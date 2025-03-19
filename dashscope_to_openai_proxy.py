"""
以Dashscope 法睿模型为例, 演示如何封装custom API为openai兼容格式
"""

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import httpx
import os
import time
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# 启用 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 替换为你的 Web 应用的域名（或 ["*"] 测试）
    allow_credentials=True,  # 是否允许跨域携带凭证（如 Cookies）
    allow_methods=["*"],  # 允许的 HTTP 方法列表，例如 ["POST", "GET", "OPTIONS"]
    allow_headers=["*"],  # 允许的请求头，如 ["Content-Type", "Authorization"]
    expose_headers=[],  # 需暴露给前端的响应头（如有）
    max_age=3600,  # 预检请求结果缓存时间
)

# Custom 大模型服务API
BASE_URL = os.getenv("CUSTOM_API_BASE_URL", "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation")
API_TOKEN = os.getenv("DASHSCOPE_API_KEY", "")
MODEL_MAP = {
    "xiaoxiang-farui": "farui-plus",
}

class OpenAIRequest(BaseModel):
    model: str
    messages: list[dict]
    temperature: float = 0.7
    max_tokens: int = 16
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: list = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

async def call_custom_api(request_data):
    async with httpx.AsyncClient() as client:     
        payload = {
            "model": MODEL_MAP[request_data.model],
            "input": {
                "messages": request_data.messages,
            },
            "parameters": {
                "result_format": "message"
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_TOKEN}"
        }
        
        # 调用自定义API（需根据实际接口调整参数）
        response = await client.post(
            f"{BASE_URL}",
            json=payload,
            headers=headers,
            timeout=30
        )
        return response.json()

@app.options("/chat/completions")
async def chat_completions_options():
    return {}

@app.post("/chat/completions")
async def chat_completions(request: Request):
    # 验证API密钥
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    api_key = auth_header.split(" ")[1]
    if api_key != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # 获取请求体
    request_data = OpenAIRequest(**await request.json())
    
    # 处理模型不存在的情况
    if request_data.model not in MODEL_MAP:
        raise HTTPException(status_code=404, detail=f"Model {request_data.model} not found")
    
    # 调用自定义API
    try:
        custom_response = await call_custom_api(request_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backend API error: {str(e)}")
    
    # 将自定义响应转换为OpenAI格式
    openai_response = {
        "id": custom_response["request_id"],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request_data.model,
        "choices": [
            {
                "index": 0,
                "message": custom_response["output"]["choices"][0]["message"],
                "finish_reason": custom_response["output"]["choices"][0]["finish_reason"]
            }
        ],
        "usage": {
            "prompt_tokens": custom_response["usage"]["input_tokens"],
            "completion_tokens": custom_response["usage"]["output_tokens"],
            "total_tokens": custom_response["usage"]["total_tokens"]
        }
    }
    
    return openai_response

@app.options("/models")
async def models_list_options():
    return {}

@app.get("/models")
async def models_list():
    return {
        "data": [
            {"id": model, "object": "model", "owned_by": "alibaba"} 
            for model in MODEL_MAP.keys()
        ]
    }
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11434)