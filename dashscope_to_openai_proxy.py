"""
以Dashscope 法睿模型为例, 演示如何封装custom API为openai兼容格式
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import os
import json
import time
from fastapi.middleware.cors import CORSMiddleware
from typing import AsyncGenerator


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
BASE_URL = os.getenv("CUSTOM_API_BASE_URL",
                     "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation")
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
    """
    调用法睿API, SSE关闭
    Ref: https://help.aliyun.com/zh/model-studio/developer-reference/tongyi-farui-api
    """
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

        response = await client.post(
            f"{BASE_URL}",
            json=payload,
            headers=headers,
            timeout=30
        )
        return response.json()


async def parse_sse_event(line_iterator: AsyncGenerator[str, None]) -> AsyncGenerator[dict, None]:
    """
    根据法睿API文档, 解析SSE响应流, 逐条生成事件字典
    Ref: https://help.aliyun.com/zh/model-studio/developer-reference/tongyi-farui-api#UzzW0
    """
    buffer = ""
    async for line in line_iterator:
        line = line.strip()
        if line == "":
            # 空行表示一个事件结束
            yield parse_event(buffer)
            buffer = ""
        else:
            buffer += f"{line}\n"
    # 处理最后一条未结束的事件
    if buffer:
        yield parse_event(buffer)


def parse_event(text: str) -> dict:
    """将单个SSE事件文本解析为字典"""
    event = {}
    lines = text.split('\n')
    for line in lines:
        if not line:
            continue
        prefix, _, value = line.partition(':')
        key = prefix.lower()
        if key == ':http_status':
            key = 'http_status'
        event[key] = value.strip()
    # 解析JSON数据
    data_json = event.pop('data', '{}')
    try:
        data = json.loads(data_json)
    except json.JSONDecodeError:
        data = {}
    event['data'] = data
    return event


async def call_custom_sse_api(request_data: OpenAIRequest) -> AsyncGenerator[str, None]:
    """
    调用法睿API, SSE开启
    Ref: https://help.aliyun.com/zh/model-studio/developer-reference/tongyi-farui-api
    """
    async with httpx.AsyncClient(timeout=None) as client:
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
            "Authorization": f"Bearer {API_TOKEN}",
            "X-DashScope-SSE": "enable"
        }

        async with client.stream("POST", f"{BASE_URL}", json=payload, headers=headers) as response:
            async for line in response.aiter_lines():
                yield line


async def stream_to_openai_format(request_data: OpenAIRequest) -> AsyncGenerator[str, None]:
    """将SSE响应转换为OpenAI流式格式"""
    prev_content = ""
    is_first = True
    start_time = int(time.time())
    openai_id = f"chatcmpl-{int(time.time())}"

    async for sse_event in parse_sse_event(call_custom_sse_api(request_data)):
        data = sse_event.get("data", {})
        output = data.get("output", {})
        choice = output.get("choices", [{}])[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "null")

        # 取消原"null"值的干扰
        if finish_reason == "null":
            finish_reason = None
        content = message.get("content", "")

        if not content:
            continue  # 跳过空内容

        # 计算增量部分
        delta_content = content[len(prev_content)
                                    :] if prev_content else content
        prev_content = content

        # 构造OpenAI响应JSON
        chunk = {
            "id": openai_id,
            "object": "chat.completion.chunk",
            "created": start_time,
            "model": request_data.model,
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant" if is_first else "",
                    # 只在首次发送role字段
                    "content": delta_content,
                },
                "finish_reason": finish_reason if finish_reason else None
            }]
        }

        # 移除空字段
        delta = chunk["choices"][0]["delta"]
        if not delta.get("role"):
            del delta["role"]

        # 发送JSON chunk
        jsonline_chunk = f"data: {json.dumps(chunk)}\n\n"
        yield jsonline_chunk.encode("utf-8")

        is_first = False

        # 当finish_reason为stop时结束流式推送
        if finish_reason == "stop":
            break


@app.options("/chat/completions")
async def chat_completions_options():
    return {}


@app.post("/chat/completions")
async def chat_completions(request: Request):
    # 验证API密钥
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Invalid authorization header")
    api_key = auth_header.split(" ")[1]
    if api_key != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API key")

    request_data = OpenAIRequest(**await request.json())

    # 处理模型不存在的情况
    if request_data.model not in MODEL_MAP:
        raise HTTPException(
            status_code=404, detail=f"Model {request_data.model} not found")

    # 判断是否需要流式输出, 并分别调用相应API
    if request_data.stream:
        return StreamingResponse(
            stream_to_openai_format(request_data),
            media_type="text/event-stream",  # 使用JSON Lines格式
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            }
        )

    # 非流式处理
    try:
        custom_response = await call_custom_api(request_data)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Backend API error: {str(e)}")

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
