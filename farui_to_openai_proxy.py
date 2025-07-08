"""
封装**法睿应用API**为OpenAI兼容格式
原始API参考: https://tongyi.aliyun.com/farui/guide/api_description_doc
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import json
import time
from fastapi.middleware.cors import CORSMiddleware
from typing import AsyncGenerator


from alibabacloud_tea_openapi_sse.client import Client as OpenApiClient
from alibabacloud_tea_openapi_sse import models as open_api_models
from alibabacloud_tea_util_sse import models as util_models
import json


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


MODEL_MAP = {
    "xiaoxiang-farui-v2": "farui",
}
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
ACCESS_KEY_ID = os.getenv("ACCESS_KEY_ID", "")
ACCESS_KEY_SECRET = os.getenv("ACCESS_KEY_SECRET", "")
WORKSPACE_ID = os.getenv("WORKSPACE_ID",  "")

endpoint = 'farui.cn-beijing.aliyuncs.com'


class Farui:
    def __init__(self, workspace_id: str, endpoint: str, access_key_id: str, access_key_secret: str) -> None:
        self._workspace_id = workspace_id
        self._endpoint = endpoint
        self._client = None
        self._api_info = self._create_api_info()
        self._runtime = util_models.RuntimeOptions(read_timeout=1000 * 100)
        self._client = self._create_client(
            access_key_id, access_key_secret, endpoint)

    def _create_client(
        self,
        access_key_id: str,
        access_key_secret: str,
        endpoint: str,
    ) -> OpenApiClient:
        config = open_api_models.Config(
            access_key_id=access_key_id,
            access_key_secret=access_key_secret
        )
        config.endpoint = endpoint
        return OpenApiClient(config)

    def _create_api_info(self) -> open_api_models.Params:
        """
        API 相关
        @param path: params
        @return: OpenApi.Params
        """
        params = open_api_models.Params(
            # 接口名称
            action='RunLegalAdviceConsultation',
            # 接口版本
            version='2024-06-28',
            # 接口协议
            protocol='HTTPS',
            # 接口 HTTP 方法
            method='POST',
            auth_type='AK',
            style='SSE',
            # 接口 PATH,
            pathname=f'/{self._workspace_id}/farui/legalAdvice/consult',
            # 接口请求体内容格式,
            req_body_type='json',
            # 接口响应体内容格式,
            body_type='sse'
        )
        return params

    async def do_sse_query(self, query: str):
        assert self._client is not None
        assert isinstance(
            query, str), '"recalling_query" is mandatory and should be str'
        assistant = {
            'type': 'legal_advice_consult',
            'version': '1.0.0'
        }
        thread = {
            'messages': [{'role': 'user', 'content': f'{query}'}]
        }
        body = {
            'appId': 'farui',
            'stream': True,
            'assistant': assistant,
            'thread': thread
        }
        request = open_api_models.OpenApiRequest(
            body=body
        )
        sse_receiver = self._client.call_sse_api_async(
            params=self._api_info, request=request, runtime=self._runtime)
        return sse_receiver


farui = Farui(WORKSPACE_ID, endpoint, ACCESS_KEY_ID, ACCESS_KEY_SECRET)


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


async def stream_to_openai_format(request_data: OpenAIRequest) -> AsyncGenerator[str, None]:
    """将SSE响应转换为OpenAI流式格式"""
    content = request_data.messages[-1].get("content", "")

    prev_content = ""
    is_first = True
    start_time = int(time.time())
    openai_id = f"chatcmpl-{int(time.time())}"

    async for res in await farui.do_sse_query(content):
        try:
            sse_event = json.loads(res.get('event').data)
            # print(res.get('event').data)
        except json.JSONDecodeError:
            # print('------json.JSONDecodeError-end--------')
            continue

        finish_reason = sse_event.get("Status", "null")
        if finish_reason == "回答生成完毕":
            finish_reason = "stop"
        else:
            finish_reason = None

        content = sse_event.get("ResponseMarkdown", "")
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
    if api_key != API_KEY:
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
    else:
        # 把 SSE 全量收完，组装成一次性 OpenAI 格式返回
        full_text = ""
        async for res in await farui.do_sse_query(request_data.messages[-1]["content"]):
            try:
                sse_event = json.loads(res.get("event").data)
                full_text = sse_event.get("ResponseMarkdown", "") or full_text
            except Exception:
                continue
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request_data.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_text
                },
                "finish_reason": "stop"
            }]
        }


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
    if not API_KEY or not ACCESS_KEY_ID or not ACCESS_KEY_SECRET or not WORKSPACE_ID:
        raise ValueError(
            "Environment variables DASHSCOPE_API_KEY, ACCESS_KEY_ID, ACCESS_KEY_SECRET, and WORKSPACE_ID must be set.")

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11435)
