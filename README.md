# open-webui x 通义法睿
该项目演示了如何封装第三方大模型API接口（支持SSE和非SSE输出）, 并结合Open WebUI实现私有化部署。动机: 
- 通义法睿的API相比通义法睿应用使用起来更加灵活也更便宜, 适合个人用户
- 私有化部署，数据更安全

Open WebUI：https://github.com/open-webui/open-webui

通义法睿：https://help.aliyun.com/zh/model-studio/developer-reference/tongyi-farui

## 安装并启动Open WebUI
```bash
# Install Open WebUI. Python version need to be >= 3.11
pip install open-webui

# Currently we are not using this. Avoid long time loading, see: https://blog.kazoottt.top/posts/openwebui-long-loading-white-screen-solution/
export ENABLE_OPENAI_API=0 # Currently we are not using this

# Running Open WebUI
export WEBUI_NAME="MY_FARUI"
open-webui serve
```

Open WebUI启动后，访问地址默认为 http://localhost:8080
这里将OPENAI_API默认关闭，防止加载时load时间过长的问题。如果后续有需要，可以在“设置-管理员设置-外部链接”中重新勾选。

## 运行通义法睿

运行通义法睿openai proxy：

```bash
export DASHSCOPE_API_KEY=sk-xxx

uvicorn dashscope_to_openai_proxy:app --host 0.0.0.0 --port 11434 --reload
```

启动后，默认服务地址为 http://localhost:11434 
接下来，在Open WebUI中设置相应的端点、API_KEY、模型名称即可。
