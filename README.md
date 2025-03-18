# open-webui x 通义法睿
open-webui：https://github.com/open-webui/open-webui

通义法睿：https://tongyi.aliyun.com/farui/home

## Install and run
```bash
# Install Open WebUI
pip install open-webui

# Running Open WebUI
open-webui serve
```

Open WebUI启动后，访问地址默认为 http://localhost:8080

启动通义法睿openai proxy：

```bash
export DASHSCOPE_API_KEY=sk-xxx
python dashscope_to_openai_proxy.py
```

启动后，默认服务地址为 http://localhost:11434。接下来，在Open WebUI中设置相应的端点、API_KEY、模型名称即可。



