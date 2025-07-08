# open-webui x 通义法睿
该项目演示了封装通义法睿**模型API**和**应用API**接口（支持SSE和非SSE输出）为OpenAI API接口, 并结合Open WebUI实现部署应用。动机: 目前通义法睿的官方API无法支持OpenAI API调用，需要实现额外封装。

**通义法睿模型API vs. 应用API对比：**

通义法睿大模型是以通义千问为基座，经法律行业数据和知识专门训练的法律行业大模型产品，具有回答法律问题、推理法律适用、推荐裁判类案、辅助案情分析、生成法律文书、检索法律知识、审查合同条款等基础功能，适用场景更加灵活。

通义法睿应用API在通义法睿大模型的基础上封装了RAG、Agent等能力，开放的法律咨询、合同审查功能相比通义法睿大模型的效果更好，如果有法律咨询、合同审查的需求，推荐优先使用通义法睿应用API。

相关链接：
- Open WebUI：https://github.com/open-webui/open-webui
- 通义法睿：https://help.aliyun.com/zh/model-studio/developer-reference/tongyi-farui
- 通义法睿模型API: https://tongyi.aliyun.com/farui/guide/model_api_doc
- 通义法睿应用API: https://tongyi.aliyun.com/farui/guide/api_description_doc


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
# 运行模型API, 模型名: xiaoxiang-farui
export DASHSCOPE_API_KEY=sk-xxx
uvicorn faruiplus_to_openai_proxy:app --host 0.0.0.0 --port 11434 --reload

# 运行应用API, 模型名: xiaoxiang-farui-v2
export DASHSCOPE_API_KEY=sk-xxx
export ACCESS_KEY_ID="LTAxxx"
export ACCESS_KEY_SECRET="xxx"
export WORKSPACE_ID="llm-xxx"
uvicorn farui_to_openai_proxy:app --host 0.0.0.0 --port 11435 --reload
```

运行后，OpenAI服务基础地址分别为 http://localhost:11434 和 http://localhost:11435。接下来，在Open WebUI中设置相应的端点、API_KEY、模型名称即可。
