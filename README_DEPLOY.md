# 如何免费部署到 Hugging Face Spaces

既然 Kaggle 的网络环境太受限，最推荐的免费部署方案是 **Hugging Face Spaces**。它提供免费的容器环境，可以直接运行你的 FastAPI + React + Ollama 应用，并生成一个永久的公网 HTTPS 网址。

## 步骤 1: 准备账号
1. 注册 [Hugging Face](https://huggingface.co/) 账号。
2. 点击右上角头像 -> **New Space**。
3. **Space Name**: 随便填，比如 `adaptive-rag-demo`。
4. **License**: `MIT`。
5. **SDK**: 选择 **Docker** (这是关键)。
6. **Space Hardware**: 选择 **Free default** (2 vCPU, 16GB RAM)。
   * *注意: 免费 CPU 跑 LLM 会比较慢，建议使用较小的模型如 `tinyllama` 或 `qwen:0.5b`，或者在代码里改用 API 调用外部模型。*
7. 点击 **Create Space**。

## 步骤 2: 上传代码
你有两种方式上传代码：

### 方法 A: 使用网页上传 (最简单)
1. 在你刚创建的 Space 页面，点击 **Files** 标签页。
2. 点击 **Add file** -> **Upload files**。
3. 把你本地项目文件夹里的所有文件（包括刚才生成的 `Dockerfile`, `server.py`, `requirements.txt` 等）拖进去。
   * *注意: 不需要上传虚拟环境文件夹 `venv` 或 `__pycache__`。*
4. 点击 **Commit changes to main**。

### 方法 B: 使用 Git 命令
```bash
git clone https://huggingface.co/spaces/你的用户名/adaptive-rag-demo
cd adaptive-rag-demo
# 把你的代码复制进来
git add .
git commit -m "Initial commit"
git push
```

## 步骤 3: 等待构建
上传后，Hugging Face 会自动检测 `Dockerfile` 并开始构建：
1. 点击 Space 页面的 **App** 标签。
2. 你会看到 "Building..." 的状态。
3. 等待几分钟（安装依赖和下载模型需要时间）。
4. 当状态变为 **Running** 时，你会直接看到你的前端页面！

## 常见问题
*   **模型下载慢/超时**: 我在 Dockerfile 里默认用了 `tinyllama`，因为它很小。如果你想用更好的模型（如 `mistral`），可以在 Dockerfile 里修改 `ollama pull mistral`，但构建时间会变长。
*   **速度慢**: 免费的 CPU 空间推理速度有限。如果想流畅体验，建议在 Space Settings 里切换到 GPU (需要付费)，或者修改代码让 LLM 部分调用 OpenAI/Groq 的 API，只在 Space 里跑 RAG 逻辑。
