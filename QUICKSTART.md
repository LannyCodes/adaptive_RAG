# 快速开始指南

## 安装依赖

```bash
pip install -r requirements.txt
```

## 环境配置

### 1. 设置API密钥

复制 `.env.example` 文件为 `.env`：
```bash
cp .env.example .env
```

编辑 `.env` 文件，填入您的实际API密钥：
```bash
# Tavily API密钥 - 用于网络搜索功能
# 从 https://tavily.com/ 获取
TAVILY_API_KEY=your_actual_tavily_api_key

# Nomic API密钥 - 用于文本嵌入服务
# 从 https://atlas.nomic.ai/ 获取
NOMIC_API_KEY=your_actual_nomic_api_key
```

### 2. 确保本地模型可用

确保您已安装并运行Ollama，并下载了Mistral模型：
```bash
# 安装Ollama
# 访问 https://ollama.ai/ 下载安装

# 下载Mistral模型
ollama pull mistral
```

## 运行系统

```bash
python main.py
```

## 项目结构

```
adaptive_RAG/
├── config.py              # 配置和环境设置
├── document_processor.py  # 文档处理和向量化
├── routers_and_graders.py # 路由器和评分器
├── workflow_nodes.py      # 工作流节点
├── main.py                # 主应用程序入口
├── requirements.txt       # 依赖管理
├── README.md              # 项目说明
└── QUICKSTART.md          # 快速开始指南
```

## 功能模块说明

1. **config.py**: 包含所有配置项、API密钥管理和环境变量设置
2. **document_processor.py**: 负责文档加载、分块、向量化和检索器设置
3. **routers_and_graders.py**: 实现查询路由、文档评分、答案质量评估等功能
4. **workflow_nodes.py**: 定义所有工作流节点和状态管理
5. **main.py**: 系统集成和用户交互界面

## 使用示例

系统启动后会自动进入交互模式，你可以：
- 询问关于LLM、提示工程、对抗性攻击的问题（使用本地知识库）
- 询问其他问题（自动路由到网络搜索）
- 输入 'quit' 或 'exit' 退出系统