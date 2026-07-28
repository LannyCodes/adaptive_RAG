---
title: Adaptive RAG System
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: Apache License 2.0
short_description: A RAG system deployed with Docker
---

# 自适应 RAG 系统 · 通用智能体 (Adaptive RAG & General Agent)

> 基于 LangGraph 的通用智能体系统：**Supervisor 多智能体编排 + MCP 标准化工具 + 三层记忆体系 + 安全护栏（HITL）**，内核为自适应 RAG（智能路由、混合检索、多跳推理、幻觉检测、GraphRAG）。

---

## 目录

- [项目简介](#项目简介)
- [系统架构](#系统架构)
- [运行模式](#运行模式)
- [核心特性](#核心特性)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [API 接口](#api-接口)
- [文件上传与索引](#文件上传与索引)
- [Docker 部署](#docker-部署)
- [项目结构](#项目结构)
- [技术栈](#技术栈)
- [工作流节点详解（Workflow 模式）](#工作流节点详解workflow-模式)
- [质量保证与安全](#质量保证与安全)
- [许可证](#许可证)

---

## 项目简介

本项目在自适应检索增强生成（Adaptive RAG）内核之上，升级为**通用智能体（General Agent）**架构：

1. 用户上传 PDF / Word 等文档，系统自动解析、分块、向量化并存入 Milvus 向量数据库
2. 用户输入查询后，**Supervisor** 按任务类型分派给子智能体：
   - **Research Agent** — 知识型问题：智能路由 → 查询分解 → 混合检索 → 重排序 → 文档评分 → 答案生成 → 幻觉检测
   - **Action Agent** — 操作型任务：代码沙箱执行、文件读写、网页抓取
   - **Verifier** — 对候选答案做最终质量校验，不通过则换策略重试
3. 工具层通过 **MCP (Model Context Protocol)** 标准化暴露，支持 stdio / http 双传输
4. **三层记忆体系**：短期记忆（Checkpointer）+ 工作记忆（自动压缩）+ 长期记忆（Milvus 事实型记忆）
5. 全流程安全防护：输入注入检测、输出敏感信息脱敏、行动任务人工审批（HITL）

### 适用场景

- 企业内部知识库问答
- 学术文献检索与问答
- 多源信息聚合分析
- 需要高准确率答案的质量保证场景
- 检索 + 计算 + 文件操作混合的复杂多步任务

---

## 系统架构

### 多智能体编排（Agent 模式）

```
用户查询
  │
  ▼
┌──────────────────────────┐
│ 输入护栏                  │
│ (提示注入检测/黑名单)      │
└────────────┬─────────────┘
             ▼
      ┌─────────────┐      长期记忆 (Milvus user_memory)
      │  Supervisor  │ ◄─── 工作记忆 (超阈值自动压缩)
      │  LLM 决策    │      短期记忆 (Redis/Memory Checkpointer)
      │ {next,task} │
      └──┬───┬───┬──┘
         │   │   │
   ┌─────▼┐ ┌▼──────┐ ┌▼────────┐
   │Research│ │Action │ │Verifier │
   │Agent  │ │Agent  │ │答案校验  │
   └──┬───┘ └──┬───┘ └────┬────┘
      │        │          │
      ▼        ▼          │ 未通过 → Supervisor 换策略重试
┌──────────┐ ┌───────────┐│        (硬性上限, 代码层保证)
│rag-tools │ │action-tools││
│MCP Server│ │MCP Server  ││
│9个RAG工具 │ │代码/文件/网页││
└──────────┘ └───────────┘▼
                    通过 → 输出护栏 (敏感信息脱敏) → 最终答案

行动任务执行前: interrupt 中断 → 人工审批 (/api/chat/approve) → 恢复执行 (HITL)
```

### RAG 内核工作流（Research Agent 策略 / Workflow 模式）

```
用户查询
  │
  ▼
┌─────────────────────┐
│  智能路由 (route)    │
│  LLM 判断信息源      │
└──────┬──────┬───────┘
       │      │
  向量检索   网络搜索
       │      │
       ▼      ▼
  查询分解   Tavily API
  (多跳推理)     │
       │      │
       ▼      │
  混合检索      │
  (向量+BM25)   │
       │      │
       ▼      │
  CrossEncoder  │
  重排序        │
       │      │
       ▼      │
  文档评分      │
  (相关性过滤)   │
       │      │
       ▼      ▼
  ┌──────────────┐
  │  答案生成     │
  │  (RAG Chain) │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  幻觉检测     │
  │  NLI+Vectara │
  └──────┬───────┘
         │
    ┌────┴────┐
    │         │
 可信答案   重新检索
```

---

## 运行模式

通过 `ENABLE_AGENT_MODE` 环境变量切换，两种模式共享同一套 RAG 内核实现：

| 模式 | 值 | 说明 |
|------|-----|------|
| **Agent 模式**（默认） | `agent` | Supervisor 多智能体编排：research / action / verifier 子 Agent 协同，支持 HITL 人工审批、三层记忆、Redis 状态持久化 |
| **Workflow 模式** | `workflow` | 固定 DAG 工作流（降级回退路径）：路由 → 检索 → 评分 → 生成 → 幻觉检测，流程确定、延迟更低 |

---

## 核心特性

### 智能体架构

| 特性 | 说明 |
|------|------|
| **Supervisor 编排** | 手写 Supervisor 路由（LLM JSON 决策，弱模型友好），子 Agent 上下文隔离，硬性重试上限由代码层保证 |
| **Research Agent** | `create_react_agent` + rag-tools（MCP），即 RAG 能力的 Agent 化封装 |
| **Action Agent** | `create_react_agent` + action-tools（MCP）：代码沙箱 / 文件读写 / 网页抓取 |
| **Verifier** | 答案质量确定性校验（复用 answer_grader），故障时放行不阻塞主流程 |
| **MCP 工具层** | 9 个 RAG 工具 + 5 个行动工具，stdio（开发）/ http（生产常驻）双传输 |
| **HITL 人工审批** | 行动任务执行前 `interrupt` 中断，经 `/api/chat/approve` 审批后恢复 |

### 记忆体系

| 层级 | 实现 | 说明 |
|------|------|------|
| **短期记忆** | RedisSaver / MemorySaver Checkpointer | 会话状态持久化，支持中断恢复、流式事件 |
| **工作记忆** | 字符阈值自动压缩 | 超过 16000 字符时保留最近 6 条消息，其余摘要压缩 |
| **长期记忆** | Milvus 独立 collection | 对话后轻量模型异步抽取事实，按语义检索注入系统提示 |

### 安全护栏

| 特性 | 说明 |
|------|------|
| **输入护栏** | 提示注入检测（中英文规则正则 + 关键词黑名单） |
| **输出护栏** | 敏感信息正则脱敏（API Key / 密码 / 私钥模式） |
| **工具侧强制约束** | 代码沙箱（禁网络/subprocess、超时、输出截断）、文件路径防穿越（限制在工作目录内） |

### RAG 内核

| 特性 | 说明 |
|------|------|
| **自适应路由** | LLM 智能判断查询应走向量检索还是网络搜索 |
| **混合检索** | 向量语义检索 + BM25 关键词检索，权重可调 |
| **多跳推理** | 复杂问题自动分解为子问题序列，逐步检索和推理 |
| **CrossEncoder 重排** | 使用 `BAAI/bge-reranker-base` 对检索结果精确重排 |
| **幻觉检测** | NLI 模型 (`nli-deberta-v3-xsmall`) + Vectara HHEM 双重检测 |
| **GraphRAG** | 知识图谱构建、社区检测、图谱检索（默认开启） |
| **查询优化** | 查询扩展 + 查询重写 + 桥接实体提取 |
| **多模态支持** | OCR + VLM（通义 qwen-vl-plus）双通道图像理解 |
| **文件上传** | 支持 PDF / Word / PPT / Excel / EPUB 等多种格式，含 LaTeX 公式感知分块 |
| **异步架构** | 全链路 async，MCP 工具异步加载，支持并发查询 |
| **LangSmith 集成** | 全链路追踪、性能监控、告警通知 |
| **Web UI** | FastAPI + React 18 + Tailwind CSS 内置前端 |

---

## 快速开始

### 环境要求

- Python 3.10+
- 推荐：NVIDIA GPU（CUDA）用于嵌入模型和重排器加速
- 可选：Ollama（如使用本地 LLM 后端）
- 可选：Redis（会话持久化与 Agent Checkpointer）

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件：

```bash
# 必需：Tavily 搜索 API 密钥
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxx

# LLM 后端选择：tongyi（推荐）/ ollama / deepseek
LLM_BACKEND=tongyi

# 通义千问配置（LLM_BACKEND=tongyi 时需要）
TONGYI_API_KEY=sk-xxxxxxxx
TONGYI_MODEL=qwen-plus

# DeepSeek 配置（LLM_BACKEND=deepseek 时需要）
# DEEPSEEK_API_KEY=sk-xxxxxxxx
# DEEPSEEK_MODEL=deepseek-chat

# Ollama 配置（LLM_BACKEND=ollama 时需要）
# LOCAL_LLM=qwen2:1.5b

# Milvus 向量数据库（默认使用本地文件模式）
# MILVUS_URI=./milvus_rag.db

# 运行模式：agent（默认，多智能体）/ workflow（固定工作流）
# ENABLE_AGENT_MODE=agent

# Redis 会话持久化（可选，不配置则降级为内存存储）
# REDIS_URL=redis://default:password@host:port/0

# HITL 人工审批（默认开启，行动任务执行前需人工确认）
# HITL_ENABLED=true

# MCP 传输模式：stdio（默认，开发）/ http（生产常驻服务）
# MCP_TRANSPORT=stdio
```

### 3. 启动系统

**方式一：Web 服务模式（推荐）**

```bash
python server.py
```

访问 `http://localhost:8000` 即可使用 Web 界面。

**方式二：命令行交互模式**

```bash
python main.py
```

**方式三：Docker 部署**

```bash
docker build -t adaptive-rag .
docker run -p 7860:7860 --env-file .env adaptive-rag
```

**方式四：MCP Server 常驻模式（生产推荐）**

```bash
# 终端 1：RAG 工具服务（端口 8100）
MCP_TRANSPORT=http python -m mcp_servers.rag_server

# 终端 2：行动工具服务（端口 8101）
MCP_TRANSPORT=http python -m mcp_servers.action_server

# 终端 3：Web 服务
MCP_TRANSPORT=http python server.py
```

---

## 配置说明

所有配置集中在 `config.py` 中，支持通过环境变量覆盖：

### LLM 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `LLM_BACKEND` | `tongyi` | LLM 后端：`tongyi` / `ollama` / `deepseek` |
| `LOCAL_LLM` | `qwen2:1.5b` | Ollama 模型名称 |
| `LIGHT_LLM` | `qwen2.5:1.5b` | 轻量模型，用于路由/评分等简单任务 |
| `TONGYI_MODEL` | `qwen-plus` | 通义千问模型名称 |
| `DEEPSEEK_MODEL` | `deepseek-chat` | DeepSeek 模型名称 |

### 向量数据库配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | 嵌入模型（支持 8192 长度） |
| `CHUNK_SIZE` | `1024` | 文档分块大小 |
| `CHUNK_OVERLAP` | `200` | 分块重叠 |
| `MILVUS_URI` | `./milvus_rag.db` | Milvus Lite 本地路径 |
| `MILVUS_INDEX_TYPE` | `HNSW` | 索引类型：HNSW / IVF_FLAT / IVF_SQ8 |

### Agent 智能体配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ENABLE_AGENT_MODE` | `agent` | 运行模式：`agent`（多智能体）/ `workflow`（固定工作流） |
| `AGENT_MAX_ITERATIONS` | `30` | Agent 递归上限（recursion_limit） |
| `HITL_ENABLED` | `true` | 行动任务执行前人工审批 |
| `REDIS_URL` | 空 | Redis 连接串（会话持久化 + Checkpointer，空则降级内存） |
| `REDIS_SESSION_TTL` | `604800` | 会话过期时间（秒），默认 7 天 |

### MCP 工具服务配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MCP_TRANSPORT` | `stdio` | 传输模式：`stdio`（client 拉起子进程）/ `http`（常驻服务，生产必用） |
| `MCP_RAG_PORT` | `8100` | RAG 工具 MCP Server 端口（http 模式） |
| `MCP_ACTION_PORT` | `8101` | 行动工具 MCP Server 端口（http 模式） |
| `ACTION_WORKSPACE_DIR` | `./workspace` | 文件工具与代码执行的根目录（路径穿越防护边界） |
| `CODE_EXEC_TIMEOUT` | `30` | 代码沙箱超时（秒） |
| `CODE_EXEC_MAX_OUTPUT` | `5000` | 代码执行输出截断长度 |

### 记忆体系配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `USER_MEMORY_COLLECTION` | `user_memory` | 长期记忆 Milvus collection |
| `MEMORY_TOP_K` | `3` | 长期记忆检索条数 |
| `CONTEXT_COMPRESS_THRESHOLD` | `16000` | 工作记忆压缩阈值（字符数，约 8000 token） |
| `CONTEXT_COMPRESS_KEEP_RECENT` | `6` | 压缩时保留的最近消息条数 |

### 功能开关

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ENABLE_GRAPHRAG` | `true` | 启用知识图谱检索 |
| `ENABLE_HYBRID_SEARCH` | `true` | 启用混合检索（向量+BM25） |
| `ENABLE_QUERY_EXPANSION` | `true` | 启用查询扩展 |
| `ENABLE_MULTIMODAL` | `true` | 启用多模态检索（OCR + VLM 双通道） |
| `ENABLE_ADVANCED_RERANKER` | `false` | 启用高级重排器（context_aware / multi_task） |

---

## API 接口

系统提供 RESTful API，启动后可访问 `http://localhost:8000/docs` 查看完整 API 文档。

### 聊天接口

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "什么是注意力机制？"}'
```

响应：
```json
{
  "answer": "注意力机制是...",
  "sources": ["参考文档片段1...", "参考文档片段2..."],
  "metrics": {
    "latency": 1.234,
    "retrieved_docs_count": 5,
    "precision_at_3": 0.85
  }
}
```

### 流式聊天（SSE）

```bash
curl -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "什么是注意力机制？", "session_id": "xxx"}'
```

### HITL 人工审批

行动任务触发审批中断后，前端展示待审批任务，用户确认后调用：

```bash
curl -X POST http://localhost:8000/api/chat/approve \
  -H "Content-Type: application/json" \
  -d '{"session_id": "xxx", "approved": true}'
```

### 会话管理

```bash
# 创建会话
curl -X POST http://localhost:8000/api/session/create

# 会话列表
curl http://localhost:8000/api/session/list

# 获取会话历史
curl http://localhost:8000/api/session/{session_id}
```

### 文件上传接口

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@/path/to/document.pdf"
```

### 其他接口

| 接口 | 说明 |
|------|------|
| `GET /api/health` | 健康检查 |
| `GET /api/documents` | 已索引文档列表 |
| `GET /api/stats` | 系统统计信息 |
| `GET /api/images/{path}` | 图片访问（多模态） |

---

## 文件上传与索引

使用独立脚本上传文件并构建向量索引：

```bash
# 上传单个文件
python upload_and_index.py /path/to/file.pdf

# 上传多个文件
python upload_and_index.py file1.pdf file2.docx file3.pdf

# 上传整个目录（递归扫描）
python upload_and_index.py /path/to/folder/

# 自定义分块参数
python upload_and_index.py --chunk-size 512 --chunk-overlap 100 file.pdf

# 使用 marker 模型做 PDF→Markdown（保留 LaTeX 公式，需 GPU）
python upload_and_index.py --use-marker file.pdf

# 仅预览，不写入向量库
python upload_and_index.py --dry-run file.pdf
```

支持的文件格式：PDF、Word (.docx)、PowerPoint (.pptx)、Excel (.xlsx)、EPUB 电子书

---

## Docker 部署

### CPU 部署

```bash
docker build -t adaptive-rag .
docker run -p 7860:7860 --env-file .env adaptive-rag
```

### GPU 部署（推荐）

```bash
docker-compose -f docker-compose.gpu.yml up -d
```

GPU 部署需要：
- 安装 NVIDIA Container Toolkit
- 具备 NVIDIA GPU 及 CUDA 驱动

### Hugging Face Spaces 部署

详见 [README_DEPLOY.md](README_DEPLOY.md)

---

## 项目结构

```
adaptive_RAG/
├── agents/                    # 智能体层
│   ├── supervisor.py          # Supervisor 多智能体编排（research/action/verifier 路由）
│   ├── research_agent.py      # Research Agent — 知识检索子智能体
│   ├── action_agent.py        # Action Agent — 行动型子智能体（代码/文件/网页）
│   ├── verifier_agent.py      # Verifier — 答案质量校验器
│   ├── runtime.py             # Agent 运行时（create_react_agent + Checkpointer）
│   ├── mcp_client.py          # MCP 工具异步加载客户端
│   └── rag_tool_impl.py       # 9 个 RAG 工具的纯函数实现（进程内/MCP 共享）
├── mcp_servers/               # MCP 工具服务层
│   ├── rag_server.py          # RAG 工具 MCP Server（9 工具，stdio/http 双模式）
│   └── action_server.py       # 行动工具 MCP Server（代码沙箱/文件/网页抓取）
├── memory/                    # 记忆体系
│   └── user_memory.py         # 长期事实型记忆（Milvus 独立 collection）
├── main.py                    # 命令行主入口（双模式：agent / workflow）
├── server.py                  # FastAPI + React Web 服务（SSE 流式/HITL/会话管理）
├── app.py                     # Python Runner 启动脚本（Kaggle/ModelScope）
├── config.py                  # 集中配置管理
├── guardrails.py              # 安全护栏（输入注入检测 + 输出脱敏）
├── document_processor.py      # 文档处理、向量化、检索核心模块
├── upload_and_index.py        # 文件上传与向量化独立脚本
├── routers_and_graders.py     # 查询路由、文档评分、答案评分、LLM 工厂
├── workflow_nodes.py          # LangGraph 工作流节点定义（workflow 模式）
├── reranker.py                # 多策略重排器（TF-IDF/BM25/CrossEncoder/混合/多样性）
├── hallucination_detector.py  # 幻觉检测（NLI/Vectara/混合）
├── lightweight_hallucination_detector.py  # 轻量级幻觉检测
├── knowledge_graph.py         # 知识图谱构建与社区检测
├── graph_retriever.py         # 图谱检索（本地/全局查询）
├── graph_indexer.py           # 图谱索引构建
├── entity_extractor.py        # 实体提取
├── cache_manager.py           # 缓存管理
├── prompt_manager.py          # 提示词管理（prompts.yaml）
├── agent_evals.py             # Agent 评估脚本
├── retrieval_evaluation.py    # 检索评估（Precision/Recall/MAP/NDCG）
├── langsmith_integration.py   # LangSmith 追踪与监控集成
├── requirements.txt           # Python 依赖
├── Dockerfile                 # Docker 镜像（CPU）
├── Dockerfile.gpu             # Docker 镜像（GPU）
├── Dockerfile.slim            # Docker 镜像（精简版，不含浏览器）
├── docker-compose.gpu.yml     # GPU Docker Compose 配置
├── start.sh                   # 启动脚本
├── entrypoint.sh              # Docker 入口脚本
├── data/                      # 数据目录（向量库、图谱索引、上传文件等）
└── source/                    # 源文件目录
```

---

## 技术栈

### 核心框架
- **LangChain** + **LangGraph** — LLM 应用编排与工作流状态管理（`create_react_agent` + 手写 Supervisor）
- **MCP (Model Context Protocol)** — 标准化工具协议，stdio / streamable-http 双传输
- **FastAPI** + **Uvicorn** — 高性能异步 Web 服务
- **React 18** + **Tailwind CSS** — 现代化前端界面

### 语言模型
- **通义千问 (Qwen)** — 推荐 LLM 后端（含 qwen-vl-plus 视觉模型）
- **Ollama** — 本地 LLM 推理引擎
- **DeepSeek** — 可选 LLM 后端

### 向量数据库与嵌入
- **Milvus** — 向量数据库（支持 Lite/Server/Zilliz Cloud 三种模式，主库 + 记忆库双 collection）
- **BAAI/bge-m3** — 嵌入模型（8192 长度，中英双语）
- **BAAI/bge-reranker-base** — CrossEncoder 重排模型

### 检索与重排
- **BM25** (rank-bm25) — 关键词检索
- **Elasticsearch** — 大规模 BM25 检索（可选）
- **CrossEncoder** — 精确重排序

### 状态持久化与记忆
- **Redis** — 会话存储 + LangGraph RedisSaver Checkpointer
- **MemorySaver** — 无 Redis 时的内存降级方案

### 幻觉检测
- **NLI 模型** (`nli-deberta-v3-xsmall`) — 自然语言推理检测
- **Vectara HHEM** — 专业幻觉评估模型

### 知识图谱
- **NetworkX** — 图结构管理
- **python-louvain** / **leidenalg** — 社区检测
- **Neo4j** — 图数据库（可选）

### 行动工具
- **子进程沙箱** — 代码执行隔离（禁网络/subprocess，超时+截断）
- **Playwright** — 网页抓取（可选依赖，未安装自动降级）

### 监控与评估
- **LangSmith** — 全链路追踪与性能监控
- **scikit-learn** — 检索评估指标计算

---

## 工作流节点详解（Workflow 模式）

| 节点 | 功能 | 输出 |
|------|------|------|
| `route_and_decompose` | 智能路由 + 查询分解 | `web_search` / `vectorstore` |
| `retrieve` | 文档检索（混合检索+查询扩展+重排序） | 文档列表 |
| `grade_documents` | 文档相关性评分与过滤 | 过滤后的相关文档 |
| `decide_to_generate` | 决策：生成/继续检索/网络搜索 | `generate` / `prepare_next_query` / `transform_query` / `web_search` |
| `prepare_next_query` | 准备下一个子查询（重写+桥接实体） | 优化后的查询 |
| `transform_query` | 查询转换/重写 | 改进后的查询 |
| `generate` | RAG 答案生成 | 生成的答案 |
| `grade_generation` | 答案质量检查（幻觉检测+有用性评分） | `useful` / `not useful` / `not supported` |
| `web_search` | 网络搜索（Tavily API） | 搜索结果 |

Agent 模式下，上述能力以 9 个 MCP 工具形式提供给 Research Agent 自主编排：`route_query` / `decompose_query` / `retrieve_from_vectorstore` / `search_web` / `grade_documents` / `rewrite_query` / `check_answerability` / `generate_answer` / `check_hallucination`。

---

## 质量保证与安全

### 多层验证
1. **文档相关性评分** — 过滤不相关文档
2. **幻觉检测** — 确保答案基于源文档（NLI + Vectara 双重检测，Research Agent 内部完成）
3. **Verifier 校验** — Supervisor 层验证答案是否切题，不通过则换策略重试（硬性上限）

### 安全防线
1. **输入护栏** — 提示注入检测（中英文模式，覆盖指令覆盖/越狱/系统提示探测）
2. **输出护栏** — 敏感信息脱敏（API Key / 密码 / 私钥正则过滤）
3. **工具约束** — 代码沙箱黑名单 + 独立子进程 + 超时截断；文件操作路径防穿越
4. **HITL 审批** — 行动任务执行前人工确认，可随时拒绝

### 迭代改进
- 查询转换 — 改进检索效果
- 重试机制 — 最大重试次数限制
- 回退策略 — 网络搜索作为备选

### 早期终止
- 答案可回答性检查 — 避免不必要的检索
- 多跳检索优化 — 提前终止已完成的子查询

---

## 许可证

[Apache License 2.0](LICENSE)
