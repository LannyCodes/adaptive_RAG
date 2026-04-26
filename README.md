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

# 自适应 RAG 系统 (Adaptive RAG)

> 基于 LangGraph 的智能检索增强生成系统，支持自适应路由、混合检索、多跳推理、幻觉检测和知识图谱检索。

---

## 目录

- [项目简介](#项目简介)
- [系统架构](#系统架构)
- [核心特性](#核心特性)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [API 接口](#api-接口)
- [文件上传与索引](#文件上传与索引)
- [Docker 部署](#docker-部署)
- [项目结构](#项目结构)
- [技术栈](#技术栈)
- [许可证](#许可证)

---

## 项目简介

本项目是一个**自适应检索增强生成（Adaptive RAG）**系统，核心流程：

1. 用户上传 PDF / Word 等文档，系统自动解析、分块、向量化并存入 Milvus 向量数据库
2. 用户输入查询后，系统通过 LangGraph 状态机完成：**智能路由 → 查询分解 → 混合检索 → 重排序 → 文档评分 → 答案生成 → 幻觉检测**
3. 全流程可回溯、可监控，避免传统 if-else 无限回调

### 适用场景

- 企业内部知识库问答
- 学术文献检索与问答
- 多源信息聚合分析
- 需要高准确率答案的质量保证场景

---

## 系统架构

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

## 核心特性

| 特性 | 说明 |
|------|------|
| **自适应路由** | LLM 智能判断查询应走向量检索还是网络搜索 |
| **混合检索** | 向量语义检索 + BM25 关键词检索，权重可调 |
| **多跳推理** | 复杂问题自动分解为子问题序列，逐步检索和推理 |
| **CrossEncoder 重排** | 使用 `BAAI/bge-reranker-base` 对检索结果精确重排 |
| **幻觉检测** | NLI 模型 (`nli-deberta-v3-xsmall`) + Vectara HHEM 双重检测 |
| **GraphRAG** | 知识图谱构建、社区检测、图谱检索（默认开启） |
| **查询优化** | 查询扩展 + 查询重写 + 桥接实体提取 |
| **多模态支持** | CLIP 模型支持文本-图像跨模态检索 |
| **文件上传** | 支持 PDF / Word / PPT / Excel / EPUB 等多种格式，含 LaTeX 公式感知分块 |
| **异步架构** | 全链路异步处理，支持并发查询 |
| **LangSmith 集成** | 全链路追踪、性能监控、告警通知 |
| **Web UI** | FastAPI + React 18 + Tailwind CSS 内置前端 |

---

## 快速开始

### 环境要求

- Python 3.10+
- 推荐：NVIDIA GPU（CUDA）用于嵌入模型和重排器加速
- 可选：Ollama（如使用本地 LLM 后端）

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

---

## 配置说明

所有配置集中在 `config.py` 中，支持通过环境变量覆盖：

### LLM 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `LLM_BACKEND` | `tongyi` | LLM 后端：`tongyi` / `ollama` / `deepseek` |
| `LOCAL_LLM` | `qwen2:1.5b` | Ollama 模型名称 |
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

### 功能开关

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ENABLE_GRAPHRAG` | `true` | 启用知识图谱检索 |
| `ENABLE_HYBRID_SEARCH` | `true` | 启用混合检索（向量+BM25） |
| `ENABLE_QUERY_EXPANSION` | `true` | 启用查询扩展 |
| `ENABLE_MULTIMODAL` | `true` | 启用多模态检索 |
| `ENABLE_ADVANCED_RERANKER` | `false` | 启用高级重排器 |

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

### 文件上传接口

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@/path/to/document.pdf"
```

### 健康检查

```bash
curl http://localhost:8000/api/health
```

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
├── main.py                  # 主入口，AdaptiveRAGSystem 类和 LangGraph 工作流
├── server.py                # FastAPI + React Web 服务
├── app.py                   # Python Runner 启动脚本（Kaggle/ModelScope）
├── config.py                # 集中配置管理
├── document_processor.py    # 文档处理、向量化、检索核心模块
├── upload_and_index.py      # 文件上传与向量化独立脚本
├── routers_and_graders.py   # 查询路由、文档评分、答案评分
├── workflow_nodes.py        # LangGraph 工作流节点定义
├── reranker.py              # 多策略重排器（TF-IDF/BM25/CrossEncoder/混合/多样性）
├── hallucination_detector.py    # 幻觉检测（NLI/Vectara/混合）
├── lightweight_hallucination_detector.py  # 轻量级幻觉检测
├── knowledge_graph.py       # 知识图谱构建与社区检测
├── graph_retriever.py       # 图谱检索（本地/全局查询）
├── graph_indexer.py         # 图谱索引构建
├── entity_extractor.py      # 实体提取
├── retrieval_evaluation.py  # 检索评估（Precision/Recall/MAP/NDCG）
├── langsmith_integration.py # LangSmith 追踪与监控集成
├── requirements.txt         # Python 依赖
├── Dockerfile               # Docker 镜像（CPU）
├── Dockerfile.gpu           # Docker 镜像（GPU）
├── docker-compose.gpu.yml   # GPU Docker Compose 配置
├── start.sh                 # 启动脚本
├── entrypoint.sh            # Docker 入口脚本
├── data/                    # 数据目录（向量库、上传文件等）
└── source/                  # 源文件目录
```

---

## 技术栈

### 核心框架
- **LangChain** + **LangGraph** — LLM 应用编排与工作流状态管理
- **FastAPI** + **Uvicorn** — 高性能异步 Web 服务
- **React 18** + **Tailwind CSS** — 现代化前端界面

### 语言模型
- **通义千问 (Qwen)** — 推荐 LLM 后端
- **Ollama** — 本地 LLM 推理引擎
- **DeepSeek** — 可选 LLM 后端

### 向量数据库与嵌入
- **Milvus** — 向量数据库（支持 Lite/Server/Zilliz Cloud 三种模式）
- **BAAI/bge-m3** — 嵌入模型（8192 长度，中英双语）
- **BAAI/bge-reranker-base** — CrossEncoder 重排模型

### 检索与重排
- **BM25** (rank-bm25) — 关键词检索
- **Elasticsearch** — 大规模 BM25 检索（可选）
- **CrossEncoder** — 精确重排序

### 幻觉检测
- **NLI 模型** (`nli-deberta-v3-xsmall`) — 自然语言推理检测
- **Vectara HHEM** — 专业幻觉评估模型

### 知识图谱
- **NetworkX** — 图结构管理
- **python-louvain** / **leidenalg** — 社区检测
- **Neo4j** — 图数据库（可选）

### 监控与评估
- **LangSmith** — 全链路追踪与性能监控
- **scikit-learn** — 检索评估指标计算

---

## 工作流节点详解

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

---

## 质量保证机制

### 多层验证
1. **文档相关性评分** — 过滤不相关文档
2. **答案质量评分** — 验证答案有用性
3. **幻觉检测** — 确保答案基于源文档（NLI + Vectara 双重检测）

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
