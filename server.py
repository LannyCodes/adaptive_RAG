"""
FastAPI + React 18 单文件全栈应用
专为 Kaggle/Colab 环境设计，展示企业级前后端分离架构

功能特点：
1. 后端：FastAPI (异步、高性能、自动文档)
2. 前端：React 18 + Tailwind CSS (现代化UI、组件化)
3. 部署：单文件运行，自动处理静态资源，支持 ngrok 穿透

使用方法：
python server.py
"""

# ── 环境变量 —— 必须在所有 import 之前设置 ──
import os

# 1. 抑制 CUDA 插件重复注册警告 (cuFFT/cuDNN/cuBLAS)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# 2. 抑制 absl 日志输出到 STDERR
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"

# 3. 抑制 protobuf MessageFactory.GetPrototype 错误
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# 4. 抑制 HuggingFace tokenizers fork 死锁警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import uvicorn
import subprocess
import time
import threading
import requests
import uuid
import json
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import shutil

# 导入项目核心模块
# 确保项目根目录在 sys.path 中
sys.path.append(os.getcwd())

try:
    from config import ENABLE_MULTIMODAL, LOCAL_LLM, LLM_BACKEND
except Exception:
    ENABLE_MULTIMODAL = False
    LOCAL_LLM = "qwen2:1.5b"
    LLM_BACKEND = "ollama"


def ensure_ollama_service(model_name: str):
    if LLM_BACKEND != "ollama":
        return
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        if response.status_code == 200:
            return
    except Exception:
        pass
    user_home = os.environ.get("HOME", "/root")
    ollama_models_dir = os.path.join(user_home, ".ollama/models")
    os.makedirs(ollama_models_dir, exist_ok=True)
    os.environ["OLLAMA_MODELS"] = ollama_models_dir
    os.environ["OLLAMA_HOST"] = "127.0.0.1:11434"
    try:
        subprocess.Popen(["ollama", "serve"])
    except Exception as e:
        print(f"❌ 启动 Ollama 失败: {e}")
        return
    time.sleep(5)
    def pull_model():
        try:
            subprocess.run(["ollama", "pull", model_name], check=False)
        except Exception as e:
            print(f"⚠️ 下载模型失败: {e}")
    threading.Thread(target=pull_model, daemon=True).start()

# ============================================================
# 会话管理器
# ============================================================

class SessionManager:
    """管理对话历史"""
    def __init__(self):
        self._sessions = {}

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = {
            "history": [],
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "title": "新对话"
        }
        return session_id

    def get_history(self, session_id: str) -> list:
        session = self._sessions.get(session_id)
        return session["history"] if session else []

    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "history": [], "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "title": "新对话"
            }
        self._sessions[session_id]["history"].append({
            "role": role, "content": content,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        if role == "user" and self._sessions[session_id]["title"] == "新对话":
            title = content[:30]
            self._sessions[session_id]["title"] = title + ("..." if len(content) > 30 else "")

    def list_sessions(self) -> list:
        return [
            {"id": sid, "title": s["title"], "created_at": s["created_at"],
             "msg_count": len(s["history"]) // 2}
            for sid, s in sorted(self._sessions.items(),
                                 key=lambda x: x[1]["created_at"], reverse=True)
        ]

    def delete_session(self, session_id: str):
        self._sessions.pop(session_id, None)

    def get_session(self, session_id: str) -> dict:
        return self._sessions.get(session_id)


# ============================================================
# 并发限流器
# ============================================================

class RateLimiter:
    """并发限流器（支持队列统计和超时）"""
    def __init__(self, max_concurrent: int = 4, max_queue: int = 20):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent
        self.max_queue = max_queue
        self._active = 0
        self._pending = 0
        self._total_processed = 0
        self._lock = threading.Lock()
        print(f"  ⏱️  速率限制器: 最大并发 {max_concurrent}, 队列上限 {max_queue}")

    @property
    def active(self) -> int:
        return self._active

    @property
    def pending(self) -> int:
        return self._pending

    async def __aenter__(self):
        with self._lock:
            if self._pending + self._active >= self.max_concurrent + self.max_queue:
                raise HTTPException(status_code=429, detail="系统繁忙，请稍后重试")
            self._pending += 1
        await self.semaphore.acquire()
        with self._lock:
            self._pending -= 1
            self._active += 1
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        with self._lock:
            self._active -= 1
            self._total_processed += 1
        self.semaphore.release()

    def stats(self) -> dict:
        with self._lock:
            return {
                "max_concurrent": self.max_concurrent,
                "max_queue": self.max_queue,
                "active": self._active,
                "pending": self._pending,
                "total_processed": self._total_processed,
            }

# ============================================================
# 缓存管理器
# ============================================================

from cache_manager import CacheManager
cache_manager = CacheManager(cache_dir="./data/cache")
print("  💾 缓存管理器已就绪")

# ============================================================
# 1. FastAPI 后端定义
# ============================================================

app = FastAPI(
    title="Adaptive RAG Enterprise API",
    description="基于 FastAPI 和 React 构建的企业级 RAG 系统演示",
    version="1.0.0"
)

# 允许跨域 (虽然单体部署不需要，但为了开发规范加上)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 调试路由 (Added for ModelScope troubleshooting) ---

@app.get("/debug/logs", response_class=PlainTextResponse)
async def get_debug_logs():
    """直接在网页查看运行日志"""
    log_files = ["server.log", "ollama.log", "startup.log"]
    content = ""
    for log_file in log_files:
        if os.path.exists(log_file):
            content += f"\n{'='*20} {log_file} {'='*20}\n"
            try:
                with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                    content += f.read()
            except Exception as e:
                content += f"Error reading file: {e}\n"
        else:
            content += f"\n{'='*20} {log_file} (Not Found) {'='*20}\n"
    
    if not content.strip():
        content = "No log files found. The application might be starting up or logs are redirected to stdout only."
    
    return content

@app.get("/debug/files", response_class=PlainTextResponse)
async def list_files():
    """查看文件系统，确认模型是否下载"""
    output = "--- /app Directory ---\n"
    try:
        output += subprocess.check_output(["ls", "-R", "/app"], text=True, stderr=subprocess.STDOUT)
    except Exception as e:
        output += f"Error listing /app: {str(e)}"
        
    # 动态获取模型目录
    user_home = os.environ.get("HOME", "/root")
    models_dir = os.path.join(user_home, ".ollama")
    
    output += f"\n\n--- {models_dir} Directory ---\n"
    try:
        output += subprocess.check_output(["ls", "-R", models_dir], text=True, stderr=subprocess.STDOUT)
    except Exception as e:
        output += f"Error listing {models_dir}: {str(e)}"
    
    return output

# --- 结束调试路由 ---

# 全局 RAG 系统实例
rag_system = None

# 会话管理器和限流器
session_manager = SessionManager()
rate_limiter = RateLimiter(max_concurrent=4)

def get_rag_system():
    global rag_system
    if rag_system is None:
        try:
            print("🔄 初始化 RAG 系统...")
            ensure_ollama_service(LOCAL_LLM)
            from main import AdaptiveRAGSystem
            rag_system = AdaptiveRAGSystem()
            print("✅ RAG 系统初始化完成")
        except Exception as e:
            print(f"❌ RAG 系统初始化失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    return rag_system

# --- 数据模型 ---

class ChatRequest(BaseModel):
    message: str
    history: List[dict] = Field(default_factory=list)
    session_id: Optional[str] = Field(default=None)

class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    metrics: Optional[dict] = Field(default=None)
    images: List[str] = Field(default_factory=list)

# --- API 路由 ---

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """文件上传接口（所有文件即时解析+向量化，图片额外 OCR+VLM）"""
    try:
        # 确保上传目录存在
        upload_dir = "./data/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 检查文件扩展名是否支持
        ext = os.path.splitext(file.filename)[1].lower()
        from upload_and_index import LOADERS, load_file, LatexAwareTextSplitter
        from config import CHUNK_SIZE, CHUNK_OVERLAP
        
        if ext not in LOADERS:
            return {
                "filename": file.filename,
                "status": "success",
                "message": f"文件已上传，但不支持 {ext} 格式的自动索引（支持: {', '.join(sorted(set(LOADERS.keys())))})",
                "indexed": False,
            }
        
        if not rag_system or not hasattr(rag_system, 'doc_processor'):
            return {"filename": file.filename, "status": "success", "message": "文件已上传，RAG 系统未就绪，稍后将自动索引", "indexed": False}
        
        try:
            # 检查文件是否已索引过（去重）
            existing = rag_system.doc_processor.check_existing_urls([file_path])
            if file_path in existing:
                return {
                    "filename": file.filename,
                    "status": "success",
                    "message": "文件已存在，无需重复索引",
                    "indexed": False,
                    "duplicate": True,
                }
            
            # 解析文件（图片自动走 OCR+VLM 双通道，其他走对应加载器）
            docs = load_file(file_path)
            if not docs:
                return {"filename": file.filename, "status": "success", "message": "文件已上传，但未提取到内容", "indexed": False}
            
            # 检测是否包含 LaTeX 公式，选择分块器
            has_latex = any(
                d.metadata.get("has_latex") or "$$" in d.page_content or (d.page_content.count("$") >= 2)
                for d in docs
            )
            if has_latex:
                splitter = LatexAwareTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            else:
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            
            doc_splits = splitter.split_documents(docs)
            rag_system.doc_processor.add_documents_to_vectorstore(doc_splits)
            
            # 生成友好提示
            file_type_desc = "图片(OCR+VLM)" if docs[0].metadata.get("file_type") == "image" else ext.lstrip(".")
            return {
                "filename": file.filename,
                "status": "success",
                "message": f"文件上传并索引成功 ({file_type_desc}, {len(docs)} 段→{len(doc_splits)} 块)",
                "indexed": True,
                "doc_count": len(doc_splits),
            }
        except Exception as idx_e:
            import traceback
            traceback.print_exc()
            return {"filename": file.filename, "status": "success", "message": f"文件上传成功，但索引失败: {str(idx_e)}", "indexed": False}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")


@app.get("/api/images/{image_path:path}")
async def serve_image(image_path: str):
    """提供图片文件访问"""
    from fastapi.responses import FileResponse
    
    # 尝试多个可能的位置
    search_paths = [
        os.path.join("./data/images", image_path),
        os.path.join("./data/uploads", image_path),
        image_path,  # 绝对路径
    ]
    
    for path in search_paths:
        if os.path.exists(path) and os.path.isfile(path):
            return FileResponse(path)
    
    raise HTTPException(status_code=404, detail=f"图片未找到: {image_path}")

# ============================================================
# SSE 流式聊天接口
# ============================================================

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """SSE 流式聊天接口（带缓存）"""
    system = get_rag_system()
    session_id = request.session_id
    question = request.message

    async def event_generator():
        # 1. 检查缓存
        cached = cache_manager.get_answer(question)
        if cached is not None:
            yield f"data: {json.dumps({'type': 'progress', 'content': '⚡ 命中缓存'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'answer'}, ensure_ascii=False)}\n\n"
            chunk_size = 4
            for i in range(0, len(cached), chunk_size):
                yield f"data: {json.dumps({'type': 'token', 'content': cached[i:i+chunk_size]}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.002)
            yield f"data: {json.dumps({'type': 'done', 'content': cached}, ensure_ascii=False)}\n\n"
            if session_id:
                session_manager.add_message(session_id, "user", question)
                session_manager.add_message(session_id, "assistant", cached)
            return

        try:
            async with rate_limiter:
                if session_id:
                    session_manager.add_message(session_id, "user", question)

                async for event in system.stream_query(question):
                    data = json.dumps(event, ensure_ascii=False)
                    yield f"data: {data}\n\n"

                    if event.get("type") == "done" and event.get("content"):
                        # 写入缓存
                        cache_manager.set_answer(question, event["content"])
                        if session_id:
                            session_manager.add_message(session_id, "assistant", event["content"])

        except HTTPException:
            raise
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no",
                                      "Connection": "keep-alive"})


# ============================================================
# 会话管理 API
# ============================================================

@app.post("/api/session/create")
async def create_session():
    """创建新会话"""
    session_id = session_manager.create_session()
    return {"session_id": session_id}


@app.get("/api/session/list")
async def list_sessions():
    """获取会话列表"""
    return {"sessions": session_manager.list_sessions()}


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """获取会话详情"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"session": session}


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    session_manager.delete_session(session_id)
    return {"status": "ok"}


# ============================================================
# 文档管理 API
# ============================================================

@app.get("/api/documents")
async def list_documents():
    """列出已索引的文档"""
    try:
        dp = get_rag_system().doc_processor
        docs = dp.list_documents()
        return {"documents": docs, "count": len(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/documents/{source:path}")
async def delete_document(source: str):
    """删除指定文档的所有索引块"""
    try:
        dp = get_rag_system().doc_processor
        deleted = dp.delete_document(source)
        if deleted == 0:
            raise HTTPException(status_code=404, detail="文档未找到")
        # 清除相关缓存
        cache_manager.clear_all()
        return {"deleted": deleted, "source": source}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 系统状态 API
# ============================================================

@app.get("/api/stats")
async def system_stats():
    """系统运行状态统计"""
    stats = {
        "rate_limiter": rate_limiter.stats(),
        "cache": cache_manager.stats(),
        "sessions": len(session_manager.list_sessions()),
        "uvicorn": {
            "host": "0.0.0.0",
            "port": 8000,
        }
    }
    # 尝试获取文档数
    try:
        dp = get_rag_system().doc_processor
        stats["documents"] = len(dp.list_documents())
    except Exception:
        stats["documents"] = -1
    return stats


# ============================================================
# 更新现有聊天接口以支持会话记录
# ============================================================

# 修改 /api/chat 端点，在返回前记录会话
# 通过 monkey-patch 原路由处理函数来添加会话记录
# 由于 FastAPI 路由已定义，我们直接修改响应处理

@app.post("/api/chat")
async def chat_endpoint_with_session(request: ChatRequest):
    """聊天接口（支持缓存和会话记录）"""
    system = get_rag_system()
    session_id = request.session_id
    question = request.message

    # 1. 检查缓存
    cached = cache_manager.get_answer(question)
    if cached is not None:
        if session_id:
            session_manager.add_message(session_id, "user", question)
            session_manager.add_message(session_id, "assistant", cached)
        return ChatResponse(answer=cached, sources=["⚡ 命中缓存"])

    if session_id:
        session_manager.add_message(session_id, "user", question)

    try:
        async with rate_limiter:
            result = await system.query(question)

        answer = result.get("answer", "无法生成回答")

        # 写入缓存
        if answer:
            cache_manager.set_answer(question, answer)

        answer = result.get("answer", "无法生成回答")
        source_documents = result.get("source_documents", [])
        sources = [doc.page_content[:200] + "..." for doc in source_documents]
        metrics = result.get("retrieval_metrics", {})

        images = []
        if ENABLE_MULTIMODAL:
            for doc in source_documents:
                stored_path = doc.metadata.get("stored_image_path", "")
                if stored_path:
                    images.append(f"/api/images/{os.path.basename(stored_path)}")
                elif doc.metadata.get("data_type") == "image" or doc.metadata.get("file_type") == "image":
                    source = doc.metadata.get("source", "")
                    if source:
                        images.append(f"/api/images/{os.path.basename(source)}")
            for img_path in result.get("image_paths", []):
                img_url = f"/api/images/{os.path.basename(img_path)}"
                if img_url not in images:
                    images.append(img_url)

        if session_id:
            session_manager.add_message(session_id, "assistant", answer)

        return ChatResponse(answer=answer, sources=sources, metrics=metrics, images=images)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")


@app.get("/api/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok", "service": "Adaptive RAG", "multimodal": ENABLE_MULTIMODAL}


@app.get("/favicon.ico", response_class=HTMLResponse)
async def favicon():
    """返回空 favicon 避免 404"""
    return HTMLResponse(content="")

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enterprise RAG System (React)</title>
    
    <!-- 引入 React 和 ReactDOM（国内 cdn.staticfile.org） -->
    <script crossorigin src="https://cdn.staticfile.org/react/18.2.0/umd/react.production.min.js"></script>
    <script crossorigin src="https://cdn.staticfile.org/react-dom/18.2.0/umd/react-dom.production.min.js"></script>
    
    <!-- 引入 Babel 用于解析 JSX -->
    <script src="https://cdn.staticfile.org/babel-standalone/7.23.0/babel.min.js"></script>
    
    <!-- 引入 Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    
    <!-- 引入 Markdown 渲染库 -->
    <script src="https://cdn.staticfile.org/marked/9.1.0/marked.min.js"></script>
    
    <!-- 引入 KaTeX 公式渲染 -->
    <link rel="stylesheet" href="https://cdn.staticfile.org/KaTeX/0.16.9/katex.min.css">
    <script src="https://cdn.staticfile.org/KaTeX/0.16.9/katex.min.js"></script>
    <script src="https://cdn.staticfile.org/KaTeX/0.16.9/contrib/auto-render.min.js"></script>
    
    <!-- 引入 FontAwesome 图标 -->
    <link rel="stylesheet" href="https://cdn.staticfile.org/font-awesome/6.4.0/css/all.min.css">
    
    <style>
        .markdown-body p { margin-bottom: 0.5rem; }
        .markdown-body ul { list-style-type: disc; margin-left: 1.5rem; }
        .markdown-body ol { list-style-type: decimal; margin-left: 1.5rem; }
        .markdown-body pre { background-color: #f3f4f6; padding: 1rem; border-radius: 0.5rem; overflow-x: auto; }
        .markdown-body code { background-color: #f3f4f6; padding: 0.2rem 0.4rem; border-radius: 0.25rem; font-family: monospace; }
        .markdown-body .katex-display { margin: 0.8rem 0; overflow-x: auto; }
        .markdown-body .katex-inline { }
        
        /* 自定义滚动条 */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #f1f1f1; }
        ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
        
        .typing-indicator span {
            display: inline-block;
            width: 6px;
            height: 6px;
            background-color: #94a3b8;
            border-radius: 50%;
            animation: typing 1.4s infinite ease-in-out both;
            margin: 0 2px;
        }
        .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
        .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
        @keyframes typing {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
    </style>
</head>
<body class="bg-slate-50 h-screen flex flex-col font-sans text-slate-800">
    <div id="root" class="h-full w-full"></div>

    <script type="text/babel">
        const { useState, useEffect, useRef } = React;

        function App() {
            const [messages, setMessages] = useState([]);
            const [inputMessage, setInputMessage] = useState('');
            const [loading, setLoading] = useState(false);
            const [sidebarOpen, setSidebarOpen] = useState(true);
            const [showSources, setShowSources] = useState(true);
            const [showMetrics, setShowMetrics] = useState(false);
            const [multimodalEnabled, setMultimodalEnabled] = useState(false);
            const [uploadStatus, setUploadStatus] = useState(null);
            const [sessionId, setSessionId] = useState(null);
            const [sessions, setSessions] = useState([]);
            
            const chatContainerRef = useRef(null);
            const fileInputRef = useRef(null);

            // 加载会话列表
            const loadSessions = async () => {
                try {
                    const resp = await fetch('/api/session/list');
                    const data = await resp.json();
                    setSessions(data.sessions || []);
                } catch(e) {}
            };

            // 切换会话
            const switchSession = async (sid) => {
                setSessionId(sid);
                try {
                    const resp = await fetch(`/api/session/${sid}`);
                    const data = await resp.json();
                    const history = data.session?.history || [];
                    setMessages(history.map(h => ({role: h.role, content: h.content})));
                } catch(e) {
                    setMessages([]);
                }
                setSidebarOpen(false);
            };

            // 创建新会话
            const newSession = async () => {
                try {
                    const resp = await fetch('/api/session/create', { method: 'POST' });
                    const data = await resp.json();
                    setSessionId(data.session_id);
                    setMessages([]);
                    loadSessions();
                } catch(e) {}
            };

            // 删除会话
            const deleteSession = async (sid, e) => {
                e.stopPropagation();
                try {
                    await fetch(`/api/session/${sid}`, { method: 'DELETE' });
                    if (sessionId === sid) {
                        setSessionId(null);
                        setMessages([]);
                    }
                    loadSessions();
                } catch(e) {}
            };

            // 每次消息更新后渲染 LaTeX 公式
            useEffect(() => {
                if (chatContainerRef.current && typeof renderMathInElement === 'function') {
                    renderMathInElement(chatContainerRef.current, {
                        delimiters: [
                            {left: '$$', right: '$$', display: true},
                            {left: '$', right: '$', display: false},
                            {left: '\\(', right: '\\)', display: false},
                            {left: '\\[', right: '\\]', display: true}
                        ],
                        throwOnError: false
                    });
                }
            }, [messages]);

            // 初始化检查健康状态 + 加载会话列表
            useEffect(() => {
                fetch('/api/health')
                    .then(res => res.json())
                    .then(data => setMultimodalEnabled(data.multimodal))
                    .catch(err => console.error("无法连接到后端", err));
                loadSessions();
            }, []);

            // 自动滚动到底部
            useEffect(() => {
                if (chatContainerRef.current) {
                    chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
                }
            }, [messages, loading]);

            const sendMessage = async () => {
                if (!inputMessage.trim() || loading) return;

                const userMsg = inputMessage.trim();
                
                // 自动创建会话
                let sid = sessionId;
                if (!sid) {
                    try {
                        const resp = await fetch('/api/session/create', { method: 'POST' });
                        const data = await resp.json();
                        sid = data.session_id;
                        setSessionId(sid);
                    } catch(e) {}
                }

                setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
                setInputMessage('');
                setLoading(true);

                // 为流式回答准备占位
                setMessages(prev => [...prev, { role: 'assistant', content: '', isStreaming: true }]);

                try {
                    const response = await fetch('/api/chat/stream', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: userMsg, session_id: sid })
                    });

                    if (!response.ok) throw new Error('Stream request failed');

                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    let accumulated = '';

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;

                        const text = decoder.decode(value, { stream: true });
                        const lines = text.split('\n');

                        for (const line of lines) {
                            if (!line.startsWith('data: ')) continue;
                            try {
                                const event = JSON.parse(line.slice(6));
                                if (event.type === 'token') {
                                    accumulated += event.content;
                                    setMessages(prev => {
                                        const newMsgs = [...prev];
                                        if (newMsgs.length > 0) {
                                            newMsgs[newMsgs.length - 1] = { 
                                                ...newMsgs[newMsgs.length - 1], 
                                                content: accumulated 
                                            };
                                        }
                                        return newMsgs;
                                    });
                                } else if (event.type === 'done' && event.content) {
                                    // 最终完成，替换为完整内容
                                    setMessages(prev => {
                                        const newMsgs = [...prev];
                                        if (newMsgs.length > 0) {
                                            newMsgs[newMsgs.length - 1] = {
                                                role: 'assistant',
                                                content: event.content,
                                                isStreaming: false
                                            };
                                        }
                                        return newMsgs;
                                    });
                                }
                            } catch(e) {}
                        }
                    }
                    
                    loadSessions();

                } catch (error) {
                    setMessages(prev => {
                        const newMsgs = [...prev];
                        if (newMsgs.length > 0 && newMsgs[newMsgs.length - 1].isStreaming) {
                            newMsgs[newMsgs.length - 1] = {
                                role: 'assistant',
                                content: '⚠️ 系统遇到错误，请稍后重试。',
                                isStreaming: false
                            };
                        }
                        return newMsgs;
                    });
                    console.error(error);
                } finally {
                    setLoading(false);
                }
            };

            const handleFileUpload = async (event) => {
                const file = event.target.files[0];
                if (!file) return;

                const formData = new FormData();
                formData.append('file', file);

                setUploadStatus({ type: 'info', message: '正在上传...' });

                try {
                    const response = await fetch('/api/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (response.ok) {
                        setUploadStatus({ type: 'success', message: '上传成功！' });
                    } else {
                        throw new Error('Upload failed');
                    }
                } catch (error) {
                    setUploadStatus({ type: 'error', message: '上传失败' });
                }

                setTimeout(() => setUploadStatus(null), 3000);
            };

            const handleKeyDown = (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            };

            return (
                <div className="flex h-full max-w-7xl mx-auto w-full shadow-2xl bg-white overflow-hidden">
                    
                    {/* 左侧侧边栏 */}
                    <div className={`bg-slate-900 text-white flex flex-col flex-shrink-0 transition-all duration-300 ${sidebarOpen ? 'w-64' : 'w-0'}`}>
                        <div className="p-4 border-b border-slate-700 flex items-center justify-between">
                            <div className="flex items-center space-x-2 font-bold text-xl overflow-hidden whitespace-nowrap">
                                <i className="fa-solid fa-brain text-blue-400"></i>
                                <span>Adaptive RAG</span>
                            </div>
                        </div>
                        
                        <div className="flex-1 overflow-y-auto p-4 space-y-6">
                            {/* 系统状态 */}
                            <div>
                                <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">系统状态</h3>
                                <div className="flex items-center space-x-2 text-sm">
                                    <span className="w-2 h-2 rounded-full bg-green-500"></span>
                                    <span>API 服务正常</span>
                                </div>
                                <div className="flex items-center space-x-2 text-sm mt-1">
                                    <span className={`w-2 h-2 rounded-full ${multimodalEnabled ? 'bg-green-500' : 'bg-gray-500'}`}></span>
                                    <span>多模态支持: {multimodalEnabled ? '开启' : '关闭'}</span>
                                </div>
                            </div>
                            
                            {/* 会话管理 */}
                            <div>
                                <div className="flex items-center justify-between mb-2">
                                    <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">会话</h3>
                                    <button onClick={newSession}
                                        className="text-xs text-blue-400 hover:text-blue-300 transition-colors">
                                        <i className="fa-solid fa-plus mr-1"></i>新建
                                    </button>
                                </div>
                                <div className="space-y-1 max-h-40 overflow-y-auto">
                                    {sessions.length === 0 && (
                                        <p className="text-xs text-slate-500 text-center py-2">暂无会话</p>
                                    )}
                                    {sessions.map(s => (
                                        <div key={s.id}
                                            className={`flex items-center justify-between px-2 py-1.5 rounded-lg cursor-pointer text-xs transition-colors ${
                                                sessionId === s.id 
                                                    ? 'bg-blue-600 text-white' 
                                                    : 'hover:bg-slate-800 text-slate-300'
                                            }`}
                                            onClick={() => switchSession(s.id)}>
                                            <div className="flex items-center space-x-2 truncate flex-1">
                                                <i className="fa-solid fa-comment text-slate-500 flex-shrink-0"></i>
                                                <span className="truncate">{s.title}</span>
                                            </div>
                                            <button onClick={(e) => deleteSession(s.id, e)}
                                                className="text-slate-500 hover:text-red-400 ml-1 flex-shrink-0">
                                                <i className="fa-solid fa-xmark"></i>
                                            </button>
                                        </div>
                                    ))}
                                </div>
                            </div>
                            
                            {/* 知识库管理 */}
                            <div>
                                <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">知识库管理</h3>
                                <div 
                                    className="bg-slate-800 rounded-lg p-3 text-center border border-dashed border-slate-600 hover:border-blue-400 transition-colors cursor-pointer"
                                    onClick={() => fileInputRef.current.click()}
                                >
                                    <input 
                                        type="file" 
                                        ref={fileInputRef} 
                                        className="hidden" 
                                        onChange={handleFileUpload} 
                                    />
                                    <i className="fa-solid fa-cloud-arrow-up text-2xl text-slate-400 mb-2"></i>
                                    <p className="text-xs text-slate-300">点击上传 PDF/图片</p>
                                </div>
                                {uploadStatus && (
                                    <p className={`text-xs mt-2 text-center ${uploadStatus.type === 'success' ? 'text-green-400' : 'text-red-400'}`}>
                                        {uploadStatus.message}
                                    </p>
                                )}
                            </div>
                            
                            {/* 设置 */}
                            <div>
                                <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">设置</h3>
                                <div className="flex items-center justify-between text-sm py-1">
                                    <span>显示检索源</span>
                                    <button 
                                        className={`w-8 h-4 rounded-full relative transition-colors ${showSources ? 'bg-blue-600' : 'bg-slate-700'}`}
                                        onClick={() => setShowSources(!showSources)}
                                    >
                                        <span className={`absolute top-0.5 left-0.5 w-3 h-3 bg-white rounded-full transition-transform ${showSources ? 'translate-x-4' : ''}`}></span>
                                    </button>
                                </div>
                                <div className="flex items-center justify-between text-sm py-1">
                                    <span>显示指标</span>
                                    <button 
                                        className={`w-8 h-4 rounded-full relative transition-colors ${showMetrics ? 'bg-blue-600' : 'bg-slate-700'}`}
                                        onClick={() => setShowMetrics(!showMetrics)}
                                    >
                                        <span className={`absolute top-0.5 left-0.5 w-3 h-3 bg-white rounded-full transition-transform ${showMetrics ? 'translate-x-4' : ''}`}></span>
                                    </button>
                                </div>
                            </div>
                        </div>
                        
                        <div className="p-4 border-t border-slate-700 text-xs text-slate-500 text-center">
                            Enterprise Edition v1.0
                        </div>
                    </div>

                    {/* 主聊天区域 */}
                    <div className="flex-1 flex flex-col h-full bg-slate-50 relative">
                        
                        {/* 顶部导航栏 */}
                        <div className="h-16 bg-white border-b border-slate-200 flex items-center justify-between px-6 shadow-sm z-10">
                            <div className="flex items-center">
                                <button onClick={() => setSidebarOpen(!sidebarOpen)} className="text-slate-500 hover:text-slate-700 focus:outline-none mr-4">
                                    <i className="fa-solid fa-bars text-xl"></i>
                                </button>
                                <h1 className="text-lg font-semibold text-slate-700">智能知识库助手</h1>
                                <button onClick={newSession}
                                    className="ml-3 px-3 py-1 text-xs text-blue-600 border border-blue-200 rounded-lg hover:bg-blue-50 transition-colors">
                                    <i className="fa-solid fa-plus mr-1"></i>新会话
                                </button>
                                {loading && (
                                    <span className="ml-3 text-xs text-slate-400">
                                        <i className="fa-solid fa-spinner fa-spin mr-1"></i>处理中...
                                    </span>
                                )}
                            </div>
                            <div className="flex items-center space-x-4">
                                <a href="/docs" target="_blank" className="text-sm text-blue-600 hover:text-blue-800 font-medium">
                                    <i className="fa-solid fa-book-open mr-1"></i> API 文档
                                </a>
                            </div>
                        </div>

                        {/* 聊天记录 */}
                        <div className="flex-1 overflow-y-auto p-6 space-y-6 scroll-smooth" ref={chatContainerRef}>
                            
                            {messages.length === 0 && (
                                <div className="flex flex-col items-center justify-center h-full text-center space-y-4 opacity-50">
                                    <div className="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center text-blue-500 text-4xl mb-4">
                                        <i className="fa-solid fa-robot"></i>
                                    </div>
                                    <h2 className="text-2xl font-bold text-slate-700">有什么可以帮您？</h2>
                                    <p className="text-slate-500 max-w-md">我可以回答关于知识库的问题，支持多模态检索和图谱分析。</p>
                                    <div className="grid grid-cols-2 gap-4 mt-8 w-full max-w-lg">
                                        <button onClick={() => setInputMessage('GraphRAG 的核心原理是什么？')} className="p-4 bg-white border border-slate-200 rounded-xl hover:border-blue-400 hover:shadow-md transition-all text-left text-sm">
                                            GraphRAG 的核心原理是什么？
                                        </button>
                                        <button onClick={() => setInputMessage('分析这些文档的主要主题')} className="p-4 bg-white border border-slate-200 rounded-xl hover:border-blue-400 hover:shadow-md transition-all text-left text-sm">
                                            分析这些文档的主要主题
                                        </button>
                                    </div>
                                </div>
                            )}

                            {messages.map((msg, index) => (
                                <div key={index} className={`flex flex-col space-y-2 ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                                    {msg.role === 'user' ? (
                                        <div className="flex items-end space-x-2 max-w-[80%]">
                                            <div className="bg-blue-600 text-white px-5 py-3 rounded-2xl rounded-tr-none shadow-md">
                                                {msg.content}
                                            </div>
                                            <div className="w-8 h-8 rounded-full bg-slate-200 flex items-center justify-center text-slate-500 flex-shrink-0">
                                                <i className="fa-solid fa-user"></i>
                                            </div>
                                        </div>
                                    ) : (
                                        <div className="flex items-start space-x-3 max-w-[90%] w-full">
                                            <div className="w-8 h-8 rounded-full bg-green-100 flex items-center justify-center text-green-600 flex-shrink-0 mt-1">
                                                <i className="fa-solid fa-robot"></i>
                                            </div>
                                            <div className="flex flex-col space-y-2 w-full">
                                                <div 
                                                    className="bg-white border border-slate-200 px-6 py-4 rounded-2xl rounded-tl-none shadow-sm w-full markdown-body"
                                                    dangerouslySetInnerHTML={{ __html: (function() {
                                                        // 先保护 LaTeX 公式不被 marked 破坏
                                                        let text = msg.content;
                                                        const latexBlocks = [];
                                                        // 保护 $$...$$ (块级公式)
                                                        text = text.replace(/\$\$([\s\S]*?)\$\$/g, (m, p1) => {
                                                            latexBlocks.push('<span class="katex-display">' + katex.renderToString(p1.trim(), {displayMode: true, throwOnError: false}) + '</span>');
                                                            return '%%LATEX_BLOCK_' + (latexBlocks.length - 1) + '%%';
                                                        });
                                                        // 保护 $...$ (行内公式)
                                                        text = text.replace(/\$([^\$\\n]+?)\$/g, (m, p1) => {
                                                            latexBlocks.push('<span class="katex-inline">' + katex.renderToString(p1.trim(), {displayMode: false, throwOnError: false}) + '</span>');
                                                            return '%%LATEX_BLOCK_' + (latexBlocks.length - 1) + '%%';
                                                        });
                                                        // marked 渲染 Markdown
                                                        let html = marked.parse(text);
                                                        // 还原 LaTeX 公式
                                                        html = html.replace(/%%LATEX_BLOCK_(\d+)%%/g, (m, idx) => latexBlocks[parseInt(idx)]);
                                                        return html;
                                                    })() }}
                                                ></div>
                                                
                                                {showSources && msg.sources && msg.sources.length > 0 && (
                                                    <div className="bg-slate-50 border border-slate-200 rounded-xl p-3 text-xs text-slate-600 mt-2">
                                                        <div className="font-semibold mb-2 flex items-center text-slate-400">
                                                            <i className="fa-solid fa-quote-left mr-2"></i> 参考来源
                                                        </div>
                                                        {msg.sources.map((source, idx) => (
                                                            <div key={idx} className="mb-2 last:mb-0 pl-3 border-l-2 border-blue-300">
                                                                {source}
                                                            </div>
                                                        ))}
                                                    </div>
                                                )}

                                                {msg.images && msg.images.length > 0 && (
                                                    <div className="bg-slate-50 border border-slate-200 rounded-xl p-3 mt-2">
                                                        <div className="font-semibold mb-2 flex items-center text-slate-400 text-xs">
                                                            <i className="fa-solid fa-image mr-2"></i> 相关图片
                                                        </div>
                                                        <div className="flex flex-wrap gap-2">
                                                            {msg.images.map((imgUrl, idx) => (
                                                                <a key={idx} href={imgUrl} target="_blank" rel="noopener noreferrer">
                                                                    <img src={imgUrl} alt={`相关图片 ${idx+1}`}
                                                                        className="h-24 w-auto rounded-lg border border-slate-200 object-cover hover:opacity-80 transition-opacity cursor-pointer" />
                                                                </a>
                                                            ))}
                                                        </div>
                                                    </div>
                                                )}

                                                {showMetrics && msg.metrics && (
                                                    <div className="flex flex-wrap gap-2 mt-1">
                                                        <span className="px-2 py-1 bg-slate-100 rounded text-xs text-slate-500 border border-slate-200">
                                                            <i className="fa-solid fa-clock mr-1"></i> {msg.metrics.latency ? msg.metrics.latency.toFixed(3) : 0}s
                                                        </span>
                                                        <span className="px-2 py-1 bg-slate-100 rounded text-xs text-slate-500 border border-slate-200">
                                                            <i className="fa-solid fa-file-lines mr-1"></i> Docs: {msg.metrics.retrieved_docs_count || 0}
                                                        </span>
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            ))}

                            {loading && (
                                <div className="flex items-start space-x-3">
                                    <div className="w-8 h-8 rounded-full bg-green-100 flex items-center justify-center text-green-600 flex-shrink-0 mt-1">
                                        <i className="fa-solid fa-robot"></i>
                                    </div>
                                    <div className="bg-white border border-slate-200 px-5 py-3 rounded-2xl rounded-tl-none shadow-sm">
                                        <div className="typing-indicator">
                                            <span></span><span></span><span></span>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* 输入区域 */}
                        <div className="bg-white p-4 border-t border-slate-200">
                            <div className="max-w-4xl mx-auto relative">
                                <textarea 
                                    value={inputMessage}
                                    onChange={(e) => setInputMessage(e.target.value)}
                                    onKeyDown={handleKeyDown}
                                    placeholder="输入您的问题... (Shift + Enter 换行)" 
                                    className="w-full pl-4 pr-12 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none shadow-inner text-sm"
                                    rows="2"
                                ></textarea>
                                <button 
                                    onClick={sendMessage} 
                                    disabled={loading || !inputMessage.trim()}
                                    className="absolute right-2 bottom-2.5 p-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-md"
                                >
                                    <i className="fa-solid fa-paper-plane"></i>
                                </button>
                            </div>
                            <div className="text-center mt-2 text-xs text-slate-400">
                                Powered by Adaptive RAG & FastAPI & React
                            </div>
                        </div>
                    </div>
                </div>
            );
        }

        const root = ReactDOM.createRoot(document.getElementById('root'));
        root.render(<App />);
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """返回单页应用前端"""
    return HTMLResponse(content=HTML_CONTENT)

if __name__ == "__main__":
    print("="*60)
    print("🚀 企业级 RAG 服务器启动中...")
    print("   后端: FastAPI")
    print("   前端: Vue 3 + Tailwind")
    print("   地址: http://0.0.0.0:8000")
    print("   文档: http://0.0.0.0:8000/docs")
    print("="*60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
