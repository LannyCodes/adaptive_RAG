"""
Kaggle RAG 服务验证启动器
========================
在 Kaggle 上一键启动 server.py（RAG 工作流 Web 服务），并验证功能。

用法（在 Kaggle Notebook 的一个 cell 中执行）:
    !python kaggle_run_server.py

脚本会依次完成:
    1. 下载前端静态资源到 ./static 与 ./webfonts（否则 Web 页面白屏）
    2. 确保 Ollama 服务运行且模型就绪（LLM_BACKEND=ollama）
    3. 安装/校验 pyngrok
    4. 后台启动 server.py（uvicorn, 0.0.0.0:8000）
    5. 冒烟测试（POST /api/chat/stream，打印 SSE 事件流）
    6. 建立 ngrok 隧道并打印公开访问地址
    7. 阻塞保持服务存活（Ctrl+C / 中断 cell 停止）

前置条件:
    * 项目代码已在 Kaggle 工作目录（脚本会自动定位到自身所在目录）
    * 已执行 `pip install -r requirements.txt`
    * Kaggle Settings → Internet 已开启（ngrok / Zilliz / CDN / 模型下载均需联网）
    * .env 已配置（TAVILY_API_KEY 等），LLM_BACKEND=ollama
    * ngrok token：在 Kaggle Add-ons → Secrets 添加 NGROK_AUTH_TOKEN
      （免费注册: https://dashboard.ngrok.com/signup）

说明:
    * server.py 跑固定 DAG RAG 工作流；main.py 可用 ENABLE_AGENT_MODE=agent
      跑单智能体 AgentRuntime。
    * 冒烟测试直接调用 API 打印 SSE 事件，即使前端资源下载不全也能验证后端逻辑。
"""

import os
import sys
import time
import json
import threading
import subprocess

# ── 定位项目根目录（脚本所在目录），保证相对路径与 .env 加载正确 ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

# ── 抑制无关警告 ──
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

SERVER_PORT = 8000
BASE_URL = f"http://127.0.0.1:{SERVER_PORT}"


def log_phase(title: str):
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64, flush=True)


# ============================================================
# 阶段 1: 下载前端静态资源
# ============================================================

# (目标相对路径, 下载 URL) —— 全部 best-effort，单个失败不中断
STATIC_FILES = [
    ("static/react.production.min.js", "https://unpkg.com/react@18.3.1/umd/react.production.min.js"),
    ("static/react-dom.production.min.js", "https://unpkg.com/react-dom@18.3.1/umd/react-dom.production.min.js"),
    ("static/babel.min.js", "https://unpkg.com/@babel/standalone@7.24.0/babel.min.js"),
    ("static/tailwind.min.js", "https://cdn.tailwindcss.com"),
    ("static/marked.min.js", "https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"),
    ("static/katex.min.css", "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"),
    ("static/katex.min.js", "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"),
    ("static/auto-render.min.js", "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"),
    ("static/fontawesome.min.css", "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css"),
]

# fontawesome webfonts（CSS 通过 ../webfonts/ 相对引用 → 挂载在 /webfonts）
WEBFONT_FILES = [
    ("webfonts/fa-solid-900.woff2", "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/webfonts/fa-solid-900.woff2"),
    ("webfonts/fa-regular-400.woff2", "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/webfonts/fa-regular-400.woff2"),
    ("webfonts/fa-brands-400.woff2", "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/webfonts/fa-brands-400.woff2"),
    ("webfonts/fa-solid-900.ttf", "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/webfonts/fa-solid-900.ttf"),
    ("webfonts/fa-regular-400.ttf", "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/webfonts/fa-regular-400.ttf"),
    ("webfonts/fa-brands-400.ttf", "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/webfonts/fa-brands-400.ttf"),
]

# KaTeX 数学字体（CSS 通过 fonts/ 相对引用 → 放在 static/fonts/）—— best-effort
KATEX_FONTS = [
    "KaTeX_AMS-Regular", "KaTeX_Caligraphic-Bold", "KaTeX_Caligraphic-Regular",
    "KaTeX_Fraktur-Bold", "KaTeX_Fraktur-Regular", "KaTeX_Main-Bold",
    "KaTeX_Main-BoldItalic", "KaTeX_Main-Italic", "KaTeX_Main-Regular",
    "KaTeX_Math-BoldItalic", "KaTeX_Math-Italic", "KaTeX_SansSerif-Bold",
    "KaTeX_SansSerif-Italic", "KaTeX_SansSerif-Regular", "KaTeX_Script-Regular",
    "KaTeX_Size1-Regular", "KaTeX_Size2-Regular", "KaTeX_Size3-Regular",
    "KaTeX_Size4-Regular", "KaTeX_Typewriter-Regular",
]
KATEX_FONT_BASE = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/fonts"


def _download(url: str, dest: str) -> bool:
    """下载单个文件（已存在则跳过）。成功返回 True。"""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    try:
        import requests
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        print(f"  ⚠️ 下载失败 {os.path.basename(dest)}: {e}")
        return False


def download_frontend_assets():
    log_phase("阶段 1/6 · 下载前端静态资源")
    ok, fail = 0, 0

    print("→ 核心 JS/CSS ...")
    for rel, url in STATIC_FILES:
        if _download(url, os.path.join(PROJECT_ROOT, rel)):
            ok += 1
        else:
            fail += 1

    print("→ fontawesome 图标字体 ...")
    for rel, url in WEBFONT_FILES:
        if _download(url, os.path.join(PROJECT_ROOT, rel)):
            ok += 1
        else:
            fail += 1

    print("→ KaTeX 数学字体 ...")
    for name in KATEX_FONTS:
        rel = f"static/fonts/{name}.woff2"
        url = f"{KATEX_FONT_BASE}/{name}.woff2"
        if _download(url, os.path.join(PROJECT_ROOT, rel)):
            ok += 1
        else:
            fail += 1

    print(f"✅ 静态资源就绪: 成功 {ok} 个, 失败 {fail} 个")
    if fail:
        print("   （失败项多为字体，不影响多智能体后端验证；UI 可能缺少部分图标/数学字体）")


# ============================================================
# 阶段 2: 确保 Ollama 服务与模型就绪
# ============================================================

def _ollama_alive() -> bool:
    try:
        import requests
        return requests.get("http://127.0.0.1:11434/api/tags", timeout=3).status_code == 200
    except Exception:
        return False


def ensure_ollama():
    log_phase("阶段 2/6 · 确保 Ollama 就绪")
    try:
        from config import LLM_BACKEND, LOCAL_LLM, LIGHT_LLM
    except Exception as e:
        print(f"  ⚠️ 无法读取 config（{e}），跳过 Ollama 检查")
        return

    if LLM_BACKEND != "ollama":
        print(f"  ℹ️ LLM_BACKEND={LLM_BACKEND}（非 ollama），跳过 Ollama 启动")
        return

    if not _ollama_alive():
        print("  → 启动 ollama serve ...")
        try:
            subprocess.Popen(["ollama", "serve"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"  ❌ 启动 ollama 失败: {e}")
            print("     请确认 Kaggle 已安装 ollama（!curl -fsSL https://ollama.com/install.sh | sh）")
            return
        for _ in range(15):
            if _ollama_alive():
                break
            time.sleep(1)
        if not _ollama_alive():
            print("  ❌ Ollama 服务未在 15s 内就绪，请手动检查")
            return
    print("  ✅ Ollama 服务运行中")

    # 检查/拉取模型
    for model in {LOCAL_LLM, LIGHT_LLM}:
        if not model:
            continue
        try:
            import requests
            tags = requests.get("http://127.0.0.1:11434/api/tags", timeout=5).json()
            present = any(m.get("name", "").startswith(model.split(":")[0]) for m in tags.get("models", []))
            if present:
                print(f"  ✅ 模型已存在: {model}")
            else:
                print(f"  → 拉取模型 {model}（首次较大，请耐心等待）...")
                subprocess.run(["ollama", "pull", model], check=False)
        except Exception as e:
            print(f"  ⚠️ 模型检查/拉取异常（{model}）: {e}")


# ============================================================
# 阶段 3: 安装/校验 pyngrok
# ============================================================

def ensure_pyngrok():
    log_phase("阶段 3/6 · 校验 pyngrok")
    try:
        import pyngrok  # noqa: F401
        print("  ✅ pyngrok 已安装")
        return True
    except ImportError:
        print("  → 安装 pyngrok ...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyngrok"], check=True)
            print("  ✅ pyngrok 安装完成")
            return True
        except Exception as e:
            print(f"  ⚠️ pyngrok 安装失败: {e}（将跳过 ngrok 隧道，仅本地访问）")
            return False


# ============================================================
# 阶段 4: 后台启动 server.py
# ============================================================

def start_server():
    log_phase("阶段 4/6 · 启动 server.py（多智能体）")
    from server import app
    import uvicorn

    def _run():
        uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, log_level="info")

    threading.Thread(target=_run, daemon=True).start()
    print(f"  → 等待服务就绪 ({BASE_URL}/api/health) ...")

    import requests
    for i in range(60):
        try:
            if requests.get(f"{BASE_URL}/api/health", timeout=2).status_code == 200:
                print("  ✅ server.py 已就绪")
                return True
        except Exception:
            pass
        time.sleep(1)
    print("  ⚠️ 60s 内未通过健康检查，仍尝试继续（首个请求会触发 RAG 系统初始化，可能较慢）")
    return False


# ============================================================
# 阶段 5: 冒烟测试（直接调用 SSE，打印工作流事件流）
# ============================================================

def smoke_test_workflow():
    log_phase("阶段 5/6 · 冒烟测试")
    print("  发送研究类问题，观察 RAG 工作流事件流")
    print("  （首次请求触发 RAG 系统初始化 + Ollama 推理，可能需 1-3 分钟）\n")

    question = "AI Agent 的核心组成部分有哪些？请简要说明。"
    try:
        import requests
        resp = requests.post(
            f"{BASE_URL}/api/chat/stream",
            json={"message": question, "session_id": "kaggle_smoke_test"},
            stream=True,
            timeout=600,
        )
        event_types = set()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            print(f"  📨 {payload}")
            try:
                etype = json.loads(payload).get("type")
                if etype:
                    event_types.add(etype)
            except Exception:
                pass

        print("\n  " + "-" * 50)
        print(f"  观测到的事件类型: {sorted(event_types)}")
        # 工作流特征事件: progress(节点进度) / token(答案流) / done(完成)
        workflow_signals = event_types & {"progress", "token", "done"}
        if workflow_signals:
            print(f"  ✅ 检测到工作流特征事件: {sorted(workflow_signals)} → RAG 工作流生效")
        else:
            print("  ⚠️ 未检测到预期事件，请检查服务日志")
    except Exception as e:
        print(f"  ❌ 冒烟测试失败: {e}")
        print("     仍可通过下方 Web 地址手动验证")


# ============================================================
# 阶段 6: ngrok 隧道
# ============================================================

def _get_ngrok_token() -> str:
    # 1) Kaggle Secrets
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("NGROK_AUTH_TOKEN")
        if token:
            print("  ✅ 从 Kaggle Secrets 读取 NGROK_AUTH_TOKEN")
            return token
    except Exception:
        pass
    # 2) 环境变量
    token = os.environ.get("NGROK_AUTH_TOKEN", "")
    if token:
        print("  ✅ 从环境变量读取 NGROK_AUTH_TOKEN")
    return token


def start_ngrok():
    log_phase("阶段 6/6 · 建立 ngrok 隧道")
    token = _get_ngrok_token()
    if not token:
        print("  ⚠️ 未找到 NGROK_AUTH_TOKEN，跳过隧道。")
        print("     添加方式: Kaggle Add-ons → Secrets → 新增 NGROK_AUTH_TOKEN")
        print(f"     服务仍在本机 {BASE_URL} 运行（可用 Kaggle 端口转发或 API 直接验证）")
        return None
    try:
        from pyngrok import ngrok, conf
        conf.get_default().auth_token = token
        ngrok.kill()
        public_url = ngrok.connect(SERVER_PORT, bind_tls=True).public_url
        print("\n  " + "=" * 56)
        print(f"  🌐 公开访问地址: {public_url}")
        print("  " + "=" * 56)
        print(f"  💬 Web 界面（多智能体 + 审批卡片）: {public_url}")
        print(f"  📋 API 文档: {public_url}/docs")
        print("  ⚠️ 链接仅在当前 Kaggle 会话运行期间有效")
        return public_url
    except Exception as e:
        print(f"  ❌ ngrok 隧道建立失败: {e}")
        return None


# ============================================================
# 主流程
# ============================================================

def main():
    print("🚀 Kaggle 多智能体验证启动器")
    print(f"   项目根目录: {PROJECT_ROOT}")

    download_frontend_assets()
    ensure_ollama()
    ensure_pyngrok()
    start_server()
    smoke_test_workflow()
    start_ngrok()

    log_phase("✅ 启动完成 · 服务保持运行中")
    print("  按 Ctrl+C 或中断当前 cell 停止服务。\n")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\n👋 已停止。")


if __name__ == "__main__":
    main()
