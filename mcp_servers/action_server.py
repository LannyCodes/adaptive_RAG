"""
行动型工具 MCP Server
=====================
以 MCP 协议暴露 Agent 的"行动"能力: 代码执行 / 文件读写 / 网页抓取。

运行: python -m mcp_servers.action_server  (stdio 传输，由 MCP client 拉起)

安全设计:
1. run_python_code: 静态黑名单检查 + 独立子进程执行 + 超时 + 输出截断。
   生产环境建议进一步以 Docker/gVisor 容器隔离（本实现为进程级沙箱基线）。
2. read_file / write_file: 路径经 realpath 解析后必须位于 ACTION_WORKSPACE_DIR
   之内，防路径穿越 (../ 逃逸)。
3. browse_webpage: Playwright 为可选依赖，未安装时返回降级提示
   （Dockerfile.slim 不安装浏览器，工具自动降级）。
"""

import os
import re
import subprocess
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ── stdout 代理: 保护 stdio 协议通道（同 rag_server）──────────
class _StdoutProxy:
    def __init__(self, original):
        self._original = original
        self.buffer = original.buffer

    def write(self, s):
        return sys.stderr.write(s)

    def flush(self):
        return sys.stderr.flush()

    def fileno(self):
        return self._original.fileno()


if not isinstance(sys.stdout, _StdoutProxy):
    sys.stdout = _StdoutProxy(sys.stdout)


def _log(msg: str):
    print(f"[action_mcp] {msg}", file=sys.stderr, flush=True)


from config import ACTION_WORKSPACE_DIR, CODE_EXEC_TIMEOUT, CODE_EXEC_MAX_OUTPUT

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("action-tools")


# ── 工作目录初始化 ───────────────────────────────────────────
def _workspace() -> str:
    path = os.path.realpath(ACTION_WORKSPACE_DIR)
    os.makedirs(path, exist_ok=True)
    return path


def _safe_path(path: str) -> str:
    """将用户路径解析为工作目录内的绝对路径，越界则抛错。"""
    base = _workspace()
    resolved = os.path.realpath(os.path.join(base, path))
    if resolved != base and not resolved.startswith(base + os.sep):
        raise ValueError(f"路径越界: {path}（必须位于工作目录内）")
    return resolved


# ── 代码执行静态黑名单 ───────────────────────────────────────
_FORBIDDEN_PATTERNS = [
    r"\bos\.system\b",
    r"\bos\.popen\b",
    r"\bsubprocess\b",
    r"\bshutil\.rmtree\b",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\b__import__\b",
    r"\binput\s*\(",
    r"\bsocket\b",
    r"\brequests\b",
    r"\burllib\b",
    r"\bhttpx\b",
]


@mcp.tool()
def run_python_code(code: str, timeout: int = CODE_EXEC_TIMEOUT) -> str:
    """在沙箱中执行 Python 代码并返回 stdout/stderr。禁止网络访问、
    subprocess、os.system、eval/exec 等危险调用（静态检查）。
    工作目录为 ACTION_WORKSPACE_DIR，超时默认 30 秒。"""
    # 1. 静态黑名单检查
    for pattern in _FORBIDDEN_PATTERNS:
        if re.search(pattern, code):
            return f"拒绝执行: 代码包含被禁止的调用模式 {pattern}"

    # 2. 子进程执行（隔离解释器，限制工作目录）
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=_workspace(),
            env={"PATH": os.environ.get("PATH", "")},  # 最小环境变量
        )
        output = ""
        if proc.stdout:
            output += proc.stdout
        if proc.stderr:
            output += f"\n[stderr]\n{proc.stderr}"
        output = output.strip() or "(无输出)"
        if len(output) > CODE_EXEC_MAX_OUTPUT:
            output = output[:CODE_EXEC_MAX_OUTPUT] + "\n...(输出截断)"
        return f"退出码 {proc.returncode}\n{output}"
    except subprocess.TimeoutExpired:
        return f"执行超时（>{timeout}s），已终止"
    except Exception as e:
        return f"执行失败: {e}"


@mcp.tool()
def read_file(path: str, max_chars: int = 10000) -> str:
    """读取工作目录内文件的文本内容。path 为相对工作目录的路径。"""
    try:
        real = _safe_path(path)
        if not os.path.isfile(real):
            return f"文件不存在: {path}"
        with open(real, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(max_chars)
        if os.path.getsize(real) > max_chars:
            content += "\n...(内容截断)"
        return content
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"读取失败: {e}"


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """将文本写入工作目录内文件（自动创建父目录）。path 为相对工作目录的路径。"""
    try:
        real = _safe_path(path)
        os.makedirs(os.path.dirname(real) or _workspace(), exist_ok=True)
        with open(real, "w", encoding="utf-8") as f:
            f.write(content)
        return f"已写入 {path}（{len(content)} 字符）"
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"写入失败: {e}"


@mcp.tool()
def list_files(subdir: str = ".") -> str:
    """列出工作目录（或其子目录）内的文件与目录。"""
    try:
        real = _safe_path(subdir)
        if not os.path.isdir(real):
            return f"目录不存在: {subdir}"
        entries = sorted(os.listdir(real))
        if not entries:
            return "(空目录)"
        lines = []
        for name in entries[:200]:
            full = os.path.join(real, name)
            tag = "[dir] " if os.path.isdir(full) else "[file]"
            lines.append(f"{tag} {name}")
        return "\n".join(lines)
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"列目录失败: {e}"


@mcp.tool()
def browse_webpage(url: str, max_chars: int = 5000) -> str:
    """用真实浏览器抓取网页正文（比搜索摘要更深的页面交互）。
    需要 playwright；未安装时返回降级提示。"""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return (
            "浏览器工具不可用: 未安装 playwright。"
            "完整版镜像运行 'playwright install --with-deps chromium' 后可用；"
            "可改用 search_web 获取网页摘要。"
        )

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=30000, wait_until="domcontentloaded")
            # 去除脚本/样式后提取正文
            text = page.evaluate(
                """() => {
                    for (const el of document.querySelectorAll(
                        'script, style, noscript, nav, footer, header')) el.remove();
                    return document.body ? document.body.innerText : '';
                }"""
            )
            browser.close()
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...(内容截断)"
        return text or "(页面无文本内容)"
    except Exception as e:
        return f"网页抓取失败: {e}"


if __name__ == "__main__":
    # 双传输模式（同 rag_server）
    from config import MCP_ACTION_PORT, MCP_TRANSPORT

    if MCP_TRANSPORT == "http":
        _log(f"Action MCP Server 启动 (streamable-http, port={MCP_ACTION_PORT})，"
             f"工作目录: {_workspace()}")
        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = MCP_ACTION_PORT
        mcp.run(transport="streamable-http")
    else:
        _log(f"Action MCP Server 启动 (stdio)，工作目录: {_workspace()}")
        mcp.run(transport="stdio")
