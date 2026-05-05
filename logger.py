"""
结构化日志模块
替换全项目的 print()，支持控制台/文件/阿里云 SLS 输出
"""

import os
import sys
import json
import time
import uuid
import logging
from datetime import datetime
from typing import Optional


# ============================================================
# 日志级别
# ============================================================

class LogLevel:
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


# ============================================================
# SLS (阿里云日志服务) 推送器
# ============================================================

class SLSPusher:
    """通过 HTTP API 推送日志到阿里云 SLS"""

    def __init__(self, project: str = "", logstore: str = "",
                 endpoint: str = "", access_key: str = "", access_secret: str = ""):
        self._enabled = bool(project and logstore)
        if not self._enabled:
            return

        self._project = project
        self._logstore = logstore
        self._endpoint = endpoint.rstrip("/")
        self._access_key = access_key
        self._access_secret = access_secret
        self._session_token = ""

        # 尝试从环境变量补充
        self._project = self._project or os.environ.get("SLS_PROJECT", "")
        self._logstore = self._logstore or os.environ.get("SLS_LOGSTORE", "")
        self._endpoint = self._endpoint or os.environ.get("SLS_ENDPOINT", "")
        self._access_key = self._access_key or os.environ.get("SLS_ACCESS_KEY", "")
        self._access_secret = self._access_secret or os.environ.get("SLS_ACCESS_SECRET", "")

        self._enabled = bool(self._project and self._logstore and self._endpoint)
        if self._enabled:
            print(f"  📡 SLS 日志推送已启用: {self._project}/{self._logstore}")

    def push(self, entries: list[dict]):
        """推送日志条目到 SLS (HTTP API)"""
        if not self._enabled or not entries:
            return

        try:
            import requests
            body = {
                "__topic__": "adaptive_rag",
                "__source__": os.uname().nodename if hasattr(os, 'uname') else "unknown",
                "__logs__": [
                    {
                        "__time__": int(e.get("timestamp", time.time())),
                        **{k: str(v) for k, v in e.items() if k != "timestamp"}
                    }
                    for e in entries
                ]
            }
            # SLS HTTP POST 接口
            url = (f"https://{self._project}.{self._endpoint}"
                   f"/logstores/{self._logstore}/track")
            requests.post(url, json=body, timeout=3)
        except Exception:
            pass  # SLS 推送失败不影响主流程


# ============================================================
# 结构化日志器
# ============================================================

class StructuredLogger:
    """
    结构化日志器
    
    用法:
        log = StructuredLogger()
        log.info("system_start", "服务启动", port=8000)
        log.error("retrieval_failed", "检索失败", query=question, error=str(e))
    """

    def __init__(self, name: str = "adaptive_rag", log_file: Optional[str] = None,
                 sls_pusher: Optional[SLSPusher] = None):
        self._name = name
        self._sls = sls_pusher or SLSPusher()
        self._log_buffer: list[dict] = []
        self._buffer_size = 10  # 积攒 10 条后批量推送

        # 文件日志
        self._file_handler = None
        if log_file:
            try:
                os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
                self._file_handler = open(log_file, "a", encoding="utf-8")
            except Exception:
                pass

    def _emit(self, level: str, event: str, message: str, **extra):
        """内部：构造并输出日志条目"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "event": event,
            "message": message,
            "logger": self._name,
            "pid": os.getpid(),
        }
        entry.update(extra)

        # 1. 控制台输出（彩色）
        self._console(entry)

        # 2. 文件输出（JSON one-per-line）
        if self._file_handler:
            self._file_handler.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._file_handler.flush()

        # 3. SLS 推送（批量）
        self._log_buffer.append(entry)
        if len(self._log_buffer) >= self._buffer_size:
            self._sls.push(self._log_buffer)
            self._log_buffer.clear()

    def _console(self, entry: dict):
        """带颜色的控制台输出"""
        level_colors = {
            "DEBUG": "\033[36m",     # 青色
            "INFO": "\033[32m",      # 绿色
            "WARNING": "\033[33m",   # 黄色
            "ERROR": "\033[31m",     # 红色
            "FATAL": "\033[35m",     # 紫色
        }
        color = level_colors.get(entry["level"], "\033[0m")
        reset = "\033[0m"
        label = entry.get("event", entry["message"])[:60]
        extra = {k: v for k, v in entry.items()
                 if k not in ("timestamp", "level", "event", "message", "logger", "pid")}
        suffix = f" | {extra}" if extra else ""
        print(f"{color}[{entry['level']}]{reset} {label}{suffix}", flush=True)

    # ── 便捷方法 ──

    def debug(self, event: str, message: str = "", **extra):
        self._emit("DEBUG", event, message, **extra)

    def info(self, event: str, message: str = "", **extra):
        self._emit("INFO", event, message, **extra)

    def warning(self, event: str, message: str = "", **extra):
        self._emit("WARNING", event, message, **extra)

    def error(self, event: str, message: str = "", **extra):
        self._emit("ERROR", event, message, **extra)

    def fatal(self, event: str, message: str = "", **extra):
        self._emit("FATAL", event, message, **extra)

    def flush(self):
        """强制推送 SLS 缓冲"""
        if self._log_buffer:
            self._sls.push(self._log_buffer)
            self._log_buffer.clear()


# ============================================================
# 全局单例
# ============================================================

_logger: Optional[StructuredLogger] = None


def get_logger() -> StructuredLogger:
    global _logger
    if _logger is None:
        _logger = StructuredLogger(
            name="adaptive_rag",
            log_file="./data/logs/adaptive_rag.log",
        )
    return _logger


def setup_sls(project: str, logstore: str, endpoint: str,
              access_key: str, access_secret: str):
    """配置 SLS 日志推送"""
    global _logger
    pusher = SLSPusher(project, logstore, endpoint, access_key, access_secret)
    _logger = StructuredLogger(
        name="adaptive_rag",
        log_file="./data/logs/adaptive_rag.log",
        sls_pusher=pusher,
    )
    _logger.info("sls_configured", "SLS 日志推送已配置")
