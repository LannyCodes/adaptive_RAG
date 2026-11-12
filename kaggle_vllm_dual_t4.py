#!/usr/bin/env python3
"""Kaggle 双 T4 启动脚本：两个 vLLM 副本 + Nginx 负载均衡。

推荐配置：
  GPU0: 7B GPTQ-Int4, max-num-seqs=4
  GPU1: 7B GPTQ-Int4, max-num-seqs=8
  Nginx: 127.0.0.1:8200 -> 8001/8002，least_conn

注意：8200 是负载均衡入口；server.py 用 8000，MCP Server 用 8100/8101。

用法：
  python kaggle_vllm_dual_t4.py

可选环境变量：
  VLLM_MODEL_PATH        HuggingFace 模型路径，默认 Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4
  VLLM_SERVED_NAME       应用侧模型名，默认 qwen2.5-7b
  VLLM_BIN               vllm 可执行文件，默认优先使用 /kaggle/working/vllm-env/bin/vllm
  VLLM_MAX_MODEL_LEN     默认 8192
  VLLM_QUANTIZATION      默认 gptq；AWQ 模型设为 awq；空字符串表示不传
  VLLM_ENABLE_TOOL_CALLING  默认 true，Agent 模式需要
  VLLM_TOOL_PARSER       默认 hermes
  VLLM_GPU0_UTIL         默认 0.65（GPU0 还要跑检索模型）
  VLLM_GPU1_UTIL         默认 0.85
  VLLM_GPU0_MAX_SEQS     默认 4
  VLLM_GPU1_MAX_SEQS     默认 8
  VLLM_LB_PORT           Nginx 负载均衡端口，默认 8200
  VLLM_HEALTH_TIMEOUT    副本就绪超时（秒），默认 900
"""

import os
import shutil
import socket
import subprocess
import textwrap
import time
from pathlib import Path

import requests


WORK_DIR = Path(os.environ.get("VLLM_WORK_DIR", "/kaggle/working"))
MODEL_PATH = os.environ.get("VLLM_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4")
SERVED_NAME = os.environ.get("VLLM_SERVED_NAME", "qwen2.5-7b")
MAX_MODEL_LEN = os.environ.get("VLLM_MAX_MODEL_LEN", "8192")
QUANTIZATION = os.environ.get("VLLM_QUANTIZATION", "gptq")
API_KEY = os.environ.get("VLLM_API_KEY", "dummy")
TOOL_PARSER = os.environ.get("VLLM_TOOL_PARSER", "hermes")
ENABLE_TOOL_CALLING = os.environ.get("VLLM_ENABLE_TOOL_CALLING", "true").lower() == "true"
ENFORCE_EAGER = os.environ.get("VLLM_ENFORCE_EAGER", "true").lower() == "true"
PREFIX_CACHING = os.environ.get("VLLM_PREFIX_CACHING", "true").lower() == "true"
LB_PORT = int(os.environ.get("VLLM_LB_PORT", "8200"))
HEALTH_TIMEOUT = int(os.environ.get("VLLM_HEALTH_TIMEOUT", "900"))
NGINX_BIN = os.environ.get("NGINX_BIN", "nginx")

_DEFAULT_VLLM_BIN = "/kaggle/working/vllm-env/bin/vllm"
VLLM_BIN = os.environ.get("VLLM_BIN") or (
    _DEFAULT_VLLM_BIN if Path(_DEFAULT_VLLM_BIN).exists() else "vllm"
)

REPLICAS = [
    {
        "gpu": 0,
        "port": 8001,
        "util": float(os.environ.get("VLLM_GPU0_UTIL", "0.65")),
        "max_seqs": int(os.environ.get("VLLM_GPU0_MAX_SEQS", "4")),
        "weight": 1,
    },
    {
        "gpu": 1,
        "port": 8002,
        "util": float(os.environ.get("VLLM_GPU1_UTIL", "0.85")),
        "max_seqs": int(os.environ.get("VLLM_GPU1_MAX_SEQS", "8")),
        "weight": 2,
    },
]


def port_open(port: int) -> bool:
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def build_serve_command(replica: dict) -> list[str]:
    cmd = [
        VLLM_BIN,
        "serve",
        MODEL_PATH,
        "--served-model-name",
        SERVED_NAME,
        "--host",
        "127.0.0.1",
        "--port",
        str(replica["port"]),
        "--api-key",
        API_KEY,
        "--dtype",
        "float16",
        "--max-model-len",
        MAX_MODEL_LEN,
        "--gpu-memory-utilization",
        str(replica["util"]),
        "--max-num-seqs",
        str(replica["max_seqs"]),
        "--disable-log-requests",
    ]
    if QUANTIZATION:
        cmd.extend(["--quantization", QUANTIZATION])
    if ENFORCE_EAGER:
        cmd.append("--enforce-eager")
    if PREFIX_CACHING:
        cmd.append("--enable-prefix-caching")
    if ENABLE_TOOL_CALLING and TOOL_PARSER:
        cmd.extend(["--enable-auto-tool-choice", "--tool-call-parser", TOOL_PARSER])
    return cmd


def start_replicas() -> list[subprocess.Popen]:
    if not (Path(VLLM_BIN).exists() or shutil.which(VLLM_BIN)):
        raise RuntimeError(
            "未找到 vllm 可执行文件，请先安装，例如：\n"
            "  python -m venv /kaggle/working/vllm-env --system-site-packages\n"
            "  /kaggle/working/vllm-env/bin/pip install vllm==0.7.3"
        )

    procs = []
    logs = []
    for replica in REPLICAS:
        port = replica["port"]
        if port_open(port):
            print(f"[skip] vLLM 端口 {port} 已在监听")
            continue

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(replica["gpu"])
        env.setdefault("VLLM_USE_V1", "0")

        log_path = WORK_DIR / f"vllm-gpu{replica['gpu']}.log"
        log_file = open(log_path, "w")
        cmd = build_serve_command(replica)
        print(f"[start] GPU{replica['gpu']} -> 127.0.0.1:{port}, log={log_path}")
        procs.append(
            subprocess.Popen(
                cmd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        )
        logs.append(log_file)
    return procs


def wait_health(timeout_seconds: int = 900) -> None:
    for replica in REPLICAS:
        port = replica["port"]
        deadline = time.time() + timeout_seconds
        last_report = time.time()
        while time.time() < deadline:
            try:
                response = requests.get(
                    f"http://127.0.0.1:{port}/health",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    timeout=2,
                )
                if response.status_code == 200:
                    print(f"[ready] vLLM {port}")
                    break
            except requests.RequestException:
                pass
            if time.time() - last_report >= 30:
                log_name = f"vllm-gpu{replica['gpu']}.log"
                print(f"[wait] vLLM {port} 尚未就绪，查看 {WORK_DIR / log_name}")
                last_report = time.time()
            time.sleep(5)
        else:
            log_name = f"vllm-gpu{replica['gpu']}.log"
            raise TimeoutError(
                f"vLLM {port} 在 {timeout_seconds}s 内未就绪，"
                f"请查看 {WORK_DIR / log_name}"
            )


def write_nginx_conf() -> Path:
    upstream_lines = "\n".join(
        "            server 127.0.0.1:{port} weight={weight} "
        "max_fails=3 fail_timeout=30s;".format(**replica)
        for replica in REPLICAS
    )
    conf = textwrap.dedent(
        f"""
        worker_processes 1;
        pid {WORK_DIR}/nginx-vllm.pid;
        error_log {WORK_DIR}/nginx-vllm-error.log;

        events {{
          worker_connections 1024;
        }}

        http {{
          access_log off;

          upstream vllm {{
            least_conn;
{upstream_lines}
          }}

          server {{
            listen {LB_PORT};

            location / {{
              proxy_pass http://vllm;
              proxy_http_version 1.1;
              proxy_set_header Connection "";
              proxy_set_header Host $host;
              proxy_buffering off;
              proxy_read_timeout 600s;
              client_max_body_size 20m;
              proxy_next_upstream error timeout http_502 http_503 http_504;
            }}
          }}
        }}
        """
    ).strip() + "\n"

    path = WORK_DIR / "nginx-vllm.conf"
    path.write_text(conf, encoding="utf-8")
    return path


def start_nginx(conf_path: Path) -> None:
    if port_open(LB_PORT):
        print(f"[skip] Nginx 端口 {LB_PORT} 已在监听")
        return

    nginx_bin = shutil.which(NGINX_BIN)
    if not nginx_bin:
        print(
            "未找到 nginx，跳过负载均衡。请先安装：\n"
            "  apt-get update && apt-get install -y nginx\n"
            "然后重新运行本脚本，或把应用指向 8001/8002。"
        )
        return

    subprocess.run([nginx_bin, "-t", "-c", str(conf_path)], check=True)
    subprocess.Popen(
        [nginx_bin, "-c", str(conf_path), "-g", "daemon off;"],
        start_new_session=True,
    )
    print(f"[ready] Nginx 负载均衡 127.0.0.1:{LB_PORT}")


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    start_replicas()
    wait_health(HEALTH_TIMEOUT)
    start_nginx(write_nginx_conf())

    print("\n应用侧环境变量：")
    print("  LLM_BACKEND=vllm")
    print(f"  VLLM_BASE_URL=http://127.0.0.1:{LB_PORT}/v1")
    print(f"  VLLM_API_KEY={API_KEY}")
    print(f"  VLLM_MODEL={SERVED_NAME}")
    print(f"  VLLM_LIGHT_MODEL={SERVED_NAME}")
    print("  MAX_CONCURRENT_REQUESTS=8")


if __name__ == "__main__":
    main()
