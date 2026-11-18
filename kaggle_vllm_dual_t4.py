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
  VLLM_MAX_NUM_BATCHED_TOKENS  单次批处理 token 上限，默认 2048（T4 上降低激活峰值）
  VLLM_CHUNKED_PREFILL   默认 true；max_num_batched_tokens < max_model_len 时必须为 true，
                         否则 vLLM 会直接拒绝启动
  VLLM_QUANTIZATION      默认 gptq；AWQ 模型设为 awq；空字符串表示不传
  VLLM_ENABLE_TOOL_CALLING  默认 true，Agent 模式需要
  VLLM_TOOL_PARSER       默认 hermes
  VLLM_GPU0_UTIL         默认 0.65（GPU0 还要跑检索模型）
  VLLM_GPU1_UTIL         默认 0.85
  VLLM_GPU0_MAX_SEQS     默认 4
  VLLM_GPU1_MAX_SEQS     默认 8
  VLLM_LB_PORT           Nginx 负载均衡端口，默认 8200
  VLLM_HEALTH_TIMEOUT    副本就绪超时（秒），默认 900
  HF_HOME                HuggingFace 缓存目录，必须可写；若 /kaggle/temp 不可写会直接失败，
                         推荐 HF_HOME=/kaggle/working/hf-cache
"""

import os
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import requests


WORK_DIR = Path(os.environ.get("VLLM_WORK_DIR", "/kaggle/working"))
MODEL_PATH = os.environ.get("VLLM_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4")
SERVED_NAME = os.environ.get("VLLM_SERVED_NAME", "qwen2.5-7b")
MAX_MODEL_LEN = os.environ.get("VLLM_MAX_MODEL_LEN", "8192")
MAX_NUM_BATCHED_TOKENS = os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS", "2048")
QUANTIZATION = os.environ.get("VLLM_QUANTIZATION", "gptq")
API_KEY = os.environ.get("VLLM_API_KEY", "dummy")
TOOL_PARSER = os.environ.get("VLLM_TOOL_PARSER", "hermes")
ENABLE_TOOL_CALLING = os.environ.get("VLLM_ENABLE_TOOL_CALLING", "true").lower() == "true"
ENFORCE_EAGER = os.environ.get("VLLM_ENFORCE_EAGER", "true").lower() == "true"
PREFIX_CACHING = os.environ.get("VLLM_PREFIX_CACHING", "true").lower() == "true"
CHUNKED_PREFILL = os.environ.get("VLLM_CHUNKED_PREFILL", "true").lower() == "true"
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
    if MAX_NUM_BATCHED_TOKENS:
        cmd.extend(["--max-num-batched-tokens", MAX_NUM_BATCHED_TOKENS])
    if CHUNKED_PREFILL:
        cmd.append("--enable-chunked-prefill")
    if QUANTIZATION:
        cmd.extend(["--quantization", QUANTIZATION])
    if ENFORCE_EAGER:
        cmd.append("--enforce-eager")
    if PREFIX_CACHING:
        cmd.append("--enable-prefix-caching")
    if ENABLE_TOOL_CALLING and TOOL_PARSER:
        cmd.extend(["--enable-auto-tool-choice", "--tool-call-parser", TOOL_PARSER])
    return cmd


def replica_log_path(replica: dict) -> Path:
    return WORK_DIR / f"vllm-gpu{replica['gpu']}.log"


def tail_log(path: Path, lines: int = 40) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.readlines()
    except OSError as exc:
        return f"<无法读取 {path}: {exc}>"
    return "".join(content[-lines:]).strip() or "<日志为空>"


def fail_with_log(port: int, log_path: Path, reason: str) -> None:
    """子进程异常退出时，把日志尾部直接打到屏幕上，避免盲等 15 分钟。"""
    print(f"[fail] {reason}")
    print("-" * 70)
    print(tail_log(log_path, 80))
    print("-" * 70)
    sys.stdout.flush()
    raise RuntimeError(f"{reason}；完整日志: {log_path}")


def health_ok(port: int) -> bool:
    try:
        response = requests.get(
            f"http://127.0.0.1:{port}/health",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=2,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def probe_engine(port: int) -> bool:
    """/health 在 engine 挂掉后仍可能返回 200，用一次真实推理确认 engine 可用。"""
    try:
        response = requests.post(
            f"http://127.0.0.1:{port}/v1/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": SERVED_NAME,
                "prompt": "ping",
                "max_tokens": 1,
                "temperature": 0,
            },
            timeout=30,
        )
        if response.status_code != 200:
            print(
                f"[probe] vLLM {port} 推理探测失败: "
                f"HTTP {response.status_code} {response.text[:200]}"
            )
            return False
        return True
    except requests.RequestException as exc:
        print(f"[probe] vLLM {port} 推理探测异常: {exc}")
        return False


def stop_replicas(procs: dict[int, subprocess.Popen]) -> None:
    for port, proc in procs.items():
        if proc.poll() is not None:
            continue
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            print(f"[stop] 已终止 vLLM {port} 进程组")
        except ProcessLookupError:
            continue
        except Exception as exc:
            print(f"[stop] 终止 vLLM {port} 失败: {exc}")
            continue
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                print(f"[stop] vLLM {port} 未响应 SIGTERM，已强制终止")
            except ProcessLookupError:
                pass


def check_hf_storage() -> None:
    """HF 缓存目录不可写时 vLLM 会在加载模型前就退出，这里提前给出明确错误。"""
    candidates = {
        os.environ.get("HF_HOME"),
        os.environ.get("TRANSFORMERS_CACHE"),
        os.environ.get("HF_HUB_CACHE"),
    }
    seen: set[str] = set()
    for raw in filter(None, candidates):
        path = Path(raw).expanduser()
        if str(path) in seen:
            continue
        seen.add(str(path))
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except Exception as exc:
            raise RuntimeError(
                f"HF 缓存目录不可写: {path}（{exc}）。"
                "请改用可写目录，例如 HF_HOME=/kaggle/working/hf-cache"
            ) from exc
        free_gb = shutil.disk_usage(path).free / (1024 ** 3)
        print(f"[hf] 缓存目录 {path} 可写，剩余 {free_gb:.1f} GB")
        if free_gb < 15:
            print("[hf] 警告：剩余空间不足 15GB，模型权重或编译缓存可能写入失败")


def check_batching_config() -> None:
    """vLLM 0.7.3 在未开 chunked prefill 时要求 max_num_batched_tokens >= max_model_len。"""
    global CHUNKED_PREFILL, MAX_NUM_BATCHED_TOKENS
    if not MAX_NUM_BATCHED_TOKENS:
        return
    try:
        batched = int(MAX_NUM_BATCHED_TOKENS)
        model_len = int(MAX_MODEL_LEN)
    except ValueError:
        return
    if batched >= model_len:
        return
    if CHUNKED_PREFILL:
        print(
            f"[config] max_num_batched_tokens={batched} < max_model_len={model_len}，"
            "已启用 --enable-chunked-prefill 分批 prefill"
        )
        return
    MAX_NUM_BATCHED_TOKENS = str(model_len)
    print(
        f"[config] vLLM 不接受 max_num_batched_tokens({batched}) < max_model_len({model_len})；"
        f"VLLM_CHUNKED_PREFILL=false，已自动提升到 {model_len}"
    )


def check_model_source() -> None:
    local = Path(MODEL_PATH).expanduser()
    if local.exists():
        print(f"[model] 本地模型路径: {local}")
    else:
        print(f"[model] HuggingFace 仓库: {MODEL_PATH}")
        print("[model] 首次运行需要下载权重，5-15 分钟属正常；若日志停在下载阶段请检查网络和磁盘")


def start_replicas() -> dict[int, subprocess.Popen]:
    if not (Path(VLLM_BIN).exists() or shutil.which(VLLM_BIN)):
        setup_script = Path(__file__).resolve().parent / "setup_vllm_env.sh"
        raise RuntimeError(
            "未找到 vllm 可执行文件，请先安装：\n"
            f"  bash {setup_script}\n"
            "或手动安装：\n"
            "  python -m venv /kaggle/working/vllm-env --system-site-packages\n"
            "  /kaggle/working/vllm-env/bin/python -m pip install -r requirements-vllm.txt"
        )

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[vllm] 使用可执行文件: {VLLM_BIN}")
    procs: dict[int, subprocess.Popen] = {}
    for replica in REPLICAS:
        port = replica["port"]
        log_path = replica_log_path(replica)
        if port_open(port):
            state = "已就绪" if health_ok(port) else "健康检查不通，可能是残留进程"
            print(f"[skip] vLLM 端口 {port} 已在监听（{state}）")
            continue

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(replica["gpu"])
        env.setdefault("VLLM_USE_V1", "0")

        log_file = open(log_path, "w")
        cmd = build_serve_command(replica)
        print(f"[start] GPU{replica['gpu']} -> 127.0.0.1:{port}, log={log_path}")
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        procs[port] = proc

        # 缺依赖、缓存不可写、端口冲突等问题通常 2-3 秒内就会退出，先快速探一次
        time.sleep(3)
        rc = proc.poll()
        if rc is not None:
            stop_replicas(procs)
            fail_with_log(port, log_path, f"vLLM {port} 启动后立即退出（exit code={rc}）")
    return procs


def wait_health(
    timeout_seconds: int = 900,
    procs: dict[int, subprocess.Popen] | None = None,
) -> None:
    procs = procs or {}
    for replica in REPLICAS:
        port = replica["port"]
        proc = procs.get(port)
        log_path = replica_log_path(replica)
        deadline = time.time() + timeout_seconds
        wait_started = time.time()
        last_report = 0.0
        while time.time() < deadline:
            if proc is not None and proc.poll() is not None:
                fail_with_log(
                    port,
                    log_path,
                    f"vLLM {port} 进程已退出（exit code={proc.returncode}）",
                )
            if health_ok(port):
                if probe_engine(port):
                    print(f"[ready] vLLM {port}")
                    break
                fail_with_log(
                    port,
                    log_path,
                    f"vLLM {port} /health 返回 200，但推理探测失败（engine 可能已退出）",
                )
            now = time.time()
            if now - last_report >= 30:
                elapsed = int(now - wait_started)
                last_line = tail_log(log_path, 1).replace("\r", " ").replace("\n", " ")[:180]
                suffix = f" | 日志最后一行: {last_line}" if last_line else ""
                print(f"[wait] vLLM {port} 尚未就绪（{elapsed}s）{suffix}")
                last_report = now
            time.sleep(5)
        else:
            hint = ""
            if proc is None:
                hint = f"；端口 {port} 在启动前已被占用，请检查是否残留旧进程：ss -ltnp | grep {port}"
            fail_with_log(
                port,
                log_path,
                f"vLLM {port} 在 {timeout_seconds}s 内未就绪{hint}",
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
    check_hf_storage()
    check_batching_config()
    check_model_source()
    procs = start_replicas()
    try:
        wait_health(HEALTH_TIMEOUT, procs)
    except BaseException:
        stop_replicas(procs)
        raise
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
