#!/usr/bin/env python3
"""并发检索效果压测。

直接请求已启动的 FastAPI /api/chat，不初始化第二套 RAG 系统。
默认使用 main.py 中的 19 个测试问题 + 第三轮验收文档问题，共 20 个。

用法:
    python benchmark_concurrent_retrieval.py --concurrency 20
    python benchmark_concurrent_retrieval.py --concurrency 20 --base-url http://127.0.0.1:8000
"""

import argparse
import json
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import requests


QUESTIONS = [
    "AI Agent的四个核心组成部分是什么？",
    "什么是Chain-of-Thought (CoT) 提示技术？",
    "大语言模型面临哪些类型的对抗攻击？",
    "AI Agent中的记忆系统分为哪两种类型？",
    "如何通过提示工程来引导LLM的行为？",
    "对抗性攻击如何影响大语言模型的安全性？",
    "AI Agent的任务规划能力包括哪些方面？",
    "什么是提示工程中的上下文提示？",
    "什么是自监督学习？",
    "什么是提示工程指南？",
    "MinMaxScaler的公式是什么？请列出公式并说明",
    "什么是主体资信供应商风险偏离度？",
    "如何提升LLM面对对抗性攻击的鲁棒性？",
    "AI Agent的工具使用能力是如何工作的？",
    "谁和 Ben Bloomberg 一起开发了 Harmoniser？",
    "Behind the Tech 播客都有哪些嘉宾？",
    "MIT 和微软之间有什么联系？",
    "Kevin Scott 和 Jacob Collier 有什么关系？",
    "Kevin Scott 是谁？",
    "BATCH-2026-11-18-V3 的负责人和部署区域是什么？",
]


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def mean_metric(rows: List[Dict[str, Any]], key: str) -> float:
    values = [
        float(row["metrics"][key])
        for row in rows
        if row.get("metrics") and row["metrics"].get(key) is not None
    ]
    return statistics.mean(values) if values else 0.0


def ask_one(base_url: str, index: int, question: str, run_id: str,
            timeout: int, cache_buster: bool) -> Dict[str, Any]:
    session_id = f"bench-{run_id}-{index:02d}"
    message = question
    if cache_buster:
        message = f"{question}（基准批次 {run_id}-{index:02d}）"

    started = time.perf_counter()
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/api/chat",
            json={"message": message, "session_id": session_id},
            timeout=timeout,
        )
        elapsed = time.perf_counter() - started

        try:
            data = response.json()
        except ValueError:
            data = {"raw": response.text[:500]}

        sources = data.get("sources") or []
        metrics = data.get("metrics") or {}
        return {
            "index": index,
            "question": question,
            "status": response.status_code,
            "elapsed": elapsed,
            "cached": sources == ["⚡ 命中缓存"],
            "answer_preview": str(data.get("answer", ""))[:200],
            "sources_count": len(sources),
            "metrics_available": bool(metrics),
            "metrics": metrics,
            "error": None,
        }
    except Exception as exc:
        return {
            "index": index,
            "question": question,
            "status": None,
            "elapsed": time.perf_counter() - started,
            "cached": False,
            "answer_preview": "",
            "sources_count": 0,
            "metrics_available": False,
            "metrics": {},
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="20 并发 RAG 检索压测")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--no-cache-buster", action="store_true",
                        help="不要在问题后追加唯一批次号（可能命中语义缓存）")
    parser.add_argument("--output", default="")
    parser.add_argument("--warmup", type=int, default=1,
                        help="压测前串行预热次数（不纳入统计），用于加载 BGE-M3/CrossEncoder 并预热 prefix cache")
    args = parser.parse_args()

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    cache_buster = not args.no_cache_buster

    print("=" * 72)
    print("并发检索压测")
    print(f"  base_url   : {args.base_url}")
    print(f"  questions  : {len(QUESTIONS)}")
    print(f"  concurrency: {args.concurrency}")
    print(f"  run_id     : {run_id}")
    print("  注意: vLLM 双副本 max-num-seqs 合计 12；")
    print("        并发超过 12 时，其余请求会在 vLLM 内部排队。")
    print("  建议: 启动 server 时设 ENABLE_USER_MEMORY=false，")
    print("        避免后台记忆抽取的 LLM 调用干扰压测时延。")
    print("=" * 72)

    # 预热：server 重启后 embedding/reranker 是懒加载的，首个请求会承担模型加载开销；
    # 若直接并发，所有请求会阻塞在同一把加载锁上，污染延迟基线。
    if args.warmup > 0:
        print(f"[warmup] 串行预热 {args.warmup} 次（不计入统计）...")
        for i in range(args.warmup):
            warm = ask_one(
                args.base_url,
                0,
                "这是一次服务预热请求，请直接回答 OK。",
                f"warmup-{run_id}-{i}",
                args.timeout,
                cache_buster=True,
            )
            state = "ok" if warm.get("status") == 200 else f"failed({warm.get('error')})"
            print(f"[warmup] {i + 1}/{args.warmup} {warm['elapsed']:.1f}s {state}")
        print("[warmup] 完成，开始正式压测")

    started = time.perf_counter()
    results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                ask_one,
                args.base_url,
                index,
                question,
                run_id,
                args.timeout,
                cache_buster,
            )
            for index, question in enumerate(QUESTIONS, 1)
        ]
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            status = row["status"] if row["status"] is not None else "ERR"
            if row.get("metrics_available"):
                m = row["metrics"]
                metric_text = (
                    f"docs={int(m.get('retrieved_docs_count') or 0)} "
                    f"graded={int(m.get('graded_relevant_count') or 0)} "
                    f"P@3={float(m.get('precision_at_3') or 0):.2f} "
                    f"R@3={float(m.get('recall_at_3') or 0):.2f} "
                    f"MAP={float(m.get('map_score') or 0):.2f} "
                    f"route={m.get('route') or '?'} "
                    f"src={m.get('metric_source', '?')}"
                )
            else:
                metric_text = "metrics=N/A"
            print(
                f"[{row['index']:02d}] status={status} "
                f"{row['elapsed']:.2f}s cached={int(row['cached'])} "
                f"{metric_text} {row.get('error') or ''}"
            )

    wall = time.perf_counter() - started
    results.sort(key=lambda item: item["index"])
    ok_rows = [row for row in results if row["status"] == 200]
    metric_rows = [row for row in ok_rows if row.get("metrics_available")]
    latencies = [row["elapsed"] for row in ok_rows]

    # 指标可信度分布：哪些样本真的完成了评分、哪些走了网络搜索/兜底
    source_counts: Dict[str, int] = {}
    for row in metric_rows:
        source = str(row["metrics"].get("metric_source") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

    grader_fallbacks = sum(
        1 for row in metric_rows if row["metrics"].get("grader_fallback")
    )
    # 排序指标只统计“评分链路真实生效”的样本；
    # web_search / 评分兜底的样本不能代表知识库检索质量，单独计数。
    graded_rows = [
        row for row in metric_rows
        if row["metrics"].get("metric_source") == "agent_graded"
    ]
    web_search_used = sum(
        1 for row in metric_rows
        if int(row["metrics"].get("web_search_calls") or 0) > 0
    )
    hallucination_passed = sum(
        1 for row in metric_rows
        if str(row["metrics"].get("hallucination_check") or "").startswith("通过")
    )
    hallucination_failed = sum(
        1 for row in metric_rows
        if str(row["metrics"].get("hallucination_check") or "").startswith("未通过")
    )

    summary = {
        "total": len(results),
        "success": len(ok_rows),
        "failed": len(results) - len(ok_rows),
        "http_429": sum(1 for row in results if row["status"] == 429),
        "cached": sum(1 for row in results if row["cached"]),
        "metrics_available": len(metric_rows),
        "ranked_samples": len(graded_rows),
        "wall_seconds": round(wall, 3),
        "latency_mean": round(statistics.mean(latencies), 3) if latencies else 0.0,
        "latency_p50": round(percentile(latencies, 50), 3),
        "latency_p95": round(percentile(latencies, 95), 3),
        "latency_max": round(max(latencies), 3) if latencies else 0.0,
        "precision_at_1": round(mean_metric(graded_rows, "precision_at_1"), 4),
        "precision_at_3": round(mean_metric(graded_rows, "precision_at_3"), 4),
        "recall_at_3": round(mean_metric(graded_rows, "recall_at_3"), 4),
        "map_score": round(mean_metric(graded_rows, "map_score"), 4),
        "mrr": round(mean_metric(graded_rows, "mrr"), 4),
        "retrieval_latency_mean": round(mean_metric(metric_rows, "latency"), 4),
        "retrieval_calls_mean": round(mean_metric(metric_rows, "retrieval_calls"), 2),
        "retrieved_docs_mean": round(mean_metric(metric_rows, "retrieved_docs_count"), 2),
        "graded_relevant_mean": round(mean_metric(metric_rows, "graded_relevant_count"), 2),
        "tool_calls_mean": round(mean_metric(metric_rows, "tool_call_count"), 2),
        "metric_source_counts": source_counts,
        "grader_fallback": grader_fallbacks,
        "web_search_used": web_search_used,
        "hallucination_passed": hallucination_passed,
        "hallucination_failed": hallucination_failed,
    }

    report = {
        "run_id": run_id,
        "base_url": args.base_url,
        "concurrency": args.concurrency,
        "questions": len(QUESTIONS),
        "summary": summary,
        "results": results,
    }

    output = Path(args.output or f"data/benchmark_concurrent_{args.concurrency}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("汇总")
    print("=" * 72)
    print(f"总数        : {summary['total']}")
    print(f"成功        : {summary['success']}")
    print(f"失败        : {summary['failed']}")
    print(f"HTTP 429    : {summary['http_429']}")
    print(f"缓存命中    : {summary['cached']}")
    print(f"有指标样本  : {summary['metrics_available']}")
    print(f"可评排序样本: {summary['ranked_samples']}（仅 agent_graded）")
    print(f"总墙钟时间  : {summary['wall_seconds']}s")
    print(f"平均延迟    : {summary['latency_mean']}s")
    print(f"P50 / P95   : {summary['latency_p50']}s / {summary['latency_p95']}s")
    print(f"最大延迟    : {summary['latency_max']}s")
    print(f"平均 P@1    : {summary['precision_at_1']}  (仅 agent_graded)")
    print(f"平均 P@3    : {summary['precision_at_3']}  (仅 agent_graded)")
    print(f"平均 R@3    : {summary['recall_at_3']}  (仅 agent_graded)")
    print(f"平均 MAP    : {summary['map_score']}  (仅 agent_graded)")
    print(f"平均 MRR    : {summary['mrr']}  (仅 agent_graded)")
    print(f"检索耗时均值: {summary['retrieval_latency_mean']}s")
    print(f"检索次数均值: {summary['retrieval_calls_mean']}")
    print(f"检索文档均值: {summary['retrieved_docs_mean']}")
    print(f"评分相关均值: {summary['graded_relevant_mean']}")
    print(f"工具调用均值: {summary['tool_calls_mean']}")
    print(f"指标来源    : {summary['metric_source_counts']}")
    print(f"评分兜底    : {summary['grader_fallback']}  | 使用网络搜索: {summary['web_search_used']}")
    print(f"幻觉检查    : 通过 {summary['hallucination_passed']} / 未通过 {summary['hallucination_failed']}")
    print(f"报告文件    : {output.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
