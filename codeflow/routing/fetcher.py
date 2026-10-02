"""PinchBench API Client，对应 EcoClaw 的 `fetcher.ts`。

这个模块从远端分别取得 Leaderboard 与每个模型最新一次 Submission Details，再把两类响应合并成
以 Model ID 为 Key 的 `BenchmarkData` Dict。Leaderboard 提供成本、速度等汇总字段，Submission
提供逐任务得分；二者共同构成路由器选择模型所需的能力与代价证据。

网络请求或单个 Submission 失败时，调用链会跳过受影响的模型而继续处理其他模型。因此返回数据
可能是部分成功的 Snapshot，不代表 PinchBench 上所有模型都被完整同步。
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from loguru import logger

from codeflow.routing.types import ModelBenchmark, ModelTaskScore

API_BASE = "https://api.pinchbench.com/api"
FETCH_CONCURRENCY = 5
FETCH_TIMEOUT_S = 30.0

# 类型别名
BenchmarkData = dict[str, ModelBenchmark]


async def fetch_leaderboard(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    resp = await client.get(f"{API_BASE}/leaderboard", timeout=FETCH_TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json()
    return data.get("leaderboard", [])


async def fetch_submission(client: httpx.AsyncClient, submission_id: str) -> dict[str, Any]:
    url = f"{API_BASE}/submissions/{submission_id}"
    resp = await client.get(url, timeout=FETCH_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()


async def fetch_latest_submission_id(client: httpx.AsyncClient, model: str) -> str | None:
    url = f"{API_BASE}/submissions"
    resp = await client.get(url, params={"model": model, "limit": 1}, timeout=FETCH_TIMEOUT_S)
    if not resp.is_success:
        return None
    data = resp.json()
    submissions = data.get("submissions", [])
    return submissions[0]["id"] if submissions else None


async def _fetch_one(
    client: httpx.AsyncClient,
    entry: dict[str, Any],
    sem: asyncio.Semaphore,
) -> dict[str, Any] | None:
    async with sem:
        try:
            latest_id = await fetch_latest_submission_id(client, entry["model"])
            if not latest_id:
                return None
            detail = await fetch_submission(client, latest_id)
            return {"entry": entry, "detail": detail, "submission_id": latest_id}
        except Exception as e:
            logger.warning("Failed to fetch submission for {}: {}", entry.get("model"), e)
            return None


async def build_benchmark_data() -> BenchmarkData:
    """从 PinchBench API 获取 Leaderboard + Submission Details，并构建 Benchmark Data。

    方法先读取 Leaderboard，再以 `FETCH_CONCURRENCY` 限制并发，为每个条目查询最新 Submission ID
    及其详情。随后把逐任务原始得分换算成百分比，计算模型 Overall Score，并返回从 Model ID 映射到
    `ModelBenchmark` 的 Dict。Cost 缺失或不大于零的条目会被排除，因为它们无法进入成本感知路由。

    同一 Model ID 仅大小写不同时视为重复，并保留 Cost 更高的条目。这沿用既有数据兼容策略，不表示
    高成本模型在质量上更好。单项 Fetch 异常会记录 Warning 并跳过，所以成功返回只证明已构建可用
    子集，不证明远端数据完整。
    """
    async with httpx.AsyncClient() as client:
        leaderboard = await fetch_leaderboard(client)
        logger.info("PinchBench leaderboard: {} entries", len(leaderboard))

        sem = asyncio.Semaphore(FETCH_CONCURRENCY)
        results = await asyncio.gather(
            *[_fetch_one(client, entry, sem) for entry in leaderboard],
        )

    data: BenchmarkData = {}

    for item in results:
        if item is None:
            continue
        entry: dict[str, Any] = item["entry"]
        detail: dict[str, Any] = item["detail"]
        submission_id: str = item["submission_id"]

        cost = entry.get("average_cost_usd")
        if cost is None or cost <= 0:
            continue

        tasks = detail.get("submission", {}).get("tasks", [])
        task_scores = [
            ModelTaskScore(
                task_id=t["task_id"],
                score=(t["score"] / (t["max_score"] or 1)) * 100,
                max_score=t["max_score"],
            )
            for t in tasks
        ]

        sum_score = sum(t["score"] for t in tasks)
        sum_max = sum(t["max_score"] for t in tasks)
        overall = (sum_score / sum_max * 100) if sum_max > 0 else 0.0

        benchmark = ModelBenchmark(
            model=entry["model"],
            provider=entry.get("provider", ""),
            overall_score=overall,
            speed=entry.get("average_execution_time_seconds"),
            cost=cost,
            task_scores=task_scores,
            submission_id=submission_id,
        )
        data[entry["model"]] = benchmark

    # 去重：多个条目映射到同一模型 ID 时保留成本更高者。
    seen: dict[str, str] = {}  # normalized_id → 原始模型键
    for model_id, benchmark in list(data.items()):
        norm = model_id.lower()
        if norm in seen:
            existing_id = seen[norm]
            if benchmark.cost > data[existing_id].cost:
                del data[existing_id]
                seen[norm] = model_id
            else:
                del data[model_id]
        else:
            seen[norm] = model_id

    logger.info("Built benchmark data: {} models", len(data))
    return data
