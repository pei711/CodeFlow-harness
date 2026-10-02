"""Benchmark 数据缓存，整体行为与 EcoClaw 的 `cache.ts` 保持对应。

模型路由依赖 Benchmark 数据比较候选模型，但联网 API 并不总是可用。这个模块因此实现 4-tier
fallback，并按照可信度和响应速度依次选择数据源：

1. In-memory：进程内已有数据时立即返回；
2. Disk cache：磁盘缓存未超过 6h 时先返回，并在后台刷新；
3. Stale disk cache：缓存已经过期、但 API 刷新失败时，继续使用旧数据保障路由可用；
4. Hardcoded `snapshot.json`：首次运行既无缓存、API 也不可用时使用随包快照。

因此，缓存命中只能说明路由器取得了一份可用的 Benchmark Snapshot，不能证明它一定是服务端的
最新版本。调用方若关心新鲜度，需要结合这里的 TTL 与刷新日志判断。
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from loguru import logger

from codeflow.product import get_product_home
from codeflow.routing.fetcher import BenchmarkData, build_benchmark_data
from codeflow.routing.types import ModelBenchmark, ModelTaskScore

CACHE_VERSION = 1
CACHE_TTL_S = 6 * 60 * 60  # 6 小时
_SNAPSHOT_PATH = Path(__file__).parent / "snapshot.json"


def _load_snapshot() -> BenchmarkData:
    """从随包的 `snapshot.json` 加载 Hardcoded Fallback Data。

    函数把 JSON 中的每个有效模型转换为 `ModelBenchmark`，并把各任务分数转换为
    `ModelTaskScore`。缺少或不大于零的 Cost 会被跳过，因为没有有效成本的模型无法参与成本约束下的
    选择。读取、解析或字段转换失败时记录错误并返回已经成功解析的部分；返回空字典表示没有可用
    快照，而不是 API 查询结果为空。
    """
    data: BenchmarkData = {}
    try:
        raw = json.loads(_SNAPSHOT_PATH.read_text(encoding="utf-8"))
        for entry in raw.get("models", []):
            cost = entry.get("cost")
            if not cost or cost <= 0:
                continue
            task_scores = [
                ModelTaskScore(
                    task_id=t["taskId"],
                    score=t["score"],
                    max_score=t["maxScore"],
                )
                for t in entry.get("taskScores", [])
            ]
            data[entry["model"]] = ModelBenchmark(
                model=entry["model"],
                provider=entry.get("provider", ""),
                overall_score=entry.get("overallScore", 0.0),
                speed=entry.get("speed"),
                cost=cost,
                task_scores=task_scores,
                submission_id=entry.get("submissionId", ""),
            )
    except Exception as e:
        logger.error("Failed to load snapshot.json: {}", e)
    return data


def _serialize(data: BenchmarkData) -> dict:
    return {
        "version": CACHE_VERSION,
        "fetched_at": time.time(),
        "models": [
            {
                "model": b.model,
                "provider": b.provider,
                "overall_score": b.overall_score,
                "speed": b.speed,
                "cost": b.cost,
                "submission_id": b.submission_id,
                "task_scores": [
                    {"task_id": t.task_id, "score": t.score, "max_score": t.max_score} for t in b.task_scores
                ],
            }
            for b in data.values()
        ],
    }


def _deserialize(raw: dict) -> tuple[BenchmarkData, float] | None:
    if raw.get("version") != CACHE_VERSION or not isinstance(raw.get("models"), list):
        return None
    data: BenchmarkData = {}
    for entry in raw["models"]:
        cost = entry.get("cost")
        if not cost or cost <= 0:
            continue
        task_scores = [
            ModelTaskScore(
                task_id=t["task_id"],
                score=t["score"],
                max_score=t["max_score"],
            )
            for t in entry.get("task_scores", [])
        ]
        data[entry["model"]] = ModelBenchmark(
            model=entry["model"],
            provider=entry.get("provider", ""),
            overall_score=entry.get("overall_score", 0.0),
            speed=entry.get("speed"),
            cost=cost,
            task_scores=task_scores,
            submission_id=entry.get("submission_id", ""),
        )
    return data, raw.get("fetched_at", 0.0)


class BenchmarkCache:
    """带 Background Refresh 的 Thread-safe Benchmark Data Cache。

    一个实例同时持有进程内 Snapshot、磁盘缓存路径与至多一个后台刷新 Task。`load` 是主要生命周期
    入口：第一次调用建立数据，后续调用复用 `_data`；磁盘新鲜数据会立即服务请求，再异步刷新，避免
    让模型选择等待网络。这里的 Thread-safe 指不会重复安排并发后台刷新，不代表多个 OS Process 会对
    同一缓存文件做事务协调；写盘失败仅降级缓存持久性，不阻断本次路由。
    """

    def __init__(self, cache_path: Path | None = None):
        self._cache_path = cache_path or get_product_home() / "routing" / "benchmark-cache.json"
        self._data: BenchmarkData | None = None
        self._refresh_task: asyncio.Task | None = None

    async def load(self) -> BenchmarkData:
        """返回 Benchmark Data，并按需要 Fetch 或 Refresh。

        调用顺序严格遵循模块说明中的四级回退：先读 In-memory，再读 6 小时内的 Disk Cache，过期缓存
        则同步尝试 API，最后才读 `snapshot.json`。新鲜磁盘缓存返回前会安排一次后台刷新；API 异常不会
        直接传给调用方，而是尽可能退回旧缓存或快照。返回值始终是以模型名为 Key 的 `BenchmarkData`，
        但可能为空，也可能不是最新数据，调用方不应把“成功返回”误解为“在线刷新成功”。
        """
        # 1. 命中内存缓存。
        if self._data is not None:
            return self._data

        # 2. 尝试磁盘缓存。
        cached = self._load_from_disk()
        if cached:
            data, fetched_at = cached
            age = time.time() - fetched_at
            if age < CACHE_TTL_S:
                self._data = data
                self._schedule_background_refresh()
                return self._data

            # 缓存已过期，优先尝试 API。
            try:
                return await self._do_refresh()
            except Exception:
                logger.warning("API refresh failed, using stale cache")
                self._data = data
                return self._data

        # 3. 未命中缓存，尝试 API。
        try:
            return await self._do_refresh()
        except Exception:
            # 4. 回退到快照。
            logger.warning("API unavailable, falling back to snapshot.json")
            self._data = _load_snapshot()
            return self._data

    def _load_from_disk(self) -> tuple[BenchmarkData, float] | None:
        try:
            if not self._cache_path.exists():
                return None
            raw = json.loads(self._cache_path.read_text(encoding="utf-8"))
            return _deserialize(raw)
        except Exception:
            return None

    def _save_to_disk(self, data: BenchmarkData) -> None:
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(
                json.dumps(_serialize(data), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("Failed to write benchmark cache: {}", e)

    async def _do_refresh(self) -> BenchmarkData:
        data = await build_benchmark_data()
        self._data = data
        self._save_to_disk(data)
        logger.info("Benchmark cache refreshed: {} models", len(data))
        return data

    def _schedule_background_refresh(self) -> None:
        if self._refresh_task and not self._refresh_task.done():
            return

        async def _bg():
            await asyncio.sleep(0.1)
            try:
                await self._do_refresh()
            except Exception:
                pass

        self._refresh_task = asyncio.create_task(_bg())

    def get_fallback(self) -> BenchmarkData:
        return _load_snapshot()
