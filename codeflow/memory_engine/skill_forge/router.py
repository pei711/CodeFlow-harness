"""`SkillForgeRouter` 并发 Fan-out 所有 Sources，再用 :func:`rrf_merge_weighted` 融合排名。

Router 而非各 Source 统一执行 Two Policies：

- **Per-source Over-fetch**：:meth:`select(k)` 向每个 Source 请求 ``k * over_fetch_factor`` Hits，RRF 再
  收窄到 Overall K。某 Source 的 #3 可能是优秀 Cross-source Merge Candidate，即使单独看进不了 Top-3；
  Default Factor=2。
- **Single-source Failure Isolation**：Source Raise 会在 :meth:`_safe_search` 内转成该轮 Empty List，并
  记录 Failure Type；其他 Sources 继续 Feed RRF，避免一个 Transient 变成 Whole-pipeline Failure。

Source List 在 Construction 时固定引用。Active Runtime 连接 Single Local Source；Deterministic
Evaluators 可使用 Multiple Sources。Router Ranking 不包含 LLM Gate 或最终 Skill Hydration。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from codeflow.memory_engine.skill_forge.fusion import rrf_merge_weighted
from codeflow.memory_engine.skill_forge.types import RouterHit, SkillSource

logger = logging.getLogger(__name__)


class SkillForgeRouter:
    """把 N 个 :class:`SkillSource` Outputs 组合为一个 Top-K Ranking。

    实例保存 Source Order、Over-fetch Factor 与 Dedup Field。Source List 按引用捕获，Host 启动后约定不再
    修改；`select` 并发调用并填充 Diagnostics，最后用 Weighted RRF 合并。Router 不验证 Skill
    Requirements，也不执行 Provider Gate。
    """

    def __init__(
        self,
        sources: list[SkillSource],
        *,
        over_fetch_factor: int = 2,
        dedup_by: str = "name",
    ) -> None:
        # 列表按引用捕获；如需禁止修改，调用方应传入已冻结的元组。此处刻意不代为冻结：
        # 宿主在启动时连接数据源，之后不再修改，冗长的不可变包装没有收益。
        self._sources = sources
        self._over_fetch_factor = max(1, over_fetch_factor)
        self._dedup_by = dedup_by

    async def select(
        self,
        query: str,
        history: list[dict[str, Any]],
        k: int = 5,
        *,
        diagnostics: dict[str, Any] | None = None,
    ) -> list[RouterHit]:
        """并发 Fan Out 每个 Source，并 Fuse 成 Top-K。

        每个 Source 获得同一 `query`、`history` 与 Over-fetched K。`diagnostics` 提供时写入
        ``failed_sources`` / ``failure_types``；Source Failure 仍可返回其他结果。`k` 传给最终 RRF 限制输出，
        空 Sources 或全失败得到空列表，而不是异常。
        """
        per_source_k = k * self._over_fetch_factor
        per_source = await asyncio.gather(*[self._safe_search(s, query, history, per_source_k) for s in self._sources])
        if diagnostics is not None:
            diagnostics["failed_sources"] = [
                source.name for source, (_, error_type) in zip(self._sources, per_source) if error_type is not None
            ]
            diagnostics["failure_types"] = {
                source.name: error_type
                for source, (_, error_type) in zip(self._sources, per_source)
                if error_type is not None
            }
        return rrf_merge_weighted(
            [(s.name, s.weight, hits) for s, (hits, _) in zip(self._sources, per_source)],
            k=k,
            dedup_by=self._dedup_by,
        )

    async def _safe_search(
        self,
        source: SkillSource,
        query: str,
        history: list[dict[str, Any]],
        k: int,
    ) -> tuple[list[RouterHit], str | None]:
        try:
            return await source.search(query, history, k), None
        except Exception as e:
            # ``exception()`` 会写入堆栈追踪；使用警告级别，让短暂抖动不会淹没 ``error`` 日志，
            # 同时仍能进入常规聚合。
            logger.warning(
                "skill source %r failed; treating as empty: %s",
                source.name,
                e,
            )
            return [], type(e).__name__


__all__ = ["SkillForgeRouter"]
