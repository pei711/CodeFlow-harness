"""在 Heterogeneous Skill Sources 之间执行 Weighted Reciprocal Rank Fusion。

不同 Source 的 Raw Score 不一定同尺度，RRF 改用各自 Rank 计算 ``weight / (RRF_K + rank)``，再按 Skill
Identity 合并。这样 Local/Remote/Memory-derived Results 可以共同排序，而不假装它们的相关度数字完全可比。
融合结果是 Retrieval Ranking Evidence，不是 Skill 正确性或任务完成证明。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from codeflow.memory_engine.skill_forge.types import RouterHit

RRF_K: int = 60


def rrf_merge_weighted(
    source_results: list[tuple[str, float, list[RouterHit]]],
    k: int,
    dedup_by: str = "name",
) -> list[RouterHit]:
    """把 Per-source Ranked Lists 融合成一个 Top-K Result。

    `source_results` 每项包含 Source Name、Weight 与 Hits；`dedup_by` 默认使用 ``name`` 形成 Identity Key。
    同一 Key 累加 Weighted RRF Score，并保留 Raw Score 最高的 Hit 作为 Representative。输出按融合分数
    Descending，`meta` 追加 ``rrf_score`` 与 ``contributing_sources``。

    `k` 限制最终数量；Unknown `dedup_by` 会由 `getattr` 抛错，调用方必须选择 `RouterHit` 真实字段。
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    best_hit: dict[str, RouterHit] = {}
    contributing: dict[str, list[str]] = defaultdict(list)

    for source_name, weight, hits in source_results:
        for rank, hit in enumerate(hits, start=1):
            key = getattr(hit, dedup_by)
            rrf_scores[key] += weight / (RRF_K + rank)
            contributing[key].append(source_name)
            previous = best_hit.get(key)
            if previous is None or hit.score > previous.score:
                best_hit[key] = hit

    ranked = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    out: list[RouterHit] = []
    for key, score in ranked[:k]:
        representative = best_hit[key]
        out.append(
            replace(
                representative,
                meta={
                    **representative.meta,
                    "rrf_score": score,
                    "contributing_sources": list(contributing[key]),
                },
            )
        )
    return out


__all__ = ["RRF_K", "rrf_merge_weighted"]
