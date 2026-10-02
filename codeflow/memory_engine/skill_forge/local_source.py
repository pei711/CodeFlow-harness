"""`LocalSkillSource` 把 :class:`LocalPool` Search Result 转成 :class:`RouterHit`。

Hot-path 上，:class:`LocalPool` 已返回 Cheap ``ScoredSkill(name, score, source)`` Triple。构造 Router Hit
还需要 ``content``，即 ``SKILL.md`` Body；:class:`SkillRegistry.get` 已把它缓存到 Memory，所以 Per-hit
Lookup 是 O(1) Dict Access，不会重新读 Disk。

若 File Watcher 在 BM25 Scoring 与 Meta Lookup 之间删除 Skill，Name 不再 Resolve，该 Hit Silently Skip。
Router Contract 是 ``at most k hits`` 而非 ``exactly k``，Drop 比发出 Empty Content 更安全。Search Hit
只表示关键词相关，不代表 LLM Gate 已选择。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from codeflow.memory_engine.skill_forge.types import RouterHit

if TYPE_CHECKING:
    from codeflow.memory_engine.skill_local.local_pool import LocalPool
    from codeflow.memory_engine.skill_local.registry import SkillRegistry


class LocalSkillSource:
    """BM25 Local Pool 的 Host-internal `SkillSource` Adapter。

    实例持有 Pool、Registry 与 Nonnegative Finite `min_score`。`search` 忽略 History，只按 Query 调用
    Local BM25，再补齐 Body、Physical Source、Always Flag、Skill Directory 与 Description。Qualified ID
    固定为 ``local/<name>``，供 Fusion/Gate/Feedback 使用。

    初始化时非法 `min_score` 抛出 `ValueError`。搜索返回数量可少于 K，包括低分过滤与 Watcher Race。
    """

    name: str = "local"
    weight: float = 1.0

    def __init__(
        self,
        pool: "LocalPool",
        registry: "SkillRegistry",
        *,
        min_score: float = 0.0,
    ) -> None:
        self._pool = pool
        self._registry = registry
        try:
            self._min_score = float(min_score)
        except (TypeError, ValueError):
            raise ValueError(
                "LocalSkillSource: min_score must be nonnegative and finite",
            ) from None
        if isinstance(min_score, bool) or self._min_score < 0 or not math.isfinite(self._min_score):
            raise ValueError(
                "LocalSkillSource: min_score must be nonnegative and finite",
            )

    async def search(
        self,
        query: str,
        history: list[dict[str, Any]],
        k: int,
    ) -> list[RouterHit]:
        # ``history`` 尚未使用，本地 BM25 不依赖先前对话。未来更智能的本地排序器
        # 可以引入它；当前签名与协议一致，以保留扩展边界。
        del history

        hits = self._pool.search(query, top_k=k)
        out: list[RouterHit] = []
        for h in hits:
            if h.score < self._min_score:
                continue
            meta = self._registry.get(h.name, source=h.source)
            if meta is None:
                # 文件监视器竞态：Skill 在 BM25 快照与元数据查找之间消失。
                # 直接跳过，不输出只填充了部分字段的命中结果。
                continue
            # ``skill_dir`` 让 SkillsSegmentBuilder 的门控后填充步骤能解析 {baseDir}
            # 和 Markdown 链接引用，无需再查一次注册表。
            path_obj = getattr(meta, "path", None)
            path_str = str(path_obj) if path_obj is not None else ""
            skill_dir: str | None = None
            if path_obj is not None and not path_str.startswith("sqlite:"):
                skill_dir = str(path_obj.parent)
            out.append(
                RouterHit(
                    qualified_id=f"local/{h.name}",
                    name=h.name,
                    content=meta.content,
                    score=h.score,
                    meta={
                        "source": "local",
                        # Local 内的物理来源（``workspace``、``builtin``、``external`` 或 ``mirror/*``）
                        # 偶尔会用于遥测，因此保留在 meta 中。
                        "physical_source": h.source,
                        "always": meta.always,
                        "skill_dir": skill_dir,
                        "description": meta.description,
                    },
                ),
            )
        return out


__all__ = ["LocalSkillSource"]
