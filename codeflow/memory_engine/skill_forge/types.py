"""`SkillForgeRouter` Data Types：:class:`RouterHit` + :class:`SkillSource` Protocol。

Two Design Points：

- :class:`RouterHit` **Self-contained**。Legacy ``ScoredSkill`` 只携带 Name + Score，Consumer 必须再从
  `SkillRegistry` Fetch Body；Router Hit 自带 Rendered ``content``，:class:`ContextBuilder` 可直接写入
  Prompt，无需 Second Round-trip。
- :class:`SkillSource` 是 Host-internal，不是 Plugin Contribution Point。``@runtime_checkable`` 让 Tests
  无需继承即可 Assert Duck-typed Conformance，代价是 Surface Matching Object 都会通过；Registration Set
  Closed，因此可接受。

Legacy ``ScoredSkill`` 定义于 :mod:`codeflow.memory_engine.skill_local.types`，历史文档也称
``skill/types.py``；它仍供 :class:`LocalPool` 与旧 ``SkillService`` Path 使用。原 Cleanup PR 计划在
移除 SkillService 后合并两者；当前 ``types.py`` 仍明确保留这道边界。类型通过不证明 Source 排序语义或
Hit 内容正确。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class RouterHit:
    """一个 :class:`SkillSource` 返回的 Ranked Skill。

    它携带 :class:`ContextBuilder` 渲染 System Prompt 所需全部字段，Consumer Side 不再 Lookup Registry。
    ``qualified_id`` 格式是 ``<source>/<native_id>``；`name` 用于跨 Source Dedup，`content` 是 Body，
    `score` 是 Source-local Relevance，`meta` 保存 Provenance/Fusion Details。Frozen 防止字段重新绑定。
    """

    qualified_id: str
    """带 Source Prefix 的 Globally-unique ID，例如 ``"local/git-resolver"`` /
    ``"mirror/git-resolver"``。Source Names 是不含 Embedded Slashes 的 Simple Identifiers，因此 Slash
    Split 无歧义。"""

    name: str
    """Skill Display Name，也是 :func:`rrf_merge_weighted` 的默认 Cross-source Dedup Key。

    SR-2 中相同 ``name`` 的 Two Hits 无论来源都会 Collapse，RRF Score Sum 后保留一个 Representative。
    因此不同技能不应意外共享同名。
    """

    content: str
    """Pre-rendered ``SKILL.md`` Body，Frontmatter 已移除，候选可进入 Prompt ``# Skills`` Block。

    Empty String 表示 Source 只有 Metadata、没有 Body；Consumer 在 Body Join 时跳过，但 Name 仍可出现在
    Summary。字段存在不表示本轮一定注入，仍需 Resolver/Gate。
    """

    score: float
    """Source-internal Relevance；各 Source 有自己的 Scale，例如 BM25 Raw / Cosine Sim / EverMem Score。

    RRF 不直接比较 Cross-source Absolute Values，因此它们跨来源无 Meaning。该值只用于同 ``name`` Hit
    Collision 时选择 Best Representative，:func:`rrf_merge_weighted` 保留更高 ``score`` 的一项。
    """

    meta: dict[str, Any] = field(default_factory=dict)
    """Source-specific Escape Hatch。

    SR-2 在这里加入 ``rrf_score`` 与 ``contributing_sources`` 供 Telemetry；Sources 可加入 Physical-origin
    Label、Native ID、Confidence、``always`` Flag 等。Consumer 应按 Key Capability 读取，不能假设所有
    Source 都提供同一 Metadata。
    """


@runtime_checkable
class SkillSource(Protocol):
    """Router 可查询的一池 Skills。

    ``weight`` 是 Class/Instance Attribute 而非 Method Param，因为 Weight 属于 Router-wide Policy，不是
    Per-call Input；Tests 与 Config 在 Construction 时设置一次。Protocol 只要求 Stable Name、Weight 与
    Async Search，具体索引和生命周期由 Source Own。
    """

    name: str
    """用于 :attr:`RouterHit.qualified_id` 的 Stable Source Identifier，不应含 Slash。"""

    weight: float
    """RRF Source Weight。Higher 表示同一 Skill 从多个 Sources Surface 时，本 Source 贡献更多 Rank Mass。"""

    async def search(
        self,
        query: str,
        history: list[dict[str, Any]],
        k: int,
    ) -> list[RouterHit]:
        """返回至多 ``k`` 条 Best-first :class:`RouterHit` Records。

        ``history`` 是 Session-level Message List，Source 可 Ignore，也可给 Smarter Ranker 作为 Context。
        Empty List 是合法响应；Router ``_safe_search`` 还会把 Exception 转成 Empty，使 Single Source
        Failure 不 Poison Whole Assembly。实现不应返回超过 K 的无界结果。
        """
        ...


__all__ = ["RouterHit", "SkillSource"]
