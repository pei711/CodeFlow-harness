"""Latency-critical Context Path 使用的 Local-only Skill Resolution。

Resolver 不调用 Provider。它先从 Router 取得 Local Candidates，再把显式点名或无文件元数据的少量 Hits
放进 ``activated``，把词法相关但未显式要求的文件 Skill 放进 ``references``。这样 Runtime 能在低延迟
路径注入最必要内容，并把其余候选作为可按需读取的引用。

Resolution 只是本轮 Context Planning；Activated 不代表 Skill Workflow 已执行，Reference 也不表示文件
已读取。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeflow.utils.bm25 import tokenize

if TYPE_CHECKING:
    from codeflow.memory_engine.skill_forge.router import SkillForgeRouter
    from codeflow.memory_engine.skill_forge.types import RouterHit


_WORD_RE = re.compile(r"[a-z0-9]{2,}")
_STOP_WORDS = {
    "and",
    "for",
    "from",
    "into",
    "that",
    "the",
    "this",
    "use",
    "using",
    "with",
    "your",
}


@dataclass(frozen=True)
class SkillResolution:
    activated: tuple["RouterHit", ...] = ()
    references: tuple["RouterHit", ...] = ()
    diagnostics: dict[str, Any] | None = None


class LocalSkillResolver:
    """在不发起 Provider Calls 的情况下 Resolve Local Skill Candidates。

    实例持有 `SkillForgeRouter`、Candidate Limit 与 Activation Limit。`resolve` 对空 Query 返回空结果；
    其他 Query 保留 Router Diagnostics，并按 Explicit Name Overlap、Basic Lexical Relevance 与 On-disk
    ``SKILL.md`` Presence 分类。无真实 Skill Path 的 Hit 可直接 Activated，以兼容 DB/Memory-derived
    Content；文件型 Skill 默认更保守地作为 Reference，除非 Query 明确点名。
    """

    def __init__(
        self,
        router: "SkillForgeRouter",
        *,
        candidate_limit: int = 5,
        activation_limit: int = 2,
    ) -> None:
        self._router = router
        self._candidate_limit = max(1, candidate_limit)
        self._activation_limit = max(0, activation_limit)

    async def resolve(
        self,
        query: str,
        history: list[dict[str, Any]],
    ) -> SkillResolution:
        if not query.strip():
            return SkillResolution(diagnostics={})

        diagnostics: dict[str, Any] = {}
        candidates = await self._router.select(
            query=query,
            history=history,
            k=self._candidate_limit,
            diagnostics=diagnostics,
        )
        activated: list[RouterHit] = []
        references: list[RouterHit] = []
        for hit in candidates:
            skill_path = self._skill_path(hit)
            if skill_path is None:
                if len(activated) < self._activation_limit:
                    activated.append(hit)
                continue
            if self._is_explicit(query, hit):
                if len(activated) < self._activation_limit:
                    activated.append(hit)
                else:
                    references.append(hit)
                continue
            if self._is_relevant(query, hit):
                references.append(hit)

        return SkillResolution(
            activated=tuple(activated),
            references=tuple(references),
            diagnostics=diagnostics,
        )

    @staticmethod
    def _is_explicit(query: str, hit: "RouterHit") -> bool:
        description = str(hit.meta.get("description") or "")
        skill_dir = hit.meta.get("skill_dir")
        if not description and not skill_dir:
            return True

        query_lower = query.lower()
        normalized_name = re.sub(r"[-_/]+", " ", hit.name.lower()).strip()
        if len(normalized_name) >= 3 and normalized_name in query_lower:
            return True
        name_words = {
            token for token in _WORD_RE.findall(normalized_name) if len(token) >= 3 and token not in _STOP_WORDS
        }
        query_words = set(_WORD_RE.findall(query_lower))
        return bool(name_words & query_words)

    @staticmethod
    def _is_relevant(query: str, hit: "RouterHit") -> bool:
        query_tokens = set(tokenize(query))
        candidate_tokens = set(
            tokenize(
                " ".join(
                    (
                        hit.name,
                        str(hit.meta.get("description") or ""),
                        hit.content[:2000],
                    )
                )
            )
        )
        shared = query_tokens & candidate_tokens
        return any(len(token) >= 2 and token not in _STOP_WORDS for token in shared) or len(shared) >= 2

    @staticmethod
    def _skill_path(hit: "RouterHit") -> Path | None:
        skill_dir = hit.meta.get("skill_dir")
        if not skill_dir:
            return None
        path = Path(str(skill_dir)) / "SKILL.md"
        return path if path.is_file() else None


__all__ = ["LocalSkillResolver", "SkillResolution"]
