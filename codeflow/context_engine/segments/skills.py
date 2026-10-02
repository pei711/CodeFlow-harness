"""构建 Segment 5：只在 Local Skill 范围内完成检索、引用解析与渲染。

`SkillsSegmentBuilder` 用 `SkillForgeRouter` 和 `LocalSkillResolver` 根据当前消息与 Session
History 选择 activated Skill 和 reference。Activated 正文进入 ``# Skills``，Reference 作为
可继续读取的线索；本地命中的相对引用由 `resolve_refs` 展开，其他 Source 不做本地路径解释。

Router 缺失时返回空 Segment 与完整零值 metadata，而不是让证据字段消失。最终 metadata
分别记录 injected/referenced ids、各 Source 命中数与 Source failure，供 Turn Outcome 核对；
检索失败边界不改变 Segment 4 always-on Skill。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import TYPE_CHECKING

from codeflow.context_engine.base import AssemblyContext, Segment
from codeflow.context_engine.segments import render
from codeflow.memory_engine.skill_forge.refs import resolve_refs
from codeflow.tracing import semconv, trace

if TYPE_CHECKING:
    from codeflow.memory_engine.skill_forge import SkillForgeRouter
    from codeflow.memory_engine.skill_forge.types import RouterHit


class SkillsSegmentBuilder:
    name = "skills"
    order = 5
    needs_prefix = False

    def __init__(
        self,
        router: "SkillForgeRouter | None",
        *,
        skill_top_k: int = 5,
        activation_max: int | None = None,
    ) -> None:
        self._router = router
        if router is None:
            self._resolver = None
        else:
            from codeflow.memory_engine.skill_forge import LocalSkillResolver

            self._resolver = LocalSkillResolver(
                router,
                candidate_limit=skill_top_k,
                activation_limit=(skill_top_k if activation_max is None else activation_max),
            )

    @trace.instrument("skill.inject", kind="skill", detached=True, extract=semconv.skill_inject_skills)
    async def build(self, ctx: AssemblyContext) -> Segment | None:
        if self._resolver is None:
            return Segment(
                text="",
                meta={
                    "injected_skill_ids": [],
                    "referenced_skill_ids": [],
                    "skill_hits_by_source": {},
                },
            )

        query = ctx.current_message or ""
        resolution = await self._resolver.resolve(query, ctx.session_messages)
        activated = self._resolve_local_refs(list(resolution.activated))
        references = list(resolution.references)
        diagnostics = resolution.diagnostics or {}
        body_parts = [
            part
            for part in (
                render.render_router_skills(activated),
                render.render_skill_references(references),
            )
            if part
        ]
        all_hits = [*activated, *references]
        meta = {
            "injected_skill_ids": [h.qualified_id for h in activated if h.qualified_id],
            "referenced_skill_ids": [h.qualified_id for h in references if h.qualified_id],
            "skill_hits_by_source": dict(Counter((h.meta.get("source") or "?") for h in all_hits)),
            "skill_source_failures": diagnostics.get("failed_sources", []),
            "skill_source_failure_types": diagnostics.get("failure_types", {}),
        }
        body = "\n\n".join(body_parts)
        text = f"# Skills\n\n{body}" if body else ""
        return Segment(text=text, meta=meta)

    @staticmethod
    def _resolve_local_refs(gated: list["RouterHit"]) -> list["RouterHit"]:
        out: list["RouterHit"] = []
        for hit in gated:
            if hit.meta.get("source") == "local":
                resolved, _ = resolve_refs(hit.content, hit.meta.get("skill_dir"))
                hit = replace(hit, content=resolved)
            out.append(hit)
        return out
