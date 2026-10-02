"""构建 Host-owned Segment 4：始终启用的 ``# Active Skills``。

`ActiveSkillsSegmentBuilder` 从 `LocalSkillCatalog.get_always_skills()` 读取无需检索即可注入的
Skill，并按配置 ``always_max``（缺失时 5）限制正文数量。目录为空或内容加载失败为空时返回
``None``；有内容才建立标题。它与 Segment 5 的 query-conditioned Skill resolution 分开，
避免检索结果决定 always-on Skill 是否存在。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeflow.context_engine.base import AssemblyContext, Segment
from codeflow.tracing import semconv, trace

if TYPE_CHECKING:
    from codeflow.memory_engine.skill_forge import LocalSkillCatalog


class ActiveSkillsSegmentBuilder:
    name = "active_skills"
    order = 4
    needs_prefix = False

    def __init__(self, skill_catalog: "LocalSkillCatalog") -> None:
        self._skills = skill_catalog

    @trace.instrument("skill.inject", kind="skill", detached=True, extract=semconv.skill_inject_active)
    async def build(self, ctx: AssemblyContext) -> Segment | None:
        always_skills = self._skills.get_always_skills()
        if not always_skills:
            return None
        cfg = getattr(self._skills, "_config", None)
        always_max = getattr(cfg, "always_max", 5) or 5
        content = self._skills.load_skills_for_context(always_skills, max_inject=always_max)
        if not content:
            return None
        return Segment(text=f"# Active Skills\n\n{content}")
