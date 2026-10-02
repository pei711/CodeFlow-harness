"""提供对已注册 Local Skill 正文与解析后资源路径的只读访问。

`SkillReadTool` 接受精确 Skill name，通过共享 `LocalSkillCatalog` 加载最多一个 Skill 的完整
Context 形状。它用于主模型先看到紧凑 reference 后按需读取正文，不重新扫描或自行解析未知
目录；找不到名称返回 Error，Catalog 的 trust 和路径规则保持唯一事实源。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from codeflow.agent.tools.base import Tool

if TYPE_CHECKING:
    from codeflow.memory_engine.skill_forge import LocalSkillCatalog


class SkillReadTool(Tool):
    def __init__(self, catalog: "LocalSkillCatalog") -> None:
        self._catalog = catalog

    @property
    def name(self) -> str:
        return "skill_read"

    @property
    def description(self) -> str:
        return "Read a registered Local Skill by name, including its resolved resource paths."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The exact Local Skill name",
                }
            },
            "required": ["name"],
        }

    async def execute(self, name: str, **_kwargs: Any) -> str:
        content = self._catalog.load_skills_for_context([name], max_inject=1)
        if not content:
            return f"Error: Skill not found: {name}"
        return content


__all__ = ["SkillReadTool"]
