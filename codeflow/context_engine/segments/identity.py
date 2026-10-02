"""构建 Host-owned Segment 1：``# CodeFlow`` identity 与 runtime 说明。

`IdentitySegmentBuilder` 从 Workspace 和可选 State 路径调用 `render.identity_text`，把 Agent
身份、目录与运行时约束放在 System Prompt 最前。它不依赖前缀，因此属于 Phase A；每轮都
返回一个 Segment，不读取 Session History，也不拥有 Memory、Skill 或 User message。
"""

from __future__ import annotations

from pathlib import Path

from codeflow.context_engine.base import AssemblyContext, Segment
from codeflow.context_engine.segments import render


class IdentitySegmentBuilder:
    name = "identity"
    order = 1
    needs_prefix = False

    def __init__(self, workspace: Path, state: Path | None = None) -> None:
        self._workspace = workspace
        self._state = state

    async def build(self, ctx: AssemblyContext) -> Segment | None:
        return Segment(text=render.identity_text(self._workspace, self._state))
