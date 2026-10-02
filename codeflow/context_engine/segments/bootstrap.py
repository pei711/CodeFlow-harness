"""构建 Host-owned Segment 2：soul、agent、TOOLS 等 bootstrap files。

`BootstrapSegmentBuilder` 在 Phase A 调用 `render.load_bootstrap_files`，从 Workspace 读取配置
的启动文件并合并为 System Prompt 贡献。没有任何可用文本时返回 ``None``，不会制造空分隔
段；它只负责 Host 静态引导材料，不选择 History，也不把文件内容写回磁盘。
"""

from __future__ import annotations

from pathlib import Path

from codeflow.context_engine.base import AssemblyContext, Segment
from codeflow.context_engine.segments import render


class BootstrapSegmentBuilder:
    name = "bootstrap"
    order = 2
    needs_prefix = False

    def __init__(self, workspace: Path, bootstrap_files: list[str] | None = None) -> None:
        self._workspace = workspace
        self._bootstrap_files = bootstrap_files

    async def build(self, ctx: AssemblyContext) -> Segment | None:
        text = render.load_bootstrap_files(self._workspace, self._bootstrap_files)
        return Segment(text=text) if text else None
