"""为 SemanticNode 汇总 driver-model transport。

支持 :mod:`.openai_compat` 的 OpenAI-compatible endpoint，以及 :mod:`.claude_cli` 的 local
Claude Code subscription。Package 只提供 transport，不拥有 schema validation 或 gate decision。
"""

from __future__ import annotations
