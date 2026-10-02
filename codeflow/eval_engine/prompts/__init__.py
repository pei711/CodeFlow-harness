"""Eval Engine Judge 与 Tool Audit Hooks 使用的 Prompt Templates。

`TASK_COMPLETION_PROMPT` 已接入单次 LLM Judge；`TOOL_SAFETY_PROMPT` 目前只是未来 Tool Safety Check 的
Scaffold，尚未 Wire-up。集中导出模板便于独立评审评估标准，但模板文字本身不是安全 Enforcement。
"""

from codeflow.eval_engine.prompts.task_completion import TASK_COMPLETION_PROMPT
from codeflow.eval_engine.prompts.tool_safety import TOOL_SAFETY_PROMPT

__all__ = ["TASK_COMPLETION_PROMPT", "TOOL_SAFETY_PROMPT"]
