"""Tool-safety Audit Prompt。

这是 LLM-driven Tool Audit 的 Stub Scaffold。当前 Default Tool-audit Behavior 仍是
``ToolAuditHook.before_execute_tools`` 内的 Deterministic Deny-list Check；未来只有在 Deny-list
Inconclusive 时，才可能调用此 Prompt。

下面的英文常量是尚未接线的 Machine-facing Template，不属于 Docstring，因此保持 ``allow`` / ``deny``
/ ``unknown`` Protocol 不变。模板存在不代表运行时已经执行 LLM Safety Review。
"""

TOOL_SAFETY_PROMPT = """You are auditing whether an AI assistant should be allowed to invoke a tool.

Tool: {tool_name}
Arguments:
{tool_arguments}

Context (last user message): {user_goal}

Answer with ONE word on a single line:
- "allow"   — the call is consistent with the user's intent and not destructive.
- "deny"    — the call is unsafe, off-topic, or violates a policy.
- "unknown" — insufficient context.

Your one-word verdict:"""


__all__ = ["TOOL_SAFETY_PROMPT"]
