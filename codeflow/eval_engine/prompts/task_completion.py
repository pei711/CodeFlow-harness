"""Task-completion Judge Prompt。

模板要求 Judge 根据 User Original Goal 与 Agent Final Response，把 Turn 分类为 ``completed``、
``failed`` 或 ``unknown``。Prompt 刻意保持 Short，使 Haiku-class Judge Model 能快速、低成本地返回
Single-word Answer。

下面的英文常量是实际发送给模型的 Machine-facing Prompt，不属于 Docstring，因此保持协议文本与三种
英文 Verdict 不变。它只检查回复文本是否看起来完成目标，不验证外部 Tool Side Effects。
"""

TASK_COMPLETION_PROMPT = """You are evaluating whether an AI assistant completed the user's task.

User asked:
\"\"\"
{user_goal}
\"\"\"

Assistant's final response:
\"\"\"
{final_response}
\"\"\"

Answer with ONE word on a single line:
- "completed" — the assistant addressed the user's request and the turn ended cleanly.
- "failed"    — the assistant explicitly errored, refused, or missed the objective.
- "unknown"   — the turn is ambiguous (mid-conversation, clarification asked, etc).

Your one-word verdict:"""


__all__ = ["TASK_COMPLETION_PROMPT"]
