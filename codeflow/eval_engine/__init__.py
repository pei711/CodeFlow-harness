"""Eval Engine，是 L3 Cognition-coord Task Judge。

它提供三个 `AgentHook` Implementations，挂在 `AgentLoop` 的不同 Lifecycle Phases，回答互相 Orthogonal
的三个问题：

- ``BeforeIterationHook``：“是否应该开始下一次 Iteration？”，负责 Token Budget / Pruning；
- ``ToolAuditHook``：“这个 Tool Call 是否可以安全执行？”，提供 Deny-list 与 Approval Workflow Stub；
- ``AfterIterationHook``：“这个 Turn 是否成功完成？”，用 LLM Judge 检查 Final Response，再通过 Memory
  Adapter 记录 Verdict。

三者 **Default 都关闭**，即 ``EvalEngineConfig.enabled = False``。它们通过 CLI Stack 挂载到
`AgentLoop`，Wire-up 见 ``cli/_eval_stack.py``。Judge Verdict 是一项独立评估证据，不应覆盖 Runtime
真实 Outcome，更不能把 LLM 的肯定判断直接等同于外部任务已经完成。

Layout：

    eval_engine/
      ├── config.py              Pydantic ``EvalEngineConfig``
      ├── engine.py              ``EvalEngine`` Orchestrator
      ├── hooks/
      │   ├── before_iteration_hook.py
      │   ├── tool_audit_hook.py
      │   └── after_iteration_hook.py
      ├── judge/
      │   └── judge.py           LLM Judge Invocation
      ├── adapter/
      │   └── adapter.py         MemoryEngine Write-back
      └── prompts/
          ├── task_completion.py
          └── tool_safety.py
"""

from codeflow.eval_engine.config import EvalEngineConfig
from codeflow.eval_engine.engine import EvalEngine
from codeflow.eval_engine.hooks.after_iteration_hook import AfterIterationHook
from codeflow.eval_engine.hooks.before_iteration_hook import BeforeIterationHook
from codeflow.eval_engine.hooks.tool_audit_hook import ToolAuditHook
from codeflow.eval_engine.judge.judge import EvalJudge, JudgeVerdict

__all__ = [
    "EvalEngine",
    "EvalEngineConfig",
    "BeforeIterationHook",
    "ToolAuditHook",
    "AfterIterationHook",
    "EvalJudge",
    "JudgeVerdict",
]
