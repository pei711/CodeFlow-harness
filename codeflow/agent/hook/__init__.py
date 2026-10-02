"""重导出 AgentLoop Lifecycle 使用的 AgentHook Abstraction。

Eval Engine 与 Caller-supplied Extension 共用 User-inbound、Iteration、Tool、Outbound Phase 契约。
:class:`AgentHook` 提供全 No-op Base；:class:`AgentHookContext` 携带 Per-turn State；
:class:`HookDecision` 表达 Pass-through、Short-circuit、Content Modification；
:class:`CompositeHook` 负责多 Hook 的 Order、Content Chain 与 Exception Isolation。External Caller
应从此入口导入，不依赖 Base/Composite 文件布局。
"""

from codeflow.agent.hook.base import AgentHook, AgentHookContext, HookDecision
from codeflow.agent.hook.composite import CompositeHook

__all__ = [
    "AgentHook",
    "AgentHookContext",
    "HookDecision",
    "CompositeHook",
]
