"""把多个 ``AgentHook`` 聚合成一个有顺序、短路与异常隔离的 CompositeHook。

每个 Phase 严格按 Registration Order 运行：``CompositeHook([A, B, C])`` 是 A→B→C，Late append
放末尾。第一个返回 ``HookDecision(short_circuit_result=…)`` 的 Hook 赢得当前 Phase，Subsequent
Hook NOT called。

只有声明支持的 Phase（当前 after_send）Chain ``modified_content``：每个 Output 写入
``ctx.outbound_content`` 成为 Next Input，最终 Decision 携带 Fully-chained Text。Hook Exception
记录后按 Pass-through No-op 继续，镜像 EventBus Contract；单个 Flaky Personalizer/Eval LLM
Timeout 不应打断 Whole Turn。
"""

from __future__ import annotations

import logging
from typing import Iterable

from codeflow.agent.hook.base import AgentHook, AgentHookContext, HookDecision

logger = logging.getLogger(__name__)


_CHAIN_MODIFIED_PHASES = frozenset({"after_send"})


class CompositeHook(AgentHook):
    """把每个 AgentHook Phase Dispatch 给有序 Child List 的 Aggregate Hook。

    实例支持 append/extend、len 与 iteration，Name 也显示 Child Order。所有公开 Phase 都委托
    `_run_phase`，因此 Short Circuit、Content Chain 与 Exception Isolation 只有一个实现，不会因
    Phase 新增产生语义漂移。空 Composite 等价 Pass-through Hook。
    """

    def __init__(self, hooks: Iterable[AgentHook] | None = None) -> None:
        self._hooks: list[AgentHook] = list(hooks or [])

    @property
    def name(self) -> str:
        if not self._hooks:
            return "CompositeHook(empty)"
        return "CompositeHook(" + ", ".join(h.name for h in self._hooks) + ")"

    def __len__(self) -> int:
        return len(self._hooks)

    def __iter__(self):
        return iter(self._hooks)

    def append(self, hook: AgentHook) -> None:
        """把一个 Hook 追加到 Chain 末尾。

        新 Hook 从之后的 Phase Invocation 起按 Registration Order 运行；方法不回放已经发生的
        Phase，也不去重同一 Instance。调用方应在 Turn 运行前完成常规接线。
        """
        self._hooks.append(hook)

    def extend(self, hooks: Iterable[AgentHook]) -> None:
        """按 Iterable Order 把多个 Hook 追加到 Chain 末尾。

        等价连续 append，并保留输入顺序。方法不复制 Hook 实例或验证名称唯一性；同一 Hook
        重复出现会被重复调用，这是 Caller 明确注册的结果。
        """
        self._hooks.extend(hooks)

    # ─────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────

    async def before_user_inbound(self, ctx: AgentHookContext) -> HookDecision:
        return await self._run_phase("before_user_inbound", ctx)

    async def before_iteration(self, ctx: AgentHookContext) -> HookDecision:
        return await self._run_phase("before_iteration", ctx)

    async def before_execute_tools(self, ctx: AgentHookContext) -> HookDecision:
        return await self._run_phase("before_execute_tools", ctx)

    async def after_iteration(self, ctx: AgentHookContext) -> HookDecision:
        return await self._run_phase("after_iteration", ctx)

    async def after_send(self, ctx: AgentHookContext) -> HookDecision:
        return await self._run_phase("after_send", ctx)

    # ─────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────

    async def _run_phase(self, phase: str, ctx: AgentHookContext) -> HookDecision:
        """依次调用每个 Child 的 ``phase``，执行 Short-circuit 与 Content-chaining Semantics。

        Child Exception 会 Log 并继续；Short Circuit 立即原样返回该 Decision。对
        `_CHAIN_MODIFIED_PHASES`，非空 modified_content 同步写入 Context 并记为最后修改；其他
        Phase 忽略修改字段。全部完成后返回 Pass-through HookDecision，并在适用时携带最终
        Chained Content。未知 Phase 的 getattr Error 不在 Hook try 内，表示 Composite 调用缺陷。
        """
        chain_content = phase in _CHAIN_MODIFIED_PHASES
        last_modified: str | None = None

        for hook in self._hooks:
            method = getattr(hook, phase)
            try:
                decision = await method(ctx)
            except Exception:
                logger.exception(
                    "hook %s.%s raised; treating as no-op and continuing",
                    hook.name,
                    phase,
                )
                continue

            if decision.short_circuit_result is not None:
                return decision

            if chain_content and decision.modified_content is not None:
                ctx.outbound_content = decision.modified_content
                last_modified = decision.modified_content

        return HookDecision(modified_content=last_modified)


__all__ = ["CompositeHook"]
