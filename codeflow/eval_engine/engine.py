"""`EvalEngine` Orchestrator，集中装配评估阶段所需依赖。

它持有 Config，构造三个 `AgentHook` Instances 以及 Judge + Adapter Dependencies，并通过唯一
:meth:`hooks` Accessor 按稳定顺序返回 Hooks。Eval-aware CLI Stack 因而可直接执行
``CompositeHook.extend(engine.hooks())``，无需重新实现 Wire-up。

没有 LLM Provider 或 MemoryEngine 的 Caller 仍可构造 Degraded `EvalEngine`，这对只想覆盖
Deterministic Deny-list Path 的 Tests 很有用。降级实例会明确使用 No-op Judge/Hook，而不是假装已经
完成评估。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeflow.eval_engine.adapter.adapter import EvalAdapter
from codeflow.eval_engine.config import EvalEngineConfig
from codeflow.eval_engine.hooks.after_iteration_hook import AfterIterationHook
from codeflow.eval_engine.hooks.before_iteration_hook import BeforeIterationHook
from codeflow.eval_engine.hooks.tool_audit_hook import ToolAuditHook
from codeflow.eval_engine.judge.judge import EvalJudge

if TYPE_CHECKING:
    from codeflow.agent.hook import AgentHook
    from codeflow.memory_engine.consolidate.consolidator import MemoryStore
    from codeflow.providers.base import LLMProvider


class EvalEngine:
    """通过 Single Factory 聚合 Eval Engine 的三个 Hooks。

    Phase B-3 将 ``memory`` Argument 从已删除的 ``MemoryEngine`` Facade 改为
    :class:`MemoryStore`，因为 Adapter 唯一使用的方法是 ``append_history``。初始化时，有 Provider 才
    建立真实 `EvalJudge`，有 Memory Store 才建立真实 After-iteration Write-back；其余阶段仍保持可用。

    实例生命周期与挂载它的 Agent Loop 一致，Config 在创建时固定。Engine 只负责装配，不自行触发
    Hook，也不合并 Verdict 与 Runtime Outcome。
    """

    def __init__(
        self,
        config: EvalEngineConfig | None = None,
        *,
        memory: "MemoryStore | None" = None,
        provider: "LLMProvider | None" = None,
    ) -> None:
        self._config = config or EvalEngineConfig()

        # Judge 需要 Provider；若未提供（例如测试框架或禁用配置的部署），
        # 则替换为始终返回 unknown 的桩，使 after-iteration hook 保持静默。
        if provider is not None:
            self._judge: EvalJudge = EvalJudge(
                provider,
                model=self._config.judge_model,
                timeout_seconds=self._config.judge_timeout_seconds,
            )
        else:
            self._judge = _NoopJudge()  # type: ignore[assignment]

        self._adapter: EvalAdapter | None = EvalAdapter(memory) if memory is not None else None

        self._before_iteration = BeforeIterationHook(self._config)
        self._tool_audit = ToolAuditHook(self._config)
        self._after_iteration = (
            AfterIterationHook(self._config, self._judge, self._adapter) if self._adapter is not None else _NoopHook()  # type: ignore[assignment]
        )

    @property
    def config(self) -> EvalEngineConfig:
        return self._config

    def hooks(self) -> list["AgentHook"]:
        """按 Canonical Iteration Order 返回三个 Hooks。

        顺序固定为 Before Iteration、Tool Audit、After Iteration，确保 `CompositeHook` 在正确生命周期
        阶段调用对应逻辑。每次返回新 List，但 Hook Instances 为 Engine 内同一对象。
        """
        return [
            self._before_iteration,
            self._tool_audit,
            self._after_iteration,
        ]


# ---------------------------------------------------------------------------
# Engine 在没有 Provider / MemoryEngine 时使用的桩回退。
# 让 ``EvalEngine.hooks()`` 仍返回真实 AgentHook 实例，确保下游
# CompositeHook 使用时满足类型契约。
# ---------------------------------------------------------------------------


from codeflow.agent.hook.base import AgentHook, AgentHookContext, HookDecision
from codeflow.eval_engine.judge.judge import JudgeVerdict


class _NoopJudge:
    """没有 LLM Provider 时使用的 Drop-in Judge。

    它始终返回 ``JudgeVerdict.unknown``，使 `AfterIterationHook` 保持 Quiet，不写下没有真实判断依据的
    Completed/Failed 结果。接口与真实 Judge 一致，便于 Engine 无条件装配。
    """

    async def judge(
        self,
        user_goal: str,
        final_response: str,
        messages=None,
    ) -> JudgeVerdict:
        return JudgeVerdict.unknown


class _NoopHook(AgentHook):
    """`EvalEngine` 没有 MemoryEngine Wire-up 时使用的 Fallback Hook。

    每个 Phase 都是 Pass-through，不产生 Verdict Write-back。它与普通 ``AgentHook()`` 的行为差别不大，
    单独命名只是为了让 Debug Logs 能清晰识别 Eval Degradation，而不是误以为真实 After-iteration
    Evaluation 已运行。
    """

    @property
    def name(self) -> str:
        return "EvalNoopHook"

    async def after_iteration(self, ctx: AgentHookContext) -> HookDecision:
        return HookDecision()


__all__ = ["EvalEngine"]
