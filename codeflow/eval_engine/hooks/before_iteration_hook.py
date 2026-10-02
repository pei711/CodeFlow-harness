"""Before-iteration Token-budget / Pruning Gate。

这是三个 Hooks 中 Cheapest 的一个：Zero LLM Calls，只对 Messages List 做 Rough Token Estimate。若
Estimate 超过 ``config.max_iteration_tokens``，Hook 会用 Synthetic ``budget exhausted`` Response
Short-circuit 当前 Iteration，阻止循环继续扩大 Context。

Estimator 刻意保持 Crude，公式是 ``len(json.dumps(msgs)) // 4``。原因是 AgentLoop 其他位置已有更严格
Budget Logic，而这里的目标是 Prevent Runaway Iteration Loop，不是做到 Millisecond-accurate 或提供
Provider Billing Evidence。估算失败返回零并 Pass-through，不能被解读为消息真的不占 Token。
"""

from __future__ import annotations

import json
import logging

from codeflow.agent.hook.base import AgentHook, AgentHookContext, HookDecision
from codeflow.eval_engine.config import EvalEngineConfig

logger = logging.getLogger(__name__)


class BeforeIterationHook(AgentHook):
    """执行 Token-budget / Pruning 的 Iteration Gate。

    只有 ``config.enabled and config.on_iteration_gate`` 同时为 `True` 才会工作，其他情况完全
    Pass-through。启用后对 ``ctx.messages`` 计算 Rough Byte/4 Estimate；预算内继续，超过时返回 Polite
    Halt String 与可诊断 Note。它只阻止下一次 Iteration，不裁剪或修改已有 Messages。
    """

    def __init__(self, config: EvalEngineConfig) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return "EvalBeforeIterationHook"

    async def before_iteration(self, ctx: AgentHookContext) -> HookDecision:
        if not (self._config.enabled and self._config.on_iteration_gate):
            return HookDecision()
        messages = ctx.messages or []
        if not messages:
            return HookDecision()

        estimate = self._estimate_tokens(messages)
        if estimate <= self._config.max_iteration_tokens:
            return HookDecision()

        logger.info(
            "EvalEngine before_iteration: token estimate %d > budget %d; halting iteration",
            estimate,
            self._config.max_iteration_tokens,
        )
        return HookDecision(
            short_circuit_result=(
                "I've hit the conversation token budget for this turn. "
                "Let me know if you'd like me to summarize or start fresh."
            ),
            notes=[f"token_budget_exceeded estimate={estimate}"],
        )

    @staticmethod
    def _estimate_tokens(messages: list[dict]) -> int:
        try:
            return len(json.dumps(messages, ensure_ascii=False, default=str)) // 4
        except Exception:  # noqa: BLE001 — 估算器出错时降级为空操作
            return 0


__all__ = ["BeforeIterationHook"]
