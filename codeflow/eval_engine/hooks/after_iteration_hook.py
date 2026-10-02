"""After-iteration Task-completion Judge Hook。

Hook 从 Context 提取第一条 User Message 与最后一条 Assistant Content，调用 LLM Judge 把结果分成
Completed / Failed / Unknown，再把非 Unknown Verdict 转给 ``EvalAdapter.record_task_completion``，由
MemoryEngine 写入 ``HISTORY.md``。

它始终保持 Pass-through，Never Short-circuits：Evaluator 只能 Annotate，不能中断 User Reply Chain。
Judge 或 Adapter 出错也会降级为 Quiet No-op。

Phase Mapping 需要特别注意：`AgentLoop` 当前在 ReAct Inner Loop 内触发 ``after_iteration``，所以本
Hook 会在每个满足提取条件的 LLM Iteration 后运行；虽然 Context 提供 ``ctx.iteration``，当前实现
**没有** 单独的 Last-seen Iteration
去重状态。真正的 Once-per-turn 语义要等 `AgentLoop` 提供与 ``after_iteration`` 分离的 Dedicated
``after_turn`` Phase，调用方不能仅凭旧命名假设每 Turn 只 Judge 一次。
"""

from __future__ import annotations

import logging

from codeflow.agent.hook.base import AgentHook, AgentHookContext, HookDecision
from codeflow.eval_engine.adapter.adapter import EvalAdapter
from codeflow.eval_engine.config import EvalEngineConfig
from codeflow.eval_engine.judge.judge import EvalJudge, JudgeVerdict

logger = logging.getLogger(__name__)


class AfterIterationHook(AgentHook):
    """对当前 Context 中最新 Assistant Response 运行 LLM Judge。

     Hook 依赖 `EvalEngineConfig` 决定是否启用，用 `EvalJudge` 生成 Verdict，再由 `EvalAdapter` 写回长期
    记录。生命周期由 Agent Loop 的 After-iteration Phase 驱动；它本身不缓存已判断 Turn，也不会修改
     Messages 或 Final Response。
    """

    def __init__(
        self,
        config: EvalEngineConfig,
        judge: EvalJudge,
        adapter: EvalAdapter,
    ) -> None:
        self._config = config
        self._judge = judge
        self._adapter = adapter

    @property
    def name(self) -> str:
        return "EvalAfterIterationHook"

    async def after_iteration(self, ctx: AgentHookContext) -> HookDecision:
        if not (self._config.enabled and self._config.on_task_completion):
            return HookDecision()

        user_goal, final_response = self._extract(ctx)
        if not user_goal or not final_response:
            return HookDecision()

        try:
            verdict = await self._judge.judge(
                user_goal=user_goal,
                final_response=final_response,
                messages=ctx.messages,
            )
        except Exception as exc:  # noqa: BLE001 — 防御性处理；judge 已自行处理内部错误
            logger.debug(
                "EvalAfterIterationHook judge error %s: %s",
                type(exc).__name__,
                exc,
            )
            verdict = JudgeVerdict.unknown

        if verdict is JudgeVerdict.unknown:
            return HookDecision()

        try:
            self._adapter.record_task_completion(
                verdict=verdict,
                user_goal=user_goal,
                session_key=ctx.session_key,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "EvalAfterIterationHook adapter error %s: %s",
                type(exc).__name__,
                exc,
            )
        return HookDecision(
            notes=[f"eval_verdict={verdict.value}"],
        )

    @staticmethod
    def _extract(ctx: AgentHookContext) -> tuple[str, str]:
        """从 Context 提取 ``user_goal`` 与 ``final_response``。

        ``user_goal`` 取 ``ctx.messages`` 中 First User Message，``final_response`` 取 Last Assistant
        Content，并从尾部反向扫描以跳过更早回复。Structural Mismatch 时返回 Empty Strings，让 Hook
        Quietly No-op 而不是 Blowing Up；Hook Chain 不适合 Loud Error Reporting。函数只接受字符串
        Content，不把 Tool-only 或结构化 Blocks 猜测成最终文本。
        """
        messages = ctx.messages or []
        user_goal = ""
        final_response = ""
        for m in messages:
            if not isinstance(m, dict):
                continue
            if m.get("role") == "user" and not user_goal:
                content = m.get("content")
                if isinstance(content, str):
                    user_goal = content
        # 从最新一条非工具 assistant 消息开始遍历响应链。
        for m in reversed(messages):
            if not isinstance(m, dict):
                continue
            if m.get("role") == "assistant":
                content = m.get("content")
                if isinstance(content, str) and content:
                    final_response = content
                    break
        return user_goal, final_response


__all__ = ["AfterIterationHook"]
