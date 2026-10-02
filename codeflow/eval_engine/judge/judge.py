"""LLM Judge，用 Verdict Tag 回答“这个 Turn 是否完成？”。

Judge 刻意保持 Minimal：One Prompt、One LLM Call、Three Possible Verdicts。真实 Production Use 可在
后续深化 Rubric，例如 Per-dimension Scores 与 Structured Failure Reasons；当前结果没有这些细粒度
证据。

Design Choices：

- **Verdict Enum，而非 Free Text**：``JudgeVerdict`` 是 String Enum，使 Adapter 与 Hook Decisions 可
  直接 Branch，Hot Path 不必解析任意 LLM Prose；
- **Pluggable Provider**：Judge 接受任何暴露 ``chat_with_retry`` 的对象；Production 使用 AgentLoop
  持有的同一 ``LLMProvider``，Tests 可传带 Canned Response 的 ``AsyncMock``；
- **Timeout + Safe Fallback**：``asyncio.wait_for`` 强制 Config 的 ``judge_timeout_seconds``。Timeout、
  Exception 或 Unparseable Response 全部回退到 ``JudgeVerdict.unknown``，Hook 随后 Cleanly
  Pass-through。

Verdict 是模型对文本的判断，不是外部副作用验证，也不应覆盖 Runtime 已知的错误事实。
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Any

from codeflow.eval_engine.prompts.task_completion import TASK_COMPLETION_PROMPT

logger = logging.getLogger(__name__)


class JudgeVerdict(str, Enum):
    """``EvalJudge.judge`` 返回的 Three-state Verdict。

    编码为 String Enum，使值能通过 Adapter Cleanly Serialize。`completed` 表示文本看起来满足目标，
    `failed` 表示可见失败，`unknown` 表示没有可靠 Signal；三态避免把未知强行归入失败或成功。
    """

    completed = "completed"  # 已满足用户目标，Turn 正常结束。
    failed = "failed"  # 出现可见错误、拒绝或未达成目标。
    unknown = "unknown"  # 无法确定（超时、解析失败、judge 禁用或 Turn 有歧义）。


class EvalJudge:
    """针对 Turn Final Response 的 Single-call LLM Judge。

    实例持有可插拔 Provider、Judge Model 与 Timeout。每次 `judge` 构造独立评估 Prompt，并以低温度、
    小输出预算请求离散 Verdict。它没有 Conversation Memory，也不会修改被评估的 Agent Messages。
    """

    def __init__(
        self,
        provider: Any,
        *,
        model: str = "claude-haiku-4-5",
        timeout_seconds: float = 8.0,
    ) -> None:
        self._provider = provider
        self._model = model
        self._timeout = timeout_seconds

    async def judge(
        self,
        user_goal: str,
        final_response: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> JudgeVerdict:
        """运行一次 Judge Call，并返回解析后的 Verdict。

        ``user_goal`` 是打开当前 Turn 的 Original User Message；``final_response`` 是 AgentLoop 即将
        返回的 Reply Content。``messages`` 是 Optional Full Message Stream，目前 **Unused**，但提前穿透
        参数可让未来 Deeper Rubric 落地而无需修改 Signature。

        Provider Response 会 Lowercase 后搜索三种 Verdict Value。任意 Error Path，包括 Timeout、Provider
        Exception 或没有可解析 Tag，都返回 ``JudgeVerdict.unknown``。Callers 必须把 `unknown` 当作
        ``no signal``，而不是 ``failed``；即使 `completed` 也只是 Judge 结论，不是任务验收收据。
        """
        prompt = TASK_COMPLETION_PROMPT.format(
            user_goal=user_goal,
            final_response=final_response or "(no response produced)",
        )

        try:
            response = await asyncio.wait_for(
                self._provider.chat_with_retry(
                    messages=[
                        {"role": "system", "content": "You are an evaluation judge."},
                        {"role": "user", "content": prompt},
                    ],
                    model=self._model,
                    max_tokens=64,
                    temperature=0.0,
                ),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            logger.debug("EvalJudge: timed out after %.1fs", self._timeout)
            return JudgeVerdict.unknown
        except Exception as exc:  # noqa: BLE001 — judge 不得导致 AgentLoop 崩溃
            logger.debug("EvalJudge: provider raised %s: %s", type(exc).__name__, exc)
            return JudgeVerdict.unknown

        text = (getattr(response, "content", "") or "").strip().lower()
        for verdict in JudgeVerdict:
            if verdict.value in text:
                return verdict
        return JudgeVerdict.unknown


__all__ = ["EvalJudge", "JudgeVerdict"]
