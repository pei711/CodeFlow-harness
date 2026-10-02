"""TokenWise 的 Core Abstractions，与其 Implementations 放在同一 Package。

每个 Strategy 与它实现的 ABC 相邻，方便从接口直接找到行为。Strategies 采用 Additive 组合方式：
同一个 Agent 可以安装多个策略，并按 Registration Order 依次调用每个 Hook。某个 Strategy 对当前
Hook 不感兴趣时继承默认 No-op，因此新增一种优化不必实现整套生命周期。

这一层只定义历史兼容协议，不决定新 Runtime 的优化策略；新的执行路径由
`codeflow.call_efficiency` 承担。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class UsageSnapshot:
    """记录一次 LLM Call 的 Token Usage 与 Cost。

    Convention：``input_tokens`` 只表示 *fresh*、即 Non-cached Prompt Tokens。
    ``codeflow.call_efficiency`` 会先统一不同 Provider 对 Total/Fresh 的计数 Convention，再把记录投影到
    这个 Historical Schema，使用者不应自行把 Cache Read 再从该字段扣除一次。

    ``trace_id`` 与 ``turn_span_id`` 把持久化 Usage Row 关联回真正花费 Token 的 Turn，相关契约见
    ``docs/specs/turn-evidence-correlation.md``。Tracing Disabled 时二者都保持 `None`；字段出现之前写入
    的旧记录也读取成不可 Join，而不是猜测关联后造成 Mis-joined Evidence。

    ``estimated_cost_usd`` 在 Pricing Unavailable 时为 `None`。数值零只保留给已知费率确实计算出
    Real Zero Cost 的调用，因此“未知成本”和“成本为零”是两个不同状态。
    """

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    estimated_cost_usd: float | None = None
    session_key: str | None = None
    trace_id: str | None = None
    turn_span_id: str | None = None


class TokenStrategy(ABC):
    """为 Token 与 Cost Optimization 提供 Cross-cutting Hooks 的统一 ABC。

    Strategies 可以 Additive 安装多个，Agent 按 Registration Order 调用每个 Hook；对某个阶段不
    感兴趣的策略直接继承默认 No-op。接口特意统一为一个类型，而不是拆成 Four Tiny ABCs，从而让
    Strategy Registry 只有一个简单 Install Point；具体策略通常只实现一到两个 Hooks。

    生命周期围绕一次 LLM Call 展开：`before_llm_call` 可改写请求，调用完成后
    `after_llm_call` 消费响应与计量快照。实现者必须保持未关注字段原样传递，避免一个优化策略意外
    改变另一个策略或 Provider 的语义。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """返回稳定的 Strategy Identifier，例如 ``cache_optimizer`` 或 ``smart_router``。

        Registry 用这个名字查找、诊断与区分策略；它应描述策略身份，而不是某次调用状态。具体实现
        必须提供该属性，重复名称如何处理由 Registry 约束。
        """

    async def before_llm_call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, str]:
        """Pre-process 即将发送的请求，返回 ``(messages, tools, model)``。

        `CacheOptimizer` 用它标记 ``cache_control``，`SmartRouter` 用它选择 Model，
        `ToolResultPruner` 用它改写旧 Tool Output Blocks。多个策略会看到前序策略的返回值，因此每个
        实现都必须返回完整三元组。Default 行为是 Pass Through，不复制也不改写输入。
        """
        return messages, tools, model

    async def after_llm_call(
        self,
        response: dict[str, Any],
        usage: UsageSnapshot,
    ) -> None:
        """执行 Post-call Hook，供 `UsageTracker`、`BudgetAlerter` 等策略消费结果。

        `response` 是已经取得的 LLM Response，`usage` 是标准化后的 `UsageSnapshot`。默认实现为
        No-op；该 Hook 没有返回值，也不负责替换模型回复，典型用途是持久化计量或发出预算告警。
        """


__all__ = ["TokenStrategy", "UsageSnapshot"]
