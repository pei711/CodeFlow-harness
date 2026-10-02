"""`StrategyRegistry` 在 Agent 的 LLM Call 前后串联 `TokenStrategy` Hooks。

Agent Loop 的调用顺序是：

    msgs, tools, model = await registry.before_llm_call(msgs, tools, model)
    response = await provider.chat_with_retry(...)
    await registry.after_llm_call(response_dict, usage_snapshot)

Ordering Guarantees：两个 Hooks 都按 Registration Order 调用。``before_llm_call`` Failure 会向上
传播，因为错误的 Pre-process 不能悄悄发送 Broken Request；``after_llm_call`` Failure 则会捕获并
记录，因为 Telemetry 或 Budget Error 绝不能中止 Main Loop。这个不对称边界让请求正确性优先于
辅助观测的完整性。
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from codeflow.token_wise.base import TokenStrategy, UsageSnapshot


class StrategyRegistry:
    """持有一组有序 `TokenStrategy` Instances，并执行它们的生命周期 Hooks。

    Registry 本身不实现 Token 优化，只维护安装顺序并协调错误边界。请求前，每个策略接收前一个
    策略输出的 Messages、Tools 与 Model；请求后，每个策略独立观察 Response 和 Usage。实例可以在
    运行期 Register 新策略，但调用方不能通过 `strategies` 属性直接修改内部列表。
    """

    def __init__(self, strategies: list[TokenStrategy] | None = None):
        self._strategies: list[TokenStrategy] = list(strategies or [])

    # ---- 内省 ----

    @property
    def strategies(self) -> list[TokenStrategy]:
        """返回 Strategy List 的浅拷贝，Callers 不能借此修改内部顺序。

        列表容器是新的，但其中仍是原 `TokenStrategy` Instances；调用方若直接修改某个策略对象，
        仍会影响后续执行。该属性适合内省和诊断，不是隔离策略状态的快照。
        """
        return list(self._strategies)

    def __len__(self) -> int:
        return len(self._strategies)

    def __bool__(self) -> bool:
        return bool(self._strategies)

    def get(self, name: str) -> TokenStrategy | None:
        """返回第一个名称等于 ``name`` 的 Strategy；没有命中时返回 `None`。

        Registry 允许列表中出现同名实例，本方法只承诺 First Match，并不会校验唯一性。需要操作
        所有同名策略时，调用方应遍历 `strategies`。
        """
        for s in self._strategies:
            if s.name == name:
                return s
        return None

    def register(self, strategy: TokenStrategy, *, first: bool = False) -> None:
        """添加 Strategy；执行 Order 的语义见 Class Docstring。

        ``first=True`` 会插入最前面，让它早于其他策略运行。例如 Tool-list Filter 必须先删减工具，
        `CacheOptimizer` 才能在删减后的 Final Tool 上标记 ``cache_control``。默认行为是追加到末尾；
        本方法不去重，也不立即执行新策略。
        """
        if first:
            self._strategies.insert(0, strategy)
        else:
            self._strategies.append(strategy)

    # ---- 钩子 ----

    async def before_llm_call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, str]:
        """按顺序运行每个 Strategy 的 Before-call Hook，Errors 向上传播。

        每轮把上一策略返回的 ``messages``、``tools``、``model`` 交给下一策略，最终返回完整三元组。
        任一策略抛错都会停止 Chain，确保 Agent 不发送只完成了一部分预处理的请求。
        """
        for s in self._strategies:
            messages, tools, model = await s.before_llm_call(messages, tools, model)
        return messages, tools, model

    async def after_llm_call(self, response: dict[str, Any], usage: UsageSnapshot) -> None:
        """按顺序运行每个 Strategy 的 After-call Hook，Errors 会 Swallowed + Logged。

        一个策略失败不会阻止后续策略继续处理同一 `response` 与 `usage`，也不会反向让已经成功的 LLM
        Call 变成 Turn Failure。代价是某项 Telemetry 或预算记录可能缺失，日志是定位这类证据缺口的
        唯一现场信息。
        """
        for s in self._strategies:
            try:
                await s.after_llm_call(response, usage)
            except Exception as e:
                logger.warning("TokenStrategy '{}' after_llm_call failed: {}", s.name, e)
