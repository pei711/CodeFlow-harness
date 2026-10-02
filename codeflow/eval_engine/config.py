"""Eval Engine 的 Pydantic Configuration Model。

模块集中声明总开关、各 Lifecycle Phase 开关、Judge Model/Timeout、Iteration Token Budget 与 Tool
Denylist。Pydantic 禁止未知字段并支持 Camel-case Alias，让配置错误尽早暴露，而不是被静默忽略。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class EvalEngineConfig(BaseModel):
    """Eval Engine 的可调参数集合。

    Default 是 Fully Off：用默认 Config 创建 ``EvalEngine`` 会得到三个不触碰 `AgentLoop` 的 No-op
    Hooks。Operator 必须设置 ``enabled = True``，再按需打开各 Per-phase Toggle 才会激活行为。这种双层
    开关避免仅配置某个细项就意外把 LLM Judge 或 Tool Gate 放进 Hot Path。
    """

    model_config = ConfigDict(extra="forbid", alias_generator=to_camel, populate_by_name=True)

    enabled: bool = False
    """Master Switch；关闭时三个 Hooks 全部为 No-op。"""

    judge_model: str = "claude-haiku-4-5"
    """LLM Judge Call 使用的低成本 Small-batch Model。Judge 位于 Per-turn Hot Path，因此默认选择
    ``haiku``；模型名称只决定调用目标，不提供 Judge 正确性保证。"""

    judge_timeout_seconds: float = 8.0
    """单次 Judge Call 的 Hard Ceiling。Time-out 时返回 ``JudgeVerdict.unknown``，Hook 按
    Pass-through 继续，不阻断正常 Turn。"""

    on_task_completion: bool = True
    """``enabled`` 时运行 After-iteration Judge 并记录 Outcomes。设为 `False` 可单独关闭 Writer，
    而不禁用 Engine 的其他阶段。"""

    on_tool_audit: bool = False
    """Tool Audit 是昂贵的 Per-tool-call Check，默认关闭，使 Engine 上线时保持 Safe but not Loud；
    Denylist 仍只有在该开关启用后才生效。"""

    on_iteration_gate: bool = False
    """每次 Iteration 前运行的 Token-budget / Pruning Gate。默认关闭，让 Common Case 没有额外开销。"""

    max_iteration_tokens: int = 40_000
    """启用 ``on_iteration_gate`` 后，Cumulative Messages 超过该 Token Budget 时拒绝开始下一轮；
    这是估算门禁，不是 Provider 的精确账单上限。"""

    tool_denylist: list[str] = Field(default_factory=list)
    """启用 ``on_tool_audit`` 后，列表内 Tool Names 会在任何 LLM Safety Check 前被 Deterministically
    Blocked；匹配语义由 Tool Audit Hook 定义。"""


__all__ = ["EvalEngineConfig"]
