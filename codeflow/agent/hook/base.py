"""定义 AgentLoop Extension 使用的 AgentHook ABC、Context 与 Decision。

Eval Engine 在此契约上构建三个具体 Hook。Initial cut 刻意保持窄：所有 Phase 默认返回 pass-through
``HookDecision()``，子类只覆盖关心阶段；接口全部 ``async``，同步实现可立即 ``return``。即使
当前 ``response_modifier`` 类逻辑是 ``(str, str) -> str``，未来 LLM Eval Judge 或 Personalizer
Classifier 也无需重做协议。

HookDecision 表达三个 Orthogonal Mode：默认 pass-through 让 Chain/Main Loop 继续；
``short_circuit_result`` 停止当前 Phase Chain，并按阶段成为 Final Answer/Outbound Reply；
``modified_content`` 仅在 ``after_send`` 改写出站文本。Hook 每次创建 Fresh Decision，
CompositeHook 负责把修改写入 ``ctx.outbound_content`` 给下一个 Hook。

AgentHookContext 按 Phase 填字段，并非每项始终有意义：``turn_request`` 在 before_user_inbound 有值，
before_iteration 可为 None。AgentLoop 负责在调用前填 Relevant Fields；Hook 必须容忍未使用字段
为 None，不能把 Mutable Context 当作跨 Turn Global State。
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from codeflow.spine.turn import TurnRequest


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@dataclass
class AgentHookContext:
    """承载一个 AgentLoop Turn 在 Hook Chain 各 Phase 间逐步出现的 State。

    ``session_key`` 是唯一 Always-required Field；turn_request、iteration、messages、tools、response、
    outbound_content 按对应 AgentHook Method 的 Phase Contract 填入。``metadata`` 允许 Extension 在
    同一 Turn 内共享命名信息，但 Context 不应跨 Session 复用。

    Context 是 Mutable：AgentLoop/CompositeHook 会随着 LLM 返回与最终回复形成而更新字段，Hook
    也可改自己拥有的 metadata。改写 Outbound Text 的 Canonical Channel 仍是
    ``HookDecision.modified_content``；在 ``after_send`` 中 CompositeHook 将它传播给 Next Hook，
    避免 Hook 只原地改 Context 却没有明确 Decision Evidence。
    """

    session_key: str

    turn_request: "TurnRequest | None" = None

    iteration: int | None = None
    messages: list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    response: Any | None = None  # 可以是 LLMResponse 或字典；保留 Any 以避免

    outbound_content: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HookDecision:
    """记录一次 Hook Invocation 对当前 Phase 做出的控制决定。

     默认 ``pass_through=True`` 且没有 Short Circuit/Modification，Chain 与 Main Loop 继续。
     ``short_circuit_result`` 有值时停止当前 Chain，AgentLoop 把它视为本 Phase Final Answer；
     ``modified_content`` 只对 ``after_send`` 有意义，Next Hook 会从 ``ctx.outbound_content`` 看到
    改写。``notes`` 可携带观察说明，不改变控制流。

     Hook 不应同时设置 short_circuit_result 与 modified_content：Short Circuit 后下游不再处理，
     Content Modification 没有消费者。CompositeHook 以 Short Circuit 优先。
    """

    pass_through: bool = True
    short_circuit_result: Any | None = None
    modified_content: str | None = None
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


class AgentHook(ABC):
    """为 AgentLoop Lifecycle Extension 提供全部 Phase 的默认 No-op Base Class。

    子类只覆盖需要的 Async Method，其他阶段自动返回 Pass-through ``HookDecision``。Hook 可以观察
    User Inbound、每次 ReAct Iteration、Tool Execute 前后与 Final Send 前，但不能绕过 AgentLoop
    直接交付第二份回复。CompositeHook 负责顺序、异常隔离、Short Circuit 与 Modification Chain。
    """

    @property
    def name(self) -> str:
        """返回出现在 Error Log 中的 Human-readable Hook Identifier。

         默认使用 Concrete Class Name；若 Class Name 无法表明 Source Subsystem，子类应覆盖为稳定
        名称。它只用于诊断，不决定 Registration Order 或 Dispatch。
        """
        return type(self).__name__

    async def before_user_inbound(self, ctx: AgentHookContext) -> HookDecision:
        """在 Fresh User Message 到达、AgentLoop Dispatch 给 LLM 前触发。

        此时 Context 已填 ``session_key`` 与 ``turn_request``，适合入站 Audit、过滤或明确 Short
        Circuit；Iteration、Response 与 Outbound 尚未产生。CRON/SUBAGENT 等非 User Origin 会由
        AgentLoop 跳过该 Phase。默认 Pass-through。
        """
        return HookDecision()

    async def before_iteration(self, ctx: AgentHookContext) -> HookDecision:
        """在 ReAct Loop 每次 LLM Call 前触发。

        Context 已填 ``session_key``、``iteration``、``messages``、``tools``。典型用途是 Token Budget
        Check，在下一轮会超限时拒绝启动；或 Pruning，在工作已经明确完成时跳过 Iteration。
        Response 尚不存在，Hook 不应读取旧轮 Response 作为当前事实。默认 Pass-through。
        """
        return HookDecision()

    async def before_execute_tools(self, ctx: AgentHookContext) -> HookDecision:
        """在 LLM 返回 ``tool_calls`` 后、任何 Tool 真正执行前触发。

        Context 已填 ``session_key``、``iteration``、``messages`` 与携带 Tool Calls 的 ``response``，
        适合 Pre-tool-call Audit/Approval。Short Circuit 可阻止执行并返回受控结果；Hook 不应自行
        调用这些 Tool，否则会绕过 Registry 的参数、超时与 Event Boundary。默认 Pass-through。
        """
        return HookDecision()

    async def after_iteration(self, ctx: AgentHookContext) -> HookDecision:
        """在每次 Iteration 的 LLM Call 与该轮 Tool Execution 都结束后触发。

        Context 已填 ``session_key``、``iteration``、更新后的 ``messages`` 与 ``response``。Hook 可
        判断 Loop Completion，或判断 Case Success 并经 memory_engine 写 ``case.md``。该判断只
        覆盖当前 Iteration Evidence；Short Circuit 才会改变后续控制流。默认 Pass-through。
        """
        return HookDecision()

    async def after_send(self, ctx: AgentHookContext) -> HookDecision:
        """在 Final Outbound Content 已组装、真正作为 Reply 发送前触发。

        Context 已填 ``session_key`` 与 ``outbound_content``。返回
        ``HookDecision(modified_content=...)`` 会改写 Outbound Text；CompositeHook 按 Registration
        Order Chain 修改，使 Downstream Hook 读取 Upstream Output。该 Phase 不应再次调用 Message
        Tool 或制造第二个 Delivery。默认不修改。
        """
        return HookDecision()


__all__ = ["AgentHook", "AgentHookContext", "HookDecision"]
