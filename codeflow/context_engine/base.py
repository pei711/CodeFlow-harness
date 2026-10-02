"""定义 Context Engine 的抽象契约与所有 SegmentBuilder 共享的数据形状。

一次 Turn 的完整 Session 往往大于模型窗口，也缺少 System Prompt、Memory、Skill 和当前媒体。
Context Engine 的任务是从这些候选材料中组装“主 Agent 本轮真正能看到的消息列表”，同时让
Token 预算、压缩所有权和注入证据保持可观察。它不调用 LLM 执行业务，也不保存 Session。

``codeflow.context_engine`` 当前只有一个 shipping implementation：:class:`ContextAssembler`。
AgentLoop 经 :func:`codeflow.context_engine.build_context_engine` 构建它，并且始终只持有一个
``self.context_engine: ContextEngine`` 引用。保留 ABC 是为了实验时可替换 Engine，而不是暗示
生产路径同时运行多个实现。

包名使用 ``context_engine`` 而非 ``context``，一方面对应 L4 ``memory_engine``，另一方面避开
:mod:`codeflow.agent.context`；后者的 :class:`ContextBuilder` 是 Engine 会调用的低层构建工具，
不是本轮窗口所有者。``AssembledContext`` 与 ``TokenBudget`` 仍位于
:mod:`codeflow.memory_engine.base`，因为它们早于本 ABC；``MemoryEngine`` 与 ``ContextEngine`` 是
peer L4 abstraction，这两个 dataclass 是共享 value object，不属于任一 Engine 私有契约。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from codeflow.memory_engine.base import AssembledContext, TokenBudget

if TYPE_CHECKING:
    # 避免运行时导入：``curator`` 会反向从本模块导入 ``ContextEngine``，
    # 因此仅在类型提示中引用 ``TurnContext``，以打破循环依赖。
    from codeflow.context_engine.curator import TurnContext
    from codeflow.spine.message import Media


# ---------------------------------------------------------------------------
# SegmentBuilder 抽象：统一的上下文贡献者模型
# ---------------------------------------------------------------------------
#
# Turn 上下文的每一部分都由 :class:`SegmentBuilder` 产生。
# seg1–5（identity / bootstrap / memory / active-skills / skills）和 Curator
# 都是 SegmentBuilder，不再另设“lane”类别。
# :class:`ContextAssembler` 分两个阶段运行它们，并路由其
# 将输出写入 system / history 槽位。


@dataclass(frozen=True)
class AssembledPrefix:
    """保存 Phase A 已确定的固定 Prompt 前缀，供 Phase B 的 Curator 精确预算。

    Phase A 的独立 Builder 先产出 System Segment；Assembler 把它们连接成 ``system_prefix``，
    再连同当前 ``user_message`` 与 ``tool_defs`` 放进本对象。Phase B Builder 因而能先计算
    fixed prompt overhead，再把剩余 Token 精确分配给 ``*history``，不会用预估的 System 大小
    重复裁剪。

    对象冻结且只在两阶段之间传递，不代表完整 `AssembledContext`；History 尚未由 Curator
    选择，最终消息也尚未按 system/history/user 槽位合并。
    """

    system_prefix: str
    user_message: dict[str, Any]
    tool_defs: list[dict[str, Any]]


@dataclass(frozen=True)
class AssemblyContext:
    """向每个 :class:`SegmentBuilder` 提供同一 Turn 的只读组装输入。

    ``session_key`` 关联会话状态，``current_message`` 与 ``media`` 是本轮用户载荷，Channel/Chat
    帮助构建运行时上下文；``session_messages`` 是候选历史而非已经裁好的模型窗口，``budget``
    则规定可用 Token。Builder 读取这些字段并返回 Segment，不能原地改写共享上下文。

    Phase A 运行彼此独立的 Builder，此时 ``prefix`` 为 ``None``，它们必须忽略该字段。
    :class:`ContextAssembler` 收集 Phase A 后填入 `AssembledPrefix`，再运行需要固定开销信息的
    Phase B Builder；后者要求 prefix 已存在。frozen dataclass 让这条阶段边界可由类型和值检查。
    """

    session_key: str
    current_message: str
    media: list[str | Media] | None
    channel: str | None
    chat_id: str | None
    session_messages: list[dict[str, Any]]
    budget: TokenBudget
    prefix: AssembledPrefix | None = None


@dataclass
class Segment:
    """统一承载一个 :class:`SegmentBuilder` 对本轮 Context 的贡献。

    ``text`` 写入 System slot，Assembler 按 Builder 的 ``order`` 连接；空字符串 ``""`` 表示
    本轮没有该 Segment。``history`` 写入 History slot，目前只有 Curator 设置，其他 Builder
    保持 ``None``，从而避免多个所有者同时裁剪历史。``meta`` 会合并进
    ``AssembledContext.metadata``，例如 ``injected_skill_ids``、``memory_hits`` 与 ``path``，
    供 Turn evidence 使用而不暴露给模型正文。

    Segment 是 Builder 与 Assembler 的内部产品，不保证单独就是合法 Prompt；只有全部 Segment
    完成排序、冲突检查和 User message 拼接后才形成模型输入。
    """

    text: str = ""
    history: list[dict[str, Any]] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SegmentBuilder(Protocol):
    """定义一个独立 Context 贡献者，seg1–5 与 Curator 都实现该协议。

    ``name`` 用于身份与诊断，``order`` 固定其在 System Prompt 中的位置，``needs_prefix`` 决定
    调度阶段：默认 ``False`` 的 Builder 进入 Phase A 并行批次；值为真表示它读取
    ``ctx.prefix``，必须等固定前缀形成后进入 Phase B。Builder 只返回自己的 `Segment` 或
    ``None``，不直接拼最终消息。

    统一 Protocol 取代 identity/bootstrap/memory/skills 与 Curator 的两套“lane”类别，使新增
    贡献者只需声明依赖和顺序。实现仍必须尊重只读 AssemblyContext 和 TokenBudget，不能因
    并行执行共享可变 Turn 状态。
    """

    name: str
    order: int
    needs_prefix: bool

    async def build(self, ctx: AssemblyContext) -> "Segment | None":
        """根据只读 ``ctx`` 构建本轮 :class:`Segment`，无贡献时返回 ``None``。

        返回 ``None`` 与空 ``Segment.text`` 都不会制造 System 文本，但前者表示整个 Builder
        本轮缺席。实现可异步读取 Memory 或 Skill；异常由 Assembler 的阶段策略处理，本协议
        不把失败自动转换为空贡献，也不允许实现直接修改最终消息列表。
        """
        ...


class ContextEngine(ABC):
    """决定主 Agent 的 LLM 在每个 Turn 中究竟能看到哪些消息。

    当前唯一实现 :class:`ContextAssembler <codeflow.context_engine.assembler.ContextAssembler>`
    接收扁平 :class:`SegmentBuilder` 列表：Phase A 并行运行 seg1–5，即 identity、bootstrap、
    memory+recall、active-skills、router-skills；Phase B 再运行 Curator，生成
    ``# Curator Working State`` 与按预算裁剪的 ``*history``。最终结果才是 Provider 输入。

    该实现 ``owns_compaction=True``，所以 AgentLoop 提供完整 append-only Session 候选并把
    compaction 延后到 :meth:`after_turn`；其他实验 Engine 可以声明不同所有权，但必须通过
    `assemble` 返回同一 `AssembledContext` 契约。Engine 不拥有 Tool 执行或回复交付。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """返回写入日志和 metadata 的短 Engine 标识，例如 ``"unified"``。

        该值用于观察与证据区分实现，不应包含 Session 或 Turn 动态状态。具体子类必须提供
        稳定字符串，调用方不应根据类名推断它。
        """

    @property
    @abstractmethod
    def owns_compaction(self) -> bool:
        """声明 History compaction 是否由当前 Context Engine 独占。

         返回 ``True`` 时 AgentLoop 跳过 ``MemoryEngine.maybe_consolidate``，让 Engine 自己管理
        历史；当前 Curator 会 out-of-band 归档消息。返回 ``False`` 时 Host 提供 consolidation 后
         切片。这个属性决定事实所有者，不能只作为性能开关，否则两个 Engine 可能重复压缩。
        """

    @abstractmethod
    async def assemble(
        self,
        session_key: str,
        session_messages: list[dict[str, Any]],
        budget: TokenBudget,
        *,
        turn: "TurnContext",
    ) -> AssembledContext:
        """构建将被原样传给主 Agent LLM 的精确消息列表。

        ``session_key`` 关联 Engine 状态，``session_messages`` 只是候选 History；它究竟来自
        ``session.messages`` 的完整 append-only log，还是 ``session.get_history()`` 的
        post-consolidation slice，由 AgentLoop 根据 :attr:`owns_compaction` 决定。``budget``
        约束窗口，``turn`` 提供当前消息、媒体与 Channel 上下文。

        返回 `AssembledContext` 同时包含最终 messages 与 metadata。实现必须让当前 User 消息、
        System Segment 和选定 History 的顺序明确，不能把候选列表直接当成已经合规的模型输入。
        """

    async def after_turn(
        self,
        session_key: str,
        outcome: dict[str, Any],
    ) -> None:
        """在 Turn 结束后给 Engine 一次可选的账本与归档更新机会。

        当前 Curator 在这里更新 manifest / archives；Legacy 忽略它。``session_key`` 指定会话，
        ``outcome`` 提供本轮结果证据，但默认实现是 no-op，允许未来 Engine 渐进 opt in，而不
        强迫不拥有 compaction 的实现伪造副作用。异常由调用方按事后流水线语义处理。
        """
        return None


__all__ = [
    "AssembledPrefix",
    "AssemblyContext",
    "ContextEngine",
    "Segment",
    "SegmentBuilder",
]
