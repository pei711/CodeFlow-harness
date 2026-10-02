"""定义 Turn 可产生的单一输出词汇，以及生命周期事件与可投递事件的边界。

Runner 只能发 `RunnerEvent`：Tool 进度、Text、MediaOut、StreamDelta、Reasoning 和 Notice；
Lane Worker 独占 TurnStarted、TurnFailed、TurnEnded 三类 lifecycle event。`TurnEvent` 把两组
合并供事件总线观察，`Deliverable` 则按 DeliveryHub 的职责再次命名 RunnerEvent。统一数据
类让 TUI、REPL 和 channel adapter 共享协议，而不需要理解 Agent Loop 内部回调。
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from codeflow.spine.message import Media, Source


@dataclass(frozen=True)
class Usage:
    """保存一个 Turn 的基础 Token 账目。

    ``prompt_tokens`` 是输入模型的 Token 数，``completion_tokens`` 是模型输出数，
    ``total_tokens`` 是 Provider 报告或规范化后的总数。该结构只承载三项跨出口稳定字段；
    成本、Context 上限等更丰富信息通过调用方的 usage sink 观察，不塞进生命周期协议。
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class NoticeKind(StrEnum):
    """枚举 Turn 向用户呈现的带外提示类型。

    ``PROGRESS`` 表示普通进度，``TOOL_HINT`` 是即将或正在使用 Tool 的简短提示，
    ``INJECTED`` 表示中途消息已合并，``DELIVERY_FAILED`` 则报告最终投递失败。Outlet 可按能力
    选择渲染或吞掉某类 Notice；它们都不是回复正文，也不改变 Turn terminal state。
    """

    PROGRESS = "progress"
    TOOL_HINT = "tool_hint"
    INJECTED = "injected"
    DELIVERY_FAILED = "delivery_failed"


class ToolPhase(StrEnum):
    """标记 ToolEvent 在一次调用生命周期中的触发阶段。

    ``START`` 在执行前携带名称与参数，``COMPLETE`` 在结束后携带结果预览、失败和耗时。
    Outlet 可以用不同 UI 渲染两阶段；该枚举只描述事件时点，不判断 Tool 是否完成了用户
    目标，COMPLETE 也可能带 ``failed=True``。
    """

    START = "start"
    COMPLETE = "complete"


# lifecycle event 由 worker 发出，runner 不得发出。


@dataclass(frozen=True)
class TurnStarted:
    """表示 Worker 已取得并开始执行一个 Turn 的生命周期标记。

    ``conversation_id`` 关联对应 Lane 与事件流；在进入 Origin 并发池之前取消的请求不会
    发出本事件。它没有业务载荷，消费者应把它理解为执行开始事实，而不是回复或成功承诺。
    """

    conversation_id: str | None = None


@dataclass(frozen=True)
class TurnFailed:
    error: str
    cancelled: bool
    conversation_id: str | None = None


@dataclass(frozen=True)
class TurnEnded:
    usage: Usage
    latency_ms: float
    explicit_reply: bool
    conversation_id: str | None = None
    tool_calls: int = 0
    tool_failures: int = 0


# 可投递 event 由 runner 发出，并路由到 Outlet。


@dataclass(frozen=True)
class ToolEvent:
    phase: ToolPhase
    tool_call_id: str
    name: str = ""
    arguments: dict[str, Any] | None = None
    result_preview: str = ""
    truncated: bool = False
    failed: bool = False
    target_call_id: str | None = None
    target_name: str | None = None
    target_arguments: dict[str, Any] | None = None
    duration_ms: float | None = None
    source: Source | None = None
    conversation_id: str | None = None


@dataclass(frozen=True)
class Text:
    content: str
    source: Source | None = None
    reply_to: str | None = None
    conversation_id: str | None = None


@dataclass(frozen=True)
class MediaOut:
    media: tuple[Media, ...]
    source: Source | None = None
    conversation_id: str | None = None


@dataclass(frozen=True)
class StreamDelta:
    delta: str
    stream_id: str | None = None
    source: Source | None = None
    conversation_id: str | None = None


@dataclass(frozen=True)
class Reasoning:
    content: str
    source: Source | None = None
    conversation_id: str | None = None


@dataclass(frozen=True)
class Notice:
    kind: NoticeKind
    source: Source | None = None
    detail: str | None = None
    conversation_id: str | None = None


RunnerEvent = ToolEvent | Text | MediaOut | StreamDelta | Reasoning | Notice
# 同一个 union，按投递职责命名：hub 路由、Outlet 渲染的内容。
Deliverable = RunnerEvent
TurnEvent = TurnStarted | TurnFailed | TurnEnded | RunnerEvent
