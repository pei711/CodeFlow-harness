"""Spine 是每个 Turn 都必须经过的单一请求与事件骨干。

请求只有一个入口 ``submit``，运行产物只有一个出口 ``emit``；每个 conversation 对应的 Lane
同时是顺序保证与取消边界。Scheduler 负责把 `TurnRequest` 交给 Runner，Runner 发
`RunnerEvent`，DeliveryHub 再按 Source 路由到 Outlet，生命周期事件则由 Lane Worker 独占。

Spine 有意不是 broadcast bus，也不恢复已停用 pub/sub ``bus`` 的多消费者语义。这个包的
`__init__` 只重导出请求、消息、事件、Runner、Scheduler 等稳定词汇，供 Host 和 Channel
共享同一协议；具体 Agent 与 Channel 实现依赖 Spine，而 Spine 不反向依赖它们。
"""

from codeflow.spine.events import (
    Deliverable,
    MediaOut,
    Notice,
    NoticeKind,
    Reasoning,
    RunnerEvent,
    StreamDelta,
    Text,
    ToolEvent,
    ToolPhase,
    TurnEnded,
    TurnEvent,
    TurnFailed,
    TurnStarted,
    Usage,
)
from codeflow.spine.message import ChatType, Media, Source
from codeflow.spine.runner import Emit, TurnOutcome, TurnRunner
from codeflow.spine.scheduler import OriginPools, Scheduler, TurnHandle
from codeflow.spine.turn import BusyPolicy, Origin, TurnRequest

__all__ = [
    "BusyPolicy",
    "ChatType",
    "Deliverable",
    "Emit",
    "Media",
    "MediaOut",
    "Notice",
    "NoticeKind",
    "Origin",
    "OriginPools",
    "Reasoning",
    "RunnerEvent",
    "Scheduler",
    "Source",
    "StreamDelta",
    "Text",
    "ToolEvent",
    "ToolPhase",
    "TurnEnded",
    "TurnEvent",
    "TurnFailed",
    "TurnHandle",
    "TurnOutcome",
    "TurnRequest",
    "TurnRunner",
    "TurnStarted",
    "Usage",
]
