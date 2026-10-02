"""`ChannelOutletAdapter`：把 Channel Outbound Send Surface 适配为 Spine Outlet。

Turn Deliverables 通过统一 ``channel.send`` 到达 Platform。这里只处理 Outbound；Inbound ``intake -> submit``
仍属于 Channel。Dependency Direction 是 Spine Never Imports Channels，Channels 在此 Import Spine Vocabulary，
避免 Core Runtime 耦合 Adapter。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeflow.spine.delivery import Capabilities
from codeflow.spine.events import Deliverable, MediaOut, Text

if TYPE_CHECKING:
    from codeflow.channels.contract import Channel


class ChannelOutletAdapter:
    """把 Channel Wrap 成 Outlet，并渲染 Final Text/Media Deliverables。

    `Text` / `MediaOut` 调用 ``channel.send(...)``；StreamDelta / Reasoning / ToolEvent / Notice 被 Consume，
    因当前 Channel Capability Non-streaming，只展示 Final Reply，尚不支持 Edit-in-place Streaming。Real Send
    Failure Raise 给 Hub Retry；不渲染中间事件不是 Failure。

    Deliverable Target 放在 ``source``，Hub 已按 ``source.channel`` 路由，Reply 回同 Channel/Chat。
    ``reply_to`` Threading 属于 Inbound，不在此处理。Deliver Return 由具体 Channel 定义，不自动确认用户已读。
    """

    def __init__(self, channel: Channel) -> None:
        self._channel = channel
        self.name = channel.name
        self.capabilities = Capabilities(streaming=False)

    async def deliver(self, out: Deliverable) -> None:
        if isinstance(out, Text):
            await self._channel.send(out.source.chat_id, out.content)
        elif isinstance(out, MediaOut):
            await self._channel.send(out.source.chat_id, "", media=[m.path for m in out.media])
        # StreamDelta / Reasoning / ToolEvent / Notice：当前渠道无法渲染，直接消费。
