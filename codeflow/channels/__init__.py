"""CodeFlow Chat Channels 的 Public Contract Surface。

Runtime 把 Inbound Platform Event 规范成 Message/Turn Request，并把 Delivery Outcome 交回 Channel Outlet。
Caller 应从这里 Import `Channel`、`ChannelSpec`、Capabilities/Optional Protocols 与 `ChannelManager`，不要
依赖 Internal File Layout，使 Adapter 组织可变化而不 Break Callers。

Channel Start/Send Call Success、Turn Completion 与 User-visible Delivery 是不同阶段；Manager/Outlet 负责
把它们明确关联。
"""

from codeflow.channels.base import ChannelBase
from codeflow.channels.contract import (
    Capabilities,
    Channel,
    ChannelSpec,
    SupportsLogin,
    SupportsStreaming,
)
from codeflow.channels.manager import ChannelManager

# 公共接口即适配器实现的契约类型。校验辅助函数
# （capability_violations）位于 channels.contract。
__all__ = [
    "Capabilities",
    "Channel",
    "ChannelBase",
    "ChannelManager",
    "ChannelSpec",
    "SupportsLogin",
    "SupportsStreaming",
]
