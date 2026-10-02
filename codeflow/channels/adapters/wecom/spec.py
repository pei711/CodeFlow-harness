"""WeCom Channel 的 Declarative `ChannelSpec`。

Import 本 Module **不会** Import ``wecom_aibot_sdk``；SDK 延迟到 Factory 构造 `WecomChannel`。Spec Maturity
为 ``beta``，只声明当前 Evidence Level，不启动 Long Connection。
"""

from __future__ import annotations

from codeflow.channels.contract import Capabilities, ChannelSpec


def _make(config):
    from codeflow.channels.adapters.wecom.channel import WecomChannel

    return WecomChannel(config)


SPEC = ChannelSpec(
    display_name="WeCom",
    factory=_make,
    capabilities=Capabilities(),
    maturity="beta",
)
