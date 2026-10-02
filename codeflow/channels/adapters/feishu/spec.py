"""Feishu Channel 的 Declarative `ChannelSpec`。

Import 本 Module **不会** Import ``lark_oapi``；Heavy SDK Import 延迟到 Factory 真正构造 `FeishuChannel`。
Spec 声明 Non-streaming Capabilities 与 ``live-gated`` Evidence Maturity，不启动连接。
"""

from __future__ import annotations

from codeflow.channels.contract import Capabilities, ChannelSpec


def _make(config):
    from codeflow.channels.adapters.feishu.channel import FeishuChannel

    return FeishuChannel(config)


SPEC = ChannelSpec(
    display_name="Feishu",
    factory=_make,
    capabilities=Capabilities(),
    maturity="live-gated",
)
