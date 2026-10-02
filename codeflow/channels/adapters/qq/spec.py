"""QQ Channel 的 Declarative `ChannelSpec`。

Import 本 Module **不会** Import ``botpy``；SDK 延迟到 Factory 构造 `QQChannel`。Spec Maturity 为 ``beta``，
表示尚不能从 Deterministic Contract 外推 Live Gate 已通过。
"""

from __future__ import annotations

from codeflow.channels.contract import Capabilities, ChannelSpec


def _make(config):
    from codeflow.channels.adapters.qq.channel import QQChannel

    return QQChannel(config)


SPEC = ChannelSpec(
    display_name="QQ",
    factory=_make,
    capabilities=Capabilities(),
    maturity="beta",
)
