"""Chat-channel Adapter 必须满足并声明的 Channel Contract。

Channel 实现 :class:`Channel` Protocol 的 ``start``/``stop``/``send``，并声明 :class:`Capabilities`；Optional
Behavior 通过独立 ``Supports*`` Protocol Opt In。每个 Package 导出 Lightweight :class:`ChannelSpec`，其
``factory`` 延迟 Heavy SDK Import，Registry 只消费 Descriptor。

设计采用 Composition over Inheritance：Adapter Structural Conformance，并注入所需 Framework Services，
如 :mod:`.intake`、Transcription。Capabilities Declaration、Protocol Implementation 与 Live Evidence 必须
一致，不能从某个 Method 存在就推断成熟度。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

# Capabilities 和 SupportsStreaming 定义在 spine.delivery（其消费者是
# delivery hub）；此处重新导出，让 channel 始终从同一入口导入。
from codeflow.spine.delivery import Capabilities, SupportsStreaming

if TYPE_CHECKING:
    from codeflow.channels.intake import Intake


@runtime_checkable
class Channel(Protocol):
    """每个 Channel 必须满足的 Minimal Runtime Contract。

    `name`/`capabilities`/`intake` 描述身份与 Inbound Boundary；Async `start`/`stop` 管理连接生命周期，
    `send` 交付 Text/Optional Media。Runtime-checkable 只验证 Surface，不执行平台健康检查。
    """

    name: str
    capabilities: Capabilities
    intake: Intake

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def send(self, chat_id: str, content: str, media: list[str] | None = None) -> None: ...


@runtime_checkable
class SupportsLogin(Protocol):
    """Opt-in Interactive QR/Scan Login Protocol。

    CLI 在 ``start`` 前按需运行一次；`force` 可要求重新鉴权。返回 Bool 是 Login Flow Result，不等于 Channel
    Long Connection 已启动。
    """

    async def login(self, force: bool = False) -> bool: ...


Maturity = Literal["beta", "live-gated"]


@dataclass(frozen=True)
class ChannelSpec:
    """Channel Package 以 ``SPEC`` 导出的 Declarative Descriptor。

    ``factory`` 延迟 Heavy SDK Import，使 Listing/Onboarding/Login Routing 的 Spec Collection 保持 Cheap。
    Descriptor 只携带其他位置无法推导的 Display Name、Factory、Capabilities、Maturity；Registry Key/Channel
    Name 来自 Package，CLI 由 Capabilities + Config Schema 推导 Dependency/Setup Guidance。

    ``maturity`` 表示 Adapter Evidence Level，而非 Code Quality：``beta`` 只有 Deterministic Contract V-C0 与
    Security V-S0 Bundles；``live-gated`` 还通过 Live Channel Gate。它是 CLI/Doctor/Onboarding Single Source，
    任何 Surface 都不能 Claim 高于 Spec 的 Level。
    """

    display_name: str
    factory: Callable[[Any], Channel]  # 配置映射到渠道实例
    capabilities: Capabilities = field(default_factory=Capabilities)
    maturity: Maturity = "beta"


# 每个 capability 标志必须与对应的 opt-in protocol 一致。新增 capability
# 时增加一行即可；下方检查会同时覆盖两个方向。
_CAP_PROTOCOLS: tuple[tuple[str, type], ...] = (
    ("interactive_login", SupportsLogin),
    ("streaming", SupportsStreaming),
)


def capability_violations(channel: object, caps: Capabilities | None = None) -> list[str]:
    """返回 Declared Capabilities 与 Implemented Protocols 的 Mismatches。

    Channel 声明 Capability 必须实现对应 ``Supports*`` Protocol，反之亦然。Empty List 表示一致，供
    Per-channel Capability-proof Tests 使用；一致只证明声明与 Surface 对齐，不证明 Live Behavior。
    """
    caps = caps if caps is not None else getattr(channel, "capabilities", Capabilities())
    out: list[str] = []
    for flag, proto in _CAP_PROTOCOLS:
        declared = getattr(caps, flag)
        implemented = isinstance(channel, proto)
        if declared and not implemented:
            out.append(f"declares {flag} but does not implement {proto.__name__}")
        elif implemented and not declared:
            out.append(f"implements {proto.__name__} but does not declare {flag}")
    return out
