"""Channel Adapters 共用的 Thin Construction Base。

``ChannelBase`` 只消除 Config/Running/Intake 初始化 Boilerplate；Runtime 真正依赖的是
:mod:`codeflow.channels.contract` 的 ``Channel`` Protocol，而不是继承关系。Inbound Normalization 属于 `Intake`，
Transcription 属于 Helper，Platform Send/Start/Stop 属于各 Adapter。
"""

from __future__ import annotations

from typing import Any

from codeflow.channels.contract import Capabilities
from codeflow.channels.intake import Intake


class ChannelBase:
    """Channel Adapter 的 Shared Plumbing Base，不拥有平台行为。

    Inbound/Transcribe 位于 ``Intake`` 与 ``transcribe`` Helper；本类只保存 Config、Running State、Injected
    Intake。系统依赖 :mod:`codeflow.channels.contract` ``Channel`` Protocol，继承这里只为 Reuse，Conformance
    仍按 Protocol。Subclass 提供 ``name``/``display_name`` 与 ``start``/``stop``/``send``，可 Override
    ``is_allowed``；``__init__`` 创建的 Injected Intake 自动调用该 Override。
    """

    name: str = ""
    display_name: str = ""
    capabilities: Capabilities = Capabilities()
    transcription_api_key: str = ""  # 由 ChannelManager 设置

    def __init__(self, config: Any):
        self.config = config
        self._running = False
        self.intake = Intake(self.name, config, allow_check=self.is_allowed)

    @property
    def is_running(self) -> bool:
        return self._running

    def is_allowed(self, sender_id: str) -> bool:
        """执行 Deny-by-default Allowlist Check。

        Empty 表示 Deny All，``"*"`` 表示 Allow All，其他值与 Stringified Sender ID Exact Match。Subclass 可
        Override Bespoke Matching，结果会流入 Intake；通过此检查不等于平台签名/业务权限已验证。
        """
        from codeflow.auth.allowlist import is_allowed as _check

        return _check(self.name, sender_id, getattr(self.config, "allow_from", None))
