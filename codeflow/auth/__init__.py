"""Authentication 与 Authorization Primitives 的总览入口。

Package 由 Three Thin Modules 组成：

- ``allowlist``：集中 Channel Sender Allowlist 判断，提供 Canonical
  ``is_allowed(channel, sender_id, allow_list)`` Helper；``channels/base.py`` 把 Deny-by-default Check
  委托到这里，不再内联另一份规则。
- ``capability_token``：Multi-agent Capability Token Design 的 Placeholder。目前只定义 Dataclass 与
  Roundtrip Helpers，**尚无 Enforcement Code**，因此生成 Token 不代表权限已经被运行时强制执行。
- ``managed_settings``：Company Deployment 中“此 Config Field 被锁定”机制的 Placeholder。目前只
  返回普通 ``ManagedSettings`` Carrier；真正 Locking Enforcement 要等存在具体 Operator Need 后在
  Upstream 实现。

External Callers 应直接从各 Sub-package Path 导入，以明确自己依赖的是现有安全检查还是尚未落地的
数据结构，避免把 Placeholder 当成已经建立的 Trust Boundary。
"""

from codeflow.auth.allowlist import is_allowed
from codeflow.auth.capability_token import CapabilityToken, issue_token, verify_token
from codeflow.auth.managed_settings import ManagedSettings

__all__ = [
    "is_allowed",
    "CapabilityToken",
    "issue_token",
    "verify_token",
    "ManagedSettings",
]
