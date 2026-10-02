"""Managed Settings，是 Company-deployment Config Locks 的 Placeholder。

`ManagedSettings` 用一个 Thin Dataclass 描述 Operator Policy 锁定哪些 Config Fields。当前它仍然
**Unconsumed**，Config Loading 尚未查询该结构；现在先定义 Shape，只是为未来工作建立 Stable Seam。

Typical Post-MVP Use Case：Managed CodeFlow Deployment 把 ``providers.openrouter.api_key`` 锁定到公司
Shared Key，并把 ``sandbox.backend`` 锁定为 ``"boxlite"``。未来 Loader 应检查
``ManagedSettings.locked_fields``，拒绝 User-side 对集合内字段的 Overrides。现阶段创建此对象并不会
产生上述 Enforcement。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ManagedSettings:
    """描述哪些 Config Paths 被锁定的 Operator Policy Carrier。

    ``locked_fields`` 中每一项都是 Dot-separated Config Path，例如
    ``providers.openrouter.api_key``；`description` 可说明策略来源或目的。`is_locked` 只做 Exact
    Membership Check，不实现父路径继承或通配符。Loader 负责 Honor Lock，这个 Frozen Dataclass 仅
    Carry Policy，不能单独阻止配置覆盖。
    """

    locked_fields: frozenset[str] = field(default_factory=frozenset)
    description: str = ""

    def is_locked(self, field_path: str) -> bool:
        return field_path in self.locked_fields


__all__ = ["ManagedSettings"]
