"""迁移到 Capability Contract 的 WeCom Channel Adapter。

Package 刻意 **不** Re-export ``WecomChannel``，否则 Import 会加载 ``wecom_aibot_sdk`` 并破坏 Cheap Spec
Discovery；``registry.discover_specs`` 只 Import ``wecom.spec``。Runtime 通过 ``spec.SPEC.factory`` 构造，
或按需直接 Import ``.channel``。
"""
