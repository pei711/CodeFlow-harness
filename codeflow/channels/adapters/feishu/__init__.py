"""迁移到 Capability Contract 的 Feishu/Lark Channel Adapter。

Package 刻意 **不** Re-export ``FeishuChannel``，否则 Package Import 会加载 ``lark_oapi``，破坏 Cheap Spec
Discovery；``registry.discover_specs`` 只 Import ``feishu.spec``。Runtime 通过 ``spec.SPEC.factory`` 构造，
明确需要实现时也可直接 Import ``.channel``。
"""
