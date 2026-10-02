"""Builtin Channel Adapters 的 Auto-discovery Namespace。

每个 Subpackage 提供导出 :class:`~codeflow.channels.contract.ChannelSpec` 的 ``spec.py``，由
:mod:`codeflow.channels.registry` 自动发现，无需 Manual Registration。Spec Factory 延迟 Heavy SDK Import；
Discovery 只扫描 One Level，不枚举 Internal Helper Modules。

发现 Spec 只证明 Metadata 可导入，Factory Construction/SDK Login/Channel Start 仍是后续阶段。
"""
