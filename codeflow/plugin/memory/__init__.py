"""Bundled Memory-backend Plugins 的 Package Namespace。

这里的 Subpackages 由 :class:`codeflow.plugin.PluginDiscovery` 通过各自 ``codeflow-plugin.toml`` Manifest 作为
Bundled Source 发现。每个 Package 的 ``__init__`` 必须保持 Empty/Cheap：Resource Resolution 在 Discovery
期间可能 Import Package，Heavy Import 会破坏 Manifest-only Discovery Guarantee，并把 Plugin Side Effect
提前到 Admission 之前。

发现 Bundled Package 只证明 Manifest 可读取；Backend Factory 仍应在 Registry 真正选中时 Lazy Import。
"""
