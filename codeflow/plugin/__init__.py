"""CodeFlow Plugin Foundation。

PG-1 引入 Manifest Schema + Plugin Context，Registry 与 Discovery 在 PG-2/PG-3 接入。最早的 Public
Contribution Point 是 ``memory_backends``，当前 Schema 也包含 Tools，并保持 Forward-compatible，使
未来新增 Contribution Types 不必破坏 Existing Manifests。

Two Load-bearing Principles：

1. **Manifests 是 Pure Data。** ``PluginManifest.from_toml_path`` 只读取 TOML；只有 Registry 真正要求
   Factory Build Backend 时才 Import Plugin Code。这样 Startup 保持 Deterministic 与 Audit-friendly，
   “发现一个插件”不会自动执行其代码。
2. **Factories 用 ``module.path:callable`` Strings 引用。** Registry Lazy Import Module 并解析 Callable；
   Manifest Parsing 永不触发 Plugin Package 的 Import-time Side Effects。

这些原则构成启动 Trust Boundary：Admission/Discovery Evidence 与最终执行权限必须分开判断。
"""

from __future__ import annotations

from codeflow.plugin.bootstrap import assemble_plugin_registry
from codeflow.plugin.context import PluginContext, ServiceLocator
from codeflow.plugin.discover import (
    DiscoveredPlugin,
    PluginCompatibilityError,
    PluginDiscovery,
    PluginIdentityError,
    Source,
)
from codeflow.plugin.manifest import (
    Contributes,
    MemoryBackendContribution,
    PluginManifest,
    ToolContribution,
)
from codeflow.plugin.registry import (
    MemoryBackendFactory,
    PluginConflictError,
    PluginError,
    PluginFactoryImportError,
    PluginNotFoundError,
    PluginRegistry,
    ToolFactory,
)

__all__ = [
    "Contributes",
    "DiscoveredPlugin",
    "assemble_plugin_registry",
    "MemoryBackendContribution",
    "MemoryBackendFactory",
    "PluginConflictError",
    "PluginCompatibilityError",
    "PluginContext",
    "PluginDiscovery",
    "PluginError",
    "PluginFactoryImportError",
    "PluginIdentityError",
    "PluginManifest",
    "PluginNotFoundError",
    "PluginRegistry",
    "ServiceLocator",
    "Source",
    "ToolContribution",
    "ToolFactory",
]
