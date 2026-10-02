"""Plugin Registry：Admit Manifests，并在真正使用时 Resolve Factories。

职责刻意拆成 Two Phases：

1. **Activation**（:meth:`activate`）依据 Config Admit Manifest、验证 Contribution Conflicts、记录尚未
   Resolve 的 Factory Refs；此阶段不 Import Plugin Python，也不扩展 ``sys.path``。
2. **Build** 只 Resolve/Cache Host 实际构造的 Backend 或 Tool Factory，并用 Fresh `PluginContext` 调用。

Across-manifest Name Conflict 会抛出 :class:`PluginConflictError`，Host 视为 Startup Failure。Discovery
已按 ID Deduplicate *Plugins*；Registry 再按 Slot Name Deduplicate *Contributions*。Activation Success 与
Factory Import/Construction Success 是独立证据边界。
"""

from __future__ import annotations

import importlib
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codeflow.plugin.discover import DiscoveredPlugin, Source
from codeflow.plugin.manifest import PluginManifest
from codeflow.tracing import semconv, trace

logger = logging.getLogger(__name__)


# 记忆后端工厂是接收 PluginContext 并返回 MemoryBackend 实现的可调用对象。
# MemoryBackend 在 MB-1 落地；在此之前返回值标为 Any，使 PG 可以独立编译。
MemoryBackendFactory = Callable[[Any], Any]

# 工具工厂接收 PluginContext 并返回单个工具。
# ``codeflow.agent.tools.base.Tool``。此处标为 Any，使插件
# 保持本层轻量导入，不依赖 agent 包。
ToolFactory = Callable[[Any], Any]


class PluginError(Exception):
    """Plugin-system Errors 的 Base Class。

    CLI / Host Startup 可捕获一个类型并渲染 Unified Diagnostic Banner，同时具体 Subclass 保留 Conflict、
    Import、Not-found 差异。
    """


class PluginConflictError(PluginError):
    """Two Activated Plugins 向同一 Slot Contribution 了相同 Name。

    Registry Fail Closed，避免 Contribution Winner 取决于偶然 Activation Order。
    """


class PluginFactoryImportError(PluginError):
    """Manifest 指向的 ``module.path:callable`` 无法 Import、Resolve 或不是 Callable。

    该异常只在 First Factory Use 暴露，因为 Activation 保持 Pure-data。
    """


class PluginNotFoundError(PluginError):
    """用户请求的 Backend/Tool Name 没有 Activated Plugin Contribution。

    错误通常同时列出 Registered Names，帮助区分拼写错误、Disabled Plugin 与 Discovery Missing。
    """


@dataclass(frozen=True)
class _ActivatedFactory:
    """Unresolved Factory Reference 与 Diagnostics Provenance。

    Record 保存 Plugin ID、Contribution Name、Factory Ref、Source 与 Manifest Location；不持有 Imported
    Module 或 Constructed Object。
    """

    plugin_id: str
    name: str
    factory_ref: str
    source: Source
    location: Path | None


class PluginRegistry:
    """Activated Contribution Factories 的 Single Registration Center。

    Registry 拥有 Activated Manifests、Memory Backend/Tool Slot Maps 与 Lazy Resolved Factory Cache。
    生命周期通常覆盖 Host Startup 到 Shutdown；它不拥有由 Factory 返回的 Backend/Tool Lifecycle。
    """

    def __init__(self) -> None:
        self._manifests: dict[str, PluginManifest] = {}
        self._memory_backends: dict[str, _ActivatedFactory] = {}
        self._tools: dict[str, _ActivatedFactory] = {}
        self._resolved_factories: dict[tuple[str, str], MemoryBackendFactory] = {}

    # ── 激活 ─────────────────────────────────────────────────────

    def activate(
        self,
        discovered: list[DiscoveredPlugin],
        *,
        disabled: frozenset[str] = frozenset(),
    ) -> None:
        """在不 Import Plugin Python Code 的情况下 Admit Manifests。

        Plugin ID 不在 ``disabled`` **AND** ``enabled_by_default`` 为 True 才 Admit；若前者命中 User
        Opt-out，**OR** 后者为 False，当前实现 Skip 并记录
        Explicit Opt-in 尚待支持。满足条件的 Manifest 进入 `_activate_one`，Contribution Conflict 立即
        抛错。PG-2/PG-3 演进说明的核心仍是配置 Admission 与 Code Import 分离。
        """
        for d in discovered:
            mf = d.manifest
            if mf.id in disabled:
                logger.info("plugin %s disabled by user config", mf.id)
                continue
            if not mf.enabled_by_default:
                logger.info(
                    "plugin %s not enabled by default; skipping (use explicit opt-in once supported)",
                    mf.id,
                )
                continue
            self._activate_one(mf, source=d.source, location=d.location)

    def _activate_one(
        self,
        mf: PluginManifest,
        *,
        source: Source,
        location: Path | None,
    ) -> None:
        if mf.id in self._manifests:
            # Discovery 理应已经去重；此处属于防御性检查。
            raise PluginConflictError(
                f"plugin id {mf.id!r} activated twice",
            )
        pending_memory_backends: dict[str, _ActivatedFactory] = {}
        pending_tools: dict[str, _ActivatedFactory] = {}

        for contribution in mf.contributes.memory_backends:
            if contribution.name in self._memory_backends:
                prev = self._memory_backends[contribution.name]
                raise PluginConflictError(
                    f"memory_backend {contribution.name!r} contributed by both {prev.plugin_id!r} and {mf.id!r}",
                )
            pending_memory_backends[contribution.name] = _ActivatedFactory(
                plugin_id=mf.id,
                name=contribution.name,
                factory_ref=contribution.factory,
                source=source,
                location=location,
            )

        for tool in mf.contributes.tools:
            if tool.name in self._tools:
                prev = self._tools[tool.name]
                raise PluginConflictError(
                    f"tool {tool.name!r} contributed by both {prev.plugin_id!r} and {mf.id!r}",
                )
            pending_tools[tool.name] = _ActivatedFactory(
                plugin_id=mf.id,
                name=tool.name,
                factory_ref=tool.factory,
                source=source,
                location=location,
            )

        self._manifests[mf.id] = mf
        self._memory_backends.update(pending_memory_backends)
        self._tools.update(pending_tools)
        for contribution in mf.contributes.memory_backends:
            logger.debug("registered memory_backend %s from %s", contribution.name, mf.id)
        for tool in mf.contributes.tools:
            logger.debug("registered tool %s from %s", tool.name, mf.id)

    @staticmethod
    def _ensure_importable(source: Source, location: Path | None) -> None:
        """在 Import 前暴露 File-based Plugin Directory。

        USER 与 Explicit PROJECT Plugins 可把 Code 放在 Manifest 旁；只有 Caller 真正 Resolve Contribution
        Factory 时才把目录 Append 到 ``sys.path``。Host Boundary 已禁用 Automatic Project Discovery。
        Bundled/Entry-point Source 使用正常 Import Path，不在此修改。
        """
        if source not in (Source.USER, Source.PROJECT) or location is None:
            return
        plugin_dir = str(location.parent)
        if plugin_dir not in sys.path:
            sys.path.append(plugin_dir)

    @staticmethod
    def _resolve_factory(plugin_id: str, ref: str) -> MemoryBackendFactory:
        """Import ``module``，并从中取得 ``callable``。

        Manifest Validation 已强制 ``module.path:callable`` Shape，因此这里只 Split、`import_module`、
        `getattr` 与 Callable Check。Import Error、Missing Attribute、Non-callable 都转换为带 Plugin ID 的
        `PluginFactoryImportError`；成功返回是 Factory Object，尚未调用。
        """
        module_path, attr = ref.split(":", 1)
        try:
            mod = importlib.import_module(module_path)
        except Exception as e:
            raise PluginFactoryImportError(
                f"plugin {plugin_id!r}: importing {module_path!r} failed: {e}",
            ) from e
        try:
            obj = getattr(mod, attr)
        except AttributeError as e:
            raise PluginFactoryImportError(
                f"plugin {plugin_id!r}: {module_path!r} has no attribute {attr!r}",
            ) from e
        if not callable(obj):
            raise PluginFactoryImportError(
                f"plugin {plugin_id!r}: {ref} resolved to a non-callable {type(obj).__name__}",
            )
        return obj  # type: ignore[return-value]

    def _get_factory(
        self,
        kind: str,
        entry: _ActivatedFactory,
    ) -> MemoryBackendFactory:
        key = (kind, entry.name)
        cached = self._resolved_factories.get(key)
        if cached is not None:
            return cached
        self._ensure_importable(entry.source, entry.location)
        factory = self._resolve_factory(entry.plugin_id, entry.factory_ref)
        self._resolved_factories[key] = factory
        return factory

    # ── 内省 ─────────────────────────────────────────────────────

    def activated_ids(self) -> list[str]:
        """返回 Stable-ordered Activated Plugin IDs；不包含 Disabled/Skipped Manifests。"""
        return sorted(self._manifests)

    def memory_backend_names(self) -> list[str]:
        """返回 Stable-ordered Registered Memory-backend Names；Factories 可能尚未 Import。"""
        return sorted(self._memory_backends)

    def get_memory_backend_factory(self, name: str) -> MemoryBackendFactory:
        """First Use 时 Resolve 并 Cache ``name`` 对应 Backend Factory。

        Name Missing 抛出 `PluginNotFoundError`；Import/Resolution Failure 抛出 `PluginFactoryImportError`。
        返回 Factory 不调用它，也不启动 Backend。
        """
        try:
            entry = self._memory_backends[name]
        except KeyError as e:
            raise PluginNotFoundError(
                f"no memory_backend named {name!r} (registered: {self.memory_backend_names()})",
            ) from e
        return self._get_factory("memory_backend", entry)

    def memory_backend_identity(self, name: str) -> tuple[str, str]:
        """返回 ``(owning_plugin_id, declared_factory_reference)``。

        只读 Activated Record，不触发 Import；Name Missing 抛出 `PluginNotFoundError`。
        """
        try:
            entry = self._memory_backends[name]
        except KeyError as e:
            raise PluginNotFoundError(
                f"no memory_backend named {name!r} (registered: {self.memory_backend_names()})",
            ) from e
        return entry.plugin_id, entry.factory_ref

    def tool_names(self) -> list[str]:
        """返回 Stable-ordered Registered Plugin-tool Names；不包含 Host Builtin Tools。"""
        return sorted(self._tools)

    def tool_plugin_id(self, name: str) -> str | None:
        """返回 Contribution Tool ``name`` 的 Plugin ID；Unknown Name 返回 `None`，且不 Import Factory。"""
        entry = self._tools.get(name)
        return entry.plugin_id if entry is not None else None

    def get_tool_factory(self, name: str) -> ToolFactory:
        """First Use 时 Resolve 并 Cache ``name`` 对应 Tool Factory。

        Failure Semantics 与 `get_memory_backend_factory` 相同；返回值尚未构造 Tool。
        """
        try:
            entry = self._tools[name]
        except KeyError as e:
            raise PluginNotFoundError(
                f"no tool named {name!r} (registered: {self.tool_names()})",
            ) from e
        return self._get_factory("tool", entry)

    def manifest_for(self, plugin_id: str) -> PluginManifest | None:
        """返回 Activated Plugin 的 Manifest；Unknown/Skipped ID 返回 `None`。

        Manifest 是 Frozen Pure Data，可用于 Diagnostics，不触发 Factory Resolution。
        """
        return self._manifests.get(plugin_id)

    # ── 构建（PG-3 入口）─────────────────────────────────────────

    @trace.instrument("plugin.load", extract=semconv.plugin_load("memory_backend"))
    def build_memory_backend(
        self,
        name: str,
        *,
        config: dict[str, Any],
        services: "ServiceLocator",
        logger: logging.Logger | None = None,
    ) -> Any:
        """Resolve Named Factory，并用 Fresh ``PluginContext`` 构造 Memory Backend。

        Construction 是 Synchronous；需要 Async Setup 的 Factory 返回 Backend，Host 稍后 Await 其
        ``start()``。Config、Service Locator 与 Plugin-bound Logger 进入 Context。Factory Exception 原样
        Propagate，让 Host 看到 Real Cause，而不是 Wrapped Error。返回对象尚未 Start。
        """
        from codeflow.plugin.context import PluginContext  # 局部导入以规避循环依赖

        factory = self.get_memory_backend_factory(name)
        ctx = PluginContext(
            config=config,
            services=services,
            logger=logger or logging.getLogger(f"codeflow.plugin.{name}"),
        )
        return factory(ctx)

    @trace.instrument("plugin.load", extract=semconv.plugin_load("tool"))
    def build_tool(
        self,
        name: str,
        *,
        config: dict[str, Any],
        services: "ServiceLocator",
        logger: logging.Logger | None = None,
    ) -> Any:
        """Resolve Named Tool Factory，用 Fresh ``PluginContext`` 返回 Constructed ``Tool``。

        与 :meth:`build_memory_backend` Symmetric：Synchronous Construction，Exceptions Propagate。Host 之后
        把返回 Tool 注册进 Agent :class:`ToolRegistry`；Factory 返回成功不表示 Registration 或 Tool
        Execution 已成功。
        """
        from codeflow.plugin.context import PluginContext  # 局部导入以规避循环依赖

        factory = self.get_tool_factory(name)
        ctx = PluginContext(
            config=config,
            services=services,
            logger=logger or logging.getLogger(f"codeflow.plugin.{name}"),
        )
        return factory(ctx)


# 为上方类型提示提供前向导入。放在模块末尾，既延后导入成本，
# 也避免模块加载时触发循环依赖（registry 会被
# 在上下文就绪前调用 __init__）。
from codeflow.plugin.context import ServiceLocator  # noqa: E402

__all__ = [
    "MemoryBackendFactory",
    "PluginConflictError",
    "PluginError",
    "PluginFactoryImportError",
    "PluginNotFoundError",
    "PluginRegistry",
    "ToolFactory",
]
