"""CLI assembly helper for the plugin / memory-backend stack.

Two functions that bridge the gap between the runtime config (user-facing
settings under ``plugins`` / ``memory``) and the runtime objects
AgentLoop expects (a ready-to-use :class:`MemoryBackend` instance):

- :func:`build_plugin_registry` — discover all trusted plugins
  (bundled + user-level + installed entry points), filter
  by ``config.plugins.disabled``, return an activated registry.
- :func:`maybe_build_memory_backend` — resolve ``config.memory.backend``
  to a concrete :class:`MemoryBackend` instance via the registry, or
  return ``None`` when no backend is selected / the requested
  contribution isn't available.

An explicit ``memory.backend`` selection is fail-closed: activation,
resolution, and construction errors remain visible to the Runtime host.
Only ``memory.backend = null`` disables Memory.

Lifecycle (``backend.start()`` / ``backend.stop()``) belongs to the concrete
``RuntimeAssembly``. These helpers only construct plugin contributions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeflow.plugin import (
    DiscoveredPlugin,
    PluginDiscovery,
    PluginIdentityError,
    PluginNotFoundError,
    PluginRegistry,
    ServiceLocator,
)
from codeflow.product import get_product_home

if TYPE_CHECKING:
    from codeflow.config.codeflow import CodeFlowConfig
    from codeflow.memory_engine import MemoryBackend

logger = logging.getLogger(__name__)

_MYNA_PLUGIN_ID = "myna-memory"
_MYNA_BACKEND = "myna"
_MYNA_FACTORY = "myna.integrations.codeflow:make_backend"


@dataclass(frozen=True)
class MemoryBackendStatus:
    backend: str | None
    state: str
    plugin_id: str | None = None
    plugin_version: str | None = None
    error: str | None = None


class MynaSetupError(RuntimeError):
    """Myna cannot start until the workspace has explicit operator setup."""


class _MynaBackendGuard:
    def __init__(self, backend: Any) -> None:
        self._backend = backend

    async def start(self) -> None:
        try:
            await self._backend.start()
        except Exception as exc:
            if getattr(exc, "code", None) in {"configuration_invalid", "myna_not_initialized"}:
                raise MynaSetupError(
                    "Myna is not initialized or its configuration is invalid; run 'myna init' in the CodeFlow "
                    "workspace, then run 'myna doctor --live'"
                ) from exc
            raise

    async def stop(self) -> None:
        await self._backend.stop()

    async def recall(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        top_k: int,
    ) -> list:
        return await self._backend.recall(
            query,
            user_id=user_id,
            agent_id=agent_id,
            top_k=top_k,
        )

    async def store(self, session_id: str, messages: list[dict]) -> None:
        await self._backend.store(session_id, messages)

    async def feedback(self, signals: dict[str, Any]) -> None:
        await self._backend.feedback(signals)


def plugin_discovery_sources() -> dict:
    """Resolve the trusted discovery sources scanned by CodeFlow hosts.

    Shared by :func:`build_plugin_registry` (live boot) and the
    ``codeflow plugins`` CLI command so both see the same set:

    - bundled - ``codeflow/plugin/memory/`` inside the package;
    - user - operator-managed ``~/.codeflow/plugins/``;
    - entry points - installed distributions in the ``codeflow.plugins`` group.

    ``project_dir`` remains explicit and disabled. A checkout must not gain an
    executable startup hook merely by containing ``.codeflow/plugins``.
    """
    import codeflow

    return {
        "bundled_dir": Path(codeflow.__path__[0]) / "plugin" / "memory",
        "user_dir": get_product_home() / "plugins",
        "project_dir": None,
        "entry_points_group": "codeflow.plugins",
    }


def build_plugin_registry(
    config: "CodeFlowConfig",
) -> PluginRegistry:
    """Discover + activate every installed plugin admitted by ``config``.

    Reads ``config.plugins.disabled`` and forwards it to
    :func:`assemble_plugin_registry`. Activation errors propagate so a
    configured backend cannot silently disappear.

    Discovery spans three trusted sources (priority bundled > user >
    entry_points):

    - **bundled** — ``codeflow/plugin/memory/<id>/`` shipped inside the
      CodeFlow package.
    - **user** - ``~/.codeflow/plugins/<id>/`` drop-in directories.
    - **entry_points** - the ``codeflow.plugins`` group, where
      third-party pip-installed plugins register their factories.

    Low-level callers can still opt into project manifests explicitly with
    :class:`PluginDiscovery(project_dir=...)`.
    """
    disabled = frozenset(config.plugins.disabled)
    discovered = PluginDiscovery(**plugin_discovery_sources()).discover()
    _validate_discovered_myna_identity(discovered, disabled=disabled)
    registry = PluginRegistry()
    registry.activate(discovered, disabled=disabled)
    return registry


def maybe_build_memory_backend(
    workspace: Path,
    config: "CodeFlowConfig",
    *,
    registry: PluginRegistry | None = None,
) -> "MemoryBackend | None":
    """Construct the configured memory backend, if any.

    Resolution order:

    1. If ``config.memory.backend`` is ``None``, return ``None``
       immediately — user explicitly disabled the plugin path.
    2. Look up the backend factory in the (possibly host-supplied)
       :class:`PluginRegistry`. Missing configured contributions raise.
    3. Resolve the per-plugin config slice from
       ``config.plugins.config`` — first by plugin id, then by backend
       contribution name as a fallback.

    The returned backend has **not** been ``await``-started. Runtime Assembly
    owns its start / stop lifecycle, while each host decides when startup fits
    its interaction policy.
    """
    name = config.memory.backend
    if name is None:
        return None
    if name in {"codecairn", "everos"}:
        raise PluginNotFoundError(
            f"memory.backend={name!r} is no longer supported; install and initialize Myna, then set "
            "memory.backend to 'myna', or set it to null. Existing memory data is left untouched"
        )
    if registry is None:
        registry = build_plugin_registry(config)
    if name == _MYNA_BACKEND:
        _validate_myna_identity(registry)
    plugin_slice = _resolve_plugin_config_slice(registry, config, name)
    services = ServiceLocator(workspace=workspace)
    backend = registry.build_memory_backend(
        name,
        config=plugin_slice,
        services=services,
    )
    return _MynaBackendGuard(backend) if name == _MYNA_BACKEND else backend


def inspect_memory_backend(config: "CodeFlowConfig") -> MemoryBackendStatus:
    """Inspect configured plugin metadata without constructing or starting it."""
    name = config.memory.backend
    if name is None:
        return MemoryBackendStatus(backend=None, state="disabled")
    if name in {"codecairn", "everos"}:
        return MemoryBackendStatus(
            backend=name,
            state="error",
            error=(
                f"memory.backend={name!r} is no longer supported; install and initialize Myna, then set "
                "memory.backend to 'myna', or set it to null"
            ),
        )
    try:
        registry = build_plugin_registry(config)
        if name == _MYNA_BACKEND:
            _validate_myna_identity(registry)
        plugin_id = _plugin_id_for_backend(registry, name)
        if plugin_id is None:
            raise PluginNotFoundError(
                f"no memory_backend named {name!r} (registered: {registry.memory_backend_names()})"
            )
        manifest = registry.manifest_for(plugin_id)
        return MemoryBackendStatus(
            backend=name,
            state="available",
            plugin_id=plugin_id,
            plugin_version=manifest.version if manifest is not None else None,
        )
    except Exception as exc:
        return MemoryBackendStatus(backend=name, state="error", error=str(exc))


def _validate_myna_identity(registry: PluginRegistry) -> None:
    try:
        identity = registry.memory_backend_identity(_MYNA_BACKEND)
    except PluginNotFoundError as exc:
        raise PluginNotFoundError(
            "configured Myna plugin is unavailable; install the myna-memory distribution and run "
            "'myna init' in the CodeFlow workspace, or set memory.backend to null"
        ) from exc
    if identity != (_MYNA_PLUGIN_ID, _MYNA_FACTORY):
        raise PluginIdentityError(
            "Myna plugin manifest identity is invalid; reinstall the official myna-memory distribution "
            "before starting CodeFlow"
        )


def _validate_discovered_myna_identity(
    discovered: list[DiscoveredPlugin],
    *,
    disabled: frozenset[str],
) -> None:
    expected = [(_MYNA_BACKEND, _MYNA_FACTORY)]
    for plugin in discovered:
        manifest = plugin.manifest
        if manifest.id in disabled:
            continue
        contributions = [
            (contribution.name, contribution.factory) for contribution in manifest.contributes.memory_backends
        ]
        if manifest.id != _MYNA_PLUGIN_ID and not any(name == _MYNA_BACKEND for name, _ in contributions):
            continue
        if manifest.id != _MYNA_PLUGIN_ID or contributions != expected:
            raise PluginIdentityError(
                "Myna plugin manifest identity is invalid; reinstall the official myna-memory distribution "
                "before starting CodeFlow"
            )


def build_plugin_tools(
    workspace: Path,
    config: "CodeFlowConfig",
    *,
    registry: PluginRegistry | None = None,
) -> list:
    """Construct every plugin-contributed tool admitted by ``config``.

    Mirrors :func:`maybe_build_memory_backend` but for the ``tools``
    contribution point: walks the activated registry's tool names,
    resolves each owning plugin's config slice, and builds the tool via
    :meth:`PluginRegistry.build_tool`. Lenient by design — a single
    tool's construction failure is logged and skipped so one bad plugin
    can't keep the agent from booting. A factory may also return ``None``
    to deliberately decline contribution (e.g. an optional dependency is
    absent); that's skipped quietly, not treated as a failure. The host
    registers the returned tools into the agent's :class:`ToolRegistry`.

    Returns an empty list when no plugin contributes a tool.
    """
    if registry is None:
        registry = build_plugin_registry(config)
    names = registry.tool_names()
    if not names:
        return []
    services = ServiceLocator(workspace=workspace)
    slices = config.plugins.config
    tools = []
    for name in names:
        plugin_id = registry.tool_plugin_id(name)
        plugin_slice = (plugin_id and slices.get(plugin_id)) or slices.get(name) or {}
        try:
            tool = registry.build_tool(
                name,
                config=plugin_slice,
                services=services,
            )
        except Exception as e:
            logger.warning(
                "plugin tool %r factory raised at construction (%s); skipping it.",
                name,
                e,
            )
            continue
            # 工厂可返回 None，在运行时拒绝提供能力（例如未安装可选依赖）。这是正常退出，
            # 不是失败；直接跳过且不警告。
        if tool is None:
            logger.debug(
                "plugin tool %r factory opted out (returned None); skipping it.",
                name,
            )
            continue
        tools.append(tool)
    return tools


def _resolve_plugin_config_slice(
    registry: PluginRegistry,
    config: "CodeFlowConfig",
    backend_name: str,
) -> dict:
    """Pick the right ``config.plugins.config[...]`` entry for a backend.

    Tries two keys, in order:

    1. The **plugin id** that contributes ``backend_name`` (canonical,
       e.g. ``"myna-memory"`` - comes from the manifest's
       ``[plugin] id`` field).
    2. The **backend contribution name** itself
       (e.g. ``"myna"`` - friendlier for handwritten config files).

    Returns an empty dict when neither key is present, so the plugin
    factory receives a deterministic shape and applies its own
    defaults.
    """
    slices = config.plugins.config
    plugin_id = _plugin_id_for_backend(registry, backend_name)
    if plugin_id is not None and plugin_id in slices:
        return slices[plugin_id]
    return slices.get(backend_name, {})


def _plugin_id_for_backend(
    registry: PluginRegistry,
    backend_name: str,
) -> str | None:
    """Reverse-lookup the plugin id that contributes ``backend_name``.

    Returns ``None`` when no activated plugin contributes the named
    backend — the caller (config resolver) treats that as "fall
    through to the contribution-name key".
    """
    for plugin_id in registry.activated_ids():
        mf = registry.manifest_for(plugin_id)
        if mf is None:
            continue
        for contribution in mf.contributes.memory_backends:
            if contribution.name == backend_name:
                return plugin_id
    return None


__all__ = [
    "build_plugin_registry",
    "build_plugin_tools",
    "inspect_memory_backend",
    "MemoryBackendStatus",
    "maybe_build_memory_backend",
]
