"""One-call Plugin Bootstrap Helper。

它把 :class:`PluginDiscovery` 与 :class:`PluginRegistry` 串起来，使 CLI / AgentLoop Construction 不必
重复 Two-step Dance。保持为 Free Function 而非 Class，让 Dataflow 明确呈线性：Discover → Activate →
Return。需要逐项审查或控制 Admission 的 Caller 应直接构造两个组件。

Bootstrap 只装配 Manifest 与 Lazy Factory References；不会在此 Import 或实例化所有 Plugin Code。
"""

from __future__ import annotations

from pathlib import Path

from codeflow.plugin.discover import PluginDiscovery
from codeflow.plugin.registry import PluginRegistry


def assemble_plugin_registry(
    *,
    bundled_dir: Path | None = None,
    user_dir: Path | None = None,
    project_dir: Path | None = None,
    entry_points_group: str | None = "codeflow.plugins",
    disabled: frozenset[str] = frozenset(),
) -> PluginRegistry:
    """Discover 所有 Manifests，Admit Enabled Items，并返回 Registry。

    ``entry_points_group`` 默认 ``"codeflow.plugins"``，这是 Third-party Plugins 在 ``pyproject.toml`` 中
    使用的 Public Group。传 `None` 可完全 Suppress Entry-point Discovery，Tests 借此保持 Hermetic。
    Bundled/User/Project Directories 与 Disabled IDs 原样传入相应组件。

    发现或激活冲突会按底层异常向上传播；成功返回表示 Registry 已建立贡献索引，不表示每个 Factory
    Import 或 Backend Construction 已成功。
    """
    discovery = PluginDiscovery(
        bundled_dir=bundled_dir,
        user_dir=user_dir,
        project_dir=project_dir,
        entry_points_group=entry_points_group,
    )
    registry = PluginRegistry()
    registry.activate(discovery.discover(), disabled=disabled)
    return registry


__all__ = ["assemble_plugin_registry"]
