"""``codeflow plugins`` — inspect installed memory / context plugins.

Reads ``CodeFlowConfig.plugins`` + the live :class:`PluginRegistry`
and prints a table of activated plugins, what each contributes, and
which memory backend the current config selects.

Use cases:

- "Where did this plugin come from?" — the command lists trusted discovery
  sources (bundled / user / entry_points) so the user can see
  where each plugin was resolved from.
- "Why isn't my plugin loading?" — disabled / failed-to-activate
  entries surface here.
- "Which backend is actually active?" — shows the
  ``config.memory.backend`` selection resolved against the registry.

This is a **read-only** command: discovery and admission parse manifests but do
not resolve factories or mutate ``sys.path``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from codeflow.cli._helpers import load_runtime_config

console = Console()


def register(app: typer.Typer) -> None:
    """Attach the ``plugins`` command to ``app``."""

    @app.command()
    def plugins(
        config_path: Optional[str] = typer.Option(
            None,
            "--config",
            "-c",
            help="Path to config file (default: ~/.codeflow/config.json)",
        ),
        verbose: bool = typer.Option(
            False,
            "--verbose",
            "-v",
            help="Show manifest paths + factory references.",
        ),
    ) -> None:
        """List installed plugins + the active memory backend."""
        # 延迟导入，避免每次调用 ``codeflow --help`` 都承担插件发现成本。
        from codeflow.cli._plugin_stack import plugin_discovery_sources
        from codeflow.plugin import (
            PluginDiscovery,
            PluginRegistry,
        )

        ec_config = _load_ec_config(config_path)

        # 将发现与激活分开，使表格不仅显示活跃集合，也能显示被遮蔽（低优先级）和已禁用插件。
        # 扫描来源与实际启动时的受信来源相同。
        discovery = PluginDiscovery(**plugin_discovery_sources())
        discovered = discovery.discover()

        registry = PluginRegistry()
        disabled = frozenset(ec_config.plugins.disabled)
        registry.activate(discovered, disabled=disabled)

        _render_plugin_table(discovered, registry, disabled, verbose=verbose)
        _render_backend_selection(ec_config, registry)


def _load_ec_config(config_path: str | None):
    """Load CodeFlowConfig with the same fallback the other CLI
    commands use. Lazy import so module import is cheap."""
    from codeflow.config.codeflow import load_codeflow_config

    # ``load_runtime_config`` 是规范的基础配置加载器；这里还需要扩展块，因此再通过专用 CodeFlow
    # 加载器读取。调用 ``load_runtime_config`` 是为了与其他 CLI 命令一致：它会设置
    # ``set_config_path``，让下游读取方看到同一文件。
    load_runtime_config(config_path)
    return load_codeflow_config(
        Path(config_path) if config_path else None,
    )


def _render_plugin_table(
    discovered,
    registry,
    disabled,
    *,
    verbose: bool,
) -> None:
    """Print one row per discovered plugin with its status."""

    if not discovered:
        console.print(
            "[yellow]No plugins discovered.[/yellow] Reinstall CodeFlow from the "
            "same distribution source, or run [bold]uv sync[/bold] from a "
            "CodeFlow source checkout. Set "
            "[bold]memory.backend[/bold] to [bold]null[/bold], or drop a "
            "third-party manifest under [bold]~/.codeflow/plugins/[/bold].",
        )
        return

    table = Table(
        title="CodeFlow plugins",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Plugin ID")
    table.add_column("Version")
    table.add_column("Source")
    table.add_column("Status")
    table.add_column("Memory backends")
    if verbose:
        table.add_column("Factory")

    activated_ids = set(registry.activated_ids())
    for record in discovered:
        mf = record.manifest
        pid = mf.id
        if pid in disabled:
            status = "[red]disabled[/red]"
        elif pid in activated_ids:
            status = "[green]activated[/green]"
        elif not mf.enabled_by_default:
            status = "[dim]inactive (opt-in)[/dim]"
        else:
            status = "[yellow]not activated[/yellow]"
        backends = ", ".join(c.name for c in mf.contributes.memory_backends) or "(none)"

        row = [
            pid,
            mf.version,
            _source_label(record.source),
            status,
            backends,
        ]
        if verbose:
            factories = "; ".join(c.factory for c in mf.contributes.memory_backends)
            row.append(factories or "(none)")
        table.add_row(*row)
    console.print(table)


def _source_label(source) -> str:
    """Friendly label for a :class:`Source` enum value."""
    from codeflow.plugin import Source

    return {
        Source.ENTRY_POINTS: "entry_points",
        Source.PROJECT: "project",
        Source.USER: "user",
        Source.BUNDLED: "bundled",
    }.get(source, str(source))


def _render_backend_selection(ec_config, registry) -> None:
    """Show which memory backend the current config activates."""
    selected = ec_config.memory.backend
    if selected is None:
        console.print(
            "\n[bold]Active memory backend:[/bold] [dim]none[/dim] "
            "([italic]Memory is disabled; Local Skills remain available[/italic])",
        )
        return

    available = registry.memory_backend_names()
    if selected not in available:
        console.print(
            f"\n[bold]Active memory backend:[/bold] [red]{selected}[/red] [red](not available)[/red]",
        )
        console.print(
            f"  [dim]Registered: {', '.join(available) or '(none)'}[/dim]",
        )
        console.print(
            "  [dim]Runtime startup will fail closed until the selected "
            "backend is installed or memory.backend is set to null.[/dim]",
        )
        return

    # 找出提供所选后端的插件 ID。
    owner_id = None
    for pid in registry.activated_ids():
        mf = registry.manifest_for(pid)
        if mf is None:
            continue
        for c in mf.contributes.memory_backends:
            if c.name == selected:
                owner_id = pid
                break
        if owner_id:
            break

    console.print(
        f"\n[bold]Active memory backend:[/bold] [green]{selected}[/green] [dim](from plugin: {owner_id})[/dim]",
    )
    console.print(
        f"  [dim]User id:  {ec_config.memory.user_id}[/dim]",
    )
