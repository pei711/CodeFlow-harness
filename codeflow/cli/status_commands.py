"""Top-level ``status`` command — show config / workspace / provider status."""

from __future__ import annotations

import typer
from rich.console import Console

from codeflow import __logo__

console = Console()


def register(app: typer.Typer) -> None:
    """Attach the ``status`` command to ``app``."""

    @app.command()
    def status():
        """Show CodeFlow status."""
        from codeflow.config.loader import get_config_path, load_config
        from codeflow.config.paths import resolve_foreground_paths

        config_path = get_config_path()
        config = load_config()
        paths = resolve_foreground_paths(config)
        from codeflow.cli._plugin_stack import inspect_memory_backend
        from codeflow.config.codeflow import load_codeflow_config

        memory = inspect_memory_backend(load_codeflow_config(config_path))

        console.print(f"{__logo__} CodeFlow Status\n")

        console.print(f"Config: {config_path} {'[green]✓[/green]' if config_path.exists() else '[red]✗[/red]'}")
        console.print(
            f"Workspace: {paths.workspace} {'[green]✓[/green]' if paths.workspace.exists() else '[red]✗[/red]'}"
        )
        console.print(f"State: {paths.state} {'[green]✓[/green]' if paths.state.exists() else '[yellow]new[/yellow]'}")
        if memory.state == "disabled":
            console.print("Memory: [dim]disabled[/dim]")
        elif memory.state == "available":
            console.print(
                f"Memory: [green]{memory.backend}[/green] [dim]({memory.plugin_id} {memory.plugin_version})[/dim]"
            )
        else:
            console.print(f"Memory: [red]{memory.backend} unavailable[/red]")
            console.print(f"  {memory.error}")

        if config_path.exists():
            from codeflow.providers.registry import PROVIDERS

            console.print(f"Model: {config.agents.defaults.model}")

            # 从注册表检查 API 密钥。
            for spec in PROVIDERS:
                p = getattr(config.providers, spec.name, None)
                if p is None:
                    continue
                if spec.is_oauth:
                    console.print(f"{spec.label}: [green]✓ (OAuth)[/green]")
                elif spec.is_local:
                    # 本地部署显示 api_base 而非 api_key。
                    if p.api_base:
                        console.print(f"{spec.label}: [green]✓ {p.api_base}[/green]")
                    else:
                        console.print(f"{spec.label}: [dim]not set[/dim]")
                else:
                    has_key = bool(p.api_key)
                    console.print(f"{spec.label}: {'[green]✓[/green]' if has_key else '[dim]not set[/dim]'}")


__all__ = ["register"]
