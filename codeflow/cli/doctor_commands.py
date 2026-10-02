"""``codeflow doctor`` — health check (static + optional --probe).

Default mode is zero-network, millisecond-fast. ``--probe`` sends one
chat exchange via :func:`codeflow.cli._helpers.send_probe`.

Exit codes:
  0  — all green (and probe ok if requested)
  1  — static check failed (config, routing, workspace, Channel SDK, or TUI bundle)
  2  — static checks ok but ``--probe`` failed (lets CI distinguish from 1)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from codeflow import __logo__
from codeflow.cli._helpers import print_probe_troubleshooting, send_probe

console = Console()


@dataclass
class PathsInfo:
    config_path: str
    config_exists: bool
    workspace_path: str = ""
    workspace_exists: bool = False
    workspace_writable: bool = False
    state_path: str = ""
    state_exists: bool = False
    state_writable: bool = False


@dataclass
class RoutingInfo:
    model: str
    provider: Optional[str]
    max_tokens: int
    context_window_tokens: int


@dataclass
class FeaturesInfo:
    channels_enabled: list[str] = field(default_factory=list)
    missing_channel_extras: list[str] = field(default_factory=list)
    skill_forge_enabled: bool = False
    tui_bundle_available: bool = False


@dataclass
class MemoryInfo:
    backend: Optional[str]
    state: str
    plugin_id: Optional[str] = None
    plugin_version: Optional[str] = None
    error: Optional[str] = None


@dataclass
class GatewayInfo:
    running: bool = False
    pid: Optional[int] = None
    started_at: Optional[float] = None


@dataclass
class ProbeResult:
    ok: bool
    text: Optional[str] = None
    tokens: Optional[int] = None
    elapsed_s: Optional[float] = None
    error: Optional[str] = None


@dataclass
class DoctorReport:
    version: int = 1
    config_loaded: bool = False
    paths: Optional[PathsInfo] = None
    routing: Optional[RoutingInfo] = None
    features: Optional[FeaturesInfo] = None
    memory: Optional[MemoryInfo] = None
    gateway: Optional[GatewayInfo] = None
    probe: Optional[ProbeResult] = None

    def exit_code(self) -> int:
        if self.paths is None or not self.paths.config_exists:
            return 1
        if not self.config_loaded:
            return 1
        if not self.paths.workspace_writable or not self.paths.state_writable:
            return 1
        if self.routing is None or self.routing.provider is None:
            return 1
        if (
            self.features is None
            or not self.features.tui_bundle_available
            or bool(self.features.missing_channel_extras)
        ):
            return 1
        if self.memory is None or self.memory.state == "error":
            return 1
        if self.probe is not None and not self.probe.ok:
            return 2
        return 0


def _gather_static_checks() -> DoctorReport:
    """Inspect config / routing / features. Strictly zero-network."""
    from codeflow.config.loader import get_config_path, load_config
    from codeflow.config.paths import resolve_foreground_paths

    config_path = get_config_path()
    paths = PathsInfo(
        config_path=str(config_path),
        config_exists=config_path.exists(),
    )
    report = DoctorReport(paths=paths)

    if not paths.config_exists:
        return report

    try:
        config = load_config()
    except Exception:
        return report
    report.config_loaded = True

    from codeflow.cli._plugin_stack import inspect_memory_backend
    from codeflow.config.codeflow import load_codeflow_config

    memory = inspect_memory_backend(load_codeflow_config(config_path))
    report.memory = MemoryInfo(
        backend=memory.backend,
        state=memory.state,
        plugin_id=memory.plugin_id,
        plugin_version=memory.plugin_version,
        error=memory.error,
    )

    runtime_paths = resolve_foreground_paths(config)
    paths.workspace_path = str(runtime_paths.workspace)
    paths.workspace_exists = runtime_paths.workspace.exists()
    paths.workspace_writable = _workspace_is_writable(runtime_paths.workspace)
    paths.state_path = str(runtime_paths.state)
    paths.state_exists = runtime_paths.state.exists()
    paths.state_writable = _workspace_is_writable(runtime_paths.state)

    defaults = config.agents.defaults
    report.routing = RoutingInfo(
        model=defaults.model,
        provider=config.get_provider_name(),
        max_tokens=defaults.max_tokens,
        context_window_tokens=defaults.context_window_tokens,
    )

    enabled: list[str] = []
    for name, value in config.channels.__dict__.items():
        if getattr(value, "enabled", False):
            enabled.append(name)

    try:
        skill_forge_on = bool(config.skill_forge.enabled)
    except Exception:
        skill_forge_on = False

    report.features = FeaturesInfo(
        channels_enabled=enabled,
        missing_channel_extras=[name for name in enabled if not _channel_extra_available(name)],
        skill_forge_enabled=skill_forge_on,
        tui_bundle_available=_tui_bundle_available(),
    )

    from codeflow.cli._gateway_lock import read_status

    info = read_status(now=time.time())
    if info is None:
        report.gateway = GatewayInfo(running=False)
    else:
        report.gateway = GatewayInfo(running=True, pid=info.pid, started_at=info.started_at)

    return report


def _workspace_is_writable(workspace: Path) -> bool:
    candidate = workspace
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            return False
        candidate = parent
    return candidate.is_dir() and os.access(candidate, os.W_OK | os.X_OK)


def _tui_bundle_available() -> bool:
    from codeflow.cli.tui_commands import resolve_dist_entry

    return resolve_dist_entry() is not None


_CHANNEL_EXTRA_MODULES = {
    "feishu": "lark_oapi",
    "qq": "botpy",
    "wecom": "wecom_aibot_sdk",
}


def _channel_extra_available(channel: str) -> bool:
    module = _CHANNEL_EXTRA_MODULES.get(channel)
    return module is not None and find_spec(module) is not None


def _channel_maturity(channel: str) -> str:
    """Evidence level declared by the channel's spec, never a stronger claim."""
    from codeflow.channels.registry import discover_specs

    spec = discover_specs().get(channel)
    return spec.maturity if spec else "unknown"


def _run_llm_probe(timeout_s: int) -> ProbeResult:
    """Wrap :func:`send_probe` so failures become a structured ProbeResult."""
    try:
        text, tokens, elapsed = send_probe(timeout_s=timeout_s)
        return ProbeResult(ok=True, text=text, tokens=tokens, elapsed_s=elapsed)
    except Exception as exc:
        return ProbeResult(ok=False, error=str(exc) or exc.__class__.__name__)


def _render_human_output(report: DoctorReport) -> None:
    console.print(f"\n{__logo__} CodeFlow Doctor\n")

    paths = report.paths
    assert paths is not None  # _gather_static_checks 始终会填充该值
    console.print("[bold]Paths[/bold]")
    if paths.config_exists:
        console.print(f"  Config:    {paths.config_path}  [green]✓[/green]")
    else:
        console.print(f"  Config:    {paths.config_path}  [red]✗  (not found)[/red]")
    if paths.config_exists:
        exists_mark = "[green]exists[/green]" if paths.workspace_exists else "[yellow]will be created[/yellow]"
        writable_mark = "[green]writable[/green]" if paths.workspace_writable else "[red]not writable[/red]"
        console.print(f"  Workspace: {paths.workspace_path}  {exists_mark}, {writable_mark}")
        state_exists_mark = "[green]exists[/green]" if paths.state_exists else "[yellow]will be created[/yellow]"
        state_writable_mark = "[green]writable[/green]" if paths.state_writable else "[red]not writable[/red]"
        console.print(f"  State:     {paths.state_path}  {state_exists_mark}, {state_writable_mark}")

    if not paths.config_exists:
        console.print("\n[yellow]⚠ CodeFlow is not configured.[/yellow] Run [cyan]codeflow onboard[/cyan] to set it up.")
        return

    if not report.config_loaded:
        console.print("\n[red]✗ Config schema invalid.[/red] Run [cyan]codeflow onboard --reset[/cyan] to recreate it.")
        return

    routing = report.routing
    if routing is not None:
        console.print("\n[bold]Routing[/bold]")
        console.print(f"  Model:        {routing.model}")
        if routing.provider:
            console.print(f"  Routes to:    {routing.provider}")
        else:
            console.print("  Routes to:    [red]<unresolved>[/red]")
        console.print(f"  Max tokens:   {routing.max_tokens}")
        console.print(f"  Context win:  {routing.context_window_tokens}")

    features = report.features
    if features is not None:
        console.print("\n[bold]Features[/bold]")
        count = len(features.channels_enabled)
        if count:
            # 转义左方括号，让 Rich 渲染字面标签而非标记。
            labelled = ", ".join(f"{name} \\[{_channel_maturity(name)}]" for name in features.channels_enabled)
            console.print(f"  Channels:    {count} enabled  ({labelled})")
        else:
            console.print("  Channels:    [dim]none enabled[/dim]")
        if features.missing_channel_extras:
            console.print("  Channel SDKs: [red]missing for " + ", ".join(features.missing_channel_extras) + "[/red]")
        sf_label = "enabled" if features.skill_forge_enabled else "[dim]disabled[/dim]"
        console.print(f"  Skill forge: {sf_label}")
        tui_label = "[green]available[/green]" if features.tui_bundle_available else "[red]missing[/red]"
        console.print(f"  TUI bundle:  {tui_label}")

    memory = report.memory
    if memory is not None:
        console.print("\n[bold]Memory[/bold]")
        if memory.state == "disabled":
            console.print("  Backend:     [dim]disabled[/dim]")
        elif memory.state == "available":
            owner = f"{memory.plugin_id} {memory.plugin_version}".strip()
            console.print(f"  Backend:     [green]{memory.backend}[/green] [dim]({owner})[/dim]")
        else:
            console.print(f"  Backend:     [red]{memory.backend} unavailable[/red]")
            console.print(f"  Remedy:      {memory.error}")

    gateway = report.gateway
    if gateway is not None:
        console.print("\n[bold]Gateway[/bold]")
        if gateway.running:
            since = (
                datetime.fromtimestamp(gateway.started_at).strftime("%Y-%m-%d %H:%M:%S") if gateway.started_at else "?"
            )
            console.print(f"  [green]✓ running[/green] (pid {gateway.pid}, since {since})")
        else:
            console.print("  [dim]not running[/dim]")

    if report.probe is not None:
        console.print("\n[bold]LLM Probe[/bold]")
        if routing:
            console.print(f"  → {routing.model}")
        if report.probe.ok:
            console.print(f'  [green]✓ Response:[/green] "{report.probe.text}"')
            extras: list[str] = []
            if report.probe.tokens:
                extras.append(f"{report.probe.tokens} tokens")
            if report.probe.elapsed_s is not None:
                extras.append(f"{report.probe.elapsed_s:.1f}s")
            if extras:
                console.print(f"  [green]✓ {', '.join(extras)}[/green]")
        else:
            console.print(f"  [red]✗ Failed:[/red] {report.probe.error}")
            print_probe_troubleshooting(routing.provider if routing else None)

    console.print()
    code = report.exit_code()
    if code == 0:
        if report.probe is None:
            console.print("[green]✓ Configuration looks healthy.[/green]")
            console.print("Run [cyan]doctor --probe[/cyan] to send a test message and verify the LLM responds.")
        else:
            console.print("[green]✓ All checks passed.[/green]")
    elif routing and routing.provider is None:
        console.print(
            f"[red]✗ Model [bold]{routing.model}[/bold] could not be routed to any configured provider.[/red]"
        )
        console.print("Run [cyan]codeflow provider list[/cyan] / [cyan]codeflow provider set[/cyan] to fix routing.")
    elif paths is not None and not paths.workspace_writable:
        console.print(f"[red]✗ Workspace is not writable:[/red] {paths.workspace_path}")
    elif paths is not None and not paths.state_writable:
        console.print(f"[red]✗ Workspace State is not writable:[/red] {paths.state_path}")
    elif report.features is not None and not report.features.tui_bundle_available:
        console.print("[red]✗ Native TUI bundle is missing.[/red] Reinstall CodeFlow or rebuild the TUI.")
    elif report.features is not None and report.features.missing_channel_extras:
        names = ", ".join(f"channel-{name}" for name in report.features.missing_channel_extras)
        console.print(f"[red]✗ Enabled Channel dependencies are missing:[/red] {names}")
    elif report.memory is not None and report.memory.state == "error":
        console.print(f"[red]✗ Memory backend is unavailable:[/red] {report.memory.error}")


def register(app: typer.Typer) -> None:
    @app.command()
    def doctor(
        probe: bool = typer.Option(False, "--probe", help="Send a test message to verify the LLM responds."),
        json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON (CI-friendly)."),
        timeout: int = typer.Option(
            15,
            "--timeout",
            help="LLM probe timeout in seconds.",
            min=1,
        ),
    ) -> None:
        """Health-check CodeFlow config, routing, and (optionally) the LLM."""
        report = _gather_static_checks()

        if probe and report.routing is not None and report.routing.provider is not None:
            report.probe = _run_llm_probe(timeout_s=timeout)

        if json_output:
            console.print_json(json.dumps(asdict(report)))
        else:
            _render_human_output(report)

        raise typer.Exit(report.exit_code())


__all__ = ["register"]
