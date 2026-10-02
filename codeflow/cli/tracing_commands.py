"""``codeflow tracing`` — open the tracing dashboard.

The dashboard is a dependency-free Node viewer bundled under
``codeflow/tracing/viewer/``. Instrumentation itself runs in-process (installed at
CLI startup, see :mod:`codeflow.tracing`); this command only launches the viewer
that reads the captured spans from ``~/.codeflow/traces``.

``codeflow tracing`` (bare) lazily starts the viewer if it is not already running,
then opens the browser. It reuses CodeFlow's Node discovery (:func:`find_node`),
so it needs the same Node >= 22 that ``codeflow`` already requires.

Registered as a top-level leaf command (not a subcommand group) so the TUI
command catalog lists it as a plain ``/tracing`` slash under "(top-level)".
Foreground mode and port are options, not subcommands.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.request
import webbrowser
from pathlib import Path

import typer
from rich.console import Console

from codeflow.tracing import config as tracing_config

console = Console()


def _viewer_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "tracing" / "viewer"


def _port_live(port: int) -> bool:
    """True if something is already listening on 127.0.0.1:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _viewer_health(port: int) -> bool:
    """True only if *our* tracing viewer is serving on ``port``.

    A live port is not enough: a stale viewer from an older layout, or an
    unrelated server (e.g. another observability tool), can hold it and 404
    every request. The viewer answers ``/api/health`` with ``{"ok": true}``;
    a foreign server does not, so this distinguishes reuse from a clash.
    """
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=0.5) as resp:
            if resp.status != 200:
                return False
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 — 任意失败都表示这不是我们的查看器
        return False
    return isinstance(data, dict) and data.get("ok") is True


def _find_free_port(start: int, span: int = 20) -> int | None:
    """First free port in ``[start, start+span)``, or ``None`` if all are taken."""
    for candidate in range(start, start + span):
        if not _port_live(candidate):
            return candidate
    return None


def _viewer_env(port: int) -> dict:
    env = dict(os.environ)
    # server.js 和 log-store.js 都通过该变量解析追踪目录。
    env["TRACING_STATE_DIR"] = str(tracing_config.state_dir())
    env["TRACING_UI_PORT"] = str(port)
    return env


def _resolve_node() -> str:
    from codeflow.cli.tui_commands import find_node

    node, _version = find_node()
    if not node:
        console.print(
            "[red]Node (>= 22) not found.[/red] The tracing dashboard needs the "
            "same Node runtime as the TUI.\n"
            "Install: https://nodejs.org/  or  brew install node@22  or  nvm install 22"
        )
        raise typer.Exit(1)
    return node


def _server_js() -> Path:
    server_js = _viewer_dir() / "server.js"
    if not server_js.exists():
        console.print(f"[red]Viewer not found at {server_js}[/red]")
        raise typer.Exit(1)
    return server_js


def _open_dashboard(port: int) -> None:
    if _port_live(port):
        if _viewer_health(port):
            url = f"http://127.0.0.1:{port}/"
            console.print(f"Tracing dashboard already running at [cyan]{url}[/cyan]")
            webbrowser.open(url)
            return
        # 端口被非本查看器的进程占用（旧实例或无关服务器）；复用会打开损坏或外部页面。
        # 改用下一个空闲端口，避免冲突。
        free = _find_free_port(port + 1)
        if free is None:
            console.print(
                f"[red]Port {port} is in use by another process, and no free port "
                f"was found nearby.[/red] Stop it, or pass --port to pick one."
            )
            raise typer.Exit(1)
        console.print(
            f"[yellow]Port {port} is held by another process (not the tracing viewer); "
            f"starting on {free} instead.[/yellow]"
        )
        port = free

    node = _resolve_node()
    server_js = _server_js()
    log_dir = tracing_config.state_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(log_dir / "viewer.log", "a", encoding="utf-8")  # noqa: SIM115 — 交给子进程持有
    subprocess.Popen(
        [node, str(server_js)],
        cwd=str(_viewer_dir()),
        env=_viewer_env(port),
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
    )

    for _ in range(24):  # 最多等待约 6 秒，直到服务器真正开始服务
        if _viewer_health(port):
            break
        time.sleep(0.25)

    url = f"http://127.0.0.1:{port}/"
    console.print(f"Tracing dashboard at [cyan]{url}[/cyan]")
    webbrowser.open(url)


def _stop_dashboard(port: int) -> None:
    if not _viewer_health(port):
        console.print(f"Tracing dashboard is not running on [cyan]http://127.0.0.1:{port}/[/cyan]")
        return

    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/shutdown",
        method="POST",
        headers={"X-CodeFlow-Viewer-Action": "shutdown"},
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        console.print(f"[red]Could not stop tracing dashboard:[/red] {exc}")
        raise typer.Exit(1) from exc

    if response.status != 200 or payload.get("stopping") is not True:
        console.print("[red]Tracing dashboard rejected the shutdown request.[/red]")
        raise typer.Exit(1)
    console.print(f"Stopped tracing dashboard at [cyan]http://127.0.0.1:{port}/[/cyan]")


def _serve_foreground(port: int) -> None:
    node = _resolve_node()
    console.print(f"Serving tracing dashboard on http://127.0.0.1:{port}/  (Ctrl-C to stop)")
    try:
        subprocess.run(
            [node, str(_server_js())],
            cwd=str(_viewer_dir()),
            env=_viewer_env(port),
            check=False,
        )
    except KeyboardInterrupt:
        pass


def register(app: typer.Typer) -> None:
    """Attach the ``tracing`` command to ``app``.

    A top-level leaf command (not a subcommand group) so the TUI command
    catalog surfaces it as a plain ``/tracing`` slash under "(top-level)",
    the same as ``/status`` / ``/doctor``.
    """

    @app.command("tracing")
    def tracing(
        port: int = typer.Option(None, "--port", "-p", help="Port to bind (default: config or 4318)."),
        foreground: bool = typer.Option(
            False, "--foreground", "-f", help="Run the viewer in the foreground (blocks; Ctrl-C to stop)."
        ),
        stop: bool = typer.Option(False, "--stop", help="Stop the viewer running on the selected port."),
    ) -> None:
        """Open the tracing dashboard (captured LLM/tool/memory spans)."""
        bind_port = port if port is not None else tracing_config.port()
        if stop:
            _stop_dashboard(bind_port)
            return
        if foreground:
            _serve_foreground(bind_port)
        else:
            _open_dashboard(bind_port)
