"""Tests for the native TUI behind the bare ``codeflow`` entry point."""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from codeflow.cli.commands import app

runner = CliRunner()


def test_tui_check_node_ok(monkeypatch):
    """When node is on PATH and version >= 22, --check exits 0."""
    monkeypatch.setattr(
        "codeflow.cli.tui_commands.find_node",
        lambda: ("/usr/bin/node", (22, 5, 0)),
    )
    monkeypatch.setattr(
        "codeflow.cli.tui_commands.run_subprocess",
        lambda *_args, **_kw: 0,
    )
    result = runner.invoke(app, ["--check"])
    assert result.exit_code == 0, result.output


def test_tui_check_node_missing(monkeypatch):
    """When node not found, --check exits 1 with friendly error."""
    monkeypatch.setattr(
        "codeflow.cli.tui_commands.find_node",
        lambda: (None, None),
    )
    result = runner.invoke(app, ["--check"])
    assert result.exit_code == 1
    assert "Node" in result.output
    assert "legacy-repl" in result.output


def test_tui_check_node_too_old(monkeypatch):
    """When node < 22, --check exits 1."""
    monkeypatch.setattr(
        "codeflow.cli.tui_commands.find_node",
        lambda: ("/usr/bin/node", (18, 0, 0)),
    )
    result = runner.invoke(app, ["--check"])
    assert result.exit_code == 1
    assert "22" in result.output


def test_tui_color_flag_forwards_to_child_env(monkeypatch):
    """`--color` exports CODEFLOW_TUI_COLOR for the Node child (read by
    colorTier.ts)."""
    import os

    monkeypatch.delenv("CODEFLOW_TUI_COLOR", raising=False)
    monkeypatch.setattr(
        "codeflow.cli.tui_commands.find_node",
        lambda: ("/usr/bin/node", (22, 5, 0)),
    )
    captured: dict[str, str | None] = {}

    def fake_run_subprocess(*_a, **_k):
        captured["color"] = os.environ.get("CODEFLOW_TUI_COLOR")
        return 0

    monkeypatch.setattr(
        "codeflow.cli.tui_commands.run_subprocess",
        fake_run_subprocess,
    )

    result = runner.invoke(app, ["--check", "--color", "256"])

    assert result.exit_code == 0, result.output
    assert captured["color"] == "256"


def test_tui_print_colors_uses_no_rpc_spawn(monkeypatch, tmp_path):
    """`--print-colors` is a no-RPC stdio spawn: it must use run_subprocess
    (not run_subprocess_with_rpc) and set CODEFLOW_TUI_PRINT_COLORS."""
    import os

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "entry.js").write_text("", encoding="utf-8")

    monkeypatch.delenv("CODEFLOW_TUI_PRINT_COLORS", raising=False)
    monkeypatch.setattr("codeflow.cli.tui_commands._UI_TUI_DIR", tmp_path)
    monkeypatch.setattr(
        "codeflow.cli.tui_commands.find_node",
        lambda: ("/usr/bin/node", (22, 5, 0)),
    )

    calls = {"plain": False, "rpc": False, "print_colors": None}

    def fake_plain(*_a, **_k):
        calls["plain"] = True
        calls["print_colors"] = os.environ.get("CODEFLOW_TUI_PRINT_COLORS")
        return 0

    def fake_rpc(*_a, **_k):
        calls["rpc"] = True
        return 3

    monkeypatch.setattr("codeflow.cli.tui_commands.run_subprocess", fake_plain)
    monkeypatch.setattr("codeflow.cli.tui_commands.run_subprocess_with_rpc", fake_rpc)

    result = runner.invoke(app, ["--print-colors"])

    assert result.exit_code == 0, result.output
    assert calls["plain"] is True
    assert calls["rpc"] is False
    assert calls["print_colors"] == "1"


def test_tui_preview_colors_uses_no_rpc_spawn(monkeypatch, tmp_path):
    """`--preview-colors` is a no-RPC stdio spawn that sets
    CODEFLOW_TUI_COLOR_PREVIEW."""
    import os

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "entry.js").write_text("", encoding="utf-8")

    monkeypatch.delenv("CODEFLOW_TUI_COLOR_PREVIEW", raising=False)
    monkeypatch.setattr("codeflow.cli.tui_commands._UI_TUI_DIR", tmp_path)
    monkeypatch.setattr(
        "codeflow.cli.tui_commands.find_node",
        lambda: ("/usr/bin/node", (22, 5, 0)),
    )

    calls = {"plain": False, "rpc": False, "preview": None}

    def fake_plain(*_a, **_k):
        calls["plain"] = True
        calls["preview"] = os.environ.get("CODEFLOW_TUI_COLOR_PREVIEW")
        return 0

    monkeypatch.setattr("codeflow.cli.tui_commands.run_subprocess", fake_plain)
    monkeypatch.setattr(
        "codeflow.cli.tui_commands.run_subprocess_with_rpc",
        lambda *_a, **_k: calls.__setitem__("rpc", True) or 3,
    )

    result = runner.invoke(app, ["--preview-colors"])

    assert result.exit_code == 0, result.output
    assert calls["plain"] is True
    assert calls["rpc"] is False
    assert calls["preview"] == "1"


def test_codeflow_node_override_no_fallback(monkeypatch, tmp_path):
    """When CODEFLOW_NODE is set but path missing, find_node returns (None, None)
    — must NOT fall back to venv/PATH."""
    from codeflow.cli.tui_commands import find_node

    monkeypatch.setenv("CODEFLOW_NODE", str(tmp_path / "nonexistent-node"))

    node_path, version = find_node()
    assert node_path is None, "CODEFLOW_NODE override must not fall back to venv/PATH"
    assert version is None


def test_find_node_discovers_windows_private_runtime(monkeypatch, tmp_path):
    """The Windows installer unpacks Node from the official zip, whose binary
    lives at ``node-v22.x.y-win-x64/node.exe`` rather than ``bin/node``."""
    if sys.platform == "win32":
        pytest.skip("uses a POSIX fake node executable")

    from codeflow.cli import tui_commands

    node_dir = tmp_path / "runtime" / "node-v22.20.0-win-x64"
    node_dir.mkdir(parents=True)
    node = node_dir / "node.exe"
    node.write_text("#!/bin/sh\necho v22.20.0\n", encoding="utf-8")
    node.chmod(node.stat().st_mode | stat.S_IXUSR)

    monkeypatch.setattr(tui_commands.sys, "platform", "win32")
    monkeypatch.setenv("CODEFLOW_HOME", str(tmp_path))
    monkeypatch.delenv("CODEFLOW_NODE", raising=False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setenv("PATH", "")

    node_path, version = tui_commands.find_node()

    assert node_path == str(node)
    assert version == (22, 20, 0)


def test_dev_npx_derived_from_node_path(monkeypatch, tmp_path):
    """`--dev` mode must derive npx from the validated node_path,
    not from PATH — so CODEFLOW_NODE's version-pin semantics are honored."""

    fake_bin = tmp_path / "fake-node" / "bin"
    fake_bin.mkdir(parents=True)
    fake_node = fake_bin / "node"
    fake_npx = fake_bin / "npx"
    fake_node.write_text("#!/bin/sh\necho ok\n")
    fake_npx.write_text("#!/bin/sh\necho ok\n")

    monkeypatch.setattr(
        "codeflow.cli.tui_commands.find_node",
        lambda: (str(fake_node), (22, 5, 0)),
    )

    captured: dict[str, object] = {}

    def fake_run_subprocess_with_rpc(node_path, args, cwd, **_kw):
        captured["node_path"] = node_path
        captured["args"] = args
        captured["cwd"] = cwd
        return 0

    monkeypatch.setattr(
        "codeflow.cli.tui_commands.run_subprocess_with_rpc",
        fake_run_subprocess_with_rpc,
    )

    monkeypatch.setattr(
        "codeflow.cli.tui_commands.shutil.which",
        lambda _name: "/usr/bin/npx",
    )

    result = runner.invoke(app, ["--dev"])
    assert result.exit_code == 0, result.output
    assert captured["node_path"] == str(fake_npx), (
        f"Expected --dev to spawn derived npx {fake_npx}, got {captured['node_path']!r}"
    )


def test_run_subprocess_returns_child_exit_code(monkeypatch):
    """Verify run_subprocess transparently returns child's exit code."""
    from codeflow.cli.tui_commands import run_subprocess

    class FakeProc:
        def __init__(self, exit_code):
            self._exit_code = exit_code

        def wait(self, timeout=None):
            return self._exit_code

        def send_signal(self, sig):
            pass

    def fake_popen(*_args, **_kw):
        return FakeProc(42)

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    monkeypatch.setattr("signal.signal", lambda *_a, **_kw: None)

    result = run_subprocess("/usr/bin/node", ["dist/entry.js"], Path("/tmp"))
    assert result == 42


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


def test_rpc_handshake_timeout(monkeypatch, tmp_path):
    """When the Node child never sends `system.hello`, the parent must
    abort with exit code 3 and a diagnostic stderr message.

    We replace `find_node` with a stub so the existing Node discovery path
    is bypassed, and replace `run_subprocess_with_rpc` with a stub that
    simulates "child spawned, never sent hello → handshake_timeout()".
    """

    fake_ui_dir = tmp_path / "ui-tui"
    (fake_ui_dir / "dist").mkdir(parents=True)
    (fake_ui_dir / "dist" / "entry.js").write_text("// stub")
    monkeypatch.setattr("codeflow.cli.tui_commands._UI_TUI_DIR", fake_ui_dir)

    monkeypatch.setattr(
        "codeflow.cli.tui_commands.find_node",
        lambda: ("/usr/bin/node", (22, 5, 0)),
    )

    monkeypatch.setattr(
        "codeflow.cli.tui_commands.run_subprocess_with_rpc",
        lambda *_a, **_kw: 3,
    )

    result = runner.invoke(app, [])
    assert result.exit_code == 3, result.output


def test_rpc_handshake_timeout_helper_real(tmp_path, monkeypatch):
    """End-to-end: invoke the actual `run_subprocess_with_rpc` helper with a
    child that exits immediately without ever sending `system.hello` —
    the helper must return exit code 3.

    Uses `sh -c 'sleep 6'` as a stand-in child: it inherits stdio + FDs but
    never writes anything to the request pipe → handshake timeout fires
    (we configure a 1s timeout in the test for speed).
    """
    from codeflow.cli import tui_commands

    monkeypatch.setattr(tui_commands, "_RPC_HANDSHAKE_TIMEOUT_S", 1.0)

    exit_code = tui_commands.run_subprocess_with_rpc(
        "/bin/sh",
        ["-c", "sleep 6"],
        cwd=tmp_path,
        forward_signals=False,
    )
    assert exit_code == 3


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
#


def test_rpc_socket_env_constants_exposed():
    """Sanity: the production transport exposes the env var names the Node
    child reads -- the TCP-loopback address and the auth token."""
    from codeflow.cli import tui_commands

    assert tui_commands._RPC_SOCKET_ENV == "CODEFLOW_RPC_SOCKET"
    assert tui_commands._RPC_TOKEN_ENV == "CODEFLOW_RPC_TOKEN"


def test_production_dispatcher_includes_all_umbrella_methods():
    """Regression: prevent drift between the umbrella registrar and the
    production ``run_subprocess_with_rpc`` spawn path.

    This test asserts that every method the umbrella registers is also
    registered by the production path. Future additions to the umbrella
    automatically extend this test.
    """
    from codeflow.tui_rpc.dispatcher import Dispatcher
    from codeflow.tui_rpc.methods import (
        register_aligned_methods,
        register_aligned_methods_except_system,
    )
    from codeflow.tui_rpc.methods.system import (
        system_hello,
        system_ping,
        system_version,
    )

    umbrella = Dispatcher()
    register_aligned_methods(umbrella)
    umbrella_methods = set(umbrella.methods())

    production = Dispatcher()
    production.register("system.hello", system_hello)
    production.register("system.ping", system_ping)
    production.register("system.version", system_version)
    register_aligned_methods_except_system(production)
    production_methods = set(production.methods())

    assert umbrella_methods == production_methods, (
        f"production dispatcher drifted from umbrella; "
        f"missing in production: {umbrella_methods - production_methods}; "
        f"extra in production: {production_methods - umbrella_methods}"
    )

    for required in (
        "session.create",
        "session.resume",
        "session.list",
        "image.attach",
        "model.options",
        "config.set",
        "terminal.resize",
    ):
        assert required in production_methods, (
            f"{required} missing from production dispatcher — `codeflow` will return -32601 for it"
        )
    for removed in ("cli.dispatch", "commands.catalog", "slash.exec", "complete.slash", "reload.mcp"):
        assert removed not in production_methods


def test_confirm_registered_when_broker_present():
    """``confirm.respond`` registers only when a ConfirmBroker is supplied
    (mirrors the emitter/turn gate). Without a broker neither the umbrella nor
    the production path registers it, so the drift test above stays balanced.
    """
    from codeflow.tui_rpc.confirm_broker import ConfirmBroker
    from codeflow.tui_rpc.dispatcher import Dispatcher
    from codeflow.tui_rpc.methods import register_aligned_methods_except_system

    async def _send(_frame):
        return None

    broker = ConfirmBroker(_send)

    without_broker = Dispatcher()
    register_aligned_methods_except_system(without_broker)
    assert "confirm.respond" not in without_broker.methods()

    with_broker = Dispatcher()
    register_aligned_methods_except_system(with_broker, confirm_broker=broker)
    assert "confirm.respond" in with_broker.methods()


def test_rpc_socket_transport_handshake_ok(tmp_path, monkeypatch):
    """End-to-end: spawn a tiny Python child that connects to the TCP-loopback
    address in `CODEFLOW_RPC_SOCKET`, sends the `CODEFLOW_RPC_TOKEN` auth line, then
    ``system.hello`` — the parent helper must complete the handshake and the
    child must exit 0.

    We use a Python child rather than the real Node demo here so the test
    doesn't depend on the ui-tui build artifact; the wire protocol (an auth
    line then newline JSON over the socket) is identical regardless of language.
    """
    from codeflow.cli import tui_commands

    monkeypatch.setattr(tui_commands, "_RPC_HANDSHAKE_TIMEOUT_S", 3.0)

    child_src = """
import json, os, socket, sys

host, port = os.environ["CODEFLOW_RPC_SOCKET"].rsplit(":", 1)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((host, int(port)))
tok = os.environ.get("CODEFLOW_RPC_TOKEN")
if tok:
    s.sendall((tok + "\\n").encode("utf-8"))
req = {"jsonrpc": "2.0", "id": 1, "method": "system.hello",
       "params": {"client_version": "0.1.0"}}
s.sendall((json.dumps(req) + "\\n").encode("utf-8"))
# Read one response frame.
buf = b""
while b"\\n" not in buf:
    chunk = s.recv(4096)
    if not chunk:
        sys.exit(2)
    buf += chunk
resp = json.loads(buf.decode("utf-8").splitlines()[0])
assert resp.get("result", {}).get("server_version"), resp
s.close()
sys.exit(0)
"""
    exit_code = tui_commands.run_subprocess_with_rpc(
        sys.executable,
        ["-c", child_src],
        cwd=tmp_path,
        forward_signals=False,
    )
    assert exit_code == 0, f"expected child to exit 0, got {exit_code}"


def test_rpc_socket_handshake_timeout_when_child_never_connects(tmp_path, monkeypatch):
    """If the spawned child never connects to the unix socket within the
    handshake deadline, parent must return exit code 3."""
    from codeflow.cli import tui_commands

    monkeypatch.setattr(tui_commands, "_RPC_HANDSHAKE_TIMEOUT_S", 1.0)

    exit_code = tui_commands.run_subprocess_with_rpc(
        "/bin/sh",
        ["-c", "sleep 6"],
        cwd=tmp_path,
        forward_signals=False,
    )
    assert exit_code == 3


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


import sys  # noqa: E402  （为后续追加的测试保留在文件末尾）
