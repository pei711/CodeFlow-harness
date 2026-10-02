"""Regression tests for the bounded local Tracing viewer data path."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VIEWER_DIR = ROOT / "codeflow" / "tracing" / "viewer"


def _node() -> str:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for Tracing viewer tests")
    return node


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _span(index: int) -> dict:
    return {
        "schemaVersion": "audit.span.v1",
        "traceId": f"trace-{index}",
        "spanId": f"span-{index}",
        "parentSpanId": None,
        "name": "llm.call",
        "startTime": f"2026-08-15T00:00:{index % 60:02d}.000Z",
        "endTime": f"2026-08-15T00:00:{index % 60:02d}.001Z",
        "status": {"code": "OK"},
        "attributes": {
            "session.id": f"session-{index}",
            "payload": "x" * 512,
        },
        "events": [],
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{json.dumps(record)}\n" for record in records), encoding="utf-8")


def test_viewer_ui_exposes_explicit_shutdown_lifecycle() -> None:
    shell = (VIEWER_DIR / "ui/shell.js").read_text(encoding="utf-8")
    app = (VIEWER_DIR / "ui/app.js").read_text(encoding="utf-8")

    assert 'id="shutdownButton"' in shell
    assert 'id="shutdownDialog"' in shell
    assert 'id="shutdownState"' in shell
    assert "fetch('/api/shutdown'" in app
    assert "'X-CodeFlow-Viewer-Action': 'shutdown'" in app
    assert "classList.add('viewer-stopped')" in app


def test_viewer_reads_a_bounded_recent_window(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    _write_jsonl(logs / "archive/2026-08-14/audit-spans-old.log", [_span(i) for i in range(20)])
    _write_jsonl(logs / "audit-spans.log", [_span(99)])
    artifact_log = logs / "audit-artifacts/tool.output/audit-spans-poison.log"
    _write_jsonl(artifact_log, [_span(777)])
    future = time.time() + 10
    os.utime(artifact_log, (future, future))

    script = """
const store = require(process.argv[1]);
const result = store.readJsonlWindow('spans');
console.log(JSON.stringify({
  spanIds: result.records.map((record) => record.spanId),
  window: result.window
}));
"""
    env = {
        **os.environ,
        "TRACING_STATE_DIR": str(tmp_path),
        "TRACE_VIEWER_MAX_BYTES": "2048",
    }
    completed = subprocess.run(
        [_node(), "-e", script, str(VIEWER_DIR / "log-store.js")],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = json.loads(completed.stdout)

    assert "span-99" in payload["spanIds"]
    assert "span-0" not in payload["spanIds"]
    assert "span-777" not in payload["spanIds"]
    assert payload["window"]["truncated"] is True
    assert payload["window"]["bytesRead"] <= 2048


def test_viewer_returns_not_modified_for_unchanged_history(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "logs/audit-spans.log", [_span(1)])
    port = _free_port()
    env = {
        **os.environ,
        "TRACING_STATE_DIR": str(tmp_path),
        "TRACING_UI_PORT": str(port),
        "TRACE_VIEWER_MAX_BYTES": "2048",
    }
    process = subprocess.Popen(
        [_node(), str(VIEWER_DIR / "server.js")],
        cwd=VIEWER_DIR,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        health_url = f"http://127.0.0.1:{port}/api/health"
        for _ in range(50):
            try:
                with urllib.request.urlopen(health_url, timeout=0.2) as response:
                    if response.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                time.sleep(0.05)
        else:
            pytest.fail(f"Tracing viewer did not start: {process.stderr.read()}")

        data_url = f"http://127.0.0.1:{port}/api/data"
        with urllib.request.urlopen(data_url, timeout=2) as response:
            assert response.status == 200
            etag = response.headers["ETag"]
            payload = json.load(response)

        assert etag
        assert payload["window"]["truncated"] is False

        request = urllib.request.Request(data_url, headers={"If-None-Match": etag})
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(request, timeout=2)
        assert exc_info.value.code == 304

        time.sleep(0.01)
        _write_jsonl(tmp_path / "logs/audit-spans.log", [_span(1), _span(2)])
        with urllib.request.urlopen(request, timeout=2) as response:
            assert response.status == 200
            assert response.headers["ETag"] != etag
            updated = json.load(response)
        assert {session["sessionId"] for session in updated["sessions"]} == {"session-1", "session-2"}
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_viewer_shutdown_requires_explicit_origin_bound_action(tmp_path: Path) -> None:
    port = _free_port()
    env = {
        **os.environ,
        "TRACING_STATE_DIR": str(tmp_path),
        "TRACING_UI_PORT": str(port),
    }
    process = subprocess.Popen(
        [_node(), str(VIEWER_DIR / "server.js")],
        cwd=VIEWER_DIR,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    health_url = f"http://127.0.0.1:{port}/api/health"
    try:
        for _ in range(50):
            try:
                with urllib.request.urlopen(health_url, timeout=0.2) as response:
                    if response.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                time.sleep(0.05)
        else:
            pytest.fail(f"Tracing viewer did not start: {process.stderr.read()}")

        untrusted = urllib.request.Request(f"http://127.0.0.1:{port}/api/shutdown", method="POST")
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(untrusted, timeout=2)
        assert exc_info.value.code == 403
        assert process.poll() is None

        cross_origin = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/shutdown",
            method="POST",
            headers={
                "Origin": "https://example.com",
                "X-CodeFlow-Viewer-Action": "shutdown",
            },
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(cross_origin, timeout=2)
        assert exc_info.value.code == 403
        assert process.poll() is None

        trusted = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/shutdown",
            method="POST",
            headers={
                "Origin": f"http://127.0.0.1:{port}",
                "X-CodeFlow-Viewer-Action": "shutdown",
            },
        )
        with urllib.request.urlopen(trusted, timeout=2) as response:
            assert response.status == 200
            assert json.load(response)["stopping"] is True
        process.wait(timeout=5)
        assert process.returncode == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
