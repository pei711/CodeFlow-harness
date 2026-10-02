"""CLI smoke entry tests: `python -m tests.tui.autotest [smoke ...]`.

Primary test path = pytest. This CLI is the ad-hoc fallback for
Bash()-driven verification without writing a pytest file.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

CLI_BASE = [sys.executable, "-m", "tests.tui.autotest"]


def _run(*args: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*CLI_BASE, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class TestHelp:
    def test_help_exits_zero(self):
        result = _run("--help")
        assert result.returncode == 0

        assert "smoke" in (result.stdout + result.stderr).lower()

    def test_help_smoke_exits_zero(self):
        result = _run("smoke", "--help")
        assert result.returncode == 0
        out = result.stdout + result.stderr
        assert "command" in out.lower() or "smoke" in out.lower()


class TestUnknownSubcommand:
    def test_unknown_subcommand_exits_two(self):
        result = _run("nonexistent-subcommand")

        assert result.returncode == 2


@pytest.mark.e2e
class TestSmokeSubcommand:
    def test_smoke_succeeds_on_tui_check(self):

        result = _run(
            "smoke",
            "--wait-readiness",
            r"(?!).*",
            "--wait-timeout",
            "5",
            "uv run codeflow --check",
        )

        assert result.returncode in (0, 1), (
            f"unexpected rc={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    def test_smoke_fails_on_nonexistent_binary(self):
        result = _run("smoke", "/nonexistent/binary/path")

        assert result.returncode == 1, (
            f"expected exit 1 (subprocess exit != 0); got {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert "[fail]" in result.stdout, f"expected [fail] trace marker; stdout:\n{result.stdout}"
