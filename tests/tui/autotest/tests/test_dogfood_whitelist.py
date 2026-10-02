"""Dogfood retained local TUI commands.

Pattern: spawn → wait readiness → type slash → wait output → Ctrl+C×2 →
expect_exit 0. These commands are handled by the contracted frontend registry
and must not depend on a dynamic command catalog or fallback dispatcher.
"""

from __future__ import annotations

import time

import pytest

from tests.tui.autotest.runner import BackendError

_WHITELIST = [
    ("status", r"Runtime status|Session|Model"),
    ("usage", r"Input tokens|Output tokens|API calls"),
    ("help", r"retained TUI commands|start a new Session"),
]


def _make_test_id(entry):
    return entry[0].replace(" ", "_")


@pytest.mark.e2e
@pytest.mark.parametrize(("slash", "expected"), _WHITELIST, ids=[_make_test_id(e) for e in _WHITELIST])
def test_dogfood_slash_command(harness, slash, expected):

    harness.spawn("uv run codeflow")
    assert harness.wait(r"CodeFlow", timeout=25.0), f"TUI not ready in 25s for /{slash}; screen=\n{harness.screen()}"
    harness.type(f"/{slash}")
    harness.press("enter")
    assert harness.wait(expected, timeout=10.0), (
        f"slash /{slash} did not produce expected output (regex={expected!r}); screen=\n{harness.screen()}"
    )

    for key in ("escape", "ctrl+c"):
        try:
            harness.press(key)
        except BackendError:
            break
        time.sleep(0.5)
    assert harness.expect_exit(0, timeout=10.0), f"TUI did not exit 0 after /{slash}; final screen=\n{harness.screen()}"
