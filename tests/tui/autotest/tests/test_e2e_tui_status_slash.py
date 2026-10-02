"""E2E: `codeflow` alt-screen `/status` round-trip — ACCEPTANCE.

This is the **live ACCEPTANCE gate**. It exercises the full harness
path:
1. spawn production `uv run codeflow` (Python parent + Node child + unix
   socket RPC + Ink alt-screen renderer)
2. wait readiness in alt-screen via Harness.wait (proves snapshot polling works
   against Ink-rendered content)
3. type a slash command into the TUI input (proves character delivery through
   PTY → Ink useInput hook)
4. press enter (proves named-key delivery)
5. wait for the locally rendered retained status panel
6. press ctrl+c twice (cancel input then exit per CodeFlow UX)
7. expect_exit 0 (proves Ink autonomy)

Zero LLM cost; exercises everything except chat streaming — see
test_e2e_codeflow_tui_chat.py for the chat streaming path.
"""

from __future__ import annotations

import pytest


@pytest.mark.e2e
def test_tui_status_slash_round_trip(harness):
    harness.spawn("uv run codeflow")
    assert harness.wait(r"CodeFlow", timeout=25.0), (
        f"TUI CodeFlow readiness banner not seen in 25s; screen=\n{harness.screen()}"
    )
    harness.type("/status")
    harness.press("enter")
    assert harness.wait(r"Runtime status|Model", timeout=10.0), (
        f"`/status` output not rendered within 10s; screen=\n{harness.screen()}"
    )

    harness.press("ctrl+c")
    import time as _t

    _t.sleep(0.5)
    harness.press("ctrl+c")
    assert harness.expect_exit(0, timeout=10.0), f"TUI did not exit 0 after Ctrl+C; final screen=\n{harness.screen()}"
