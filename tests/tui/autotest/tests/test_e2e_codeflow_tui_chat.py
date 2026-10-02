"""Optional E2E smoke: `codeflow` alt-screen chat pipeline liveness.

If this passes, the harness can drive a full Ink alt-screen TUI through
RPC + streaming + slash routing + Ctrl+C autonomy — i.e., Claude Code can
independently reproduce any TUI bug from `Bash()`.

This checkout-based UI smoke is not an installed-wheel or live-Provider
acceptance gate.

This asserts the chat PIPELINE is alive (a prompt is accepted, a turn runs
through RPC + agent-loop + streaming, and the app exits cleanly), NOT any
specific model output. Asserting the model produced a particular answer (its
own name, a fact, a colour) is non-deterministic and therefore an illegitimate
e2e assertion — the model may or may not self-name, and any factual answer can
vary run to run.

Liveness is read from the status bar's turn state (working -> ready), which is
content-agnostic and robust: the Ink alt-screen redraws the entire frame each
tick (welcome art, borders, side panel), so a naive screen-text delta reports
chrome as if it were a reply. The turn-state cycle proves the pipeline ran the
turn regardless of what — or whether — the model rendered any text.

Requires:
- `tui-use` >=0.1.20 on PATH (npm install -g tui-use)
- Built `ui-tui/dist/entry.js` (npm install + npm run build in ui-tui/)
- An accessible default model configured (else the run skips, not fails)

Ink Ctrl+C autonomy yields exit 0, NOT 130. expect_exit(0) is correct.
"""

from __future__ import annotations

import re
import time

import pytest

from tests.tui.autotest.runner import BackendError

pytestmark = pytest.mark.real_llm


_PROMPT = "Reply with a short friendly sentence."


_WORKING_RE = re.compile(
    r"\b(pondering|contemplating|musing|cogitating|ruminating|deliberating|"
    r"mulling|reflecting|processing|reasoning|analyzing|computing|"
    r"synthesizing|formulating|brainstorming)…",
    re.IGNORECASE,
)

_READY_RE = re.compile(r"\bready\b", re.IGNORECASE)


@pytest.mark.e2e
def test_tui_chat_round_trip(harness):

    harness.spawn("uv run codeflow")

    assert harness.wait(r"CodeFlow", timeout=25.0), (
        f"TUI CodeFlow readiness banner not seen in 25s; screen=\n{harness.screen()}"
    )

    harness.type(_PROMPT)
    harness.press("enter")

    started = False
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        screen = harness.screen()
        if re.search(r"error:\s*model_not_available", screen):
            pytest.skip(
                "default model returned model_not_available — the pipeline "
                "could not run a turn; configure an accessible default model "
                "and re-run."
            )
        if _WORKING_RE.search(screen):
            started = True
            break
        time.sleep(0.2)
    assert started, (
        "pipeline liveness failed: no turn started (status bar never entered a "
        f"working state) within 20s of submitting.\nscreen=\n{harness.screen()}"
    )
    assert harness.wait(_READY_RE, timeout=60.0), (
        "pipeline liveness failed: the turn never completed (status bar did not "
        f"return to ready) within 60s.\nscreen=\n{harness.screen()}"
    )

    harness.press("ctrl+c")
    time.sleep(0.5)
    try:
        harness.press("ctrl+c")
    except BackendError:
        pass
    assert harness.expect_exit(0, timeout=10.0), f"TUI did not exit 0 after Ctrl+C; final screen=\n{harness.screen()}"
