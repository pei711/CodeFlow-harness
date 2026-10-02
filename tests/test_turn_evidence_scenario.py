"""The deterministic scenario `V-TE0` validates.

Drives six real Spine Turns through scripted runners, tools, and outlets --
success, tool failure, provider failure, runner error, cancellation, and
delivery exhaustion -- into an isolated evidence root, then leaves the
artifacts on disk for ``scripts/verify_turn_evidence.py`` to join.

The root is always pytest's ``tmp_path``: the suite's autouse home-isolation
fixture strips every ``CODEFLOW_*`` variable, so an environment handshake would be
both unreliable and a hole in that guard. The verifier instead points
``--basetemp`` at its own output root and reads :data:`MANIFEST_FILENAME`,
written beside it, to learn the exact directory this run used.

Everything here is scripted: no Provider, no platform, no network. Its evidence
class is ``deterministic`` and it must never be reported as a live result.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from codeflow.agent.loop.main import ProviderTurnError
from codeflow.call_efficiency import CallEfficiency
from codeflow.providers.base import LLMResponse
from codeflow.spine import ChatType, Origin, Source, Text, TurnOutcome, TurnRequest, Usage
from codeflow.spine import delivery as delivery_mod
from codeflow.spine.delivery import Capabilities, DeliveryHub, make_hub_sink
from codeflow.spine.scheduler import Lane, OriginPools
from codeflow.token_wise.base import UsageSnapshot
from codeflow.token_wise.usage_tracker import UsageTracker
from codeflow.tracing import spans as _spans
from codeflow.tracing import trace

COMPLETED = "scenario:completed"
TOOL_FAILURE = "scenario:tool_failure"
PROVIDER_FAILURE = "scenario:provider_failure"
RUNNER_ERROR = "scenario:runner_error"
CANCELLED = "scenario:cancelled"
DELIVERY_EXHAUSTED = "scenario:delivery_exhausted"

SCENARIO_CONVERSATIONS = (
    COMPLETED,
    TOOL_FAILURE,
    PROVIDER_FAILURE,
    RUNNER_ERROR,
    CANCELLED,
    DELIVERY_EXHAUSTED,
)

NOTICES_FILENAME = "notices.jsonl"
SPANS_RELPATH = "traces/logs/audit-spans.log"
TELEMETRY_DIRNAME = "telemetry"
MANIFEST_FILENAME = "turn-evidence-manifest.json"


@pytest.fixture
def evidence_root(tmp_path, monkeypatch) -> Path:
    root = Path(tmp_path)
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CODEFLOW_TRACING", "1")
    monkeypatch.setenv("CODEFLOW_TRACING_DIR", str(root / "traces"))
    monkeypatch.setattr(delivery_mod, "_RETRY_BASE_DELAY", 0)
    _spans._store = None

    (root.parent / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "root": str(root),
                "spans": str(root / SPANS_RELPATH),
                "telemetry": str(root / TELEMETRY_DIRNAME),
                "notices": str(root / NOTICES_FILENAME),
                "conversations": list(SCENARIO_CONVERSATIONS),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    yield root
    _spans._store = None


class _Outlet:
    """Succeeds, or always raises when ``broken`` -- the delivery failure path."""

    def __init__(self, name: str, *, broken: bool = False) -> None:
        self.name = name
        self.capabilities = Capabilities()
        self.received: list = []
        self.broken = broken

    async def deliver(self, out) -> None:
        if self.broken:
            raise RuntimeError("transport down")
        self.received.append(out)


class _ScriptedRunner:
    """Opens the spans a real Turn opens, records a usage row, then reaches the
    scripted terminal. Scripted rather than model-backed so the scenario is
    deterministic; the real Agent Loop nesting is covered by
    ``tests/test_turn_evidence_correlation.py``."""

    def __init__(
        self,
        tracker: UsageTracker,
        call_efficiency: CallEfficiency,
        *,
        tool_failures: int = 0,
        raises: BaseException | None = None,
        hang: asyncio.Event | None = None,
    ) -> None:
        self._tracker = tracker
        self._call_efficiency = call_efficiency
        self._tool_failures = tool_failures
        self._raises = raises
        self._hang = hang

    async def run(self, req, emit, drain) -> TurnOutcome:
        with trace.span("session.turn", {"turn.input_preview": req.text}):
            with trace.span("llm.call", {"llm.provider": "scripted", "llm.model": "scripted-1"}) as call:
                call.set({"llm.call_id": call.span_id, "llm.usage.total_tokens": 16})
            ctx = trace.current()
            await self._tracker.after_llm_call(
                {},
                UsageSnapshot(
                    model="scripted-1",
                    input_tokens=11,
                    output_tokens=5,
                    estimated_cost_usd=0.0,
                    session_key=req.conversation,
                    trace_id=getattr(ctx, "trace_id", None),
                    turn_span_id=getattr(ctx, "parent_span_id", None),
                ),
            )
            self._call_efficiency.record(
                LLMResponse(
                    content="scripted",
                    usage={"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
                    model="deepseek/deepseek-v4-flash",
                ),
                requested_model="deepseek/deepseek-v4-flash",
                session_key=req.conversation,
            )
            with trace.span("tool.call", {"tool.name": "grep"}) as tool:
                tool.set({"tool.call_id": "call_scenario_1"})
                if self._tool_failures:
                    tool.error("Error: scripted tool failure")
            if self._hang is not None:
                self._hang.set()
                await asyncio.Event().wait()
            if self._raises is not None:
                raise self._raises
        await emit(Text(content="reply"))
        return TurnOutcome(
            usage=Usage(11, 5, 16),
            explicit_reply=True,
            tool_calls=1,
            tool_failures=self._tool_failures,
        )


def _request(conversation: str, channel: str) -> TurnRequest:
    return TurnRequest(
        origin=Origin.USER,
        source=Source(channel=channel, chat_id="c", sender_id="u", chat_type=ChatType.DM),
        text=f"drive {conversation}",
        conversation=conversation,
    )


async def test_turn_evidence_scenario(evidence_root: Path) -> None:
    notices: list[dict] = []

    async def failure_sink(notice) -> None:
        notices.append({k: (str(v) if v is not None else None) for k, v in asdict(notice).items() if k != "source"})

    hub = DeliveryHub(on_delivery_failure=failure_sink)
    hub.register(_Outlet("ok"))
    hub.register(_Outlet("broken", broken=True))
    sink = make_hub_sink(hub)
    pools = OriginPools(user=4, system=1)
    tracker = UsageTracker(telemetry_dir=evidence_root / "telemetry", persist=True)
    call_efficiency = CallEfficiency(
        mode="observe",
        telemetry_dir=evidence_root / "telemetry",
        persist=True,
    )

    def lane(conversation: str, runner) -> Lane:
        return Lane(runner=runner, pools=pools, sink=sink, conversation_id=conversation)

    try:
        await lane(COMPLETED, _ScriptedRunner(tracker, call_efficiency)).submit(_request(COMPLETED, "ok"))
        await lane(TOOL_FAILURE, _ScriptedRunner(tracker, call_efficiency, tool_failures=1)).submit(
            _request(TOOL_FAILURE, "ok")
        )
        await lane(
            PROVIDER_FAILURE,
            _ScriptedRunner(
                tracker,
                call_efficiency,
                raises=ProviderTurnError("scripted_provider_error"),
            ),
        ).submit(_request(PROVIDER_FAILURE, "ok"))
        await lane(
            RUNNER_ERROR,
            _ScriptedRunner(tracker, call_efficiency, raises=RuntimeError("scripted runner failure")),
        ).submit(_request(RUNNER_ERROR, "ok"))

        hang = asyncio.Event()
        cancel_lane = lane(CANCELLED, _ScriptedRunner(tracker, call_efficiency, hang=hang))
        cancel_future = cancel_lane.submit(_request(CANCELLED, "ok"))
        await hang.wait()
        assert cancel_lane.cancel_running() == 1
        await cancel_future

        await lane(DELIVERY_EXHAUSTED, _ScriptedRunner(tracker, call_efficiency)).submit(
            _request(DELIVERY_EXHAUSTED, "broken")
        )
        await hub.wait_idle("ok")
        await hub.wait_idle("broken")
    finally:
        tracker.close()
        call_efficiency.close()
        await hub.aclose()

    (evidence_root / NOTICES_FILENAME).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in notices),
        encoding="utf-8",
    )

    assert (evidence_root / SPANS_RELPATH).exists()
    assert list((evidence_root / TELEMETRY_DIRNAME).glob("usage-*.jsonl"))
    assert list((evidence_root / TELEMETRY_DIRNAME).glob("call-efficiency-*.jsonl"))
    assert (evidence_root / TELEMETRY_DIRNAME / "call-efficiency-ledger-health.json").exists()
    assert len(notices) == 1
