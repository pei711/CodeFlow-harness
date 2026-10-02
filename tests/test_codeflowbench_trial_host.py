"""Trial-host tests including isolation for frozen historical campaigns."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from benchmarks.codeflowbench.budget import (
    BudgetGuardedProvider,
    ProviderBudgetConfig,
    ProviderBudgetError,
    ProviderBudgetLedger,
    provider_call_budget_scope,
)
from benchmarks.codeflowbench.host import RecordingOutlet, RuntimeTrialHost
from benchmarks.codeflowbench.isolation import TrialIsolation
from benchmarks.codeflowbench.records import VerificationState
from benchmarks.codeflowbench.usage import RecordingProvider, UsageRecorder
from benchmarks.codeflowbench.verifier import (
    JsonArtifactVerifier,
    VerifierSeal,
    run_sealed_verifier,
)
from codeflow.agent.loop.main import ProviderTurnError
from codeflow.agent.spine_runner import AgentTurnRunner
from codeflow.config.codeflow import CodeFlowConfig
from codeflow.config.schema import Config
from codeflow.providers.base import (
    ErrorClassification,
    LLMProvider,
    LLMResponse,
    StreamDelta,
)
from codeflow.spine import ChatType, Origin, Source, TurnRequest
from codeflow.spine.events import Text, Usage
from codeflow.spine.runner import TurnOutcome


def test_trial_isolation_creates_non_overlapping_state_roots(tmp_path: Path) -> None:
    first = TrialIsolation.create(tmp_path, "attempt-a")
    second = TrialIsolation.create(tmp_path, "attempt-b")

    first.prepare()
    second.prepare()

    assert first.workspace != second.workspace
    assert first.codeflow_home != second.codeflow_home
    assert first.everos_root != second.everos_root
    assert first.trace_root != second.trace_root
    assert first.workspace.is_dir()
    assert first.child_environment()["CODEFLOW_HOME"] == str(first.codeflow_home)
    assert first.child_environment()["EVEROS_ROOT"] == str(first.everos_root)


class _UsageProvider:
    generation = object()

    def get_default_model(self) -> str:
        return "scripted/usage"

    async def chat(self, **kwargs) -> LLMResponse:
        return LLMResponse(
            content="ok",
            model="scripted/actual-usage",
            finish_reason="stop",
            usage={
                "prompt_tokens": 11,
                "completion_tokens": 3,
                "total_tokens": 14,
            },
        )


class _BudgetDelegate(LLMProvider):
    def get_default_model(self) -> str:
        return "scripted/budget"

    async def chat(self, **kwargs) -> LLMResponse:
        raise AssertionError("logical call ceiling must block the delegate")


@pytest.mark.asyncio
async def test_recording_provider_aggregates_every_call_once() -> None:
    recorder = UsageRecorder()
    provider = RecordingProvider(_UsageProvider(), recorder=recorder)

    await provider.chat(messages=[], model="scripted/usage")
    await provider.chat(messages=[], model="scripted/usage")

    records = recorder.records()
    aggregate = recorder.aggregate()
    assert len(records) == 2
    assert len({record.call_id for record in records}) == 2
    assert {record.requested_model for record in records} == {
        "scripted/usage",
    }
    assert {record.model for record in records} == {
        "scripted/actual-usage",
    }
    assert aggregate.input_tokens == 22
    assert aggregate.output_tokens == 6
    assert aggregate.total_tokens == 28
    assert aggregate.usage_complete is True


class _FailingUsageProvider(_UsageProvider):
    async def chat(self, **kwargs) -> LLMResponse:
        raise RuntimeError("provider unavailable")

    def classify_error(self, exc, content=None) -> ErrorClassification:
        return ErrorClassification(category="provider_unavailable")


@pytest.mark.asyncio
async def test_recording_provider_keeps_failed_attempt_and_blocks_complete_usage() -> None:
    recorder = UsageRecorder()
    provider = RecordingProvider(
        _FailingUsageProvider(),
        recorder=recorder,
    )

    with pytest.raises(RuntimeError, match="provider unavailable"):
        await provider.chat(messages=[], model="scripted/usage")

    records = recorder.records()
    assert len(records) == 1
    assert records[0].succeeded is False
    assert records[0].requested_model == "scripted/usage"
    assert records[0].model is None
    assert recorder.aggregate().usage_complete is False


def test_recording_provider_accepts_complete_usage_from_error_response() -> None:
    recorder = UsageRecorder()

    recorder.record_response(
        LLMResponse(
            content="provider rejected the request",
            finish_reason="error",
            error_classification=ErrorClassification("provider_rejected"),
            usage={
                "prompt_tokens": 9,
                "completion_tokens": 1,
                "total_tokens": 10,
            },
        ),
        model="scripted/error-usage",
    )

    aggregate = recorder.aggregate()
    assert aggregate.calls == 1
    assert aggregate.input_tokens == 9
    assert aggregate.output_tokens == 1
    assert aggregate.total_tokens == 10
    assert aggregate.usage_complete is True


class _CancelledUsageProvider(_UsageProvider):
    async def chat(self, **kwargs) -> LLMResponse:
        raise asyncio.CancelledError


@pytest.mark.asyncio
async def test_recording_provider_records_dispatched_cancellation() -> None:
    recorder = UsageRecorder()
    provider = RecordingProvider(
        _CancelledUsageProvider(),
        recorder=recorder,
    )

    with pytest.raises(asyncio.CancelledError):
        await provider.chat(messages=[], model="scripted/usage")

    records = recorder.records()
    assert len(records) == 1
    assert records[0].provider_dispatched is True
    assert records[0].succeeded is False
    assert recorder.aggregate().calls == 1
    assert recorder.aggregate().usage_complete is False


class _CancelledStreamingUsageProvider(_UsageProvider):
    async def chat_stream(self, **kwargs):
        raise asyncio.CancelledError
        yield


@pytest.mark.asyncio
async def test_recording_provider_records_dispatched_stream_cancellation() -> None:
    recorder = UsageRecorder()
    provider = RecordingProvider(
        _CancelledStreamingUsageProvider(),
        recorder=recorder,
    )

    with pytest.raises(asyncio.CancelledError):
        async for _delta in provider.chat_stream(
            messages=[],
            model="scripted/usage",
        ):
            pass

    records = recorder.records()
    assert len(records) == 1
    assert records[0].provider_dispatched is True
    assert recorder.aggregate().calls == 1
    assert recorder.aggregate().usage_complete is False


class _PostDispatchTaskBudgetProvider(_UsageProvider):
    async def chat(self, **kwargs) -> LLMResponse:
        raise RuntimeError("remote budget signal")

    def classify_error(self, exc, content=None) -> ErrorClassification:
        del exc, content
        return ErrorClassification("task_budget_exhausted")


@pytest.mark.asyncio
async def test_recording_provider_does_not_infer_dispatch_from_category() -> None:
    recorder = UsageRecorder()
    provider = RecordingProvider(
        _PostDispatchTaskBudgetProvider(),
        recorder=recorder,
    )

    with pytest.raises(RuntimeError, match="remote budget signal"):
        await provider.chat(messages=[], model="scripted/usage")

    records = recorder.records()
    assert len(records) == 1
    assert records[0].error_category == "task_budget_exhausted"
    assert records[0].provider_dispatched is True
    assert recorder.aggregate().calls == 1
    assert recorder.aggregate().usage_complete is False


class _StreamingUsageProvider(_UsageProvider):
    async def chat_stream(self, **kwargs):
        yield StreamDelta(
            content="ok",
            model="scripted/actual-stream",
            usage={
                "prompt_tokens": 7,
                "completion_tokens": 2,
                "total_tokens": 9,
            },
        )


@pytest.mark.asyncio
async def test_recording_provider_records_actual_stream_model() -> None:
    recorder = UsageRecorder()
    provider = RecordingProvider(
        _StreamingUsageProvider(),
        recorder=recorder,
    )

    assert [
        delta.content
        async for delta in provider.chat_stream(
            messages=[],
            model="scripted/requested-stream",
        )
    ] == ["ok"]

    record = recorder.records()[0]
    assert record.requested_model == "scripted/requested-stream"
    assert record.model == "scripted/actual-stream"
    assert recorder.aggregate().usage_complete is True


def test_recording_provider_does_not_invent_actual_model() -> None:
    recorder = UsageRecorder()

    record = recorder.record_response(
        LLMResponse(
            content="ok",
            finish_reason="stop",
            usage={
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        ),
        model="scripted/requested-only",
    )

    assert record.requested_model == "scripted/requested-only"
    assert record.model is None


@pytest.mark.asyncio
async def test_recording_provider_preserves_unknown_usage() -> None:
    recorder = UsageRecorder()
    recorder.record_response(
        LLMResponse(content="ok", finish_reason="stop", usage={}),
        model="scripted/unknown",
    )

    aggregate = recorder.aggregate()
    assert aggregate.input_tokens is None
    assert aggregate.output_tokens is None
    assert aggregate.total_tokens is None
    assert aggregate.usage_complete is False


@pytest.mark.asyncio
async def test_recording_provider_preserves_budget_category_through_host(
    tmp_path: Path,
) -> None:
    config = Config(
        agents={
            "defaults": {
                "workspace": str(tmp_path / "workspace"),
                "model": "scripted/budget",
                "max_tokens": 64,
                "max_tool_iterations": 1,
            },
        },
    )
    codeflow_config = CodeFlowConfig(base=config)
    codeflow_config.memory.backend = None
    guarded = BudgetGuardedProvider(
        _BudgetDelegate(),
        ledger=ProviderBudgetLedger(
            tmp_path / "provider-budget.jsonl",
            ProviderBudgetConfig(
                hard_cap_cny=100.0,
                external_service_reserve_cny=0.0,
                max_total_request_attempts=1,
                max_input_tokens_per_call=20_000,
                max_output_tokens_per_call=8_192,
                input_cache_miss_usd_per_million=1.0,
                output_usd_per_million=1.0,
                conservative_usd_to_cny_multiplier=8.0,
            ),
        ),
    )
    recorder = UsageRecorder()
    provider = RecordingProvider(
        guarded,
        recorder=recorder,
    )
    host = await RuntimeTrialHost.build(
        config=config,
        codeflow_config=codeflow_config,
        provider=provider,
        cron_service=None,
        outlet=RecordingOutlet("bench"),
    )

    try:
        with provider_call_budget_scope(
            trial_id="budget-category",
            max_logical_calls=0,
            max_attempts_per_call=1,
        ):
            observation = await host.run(_request())
    finally:
        await host.close()

    assert observation.runtime_state.value == "provider_failed"
    assert observation.failure_category == "task_budget_exhausted"
    assert recorder.has_error_category("task_budget_exhausted")
    assert recorder.records()[0].provider_dispatched is False
    assert recorder.aggregate().calls == 0


@pytest.mark.asyncio
async def test_recording_provider_excludes_predispatch_input_budget_rejection(
    tmp_path: Path,
) -> None:
    guarded = BudgetGuardedProvider(
        _UsageProvider(),
        ledger=ProviderBudgetLedger(
            tmp_path / "provider-budget.jsonl",
            ProviderBudgetConfig(
                hard_cap_cny=100.0,
                external_service_reserve_cny=0.0,
                max_total_request_attempts=2,
                max_input_tokens_per_call=500,
                max_output_tokens_per_call=64,
                input_cache_miss_usd_per_million=1.0,
                output_usd_per_million=1.0,
                conservative_usd_to_cny_multiplier=8.0,
            ),
        ),
    )
    recorder = UsageRecorder()
    provider = RecordingProvider(guarded, recorder=recorder)

    with provider_call_budget_scope(
        trial_id="predispatch-input-budget",
        max_logical_calls=2,
        max_attempts_per_call=1,
    ):
        await provider.chat(
            messages=[],
            model="scripted/usage",
            max_tokens=64,
        )
        with pytest.raises(
            ProviderBudgetError,
            match="estimated input tokens exceed the per-call ceiling",
        ):
            await provider.chat(
                messages=[
                    {
                        "role": "user",
                        "content": "x" * 5_000,
                    },
                ],
                model="scripted/usage",
                max_tokens=64,
            )

    aggregate = recorder.aggregate()
    assert len(recorder.records()) == 2
    assert recorder.records()[1].provider_dispatched is False
    assert recorder.has_error_category("task_budget_exhausted")
    assert aggregate.calls == 1
    assert aggregate.input_tokens == 11
    assert aggregate.output_tokens == 3
    assert aggregate.total_tokens == 14
    assert aggregate.usage_complete is True


@pytest.mark.asyncio
async def test_recording_provider_counts_only_predispatch_rejection_as_zero_usage(
    tmp_path: Path,
) -> None:
    guarded = BudgetGuardedProvider(
        _UsageProvider(),
        ledger=ProviderBudgetLedger(
            tmp_path / "provider-budget.jsonl",
            ProviderBudgetConfig(
                hard_cap_cny=100.0,
                external_service_reserve_cny=0.0,
                max_total_request_attempts=1,
                max_input_tokens_per_call=100,
                max_output_tokens_per_call=64,
                input_cache_miss_usd_per_million=1.0,
                output_usd_per_million=1.0,
                conservative_usd_to_cny_multiplier=8.0,
            ),
        ),
    )
    recorder = UsageRecorder()
    provider = RecordingProvider(guarded, recorder=recorder)

    with provider_call_budget_scope(
        trial_id="only-predispatch-input-budget",
        max_logical_calls=1,
        max_attempts_per_call=1,
    ):
        with pytest.raises(
            ProviderBudgetError,
            match="estimated input tokens exceed the per-call ceiling",
        ):
            await provider.chat(
                messages=[{"role": "user", "content": "x" * 1_000}],
                model="scripted/usage",
                max_tokens=64,
            )

    aggregate = recorder.aggregate()
    assert aggregate.calls == 0
    assert aggregate.input_tokens == 0
    assert aggregate.output_tokens == 0
    assert aggregate.total_tokens == 0
    assert aggregate.usage_complete is True


@pytest.mark.asyncio
async def test_sealed_json_verifier_accepts_valid_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    verifier_root = tmp_path / "sealed"
    verifier_root.mkdir()
    expected = verifier_root / "expected.json"
    expected.write_text(json.dumps({"status": "ready", "count": 3}))
    seal = VerifierSeal.capture(expected)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "result.json").write_text(json.dumps({"status": "ready", "count": 3}))
    verifier = JsonArtifactVerifier(
        expected_path=expected,
        artifact_path="result.json",
    )

    valid = await run_sealed_verifier(verifier, workspace=workspace, seal=seal)
    assert valid.infrastructure_error is None
    assert valid.result.state.value == "passed"

    (workspace / "result.json").write_text(json.dumps({"status": "wrong", "count": 3}))
    invalid = await run_sealed_verifier(verifier, workspace=workspace, seal=seal)
    assert invalid.infrastructure_error is None
    assert invalid.result.state.value == "failed"

    expected.write_text("{}")
    tampered = await run_sealed_verifier(verifier, workspace=workspace, seal=seal)
    assert tampered.infrastructure_error == "verifier_digest_changed"


@pytest.mark.parametrize(
    ("field_name", "path"),
    [
        ("artifact_path", "/tmp/result.json"),
        ("artifact_path", "../result.json"),
        ("forbidden_paths", "/tmp/protected.json"),
        ("forbidden_paths", "../protected.json"),
    ],
)
def test_json_artifact_verifier_rejects_non_relative_or_escaping_paths(
    tmp_path: Path,
    field_name: str,
    path: str,
) -> None:
    expected = tmp_path / "expected.json"
    expected.write_text("{}")
    kwargs = {
        "expected_path": expected,
        "artifact_path": "result.json",
        "forbidden_paths": (),
    }
    if field_name == "artifact_path":
        kwargs["artifact_path"] = path
    else:
        kwargs["forbidden_paths"] = (path,)

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be a normalized relative path",
    ):
        JsonArtifactVerifier(**kwargs)


@pytest.mark.asyncio
async def test_json_artifact_verifier_rejects_resolved_workspace_escapes(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "expected.json"
    expected.write_text("{}")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "result.json").write_text("{}")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "escape").symlink_to(outside, target_is_directory=True)

    artifact_escape = await JsonArtifactVerifier(
        expected_path=expected,
        artifact_path="escape/result.json",
    ).verify(workspace)
    forbidden_escape = await JsonArtifactVerifier(
        expected_path=expected,
        artifact_path="result.json",
        forbidden_paths=("escape/protected.json",),
    ).verify(workspace)

    assert artifact_escape.state is VerificationState.FAILED
    assert artifact_escape.findings == ("artifact_path_outside_workspace:escape/result.json",)
    assert forbidden_escape.state is VerificationState.FAILED
    assert forbidden_escape.findings == ("forbidden_path_outside_workspace:escape/protected.json",)


class _Loop:
    async def run_turn(self, req, emit, drain, *, stream):
        await emit(
            Text(
                content="done",
                source=req.source,
                conversation_id=req.conversation,
            )
        )
        return TurnOutcome(
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            explicit_reply=True,
        )


class _Assembly:
    def __init__(self, loop=None) -> None:
        self.agent_loop = loop or _Loop()
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _request(channel: str = "bench") -> TurnRequest:
    return TurnRequest(
        origin=Origin.USER,
        source=Source(
            channel=channel,
            chat_id="chat",
            sender_id="user",
            chat_type=ChatType.DM,
        ),
        text="run",
        conversation=f"{channel}:chat",
    )


@pytest.mark.asyncio
async def test_runtime_trial_host_enters_through_agent_turn_runner_and_scheduler() -> None:
    assembly = _Assembly()
    outlet = RecordingOutlet("bench")
    host = RuntimeTrialHost(assembly=assembly, outlet=outlet)

    observation = await host.run(_request())
    await host.close()

    assert isinstance(host.runner, AgentTurnRunner)
    assert observation.runtime_state.value == "completed"
    assert observation.delivery_state.value == "delivered"
    assert observation.failure_category is None
    assert [event.content for event in outlet.events if isinstance(event, Text)] == ["done"]
    assert assembly.closed is True


@pytest.mark.asyncio
async def test_runtime_trial_host_distinguishes_dropped_and_no_outlet() -> None:
    dropped_host = RuntimeTrialHost(
        assembly=_Assembly(),
        outlet=RecordingOutlet("bench", fail=True),
        delivery_retries=0,
    )
    dropped = await dropped_host.run(_request())
    await dropped_host.close()

    no_outlet_host = RuntimeTrialHost(
        assembly=_Assembly(),
        outlet=RecordingOutlet("other"),
    )
    no_outlet = await no_outlet_host.run(_request())
    await no_outlet_host.close()

    assert dropped.delivery_state.value == "dropped"
    assert no_outlet.delivery_state.value == "no_outlet"


class _ProviderFailureLoop:
    async def run_turn(self, req, emit, drain, *, stream):
        raise ProviderTurnError("auth")


class _SilentLoop:
    async def run_turn(self, req, emit, drain, *, stream):
        return TurnOutcome(
            usage=Usage(prompt_tokens=1, completion_tokens=0, total_tokens=1),
            explicit_reply=False,
        )


@pytest.mark.asyncio
async def test_runtime_trial_host_classifies_provider_failure_and_missing_delivery() -> None:
    provider_host = RuntimeTrialHost(
        assembly=_Assembly(_ProviderFailureLoop()),
        outlet=RecordingOutlet("bench"),
    )
    provider_failure = await provider_host.run(_request())
    await provider_host.close()

    silent_host = RuntimeTrialHost(
        assembly=_Assembly(_SilentLoop()),
        outlet=RecordingOutlet("bench"),
    )
    silent = await silent_host.run(_request())
    await silent_host.close()

    assert provider_failure.runtime_state.value == "provider_failed"
    assert provider_failure.failure_category == "auth"
    assert provider_failure.delivery_state.value == "dropped"
    assert silent.runtime_state.value == "completed"
    assert silent.failure_category is None
    assert silent.delivery_state.value == "dropped"


@pytest.mark.asyncio
async def test_runtime_trial_host_builds_through_runtime_assembly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assembly = _Assembly()
    calls: list[tuple[Config, CodeFlowConfig]] = []

    def _assemble(config, codeflow_config, **kwargs):
        calls.append((config, codeflow_config))
        return assembly

    monkeypatch.setattr("codeflow.cli._runtime_assembly.assemble_runtime", _assemble)
    config = Config(agents={"defaults": {"workspace": str(tmp_path)}})
    codeflow_config = CodeFlowConfig()

    host = await RuntimeTrialHost.build(
        config=config,
        codeflow_config=codeflow_config,
        provider=object(),
        cron_service=None,
        outlet=RecordingOutlet("bench"),
    )
    await host.close()

    assert calls == [(config, codeflow_config)]
