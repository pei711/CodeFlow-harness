from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.codeflowbench.packs.context import (
    CALIBRATION_CONTEXT_TASK_COUNT,
    CONTEXT_BENCHMARK_CURATOR_MAX_STEPS,
    CONTEXT_BENCHMARK_OUTPUT_TOKENS,
    CONTEXT_BENCHMARK_PROTECT_FIRST_N,
    FORMAL_CONTEXT_TASK_COUNT,
    ContextPack,
    ContextPairMeasurement,
    ContextTrack,
    FifoHistoryManager,
    FullHistoryManager,
    SealedContextTaskVerifier,
    assess_context_claim,
    load_context_tasks,
    reduce_context_artifacts,
)
from benchmarks.codeflowbench.packs.context.runner import (
    _RoleScopedProvider,
    _trial_configs,
)
from benchmarks.codeflowbench.packs.context.runner import (
    _trial_status as context_trial_status,
)
from benchmarks.codeflowbench.plan import compile_plan
from benchmarks.codeflowbench.protocol import TrialContext, TrialExecution
from benchmarks.codeflowbench.records import (
    DeliveryOutcome,
    TrialKey,
    TrialStatus,
    TurnTerminalState,
    VerificationState,
    VerifierResult,
)
from benchmarks.codeflowbench.schema import ExperimentSpec
from codeflow.config.codeflow import CodeFlowConfig
from codeflow.config.schema import Config
from codeflow.context_engine.base import AssembledPrefix, AssemblyContext
from codeflow.context_engine.segments.curator import CuratorSegmentBuilder
from codeflow.memory_engine.base import TokenBudget
from codeflow.providers.base import ErrorClassification, GenerationSettings


def _experiment(
    tmp_path: Path,
    *,
    pack_id: str,
    repetitions: int,
) -> ExperimentSpec:
    return ExperimentSpec(
        suite=f"{pack_id}-suite",
        repetitions=repetitions,
        pack_ids=(pack_id,),
        output_root=tmp_path,
        identity={
            "codeflow_commit": "0" * 40,
            "model": "scripted/context",
            "budget_cap_cny": 100,
        },
    )


def _message_text(messages: list[dict]) -> str:
    return "\n".join(str(message.get("content", "")) for message in messages)


def test_context_task_budget_exhaustion_is_measurable_timeout() -> None:
    assert (
        context_trial_status(
            runtime_state=TurnTerminalState.PROVIDER_FAILED,
            failure_category="task_budget_exhausted",
            verification_state=VerificationState.FAILED,
            verifier_infrastructure_error=None,
        )
        is TrialStatus.TASK_TIMEOUT
    )


def test_context_role_wrapper_preserves_delegate_error_category() -> None:
    class Delegate:
        generation = GenerationSettings()

        def classify_error(self, exc=None, content=None):
            del exc, content
            return ErrorClassification("task_budget_exhausted")

    provider = _RoleScopedProvider(Delegate(), "context_auxiliary")

    assert provider.classify_error(RuntimeError("budget")).category == "task_budget_exhausted"


def test_context_trial_binds_curator_to_frozen_campaign_model(
    tmp_path: Path,
) -> None:
    config = Config()
    codeflow_config = CodeFlowConfig(base=config)
    codeflow_config.context.curator_model = "gemini-2.5-flash"

    trial_config, trial_codeflow = _trial_configs(
        config,
        codeflow_config,
        workspace=tmp_path,
        model="deepseek/deepseek-v4-flash",
    )

    assert trial_codeflow.context.curator_model == "deepseek/deepseek-v4-flash"
    assert trial_config.agents.defaults.max_tokens == 500
    assert trial_codeflow.context.protect_first_n == 1


def test_context_pack_freezes_prompt_budget_and_initial_protection() -> None:
    identity = ContextPack().definition().identity

    assert CONTEXT_BENCHMARK_OUTPUT_TOKENS == 500
    assert CONTEXT_BENCHMARK_PROTECT_FIRST_N == 1
    assert identity["reserved_output_tokens"] == 500
    assert identity["protected_initial_turns"] == 1


def test_context_pack_freezes_benchmark_curator_steps_only() -> None:
    assert CONTEXT_BENCHMARK_CURATOR_MAX_STEPS == 4
    assert ContextPack().definition().identity["curator_max_steps"] == 4
    assert CuratorSegmentBuilder.__init__.__defaults__[-2] == 12


def test_context_task_sets_are_frozen_and_disjoint() -> None:
    formal = load_context_tasks(ContextTrack.FORMAL)
    calibration = load_context_tasks(ContextTrack.CALIBRATION)

    assert len(formal) == FORMAL_CONTEXT_TASK_COUNT == 8
    assert len(calibration) == CALIBRATION_CONTEXT_TASK_COUNT == 4
    assert {task.task_id for task in formal}.isdisjoint(task.task_id for task in calibration)
    assert len({task.history_digest for task in formal + calibration}) == 12


@pytest.mark.parametrize(
    "track",
    [ContextTrack.FORMAL, ContextTrack.CALIBRATION],
)
def test_every_context_task_materializes_required_long_session_shape(
    track: ContextTrack,
) -> None:
    for task in load_context_tasks(track):
        history = task.materialize_history()
        content = _message_text(history)
        tool_results = [message for message in history if message.get("role") == "tool"]

        assert 30 <= len(history) <= 80
        assert task.early_constraint in content
        assert task.superseded_before in content
        assert task.superseded_after in content
        assert content.index(task.superseded_before) < content.index(task.superseded_after)
        assert len(tool_results) >= 2
        assert max(len(str(result["content"])) for result in tool_results) >= 800
        assert any("[irrelevant-noise:" in str(message.get("content", "")) for message in history)
        assert task.expected_path.is_file()
        assert set(task.constraint_keys).isdisjoint(task.decision_keys)
        assert set(task.constraint_keys) | set(task.decision_keys) == set(
            json.loads(task.expected_path.read_text(encoding="utf-8")),
        )
        assert task.history_digest == task.compute_history_digest()


def test_context_pack_freezes_only_fifo_vs_curator_history_manager_axis(
    tmp_path: Path,
) -> None:
    formal_pack = ContextPack(ContextTrack.FORMAL)
    calibration_pack = ContextPack(ContextTrack.CALIBRATION)

    formal = formal_pack.definition()
    calibration = calibration_pack.definition()

    assert formal.pack_id == "context"
    assert calibration.pack_id == "context-calibration"
    assert [dict(variant.settings) for variant in formal.variants] == [
        {"history_manager": "fifo_tail"},
        {"history_manager": "curator"},
    ]
    assert formal.pairs[0].treatment_axis == "history_manager"
    assert formal.pairs[0].control_variant_id == "context-fifo"
    assert formal.pairs[0].treatment_variant_id == "context-curator"
    assert formal.identity["result_scope"] == "exploratory_eight_task_pack"

    formal_plan = compile_plan(
        _experiment(tmp_path, pack_id=formal.pack_id, repetitions=3),
        (formal_pack,),
    )
    calibration_plan = compile_plan(
        _experiment(
            tmp_path,
            pack_id=calibration.pack_id,
            repetitions=2,
        ),
        (calibration_pack,),
    )
    assert len(formal_plan.trials) == 8 * 2 * 3 == 48
    assert len(formal_plan.pairs) == 8 * 3 == 24
    assert len(calibration_plan.trials) == 4 * 2 * 2 == 16
    assert len(calibration_plan.pairs) == 4 * 2 == 8


class _TokenCounter:
    def estimate_prompt_tokens(self, messages, tools, model):
        del tools, model
        return (
            sum(len(str(message.get("content", ""))) for message in messages),
            "test_counter",
        )


def _assembly_context(history: list[dict]) -> AssemblyContext:
    return AssemblyContext(
        session_key="context-test",
        current_message="produce the final artifact",
        media=None,
        channel="bench",
        chat_id="context-test",
        session_messages=history,
        budget=TokenBudget(
            context_length=720,
            reserved_output=120,
            reserved_tools=0,
            reserved_system=40,
            available_history=560,
        ),
        prefix=AssembledPrefix(
            system_prefix="system",
            user_message={"role": "user", "content": "produce the final artifact"},
            tool_defs=[],
        ),
    )


@pytest.mark.asyncio
async def test_fifo_history_manager_keeps_budgeted_tail_not_early_history() -> None:
    history = [
        {"role": "user", "content": f"turn-{index}-" + ("x" * 90)}
        if index % 2 == 0
        else {"role": "assistant", "content": f"ack-{index}-" + ("y" * 90)}
        for index in range(12)
    ]
    manager = FifoHistoryManager(
        provider=_TokenCounter(),
        model="scripted/context",
        get_tool_definitions=lambda: [],
        context_window_tokens=720,
    )

    segment = await manager.build(_assembly_context(history))

    assert segment is not None
    assert segment.meta["path"] == "fifo_tail"
    assert segment.meta["history_manager"] == "fifo_tail"
    assert "turn-0-" not in _message_text(segment.history or [])
    assert "ack-11-" in _message_text(segment.history or [])
    assert segment.meta["estimated_prompt_tokens"] <= 600


@pytest.mark.asyncio
async def test_full_history_manager_is_exploratory_and_never_a_formal_variant() -> None:
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
    ]
    manager = FullHistoryManager()

    segment = await manager.build(_assembly_context(history))
    formal = ContextPack(ContextTrack.FORMAL).definition()

    assert segment is not None
    assert segment.history == history
    assert segment.meta["path"] == "full_history"
    assert "full_history" not in {variant.settings["history_manager"] for variant in formal.variants}


@pytest.mark.parametrize(
    "task",
    load_context_tasks(ContextTrack.FORMAL),
    ids=lambda task: task.task_id,
)
@pytest.mark.asyncio
async def test_every_formal_context_verifier_accepts_known_valid_artifact(
    task,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    artifact = workspace / task.artifact_path
    artifact.parent.mkdir(parents=True)
    artifact.write_text(task.expected_path.read_text(encoding="utf-8"), encoding="utf-8")
    verifier = SealedContextTaskVerifier.capture(task)

    execution = await verifier.verify(workspace)

    assert execution.infrastructure_error is None
    assert execution.result.state.value == "passed"


@pytest.mark.parametrize(
    "task",
    load_context_tasks(ContextTrack.FORMAL),
    ids=lambda task: task.task_id,
)
@pytest.mark.asyncio
async def test_every_formal_context_verifier_rejects_missing_altered_and_forbidden(
    task,
    tmp_path: Path,
) -> None:
    verifier = SealedContextTaskVerifier.capture(task)

    missing_workspace = tmp_path / "missing"
    missing_workspace.mkdir()
    missing = await verifier.verify(missing_workspace)

    altered_workspace = tmp_path / "altered"
    altered_artifact = altered_workspace / task.artifact_path
    altered_artifact.parent.mkdir(parents=True)
    altered_artifact.write_text(json.dumps({"tampered": True}), encoding="utf-8")
    altered = await verifier.verify(altered_workspace)

    forbidden_workspace = tmp_path / "forbidden"
    forbidden_artifact = forbidden_workspace / task.artifact_path
    forbidden_artifact.parent.mkdir(parents=True)
    forbidden_artifact.write_text(
        task.expected_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    forbidden = forbidden_workspace / task.forbidden_paths[0]
    forbidden.parent.mkdir(parents=True)
    forbidden.write_text("mutation", encoding="utf-8")
    forbidden_result = await verifier.verify(forbidden_workspace)

    assert missing.infrastructure_error is None
    assert missing.result.state.value == "failed"
    assert altered.infrastructure_error is None
    assert altered.result.state.value == "failed"
    assert forbidden_result.infrastructure_error is None
    assert forbidden_result.result.state.value == "failed"


@pytest.mark.parametrize(
    "task",
    load_context_tasks(ContextTrack.FORMAL),
    ids=lambda task: task.task_id,
)
@pytest.mark.asyncio
async def test_every_formal_context_verifier_reports_own_tampering_as_infrastructure(
    task,
    tmp_path: Path,
) -> None:
    expected_copy = tmp_path / f"{task.task_id}-expected.json"
    expected_copy.write_text(task.expected_path.read_text(encoding="utf-8"), encoding="utf-8")
    verifier = SealedContextTaskVerifier.capture(
        task,
        expected_path=expected_copy,
    )
    expected_copy.write_text("{}", encoding="utf-8")

    execution = await verifier.verify(tmp_path / "workspace")

    assert execution.result.state.value == "not_run"
    assert execution.infrastructure_error == "verifier_digest_changed"


@pytest.mark.asyncio
async def test_context_verifier_reports_constraint_and_decision_separately(
    tmp_path: Path,
) -> None:
    task = load_context_tasks(ContextTrack.FORMAL)[0]
    expected = json.loads(task.expected_path.read_text(encoding="utf-8"))
    expected[task.decision_keys[0]] = "stale-value"
    artifact = tmp_path / task.artifact_path
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps(expected), encoding="utf-8")
    verifier = SealedContextTaskVerifier.capture(task)

    metrics = verifier.diagnostic_metrics(tmp_path)

    assert metrics == {
        "artifact_valid_json": True,
        "active_constraint_applied": True,
        "latest_decision_applied": False,
        "artifact_exact": False,
        "forbidden_paths_clean": True,
    }


def _positive_measurements() -> tuple[ContextPairMeasurement, ...]:
    return tuple(
        ContextPairMeasurement(
            task_id=task.task_id,
            repetition=repetition,
            control_passed=True,
            treatment_passed=True,
            control_main_agent_input_tokens=900,
            treatment_main_agent_input_tokens=650,
            control_trial_total_input_tokens=1_000,
            treatment_trial_total_input_tokens=800,
            control_context_auxiliary_input_tokens=0,
            treatment_context_auxiliary_input_tokens=150,
            usage_complete=True,
            valid=True,
        )
        for task in load_context_tasks(ContextTrack.FORMAL)
        for repetition in range(3)
    )


def test_context_claim_uses_equal_task_trial_total_macro_and_is_exploratory() -> None:
    assessment = assess_context_claim(_positive_measurements())

    assert assessment.claim_eligible is True
    assert assessment.covered_tasks == 8
    assert assessment.tasks_with_lower_trial_total == 8
    assert assessment.treatment_passes == assessment.control_passes == 24
    assert assessment.equal_task_macro_reduction == pytest.approx(0.20)
    assert assessment.exploratory is True


def test_context_claim_cannot_hide_curator_auxiliary_usage() -> None:
    measurements = tuple(
        ContextPairMeasurement(
            task_id=measurement.task_id,
            repetition=measurement.repetition,
            control_passed=True,
            treatment_passed=True,
            control_main_agent_input_tokens=900,
            treatment_main_agent_input_tokens=500,
            control_trial_total_input_tokens=1_000,
            treatment_trial_total_input_tokens=1_050,
            control_context_auxiliary_input_tokens=0,
            treatment_context_auxiliary_input_tokens=550,
            usage_complete=True,
            valid=True,
        )
        for measurement in _positive_measurements()
    )

    assessment = assess_context_claim(measurements)

    assert assessment.claim_eligible is False
    assert assessment.equal_task_macro_reduction == pytest.approx(-0.05)
    assert "trial_total_input_reduction_below_15_percent" in assessment.findings


def test_context_artifact_reducer_rebuilds_flat_claim_metrics() -> None:
    trials, pairs = _positive_context_artifacts()

    result = reduce_context_artifacts(
        trial_records=trials,
        pair_results=pairs,
    )

    assert result["context.measurement_valid"] is True
    assert result["context.capability_measurement_valid"] is True
    assert result["context.efficiency_measurement_valid"] is True
    assert result["context.positive_claim_eligible"] is True
    assert result["context.trial_total_input_token_reduction_percent"] == pytest.approx(20.0)
    assert result["context.coverage_valid"] is True
    assert result["context.usage_complete"] is True
    assert result["context.single_axis_valid"] is True
    assert result["context.capability_evidence_complete"] is True
    assert result["context.treatment_capability_score_rate"] == 1.0
    assert result["context.treatment_early_constraint_retained_rate"] == 1.0
    assert result["context.treatment_active_constraint_applied_rate"] == 1.0
    assert result["context.treatment_latest_decision_applied_rate"] == 1.0
    assert result["context.treatment_artifact_exact_rate"] == 1.0
    assert result["context.findings"] == []


def test_context_artifact_reducer_fails_closed_on_attempt_drift() -> None:
    trials, pairs = _positive_context_artifacts()
    pairs[0] = {
        **pairs[0],
        "selected_block_attempt": 2,
    }

    result = reduce_context_artifacts(
        trial_records=trials,
        pair_results=pairs,
    )

    assert result["context.measurement_valid"] is False
    assert result["context.positive_claim_eligible"] is False
    assert any(str(finding).startswith("context_selected_attempt_drift:") for finding in result["context.findings"])


def test_context_artifact_reducer_reports_incomplete_measurable_usage() -> None:
    trials, pairs = _positive_context_artifacts()
    treatment = next(
        trial
        for trial in trials
        if trial["key"]["task_id"] == pairs[0]["key"]["task_id"]
        and trial["key"]["repetition"] == pairs[0]["key"]["repetition"]
        and trial["key"]["variant_id"] == "context-curator"
    )
    treatment["metrics"]["trial_total_input_tokens"] = None
    treatment["metrics"]["context_auxiliary_input_tokens"] = None
    treatment["metrics"]["usage_complete"] = False

    result = reduce_context_artifacts(
        trial_records=trials,
        pair_results=pairs,
    )

    assert result["context.usage_complete"] is False
    assert result["context.valid_pair_measurement_count"] == 23
    assert result["context.measurement_valid"] is False
    assert result["context.capability_measurement_valid"] is True
    assert result["context.efficiency_measurement_valid"] is False
    assert any(str(finding).startswith("context_usage_incomplete:") for finding in result["context.findings"])
    assert not any(
        str(finding).startswith("context_usage_subtotal_exceeds_total:") for finding in result["context.findings"]
    )


def test_context_artifact_reducer_rejects_real_usage_subtotal_overflow() -> None:
    trials, pairs = _positive_context_artifacts()
    treatment = next(
        trial
        for trial in trials
        if trial["key"]["task_id"] == pairs[0]["key"]["task_id"]
        and trial["key"]["repetition"] == pairs[0]["key"]["repetition"]
        and trial["key"]["variant_id"] == "context-curator"
    )
    treatment["metrics"]["main_agent_input_tokens"] = 900
    treatment["metrics"]["context_auxiliary_input_tokens"] = 200
    treatment["metrics"]["trial_total_input_tokens"] = 1_000

    result = reduce_context_artifacts(
        trial_records=trials,
        pair_results=pairs,
    )

    assert result["context.measurement_valid"] is False
    assert any(
        str(finding).startswith(
            "context_usage_subtotal_exceeds_total:treatment:",
        )
        for finding in result["context.findings"]
    )


def test_context_artifact_reducer_allows_two_distributed_operational_pairs() -> None:
    trials, pairs = _positive_context_artifacts()
    for pair in (pairs[0], pairs[3]):
        _mark_context_pair_provider_failure(trials, pair)

    result = reduce_context_artifacts(
        trial_records=trials,
        pair_results=pairs,
    )

    assert result["context.measurement_valid"] is True
    assert result["context.coverage_valid"] is True
    assert result["context.usage_complete"] is True
    assert result["context.valid_pair_measurement_count"] == 22


def test_context_artifact_reducer_rejects_insufficient_formal_coverage() -> None:
    trials, pairs = _positive_context_artifacts()
    for pair in (pairs[0], pairs[3], pairs[6]):
        _mark_context_pair_provider_failure(trials, pair)

    result = reduce_context_artifacts(
        trial_records=trials,
        pair_results=pairs,
    )

    assert result["context.measurement_valid"] is False
    assert result["context.coverage_valid"] is False
    assert result["context.valid_pair_measurement_count"] == 21


def test_context_artifact_reducer_requires_two_pairs_per_task() -> None:
    trials, pairs = _positive_context_artifacts()
    for pair in pairs[:2]:
        _mark_context_pair_provider_failure(trials, pair)

    result = reduce_context_artifacts(
        trial_records=trials,
        pair_results=pairs,
    )

    assert result["context.measurement_valid"] is False
    assert result["context.coverage_valid"] is False
    assert result["context.valid_pair_measurement_count"] == 22


def test_context_calibration_requires_all_eight_pairs() -> None:
    trials, pairs = _positive_context_artifacts(
        track=ContextTrack.CALIBRATION,
        pack_id="context-calibration",
        repetitions=2,
    )
    _mark_context_pair_provider_failure(trials, pairs[0])

    result = reduce_context_artifacts(
        trial_records=trials,
        pair_results=pairs,
    )

    assert result["context.measurement_valid"] is False
    assert result["context.coverage_valid"] is False
    assert result["context.valid_pair_measurement_count"] == 7


def _mark_context_pair_provider_failure(
    trials: list[dict],
    pair: dict,
) -> None:
    pair["valid"] = False
    key = pair["key"]
    for trial in trials:
        trial_key = trial["key"]
        if (
            trial_key["pack_id"] == key["pack_id"]
            and trial_key["task_id"] == key["task_id"]
            and trial_key["repetition"] == key["repetition"]
        ):
            trial["status"] = "provider_failure"
            trial["metrics"] = {}


def _positive_context_artifacts(
    *,
    track: ContextTrack = ContextTrack.FORMAL,
    pack_id: str = "context",
    repetitions: int = 3,
) -> tuple[list[dict], list[dict]]:
    trials: list[dict] = []
    pairs: list[dict] = []
    for task in load_context_tasks(track):
        for repetition in range(repetitions):
            selected_attempt = repetition + 1
            for variant_id, main_tokens, total_tokens, auxiliary_tokens in (
                ("context-fifo", 900, 1_000, 0),
                ("context-curator", 650, 800, 150),
            ):
                trials.append(
                    {
                        "key": {
                            "pack_id": pack_id,
                            "task_id": task.task_id,
                            "variant_id": variant_id,
                            "repetition": repetition,
                        },
                        "selected_block_attempt": selected_attempt,
                        "status": "passed",
                        "metrics": {
                            "main_agent_input_tokens": main_tokens,
                            "trial_total_input_tokens": total_tokens,
                            "context_auxiliary_input_tokens": (auxiliary_tokens),
                            "usage_complete": True,
                            "early_constraint_retained": True,
                            "active_constraint_applied": True,
                            "latest_decision_applied": True,
                            "artifact_exact": True,
                        },
                    }
                )
            pairs.append(
                {
                    "key": {
                        "pack_id": pack_id,
                        "treatment_axis": "history_manager",
                        "task_id": task.task_id,
                        "repetition": repetition,
                        "control_variant_id": "context-fifo",
                        "treatment_variant_id": "context-curator",
                    },
                    "selected_block_attempt": selected_attempt,
                    "valid": True,
                    "actual_variant_diff": {
                        "history_manager": {
                            "control": "fifo_tail",
                            "treatment": "curator",
                        },
                    },
                }
            )
    return trials, pairs


@pytest.mark.asyncio
async def test_context_pack_resolves_factory_and_rejects_observed_axis_drift(
    tmp_path: Path,
) -> None:
    observed_factories: list[str] = []

    async def runner(*, context, task, context_engine_factory):
        del task
        observed_factories.append(context_engine_factory.__name__)
        return TrialExecution(
            status=TrialStatus.PASSED,
            runtime_state=TurnTerminalState.COMPLETED,
            delivery_state=DeliveryOutcome.DELIVERED,
            verification=VerifierResult(state=VerificationState.PASSED),
            observed_variant_settings={"history_manager": "curator"},
            metrics={
                "main_agent_input_tokens": 100,
                "trial_total_input_tokens": 100,
                "context_auxiliary_input_tokens": 0,
                "usage_complete": True,
                "context_path": "fifo_tail",
                "early_constraint_retained": False,
                "artifact_valid_json": True,
                "active_constraint_applied": True,
                "latest_decision_applied": True,
                "artifact_exact": True,
                "forbidden_paths_clean": True,
                "capability_criteria_passed": 3,
                "capability_criteria_total": 4,
                "end_to_end_latency_ms": 1,
            },
        )

    pack = ContextPack(ContextTrack.FORMAL, runner=runner)
    definition = pack.definition()
    experiment = _experiment(
        tmp_path,
        pack_id=definition.pack_id,
        repetitions=1,
    )
    task = definition.tasks[0]
    variant = definition.variants[0]
    context = TrialContext(
        experiment_id="context-axis-test",
        plan_digest="context-axis-test",
        key=TrialKey(
            experiment_id="context-axis-test",
            pack_id=definition.pack_id,
            task_id=task.task_id,
            variant_id=variant.variant_id,
            repetition=0,
        ),
        block_attempt=1,
        experiment=experiment,
        task=task,
        variant=variant,
    )

    execution = await pack.run_trial(context)

    assert observed_factories == ["build_fifo_context_engine"]
    assert execution.status is TrialStatus.INFRASTRUCTURE_FAILURE
    assert execution.findings == ("observed_variant_settings_drift",)
