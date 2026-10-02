from __future__ import annotations

import json
from pathlib import Path

from benchmarks.codeflowbench.packs.tool_mcp import (
    CALIBRATION_TOOL_MCP_TASK_COUNT,
    FORMAL_TOOL_MCP_TASK_COUNT,
    SealedMCPReceiptVerifier,
    ToolMCPPack,
    ToolMCPPairMeasurement,
    ToolMCPTrack,
    assess_tool_mcp_claim,
    catalog_definitions,
    load_tool_mcp_tasks,
    normalize_target_calls,
    reduce_tool_mcp_claim_from_artifacts,
)
from benchmarks.codeflowbench.packs.tool_mcp.runner import (
    _effective_failure_category,
    _RecordingToolMCPProvider,
)
from benchmarks.codeflowbench.packs.tool_mcp.runner import (
    _trial_status as tool_mcp_trial_status,
)
from benchmarks.codeflowbench.plan import compile_plan
from benchmarks.codeflowbench.records import (
    TrialStatus,
    TurnTerminalState,
    VerificationState,
)
from benchmarks.codeflowbench.schema import ExperimentSpec
from codeflow.providers.base import (
    ErrorClassification,
    GenerationSettings,
    LLMResponse,
)
from codeflow.spine import ToolEvent, ToolPhase


def _experiment(tmp_path: Path, *, pack_id: str) -> ExperimentSpec:
    return ExperimentSpec(
        suite=f"{pack_id}-suite",
        repetitions=3,
        pack_ids=(pack_id,),
        output_root=tmp_path,
        identity={
            "codeflow_commit": "0" * 40,
            "model": "scripted/tool-mcp",
            "budget_cap_cny": 100,
        },
    )


def test_tool_mcp_task_budget_exhaustion_is_measurable_timeout() -> None:
    assert (
        tool_mcp_trial_status(
            runtime_state=TurnTerminalState.PROVIDER_FAILED,
            failure_category="task_budget_exhausted",
            verification_state=VerificationState.FAILED,
            mcp_connected=True,
        )
        is TrialStatus.TASK_TIMEOUT
    )


def test_tool_mcp_infrastructure_failure_precedes_task_budget_timeout() -> None:
    assert (
        tool_mcp_trial_status(
            runtime_state=TurnTerminalState.PROVIDER_FAILED,
            failure_category="task_budget_exhausted",
            verification_state=VerificationState.NOT_RUN,
            mcp_connected=False,
        )
        is TrialStatus.INFRASTRUCTURE_FAILURE
    )


def test_tool_mcp_provider_wrapper_preserves_delegate_error_category() -> None:
    class Delegate:
        def classify_error(self, exc=None, content=None):
            del exc, content
            return ErrorClassification("task_budget_exhausted")

    provider = _RecordingToolMCPProvider(
        Delegate(),
        model="provider/exact-model",
        generation=GenerationSettings(),
    )

    assert provider.classify_error(RuntimeError("budget")).category == "task_budget_exhausted"


async def test_tool_mcp_provider_wrapper_does_not_invent_actual_model() -> None:
    class Delegate:
        async def chat(self, **kwargs):
            del kwargs
            return LLMResponse(
                content="done",
                usage={
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            )

        def classify_error(self, exc=None, content=None):
            del exc, content
            return ErrorClassification("unknown")

    provider = _RecordingToolMCPProvider(
        Delegate(),
        model="provider/exact-model",
        generation=GenerationSettings(),
    )

    await provider.chat(messages=[{"role": "user", "content": "test"}])

    assert provider.call_records[0]["requested_model"] == "provider/exact-model"
    assert provider.call_records[0]["model"] is None


def test_tool_mcp_detects_budget_exhaustion_hidden_by_final_synthesis() -> None:
    assert (
        _effective_failure_category(
            None,
            [
                {
                    "finish_reason": "exception",
                    "error_category": "task_budget_exhausted",
                }
            ],
        )
        == "task_budget_exhausted"
    )


def test_tool_mcp_pack_freezes_catalog_tasks_and_single_axis(
    tmp_path: Path,
) -> None:
    catalog = catalog_definitions()
    formal = load_tool_mcp_tasks(ToolMCPTrack.FORMAL)
    calibration = load_tool_mcp_tasks(ToolMCPTrack.CALIBRATION)

    assert len(catalog) == 64
    assert len({tool.name for tool in catalog}) == 64
    assert len({tool.description_prefix for tool in catalog}) == 1
    assert len({tool.parameters_digest for tool in catalog}) == 1
    assert len(formal) == FORMAL_TOOL_MCP_TASK_COUNT == 8
    assert len(calibration) == CALIBRATION_TOOL_MCP_TASK_COUNT == 4
    assert {task.task_id for task in formal}.isdisjoint(task.task_id for task in calibration)
    assert all(1 <= len(task.targets) <= 3 for task in formal + calibration)
    for task in formal + calibration:
        for target in task.targets:
            assert target.tool_name.removeprefix("catalog_probe_") in task.prompt
            assert target.arguments["resource"] in task.prompt
            assert target.arguments["operation"] in task.prompt
            assert str(target.arguments["value"]) in task.prompt

    pack = ToolMCPPack()
    definition = pack.definition()
    assert definition.pack_id == "tool-mcp"
    assert [dict(variant.settings) for variant in definition.variants] == [
        {"tool_disclosure": "all_tools"},
        {"tool_disclosure": "progressive_disclosure"},
    ]
    assert definition.pairs[0].treatment_axis == "tool_disclosure"
    assert definition.identity["claim_reducer"] == "tool_mcp_v1"
    assert definition.identity["mcp_transport"] == "stdio"
    assert definition.identity["mcp_catalog_size"] == 64

    plan = compile_plan(
        _experiment(tmp_path, pack_id=definition.pack_id),
        (pack,),
    )
    assert len(plan.trials) == 8 * 2 * 3 == 48
    assert len(plan.pairs) == 8 * 3 == 24


def test_target_call_normalization_keeps_invalid_unknown_and_exact_repeats() -> None:
    target = catalog_definitions()[0].runtime_name
    visible = frozenset()
    events = (
        ToolEvent(
            phase=ToolPhase.START,
            tool_call_id="direct-1",
            name=target,
            arguments={"resource": "alpha", "operation": "inspect", "value": 1},
        ),
        ToolEvent(
            phase=ToolPhase.COMPLETE,
            tool_call_id="direct-1",
            name=target,
            result_preview='{"receipt":"ok"}',
        ),
        ToolEvent(
            phase=ToolPhase.START,
            tool_call_id="direct-1",
            name=target,
            arguments={"operation": "inspect", "resource": "alpha", "value": 1},
        ),
        ToolEvent(
            phase=ToolPhase.COMPLETE,
            tool_call_id="direct-1",
            name=target,
            result_preview='{"receipt":"ok"}',
        ),
        ToolEvent(
            phase=ToolPhase.START,
            tool_call_id="invalid-envelope",
            name="tool_call",
            arguments={"name": target, "arguments": "not-json"},
        ),
        ToolEvent(
            phase=ToolPhase.COMPLETE,
            tool_call_id="invalid-envelope",
            name="tool_call",
            failed=True,
            result_preview="Error: 'arguments' must be a JSON object.",
        ),
        ToolEvent(
            phase=ToolPhase.START,
            tool_call_id="unknown",
            name="tool_call",
            arguments={
                "name": "mcp_codeflowbench_catalog_probe_99",
                "arguments": {},
            },
        ),
        ToolEvent(
            phase=ToolPhase.COMPLETE,
            tool_call_id="unknown",
            name="tool_call",
            failed=True,
            result_preview="Error: tool not found",
        ),
        ToolEvent(
            phase=ToolPhase.START,
            tool_call_id="search",
            name="tool_search",
            arguments={"query": target},
        ),
        ToolEvent(
            phase=ToolPhase.COMPLETE,
            tool_call_id="search",
            name="tool_search",
            result_preview="[]",
        ),
    )

    result = normalize_target_calls(
        events,
        catalog_names=frozenset(tool.runtime_name for tool in catalog_definitions()),
        initially_visible_names=visible,
        expected_first_target=target,
    )

    assert len(result.records) == 4
    assert result.records[0].route == "direct_hidden"
    assert result.records[0].canonical_key == result.records[1].canonical_key
    assert result.first_target_accuracy == 1.0
    assert result.invalid_target_call_rate == 0.5
    assert result.exact_target_repeat_rate == 0.25
    assert result.meta_tool_invocations == {"tool_call": 2, "tool_search": 1}


def _positive_measurements() -> tuple[ToolMCPPairMeasurement, ...]:
    return tuple(
        ToolMCPPairMeasurement(
            task_id=task.task_id,
            repetition=repetition,
            control_passed=True,
            treatment_passed=True,
            control_schema_tokens=1_000,
            treatment_schema_tokens=300,
            control_invalid_target_call_rate=0.0,
            treatment_invalid_target_call_rate=0.0,
            control_exact_target_repeat_rate=0.0,
            treatment_exact_target_repeat_rate=0.0,
            usage_complete=True,
            valid=True,
            initial_visible_sets_differ=True,
            control_mcp_connected=True,
            treatment_mcp_connected=True,
        )
        for task in load_tool_mcp_tasks(ToolMCPTrack.FORMAL)
        for repetition in range(3)
    )


def test_tool_mcp_claim_uses_equal_task_macro_and_rejects_null_rates() -> None:
    positive = assess_tool_mcp_claim(_positive_measurements())

    assert positive.claim_eligible is True
    assert positive.covered_tasks == 8
    assert positive.tasks_with_lower_schema_tokens == 8
    assert positive.equal_task_macro_reduction == 0.7

    first = _positive_measurements()[0]
    null_rate = ToolMCPPairMeasurement(
        **{
            **first.__dict__,
            "treatment_invalid_target_call_rate": None,
        }
    )
    ineligible = assess_tool_mcp_claim((null_rate, *_positive_measurements()[1:]))

    assert ineligible.claim_eligible is False
    assert "null_target_call_rate" in ineligible.findings


def test_sealed_mcp_verifier_rejects_missing_altered_and_drifted_evidence(
    tmp_path: Path,
) -> None:
    task = load_tool_mcp_tasks(ToolMCPTrack.FORMAL)[0]
    receipt_path = tmp_path / "receipts.jsonl"
    verifier = SealedMCPReceiptVerifier.capture(task)

    missing, _ = verifier.verify(receipt_path)
    assert missing.state is VerificationState.FAILED
    assert missing.findings == ("missing_mcp_receipt_log",)

    receipt_path.write_text('{"receipt":"altered"}\n', encoding="utf-8")
    altered, _ = verifier.verify(receipt_path)
    assert altered.state is VerificationState.FAILED
    assert altered.findings == ("expected_mcp_receipt_missing",)

    drifted = SealedMCPReceiptVerifier(
        task=task,
        expected_data_digest="0" * 64,
        verifier_code_digest=verifier.verifier_code_digest,
    )
    not_run, _ = drifted.verify(receipt_path)
    assert not_run.state is VerificationState.NOT_RUN
    assert not_run.findings == ("mcp_expected_data_digest_changed",)


def test_sealed_mcp_verifier_rejects_unexpected_extra_receipt(
    tmp_path: Path,
) -> None:
    tasks = load_tool_mcp_tasks(ToolMCPTrack.FORMAL)
    task = tasks[0]
    receipts = (
        task.targets[0].expected_receipt,
        tasks[1].targets[0].expected_receipt,
    )
    receipt_path = tmp_path / "receipts.jsonl"
    receipt_path.write_text(
        "".join(f"{json.dumps(receipt)}\n" for receipt in receipts),
        encoding="utf-8",
    )

    result, observed = SealedMCPReceiptVerifier.capture(task).verify(receipt_path)

    assert result.state is VerificationState.FAILED
    assert result.findings == ("unexpected_mcp_receipt",)
    assert result.metrics == {
        "expected_receipt_count": 1,
        "observed_receipt_count": 2,
    }
    assert observed == receipts


def test_sealed_mcp_verifier_rejects_code_drift_bad_json_and_non_object_receipts(
    tmp_path: Path,
) -> None:
    task = load_tool_mcp_tasks(ToolMCPTrack.FORMAL)[0]
    verifier = SealedMCPReceiptVerifier.capture(task)
    receipt_path = tmp_path / "receipts.jsonl"
    receipt_path.write_text(
        json.dumps(task.targets[0].expected_receipt) + "\n",
        encoding="utf-8",
    )
    drifted = SealedMCPReceiptVerifier(
        task=task,
        expected_data_digest=verifier.expected_data_digest,
        verifier_code_digest="0" * 64,
    )

    code_drift, observed = drifted.verify(receipt_path)

    assert code_drift.state is VerificationState.NOT_RUN
    assert code_drift.findings == ("mcp_verifier_code_digest_changed",)
    assert observed == ()

    receipt_path.write_text('{"receipt":\n', encoding="utf-8")
    bad_json, observed = verifier.verify(receipt_path)

    assert bad_json.state is VerificationState.NOT_RUN
    assert bad_json.findings == ("mcp_receipt_verifier_error:JSONDecodeError",)
    assert observed == ()

    receipt_path.write_text('["not-an-object"]\n', encoding="utf-8")
    non_object, observed = verifier.verify(receipt_path)

    assert non_object.state is VerificationState.FAILED
    assert non_object.findings == ("invalid_mcp_receipt_shape",)
    assert observed == (["not-an-object"],)


def test_tool_mcp_artifact_reducer_reconstructs_and_enforces_claim_inputs() -> None:
    trials, pairs = _positive_artifacts()

    positive = reduce_tool_mcp_claim_from_artifacts(
        trial_records=trials,
        pair_results=pairs,
    )

    assert positive["tool_mcp.measurement_valid"] is True
    assert positive["tool_mcp.positive_claim_eligible"] is True
    assert positive["tool_mcp.covered_tasks"] == 8
    assert positive["tool_mcp.equal_task_macro_schema_token_reduction_percent"] == 70.0

    broken_trials = [dict(record) for record in trials]
    treatment = next(record for record in broken_trials if record["key"]["variant_id"] == "tool-mcp-progressive")
    treatment["metrics"] = {
        **treatment["metrics"],
        "invalid_target_call_rate": None,
        "initial_visible_tool_names": treatment["metrics"]["initial_visible_tool_names"],
    }
    ineligible = reduce_tool_mcp_claim_from_artifacts(
        trial_records=broken_trials,
        pair_results=pairs,
    )

    assert ineligible["tool_mcp.positive_claim_eligible"] is False
    assert "null_target_call_rate" in ineligible["tool_mcp.findings"]

    calibration_trials, calibration_pairs = _positive_artifacts(
        track=ToolMCPTrack.CALIBRATION,
        pack_id="tool-mcp-calibration",
        repetitions=2,
    )
    calibration = reduce_tool_mcp_claim_from_artifacts(
        trial_records=calibration_trials,
        pair_results=calibration_pairs,
    )

    assert calibration["tool_mcp.measurement_valid"] is True
    assert calibration["tool_mcp.positive_claim_eligible"] is False
    assert calibration["tool_mcp.pair_measurement_count"] == 8


def test_tool_mcp_artifact_reducer_rejects_missing_actual_model() -> None:
    trials, pairs = _positive_artifacts()
    trial = trials[0]
    trial["metrics"]["model_call_records"][0]["model"] = None
    trial["metrics"]["actual_model_names"] = []

    result = reduce_tool_mcp_claim_from_artifacts(
        trial_records=trials,
        pair_results=pairs,
    )

    assert result["tool_mcp.measurement_valid"] is False
    assert result["tool_mcp.positive_claim_eligible"] is False
    assert any(finding.startswith("missing_tool_mcp_actual_model:") for finding in result["tool_mcp.findings"])


def test_tool_mcp_artifact_reducer_rejects_actual_model_drift() -> None:
    trials, pairs = _positive_artifacts()
    trial = trials[0]
    trial["metrics"]["model_call_records"][0]["model"] = "other/provider-model"
    trial["metrics"]["actual_model_names"] = ["other/provider-model"]

    result = reduce_tool_mcp_claim_from_artifacts(
        trial_records=trials,
        pair_results=pairs,
    )

    assert result["tool_mcp.measurement_valid"] is False
    assert result["tool_mcp.positive_claim_eligible"] is False
    assert any(finding.startswith("tool_mcp_actual_model_drift:") for finding in result["tool_mcp.findings"])


def test_tool_mcp_artifact_reducer_rejects_actual_model_summary_drift() -> None:
    trials, pairs = _positive_artifacts()
    trials[0]["metrics"]["actual_model_names"] = ["provider/exact-model"]

    result = reduce_tool_mcp_claim_from_artifacts(
        trial_records=trials,
        pair_results=pairs,
    )

    assert result["tool_mcp.measurement_valid"] is False
    assert any(finding.startswith("tool_mcp_actual_model_names_drift:") for finding in result["tool_mcp.findings"])


def test_tool_mcp_artifact_reducer_allows_two_distributed_operational_pairs() -> None:
    trials, pairs = _positive_artifacts()
    for pair in (pairs[0], pairs[3]):
        _mark_tool_pair_provider_failure(trials, pair)

    result = reduce_tool_mcp_claim_from_artifacts(
        trial_records=trials,
        pair_results=pairs,
    )

    assert result["tool_mcp.measurement_valid"] is True
    assert result["tool_mcp.coverage_valid"] is True
    assert result["tool_mcp.valid_pair_measurement_count"] == 22


def test_tool_mcp_calibration_requires_all_eight_pairs() -> None:
    trials, pairs = _positive_artifacts(
        track=ToolMCPTrack.CALIBRATION,
        pack_id="tool-mcp-calibration",
        repetitions=2,
    )
    _mark_tool_pair_provider_failure(trials, pairs[0])

    result = reduce_tool_mcp_claim_from_artifacts(
        trial_records=trials,
        pair_results=pairs,
    )

    assert result["tool_mcp.measurement_valid"] is False
    assert result["tool_mcp.coverage_valid"] is False
    assert result["tool_mcp.valid_pair_measurement_count"] == 7


def _mark_tool_pair_provider_failure(
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


def _positive_artifacts(
    *,
    track: ToolMCPTrack = ToolMCPTrack.FORMAL,
    pack_id: str = "tool-mcp",
    repetitions: int = 3,
) -> tuple[list[dict], list[dict]]:
    trials: list[dict] = []
    pairs: list[dict] = []
    for task in load_tool_mcp_tasks(track):
        for repetition in range(repetitions):
            selected_attempt = repetition + 1
            for variant_id, schema_tokens, visible_names in (
                (
                    "tool-mcp-all-tools",
                    1_000,
                    ["mcp_codeflowbench_catalog_probe_00"],
                ),
                (
                    "tool-mcp-progressive",
                    300,
                    ["tool_call", "tool_search"],
                ),
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
                            "trial_total_estimated_visible_tool_schema_tokens": schema_tokens,
                            "invalid_target_call_rate": 0.0,
                            "exact_target_repeat_rate": 0.0,
                            "usage_complete": True,
                            "initial_visible_tool_names": visible_names,
                            "mcp_transport": "stdio",
                            "mcp_connected": True,
                            "mcp_catalog_count": 64,
                            "mcp_catalog_digest": ToolMCPPack().definition().identity["mcp_catalog_digest"],
                            "mcp_catalog_source_digest": ToolMCPPack().definition().identity["mcp_catalog_digest"],
                            "provider_model": "provider/exact-model",
                            "actual_model_names": ["exact-model"],
                            "model_call_records": [
                                {
                                    "requested_model": "provider/exact-model",
                                    "model": "exact-model",
                                    "finish_reason": "stop",
                                }
                            ],
                        },
                    }
                )
            pairs.append(
                {
                    "key": {
                        "pack_id": pack_id,
                        "treatment_axis": "tool_disclosure",
                        "task_id": task.task_id,
                        "repetition": repetition,
                        "control_variant_id": "tool-mcp-all-tools",
                        "treatment_variant_id": "tool-mcp-progressive",
                    },
                    "selected_block_attempt": selected_attempt,
                    "valid": True,
                    "actual_variant_diff": {
                        "tool_disclosure": [
                            "all_tools",
                            "progressive_disclosure",
                        ]
                    },
                }
            )
    return trials, pairs
