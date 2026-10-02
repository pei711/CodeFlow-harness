from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.codeflowbench import rebuild_report, run
from benchmarks.codeflowbench.claims import ClaimRuleResult, evaluate_claim_rules
from benchmarks.codeflowbench.coverage import assess_pair_coverage
from benchmarks.codeflowbench.protocol import (
    RetrievalContext,
    RetrievalExecution,
    TrialContext,
    TrialExecution,
)
from benchmarks.codeflowbench.records import (
    DeliveryOutcome,
    RetrievalStatus,
    TrialStatus,
    TurnTerminalState,
    VerificationState,
    VerifierResult,
)
from benchmarks.codeflowbench.registry import PackRegistry
from benchmarks.codeflowbench.report import (
    FullReport,
    _eligible_claim_metrics,
    rebuild_full_report,
)
from benchmarks.codeflowbench.schema import (
    ClaimRule,
    ExecutionPolicy,
    ExperimentSpec,
    PackDefinition,
    PairSpec,
    RetrievalConfigurationSpec,
    RetrievalQuerySpec,
    RetrievalSuiteSpec,
    TaskSpec,
    VariantSpec,
)
from benchmarks.codeflowbench.statistics import clustered_bootstrap_interval


def test_clustered_bootstrap_is_task_level_and_reproducible() -> None:
    per_task = {
        "task-a": (1.0, 1.0, 1.0),
        "task-b": (-1.0, -1.0, -1.0),
        "task-c": (0.5, 0.5, 0.5),
        "task-d": (0.0, 0.0, 0.0),
    }

    first = clustered_bootstrap_interval(
        per_task,
        samples=10_000,
        seed=1729,
    )
    second = clustered_bootstrap_interval(
        per_task,
        samples=10_000,
        seed=1729,
    )

    assert first == second
    assert first.unit == "task"
    assert first.tasks == 4
    assert first.samples == 10_000
    assert first.exploratory is True
    assert first.lower <= first.estimate <= first.upper


def test_claim_rules_keep_complete_negative_experiment_ineligible() -> None:
    rules = (
        ClaimRule(
            rule_id="context-token-reduction",
            metric="context.token_reduction_percent",
            operator="ge",
            threshold=15,
            prerequisites=("context.coverage_valid",),
        ),
    )

    result = evaluate_claim_rules(
        rules,
        metrics={
            "context.token_reduction_percent": 8.5,
            "context.coverage_valid": True,
        },
        ship_complete=True,
        measurement_valid=True,
    )

    assert result.ship_complete is True
    assert result.measurement_valid is True
    assert result.positive_claim_eligible is False
    assert result.rules[0].passed is False
    assert result.rules[0].reason == "threshold_not_met"


def test_claim_rules_fail_closed_on_missing_prerequisite_or_measurement() -> None:
    rules = (
        ClaimRule(
            rule_id="memory-recall",
            metric="memory.recall_at_5",
            operator="ge",
            threshold=0.8,
            prerequisites=("memory.coverage_valid",),
        ),
    )

    missing_prerequisite = evaluate_claim_rules(
        rules,
        metrics={"memory.recall_at_5": 0.9},
        ship_complete=True,
        measurement_valid=True,
    )
    invalid_measurement = evaluate_claim_rules(
        rules,
        metrics={
            "memory.recall_at_5": 0.9,
            "memory.coverage_valid": True,
        },
        ship_complete=True,
        measurement_valid=False,
    )

    assert missing_prerequisite.rules[0].reason == "prerequisite_not_met"
    assert invalid_measurement.positive_claim_eligible is False


def test_cv_metrics_require_every_rule_in_a_capability_group_to_pass() -> None:
    metrics = {
        "memory_retrieval.final_injection_recall_at_5": 0.9,
        "memory_retrieval.irrelevant_injection_rate": 0.2,
        "context.trial_total_input_token_reduction_percent": 18.0,
    }
    report = FullReport(
        experiment_id="group-gate",
        report_digest="digest",
        ship_complete=True,
        measurement_valid=True,
        positive_claim_eligible=False,
        planned_trials=0,
        terminal_trials=0,
        planned_retrieval_cases=0,
        terminal_retrieval_cases=0,
        metrics=metrics,
        pair_summaries=(),
        claim_results=(
            ClaimRuleResult(
                rule_id="memory-recall-at-five",
                metric="memory_retrieval.final_injection_recall_at_5",
                passed=True,
                observed=0.9,
                threshold=0.8,
                reason="passed",
            ),
            ClaimRuleResult(
                rule_id="memory-irrelevant-injection",
                metric="memory_retrieval.irrelevant_injection_rate",
                passed=False,
                observed=0.2,
                threshold=0.05,
                reason="threshold_not_met",
            ),
            ClaimRuleResult(
                rule_id="context-token-reduction",
                metric="context.trial_total_input_token_reduction_percent",
                passed=True,
                observed=18.0,
                threshold=15,
                reason="passed",
            ),
        ),
        selected_status_counts={},
        first_attempt_status_counts={},
        all_attempt_status_counts={},
        retrieval_status_counts={},
        retrieval_first_attempt_status_counts={},
        retrieval_all_attempt_status_counts={},
        findings=(),
    )

    assert _eligible_claim_metrics(report) == {
        "context.trial_total_input_token_reduction_percent": 18.0,
    }


def test_cv_metrics_never_export_fixture_retrieval_claims() -> None:
    metric = "memory_retrieval.final_injection_recall_at_5"
    report = FullReport(
        experiment_id="fixture-retrieval-gate",
        report_digest="digest",
        ship_complete=True,
        measurement_valid=True,
        positive_claim_eligible=True,
        planned_trials=0,
        terminal_trials=0,
        planned_retrieval_cases=80,
        terminal_retrieval_cases=80,
        metrics={
            metric: 1.0,
            "memory_retrieval.real_semantic_claim_eligible": False,
            "evidence.retrieval_level": "deterministic_contract",
        },
        pair_summaries=(),
        claim_results=(
            ClaimRuleResult(
                rule_id="memory-recall-at-five",
                metric=metric,
                passed=True,
                observed=1.0,
                threshold=0.8,
                reason="passed",
            ),
        ),
        selected_status_counts={},
        first_attempt_status_counts={},
        all_attempt_status_counts={},
        retrieval_status_counts={"measurable": 80},
        retrieval_first_attempt_status_counts={"measurable": 80},
        retrieval_all_attempt_status_counts={"measurable": 80},
        findings=(),
    )

    assert _eligible_claim_metrics(report) == {}


class _TokenPack:
    def definition(self) -> PackDefinition:
        return PackDefinition(
            pack_id="context-v1",
            tasks=tuple(TaskSpec(task_id=f"context-{index}") for index in range(8)),
            variants=(
                VariantSpec(
                    variant_id="fifo",
                    settings={"history_manager": "fifo"},
                ),
                VariantSpec(
                    variant_id="curator",
                    settings={"history_manager": "curator"},
                ),
            ),
            pairs=(
                PairSpec(
                    treatment_axis="history_manager",
                    control_variant_id="fifo",
                    treatment_variant_id="curator",
                ),
            ),
        )

    async def run_trial(self, context: TrialContext) -> TrialExecution:
        tokens = 100 if context.variant.variant_id == "fifo" else 50
        return TrialExecution(
            status=TrialStatus.PASSED,
            runtime_state=TurnTerminalState.COMPLETED,
            delivery_state=DeliveryOutcome.DELIVERED,
            verification=VerifierResult(state=VerificationState.PASSED),
            observed_variant_settings=dict(context.variant.settings),
            metrics={"context.trial_total_input_tokens": tokens},
        )


class _OperationalFailureTokenPack(_TokenPack):
    async def run_trial(self, context: TrialContext) -> TrialExecution:
        if context.key.repetition == 0 and context.key.task_id in {
            "context-0",
            "context-1",
        }:
            return TrialExecution(
                status=TrialStatus.PROVIDER_FAILURE,
                runtime_state=TurnTerminalState.PROVIDER_FAILED,
                delivery_state=None,
                verification=VerifierResult(
                    state=VerificationState.NOT_RUN,
                ),
                observed_variant_settings=dict(context.variant.settings),
            )
        return await super().run_trial(context)


class _RetryingTokenPack(_TokenPack):
    async def run_trial(self, context: TrialContext) -> TrialExecution:
        if (
            context.block_attempt == 1
            and context.key.task_id == "context-0"
            and context.key.repetition == 0
            and context.variant.variant_id == "curator"
        ):
            return TrialExecution(
                status=TrialStatus.PROVIDER_FAILURE,
                runtime_state=TurnTerminalState.PROVIDER_FAILED,
                delivery_state=DeliveryOutcome.NO_OUTLET,
                verification=VerifierResult(
                    state=VerificationState.NOT_RUN,
                ),
                observed_variant_settings=dict(context.variant.settings),
            )
        return await super().run_trial(context)


class _DeclaredContextReducerPack:
    def definition(self) -> PackDefinition:
        return PackDefinition(
            pack_id="context-calibration",
            tasks=tuple(TaskSpec(task_id=f"context-calibration-{index}") for index in range(4)),
            variants=(
                VariantSpec(
                    variant_id="context-fifo",
                    settings={"history_manager": "fifo_tail"},
                ),
                VariantSpec(
                    variant_id="context-curator",
                    settings={"history_manager": "codeflow_curator"},
                ),
            ),
            pairs=(
                PairSpec(
                    treatment_axis="history_manager",
                    control_variant_id="context-fifo",
                    treatment_variant_id="context-curator",
                ),
            ),
            identity={"claim_reducer": "context_v1"},
        )

    async def run_trial(self, context: TrialContext) -> TrialExecution:
        curator = context.variant.variant_id == "context-curator"
        return TrialExecution(
            status=TrialStatus.PASSED,
            runtime_state=TurnTerminalState.COMPLETED,
            delivery_state=DeliveryOutcome.DELIVERED,
            verification=VerifierResult(state=VerificationState.PASSED),
            observed_variant_settings=dict(context.variant.settings),
            metrics={
                "main_agent_input_tokens": 50 if curator else 100,
                "trial_total_input_tokens": 60 if curator else 110,
                "context_auxiliary_input_tokens": 10,
                "usage_complete": True,
            },
        )


class _DeclaredContextCapabilityReducerPack(_DeclaredContextReducerPack):
    def definition(self) -> PackDefinition:
        definition = super().definition()
        return replace(
            definition,
            identity={"claim_reducer": "context_v2"},
        )

    async def run_trial(self, context: TrialContext) -> TrialExecution:
        if context.variant.variant_id == "context-fifo":
            return TrialExecution(
                status=TrialStatus.TASK_FAILED,
                runtime_state=TurnTerminalState.COMPLETED,
                delivery_state=DeliveryOutcome.DELIVERED,
                verification=VerifierResult(state=VerificationState.FAILED),
                observed_variant_settings=dict(context.variant.settings),
            )
        return await super().run_trial(context)


class _UnknownDeclaredReducerPack:
    def definition(self) -> PackDefinition:
        return PackDefinition(
            pack_id="unknown-reducer-pack",
            tasks=(TaskSpec(task_id="unknown-reducer-task"),),
            variants=(
                VariantSpec(
                    variant_id="unknown-reducer-variant",
                    settings={"axis": "control"},
                ),
            ),
            pairs=(),
            identity={"claim_reducer": "unknown_v1"},
        )

    async def run_trial(self, context: TrialContext) -> TrialExecution:
        return TrialExecution(
            status=TrialStatus.PASSED,
            runtime_state=TurnTerminalState.COMPLETED,
            delivery_state=DeliveryOutcome.DELIVERED,
            verification=VerifierResult(state=VerificationState.PASSED),
            observed_variant_settings=dict(context.variant.settings),
        )


class _RetrievalEvidencePack:
    def definition(self) -> PackDefinition:
        return PackDefinition(
            pack_id="retrieval-evidence",
            tasks=(),
            variants=(),
            pairs=(),
            retrieval_suites=(
                RetrievalSuiteSpec(
                    retrieval_suite_id="retrieval-evidence-suite",
                    queries=(
                        RetrievalQuerySpec(
                            query_id="retrieval-evidence-query",
                            label="positive",
                            expected_item_ids=("expected-item",),
                        ),
                    ),
                    configurations=(
                        RetrievalConfigurationSpec(
                            configuration_id="retrieval-evidence-config",
                            settings={"source": "fixture"},
                        ),
                    ),
                    corpus_digest="a" * 64,
                    query_labels_digest="b" * 64,
                ),
            ),
        )

    async def run_trial(self, context: TrialContext) -> TrialExecution:
        raise AssertionError(f"unexpected Trial: {context.key}")

    async def run_retrieval_case(
        self,
        context: RetrievalContext,
    ) -> RetrievalExecution:
        return RetrievalExecution(
            status=RetrievalStatus.MEASURABLE,
            ranked_results=({"item_id": "unrelated-item", "score": 0.5},),
            injected_results=(),
            usage={"input_tokens": 10},
        )


class _MultiQueryRetrievalPack(_RetrievalEvidencePack):
    def definition(self) -> PackDefinition:
        return PackDefinition(
            pack_id="multi-query-retrieval",
            tasks=(),
            variants=(),
            pairs=(),
            retrieval_suites=(
                RetrievalSuiteSpec(
                    retrieval_suite_id="multi-query-suite",
                    queries=tuple(
                        RetrievalQuerySpec(
                            query_id=f"multi-query-{index}",
                            label="positive",
                            expected_item_ids=(f"expected-item-{index}",),
                        )
                        for index in range(2)
                    ),
                    configurations=(
                        RetrievalConfigurationSpec(
                            configuration_id="fixture",
                            settings={"source": "fixture"},
                        ),
                    ),
                    corpus_digest="c" * 64,
                    query_labels_digest="d" * 64,
                ),
            ),
        )


@pytest.mark.asyncio
async def test_full_report_rebuilds_pair_metrics_and_claims_from_artifacts(
    tmp_path: Path,
) -> None:
    metric = "pair.context-v1.history_manager.context.trial_total_input_tokens.relative_reduction_percent"
    coverage = "pair.context-v1.history_manager.coverage_valid"
    spec = ExperimentSpec(
        suite="report-suite",
        repetitions=3,
        pack_ids=("context-v1",),
        output_root=tmp_path,
        identity={"codeflow_commit": "b" * 40, "model": "scripted/report"},
        claim_rules=(
            ClaimRule(
                rule_id="context-token-reduction",
                metric=metric,
                operator="ge",
                threshold=15,
                prerequisites=(coverage,),
            ),
        ),
    )
    registry = PackRegistry()
    registry.register(_TokenPack())
    ref = await run(spec, registry=registry)

    first = rebuild_full_report(ref)
    second = rebuild_full_report(ref)

    assert first == second
    assert first.ship_complete is True
    assert first.measurement_valid is True
    assert first.positive_claim_eligible is True
    assert first.metrics[coverage] is True
    assert first.metrics[metric] == 50.0
    assert first.metrics["trial.run_level_pass_rate"] == 1.0
    assert first.metrics["trial.task_level_pass_rate"] == 1.0
    assert first.pair_summaries[0].valid_pairs == 24
    assert first.pair_summaries[0].bootstrap.unit == "task"
    assert (ref.root / "cv-metrics.json").is_file()
    assert (ref.root / "REPORT.md").is_file()

    public = rebuild_report(ref)
    assert public.report_digest == first.report_digest
    assert public.positive_claim_eligible is True


@pytest.mark.parametrize(
    "artifact_kind",
    (
        "pair",
        "comparison_block",
    ),
)
@pytest.mark.asyncio
async def test_ship_completeness_requires_retained_pair_and_comparison_block(
    tmp_path: Path,
    artifact_kind: str,
) -> None:
    spec = ExperimentSpec(
        suite=f"retained-{artifact_kind}-suite",
        repetitions=1,
        pack_ids=("context-v1",),
        output_root=tmp_path,
        identity={
            "codeflow_commit": "9" * 40,
            "model": "scripted/report",
        },
    )
    registry = PackRegistry()
    registry.register(_TokenPack())
    ref = await run(
        spec,
        registry=registry,
    )
    pattern = "pairs/**/pair-result.json" if artifact_kind == "pair" else "blocks/**/block-result.json"
    next(ref.root.glob(pattern)).unlink()

    report = rebuild_full_report(ref)

    assert report.ship_complete is False
    assert report.measurement_valid is False
    assert report.positive_claim_eligible is False
    assert "incomplete_terminal_records" in report.findings


@pytest.mark.parametrize(
    "denominator",
    (
        "pairs",
        "comparison_blocks",
    ),
)
@pytest.mark.asyncio
async def test_ship_completeness_rejects_truncated_task_denominator(
    tmp_path: Path,
    denominator: str,
) -> None:
    spec = ExperimentSpec(
        suite=f"truncated-{denominator}-suite",
        repetitions=1,
        pack_ids=("context-v1",),
        output_root=tmp_path,
        identity={
            "codeflow_commit": "8" * 40,
            "model": "scripted/report",
        },
    )
    registry = PackRegistry()
    registry.register(_TokenPack())
    ref = await run(
        spec,
        registry=registry,
    )
    manifest_path = ref.root / "manifest.json"
    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8",
        )
    )
    manifest[denominator].pop()
    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    report = rebuild_full_report(ref)

    assert report.ship_complete is False
    assert report.measurement_valid is False
    assert "incomplete_terminal_records" in report.findings


@pytest.mark.asyncio
async def test_ship_completeness_rejects_truncated_retrieval_denominator(
    tmp_path: Path,
) -> None:
    spec = ExperimentSpec(
        suite="truncated-retrieval-denominator-suite",
        repetitions=1,
        pack_ids=("retrieval-evidence",),
        output_root=tmp_path,
        identity={
            "codeflow_commit": "7" * 40,
            "model": "scripted/report",
        },
    )
    registry = PackRegistry()
    registry.register(_RetrievalEvidencePack())
    ref = await run(
        spec,
        registry=registry,
    )
    manifest_path = ref.root / "manifest.json"
    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8",
        )
    )
    manifest["retrieval_cases"].pop()
    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    report = rebuild_full_report(ref)

    assert report.ship_complete is False
    assert report.measurement_valid is False
    assert "incomplete_terminal_records" in report.findings


@pytest.mark.asyncio
async def test_ship_completeness_rejects_reordered_block_trial_keys(
    tmp_path: Path,
) -> None:
    spec = ExperimentSpec(
        suite="reordered-block-trial-keys-suite",
        repetitions=1,
        pack_ids=("context-v1",),
        output_root=tmp_path,
        identity={
            "codeflow_commit": "6" * 40,
            "model": "scripted/report",
        },
    )
    registry = PackRegistry()
    registry.register(_TokenPack())
    ref = await run(
        spec,
        registry=registry,
    )
    manifest_path = ref.root / "manifest.json"
    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8",
        )
    )
    manifest["comparison_blocks"][0]["trial_keys"].reverse()
    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    report = rebuild_full_report(ref)

    assert report.ship_complete is False
    assert report.measurement_valid is False
    assert "incomplete_terminal_records" in report.findings


@pytest.mark.asyncio
async def test_ship_completeness_rejects_cross_block_retrieval_case_keys(
    tmp_path: Path,
) -> None:
    spec = ExperimentSpec(
        suite="cross-block-retrieval-case-keys-suite",
        repetitions=1,
        pack_ids=("multi-query-retrieval",),
        output_root=tmp_path,
        identity={
            "codeflow_commit": "5" * 40,
            "model": "scripted/report",
        },
    )
    registry = PackRegistry()
    registry.register(_MultiQueryRetrievalPack())
    ref = await run(
        spec,
        registry=registry,
    )
    manifest_path = ref.root / "manifest.json"
    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8",
        )
    )
    blocks = manifest["retrieval_query_blocks"]
    blocks[0]["case_keys"] = blocks[1]["case_keys"]
    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    report = rebuild_full_report(ref)

    assert report.ship_complete is False
    assert report.measurement_valid is False
    assert "incomplete_terminal_records" in report.findings


@pytest.mark.parametrize(
    "tamper",
    (
        "claim_rules",
        "report_title",
        "minimum_valid_pairs_per_task",
    ),
)
@pytest.mark.asyncio
async def test_report_rebuild_rejects_manifest_identity_tampering(
    tmp_path: Path,
    tamper: str,
) -> None:
    spec = ExperimentSpec(
        suite=f"manifest-{tamper}-tamper-suite",
        repetitions=1,
        pack_ids=("context-v1",),
        output_root=tmp_path,
        identity={
            "codeflow_commit": "0" * 40,
            "model": "scripted/report",
        },
    )
    registry = PackRegistry()
    registry.register(_TokenPack())
    ref = await run(spec, registry=registry)
    manifest_path = ref.root / "manifest.json"
    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8",
        )
    )
    if tamper == "claim_rules":
        manifest["spec"]["claim_rules"] = [
            {
                "rule_id": "tampered-rule",
                "metric": "trial.run_level_pass_rate",
                "operator": "ge",
                "threshold": 0,
                "prerequisites": [],
            }
        ]
    else:
        manifest["pack_definitions"][0]["identity"][tamper] = "Tampered report" if tamper == "report_title" else 1
    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="manifest identity",
    ):
        rebuild_full_report(ref)


@pytest.mark.parametrize(
    "field",
    (
        "schema",
        "evidence_schema",
    ),
)
@pytest.mark.asyncio
async def test_report_rebuild_requires_manifest_schema_fields(
    tmp_path: Path,
    field: str,
) -> None:
    spec = ExperimentSpec(
        suite=f"missing-{field}-suite",
        repetitions=1,
        pack_ids=("context-v1",),
        output_root=tmp_path,
        identity={
            "codeflow_commit": "6" * 40,
            "model": "scripted/report",
        },
    )
    registry = PackRegistry()
    registry.register(_TokenPack())
    ref = await run(
        spec,
        registry=registry,
    )
    manifest_path = ref.root / "manifest.json"
    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8",
        )
    )
    del manifest[field]
    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="manifest identity schema",
    ):
        rebuild_full_report(ref)


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    (
        (
            "metrics",
            {"context.trial_total_input_tokens": 1},
        ),
        ("status", TrialStatus.TASK_FAILED.value),
    ),
)
@pytest.mark.asyncio
async def test_report_rebuild_rejects_same_plan_trial_summary_tampering(
    tmp_path: Path,
    field: str,
    tampered_value: object,
) -> None:
    spec = ExperimentSpec(
        suite="trial-tamper-suite",
        repetitions=1,
        pack_ids=("context-v1",),
        output_root=tmp_path,
        identity={"codeflow_commit": "1" * 40, "model": "scripted/report"},
    )
    registry = PackRegistry()
    registry.register(_TokenPack())
    ref = await run(spec, registry=registry)
    trial_path = ref.root / "trials" / "context-v1" / "context-0" / "0" / "curator" / "trial-record.json"
    payload = json.loads(trial_path.read_text(encoding="utf-8"))
    payload[field] = tampered_value
    trial_path.write_text(json.dumps(payload), encoding="utf-8")

    report = rebuild_full_report(ref)

    assert report.ship_complete is False
    assert report.measurement_valid is False
    assert report.positive_claim_eligible is False
    assert ("trial_summary_evidence_mismatch:context-v1/context-0/0/curator") in report.findings


@pytest.mark.asyncio
async def test_report_rebuild_rejects_same_plan_pair_summary_tampering(
    tmp_path: Path,
) -> None:
    spec = ExperimentSpec(
        suite="pair-tamper-suite",
        repetitions=1,
        pack_ids=("context-v1",),
        output_root=tmp_path,
        identity={"codeflow_commit": "2" * 40, "model": "scripted/report"},
    )
    registry = PackRegistry()
    registry.register(_TokenPack())
    ref = await run(spec, registry=registry)
    pair_path = ref.root / "pairs" / "context-v1" / "history_manager" / "context-0" / "0" / "pair-result.json"
    payload = json.loads(pair_path.read_text(encoding="utf-8"))
    payload["actual_variant_diff"] = {
        "history_manager": {
            "control": "tampered",
            "treatment": "tampered",
        },
    }
    pair_path.write_text(json.dumps(payload), encoding="utf-8")

    report = rebuild_full_report(ref)

    assert report.ship_complete is False
    assert report.measurement_valid is False
    assert report.positive_claim_eligible is False
    assert ("pair_summary_evidence_mismatch:context-v1/history_manager/context-0/0") in report.findings


@pytest.mark.asyncio
async def test_report_rebuild_rejects_same_plan_retrieval_summary_tampering(
    tmp_path: Path,
) -> None:
    spec = ExperimentSpec(
        suite="retrieval-tamper-suite",
        repetitions=1,
        pack_ids=("retrieval-evidence",),
        output_root=tmp_path,
        identity={"codeflow_commit": "3" * 40, "model": "scripted/report"},
    )
    registry = PackRegistry()
    registry.register(_RetrievalEvidencePack())
    ref = await run(spec, registry=registry)
    case_path = (
        ref.root
        / "retrieval"
        / "retrieval-evidence-suite"
        / "retrieval-evidence-query"
        / "retrieval-evidence-config"
        / "retrieval-case-record.json"
    )
    payload = json.loads(case_path.read_text(encoding="utf-8"))
    payload["injected_results"] = [{"item_id": "expected-item"}]
    case_path.write_text(json.dumps(payload), encoding="utf-8")

    report = rebuild_full_report(ref)

    assert report.ship_complete is False
    assert report.measurement_valid is False
    assert report.positive_claim_eligible is False
    assert (
        "retrieval_summary_evidence_mismatch:"
        "retrieval-evidence-suite/retrieval-evidence-query/"
        "retrieval-evidence-config"
    ) in report.findings


def test_pair_coverage_policy_accepts_only_well_distributed_formal_loss() -> None:
    planned = tuple((f"task-{task}", repetition) for task in range(8) for repetition in range(3))
    formal_22 = planned[1:3] + planned[4:]
    formal_21 = planned[1:3] + planned[4:6] + planned[7:]
    one_task_below_two = planned[2:]
    calibration = tuple((f"calibration-{task}", repetition) for task in range(4) for repetition in range(2))

    assert (
        assess_pair_coverage(
            expected_pairs=24,
            planned_pair_keys=planned,
            valid_pair_keys=formal_22,
        ).valid
        is True
    )
    assert (
        assess_pair_coverage(
            expected_pairs=24,
            planned_pair_keys=planned,
            valid_pair_keys=formal_21,
        ).valid
        is False
    )
    assert (
        assess_pair_coverage(
            expected_pairs=24,
            planned_pair_keys=planned,
            valid_pair_keys=one_task_below_two,
        ).valid
        is False
    )
    assert (
        assess_pair_coverage(
            expected_pairs=8,
            planned_pair_keys=calibration,
            valid_pair_keys=calibration[:-1],
        ).valid
        is False
    )


@pytest.mark.asyncio
async def test_selected_operational_failures_only_affect_pair_coverage(
    tmp_path: Path,
) -> None:
    spec = ExperimentSpec(
        suite="operational-coverage-suite",
        repetitions=3,
        pack_ids=("context-v1",),
        output_root=tmp_path,
        identity={"codeflow_commit": "a" * 40, "model": "scripted/report"},
        execution=ExecutionPolicy(max_comparison_block_attempts=1),
    )
    registry = PackRegistry()
    registry.register(_OperationalFailureTokenPack())

    ref = await run(spec, registry=registry)
    report = rebuild_full_report(ref)

    assert report.ship_complete is True
    assert report.measurement_valid is True
    assert report.pair_summaries[0].valid_pairs == 22
    assert report.pair_summaries[0].coverage_valid is True
    assert report.selected_status_counts["provider_failure"] == 4
    assert "selected_operational_failure" not in report.findings


@pytest.mark.parametrize("tamper", ("missing", "corrupt"))
@pytest.mark.asyncio
async def test_report_rebuild_rejects_missing_or_corrupt_retry_claim(
    tmp_path: Path,
    tamper: str,
) -> None:
    spec = ExperimentSpec(
        suite="retry-claim-evidence-suite",
        repetitions=1,
        pack_ids=("context-v1",),
        output_root=tmp_path,
        identity={"codeflow_commit": "4" * 40, "model": "scripted/report"},
        execution=ExecutionPolicy(
            max_comparison_block_attempts=2,
            max_comparison_block_retries_total=1,
        ),
    )
    registry = PackRegistry()
    registry.register(_RetryingTokenPack())
    ref = await run(spec, registry=registry)
    claim_path = next(ref.root.glob("blocks/**/retry-claims/2.json"))
    if tamper == "missing":
        claim_path.unlink()
    else:
        claim_path.write_text("{", encoding="utf-8")

    report = rebuild_full_report(ref)

    assert report.ship_complete is True
    assert report.measurement_valid is False
    assert report.positive_claim_eligible is False
    assert any(finding.startswith("comparison_retry_claim_missing_or_invalid:") for finding in report.findings)


@pytest.mark.asyncio
async def test_report_rebuild_rejects_unexpected_over_quota_retry_claim(
    tmp_path: Path,
) -> None:
    spec = ExperimentSpec(
        suite="retry-claim-quota-suite",
        repetitions=1,
        pack_ids=("context-v1",),
        output_root=tmp_path,
        identity={"codeflow_commit": "5" * 40, "model": "scripted/report"},
        execution=ExecutionPolicy(
            max_comparison_block_attempts=2,
            max_comparison_block_retries_total=1,
        ),
    )
    registry = PackRegistry()
    registry.register(_RetryingTokenPack())
    ref = await run(spec, registry=registry)
    original_path = next(ref.root.glob("blocks/**/retry-claims/2.json"))
    extra_path = ref.root / "blocks" / "context-v1" / "context-1" / "0" / "retry-claims" / "2.json"
    payload = json.loads(original_path.read_text(encoding="utf-8"))
    payload["key"]["task_id"] = "context-1"
    extra_path.parent.mkdir(parents=True, exist_ok=True)
    extra_path.write_text(json.dumps(payload), encoding="utf-8")

    report = rebuild_full_report(ref)

    assert report.ship_complete is True
    assert report.measurement_valid is False
    assert report.positive_claim_eligible is False
    assert any(finding.startswith("comparison_retry_claim_unexpected:") for finding in report.findings)
    assert "comparison_retry_claim_quota_exceeded:2>1" in report.findings


@pytest.mark.asyncio
async def test_declared_pack_reducer_metrics_are_in_report(
    tmp_path: Path,
) -> None:
    spec = ExperimentSpec(
        suite="declared-reducer-suite",
        repetitions=2,
        pack_ids=("context-calibration",),
        output_root=tmp_path,
        identity={"codeflow_commit": "d" * 40, "model": "scripted/report"},
    )
    registry = PackRegistry()
    registry.register(_DeclaredContextReducerPack())

    ref = await run(spec, registry=registry)
    report = rebuild_full_report(ref)

    assert report.ship_complete is True
    assert report.measurement_valid is True
    assert report.metrics["context.measurement_valid"] is True
    assert report.metrics["context.expected_pair_count"] == 8
    assert report.metrics["context.trial_total_input_token_reduction_percent"] == pytest.approx(45.454545)


@pytest.mark.asyncio
async def test_context_v2_capability_validity_does_not_require_efficiency_pair_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = ExperimentSpec(
        suite="context-capability-reducer-suite",
        repetitions=2,
        pack_ids=("context-calibration",),
        output_root=tmp_path,
        identity={"codeflow_commit": "e" * 40, "model": "scripted/report"},
    )
    registry = PackRegistry()
    registry.register(_DeclaredContextCapabilityReducerPack())
    monkeypatch.setattr(
        "benchmarks.codeflowbench.packs.context.reduce_context_artifacts",
        lambda **_kwargs: {
            "context.capability_measurement_valid": True,
            "context.efficiency_measurement_valid": False,
        },
    )

    ref = await run(spec, registry=registry)
    report = rebuild_full_report(ref)

    assert report.ship_complete is True
    assert report.measurement_valid is True
    assert report.metrics["context.capability_measurement_valid"] is True
    assert report.metrics["context.efficiency_measurement_valid"] is False
    assert "pair_coverage_below_gate" not in report.findings


@pytest.mark.asyncio
async def test_declared_pack_ship_state_fails_closed_at_top_level(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = ExperimentSpec(
        suite="declared-ship-incomplete-suite",
        repetitions=1,
        pack_ids=("context-calibration",),
        output_root=tmp_path,
        identity={
            "codeflow_commit": "5" * 40,
            "model": "scripted/report",
        },
    )
    registry = PackRegistry()
    registry.register(_DeclaredContextReducerPack())
    ref = await run(
        spec,
        registry=registry,
    )
    from benchmarks.codeflowbench.packs.context import (
        reduce_context_artifacts,
    )

    def reduce_with_incomplete_ship(**kwargs):
        metrics = reduce_context_artifacts(**kwargs)
        metrics["context.ship_complete"] = False
        return metrics

    monkeypatch.setattr(
        "benchmarks.codeflowbench.packs.context.reduce_context_artifacts",
        reduce_with_incomplete_ship,
    )

    report = rebuild_full_report(ref)

    assert report.ship_complete is False
    assert report.measurement_valid is False
    assert "declared_pack_ship_incomplete:context.ship_complete" in report.findings


@pytest.mark.asyncio
async def test_unknown_declared_pack_reducer_fails_closed(
    tmp_path: Path,
) -> None:
    spec = ExperimentSpec(
        suite="unknown-reducer-suite",
        repetitions=1,
        pack_ids=("unknown-reducer-pack",),
        output_root=tmp_path,
        identity={"codeflow_commit": "e" * 40, "model": "scripted/report"},
    )
    registry = PackRegistry()
    registry.register(_UnknownDeclaredReducerPack())

    ref = await run(spec, registry=registry)
    report = rebuild_full_report(ref)

    assert report.ship_complete is True
    assert report.measurement_valid is False
    assert report.findings == ("unknown_declared_claim_reducer:unknown_v1",)


@pytest.mark.asyncio
async def test_pack_reducer_collision_with_base_metric_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = ExperimentSpec(
        suite="colliding-reducer-suite",
        repetitions=2,
        pack_ids=("context-calibration",),
        output_root=tmp_path,
        identity={"codeflow_commit": "f" * 40, "model": "scripted/report"},
    )
    registry = PackRegistry()
    registry.register(_DeclaredContextReducerPack())
    ref = await run(spec, registry=registry)

    monkeypatch.setattr(
        "benchmarks.codeflowbench.packs.context.reduce_context_artifacts",
        lambda **_kwargs: {
            "context.measurement_valid": True,
            "trial.planned": 999,
        },
    )
    report = rebuild_full_report(ref)

    assert report.ship_complete is True
    assert report.measurement_valid is False
    assert report.metrics["trial.planned"] == 16
    assert report.findings == ("pack_base_metric_collision:trial.planned",)


@pytest.mark.asyncio
async def test_invalid_measurement_never_exports_passing_metric(
    tmp_path: Path,
) -> None:
    metric = "pair.context-v1.history_manager.context.trial_total_input_tokens.relative_reduction_percent"
    coverage = "pair.context-v1.history_manager.coverage_valid"
    spec = ExperimentSpec(
        suite="invalid-report-suite",
        repetitions=3,
        pack_ids=("context-v1",),
        output_root=tmp_path,
        identity={"codeflow_commit": "c" * 40, "model": "scripted/report"},
        claim_rules=(
            ClaimRule(
                rule_id="context-token-reduction",
                metric=metric,
                operator="ge",
                threshold=15,
                prerequisites=(coverage,),
            ),
        ),
    )
    registry = PackRegistry()
    registry.register(_TokenPack())
    ref = await run(spec, registry=registry)
    pair_paths = sorted(ref.root.glob("pairs/**/context-0/*/pair-result.json"))
    assert len(pair_paths) == 3
    for pair_path in pair_paths:
        pair_path.write_text("{}", encoding="utf-8")

    report = rebuild_full_report(ref)
    cv_metrics = json.loads((ref.root / "cv-metrics.json").read_text(encoding="utf-8"))

    assert report.ship_complete is False
    assert report.measurement_valid is False
    assert report.positive_claim_eligible is False
    assert cv_metrics["eligible_metrics"] == {}
