from __future__ import annotations

import pytest

from codeflow.evolver.candidate_evidence import (
    EVALUATOR_BINDINGS,
    FIXTURE_BINDINGS,
    AcceptedRuntimeEvidence,
    CandidateEvidenceError,
    evaluate_candidate_evidence,
    recompute_accepted_runtime_evidence,
)
from codeflow.evolver.candidate_manifest import (
    LABEL_POLICIES,
    CandidateLabel,
    manifest_for_patch,
)
from codeflow.evolver.judge.schema import PatchWhere, PatchWhy
from codeflow.evolver.orchestrator.gates.pipeline import GateResult
from codeflow.evolver.orchestrator.gates.policy import CandidateOutcome
from codeflow.evolver.orchestrator.scoring import (
    EvaluationVerdict,
    MeasurementStatus,
    MeasurementValidity,
    TaskEval,
)
from codeflow.evolver.tree.node import AppliedPatch, NodeStatus, PatchComponent

TARGET = "benchmarks/appworld/agent_cli.py"
BEFORE = {TARGET: b"VALUE = 1\n"}
AFTER = {TARGET: b"VALUE = 2\n"}


def _manifest():
    patch = AppliedPatch(
        patch_where=PatchWhere.loop_override,
        patch_why=PatchWhy.other,
        patch_why_extra="fixture",
        components=[
            PatchComponent(
                component_id="comp_1",
                target_file=TARGET,
                diff=f"--- a/{TARGET}\n+++ b/{TARGET}\n",
                rationale="fixture",
            )
        ],
        overall_reasoning="fixture",
    )
    return manifest_for_patch(
        "candidate-1",
        CandidateLabel.runtime,
        patch,
        before_files=BEFORE,
        after_files=AFTER,
    )


def test_supported_label_policy_resolves_to_executable_bindings() -> None:
    policy = LABEL_POLICIES[CandidateLabel.runtime]

    assert policy.fixture in FIXTURE_BINDINGS
    assert policy.evaluator in EVALUATOR_BINDINGS


def test_fabricated_accepted_status_without_bound_gate_is_rejected() -> None:
    outcome = CandidateOutcome(
        node_id="candidate-1",
        status=NodeStatus.promoted_to_baseline,
        score=1.0,
        confirm_evals={"task-1": TaskEval("task-1", passes=1, attempts=1)},
        stats={"full_lift": 1.0},
        verdict=EvaluationVerdict.accepted,
    )

    with pytest.raises(CandidateEvidenceError, match="three-shield"):
        evaluate_candidate_evidence(
            _manifest(),
            outcome,
            before=BEFORE,
            after=AFTER,
        )


def test_bound_runtime_gate_produces_accepted_evidence() -> None:
    candidate = {"task-1": TaskEval("task-1", passes=1, attempts=1)}
    control = {"task-1": TaskEval("task-1", passes=0, attempts=1)}
    from codeflow.evolver.orchestrator.gates.pipeline import run_gates

    gate = run_gates(
        candidate_evals=candidate,
        control_evals=control,
        task_ids=["task-1"],
        expected_attempts=1,
    )
    outcome = CandidateOutcome(
        node_id="candidate-1",
        status=NodeStatus.promoted_to_baseline,
        score=1.0,
        confirm_evals=candidate,
        gate=gate,
        stats={"full_lift": 1.0},
        verdict=EvaluationVerdict.accepted,
    )

    decision = evaluate_candidate_evidence(
        _manifest(),
        outcome,
        before=BEFORE,
        after=AFTER,
        control_evals=control,
        task_ids=["task-1"],
        expected_attempts=1,
    )

    assert isinstance(decision, AcceptedRuntimeEvidence)
    assert recompute_accepted_runtime_evidence(_manifest(), decision).outcome.value == "accepted"


def test_fabricated_gate_cannot_hide_partial_k_measurements() -> None:
    validity = MeasurementValidity(status=MeasurementStatus.measured)
    gate = GateResult(
        promoted=True,
        verdict=EvaluationVerdict.accepted,
        paired=None,
        eligible_tasks=["task-1"],
        candidate_validity=validity,
        control_validity=validity,
    )
    outcome = CandidateOutcome(
        node_id="candidate-1",
        status=NodeStatus.promoted_to_baseline,
        score=1.0,
        confirm_evals={"task-1": TaskEval("task-1", passes=1, attempts=1)},
        gate=gate,
        stats={"full_lift": 1.0},
        verdict=EvaluationVerdict.accepted,
    )

    with pytest.raises(CandidateEvidenceError, match="exactly 3 attempts"):
        evaluate_candidate_evidence(
            _manifest(),
            outcome,
            before=BEFORE,
            after=AFTER,
            control_evals={"task-1": TaskEval("task-1", passes=0, attempts=1)},
            task_ids=["task-1"],
            expected_attempts=3,
        )


def test_reported_full_lift_is_recomputed_from_raw_measurements() -> None:
    candidate = {"task-1": TaskEval("task-1", passes=1, attempts=1)}
    control = {"task-1": TaskEval("task-1", passes=0, attempts=1)}
    from codeflow.evolver.orchestrator.gates.pipeline import run_gates

    outcome = CandidateOutcome(
        node_id="candidate-1",
        status=NodeStatus.promoted_to_baseline,
        score=1.0,
        confirm_evals=candidate,
        gate=run_gates(
            candidate_evals=candidate,
            control_evals=control,
            task_ids=["task-1"],
            expected_attempts=1,
        ),
        stats={"full_lift": 999.0},
        verdict=EvaluationVerdict.accepted,
    )

    with pytest.raises(CandidateEvidenceError, match="full-train lift"):
        evaluate_candidate_evidence(
            _manifest(),
            outcome,
            before=BEFORE,
            after=AFTER,
            control_evals=control,
            task_ids=["task-1"],
            expected_attempts=1,
        )


@pytest.mark.parametrize("arm", ["candidate", "control"])
def test_accepted_measurement_order_must_match_declared_task_order(arm: str) -> None:
    candidate = (
        TaskEval("task-1", passes=1, attempts=1),
        TaskEval("task-2", passes=1, attempts=1),
    )
    control = (
        TaskEval("task-1", passes=0, attempts=1),
        TaskEval("task-2", passes=0, attempts=1),
    )
    evidence = AcceptedRuntimeEvidence(
        schema_version=1,
        evaluator="appworld_focused_fisher_v1",
        task_ids=("task-1", "task-2"),
        expected_attempts=1,
        eligible_tasks=("task-1", "task-2"),
        candidate_evals=tuple(reversed(candidate)) if arm == "candidate" else candidate,
        control_evals=tuple(reversed(control)) if arm == "control" else control,
    )

    with pytest.raises(CandidateEvidenceError, match="order is not canonical"):
        recompute_accepted_runtime_evidence(_manifest(), evidence)


def test_runtime_fixture_rejects_snapshot_drift_from_manifest() -> None:
    outcome = CandidateOutcome(
        node_id="candidate-1",
        status=NodeStatus.pruned_at_confirm,
        verdict=EvaluationVerdict.rejected,
    )

    with pytest.raises(CandidateEvidenceError, match="content digests"):
        evaluate_candidate_evidence(
            _manifest(),
            outcome,
            before=BEFORE,
            after={TARGET: b"VALUE = 3\n"},
        )


@pytest.mark.parametrize(
    ("validity_field", "failure_key", "expected"),
    [
        ("control_validity", "provider_failures", "provider"),
        ("control_validity", "infrastructure_failures", "infrastructure"),
    ],
)
def test_nonaccepted_evidence_preserves_control_failure_class(
    validity_field: str,
    failure_key: str,
    expected: str,
) -> None:
    outcome = CandidateOutcome(
        node_id="candidate-1",
        status=NodeStatus.errored,
        stats={
            validity_field: {
                failure_key: ["task-1"],
            }
        },
        verdict=EvaluationVerdict.failed,
    )

    decision = evaluate_candidate_evidence(
        _manifest(),
        outcome,
        before=BEFORE,
        after=AFTER,
    )

    assert decision.failure_class == expected
