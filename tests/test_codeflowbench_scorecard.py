from __future__ import annotations

import json

import pytest

from benchmarks.codeflowbench.canonical import canonical_digest
from benchmarks.codeflowbench.scorecard import (
    _formal_inputs,
    _runtime_inputs,
    _tokenwise_inputs,
    compute_scorecard,
)


def _formal_summary() -> dict[str, object]:
    return {
        "ship_complete": True,
        "measurement_valid": True,
        "metrics": {
            "context.expected_pair_count": 24,
            "context.treatment_pass_count": 0,
            "context.capability_evidence_complete": True,
            "context.treatment_capability_score_rate": 0.25,
            "context.positive_claim_eligible": False,
            "tool_mcp.pair_measurement_count": 24,
            "tool_mcp.treatment_pass_count": 22,
            "tool_mcp.positive_claim_eligible": False,
            "tool_mcp.all_initial_visible_sets_differ": True,
            "tool_mcp.all_mcp_arms_connected": True,
            "tool_mcp.invalid_target_call_rate_noninferior": True,
            "tool_mcp.exact_target_repeat_rate_noninferior": True,
        },
    }


def test_scorecard_reports_current_diagnostic_without_certifying_it() -> None:
    result = compute_scorecard(
        _formal_summary(),
        {"claim_eligible": True},
    )

    assert result.schema == "codeflow.codeflowbench.multidimensional-score.v1"
    assert result.diagnostic_score == pytest.approx(49.4444)
    assert result.certified_score is None
    assert result.dimensions["capability"].earned == pytest.approx(19.4444)
    assert result.dimensions["reliability"].earned == 20
    assert result.dimensions["efficiency"].earned == 0
    assert result.dimensions["process"].earned == 10
    assert result.certification_gates["evidence_complete"] is False


def test_scorecard_certifies_only_complete_preregistered_evidence() -> None:
    summary = _formal_summary()
    metrics = summary["metrics"]
    assert isinstance(metrics, dict)
    metrics["context.positive_claim_eligible"] = True
    metrics["tool_mcp.positive_claim_eligible"] = True
    metrics["context.treatment_capability_score_rate"] = 1.0

    result = compute_scorecard(
        summary,
        {"claim_eligible": True},
        memory_treatment_pass_rate=0.75,
        memory_safety_current=True,
        tokenwise_current_evidence=True,
        tokenwise_current_claim_eligible=True,
        turn_efficiency_current_evidence=True,
        turn_efficiency_current_claim_eligible=True,
        scoring_spec_preregistered=True,
    )

    assert result.diagnostic_score == pytest.approx(94.4444)
    assert result.certified_score == result.diagnostic_score


def test_scorecard_rejects_invalid_rates() -> None:
    summary = _formal_summary()
    metrics = summary["metrics"]
    assert isinstance(metrics, dict)
    metrics["tool_mcp.treatment_pass_count"] = 25

    with pytest.raises(ValueError, match="rates must be between"):
        compute_scorecard(summary, {"claim_eligible": True})


def test_scorecard_accepts_current_negative_runtime_evidence() -> None:
    codeflow_commit = "a" * 40
    payload = {
        "schema": "codeflow.codeflowbench.runtime-scheduler-experiments.v1",
        "source_commit": codeflow_commit,
        "claim_eligible": False,
    }
    evidence = {**payload, "evidence_digest": canonical_digest(payload)}

    assert _runtime_inputs(evidence, codeflow_commit=codeflow_commit) == (True, False)


def test_scorecard_accepts_current_tokenwise_report() -> None:
    codeflow_commit = "b" * 40
    payload = {
        "schema": "codeflow.codeflowbench.tokenwise-cost.report.v1",
        "campaign": {"codeflow_commit": codeflow_commit},
        "claim": {"claim_eligible": True},
    }
    report = {**payload, "report_digest": canonical_digest(payload)}

    assert _tokenwise_inputs(report, codeflow_commit=codeflow_commit) == (True, True)


def test_scorecard_requires_formal_manifest_from_current_commit(tmp_path) -> None:
    codeflow_commit = "a" * 40
    spec = {
        "schema": "codeflow.codeflowbench.experiment.v1",
        "evidence_schema": "codeflow.codeflowbench.evidence.v1",
        "identity": {
            "campaign_mode": "formal",
            "campaign_suite": "agent-application-scorecard-v1",
            "codeflow_commit": codeflow_commit,
        },
    }
    experiment_id = canonical_digest({"spec": spec, "pack_definitions": []})
    manifest = {
        "schema": spec["schema"],
        "evidence_schema": spec["evidence_schema"],
        "experiment_id": experiment_id,
        "plan_digest": experiment_id,
        "spec": spec,
        "pack_definitions": [],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    payload = {**_formal_summary(), "experiment_id": experiment_id}
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps({**payload, "report_digest": canonical_digest(payload)}),
        encoding="utf-8",
    )

    assert _formal_inputs(summary_path, codeflow_commit=codeflow_commit)["experiment_id"] == experiment_id

    with pytest.raises(ValueError, match="current Scorecard subject"):
        _formal_inputs(summary_path, codeflow_commit="b" * 40)


def test_scorecard_counts_negative_runtime_as_current_evidence() -> None:
    result = compute_scorecard(
        _formal_summary(),
        {"claim_eligible": False},
        runtime_current_evidence=True,
        turn_efficiency_current_evidence=True,
    )

    assert result.dimensions["reliability"].earned == 0
    assert result.evidence["runtime_current"] is True
    assert result.evidence["turn_efficiency_current"] is True
