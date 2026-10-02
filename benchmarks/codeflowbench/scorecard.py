from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .canonical import canonical_digest
from .plan import validate_manifest_identity


@dataclass(frozen=True)
class DimensionScore:
    earned: float
    maximum: float


@dataclass(frozen=True)
class ScorecardResult:
    schema: str
    diagnostic_score: float
    certified_score: float | None
    dimensions: dict[str, DimensionScore]
    certification_gates: dict[str, bool]
    evidence: dict[str, bool]


def compute_scorecard(
    formal_summary: Mapping[str, Any],
    runtime_evidence: Mapping[str, Any],
    *,
    runtime_current_evidence: bool | None = None,
    memory_treatment_pass_rate: float | None = None,
    memory_safety_current: bool = False,
    tokenwise_current_evidence: bool = False,
    tokenwise_current_claim_eligible: bool = False,
    turn_efficiency_current_evidence: bool = False,
    turn_efficiency_current_claim_eligible: bool = False,
    scoring_spec_preregistered: bool = False,
) -> ScorecardResult:
    metrics = _mapping(formal_summary.get("metrics"), "formal metrics")
    context_capability_rate = metrics.get(
        "context.treatment_capability_score_rate",
    )
    context_capability_current = metrics.get("context.capability_evidence_complete") is True and isinstance(
        context_capability_rate, (int, float)
    )
    context_rate = (
        _rate(float(context_capability_rate))
        if context_capability_current
        else _pass_rate(
            metrics,
            numerator="context.treatment_pass_count",
            denominator="context.expected_pair_count",
        )
    )
    tool_rate = _pass_rate(
        metrics,
        numerator="tool_mcp.treatment_pass_count",
        denominator="tool_mcp.expected_pair_count",
        fallback_denominator="tool_mcp.pair_measurement_count",
    )
    memory_available = memory_treatment_pass_rate is not None
    memory_rate = _rate(memory_treatment_pass_rate or 0.0)
    capability = 50.0 * (context_rate + tool_rate + memory_rate) / 3.0

    runtime_claim_eligible = runtime_evidence.get("claim_eligible") is True
    runtime_current = bool(runtime_evidence) if runtime_current_evidence is None else runtime_current_evidence
    reliability = 20.0 if runtime_claim_eligible else 0.0

    context_efficiency = metrics.get("context.positive_claim_eligible") is True
    tool_efficiency = metrics.get("tool_mcp.positive_claim_eligible") is True
    efficiency_lanes = (
        tokenwise_current_claim_eligible,
        context_efficiency,
        tool_efficiency,
        turn_efficiency_current_claim_eligible,
    )
    efficiency = 5.0 * sum(bool(value) for value in efficiency_lanes)

    process_gates = (
        metrics.get("tool_mcp.all_initial_visible_sets_differ") is True,
        metrics.get("tool_mcp.all_mcp_arms_connected") is True,
        metrics.get("tool_mcp.invalid_target_call_rate_noninferior") is True,
        metrics.get("tool_mcp.exact_target_repeat_rate_noninferior") is True,
    )
    process = 10.0 * sum(process_gates) / len(process_gates)

    dimensions = {
        "capability": DimensionScore(round(capability, 4), 50.0),
        "reliability": DimensionScore(round(reliability, 4), 20.0),
        "efficiency": DimensionScore(round(efficiency, 4), 20.0),
        "process": DimensionScore(round(process, 4), 10.0),
    }
    diagnostic = round(sum(value.earned for value in dimensions.values()), 4)
    evidence = {
        "context_current": context_capability_current,
        "tool_mcp_current": True,
        "memory_current": memory_available,
        "runtime_current": runtime_current,
        "tokenwise_current": tokenwise_current_evidence,
        "turn_efficiency_current": turn_efficiency_current_evidence,
    }
    certification_gates = {
        "scoring_spec_preregistered": scoring_spec_preregistered,
        "ship_complete": formal_summary.get("ship_complete") is True,
        "measurement_valid": formal_summary.get("measurement_valid") is True,
        "evidence_complete": all(evidence.values()),
        "safety_evidence_complete": memory_safety_current,
    }
    certified = diagnostic if all(certification_gates.values()) else None
    return ScorecardResult(
        schema="codeflow.codeflowbench.multidimensional-score.v1",
        diagnostic_score=diagnostic,
        certified_score=certified,
        dimensions=dimensions,
        certification_gates=certification_gates,
        evidence=evidence,
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _pass_rate(
    metrics: Mapping[str, Any],
    *,
    numerator: str,
    denominator: str,
    fallback_denominator: str | None = None,
) -> float:
    raw_denominator = metrics.get(denominator)
    if raw_denominator is None and fallback_denominator is not None:
        raw_denominator = metrics.get(fallback_denominator)
    if not isinstance(raw_denominator, (int, float)) or raw_denominator <= 0:
        raise ValueError(f"{denominator} must be positive")
    raw_numerator = metrics.get(numerator)
    if not isinstance(raw_numerator, (int, float)):
        raise ValueError(f"{numerator} must be numeric")
    return _rate(float(raw_numerator) / float(raw_denominator))


def _rate(value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError("rates must be between zero and one")
    return value


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(value, str(path))


def _formal_inputs(
    summary_path: Path,
    *,
    codeflow_commit: str,
) -> Mapping[str, Any]:
    summary = _read_json(summary_path)
    manifest_path = summary_path.with_name("manifest.json")
    if not manifest_path.is_file():
        raise ValueError("formal summary manifest is missing")
    manifest = _read_json(manifest_path)
    experiment_id = summary.get("experiment_id")
    if not isinstance(experiment_id, str):
        raise ValueError("formal summary experiment identity is missing")
    validate_manifest_identity(manifest, experiment_id=experiment_id)
    spec = manifest.get("spec")
    identity = spec.get("identity") if isinstance(spec, Mapping) else None
    if (
        not isinstance(identity, Mapping)
        or identity.get("codeflow_commit") != codeflow_commit
        or identity.get("campaign_suite") != "agent-application-scorecard-v1"
        or identity.get("campaign_mode") != "formal"
    ):
        raise ValueError("formal summary does not match the current Scorecard subject")
    report_digest = summary.get("report_digest")
    payload = dict(summary)
    payload.pop("report_digest", None)
    if not isinstance(report_digest, str) or canonical_digest(payload) != report_digest:
        raise ValueError("formal summary digest does not match")
    return summary


def _current_codeflow_commit() -> str:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to resolve the CodeFlow commit")
    completed = subprocess.run(
        (git, "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
    )
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute the CodeFlowBench multidimensional diagnostic score.",
    )
    parser.add_argument("--formal-summary", required=True, type=Path)
    parser.add_argument("--runtime-evidence", type=Path)
    parser.add_argument("--tokenwise-report", type=Path)
    parser.add_argument(
        "--scoring-spec-preregistered",
        action="store_true",
    )
    args = parser.parse_args()
    current_commit = _current_codeflow_commit()
    formal_summary = _formal_inputs(
        args.formal_summary,
        codeflow_commit=current_commit,
    )
    runtime = _read_json(args.runtime_evidence) if args.runtime_evidence else {}
    runtime_current = False
    runtime_claim_eligible = False
    if args.runtime_evidence is not None:
        runtime_current, runtime_claim_eligible = _runtime_inputs(
            runtime,
            codeflow_commit=current_commit,
        )
    tokenwise_current = False
    tokenwise_claim_eligible = False
    if args.tokenwise_report is not None:
        tokenwise_current, tokenwise_claim_eligible = _tokenwise_inputs(
            _read_json(args.tokenwise_report),
            codeflow_commit=current_commit,
        )
    result = compute_scorecard(
        formal_summary,
        runtime,
        runtime_current_evidence=runtime_current,
        tokenwise_current_evidence=tokenwise_current,
        tokenwise_current_claim_eligible=tokenwise_claim_eligible,
        turn_efficiency_current_evidence=runtime_current,
        turn_efficiency_current_claim_eligible=runtime_claim_eligible,
        scoring_spec_preregistered=args.scoring_spec_preregistered,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


def _runtime_inputs(
    evidence: Mapping[str, Any],
    *,
    codeflow_commit: str,
) -> tuple[bool, bool]:
    schema = evidence.get("schema")
    commit_key = "codeflow_commit" if schema == "codeflow.codeflowbench.runtime-evidence.v2" else "source_commit"
    if schema not in {
        "codeflow.codeflowbench.runtime-evidence.v2",
        "codeflow.codeflowbench.runtime-scheduler-experiments.v1",
        "codeflow.codeflowbench.runtime-live-scheduler.v2",
    }:
        raise ValueError("runtime evidence has the wrong schema")
    _verify_current_digest_bound_evidence(
        evidence,
        digest_key="evidence_digest",
        commit_key=commit_key,
        codeflow_commit=codeflow_commit,
    )
    return True, evidence.get("claim_eligible") is True


def _tokenwise_inputs(
    report: Mapping[str, Any],
    *,
    codeflow_commit: str,
) -> tuple[bool, bool]:
    if report.get("schema") != "codeflow.codeflowbench.tokenwise-cost.report.v1":
        raise ValueError("TokenWise report has the wrong schema")
    campaign = _mapping(report.get("campaign"), "TokenWise campaign")
    if campaign.get("codeflow_commit") != codeflow_commit:
        raise ValueError("TokenWise report does not match the current CodeFlow commit")
    recorded_digest = report.get("report_digest")
    if not isinstance(recorded_digest, str):
        raise ValueError("TokenWise report is missing report_digest")
    payload = dict(report)
    payload.pop("report_digest")
    if canonical_digest(payload) != recorded_digest:
        raise ValueError("TokenWise report digest does not match")
    claim = _mapping(report.get("claim"), "TokenWise claim")
    return True, claim.get("claim_eligible") is True


def _verify_current_digest_bound_evidence(
    evidence: Mapping[str, Any],
    *,
    digest_key: str,
    commit_key: str,
    codeflow_commit: str,
) -> None:
    if evidence.get(commit_key) != codeflow_commit:
        raise ValueError("evidence does not match the current CodeFlow commit")
    recorded_digest = evidence.get(digest_key)
    if not isinstance(recorded_digest, str):
        raise ValueError("evidence digest is missing")
    payload = dict(evidence)
    payload.pop(digest_key)
    if canonical_digest(payload) != recorded_digest:
        raise ValueError("evidence digest does not match")


if __name__ == "__main__":
    raise SystemExit(main())
