"""Offline CallEfficiency replay for frozen TokenWise DeepSeek reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from benchmarks.codeflowbench.canonical import canonical_digest, to_primitive
from codeflow.call_efficiency.pricing import estimate_cost_from_rates
from codeflow.utils.atomic_io import atomic_replace

from .live import REPORT_SCHEMA
from .models import TokenWiseCostMeasurement
from .pack import TokenWiseCostPack
from .reducer import assess_tokenwise_cost_claim

REPLAY_REPORT_SCHEMA = "codeflow.codeflowbench.call-efficiency-replay.report.v1"


def verify_legacy_report(
    source: dict[str, Any],
    *,
    expected_source_digest: str | None = None,
) -> dict[str, Any]:
    if source.get("schema") != REPORT_SCHEMA:
        raise ValueError("unsupported historical TokenWise report schema")
    source_digest = str(source.get("report_digest") or "")
    digest_payload = {key: value for key, value in source.items() if key != "report_digest"}
    digest_valid = source_digest == canonical_digest(digest_payload)
    findings: list[str] = []
    if not digest_valid:
        findings.append("source_report_digest_mismatch")
    expected_digest_matches = None
    if expected_source_digest is None:
        findings.append("expected_source_digest_required")
    else:
        expected_digest_matches = source_digest == expected_source_digest
        if not expected_digest_matches:
            findings.append("source_report_digest_not_expected")

    campaign = source.get("campaign")
    trials = source.get("trials")
    if not isinstance(campaign, dict) or not isinstance(trials, list):
        raise ValueError("historical TokenWise report is missing campaign or trials")
    price = campaign.get("price_snapshot")
    if not isinstance(price, dict):
        raise ValueError("historical TokenWise report has no frozen price snapshot")

    definition = TokenWiseCostPack().definition()
    expected_workloads = {task.task_id: str(task.payload["workload_class"]) for task in definition.tasks}
    if int(campaign.get("repetitions", 0)) != 3 or len(trials) != 72:
        findings.append("historical_trial_matrix_incomplete")
    measurements: list[TokenWiseCostMeasurement] = []
    observed_workloads: dict[str, str] = {}
    cost_equivalent = True
    for index, trial in enumerate(trials):
        if not isinstance(trial, dict):
            raise ValueError(f"historical trial {index} is not an object")
        task_id = str(trial["task_id"])
        workload_class = str(trial["workload_class"])
        prior_workload = observed_workloads.setdefault(task_id, workload_class)
        if prior_workload != workload_class:
            findings.append(f"task_workload_mismatch:{task_id}")
        if expected_workloads.get(task_id) != workload_class:
            findings.append(f"unexpected_task_workload:{task_id}")
        replayed_cost = estimate_cost_from_rates(
            input_tokens=int(trial["fresh_input_tokens"]),
            output_tokens=int(trial["output_tokens"]),
            cache_read_tokens=int(trial["cache_read_tokens"]),
            cache_write_tokens=int(trial["cache_write_tokens"]),
            input_usd_per_token=float(price["cache_miss_usd_per_token"]),
            output_usd_per_token=float(price["output_usd_per_token"]),
            cache_read_usd_per_token=float(price["cache_hit_usd_per_token"]),
        )
        source_cost = float(trial["cost_usd"])
        if not math.isclose(replayed_cost, source_cost, rel_tol=1e-9, abs_tol=1e-12):
            findings.append(f"trial_cost_mismatch:{task_id}:{trial['repetition']}:{trial['cache_policy']}")
            cost_equivalent = False
        measurements.append(
            TokenWiseCostMeasurement(
                task_id=task_id,
                workload_class=workload_class,
                repetition=int(trial["repetition"]),
                cache_policy=str(trial["cache_policy"]),
                task_passed=bool(trial["task_passed"]),
                usage_complete=bool(trial["usage_complete"]),
                cost_complete=bool(trial["cost_complete"]),
                requested_model=str(trial["requested_model"]),
                actual_model=str(trial["actual_model"]) if trial.get("actual_model") is not None else None,
                fallback_used=bool(trial["fallback_used"]),
                fresh_input_tokens=int(trial["fresh_input_tokens"]),
                cache_write_tokens=int(trial["cache_write_tokens"]),
                cache_read_tokens=int(trial["cache_read_tokens"]),
                output_tokens=int(trial["output_tokens"]),
                cost_usd=replayed_cost,
            )
        )

    claim = assess_tokenwise_cost_claim(
        tuple(measurements),
        expected_workloads=expected_workloads,
        repetitions=int(campaign["repetitions"]),
    )
    rebuilt_claim = to_primitive(claim)
    claim_equivalent = rebuilt_claim == source.get("claim")
    if not claim_equivalent:
        findings.append("claim_reduction_mismatch")
    equivalent = (
        digest_valid and expected_digest_matches is True and cost_equivalent and claim_equivalent and not findings
    )
    report = {
        "schema": REPLAY_REPORT_SCHEMA,
        "source_schema": REPORT_SCHEMA,
        "source_report_digest": source_digest,
        "source_report_digest_valid": digest_valid,
        "expected_source_digest": expected_source_digest,
        "expected_source_digest_matches": expected_digest_matches,
        "source_codeflow_commit": campaign.get("codeflow_commit"),
        "source_plan_digest": campaign.get("plan_digest"),
        "price_snapshot_id": price.get("snapshot_id"),
        "trial_count": len(measurements),
        "provider_call_count": sum(int(trial.get("provider_calls", 0)) for trial in trials),
        "cost_equivalent": cost_equivalent,
        "claim_equivalent": claim_equivalent,
        "equivalent": equivalent,
        "findings": findings,
        "replayed_claim": rebuilt_claim,
    }
    report["replay_digest"] = canonical_digest(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--expected-source-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.source_report.resolve() == args.output.resolve():
        parser.error("--output must not overwrite --source-report")
    source = json.loads(args.source_report.read_text(encoding="utf-8"))
    report = verify_legacy_report(source, expected_source_digest=args.expected_source_digest)
    atomic_replace(
        args.output,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        increment_epoch=True,
    )
    return 0 if report["equivalent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["REPLAY_REPORT_SCHEMA", "main", "verify_legacy_report"]
