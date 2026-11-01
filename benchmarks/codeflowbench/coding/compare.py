"""Compare two immutable sets of coding run artifacts by task and repetition."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def compare_runs(reference_root: Path, candidate_root: Path, output: Path | None = None) -> dict[str, Any]:
    reference = _load_runs(reference_root)
    candidate = _load_runs(candidate_root)
    valid_ref = [r for r in reference if r.get("status") in {"passed", "failed"}]
    valid_cand = [r for r in candidate if r.get("status") in {"passed", "failed"}]
    ref_index = _index(valid_ref)
    cand_index = _index(valid_cand)
    common = sorted(ref_index.keys() & cand_index.keys())
    pairs = []
    incompatible = []
    for key in common:
        old, new = ref_index[key], cand_index[key]
        if old.get("task_digest") != new.get("task_digest"):
            incompatible.append({"task_id": key[0], "repetition": key[1], "reason": "task_definition_digest_mismatch"})
            continue
        if old.get("status") not in {"passed", "failed"} or new.get("status") not in {"passed", "failed"}:
            continue
        old_ok = old.get("status") == "passed"
        new_ok = new.get("status") == "passed"
        pairs.append(
            {
                "task_id": key[0],
                "repetition": key[1],
                "reference_status": old.get("status"),
                "candidate_status": new.get("status"),
                "transition": "regression"
                if old_ok and not new_ok
                else "recovered"
                if not old_ok and new_ok
                else "stable_pass"
                if old_ok
                else "stable_fail",
                "reference_latency_ms": _metric(old, "end_to_end_latency_ms"),
                "candidate_latency_ms": _metric(new, "end_to_end_latency_ms"),
                "reference_tokens": _metric(old, "tokens"),
                "candidate_tokens": _metric(new, "tokens"),
            }
        )
    report = {
        "schema": "codeflow.coding-version-comparison.v1",
        "reference_runs": len(reference),
        "candidate_runs": len(candidate),
        "paired_runs": len(pairs),
        "incompatible_task_repetitions": incompatible,
        "invalid_reference_runs": len(reference) - len(valid_ref),
        "invalid_candidate_runs": len(candidate) - len(valid_cand),
        "unpaired_reference_runs": len(valid_ref) - len(pairs),
        "unpaired_candidate_runs": len(valid_cand) - len(pairs),
        "reference_pass_rate": _rate(valid_ref),
        "candidate_pass_rate": _rate(valid_cand),
        "pass_rate_delta": _delta(_rate(valid_ref), _rate(valid_cand)),
        "paired_regressions": sum(p["transition"] == "regression" for p in pairs),
        "paired_recoveries": sum(p["transition"] == "recovered" for p in pairs),
        "paired_latency_delta_ms": _mean_delta(pairs, "reference_latency_ms", "candidate_latency_ms"),
        "paired_token_delta": _mean_delta(pairs, "reference_tokens", "candidate_tokens"),
        "reference_pass_at_1": _pass_at_n(reference, 1),
        "candidate_pass_at_1": _pass_at_n(candidate, 1),
        "reference_pass_at_3": _pass_at_n(reference, 3),
        "candidate_pass_at_3": _pass_at_n(candidate, 3),
        "pairs": pairs,
        "findings": _findings(reference, candidate, pairs, incompatible),
    }
    canonical = json.dumps(report, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    report["report_digest"] = hashlib.sha256(canonical).hexdigest()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _load_runs(root: Path) -> list[dict[str, Any]]:
    paths = sorted(Path(root).rglob("result.json")) if Path(root).is_dir() else [Path(root)]
    result = []
    for path in paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value.get("schema") == "codeflow.coding-run.v1":
            result.append(value)
    return result


def _index(runs: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for run in runs:
        if not isinstance(run.get("task_id"), str) or not isinstance(run.get("repetition"), int):
            continue
        key = (run["task_id"], run["repetition"])
        if key in indexed:
            raise ValueError(f"duplicate task/repetition in comparison input: {key}")
        indexed[key] = run
    return indexed


def _metric(run: dict[str, Any], name: str) -> float | None:
    value = run.get("metrics", {}).get(name)
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _rate(runs: list[dict[str, Any]]) -> float | None:
    return sum(r["status"] == "passed" for r in runs) / len(runs) if runs else None


def _delta(old: float | None, new: float | None) -> float | None:
    return new - old if old is not None and new is not None else None


def _mean_delta(pairs: list[dict[str, Any]], old_key: str, new_key: str) -> float | None:
    values = [p[new_key] - p[old_key] for p in pairs if p[old_key] is not None and p[new_key] is not None]
    return sum(values) / len(values) if values else None


def _pass_at_n(runs: list[dict[str, Any]], n: int) -> float | None:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        if run.get("status") in {"passed", "failed"}:
            by_task.setdefault(str(run.get("task_id")), []).append(run)
    eligible = [
        sorted(items, key=lambda run: run.get("repetition", 0)) for items in by_task.values() if len(items) >= n
    ]
    if not eligible:
        return None
    return sum(any(run.get("status") == "passed" for run in items[:n]) for items in eligible) / len(eligible)


def _findings(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    incompatible: list[dict[str, Any]],
) -> list[str]:
    findings = []
    if not pairs:
        findings.append("no_paired_task_repetitions; paired regression and cost deltas unavailable")
    if any(run.get("status") in {"invalid_baseline", "invalid_verifier"} for run in (*reference, *candidate)):
        findings.append("baseline_or_verifier_integrity_runs_excluded_from_pass_rates")
    if any(run.get("metrics", {}).get("token_budget_exceeded") is None for run in (*reference, *candidate)):
        findings.append("some_token_usage_or_budget_evidence_missing")
    if sum(p["transition"] == "regression" for p in pairs):
        findings.append("candidate_introduces_paired_task_regressions")
    if any(r.get("task_digest") is None for r in (*reference, *candidate)):
        findings.append("task_identity_digest_missing")
    if incompatible:
        findings.append("task_definition_mismatch_excluded_from_paired_deltas")
    return findings
