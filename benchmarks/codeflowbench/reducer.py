from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from .artifacts import ArtifactStore
from .canonical import canonical_digest
from .coverage import assess_pair_coverage
from .plan import (
    manifest_derived_plan_matches_identity,
    validate_manifest_identity,
    variant_diff,
)
from .records import (
    MEASURABLE_RETRIEVAL_STATUSES,
    MEASURABLE_TRIAL_STATUSES,
    TrialStatus,
)
from .schema import ExperimentRef
from .statistics import BootstrapInterval, clustered_bootstrap_interval


@dataclass(frozen=True)
class PairSummary:
    pack_id: str
    treatment_axis: str
    planned_pairs: int
    valid_pairs: int
    covered_tasks: int
    coverage_valid: bool
    paired_pass_delta: float | None
    bootstrap: BootstrapInterval | None


@dataclass(frozen=True)
class Reduction:
    experiment_id: str
    ship_complete: bool
    measurement_valid: bool
    planned_trials: int
    terminal_trials: int
    planned_retrieval_cases: int
    terminal_retrieval_cases: int
    selected_status_counts: dict[str, int]
    first_attempt_status_counts: dict[str, int]
    all_attempt_status_counts: dict[str, int]
    retrieval_status_counts: dict[str, int]
    retrieval_first_attempt_status_counts: dict[str, int]
    retrieval_all_attempt_status_counts: dict[str, int]
    pair_summaries: tuple[PairSummary, ...]
    metrics: dict[str, Any]
    findings: tuple[str, ...]


def reduce_experiment(ref: ExperimentRef) -> Reduction:
    store = ArtifactStore(ref)
    manifest = store.read_json(store.manifest_path)
    validate_manifest_identity(
        manifest,
        experiment_id=ref.experiment_id,
    )
    plan_digest = str(manifest.get("plan_digest", ""))

    trial_records, trial_evidence_findings = _load_trial_records(
        ref.root,
        manifest,
        store,
        plan_digest,
    )
    retrieval_records, retrieval_evidence_findings = _load_retrieval_records(
        ref.root,
        manifest,
        store,
        plan_digest,
    )
    pair_records, pair_evidence_findings = _load_pair_records(
        ref.root,
        manifest,
        store,
        plan_digest,
        trial_records,
    )
    comparison_block_records, comparison_block_evidence_findings = _load_comparison_block_records(
        root=ref.root,
        manifest=manifest,
        store=store,
        plan_digest=plan_digest,
    )
    retry_claim_findings = _validate_comparison_retry_claims(
        root=ref.root,
        manifest=manifest,
        store=store,
        plan_digest=plan_digest,
    )
    attempt_records = _load_attempt_records(ref.root, store, plan_digest)
    retrieval_attempt_records = _load_retrieval_attempt_records(
        ref.root,
        store,
        plan_digest,
    )

    planned_trials = len(manifest.get("trials", []))
    planned_retrieval = len(manifest.get("retrieval_cases", []))
    planned_pairs = len(manifest.get("pairs", []))
    planned_comparison_blocks = len(
        manifest.get(
            "comparison_blocks",
            [],
        )
    )
    plan_denominators_complete = manifest_derived_plan_matches_identity(
        manifest,
    )
    artifact_ship_complete = (
        plan_denominators_complete
        and len(trial_records) == planned_trials
        and len(retrieval_records) == planned_retrieval
        and len(pair_records) == planned_pairs
        and len(comparison_block_records) == planned_comparison_blocks
    )
    selected_statuses = Counter(str(record["status"]) for record in trial_records.values())
    retrieval_statuses = Counter(str(record["status"]) for record in retrieval_records.values())
    first_attempt_statuses = Counter(
        str(record["status"]) for record in attempt_records if int(record["key"]["block_attempt"]) == 1
    )
    all_attempt_statuses = Counter(str(record["status"]) for record in attempt_records)
    retrieval_first_attempt_statuses = Counter(
        str(record["status"]) for record in retrieval_attempt_records if int(record["key"]["query_block_attempt"]) == 1
    )
    retrieval_all_attempt_statuses = Counter(str(record["status"]) for record in retrieval_attempt_records)
    task_passes: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (pack_id, task_id, _repetition, _variant_id), record in trial_records.items():
        task_passes[(pack_id, task_id)].append(float(record["status"] == TrialStatus.PASSED.value))

    pair_summaries, pair_metrics = _reduce_pairs(
        manifest=manifest,
        trial_records=trial_records,
        pair_records=pair_records,
        seed=int(ref.experiment_id[:16], 16),
    )
    retrieval_metrics = _reduce_retrieval(retrieval_records)
    pack_metrics, pack_measurement_valid, pack_findings = _reduce_declared_pack_claims(
        manifest=manifest,
        trial_records=trial_records,
        retrieval_records=retrieval_records,
        pair_records=pair_records,
    )
    incomplete_pack_ship_metrics = sorted(
        key for key, value in pack_metrics.items() if key.endswith(".ship_complete") and value is not True
    )
    ship_complete = artifact_ship_complete and not incomplete_pack_ship_metrics
    base_metrics: dict[str, Any] = {
        "trial.planned": planned_trials,
        "trial.terminal": len(trial_records),
        "trial.product_pass_rate": _rate(
            selected_statuses[TrialStatus.PASSED.value],
            sum(selected_statuses[status.value] for status in MEASURABLE_TRIAL_STATUSES),
        ),
        "trial.run_level_pass_rate": _rate(
            selected_statuses[TrialStatus.PASSED.value],
            len(trial_records),
        ),
        "trial.task_level_pass_rate": (mean(mean(values) for values in task_passes.values()) if task_passes else None),
        "trial.first_attempt_end_to_end_pass_rate": _rate(
            first_attempt_statuses[TrialStatus.PASSED.value],
            planned_trials,
        ),
        "trial.first_attempt_provider_failure_rate": _rate(
            first_attempt_statuses[TrialStatus.PROVIDER_FAILURE.value],
            planned_trials,
        ),
        "trial.first_attempt_infrastructure_failure_rate": _rate(
            first_attempt_statuses[TrialStatus.INFRASTRUCTURE_FAILURE.value],
            planned_trials,
        ),
        "trial.all_attempt_operational_failure_rate": _rate(
            all_attempt_statuses[TrialStatus.PROVIDER_FAILURE.value]
            + all_attempt_statuses[TrialStatus.INFRASTRUCTURE_FAILURE.value],
            len(attempt_records),
        ),
        "retrieval.planned": planned_retrieval,
        "retrieval.terminal": len(retrieval_records),
        "retrieval.first_attempt_measurable_rate": _rate(
            retrieval_first_attempt_statuses["measurable"],
            planned_retrieval,
        ),
        "retrieval.first_attempt_provider_failure_rate": _rate(
            retrieval_first_attempt_statuses["provider_failure"],
            planned_retrieval,
        ),
        "retrieval.first_attempt_infrastructure_failure_rate": _rate(
            retrieval_first_attempt_statuses["infrastructure_failure"],
            planned_retrieval,
        ),
        "retrieval.all_attempt_operational_failure_rate": _rate(
            retrieval_all_attempt_statuses["provider_failure"]
            + retrieval_all_attempt_statuses["infrastructure_failure"],
            len(retrieval_attempt_records),
        ),
        **pair_metrics,
        **retrieval_metrics,
    }
    base_collisions = sorted(base_metrics.keys() & pack_metrics.keys())
    if base_collisions:
        pack_measurement_valid = False
        pack_findings = (
            *pack_findings,
            "pack_base_metric_collision:" + ",".join(base_collisions),
        )
        pack_metrics = {key: value for key, value in pack_metrics.items() if key not in base_collisions}
    metrics = {**base_metrics, **pack_metrics}
    capability_only_pair_packs = {
        str(definition.get("pack_id"))
        for definition in manifest.get("pack_definitions", ())
        if isinstance(definition, dict)
        and isinstance(definition.get("identity"), dict)
        and definition["identity"].get("claim_reducer") == "context_v2"
    }
    all_pair_coverage = all(
        summary.coverage_valid or summary.pack_id in capability_only_pair_packs for summary in pair_summaries
    )
    invalid_retrieval = any(
        retrieval_statuses[status]
        for status in (
            "infrastructure_failure",
            "provider_failure",
            "cancelled",
            "inconclusive",
        )
    )
    measurement_valid = (
        ship_complete
        and not invalid_retrieval
        and all_pair_coverage
        and pack_measurement_valid
        and not trial_evidence_findings
        and not retrieval_evidence_findings
        and not pair_evidence_findings
        and not comparison_block_evidence_findings
        and not retry_claim_findings
    )
    findings: list[str] = [
        *trial_evidence_findings,
        *retrieval_evidence_findings,
        *pair_evidence_findings,
        *comparison_block_evidence_findings,
        *retry_claim_findings,
        *pack_findings,
    ]
    if incomplete_pack_ship_metrics:
        findings.append(
            "declared_pack_ship_incomplete:"
            + ",".join(
                incomplete_pack_ship_metrics,
            )
        )
    if not ship_complete:
        findings.append("incomplete_terminal_records")
    if not all_pair_coverage:
        findings.append("pair_coverage_below_gate")
    if invalid_retrieval:
        findings.append("retrieval_not_fully_measurable")
    return Reduction(
        experiment_id=ref.experiment_id,
        ship_complete=ship_complete,
        measurement_valid=measurement_valid,
        planned_trials=planned_trials,
        terminal_trials=len(trial_records),
        planned_retrieval_cases=planned_retrieval,
        terminal_retrieval_cases=len(retrieval_records),
        selected_status_counts=dict(sorted(selected_statuses.items())),
        first_attempt_status_counts=dict(sorted(first_attempt_statuses.items())),
        all_attempt_status_counts=dict(sorted(all_attempt_statuses.items())),
        retrieval_status_counts=dict(sorted(retrieval_statuses.items())),
        retrieval_first_attempt_status_counts=dict(sorted(retrieval_first_attempt_statuses.items())),
        retrieval_all_attempt_status_counts=dict(sorted(retrieval_all_attempt_statuses.items())),
        pair_summaries=pair_summaries,
        metrics=metrics,
        findings=tuple(findings),
    )


def _reduce_declared_pack_claims(
    *,
    manifest: dict[str, Any],
    trial_records: dict[tuple[str, str, int, str], dict[str, Any]],
    retrieval_records: dict[tuple[str, str, str], dict[str, Any]],
    pair_records: dict[tuple[str, str, str, int], dict[str, Any]],
) -> tuple[dict[str, Any], bool, tuple[str, ...]]:
    declared: list[str] = []
    for definition in manifest.get("pack_definitions", ()):
        identity = definition.get("identity", {})
        if not isinstance(identity, dict):
            continue
        reducer_id = identity.get("claim_reducer")
        if isinstance(reducer_id, str) and reducer_id:
            declared.append(reducer_id)
    if not declared:
        return {}, True, ()

    metrics: dict[str, Any] = {}
    findings: list[str] = []
    valid = True
    for reducer_id in declared:
        if declared.count(reducer_id) > 1:
            findings.append(
                f"duplicate_declared_claim_reducer:{reducer_id}",
            )
            valid = False
            continue
        if reducer_id in {"context_v1", "context_v2"}:
            from .packs.context import reduce_context_artifacts

            reduced = reduce_context_artifacts(
                trial_records=trial_records.values(),
                pair_results=pair_records.values(),
            )
            validity_key = (
                "context.capability_measurement_valid" if reducer_id == "context_v2" else "context.measurement_valid"
            )
        elif reducer_id == "memory_skill_v1":
            from .packs.memory_skill import reduce_memory_skill_claims

            reduced = reduce_memory_skill_claims(
                trial_records,
                retrieval_records,
                pair_records,
            )
            validity_key = "memory_skill.measurement_valid"
        elif reducer_id == "memory_effect_v1":
            from .packs.memory_skill import (
                reduce_semantic_memory_effect_claims,
            )

            reduced = reduce_semantic_memory_effect_claims(
                trial_records,
                pair_records,
            )
            validity_key = "semantic_memory_effect.measurement_valid"
        elif reducer_id == "tool_mcp_v1":
            from .packs.tool_mcp import (
                reduce_tool_mcp_claim_from_artifacts,
            )

            reduced = reduce_tool_mcp_claim_from_artifacts(
                trial_records=trial_records.values(),
                pair_results=pair_records.values(),
            )
            validity_key = "tool_mcp.measurement_valid"
        else:
            findings.append(
                f"unknown_declared_claim_reducer:{reducer_id}",
            )
            valid = False
            continue
        collisions = sorted(metrics.keys() & reduced.keys())
        if collisions:
            findings.append(
                "pack_claim_metric_collision:" + ",".join(collisions),
            )
            valid = False
            continue
        metrics.update(reduced)
        if reduced.get(validity_key) is not True:
            findings.append(
                f"declared_pack_measurement_invalid:{reducer_id}",
            )
            valid = False
    return metrics, valid, tuple(findings)


def _reduce_pairs(
    *,
    manifest: dict[str, Any],
    trial_records: dict[tuple[str, str, int, str], dict[str, Any]],
    pair_records: dict[tuple[str, str, str, int], dict[str, Any]],
    seed: int,
) -> tuple[tuple[PairSummary, ...], dict[str, Any]]:
    minimum_valid_pairs_by_pack = {
        str(definition.get("pack_id")): _minimum_valid_pairs_per_task(
            definition,
        )
        for definition in manifest.get("pack_definitions", ())
        if isinstance(definition, dict)
    }
    planned_by_axis: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for pair in manifest.get("pairs", []):
        planned_by_axis[(pair["pack_id"], pair["treatment_axis"])].append(pair)
    summaries: list[PairSummary] = []
    metrics: dict[str, Any] = {}
    for (pack_id, axis), planned in sorted(planned_by_axis.items()):
        valid = [
            pair_records[(pack_id, axis, pair["task_id"], int(pair["repetition"]))]
            for pair in planned
            if (
                pack_id,
                axis,
                pair["task_id"],
                int(pair["repetition"]),
            )
            in pair_records
            and pair_records[(pack_id, axis, pair["task_id"], int(pair["repetition"]))].get("valid") is True
        ]
        valid_keys = {
            (
                record["key"]["task_id"],
                int(record["key"]["repetition"]),
            )
            for record in valid
        }
        per_task_pass: dict[str, list[float]] = defaultdict(list)
        numeric_reductions: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for pair in planned:
            task_id = str(pair["task_id"])
            repetition = int(pair["repetition"])
            if (task_id, repetition) not in valid_keys:
                continue
            control = trial_records.get(
                (
                    pack_id,
                    task_id,
                    repetition,
                    str(pair["control_variant_id"]),
                )
            )
            treatment = trial_records.get(
                (
                    pack_id,
                    task_id,
                    repetition,
                    str(pair["treatment_variant_id"]),
                )
            )
            if control is None or treatment is None:
                continue
            per_task_pass[task_id].append(
                float(treatment["status"] == TrialStatus.PASSED.value)
                - float(control["status"] == TrialStatus.PASSED.value)
            )
            control_metrics = control.get("metrics", {})
            treatment_metrics = treatment.get("metrics", {})
            for metric in sorted(set(control_metrics) & set(treatment_metrics)):
                control_value = _number(control_metrics[metric])
                treatment_value = _number(treatment_metrics[metric])
                if control_value is None or treatment_value is None or control_value == 0:
                    continue
                numeric_reductions[metric][task_id].append(1.0 - (treatment_value / control_value))
        coverage = assess_pair_coverage(
            expected_pairs=len(planned),
            planned_pair_keys=((str(pair["task_id"]), int(pair["repetition"])) for pair in planned),
            valid_pair_keys=valid_keys,
            minimum_valid_pairs_per_task=minimum_valid_pairs_by_pack.get(
                pack_id,
                2,
            ),
        )
        coverage_valid = coverage.valid
        prefix = f"pair.{pack_id}.{axis}"
        metrics[f"{prefix}.coverage_valid"] = coverage_valid
        metrics[f"{prefix}.valid_pairs"] = len(valid)
        metrics[f"{prefix}.planned_pairs"] = len(planned)
        bootstrap = (
            clustered_bootstrap_interval(
                {task: tuple(values) for task, values in per_task_pass.items()},
                samples=10_000,
                seed=seed ^ int(canonical_digest(prefix)[:16], 16),
            )
            if per_task_pass
            else None
        )
        pass_delta = mean(mean(values) for values in per_task_pass.values()) if per_task_pass else None
        metrics[f"{prefix}.paired_task_pass_delta"] = pass_delta
        for metric, per_task in numeric_reductions.items():
            macro = mean(mean(values) for values in per_task.values()) * 100.0
            metrics[f"{prefix}.{metric}.relative_reduction_percent"] = macro
            metrics[f"{prefix}.{metric}.covered_tasks"] = len(per_task)
        summaries.append(
            PairSummary(
                pack_id=pack_id,
                treatment_axis=axis,
                planned_pairs=len(planned),
                valid_pairs=len(valid),
                covered_tasks=len(per_task_pass),
                coverage_valid=coverage_valid,
                paired_pass_delta=pass_delta,
                bootstrap=bootstrap,
            )
        )
    return tuple(summaries), metrics


def _reduce_retrieval(
    records: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    by_configuration: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records.values():
        key = record["key"]
        by_configuration[(str(key["retrieval_suite_id"]), str(key["configuration_id"]))].append(record)
    metrics: dict[str, Any] = {}
    for (suite_id, configuration_id), rows in sorted(by_configuration.items()):
        prefix = f"retrieval.{suite_id}.{configuration_id}"
        positive = [row for row in rows if row.get("label") == "positive"]
        negative = [row for row in rows if row.get("label") == "hard_negative"]
        recalls: list[float] = []
        reciprocal_ranks: list[float] = []
        for row in positive:
            expected = set(row.get("expected_item_ids", ()))
            injected = [_item_id(item) for item in row.get("injected_results", ())[:5] if _item_id(item) is not None]
            recalls.append(len(expected.intersection(injected)) / len(expected) if expected else 0.0)
            ranks = [index + 1 for index, item_id in enumerate(injected) if item_id in expected]
            reciprocal_ranks.append(1.0 / min(ranks) if ranks else 0.0)
        negative_injections = sum(bool(row.get("injected_results")) for row in negative)
        metrics[f"{prefix}.recall_at_5"] = mean(recalls) if recalls else None
        metrics[f"{prefix}.mrr_at_5"] = mean(reciprocal_ranks) if reciprocal_ranks else None
        metrics[f"{prefix}.hard_negative_injection_rate"] = _rate(
            negative_injections,
            len(negative),
        )
        metrics[f"{prefix}.measurable_cases"] = len(rows)
    return metrics


def _load_trial_records(root, manifest, store, plan_digest):
    records = {}
    findings: list[str] = []
    for planned in manifest.get("trials", []):
        key = planned["key"]
        identity = (
            str(key["pack_id"]),
            str(key["task_id"]),
            int(key["repetition"]),
            str(key["variant_id"]),
        )
        path = Path(root) / "trials" / identity[0] / identity[1] / str(identity[2]) / identity[3] / "trial-record.json"
        record = store.read_if_valid(path, plan_digest=plan_digest)
        if record is None:
            if path.exists():
                findings.append(
                    "trial_summary_evidence_mismatch:"
                    + "/".join(
                        (
                            identity[0],
                            identity[1],
                            str(identity[2]),
                            identity[3],
                        )
                    )
                )
            continue
        if not _trial_summary_matches_selected_attempt(
            root=Path(root),
            manifest=manifest,
            store=store,
            plan_digest=plan_digest,
            planned=planned,
            summary=record,
        ):
            finding = "trial_summary_evidence_mismatch:" + "/".join(
                (
                    identity[0],
                    identity[1],
                    str(identity[2]),
                    identity[3],
                )
            )
            findings.append(finding)
            continue
        records[identity] = record
    return records, tuple(findings)


def _trial_summary_matches_selected_attempt(
    *,
    root: Path,
    manifest: dict[str, Any],
    store: ArtifactStore,
    plan_digest: str,
    planned: dict[str, Any],
    summary: dict[str, Any],
) -> bool:
    key = planned.get("key")
    if not isinstance(key, dict) or summary.get("key") != key:
        return False
    if summary.get("pair_memberships") != planned.get("pair_memberships", []):
        return False
    selected = summary.get("selected_block_attempt")
    if not isinstance(selected, int) or isinstance(selected, bool) or selected < 1:
        return False
    block_key = _comparison_block_key(key)
    block_path = _comparison_block_path(root, block_key)
    block = store.read_if_valid(block_path, plan_digest=plan_digest)
    if block is None or block.get("selected_block_attempt") != selected:
        return False
    if not _comparison_block_matches_attempts(
        root=root,
        manifest=manifest,
        store=store,
        plan_digest=plan_digest,
        block_key=block_key,
        block=block,
    ):
        return False

    expected_refs: list[str] = []
    selected_attempt: dict[str, Any] | None = None
    for attempt_number in range(1, selected + 1):
        attempt_key = _comparison_attempt_key(
            block_key,
            str(key.get("variant_id")),
            attempt_number,
        )
        attempt_path = _comparison_attempt_path(
            root,
            attempt_key,
        )
        attempt = store.read_if_valid(attempt_path, plan_digest=plan_digest)
        if attempt is None or attempt.get("key") != attempt_key:
            return False
        expected_refs.append(store.relative(attempt_path))
        if attempt_number == selected:
            selected_attempt = attempt
    if summary.get("attempt_refs") != expected_refs or selected_attempt is None:
        return False
    return all(
        summary.get(field) == selected_attempt.get(field)
        for field in (
            "status",
            "runtime_state",
            "delivery_state",
            "verification",
            "declared_variant_settings",
            "observed_variant_settings",
            "metrics",
            "findings",
            "artifact_refs",
        )
    )


def _load_retrieval_records(root, manifest, store, plan_digest):
    records = {}
    findings: list[str] = []
    for planned in manifest.get("retrieval_cases", []):
        key = planned["key"]
        identity = (
            str(key["retrieval_suite_id"]),
            str(key["query_id"]),
            str(key["configuration_id"]),
        )
        path = Path(root) / "retrieval" / identity[0] / identity[1] / identity[2] / "retrieval-case-record.json"
        record = store.read_if_valid(path, plan_digest=plan_digest)
        if record is None:
            if path.exists():
                findings.append("retrieval_summary_evidence_mismatch:" + "/".join(identity))
            continue
        if not _retrieval_summary_matches_selected_attempt(
            root=Path(root),
            manifest=manifest,
            store=store,
            plan_digest=plan_digest,
            planned=planned,
            summary=record,
        ):
            findings.append("retrieval_summary_evidence_mismatch:" + "/".join(identity))
            continue
        records[identity] = record
    return records, tuple(findings)


def _load_comparison_block_records(
    *,
    root: Path,
    manifest: dict[str, Any],
    store: ArtifactStore,
    plan_digest: str,
) -> tuple[
    dict[tuple[str, str, int], dict[str, Any]],
    tuple[str, ...],
]:
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    findings: list[str] = []
    for planned in manifest.get("comparison_blocks", ()):
        if not isinstance(planned, dict):
            continue
        key = planned.get("key")
        if not isinstance(key, dict):
            continue
        identity = (
            str(key.get("pack_id")),
            str(key.get("task_id")),
            int(key.get("repetition")),
        )
        path = _comparison_block_path(
            root,
            key,
        )
        record = store.read_if_valid(
            path,
            plan_digest=plan_digest,
        )
        if record is None:
            if path.exists():
                findings.append(
                    "comparison_block_evidence_mismatch:"
                    + "/".join(
                        (
                            identity[0],
                            identity[1],
                            str(identity[2]),
                        )
                    )
                )
            continue
        if not _comparison_block_matches_attempts(
            root=root,
            manifest=manifest,
            store=store,
            plan_digest=plan_digest,
            block_key=key,
            block=record,
        ):
            findings.append(
                "comparison_block_evidence_mismatch:"
                + "/".join(
                    (
                        identity[0],
                        identity[1],
                        str(identity[2]),
                    )
                )
            )
            continue
        records[identity] = record
    return records, tuple(findings)


def _load_pair_records(
    root,
    manifest,
    store,
    plan_digest,
    trial_records,
):
    records = {}
    findings: list[str] = []
    for key in manifest.get("pairs", []):
        identity = (
            str(key["pack_id"]),
            str(key["treatment_axis"]),
            str(key["task_id"]),
            int(key["repetition"]),
        )
        path = Path(root) / "pairs" / identity[0] / identity[1] / identity[2] / str(identity[3]) / "pair-result.json"
        record = store.read_if_valid(path, plan_digest=plan_digest)
        if record is None:
            if path.exists():
                findings.append(
                    "pair_summary_evidence_mismatch:"
                    + "/".join(
                        (
                            identity[0],
                            identity[1],
                            identity[2],
                            str(identity[3]),
                        )
                    )
                )
            continue
        if not _pair_summary_matches_selected_attempts(
            root=Path(root),
            manifest=manifest,
            store=store,
            plan_digest=plan_digest,
            key=key,
            summary=record,
            trial_records=trial_records,
        ):
            findings.append(
                "pair_summary_evidence_mismatch:"
                + "/".join(
                    (
                        identity[0],
                        identity[1],
                        identity[2],
                        str(identity[3]),
                    )
                )
            )
            continue
        records[identity] = record
    return records, tuple(findings)


def _validate_comparison_retry_claims(
    *,
    root: Path,
    manifest: dict[str, Any],
    store: ArtifactStore,
    plan_digest: str,
) -> tuple[str, ...]:
    expected: dict[Path, dict[str, Any]] = {}
    block_keys: dict[tuple[str, str, int], dict[str, Any]] = {}
    for planned in manifest.get("trials", ()):
        key = planned.get("key") if isinstance(planned, dict) else None
        if not isinstance(key, dict):
            continue
        block_key = _comparison_block_key(key)
        identity = (
            str(block_key.get("pack_id")),
            str(block_key.get("task_id")),
            int(block_key.get("repetition")),
        )
        block_keys[identity] = block_key
    for block_key in block_keys.values():
        block = store.read_if_valid(
            _comparison_block_path(root, block_key),
            plan_digest=plan_digest,
        )
        if block is None:
            continue
        selected = block.get("selected_block_attempt")
        if not isinstance(selected, int) or isinstance(selected, bool) or selected < 1:
            continue
        for attempt_number in range(2, selected + 1):
            path = _comparison_retry_claim_path(
                root,
                block_key,
                attempt_number,
            )
            expected[path] = {
                "kind": "comparison_block_retry_claim",
                "plan_digest": plan_digest,
                "key": block_key,
                "block_attempt": attempt_number,
            }

    claims_root = root / "blocks"
    actual = set(claims_root.glob("*/*/*/retry-claims/*.json")) if claims_root.exists() else set()
    findings: list[str] = []
    for path, expected_claim in sorted(
        expected.items(),
        key=lambda item: item[0].as_posix(),
    ):
        claim = store.read_if_valid(path, plan_digest=plan_digest)
        if claim != expected_claim:
            findings.append(
                "comparison_retry_claim_missing_or_invalid:" + store.relative(path),
            )
    for path in sorted(actual - set(expected), key=Path.as_posix):
        findings.append(
            "comparison_retry_claim_unexpected:" + store.relative(path),
        )

    maximum = manifest.get("spec", {}).get("execution", {}).get("max_comparison_block_retries_total")
    if isinstance(maximum, int) and not isinstance(maximum, bool) and len(actual) > maximum:
        findings.append(
            f"comparison_retry_claim_quota_exceeded:{len(actual)}>{maximum}",
        )
    return tuple(findings)


def _comparison_block_matches_attempts(
    *,
    root: Path,
    manifest: dict[str, Any],
    store: ArtifactStore,
    plan_digest: str,
    block_key: dict[str, Any],
    block: dict[str, Any],
) -> bool:
    selected = block.get("selected_block_attempt")
    maximum = manifest.get("spec", {}).get("execution", {}).get("max_comparison_block_attempts")
    if (
        block.get("key") != block_key
        or not isinstance(selected, int)
        or isinstance(selected, bool)
        or selected < 1
        or not isinstance(maximum, int)
        or isinstance(maximum, bool)
        or selected > maximum
    ):
        return False
    resolved = block.get("resolved") is True
    exhausted = block.get("exhausted") is True
    if resolved == exhausted:
        return False
    planned_trials = [
        planned
        for planned in manifest.get("trials", ())
        if isinstance(planned, dict)
        and isinstance(planned.get("key"), dict)
        and _comparison_block_key(planned["key"]) == block_key
    ]
    if not planned_trials:
        return False
    expected_refs: list[str] = []
    selected_attempts: list[dict[str, Any]] = []
    for attempt_number in range(1, selected + 1):
        for planned in planned_trials:
            trial_key = planned["key"]
            attempt_key = _comparison_attempt_key(
                block_key,
                str(trial_key.get("variant_id")),
                attempt_number,
            )
            attempt_path = _comparison_attempt_path(root, attempt_key)
            attempt = store.read_if_valid(
                attempt_path,
                plan_digest=plan_digest,
            )
            if attempt is None or attempt.get("key") != attempt_key:
                return False
            expected_refs.append(store.relative(attempt_path))
            if attempt_number == selected:
                selected_attempts.append(attempt)
    if block.get("variant_attempt_refs") != expected_refs:
        return False
    measurable = all(
        attempt.get("status") in {status.value for status in MEASURABLE_TRIAL_STATUSES} for attempt in selected_attempts
    )
    expected_findings = [] if resolved else ["comparison_block_attempts_exhausted"]
    return resolved == measurable and block.get("findings", []) == expected_findings


def _pair_summary_matches_selected_attempts(
    *,
    root: Path,
    manifest: dict[str, Any],
    store: ArtifactStore,
    plan_digest: str,
    key: dict[str, Any],
    summary: dict[str, Any],
    trial_records: dict[tuple[str, str, int, str], dict[str, Any]],
) -> bool:
    if summary.get("key") != key:
        return False
    block_key = _comparison_block_key(key)
    block = store.read_if_valid(
        _comparison_block_path(root, block_key),
        plan_digest=plan_digest,
    )
    if block is None or not _comparison_block_matches_attempts(
        root=root,
        manifest=manifest,
        store=store,
        plan_digest=plan_digest,
        block_key=block_key,
        block=block,
    ):
        return False
    identity = (
        str(key.get("pack_id")),
        str(key.get("task_id")),
        int(key.get("repetition")),
    )
    control = trial_records.get((*identity, str(key.get("control_variant_id"))))
    treatment = trial_records.get((*identity, str(key.get("treatment_variant_id"))))
    if control is None or treatment is None:
        return False
    control_settings = control.get("observed_variant_settings")
    treatment_settings = treatment.get("observed_variant_settings")
    if not isinstance(control_settings, dict) or not isinstance(
        treatment_settings,
        dict,
    ):
        return False
    actual_diff = variant_diff(control_settings, treatment_settings)
    resolved = block.get("resolved") is True
    expected_valid = (
        resolved
        and set(actual_diff) == {str(key.get("treatment_axis"))}
        and control.get("status") in {status.value for status in MEASURABLE_TRIAL_STATUSES}
        and treatment.get("status") in {status.value for status in MEASURABLE_TRIAL_STATUSES}
    )
    expected_findings = [] if expected_valid else ["pair_invalid_or_variant_drift"]
    expected_selected = block.get("selected_block_attempt") if resolved else None
    return (
        summary.get("selected_block_attempt") == expected_selected
        and summary.get("valid") is expected_valid
        and summary.get("actual_variant_diff") == actual_diff
        and summary.get("findings", []) == expected_findings
    )


def _retrieval_summary_matches_selected_attempt(
    *,
    root: Path,
    manifest: dict[str, Any],
    store: ArtifactStore,
    plan_digest: str,
    planned: dict[str, Any],
    summary: dict[str, Any],
) -> bool:
    key = planned.get("key")
    block_key = planned.get("query_block")
    if (
        not isinstance(key, dict)
        or not isinstance(block_key, dict)
        or summary.get("key") != key
        or summary.get("query_block") != block_key
    ):
        return False
    selected = summary.get("selected_query_block_attempt")
    if not isinstance(selected, int) or isinstance(selected, bool) or selected < 1:
        return False
    block = store.read_if_valid(
        _retrieval_block_path(root, block_key),
        plan_digest=plan_digest,
    )
    if block is None or block.get("selected_query_block_attempt") != selected:
        return False
    if not _retrieval_block_matches_attempts(
        root=root,
        manifest=manifest,
        store=store,
        plan_digest=plan_digest,
        block_key=block_key,
        block=block,
    ):
        return False

    expected_refs: list[str] = []
    selected_attempt: dict[str, Any] | None = None
    for attempt_number in range(1, selected + 1):
        attempt_key = _retrieval_attempt_key(
            block_key,
            str(key.get("configuration_id")),
            attempt_number,
        )
        attempt_path = _retrieval_attempt_path(root, attempt_key)
        attempt = store.read_if_valid(
            attempt_path,
            plan_digest=plan_digest,
        )
        if attempt is None or attempt.get("key") != attempt_key:
            return False
        expected_refs.append(store.relative(attempt_path))
        if attempt_number == selected:
            selected_attempt = attempt
    if summary.get("attempt_refs") != expected_refs or selected_attempt is None:
        return False
    return all(
        summary.get(field) == selected_attempt.get(field)
        for field in (
            "status",
            "label",
            "expected_item_ids",
            "ranked_results",
            "injected_results",
            "usage",
            "findings",
        )
    ) and _optional_metadata_matches(summary, selected_attempt)


def _retrieval_block_matches_attempts(
    *,
    root: Path,
    manifest: dict[str, Any],
    store: ArtifactStore,
    plan_digest: str,
    block_key: dict[str, Any],
    block: dict[str, Any],
) -> bool:
    selected = block.get("selected_query_block_attempt")
    maximum = manifest.get("spec", {}).get("execution", {}).get("max_retrieval_query_block_attempts")
    if (
        block.get("key") != block_key
        or not isinstance(selected, int)
        or isinstance(selected, bool)
        or selected < 1
        or not isinstance(maximum, int)
        or isinstance(maximum, bool)
        or selected > maximum
    ):
        return False
    resolved = block.get("resolved") is True
    exhausted = block.get("exhausted") is True
    if resolved == exhausted:
        return False
    planned_cases = [
        planned
        for planned in manifest.get("retrieval_cases", ())
        if isinstance(planned, dict)
        and planned.get("query_block") == block_key
        and isinstance(planned.get("key"), dict)
    ]
    if not planned_cases:
        return False
    expected_refs: list[str] = []
    selected_attempts: list[dict[str, Any]] = []
    for attempt_number in range(1, selected + 1):
        for planned in planned_cases:
            attempt_key = _retrieval_attempt_key(
                block_key,
                str(planned["key"].get("configuration_id")),
                attempt_number,
            )
            attempt_path = _retrieval_attempt_path(root, attempt_key)
            attempt = store.read_if_valid(
                attempt_path,
                plan_digest=plan_digest,
            )
            if attempt is None or attempt.get("key") != attempt_key:
                return False
            expected_refs.append(store.relative(attempt_path))
            if attempt_number == selected:
                selected_attempts.append(attempt)
    if block.get("configuration_attempt_refs") != expected_refs:
        return False
    measurable = all(
        attempt.get("status") in {status.value for status in MEASURABLE_RETRIEVAL_STATUSES}
        for attempt in selected_attempts
    )
    expected_findings = [] if resolved else ["retrieval_query_block_attempts_exhausted"]
    return resolved == measurable and block.get("findings", []) == expected_findings


def _comparison_block_key(key: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": key.get("experiment_id"),
        "pack_id": key.get("pack_id"),
        "task_id": key.get("task_id"),
        "repetition": key.get("repetition"),
    }


def _comparison_attempt_key(
    block_key: dict[str, Any],
    variant_id: str,
    attempt_number: int,
) -> dict[str, Any]:
    return {
        "block": block_key,
        "variant_id": variant_id,
        "block_attempt": attempt_number,
    }


def _comparison_block_path(
    root: Path,
    block_key: dict[str, Any],
) -> Path:
    return (
        root
        / "blocks"
        / str(block_key.get("pack_id"))
        / str(block_key.get("task_id"))
        / str(block_key.get("repetition"))
        / "block-result.json"
    )


def _comparison_retry_claim_path(
    root: Path,
    block_key: dict[str, Any],
    attempt_number: int,
) -> Path:
    return (
        root
        / "blocks"
        / str(block_key.get("pack_id"))
        / str(block_key.get("task_id"))
        / str(block_key.get("repetition"))
        / "retry-claims"
        / f"{attempt_number}.json"
    )


def _comparison_attempt_path(
    root: Path,
    attempt_key: dict[str, Any],
) -> Path:
    block = attempt_key["block"]
    return (
        root
        / "trials"
        / str(block.get("pack_id"))
        / str(block.get("task_id"))
        / str(block.get("repetition"))
        / str(attempt_key.get("variant_id"))
        / "attempts"
        / str(attempt_key.get("block_attempt"))
        / "attempt-record.json"
    )


def _retrieval_attempt_key(
    block_key: dict[str, Any],
    configuration_id: str,
    attempt_number: int,
) -> dict[str, Any]:
    return {
        "block": block_key,
        "configuration_id": configuration_id,
        "query_block_attempt": attempt_number,
    }


def _retrieval_block_path(
    root: Path,
    block_key: dict[str, Any],
) -> Path:
    return (
        root
        / "retrieval"
        / "query-blocks"
        / str(block_key.get("retrieval_suite_id"))
        / str(block_key.get("query_id"))
        / "retrieval-query-block-result.json"
    )


def _retrieval_attempt_path(
    root: Path,
    attempt_key: dict[str, Any],
) -> Path:
    block = attempt_key["block"]
    return (
        root
        / "retrieval"
        / str(block.get("retrieval_suite_id"))
        / str(block.get("query_id"))
        / str(attempt_key.get("configuration_id"))
        / "attempts"
        / str(attempt_key.get("query_block_attempt"))
        / "retrieval-attempt-record.json"
    )


def _load_attempt_records(root, store, plan_digest):
    records = []
    for path in Path(root).glob("trials/**/attempt-record.json"):
        record = store.read_if_valid(path, plan_digest=plan_digest)
        if record is not None:
            records.append(record)
    return records


def _load_retrieval_attempt_records(root, store, plan_digest):
    records = []
    for path in Path(root).glob("retrieval/**/retrieval-attempt-record.json"):
        record = store.read_if_valid(path, plan_digest=plan_digest)
        if record is not None:
            records.append(record)
    return records


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _minimum_valid_pairs_per_task(
    definition: dict[str, Any],
) -> int:
    identity = definition.get("identity")
    if not isinstance(identity, dict):
        return 2
    value = identity.get("minimum_valid_pairs_per_task", 2)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        return 2
    return value


def _optional_metadata_matches(
    left: dict[str, Any],
    right: dict[str, Any],
) -> bool:
    left_metadata = left.get("metadata", {})
    right_metadata = right.get("metadata", {})
    return isinstance(left_metadata, dict) and isinstance(right_metadata, dict) and left_metadata == right_metadata


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _item_id(item: Any) -> str | None:
    if not isinstance(item, dict):
        return None
    value = item.get("item_id") or item.get("id_hash") or item.get("id")
    return str(value) if value is not None else None
