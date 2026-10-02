"""Reducer for frozen historical Memory/Skill Trial records."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from statistics import mean
from typing import Any

from benchmarks.codeflowbench.coverage import assess_pair_coverage
from benchmarks.codeflowbench.schema import JsonValue

from .fixtures import anonymous_item_id

_MEASURABLE_TRIAL_STATUSES = {"passed", "task_failed", "task_timeout"}
_FORMAL_PACK_ID = "memory-skill-v1"
_CALIBRATION_PACK_ID = "memory-skill-calibration-v1"


def reduce_memory_skill_claims(
    trial_records: Mapping[Any, Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    retrieval_records: Mapping[Any, Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    pair_results: Mapping[Any, Mapping[str, Any]] | Iterable[Mapping[str, Any]],
) -> dict[str, JsonValue]:
    trials = _materialize(trial_records)
    retrieval = _materialize(retrieval_records)
    pairs = _materialize(pair_results)
    metrics: dict[str, JsonValue] = {}
    metrics.update(
        _reduce_e2e(
            trials,
            pairs,
            prefix="memory_e2e",
            treatment_axis="user_memory_recall",
            control_variant_id="user_memory_off",
            treatment_variant_id="user_memory_on_local_plus_everos",
        )
    )
    metrics.update(
        _reduce_e2e(
            trials,
            pairs,
            prefix="skill_e2e",
            treatment_axis="skill_sources",
            control_variant_id="user_memory_on_local_only",
            treatment_variant_id="user_memory_on_local_plus_everos",
        )
    )
    metrics.update(_reduce_memory_retrieval(retrieval))
    metrics.update(_reduce_skill_fusion(retrieval))
    metrics["memory_skill.measurement_valid"] = all(
        bool(metrics[name])
        for name in (
            "memory_e2e.coverage_valid",
            "memory_e2e.attempt_consistency_valid",
            "memory_e2e.variant_axis_valid",
            "skill_e2e.coverage_valid",
            "skill_e2e.attempt_consistency_valid",
            "skill_e2e.variant_axis_valid",
            "memory_retrieval.coverage_valid",
            "skill_fusion.coverage_valid",
        )
    )
    metrics["evidence.task_effect_level"] = (
        "real_agent_task"
        if metrics["memory_e2e.real_agent_task_effect_claim_eligible"]
        and metrics["skill_e2e.real_agent_task_effect_claim_eligible"]
        else "deterministic_contract"
    )
    metrics["evidence.retrieval_level"] = (
        "configured_everos"
        if metrics["memory_retrieval.real_semantic_claim_eligible"]
        and metrics["skill_fusion.real_semantic_claim_eligible"]
        else "deterministic_contract"
    )
    return metrics


def _reduce_e2e(
    records: list[Mapping[str, Any]],
    pair_results: list[Mapping[str, Any]],
    *,
    prefix: str,
    treatment_axis: str,
    control_variant_id: str,
    treatment_variant_id: str,
) -> dict[str, JsonValue]:
    by_trial: dict[tuple[str, str, str, int, str], Mapping[str, Any]] = {}
    for record in records:
        key = _mapping(record.get("key"))
        pack_id = str(key.get("pack_id", ""))
        if pack_id not in {_FORMAL_PACK_ID, _CALIBRATION_PACK_ID}:
            continue
        variant_id = str(key.get("variant_id", ""))
        if variant_id not in {control_variant_id, treatment_variant_id}:
            continue
        identity = (
            str(key.get("experiment_id", "")),
            pack_id,
            str(key.get("task_id", "")),
            int(key.get("repetition", -1)),
            variant_id,
        )
        by_trial[identity] = record

    declared_pairs: list[Mapping[str, Any]] = []
    pack_ids: set[str] = set()
    for pair in pair_results:
        key = _mapping(pair.get("key"))
        pack_id = str(key.get("pack_id", ""))
        if (
            pack_id in {_FORMAL_PACK_ID, _CALIBRATION_PACK_ID}
            and str(key.get("treatment_axis", "")) == treatment_axis
            and str(key.get("control_variant_id", "")) == control_variant_id
            and str(key.get("treatment_variant_id", "")) == treatment_variant_id
        ):
            declared_pairs.append(pair)
            pack_ids.add(pack_id)

    planned_pairs = sum(24 if pack_id == _FORMAL_PACK_ID else 8 for pack_id in pack_ids)
    deltas_by_task: dict[tuple[str, str], list[int]] = defaultdict(list)
    losses_by_task: dict[tuple[str, str], int] = defaultdict(int)
    valid_pairs = 0
    net_gains = 0
    evidence_rows: list[Mapping[str, Any]] = []
    accepted_pair_identities: set[tuple[str, str, str, int]] = set()
    attempt_consistency = True
    variant_axis_valid = True
    for pair in declared_pairs:
        key = _mapping(pair.get("key"))
        experiment_id = str(key.get("experiment_id", ""))
        pack_id = str(key.get("pack_id", ""))
        task_id = str(key.get("task_id", ""))
        repetition = int(key.get("repetition", -1))
        pair_identity = (experiment_id, pack_id, task_id, repetition)
        if pair_identity in accepted_pair_identities:
            attempt_consistency = False
            variant_axis_valid = False
            continue
        selected_attempt = pair.get("selected_block_attempt")
        control = by_trial.get(
            (
                experiment_id,
                pack_id,
                task_id,
                repetition,
                control_variant_id,
            )
        )
        treatment = by_trial.get(
            (
                experiment_id,
                pack_id,
                task_id,
                repetition,
                treatment_variant_id,
            )
        )
        if control is None or treatment is None:
            attempt_consistency = False
            continue
        actual_diff = _variant_diff(control, treatment)
        if set(actual_diff) != {treatment_axis} or pair.get("actual_variant_diff") != actual_diff:
            variant_axis_valid = False
            continue
        if (
            pair.get("valid") is not True
            or not isinstance(selected_attempt, int)
            or isinstance(selected_attempt, bool)
            or selected_attempt < 1
        ):
            continue
        if (
            control.get("selected_block_attempt") != selected_attempt
            or treatment.get("selected_block_attempt") != selected_attempt
            or not _trial_belongs_to_pair(control, key)
            or not _trial_belongs_to_pair(treatment, key)
            or not _same_plan_digest(pair, control, treatment)
        ):
            attempt_consistency = False
            continue
        if not _measurable_trial(control) or not _measurable_trial(treatment):
            continue
        accepted_pair_identities.add(pair_identity)
        valid_pairs += 1
        control_passed = str(control.get("status")) == "passed"
        treatment_passed = str(treatment.get("status")) == "passed"
        delta = int(treatment_passed) - int(control_passed)
        net_gains += delta
        deltas_by_task[(pack_id, task_id)].append(delta)
        if delta < 0:
            losses_by_task[(pack_id, task_id)] += 1
        evidence_rows.extend((control, treatment))

    positive_tasks = sum(sum(deltas) > 0 for deltas in deltas_by_task.values())
    regressed_two_of_three = sum(losses >= 2 for losses in losses_by_task.values())
    coverage = assess_pair_coverage(
        expected_pairs=planned_pairs,
        planned_pair_keys=(
            (
                ":".join(
                    (
                        str(_mapping(pair.get("key")).get("experiment_id", "")),
                        str(_mapping(pair.get("key")).get("pack_id", "")),
                        str(_mapping(pair.get("key")).get("task_id", "")),
                    )
                ),
                int(_mapping(pair.get("key")).get("repetition", -1)),
            )
            for pair in declared_pairs
        ),
        valid_pair_keys=(
            (
                ":".join((experiment_id, pack_id, task_id)),
                repetition,
            )
            for experiment_id, pack_id, task_id, repetition in accepted_pair_identities
        ),
    )
    coverage_valid = coverage.valid
    contract_valid = coverage_valid and net_gains >= 2 and positive_tasks >= 2 and regressed_two_of_three == 0
    real_effect_evidence = bool(evidence_rows) and all(
        _real_agent_task_effect_evidence_valid(record) for record in evidence_rows
    )
    attempt_consistency = bool(declared_pairs) and attempt_consistency
    variant_axis_valid = bool(declared_pairs) and variant_axis_valid
    return {
        f"{prefix}.planned_pairs": planned_pairs,
        f"{prefix}.valid_pairs": valid_pairs,
        f"{prefix}.coverage_valid": coverage_valid,
        f"{prefix}.attempt_consistency_valid": attempt_consistency,
        f"{prefix}.variant_axis_valid": variant_axis_valid,
        f"{prefix}.net_verifier_gains": net_gains,
        f"{prefix}.positive_tasks": positive_tasks,
        f"{prefix}.tasks_with_two_of_three_regressions": regressed_two_of_three,
        f"{prefix}.no_two_of_three_regressions": regressed_two_of_three == 0,
        f"{prefix}.claim_contract_valid": contract_valid,
        f"{prefix}.real_agent_task_effect_evidence_valid": real_effect_evidence,
        f"{prefix}.real_agent_task_effect_claim_eligible": (contract_valid and real_effect_evidence),
    }


def _real_agent_task_effect_evidence_valid(
    record: Mapping[str, Any],
) -> bool:
    metrics = _mapping(record.get("metrics"))
    return (
        metrics.get("real_agent_task_effect_claim_eligible") is True
        and metrics.get("memory.backend_class") == "EverosBackend"
        and metrics.get("memory.backend_adapter") == "production"
        and metrics.get("usage.complete") is True
        and metrics.get("cost.complete") is True
    )


def _reduce_memory_retrieval(
    records: list[Mapping[str, Any]],
) -> dict[str, JsonValue]:
    rows = [
        record
        for record in records
        if str(_mapping(record.get("key")).get("retrieval_suite_id", "")).startswith("user-memory-retrieval-")
    ]
    positive = [record for record in rows if record.get("label") == "positive"]
    negative = [record for record in rows if record.get("label") == "hard_negative"]
    expected_cases = _expected_retrieval_cases(rows, formal=80, calibration=10)
    coverage_valid = (
        expected_cases > 0
        and len(rows) == expected_cases
        and all(str(record.get("status")) == "measurable" for record in rows)
    )
    backend_recall_at_1 = _mean_recall(positive, field="ranked_results", limit=1)
    backend_recall_at_5 = _mean_recall(positive, field="ranked_results", limit=5)
    final_recall_at_5 = _mean_recall(
        positive,
        field="injected_results",
        limit=5,
    )
    final_precision_at_5 = _mean_precision(positive)
    irrelevant_injection_rate = 1.0 - final_precision_at_5 if final_precision_at_5 is not None else None
    mrr_at_5 = _mean_mrr(positive)
    hard_negative_rate = _rate(
        sum(bool(record.get("injected_results")) for record in negative),
        len(negative),
    )
    stale_ids, cross_ids = _memory_forbidden_ids(rows)
    stale_count = _selected_id_count(rows, stale_ids)
    cross_count = _selected_id_count(rows, cross_ids)
    quality_gate = (
        final_recall_at_5 is not None
        and final_recall_at_5 >= 0.80
        and irrelevant_injection_rate is not None
        and irrelevant_injection_rate <= 0.05
        and hard_negative_rate is not None
        and hard_negative_rate <= 0.05
        and stale_count == 0
        and cross_count == 0
    )
    semantic_evidence = bool(rows) and all(_real_semantic_evidence_valid(record) for record in rows)
    return {
        "memory_retrieval.planned_cases": expected_cases,
        "memory_retrieval.measurable_cases": sum(str(record.get("status")) == "measurable" for record in rows),
        "memory_retrieval.coverage_valid": coverage_valid,
        "memory_retrieval.backend_recall_at_1": backend_recall_at_1,
        "memory_retrieval.backend_recall_at_5": backend_recall_at_5,
        "memory_retrieval.final_injection_recall_at_5": final_recall_at_5,
        "memory_retrieval.final_injection_precision_at_5": (final_precision_at_5),
        "memory_retrieval.irrelevant_injection_rate": (irrelevant_injection_rate),
        "memory_retrieval.mrr_at_5": mrr_at_5,
        "memory_retrieval.hard_negative_injection_rate": hard_negative_rate,
        "memory_retrieval.stale_injection_count": stale_count,
        "memory_retrieval.cross_workspace_leakage_count": cross_count,
        "memory_retrieval.quality_gate_passed": quality_gate,
        "memory_retrieval.claim_contract_valid": coverage_valid and quality_gate,
        "memory_retrieval.deterministic_contract_claim_eligible": (coverage_valid and quality_gate),
        "memory_retrieval.real_semantic_evidence_valid": semantic_evidence,
        "memory_retrieval.real_semantic_claim_eligible": (coverage_valid and quality_gate and semantic_evidence),
    }


def _reduce_skill_fusion(
    records: list[Mapping[str, Any]],
) -> dict[str, JsonValue]:
    rows = [
        record
        for record in records
        if str(_mapping(record.get("key")).get("retrieval_suite_id", "")).startswith("skill-source-fusion-")
    ]
    expected_cases = _expected_retrieval_cases(rows, formal=180, calibration=24)
    coverage_valid = (
        expected_cases > 0
        and len(rows) == expected_cases
        and all(str(record.get("status")) == "measurable" for record in rows)
    )
    positive_by_configuration = {
        configuration: [
            record
            for record in rows
            if record.get("label") == "positive" and _configuration_id(record) == configuration
        ]
        for configuration in ("local_only", "everos_only", "fused")
    }
    recall = {
        configuration: _mean_recall(
            configuration_rows,
            field="injected_results",
            limit=5,
        )
        for configuration, configuration_rows in positive_by_configuration.items()
    }
    fused_recall = recall["fused"]
    singles = [value for value in (recall["local_only"], recall["everos_only"]) if value is not None]
    improvement = fused_recall - max(singles) if fused_recall is not None and singles else None
    fused_positive = positive_by_configuration["fused"]
    fused_negative = [
        record for record in rows if record.get("label") == "hard_negative" and _configuration_id(record) == "fused"
    ]
    hard_negative_rate = _rate(
        sum(bool(record.get("injected_results")) for record in fused_negative),
        len(fused_negative),
    )
    cross_count = _selected_id_count(rows, _skill_cross_ids(rows))
    source_contribution = {"local": 0, "everos": 0}
    for record in rows:
        if _configuration_id(record) != "fused":
            continue
        for result in _result_list(record, "injected_results"):
            for source in result.get("contributing_sources", []):
                source_name = str(source)
                if source_name in source_contribution:
                    source_contribution[source_name] += 1
    quality_gate = (
        improvement is not None
        and improvement >= 0.05
        and hard_negative_rate is not None
        and hard_negative_rate <= 0.05
        and cross_count == 0
    )
    semantic_evidence = bool(rows) and all(_real_semantic_evidence_valid(record) for record in rows)
    return {
        "skill_fusion.planned_cases": expected_cases,
        "skill_fusion.measurable_cases": sum(str(record.get("status")) == "measurable" for record in rows),
        "skill_fusion.coverage_valid": coverage_valid,
        "skill_fusion.local_recall_at_5": recall["local_only"],
        "skill_fusion.everos_recall_at_5": recall["everos_only"],
        "skill_fusion.fused_recall_at_5": fused_recall,
        "skill_fusion.fused_mrr_at_5": _mean_mrr(fused_positive),
        "skill_fusion.improvement_over_best_single_source": improvement,
        "skill_fusion.hard_negative_injection_rate": hard_negative_rate,
        "skill_fusion.cross_workspace_leakage_count": cross_count,
        "skill_fusion.local_source_contribution": source_contribution["local"],
        "skill_fusion.everos_source_contribution": source_contribution["everos"],
        "skill_fusion.quality_gate_passed": quality_gate,
        "skill_fusion.claim_contract_valid": coverage_valid and quality_gate,
        "skill_fusion.deterministic_contract_claim_eligible": (coverage_valid and quality_gate),
        "skill_fusion.real_semantic_evidence_valid": semantic_evidence,
        "skill_fusion.real_semantic_claim_eligible": (coverage_valid and quality_gate and semantic_evidence),
    }


def _real_semantic_evidence_valid(record: Mapping[str, Any]) -> bool:
    usage = _mapping(record.get("usage"))
    return (
        usage.get("everos_semantic_quality_claim_eligible") is True
        and str(usage.get("retrieval_evidence_level", "")) not in {"", "deterministic_contract"}
        and str(usage.get("backend_adapter", "")) not in {"", "injected_fixture"}
    )


def _materialize(
    records: Mapping[Any, Mapping[str, Any]] | Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if isinstance(records, Mapping):
        return list(records.values())
    return list(records)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _measurable_trial(record: Mapping[str, Any]) -> bool:
    return str(record.get("status")) in _MEASURABLE_TRIAL_STATUSES


def _variant_diff(
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
) -> dict[str, JsonValue]:
    control_settings = _mapping(control.get("observed_variant_settings"))
    treatment_settings = _mapping(treatment.get("observed_variant_settings"))
    return {
        key: {
            "control": control_settings.get(key),
            "treatment": treatment_settings.get(key),
        }
        for key in sorted(set(control_settings) | set(treatment_settings))
        if control_settings.get(key) != treatment_settings.get(key)
    }


def _trial_belongs_to_pair(
    trial: Mapping[str, Any],
    pair_key: Mapping[str, Any],
) -> bool:
    memberships = trial.get("pair_memberships", ())
    return isinstance(memberships, list | tuple) and any(_mapping(membership) == pair_key for membership in memberships)


def _same_plan_digest(
    pair: Mapping[str, Any],
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
) -> bool:
    digest = pair.get("plan_digest")
    return (
        isinstance(digest, str)
        and bool(digest)
        and control.get("plan_digest") == digest
        and treatment.get("plan_digest") == digest
    )


def _configuration_id(record: Mapping[str, Any]) -> str:
    return str(_mapping(record.get("key")).get("configuration_id", ""))


def _result_list(
    record: Mapping[str, Any],
    field: str,
) -> list[Mapping[str, Any]]:
    value = record.get(field)
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _expected_ids(record: Mapping[str, Any]) -> set[str]:
    values = record.get("expected_item_ids")
    if not isinstance(values, (list, tuple)):
        return set()
    return {str(value) for value in values}


def _mean_recall(
    records: list[Mapping[str, Any]],
    *,
    field: str,
    limit: int,
) -> float | None:
    values: list[float] = []
    for record in records:
        expected = _expected_ids(record)
        if not expected:
            continue
        observed = {str(result.get("item_id")) for result in _result_list(record, field)[:limit]}
        values.append(len(observed & expected) / len(expected))
    return mean(values) if values else None


def _mean_precision(records: list[Mapping[str, Any]]) -> float | None:
    values: list[float] = []
    for record in records:
        expected = _expected_ids(record)
        observed = {str(result.get("item_id")) for result in _result_list(record, "injected_results")[:5]}
        values.append(len(observed & expected) / len(observed) if observed else 0.0)
    return mean(values) if values else None


def _mean_mrr(records: list[Mapping[str, Any]]) -> float | None:
    values: list[float] = []
    for record in records:
        expected = _expected_ids(record)
        rank = next(
            (
                index
                for index, result in enumerate(
                    _result_list(record, "injected_results")[:5],
                    start=1,
                )
                if str(result.get("item_id")) in expected
            ),
            None,
        )
        values.append(0.0 if rank is None else 1.0 / rank)
    return mean(values) if values else None


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _selected_id_count(
    records: list[Mapping[str, Any]],
    selected_ids: set[str],
) -> int:
    return sum(
        str(result.get("item_id")) in selected_ids
        for record in records
        for result in _result_list(record, "injected_results")
    )


def _expected_retrieval_cases(
    records: list[Mapping[str, Any]],
    *,
    formal: int,
    calibration: int,
) -> int:
    suite_ids = {str(_mapping(record.get("key")).get("retrieval_suite_id", "")) for record in records}
    return sum(calibration if "calibration" in suite_id else formal for suite_id in suite_ids)


def _memory_forbidden_ids(
    records: list[Mapping[str, Any]],
) -> tuple[set[str], set[str]]:
    stale: set[str] = set()
    cross: set[str] = set()
    suite_ids = {str(_mapping(record.get("key")).get("retrieval_suite_id", "")) for record in records}
    for suite_id in suite_ids:
        if "calibration" in suite_id:
            stale.update(anonymous_item_id(suite_id, f"cal-memory-stale-{index:02d}") for index in range(6))
            cross.update(anonymous_item_id(suite_id, f"cal-memory-cross-{index:02d}") for index in range(4))
        else:
            stale.update(anonymous_item_id(suite_id, f"memory-stale-{index:03d}") for index in range(50))
            cross.update(anonymous_item_id(suite_id, f"memory-cross-{index:03d}") for index in range(30))
    return stale, cross


def _skill_cross_ids(records: list[Mapping[str, Any]]) -> set[str]:
    selected: set[str] = set()
    suite_ids = {str(_mapping(record.get("key")).get("retrieval_suite_id", "")) for record in records}
    for suite_id in suite_ids:
        if "calibration" in suite_id:
            selected.update(anonymous_item_id(suite_id, f"cal-skill-cross-{index:02d}") for index in range(4))
        else:
            selected.update(anonymous_item_id(suite_id, f"skill-cross-{index:03d}") for index in range(20))
    return selected


__all__ = ["reduce_memory_skill_claims"]
