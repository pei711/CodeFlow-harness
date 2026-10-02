from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from benchmarks.codeflowbench.coverage import assess_pair_coverage
from benchmarks.codeflowbench.schema import JsonValue

from .metrics import ContextPairMeasurement, assess_context_claim

_SUPPORTED_PACKS = {
    "context": (24, 8, 3),
    "context-calibration": (8, 4, 2),
}
_TREATMENT_AXIS = "history_manager"
_CONTROL_VARIANT = "context-fifo"
_TREATMENT_VARIANT = "context-curator"
_MEASURABLE_STATUSES = {"passed", "task_failed", "task_timeout"}
_CAPABILITY_CRITERIA = (
    "early_constraint_retained",
    "active_constraint_applied",
    "latest_decision_applied",
    "artifact_exact",
)


def reduce_context_artifacts(
    *,
    trial_records: Iterable[Mapping[str, Any]],
    pair_results: Iterable[Mapping[str, Any]],
) -> dict[str, JsonValue]:
    trials, structural_findings = _index_trials(trial_records)
    operational_findings: list[str] = []
    measurements: list[ContextPairMeasurement] = []
    pair_pack_ids: set[str] = set()
    seen_pairs: set[tuple[str, str, int]] = set()
    single_axis_valid = True

    for pair in pair_results:
        key = _mapping(pair.get("key"))
        if key is None:
            structural_findings.append("malformed_context_pair_key")
            continue
        pack_id = _string(key.get("pack_id"))
        if pack_id not in _SUPPORTED_PACKS:
            continue
        pair_pack_ids.add(pack_id)
        if key.get("treatment_axis") != _TREATMENT_AXIS:
            structural_findings.append("unexpected_context_treatment_axis")
            single_axis_valid = False
            continue
        task_id = _string(key.get("task_id"))
        repetition = _integer(key.get("repetition"))
        control_variant = _string(key.get("control_variant_id"))
        treatment_variant = _string(key.get("treatment_variant_id"))
        if (
            task_id is None
            or repetition is None
            or control_variant != _CONTROL_VARIANT
            or treatment_variant != _TREATMENT_VARIANT
        ):
            structural_findings.append("malformed_context_pair_identity")
            continue
        pair_identity = (pack_id, task_id, repetition)
        if pair_identity in seen_pairs:
            structural_findings.append(
                f"duplicate_context_pair:{pack_id}:{task_id}:{repetition}",
            )
            continue
        seen_pairs.add(pair_identity)
        control = trials.get(
            (pack_id, task_id, repetition, control_variant),
        )
        treatment = trials.get(
            (pack_id, task_id, repetition, treatment_variant),
        )
        if control is None or treatment is None:
            operational_findings.append(
                f"missing_context_pair_arm:{pack_id}:{task_id}:{repetition}",
            )
            continue
        actual_diff = _mapping(pair.get("actual_variant_diff"))
        if actual_diff is None or set(actual_diff) != {_TREATMENT_AXIS}:
            single_axis_valid = False
        measurement, integrity_findings, pair_findings = _measurement(
            pair=pair,
            control=control,
            treatment=treatment,
            task_id=task_id,
            repetition=repetition,
        )
        measurements.append(measurement)
        structural_findings.extend(integrity_findings)
        operational_findings.extend(pair_findings)

    if len(pair_pack_ids) > 1:
        structural_findings.append("mixed_context_tracks")
    expected_pairs = sum(_SUPPORTED_PACKS[pack_id][0] for pack_id in pair_pack_ids)
    planned_pair_keys: set[tuple[str, int]] = set()
    for pack_id in pair_pack_ids:
        _expected_pairs, expected_tasks, repetitions = _SUPPORTED_PACKS[pack_id]
        task_ids = {
            task_id for indexed_pack_id, task_id, _repetition, _variant_id in trials if indexed_pack_id == pack_id
        }
        if len(task_ids) != expected_tasks:
            structural_findings.append(
                f"context_task_denominator_mismatch:{len(task_ids)}:{expected_tasks}",
            )
        planned_pair_keys.update((task_id, repetition) for task_id in task_ids for repetition in range(repetitions))
    valid_measurements = [measurement for measurement in measurements if measurement.valid]
    coverage = assess_pair_coverage(
        expected_pairs=expected_pairs,
        planned_pair_keys=planned_pair_keys,
        valid_pair_keys=((measurement.task_id, measurement.repetition) for measurement in valid_measurements),
    )
    coverage_valid = coverage.valid
    if expected_pairs == 0:
        operational_findings.append("no_context_pairs")
    elif len(measurements) != expected_pairs:
        operational_findings.append(
            f"context_pair_denominator_mismatch:{len(measurements)}:{expected_pairs}",
        )

    assessment = assess_context_claim(tuple(measurements))
    measurable_measurements = [measurement for measurement in measurements if measurement.measurable]
    usage_complete = bool(measurable_measurements) and all(
        measurement.usage_complete for measurement in measurable_measurements
    )
    pass_count_noninferior = assessment.treatment_passes >= assessment.control_passes
    minimum_success_matched_coverage = assessment.covered_tasks >= 6
    minimum_lower_input_coverage = assessment.tasks_with_lower_trial_total >= 6
    no_two_of_three_regressions = not any(
        finding.startswith("task_lost_at_least_two_of_three_passes:") for finding in assessment.findings
    )
    measurement_valid = coverage_valid and usage_complete and single_axis_valid and not structural_findings
    combined_findings = list(
        dict.fromkeys(
            (
                *structural_findings,
                *operational_findings,
                *assessment.findings,
            )
        ),
    )
    reduction_percent = (
        assessment.equal_task_macro_reduction * 100.0 if assessment.equal_task_macro_reduction is not None else None
    )
    capability_metrics = _capability_metrics(
        trials=trials,
        pack_ids=pair_pack_ids,
        planned_pair_keys=planned_pair_keys,
    )
    capability_integrity_findings = (
        "malformed_context_trial_key",
        "malformed_context_trial_identity",
        "duplicate_context_trial:",
        "mixed_context_tracks",
        "context_task_denominator_mismatch:",
    )
    capability_measurement_valid = bool(
        capability_metrics["context.capability_evidence_complete"]
        and len(planned_pair_keys) == expected_pairs
        and single_axis_valid
        and not any(
            finding == prefix or finding.startswith(prefix)
            for finding in structural_findings
            for prefix in capability_integrity_findings
        )
    )
    return {
        "context.measurement_valid": measurement_valid,
        "context.capability_measurement_valid": capability_measurement_valid,
        "context.efficiency_measurement_valid": measurement_valid,
        "context.positive_claim_eligible": (measurement_valid and assessment.claim_eligible),
        "context.trial_total_input_token_reduction_percent": (reduction_percent),
        "context.coverage_valid": coverage_valid,
        "context.usage_complete": usage_complete,
        "context.single_axis_valid": single_axis_valid,
        "context.pass_count_noninferior": pass_count_noninferior,
        "context.minimum_success_matched_coverage": (minimum_success_matched_coverage),
        "context.minimum_lower_input_coverage": (minimum_lower_input_coverage),
        "context.no_two_of_three_pass_regressions": (no_two_of_three_regressions),
        "context.pair_measurement_count": len(measurements),
        "context.valid_pair_measurement_count": len(valid_measurements),
        "context.expected_pair_count": expected_pairs,
        "context.covered_tasks": assessment.covered_tasks,
        "context.tasks_with_lower_trial_total_input": (assessment.tasks_with_lower_trial_total),
        "context.control_pass_count": assessment.control_passes,
        "context.treatment_pass_count": assessment.treatment_passes,
        **capability_metrics,
        "context.result_scope": ("exploratory_eight_task_pack" if pair_pack_ids == {"context"} else "calibration_only"),
        "context.findings": combined_findings,
    }


def _index_trials(
    records: Iterable[Mapping[str, Any]],
) -> tuple[
    dict[tuple[str, str, int, str], Mapping[str, Any]],
    list[str],
]:
    indexed: dict[
        tuple[str, str, int, str],
        Mapping[str, Any],
    ] = {}
    findings: list[str] = []
    for record in records:
        key = _mapping(record.get("key"))
        if key is None:
            findings.append("malformed_context_trial_key")
            continue
        pack_id = _string(key.get("pack_id"))
        if pack_id not in _SUPPORTED_PACKS:
            continue
        task_id = _string(key.get("task_id"))
        repetition = _integer(key.get("repetition"))
        variant_id = _string(key.get("variant_id"))
        if task_id is None or repetition is None or variant_id not in {_CONTROL_VARIANT, _TREATMENT_VARIANT}:
            findings.append("malformed_context_trial_identity")
            continue
        identity = (pack_id, task_id, repetition, variant_id)
        if identity in indexed:
            findings.append(
                f"duplicate_context_trial:{pack_id}:{task_id}:{repetition}:{variant_id}",
            )
            continue
        indexed[identity] = record
    return indexed, findings


def _measurement(
    *,
    pair: Mapping[str, Any],
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
    task_id: str,
    repetition: int,
) -> tuple[ContextPairMeasurement, list[str], list[str]]:
    integrity_findings: list[str] = []
    operational_findings: list[str] = []
    pair_valid = pair.get("valid") is True
    if not isinstance(pair.get("valid"), bool):
        integrity_findings.append(
            f"invalid_context_pair_validity:{task_id}:{repetition}",
        )
    elif not pair_valid:
        operational_findings.append(
            f"context_pair_marked_invalid:{task_id}:{repetition}",
        )

    selected_attempt = _integer(pair.get("selected_block_attempt"))
    control_attempt = _integer(control.get("selected_block_attempt"))
    treatment_attempt = _integer(
        treatment.get("selected_block_attempt"),
    )
    attempts_match = selected_attempt is not None and selected_attempt == control_attempt == treatment_attempt
    if not attempts_match:
        integrity_findings.append(
            f"context_selected_attempt_drift:{task_id}:{repetition}",
        )

    actual_diff = _mapping(pair.get("actual_variant_diff"))
    single_axis = actual_diff is not None and set(actual_diff) == {_TREATMENT_AXIS}
    if not single_axis:
        integrity_findings.append(
            f"context_variant_axis_drift:{task_id}:{repetition}",
        )

    statuses_measurable = (
        control.get("status") in _MEASURABLE_STATUSES and treatment.get("status") in _MEASURABLE_STATUSES
    )
    if not statuses_measurable:
        operational_findings.append(
            f"nonmeasurable_context_pair:{task_id}:{repetition}",
        )
        return (
            ContextPairMeasurement(
                task_id=task_id,
                repetition=repetition,
                control_passed=False,
                treatment_passed=False,
                control_main_agent_input_tokens=0,
                treatment_main_agent_input_tokens=0,
                control_trial_total_input_tokens=0,
                treatment_trial_total_input_tokens=0,
                control_context_auxiliary_input_tokens=0,
                treatment_context_auxiliary_input_tokens=0,
                usage_complete=False,
                valid=False,
                measurable=False,
            ),
            integrity_findings,
            operational_findings,
        )

    control_metrics = _mapping(control.get("metrics"))
    treatment_metrics = _mapping(treatment.get("metrics"))
    if control_metrics is None or treatment_metrics is None:
        integrity_findings.append(
            f"missing_context_metrics:{task_id}:{repetition}",
        )
        control_metrics = control_metrics or {}
        treatment_metrics = treatment_metrics or {}

    numeric_fields = (
        "main_agent_input_tokens",
        "trial_total_input_tokens",
        "context_auxiliary_input_tokens",
    )
    parsed: dict[str, int] = {}
    numeric_valid = True
    for arm, metrics in (
        ("control", control_metrics),
        ("treatment", treatment_metrics),
    ):
        arm_numeric_valid = True
        for field in numeric_fields:
            value = _nonnegative_integer(metrics.get(field))
            if value is None:
                numeric_valid = False
                arm_numeric_valid = False
                integrity_findings.append(
                    f"invalid_context_{field}:{arm}:{task_id}:{repetition}",
                )
                value = 0
            parsed[f"{arm}_{field}"] = value
        if arm_numeric_valid and (
            parsed[f"{arm}_trial_total_input_tokens"]
            < parsed[f"{arm}_main_agent_input_tokens"] + parsed[f"{arm}_context_auxiliary_input_tokens"]
        ):
            numeric_valid = False
            integrity_findings.append(
                f"context_usage_subtotal_exceeds_total:{arm}:{task_id}:{repetition}",
            )

    control_usage_complete = control_metrics.get("usage_complete")
    treatment_usage_complete = treatment_metrics.get("usage_complete")
    if not isinstance(control_usage_complete, bool) or not isinstance(
        treatment_usage_complete,
        bool,
    ):
        integrity_findings.append(
            f"invalid_context_usage_flag:{task_id}:{repetition}",
        )
    usage_complete = control_usage_complete is True and treatment_usage_complete is True
    if not usage_complete:
        integrity_findings.append(
            f"context_usage_incomplete:{task_id}:{repetition}",
        )

    valid = pair_valid and attempts_match and single_axis and numeric_valid and statuses_measurable and usage_complete
    return (
        ContextPairMeasurement(
            task_id=task_id,
            repetition=repetition,
            control_passed=control.get("status") == "passed",
            treatment_passed=treatment.get("status") == "passed",
            control_main_agent_input_tokens=parsed["control_main_agent_input_tokens"],
            treatment_main_agent_input_tokens=parsed["treatment_main_agent_input_tokens"],
            control_trial_total_input_tokens=parsed["control_trial_total_input_tokens"],
            treatment_trial_total_input_tokens=parsed["treatment_trial_total_input_tokens"],
            control_context_auxiliary_input_tokens=parsed["control_context_auxiliary_input_tokens"],
            treatment_context_auxiliary_input_tokens=parsed["treatment_context_auxiliary_input_tokens"],
            usage_complete=usage_complete,
            valid=valid,
            measurable=True,
        ),
        integrity_findings,
        operational_findings,
    )


def _capability_metrics(
    *,
    trials: Mapping[tuple[str, str, int, str], Mapping[str, Any]],
    pack_ids: set[str],
    planned_pair_keys: set[tuple[str, int]],
) -> dict[str, JsonValue]:
    expected_trials = len(planned_pair_keys)
    passed = {criterion: 0 for criterion in _CAPABILITY_CRITERIA}
    evidence_complete = len(pack_ids) == 1
    pack_id = next(iter(pack_ids), "")
    for task_id, repetition in sorted(planned_pair_keys):
        trial = trials.get(
            (pack_id, task_id, repetition, _TREATMENT_VARIANT),
        )
        if trial is None:
            evidence_complete = False
            continue
        metrics = _mapping(trial.get("metrics")) or {}
        for criterion in _CAPABILITY_CRITERIA:
            value = metrics.get(criterion)
            if isinstance(value, bool):
                passed[criterion] += int(value)
            elif trial.get("status") in _MEASURABLE_STATUSES:
                evidence_complete = False
    denominator = expected_trials * len(_CAPABILITY_CRITERIA)
    result: dict[str, JsonValue] = {
        "context.capability_evidence_complete": (evidence_complete and expected_trials > 0),
        "context.treatment_capability_score_rate": (sum(passed.values()) / denominator if denominator else None),
    }
    for criterion, count in passed.items():
        result[f"context.treatment_{criterion}_rate"] = count / expected_trials if expected_trials else None
    return result


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _nonnegative_integer(value: Any) -> int | None:
    parsed = _integer(value)
    return parsed if parsed is not None and parsed >= 0 else None


__all__ = ["reduce_context_artifacts"]
