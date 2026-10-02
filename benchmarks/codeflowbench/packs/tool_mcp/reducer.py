from __future__ import annotations

from collections.abc import Iterable, Mapping
from statistics import mean
from typing import Any

from benchmarks.codeflowbench.coverage import assess_pair_coverage
from benchmarks.codeflowbench.fixtures.mcp import (
    MCP_CATALOG_SIZE,
    catalog_digest,
)

from .metrics import ToolMCPPairMeasurement, assess_tool_mcp_claim

_PACK_IDS = frozenset({"tool-mcp", "tool-mcp-calibration"})
_PACK_POLICIES = {
    "tool-mcp": (24, 8, 3),
    "tool-mcp-calibration": (8, 4, 2),
}
_TREATMENT_AXIS = "tool_disclosure"
_MEASURABLE_STATUSES = {"passed", "task_failed", "task_timeout"}


def reduce_tool_mcp_claim_from_artifacts(
    *,
    trial_records: Iterable[Mapping[str, Any]],
    pair_results: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    trials, structural_findings = _index_trials(trial_records)
    operational_findings: list[str] = []
    measurements: list[ToolMCPPairMeasurement] = []
    seen_pairs: set[tuple[str, str, int]] = set()
    selected_pack_ids: set[str] = set()
    for pair in pair_results:
        key = _mapping(pair.get("key"))
        if key is None or key.get("pack_id") not in _PACK_IDS:
            continue
        pack_id = str(key["pack_id"])
        selected_pack_ids.add(pack_id)
        if key.get("treatment_axis") != _TREATMENT_AXIS:
            structural_findings.append("unexpected_tool_mcp_treatment_axis")
            continue
        task_id = _string(key.get("task_id"))
        repetition = _integer(key.get("repetition"))
        control_variant = _string(key.get("control_variant_id"))
        treatment_variant = _string(key.get("treatment_variant_id"))
        if None in {
            task_id,
            repetition,
            control_variant,
            treatment_variant,
        }:
            structural_findings.append("malformed_tool_mcp_pair_key")
            continue
        pair_identity = (pack_id, task_id, repetition)
        if pair_identity in seen_pairs:
            structural_findings.append(
                f"duplicate_tool_mcp_pair:{task_id}:{repetition}",
            )
            continue
        seen_pairs.add(pair_identity)
        control = trials.get((pack_id, task_id, repetition, control_variant))
        treatment = trials.get((pack_id, task_id, repetition, treatment_variant))
        if control is None or treatment is None:
            operational_findings.append(
                f"missing_tool_mcp_pair_arm:{task_id}:{repetition}",
            )
            continue
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

    if len(selected_pack_ids) > 1:
        structural_findings.append("mixed_tool_mcp_pack_artifacts")
    expected_pairs = sum(_PACK_POLICIES[pack_id][0] for pack_id in selected_pack_ids)
    planned_pair_keys: set[tuple[str, int]] = set()
    for pack_id in selected_pack_ids:
        _pair_count, expected_tasks, repetitions = _PACK_POLICIES[pack_id]
        task_ids = {
            task_id for indexed_pack_id, task_id, _repetition, _variant_id in trials if indexed_pack_id == pack_id
        }
        if len(task_ids) != expected_tasks:
            structural_findings.append(
                f"tool_mcp_task_denominator_mismatch:{len(task_ids)}:{expected_tasks}",
            )
        planned_pair_keys.update((task_id, repetition) for task_id in task_ids for repetition in range(repetitions))
    assessment = assess_tool_mcp_claim(tuple(measurements))
    valid_measurements = [measurement for measurement in measurements if measurement.valid]
    coverage = assess_pair_coverage(
        expected_pairs=expected_pairs,
        planned_pair_keys=planned_pair_keys,
        valid_pair_keys=((measurement.task_id, measurement.repetition) for measurement in valid_measurements),
    )
    coverage_valid = coverage.valid
    success_matched = [
        measurement for measurement in valid_measurements if measurement.control_passed and measurement.treatment_passed
    ]
    usage_complete = [measurement for measurement in success_matched if measurement.usage_complete]
    rate_complete = [
        measurement
        for measurement in usage_complete
        if measurement.control_invalid_target_call_rate is not None
        and measurement.treatment_invalid_target_call_rate is not None
        and measurement.control_exact_target_repeat_rate is not None
        and measurement.treatment_exact_target_repeat_rate is not None
    ]
    visible_sets_valid = bool(valid_measurements) and all(
        measurement.initial_visible_sets_differ for measurement in valid_measurements
    )
    mcp_arms_valid = bool(valid_measurements) and all(
        measurement.control_mcp_connected and measurement.treatment_mcp_connected for measurement in valid_measurements
    )
    if len(usage_complete) != len(success_matched):
        structural_findings.append("incomplete_success_matched_usage")
    if len(rate_complete) != len(usage_complete):
        structural_findings.append("null_success_matched_target_call_rate")
    if not visible_sets_valid:
        structural_findings.append("initial_visible_tool_sets_do_not_differ")
    if not mcp_arms_valid:
        structural_findings.append("silent_mcp_connection_failure")
    if len(measurements) != expected_pairs:
        operational_findings.append(
            f"tool_mcp_pair_denominator_mismatch:{len(measurements)}:{expected_pairs}",
        )
    measurement_valid = coverage_valid and not structural_findings
    combined_findings = tuple(
        dict.fromkeys(
            (
                *structural_findings,
                *operational_findings,
                *assessment.findings,
            )
        )
    )
    no_two_of_three_regressions = not any(
        finding.startswith("task_lost_at_least_two_of_three_passes:") for finding in assessment.findings
    )
    invalid_target_call_rate_delta = _rate_delta(
        rate_complete,
        "control_invalid_target_call_rate",
        "treatment_invalid_target_call_rate",
    )
    exact_target_repeat_rate_delta = _rate_delta(
        rate_complete,
        "control_exact_target_repeat_rate",
        "treatment_exact_target_repeat_rate",
    )
    calibration = selected_pack_ids == {"tool-mcp-calibration"}
    return {
        "tool_mcp.measurement_valid": measurement_valid,
        "tool_mcp.positive_claim_eligible": (measurement_valid and assessment.claim_eligible),
        "tool_mcp.coverage_valid": coverage_valid,
        "tool_mcp.pair_measurement_count": len(measurements),
        "tool_mcp.valid_pair_measurement_count": len(valid_measurements),
        "tool_mcp.success_matched_pair_count": len(success_matched),
        "tool_mcp.success_matched_usage_complete_pair_count": len(usage_complete),
        "tool_mcp.rate_complete_success_matched_pair_count": len(rate_complete),
        "tool_mcp.covered_tasks": assessment.covered_tasks,
        "tool_mcp.tasks_with_lower_schema_tokens": (assessment.tasks_with_lower_schema_tokens),
        "tool_mcp.control_pass_count": assessment.control_passes,
        "tool_mcp.treatment_pass_count": assessment.treatment_passes,
        "tool_mcp.equal_task_macro_schema_token_reduction_percent": (
            assessment.equal_task_macro_reduction * 100.0 if assessment.equal_task_macro_reduction is not None else None
        ),
        "tool_mcp.minimum_success_matched_coverage": assessment.covered_tasks >= 6,
        "tool_mcp.pass_count_noninferior": assessment.treatment_passes >= assessment.control_passes,
        "tool_mcp.no_two_of_three_pass_regressions": no_two_of_three_regressions,
        "tool_mcp.invalid_target_call_rate_noninferior": (
            invalid_target_call_rate_delta is not None and invalid_target_call_rate_delta <= 0
        ),
        "tool_mcp.exact_target_repeat_rate_noninferior": (
            exact_target_repeat_rate_delta is not None and exact_target_repeat_rate_delta <= 0
        ),
        "tool_mcp.invalid_target_call_rate_delta": invalid_target_call_rate_delta,
        "tool_mcp.exact_target_repeat_rate_delta": exact_target_repeat_rate_delta,
        "tool_mcp.all_initial_visible_sets_differ": visible_sets_valid,
        "tool_mcp.all_mcp_arms_connected": mcp_arms_valid,
        "tool_mcp.result_scope": ("calibration_only" if calibration else "exploratory_eight_task_pack"),
        "tool_mcp.findings": list(combined_findings),
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
        if key is None or key.get("pack_id") not in _PACK_IDS:
            continue
        pack_id = str(key["pack_id"])
        task_id = _string(key.get("task_id"))
        repetition = _integer(key.get("repetition"))
        variant_id = _string(key.get("variant_id"))
        if task_id is None or repetition is None or variant_id is None:
            findings.append("malformed_tool_mcp_trial_key")
            continue
        identity = (pack_id, task_id, repetition, variant_id)
        if identity in indexed:
            findings.append(f"duplicate_tool_mcp_trial:{task_id}:{repetition}:{variant_id}")
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
) -> tuple[ToolMCPPairMeasurement, list[str], list[str]]:
    integrity_findings: list[str] = []
    operational_findings: list[str] = []
    pair_valid = pair.get("valid") is True
    if not isinstance(pair.get("valid"), bool):
        integrity_findings.append(
            f"invalid_tool_mcp_pair_validity:{task_id}:{repetition}",
        )
    elif not pair_valid:
        operational_findings.append(
            f"tool_mcp_pair_marked_invalid:{task_id}:{repetition}",
        )
    selected_attempt = _integer(pair.get("selected_block_attempt"))
    control_attempt = _integer(control.get("selected_block_attempt"))
    treatment_attempt = _integer(treatment.get("selected_block_attempt"))
    attempts_match = selected_attempt is not None and selected_attempt == control_attempt == treatment_attempt
    if not attempts_match:
        integrity_findings.append(
            f"tool_mcp_selected_attempt_drift:{task_id}:{repetition}",
        )
    actual_diff = _mapping(pair.get("actual_variant_diff"))
    single_axis = actual_diff is not None and set(actual_diff) == {_TREATMENT_AXIS}
    if not single_axis:
        integrity_findings.append(
            f"tool_mcp_variant_axis_drift:{task_id}:{repetition}",
        )
    statuses_measurable = (
        control.get("status") in _MEASURABLE_STATUSES and treatment.get("status") in _MEASURABLE_STATUSES
    )
    if not statuses_measurable:
        operational_findings.append(
            f"nonmeasurable_tool_mcp_pair:{task_id}:{repetition}",
        )
        return (
            ToolMCPPairMeasurement(
                task_id=task_id,
                repetition=repetition,
                control_passed=False,
                treatment_passed=False,
                control_schema_tokens=0,
                treatment_schema_tokens=0,
                control_invalid_target_call_rate=None,
                treatment_invalid_target_call_rate=None,
                control_exact_target_repeat_rate=None,
                treatment_exact_target_repeat_rate=None,
                usage_complete=False,
                valid=False,
                initial_visible_sets_differ=False,
                control_mcp_connected=False,
                treatment_mcp_connected=False,
            ),
            integrity_findings,
            operational_findings,
        )

    control_metrics = _mapping(control.get("metrics"))
    treatment_metrics = _mapping(treatment.get("metrics"))
    if control_metrics is None or treatment_metrics is None:
        integrity_findings.append(
            f"missing_tool_mcp_metrics:{task_id}:{repetition}",
        )
        control_metrics = control_metrics or {}
        treatment_metrics = treatment_metrics or {}

    control_schema = _nonnegative_integer(control_metrics.get("trial_total_estimated_visible_tool_schema_tokens"))
    treatment_schema = _nonnegative_integer(treatment_metrics.get("trial_total_estimated_visible_tool_schema_tokens"))
    if control_schema is None or treatment_schema is None:
        integrity_findings.append(
            f"invalid_tool_mcp_schema_tokens:{task_id}:{repetition}",
        )

    control_visible = _string_set(control_metrics.get("initial_visible_tool_names"))
    treatment_visible = _string_set(treatment_metrics.get("initial_visible_tool_names"))
    if control_visible is None or treatment_visible is None:
        integrity_findings.append(
            f"missing_tool_mcp_visible_tool_set:{task_id}:{repetition}",
        )
    visible_sets_differ = (
        control_visible is not None and treatment_visible is not None and control_visible != treatment_visible
    )

    control_usage = control_metrics.get("usage_complete")
    treatment_usage = treatment_metrics.get("usage_complete")
    if not isinstance(control_usage, bool) or not isinstance(
        treatment_usage,
        bool,
    ):
        integrity_findings.append(
            f"invalid_tool_mcp_usage_flag:{task_id}:{repetition}",
        )
    usage_complete = control_usage is True and treatment_usage is True

    if control_metrics.get("mcp_catalog_digest") != treatment_metrics.get("mcp_catalog_digest"):
        integrity_findings.append(
            f"tool_mcp_catalog_digest_drift:{task_id}:{repetition}",
        )

    control_model_evidence = _model_evidence_findings(
        control_metrics,
        arm="control",
        task_id=task_id,
        repetition=repetition,
    )
    treatment_model_evidence = _model_evidence_findings(
        treatment_metrics,
        arm="treatment",
        task_id=task_id,
        repetition=repetition,
    )
    integrity_findings.extend(control_model_evidence)
    integrity_findings.extend(treatment_model_evidence)
    model_evidence_valid = not control_model_evidence and not treatment_model_evidence

    return (
        ToolMCPPairMeasurement(
            task_id=task_id,
            repetition=repetition,
            control_passed=control.get("status") == "passed",
            treatment_passed=treatment.get("status") == "passed",
            control_schema_tokens=control_schema or 0,
            treatment_schema_tokens=treatment_schema or 0,
            control_invalid_target_call_rate=_rate(control_metrics.get("invalid_target_call_rate")),
            treatment_invalid_target_call_rate=_rate(treatment_metrics.get("invalid_target_call_rate")),
            control_exact_target_repeat_rate=_rate(control_metrics.get("exact_target_repeat_rate")),
            treatment_exact_target_repeat_rate=_rate(treatment_metrics.get("exact_target_repeat_rate")),
            usage_complete=usage_complete,
            valid=(
                pair_valid
                and attempts_match
                and single_axis
                and control_schema is not None
                and treatment_schema is not None
                and model_evidence_valid
            ),
            initial_visible_sets_differ=visible_sets_differ,
            control_mcp_connected=_mcp_connected(control_metrics),
            treatment_mcp_connected=_mcp_connected(treatment_metrics),
        ),
        integrity_findings,
        operational_findings,
    )


def _model_evidence_findings(
    metrics: Mapping[str, Any],
    *,
    arm: str,
    task_id: str,
    repetition: int,
) -> list[str]:
    provider_model = _string(metrics.get("provider_model"))
    records = metrics.get("model_call_records")
    actual_model_names = _string_set(metrics.get("actual_model_names"))
    prefix = f"{arm}:{task_id}:{repetition}"
    if provider_model is None:
        return [f"missing_tool_mcp_provider_model:{prefix}"]
    if not isinstance(records, list) or not records:
        return [f"missing_tool_mcp_model_call_records:{prefix}"]
    findings: list[str] = []
    observed_actual_models: set[str] = set()
    for index, raw_record in enumerate(records):
        record = _mapping(raw_record)
        if record is None:
            findings.append(
                f"malformed_tool_mcp_model_call_record:{prefix}:{index}",
            )
            continue
        requested_model = _string(record.get("requested_model"))
        if requested_model != provider_model:
            findings.append(
                f"tool_mcp_requested_model_drift:{prefix}:{index}",
            )
        if record.get("finish_reason") == "exception":
            continue
        actual_model = _string(record.get("model"))
        if actual_model is None:
            findings.append(
                f"missing_tool_mcp_actual_model:{prefix}:{index}",
            )
            continue
        observed_actual_models.add(actual_model)
        if not _models_equivalent(provider_model, actual_model):
            findings.append(
                f"tool_mcp_actual_model_drift:{prefix}:{index}",
            )
    if actual_model_names is None:
        findings.append(f"missing_tool_mcp_actual_model_names:{prefix}")
    elif actual_model_names != observed_actual_models:
        findings.append(f"tool_mcp_actual_model_names_drift:{prefix}")
    return findings


def _models_equivalent(requested_model: str, actual_model: str) -> bool:
    if requested_model == actual_model:
        return True
    return "/" in requested_model and "/" not in actual_model and requested_model.rsplit("/", 1)[-1] == actual_model


def _mcp_connected(metrics: Mapping[str, Any]) -> bool:
    runtime_digest = metrics.get("mcp_catalog_digest")
    return (
        metrics.get("mcp_connected") is True
        and metrics.get("mcp_transport") == "stdio"
        and metrics.get("mcp_catalog_count") == MCP_CATALOG_SIZE
        and metrics.get("mcp_catalog_source_digest") == catalog_digest()
        and isinstance(runtime_digest, str)
        and len(runtime_digest) == 64
        and all(character in "0123456789abcdef" for character in runtime_digest)
    )


def _rate_delta(
    measurements: list[ToolMCPPairMeasurement],
    control_field: str,
    treatment_field: str,
) -> float | None:
    if not measurements:
        return None
    return mean(
        float(getattr(measurement, treatment_field)) - float(getattr(measurement, control_field))
        for measurement in measurements
    )


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _nonnegative_integer(value: Any) -> int | None:
    parsed = _integer(value)
    return parsed if parsed is not None and parsed >= 0 else None


def _rate(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    converted = float(value)
    return converted if 0.0 <= converted <= 1.0 else None


def _string_set(value: Any) -> frozenset[str] | None:
    if not isinstance(value, list | tuple):
        return None
    if any(not isinstance(item, str) for item in value):
        return None
    return frozenset(value)


__all__ = ["reduce_tool_mcp_claim_from_artifacts"]
