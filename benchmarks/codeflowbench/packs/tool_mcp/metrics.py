from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from importlib.metadata import version
from statistics import mean
from typing import Any

import tiktoken

from benchmarks.codeflowbench.canonical import canonical_digest, canonical_json
from codeflow.spine import ToolEvent, ToolPhase

from .models import TargetCallRecord, TargetCallSummary

TOOL_SCHEMA_ESTIMATOR_ID = f"tiktoken:{version('tiktoken')}:cl100k_base"
TOOL_SCHEMA_ESTIMATOR_DIGEST = canonical_digest(TOOL_SCHEMA_ESTIMATOR_ID)
_INVALID_STATUSES = {
    "invalid_envelope",
    "unknown_target",
    "validation_failed",
}


def estimate_visible_tool_schema_tokens(
    tools: list[dict[str, Any]],
) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(canonical_json(tools)))


def normalize_target_calls(
    events: tuple[ToolEvent, ...],
    *,
    catalog_names: frozenset[str],
    initially_visible_names: frozenset[str],
    expected_first_target: str,
) -> TargetCallSummary:
    completions: dict[str, deque[ToolEvent]] = defaultdict(deque)
    for event in events:
        if event.phase is ToolPhase.COMPLETE:
            completions[event.tool_call_id].append(event)
    meta_invocations: Counter[str] = Counter()
    meta_failures: Counter[str] = Counter()
    records: list[TargetCallRecord] = []
    for event in events:
        if event.phase is not ToolPhase.START:
            continue
        queued = completions[event.tool_call_id]
        completion = queued.popleft() if queued else None
        if event.name in {"tool_search", "tool_call"}:
            meta_invocations[event.name] += 1
            if completion is not None and completion.failed:
                meta_failures[event.name] += 1
        if event.name == "tool_search":
            continue
        normalized = _normalize_start(
            event,
            completion,
            catalog_names=catalog_names,
            initially_visible_names=initially_visible_names,
        )
        if normalized is not None:
            records.append(normalized)

    first_accuracy = float(records[0].target_name == expected_first_target) if records else None
    invalid_rate = (
        sum(record.dispatch_status in _INVALID_STATUSES for record in records) / len(records) if records else None
    )
    seen: Counter[str] = Counter()
    repeats = 0
    for record in records:
        seen[record.canonical_key] += 1
        if seen[record.canonical_key] > 1:
            repeats += 1
    repeat_rate = repeats / len(records) if records else None
    return TargetCallSummary(
        records=tuple(records),
        first_target_accuracy=first_accuracy,
        invalid_target_call_rate=invalid_rate,
        exact_target_repeat_rate=repeat_rate,
        meta_tool_invocations={
            "tool_call": meta_invocations["tool_call"],
            "tool_search": meta_invocations["tool_search"],
        },
        meta_tool_failures={
            "tool_call": meta_failures["tool_call"],
            "tool_search": meta_failures["tool_search"],
        },
    )


def _normalize_start(
    event: ToolEvent,
    completion: ToolEvent | None,
    *,
    catalog_names: frozenset[str],
    initially_visible_names: frozenset[str],
) -> TargetCallRecord | None:
    raw_arguments = event.arguments or {}
    route = "direct_catalog"
    status = "succeeded"
    if event.name == "tool_call":
        route = "tool_call"
        target_name = raw_arguments.get("name")
        target_arguments = raw_arguments.get("arguments", {})
        if isinstance(target_arguments, str):
            try:
                target_arguments = json.loads(target_arguments)
            except json.JSONDecodeError:
                target_arguments = None
        if not isinstance(target_name, str) or not isinstance(
            target_arguments,
            dict,
        ):
            target_name = target_name if isinstance(target_name, str) else "__invalid__"
            target_arguments = {}
            status = "invalid_envelope"
    elif event.name in catalog_names or event.name.startswith("mcp_codeflowbench_"):
        target_name = event.name
        target_arguments = raw_arguments
        if event.name not in initially_visible_names:
            route = "direct_hidden"
    else:
        return None

    if status != "invalid_envelope":
        if target_name not in catalog_names:
            status = "unknown_target"
        elif completion is None:
            status = "target_failed"
        elif completion.failed:
            status = "validation_failed" if "Invalid parameters" in completion.result_preview else "target_failed"
    canonical_key = (
        f"{target_name}:{canonical_json(target_arguments)}"
        if status != "invalid_envelope"
        else f"invalid:{canonical_digest(raw_arguments)}"
    )
    result_preview = completion.result_preview if completion is not None else ""
    receipt = _receipt_from_preview(result_preview)
    return TargetCallRecord(
        call_id=event.tool_call_id,
        target_name=target_name,
        arguments=target_arguments,
        route=route,
        dispatch_status=status,
        canonical_key=canonical_key,
        result_preview=result_preview,
        receipt=receipt,
    )


def _receipt_from_preview(preview: str) -> str | None:
    try:
        payload = json.loads(preview)
    except (json.JSONDecodeError, TypeError):
        return None
    receipt = payload.get("receipt") if isinstance(payload, dict) else None
    return receipt if isinstance(receipt, str) else None


@dataclass(frozen=True)
class ToolMCPPairMeasurement:
    task_id: str
    repetition: int
    control_passed: bool
    treatment_passed: bool
    control_schema_tokens: int
    treatment_schema_tokens: int
    control_invalid_target_call_rate: float | None
    treatment_invalid_target_call_rate: float | None
    control_exact_target_repeat_rate: float | None
    treatment_exact_target_repeat_rate: float | None
    usage_complete: bool
    valid: bool
    initial_visible_sets_differ: bool
    control_mcp_connected: bool
    treatment_mcp_connected: bool


@dataclass(frozen=True)
class ToolMCPClaimAssessment:
    claim_eligible: bool
    covered_tasks: int
    tasks_with_lower_schema_tokens: int
    control_passes: int
    treatment_passes: int
    equal_task_macro_reduction: float | None
    findings: tuple[str, ...]
    exploratory: bool = True


def assess_tool_mcp_claim(
    measurements: tuple[ToolMCPPairMeasurement, ...],
) -> ToolMCPClaimAssessment:
    by_task: dict[str, list[ToolMCPPairMeasurement]] = defaultdict(list)
    for measurement in measurements:
        by_task[measurement.task_id].append(measurement)
    valid = [measurement for measurement in measurements if measurement.valid]
    control_passes = sum(measurement.control_passed for measurement in valid)
    treatment_passes = sum(measurement.treatment_passed for measurement in valid)
    reductions: list[float] = []
    tasks_with_lower = 0
    regression_tasks: list[str] = []
    usable_pairs: list[ToolMCPPairMeasurement] = []
    for task_id, task_measurements in sorted(by_task.items()):
        usable = [
            measurement
            for measurement in task_measurements
            if measurement.valid
            and measurement.usage_complete
            and measurement.control_passed
            and measurement.treatment_passed
        ]
        if len(usable) >= 2:
            control_mean = mean(measurement.control_schema_tokens for measurement in usable)
            treatment_mean = mean(measurement.treatment_schema_tokens for measurement in usable)
            if control_mean > 0:
                reductions.append(1 - treatment_mean / control_mean)
                usable_pairs.extend(usable)
                if treatment_mean < control_mean:
                    tasks_with_lower += 1
        if (
            sum(measurement.control_passed for measurement in task_measurements if measurement.valid)
            - sum(measurement.treatment_passed for measurement in task_measurements if measurement.valid)
            >= 2
        ):
            regression_tasks.append(task_id)

    macro = mean(reductions) if reductions else None
    findings: list[str] = []
    if len(reductions) < 6:
        findings.append("fewer_than_six_tasks_have_two_success_matched_pairs")
    if macro is None or macro < 0.5:
        findings.append("schema_token_reduction_below_50_percent")
    if tasks_with_lower < 6:
        findings.append("fewer_than_six_tasks_use_less_schema_tokens")
    if treatment_passes < control_passes:
        findings.append("treatment_pass_count_below_control")
    if regression_tasks:
        findings.append("task_lost_at_least_two_of_three_passes:" + ",".join(regression_tasks))
    if any(
        measurement.control_invalid_target_call_rate is None
        or measurement.treatment_invalid_target_call_rate is None
        or measurement.control_exact_target_repeat_rate is None
        or measurement.treatment_exact_target_repeat_rate is None
        for measurement in usable_pairs
    ):
        findings.append("null_target_call_rate")
    elif usable_pairs:
        if mean(measurement.treatment_invalid_target_call_rate for measurement in usable_pairs) > mean(
            measurement.control_invalid_target_call_rate for measurement in usable_pairs
        ):
            findings.append("invalid_target_call_rate_worse")
        if mean(measurement.treatment_exact_target_repeat_rate for measurement in usable_pairs) > mean(
            measurement.control_exact_target_repeat_rate for measurement in usable_pairs
        ):
            findings.append("exact_target_repeat_rate_worse")
    if any(not measurement.initial_visible_sets_differ for measurement in valid):
        findings.append("initial_visible_tool_sets_do_not_differ")
    if any(not measurement.control_mcp_connected or not measurement.treatment_mcp_connected for measurement in valid):
        findings.append("silent_mcp_connection_failure")

    return ToolMCPClaimAssessment(
        claim_eligible=not findings,
        covered_tasks=len(reductions),
        tasks_with_lower_schema_tokens=tasks_with_lower,
        control_passes=control_passes,
        treatment_passes=treatment_passes,
        equal_task_macro_reduction=macro,
        findings=tuple(findings),
    )


__all__ = [
    "TOOL_SCHEMA_ESTIMATOR_DIGEST",
    "TOOL_SCHEMA_ESTIMATOR_ID",
    "ToolMCPClaimAssessment",
    "ToolMCPPairMeasurement",
    "assess_tool_mcp_claim",
    "estimate_visible_tool_schema_tokens",
    "normalize_target_calls",
]
