from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class ContextPairMeasurement:
    task_id: str
    repetition: int
    control_passed: bool
    treatment_passed: bool
    control_main_agent_input_tokens: int
    treatment_main_agent_input_tokens: int
    control_trial_total_input_tokens: int
    treatment_trial_total_input_tokens: int
    control_context_auxiliary_input_tokens: int
    treatment_context_auxiliary_input_tokens: int
    usage_complete: bool
    valid: bool
    measurable: bool = True


@dataclass(frozen=True)
class ContextClaimAssessment:
    claim_eligible: bool
    covered_tasks: int
    tasks_with_lower_trial_total: int
    control_passes: int
    treatment_passes: int
    equal_task_macro_reduction: float | None
    findings: tuple[str, ...]
    exploratory: bool = True


def assess_context_claim(
    measurements: tuple[ContextPairMeasurement, ...],
) -> ContextClaimAssessment:
    by_task: dict[str, list[ContextPairMeasurement]] = defaultdict(list)
    for measurement in measurements:
        by_task[measurement.task_id].append(measurement)

    control_passes = sum(measurement.control_passed for measurement in measurements if measurement.valid)
    treatment_passes = sum(measurement.treatment_passed for measurement in measurements if measurement.valid)
    covered_reductions: list[float] = []
    tasks_with_lower = 0
    regression_tasks: list[str] = []
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
            control_mean = mean(measurement.control_trial_total_input_tokens for measurement in usable)
            treatment_mean = mean(measurement.treatment_trial_total_input_tokens for measurement in usable)
            if control_mean > 0:
                covered_reductions.append(1 - treatment_mean / control_mean)
                if treatment_mean < control_mean:
                    tasks_with_lower += 1
        control_task_passes = sum(measurement.control_passed for measurement in task_measurements if measurement.valid)
        treatment_task_passes = sum(
            measurement.treatment_passed for measurement in task_measurements if measurement.valid
        )
        if control_task_passes - treatment_task_passes >= 2:
            regression_tasks.append(task_id)

    macro = mean(covered_reductions) if covered_reductions else None
    findings: list[str] = []
    if len(covered_reductions) < 6:
        findings.append("fewer_than_six_tasks_have_two_success_matched_pairs")
    if macro is None or macro < 0.15:
        findings.append("trial_total_input_reduction_below_15_percent")
    if treatment_passes < control_passes:
        findings.append("treatment_pass_count_below_control")
    if tasks_with_lower < 6:
        findings.append("fewer_than_six_tasks_use_less_trial_total_input")
    if regression_tasks:
        findings.append("task_lost_at_least_two_of_three_passes:" + ",".join(regression_tasks))

    return ContextClaimAssessment(
        claim_eligible=not findings,
        covered_tasks=len(covered_reductions),
        tasks_with_lower_trial_total=tasks_with_lower,
        control_passes=control_passes,
        treatment_passes=treatment_passes,
        equal_task_macro_reduction=macro,
        findings=tuple(findings),
    )


__all__ = [
    "ContextClaimAssessment",
    "ContextPairMeasurement",
    "assess_context_claim",
]
