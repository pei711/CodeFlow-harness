from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from statistics import mean

from benchmarks.codeflowbench.statistics import clustered_bootstrap_interval

from .models import (
    TokenWiseArmSummary,
    TokenWiseCostClaim,
    TokenWiseCostMeasurement,
)

CACHE_POLICY_PREFIX_DISRUPTED = "prefix_disrupted"
CACHE_POLICY_PREFIX_STABLE = "prefix_stable"
CACHE_POLICIES = (
    CACHE_POLICY_PREFIX_DISRUPTED,
    CACHE_POLICY_PREFIX_STABLE,
)


def assess_tokenwise_cost_claim(
    measurements: tuple[TokenWiseCostMeasurement, ...],
    *,
    expected_workloads: Mapping[str, str],
    repetitions: int,
) -> TokenWiseCostClaim:
    if not expected_workloads:
        raise ValueError("expected_workloads must not be empty")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    expected_block_keys = {(task_id, repetition) for task_id in expected_workloads for repetition in range(repetitions)}

    grouped: dict[tuple[str, int], list[TokenWiseCostMeasurement]] = defaultdict(list)
    for measurement in measurements:
        grouped[(measurement.task_id, measurement.repetition)].append(measurement)

    valid_blocks: list[tuple[TokenWiseCostMeasurement, ...]] = []
    for block_key in expected_block_keys:
        block = grouped.get(block_key, [])
        by_policy = {measurement.cache_policy: measurement for measurement in block}
        if set(by_policy) != set(CACHE_POLICIES) or len(block) != len(CACHE_POLICIES):
            continue
        ordered = tuple(by_policy[policy] for policy in CACHE_POLICIES)
        models = {measurement.actual_model for measurement in ordered}
        workload_classes = {measurement.workload_class for measurement in ordered}
        if (
            not all(measurement.usage_complete and measurement.cost_complete for measurement in ordered)
            or any(measurement.fallback_used for measurement in ordered)
            or None in models
            or len(models) != 1
            or len(workload_classes) != 1
            or workload_classes != {expected_workloads[block_key[0]]}
            or not all(measurement.requested_model and measurement.actual_model for measurement in ordered)
            or any(measurement.actual_model != measurement.requested_model for measurement in ordered)
        ):
            continue
        valid_blocks.append(ordered)

    retained = tuple(measurement for block in valid_blocks for measurement in block)
    arms = {
        policy: _summarize_arm(
            tuple(measurement for measurement in retained if measurement.cache_policy == policy),
            policy,
        )
        for policy in CACHE_POLICIES
    }
    disrupted = arms[CACHE_POLICY_PREFIX_DISRUPTED]
    stable = arms[CACHE_POLICY_PREFIX_STABLE]
    cost_reduction_vs_disrupted = _cost_reduction(disrupted, stable)
    cache_hit_rate_lift = stable.conservative_cache_hit_rate - disrupted.conservative_cache_hit_rate

    findings: list[str] = []
    if set(grouped) != expected_block_keys or len(valid_blocks) != len(expected_block_keys):
        findings.append("incomplete_comparison_blocks")
    if not _covers_all_workloads(retained):
        findings.append("workload_coverage_incomplete")
    if not _success_non_regression(retained):
        findings.append("task_success_regression")
    if cost_reduction_vs_disrupted is None or cost_reduction_vs_disrupted <= 0:
        findings.append("no_cost_reduction_vs_disrupted_prefix")
    if cache_hit_rate_lift <= 0:
        findings.append("no_cache_hit_rate_lift")
    if stable.conservative_cache_hit_rate <= 0:
        findings.append("no_stable_prefix_cache_reads")

    claim_eligible = not findings
    cv_metrics: dict[str, int | float] = {}
    if claim_eligible:
        cv_metrics = {
            "cost_per_verified_success_usd": stable.cost_per_verified_success_usd or 0.0,
            "cost_reduction_vs_disrupted": cost_reduction_vs_disrupted or 0.0,
            "conservative_cache_hit_rate": stable.conservative_cache_hit_rate,
            "cache_hit_rate_lift": cache_hit_rate_lift,
            "task_pass_rate": stable.task_pass_rate,
            "valid_comparison_blocks": len(valid_blocks),
            "trial_count": len(retained),
        }
    return TokenWiseCostClaim(
        claim_eligible=claim_eligible,
        expected_blocks=len(expected_block_keys),
        valid_blocks=len(valid_blocks),
        arms=arms,
        cost_reduction_vs_disrupted=cost_reduction_vs_disrupted,
        cache_hit_rate_lift=cache_hit_rate_lift,
        findings=tuple(findings),
        cv_metrics=cv_metrics,
    )


def paired_cost_reduction_interval(
    measurements: tuple[TokenWiseCostMeasurement, ...],
    *,
    samples: int,
    seed: int,
) -> dict[str, int | float | bool | str] | None:
    grouped: dict[tuple[str, int], dict[str, TokenWiseCostMeasurement]] = defaultdict(dict)
    for measurement in measurements:
        grouped[(measurement.task_id, measurement.repetition)][measurement.cache_policy] = measurement
    per_task: dict[str, list[float]] = defaultdict(list)
    for (task_id, _repetition), policies in grouped.items():
        if set(policies) != set(CACHE_POLICIES):
            return None
        control = policies[CACHE_POLICY_PREFIX_DISRUPTED]
        treatment = policies[CACHE_POLICY_PREFIX_STABLE]
        if control.cost_usd <= 0:
            return None
        per_task[task_id].append((control.cost_usd - treatment.cost_usd) / control.cost_usd)
    if not per_task:
        return None
    interval = clustered_bootstrap_interval(
        {task_id: tuple(values) for task_id, values in per_task.items()},
        samples=samples,
        seed=seed,
    )
    paired_values = [value for values in per_task.values() for value in values]
    return {
        "estimate": mean(paired_values),
        "lower": interval.lower,
        "upper": interval.upper,
        "tasks": interval.tasks,
        "pairs": len(paired_values),
        "samples": interval.samples,
        "seed": interval.seed,
        "unit": "task_clustered_pair",
        "exploratory": interval.exploratory,
    }


def _summarize_arm(
    measurements: tuple[TokenWiseCostMeasurement, ...],
    cache_policy: str,
) -> TokenWiseArmSummary:
    trials = len(measurements)
    successes = sum(measurement.task_passed for measurement in measurements)
    total_cost = sum(measurement.cost_usd for measurement in measurements)
    input_denominator = sum(
        measurement.fresh_input_tokens + measurement.cache_write_tokens + measurement.cache_read_tokens
        for measurement in measurements
    )
    cache_reads = sum(measurement.cache_read_tokens for measurement in measurements)
    return TokenWiseArmSummary(
        cache_policy=cache_policy,
        trials=trials,
        verified_successes=successes,
        task_pass_rate=successes / trials if trials else 0.0,
        total_cost_usd=total_cost,
        cost_per_verified_success_usd=total_cost / successes if successes else None,
        conservative_cache_hit_rate=cache_reads / input_denominator if input_denominator else 0.0,
    )


def _cost_reduction(
    control: TokenWiseArmSummary,
    treatment: TokenWiseArmSummary,
) -> float | None:
    control_cost = control.cost_per_verified_success_usd
    treatment_cost = treatment.cost_per_verified_success_usd
    if control_cost is None or treatment_cost is None or control_cost == 0:
        return None
    return 1 - treatment_cost / control_cost


def _covers_all_workloads(
    measurements: tuple[TokenWiseCostMeasurement, ...],
) -> bool:
    return {measurement.workload_class for measurement in measurements} == {
        "stable_dialogue",
        "long_history",
        "tool_accumulation",
        "intra_turn_tool_chain",
    }


def _success_non_regression(
    measurements: tuple[TokenWiseCostMeasurement, ...],
) -> bool:
    by_workload: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for measurement in measurements:
        by_workload[measurement.workload_class][measurement.cache_policy].append(measurement.task_passed)
    for policies in by_workload.values():
        stable = policies[CACHE_POLICY_PREFIX_STABLE]
        disrupted = policies[CACHE_POLICY_PREFIX_DISRUPTED]
        if not stable or not disrupted:
            return False
        if sum(stable) / len(stable) < sum(disrupted) / len(disrupted):
            return False
    return True


__all__ = [
    "CACHE_POLICIES",
    "CACHE_POLICY_PREFIX_DISRUPTED",
    "CACHE_POLICY_PREFIX_STABLE",
    "assess_tokenwise_cost_claim",
    "paired_cost_reduction_interval",
]
