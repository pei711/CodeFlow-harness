from __future__ import annotations

from dataclasses import replace
from math import inf, nan
from pathlib import Path

import pytest

from benchmarks.codeflowbench.packs.tokenwise_cost import (
    CACHE_POLICY_PREFIX_DISRUPTED,
    CACHE_POLICY_PREFIX_STABLE,
    TokenWiseCostMeasurement,
    TokenWiseCostPack,
    assess_tokenwise_cost_claim,
)
from benchmarks.codeflowbench.plan import compile_plan
from benchmarks.codeflowbench.schema import ExperimentSpec


def _experiment(tmp_path: Path) -> ExperimentSpec:
    return ExperimentSpec(
        suite="tokenwise-cost-formal",
        repetitions=3,
        pack_ids=("tokenwise-cost",),
        output_root=tmp_path,
        identity={
            "codeflow_commit": "0" * 40,
            "model": "deepseek/deepseek-v4-flash",
            "provider": "deepseek",
        },
    )


def _measurements() -> tuple[TokenWiseCostMeasurement, ...]:
    measurements: list[TokenWiseCostMeasurement] = []
    tasks = TokenWiseCostPack().definition().tasks
    costs = {
        CACHE_POLICY_PREFIX_DISRUPTED: 1.0,
        CACHE_POLICY_PREFIX_STABLE: 0.5,
    }
    cache_reads = {
        CACHE_POLICY_PREFIX_DISRUPTED: 0,
        CACHE_POLICY_PREFIX_STABLE: 600,
    }
    for task in tasks:
        workload_class = str(task.payload["workload_class"])
        for repetition in range(3):
            for cache_policy, cost_usd in costs.items():
                measurements.append(
                    TokenWiseCostMeasurement(
                        task_id=task.task_id,
                        workload_class=workload_class,
                        repetition=repetition,
                        cache_policy=cache_policy,
                        task_passed=True,
                        usage_complete=True,
                        cost_complete=True,
                        requested_model="deepseek/deepseek-v4-flash",
                        actual_model="deepseek/deepseek-v4-flash",
                        fallback_used=False,
                        fresh_input_tokens=1_000 - cache_reads[cache_policy],
                        cache_write_tokens=0,
                        cache_read_tokens=cache_reads[cache_policy],
                        output_tokens=20,
                        cost_usd=cost_usd,
                    )
                )
    return tuple(measurements)


def _assess(
    measurements: tuple[TokenWiseCostMeasurement, ...],
):
    expected_workloads = {
        task.task_id: str(task.payload["workload_class"]) for task in TokenWiseCostPack().definition().tasks
    }
    return assess_tokenwise_cost_claim(
        measurements,
        expected_workloads=expected_workloads,
        repetitions=3,
    )


def test_tokenwise_cost_pack_freezes_four_workloads_and_two_arms(tmp_path: Path) -> None:
    definition = TokenWiseCostPack().definition()

    assert definition.pack_id == "tokenwise-cost"
    assert len(definition.tasks) == 12
    assert {task.payload["workload_class"] for task in definition.tasks} == {
        "stable_dialogue",
        "long_history",
        "tool_accumulation",
        "intra_turn_tool_chain",
    }
    assert [variant.settings["cache_policy"] for variant in definition.variants] == [
        CACHE_POLICY_PREFIX_DISRUPTED,
        CACHE_POLICY_PREFIX_STABLE,
    ]
    assert len(definition.pairs) == 1

    plan = compile_plan(_experiment(tmp_path), (TokenWiseCostPack(),))
    assert len(plan.comparison_blocks) == 12 * 3 == 36
    assert len(plan.trials) == 12 * 2 * 3 == 72
    assert len(plan.pairs) == 12 * 3 == 36


def test_tokenwise_cost_claim_uses_conservative_hit_rate_and_success_cost() -> None:
    result = _assess(_measurements())

    stable = result.arms[CACHE_POLICY_PREFIX_STABLE]
    assert result.claim_eligible is True
    assert result.valid_blocks == 36
    assert stable.task_pass_rate == 1.0
    assert stable.cost_per_verified_success_usd == 0.5
    assert stable.conservative_cache_hit_rate == 0.6
    assert result.cost_reduction_vs_disrupted == 0.5
    assert result.cache_hit_rate_lift == 0.6
    assert result.cv_metrics == {
        "cost_per_verified_success_usd": 0.5,
        "cost_reduction_vs_disrupted": 0.5,
        "conservative_cache_hit_rate": 0.6,
        "cache_hit_rate_lift": 0.6,
        "task_pass_rate": 1.0,
        "valid_comparison_blocks": 36,
        "trial_count": 72,
    }


def test_tokenwise_cost_claim_rejects_incomplete_usage() -> None:
    measurements = list(_measurements())
    measurements[0] = replace(measurements[0], usage_complete=False)

    result = _assess(tuple(measurements))

    assert result.claim_eligible is False
    assert result.valid_blocks == 35
    assert "incomplete_comparison_blocks" in result.findings
    assert result.cv_metrics == {}


def test_tokenwise_cost_claim_rejects_task_success_regression() -> None:
    measurements = list(_measurements())
    index = next(
        index
        for index, measurement in enumerate(measurements)
        if measurement.cache_policy == CACHE_POLICY_PREFIX_STABLE
    )
    measurements[index] = replace(measurements[index], task_passed=False)

    result = _assess(tuple(measurements))

    assert result.claim_eligible is False
    assert "task_success_regression" in result.findings
    assert result.cv_metrics == {}


def test_tokenwise_cost_claim_requires_observed_cache_reads() -> None:
    measurements = tuple(
        replace(measurement, cache_read_tokens=0)
        if measurement.cache_policy == CACHE_POLICY_PREFIX_STABLE
        else measurement
        for measurement in _measurements()
    )

    result = _assess(measurements)

    assert result.claim_eligible is False
    assert "no_cache_hit_rate_lift" in result.findings
    assert result.cv_metrics == {}


def test_tokenwise_cost_claim_rejects_wrong_block_identity() -> None:
    measurements = list(_measurements())
    measurements[0] = replace(measurements[0], task_id="tw-unplanned-case")

    result = _assess(tuple(measurements))

    assert result.claim_eligible is False
    assert result.valid_blocks == 35
    assert "incomplete_comparison_blocks" in result.findings
    assert result.cv_metrics == {}


@pytest.mark.parametrize("cost_usd", [nan, inf])
def test_tokenwise_cost_measurement_rejects_non_finite_cost(cost_usd: float) -> None:
    measurement = _measurements()[0]

    with pytest.raises(ValueError, match="finite"):
        replace(measurement, cost_usd=cost_usd)
