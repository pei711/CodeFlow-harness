import asyncio
import re
from pathlib import Path

import pytest

from benchmarks.codeflowbench.canonical import canonical_digest, canonical_json
from benchmarks.codeflowbench.packs.runtime.live_scheduler_experiment import (
    LiveSchedulerConfig,
    _analyze_live_pairs,
    _bootstrap_median_interval,
    _live_work,
    _run_live_arm,
    build_live_scheduler_plan,
    verify_live_scheduler_evidence,
)
from codeflow.config.schema import Config
from codeflow.providers.base import LLMProvider, LLMResponse


class _MarkerProvider(LLMProvider):
    def get_default_model(self) -> str:
        return "fake/live-agent"

    async def chat(self, messages, **kwargs) -> LLMResponse:
        markers = re.findall(r"CODEFLOW_LIVE_PERF_R\d{2}_(?:HOT|FG)_\d{2}", str(messages[-1]["content"]))
        marker = markers[-1]
        await asyncio.sleep(0.002 if "_HOT_" in marker else 0.001)
        return LLMResponse(
            content=marker,
            model="fake/live-agent",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )


def _small_config() -> LiveSchedulerConfig:
    return LiveSchedulerConfig(
        repetitions=2,
        user_slots=2,
        hot_turns=1,
        foreground_sessions=3,
        hard_cap_cny=2.0,
        bootstrap_resamples=100,
    )


def test_live_plan_freezes_workload_and_budget_without_credentials() -> None:
    config = LiveSchedulerConfig()
    plan = build_live_scheduler_plan(
        config,
        repository_root=Path(__file__).resolve().parents[1],
        provider_name="deepseek",
        model="deepseek/deepseek-v4-flash",
    )

    assert config.planned_turns == 4_160
    assert config.maximum_provider_request_attempts == 33_280
    assert config.maximum_cost_cny < config.hard_cap_cny == 320.0
    assert plan["budget"]["planned_turns"] == 4_160
    assert plan["analysis"] == {
        "confidence_interval": "paired_repetition_bootstrap_percentile",
        "confidence_level": 0.95,
        "resamples": 10_000,
        "seed": 20_260_806,
    }
    assert "api_key" not in canonical_json(plan).lower()


def test_live_plan_reserves_the_agent_exhaustion_call() -> None:
    with pytest.raises(ValueError, match="exhaustion synthesis"):
        LiveSchedulerConfig(max_agent_iterations=2, max_logical_calls_per_turn=2)


def test_live_plan_requires_balanced_arm_order() -> None:
    with pytest.raises(ValueError, match="balanced arm order"):
        LiveSchedulerConfig(repetitions=3)


async def test_live_arm_runs_real_agent_loop_with_marker_verification(tmp_path) -> None:
    config = _small_config()
    base = Config()
    base.agents.defaults.model = "fake/live-agent"
    work = _live_work(config, 0)

    result = await _run_live_arm(
        config,
        work=work,
        scheduler_kind="session_lanes",
        base_config=base,
        provider=_MarkerProvider(),
        workspace=tmp_path,
    )

    assert result["accepted_requests"] == 4
    assert result["measured_foreground_requests"] == 3
    assert result["task_failures"] == 0
    assert result["runner_exceptions"] == 0
    assert result["total_tokens"] == 60
    assert len(result["turn_records"]) == 4
    assert [record["submission_index"] for record in result["turn_records"]] == [0, 1, 2, 3]
    measured = [record for record in result["turn_records"] if record["measured"]]
    assert len(measured) == 3
    assert all(record["queue_wait_ms"] is not None for record in measured)
    assert all(record["execution_ms"] is not None for record in measured)
    assert all(record["end_to_end_ms"] is not None for record in measured)
    assert all(record["verifier_failures"] == [] for record in result["turn_records"])

    pairs = [
        {
            "repetition": repetition,
            "arm_order": ["global_fifo", "session_lanes"],
            "global_fifo": result,
            "session_lanes": result,
            "p95_reduction_percent": 0.0,
        }
        for repetition in range(2)
    ]
    analysis = {
        "confidence_interval": "paired_repetition_bootstrap_percentile",
        "confidence_level": config.confidence_level,
        "resamples": config.bootstrap_resamples,
        "seed": config.bootstrap_seed,
    }
    payload = {
        "schema": "codeflow.codeflowbench.runtime-live-scheduler.v2",
        "analysis": analysis,
        "repetitions": pairs,
        "summary": _analyze_live_pairs(
            pairs,
            bootstrap_resamples=config.bootstrap_resamples,
            bootstrap_seed=config.bootstrap_seed,
            confidence_level=config.confidence_level,
        ),
    }
    evidence = {**payload, "evidence_digest": canonical_digest(payload)}

    verification = verify_live_scheduler_evidence(evidence)

    assert verification["passed"] is True
    assert all(verification["gates"].values())


def test_bootstrap_interval_is_deterministic_and_positive() -> None:
    first = _bootstrap_median_interval(
        [4.0, 6.0, 8.0, 10.0],
        resamples=1_000,
        seed=42,
        confidence_level=0.95,
    )
    second = _bootstrap_median_interval(
        [4.0, 6.0, 8.0, 10.0],
        resamples=1_000,
        seed=42,
        confidence_level=0.95,
    )

    assert first == second
    assert first["lower_percent"] > 0
    assert first["upper_percent"] >= first["lower_percent"]
