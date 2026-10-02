from benchmarks.codeflowbench.packs.runtime.scheduler_experiments import (
    SchedulerExperimentConfig,
    run_bulkhead_experiment,
    run_head_of_line_experiment,
)


def _small_config() -> SchedulerExperimentConfig:
    return SchedulerExperimentConfig(
        repetitions=2,
        fate_repetitions=1,
        worker_slots=4,
        system_slots=2,
        hol_cycles=2,
        hol_regular_sessions=8,
        hol_hot_turns_per_cycle=3,
        foreground_delay_ms=1,
        hot_delay_ms=3,
        bulkhead_user_sessions=8,
        bulkhead_turns_per_session=2,
        background_tasks=6,
        background_delay_ms=10,
    )


async def test_session_lanes_reduce_global_fifo_head_of_line_wait() -> None:
    result = await run_head_of_line_experiment(_small_config())

    assert result["summary"]["correctness_passed"]
    assert result["summary"]["foreground_p95_reduction_percent"] > 0
    assert len(result["repetitions"]) == 2


async def test_origin_pools_reduce_background_saturation_slowdown() -> None:
    result = await run_bulkhead_experiment(_small_config())

    assert result["summary"]["correctness_passed"]
    assert result["summary"]["shared_pool_slowdown"] > result["summary"]["isolated_pools_slowdown"]
    assert len(result["repetitions"]) == 2
