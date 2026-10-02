from benchmarks.codeflowbench.packs.runtime.tool_execution_experiments import (
    ToolExecutionExperimentConfig,
    run_tool_execution_experiment,
)


async def test_capability_parallel_arm_preserves_results_and_reduces_latency() -> None:
    result = await run_tool_execution_experiment(
        ToolExecutionExperimentConfig(
            repetitions=3,
            tool_calls=4,
            delay_ms=5,
        )
    )

    summary = result["summary"]
    assert summary["correctness_passed"] is True
    assert summary["serial_peak_concurrency"] == 1
    assert summary["capability_parallel_peak_concurrency"] == 4
    assert summary["median_latency_reduction_percent"] > 50
    assert len(result["repetitions"]) == 3
    assert result["evidence_scope"] == "synthetic_async_scheduler_microbenchmark"
    assert result["positive_claim_eligible"] is False
