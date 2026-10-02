from benchmarks.codeflowbench.packs.runtime import (
    R1_FULL_PATH_TURNS,
    run_r1_full_runtime_track,
)


async def test_r1_runs_exact_full_path_runtime_campaign(tmp_path) -> None:
    result = await run_r1_full_runtime_track(tmp_path)

    assert result.turns == R1_FULL_PATH_TURNS == 100
    assert result.submitted_through_scheduler == result.turns
    assert result.runner_type == "AgentTurnRunner"
    assert result.terminal_counts == {
        "completed": 84,
        "completed_with_tool_failure": 8,
        "provider_failed": 4,
        "cancelled": 4,
    }
    assert result.model_calls == 118
    assert result.tool_event_turns == 18
    assert result.readable_sessions == 92
    assert result.delivery_counts == {
        "delivered": 88,
        "dropped": 2,
        "no_outlet": 2,
    }
    assert result.identity_contradictions == 0
    assert result.lifecycle_contradictions == 0
    assert result.unresolved_handles == 0
    assert result.resources_closed
    assert result.passed
    assert result.evidence_label == "deterministic_benchmark_host_full_path"
