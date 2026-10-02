from benchmarks.codeflowbench.packs.runtime import (
    R0_ACCEPTED_REQUESTS,
    R0_CONVERSATIONS,
    R0_REJECTION_PROBES,
    RequestFate,
    run_r0_scheduler_track,
)


async def test_r0_accounts_for_every_request_and_scheduler_invariant() -> None:
    result = await run_r0_scheduler_track()

    assert result.accepted_requests == R0_ACCEPTED_REQUESTS == 2_000
    assert result.rejected_requests == R0_REJECTION_PROBES == 64
    assert result.conversation_count == R0_CONVERSATIONS == 64
    assert sum(result.fate_counts.values()) == result.accepted_requests
    assert all(result.fate_counts[fate.value] > 0 for fate in RequestFate)
    assert result.rejected_without_handle == result.rejected_requests
    assert result.lost_requests == 0
    assert result.unresolved_handles == 0
    assert result.unexpected_duplicate_executions == 0
    assert result.lifecycle_contradictions == 0
    assert result.pool_limit_violations == 0
    assert result.peak_user_concurrency == 16
    assert result.peak_system_concurrency == 4
    assert result.shutdown_cancelled_requests > 0
    assert result.max_observed_pending_depth > 0
    assert result.dispatch_overhead_ms.count == 16
    assert result.queue_wait_ms.count > 0
    assert result.execution_latency_ms.count > 0
    assert result.passed
    assert result.evidence_label == "deterministic_scheduler_runtime"
