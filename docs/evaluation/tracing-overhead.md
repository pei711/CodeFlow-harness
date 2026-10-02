# Tracing Runtime overhead

> **Status: formal run completed on 2026-08-13.** Generated evidence remains
> outside Git; this page records its candidate binding and digests.

This CodeFlowBench track measures the local Runtime tax of CodeFlow's in-tree Tracing.
It does not test an external Provider or claim production latency.

## Treatment and workload

The campaign runs 20 balanced blocks of 50 paired Turns, for 1,000 pairs and
2,000 total Turns. Every Turn crosses the shared Runtime Assembly and Agent
Loop, makes two deterministic local Provider calls, executes `trace_lookup`
once, and returns `TRACE_OK`. The sole treatment axis is `CODEFLOW_TRACING=0` versus
`CODEFLOW_TRACING=1`.

Each enabled Turn must retain exactly one terminal `session.turn` trace joined
to two `llm.call` spans and one `tool.call` span. The disabled arm must emit no
trace bytes. Both arms must complete the same workload with identical replies
and call counts.

## Metrics and evidence

The aggregate reports P50 and P95 latency for both arms, the relative overhead,
a block-clustered bootstrap 95% interval for the P95 ratio, and bytes per traced
Turn. Measurement validity requires all correctness, correlation, Pair-count,
arm-balance, and disabled-no-output Gates to pass. There is no preregistered
"good overhead" threshold; the result is an estimated operational cost, not an
optimization claim.

The immutable manifest binds the CodeFlow commit, Python and platform identity,
workload, Pair count, and bootstrap settings. Per-block receipts retain raw
Turn latency and terminal outcomes plus SHA-256 receipts for every trace file.
The offline verifier rebuilds `raw-outcomes.jsonl`, `aggregate.json`,
`claim-eligibility.json`, `verifier-report.json`, and `inventory.json`.

## Result

All 1,000 pairs and 2,000 Turns were valid. The enabled arm retained exactly
1,000 traces and 6,000 spans with 100 percent correlation; the disabled arm
emitted zero trace bytes. Tracing wrote 25,717.2 bytes per enabled Turn.

| Metric | Tracing off | Tracing on |
| --- | ---: | ---: |
| P50 Turn latency | 2.061208 ms | 4.284334 ms |
| P95 Turn latency | 2.912292 ms | 5.157333 ms |

The observed P95 tax was 2.245041 ms, or 77.0885 percent relative to the small
local baseline. The block-clustered relative P95 interval was -9.9969 to
101.9378 percent, so the result does not support a stable relative-overhead
claim. It does support exact correlation and an absolute local tax estimate.

The aggregate SHA-256 is
`29b855459d6039f7630f410a623875a37a7e454f4a978aae265330fb023cec65`.
Raw manifests and inventories are intentionally not published in this repository.

## Operator commands

```bash
make codeflowbench-tracing-plan
make codeflowbench-tracing-run
make codeflowbench-tracing-verify
```

Set `CODEFLOW_TRACING_OUTPUT` to use a new evidence root. After the repository moves
past the measured candidate, set `CODEFLOW_TRACING_COMMIT` to the full commit bound
by the retained manifest before running the offline verifier.
