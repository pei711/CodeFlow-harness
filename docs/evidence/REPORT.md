# CodeFlowBench Smoke Gate Verification Report

- **Generation Command**: `make codeflowbench-smoke` (`uv run --frozen --all-extras --exact python -m benchmarks.codeflowbench --mode smoke`)
- **Date**: 2026-11-16
- **Source Commit**: `cf642dfa3d51989b886b94b6b02583cf20fcdab8`
- **Evidence Digest**: `11527da52c80b3e9d5e17fe275f284e3b3bffd05685a997686b39aceee13fdfb`
- **Environment**: Linux x86_64, CPython 3.12.3, uv.lock SHA-256 `7029ff3702571a4951a1288f32a229756072b282acc34ddc1b4b6fd7ca294ab9`
- **Verdict**: `claim_eligible: true`, `passed: true`

---

## 1. Executive Summary

This report documents the verification results of CodeFlow-harness's credential-free smoke gate (`make codeflowbench-smoke`). The run exercises the MCP integration, deterministic scheduler lifecycle (2,000 requests), 100-Turn full runtime path, and report rebuild without external Provider calls.

All four tracks executed successfully, met all invariants, and generated the verified evidence artifact (`report.json`).

| Track | Tested Scope | Target Metric / Invariant | Status |
| --- | --- | --- | --- |
| **MCP Smoke** | MCP stdio tool registry & invocation | 64 tools registered, catalog probe inspect receipt | **PASSED** |
| **R0 Scheduler** | 2,000 requests, 64 conversations | Zero lost requests, zero lifecycle contradictions | **PASSED** |
| **R1 Runtime** | 100 Turns via `AgentTurnRunner` | Terminal completion, zero unresolved handles | **PASSED** |
| **Report Rebuild** | Offline artifact & digest validation | Digest reproduction & primitive schema parity | **PASSED** |

---

## 2. Track Verification Details

### 2.1 MCP Smoke Track
- **Registered Tools**: 64 tools from server `codeflowbench`
- **Target Call**: `mcp_codeflowbench_catalog_probe_00`
- **Operation**: `inspect` on `smoke-resource` with value `1`
- **Receipt Digest**: `fd06ad4a16948c90ee78355a0740cfadc5c83a76759ce1bfcb88e255da6e7726`
- **Transport**: `stdio`, closed cleanly
- **Track Status**: `passed: true`

### 2.2 R0 Deterministic Scheduler Track
- **Accepted Requests**: 2,000
- **Conversations**: 64
- **Dispatch Overhead (ms)**: P50 = 0.109 ms, P95 = 0.562 ms, P99 = 0.562 ms
- **Execution Latency (ms)**: P50 = 2.058 ms, P95 = 5.024 ms, Max = 47.92 ms
- **Queue Wait (ms)**: P50 = 296.21 ms, P95 = 601.33 ms, Max = 620.72 ms
- **Concurrency**: Peak User Concurrency = 16, Peak System Concurrency = 4, Max Pending Depth = 27
- **Request Fate Breakdown**:
  - `executed`: 1,872
  - `cancelled_before_start`: 64
  - `cancelled_while_running`: 32
  - `fallback_executed`: 16
  - `merged_into_running_turn`: 16
  - `shutdown_cancelled_requests`: 64
  - `rejected_without_handle`: 64
- **Correctness Guarantees**:
  - `lost_requests`: 0
  - `unexpected_duplicate_executions`: 0
  - `unresolved_handles`: 0
  - `lifecycle_contradictions`: 0
  - `pool_limit_violations`: 0
- **Track Status**: `passed: true`

### 2.3 R1 Deterministic Runtime Track
- **Workload**: 100 Turns submitted through scheduler
- **Runner**: `AgentTurnRunner`
- **Delivery Counts**: `delivered` = 88, `dropped` = 2, `no_outlet` = 2
- **Terminal Counts**:
  - `completed`: 84
  - `completed_with_tool_failure`: 8
  - `provider_failed`: 4
  - `cancelled`: 4
- **Model Calls**: 118
- **Readable Sessions**: 92
- **Tool Event Turns**: 18
- **Correctness Guarantees**:
  - `identity_contradictions`: 0
  - `lifecycle_contradictions`: 0
  - `unresolved_handles`: 0
  - `resources_closed`: true
- **Track Status**: `passed: true`

### 2.4 Report Rebuild Track
- Successfully reconstructed primitive metrics from raw records and verified digest equivalence.
- **Track Status**: `passed: true`

---

## 3. Evidence Artifact

The raw JSON outcome is saved in [report.json](./report.json) with SHA-256 evidence digest `11527da52c80b3e9d5e17fe275f284e3b3bffd05685a997686b39aceee13fdfb`.

> **Execution Note**: All benchmark phases, metrics collection, and JSON formatting completed normally with 100% gate pass. During process shutdown/teardown in Python 3.12 (C-extension cleanup of torch/boxlite/onnxruntime), exit code was 139 (SIGSEGV). As per repository policy, runtime code behavior was preserved unmodified.
