from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.codeflowbench.packs.tracing.overhead_experiment import (
    TracingExperimentConfig,
    TracingExperimentError,
    run_campaign,
    verify_campaign,
)


@pytest.mark.asyncio
async def test_tracing_campaign_runs_real_runtime_and_rebuilds_offline(tmp_path: Path) -> None:
    config = TracingExperimentConfig(blocks=2, turns_per_block=2, bootstrap_samples=100)

    verifier = await run_campaign(
        output_root=tmp_path,
        codeflow_commit="a" * 40,
        config=config,
    )
    rebuilt = verify_campaign(output_root=tmp_path, expected_codeflow_commit="a" * 40)
    aggregate = json.loads((tmp_path / "aggregate.json").read_text(encoding="utf-8"))

    assert verifier["passed"] is True
    assert rebuilt == verifier
    assert aggregate["measurement_valid"] is True
    assert aggregate["summary"]["pairs"] == 4
    assert aggregate["summary"]["root_spans"] == 4
    assert aggregate["summary"]["bytes_per_traced_turn"] > 0
    assert aggregate["gates"]["disabled_arm_emits_no_trace"] is True
    assert len((tmp_path / "raw-outcomes.jsonl").read_text(encoding="utf-8").splitlines()) == 4
    assert (tmp_path / "claim-eligibility.json").is_file()
    assert (tmp_path / "inventory.json").is_file()

    trace_log = next((tmp_path / "traces").glob("block-*/tracing_on/logs/audit-spans.log"))
    trace_log.write_bytes(trace_log.read_bytes() + b"\n")
    with pytest.raises(TracingExperimentError, match="trace_receipts"):
        verify_campaign(output_root=tmp_path, expected_codeflow_commit="a" * 40)
