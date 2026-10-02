from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any

from codeflow.agent.tools import (
    Tool,
    ToolCapability,
    ToolEffect,
    ToolInvocation,
    ToolRegistry,
)

TOOL_EXECUTION_EXPERIMENT_SCHEMA = "codeflow.codeflowbench.tool-execution-experiment.v1"


@dataclass(frozen=True)
class ToolExecutionExperimentConfig:
    repetitions: int = 7
    tool_calls: int = 8
    delay_ms: float = 20.0
    max_parallel: int = ToolRegistry.DEFAULT_MAX_PARALLEL

    def __post_init__(self) -> None:
        if self.repetitions < 1:
            raise ValueError("repetitions must be positive")
        if self.tool_calls < 2:
            raise ValueError("tool_calls must be at least two")
        if self.delay_ms <= 0:
            raise ValueError("delay_ms must be positive")
        if self.max_parallel < 1:
            raise ValueError("max_parallel must be positive")


class _DelayReadTool(Tool):
    capability = ToolCapability(effect=ToolEffect.READ, concurrency_safe=True)

    def __init__(self, delay_ms: float) -> None:
        self._delay_s = delay_ms / 1_000
        self.active = 0
        self.peak = 0

    @property
    def name(self) -> str:
        return "delay_read"

    @property
    def description(self) -> str:
        return "return a label after a deterministic delay"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": ["label"],
        }

    async def execute(self, label: str, **kwargs: Any) -> str:
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self._delay_s)
            return label
        finally:
            self.active -= 1


async def run_tool_execution_experiment(config: ToolExecutionExperimentConfig) -> dict[str, Any]:
    repetitions: list[dict[str, Any]] = []
    for repetition in range(config.repetitions):
        arm_order = ("serial", "capability_parallel") if repetition % 2 == 0 else ("capability_parallel", "serial")
        arms: dict[str, dict[str, Any]] = {}
        for arm in arm_order:
            arms[arm] = await _run_arm(config, parallel_safe=arm == "capability_parallel")
        repetitions.append(
            {
                "repetition": repetition,
                "arm_order": list(arm_order),
                "serial": arms["serial"],
                "capability_parallel": arms["capability_parallel"],
            }
        )

    serial_latencies = [item["serial"]["elapsed_ms"] for item in repetitions]
    parallel_latencies = [item["capability_parallel"]["elapsed_ms"] for item in repetitions]
    serial_median = median(serial_latencies)
    parallel_median = median(parallel_latencies)
    serial_peak = max(item["serial"]["peak_concurrency"] for item in repetitions)
    parallel_peak = max(item["capability_parallel"]["peak_concurrency"] for item in repetitions)
    correctness_passed = all(item[arm]["correct"] for item in repetitions for arm in ("serial", "capability_parallel"))
    expected_parallel_peak = min(config.tool_calls, config.max_parallel)
    correctness_passed = correctness_passed and serial_peak == 1 and parallel_peak == expected_parallel_peak
    reduction = (serial_median - parallel_median) / serial_median * 100

    return {
        "schema": TOOL_EXECUTION_EXPERIMENT_SCHEMA,
        "evidence_scope": "synthetic_async_scheduler_microbenchmark",
        "positive_claim_eligible": False,
        "config": asdict(config),
        "repetitions": repetitions,
        "summary": {
            "correctness_passed": correctness_passed,
            "serial_median_ms": serial_median,
            "capability_parallel_median_ms": parallel_median,
            "median_latency_reduction_percent": reduction,
            "serial_peak_concurrency": serial_peak,
            "capability_parallel_peak_concurrency": parallel_peak,
        },
        "passed": correctness_passed and reduction > 0,
    }


async def _run_arm(
    config: ToolExecutionExperimentConfig,
    *,
    parallel_safe: bool,
) -> dict[str, Any]:
    tool = _DelayReadTool(config.delay_ms)
    registry = ToolRegistry(max_parallel=config.max_parallel)
    registry.register(tool)
    labels = [f"call-{index}" for index in range(config.tool_calls)]
    invocations = [ToolInvocation(tool.name, {"label": label}) for label in labels]

    started = time.perf_counter_ns()
    executions = await registry.execute_many(invocations, parallel_safe=parallel_safe)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    outputs = [str(execution.result) for execution in executions]

    return {
        "elapsed_ms": elapsed_ms,
        "peak_concurrency": tool.peak,
        "outputs": outputs,
        "correct": outputs == labels and all(not execution.result.failed for execution in executions),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Tool execution serial-vs-capability A/B experiment")
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--tool-calls", type=int, default=8)
    parser.add_argument("--delay-ms", type=float, default=20.0)
    parser.add_argument("--max-parallel", type=int, default=ToolRegistry.DEFAULT_MAX_PARALLEL)
    return parser.parse_args()


async def _main() -> int:
    args = _parse_args()
    result = await run_tool_execution_experiment(
        ToolExecutionExperimentConfig(
            repetitions=args.repetitions,
            tool_calls=args.tool_calls,
            delay_ms=args.delay_ms,
            max_parallel=args.max_parallel,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
