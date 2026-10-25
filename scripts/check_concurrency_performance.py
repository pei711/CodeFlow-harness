"""Local scheduler regression comparison; never calls an LLM or creates a worktree.

Compares only the committed Scheduler implementation with the working version.
The benchmark, dependencies and the rest of the checkout are shared. This is
deliberately not a clean-checkout release performance claim.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import types
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

from benchmarks.codeflowbench.packs.runtime import r0, scheduler_experiments  # noqa: E402


def git(*args: str) -> str:
    executable = shutil.which("git")
    if executable is None:
        raise RuntimeError("Git is required")
    return subprocess.check_output([executable, *args], cwd=ROOT).decode("utf-8")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="HEAD")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / ".codeflow/evidence/concurrency-validation" / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    output.mkdir(parents=True, exist_ok=False)
    source = git("show", f"{args.baseline_ref}:codeflow/spine/scheduler.py")
    module = types.ModuleType("codeflow.spine._performance_baseline")
    sys.modules[module.__name__] = module
    exec(compile(source, "<committed-scheduler>", "exec"), module.__dict__)
    current = scheduler_experiments.Scheduler
    current_draining = r0.SchedulerDrainingError
    os.environ["CODEFLOW_TRACING"] = "0"
    logger.remove()
    config = scheduler_experiments.SchedulerExperimentConfig()
    identity = {
        "baseline_commit": git("rev-parse", args.baseline_ref).strip(),
        "dirty_checkout": bool(git("status", "--porcelain")),
        "scope": "scheduler_source_only_comparison_shared_current_benchmark_and_dependencies",
        "config": asdict(config),
    }
    summaries = {}
    try:
        for name, implementation in (("baseline", module.Scheduler), ("current", current)):
            scheduler_experiments.Scheduler = r0.Scheduler = implementation
            r0.SchedulerDrainingError = module.SchedulerDrainingError if name == "baseline" else current_draining
            result = await scheduler_experiments.run_scheduler_experiment_suite(config)
            (output / f"{name}.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            summaries[name] = {"passed": result["passed"], **result["resume_metrics"]}
            print(json.dumps({name: summaries[name]}, ensure_ascii=False), flush=True)
    finally:
        scheduler_experiments.Scheduler = r0.Scheduler = current
        r0.SchedulerDrainingError = current_draining
    (output / "comparison.json").write_text(
        json.dumps({**identity, "results": summaries}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Local evidence: {output}")
    if not all(result["passed"] for result in summaries.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
