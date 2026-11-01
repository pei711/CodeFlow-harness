from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compare import compare_runs
from .runner import load_task, run_task
from .trajectory import score_trace


def main() -> int:
    parser = argparse.ArgumentParser(description="CodeFlowBench repository task regression and trajectory evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run one frozen repository task")
    run.add_argument("task", type=Path, help="JSON codeflow.coding-task.v1 file")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--repetition", type=int, default=1)
    compare = sub.add_parser("compare", help="compare two version artifact roots")
    compare.add_argument("reference", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--output", type=Path)
    trace = sub.add_parser("score-trace", help="score an audit-spans.log file")
    trace.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        result = run_task(load_task(args.task), args.output, repetition=args.repetition)
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
        return 0 if result.get("status") == "passed" else 1
    if args.command == "compare":
        result = compare_runs(args.reference, args.candidate, args.output)
    else:
        result = score_trace(args.path)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
