from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from benchmarks.codeflowbench.canonical import to_primitive

from .models import CrossSessionTask
from .runtime import run_runtime_stage


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("learning", "evaluation"), required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(args.spec.read_text(encoding="utf-8"))
    task = CrossSessionTask(**payload["task"])
    return await run_runtime_stage(
        stage=args.stage,
        task=task,
        variant_settings=dict(payload["variant_settings"]),
        workspace=Path(payload["workspace"]),
        state_path=Path(payload["state_path"]),
        provider_spec=dict(payload["provider"]),
    )


def main() -> int:
    args = _parser().parse_args()
    try:
        result = asyncio.run(_run(args))
    except BaseException as exc:
        result = {
            "status": "infrastructure_failure",
            "error_code": "worker_exception",
            "error_type": type(exc).__name__,
        }
        exit_code = 1
    else:
        result = {"status": "completed", **result}
        exit_code = 0
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(
            to_primitive(result),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
