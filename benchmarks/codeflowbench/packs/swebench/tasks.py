from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from benchmarks.codeflowbench.canonical import canonical_digest
from .models import SwebenchTask, SwebenchTrack

FORMAL_SWEBENCH_TASK_COUNT = 50
SMOKE_SWEBENCH_TASK_COUNT = 2

_TASK_ROOT = Path(__file__).resolve().parents[2] / "tasks" / "swebench"


def load_swebench_manifest() -> dict[str, Any]:
    manifest_path = _TASK_ROOT / "verified_50_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing SWE-bench manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def load_swebench_tasks(track: SwebenchTrack = SwebenchTrack.FORMAL) -> tuple[SwebenchTask, ...]:
    track = SwebenchTrack(track)
    source_path = _TASK_ROOT / f"{track.value}.json"
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing SWE-bench tasks file: {source_path}")

    raw = json.loads(source_path.read_text(encoding="utf-8"))
    entries = raw.get("tasks")
    if not isinstance(entries, list):
        raise ValueError(f"tasks list missing in {source_path}")

    tasks = tuple(_parse_task(track, entry) for entry in entries)
    expected_count = (
        FORMAL_SWEBENCH_TASK_COUNT if track is SwebenchTrack.FORMAL else SMOKE_SWEBENCH_TASK_COUNT
    )
    if len(tasks) != expected_count:
        raise ValueError(f"{track.value} requires exactly {expected_count} tasks, got {len(tasks)}")

    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError(f"duplicate task id in {source_path}")

    return tasks


def swebench_task_set_digest(track: SwebenchTrack = SwebenchTrack.FORMAL) -> str:
    tasks = load_swebench_tasks(track)
    return canonical_digest(
        [
            {
                "task_id": task.task_id,
                "instance_id": task.instance_id,
                "repo": task.repo,
                "base_commit": task.base_commit,
                "version": task.version,
                "fail_to_pass": sorted(task.fail_to_pass),
                "pass_to_pass": sorted(task.pass_to_pass),
                "gold_patch_sha256": task.gold_patch_sha256,
            }
            for task in tasks
        ]
    )


def _parse_task(track: SwebenchTrack, raw: dict[str, Any]) -> SwebenchTask:
    instance_id = str(raw["instance_id"])
    task_id = raw.get("task_id") or f"swe-{instance_id.replace('__', '-').replace('_', '-')}"
    fail_to_pass = tuple(str(x) for x in raw.get("fail_to_pass", []))
    pass_to_pass = tuple(str(x) for x in raw.get("pass_to_pass", []))

    return SwebenchTask(
        task_id=task_id,
        instance_id=instance_id,
        repo=str(raw["repo"]),
        base_commit=str(raw["base_commit"]),
        version=str(raw.get("version", "")),
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        problem_statement=str(raw.get("problem_statement", "")),
        hints_text=str(raw.get("hints_text", "")),
        gold_patch_sha256=str(raw.get("gold_patch_sha256", "")),
        gold_patch_lines=int(raw.get("gold_patch_lines", 0)),
        track=track,
    )
