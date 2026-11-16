from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from benchmarks.codeflowbench.canonical import canonical_digest
from .models import TrajectoryCategory, TrajectoryTask, TrajectoryTrack

FORMAL_TRAJECTORY_TASK_COUNT = 6
SMOKE_TRAJECTORY_TASK_COUNT = 3

_TASK_ROOT = Path(__file__).resolve().parents[2] / "tasks" / "trajectory"


@lru_cache(maxsize=4)
def load_trajectory_tasks(
    track: TrajectoryTrack = TrajectoryTrack.FORMAL,
) -> tuple[TrajectoryTask, ...]:
    track = TrajectoryTrack(track)
    source_path = _TASK_ROOT / f"{track.value}.json"
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing trajectory tasks file: {source_path}")

    raw = json.loads(source_path.read_text(encoding="utf-8"))
    entries = raw.get("tasks")
    if not isinstance(entries, list):
        raise ValueError(f"tasks list missing in {source_path}")

    tasks = tuple(_parse_task(track, entry) for entry in entries)
    expected_count = (
        FORMAL_TRAJECTORY_TASK_COUNT
        if track is TrajectoryTrack.FORMAL
        else SMOKE_TRAJECTORY_TASK_COUNT
    )
    if len(tasks) != expected_count:
        raise ValueError(
            f"{track.value} requires exactly {expected_count} trajectory tasks, got {len(tasks)}"
        )

    if len({t.task_id for t in tasks}) != len(tasks):
        raise ValueError(f"duplicate task id in {source_path}")

    return tasks


def trajectory_task_set_digest(track: TrajectoryTrack = TrajectoryTrack.FORMAL) -> str:
    tasks = load_trajectory_tasks(track)
    return canonical_digest(
        [
            {
                "task_id": t.task_id,
                "category": t.category.value,
                "message_count": t.message_count,
                "early_constraint": t.early_constraint,
                "active_constraint": t.active_constraint,
                "latest_decision": t.latest_decision,
            }
            for t in tasks
        ]
    )


def _parse_task(track: TrajectoryTrack, raw: dict[str, Any]) -> TrajectoryTask:
    return TrajectoryTask(
        task_id=str(raw["task_id"]),
        category=TrajectoryCategory(str(raw["category"])),
        title=str(raw["title"]),
        message_count=int(raw["message_count"]),
        early_constraint=str(raw["early_constraint"]),
        active_constraint=str(raw["active_constraint"]),
        latest_decision=str(raw["latest_decision"]),
        noise_topic=str(raw.get("noise_topic", "")),
        turns=tuple(dict(t) for t in raw.get("turns", [])),
        track=track,
    )
