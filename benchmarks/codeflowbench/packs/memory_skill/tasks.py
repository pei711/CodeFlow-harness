from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from .models import CrossSessionTask

_SCHEMA = "codeflow.codeflowbench.memory-skill-tasks.v1"
_EXPECTED_COUNTS = {
    "formal": 8,
    "calibration": 4,
}


@lru_cache(maxsize=2)
def load_cross_session_tasks(
    definition_kind: Literal["formal", "calibration"],
) -> tuple[CrossSessionTask, ...]:
    path = Path(__file__).resolve().parents[2] / "tasks" / "memory_skill" / f"{definition_kind}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != _SCHEMA:
        raise ValueError(f"unsupported memory-skill task schema: {path}")
    rows = payload.get("tasks")
    if not isinstance(rows, list):
        raise ValueError(f"memory-skill tasks must be a list: {path}")
    tasks = tuple(CrossSessionTask(**dict(row)) for row in rows)
    if len(tasks) != _EXPECTED_COUNTS[definition_kind]:
        raise ValueError(f"unexpected memory-skill task count: {path}")
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"duplicate memory-skill task id: {path}")
    return tasks


__all__ = ["load_cross_session_tasks"]
