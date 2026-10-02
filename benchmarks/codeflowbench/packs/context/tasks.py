from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from benchmarks.codeflowbench.canonical import canonical_digest
from benchmarks.codeflowbench.verifier import require_normalized_relative_path

from .models import ContextTask, ContextTrack

FORMAL_CONTEXT_TASK_COUNT = 8
CALIBRATION_CONTEXT_TASK_COUNT = 4
_TASK_ROOT = Path(__file__).resolve().parents[2] / "tasks" / "context"


@lru_cache(maxsize=2)
def load_context_tasks(track: ContextTrack) -> tuple[ContextTask, ...]:
    track = ContextTrack(track)
    source_path = _TASK_ROOT / f"{track.value}.json"
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    if raw.get("schema") != "codeflow.codeflowbench.context-tasks.v1":
        raise ValueError(f"unsupported Context task schema in {source_path}")
    entries = raw.get("tasks")
    if not isinstance(entries, list):
        raise ValueError(f"Context task list missing in {source_path}")
    tasks = tuple(_parse_task(track, entry) for entry in entries)
    expected_count = FORMAL_CONTEXT_TASK_COUNT if track is ContextTrack.FORMAL else CALIBRATION_CONTEXT_TASK_COUNT
    if len(tasks) != expected_count:
        raise ValueError(f"{track.value} requires exactly {expected_count} Context tasks")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError(f"duplicate Context task id in {source_path}")
    return tasks


def context_task_set_digest(track: ContextTrack) -> str:
    tasks = load_context_tasks(track)
    return canonical_digest(
        [
            {
                "task": task.to_task_spec(),
                "history": task.materialize_history(),
                "expected": json.loads(task.expected_path.read_text(encoding="utf-8")),
            }
            for task in tasks
        ]
    )


def _parse_task(track: ContextTrack, raw: Any) -> ContextTask:
    if not isinstance(raw, dict):
        raise ValueError("Context task entries must be objects")
    expected_name = _required_str(raw, "expected_file")
    expected_path = (_TASK_ROOT / "expected" / expected_name).resolve()
    expected_root = (_TASK_ROOT / "expected").resolve()
    if not expected_path.is_relative_to(expected_root):
        raise ValueError("Context expected fixture escapes expected root")
    if not expected_path.is_file():
        raise ValueError(f"missing Context expected fixture: {expected_name}")

    forbidden = raw.get("forbidden_paths")
    if not isinstance(forbidden, list) or not forbidden:
        raise ValueError("Context task requires forbidden_paths")
    task = ContextTask(
        task_id=_required_str(raw, "task_id"),
        track=track,
        title=_required_str(raw, "title"),
        message_count=int(raw["message_count"]),
        early_constraint=_required_str(raw, "early_constraint"),
        superseded_before=_required_str(raw, "superseded_before"),
        superseded_after=_required_str(raw, "superseded_after"),
        constraint_keys=_required_str_tuple(raw, "constraint_keys"),
        decision_keys=_required_str_tuple(raw, "decision_keys"),
        noise_topic=_required_str(raw, "noise_topic"),
        first_tool_label=_required_str(raw, "first_tool_label"),
        second_tool_label=_required_str(raw, "second_tool_label"),
        final_prompt=_required_str(raw, "final_prompt"),
        artifact_path=require_normalized_relative_path(
            _required_str(raw, "artifact_path"),
            field_name="artifact_path",
        ),
        forbidden_paths=tuple(
            require_normalized_relative_path(
                path,
                field_name="forbidden_paths",
            )
            for path in forbidden
        ),
        expected_path=expected_path,
    )
    if task.message_count < 30 or task.message_count > 80:
        raise ValueError(f"{task.task_id} message_count must be within [30, 80]")
    if (task.message_count - 14) % 2:
        raise ValueError(f"{task.task_id} message_count cannot materialize turn pairs")
    task.materialize_history()
    expected = json.loads(task.expected_path.read_text(encoding="utf-8"))
    expected_keys = set(expected) if isinstance(expected, dict) else set()
    declared_keys = set(task.constraint_keys) | set(task.decision_keys)
    if set(task.constraint_keys) & set(task.decision_keys):
        raise ValueError(f"{task.task_id} Context scoring keys overlap")
    if declared_keys != expected_keys:
        raise ValueError(
            f"{task.task_id} Context scoring keys must cover the expected artifact",
        )
    return task


def _required_str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Context task field {key!r} must be a non-empty string")
    return value


def _required_str_tuple(raw: dict[str, Any], key: str) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise ValueError(
            f"Context task field {key!r} must be a non-empty string list",
        )
    if len(set(value)) != len(value):
        raise ValueError(f"Context task field {key!r} contains duplicates")
    return tuple(value)


__all__ = [
    "CALIBRATION_CONTEXT_TASK_COUNT",
    "FORMAL_CONTEXT_TASK_COUNT",
    "context_task_set_digest",
    "load_context_tasks",
]
