"""The task split: deterministic assertion cases over the subject module.

Each task is a small bundle of ``(function, args, expected)`` cases. A task
passes only when every one of its cases matches, so a trial is a single
pass/fail observation exactly like a benchmark task.

The train and test id sets are disjoint and every id is globally unique, so a
scorer can resolve a task from its id alone without being told the split.
"""

from __future__ import annotations

from dataclasses import dataclass

PUBLIC_SURFACE = ("normalize_number", "parse_duration", "slugify")

WHY_DEFINITIONS = {
    "slug_separator_collapse": (
        "slugify emits one separator per non-alphanumeric character, so runs of "
        "punctuation or whitespace become runs of separators and edge separators "
        "are left in place."
    ),
    "duration_hour_unit": (
        "parse_duration silently ignores unit suffixes it does not know, so hours contribute nothing."
    ),
    "number_group_separator": (
        "normalize_number does not remove digit-group separators, so a grouped number falls into the 0.0 fallback."
    ),
}


@dataclass(frozen=True)
class Case:
    fn: str
    args: tuple
    expect: object


@dataclass(frozen=True)
class Task:
    task_id: str
    why: str
    description: str
    cases: tuple[Case, ...]

    def to_payload(self) -> dict:
        return {
            "task_id": self.task_id,
            "cases": [{"fn": c.fn, "args": list(c.args), "expect": c.expect} for c in self.cases],
        }


TRAIN_TASKS: tuple[Task, ...] = (
    Task(
        task_id="slug-basic",
        why="slug_separator_collapse",
        description="slugify a plain title with single spaces",
        cases=(
            Case("slugify", ("CodeFlow Harness",), "codeflow-harness"),
            Case("slugify", ("release",), "release"),
        ),
    ),
    Task(
        task_id="slug-trim",
        why="slug_separator_collapse",
        description="slugify a title padded with spaces and trailing punctuation",
        cases=(Case("slugify", ("  Hello,  World!  ",), "hello-world"),),
    ),
    Task(
        task_id="slug-collapse",
        why="slug_separator_collapse",
        description="slugify a title containing a punctuation run",
        cases=(Case("slugify", ("Release v1.2 -- final",), "release-v1-2-final"),),
    ),
    Task(
        task_id="duration-minutes",
        why="duration_hour_unit",
        description="parse a single minute or second chunk",
        cases=(
            Case("parse_duration", ("30m",), 1800),
            Case("parse_duration", ("45s",), 45),
        ),
    ),
    Task(
        task_id="duration-hours",
        why="duration_hour_unit",
        description="parse a bare hour chunk",
        cases=(Case("parse_duration", ("2h",), 7200),),
    ),
    Task(
        task_id="duration-mixed",
        why="duration_hour_unit",
        description="parse an hours-and-minutes duration",
        cases=(Case("parse_duration", ("1h 30m",), 5400),),
    ),
    Task(
        task_id="number-decimal",
        why="number_group_separator",
        description="normalise a plain decimal and a currency amount",
        cases=(
            Case("normalize_number", ("12.5",), 12.5),
            Case("normalize_number", ("$3.00",), 3.0),
        ),
    ),
    Task(
        task_id="number-percent",
        why="number_group_separator",
        description="normalise a percentage",
        cases=(Case("normalize_number", ("42%",), 42.0),),
    ),
    Task(
        task_id="number-grouped",
        why="number_group_separator",
        description="normalise a grouped decimal amount",
        cases=(Case("normalize_number", ("1,234.50",), 1234.5),),
    ),
    Task(
        task_id="number-grouped-int",
        why="number_group_separator",
        description="normalise a grouped integer amount",
        cases=(Case("normalize_number", ("2,000",), 2000.0),),
    ),
)

TEST_TASKS: tuple[Task, ...] = (
    Task(
        task_id="sealed-slug-punctuation",
        why="slug_separator_collapse",
        description="slugify a title with a spaced slash",
        cases=(Case("slugify", ("Data / Ops!",), "data-ops"),),
    ),
    Task(
        task_id="sealed-duration-hour-seconds",
        why="duration_hour_unit",
        description="parse an hours-and-seconds duration",
        cases=(Case("parse_duration", ("1h 5s",), 3605),),
    ),
    Task(
        task_id="sealed-number-currency-grouped",
        why="number_group_separator",
        description="normalise a grouped currency amount",
        cases=(Case("normalize_number", ("$1,000,000",), 1000000.0),),
    ),
    Task(
        task_id="sealed-slug-plain",
        why="slug_separator_collapse",
        description="slugify a single word",
        cases=(Case("slugify", ("Alpha",), "alpha"),),
    ),
)

ALL_TASKS: dict[str, Task] = {task.task_id: task for task in TRAIN_TASKS + TEST_TASKS}


def train_task_ids() -> list[str]:
    return [task.task_id for task in TRAIN_TASKS]


def test_task_ids() -> list[str]:
    return [task.task_id for task in TEST_TASKS]


def task_for(task_id: str) -> Task:
    try:
        return ALL_TASKS[task_id]
    except KeyError as exc:
        raise ValueError(f"unknown task id {task_id!r}; known ids: {sorted(ALL_TASKS)}") from exc


__all__ = [
    "ALL_TASKS",
    "Case",
    "PUBLIC_SURFACE",
    "TEST_TASKS",
    "TRAIN_TASKS",
    "Task",
    "WHY_DEFINITIONS",
    "task_for",
    "test_task_ids",
    "train_task_ids",
]
