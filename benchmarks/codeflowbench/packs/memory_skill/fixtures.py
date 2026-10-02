"""Frozen historical Memory/Skill fixtures; source labels are evidence identity."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from benchmarks.codeflowbench.canonical import canonical_digest, to_primitive
from benchmarks.codeflowbench.schema import (
    RetrievalConfigurationSpec,
    RetrievalQuerySpec,
    RetrievalSuiteSpec,
    TaskSpec,
)

from .models import CrossSessionTask, MemoryFact, SkillItem
from .tasks import load_cross_session_tasks


@lru_cache(maxsize=1)
def retrieval_fixture_manifest() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[2] / "fixtures" / "retrieval" / "memory_skill_v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "codeflow.codeflowbench.retrieval-fixture.v1":
        raise ValueError(f"unsupported retrieval fixture schema: {path}")
    return dict(payload)


def anonymous_item_id(suite_id: str, item_id: str) -> str:
    return hashlib.sha256(f"{suite_id}:{item_id}".encode()).hexdigest()[:20]


def formal_memory_corpus() -> tuple[MemoryFact, ...]:
    active = tuple(
        MemoryFact(
            item_id=f"memory-active-{index:03d}",
            workspace_id=f"workspace-{index % 8}",
            text=f"account {index:03d} deployment nonce orchid-{index:03d}",
        )
        for index in range(50)
    )
    stale = tuple(
        MemoryFact(
            item_id=f"memory-stale-{index:03d}",
            workspace_id=f"workspace-{index % 8}",
            text=f"account {index:03d} deployment nonce obsolete-{index:03d}",
            active=False,
            superseded=True,
        )
        for index in range(50)
    )
    cross_workspace = tuple(
        MemoryFact(
            item_id=f"memory-cross-{index:03d}",
            workspace_id=f"workspace-{(index + 1) % 8}",
            text=f"account {index:03d} deployment nonce orchid-{index:03d}",
        )
        for index in range(30)
    )
    unrelated = tuple(
        MemoryFact(
            item_id=f"memory-noise-{index:03d}",
            workspace_id=f"workspace-{index % 8}",
            text=f"unrelated service preference cobaltnoise{index:03d}",
        )
        for index in range(30)
    )
    return (*active, *stale, *cross_workspace, *unrelated)


def formal_memory_queries() -> tuple[RetrievalQuerySpec, ...]:
    suite_id = "user-memory-retrieval-v1"
    positives = tuple(
        RetrievalQuerySpec(
            query_id=f"memory-positive-{index:03d}",
            label="positive",
            expected_item_ids=(anonymous_item_id(suite_id, f"memory-active-{index:03d}"),),
            payload={
                "query_text": f"deployment nonce for account {index:03d} orchid-{index:03d}",
                "workspace_id": f"workspace-{index % 8}",
                "consuming_turn": f"memory-turn-{index:03d}",
            },
        )
        for index in range(50)
    )
    negatives = tuple(
        RetrievalQuerySpec(
            query_id=f"memory-negative-{index:03d}",
            label="hard_negative",
            payload={
                "query_text": f"unknown account zephyrnotfound{index:03d}",
                "workspace_id": f"workspace-{index % 8}",
                "consuming_turn": f"memory-negative-turn-{index:03d}",
            },
        )
        for index in range(30)
    )
    return (*positives, *negatives)


def formal_skill_corpus() -> tuple[SkillItem, ...]:
    items: list[SkillItem] = []
    for index in range(40):
        workspace_id = f"workspace-{index % 8}"
        if index < 25:
            items.append(
                SkillItem(
                    item_id=f"skill-local-{index:03d}",
                    logical_id=f"skill-{index:03d}",
                    workspace_id=workspace_id,
                    source="local",
                    text=(f"repair workflow skilltoken{index:03d} local procedure ambertoken{index:03d}"),
                )
            )
        if index >= 15:
            items.append(
                SkillItem(
                    item_id=f"skill-everos-{index:03d}",
                    logical_id=f"skill-{index:03d}",
                    workspace_id=workspace_id,
                    source="everos",
                    text=(f"repair workflow skilltoken{index:03d} semantic procedure ambertoken{index:03d}"),
                )
            )
    items.extend(
        SkillItem(
            item_id=f"skill-cross-{index:03d}",
            logical_id=f"skill-cross-{index:03d}",
            workspace_id=f"workspace-{(index + 1) % 8}",
            source="everos" if index % 2 else "local",
            text=f"foreign distractor crossamber{index:03d}",
        )
        for index in range(20)
    )
    items.extend(
        SkillItem(
            item_id=f"skill-noise-{index:03d}",
            logical_id=f"skill-noise-{index:03d}",
            workspace_id=f"workspace-{index % 8}",
            source="everos" if index % 2 else "local",
            text=f"unrelated documentation workflow indigo-{index:03d}",
        )
        for index in range(20)
    )
    return tuple(items)


def formal_skill_queries() -> tuple[RetrievalQuerySpec, ...]:
    suite_id = "skill-source-fusion-v1"
    positives = tuple(
        RetrievalQuerySpec(
            query_id=f"skill-positive-{index:03d}",
            label="positive",
            expected_item_ids=(anonymous_item_id(suite_id, f"skill-{index:03d}"),),
            payload={
                "query_text": f"ambertoken{index:03d} skilltoken{index:03d}",
                "workspace_id": f"workspace-{index % 8}",
                "consuming_turn": f"skill-turn-{index:03d}",
            },
        )
        for index in range(40)
    )
    negatives = tuple(
        RetrievalQuerySpec(
            query_id=f"skill-negative-{index:03d}",
            label="hard_negative",
            payload={
                "query_text": f"unsupported quartznotfound{index:03d}",
                "workspace_id": f"workspace-{index % 8}",
                "consuming_turn": f"skill-negative-turn-{index:03d}",
            },
        )
        for index in range(20)
    )
    return (*positives, *negatives)


def formal_retrieval_suites() -> tuple[RetrievalSuiteSpec, ...]:
    memory_corpus = formal_memory_corpus()
    memory_queries = formal_memory_queries()
    skill_corpus = formal_skill_corpus()
    skill_queries = formal_skill_queries()
    return (
        RetrievalSuiteSpec(
            retrieval_suite_id="user-memory-retrieval-v1",
            queries=memory_queries,
            configurations=(
                RetrievalConfigurationSpec(
                    configuration_id="user_memory_on",
                    settings={"user_memory_recall": "enabled"},
                ),
            ),
            corpus_digest=canonical_digest(memory_corpus),
            query_labels_digest=canonical_digest(memory_queries),
        ),
        RetrievalSuiteSpec(
            retrieval_suite_id="skill-source-fusion-v1",
            queries=skill_queries,
            configurations=tuple(
                RetrievalConfigurationSpec(
                    configuration_id=configuration_id,
                    settings={"retrieval_configuration": configuration_id},
                )
                for configuration_id in ("local_only", "everos_only", "fused")
            ),
            corpus_digest=canonical_digest(skill_corpus),
            query_labels_digest=canonical_digest(skill_queries),
        ),
    )


def formal_cross_session_tasks() -> tuple[CrossSessionTask, ...]:
    return load_cross_session_tasks("formal")


def formal_task_specs() -> tuple[TaskSpec, ...]:
    return tuple(
        TaskSpec(
            task_id=task.task_id,
            payload=to_primitive(task),
        )
        for task in formal_cross_session_tasks()
    )


def calibration_memory_corpus() -> tuple[MemoryFact, ...]:
    active = tuple(
        MemoryFact(
            item_id=f"cal-memory-active-{index:02d}",
            workspace_id=f"cal-workspace-{index % 4}",
            text=f"calibration tenant {index:02d} routing token marigold-{index:02d}",
        )
        for index in range(6)
    )
    stale = tuple(
        MemoryFact(
            item_id=f"cal-memory-stale-{index:02d}",
            workspace_id=f"cal-workspace-{index % 4}",
            text=f"calibration tenant {index:02d} routing token retired-{index:02d}",
            active=False,
            superseded=True,
        )
        for index in range(6)
    )
    cross_workspace = tuple(
        MemoryFact(
            item_id=f"cal-memory-cross-{index:02d}",
            workspace_id=f"cal-workspace-{(index + 1) % 4}",
            text=f"calibration tenant {index:02d} routing token marigold-{index:02d}",
        )
        for index in range(4)
    )
    return (*active, *stale, *cross_workspace)


def calibration_memory_queries() -> tuple[RetrievalQuerySpec, ...]:
    suite_id = "user-memory-retrieval-calibration-v1"
    positives = tuple(
        RetrievalQuerySpec(
            query_id=f"cal-memory-positive-{index:02d}",
            label="positive",
            expected_item_ids=(anonymous_item_id(suite_id, f"cal-memory-active-{index:02d}"),),
            payload={
                "query_text": (f"routing token for calibration tenant {index:02d} marigold-{index:02d}"),
                "workspace_id": f"cal-workspace-{index % 4}",
                "consuming_turn": f"cal-memory-turn-{index:02d}",
            },
        )
        for index in range(6)
    )
    negatives = tuple(
        RetrievalQuerySpec(
            query_id=f"cal-memory-negative-{index:02d}",
            label="hard_negative",
            payload={
                "query_text": f"missing routing token heliotropenotfound{index:02d}",
                "workspace_id": f"cal-workspace-{index % 4}",
                "consuming_turn": f"cal-memory-negative-turn-{index:02d}",
            },
        )
        for index in range(4)
    )
    return (*positives, *negatives)


def calibration_skill_corpus() -> tuple[SkillItem, ...]:
    items: list[SkillItem] = []
    for index in range(5):
        workspace_id = f"cal-workspace-{index % 4}"
        if index < 3:
            items.append(
                SkillItem(
                    item_id=f"cal-skill-local-{index:02d}",
                    logical_id=f"cal-skill-{index:02d}",
                    workspace_id=workspace_id,
                    source="local",
                    text=(f"calibration repair calskilltoken{index:02d} local procedure saffrontoken{index:02d}"),
                )
            )
        if index >= 2:
            items.append(
                SkillItem(
                    item_id=f"cal-skill-everos-{index:02d}",
                    logical_id=f"cal-skill-{index:02d}",
                    workspace_id=workspace_id,
                    source="everos",
                    text=(f"calibration repair calskilltoken{index:02d} semantic procedure saffrontoken{index:02d}"),
                )
            )
    items.extend(
        SkillItem(
            item_id=f"cal-skill-cross-{index:02d}",
            logical_id=f"cal-skill-cross-{index:02d}",
            workspace_id=f"cal-workspace-{(index + 1) % 4}",
            source="everos" if index % 2 else "local",
            text=f"calibration foreign crosssaffron{index:02d}",
        )
        for index in range(4)
    )
    return tuple(items)


def calibration_skill_queries() -> tuple[RetrievalQuerySpec, ...]:
    suite_id = "skill-source-fusion-calibration-v1"
    positives = tuple(
        RetrievalQuerySpec(
            query_id=f"cal-skill-positive-{index:02d}",
            label="positive",
            expected_item_ids=(anonymous_item_id(suite_id, f"cal-skill-{index:02d}"),),
            payload={
                "query_text": f"saffrontoken{index:02d} calskilltoken{index:02d}",
                "workspace_id": f"cal-workspace-{index % 4}",
                "consuming_turn": f"cal-skill-turn-{index:02d}",
            },
        )
        for index in range(5)
    )
    negatives = tuple(
        RetrievalQuerySpec(
            query_id=f"cal-skill-negative-{index:02d}",
            label="hard_negative",
            payload={
                "query_text": f"unsupported carnelianunknown{index:02d}",
                "workspace_id": f"cal-workspace-{index % 4}",
                "consuming_turn": f"cal-skill-negative-turn-{index:02d}",
            },
        )
        for index in range(3)
    )
    return (*positives, *negatives)


def calibration_retrieval_suites() -> tuple[RetrievalSuiteSpec, ...]:
    memory_corpus = calibration_memory_corpus()
    memory_queries = calibration_memory_queries()
    skill_corpus = calibration_skill_corpus()
    skill_queries = calibration_skill_queries()
    return (
        RetrievalSuiteSpec(
            retrieval_suite_id="user-memory-retrieval-calibration-v1",
            queries=memory_queries,
            configurations=(
                RetrievalConfigurationSpec(
                    configuration_id="user_memory_on",
                    settings={"user_memory_recall": "enabled"},
                ),
            ),
            corpus_digest=canonical_digest(memory_corpus),
            query_labels_digest=canonical_digest(memory_queries),
        ),
        RetrievalSuiteSpec(
            retrieval_suite_id="skill-source-fusion-calibration-v1",
            queries=skill_queries,
            configurations=tuple(
                RetrievalConfigurationSpec(
                    configuration_id=configuration_id,
                    settings={"retrieval_configuration": configuration_id},
                )
                for configuration_id in ("local_only", "everos_only", "fused")
            ),
            corpus_digest=canonical_digest(skill_corpus),
            query_labels_digest=canonical_digest(skill_queries),
        ),
    )


def calibration_cross_session_tasks() -> tuple[CrossSessionTask, ...]:
    return load_cross_session_tasks("calibration")


def calibration_task_specs() -> tuple[TaskSpec, ...]:
    return tuple(
        TaskSpec(
            task_id=task.task_id,
            payload=to_primitive(task),
        )
        for task in calibration_cross_session_tasks()
    )
