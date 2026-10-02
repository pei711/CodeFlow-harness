from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryFact:
    item_id: str
    workspace_id: str
    text: str
    active: bool = True
    superseded: bool = False


@dataclass(frozen=True)
class SkillItem:
    item_id: str
    logical_id: str
    workspace_id: str
    source: str
    text: str


@dataclass(frozen=True)
class CrossSessionTask:
    task_id: str
    workspace_id: str
    learned_fact: str
    expected_value: str
    required_skill: str
    evaluation_request: str


@dataclass(frozen=True)
class SemanticMemoryEffectTask:
    task_id: str
    workspace_id: str
    prior_session_id: str
    customer: str
    service: str
    memory_text: str
    distractor_memories: tuple[str, ...]
    expected_region: str
    expected_retention: str
    expected_approval_code: str
    evaluation_request: str
