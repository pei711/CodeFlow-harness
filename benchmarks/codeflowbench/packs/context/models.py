from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from benchmarks.codeflowbench.canonical import canonical_digest
from benchmarks.codeflowbench.schema import TaskSpec


class ContextTrack(StrEnum):
    FORMAL = "formal"
    CALIBRATION = "calibration"


@dataclass(frozen=True)
class ContextTask:
    task_id: str
    track: ContextTrack
    title: str
    message_count: int
    early_constraint: str
    superseded_before: str
    superseded_after: str
    constraint_keys: tuple[str, ...]
    decision_keys: tuple[str, ...]
    noise_topic: str
    first_tool_label: str
    second_tool_label: str
    final_prompt: str
    artifact_path: str
    forbidden_paths: tuple[str, ...]
    expected_path: Path

    def materialize_history(self) -> list[dict[str, Any]]:
        filler_turns = (self.message_count - 14) // 2
        first_call_id = f"{self.task_id}-tool-a"
        second_call_id = f"{self.task_id}-tool-b"
        history: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": f"[required-early-constraint] {self.early_constraint}",
            },
            {
                "role": "assistant",
                "content": "I recorded the non-negotiable output constraint.",
            },
            {
                "role": "user",
                "content": f"[decision-v1] {self.superseded_before}",
            },
            {
                "role": "assistant",
                "content": "I recorded the initial decision and will update it if superseded.",
            },
            {
                "role": "user",
                "content": f"Inspect the first diagnostic source: {self.first_tool_label}.",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": first_call_id,
                        "type": "function",
                        "function": {
                            "name": "read_diagnostic",
                            "arguments": json.dumps(
                                {"source": self.first_tool_label},
                                sort_keys=True,
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": first_call_id,
                "name": "read_diagnostic",
                "content": _long_tool_result(self.first_tool_label, 1_200),
            },
            {
                "role": "assistant",
                "content": "The diagnostic was recorded; most of it is background evidence.",
            },
        ]
        for index in range(filler_turns):
            history.extend(
                [
                    {
                        "role": "user",
                        "content": (
                            f"[irrelevant-noise:{self.noise_topic}:{index:02d}] "
                            "This side discussion is unrelated to the final artifact "
                            "and must not replace earlier requirements."
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": (f"Noise item {index:02d} acknowledged without changing the task decisions."),
                    },
                ]
            )
        history.extend(
            [
                {
                    "role": "user",
                    "content": f"[decision-v2-supersedes-v1] {self.superseded_after}",
                },
                {
                    "role": "assistant",
                    "content": "The later decision supersedes the earlier one.",
                },
                {
                    "role": "user",
                    "content": f"Inspect the final diagnostic source: {self.second_tool_label}.",
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": second_call_id,
                            "type": "function",
                            "function": {
                                "name": "read_diagnostic",
                                "arguments": json.dumps(
                                    {"source": self.second_tool_label},
                                    sort_keys=True,
                                ),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": second_call_id,
                    "name": "read_diagnostic",
                    "content": _long_tool_result(self.second_tool_label, 900),
                },
                {
                    "role": "assistant",
                    "content": "All evidence is available; apply the final decision and early constraint.",
                },
            ]
        )
        if len(history) != self.message_count:
            raise ValueError(f"{self.task_id} materialized {len(history)} messages, expected {self.message_count}")
        return history

    @property
    def history_digest(self) -> str:
        return self.compute_history_digest()

    def compute_history_digest(self) -> str:
        return canonical_digest(self.materialize_history())

    @property
    def expected_digest(self) -> str:
        return canonical_digest(json.loads(self.expected_path.read_text(encoding="utf-8")))

    def to_task_spec(self) -> TaskSpec:
        return TaskSpec(
            task_id=self.task_id,
            payload={
                "track": self.track.value,
                "title": self.title,
                "message_count": self.message_count,
                "history_digest": self.history_digest,
                "final_prompt": self.final_prompt,
                "constraint_keys": list(self.constraint_keys),
                "decision_keys": list(self.decision_keys),
                "artifact_path": self.artifact_path,
                "forbidden_paths": list(self.forbidden_paths),
                "expected_digest": self.expected_digest,
                "verifier": "external_sealed_json",
            },
        )


def _long_tool_result(label: str, minimum_chars: int) -> str:
    line = f"{label} telemetry datum: component healthy, sample retained for audit, no task decision changed. "
    repetitions = minimum_chars // len(line) + 1
    return f"[tool-result:{label}]\n" + line * repetitions


__all__ = ["ContextTask", "ContextTrack"]
