from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from benchmarks.codeflowbench.canonical import canonical_digest
from .models import SecurityCategory, SecurityScenario, SecurityTrack

FORMAL_SECURITY_SCENARIO_COUNT = 30
SMOKE_SECURITY_SCENARIO_COUNT = 5

_TASK_ROOT = Path(__file__).resolve().parents[2] / "tasks" / "security"


@lru_cache(maxsize=4)
def load_security_scenarios(
    track: SecurityTrack = SecurityTrack.FORMAL,
) -> tuple[SecurityScenario, ...]:
    track = SecurityTrack(track)
    source_path = _TASK_ROOT / f"{track.value}.json"
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing Security scenarios file: {source_path}")

    raw = json.loads(source_path.read_text(encoding="utf-8"))
    entries = raw.get("scenarios")
    if not isinstance(entries, list):
        raise ValueError(f"scenarios list missing in {source_path}")

    scenarios = tuple(_parse_scenario(track, entry) for entry in entries)
    expected_count = (
        FORMAL_SECURITY_SCENARIO_COUNT
        if track is SecurityTrack.FORMAL
        else SMOKE_SECURITY_SCENARIO_COUNT
    )
    if len(scenarios) != expected_count:
        raise ValueError(
            f"{track.value} requires exactly {expected_count} security scenarios, got {len(scenarios)}"
        )

    if len({s.task_id for s in scenarios}) != len(scenarios):
        raise ValueError(f"duplicate scenario id in {source_path}")

    return scenarios


def security_task_set_digest(track: SecurityTrack = SecurityTrack.FORMAL) -> str:
    scenarios = load_security_scenarios(track)
    return canonical_digest(
        [
            {
                "task_id": s.task_id,
                "category": s.category.value,
                "target_action": s.target_action,
                "payload": s.payload,
                "expected_guard": s.expected_guard,
            }
            for s in scenarios
        ]
    )


def _parse_scenario(track: SecurityTrack, raw: dict[str, Any]) -> SecurityScenario:
    return SecurityScenario(
        task_id=str(raw["task_id"]),
        category=SecurityCategory(str(raw["category"])),
        title=str(raw["title"]),
        target_action=str(raw["target_action"]),
        payload=dict(raw.get("payload", {})),
        expected_guard=str(raw["expected_guard"]),
        description=str(raw.get("description", "")),
        track=track,
    )
