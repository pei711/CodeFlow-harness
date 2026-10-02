"""Round journal — append-only checkpoint so a killed run resumes.

Each completed round appends one JSON line with just enough to reconstruct the
loop's control state on restart: the round index, the parent it evolved to, and
whether it promoted (which drives the termination counter). On resume the loop
replays these records to seed the :class:`TerminationTracker`, the current
parent, and the round counter, then continues from the next round without
re-running the expensive evals it already did.

The node ledger (``nodes/*.json``) is the durable record of the *nodes*; this
journal is the durable record of the *loop's progress* over them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 避免与 loop 形成运行时导入环
    from codeflow.evolver.orchestrator.loop import RoundResult


@dataclass
class RoundJournal:
    """Append-only JSONL of completed-round checkpoints."""

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def append(self, rr: "RoundResult") -> None:
        """Persist a compact checkpoint for one completed round."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "round_index": rr.round_index,
            "parent_id": rr.parent_id,
            "next_parent_id": rr.next_parent_id,
            "promoted": rr.promoted,
            # 两个终止信号：SOP 规定耐心与原始版本比较，错误轮次另有计数器；恢复时会重放这些状态。
            "beat_vanilla": rr.beat_vanilla,
            "errored": rr.errored,
            "verdict": rr.verdict,
            # 用于事后密封解封（C3）：记录可交付项的提交和训练 pass@1，使运行后可重建测试曲线。
            "next_parent_sha": rr.next_parent_sha,
            "next_parent_train": rr.next_parent_train,
            "candidates": [
                {
                    "node_id": o.node_id,
                    "status": o.status.value,
                    "verdict": o.verdict.value if o.verdict is not None else "inconclusive",
                }
                for o in rr.outcomes
            ],
        }
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def load(self) -> list[dict]:
        """Read completed-round checkpoints in order (empty if none).

        A malformed FINAL line is tolerated and dropped — a kill/disk-full mid
        ``append`` leaves a truncated last record, and the journal exists
        precisely to survive that; the interrupted round simply re-runs.
        Corruption anywhere earlier still raises (the history is not trustable).
        """
        if not self.path.exists():
            return []
        lines = [ln.strip() for ln in self.path.read_text().splitlines() if ln.strip()]
        out = []
        for i, line in enumerate(lines):
            try:
                out.append(json.loads(line))
            except ValueError as exc:
                if i == len(lines) - 1:
                    break
                raise ValueError(f"corrupt journal record at line {i + 1} of {self.path}: {exc}") from exc
        return out


__all__ = ["RoundJournal"]
