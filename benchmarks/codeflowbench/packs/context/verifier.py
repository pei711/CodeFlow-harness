from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.codeflowbench.verifier import (
    JsonArtifactVerifier,
    VerifierExecution,
    VerifierSeal,
    run_sealed_verifier,
)

from .models import ContextTask


@dataclass(frozen=True)
class SealedContextTaskVerifier:
    task: ContextTask
    expected_path: Path
    expected: dict[str, Any]
    seal: VerifierSeal

    @classmethod
    def capture(
        cls,
        task: ContextTask,
        *,
        expected_path: Path | None = None,
    ) -> "SealedContextTaskVerifier":
        resolved = Path(expected_path or task.expected_path)
        return cls(
            task=task,
            expected_path=resolved,
            expected=json.loads(resolved.read_text(encoding="utf-8")),
            seal=VerifierSeal.capture(resolved),
        )

    async def verify(self, workspace: Path) -> VerifierExecution:
        verifier = JsonArtifactVerifier(
            expected_path=self.expected_path,
            artifact_path=self.task.artifact_path,
            forbidden_paths=self.task.forbidden_paths,
        )
        return await run_sealed_verifier(
            verifier,
            workspace=workspace,
            seal=self.seal,
        )

    def diagnostic_metrics(self, workspace: Path) -> dict[str, bool]:
        workspace = Path(workspace).resolve()
        artifact = (workspace / self.task.artifact_path).resolve()
        forbidden_paths_clean = all(not (workspace / path).resolve().exists() for path in self.task.forbidden_paths)
        metrics = {
            "artifact_valid_json": False,
            "active_constraint_applied": False,
            "latest_decision_applied": False,
            "artifact_exact": False,
            "forbidden_paths_clean": forbidden_paths_clean,
        }
        if not self.seal.intact() or not artifact.is_relative_to(workspace) or not artifact.is_file():
            return metrics
        try:
            actual = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return metrics
        if not isinstance(actual, dict):
            return metrics
        metrics["artifact_valid_json"] = True
        metrics["active_constraint_applied"] = all(
            actual.get(key) == self.expected.get(key) for key in self.task.constraint_keys
        )
        metrics["latest_decision_applied"] = all(
            actual.get(key) == self.expected.get(key) for key in self.task.decision_keys
        )
        metrics["artifact_exact"] = actual == self.expected
        return metrics


__all__ = ["SealedContextTaskVerifier"]
