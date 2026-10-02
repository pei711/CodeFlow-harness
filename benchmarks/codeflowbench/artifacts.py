from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from fcntl import LOCK_EX, LOCK_NB, LOCK_UN, flock
from pathlib import Path
from typing import Any, Iterator

from .canonical import canonical_bytes, canonical_json, to_primitive
from .records import (
    AttemptKey,
    ComparisonBlockKey,
    PairKey,
    RetrievalAttemptKey,
    RetrievalCaseKey,
    RetrievalQueryBlockKey,
    TrialKey,
)
from .schema import ExperimentRef


class ArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArtifactStore:
    ref: ExperimentRef

    @property
    def root(self) -> Path:
        return self.ref.root

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def journal_path(self) -> Path:
        return self.root / "journal.jsonl"

    @contextmanager
    def exclusive_run_lock(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / ".run.lock"
        with lock_path.open("a+b") as lock:
            try:
                flock(lock.fileno(), LOCK_EX | LOCK_NB)
            except BlockingIOError as exc:
                raise ArtifactError(
                    f"experiment already has an active writer: {self.ref.experiment_id}",
                ) from exc
            try:
                yield
            finally:
                flock(lock.fileno(), LOCK_UN)

    def freeze_manifest(self, manifest: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        expected = canonical_bytes(manifest)
        if self.manifest_path.exists():
            try:
                actual = canonical_bytes(self.read_json(self.manifest_path))
            except ArtifactError as exc:
                raise ArtifactError("existing manifest is corrupt") from exc
            if actual != expected:
                raise ArtifactError("existing manifest does not match the experiment plan")
            return
        self._write_atomic(self.manifest_path, expected)

    def append_journal(self, event: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = canonical_json(event) + "\n"
        with self.journal_path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    def append_immutable(self, path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = canonical_bytes(value)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temp_path, path)
            self._fsync_directory(path.parent)
        except FileExistsError:
            try:
                existing = canonical_bytes(self.read_json(path))
            except ArtifactError as exc:
                raise ArtifactError(f"immutable artifact is corrupt: {path}") from exc
            if existing != payload:
                raise ArtifactError(f"immutable artifact already exists with different data: {path}")
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def write_summary(self, path: Path, value: Any) -> None:
        self._write_atomic(path, canonical_bytes(value))

    def read_json(self, path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"cannot read artifact: {path}") from exc
        if not isinstance(value, dict):
            raise ArtifactError(f"artifact must contain a JSON object: {path}")
        return value

    def read_if_valid(self, path: Path, *, plan_digest: str) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            value = self.read_json(path)
        except ArtifactError:
            return None
        if value.get("plan_digest") != plan_digest:
            return None
        return value

    def attempt_path(self, key: AttemptKey) -> Path:
        return (
            self.root
            / "trials"
            / key.block.pack_id
            / key.block.task_id
            / str(key.block.repetition)
            / key.variant_id
            / "attempts"
            / str(key.block_attempt)
            / "attempt-record.json"
        )

    def trial_path(self, key: TrialKey) -> Path:
        return (
            self.root
            / "trials"
            / key.pack_id
            / key.task_id
            / str(key.repetition)
            / key.variant_id
            / "trial-record.json"
        )

    def block_path(self, key: ComparisonBlockKey) -> Path:
        return self.root / "blocks" / key.pack_id / key.task_id / str(key.repetition) / "block-result.json"

    def comparison_block_retry_claim_path(
        self,
        key: ComparisonBlockKey,
        block_attempt: int,
    ) -> Path:
        return (
            self.root
            / "blocks"
            / key.pack_id
            / key.task_id
            / str(key.repetition)
            / "retry-claims"
            / f"{block_attempt}.json"
        )

    def claim_comparison_block_retry(
        self,
        *,
        key: ComparisonBlockKey,
        block_attempt: int,
        plan_digest: str,
        maximum_claims: int | None,
    ) -> bool:
        if block_attempt < 2:
            raise ValueError("comparison block retry attempts start at two")
        if maximum_claims is not None and maximum_claims < 0:
            raise ValueError("maximum retry claims must not be negative")
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / ".comparison-block-retries.lock"
        claim_path = self.comparison_block_retry_claim_path(
            key,
            block_attempt,
        )
        claim = {
            "kind": "comparison_block_retry_claim",
            "plan_digest": plan_digest,
            "key": artifact_dict(key),
            "block_attempt": block_attempt,
        }
        with lock_path.open("a+b") as lock:
            flock(lock.fileno(), LOCK_EX)
            try:
                if claim_path.exists():
                    existing = self.read_json(claim_path)
                    if canonical_bytes(existing) != canonical_bytes(claim):
                        raise ArtifactError(
                            f"comparison block retry claim does not match the experiment plan: {claim_path}",
                        )
                    return True
                claimed = self._count_comparison_block_retry_claims(
                    plan_digest=plan_digest,
                )
                if maximum_claims is not None and claimed >= maximum_claims:
                    return False
                self.append_immutable(claim_path, claim)
                return True
            finally:
                flock(lock.fileno(), LOCK_UN)

    def pair_path(self, key: PairKey) -> Path:
        return (
            self.root
            / "pairs"
            / key.pack_id
            / key.treatment_axis
            / key.task_id
            / str(key.repetition)
            / "pair-result.json"
        )

    def retrieval_attempt_path(self, key: RetrievalAttemptKey) -> Path:
        return (
            self.root
            / "retrieval"
            / key.block.retrieval_suite_id
            / key.block.query_id
            / key.configuration_id
            / "attempts"
            / str(key.query_block_attempt)
            / "retrieval-attempt-record.json"
        )

    def retrieval_case_path(self, key: RetrievalCaseKey) -> Path:
        return (
            self.root
            / "retrieval"
            / key.retrieval_suite_id
            / key.query_id
            / key.configuration_id
            / "retrieval-case-record.json"
        )

    def retrieval_block_path(self, key: RetrievalQueryBlockKey) -> Path:
        return (
            self.root
            / "retrieval"
            / "query-blocks"
            / key.retrieval_suite_id
            / key.query_id
            / "retrieval-query-block-result.json"
        )

    def relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def next_block_attempt(self, key: ComparisonBlockKey) -> int:
        attempts = self.root / "trials" / key.pack_id / key.task_id / str(key.repetition)
        seen: list[int] = []
        if attempts.exists():
            for path in attempts.glob("*/attempts/*"):
                if path.name.isdigit():
                    seen.append(int(path.name))
        return max(seen, default=0) + 1

    def next_retrieval_attempt(self, key: RetrievalQueryBlockKey) -> int:
        root = self.root / "retrieval" / key.retrieval_suite_id / key.query_id
        seen: list[int] = []
        if root.exists():
            for path in root.glob("*/attempts/*"):
                if path.name.isdigit():
                    seen.append(int(path.name))
        return max(seen, default=0) + 1

    def _count_comparison_block_retry_claims(
        self,
        *,
        plan_digest: str,
    ) -> int:
        claims_root = self.root / "blocks"
        if not claims_root.exists():
            return 0
        count = 0
        for path in claims_root.glob("*/*/*/retry-claims/*.json"):
            claim = self.read_json(path)
            if (
                claim.get("kind") != "comparison_block_retry_claim"
                or claim.get("plan_digest") != plan_digest
                or not isinstance(claim.get("block_attempt"), int)
                or isinstance(claim.get("block_attempt"), bool)
                or int(claim["block_attempt"]) < 2
            ):
                raise ArtifactError(
                    f"invalid comparison block retry claim: {path}",
                )
            count += 1
        return count

    def _write_atomic(self, path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(temp_path, path)
            self._fsync_directory(path.parent)
        finally:
            temp_path.unlink(missing_ok=True)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        directory_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def artifact_dict(value: Any) -> dict[str, Any]:
    primitive = to_primitive(value)
    if not isinstance(primitive, dict):
        raise TypeError("artifact record must serialize to an object")
    return primitive
