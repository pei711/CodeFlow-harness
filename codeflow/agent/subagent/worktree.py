"""Opt-in development workspaces and explicit, verified integration.

One runtime owns this manager. No task queue, distributed claim, automatic
commit, destructive cleanup, or conflict resolution is implied by this API.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from codeflow.spine._barrier import finish_barrier
from codeflow.utils.async_io import run_blocking_io
from codeflow.utils.atomic_file import atomic_write_bytes
from codeflow.utils.subprocess import stop_process

Verifier = Callable[[Path], Awaitable[bool]]


class WorktreeError(RuntimeError):
    """An operation was refused; all existing workspace contents are retained."""


@dataclass(frozen=True)
class WorkspaceRecord:
    task_id: str
    path: Path
    branch: str
    base: str
    status: str
    head: str = ""
    dirty: bool | None = None
    error: str = ""
    target_ref: str = ""
    reviewed_heads: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReviewedChange:
    task_id: str
    head: str
    base: str
    diff: str


@dataclass(frozen=True)
class IntegrationCandidate:
    task_id: str
    target_ref: str
    target_base: str
    head: str


class GitWorktreeManager:
    def __init__(self, repository: Path, state_dir: Path):
        self.repository = repository.resolve()
        # A sibling root avoids turning a parent's `git add -A` into an
        # accidental embedded-repository/submodule addition in arbitrary repos.
        self.root = self.repository.parent / f".{self.repository.name}.codeflow-worktrees"
        self.state_dir = state_dir.resolve()
        self.records: dict[str, WorkspaceRecord] = {}
        self._verified: dict[str, IntegrationCandidate] = {}
        self._management_lock = asyncio.Lock()

    async def _git(self, cwd: Path, *args: str, check: bool = True) -> str:
        # An embedding CLI can inherit Git-specific environment variables.
        env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        process = await asyncio.create_subprocess_exec(
            "git",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "user.name=CodeFlow",
            "-c",
            "user.email=integration@codeflow.local",
            *args,
            cwd=str(cwd),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(process.communicate(), 60)
        except (asyncio.CancelledError, TimeoutError):
            await stop_process(process)
            raise
        if check and process.returncode != 0:
            raise WorktreeError(err.decode("utf-8", "replace").strip() or f"git {args[0]} failed")
        return out.decode("utf-8", "replace").strip()

    async def _save(self, record: WorkspaceRecord) -> None:
        self.records[record.task_id] = record
        data = {**asdict(record), "path": str(record.path)}
        await run_blocking_io(
            atomic_write_bytes,
            self.state_dir / f"{record.task_id}.json",
            (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        )

    async def create(self, task_id: str, *, base: str = "HEAD") -> WorkspaceRecord:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", task_id):
            raise ValueError("task_id must be a simple, unique identifier")
        async with self._management_lock:
            top = Path(await self._git(self.repository, "rev-parse", "--show-toplevel")).resolve()
            if top != self.repository:
                raise WorktreeError("development workspace must be the repository root")
            path = (self.root / task_id).resolve()
            if not path.is_relative_to(self.root.resolve()) or path == self.root.resolve():
                raise WorktreeError("worktree path escapes managed root")
            if task_id in self.records or path.exists() or (self.state_dir / f"{task_id}.json").exists():
                raise WorktreeError(f"task workspace already exists: {task_id}")
            base_sha = await self._git(
                self.repository, "rev-parse", "--verify", "--end-of-options", f"{base}^{{commit}}"
            )
            branch = f"codeflow/task/{task_id}-{uuid.uuid4().hex[:8]}"
            record = WorkspaceRecord(task_id, path, branch, base_sha, "preparing")
            try:
                await self._save(record)
                # Starts from committed history; uncommitted parent changes are
                # intentionally not copied, staged, stashed, or discarded.
                await self._git(self.repository, "worktree", "add", "-b", branch, str(path), base_sha)
                return await self.finish(task_id, "running")
            except BaseException as exc:
                await finish_barrier(asyncio.create_task(self._record_failure(task_id, exc)))
                raise

    async def _record_failure(self, task_id: str, exc: BaseException) -> None:
        await self.finish(task_id, "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed", str(exc))

    async def finish(self, task_id: str, status: str, error: str | None = None) -> WorkspaceRecord:
        record = self.records[task_id]
        error = record.error if error is None else error
        try:
            top = Path(await self._git(record.path, "rev-parse", "--show-toplevel")).resolve()
            if top != record.path:
                raise WorktreeError("worktree is incomplete or was moved")
            branch = await self._git(record.path, "symbolic-ref", "--short", "HEAD")
            if branch != record.branch:
                raise WorktreeError("workspace is no longer on its assigned feature branch")
            head = await self._git(record.path, "rev-parse", "HEAD")
            dirty = bool(await self._git(record.path, "status", "--porcelain", "--untracked-files=all"))
        except (OSError, WorktreeError) as exc:
            head, dirty = record.head, None
            error = f"{error}\nUnable to inspect workspace: {exc}".strip()
        record = replace(record, status=status, head=head, dirty=dirty, error=error)
        await self._save(record)
        return record

    async def review(self, task_id: str) -> ReviewedChange:
        record = await self.finish(task_id, self.records[task_id].status)
        if record.dirty is not False or not record.head:
            raise WorktreeError("review requires a committed, clean workspace; pending edits have been retained")
        diff = await self._git(record.path, "diff", "--no-ext-diff", "--no-textconv", f"{record.base}..{record.head}")
        return ReviewedChange(task_id, record.head, record.base, diff)

    async def integrate(
        self, approved_reviews: Sequence[ReviewedChange], *, target: str, verifier: Verifier
    ) -> IntegrationCandidate:
        """Caller reviews the pinned diffs before passing approved snapshots.

        Merge failures, rejected verification and cancellation retain the entire
        integration worktree (including MERGE_HEAD) and never change target.
        """
        if not approved_reviews:
            raise WorktreeError("at least one reviewed change is required")
        await self._git(self.repository, "check-ref-format", "--branch", target)
        target_ref = f"refs/heads/{target}"
        target_base = await self._git(self.repository, "rev-parse", "--verify", target_ref)
        for approved in approved_reviews:
            current = await self.review(approved.task_id)
            if current != approved:
                raise WorktreeError("branch changed after review; review the new diff")
        record = await self.create(f"integration-{uuid.uuid4().hex}", base=target_base)
        try:
            record = replace(
                record,
                status="integrating",
                target_ref=target_ref,
                reviewed_heads=tuple(r.head for r in approved_reviews),
            )
            await self._save(record)
            for approved in approved_reviews:
                await self._git(record.path, "merge", "--no-ff", "--no-edit", approved.head)
            head = await self._git(record.path, "rev-parse", "HEAD")
            # Only an explicit True is acceptance, not an arbitrary truthy
            # object, exit-code string, or an Agent's completion prose.
            if await verifier(record.path) is not True:
                raise WorktreeError("integration verifier rejected the candidate")
            final = await self.finish(record.task_id, "verified")
            if final.head != head or final.dirty is not False:
                raise WorktreeError("verifier changed the candidate; review and verify again")
            candidate = IntegrationCandidate(record.task_id, target_ref, target_base, head)
            self._verified[record.task_id] = candidate
            return candidate
        except BaseException as exc:
            await finish_barrier(asyncio.create_task(self._record_failure(record.task_id, exc)))
            raise

    async def promote(self, candidate: IntegrationCandidate) -> None:
        """Explicitly advance an unchecked-out target with compare-and-swap.

        Refuse checked-out branches rather than desynchronizing another user's
        index/worktree. External Git management must not run concurrently.
        """
        async with self._management_lock:
            if self._verified.get(candidate.task_id) != candidate:
                raise WorktreeError("candidate is not verified by this runtime")
            record = await self.finish(candidate.task_id, "verified")
            if record.head != candidate.head or record.dirty is not False:
                self._verified.pop(candidate.task_id, None)
                raise WorktreeError("candidate changed since verification")
            listing = await self._git(self.repository, "worktree", "list", "--porcelain")
            if f"branch {candidate.target_ref}" in listing.splitlines():
                raise WorktreeError(
                    "target is checked out; retain candidate for explicit integration in that workspace"
                )

            async def publish() -> None:
                await self._git(
                    self.repository, "update-ref", candidate.target_ref, candidate.head, candidate.target_base
                )
                self._verified.pop(candidate.task_id, None)
                await self.finish(candidate.task_id, "promoted")

            # Once the short CAS transaction starts, cancellation must not
            # separate a successful ref update from its status record.
            await finish_barrier(asyncio.create_task(publish()))
