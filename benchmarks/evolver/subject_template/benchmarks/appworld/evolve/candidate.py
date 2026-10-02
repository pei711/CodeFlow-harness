"""Candidate representation and the G5 pre-commit manifest binding.

A candidate is the full new bytes of the one mutable file. Turning it into a
Candidate Manifest happens *before* the child commit is created, so a G5
rejection is a reproducible ``rejected`` verdict that leaves no commit behind.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import unified_diff
from pathlib import Path

from benchmarks.appworld.evolve.adapter import MODULE_PATH
from codeflow.evolver.candidate_manifest import (
    CandidateLabel,
    CandidateManifest,
    ManifestGateError,
    assert_manifest_gate,
    manifest_for_patch,
)
from codeflow.evolver.judge.schema import PatchWhere, PatchWhy
from codeflow.evolver.tree import git_ops
from codeflow.evolver.tree.git_ops import GitOpError
from codeflow.evolver.tree.node import AppliedPatch, PatchComponent


@dataclass
class Candidate:
    """One designed rewrite of the subject module."""

    files: dict[str, bytes]
    why: str
    focused_task_ids: list[str] = field(default_factory=list)
    summary: str = ""
    deletions: list[str] = field(default_factory=list)
    has_beacon: bool = False
    label: CandidateLabel | None = None
    before_files: dict[str, bytes | None] = field(default_factory=dict, repr=False)
    applied_patch: AppliedPatch | None = field(default=None, repr=False)
    manifest: CandidateManifest | None = None
    candidate_id: str = ""


def files_of(candidate: Candidate) -> dict[str, bytes]:
    return candidate.files


def _component_diff(path: str, before: bytes | None, after: bytes | None) -> str:
    try:
        before_text = "" if before is None else before.decode("utf-8")
        after_text = "" if after is None else after.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"candidate target must contain UTF-8 text for manifest binding: {path!r}") from exc
    return "".join(
        unified_diff(
            before_text.splitlines(keepends=True),
            after_text.splitlines(keepends=True),
            fromfile="/dev/null" if before is None else f"a/{path}",
            tofile="/dev/null" if after is None else f"b/{path}",
        )
    )


def materialize_candidate_patch(candidate: Candidate, before_files: dict[str, bytes | None]) -> AppliedPatch:
    paths = sorted(candidate.files)
    if not paths:
        raise ValueError("candidate has no target files")
    outside = [path for path in paths if path != MODULE_PATH]
    if outside:
        raise ValueError(f"candidate targets outside the subject mutable surface: {outside}")
    try:
        why = PatchWhy(candidate.why)
        why_extra = None
    except ValueError:
        why = PatchWhy.other
        why_extra = candidate.why or "unspecified"
    patch = AppliedPatch(
        patch_where=PatchWhere.loop_override,
        patch_why=why,
        patch_why_extra=why_extra,
        components=[
            PatchComponent(
                component_id=f"comp_{index}",
                target_file=path,
                diff=_component_diff(path, before_files.get(path), candidate.files.get(path)),
                rationale=candidate.summary or candidate.why,
            )
            for index, path in enumerate(paths, start=1)
        ],
        overall_reasoning=candidate.summary or candidate.why,
    )
    candidate.label = CandidateLabel.runtime
    candidate.before_files = {path: before_files.get(path) for path in paths}
    candidate.applied_patch = patch
    return patch


def prepare_candidate_manifest(
    candidate_id: str,
    parent_sha: str,
    candidate: Candidate,
    *,
    repo_root: str | Path,
) -> CandidateManifest:
    paths = sorted(candidate.files)
    before_files: dict[str, bytes | None] = {}
    for path in paths:
        try:
            before_files[path] = git_ops.read_file_at(Path(repo_root), parent_sha, path)
        except GitOpError:
            before_files[path] = None
    try:
        patch = materialize_candidate_patch(candidate, before_files)
    except ValueError as exc:
        candidate.candidate_id = candidate_id
        raise ManifestGateError(f"G5 manifest gate failed: {exc}") from exc
    after_files = {path: candidate.files[path] for path in paths}
    manifest = manifest_for_patch(
        candidate_id,
        CandidateLabel.runtime,
        patch,
        before_files=before_files,
        after_files=after_files,
    )
    candidate.candidate_id = candidate_id
    candidate.manifest = manifest
    assert_manifest_gate(
        manifest,
        patch,
        before_files=before_files,
        after_files=after_files,
        repo_root=repo_root,
        treeish=parent_sha,
    )
    return manifest


__all__ = [
    "Candidate",
    "files_of",
    "materialize_candidate_patch",
    "prepare_candidate_manifest",
]
