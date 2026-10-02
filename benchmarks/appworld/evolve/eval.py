"""AppWorld eval + candidate representation for the in-package evolution.

A candidate is the driver's edited harness files (``Candidate.files`` =
``{repo-rel path: bytes}``); ``make_git_commit_apply_fn`` (with ``files_of``)
turns them into a real child commit. Eval then checks that commit out into an
ephemeral worktree and runs ``batch.py`` with ``cwd=worktree`` so the candidate's
committed harness is what imports — no writing into the live repo (the
zero-contamination replacement for RealPathSync).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import unified_diff
from pathlib import Path

from benchmarks.appworld.evolve import adapter as aw_adapter
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
    """One candidate harness edit produced by the bash-editor design step.

    ``has_beacon`` marks a code edit carrying an ``activation_beacon`` call —
    only those get Gate-b per-task attribution (prompt/config edits have no
    code execution point and fail open). ``activation_spec`` is the driver's
    optional self-declared trigger predicate, consumed by the zero-hit
    preflight.
    """

    files: dict[str, bytes]  # 各编辑后仓库相对路径的完整新字节。
    why: str  # 此候选针对的 WHY。
    focused_task_ids: list[str] = field(default_factory=list)  # WHY 的证据子集。
    summary: str = ""  # 编辑器对改动内容的单行摘要。
    deletions: list[str] = field(default_factory=list)  # 已删除的仓库相对路径。
    has_beacon: bool = False
    activation_spec: dict | None = None
    label: CandidateLabel | None = None
    before_files: dict[str, bytes | None] = field(default_factory=dict, repr=False)
    applied_patch: AppliedPatch | None = field(default=None, repr=False)
    manifest: CandidateManifest | None = None
    candidate_id: str = ""


def _patch_where(label: CandidateLabel, paths: list[str]) -> PatchWhere:
    if label is CandidateLabel.skill:
        return PatchWhere.skill
    if label is CandidateLabel.prompt:
        if all(path.startswith("codeflow/templates/") for path in paths):
            return PatchWhere.system_prompt_template
        return PatchWhere.task_wrapper_prompt
    if label in {CandidateLabel.model_profile, CandidateLabel.route}:
        return PatchWhere.config
    if label is CandidateLabel.policy:
        if all(Path(path).suffix.lower() in {".json", ".toml", ".yaml", ".yml"} for path in paths):
            return PatchWhere.config
        return PatchWhere.hook_modify
    if paths and all(path == "benchmarks/appworld/agent_cli.py" for path in paths):
        return PatchWhere.loop_override
    return PatchWhere.tool_override


def _candidate_label(paths: list[str]) -> CandidateLabel:
    if paths and all(path.startswith("codeflow/memory_engine/skills/") for path in paths):
        return CandidateLabel.skill
    if paths and all(path.startswith("codeflow/templates/") for path in paths):
        return CandidateLabel.prompt
    return CandidateLabel.runtime


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


def materialize_candidate_patch(
    candidate: Candidate,
    before_files: dict[str, bytes | None],
) -> AppliedPatch:
    files = set(candidate.files)
    deletions = set(candidate.deletions)
    overlap = sorted(files & deletions)
    if overlap:
        raise ValueError(f"candidate paths cannot be both written and deleted: {overlap}")
    paths = sorted(files | deletions)
    if not paths:
        raise ValueError("candidate has no target files")
    declared_label = getattr(candidate, "label", None)
    label = CandidateLabel(declared_label) if declared_label is not None else _candidate_label(paths)
    where = _patch_where(label, paths)
    try:
        why = PatchWhy(candidate.why)
        why_extra = None
    except ValueError:
        why = PatchWhy.other
        why_extra = candidate.why or "unspecified"
    patch = AppliedPatch(
        patch_where=where,
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
    candidate.label = label
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
    files = set(candidate.files)
    deletions = set(getattr(candidate, "deletions", ()))
    overlap = sorted(files & deletions)
    if overlap:
        candidate.candidate_id = candidate_id
        raise ManifestGateError(
            f"G5 manifest gate failed: candidate paths cannot be both written and deleted: {overlap}"
        )
    paths = sorted(files | deletions)
    before_files: dict[str, bytes | None] = {}
    for path in paths:
        try:
            before_files[path] = git_ops.read_file_at(repo_root, parent_sha, path)
        except GitOpError:
            before_files[path] = None
    try:
        patch = materialize_candidate_patch(candidate, before_files)
    except ValueError as exc:
        candidate.candidate_id = candidate_id
        raise ManifestGateError(f"G5 manifest gate failed: {exc}") from exc
    after_files = {path: candidate.files[path] if path in candidate.files else None for path in paths}
    manifest = manifest_for_patch(
        candidate_id,
        candidate.label,
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


def files_of(cand: Candidate) -> dict[str, bytes]:
    """Extract the edited file bytes for ``make_git_commit_apply_fn``."""
    return cand.files


def deletions_of(cand: Candidate) -> list[str]:
    """Extract the deleted paths for ``make_git_commit_apply_fn``."""
    return cand.deletions


def make_appworld_eval_fn(aw: "aw_adapter.AppWorldConfig", repo_root: str | Path):
    """Eval a node by checking its commit out into a worktree and running
    ``batch.py`` there (``cwd=worktree``). No activation env, no live-repo writes.
    """
    root = Path(repo_root)

    def eval_fn(node, task_ids, k, job_name, *, split="train"):
        cfg = aw
        if split != aw.split:
            from dataclasses import replace

            cfg = replace(aw, split=split)
        with git_ops.worktree_at(root, node.git_commit_sha) as wt:
            return aw_adapter.run_eval(cfg, K=k, experiment=job_name, task_ids=task_ids, cwd=wt)

    return eval_fn


__all__ = [
    "Candidate",
    "deletions_of",
    "files_of",
    "make_appworld_eval_fn",
    "materialize_candidate_patch",
    "prepare_candidate_manifest",
]
