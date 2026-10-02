from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.appworld.evolve.eval import (
    Candidate,
    files_of,
    materialize_candidate_patch,
    prepare_candidate_manifest,
)
from codeflow.evolver.applier.path_guard import (
    ImmutablePathError,
    assert_patch_allowed,
    check_patch_paths,
)
from codeflow.evolver.candidate_manifest import (
    LABEL_POLICIES,
    ActivationPolicy,
    CandidateLabel,
    CandidateManifest,
    ManifestGateError,
    assert_manifest_gate,
    evaluate_manifest_gate,
    manifest_for_patch,
)
from codeflow.evolver.judge.schema import PatchWhere, PatchWhy
from codeflow.evolver.orchestrator.production import make_git_commit_apply_fn
from codeflow.evolver.tree.node import AppliedPatch, HarnessNode, PatchComponent
from codeflow.evolver.tree.store import EvolverTreeStore


def _patch(where: PatchWhere, target: str) -> AppliedPatch:
    return AppliedPatch(
        patch_where=where,
        patch_why=PatchWhy.skill_gap_fill,
        components=[
            PatchComponent(
                component_id="comp_1",
                target_file=target,
                diff=f"--- a/{target}\n+++ b/{target}\n",
                rationale="fixture",
            )
        ],
        overall_reasoning="deterministic fixture",
    )


def _states(
    patch: AppliedPatch,
) -> tuple[dict[str, bytes | None], dict[str, bytes | None]]:
    targets = [component.target_file for component in patch.components]
    return (
        {target: b"before\n" for target in targets},
        {target: b"after\n" for target in targets},
    )


def _manifest(
    candidate_id: str,
    label: CandidateLabel | str,
    patch: AppliedPatch,
) -> CandidateManifest:
    before, after = _states(patch)
    return manifest_for_patch(
        candidate_id,
        label,
        patch,
        before_files=before,
        after_files=after,
    )


def _evaluate(
    manifest: CandidateManifest,
    patch: AppliedPatch,
):
    before, after = _states(patch)
    return evaluate_manifest_gate(
        manifest,
        patch,
        before_files=before,
        after_files=after,
    )


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _candidate_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "subject"
    (repo / "benchmarks/appworld").mkdir(parents=True)
    (repo / "benchmarks/appworld/agent_cli.py").write_text("VALUE = 1\n")
    (repo / "codeflow/evolver").mkdir(parents=True)
    (repo / "codeflow/evolver/kernel.py").write_text("SEALED = True\n")
    (repo / "mutable/real").mkdir(parents=True)
    (repo / "mutable/link").symlink_to("real", target_is_directory=True)
    _git(repo, "init", "-q")
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "CodeFlow Test",
        "GIT_AUTHOR_EMAIL": "codeflow-test@example.invalid",
        "GIT_COMMITTER_NAME": "CodeFlow Test",
        "GIT_COMMITTER_EMAIL": "codeflow-test@example.invalid",
    }
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True, env=env)
    return repo, _git(repo, "rev-parse", "HEAD")


@pytest.mark.parametrize("label", list(CandidateLabel))
def test_every_public_label_has_an_explicit_policy(label: CandidateLabel) -> None:
    policy = LABEL_POLICIES[label]
    assert policy.patch_where
    assert policy.activation_policy in ActivationPolicy
    if policy.supported:
        assert policy.fixture
        assert policy.mutable_paths
        assert policy.evaluator
    else:
        assert policy.unsupported_reason


def test_manifest_is_a_deterministic_view_over_applied_patch() -> None:
    patch = _patch(PatchWhere.loop_override, "benchmarks/appworld/agent_cli.py")
    manifest = _manifest("candidate-1", CandidateLabel.runtime, patch)

    assert _manifest("candidate-1", "runtime", patch).to_json() == manifest.to_json()
    assert CandidateManifest.from_json(manifest.to_json()) == manifest
    assert manifest.patch_where == patch.patch_where
    assert manifest.target_files == (patch.components[0].target_file,)
    assert _evaluate(manifest, patch).passed


def test_manifest_gate_rejects_tampered_patch_metadata() -> None:
    patch = _patch(PatchWhere.loop_override, "benchmarks/appworld/agent_cli.py")
    manifest = _manifest("candidate-1", CandidateLabel.runtime, patch)
    tampered = replace(manifest, target_files=("codeflow/templates/AGENTS.md",))

    result = _evaluate(tampered, patch)

    assert result.gate == "G5"
    assert result.passed is False
    assert any("target_files" in reason for reason in result.reasons)
    before, after = _states(patch)
    with pytest.raises(ManifestGateError, match="G5"):
        assert_manifest_gate(
            tampered,
            patch,
            before_files=before,
            after_files=after,
        )


def test_manifest_gate_binds_raw_parent_and_result_bytes() -> None:
    patch = _patch(PatchWhere.loop_override, "benchmarks/appworld/agent_cli.py")
    before, after = _states(patch)
    manifest = manifest_for_patch(
        "candidate-1",
        CandidateLabel.runtime,
        patch,
        before_files=before,
        after_files=after,
    )
    changed_after = {patch.target_file: b"different\n"}

    result = evaluate_manifest_gate(
        manifest,
        patch,
        before_files=before,
        after_files=changed_after,
    )

    assert result.passed is False
    assert any("after_sha256" in reason for reason in result.reasons)
    assert any("patch_digest" in reason for reason in result.reasons)


def test_supported_label_requires_executable_evaluator_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codeflow.evolver import candidate_evidence

    patch = _patch(PatchWhere.loop_override, "benchmarks/appworld/agent_cli.py")
    manifest = _manifest("candidate-1", CandidateLabel.runtime, patch)
    monkeypatch.delitem(
        candidate_evidence.EVALUATOR_BINDINGS,
        "appworld_focused_fisher_v1",
    )

    result = _evaluate(manifest, patch)

    assert result.passed is False
    assert any("executable evaluator binding" in reason for reason in result.reasons)


@pytest.mark.parametrize(
    ("label", "where", "target"),
    [
        (
            CandidateLabel.skill,
            PatchWhere.skill,
            "codeflow/memory_engine/skills/example/SKILL.md",
        ),
        (
            CandidateLabel.prompt,
            PatchWhere.system_prompt_template,
            "codeflow/templates/AGENTS.md",
        ),
        (CandidateLabel.policy, PatchWhere.config, "config/policy.yaml"),
        (CandidateLabel.model_profile, PatchWhere.config, "config/model_profile.json"),
        (CandidateLabel.route, PatchWhere.config, "config/route.toml"),
    ],
)
def test_labels_without_safe_evaluators_fail_g5(
    label: CandidateLabel,
    where: PatchWhere,
    target: str,
) -> None:
    patch = _patch(where, target)
    result = _evaluate(_manifest("candidate-1", label, patch), patch)

    assert result.passed is False
    assert any("evaluator" in reason.lower() for reason in result.reasons)


@pytest.mark.parametrize("label", [CandidateLabel.model_profile, CandidateLabel.route])
def test_model_profile_and_route_are_config_only_and_never_weights(label: CandidateLabel) -> None:
    code_patch = _patch(PatchWhere.config, "codeflow/routing/profiles.py")
    weight_patch = _patch(PatchWhere.config, "models/adapter.safetensors")

    code_result = _evaluate(_manifest("code", label, code_patch), code_patch)
    weight_result = _evaluate(_manifest("weights", label, weight_patch), weight_patch)

    assert any("configuration files only" in reason for reason in code_result.reasons)
    assert any("model weights" in reason for reason in weight_result.reasons)


def test_runtime_requires_human_review_and_exact_mutable_surface() -> None:
    patch = _patch(PatchWhere.loop_override, "benchmarks/appworld/agent_cli.py")
    manifest = _manifest("runtime-1", CandidateLabel.runtime, patch)
    outside = _patch(PatchWhere.loop_override, "codeflow/agent/loop/recovery.py")

    assert manifest.activation_policy == ActivationPolicy.human_review
    assert _evaluate(manifest, patch).passed
    result = _evaluate(
        _manifest("runtime-2", CandidateLabel.runtime, outside),
        outside,
    )
    assert result.passed is False
    assert any("mutable allowlist" in reason for reason in result.reasons)


@pytest.mark.parametrize(
    "target",
    [
        "../codeflow/templates/AGENTS.md",
        "codeflow/templates/../../pyproject.toml",
        "/tmp/candidate.py",
        r"C:\tmp\candidate.py",
        r"\\server\share\candidate.py",
        "codeflow//templates/AGENTS.md",
    ],
)
def test_path_guard_rejects_traversal_and_absolute_paths(target: str) -> None:
    assert check_patch_paths([target]) == [target]
    with pytest.raises(ImmutablePathError):
        assert_patch_allowed([target])


def test_path_guard_rejects_symlink_components(tmp_path: Path) -> None:
    (tmp_path / "codeflow").mkdir()
    (tmp_path / "outside").mkdir()
    (tmp_path / "codeflow" / "templates").symlink_to(tmp_path / "outside", target_is_directory=True)
    target = "codeflow/templates/AGENTS.md"

    assert check_patch_paths([target], repo_root=tmp_path) == [target]
    with pytest.raises(ImmutablePathError):
        assert_patch_allowed([target], repo_root=tmp_path)


def test_path_guard_reads_symlinks_from_parent_git_tree(tmp_path: Path) -> None:
    repo, parent_sha = _candidate_repo(tmp_path)
    target = "mutable/link/candidate.py"
    (repo / "mutable/link").unlink()
    (repo / "mutable/link").mkdir()

    assert check_patch_paths([target], repo_root=repo) == []
    assert check_patch_paths(
        [target],
        repo_root=repo,
        treeish=parent_sha,
    ) == [target]
    with pytest.raises(ImmutablePathError):
        assert_patch_allowed(
            [target],
            repo_root=repo,
            treeish=parent_sha,
        )


def test_tree_store_rejects_parent_tree_symlink_before_child_node(
    tmp_path: Path,
) -> None:
    repo, parent_sha = _candidate_repo(tmp_path)
    store = EvolverTreeStore(repo, tmp_path / "nodes")
    root = HarnessNode(
        node_id="C0",
        parent_id=None,
        git_commit_sha=parent_sha,
        git_branch="",
        created_at=HarnessNode.utc_now(),
        created_at_iter=0,
    )
    store.save_node(root)
    target = "mutable/link/candidate.py"
    patch = AppliedPatch(
        patch_where=PatchWhere.tool_override,
        patch_why=PatchWhy.tool_clarity,
        components=[
            PatchComponent(
                component_id="comp_1",
                target_file=target,
                diff=(f"--- /dev/null\n+++ b/{target}\n@@ -0,0 +1 @@\n+VALUE = 1\n"),
                rationale="fixture",
            )
        ],
        overall_reasoning="fixture",
    )

    with pytest.raises(ImmutablePathError):
        store.create_child_node(
            "C0",
            patch,
            1,
            "fixture",
        )

    assert sorted(path.name for path in (tmp_path / "nodes").iterdir()) == ["C0.json"]


@pytest.mark.parametrize(
    "target",
    [
        "codeflow/evolver/candidate_manifest.py",
        "codeflow/eval_engine/engine.py",
        "tests/test_evolver_candidate_manifest.py",
        "benchmarks/appworld/evolve/grade.py",
        "benchmarks/appworld/batch.py",
        "CODEFLOW/EVOLVER/candidate_manifest.py",
    ],
)
def test_path_guard_keeps_kernel_and_evidence_surfaces_immutable(target: str) -> None:
    assert check_patch_paths([target]) == [target]


def test_appworld_candidate_gets_g5_manifest_before_child_commit(tmp_path: Path) -> None:
    repo, parent_sha = _candidate_repo(tmp_path)
    candidate = Candidate(
        files={"benchmarks/appworld/agent_cli.py": b"VALUE = 2\n"},
        why="runtime_fixture",
        summary="change the fixture runtime",
    )

    def prepare(node_id: str, _parent_id: str, base_sha: str, value: Candidate) -> None:
        prepare_candidate_manifest(node_id, base_sha, value, repo_root=repo)

    apply_candidate = make_git_commit_apply_fn(
        repo,
        files_of,
        root_node_id="C0",
        base_sha=parent_sha,
        node_id_salt="fixture",
        before_commit=prepare,
        applied_patch_of=lambda value: value.applied_patch,
    )
    node = apply_candidate("C0", candidate, 1)

    assert candidate.manifest is not None
    assert candidate.manifest.candidate_id == node.node_id
    assert candidate.manifest.label is CandidateLabel.runtime
    assert candidate.manifest.activation_policy is ActivationPolicy.human_review
    assert node.patch is candidate.applied_patch
    assert _git(repo, "show", f"{node.git_commit_sha}:benchmarks/appworld/agent_cli.py") == "VALUE = 2"
    assert _git(repo, "rev-parse", "HEAD") == parent_sha
    assert (repo / "benchmarks/appworld/agent_cli.py").read_text() == "VALUE = 1\n"


def test_precommit_rematerializes_stale_candidate_from_parent_bytes(tmp_path: Path) -> None:
    repo, parent_sha = _candidate_repo(tmp_path)
    target = "benchmarks/appworld/agent_cli.py"
    candidate = Candidate(
        files={target: b"VALUE = 2\n"},
        why="runtime_fixture",
    )
    stale_patch = materialize_candidate_patch(
        candidate,
        {target: b"stale parent\n"},
    )
    candidate.files[target] = b"VALUE = 3\n"

    manifest = prepare_candidate_manifest(
        "candidate-1",
        parent_sha,
        candidate,
        repo_root=repo,
    )

    assert candidate.applied_patch is not stale_patch
    assert "-VALUE = 1" in candidate.applied_patch.diff
    assert "+VALUE = 3" in candidate.applied_patch.diff
    assert manifest.before_sha256 == (hashlib.sha256(b"VALUE = 1\n").hexdigest(),)
    assert manifest.after_sha256 == (hashlib.sha256(b"VALUE = 3\n").hexdigest(),)


def test_g5_rejects_write_delete_overlap(tmp_path: Path) -> None:
    repo, parent_sha = _candidate_repo(tmp_path)
    target = "benchmarks/appworld/agent_cli.py"
    candidate = Candidate(
        files={target: b"VALUE = 2\n"},
        deletions=[target],
        why="runtime_fixture",
    )

    with pytest.raises(ManifestGateError, match="both written and deleted"):
        prepare_candidate_manifest(
            "candidate-1",
            parent_sha,
            candidate,
            repo_root=repo,
        )


def test_g5_rejects_non_utf8_candidate_content(tmp_path: Path) -> None:
    repo, parent_sha = _candidate_repo(tmp_path)
    candidate = Candidate(
        files={"benchmarks/appworld/agent_cli.py": b"\xff\xfe"},
        why="runtime_fixture",
    )

    with pytest.raises(ManifestGateError, match="UTF-8"):
        prepare_candidate_manifest(
            "candidate-1",
            parent_sha,
            candidate,
            repo_root=repo,
        )


def test_g5_rejection_creates_no_candidate_commit(tmp_path: Path) -> None:
    repo, parent_sha = _candidate_repo(tmp_path)
    candidate = Candidate(
        files={"codeflow/evolver/kernel.py": b"SEALED = False\n"},
        why="unsafe_fixture",
        summary="edit the Evolver kernel",
    )

    def prepare(node_id: str, _parent_id: str, base_sha: str, value: Candidate) -> None:
        prepare_candidate_manifest(node_id, base_sha, value, repo_root=repo)

    apply_candidate = make_git_commit_apply_fn(
        repo,
        files_of,
        root_node_id="C0",
        base_sha=parent_sha,
        node_id_salt="fixture",
        before_commit=prepare,
        applied_patch_of=lambda value: value.applied_patch,
    )
    commit_count = _git(repo, "rev-list", "--all", "--count")

    with pytest.raises(ManifestGateError, match="G5"):
        apply_candidate("C0", candidate, 1)

    assert candidate.candidate_id == "v1-c1-fixture"
    assert _git(repo, "rev-list", "--all", "--count") == commit_count
    assert _git(repo, "for-each-ref", "--format=%(refname)", "refs/evolver") == ""
