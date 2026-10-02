from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from codeflow.evolver.activation.artifacts import (
    ActivationState,
    EvidenceDecision,
    EvidenceOutcome,
    _canonical_json,
    _identity_digest,
    create_activation_artifacts,
    load_activation_record,
    set_activation_state,
    verify_activation_artifacts,
)
from codeflow.evolver.activation.summary import (
    build_evolution_summary,
    write_evolution_summary,
)
from codeflow.evolver.candidate_evidence import AcceptedRuntimeEvidence
from codeflow.evolver.candidate_manifest import LABEL_POLICIES, CandidateLabel
from codeflow.evolver.orchestrator.scoring import TaskEval

PARENT_SHA = "1" * 40
CANDIDATE_SHA = "2" * 40

_WHERE_BY_LABEL = {
    CandidateLabel.skill: "skill",
    CandidateLabel.prompt: "task_wrapper_prompt",
    CandidateLabel.policy: "config",
    CandidateLabel.runtime: "loop_override",
    CandidateLabel.model_profile: "config",
    CandidateLabel.route: "config",
}
_TARGET_BY_LABEL = {
    CandidateLabel.skill: "codeflow/memory_engine/skills/example/SKILL.md",
    CandidateLabel.prompt: "codeflow/templates/AGENTS.md",
    CandidateLabel.policy: "config/policy.yaml",
    CandidateLabel.runtime: "benchmarks/appworld/agent_cli.py",
    CandidateLabel.model_profile: "config/model_profile.json",
    CandidateLabel.route: "config/route.toml",
}


def _manifest(
    candidate_id: str,
    *,
    label: CandidateLabel | str = CandidateLabel.runtime,
    target_files: list[str] | None = None,
    before: dict[str, str | bytes | None] | None = None,
    after: dict[str, str | bytes | None] | None = None,
) -> dict:
    parsed_label = CandidateLabel(label)
    policy = LABEL_POLICIES[parsed_label]
    targets = target_files if target_files is not None else [_TARGET_BY_LABEL[parsed_label]]
    if before is None:
        before = {path: "VALUE = 1\n" if parsed_label is CandidateLabel.runtime else "before" for path in targets}
    if after is None:
        after = {path: "VALUE = 2\n" if parsed_label is CandidateLabel.runtime else "after" for path in targets}

    def digest(value: str | bytes | None) -> str | None:
        if value is None:
            return None
        content = value.encode() if isinstance(value, str) else value
        return hashlib.sha256(content).hexdigest()

    return {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "label": parsed_label.value,
        "patch_where": _WHERE_BY_LABEL[parsed_label],
        "target_files": targets,
        "before_sha256": [digest(before[path]) for path in targets],
        "after_sha256": [digest(after[path]) for path in targets],
        "patch_digest": "a" * 64,
        "fixture": policy.fixture,
        "evaluator": policy.evaluator,
        "activation_policy": policy.activation_policy.value,
    }


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _accepted(
    *,
    task_ids: tuple[str, ...] = ("task-1",),
    expected_attempts: int = 1,
) -> AcceptedRuntimeEvidence:
    return AcceptedRuntimeEvidence(
        schema_version=1,
        evaluator="appworld_focused_fisher_v1",
        task_ids=task_ids,
        expected_attempts=expected_attempts,
        eligible_tasks=task_ids,
        candidate_evals=tuple(
            TaskEval(task_id, passes=expected_attempts, attempts=expected_attempts) for task_id in task_ids
        ),
        control_evals=tuple(TaskEval(task_id, passes=0, attempts=expected_attempts) for task_id in task_ids),
    )


def _runtime_snapshots() -> tuple[dict[str, str], dict[str, str]]:
    target = _TARGET_BY_LABEL[CandidateLabel.runtime]
    return {target: "VALUE = 1\n"}, {target: "VALUE = 2\n"}


def _runtime_repo(
    path: Path,
    before: dict[str, str | bytes | None],
    after: dict[str, str | bytes | None],
) -> tuple[Path, str, str]:
    path.mkdir()
    _git(path, "init")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test User")

    def materialize(files: dict[str, str | bytes | None]) -> None:
        for relative, content in files.items():
            target = path / relative
            if content is None:
                if target.exists():
                    target.unlink()
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                target.write_bytes(content)
            else:
                target.write_text(content)

    materialize(before)
    _git(path, "add", "-A")
    _git(path, "commit", "-m", "parent")
    parent_sha = _git(path, "rev-parse", "HEAD")
    materialize(after)
    _git(path, "add", "-A")
    _git(path, "commit", "-m", "candidate")
    candidate_sha = _git(path, "rev-parse", "HEAD")
    _git(path, "checkout", "--detach", parent_sha)
    return path, parent_sha, candidate_sha


def _create_runtime(
    work_dir: Path,
    *,
    candidate_id: str = "runtime-001",
) -> Path:
    before, after = _runtime_snapshots()
    repo, parent_sha, candidate_sha = _runtime_repo(
        work_dir.parent / f"{work_dir.name}-{candidate_id}-subject",
        before,
        after,
    )
    return create_activation_artifacts(
        work_dir,
        candidate_id=candidate_id,
        label="runtime",
        manifest=_manifest(candidate_id),
        evidence=_accepted(),
        before=before,
        after=after,
        parent_sha=parent_sha,
        candidate_sha=candidate_sha,
        repo_root=repo,
    )


def _create_rejected_prompt(
    work_dir: Path,
    *,
    candidate_id: str,
) -> Path:
    target = _TARGET_BY_LABEL[CandidateLabel.prompt]
    return create_activation_artifacts(
        work_dir,
        candidate_id=candidate_id,
        label="prompt",
        manifest=_manifest(candidate_id, label="prompt"),
        evidence=EvidenceDecision(outcome="rejected", gate_passed=False),
        before={target: "before"},
        after={target: "after"},
        require_human=False,
    )


def test_runtime_artifacts_default_to_pending_human_without_moving_checkout(
    tmp_path,
):
    before, after = _runtime_snapshots()
    repo, parent_sha, candidate_sha = _runtime_repo(tmp_path / "repo", before, after)
    before_head = _git(repo, "rev-parse", "HEAD")
    before_status = _git(repo, "status", "--porcelain")

    artifact_dir = create_activation_artifacts(
        tmp_path / "run",
        candidate_id="runtime-001",
        label="runtime",
        manifest=_manifest("runtime-001"),
        evidence=_accepted(),
        before=before,
        after=after,
        parent_sha=parent_sha,
        candidate_sha=candidate_sha,
        repo_root=repo,
    )

    assert sorted(path.name for path in artifact_dir.iterdir()) == [
        "activation.json",
        "after.json",
        "before.json",
        "candidate_manifest.json",
        "evidence.json",
        "rollback.json",
    ]
    assert (artifact_dir / "before.json").read_bytes() == (artifact_dir / "rollback.json").read_bytes()
    record = load_activation_record(artifact_dir)
    assert record["state"] == ActivationState.pending_human.value
    assert record["requires_human"] is True
    assert record["parent_sha"] == parent_sha
    assert record["candidate_sha"] == candidate_sha
    assert _git(repo, "rev-parse", "HEAD") == before_head
    assert _git(repo, "status", "--porcelain") == before_status
    assert (repo / _TARGET_BY_LABEL[CandidateLabel.runtime]).read_text() == "VALUE = 1\n"


def test_creation_rejects_symlinked_activation_candidate_directory(tmp_path):
    work_dir = tmp_path / "run"
    activation_root = work_dir / "activation"
    outside = tmp_path / "outside"
    activation_root.mkdir(parents=True)
    outside.mkdir()
    (activation_root / "runtime-001").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        _create_rejected_prompt(work_dir, candidate_id="runtime-001")

    assert list(outside.iterdir()) == []


def test_creation_rejects_symlinked_activation_root(tmp_path):
    work_dir = tmp_path / "run"
    outside = tmp_path / "outside"
    work_dir.mkdir()
    outside.mkdir()
    (work_dir / "activation").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="activation root must not be a symlink"):
        _create_rejected_prompt(work_dir, candidate_id="runtime-001")

    assert list(outside.iterdir()) == []


def test_verification_rejects_symlinked_activation_candidate_directory(tmp_path):
    candidate_id = "runtime-symlinked-read"
    outside_artifact = _create_rejected_prompt(
        tmp_path / "outside-run",
        candidate_id=candidate_id,
    )
    activation_root = tmp_path / "run" / "activation"
    activation_root.mkdir(parents=True)
    artifact_dir = activation_root / candidate_id
    artifact_dir.symlink_to(outside_artifact, target_is_directory=True)

    with pytest.raises(ValueError, match="candidate directory must not be a symlink"):
        load_activation_record(artifact_dir)


def test_verification_rejects_symlinked_activation_payload(tmp_path):
    artifact_dir = _create_rejected_prompt(
        tmp_path / "run",
        candidate_id="prompt-symlinked-payload",
    )
    payload = artifact_dir / "after.json"
    outside = tmp_path / "outside-after.json"
    outside.write_bytes(payload.read_bytes())
    payload.unlink()
    payload.symlink_to(outside)

    with pytest.raises(ValueError, match="artifact file must not be a symlink"):
        verify_activation_artifacts(artifact_dir)


@pytest.mark.parametrize("outcome", ["rejected", "failed", "inconclusive"])
def test_non_accepted_evidence_is_ineligible_for_activation(tmp_path, outcome):
    candidate_id = f"candidate-{outcome}"
    target = _TARGET_BY_LABEL[CandidateLabel.prompt]
    artifact_dir = create_activation_artifacts(
        tmp_path / outcome,
        candidate_id=candidate_id,
        label="prompt",
        manifest=_manifest(candidate_id, label="prompt"),
        evidence=EvidenceDecision(
            outcome=outcome,
            gate_passed=False,
            failure_class="provider" if outcome == "failed" else None,
        ),
        before={target: "before"},
        after={target: "after"},
        require_human=False,
    )

    assert load_activation_record(artifact_dir)["state"] == "ineligible"


@pytest.mark.parametrize(
    "label",
    ["skill", "prompt", "policy", "model_profile", "route"],
)
def test_unsupported_label_cannot_carry_accepted_evidence(tmp_path, label):
    candidate_id = f"{label}-001"
    target = _TARGET_BY_LABEL[CandidateLabel(label)]
    with pytest.raises(ValueError, match="accepted evidence is forbidden"):
        create_activation_artifacts(
            tmp_path / label,
            candidate_id=candidate_id,
            label=label,
            manifest=_manifest(candidate_id, label=label),
            evidence=_accepted(),
            before={target: "before"},
            after={target: "after"},
            parent_sha=PARENT_SHA,
            candidate_sha=CANDIDATE_SHA,
        )


def test_manifest_must_be_complete_and_match_canonical_policy(tmp_path):
    before, after = _runtime_snapshots()
    with pytest.raises(ValueError, match="fields must be exactly"):
        create_activation_artifacts(
            tmp_path / "missing",
            candidate_id="runtime-001",
            label="runtime",
            manifest={"candidate_id": "runtime-001", "label": "runtime"},
            evidence=_accepted(),
            before=before,
            after=after,
            parent_sha=PARENT_SHA,
            candidate_sha=CANDIDATE_SHA,
        )

    manifest = _manifest("runtime-001")
    manifest["evaluator"] = "self_reported"
    with pytest.raises(ValueError, match="canonical label policy"):
        create_activation_artifacts(
            tmp_path / "policy-drift",
            candidate_id="runtime-001",
            label="runtime",
            manifest=manifest,
            evidence=_accepted(),
            before=before,
            after=after,
            parent_sha=PARENT_SHA,
            candidate_sha=CANDIDATE_SHA,
        )


def test_manifest_content_digests_must_match_snapshots(tmp_path):
    before, after = _runtime_snapshots()
    manifest = _manifest("runtime-001")
    manifest["after_sha256"] = ["f" * 64]

    with pytest.raises(ValueError, match="after_sha256.*after snapshot"):
        create_activation_artifacts(
            tmp_path / "digest-drift",
            candidate_id="runtime-001",
            label="runtime",
            manifest=manifest,
            evidence=_accepted(),
            before=before,
            after=after,
            parent_sha=PARENT_SHA,
            candidate_sha=CANDIDATE_SHA,
        )


@pytest.mark.parametrize(
    "decision_kwargs",
    [
        {"outcome": "accepted", "gate_passed": False},
        {"outcome": "rejected", "gate_passed": True},
        {"outcome": "failed", "gate_passed": True},
        {"outcome": "inconclusive", "gate_passed": True},
        {"outcome": "accepted", "gate_passed": True, "regression": True},
        {
            "outcome": "accepted",
            "gate_passed": True,
            "failure_class": "provider",
        },
        {
            "outcome": "accepted",
            "gate_passed": True,
            "failure_class": "infrastructure",
        },
    ],
)
def test_invalid_improvement_claims_are_rejected(decision_kwargs):
    with pytest.raises(ValueError):
        EvidenceDecision(**decision_kwargs)


@pytest.mark.parametrize("outcome", ["rejected", "inconclusive"])
@pytest.mark.parametrize("failure_class", ["provider", "infrastructure"])
def test_failure_provenance_requires_failed_outcome(
    outcome,
    failure_class,
):
    with pytest.raises(ValueError, match="requires a failed outcome"):
        EvidenceDecision(
            outcome=outcome,
            gate_passed=False,
            failure_class=failure_class,
        )


@pytest.mark.parametrize("failure_class", ["provider", "infrastructure"])
def test_failed_evidence_preserves_failure_provenance(failure_class):
    decision = EvidenceDecision(
        outcome="failed",
        gate_passed=False,
        failure_class=failure_class,
    )

    assert decision.outcome is EvidenceOutcome.failed
    assert decision.failure_class == failure_class


@pytest.mark.parametrize("evidence", ["accepted", EvidenceOutcome.accepted])
def test_scalar_accepted_evidence_is_rejected(tmp_path, evidence):
    before, after = _runtime_snapshots()

    with pytest.raises(ValueError, match="canonical candidate and control measurements"):
        create_activation_artifacts(
            tmp_path / "run",
            candidate_id="runtime-001",
            label="runtime",
            manifest=_manifest("runtime-001"),
            evidence=evidence,
            before=before,
            after=after,
            parent_sha=PARENT_SHA,
            candidate_sha=CANDIDATE_SHA,
        )


def test_raw_structured_accepted_decision_is_rejected(tmp_path):
    before, after = _runtime_snapshots()

    with pytest.raises(ValueError, match="canonical candidate and control measurements"):
        create_activation_artifacts(
            tmp_path / "run",
            candidate_id="runtime-001",
            label="runtime",
            manifest=_manifest("runtime-001"),
            evidence=EvidenceDecision(outcome="accepted", gate_passed=True),
            before=before,
            after=after,
            parent_sha=PARENT_SHA,
            candidate_sha=CANDIDATE_SHA,
        )


@pytest.mark.parametrize("field", ["gate_passed", "regression"])
def test_evidence_boolean_fields_reject_string_coercion(tmp_path, field):
    before, after = _runtime_snapshots()
    evidence = {
        "outcome": "accepted",
        "gate_passed": True,
        "reason": "",
        "regression": False,
        "failure_class": None,
    }
    evidence[field] = "false"

    with pytest.raises(ValueError, match="must be a boolean"):
        create_activation_artifacts(
            tmp_path / "run",
            candidate_id="runtime-001",
            label="runtime",
            manifest=_manifest("runtime-001"),
            evidence=evidence,
            before=before,
            after=after,
            parent_sha=PARENT_SHA,
            candidate_sha=CANDIDATE_SHA,
        )


@pytest.mark.parametrize(
    ("parent_sha", "candidate_sha"),
    [
        (None, CANDIDATE_SHA),
        (PARENT_SHA, None),
        ("short", CANDIDATE_SHA),
        (PARENT_SHA, PARENT_SHA),
    ],
)
def test_accepted_evidence_requires_distinct_commit_identities(
    tmp_path,
    parent_sha,
    candidate_sha,
):
    before, after = _runtime_snapshots()
    with pytest.raises(ValueError, match="sha|commit|together"):
        create_activation_artifacts(
            tmp_path / "run",
            candidate_id="runtime-001",
            label="runtime",
            manifest=_manifest("runtime-001"),
            evidence=_accepted(),
            before=before,
            after=after,
            parent_sha=parent_sha,
            candidate_sha=candidate_sha,
        )


def test_accepted_evidence_rejects_invented_commit_ids(tmp_path):
    before, after = _runtime_snapshots()
    repo, _, _ = _runtime_repo(tmp_path / "subject", before, after)

    with pytest.raises(ValueError, match="parent_sha is not a commit"):
        create_activation_artifacts(
            tmp_path / "run",
            candidate_id="runtime-001",
            label="runtime",
            manifest=_manifest("runtime-001"),
            evidence=_accepted(),
            before=before,
            after=after,
            parent_sha=PARENT_SHA,
            candidate_sha=CANDIDATE_SHA,
            repo_root=repo,
        )


def test_accepted_evidence_requires_direct_parent_child_commits(tmp_path):
    before, after = _runtime_snapshots()
    repo, parent_sha, child_sha = _runtime_repo(tmp_path / "subject", before, after)
    target = _TARGET_BY_LABEL[CandidateLabel.runtime]
    _git(repo, "checkout", "--detach", child_sha)
    (repo / target).write_text("VALUE = 3\n")
    _git(repo, "add", target)
    _git(repo, "commit", "-m", "grandchild")
    grandchild_sha = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "--detach", parent_sha)
    grandchild_after = {target: "VALUE = 3\n"}

    with pytest.raises(ValueError, match="not a direct child"):
        create_activation_artifacts(
            tmp_path / "run",
            candidate_id="runtime-001",
            label="runtime",
            manifest=_manifest("runtime-001", before=before, after=grandchild_after),
            evidence=_accepted(),
            before=before,
            after=grandchild_after,
            parent_sha=parent_sha,
            candidate_sha=grandchild_sha,
            repo_root=repo,
        )


def test_accepted_evidence_snapshots_must_match_commit_blobs(tmp_path):
    before, committed_after = _runtime_snapshots()
    repo, parent_sha, candidate_sha = _runtime_repo(
        tmp_path / "subject",
        before,
        committed_after,
    )
    target = _TARGET_BY_LABEL[CandidateLabel.runtime]
    claimed_after = {target: "VALUE = 3\n"}

    with pytest.raises(ValueError, match="after snapshot differs"):
        create_activation_artifacts(
            tmp_path / "run",
            candidate_id="runtime-001",
            label="runtime",
            manifest=_manifest("runtime-001", before=before, after=claimed_after),
            evidence=_accepted(),
            before=before,
            after=claimed_after,
            parent_sha=parent_sha,
            candidate_sha=candidate_sha,
            repo_root=repo,
        )


def test_runtime_state_transition_requires_human_and_preserves_rollback(
    tmp_path,
):
    artifact_dir = _create_runtime(tmp_path / "run")
    rollback = (artifact_dir / "rollback.json").read_bytes()

    with pytest.raises(ValueError, match="human_actor"):
        set_activation_state(artifact_dir, ActivationState.ready)
    with pytest.raises(ValueError, match="human_actor must be a string"):
        set_activation_state(
            artifact_dir,
            ActivationState.ready,
            human_actor=123,
        )

    ready = set_activation_state(
        artifact_dir,
        ActivationState.ready,
        human_actor="reviewer@example.com",
        reason="reviewed evidence",
    )
    activated = set_activation_state(artifact_dir, ActivationState.activated)
    rolled_back = set_activation_state(
        artifact_dir,
        ActivationState.rolled_back,
        reason="regression observed after activation",
    )

    assert ready["state"] == "ready"
    assert activated["state"] == "activated"
    assert rolled_back["state"] == "rolled_back"
    assert [item["to"] for item in rolled_back["state_history"]] == [
        "ready",
        "activated",
        "rolled_back",
    ]
    assert (artifact_dir / "rollback.json").read_bytes() == rollback


def test_resume_preserves_state_and_rejects_tampered_payload(tmp_path):
    before, after = _runtime_snapshots()
    repo, parent_sha, candidate_sha = _runtime_repo(tmp_path / "subject", before, after)
    kwargs = {
        "candidate_id": "runtime-001",
        "label": "runtime",
        "manifest": _manifest("runtime-001"),
        "evidence": _accepted(),
        "before": before,
        "after": after,
        "parent_sha": parent_sha,
        "candidate_sha": candidate_sha,
        "repo_root": repo,
    }
    artifact_dir = create_activation_artifacts(tmp_path / "run", **kwargs)
    set_activation_state(
        artifact_dir,
        ActivationState.ready,
        human_actor="reviewer@example.com",
    )

    resumed = create_activation_artifacts(tmp_path / "run", **kwargs)

    assert resumed == artifact_dir
    assert load_activation_record(resumed)["state"] == "ready"
    (artifact_dir / "rollback.json").write_text("{}\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_activation_artifacts(artifact_dir)
    with pytest.raises(ValueError, match="digest mismatch"):
        set_activation_state(artifact_dir, ActivationState.activated)


def test_digest_manifest_cannot_drop_required_artifacts(tmp_path):
    artifact_dir = _create_runtime(tmp_path / "run")
    for name in ("before.json", "after.json", "rollback.json"):
        (artifact_dir / name).unlink()
    record_path = artifact_dir / "activation.json"
    record = json.loads(record_path.read_text())
    record["artifact_digests"] = {"candidate_manifest.json": record["artifact_digests"]["candidate_manifest.json"]}
    record_path.write_bytes(_canonical_json(record))

    with pytest.raises(ValueError, match="must contain exactly"):
        verify_activation_artifacts(artifact_dir)
    with pytest.raises(ValueError, match="must contain exactly"):
        set_activation_state(artifact_dir, ActivationState.activated)


def test_activation_state_must_equal_replayed_history(tmp_path):
    artifact_dir = _create_runtime(tmp_path / "run")
    record_path = artifact_dir / "activation.json"
    record = json.loads(record_path.read_text())
    record["state"] = "ready"
    record_path.write_bytes(_canonical_json(record))

    with pytest.raises(ValueError, match="replayed state_history"):
        verify_activation_artifacts(artifact_dir)


def test_activation_history_must_be_contiguous_and_authorized(tmp_path):
    artifact_dir = _create_runtime(tmp_path / "run")
    record_path = artifact_dir / "activation.json"
    record = json.loads(record_path.read_text())
    record["state"] = "activated"
    record["state_history"] = [
        {
            "from": "ready",
            "to": "activated",
            "human_actor": None,
            "reason": "",
        }
    ]
    record_path.write_bytes(_canonical_json(record))

    with pytest.raises(ValueError, match="does not continue"):
        verify_activation_artifacts(artifact_dir)


def test_commit_identity_tamper_breaks_evidence_cross_binding(tmp_path):
    artifact_dir = _create_runtime(tmp_path / "run")
    record_path = artifact_dir / "activation.json"
    record = json.loads(record_path.read_text())
    record["candidate_sha"] = "3" * 40
    record["identity_digest"] = _identity_digest(record)
    record_path.write_bytes(_canonical_json(record))

    with pytest.raises(ValueError, match="does not match candidate"):
        verify_activation_artifacts(artifact_dir)


def test_artifact_json_is_canonical_across_input_order(tmp_path):
    targets = [
        "benchmarks/appworld/tool.py",
        "benchmarks/appworld/agent_cli.py",
    ]
    manifest = _manifest(
        "runtime-001",
        target_files=targets,
        before={targets[0]: "a", targets[1]: "b"},
        after={targets[0]: "A", targets[1]: "B"},
    )
    repo, parent_sha, candidate_sha = _runtime_repo(
        tmp_path / "subject",
        {targets[0]: "a", targets[1]: "b"},
        {targets[0]: "A", targets[1]: "B"},
    )
    first = create_activation_artifacts(
        tmp_path / "one",
        candidate_id="runtime-001",
        label="runtime",
        manifest=manifest,
        evidence=_accepted(),
        before={targets[1]: "b", targets[0]: "a"},
        after={targets[1]: "B", targets[0]: "A"},
        parent_sha=parent_sha,
        candidate_sha=candidate_sha,
        repo_root=repo,
    )
    second = create_activation_artifacts(
        tmp_path / "two",
        candidate_id="runtime-001",
        label="runtime",
        manifest=dict(reversed(list(manifest.items()))),
        evidence=_accepted(),
        before={targets[0]: "a", targets[1]: "b"},
        after={targets[0]: "A", targets[1]: "B"},
        parent_sha=parent_sha,
        candidate_sha=candidate_sha,
        repo_root=repo,
    )

    for name in sorted(path.name for path in first.iterdir()):
        assert (first / name).read_bytes() == (second / name).read_bytes()


@pytest.mark.parametrize(
    "path",
    [
        "../agent.py",
        "codeflow/../agent.py",
        "/tmp/agent.py",
        r"C:\tmp\agent.py",
        r"\\server\share\agent.py",
        "codeflow//agent.py",
    ],
)
def test_snapshot_paths_must_be_unambiguous_and_repo_relative(tmp_path, path):
    with pytest.raises(ValueError, match="repo-relative"):
        create_activation_artifacts(
            tmp_path / "run",
            candidate_id="runtime-001",
            label="runtime",
            manifest=_manifest("runtime-001", target_files=[path]),
            evidence=EvidenceDecision(
                outcome="rejected",
                gate_passed=False,
            ),
            before={path: "before"},
            after={path: "after"},
        )


def test_manifest_targets_require_complete_before_and_after_snapshots(tmp_path):
    target = _TARGET_BY_LABEL[CandidateLabel.runtime]
    with pytest.raises(ValueError, match="paths do not match"):
        create_activation_artifacts(
            tmp_path / "run",
            candidate_id="runtime-001",
            label="runtime",
            manifest=_manifest("runtime-001"),
            evidence=_accepted(),
            before={},
            after={target: "after"},
            parent_sha=PARENT_SHA,
            candidate_sha=CANDIDATE_SHA,
        )


def _write_summary_fixture(work_dir: Path) -> None:
    nodes = work_dir / "nodes"
    journal = work_dir / "journal"
    nodes.mkdir(parents=True)
    journal.mkdir()
    before, after = _runtime_snapshots()
    repo, parent_sha, candidate_sha = _runtime_repo(
        work_dir.parent / f"{work_dir.name}-summary-subject",
        before,
        after,
    )
    create_activation_artifacts(
        work_dir,
        candidate_id="candidate-accepted",
        label="runtime",
        manifest=_manifest("candidate-accepted"),
        evidence=_accepted(),
        before=before,
        after=after,
        parent_sha=parent_sha,
        candidate_sha=candidate_sha,
        repo_root=repo,
    )
    records = [
        ("candidate-accepted", "promoted_to_baseline"),
        ("candidate-rejected", "pruned_at_confirm"),
        ("candidate-failed", "errored"),
        ("candidate-inconclusive", "blocked_l1"),
        ("candidate-resume-inconclusive", "pruned_at_screen"),
        ("candidate-infra-legacy", "active"),
    ]
    for candidate_id, status in reversed(records):
        record = {"node_id": candidate_id, "status": status}
        if candidate_id == "candidate-accepted":
            record["git_commit_sha"] = candidate_sha
            record["candidate"] = {"manifest": _manifest("candidate-accepted")}
        if candidate_id == "candidate-infra-legacy":
            record["stats"] = {"failure_class": "infrastructure"}
        (nodes / f"{candidate_id}.json").write_text(json.dumps(record))
    (journal / "rounds.jsonl").write_text(
        json.dumps(
            {
                "round_index": 0,
                "candidates": [
                    {
                        "node_id": "candidate-journal-failed",
                        "status": "errored",
                    },
                    {
                        "node_id": "candidate-provider",
                        "status": "errored",
                        "verdict": "failed",
                    },
                    {
                        "node_id": "candidate-resume-inconclusive",
                        "status": "pruned_at_screen",
                        "verdict": "inconclusive",
                    },
                ],
            }
        )
        + "\n"
    )
    target = _TARGET_BY_LABEL[CandidateLabel.prompt]
    create_activation_artifacts(
        work_dir,
        candidate_id="candidate-rejected",
        label="prompt",
        manifest=_manifest("candidate-rejected", label="prompt"),
        evidence=EvidenceDecision(outcome="rejected", gate_passed=False),
        before={target: "before"},
        after={target: "after"},
    )


def test_summary_is_deterministic_across_resume_and_finalize(tmp_path):
    work_dir = tmp_path / "run"
    _write_summary_fixture(work_dir)

    resumed = build_evolution_summary(work_dir)
    output = write_evolution_summary(work_dir)
    finalized = build_evolution_summary(work_dir)

    assert resumed == finalized == json.loads(output.read_text())
    assert resumed["outcomes"] == {
        "accepted": ["candidate-accepted"],
        "rejected": ["candidate-rejected"],
        "failed": [
            "candidate-failed",
            "candidate-infra-legacy",
            "candidate-journal-failed",
            "candidate-provider",
        ],
        "inconclusive": [
            "candidate-inconclusive",
            "candidate-resume-inconclusive",
        ],
    }
    assert resumed["outcome_counts"] == {
        "accepted": 1,
        "rejected": 1,
        "failed": 4,
        "inconclusive": 2,
    }
    assert resumed["activation_states"] == {
        "ineligible": ["candidate-rejected"],
        "pending_human": ["candidate-accepted"],
        "ready": [],
        "activated": [],
        "rolled_back": [],
    }
    assert resumed["integrity_error_count"] == 0
    assert resumed["integrity_errors"] == []


def test_summary_fails_closed_on_corrupt_activation_bundle(tmp_path):
    work_dir = tmp_path / "run"
    _write_summary_fixture(work_dir)
    artifact_dir = work_dir / "activation" / "candidate-accepted"
    (artifact_dir / "after.json").write_text("{}\n")

    first = build_evolution_summary(work_dir)
    second = build_evolution_summary(work_dir)

    assert first == second
    assert "candidate-accepted" not in first["outcomes"]["accepted"]
    assert "candidate-accepted" in first["outcomes"]["failed"]
    assert first["integrity_error_count"] == 1
    assert first["integrity_errors"][0]["source"] == "activation"
    assert first["activation_states"]["pending_human"] == []


def test_summary_rejects_accepted_activation_without_node_ledger(tmp_path):
    work_dir = tmp_path / "run"
    _create_runtime(work_dir, candidate_id="candidate-orphan-activation")

    summary = build_evolution_summary(work_dir)

    assert summary["outcomes"]["accepted"] == []
    assert summary["outcomes"]["failed"] == ["candidate-orphan-activation"]
    assert summary["integrity_errors"][0]["error"] == ("accepted activation has no matching node ledger")


def test_summary_rejects_symlinked_activation_candidate_directory(tmp_path):
    candidate_id = "candidate-symlinked-activation"
    outside_artifact = tmp_path / "outside-artifact"
    outside_artifact.mkdir()
    work_dir = tmp_path / "run"
    nodes = work_dir / "nodes"
    activation_root = work_dir / "activation"
    nodes.mkdir(parents=True)
    activation_root.mkdir()
    (activation_root / candidate_id).symlink_to(
        outside_artifact,
        target_is_directory=True,
    )
    (nodes / f"{candidate_id}.json").write_text(
        json.dumps(
            {
                "node_id": candidate_id,
                "git_commit_sha": CANDIDATE_SHA,
                "status": "promoted_to_baseline",
                "verdict": "accepted",
                "candidate": {"manifest": _manifest(candidate_id)},
            }
        )
    )

    summary = build_evolution_summary(work_dir)

    assert summary["outcomes"]["accepted"] == []
    assert summary["outcomes"]["failed"] == [candidate_id]
    assert summary["integrity_errors"][0]["source"] == "activation"
    assert "symlink" in summary["integrity_errors"][0]["error"]


def test_summary_rejects_symlinked_activation_root_without_traversing_it(tmp_path):
    work_dir = tmp_path / "run"
    outside = tmp_path / "outside"
    work_dir.mkdir()
    outside.mkdir()
    (work_dir / "activation").symlink_to(outside, target_is_directory=True)

    summary = build_evolution_summary(work_dir)

    assert summary["integrity_error_count"] == 1
    assert summary["integrity_errors"] == [
        {
            "source": "activation",
            "path": "activation",
            "candidate_id": "activation:root",
            "error": "activation root must not be a symlink",
        }
    ]


def test_summary_rejects_accepted_manifest_without_activation_bundle(tmp_path):
    work_dir = tmp_path / "run"
    nodes = work_dir / "nodes"
    nodes.mkdir(parents=True)
    candidate_id = "candidate-missing-activation"
    (nodes / f"{candidate_id}.json").write_text(
        json.dumps(
            {
                "node_id": candidate_id,
                "git_commit_sha": CANDIDATE_SHA,
                "status": "promoted_to_baseline",
                "verdict": "accepted",
                "candidate": {"manifest": _manifest(candidate_id)},
            }
        )
    )

    summary = build_evolution_summary(work_dir)

    assert summary["outcomes"]["accepted"] == []
    assert summary["outcomes"]["failed"] == [candidate_id]
    assert summary["integrity_errors"][0]["error"] == ("accepted candidate has no verified activation bundle")


def test_summary_rejects_accepted_node_without_activation_evidence(tmp_path):
    work_dir = tmp_path / "run"
    nodes = work_dir / "nodes"
    nodes.mkdir(parents=True)
    candidate_id = "candidate-without-activation-evidence"
    (nodes / f"{candidate_id}.json").write_text(
        json.dumps(
            {
                "node_id": candidate_id,
                "git_commit_sha": CANDIDATE_SHA,
                "status": "promoted_to_baseline",
                "verdict": "accepted",
            }
        )
    )

    summary = build_evolution_summary(work_dir)

    assert summary["outcomes"]["accepted"] == []
    assert summary["outcomes"]["failed"] == [candidate_id]
    assert summary["integrity_errors"][0]["error"] == ("accepted candidate has no verified activation bundle")


def test_summary_rejects_journal_only_accepted_candidate(tmp_path):
    work_dir = tmp_path / "run"
    journal = work_dir / "journal"
    journal.mkdir(parents=True)
    candidate_id = "candidate-journal-only-accepted"
    (journal / "rounds.jsonl").write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "node_id": candidate_id,
                        "status": "promoted_to_baseline",
                        "verdict": "accepted",
                    }
                ]
            }
        )
        + "\n"
    )

    summary = build_evolution_summary(work_dir)

    assert summary["outcomes"]["accepted"] == []
    assert summary["outcomes"]["failed"] == [candidate_id]
    assert summary["integrity_errors"][0]["error"] == ("accepted candidate has no verified activation bundle")


def test_summary_fails_closed_on_corrupt_node_evidence(tmp_path):
    work_dir = tmp_path / "run"
    _write_summary_fixture(work_dir)
    node_path = work_dir / "nodes" / "candidate-accepted.json"
    node_path.write_text("{")

    summary = build_evolution_summary(work_dir)

    assert "candidate-accepted" not in summary["outcomes"]["accepted"]
    assert "candidate-accepted" in summary["outcomes"]["failed"]
    assert any(
        error["source"] == "node" and error["candidate_id"] == "candidate-accepted"
        for error in summary["integrity_errors"]
    )


def test_summary_surfaces_malformed_durable_journal_records(tmp_path):
    work_dir = tmp_path / "run"
    journal = work_dir / "journal"
    journal.mkdir(parents=True)
    (journal / "rounds.jsonl").write_text(
        json.dumps({"candidates": [{"node_id": "candidate-valid"}]})
        + "\n"
        + "not-json\n"
        + json.dumps({"candidates": [None]})
        + "\n"
    )

    summary = build_evolution_summary(work_dir)

    assert summary["integrity_error_count"] == 2
    assert {error["source"] for error in summary["integrity_errors"]} == {"journal"}
    assert len(summary["outcomes"]["failed"]) == 2
    assert summary["outcomes"]["inconclusive"] == ["candidate-valid"]


def test_summary_tolerates_only_a_truncated_final_journal_line(tmp_path):
    work_dir = tmp_path / "run"
    journal = work_dir / "journal"
    journal.mkdir(parents=True)
    (journal / "rounds.jsonl").write_text(
        json.dumps({"candidates": [{"node_id": "candidate-valid"}]}) + "\n" + '{"candidates":'
    )

    summary = build_evolution_summary(work_dir)

    assert summary["integrity_error_count"] == 0
    assert summary["outcomes"]["inconclusive"] == ["candidate-valid"]


def test_summary_rejects_node_outcome_that_disagrees_with_journal(tmp_path):
    work_dir = tmp_path / "run"
    nodes = work_dir / "nodes"
    journal = work_dir / "journal"
    nodes.mkdir(parents=True)
    journal.mkdir()
    candidate_id = "candidate-ledger-mismatch"
    (nodes / f"{candidate_id}.json").write_text(
        json.dumps(
            {
                "node_id": candidate_id,
                "status": "promoted_to_baseline",
                "verdict": "accepted",
            }
        )
    )
    (journal / "rounds.jsonl").write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "node_id": candidate_id,
                        "status": "pruned_at_confirm",
                        "verdict": "rejected",
                    }
                ]
            }
        )
        + "\n"
    )

    summary = build_evolution_summary(work_dir)

    assert summary["outcomes"]["accepted"] == []
    assert summary["outcomes"]["failed"] == [candidate_id]
    assert summary["integrity_errors"] == [
        {
            "source": "activation",
            "path": f"activation/{candidate_id}",
            "candidate_id": candidate_id,
            "error": "accepted candidate has no verified activation bundle",
        },
        {
            "source": "ledger",
            "path": f"nodes/{candidate_id}.json",
            "candidate_id": candidate_id,
            "error": "node evidence outcome does not match journal ledger",
        },
    ]


def test_summary_rejects_activation_outcome_that_disagrees_with_ledger(
    tmp_path,
):
    work_dir = tmp_path / "run"
    nodes = work_dir / "nodes"
    journal = work_dir / "journal"
    nodes.mkdir(parents=True)
    journal.mkdir()
    candidate_id = "candidate-mismatch"
    (nodes / f"{candidate_id}.json").write_text(
        json.dumps(
            {
                "node_id": candidate_id,
                "status": "promoted_to_baseline",
                "verdict": "accepted",
            }
        )
    )
    (journal / "rounds.jsonl").write_text(
        json.dumps(
            {
                "round_index": 1,
                "candidates": [
                    {
                        "node_id": candidate_id,
                        "status": "promoted_to_baseline",
                        "verdict": "accepted",
                    }
                ],
            }
        )
        + "\n"
    )
    target = _TARGET_BY_LABEL[CandidateLabel.runtime]
    create_activation_artifacts(
        work_dir,
        candidate_id=candidate_id,
        label="runtime",
        manifest=_manifest(
            candidate_id,
            before={target: "before"},
            after={target: "after"},
        ),
        evidence=EvidenceDecision(outcome="rejected", gate_passed=False),
        before={target: "before"},
        after={target: "after"},
    )

    summary = build_evolution_summary(work_dir)

    assert summary["outcomes"]["accepted"] == []
    assert summary["outcomes"]["failed"] == [candidate_id]
    assert summary["integrity_error_count"] == 1
    assert "does not match" in summary["integrity_errors"][0]["error"]
