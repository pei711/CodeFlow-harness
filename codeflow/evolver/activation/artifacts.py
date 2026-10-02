"""为 candidate activation 创建、校验和推进 durable evidence/rollback bundle。

本模块把 candidate identity、label policy、manifest、before/after snapshot、rollback、Runtime
evidence、Git parent/child binding 与 activation state machine 固化为 canonical JSON。所有
payload 都有 SHA-256，path 必须位于 activation root 内且不能经过 symlink；accepted evidence
还必须绑定 repo 中 direct parent/child commit、exact changed paths 与 snapshot bytes。

``create_activation_artifacts`` 只生成 immutable bundle，不把 candidate 应用到 checkout；
``verify_activation_artifacts`` 从磁盘重新计算所有关系；``set_activation_state`` 只按允许的
``ineligible/pending_human/ready/activated/rolled_back`` transition 更新 ledger。rollback payload
必须与 before snapshot 完全一致。

关键证据边界：文件写入成功不等于 bundle 验证通过；evidence accepted 不等于已 activated；
human review 只解除对应 gate；activated 也不证明后续 Runtime 调用或业务任务完成。任何正向
结论必须来自 canonical measurements、Git binding、artifact digest 和 state history 的联合验证。
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from codeflow.evolver.candidate_manifest import (
    LABEL_POLICIES,
    ActivationPolicy,
    CandidateLabel,
    CandidateManifest,
)

SCHEMA_VERSION = 1
ACTIVATION_DIRNAME = "activation"
RECOGNIZED_LABELS = frozenset(label.value for label in CandidateLabel)
SUPPORTED_LABELS = frozenset(label.value for label, policy in LABEL_POLICIES.items() if policy.supported)
PAYLOAD_FILENAMES = frozenset(
    {
        "candidate_manifest.json",
        "evidence.json",
        "before.json",
        "after.json",
        "rollback.json",
    }
)

_CANDIDATE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "label",
        "patch_where",
        "target_files",
        "before_sha256",
        "after_sha256",
        "patch_digest",
        "fixture",
        "evaluator",
        "activation_policy",
    }
)
_DECISION_KEYS = frozenset(
    {
        "outcome",
        "gate_passed",
        "reason",
        "regression",
        "failure_class",
    }
)
_EVIDENCE_KEYS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "label",
        "parent_sha",
        "candidate_sha",
        "manifest_sha256",
        "before_sha256",
        "after_sha256",
        "rollback_sha256",
        "decision",
        "evaluation",
    }
)
_SNAPSHOT_KEYS = frozenset({"schema_version", "files"})
_SNAPSHOT_ENTRY_KEYS = frozenset(
    {
        "path",
        "exists",
        "sha256",
        "content_base64",
    }
)
_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "label",
        "parent_sha",
        "candidate_sha",
        "evidence_outcome",
        "state",
        "requires_human",
        "artifact_digests",
        "identity_digest",
        "state_history",
    }
)
_TRANSITION_KEYS = frozenset({"from", "to", "human_actor", "reason"})


class EvidenceOutcome(str, Enum):
    """candidate evidence 的四种终态值。

    ``accepted`` 表示 canonical gate measurement 通过；``rejected`` 表示明确未通过；``failed``
    可携带 failure_class；``inconclusive`` 表示证据不足。只有 accepted 允许进入 ready 路径。
    """

    accepted = "accepted"
    rejected = "rejected"
    failed = "failed"
    inconclusive = "inconclusive"


class ActivationState(str, Enum):
    """activation bundle 的 durable lifecycle state。

    ``ineligible`` 与 ``rolled_back`` 是终态；``pending_human -> ready -> activated ->
    rolled_back`` 是允许主链。状态不允许跳跃或倒退，实际合法性由 state history replay 验证。
    """

    ineligible = "ineligible"
    pending_human = "pending_human"
    ready = "ready"
    activated = "activated"
    rolled_back = "rolled_back"


_TRANSITIONS = {
    ActivationState.ineligible: frozenset(),
    ActivationState.pending_human: frozenset({ActivationState.ready}),
    ActivationState.ready: frozenset({ActivationState.activated}),
    ActivationState.activated: frozenset({ActivationState.rolled_back}),
    ActivationState.rolled_back: frozenset(),
}


@dataclass(frozen=True)
class EvidenceDecision:
    """把 evidence outcome 与 gate 语义绑定为不可变 decision。

    ``gate_passed`` 当且仅当 outcome 为 ``accepted``；accepted 不能同时 regression 或携带
    failure_class，failure_class 只允许用于 ``failed``。``reason`` 保存解释，``regression``
    标记明确退化。构造时会规范化 Enum 与小写 failure class，并拒绝矛盾组合。
    """

    outcome: EvidenceOutcome | str
    gate_passed: bool
    reason: str = ""
    regression: bool = False
    failure_class: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcome", EvidenceOutcome(self.outcome))
        if not isinstance(self.gate_passed, bool):
            raise ValueError("evidence gate_passed must be a boolean")
        if not isinstance(self.reason, str):
            raise ValueError("evidence reason must be a string")
        if not isinstance(self.regression, bool):
            raise ValueError("evidence regression must be a boolean")
        if self.failure_class is not None and not isinstance(self.failure_class, str):
            raise ValueError("evidence failure_class must be a string or null")
        failure_class = self.failure_class.strip().lower() if self.failure_class is not None else None
        object.__setattr__(self, "failure_class", failure_class)
        self._validate()

    def _validate(self) -> None:
        accepted = self.outcome is EvidenceOutcome.accepted
        if self.gate_passed is not accepted:
            raise ValueError("evidence gate_passed must be true exactly when outcome is accepted")
        if accepted and self.regression:
            raise ValueError("regression evidence cannot be accepted")
        if accepted and self.failure_class:
            raise ValueError("failed evidence cannot be accepted")
        if self.failure_class and self.outcome is not EvidenceOutcome.failed:
            raise ValueError("evidence failure_class requires a failed outcome")

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "gate_passed": self.gate_passed,
            "reason": self.reason,
            "regression": self.regression,
            "failure_class": self.failure_class,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceDecision":
        if set(value) != _DECISION_KEYS:
            raise ValueError(f"evidence decision fields must be exactly {sorted(_DECISION_KEYS)}")
        if not isinstance(value["gate_passed"], bool):
            raise ValueError("evidence gate_passed must be a boolean")
        if not isinstance(value["regression"], bool):
            raise ValueError("evidence regression must be a boolean")
        if not isinstance(value["reason"], str):
            raise ValueError("evidence reason must be a string")
        failure_class = value["failure_class"]
        if failure_class is not None and not isinstance(failure_class, str):
            raise ValueError("evidence failure_class must be a string or null")
        return cls(
            outcome=value["outcome"],
            gate_passed=value["gate_passed"],
            reason=value["reason"],
            regression=value["regression"],
            failure_class=failure_class,
        )


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ": "),
        )
        + "\n"
    ).encode()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _assert_resolved_within(path: Path, root: Path, *, kind: str) -> None:
    resolved_root = root.resolve(strict=False)
    resolved_path = path.resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{kind} escapes its activation root: {path}") from exc


def _activation_directory(work_dir: Path | str, candidate_id: str) -> Path:
    work_dir = Path(work_dir)
    root = work_dir / ACTIVATION_DIRNAME
    if root.is_symlink():
        raise ValueError(f"activation root must not be a symlink: {root}")
    _assert_resolved_within(root, work_dir, kind="activation root")
    artifact_dir = root / candidate_id
    if artifact_dir.is_symlink():
        raise ValueError(f"activation candidate directory must not be a symlink: {artifact_dir}")
    _assert_resolved_within(
        artifact_dir,
        root,
        kind="activation candidate directory",
    )
    return artifact_dir


def _validated_artifact_directory(artifact_dir: Path | str) -> Path:
    artifact_dir = Path(artifact_dir)
    root = artifact_dir.parent
    if root.is_symlink():
        raise ValueError(f"activation root must not be a symlink: {root}")
    if artifact_dir.is_symlink():
        raise ValueError(f"activation candidate directory must not be a symlink: {artifact_dir}")
    _assert_resolved_within(root, root.parent, kind="activation root")
    _assert_resolved_within(
        artifact_dir,
        root,
        kind="activation candidate directory",
    )
    return artifact_dir


def _activation_artifact_path(artifact_dir: Path, name: str) -> Path:
    path = artifact_dir / name
    if path.is_symlink():
        raise ValueError(f"activation artifact file must not be a symlink: {path}")
    _assert_resolved_within(path, artifact_dir, kind="activation artifact file")
    return path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json_object(path: Path, kind: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid {kind}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid {kind}: {path}")
    if raw != _canonical_json(value):
        raise ValueError(f"{kind} is not canonical JSON: {path}")
    return value, raw


def _validate_candidate_id(candidate_id: str) -> str:
    if not isinstance(candidate_id, str):
        raise ValueError("candidate_id must be a string")
    if not _CANDIDATE_ID_RE.fullmatch(candidate_id) or candidate_id in {".", ".."}:
        raise ValueError(f"invalid candidate_id: {candidate_id!r}")
    return candidate_id


def _validate_commit_sha(
    value: str | None,
    field: str,
    *,
    required: bool,
) -> str | None:
    if value is None:
        if required:
            raise ValueError(f"{field} is required for accepted evidence")
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a 40-character hexadecimal commit id")
    normalized = value.strip().lower()
    if not _COMMIT_SHA_RE.fullmatch(normalized):
        raise ValueError(f"{field} must be a 40-character hexadecimal commit id")
    return normalized


def _validate_commit_pair(
    parent_sha: str | None,
    candidate_sha: str | None,
    *,
    required: bool,
) -> tuple[str | None, str | None]:
    parent = _validate_commit_sha(parent_sha, "parent_sha", required=required)
    candidate = _validate_commit_sha(
        candidate_sha,
        "candidate_sha",
        required=required,
    )
    if (parent is None) is not (candidate is None):
        raise ValueError("parent_sha and candidate_sha must be provided together")
    if parent is not None and parent == candidate:
        raise ValueError("parent_sha and candidate_sha must identify different commits")
    return parent, candidate


def _validate_snapshot_path(path: str) -> str:
    if not isinstance(path, str):
        raise ValueError("snapshot path must be a string")
    normalized = path.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if (
        not normalized
        or "\x00" in normalized
        or pure.is_absolute()
        or re.match(r"^[A-Za-z]:/", normalized)
        or any(part in {"", ".", ".."} for part in normalized.split("/"))
    ):
        raise ValueError(f"snapshot path must be repo-relative: {path!r}")
    return normalized


def _path_matches(path: str, pattern: str) -> bool:
    if pattern.endswith("/"):
        return path.startswith(pattern)
    return path == pattern


def _validate_manifest(
    value: CandidateManifest | Mapping[str, Any],
    *,
    candidate_id: str,
    label: str,
    accepted: bool,
) -> CandidateManifest:
    raw = value.to_dict() if isinstance(value, CandidateManifest) else dict(value)
    if set(raw) != _MANIFEST_KEYS:
        raise ValueError(f"candidate manifest fields must be exactly {sorted(_MANIFEST_KEYS)}")
    if type(raw["schema_version"]) is not int:
        raise ValueError("candidate manifest schema_version must be an integer")
    if not isinstance(raw["target_files"], list) or not all(isinstance(path, str) for path in raw["target_files"]):
        raise ValueError("candidate manifest target_files must be a list of strings")
    for field in ("before_sha256", "after_sha256"):
        digests = raw[field]
        if not isinstance(digests, list) or not all(
            digest is None or (isinstance(digest, str) and _DIGEST_RE.fullmatch(digest)) for digest in digests
        ):
            raise ValueError(f"candidate manifest {field} must be a list of SHA-256 values or null")
    for key in (
        "candidate_id",
        "label",
        "patch_where",
        "patch_digest",
        "activation_policy",
    ):
        if not isinstance(raw[key], str):
            raise ValueError(f"candidate manifest {key} must be a string")
    for key in ("fixture", "evaluator"):
        if raw[key] is not None and not isinstance(raw[key], str):
            raise ValueError(f"candidate manifest {key} must be a string or null")

    try:
        manifest = CandidateManifest.from_dict(raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid candidate manifest") from exc
    if raw != manifest.to_dict():
        raise ValueError("candidate manifest is not canonical")
    if manifest.schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported candidate manifest schema: {manifest.schema_version}")
    if _validate_candidate_id(manifest.candidate_id) != candidate_id:
        raise ValueError("manifest candidate_id does not match activation candidate_id")
    if manifest.label.value != label:
        raise ValueError("manifest label does not match activation label")
    if not _DIGEST_RE.fullmatch(manifest.patch_digest):
        raise ValueError("candidate manifest patch_digest must be lowercase SHA-256")

    targets = tuple(_validate_snapshot_path(path) for path in manifest.target_files)
    if not targets:
        raise ValueError("candidate manifest target_files must not be empty")
    if len(targets) != len(set(targets)):
        raise ValueError("candidate manifest target_files must be unique")
    if len(manifest.before_sha256) != len(targets):
        raise ValueError("candidate manifest before_sha256 must align with target_files")
    if len(manifest.after_sha256) != len(targets):
        raise ValueError("candidate manifest after_sha256 must align with target_files")

    policy = LABEL_POLICIES[manifest.label]
    if manifest.patch_where not in policy.patch_where:
        raise ValueError(f"label {label!r} does not allow PatchWhere {manifest.patch_where.value!r}")
    if manifest.fixture != policy.fixture:
        raise ValueError("manifest fixture does not match canonical label policy")
    if manifest.evaluator != policy.evaluator:
        raise ValueError("manifest evaluator does not match canonical label policy")
    if manifest.activation_policy != policy.activation_policy:
        raise ValueError("manifest activation policy does not match canonical label policy")

    if accepted:
        if not policy.supported:
            raise ValueError(f"accepted evidence is forbidden for unsupported label {label!r}")
        for target in targets:
            if not any(_path_matches(target, pattern) for pattern in policy.mutable_paths):
                raise ValueError(f"accepted target is outside the {label} mutable allowlist: {target}")
    return manifest


def _snapshot_payload(
    files: Mapping[str, bytes | str | None],
    *,
    target_order: tuple[str, ...],
) -> dict[str, Any]:
    entries = []
    normalized_paths: set[str] = set()
    for raw_path, raw_content in files.items():
        path = _validate_snapshot_path(raw_path)
        if path in normalized_paths:
            raise ValueError(f"duplicate normalized snapshot path: {path!r}")
        normalized_paths.add(path)
        if raw_content is None:
            entries.append(
                {
                    "path": path,
                    "exists": False,
                    "sha256": None,
                    "content_base64": None,
                }
            )
            continue
        content = raw_content.encode() if isinstance(raw_content, str) else raw_content
        if not isinstance(content, bytes):
            raise TypeError(f"snapshot content for {path!r} must be bytes, str, or None")
        entries.append(
            {
                "path": path,
                "exists": True,
                "sha256": _sha256(content),
                "content_base64": base64.b64encode(content).decode("ascii"),
            }
        )
    rank = {path: index for index, path in enumerate(target_order)}
    entries.sort(
        key=lambda entry: (
            rank.get(entry["path"], len(rank)),
            entry["path"],
        )
    )
    return {"schema_version": SCHEMA_VERSION, "files": entries}


def _validate_snapshot_payload(
    value: Mapping[str, Any],
    *,
    expected_targets: tuple[str, ...],
    name: str,
) -> dict[str, Any]:
    payload = dict(value)
    if set(payload) != _SNAPSHOT_KEYS:
        raise ValueError(f"{name} snapshot has unexpected fields")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"{name} snapshot has an unsupported schema")
    entries = payload["files"]
    if not isinstance(entries, list):
        raise ValueError(f"{name} snapshot files must be a list")

    paths: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != _SNAPSHOT_ENTRY_KEYS:
            raise ValueError(f"{name} snapshot has an invalid file entry")
        if not isinstance(entry["path"], str):
            raise ValueError(f"{name} snapshot path must be a string")
        path = _validate_snapshot_path(entry["path"])
        if not isinstance(entry["exists"], bool):
            raise ValueError(f"{name} snapshot exists must be a boolean")
        if entry["exists"]:
            digest = entry["sha256"]
            encoded = entry["content_base64"]
            if not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
                raise ValueError(f"{name} snapshot has an invalid content digest")
            if not isinstance(encoded, str):
                raise ValueError(f"{name} snapshot has invalid base64 content")
            try:
                content = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError(f"{name} snapshot has invalid base64 content") from exc
            if _sha256(content) != digest:
                raise ValueError(f"{name} snapshot content digest mismatch")
        elif entry["sha256"] is not None or entry["content_base64"] is not None:
            raise ValueError(f"{name} snapshot marks a missing file with content")
        paths.append(path)

    if tuple(paths) != expected_targets:
        raise ValueError(f"{name} snapshot paths do not match manifest target_files")
    if payload != {"schema_version": SCHEMA_VERSION, "files": entries}:
        raise ValueError(f"{name} snapshot is not canonical")
    return payload


def _snapshot_content_digests(
    payload: Mapping[str, Any],
) -> tuple[str | None, ...]:
    return tuple(entry["sha256"] for entry in payload["files"])


def _validate_manifest_snapshot_digests(
    manifest: CandidateManifest,
    *,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> None:
    if manifest.before_sha256 != _snapshot_content_digests(before):
        raise ValueError("candidate manifest before_sha256 does not match before snapshot")
    if manifest.after_sha256 != _snapshot_content_digests(after):
        raise ValueError("candidate manifest after_sha256 does not match after snapshot")


def _snapshot_contents(payload: Mapping[str, Any]) -> dict[str, bytes | None]:
    contents: dict[str, bytes | None] = {}
    for entry in payload["files"]:
        contents[entry["path"]] = base64.b64decode(entry["content_base64"], validate=True) if entry["exists"] else None
    return contents


def _validate_git_binding(
    repo_root: Path | str | None,
    *,
    parent_sha: str,
    candidate_sha: str,
    manifest: CandidateManifest,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> None:
    if repo_root is None:
        raise ValueError("accepted evidence requires repo_root for Git verification")
    from codeflow.evolver.tree import git_ops

    root = Path(repo_root)
    try:
        if not git_ops.commit_exists(root, parent_sha):
            raise ValueError("accepted evidence parent_sha is not a commit in repo_root")
        if not git_ops.commit_exists(root, candidate_sha):
            raise ValueError("accepted evidence candidate_sha is not a commit in repo_root")
        if git_ops.get_commit_parents(root, candidate_sha) != [parent_sha]:
            raise ValueError("accepted evidence candidate commit is not a direct child of parent_sha")
        expected_paths = set(manifest.target_files)
        actual_paths = set(git_ops.changed_paths_between(root, parent_sha, candidate_sha))
        if actual_paths != expected_paths:
            raise ValueError(
                "accepted evidence Git paths differ from manifest targets: "
                f"expected={sorted(expected_paths)}, actual={sorted(actual_paths)}"
            )

        before_contents = _snapshot_contents(before)
        after_contents = _snapshot_contents(after)
        for path in manifest.target_files:
            parent_mode = git_ops.tree_file_mode(root, parent_sha, path)
            candidate_mode = git_ops.tree_file_mode(root, candidate_sha, path)
            if parent_mode not in {None, "100644", "100755"}:
                raise ValueError(f"accepted evidence parent target is not a regular file: {path!r}")
            if candidate_mode not in {None, "100644", "100755"}:
                raise ValueError(f"accepted evidence candidate target is not a regular file: {path!r}")
            parent_blob = git_ops.read_file_at(root, parent_sha, path) if parent_mode is not None else None
            candidate_blob = git_ops.read_file_at(root, candidate_sha, path) if candidate_mode is not None else None
            if parent_blob != before_contents[path]:
                raise ValueError(f"accepted evidence before snapshot differs from parent commit: {path!r}")
            if candidate_blob != after_contents[path]:
                raise ValueError(f"accepted evidence after snapshot differs from candidate commit: {path!r}")
            if parent_blob == candidate_blob:
                raise ValueError(f"accepted evidence target has no content change: {path!r}")
    except (OSError, git_ops.GitOpError) as exc:
        raise ValueError("accepted evidence Git binding could not be verified") from exc


def _coerce_evidence(
    manifest: CandidateManifest,
    evidence: EvidenceDecision | EvidenceOutcome | str | Mapping[str, Any],
) -> tuple[EvidenceDecision, dict[str, Any] | None]:
    from codeflow.evolver.candidate_evidence import (
        AcceptedRuntimeEvidence,
        recompute_accepted_runtime_evidence,
    )

    if isinstance(evidence, AcceptedRuntimeEvidence):
        evaluation = evidence.to_dict()
        return recompute_accepted_runtime_evidence(manifest, evidence), evaluation
    if isinstance(evidence, Mapping) and "candidate_evals" in evidence:
        evaluation_value = dict(evidence)
        accepted = AcceptedRuntimeEvidence.from_dict(evaluation_value)
        canonical = accepted.to_dict()
        if evaluation_value != canonical:
            raise ValueError("accepted runtime evidence is not canonical")
        return recompute_accepted_runtime_evidence(manifest, accepted), canonical
    if isinstance(evidence, EvidenceDecision):
        decision = evidence
    elif isinstance(evidence, (EvidenceOutcome, str)):
        outcome = EvidenceOutcome(evidence)
        if outcome is EvidenceOutcome.accepted:
            raise ValueError("accepted evidence must contain canonical candidate and control measurements")
        decision = EvidenceDecision(
            outcome=outcome,
            gate_passed=False,
        )
    else:
        decision = EvidenceDecision.from_dict(evidence)
    if decision.outcome is EvidenceOutcome.accepted:
        raise ValueError("accepted evidence must contain canonical candidate and control measurements")
    return decision, None


def _evidence_payload(
    *,
    candidate_id: str,
    label: str,
    parent_sha: str | None,
    candidate_sha: str | None,
    manifest_digest: str,
    before_digest: str,
    after_digest: str,
    decision: EvidenceDecision,
    evaluation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "label": label,
        "parent_sha": parent_sha,
        "candidate_sha": candidate_sha,
        "manifest_sha256": manifest_digest,
        "before_sha256": before_digest,
        "after_sha256": after_digest,
        "rollback_sha256": before_digest,
        "decision": decision.to_dict(),
        "evaluation": dict(evaluation) if evaluation is not None else None,
    }


def _identity_digest(record: Mapping[str, Any]) -> str:
    identity = {
        "schema_version": record["schema_version"],
        "candidate_id": record["candidate_id"],
        "label": record["label"],
        "parent_sha": record["parent_sha"],
        "candidate_sha": record["candidate_sha"],
        "evidence_outcome": record["evidence_outcome"],
        "requires_human": record["requires_human"],
        "artifact_digests": record["artifact_digests"],
    }
    return _sha256(_canonical_json(identity))


def _initial_state(
    decision: EvidenceDecision,
    *,
    requires_human: bool,
) -> ActivationState:
    if decision.outcome is not EvidenceOutcome.accepted:
        return ActivationState.ineligible
    return ActivationState.pending_human if requires_human else ActivationState.ready


def _validate_state_history(
    record: Mapping[str, Any],
    decision: EvidenceDecision,
) -> ActivationState:
    requires_human = record["requires_human"]
    if not isinstance(requires_human, bool):
        raise ValueError("activation requires_human must be a boolean")
    current = _initial_state(decision, requires_human=requires_human)
    history = record["state_history"]
    if not isinstance(history, list):
        raise ValueError("activation state_history must be a list")

    for index, transition in enumerate(history):
        if not isinstance(transition, dict) or set(transition) != _TRANSITION_KEYS:
            raise ValueError(f"activation state_history entry {index} is invalid")
        try:
            source = ActivationState(transition["from"])
            target = ActivationState(transition["to"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"activation state_history entry {index} has an invalid state") from exc
        if source is not current:
            raise ValueError(f"activation state_history entry {index} does not continue the previous state")
        if target not in _TRANSITIONS[current]:
            raise ValueError(f"activation state_history entry {index} has an invalid transition")
        actor = transition["human_actor"]
        reason = transition["reason"]
        if actor is not None and not isinstance(actor, str):
            raise ValueError(f"activation state_history entry {index} has an invalid human_actor")
        if not isinstance(reason, str):
            raise ValueError(f"activation state_history entry {index} has an invalid reason")
        if current is ActivationState.pending_human and not str(actor or "").strip():
            raise ValueError("pending_human -> ready transition requires human_actor")
        current = target

    try:
        recorded = ActivationState(record["state"])
    except (TypeError, ValueError) as exc:
        raise ValueError("activation record has an invalid state") from exc
    if recorded is not current:
        raise ValueError("activation state does not match replayed state_history")
    return current


def create_activation_artifacts(
    work_dir: Path | str,
    *,
    candidate_id: str,
    label: str,
    manifest: CandidateManifest | Mapping[str, Any],
    evidence: EvidenceDecision | EvidenceOutcome | str | Mapping[str, Any],
    before: Mapping[str, bytes | str | None],
    after: Mapping[str, bytes | str | None],
    require_human: bool = True,
    parent_sha: str | None = None,
    candidate_sha: str | None = None,
    repo_root: Path | str | None = None,
) -> Path:
    """创建 immutable activation bundle，但不应用 candidate。

    ``work_dir/candidate_id`` 经过 traversal/symlink 防护。函数验证 label 与 canonical
    ``CandidateManifest``，把 evidence 归一化为 ``EvidenceDecision``；accepted evidence 必须
    包含 canonical candidate/control measurements、合法 label policy、parent/candidate 40-char
    SHA，并使用 ``repo_root`` 验证 direct child、changed paths、file mode 与 before/after bytes。

    ``before``/``after`` 只允许 manifest target，内容编码为 base64 并逐项校验 SHA-256；
    ``rollback.json`` 等于 ``before.json``。payload 先 atomic write，再写带 identity digest 与
    initial state 的 ``activation.json``，最后全量回读验证。已存在相同 durable identity 时
    幂等返回；任一 immutable field 不同则抛出 ``ValueError``。

    返回 artifact directory。成功表示 bundle 已持久化且当前验证通过，不表示 candidate 已
    activated、checkout 已改变或效果已在线上出现。
    """

    candidate_id = _validate_candidate_id(candidate_id)
    try:
        parsed_label = CandidateLabel(label)
    except ValueError as exc:
        raise ValueError(f"unrecognized candidate label: {label!r}") from exc
    label = parsed_label.value
    if not isinstance(require_human, bool):
        raise ValueError("require_human must be a boolean")
    manifest_value = _validate_manifest(
        manifest,
        candidate_id=candidate_id,
        label=label,
        accepted=False,
    )
    from codeflow.evolver.candidate_evidence import AcceptedRuntimeEvidence

    canonical_accepted = isinstance(evidence, AcceptedRuntimeEvidence) or (
        isinstance(evidence, Mapping) and "candidate_evals" in evidence
    )
    if canonical_accepted:
        manifest_value = _validate_manifest(
            manifest,
            candidate_id=candidate_id,
            label=label,
            accepted=True,
        )
    decision, evaluation = _coerce_evidence(manifest_value, evidence)
    parent_sha, candidate_sha = _validate_commit_pair(
        parent_sha,
        candidate_sha,
        required=decision.outcome is EvidenceOutcome.accepted,
    )
    policy = LABEL_POLICIES[manifest_value.label]
    requires_human = bool(require_human) or (policy.activation_policy is ActivationPolicy.human_review)
    initial_state = _initial_state(decision, requires_human=requires_human)

    expected_targets = manifest_value.target_files
    before_payload = _snapshot_payload(
        before,
        target_order=expected_targets,
    )
    after_payload = _snapshot_payload(
        after,
        target_order=expected_targets,
    )
    _validate_snapshot_payload(
        before_payload,
        expected_targets=expected_targets,
        name="before",
    )
    _validate_snapshot_payload(
        after_payload,
        expected_targets=expected_targets,
        name="after",
    )
    _validate_manifest_snapshot_digests(
        manifest_value,
        before=before_payload,
        after=after_payload,
    )
    if decision.outcome is EvidenceOutcome.accepted:
        if parent_sha is None or candidate_sha is None:
            raise ValueError("accepted evidence is missing validated commit identities")
        _validate_git_binding(
            repo_root,
            parent_sha=parent_sha,
            candidate_sha=candidate_sha,
            manifest=manifest_value,
            before=before_payload,
            after=after_payload,
        )

    manifest_bytes = _canonical_json(manifest_value.to_dict())
    before_bytes = _canonical_json(before_payload)
    after_bytes = _canonical_json(after_payload)
    evidence_value = _evidence_payload(
        candidate_id=candidate_id,
        label=label,
        parent_sha=parent_sha,
        candidate_sha=candidate_sha,
        manifest_digest=_sha256(manifest_bytes),
        before_digest=_sha256(before_bytes),
        after_digest=_sha256(after_bytes),
        decision=decision,
        evaluation=evaluation,
    )
    payloads = {
        "candidate_manifest.json": manifest_bytes,
        "evidence.json": _canonical_json(evidence_value),
        "before.json": before_bytes,
        "after.json": after_bytes,
        "rollback.json": before_bytes,
    }
    artifact_digests = {name: _sha256(content) for name, content in sorted(payloads.items())}
    artifact_dir = _activation_directory(work_dir, candidate_id)
    activation_path = _activation_artifact_path(artifact_dir, "activation.json")
    record = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "label": label,
        "parent_sha": parent_sha,
        "candidate_sha": candidate_sha,
        "evidence_outcome": decision.outcome.value,
        "state": initial_state.value,
        "requires_human": requires_human,
        "artifact_digests": artifact_digests,
        "identity_digest": "",
        "state_history": [],
    }
    record["identity_digest"] = _identity_digest(record)

    if activation_path.is_file():
        existing = verify_activation_artifacts(artifact_dir, repo_root=repo_root)
        immutable_keys = (
            "schema_version",
            "candidate_id",
            "label",
            "parent_sha",
            "candidate_sha",
            "evidence_outcome",
            "requires_human",
            "artifact_digests",
            "identity_digest",
        )
        if any(existing[key] != record[key] for key in immutable_keys):
            raise ValueError(f"activation artifacts for {candidate_id!r} differ from durable state")
        return artifact_dir

    for name, content in payloads.items():
        _atomic_write(_activation_artifact_path(artifact_dir, name), content)
    _atomic_write(activation_path, _canonical_json(record))
    verify_activation_artifacts(artifact_dir, repo_root=repo_root)
    return artifact_dir


def _read_activation_record(artifact_dir: Path) -> dict[str, Any]:
    record, _ = _read_json_object(
        _activation_artifact_path(artifact_dir, "activation.json"),
        "activation record",
    )
    if set(record) != _RECORD_KEYS:
        raise ValueError("activation record has unexpected fields")
    if type(record["schema_version"]) is not int or record["schema_version"] != SCHEMA_VERSION:
        raise ValueError("activation record has an unsupported schema")
    for field in (
        "candidate_id",
        "label",
        "evidence_outcome",
        "state",
        "identity_digest",
    ):
        if not isinstance(record[field], str):
            raise ValueError(f"activation record {field} must be a string")
    candidate_id = _validate_candidate_id(record["candidate_id"])
    if artifact_dir.name != candidate_id:
        raise ValueError("activation directory does not match record candidate_id")
    try:
        CandidateLabel(record["label"])
        EvidenceOutcome(record["evidence_outcome"])
    except (TypeError, ValueError) as exc:
        raise ValueError("activation record has an invalid label or outcome") from exc
    return record


def load_activation_record(
    artifact_dir: Path | str,
    *,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """验证并加载 activation record。

    该入口直接委托 ``verify_activation_artifacts``，因此不是宽松 read：digest、canonical JSON、
    manifest/snapshot/evidence/state relationship 任一异常都会抛出 ``ValueError``。提供
    ``repo_root`` 时还会重新验证 accepted Git binding。返回的是验证后的 record dict。
    """
    return verify_activation_artifacts(artifact_dir, repo_root=repo_root)


def verify_activation_artifacts(
    artifact_dir: Path | str,
    *,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """从磁盘全量验证 activation bundle，并返回 durable record。

    验证覆盖目录边界、required payload set、每个 SHA-256、canonical JSON、rollback=before、
    identity digest、manifest policy、snapshot path/content、accepted Runtime measurement、commit
    identity、可选 Git tree binding，以及 state history replay。non-accepted evidence 不得夹带
    accepted measurement；human-review label 必须保留 ``requires_human``。

    任一不一致抛出 ``ValueError``，不会尝试修复文件。返回成功说明 bundle 在当前读取时
    self-consistent；若省略 ``repo_root``，不会重新访问 Git repo 验证 commit contents。
    """
    artifact_dir = _validated_artifact_directory(artifact_dir)
    record = _read_activation_record(artifact_dir)
    decision_outcome = EvidenceOutcome(record["evidence_outcome"])
    parent_sha, candidate_sha = _validate_commit_pair(
        record["parent_sha"],
        record["candidate_sha"],
        required=decision_outcome is EvidenceOutcome.accepted,
    )
    if parent_sha != record["parent_sha"] or candidate_sha != record["candidate_sha"]:
        raise ValueError("activation commit identities are not canonical")

    digests = record["artifact_digests"]
    if not isinstance(digests, dict) or set(digests) != PAYLOAD_FILENAMES:
        raise ValueError(f"activation artifact digests must contain exactly {sorted(PAYLOAD_FILENAMES)}")
    payload_bytes: dict[str, bytes] = {}
    for name in sorted(PAYLOAD_FILENAMES):
        expected = digests[name]
        if not isinstance(expected, str) or not _DIGEST_RE.fullmatch(expected):
            raise ValueError(f"invalid activation artifact digest for {name}")
        path = _activation_artifact_path(artifact_dir, name)
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"missing activation artifact: {path}") from exc
        if _sha256(raw) != expected:
            raise ValueError(f"activation artifact digest mismatch: {path}")
        payload_bytes[name] = raw

    if payload_bytes["before.json"] != payload_bytes["rollback.json"]:
        raise ValueError("rollback artifact does not match before artifact")
    if not _DIGEST_RE.fullmatch(record["identity_digest"]):
        raise ValueError("activation identity digest is invalid")
    if record["identity_digest"] != _identity_digest(record):
        raise ValueError("activation identity digest mismatch")

    manifest_raw = json.loads(payload_bytes["candidate_manifest.json"])
    if not isinstance(manifest_raw, dict):
        raise ValueError("candidate manifest artifact must contain an object")
    manifest = _validate_manifest(
        manifest_raw,
        candidate_id=record["candidate_id"],
        label=record["label"],
        accepted=decision_outcome is EvidenceOutcome.accepted,
    )
    if payload_bytes["candidate_manifest.json"] != _canonical_json(manifest.to_dict()):
        raise ValueError("candidate manifest artifact is not canonical")

    expected_targets = manifest.target_files
    snapshots: dict[str, dict[str, Any]] = {}
    for name in ("before", "after", "rollback"):
        raw = payload_bytes[f"{name}.json"]
        try:
            value = json.loads(raw)
        except ValueError as exc:
            raise ValueError(f"invalid {name} snapshot JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"invalid {name} snapshot JSON")
        snapshots[name] = _validate_snapshot_payload(
            value,
            expected_targets=expected_targets,
            name=name,
        )
        if raw != _canonical_json(snapshots[name]):
            raise ValueError(f"{name} snapshot artifact is not canonical")
    _validate_manifest_snapshot_digests(
        manifest,
        before=snapshots["before"],
        after=snapshots["after"],
    )

    try:
        evidence_raw = json.loads(payload_bytes["evidence.json"])
    except ValueError as exc:
        raise ValueError("invalid evidence artifact JSON") from exc
    if not isinstance(evidence_raw, dict) or set(evidence_raw) != _EVIDENCE_KEYS:
        raise ValueError("evidence artifact has unexpected fields")
    if type(evidence_raw["schema_version"]) is not int or evidence_raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError("evidence artifact has an unsupported schema")
    decision_raw = evidence_raw["decision"]
    if not isinstance(decision_raw, dict):
        raise ValueError("evidence decision must contain an object")
    decision = EvidenceDecision.from_dict(decision_raw)
    evaluation_raw = evidence_raw["evaluation"]
    evaluation: dict[str, Any] | None
    if decision.outcome is EvidenceOutcome.accepted:
        if not isinstance(evaluation_raw, dict):
            raise ValueError("accepted evidence must contain canonical candidate and control measurements")
        from codeflow.evolver.candidate_evidence import (
            AcceptedRuntimeEvidence,
            recompute_accepted_runtime_evidence,
        )

        accepted_evidence = AcceptedRuntimeEvidence.from_dict(evaluation_raw)
        evaluation = accepted_evidence.to_dict()
        if evaluation_raw != evaluation:
            raise ValueError("accepted runtime evidence is not canonical")
        if recompute_accepted_runtime_evidence(manifest, accepted_evidence) != decision:
            raise ValueError("accepted evidence decision does not match canonical measurements")
    else:
        if evaluation_raw is not None:
            raise ValueError("non-accepted evidence must not contain accepted measurements")
        evaluation = None
    expected_evidence = _evidence_payload(
        candidate_id=record["candidate_id"],
        label=record["label"],
        parent_sha=parent_sha,
        candidate_sha=candidate_sha,
        manifest_digest=digests["candidate_manifest.json"],
        before_digest=digests["before.json"],
        after_digest=digests["after.json"],
        decision=decision,
        evaluation=evaluation,
    )
    if evidence_raw != expected_evidence:
        raise ValueError("evidence artifact does not match candidate, manifest, or snapshots")
    if payload_bytes["evidence.json"] != _canonical_json(expected_evidence):
        raise ValueError("evidence artifact is not canonical")
    if decision.outcome.value != record["evidence_outcome"]:
        raise ValueError("activation outcome does not match evidence decision")
    if decision.outcome is EvidenceOutcome.accepted and repo_root is not None:
        if parent_sha is None or candidate_sha is None:
            raise ValueError("accepted evidence is missing validated commit identities")
        _validate_git_binding(
            repo_root,
            parent_sha=parent_sha,
            candidate_sha=candidate_sha,
            manifest=manifest,
            before=snapshots["before"],
            after=snapshots["after"],
        )

    policy = LABEL_POLICIES[manifest.label]
    if policy.activation_policy is ActivationPolicy.human_review and not record["requires_human"]:
        raise ValueError("label policy requires human activation review")
    _validate_state_history(record, decision)
    return record


def set_activation_state(
    artifact_dir: Path | str,
    state: ActivationState | str,
    *,
    human_actor: str | None = None,
    reason: str = "",
) -> dict[str, Any]:
    """推进 artifact state，同时保持 caller checkout untouched。

    方法先全量 verify，same-state request 幂等返回；其他请求必须属于 ``_TRANSITIONS``。
    ``pending_human -> ready`` 强制要求 non-empty ``human_actor``，``reason`` 必须是 string。
    transition 追加到 immutable history 语义的 list，atomic rewrite ``activation.json`` 后再次
    verify。

    返回更新后的 record。state 变为 ``activated`` 仍只表示 durable approval 状态，本方法
    不应用 patch；真正 checkout mutation 属于 applier/launch 路径。
    """

    artifact_dir = Path(artifact_dir)
    record = verify_activation_artifacts(artifact_dir)
    current = ActivationState(record["state"])
    requested = ActivationState(state)
    if current is requested:
        return record
    if requested not in _TRANSITIONS[current]:
        raise ValueError(f"invalid activation transition: {current.value} -> {requested.value}")
    if human_actor is not None and not isinstance(human_actor, str):
        raise ValueError("activation transition human_actor must be a string")
    actor = human_actor.strip() if human_actor is not None else None
    if current is ActivationState.pending_human and not actor:
        raise ValueError("pending_human -> ready requires human_actor")
    if not isinstance(reason, str):
        raise ValueError("activation transition reason must be a string")

    transition = {
        "from": current.value,
        "to": requested.value,
        "human_actor": actor,
        "reason": reason,
    }
    record["state"] = requested.value
    record["state_history"] = [*record["state_history"], transition]
    _atomic_write(
        _activation_artifact_path(artifact_dir, "activation.json"),
        _canonical_json(record),
    )
    return verify_activation_artifacts(artifact_dir)


__all__ = [
    "ACTIVATION_DIRNAME",
    "ActivationState",
    "EvidenceDecision",
    "EvidenceOutcome",
    "PAYLOAD_FILENAMES",
    "RECOGNIZED_LABELS",
    "SCHEMA_VERSION",
    "SUPPORTED_LABELS",
    "create_activation_artifacts",
    "load_activation_record",
    "set_activation_state",
    "verify_activation_artifacts",
]
