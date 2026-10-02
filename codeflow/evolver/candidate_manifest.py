"""定义 Evolver candidate 的 reviewable label、manifest 与 G5 policy gate。

``CandidateLabel`` 把 patch 按 Skill、prompt、policy、Runtime、model profile、route 分类；
``LABEL_POLICIES`` 为每类固定允许的 ``PatchWhere``、mutable path、fixture、evaluator 与
activation policy。当前只有 ``runtime`` 具备 executable fixture/evaluator，其他 label 明确
unsupported，不能因 manifest shape 合法就获得 accepted evidence。

``manifest_for_patch`` 把 AppliedPatch 与 before/after bytes 绑定为 SHA-256；
``evaluate_manifest_gate`` 重新验证 schema、label policy、digest、path trust boundary、fixture/
evaluator binding 与 content change，返回 G5 result。G5 通过只证明 candidate 可进入 evidence
评估，不证明 patch 正确、效果为正、已 activated 或任务完成。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from codeflow.evolver.applier.path_guard import check_patch_paths
from codeflow.evolver.judge.schema import PatchWhere
from codeflow.evolver.tree.node import AppliedPatch


class CandidateLabel(str, Enum):
    """candidate patch 的 review/evidence 类别。

    Enum value 是 durable manifest 字面值；新增 label 必须同步 ``LABEL_POLICIES`` 与可执行
    evidence binding，否则应保持 unsupported。
    """

    skill = "skill"
    prompt = "prompt"
    policy = "policy"
    runtime = "runtime"
    model_profile = "model_profile"
    route = "route"


class ActivationPolicy(str, Enum):
    """candidate 通过 evidence 后的 activation 审批策略。

    ``gated`` 可按自动 gate 进入 ready；``human_review`` 要求 human actor 明确批准。
    """

    gated = "gated"
    human_review = "human_review"


@dataclass(frozen=True)
class LabelPolicy:
    """一个 CandidateLabel 的 canonical mutable/evidence policy。

    supported policy 必须同时声明 fixture、mutable_paths 与 evaluator；unsupported policy 必须
    给出 reason。``config_only`` 为 model_profile/route 附加文件类型限制。
    """

    label: CandidateLabel
    patch_where: frozenset[PatchWhere]
    mutable_paths: tuple[str, ...]
    fixture: str | None
    evaluator: str | None
    activation_policy: ActivationPolicy
    supported: bool
    config_only: bool = False
    unsupported_reason: str | None = None

    def __post_init__(self) -> None:
        if self.supported and (not self.fixture or not self.mutable_paths or not self.evaluator):
            raise ValueError(f"Supported label {self.label.value!r} requires fixture, allowlist, and evaluator")
        if not self.supported and not self.unsupported_reason:
            raise ValueError(f"Unsupported label {self.label.value!r} requires a reason")


LABEL_POLICIES: dict[CandidateLabel, LabelPolicy] = {
    CandidateLabel.skill: LabelPolicy(
        label=CandidateLabel.skill,
        patch_where=frozenset({PatchWhere.skill}),
        mutable_paths=(),
        fixture=None,
        evaluator=None,
        activation_policy=ActivationPolicy.gated,
        supported=False,
        unsupported_reason="No deterministic Skill routing fixture and evaluator are wired",
    ),
    CandidateLabel.prompt: LabelPolicy(
        label=CandidateLabel.prompt,
        patch_where=frozenset(
            {
                PatchWhere.system_prompt_template,
                PatchWhere.task_wrapper_prompt,
            }
        ),
        mutable_paths=(),
        fixture=None,
        evaluator=None,
        activation_policy=ActivationPolicy.gated,
        supported=False,
        unsupported_reason="No deterministic prompt rendering fixture and evaluator are wired",
    ),
    CandidateLabel.policy: LabelPolicy(
        label=CandidateLabel.policy,
        patch_where=frozenset(
            {
                PatchWhere.config,
                PatchWhere.hook_new,
                PatchWhere.hook_modify,
            }
        ),
        mutable_paths=(),
        fixture=None,
        evaluator=None,
        activation_policy=ActivationPolicy.human_review,
        supported=False,
        unsupported_reason="No deterministic policy evaluator is wired to the retained Runtime",
    ),
    CandidateLabel.runtime: LabelPolicy(
        label=CandidateLabel.runtime,
        patch_where=frozenset(
            {
                PatchWhere.tool_description,
                PatchWhere.hook_new,
                PatchWhere.hook_modify,
                PatchWhere.tool_new,
                PatchWhere.loop_override,
                PatchWhere.context_override,
                PatchWhere.tool_override,
            }
        ),
        mutable_paths=(
            "benchmarks/appworld/agent_cli.py",
            "benchmarks/appworld/tool.py",
        ),
        fixture="appworld_runtime_v1",
        evaluator="appworld_focused_fisher_v1",
        activation_policy=ActivationPolicy.human_review,
        supported=True,
    ),
    CandidateLabel.model_profile: LabelPolicy(
        label=CandidateLabel.model_profile,
        patch_where=frozenset({PatchWhere.config}),
        mutable_paths=(),
        fixture=None,
        evaluator=None,
        activation_policy=ActivationPolicy.human_review,
        supported=False,
        config_only=True,
        unsupported_reason="No config-only model-profile fixture and evaluator are wired",
    ),
    CandidateLabel.route: LabelPolicy(
        label=CandidateLabel.route,
        patch_where=frozenset({PatchWhere.config}),
        mutable_paths=(),
        fixture=None,
        evaluator=None,
        activation_policy=ActivationPolicy.human_review,
        supported=False,
        config_only=True,
        unsupported_reason="No config-only route fixture and evaluator are wired",
    ),
}


@dataclass(frozen=True)
class CandidateManifest:
    """把 candidate identity、patch 与 before/after content 绑定的不可变 manifest。

    ``target_files`` 与两组 digest 按相同顺序一一对应；``patch_digest`` 还包含 AppliedPatch
    结构。fixture/evaluator/activation_policy 必须等于 label canonical policy。对象本身不验证
    Git commit 或执行 evaluator，需经过 G5 与 activation artifact 验证。
    """

    schema_version: int
    candidate_id: str
    label: CandidateLabel
    patch_where: PatchWhere
    target_files: tuple[str, ...]
    before_sha256: tuple[str | None, ...]
    after_sha256: tuple[str | None, ...]
    patch_digest: str
    fixture: str | None
    evaluator: str | None
    activation_policy: ActivationPolicy

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "label": self.label.value,
            "patch_where": self.patch_where.value,
            "target_files": list(self.target_files),
            "before_sha256": list(self.before_sha256),
            "after_sha256": list(self.after_sha256),
            "patch_digest": self.patch_digest,
            "fixture": self.fixture,
            "evaluator": self.evaluator,
            "activation_policy": self.activation_policy.value,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CandidateManifest":
        """从 mapping 恢复 manifest 的 typed fields。

        Enum、int 与 sequence 转换失败会传播异常。该方法不检查 exact field set 或 canonical
        policy；durable artifact reader 会在外层完成严格验证。
        """
        return cls(
            schema_version=int(value["schema_version"]),
            candidate_id=str(value["candidate_id"]),
            label=CandidateLabel(value["label"]),
            patch_where=PatchWhere(value["patch_where"]),
            target_files=tuple(str(path) for path in value["target_files"]),
            before_sha256=tuple(str(digest) if digest is not None else None for digest in value["before_sha256"]),
            after_sha256=tuple(str(digest) if digest is not None else None for digest in value["after_sha256"]),
            patch_digest=str(value["patch_digest"]),
            fixture=value.get("fixture"),
            evaluator=value.get("evaluator"),
            activation_policy=ActivationPolicy(value["activation_policy"]),
        )

    @classmethod
    def from_json(cls, value: str) -> "CandidateManifest":
        """解析 JSON object 并调用 ``from_dict``。

        JSON root 非 object 时抛出 ``ValueError``。解析成功不等于 G5 通过。
        """
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("Candidate manifest JSON must contain an object")
        return cls.from_dict(parsed)


@dataclass(frozen=True)
class ManifestGateResult:
    """G5 manifest gate 的 immutable result。

    ``passed`` 当且仅当 ``reasons`` 为空；``gate`` 当前固定为 ``G5``。
    """

    gate: str
    passed: bool
    reasons: tuple[str, ...]


class ManifestGateError(ValueError):
    """candidate 未通过 G5 manifest gate 时抛出的聚合错误。"""


def _content_digest(content: bytes | None) -> str | None:
    return hashlib.sha256(content).hexdigest() if content is not None else None


def _target_digests(
    target_files: tuple[str, ...],
    files: Mapping[str, bytes | None],
    *,
    field_name: str,
) -> tuple[str | None, ...]:
    expected = set(target_files)
    if len(expected) != len(target_files):
        raise ValueError("patch targets contain duplicates")
    actual = set(files)
    if actual != expected:
        raise ValueError(
            f"{field_name} paths do not match patch targets: expected={sorted(expected)}, actual={sorted(actual)}"
        )
    digests: list[str | None] = []
    for path in target_files:
        content = files[path]
        if content is not None and not isinstance(content, bytes):
            raise TypeError(f"{field_name}[{path!r}] must be bytes or None")
        digests.append(_content_digest(content))
    return tuple(digests)


def _patch_digest(
    patch: AppliedPatch,
    before_sha256: tuple[str | None, ...],
    after_sha256: tuple[str | None, ...],
) -> str:
    payload = json.dumps(
        {
            "patch": patch.to_dict(),
            "before_sha256": before_sha256,
            "after_sha256": after_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def manifest_for_patch(
    candidate_id: str,
    label: CandidateLabel | str,
    patch: AppliedPatch,
    *,
    before_files: Mapping[str, bytes | None],
    after_files: Mapping[str, bytes | None],
) -> CandidateManifest:
    """为 ``AppliedPatch`` 与 before/after snapshot 构造 canonical manifest。

    ``candidate_id`` 不能为空；label 解析后读取 canonical policy。每个 patch component 的
    ``target_file`` 决定顺序，before/after mapping 必须精确覆盖且无 duplicate target，内容
    必须为 bytes 或 None。函数计算 per-file SHA-256 与包含 patch structure 的 patch digest。

    返回 manifest 还未经过 G5；unsupported label 也能被描述，但不能通过 gate。
    """
    if not candidate_id.strip():
        raise ValueError("candidate_id must not be empty")
    parsed_label = CandidateLabel(label)
    policy = LABEL_POLICIES[parsed_label]
    target_files = tuple(component.target_file for component in patch.components)
    before_sha256 = _target_digests(
        target_files,
        before_files,
        field_name="before_files",
    )
    after_sha256 = _target_digests(
        target_files,
        after_files,
        field_name="after_files",
    )
    return CandidateManifest(
        schema_version=1,
        candidate_id=candidate_id,
        label=parsed_label,
        patch_where=patch.patch_where,
        target_files=target_files,
        before_sha256=before_sha256,
        after_sha256=after_sha256,
        patch_digest=_patch_digest(patch, before_sha256, after_sha256),
        fixture=policy.fixture,
        evaluator=policy.evaluator,
        activation_policy=policy.activation_policy,
    )


def _path_matches(path: str, pattern: str) -> bool:
    if pattern.endswith("/"):
        return path.startswith(pattern)
    return path == pattern


def _is_config_path(path: str) -> bool:
    return Path(path).suffix.lower() in {".json", ".toml", ".yaml", ".yml"}


def _looks_like_weights(path: str) -> bool:
    suffix = Path(path).suffix.lower()
    return suffix in {".bin", ".ckpt", ".gguf", ".onnx", ".pt", ".pth", ".safetensors"}


def evaluate_manifest_gate(
    manifest: CandidateManifest,
    patch: AppliedPatch,
    *,
    before_files: Mapping[str, bytes | None],
    after_files: Mapping[str, bytes | None],
    repo_root: str | Path | None = None,
    treeish: str | None = None,
) -> ManifestGateResult:
    """重新计算 candidate 的 G5 manifest gate。

    验证 schema/candidate ID、label supported/binding、PatchWhere、target order、before/after
    digest、content change、patch digest、fixture/evaluator/activation policy、immutable/symlink
    path、label mutable allowlist、config-only 与 no-model-weights 规则。``repo_root``/``treeish``
    控制 path guard 检查哪个 tree。

    所有 violation 聚合进 ``reasons``，不会在首项失败时抛出；返回 ``passed=not reasons``。
    该结果不运行 benchmark evaluator。
    """
    reasons: list[str] = []
    policy = LABEL_POLICIES[manifest.label]
    patch_targets = tuple(component.target_file for component in patch.components)
    try:
        before_sha256 = _target_digests(
            patch_targets,
            before_files,
            field_name="before_files",
        )
        after_sha256 = _target_digests(
            patch_targets,
            after_files,
            field_name="after_files",
        )
    except (TypeError, ValueError) as exc:
        before_sha256 = ()
        after_sha256 = ()
        reasons.append(str(exc))

    if manifest.schema_version != 1:
        reasons.append(f"unsupported manifest schema version: {manifest.schema_version}")
    if not manifest.candidate_id.strip():
        reasons.append("candidate_id is empty")
    if not policy.supported:
        reasons.append(policy.unsupported_reason or f"label {manifest.label.value!r} is unsupported")
    else:
        from codeflow.evolver.candidate_evidence import (
            EVALUATOR_BINDINGS,
            FIXTURE_BINDINGS,
        )

        if policy.fixture not in FIXTURE_BINDINGS:
            reasons.append(f"label {manifest.label.value!r} has no executable fixture binding")
        if policy.evaluator not in EVALUATOR_BINDINGS:
            reasons.append(f"label {manifest.label.value!r} has no executable evaluator binding")
    if manifest.patch_where != patch.patch_where:
        reasons.append("manifest patch_where does not match the candidate patch")
    if patch.patch_where not in policy.patch_where:
        reasons.append(f"label {manifest.label.value!r} does not allow PatchWhere {patch.patch_where.value!r}")
    if manifest.target_files != patch_targets:
        reasons.append("manifest target_files do not match the candidate patch")
    if manifest.before_sha256 != before_sha256:
        reasons.append("manifest before_sha256 does not match the candidate parent bytes")
    if manifest.after_sha256 != after_sha256:
        reasons.append("manifest after_sha256 does not match the candidate result bytes")
    for target in patch_targets:
        if target in before_files and target in after_files and before_files[target] == after_files[target]:
            reasons.append(f"candidate target has no content change: {target}")
    if manifest.patch_digest != _patch_digest(patch, before_sha256, after_sha256):
        reasons.append("manifest patch_digest does not match the candidate patch")
    if manifest.fixture != policy.fixture:
        reasons.append("manifest fixture does not match the canonical label policy")
    if manifest.evaluator != policy.evaluator:
        reasons.append("manifest evaluator does not match the canonical label policy")
    if manifest.activation_policy != policy.activation_policy:
        reasons.append("manifest activation policy does not match the canonical label policy")

    unsafe = check_patch_paths(
        patch_targets,
        repo_root=repo_root,
        treeish=treeish,
    )
    if unsafe:
        reasons.append(f"candidate targets unsafe or immutable paths: {unsafe}")
    for target in patch_targets:
        if policy.mutable_paths and not any(_path_matches(target, pattern) for pattern in policy.mutable_paths):
            reasons.append(f"target is outside the {manifest.label.value} mutable allowlist: {target}")
        if policy.config_only and not _is_config_path(target):
            reasons.append(f"{manifest.label.value} candidates may modify configuration files only: {target}")
        if manifest.label in {CandidateLabel.model_profile, CandidateLabel.route} and _looks_like_weights(target):
            reasons.append(f"{manifest.label.value} candidates may never modify model weights: {target}")

    return ManifestGateResult(gate="G5", passed=not reasons, reasons=tuple(reasons))


def assert_manifest_gate(
    manifest: CandidateManifest,
    patch: AppliedPatch,
    *,
    before_files: Mapping[str, bytes | None],
    after_files: Mapping[str, bytes | None],
    repo_root: str | Path | None = None,
    treeish: str | None = None,
) -> None:
    """要求 candidate 通过 G5，否则抛出 ``ManifestGateError``。

    参数与 ``evaluate_manifest_gate`` 相同。通过时无返回值；失败 message 合并全部 reason，
    便于 candidate 诊断。函数不应用 patch。
    """
    result = evaluate_manifest_gate(
        manifest,
        patch,
        before_files=before_files,
        after_files=after_files,
        repo_root=repo_root,
        treeish=treeish,
    )
    if not result.passed:
        raise ManifestGateError("G5 manifest gate failed: " + "; ".join(result.reasons))


__all__ = [
    "ActivationPolicy",
    "CandidateLabel",
    "CandidateManifest",
    "LABEL_POLICIES",
    "LabelPolicy",
    "ManifestGateError",
    "ManifestGateResult",
    "assert_manifest_gate",
    "evaluate_manifest_gate",
    "manifest_for_patch",
]
