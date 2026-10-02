"""从 durable artifacts 重建 deterministic Evolution Run summary。

summary 不信任单一 ledger。它合并 round journal、per-node JSON 与 verified activation bundle，
检查 candidate ID/filename、显式 outcome、status type、journal-node outcome、Git commit identity、
manifest、accepted activation presence 等 cross-source integrity。任何损坏或矛盾 candidate 都
归入 ``failed``，同时保留结构化 ``integrity_errors``，避免把缺失证据默认为成功。

``build_evolution_summary`` 在 resume 与 finalize 都从磁盘重算相同结果，按
``accepted/rejected/failed/inconclusive`` 与 activation state 分组排序；
``write_evolution_summary`` 以 canonical JSON atomic write。summary 是 evidence index，不重跑
benchmark，也不能把 ``accepted``、``activated`` 与任务交付成功或 sealed 正向结论混为一谈。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from codeflow.evolver.activation.artifacts import (
    ACTIVATION_DIRNAME,
    ActivationState,
    EvidenceOutcome,
    _atomic_write,
    _canonical_json,
    verify_activation_artifacts,
)

SCHEMA_VERSION = 1
SUMMARY_FILENAME = "evolution_summary.json"

_OUTCOMES = tuple(outcome.value for outcome in EvidenceOutcome)
_ACTIVATION_STATES = tuple(state.value for state in ActivationState)
_ACCEPTED_STATUSES = frozenset({"promoted_to_baseline"})
_REJECTED_STATUSES = frozenset(
    {
        "pruned_low_score",
        "pruned_inert",
        "pruned_at_screen",
        "pruned_at_confirm",
        "rejected_at_manifest",
    }
)
_INCONCLUSIVE_STATUSES = frozenset({"active", "blocked_l1", "archived-methodology-failure"})
_INFRA_FAILURE_CLASSES = frozenset({"provider", "infrastructure"})


def _integrity_error(
    *,
    source: str,
    path: str,
    candidate_id: str,
    error: str,
) -> dict[str, str]:
    return {
        "source": source,
        "path": path,
        "candidate_id": candidate_id,
        "error": error,
    }


def _load_dict(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text())
    except OSError as exc:
        return None, f"unreadable JSON: {type(exc).__name__}"
    except ValueError:
        return None, "invalid JSON"
    if not isinstance(value, dict):
        return None, "JSON root is not an object"
    return value, None


def _failure_class(record: dict[str, Any]) -> str | None:
    for key in ("failure_class", "error_class"):
        value = record.get(key)
        if value:
            return str(value).strip().lower()
    stats = record.get("stats")
    if isinstance(stats, dict):
        for key in ("failure_class", "error_class"):
            value = stats.get(key)
            if value:
                return str(value).strip().lower()
    return None


def _classify(record: dict[str, Any]) -> str:
    explicit = record.get("evidence_outcome") or record.get("verdict")
    if explicit in _OUTCOMES:
        return str(explicit)
    status = str(record.get("status") or "")
    if status in _ACCEPTED_STATUSES:
        return EvidenceOutcome.accepted.value
    if status in _REJECTED_STATUSES:
        return EvidenceOutcome.rejected.value
    if _failure_class(record) in _INFRA_FAILURE_CLASSES:
        return EvidenceOutcome.failed.value
    if status in _INCONCLUSIVE_STATUSES:
        return EvidenceOutcome.inconclusive.value
    if status == "errored":
        return EvidenceOutcome.failed.value
    return EvidenceOutcome.inconclusive.value


def _explicit_outcome(record: dict[str, Any]) -> str | None:
    explicit = record.get("evidence_outcome") or record.get("verdict")
    return str(explicit) if explicit in _OUTCOMES else None


def _node_records(
    work_dir: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, str]],
    set[str],
]:
    records: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    failed_ids: set[str] = set()
    root = work_dir / "nodes"
    for path in sorted(root.glob("*.json")):
        relative = path.relative_to(work_dir).as_posix()
        fallback_id = path.stem
        record, error = _load_dict(path)
        if error is not None:
            failed_ids.add(fallback_id)
            errors.append(
                _integrity_error(
                    source="node",
                    path=relative,
                    candidate_id=fallback_id,
                    error=error,
                )
            )
            continue
        candidate_id = record.get("node_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            failed_ids.add(fallback_id)
            errors.append(
                _integrity_error(
                    source="node",
                    path=relative,
                    candidate_id=fallback_id,
                    error="missing node_id",
                )
            )
            continue
        if candidate_id != fallback_id:
            failed_ids.update({fallback_id, candidate_id})
            errors.append(
                _integrity_error(
                    source="node",
                    path=relative,
                    candidate_id=candidate_id,
                    error="node_id does not match filename",
                )
            )
            continue
        invalid_outcome_field = next(
            (
                field
                for field in ("evidence_outcome", "verdict")
                if record.get(field) is not None and record.get(field) not in _OUTCOMES
            ),
            None,
        )
        if invalid_outcome_field is not None:
            failed_ids.add(candidate_id)
            errors.append(
                _integrity_error(
                    source="node",
                    path=relative,
                    candidate_id=candidate_id,
                    error=f"node {invalid_outcome_field} is invalid",
                )
            )
            continue
        status = record.get("status")
        if status is not None and not isinstance(status, str):
            failed_ids.add(candidate_id)
            errors.append(
                _integrity_error(
                    source="node",
                    path=relative,
                    candidate_id=candidate_id,
                    error="node status must be a string",
                )
            )
            continue
        records[candidate_id] = record
    return records, errors, failed_ids


def _journal_error_id(
    relative: str,
    *,
    line: int | None = None,
    candidate_index: int | None = None,
) -> str:
    identifier = f"journal:{relative}"
    if line is not None:
        identifier = f"{identifier}:line-{line}"
    if candidate_index is not None:
        identifier = f"{identifier}:candidate-{candidate_index}"
    return identifier


def _journal_records(
    work_dir: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, str]],
    set[str],
]:
    records: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    failed_ids: set[str] = set()
    for path in sorted((work_dir / "journal").glob("*.jsonl")):
        relative = path.relative_to(work_dir).as_posix()
        try:
            lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        except OSError as exc:
            candidate_id = _journal_error_id(relative)
            failed_ids.add(candidate_id)
            errors.append(
                _integrity_error(
                    source="journal",
                    path=relative,
                    candidate_id=candidate_id,
                    error=f"unreadable journal: {type(exc).__name__}",
                )
            )
            continue
        for index, line in enumerate(lines):
            line_number = index + 1
            try:
                round_record = json.loads(line)
            except ValueError:
                if index == len(lines) - 1:
                    break
                candidate_id = _journal_error_id(
                    relative,
                    line=line_number,
                )
                failed_ids.add(candidate_id)
                errors.append(
                    _integrity_error(
                        source="journal",
                        path=relative,
                        candidate_id=candidate_id,
                        error=f"invalid JSON at line {line_number}",
                    )
                )
                continue
            if not isinstance(round_record, dict):
                candidate_id = _journal_error_id(
                    relative,
                    line=line_number,
                )
                failed_ids.add(candidate_id)
                errors.append(
                    _integrity_error(
                        source="journal",
                        path=relative,
                        candidate_id=candidate_id,
                        error=f"record at line {line_number} is not an object",
                    )
                )
                continue
            candidates = round_record.get("candidates")
            if not isinstance(candidates, list):
                candidate_id = _journal_error_id(
                    relative,
                    line=line_number,
                )
                failed_ids.add(candidate_id)
                errors.append(
                    _integrity_error(
                        source="journal",
                        path=relative,
                        candidate_id=candidate_id,
                        error=(f"record at line {line_number} has invalid candidates"),
                    )
                )
                continue
            for candidate_index, candidate in enumerate(candidates):
                fallback_id = _journal_error_id(
                    relative,
                    line=line_number,
                    candidate_index=candidate_index,
                )
                if not isinstance(candidate, dict):
                    failed_ids.add(fallback_id)
                    errors.append(
                        _integrity_error(
                            source="journal",
                            path=relative,
                            candidate_id=fallback_id,
                            error=(f"candidate {candidate_index} at line {line_number} is not an object"),
                        )
                    )
                    continue
                candidate_id = candidate.get("node_id")
                if not isinstance(candidate_id, str) or not candidate_id:
                    failed_ids.add(fallback_id)
                    errors.append(
                        _integrity_error(
                            source="journal",
                            path=relative,
                            candidate_id=fallback_id,
                            error=(f"candidate {candidate_index} at line {line_number} has invalid node_id"),
                        )
                    )
                    continue
                status = candidate.get("status")
                verdict = candidate.get("verdict")
                if status is not None and not isinstance(status, str):
                    failed_ids.add(candidate_id)
                    errors.append(
                        _integrity_error(
                            source="journal",
                            path=relative,
                            candidate_id=candidate_id,
                            error="candidate status must be a string",
                        )
                    )
                    continue
                if verdict is not None and verdict not in _OUTCOMES:
                    failed_ids.add(candidate_id)
                    errors.append(
                        _integrity_error(
                            source="journal",
                            path=relative,
                            candidate_id=candidate_id,
                            error="candidate verdict is invalid",
                        )
                    )
                    continue
                records[candidate_id] = candidate
    return records, errors, failed_ids


def _ledger_merge_errors(
    *,
    work_dir: Path,
    journal_records: dict[str, dict[str, Any]],
    node_records: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, str]], set[str]]:
    errors: list[dict[str, str]] = []
    failed_ids: set[str] = set()
    for candidate_id in sorted(journal_records.keys() & node_records.keys()):
        journal_outcome = _explicit_outcome(journal_records[candidate_id])
        node_outcome = _explicit_outcome(node_records[candidate_id])
        if journal_outcome is None or node_outcome is None or journal_outcome == node_outcome:
            continue
        failed_ids.add(candidate_id)
        errors.append(
            _integrity_error(
                source="ledger",
                path=(work_dir / "nodes" / f"{candidate_id}.json").relative_to(work_dir).as_posix(),
                candidate_id=candidate_id,
                error="node evidence outcome does not match journal ledger",
            )
        )
    return errors, failed_ids


def _activation_records(
    work_dir: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, str]],
    set[str],
]:
    records: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    failed_ids: set[str] = set()
    root = work_dir / ACTIVATION_DIRNAME
    if root.is_symlink():
        errors.append(
            _integrity_error(
                source="activation",
                path=ACTIVATION_DIRNAME,
                candidate_id="activation:root",
                error="activation root must not be a symlink",
            )
        )
        return records, errors, failed_ids
    if not root.is_dir():
        return records, errors, failed_ids

    for artifact_dir in sorted(root.iterdir(), key=lambda path: path.name):
        candidate_id = artifact_dir.name
        relative = artifact_dir.relative_to(work_dir).as_posix()
        if artifact_dir.is_symlink():
            failed_ids.add(candidate_id)
            errors.append(
                _integrity_error(
                    source="activation",
                    path=relative,
                    candidate_id=candidate_id,
                    error="activation candidate directory must not be a symlink",
                )
            )
            continue
        if not artifact_dir.is_dir():
            continue
        try:
            record = verify_activation_artifacts(artifact_dir)
        except (OSError, TypeError, ValueError) as exc:
            failed_ids.add(candidate_id)
            message = str(exc).replace(str(work_dir), "<work_dir>")
            errors.append(
                _integrity_error(
                    source="activation",
                    path=relative,
                    candidate_id=candidate_id,
                    error=message,
                )
            )
            continue
        records[candidate_id] = record
    return records, errors, failed_ids


def _external_activation_errors(
    *,
    work_dir: Path,
    activation_records: dict[str, dict[str, Any]],
    node_records: dict[str, dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, str]], set[str]]:
    errors: list[dict[str, str]] = []
    failed_ids: set[str] = set()
    for candidate_id, activation in sorted(activation_records.items()):
        relative = (Path(ACTIVATION_DIRNAME) / candidate_id).as_posix()
        node = node_records.get(candidate_id)
        if activation["evidence_outcome"] == EvidenceOutcome.accepted.value and node is None:
            failed_ids.add(candidate_id)
            errors.append(
                _integrity_error(
                    source="activation",
                    path=relative,
                    candidate_id=candidate_id,
                    error="accepted activation has no matching node ledger",
                )
            )
        if node is not None:
            node_sha = node.get("git_commit_sha")
            candidate_sha = activation.get("candidate_sha")
            if (activation["evidence_outcome"] == EvidenceOutcome.accepted.value and node_sha != candidate_sha) or (
                candidate_sha is not None and node_sha is not None and candidate_sha != node_sha
            ):
                failed_ids.add(candidate_id)
                errors.append(
                    _integrity_error(
                        source="activation",
                        path=relative,
                        candidate_id=candidate_id,
                        error="candidate_sha does not match node git_commit_sha",
                    )
                )
            node_manifest = (
                (node.get("candidate") or {}).get("manifest") if isinstance(node.get("candidate"), dict) else None
            )
            if activation["evidence_outcome"] == EvidenceOutcome.accepted.value and node_manifest is None:
                failed_ids.add(candidate_id)
                errors.append(
                    _integrity_error(
                        source="activation",
                        path=relative,
                        candidate_id=candidate_id,
                        error="accepted activation has no node candidate manifest",
                    )
                )
            if node_manifest is not None:
                manifest_path = work_dir / relative / "candidate_manifest.json"
                manifest, manifest_error = _load_dict(manifest_path)
                if manifest_error is not None or manifest != node_manifest:
                    failed_ids.add(candidate_id)
                    errors.append(
                        _integrity_error(
                            source="activation",
                            path=relative,
                            candidate_id=candidate_id,
                            error=("candidate manifest does not match node ledger"),
                        )
                    )

        prior = candidates.get(candidate_id)
        prior_outcome = _classify(prior) if prior is not None else None
        activation_outcome = activation["evidence_outcome"]
        if prior_outcome is not None and prior_outcome != activation_outcome:
            failed_ids.add(candidate_id)
            errors.append(
                _integrity_error(
                    source="activation",
                    path=relative,
                    candidate_id=candidate_id,
                    error=("activation evidence outcome does not match journal or node ledger"),
                )
            )
    return errors, failed_ids


def _missing_activation_errors(
    *,
    candidates: dict[str, dict[str, Any]],
    activation_records: dict[str, dict[str, Any]],
    failed_activation_ids: set[str],
) -> tuple[list[dict[str, str]], set[str]]:
    errors: list[dict[str, str]] = []
    failed_ids: set[str] = set()
    for candidate_id, record in sorted(candidates.items()):
        if (
            _classify(record) != EvidenceOutcome.accepted.value
            or candidate_id in activation_records
            or candidate_id in failed_activation_ids
        ):
            continue
        failed_ids.add(candidate_id)
        relative = (Path(ACTIVATION_DIRNAME) / candidate_id).as_posix()
        errors.append(
            _integrity_error(
                source="activation",
                path=relative,
                candidate_id=candidate_id,
                error=("accepted candidate has no verified activation bundle"),
            )
        )
    return errors, failed_ids


def build_evolution_summary(work_dir: Path | str) -> dict[str, Any]:
    """从 ``work_dir`` durable state 重建 resume/finalize 共用 summary。

    函数依次读取 journal、node 与 activation records，验证 source 之间的 outcome、candidate
    SHA、manifest 和 required bundle 关系。journal 最后一行若是 partial invalid JSON 被视为
    crash residue 并忽略；中间损坏、shape 错误或 cross-ledger mismatch 都形成 integrity error。

    failed integrity ID 会覆盖为 ``EvidenceOutcome.failed``；其余 candidate 按 explicit outcome
    或 status/failure_class 保守分类。返回 schema version、各 outcome/state count 与排序 ID，
    以及确定性排序的 error list。调用不修改源 artifact。
    """

    work_dir = Path(work_dir)
    candidates, journal_errors, failed_journal_ids = _journal_records(work_dir)
    node_records, node_errors, failed_node_ids = _node_records(work_dir)
    ledger_errors, failed_ledger_ids = _ledger_merge_errors(
        work_dir=work_dir,
        journal_records=candidates,
        node_records=node_records,
    )
    for candidate_id, node_record in node_records.items():
        merged = dict(candidates.get(candidate_id, {}))
        merged.update(node_record)
        candidates[candidate_id] = merged

    activation_records, activation_errors, failed_activation_ids = _activation_records(work_dir)
    external_errors, failed_external_ids = _external_activation_errors(
        work_dir=work_dir,
        activation_records=activation_records,
        node_records=node_records,
        candidates=candidates,
    )
    missing_activation_errors, failed_missing_activation_ids = _missing_activation_errors(
        candidates=candidates,
        activation_records=activation_records,
        failed_activation_ids=failed_activation_ids,
    )
    for candidate_id, activation in activation_records.items():
        merged = dict(candidates.get(candidate_id, {}))
        merged.update(
            {
                "node_id": candidate_id,
                "evidence_outcome": activation["evidence_outcome"],
            }
        )
        candidates[candidate_id] = merged

    failed_integrity_ids = (
        failed_journal_ids
        | failed_node_ids
        | failed_ledger_ids
        | failed_activation_ids
        | failed_external_ids
        | failed_missing_activation_ids
    )
    for candidate_id in failed_integrity_ids:
        merged = dict(candidates.get(candidate_id, {}))
        merged.update(
            {
                "node_id": candidate_id,
                "evidence_outcome": EvidenceOutcome.failed.value,
            }
        )
        candidates[candidate_id] = merged

    outcomes = {name: [] for name in _OUTCOMES}
    for candidate_id, record in candidates.items():
        outcomes[_classify(record)].append(candidate_id)
    for candidate_ids in outcomes.values():
        candidate_ids.sort()

    activation_states = {name: [] for name in _ACTIVATION_STATES}
    for candidate_id, record in activation_records.items():
        if candidate_id in failed_integrity_ids:
            continue
        state = record["state"]
        activation_states[state].append(candidate_id)
    for candidate_ids in activation_states.values():
        candidate_ids.sort()

    integrity_errors = sorted(
        [
            *journal_errors,
            *node_errors,
            *ledger_errors,
            *activation_errors,
            *external_errors,
            *missing_activation_errors,
        ],
        key=lambda item: (
            item["candidate_id"],
            item["source"],
            item["path"],
            item["error"],
        ),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "outcome_counts": {name: len(outcomes[name]) for name in _OUTCOMES},
        "outcomes": outcomes,
        "activation_counts": {name: len(activation_states[name]) for name in _ACTIVATION_STATES},
        "activation_states": activation_states,
        "integrity_error_count": len(integrity_errors),
        "integrity_errors": integrity_errors,
    }


def write_evolution_summary(
    work_dir: Path | str,
    output_path: Path | str | None = None,
) -> Path:
    """构建 summary，并以 canonical JSON atomic write 到目标路径。

    ``output_path`` 缺失时使用 ``work_dir/evolution_summary.json``。函数返回实际 path；返回
    成功表示 summary file 已写入，不表示 ``integrity_error_count`` 为 0，调用方仍需检查内容。
    """
    work_dir = Path(work_dir)
    path = Path(output_path) if output_path is not None else work_dir / SUMMARY_FILENAME
    _atomic_write(path, _canonical_json(build_evolution_summary(work_dir)))
    return path


__all__ = [
    "SCHEMA_VERSION",
    "SUMMARY_FILENAME",
    "build_evolution_summary",
    "write_evolution_summary",
]
