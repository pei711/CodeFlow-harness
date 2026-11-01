"""Deterministic, evidence-aware scoring for CodeFlow raw span logs."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def read_spans(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Load audit-spans JSONL, returning valid records and malformed line count."""
    spans: list[dict[str, Any]] = []
    malformed = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(value, dict):
                spans.append(value)
            else:
                malformed += 1
    return spans, malformed


def score_trace(spans_or_path: Iterable[dict[str, Any]] | Path) -> dict[str, Any]:
    """Score observable trace signals; absent evidence remains ``None``/unknown.

    Repeated reads require the same tool, normalized path/range arguments and
    identical output artifact digest. This avoids treating a reread after a file
    changed as redundant. It is a diagnostic signal, not a penalty by itself.
    """
    malformed = 0
    if isinstance(spans_or_path, Path):
        spans, malformed = read_spans(spans_or_path)
    else:
        spans = list(spans_or_path)
    tool_spans = [s for s in spans if _attrs(s).get("tool.name")]
    llm_spans = [s for s in spans if _attrs(s).get("llm.model") or _has_usage(_attrs(s))]

    read_identities: Counter[str] = Counter()
    failures = 0
    tool_missing_failure_signal = not tool_spans
    verification_calls = 0
    for span in tool_spans:
        attrs = _attrs(span)
        name = str(attrs.get("tool.name", "")).lower()
        if _is_read_tool(name):
            identity = _read_identity(attrs)
            if identity:
                read_identities[identity] += 1
        if _tool_failed(span, attrs):
            failures += 1
        command_text = _command_text(span, attrs).lower()
        if _is_verification_command(command_text):
            verification_calls += 1

    usage_fields_present = [
        isinstance(_attrs(span).get("llm.usage.input_tokens"), int)
        and isinstance(_attrs(span).get("llm.usage.output_tokens"), int)
        for span in llm_spans
    ]
    usage_complete = bool(llm_spans) and all(usage_fields_present)
    input_tokens = sum(
        int(_attrs(s)["llm.usage.input_tokens"])
        for s in llm_spans
        if isinstance(_attrs(s).get("llm.usage.input_tokens"), int)
    )
    output_tokens = sum(
        int(_attrs(s)["llm.usage.output_tokens"])
        for s in llm_spans
        if isinstance(_attrs(s).get("llm.usage.output_tokens"), int)
    )
    compact_spans = [s for s in spans if _is_compaction_span(s)]

    verifier_evidence = _verifier_evidence(spans)
    resume_evidence = _resume_evidence(spans)
    tool_metrics = {
        "call_count": len(tool_spans),
        "failed_calls": failures if tool_spans else None,
        "failure_rate": failures / len(tool_spans) if tool_spans else None,
        "repeated_identical_file_reads": sum(max(0, count - 1) for count in read_identities.values())
        if read_identities
        else None,
        "unique_file_read_identities": len(read_identities) if read_identities else None,
    }
    context_metrics = {
        "input_tokens": input_tokens if usage_complete else None,
        "output_tokens": output_tokens if usage_complete else None,
        "total_tokens": input_tokens + output_tokens if usage_complete else None,
        "compaction_span_count": len(compact_spans) if compact_spans else None,
    }
    return {
        "schema": "codeflow.trajectory-score.v1",
        "evidence": {
            "span_count": len(spans),
            "malformed_span_lines": malformed,
            "tool_span_count": len(tool_spans),
            "llm_span_count": len(llm_spans),
            "usage_complete": usage_complete,
            "tool_failure_signal_present": not tool_missing_failure_signal,
            "verifier_signal_present": verifier_evidence is not None,
            "resume_signal_present": resume_evidence is not None,
        },
        "tools": tool_metrics,
        "context": context_metrics,
        "execution": {
            "verification_command_calls": verification_calls if tool_spans else None,
            "verifier_result": verifier_evidence,
            "resume_result": resume_evidence,
        },
        "dimensions": {
            "tool_efficiency": {
                "evidence_state": "observed" if tool_spans else "unknown",
                "metrics": {"repeated_identical_file_reads": tool_metrics["repeated_identical_file_reads"]},
            },
            "tool_reliability": {
                "evidence_state": "observed" if tool_spans else "unknown",
                "metrics": {"failed_calls": tool_metrics["failed_calls"], "failure_rate": tool_metrics["failure_rate"]},
            },
            "context_efficiency": {
                "evidence_state": "complete" if usage_complete else "partial" if llm_spans else "unknown",
                "metrics": {
                    "total_tokens": context_metrics["total_tokens"],
                    "compaction_span_count": context_metrics["compaction_span_count"],
                },
            },
            "execution_completeness": {
                "evidence_state": "explicit"
                if verifier_evidence is not None
                else "partial"
                if verification_calls
                else "unknown",
                "metrics": {
                    "verification_command_calls": verification_calls if tool_spans else None,
                    "verifier_result": verifier_evidence,
                },
            },
            "task_recovery": {
                "evidence_state": "explicit" if resume_evidence is not None else "unknown",
                "metrics": {"resume_result": resume_evidence},
            },
        },
        "findings": _findings(spans, malformed, usage_complete, verifier_evidence, resume_evidence),
    }


def _attrs(span: dict[str, Any]) -> dict[str, Any]:
    attrs = span.get("attributes")
    return attrs if isinstance(attrs, dict) else {}


def _has_usage(attrs: dict[str, Any]) -> bool:
    return "llm.usage.input_tokens" in attrs or "llm.usage.output_tokens" in attrs


def _is_read_tool(name: str) -> bool:
    return name in {"read_file", "read", "cat", "get_file"} or name.endswith(".read_file")


def _read_identity(attrs: dict[str, Any]) -> str | None:
    payload = _artifact_json(attrs.get("tool.input.artifact_path"))
    arguments = payload.get("params", payload) if isinstance(payload, dict) else {}
    if not isinstance(arguments, dict):
        arguments = {}
    path = next((arguments[k] for k in ("path", "file_path", "filename") if isinstance(arguments.get(k), str)), None)
    if path is None:
        path = attrs.get("tool.input.path")
    if not isinstance(path, str):
        return None
    normalized_path = re.sub(r"[\\/]+", "/", path).casefold()
    ranges = {key: arguments[key] for key in ("start_line", "end_line", "offset", "limit", "line") if key in arguments}
    digest = attrs.get("tool.output.artifact_sha1") or attrs.get("tool.output.artifact_sha256")
    if not isinstance(digest, str):
        output = _artifact_json(attrs.get("tool.output.artifact_path"))
        digest = (
            hashlib.sha256(json.dumps(output, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()
            if output
            else None
        )
    if not digest:
        return None
    return json.dumps([normalized_path, ranges, digest], sort_keys=True, ensure_ascii=False)


def _artifact_json(value: Any) -> Any:
    if not isinstance(value, str):
        return None
    try:
        path = Path(value)
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    return None


def _tool_failed(span: dict[str, Any], attrs: dict[str, Any]) -> bool:
    status = span.get("status")
    status_code = status.get("code") if isinstance(status, dict) else status
    return bool(
        attrs.get("tool.failed") is True or attrs.get("error.type") or status_code == "ERROR" or attrs.get("tool.error")
    )


def _command_text(span: dict[str, Any], attrs: dict[str, Any]) -> str:
    payload = _artifact_json(attrs.get("tool.input.artifact_path"))
    return (
        json.dumps(payload, ensure_ascii=False, default=str)
        if payload is not None
        else str(attrs.get("tool.input_preview", ""))
    )


def _is_verification_command(command: str) -> bool:
    return bool(
        re.search(
            r"\b(pytest|unittest|tox|nox|mypy|ruff|flake8|cargo\s+test|go\s+test|npm\s+test|pnpm\s+test|make\s+test)\b",
            command,
        )
    )


def _is_compaction_span(span: dict[str, Any]) -> bool:
    name = str(span.get("name", "")).lower()
    attrs = _attrs(span)
    return any(word in name for word in ("compact", "compaction", "history.trim", "context.trim")) or isinstance(
        attrs.get("context.compact_count"), int
    )


def _verifier_evidence(spans: list[dict[str, Any]]) -> bool | None:
    for span in spans:
        attrs = _attrs(span)
        for key in ("verification.passed", "verifier.passed", "task.verifier_passed"):
            if isinstance(attrs.get(key), bool):
                return attrs[key]
    return None


def _resume_evidence(spans: list[dict[str, Any]]) -> bool | None:
    for span in spans:
        attrs = _attrs(span)
        for key in ("checkpoint.resume_success", "recovery.resume_success", "task.resume_passed"):
            if isinstance(attrs.get(key), bool):
                return attrs[key]
    return None


def _findings(
    spans: list[dict[str, Any]], malformed: int, usage_complete: bool, verifier: bool | None, resume: bool | None
) -> list[str]:
    findings: list[str] = []
    if not spans:
        findings.append("trace_missing")
    if malformed:
        findings.append("trace_contains_malformed_lines")
    if spans and not usage_complete:
        findings.append("token_usage_incomplete")
    if verifier is None:
        findings.append("verifier_outcome_not_in_trace; use task-level external verifier")
    if resume is None:
        findings.append("resume_outcome_not_in_trace")
    return findings
