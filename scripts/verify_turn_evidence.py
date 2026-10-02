#!/usr/bin/env python3
"""V-TE0: prove that one Turn's tracing, usage, and delivery evidence joins.

Runs the deterministic scenario in ``tests/test_turn_evidence_scenario.py``
inside a temporary evidence root, then validates the artifacts it produced.
Every validation is a pure function over parsed records, so each detector is
unit-tested against corrupted fixtures in ``tests/test_verify_turn_evidence.py``
rather than only against a healthy run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA = "codeflow.turn.evidence.v1"
GATE = "V-TE0"
RERUN = "make verify-turn-evidence"
REPORT_FILENAME = "turn-evidence-report.json"
MANIFEST_FILENAME = "turn-evidence-manifest.json"
SCENARIO_TEST = "tests/test_turn_evidence_scenario.py"
CALL_EFFICIENCY_HEALTH_SCHEMA = "codeflow.call-efficiency.ledger-health.v1"

PASSED = "passed"
FAILED = "failed"
SKIPPED = "skipped"
INCONCLUSIVE = "inconclusive"
PROVIDER_FAILURE = "provider_failure"
INFRASTRUCTURE_FAILURE = "infrastructure_failure"

DETERMINISTIC = "deterministic"
CONTRACT = "contract"

SPINE_ROOT = "spine.turn"
SESSION_TURN = "session.turn"
DELIVERY = "channel.deliver"
LEAF_SPANS = ("llm.call", "tool.call")

SPINE_OUTCOMES = (
    "completed",
    "completed_with_tool_failure",
    "provider_failed",
    "error",
    "cancelled",
)
_OK_OUTCOMES = {"completed", "completed_with_tool_failure"}
CHANNEL_OUTCOMES = ("delivered", "dropped", "no_outlet")

# 门自身的契约：场景必须把哪个轮次驱动到哪个终态。契约位于此处而非场景中，因此场景若静默
# 停止产生某状态，会让门失败而不是重定义契约。
EXPECTED_TERMINALS = {
    "scenario:completed": "completed",
    "scenario:tool_failure": "completed_with_tool_failure",
    "scenario:provider_failure": "provider_failed",
    "scenario:runner_error": "error",
    "scenario:cancelled": "cancelled",
    "scenario:delivery_exhausted": "completed",
}
# 结果必须两两不同的轮次，每个分类条目一个。刻意排除 ``scenario:delivery_exhausted``：
# 它是已完成轮次，区分证据是渠道结果，而非轮次结果。
TERMINAL_WITNESSES = (
    "scenario:completed",
    "scenario:tool_failure",
    "scenario:provider_failure",
    "scenario:runner_error",
    "scenario:cancelled",
)
EXPECTED_DELIVERIES = {
    "scenario:completed": "delivered",
    "scenario:tool_failure": "delivered",
    "scenario:delivery_exhausted": "dropped",
}

_COUNT_PATTERNS = {
    "passed": re.compile(r"(\d+) passed"),
    "failed": re.compile(r"(\d+) failed"),
    "errors": re.compile(r"(\d+) errors?"),
    "skipped": re.compile(r"(\d+) skipped"),
}


# --------------------------------------------------------------------------
# 加载
# --------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse a JSON-lines file; a missing file reads as empty, not as an error."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_usage_rows(telemetry_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pattern in ("usage-*.jsonl", "call-efficiency-*.jsonl"):
        for path in sorted(telemetry_dir.glob(pattern)):
            rows.extend(read_jsonl(path))
    return rows


def check_call_efficiency_health(telemetry_dir: Path) -> dict[str, Any]:
    path = telemetry_dir / "call-efficiency-ledger-health.json"
    if not path.exists():
        return _result([], observed=0, evidence_class=CONTRACT, unit="ledger health")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != CALL_EFFICIENCY_HEALTH_SCHEMA:
            raise ValueError("unsupported ledger health schema")
        status = payload.get("status")
        if status not in {"healthy", "degraded"}:
            raise ValueError("invalid ledger health status")
        counts = []
        for key in ("accepted_records", "persisted_records", "lost_records"):
            value = payload.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"invalid ledger health field: {key}")
            counts.append(value)
        accepted_records, persisted_records, lost_records = counts
        if persisted_records > accepted_records or lost_records != accepted_records - persisted_records:
            raise ValueError("inconsistent ledger health counts")
        if status == "healthy" and lost_records:
            raise ValueError("healthy ledger reports loss")
        ledgers = payload.get("ledgers")
        if ledgers is not None:
            if not isinstance(ledgers, dict) or not ledgers:
                raise ValueError("invalid ledger health entries")
            entry_counts = [0, 0, 0]
            entry_degraded = False
            for entry in ledgers.values():
                if not isinstance(entry, dict) or entry.get("status") not in {"healthy", "degraded"}:
                    raise ValueError("invalid ledger health entry")
                values = []
                for key in ("accepted_records", "persisted_records", "lost_records"):
                    value = entry.get(key)
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        raise ValueError(f"invalid ledger health entry field: {key}")
                    values.append(value)
                if values[1] > values[0] or values[2] != values[0] - values[1]:
                    raise ValueError("inconsistent ledger health entry counts")
                if entry["status"] == "healthy" and values[2]:
                    raise ValueError("healthy ledger entry reports loss")
                entry_counts = [left + right for left, right in zip(entry_counts, values, strict=True)]
                entry_degraded = entry_degraded or entry["status"] == "degraded"
            if entry_counts != counts or (status == "degraded") != (entry_degraded or lost_records > 0):
                raise ValueError("ledger health aggregate mismatch")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        findings = [
            _finding(
                "call_efficiency_ledger_health_invalid",
                "CallEfficiency ledger health artifact is invalid",
                error_type=type(exc).__name__,
            )
        ]
        return _result(findings, observed=1, evidence_class=CONTRACT, unit="ledger health")
    findings = []
    if status != "healthy" or lost_records:
        findings.append(
            _finding(
                "call_efficiency_ledger_degraded",
                "CallEfficiency reports incomplete persistence",
                accepted_records=payload.get("accepted_records"),
                persisted_records=payload.get("persisted_records"),
                lost_records=payload.get("lost_records"),
            )
        )
    return _result(findings, observed=1, evidence_class=CONTRACT, unit="ledger health")


def dedupe_spans(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse re-emitted spans, last write wins.

    A long root checkpoints itself mid-flight under the same span id so children
    have a root to group under while the turn is open; the final emit supersedes
    that draft.
    """
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        latest[(row.get("traceId"), row.get("spanId"))] = row
    return list(latest.values())


# --------------------------------------------------------------------------
# 索引
# --------------------------------------------------------------------------


def _finding(detector: str, detail: str, **fields: Any) -> dict[str, Any]:
    return {"detector": detector, "detail": detail, **fields}


def index_turns(spans: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group deduped spans by trace and summarize each Turn.

    Keyed by trace id because the trace *is* the Turn identity; a trace with no
    ``spine.turn`` is still indexed so the correlation check can report it.
    """
    by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for span in spans:
        by_trace[span.get("traceId")].append(span)

    turns: dict[str, dict[str, Any]] = {}
    for trace_id, rows in by_trace.items():
        roots = [r for r in rows if r.get("name") == SPINE_ROOT]
        root = roots[0] if roots else None
        attrs = (root or {}).get("attributes") or {}
        deliveries = [r for r in rows if r.get("name") == DELIVERY]
        turns[trace_id] = {
            "trace_id": trace_id,
            "conversation_id": attrs.get("spine.conversation_id"),
            "spine_span_id": (root or {}).get("spanId"),
            "root_count": len(roots),
            "outcome": attrs.get("spine.outcome"),
            "terminal_event": attrs.get("spine.terminal_event"),
            "error_class": attrs.get("spine.error_class"),
            "status": ((root or {}).get("status") or {}).get("code"),
            "session_turn_spans": sum(1 for r in rows if r.get("name") == SESSION_TURN),
            "llm_call_spans": sum(1 for r in rows if r.get("name") == "llm.call"),
            "tool_call_spans": sum(1 for r in rows if r.get("name") == "tool.call"),
            "delivery_outcomes": [(r.get("attributes") or {}).get("channel.outcome") for r in deliveries],
            "span_ids": {r.get("spanId") for r in rows},
            "spans": rows,
        }
    return turns


def _public_turn(turn: dict[str, Any], usage_rows: int) -> dict[str, Any]:
    """The report view: ids, labels, and counts only -- never message content."""
    return {
        "conversation_id": turn["conversation_id"],
        "trace_id": turn["trace_id"],
        "spine_span_id": turn["spine_span_id"],
        "outcome": turn["outcome"],
        "terminal_event": turn["terminal_event"],
        "error_class": turn["error_class"],
        "span_status": turn["status"],
        "session_turn_spans": turn["session_turn_spans"],
        "llm_call_spans": turn["llm_call_spans"],
        "tool_call_spans": turn["tool_call_spans"],
        "delivery_outcomes": turn["delivery_outcomes"],
        "usage_rows": usage_rows,
    }


# --------------------------------------------------------------------------
# 检测器
# --------------------------------------------------------------------------


def check_turn_correlation(turns: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Each Turn is one trace chaining spine.turn -> session.turn -> leaves."""
    findings: list[dict[str, Any]] = []
    for trace_id, turn in sorted(turns.items(), key=lambda kv: str(kv[0])):
        if turn["root_count"] == 0:
            findings.append(_finding("trace_without_spine_root", "no spine.turn span", trace_id=trace_id))
            continue
        if turn["root_count"] > 1:
            findings.append(
                _finding("duplicate_spine_root", f"{turn['root_count']} spine.turn spans", trace_id=trace_id)
            )
        if turn["session_turn_spans"] == 0:
            findings.append(_finding("turn_without_session_turn", "no session.turn span", trace_id=trace_id))
        if turn["llm_call_spans"] == 0:
            findings.append(_finding("turn_without_model_call", "no llm.call span", trace_id=trace_id))

        root_id = turn["spine_span_id"]
        session_ids = {r["spanId"] for r in turn["spans"] if r.get("name") == SESSION_TURN}
        for span in turn["spans"]:
            name = span.get("name")
            parent = span.get("parentSpanId")
            if name == SESSION_TURN and parent != root_id:
                findings.append(
                    _finding(
                        "session_turn_orphaned",
                        "session.turn does not parent onto spine.turn",
                        trace_id=trace_id,
                        span_id=span.get("spanId"),
                    )
                )
            elif name in LEAF_SPANS and parent not in session_ids:
                findings.append(
                    _finding(
                        "leaf_span_orphaned",
                        f"{name} does not parent onto a session.turn",
                        trace_id=trace_id,
                        span_id=span.get("spanId"),
                    )
                )
            elif name == SPINE_ROOT and parent is not None:
                findings.append(_finding("spine_root_not_a_root", "spine.turn has a parent", trace_id=trace_id))
    return _result(findings, observed=len(turns), evidence_class=CONTRACT, unit="turns")


def check_usage_join(turns: dict[str, dict[str, Any]], usage_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Every persisted usage row resolves to a Turn and to a span in it."""
    findings: list[dict[str, Any]] = []
    for index, row in enumerate(usage_rows):
        trace_id = row.get("trace_id")
        if not trace_id:
            findings.append(_finding("usage_row_without_trace_id", f"row {index} carries no trace id", row=index))
            continue
        turn = turns.get(trace_id)
        # 没有 spine.turn 的追踪组不是轮次；对其解析会报告证据实际不支持的关联。
        if turn is None or turn["root_count"] == 0:
            findings.append(
                _finding("usage_row_unjoinable_trace", "trace id has no Turn", row=index, trace_id=trace_id)
            )
            continue
        span_id = row.get("turn_span_id")
        if span_id not in turn["span_ids"]:
            findings.append(
                _finding(
                    "usage_row_unjoinable_span",
                    "turn_span_id is not a span of this trace",
                    row=index,
                    trace_id=trace_id,
                )
            )
    return _result(findings, observed=len(usage_rows), evidence_class=CONTRACT, unit="usage rows")


def check_delivery_join(
    turns: dict[str, dict[str, Any]],
    spans: list[dict[str, Any]],
    notices: list[dict[str, Any]],
) -> dict[str, Any]:
    """Delivery spans resolve to a Turn, and an exhausted send has its notice."""
    findings: list[dict[str, Any]] = []
    delivery_spans = [s for s in spans if s.get("name") == DELIVERY]
    dropped_conversations = set()
    for span in delivery_spans:
        attrs = span.get("attributes") or {}
        trace_id = span.get("traceId")
        outcome = attrs.get("channel.outcome")
        if outcome not in CHANNEL_OUTCOMES:
            findings.append(
                _finding("delivery_outcome_unknown", f"unknown channel outcome {outcome!r}", trace_id=trace_id)
            )
        if outcome == "no_outlet":
            continue  # 从未入队，因此没有可作为父级的轮次跨度
        turn = turns.get(trace_id)
        if turn is None or turn["root_count"] == 0:
            findings.append(_finding("delivery_span_unjoinable_trace", "delivery trace has no Turn", trace_id=trace_id))
            continue
        if outcome == "dropped":
            dropped_conversations.add(attrs.get("channel.conversation_id"))

    notified = {n.get("conversation_id") for n in notices if n.get("kind") == "delivery_failed"}
    for conversation in sorted(dropped_conversations - notified, key=str):
        findings.append(
            _finding(
                "dropped_delivery_without_notice",
                "a dropped delivery raised no DELIVERY_FAILED notice",
                conversation_id=conversation,
            )
        )
    for conversation in sorted(notified - dropped_conversations, key=str):
        findings.append(
            _finding(
                "notice_without_dropped_delivery",
                "a DELIVERY_FAILED notice has no dropped delivery span",
                conversation_id=conversation,
            )
        )
    return _result(findings, observed=len(delivery_spans), evidence_class=CONTRACT, unit="delivery spans")


def check_terminal_states(turns: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Terminal states are known, self-consistent, covered, and distinct."""
    findings: list[dict[str, Any]] = []
    observed: dict[str, str] = {}
    if not any(t["root_count"] for t in turns.values()):
        return _result([], observed=0, evidence_class=CONTRACT, unit="terminal states")
    for turn in turns.values():
        conversation = turn["conversation_id"]
        outcome = turn["outcome"]
        if turn["root_count"] == 0:
            continue
        if not outcome:
            findings.append(
                _finding(
                    "spine_turn_without_outcome",
                    "spine.turn reached no terminal state",
                    trace_id=turn["trace_id"],
                    conversation_id=conversation,
                )
            )
            continue
        if outcome not in SPINE_OUTCOMES:
            findings.append(
                _finding("spine_outcome_unknown", f"unknown outcome {outcome!r}", trace_id=turn["trace_id"])
            )
            continue
        expected_status = "OK" if outcome in _OK_OUTCOMES else "ERROR"
        if turn["status"] != expected_status:
            findings.append(
                _finding(
                    "spine_status_contradicts_outcome",
                    f"outcome {outcome} with span status {turn['status']}",
                    trace_id=turn["trace_id"],
                    conversation_id=conversation,
                )
            )
        expected_event = "TurnEnded" if outcome in _OK_OUTCOMES else "TurnFailed"
        if turn["terminal_event"] != expected_event:
            findings.append(
                _finding(
                    "spine_terminal_event_contradicts_outcome",
                    f"outcome {outcome} with terminal event {turn['terminal_event']}",
                    trace_id=turn["trace_id"],
                    conversation_id=conversation,
                )
            )
        if conversation is not None:
            observed[conversation] = outcome

    for conversation, expected in sorted(EXPECTED_TERMINALS.items()):
        actual = observed.get(conversation)
        if actual is None:
            findings.append(
                _finding("scenario_turn_missing", "expected scenario Turn absent", conversation_id=conversation)
            )
        elif actual != expected:
            findings.append(
                _finding(
                    "scenario_terminal_mismatch",
                    f"expected {expected}, observed {actual}",
                    conversation_id=conversation,
                )
            )

    witnesses = [observed[c] for c in TERMINAL_WITNESSES if c in observed]
    if len(set(witnesses)) != len(witnesses):
        findings.append(_finding("terminal_states_not_distinct", "two witness Turns share a terminal state"))
    missing = sorted(set(SPINE_OUTCOMES) - set(witnesses))
    if missing:
        findings.append(_finding("terminal_state_not_covered", f"no scenario Turn reaches {', '.join(missing)}"))
    return _result(findings, observed=len(observed), evidence_class=CONTRACT, unit="terminal states")


def check_scenario_deliveries(turns: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The scenario's delivery outcomes are the ones the gate contracts for."""
    findings: list[dict[str, Any]] = []
    by_conversation = {t["conversation_id"]: t for t in turns.values() if t["conversation_id"]}
    if not by_conversation:
        # 完全没有轮次：契约并非被违反，而是从未被观察到。
        return _result([], observed=0, evidence_class=CONTRACT, unit="scenario deliveries")
    for conversation, expected in sorted(EXPECTED_DELIVERIES.items()):
        turn = by_conversation.get(conversation)
        if turn is None:
            findings.append(
                _finding("scenario_turn_missing", "expected scenario Turn absent", conversation_id=conversation)
            )
            continue
        if expected not in turn["delivery_outcomes"]:
            findings.append(
                _finding(
                    "scenario_delivery_mismatch",
                    f"expected a {expected} delivery, observed {turn['delivery_outcomes']}",
                    conversation_id=conversation,
                )
            )
    return _result(findings, observed=len(EXPECTED_DELIVERIES), evidence_class=CONTRACT, unit="scenario deliveries")


def _result(findings: list[dict[str, Any]], *, observed: int, evidence_class: str, unit: str) -> dict[str, Any]:
    # 发现优先于观察计数：检测器可能在单元计数器从未看到的记录上发现矛盾，例如没有投递跨度的
    # 通知；若报告为无结论，就会把已违反契约说成未执行契约。
    if findings:
        status = FAILED
    elif observed == 0:
        status = INCONCLUSIVE
    else:
        status = PASSED
    return {
        "status": status,
        "evidence_class": evidence_class,
        "observed": observed,
        "unit": unit,
        "findings": findings,
    }


# --------------------------------------------------------------------------
# 场景执行与报告
# --------------------------------------------------------------------------


def _counts(output: str) -> dict[str, int]:
    return {
        name: int(match.group(1)) if (match := pattern.search(output)) else 0
        for name, pattern in _COUNT_PATTERNS.items()
    }


def run_scenario(output_root: Path) -> tuple[dict[str, Any], Path | None]:
    """Run the scenario under its own basetemp; return the check and its root."""
    basetemp = output_root / "scenario"
    log_path = output_root / "scenario.log"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--strict-markers",
        f"--basetemp={basetemp}",
        SCENARIO_TEST,
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        log_path.write_text(output, encoding="utf-8")
        return (
            {
                "status": INFRASTRUCTURE_FAILURE,
                "evidence_class": DETERMINISTIC,
                "command": command,
                "exit_code": None,
                "counts": {},
                "log": log_path.name,
                "log_sha256": _sha256(log_path),
            },
            None,
        )

    output = completed.stdout + completed.stderr
    log_path.write_text(output, encoding="utf-8")
    counts = _counts(output)
    manifest_path = basetemp / MANIFEST_FILENAME
    if completed.returncode == 0 and counts["passed"] > 0 and not counts["failed"] and not counts["errors"]:
        status = PASSED
    elif counts["passed"] == 0 and counts["skipped"] > 0:
        status = SKIPPED
    else:
        status = FAILED
    if status == PASSED and not manifest_path.exists():
        status = INFRASTRUCTURE_FAILURE
    check = {
        "status": status,
        "evidence_class": DETERMINISTIC,
        "command": command,
        "exit_code": completed.returncode,
        "counts": counts,
        "log": log_path.name,
        "log_sha256": _sha256(log_path),
    }
    return check, (manifest_path if manifest_path.exists() else None)


def build_report(scenario: dict[str, Any], checks: dict[str, dict[str, Any]], turns: list[dict[str, Any]]) -> dict:
    all_checks = {"scenario": scenario, **checks}
    findings = [{"check": name, **finding} for name, check in checks.items() for finding in check.get("findings", [])]
    if scenario["status"] == INFRASTRUCTURE_FAILURE:
        status = INFRASTRUCTURE_FAILURE
    elif all(check["status"] == PASSED for check in all_checks.values()):
        status = PASSED
    elif any(check["status"] == FAILED for check in all_checks.values()):
        status = FAILED
    else:
        status = INCONCLUSIVE
    return {
        "checks": all_checks,
        "evidence_class": DETERMINISTIC,
        "findings": findings,
        "gate": GATE,
        "rerun": RERUN,
        "schema": SCHEMA,
        "status": status,
        "turns": turns,
    }


def evaluate(manifest: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Validate the artifacts a scenario run described; the pure gate core."""
    spans = dedupe_spans(read_jsonl(Path(manifest["spans"])))
    telemetry_dir = Path(manifest["telemetry"])
    usage_rows = load_usage_rows(telemetry_dir)
    notices = read_jsonl(Path(manifest["notices"]))
    turns = index_turns(spans)

    checks = {
        "turn_correlation": check_turn_correlation(turns),
        "usage_join": check_usage_join(turns, usage_rows),
        "call_efficiency_health": check_call_efficiency_health(telemetry_dir),
        "delivery_join": check_delivery_join(turns, spans, notices),
        "terminal_states": check_terminal_states(turns),
        "scenario_deliveries": check_scenario_deliveries(turns),
    }
    usage_by_trace: dict[str, int] = defaultdict(int)
    for row in usage_rows:
        usage_by_trace[row.get("trace_id")] += 1
    public = sorted(
        (_public_turn(turn, usage_by_trace.get(trace_id, 0)) for trace_id, turn in turns.items()),
        key=lambda t: str(t["conversation_id"]),
    )
    return checks, public


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the V-TE0 turn-evidence correlation gate.")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    scenario, manifest_path = run_scenario(output_root)

    if manifest_path is None:
        checks = {
            name: {
                "status": SKIPPED,
                "evidence_class": CONTRACT,
                "observed": 0,
                "unit": "artifacts",
                "findings": [],
                "reason": "scenario produced no manifest",
            }
            for name in (
                "turn_correlation",
                "usage_join",
                "call_efficiency_health",
                "delivery_join",
                "terminal_states",
                "scenario_deliveries",
            )
        }
        turns: list[dict[str, Any]] = []
    else:
        checks, turns = evaluate(json.loads(manifest_path.read_text(encoding="utf-8")))

    report = build_report(scenario, checks, turns)
    report_path = output_root / REPORT_FILENAME
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{GATE} {report['status']}: {report_path}")
    return 0 if report["status"] == PASSED else 1


if __name__ == "__main__":
    raise SystemExit(main())
