"""Unit tests for the V-TE0 detectors.

Each detector is exercised against a healthy fixture and against a corrupted
one, because a gate that only ever sees a healthy run proves nothing about what
it would catch.
"""

from __future__ import annotations

import copy
import json

import pytest

from scripts.verify_turn_evidence import (
    CHANNEL_OUTCOMES,
    SPINE_OUTCOMES,
    build_report,
    check_call_efficiency_health,
    check_delivery_join,
    check_scenario_deliveries,
    check_terminal_states,
    check_turn_correlation,
    check_usage_join,
    dedupe_spans,
    evaluate,
    index_turns,
    load_usage_rows,
    read_jsonl,
)

_TRACE = "trace-1"


def _span(name, span_id, parent, *, trace_id=_TRACE, status="OK", attributes=None):
    return {
        "traceId": trace_id,
        "spanId": span_id,
        "parentSpanId": parent,
        "name": name,
        "status": {"code": status, "message": ""},
        "attributes": attributes or {},
    }


def _turn_spans(
    conversation,
    trace_id,
    *,
    outcome="completed",
    status="OK",
    terminal_event="TurnEnded",
    delivery=None,
):
    root, session = f"{trace_id}-root", f"{trace_id}-session"
    spans = [
        _span(
            "spine.turn",
            root,
            None,
            trace_id=trace_id,
            status=status,
            attributes={
                "spine.conversation_id": conversation,
                "spine.outcome": outcome,
                "spine.terminal_event": terminal_event,
            },
        ),
        _span("session.turn", session, root, trace_id=trace_id),
        _span("llm.call", f"{trace_id}-llm", session, trace_id=trace_id),
        _span("tool.call", f"{trace_id}-tool", session, trace_id=trace_id),
    ]
    if delivery is not None:
        spans.append(
            _span(
                "channel.deliver",
                f"{trace_id}-deliver",
                root,
                trace_id=trace_id,
                status="OK" if delivery == "delivered" else "ERROR",
                attributes={"channel.outcome": delivery, "channel.conversation_id": conversation},
            )
        )
    return spans


_SCENARIO = [
    *_turn_spans("scenario:completed", "t-ok", delivery="delivered"),
    *_turn_spans("scenario:tool_failure", "t-tool", outcome="completed_with_tool_failure", delivery="delivered"),
    *_turn_spans(
        "scenario:provider_failure", "t-prov", outcome="provider_failed", status="ERROR", terminal_event="TurnFailed"
    ),
    *_turn_spans("scenario:runner_error", "t-err", outcome="error", status="ERROR", terminal_event="TurnFailed"),
    *_turn_spans("scenario:cancelled", "t-can", outcome="cancelled", status="ERROR", terminal_event="TurnFailed"),
    *_turn_spans("scenario:delivery_exhausted", "t-drop", delivery="dropped"),
]

_USAGE = [
    {"trace_id": "t-ok", "turn_span_id": "t-ok-session"},
    {"trace_id": "t-tool", "turn_span_id": "t-tool-session"},
]

_NOTICES = [{"kind": "delivery_failed", "conversation_id": "scenario:delivery_exhausted"}]


def _detectors(spans=None, usage=None, notices=None):
    spans = _SCENARIO if spans is None else spans
    turns = index_turns(spans)
    return {
        "turn_correlation": check_turn_correlation(turns),
        "usage_join": check_usage_join(turns, _USAGE if usage is None else usage),
        "delivery_join": check_delivery_join(turns, spans, _NOTICES if notices is None else notices),
        "terminal_states": check_terminal_states(turns),
        "scenario_deliveries": check_scenario_deliveries(turns),
    }


def _detector_names(check):
    return sorted(f["detector"] for f in check["findings"])


def test_a_healthy_scenario_passes_every_detector():
    for name, check in _detectors().items():
        assert check["status"] == "passed", (name, check["findings"])
        assert check["evidence_class"] == "contract"


def test_an_empty_artifact_set_is_inconclusive_not_passed():
    checks = _detectors(spans=[], usage=[], notices=[])
    assert checks["turn_correlation"]["status"] == "inconclusive"
    assert checks["usage_join"]["status"] == "inconclusive"
    assert checks["delivery_join"]["status"] == "inconclusive"

    assert checks["terminal_states"]["status"] == "inconclusive"
    assert checks["scenario_deliveries"]["status"] == "inconclusive"


def test_a_contradiction_outranks_an_empty_unit_count():

    root_without_outcome = [
        _span("spine.turn", "t-x-root", None, trace_id="t-x", attributes={"spine.conversation_id": "scenario:x"})
    ]
    terminal = check_terminal_states(index_turns(root_without_outcome))
    assert terminal["observed"] == 0
    assert terminal["status"] == "failed"
    assert "spine_turn_without_outcome" in _detector_names(terminal)

    delivery = check_delivery_join({}, [], [{"kind": "delivery_failed", "conversation_id": "scenario:x"}])
    assert delivery["observed"] == 0
    assert delivery["status"] == "failed"
    assert _detector_names(delivery) == ["notice_without_dropped_delivery"]


def test_a_trace_without_a_spine_root_is_detected():
    spans = [s for s in _SCENARIO if not (s["traceId"] == "t-ok" and s["name"] == "spine.turn")]
    check = _detectors(spans=spans)["turn_correlation"]
    assert check["status"] == "failed"
    assert "trace_without_spine_root" in _detector_names(check)


def test_two_spine_roots_in_one_trace_are_detected():
    spans = [*_SCENARIO, _span("spine.turn", "t-ok-root2", None, trace_id="t-ok")]
    check = _detectors(spans=spans)["turn_correlation"]
    assert "duplicate_spine_root" in _detector_names(check)


def test_a_session_turn_detached_from_its_root_is_detected():
    spans = copy.deepcopy(_SCENARIO)
    next(s for s in spans if s["spanId"] == "t-ok-session")["parentSpanId"] = "somewhere-else"
    check = _detectors(spans=spans)["turn_correlation"]
    assert check["status"] == "failed"
    assert "session_turn_orphaned" in _detector_names(check)


def test_a_leaf_span_detached_from_its_turn_is_detected():
    spans = copy.deepcopy(_SCENARIO)
    next(s for s in spans if s["spanId"] == "t-ok-llm")["parentSpanId"] = "orphan"
    check = _detectors(spans=spans)["turn_correlation"]
    assert "leaf_span_orphaned" in _detector_names(check)


def test_a_turn_missing_its_model_call_is_detected():
    spans = [s for s in _SCENARIO if s["spanId"] != "t-ok-llm"]
    check = _detectors(spans=spans)["turn_correlation"]
    assert "turn_without_model_call" in _detector_names(check)


def test_a_parented_spine_root_is_detected():
    spans = copy.deepcopy(_SCENARIO)
    next(s for s in spans if s["spanId"] == "t-ok-root")["parentSpanId"] = "not-a-root"
    check = _detectors(spans=spans)["turn_correlation"]
    assert "spine_root_not_a_root" in _detector_names(check)


def test_a_usage_row_without_a_trace_id_is_detected():
    check = _detectors(usage=[{"trace_id": None, "turn_span_id": None}])["usage_join"]
    assert check["status"] == "failed"
    assert _detector_names(check) == ["usage_row_without_trace_id"]


def test_a_usage_row_pointing_at_an_unknown_trace_is_detected():
    check = _detectors(usage=[{"trace_id": "ghost", "turn_span_id": "x"}])["usage_join"]
    assert _detector_names(check) == ["usage_row_unjoinable_trace"]


def test_a_usage_row_pointing_at_a_foreign_span_is_detected():
    check = _detectors(usage=[{"trace_id": "t-ok", "turn_span_id": "t-tool-session"}])["usage_join"]
    assert _detector_names(check) == ["usage_row_unjoinable_span"]


def test_a_delivery_span_on_an_unknown_trace_is_detected():
    spans = [
        *_SCENARIO,
        _span(
            "channel.deliver",
            "ghost-deliver",
            None,
            trace_id="ghost",
            attributes={"channel.outcome": "delivered", "channel.conversation_id": "x"},
        ),
    ]
    check = _detectors(spans=spans)["delivery_join"]
    assert "delivery_span_unjoinable_trace" in _detector_names(check)


def test_an_unknown_channel_outcome_is_detected():
    spans = copy.deepcopy(_SCENARIO)
    next(s for s in spans if s["spanId"] == "t-ok-deliver")["attributes"]["channel.outcome"] = "maybe"
    check = _detectors(spans=spans)["delivery_join"]
    assert "delivery_outcome_unknown" in _detector_names(check)


def test_a_dropped_delivery_without_its_notice_is_detected():
    check = _detectors(notices=[])["delivery_join"]
    assert check["status"] == "failed"
    assert _detector_names(check) == ["dropped_delivery_without_notice"]


def test_a_notice_with_no_dropped_delivery_is_detected():
    notices = [*_NOTICES, {"kind": "delivery_failed", "conversation_id": "scenario:completed"}]
    check = _detectors(notices=notices)["delivery_join"]
    assert _detector_names(check) == ["notice_without_dropped_delivery"]


def test_a_no_outlet_delivery_needs_no_turn():
    spans = [
        *_SCENARIO,
        _span(
            "channel.deliver",
            "orphan-deliver",
            None,
            trace_id="unrouted",
            attributes={"channel.outcome": "no_outlet", "channel.conversation_id": "ghost:c"},
        ),
    ]
    check = _detectors(spans=spans)["delivery_join"]
    assert check["status"] == "passed"


def test_a_turn_with_no_terminal_state_is_detected():
    spans = copy.deepcopy(_SCENARIO)
    del next(s for s in spans if s["spanId"] == "t-ok-root")["attributes"]["spine.outcome"]
    check = _detectors(spans=spans)["terminal_states"]
    assert "spine_turn_without_outcome" in _detector_names(check)


def test_an_unknown_terminal_state_is_detected():
    spans = copy.deepcopy(_SCENARIO)
    next(s for s in spans if s["spanId"] == "t-ok-root")["attributes"]["spine.outcome"] = "mostly_fine"
    check = _detectors(spans=spans)["terminal_states"]
    assert "spine_outcome_unknown" in _detector_names(check)


def test_a_completed_turn_on_an_error_span_is_contradictory():
    spans = copy.deepcopy(_SCENARIO)
    next(s for s in spans if s["spanId"] == "t-ok-root")["status"]["code"] = "ERROR"
    check = _detectors(spans=spans)["terminal_states"]
    assert "spine_status_contradicts_outcome" in _detector_names(check)


def test_a_failed_turn_reporting_turnended_is_contradictory():
    spans = copy.deepcopy(_SCENARIO)
    next(s for s in spans if s["spanId"] == "t-can-root")["attributes"]["spine.terminal_event"] = "TurnEnded"
    check = _detectors(spans=spans)["terminal_states"]
    assert "spine_terminal_event_contradicts_outcome" in _detector_names(check)


def test_collapsing_two_terminal_states_is_detected():
    spans = copy.deepcopy(_SCENARIO)
    root = next(s for s in spans if s["spanId"] == "t-tool-root")
    root["attributes"]["spine.outcome"] = "completed"
    check = _detectors(spans=spans)["terminal_states"]
    names = _detector_names(check)
    assert "terminal_states_not_distinct" in names
    assert "terminal_state_not_covered" in names
    assert "scenario_terminal_mismatch" in names


def test_a_missing_scenario_turn_is_detected():
    spans = [s for s in _SCENARIO if s["traceId"] != "t-can"]
    check = _detectors(spans=spans)["terminal_states"]
    assert "scenario_turn_missing" in _detector_names(check)


def test_the_witness_set_covers_the_whole_taxonomy():
    turns = index_turns(_SCENARIO)
    observed = {t["conversation_id"]: t["outcome"] for t in turns.values()}
    assert set(SPINE_OUTCOMES) <= set(observed.values())


def test_the_scenario_delivery_contract_is_enforced():
    spans = copy.deepcopy(_SCENARIO)
    next(s for s in spans if s["spanId"] == "t-drop-deliver")["attributes"]["channel.outcome"] = "delivered"
    check = _detectors(spans=spans)["scenario_deliveries"]
    assert check["status"] == "failed"
    assert "scenario_delivery_mismatch" in _detector_names(check)


def test_channel_outcome_vocabulary_matches_the_runtime():
    from codeflow.tracing import semconv

    assert set(CHANNEL_OUTCOMES) == set(semconv.CHANNEL_OUTCOMES)
    assert set(SPINE_OUTCOMES) == set(semconv.SPINE_OUTCOMES)


def test_dedupe_keeps_the_last_write_of_a_checkpointed_span():
    draft = _span("session.turn", "s1", "r1", attributes={"turn.in_progress": True})
    final = _span("session.turn", "s1", "r1", attributes={"turn.in_progress": False})
    deduped = dedupe_spans([draft, final])
    assert len(deduped) == 1
    assert deduped[0]["attributes"]["turn.in_progress"] is False


def test_read_jsonl_treats_a_missing_file_as_empty(tmp_path):
    assert read_jsonl(tmp_path / "nope.jsonl") == []


def test_read_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert read_jsonl(path) == [{"a": 1}, {"a": 2}]


def _scenario_check(status="passed"):
    return {"status": status, "evidence_class": "deterministic", "counts": {}, "exit_code": 0}


def test_the_report_passes_only_when_every_check_passes():
    report = build_report(_scenario_check(), _detectors(), [])
    assert report["status"] == "passed"
    assert report["schema"] == "codeflow.turn.evidence.v1"
    assert report["gate"] == "V-TE0"
    assert report["rerun"] == "make verify-turn-evidence"
    assert report["findings"] == []


def test_a_failing_detector_fails_the_report_and_surfaces_its_finding():
    checks = _detectors(notices=[])
    report = build_report(_scenario_check(), checks, [])
    assert report["status"] == "failed"
    assert [f["check"] for f in report["findings"]] == ["delivery_join"]
    assert report["findings"][0]["detector"] == "dropped_delivery_without_notice"


def test_an_inconclusive_check_never_passes_the_gate():
    report = build_report(_scenario_check(), _detectors(spans=[], usage=[], notices=[]), [])
    assert report["status"] == "inconclusive"


def test_a_failed_scenario_fails_the_gate():
    report = build_report(_scenario_check("failed"), _detectors(), [])
    assert report["status"] == "failed"


def test_an_unrunnable_scenario_is_infrastructure_failure_not_a_pass():
    report = build_report(_scenario_check("infrastructure_failure"), _detectors(), [])
    assert report["status"] == "infrastructure_failure"


def test_the_report_never_serializes_message_content():
    spans = copy.deepcopy(_SCENARIO)
    next(s for s in spans if s["spanId"] == "t-ok-session")["attributes"].update(
        {"turn.input_preview": "SECRET USER TEXT", "turn.output_preview": "SECRET REPLY"}
    )
    checks = _detectors(spans=spans)
    report = build_report(_scenario_check(), checks, [])
    assert "SECRET" not in json.dumps(report)


def test_usage_loader_accepts_legacy_and_call_efficiency_ledgers(tmp_path):
    telemetry = tmp_path / "telemetry"
    telemetry.mkdir()
    legacy = {"trace_id": "legacy", "turn_span_id": "turn-1"}
    current = {
        "schema": "codeflow.call-efficiency.call.v1",
        "trace_id": "current",
        "turn_span_id": "turn-2",
    }
    (telemetry / "usage-2026-01-01.jsonl").write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    (telemetry / "call-efficiency-2026-01-01.jsonl").write_text(
        json.dumps(current) + "\n",
        encoding="utf-8",
    )

    assert load_usage_rows(telemetry) == [legacy, current]


def test_call_efficiency_health_fails_when_ledger_reports_loss(tmp_path):
    (tmp_path / "call-efficiency-ledger-health.json").write_text(
        json.dumps(
            {
                "schema": "codeflow.call-efficiency.ledger-health.v1",
                "status": "degraded",
                "accepted_records": 2,
                "persisted_records": 1,
                "lost_records": 1,
            }
        ),
        encoding="utf-8",
    )

    check = check_call_efficiency_health(tmp_path)

    assert check["status"] == "failed"
    assert check["findings"][0]["detector"] == "call_efficiency_ledger_degraded"


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        json.dumps({"schema": "wrong", "status": "healthy", "lost_records": 0}),
        json.dumps({"schema": "codeflow.call-efficiency.ledger-health.v1", "status": "healthy"}),
        json.dumps(
            {
                "schema": "codeflow.call-efficiency.ledger-health.v1",
                "status": "healthy",
                "accepted_records": 2,
                "persisted_records": 1,
                "lost_records": 0,
            }
        ),
        json.dumps(
            {
                "schema": "codeflow.call-efficiency.ledger-health.v1",
                "status": "healthy",
                "accepted_records": True,
                "persisted_records": 1,
                "lost_records": 0,
            }
        ),
    ],
)
def test_call_efficiency_health_fails_closed_for_invalid_health_artifact(tmp_path, payload):
    (tmp_path / "call-efficiency-ledger-health.json").write_text(payload, encoding="utf-8")

    check = check_call_efficiency_health(tmp_path)

    assert check["status"] == "failed"
    assert check["findings"][0]["detector"] == "call_efficiency_ledger_health_invalid"


def test_call_efficiency_health_accepts_consistent_terminal_counts(tmp_path):
    (tmp_path / "call-efficiency-ledger-health.json").write_text(
        json.dumps(
            {
                "schema": "codeflow.call-efficiency.ledger-health.v1",
                "status": "healthy",
                "accepted_records": 2,
                "persisted_records": 2,
                "lost_records": 0,
            }
        ),
        encoding="utf-8",
    )

    check = check_call_efficiency_health(tmp_path)

    assert check["status"] == "passed"


@pytest.mark.parametrize("missing", ["spans", "telemetry", "notices"])
def test_evaluate_survives_a_missing_artifact(tmp_path, missing):
    manifest = {
        "spans": str(tmp_path / "spans.log"),
        "telemetry": str(tmp_path / "telemetry"),
        "notices": str(tmp_path / "notices.jsonl"),
    }
    if missing != "spans":
        (tmp_path / "spans.log").write_text("".join(json.dumps(s) + "\n" for s in _SCENARIO), encoding="utf-8")
    if missing != "telemetry":
        (tmp_path / "telemetry").mkdir()
        (tmp_path / "telemetry" / "usage-2026-01-01.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in _USAGE), encoding="utf-8"
        )
    if missing != "notices":
        (tmp_path / "notices.jsonl").write_text("".join(json.dumps(n) + "\n" for n in _NOTICES), encoding="utf-8")

    checks, turns = evaluate(manifest)
    assert build_report(_scenario_check(), checks, turns)["status"] != "passed"
