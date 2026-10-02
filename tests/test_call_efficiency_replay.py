"""Offline equivalence checks for historical DeepSeek TokenWise trials."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.codeflowbench.packs.tokenwise_cost.live import (
    CampaignConfig,
    LiveTrialResult,
    build_campaign_report,
    load_task_corpus,
)
from benchmarks.codeflowbench.packs.tokenwise_cost.replay import main, verify_legacy_report
from codeflow.call_efficiency.pricing import estimate_cost_from_rates

CORPUS = Path(__file__).resolve().parents[1] / "benchmarks" / "codeflowbench" / "tasks" / "tokenwise_cost" / "formal.json"


def _historical_report(tmp_path: Path) -> dict:
    corpus = load_task_corpus(CORPUS)
    config = CampaignConfig(output_root=tmp_path)
    price = {
        "input_usd_per_token": 0.14e-6,
        "output_usd_per_token": 0.28e-6,
        "cache_read_usd_per_token": 0.0028e-6,
    }
    trials: list[LiveTrialResult] = []
    for task in corpus.tasks:
        for repetition in range(3):
            for policy in ("prefix_disrupted", "prefix_stable"):
                fresh = 1000 if policy == "prefix_disrupted" else 200
                cache_read = 0 if policy == "prefix_disrupted" else 800
                cost = estimate_cost_from_rates(
                    input_tokens=fresh,
                    output_tokens=20,
                    cache_read_tokens=cache_read,
                    cache_write_tokens=0,
                    **price,
                )
                trials.append(
                    LiveTrialResult(
                        task_id=task.task_id,
                        workload_class=task.workload_class,
                        repetition=repetition,
                        cache_policy=policy,
                        task_passed=True,
                        usage_complete=True,
                        cost_complete=True,
                        requested_model=config.model,
                        actual_model=config.model,
                        fallback_used=False,
                        fresh_input_tokens=fresh,
                        cache_write_tokens=0,
                        cache_read_tokens=cache_read,
                        output_tokens=20,
                        cost_usd=cost,
                        provider_calls=1,
                        latency_ms=1,
                        findings=(),
                    )
                )
    return build_campaign_report(config=config, corpus=corpus, trials=tuple(trials))


def test_replay_reprices_trials_and_rebuilds_the_historical_claim(tmp_path: Path) -> None:
    source = _historical_report(tmp_path)

    replay = verify_legacy_report(source, expected_source_digest=source["report_digest"])

    assert replay["schema"] == "codeflow.codeflowbench.call-efficiency-replay.report.v1"
    assert replay["source_schema"] == "codeflow.codeflowbench.tokenwise-cost.report.v1"
    assert replay["source_report_digest"] == source["report_digest"]
    assert replay["source_report_digest_valid"] is True
    assert replay["expected_source_digest_matches"] is True
    assert replay["trial_count"] == 72
    assert replay["cost_equivalent"] is True
    assert replay["claim_equivalent"] is True
    assert replay["equivalent"] is True
    assert replay["findings"] == []
    assert len(replay["replay_digest"]) == 64


def test_replay_fails_closed_when_historical_cost_is_tampered(tmp_path: Path) -> None:
    source = deepcopy(_historical_report(tmp_path))
    source["trials"][0]["cost_usd"] += 1.0

    replay = verify_legacy_report(source, expected_source_digest=source["report_digest"])

    assert replay["equivalent"] is False
    assert replay["source_report_digest_valid"] is False
    assert replay["cost_equivalent"] is False
    assert "source_report_digest_mismatch" in replay["findings"]
    assert any(finding.startswith("trial_cost_mismatch:") for finding in replay["findings"])


def test_replay_can_bind_lineage_to_an_expected_source_digest(tmp_path: Path) -> None:
    source = _historical_report(tmp_path)

    replay = verify_legacy_report(source, expected_source_digest="0" * 64)

    assert replay["equivalent"] is False
    assert replay["expected_source_digest_matches"] is False
    assert "source_report_digest_not_expected" in replay["findings"]


def test_replay_requires_external_source_digest_binding(tmp_path: Path) -> None:
    source = _historical_report(tmp_path)

    replay = verify_legacy_report(source)

    assert replay["equivalent"] is False
    assert "expected_source_digest_required" in replay["findings"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda source: source.update(schema="unsupported"), "unsupported historical TokenWise report schema"),
        (lambda source: source.update(campaign=None), "missing campaign or trials"),
        (lambda source: source.update(trials=None), "missing campaign or trials"),
        (lambda source: source["campaign"].pop("price_snapshot"), "no frozen price snapshot"),
        (lambda source: source["trials"].__setitem__(0, None), "historical trial 0 is not an object"),
    ],
)
def test_replay_rejects_malformed_artifacts(tmp_path: Path, mutation, message: str) -> None:
    source = _historical_report(tmp_path)
    mutation(source)

    with pytest.raises(ValueError, match=message):
        verify_legacy_report(source, expected_source_digest=source["report_digest"])


def test_replay_cli_writes_bound_report_and_returns_equivalence_status(tmp_path: Path) -> None:
    source = _historical_report(tmp_path)
    source_path = tmp_path / "source.json"
    output_path = tmp_path / "replay.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")

    result = main(
        [
            "--source-report",
            str(source_path),
            "--expected-source-digest",
            source["report_digest"],
            "--output",
            str(output_path),
        ]
    )

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert result == 0
    assert written["schema"] == "codeflow.codeflowbench.call-efficiency-replay.report.v1"
    assert written["source_report_digest"] == source["report_digest"]
    assert written["equivalent"] is True


def test_replay_cli_refuses_to_overwrite_the_source_report(tmp_path: Path) -> None:
    source = _historical_report(tmp_path)
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(SystemExit):
        main(
            [
                "--source-report",
                str(source_path),
                "--expected-source-digest",
                source["report_digest"],
                "--output",
                str(source_path),
            ]
        )

    assert json.loads(source_path.read_text(encoding="utf-8"))["report_digest"] == source["report_digest"]
