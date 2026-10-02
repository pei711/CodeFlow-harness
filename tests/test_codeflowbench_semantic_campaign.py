"""Tests for the historical EverOS semantic campaign boundary."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from benchmarks.codeflowbench.canonical import canonical_bytes, canonical_digest
from benchmarks.codeflowbench.packs.memory_skill.semantic_fixtures import (
    semantic_fixture,
)
from benchmarks.codeflowbench.packs.memory_skill.semantic_runtime import (
    semantic_runtime_identity,
)
from benchmarks.codeflowbench.semantic_campaign import (
    DEFAULT_SUITE_PATH_V2,
    SemanticCampaignError,
    _expected_record_keys,
    _runtime_config,
    _semantic_record_seal,
    _skill_claim_passed,
    build_semantic_preflight,
    load_semantic_suite,
    rebuild_semantic_report,
    run_semantic_track,
)


def _provider_settings() -> SimpleNamespace:
    return SimpleNamespace(
        model="text-embedding-test",
        base_url="https://embedding.example/v1",
        api_key=SecretStr("configured"),
    )


def test_semantic_preflight_is_fail_closed_and_below_five_cny() -> None:
    calibration = build_semantic_preflight(
        "calibration",
        provider_settings=_provider_settings(),
    )
    calibration_report = _calibration_report(calibration)
    formal = build_semantic_preflight(
        "formal",
        provider_settings=_provider_settings(),
        calibration_report=calibration_report,
    )

    assert calibration.provider_configured is True
    assert calibration.planned_cases == 34
    assert formal.planned_cases == 260
    assert 0 < calibration.estimated_embedding_calls
    assert calibration.estimated_worst_case_cny <= calibration.hard_cap_cny <= 5.0
    assert formal.estimated_worst_case_cny <= formal.hard_cap_cny <= 5.0
    assert formal.calibration_embedding_calls == 23
    assert formal.calibration_billable_characters == 4096
    assert formal.estimated_billable_characters == formal.track_estimated_billable_characters + 4096
    assert len(formal.approval_digest) == 64
    assert len(formal.provider_config_digest) == 64


def test_fresh_semantic_preflight_bootstraps_budget_high_water(
    tmp_path: Path,
) -> None:
    preflight = build_semantic_preflight(
        "calibration",
        provider_settings=_provider_settings(),
        budget_root=tmp_path,
    )
    ledger_path = Path(preflight.budget_ledger_path)
    high_water_path = ledger_path.with_suffix(".high-water.json")

    assert ledger_path.exists()
    assert json.loads(high_water_path.read_text(encoding="utf-8")) == {
        "schema": "codeflow.codeflowbench.provider-budget-high-water.v1",
        "ledger_path": str(ledger_path),
        "event_count": 0,
        "ledger_digest": canonical_digest([]),
        "provider_charged_cny": 0.0,
    }


def test_formal_preflight_rejects_mismatched_calibration() -> None:
    calibration = build_semantic_preflight(
        "calibration",
        provider_settings=_provider_settings(),
    )
    report = {
        **_calibration_report(calibration),
        "provider_config_digest": "f" * 64,
    }

    with pytest.raises(
        SemanticCampaignError,
        match="same commit, suite, and embedding provider",
    ):
        build_semantic_preflight(
            "formal",
            provider_settings=_provider_settings(),
            calibration_report=report,
        )


def test_semantic_suite_weights_match_executed_runtime_config() -> None:
    suite = load_semantic_suite()
    config = _runtime_config(suite)

    assert config.local_weight == 1.0
    assert config.everos_weight == 0.9
    assert config.user_min_score == 0.35
    assert config.agent_radius == 0.55


@pytest.mark.asyncio
async def test_semantic_run_requires_exact_paid_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "benchmarks.codeflowbench.semantic_campaign._load_embedding_settings",
        _provider_settings,
    )
    monkeypatch.delenv(
        "CODEFLOWBENCH_SEMANTIC_PAID_APPROVAL",
        raising=False,
    )
    monkeypatch.setattr(
        "benchmarks.codeflowbench.semantic_campaign._git_output",
        lambda _root, *args: "a" * 40 if args == ("rev-parse", "HEAD") else "",
    )

    with pytest.raises(SemanticCampaignError, match="paid approval"):
        await run_semantic_track("calibration")


@pytest.mark.asyncio
async def test_complete_existing_root_rebuilds_before_paid_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "benchmarks.codeflowbench.semantic_campaign._load_embedding_settings",
        _provider_settings,
    )
    monkeypatch.setattr(
        "benchmarks.codeflowbench.semantic_campaign._git_output",
        lambda _root, *args: "a" * 40 if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.delenv(
        "CODEFLOWBENCH_SEMANTIC_PAID_APPROVAL",
        raising=False,
    )
    preflight = build_semantic_preflight(
        "calibration",
        provider_settings=_provider_settings(),
    )
    _complete_artifact_root(
        tmp_path,
        plan_digest=preflight.plan_digest,
        root_name=preflight.experiment_id,
        codeflow_commit=preflight.codeflow_commit,
        provider_config_digest=preflight.provider_config_digest,
    )

    report = await run_semantic_track(
        "calibration",
        output_root=tmp_path,
    )

    assert report["experiment_id"] == preflight.experiment_id
    assert report["ship_complete"] is True


@pytest.mark.asyncio
async def test_partial_existing_root_fails_before_paid_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "benchmarks.codeflowbench.semantic_campaign._load_embedding_settings",
        _provider_settings,
    )
    monkeypatch.setattr(
        "benchmarks.codeflowbench.semantic_campaign._git_output",
        lambda _root, *args: "a" * 40 if args == ("rev-parse", "HEAD") else "",
    )
    preflight = build_semantic_preflight(
        "calibration",
        provider_settings=_provider_settings(),
    )
    (tmp_path / preflight.experiment_id).mkdir()

    with pytest.raises(
        SemanticCampaignError,
        match="partial or stale",
    ):
        await run_semantic_track(
            "calibration",
            output_root=tmp_path,
        )


def test_semantic_rebuild_is_stable_and_keeps_claims_separate(
    tmp_path: Path,
) -> None:
    fixture = semantic_fixture("calibration")
    records = _empty_production_records(fixture)
    suite = load_semantic_suite()
    runtime_identity = semantic_runtime_identity(
        fixture,
        _runtime_config(suite),
    )
    manifest = {
        "schema": "codeflow.codeflowbench.semantic-addendum.v1",
        "experiment_id": "p" * 64,
        "plan_digest": "p" * 64,
        "track": "calibration",
        "preflight": {
            "plan_digest": "p" * 64,
            "experiment_id": "p" * 64,
            "planned_cases": 34,
            "codeflow_commit": "a" * 40,
            "worktree_clean": True,
            "approval_digest": "d" * 64,
            "budget_request_attempt_baseline": 0,
        },
        "suite": suite,
        "suite_digest": canonical_digest(suite),
        "runtime_identity": runtime_identity,
        "record_keys_digest": canonical_digest(_expected_record_keys(fixture)),
        "calibration_report_digest": None,
        "indexed_rows": 1,
        "track_embedding_calls": 10,
        "track_embedded_characters": 100,
        "embedding_calls": 10,
        "embedded_characters": 100,
        "provider_identity": "everos/test",
        "provider_config_digest": "c" * 64,
        "real_provider": True,
        "budget_snapshot": _budget_snapshot("d" * 64),
        "estimated_actual_cny": 0.5,
    }
    runtime_digest = canonical_digest(runtime_identity)
    records = [{**record, "runtime_identity_digest": runtime_digest} for record in records]
    root = tmp_path / "experiment"
    (root / "manifest.json").parent.mkdir(parents=True)
    (root / "manifest.json").write_bytes(canonical_bytes(manifest))
    for index, record in enumerate(records):
        path = root / "records" / "suite" / f"query-{index}" / "case.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_bytes(record))
    (root / "record-seal.json").write_bytes(canonical_bytes(_semantic_record_seal(root)))

    first = rebuild_semantic_report(root)
    first_summary = (root / "summary.json").read_bytes()
    second = rebuild_semantic_report(root)

    assert first == second
    assert first_summary == (root / "summary.json").read_bytes()
    assert first["ship_complete"] is True
    assert first["production_evidence_valid"] is True
    assert first["memory_claim_eligible"] is False
    assert first["skill_claim_eligible"] is False
    cv_metrics = json.loads((root / "cv-metrics-semantic.json").read_text(encoding="utf-8"))
    assert cv_metrics["eligible_metrics"] == {}
    assert not (root / "cv-metrics.json").exists()


def test_semantic_rebuild_rejects_missing_planned_key(
    tmp_path: Path,
) -> None:
    root = _complete_artifact_root(tmp_path)
    missing = next((root / "records").glob("*/*/*.json"))
    missing.unlink()

    with pytest.raises(SemanticCampaignError, match="record key set"):
        rebuild_semantic_report(root)


def test_semantic_rebuild_rejects_tampered_record(
    tmp_path: Path,
) -> None:
    root = _complete_artifact_root(tmp_path)
    record_path = next((root / "records").glob("*/*/*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["status"] = "tampered"
    record_path.write_bytes(canonical_bytes(record))

    with pytest.raises(SemanticCampaignError, match="record seal"):
        rebuild_semantic_report(root)


def test_semantic_v2_complete_artifact_rebuild_is_stable(
    tmp_path: Path,
) -> None:
    root = _complete_artifact_root(
        tmp_path,
        root_name="semantic-v2-complete",
        version="v2",
    )

    first = rebuild_semantic_report(root)
    first_summary = (root / "summary.json").read_bytes()
    first_cv_metrics = (root / "cv-metrics-semantic.json").read_bytes()
    second = rebuild_semantic_report(root)

    assert first == second
    assert first_summary == (root / "summary.json").read_bytes()
    assert first_cv_metrics == (root / "cv-metrics-semantic.json").read_bytes()
    assert first["schema"] == "codeflow.codeflowbench.semantic-report.v2"
    assert first["planned_cases"] == first["terminal_cases"] == 48
    assert first["ship_complete"] is True
    assert first["production_evidence_valid"] is True
    assert first["memory_claim_eligible"] is False
    assert first["skill_candidate_claim_eligible"] is False
    assert first["skill_raw_abstention_claim_eligible"] is False
    assert first["skill_evidence_stage"] == "candidate_retrieval_pre_llm_gate"
    assert first["skill_final_injection_claim_eligible"] is False
    cv_metrics = json.loads(first_cv_metrics)
    assert cv_metrics["eligible_metrics"] == {}


def test_semantic_v2_formal_positive_export_is_stage_scoped(
    tmp_path: Path,
) -> None:
    root = _positive_formal_v2_artifact_root(tmp_path)

    report = rebuild_semantic_report(root)
    cv_metrics = json.loads((root / "cv-metrics-semantic.json").read_text(encoding="utf-8"))
    eligible = cv_metrics["eligible_metrics"]

    assert report["ship_complete"] is True
    assert report["production_evidence_valid"] is True
    assert report["memory_claim_eligible"] is True
    assert report["skill_candidate_claim_eligible"] is True
    assert report["skill_raw_abstention_claim_eligible"] is False
    assert report["skill_final_injection_claim_eligible"] is False
    assert eligible["memory.recall_at_1"] == 1.0
    assert eligible["skill.candidate.fused_recall_at_10"] == 1.0
    assert eligible["skill.candidate.fused_recall_improvement_over_best_single_source"] == 1.0
    assert all(key.startswith(("memory.", "skill.candidate.")) for key in eligible)
    assert not any(key.startswith("skill.raw.") for key in eligible)
    assert not any("final_injection" in key for key in eligible)


def test_semantic_v2_record_tamper_breaks_seal(
    tmp_path: Path,
) -> None:
    root = _complete_artifact_root(
        tmp_path,
        root_name="semantic-v2-tamper",
        version="v2",
    )
    record_path = next((root / "records").glob("*/*/*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["status"] = "tampered"
    record_path.write_bytes(canonical_bytes(record))

    with pytest.raises(SemanticCampaignError, match="record seal"):
        rebuild_semantic_report(root)


def test_skill_claim_requires_fusion_gain() -> None:
    suite = load_semantic_suite()
    metrics = {
        "skill.local_recall_at_5": 0.85,
        "skill.everos_recall_at_5": 0.80,
        "skill.fused_recall_at_5": 0.88,
        "skill.fused_recall_improvement_over_best_single_source": 0.03,
        "skill.hard_negative_injection_rate": 0.0,
        "skill.cross_workspace_leakage_count": 0,
        "skill.weighted_rrf_auditable": True,
    }

    assert _skill_claim_passed(metrics, suite["claims"]) is False

    metrics["skill.fused_recall_improvement_over_best_single_source"] = 0.05
    assert _skill_claim_passed(metrics, suite["claims"]) is True


def _empty_production_records(
    fixture,
    *,
    version: str = "v1",
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for query in fixture.memory_queries:
        records.append(
            _empty_record(
                fixture.memory_suite_id,
                query,
                "user_memory_on",
                version=version,
            )
        )
    for query in fixture.skill_queries:
        for configuration in ("local_only", "everos_only", "fused"):
            records.append(
                _empty_record(
                    fixture.skill_suite_id,
                    query,
                    configuration,
                    version=version,
                )
            )
    return records


def _empty_record(
    suite_id: str,
    query,
    configuration: str,
    *,
    version: str = "v1",
) -> dict[str, object]:
    local_only = configuration == "local_only"
    memory = configuration == "user_memory_on"
    record: dict[str, object] = {
        "schema": f"codeflow.codeflowbench.semantic-retrieval-record.{version}",
        "plan_digest": "p" * 64,
        "key": {
            "retrieval_suite_id": suite_id,
            "query_id": query.query_id,
            "configuration_id": configuration,
        },
        "label": query.label,
        "expected_item_ids": list(query.expected_item_ids),
        "status": "measurable",
        "ranked_results": [],
        "usage": {
            "embedding_calls": 0 if local_only else 1,
            "provider_calls": 0 if local_only else 1,
            "retrieval_evidence_level": ("local_bm25" if local_only else "production_everos_real_vector"),
            "retrieval_path": ("local_bm25" if local_only else "production"),
            "backend_class": ("LocalSkillSource" if local_only else "EverosBackend"),
            "backend_adapter": (None if local_only else "everos_search_manager_real_embedding"),
            "provider_identity": "everos/test",
            "provider_config_digest": "c" * 64,
            "everos_semantic_quality_claim_eligible": not local_only,
        },
    }
    if version == "v2":
        record["evidence_stage"] = "memory_context_injection" if memory else "candidate_retrieval"
    record["injected_results" if memory or version == "v1" else "candidate_results"] = []
    return record


def _calibration_report(preflight) -> dict[str, object]:
    return {
        "track": "calibration",
        "ship_complete": True,
        "production_evidence_valid": True,
        "worktree_clean": True,
        "codeflow_commit": preflight.codeflow_commit,
        "suite_digest": canonical_digest(load_semantic_suite()),
        "provider_config_digest": preflight.provider_config_digest,
        "report_digest": "r" * 64,
        "embedding_calls": 23,
        "embedded_characters": 4096,
    }


def _complete_artifact_root(
    tmp_path: Path,
    *,
    plan_digest: str = "p" * 64,
    root_name: str = "complete",
    codeflow_commit: str = "a" * 40,
    provider_config_digest: str = "c" * 64,
    version: str = "v1",
    track: str = "calibration",
) -> Path:
    fixture = semantic_fixture(track, version=version)
    suite = load_semantic_suite(DEFAULT_SUITE_PATH_V2) if version == "v2" else load_semantic_suite()
    runtime_identity = semantic_runtime_identity(
        fixture,
        _runtime_config(suite),
    )
    runtime_digest = canonical_digest(runtime_identity)
    records = [
        {**record, "runtime_identity_digest": runtime_digest}
        for record in _empty_production_records(
            fixture,
            version=version,
        )
    ]
    manifest = {
        "schema": f"codeflow.codeflowbench.semantic-addendum.{version}",
        "experiment_id": plan_digest,
        "plan_digest": plan_digest,
        "track": track,
        "preflight": {
            "plan_digest": plan_digest,
            "experiment_id": plan_digest,
            "planned_cases": fixture.planned_cases,
            "codeflow_commit": codeflow_commit,
            "worktree_clean": True,
            "approval_digest": "d" * 64,
            "budget_request_attempt_baseline": 0,
        },
        "suite": suite,
        "suite_digest": canonical_digest(suite),
        "runtime_identity": runtime_identity,
        "record_keys_digest": canonical_digest(_expected_record_keys(fixture)),
        "calibration_report_digest": None,
        "indexed_rows": 1,
        "track_embedding_calls": 10,
        "track_embedded_characters": 100,
        "embedding_calls": 10,
        "embedded_characters": 100,
        "provider_identity": "everos/test",
        "provider_config_digest": provider_config_digest,
        "real_provider": True,
        "budget_snapshot": _budget_snapshot("d" * 64),
        "estimated_actual_cny": 0.5,
    }
    records = [
        {
            **record,
            "plan_digest": plan_digest,
            "usage": {
                **record["usage"],
                "provider_config_digest": provider_config_digest,
            },
        }
        for record in records
    ]
    root = tmp_path / root_name
    (root / "manifest.json").parent.mkdir(parents=True)
    (root / "manifest.json").write_bytes(canonical_bytes(manifest))
    for index, record in enumerate(records):
        path = root / "records" / "suite" / f"query-{index}" / "case.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_bytes(record))
    (root / "record-seal.json").write_bytes(
        canonical_bytes(
            _semantic_record_seal(
                root,
                semantic_schema=str(suite["schema"]),
            )
        )
    )
    return root


def _positive_formal_v2_artifact_root(tmp_path: Path) -> Path:
    root = _complete_artifact_root(
        tmp_path,
        root_name="semantic-v2-formal-positive",
        version="v2",
        track="formal",
    )
    fixture = semantic_fixture("formal", version="v2")
    queries = {query.query_id: query for query in (*fixture.memory_queries, *fixture.skill_queries)}

    for path in (root / "records").glob("*/*/*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        query = queries[record["key"]["query_id"]]
        workspace_id = str(query.payload["workspace_id"])
        configuration = record["key"]["configuration_id"]
        selected: list[dict[str, object]] = []
        if record["label"] == "positive" and configuration == "user_memory_on":
            selected = [
                _synthetic_result(
                    record["expected_item_ids"][0],
                    workspace_id,
                )
            ]
            record["injected_results"] = selected
        elif record["label"] == "positive" and configuration == "fused":
            selected = [
                _synthetic_result(
                    record["expected_item_ids"][0],
                    workspace_id,
                    sources=("local", "everos"),
                    rrf_score=0.5,
                )
            ]
            record["candidate_results"] = selected
        elif record["label"] == "hard_negative" and configuration == "fused":
            selected = [
                _synthetic_result(
                    f"negative-{query.query_id}",
                    workspace_id,
                    sources=("local", "everos"),
                    rrf_score=0.1,
                )
            ]
            record["candidate_results"] = selected
        record["ranked_results"] = selected
        path.write_bytes(canonical_bytes(record))

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    (root / "record-seal.json").write_bytes(
        canonical_bytes(
            _semantic_record_seal(
                root,
                semantic_schema=str(manifest["schema"]),
            )
        )
    )
    return root


def _synthetic_result(
    item_id: str,
    workspace_id: str,
    *,
    sources: tuple[str, ...] = (),
    rrf_score: float | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "item_id": item_id,
        "query_workspace_id": workspace_id,
        "selected_workspace_id": workspace_id,
    }
    if sources:
        result["contributing_sources"] = list(sources)
        result["rrf_score"] = rrf_score
    return result


def _budget_snapshot(approval_digest: str) -> dict[str, object]:
    return {
        "accounting_complete": True,
        "open_reservations": 0,
        "total_committed_cny": 0.5,
        "hard_cap_cny": 5.0,
        "approval_digest": approval_digest,
        "request_attempts": 10,
        "request_attempt_lifetime_ceiling": 100,
        "request_attempt_baseline": 0,
        "ledger_digest": "l" * 64,
        "high_water_digest": "h" * 64,
    }
