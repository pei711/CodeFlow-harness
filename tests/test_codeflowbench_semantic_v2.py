"""Tests for the frozen historical semantic v2 evidence schema."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

from pydantic import SecretStr

from benchmarks.codeflowbench.packs.memory_skill.semantic_fixtures import (
    semantic_fixture,
)
from benchmarks.codeflowbench.semantic_campaign import (
    DEFAULT_OUTPUT_ROOT_V2,
    DEFAULT_SUITE_PATH_V2,
    _runtime_config,
    _selection_precision_from_field,
    _skill_raw_abstention_claim_passed,
    build_semantic_preflight,
    load_semantic_suite,
)


def _provider_settings() -> SimpleNamespace:
    return SimpleNamespace(
        model="text-embedding-test",
        base_url="https://embedding.example/v1",
        api_key=SecretStr("configured"),
    )


def test_semantic_v2_is_disjoint_and_preregistered() -> None:
    v1 = semantic_fixture("formal")
    calibration = semantic_fixture("calibration", version="v2")
    formal = semantic_fixture("formal", version="v2")
    suite = load_semantic_suite(DEFAULT_SUITE_PATH_V2)
    config = _runtime_config(suite)

    assert formal.planned_cases == 260
    assert DEFAULT_OUTPUT_ROOT_V2.as_posix() == (".codeflow/evidence/codeflowbench-semantic-v2")
    assert config.effective_memory_top_k == 1
    assert config.effective_skill_top_k == 10
    assert config.user_min_score == 0.65
    assert config.local_min_score == 4.0
    assert config.skill_evidence_stage == "candidate_retrieval"
    assert suite["development_evidence"] == {
        "suite": "agent-application-ship-1-semantic",
        "experiment_id": ("1336036d67d0c4ddf6142d894ed1cbfc1019d85d4624bd878d6a8c4e3b88018a"),
        "report_digest": ("7d5bff56fca4d1cb5cc64984cb8ce0aa5f1c6f6d1b5ea4500f838d933067d0e1"),
        "disposition": "retained_negative_development_result",
    }

    for previous, held_out in (
        (v1.memory_corpus, formal.memory_corpus),
        (v1.skill_corpus, formal.skill_corpus),
        (calibration.memory_corpus, formal.memory_corpus),
        (calibration.skill_corpus, formal.skill_corpus),
    ):
        assert {item.item_id for item in previous}.isdisjoint(item.item_id for item in held_out)
        assert {item.text for item in previous}.isdisjoint(item.text for item in held_out)
    assert {item.logical_id for item in v1.skill_corpus}.isdisjoint(item.logical_id for item in formal.skill_corpus)
    assert {item.logical_id for item in calibration.skill_corpus}.isdisjoint(
        item.logical_id for item in formal.skill_corpus
    )
    for previous, held_out in (
        (v1.memory_queries, formal.memory_queries),
        (v1.skill_queries, formal.skill_queries),
        (calibration.memory_queries, formal.memory_queries),
        (calibration.skill_queries, formal.skill_queries),
    ):
        assert {query.query_id for query in previous}.isdisjoint(query.query_id for query in held_out)
        assert {str(query.payload["query_text"]) for query in previous}.isdisjoint(
            str(query.payload["query_text"]) for query in held_out
        )


def test_semantic_v2_calibration_density_matches_candidate_depth() -> None:
    fixture = semantic_fixture("calibration", version="v2")
    counts = Counter((item.workspace_id, item.source) for item in fixture.skill_corpus)
    workspaces = {str(query.payload["workspace_id"]) for query in fixture.skill_queries}

    assert fixture.planned_cases == 48
    assert all(counts[(workspace, "local")] >= 10 for workspace in workspaces)
    assert all(counts[(workspace, "everos")] >= 10 for workspace in workspaces)


def test_semantic_v2_preflight_uses_new_schema_and_denominators() -> None:
    calibration = build_semantic_preflight(
        "calibration",
        suite_path=DEFAULT_SUITE_PATH_V2,
        provider_settings=_provider_settings(),
    )

    assert calibration.planned_cases == 48
    assert calibration.provider_configured is True
    assert calibration.estimated_worst_case_cny <= 5.0


def test_semantic_v2_raw_abstention_requires_positive_candidate_recall() -> None:
    suite = load_semantic_suite(DEFAULT_SUITE_PATH_V2)
    metrics = {
        "skill.candidate.fused_recall_at_10": 0.0,
        ("skill.candidate.fused_recall_improvement_over_best_single_source"): 0.0,
        "skill.candidate.cross_workspace_leakage_count": 0,
        "skill.candidate.weighted_rrf_auditable": True,
        "skill.raw.local_hard_negative_abstention_rate": 1.0,
        "skill.raw.everos_hard_negative_abstention_rate": 1.0,
        "skill.raw.fused_hard_negative_abstention_rate": 1.0,
    }

    assert (
        _skill_raw_abstention_claim_passed(
            metrics,
            suite["claims"],
        )
        is False
    )


def test_semantic_v2_irrelevant_injection_excludes_abstentions() -> None:
    abstained = {
        "expected_item_ids": ["expected"],
        "injected_results": [],
    }
    correct = {
        "expected_item_ids": ["expected"],
        "injected_results": [{"item_id": "expected"}],
    }
    wrong = {
        "expected_item_ids": ["expected"],
        "injected_results": [{"item_id": "other"}],
    }

    assert (
        1.0
        - _selection_precision_from_field(
            [abstained, correct],
            "injected_results",
            1,
        )
        == 0.0
    )
    assert (
        1.0
        - _selection_precision_from_field(
            [abstained, wrong],
            "injected_results",
            1,
        )
        == 1.0
    )
