"""Historical EverOS semantic campaign retained for artifact reconstruction."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from benchmarks.codeflowbench.budget import (
    ProviderBudgetConfig,
    ProviderBudgetError,
    ProviderBudgetLedger,
)
from benchmarks.codeflowbench.canonical import (
    canonical_bytes,
    canonical_digest,
)
from benchmarks.codeflowbench.packs.memory_skill.fixtures import (
    anonymous_item_id,
)
from benchmarks.codeflowbench.packs.memory_skill.semantic_fixtures import (
    SemanticFixture,
    semantic_fixture,
)
from benchmarks.codeflowbench.packs.memory_skill.semantic_runtime import (
    CountingEmbeddingProvider,
    SemanticRuntimeConfig,
    run_semantic_runtime,
    semantic_runtime_identity,
)
from codeflow.product import get_product_home

SEMANTIC_SCHEMA = "codeflow.codeflowbench.semantic-addendum.v1"
SEMANTIC_SCHEMA_V2 = "codeflow.codeflowbench.semantic-addendum.v2"
DEFAULT_SUITE_PATH = Path(__file__).resolve().parent / "suites" / "agent_application_ship_1_semantic.yaml"
DEFAULT_SUITE_PATH_V2 = Path(__file__).resolve().parent / "suites" / "agent_application_ship_1_semantic_v2.yaml"
DEFAULT_OUTPUT_ROOT = Path(".codeflow/evidence/codeflowbench-semantic")
DEFAULT_OUTPUT_ROOT_V2 = Path(".codeflow/evidence/codeflowbench-semantic-v2")
APPROVAL_ENV = "CODEFLOWBENCH_SEMANTIC_PAID_APPROVAL"
APPROVAL_ENV_V2 = "CODEFLOWBENCH_SEMANTIC_V2_PAID_APPROVAL"

_SUITE_PATHS = {
    "v1": DEFAULT_SUITE_PATH,
    "v2": DEFAULT_SUITE_PATH_V2,
}
_OUTPUT_ROOTS = {
    "v1": DEFAULT_OUTPUT_ROOT,
    "v2": DEFAULT_OUTPUT_ROOT_V2,
}
_SCHEMAS = {
    "v1": SEMANTIC_SCHEMA,
    "v2": SEMANTIC_SCHEMA_V2,
}


class SemanticCampaignError(RuntimeError):
    pass


@dataclass(frozen=True)
class SemanticPreflight:
    track: str
    experiment_id: str
    plan_digest: str
    approval_digest: str
    provider_identity: str
    provider_config_digest: str
    provider_configured: bool
    planned_cases: int
    track_estimated_embedding_calls: int
    track_estimated_billable_characters: int
    calibration_embedding_calls: int
    calibration_billable_characters: int
    estimated_embedding_calls: int
    estimated_billable_characters: int
    estimated_input_tokens: int
    estimated_worst_case_cny: float
    hard_cap_cny: float
    budget_ledger_path: str
    budget_request_attempt_baseline: int
    budget_request_attempt_lifetime_ceiling: int
    budget_ledger_prefix_event_count: int
    budget_ledger_prefix_digest: str
    budget_ledger_prefix_charged_cny: float
    worktree_clean: bool
    codeflow_commit: str


def _semantic_version(schema: str) -> str:
    for version, expected in _SCHEMAS.items():
        if schema == expected:
            return version
    raise SemanticCampaignError(f"unsupported semantic schema: {schema}")


def load_semantic_suite(path: Path = DEFAULT_SUITE_PATH) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") not in _SCHEMAS.values():
        raise SemanticCampaignError(f"invalid semantic suite: {path}")
    version = _semantic_version(str(payload["schema"]))
    hard_cap = float(payload["budget"]["hard_cap_cny"])
    if hard_cap <= 0 or hard_cap > 5.0:
        raise SemanticCampaignError("semantic addendum hard cap must be in (0, 5] CNY")
    for track in ("calibration", "formal"):
        fixture = semantic_fixture(track, version=version)
        expected = int(payload["tracks"][track]["expected_total_cases"])
        if fixture.planned_cases != expected:
            raise SemanticCampaignError(f"{track} case count drift: {fixture.planned_cases} != {expected}")
    return payload


def build_semantic_preflight(
    track: str,
    *,
    suite_path: Path = DEFAULT_SUITE_PATH,
    provider_settings: Any | None = None,
    repository_root: Path | None = None,
    calibration_report: dict[str, Any] | None = None,
    budget_root: Path | None = None,
) -> SemanticPreflight:
    if track not in {"calibration", "formal"}:
        raise SemanticCampaignError(f"unsupported semantic track: {track}")
    suite = load_semantic_suite(suite_path)
    version = _semantic_version(str(suite["schema"]))
    fixture = semantic_fixture(track, version=version)
    settings = provider_settings if provider_settings is not None else _load_embedding_settings()
    provider_identity = f"everos/{settings.model}" if settings.model else "everos/unconfigured"
    has_key = bool(settings.api_key and settings.api_key.get_secret_value())
    configured = bool(settings.model and settings.base_url and has_key)
    transport = _embedding_transport_config(suite, settings)
    provider_config_digest = canonical_digest(transport)
    suite_digest = canonical_digest(suite)
    budget = suite["budget"]
    ledger_path = _semantic_budget_ledger_path(
        suite,
        provider_config_digest,
        budget_root=budget_root,
    )
    budget_prefix = _inspect_semantic_budget_ledger(
        ledger_path,
        budget=budget,
    )
    root = repository_root or Path(__file__).resolve().parents[2]
    revision = _git_output(root, "rev-parse", "HEAD")
    clean = not bool(_git_output(root, "status", "--porcelain"))
    calibration_calls = 0
    calibration_characters = 0
    calibration_report_digest = None
    if track == "formal":
        if calibration_report is None:
            raise SemanticCampaignError("formal semantic preflight requires a calibration report")
        _validate_calibration_report(
            calibration_report,
            codeflow_commit=revision,
            suite_digest=suite_digest,
            provider_config_digest=provider_config_digest,
        )
        calibration_calls = int(calibration_report["embedding_calls"])
        calibration_characters = int(calibration_report["embedded_characters"])
        calibration_report_digest = str(calibration_report["report_digest"])
    track_characters, track_calls = _estimate_workload(
        fixture,
        suite=suite,
    )
    billable_characters = calibration_characters + track_characters
    embedding_calls = calibration_calls + track_calls
    estimated_tokens = math.ceil(billable_characters / float(budget["characters_per_token"]))
    estimated_cny = _estimate_cost_cny(
        estimated_tokens,
        budget,
    )
    hard_cap = float(budget["hard_cap_cny"])
    if estimated_cny > hard_cap:
        raise SemanticCampaignError(f"semantic estimate {estimated_cny:.4f} CNY exceeds {hard_cap:.2f} CNY hard cap")
    runtime_config = _runtime_config(suite)
    plan = {
        "schema": suite["schema"],
        "suite": suite["suite"],
        "track": track,
        "codeflow_commit": revision,
        "worktree_clean": clean,
        "suite_digest": suite_digest,
        "provider_identity": provider_identity,
        "provider_config_digest": provider_config_digest,
        "fixture": semantic_runtime_identity(fixture, runtime_config),
        "planned_cases": fixture.planned_cases,
        "track_estimated_embedding_calls": track_calls,
        "track_estimated_billable_characters": track_characters,
        "calibration_embedding_calls": calibration_calls,
        "calibration_billable_characters": calibration_characters,
        "estimated_embedding_calls": embedding_calls,
        "estimated_billable_characters": billable_characters,
        "estimated_input_tokens": estimated_tokens,
        "estimated_worst_case_cny": round(estimated_cny, 8),
        "hard_cap_cny": hard_cap,
        "budget_request_attempt_baseline": (budget_prefix["request_attempts"]),
        "budget_request_attempt_lifetime_ceiling": (budget_prefix["request_attempts"] + track_calls),
        "budget_ledger_prefix_event_count": (budget_prefix["event_count"]),
        "budget_ledger_prefix_digest": budget_prefix["ledger_digest"],
        "budget_ledger_prefix_charged_cny": (budget_prefix["provider_charged_cny"]),
        "calibration_report_digest": calibration_report_digest,
    }
    plan_digest = canonical_digest(plan)
    approval_digest = canonical_digest(
        {
            "plan_digest": plan_digest,
            "hard_cap_cny": hard_cap,
            "provider_config_digest": provider_config_digest,
            "budget_ledger_prefix_event_count": (budget_prefix["event_count"]),
            "budget_ledger_prefix_digest": (budget_prefix["ledger_digest"]),
            "budget_ledger_prefix_charged_cny": (budget_prefix["provider_charged_cny"]),
        }
    )
    return SemanticPreflight(
        track=track,
        experiment_id=plan_digest,
        plan_digest=plan_digest,
        approval_digest=approval_digest,
        provider_identity=provider_identity,
        provider_config_digest=provider_config_digest,
        provider_configured=configured,
        planned_cases=fixture.planned_cases,
        track_estimated_embedding_calls=track_calls,
        track_estimated_billable_characters=track_characters,
        calibration_embedding_calls=calibration_calls,
        calibration_billable_characters=calibration_characters,
        estimated_embedding_calls=embedding_calls,
        estimated_billable_characters=billable_characters,
        estimated_input_tokens=estimated_tokens,
        estimated_worst_case_cny=round(estimated_cny, 8),
        hard_cap_cny=hard_cap,
        budget_ledger_path=str(ledger_path),
        budget_request_attempt_baseline=int(budget_prefix["request_attempts"]),
        budget_request_attempt_lifetime_ceiling=int(budget_prefix["request_attempts"] + track_calls),
        budget_ledger_prefix_event_count=int(budget_prefix["event_count"]),
        budget_ledger_prefix_digest=str(budget_prefix["ledger_digest"]),
        budget_ledger_prefix_charged_cny=float(budget_prefix["provider_charged_cny"]),
        worktree_clean=clean,
        codeflow_commit=revision,
    )


async def run_semantic_track(
    track: str,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    suite_path: Path = DEFAULT_SUITE_PATH,
    calibration_root: Path | None = None,
    budget_root: Path | None = None,
) -> dict[str, Any]:
    if track not in {"calibration", "formal"}:
        raise SemanticCampaignError(f"unsupported semantic track: {track}")
    calibration: dict[str, Any] | None = None
    if track == "formal":
        if calibration_root is None:
            raise SemanticCampaignError("formal semantic run requires --calibration-root")
        calibration = rebuild_semantic_report(calibration_root)
    suite = load_semantic_suite(suite_path)
    settings = _load_embedding_settings()
    preflight = build_semantic_preflight(
        track,
        suite_path=suite_path,
        provider_settings=settings,
        calibration_report=calibration,
        budget_root=budget_root,
    )
    if not preflight.provider_configured:
        raise SemanticCampaignError("EverOS embedding model, base URL, and API key are required")
    if not preflight.worktree_clean:
        raise SemanticCampaignError(f"{track} semantic campaign requires a clean worktree")
    with _exclusive_semantic_lock(output_root, preflight.experiment_id):
        return await _run_semantic_track_locked(
            track=track,
            output_root=output_root,
            suite=suite,
            settings=settings,
            preflight=preflight,
            calibration=calibration,
        )


async def _run_semantic_track_locked(
    *,
    track: str,
    output_root: Path,
    suite: dict[str, Any],
    settings: Any,
    preflight: SemanticPreflight,
    calibration: dict[str, Any] | None,
) -> dict[str, Any]:
    experiment_root = output_root / preflight.experiment_id
    if experiment_root.exists():
        try:
            existing = rebuild_semantic_report(experiment_root)
        except SemanticCampaignError as exc:
            raise SemanticCampaignError("existing semantic experiment root is partial or stale") from exc
        if (
            existing.get("experiment_id") != preflight.experiment_id
            or existing.get("codeflow_commit") != preflight.codeflow_commit
            or existing.get("provider_config_digest") != preflight.provider_config_digest
            or existing.get("ship_complete") is not True
            or existing.get("production_evidence_valid") is not True
        ):
            raise SemanticCampaignError("existing semantic experiment root is not reusable")
        return existing
    version = _semantic_version(str(suite["schema"]))
    approval_env = APPROVAL_ENV_V2 if version == "v2" else APPROVAL_ENV
    if os.environ.get(approval_env) != preflight.approval_digest:
        raise SemanticCampaignError(
            f"paid approval missing or stale; set {approval_env} to {preflight.approval_digest}"
        )

    from everos.component.embedding import OpenAIEmbeddingProvider

    transport = _embedding_transport_config(suite, settings)
    delegate = OpenAIEmbeddingProvider(
        model=str(settings.model),
        api_key=settings.api_key.get_secret_value(),
        base_url=str(settings.base_url),
        dim=int(transport["dimension"]),
        timeout=float(transport["timeout_seconds"]),
        max_retries=int(transport["max_retries"]),
        batch_size=int(transport["batch_size"]),
        max_concurrent=int(transport["max_concurrent"]),
    )
    budget = suite["budget"]
    maximum_input_tokens = math.ceil(
        int(budget["max_characters_per_embedding_call"]) / float(budget["characters_per_token"])
    )
    ledger = ProviderBudgetLedger(
        Path(preflight.budget_ledger_path),
        ProviderBudgetConfig(
            hard_cap_cny=preflight.hard_cap_cny,
            external_service_reserve_cny=float(budget["external_reserve_cny"]),
            max_total_request_attempts=(preflight.budget_request_attempt_lifetime_ceiling),
            max_input_tokens_per_call=maximum_input_tokens,
            max_output_tokens_per_call=1,
            input_cache_miss_usd_per_million=float(budget["input_usd_per_million_tokens"]),
            output_usd_per_million=0.0,
            conservative_usd_to_cny_multiplier=(
                float(budget["usd_to_cny_multiplier"]) * float(budget["safety_multiplier"])
            ),
            max_additional_request_attempts=0,
            request_attempt_baseline=(preflight.budget_request_attempt_baseline),
            approval_digest=preflight.approval_digest,
            ledger_prefix_event_count=(preflight.budget_ledger_prefix_event_count),
            ledger_prefix_digest=(preflight.budget_ledger_prefix_digest),
            ledger_prefix_charged_cny=(preflight.budget_ledger_prefix_charged_cny),
        ),
    )
    initial_budget = ledger.snapshot()
    if (
        not initial_budget.accounting_complete
        or initial_budget.request_attempts != preflight.budget_request_attempt_baseline
        or not math.isclose(
            initial_budget.provider_charged_cny,
            preflight.budget_ledger_prefix_charged_cny,
            rel_tol=0,
            abs_tol=1e-12,
        )
    ):
        raise SemanticCampaignError("semantic embedding budget does not match paid approval")
    maximum_characters = _maximum_billable_characters(budget) - preflight.calibration_billable_characters
    if maximum_characters <= 0:
        raise SemanticCampaignError("calibration consumed the semantic embedding character budget")
    embedder = CountingEmbeddingProvider(
        delegate,
        provider_identity=preflight.provider_identity,
        provider_config_digest=preflight.provider_config_digest,
        real_provider=True,
        maximum_characters=maximum_characters,
        maximum_calls=preflight.track_estimated_embedding_calls,
        characters_per_token=float(budget["characters_per_token"]),
        budget_ledger=ledger,
        budget_trial_id=f"{preflight.experiment_id}:{track}",
        maximum_input_tokens_per_call=maximum_input_tokens,
    )
    fixture = semantic_fixture(track, version=version)
    with tempfile.TemporaryDirectory(prefix=f"codeflow-semantic-{track}-") as directory:
        result = await run_semantic_runtime(
            fixture=fixture,
            config=_runtime_config(suite),
            isolated_root=Path(directory) / "everos",
            embedder=embedder,
        )
    if (
        result.embedding_calls > preflight.track_estimated_embedding_calls
        or result.embedded_characters > preflight.track_estimated_billable_characters
    ):
        raise SemanticCampaignError("semantic runtime exceeded the preregistered workload ceiling")
    final_budget = ledger.snapshot()
    if (
        not final_budget.accounting_complete
        or final_budget.request_attempts > final_budget.request_attempt_lifetime_ceiling
        or final_budget.total_committed_cny > final_budget.hard_cap_cny
        or final_budget.approval_digest != preflight.approval_digest
    ):
        raise SemanticCampaignError("semantic embedding budget accounting is incomplete")

    cumulative_characters = preflight.calibration_billable_characters + result.embedded_characters
    cumulative_calls = preflight.calibration_embedding_calls + result.embedding_calls
    manifest = {
        "schema": suite["schema"],
        "experiment_id": preflight.experiment_id,
        "plan_digest": preflight.plan_digest,
        "track": track,
        "preflight": asdict(preflight),
        "suite": suite,
        "suite_digest": canonical_digest(suite),
        "runtime_identity": semantic_runtime_identity(
            fixture,
            _runtime_config(suite),
        ),
        "record_keys_digest": canonical_digest(_expected_record_keys(fixture)),
        "calibration_report_digest": (calibration["report_digest"] if calibration is not None else None),
        "indexed_rows": result.indexed_rows,
        "track_embedding_calls": result.embedding_calls,
        "track_embedded_characters": result.embedded_characters,
        "embedding_calls": cumulative_calls,
        "embedded_characters": cumulative_characters,
        "provider_identity": result.provider_identity,
        "provider_config_digest": result.provider_config_digest,
        "real_provider": result.real_provider,
        "budget_snapshot": asdict(final_budget),
        "estimated_actual_cny": round(
            _estimate_cost_cny(
                math.ceil(cumulative_characters / float(suite["budget"]["characters_per_token"])),
                suite["budget"],
            ),
            8,
        ),
    }
    _freeze_json(experiment_root / "manifest.json", manifest)
    for record in result.records:
        key = record["key"]
        payload = {
            **record,
            "plan_digest": preflight.plan_digest,
            "runtime_identity_digest": canonical_digest(manifest["runtime_identity"]),
        }
        _freeze_json(
            experiment_root
            / "records"
            / str(key["retrieval_suite_id"])
            / str(key["query_id"])
            / f"{key['configuration_id']}.json",
            payload,
        )
    _freeze_json(
        experiment_root / "record-seal.json",
        _semantic_record_seal(
            experiment_root,
            semantic_schema=str(suite["schema"]),
        ),
    )
    return rebuild_semantic_report(experiment_root)


def rebuild_semantic_report(experiment_root: Path) -> dict[str, Any]:
    manifest = _read_object(experiment_root / "manifest.json")
    manifest_schema = str(manifest.get("schema", ""))
    if manifest_schema not in _SCHEMAS.values():
        raise SemanticCampaignError("semantic manifest schema mismatch")
    version = _semantic_version(manifest_schema)
    frozen_suite = manifest.get("suite")
    if not isinstance(frozen_suite, dict):
        raise SemanticCampaignError("semantic manifest has no frozen suite")
    frozen_suite_digest = canonical_digest(frozen_suite)
    current_suite = load_semantic_suite(_SUITE_PATHS[version])
    if manifest.get("suite_digest") != frozen_suite_digest or canonical_digest(current_suite) != frozen_suite_digest:
        raise SemanticCampaignError("semantic suite digest mismatch")
    track = str(manifest.get("track", ""))
    if track not in {"calibration", "formal"}:
        raise SemanticCampaignError("semantic manifest track mismatch")
    fixture = semantic_fixture(track, version=version)
    runtime_identity = semantic_runtime_identity(
        fixture,
        _runtime_config(frozen_suite),
    )
    if manifest.get("runtime_identity") != runtime_identity:
        raise SemanticCampaignError("semantic runtime identity mismatch")
    expected_keys = _expected_record_keys(fixture)
    if manifest.get("record_keys_digest") != canonical_digest(expected_keys):
        raise SemanticCampaignError("semantic planned key digest mismatch")
    record_paths = sorted((experiment_root / "records").glob("*/*/*.json"))
    records = [_read_object(path) for path in record_paths]
    actual_keys = [_record_key_tuple(record) for record in records]
    expected_key_tuples = {_record_key_tuple({"key": key}) for key in expected_keys}
    if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != expected_key_tuples:
        raise SemanticCampaignError("semantic record key set mismatch")
    if _read_object(experiment_root / "record-seal.json") != _semantic_record_seal(
        experiment_root,
        semantic_schema=manifest_schema,
    ):
        raise SemanticCampaignError("semantic record seal mismatch")
    plan_digest = str(manifest.get("plan_digest", ""))
    runtime_identity_digest = canonical_digest(runtime_identity)
    if not plan_digest or any(
        record.get("schema") != f"codeflow.codeflowbench.semantic-retrieval-record.{version}"
        or record.get("plan_digest") != plan_digest
        or record.get("runtime_identity_digest") != runtime_identity_digest
        for record in records
    ):
        raise SemanticCampaignError("semantic record binding mismatch")
    preflight = manifest.get("preflight")
    if (
        not isinstance(preflight, dict)
        or manifest.get("experiment_id") != plan_digest
        or preflight.get("plan_digest") != plan_digest
        or preflight.get("experiment_id") != manifest.get("experiment_id")
        or int(preflight.get("planned_cases", -1)) != len(expected_keys)
    ):
        raise SemanticCampaignError("semantic preflight binding mismatch")
    expected_cases = int(frozen_suite["tracks"][track]["expected_total_cases"])
    metrics = _reduce_semantic_records(
        fixture,
        records,
        version=version,
    )
    production_evidence = _production_evidence_valid(
        manifest,
        records,
    )
    ship_complete = len(records) == expected_cases and all(record.get("status") == "measurable" for record in records)
    memory_eligible = (
        track == "formal"
        and ship_complete
        and production_evidence
        and (
            _memory_claim_passed(metrics, frozen_suite["claims"])
            if version == "v1"
            else _memory_v2_claim_passed(
                metrics,
                frozen_suite["claims"],
            )
        )
    )
    skill_eligible = (
        track == "formal"
        and ship_complete
        and production_evidence
        and version == "v1"
        and _skill_claim_passed(metrics, frozen_suite["claims"])
    )
    skill_candidate_eligible = (
        track == "formal"
        and ship_complete
        and production_evidence
        and version == "v2"
        and _skill_candidate_claim_passed(
            metrics,
            frozen_suite["claims"],
        )
    )
    skill_raw_abstention_eligible = (
        track == "formal"
        and ship_complete
        and production_evidence
        and version == "v2"
        and _skill_raw_abstention_claim_passed(
            metrics,
            frozen_suite["claims"],
        )
    )
    payload = {
        "schema": f"codeflow.codeflowbench.semantic-report.{version}",
        "experiment_id": manifest["experiment_id"],
        "track": track,
        "ship_complete": ship_complete,
        "planned_cases": expected_cases,
        "terminal_cases": len(records),
        "production_evidence_valid": production_evidence,
        "memory_claim_eligible": memory_eligible,
        "skill_claim_eligible": skill_eligible,
        "metrics": metrics,
        "provider_identity": manifest.get("provider_identity"),
        "provider_config_digest": manifest.get("provider_config_digest"),
        "codeflow_commit": preflight.get("codeflow_commit"),
        "worktree_clean": preflight.get("worktree_clean"),
        "suite_digest": frozen_suite_digest,
        "record_keys_digest": manifest.get("record_keys_digest"),
        "calibration_report_digest": manifest.get("calibration_report_digest"),
        "embedding_calls": manifest.get("embedding_calls"),
        "embedded_characters": manifest.get("embedded_characters"),
        "track_embedding_calls": manifest.get("track_embedding_calls"),
        "track_embedded_characters": manifest.get("track_embedded_characters"),
        "estimated_actual_cny": manifest.get("estimated_actual_cny"),
        "budget_snapshot": manifest.get("budget_snapshot"),
    }
    if version == "v2":
        payload.update(
            {
                "skill_evidence_stage": ("candidate_retrieval_pre_llm_gate"),
                "skill_candidate_claim_eligible": (skill_candidate_eligible),
                "skill_raw_abstention_claim_eligible": (skill_raw_abstention_eligible),
                "skill_final_injection_claim_eligible": False,
            }
        )
    report = {
        **payload,
        "report_digest": canonical_digest(payload),
    }
    _write_json(experiment_root / "summary.json", report)
    eligible: dict[str, Any] = {}
    if memory_eligible:
        eligible.update({key: value for key, value in metrics.items() if key.startswith("memory.")})
    if skill_eligible:
        eligible.update({key: value for key, value in metrics.items() if key.startswith("skill.")})
    if skill_candidate_eligible:
        eligible.update({key: value for key, value in metrics.items() if key.startswith("skill.candidate.")})
    if skill_raw_abstention_eligible:
        eligible.update({key: value for key, value in metrics.items() if key.startswith("skill.raw.")})
    cv_payload = {
        "schema": f"codeflow.codeflowbench.semantic-cv-metrics.{version}",
        "experiment_id": manifest["experiment_id"],
        "report_digest": report["report_digest"],
        "ship_complete": ship_complete,
        "memory_claim_eligible": memory_eligible,
        "skill_claim_eligible": skill_eligible,
        "eligible_metrics": eligible,
    }
    if version == "v2":
        cv_payload.update(
            {
                "skill_evidence_stage": ("candidate_retrieval_pre_llm_gate"),
                "skill_candidate_claim_eligible": (skill_candidate_eligible),
                "skill_raw_abstention_claim_eligible": (skill_raw_abstention_eligible),
                "skill_final_injection_claim_eligible": False,
            }
        )
    _write_json(
        experiment_root / "cv-metrics-semantic.json",
        cv_payload,
    )
    return report


def _runtime_config(suite: dict[str, Any]) -> SemanticRuntimeConfig:
    retrieval = suite["retrieval"]
    version = _semantic_version(str(suite["schema"]))
    if version == "v2":
        memory_top_k = int(retrieval["memory_top_k"])
        skill_top_k = int(retrieval["skill_candidate_top_k"])
        return SemanticRuntimeConfig(
            top_k=max(memory_top_k, skill_top_k),
            user_min_score=float(retrieval["user_min_score"]),
            agent_radius=float(retrieval["agent_radius"]),
            local_weight=float(retrieval["local_weight"]),
            everos_weight=float(retrieval["everos_weight"]),
            memory_top_k=memory_top_k,
            skill_candidate_top_k=skill_top_k,
            local_min_score=float(retrieval["local_min_score"]),
            semantic_schema=str(suite["schema"]),
            skill_evidence_stage=str(suite["evidence"]["skill_stage"]),
        )
    return SemanticRuntimeConfig(
        top_k=int(retrieval["top_k"]),
        user_min_score=float(retrieval["user_min_score"]),
        agent_radius=float(retrieval["agent_radius"]),
        local_weight=float(retrieval["local_weight"]),
        everos_weight=float(retrieval["everos_weight"]),
    )


def _load_embedding_settings() -> Any:
    from everos.config import load_settings
    from codeflow.config.update_everos import configure_everos_env

    configure_everos_env()
    load_settings.cache_clear()
    return load_settings().embedding


def _embedding_transport_config(
    suite: dict[str, Any],
    settings: Any,
) -> dict[str, Any]:
    transport = suite["embedding_transport"]
    return {
        "provider_class": "everos.component.embedding.OpenAIEmbeddingProvider",
        "model": settings.model or "",
        "base_url": settings.base_url or "",
        "dimension": int(transport["dimension"]),
        "timeout_seconds": float(transport["timeout_seconds"]),
        "max_retries": int(transport["max_retries"]),
        "batch_size": int(transport["batch_size"]),
        "max_concurrent": int(transport["max_concurrent"]),
    }


def _semantic_budget_ledger_path(
    suite: dict[str, Any],
    provider_config_digest: str,
    *,
    budget_root: Path | None,
) -> Path:
    approval_id = canonical_digest(
        {
            "schema": suite["schema"],
            "suite": suite["suite"],
            "provider_config_digest": provider_config_digest,
        }
    )
    root = (
        budget_root if budget_root is not None else get_product_home() / "evidence" / "codeflowbench" / "semantic-approvals"
    )
    return root / approval_id / "provider-budget.jsonl"


def _inspect_semantic_budget_ledger(
    path: Path,
    *,
    budget: dict[str, Any],
) -> dict[str, Any]:
    if not path.exists():
        high_water_path = path.with_suffix(".high-water.json")
        if high_water_path.exists():
            raise SemanticCampaignError(
                "semantic embedding budget ledger is missing",
            )
        maximum_input_tokens = math.ceil(
            int(budget["max_characters_per_embedding_call"]) / float(budget["characters_per_token"])
        )
        inspection = ProviderBudgetLedger(
            path,
            ProviderBudgetConfig(
                hard_cap_cny=float(budget["hard_cap_cny"]),
                external_service_reserve_cny=float(
                    budget["external_reserve_cny"],
                ),
                max_total_request_attempts=2**63 - 1,
                max_input_tokens_per_call=maximum_input_tokens,
                max_output_tokens_per_call=1,
                input_cache_miss_usd_per_million=float(
                    budget["input_usd_per_million_tokens"],
                ),
                output_usd_per_million=0.0,
                conservative_usd_to_cny_multiplier=(
                    float(budget["usd_to_cny_multiplier"]) * float(budget["safety_multiplier"])
                ),
            ),
        )
        try:
            inspection.snapshot()
        except ProviderBudgetError as exc:
            raise SemanticCampaignError(
                "semantic embedding budget ledger bootstrap failed",
            ) from exc
    try:
        with path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            events = [json.loads(line) for line in handle if line.strip()]
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except (OSError, json.JSONDecodeError) as exc:
        raise SemanticCampaignError("semantic embedding budget ledger is unreadable") from exc
    reservations: dict[str, float] = {}
    terminal: dict[str, float] = {}
    for event in events:
        if not isinstance(event, dict):
            raise SemanticCampaignError("semantic embedding budget ledger is invalid")
        request_id = str(event.get("request_id", ""))
        kind = event.get("kind")
        if kind == "reserved" and request_id not in reservations:
            reservations[request_id] = float(event["reserved_cny"])
        elif kind in {"settled", "failed"} and request_id in reservations and request_id not in terminal:
            terminal[request_id] = float(event["charged_cny"])
        else:
            raise SemanticCampaignError("semantic embedding budget ledger is invalid")
    if reservations.keys() != terminal.keys():
        raise SemanticCampaignError("semantic embedding budget ledger has open reservations")
    charged = sum(terminal.values())
    digest = canonical_digest(events)
    high_water_path = path.with_suffix(".high-water.json")
    try:
        high_water = json.loads(high_water_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SemanticCampaignError("semantic embedding budget high-water record is missing") from exc
    expected_high_water = {
        "schema": "codeflow.codeflowbench.provider-budget-high-water.v1",
        "ledger_path": str(path),
        "event_count": len(events),
        "ledger_digest": digest,
        "provider_charged_cny": charged,
    }
    if high_water != expected_high_water:
        raise SemanticCampaignError("semantic embedding budget high-water record does not match")
    return {
        "event_count": len(events),
        "ledger_digest": digest,
        "request_attempts": len(reservations),
        "provider_charged_cny": charged,
    }


def _estimate_workload(
    fixture: SemanticFixture,
    *,
    suite: dict[str, Any],
) -> tuple[int, int]:
    memory_documents = sum(len(item.text) for item in fixture.memory_corpus)
    skill_documents = sum(
        len(item.logical_id) + 1 + len(item.text) for item in fixture.skill_corpus if item.source == "everos"
    )
    memory_queries = sum(len(str(query.payload["query_text"])) for query in fixture.memory_queries)
    skill_queries = 2 * sum(len(str(query.payload["query_text"])) for query in fixture.skill_queries)
    characters = memory_documents + skill_documents + memory_queries + skill_queries
    calls = (
        len(fixture.memory_corpus)
        + sum(item.source == "everos" for item in fixture.skill_corpus)
        + len(fixture.memory_queries)
        + 2 * len(fixture.skill_queries)
    )
    budget = suite["budget"]
    return (
        math.ceil(characters * float(budget["workload_character_multiplier"])),
        math.ceil(calls * float(budget["workload_call_multiplier"])),
    )


def _estimate_cost_cny(
    input_tokens: int,
    budget: dict[str, Any],
) -> float:
    variable = (
        input_tokens
        / 1_000_000
        * float(budget["input_usd_per_million_tokens"])
        * float(budget["usd_to_cny_multiplier"])
        * float(budget["safety_multiplier"])
    )
    return variable + float(budget["external_reserve_cny"])


def _maximum_billable_characters(budget: dict[str, Any]) -> int:
    available = float(budget["hard_cap_cny"]) - float(budget["external_reserve_cny"])
    per_million = (
        float(budget["input_usd_per_million_tokens"])
        * float(budget["usd_to_cny_multiplier"])
        * float(budget["safety_multiplier"])
    )
    maximum_tokens = available / per_million * 1_000_000
    return math.floor(maximum_tokens * float(budget["characters_per_token"]))


def _reduce_semantic_records(
    fixture: SemanticFixture,
    records: list[dict[str, Any]],
    *,
    version: str = "v1",
) -> dict[str, Any]:
    if version == "v2":
        return _reduce_semantic_records_v2(fixture, records)
    return _reduce_semantic_records_v1(fixture, records)


def _reduce_semantic_records_v1(
    fixture: SemanticFixture,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    memory = [record for record in records if record["key"]["retrieval_suite_id"] == fixture.memory_suite_id]
    skill = [record for record in records if record["key"]["retrieval_suite_id"] == fixture.skill_suite_id]
    memory_positive = [record for record in memory if record["label"] == "positive"]
    memory_negative = [record for record in memory if record["label"] == "hard_negative"]
    skill_positive = [record for record in skill if record["label"] == "positive"]
    skill_fused_positive = [record for record in skill_positive if record["key"]["configuration_id"] == "fused"]
    skill_fused_negative = [
        record
        for record in skill
        if record["label"] == "hard_negative" and record["key"]["configuration_id"] == "fused"
    ]
    stale_ids = {
        anonymous_item_id(fixture.memory_suite_id, item.item_id) for item in fixture.memory_corpus if item.superseded
    }
    source_contribution = {"local": 0, "everos": 0}
    fused_results = []
    for record in skill:
        if record["key"]["configuration_id"] != "fused":
            continue
        for result in record["injected_results"]:
            fused_results.append(result)
            for source in result["contributing_sources"]:
                if source in source_contribution:
                    source_contribution[source] += 1
    memory_precision = _mean_precision(memory_positive)
    local_recall = _mean_recall(
        [record for record in skill_positive if record["key"]["configuration_id"] == "local_only"],
        5,
    )
    everos_recall = _mean_recall(
        [record for record in skill_positive if record["key"]["configuration_id"] == "everos_only"],
        5,
    )
    fused_recall = _mean_recall(skill_fused_positive, 5)
    return {
        "memory.recall_at_1": _mean_recall(memory_positive, 1),
        "memory.recall_at_5": _mean_recall(memory_positive, 5),
        "memory.precision_at_5": memory_precision,
        "memory.irrelevant_injection_rate": 1.0 - memory_precision,
        "memory.hard_negative_injection_rate": _nonempty_rate(memory_negative),
        "memory.stale_injection_count": _selected_count(
            memory,
            stale_ids,
        ),
        "memory.cross_workspace_leakage_count": _selected_count(
            memory,
            selected_ids=None,
        ),
        "skill.local_recall_at_5": local_recall,
        "skill.everos_recall_at_5": everos_recall,
        "skill.fused_recall_at_5": fused_recall,
        "skill.fused_recall_improvement_over_best_single_source": round(
            fused_recall - max(local_recall, everos_recall),
            8,
        ),
        "skill.hard_negative_injection_rate": _nonempty_rate(skill_fused_negative),
        "skill.cross_workspace_leakage_count": _selected_count(
            skill,
            selected_ids=None,
        ),
        "skill.local_source_contribution": source_contribution["local"],
        "skill.everos_source_contribution": source_contribution["everos"],
        "skill.weighted_rrf_auditable": bool(fused_results)
        and all(result["rrf_score"] is not None and result["contributing_sources"] for result in fused_results),
    }


def _reduce_semantic_records_v2(
    fixture: SemanticFixture,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    memory = [record for record in records if record["key"]["retrieval_suite_id"] == fixture.memory_suite_id]
    skill = [record for record in records if record["key"]["retrieval_suite_id"] == fixture.skill_suite_id]
    memory_positive = [record for record in memory if record["label"] == "positive"]
    memory_negative = [record for record in memory if record["label"] == "hard_negative"]
    skill_positive = [record for record in skill if record["label"] == "positive"]
    skill_negative = [record for record in skill if record["label"] == "hard_negative"]
    positive_by_configuration = {
        configuration: [record for record in skill_positive if record["key"]["configuration_id"] == configuration]
        for configuration in ("local_only", "everos_only", "fused")
    }
    negative_by_configuration = {
        configuration: [record for record in skill_negative if record["key"]["configuration_id"] == configuration]
        for configuration in ("local_only", "everos_only", "fused")
    }
    stale_ids = {
        anonymous_item_id(fixture.memory_suite_id, item.item_id) for item in fixture.memory_corpus if item.superseded
    }
    source_contribution = {"local": 0, "everos": 0}
    fused_candidates = []
    for record in skill:
        if record["key"]["configuration_id"] != "fused":
            continue
        for result in record["candidate_results"]:
            fused_candidates.append(result)
            for source in result["contributing_sources"]:
                if source in source_contribution:
                    source_contribution[source] += 1
    local_recall = _mean_recall_from_field(
        positive_by_configuration["local_only"],
        "candidate_results",
        10,
    )
    everos_recall = _mean_recall_from_field(
        positive_by_configuration["everos_only"],
        "candidate_results",
        10,
    )
    fused_recall = _mean_recall_from_field(
        positive_by_configuration["fused"],
        "candidate_results",
        10,
    )
    memory_precision = _selection_precision_from_field(
        memory_positive,
        "injected_results",
        1,
    )
    return {
        "memory.recall_at_1": _mean_recall_from_field(
            memory_positive,
            "injected_results",
            1,
        ),
        "memory.precision_at_1": memory_precision,
        "memory.irrelevant_injection_rate": 1.0 - memory_precision,
        "memory.hard_negative_injection_rate": _nonempty_rate_from_field(
            memory_negative,
            "injected_results",
        ),
        "memory.stale_injection_count": _selected_count_from_field(
            memory,
            "injected_results",
            stale_ids,
        ),
        "memory.cross_workspace_leakage_count": (
            _selected_count_from_field(
                memory,
                "injected_results",
                None,
            )
        ),
        "skill.candidate.local_recall_at_10": local_recall,
        "skill.candidate.everos_recall_at_10": everos_recall,
        "skill.candidate.fused_recall_at_10": fused_recall,
        ("skill.candidate.fused_recall_improvement_over_best_single_source"): round(
            fused_recall - max(local_recall, everos_recall),
            8,
        ),
        "skill.candidate.cross_workspace_leakage_count": (
            _selected_count_from_field(
                skill,
                "candidate_results",
                None,
            )
        ),
        "skill.candidate.local_source_contribution": (source_contribution["local"]),
        "skill.candidate.everos_source_contribution": (source_contribution["everos"]),
        "skill.candidate.weighted_rrf_auditable": bool(fused_candidates)
        and all(result["rrf_score"] is not None and result["contributing_sources"] for result in fused_candidates),
        "skill.raw.local_hard_negative_abstention_rate": (
            1.0
            - _nonempty_rate_from_field(
                negative_by_configuration["local_only"],
                "candidate_results",
            )
        ),
        "skill.raw.everos_hard_negative_abstention_rate": (
            1.0
            - _nonempty_rate_from_field(
                negative_by_configuration["everos_only"],
                "candidate_results",
            )
        ),
        "skill.raw.fused_hard_negative_abstention_rate": (
            1.0
            - _nonempty_rate_from_field(
                negative_by_configuration["fused"],
                "candidate_results",
            )
        ),
    }


def _mean_recall(records: list[dict[str, Any]], limit: int) -> float:
    values = []
    for record in records:
        expected = set(record["expected_item_ids"])
        observed = {result["item_id"] for result in record["injected_results"][:limit]}
        values.append(len(expected & observed) / len(expected) if expected else 1.0)
    return sum(values) / len(values) if values else 0.0


def _mean_recall_from_field(
    records: list[dict[str, Any]],
    field: str,
    limit: int,
) -> float:
    values = []
    for record in records:
        expected = set(record["expected_item_ids"])
        observed = {result["item_id"] for result in record[field][:limit]}
        values.append(len(expected & observed) / len(expected) if expected else 1.0)
    return sum(values) / len(values) if values else 0.0


def _mean_precision(records: list[dict[str, Any]]) -> float:
    values = []
    for record in records:
        expected = set(record["expected_item_ids"])
        observed = {result["item_id"] for result in record["injected_results"][:5]}
        values.append(len(expected & observed) / len(observed) if observed else 0.0)
    return sum(values) / len(values) if values else 0.0


def _selection_precision_from_field(
    records: list[dict[str, Any]],
    field: str,
    limit: int,
) -> float:
    selected = 0
    relevant = 0
    for record in records:
        expected = set(record["expected_item_ids"])
        observed = [result["item_id"] for result in record[field][:limit]]
        selected += len(observed)
        relevant += sum(item_id in expected for item_id in observed)
    return relevant / selected if selected else 1.0


def _nonempty_rate(records: list[dict[str, Any]]) -> float:
    return sum(bool(record["injected_results"]) for record in records) / len(records) if records else 0.0


def _nonempty_rate_from_field(
    records: list[dict[str, Any]],
    field: str,
) -> float:
    return sum(bool(record[field]) for record in records) / len(records) if records else 0.0


def _selected_count(
    records: list[dict[str, Any]],
    selected_ids: set[str] | None,
) -> int:
    if selected_ids is None:
        return sum(
            not result.get("query_workspace_id")
            or not result.get("selected_workspace_id")
            or result["query_workspace_id"] != result["selected_workspace_id"]
            for record in records
            for result in record["injected_results"]
        )
    return sum(result["item_id"] in selected_ids for record in records for result in record["injected_results"])


def _selected_count_from_field(
    records: list[dict[str, Any]],
    field: str,
    selected_ids: set[str] | None,
) -> int:
    if selected_ids is None:
        return sum(
            not result.get("query_workspace_id")
            or not result.get("selected_workspace_id")
            or result["query_workspace_id"] != result["selected_workspace_id"]
            for record in records
            for result in record[field]
        )
    return sum(result["item_id"] in selected_ids for record in records for result in record[field])


def _production_evidence_valid(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
) -> bool:
    provider_digest = str(manifest.get("provider_config_digest", ""))
    local_records = [record for record in records if record["key"]["configuration_id"] == "local_only"]
    vector_records = [record for record in records if record["key"]["configuration_id"] != "local_only"]
    selections = [result for record in records for result in record["ranked_results"]]
    return (
        manifest.get("real_provider") is True
        and _semantic_budget_evidence_valid(manifest)
        and int(manifest.get("indexed_rows", 0)) > 0
        and int(manifest.get("embedding_calls", 0)) > 0
        and len(provider_digest) == 64
        and bool(vector_records)
        and all(
            isinstance(result.get("query_workspace_id"), str)
            and bool(result["query_workspace_id"])
            and isinstance(result.get("selected_workspace_id"), str)
            and bool(result["selected_workspace_id"])
            for result in selections
        )
        and all(
            record["usage"].get("retrieval_evidence_level") == "production_everos_real_vector"
            and record["usage"].get("backend_adapter") == "everos_search_manager_real_embedding"
            and record["usage"].get("provider_config_digest") == provider_digest
            and record["usage"].get("everos_semantic_quality_claim_eligible") is True
            for record in vector_records
        )
        and all(
            record["usage"].get("retrieval_evidence_level") == "local_bm25"
            and record["usage"].get("backend_class") == "LocalSkillSource"
            and record["usage"].get("backend_adapter") is None
            and int(record["usage"].get("provider_calls", -1)) == 0
            and record["usage"].get("everos_semantic_quality_claim_eligible") is False
            for record in local_records
        )
    )


def _semantic_budget_evidence_valid(manifest: dict[str, Any]) -> bool:
    snapshot = manifest.get("budget_snapshot")
    preflight = manifest.get("preflight")
    if not isinstance(snapshot, dict) or not isinstance(preflight, dict):
        return False
    try:
        return (
            snapshot.get("accounting_complete") is True
            and int(snapshot["open_reservations"]) == 0
            and float(snapshot["total_committed_cny"]) <= float(snapshot["hard_cap_cny"])
            and snapshot.get("approval_digest") == preflight.get("approval_digest")
            and int(snapshot["request_attempts"]) <= int(snapshot["request_attempt_lifetime_ceiling"])
            and int(snapshot["request_attempt_baseline"]) == int(preflight["budget_request_attempt_baseline"])
            and str(snapshot["ledger_digest"])
            and str(snapshot["high_water_digest"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def _memory_claim_passed(
    metrics: dict[str, Any],
    claims: dict[str, Any],
) -> bool:
    return (
        metrics["memory.recall_at_5"] >= float(claims["memory_recall_at_5_min"])
        and metrics["memory.irrelevant_injection_rate"] <= float(claims["memory_irrelevant_injection_rate_max"])
        and metrics["memory.hard_negative_injection_rate"] <= float(claims["memory_hard_negative_injection_rate_max"])
        and metrics["memory.stale_injection_count"] <= int(claims["stale_injection_count_max"])
        and metrics["memory.cross_workspace_leakage_count"] <= int(claims["cross_workspace_leakage_count_max"])
    )


def _memory_v2_claim_passed(
    metrics: dict[str, Any],
    claims: dict[str, Any],
) -> bool:
    return (
        metrics["memory.recall_at_1"] >= float(claims["memory_recall_at_1_min"])
        and metrics["memory.irrelevant_injection_rate"] <= float(claims["memory_irrelevant_injection_rate_max"])
        and metrics["memory.hard_negative_injection_rate"] <= float(claims["memory_hard_negative_injection_rate_max"])
        and metrics["memory.stale_injection_count"] <= int(claims["stale_injection_count_max"])
        and metrics["memory.cross_workspace_leakage_count"] <= int(claims["cross_workspace_leakage_count_max"])
    )


def _skill_claim_passed(
    metrics: dict[str, Any],
    claims: dict[str, Any],
) -> bool:
    return (
        metrics["skill.fused_recall_at_5"] >= float(claims["skill_fused_recall_at_5_min"])
        and metrics["skill.fused_recall_improvement_over_best_single_source"]
        >= float(claims["skill_fused_recall_improvement_over_best_single_source_min"])
        and metrics["skill.hard_negative_injection_rate"] <= float(claims["skill_hard_negative_injection_rate_max"])
        and metrics["skill.cross_workspace_leakage_count"] <= int(claims["cross_workspace_leakage_count_max"])
        and metrics["skill.weighted_rrf_auditable"] is True
    )


def _skill_candidate_claim_passed(
    metrics: dict[str, Any],
    claims: dict[str, Any],
) -> bool:
    return (
        metrics["skill.candidate.fused_recall_at_10"] >= float(claims["skill_candidate_fused_recall_at_10_min"])
        and metrics["skill.candidate.fused_recall_improvement_over_best_single_source"]
        >= float(claims["skill_candidate_fused_recall_improvement_over_best_single_source_min"])
        and metrics["skill.candidate.cross_workspace_leakage_count"]
        <= int(claims["skill_candidate_cross_workspace_leakage_count_max"])
        and metrics["skill.candidate.weighted_rrf_auditable"] is True
    )


def _skill_raw_abstention_claim_passed(
    metrics: dict[str, Any],
    claims: dict[str, Any],
) -> bool:
    return _skill_candidate_claim_passed(
        metrics,
        claims,
    ) and all(
        metrics[f"skill.raw.{source}_hard_negative_abstention_rate"]
        >= float(claims[f"skill_raw_{source}_hard_negative_abstention_rate_min"])
        for source in ("local", "everos", "fused")
    )


def _validate_calibration_report(
    report: dict[str, Any],
    *,
    codeflow_commit: str,
    suite_digest: str,
    provider_config_digest: str,
) -> None:
    if (
        report.get("track") != "calibration"
        or report.get("ship_complete") is not True
        or report.get("production_evidence_valid") is not True
        or report.get("worktree_clean") is not True
        or report.get("codeflow_commit") != codeflow_commit
        or report.get("suite_digest") != suite_digest
        or report.get("provider_config_digest") != provider_config_digest
        or not isinstance(report.get("report_digest"), str)
        or len(str(report["report_digest"])) != 64
        or int(report.get("embedding_calls", 0)) <= 0
        or int(report.get("embedded_characters", 0)) <= 0
    ):
        raise SemanticCampaignError(
            "formal semantic run requires a clean, complete calibration "
            "from the same commit, suite, and embedding provider"
        )


def _expected_record_keys(
    fixture: SemanticFixture,
) -> list[dict[str, str]]:
    keys = [
        {
            "retrieval_suite_id": fixture.memory_suite_id,
            "query_id": query.query_id,
            "configuration_id": "user_memory_on",
        }
        for query in fixture.memory_queries
    ]
    keys.extend(
        {
            "retrieval_suite_id": fixture.skill_suite_id,
            "query_id": query.query_id,
            "configuration_id": configuration,
        }
        for query in fixture.skill_queries
        for configuration in ("local_only", "everos_only", "fused")
    )
    return sorted(
        keys,
        key=lambda key: (
            key["retrieval_suite_id"],
            key["query_id"],
            key["configuration_id"],
        ),
    )


def _record_key_tuple(
    record: dict[str, Any],
) -> tuple[str, str, str]:
    key = record.get("key")
    if not isinstance(key, dict):
        raise SemanticCampaignError("semantic record has no key")
    values = (
        key.get("retrieval_suite_id"),
        key.get("query_id"),
        key.get("configuration_id"),
    )
    if not all(isinstance(value, str) and value for value in values):
        raise SemanticCampaignError("semantic record key is invalid")
    return values


def _git_output(root: Path, *args: str) -> str:
    executable = shutil.which("git")
    if executable is None:
        raise SemanticCampaignError("git executable is unavailable")
    completed = subprocess.run(
        [executable, *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


@contextlib.contextmanager
def _exclusive_semantic_lock(
    output_root: Path,
    experiment_id: str,
):
    lock_root = output_root / ".locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / f"{experiment_id}.lock"
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(
                handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise SemanticCampaignError("semantic experiment already has an active writer") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _freeze_json(path: Path, value: dict[str, Any]) -> None:
    payload = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise SemanticCampaignError(f"immutable semantic artifact differs: {path}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, canonical_bytes(value))


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SemanticCampaignError(f"cannot read semantic artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise SemanticCampaignError(f"semantic artifact is not an object: {path}")
    return payload


def _semantic_record_seal(
    experiment_root: Path,
    *,
    semantic_schema: str = SEMANTIC_SCHEMA,
) -> dict[str, Any]:
    entries = []
    for path in sorted((experiment_root / "records").glob("*/*/*.json")):
        payload = _read_object(path)
        entries.append(
            {
                "path": path.relative_to(
                    experiment_root,
                ).as_posix(),
                "digest": canonical_digest(payload),
            }
        )
    return {
        "schema": (f"codeflow.codeflowbench.semantic-record-seal.{_semantic_version(semantic_schema)}"),
        "record_count": len(entries),
        "records_digest": canonical_digest(entries),
        "records": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the checkout-only CodeFlowBench semantic addendum.",
    )
    parser.add_argument(
        "--mode",
        choices=("preflight", "calibration", "formal", "rebuild"),
        default="preflight",
    )
    parser.add_argument(
        "--track",
        choices=("calibration", "formal"),
        default="calibration",
    )
    parser.add_argument(
        "--suite-version",
        choices=("v1", "v2"),
        default="v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
    )
    parser.add_argument("--budget-root", type=Path)
    parser.add_argument("--calibration-root", type=Path)
    parser.add_argument("--experiment-root", type=Path)
    args = parser.parse_args()
    suite_path = _SUITE_PATHS[args.suite_version]
    output_root = args.output_root or _OUTPUT_ROOTS[args.suite_version]
    try:
        if args.mode == "preflight":
            calibration = None
            if args.track == "formal":
                if args.calibration_root is None:
                    raise SemanticCampaignError("formal preflight requires --calibration-root")
                calibration = rebuild_semantic_report(args.calibration_root)
            outcome: dict[str, Any] = asdict(
                build_semantic_preflight(
                    args.track,
                    suite_path=suite_path,
                    calibration_report=calibration,
                    budget_root=args.budget_root,
                )
            )
        elif args.mode == "rebuild":
            if args.experiment_root is None:
                raise SemanticCampaignError("rebuild requires --experiment-root")
            outcome = rebuild_semantic_report(args.experiment_root)
        else:
            outcome = asyncio.run(
                run_semantic_track(
                    args.mode,
                    output_root=output_root,
                    suite_path=suite_path,
                    calibration_root=args.calibration_root,
                    budget_root=args.budget_root,
                )
            )
    except SemanticCampaignError as exc:
        parser.exit(2, f"CodeFlowBench semantic addendum aborted: {exc}\n")
    print(json.dumps(outcome, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "APPROVAL_ENV",
    "APPROVAL_ENV_V2",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_OUTPUT_ROOT_V2",
    "DEFAULT_SUITE_PATH_V2",
    "SemanticCampaignError",
    "SemanticPreflight",
    "build_semantic_preflight",
    "load_semantic_suite",
    "rebuild_semantic_report",
    "run_semantic_track",
]
