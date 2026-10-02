"""Historical EverOS task-effect harness retained fail-closed after removal."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Protocol

from sqlmodel import SQLModel

from benchmarks.codeflowbench.budget import (
    BudgetGuardedProvider,
    ProviderBudgetConfig,
    ProviderBudgetLedger,
    provider_call_budget_scope,
)
from benchmarks.codeflowbench.canonical import canonical_digest, to_primitive
from benchmarks.codeflowbench.coverage import assess_pair_coverage
from benchmarks.codeflowbench.host import RecordingOutlet, RuntimeTrialHost
from benchmarks.codeflowbench.isolation import TrialIsolation
from benchmarks.codeflowbench.protocol import TrialContext, TrialExecution
from benchmarks.codeflowbench.records import (
    DeliveryOutcome,
    TrialStatus,
    TurnTerminalState,
    VerificationState,
    VerifierResult,
)
from benchmarks.codeflowbench.schema import (
    JsonValue,
    PackDefinition,
    PairSpec,
    TaskSpec,
    VariantSpec,
)
from benchmarks.codeflowbench.usage import (
    RecordingProvider,
    UsageRecorder,
    usage_scope,
)
from codeflow.config.codeflow import MemoryConfig, CodeFlowConfig
from codeflow.config.schema import Config
from codeflow.context_engine.assembler import ContextAssembler
from codeflow.context_engine.segments import (
    BootstrapSegmentBuilder,
    IdentitySegmentBuilder,
    MemorySegmentBuilder,
)
from codeflow.context_engine.segments.curator import CuratorSegmentBuilder
from codeflow.memory_engine import Memory
from codeflow.plugin import PluginContext, ServiceLocator
from codeflow.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from codeflow.spine import ChatType, Origin, Source, TurnRequest
from codeflow.utils.helpers import estimate_prompt_tokens

from .models import SemanticMemoryEffectTask
from .semantic_runtime import CountingEmbeddingProvider

_TASK_SCHEMA = "codeflow.codeflowbench.semantic-memory-effect-tasks.v1"
_FORMAL_PACK_ID = "semantic-memory-effect-v1"
_CALIBRATION_PACK_ID = "semantic-memory-effect-calibration-v1"
_EXPECTED_TASK_COUNTS = {
    "calibration": 2,
    "formal": 8,
}
_EXPECTED_PAIR_COUNTS = {
    _CALIBRATION_PACK_ID: 4,
    _FORMAL_PACK_ID: 24,
}
_MEASURABLE_TRIAL_STATUSES = {
    TrialStatus.PASSED.value,
    TrialStatus.TASK_FAILED.value,
    TrialStatus.TASK_TIMEOUT.value,
}
_CONTROL_VARIANT_ID = "user_memory_off"
_TREATMENT_VARIANT_ID = "user_memory_on"
_TREATMENT_AXIS = "user_memory_recall"
_PROCESS_TERMINATION_GRACE_SECONDS = 5.0
_AGENT_PROVIDER_CALL_CEILING = 2
_EMBEDDING_PROVIDER_CALL_CEILING = 6
_TOTAL_PROVIDER_CALL_CEILING = _AGENT_PROVIDER_CALL_CEILING + _EMBEDDING_PROVIDER_CALL_CEILING
_GLOBAL_AUTHORIZED_HARD_CAP_CNY = 100.0
_MAIN_EXTERNAL_SERVICE_RESERVE_CNY = 5.0
_SEMANTIC_ADDENDUM_HARD_CAP_CNY = 5.0
_MAIN_PROVIDER_SPEND_CEILING_CNY = _GLOBAL_AUTHORIZED_HARD_CAP_CNY - _MAIN_EXTERNAL_SERVICE_RESERVE_CNY
_EMBEDDING_ACCOUNTING_BASIS = "shared_main_provider_ledger_conservative_proxy"
_DISABLED_TOOLS = [
    "ask_user",
    "edit_file",
    "exec",
    "find",
    "grep",
    "list_dir",
    "message",
    "read_file",
    "spawn",
    "understand_media",
    "web_fetch",
    "web_search",
]


class SemanticMemoryEffectRunner(Protocol):
    kind: str

    async def run(self, context: TrialContext) -> TrialExecution: ...


@lru_cache(maxsize=2)
def load_semantic_memory_effect_tasks(
    definition_kind: Literal["formal", "calibration"],
) -> tuple[SemanticMemoryEffectTask, ...]:
    path = Path(__file__).resolve().parents[2] / "tasks" / "memory_skill" / f"semantic_effect_{definition_kind}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != _TASK_SCHEMA:
        raise ValueError(f"unsupported semantic memory effect task schema: {path}")
    rows = payload.get("tasks")
    if not isinstance(rows, list):
        raise ValueError(f"semantic memory effect tasks must be a list: {path}")
    tasks = tuple(_task_from_payload(dict(row)) for row in rows)
    if len(tasks) != _EXPECTED_TASK_COUNTS[definition_kind]:
        raise ValueError(f"unexpected semantic memory effect task count: {path}")
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"duplicate semantic memory effect task id: {path}")
    return tasks


class SemanticMemoryEffectPack:
    def __init__(
        self,
        runner: SemanticMemoryEffectRunner,
        *,
        definition_kind: Literal["formal", "calibration"] = "formal",
    ) -> None:
        self._runner = runner
        self._definition_kind = definition_kind
        self._pack_id = _FORMAL_PACK_ID if definition_kind == "formal" else _CALIBRATION_PACK_ID
        self._tasks = load_semantic_memory_effect_tasks(definition_kind)

    def definition(self) -> PackDefinition:
        invariant_settings = {
            "context_pipeline": "identity-bootstrap-memory-curator",
            "memory_top_k": 5,
            "skill_sources": [],
            "tool_surface": ["write_file"],
        }
        return PackDefinition(
            pack_id=self._pack_id,
            tasks=tuple(
                TaskSpec(
                    task_id=task.task_id,
                    payload=to_primitive(task),
                )
                for task in self._tasks
            ),
            variants=(
                VariantSpec(
                    variant_id="user_memory_off",
                    settings={
                        "user_memory_recall": "disabled",
                        **invariant_settings,
                    },
                ),
                VariantSpec(
                    variant_id="user_memory_on",
                    settings={
                        "user_memory_recall": "enabled",
                        **invariant_settings,
                    },
                ),
            ),
            pairs=(
                PairSpec(
                    treatment_axis="user_memory_recall",
                    control_variant_id="user_memory_off",
                    treatment_variant_id="user_memory_on",
                ),
            ),
            identity={
                "runner_kind": self._runner.kind,
                "claim_reducer": "memory_effect_v1",
                "task_schema": _TASK_SCHEMA,
                "task_manifest_digest": canonical_digest(self._tasks),
                "task_effect_boundary": ("production_indexed_prior_session_memory_to_real_runtime_outcome"),
                "automatic_memory_extraction_claimed": False,
                "production_backend_required_for_claim": True,
                "real_embedding_required_for_claim": True,
                "real_agent_provider_required_for_claim": True,
            },
        )

    async def run_trial(self, context: TrialContext) -> TrialExecution:
        return await self._runner.run(context)


def create_semantic_memory_effect_pack(
    runner: SemanticMemoryEffectRunner,
) -> SemanticMemoryEffectPack:
    return SemanticMemoryEffectPack(runner)


def create_semantic_memory_effect_calibration_pack(
    runner: SemanticMemoryEffectRunner,
) -> SemanticMemoryEffectPack:
    return SemanticMemoryEffectPack(
        runner,
        definition_kind="calibration",
    )


def reduce_semantic_memory_effect_claims(
    trial_records: (Mapping[Any, Mapping[str, Any]] | Iterable[Mapping[str, Any]]),
    pair_results: (Mapping[Any, Mapping[str, Any]] | Iterable[Mapping[str, Any]]),
) -> dict[str, JsonValue]:
    trials = _materialize_records(trial_records)
    pairs = _materialize_records(pair_results)
    relevant_pack_ids = {
        pack_id
        for record in (*trials, *pairs)
        if (
            pack_id := str(
                _record_mapping(record.get("key")).get("pack_id", ""),
            )
        )
        in _EXPECTED_PAIR_COUNTS
    }
    expected_pairs = sum(_EXPECTED_PAIR_COUNTS[pack_id] for pack_id in relevant_pack_ids)

    by_trial: dict[
        tuple[str, str, str, int, str],
        Mapping[str, Any],
    ] = {}
    attempt_consistency_valid = True
    for trial in trials:
        key = _record_mapping(trial.get("key"))
        pack_id = str(key.get("pack_id", ""))
        variant_id = str(key.get("variant_id", ""))
        if pack_id not in _EXPECTED_PAIR_COUNTS or variant_id not in {_CONTROL_VARIANT_ID, _TREATMENT_VARIANT_ID}:
            continue
        identity = (
            str(key.get("experiment_id", "")),
            pack_id,
            str(key.get("task_id", "")),
            int(key.get("repetition", -1)),
            variant_id,
        )
        if identity in by_trial:
            attempt_consistency_valid = False
            continue
        by_trial[identity] = trial

    declared_pairs: list[Mapping[str, Any]] = []
    variant_axis_valid = True
    for pair in pairs:
        key = _record_mapping(pair.get("key"))
        pack_id = str(key.get("pack_id", ""))
        if pack_id not in _EXPECTED_PAIR_COUNTS:
            continue
        if (
            str(key.get("treatment_axis", "")) != _TREATMENT_AXIS
            or str(key.get("control_variant_id", "")) != _CONTROL_VARIANT_ID
            or str(key.get("treatment_variant_id", "")) != _TREATMENT_VARIANT_ID
        ):
            variant_axis_valid = False
            continue
        declared_pairs.append(pair)

    accepted_pair_identities: set[tuple[str, str, str, int]] = set()
    deltas_by_task: dict[tuple[str, str], list[int]] = defaultdict(
        list,
    )
    losses_by_task: dict[tuple[str, str], int] = defaultdict(int)
    control_records: list[Mapping[str, Any]] = []
    treatment_records: list[Mapping[str, Any]] = []
    valid_pairs = 0
    control_passes = 0
    treatment_passes = 0
    net_gains = 0
    seen_pair_identities: set[tuple[str, str, str, int]] = set()
    for pair in declared_pairs:
        key = _record_mapping(pair.get("key"))
        experiment_id = str(key.get("experiment_id", ""))
        pack_id = str(key.get("pack_id", ""))
        task_id = str(key.get("task_id", ""))
        repetition = int(key.get("repetition", -1))
        pair_identity = (
            experiment_id,
            pack_id,
            task_id,
            repetition,
        )
        if pair_identity in seen_pair_identities:
            attempt_consistency_valid = False
            continue
        seen_pair_identities.add(pair_identity)
        control = by_trial.get(
            (
                experiment_id,
                pack_id,
                task_id,
                repetition,
                _CONTROL_VARIANT_ID,
            )
        )
        treatment = by_trial.get(
            (
                experiment_id,
                pack_id,
                task_id,
                repetition,
                _TREATMENT_VARIANT_ID,
            )
        )
        if control is None or treatment is None:
            attempt_consistency_valid = False
            continue
        actual_diff = _observed_variant_diff(control, treatment)
        if set(actual_diff) != {_TREATMENT_AXIS} or pair.get("actual_variant_diff") != actual_diff:
            variant_axis_valid = False
            continue
        if pair.get("valid") is not True:
            continue
        selected_attempt = pair.get("selected_block_attempt")
        if not isinstance(selected_attempt, int) or isinstance(selected_attempt, bool) or selected_attempt < 1:
            attempt_consistency_valid = False
            continue
        if (
            control.get("selected_block_attempt") != selected_attempt
            or treatment.get("selected_block_attempt") != selected_attempt
            or not _trial_belongs_to_effect_pair(control, key)
            or not _trial_belongs_to_effect_pair(treatment, key)
            or not _effect_pair_plan_digest_matches(
                pair,
                control,
                treatment,
            )
        ):
            attempt_consistency_valid = False
            continue
        if not _effect_trial_is_measurable(control) or not _effect_trial_is_measurable(treatment):
            continue

        accepted_pair_identities.add(pair_identity)
        valid_pairs += 1
        control_passed = str(control.get("status")) == TrialStatus.PASSED.value
        treatment_passed = str(treatment.get("status")) == TrialStatus.PASSED.value
        control_passes += int(control_passed)
        treatment_passes += int(treatment_passed)
        delta = int(treatment_passed) - int(control_passed)
        net_gains += delta
        task_identity = (pack_id, task_id)
        deltas_by_task[task_identity].append(delta)
        if delta < 0:
            losses_by_task[task_identity] += 1
        control_records.append(control)
        treatment_records.append(treatment)

    coverage = assess_pair_coverage(
        expected_pairs=expected_pairs,
        planned_pair_keys=(
            (
                ":".join(
                    (
                        str(
                            _record_mapping(pair.get("key")).get(
                                "experiment_id",
                                "",
                            )
                        ),
                        str(
                            _record_mapping(pair.get("key")).get(
                                "pack_id",
                                "",
                            )
                        ),
                        str(
                            _record_mapping(pair.get("key")).get(
                                "task_id",
                                "",
                            )
                        ),
                    )
                ),
                int(
                    _record_mapping(pair.get("key")).get(
                        "repetition",
                        -1,
                    )
                ),
            )
            for pair in declared_pairs
        ),
        valid_pair_keys=(
            (
                ":".join((experiment_id, pack_id, task_id)),
                repetition,
            )
            for experiment_id, pack_id, task_id, repetition in accepted_pair_identities
        ),
    )
    attempt_consistency_valid = bool(declared_pairs) and (attempt_consistency_valid)
    variant_axis_valid = bool(declared_pairs) and variant_axis_valid
    measurement_valid = bool(relevant_pack_ids) and all(
        (
            coverage.valid,
            attempt_consistency_valid,
            variant_axis_valid,
        )
    )

    positive_tasks = sum(sum(deltas) > 0 for deltas in deltas_by_task.values())
    regressed_two_of_three = sum(losses >= 2 for losses in losses_by_task.values())
    control_pass_rate = _effect_rate(control_passes, valid_pairs)
    treatment_pass_rate = _effect_rate(treatment_passes, valid_pairs)
    success_delta_pp = (
        (treatment_pass_rate - control_pass_rate) * 100.0
        if treatment_pass_rate is not None and control_pass_rate is not None
        else None
    )

    memory_off_evidence_complete = bool(control_records) and all(
        _non_negative_integer_metric(
            record,
            "memory.user_recall_calls",
        )
        is not None
        for record in control_records
    )
    memory_off_backend_call_count = sum(
        value
        for record in control_records
        if (
            value := _non_negative_integer_metric(
                record,
                "memory.user_recall_calls",
            )
        )
        is not None
    )
    memory_off_backend_calls_zero = memory_off_evidence_complete and memory_off_backend_call_count == 0
    treatment_target_hit_evidence_complete = bool(
        treatment_records,
    ) and all(
        isinstance(
            _record_mapping(record.get("metrics")).get(
                "memory.target_hit",
            ),
            bool,
        )
        for record in treatment_records
    )
    treatment_target_hits = sum(
        _record_mapping(record.get("metrics")).get(
            "memory.target_hit",
        )
        is True
        for record in treatment_records
    )
    treatment_target_hits_complete = treatment_target_hit_evidence_complete and treatment_target_hits == valid_pairs

    evidence_records = [*control_records, *treatment_records]
    production_memory_evidence_valid = bool(evidence_records) and all(
        _production_memory_effect_evidence_valid(record) for record in evidence_records
    )
    production_real_agent_evidence_valid = bool(
        evidence_records,
    ) and all(_production_real_agent_evidence_valid(record) for record in evidence_records)
    production_embedding_evidence_valid = bool(
        evidence_records,
    ) and all(_production_embedding_effect_evidence_valid(record) for record in evidence_records)
    production_cost_evidence_valid = bool(evidence_records) and all(
        _production_effect_cost_evidence_valid(record) for record in evidence_records
    )
    production_model_evidence_valid = bool(evidence_records) and all(
        _production_effect_model_evidence_valid(record) for record in evidence_records
    )
    real_agent_task_effect_evidence_valid = all(
        (
            production_memory_evidence_valid,
            production_real_agent_evidence_valid,
            production_embedding_evidence_valid,
            production_cost_evidence_valid,
            production_model_evidence_valid,
        )
    )

    formal_claim_denominator = relevant_pack_ids == {_FORMAL_PACK_ID}
    claim_contract_valid = all(
        (
            measurement_valid,
            formal_claim_denominator,
            net_gains >= 6,
            positive_tasks >= 4,
            regressed_two_of_three == 0,
            memory_off_backend_calls_zero,
            treatment_target_hits_complete,
        )
    )
    claim_eligible = claim_contract_valid and real_agent_task_effect_evidence_valid
    return {
        "semantic_memory_e2e.planned_pairs": expected_pairs,
        "semantic_memory_e2e.valid_pairs": valid_pairs,
        "semantic_memory_e2e.coverage_valid": coverage.valid,
        "semantic_memory_e2e.attempt_consistency_valid": (attempt_consistency_valid),
        "semantic_memory_e2e.variant_axis_valid": variant_axis_valid,
        "semantic_memory_e2e.control_passes": control_passes,
        "semantic_memory_e2e.treatment_passes": treatment_passes,
        "semantic_memory_e2e.control_pass_rate": control_pass_rate,
        "semantic_memory_e2e.treatment_pass_rate": (treatment_pass_rate),
        "semantic_memory_e2e.success_delta_pp": success_delta_pp,
        "semantic_memory_e2e.net_verifier_gains": net_gains,
        "semantic_memory_e2e.positive_tasks": positive_tasks,
        "semantic_memory_e2e.tasks_with_two_of_three_regressions": (regressed_two_of_three),
        "semantic_memory_e2e.no_two_of_three_regressions": (regressed_two_of_three == 0),
        "semantic_memory_e2e.memory_off_backend_call_evidence_complete": (memory_off_evidence_complete),
        "semantic_memory_e2e.memory_off_backend_call_count": (memory_off_backend_call_count),
        "semantic_memory_e2e.memory_off_backend_calls_zero": (memory_off_backend_calls_zero),
        "semantic_memory_e2e.treatment_target_hit_evidence_complete": (treatment_target_hit_evidence_complete),
        "semantic_memory_e2e.treatment_target_hits": (treatment_target_hits),
        "semantic_memory_e2e.treatment_target_hits_complete": (treatment_target_hits_complete),
        "semantic_memory_e2e.production_memory_evidence_valid": (production_memory_evidence_valid),
        "semantic_memory_e2e.production_real_agent_evidence_valid": (production_real_agent_evidence_valid),
        "semantic_memory_e2e.production_embedding_evidence_valid": (production_embedding_evidence_valid),
        "semantic_memory_e2e.production_cost_evidence_valid": (production_cost_evidence_valid),
        "semantic_memory_e2e.production_model_evidence_valid": (production_model_evidence_valid),
        "semantic_memory_e2e.real_agent_task_effect_evidence_valid": (real_agent_task_effect_evidence_valid),
        "semantic_memory_e2e.claim_contract_valid": (claim_contract_valid),
        "semantic_memory_e2e.real_agent_task_effect_claim_eligible": (claim_eligible),
        "semantic_memory_e2e.measurement_valid": measurement_valid,
        "semantic_memory_effect.measurement_valid": measurement_valid,
    }


def _materialize_records(
    records: (Mapping[Any, Mapping[str, Any]] | Iterable[Mapping[str, Any]]),
) -> list[Mapping[str, Any]]:
    if isinstance(records, Mapping):
        return list(records.values())
    return list(records)


def _record_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _observed_variant_diff(
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
) -> dict[str, JsonValue]:
    control_settings = _record_mapping(
        control.get("observed_variant_settings"),
    )
    treatment_settings = _record_mapping(
        treatment.get("observed_variant_settings"),
    )
    return {
        key: {
            "control": control_settings.get(key),
            "treatment": treatment_settings.get(key),
        }
        for key in sorted(
            set(control_settings) | set(treatment_settings),
        )
        if control_settings.get(key) != treatment_settings.get(key)
    }


def _trial_belongs_to_effect_pair(
    trial: Mapping[str, Any],
    pair_key: Mapping[str, Any],
) -> bool:
    memberships = trial.get("pair_memberships", ())
    return isinstance(memberships, list | tuple) and any(
        _record_mapping(membership) == pair_key for membership in memberships
    )


def _effect_pair_plan_digest_matches(
    pair: Mapping[str, Any],
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
) -> bool:
    digest = pair.get("plan_digest")
    return (
        isinstance(digest, str)
        and bool(digest)
        and control.get("plan_digest") == digest
        and treatment.get("plan_digest") == digest
    )


def _effect_trial_is_measurable(
    record: Mapping[str, Any],
) -> bool:
    return str(record.get("status")) in _MEASURABLE_TRIAL_STATUSES


def _non_negative_integer_metric(
    record: Mapping[str, Any],
    name: str,
) -> int | None:
    value = _record_mapping(record.get("metrics")).get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return None
    return value


def _production_memory_effect_evidence_valid(
    record: Mapping[str, Any],
) -> bool:
    metrics = _record_mapping(record.get("metrics"))
    return (
        metrics.get("memory.backend_class") == "EverosBackend"
        and metrics.get("memory.backend_adapter") == "production"
        and metrics.get("memory.backend_adapter_class") == "_RealEverosAdapter"
        and metrics.get("runtime.fresh_process") is True
    )


def _production_real_agent_evidence_valid(
    record: Mapping[str, Any],
) -> bool:
    metrics = _record_mapping(record.get("metrics"))
    return (
        metrics.get("paid_campaign_eligible") is True and metrics.get("real_agent_task_effect_claim_eligible") is True
    )


def _production_embedding_effect_evidence_valid(
    record: Mapping[str, Any],
) -> bool:
    metrics = _record_mapping(record.get("metrics"))
    return (
        metrics.get("embedding.real_provider") is True
        and _non_empty_string(
            metrics.get("embedding.provider_identity"),
        )
        and _non_empty_string(
            metrics.get("embedding.provider_config_digest"),
        )
    )


def _production_effect_cost_evidence_valid(
    record: Mapping[str, Any],
) -> bool:
    metrics = _record_mapping(record.get("metrics"))
    return (
        metrics.get("usage.complete") is True
        and metrics.get("cost.agent_provider_complete") is True
        and metrics.get("cost.embedding_budget_complete") is True
        and metrics.get("cost.ledger_reconciliation_valid") is True
        and metrics.get("cost.global_authority_proof_valid") is True
        and metrics.get("cost.embedding_accounting_basis") == _EMBEDDING_ACCOUNTING_BASIS
        and metrics.get("cost.embedding_uses_shared_main_ledger") is True
        and metrics.get("cost.embedding_has_separate_authority") is False
        and metrics.get("cost.main_provider_spend_ceiling_cny") == _MAIN_PROVIDER_SPEND_CEILING_CNY
        and metrics.get("cost.semantic_addendum_hard_cap_cny") == _SEMANTIC_ADDENDUM_HARD_CAP_CNY
        and metrics.get("cost.global_authorized_hard_cap_cny") == _GLOBAL_AUTHORIZED_HARD_CAP_CNY
        and metrics.get("cost.complete") is True
    )


def _production_effect_model_evidence_valid(
    record: Mapping[str, Any],
) -> bool:
    metrics = _record_mapping(record.get("metrics"))
    requested_model = metrics.get("provider.kind")
    actual_models = metrics.get("provider.actual_models")
    return (
        _non_empty_string(requested_model)
        and isinstance(actual_models, list | tuple)
        and bool(actual_models)
        and all(_non_empty_string(model) for model in actual_models)
        and all(
            _provider_model_equivalent(
                str(requested_model),
                str(model),
            )
            for model in actual_models
        )
    )


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _effect_rate(
    numerator: int,
    denominator: int,
) -> float | None:
    return numerator / denominator if denominator else None


class ScriptedSemanticMemoryEffectRunner:
    kind = "semantic_memory_effect_scripted_fake_embedding"

    async def run(self, context: TrialContext) -> TrialExecution:
        return await _run_semantic_memory_effect_trial(
            context,
            provider_spec={
                "mode": "scripted",
                "provider_identity": "scripted/codeflowbench-semantic-memory-effect",
                "paid_campaign_eligible": False,
            },
            embedding_spec={
                "mode": "fake",
                "provider_identity": "fake/hash-bag-1024",
                "provider_config_digest": canonical_digest(
                    "fake/hash-bag-1024-v1",
                ),
                "maximum_characters": 20_000,
                "maximum_calls": _EMBEDDING_PROVIDER_CALL_CEILING,
                "characters_per_token": 2.0,
            },
        )


class ProductionSemanticMemoryEffectRunner:
    kind = "semantic_memory_effect_real_provider_real_embedding"

    def __init__(
        self,
        *,
        config: Config,
        codeflow_config: CodeFlowConfig,
        provider: LLMProvider,
        maximum_embedding_characters_per_trial: int = 20_000,
    ) -> None:
        if not isinstance(provider, BudgetGuardedProvider):
            raise ValueError(
                "ProductionSemanticMemoryEffectRunner requires BudgetGuardedProvider",
            )
        if maximum_embedding_characters_per_trial < 1:
            raise ValueError(
                "maximum_embedding_characters_per_trial must be positive",
            )
        self._config = config
        self._codeflow_config = codeflow_config
        self._provider = provider
        (
            self._embedding_environment,
            identity,
            self._embedding_config,
        ) = _configured_embedding_environment()
        self._embedding_identity = identity
        self._maximum_embedding_characters = maximum_embedding_characters_per_trial

    async def run(self, context: TrialContext) -> TrialExecution:
        pack_budget = context.experiment.execution.provider_trial_budget_for(
            context.key.pack_id,
        )
        if pack_budget is None:
            return _preparation_failure(
                context,
                "semantic_memory_effect_provider_budget_missing",
            )
        if (
            pack_budget.max_provider_calls_per_trial != _TOTAL_PROVIDER_CALL_CEILING
            or pack_budget.max_input_tokens_per_call != 15_000
            or pack_budget.max_output_tokens_per_call != 1_500
        ):
            return _preparation_failure(
                context,
                "semantic_memory_effect_provider_budget_not_frozen",
            )
        ledger = self._provider.ledger
        with tempfile.TemporaryDirectory(
            prefix="codeflowbench-semantic-memory-private-",
        ) as private_root:
            config_path = Path(private_root) / "runtime-config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "config": self._config.model_dump(mode="json"),
                        "codeflow_config": self._codeflow_config.model_dump(mode="json"),
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            config_path.chmod(0o600)
            trial_id = (
                f"{context.experiment_id}/"
                f"{context.key.pack_id}/{context.key.task_id}/"
                f"{context.key.variant_id}/{context.key.repetition}/"
                f"{context.block_attempt}"
            )
            return await _run_semantic_memory_effect_trial(
                context,
                provider_spec={
                    "mode": "real",
                    "provider_identity": str(
                        self._config.agents.defaults.model,
                    ),
                    "private_config_path": str(config_path),
                    "ledger_path": str(ledger.path.resolve()),
                    "ledger_config": asdict(ledger.config),
                    "trial_id": trial_id,
                    "max_logical_calls": (_AGENT_PROVIDER_CALL_CEILING),
                    "max_attempts_per_call": (context.experiment.execution.provider_call_max_attempts),
                    "max_input_tokens_per_call": (pack_budget.max_input_tokens_per_call),
                    "max_output_tokens_per_call": (pack_budget.max_output_tokens_per_call),
                    "paid_campaign_eligible": True,
                },
                embedding_spec={
                    "mode": "real",
                    "provider_identity": self._embedding_identity,
                    "provider_config_digest": canonical_digest(self._embedding_config),
                    "maximum_characters": (self._maximum_embedding_characters),
                    "maximum_calls": _EMBEDDING_PROVIDER_CALL_CEILING,
                    "maximum_input_tokens_per_call": (pack_budget.max_input_tokens_per_call),
                    "characters_per_token": 2.0,
                    "budget_trial_id_prefix": (f"{trial_id}:embedding"),
                    "ledger_path": str(ledger.path.resolve()),
                    "ledger_config": asdict(ledger.config),
                },
                private_environment=self._embedding_environment,
            )


async def _run_semantic_memory_effect_trial(
    context: TrialContext,
    *,
    provider_spec: dict[str, Any],
    embedding_spec: dict[str, Any],
    private_environment: dict[str, str] | None = None,
) -> TrialExecution:
    task = _task_from_payload(dict(context.task.payload))
    attempt_id = f"{task.task_id}-{context.variant.variant_id}-r{context.key.repetition}-b{context.block_attempt}"
    isolation = TrialIsolation.create(
        (context.experiment.output_root / ".codeflowbench-semantic-memory-effect" / context.experiment_id),
        attempt_id,
    )
    isolation.prepare()
    spec_path = isolation.root / "worker-spec.json"
    seed_result_path = isolation.root / "seed-result.json"
    spec_path.write_text(
        json.dumps(
            {
                "task": to_primitive(task),
                "variant_settings": dict(context.variant.settings),
                "workspace": str(isolation.workspace),
                "seed_result_path": str(seed_result_path),
                "provider": provider_spec,
                "embedding": embedding_spec,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    trial_budget_before = _shared_budget_snapshot(provider_spec)
    started = time.perf_counter()
    seed = await _run_worker_stage(
        stage="seed",
        spec_path=spec_path,
        result_path=seed_result_path,
        isolation=isolation,
        private_environment=private_environment,
    )
    if seed.get("status") != "completed":
        return _infrastructure_failure(
            context,
            isolation,
            f"semantic_seed_worker_failed:{seed.get('error_type', 'unknown')}",
        )
    remaining_embedding_characters = int(embedding_spec["maximum_characters"]) - int(seed["embedded_characters"])
    if remaining_embedding_characters < 1:
        return _infrastructure_failure(
            context,
            isolation,
            "semantic_embedding_trial_budget_exhausted_after_seed",
        )
    remaining_embedding_calls = int(embedding_spec["maximum_calls"]) - int(seed["embedding_calls"])
    if remaining_embedding_calls < 1:
        return _infrastructure_failure(
            context,
            isolation,
            "semantic_embedding_call_budget_exhausted_after_seed",
        )
    evaluation_spec = json.loads(
        spec_path.read_text(encoding="utf-8"),
    )
    evaluation_spec["embedding"]["maximum_characters"] = remaining_embedding_characters
    evaluation_spec["embedding"]["maximum_calls"] = remaining_embedding_calls
    spec_path.write_text(
        json.dumps(
            evaluation_spec,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    evaluation_result_path = isolation.root / "evaluation-result.json"
    evaluation = await _run_worker_stage(
        stage="evaluation",
        spec_path=spec_path,
        result_path=evaluation_result_path,
        isolation=isolation,
        private_environment=private_environment,
    )
    if evaluation.get("status") != "completed":
        return _infrastructure_failure(
            context,
            isolation,
            (f"semantic_evaluation_worker_failed:{evaluation.get('error_type', 'unknown')}"),
        )

    elapsed_ms = (time.perf_counter() - started) * 1_000
    verification = _verify_result(isolation.workspace, task)
    findings = _integrity_findings(
        context=context,
        seed=seed,
        evaluation=evaluation,
    )
    runtime_state = TurnTerminalState(str(evaluation["runtime_state"]))
    delivery_state = DeliveryOutcome(str(evaluation["delivery_state"]))
    failure_category = _safe_failure_category(evaluation)
    if findings:
        status = TrialStatus.INFRASTRUCTURE_FAILURE
        verification = VerifierResult(
            state=VerificationState.NOT_RUN,
            findings=tuple(findings),
        )
    elif failure_category == "task_budget_exhausted":
        status = TrialStatus.TASK_TIMEOUT
    elif runtime_state is TurnTerminalState.PROVIDER_FAILED:
        status = TrialStatus.PROVIDER_FAILURE
    elif runtime_state is TurnTerminalState.CANCELLED:
        status = TrialStatus.CANCELLED
    elif runtime_state not in {
        TurnTerminalState.COMPLETED,
        TurnTerminalState.COMPLETED_WITH_TOOL_FAILURE,
    }:
        status = TrialStatus.TASK_FAILED
    elif verification.state is VerificationState.PASSED:
        status = TrialStatus.PASSED
    else:
        status = TrialStatus.TASK_FAILED

    usage = dict(evaluation["usage"])
    embedding_calls = int(seed["embedding_calls"]) + int(
        evaluation["embedding_calls"],
    )
    embedded_characters = int(seed["embedded_characters"]) + int(
        evaluation["embedded_characters"],
    )
    embedding_budgeted_input_tokens = int(
        seed["embedding_budgeted_input_tokens"],
    ) + int(evaluation["embedding_budgeted_input_tokens"])
    fresh_process = int(seed["pid"]) != int(evaluation["pid"])
    production_backend = (
        evaluation["backend_class"] == "EverosBackend" and evaluation["backend_adapter_class"] == "_RealEverosAdapter"
    )
    budget_snapshot = _shared_budget_snapshot(provider_spec)
    actual_models_valid = bool(evaluation["actual_models"]) and all(
        _provider_model_equivalent(
            str(provider_spec["provider_identity"]),
            str(actual_model),
        )
        for actual_model in evaluation["actual_models"]
    )
    budget_complete = budget_snapshot is None or (
        budget_snapshot.accounting_complete
        and budget_snapshot.open_reservations == 0
        and budget_snapshot.total_committed_cny <= budget_snapshot.hard_cap_cny
    )
    embedding_budgeted_cny = _embedding_budgeted_cny(
        provider_spec=provider_spec,
        input_tokens=embedding_budgeted_input_tokens,
    )
    trial_provider_charged_cny = (
        budget_snapshot.provider_charged_cny - trial_budget_before.provider_charged_cny
        if budget_snapshot is not None and trial_budget_before is not None
        else 0.0
    )
    evaluation_provider_charged_cny = float(
        evaluation["provider_charged_cny"],
    )
    evaluation_embedding_budgeted_cny = _embedding_budgeted_cny(
        provider_spec=provider_spec,
        input_tokens=int(
            evaluation["embedding_budgeted_input_tokens"],
        ),
    )
    agent_provider_charged_cny = max(
        0.0,
        evaluation_provider_charged_cny - evaluation_embedding_budgeted_cny,
    )
    ledger_reconciliation_valid = math.isclose(
        trial_provider_charged_cny,
        agent_provider_charged_cny + embedding_budgeted_cny,
        rel_tol=0,
        abs_tol=1e-12,
    )
    global_authority_proof_valid = _global_authority_proof_valid(
        budget_snapshot,
    )
    real_effect_eligible = bool(
        production_backend
        and fresh_process
        and seed["real_embedding"]
        and evaluation["real_embedding"]
        and evaluation["real_agent_provider"]
        and usage["usage_complete"]
        and evaluation["agent_cost_complete"]
        and actual_models_valid
        and budget_complete
        and ledger_reconciliation_valid
        and global_authority_proof_valid
    )
    relative_root = isolation.root.relative_to(
        context.experiment.output_root,
    )
    return TrialExecution(
        status=status,
        runtime_state=runtime_state,
        delivery_state=delivery_state,
        verification=verification,
        observed_variant_settings=dict(context.variant.settings),
        metrics={
            "memory.user_recall_calls": int(
                evaluation["user_recall_calls"],
            ),
            "memory.suppressed_user_recall_calls": int(
                evaluation["suppressed_user_recall_calls"],
            ),
            "memory.hits": int(evaluation["memory_hits"]),
            "memory.hit_ids": list(evaluation["memory_hit_ids"]),
            "memory.target_hit": bool(evaluation["target_hit"]),
            "memory.backend_class": str(evaluation["backend_class"]),
            "memory.backend_adapter": ("production" if production_backend else "non_production"),
            "memory.backend_adapter_class": str(
                evaluation["backend_adapter_class"],
            ),
            "memory.seed_storage_path": ("EpisodeWriter->CascadeOrchestrator"),
            "memory.seed_indexed_rows": int(seed["indexed_rows"]),
            "runtime.prior_session_id": task.prior_session_id,
            "runtime.evaluation_session_id": str(
                evaluation["conversation"],
            ),
            "runtime.seed_pid": int(seed["pid"]),
            "runtime.evaluation_pid": int(evaluation["pid"]),
            "runtime.fresh_process": fresh_process,
            "runtime.end_to_end_latency_ms": elapsed_ms,
            "runtime.failure_category": failure_category,
            "provider.kind": str(evaluation["provider_identity"]),
            "provider.actual_models": list(
                evaluation["actual_models"],
            ),
            "provider.memory_observed": evaluation["provider_memory_observed"],
            "embedding.calls": embedding_calls,
            "embedding.characters": embedded_characters,
            "embedding.budgeted_input_tokens": (embedding_budgeted_input_tokens),
            "embedding.trial_character_cap": int(
                embedding_spec["maximum_characters"],
            ),
            "embedding.provider_identity": str(
                evaluation["embedding_provider_identity"],
            ),
            "embedding.provider_config_digest": str(
                evaluation["embedding_provider_config_digest"],
            ),
            "embedding.real_provider": bool(seed["real_embedding"] and evaluation["real_embedding"]),
            "paid_campaign_eligible": bool(evaluation["paid_campaign_eligible"]),
            "real_agent_task_effect_claim_eligible": (real_effect_eligible),
            "cost.agent_provider_complete": bool(
                evaluation["agent_cost_complete"],
            ),
            "cost.embedding_budget_complete": bool(
                budget_complete and ledger_reconciliation_valid and global_authority_proof_valid
            ),
            "cost.ledger_reconciliation_valid": (ledger_reconciliation_valid),
            "cost.global_authority_proof_valid": (global_authority_proof_valid),
            "cost.embedding_accounting_basis": (_EMBEDDING_ACCOUNTING_BASIS),
            "cost.embedding_uses_shared_main_ledger": (budget_snapshot is not None),
            "cost.embedding_has_separate_authority": False,
            "cost.main_hard_cap_cny": (budget_snapshot.hard_cap_cny if budget_snapshot is not None else None),
            "cost.main_external_service_reserve_cny": (
                budget_snapshot.external_service_reserve_cny if budget_snapshot is not None else None
            ),
            "cost.main_provider_spend_ceiling_cny": (_MAIN_PROVIDER_SPEND_CEILING_CNY),
            "cost.semantic_addendum_hard_cap_cny": (_SEMANTIC_ADDENDUM_HARD_CAP_CNY),
            "cost.global_authorized_hard_cap_cny": (_GLOBAL_AUTHORIZED_HARD_CAP_CNY),
            "cost.complete": bool(
                evaluation["agent_cost_complete"]
                and budget_complete
                and ledger_reconciliation_valid
                and global_authority_proof_valid
            ),
            "cost.agent_provider_charged_cny": (agent_provider_charged_cny),
            "cost.embedding_shared_ledger_budgeted_cny": (embedding_budgeted_cny),
            "cost.trial_provider_charged_cny": (trial_provider_charged_cny),
            "cost.embedding_accounting": (
                "shared_main_provider_ledger" if evaluation["real_embedding"] else "zero_cost_fixture"
            ),
            "usage.main_agent_input_tokens": usage["input_tokens"],
            "usage.trial_total_input_tokens": usage["input_tokens"],
            "usage.trial_total_output_tokens": usage["output_tokens"],
            "usage.trial_total_tokens": usage["total_tokens"],
            "usage.model_calls": int(usage["calls"]),
            "usage.complete": bool(usage["usage_complete"]),
        },
        findings=tuple(findings),
        artifact_refs=(relative_root.as_posix(),),
    )


async def _run_worker_stage(
    *,
    stage: str,
    spec_path: Path,
    result_path: Path,
    isolation: TrialIsolation,
    private_environment: dict[str, str] | None,
) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[4]
    environment = {
        **os.environ,
        **isolation.child_environment(),
        **(private_environment or {}),
        "EVEROS_MEMORIZE__MODE": "agent",
        "CODEFLOW_TRACING_DIR": str(isolation.trace_root),
    }
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(repository_root)
        if not existing_pythonpath
        else os.pathsep.join((str(repository_root), existing_pythonpath))
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "benchmarks.codeflowbench.packs.memory_skill.semantic_effect",
        "--worker-stage",
        stage,
        "--spec",
        str(spec_path),
        "--result",
        str(result_path),
        cwd=repository_root,
        env=environment,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await _communicate_and_reap(process)
    if not result_path.exists():
        return {
            "status": "infrastructure_failure",
            "error_type": f"worker_exit_{process.returncode}",
        }
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "status": "infrastructure_failure",
            "error_type": "worker_result_corrupt",
        }
    if process.returncode != 0:
        result["status"] = "infrastructure_failure"
    return result


async def _communicate_and_reap(
    process: asyncio.subprocess.Process,
) -> None:
    try:
        await process.communicate()
    except BaseException:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(
                    process.wait(),
                    timeout=_PROCESS_TERMINATION_GRACE_SECONDS,
                )
            except TimeoutError:
                if process.returncode is None:
                    process.kill()
                await process.wait()
        raise


def _verify_result(
    workspace: Path,
    task: SemanticMemoryEffectTask,
) -> VerifierResult:
    artifact = workspace / "result.json"
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return VerifierResult(
            state=VerificationState.FAILED,
            findings=("result_artifact_missing_or_invalid",),
        )
    expected = {
        "task_id": task.task_id,
        "region": task.expected_region,
        "retention": task.expected_retention,
        "approval_code": task.expected_approval_code,
    }
    if not isinstance(payload, dict):
        return VerifierResult(
            state=VerificationState.FAILED,
            findings=("result_artifact_mismatch",),
            metrics={
                "region_matched": False,
                "retention_matched": False,
                "approval_code_matched": False,
            },
        )
    if payload != expected:
        return VerifierResult(
            state=VerificationState.FAILED,
            findings=("result_artifact_mismatch",),
            metrics={
                "region_matched": (payload.get("region") == task.expected_region),
                "retention_matched": (payload.get("retention") == task.expected_retention),
                "approval_code_matched": (payload.get("approval_code") == task.expected_approval_code),
            },
        )
    return VerifierResult(
        state=VerificationState.PASSED,
        metrics={
            "region_matched": True,
            "retention_matched": True,
            "approval_code_matched": True,
        },
    )


def _integrity_findings(
    *,
    context: TrialContext,
    seed: dict[str, Any],
    evaluation: dict[str, Any],
) -> list[str]:
    findings: list[str] = []
    if int(seed["pid"]) == int(evaluation["pid"]):
        findings.append("evaluation_not_fresh_process")
    if seed["prior_session_id"] == evaluation["conversation"]:
        findings.append("prior_and_evaluation_session_not_distinct")
    if seed["storage_path"] != "EpisodeWriter->CascadeOrchestrator":
        findings.append("semantic_seed_did_not_use_production_storage_path")
    if evaluation["backend_class"] != "EverosBackend":
        findings.append("semantic_effect_backend_not_everos")
    if evaluation["backend_adapter_class"] != "_RealEverosAdapter":
        findings.append("semantic_effect_adapter_not_production")
    recall_setting = context.variant.settings["user_memory_recall"]
    delegated = int(evaluation["user_recall_calls"])
    suppressed = int(evaluation["suppressed_user_recall_calls"])
    if recall_setting == "disabled" and delegated != 0:
        findings.append("memory_off_delegated_user_recall")
    if recall_setting == "disabled" and suppressed == 0:
        findings.append("memory_off_suppression_not_observed")
    if recall_setting == "enabled" and delegated == 0:
        findings.append("memory_on_user_recall_not_observed")
    if recall_setting == "enabled" and not evaluation["target_hit"]:
        findings.append("memory_on_target_not_recalled")
    runtime_state = TurnTerminalState(str(evaluation["runtime_state"]))
    if (
        runtime_state
        in {
            TurnTerminalState.COMPLETED,
            TurnTerminalState.COMPLETED_WITH_TOOL_FAILURE,
        }
        and (_safe_failure_category(evaluation) != "task_budget_exhausted")
        and not evaluation["usage"]["usage_complete"]
    ):
        findings.append("trial_usage_incomplete")
    if seed["embedding_provider_config_digest"] != evaluation["embedding_provider_config_digest"]:
        findings.append("embedding_configuration_changed_between_processes")
    if context.experiment.identity.get("provider"):
        requested_model = str(evaluation["provider_identity"])
        actual_models = evaluation["actual_models"]
        if not actual_models:
            findings.append("provider_actual_model_missing")
        elif not all(
            _provider_model_equivalent(
                requested_model,
                str(actual_model),
            )
            for actual_model in actual_models
        ):
            findings.append("provider_actual_model_mismatch")
    return findings


def _infrastructure_failure(
    context: TrialContext,
    isolation: TrialIsolation,
    finding: str,
) -> TrialExecution:
    relative_root = isolation.root.relative_to(
        context.experiment.output_root,
    )
    return TrialExecution(
        status=TrialStatus.INFRASTRUCTURE_FAILURE,
        runtime_state=None,
        delivery_state=None,
        verification=VerifierResult(
            state=VerificationState.NOT_RUN,
            findings=(finding,),
        ),
        observed_variant_settings=dict(context.variant.settings),
        findings=(finding,),
        artifact_refs=(relative_root.as_posix(),),
    )


def _preparation_failure(
    context: TrialContext,
    finding: str,
) -> TrialExecution:
    return TrialExecution(
        status=TrialStatus.INFRASTRUCTURE_FAILURE,
        runtime_state=None,
        delivery_state=None,
        verification=VerifierResult(
            state=VerificationState.NOT_RUN,
            findings=(finding,),
        ),
        observed_variant_settings=dict(context.variant.settings),
        findings=(finding,),
    )


def _safe_failure_category(
    stage: dict[str, Any],
) -> str | None:
    category = stage.get("failure_category")
    if not isinstance(category, str):
        return None
    normalized = category.strip().lower()
    if not normalized or len(normalized) > 64:
        return None
    if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in normalized):
        return None
    return normalized


def _provider_model_equivalent(
    requested: str,
    actual: str,
) -> bool:
    return bool(
        requested
        and actual
        and (requested == actual or requested.endswith(f"/{actual}") or actual.endswith(f"/{requested}"))
    )


def _shared_budget_snapshot(
    provider_spec: dict[str, Any],
) -> Any | None:
    if provider_spec["mode"] != "real":
        return None
    return ProviderBudgetLedger(
        Path(str(provider_spec["ledger_path"])),
        ProviderBudgetConfig(
            **dict(provider_spec["ledger_config"]),
        ),
    ).snapshot()


def _embedding_budgeted_cny(
    *,
    provider_spec: Mapping[str, Any],
    input_tokens: int,
) -> float:
    if provider_spec["mode"] != "real":
        return 0.0
    config = ProviderBudgetConfig(
        **dict(provider_spec["ledger_config"]),
    )
    return config.cost_cny(input_tokens, 0)


def _global_authority_proof_valid(
    budget_snapshot: Any | None,
) -> bool:
    if budget_snapshot is None:
        return False
    return all(
        (
            math.isclose(
                budget_snapshot.hard_cap_cny,
                _GLOBAL_AUTHORIZED_HARD_CAP_CNY,
                rel_tol=0,
                abs_tol=1e-12,
            ),
            math.isclose(
                budget_snapshot.external_service_reserve_cny,
                _MAIN_EXTERNAL_SERVICE_RESERVE_CNY,
                rel_tol=0,
                abs_tol=1e-12,
            ),
            math.isclose(
                budget_snapshot.hard_cap_cny - budget_snapshot.external_service_reserve_cny,
                _MAIN_PROVIDER_SPEND_CEILING_CNY,
                rel_tol=0,
                abs_tol=1e-12,
            ),
            math.isclose(
                _SEMANTIC_ADDENDUM_HARD_CAP_CNY,
                budget_snapshot.external_service_reserve_cny,
                rel_tol=0,
                abs_tol=1e-12,
            ),
            budget_snapshot.provider_charged_cny <= _MAIN_PROVIDER_SPEND_CEILING_CNY,
            budget_snapshot.total_committed_cny <= _GLOBAL_AUTHORIZED_HARD_CAP_CNY,
            budget_snapshot.accounting_complete,
            budget_snapshot.open_reservations == 0,
        )
    )


class _HashBagEmbeddingProvider:
    dim = 1024

    async def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        terms = re.findall(r"[a-z0-9]+", text.lower())
        for term in terms:
            digest = hashlib.sha256(term.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            vector[index] += 1.0
        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude:
            vector = [value / magnitude for value in vector]
        return vector

    async def embed_batch(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        return [await self.embed(text) for text in texts]


def _embedding_provider(
    spec: dict[str, Any],
) -> CountingEmbeddingProvider:
    mode = str(spec["mode"])
    if mode == "fake":
        delegate: Any = _HashBagEmbeddingProvider()
        real_provider = False
    elif mode == "real":
        from everos.component.embedding import OpenAIEmbeddingProvider
        from everos.config import load_settings

        settings = load_settings().embedding
        if not settings.model or settings.api_key is None or not settings.api_key.get_secret_value():
            raise RuntimeError("real embedding provider is not configured")
        transport = _embedding_transport_identity(settings)
        if canonical_digest(transport) != str(spec["provider_config_digest"]):
            raise RuntimeError("embedding transport differs from the frozen Trial plan")
        delegate = OpenAIEmbeddingProvider(
            model=str(settings.model),
            api_key=settings.api_key.get_secret_value(),
            base_url=str(settings.base_url),
            dim=1024,
            timeout=float(settings.timeout_seconds),
            max_retries=0,
            batch_size=int(settings.batch_size),
            max_concurrent=int(settings.max_concurrent),
        )
        real_provider = True
    else:
        raise ValueError(f"unknown embedding mode: {mode}")
    ledger = None
    budget_trial_id = None
    if mode == "real":
        ledger = ProviderBudgetLedger(
            Path(str(spec["ledger_path"])),
            ProviderBudgetConfig(
                **dict(spec["ledger_config"]),
            ),
        )
        budget_trial_id = str(spec["budget_trial_id"])
    return CountingEmbeddingProvider(
        delegate,
        provider_identity=str(spec["provider_identity"]),
        provider_config_digest=str(spec["provider_config_digest"]),
        real_provider=real_provider,
        maximum_characters=int(spec["maximum_characters"]),
        maximum_calls=int(spec["maximum_calls"]),
        characters_per_token=float(spec["characters_per_token"]),
        budget_ledger=ledger,
        budget_trial_id=budget_trial_id,
        maximum_input_tokens_per_call=(
            int(spec["maximum_input_tokens_per_call"]) if "maximum_input_tokens_per_call" in spec else None
        ),
    )


async def _seed_stage(
    *,
    task: SemanticMemoryEffectTask,
    embedding_spec: dict[str, Any],
) -> dict[str, Any]:
    from everos.component.tokenizer import build_tokenizer
    from everos.config import load_settings
    from everos.core.persistence import MemoryRoot
    from everos.infra.persistence.lancedb import (
        dispose_connection,
        ensure_business_indexes,
        get_connection,
        verify_business_schemas,
    )
    from everos.infra.persistence.markdown import EpisodeWriter
    from everos.infra.persistence.sqlite import dispose_engine, get_engine
    from everos.memory.cascade import CascadeOrchestrator

    root_path = Path(os.environ["EVEROS_ROOT"])
    _prepare_everos_root(root_path)
    load_settings.cache_clear()
    root = MemoryRoot(root_path)
    root.ensure()
    engine = get_engine()
    embedder = _embedding_provider(embedding_spec)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(SQLModel.metadata.create_all)
        await get_connection()
        await verify_business_schemas()
        await ensure_business_indexes()
        now = dt.datetime.now(dt.UTC).isoformat()
        writer = EpisodeWriter(root)
        texts = (task.memory_text, *task.distractor_memories)
        entry_ids = await writer.append_entries(
            task.workspace_id,
            [
                (
                    {
                        "owner_id": task.workspace_id,
                        "session_id": task.prior_session_id,
                        "timestamp": now,
                        "parent_type": "memcell",
                        "parent_id": (f"{task.task_id}-memory-{index}"),
                        "sender_ids": [task.workspace_id],
                    },
                    {"Content": text},
                )
                for index, text in enumerate(texts)
            ],
        )
        orchestrator = CascadeOrchestrator(
            memory_root=root,
            embedder=embedder,
            tokenizer=build_tokenizer(),
        )
        indexed_rows = await orchestrator.sync_once()
        target_native_id = f"{task.workspace_id}_{entry_ids[0]}"
        return {
            "pid": os.getpid(),
            "prior_session_id": task.prior_session_id,
            "target_native_id": target_native_id,
            "indexed_rows": int(indexed_rows),
            "storage_path": "EpisodeWriter->CascadeOrchestrator",
            "embedding_calls": embedder.calls,
            "embedded_characters": embedder.embedded_characters,
            "embedding_budgeted_input_tokens": (embedder.budgeted_input_tokens),
            "embedding_provider_identity": (embedder.provider_identity),
            "embedding_provider_config_digest": (embedder.provider_config_digest),
            "real_embedding": embedder.real_provider,
        }
    finally:
        await dispose_connection()
        await dispose_engine()
        load_settings.cache_clear()


class _RecordingProductionEverosBackend:
    def __init__(
        self,
        *,
        root: Path,
        task: SemanticMemoryEffectTask,
        user_recall_enabled: bool,
        drive_production_lifespan: bool,
    ) -> None:
        from codeflow.plugin.memory.everos.backend import EverosBackend

        self._delegate = EverosBackend(
            PluginContext(
                config={
                    "mode": "embedded",
                    "user_id": task.workspace_id,
                    "agent_id": task.workspace_id,
                    "flush_every_turns": 0,
                },
                services=ServiceLocator(workspace=root),
            )
        )
        self._user_recall_enabled = user_recall_enabled
        self._drive_production_lifespan = drive_production_lifespan
        self.user_recall_calls = 0
        self.suppressed_user_recall_calls = 0
        self.memory_hit_ids: list[str] = []
        self.store_calls = 0

    async def start(self) -> None:
        if self._drive_production_lifespan:
            await self._delegate.start()
            return
        from codeflow.plugin.memory.everos.backend import (
            _try_make_real_adapter,
        )

        self._delegate._adapter = await asyncio.to_thread(
            _try_make_real_adapter,
        )

    async def stop(self) -> None:
        await self._delegate.stop()

    async def feedback(self, signals: dict[str, Any]) -> None:
        await self._delegate.feedback(signals)

    async def recall(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        top_k: int,
    ) -> list[Memory]:
        if user_id is not None:
            if not self._user_recall_enabled:
                self.suppressed_user_recall_calls += 1
                return []
            self.user_recall_calls += 1
        memories = await self._delegate.recall(
            query,
            user_id=user_id,
            agent_id=agent_id,
            top_k=top_k,
        )
        if user_id is not None:
            self.memory_hit_ids.extend(
                str(memory.metadata.get("id")) for memory in memories if memory.metadata.get("id") is not None
            )
        return memories

    async def store(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
    ) -> None:
        del session_id, messages
        self.store_calls += 1

    @property
    def adapter_class(self) -> str:
        adapter = self._delegate._adapter
        return type(adapter).__name__ if adapter is not None else "None"


class _ScriptedSemanticEffectProvider(LLMProvider):
    def __init__(self, task: SemanticMemoryEffectTask) -> None:
        super().__init__(api_key="codeflowbench")
        self._task = task
        self._calls = 0
        self.memory_observed = False

    def get_default_model(self) -> str:
        return "scripted/codeflowbench-semantic-memory-effect"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        del max_tokens, temperature, reasoning_effort, tool_choice
        self._calls += 1
        prompt = "\n".join(str(message.get("content", "")) for message in messages)
        self.memory_observed = (
            self._task.expected_approval_code in prompt
            and self._task.expected_region in prompt
            and self._task.expected_retention in prompt
        )
        usage = {
            "prompt_tokens": estimate_prompt_tokens(messages, tools),
            "completion_tokens": 24,
        }
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        actual_model = model or self.get_default_model()
        if self._calls == 1:
            artifact = {
                "task_id": self._task.task_id,
                "region": (self._task.expected_region if self.memory_observed else None),
                "retention": (self._task.expected_retention if self.memory_observed else None),
                "approval_code": (self._task.expected_approval_code if self.memory_observed else None),
            }
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id=f"{self._task.task_id}-write-result",
                        name="write_file",
                        arguments={
                            "path": "result.json",
                            "content": json.dumps(
                                artifact,
                                sort_keys=True,
                            ),
                        },
                    )
                ],
                finish_reason="tool_calls",
                usage=usage,
                model=actual_model,
            )
        return LLMResponse(
            content="Evaluation artifact written.",
            finish_reason="stop",
            usage=usage,
            model=actual_model,
        )


def _semantic_memory_context_engine_factory():
    def build(**kwargs: Any) -> ContextAssembler:
        backend = kwargs["backend"]
        builder = kwargs["builder"]
        memory_config = kwargs.get("memory_config") or MemoryConfig()
        builders = [
            IdentitySegmentBuilder(kwargs["workspace"]),
            BootstrapSegmentBuilder(kwargs["workspace"]),
            MemorySegmentBuilder(
                builder.memory,
                backend,
                user_id=memory_config.user_id,
                memory_top_k=memory_config.memory_top_k,
                enabled=True,
            ),
            CuratorSegmentBuilder(
                workspace=kwargs["workspace"],
                config=kwargs["config"],
                provider=kwargs["provider"],
                model=kwargs["model"],
                context_window_tokens=kwargs["context_window_tokens"],
                get_tool_definitions=kwargs["get_tool_definitions"],
                now_fn=kwargs.get("now_fn"),
                memory_enabled=True,
            ),
        ]
        return ContextAssembler(
            builders,
            kwargs["get_tool_definitions"],
            now_fn=kwargs.get("now_fn"),
        )

    return build


async def _evaluation_stage(
    *,
    task: SemanticMemoryEffectTask,
    variant_settings: dict[str, Any],
    workspace: Path,
    provider_spec: dict[str, Any],
    embedding_spec: dict[str, Any],
    seed_result: dict[str, Any],
) -> dict[str, Any]:
    import importlib
    from unittest.mock import patch

    from everos.config import load_settings

    _prepare_everos_root(Path(os.environ["EVEROS_ROOT"]))
    load_settings.cache_clear()
    embedder = _embedding_provider(embedding_spec)
    search_module = importlib.import_module("everos.service.search")
    search_module._manager = None
    search_module._embedding = embedder
    search_module._embedding_resolved = True
    search_module._reranker = None
    search_module._rerank_resolved = True
    search_module._llm_client = None
    search_module._llm_resolved = True

    user_recall_enabled = variant_settings["user_memory_recall"] == "enabled"
    backend = _RecordingProductionEverosBackend(
        root=workspace,
        task=task,
        user_recall_enabled=user_recall_enabled,
        drive_production_lifespan=(embedding_spec["mode"] == "real"),
    )
    (
        config,
        codeflow_config,
        delegate,
        budget_ledger,
        provider_identity,
    ) = _runtime_dependencies(
        provider_spec=provider_spec,
        workspace=workspace,
        task=task,
    )
    recorder = UsageRecorder()
    provider = RecordingProvider(delegate, recorder=recorder)
    outlet = RecordingOutlet("codeflowbench-semantic-memory-effect")
    conversation = f"{task.task_id}:evaluation"
    request = TurnRequest(
        origin=Origin.USER,
        source=Source(
            channel="codeflowbench-semantic-memory-effect",
            chat_id=conversation,
            sender_id=task.workspace_id,
            chat_type=ChatType.DM,
        ),
        text=task.evaluation_request,
        message_id=f"{conversation}-message",
        conversation=conversation,
    )
    budget_before = budget_ledger.snapshot() if budget_ledger else None
    if provider_spec["mode"] == "real":
        budget_scope = provider_call_budget_scope(
            trial_id=str(provider_spec["trial_id"]),
            max_logical_calls=int(
                provider_spec["max_logical_calls"],
            ),
            max_attempts_per_call=int(
                provider_spec["max_attempts_per_call"],
            ),
            max_input_tokens_per_call=int(
                provider_spec["max_input_tokens_per_call"],
            ),
            max_output_tokens_per_call=int(
                provider_spec["max_output_tokens_per_call"],
            ),
        )
    else:
        budget_scope = contextlib.nullcontext()
    with (
        patch(
            "codeflow.cli._plugin_stack.maybe_build_memory_backend",
            return_value=backend,
        ),
        patch(
            "codeflow.cli._plugin_stack.build_plugin_tools",
            return_value=[],
        ),
    ):
        with budget_scope:
            host = await RuntimeTrialHost.build(
                config=config,
                codeflow_config=codeflow_config,
                provider=provider,
                cron_service=None,
                outlet=outlet,
                context_engine_factory=(_semantic_memory_context_engine_factory()),
            )
            try:
                with usage_scope(call_role="main_agent"):
                    observation = await host.run(request)
                session = host.assembly.session_manager.peek(
                    conversation,
                )
            finally:
                await host.close()
    aggregate = recorder.aggregate()
    budget_after = budget_ledger.snapshot() if budget_ledger else None
    failure_category = observation.failure_category
    if recorder.has_error_category("task_budget_exhausted"):
        failure_category = "task_budget_exhausted"
    scripted = (
        delegate
        if isinstance(
            delegate,
            _ScriptedSemanticEffectProvider,
        )
        else None
    )
    target_native_id = str(seed_result["target_native_id"])
    return {
        "pid": os.getpid(),
        "conversation": conversation,
        "session_message_count": (len(session.messages) if session is not None else 0),
        "runtime_state": observation.runtime_state.value,
        "delivery_state": observation.delivery_state.value,
        "failure_category": failure_category,
        "memory_hits": (observation.outcome.memory_hits if observation.outcome is not None else 0),
        "memory_hit_ids": list(backend.memory_hit_ids),
        "target_hit": target_native_id in backend.memory_hit_ids,
        "user_recall_calls": backend.user_recall_calls,
        "suppressed_user_recall_calls": (backend.suppressed_user_recall_calls),
        "store_calls": backend.store_calls,
        "backend_class": "EverosBackend",
        "backend_adapter_class": backend.adapter_class,
        "provider_memory_observed": (scripted.memory_observed if scripted is not None else None),
        "provider_identity": provider_identity,
        "actual_models": sorted({record.model for record in recorder.records() if record.model is not None}),
        "real_agent_provider": provider_spec["mode"] == "real",
        "paid_campaign_eligible": bool(provider_spec["paid_campaign_eligible"]),
        "agent_cost_complete": (budget_after.accounting_complete if budget_after is not None else True),
        "provider_charged_cny": (
            budget_after.provider_charged_cny - budget_before.provider_charged_cny
            if budget_after is not None and budget_before is not None
            else 0.0
        ),
        "embedding_calls": embedder.calls,
        "embedded_characters": embedder.embedded_characters,
        "embedding_budgeted_input_tokens": (embedder.budgeted_input_tokens),
        "embedding_provider_identity": embedder.provider_identity,
        "embedding_provider_config_digest": (embedder.provider_config_digest),
        "real_embedding": embedder.real_provider,
        "usage": {
            "calls": aggregate.calls,
            "input_tokens": aggregate.input_tokens,
            "output_tokens": aggregate.output_tokens,
            "total_tokens": aggregate.total_tokens,
            "usage_complete": aggregate.usage_complete,
        },
    }


def _runtime_dependencies(
    *,
    provider_spec: dict[str, Any],
    workspace: Path,
    task: SemanticMemoryEffectTask,
) -> tuple[
    Config,
    CodeFlowConfig,
    LLMProvider,
    ProviderBudgetLedger | None,
    str,
]:
    if provider_spec["mode"] == "scripted":
        delegate = _ScriptedSemanticEffectProvider(task)
        config = _freeze_runtime_config(
            Config(),
            workspace=workspace,
            model=delegate.get_default_model(),
        )
        return (
            config,
            _freeze_codeflow_config(
                CodeFlowConfig(base=config),
                config,
                task,
            ),
            delegate,
            None,
            delegate.get_default_model(),
        )
    if provider_spec["mode"] != "real":
        raise ValueError(
            f"unknown provider mode: {provider_spec['mode']}",
        )
    from codeflow.cli._helpers import make_provider

    private_config_path = Path(str(provider_spec["private_config_path"]))
    payload = json.loads(
        private_config_path.read_text(encoding="utf-8"),
    )
    model = str(provider_spec["provider_identity"])
    config = _freeze_runtime_config(
        Config.model_validate(payload["config"]),
        workspace=workspace,
        model=model,
    )
    codeflow_config = _freeze_codeflow_config(
        CodeFlowConfig.model_validate(payload["codeflow_config"]),
        config,
        task,
    )
    ledger = ProviderBudgetLedger(
        Path(str(provider_spec["ledger_path"])),
        ProviderBudgetConfig(
            **dict(provider_spec["ledger_config"]),
        ),
    )
    delegate = BudgetGuardedProvider(
        make_provider(config),
        ledger=ledger,
    )
    return config, codeflow_config, delegate, ledger, model


def _freeze_runtime_config(
    source: Config,
    *,
    workspace: Path,
    model: str,
) -> Config:
    config = Config.model_validate(
        source.model_dump(mode="json"),
    )
    config.agents.defaults.workspace = str(workspace)
    config.agents.defaults.model = model
    config.agents.defaults.max_tokens = 1_500
    config.agents.defaults.max_tool_iterations = 3
    config.agents.defaults.enable_personalization = False
    config.routing.enabled = False
    config.tools.restrict_to_workspace = True
    config.tools.disabled_tools = list(_DISABLED_TOOLS)
    config.tools.mcp_servers = {}
    config.tools.tool_search.enabled = False
    return config


def _freeze_codeflow_config(
    codeflow_config: CodeFlowConfig,
    config: Config,
    task: SemanticMemoryEffectTask,
) -> CodeFlowConfig:
    codeflow_config.base = config
    codeflow_config.memory.backend = None
    codeflow_config.memory.user_id = task.workspace_id
    codeflow_config.memory.agent_id = task.workspace_id
    codeflow_config.memory.memory_top_k = 5
    codeflow_config.skill_forge.rewrite_enabled = False
    codeflow_config.skill_forge.llm_gate_enabled = False
    codeflow_config.token_wise.smart_routing.enabled = False
    codeflow_config.runtime.checkpoint.policy = "never"
    return codeflow_config


def _prepare_everos_root(root: Path) -> None:
    import everos

    root.mkdir(parents=True, exist_ok=True)
    target = root / "ome.toml"
    if target.exists():
        return
    source = Path(everos.__file__).resolve().parent / "config" / "default_ome.toml"
    shutil.copy2(source, target)


def _configured_embedding_environment() -> tuple[
    dict[str, str],
    str,
    dict[str, Any],
]:
    from everos.config import load_settings
    from codeflow.config.update_everos import get_everos_config_path

    previous_root = os.environ.get("EVEROS_ROOT")
    if previous_root is None:
        os.environ["EVEROS_ROOT"] = str(
            get_everos_config_path().parent,
        )
    try:
        load_settings.cache_clear()
        settings = load_settings()
    finally:
        if previous_root is None:
            os.environ.pop("EVEROS_ROOT", None)
        else:
            os.environ["EVEROS_ROOT"] = previous_root
        load_settings.cache_clear()
    embedding = settings.embedding
    if not embedding.model or embedding.api_key is None or not embedding.api_key.get_secret_value():
        raise ValueError("EverOS embedding provider is not configured")
    environment = {
        "EVEROS_EMBEDDING__MODEL": str(embedding.model),
        "EVEROS_EMBEDDING__API_KEY": (embedding.api_key.get_secret_value()),
        "EVEROS_EMBEDDING__TIMEOUT_SECONDS": str(
            embedding.timeout_seconds,
        ),
        "EVEROS_EMBEDDING__MAX_RETRIES": "0",
        "EVEROS_EMBEDDING__BATCH_SIZE": str(
            embedding.batch_size,
        ),
        "EVEROS_EMBEDDING__MAX_CONCURRENT": str(
            embedding.max_concurrent,
        ),
    }
    if embedding.base_url:
        environment["EVEROS_EMBEDDING__BASE_URL"] = str(
            embedding.base_url,
        )
    llm = settings.llm
    if llm.model and llm.api_key is not None and llm.api_key.get_secret_value():
        environment["EVEROS_LLM__MODEL"] = str(llm.model)
        environment["EVEROS_LLM__API_KEY"] = llm.api_key.get_secret_value()
        if llm.base_url:
            environment["EVEROS_LLM__BASE_URL"] = str(
                llm.base_url,
            )
    return (
        environment,
        str(embedding.model),
        _embedding_transport_identity(embedding),
    )


def _embedding_transport_identity(settings: Any) -> dict[str, Any]:
    return {
        "provider_class": ("everos.component.embedding.OpenAIEmbeddingProvider"),
        "model": str(settings.model or ""),
        "base_url": str(settings.base_url or ""),
        "dimension": 1024,
        "timeout_seconds": float(settings.timeout_seconds),
        "max_retries": 0,
        "batch_size": int(settings.batch_size),
        "max_concurrent": int(settings.max_concurrent),
    }


def _task_from_payload(
    payload: dict[str, Any],
) -> SemanticMemoryEffectTask:
    return SemanticMemoryEffectTask(
        task_id=str(payload["task_id"]),
        workspace_id=str(payload["workspace_id"]),
        prior_session_id=str(payload["prior_session_id"]),
        customer=str(payload["customer"]),
        service=str(payload["service"]),
        memory_text=str(payload["memory_text"]),
        distractor_memories=tuple(str(item) for item in payload["distractor_memories"]),
        expected_region=str(payload["expected_region"]),
        expected_retention=str(payload["expected_retention"]),
        expected_approval_code=str(
            payload["expected_approval_code"],
        ),
        evaluation_request=str(payload["evaluation_request"]),
    )


def _worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker-stage",
        choices=("seed", "evaluation"),
        required=True,
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser


async def _run_worker(
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = json.loads(
        args.spec.read_text(encoding="utf-8"),
    )
    task = _task_from_payload(dict(payload["task"]))
    embedding_spec = dict(payload["embedding"])
    budget_prefix = embedding_spec.get("budget_trial_id_prefix")
    if budget_prefix is not None:
        embedding_spec["budget_trial_id"] = f"{budget_prefix}:{args.worker_stage}"
    if args.worker_stage == "seed":
        return await _seed_stage(
            task=task,
            embedding_spec=embedding_spec,
        )
    seed_result = json.loads(
        Path(payload["seed_result_path"]).read_text(
            encoding="utf-8",
        ),
    )
    return await _evaluation_stage(
        task=task,
        variant_settings=dict(payload["variant_settings"]),
        workspace=Path(payload["workspace"]),
        provider_spec=dict(payload["provider"]),
        embedding_spec=embedding_spec,
        seed_result=seed_result,
    )


def main() -> int:
    args = _worker_parser().parse_args()
    try:
        result = asyncio.run(_run_worker(args))
    except BaseException as exc:
        result = {
            "status": "infrastructure_failure",
            "error_code": "worker_exception",
            "error_type": type(exc).__name__,
        }
        exit_code = 1
    else:
        result = {"status": "completed", **result}
        exit_code = 0
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(
            to_primitive(result),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
