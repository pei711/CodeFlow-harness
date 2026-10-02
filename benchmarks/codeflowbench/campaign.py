from __future__ import annotations

import asyncio
import importlib.metadata
import json
import shutil
import subprocess
import tempfile
from collections.abc import Awaitable, Callable, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterator, Protocol

import yaml

from codeflow.product import get_product_home
from codeflow.providers.base import LLMProvider, ToolCallRequest
from codeflow.utils.portable_lock import LockTimeoutError, file_lock

from .artifacts import ArtifactError, ArtifactStore
from .budget import (
    BudgetGuardedProvider,
    ProviderBudgetConfig,
    ProviderBudgetLedger,
    ProviderBudgetSnapshot,
    provider_call_budget_scope,
)
from .canonical import canonical_digest, to_primitive
from .environment import (
    EnvironmentIdentityError,
    capture_environment_identity,
    validate_environment_identity,
)
from .harness import rebuild_report as rebuild_experiment_report
from .harness import run as run_experiment
from .plan import ExperimentPlan, compile_plan
from .registry import PackRegistry
from .schema import (
    ClaimRule,
    ExecutionPolicy,
    ExperimentRef,
    ExperimentSpec,
    ProviderTrialBudget,
)

CAMPAIGN_SCHEMA = "codeflow.codeflowbench.campaign.v1"
DEFAULT_SUITE_PATH = Path(__file__).resolve().parent / "suites" / "agent_application_ship_1.yaml"
DEFAULT_OUTPUT_ROOT = Path(".codeflow/evidence/codeflowbench")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_REQUIRED_USAGE_FIELDS = frozenset(
    {"prompt_tokens", "completion_tokens", "total_tokens"},
)
_PROVIDER_PROBE_MAX_ATTEMPTS = 2
_PROVIDER_BUDGET_APPROVAL_SCHEMA = "codeflow.codeflowbench.provider-budget-approval.v3"
_PROVIDER_PREFLIGHT_CACHE_SCHEMA = "codeflow.codeflowbench.provider-preflight-cache.v1"
_RUNTIME_EVIDENCE_SCHEMA = "codeflow.codeflowbench.runtime-evidence.v2"


class CampaignError(RuntimeError):
    pass


class CampaignMode(StrEnum):
    SMOKE = "smoke"
    CALIBRATION = "calibration"
    FORMAL = "formal"
    SHIP = "ship"


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    model: str
    allow_fallback: bool


@dataclass(frozen=True)
class BudgetConfig:
    warning_cny: float
    hard_cap_cny: float
    max_provider_calls_per_trial: int
    max_input_tokens_per_call: int
    max_output_tokens_per_call: int
    max_additional_provider_attempts: int
    input_cache_miss_usd_per_million: float
    output_usd_per_million: float
    conservative_usd_to_cny_multiplier: float
    external_service_reserve_cny: float
    pricing_source: str
    provider_trial_budgets: tuple[ProviderTrialBudget, ...]

    def __post_init__(self) -> None:
        if self.warning_cny <= 0:
            raise ValueError("warning_cny must be positive")
        for name in (
            "max_provider_calls_per_trial",
            "max_input_tokens_per_call",
            "max_output_tokens_per_call",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_additional_provider_attempts < 0:
            raise ValueError(
                "max_additional_provider_attempts must not be negative",
            )
        for name in (
            "input_cache_miss_usd_per_million",
            "output_usd_per_million",
            "conservative_usd_to_cny_multiplier",
            "external_service_reserve_cny",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")
        if not self.provider_trial_budgets:
            raise ValueError("provider_trial_budgets must not be empty")
        pack_ids = tuple(budget.pack_id for budget in self.provider_trial_budgets)
        if len(set(pack_ids)) != len(pack_ids):
            raise ValueError(
                "provider_trial_budgets must use unique pack IDs",
            )
        if (
            max(budget.max_provider_calls_per_trial for budget in self.provider_trial_budgets)
            != self.max_provider_calls_per_trial
        ):
            raise ValueError(
                "max_provider_calls_per_trial must equal the largest pack ceiling",
            )
        if (
            max(budget.max_input_tokens_per_call for budget in self.provider_trial_budgets)
            != self.max_input_tokens_per_call
        ):
            raise ValueError(
                "max_input_tokens_per_call must equal the largest pack ceiling",
            )
        if (
            max(budget.max_output_tokens_per_call for budget in self.provider_trial_budgets)
            != self.max_output_tokens_per_call
        ):
            raise ValueError(
                "max_output_tokens_per_call must equal the largest pack ceiling",
            )

    def provider_trial_budget_for(
        self,
        pack_id: str,
    ) -> ProviderTrialBudget:
        budget = next(
            (budget for budget in self.provider_trial_budgets if budget.pack_id == pack_id),
            None,
        )
        if budget is None:
            raise ValueError(
                f"missing Provider Trial budget for pack: {pack_id}",
            )
        return budget


@dataclass(frozen=True)
class TrackConfig:
    suite: str
    repetitions: int
    pack_ids: tuple[str, ...]
    expected_trials: int
    expected_retrieval_cases: int
    max_comparison_block_retries_total: int
    expected_trials_by_pack: tuple[tuple[str, int], ...]
    expected_comparison_blocks_by_pack: tuple[tuple[str, int], ...]
    expected_retrieval_cases_by_pack: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if self.repetitions < 1:
            raise ValueError("repetitions must be positive")
        if not self.pack_ids:
            raise ValueError("pack_ids must not be empty")
        if self.expected_trials < 0 or self.expected_retrieval_cases < 0:
            raise ValueError("expected denominators must not be negative")
        if self.max_comparison_block_retries_total < 0:
            raise ValueError(
                "max_comparison_block_retries_total must not be negative",
            )
        for field, expected_total in (
            ("expected_trials_by_pack", self.expected_trials),
            (
                "expected_retrieval_cases_by_pack",
                self.expected_retrieval_cases,
            ),
        ):
            counts = getattr(self, field)
            pack_ids = tuple(pack_id for pack_id, _ in counts)
            if pack_ids != self.pack_ids:
                raise ValueError(
                    f"{field} must match pack_ids in order",
                )
            if any(count < 0 for _, count in counts):
                raise ValueError(f"{field} must not contain negative counts")
            if sum(count for _, count in counts) != expected_total:
                raise ValueError(
                    f"{field} must sum to its track denominator",
                )
        block_pack_ids = tuple(pack_id for pack_id, _count in self.expected_comparison_blocks_by_pack)
        if block_pack_ids != self.pack_ids:
            raise ValueError(
                "expected_comparison_blocks_by_pack must match pack_ids in order",
            )
        if any(count < 0 for _pack_id, count in self.expected_comparison_blocks_by_pack):
            raise ValueError(
                "expected_comparison_blocks_by_pack must not contain negative counts",
            )
        if self.max_comparison_block_retries_total > sum(
            count for _pack_id, count in self.expected_comparison_blocks_by_pack
        ):
            raise ValueError(
                "comparison block retry cap exceeds the planned block count",
            )


@dataclass(frozen=True)
class CampaignSuite:
    suite: str
    provider: ProviderConfig
    execution: ExecutionPolicy
    budget: BudgetConfig
    calibration: TrackConfig
    formal: TrackConfig
    claim_rules: tuple[ClaimRule, ...]
    schema: str = CAMPAIGN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CAMPAIGN_SCHEMA:
            raise ValueError(f"unsupported campaign schema: {self.schema}")
        pack_ids = {
            *self.calibration.pack_ids,
            *self.formal.pack_ids,
        }
        budget_pack_ids = {budget.pack_id for budget in self.budget.provider_trial_budgets}
        if budget_pack_ids != pack_ids:
            raise ValueError(
                "provider_trial_budgets must cover every campaign pack exactly",
            )
        if self.execution.provider_call_max_attempts == 1 and self.budget.max_additional_provider_attempts:
            raise ValueError(
                "additional Provider attempts require per-call retries",
            )
        if self.execution.max_comparison_block_attempts == 1 and (
            self.calibration.max_comparison_block_retries_total or self.formal.max_comparison_block_retries_total
        ):
            raise ValueError(
                "comparison block retries require a second block attempt",
            )

    def canonical_payload(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True)
class CostEstimate:
    trial_count: int
    logical_provider_calls: int
    additional_provider_attempts: int
    provider_request_attempts: int
    maximum_input_tokens: int
    maximum_output_tokens: int
    estimated_usd: float
    estimated_cny: float
    pricing_source: str


@dataclass(frozen=True)
class ResolvedProvider:
    provider_name: str
    model: str
    provider: object
    runtime_config_digest: str
    config: object | None = None
    codeflow_config: object | None = None
    configured_model: str | None = None
    budget_ledger: ProviderBudgetLedger | None = None


@dataclass(frozen=True)
class ProviderProbeRequest:
    provider_name: str
    model: str
    codeflow_commit: str
    worktree_clean: bool | None
    suite_digest: str
    claim_rules_digest: str
    estimated_cost: CostEstimate
    max_provider_calls_per_trial: int
    max_input_tokens_per_call: int
    max_output_tokens_per_call: int


@dataclass(frozen=True)
class ProviderProbeResult:
    provider_name: str
    requested_model: str
    resolved_model: str
    tool_calling_supported: bool
    usage_fields: tuple[str, ...]
    seed_supported: bool
    tokenizer_identity: str
    tokenizer_version: str
    tokenizer_digest: str
    pricing_source: str
    fallback_used: bool


@dataclass(frozen=True)
class ProviderPreflightRecord:
    provider_name: str
    requested_model: str
    resolved_model: str
    codeflow_commit: str
    worktree_clean: bool | None
    suite_digest: str
    claim_rules_digest: str
    runtime_config_digest: str
    tokenizer_identity: str
    tokenizer_version: str
    tokenizer_digest: str
    usage_fields: tuple[str, ...]
    seed_supported: bool
    pricing_source: str
    estimated_worst_case_cny: float
    budget_warning_cny: float
    budget_hard_cap_cny: float


@dataclass(frozen=True)
class DeterministicGateResult:
    passed: bool
    details: dict[str, Any]
    evidence_path: str | None = None
    evidence_digest: str | None = None


class CampaignReport(Protocol):
    planned_trials: int
    terminal_trials: int
    planned_retrieval_cases: int
    terminal_retrieval_cases: int
    ship_complete: bool
    measurement_valid: bool


@dataclass(frozen=True)
class CampaignServices:
    run_deterministic: Callable[[Path], Awaitable[DeterministicGateResult]]
    worktree_is_clean: Callable[[], bool]
    resolve_codeflow_commit: Callable[[], str]
    resolve_provider: Callable[[], ResolvedProvider]
    build_registry: Callable[
        [CampaignMode, ResolvedProvider],
        PackRegistry,
    ]
    probe_provider: Callable[
        [ProviderProbeRequest, ResolvedProvider],
        Awaitable[ProviderProbeResult],
    ]
    execute_experiment: Callable[
        [ExperimentSpec, PackRegistry],
        Awaitable[ExperimentRef],
    ]
    rebuild_report: Callable[[ExperimentRef], CampaignReport]
    prepare_budget_guard: (
        Callable[
            [
                ResolvedProvider,
                CampaignSuite,
                CampaignMode,
                CostEstimate,
                Path,
                str,
                str,
            ],
            ResolvedProvider,
        ]
        | None
    ) = None


@dataclass(frozen=True)
class CampaignOutcome:
    mode: CampaignMode
    deterministic: DeterministicGateResult
    preflight: ProviderPreflightRecord | None
    experiments: tuple[ExperimentRef, ...] = ()
    reports: tuple[CampaignReport, ...] = ()
    warnings: tuple[str, ...] = ()
    budget_snapshot: ProviderBudgetSnapshot | None = None
    campaign_artifact_path: str | None = None


def load_campaign_suite(path: Path = DEFAULT_SUITE_PATH) -> CampaignSuite:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CampaignError(f"cannot load campaign suite: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise CampaignError("campaign suite must be a mapping")
    try:
        _require_keys(
            raw,
            {
                "schema",
                "suite",
                "provider",
                "execution",
                "budget",
                "calibration",
                "formal",
                "claim_rules",
            },
            "campaign suite",
        )
        provider_raw = _mapping(raw["provider"], "provider")
        execution_raw = _mapping(raw["execution"], "execution")
        budget_raw = _mapping(raw["budget"], "budget")
        calibration_raw = _mapping(raw["calibration"], "calibration")
        formal_raw = _mapping(raw["formal"], "formal")
        return CampaignSuite(
            schema=str(raw["schema"]),
            suite=str(raw["suite"]),
            provider=ProviderConfig(
                name=str(provider_raw["name"]),
                model=str(provider_raw["model"]),
                allow_fallback=_bool(
                    provider_raw["allow_fallback"],
                    "provider.allow_fallback",
                ),
            ),
            execution=ExecutionPolicy(
                timeout_seconds=float(execution_raw["timeout_seconds"]),
                retry_policy=str(execution_raw["retry_policy"]),
                provider_call_max_attempts=int(
                    execution_raw["provider_call_max_attempts"],
                ),
                max_comparison_block_attempts=int(
                    execution_raw["max_comparison_block_attempts"],
                ),
                max_retrieval_query_block_attempts=int(
                    execution_raw["max_retrieval_query_block_attempts"],
                ),
            ),
            budget=BudgetConfig(
                warning_cny=float(budget_raw["warning_cny"]),
                hard_cap_cny=float(budget_raw["hard_cap_cny"]),
                max_provider_calls_per_trial=int(
                    budget_raw["max_provider_calls_per_trial"],
                ),
                max_input_tokens_per_call=int(
                    budget_raw["max_input_tokens_per_call"],
                ),
                max_output_tokens_per_call=int(
                    budget_raw["max_output_tokens_per_call"],
                ),
                max_additional_provider_attempts=int(
                    budget_raw["max_additional_provider_attempts"],
                ),
                input_cache_miss_usd_per_million=float(
                    budget_raw["input_cache_miss_usd_per_million"],
                ),
                output_usd_per_million=float(
                    budget_raw["output_usd_per_million"],
                ),
                conservative_usd_to_cny_multiplier=float(
                    budget_raw["conservative_usd_to_cny_multiplier"],
                ),
                external_service_reserve_cny=float(
                    budget_raw["external_service_reserve_cny"],
                ),
                pricing_source=str(budget_raw["pricing_source"]),
                provider_trial_budgets=_provider_trial_budgets(
                    budget_raw["provider_trial_budgets"],
                ),
            ),
            calibration=_track_config(calibration_raw),
            formal=_track_config(formal_raw),
            claim_rules=tuple(_claim_rule(item) for item in _sequence(raw["claim_rules"], "claim_rules")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CampaignError(f"invalid campaign suite: {exc}") from exc


def estimate_worst_case_cost(
    suite: CampaignSuite,
    *,
    modes: tuple[CampaignMode, ...],
) -> CostEstimate:
    normalized = tuple(CampaignMode(mode) for mode in modes)
    if not normalized:
        raise ValueError("at least one paid campaign mode is required")
    if CampaignMode.SMOKE in normalized or CampaignMode.SHIP in normalized:
        raise ValueError("cost estimate requires calibration/formal modes")
    tracks = tuple(suite.calibration if mode is CampaignMode.CALIBRATION else suite.formal for mode in normalized)
    trial_count = sum(track.expected_trials for track in tracks)
    logical_calls = 0
    input_tokens = 0
    output_tokens = 0
    retry_block_candidates: list[tuple[float, int, int, int]] = []
    provider_attempt_candidates: list[tuple[float, ProviderTrialBudget]] = []
    for track in tracks:
        blocks_by_pack = dict(
            track.expected_comparison_blocks_by_pack,
        )
        track_retry_candidates: list[tuple[float, int, int, int]] = []
        for pack_id, pack_trials in track.expected_trials_by_pack:
            pack_budget = suite.budget.provider_trial_budget_for(pack_id)
            pack_logical_calls = pack_trials * pack_budget.max_provider_calls_per_trial
            logical_calls += pack_logical_calls
            input_tokens += pack_logical_calls * pack_budget.max_input_tokens_per_call
            output_tokens += pack_logical_calls * pack_budget.max_output_tokens_per_call
            block_count = blocks_by_pack[pack_id]
            if block_count < 1 or pack_trials % block_count:
                raise ValueError(
                    f"{pack_id} Trial count is not divisible by its block count",
                )
            arms_per_block = pack_trials // block_count
            block_calls = arms_per_block * pack_budget.max_provider_calls_per_trial
            block_input = block_calls * pack_budget.max_input_tokens_per_call
            block_output = block_calls * pack_budget.max_output_tokens_per_call
            block_cost = _token_cost_usd(
                suite,
                input_tokens=block_input,
                output_tokens=block_output,
            )
            track_retry_candidates.extend(
                (block_cost, block_calls, block_input, block_output) for _ in range(block_count)
            )
            provider_attempt_candidates.append(
                (
                    _token_cost_usd(
                        suite,
                        input_tokens=(pack_budget.max_input_tokens_per_call),
                        output_tokens=(pack_budget.max_output_tokens_per_call),
                    ),
                    pack_budget,
                )
            )
        retry_block_candidates.extend(
            sorted(
                track_retry_candidates,
                key=lambda item: item[0],
                reverse=True,
            )[: track.max_comparison_block_retries_total]
        )
    for _cost, block_calls, block_input, block_output in retry_block_candidates:
        logical_calls += block_calls
        input_tokens += block_input
        output_tokens += block_output

    additional_attempts = suite.budget.max_additional_provider_attempts
    maximum_additional_attempts = logical_calls * (suite.execution.provider_call_max_attempts - 1)
    if additional_attempts > maximum_additional_attempts:
        raise ValueError(
            "additional Provider attempt cap exceeds per-call retry capacity",
        )
    if additional_attempts:
        _cost, most_expensive = max(
            provider_attempt_candidates,
            key=lambda item: item[0],
        )
        input_tokens += additional_attempts * most_expensive.max_input_tokens_per_call
        output_tokens += additional_attempts * most_expensive.max_output_tokens_per_call
    request_attempts = logical_calls + additional_attempts
    estimated_usd = _token_cost_usd(
        suite,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    estimated_cny = (
        estimated_usd * suite.budget.conservative_usd_to_cny_multiplier + suite.budget.external_service_reserve_cny
    )
    return CostEstimate(
        trial_count=trial_count,
        logical_provider_calls=logical_calls,
        additional_provider_attempts=additional_attempts,
        provider_request_attempts=request_attempts,
        maximum_input_tokens=input_tokens,
        maximum_output_tokens=output_tokens,
        estimated_usd=estimated_usd,
        estimated_cny=estimated_cny,
        pricing_source=suite.budget.pricing_source,
    )


def _token_cost_usd(
    suite: CampaignSuite,
    *,
    input_tokens: int,
    output_tokens: int,
) -> float:
    return (
        input_tokens / 1_000_000 * suite.budget.input_cache_miss_usd_per_million
        + output_tokens / 1_000_000 * suite.budget.output_usd_per_million
    )


async def run_campaign(
    mode: CampaignMode,
    *,
    output_root: Path,
    suite_path: Path = DEFAULT_SUITE_PATH,
    suite: CampaignSuite | None = None,
    services: CampaignServices | None = None,
) -> CampaignOutcome:
    selected_mode = CampaignMode(mode)
    if selected_mode is CampaignMode.FORMAL:
        raise CampaignError(
            "standalone formal campaign is disabled; run ship so calibration gates formal execution",
        )
    selected_suite = suite or load_campaign_suite(suite_path)
    if selected_mode is CampaignMode.SMOKE:
        return await _run_campaign(
            selected_mode,
            output_root=output_root,
            suite=selected_suite,
            services=services,
        )
    suite_digest = canonical_digest(
        selected_suite.canonical_payload(),
    )
    with _exclusive_paid_campaign_lock(suite_digest):
        return await _run_campaign(
            selected_mode,
            output_root=output_root,
            suite=selected_suite,
            services=services,
        )


async def _run_campaign(
    mode: CampaignMode,
    *,
    output_root: Path,
    suite: CampaignSuite,
    services: CampaignServices | None,
) -> CampaignOutcome:
    selected_mode = CampaignMode(mode)
    selected_suite = suite
    using_default_services = services is None
    selected_services = services or default_campaign_services()
    output_root = Path(output_root)

    deterministic = await selected_services.run_deterministic(output_root)
    if not deterministic.passed:
        raise CampaignError("deterministic gate failed")
    if selected_mode is CampaignMode.SMOKE:
        return CampaignOutcome(
            mode=selected_mode,
            deterministic=deterministic,
            preflight=None,
        )

    clean_worktree_required = selected_mode in {
        CampaignMode.FORMAL,
        CampaignMode.SHIP,
    }
    worktree_clean: bool | None = None
    if clean_worktree_required:
        worktree_clean = selected_services.worktree_is_clean()
        if not worktree_clean:
            raise CampaignError(
                "formal campaign requires a clean worktree",
            )
    codeflow_commit = _validated_codeflow_commit(
        selected_services.resolve_codeflow_commit(),
    )

    paid_modes = _paid_modes(selected_mode)
    estimate = estimate_worst_case_cost(
        selected_suite,
        modes=paid_modes,
    )
    if estimate.estimated_cny > selected_suite.budget.hard_cap_cny:
        raise CampaignError(
            "estimated worst-case cost exceeds the approved cost ceiling: "
            f"{estimate.estimated_cny:.2f} CNY > "
            f"{selected_suite.budget.hard_cap_cny:.2f} CNY",
        )
    suite_digest = canonical_digest(
        selected_suite.canonical_payload(),
    )
    claim_rules_digest = canonical_digest(
        selected_suite.claim_rules,
    )

    resolved = selected_services.resolve_provider()
    _validate_resolved_provider(selected_suite, resolved)
    environment_identity = _resolved_environment_identity(resolved)
    if selected_services.prepare_budget_guard is not None:
        resolved = selected_services.prepare_budget_guard(
            resolved,
            selected_suite,
            selected_mode,
            estimate,
            output_root,
            suite_digest,
            codeflow_commit,
        )
    if using_default_services and resolved.budget_ledger is None:
        raise CampaignError(
            "default paid campaign requires an actual Provider budget ledger",
        )
    registries = {
        paid_mode: selected_services.build_registry(
            paid_mode,
            resolved,
        )
        for paid_mode in paid_modes
    }
    provisional_specs = {
        paid_mode: _experiment_spec(
            selected_suite,
            paid_mode,
            output_root=output_root,
            suite_digest="provisional",
            claim_rules_digest="provisional",
            runtime_config_digest=resolved.runtime_config_digest,
            codeflow_commit=codeflow_commit,
            worktree_clean=worktree_clean,
            environment_identity=environment_identity,
        )
        for paid_mode in paid_modes
    }
    plans = {
        paid_mode: _validated_plan(
            provisional_specs[paid_mode],
            registries[paid_mode],
            _track_for(selected_suite, paid_mode),
        )
        for paid_mode in paid_modes
    }
    if selected_mode is CampaignMode.SHIP:
        _require_disjoint_calibration_and_formal(
            plans[CampaignMode.CALIBRATION],
            plans[CampaignMode.FORMAL],
        )

    request = ProviderProbeRequest(
        provider_name=selected_suite.provider.name,
        model=selected_suite.provider.model,
        codeflow_commit=codeflow_commit,
        worktree_clean=worktree_clean,
        suite_digest=suite_digest,
        claim_rules_digest=claim_rules_digest,
        estimated_cost=estimate,
        max_provider_calls_per_trial=(selected_suite.budget.max_provider_calls_per_trial),
        max_input_tokens_per_call=(selected_suite.budget.max_input_tokens_per_call),
        max_output_tokens_per_call=(selected_suite.budget.max_output_tokens_per_call),
    )
    probe = _read_cached_provider_probe(request, resolved) if using_default_services else None
    if probe is None:
        probe = await selected_services.probe_provider(request, resolved)
    _validate_probe(selected_suite, probe)
    if using_default_services:
        _persist_provider_probe(request, resolved, probe)
    _require_environment_identity(
        resolved,
        expected=environment_identity,
    )
    preflight = ProviderPreflightRecord(
        provider_name=probe.provider_name,
        requested_model=probe.requested_model,
        resolved_model=probe.resolved_model,
        codeflow_commit=codeflow_commit,
        worktree_clean=worktree_clean,
        suite_digest=suite_digest,
        claim_rules_digest=claim_rules_digest,
        runtime_config_digest=resolved.runtime_config_digest,
        tokenizer_identity=probe.tokenizer_identity,
        tokenizer_version=probe.tokenizer_version,
        tokenizer_digest=probe.tokenizer_digest,
        usage_fields=probe.usage_fields,
        seed_supported=probe.seed_supported,
        pricing_source=probe.pricing_source,
        estimated_worst_case_cny=estimate.estimated_cny,
        budget_warning_cny=selected_suite.budget.warning_cny,
        budget_hard_cap_cny=selected_suite.budget.hard_cap_cny,
    )

    warnings = []
    if estimate.estimated_cny >= selected_suite.budget.warning_cny:
        warnings.append(
            f"estimated worst-case cost reached the warning threshold: {estimate.estimated_cny:.2f} CNY",
        )

    experiments: list[ExperimentRef] = []
    reports: list[CampaignReport] = []
    for paid_mode in paid_modes:
        if clean_worktree_required:
            _revalidate_formal_git_state(
                selected_services,
                expected_commit=codeflow_commit,
            )
        spec = _experiment_spec(
            selected_suite,
            paid_mode,
            output_root=output_root,
            suite_digest=suite_digest,
            claim_rules_digest=claim_rules_digest,
            runtime_config_digest=resolved.runtime_config_digest,
            codeflow_commit=codeflow_commit,
            worktree_clean=worktree_clean,
            environment_identity=environment_identity,
            probe=probe,
        )
        _validated_plan(
            spec,
            registries[paid_mode],
            _track_for(selected_suite, paid_mode),
        )
        ref = await selected_services.execute_experiment(
            spec,
            registries[paid_mode],
        )
        report = selected_services.rebuild_report(ref)
        _validate_report(
            report,
            _track_for(selected_suite, paid_mode),
        )
        _require_environment_identity(
            resolved,
            expected=environment_identity,
        )
        _budget_snapshot(resolved)
        if clean_worktree_required:
            _revalidate_formal_git_state(
                selected_services,
                expected_commit=codeflow_commit,
            )
        experiments.append(ref)
        reports.append(report)

    campaign_artifact_path = (
        output_root / "campaigns" / suite_digest / codeflow_commit / selected_mode.value / "campaign-outcome.json"
    )
    outcome = CampaignOutcome(
        mode=selected_mode,
        deterministic=deterministic,
        preflight=preflight,
        experiments=tuple(experiments),
        reports=tuple(reports),
        warnings=tuple(warnings),
        budget_snapshot=_budget_snapshot(resolved),
        campaign_artifact_path=str(campaign_artifact_path),
    )
    if clean_worktree_required:
        _revalidate_formal_git_state(
            selected_services,
            expected_commit=codeflow_commit,
        )
    try:
        ArtifactStore(
            ExperimentRef(
                experiment_id=suite_digest,
                root=campaign_artifact_path.parent,
            )
        ).write_summary(
            campaign_artifact_path,
            to_primitive(outcome),
        )
    except OSError as exc:
        raise CampaignError(
            f"cannot persist campaign outcome: {exc}",
        ) from exc
    return outcome


@contextmanager
def _exclusive_paid_campaign_lock(suite_digest: str) -> Iterator[None]:
    lock_path = get_product_home() / "evidence" / "codeflowbench" / "campaign-locks" / f"{suite_digest}.lock"
    stack = ExitStack()
    try:
        stack.enter_context(file_lock(lock_path, blocking=False))
    except LockTimeoutError as exc:
        raise CampaignError(
            "suite already has an active paid campaign writer",
        ) from exc
    with stack:
        yield


def default_campaign_services() -> CampaignServices:
    return CampaignServices(
        run_deterministic=_run_deterministic_gate,
        worktree_is_clean=_worktree_is_clean,
        resolve_codeflow_commit=_resolve_codeflow_commit,
        resolve_provider=_resolve_provider,
        build_registry=_build_registry,
        probe_provider=_probe_provider,
        execute_experiment=_execute_experiment,
        rebuild_report=rebuild_experiment_report,
        prepare_budget_guard=_prepare_budget_guard,
    )


async def _run_deterministic_gate(
    output_root: Path,
) -> DeterministicGateResult:
    from codeflow.sandbox.config import SandboxConfig

    from .packs.runtime import (
        run_r0_scheduler_track,
        run_r1_full_runtime_track,
    )
    from .packs.tool_mcp import run_mcp_transport_smoke

    Path(output_root).mkdir(parents=True, exist_ok=True)
    codeflow_commit = _validated_codeflow_commit(_resolve_codeflow_commit())
    worktree_clean = _worktree_is_clean()
    environment_identity = capture_environment_identity(
        _REPOSITORY_ROOT,
        sandbox_config=SandboxConfig(),
    )
    with tempfile.TemporaryDirectory(
        prefix="codeflowbench-smoke-",
        dir=output_root,
    ) as temporary:
        root = Path(temporary)
        r0 = await run_r0_scheduler_track()
        r1 = await run_r1_full_runtime_track(root / "r1")
        mcp = await run_mcp_transport_smoke(root / "mcp")
        report_stable = await _exercise_report_rebuild(root / "report")
    details = {
        "r0": {
            "passed": r0.passed,
            "metrics": to_primitive(r0),
        },
        "r1": {
            "passed": r1.passed,
            "metrics": to_primitive(r1),
        },
        "mcp": {
            "passed": mcp.transport_closed and mcp.catalog_count > 0,
            "metrics": to_primitive(mcp),
        },
        "report_rebuild": {
            "passed": report_stable,
        },
    }
    final_codeflow_commit = _validated_codeflow_commit(_resolve_codeflow_commit())
    final_worktree_clean = _worktree_is_clean()
    if final_codeflow_commit != codeflow_commit or final_worktree_clean != worktree_clean:
        raise CampaignError(
            "Git state changed during deterministic Runtime tracks",
        )
    final_environment_identity = capture_environment_identity(
        _REPOSITORY_ROOT,
        sandbox_config=SandboxConfig(),
    )
    if final_environment_identity != environment_identity:
        raise CampaignError(
            "Execution environment changed during deterministic Runtime tracks",
        )
    passed = all(bool(value["passed"]) for value in details.values())
    evidence_payload = {
        "schema": _RUNTIME_EVIDENCE_SCHEMA,
        "codeflow_commit": codeflow_commit,
        "worktree_clean": worktree_clean,
        "environment": environment_identity,
        "evidence_scope": "deterministic_runtime_only",
        "claim_eligible": passed and worktree_clean,
        "details": details,
        "cv_metrics": (_runtime_cv_metrics(details) if passed and worktree_clean else {}),
    }
    evidence_digest = canonical_digest(evidence_payload)
    evidence = {
        **evidence_payload,
        "evidence_digest": evidence_digest,
    }
    evidence_root = output_root / "runtime-evidence" / codeflow_commit
    evidence_path = evidence_root / f"cv-metrics-runtime.{evidence_digest}.json"
    store = ArtifactStore(
        ExperimentRef(
            experiment_id=evidence_digest,
            root=evidence_root,
        )
    )
    try:
        store.append_immutable(evidence_path, evidence)
        _read_runtime_evidence(
            evidence_path,
            expected_digest=evidence_digest,
        )
    except (ArtifactError, OSError) as exc:
        raise CampaignError(
            f"cannot persist deterministic Runtime evidence: {exc}",
        ) from exc
    return DeterministicGateResult(
        passed=passed,
        details=details,
        evidence_path=str(evidence_path),
        evidence_digest=evidence_digest,
    )


def _runtime_cv_metrics(
    details: Mapping[str, Any],
) -> dict[str, Any]:
    r0 = details["r0"]["metrics"]
    r1 = details["r1"]["metrics"]
    return {
        "runtime.r0.accepted_requests": r0["accepted_requests"],
        "runtime.r0.conversation_count": r0["conversation_count"],
        "runtime.r0.lost_requests": r0["lost_requests"],
        "runtime.r0.unexpected_duplicate_executions": (r0["unexpected_duplicate_executions"]),
        "runtime.r0.unresolved_handles": r0["unresolved_handles"],
        "runtime.r0.lifecycle_contradictions": (r0["lifecycle_contradictions"]),
        "runtime.r0.pool_limit_violations": (r0["pool_limit_violations"]),
        "runtime.r0.peak_user_concurrency": (r0["peak_user_concurrency"]),
        "runtime.r0.peak_system_concurrency": (r0["peak_system_concurrency"]),
        "runtime.r0.max_observed_pending_depth": (r0["max_observed_pending_depth"]),
        "runtime.r0.dispatch_overhead_p95_ms": (r0["dispatch_overhead_ms"]["p95"]),
        "runtime.r0.queue_wait_p95_ms": r0["queue_wait_ms"]["p95"],
        "runtime.r0.execution_latency_p95_ms": (r0["execution_latency_ms"]["p95"]),
        "runtime.r1.turns": r1["turns"],
        "runtime.r1.submitted_through_scheduler": (r1["submitted_through_scheduler"]),
        "runtime.r1.runner_type": r1["runner_type"],
        "runtime.r1.terminal_counts": r1["terminal_counts"],
        "runtime.r1.model_calls": r1["model_calls"],
        "runtime.r1.tool_event_turns": r1["tool_event_turns"],
        "runtime.r1.readable_sessions": r1["readable_sessions"],
        "runtime.r1.delivery_counts": r1["delivery_counts"],
        "runtime.r1.identity_contradictions": (r1["identity_contradictions"]),
        "runtime.r1.lifecycle_contradictions": (r1["lifecycle_contradictions"]),
        "runtime.r1.unresolved_handles": r1["unresolved_handles"],
        "runtime.r1.resources_closed": r1["resources_closed"],
    }


def _read_runtime_evidence(
    path: Path,
    *,
    expected_digest: str,
) -> dict[str, Any]:
    store = ArtifactStore(
        ExperimentRef(
            experiment_id=expected_digest,
            root=path.parent,
        )
    )
    evidence = store.read_json(path)
    if evidence.get("schema") != _RUNTIME_EVIDENCE_SCHEMA:
        raise ArtifactError("Runtime evidence schema is invalid")
    recorded_digest = evidence.get("evidence_digest")
    if not isinstance(recorded_digest, str):
        raise ArtifactError("Runtime evidence digest is missing")
    payload = dict(evidence)
    payload.pop("evidence_digest")
    if recorded_digest != canonical_digest(payload):
        raise ArtifactError("Runtime evidence digest is invalid")
    if recorded_digest != expected_digest:
        raise ArtifactError("Runtime evidence digest does not match the expected value")
    if path.name != f"cv-metrics-runtime.{expected_digest}.json":
        raise ArtifactError("Runtime evidence path does not match the expected digest")
    environment = evidence.get("environment")
    if not isinstance(environment, Mapping):
        raise ArtifactError("Runtime evidence environment identity is missing")
    try:
        validate_environment_identity(environment)
    except EnvironmentIdentityError as exc:
        raise ArtifactError(
            f"Runtime evidence environment identity is invalid: {exc}",
        ) from exc
    return evidence


async def _exercise_report_rebuild(root: Path) -> bool:
    from .protocol import TrialContext, TrialExecution
    from .records import (
        DeliveryOutcome,
        TrialStatus,
        TurnTerminalState,
        VerificationState,
        VerifierResult,
    )
    from .schema import PackDefinition, TaskSpec, VariantSpec

    class SmokePack:
        def definition(self) -> PackDefinition:
            return PackDefinition(
                pack_id="smoke-report",
                tasks=(TaskSpec(task_id="smoke-report-task"),),
                variants=(
                    VariantSpec(
                        variant_id="smoke-report-variant",
                        settings={"smoke_axis": "stable"},
                    ),
                ),
                pairs=(),
            )

        async def run_trial(
            self,
            context: TrialContext,
        ) -> TrialExecution:
            return TrialExecution(
                status=TrialStatus.PASSED,
                runtime_state=TurnTerminalState.COMPLETED,
                delivery_state=DeliveryOutcome.DELIVERED,
                verification=VerifierResult(
                    state=VerificationState.PASSED,
                ),
                observed_variant_settings=dict(
                    context.variant.settings,
                ),
            )

    registry = PackRegistry()
    registry.register(SmokePack())
    spec = ExperimentSpec(
        suite="smoke-report",
        repetitions=1,
        pack_ids=("smoke-report",),
        output_root=root,
        identity={"kind": "credential-free-report-rebuild"},
    )
    ref = await run_experiment(spec, registry=registry)
    first = rebuild_experiment_report(ref)
    second = rebuild_experiment_report(ref)
    return first.report_digest == second.report_digest and first.ship_complete


def _worktree_is_clean() -> bool:
    git = shutil.which("git")
    if git is None:
        return False
    completed = subprocess.run(
        [git, "status", "--short"],
        check=False,
        capture_output=True,
        text=True,
        cwd=_REPOSITORY_ROOT,
    )
    return completed.returncode == 0 and not completed.stdout.strip()


def _resolve_codeflow_commit() -> str:
    git = shutil.which("git")
    if git is None:
        raise CampaignError("git is required to bind CodeFlowBench evidence")
    completed = subprocess.run(
        [git, "rev-parse", "--verify", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        cwd=_REPOSITORY_ROOT,
    )
    if completed.returncode != 0:
        raise CampaignError("cannot resolve the current CodeFlow commit")
    return _validated_codeflow_commit(completed.stdout.strip())


def _resolve_provider() -> ResolvedProvider:
    from codeflow.cli._helpers import make_provider
    from codeflow.config.loader import load_config
    from codeflow.config.codeflow import load_codeflow_config

    config = load_config()
    codeflow_config = load_codeflow_config()
    configured_model = config.agents.defaults.model
    provider_name = config.get_provider_name(configured_model)
    provider = make_provider(config)
    return ResolvedProvider(
        provider_name=provider_name,
        model=_model_identity(configured_model, provider_name),
        provider=provider,
        runtime_config_digest=canonical_digest(
            _redacted_runtime_config(config, codeflow_config),
        ),
        config=config,
        codeflow_config=codeflow_config,
        configured_model=configured_model,
    )


def _prepare_budget_guard(
    resolved: ResolvedProvider,
    suite: CampaignSuite,
    mode: CampaignMode,
    estimate: CostEstimate,
    output_root: Path,
    suite_digest: str,
    codeflow_commit: str,
) -> ResolvedProvider:
    del mode, output_root, estimate
    provider = resolved.provider
    if not isinstance(provider, LLMProvider):
        raise CampaignError("resolved Provider is not an LLMProvider")
    approval_id = canonical_digest(
        {
            "schema": suite.schema,
            "suite": suite.suite,
        }
    )
    approval_root = get_product_home() / "evidence" / "codeflowbench" / "approvals" / approval_id
    ledger_path = approval_root / "provider-budget.jsonl"
    common_config = {
        "hard_cap_cny": suite.budget.hard_cap_cny,
        "external_service_reserve_cny": (suite.budget.external_service_reserve_cny),
        "max_input_tokens_per_call": (suite.budget.max_input_tokens_per_call),
        "max_output_tokens_per_call": (suite.budget.max_output_tokens_per_call),
        "input_cache_miss_usd_per_million": (suite.budget.input_cache_miss_usd_per_million),
        "output_usd_per_million": (suite.budget.output_usd_per_million),
        "conservative_usd_to_cny_multiplier": (suite.budget.conservative_usd_to_cny_multiplier),
    }
    authorization_estimate = estimate_worst_case_cost(
        suite,
        modes=(
            CampaignMode.CALIBRATION,
            CampaignMode.FORMAL,
        ),
    )
    authorized_new_request_attempts = authorization_estimate.provider_request_attempts + _PROVIDER_PROBE_MAX_ATTEMPTS
    max_additional_request_attempts = suite.budget.max_additional_provider_attempts + _PROVIDER_PROBE_MAX_ATTEMPTS - 1
    approval_generation = canonical_digest(
        {
            "schema": _PROVIDER_BUDGET_APPROVAL_SCHEMA,
            "suite_digest": suite_digest,
            "codeflow_commit": codeflow_commit,
            "runtime_config_digest": resolved.runtime_config_digest,
        }
    )
    approval_history_root = approval_root / "provider-budget-approvals"
    approval_path = approval_history_root / f"{approval_generation}.v3.json"
    inspection_overrides: dict[str, Any] = {
        "max_total_request_attempts": 2**63 - 1,
    }
    if approval_path.exists():
        frozen_approval = _read_provider_budget_approval(
            approval_path,
        )
        _validate_provider_budget_approval(
            frozen_approval,
            suite_digest=suite_digest,
            codeflow_commit=codeflow_commit,
            runtime_config_digest=resolved.runtime_config_digest,
            authorized_new_request_attempts=(authorized_new_request_attempts),
            max_additional_request_attempts=(max_additional_request_attempts),
        )
        inspection_overrides = {
            "max_total_request_attempts": int(
                frozen_approval["request_attempt_lifetime_ceiling"],
            ),
            "request_attempt_baseline": int(
                frozen_approval["request_attempt_baseline"],
            ),
            "approval_digest": str(
                frozen_approval["approval_digest"],
            ),
            "max_additional_request_attempts": int(
                frozen_approval["max_additional_request_attempts"],
            ),
            "ledger_prefix_event_count": int(
                frozen_approval["ledger_prefix_event_count"],
            ),
            "ledger_prefix_digest": str(
                frozen_approval["ledger_prefix_digest"],
            ),
            "ledger_prefix_charged_cny": float(
                frozen_approval["ledger_prefix_charged_cny"],
            ),
        }
    elif approval_history_root.exists() and any(
        approval_history_root.glob("*.v3.json"),
    ):
        inspection_overrides["approval_digest"] = "frozen-approval-history"
    inspection_ledger = ProviderBudgetLedger(
        ledger_path,
        ProviderBudgetConfig(
            **common_config,
            **inspection_overrides,
        ),
    )
    existing = inspection_ledger.snapshot()
    if not existing.accounting_complete or existing.total_committed_cny > suite.budget.hard_cap_cny:
        raise CampaignError(
            "existing Provider budget ledger is incomplete or exceeds the campaign budget",
        )
    if not approval_path.exists():
        projected_total_cny = (
            existing.provider_charged_cny
            + authorization_estimate.estimated_cny
            + (_PROVIDER_PROBE_MAX_ATTEMPTS * inspection_ledger.config.maximum_request_cny)
        )
        if projected_total_cny > suite.budget.hard_cap_cny:
            raise CampaignError(
                "existing Provider spend leaves insufficient campaign budget",
            )
    approval = _provider_budget_approval(
        approval_path,
        suite_digest=suite_digest,
        codeflow_commit=codeflow_commit,
        runtime_config_digest=resolved.runtime_config_digest,
        request_attempt_baseline=existing.request_attempts,
        authorized_new_request_attempts=(authorized_new_request_attempts),
        max_additional_request_attempts=(max_additional_request_attempts),
        ledger_prefix_event_count=existing.ledger_event_count,
        ledger_prefix_digest=existing.ledger_digest,
        ledger_prefix_charged_cny=existing.provider_charged_cny,
    )
    ledger = ProviderBudgetLedger(
        ledger_path,
        ProviderBudgetConfig(
            **common_config,
            max_total_request_attempts=int(
                approval["request_attempt_lifetime_ceiling"],
            ),
            request_attempt_baseline=int(
                approval["request_attempt_baseline"],
            ),
            approval_digest=str(approval["approval_digest"]),
            max_additional_request_attempts=int(
                approval["max_additional_request_attempts"],
            ),
            ledger_prefix_event_count=int(
                approval["ledger_prefix_event_count"],
            ),
            ledger_prefix_digest=str(
                approval["ledger_prefix_digest"],
            ),
            ledger_prefix_charged_cny=float(
                approval["ledger_prefix_charged_cny"],
            ),
        ),
    )
    snapshot = ledger.snapshot()
    if (
        not snapshot.accounting_complete
        or snapshot.total_committed_cny > snapshot.hard_cap_cny
        or snapshot.request_attempts < snapshot.request_attempt_baseline
        or snapshot.request_attempts > snapshot.request_attempt_lifetime_ceiling
        or (
            snapshot.additional_request_attempt_ceiling is not None
            and snapshot.additional_request_attempts > snapshot.additional_request_attempt_ceiling
        )
    ):
        raise CampaignError(
            "Provider request-attempt ledger does not match its frozen approval",
        )
    return replace(
        resolved,
        provider=BudgetGuardedProvider(
            provider,
            ledger=ledger,
        ),
        budget_ledger=ledger,
    )


def _provider_budget_approval(
    path: Path,
    *,
    suite_digest: str,
    codeflow_commit: str,
    runtime_config_digest: str,
    request_attempt_baseline: int,
    authorized_new_request_attempts: int,
    max_additional_request_attempts: int,
    ledger_prefix_event_count: int,
    ledger_prefix_digest: str,
    ledger_prefix_charged_cny: float,
) -> dict[str, Any]:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CampaignError(
                "Provider budget approval is unreadable",
            ) from exc
        _validate_provider_budget_approval(
            existing,
            suite_digest=suite_digest,
            codeflow_commit=codeflow_commit,
            runtime_config_digest=runtime_config_digest,
            authorized_new_request_attempts=(authorized_new_request_attempts),
            max_additional_request_attempts=(max_additional_request_attempts),
        )
        return existing

    payload = {
        "schema": _PROVIDER_BUDGET_APPROVAL_SCHEMA,
        "suite_digest": suite_digest,
        "codeflow_commit": codeflow_commit,
        "runtime_config_digest": runtime_config_digest,
        "request_attempt_baseline": request_attempt_baseline,
        "authorized_new_request_attempts": (authorized_new_request_attempts),
        "request_attempt_lifetime_ceiling": (request_attempt_baseline + authorized_new_request_attempts),
        "max_additional_request_attempts": (max_additional_request_attempts),
        "ledger_prefix_event_count": ledger_prefix_event_count,
        "ledger_prefix_digest": ledger_prefix_digest,
        "ledger_prefix_charged_cny": ledger_prefix_charged_cny,
    }
    record = {
        **payload,
        "approval_digest": canonical_digest(payload),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(
                to_primitive(record),
                handle,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write("\n")
    except FileExistsError:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CampaignError(
                "Provider budget approval is unreadable",
            ) from exc
        _validate_provider_budget_approval(
            existing,
            suite_digest=suite_digest,
            codeflow_commit=codeflow_commit,
            runtime_config_digest=runtime_config_digest,
            authorized_new_request_attempts=(authorized_new_request_attempts),
            max_additional_request_attempts=(max_additional_request_attempts),
        )
        return existing
    return record


def _read_provider_budget_approval(
    path: Path,
) -> dict[str, Any]:
    try:
        approval = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(
            "Provider budget approval is unreadable",
        ) from exc
    if not isinstance(approval, dict):
        raise CampaignError("Provider budget approval is invalid")
    return approval


def _validate_provider_budget_approval(
    approval: object,
    *,
    suite_digest: str,
    codeflow_commit: str,
    runtime_config_digest: str,
    authorized_new_request_attempts: int,
    max_additional_request_attempts: int,
) -> None:
    if not isinstance(approval, dict):
        raise CampaignError("Provider budget approval is invalid")
    try:
        baseline = int(approval["request_attempt_baseline"])
        authorized = int(
            approval["authorized_new_request_attempts"],
        )
        ceiling = int(
            approval["request_attempt_lifetime_ceiling"],
        )
        additional = int(
            approval["max_additional_request_attempts"],
        )
        prefix_event_count = int(
            approval["ledger_prefix_event_count"],
        )
        prefix_digest = str(
            approval["ledger_prefix_digest"],
        )
        prefix_charged_cny = approval["ledger_prefix_charged_cny"]
        if not isinstance(prefix_charged_cny, int | float) or isinstance(
            prefix_charged_cny,
            bool,
        ):
            raise TypeError("ledger prefix charge must be numeric")
        payload = {
            "schema": str(approval["schema"]),
            "suite_digest": str(approval["suite_digest"]),
            "codeflow_commit": str(approval["codeflow_commit"]),
            "runtime_config_digest": str(
                approval["runtime_config_digest"],
            ),
            "request_attempt_baseline": baseline,
            "authorized_new_request_attempts": authorized,
            "request_attempt_lifetime_ceiling": ceiling,
            "max_additional_request_attempts": additional,
            "ledger_prefix_event_count": prefix_event_count,
            "ledger_prefix_digest": prefix_digest,
            "ledger_prefix_charged_cny": prefix_charged_cny,
        }
        approval_digest = str(approval["approval_digest"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CampaignError(
            "Provider budget approval is invalid",
        ) from exc
    if (
        payload["schema"] != _PROVIDER_BUDGET_APPROVAL_SCHEMA
        or payload["suite_digest"] != suite_digest
        or payload["codeflow_commit"] != codeflow_commit
        or payload["runtime_config_digest"] != runtime_config_digest
        or authorized != authorized_new_request_attempts
        or additional != max_additional_request_attempts
        or ceiling != baseline + authorized
        or prefix_event_count < 0
        or not prefix_digest
        or float(prefix_charged_cny) < 0
        or approval_digest != canonical_digest(payload)
    ):
        raise CampaignError(
            "Provider budget approval does not match the frozen campaign",
        )


def _build_registry(
    mode: CampaignMode,
    resolved: ResolvedProvider,
) -> PackRegistry:
    from .packs.context import (
        ContextPack,
        ContextTrack,
        RuntimeContextTrialRunner,
    )
    from .packs.memory_skill import (
        ProductionSemanticMemoryEffectRunner,
        create_calibration_pack,
        create_formal_pack,
        create_semantic_memory_effect_calibration_pack,
        create_semantic_memory_effect_pack,
    )
    from .packs.tool_mcp import (
        MCPRuntimeTrialRunner,
        ToolMCPPack,
        ToolMCPTrack,
    )

    if resolved.config is None or resolved.codeflow_config is None:
        raise CampaignError(
            "resolved Provider is missing Runtime configuration",
        )
    provider = resolved.provider
    if not isinstance(provider, LLMProvider):
        raise CampaignError("resolved Provider is not an LLMProvider")
    try:
        from .packs.memory_skill import RuntimeCrossProcessRunner
    except ImportError as exc:
        raise CampaignError(
            "real Memory/Skill cross-session runner is not available; formal campaign fails closed",
        ) from exc

    calibration = mode is CampaignMode.CALIBRATION
    context_track = ContextTrack.CALIBRATION if calibration else ContextTrack.FORMAL
    tool_track = ToolMCPTrack.CALIBRATION if calibration else ToolMCPTrack.FORMAL
    memory_runner = RuntimeCrossProcessRunner(
        config=resolved.config,
        codeflow_config=resolved.codeflow_config,
        provider=provider,
    )
    memory_pack = create_calibration_pack(memory_runner) if calibration else create_formal_pack(memory_runner)
    semantic_memory_runner = ProductionSemanticMemoryEffectRunner(
        config=resolved.config,
        codeflow_config=resolved.codeflow_config,
        provider=provider,
    )
    semantic_memory_pack = (
        create_semantic_memory_effect_calibration_pack(
            semantic_memory_runner,
        )
        if calibration
        else create_semantic_memory_effect_pack(
            semantic_memory_runner,
        )
    )
    if getattr(memory_pack, "retrieval_cost_mode", None) != "local_zero_provider":
        raise CampaignError(
            "retrieval cost mode is not frozen as local zero-Provider",
        )
    if (
        resolved.budget_ledger is None
        or not isinstance(provider, BudgetGuardedProvider)
        or provider.ledger is not resolved.budget_ledger
    ):
        raise CampaignError(
            "Memory subprocesses must share the campaign Provider ledger",
        )
    registry = PackRegistry()
    registry.register(
        ContextPack(
            context_track,
            runner=RuntimeContextTrialRunner(
                config=resolved.config,
                codeflow_config=resolved.codeflow_config,
                provider=provider,
            ),
        )
    )
    registry.register(memory_pack)
    registry.register(semantic_memory_pack)
    registry.register(
        ToolMCPPack(
            tool_track,
            runner=MCPRuntimeTrialRunner(
                provider=provider,
                model=(resolved.configured_model or resolved.provider_name + "/" + resolved.model),
            ),
        )
    )
    return registry


def _provider_probe_cache_path(
    resolved: ResolvedProvider,
) -> Path | None:
    ledger = resolved.budget_ledger
    if ledger is None or not ledger.config.approval_digest:
        return None
    return ledger.path.parent / "provider-preflights" / f"{ledger.config.approval_digest}.json"


def _provider_probe_cache_identity(
    request: ProviderProbeRequest,
    resolved: ResolvedProvider,
) -> str:
    ledger = resolved.budget_ledger
    return canonical_digest(
        {
            "request": to_primitive(request),
            "runtime_config_digest": resolved.runtime_config_digest,
            "approval_digest": (ledger.config.approval_digest if ledger is not None else None),
        }
    )


def _read_cached_provider_probe(
    request: ProviderProbeRequest,
    resolved: ResolvedProvider,
) -> ProviderProbeResult | None:
    path = _provider_probe_cache_path(resolved)
    if path is None or not path.exists():
        return None
    store = ArtifactStore(
        ExperimentRef(
            experiment_id=path.stem,
            root=path.parent,
        )
    )
    try:
        record = store.read_json(path)
    except ArtifactError as exc:
        raise CampaignError(
            "cached Provider preflight is unreadable",
        ) from exc
    if set(record) != {
        "schema",
        "identity_digest",
        "result",
        "record_digest",
    }:
        raise CampaignError("cached Provider preflight is invalid")
    payload = {
        "schema": record.get("schema"),
        "identity_digest": record.get("identity_digest"),
        "result": record.get("result"),
    }
    if (
        payload["schema"] != _PROVIDER_PREFLIGHT_CACHE_SCHEMA
        or payload["identity_digest"] != _provider_probe_cache_identity(request, resolved)
        or record.get("record_digest") != canonical_digest(payload)
        or not isinstance(payload["result"], dict)
    ):
        raise CampaignError("cached Provider preflight is invalid")
    result = payload["result"]
    usage_fields = result.get("usage_fields")
    string_fields = (
        "provider_name",
        "requested_model",
        "resolved_model",
        "tokenizer_identity",
        "tokenizer_version",
        "tokenizer_digest",
        "pricing_source",
    )
    if (
        set(result)
        != {
            *string_fields,
            "tool_calling_supported",
            "usage_fields",
            "seed_supported",
            "fallback_used",
        }
        or any(not isinstance(result.get(field), str) for field in string_fields)
        or not isinstance(usage_fields, list)
        or any(not isinstance(field, str) for field in usage_fields)
        or not all(
            isinstance(result.get(field), bool)
            for field in (
                "tool_calling_supported",
                "seed_supported",
                "fallback_used",
            )
        )
    ):
        raise CampaignError("cached Provider preflight is invalid")
    return ProviderProbeResult(
        provider_name=result["provider_name"],
        requested_model=result["requested_model"],
        resolved_model=result["resolved_model"],
        tool_calling_supported=result["tool_calling_supported"],
        usage_fields=tuple(usage_fields),
        seed_supported=result["seed_supported"],
        tokenizer_identity=result["tokenizer_identity"],
        tokenizer_version=result["tokenizer_version"],
        tokenizer_digest=result["tokenizer_digest"],
        pricing_source=result["pricing_source"],
        fallback_used=result["fallback_used"],
    )


def _persist_provider_probe(
    request: ProviderProbeRequest,
    resolved: ResolvedProvider,
    probe: ProviderProbeResult,
) -> None:
    path = _provider_probe_cache_path(resolved)
    if path is None:
        raise CampaignError(
            "default paid campaign cannot bind Provider preflight",
        )
    payload = {
        "schema": _PROVIDER_PREFLIGHT_CACHE_SCHEMA,
        "identity_digest": _provider_probe_cache_identity(
            request,
            resolved,
        ),
        "result": to_primitive(probe),
    }
    record = {
        **payload,
        "record_digest": canonical_digest(payload),
    }
    store = ArtifactStore(
        ExperimentRef(
            experiment_id=path.stem,
            root=path.parent,
        )
    )
    try:
        store.append_immutable(path, record)
    except ArtifactError as exc:
        raise CampaignError(
            "cannot persist immutable Provider preflight",
        ) from exc


async def _probe_provider(
    request: ProviderProbeRequest,
    resolved: ResolvedProvider,
) -> ProviderProbeResult:
    provider = resolved.provider
    if not isinstance(provider, LLMProvider):
        raise CampaignError("resolved Provider is not callable")
    configured_model = resolved.configured_model or request.model
    tools = [
        {
            "type": "function",
            "function": {
                "name": "codeflowbench_preflight_echo",
                "description": ("Return the exact probe value to verify Tool Calling."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                    },
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        },
    ]

    def valid_probe_call(call: ToolCallRequest) -> bool:
        return (
            call.name == "codeflowbench_preflight_echo"
            and isinstance(call.arguments, dict)
            and call.arguments.get("value") == "ready"
        )

    with provider_call_budget_scope(
        trial_id=(f"preflight/{request.codeflow_commit}/{request.suite_digest}/{resolved.runtime_config_digest}"),
        max_logical_calls=1,
        max_attempts_per_call=_PROVIDER_PROBE_MAX_ATTEMPTS,
    ):
        for _ in range(_PROVIDER_PROBE_MAX_ATTEMPTS):
            response = await asyncio.wait_for(
                provider.chat(
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                'Call codeflowbench_preflight_echo exactly once with value "ready". Do not answer with text.'
                            ),
                        },
                    ],
                    tools=tools,
                    model=configured_model,
                    max_tokens=request.max_output_tokens_per_call,
                    temperature=0,
                ),
                timeout=60,
            )
            if response.finish_reason == "error" or response.error_classification is not None:
                classification = response.error_classification
                category = classification.category if classification is not None else "unknown"
                raise CampaignError(
                    f"Provider preflight call failed ({category})",
                )
            if any(valid_probe_call(call) for call in response.tool_calls):
                break
    usage_fields = tuple(sorted(field for field in _REQUIRED_USAGE_FIELDS if response.usage.get(field) is not None))
    tool_calling_supported = any(valid_probe_call(call) for call in response.tool_calls)
    if not response.model:
        raise CampaignError(
            "Provider preflight did not report actual model identity",
        )
    actual_model = _model_identity(response.model, resolved.provider_name)
    tokenizer_identity, tokenizer_version, tokenizer_digest = _tokenizer_identity(configured_model)
    return ProviderProbeResult(
        provider_name=resolved.provider_name,
        requested_model=request.model,
        resolved_model=actual_model,
        tool_calling_supported=tool_calling_supported,
        usage_fields=usage_fields,
        seed_supported=False,
        tokenizer_identity=tokenizer_identity,
        tokenizer_version=tokenizer_version,
        tokenizer_digest=tokenizer_digest,
        pricing_source=request.estimated_cost.pricing_source,
        fallback_used=actual_model != resolved.model,
    )


async def _execute_experiment(
    spec: ExperimentSpec,
    registry: PackRegistry,
) -> ExperimentRef:
    return await run_experiment(spec, registry=registry)


def _paid_modes(mode: CampaignMode) -> tuple[CampaignMode, ...]:
    if mode is CampaignMode.SHIP:
        return (
            CampaignMode.CALIBRATION,
            CampaignMode.FORMAL,
        )
    if mode in {CampaignMode.CALIBRATION, CampaignMode.FORMAL}:
        return (mode,)
    raise ValueError(f"{mode.value} has no paid track")


def _track_for(
    suite: CampaignSuite,
    mode: CampaignMode,
) -> TrackConfig:
    if mode is CampaignMode.CALIBRATION:
        return suite.calibration
    if mode is CampaignMode.FORMAL:
        return suite.formal
    raise ValueError(f"{mode.value} is not a track")


def _experiment_spec(
    suite: CampaignSuite,
    mode: CampaignMode,
    *,
    output_root: Path,
    suite_digest: str,
    claim_rules_digest: str,
    runtime_config_digest: str,
    codeflow_commit: str,
    worktree_clean: bool | None,
    environment_identity: Mapping[str, Any],
    probe: ProviderProbeResult | None = None,
) -> ExperimentSpec:
    track = _track_for(suite, mode)
    identity = {
        "campaign_schema": suite.schema,
        "campaign_suite": suite.suite,
        "campaign_mode": mode.value,
        "codeflow_commit": codeflow_commit,
        "worktree_clean": worktree_clean,
        "clean_worktree_required": worktree_clean is True,
        "suite_digest": suite_digest,
        "claim_rules_digest": claim_rules_digest,
        "provider": suite.provider.name,
        "model": suite.provider.model,
        "runtime_config_digest": runtime_config_digest,
        "environment": dict(environment_identity),
        "budget_hard_cap_cny": suite.budget.hard_cap_cny,
        "provider_call_max_attempts": (suite.execution.provider_call_max_attempts),
        "max_comparison_block_attempts": (suite.execution.max_comparison_block_attempts),
        "max_comparison_block_retries_total": (track.max_comparison_block_retries_total),
        "max_retrieval_query_block_attempts": (suite.execution.max_retrieval_query_block_attempts),
    }
    if probe is not None:
        identity.update(
            {
                "resolved_model": probe.resolved_model,
                "tokenizer_identity": probe.tokenizer_identity,
                "tokenizer_version": probe.tokenizer_version,
                "tokenizer_digest": probe.tokenizer_digest,
                "seed_supported": probe.seed_supported,
                "pricing_source": probe.pricing_source,
            }
        )
    return ExperimentSpec(
        suite=track.suite,
        repetitions=track.repetitions,
        pack_ids=track.pack_ids,
        output_root=output_root,
        identity=identity,
        execution=replace(
            suite.execution,
            max_comparison_block_retries_total=(track.max_comparison_block_retries_total),
            max_provider_calls_per_trial=(suite.budget.max_provider_calls_per_trial),
            provider_trial_budgets=tuple(suite.budget.provider_trial_budget_for(pack_id) for pack_id in track.pack_ids),
        ),
        claim_rules=(suite.claim_rules if mode is CampaignMode.FORMAL else ()),
    )


def _resolved_environment_identity(
    resolved: ResolvedProvider,
) -> dict[str, Any]:
    config = resolved.config
    if config is None and resolved.codeflow_config is not None:
        config = getattr(resolved.codeflow_config, "base", None)
    tools = getattr(config, "tools", None)
    sandbox_config = getattr(tools, "sandbox", None)
    try:
        return capture_environment_identity(
            _REPOSITORY_ROOT,
            sandbox_config=sandbox_config,
        )
    except EnvironmentIdentityError as exc:
        raise CampaignError(
            f"cannot seal execution environment identity: {exc}",
        ) from exc


def _require_environment_identity(
    resolved: ResolvedProvider,
    *,
    expected: Mapping[str, Any],
) -> None:
    actual = _resolved_environment_identity(resolved)
    if actual != dict(expected):
        raise CampaignError(
            "Execution environment changed during paid campaign",
        )


def _validated_plan(
    spec: ExperimentSpec,
    registry: PackRegistry,
    track: TrackConfig,
) -> ExperimentPlan:
    try:
        plan = compile_plan(
            spec,
            registry.resolve(spec.pack_ids),
        )
    except (KeyError, ValueError) as exc:
        raise CampaignError(
            f"cannot compile {track.suite}: {exc}",
        ) from exc
    if len(plan.trials) != track.expected_trials:
        raise CampaignError(
            f"{track.suite} planned Trial denominator drift: {len(plan.trials)} != {track.expected_trials}",
        )
    if len(plan.retrieval_cases) != track.expected_retrieval_cases:
        raise CampaignError(
            f"{track.suite} planned Retrieval Case denominator drift: "
            f"{len(plan.retrieval_cases)} != "
            f"{track.expected_retrieval_cases}",
        )
    expected_trials_by_pack = dict(track.expected_trials_by_pack)
    expected_blocks_by_pack = dict(
        track.expected_comparison_blocks_by_pack,
    )
    expected_retrieval_cases_by_pack = dict(
        track.expected_retrieval_cases_by_pack,
    )
    for definition in plan.pack_definitions:
        actual_trials = sum(1 for trial in plan.trials if trial.key.pack_id == definition.pack_id)
        expected_trials = expected_trials_by_pack[definition.pack_id]
        if actual_trials != expected_trials:
            raise CampaignError(
                f"{definition.pack_id} planned Trial denominator drift: {actual_trials} != {expected_trials}",
            )
        actual_blocks = sum(1 for block in plan.comparison_blocks if block.key.pack_id == definition.pack_id)
        expected_blocks = expected_blocks_by_pack[definition.pack_id]
        if actual_blocks != expected_blocks:
            raise CampaignError(
                f"{definition.pack_id} planned Comparison Block denominator drift: "
                f"{actual_blocks} != {expected_blocks}",
            )
        actual_retrieval_cases = sum(
            len(suite.queries) * len(suite.configurations) for suite in definition.retrieval_suites
        )
        expected_retrieval_cases = expected_retrieval_cases_by_pack[definition.pack_id]
        if actual_retrieval_cases != expected_retrieval_cases:
            raise CampaignError(
                f"{definition.pack_id} planned Retrieval Case denominator drift: "
                f"{actual_retrieval_cases} != "
                f"{expected_retrieval_cases}",
            )
    return plan


def _require_disjoint_calibration_and_formal(
    calibration: ExperimentPlan,
    formal: ExperimentPlan,
) -> None:
    calibration_tasks = {trial.key.task_id for trial in calibration.trials}
    formal_tasks = {trial.key.task_id for trial in formal.trials}
    calibration_queries = {case.key.query_id for case in calibration.retrieval_cases}
    formal_queries = {case.key.query_id for case in formal.retrieval_cases}
    overlap = (calibration_tasks & formal_tasks) | (calibration_queries & formal_queries)
    if overlap:
        raise CampaignError(
            "calibration and formal task/query ids must be disjoint: " + ", ".join(sorted(overlap)),
        )


def _validate_resolved_provider(
    suite: CampaignSuite,
    resolved: ResolvedProvider,
) -> None:
    if resolved.provider_name != suite.provider.name:
        raise CampaignError(
            "configured Provider identity does not match the frozen suite: "
            f"{resolved.provider_name!r} != {suite.provider.name!r}",
        )
    if resolved.model != suite.provider.model:
        raise CampaignError(
            "configured model identity does not match the frozen suite: "
            f"{resolved.model!r} != {suite.provider.model!r}",
        )


def _validate_probe(
    suite: CampaignSuite,
    probe: ProviderProbeResult,
) -> None:
    if probe.provider_name != suite.provider.name:
        raise CampaignError("Provider identity changed during preflight")
    if probe.requested_model != suite.provider.model or probe.resolved_model != suite.provider.model:
        raise CampaignError("model identity changed during preflight")
    if not probe.tool_calling_supported:
        raise CampaignError(
            "Provider preflight did not prove Tool Calling",
        )
    missing_usage = _REQUIRED_USAGE_FIELDS - set(probe.usage_fields)
    if missing_usage:
        raise CampaignError(
            "Provider preflight is missing required usage fields: " + ", ".join(sorted(missing_usage)),
        )
    if not probe.tokenizer_identity or not probe.tokenizer_version or not probe.tokenizer_digest:
        raise CampaignError(
            "Provider preflight did not freeze a tokenizer identity",
        )
    if probe.fallback_used:
        raise CampaignError(
            "Provider preflight used a fallback model",
        )
    if not probe.pricing_source:
        raise CampaignError(
            "Provider preflight did not freeze a pricing source",
        )


def _validated_codeflow_commit(value: str) -> str:
    commit = value.strip().lower()
    if len(commit) not in {40, 64} or any(character not in "0123456789abcdef" for character in commit):
        raise CampaignError("current CodeFlow commit is not a full Git object id")
    return commit


def _revalidate_formal_git_state(
    services: CampaignServices,
    *,
    expected_commit: str,
) -> None:
    if not services.worktree_is_clean():
        raise CampaignError(
            "formal campaign worktree changed after preflight",
        )
    current_commit = _validated_codeflow_commit(
        services.resolve_codeflow_commit(),
    )
    if current_commit != expected_commit:
        raise CampaignError(
            "formal campaign commit changed after preflight",
        )


def _validate_report(
    report: CampaignReport,
    track: TrackConfig,
) -> None:
    if report.planned_trials != track.expected_trials:
        raise CampaignError(
            f"{track.suite} report Trial denominator drift",
        )
    if report.terminal_trials != track.expected_trials:
        raise CampaignError(
            f"{track.suite} is missing terminal Trial records",
        )
    if report.planned_retrieval_cases != track.expected_retrieval_cases:
        raise CampaignError(
            f"{track.suite} report Retrieval Case denominator drift",
        )
    if report.terminal_retrieval_cases != track.expected_retrieval_cases:
        raise CampaignError(
            f"{track.suite} is missing terminal Retrieval Case records",
        )
    if not report.ship_complete:
        raise CampaignError(
            f"{track.suite} report is not Ship-complete",
        )
    if not report.measurement_valid:
        raise CampaignError(
            f"{track.suite} report measurement is invalid",
        )


def _budget_snapshot(
    resolved: ResolvedProvider,
) -> ProviderBudgetSnapshot | None:
    if resolved.budget_ledger is None:
        return None
    snapshot = resolved.budget_ledger.snapshot()
    if not snapshot.accounting_complete:
        raise CampaignError(
            "actual Provider usage accounting is incomplete",
        )
    if snapshot.total_committed_cny > snapshot.hard_cap_cny:
        raise CampaignError(
            "actual Provider spending crossed the CNY hard cap",
        )
    return snapshot


def _track_config(raw: Mapping[str, Any]) -> TrackConfig:
    return TrackConfig(
        suite=str(raw["suite"]),
        repetitions=int(raw["repetitions"]),
        pack_ids=tuple(str(value) for value in _sequence(raw["pack_ids"], "pack_ids")),
        expected_trials=int(raw["expected_trials"]),
        expected_retrieval_cases=int(
            raw["expected_retrieval_cases"],
        ),
        max_comparison_block_retries_total=int(
            raw["max_comparison_block_retries_total"],
        ),
        expected_trials_by_pack=_count_items(
            raw["expected_trials_by_pack"],
            "expected_trials_by_pack",
        ),
        expected_comparison_blocks_by_pack=_count_items(
            raw["expected_comparison_blocks_by_pack"],
            "expected_comparison_blocks_by_pack",
        ),
        expected_retrieval_cases_by_pack=_count_items(
            raw["expected_retrieval_cases_by_pack"],
            "expected_retrieval_cases_by_pack",
        ),
    )


def _provider_trial_budgets(
    value: object,
) -> tuple[ProviderTrialBudget, ...]:
    budgets = _mapping(
        value,
        "provider_trial_budgets",
    )
    return tuple(
        ProviderTrialBudget(
            pack_id=str(pack_id),
            max_provider_calls_per_trial=int(
                _mapping(
                    raw_budget,
                    f"provider_trial_budgets.{pack_id}",
                )["max_provider_calls_per_trial"],
            ),
            max_input_tokens_per_call=int(
                _mapping(
                    raw_budget,
                    f"provider_trial_budgets.{pack_id}",
                )["max_input_tokens_per_call"],
            ),
            max_output_tokens_per_call=int(
                _mapping(
                    raw_budget,
                    f"provider_trial_budgets.{pack_id}",
                )["max_output_tokens_per_call"],
            ),
        )
        for pack_id, raw_budget in sorted(
            budgets.items(),
            key=lambda item: str(item[0]),
        )
    )


def _claim_rule(raw: object) -> ClaimRule:
    mapping = _mapping(raw, "claim rule")
    return ClaimRule(
        rule_id=str(mapping["rule_id"]),
        metric=str(mapping["metric"]),
        operator=str(mapping["operator"]),
        threshold=_number(mapping["threshold"], "claim threshold"),
        prerequisites=tuple(
            str(value)
            for value in _sequence(
                mapping.get("prerequisites", ()),
                "claim prerequisites",
            )
        ),
    )


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _sequence(value: object, field: str) -> tuple[object, ...]:
    if isinstance(value, str) or not isinstance(value, list | tuple):
        raise TypeError(f"{field} must be a sequence")
    return tuple(value)


def _count_items(
    value: object,
    field: str,
) -> tuple[tuple[str, int], ...]:
    mapping = _mapping(value, field)
    return tuple((str(key), int(count)) for key, count in mapping.items())


def _bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be a boolean")
    return value


def _number(value: object, field: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be a number")
    return value


def _require_keys(
    mapping: Mapping[str, Any],
    required: set[str],
    field: str,
) -> None:
    missing = required - set(mapping)
    if missing:
        raise KeyError(
            f"{field} is missing: {', '.join(sorted(missing))}",
        )


def _model_identity(model: str, provider_name: str) -> str:
    prefix = provider_name + "/"
    return model.removeprefix(prefix)


def _tokenizer_identity(
    model: str,
) -> tuple[str, str, str]:
    try:
        from codeflow.providers.litellm_setup import import_litellm

        litellm = import_litellm()
        token_count = litellm.token_counter(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "CodeFlowBench tokenizer preflight",
                },
            ],
        )
        if not isinstance(token_count, int) or token_count <= 0:
            raise ValueError("tokenizer returned no tokens")
        version = importlib.metadata.version("litellm")
    except Exception as exc:
        raise CampaignError(
            f"tokenizer preflight failed: {type(exc).__name__}",
        ) from exc
    identity = f"litellm.token_counter/{model}"
    digest = canonical_digest(
        {
            "identity": identity,
            "version": version,
            "probe_token_count": token_count,
        }
    )
    return identity, version, digest


def _redacted_runtime_config(
    config: object,
    codeflow_config: object,
) -> dict[str, Any]:
    return {
        "base": _redact_secrets(_model_dump(config)),
        "codeflow": _redact_secrets(_model_dump(codeflow_config)),
    }


def _model_dump(value: object) -> Any:
    method = getattr(value, "model_dump", None)
    if callable(method):
        return method(mode="json")
    return json.loads(json.dumps(value, default=str))


def _redact_secrets(value: object) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if _secret_key(name):
                redacted[name] = "<redacted>"
            elif _secret_container(name):
                redacted[name] = _redact_container(item)
            else:
                redacted[name] = _redact_secrets(item)
        return redacted
    if isinstance(value, list | tuple):
        return [_redact_secrets(item) for item in value]
    return value


def _redact_container(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): "<redacted>" for key in value}
    if isinstance(value, list | tuple):
        return ["<redacted>" for _ in value]
    return "<redacted>"


def _secret_container(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return normalized in {
        "env",
        "environment",
        "extra_headers",
        "headers",
        "http_headers",
    }


def _secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(
        marker in normalized
        for marker in (
            "api_key",
            "app_code",
            "authorization",
            "client_secret",
            "cookie",
            "secret",
            "token",
            "password",
            "credential",
        )
    )


__all__ = [
    "BudgetConfig",
    "CampaignError",
    "CampaignMode",
    "CampaignOutcome",
    "CampaignServices",
    "CampaignSuite",
    "CostEstimate",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_SUITE_PATH",
    "DeterministicGateResult",
    "ProviderPreflightRecord",
    "ProviderProbeRequest",
    "ProviderProbeResult",
    "ResolvedProvider",
    "TrackConfig",
    "default_campaign_services",
    "estimate_worst_case_cost",
    "load_campaign_suite",
    "run_campaign",
]
