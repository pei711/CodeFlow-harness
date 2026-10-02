"""Contract tests for frozen historical CodeFlowBench campaign identities."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

import benchmarks.codeflowbench.campaign as campaign
from benchmarks.codeflowbench.__main__ import _CLI_MODES
from benchmarks.codeflowbench.budget import (
    BudgetGuardedProvider,
    ProviderBudgetConfig,
    ProviderBudgetError,
    ProviderBudgetLedger,
    provider_call_budget_scope,
)
from benchmarks.codeflowbench.campaign import (
    CampaignError,
    CampaignMode,
    CampaignServices,
    DeterministicGateResult,
    ProviderProbeRequest,
    ProviderProbeResult,
    ResolvedProvider,
    _prepare_budget_guard,
    _probe_provider,
    estimate_worst_case_cost,
    load_campaign_suite,
    run_campaign,
)
from benchmarks.codeflowbench.canonical import canonical_digest
from benchmarks.codeflowbench.claims import evaluate_claim_rules
from benchmarks.codeflowbench.registry import PackRegistry
from benchmarks.codeflowbench.schema import (
    ExperimentRef,
    PackDefinition,
    PairSpec,
    RetrievalConfigurationSpec,
    RetrievalQuerySpec,
    RetrievalSuiteSpec,
    TaskSpec,
    VariantSpec,
)
from codeflow.providers.base import ErrorClassification, LLMProvider, LLMResponse, ToolCallRequest

_SUITE_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "codeflowbench" / "suites" / "agent_application_ship_1.yaml"
)


class _Pack:
    def __init__(self, definition: PackDefinition) -> None:
        self._definition = definition

    def definition(self) -> PackDefinition:
        return self._definition

    async def run_trial(self, context):
        raise AssertionError("campaign orchestration test must not execute Packs")


class _BudgetProvider(LLMProvider):
    def __init__(self, usage: dict[str, int]) -> None:
        super().__init__()
        self.usage = usage
        self.calls = 0

    async def chat(
        self,
        messages,
        tools=None,
        model=None,
        max_tokens=4096,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
    ) -> LLMResponse:
        del messages, tools, model, max_tokens, temperature
        del reasoning_effort, tool_choice
        self.calls += 1
        return LLMResponse(content="ok", usage=dict(self.usage))

    def get_default_model(self) -> str:
        return "test/budget"


class _ProbeProvider(LLMProvider):
    def __init__(self, response: LLMResponse | list[LLMResponse]) -> None:
        super().__init__()
        self.responses = response if isinstance(response, list) else [response]
        self.calls: list[dict[str, object]] = []

    async def chat(
        self,
        messages,
        tools=None,
        model=None,
        max_tokens=4096,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "reasoning_effort": reasoning_effort,
                "tool_choice": tool_choice,
            }
        )
        return self.responses[min(len(self.calls), len(self.responses)) - 1]

    def get_default_model(self) -> str:
        return "deepseek/deepseek-v4-flash"


@dataclass(frozen=True)
class _ShipReport:
    experiment_id: str
    planned_trials: int
    terminal_trials: int
    planned_retrieval_cases: int
    terminal_retrieval_cases: int
    status_counts: dict[str, int]
    retrieval_status_counts: dict[str, int]
    ship_complete: bool
    measurement_valid: bool


def _tasks(prefix: str, count: int) -> tuple[TaskSpec, ...]:
    return tuple(TaskSpec(task_id=f"{prefix}-{index:03d}") for index in range(count))


def _context_definition(*, calibration: bool) -> PackDefinition:
    suffix = "-calibration" if calibration else ""
    return PackDefinition(
        pack_id=f"context{suffix}",
        tasks=_tasks(f"context{suffix}", 4 if calibration else 8),
        variants=(
            VariantSpec("fifo", {"history_manager": "fifo_tail"}),
            VariantSpec("curator", {"history_manager": "curator"}),
        ),
        pairs=(PairSpec("history_manager", "fifo", "curator"),),
    )


def _memory_definition(*, calibration: bool) -> PackDefinition:
    suffix = "-calibration-v1" if calibration else "-v1"
    memory_queries = 10 if calibration else 80
    skill_queries = 8 if calibration else 60
    return PackDefinition(
        pack_id=f"memory-skill{suffix}",
        tasks=_tasks(f"memory-skill{suffix}", 4 if calibration else 8),
        variants=(
            VariantSpec("memory-off", {"user_memory_recall": "disabled", "skill_sources": ["local", "everos"]}),
            VariantSpec("local-only", {"user_memory_recall": "enabled", "skill_sources": ["local"]}),
            VariantSpec("fused", {"user_memory_recall": "enabled", "skill_sources": ["local", "everos"]}),
        ),
        pairs=(
            PairSpec("user_memory_recall", "memory-off", "fused"),
            PairSpec("skill_sources", "local-only", "fused"),
        ),
        retrieval_suites=(
            RetrievalSuiteSpec(
                retrieval_suite_id=f"user-memory{suffix}",
                queries=tuple(
                    RetrievalQuerySpec(
                        query_id=f"user-memory{suffix}-{index:03d}",
                        label="positive",
                    )
                    for index in range(memory_queries)
                ),
                configurations=(RetrievalConfigurationSpec("memory-on", {"enabled": True}),),
                corpus_digest="a" * 64,
                query_labels_digest="b" * 64,
            ),
            RetrievalSuiteSpec(
                retrieval_suite_id=f"skill-fusion{suffix}",
                queries=tuple(
                    RetrievalQuerySpec(
                        query_id=f"skill-fusion{suffix}-{index:03d}",
                        label="positive",
                    )
                    for index in range(skill_queries)
                ),
                configurations=tuple(
                    RetrievalConfigurationSpec(name, {"configuration": name}) for name in ("local", "everos", "fused")
                ),
                corpus_digest="c" * 64,
                query_labels_digest="d" * 64,
            ),
        ),
    )


def _tool_definition(*, calibration: bool) -> PackDefinition:
    suffix = "-calibration" if calibration else ""
    return PackDefinition(
        pack_id=f"tool-mcp{suffix}",
        tasks=_tasks(f"tool-mcp{suffix}", 4 if calibration else 8),
        variants=(
            VariantSpec("all-tools", {"tool_disclosure": "all_tools"}),
            VariantSpec("progressive", {"tool_disclosure": "progressive_disclosure"}),
        ),
        pairs=(PairSpec("tool_disclosure", "all-tools", "progressive"),),
    )


def _semantic_memory_definition(
    *,
    calibration: bool,
) -> PackDefinition:
    suffix = "-calibration-v1" if calibration else "-v1"
    return PackDefinition(
        pack_id=f"semantic-memory-effect{suffix}",
        tasks=_tasks(
            f"semantic-memory-effect{suffix}",
            2 if calibration else 8,
        ),
        variants=(
            VariantSpec(
                "user_memory_off",
                {"user_memory_recall": "disabled"},
            ),
            VariantSpec(
                "user_memory_on",
                {"user_memory_recall": "enabled"},
            ),
        ),
        pairs=(
            PairSpec(
                "user_memory_recall",
                "user_memory_off",
                "user_memory_on",
            ),
        ),
        identity={"claim_reducer": "memory_effect_v1"},
    )


def _registry(
    mode: CampaignMode,
    *,
    overlap: bool = False,
    redistribute_trials: bool = False,
    comparison_block_drift: bool = False,
) -> PackRegistry:
    calibration = mode is CampaignMode.CALIBRATION
    registry = PackRegistry()
    definitions = [
        _context_definition(calibration=calibration),
        _memory_definition(calibration=calibration),
        _semantic_memory_definition(calibration=calibration),
        _tool_definition(calibration=calibration),
    ]
    if overlap and calibration:
        formal = _context_definition(calibration=False)
        definitions[0] = replace(
            definitions[0],
            tasks=(formal.tasks[0], *definitions[0].tasks[1:]),
        )
    if redistribute_trials and not calibration:
        definitions[0] = replace(
            definitions[0],
            tasks=definitions[0].tasks[:-3],
        )
        definitions[1] = replace(
            definitions[1],
            tasks=(
                *definitions[1].tasks,
                TaskSpec(task_id="memory-skill-v1-extra-000"),
                TaskSpec(task_id="memory-skill-v1-extra-001"),
            ),
        )
    if comparison_block_drift and not calibration:
        definitions[0] = replace(
            definitions[0],
            tasks=definitions[0].tasks[:4],
            variants=(
                *definitions[0].variants,
                VariantSpec(
                    "context-extra-a",
                    {"history_manager": "extra_a"},
                ),
                VariantSpec(
                    "context-extra-b",
                    {"history_manager": "extra_b"},
                ),
            ),
        )
    for definition in definitions:
        registry.register(_Pack(definition))
    return registry


def _probe_result() -> ProviderProbeResult:
    return ProviderProbeResult(
        provider_name="deepseek",
        requested_model="deepseek-v4-flash",
        resolved_model="deepseek-v4-flash",
        tool_calling_supported=True,
        usage_fields=("prompt_tokens", "completion_tokens", "total_tokens"),
        seed_supported=False,
        tokenizer_identity="litellm-token-counter/deepseek-v4-flash",
        tokenizer_version="1",
        tokenizer_digest="e" * 64,
        pricing_source="https://api-docs.deepseek.com/quick_start/pricing/",
        fallback_used=False,
    )


def _probe_request() -> ProviderProbeRequest:
    suite = load_campaign_suite(_SUITE_PATH)
    return ProviderProbeRequest(
        provider_name=suite.provider.name,
        model=suite.provider.model,
        codeflow_commit="a" * 40,
        worktree_clean=True,
        suite_digest=canonical_digest(suite.canonical_payload()),
        claim_rules_digest=canonical_digest(suite.claim_rules),
        estimated_cost=estimate_worst_case_cost(
            suite,
            modes=(CampaignMode.CALIBRATION, CampaignMode.FORMAL),
        ),
        max_provider_calls_per_trial=suite.budget.max_provider_calls_per_trial,
        max_input_tokens_per_call=suite.budget.max_input_tokens_per_call,
        max_output_tokens_per_call=suite.budget.max_output_tokens_per_call,
    )


@dataclass(frozen=True)
class _Calls:
    entries: list[str]
    probe_requests: list[object]
    specs: list[object]


def _services(
    calls: _Calls,
    *,
    clean: bool = True,
    probe_result: ProviderProbeResult | None = None,
    overlap: bool = False,
    redistribute_trials: bool = False,
    comparison_block_drift: bool = False,
    measurement_valid: bool = True,
) -> CampaignServices:
    async def deterministic(output_root: Path) -> DeterministicGateResult:
        calls.entries.append("deterministic")
        return DeterministicGateResult(
            passed=True,
            details={
                "r0": "passed",
                "r1": "passed",
                "mcp": "passed",
                "report_rebuild": "passed",
            },
        )

    def clean_worktree() -> bool:
        calls.entries.append("clean")
        return clean

    def resolve_commit() -> str:
        calls.entries.append("commit")
        return "a" * 40

    def resolve() -> ResolvedProvider:
        calls.entries.append("resolve")
        return ResolvedProvider(
            provider_name="deepseek",
            model="deepseek-v4-flash",
            provider=object(),
            runtime_config_digest="f" * 64,
        )

    def build_registry(mode: CampaignMode, resolved: ResolvedProvider) -> PackRegistry:
        del resolved
        calls.entries.append(f"registry:{mode.value}")
        return _registry(
            mode,
            overlap=overlap,
            redistribute_trials=redistribute_trials,
            comparison_block_drift=comparison_block_drift,
        )

    async def probe(request, resolved: ResolvedProvider) -> ProviderProbeResult:
        del resolved
        calls.entries.append("probe")
        calls.probe_requests.append(request)
        return probe_result or _probe_result()

    async def execute(spec, registry) -> ExperimentRef:
        del registry
        calls.entries.append(f"execute:{spec.suite}")
        calls.specs.append(spec)
        return ExperimentRef(
            experiment_id=canonical_digest(spec.canonical_payload()),
            root=spec.output_root / canonical_digest(spec.canonical_payload()),
        )

    def report(ref: ExperimentRef) -> _ShipReport:
        calls.entries.append("report")
        spec = calls.specs[-1]
        expected_trials = 64 if "calibration" in spec.suite else 216
        expected_retrieval = 34 if "calibration" in spec.suite else 260
        return _ShipReport(
            experiment_id=ref.experiment_id,
            planned_trials=expected_trials,
            terminal_trials=expected_trials,
            planned_retrieval_cases=expected_retrieval,
            terminal_retrieval_cases=expected_retrieval,
            status_counts={"passed": expected_trials},
            retrieval_status_counts={"measurable": expected_retrieval},
            ship_complete=True,
            measurement_valid=measurement_valid,
        )

    return CampaignServices(
        run_deterministic=deterministic,
        worktree_is_clean=clean_worktree,
        resolve_codeflow_commit=resolve_commit,
        resolve_provider=resolve,
        build_registry=build_registry,
        probe_provider=probe,
        execute_experiment=execute,
        rebuild_report=report,
    )


def test_suite_freezes_exact_denominators_retries_model_and_budget() -> None:
    suite = load_campaign_suite(_SUITE_PATH)

    assert suite.provider.name == "deepseek"
    assert suite.provider.model == "deepseek-v4-flash"
    assert suite.provider.allow_fallback is False
    assert suite.execution.provider_call_max_attempts == 2
    assert suite.execution.max_comparison_block_attempts == 2
    assert suite.execution.max_retrieval_query_block_attempts == 2
    assert suite.budget.warning_cny == 80
    assert suite.budget.hard_cap_cny == 100
    assert suite.budget.max_provider_calls_per_trial == 8
    assert suite.budget.max_input_tokens_per_call == 42_000
    assert suite.budget.max_additional_provider_attempts == 64
    assert {
        budget.pack_id: (
            budget.max_provider_calls_per_trial,
            budget.max_input_tokens_per_call,
            budget.max_output_tokens_per_call,
        )
        for budget in suite.budget.provider_trial_budgets
    } == {
        "context-calibration": (8, 42_000, 1_200),
        "context": (8, 42_000, 1_200),
        "memory-skill-calibration-v1": (4, 15_000, 1_500),
        "memory-skill-v1": (4, 15_000, 1_500),
        "semantic-memory-effect-calibration-v1": (
            8,
            15_000,
            1_500,
        ),
        "semantic-memory-effect-v1": (8, 15_000, 1_500),
        "tool-mcp-calibration": (4, 40_000, 1_500),
        "tool-mcp": (4, 40_000, 1_500),
    }
    assert suite.calibration.expected_trials == 64
    assert suite.calibration.expected_retrieval_cases == 34
    assert suite.calibration.max_comparison_block_retries_total == 1
    assert dict(suite.calibration.expected_trials_by_pack) == {
        "context-calibration": 16,
        "memory-skill-calibration-v1": 24,
        "semantic-memory-effect-calibration-v1": 8,
        "tool-mcp-calibration": 16,
    }
    assert dict(suite.calibration.expected_comparison_blocks_by_pack) == {
        "context-calibration": 8,
        "memory-skill-calibration-v1": 8,
        "semantic-memory-effect-calibration-v1": 4,
        "tool-mcp-calibration": 8,
    }
    assert dict(suite.calibration.expected_retrieval_cases_by_pack) == {
        "context-calibration": 0,
        "memory-skill-calibration-v1": 34,
        "semantic-memory-effect-calibration-v1": 0,
        "tool-mcp-calibration": 0,
    }
    assert suite.formal.expected_trials == 216
    assert suite.formal.expected_retrieval_cases == 260
    assert suite.formal.max_comparison_block_retries_total == 4
    assert dict(suite.formal.expected_trials_by_pack) == {
        "context": 48,
        "memory-skill-v1": 72,
        "semantic-memory-effect-v1": 48,
        "tool-mcp": 48,
    }
    assert dict(suite.formal.expected_comparison_blocks_by_pack) == {
        "context": 24,
        "memory-skill-v1": 24,
        "semantic-memory-effect-v1": 24,
        "tool-mcp": 24,
    }
    assert dict(suite.formal.expected_retrieval_cases_by_pack) == {
        "context": 0,
        "memory-skill-v1": 260,
        "semantic-memory-effect-v1": 0,
        "tool-mcp": 0,
    }
    assert {rule.metric for rule in suite.claim_rules} == {
        "context.trial_total_input_token_reduction_percent",
        "context.tasks_with_lower_trial_total_input",
        "semantic_memory_e2e.net_verifier_gains",
        "semantic_memory_e2e.positive_tasks",
        "skill_e2e.net_verifier_gains",
        "skill_e2e.positive_tasks",
        "memory_retrieval.final_injection_recall_at_5",
        "memory_retrieval.irrelevant_injection_rate",
        "memory_retrieval.hard_negative_injection_rate",
        "memory_retrieval.stale_injection_count",
        "memory_retrieval.cross_workspace_leakage_count",
        "skill_fusion.improvement_over_best_single_source",
        "skill_fusion.hard_negative_injection_rate",
        "skill_fusion.cross_workspace_leakage_count",
        "tool_mcp.equal_task_macro_schema_token_reduction_percent",
        "tool_mcp.tasks_with_lower_schema_tokens",
    }
    assert all(rule.prerequisites for rule in suite.claim_rules)
    threshold_coupled_prerequisites = {
        "context.positive_claim_eligible",
        "semantic_memory_e2e.real_agent_task_effect_claim_eligible",
        "skill_e2e.real_agent_task_effect_claim_eligible",
        "memory_retrieval.deterministic_contract_claim_eligible",
        "skill_fusion.deterministic_contract_claim_eligible",
        "tool_mcp.positive_claim_eligible",
    }
    assert not threshold_coupled_prerequisites.intersection(
        prerequisite for rule in suite.claim_rules for prerequisite in rule.prerequisites
    )
    memory_retrieval_rules = [rule for rule in suite.claim_rules if rule.metric.startswith("memory_retrieval.")]
    skill_fusion_rules = [rule for rule in suite.claim_rules if rule.metric.startswith("skill_fusion.")]
    assert memory_retrieval_rules
    assert skill_fusion_rules
    assert all("memory_retrieval.real_semantic_evidence_valid" in rule.prerequisites for rule in memory_retrieval_rules)
    assert all("skill_fusion.real_semantic_evidence_valid" in rule.prerequisites for rule in skill_fusion_rules)

    estimate = estimate_worst_case_cost(
        suite,
        modes=(CampaignMode.CALIBRATION, CampaignMode.FORMAL),
    )
    assert 0 < estimate.estimated_cny <= 100
    assert estimate.logical_provider_calls == 1_680
    assert estimate.additional_provider_attempts == 64
    assert estimate.provider_request_attempts == 1_744
    assert estimate.maximum_input_tokens == 50_272_000
    assert estimate.maximum_output_tokens == 2_419_200
    assert estimate.estimated_cny == pytest.approx(62.86592)


def test_suite_negative_result_remains_ship_complete_but_ineligible() -> None:
    suite = load_campaign_suite(_SUITE_PATH)
    metrics = {prerequisite: True for rule in suite.claim_rules for prerequisite in rule.prerequisites}
    metrics.update({rule.metric: rule.threshold for rule in suite.claim_rules})
    metrics["context.trial_total_input_token_reduction_percent"] = 14.9

    result = evaluate_claim_rules(
        suite.claim_rules,
        metrics=metrics,
        ship_complete=True,
        measurement_valid=True,
    )

    assert result.ship_complete is True
    assert result.measurement_valid is True
    assert result.positive_claim_eligible is False
    failed = [rule for rule in result.rules if not rule.passed]
    assert [rule.rule_id for rule in failed] == [
        "context-token-reduction",
    ]
    assert failed[0].reason == "threshold_not_met"


def test_suite_thresholds_are_authoritative_for_negative_results() -> None:
    suite = load_campaign_suite(_SUITE_PATH)
    structural_metrics = {prerequisite: True for rule in suite.claim_rules for prerequisite in rule.prerequisites}

    for selected_rule in suite.claim_rules:
        metrics = {
            **structural_metrics,
            **{rule.metric: rule.threshold for rule in suite.claim_rules},
        }
        metrics[selected_rule.metric] = (
            selected_rule.threshold - 0.1
            if selected_rule.operator in {"ge", "gt", "eq"}
            else selected_rule.threshold + 0.1
        )

        result = evaluate_claim_rules(
            suite.claim_rules,
            metrics=metrics,
            ship_complete=True,
            measurement_valid=True,
        )

        failed = [rule for rule in result.rules if not rule.passed]
        assert [rule.rule_id for rule in failed] == [
            selected_rule.rule_id,
        ]
        assert failed[0].reason == "threshold_not_met"


@pytest.mark.asyncio
async def test_provider_probe_reports_provider_failure_before_capability_checks() -> None:
    provider = _ProbeProvider(
        LLMResponse(
            content="redacted provider error",
            finish_reason="error",
            error_classification=ErrorClassification("invalid_request"),
        )
    )
    resolved = ResolvedProvider(
        provider_name="deepseek",
        model="deepseek-v4-flash",
        provider=provider,
        runtime_config_digest="config",
        configured_model="deepseek/deepseek-v4-flash",
    )

    with pytest.raises(CampaignError, match=r"Provider preflight call failed \(invalid_request\)"):
        await _probe_provider(_probe_request(), resolved)

    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_provider_probe_mirrors_runtime_tool_choice_and_requires_exact_call() -> None:
    provider = _ProbeProvider(
        LLMResponse(
            content=None,
            model="deepseek-v4-flash",
            tool_calls=[
                ToolCallRequest(
                    id="probe-call",
                    name="codeflowbench_preflight_echo",
                    arguments={"value": "ready"},
                )
            ],
            finish_reason="tool_calls",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
            },
        )
    )
    resolved = ResolvedProvider(
        provider_name="deepseek",
        model="deepseek-v4-flash",
        provider=provider,
        runtime_config_digest="config",
        configured_model="deepseek/deepseek-v4-flash",
    )

    request = _probe_request()
    result = await _probe_provider(request, resolved)

    assert result.tool_calling_supported is True
    assert provider.calls[0]["tool_choice"] is None
    assert provider.calls[0]["max_tokens"] == request.max_output_tokens_per_call


@pytest.mark.asyncio
async def test_provider_probe_retries_one_text_response_before_proving_tool_calling() -> None:
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
    }
    provider = _ProbeProvider(
        [
            LLMResponse(
                content="ready",
                model="deepseek-v4-flash",
                usage=usage,
            ),
            LLMResponse(
                content=None,
                model="deepseek-v4-flash",
                tool_calls=[
                    ToolCallRequest(
                        id="probe-call",
                        name="codeflowbench_preflight_echo",
                        arguments={"value": "ready"},
                    )
                ],
                finish_reason="tool_calls",
                usage=usage,
            ),
        ]
    )
    resolved = ResolvedProvider(
        provider_name="deepseek",
        model="deepseek-v4-flash",
        provider=provider,
        runtime_config_digest="config",
        configured_model="deepseek/deepseek-v4-flash",
    )

    result = await _probe_provider(_probe_request(), resolved)

    assert result.tool_calling_supported is True
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_provider_probe_retries_malformed_tool_arguments() -> None:
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
    }
    provider = _ProbeProvider(
        [
            LLMResponse(
                content=None,
                model="deepseek-v4-flash",
                tool_calls=[
                    ToolCallRequest(
                        id="malformed-probe-call",
                        name="codeflowbench_preflight_echo",
                        arguments=[{"value": "ready"}],
                    )
                ],
                finish_reason="tool_calls",
                usage=usage,
            ),
            LLMResponse(
                content=None,
                model="deepseek-v4-flash",
                tool_calls=[
                    ToolCallRequest(
                        id="probe-call",
                        name="codeflowbench_preflight_echo",
                        arguments={"value": "ready"},
                    )
                ],
                finish_reason="tool_calls",
                usage=usage,
            ),
        ]
    )
    resolved = ResolvedProvider(
        provider_name="deepseek",
        model="deepseek-v4-flash",
        provider=provider,
        runtime_config_digest="config",
        configured_model="deepseek/deepseek-v4-flash",
    )

    result = await _probe_provider(_probe_request(), resolved)

    assert result.tool_calling_supported is True
    assert len(provider.calls) == 2


def test_provider_probe_cache_reuses_exact_approval_generation(
    tmp_path: Path,
) -> None:
    ledger = ProviderBudgetLedger(
        tmp_path / "provider-budget.jsonl",
        replace(
            _budget_ledger(tmp_path).config,
            approval_digest="a" * 64,
        ),
    )
    resolved = ResolvedProvider(
        provider_name="deepseek",
        model="deepseek-v4-flash",
        provider=object(),
        runtime_config_digest="f" * 64,
        budget_ledger=ledger,
    )
    request = _probe_request()
    expected = _probe_result()

    campaign._persist_provider_probe(request, resolved, expected)

    assert (
        campaign._read_cached_provider_probe(
            request,
            resolved,
        )
        == expected
    )
    changed = replace(
        resolved,
        runtime_config_digest="e" * 64,
    )
    with pytest.raises(
        CampaignError,
        match="cached Provider preflight is invalid",
    ):
        campaign._read_cached_provider_probe(request, changed)


@pytest.mark.asyncio
async def test_provider_probe_requires_actual_model_identity() -> None:
    provider = _ProbeProvider(
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="probe-call",
                    name="codeflowbench_preflight_echo",
                    arguments={"value": "ready"},
                )
            ],
            finish_reason="tool_calls",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
            },
        )
    )
    resolved = ResolvedProvider(
        provider_name="deepseek",
        model="deepseek-v4-flash",
        provider=provider,
        runtime_config_digest="config",
        configured_model="deepseek/deepseek-v4-flash",
    )

    with pytest.raises(
        CampaignError,
        match="actual model identity",
    ):
        await _probe_provider(_probe_request(), resolved)


@pytest.mark.asyncio
async def test_provider_probe_marks_actual_model_drift_as_fallback() -> None:
    provider = _ProbeProvider(
        LLMResponse(
            content=None,
            model="deepseek-v4-pro",
            tool_calls=[
                ToolCallRequest(
                    id="probe-call",
                    name="codeflowbench_preflight_echo",
                    arguments={"value": "ready"},
                )
            ],
            finish_reason="tool_calls",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
            },
        )
    )
    resolved = ResolvedProvider(
        provider_name="deepseek",
        model="deepseek-v4-flash",
        provider=provider,
        runtime_config_digest="config",
        configured_model="deepseek/deepseek-v4-flash",
    )

    result = await _probe_provider(_probe_request(), resolved)

    assert result.resolved_model == "deepseek-v4-pro"
    assert result.fallback_used is True


@pytest.mark.asyncio
async def test_smoke_is_credential_free_and_never_resolves_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product_home = tmp_path / "codeflow-home"
    monkeypatch.setenv("CODEFLOW_HOME", str(product_home))
    calls = _Calls([], [], [])

    outcome = await run_campaign(
        CampaignMode.SMOKE,
        suite_path=_SUITE_PATH,
        output_root=tmp_path,
        services=_services(calls),
    )

    assert outcome.mode is CampaignMode.SMOKE
    assert outcome.preflight is None
    assert calls.entries == ["deterministic"]
    assert not (product_home / "evidence" / "codeflowbench" / "campaign-locks").exists()


@pytest.mark.asyncio
async def test_paid_campaign_rejects_concurrent_writer_before_reservation_or_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEFLOW_HOME", str(tmp_path / "codeflow-home"))
    first_calls = _Calls([], [], [])
    first_services = _services(first_calls)
    first_probe_started = asyncio.Event()
    release_first_probe = asyncio.Event()
    original_first_probe = first_services.probe_provider

    async def blocking_probe(request, resolved):
        first_probe_started.set()
        await release_first_probe.wait()
        return await original_first_probe(request, resolved)

    first_task = asyncio.create_task(
        run_campaign(
            CampaignMode.CALIBRATION,
            suite_path=_SUITE_PATH,
            output_root=tmp_path / "first",
            services=replace(
                first_services,
                probe_provider=blocking_probe,
            ),
        ),
    )
    await first_probe_started.wait()

    second_calls = _Calls([], [], [])
    reservation_calls = 0

    def reserve_budget(
        resolved,
        suite,
        mode,
        estimate,
        output_root,
        suite_digest,
        codeflow_commit,
    ):
        nonlocal reservation_calls
        del suite, mode, estimate, output_root, suite_digest, codeflow_commit
        reservation_calls += 1
        return resolved

    try:
        with pytest.raises(CampaignError, match="active paid campaign writer"):
            await run_campaign(
                CampaignMode.CALIBRATION,
                suite_path=_SUITE_PATH,
                output_root=tmp_path / "second",
                services=replace(
                    _services(second_calls),
                    prepare_budget_guard=reserve_budget,
                ),
            )
    finally:
        release_first_probe.set()
        await first_task

    assert reservation_calls == 0
    assert second_calls.probe_requests == []


@pytest.mark.asyncio
async def test_paid_campaign_rejects_nested_writer_before_campaign_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEFLOW_HOME", str(tmp_path / "codeflow-home"))
    suite = load_campaign_suite(_SUITE_PATH)
    suite_digest = canonical_digest(suite.canonical_payload())
    calls = _Calls([], [], [])

    with campaign._exclusive_paid_campaign_lock(suite_digest):
        with pytest.raises(CampaignError, match="active paid campaign writer"):
            await run_campaign(
                CampaignMode.CALIBRATION,
                suite=suite,
                output_root=tmp_path / "nested",
                services=_services(calls),
            )

    assert calls.entries == []


@pytest.mark.asyncio
async def test_paid_campaign_holds_writer_lock_through_outcome_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEFLOW_HOME", str(tmp_path / "codeflow-home"))
    suite = load_campaign_suite(_SUITE_PATH)
    suite_digest = canonical_digest(suite.canonical_payload())
    original_write_summary = campaign.ArtifactStore.write_summary
    lock_observed = False

    def write_summary(store, path, value):
        nonlocal lock_observed
        with pytest.raises(CampaignError, match="active paid campaign writer"):
            with campaign._exclusive_paid_campaign_lock(suite_digest):
                raise AssertionError("paid campaign lock was released before persistence")
        lock_observed = True
        original_write_summary(store, path, value)

    monkeypatch.setattr(
        campaign.ArtifactStore,
        "write_summary",
        write_summary,
    )

    await run_campaign(
        CampaignMode.CALIBRATION,
        suite=suite,
        output_root=tmp_path / "paid",
        services=_services(_Calls([], [], [])),
    )

    assert lock_observed is True


@pytest.mark.asyncio
async def test_default_smoke_preserves_runtime_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmarks.codeflowbench.campaign import _run_deterministic_gate

    monkeypatch.setattr(campaign, "_resolve_codeflow_commit", lambda: "a" * 40)
    monkeypatch.setattr(campaign, "_worktree_is_clean", lambda: True)
    result = await _run_deterministic_gate(tmp_path)

    assert result.passed is True
    assert result.details["r0"]["metrics"]["accepted_requests"] == 2_000
    assert result.details["r0"]["metrics"]["lost_requests"] == 0
    assert result.details["r0"]["metrics"]["dispatch_overhead_ms"]["p95"] is not None
    assert result.details["r1"]["metrics"]["turns"] == 100
    assert result.details["r1"]["metrics"]["unresolved_handles"] == 0
    assert result.details["mcp"]["metrics"]["transport"] == "stdio"
    assert result.evidence_path is not None
    assert result.evidence_digest is not None
    evidence_path = Path(result.evidence_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence_digest = evidence.pop("evidence_digest")

    assert evidence_digest == result.evidence_digest
    assert evidence_digest == canonical_digest(evidence)
    assert evidence["evidence_scope"] == "deterministic_runtime_only"
    assert evidence["environment"]["dependency_lock"]["file"] == "uv.lock"
    assert len(evidence["environment"]["dependency_lock"]["sha256"]) == 64
    assert evidence["environment"]["execution"] == {
        "configured_backend": "none",
        "sandbox_identity": "direct_host",
    }
    assert evidence["cv_metrics"]["runtime.r0.accepted_requests"] == 2_000
    assert evidence["cv_metrics"]["runtime.r0.lost_requests"] == 0
    assert evidence["cv_metrics"]["runtime.r1.turns"] == 100


@pytest.mark.asyncio
async def test_runtime_evidence_rejects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(campaign, "_resolve_codeflow_commit", lambda: "a" * 40)
    monkeypatch.setattr(campaign, "_worktree_is_clean", lambda: True)
    result = await campaign._run_deterministic_gate(tmp_path)
    assert result.evidence_path is not None
    assert result.evidence_digest is not None
    evidence_path = Path(result.evidence_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["environment"]["execution"]["configured_backend"] = "boxlite"
    tampered_path = tmp_path / "tampered-runtime-evidence.json"
    tampered_path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(
        campaign.ArtifactError,
        match="Runtime evidence digest is invalid",
    ):
        campaign._read_runtime_evidence(
            tampered_path,
            expected_digest=result.evidence_digest,
        )


@pytest.mark.asyncio
async def test_runtime_evidence_rejects_environment_drift_during_tracks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(campaign, "_resolve_codeflow_commit", lambda: "a" * 40)
    monkeypatch.setattr(campaign, "_worktree_is_clean", lambda: True)
    identities = iter(
        (
            {"dependency_lock": {"sha256": "a" * 64}},
            {"dependency_lock": {"sha256": "b" * 64}},
        ),
    )
    monkeypatch.setattr(
        campaign,
        "capture_environment_identity",
        lambda *args, **kwargs: next(identities),
    )

    with pytest.raises(
        CampaignError,
        match="Execution environment changed during deterministic Runtime tracks",
    ):
        await campaign._run_deterministic_gate(tmp_path)


@pytest.mark.asyncio
async def test_runtime_evidence_withholds_cv_metrics_for_dirty_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(campaign, "_resolve_codeflow_commit", lambda: "a" * 40)
    monkeypatch.setattr(campaign, "_worktree_is_clean", lambda: False)

    result = await campaign._run_deterministic_gate(tmp_path)

    assert result.evidence_path is not None
    assert result.evidence_digest is not None
    evidence = campaign._read_runtime_evidence(
        Path(result.evidence_path),
        expected_digest=result.evidence_digest,
    )
    assert evidence["claim_eligible"] is False
    assert evidence["cv_metrics"] == {}


@pytest.mark.asyncio
async def test_runtime_evidence_rejects_git_drift_during_tracks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commits = iter(("a" * 40, "b" * 40))
    monkeypatch.setattr(campaign, "_resolve_codeflow_commit", lambda: next(commits))
    monkeypatch.setattr(campaign, "_worktree_is_clean", lambda: True)

    with pytest.raises(
        CampaignError,
        match="Git state changed during deterministic Runtime tracks",
    ):
        await campaign._run_deterministic_gate(tmp_path)


@pytest.mark.asyncio
async def test_runtime_evidence_rejects_recomputed_digest_for_old_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(campaign, "_resolve_codeflow_commit", lambda: "a" * 40)
    monkeypatch.setattr(campaign, "_worktree_is_clean", lambda: True)
    result = await campaign._run_deterministic_gate(tmp_path)
    assert result.evidence_path is not None
    assert result.evidence_digest is not None
    evidence = json.loads(Path(result.evidence_path).read_text(encoding="utf-8"))
    evidence["cv_metrics"]["runtime.r0.accepted_requests"] = 1_999
    payload = dict(evidence)
    payload.pop("evidence_digest")
    recomputed_digest = canonical_digest(payload)
    evidence["evidence_digest"] = recomputed_digest
    recomputed_path = tmp_path / f"cv-metrics-runtime.{result.evidence_digest}.json"
    recomputed_path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(
        campaign.ArtifactError,
        match="Runtime evidence path does not match the expected digest",
    ):
        campaign._read_runtime_evidence(
            recomputed_path,
            expected_digest=recomputed_digest,
        )


@pytest.mark.asyncio
async def test_runtime_evidence_rejects_recomputed_invalid_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(campaign, "_resolve_codeflow_commit", lambda: "a" * 40)
    monkeypatch.setattr(campaign, "_worktree_is_clean", lambda: True)
    result = await campaign._run_deterministic_gate(tmp_path)
    assert result.evidence_path is not None
    evidence = json.loads(Path(result.evidence_path).read_text(encoding="utf-8"))
    evidence["environment"]["execution"]["sandbox_identity"] = "direct_host"
    evidence["environment"]["execution"]["configured_backend"] = "boxlite"
    payload = dict(evidence)
    payload.pop("evidence_digest")
    recomputed_digest = canonical_digest(payload)
    evidence["evidence_digest"] = recomputed_digest
    recomputed_path = tmp_path / f"cv-metrics-runtime.{recomputed_digest}.json"
    recomputed_path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(
        campaign.ArtifactError,
        match="Runtime evidence environment identity is invalid",
    ):
        campaign._read_runtime_evidence(
            recomputed_path,
            expected_digest=recomputed_digest,
        )


@pytest.mark.asyncio
async def test_paid_campaign_rejects_environment_drift_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _Calls([], [], [])
    identities = iter(
        (
            {
                "schema": "codeflow.codeflowbench.environment.v1",
                "dependency_lock": {"sha256": "a" * 64},
            },
            {
                "schema": "codeflow.codeflowbench.environment.v1",
                "dependency_lock": {"sha256": "b" * 64},
            },
        ),
    )
    monkeypatch.setattr(
        campaign,
        "_resolved_environment_identity",
        lambda resolved: next(identities),
    )

    with pytest.raises(
        CampaignError,
        match="Execution environment changed during paid campaign",
    ):
        await run_campaign(
            CampaignMode.CALIBRATION,
            suite_path=_SUITE_PATH,
            output_root=tmp_path,
            services=_services(calls),
        )

    assert not any(entry.startswith("execute:") for entry in calls.entries)


@pytest.mark.asyncio
async def test_ship_runs_deterministic_then_one_preflight_then_calibration_and_formal(
    tmp_path: Path,
) -> None:
    calls = _Calls([], [], [])

    outcome = await run_campaign(
        CampaignMode.SHIP,
        suite_path=_SUITE_PATH,
        output_root=tmp_path,
        services=_services(calls),
    )

    assert calls.entries == [
        "deterministic",
        "clean",
        "commit",
        "resolve",
        "registry:calibration",
        "registry:formal",
        "probe",
        "clean",
        "commit",
        "execute:agent-application-ship-1-calibration",
        "report",
        "clean",
        "commit",
        "clean",
        "commit",
        "execute:agent-application-ship-1-formal",
        "report",
        "clean",
        "commit",
        "clean",
        "commit",
    ]
    assert outcome.preflight is not None
    assert outcome.preflight.estimated_worst_case_cny <= 100
    assert len(outcome.experiments) == 2
    assert len(calls.probe_requests) == 1
    request = calls.probe_requests[0]
    assert request.provider_name == "deepseek"
    assert request.model == "deepseek-v4-flash"
    assert request.claim_rules_digest
    assert request.suite_digest
    assert request.codeflow_commit == "a" * 40
    assert request.worktree_clean is True
    assert request.estimated_cost.provider_request_attempts == 1_744
    assert all(spec.execution.provider_call_max_attempts == 2 for spec in calls.specs)
    assert {
        spec.suite: {
            budget.pack_id: (
                budget.max_provider_calls_per_trial,
                budget.max_input_tokens_per_call,
                budget.max_output_tokens_per_call,
            )
            for budget in spec.execution.provider_trial_budgets
        }
        for spec in calls.specs
    } == {
        "agent-application-ship-1-calibration": {
            "context-calibration": (8, 42_000, 1_200),
            "memory-skill-calibration-v1": (4, 15_000, 1_500),
            "semantic-memory-effect-calibration-v1": (
                8,
                15_000,
                1_500,
            ),
            "tool-mcp-calibration": (4, 40_000, 1_500),
        },
        "agent-application-ship-1-formal": {
            "context": (8, 42_000, 1_200),
            "memory-skill-v1": (4, 15_000, 1_500),
            "semantic-memory-effect-v1": (8, 15_000, 1_500),
            "tool-mcp": (4, 40_000, 1_500),
        },
    }
    assert all(spec.identity["claim_rules_digest"] == request.claim_rules_digest for spec in calls.specs)
    assert all(spec.identity["codeflow_commit"] == "a" * 40 for spec in calls.specs)
    assert all(spec.identity["worktree_clean"] is True for spec in calls.specs)
    assert all(spec.identity["clean_worktree_required"] is True for spec in calls.specs)
    assert all(spec.identity["environment"]["schema"] for spec in calls.specs)
    assert all(spec.identity["environment"]["dependency_lock"]["file"] == "uv.lock" for spec in calls.specs)
    assert all(spec.identity["environment"]["execution"]["configured_backend"] == "unresolved" for spec in calls.specs)
    assert outcome.preflight.codeflow_commit == "a" * 40
    assert outcome.preflight.worktree_clean is True
    assert outcome.campaign_artifact_path is not None
    assert Path(outcome.campaign_artifact_path).is_file()


@pytest.mark.asyncio
async def test_default_ship_services_parse_config_and_assemble_real_pack_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    product_home = tmp_path / "codeflow-home"
    product_home.mkdir()
    (product_home / "config.json").write_text(
        json.dumps(
            {
                "agents": {
                    "defaults": {
                        "model": "deepseek/deepseek-v4-flash",
                        "provider": "deepseek",
                    }
                },
                "providers": {
                    "deepseek": {
                        "apiKey": "offline-test-key",
                    }
                },
                "memory": {
                    "backend": None,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEFLOW_HOME", str(product_home))
    provider = _ProbeProvider(
        LLMResponse(
            content=None,
            model="deepseek-v4-flash",
            tool_calls=[
                ToolCallRequest(
                    id="default-ship-probe",
                    name="codeflowbench_preflight_echo",
                    arguments={"value": "ready"},
                )
            ],
            finish_reason="tool_calls",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
            },
        )
    )
    monkeypatch.setattr(
        "codeflow.cli._helpers.make_provider",
        lambda config: provider,
    )
    monkeypatch.setattr(
        ("benchmarks.codeflowbench.packs.memory_skill.semantic_effect._configured_embedding_environment"),
        lambda: (
            {},
            "offline/test-embedding",
            {"model": "offline/test-embedding"},
        ),
    )

    async def deterministic(output_root: Path) -> DeterministicGateResult:
        del output_root
        return DeterministicGateResult(
            passed=True,
            details={"offline": {"passed": True}},
        )

    specs_by_id = {}
    assembled = {}

    async def execute(
        spec,
        registry: PackRegistry,
    ) -> ExperimentRef:
        experiment_id = canonical_digest(spec.canonical_payload())
        specs_by_id[experiment_id] = spec
        packs = registry.resolve(spec.pack_ids)
        assembled[spec.suite] = tuple(
            (
                pack.definition().pack_id,
                type(pack).__name__,
                type(pack._runner).__name__,
            )
            for pack in packs
        )
        return ExperimentRef(
            experiment_id=experiment_id,
            root=spec.output_root / experiment_id,
        )

    def report(ref: ExperimentRef) -> _ShipReport:
        spec = specs_by_id[ref.experiment_id]
        track = (
            load_campaign_suite(_SUITE_PATH).calibration
            if "calibration" in spec.suite
            else load_campaign_suite(_SUITE_PATH).formal
        )
        return _ShipReport(
            experiment_id=ref.experiment_id,
            planned_trials=track.expected_trials,
            terminal_trials=track.expected_trials,
            planned_retrieval_cases=track.expected_retrieval_cases,
            terminal_retrieval_cases=track.expected_retrieval_cases,
            status_counts={"passed": track.expected_trials},
            retrieval_status_counts={
                "measurable": track.expected_retrieval_cases,
            },
            ship_complete=True,
            measurement_valid=True,
        )

    monkeypatch.setattr(campaign, "_run_deterministic_gate", deterministic)
    monkeypatch.setattr(campaign, "_worktree_is_clean", lambda: True)
    monkeypatch.setattr(campaign, "_resolve_codeflow_commit", lambda: "a" * 40)
    monkeypatch.setattr(campaign, "_execute_experiment", execute)
    monkeypatch.setattr(campaign, "rebuild_experiment_report", report)

    outcome = await run_campaign(
        CampaignMode.SHIP,
        suite_path=_SUITE_PATH,
        output_root=tmp_path / "evidence",
    )

    assert outcome.preflight is not None
    assert outcome.preflight.runtime_config_digest
    assert provider.calls[0]["model"] == "deepseek/deepseek-v4-flash"
    assert assembled == {
        "agent-application-ship-1-calibration": (
            (
                "context-calibration",
                "ContextPack",
                "RuntimeContextTrialRunner",
            ),
            (
                "memory-skill-calibration-v1",
                "MemorySkillPack",
                "RuntimeCrossProcessRunner",
            ),
            (
                "semantic-memory-effect-calibration-v1",
                "SemanticMemoryEffectPack",
                "ProductionSemanticMemoryEffectRunner",
            ),
            (
                "tool-mcp-calibration",
                "ToolMCPPack",
                "MCPRuntimeTrialRunner",
            ),
        ),
        "agent-application-ship-1-formal": (
            ("context", "ContextPack", "RuntimeContextTrialRunner"),
            (
                "memory-skill-v1",
                "MemorySkillPack",
                "RuntimeCrossProcessRunner",
            ),
            (
                "semantic-memory-effect-v1",
                "SemanticMemoryEffectPack",
                "ProductionSemanticMemoryEffectRunner",
            ),
            ("tool-mcp", "ToolMCPPack", "MCPRuntimeTrialRunner"),
        ),
    }
    assert outcome.budget_snapshot is not None
    assert outcome.budget_snapshot.request_attempts == 1
    assert outcome.budget_snapshot.accounting_complete is True


@pytest.mark.asyncio
async def test_ship_rejects_invalid_calibration_before_formal(
    tmp_path: Path,
) -> None:
    calls = _Calls([], [], [])

    with pytest.raises(CampaignError, match="measurement is invalid"):
        await run_campaign(
            CampaignMode.SHIP,
            suite_path=_SUITE_PATH,
            output_root=tmp_path,
            services=_services(calls, measurement_valid=False),
        )

    assert "execute:agent-application-ship-1-calibration" in calls.entries
    assert "execute:agent-application-ship-1-formal" not in calls.entries


@pytest.mark.asyncio
async def test_campaign_artifact_binds_final_budget_ledger_snapshot(
    tmp_path: Path,
) -> None:
    calls = _Calls([], [], [])
    services = _services(calls)
    ledger = _budget_ledger(tmp_path)

    def prepare_guard(
        resolved,
        suite,
        mode,
        estimate,
        output_root,
        suite_digest,
        codeflow_commit,
    ):
        del suite, mode, estimate, output_root, suite_digest, codeflow_commit
        return replace(resolved, budget_ledger=ledger)

    outcome = await run_campaign(
        CampaignMode.CALIBRATION,
        suite_path=_SUITE_PATH,
        output_root=tmp_path,
        services=replace(
            services,
            prepare_budget_guard=prepare_guard,
        ),
    )

    assert outcome.budget_snapshot is not None
    assert outcome.campaign_artifact_path is not None
    artifact = json.loads(Path(outcome.campaign_artifact_path).read_text(encoding="utf-8"))
    assert artifact["budget_snapshot"]["ledger_path"] == str(ledger.path)
    assert artifact["budget_snapshot"]["ledger_digest"]


@pytest.mark.asyncio
async def test_standalone_formal_is_rejected_before_any_campaign_work(
    tmp_path: Path,
) -> None:
    calls = _Calls([], [], [])

    with pytest.raises(CampaignError, match="standalone formal"):
        await run_campaign(
            CampaignMode.FORMAL,
            suite_path=_SUITE_PATH,
            output_root=tmp_path,
            services=_services(calls),
        )

    assert calls.entries == []


@pytest.mark.asyncio
async def test_ship_dirty_worktree_aborts_before_provider_probe(
    tmp_path: Path,
) -> None:
    calls = _Calls([], [], [])

    with pytest.raises(CampaignError, match="clean worktree"):
        await run_campaign(
            CampaignMode.SHIP,
            suite_path=_SUITE_PATH,
            output_root=tmp_path,
            services=_services(calls, clean=False),
        )

    assert calls.entries == ["deterministic", "clean"]


@pytest.mark.asyncio
async def test_ship_revalidates_clean_commit_after_preflight(
    tmp_path: Path,
) -> None:
    calls = _Calls([], [], [])
    services = _services(calls)
    clean_states = iter((True, False))

    with pytest.raises(
        CampaignError,
        match="worktree changed after preflight",
    ):
        await run_campaign(
            CampaignMode.SHIP,
            suite_path=_SUITE_PATH,
            output_root=tmp_path,
            services=replace(
                services,
                worktree_is_clean=lambda: next(clean_states),
            ),
        )

    assert not any(entry.startswith("execute:") for entry in calls.entries)


@pytest.mark.asyncio
async def test_ship_rejects_commit_drift_after_preflight(
    tmp_path: Path,
) -> None:
    calls = _Calls([], [], [])
    services = _services(calls)
    commits = iter(("a" * 40, "b" * 40))

    with pytest.raises(
        CampaignError,
        match="commit changed after preflight",
    ):
        await run_campaign(
            CampaignMode.SHIP,
            suite_path=_SUITE_PATH,
            output_root=tmp_path,
            services=replace(
                services,
                resolve_codeflow_commit=lambda: next(commits),
            ),
        )

    assert not any(entry.startswith("execute:") for entry in calls.entries)


@pytest.mark.asyncio
async def test_ship_rejects_commit_drift_after_calibration_track(
    tmp_path: Path,
) -> None:
    calls = _Calls([], [], [])
    services = _services(calls)
    commits = iter(("a" * 40, "a" * 40, "b" * 40))

    with pytest.raises(
        CampaignError,
        match="commit changed after preflight",
    ):
        await run_campaign(
            CampaignMode.SHIP,
            suite_path=_SUITE_PATH,
            output_root=tmp_path,
            services=replace(
                services,
                resolve_codeflow_commit=lambda: next(commits),
            ),
        )

    assert "execute:agent-application-ship-1-calibration" in calls.entries
    assert "execute:agent-application-ship-1-formal" not in calls.entries


@pytest.mark.asyncio
async def test_ship_rejects_dirty_worktree_before_persisting_outcome(
    tmp_path: Path,
) -> None:
    calls = _Calls([], [], [])
    services = _services(calls)
    clean_states = iter((True, True, True, True, True, False))

    with pytest.raises(
        CampaignError,
        match="worktree changed after preflight",
    ):
        await run_campaign(
            CampaignMode.SHIP,
            suite_path=_SUITE_PATH,
            output_root=tmp_path,
            services=replace(
                services,
                worktree_is_clean=lambda: next(clean_states),
            ),
        )

    assert "execute:agent-application-ship-1-formal" in calls.entries
    campaign_artifacts = tuple(tmp_path.glob("campaigns/**/campaign-outcome.json"))
    assert campaign_artifacts == ()


@pytest.mark.asyncio
async def test_paid_campaign_rejects_unbound_commit_before_provider(
    tmp_path: Path,
) -> None:
    calls = _Calls([], [], [])
    services = _services(calls)

    with pytest.raises(CampaignError, match="full Git object id"):
        await run_campaign(
            CampaignMode.CALIBRATION,
            suite_path=_SUITE_PATH,
            output_root=tmp_path,
            services=replace(
                services,
                resolve_codeflow_commit=lambda: "main",
            ),
        )

    assert "resolve" not in calls.entries
    assert "probe" not in calls.entries


@pytest.mark.asyncio
async def test_budget_aborts_before_provider_probe(
    tmp_path: Path,
) -> None:
    calls = _Calls([], [], [])
    suite = load_campaign_suite(_SUITE_PATH)
    tiny_cap = replace(
        suite,
        budget=replace(suite.budget, hard_cap_cny=1),
    )

    with pytest.raises(CampaignError, match="cost ceiling"):
        await run_campaign(
            CampaignMode.SHIP,
            suite=tiny_cap,
            output_root=tmp_path,
            services=_services(calls),
        )

    assert "probe" not in calls.entries


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("probe_result", "message"),
    [
        (
            replace(_probe_result(), resolved_model="deepseek-v4-pro"),
            "model identity",
        ),
        (
            replace(_probe_result(), tool_calling_supported=False),
            "Tool Calling",
        ),
        (
            replace(_probe_result(), usage_fields=("total_tokens",)),
            "usage fields",
        ),
        (
            replace(_probe_result(), tokenizer_digest=""),
            "tokenizer",
        ),
        (
            replace(_probe_result(), fallback_used=True),
            "fallback",
        ),
    ],
)
async def test_preflight_capabilities_fail_closed_before_trials(
    tmp_path: Path,
    probe_result: ProviderProbeResult,
    message: str,
) -> None:
    calls = _Calls([], [], [])

    with pytest.raises(CampaignError, match=message):
        await run_campaign(
            CampaignMode.CALIBRATION,
            suite_path=_SUITE_PATH,
            output_root=tmp_path,
            services=_services(calls, probe_result=probe_result),
        )

    assert "probe" in calls.entries
    assert not any(entry.startswith("execute:") for entry in calls.entries)


@pytest.mark.asyncio
async def test_ship_rejects_calibration_ids_that_overlap_formal_ids(
    tmp_path: Path,
) -> None:
    calls = _Calls([], [], [])

    with pytest.raises(CampaignError, match="disjoint"):
        await run_campaign(
            CampaignMode.SHIP,
            suite_path=_SUITE_PATH,
            output_root=tmp_path,
            services=_services(calls, overlap=True),
        )

    assert "probe" not in calls.entries


@pytest.mark.asyncio
async def test_ship_rejects_per_pack_trial_redistribution_before_preflight(
    tmp_path: Path,
) -> None:
    calls = _Calls([], [], [])

    with pytest.raises(CampaignError, match="context planned Trial denominator drift"):
        await run_campaign(
            CampaignMode.SHIP,
            suite_path=_SUITE_PATH,
            output_root=tmp_path,
            services=_services(
                calls,
                redistribute_trials=True,
            ),
        )

    assert "probe" not in calls.entries


@pytest.mark.asyncio
async def test_ship_rejects_comparison_block_denominator_drift_before_preflight(
    tmp_path: Path,
) -> None:
    calls = _Calls([], [], [])

    with pytest.raises(
        CampaignError,
        match="context planned Comparison Block denominator drift",
    ):
        await run_campaign(
            CampaignMode.SHIP,
            suite_path=_SUITE_PATH,
            output_root=tmp_path,
            services=_services(
                calls,
                comparison_block_drift=True,
            ),
        )

    assert "probe" not in calls.entries


def test_budget_guard_uses_one_approval_ledger_across_modes_and_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product_home = tmp_path / "codeflow-home"
    monkeypatch.setenv("CODEFLOW_HOME", str(product_home))
    suite = load_campaign_suite(_SUITE_PATH)
    estimate = estimate_worst_case_cost(
        suite,
        modes=(CampaignMode.CALIBRATION, CampaignMode.FORMAL),
    )
    resolved = ResolvedProvider(
        provider_name="deepseek",
        model="deepseek-v4-flash",
        provider=_BudgetProvider(
            {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            }
        ),
        runtime_config_digest="f" * 64,
    )

    calibration = _prepare_budget_guard(
        resolved,
        suite,
        CampaignMode.CALIBRATION,
        estimate,
        tmp_path / "output-a",
        "a" * 64,
        "1" * 40,
    )
    formal = _prepare_budget_guard(
        resolved,
        suite,
        CampaignMode.FORMAL,
        estimate,
        tmp_path / "output-b",
        "a" * 64,
        "1" * 40,
    )

    assert calibration.budget_ledger is not None
    assert formal.budget_ledger is not None
    assert calibration.budget_ledger.path == formal.budget_ledger.path
    assert calibration.budget_ledger.path.is_relative_to(product_home)
    snapshot = formal.budget_ledger.snapshot()
    assert snapshot.request_attempt_baseline == 0
    assert snapshot.request_attempt_lifetime_ceiling == 1_746
    assert snapshot.approval_digest

    for index in range(210):
        request_id = formal.budget_ledger.reserve(
            trial_id=f"resume-proof-{index}",
            request_digest=f"{index:064x}",
            model="deepseek-v4-flash",
            estimated_input_tokens=42_000,
        )
        formal.budget_ledger.settle(
            request_id,
            input_tokens=42_000,
            output_tokens=1_500,
        )
    resumed = _prepare_budget_guard(
        resolved,
        suite,
        CampaignMode.FORMAL,
        estimate,
        tmp_path / "output-c",
        "a" * 64,
        "1" * 40,
    )

    assert resumed.budget_ledger is not None
    resumed_snapshot = resumed.budget_ledger.snapshot()
    assert resumed_snapshot.request_attempts == 210
    assert resumed_snapshot.provider_charged_cny > 9.5
    assert resumed_snapshot.request_attempt_baseline == 0
    assert resumed_snapshot.request_attempt_lifetime_ceiling == 1_746
    assert resumed_snapshot.request_attempt_lifetime_ceiling - resumed_snapshot.request_attempts == 1_536


def test_budget_guard_includes_existing_spend_and_probe_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEFLOW_HOME", str(tmp_path / "codeflow-home"))
    suite = load_campaign_suite(_SUITE_PATH)
    estimate = estimate_worst_case_cost(
        suite,
        modes=(CampaignMode.CALIBRATION, CampaignMode.FORMAL),
    )
    resolved = ResolvedProvider(
        provider_name="deepseek",
        model="deepseek-v4-flash",
        provider=_BudgetProvider(
            {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            }
        ),
        runtime_config_digest="f" * 64,
    )
    guarded = _prepare_budget_guard(
        resolved,
        suite,
        CampaignMode.SHIP,
        estimate,
        tmp_path / "output",
        "a" * 64,
        "1" * 40,
    )
    assert guarded.budget_ledger is not None
    request_id = guarded.budget_ledger.reserve(
        trial_id="prior-spend",
        request_digest="d" * 64,
        model="deepseek-v4-flash",
        estimated_input_tokens=100,
    )
    guarded.budget_ledger.settle(
        request_id,
        input_tokens=42_000,
        output_tokens=1_500,
    )
    constrained_suite = replace(
        suite,
        budget=replace(
            suite.budget,
            warning_cny=50,
            hard_cap_cny=54.5,
        ),
    )

    with pytest.raises(
        CampaignError,
        match="insufficient campaign budget",
    ):
        _prepare_budget_guard(
            resolved,
            constrained_suite,
            CampaignMode.SHIP,
            estimate_worst_case_cost(
                constrained_suite,
                modes=(
                    CampaignMode.CALIBRATION,
                    CampaignMode.FORMAL,
                ),
            ),
            tmp_path / "output",
            "b" * 64,
            "2" * 40,
        )


def test_budget_guard_rejects_approved_ledger_and_high_water_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEFLOW_HOME", str(tmp_path / "codeflow-home"))
    suite = load_campaign_suite(_SUITE_PATH)
    estimate = estimate_worst_case_cost(
        suite,
        modes=(CampaignMode.CALIBRATION, CampaignMode.FORMAL),
    )
    resolved = ResolvedProvider(
        provider_name="deepseek",
        model="deepseek-v4-flash",
        provider=_BudgetProvider(
            {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            }
        ),
        runtime_config_digest="f" * 64,
    )
    guarded = _prepare_budget_guard(
        resolved,
        suite,
        CampaignMode.SHIP,
        estimate,
        tmp_path / "output",
        "a" * 64,
        "1" * 40,
    )
    assert guarded.budget_ledger is not None
    request_id = guarded.budget_ledger.reserve(
        trial_id="paid-attempt",
        request_digest="d" * 64,
        model="deepseek-v4-flash",
        estimated_input_tokens=100,
    )
    guarded.budget_ledger.settle(
        request_id,
        input_tokens=100,
        output_tokens=1,
    )
    guarded.budget_ledger.path.write_text("", encoding="utf-8")
    guarded.budget_ledger.high_water_path.unlink()

    with pytest.raises(
        ProviderBudgetError,
        match="high-water record is missing",
    ):
        _prepare_budget_guard(
            resolved,
            suite,
            CampaignMode.SHIP,
            estimate,
            tmp_path / "output",
            "a" * 64,
            "1" * 40,
        )


def test_budget_guard_preserves_commit_scoped_approval_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEFLOW_HOME", str(tmp_path / "codeflow-home"))
    suite = load_campaign_suite(_SUITE_PATH)
    estimate = estimate_worst_case_cost(
        suite,
        modes=(CampaignMode.CALIBRATION, CampaignMode.FORMAL),
    )
    resolved = ResolvedProvider(
        provider_name="deepseek",
        model="deepseek-v4-flash",
        provider=_BudgetProvider(
            {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            }
        ),
        runtime_config_digest="f" * 64,
    )
    first_digest = "a" * 64
    first_commit = "1" * 40
    first = _prepare_budget_guard(
        resolved,
        suite,
        CampaignMode.SHIP,
        estimate,
        tmp_path / "output",
        first_digest,
        first_commit,
    )
    assert first.budget_ledger is not None
    request_id = first.budget_ledger.reserve(
        trial_id="old-suite-attempt",
        request_digest="d" * 64,
        model="deepseek-v4-flash",
        estimated_input_tokens=100,
    )
    first.budget_ledger.settle(
        request_id,
        input_tokens=100,
        output_tokens=1,
    )
    approval_root = first.budget_ledger.path.parent
    approval_paths = set(
        (approval_root / "provider-budget-approvals").glob("*.v3.json"),
    )
    assert len(approval_paths) == 1
    first_path = approval_paths.pop()
    first_approval = json.loads(first_path.read_text(encoding="utf-8"))

    second_commit = "2" * 40
    second = _prepare_budget_guard(
        resolved,
        suite,
        CampaignMode.SHIP,
        estimate,
        tmp_path / "output",
        first_digest,
        second_commit,
    )

    assert second.budget_ledger is not None
    second_paths = set(
        (approval_root / "provider-budget-approvals").glob("*.v3.json"),
    ) - {first_path}
    assert len(second_paths) == 1
    second_path = second_paths.pop()
    second_approval = json.loads(second_path.read_text(encoding="utf-8"))
    assert json.loads(first_path.read_text(encoding="utf-8")) == first_approval
    assert first_approval["request_attempt_baseline"] == 0
    assert first_approval["codeflow_commit"] == first_commit
    assert first_approval["request_attempt_lifetime_ceiling"] == 1_746
    assert second_approval["request_attempt_baseline"] == 1
    assert second_approval["codeflow_commit"] == second_commit
    assert second_approval["request_attempt_lifetime_ceiling"] == 1_747
    assert second.budget_ledger.snapshot().request_attempt_baseline == 1
    repeated_request = second.budget_ledger.reserve(
        trial_id="old-suite-attempt",
        request_digest="d" * 64,
        model="deepseek-v4-flash",
        estimated_input_tokens=100,
    )
    second.budget_ledger.settle(
        repeated_request,
        input_tokens=100,
        output_tokens=1,
    )
    assert second.budget_ledger.snapshot().request_attempts == 2


def test_makefile_exposes_checkout_only_smoke_and_ship_targets() -> None:
    makefile = (Path(__file__).resolve().parents[1] / "Makefile").read_text(
        encoding="utf-8",
    )

    assert "codeflowbench-smoke:" in makefile
    assert "codeflowbench:" in makefile
    assert ("uv run --frozen --all-extras --exact python -m benchmarks.codeflowbench --mode smoke") in makefile
    assert ("uv run --frozen --all-extras --exact python -m benchmarks.codeflowbench --mode ship") in makefile


def test_cli_exposes_only_smoke_and_calibration_gated_ship() -> None:
    assert _CLI_MODES == (
        CampaignMode.SMOKE,
        CampaignMode.SHIP,
    )


@pytest.mark.asyncio
async def test_provider_guard_rejects_call_beyond_per_trial_cap(
    tmp_path: Path,
) -> None:
    delegate = _BudgetProvider(
        {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
        }
    )
    guarded = BudgetGuardedProvider(
        delegate,
        ledger=_budget_ledger(tmp_path),
    )

    with provider_call_budget_scope(
        trial_id="trial-cap",
        max_logical_calls=1,
        max_attempts_per_call=2,
    ):
        await guarded.chat(
            [{"role": "user", "content": "first"}],
            max_tokens=10,
        )
        with pytest.raises(
            ProviderBudgetError,
            match="per-Trial",
        ):
            await guarded.chat(
                [{"role": "user", "content": "second"}],
                max_tokens=10,
            )

    assert delegate.calls == 1


@pytest.mark.asyncio
async def test_provider_guard_preauthorizes_against_cny_hard_cap(
    tmp_path: Path,
) -> None:
    config = ProviderBudgetConfig(
        hard_cap_cny=100,
        external_service_reserve_cny=0,
        max_total_request_attempts=4,
        max_input_tokens_per_call=1_000_000,
        max_output_tokens_per_call=1_000_000,
        input_cache_miss_usd_per_million=30,
        output_usd_per_million=30,
        conservative_usd_to_cny_multiplier=1,
    )
    delegate = _BudgetProvider(
        {
            "prompt_tokens": 1_000_000,
            "completion_tokens": 1_000_000,
            "total_tokens": 2_000_000,
        }
    )
    ledger = ProviderBudgetLedger(
        tmp_path / "hard-cap.jsonl",
        config,
    )
    guarded = BudgetGuardedProvider(delegate, ledger=ledger)

    with provider_call_budget_scope(
        trial_id="hard-cap",
        max_logical_calls=2,
        max_attempts_per_call=1,
    ):
        await guarded.chat(
            [{"role": "user", "content": "first"}],
            max_tokens=1_000_000,
        )
        with pytest.raises(
            ProviderBudgetError,
            match="hard cap",
        ):
            await guarded.chat(
                [{"role": "user", "content": "second"}],
                max_tokens=1_000_000,
            )

    assert delegate.calls == 1
    assert ledger.snapshot().total_committed_cny == 60


@pytest.mark.asyncio
async def test_provider_guard_blocks_campaign_after_missing_usage(
    tmp_path: Path,
) -> None:
    delegate = _BudgetProvider({})
    ledger = _budget_ledger(tmp_path)
    guarded = BudgetGuardedProvider(delegate, ledger=ledger)

    with provider_call_budget_scope(
        trial_id="missing-usage",
        max_logical_calls=2,
        max_attempts_per_call=1,
    ):
        with pytest.raises(
            ProviderBudgetError,
            match="usage accounting",
        ):
            await guarded.chat(
                [{"role": "user", "content": "first"}],
                max_tokens=10,
            )
        with pytest.raises(
            ProviderBudgetError,
            match="accounting is incomplete",
        ):
            await guarded.chat(
                [{"role": "user", "content": "second"}],
                max_tokens=10,
            )

    assert delegate.calls == 1
    assert ledger.snapshot().accounting_complete is False


def test_runtime_config_redaction_covers_nested_headers_and_environment() -> None:
    first = {
        "provider": {
            "extra_headers": {
                "Authorization": "Bearer first-secret",
                "APP-Code": "first-app-code",
            },
        },
        "mcp_servers": {
            "private": {
                "headers": {"X-Custom-Auth": "first-header-secret"},
                "env": {
                    "CUSTOM_LOGIN": "first-login-secret",
                    "SAFE_SETTING": "still-private",
                },
            },
        },
        "ordinary": {"mode": "fixed"},
    }
    second = json.loads(
        json.dumps(first)
        .replace("first-secret", "second-secret")
        .replace("first-app-code", "second-app-code")
        .replace("first-header-secret", "second-header-secret")
        .replace("first-login-secret", "second-login-secret")
        .replace("still-private", "different-private-value")
    )

    redacted = campaign._redact_secrets(first)
    serialized = json.dumps(redacted, sort_keys=True)

    assert "first-" not in serialized
    assert "still-private" not in serialized
    assert redacted["ordinary"] == {"mode": "fixed"}
    assert canonical_digest(redacted) == canonical_digest(campaign._redact_secrets(second))


def _budget_ledger(tmp_path: Path) -> ProviderBudgetLedger:
    return ProviderBudgetLedger(
        tmp_path / "provider-budget.jsonl",
        ProviderBudgetConfig(
            hard_cap_cny=100,
            external_service_reserve_cny=5,
            max_total_request_attempts=8,
            max_input_tokens_per_call=16_000,
            max_output_tokens_per_call=1_500,
            input_cache_miss_usd_per_million=0.14,
            output_usd_per_million=0.28,
            conservative_usd_to_cny_multiplier=7.5,
        ),
    )
