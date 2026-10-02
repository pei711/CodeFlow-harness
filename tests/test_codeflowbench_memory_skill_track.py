"""Tests for the frozen historical Memory/Skill Pack and evidence labels."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.codeflowbench import budget as codeflowbench_budget
from benchmarks.codeflowbench.budget import (
    BudgetGuardedProvider,
    ProviderBudgetConfig,
    ProviderBudgetLedger,
)
from benchmarks.codeflowbench.canonical import to_primitive
from benchmarks.codeflowbench.packs.memory_skill import (
    DeterministicCrossProcessRunner,
    RuntimeCrossProcessRunner,
    create_calibration_pack,
    create_formal_pack,
    reduce_memory_skill_claims,
    run_retrieval_micro_suite,
    summarize_retrieval,
)
from benchmarks.codeflowbench.packs.memory_skill import e2e as memory_skill_e2e
from benchmarks.codeflowbench.packs.memory_skill import runtime as memory_skill_runtime
from benchmarks.codeflowbench.packs.memory_skill import worker as memory_skill_worker
from benchmarks.codeflowbench.packs.memory_skill.models import CrossSessionTask
from benchmarks.codeflowbench.packs.memory_skill.retrieval import _FixtureMemoryBackend
from benchmarks.codeflowbench.packs.memory_skill.runtime import (
    ScriptedCrossSessionProvider,
    _freeze_codeflow_config,
    _freeze_runtime_config,
)
from benchmarks.codeflowbench.packs.memory_skill.semantic_fixtures import (
    semantic_fixture,
)
from benchmarks.codeflowbench.protocol import RetrievalContext, TrialContext
from benchmarks.codeflowbench.records import (
    RetrievalCaseKey,
    RetrievalStatus,
    TrialKey,
    TrialStatus,
    VerificationState,
)
from benchmarks.codeflowbench.schema import (
    ExecutionPolicy,
    ExperimentSpec,
    ProviderTrialBudget,
)
from codeflow.config.codeflow import CodeFlowConfig
from codeflow.config.schema import Config
from codeflow.context_engine.segments import MemorySegmentBuilder
from codeflow.memory_engine.skill_forge import SkillForgeRouter
from codeflow.providers.base import GenerationSettings


def test_formal_memory_skill_definition_freezes_independent_matrices() -> None:
    pack = create_formal_pack(DeterministicCrossProcessRunner())
    definition = pack.definition()

    assert len(definition.tasks) == 8
    assert definition.identity["claim_reducer"] == "memory_skill_v1"
    assert definition.identity["retrieval_evidence_level"] == "deterministic_contract"
    assert definition.identity["everos_semantic_quality_claim_eligible"] is False
    assert definition.identity["retrieval_paid_provider_calls"] == 0
    assert definition.identity["retrieval_embedding_calls"] == 0
    assert len(definition.identity["retrieval_fixture_digest"]) == 64
    assert definition.identity["memory_corpus_items"] == 160
    assert definition.identity["skill_corpus_items"] == 90
    assert [variant.variant_id for variant in definition.variants] == [
        "user_memory_off",
        "user_memory_on_local_only",
        "user_memory_on_local_plus_everos",
    ]
    assert [
        (
            pair.treatment_axis,
            pair.control_variant_id,
            pair.treatment_variant_id,
        )
        for pair in definition.pairs
    ] == [
        (
            "user_memory_recall",
            "user_memory_off",
            "user_memory_on_local_plus_everos",
        ),
        (
            "skill_sources",
            "user_memory_on_local_only",
            "user_memory_on_local_plus_everos",
        ),
    ]

    memory_suite, skill_suite = definition.retrieval_suites
    assert memory_suite.retrieval_suite_id == "user-memory-retrieval-v1"
    assert len(memory_suite.queries) == 80
    assert sum(query.label == "positive" for query in memory_suite.queries) == 50
    assert sum(query.label == "hard_negative" for query in memory_suite.queries) == 30
    assert len(memory_suite.configurations) == 1

    assert skill_suite.retrieval_suite_id == "skill-source-fusion-v1"
    assert len(skill_suite.queries) == 60
    assert sum(query.label == "positive" for query in skill_suite.queries) == 40
    assert sum(query.label == "hard_negative" for query in skill_suite.queries) == 20
    assert [configuration.configuration_id for configuration in skill_suite.configurations] == [
        "local_only",
        "everos_only",
        "fused",
    ]


def test_calibration_matrix_is_smaller_and_disjoint_from_formal() -> None:
    runner = DeterministicCrossProcessRunner()
    formal = create_formal_pack(runner).definition()
    calibration = create_calibration_pack(runner).definition()

    assert len(calibration.tasks) == 4
    assert calibration.pack_id != formal.pack_id
    assert {task.task_id for task in calibration.tasks}.isdisjoint(task.task_id for task in formal.tasks)

    memory_suite, skill_suite = calibration.retrieval_suites
    assert len(memory_suite.queries) == 10
    assert len(memory_suite.configurations) == 1
    assert len(skill_suite.queries) == 8
    assert len(skill_suite.configurations) == 3
    assert 10 + 8 * 3 == 34
    formal_query_ids = {query.query_id for suite in formal.retrieval_suites for query in suite.queries}
    calibration_query_ids = {query.query_id for suite in calibration.retrieval_suites for query in suite.queries}
    assert calibration_query_ids.isdisjoint(formal_query_ids)


def test_semantic_addendum_freezes_natural_language_case_counts() -> None:
    calibration = semantic_fixture("calibration")
    formal = semantic_fixture("formal")

    assert len(calibration.memory_corpus) == 16
    assert len(calibration.skill_corpus) == 10
    assert len(calibration.memory_queries) == 10
    assert len(calibration.skill_queries) == 8
    assert calibration.planned_cases == 34
    assert len(formal.memory_corpus) == 160
    assert len(formal.skill_corpus) == 90
    assert len(formal.memory_queries) == 80
    assert len(formal.skill_queries) == 60
    assert formal.planned_cases == 260
    formal_positive_skill_queries = tuple(
        str(query.payload["query_text"]) for query in formal.skill_queries if query.label == "positive"
    )
    assert len(set(formal_positive_skill_queries)) == 40

    queries = (
        *calibration.memory_queries,
        *calibration.skill_queries,
        *formal.memory_queries,
        *formal.skill_queries,
    )
    assert all(len(str(query.payload["query_text"]).split()) >= 8 for query in queries)
    forbidden_markers = (
        "token",
        "nonce",
        "notfound",
        "unknown",
        "orchid",
        "saffron",
    )
    assert not any(
        marker in str(query.payload["query_text"]).lower() for query in queries for marker in forbidden_markers
    )


def test_worker_artifact_does_not_persist_exception_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "result.json"

    async def fail(_args: object) -> dict[str, object]:
        raise RuntimeError("api_key=never-write-this-secret")

    monkeypatch.setattr(memory_skill_worker, "_run", fail)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "codeflowbench-memory-worker",
            "--stage",
            "learning",
            "--spec",
            str(tmp_path / "unused-spec.json"),
            "--result",
            str(result_path),
        ],
    )

    assert memory_skill_worker.main() == 1
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload == {
        "error_code": "worker_exception",
        "error_type": "RuntimeError",
        "status": "infrastructure_failure",
    }
    assert "never-write-this-secret" not in result_path.read_text(encoding="utf-8")


def test_memory_skill_tasks_are_loaded_from_frozen_manifests() -> None:
    definition = create_formal_pack(DeterministicCrossProcessRunner()).definition()
    manifest_path = Path(__file__).parents[1] / "benchmarks" / "codeflowbench" / "tasks" / "memory_skill" / "formal.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["schema"] == "codeflow.codeflowbench.memory-skill-tasks.v1"
    assert [task.payload["evaluation_request"] for task in definition.tasks] == [
        task["evaluation_request"] for task in payload["tasks"]
    ]


def test_real_runner_freezes_model_budget_and_routing(
    tmp_path: Path,
) -> None:
    task = CrossSessionTask(
        task_id="real-runner-test",
        workspace_id="workspace",
        learned_fact="customer-region=test-region",
        expected_value="test-region",
        required_skill="local-skill",
        evaluation_request="write the verified result",
    )
    delegate = ScriptedCrossSessionProvider(task=task, stage="learning")
    guarded = BudgetGuardedProvider(
        delegate,
        ledger=ProviderBudgetLedger(
            tmp_path / "provider-budget.jsonl",
            ProviderBudgetConfig(
                hard_cap_cny=100.0,
                external_service_reserve_cny=0.0,
                max_total_request_attempts=4,
                max_input_tokens_per_call=20_000,
                max_output_tokens_per_call=1_500,
                input_cache_miss_usd_per_million=1.0,
                output_usd_per_million=1.0,
                conservative_usd_to_cny_multiplier=8.0,
            ),
        ),
    )
    source = Config()
    source.routing.enabled = True
    frozen = _freeze_runtime_config(
        source,
        workspace=tmp_path / "workspace",
        model="provider/exact-model",
        stage="evaluation",
    )

    assert (
        RuntimeCrossProcessRunner(
            config=frozen,
            codeflow_config=object(),
            provider=guarded,
        ).kind
        == "runtime_cross_process_real_provider"
    )
    assert frozen.agents.defaults.model == "provider/exact-model"
    assert frozen.agents.defaults.max_tokens == 1_500
    assert frozen.agents.defaults.max_tool_iterations == 3
    assert frozen.routing.enabled is False
    assert frozen.tools.tool_search.enabled is False
    assert "understand_media" in frozen.tools.disabled_tools
    assert "write_file" not in frozen.tools.disabled_tools
    learning = _freeze_runtime_config(
        source,
        workspace=tmp_path / "workspace",
        model="provider/exact-model",
        stage="learning",
    )
    assert "understand_media" in learning.tools.disabled_tools
    assert "write_file" in learning.tools.disabled_tools
    codeflow_config = CodeFlowConfig(base=frozen)
    codeflow_config.token_wise.smart_routing.enabled = True
    frozen_codeflow = _freeze_codeflow_config(
        codeflow_config,
        frozen,
        task,
    )
    assert frozen_codeflow.token_wise.smart_routing.enabled is False
    with pytest.raises(ValueError, match="BudgetGuardedProvider"):
        RuntimeCrossProcessRunner(
            config=frozen,
            codeflow_config=object(),
            provider=delegate,
        )


@pytest.mark.asyncio
async def test_real_runner_uses_frozen_pack_budget_and_four_call_stage_split(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pack = create_formal_pack(DeterministicCrossProcessRunner())
    definition = pack.definition()
    task = CrossSessionTask(**dict(definition.tasks[0].payload))
    delegate = ScriptedCrossSessionProvider(task=task, stage="learning")
    ledger = ProviderBudgetLedger(
        tmp_path / "shared-provider-budget.jsonl",
        ProviderBudgetConfig(
            hard_cap_cny=100.0,
            external_service_reserve_cny=0.0,
            max_total_request_attempts=8,
            max_input_tokens_per_call=20_000,
            max_output_tokens_per_call=1_500,
            input_cache_miss_usd_per_million=1.0,
            output_usd_per_million=1.0,
            conservative_usd_to_cny_multiplier=8.0,
        ),
    )
    guarded = BudgetGuardedProvider(delegate, ledger=ledger)
    config = _freeze_runtime_config(
        Config(),
        workspace=tmp_path / "workspace",
        model="provider/exact-model",
        stage="evaluation",
    )
    runner = RuntimeCrossProcessRunner(
        config=config,
        codeflow_config=CodeFlowConfig(base=config),
        provider=guarded,
    )
    experiment = ExperimentSpec(
        suite="memory-skill-real-runner-contract",
        repetitions=1,
        pack_ids=(definition.pack_id,),
        output_root=tmp_path,
        identity={"codeflow_commit": "1" * 40, "model": "provider/exact-model"},
        execution=ExecutionPolicy(
            max_provider_calls_per_trial=8,
            provider_trial_budgets=(
                ProviderTrialBudget(
                    pack_id=definition.pack_id,
                    max_provider_calls_per_trial=4,
                    max_input_tokens_per_call=15_000,
                    max_output_tokens_per_call=1_500,
                ),
            ),
        ),
    )
    variant = definition.variants[0]
    context = TrialContext(
        experiment_id="real-runner-contract",
        plan_digest="a" * 64,
        key=TrialKey(
            experiment_id="real-runner-contract",
            pack_id=definition.pack_id,
            task_id=definition.tasks[0].task_id,
            variant_id=variant.variant_id,
            repetition=0,
        ),
        block_attempt=1,
        experiment=experiment,
        task=definition.tasks[0],
        variant=variant,
    )
    captured: dict[str, object] = {}

    async def fake_run(
        fake_context: TrialContext,
        *,
        provider_spec: dict[str, object],
    ):
        captured.update(provider_spec)
        assert Path(str(provider_spec["private_config_path"])).exists()
        return memory_skill_e2e._preparation_failure(
            fake_context,
            "captured",
        )

    monkeypatch.setattr(
        memory_skill_e2e,
        "_run_cross_process_trial",
        fake_run,
    )

    execution = await runner.run(context)

    assert execution.status is TrialStatus.INFRASTRUCTURE_FAILURE
    assert captured["ledger_path"] == str(ledger.path.resolve())
    assert captured["provider_identity"] == "provider/exact-model"
    assert captured["stage_logical_call_caps"] == {
        "learning": 1,
        "evaluation": 3,
    }
    assert captured["max_input_tokens_per_call"] == 15_000
    assert captured["max_output_tokens_per_call"] == 1_500
    assert "api_key" not in captured
    assert "config" not in captured


@pytest.mark.asyncio
async def test_real_provider_worker_bootstrap_uses_stage_budget_scopes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = CrossSessionTask(
        task_id="worker-budget-scope",
        workspace_id="worker-workspace",
        learned_fact="customer-region=worker-test",
        expected_value="worker-test",
        required_skill="local-worker-budget",
        evaluation_request="write the verified result",
    )
    workspace = tmp_path / "workspace"
    memory_skill_e2e._prepare_workspace(workspace, task)
    state_path = tmp_path / "memory-state.json"
    config = Config()
    config.agents.defaults.model = "provider/exact-model"
    codeflow_config = CodeFlowConfig(base=config)
    private_config_path = tmp_path / "runtime-config.json"
    private_config_path.write_text(
        json.dumps(
            {
                "config": config.model_dump(mode="json"),
                "codeflow_config": codeflow_config.model_dump(mode="json"),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    ledger_config = ProviderBudgetConfig(
        hard_cap_cny=100.0,
        external_service_reserve_cny=0.0,
        max_total_request_attempts=8,
        max_input_tokens_per_call=15_000,
        max_output_tokens_per_call=1_500,
        input_cache_miss_usd_per_million=1.0,
        output_usd_per_million=1.0,
        conservative_usd_to_cny_multiplier=8.0,
    )
    provider_spec = {
        "mode": "real",
        "provider_identity": "provider/exact-model",
        "private_config_path": str(private_config_path),
        "ledger_path": str(tmp_path / "provider-budget.jsonl"),
        "ledger_config": asdict(ledger_config),
        "trial_id": "memory-skill-v1/worker-budget-scope/full/0/1",
        "max_attempts_per_call": 2,
        "max_input_tokens_per_call": 15_000,
        "max_output_tokens_per_call": 1_500,
        "stage_logical_call_caps": {
            "learning": 1,
            "evaluation": 3,
        },
        "paid_campaign_eligible": True,
        "real_agent_task_effect_claim_eligible": True,
    }
    spec_path = tmp_path / "worker-spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "task": to_primitive(task),
                "variant_settings": {
                    "user_memory_recall": "enabled",
                    "skill_sources": ["local", "everos"],
                },
                "workspace": str(workspace),
                "state_path": str(state_path),
                "provider": provider_spec,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    active_stage = ""
    budget_exhaustion_case = False
    delegates = []

    class ExhaustingCrossSessionProvider(
        ScriptedCrossSessionProvider,
    ):
        async def chat(self, *args, **kwargs):
            self._calls = 0
            return await super().chat(*args, **kwargs)

    def make_provider(config: Config) -> ScriptedCrossSessionProvider:
        del config
        provider_class = ExhaustingCrossSessionProvider if budget_exhaustion_case else ScriptedCrossSessionProvider
        delegate = provider_class(
            task=task,
            stage=active_stage,
        )
        delegate.generation = GenerationSettings(
            max_tokens=1_500,
            temperature=0.0,
        )
        delegates.append(delegate)
        return delegate

    monkeypatch.setattr(
        "codeflow.cli._helpers.make_provider",
        make_provider,
    )
    scopes = []
    actual_budget_scope = codeflowbench_budget.provider_call_budget_scope

    def recording_budget_scope(**kwargs):
        scopes.append(kwargs)
        return actual_budget_scope(**kwargs)

    monkeypatch.setattr(
        memory_skill_runtime,
        "provider_call_budget_scope",
        recording_budget_scope,
    )
    results = {}
    for stage in ("learning", "evaluation"):
        active_stage = stage
        results[stage] = await memory_skill_worker._run(
            SimpleNamespace(
                stage=stage,
                spec=spec_path,
            )
        )

    assert [result["runtime_state"] for result in results.values()] == [
        "completed",
        "completed",
    ]
    assert results["learning"]["provider_identity"] == ("provider/exact-model")
    assert results["evaluation"]["paid_campaign_eligible"] is True
    assert len(delegates) == 2
    assert scopes == [
        {
            "trial_id": ("memory-skill-v1/worker-budget-scope/full/0/1/learning"),
            "max_logical_calls": 1,
            "max_attempts_per_call": 2,
            "max_input_tokens_per_call": 15_000,
            "max_output_tokens_per_call": 1_500,
        },
        {
            "trial_id": ("memory-skill-v1/worker-budget-scope/full/0/1/evaluation"),
            "max_logical_calls": 3,
            "max_attempts_per_call": 2,
            "max_input_tokens_per_call": 15_000,
            "max_output_tokens_per_call": 1_500,
        },
    ]
    ledger_events = [
        json.loads(line) for line in Path(provider_spec["ledger_path"]).read_text(encoding="utf-8").splitlines()
    ]
    reservations = [event for event in ledger_events if event["kind"] == "reserved"]
    assert [event["trial_id"] for event in reservations] == [
        "memory-skill-v1/worker-budget-scope/full/0/1/learning",
        "memory-skill-v1/worker-budget-scope/full/0/1/evaluation",
        "memory-skill-v1/worker-budget-scope/full/0/1/evaluation",
    ]
    assert all(
        event["maximum_input_tokens"] == 15_000 and event["maximum_output_tokens"] == 1_500 for event in reservations
    )

    provider_spec["trial_id"] = "memory-skill-v1/worker-budget-ceiling/full/0/1"
    spec_payload = json.loads(spec_path.read_text(encoding="utf-8"))
    spec_payload["provider"] = provider_spec
    spec_path.write_text(
        json.dumps(spec_payload, sort_keys=True),
        encoding="utf-8",
    )
    active_stage = "evaluation"
    budget_exhaustion_case = True
    budget_exhausted = await memory_skill_worker._run(
        SimpleNamespace(
            stage="evaluation",
            spec=spec_path,
        )
    )

    assert budget_exhausted["runtime_state"] == "completed"
    assert budget_exhausted["failure_category"] == "task_budget_exhausted"
    assert budget_exhausted["usage"]["calls"] == 3
    assert budget_exhausted["usage"]["input_tokens"] > 0
    assert budget_exhausted["usage"]["output_tokens"] == 36
    assert budget_exhausted["usage"]["usage_complete"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("runtime_state", "failure_category", "expected_status"),
    (
        (
            "provider_failed",
            "network",
            TrialStatus.PROVIDER_FAILURE,
        ),
        (
            "provider_failed",
            "task_budget_exhausted",
            TrialStatus.TASK_TIMEOUT,
        ),
        (
            "error",
            "runtime_error",
            TrialStatus.TASK_FAILED,
        ),
        (
            "cancelled",
            "cancelled",
            TrialStatus.CANCELLED,
        ),
    ),
)
async def test_learning_runtime_failure_short_circuits_paid_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    runtime_state: str,
    failure_category: str,
    expected_status: TrialStatus,
) -> None:
    definition = create_calibration_pack(
        DeterministicCrossProcessRunner(),
    ).definition()
    variant = definition.variants[0]
    context = TrialContext(
        experiment_id="learning-short-circuit",
        plan_digest="a" * 64,
        key=TrialKey(
            experiment_id="learning-short-circuit",
            pack_id=definition.pack_id,
            task_id=definition.tasks[0].task_id,
            variant_id=variant.variant_id,
            repetition=0,
        ),
        block_attempt=1,
        experiment=ExperimentSpec(
            suite="learning-short-circuit",
            repetitions=1,
            pack_ids=(definition.pack_id,),
            output_root=tmp_path,
            identity={
                "codeflow_commit": "1" * 40,
                "model": "provider/exact-model",
            },
            execution=ExecutionPolicy(
                max_provider_calls_per_trial=4,
            ),
        ),
        task=definition.tasks[0],
        variant=variant,
    )
    stages: list[str] = []

    async def fake_stage(
        *,
        stage: str,
        spec_path: Path,
        isolation: object,
    ) -> dict[str, object]:
        del spec_path, isolation
        stages.append(stage)
        assert stage == "learning"
        usage_complete = failure_category != "network"
        return {
            "status": "completed",
            "runtime_state": runtime_state,
            "delivery_state": "dropped",
            "failure_category": failure_category,
            "session_message_count": 1,
            "provider_identity": "provider/exact-model",
            "paid_campaign_eligible": True,
            "cost_complete": True,
            "provider_charged_cny": 0.25,
            "usage": {
                "calls": 1,
                "input_tokens": (100 if usage_complete else None),
                "output_tokens": (20 if usage_complete else None),
                "total_tokens": (120 if usage_complete else None),
                "usage_complete": usage_complete,
            },
        }

    monkeypatch.setattr(
        memory_skill_e2e,
        "_run_stage",
        fake_stage,
    )

    execution = await memory_skill_e2e._run_cross_process_trial(
        context,
        provider_spec={
            "mode": "real",
            "paid_campaign_eligible": True,
        },
    )

    assert stages == ["learning"]
    assert execution.status is expected_status
    assert execution.runtime_state.value == runtime_state
    assert execution.metrics["runtime.evaluation_skipped"] is True
    assert execution.metrics["runtime.failure_category"] == failure_category
    assert execution.metrics["usage.model_calls"] == 1
    if failure_category == "network":
        assert execution.metrics["usage.main_agent_input_tokens"] is None
        assert execution.metrics["usage.complete"] is False
    else:
        assert execution.metrics["usage.main_agent_input_tokens"] == 100
        assert execution.metrics["usage.complete"] is True
    assert execution.metrics["cost.estimated_cny"] == 0.25
    assert execution.findings == (f"learning_runtime_short_circuit:{failure_category}",)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("runtime_state", "failure_category", "expected_status"),
    (
        (
            "provider_failed",
            "network",
            TrialStatus.PROVIDER_FAILURE,
        ),
        (
            "provider_failed",
            "task_budget_exhausted",
            TrialStatus.TASK_TIMEOUT,
        ),
        (
            "completed",
            "task_budget_exhausted",
            TrialStatus.TASK_TIMEOUT,
        ),
    ),
)
async def test_evaluation_terminal_failure_preserves_null_usage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    runtime_state: str,
    failure_category: str,
    expected_status: TrialStatus,
) -> None:
    definition = create_calibration_pack(
        DeterministicCrossProcessRunner(),
    ).definition()
    variant = definition.variants[0]
    context = TrialContext(
        experiment_id="evaluation-provider-failure",
        plan_digest="b" * 64,
        key=TrialKey(
            experiment_id="evaluation-provider-failure",
            pack_id=definition.pack_id,
            task_id=definition.tasks[0].task_id,
            variant_id=variant.variant_id,
            repetition=0,
        ),
        block_attempt=1,
        experiment=ExperimentSpec(
            suite="evaluation-provider-failure",
            repetitions=1,
            pack_ids=(definition.pack_id,),
            output_root=tmp_path,
            identity={
                "codeflow_commit": "2" * 40,
                "model": "provider/exact-model",
            },
            execution=ExecutionPolicy(
                max_provider_calls_per_trial=4,
            ),
        ),
        task=definition.tasks[0],
        variant=variant,
    )
    stages: list[str] = []

    async def fake_stage(
        *,
        stage: str,
        spec_path: Path,
        isolation: object,
    ) -> dict[str, object]:
        del spec_path, isolation
        stages.append(stage)
        completed = stage == "learning"
        return {
            "status": "completed",
            "pid": 1 if completed else 2,
            "conversation": stage,
            "runtime_state": ("completed" if completed else runtime_state),
            "delivery_state": ("delivered" if completed else "dropped"),
            "failure_category": (None if completed else failure_category),
            "session_message_count": 1,
            "backend_quiescent": True,
            "user_recall_calls": 0,
            "suppressed_user_recall_calls": 1,
            "agent_recall_calls": 0,
            "memory_hits": 0,
            "injected_skill_ids": [],
            "backend_class": "EverosBackend",
            "backend_adapter": "injected_fixture",
            "everos_semantic_quality_claim_eligible": False,
            "provider_memory_observed": None,
            "provider_skill_observed": None,
            "provider_identity": "provider/exact-model",
            "paid_campaign_eligible": True,
            "real_agent_task_effect_claim_eligible": True,
            "cost_complete": completed,
            "provider_charged_cny": 0.1,
            "usage": {
                "calls": 1,
                "input_tokens": 100 if completed else None,
                "output_tokens": 20 if completed else None,
                "total_tokens": 120 if completed else None,
                "usage_complete": completed,
            },
        }

    monkeypatch.setattr(
        memory_skill_e2e,
        "_run_stage",
        fake_stage,
    )

    execution = await memory_skill_e2e._run_cross_process_trial(
        context,
        provider_spec={
            "mode": "real",
            "paid_campaign_eligible": True,
        },
    )

    assert stages == ["learning", "evaluation"]
    assert execution.status is expected_status
    assert execution.metrics["runtime.failure_category"] == failure_category
    assert execution.metrics["usage.main_agent_input_tokens"] is None
    assert execution.metrics["usage.trial_total_tokens"] is None
    assert execution.metrics["usage.complete"] is False
    assert execution.findings == ()


def test_memory_skill_verifier_rejects_non_object_json(
    tmp_path: Path,
) -> None:
    definition = create_calibration_pack(
        DeterministicCrossProcessRunner(),
    ).definition()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "result.json").write_text(
        json.dumps(["not", "an", "object"]),
        encoding="utf-8",
    )

    verification = memory_skill_e2e._verify_result(
        workspace,
        CrossSessionTask(**dict(definition.tasks[0].payload)),
    )

    assert verification.state is VerificationState.FAILED
    assert verification.findings == ("result_artifact_mismatch",)
    assert verification.metrics == {
        "region_matched": False,
        "skill_applied": False,
    }


@pytest.mark.asyncio
async def test_cancelled_worker_is_terminated_killed_and_reaped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubbornProcess:
        def __init__(self) -> None:
            self.returncode: int | None = None
            self.communicate_started = asyncio.Event()
            self.terminated = False
            self.killed = False
            self.wait_calls = 0

        async def communicate(self):
            self.communicate_started.set()
            await asyncio.Event().wait()

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        async def wait(self) -> int:
            self.wait_calls += 1
            if self.returncode is None:
                await asyncio.Event().wait()
            return self.returncode

    monkeypatch.setattr(
        memory_skill_e2e,
        "_PROCESS_TERMINATION_GRACE_SECONDS",
        0.001,
    )
    process = StubbornProcess()
    communicate = asyncio.create_task(
        memory_skill_e2e._communicate_and_reap(process),
    )
    await process.communicate_started.wait()

    communicate.cancel()
    with pytest.raises(asyncio.CancelledError):
        await communicate

    assert process.terminated is True
    assert process.killed is True
    assert process.returncode == -9
    assert process.wait_calls == 2


@pytest.mark.asyncio
async def test_retrieval_records_are_anonymous_and_keep_memory_and_skill_separate(
    tmp_path: Path,
) -> None:
    pack = create_formal_pack(DeterministicCrossProcessRunner())
    definition = pack.definition()
    experiment = ExperimentSpec(
        suite="memory-skill-test",
        repetitions=1,
        pack_ids=(definition.pack_id,),
        output_root=tmp_path,
        identity={"codeflow_commit": "a" * 40, "model": "scripted/deterministic"},
    )
    memory_suite, skill_suite = definition.retrieval_suites
    memory_query = memory_suite.queries[0]
    memory_configuration = memory_suite.configurations[0]
    memory = await pack.run_retrieval_case(
        RetrievalContext(
            experiment_id="e" * 64,
            plan_digest="e" * 64,
            key=RetrievalCaseKey(
                experiment_id="e" * 64,
                retrieval_suite_id=memory_suite.retrieval_suite_id,
                query_id=memory_query.query_id,
                configuration_id=memory_configuration.configuration_id,
            ),
            query_block_attempt=1,
            experiment=experiment,
            query=memory_query,
            configuration=memory_configuration,
        )
    )

    assert memory.status is RetrievalStatus.MEASURABLE
    assert memory.usage["backend_class"] == "EverosBackend"
    assert memory.usage["backend_adapter"] == "injected_fixture"
    assert memory.usage["retrieval_evidence_level"] == "deterministic_contract"
    assert memory.usage["everos_semantic_quality_claim_eligible"] is False
    assert memory.injected_results[0]["item_id"] == memory_query.expected_item_ids[0]
    assert memory.injected_results[0]["source"] == "user_memory"
    assert "text" not in memory.injected_results[0]
    assert set(memory.injected_results[0]) == {
        "query_id",
        "item_id",
        "source",
        "rank",
        "raw_score",
        "rrf_score",
        "contributing_sources",
        "injected",
        "consuming_turn",
    }

    overlap_query = skill_suite.queries[20]
    fused_configuration = skill_suite.configurations[2]
    fused = await pack.run_retrieval_case(
        RetrievalContext(
            experiment_id="e" * 64,
            plan_digest="e" * 64,
            key=RetrievalCaseKey(
                experiment_id="e" * 64,
                retrieval_suite_id=skill_suite.retrieval_suite_id,
                query_id=overlap_query.query_id,
                configuration_id=fused_configuration.configuration_id,
            ),
            query_block_attempt=1,
            experiment=experiment,
            query=overlap_query,
            configuration=fused_configuration,
        )
    )

    assert fused.status is RetrievalStatus.MEASURABLE
    assert fused.usage["backend_class"] == "EverosBackend"
    assert fused.usage["backend_adapter"] == "injected_fixture"
    assert fused.usage["everos_semantic_quality_claim_eligible"] is False
    assert fused.injected_results[0]["item_id"] == overlap_query.expected_item_ids[0]
    assert fused.injected_results[0]["source"] == "fused"
    assert fused.injected_results[0]["contributing_sources"] == ["local", "everos"]
    assert fused.injected_results[0]["rrf_score"] > 0


@pytest.mark.asyncio
async def test_retrieval_adapter_executes_product_segments_and_router(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = {"memory_segment": 0, "skill_router": 0, "everos_backend": 0}
    original_memory_build = MemorySegmentBuilder.build
    original_router_select = SkillForgeRouter.select
    original_everos_recall = _FixtureMemoryBackend.recall

    async def memory_build(self, context):
        calls["memory_segment"] += 1
        return await original_memory_build(self, context)

    async def router_select(self, query, history, k=5, *, diagnostics=None):
        calls["skill_router"] += 1
        return await original_router_select(
            self,
            query,
            history,
            k,
            diagnostics=diagnostics,
        )

    async def everos_recall(
        self,
        query,
        *,
        user_id=None,
        agent_id=None,
        top_k,
    ):
        calls["everos_backend"] += 1
        return await original_everos_recall(
            self,
            query,
            user_id=user_id,
            agent_id=agent_id,
            top_k=top_k,
        )

    monkeypatch.setattr(MemorySegmentBuilder, "build", memory_build)
    monkeypatch.setattr(SkillForgeRouter, "select", router_select)
    monkeypatch.setattr(_FixtureMemoryBackend, "recall", everos_recall)
    pack = create_formal_pack(DeterministicCrossProcessRunner())
    definition = pack.definition()
    experiment = ExperimentSpec(
        suite="memory-skill-product-path",
        repetitions=1,
        pack_ids=(definition.pack_id,),
        output_root=tmp_path,
        identity={"codeflow_commit": "c" * 40, "model": "scripted/deterministic"},
    )
    for suite in definition.retrieval_suites:
        query = suite.queries[0]
        configuration = suite.configurations[-1]
        await pack.run_retrieval_case(
            RetrievalContext(
                experiment_id="e" * 64,
                plan_digest="e" * 64,
                key=RetrievalCaseKey(
                    experiment_id="e" * 64,
                    retrieval_suite_id=suite.retrieval_suite_id,
                    query_id=query.query_id,
                    configuration_id=configuration.configuration_id,
                ),
                query_block_attempt=1,
                experiment=experiment,
                query=query,
                configuration=configuration,
            )
        )

    assert calls["memory_segment"] == 1
    assert calls["skill_router"] >= 2
    assert calls["everos_backend"] >= 2


@pytest.mark.asyncio
async def test_formal_retrieval_micro_suite_has_separate_claim_gates(
    tmp_path: Path,
) -> None:
    pack = create_formal_pack(DeterministicCrossProcessRunner())
    experiment = ExperimentSpec(
        suite="memory-skill-micro",
        repetitions=1,
        pack_ids=(pack.definition().pack_id,),
        output_root=tmp_path,
        identity={"codeflow_commit": "b" * 40, "model": "scripted/deterministic"},
    )

    measurements = await run_retrieval_micro_suite(pack, experiment)
    summary = summarize_retrieval(measurements)

    assert len(measurements) == 260
    assert summary.measurable_cases == 260
    assert summary.memory_final_injection_recall_at_5 >= 0.80
    assert summary.memory_hard_negative_injection_rate <= 0.05
    assert summary.memory_stale_injection_count == 0
    assert summary.memory_cross_workspace_leakage_count == 0
    assert summary.memory_contract_gate_passed
    assert summary.memory_positive_claim_eligible is False

    best_single = max(summary.skill_local_recall_at_5, summary.skill_everos_recall_at_5)
    assert summary.skill_fused_recall_at_5 - best_single >= 0.05
    assert summary.skill_hard_negative_injection_rate <= 0.05
    assert summary.skill_cross_workspace_leakage_count == 0
    assert summary.skill_contract_gate_passed
    assert summary.skill_positive_claim_eligible is False


@pytest.mark.asyncio
async def test_artifact_reducer_separates_contract_and_real_claims(
    tmp_path: Path,
) -> None:
    pack = create_formal_pack(DeterministicCrossProcessRunner())
    definition = pack.definition()
    experiment = ExperimentSpec(
        suite="memory-skill-reducer",
        repetitions=3,
        pack_ids=(definition.pack_id,),
        output_root=tmp_path,
        identity={"codeflow_commit": "f" * 40, "model": "scripted/deterministic"},
    )
    measurements = await run_retrieval_micro_suite(pack, experiment)
    retrieval_records = [
        {
            "key": {
                "retrieval_suite_id": measurement.retrieval_suite_id,
                "query_id": measurement.query_id,
                "configuration_id": measurement.configuration_id,
            },
            "status": measurement.execution.status.value,
            "label": measurement.label,
            "expected_item_ids": list(measurement.expected_item_ids),
            "ranked_results": to_primitive(measurement.execution.ranked_results),
            "injected_results": to_primitive(measurement.execution.injected_results),
            "usage": to_primitive(measurement.execution.usage),
        }
        for measurement in measurements
    ]
    for record in retrieval_records:
        record["usage"]["everos_semantic_quality_claim_eligible"] = True
    trial_records = []
    pair_results = []
    plan_digest = "a" * 64
    experiment_id = "e" * 64
    for task in definition.tasks:
        for repetition in range(3):
            pair_keys = [
                {
                    "experiment_id": experiment_id,
                    "pack_id": definition.pack_id,
                    "treatment_axis": pair.treatment_axis,
                    "task_id": task.task_id,
                    "repetition": repetition,
                    "control_variant_id": pair.control_variant_id,
                    "treatment_variant_id": pair.treatment_variant_id,
                }
                for pair in definition.pairs
            ]
            for variant in definition.variants:
                passed = variant.variant_id == "user_memory_on_local_plus_everos"
                trial_records.append(
                    {
                        "key": {
                            "experiment_id": experiment_id,
                            "pack_id": definition.pack_id,
                            "task_id": task.task_id,
                            "variant_id": variant.variant_id,
                            "repetition": repetition,
                        },
                        "plan_digest": plan_digest,
                        "pair_memberships": [
                            pair_key
                            for pair_key in pair_keys
                            if variant.variant_id
                            in {
                                pair_key["control_variant_id"],
                                pair_key["treatment_variant_id"],
                            }
                        ],
                        "status": "passed" if passed else "task_failed",
                        "selected_block_attempt": 1,
                        "observed_variant_settings": dict(variant.settings),
                        "metrics": {
                            "real_agent_task_effect_claim_eligible": False,
                        },
                    }
                )
            settings = {variant.variant_id: dict(variant.settings) for variant in definition.variants}
            for pair_key in pair_keys:
                control = settings[pair_key["control_variant_id"]]
                treatment = settings[pair_key["treatment_variant_id"]]
                actual_diff = {
                    key: {
                        "control": control.get(key),
                        "treatment": treatment.get(key),
                    }
                    for key in sorted(control.keys() | treatment.keys())
                    if control.get(key) != treatment.get(key)
                }
                pair_results.append(
                    {
                        "key": pair_key,
                        "plan_digest": plan_digest,
                        "selected_block_attempt": 1,
                        "valid": True,
                        "actual_variant_diff": actual_diff,
                    }
                )

    metrics = reduce_memory_skill_claims(
        trial_records,
        retrieval_records,
        pair_results,
    )

    assert metrics["memory_skill.measurement_valid"] is True
    assert metrics["memory_e2e.claim_contract_valid"] is True
    assert metrics["skill_e2e.claim_contract_valid"] is True
    assert metrics["memory_retrieval.claim_contract_valid"] is True
    assert metrics["memory_retrieval.irrelevant_injection_rate"] == 0.0
    assert metrics["skill_fusion.claim_contract_valid"] is True
    assert metrics["memory_e2e.real_agent_task_effect_claim_eligible"] is False
    assert metrics["memory_retrieval.real_semantic_evidence_valid"] is False
    assert metrics["memory_retrieval.real_semantic_claim_eligible"] is False
    assert metrics["skill_fusion.real_semantic_evidence_valid"] is False
    assert metrics["skill_fusion.real_semantic_claim_eligible"] is False
    assert metrics["evidence.task_effect_level"] == "deterministic_contract"
    assert metrics["evidence.retrieval_level"] == "deterministic_contract"


@pytest.mark.asyncio
async def test_artifact_reducer_rejects_pair_trial_attempt_mismatch(
    tmp_path: Path,
) -> None:
    pack = create_formal_pack(DeterministicCrossProcessRunner())
    definition = pack.definition()
    experiment = ExperimentSpec(
        suite="memory-skill-pair-guard",
        repetitions=3,
        pack_ids=(definition.pack_id,),
        output_root=tmp_path,
        identity={"codeflow_commit": "f" * 40, "model": "scripted/deterministic"},
    )
    measurements = await run_retrieval_micro_suite(pack, experiment)
    retrieval_records = [
        {
            "key": {
                "retrieval_suite_id": measurement.retrieval_suite_id,
                "query_id": measurement.query_id,
                "configuration_id": measurement.configuration_id,
            },
            "status": measurement.execution.status.value,
            "label": measurement.label,
            "expected_item_ids": list(measurement.expected_item_ids),
            "ranked_results": to_primitive(measurement.execution.ranked_results),
            "injected_results": to_primitive(measurement.execution.injected_results),
            "usage": to_primitive(measurement.execution.usage),
        }
        for measurement in measurements
    ]
    plan_digest = "a" * 64
    experiment_id = "e" * 64
    trials = []
    pairs = []
    for task in definition.tasks:
        for repetition in range(3):
            pair = definition.pairs[0]
            pair_key = {
                "experiment_id": experiment_id,
                "pack_id": definition.pack_id,
                "treatment_axis": pair.treatment_axis,
                "task_id": task.task_id,
                "repetition": repetition,
                "control_variant_id": pair.control_variant_id,
                "treatment_variant_id": pair.treatment_variant_id,
            }
            variants = {variant.variant_id: variant for variant in definition.variants}
            for variant_id in (
                pair.control_variant_id,
                pair.treatment_variant_id,
            ):
                variant = variants[variant_id]
                trials.append(
                    {
                        "key": {
                            "experiment_id": experiment_id,
                            "pack_id": definition.pack_id,
                            "task_id": task.task_id,
                            "variant_id": variant_id,
                            "repetition": repetition,
                        },
                        "plan_digest": plan_digest,
                        "pair_memberships": [pair_key],
                        "status": "passed",
                        "selected_block_attempt": 1,
                        "observed_variant_settings": dict(variant.settings),
                        "metrics": {
                            "real_agent_task_effect_claim_eligible": True,
                        },
                    }
                )
            control = dict(variants[pair.control_variant_id].settings)
            treatment = dict(variants[pair.treatment_variant_id].settings)
            pairs.append(
                {
                    "key": pair_key,
                    "plan_digest": plan_digest,
                    "selected_block_attempt": (
                        2 if task.task_id == definition.tasks[0].task_id and repetition == 0 else 1
                    ),
                    "valid": not (task.task_id == definition.tasks[0].task_id and repetition == 1),
                    "actual_variant_diff": {
                        key: {
                            "control": control.get(key),
                            "treatment": treatment.get(key),
                        }
                        for key in sorted(control.keys() | treatment.keys())
                        if control.get(key) != treatment.get(key)
                    },
                }
            )
    pairs[2]["key"] = {
        **pairs[2]["key"],
        "treatment_axis": "wrong_axis",
    }

    metrics = reduce_memory_skill_claims(
        trials,
        retrieval_records,
        pairs,
    )

    assert metrics["memory_e2e.valid_pairs"] == 21
    assert metrics["memory_e2e.coverage_valid"] is False
    assert metrics["memory_e2e.attempt_consistency_valid"] is False
    assert metrics["memory_skill.measurement_valid"] is False


@pytest.mark.parametrize(
    ("calibration", "invalid_indices", "expected_valid", "coverage_valid"),
    (
        (False, (0, 3), 22, True),
        (False, (0, 3, 6), 21, False),
        (False, (0, 1), 22, False),
        (True, (0,), 7, False),
    ),
)
def test_memory_e2e_pair_coverage_policy(
    calibration: bool,
    invalid_indices: tuple[int, ...],
    expected_valid: int,
    coverage_valid: bool,
) -> None:
    pack = (
        create_calibration_pack(DeterministicCrossProcessRunner())
        if calibration
        else create_formal_pack(DeterministicCrossProcessRunner())
    )
    trials, pairs = _memory_pair_artifacts(
        pack.definition(),
        repetitions=2 if calibration else 3,
        invalid_indices=frozenset(invalid_indices),
    )

    metrics = reduce_memory_skill_claims(trials, [], pairs)

    assert metrics["memory_e2e.valid_pairs"] == expected_valid
    assert metrics["memory_e2e.coverage_valid"] is coverage_valid
    assert metrics["memory_e2e.attempt_consistency_valid"] is True
    assert metrics["memory_e2e.variant_axis_valid"] is True


def test_memory_e2e_rejects_axis_drift_inside_coverage_tolerance() -> None:
    definition = create_formal_pack(DeterministicCrossProcessRunner()).definition()
    trials, pairs = _memory_pair_artifacts(
        definition,
        repetitions=3,
        invalid_indices=frozenset(),
    )
    pair = pairs[0]
    pair_key = pair["key"]
    pair["valid"] = False
    pair["selected_block_attempt"] = None
    arms = [
        trial
        for trial in trials
        if trial["key"]["task_id"] == pair_key["task_id"] and trial["key"]["repetition"] == pair_key["repetition"]
    ]
    for arm in arms:
        arm["status"] = "provider_failure"
    treatment = next(arm for arm in arms if arm["key"]["variant_id"] == pair_key["treatment_variant_id"])
    treatment["observed_variant_settings"]["unexpected_axis"] = "enabled"
    control = next(arm for arm in arms if arm["key"]["variant_id"] == pair_key["control_variant_id"])
    pair["actual_variant_diff"] = {
        key: {
            "control": control["observed_variant_settings"].get(key),
            "treatment": treatment["observed_variant_settings"].get(key),
        }
        for key in sorted(control["observed_variant_settings"].keys() | treatment["observed_variant_settings"].keys())
        if control["observed_variant_settings"].get(key) != treatment["observed_variant_settings"].get(key)
    }

    metrics = reduce_memory_skill_claims(trials, [], pairs)

    assert metrics["memory_e2e.valid_pairs"] == 23
    assert metrics["memory_e2e.coverage_valid"] is True
    assert metrics["memory_e2e.variant_axis_valid"] is False


def _memory_pair_artifacts(
    definition,
    *,
    repetitions: int,
    invalid_indices: frozenset[int],
) -> tuple[list[dict], list[dict]]:
    pair_spec = next(pair for pair in definition.pairs if pair.treatment_axis == "user_memory_recall")
    variants = {variant.variant_id: variant for variant in definition.variants}
    plan_digest = "b" * 64
    experiment_id = "c" * 64
    trials: list[dict] = []
    pairs: list[dict] = []
    pair_index = 0
    for task in definition.tasks:
        for repetition in range(repetitions):
            pair_key = {
                "experiment_id": experiment_id,
                "pack_id": definition.pack_id,
                "treatment_axis": pair_spec.treatment_axis,
                "task_id": task.task_id,
                "repetition": repetition,
                "control_variant_id": pair_spec.control_variant_id,
                "treatment_variant_id": pair_spec.treatment_variant_id,
            }
            operational_failure = pair_index in invalid_indices
            for variant_id in (
                pair_spec.control_variant_id,
                pair_spec.treatment_variant_id,
            ):
                variant = variants[variant_id]
                trials.append(
                    {
                        "key": {
                            "experiment_id": experiment_id,
                            "pack_id": definition.pack_id,
                            "task_id": task.task_id,
                            "variant_id": variant_id,
                            "repetition": repetition,
                        },
                        "plan_digest": plan_digest,
                        "pair_memberships": [pair_key],
                        "status": ("provider_failure" if operational_failure else "passed"),
                        "selected_block_attempt": 1,
                        "observed_variant_settings": dict(
                            variant.settings,
                        ),
                        "metrics": {
                            "real_agent_task_effect_claim_eligible": True,
                        },
                    }
                )
            control = dict(
                variants[pair_spec.control_variant_id].settings,
            )
            treatment = dict(
                variants[pair_spec.treatment_variant_id].settings,
            )
            pairs.append(
                {
                    "key": pair_key,
                    "plan_digest": plan_digest,
                    "selected_block_attempt": (None if operational_failure else 1),
                    "valid": not operational_failure,
                    "actual_variant_diff": {
                        key: {
                            "control": control.get(key),
                            "treatment": treatment.get(key),
                        }
                        for key in sorted(control.keys() | treatment.keys())
                        if control.get(key) != treatment.get(key)
                    },
                }
            )
            pair_index += 1
    return trials, pairs
