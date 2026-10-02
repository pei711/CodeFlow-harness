from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.codeflowbench.packs.tokenwise_cost import (
    CACHE_POLICY_PREFIX_DISRUPTED,
    CACHE_POLICY_PREFIX_STABLE,
)
from benchmarks.codeflowbench.packs.tokenwise_cost.live import (
    CampaignBudget,
    CampaignConfig,
    CampaignError,
    LiveTrialResult,
    build_arm,
    build_campaign_report,
    build_current_campaign_report,
    load_task_corpus,
    rotated_cache_policies,
)
from benchmarks.codeflowbench.packs.tokenwise_cost.runner import (
    ProviderCallRecord,
    TrialArtifact,
    _seed_workspace,
    execute_live_trial,
    load_deepseek_key,
    run_formal_campaign,
    verify_retained_campaign,
)

CORPUS = Path(__file__).resolve().parents[1] / "benchmarks" / "codeflowbench" / "tasks" / "tokenwise_cost" / "formal.json"


def test_formal_corpus_is_sealed_and_matches_the_frozen_matrix() -> None:
    corpus = load_task_corpus(CORPUS)

    assert corpus.schema == "codeflow.codeflowbench.tokenwise-cost.tasks.v1"
    assert len(corpus.tasks) == 12
    assert {task.workload_class for task in corpus.tasks} == {
        "stable_dialogue",
        "long_history",
        "tool_accumulation",
        "intra_turn_tool_chain",
    }
    assert all(len(task.expected_outputs) == task.turn_count for task in corpus.tasks)
    assert len(corpus.digest) == 64


@pytest.mark.parametrize(
    ("policy", "strategy_name"),
    [
        (CACHE_POLICY_PREFIX_DISRUPTED, "prefix_disruptor"),
        (CACHE_POLICY_PREFIX_STABLE, None),
    ],
)
def test_build_arm_exposes_only_the_declared_cache_treatment(
    policy: str,
    strategy_name: str | None,
) -> None:
    arm = build_arm(policy)

    assert arm.cache_policy == policy
    assert getattr(arm.strategy, "name", None) == strategy_name


def test_arm_order_rotates_deterministically_without_dropping_policies() -> None:
    orders = [rotated_cache_policies("0" * 64, index) for index in range(2)]

    assert [order[0] for order in orders] == [
        CACHE_POLICY_PREFIX_DISRUPTED,
        CACHE_POLICY_PREFIX_STABLE,
    ]
    assert all(set(order) == set(orders[0]) for order in orders)


def test_campaign_budget_fails_before_the_next_provider_call() -> None:
    budget = CampaignBudget(hard_cap_usd=0.10, max_provider_calls=2)
    budget.reserve_call()
    budget.record_call(cost_usd=0.04)
    budget.reserve_call()
    budget.record_call(cost_usd=0.04)

    with pytest.raises(CampaignError, match="provider call ceiling"):
        budget.reserve_call()

    budget = CampaignBudget(hard_cap_usd=0.10, max_provider_calls=5)
    budget.reserve_call()
    budget.record_call(cost_usd=0.10)

    with pytest.raises(CampaignError, match="cost ceiling"):
        budget.reserve_call()


def test_campaign_report_rebuilds_cv_metrics_from_terminal_trial_records(tmp_path: Path) -> None:
    corpus = load_task_corpus(CORPUS)
    config = CampaignConfig(
        model="deepseek/deepseek-v4-flash",
        repetitions=3,
        hard_cap_usd=2.0,
        output_root=tmp_path,
    )
    costs = {
        CACHE_POLICY_PREFIX_DISRUPTED: 1.0,
        CACHE_POLICY_PREFIX_STABLE: 0.5,
    }
    records: list[LiveTrialResult] = []
    for task in corpus.tasks:
        for repetition in range(3):
            for policy, cost in costs.items():
                records.append(
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
                        fresh_input_tokens=1000,
                        cache_write_tokens=0,
                        cache_read_tokens=600 if policy == CACHE_POLICY_PREFIX_STABLE else 0,
                        output_tokens=20,
                        cost_usd=cost,
                        provider_calls=2,
                        latency_ms=10,
                        findings=(),
                    )
                )

    report = build_campaign_report(config=config, corpus=corpus, trials=tuple(records))
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":"))

    assert report["claim"]["claim_eligible"] is True
    assert report["claim"]["cv_metrics"]["trial_count"] == 72
    assert report["claim"]["cv_metrics"]["valid_comparison_blocks"] == 36
    assert report["campaign"]["task_corpus_digest"] == corpus.digest
    assert report["campaign"]["price_snapshot"]["cache_miss_usd_per_token"] == 0.14e-6
    assert "generated_at" not in encoded


def test_current_report_requires_runtime_call_records_and_positive_paired_interval(tmp_path: Path) -> None:
    corpus = load_task_corpus(CORPUS)
    config = CampaignConfig(output_root=tmp_path)
    records: list[LiveTrialResult] = []
    for task in corpus.tasks:
        for repetition in range(3):
            for policy, cost in (
                (CACHE_POLICY_PREFIX_DISRUPTED, 1.0),
                (CACHE_POLICY_PREFIX_STABLE, 0.5),
            ):
                records.append(
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
                        fresh_input_tokens=1000,
                        cache_write_tokens=0,
                        cache_read_tokens=600 if policy == CACHE_POLICY_PREFIX_STABLE else 0,
                        output_tokens=20,
                        cost_usd=cost,
                        provider_calls=2,
                        latency_ms=10,
                        findings=(),
                        runtime_assembly=True,
                        call_efficiency_mode="observe",
                        call_efficiency_records=2,
                        call_efficiency_records_complete=True,
                        call_efficiency_ledger_status="healthy",
                        call_efficiency_accepted_records=2,
                        call_efficiency_persisted_records=2,
                        call_efficiency_lost_records=0,
                    )
                )

    report = build_current_campaign_report(config=config, corpus=corpus, trials=tuple(records))

    assert report["schema"] == "codeflow.codeflowbench.call-efficiency-cost.report.v2"
    assert report["claim"]["claim_eligible"] is True
    assert report["claim"]["paired_cost_reduction_95_ci"]["lower"] == 0.5
    assert report["claim"]["paired_cost_reduction_95_ci"]["upper"] == 0.5
    assert all(report["gates"].values())

    records[0] = records[0].__class__(**{**records[0].__dict__, "call_efficiency_records_complete": False})
    rejected = build_current_campaign_report(config=config, corpus=corpus, trials=tuple(records))
    assert rejected["claim"]["claim_eligible"] is False
    assert rejected["gates"]["call_efficiency_per_attempt_complete"] is False


def test_deepseek_key_preflight_fails_closed_without_a_credential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(CampaignError, match="DeepSeek credential"):
        load_deepseek_key(config_path=tmp_path / "missing.json")


def test_workspace_seed_uses_current_context_layout(tmp_path: Path) -> None:
    _seed_workspace(tmp_path, namespace="trial-namespace", workload_class="tool_accumulation")

    state = tmp_path / "state"
    soul = (state / "agent_memory" / "profile" / "soul.md").read_text(encoding="utf-8")
    assert "Namespace: trial-namespace" in soul
    assert "call lookup_record exactly once" in soul
    assert "Rule 179" in soul
    assert (state / "agent_memory" / "profile" / "agent.md").is_file()
    assert (state / "user_memory" / "profile" / "user.md").is_file()
    assert (state / "TOOLS.md").is_file()


@pytest.mark.asyncio
async def test_formal_campaign_persists_each_trial_and_rebuilds_on_resume(tmp_path: Path) -> None:
    corpus = load_task_corpus(CORPUS)
    config = CampaignConfig(output_root=tmp_path)
    calls: list[tuple[str, int, str]] = []

    async def fake_trial(task, repetition, arm, **_kwargs):
        calls.append((task.task_id, repetition, arm.cache_policy))
        cost = {
            CACHE_POLICY_PREFIX_DISRUPTED: 1.0,
            CACHE_POLICY_PREFIX_STABLE: 0.5,
        }[arm.cache_policy]
        provider_call = ProviderCallRecord(
            request_digest="a" * 64,
            requested_model=config.model,
            actual_model=config.model,
            usage_complete=True,
            fresh_input_tokens=1000,
            cache_write_tokens=0,
            cache_read_tokens=600 if arm.cache_policy == CACHE_POLICY_PREFIX_STABLE else 0,
            output_tokens=10,
            cost_usd=cost,
            finish_reason="stop",
            latency_ms=1,
        )
        call_efficiency_record = {
            "outcome": "success",
            "requested_model": config.model,
            "actual_model": "deepseek-v4-flash",
            "usage": {
                "complete": True,
                "input_tokens": 1000,
                "cache_read_tokens": provider_call.cache_read_tokens,
                "cache_write_tokens": 0,
                "output_tokens": 10,
            },
            "estimated_cost_usd": cost,
        }
        return TrialArtifact(
            result=LiveTrialResult(
                task_id=task.task_id,
                workload_class=task.workload_class,
                repetition=repetition,
                cache_policy=arm.cache_policy,
                task_passed=True,
                usage_complete=True,
                cost_complete=True,
                requested_model=config.model,
                actual_model=config.model,
                fallback_used=False,
                fresh_input_tokens=1000,
                cache_write_tokens=0,
                cache_read_tokens=600 if arm.cache_policy == CACHE_POLICY_PREFIX_STABLE else 0,
                output_tokens=10,
                cost_usd=cost,
                provider_calls=1,
                latency_ms=1,
                findings=(),
                runtime_assembly=True,
                call_efficiency_mode="observe",
                call_efficiency_records=1,
                call_efficiency_records_complete=True,
                call_efficiency_ledger_status="healthy",
                call_efficiency_accepted_records=1,
                call_efficiency_persisted_records=1,
                call_efficiency_lost_records=0,
            ),
            provider_calls=(provider_call,),
            outputs=task.expected_outputs,
            call_efficiency_records=(call_efficiency_record,),
        )

    preflight = {"schema": "test.preflight.v1", "passed": True}
    (tmp_path / "preflight.json").write_text(json.dumps(preflight), encoding="utf-8")
    first = await run_formal_campaign(
        config=config,
        corpus=corpus,
        api_key="deepseek-test-key",
        codeflow_commit="f" * 40,
        trial_executor=fake_trial,
        preflight_report=preflight,
    )
    second = await run_formal_campaign(
        config=config,
        corpus=corpus,
        api_key="deepseek-test-key",
        codeflow_commit="f" * 40,
        trial_executor=fake_trial,
        preflight_report=preflight,
    )
    verifier = verify_retained_campaign(
        output_root=tmp_path,
        corpus=corpus,
        codeflow_commit="f" * 40,
    )

    assert len(calls) == 72
    assert first == second
    assert first["claim"]["claim_eligible"] is True
    assert len(tuple((tmp_path / "trials").glob("*.json"))) == 72
    assert verifier["passed"] is True
    assert (tmp_path / "raw-outcomes.jsonl").is_file()
    assert (tmp_path / "aggregate.json").is_file()
    assert (tmp_path / "claim-eligibility.json").is_file()
    assert (tmp_path / "inventory.json").is_file()


@pytest.mark.asyncio
async def test_live_trial_crosses_runtime_assembly_and_persists_every_call_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codeflow.providers.base import LLMProvider, LLMResponse

    class FakeDeepSeekProvider(LLMProvider):
        def __init__(self, *, model, budget, **_kwargs):
            super().__init__()
            self.model = model
            self.budget = budget
            self.records: list[ProviderCallRecord] = []

        def get_default_model(self) -> str:
            return self.model

        async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7, **_kwargs):
            import re

            requested_model = model or self.model
            match = re.search(r"Reply only ([A-Z]+\d+)\.", str(messages[-1].get("content", "")))
            content = match.group(1) if match else "UNKNOWN"
            usage = {
                "prompt_tokens": 100,
                "completion_tokens": 1,
                "total_tokens": 101,
                "cache_read_input_tokens": 0,
                "cache_miss_input_tokens": 100,
            }
            cost = 100 * 0.14e-6 + 1 * 0.28e-6
            self.budget.reserve_call()
            self.budget.record_call(cost_usd=cost)
            self.records.append(
                ProviderCallRecord(
                    request_digest="0" * 64,
                    requested_model=requested_model,
                    actual_model=requested_model,
                    usage_complete=True,
                    fresh_input_tokens=100,
                    cache_write_tokens=0,
                    cache_read_tokens=0,
                    output_tokens=1,
                    cost_usd=cost,
                    finish_reason="stop",
                    latency_ms=1,
                )
            )
            return LLMResponse(content=content, finish_reason="stop", usage=usage, model=requested_model)

    monkeypatch.setattr(
        "benchmarks.codeflowbench.packs.tokenwise_cost.runner._RecordingDeepSeekProvider",
        FakeDeepSeekProvider,
    )
    corpus = load_task_corpus(CORPUS)
    config = CampaignConfig(output_root=tmp_path)
    budget = CampaignBudget(hard_cap_usd=1.0, max_provider_calls=20)

    artifact = await execute_live_trial(
        corpus.tasks[0],
        0,
        build_arm(CACHE_POLICY_PREFIX_STABLE),
        api_key="unused",
        config=config,
        plan_digest="f" * 64,
        budget=budget,
    )

    assert artifact.result.task_passed is True
    assert artifact.result.runtime_assembly is True
    assert artifact.result.call_efficiency_mode == "observe"
    assert artifact.result.provider_calls == 6
    assert artifact.result.call_efficiency_records == 6
    assert artifact.result.call_efficiency_records_complete is True
    assert artifact.result.call_efficiency_ledger_status == "healthy"
    assert artifact.result.call_efficiency_accepted_records == 6
    assert artifact.result.call_efficiency_persisted_records == 6
    assert artifact.result.call_efficiency_lost_records == 0
    assert len(artifact.call_efficiency_records) == 6
