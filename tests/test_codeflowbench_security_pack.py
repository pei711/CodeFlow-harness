from __future__ import annotations

import pytest

from benchmarks.codeflowbench.packs.security import (
    FORMAL_SECURITY_SCENARIO_COUNT,
    SMOKE_SECURITY_SCENARIO_COUNT,
    DeterministicSecurityRunner,
    SecurityCategory,
    SecurityMetrics,
    SecurityPack,
    SecurityRunOutcome,
    SecurityScenario,
    SecuritySummary,
    SecurityTrack,
    SecurityVerifier,
    load_security_scenarios,
    security_task_set_digest,
)
from benchmarks.codeflowbench.plan import compile_plan
from benchmarks.codeflowbench.protocol import TrialContext
from benchmarks.codeflowbench.records import (
    TrialKey,
    TrialStatus,
    VerificationState,
)
from benchmarks.codeflowbench.schema import ExperimentSpec, VariantSpec


def test_security_scenarios_loading() -> None:
    formal = load_security_scenarios(SecurityTrack.FORMAL)
    assert len(formal) == FORMAL_SECURITY_SCENARIO_COUNT == 30

    smoke = load_security_scenarios(SecurityTrack.SMOKE)
    assert len(smoke) == SMOKE_SECURITY_SCENARIO_COUNT == 5

    categories = {s.category for s in formal}
    assert len(categories) == 5

    # Exactly 6 scenarios per category
    for cat in SecurityCategory:
        cat_scenarios = [s for s in formal if s.category == cat]
        assert len(cat_scenarios) == 6


def test_security_task_set_digest_deterministic() -> None:
    digest1 = security_task_set_digest(SecurityTrack.FORMAL)
    digest2 = security_task_set_digest(SecurityTrack.FORMAL)
    assert digest1 == digest2
    assert len(digest1) == 64


def test_security_verifier_outcomes() -> None:
    verifier = SecurityVerifier()
    scenario = load_security_scenarios(SecurityTrack.SMOKE)[0]

    # Clean intercept
    outcome_ok = SecurityRunOutcome(
        task_id=scenario.task_id,
        category=scenario.category.value,
        repetition=1,
        intercepted=True,
        executed_unapproved_dangerous=0,
        executed_outside_workspace_write=0,
        guard_mechanism="shell_deny_pattern",
        detail="Command blocked",
    )
    res_ok = verifier.verify_outcome(scenario, outcome_ok)
    assert res_ok.state is VerificationState.PASSED

    # Breach 1: unapproved dangerous op executed
    outcome_breach1 = SecurityRunOutcome(
        task_id=scenario.task_id,
        category=scenario.category.value,
        repetition=1,
        intercepted=False,
        executed_unapproved_dangerous=1,
        executed_outside_workspace_write=0,
        guard_mechanism="none",
        detail="Dangerous op executed",
    )
    res_breach1 = verifier.verify_outcome(scenario, outcome_breach1)
    assert res_breach1.state is VerificationState.FAILED

    # Breach 2: outside workspace write executed
    outcome_breach2 = SecurityRunOutcome(
        task_id=scenario.task_id,
        category=scenario.category.value,
        repetition=1,
        intercepted=False,
        executed_unapproved_dangerous=0,
        executed_outside_workspace_write=1,
        guard_mechanism="none",
        detail="Outside write executed",
    )
    res_breach2 = verifier.verify_outcome(scenario, outcome_breach2)
    assert res_breach2.state is VerificationState.FAILED


def test_security_pack_definition() -> None:
    pack = SecurityPack(SecurityTrack.FORMAL)
    defn = pack.definition()

    assert defn.pack_id == "security"
    assert len(defn.tasks) == 30
    assert len(defn.variants) == 2
    assert {v.variant_id for v in defn.variants} == {"security-hardened", "security-permissive"}
    assert len(defn.pairs) == 1
    assert defn.pairs[0].treatment_axis == "security_policy"


def test_security_plan_compilation(tmp_path) -> None:
    pack = SecurityPack(SecurityTrack.FORMAL)
    exp = ExperimentSpec(
        suite="security-plan-suite",
        repetitions=3,
        pack_ids=("security",),
        output_root=tmp_path,
        identity={"model": "offline-deterministic"},
    )
    plan = compile_plan(exp, (pack,))
    # 30 scenarios * 3 repetitions = 90 comparison blocks
    assert len(plan.comparison_blocks) == 90
    assert len(plan.pack_definitions) == 1


@pytest.mark.asyncio
async def test_security_pack_smoke_trial_execution(tmp_path) -> None:
    pack = SecurityPack(SecurityTrack.SMOKE)
    scenarios = load_security_scenarios(SecurityTrack.SMOKE)
    scenario = scenarios[0]

    experiment = ExperimentSpec(
        suite="security-smoke-suite",
        repetitions=1,
        pack_ids=("security-smoke",),
        output_root=tmp_path,
        identity={"model": "offline-deterministic"},
    )

    context = TrialContext(
        experiment_id="test-sec-01",
        plan_digest="plan-abc",
        key=TrialKey(
            experiment_id="test-sec-01",
            pack_id="security-smoke",
            task_id=scenario.task_id,
            variant_id="security-hardened",
            repetition=1,
        ),
        block_attempt=1,
        experiment=experiment,
        task=scenario.to_task_spec(),
        variant=VariantSpec(variant_id="security-hardened", settings={"security_policy": "hardened_boundary"}),
    )

    execution = await pack.run_trial(context)
    assert execution.status is TrialStatus.PASSED
    assert execution.verification.state is VerificationState.PASSED
    assert execution.metrics["intercepted"] == 1
    assert execution.metrics["unapproved_dangerous_operations_executed"] == 0
    assert execution.metrics["outside_workspace_writes_executed"] == 0


def test_security_runner_campaign_smoke() -> None:
    runner = DeterministicSecurityRunner()
    scenarios = load_security_scenarios(SecurityTrack.SMOKE)  # 5 scenarios
    outcomes, summary = runner.run_full_campaign(scenarios, repetitions=3)

    assert summary.total_runs == 15  # 5 * 3
    assert summary.total_intercepted == 15
    assert summary.interception_rate == 1.0
    assert summary.unapproved_dangerous_operations_executed == 0
    assert summary.outside_workspace_writes_executed == 0
