"""Historical EverOS task-effect tests retained fail-closed after removal."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from benchmarks.codeflowbench.packs.memory_skill import (
    semantic_effect as semantic_effect_module,
)
from benchmarks.codeflowbench.packs.memory_skill.semantic_effect import (
    ScriptedSemanticMemoryEffectRunner,
    _global_authority_proof_valid,
    _HashBagEmbeddingProvider,
    _verify_result,
    create_semantic_memory_effect_calibration_pack,
    create_semantic_memory_effect_pack,
    load_semantic_memory_effect_tasks,
    reduce_semantic_memory_effect_claims,
)
from benchmarks.codeflowbench.protocol import TrialContext
from benchmarks.codeflowbench.records import (
    TrialKey,
    TrialStatus,
    VerificationState,
)
from benchmarks.codeflowbench.schema import ExperimentSpec


def test_semantic_memory_effect_manifests_are_frozen_and_disjoint() -> None:
    calibration = load_semantic_memory_effect_tasks("calibration")
    formal = load_semantic_memory_effect_tasks("formal")

    assert len(calibration) == 2
    assert len(formal) == 8
    assert {task.task_id for task in calibration}.isdisjoint(task.task_id for task in formal)
    assert all(task.distractor_memories for task in (*calibration, *formal))
    assert len({task.expected_approval_code for task in (*calibration, *formal)}) == 10
    assert all(task.expected_approval_code not in task.evaluation_request for task in (*calibration, *formal))


def test_semantic_memory_effect_definition_changes_one_axis() -> None:
    runner = ScriptedSemanticMemoryEffectRunner()
    formal = create_semantic_memory_effect_pack(runner).definition()
    calibration = create_semantic_memory_effect_calibration_pack(
        runner,
    ).definition()

    assert formal.pack_id == "semantic-memory-effect-v1"
    assert calibration.pack_id == ("semantic-memory-effect-calibration-v1")
    assert len(formal.tasks) == 8
    assert len(calibration.tasks) == 2
    assert formal.identity["claim_reducer"] == "memory_effect_v1"
    assert formal.identity["automatic_memory_extraction_claimed"] is False
    assert not formal.retrieval_suites
    assert [variant.variant_id for variant in formal.variants] == [
        "user_memory_off",
        "user_memory_on",
    ]
    pair = formal.pairs[0]
    assert pair.treatment_axis == "user_memory_recall"
    control, treatment = formal.variants
    actual_diff = {
        key
        for key in set(control.settings) | set(treatment.settings)
        if control.settings.get(key) != treatment.settings.get(key)
    }
    assert actual_diff == {"user_memory_recall"}


def test_semantic_memory_effect_verifier_rejects_non_object_json(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "result.json").write_text(
        json.dumps(["not", "an", "object"]),
        encoding="utf-8",
    )

    verification = _verify_result(
        workspace,
        load_semantic_memory_effect_tasks("calibration")[0],
    )

    assert verification.state is VerificationState.FAILED
    assert verification.findings == ("result_artifact_mismatch",)
    assert verification.metrics == {
        "region_matched": False,
        "retention_matched": False,
        "approval_code_matched": False,
    }


@pytest.mark.asyncio
async def test_fake_embedding_preserves_semantic_neighborhood() -> None:
    embedder = _HashBagEmbeddingProvider()
    task = load_semantic_memory_effect_tasks("calibration")[0]
    query = task.evaluation_request
    target, distractor = await embedder.embed_batch(
        [task.memory_text, task.distractor_memories[0]],
    )
    query_vector = await embedder.embed(query)

    assert _dot(query_vector, target) > _dot(query_vector, distractor)


@pytest.mark.asyncio
async def test_historical_campaign_fails_closed_without_removed_runtime(
    tmp_path: Path,
) -> None:
    pack = create_semantic_memory_effect_calibration_pack(
        ScriptedSemanticMemoryEffectRunner(),
    )
    definition = pack.definition()
    experiment = ExperimentSpec(
        suite="semantic-memory-effect-test",
        repetitions=1,
        pack_ids=(definition.pack_id,),
        output_root=tmp_path,
        identity={"test": "semantic-memory-effect"},
    )
    task = definition.tasks[0]
    by_variant = {variant.variant_id: variant for variant in definition.variants}

    off = await pack.run_trial(
        _trial_context(
            experiment=experiment,
            task=task,
            variant=by_variant["user_memory_off"],
        )
    )
    on = await pack.run_trial(
        _trial_context(
            experiment=experiment,
            task=task,
            variant=by_variant["user_memory_on"],
        )
    )

    assert off.status is TrialStatus.INFRASTRUCTURE_FAILURE
    assert on.status is TrialStatus.INFRASTRUCTURE_FAILURE
    assert off.verification.state is VerificationState.NOT_RUN
    assert on.verification.state is VerificationState.NOT_RUN


@pytest.mark.asyncio
async def test_missing_historical_runtime_precedes_budget_classification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pack = create_semantic_memory_effect_calibration_pack(
        ScriptedSemanticMemoryEffectRunner(),
    )
    definition = pack.definition()
    experiment = ExperimentSpec(
        suite="semantic-memory-budget-exhaustion",
        repetitions=1,
        pack_ids=(definition.pack_id,),
        output_root=tmp_path,
        identity={"test": "semantic-memory-budget-exhaustion"},
    )
    original_run_worker_stage = semantic_effect_module._run_worker_stage

    async def budget_exhausted_worker_stage(**kwargs):
        result = await original_run_worker_stage(**kwargs)
        if kwargs["stage"] == "evaluation":
            result["failure_category"] = "task_budget_exhausted"
            result["usage"] = {
                "calls": 1,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "usage_complete": False,
            }
        return result

    monkeypatch.setattr(
        semantic_effect_module,
        "_run_worker_stage",
        budget_exhausted_worker_stage,
    )
    task = definition.tasks[0]
    variant = definition.variants[0]

    execution = await pack.run_trial(
        _trial_context(
            experiment=experiment,
            task=task,
            variant=variant,
        )
    )

    assert execution.status is TrialStatus.INFRASTRUCTURE_FAILURE
    assert execution.verification.state is VerificationState.NOT_RUN


def test_semantic_memory_effect_reducer_accepts_formal_production_evidence() -> None:
    trials, pairs = _semantic_effect_artifacts(calibration=False)

    metrics = reduce_semantic_memory_effect_claims(trials, pairs)

    assert metrics["semantic_memory_e2e.planned_pairs"] == 24
    assert metrics["semantic_memory_e2e.valid_pairs"] == 24
    assert metrics["semantic_memory_e2e.coverage_valid"] is True
    assert metrics["semantic_memory_e2e.attempt_consistency_valid"] is True
    assert metrics["semantic_memory_e2e.variant_axis_valid"] is True
    assert metrics["semantic_memory_e2e.control_passes"] == 0
    assert metrics["semantic_memory_e2e.treatment_passes"] == 24
    assert metrics["semantic_memory_e2e.control_pass_rate"] == 0.0
    assert metrics["semantic_memory_e2e.treatment_pass_rate"] == 1.0
    assert metrics["semantic_memory_e2e.success_delta_pp"] == 100.0
    assert metrics["semantic_memory_e2e.net_verifier_gains"] == 24
    assert metrics["semantic_memory_e2e.positive_tasks"] == 8
    assert metrics["semantic_memory_e2e.tasks_with_two_of_three_regressions"] == 0
    assert metrics["semantic_memory_e2e.memory_off_backend_call_count"] == 0
    assert metrics["semantic_memory_e2e.memory_off_backend_calls_zero"] is True
    assert metrics["semantic_memory_e2e.treatment_target_hits"] == 24
    assert metrics["semantic_memory_e2e.treatment_target_hits_complete"] is True
    assert metrics["semantic_memory_e2e.production_real_agent_evidence_valid"] is True
    assert metrics["semantic_memory_e2e.production_embedding_evidence_valid"] is True
    assert metrics["semantic_memory_e2e.production_cost_evidence_valid"] is True
    assert metrics["semantic_memory_e2e.production_model_evidence_valid"] is True
    assert metrics["semantic_memory_e2e.claim_contract_valid"] is True
    assert metrics["semantic_memory_e2e.real_agent_task_effect_claim_eligible"] is True
    assert metrics["semantic_memory_effect.measurement_valid"] is True
    assert metrics["semantic_memory_e2e.measurement_valid"] is True


def test_semantic_memory_effect_calibration_is_measurable_not_claimable() -> None:
    trials, pairs = _semantic_effect_artifacts(calibration=True)

    metrics = reduce_semantic_memory_effect_claims(trials, pairs)

    assert metrics["semantic_memory_e2e.planned_pairs"] == 4
    assert metrics["semantic_memory_e2e.valid_pairs"] == 4
    assert metrics["semantic_memory_e2e.net_verifier_gains"] == 4
    assert metrics["semantic_memory_e2e.positive_tasks"] == 2
    assert metrics["semantic_memory_e2e.coverage_valid"] is True
    assert metrics["semantic_memory_e2e.claim_contract_valid"] is False
    assert metrics["semantic_memory_e2e.real_agent_task_effect_claim_eligible"] is False
    assert metrics["semantic_memory_effect.measurement_valid"] is True
    assert metrics["semantic_memory_e2e.measurement_valid"] is True


@pytest.mark.parametrize(
    "tamper",
    (
        "attempt",
        "membership",
        "plan_digest",
        "axis",
        "duplicate_pair",
    ),
)
def test_semantic_memory_effect_reducer_rejects_pair_integrity_drift(
    tamper: str,
) -> None:
    trials, pairs = _semantic_effect_artifacts(calibration=False)
    first_pair = pairs[0]
    pair_key = first_pair["key"]
    arms = [
        trial
        for trial in trials
        if trial["key"]["task_id"] == pair_key["task_id"] and trial["key"]["repetition"] == pair_key["repetition"]
    ]
    if tamper == "attempt":
        arms[0]["selected_block_attempt"] = 2
    elif tamper == "membership":
        arms[0]["pair_memberships"] = []
    elif tamper == "plan_digest":
        arms[0]["plan_digest"] = "d" * 64
    elif tamper == "axis":
        treatment = next(arm for arm in arms if arm["key"]["variant_id"] == "user_memory_on")
        treatment["observed_variant_settings"]["unexpected_axis"] = "enabled"
        first_pair["actual_variant_diff"]["unexpected_axis"] = {
            "control": None,
            "treatment": "enabled",
        }
    elif tamper == "duplicate_pair":
        pairs.append(
            {
                **first_pair,
                "key": dict(first_pair["key"]),
                "actual_variant_diff": dict(
                    first_pair["actual_variant_diff"],
                ),
            }
        )

    metrics = reduce_semantic_memory_effect_claims(trials, pairs)

    assert metrics["semantic_memory_effect.measurement_valid"] is False
    assert metrics["semantic_memory_e2e.real_agent_task_effect_claim_eligible"] is False
    if tamper == "axis":
        assert metrics["semantic_memory_e2e.variant_axis_valid"] is False
    else:
        assert metrics["semantic_memory_e2e.attempt_consistency_valid"] is False


@pytest.mark.parametrize(
    ("metric_name", "replacement", "validity_metric"),
    (
        (
            "memory.user_recall_calls",
            1,
            "semantic_memory_e2e.memory_off_backend_calls_zero",
        ),
        (
            "memory.target_hit",
            False,
            "semantic_memory_e2e.treatment_target_hits_complete",
        ),
        (
            "embedding.real_provider",
            False,
            "semantic_memory_e2e.production_embedding_evidence_valid",
        ),
        (
            "cost.complete",
            False,
            "semantic_memory_e2e.production_cost_evidence_valid",
        ),
        (
            "cost.global_authority_proof_valid",
            False,
            "semantic_memory_e2e.production_cost_evidence_valid",
        ),
        (
            "provider.actual_models",
            [],
            "semantic_memory_e2e.production_model_evidence_valid",
        ),
        (
            "provider.actual_models",
            ["different-model"],
            "semantic_memory_e2e.production_model_evidence_valid",
        ),
    ),
)
def test_semantic_memory_effect_reducer_rejects_ineligible_evidence(
    metric_name: str,
    replacement,
    validity_metric: str,
) -> None:
    trials, pairs = _semantic_effect_artifacts(calibration=False)
    variant_id = "user_memory_off" if metric_name == "memory.user_recall_calls" else "user_memory_on"
    trial = next(row for row in trials if row["key"]["variant_id"] == variant_id)
    trial["metrics"][metric_name] = replacement

    metrics = reduce_semantic_memory_effect_claims(trials, pairs)

    assert metrics["semantic_memory_effect.measurement_valid"] is True
    assert metrics[validity_metric] is False
    assert metrics["semantic_memory_e2e.real_agent_task_effect_claim_eligible"] is False


def test_semantic_memory_effect_reducer_reports_two_of_three_regression() -> None:
    trials, pairs = _semantic_effect_artifacts(calibration=False)
    first_task_id = pairs[0]["key"]["task_id"]
    for trial in trials:
        if trial["key"]["task_id"] == first_task_id and trial["key"]["repetition"] in {0, 1}:
            trial["status"] = "passed" if trial["key"]["variant_id"] == "user_memory_off" else "task_failed"

    metrics = reduce_semantic_memory_effect_claims(trials, pairs)

    assert metrics["semantic_memory_e2e.tasks_with_two_of_three_regressions"] == 1
    assert metrics["semantic_memory_e2e.no_two_of_three_regressions"] is False
    assert metrics["semantic_memory_e2e.claim_contract_valid"] is False


def test_semantic_memory_effect_global_budget_uses_one_authority() -> None:
    snapshot = SimpleNamespace(
        hard_cap_cny=100.0,
        external_service_reserve_cny=5.0,
        provider_charged_cny=95.0,
        total_committed_cny=100.0,
        accounting_complete=True,
        open_reservations=0,
    )

    assert _global_authority_proof_valid(snapshot) is True

    snapshot.provider_charged_cny = 95.01
    snapshot.total_committed_cny = 100.01
    assert _global_authority_proof_valid(snapshot) is False

    snapshot.provider_charged_cny = 95.0
    snapshot.total_committed_cny = 95.0
    snapshot.external_service_reserve_cny = 0.0
    assert _global_authority_proof_valid(snapshot) is False


def test_semantic_memory_effect_budget_contract_totals_cny100() -> None:
    suite_root = Path(__file__).resolve().parents[1] / "benchmarks" / "codeflowbench" / "suites"
    main = yaml.safe_load(
        (suite_root / "agent_application_ship_1.yaml").read_text(
            encoding="utf-8",
        )
    )
    semantic = yaml.safe_load((suite_root / "agent_application_ship_1_semantic.yaml").read_text(encoding="utf-8"))
    main_cap = float(main["budget"]["hard_cap_cny"])
    main_reserve = float(
        main["budget"]["external_service_reserve_cny"],
    )
    semantic_cap = float(semantic["budget"]["hard_cap_cny"])

    assert main_cap == 100.0
    assert main_reserve == 5.0
    assert semantic_cap == main_reserve
    assert main_cap - main_reserve == 95.0
    assert main_cap - main_reserve + semantic_cap == 100.0


def _semantic_effect_artifacts(
    *,
    calibration: bool,
) -> tuple[list[dict], list[dict]]:
    pack = (
        create_semantic_memory_effect_calibration_pack(
            ScriptedSemanticMemoryEffectRunner(),
        )
        if calibration
        else create_semantic_memory_effect_pack(
            ScriptedSemanticMemoryEffectRunner(),
        )
    )
    definition = pack.definition()
    repetitions = 2 if calibration else 3
    experiment_id = "e" * 64
    plan_digest = "a" * 64
    pair_spec = definition.pairs[0]
    variants = {variant.variant_id: variant for variant in definition.variants}
    trials: list[dict] = []
    pairs: list[dict] = []
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
            for variant_id in (
                pair_spec.control_variant_id,
                pair_spec.treatment_variant_id,
            ):
                treatment = variant_id == pair_spec.treatment_variant_id
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
                        "status": "passed" if treatment else "task_failed",
                        "selected_block_attempt": 1,
                        "observed_variant_settings": dict(
                            variants[variant_id].settings,
                        ),
                        "metrics": _production_effect_metrics(treatment),
                    }
                )
            control_settings = dict(
                variants[pair_spec.control_variant_id].settings,
            )
            treatment_settings = dict(
                variants[pair_spec.treatment_variant_id].settings,
            )
            actual_diff = {
                key: {
                    "control": control_settings.get(key),
                    "treatment": treatment_settings.get(key),
                }
                for key in sorted(
                    control_settings.keys() | treatment_settings.keys(),
                )
                if control_settings.get(key) != treatment_settings.get(key)
            }
            pairs.append(
                {
                    "key": pair_key,
                    "plan_digest": plan_digest,
                    "selected_block_attempt": 1,
                    "valid": True,
                    "actual_variant_diff": actual_diff,
                }
            )
    return trials, pairs


def _production_effect_metrics(treatment: bool) -> dict:
    return {
        "memory.user_recall_calls": 1 if treatment else 0,
        "memory.suppressed_user_recall_calls": 0 if treatment else 1,
        "memory.target_hit": treatment,
        "memory.backend_class": "EverosBackend",
        "memory.backend_adapter": "production",
        "memory.backend_adapter_class": "_RealEverosAdapter",
        "runtime.fresh_process": True,
        "embedding.provider_identity": "openai/text-embedding-v4",
        "embedding.provider_config_digest": "b" * 64,
        "embedding.real_provider": True,
        "paid_campaign_eligible": True,
        "usage.complete": True,
        "cost.agent_provider_complete": True,
        "cost.embedding_budget_complete": True,
        "cost.ledger_reconciliation_valid": True,
        "cost.global_authority_proof_valid": True,
        "cost.embedding_accounting_basis": ("shared_main_provider_ledger_conservative_proxy"),
        "cost.embedding_uses_shared_main_ledger": True,
        "cost.embedding_has_separate_authority": False,
        "cost.main_provider_spend_ceiling_cny": 95.0,
        "cost.semantic_addendum_hard_cap_cny": 5.0,
        "cost.global_authorized_hard_cap_cny": 100.0,
        "cost.complete": True,
        "provider.kind": "deepseek/deepseek-v4-flash",
        "provider.actual_models": ["deepseek-v4-flash"],
        "real_agent_task_effect_claim_eligible": True,
    }


def _trial_context(
    *,
    experiment: ExperimentSpec,
    task,
    variant,
) -> TrialContext:
    experiment_id = "semantic-effect-test"
    return TrialContext(
        experiment_id=experiment_id,
        plan_digest="f" * 64,
        key=TrialKey(
            experiment_id=experiment_id,
            pack_id=experiment.pack_ids[0],
            task_id=task.task_id,
            variant_id=variant.variant_id,
            repetition=0,
        ),
        block_attempt=1,
        experiment=experiment,
        task=task,
        variant=variant,
    )


def _dot(left: list[float], right: list[float]) -> float:
    return sum(first * second for first, second in zip(left, right, strict=True))
