"""Historical Memory/Skill E2E artifact contract retained after removal."""

from __future__ import annotations

from pathlib import Path

from benchmarks.codeflowbench.packs.memory_skill import (
    DeterministicCrossProcessRunner,
    create_formal_pack,
)
from benchmarks.codeflowbench.protocol import TrialContext
from benchmarks.codeflowbench.records import TrialKey, TrialStatus
from benchmarks.codeflowbench.schema import ExperimentSpec


async def test_memory_skill_variants_use_fresh_runtime_processes(
    tmp_path: Path,
) -> None:
    pack = create_formal_pack(DeterministicCrossProcessRunner())
    definition = pack.definition()
    task = definition.tasks[4]
    experiment = ExperimentSpec(
        suite="memory-skill-runtime-e2e",
        repetitions=1,
        pack_ids=(definition.pack_id,),
        output_root=tmp_path,
        identity={
            "codeflow_commit": "d" * 40,
            "provider": "scripted",
            "model": "scripted/codeflowbench-memory-skill",
        },
    )
    executions = {}
    for variant in definition.variants:
        context = TrialContext(
            experiment_id="memory-skill-runtime-e2e",
            plan_digest="e" * 64,
            key=TrialKey(
                experiment_id="memory-skill-runtime-e2e",
                pack_id=definition.pack_id,
                task_id=task.task_id,
                variant_id=variant.variant_id,
                repetition=0,
            ),
            block_attempt=1,
            experiment=experiment,
            task=task,
            variant=variant,
        )
        executions[variant.variant_id] = await pack.run_trial(context)

    memory_off = executions["user_memory_off"]
    local_only = executions["user_memory_on_local_only"]
    full = executions["user_memory_on_local_plus_everos"]

    assert memory_off.status is TrialStatus.TASK_FAILED
    assert memory_off.metrics["memory.user_recall_calls"] == 0
    assert memory_off.metrics["memory.suppressed_user_recall_calls"] > 0
    assert memory_off.metrics["provider.memory_observed"] is False
    assert memory_off.metrics["provider.skill_observed"] is True

    assert local_only.status is TrialStatus.TASK_FAILED
    assert local_only.metrics["memory.user_recall_calls"] > 0
    assert local_only.metrics["provider.memory_observed"] is True
    assert local_only.metrics["provider.skill_observed"] is False
    assert local_only.metrics["skill.source_contribution"]["everos"] == 0

    assert full.status is TrialStatus.PASSED
    assert full.verification.state.value == "passed"
    assert full.metrics["runtime.fresh_process"] is True
    assert full.metrics["runtime.backend_quiescent"] is True
    assert full.metrics["memory.backend_class"] == "EverosBackend"
    assert full.metrics["memory.backend_adapter"] == "injected_fixture"
    assert full.metrics["memory.everos_semantic_quality_claim_eligible"] is False
    assert full.metrics["provider.memory_observed"] is True
    assert full.metrics["provider.skill_observed"] is True
    assert full.metrics["provider.kind"] == "scripted/codeflowbench-memory-skill"
    assert full.metrics["paid_campaign_eligible"] is False
    assert full.metrics["real_agent_task_effect_claim_eligible"] is False
    assert full.metrics["cost.complete"] is True
    assert full.metrics["cost.estimated_cny"] == 0.0
    assert full.metrics["skill.source_contribution"]["everos"] > 0
    assert full.metrics["usage.complete"] is True
    assert (
        len(
            {
                memory_off.artifact_refs[0],
                local_only.artifact_refs[0],
                full.artifact_refs[0],
            }
        )
        == 3
    )
