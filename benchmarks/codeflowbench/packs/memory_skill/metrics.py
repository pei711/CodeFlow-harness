"""Reducers for the frozen historical Memory/Skill evidence schema."""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.codeflowbench.canonical import canonical_digest
from benchmarks.codeflowbench.protocol import RetrievalContext, RetrievalExecution
from benchmarks.codeflowbench.records import RetrievalCaseKey, RetrievalStatus
from benchmarks.codeflowbench.schema import ExperimentSpec

from .fixtures import anonymous_item_id
from .pack import MemorySkillPack


@dataclass(frozen=True)
class RetrievalMeasurement:
    retrieval_suite_id: str
    query_id: str
    configuration_id: str
    label: str
    expected_item_ids: tuple[str, ...]
    execution: RetrievalExecution


@dataclass(frozen=True)
class MemorySkillRetrievalSummary:
    planned_cases: int
    measurable_cases: int
    memory_backend_recall_at_1: float
    memory_backend_recall_at_5: float
    memory_final_injection_recall_at_5: float
    memory_final_injection_precision_at_5: float
    memory_mrr_at_5: float
    memory_hard_negative_injection_rate: float
    memory_stale_injection_count: int
    memory_cross_workspace_leakage_count: int
    memory_contract_gate_passed: bool
    memory_positive_claim_eligible: bool
    skill_local_recall_at_5: float
    skill_everos_recall_at_5: float
    skill_fused_recall_at_5: float
    skill_fused_mrr_at_5: float
    skill_hard_negative_injection_rate: float
    skill_cross_workspace_leakage_count: int
    skill_source_contribution: dict[str, int]
    skill_contract_gate_passed: bool
    skill_positive_claim_eligible: bool


async def run_retrieval_micro_suite(
    pack: MemorySkillPack,
    experiment: ExperimentSpec,
) -> tuple[RetrievalMeasurement, ...]:
    experiment_id = canonical_digest(
        {
            "experiment": experiment.canonical_payload(),
            "pack": pack.definition(),
        }
    )
    measurements: list[RetrievalMeasurement] = []
    for suite in pack.definition().retrieval_suites:
        for query in suite.queries:
            for configuration in suite.configurations:
                execution = await pack.run_retrieval_case(
                    RetrievalContext(
                        experiment_id=experiment_id,
                        plan_digest=experiment_id,
                        key=RetrievalCaseKey(
                            experiment_id=experiment_id,
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
                measurements.append(
                    RetrievalMeasurement(
                        retrieval_suite_id=suite.retrieval_suite_id,
                        query_id=query.query_id,
                        configuration_id=configuration.configuration_id,
                        label=query.label,
                        expected_item_ids=query.expected_item_ids,
                        execution=execution,
                    )
                )
    return tuple(measurements)


def summarize_retrieval(
    measurements: tuple[RetrievalMeasurement, ...],
) -> MemorySkillRetrievalSummary:
    memory = tuple(
        measurement for measurement in measurements if measurement.retrieval_suite_id == "user-memory-retrieval-v1"
    )
    skill = tuple(
        measurement for measurement in measurements if measurement.retrieval_suite_id == "skill-source-fusion-v1"
    )
    memory_positive = tuple(item for item in memory if item.label == "positive")
    memory_negative = tuple(item for item in memory if item.label == "hard_negative")
    skill_positive = tuple(item for item in skill if item.label == "positive")
    skill_negative_fused = tuple(
        item for item in skill if item.label == "hard_negative" and item.configuration_id == "fused"
    )

    memory_ranked_recall_1 = _mean_recall(memory_positive, injected=False, limit=1)
    memory_ranked_recall_5 = _mean_recall(memory_positive, injected=False, limit=5)
    memory_injected_recall_5 = _mean_recall(memory_positive, injected=True, limit=5)
    memory_precision = _mean_precision(memory_positive)
    memory_mrr = _mean_mrr(memory_positive, configuration_id="user_memory_on")
    memory_hard_negative_rate = _hard_negative_rate(memory_negative)
    stale_ids = {anonymous_item_id("user-memory-retrieval-v1", f"memory-stale-{index:03d}") for index in range(50)}
    memory_cross_ids = {
        anonymous_item_id("user-memory-retrieval-v1", f"memory-cross-{index:03d}") for index in range(30)
    }
    memory_stale_count = _injected_id_count(memory, stale_ids)
    memory_cross_count = _injected_id_count(memory, memory_cross_ids)
    memory_coverage = len(memory) == 80 and all(item.execution.status is RetrievalStatus.MEASURABLE for item in memory)
    memory_eligible = (
        memory_coverage
        and memory_injected_recall_5 >= 0.80
        and memory_hard_negative_rate <= 0.05
        and memory_stale_count == 0
        and memory_cross_count == 0
    )

    skill_local = _mean_recall(
        tuple(item for item in skill_positive if item.configuration_id == "local_only"),
        injected=True,
        limit=5,
    )
    skill_everos = _mean_recall(
        tuple(item for item in skill_positive if item.configuration_id == "everos_only"),
        injected=True,
        limit=5,
    )
    skill_fused = _mean_recall(
        tuple(item for item in skill_positive if item.configuration_id == "fused"),
        injected=True,
        limit=5,
    )
    skill_mrr = _mean_mrr(skill_positive, configuration_id="fused")
    skill_hard_negative_rate = _hard_negative_rate(skill_negative_fused)
    skill_cross_ids = {anonymous_item_id("skill-source-fusion-v1", f"skill-cross-{index:03d}") for index in range(20)}
    skill_cross_count = _injected_id_count(skill, skill_cross_ids)
    contribution = {"local": 0, "everos": 0}
    for measurement in skill:
        if measurement.configuration_id != "fused":
            continue
        for result in measurement.execution.injected_results:
            for source in result["contributing_sources"]:
                contribution[source] += 1
    skill_coverage = len(skill) == 180 and all(item.execution.status is RetrievalStatus.MEASURABLE for item in skill)
    skill_eligible = (
        skill_coverage
        and skill_fused - max(skill_local, skill_everos) >= 0.05
        and skill_hard_negative_rate <= 0.05
        and skill_cross_count == 0
    )
    return MemorySkillRetrievalSummary(
        planned_cases=len(measurements),
        measurable_cases=sum(
            measurement.execution.status is RetrievalStatus.MEASURABLE for measurement in measurements
        ),
        memory_backend_recall_at_1=memory_ranked_recall_1,
        memory_backend_recall_at_5=memory_ranked_recall_5,
        memory_final_injection_recall_at_5=memory_injected_recall_5,
        memory_final_injection_precision_at_5=memory_precision,
        memory_mrr_at_5=memory_mrr,
        memory_hard_negative_injection_rate=memory_hard_negative_rate,
        memory_stale_injection_count=memory_stale_count,
        memory_cross_workspace_leakage_count=memory_cross_count,
        memory_contract_gate_passed=memory_eligible,
        memory_positive_claim_eligible=False,
        skill_local_recall_at_5=skill_local,
        skill_everos_recall_at_5=skill_everos,
        skill_fused_recall_at_5=skill_fused,
        skill_fused_mrr_at_5=skill_mrr,
        skill_hard_negative_injection_rate=skill_hard_negative_rate,
        skill_cross_workspace_leakage_count=skill_cross_count,
        skill_source_contribution=contribution,
        skill_contract_gate_passed=skill_eligible,
        skill_positive_claim_eligible=False,
    )


def _mean_recall(
    measurements: tuple[RetrievalMeasurement, ...],
    *,
    injected: bool,
    limit: int,
) -> float:
    if not measurements:
        return 0.0
    values = []
    for measurement in measurements:
        results = measurement.execution.injected_results if injected else measurement.execution.ranked_results
        observed = {str(result["item_id"]) for result in results[:limit]}
        expected = set(measurement.expected_item_ids)
        values.append(len(observed & expected) / len(expected) if expected else 1.0)
    return sum(values) / len(values)


def _mean_precision(measurements: tuple[RetrievalMeasurement, ...]) -> float:
    if not measurements:
        return 0.0
    values = []
    for measurement in measurements:
        observed = {str(result["item_id"]) for result in measurement.execution.injected_results[:5]}
        expected = set(measurement.expected_item_ids)
        values.append(len(observed & expected) / len(observed) if observed else 0.0)
    return sum(values) / len(values)


def _mean_mrr(
    measurements: tuple[RetrievalMeasurement, ...],
    *,
    configuration_id: str,
) -> float:
    selected = tuple(
        measurement
        for measurement in measurements
        if measurement.configuration_id == configuration_id and measurement.label == "positive"
    )
    if not selected:
        return 0.0
    reciprocal_ranks = []
    for measurement in selected:
        expected = set(measurement.expected_item_ids)
        rank = next(
            (
                index
                for index, result in enumerate(
                    measurement.execution.injected_results[:5],
                    start=1,
                )
                if result["item_id"] in expected
            ),
            None,
        )
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def _hard_negative_rate(measurements: tuple[RetrievalMeasurement, ...]) -> float:
    if not measurements:
        return 0.0
    injected = sum(bool(measurement.execution.injected_results) for measurement in measurements)
    return injected / len(measurements)


def _injected_id_count(
    measurements: tuple[RetrievalMeasurement, ...],
    forbidden_ids: set[str],
) -> int:
    return sum(
        result["item_id"] in forbidden_ids
        for measurement in measurements
        for result in measurement.execution.injected_results
    )
