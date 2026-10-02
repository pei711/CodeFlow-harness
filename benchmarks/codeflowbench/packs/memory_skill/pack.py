"""Frozen historical Memory/Skill Pack definition."""

from __future__ import annotations

from typing import Protocol

from benchmarks.codeflowbench.canonical import canonical_digest
from benchmarks.codeflowbench.protocol import (
    RetrievalContext,
    RetrievalExecution,
    TrialContext,
    TrialExecution,
)
from benchmarks.codeflowbench.schema import PackDefinition, PairSpec, VariantSpec

from .fixtures import (
    calibration_memory_corpus,
    calibration_retrieval_suites,
    calibration_skill_corpus,
    calibration_task_specs,
    formal_memory_corpus,
    formal_retrieval_suites,
    formal_skill_corpus,
    formal_task_specs,
    retrieval_fixture_manifest,
)
from .retrieval import ProductRetrievalAdapter


class CrossSessionRunner(Protocol):
    kind: str

    async def run(self, context: TrialContext) -> TrialExecution: ...


class MemorySkillPack:
    retrieval_cost_mode = "local_zero_provider"

    def __init__(
        self,
        runner: CrossSessionRunner,
        *,
        definition_kind: str = "formal",
    ) -> None:
        self._runner = runner
        if definition_kind == "formal":
            self._pack_id = "memory-skill-v1"
            self._tasks = formal_task_specs()
            self._retrieval_suites = formal_retrieval_suites()
            memory_corpus = formal_memory_corpus()
            skill_corpus = formal_skill_corpus()
        elif definition_kind == "calibration":
            self._pack_id = "memory-skill-calibration-v1"
            self._tasks = calibration_task_specs()
            self._retrieval_suites = calibration_retrieval_suites()
            memory_corpus = calibration_memory_corpus()
            skill_corpus = calibration_skill_corpus()
        else:
            raise ValueError(f"unknown definition_kind: {definition_kind}")
        self._memory_corpus = memory_corpus
        self._skill_corpus = skill_corpus
        self._retrieval = ProductRetrievalAdapter(
            memory_corpus=memory_corpus,
            skill_corpus=skill_corpus,
        )

    def definition(self) -> PackDefinition:
        invariant_settings = {
            "host_memory_rendering_digest": canonical_digest("host-user-memory-v1"),
            "curator_memory_config_digest": canonical_digest("curator-memory-enabled-v1"),
        }
        return PackDefinition(
            pack_id=self._pack_id,
            tasks=self._tasks,
            variants=(
                VariantSpec(
                    variant_id="user_memory_off",
                    settings={
                        "user_memory_recall": "disabled",
                        "skill_sources": ["local", "everos"],
                        **invariant_settings,
                    },
                ),
                VariantSpec(
                    variant_id="user_memory_on_local_only",
                    settings={
                        "user_memory_recall": "enabled",
                        "skill_sources": ["local"],
                        **invariant_settings,
                    },
                ),
                VariantSpec(
                    variant_id="user_memory_on_local_plus_everos",
                    settings={
                        "user_memory_recall": "enabled",
                        "skill_sources": ["local", "everos"],
                        **invariant_settings,
                    },
                ),
            ),
            pairs=(
                PairSpec(
                    treatment_axis="user_memory_recall",
                    control_variant_id="user_memory_off",
                    treatment_variant_id="user_memory_on_local_plus_everos",
                ),
                PairSpec(
                    treatment_axis="skill_sources",
                    control_variant_id="user_memory_on_local_only",
                    treatment_variant_id="user_memory_on_local_plus_everos",
                ),
            ),
            retrieval_suites=self._retrieval_suites,
            identity={
                "runner_kind": self._runner.kind,
                "claim_reducer": "memory_skill_v1",
                "retrieval_evidence_level": "deterministic_contract",
                "everos_semantic_quality_claim_eligible": False,
                "retrieval_paid_provider_calls": 0,
                "retrieval_embedding_calls": 0,
                "retrieval_fixture_digest": canonical_digest(retrieval_fixture_manifest()),
                "memory_corpus_items": len(self._memory_corpus),
                "skill_corpus_items": len(self._skill_corpus),
                "memory_corpus_digest": canonical_digest(self._memory_corpus),
                "skill_corpus_digest": canonical_digest(self._skill_corpus),
            },
        )

    async def run_trial(self, context: TrialContext) -> TrialExecution:
        return await self._runner.run(context)

    async def run_retrieval_case(
        self,
        context: RetrievalContext,
    ) -> RetrievalExecution:
        return await self._retrieval.run(context)


def create_formal_pack(runner: CrossSessionRunner) -> MemorySkillPack:
    return MemorySkillPack(runner)


def create_calibration_pack(
    runner: CrossSessionRunner,
) -> MemorySkillPack:
    return MemorySkillPack(runner, definition_kind="calibration")
