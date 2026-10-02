"""Deterministic reconstruction of the historical Memory/Skill retrieval Pack."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from benchmarks.codeflowbench.protocol import RetrievalContext, RetrievalExecution
from benchmarks.codeflowbench.records import RetrievalStatus
from codeflow.context_engine.base import AssemblyContext
from codeflow.context_engine.segments import MemorySegmentBuilder, SkillsSegmentBuilder
from codeflow.memory_engine import Memory, TokenBudget
from codeflow.memory_engine.skill_forge import (
    LocalSkillSource,
    RouterHit,
    SkillForgeRouter,
)
from codeflow.memory_engine.skill_local import LocalPool, SkillMeta
from codeflow.utils.bm25 import BM25Okapi, tokenize

from .fixtures import anonymous_item_id
from .models import MemoryFact, SkillItem

_SEMANTIC_MARKER = re.compile(r"(?:ambertoken|saffrontoken)\d+")
_MEMORY_RAW_SCORE_THRESHOLD = 3.0


class ProductRetrievalAdapter:
    def __init__(
        self,
        *,
        memory_corpus: tuple[MemoryFact, ...],
        skill_corpus: tuple[SkillItem, ...],
    ) -> None:
        self._backend = _FixtureMemoryBackend(
            memory_corpus=memory_corpus,
            skill_corpus=skill_corpus,
        )
        self._skill_corpus = skill_corpus
        self._local_sources: dict[str, LocalSkillSource] = {}

    async def run(self, context: RetrievalContext) -> RetrievalExecution:
        suite_id = context.key.retrieval_suite_id
        if suite_id.startswith("user-memory-retrieval-"):
            return await self._run_memory(context)
        if suite_id.startswith("skill-source-fusion-"):
            return await self._run_skill(context)
        return RetrievalExecution(
            status=RetrievalStatus.INFRASTRUCTURE_FAILURE,
            findings=(f"unknown_retrieval_suite:{suite_id}",),
        )

    async def _run_memory(
        self,
        context: RetrievalContext,
    ) -> RetrievalExecution:
        query_text = str(context.query.payload["query_text"])
        workspace_id = str(context.query.payload["workspace_id"])
        consuming_turn = str(context.query.payload["consuming_turn"])
        before = self._backend.user_recall_calls
        builder = MemorySegmentBuilder(
            _EmptyMemoryStore(),
            self._backend,
            user_id=workspace_id,
            memory_top_k=5,
        )
        segment = await builder.build(_assembly_context(query_text, consuming_turn))
        recalled = self._backend.last_user_hits
        injected_ids = {
            str(hit.metadata["id"]) for hit in recalled if hit.text and segment is not None and segment.text
        }
        ranked = tuple(
            _selection(
                query_id=context.query.query_id,
                item_id=anonymous_item_id(
                    context.key.retrieval_suite_id,
                    str(hit.metadata["id"]),
                ),
                source="user_memory",
                rank=rank,
                raw_score=hit.score,
                rrf_score=None,
                contributing_sources=("user_memory",),
                injected=str(hit.metadata["id"]) in injected_ids,
                consuming_turn=consuming_turn,
            )
            for rank, hit in enumerate(recalled, start=1)
        )
        injected = tuple(result for result in ranked if result["injected"])
        return RetrievalExecution(
            status=RetrievalStatus.MEASURABLE,
            ranked_results=ranked,
            injected_results=injected,
            usage={
                "backend_recall_calls": self._backend.user_recall_calls - before,
                "provider_calls": 0,
                "embedding_calls": 0,
                "retrieval_evidence_level": "deterministic_contract",
                "backend_class": "EverosBackend",
                "backend_adapter": "injected_fixture",
                "everos_semantic_quality_claim_eligible": False,
            },
        )

    async def _run_skill(
        self,
        context: RetrievalContext,
    ) -> RetrievalExecution:
        configuration_id = context.configuration.configuration_id
        query_text = str(context.query.payload["query_text"])
        workspace_id = str(context.query.payload["workspace_id"])
        consuming_turn = str(context.query.payload["consuming_turn"])
        local = self._local_source(workspace_id)
        everos = _HistoricalSkillSource(
            backend=self._backend,
            agent_id=workspace_id,
        )
        if configuration_id == "local_only":
            sources = [local]
        elif configuration_id == "everos_only":
            sources = [everos]
        elif configuration_id == "fused":
            sources = [local, everos]
        else:
            return RetrievalExecution(
                status=RetrievalStatus.INFRASTRUCTURE_FAILURE,
                findings=(f"unknown_retrieval_configuration:{configuration_id}",),
            )
        router = SkillForgeRouter(sources=sources)
        before = self._backend.agent_recall_calls
        ranked_hits = await router.select(
            query=query_text,
            history=[],
            k=5,
        )
        segment = await SkillsSegmentBuilder(
            router,
            skill_top_k=5,
        ).build(_assembly_context(query_text, consuming_turn))
        injected_qualified_ids = set(segment.meta["injected_skill_ids"]) if segment is not None else set()
        ranked = tuple(
            _selection(
                query_id=context.query.query_id,
                item_id=anonymous_item_id(
                    context.key.retrieval_suite_id,
                    hit.name,
                ),
                source=("fused" if configuration_id == "fused" else configuration_id.removesuffix("_only")),
                rank=rank,
                raw_score=hit.score,
                rrf_score=(float(hit.meta["rrf_score"]) if configuration_id == "fused" else None),
                contributing_sources=tuple(
                    str(source)
                    for source in hit.meta.get(
                        "contributing_sources",
                        [configuration_id.removesuffix("_only")],
                    )
                ),
                injected=hit.qualified_id in injected_qualified_ids,
                consuming_turn=consuming_turn,
            )
            for rank, hit in enumerate(ranked_hits, start=1)
        )
        injected = tuple(result for result in ranked if result["injected"])
        return RetrievalExecution(
            status=RetrievalStatus.MEASURABLE,
            ranked_results=ranked,
            injected_results=injected,
            usage={
                "backend_agent_recall_calls": (self._backend.agent_recall_calls - before),
                "provider_calls": 0,
                "embedding_calls": 0,
                "semantic_fixture_calls": (self._backend.agent_recall_calls - before),
                "retrieval_evidence_level": "deterministic_contract",
                "backend_class": "EverosBackend",
                "backend_adapter": "injected_fixture",
                "everos_semantic_quality_claim_eligible": False,
            },
        )

    def _local_source(self, workspace_id: str) -> LocalSkillSource:
        existing = self._local_sources.get(workspace_id)
        if existing is not None:
            return existing
        metas = [
            SkillMeta(
                id=item.item_id,
                name=item.logical_id,
                description=item.text,
                path=Path("/codeflowbench") / item.item_id / "SKILL.md",
                content=item.text,
                source="local",
            )
            for item in self._skill_corpus
            if item.source == "local" and item.workspace_id == workspace_id
        ]
        registry = _FixtureSkillRegistry(metas)
        source = LocalSkillSource(LocalPool(registry), registry)
        self._local_sources[workspace_id] = source
        return source


class _FixtureMemoryBackend:
    def __init__(
        self,
        *,
        memory_corpus: tuple[MemoryFact, ...],
        skill_corpus: tuple[SkillItem, ...],
    ) -> None:
        self._memory_corpus = memory_corpus
        self._skill_corpus = skill_corpus
        self.user_recall_calls = 0
        self.agent_recall_calls = 0
        self.last_user_hits: list[Memory] = []

    async def recall(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        top_k: int,
    ) -> list[Memory]:
        if (user_id is None) == (agent_id is None):
            return []
        if user_id is not None:
            self.user_recall_calls += 1
            candidates = [
                fact
                for fact in self._memory_corpus
                if fact.workspace_id == user_id and fact.active and not fact.superseded
            ]
            if not candidates:
                self.last_user_hits = []
                return []
            index = BM25Okapi([tokenize(fact.text) for fact in candidates])
            scores = index.get_scores(tokenize(query))
            ranked = sorted(zip(candidates, scores), key=lambda item: (-item[1], item[0].item_id))
            selected = [(fact, score) for fact, score in ranked if score >= _MEMORY_RAW_SCORE_THRESHOLD][:top_k]
            max_score = max((score for _, score in selected), default=1.0)
            hits = [
                Memory(
                    text=fact.text,
                    score=score / max_score,
                    metadata={"id": fact.item_id},
                )
                for fact, score in selected
            ]
            self.last_user_hits = hits
            return hits
        self.agent_recall_calls += 1
        marker_match = _SEMANTIC_MARKER.search(query.lower())
        if marker_match is None:
            return []
        marker = marker_match.group(0)
        selected = [
            item
            for item in self._skill_corpus
            if item.source == "everos" and item.workspace_id == agent_id and marker in item.text.lower()
        ][:top_k]
        return [
            Memory(
                text=item.text,
                score=1.0,
                metadata={"id": item.item_id, "name": item.logical_id},
            )
            for item in selected
        ]


class _HistoricalSkillSource:
    name = "everos"
    weight = 0.9

    def __init__(self, backend: _FixtureMemoryBackend, agent_id: str) -> None:
        self._backend = backend
        self._agent_id = agent_id

    async def search(
        self,
        query: str,
        history: list[dict[str, Any]],
        k: int,
    ) -> list[RouterHit]:
        del history
        memories = await self._backend.recall(
            query,
            agent_id=self._agent_id,
            top_k=k,
        )
        return [
            RouterHit(
                qualified_id=f"everos/{memory.metadata['id']}",
                name=str(memory.metadata["name"]),
                content=memory.text,
                score=memory.score,
                meta={"source": "everos"},
            )
            for memory in memories
        ]


class _FixtureSkillRegistry:
    def __init__(self, metas: list[SkillMeta]) -> None:
        self._metas = metas

    def list_all(self) -> list[SkillMeta]:
        return list(self._metas)

    def get(
        self,
        name: str,
        *,
        source: str | None = None,
    ) -> SkillMeta | None:
        return next(
            (meta for meta in self._metas if meta.name == name and (source is None or meta.source == source)),
            None,
        )


class _EmptyMemoryStore:
    def get_memory_context(self, current_message: str = "") -> str:
        return ""


def _assembly_context(
    query: str,
    consuming_turn: str,
) -> AssemblyContext:
    return AssemblyContext(
        session_key=consuming_turn,
        current_message=query,
        media=None,
        channel="codeflowbench",
        chat_id=consuming_turn,
        session_messages=[],
        budget=TokenBudget(
            context_length=100_000,
            reserved_output=4_000,
            reserved_tools=2_000,
            reserved_system=1_000,
            available_history=93_000,
        ),
    )


def _selection(
    *,
    query_id: str,
    item_id: str,
    source: str,
    rank: int,
    raw_score: float,
    rrf_score: float | None,
    contributing_sources: tuple[str, ...],
    injected: bool,
    consuming_turn: str,
) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "item_id": item_id,
        "source": source,
        "rank": rank,
        "raw_score": round(raw_score, 8),
        "rrf_score": None if rrf_score is None else round(rrf_score, 8),
        "contributing_sources": list(contributing_sources),
        "injected": injected,
        "consuming_turn": consuming_turn,
    }
