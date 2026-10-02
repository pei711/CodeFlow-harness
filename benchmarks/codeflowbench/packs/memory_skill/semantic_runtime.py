"""Historical EverOS semantic Runtime retained for evidence reconstruction."""

from __future__ import annotations

import datetime as dt
import math
import os
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlmodel import SQLModel

from benchmarks.codeflowbench.budget import ProviderBudgetLedger
from benchmarks.codeflowbench.canonical import canonical_digest
from codeflow.context_engine.base import AssemblyContext
from codeflow.context_engine.segments import MemorySegmentBuilder, SkillsSegmentBuilder
from codeflow.memory_engine import Memory, TokenBudget
from codeflow.memory_engine.skill_forge import (
    LocalSkillSource,
    RouterHit,
    SkillForgeRouter,
)
from codeflow.memory_engine.skill_local import LocalPool, SkillMeta
from codeflow.plugin import PluginContext, ServiceLocator

from .fixtures import anonymous_item_id
from .models import MemoryFact, SkillItem
from .semantic_fixtures import SemanticFixture


@dataclass(frozen=True)
class SemanticRuntimeConfig:
    top_k: int
    user_min_score: float
    agent_radius: float
    local_weight: float
    everos_weight: float
    memory_top_k: int | None = None
    skill_candidate_top_k: int | None = None
    local_min_score: float = 0.0
    semantic_schema: str = "codeflow.codeflowbench.semantic-addendum.v1"
    skill_evidence_stage: str = "context_injection"

    @property
    def effective_memory_top_k(self) -> int:
        return self.memory_top_k or self.top_k

    @property
    def effective_skill_top_k(self) -> int:
        return self.skill_candidate_top_k or self.top_k


@dataclass(frozen=True)
class SemanticRuntimeResult:
    records: tuple[dict[str, Any], ...]
    indexed_rows: int
    embedding_calls: int
    embedded_characters: int
    provider_identity: str
    provider_config_digest: str
    real_provider: bool


class CountingEmbeddingProvider:
    def __init__(
        self,
        delegate: Any,
        *,
        provider_identity: str,
        provider_config_digest: str,
        real_provider: bool,
        maximum_characters: int | None = None,
        maximum_calls: int | None = None,
        characters_per_token: float = 2.0,
        budget_ledger: ProviderBudgetLedger | None = None,
        budget_trial_id: str | None = None,
        maximum_input_tokens_per_call: int | None = None,
    ) -> None:
        self._delegate = delegate
        self.dim = int(delegate.dim)
        self.provider_identity = provider_identity
        self.provider_config_digest = provider_config_digest
        self.real_provider = real_provider
        self.maximum_characters = maximum_characters
        self.maximum_calls = maximum_calls
        self.characters_per_token = characters_per_token
        self.budget_ledger = budget_ledger
        self.budget_trial_id = budget_trial_id
        self.maximum_input_tokens_per_call = maximum_input_tokens_per_call
        self.calls = 0
        self.embedded_characters = 0
        self.budgeted_input_tokens = 0

    async def embed(self, text: str) -> list[float]:
        request_id = self._reserve((text,))
        try:
            result = await self._delegate.embed(text)
        except BaseException as exc:
            self._fail(request_id, exc)
            raise
        self._settle(request_id, len(text))
        return result

    async def embed_batch(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        request_id = self._reserve(texts)
        characters = sum(len(text) for text in texts)
        try:
            result = await self._delegate.embed_batch(texts)
        except BaseException as exc:
            self._fail(request_id, exc)
            raise
        self._settle(request_id, characters)
        return result

    def _reserve(self, texts: Sequence[str]) -> str | None:
        characters = sum(len(text) for text in texts)
        if self.maximum_characters is not None and self.embedded_characters + characters > self.maximum_characters:
            raise RuntimeError("semantic embedding budget exhausted")
        if self.maximum_calls is not None and self.calls >= self.maximum_calls:
            raise RuntimeError("semantic embedding call budget exhausted")
        call_ordinal = self.calls + 1
        request_id = None
        estimated_tokens = self._estimated_tokens(characters)
        if self.budget_ledger is not None:
            if self.budget_trial_id is None:
                raise RuntimeError("semantic budget trial identity is missing")
            request_id = self.budget_ledger.reserve(
                trial_id=self.budget_trial_id,
                request_digest=canonical_digest(
                    {
                        "provider_config_digest": (self.provider_config_digest),
                        "call_ordinal": call_ordinal,
                        "texts": [canonical_digest({"text": text}) for text in texts],
                    }
                ),
                model=self.provider_identity,
                estimated_input_tokens=estimated_tokens,
                maximum_input_tokens=(self.maximum_input_tokens_per_call),
                maximum_output_tokens=1,
                max_logical_calls=self.maximum_calls,
                max_attempts_per_call=1,
            )
        self.calls = call_ordinal
        self.embedded_characters += characters
        self.budgeted_input_tokens += estimated_tokens
        return request_id

    def _settle(self, request_id: str | None, characters: int) -> None:
        if request_id is None or self.budget_ledger is None:
            return
        self.budget_ledger.settle(
            request_id,
            input_tokens=self._estimated_tokens(characters),
            output_tokens=0,
        )

    def _fail(
        self,
        request_id: str | None,
        exc: BaseException,
    ) -> None:
        if request_id is None or self.budget_ledger is None:
            return
        self.budget_ledger.fail(
            request_id,
            reason=f"embedding_provider_exception:{type(exc).__name__}",
        )

    def _estimated_tokens(self, characters: int) -> int:
        return max(1, math.ceil(characters / self.characters_per_token))


async def run_semantic_runtime(
    *,
    fixture: SemanticFixture,
    config: SemanticRuntimeConfig,
    isolated_root: Path,
    embedder: CountingEmbeddingProvider,
) -> SemanticRuntimeResult:
    from everos.component.tokenizer import build_tokenizer
    from everos.config import load_settings
    from everos.core.persistence import MemoryRoot
    from everos.infra.persistence.lancedb import (
        dispose_connection,
        ensure_business_indexes,
        get_connection,
        verify_business_schemas,
    )
    from everos.infra.persistence.sqlite import dispose_engine, get_engine

    previous_root = os.environ.get("EVEROS_ROOT")
    os.environ["EVEROS_ROOT"] = str(isolated_root)
    load_settings.cache_clear()
    root = MemoryRoot(isolated_root)
    root.ensure()
    engine = get_engine()
    try:
        async with engine.begin() as connection:
            await connection.run_sync(SQLModel.metadata.create_all)
        await get_connection()
        await verify_business_schemas()
        await ensure_business_indexes()
        native_to_logical = await _write_corpus(root, fixture)
        tokenizer = build_tokenizer()
        indexed_rows = await _sync_corpus(root, embedder, tokenizer)
        manager = _build_search_manager(embedder, tokenizer)
        adapter = _SearchManagerAdapter(
            manager,
            user_min_score=(None if config.semantic_schema.endswith(".v2") else config.user_min_score),
            agent_radius=config.agent_radius,
        )
        backend = _RecordingEverosBackend(
            isolated_root,
            adapter,
            user_min_score=config.user_min_score,
        )
        records = await _run_cases(
            fixture=fixture,
            config=config,
            backend=backend,
            native_to_logical=native_to_logical,
            embedder=embedder,
        )
        return SemanticRuntimeResult(
            records=records,
            indexed_rows=indexed_rows,
            embedding_calls=embedder.calls,
            embedded_characters=embedder.embedded_characters,
            provider_identity=embedder.provider_identity,
            provider_config_digest=embedder.provider_config_digest,
            real_provider=embedder.real_provider,
        )
    finally:
        await dispose_connection()
        await dispose_engine()
        if previous_root is None:
            os.environ.pop("EVEROS_ROOT", None)
        else:
            os.environ["EVEROS_ROOT"] = previous_root
        load_settings.cache_clear()


async def _write_corpus(
    root: Any,
    fixture: SemanticFixture,
) -> dict[str, tuple[str, str]]:
    from everos.infra.persistence.markdown import (
        AgentSkillFrontmatter,
        AgentSkillWriter,
        EpisodeWriter,
    )

    native_to_logical: dict[str, tuple[str, str]] = {}
    episode_writer = EpisodeWriter(root)
    memory_by_workspace: dict[str, list[MemoryFact]] = defaultdict(list)
    for item in fixture.memory_corpus:
        memory_by_workspace[item.workspace_id].append(item)
    now = dt.datetime.now(dt.UTC).isoformat()
    for workspace_id, items in memory_by_workspace.items():
        entry_ids = await episode_writer.append_entries(
            workspace_id,
            [
                (
                    {
                        "owner_id": workspace_id,
                        "session_id": f"{fixture.track}-semantic-corpus",
                        "timestamp": now,
                        "parent_type": "memcell",
                        "parent_id": item.item_id,
                        "sender_ids": [workspace_id],
                    },
                    {"Content": item.text},
                )
                for item in items
            ],
        )
        entry_by_logical = {item.item_id: str(entry_id) for item, entry_id in zip(items, entry_ids, strict=True)}
        for item in items:
            native_to_logical[f"{workspace_id}_{entry_by_logical[item.item_id]}"] = (item.item_id, workspace_id)
        deprecated: dict[str, str] = {}
        for item in items:
            if not item.superseded:
                continue
            active_id = item.item_id.replace("stale", "active")
            if active_id in entry_by_logical:
                deprecated[entry_by_logical[item.item_id]] = entry_by_logical[active_id]
        if deprecated:
            await episode_writer.patch_frontmatter(
                episode_writer.path_for(workspace_id),
                {"deprecated_entries": deprecated},
            )

    skill_writer = AgentSkillWriter(root)
    for item in fixture.skill_corpus:
        if item.source != "everos":
            continue
        frontmatter = AgentSkillFrontmatter(
            id=f"{item.workspace_id}_{item.logical_id}",
            agent_id=item.workspace_id,
            name=item.logical_id,
            description=item.text,
            confidence=1.0,
            maturity_score=1.0,
        )
        await skill_writer.write_main(
            item.workspace_id,
            item.logical_id,
            frontmatter=frontmatter,
            body=item.text,
        )
        native_to_logical[f"{item.workspace_id}_{item.logical_id}"] = (item.logical_id, item.workspace_id)
    return native_to_logical


async def _sync_corpus(
    root: Any,
    embedder: CountingEmbeddingProvider,
    tokenizer: Any,
) -> int:
    from everos.memory.cascade import CascadeOrchestrator

    orchestrator = CascadeOrchestrator(
        memory_root=root,
        embedder=embedder,
        tokenizer=tokenizer,
    )
    return await orchestrator.sync_once()


def _build_search_manager(
    embedder: CountingEmbeddingProvider,
    tokenizer: Any,
) -> Any:
    from everos.memory.search.manager import SearchManager
    from everos.memory.search.recall import (
        AgentCaseRecaller,
        AgentSkillRecaller,
        AtomicFactRecaller,
        EpisodeRecaller,
        ProfileRecaller,
        RecallerDeps,
    )

    deps = RecallerDeps(tokenizer=tokenizer)
    return SearchManager(
        episode_recaller=EpisodeRecaller(deps),
        atomic_fact_recaller=AtomicFactRecaller(deps),
        agent_case_recaller=AgentCaseRecaller(deps),
        agent_skill_recaller=AgentSkillRecaller(deps),
        profile_recaller=ProfileRecaller(),
        embedding=embedder,
        reranker=None,
        llm_client=None,
        search_tokenizer=tokenizer,
    )


class _SearchManagerAdapter:
    def __init__(
        self,
        manager: Any,
        *,
        user_min_score: float | None,
        agent_radius: float,
    ) -> None:
        self._manager = manager
        self._user_min_score = user_min_score
        self._agent_radius = agent_radius

    async def search(
        self,
        *,
        user_id: str | None,
        agent_id: str | None,
        query: str,
        top_k: int,
    ) -> Any:
        from everos.memory.search import SearchMethod, SearchRequest

        if user_id is not None:
            request = SearchRequest(
                user_id=user_id,
                query=query,
                method=SearchMethod.HYBRID,
                top_k=top_k,
                min_score=self._user_min_score,
            )
        else:
            request = SearchRequest(
                agent_id=agent_id,
                query=query,
                method=SearchMethod.VECTOR,
                top_k=top_k,
                radius=self._agent_radius,
            )
        response = await self._manager.search(request)
        return response.data

    async def memorize(
        self,
        session_id: str,
        payload_messages: list[dict[str, Any]],
        *,
        is_final: bool = False,
    ) -> None:
        del session_id, payload_messages, is_final


class _RecordingEverosBackend:
    def __init__(
        self,
        root: Path,
        adapter: _SearchManagerAdapter,
        *,
        user_min_score: float,
    ) -> None:
        from codeflow.plugin.memory.everos.backend import EverosBackend

        self._delegate = EverosBackend(
            PluginContext(
                config={
                    "mode": "embedded",
                    "user_min_score": user_min_score,
                },
                services=ServiceLocator(workspace=root),
            ),
            adapter=adapter,
        )
        self.last_hits: list[Memory] = []

    async def recall(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        top_k: int,
    ) -> list[Memory]:
        hits = await self._delegate.recall(
            query,
            user_id=user_id,
            agent_id=agent_id,
            top_k=top_k,
        )
        self.last_hits = hits
        return hits


class _HistoricalSkillSource:
    name = "everos"
    weight = 0.9

    def __init__(self, backend: _RecordingEverosBackend, agent_id: str) -> None:
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
                name=str(memory.metadata.get("name") or memory.metadata["id"]),
                content=memory.text,
                score=memory.score,
                meta={"source": "everos"},
            )
            for memory in memories
            if memory.metadata.get("id") is not None
        ]


async def _run_cases(
    *,
    fixture: SemanticFixture,
    config: SemanticRuntimeConfig,
    backend: _RecordingEverosBackend,
    native_to_logical: dict[str, tuple[str, str]],
    embedder: CountingEmbeddingProvider,
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    empty_store = _EmptyMemoryStore()
    skill_workspaces: dict[str, str | None] = {}
    for item in fixture.skill_corpus:
        previous = skill_workspaces.get(item.logical_id)
        if previous is None and item.logical_id not in skill_workspaces:
            skill_workspaces[item.logical_id] = item.workspace_id
        elif previous != item.workspace_id:
            skill_workspaces[item.logical_id] = None
    for query in fixture.memory_queries:
        before = embedder.calls
        segment = await MemorySegmentBuilder(
            empty_store,
            backend,
            user_id=str(query.payload["workspace_id"]),
            memory_top_k=config.effective_memory_top_k,
        ).build(
            _assembly_context(
                str(query.payload["query_text"]),
                str(query.payload["consuming_turn"]),
            )
        )
        records.append(
            _memory_record(
                fixture,
                query,
                backend.last_hits,
                native_to_logical,
                segment is not None,
                embedder.calls - before,
                embedder,
                config,
            )
        )

    local_sources: dict[str, LocalSkillSource] = {}
    for query in fixture.skill_queries:
        workspace_id = str(query.payload["workspace_id"])
        local = local_sources.get(workspace_id)
        if local is None:
            local = _local_source(
                fixture.skill_corpus,
                workspace_id,
                min_score=config.local_min_score,
            )
            local_sources[workspace_id] = local
        everos = _HistoricalSkillSource(backend, workspace_id)
        local.weight = config.local_weight
        everos.weight = config.everos_weight
        for configuration_id, sources in (
            ("local_only", [local]),
            ("everos_only", [everos]),
            ("fused", [local, everos]),
        ):
            before = embedder.calls
            router = _RecordingRouter(SkillForgeRouter(sources=sources))
            segment = None
            if config.skill_evidence_stage == "candidate_retrieval":
                await router.select(
                    str(query.payload["query_text"]),
                    [],
                    config.effective_skill_top_k,
                )
            else:
                segment = await SkillsSegmentBuilder(
                    router,
                    skill_top_k=config.effective_skill_top_k,
                ).build(
                    _assembly_context(
                        str(query.payload["query_text"]),
                        str(query.payload["consuming_turn"]),
                    )
                )
            records.append(
                _skill_record(
                    fixture,
                    query,
                    configuration_id,
                    router.last_hits,
                    segment,
                    skill_workspaces,
                    embedder.calls - before,
                    embedder,
                    config,
                )
            )
    return tuple(records)


class _RecordingRouter:
    def __init__(self, delegate: SkillForgeRouter) -> None:
        self._delegate = delegate
        self.last_hits: list[Any] = []

    async def select(
        self,
        query: str,
        history: list[dict[str, Any]],
        k: int = 5,
        *,
        diagnostics: dict[str, Any] | None = None,
    ) -> list[Any]:
        self.last_hits = await self._delegate.select(
            query,
            history,
            k,
            diagnostics=diagnostics,
        )
        return self.last_hits


def _memory_record(
    fixture: SemanticFixture,
    query: Any,
    hits: list[Memory],
    native_to_logical: dict[str, tuple[str, str]],
    injected: bool,
    embedding_calls: int,
    embedder: CountingEmbeddingProvider,
    config: SemanticRuntimeConfig,
) -> dict[str, Any]:
    ranked = []
    for rank, hit in enumerate(hits, start=1):
        native_id = str(hit.metadata.get("id", ""))
        logical_id, selected_workspace_id = native_to_logical.get(
            native_id,
            (native_id, ""),
        )
        ranked.append(
            _selection(
                query.query_id,
                anonymous_item_id(fixture.memory_suite_id, logical_id),
                "user_memory",
                rank,
                hit.score,
                None,
                ("everos_hybrid",),
                injected,
                str(query.payload["consuming_turn"]),
                str(query.payload["workspace_id"]),
                selected_workspace_id,
            )
        )
    return _record(
        fixture.memory_suite_id,
        query,
        "user_memory_on",
        ranked,
        embedding_calls,
        embedder,
        "everos_hybrid",
        semantic_schema=config.semantic_schema,
        evidence_stage="memory_context_injection",
    )


def _skill_record(
    fixture: SemanticFixture,
    query: Any,
    configuration_id: str,
    hits: list[Any],
    segment: Any,
    skill_workspaces: dict[str, str | None],
    embedding_calls: int,
    embedder: CountingEmbeddingProvider,
    config: SemanticRuntimeConfig,
) -> dict[str, Any]:
    injected_ids = set(segment.meta.get("injected_skill_ids", ())) if segment is not None else set()
    ranked = [
        _selection(
            query.query_id,
            anonymous_item_id(fixture.skill_suite_id, hit.name),
            configuration_id.removesuffix("_only"),
            rank,
            hit.score,
            (float(hit.meta["rrf_score"]) if "rrf_score" in hit.meta else None),
            tuple(
                str(source)
                for source in hit.meta.get(
                    "contributing_sources",
                    [configuration_id.removesuffix("_only")],
                )
            ),
            hit.qualified_id in injected_ids,
            str(query.payload["consuming_turn"]),
            str(query.payload["workspace_id"]),
            skill_workspaces.get(hit.name),
        )
        for rank, hit in enumerate(hits, start=1)
    ]
    return _record(
        fixture.skill_suite_id,
        query,
        configuration_id,
        ranked,
        embedding_calls,
        embedder,
        (
            "local_bm25"
            if configuration_id == "local_only"
            else "everos_vector"
            if configuration_id == "everos_only"
            else "local_bm25_plus_everos_vector_weighted_rrf"
        ),
        semantic_schema=config.semantic_schema,
        evidence_stage=config.skill_evidence_stage,
    )


def _record(
    suite_id: str,
    query: Any,
    configuration_id: str,
    ranked: list[dict[str, Any]],
    embedding_calls: int,
    embedder: CountingEmbeddingProvider,
    retrieval_path: str,
    *,
    semantic_schema: str,
    evidence_stage: str,
) -> dict[str, Any]:
    uses_everos_vector = configuration_id != "local_only"
    version = semantic_schema.rsplit(".", 1)[-1]
    record = {
        "schema": f"codeflow.codeflowbench.semantic-retrieval-record.{version}",
        "key": {
            "retrieval_suite_id": suite_id,
            "query_id": query.query_id,
            "configuration_id": configuration_id,
        },
        "label": query.label,
        "expected_item_ids": list(query.expected_item_ids),
        "status": "measurable",
        "ranked_results": ranked,
        "usage": {
            "embedding_calls": embedding_calls,
            "provider_calls": embedding_calls,
            "retrieval_evidence_level": ("production_everos_real_vector" if uses_everos_vector else "local_bm25"),
            "retrieval_path": retrieval_path,
            "backend_class": ("EverosBackend" if uses_everos_vector else "LocalSkillSource"),
            "backend_adapter": ("everos_search_manager_real_embedding" if uses_everos_vector else None),
            "provider_identity": embedder.provider_identity,
            "provider_config_digest": embedder.provider_config_digest,
            "everos_semantic_quality_claim_eligible": (embedder.real_provider and uses_everos_vector),
        },
    }
    if version == "v2":
        record["evidence_stage"] = evidence_stage
    if evidence_stage == "candidate_retrieval":
        record["candidate_results"] = ranked
    else:
        record["injected_results"] = [result for result in ranked if result["injected"]]
    return record


def _selection(
    query_id: str,
    item_id: str,
    source: str,
    rank: int,
    raw_score: float,
    rrf_score: float | None,
    contributing_sources: tuple[str, ...],
    injected: bool,
    consuming_turn: str,
    query_workspace_id: str,
    selected_workspace_id: str | None,
) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "item_id": item_id,
        "source": source,
        "rank": rank,
        "raw_score": round(float(raw_score), 8),
        "rrf_score": (None if rrf_score is None else round(float(rrf_score), 8)),
        "contributing_sources": list(contributing_sources),
        "injected": injected,
        "consuming_turn": consuming_turn,
        "query_workspace_id": query_workspace_id,
        "selected_workspace_id": selected_workspace_id,
    }


def _local_source(
    corpus: tuple[SkillItem, ...],
    workspace_id: str,
    *,
    min_score: float = 0.0,
) -> LocalSkillSource:
    metas = [
        SkillMeta(
            id=item.item_id,
            name=item.logical_id,
            description=item.text,
            path=Path("/codeflowbench-semantic") / item.item_id / "SKILL.md",
            content=item.text,
            source="local",
        )
        for item in corpus
        if item.source == "local" and item.workspace_id == workspace_id
    ]
    registry = _SkillRegistry(metas)
    return LocalSkillSource(
        LocalPool(registry),
        registry,
        min_score=min_score,
    )


class _SkillRegistry:
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
        del current_message
        return ""


def _assembly_context(query: str, consuming_turn: str) -> AssemblyContext:
    return AssemblyContext(
        session_key=consuming_turn,
        current_message=query,
        media=None,
        channel="codeflowbench-semantic",
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


def semantic_runtime_identity(
    fixture: SemanticFixture,
    config: SemanticRuntimeConfig,
) -> dict[str, Any]:
    identity = {
        "track": fixture.track,
        "memory_suite_id": fixture.memory_suite_id,
        "skill_suite_id": fixture.skill_suite_id,
        "memory_corpus_digest": canonical_digest(fixture.memory_corpus),
        "skill_corpus_digest": canonical_digest(fixture.skill_corpus),
        "memory_query_digest": canonical_digest(fixture.memory_queries),
        "skill_query_digest": canonical_digest(fixture.skill_queries),
        "runtime_config_digest": canonical_digest(_semantic_runtime_config_identity(config)),
    }
    if config.semantic_schema.endswith(".v2"):
        identity["semantic_schema"] = config.semantic_schema
    return identity


def _semantic_runtime_config_identity(
    config: SemanticRuntimeConfig,
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "top_k": config.top_k,
        "user_min_score": config.user_min_score,
        "agent_radius": config.agent_radius,
        "local_weight": config.local_weight,
        "everos_weight": config.everos_weight,
    }
    if config.semantic_schema.endswith(".v2"):
        identity.update(
            {
                "memory_top_k": config.effective_memory_top_k,
                "skill_candidate_top_k": config.effective_skill_top_k,
                "local_min_score": config.local_min_score,
                "semantic_schema": config.semantic_schema,
                "skill_evidence_stage": config.skill_evidence_stage,
            }
        )
    return identity


__all__ = [
    "CountingEmbeddingProvider",
    "SemanticRuntimeConfig",
    "SemanticRuntimeResult",
    "run_semantic_runtime",
    "semantic_runtime_identity",
]
