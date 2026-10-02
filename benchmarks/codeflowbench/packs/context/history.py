from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from codeflow.agent.context import ContextBuilder
from codeflow.context_engine.assembler import ContextAssembler
from codeflow.context_engine.base import AssemblyContext, ContextEngine, Segment
from codeflow.context_engine.factory import _build_router
from codeflow.context_engine.history_trimmer import HistoryTrimmer
from codeflow.context_engine.segments import (
    ActiveSkillsSegmentBuilder,
    BootstrapSegmentBuilder,
    IdentitySegmentBuilder,
    MemorySegmentBuilder,
    SkillsSegmentBuilder,
)
from codeflow.context_engine.segments.curator import CuratorSegmentBuilder
from codeflow.providers.base import LLMProvider

if TYPE_CHECKING:
    from codeflow.config.codeflow import (
        ContextConfig,
        MemoryConfig,
        SkillForgeConfig,
        SkillForgeRouterConfig,
    )
    from codeflow.context_engine.factory import ContextEngineFactory
    from codeflow.memory_engine.backend import MemoryBackend

CONTEXT_BENCHMARK_CURATOR_MAX_STEPS = 4


class FifoHistoryManager:
    name = "fifo_history_manager"
    order = 6
    needs_prefix = True

    def __init__(
        self,
        *,
        provider: LLMProvider,
        model: str,
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        context_window_tokens: int,
    ) -> None:
        self._trimmer = HistoryTrimmer(
            provider,
            model,
            get_tool_definitions,
            context_window_tokens,
        )

    async def build(self, ctx: AssemblyContext) -> Segment:
        if ctx.prefix is None:
            raise RuntimeError("FifoHistoryManager requires phase-B prefix")

        def build_messages(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [
                {"role": "system", "content": ctx.prefix.system_prefix},
                *history,
                ctx.prefix.user_message,
            ]

        _, outcome = self._trimmer.trim(
            session_messages=ctx.session_messages,
            ids=list(range(len(ctx.session_messages))),
            protected_ids=set(),
            reserved_output=ctx.budget.reserved_output,
            build_messages=build_messages,
            priority_scores={index: float(index) for index in range(len(ctx.session_messages))},
        )
        return Segment(
            history=outcome.history,
            meta={
                "path": "fifo_tail",
                "history_manager": "fifo_tail",
                "included_message_ids": outcome.included_ids,
                "estimated_prompt_tokens": outcome.estimated_tokens,
                "max_prompt_tokens": outcome.max_prompt_tokens,
                "token_estimation_source": outcome.source,
                "history_warnings": outcome.warnings,
            },
        )


class FullHistoryManager:
    name = "full_history_manager"
    order = 6
    needs_prefix = True

    async def build(self, ctx: AssemblyContext) -> Segment:
        if ctx.prefix is None:
            raise RuntimeError("FullHistoryManager requires phase-B prefix")
        included_ids = list(range(len(ctx.session_messages)))
        history = HistoryTrimmer.history_from_ids(
            ctx.session_messages,
            included_ids,
        )
        return Segment(
            history=history,
            meta={
                "path": "full_history",
                "history_manager": "full_history",
                "included_message_ids": included_ids,
            },
        )


def context_engine_factory_for(history_manager: str) -> "ContextEngineFactory":
    if history_manager == "curator":
        return build_benchmark_curator_context_engine
    if history_manager == "fifo_tail":
        return build_fifo_context_engine
    if history_manager == "full_history":
        return build_full_history_context_engine
    raise ValueError(f"unsupported Context history manager: {history_manager}")


def build_benchmark_curator_context_engine(
    *,
    workspace: Path,
    config: "ContextConfig",
    builder: ContextBuilder,
    provider: LLMProvider,
    model: str,
    context_window_tokens: int,
    get_tool_definitions: Callable[[], list[dict]],
    now_fn: Callable[[], datetime] | None = None,
    backend: "MemoryBackend | None" = None,
    memory_config: "MemoryConfig | None" = None,
    skill_forge_router_config: "SkillForgeRouterConfig | None" = None,
    skill_forge_config: "SkillForgeConfig | None" = None,
) -> ContextEngine:
    return _build_benchmark_context_engine(
        workspace=workspace,
        builder=builder,
        provider=provider,
        model=model,
        context_window_tokens=context_window_tokens,
        get_tool_definitions=get_tool_definitions,
        now_fn=now_fn,
        backend=backend,
        memory_config=memory_config,
        skill_forge_router_config=skill_forge_router_config,
        skill_forge_config=skill_forge_config,
        history_manager=CuratorSegmentBuilder(
            workspace=workspace,
            config=config,
            provider=provider,
            model=model,
            context_window_tokens=context_window_tokens,
            get_tool_definitions=get_tool_definitions,
            now_fn=now_fn,
            memory_enabled=backend is not None,
            max_steps=CONTEXT_BENCHMARK_CURATOR_MAX_STEPS,
        ),
    )


def build_fifo_context_engine(
    *,
    workspace: Path,
    config: "ContextConfig",
    builder: ContextBuilder,
    provider: LLMProvider,
    model: str,
    context_window_tokens: int,
    get_tool_definitions: Callable[[], list[dict]],
    now_fn: Callable[[], datetime] | None = None,
    backend: "MemoryBackend | None" = None,
    memory_config: "MemoryConfig | None" = None,
    skill_forge_router_config: "SkillForgeRouterConfig | None" = None,
    skill_forge_config: "SkillForgeConfig | None" = None,
) -> ContextEngine:
    del config
    return _build_benchmark_context_engine(
        workspace=workspace,
        builder=builder,
        provider=provider,
        model=model,
        context_window_tokens=context_window_tokens,
        get_tool_definitions=get_tool_definitions,
        now_fn=now_fn,
        backend=backend,
        memory_config=memory_config,
        skill_forge_router_config=skill_forge_router_config,
        skill_forge_config=skill_forge_config,
        history_manager=FifoHistoryManager(
            provider=provider,
            model=model,
            get_tool_definitions=get_tool_definitions,
            context_window_tokens=context_window_tokens,
        ),
    )


def build_full_history_context_engine(
    *,
    workspace: Path,
    config: "ContextConfig",
    builder: ContextBuilder,
    provider: LLMProvider,
    model: str,
    context_window_tokens: int,
    get_tool_definitions: Callable[[], list[dict]],
    now_fn: Callable[[], datetime] | None = None,
    backend: "MemoryBackend | None" = None,
    memory_config: "MemoryConfig | None" = None,
    skill_forge_router_config: "SkillForgeRouterConfig | None" = None,
    skill_forge_config: "SkillForgeConfig | None" = None,
) -> ContextEngine:
    del config
    return _build_benchmark_context_engine(
        workspace=workspace,
        builder=builder,
        provider=provider,
        model=model,
        context_window_tokens=context_window_tokens,
        get_tool_definitions=get_tool_definitions,
        now_fn=now_fn,
        backend=backend,
        memory_config=memory_config,
        skill_forge_router_config=skill_forge_router_config,
        skill_forge_config=skill_forge_config,
        history_manager=FullHistoryManager(),
    )


def _build_benchmark_context_engine(
    *,
    workspace: Path,
    builder: ContextBuilder,
    provider: LLMProvider,
    model: str,
    context_window_tokens: int,
    get_tool_definitions: Callable[[], list[dict]],
    now_fn: Callable[[], datetime] | None,
    backend: "MemoryBackend | None",
    memory_config: "MemoryConfig | None",
    skill_forge_router_config: "SkillForgeRouterConfig | None",
    skill_forge_config: "SkillForgeConfig | None",
    history_manager: Any,
) -> ContextEngine:
    from codeflow.config.codeflow import MemoryConfig, SkillForgeRouterConfig

    resolved_memory = memory_config or MemoryConfig()
    resolved_router = skill_forge_router_config or SkillForgeRouterConfig()
    router = _build_router(
        builder=builder,
        skill_forge_router_config=resolved_router,
    )
    configured_inject_max = int(getattr(skill_forge_config, "inject_max", 2)) if skill_forge_config is not None else 2
    summary_only = (
        skill_forge_config is not None and getattr(skill_forge_config, "injection_mode", "full_body") == "summary"
    )
    activation_max = 0 if summary_only else configured_inject_max or resolved_router.top_k
    builders = [
        IdentitySegmentBuilder(workspace),
        BootstrapSegmentBuilder(workspace),
        MemorySegmentBuilder(
            builder.memory,
            backend,
            user_id=resolved_memory.user_id,
            memory_top_k=resolved_memory.memory_top_k,
            enabled=backend is not None,
        ),
        ActiveSkillsSegmentBuilder(builder.skills),
        SkillsSegmentBuilder(
            router,
            skill_top_k=resolved_router.top_k,
            activation_max=activation_max,
        ),
        history_manager,
    ]
    return ContextAssembler(builders, get_tool_definitions, now_fn=now_fn)


__all__ = [
    "CONTEXT_BENCHMARK_CURATOR_MAX_STEPS",
    "FifoHistoryManager",
    "FullHistoryManager",
    "build_benchmark_curator_context_engine",
    "build_fifo_context_engine",
    "build_full_history_context_engine",
    "context_engine_factory_for",
]
