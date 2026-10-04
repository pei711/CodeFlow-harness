"""从共享依赖构建唯一 ContextAssembler 及其扁平 SegmentBuilder 列表。

旧 ``legacy`` / ``curator`` / ``default`` Engine 分派已经移除。当前每轮由一套 Assembler 协调：
CodeFlow-style history manager 依据 token pressure 分层裁剪并在高压时总结历史，独占 ``*history``；
Memory lane 合并 Host Memory、可选 ``backend.recall(user_id=...)`` 与本地混合检索形成 segment 3
``# Memory``；Local Skill lane 用 :class:`SkillForgeRouter` 形成 segment 5 ``# Skills``；Host
通过 :class:`ContextBuilder` 提供 identity、bootstrap 与 always-skills。

SkillForgeRouter 包装 Builder 已有 ``LocalPool`` 与 ``SkillRegistry``，不会再扫一次磁盘。
外部 Memory Backend 可选；Host Memory 与内置轮次摘要召回始终可用，也不改变 Local Skill 可用性。Factory 的责任是接线和
默认配置，不执行某一 Turn 的选择；最终总是返回同一个 :class:`ContextAssembler` 类型。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol

from codeflow.agent.context import ContextBuilder
from codeflow.context_engine.assembler import ContextAssembler
from codeflow.context_engine.base import ContextEngine
from codeflow.context_engine.segments import (
    ActiveSkillsSegmentBuilder,
    BootstrapSegmentBuilder,
    IdentitySegmentBuilder,
    MemorySegmentBuilder,
    RelevantTurnMemorySegmentBuilder,
    SkillsSegmentBuilder,
)
from codeflow.context_engine.segments.codeflow_history import CodeFlowHistorySegmentBuilder
from codeflow.providers.base import LLMProvider

if TYPE_CHECKING:
    from codeflow.config.codeflow import (
        ContextConfig,
        MemoryConfig,
        SkillForgeConfig,
        SkillForgeRouterConfig,
    )
    from codeflow.memory_engine.backend import MemoryBackend
    from codeflow.memory_engine.skill_forge import SkillForgeRouter


class ContextEngineFactory(Protocol):
    def __call__(
        self,
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
    ) -> ContextEngine: ...


def build_context_engine(
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
    """从扁平 SegmentBuilder 列表构建唯一 :class:`ContextAssembler`。

    ``config.engine`` 已不再是 dispatch key，因为只有一个 Engine；字段留在
    :class:`ContextConfig` 仅为配置 back-compat，本函数有意忽略。``builder`` 当前只作为共享
    ``MemoryStore`` / ``LocalSkillCatalog`` 的持有者，待低层兼容结构退休后可移除。

    缺失 Memory 与 SkillForgeRouter 配置时创建默认值。Memory Segment 总是提供 Host Memory，
    并在 Backend 存在时额外执行外部召回；轮次摘要召回独立启用。Skill injection_mode 为 summary
    时 ``activation_max=0``，否则采用 inject_max 或 router top_k。Builder 按 identity、bootstrap、
    memory、relevant turn memory、active skills、router skills、CodeFlow history 顺序交给 Assembler，
    Tool definitions 使用延迟 callable 保持注册表为当前值。
    """
    from codeflow.config.codeflow import (
        MemoryConfig as _MemoryConfig,
    )
    from codeflow.config.codeflow import (
        SkillForgeRouterConfig as _SkillForgeRouterConfig,
    )

    if memory_config is None:
        memory_config = _MemoryConfig()
    if skill_forge_router_config is None:
        skill_forge_router_config = _SkillForgeRouterConfig()

    router = _build_router(
        builder=builder,
        skill_forge_router_config=skill_forge_router_config,
    )
    configured_inject_max = int(getattr(skill_forge_config, "inject_max", 2)) if skill_forge_config is not None else 2
    summary_only = (
        skill_forge_config is not None and getattr(skill_forge_config, "injection_mode", "full_body") == "summary"
    )
    activation_max = 0 if summary_only else configured_inject_max or skill_forge_router_config.top_k

    builders = [
        IdentitySegmentBuilder(workspace, builder.state),
        BootstrapSegmentBuilder(builder.state),
        MemorySegmentBuilder(
            builder.memory,
            backend,
            user_id=memory_config.user_id,
            memory_top_k=memory_config.memory_top_k,
            enabled=True,
        ),
        RelevantTurnMemorySegmentBuilder(builder.state),
        ActiveSkillsSegmentBuilder(builder.skills),
        SkillsSegmentBuilder(
            router,
            skill_top_k=skill_forge_router_config.top_k,
            activation_max=activation_max,
        ),
        CodeFlowHistorySegmentBuilder(
            state_root=builder.state,
            config=config,
            provider=provider,
            model=model,
            context_window_tokens=context_window_tokens,
            get_tool_definitions=get_tool_definitions,
        ),
    ]
    return ContextAssembler(builders, get_tool_definitions, now_fn=now_fn)


def _build_router(
    *,
    builder: ContextBuilder,
    skill_forge_router_config: "SkillForgeRouterConfig",
) -> "SkillForgeRouter":
    """为 segment 5 组装只包含 operator-managed Local Skill 的路由器。

    `LocalSkillSource` 直接复用 ``builder.skills.pool`` 与 ``builder.skills.registry``，并应用
    ``local_min_score``；这避免重新扫描磁盘或建立第二份 Registry。返回的 `SkillForgeRouter`
    当前只有该 Source，top-k 与 activation 数量由上层 `SkillsSegmentBuilder` 控制，本函数不
    执行检索。
    """
    from codeflow.memory_engine.skill_forge import (
        LocalSkillSource,
        SkillForgeRouter,
    )

    local_source = LocalSkillSource(
        pool=builder.skills.pool,
        registry=builder.skills.registry,
        min_score=skill_forge_router_config.local_min_score,
    )
    return SkillForgeRouter(sources=[local_source])
