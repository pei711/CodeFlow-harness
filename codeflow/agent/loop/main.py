"""把一条用户请求推进为模型回复、工具执行和可交付结果的 Agent Loop 核心引擎。

没有 Agent 开发经验时，可以先把这里理解为一次请求的“现场总调度”：它从 Spine 接收
`TurnRequest`，借助 Context Engine 拼出模型本轮可见的历史、Memory 与 Skill，调用
Provider 获得模型输出，再执行模型选择的 Tool。只要模型继续请求工具，这条链路就会
在受限迭代预算内重复；得到最终文本、达到上限或遇到 Provider 错误后，结果再沿 Spine
的单一 `emit` 边界交给 TUI、REPL 或其他出口。

本模块还拥有几条不能被“循环调用模型”这个表面描述掩盖的边界：流式文本与非流式文本
不能重复交付，工具输出过大时要在持久化和上下文窗口处分别收缩，MCP 与 Sandbox 资源
按运行时生命周期启动和关闭，Checkpoint 只为符合策略的 Turn 留下 shadow-git 恢复证据。
阅读时可先看 `AgentLoop.run_turn` 的入口与 `_run_agent_loop` 的迭代，再回到上下文组装、
工具注册、恢复和关闭方法理解各自所有权。
"""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from codeflow.agent.context import ContextBuilder
from codeflow.agent.loop.recovery import (
    POST_TOOL_NUDGE,
    RecoveryAction,
    RecoveryLimits,
    classify_empty_response,
)
from codeflow.agent.subagent import SubagentManager
from codeflow.agent.tools.ask_user import AskUserTool
from codeflow.agent.tools.base import ToolResult
from codeflow.agent.tools.execution import ToolExecution, ToolExecutionContext, ToolInvocation
from codeflow.agent.tools.file_search import FindTool, GrepTool
from codeflow.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from codeflow.agent.tools.message import MessageTool
from codeflow.agent.tools.registry import ToolRegistry
from codeflow.agent.tools.resource_locks import ResourceLockManager
from codeflow.agent.tools.shell import ExecTool
from codeflow.agent.tools.skill import SkillReadTool
from codeflow.agent.tools.spawn import SpawnTool
from codeflow.agent.tools.web import WebFetchTool, WebSearchTool
from codeflow.call_efficiency.pricing import resolve_context_window
from codeflow.memory_engine.base import TokenBudget
from codeflow.memory_engine.consolidate.consolidator import MemoryConsolidator, MemoryStore
from codeflow.providers.base import ErrorClassification, LLMProvider, LLMResponse, ToolCallRequest
from codeflow.sandbox import SandboxConfig, SandboxExecutor, SandboxInitError, build_executor
from codeflow.session.manager import Session, SessionManager
from codeflow.spine.message import Media
from codeflow.spine.turn import Origin
from codeflow.tracing import semconv, trace
from codeflow.utils.helpers import estimate_prompt_tokens
from codeflow.utils.persisted_payload import sanitize_persisted_payload

# 刻意在 ``__init__`` 和 ``_assemble_context_messages`` 内延迟导入 ``codeflow.context_engine``，以打破运行时
# 循环导入：``codeflow.agent.__init__`` 急切加载 AgentLoop，而 ``codeflow.context_engine.curator``
# 又从 ``codeflow.agent.context`` 导入 ``ContextBuilder``。如果在此模块顶层导入，会重新进入
# 只初始化了一部分的包，并在 ``TurnContext`` 上抛出 ImportError。

if TYPE_CHECKING:
    from codeflow.agent.hook import CompositeHook
    from codeflow.agent.tools.base import Tool
    from codeflow.call_efficiency import CallEfficiency
    from codeflow.config.codeflow import (
        ContextConfig,
        MemoryConfig,
        RuntimeConfig,
        SkillForgeRouterConfig,
    )
    from codeflow.config.schema import ChannelsConfig, ExecToolConfig
    from codeflow.context_engine import ContextEngine
    from codeflow.context_engine.factory import ContextEngineFactory
    from codeflow.memory_engine.backend import MemoryBackend
    from codeflow.proactive_engine.schedulers.cron.service import CronService
    from codeflow.routing.router import ModelRouter
    from codeflow.sandbox.debug_server import SandboxDebugServer
    from codeflow.spine.runner import Drain, Emit, TurnOutcome
    from codeflow.spine.turn import TurnRequest
    from codeflow.token_wise.base import UsageSnapshot
    from codeflow.token_wise.registry import StrategyRegistry


@dataclass
class TurnOutcome:
    """记录一次 ``_run_agent_loop`` 除文本回复之外的终态与恢复证据。

    Turn 是 Agent 围绕一条请求执行的一轮完整工作，不等同于模型生成的一段文字。调用方
    除了需要最终文本，还必须知道循环为何停止、是否留下部分编辑以及下轮应从哪里核对。
    这个值对象把这些控制面事实从回复正文中分离出来，避免调用方根据自然语言猜测状态。

    ``status`` 明确区分三种终态：``"completed"`` 表示正常完成，``"interrupted"`` 表示
    达到最大迭代预算后中断，``"error"`` 表示 LLM 调用出错。该区分落实 Bug2 / decision B：
    调用方绝不能把 "ran out of budget" 当成 "done"。``error_category`` 在错误终态下携带
    稳定分类，供边界外记录或展示，而不是要求读者解析 Provider 的原始异常文本。

    ``checkpoint_id`` 与 ``edited_files`` 携带本轮 shadow-git 快照标识和已编辑文件清单。
    `_stash_recovery` 只会为可恢复的中断保存它们，下一 Turn 的 `_inject_recovery_block`
    再据此构造恢复提示。它们提供的是“先检查哪些现场”的证据，不保证下一轮一定能自动
    恢复成功，也不会把模型回复本身当作文件状态的事实来源。
    """

    status: str = "completed"  # 可选 "completed" | "interrupted" | "error"
    checkpoint_id: str | None = None
    edited_files: list[str] = field(default_factory=list)
    error_category: str | None = None


class ProviderTurnError(RuntimeError):
    """在 Provider 无法完成 Turn 时，向运行边界传播安全、稳定的终止错误。

    Provider 是 Agent Loop 与具体 LLM 服务之间的适配层。底层服务可能返回包含供应商细节
    的异常，但上层只需要可记录、可比较的错误类别；因此本异常把 ``category`` 保存为字段，
    并把消息规范化为 ``provider_error:{category}``。它表示本轮已经进入错误终态，不是可供
    循环继续消费的普通模型回复；调用方应让 Spine 的失败事件处理这条路径。
    """

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(f"provider_error:{category}")


# 迭代预算用尽后，让模型尽力收尾。本次调用不提供工具，因此提示词不能诱导
# 模型再次调用工具或提问，因为不会再有下一轮回答。
_MAX_ITER_SYNTHESIS_PROMPT = (
    "You've used up the tool-calling budget for this turn, so no tools are "
    "available now. Using only what you've already gathered, give your best "
    "final answer: summarize what you accomplished, deliver any partial "
    "results, and briefly note what's left undone. Do not ask questions — "
    "there is no further turn to answer them. Reply in the same language as "
    "the user's request (this instruction is in English, but it is not the "
    "conversation language)."
)

# 只在合成调用自身失败时返回，绝不让 Turn 没有响应。
_MAX_ITER_STATIC_FALLBACK = (
    "I reached the maximum number of tool call iterations ({n}) without "
    "completing the task. You can try breaking the task into smaller steps."
)

# 这些来源的内容并非用户输入，所以对应 Turn 跳过用户入站 Hook。
_SKIP_USER_INBOUND_ORIGINS = frozenset({Origin.CRON, Origin.SUBAGENT})

# 普通重试很可能清除这些失败标记，因此不得计入工具失败连续记录；
# 对会自愈的 429 发出提示只会制造噪声。
_TRANSIENT_FAILURE_MARKERS = (
    "429",
    "rate limit",
    "timed out",
    "timeout",
    "no healthy upstream",
    "502",
    "503",
)
# 成功但为空的结果表示工具正常运行，只是没找到内容。重复空搜索是合法探索，
# 而不是卡死的无效调用，因此不得计入失败连续记录。
_EMPTY_SUCCESS_MARKERS = ("no matches found", "no files found")


def _is_tool_failure(result: object) -> bool:
    if isinstance(result, ToolResult):
        return result.failed

    s = str(result).strip()
    match = re.search(r"(?:^|\n)Exit code:\s*(-?\d+)(?:\s|$)", s)
    if match:
        return match.group(1) != "0"
    if s.startswith("{"):
        try:
            payload = json.loads(s)
        except (json.JSONDecodeError, TypeError):
            pass
        else:
            if isinstance(payload, dict) and payload.get("error"):
                return True
    low = s.lower()
    return (
        low.startswith("error:")
        or low.startswith("error ")
        or low.startswith("proxy error:")
        or low.startswith("(mcp tool call failed:")
        or low.startswith("(mcp tool call timed out ")
        or low.startswith("(mcp tool call was cancelled)")
    )


def _is_hard_tool_failure(result: object) -> bool:
    """判断工具结果是否属于“原样重试仍会复现”的确定性失败。

    返回 ``True`` 表示同一 Tool 和相同参数再次执行通常只会得到同类错误，可以计入连续
    失败并触发循环打断；成功结果、可重试的临时错误以及合法的空搜索结果返回 ``False``。
    429、超时、502/503 等标记会先被排除，因为外部依赖恢复后重试可能成功；"no matches
    found" 和 "no files found" 也不是失败。最终的失败判定复用 `_is_tool_failure`，本函数
    只负责增加“是否值得打断重复尝试”这一层语义。
    """
    s = str(result)
    low = s.lower()
    if any(m in low for m in _TRANSIENT_FAILURE_MARKERS):
        return False
    if s.strip().rstrip(".").lower() in _EMPTY_SUCCESS_MARKERS:
        return False
    return _is_tool_failure(result)


def _loop_break_nudge(tool: str, n: int) -> str:
    """生成同一工具连续确定性失败 ``n`` 次后注入模型的改路提示。

    ``tool`` 是失败的 Tool 名称，``n`` 是当前连续次数。返回文本不会替模型选择具体方案，
    而是明确禁止不变重试，并按外部依赖、文件路径和其他失败三类给出检查方向。它只在
    `_is_hard_tool_failure` 已确认失败可复现后使用，目的是让模型改变工具、命令或策略，
    而不是把瞬时网络错误误判为 Agent 卡死。
    """
    return (
        f"[loop] `{tool}` has failed {n} times in a row with the same kind of error. "
        "Stop repeating it. If it is an external dependency (network/API/search), "
        "complete what you can offline from local data and report what stayed blocked. "
        "If it is a file or path error, re-examine the EXACT path before any retry — "
        "do not call it again unchanged. Otherwise change approach: a different tool, "
        "command, or strategy."
    )


class AgentLoop:
    """统筹一个 Agent Turn 从 Spine 入站到回复出站的核心处理引擎。

    对初学者而言，`AgentLoop` 不是“无限让模型思考”的循环，而是有预算、有资源生命周期、
    有单一交付边界的请求协调器。它依次接收 Spine 消息，使用历史、Memory 和 Skill 构建
    Context，调用 LLM，执行模型发出的 Tool 调用，并把文本、推理增量、工具事件、媒体与
    Usage 送回同一个 `emit`。模型若请求工具，新的工具结果会进入下一次模型调用，直到得到
    最终回答或抵达明确终态。

    实例通常由 Gateway 等 Host 长期持有，因此它同时拥有 ToolRegistry、Context Engine、
    SessionManager、SubagentManager、Sandbox executor、MCP 连接和后台个性化任务。`run`
    只负责把这些运行时资源拉起并保持存活，真正的一轮请求从 `run_turn` 进入；`close` 与
    `stop` 则分别完成异步资源清理和停止保活循环。跨 Turn 的可变状态必须显式按
    `session_key` 隔离，不能把一次请求的恢复或失败计数泄漏给另一段会话。
    """

    _TOOL_RESULT_MAX_CHARS = 16_000
    # 每个 Turn 在上下文溢出成为致命错误前，最多执行的紧急收缩次数。
    _MAX_COMPRESS_RETRIES = 2
    # 紧急收缩时保持完整的最新工具结果数；更旧结果被省略，
    # 因为它们的正文是 Turn 中途上下文增长的主体。
    _SHRINK_KEEP_RECENT_TOOL_RESULTS = 3
    # 工具失败循环打断：同一工具连续确定性失败达到此次数后发出提示；
    # 同时限制每个 Turn 的提示次数，避免提示本身形成循环。
    _LOOP_BREAK_THRESHOLD = 2
    _LOOP_BREAK_MAX = 2

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        context_window_tokens: int = 65_536,
        brave_api_key: str | None = None,
        web_proxy: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        sandbox_config: SandboxConfig | None = None,
        channels_config: ChannelsConfig | None = None,
        router: "ModelRouter | None" = None,
        strategies: "StrategyRegistry | None" = None,
        call_efficiency: "CallEfficiency | None" = None,
        skill_forge_config: Any = None,
        hooks: "CompositeHook | None" = None,
        now_fn: Callable | None = None,
        context_config: "ContextConfig | None" = None,
        runtime_config: "RuntimeConfig | None" = None,
        interactive: bool = True,
        jina_api_key: str | None = None,
        max_concurrent_subagents: int = 4,
        max_subagent_spawns_per_hour: int = 30,
        disabled_tools: list[str] | None = None,
        tool_search_config: Any = None,
        # 可选的插件 MemoryBackend。None 禁用后端回忆和存储，但保留 Session 和本地 Skill。
        backend: "MemoryBackend | None" = None,
        # 转发给 ``build_context_engine``，供 Memory 通道和本地 Skill 路由器使用。
        memory_config: "MemoryConfig | None" = None,
        skill_forge_router_config: "SkillForgeRouterConfig | None" = None,
        # 已激活插件贡献的工具，由 CLI 通过 ``build_plugin_tools`` 构建，并在
        # ``_register_default_tools`` 中与内置工具一起注册。None 或空列表表示无插件工具，
        # 默认行为不变。
        plugin_tools: "list[Tool] | None" = None,
        empty_recovery: RecoveryLimits | None = None,
        context_engine_factory: "ContextEngineFactory | None" = None,
        state: Path | None = None,
    ):
        from codeflow.agent.hook import CompositeHook
        from codeflow.call_efficiency import CallEfficiency
        from codeflow.config.schema import ExecToolConfig
        from codeflow.token_wise.registry import StrategyRegistry

        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.state = state or workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        # 空响应恢复预算。None 表示使用已启用的默认值。
        self._recovery_limits = empty_recovery if empty_recovery is not None else RecoveryLimits()
        self.context_window_tokens = context_window_tokens
        self.brave_api_key = brave_api_key
        self.jina_api_key = jina_api_key
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        # 旧版注册表继续作为工具列表和基准扩展边界。
        self.strategies = strategies if strategies is not None else StrategyRegistry([])
        self.call_efficiency = call_efficiency or CallEfficiency.disabled()
        # 基准和模拟框架的伪时钟注入点。默认使用墙上时钟，不影响网关和 REPL 等生产路径。
        # 它同时用于会话项时间戳并传入 ContextBuilder，使 LLM 提示词中的 ``Current Time:``
        # 与持久化消息记录的时间保持同步。
        self._now_fn = now_fn or datetime.now

        self.backend: "MemoryBackend | None" = backend
        self.memory_enabled = backend is not None

        # 已激活插件贡献的工具，由 ``_register_default_tools`` 注册到 ToolRegistry。
        self.plugin_tools: "list[Tool]" = list(plugin_tools or [])

        self.context = ContextBuilder(
            workspace,
            state=self.state,
            skill_forge_config=skill_forge_config,
            llm_provider=self.provider,
            now_fn=now_fn,
        )
        self.sessions = session_manager or SessionManager(self.state)
        # 从注册表中排除的工具名称。在默认工具注册和 MCP 连接后均应用，因此可同时限制两组。
        # 供 BCP 等需要严格工具子集的评测框架使用。
        self._disabled_tools = set(disabled_tools or [])
        self._tool_search_config = tool_search_config
        self._resource_locks = ResourceLockManager()
        self.tools = ToolRegistry(resource_locks=self._resource_locks)

        # Context Engine 是唯一的 ContextAssembler。在 self.tools 之后于此构建，使工厂能将
        # ``self.tools.get_definitions`` 捕获为延迟可调用对象；真正的工具注册表内容
        # 稍后由同一构造函数中的 ``_register_default_tools`` 填充。
        #
        # 延迟导入 ``codeflow.context_engine`` 的原因见模块顶部关于 ``codeflow.agent.__init__`` 循环导入的说明。
        if context_config is None:
            from codeflow.config.codeflow import ContextConfig

            context_config = ContextConfig()
        if context_engine_factory is None:
            from codeflow.context_engine import build_context_engine

            context_engine_factory = build_context_engine
        self.context_config = context_config

        self.context_engine: "ContextEngine" = context_engine_factory(
            workspace=workspace,
            config=context_config,
            builder=self.context,
            provider=self.provider,
            model=self.model,
            context_window_tokens=context_window_tokens,
            get_tool_definitions=self.tools.get_definitions,
            now_fn=now_fn,
            # 工厂使用这些参数组装统一的 Memory 和本地 Skill 通道。
            backend=backend,
            memory_config=memory_config,
            skill_forge_router_config=skill_forge_router_config,
            skill_forge_config=skill_forge_config,
        )

        # 运行时约束（第五支柱）。检查点受 policy 和 interactive 联合门控，见 ``_checkpoint_active``。
        # 门禁关闭时，Agent Loop 与基线字节级一致。
        if runtime_config is None:
            from codeflow.config.codeflow import RuntimeConfig

            runtime_config = RuntimeConfig()
        self.runtime_config = runtime_config
        self.interactive = interactive
        self._checkpoint = None
        if self._checkpoint_active(runtime_config.checkpoint.policy, interactive):
            from codeflow.agent.loop.checkpoint import CheckpointService

            try:
                self._checkpoint = CheckpointService(
                    workspace,
                    shadow_dir=runtime_config.checkpoint.shadow_dir,
                    state=self.state if self.state != workspace else None,
                )
            except ValueError as exc:
                # shadow_dir 无效（如 ``../escape`` 或绝对路径）时 CheckpointService 拒绝构建。
                # 不因配置笔误让整个 Agent 崩溃；记录日志并禁用安全网，使 Turn 仍可运行。
                logger.warning("runtime.checkpoint disabled — {}", exc)
        # Turn 因迭代上限中断时，保存 session_key -> {"checkpoint_id", "files"}；
        # 下一个 Turn 的恢复提示词会消费它。
        self._pending_recovery: dict[str, dict] = {}

        self._sandbox_config = sandbox_config
        self._owned_ids: set[str] = set()
        self.subagents = SubagentManager(
            provider=self.provider,
            workspace=workspace,
            state=self.state,
            model=self.model,
            brave_api_key=brave_api_key,
            jina_api_key=jina_api_key,
            web_proxy=web_proxy,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
            sandbox_config=sandbox_config,
            owned_ids=self._owned_ids,
            max_concurrent=max_concurrent_subagents,
            max_spawns_per_hour=max_subagent_spawns_per_hour,
            resource_locks=self._resource_locks,
        )

        # 执行器此处只同步构建，虚拟机在 _start_executor() 中启动。
        self._executor: SandboxExecutor = build_executor(sandbox_config, workspace, self._owned_ids)
        self._executor_stack: AsyncExitStack | None = None
        self._executor_started: bool = False
        self._executor_start_lock = asyncio.Lock()
        self._debug_server: SandboxDebugServer | None = None

        self.router = router
        self.enable_personalization = False  # 通过 configure_personalization() 设置
        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._processing_lock = asyncio.Lock()
        # 每个已分派 Turn 结束后触发，无论成功、错误还是取消。回调必须低成本且不得抛错。
        self.on_turn_complete: list[Callable[[], None]] = []
        self.memory_consolidator = MemoryConsolidator(
            workspace=self.state,
            provider=self.provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=context_window_tokens,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
            now_fn=now_fn,
        )

        self._personalization_tasks: set[asyncio.Task[Any]] = set()
        self._personalization_closed = False
        self._close_lock = asyncio.Lock()
        self._closed = False

        # B-3 阶段已移除 L4 外观（``DefaultMemoryEngine`` / ``MemoryEngine`` 抽象基类）。
        # AgentLoop 现在直接持有底层子系统：
        #
        # - ``self.memory_consolidator`` 负责 Markdown 压缩策略，并拥有它构建的 ``MemoryStore``；
        #   需要时通过 ``self.memory_consolidator.store`` 访问。
        # - ``self.context.skills`` 是 :class:`LocalSkillCatalog`，负责常驻 Skill 和 ``# Skills`` 渲染路径。
        #   由 ``context_engine.factory`` 组装的 SkillForgeRouter 栈拥有检索职责。

        # AgentHook 生命周期链。评测 Hook 和调用方 Hook 共享同一个有序接口，
        # 无需按功能定制回调适配器。
        self.hooks: "CompositeHook" = CompositeHook()
        if hooks is not None:
            self.hooks.extend(hooks)

        self._register_default_tools()
        self._apply_disabled_tools()

    def _apply_disabled_tools(self) -> None:
        """从当前注册表移除 ``tools.disabled_tools`` 指定的 Tool。

        这一步必须在 :meth:`_register_default_tools` 注册内置与插件工具后执行，也必须在
        :meth:`_connect_mcp` 动态加入 MCP 工具后再次执行，否则同一黑名单只能覆盖其中一组。
        名称未注册时保持静默：评测配置经常给出比当前构建实际工具更宽的列表，缺失项应当是
        no-op，而不是让 Agent 无法启动。方法原地修改 `ToolRegistry`，没有返回值。
        """
        if not self._disabled_tools:
            return
        for name in self._disabled_tools:
            if self.tools.has(name):
                self.tools.unregister(name)

    def configure_personalization(self, enable: bool) -> None:
        """设置全局四阶段个性化流程开关，该流程受 PAHF 思路启发。

        开启后，一条消息先由 ``classify()`` 判断是否需要询问偏好；需要时在执行前只问一个
        问题并提取、保存答案；随后进入完全不变的普通 Agent Loop；回复完成后再把
        ``post_learn()`` 作为后台任务运行。这样偏好交互围绕主执行链展开，而不是替换它。

        ``enable`` 只是调用方意图，真正状态还受 `memory_enabled` 约束：没有 MemoryBackend
        时无法持久化学习结果，即使传入 ``True`` 也会保持关闭。功能默认禁用，可由配置
        ``agents.defaults.enable_personalization: true`` 开启。方法只更新实例状态并记录日志。
        """
        self.enable_personalization = bool(enable and self.memory_enabled)
        logger.info("Personalization flow: {}", "enabled" if self.enable_personalization else "disabled")

    def _start_personalization_task(self, factory: Callable[[], Awaitable[Any]]) -> None:
        if self._personalization_closed:
            return
        task = asyncio.create_task(factory())
        self._personalization_tasks.add(task)
        task.add_done_callback(self._personalization_tasks.discard)

    def begin_close(self) -> None:
        """同步封住新的个性化后台任务，并取消已经启动的任务。

        关闭流程需要先阻止 `_start_personalization_task` 再接纳工作，否则等待旧任务时仍可能
        产生新任务。该方法把 `_personalization_closed` 设为真并对当前任务快照调用
        `cancel()`，整个过程不发生 ``await``，所以调用方可在进入异步 drain 前建立清晰屏障。
        重复调用是 no-op；真正等待取消完成由 `close` 负责。
        """
        if self._personalization_closed:
            return
        self._personalization_closed = True
        for task in tuple(self._personalization_tasks):
            task.cancel()

    def _register_default_tools(self) -> None:
        """按运行时约束组装并注册 Agent 默认可用的 Tool 集合。

        文件、Shell、Web、消息、子 Agent、提问和可选 Cron Tool 在这里绑定工作区、代理、
        Sandbox executor 等依赖。插件工具随后以 ``replace=True`` 注册，因此插件可有意覆盖
        同名内置实现；渐进式 Tool Search 最后建立，才能看到此前完整目录。MCP Tool 不在
        此处连接，而由 `_connect_mcp` 在首次 Turn 前延迟加入。

        方法只完成注册，不代表所有工具都最终可见：调用方紧接着执行
        `_apply_disabled_tools`，Tool Search 策略也可能按 Turn 缩小暴露集合。若启用 Cron，
        函数内延迟导入是为避开 `codeflow.agent.__init__` 的循环导入边界。
        """
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool, GrepTool, FindTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(SkillReadTool(self.context.skills))
        self.tools.register(
            ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
                path_append=self.exec_config.path_append,
                executor=self._executor,
            )
        )
        self.tools.register(WebSearchTool(api_key=self.brave_api_key, proxy=self.web_proxy))
        self.tools.register(WebFetchTool(api_key=self.jina_api_key, proxy=self.web_proxy))
        self.tools.register(MessageTool())
        self.tools.register(SpawnTool(manager=self.subagents))
        # QuestionBroker 按传输层为单例；当传输层（TUI RPC 服务器或网关 hub）存在后，
        # 通过 set_broker 延迟绑定。
        self.tools.register(AskUserTool())
        if self.cron_service:
            # 延迟导入：CronTool 所在模块会导入 codeflow.agent.tools.base，触发 codeflow.agent.__init__，
            # 后者又导入当前循环模块。在函数作用域导入可打破循环，因为执行
            # _register_default_tools 时 loop.py 已完全加载。
            from codeflow.proactive_engine.schedulers.cron.tool import CronTool

            self.tools.register(CronTool(self.cron_service))

        # 插件贡献的工具最后注册，使插件在有意提供同名工具时能覆盖内置实现。
        # 随后仍会运行 ``_apply_disabled_tools``，因此任何工具都可被移除。
        for tool in self.plugin_tools:
            self.tools.register(tool, replace=True)

        # 渐进式工具披露最后注册，使其搜索目录覆盖上方全部内置和插件工具。
        # MCP 工具稍后在 ``_connect_mcp`` 中加入；策略每个 Turn 都重读注册表，因此能自动获取。
        cfg = self._tool_search_config
        if cfg is not None and cfg.enabled:
            from codeflow.agent.tools.tool_search import (
                DEFAULT_ALWAYS_VISIBLE,
                ToolCallTool,
                ToolSearchController,
                ToolSearchStrategy,
                ToolSearchTool,
            )

            always = set(DEFAULT_ALWAYS_VISIBLE) | set(cfg.always_visible)
            self.tool_search_controller = ToolSearchController(
                self.tools,
                always_visible=always,
                search_result_limit=cfg.search_result_limit,
            )
            self.tools.register(ToolSearchTool(self.tool_search_controller))
            self.tools.register(ToolCallTool(self.tool_search_controller))
            # ``first=True`` 表示在 CacheOptimizer 用 ``cache_control`` 标记最后一个工具前先过滤列表；
            # 否则已标记的工具可能被过滤，导致缓存断点丢失。
            self.strategies.register(
                ToolSearchStrategy(
                    self.tool_search_controller,
                    compaction_threshold=cfg.compaction_threshold,
                ),
                first=True,
            )

    # ── 上下文引擎辅助方法 ─────────────────────────────────────────────

    def _context_messages_for_session(self, session: Session) -> list[dict[str, Any]]:
        """按当前 Context Engine 的所有权返回待组装会话消息视图。

        `Session` 保存追加式消息记录，但不同引擎对压缩的责任不同。Curator 在
        ``owns_compaction=True`` 时必须看到完整 append-only log，才能自行决定哪些内容归档；
        Legacy 引擎不拥有该决策，只读取 `get_history(max_messages=0)` 提供的 consolidation 后
        切片，以保持引入 Curator 前的行为。返回的是新列表视图，方法不会修改 Session。
        """
        if self.context_engine.owns_compaction:
            return list(session.messages)
        return session.get_history(max_messages=0)

    def _make_token_budget(self, selected_skills: list[Any] | None = None) -> TokenBudget:
        """为当前 Context Engine 计算一份保守的单 Turn Prompt 预算。

        预算从 `context_window_tokens` 总窗口中依次预留模型最大输出、当前 Tool definitions
        和 System Prompt 所需 Token，剩余值才写入 `available_history` 供历史消息使用。
        ``selected_skills`` 会影响 System Prompt 大小，因此必须在估算时传入；Provider 未
        声明生成上限时按 4096 预留。任何扣减导致的负数都会收敛为 0，避免把超额空间伪装
        成可用历史。返回 `TokenBudget`，不裁剪消息本身。
        """
        reserved_output = int(getattr(getattr(self.provider, "generation", None), "max_tokens", 4096) or 4096)
        tool_tokens = estimate_prompt_tokens([], self.tools.get_definitions())
        system_prompt = self.context.build_system_prompt(
            selected_skills,
            include_memory=self.memory_enabled,
        )
        system_tokens = estimate_prompt_tokens([{"role": "system", "content": system_prompt}])
        available_history = max(
            0,
            self.context_window_tokens - reserved_output - tool_tokens - system_tokens,
        )
        return TokenBudget(
            context_length=self.context_window_tokens,
            reserved_output=reserved_output,
            reserved_tools=tool_tokens,
            reserved_system=system_tokens,
            available_history=available_history,
        )

    async def _select_skills_for_turn(
        self,
        current_message: str,
        history: list[dict],
    ) -> list[Any] | None:
        """保留旧调用形状，但不在 Host 侧预选 Skill。

         统一 Context Engine 内部的 `SkillForgeRouter` 同时拥有 Skill 选择与渲染，随后通过
        每 Turn 的 assembly metadata 暴露 ``injected_skill_ids``。因此 ``current_message``
         与 ``history`` 在这条兼容入口中不会被再次消费，也没有 `SkillMeta` 列表流回 Host；
         方法始终返回 ``None``。把选择留在 Engine 内可避免 Host 与 Context 对同一预算做两次
         决策。
        """
        return None

    async def _assemble_context_messages(
        self,
        *,
        session: Session,
        session_key: str,
        current_message: str,
        media: list[str | Media] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        selected_skills: list[Any] | None = None,
        metadata_sink: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """请求当前 Context Engine 组装主 Agent 本轮实际可见的消息窗口。

        方法先依据压缩所有权取得 Session 候选消息，再把 ``current_message``、媒体、通道、
        会话标识和已选 Skill 封装为 `TurnContext`，连同 `_make_token_budget` 的结果交给
        `context_engine.assemble`。返回值是可直接发给 Provider 的消息列表，不是完整 Session。

        若提供 ``metadata_sink``，Engine 产出的 Memory 命中、Skill 注入和回退路径等元数据
        会原地写入该字典，供 `run_turn` 形成证据；随后 `_inject_recovery_block` 可能把上轮
        中断通知加到当前用户消息前。`TurnContext` 使用延迟导入，以维持模块顶部说明的循环
        导入边界。
        """
        from codeflow.context_engine import TurnContext  # 延迟导入，原因见模块说明

        session_messages = self._context_messages_for_session(session)
        assembled = await self.context_engine.assemble(
            session_key,
            session_messages,
            self._make_token_budget(selected_skills),
            turn=TurnContext(
                current_message=current_message,
                media=media,
                channel=channel,
                chat_id=chat_id,
                selected_skills=selected_skills,
            ),
        )
        if metadata_sink is not None:
            metadata_sink.update(assembled.metadata or {})
        messages = assembled.messages
        self._inject_recovery_block(session_key, messages)
        return messages

    @staticmethod
    def _checkpoint_active(policy: str, interactive: bool) -> bool:
        """结合调用现场是否交互，解析 ``runtime.checkpoint.policy`` 是否启用快照。

        默认策略 ``"interactive"`` 直接采用 ``interactive`` 信号：持续会话可能有 "next
        turn" 可注入恢复信息，而一次性 ``-m`` 调用没有下一轮，创建快照只会增加无用成本。
        ``"always"`` 无视调用形态强制开启，``"never"`` 无条件关闭。返回布尔值，只决定本次
        AgentLoop 是否构建 CheckpointService，不创建任何快照。
        """
        if policy == "never":
            return False
        if policy == "always":
            return True
        return interactive  # policy 为 interactive

    def _stash_recovery(self, session_key: str, outcome: "TurnOutcome") -> None:
        """暂存一次中断 Turn 的快照证据，供同一 Session 的下一轮生成恢复提示。

        只有 Checkpoint 已启用、``outcome.status`` 精确为 ``"interrupted"``，并且确实存在
        `checkpoint_id` 或 `edited_files` 时，方法才会把证据写入按 ``session_key`` 隔离的
        `_pending_recovery`。其他情况是 no-op，避免为空现场制造恢复通知。

        状态过滤是有意的：``"error"`` Turn 仍可能产生逐 Turn shadow commit 用于审计，但
        Provider 400 等错误通常没有可继续的部分编辑轨迹；此时向用户展示 "Files modified
        last turn" 会造成误导。这里保存的是下一轮要核验的线索，不把快照等同于恢复成功。
        """
        if self._checkpoint is None or outcome.status != "interrupted":
            return
        if outcome.edited_files or outcome.checkpoint_id:
            self._pending_recovery[session_key] = {
                "checkpoint_id": outcome.checkpoint_id,
                "files": outcome.edited_files,
            }

    def _inject_recovery_block(self, session_key: str, messages: list[dict]) -> None:
        """把上轮中断现场的恢复通知前置到当前用户消息，并在成功后一次性消费。

        方法按 ``session_key`` 查找待恢复记录，只在消息列表以 ``role="user"`` 结尾时处理。
        通知包含可用的文件清单、Checkpoint 标识以及“先核对现状再继续”的要求；字符串内容
        直接前置，多模态列表则插入一个文本块。这样 Provider 在读取用户本轮要求前先看到
        上次未完成工作的证据边界。

        若 ``content`` 是 None、dict 或其他未知形状，方法不会猜测如何改写，也不会删除待
        恢复记录；后续一次正常 assembly 仍可注入。只有内容已安全写入后才从
        `_pending_recovery` 弹出，保证异常形状不会让恢复线索静默丢失。
        """
        recovery = self._pending_recovery.get(session_key)
        if not recovery or not messages:
            return
        last = messages[-1]
        if last.get("role") != "user":
            # 最后一条消息不是用户 Turn 时，保持恢复待处理，使下次以用户消息结尾的组装注入它。
            return
        content = last.get("content")
        files = recovery.get("files") or []
        cid = recovery.get("checkpoint_id")
        lines = ["[Recovery — the previous turn was interrupted before finishing]"]
        if files:
            lines.append("Files modified last turn: " + ", ".join(files))
        if cid:
            lines.append(f"Checkpoint: {cid}")
        lines.append("Verify the current state of these files before continuing.")
        block = "\n".join(lines)
        # 先修改、后弹出，从调用方视角看是原子操作。如果无法安全写入形状未知的 ``content``，
        # 恢复保持待处理，而不是被静默丢弃。
        if isinstance(content, str):
            last["content"] = f"{block}\n\n{content}"
        elif isinstance(content, list):
            last["content"] = [{"type": "text", "text": block}] + content
        else:
            return  # 内容形状异常时保持待恢复状态
        self._pending_recovery.pop(session_key, None)

    @trace.instrument("memory.store", extract=semconv.memory_store)
    async def _dispatch_backend_store(
        self,
        session_key: str,
        messages_slice: list[dict],
    ) -> None:
        """AG-1：把本轮新增消息转交插件 :class:`MemoryBackend` 持久化或建立索引。

        一轮结束后有三个并列步骤：``context_engine.after_turn`` 处理 Engine 自身账本，
        ``memory.maybe_consolidate`` 执行 CodeFlow compaction，本方法负责插件 Backend。传入的
        ``messages_slice`` 会先经 `sanitize_persisted_payload` 清理再调用 `store`，而不是让
        后端接触未经约束的运行时对象。

        当 ``self.backend is None`` 或消息切片为空时返回 no-op，使从未注册插件的旧调用方
        与 pre-AG-1 行为一致。Backend failure 会在 append-only Session 已保存之后继续向外
        传播：Session 事实不会回滚，但 Turn 也不能在索引失败时被报告为成功。
        """
        if self.backend is None or not messages_slice:
            return
        await self.backend.store(
            session_key,
            sanitize_persisted_payload(messages_slice),
        )

    def _collect_injected_skill_ids(
        self,
        selected: list[Any] | None,
    ) -> list[str]:
        """合并本轮 selector top-K 与 always-skills，生成去重后的 Skill 证据标识。

        ``selected`` 是检索 selector 返回的 :class:`SkillMeta` 列表；selector 关闭或无命中时
        可以是 ``None``。always-skills 无论检索结果如何都会由 :class:`LocalSkillCatalog`
        渲染，因此还要从 `self.context.skills` 单独读取，不能只记录 top-K。

        不同 SkillMeta 生产者对 ``meta.id`` 的填法并不一致：有的已经带 source，有的只有
        stable key。方法统一规范为 ``{source}/{stable_key}``，按首次出现顺序去重后返回，
        供 Turn evidence 写入 ``injected_skill_ids``。目录不存在或读取 always-skills 失败时
        保守返回已有结果，不让证据收集阻断主请求。
        """
        skills_svc = getattr(self.context, "skills", None)
        if skills_svc is None:
            return []

        seen: set[str] = set()
        ids: list[str] = []

        def _add(meta: Any) -> None:
            src = getattr(meta, "source", None)
            mid = getattr(meta, "id", None)
            if not src or not mid:
                return
            canonical = mid if "/" in mid else f"{src}/{mid}"
            if canonical not in seen:
                seen.add(canonical)
                ids.append(canonical)

        for meta in selected or []:
            _add(meta)
        try:
            always = skills_svc.get_always_skills()
        except Exception:
            always = []
        for meta in always:
            _add(meta)
        return ids

    async def _start_executor(self) -> None:
        """在首次使用前启动 Sandbox executor，并保证并发调用下只启动一次。

        `_executor_start_lock` 串行化竞争者；已经启动时立即返回。首次启动会创建并进入
        `AsyncExitStack`，再把 executor 作为异步上下文压栈，使关闭责任集中到
        `close_executor`。若进入过程中抛出异常，临时 stack 会先清理再把异常原样传播，
        `_executor_started` 也不会被误设为真。
        """
        async with self._executor_start_lock:
            if self._executor_started:
                return
            stack = AsyncExitStack()
            try:
                await stack.__aenter__()
                await stack.enter_async_context(self._executor)
            except Exception:
                await stack.aclose()
                raise
            self._executor_stack = stack
            self._executor_started = True

    async def _start_debug_server(self) -> None:
        """在 Sandbox 调试模式启用时启动本地 socket 调试服务器。

        未配置 Sandbox、``debug.enabled`` 为假或 backend 为 ``"none"`` 时不会创建服务器；
        最后一种情况会记录警告，因为没有 BoxLite runtime 可供调试。启用路径根据数据目录
        解析 socket，传入当前 executor 拥有的实例 ID 和消息大小上限，再保存已启动的
        `SandboxDebugServer` 供关闭阶段使用。

        用户显式选择调试模式后，启动失败会记录具体错误但不终止整个 Agent runtime；这让
        主请求仍可运行，同时避免稍后 `codeflow sandbox` 只看到“套接字不存在”却没有原因。
        """
        cfg = self._sandbox_config
        if cfg is None or not cfg.debug.enabled:
            return
        if cfg.backend == "none":
            logger.warning(
                "sandbox.debug.enabled=true is ignored because backend='none' (no boxlite runtime is active)"
            )
            return
        try:
            from codeflow.config.paths import get_data_dir
            from codeflow.sandbox.debug_server import SandboxDebugServer

            socket_path = SandboxDebugServer.resolve_socket_path(cfg.debug.socket, get_data_dir())
            server = SandboxDebugServer(
                socket_path=socket_path,
                owned_ids=self._owned_ids,
                max_message_bytes=cfg.debug.max_message_bytes,
            )
            await server.start()
            self._debug_server = server
        except Exception as exc:
            # 用户已明确选择调试模式；此处静默失败会让用户在稍后看到 `codeflow sandbox`
            # 报“套接字不存在”时困惑。需明确记录原因。
            logger.error("Failed to start sandbox debug server: %s", exc)

    async def close_executor(self) -> None:
        """关闭调试服务器和 Sandbox executor，并把实例状态复位为可重新启动。

        调试 socket 先停止，随后关闭保存 executor 生命周期的 `AsyncExitStack`。停止阶段的
        普通异常会记录或被容忍，`RuntimeError` 与 `BaseExceptionGroup` 也不会让 shutdown
        卡住；无论资源此前是否完整启动，最终都会清空引用并把 `_executor_started` 设为假。
        方法可重复调用，适用于 Turn 失败后的急停和正常 runtime 关闭两条路径。
        """
        if self._debug_server is not None:
            try:
                await self._debug_server.stop()
            except Exception as exc:
                logger.warning("Error stopping sandbox debug server: %s", exc)
            self._debug_server = None
        if self._executor_stack:
            try:
                await self._executor_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass
            self._executor_stack = None
        self._executor_started = False

    async def _connect_mcp(self) -> None:
        """在首次需要时一次性连接已配置的 MCP servers，并注册其 Tool。

        已连接、正在连接或没有配置时立即返回。方法在第一个 ``await`` 前设置
        `_mcp_connecting`，利用 asyncio 单线程任务切换边界阻止重复连接；随后确保 Sandbox
        executor 已启动，用独立 `AsyncExitStack` 承担 MCP 连接的关闭责任，并把远端 Tool
        加入当前 `ToolRegistry`。连接后再次执行 `_apply_disabled_tools`，使黑名单也覆盖
        动态加入的 ``mcp_<server>_search`` 等名称。

        任一步失败都会复位 connecting 标志、关闭部分建立的 stack 并重新抛出异常，因此
        当前 Turn 能准确失败，后续调用也仍有机会重试，而不会永久停在“正在连接”状态。
        """
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        # 在第一个 await 之前同步设置标志。asyncio 单线程执行，此处不会发生上下文切换，
        # 该互斥模式无需锁。
        self._mcp_connecting = True
        try:
            await self._start_executor()  # MCP 服务器连接前，执行器必须已启动
            from codeflow.agent.tools.mcp import connect_mcp_servers

            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(
                self._mcp_servers,
                self.tools,
                self._mcp_stack,
                executor=self._executor,
            )
            # 重新应用黑名单：MCP 服务器可能注册也出现在 ``disabled_tools`` 中的工具名，
            # 例如 ``mcp_<server>_search``。
            self._apply_disabled_tools()
            self._mcp_connected = True
            self._mcp_connecting = False
        except Exception:
            # 重置进行中标志，使后续调用可重试。
            self._mcp_connecting = False
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
            raise

    def _set_tool_context(
        self, channel: str, chat_id: str, message_id: str | None = None, session_key: str | None = None
    ) -> None:
        """把当前消息的路由上下文写入需要感知出口的 Tool。

        ``channel``、``chat_id`` 和可选 ``message_id`` 交给 message Tool，使它能把回复送回
        原会话；spawn Tool 还需要 ``session_key`` 关联父子 Turn，未提供时退化为
        ``{channel}:{chat_id}``；cron Tool 只接收通道与聊天标识。没有 `set_context` 的工具
        会被跳过。方法只更新当前注册表中的 message、spawn、cron 三类，不向所有 Tool
        强加路由协议。
        """
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                if not hasattr(tool, "set_context"):
                    continue
                if name == "message":
                    tool.set_context(channel, chat_id, message_id)
                elif name == "spawn":
                    tool.set_context(channel, chat_id, session_key or f"{channel}:{chat_id}")
                else:
                    tool.set_context(channel, chat_id)

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """移除部分模型混入最终内容的 ``<think>…</think>`` 推理块。

        输入为空时返回 ``None``；有文本时跨行删除所有成对标签及内部内容，再去除首尾空白。
        清理后没有可见正文也返回 ``None``，让上层进入空响应恢复而不是交付空字符串。该方法
        只处理显式标签，不猜测普通文本是否属于推理，也不修改独立的
        `reasoning_content` 字段。
        """
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """把一组 Tool 调用压缩成供进度出口展示的简短提示。

        每个调用优先取参数字典中的第一个字符串值，渲染成例如
        ``'web_search("query")'`` 的形式；值超过 40 个字符时截断并追加省略号。参数不是
        字典、没有值或首值不是字符串时只展示 Tool 名，避免把复杂载荷泄漏到进度提示。
        多个调用以逗号连接，返回值仅用于人类可读状态，不参与实际 Tool 执行。
        """

        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'

        return ", ".join(_fmt(tc) for tc in tool_calls)

    @staticmethod
    def _build_usage_snapshot(response, model: str, session_key: str) -> "UsageSnapshot":
        """经统一计费归一化构建历史 TokenWise 兼容视图。

        方法用禁用上报的 `CallEfficiency` 记录 ``response``，同时带入请求 ``model`` 与
        ``session_key``，让模型名称、Token 和成本字段先走当前 canonical normalization；
        随后调用 `to_legacy_snapshot()` 转成旧 TokenWise 消费者期望的 `UsageSnapshot`。
        它不产生外部遥测，也不绕过统一计费规则直接拼字段。
        """
        from codeflow.call_efficiency import CallEfficiency

        return (
            CallEfficiency.disabled()
            .record(
                response,
                requested_model=model,
                session_key=session_key,
            )
            .to_legacy_snapshot()
        )

    @trace.instrument("llm.call", extract=semconv.llm_call_stream)
    async def _llm_call_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        model: str | None,
        on_token_delta: Callable[[str], Awaitable[None]] | None = None,
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """通过 ``provider.chat_stream`` 接收增量，并重组为完整 `LLMResponse`。

        按 design.md §D3，Turn 调用方连接 ``on_token_delta`` 或推理回调时，AgentLoop 会走
        本方法而不是 ``chat_with_retry``。每个非空内容片段立即触发对应回调，同时累积文本、
        reasoning、usage、finish reason、模型和错误分类；Tool call 片段交给
        `_merge_tool_call_fragments` 按 ``index`` 汇合，最终对象与普通 ``chat()`` 返回形状兼容。

        v0.1 first-cut 合并边界仍然保留：同一位置只表示一个 Tool call，片段按顺序抵达，
        ``id`` / ``function.name`` 通常在该位置首片段出现，``function.arguments`` 由各片段
        JSON 字符串直接连接。Multi-tool 的 index 已区分，但更一般的 out-of-order 修复仍是
        v0.2 ask，不能把当前实现解释成任意乱序协议重组器。

        v0.1 stream mode 对临时错误不重试。已经向用户流出部分内容后，重头调用会重复输出，
        从 offset 恢复又依赖具体 Provider；在没有明确协议前选择传播错误，而不是制造看似
        完整但重复的回复。
        """
        content_buf: list[str] = []
        reasoning_buf: list[str] = []
        tool_call_slots: list[dict[str, Any]] = []
        final_usage: dict[str, Any] | None = None
        final_finish_reason: str | None = None
        final_error_classification: ErrorClassification | None = None
        actual_model: str | None = None
        cache_policy: str | None = None
        call_record = None

        async for delta in self.provider.chat_stream(
            messages=messages,
            tools=tools,
            model=model,
        ):
            delta_finish_reason = getattr(delta, "finish_reason", None)
            if delta_finish_reason is not None:
                final_finish_reason = delta_finish_reason
            delta_error_classification = getattr(delta, "error_classification", None)
            if delta_error_classification is not None:
                final_error_classification = delta_error_classification
            delta_model = getattr(delta, "model", None)
            if delta_model:
                actual_model = delta_model
            delta_cache_policy = getattr(delta, "cache_policy", None)
            if delta_cache_policy:
                cache_policy = delta_cache_policy
            delta_call_record = getattr(delta, "call_record", None)
            if delta_call_record is not None:
                call_record = delta_call_record
            is_error_delta = delta_finish_reason == "error"
            reasoning_delta = getattr(delta, "reasoning_content", None)
            if reasoning_delta:
                reasoning_buf.append(reasoning_delta)
                if on_reasoning_delta is not None and not is_error_delta:
                    await on_reasoning_delta(reasoning_delta)
            if delta.content:
                content_buf.append(delta.content)
                if on_token_delta is not None and not is_error_delta:
                    await on_token_delta(delta.content)
            if delta.tool_call_delta:
                _merge_tool_call_fragments(
                    tool_call_slots,
                    delta.tool_call_delta,
                )
            if delta.usage is not None:
                final_usage = delta.usage

        tool_calls = _finalize_tool_calls(tool_call_slots)
        finish_reason = final_finish_reason or ("tool_calls" if tool_calls else "stop")

        return LLMResponse(
            content="".join(content_buf),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=final_usage or {},
            reasoning_content="".join(reasoning_buf) or None,
            error_classification=final_error_classification,
            model=actual_model or model,
            cache_policy=cache_policy,
            call_record=call_record,
        )

    @classmethod
    def _emergency_shrink(cls, messages: list[dict]) -> tuple[list[dict], int]:
        """在 Turn 中途上下文溢出时，省略较旧 Tool 结果正文以腾出窗口。

        中途增长通常来自累积的 ``role="tool"`` 输出。方法保留最近
        `_SHRINK_KEEP_RECENT_TOOL_RESULTS` 条完整结果，把更旧且非空的正文替换成固定占位符；
        System、User、Assistant 推理以及消息顺序保持不变。它基于输入生成新列表和复制后的
        被改消息，不额外调用 LLM，因此收缩是确定且可计数的。

        返回 ``(new_messages, num_elided)``。当 ``num_elided == 0`` 时没有值得省略的旧结果，
        调用方不应靠相同重试期待窗口变小；原列表会直接返回。该机制保留最新操作现场，但
        明确牺牲旧 Tool 正文，不等同于无损压缩。
        """
        placeholder = "[earlier tool output elided to fit the context window]"
        tool_idxs = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
        if len(tool_idxs) <= cls._SHRINK_KEEP_RECENT_TOOL_RESULTS:
            return messages, 0
        elide = set(tool_idxs[: -cls._SHRINK_KEEP_RECENT_TOOL_RESULTS])
        shrunk: list[dict] = []
        elided = 0
        for i, m in enumerate(messages):
            if i in elide and m.get("content") and m.get("content") != placeholder:
                clean = dict(m)
                clean["content"] = placeholder
                shrunk.append(clean)
                elided += 1
            else:
                shrunk.append(m)
        return shrunk, elided

    async def _synthesize_final_on_exhaustion(
        self,
        messages: list[dict],
        model: str | None,
        fallback_models: list[str] | None,
        session_key: str = "",
        on_token_delta: Callable[[str], Awaitable[None]] | None = None,
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """迭代预算耗尽后，用一次禁用 Tool 的 LLM 调用生成本轮收尾回复。

        与直接返回固定道歉不同，方法把 `_MAX_ITER_SYNTHESIS_PROMPT` 追加为 User 消息，请模型
        只根据已经收集的材料总结完成项、部分结果和未完成项。调用明确传入 ``tools=None``，
        因而模型不能在预算悬崖边再发起 Tool call 或 ``ask_user``。响应出错、为空或自身抛出
        异常时返回 `_MAX_ITER_STATIC_FALLBACK`，保证 Turn 不会静默结束。

        调用方若连接了 streaming callbacks，合成回复也必须走 `_llm_call_stream` 并继续流出；
        否则 `run_turn` 在已有增量后会抑制结尾 ``Text``，非流式生成的收尾就会被出口丢弃。
        未连接回调时使用 Provider 的 `chat_with_retry`。成功响应仍经 CallEfficiency 记录和
        `_strip_think` 清理，返回值是最终可交付文本，不包含新增 Tool 结果。
        """
        synth_messages = messages + [{"role": "user", "content": _MAX_ITER_SYNTHESIS_PROMPT}]
        try:
            if on_token_delta is not None or on_reasoning_delta is not None:
                response = await self._llm_call_stream(
                    messages=synth_messages,
                    tools=None,
                    model=model,
                    on_token_delta=on_token_delta,
                    on_reasoning_delta=on_reasoning_delta,
                )
            else:
                response = await self.provider.chat_with_retry(
                    messages=synth_messages,
                    tools=None,
                    model=model,
                    fallback_models=fallback_models,
                )
            if response.call_record is None:
                response.call_record = self.call_efficiency.record(
                    response,
                    requested_model=model or self.model,
                    session_key=session_key,
                    cache_policy=response.cache_policy,
                )
            text = self._strip_think(response.content)
            if response.finish_reason != "error" and text:
                return text
            logger.warning(
                "Max-iter synthesis returned no usable content (finish_reason={})",
                response.finish_reason,
            )
        except Exception as exc:
            logger.warning("Max-iter synthesis call failed: {}", exc)
        fallback = _MAX_ITER_STATIC_FALLBACK.format(n=self.max_iterations)
        # 流式成功路径已通过 ``on_token_delta`` 交付文本，此回退路径则没有。因此也要将它推入流；
        # 否则一旦已有内容流出，run_turn 边界会抑制结尾 ``Text``，导致流式出口丢失回退文本。
        if on_token_delta is not None:
            await on_token_delta(fallback)
        return fallback

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        session_key: str | None = None,
        model: str | None = None,
        fallback_models: list[str] | None = None,
        injected_skill_ids: list[str] | None = None,
        on_token_delta: Callable[[str], Awaitable[None]] | None = None,
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
        on_tool_event: Callable[[str, dict], Awaitable[None]] | None = None,
        usage_sink: dict[str, Any] | None = None,
        drain: Drain | None = None,
        origin: Origin | None = None,
    ) -> tuple[str | None, list[str], list[dict], TurnOutcome]:
        """执行一次有预算的模型—Tool 迭代，并返回回复、证据与明确终态。

        `initial_messages` 是已经由 Context Engine 组装的模型窗口。每轮先读取 Tool
        definitions 和策略变换后的调用参数，再请求 LLM；若响应含 Tool call，就执行并把结果
        追加回消息，进入下一轮；若得到可用正文则结束。Provider error、空响应恢复耗尽或最大
        迭代数到达都会走各自的显式终止路径，不会被伪装成普通完成。

        连接 ``drain`` 时，每次迭代顶部都会拉取 BusyPolicy.INJECT 在 Turn 中途注入的用户
        消息，并在下一次 LLM 调用前合并为 User turn。``session_key`` 把 Usage、
        CallEfficiency 与 Checkpoint evidence 归属到 Host 对话；省略时使用空字符串，而不是
        猜测另一个会话。流式文本、推理和 Tool 事件分别经对应回调交付。

        返回四元组依次是最终文本或 ``None``、本轮使用的 Tool 名列表、清理过合成恢复脚手架
        的消息历史，以及本模块的 `TurnOutcome`。其中 ``status`` 区分 ``completed``、
        ``interrupted``、``error``；达到上限时还会尝试一次 tools-disabled 收尾，但有回复不
        会把“尚未完成”的 interrupted 事实改写成 completed。
        """
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        session_key = session_key or ""
        effective_model = model or self.model

        # 记录 Turn 是正常退出还是因迭代上限中断。下游只读取 ``status``，
        # 用于标记影子 Git 提交并写入 ``TurnOutcome``。
        status = "completed"
        error_category: str | None = None

        # 上下文溢出恢复：限制紧急收缩次数，避免省略后仍溢出的 Turn 永久循环。
        compress_retries = 0
        # 工具失败循环打断：跨迭代跟踪同一工具的连续硬失败；
        # 每个新连续段提示一次，并按 Turn 限制总数。
        loop_fail_tool: str | None = None
        loop_fail_streak = 0
        loop_nudges = 0
        # 空响应恢复状态属于当前 Turn。AgentLoop 是跨会话共享的长生命周期单例，
        # 实例级计数器会跨 Turn 泄漏；此处重置可为每个 Turn 提供干净预算。
        prev_had_tool_calls = False
        post_tool_nudges = 0
        prefill_retries = 0
        empty_retries = 0

        while iteration < self.max_iterations:
            iteration += 1
            logger.info(
                "Iteration {}/{} model={}",
                iteration,
                self.max_iterations,
                effective_model,
            )

            # 在本次迭代的 LLM 调用前合并所有通过 BusyPolicy.INJECT 注入的用户消息。
            # 携带媒体的注入会在文本中保留文件路径，避免内容静默丢失。
            if drain is not None:
                for inj in drain():
                    inj_text = inj.text or ""
                    inj_paths = [m.path for m in inj.media]
                    if inj_paths:
                        prefix = inj_text + "\n" if inj_text else ""
                        inj_text = f"{prefix}[injected message; attached files: {', '.join(inj_paths)}]"
                    if inj_text:
                        messages.append({"role": "user", "content": inj_text})
                        logger.info("inject: merged a mid-turn user message")

            tool_defs = self.tools.get_definitions()

            # 先运行历史策略，使工具过滤先于缓存规划。
            call_messages, call_tools, call_model = await self.strategies.before_llm_call(
                messages,
                tool_defs,
                effective_model,
            )
            if on_token_delta is not None or on_reasoning_delta is not None:
                response = await self._llm_call_stream(
                    messages=call_messages,
                    tools=call_tools,
                    model=call_model,
                    on_token_delta=on_token_delta,
                    on_reasoning_delta=on_reasoning_delta,
                )
            else:
                response = await self.provider.chat_with_retry(
                    messages=call_messages,
                    tools=call_tools,
                    model=call_model,
                    fallback_models=fallback_models,
                )
            call_record = response.call_record
            if call_record is None:
                call_record = self.call_efficiency.record(
                    response,
                    requested_model=call_model,
                    session_key=session_key,
                    cache_policy=response.cache_policy,
                )
                response.call_record = call_record
            usage_snapshot = call_record.to_legacy_snapshot()
            await self.strategies.after_llm_call(
                {
                    "content": response.content,
                    "finish_reason": response.finish_reason,
                    "usage": response.usage,
                },
                usage_snapshot,
            )
            # tui-chat L2-A 线路：流式调用方（turn.* 处理器）可能需要用最后一次迭代用量
            # 按 CAP-CHAT-1 形状填充 `message.complete.payload.usage`。应使用线上合约 UsageSnapshot
            # 的 prompt_tokens、completion_tokens 和 total_tokens，而非带模型、缓存和成本字段的
            # Agent 内部快照。
            if usage_sink is not None and response.usage:
                prompt_tokens = int(response.usage.get("prompt_tokens", 0) or 0)
                completion_tokens = int(response.usage.get("completion_tokens", 0) or 0)
                # LiteLLM 信息滞后时，从模型提供商表获取真实窗口；否则使用配置默认值。
                context_max = (
                    resolve_context_window(
                        call_record.accounting_model,
                        allow_litellm_import=False,
                    )
                    or self.context_window_tokens
                )
                context_used = prompt_tokens + completion_tokens
                usage_sink.clear()
                usage_sink["prompt_tokens"] = prompt_tokens
                usage_sink["completion_tokens"] = completion_tokens
                usage_sink["total_tokens"] = int(response.usage.get("total_tokens", 0) or 0)
                if usage_snapshot.estimated_cost_usd is not None:
                    usage_sink["cost_usd"] = usage_snapshot.estimated_cost_usd
                else:
                    usage_sink.pop("cost_usd", None)
                usage_sink["context_max"] = context_max
                usage_sink["context_used"] = context_used
                usage_sink["context_percent"] = round(100 * context_used / context_max) if context_max else 0

                # 上下文窗口溢出恢复：结构化分类器标记 should_compress。更小窗口无济于事，
                # 但省略大量累积工具输出有效。就地收缩并重试当前迭代，而不暴露为致命错误；
                # 重试次数受限。
            cls_ = response.error_classification
            if (
                response.finish_reason == "error"
                and cls_ is not None
                and cls_.should_compress
                and compress_retries < self._MAX_COMPRESS_RETRIES
            ):
                shrunk, elided = self._emergency_shrink(messages)
                if elided > 0:
                    messages = shrunk
                    compress_retries += 1
                    iteration -= 1  # 溢出的调用没有执行工作，不计入迭代次数
                    logger.warning(
                        "Context overflow; elided {} old tool result(s), retrying ({}/{})",
                        elided,
                        compress_retries,
                        self._MAX_COMPRESS_RETRIES,
                    )
                    continue

            if response.has_tool_calls:
                if on_progress:
                    thought = self._strip_think(response.content)
                    if thought:
                        await on_progress(thought)
                    await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)

                tool_call_dicts = [tc.to_openai_tool_call() for tc in response.tool_calls]
                messages = self.context.add_assistant_message(
                    messages,
                    response.content,
                    tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

                invocations: list[ToolInvocation] = []
                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    invocations.append(
                        ToolInvocation(
                            name=tool_call.name,
                            arguments=tool_call.arguments,
                            context=ToolExecutionContext(
                                call_id=tool_call.id,
                                session_key=session_key,
                                iteration=iteration,
                                origin=origin.value if origin is not None else None,
                            ),
                        )
                    )

                async def _tool_started(invocation: ToolInvocation) -> None:
                    if on_tool_event is not None:
                        nested = invocation.context.parent_call_id is not None
                        await on_tool_event(
                            "start",
                            {
                                "tool_call_id": (
                                    invocation.context.parent_call_id if nested else invocation.context.call_id
                                ),
                                "name": "tool_call" if nested else invocation.name,
                                "arguments": (
                                    {"name": invocation.name, "arguments": invocation.arguments}
                                    if nested
                                    else invocation.arguments
                                ),
                                "target_call_id": invocation.context.call_id if nested else None,
                                "target_name": invocation.name if nested else None,
                                "target_arguments": invocation.arguments if nested else None,
                            },
                        )

                async def _tool_completed(execution: ToolExecution) -> None:
                    invocation = execution.invocation
                    result = execution.result
                    result_str = str(result)
                    preview = result_str.replace("\n", " ")[:200]
                    logger.info(
                        "Tool result: {} duration={}ms result={}",
                        invocation.name,
                        int(execution.duration_ms),
                        preview,
                    )
                    if on_tool_event is not None:
                        nested = invocation.context.parent_call_id is not None
                        await on_tool_event(
                            "complete",
                            {
                                "tool_call_id": (
                                    invocation.context.parent_call_id if nested else invocation.context.call_id
                                ),
                                "name": "tool_call" if nested else invocation.name,
                                "result_preview": preview,
                                "truncated": len(result_str) > 200,
                                "failed": _is_tool_failure(result),
                                "target_call_id": invocation.context.call_id if nested else None,
                                "target_name": invocation.name if nested else None,
                                "target_arguments": invocation.arguments if nested else None,
                                "duration_ms": execution.duration_ms,
                            },
                        )

                executions = await self.tools.execute_many(
                    invocations,
                    on_start=_tool_started,
                    on_complete=_tool_completed,
                )

                for tool_call, execution in zip(response.tool_calls, executions, strict=True):
                    result = execution.result
                    messages = self.context.add_tool_result(messages, tool_call.id, tool_call.name, result)
                    # 跟踪同一工具的连续确定性失败；排除可通过重试清除的短暂错误。
                    if _is_hard_tool_failure(result):
                        if tool_call.name == loop_fail_tool:
                            loop_fail_streak += 1
                        else:
                            loop_fail_tool, loop_fail_streak = tool_call.name, 1
                    else:
                        loop_fail_tool, loop_fail_streak = None, 0

                # 失败循环打断：同一工具连续确定性失败 `threshold` 次后，
                # 向最后一个工具结果附加改变方法的提示，让模型停止重复无效调用。
                if (
                    loop_fail_streak >= self._LOOP_BREAK_THRESHOLD
                    and loop_nudges < self._LOOP_BREAK_MAX
                    and messages
                    and messages[-1].get("role") == "tool"
                ):
                    loop_nudges += 1
                    messages[-1]["content"] = (
                        str(messages[-1].get("content", ""))
                        + "\n\n"
                        + _loop_break_nudge(loop_fail_tool, loop_fail_streak)
                    )
                    loop_fail_streak = 0  # 每轮新的连续失败只触发一次
                prev_had_tool_calls = True
            else:
                clean = self._strip_think(response.content)
                # 不将错误响应持久化到会话历史，因为它们可能污染上下文，导致永久 400 循环。
                if response.finish_reason == "error":
                    logger.error("LLM returned error: {}", (clean or "")[:200])
                    final_content = clean or "Sorry, I encountered an error calling the AI model."
                    status = "error"
                    classification = response.error_classification
                    if classification is None:
                        classifier = getattr(self.provider, "classify_error", None)
                        classification = classifier(content=clean) if classifier is not None else None
                    error_category = classification.category if classification is not None else "unknown"
                    break

                # 空响应恢复：如果不处理，空的 Assistant Turn 会在此退出，暴露“无回复可给”的无效结果。
                # 放弃前先尝试恢复。合成脚手架用 ``_recovery_synthetic`` 标记，并在持久化和提取前移除，
                # 避免污染未来上下文。
                action = classify_empty_response(
                    response,
                    clean,
                    prev_had_tool_calls=prev_had_tool_calls,
                    nudges_done=post_tool_nudges,
                    prefill_retries=prefill_retries,
                    empty_retries=empty_retries,
                    limits=self._recovery_limits,
                )
                if action is RecoveryAction.PREFILL:
                    prefill_retries += 1
                    logger.warning(
                        "empty-recovery: thinking-only prefill {}/{}",
                        prefill_retries,
                        self._recovery_limits.thinking_prefill_max_retries,
                    )
                    # 将模型自身未删减的推理回填，使其继续生成正文。消息标记为合成，
                    # 在持久化和提取前丢弃；提供商的键白名单会从线上请求中移除推理字段。
                    messages = self.context.add_assistant_message(
                        messages,
                        response.content,
                        reasoning_content=response.reasoning_content,
                        thinking_blocks=response.thinking_blocks,
                    )
                    messages[-1]["_recovery_synthetic"] = True
                    prev_had_tool_calls = False
                    continue
                if action is RecoveryAction.NUDGE:
                    post_tool_nudges += 1
                    logger.warning("empty-recovery: post-tool empty nudge")
                    # 空 Assistant 消息必须位于工具结果和提示之间，因为多数 API 会对裸的
                    # tool → user 序列返回 400。
                    messages = self.context.add_assistant_message(messages, "(empty)")
                    messages[-1]["_recovery_synthetic"] = True
                    messages.append({"role": "user", "content": POST_TOOL_NUDGE, "_recovery_synthetic": True})
                    prev_had_tool_calls = False
                    continue
                if action is RecoveryAction.RETRY:
                    empty_retries += 1
                    logger.warning(
                        "empty-recovery: plain empty retry {}/{}",
                        empty_retries,
                        self._recovery_limits.empty_content_max_retries,
                    )
                    prev_had_tool_calls = False
                    continue

                messages = self.context.add_assistant_message(
                    messages,
                    clean,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                final_content = clean
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached; synthesizing final answer", self.max_iterations)
            # 耗尽包含两个彼此独立的事实，并非二选一：
            #   1. 本轮尚未完成——将其标记为 ``interrupted``，让影子 Git 检查点提交带上
            #      对应标签，并让下一轮的恢复提示展示提交 SHA 和已编辑文件以便续作。
            #   2. 用户现在仍应得到有用的回复——因此无论是否存在检查点，都尽力生成收尾
            #      （调用一次禁用工具的模型，总结已完成和待完成事项），而不是返回固定道歉。
            #      如果生成失败，辅助方法内部会回退到静态消息，确保本轮不会静默结束。
            status = "interrupted"
            final_content = await self._synthesize_final_on_exhaustion(
                messages,
                effective_model,
                fallback_models,
                session_key,
                on_token_delta=on_token_delta,
                on_reasoning_delta=on_reasoning_delta,
            )
            # 像普通最终回复一样，把收尾内容持久化到历史中。下游持久化只读取返回的
            # ``messages`` 列表；若不这样做，生成的答案虽会经由流到达用户，却不会进入
            # 对话，下一轮（尤其是中断恢复轮）便看不到本次总结。生成提示本身只保留在
            # 辅助方法内部，因此这里只落下回复。
            if final_content:
                messages = self.context.add_assistant_message(messages, final_content)

        # 持久化和返回前移除临时的空响应恢复脚手架。CodeFlow 会用
        # ``_recovery_synthetic`` 标记合成的推动/预填消息，必须移除以免持久化。
        if any(m.get("_recovery_synthetic") for m in messages):
            messages = [m for m in messages if not m.get("_recovery_synthetic")]

        outcome = TurnOutcome(
            status=status,
            error_category=error_category,
        )
        if self._checkpoint is not None:
            # 每轮快照：一次提交覆盖本轮全部编辑，正常退出和中断退出均如此
            # （与 Claude Code/Cursor 的粒度一致）。这里只尽力而为，commit_turn 从不抛错。
            label = f"turn {session_key or 'anon'} [{status}]"
            cid, changed = await self._checkpoint.commit_turn(label)
            outcome.checkpoint_id = cid
            if status == "interrupted":
                outcome.edited_files = changed

        return final_content, tools_used, messages, outcome

    async def run(self) -> None:
        """拉起 Agent runtime 依赖并作为长期任务保持存活。

        Turn 已经统一从 Spine 的 ``run_turn`` 进入，本协程不再消费 inbound bus。它依次启动
        executor、可选 debug server 和 MCP 连接，然后在 ``self._running`` 为真时短暂 sleep，
        让 Gateway 能把它作为长期任务 gather。shutdown 通过 ``stop()`` 清除运行标志；真正
        的 MCP、Sandbox 和后台任务清理由 `close` 系列方法负责。

        若启动依赖时抛出异常，异常向 Host 传播而不是进入空保活循环。正常退出循环后也不会
        自动重启资源，因此生命周期所有者必须显式决定何时再次调用 `run`。
        """
        self._running = True
        try:
            await self._start_executor()
            await self._start_debug_server()
            await self._connect_mcp()
        except SandboxInitError as exc:
            logger.error("Sandbox failed to start: {}", exc)
            await self.close_executor()
            self._running = False
            return
        except Exception:
            await self.close_executor()
            raise
        logger.info("Agent loop started")

        while self._running:
            await asyncio.sleep(1.0)

    @property
    def is_processing(self) -> bool:
        """返回当前是否有 Turn 正在全局处理锁内分派。

        结果来自 `_processing_lock.locked()`，表示这个 AgentLoop 实例的临界区占用状态，供
        Host 判断 BusyPolicy 等行为。它不是队列长度，也不证明模型或 Tool 此刻正在运行；
        锁从 Turn 进入到分派结束覆盖整条请求链，因此 ``True`` 只说明已有工作拥有该边界。
        """
        return self._processing_lock.locked()

    def _notify_turn_complete(self) -> None:
        for callback in self.on_turn_complete:
            try:
                callback()
            except Exception:
                logger.exception("on_turn_complete callback failed")

    async def close_mcp(self) -> None:
        """关闭 MCP 连接栈，并无条件继续关闭 Sandbox executor。

        已建立的 `_mcp_stack` 会先执行 `aclose()`；关闭期异常只记录为调试噪声，不阻断整体
        shutdown。随后 `_mcp_connected` 与 `_mcp_connecting` 都复位，使实例若被重新使用仍可
        连接。即使从未配置 MCP，也始终调用 `close_executor`，因为 Sandbox 生命周期不应
        被 MCP 是否存在所绑架。
        """
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK 的取消作用域清理会产生噪声，但无害
            self._mcp_stack = None
        self._mcp_connected = False  # 重置后 _connect_mcp() 才能在关闭后重连
        self._mcp_connecting = False  # 重置以免并发调用方永久阻塞
        await self.close_executor()  # 即使未配置 MCP 服务器也始终执行

    async def close(self) -> None:
        """先排空后台个性化任务，再关闭 Agent runtime 资源。

        `_close_lock` 让并发关闭串行化，`_closed` 使重复调用成为 no-op。首次关闭先执行
        `begin_close`，同步封住新个性化任务并取消旧任务；随后用 `gather(...,
        return_exceptions=True)` 等待其完成，最后调用 `close_mcp` 释放 MCP 与 Sandbox。
        只有这些步骤结束后才把实例标记为已关闭，避免 cleanup 尚未完成就对外宣称终止。
        """
        async with self._close_lock:
            if self._closed:
                return
            self.begin_close()
            tasks = tuple(self._personalization_tasks)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await self.close_mcp()
            self._closed = True

    def stop(self) -> None:
        """请求长期 `run` 保活循环在下一次检查时停止。

        方法只把 `_running` 设为 ``False`` 并记录日志，不等待循环退出，也不释放 MCP、
        Sandbox 或后台任务。因此它适合 Host 的同步停止信号；需要完整资源清理时调用方仍应
        await `close()`。重复调用不会改变额外状态。
        """
        self._running = False
        logger.info("Agent loop stopping")

    def replace_provider(self, provider: LLMProvider, *, model: str | None = None) -> None:
        from codeflow.call_efficiency.provider import CallEfficiencyProvider

        if isinstance(self.provider, CallEfficiencyProvider):
            self.provider.replace(provider)
        else:
            self.provider = provider
        if model is None:
            return
        self.model = model
        self.subagents.model = model
        self.memory_consolidator.model = model
        replace_model = getattr(self.context_engine, "replace_model", None)
        if callable(replace_model):
            replace_model(model)

    @trace.instrument("session.turn", seed=semconv.turn_seed, on_open=semconv.turn_open, extract=semconv.turn)
    async def _process_message(
        self,
        req: TurnRequest,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_token_delta: Callable[[str], Awaitable[None]] | None = None,
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
        on_tool_event: Callable[[str, dict], Awaitable[None]] | None = None,
        usage_sink: dict[str, Any] | None = None,
        origin: Origin | None = None,
        drain: Drain | None = None,
        context_metadata_sink: dict[str, Any] | None = None,
    ) -> tuple[str | None, list[str]] | None:
        """处理一个 `TurnRequest`，完成会话、上下文、迭代和事后流水线并返回回复。

        方法从 ``req.source`` 取得 channel、sender、chat 与 extras，解析当前 Session，运行允许
        的入站 Hook 和个性化前置步骤，再调用 Context Engine 组装消息并进入
        `_run_agent_loop`。循环结束后，它保存本轮消息、执行 Engine/Memory/Backend 的事后
        工作，并根据请求 ``origin`` 维持用户输入与 CRON、SUBAGENT 等系统来源的边界。

        有待交付回复时返回 ``(reply_content, media_paths)``；返回 ``None`` 表示 silent turn，
        例如 message Tool 已直接发送，或 Hook 短路选择了 None。这里的 silent 只说明本方法
        不再返回第二份内容，不代表用户一定没有收到消息。``origin`` 是 Spine
        `TurnRequest` 的来源事实，不能从文本内容反推。Provider 无法完成 Turn 时异常继续
        传播给 `run_turn` 和 Lane，而不是转换成成功字符串。
        """
        from codeflow.agent.hook import AgentHookContext

        channel = req.source.channel
        sender_id = req.source.sender_id
        chat_id = req.source.chat_id
        content = req.text
        metadata = dict(req.source.extras)
        turn_media = list(req.media)
        msg_session_key = req.conversation or f"{channel}:{chat_id}"

        # AgentHook 的 ``before_user_inbound`` 链。
        #
        # 观察型钩子和短路型钩子共用同一条有序链。来源不是用户输入时跳过本阶段。
        skip_user_inbound = origin in _SKIP_USER_INBOUND_ORIGINS
        if len(self.hooks) > 0 and not skip_user_inbound:
            _hook_ctx = AgentHookContext(
                session_key=msg_session_key,
                turn_request=req,
            )
            _decision = await self.hooks.before_user_inbound(_hook_ctx)
            if _decision.short_circuit_result is not None:
                return _decision.short_circuit_result

        preview = content[:80] + "..." if len(content) > 80 else content
        logger.info("Processing message from {}:{}: {}", channel, sender_id, preview)

        key = session_key or msg_session_key
        session = self.sessions.get_or_create(key)

        # 斜杠命令
        cmd = content.strip().lower()
        if cmd == "/new":
            try:
                if not await self.memory_consolidator.archive_unconsolidated(session):
                    return (
                        "Memory archival failed, session not cleared. Please try again.",
                        [],
                    )
            except Exception:
                logger.exception("/new archival failed for {}", session.key)
                return (
                    "Memory archival failed, session not cleared. Please try again.",
                    [],
                )

            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            return ("New session started.", [])
        if cmd == "/help":
            lines = [
                "✦ CodeFlow commands:",
                "/new — Start a new conversation",
                "/stop — Stop the current task",
                "/restart — Restart the bot",
                "/help — Show available commands",
            ]
            return ("\n".join(lines), [])
        if not self.context_engine.owns_compaction:
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)

        # ── 个性化流程（全局开关：self.enable_personalization）────────────────
        # 子智能体结果回注时跳过：其内容是系统生成的通知而非用户输入；个性化处理会污染
        # 用户画像，或针对通知触发澄清。此处仅 SUBAGENT 跳过（不是更宽泛的用户输入集合）：
        # Cron 轮次目前仍会进入并保留该流程。
        if self.enable_personalization and origin is not Origin.SUBAGENT:
            from datetime import datetime as _dt

            from codeflow.agent.personalizer import Personalizer

            _personalizer = Personalizer(MemoryStore(self.state), self.provider, self.model)

            # ── 步骤 2 完成阶段：用户正在回答待处理的澄清问题 ──
            if session.pending_clarification:
                _pending = session.pending_clarification

                # 判断用户是在回答上一个问题，还是发起新请求。新请求通常包含动作动词，
                # 且与原请求无关；重新分类以作判断：若仍需澄清，就按新请求处理。
                _recent = session.get_history(max_messages=4)
                _recheck = await _personalizer.classify(content, history=_recent)
                _is_new_request = _recheck.get("needs_clarification", False)

                if _is_new_request:
                    # 用户发起了新请求；丢弃旧的待处理状态并重新分类。
                    session.pending_clarification = None
                    self.sessions.save(session)
                    logger.info("Personalization: new request detected, discarding old pending_clarification")

                    _question = await _personalizer.generate_question(
                        content,
                        _recheck.get("domain", ""),
                    )
                    if _question:
                        _ts = _dt.now().isoformat()
                        session.record({"role": "user", "content": content, "timestamp": _ts})
                        session.record({"role": "assistant", "content": _question, "timestamp": _ts})
                        session.pending_clarification = {
                            "original_message": content,
                            "question": _question,
                            "domain": _recheck.get("domain", ""),
                        }
                        self.sessions.save(session)
                        logger.info(
                            "Personalization: asked clarification for new request, session {}",
                            session.key,
                        )
                        return (_question, [])
                        # 问题生成失败时清除待处理状态，并按正常流程继续。
                    session.pending_clarification = None

                else:
                    # 用户正在回答上一个问题；提取偏好并恢复原任务。
                    session.pending_clarification = None

                    # 后台提取偏好并写入 MEMORY.md，不阻塞响应。
                    async def _extract():
                        await _personalizer.extract_and_store_preference(
                            original_message=_pending["original_message"],
                            question=_pending["question"],
                            answer=content,
                        )

                    self._start_personalization_task(_extract)
                    # 正常继续：LLM 通过对话历史理解任务。

            else:
                # ── 步骤 1：分类请求——判断是否需要澄清 ──
                _recent = session.get_history(max_messages=4)
                _classification = await _personalizer.classify(content, history=_recent)

                if _classification.get("needs_clarification"):
                    # ── 步骤 2：行动前交互——生成并返回澄清问题 ──
                    _question = await _personalizer.generate_question(
                        content,
                        _classification.get("domain", ""),
                    )

                    if _question:
                        # 将原请求和澄清问题写入历史，保持对话连贯。
                        _ts = _dt.now().isoformat()
                        session.record({"role": "user", "content": content, "timestamp": _ts})
                        session.record({"role": "assistant", "content": _question, "timestamp": _ts})

                        # 保存待处理状态，让下一条消息可以恢复流程。
                        session.pending_clarification = {
                            "original_message": content,
                            "question": _question,
                            "domain": _classification.get("domain", ""),
                        }
                        self.sessions.save(session)

                        logger.info("Personalization: asked clarification for session {}", session.key)
                        return (_question, [])
                        # generate_question 失败：静默跳过并继续
            # ── 个性化流程结束 ────────────────────────────────────────────────

        self._set_tool_context(channel, chat_id, metadata.get("message_id"), session_key=key)
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()
            # ask_user 使用真实 conversation_id（即通道/门控键）作为键；该标识感知话题
            # （req.conversation），而非仅使用 channel:chat_id。
        if (ask_tool := self.tools.get("ask_user")) and isinstance(ask_tool, AskUserTool):
            ask_tool.set_context(key)

        context_messages = self._context_messages_for_session(session)
        # SkillForge：选择器选出前 K 个。参见上方系统消息分支中的说明——返回空列表时
        # 回退到完整目录。B-3 阶段统一经由 ``_select_skills_for_turn`` 路由，使新的
        # ``default`` 引擎可在此直接结束选择。
        selected_skills = await self._select_skills_for_turn(
            content,
            context_messages,
        )
        context_metadata = context_metadata_sink if context_metadata_sink is not None else {}
        initial_messages = await self._assemble_context_messages(
            session=session,
            session_key=key,
            current_message=content,
            media=turn_media if turn_media else None,
            channel=channel,
            chat_id=chat_id,
            selected_skills=selected_skills or None,
            metadata_sink=context_metadata,
        )
        injected_skill_ids = list(
            context_metadata.get("injected_skill_ids") or self._collect_injected_skill_ids(selected_skills)
        )

        # ── 模型路由（EcoClaw 风格）──────────────────────────────────────────
        routed_model: str | None = None
        fallback_models: list[str] = []
        if self.router is not None:
            routed_model, fallback_models = await self.router.select_model_chain(content)
            if routed_model and routed_model != self.model:
                logger.info("Router: {} → {}", self.model, routed_model)
            if fallback_models:
                logger.info("Router fallback chain: {}", fallback_models)

        turn_start_idx = len(initial_messages) - 1
        final_content, _, all_msgs, outcome = await self._run_agent_loop(
            initial_messages,
            on_progress=on_progress,
            session_key=key,
            model=routed_model,
            fallback_models=fallback_models,
            injected_skill_ids=injected_skill_ids,
            on_token_delta=on_token_delta,
            on_reasoning_delta=on_reasoning_delta,
            on_tool_event=on_tool_event,
            usage_sink=usage_sink,
            drain=drain,
            origin=origin,
        )
        self._stash_recovery(key, outcome)
        if outcome.status == "error":
            raise ProviderTurnError(outcome.error_category or "unknown")

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

            # AgentHook 的 ``after_send`` 是通用出站阶段，适用于所有来源。
        if len(self.hooks) > 0:
            from codeflow.agent.hook import AgentHookContext

            _send_ctx = AgentHookContext(
                session_key=key,
                outbound_content=final_content,
            )
            _send_decision = await self.hooks.after_send(_send_ctx)
            if _send_decision.modified_content is not None:
                final_content = _send_decision.modified_content
                for message in reversed(all_msgs[turn_start_idx:]):
                    if message.get("role") == "assistant" and not message.get("tool_calls"):
                        message["content"] = final_content
                        break

        turn_artifact_messages = self._save_turn(
            session,
            all_msgs,
            turn_start_idx,
            origin,
        )
        self.sessions.save(session)
        await self.context_engine.after_turn(
            key,
            {
                "final_content": final_content,
                "messages": turn_artifact_messages,
                "context": context_metadata,
            },
        )
        await self._dispatch_backend_store(
            key,
            turn_artifact_messages,
        )
        if not self.context_engine.owns_compaction:
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)

            # ── 步骤 4：行动后学习（后台、非阻塞）──────────────────────────────
            # 子智能体结果回注时跳过（参见上方轮次前流程）：其内容是系统生成的通知，
            # 不是可供学习的用户输入。
        if self.enable_personalization and origin is not Origin.SUBAGENT:
            from codeflow.agent.personalizer import Personalizer

            _p4 = Personalizer(MemoryStore(self.state), self.provider, self.model)

            async def _post_learn():
                await _p4.post_learn(content, final_content)

            self._start_personalization_task(_post_learn)
            # ── 步骤 4 结束 ────────────────────────────────────────────────────

        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt.sent_in_turn:
            # 防御性指纹。过去智能体通过 message 工具回复并静默返回 None 时不会留下
            # 痕迹，导致随机出现的无效轮次无法通过 grep 发现。记录原本要返回的响应，
            # 为后续调查留下与下方 "Response to ..." 并行的线索。
            if final_content:
                preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
                logger.info(
                    "MessageTool sent in turn for {}:{}: {}",
                    channel,
                    sender_id,
                    preview,
                )
            return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", channel, sender_id, preview)
        return (final_content, [])

    def _save_turn(
        self,
        session: Session,
        messages: list[dict],
        skip: int,
        origin: Origin | None = None,
    ) -> list[dict]:
        """把本轮新增消息清理后追加到 Session，并返回实际持久化的切片。

        ``messages[skip:]`` 是相对组装窗口新增的部分。SUBAGENT 来源的首条系统回注、带
        `_recovery_synthetic` 的空响应恢复脚手架，以及没有正文和 Tool call 的 Assistant
        空消息都不会落盘。Tool 正文超过 `_TOOL_RESULT_MAX_CHARS` 时截断，避免单次外部输出
        无限膨胀历史；该截断只影响 Session 副本，不修改调用方的原消息。

        User 消息会剥离 `ContextBuilder._RUNTIME_CONTEXT_TAG` 前缀，多模态 data:image 只记录
        ``[image]`` 占位，随后统一经过 `sanitize_persisted_payload`。每条记录补入时间戳并
        调用 `session.record`，最后更新 `session.updated_at`。返回列表精确表示后续 Backend
        可以消费的持久化事实，而不是原始运行时消息全集。
        """
        persisted: list[dict] = []
        for index, m in enumerate(messages[skip:]):
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if origin is Origin.SUBAGENT and index == 0 and role == "user":
                continue
            if entry.get("_recovery_synthetic"):
                continue  # #1a 合成恢复推动消息——绝不持久化脚手架
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # 跳过空助手消息——它们会污染会话上下文
            if role == "tool" and isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                entry["content"] = content[: self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    # 去除运行时上下文前缀，只保留用户文本。
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = []
                    for c in content:
                        if (
                            c.get("type") == "text"
                            and isinstance(c.get("text"), str)
                            and c["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG)
                        ):
                            continue  # 从多模态消息中去除运行时上下文
                        if c.get("type") == "image_url" and c.get("image_url", {}).get("url", "").startswith(
                            "data:image/"
                        ):
                            filtered.append({"type": "text", "text": "[image]"})
                        else:
                            filtered.append(c)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry = sanitize_persisted_payload(entry)
            entry.setdefault("timestamp", self._now_fn().isoformat())
            session.record(entry)
            persisted.append(entry)
        session.updated_at = self._now_fn()
        return persisted

    async def run_turn(
        self,
        req: TurnRequest,
        emit: Emit,
        drain: Drain,
        *,
        stream: bool = True,
        usage_sink: dict[str, Any] | None = None,
        text_sink: dict[str, Any] | None = None,
    ) -> TurnOutcome:
        """作为 Spine 原生入口消费 `TurnRequest`，经单一 ``emit`` 扇出事件并返回 TurnOutcome。

        这个边界把旧实现的字符串返回值和五类 callback 收拢为一条可观察事件流。名称使用
        ``run_turn`` 而不是 ``run``，因为后者只负责 executor、debug server、MCP 拉起后的
        runtime keep-alive；Spine runner 包装本方法以满足 `TurnRunner` protocol，真实请求也
        从这里进入 `_process_message`。

        ``stream`` 是 canon Q2-D assembly switch。TUI 等 streaming outlet 传 ``True`` 时，
        回复以 StreamDelta 发送并在边界处 dissolves（b2 — no trailing Text）；REPL 等非流式
        出口传 ``False`` 时只发送一个 Text。它同时控制 LLM callbacks 与 message-tool routing，
        确保整份回复只走一种交付方式。媒体独立发为 MediaOut，Tool、Reasoning、Notice 和
        Usage 也都使用同一个 ``emit``。

        ``usage_sink`` 允许调用方观察包含 cost / context 的完整 Token 账目，比三字段
        TurnOutcome.usage 更丰富；TUI 用它填充 message.complete，REPL 可省略并读取返回值。
        ``text_sink`` 是回复文本的同类观察副本：最终内容写入 ``text_sink["text"]``，但仍只
        经 emit 交付一次，Cron Host 可在 Turn 后据此广播。``drain`` 则把
        BusyPolicy.INJECT 的中途消息传入每次迭代顶部。

        Sandbox init 等异常不会在这里转成错误字符串，而是继续传播，让 Lane 产生
        TurnFailed；这保留了 Spine 的失败事实。正常返回的 `TurnOutcome` 汇总 Usage、是否
        显式回复、Tool 次数与失败数、Memory 命中、Skill 注入和 Context 回退证据，不包含
        第二份用户回复。
        """
        from codeflow.proactive_engine.schedulers.cron.tool import CronTool
        from codeflow.spine.events import (
            MediaOut,
            Notice,
            NoticeKind,
            Reasoning,
            StreamDelta,
            Text,
            ToolEvent,
            ToolPhase,
            Usage,
        )
        from codeflow.spine.message import Media
        from codeflow.spine.runner import TurnOutcome

        cid = req.conversation or f"{req.source.channel}:{req.source.chat_id}"

        streamed = False
        tool_calls = 0
        tool_failures = 0
        context_metadata: dict[str, Any] = {}

        async def on_token(text: str) -> None:
            nonlocal streamed
            if not text:
                return
            streamed = True
            await emit(StreamDelta(delta=text))

        async def on_reasoning(text: str) -> None:
            if text:
                await emit(Reasoning(content=text))

        async def on_tool(phase: str, info: dict[str, Any]) -> None:
            nonlocal tool_calls, tool_failures
            if phase == "start":
                tool_calls += 1
                if info["name"] != "message":
                    await emit(
                        ToolEvent(
                            phase=ToolPhase.START,
                            tool_call_id=info["tool_call_id"],
                            name=info["name"],
                            arguments=info["arguments"],
                            target_call_id=info.get("target_call_id"),
                            target_name=info.get("target_name"),
                            target_arguments=info.get("target_arguments"),
                        )
                    )
            else:
                if info["failed"]:
                    tool_failures += 1
                if info["name"] != "message" or info["failed"]:
                    await emit(
                        ToolEvent(
                            phase=ToolPhase.COMPLETE,
                            tool_call_id=info["tool_call_id"],
                            name=info["name"],
                            result_preview=info["result_preview"],
                            truncated=info["truncated"],
                            failed=info["failed"],
                            target_call_id=info.get("target_call_id"),
                            target_name=info.get("target_name"),
                            target_arguments=info.get("target_arguments"),
                            duration_ms=info.get("duration_ms"),
                        )
                    )

        async def on_progress(text: str, tool_hint: bool = False) -> None:
            # 保留进度与工具提示的区别，让出口像总线路径一样，分别通过配置开关
            # （send_progress 与 send_tool_hints）控制：工具提示文本使用
            # NoticeKind.TOOL_HINT，进度使用 PROGRESS。不渲染两者的出口仍会吞掉这两类通知。
            if text:
                await emit(
                    Notice(
                        kind=NoticeKind.TOOL_HINT if tool_hint else NoticeKind.PROGRESS,
                        detail=text,
                    )
                )

        async def _emit_media(paths: list[str]) -> None:
            await emit(
                MediaOut(media=tuple(Media(path=p, mime="application/octet-stream", kind="file") for p in paths))
            )

        # 将 message 工具的回复路由到令牌流，使工具驱动的回复像主响应一样流式输出；
        # 此时 _process_message 返回 None，因此下方边界不会再次发出内容。回调属于当前轮次
        # （MessageTool 中的 ContextVar），并发轮次无法覆盖本轮路由，无需保存和恢复。
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):

            async def _route_to_stream(content: str, media: list[str]) -> None:
                # message 工具的回复可以附带媒体；需独立发出以免丢失（工具回复会让
                # _process_message 返回 None，下方边界看不到它）。内容与主回复遵循同一流式开关：
                # 流式时使用 StreamDelta，否则使用单个 Text；不然非流式出口会吞掉增量。
                if media:
                    await _emit_media(media)
                if text_sink is not None and content:
                    text_sink["text"] = content
                if stream:
                    await on_token(content)
                elif content:
                    await emit(Text(content=content))

            message_tool.set_send_callback(_route_to_stream)

        # CRON 轮次不得让智能体在运行途中调度新的 cron 任务。CronTool 通过 ContextVar
        # 防护；需在此处、即实际运行本轮的通道任务中设置，才能传播到工具。cron 回调在
        # 另一个任务中设置该值，无法传到本任务。
        cron_tool = self.tools.get("cron")
        cron_token = None
        if req.origin is Origin.CRON and isinstance(cron_tool, CronTool):
            cron_token = cron_tool.set_cron_context(True)

        if usage_sink is None:
            usage_sink = {}
        try:
            await self._start_executor()
            await self._connect_mcp()
            out = await self._process_message(
                req,
                session_key=cid,
                on_progress=on_progress,
                on_token_delta=on_token if stream else None,
                on_reasoning_delta=on_reasoning if stream else None,
                on_tool_event=on_tool,
                usage_sink=usage_sink,
                origin=req.origin,
                drain=drain,
                context_metadata_sink=context_metadata,
            )
        except Exception:
            await self.close_executor()
            raise
        finally:
            if cron_token is not None and isinstance(cron_tool, CronTool):
                cron_tool.reset_cron_context(cron_token)

        # 单一的返回到发出边界（N-UNIFORM）。MediaOut 独立于流，并先于 Text
        # （G-MEDIA-2(a)：当前顺序为媒体优先）。
        if out is not None:
            reply_content, reply_media = out
            if reply_media:
                await _emit_media(reply_media)
            if not streamed and reply_content:
                await emit(Text(content=reply_content))
            if text_sink is not None and reply_content:
                text_sink["text"] = reply_content

        usage = Usage(
            prompt_tokens=int(usage_sink.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage_sink.get("completion_tokens", 0) or 0),
            total_tokens=int(usage_sink.get("total_tokens", 0) or 0),
        )
        # message 工具回复虽会让 _process_message 返回 None，但确实已回复，
        # 因此也计作显式回复。
        replied_via_tool = isinstance(message_tool, MessageTool) and message_tool.sent_in_turn
        return TurnOutcome(
            usage=usage,
            explicit_reply=out is not None or replied_via_tool,
            tool_calls=tool_calls,
            tool_failures=tool_failures,
            memory_hits=int(context_metadata.get("memory_hits", 0) or 0),
            injected_skill_ids=tuple(context_metadata.get("injected_skill_ids") or ()),
            context_path=context_metadata.get("path"),
            context_fallback_reason=context_metadata.get("fallback_reason"),
            skill_source_failures=tuple(context_metadata.get("skill_source_failures") or ()),
        )


def _merge_tool_call_fragments(
    slots: list[dict[str, Any]],
    delta: dict[str, Any],
) -> None:
    """把一次 chat_stream ``tool_call_delta`` 合并进按位置保存的累积槽。

    每个 slot 使用 ``{id, function: {name, arguments_buf: [str]}}`` 形状。按照
    OpenAI/LiteLLM Provider chunk 语义，片段携带 ``index``；对应 ``id`` 与
    ``function.name`` 通常只在该 index 的首片段出现，``function.arguments`` 则是分段到达的
    JSON string。方法只在槽中尚无 id/name 时写入它们，并按抵达顺序追加参数片段。

    ``index`` 决定扩展和选择哪个 slot，因此并行 multi-tool stream 不会全部塌缩进
    ``slots[0]``。缺少 ``index`` 时默认为 0，以兼容 single-tool case。delta 没有 Tool call
    时直接返回；本函数只累积，不解析最终 JSON，也不构造 `ToolCallRequest`。
    """
    incoming = delta.get("tool_calls") or []
    if not incoming:
        return
    for tc in incoming:
        idx = int(tc.get("index", 0) or 0)
        while len(slots) <= idx:
            slots.append({"id": None, "function": {"name": None, "arguments_buf": []}})
        slot = slots[idx]
        if tc.get("id") and not slot["id"]:
            slot["id"] = tc["id"]
        fn = tc.get("function") or {}
        if fn.get("name") and not slot["function"]["name"]:
            slot["function"]["name"] = fn["name"]
        if fn.get("arguments"):
            slot["function"]["arguments_buf"].append(fn["arguments"])


def _finalize_tool_calls(slots: list[dict[str, Any]]) -> list[ToolCallRequest]:
    """把流式累积槽转换为可交给执行器的 `ToolCallRequest` 列表。

    没有 ``function.name`` 的不完整槽会被跳过。其余槽按原顺序连接
    ``arguments_buf``，非空时尝试作为 JSON 解析，空参数得到空字典；JSON 不完整或非法时不
    丢弃调用，而以 ``{"_raw_arguments": args_text}`` 保存原始文本，交由后续参数校验给出
    可解释失败。缺失 id 使用空字符串。返回值只完成协议形状归一化，不执行 Tool。
    """
    result: list[ToolCallRequest] = []
    for slot in slots:
        name = slot["function"]["name"]
        if not name:
            continue
        args_text = "".join(slot["function"]["arguments_buf"])
        try:
            args = json.loads(args_text) if args_text else {}
        except json.JSONDecodeError:
            args = {"_raw_arguments": args_text}
        result.append(
            ToolCallRequest(
                id=slot["id"] or "",
                name=name,
                arguments=args,
            )
        )
    return result
