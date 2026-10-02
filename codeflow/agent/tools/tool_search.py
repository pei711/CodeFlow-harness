"""为大型 Tool Catalog 提供 Progressive disclosure 与稳定 Prompt cache。

当 built-ins + plugins + MCP 超过阈值，每轮注入全部 Schema 会让 Context 成本随 Tool 数量增长。
本模块只保留核心 Tool 与两个 Meta Tool：``tool_search`` 用 BM25 搜索隐藏 Catalog，并返回 name、
description、parameter schema；``tool_call`` 按名称转发 Registry，后者校验 arguments 并返回可
修正错误。模型无需第二次 describe lookup 即可调用命中能力。

每 Turn 发送的 Tool list 固定为 Core + Meta，Prompt cache 因而跨 Session 稳定；Tools 位于 cached
prefix 的 System+Messages 之前，动态列表会使后续全部失效。代价是 Cataloged Tool 通过
``tool_call`` 而不是 Native function-calling 执行。

:class:`ToolSearchStrategy` 维护两个 Visibility tier：always-visible 每轮带完整 Schema，Cataloged
其余能力仅在搜索时披露。小于阈值时 Meta Tool 被移除并保持旧行为；Meta 被 disabled 时回退
暴露全部 Tool，不能把能力困在不可调用搜索之后。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from codeflow.agent.tools.base import Tool
from codeflow.agent.tools.execution import ToolExecutionContext, ToolInvocation
from codeflow.agent.tools.tool_index import ToolIndex
from codeflow.token_wise.base import TokenStrategy

if TYPE_CHECKING:
    from codeflow.agent.tools.registry import ToolRegistry

# 核心工具在每个 Turn 都保持可见，否则 Agent 还得搜索它们，基本能力会受限。除文件、
# 搜索和执行原语外，``message``、``ask_user`` 和 ``spawn`` 是任何 Turn 都可能需要的
# 交互和编排原语：回复、通过提问解除阻塞、委托子 Agent。隐藏后模型可能根本想不到搜索它们。
# ``tools.tool_search.always_visible`` 配置可扩展此集合。
DEFAULT_ALWAYS_VISIBLE: tuple[str, ...] = (
    "read_file",
    "write_file",
    "edit_file",
    "list_dir",
    "grep",
    "find",
    "exec",
    "message",
    "ask_user",
    "spawn",
)

TOOL_CALL_NAME: str = "tool_call"
# 元工具在功能开启时始终注册，但绝不进入目录。
META_TOOL_NAMES: frozenset[str] = frozenset({"tool_search", TOOL_CALL_NAME})


class ToolSearchController:
    """共享 Meta Tool 与 Strategy 所需的 Live Catalog、Index 和 Visibility State。

    Registry 是 Tool Catalog Source of Truth，ToolIndex 只保存可搜索 BM25 投影。``always_visible`` 在
    构造时强制加入 META_TOOL_NAMES，所以大型 Catalog 的 per-turn Tool list 始终是 Core + Meta，
    Prompt cache 得以稳定。Controller 负责搜索结果组装与 Nested invocation 解析，不直接参与
    LLM Strategy 生命周期。
    """

    def __init__(
        self,
        registry: "ToolRegistry",
        *,
        always_visible: set[str],
        search_result_limit: int = 10,
    ) -> None:
        self._registry = registry
        self.always_visible = set(always_visible) | META_TOOL_NAMES
        self.search_result_limit = search_result_limit
        self._index = ToolIndex()

    def _catalog_tools(self) -> list[Tool]:
        """返回所有已注册但非 Meta 的 Tool，防止搜索与调用工具自我编目。

        方法按 Registry 顺序读取 Live instance，热删除后 get 为 None 的名称跳过。``tool_search``
        与 ``tool_call`` 永远不进入自身 BM25 结果，否则模型可能形成递归 Meta 调用。
        """
        out = []
        for name in self._registry.tool_names:
            if name in META_TOOL_NAMES:
                continue
            tool = self._registry.get(name)
            if tool is not None:
                out.append(tool)
        return out

    def refresh(self) -> None:
        """用当前 Registry Catalog 同步 BM25 Index，Signature 未变时 no-op。

        每次大型 Catalog LLM call 前调用，确保 Plugin/MCP 热加入可搜索；昂贵 Rebuild 是否需要由
        ToolIndex.ensure 的 name/description/parameters Signature 决定。
        """
        self._index.ensure(self._catalog_tools())

    def visible_names(self) -> set[str]:
        return self.always_visible

    def search(self, query: str, limit: int | None = None) -> list[dict[str, Any]]:
        """搜索 Catalog，并为每个 Hit 返回 name、description 与 parameter schema。

        Limit 缺失时使用 Controller 默认值。BM25 先只返回 Name，随后重新从 Live Registry 获取
        Tool，已热删除项跳过；完整 Schema 让模型可直接进入 ``tool_call``，无需 separate describe
        round-trip。结果顺序沿用 BM25 rank，方法不把 Tool 加入 always-visible set。
        """
        names = self._index.search(query, limit or self.search_result_limit)
        hits = []
        for name in names:
            tool = self._registry.get(name)
            if tool is None:
                continue
            hits.append(
                {
                    "name": name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            )
        return hits

    def resolve_invocation(
        self,
        name: str,
        arguments: dict[str, Any] | str | None,
        context: ToolExecutionContext,
    ) -> ToolInvocation | None:
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        if not isinstance(arguments, (dict, type(None))):
            return None
        if name in META_TOOL_NAMES or not self._registry.has(name):
            return None
        child_call_id = f"{context.call_id}:{name}" if context.call_id else None
        return ToolInvocation(
            name=name,
            arguments=arguments or {},
            context=context.child(child_call_id),
        )

    async def call(
        self,
        name: str,
        arguments: dict[str, Any] | None,
        context: ToolExecutionContext | None = None,
    ) -> str:
        """调用一个 Cataloged Tool，并把 Argument validation 与执行转交 Registry。

        模型有时把 Nested ``arguments`` 发成 JSON string，方法先解析这种形状；非法 JSON、非
        Object、Meta Tool 自调用或未知 Name 均返回可修正 Error。合法输入经 `resolve_invocation`
        生成 Child ToolExecutionContext，再由 Registry execute_invocation，保留 timeout、Schema、
        failure 与 tracing 语义。返回目标 Tool Result，不伪装 Meta Tool 自己完成业务。
        """
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return "Error: 'arguments' must be a JSON object."
        if name in META_TOOL_NAMES:
            return f"Error: '{name}' cannot be invoked via tool_call."
        if not self._registry.has(name):
            return f"Error: tool '{name}' not found. Use tool_search to find it."
        invocation = self.resolve_invocation(name, arguments, context or ToolExecutionContext())
        if invocation is None:
            return "Error: 'arguments' must be a JSON object."
        execution = await self._registry.execute_invocation(invocation)
        return execution.result


class ToolSearchTool(Tool):
    """对当前未加载 Schema 的 Tool Catalog 执行 Keyword Search。

    Query 应描述所需 Task capability，可中英文；结果是包含 description 与 Schema 的 JSON，直接
    可交给 `tool_call`。无命中返回提示扩大关键词的普通文本。Tool 只搜索本地 Index，不执行
    命中能力，也不改变 Visible Set。
    """

    def __init__(self, controller: ToolSearchController) -> None:
        self._ctrl = controller

    @property
    def name(self) -> str:
        return "tool_search"

    @property
    def description(self) -> str:
        return (
            "Search the catalog of additional tools that are available but not "
            "currently loaded. Returns matching tools with their description and "
            "parameter schema, ready to invoke with tool_call. Query with task "
            "keywords, e.g. 'create github issue' or '生成图片'."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Task keywords describing the capability you need.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of results.",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str, limit: int | None = None) -> str:
        hits = self._ctrl.search(query, limit)
        if not hits:
            return f"No tools matched '{query}'. Try broader or different keywords."
        return json.dumps(hits, ensure_ascii=False)


class ToolCallTool(Tool):
    """按精确名称调用 Cataloged Tool，Arguments 由 Registry 校验。

    目标 Name 必须来自 ``tool_search`` 结果，Nested arguments 为对应 Schema Object。普通 execute
    使用空 Context；`execute_with_context` 保留 Meta call id，并由 Controller 生成 Child call id，
    让 ToolEvent 显示实际目标。`resolve_invocation` 还使 execute_many 在并发判定前看到目标 Tool
    capability，而不是错误把 Meta Tool 当作安全 Read。
    """

    def __init__(self, controller: ToolSearchController) -> None:
        self._ctrl = controller

    @property
    def name(self) -> str:
        return TOOL_CALL_NAME

    @property
    def description(self) -> str:
        return (
            "Invoke a tool found via tool_search by name, passing its arguments. "
            "If the arguments don't fit the tool's schema the registry returns a "
            "validation error describing the fix; adjust and call again."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Exact tool name from a tool_search result.",
                },
                "arguments": {
                    "type": "object",
                    "description": "Arguments object for the target tool.",
                },
            },
            "required": ["name"],
        }

    async def execute(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        return await self._ctrl.call(name, arguments)

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> str:
        return await self._ctrl.call(name, arguments, context)

    def resolve_invocation(self, invocation: ToolInvocation) -> ToolInvocation:
        name = invocation.arguments.get("name")
        if not isinstance(name, str):
            return invocation
        resolved = self._ctrl.resolve_invocation(
            name,
            invocation.arguments.get("arguments"),
            invocation.context,
        )
        return resolved or invocation


class ToolSearchStrategy(TokenStrategy):
    """在 ``before_llm_call`` 为大型 Catalog 压缩 Tool list 的 Token Strategy。

    Catalog 数量不超过 ``compaction_threshold`` 时移除 Meta Tool，其余 definitions 原样通过，
    让小型 Setup 保持 byte-for-byte 旧行为。超过阈值时先 refresh Index，再只保留 always-visible
    Core + Meta Schema，其余仍可经 ``tool_search`` / ``tool_call`` 到达。

    若两 Meta definitions 未同时出现，例如被 disabled_tools 移除，Strategy 直接暴露全部列表，
    不实施不可逆隐藏。Messages 与 Model 始终原样返回，策略只改变 Tool definitions。
    """

    def __init__(self, controller: ToolSearchController, *, compaction_threshold: int = 50) -> None:
        self._ctrl = controller
        self._compaction_threshold = compaction_threshold

    @property
    def name(self) -> str:
        return "tool_search"

    async def before_llm_call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, str]:
        if not tools:
            return messages, tools, model
        self._ctrl.refresh()
        catalog_size = sum(1 for t in tools if t["function"]["name"] not in META_TOOL_NAMES)
        if catalog_size <= self._compaction_threshold:
            out = [t for t in tools if t["function"]["name"] not in META_TOOL_NAMES]
            return messages, out, model
        present = {t["function"]["name"] for t in tools}
        if not META_TOOL_NAMES.issubset(present):
            # 元工具不可用时，例如被 disabled_tools 移除，直接暴露全部工具；
            # 不要将已编目的工具困在模型无法调用的搜索之后。
            return messages, tools, model
        visible = self._ctrl.visible_names()
        out = [t for t in tools if t["function"]["name"] in visible]
        return messages, out, model
