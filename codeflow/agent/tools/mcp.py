"""连接 MCP Server，并把远端 Tool 适配成原生 CodeFlow Tool。

`connect_mcp_servers` 支持 stdio、SSE 与 streamableHttp transport，把连接生命周期压入调用方
AsyncExitStack；每个远端定义由 `MCPToolWrapper` 加上 ``mcp_<server>_`` namespace 后注册，
避免不同 Server 名称碰撞。Sandbox 不支持 process spawning 时 stdio 配置硬失败，不能静默
启动一个不可用 Agent；单 Server 普通连接失败则记录并继续其他 Server。
"""

import asyncio
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

import httpx
from loguru import logger

from codeflow.agent.tools.base import Tool, ToolResult
from codeflow.agent.tools.registry import ToolRegistry
from codeflow.sandbox import SandboxInitError

if TYPE_CHECKING:
    from codeflow.sandbox import SandboxExecutor


class MCPToolWrapper(Tool):
    """把一个 MCP Server Tool 包装为符合 CodeFlow Registry 契约的 Tool。

     Wrapper 保留 Server 原始 name 用于 `call_tool`，对模型暴露带 Server prefix 的名称、远端
     description 与 inputSchema。每次调用有独立 ``tool_timeout``；timeout、SDK/Server cancel 与
    普通异常都返回 failed ToolResult，只有当前 asyncio Task 确实在 cancelling 时才重新传播
     CancelledError。响应的 TextContent 提取正文，其他 block 转字符串，``result.isError`` 决定
    显式失败状态。
    """

    def __init__(self, session, server_name: str, tool_def, tool_timeout: int = 30):
        self._session = session
        self._original_name = tool_def.name
        self._name = f"mcp_{server_name}_{tool_def.name}"
        self._description = tool_def.description or tool_def.name
        self._parameters = tool_def.inputSchema or {"type": "object", "properties": {}}
        self._tool_timeout = tool_timeout

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, **kwargs: Any) -> str:
        from mcp import types

        try:
            result = await asyncio.wait_for(
                self._session.call_tool(self._original_name, arguments=kwargs),
                timeout=self._tool_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("MCP tool '{}' timed out after {}s", self._name, self._tool_timeout)
            return ToolResult(
                f"(MCP tool call timed out after {self._tool_timeout}s)",
                failed=True,
            )
        except asyncio.CancelledError:
            # MCP SDK 的 anyio 取消作用域可能在超时或失败时泄漏 CancelledError。
            # 只有当前任务确实被外部取消（如 /stop）时才重新抛出。
            task = asyncio.current_task()
            if task is not None and task.cancelling() > 0:
                raise
            logger.warning("MCP tool '{}' was cancelled by server/SDK", self._name)
            return ToolResult("(MCP tool call was cancelled)", failed=True)
        except Exception as exc:
            logger.exception(
                "MCP tool '{}' failed: {}: {}",
                self._name,
                type(exc).__name__,
                exc,
            )
            return ToolResult(
                f"(MCP tool call failed: {type(exc).__name__})",
                failed=True,
            )

        parts = []
        for block in result.content:
            if isinstance(block, types.TextContent):
                parts.append(block.text)
            else:
                parts.append(str(block))
        return ToolResult(
            "\n".join(parts) or "(no output)",
            failed=bool(result.isError),
        )


async def connect_mcp_servers(
    mcp_servers: dict,
    registry: ToolRegistry,
    stack: AsyncExitStack,
    executor: "SandboxExecutor | None" = None,
) -> None:
    """连接所有已配置 MCP Servers，并把列出的 Tool 注册到 ``registry``。

    Transport 可显式声明，也可从 command/url 推断为 stdio、sse 或 streamableHttp。stdio 在支持
    process spawning 的 Sandbox 内由 Executor 启动，否则使用 Host MCP client；HTTP transport
    合并配置 headers 并跟随 redirect。Session initialize 后调用 list_tools，为每项建立
    `MCPToolWrapper`。

    ``stack`` 统一拥有 Client、HTTP 和 transport 生命周期。Sandbox capability mismatch 抛
    `SandboxInitError` 给 Agent 启动边界；未知 transport 或单 Server 的 Exception/
    BaseExceptionGroup 记录后跳过，已连接 Server 保持可用。
    """
    for name, cfg in mcp_servers.items():
        # 在 try/except 之前解析传输类型，使下方的沙箱保护能抛错，
        # 而不被单服务器错误处理器吞掉。
        transport_type = cfg.type
        if not transport_type:
            if cfg.command:
                transport_type = "stdio"
            elif cfg.url:
                transport_type = "sse" if cfg.url.rstrip("/").endswith("/sse") else "streamableHttp"
            else:
                logger.warning("MCP server '{}': no command or url configured, skipping", name)
                continue

        # 沙箱保护采用硬失败，避免 Agent 带着静默损坏的 MCP 服务器启动。该逻辑在
        # try/except 外运行，SandboxInitError 会传播到 _connect_mcp()，再作为启动错误暴露。
        if (
            transport_type == "stdio"
            and executor is not None
            and executor.is_sandboxed
            and not executor.supports_process_spawning
        ):
            raise SandboxInitError(
                f"MCP server '{name}' uses stdio transport, but the active sandbox "
                f"({type(executor).__name__}) does not yet support process spawning. "
                "Either switch to an HTTP/SSE MCP server or set sandbox.backend='none'."
            )

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.sse import sse_client
            from mcp.client.stdio import stdio_client
            from mcp.client.streamable_http import streamable_http_client

            if transport_type == "stdio":
                if executor is not None and executor.supports_process_spawning:
                    read, write = await executor.start_process(cfg.command, cfg.args, env=cfg.env or None)
                else:
                    params = StdioServerParameters(command=cfg.command, args=cfg.args, env=cfg.env or None)
                    read, write = await stack.enter_async_context(stdio_client(params))
            elif transport_type == "sse":

                def httpx_client_factory(
                    headers: dict[str, str] | None = None,
                    timeout: httpx.Timeout | None = None,
                    auth: httpx.Auth | None = None,
                ) -> httpx.AsyncClient:
                    merged_headers = {**(cfg.headers or {}), **(headers or {})}
                    return httpx.AsyncClient(
                        headers=merged_headers or None,
                        follow_redirects=True,
                        timeout=timeout,
                        auth=auth,
                    )

                read, write = await stack.enter_async_context(
                    sse_client(cfg.url, httpx_client_factory=httpx_client_factory)
                )
            elif transport_type == "streamableHttp":
                http_client = await stack.enter_async_context(
                    httpx.AsyncClient(
                        headers=cfg.headers or None,
                        follow_redirects=True,
                        timeout=None,
                    )
                )
                read, write, _ = await stack.enter_async_context(
                    streamable_http_client(cfg.url, http_client=http_client)
                )
            else:
                logger.warning("MCP server '{}': unknown transport type '{}'", name, transport_type)
                continue

            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            tools = await session.list_tools()
            for tool_def in tools.tools:
                wrapper = MCPToolWrapper(session, name, tool_def, tool_timeout=cfg.tool_timeout)
                registry.register(wrapper)
                logger.debug("MCP: registered tool '{}' from server '{}'", wrapper.name, name)

            logger.info("MCP server '{}': connected, {} tools registered", name, len(tools.tools))
        except (Exception, BaseExceptionGroup) as e:
            # anyio 任务组可能抛出 BaseExceptionGroup，例如 streamableHttp 取消作用域失败；
            # 它在 Python 3.11+ 中不是 Exception 的子类。
            logger.error("MCP server '{}': failed to connect: {}", name, e)
