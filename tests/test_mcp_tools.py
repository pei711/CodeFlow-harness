from __future__ import annotations

from types import SimpleNamespace

import pytest
from mcp import types

from codeflow.agent.tools.mcp import MCPToolWrapper
from codeflow.agent.tools.registry import ToolRegistry


class _Session:
    def __init__(self, result: types.CallToolResult) -> None:
        self._result = result

    async def call_tool(self, name: str, arguments: dict):
        return self._result


@pytest.mark.parametrize("is_error", [False, True])
async def test_mcp_result_preserves_server_error_status(is_error: bool):
    session = _Session(
        types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text="permission denied",
                )
            ],
            isError=is_error,
        )
    )
    wrapper = MCPToolWrapper(
        session,
        "docs",
        SimpleNamespace(
            name="read",
            description="Read docs",
            inputSchema={"type": "object", "properties": {}},
        ),
    )
    registry = ToolRegistry()
    registry.register(wrapper)

    result = await registry.execute(wrapper.name, {})

    assert result == "permission denied"
    assert result.failed is is_error
