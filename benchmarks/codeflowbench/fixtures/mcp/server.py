from __future__ import annotations

import json
import os
from pathlib import Path

from catalog import catalog_definitions, receipt_payload

from mcp.server.fastmcp import FastMCP

_RECEIPT_ENV = "CODEFLOWBENCH_MCP_RECEIPTS"


def _handler_for(tool_name: str):
    async def handler(
        resource: str,
        operation: str,
        value: int,
    ) -> str:
        receipt = receipt_payload(
            tool_name,
            {
                "resource": resource,
                "operation": operation,
                "value": value,
            },
        )
        receipt_path = os.environ.get(_RECEIPT_ENV)
        if receipt_path:
            path = Path(receipt_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(receipt, sort_keys=True) + "\n")
        return json.dumps(receipt, sort_keys=True)

    return handler


def main() -> None:
    server = FastMCP("CodeFlowBench deterministic catalog")
    for tool in catalog_definitions():
        server.add_tool(
            _handler_for(tool.name),
            name=tool.name,
            description=tool.description,
            structured_output=False,
        )
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
