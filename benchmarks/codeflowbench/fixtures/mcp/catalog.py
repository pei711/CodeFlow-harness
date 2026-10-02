from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

MCP_CATALOG_SIZE = 64
MCP_SERVER_NAME = "codeflowbench"
_DESCRIPTION_PREFIX = (
    "Resolve one deterministic catalog record using the requested resource, operation, and numeric value."
)
_PARAMETERS = {
    "type": "object",
    "properties": {
        "resource": {
            "type": "string",
            "description": "Deterministic resource identifier.",
            "minLength": 1,
        },
        "operation": {
            "type": "string",
            "description": "Catalog operation to perform.",
            "enum": ["inspect", "transform", "validate"],
        },
        "value": {
            "type": "integer",
            "description": "Deterministic numeric payload.",
            "minimum": 0,
            "maximum": 10_000,
        },
    },
    "required": ["resource", "operation", "value"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class CatalogToolDefinition:
    index: int
    name: str
    description: str
    parameters: dict[str, Any]

    @property
    def runtime_name(self) -> str:
        return f"mcp_{MCP_SERVER_NAME}_{self.name}"

    @property
    def description_prefix(self) -> str:
        return _DESCRIPTION_PREFIX

    @property
    def parameters_digest(self) -> str:
        return _digest(self.parameters)


@lru_cache(maxsize=1)
def catalog_definitions() -> tuple[CatalogToolDefinition, ...]:
    return tuple(
        CatalogToolDefinition(
            index=index,
            name=f"catalog_probe_{index:02d}",
            description=f"{_DESCRIPTION_PREFIX} Catalog slot {index:02d}.",
            parameters=json.loads(json.dumps(_PARAMETERS)),
        )
        for index in range(MCP_CATALOG_SIZE)
    )


def catalog_digest() -> str:
    return _digest(
        [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in catalog_definitions()
        ]
    )


def receipt_payload(
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    normalized = {
        "tool": tool_name,
        "resource": arguments["resource"],
        "operation": arguments["operation"],
        "value": arguments["value"],
    }
    return {
        **normalized,
        "receipt": _digest(normalized),
    }


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
