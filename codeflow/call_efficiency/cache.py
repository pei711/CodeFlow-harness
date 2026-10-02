"""Provider-aware Cache Ownership 与 Request Planning。

不同 Provider 对 Prompt Cache 的控制方式不同：有些自动缓存，有些要求 Anthropic-style 显式
``cache_control``，还有些完全不支持。本模块先通过 `cache_capability` 判定所有权，再负责验证、移除
或补充请求中的 Cache Markers。这样 Runtime 不会把某家 Provider 的协议字段错误发送给另一家，
也不会覆盖调用方已经明确给出的合法缓存计划。

这些函数只规划请求结构；Cache Marker 存在不代表远端一定命中缓存，实际 Read/Write Tokens 仍以
Provider 返回的 Usage Evidence 为准。
"""

from __future__ import annotations

import copy
from enum import Enum
from typing import Any

from codeflow.providers.registry import find_by_model

_CACHE_CONTROL = {"type": "ephemeral"}


class CacheCapability(str, Enum):
    NONE = "none"
    PROVIDER_AUTOMATIC = "provider_automatic"
    ANTHROPIC_EXPLICIT = "anthropic_explicit"


def cache_capability(
    model: str,
    *,
    explicit_cache_control_supported: bool | None = None,
) -> CacheCapability:
    key = model.lower().removeprefix("openrouter/")
    spec = find_by_model(key)
    if spec is not None and spec.name in {"deepseek", "openai", "openai_codex"}:
        return CacheCapability.PROVIDER_AUTOMATIC
    supports_explicit = (
        spec is not None and spec.supports_prompt_caching
        if explicit_cache_control_supported is None
        else explicit_cache_control_supported
    )
    if spec is not None and spec.name == "anthropic" and supports_explicit:
        return CacheCapability.ANTHROPIC_EXPLICIT
    return CacheCapability.NONE


def has_cache_control(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> bool:
    return bool(_cache_control_values(messages, tools))


def valid_cache_control(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    max_breakpoints: int,
) -> bool:
    values = _cache_control_values(messages, tools)
    return bool(values) and len(values) <= max_breakpoints and all(value == _CACHE_CONTROL for value in values)


def strip_cache_control(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    new_messages: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list) or not any(
            isinstance(block, dict) and "cache_control" in block for block in content
        ):
            new_messages.append(message)
            continue
        blocks = [
            {key: value for key, value in block.items() if key != "cache_control"} if isinstance(block, dict) else block
            for block in content
        ]
        new_messages.append({**message, "content": blocks})

    if tools is None:
        new_tools = None
    else:
        new_tools = [{key: value for key, value in tool.items() if key != "cache_control"} for tool in tools]
    return new_messages, new_tools


def _cache_control_values(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> list[Any]:
    values: list[Any] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            values.extend(
                block["cache_control"] for block in content if isinstance(block, dict) and "cache_control" in block
            )
    if tools:
        values.extend(tool["cache_control"] for tool in tools if "cache_control" in tool)
    return values


def apply_anthropic_cache_plan(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    max_breakpoints: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    if has_cache_control(messages, tools):
        return messages, tools

    budget = max_breakpoints
    new_messages = list(messages)
    new_tools = tools

    if tools and budget:
        new_tools = copy.deepcopy(tools)
        new_tools[-1] = {**new_tools[-1], "cache_control": _CACHE_CONTROL}
        budget -= 1

    system_index = _last_role_index(new_messages, "system")
    if system_index is not None and budget:
        new_messages[system_index] = _mark_message_tail(new_messages[system_index])
        budget -= 1

    if budget:
        non_system = [index for index, message in enumerate(new_messages) if message.get("role") != "system"]
        for index in non_system[-budget:]:
            new_messages[index] = _mark_message_tail(new_messages[index])

    return new_messages, new_tools


def _last_role_index(messages: list[dict[str, Any]], role: str) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == role:
            return index
    return None


def _mark_message_tail(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if isinstance(content, str):
        blocks = [{"type": "text", "text": content, "cache_control": _CACHE_CONTROL}]
    elif isinstance(content, list) and content and isinstance(content[-1], dict):
        blocks = copy.deepcopy(content)
        blocks[-1] = {**blocks[-1], "cache_control": _CACHE_CONTROL}
    else:
        return message
    return {**message, "content": blocks}


__all__ = [
    "CacheCapability",
    "apply_anthropic_cache_plan",
    "cache_capability",
    "has_cache_control",
    "strip_cache_control",
    "valid_cache_control",
]
