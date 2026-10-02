"""忠实复现 Hermes Agent 的 ``system_and_3`` Prompt-caching Strategy。

Source：``NousResearch/hermes-agent/agent/prompt_caching.py``（v0.9.0，2026-04-13）。

Strategy 最多放置 4 个 ``cache_control`` Breakpoints：

1. System Prompt，即 Index 0 且 Role 等于 System 的消息；
2. 最后三条 **Non-system Messages**，构成位于 Tail 的 Rolling Window。

本模块把 Hermes Logic 包装为 `TokenStrategy`，使它能与 CodeFlow 自己的 `CacheOptimizer` 一起安装进
`StrategyRegistry`，进行 A/B Comparison。

相对 CodeFlow `CacheOptimizer` 的 Key Behavioral Differences：它 **不** 标记 Tools Schema，四个断点
全部用于 Messages；**不** 设置 Mid-history Breakpoint；使用每次 Iteration / Turn 都会移动的
**Rolling Tail Window**；并像 Hermes Original 一样先 Deep-copy 再修改。
"""

from __future__ import annotations

import copy
from typing import Any

from loguru import logger

from codeflow.providers.registry import find_by_model
from codeflow.token_wise.base import TokenStrategy

_CACHE_CONTROL = {"type": "ephemeral"}


def _supports_cache_control(model: str) -> bool:
    if not model:
        return False
    spec = find_by_model(model)
    return spec is not None and spec.supports_prompt_caching


def _apply_cache_marker(msg: dict[str, Any]) -> None:
    """给单条 Message 添加 ``cache_control``，支持 `str` / `list` / `None` Content。

    逻辑与 Hermes 的 ``_apply_cache_marker`` 相同，但去掉了只影响 ``tool`` Role 的
    ``native_anthropic`` Flag。本复现不单独标记 Tool Messages：Hermes 选择的是 Non-system
    Messages，而 Tool-role 与 Assistant/User 一样通过 Content-list Path 处理。

    函数会原地修改传入 Dict；安全性依赖调用方先 Deep-copy Messages。空 Content 在消息顶层加
    Marker，字符串会包装成 Text Block，非空列表则只修改最后一个 Dict Block。
    """
    content = msg.get("content")

    if content is None or content == "":
        msg["cache_control"] = _CACHE_CONTROL
        return

    if isinstance(content, str):
        msg["content"] = [{"type": "text", "text": content, "cache_control": _CACHE_CONTROL}]
        return

    if isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict):
            last["cache_control"] = _CACHE_CONTROL


class SystemAndTailCacheStrategy(TokenStrategy):
    """``system + tail`` Cache Placement，忠实复现 Hermes Agent ``system_and_3``。

    Source 与 Credit 见 Module Docstring。Breakpoints 分配为：`bp1` 放在 Index 0 的 System Message，
    `bp2` 到 `bp4` 放在最后三条 Non-system Messages 上并随 Tail Rolling。若没有 System Message，剩余
    四个名额都可用于尾部消息。

    Strategy 在修改前 Deep-copies 所有 Messages，因此 Caller 的 Original List 与嵌套 Blocks 都不会
    被触碰。Tools 原样返回且不加 Marker；不支持 Prompt Caching 的模型则整组输入直接透传。
    """

    name = "system_and_tail"

    async def before_llm_call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, str]:
        if not _supports_cache_control(model):
            return messages, tools, model

        new_messages = copy.deepcopy(messages)

        breakpoints_used = 0

        # bp1：系统提示
        if new_messages and new_messages[0].get("role") == "system":
            _apply_cache_marker(new_messages[0])
            breakpoints_used += 1

        # bp2–4：最后 N 条非 system 消息（N = 4 - breakpoints_used）
        remaining = 4 - breakpoints_used
        non_sys_indices = [i for i in range(len(new_messages)) if new_messages[i].get("role") != "system"]

        for idx in non_sys_indices[-remaining:]:
            _apply_cache_marker(new_messages[idx])

        used = breakpoints_used + min(remaining, len(non_sys_indices))
        logger.debug("SystemAndTailCacheStrategy: placed {} breakpoint(s) on model={}", used, model)

        # Hermes 不标记工具，因此 4 个断点全部用于消息。
        return new_messages, tools, model
