"""`CacheOptimizer` 为 Anthropic 合理放置 ``cache_control`` Breakpoints。

Anthropic 每个请求最多允许 4 个 Ephemeral Cache Breakpoints。每个断点的 Cache Key 覆盖从请求开头
到该断点（*包含断点自身*）的所有 Blocks，因此断点位置直接决定哪些前缀真正可缓存。当前策略来自
与 Hermes Agent ``system_and_3`` Strategy 的 Head-to-head Benchmark，详见
``EXPERIMENT_REPORT_CACHE_STRATEGIES.md``。

存在 Tools 时，这是常见 Agent Scenario，四个位置依次是：

1. Tools List End：Tool Schemas 很少变化，缓存后可省去每次重发完整 Schema 的成本；
2. System Prompt Tail：SOUL + USER + MEMORY + Built-ins 通常稳定；
3. ``messages[-2]``：Rolling Tail 覆盖 Intra-turn Tool-chain Prefix，使每轮只为最新 Result 支付
   Fresh Tokens；
4. ``messages[-1]``：本次写入缓存、下次作为 Cache Read，补全滚动覆盖。

不存在 Tools、即 Pure Conversation 时：

1. System Prompt Tail；
2. 最后三条 Non-system Messages，组成与 Hermes ``system_and_3`` 相同的 Rolling Window；基准结果
   表明它适合 Cross-turn Prefix Matching。

不支持 Prompt Caching 的模型使用本策略时是 No-op。Original Messages 与 Tools *绝不原地修改*；
每个新增 ``cache_control`` Marker 的 Block 都通过复制产生，调用方可以安全保留原请求对象。
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


def _last_index(messages: list[dict[str, Any]], *, role: str) -> int | None:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == role:
            return i
    return None


def _mark_cache(block: dict[str, Any]) -> dict[str, Any]:
    return {**block, "cache_control": _CACHE_CONTROL}


def _mark_message_tail(msg: dict[str, Any]) -> dict[str, Any]:
    """复制 ``msg``，并在最后一个 Content Block 上放置 ``cache_control``。

    - `str` Content：包装成一个带 Cache Control 的 Text Block；
    - `list` Content：复制列表，只给 Last Block 添加 Cache Control；
    - 其他类型，包括 `None`、`dict` 与 Unknown：原样返回。

    如果 List 的末项不是 Dict，也会返回原消息，因为无法安全附加协议字段。函数只复制需要改动的
    外层结构，不承诺对所有嵌套内容做完整 Deep Copy。
    """
    content = msg.get("content")
    if isinstance(content, str):
        new_content = [{"type": "text", "text": content, "cache_control": _CACHE_CONTROL}]
    elif isinstance(content, list) and content:
        new_content = list(content)
        last = new_content[-1]
        if isinstance(last, dict):
            new_content[-1] = _mark_cache(last)
        else:
            return msg
    else:
        return msg
    return {**msg, "content": new_content}


class CacheOptimizer(TokenStrategy):
    """根据请求结构自适应放置 Cache Breakpoints 的 Token Strategy。

    默认使用 Anthropic 全部 4 个 Breakpoints，并根据是否存在 Tools 分配：

    - **With tools**：Tools + System + ``msg[-2]`` + ``msg[-1]``，其中 2 个 Rolling；
    - **Without tools**：System + ``msg[-3]`` + ``msg[-2]`` + ``msg[-1]``，其中 3 个 Rolling，等价于
      Hermes ``system_and_3``。

    Rolling Tail 让 Intra-turn Tool Chains 渐进缓存：每次迭代新增的 ``tool_result`` 会成为下一次的
    Cached Prefix；Turn N 的尾部与 Turn N+1 的窗口自然重叠，也让 Cross-turn Prefix 命中缓存。

    实例生命周期中 `max_breakpoints` 固定在 1 到 4，越小会按相同优先级提前停止。每次
    `before_llm_call` 都返回新的受影响结构与原 Model；不支持 Cache Control 时完全透传。
    """

    name = "cache_optimizer"

    def __init__(self, max_breakpoints: int = 4):
        if not 1 <= max_breakpoints <= 4:
            raise ValueError("max_breakpoints must be between 1 and 4")
        self.max_breakpoints = max_breakpoints

    async def before_llm_call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, str]:
        if not _supports_cache_control(model):
            return messages, tools, model

        budget = self.max_breakpoints
        new_tools = tools
        new_messages = list(messages)

        # ── bp1：工具（仅在存在工具时）──
        if tools and budget > 0:
            new_tools = copy.deepcopy(tools)
            new_tools[-1] = _mark_cache(new_tools[-1])
            budget -= 1

        # ── bp2：系统提示尾部 ──
        sys_idx = _last_index(new_messages, role="system")
        if sys_idx is not None and budget > 0:
            new_messages[sys_idx] = _mark_message_tail(new_messages[sys_idx])
            budget -= 1

        # ── bp3..4（无工具时为 bp2..4）：滚动尾部窗口 ──
        # 在最后 N 条非 system 消息上放置断点，其中 N = 剩余预算。
        # 这种滚动窗口方式同时覆盖 Turn 内工具链和跨 Turn 前缀复用。
        if budget > 0:
            non_sys = [i for i in range(len(new_messages)) if new_messages[i].get("role") != "system"]
            for idx in non_sys[-budget:]:
                new_messages[idx] = _mark_message_tail(new_messages[idx])
                budget -= 1

        used = self.max_breakpoints - budget
        logger.debug("CacheOptimizer: placed {} breakpoint(s) on model={}", used, model)
        return new_messages, new_tools, model
