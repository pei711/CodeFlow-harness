"""Tool-call Audit Hook。

Default Policy 是 Deterministic Deny-list。启用 ``config.on_tool_audit`` 后，任何 Name 出现在
``config.tool_denylist`` 的 Tool Call 都会用解释信息 Short-circuit Iteration。未来可继续调用 LLM
Safety Check，Prompt Scaffold 见 ``prompts/tool_safety.py``；但在出现 Concrete Need 前该路径仍然
**Unwired**，当前不存在模型安全审查。

Hook 从 ``ctx.response`` 读取 ``tool_calls``，因为 ``before_execute_tools`` 在 LLM 已生成本 Turn 的
Tool-call List **之后（AFTER）**触发，`AgentHookContext` 将 Response 传入此阶段。通过 Denylist 只证明名称未被
明确禁止，不代表 Tool Parameters 或副作用安全。
"""

from __future__ import annotations

import logging
from typing import Any

from codeflow.agent.hook.base import AgentHook, AgentHookContext, HookDecision
from codeflow.eval_engine.config import EvalEngineConfig

logger = logging.getLogger(__name__)


class ToolAuditHook(AgentHook):
    """基于 Deny-list 的 Tool Audit，Deterministic 且 No LLM。

    实例持有 `EvalEngineConfig`，每次 Tool Execution 前读取当前 Denylist 并提取违规名称。关闭总开关、
    关闭阶段开关、名单为空或没有 Offenders 时均 Pass-through；命中时返回用户可见 Halt Message 与
    诊断 Note，不执行任何被拦截 Tool。
    """

    def __init__(self, config: EvalEngineConfig) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return "EvalToolAuditHook"

    async def before_execute_tools(self, ctx: AgentHookContext) -> HookDecision:
        if not (self._config.enabled and self._config.on_tool_audit):
            return HookDecision()
        denylist = set(self._config.tool_denylist)
        if not denylist:
            return HookDecision()

        offenders = _extract_offending_tool_names(ctx.response, denylist)
        if not offenders:
            return HookDecision()
        sorted_offenders = sorted(offenders)

        logger.warning(
            "EvalEngine tool audit: blocking tool calls %s",
            sorted_offenders,
        )
        return HookDecision(
            short_circuit_result=(
                "I tried to invoke a tool that's been blocked by policy: "
                f"{', '.join(sorted_offenders)}. "
                "Please rephrase or escalate if you believe this is intended."
            ),
            notes=[f"tool_denylist_hit names={sorted_offenders}"],
        )


def _extract_offending_tool_names(response: Any, denylist: set[str]) -> set[str]:
    """遍历 ``response.tool_calls`` 或 ``response["tool_calls"]``，返回 ``denylist`` 命中的 Names。

    前者覆盖 `LLMResponse`，后者覆盖 Dict；每个 Tool Call 还兼容对象 ``name``、顶层 Dict ``name`` 与
    OpenAI-style ``function.name`` 三种结构。结果用 Set 去重。任何 Structural Mismatch 都返回 Empty
    Set，使 Malformed Response 不会 Crash Hook；这是一种可用性降级，也意味着畸形结构不会被此
    Denylist 拦截，Schema Validation 仍应由 Provider 层承担。
    """
    if response is None:
        return set()
    tool_calls = getattr(response, "tool_calls", None)
    if tool_calls is None and isinstance(response, dict):
        tool_calls = response.get("tool_calls")
    if not tool_calls:
        return set()

    offenders: set[str] = set()
    for tc in tool_calls:
        name = (
            getattr(tc, "name", None)
            or (tc.get("name") if isinstance(tc, dict) else None)
            or (tc.get("function", {}).get("name") if isinstance(tc, dict) else None)
        )
        if isinstance(name, str) and name in denylist:
            offenders.add(name)
    return offenders


__all__ = ["ToolAuditHook"]
