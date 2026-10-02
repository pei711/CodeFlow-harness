"""Memory Engine 保留的 Host-owned Data Carriers。

Phase B-3 已删除 :class:`MemoryEngine` ABC 与 :class:`DefaultMemoryEngine` Facade。L4 Indirection 暴露了
过多 Surface，例如 Third-party Plugins 无法满足的 Subsystem Accessors；Host 现在直接调用
:class:`MemoryStore`、:class:`MemoryConsolidator`、:class:`SkillService`，并把更窄的
:class:`MemoryBackend` Protocol（:mod:`codeflow.memory_engine.backend`）作为 Plugin Contract。

本文件只保留 :class:`ContextEngine.assemble` 返回/消费的 Two Dataclasses：

- :class:`AssembledContext`：交给 AgentLoop 发起 LLM Call 的 Messages + Metadata；
- :class:`TokenBudget`：Per-turn Budget Breakdown，帮助 Engine 决定哪些内容能进入 Prompt。

它们因 Historical Reasons 留在这里，而非 `ContextEngine` 旁；未来可整理为
``codeflow.context_engine.types``。大量 Importers 使用旧路径，所以 Phase B Cleanup 保持 Location Stable。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AssembledContext:
    """一次 ``ContextEngine.assemble()`` Call 的 Output。

    Agent 的 LLM Call **只**使用 `messages` 中内容，Session History 的其他部分不会直接到达 Model。
    `system_prompt_addition` 携带摘要/工作状态，`include_indices` 记录保留的原 History Indices，`metadata`
    服务 Debug/Telemetry。对象表示 Context 已装配，不表示 Provider Call 已成功或任务已完成。
    """

    messages: list[dict[str, Any]]
    system_prompt_addition: str | None = None  # 注入的摘要和工作状态
    include_indices: list[int] | None = None  # 保留下来的会话消息索引
    metadata: dict[str, Any] = field(default_factory=dict)  # 调试和遥测


@dataclass
class TokenBudget:
    """一个 Turn 的 Token Budget Breakdown。

    总 Context Window 先为 Output、Tools 与 System Prompt 保留额度，剩余 `available_history` 才能给
    Session History 与 Archive Injection。Properties 提供 Reserved Total 和默认 75% Compaction
    Threshold；这些值是 Context Planning 预算，不是 Provider 最终 Usage Receipt。
    """

    context_length: int  # 模型上下文窗口
    reserved_output: int  # 为补全预留
    reserved_tools: int  # 提示词中的工具模式和结果
    reserved_system: int  # 系统提示词开销
    available_history: int  # 为会话历史和归档注入留下的余额

    @property
    def total_reserved(self) -> int:
        return self.reserved_output + self.reserved_tools + self.reserved_system

    @property
    def threshold(self) -> int:
        """返回 Compaction Trigger，默认是 ``available_history`` 的 75%。

        超过阈值提示 Engine 应开始压缩，而不是等到 Window 完全用尽。结果向下取整；负数或不合理预算
        不在此验证，应由创建 `TokenBudget` 的上游保证。
        """
        return int(self.available_history * 0.75)


__all__ = [
    "AssembledContext",
    "TokenBudget",
]
