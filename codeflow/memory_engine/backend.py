"""`MemoryBackend` Protocol，是每个 Memory Plugin 都要实现的 Single Contract。

MB-1 引入这条 **New Seam**，连接 `AgentLoop` 与 Memory Subsystem。它刻意区别于
:mod:`codeflow.memory_engine.base` 中旧的 :class:`MemoryEngine` ABC，使代码库迁移期间两套接口可以共存。

Plugin Authors 必须理解三个 Design Points：

- ``recall`` 显式命名 Track，接收 ``user_id`` XOR ``agent_id``。Active Host 的 Memory 使用 User Track；
  Agent Track 为 Public Plugin Protocol Compatibility 保留。
- ``Memory.metadata`` 是 **Escape Hatch**。``text`` 与 ``score`` 已标准化；Categories、Episode Type、
  Native ID、Source Labels 等 Backend-specific 信息全部放入 ``metadata``。Host Context Assembler **不**
  读取 Metadata，只有 Pre-rendered ``text`` 进入 Prompt；把 Memory Hit 重新发为 `ScoredSkill` 的
  Skill-source Adapter 才读取 Metadata 构造 Qualified ID。
- ``feedback`` **允许 No-op**。它为 Public Protocol Compatibility 保留，但 Active Host 当前不 Dispatch。

Protocol 使用 :func:`typing.runtime_checkable`，所以 Tests 可执行 ``isinstance(x, MemoryBackend)``；代价
是任何 Surface Matching 的 Class，包括 Duck-typed Mocks，都会通过。Contract Tests 因此无需继承 Base
Class，但 Runtime Check 也不证明语义实现正确。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# 数据载体
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Memory:
    """一次 :meth:`MemoryBackend.recall` 返回的单条 Hit。

    ``frozen=True`` 让 Host 可以在组件间传递 Memory List，而无需担心 Adapter Code 重新绑定字段、修改
    他人视图。`text` 是可注入内容，`score` 是标准化相关度，`metadata` 保留 Backend Details；Frozen
    不会深度冻结 Metadata Dict，Consumer 仍应把它当作只读。
    """

    text: str
    """LLM 在 Prompt 中 Verbatim 看到的 Pre-rendered Content。

    Adapter 负责 Formatting，例如 EverMem 返回 Natural-sentence Facts，Mem0 返回 Category-tagged Blobs。
    Host 除了把多个 Hits Join 成 Block 外，**永不** Post-process ``text``；因此去敏、边界标记与可读性
    必须在进入该字段前完成。
    """

    score: float = 0.0
    """由 Adapter 归一化到 ``[0, 1]`` 的 Relevance。

    Memory Hit 被重新发为 `ScoredSkill` 时，:class:`SkillForgeRouter` 用它做 Cross-source RRF。普通
    ``# Recalled memory`` Injection 中该值仅 Informational，不决定文本是否进入 Prompt。
    """

    metadata: dict[str, Any] = field(default_factory=dict)
    """Adapter-specific Escape Hatch。各 Backend Examples：

    - EverMem: ``{"id": ..., "episode_type": ..., "name": ...,
      "owner_type": "user" | "agent"}``
    - mem0: ``{"id": ..., "categories": [...], "memory_type": ...}``
    - MemOS: ``{"id": ..., "mem_cube_id": ..., "metadata": {...}}``
    - Letta: ``{"archival_memory_id": ...}``
    Host Context Injection 不读取这些字段；它们主要用于 Provenance、Qualified ID 与 Backend-native
    Correlation，不能代替 `text` 中面向模型的内容。
    """


# ---------------------------------------------------------------------------
# 协议
# ---------------------------------------------------------------------------


@runtime_checkable
class MemoryBackend(Protocol):
    """所有 Memory Plugins 实现的 Single Runtime Contract。

    Five Methods 按 Hot-path 排列：

    1. :meth:`recall`：``ContextEngine.assemble`` 每 Turn 以 ``user_id`` 读取 User-track Memory；
    2. :meth:`store`：AgentLoop 每 Turn 后持久化 Conversation Slice；
    3. :meth:`feedback`：允许 No-op 的 Compatibility Hook；
    4. :meth:`start` / :meth:`stop`：由 Host Await 的 Lifecycle。

    Backend 拥有 Transport/Storage State，Host 拥有调用时机与 Context Assembly。协议方法返回不自动证明
    Store 已 Durable 或 Recalled Text 适合正向结论，具体 Adapter 必须提供这些保证。
    """

    async def recall(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        top_k: int,
    ) -> list[Memory]:
        """为一个 Track 检索匹配 ``query`` 的 Memories。

        ``user_id`` / ``agent_id`` 必须 Exactly One Set，即 XOR。Caller 在构造路径已经知道 Track，所以
        显式命名，而不是塞进带 Prefix 的 Opaque String。Backend 可以只支持 ``user_id``，对 Agent Track
        返回 ``[]``；Neither/Both 都是 Caller Bug，也应返回空列表。

        Empty Result 是合法的 No Hits；Transport Error、Auth Failure 等应 Raise。Host 会通过
        ``SkillForgeRouter._safe_search`` 降级到其他 Sources。`top_k` 是最大候选意图，不保证 Backend
        一定返回该数量，成功返回也不表示内容已进入最终 Prompt。
        """
        ...

    async def store(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
    ) -> None:
        """持久化一个 Session Slice。

        ``messages`` 使用 AgentLoop 已产生的 ``{"role", "content", ...}`` List-of-dicts Shape，Adapter
        无需中间转换。Backend 可自行 Chunk、Deduplicate 或 Extract；Protocol 对每次调用采用
        Fire-and-forget 结果形态，不返回对象。

        Transport/Auth Error 必须 Raise，让 AgentLoop Surface；Host **不会** Silently Swallow Store
        Failure。正常返回的 Durable 含义由 Backend 实现定义，若远端只接受异步队列，Adapter 应在自身
        文档中说明证据边界。
        """
        ...

    async def feedback(self, signals: dict[str, Any]) -> None:
        """消费 Free-form Signal Dict，例如 Injected/Used Skill IDs。

        No-op Implementation 完全 Valid 且 Idiomatic，因为 Active Host 当前不 Dispatch 该 Hook。未来使用
        时，各 Backend 必须自行验证 Signal Schema；调用不应被视为学习或演化已完成。
        """
        ...

    async def start(self) -> None:
        """执行 One-time / Idempotent Initialization，例如 Open Connections、Warm Caches、Run Migrations。

        Host 在 Agent Boot 时 Exactly Once Await；Failure 会 Abort Startup。实现仍应保持 Idempotent，便于
        部分初始化后的清理或防御性重试。
        """
        ...

    async def stop(self) -> None:
        """执行 One-time / Idempotent Teardown。

        Adapter 必须让它在 Failed ``start`` 后也能安全调用，以清理 Partial-init State。Stop 返回只说明
        Backend Lifecycle 已收尾，不影响已写入的 Durable Memory。
        """
        ...


__all__ = ["Memory", "MemoryBackend"]
