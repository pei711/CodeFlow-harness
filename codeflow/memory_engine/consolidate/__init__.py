"""使用 ``MEMORY.md`` 与 ``HISTORY.md`` 的 Two-layer Long-term Memory。

``MemoryStore`` 提供 Locked Reads/Writes，拥有两份 Markdown 的 Durable IO；``MemoryConsolidator`` 在此
基础上增加 Boundary-aware、Token-driven Compaction，把较早 History 提炼到 Stable Memory。写入成功、
压缩完成与新 Memory 实际注入后续 Prompt 是三个不同阶段，不能混为一谈。
"""

from codeflow.memory_engine.consolidate.consolidator import (
    MemoryConsolidator,
    MemoryStore,
)

__all__ = ["MemoryStore", "MemoryConsolidator"]
