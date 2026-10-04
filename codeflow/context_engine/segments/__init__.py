"""集中导出组成 System Prompt 与 History 的 SegmentBuilder 实现。

每个子模块定义一个 :class:`SegmentBuilder`，共同覆盖上下文片段与 CodeFlow History manager。它们使用同一
AssemblyContext/Segment 接口，并由 :class:`ContextAssembler` 按 order 与 needs_prefix 统一
调度，而不是由 Host 手写不同分支。``render.py`` 保存共享低层渲染函数，这些函数以前属于
``ContextBuilder`` methods；Builder 负责数据来源与 Segment 所有权，render 只负责文本形状。
"""

from codeflow.context_engine.segments.active_skills import ActiveSkillsSegmentBuilder
from codeflow.context_engine.segments.bootstrap import BootstrapSegmentBuilder
from codeflow.context_engine.segments.identity import IdentitySegmentBuilder
from codeflow.context_engine.segments.memory import MemorySegmentBuilder
from codeflow.context_engine.segments.memory import RelevantTurnMemorySegmentBuilder
from codeflow.context_engine.segments.skills import SkillsSegmentBuilder

__all__ = [
    "ActiveSkillsSegmentBuilder",
    "BootstrapSegmentBuilder",
    "IdentitySegmentBuilder",
    "MemorySegmentBuilder",
    "RelevantTurnMemorySegmentBuilder",
    "SkillsSegmentBuilder",
]
