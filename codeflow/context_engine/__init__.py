"""提供 Context Management Engine 的稳定公开入口。

当前只有一个 Engine：:class:`ContextAssembler`。:func:`build_context_engine` 用扁平
:class:`SegmentBuilder` 列表组装它，列表包含 seg1–5 与 Curator；历史上的 ``legacy`` /
``curator`` / ``default`` 分裂已经收敛。这里重导出组装协议、TurnContext、HistoryTrimmer 和
Factory，调用方无需依赖内部文件布局，也不应再按旧 Engine 名称分支。
"""

from codeflow.context_engine.assembler import ContextAssembler
from codeflow.context_engine.base import (
    AssembledPrefix,
    AssemblyContext,
    ContextEngine,
    Segment,
    SegmentBuilder,
)
from codeflow.context_engine.curator import TurnContext
from codeflow.context_engine.factory import build_context_engine
from codeflow.context_engine.history_trimmer import HistoryTrimmer

__all__ = [
    "AssembledPrefix",
    "AssemblyContext",
    "ContextAssembler",
    "ContextEngine",
    "HistoryTrimmer",
    "Segment",
    "SegmentBuilder",
    "TurnContext",
    "build_context_engine",
]
