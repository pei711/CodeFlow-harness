"""Eval Engine 写回 `MemoryEngine` 的 Write-back Adapter 入口。

这里公开 `EvalAdapter`，把 Judge 产生的 Verdict 转换成 Memory Engine 能持久化的 Feedback Event。它
只负责跨模块协议适配，不生成 Verdict，也不决定某条评估是否足以支持正向能力结论。
"""

from codeflow.eval_engine.adapter.adapter import EvalAdapter

__all__ = ["EvalAdapter"]
