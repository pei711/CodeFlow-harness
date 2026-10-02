"""由 Runtime 统一拥有的 Provider Call Efficiency 与 Evidence 入口。

这个 Package 覆盖一次模型调用从请求预处理、Provider 包装、Usage 标准化到 Call Record 持久化的
完整计量边界。外部模块通常只需导入 `CallEfficiency` 或 `CallEfficiencyProvider`；`PreparedCall`、
`CallUsage` 与 `CallRecord` 分别描述调用前、调用后和可持久化证据。它记录模型调用事实，但不把
Token/Cost Receipt 误当成用户任务已经完成的证明。
"""

from codeflow.call_efficiency.models import (
    CALL_RECORD_SCHEMA,
    CallRecord,
    CallUsage,
    PreparedCall,
)
from codeflow.call_efficiency.provider import CallEfficiencyProvider
from codeflow.call_efficiency.runtime import CallEfficiency

__all__ = [
    "CALL_RECORD_SCHEMA",
    "CallEfficiency",
    "CallEfficiencyProvider",
    "CallRecord",
    "CallUsage",
    "PreparedCall",
]
