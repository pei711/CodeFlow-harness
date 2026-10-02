"""Memory Contracts、Host-owned Context Carriers 与 Local Skill Routing 的公开入口。

Memory Engine 把长期记忆能力拆成稳定 Backend Protocol、Host 负责的 `AssembledContext` / `TokenBudget`
数据载体，以及 Skill Forge/Local Skill 子系统。Runtime 通过这些接口读取、搜索、更新或注入记忆，不应
直接依赖某个具体存储实现。

Contract Test Classes 采用 Lazy Export，避免 Production ``import codeflow.memory_engine`` 被仅开发环境的
`pytest` 依赖污染。Memory Backend 调用成功只说明完成协议操作；上下文是否实际注入、持久化是否耐久、
以及记忆是否能支持正向任务结论，仍由后续阶段分别验证。
"""

from typing import TYPE_CHECKING

from codeflow.memory_engine.backend import Memory, MemoryBackend
from codeflow.memory_engine.base import AssembledContext, TokenBudget

if TYPE_CHECKING:
    from codeflow.memory_engine.contract_test import (
        LifecycleContractTests,
        MemoryBackendContractTests,
    )

__all__ = [
    "AssembledContext",
    "LifecycleContractTests",
    "Memory",
    "MemoryBackend",
    "MemoryBackendContractTests",
    "TokenBudget",
]


# 合约测试基类位于 ``contract_test``，该模块会在顶层导入仅开发环境依赖的 ``pytest``。
# 如果在此处急切导入，每次 ``import codeflow.memory_engine`` 都会引入 pytest，导致未安装
# pytest 的生产环境（如打包后的 `codeflow`）以 ``ModuleNotFoundError`` 失败。通过 PEP 562
# 延迟暴露，只在测试套件真正访问这些类时解析，此时 pytest 已可用。
def __getattr__(name: str):
    if name in ("LifecycleContractTests", "MemoryBackendContractTests"):
        from codeflow.memory_engine import contract_test

        return getattr(contract_test, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
