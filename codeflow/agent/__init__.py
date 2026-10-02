"""提供 Agent Core 的轻量入口，并延迟重导出高成本 Runtime 类型。

`AgentLoop`、`ContextBuilder` 与 `MemoryStore` 通过 PEP 562 ``__getattr__`` 按需 Import；普通
``codeflow.agent`` Submodule 导入不会急切加载 AgentLoop 与 LiteLLM，降低 CLI Cold Start。`__dir__`
仍返回稳定 Public Surface，未知名称明确抛 AttributeError。
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeflow.agent.context import ContextBuilder
    from codeflow.agent.loop import AgentLoop
    from codeflow.memory_engine.consolidate.consolidator import MemoryStore

__all__ = ["AgentLoop", "ContextBuilder", "MemoryStore"]

# 通过 PEP 562 延迟重新导出：导入 ``codeflow.agent`` 子模块时不应急切构建 ``AgentLoop``，
# 否则会进一步导入 litellm，成为 CLI 冷启动的主要开销。
_LAZY_EXPORTS = {
    "ContextBuilder": "codeflow.agent.context",
    "AgentLoop": "codeflow.agent.loop",
    "MemoryStore": "codeflow.memory_engine.consolidate.consolidator",
}


def __getattr__(name: str) -> object:
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted(__all__)
