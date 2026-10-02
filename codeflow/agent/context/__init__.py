"""重导出为 AgentLoop 组装 System Prompt 与 History 的 ContextBuilder。

实现位于 ``builder.py``。当前 Turn Request Path 的最终 Context 由 ContextAssembler 所有，但
External Callers 仍应保持稳定 Import：

    from codeflow.agent.context import ContextBuilder

这个 Module 不再导出其他内部 Helper，避免调用方绕开 Builder/Assembler 的 Ownership Boundary。
"""

from codeflow.agent.context.builder import ContextBuilder

__all__ = ["ContextBuilder"]
