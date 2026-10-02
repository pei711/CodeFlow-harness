"""重导出为 Delegated Task 创建 Child AgentLoops 的 SubagentManager。

Implementation 位于 ``manager.py``。External Caller 应保持稳定 Import：

    from codeflow.agent.subagent import SubagentManager

该入口不启动 Child、Sandbox 或 Background Task，只稳定 Package Surface。
"""

from codeflow.agent.subagent.manager import SubagentManager

__all__ = ["SubagentManager"]
