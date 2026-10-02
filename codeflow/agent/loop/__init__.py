"""重导出 CodeFlow L2 ReAct Executor `AgentLoop` 与内部 TurnOutcome。

完整 ``AgentLoop`` Implementation 位于 ``main.py``。Package Shape 预留未来拆分为 ``main.py`` /
``dispatch.py`` / ``runner.py`` 的空间，不让 External Import 再次变化。Caller 应持续使用：

    from codeflow.agent.loop import AgentLoop

该入口只稳定 Symbol Path，不改变 AgentLoop 的 Runtime/Turn Ownership。
"""

from codeflow.agent.loop.main import AgentLoop, TurnOutcome

__all__ = ["AgentLoop", "TurnOutcome"]
