"""在 Agent Side 把 AgentLoop 适配为 Spine ``TurnRunner``。

每个 Turn 直接委托 ``AgentLoop.run_turn``。Adapter 持有具体 Loop，所以必须位于 Agent Package；
Spine never imports Agent，依赖反转保持成立。Constructor 的 ``stream`` 是 Canon Q2-D Assembly
Switch：TUI 等 Streaming Outlet 传 True，回复以 StreamDelta 发送并 dissolves；REPL 传 False，
回复只形成一个 Text。Adapter 不改变 Request、Emit、Drain 或 TurnOutcome。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeflow.agent.loop import AgentLoop
    from codeflow.spine.runner import Drain, Emit, TurnOutcome
    from codeflow.spine.turn import TurnRequest


class AgentTurnRunner:
    def __init__(self, agent_loop: AgentLoop, *, stream: bool) -> None:
        self._loop = agent_loop
        self._stream = stream

    async def run(self, req: TurnRequest, emit: Emit, drain: Drain) -> TurnOutcome:
        return await self._loop.run_turn(req, emit, drain, stream=self._stream)
