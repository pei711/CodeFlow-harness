import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from codeflow.agent.loop import AgentLoop
from codeflow.sandbox.interfaces import SandboxExecutor
from codeflow.spine import ChatType, Origin, Source, TurnRequest


class Provider:
    def get_default_model(self):
        return "stub"


async def noop(_event):
    pass


def request(conversation):
    return TurnRequest(
        Origin.USER, Source("test", conversation, "u", ChatType.DM), conversation, conversation=conversation
    )


async def test_turn_failure_defers_shared_executor_reset_until_other_turn_finishes(tmp_path, monkeypatch):
    stopped = []

    class Executor(SandboxExecutor):
        async def exec(self, *args, **kwargs):
            raise NotImplementedError

        async def stop(self):
            stopped.append(True)

    agent = AgentLoop(Provider(), tmp_path)
    agent._executor = Executor()
    agent._mcp_connected = True
    first_entered, second_entered = asyncio.Event(), asyncio.Event()
    fail, release = asyncio.Event(), asyncio.Event()

    async def process(req, **kwargs):
        if req.text == "a":
            first_entered.set()
            await fail.wait()
            raise ValueError("one conversation failed")
        second_entered.set()
        await release.wait()
        assert not stopped
        return "second completed", []

    monkeypatch.setattr(agent, "_process_message", process)
    first = asyncio.create_task(agent.run_turn(request("a"), noop, lambda: [], stream=False))
    second = asyncio.create_task(agent.run_turn(request("b"), noop, lambda: [], stream=False))
    try:
        await asyncio.wait_for(first_entered.wait(), 5)
        await asyncio.wait_for(second_entered.wait(), 5)
        fail.set()
        with pytest.raises(ValueError, match="one conversation"):
            await first
        assert not stopped and agent._executor_started
        release.set()
        assert (await second).explicit_reply
        assert stopped == [True]
        assert agent._executor_turn_users == 0
        assert not agent._executor_started
        assert not agent._mcp_connected
    finally:
        fail.set()
        release.set()
        await asyncio.gather(first, second, return_exceptions=True)
        await agent.close()


@pytest.mark.parametrize("cancel_first", [False, True])
async def test_mcp_initialization_is_awaited_and_cancelled_setup_can_retry(tmp_path, monkeypatch, cancel_first):
    import codeflow.agent.tools.mcp as mcp

    agent = AgentLoop(Provider(), tmp_path, mcp_servers={"svc": object()})
    monkeypatch.setattr(agent, "_start_executor", AsyncMock())
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def connect(*args, **kwargs):
        calls.append(True)
        agent.tools.register(
            mcp.MCPToolWrapper(object(), "svc", SimpleNamespace(name="read", description="read", inputSchema={}))
        )
        entered.set()
        await release.wait()

    monkeypatch.setattr(mcp, "connect_mcp_servers", connect)
    first = asyncio.create_task(agent._connect_mcp())
    await asyncio.wait_for(entered.wait(), 5)
    second = asyncio.create_task(agent._connect_mcp())
    try:
        await asyncio.sleep(0)
        assert not second.done()
        if cancel_first:
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
        release.set()
        await asyncio.gather(first, second, return_exceptions=cancel_first)
        assert agent._mcp_connected and not agent._mcp_connecting
        assert agent.tools.has("mcp_svc_read")
        assert len(calls) == (2 if cancel_first else 1)
    finally:
        release.set()
        await asyncio.gather(first, second, return_exceptions=True)
        await agent.close()
        assert not agent.tools.has("mcp_svc_read")
