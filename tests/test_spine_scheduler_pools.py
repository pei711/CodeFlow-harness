import asyncio

import pytest

from codeflow.spine import ChatType, Origin, Source, TurnOutcome, TurnRequest, Usage
from codeflow.spine.scheduler import Lane, OriginPools


def _req(origin: Origin, text: str = "x") -> TurnRequest:
    src = Source(channel="t", chat_id=text, sender_id="u", chat_type=ChatType.DM)
    return TurnRequest(origin=origin, source=src, text=text)


async def _sink(event) -> None:
    pass


class Quick:
    async def run(self, req, emit, drain) -> TurnOutcome:
        return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=False)


def test_for_origin_maps_user_separately_from_system_origins():
    pools = OriginPools(user=1, system=1)
    user = pools.for_origin(Origin.USER)
    system = pools.for_origin(Origin.CRON)
    assert user is not system
    for origin in (Origin.CRON, Origin.SUBAGENT):
        assert pools.for_origin(origin) is system


def test_origin_contract_contains_only_retained_sources():
    assert set(Origin) == {Origin.USER, Origin.CRON, Origin.SUBAGENT}


def test_for_origin_rejects_unknown_origin():

    pools = OriginPools(user=1, system=1)
    with pytest.raises(ValueError):
        pools.for_origin("not-an-origin")


async def test_user_pool_is_independent_of_a_full_system_pool():

    pools = OriginPools(user=1, system=1)
    await pools.for_origin(Origin.CRON).acquire()
    lane = Lane(runner=Quick(), pools=pools, sink=_sink, conversation_id="c")
    fut = lane.submit(_req(Origin.USER, "user"))
    assert isinstance(await asyncio.wait_for(fut, timeout=1.0), TurnOutcome)


async def test_no_cross_pool_borrow():

    pools = OriginPools(user=1, system=5)
    await pools.for_origin(Origin.USER).acquire()
    lane = Lane(runner=Quick(), pools=pools, sink=_sink, conversation_id="c")
    blocked = lane.submit(_req(Origin.USER, "b"))
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(blocked), timeout=0.1)
    pools.for_origin(Origin.USER).release()
    assert isinstance(await asyncio.wait_for(blocked, timeout=1.0), TurnOutcome)
