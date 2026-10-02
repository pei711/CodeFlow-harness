"""Tests for tui_rpc confirm round-trip.

Covers the ConfirmBroker (notification emit + request_id→Future registry +
fail-safe), the confirm.respond handler + umbrella registration, and the
typer.confirm injection layer. Destructive TUI confirms become answerable
via a generic RPC round-trip.
"""

from __future__ import annotations

import asyncio

import pytest

from codeflow.tui_rpc import confirm_broker as cb
from codeflow.tui_rpc.confirm_broker import ConfirmBroker
from codeflow.tui_rpc.methods.confirm import confirm_respond, register_confirm_methods


def _frame_collector() -> tuple[list[dict], object]:
    frames: list[dict] = []

    async def send_frame(frame: dict) -> None:
        frames.append(frame)

    return frames, send_frame


async def _wait_for_frame(frames: list[dict], timeout: float = 1.0) -> dict:
    """Poll until the broker has emitted its confirm.request frame."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not frames:
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("confirm.request frame never emitted")
        await asyncio.sleep(0.005)
    return frames[0]


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


async def test_confirm_request_notification_emitted() -> None:
    frames, send_frame = _frame_collector()
    broker = ConfirmBroker(send_frame)

    task = asyncio.create_task(broker.await_confirm("Continue?", default=False))
    frame = await _wait_for_frame(frames)

    assert "id" not in frame
    assert frame["jsonrpc"] == "2.0"
    assert frame["method"] == "confirm.request"
    params = frame["params"]
    assert isinstance(params["request_id"], str) and params["request_id"]
    assert params["prompt"] == "Continue?"
    assert params["default"] is False

    broker.resolve(params["request_id"], True)
    await task


async def test_broker_await_returns_answer() -> None:
    frames, send_frame = _frame_collector()
    broker = ConfirmBroker(send_frame)

    task = asyncio.create_task(broker.await_confirm("Reset?", default=False))
    frame = await _wait_for_frame(frames)
    broker.resolve(frame["params"]["request_id"], True)

    assert await task is True


async def test_confirm_respond_resolves_future() -> None:
    frames, send_frame = _frame_collector()
    broker = ConfirmBroker(send_frame)

    task = asyncio.create_task(broker.await_confirm("Reset?", default=False))
    frame = await _wait_for_frame(frames)
    rid = frame["params"]["request_id"]

    assert broker.resolve(rid, False) is True
    assert await task is False

    assert broker.resolve(rid, True) is False


async def test_confirm_respond_unknown_id_idempotent() -> None:
    _frames, send_frame = _frame_collector()
    broker = ConfirmBroker(send_frame)

    assert broker.resolve("does-not-exist", True) is False


async def test_confirm_hard_limit_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cb, "_CONFIRM_HARD_LIMIT_S", 0.05)
    frames, send_frame = _frame_collector()
    broker = ConfirmBroker(send_frame)

    result = await broker.await_confirm("Continue?", default=False)
    assert result is False
    await _wait_for_frame(frames)


async def test_confirm_task_cancellation_propagates() -> None:
    frames, send_frame = _frame_collector()
    broker = ConfirmBroker(send_frame)

    task = asyncio.create_task(broker.await_confirm("Continue?", default=False))
    await _wait_for_frame(frames)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


async def test_broker_cancel_all_failsafe() -> None:
    frames, send_frame = _frame_collector()
    broker = ConfirmBroker(send_frame)

    task = asyncio.create_task(broker.await_confirm("Continue?", default=False))
    await _wait_for_frame(frames)

    broker.cancel_all()
    assert await task is False


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


async def test_confirm_respond_handler_resolves() -> None:
    frames, send_frame = _frame_collector()
    broker = ConfirmBroker(send_frame)
    task = asyncio.create_task(broker.await_confirm("Reset?", default=False))
    frame = await _wait_for_frame(frames)
    rid = frame["params"]["request_id"]

    result = await confirm_respond({"request_id": rid, "answer": True}, confirm_broker=broker)

    assert result == {"ok": True}
    assert await task is True


async def test_confirm_respond_handler_unknown_id_returns_not_ok() -> None:
    _frames, send_frame = _frame_collector()
    broker = ConfirmBroker(send_frame)

    result = await confirm_respond({"request_id": "nope", "answer": True}, confirm_broker=broker)

    assert result == {"ok": False}


async def test_register_confirm_methods_adds_respond() -> None:
    from codeflow.tui_rpc.dispatcher import Dispatcher

    _frames, send_frame = _frame_collector()
    broker = ConfirmBroker(send_frame)
    dispatcher = Dispatcher()
    register_confirm_methods(dispatcher, confirm_broker=broker)

    assert "confirm.respond" in dispatcher.methods()
