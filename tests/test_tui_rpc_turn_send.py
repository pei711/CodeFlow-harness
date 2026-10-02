"""Tests for ``turn.send`` real handler.

``turn.send`` submits a turn onto the spine (build_tui Scheduler) and returns
``{turn_id, accepted}`` synchronously; the turn streams out via the hub/sink.
These tests drive the handler with a fake Scheduler + emitter (the spine path
itself is covered in ``test_tui_rpc_spine.py``).

Spec source:
- ``codeflow/tui_rpc/models.py`` ``TurnSendParams`` / ``TurnSendResult``
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from codeflow.tui_rpc.dispatcher import Dispatcher
from codeflow.tui_rpc.errors import ModelNotAvailableError, RpcError, TurnInProgressError
from codeflow.tui_rpc.methods.image import image_attach, pending_images
from codeflow.tui_rpc.methods.turn import register_turn_methods, turn_send


class FakeHandle:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    async def result(self):
        return None


class FakeScheduler:
    """Records submitted requests; returns a handle. Optionally raises."""

    def __init__(self, raises: Exception | None = None, *, busy: bool = False) -> None:
        self.submitted: list = []
        self._raises = raises
        self._busy = busy

    def submit(self, req):
        if self._raises is not None:
            raise self._raises
        self.submitted.append(req)
        return FakeHandle()

    def has_pending_or_running(self, conversation_id: str) -> bool:
        return self._busy


class FakeEmitter:
    def __init__(self) -> None:
        self.emitted: list[tuple[str, dict]] = []

    async def emit(self, session_key: str, event: dict) -> None:
        self.emitted.append((session_key, event))

    def types(self) -> list[str]:
        return [e["type"] for _k, e in self.emitted]


@pytest.fixture(autouse=True)
def _clear_active_turns():
    from codeflow.tui_rpc.methods import turn as _turn_mod

    _turn_mod._active_turns.clear()
    _turn_mod._active_request_keys.clear()
    yield
    _turn_mod._active_turns.clear()
    _turn_mod._active_request_keys.clear()


@pytest.fixture
def dispatcher() -> Dispatcher:
    d = Dispatcher()
    register_turn_methods(
        d,
        emitter=FakeEmitter(),
        scheduler=FakeScheduler(),
        turn_ids={},
        submission_ids={},
    )
    return d


async def test_turn_send_happy_path_returns_turn_id_and_accepted() -> None:
    scheduler = FakeScheduler()
    turn_ids: dict[int, str] = {}
    submission_ids: dict[int, str] = {}
    emitter = FakeEmitter()

    result = await turn_send(
        {
            "session_key": "tui:default",
            "content": "hello",
            "submission_id": "submission-1",
        },
        emitter=emitter,
        scheduler=scheduler,
        turn_ids=turn_ids,
        submission_ids=submission_ids,
    )

    assert set(result) == {"turn_id", "accepted"}
    assert result["accepted"] is True
    assert isinstance(result["turn_id"], str) and len(result["turn_id"]) >= 16

    assert len(scheduler.submitted) == 1
    assert scheduler.submitted[0].conversation == "tui:default"
    request_key = id(scheduler.submitted[0])
    assert turn_ids[request_key] == result["turn_id"]
    assert submission_ids[request_key] == "submission-1"
    assert emitter.types() == []


async def test_turn_send_acquires_scheduler_from_runtime_host() -> None:
    scheduler = FakeScheduler()
    acquired: list[str] = []

    async def _acquire_scheduler():
        acquired.append("ready")
        return scheduler

    result = await turn_send(
        {"session_key": "tui:default", "content": "hello"},
        scheduler_factory=_acquire_scheduler,
        turn_ids={},
        submission_ids={},
    )

    assert result["accepted"] is True
    assert acquired == ["ready"]
    assert len(scheduler.submitted) == 1


async def test_turn_send_generates_unique_turn_ids() -> None:
    scheduler = FakeScheduler()
    turn_ids: dict[int, str] = {}
    r1 = await turn_send({"session_key": "tui:a", "content": "x"}, scheduler=scheduler, turn_ids=turn_ids)
    r2 = await turn_send({"session_key": "tui:b", "content": "x"}, scheduler=scheduler, turn_ids=turn_ids)
    assert r1["turn_id"] != r2["turn_id"]


async def test_turn_send_binds_active_slot_after_submit() -> None:
    from codeflow.tui_rpc.methods import turn as turn_mod

    scheduler = FakeScheduler()
    await turn_send({"session_key": "tui:default", "content": "hi"}, scheduler=scheduler, turn_ids={})
    assert turn_mod.is_turn_active("tui:default") is True


async def test_turn_send_rejects_active_turn_with_minus_32003() -> None:
    scheduler = FakeScheduler()
    turn_ids: dict[int, str] = {}
    await turn_send({"session_key": "tui:default", "content": "first"}, scheduler=scheduler, turn_ids=turn_ids)

    with pytest.raises(TurnInProgressError) as excinfo:
        await turn_send({"session_key": "tui:default", "content": "second"}, scheduler=scheduler, turn_ids=turn_ids)

    assert excinfo.value.CODE == -32003
    assert excinfo.value.MESSAGE == "turn_in_progress"


async def test_turn_send_rejects_scheduler_lane_work_with_minus_32003() -> None:
    scheduler = FakeScheduler(busy=True)

    with pytest.raises(TurnInProgressError) as excinfo:
        await turn_send(
            {"session_key": "tui:default", "content": "queued behind system"},
            scheduler=scheduler,
            turn_ids={},
            submission_ids={},
        )

    assert excinfo.value.CODE == -32003
    assert scheduler.submitted == []


async def test_turn_send_rejects_unknown_model_with_minus_32008() -> None:
    with patch(
        "codeflow.tui_rpc.methods.turn._resolve_model",
        side_effect=ModelNotAvailableError("no provider configured"),
    ):
        with pytest.raises(ModelNotAvailableError) as excinfo:
            await turn_send({"session_key": "tui:default", "content": "x"}, scheduler=FakeScheduler())

    assert excinfo.value.CODE == -32008


async def test_turn_send_without_scheduler_emits_model_not_available() -> None:

    emitter = FakeEmitter()
    result = await turn_send(
        {
            "session_key": "tui:default",
            "content": "x",
            "submission_id": "submission-1",
        },
        emitter=emitter,
        scheduler=None,
    )
    assert result["accepted"] is True
    assert emitter.types() == ["message.start", "error"]
    assert emitter.emitted[-1][1]["payload"]["code"] == -32008
    assert emitter.emitted[-1][1]["payload"]["submission_id"] == "submission-1"
    assert emitter.emitted[-1][1]["payload"]["turn_id"] == result["turn_id"]


async def test_turn_send_without_scheduler_reports_discarded_attachment(tmp_path) -> None:
    image = tmp_path / "pending.svg"
    image.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    await image_attach({"session_id": "tui:default", "path": str(image)})
    emitter = FakeEmitter()

    await turn_send({"session_key": "tui:default", "content": "x"}, emitter=emitter, scheduler=None)

    assert emitter.emitted[-1][1]["payload"]["attachments_discarded"] is True
    assert pending_images("tui:default") == ()


async def test_turn_send_when_submit_rejected_surfaces_turn_failed() -> None:

    from codeflow.spine.scheduler import SchedulerDrainingError
    from codeflow.tui_rpc.methods import turn as turn_mod

    emitter = FakeEmitter()
    turn_ids: dict[int, str] = {}
    result = await turn_send(
        {"session_key": "tui:default", "content": "x"},
        emitter=emitter,
        scheduler=FakeScheduler(raises=SchedulerDrainingError("draining")),
        turn_ids=turn_ids,
    )
    assert result["accepted"] is True
    assert emitter.types() == ["message.start", "error"]
    assert emitter.emitted[-1][1]["payload"]["message"] == "turn_failed"

    assert turn_ids == {} and "tui:default" not in turn_mod._active_turns


async def test_turn_send_without_scheduler_surfaces_build_error_code() -> None:

    class _BuildErr(RpcError):
        CODE = -32603
        MESSAGE = "internal_error"

    emitter = FakeEmitter()
    build_error = _BuildErr(
        "secret detail",
        data={"public_message": "memory backend configuration requires migration"},
    )
    await turn_send(
        {"session_key": "tui:default", "content": "x"},
        emitter=emitter,
        scheduler=None,
        build_error=build_error,
    )
    assert emitter.types() == ["message.start", "error"]
    payload = emitter.emitted[-1][1]["payload"]
    assert payload["code"] == -32603
    assert payload["message"] == "memory backend configuration requires migration"
    assert "secret detail" not in payload["message"]


async def test_turn_send_without_scheduler_hides_unmarked_build_error_detail() -> None:
    class _BuildErr(RpcError):
        CODE = -32603
        MESSAGE = "internal_error"

    emitter = FakeEmitter()
    await turn_send(
        {"session_key": "tui:default", "content": "x"},
        emitter=emitter,
        scheduler=None,
        build_error=_BuildErr("secret detail"),
    )

    payload = emitter.emitted[-1][1]["payload"]
    assert payload["message"] == "internal_error"
    assert "secret detail" not in payload["message"]


async def test_turn_send_rejects_missing_session_key() -> None:
    with pytest.raises(Exception):  # noqa: BLE001
        await turn_send({"content": "missing session_key"}, scheduler=FakeScheduler())


async def test_turn_send_rejects_missing_content() -> None:
    with pytest.raises(Exception):  # noqa: BLE001
        await turn_send({"session_key": "tui:default"}, scheduler=FakeScheduler())


async def test_turn_send_accepts_optional_channel_chat_id_sender_id() -> None:
    scheduler = FakeScheduler()
    result = await turn_send(
        {
            "session_key": "tui:default",
            "content": "hi",
            "channel": "tui",
            "chat_id": "default",
            "sender_id": "user",
        },
        scheduler=scheduler,
        turn_ids={},
    )
    assert result["accepted"] is True
    src = scheduler.submitted[0].source
    assert (src.channel, src.chat_id, src.sender_id) == ("tui", "default", "user")


async def test_turn_send_dispatches_via_dispatcher(dispatcher: Dispatcher) -> None:
    resp = await dispatcher.dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "turn.send",
            "params": {"session_key": "tui:default", "content": "hello"},
        }
    )

    assert "error" not in resp, f"turn.send unexpectedly raised: {resp}"
    assert set(resp["result"]) == {"turn_id", "accepted"}
    assert resp["result"]["accepted"] is True


async def test_turn_send_dispatcher_returns_minus_32003_on_concurrent_send(
    dispatcher: Dispatcher,
) -> None:
    await dispatcher.dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "turn.send",
            "params": {"session_key": "tui:default", "content": "first"},
        }
    )
    resp = await dispatcher.dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "turn.send",
            "params": {"session_key": "tui:default", "content": "second"},
        }
    )

    assert "error" in resp
    assert resp["error"]["code"] == -32003
    assert resp["error"]["message"] == "turn_in_progress"
