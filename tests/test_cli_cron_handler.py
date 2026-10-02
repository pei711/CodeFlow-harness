"""Trigger-path tests for ``make_on_cron_job`` (codeflow/cli/_cron_handler.py).

Distinct scope from:
  - ``test_cron_delivery.py`` — unit-tests the ``resolve_cron_delivery``
    helper in isolation.

This file targets the factory closure itself: trigger-time dynamic
config read, outbound expansion, ephemeral vs pass-through routing.
Heavily mock-based so it stays a pure unit test (no real channels,
no real agent loop, no real session disk I/O).

Every cron turn now runs through the spine ``submit``. A single-target
delivering job rides the hub (the test asserts the submitted request's
source, since the outlets are wired only in the assembly tests); a
broadcast (len > 1) submits with an ephemeral source the hub drops and is
delivered explicitly via ``hub.post`` (one spine Text per target); a silent
job delivers nothing.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from codeflow.cli import _cron_handler
from codeflow.cli._cron_handler import make_on_cron_job
from codeflow.proactive_engine.schedulers import cron as cron_package
from codeflow.proactive_engine.schedulers.cron.service import CronService
from codeflow.proactive_engine.schedulers.cron.types import (
    CronJob,
    CronJobState,
    CronPayload,
    CronSchedule,
)
from codeflow.spine import Origin, Text


def _make_job(
    *,
    channel: str = "cli",
    to: str = "direct",
    deliver: bool = True,
    name: str = "test_job",
) -> CronJob:
    return CronJob(
        id=f"job_{name}",
        name=name,
        enabled=True,
        schedule=CronSchedule(kind="at", at_ms=1000),
        payload=CronPayload(
            kind="agent_turn",
            message="reminder body source",
            deliver=deliver,
            channel=channel,
            to=to,
        ),
        state=CronJobState(),
    )


@pytest.fixture
def fake_hub() -> MagicMock:
    hub = MagicMock()
    hub.post = AsyncMock()
    return hub


@pytest.fixture
def spine() -> SimpleNamespace:
    """Spine submit mock + readback map, mimicking the gateway capturing runner:
    submit records the request and stores the reply text under req.conversation
    before result() resolves (so the handler reads it back)."""
    readback: dict[str, str] = {}
    captured: list = []

    class _Handle:
        async def result(self):
            return None

    def _submit(req):
        captured.append(req)
        readback[req.conversation] = "resolved body"
        return _Handle()

    return SimpleNamespace(submit=_submit, readback=readback, captured=captured)


@pytest.fixture
def fake_session_mgr() -> MagicMock:
    """SessionManager mock — find_most_recent_chat_id returns fixed ids."""
    mgr = MagicMock()
    mgr.find_most_recent_chat_id.side_effect = lambda ch: {
        "telegram": "tg_user_1",
        "feishu": "ou_user_1",
    }.get(ch)
    return mgr


@pytest.fixture
def patch_cron_config(monkeypatch):
    """Patch ``load_config()`` to return a stub with controllable
    ``cron.forward_channels``. Returns a setter the test calls."""

    def _patch(forward_channels: list[str]):
        config = MagicMock()
        config.cron = SimpleNamespace(forward_channels=forward_channels)
        monkeypatch.setattr(
            "codeflow.config.loader.load_config",
            lambda: config,
        )

    return _patch


def test_cron_handler_has_no_proactive_collaborators() -> None:
    params = inspect.signature(make_on_cron_job).parameters
    assert "agent" not in params
    assert {"sentinel_runner", "system_events", "wake"}.isdisjoint(params)


def test_cron_production_imports_have_no_proactive_side_effect_dependencies() -> None:
    forbidden = (
        "codeflow.proactive_engine.sentinel",
        "codeflow.proactive_engine.schedulers.heartbeat",
        "codeflow.proactive_engine.system_events",
        "codeflow.proactive_engine.wake",
    )
    paths = [Path(_cron_handler.__file__)]
    paths.extend(Path(cron_package.__file__).parent.glob("*.py"))

    imported: list[tuple[Path, str]] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.append((path, node.module))
            elif isinstance(node, ast.Import):
                imported.extend((path, alias.name) for alias in node.names)

    violations = [(path.name, module) for path, module in imported if module.startswith(forbidden)]
    assert violations == []


# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────


async def test_trigger_ephemeral_broadcasts_to_all_enabled(
    fake_hub,
    spine,
    fake_session_mgr,
    patch_cron_config,
):
    """forward_channels=['*'] + cli job → broadcast to every enabled channel.

    More than one target → the turn submits with the ephemeral source (hub
    drops the reply) and the resolved text is posted to each target.
    """
    patch_cron_config(["*"])
    channel_manager = SimpleNamespace(enabled_channels=["telegram", "feishu"])

    handler = make_on_cron_job(
        fake_hub,
        submit=spine.submit,
        readback_texts=spine.readback,
        channel_manager=channel_manager,
        session_manager=fake_session_mgr,
    )

    await handler(_make_job(channel="cli"))

    assert len(spine.captured) == 1 and spine.captured[0].origin is Origin.CRON
    assert fake_hub.post.await_count == 2
    sent = [call.args[0] for call in fake_hub.post.await_args_list]
    assert all(isinstance(out, Text) for out in sent)
    by_channel = {out.source.channel: out.source.chat_id for out in sent}
    assert by_channel == {"telegram": "tg_user_1", "feishu": "ou_user_1"}
    assert all(out.content == "resolved body" for out in sent)


# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────


async def test_trigger_ephemeral_restricted_to_single_target_rides_hub(
    fake_hub,
    spine,
    fake_session_mgr,
    patch_cron_config,
):
    """forward_channels=['telegram'] resolves to one target → the turn submits
    with that target as the source (hub delivers); no explicit broadcast post."""
    patch_cron_config(["telegram"])
    channel_manager = SimpleNamespace(enabled_channels=["telegram", "feishu"])

    handler = make_on_cron_job(
        fake_hub,
        submit=spine.submit,
        readback_texts=spine.readback,
        channel_manager=channel_manager,
        session_manager=fake_session_mgr,
    )

    await handler(_make_job(channel="cli"))

    assert len(spine.captured) == 1
    assert spine.captured[0].source.channel == "telegram"
    assert spine.captured[0].source.chat_id == "tg_user_1"
    fake_hub.post.assert_not_awaited()


async def test_trigger_tui_treated_same_as_cli(
    fake_hub,
    spine,
    fake_session_mgr,
    patch_cron_config,
):
    """channel='tui' is also ephemeral and gets forwarded (single target → hub)."""
    patch_cron_config(["feishu"])
    channel_manager = SimpleNamespace(enabled_channels=["telegram", "feishu"])

    handler = make_on_cron_job(
        fake_hub,
        submit=spine.submit,
        readback_texts=spine.readback,
        channel_manager=channel_manager,
        session_manager=fake_session_mgr,
    )

    await handler(_make_job(channel="tui", to="default"))

    assert spine.captured[0].source.channel == "feishu"
    fake_hub.post.assert_not_awaited()


async def test_trigger_real_channel_passthrough(
    fake_hub,
    spine,
    fake_session_mgr,
    patch_cron_config,
):
    """telegram job ignores forward_channels — delivered to its own (channel, to)
    via the hub (single pass-through target).

    Guards against the regression where someone wires forward_channels
    into every path; per-job binding for real channels must stay sacred.
    """
    patch_cron_config(["feishu"])
    channel_manager = SimpleNamespace(enabled_channels=["telegram", "feishu"])

    handler = make_on_cron_job(
        fake_hub,
        submit=spine.submit,
        readback_texts=spine.readback,
        channel_manager=channel_manager,
        session_manager=fake_session_mgr,
    )

    await handler(_make_job(channel="telegram", to="explicit_chat_999"))

    assert spine.captured[0].source.channel == "telegram"
    assert spine.captured[0].source.chat_id == "explicit_chat_999"
    fake_hub.post.assert_not_awaited()


# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────


async def test_trigger_no_overlap_emits_no_outbound(
    fake_hub,
    spine,
    fake_session_mgr,
    patch_cron_config,
):
    """forward_channels=['xyz'] but enabled={'telegram'} → 0 targets.

    The turn still submits (for side-effects); the ephemeral source has no
    outlet so the hub drops it and there is no broadcast.
    """
    patch_cron_config(["xyz"])
    channel_manager = SimpleNamespace(enabled_channels=["telegram"])

    handler = make_on_cron_job(
        fake_hub,
        submit=spine.submit,
        readback_texts=spine.readback,
        channel_manager=channel_manager,
        session_manager=fake_session_mgr,
    )

    await handler(_make_job(channel="cli"))

    assert len(spine.captured) == 1
    fake_hub.post.assert_not_awaited()


async def test_trigger_deliver_false_emits_no_outbound(
    fake_hub,
    spine,
    fake_session_mgr,
    patch_cron_config,
):
    """A job with deliver=False is run for side-effects only — the turn submits
    but nothing is delivered (no hub outlet for the ephemeral source, no
    broadcast)."""
    patch_cron_config(["*"])
    channel_manager = SimpleNamespace(enabled_channels=["telegram"])

    handler = make_on_cron_job(
        fake_hub,
        submit=spine.submit,
        readback_texts=spine.readback,
        channel_manager=channel_manager,
        session_manager=fake_session_mgr,
    )

    await handler(_make_job(channel="cli", deliver=False))

    assert len(spine.captured) == 1
    fake_hub.post.assert_not_awaited()


# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────


async def test_trigger_dynamic_config_reload(
    fake_hub,
    spine,
    fake_session_mgr,
    monkeypatch,
):
    """``cron config set forward_channels`` must take effect on the next
    trigger WITHOUT recreating the handler — the closure reads config
    fresh each fire. Each resolves to a single target → rides the hub."""
    state = {"forward_channels": ["telegram"]}

    def _make_config():
        config = MagicMock()
        config.cron = SimpleNamespace(forward_channels=state["forward_channels"])
        return config

    monkeypatch.setattr("codeflow.config.loader.load_config", _make_config)
    channel_manager = SimpleNamespace(enabled_channels=["telegram", "feishu"])

    handler = make_on_cron_job(
        fake_hub,
        submit=spine.submit,
        readback_texts=spine.readback,
        channel_manager=channel_manager,
        session_manager=fake_session_mgr,
    )

    await handler(_make_job(channel="cli", name="t1"))
    assert spine.captured[-1].source.channel == "telegram"

    state["forward_channels"] = ["feishu"]

    await handler(_make_job(channel="cli", name="t2"))
    assert spine.captured[-1].source.channel == "feishu"
    fake_hub.post.assert_not_awaited()


# ─────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────


async def test_spine_path_returns_and_consumes_readback(
    fake_hub,
    fake_session_mgr,
    patch_cron_config,
):
    patch_cron_config(["*"])
    channel_manager = SimpleNamespace(enabled_channels=["telegram"])
    readback_texts: dict[str, str] = {}
    captured: dict[str, object] = {}

    class _Handle:
        async def result(self):

            readback_texts["cron:job_t1"] = "reminder done at 17:05"
            return None

    def _submit(req):
        captured["req"] = req
        return _Handle()

    handler = make_on_cron_job(
        fake_hub,
        submit=_submit,
        readback_texts=readback_texts,
        channel_manager=channel_manager,
        session_manager=fake_session_mgr,
    )

    response = await handler(_make_job(channel="telegram", to="c1", name="t1"))

    fake_hub.post.assert_not_awaited()
    assert captured["req"].origin is Origin.CRON
    assert captured["req"].conversation == "cron:job_t1"
    assert response == "reminder done at 17:05"
    assert "cron:job_t1" not in readback_texts


async def test_due_job_runs_through_spine_seam(
    tmp_path,
    fake_hub,
    spine,
    fake_session_mgr,
    patch_cron_config,
):
    patch_cron_config(["*"])
    now = {"value": datetime(2026, 1, 1, tzinfo=UTC)}
    handler = make_on_cron_job(
        fake_hub,
        submit=spine.submit,
        readback_texts=spine.readback,
        channel_manager=SimpleNamespace(enabled_channels=["feishu"]),
        session_manager=fake_session_mgr,
    )
    service = CronService(
        tmp_path / "jobs.json",
        on_job=handler,
        now_fn=lambda: now["value"],
    )
    job = service.add_job(
        name="spine delivery",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="run",
        deliver=True,
        channel="feishu",
        to="ou_user_1",
    )
    now["value"] += timedelta(minutes=1)

    await service.run_due()

    assert len(spine.captured) == 1
    assert spine.captured[0].origin is Origin.CRON
    assert spine.captured[0].conversation == f"cron:{job.id}"
    persisted = service.list_jobs(include_disabled=True)
    assert persisted[0].state.last_status == "ok"
    fake_hub.post.assert_not_awaited()


async def test_spine_error_propagates_without_proactive_side_effects(
    fake_hub,
    patch_cron_config,
):
    patch_cron_config(["*"])

    class _Handle:
        async def result(self):
            raise RuntimeError("provider down")

    handler = make_on_cron_job(
        fake_hub,
        submit=lambda req: _Handle(),
    )

    with pytest.raises(RuntimeError, match="provider down"):
        await handler(_make_job(channel="cli", name="failed"))


async def test_spine_cancellation_propagates_without_proactive_side_effects(
    fake_hub,
    patch_cron_config,
):
    patch_cron_config(["*"])

    class _Handle:
        async def result(self):
            raise asyncio.CancelledError

    handler = make_on_cron_job(
        fake_hub,
        submit=lambda req: _Handle(),
    )

    with pytest.raises(asyncio.CancelledError):
        await handler(_make_job(channel="cli", name="cancelled"))


async def test_silent_job_on_non_ephemeral_channel_warns(
    fake_hub,
    spine,
    fake_session_mgr,
    patch_cron_config,
):
    """gap2 edge: a silent job (deliver=False) on a non-ephemeral channel is only
    reachable by hand-editing jobs.json (every creation path sets deliver=True).
    Under the spine its reply IS delivered (no outlet-less suppression for real
    channels) — the handler warns so the edge is visible."""
    from loguru import logger

    patch_cron_config(["*"])
    channel_manager = SimpleNamespace(enabled_channels=["telegram"])

    handler = make_on_cron_job(
        fake_hub,
        submit=spine.submit,
        readback_texts=spine.readback,
        channel_manager=channel_manager,
        session_manager=fake_session_mgr,
    )

    warnings: list[str] = []
    sink_id = logger.add(lambda m: warnings.append(str(m)), level="WARNING")
    try:
        await handler(_make_job(channel="telegram", to="c1", deliver=False, name="silent_im"))
    finally:
        logger.remove(sink_id)

    assert len(spine.captured) == 1
    assert any("non-ephemeral" in m for m in warnings)


# ─────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────


async def test_repl_assembly_cron_renders_once_via_clioutlet_not_broadcast(fake_hub, patch_cron_config):
    from codeflow.cli._repl_spine import build_repl
    from codeflow.spine import Text, TurnOutcome, Usage

    patch_cron_config(["*"])

    class _CronEchoLoop:
        async def run_turn(self, req, emit, drain, *, stream) -> TurnOutcome:
            await emit(Text(content=f"cron-reply<{req.conversation}>", source=req.source))
            return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=True)

    rendered: list[str] = []
    scheduler, hub, teardown = build_repl(_CronEchoLoop(), "cli", rendered.append)

    handler = make_on_cron_job(
        hub,
        submit=scheduler.submit,
        channel_manager=SimpleNamespace(enabled_channels=["cli"]),
        session_manager=None,
        default_channel="cli",
    )

    try:
        await handler(_make_job(channel="cli", to="direct", name="repl1"))
        await hub.wait_idle("cli")
    finally:
        await teardown()

    assert rendered == ["cron-reply<cron:job_repl1>"]


# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────


async def test_fan_out_tui_offline_no_im_configured_drops_with_warning(
    fake_hub,
    spine,
    fake_session_mgr,
    patch_cron_config,
):
    """When gateway claims a stale TUI cron job but the user has no IM channel
    enabled, ``resolve_cron_delivery`` SHALL return zero targets and the
    handler SHALL emit no outbound (reminder is dropped). Verifies the
    gateway-fallback degraded path: TUI offline + no IM channel configured.
    """
    patch_cron_config(["*"])
    channel_manager = SimpleNamespace(enabled_channels=[])

    handler = make_on_cron_job(
        fake_hub,
        submit=spine.submit,
        readback_texts=spine.readback,
        channel_manager=channel_manager,
        session_manager=fake_session_mgr,
    )
    await handler(_make_job(channel="tui", name="hydrate_no_im"))

    assert len(spine.captured) == 1
    fake_hub.post.assert_not_awaited()


async def test_fan_out_posts_text_per_target(
    fake_hub,
    spine,
    fake_session_mgr,
    patch_cron_config,
):
    """A broadcast (more than one target) posts one spine Text per target,
    carrying the resolved reply content and a cron-stamped source."""
    patch_cron_config(["*"])
    channel_manager = SimpleNamespace(enabled_channels=["telegram", "feishu"])

    handler = make_on_cron_job(
        fake_hub,
        submit=spine.submit,
        readback_texts=spine.readback,
        channel_manager=channel_manager,
        session_manager=fake_session_mgr,
    )

    await handler(_make_job(channel="cli", name="hydrate"))

    assert fake_hub.post.await_count == 2
    out = fake_hub.post.await_args_list[0].args[0]
    assert isinstance(out, Text)
    assert out.content == "resolved body"
    assert out.source.sender_id == "cron"
    assert out.source.channel in {"telegram", "feishu"}
