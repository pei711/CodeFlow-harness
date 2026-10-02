"""CronService claims only jobs whose channel is in its allowed_channels.

The gateway's allowed_channels is IM-only (no "tui"), so a TUI-originated cron
job is fired by the TUI process, never claimed/forwarded by the gateway — a
TUI-set reminder always delivers to the TUI instead of racing to an IM channel.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from codeflow.proactive_engine.schedulers.cron.service import _CLAIM_TTL_MS, CronService
from codeflow.proactive_engine.schedulers.cron.types import CronSchedule

_START = datetime(2026, 1, 1, tzinfo=UTC)
_DUE = _START + timedelta(minutes=1)


def _add_due_tui_job(store_path: Path) -> str:
    svc = CronService(store_path, now_fn=lambda: _START)
    job = svc.add_job(
        name="tui reminder",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="drink water",
        deliver=True,
        channel="tui",
        to="default",
    )
    return job.id


async def _fired_ids(allowed: set[str], store_path: Path) -> list[str]:
    fired: list[str] = []

    async def on_job(job) -> None:
        fired.append(job.id)

    svc = CronService(store_path, allowed_channels=allowed, now_fn=lambda: _DUE)
    svc.on_job = on_job
    await svc.run_due()
    return fired


async def test_gateway_does_not_claim_tui_job(tmp_path: Path) -> None:
    store = tmp_path / "jobs.json"
    job_id = _add_due_tui_job(store)

    fired = await _fired_ids({"weixin"}, store)
    assert job_id not in fired


async def test_owning_process_claims_its_tui_job(tmp_path: Path) -> None:
    store = tmp_path / "jobs.json"
    job_id = _add_due_tui_job(store)

    fired = await _fired_ids({"tui"}, store)
    assert job_id in fired


async def test_two_workers_in_one_process_deliver_once(tmp_path: Path) -> None:
    now = {"value": datetime(2026, 1, 1, tzinfo=UTC)}
    started = asyncio.Event()
    release = asyncio.Event()
    deliveries: list[str] = []

    def now_fn() -> datetime:
        return now["value"]

    async def callback_a(job) -> None:
        deliveries.append(f"a:{job.id}")
        started.set()
        await release.wait()

    async def callback_b(job) -> None:
        deliveries.append(f"b:{job.id}")

    store = tmp_path / "jobs.json"
    worker_a = CronService(store, on_job=callback_a, now_fn=now_fn)
    worker_b = CronService(store, on_job=callback_b, now_fn=now_fn)
    job = worker_a.add_job(
        name="one delivery",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="run",
    )
    now["value"] += timedelta(minutes=1)

    task_a = asyncio.create_task(worker_a.run_due())
    await started.wait()
    await worker_b.run_due()
    release.set()
    await task_a

    assert deliveries == [f"a:{job.id}"]


async def test_expired_owner_cannot_overwrite_new_claim(tmp_path: Path) -> None:
    now = {"value": _START}
    started_a = asyncio.Event()
    started_b = asyncio.Event()
    release_a = asyncio.Event()
    release_b = asyncio.Event()

    def now_fn() -> datetime:
        return now["value"]

    async def callback_a(_job) -> None:
        started_a.set()
        await release_a.wait()
        raise RuntimeError("late failure")

    async def callback_b(_job) -> None:
        started_b.set()
        await release_b.wait()

    store = tmp_path / "jobs.json"
    worker_a = CronService(store, on_job=callback_a, now_fn=now_fn)
    worker_b = CronService(store, on_job=callback_b, now_fn=now_fn)
    worker_a.add_job(
        name="claim transfer",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="run",
    )
    now["value"] = _DUE

    task_a = asyncio.create_task(worker_a.run_due())
    await started_a.wait()
    now["value"] += timedelta(milliseconds=_CLAIM_TTL_MS + 1)
    task_b = asyncio.create_task(worker_b.run_due())
    await started_b.wait()

    release_a.set()
    await task_a
    release_b.set()
    await task_b

    persisted = CronService(store).list_jobs(include_disabled=True)
    assert persisted[0].state.last_status == "ok"
    assert persisted[0].state.last_error is None
    assert persisted[0].state.claimed_by_pid is None
