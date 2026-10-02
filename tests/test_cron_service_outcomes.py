import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from codeflow.proactive_engine.schedulers.cron.service import CronService
from codeflow.proactive_engine.schedulers.cron.types import CronSchedule


def _stored_job(job_id: str, schedule: dict) -> dict:
    return {
        "id": job_id,
        "name": job_id,
        "enabled": True,
        "schedule": schedule,
        "payload": {"kind": "agent_turn", "message": "run"},
        "state": {},
        "createdAtMs": 0,
        "updatedAtMs": 0,
        "deleteAfterRun": False,
    }


async def test_cancelled_job_is_recoverable_and_releases_claim(tmp_path: Path) -> None:
    now = {"value": datetime(2026, 1, 1, tzinfo=UTC)}

    def now_fn() -> datetime:
        return now["value"]

    async def cancel(_job) -> None:
        raise asyncio.CancelledError

    store = tmp_path / "jobs.json"
    service = CronService(store, on_job=cancel, now_fn=now_fn)
    job = service.add_job(
        name="cancelled turn",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="run",
    )
    now["value"] += timedelta(minutes=1)

    with pytest.raises(asyncio.CancelledError):
        await service.run_due()

    persisted = CronService(store).list_jobs(include_disabled=True)
    assert len(persisted) == 1
    assert persisted[0].id == job.id
    assert persisted[0].enabled is True
    assert persisted[0].state.last_status == "cancelled"
    assert persisted[0].state.last_error == "execution cancelled"
    assert persisted[0].state.claimed_by_pid is None
    assert persisted[0].state.claimed_at_ms is None
    assert persisted[0].state.next_run_at_ms == job.state.next_run_at_ms


async def test_cancelled_manual_run_releases_claim(tmp_path: Path) -> None:
    now = {"value": datetime(2026, 1, 1, tzinfo=UTC)}

    def now_fn() -> datetime:
        return now["value"]

    async def cancel(_job) -> None:
        raise asyncio.CancelledError

    store = tmp_path / "jobs.json"
    service = CronService(store, on_job=cancel, now_fn=now_fn)
    job = service.add_job(
        name="manual cancellation",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="run",
    )

    with pytest.raises(asyncio.CancelledError):
        await service.run_job(job.id)

    persisted = CronService(store).list_jobs(include_disabled=True)
    assert persisted[0].state.last_status == "cancelled"
    assert persisted[0].state.claimed_by_pid is None
    assert persisted[0].state.claimed_at_ms is None


async def test_failed_job_persists_diagnostic_and_releases_claim(tmp_path: Path) -> None:
    now = {"value": datetime(2026, 1, 1, tzinfo=UTC)}

    def now_fn() -> datetime:
        return now["value"]

    async def fail(_job) -> None:
        raise RuntimeError("provider unavailable")

    store = tmp_path / "jobs.json"
    service = CronService(store, on_job=fail, now_fn=now_fn)
    job = service.add_job(
        name="failed turn",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="run",
    )
    now["value"] += timedelta(minutes=1)

    await service.run_due()

    persisted = CronService(store).list_jobs(include_disabled=True)
    assert len(persisted) == 1
    assert persisted[0].id == job.id
    assert persisted[0].state.last_status == "error"
    assert persisted[0].state.last_error == "provider unavailable"
    assert persisted[0].state.claimed_by_pid is None
    assert persisted[0].state.claimed_at_ms is None


async def test_expired_schedule_is_disabled_with_diagnostic(tmp_path: Path) -> None:
    store = tmp_path / "jobs.json"
    store.write_text(
        json.dumps(
            {
                "version": 1,
                "jobs": [
                    _stored_job(
                        "expired",
                        {"kind": "every", "everyMs": 0},
                    )
                ],
            }
        ),
        encoding="utf-8",
    )
    service = CronService(
        store,
        now_fn=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    await service.start()
    service.stop()

    jobs = service.list_jobs(include_disabled=True)
    assert len(jobs) == 1
    assert jobs[0].enabled is False
    assert jobs[0].state.last_status == "expired"
    assert jobs[0].state.last_error == "schedule has no future fire"
    assert jobs[0].state.next_run_at_ms is None
    enabled = service.enable_job("expired", enabled=True)
    assert enabled is not None
    assert enabled.enabled is False
    assert enabled.state.last_status == "expired"
    assert await service.run_job("expired", force=True) is False


async def test_incompatible_job_is_isolated_with_diagnostic(tmp_path: Path) -> None:
    store = tmp_path / "jobs.json"
    incompatible_record = _stored_job(
        "future-format",
        {"kind": "calendar", "calendarId": "team"},
    )
    store.write_text(
        json.dumps(
            {
                "version": 1,
                "jobs": [
                    _stored_job("valid", {"kind": "every", "everyMs": 60_000}),
                    incompatible_record,
                ],
            }
        ),
        encoding="utf-8",
    )
    service = CronService(
        store,
        now_fn=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    await service.start()
    service.stop()

    jobs = {job.id: job for job in service.list_jobs(include_disabled=True)}
    assert jobs["valid"].enabled is True
    assert jobs["future-format"].enabled is False
    assert jobs["future-format"].state.last_status == "incompatible"
    assert jobs["future-format"].state.last_error == "unsupported schedule kind 'calendar'"
    assert jobs["future-format"].state.next_run_at_ms is None
    assert await service.run_job("future-format", force=True) is False
    stored_records = json.loads(store.read_text(encoding="utf-8"))["jobs"]
    assert stored_records[1] == incompatible_record


async def test_invalid_field_type_is_isolated_from_valid_jobs(tmp_path: Path) -> None:
    store = tmp_path / "jobs.json"
    invalid_record = _stored_job("bad-time", {"kind": "every", "everyMs": "60000"})
    store.write_text(
        json.dumps(
            {
                "version": 1,
                "jobs": [
                    _stored_job("valid", {"kind": "every", "everyMs": 60_000}),
                    invalid_record,
                ],
            }
        ),
        encoding="utf-8",
    )
    service = CronService(
        store,
        now_fn=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    await service.start()
    service.stop()

    jobs = {job.id: job for job in service.list_jobs(include_disabled=True)}
    assert jobs["valid"].enabled is True
    assert jobs["bad-time"].enabled is False
    assert jobs["bad-time"].state.last_status == "incompatible"
    assert jobs["bad-time"].state.last_error == "schedule.everyMs must be an integer or null"
    stored_records = json.loads(store.read_text(encoding="utf-8"))["jobs"]
    assert stored_records[1] == invalid_record


async def test_future_store_version_is_read_only_and_preserved(tmp_path: Path) -> None:
    store = tmp_path / "jobs.json"
    future_store = {
        "version": 2,
        "metadata": {"writer": "future-runtime"},
        "jobs": [_stored_job("future-job", {"kind": "every", "everyMs": 60_000})],
    }
    store.write_text(json.dumps(future_store), encoding="utf-8")
    service = CronService(
        store,
        now_fn=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    await service.start()
    service.stop()

    jobs = service.list_jobs(include_disabled=True)
    assert jobs[0].state.last_status == "incompatible"
    assert jobs[0].state.last_error == "unsupported store version '2'"
    assert json.loads(store.read_text(encoding="utf-8")) == future_store
    with pytest.raises(ValueError, match="unsupported cron store version '2'"):
        service.add_job(
            name="must not mix schemas",
            schedule=CronSchedule(kind="every", every_ms=60_000),
            message="run",
        )


async def test_restart_recovers_one_pending_interval_without_backfill(tmp_path: Path) -> None:
    now = {"value": datetime(2026, 1, 1, tzinfo=UTC)}

    def now_fn() -> datetime:
        return now["value"]

    store = tmp_path / "jobs.json"
    initial = CronService(store, now_fn=now_fn)
    job = initial.add_job(
        name="interval",
        schedule=CronSchedule(kind="every", every_ms=300_000),
        message="run",
    )
    now["value"] += timedelta(minutes=20)
    fired: list[str] = []

    async def callback(cron_job) -> None:
        fired.append(cron_job.id)

    recovered = CronService(store, on_job=callback, now_fn=now_fn)
    await recovered.start()
    recovered.stop()
    await recovered.run_due()

    assert fired == [job.id]
    persisted = recovered.list_jobs(include_disabled=True)
    assert persisted[0].state.next_run_at_ms == int(datetime(2026, 1, 1, 0, 25, tzinfo=UTC).timestamp() * 1000)


async def test_restart_does_not_repeat_completed_single_fire(tmp_path: Path) -> None:
    store = tmp_path / "jobs.json"
    completed = _stored_job(
        "completed",
        {
            "kind": "at",
            "atMs": int(datetime(2026, 1, 1, 0, 5, tzinfo=UTC).timestamp() * 1000),
        },
    )
    completed["state"] = {
        "nextRunAtMs": None,
        "lastRunAtMs": int(datetime(2026, 1, 1, 0, 5, tzinfo=UTC).timestamp() * 1000),
        "lastStatus": "ok",
    }
    store.write_text(
        json.dumps({"version": 1, "jobs": [completed]}),
        encoding="utf-8",
    )
    fired: list[str] = []

    async def callback(job) -> None:
        fired.append(job.id)

    service = CronService(
        store,
        on_job=callback,
        now_fn=lambda: datetime(2026, 1, 1, 0, 10, tzinfo=UTC),
    )
    await service.start()
    service.stop()
    await service.run_due()

    jobs = service.list_jobs(include_disabled=True)
    assert fired == []
    assert jobs[0].enabled is False
    assert jobs[0].state.next_run_at_ms is None
    assert jobs[0].state.last_status == "ok"
