"""CronService.add_job rejects non-runnable schedules at add time.

A schedule that maps to no next run (an `at` in the past, every_seconds <= 0,
an unparseable cron expr) used to be stored as a job that silently never fires
— a false success to every caller (cron tool / CLI). add_job now
raises ValueError so each caller surfaces it. This is the single service-layer
invariant covering all three callers.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from codeflow.proactive_engine.schedulers.cron.service import CronService
from codeflow.proactive_engine.schedulers.cron.types import CronSchedule


@pytest.fixture
def svc(tmp_path: Path) -> CronService:
    return CronService(tmp_path / "jobs.json")


def _add(svc: CronService, schedule: CronSchedule, message: str = "m"):
    return svc.add_job(name="t", schedule=schedule, message=message, deliver=True, channel="cli", to="direct")


def test_past_at_is_rejected(svc: CronService) -> None:
    past_ms = int(time.time() * 1000) - 60_000
    with pytest.raises(ValueError, match="at time is in the past"):
        _add(svc, CronSchedule(kind="at", at_ms=past_ms))
    assert svc.list_jobs() == []


def test_non_positive_every_is_rejected(svc: CronService) -> None:
    with pytest.raises(ValueError, match="every_seconds must be positive"):
        _add(svc, CronSchedule(kind="every", every_ms=0))
    with pytest.raises(ValueError, match="every_seconds must be positive"):
        _add(svc, CronSchedule(kind="every", every_ms=-5_000))
    assert svc.list_jobs() == []


def test_invalid_cron_expr_is_rejected(svc: CronService) -> None:
    with pytest.raises(ValueError, match="invalid cron expression"):
        _add(svc, CronSchedule(kind="cron", expr="not a cron expr"))
    assert svc.list_jobs() == []


def test_runnable_schedules_still_created(tmp_path: Path) -> None:

    future_ms = int(time.time() * 1000) + 60_000
    for schedule in (
        CronSchedule(kind="at", at_ms=future_ms),
        CronSchedule(kind="every", every_ms=60_000),
        CronSchedule(kind="cron", expr="0 9 * * *"),
    ):
        svc = CronService(tmp_path / f"{schedule.kind}.json")
        job = _add(svc, schedule)
        assert job.state.next_run_at_ms is not None
        assert len(svc.list_jobs()) == 1


def test_schedule_kinds_and_dst_use_fake_clock(tmp_path: Path) -> None:
    now = {"value": datetime(2026, 3, 7, 12, 0, tzinfo=UTC)}

    def now_fn() -> datetime:
        return now["value"]

    at_ms = int(datetime(2026, 3, 7, 12, 5, tzinfo=UTC).timestamp() * 1000)
    at_job = _add(
        CronService(tmp_path / "at.json", now_fn=now_fn),
        CronSchedule(kind="at", at_ms=at_ms),
    )
    interval_job = _add(
        CronService(tmp_path / "interval.json", now_fn=now_fn),
        CronSchedule(kind="every", every_ms=90_000),
    )
    before_dst = _add(
        CronService(tmp_path / "before-dst.json", now_fn=now_fn),
        CronSchedule(kind="cron", expr="0 9 * * *", tz="America/New_York"),
    )

    assert at_job.state.next_run_at_ms == at_ms
    assert interval_job.state.next_run_at_ms == int(datetime(2026, 3, 7, 12, 1, 30, tzinfo=UTC).timestamp() * 1000)
    assert before_dst.state.next_run_at_ms == int(datetime(2026, 3, 7, 14, 0, tzinfo=UTC).timestamp() * 1000)

    now["value"] = datetime(2026, 3, 8, 12, 0, tzinfo=UTC)
    after_dst = _add(
        CronService(tmp_path / "after-dst.json", now_fn=now_fn),
        CronSchedule(kind="cron", expr="0 9 * * *", tz="America/New_York"),
    )
    assert after_dst.state.next_run_at_ms == int(datetime(2026, 3, 8, 13, 0, tzinfo=UTC).timestamp() * 1000)
