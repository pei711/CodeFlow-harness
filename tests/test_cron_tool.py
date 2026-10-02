"""Unit tests for CronTool argument handling (codeflow/proactive_engine/schedulers/cron/tool.py).

Focus: ``tz`` is only meaningful for a cron-expression schedule. For an
every/at schedule the tool ignores it (rather than erroring) so the agent does
not have to retry just to drop a no-op arg.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from codeflow.proactive_engine.schedulers.cron.tool import CronTool


def _tool() -> tuple[CronTool, MagicMock]:
    cron = MagicMock()
    cron.add_job.return_value = SimpleNamespace(id="j1", name="reminder")
    tool = CronTool(cron)
    tool.set_context("tui", "default")
    return tool, cron


async def test_add_every_with_tz_is_tolerated_and_dropped() -> None:
    tool, cron = _tool()
    result = await tool.execute(action="add", message="drink water", every_seconds=60, tz="Asia/Shanghai")
    assert result.startswith("Created job")
    schedule = cron.add_job.call_args.kwargs["schedule"]
    assert schedule.kind == "every"
    assert getattr(schedule, "tz", None) is None


async def test_add_naive_at_with_tz_anchors_to_that_zone() -> None:

    from datetime import datetime
    from zoneinfo import ZoneInfo

    tool, cron = _tool()
    result = await tool.execute(action="add", message="ping", at="2026-06-24T15:00:00", tz="Asia/Shanghai")
    assert result.startswith("Created job")
    schedule = cron.add_job.call_args.kwargs["schedule"]
    assert schedule.kind == "at"
    expected_ms = int(datetime(2026, 6, 24, 15, 0, 0, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp() * 1000)
    assert schedule.at_ms == expected_ms


async def test_add_offset_aware_at_ignores_tz_param() -> None:

    from datetime import datetime

    tool, cron = _tool()
    await tool.execute(action="add", message="ping", at="2026-06-24T15:00:00+09:00", tz="Asia/Shanghai")
    schedule = cron.add_job.call_args.kwargs["schedule"]
    expected_ms = int(datetime.fromisoformat("2026-06-24T15:00:00+09:00").timestamp() * 1000)
    assert schedule.at_ms == expected_ms


async def test_non_runnable_schedule_surfaces_service_error() -> None:

    tool, cron = _tool()
    cron.add_job.side_effect = ValueError("at time is in the past")
    result = await tool.execute(action="add", message="late", at="2020-01-01T00:00:00")
    assert result == "Error: at time is in the past"


async def test_add_cron_expr_with_valid_tz_uses_it() -> None:
    tool, cron = _tool()
    result = await tool.execute(action="add", message="daily standup", cron_expr="0 9 * * *", tz="Asia/Shanghai")
    assert result.startswith("Created job")
    schedule = cron.add_job.call_args.kwargs["schedule"]
    assert schedule.kind == "cron" and schedule.tz == "Asia/Shanghai"


async def test_add_cron_expr_with_bad_tz_still_errors() -> None:
    tool, cron = _tool()
    result = await tool.execute(action="add", message="daily", cron_expr="0 9 * * *", tz="Not/AZone")
    assert "unknown timezone" in result
    cron.add_job.assert_not_called()


async def test_cron_context_cannot_create_another_job() -> None:
    tool, cron = _tool()
    token = tool.set_cron_context(True)
    try:
        result = await tool.execute(action="add", message="recursive", every_seconds=60)
    finally:
        tool.reset_cron_context(token)

    assert result == "Error: cannot schedule new jobs from within a cron job execution"
    cron.add_job.assert_not_called()
