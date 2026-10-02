"""用于 Scheduling Reminders 与 Tasks 的 Cron Tool。

模块包含 Two Layers：

1. ``CronTool``：面向 LLM 的 Agent Tool。创建 Job 时原样保存 Request-time ``channel/chat_id``，这里不做
   Forwarding Decision；
2. ``resolve_cron_delivery``：在 **TRIGGER Time** 由 ``codeflow.cli._cron_handler`` 消费的 Delivery
   Resolver。Real Channels 直接 Pass-through，Ephemeral Channels（CLI/TUI）才 Broadcast/Forward。

把创建与交付分开，保证任务触发时使用最新 Enabled Channels 和 Recent Session，而不是把易变路由冻结
在 Job 中。Tool 返回 Created 只证明 Job 进入 Store，不证明未来 Trigger 或 Delivery 成功。
"""

from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from codeflow.agent.tools.base import Tool
from codeflow.proactive_engine.schedulers.cron.service import CronService
from codeflow.proactive_engine.schedulers.cron.types import CronSchedule

if TYPE_CHECKING:
    from codeflow.session.manager import SessionManager


# ────────────────────────────────────────────────────────────────────
# 投递解析：在触发时消费，而不是创建任务时。
# ────────────────────────────────────────────────────────────────────


@dataclass
class DeliveryTarget:
    """Cron Handler 应交付到的一个 Concrete ``(channel, chat_id)`` Target。

    `channel` 选择 Platform Adapter，`chat_id` 标识该平台内 Recipient/Conversation。Dataclass 只携带已
    解析目标，不包含 Delivery Result；一个 Cron Fire 可解析成多个 `DeliveryTarget`。
    """

    channel: str
    chat_id: str


def is_ephemeral_channel(channel: str, enabled_channels: set[str]) -> bool:
    """判断 Channel 是否为 Host Process 退出后无法自行 Deliver 的 Ephemeral Channel。

    CLI/TUI REPL 关闭、WebUI Session 结束后，原进程内通道不再存在。Rule 是：任何不在
    ``ChannelManager.enabled_channels`` 中的 Channel 都视为 Ephemeral，并需要 Forwarding。Today
    主要匹配 ``cli`` + ``tui``；Future WebUI / Desktop Frontends 会按同一规则自动覆盖，无需新增名称
    分支。返回值只反映当前 Enabled Set，不探测进程存活。
    """
    return channel not in enabled_channels


def resolve_cron_delivery(
    *,
    channel: str,
    chat_id: str,
    forward_channels: list[str],
    enabled_channels: set[str],
    session_manager: "SessionManager | None" = None,
) -> tuple[list[DeliveryTarget], list[str]]:
    """在 Trigger Time 为 Cron Job 解析 Final Delivery Targets。

    返回 ``(targets, warnings)``：Non-ephemeral Channels 依据 Per-job Binding 直接 Pass Through；
    Ephemeral CLI/TUI 按 ``forward_channels`` Broadcast。``["*"]`` 展开为全部 ``enabled_channels``，
    Specific Names 则限制范围；每个 Target 的 ``chat_id`` 由 `session_manager` 中 Most-recent Session
    提供。没有 Recent Session 的 Channel 被跳过并产生 Warning。

    Warnings 供日志暴露，Never Raised；一个 Stale Forward Target 不应破坏仍有其他 Valid Targets 的
    Fire。空 Targets 表示当前没有可交付位置，不等同于 Cron Callback 本身未执行。
    """
    if not is_ephemeral_channel(channel, enabled_channels):
        return [DeliveryTarget(channel=channel, chat_id=chat_id)], []

    if not forward_channels:
        return [], [f"{channel}: no forward_channels configured"]

    if "*" in forward_channels:
        targets_channels = list(enabled_channels)
    else:
        targets_channels = [c for c in forward_channels if c in enabled_channels]

    if not targets_channels:
        return [], [f"{channel}: forward_channels has no overlap with enabled"]

    results: list[DeliveryTarget] = []
    warnings: list[str] = []
    for ch in targets_channels:
        cid = session_manager.find_most_recent_chat_id(ch) if session_manager is not None else None
        if cid:
            results.append(DeliveryTarget(channel=ch, chat_id=cid))
        else:
            warnings.append(f"{ch}: no recent session, skipped")
    return results, warnings


# ────────────────────────────────────────────────────────────────────
# CronTool（面向 LLM）
# ────────────────────────────────────────────────────────────────────


class CronTool(Tool):
    """供 Agent 创建、列出与删除 Reminders/Recurring Tasks 的 LLM-facing Tool。

    `set_context` 在每个请求前绑定 Source Channel 与 Chat ID，`execute` 再把它们写入新 Job。Tool 同时用
    `ContextVar` 标记 Cron Callback Context，阻止计划任务在自身执行时继续创建新 Cron，避免自我复制。
    Schedule 选择、Dedup 与持久化委托给 `CronService`；Description 中的长协议指导模型区分 One-shot
    与 Explicit Recurrence。
    """

    def __init__(self, cron_service: CronService):
        self._cron = cron_service
        self._channel = ""
        self._chat_id = ""
        self._in_cron_context: ContextVar[bool] = ContextVar("cron_in_context", default=False)

    def set_context(self, channel: str, chat_id: str) -> None:
        """设置当前 Session Context，供后续创建 Job 绑定 Delivery。

        `channel` 与 `chat_id` 原样保存在 Tool Instance，直到下一次设置；方法不验证目标是否可达，也不
        修改已有 Jobs。调用栈必须在每个请求边界正确刷新，避免把提醒绑定到前一个 Session。
        """
        self._channel = channel
        self._chat_id = chat_id

    def set_cron_context(self, active: bool):
        """标记 Tool 当前是否在 Cron Job Callback 内执行。

        使用 `ContextVar` 使并发 Async Context 各自隔离；返回的 Token 必须交给 `reset_cron_context`，以
        在 Callback 结束后恢复先前状态。Active 时 `add` 会被拒绝，List/Remove 仍按正常路径处理。
        """
        return self._in_cron_context.set(active)

    def reset_cron_context(self, token) -> None:
        """使用 `ContextVar` Token 恢复 Previous Cron Context。

        应与 `set_cron_context` 成对放在 `finally` 中调用，确保异常或取消不会让当前 Task 永久停留在
        Cron Context。传入不属于该 ContextVar 的 Token 时底层异常会向上传播。
        """
        self._in_cron_context.reset(token)

    @property
    def name(self) -> str:
        return "cron"

    @property
    def description(self) -> str:
        return (
            "Schedule reminders and recurring tasks. Actions: add, list, remove.\n"
            "\n"
            "BEFORE displaying any reminder list, table, or status summary (CRITICAL):\n"
            "ALWAYS call `cron(action='list')` first to fetch the live job set.\n"
            "Do NOT reconstruct the list from conversation history — fired one-shot\n"
            "reminders, expired schedules, and reminders created in cron-triggered\n"
            "callback sessions all change the job set behind your back. Conversation\n"
            "history will show 'I scheduled X at 14:33' but a job that fired at 14:35\n"
            "is GONE from jobs.json (delete_after_run=true for ``at`` schedules).\n"
            "Without a fresh `cron.list` you will hallucinate '⏳ pending' rows for\n"
            "jobs that already completed — this confuses the user about real state.\n"
            "\n"
            "When NOT to create a recurring cron (F-K.1 — explicit ask required):\n"
            "Recurring crons (`cron_expr` or `every_seconds`) require the user to "
            "EXPLICITLY ask for a repeating schedule. Phrases like '每天提醒...', "
            "'remind me weekly', '每小时...', 'every Monday' — these imply repetition. "
            "WITHOUT such phrasing, default to `at` (one-shot) OR no cron at all.\n"
            "\n"
            "When NOT to use cron at all (F-K.3 — complaints ≠ reminders):\n"
            "Users often vent about chronic problems ('脖子又僵了', '总忘了喝水', "
            "'I keep forgetting to stretch'). This is a COMPLAINT, not a reminder "
            "request. Replying with empathy + ad-hoc action is usually correct. "
            "Do NOT create a recurring cron unless the user follows up with "
            "an explicit '请帮我设个定时提醒' / 'please set up a recurring reminder'. "
            "Otherwise the cron will fire forever, including at night, weekends, "
            "and during focus time — the very situations the user was complaining "
            "about.\n"
            "\n"
            "When to use which schedule field (CRITICAL — choose carefully):\n"
            "- `at`: ONE-TIME reminders. Use this for any 'X minutes/hours/days from now' "
            "intent. Compute the absolute target time yourself (current_time + duration) "
            "and pass it as ISO datetime. ⚠️ DO NOT use `every_seconds` to express "
            "'X minutes later' — that creates an infinite repeating reminder.\n"
            "- `cron_expr`: Fixed-time recurring patterns ('每天 7:00', 'every Monday 9 AM'). "
            "User must explicitly want a repeating schedule on a CLOCK basis.\n"
            "- `every_seconds`: Periodic interval that should repeat indefinitely "
            "(e.g. 'every hour during work', 'check every 30 min'). Rarely correct — "
            "user usually means a one-shot reminder, not infinite repetition. Default to `at` "
            "unless user explicitly asks for ongoing repetition.\n"
            "\n"
            "Anti-pattern examples to avoid:\n"
            "- ❌ '50 分钟后提醒我休息' → every_seconds=3000 (recurring forever, fires at 3am)\n"
            "- ✅ '50 分钟后提醒我休息' → at=(now+50min) (fires once)\n"
            "- ❌ '每天提醒吃药' → at=tomorrow_07:00 (fires once, never repeats)\n"
            "- ✅ '每天提醒吃药' → cron_expr='0 7 * * *' (fires daily forever)\n"
            "- ❌ '脖子总是僵' → cron_expr='*/45 * * * *' (assumes recurring without ask, "
            "fires at midnight) — instead reply with empathy + suggest one-shot stretch\n"
            "- ✅ '脖子总是僵' → no cron; sympathize + offer concrete now-action\n"
            "- ❌ '每次写作我都忘记喝水' → recurring cron (no explicit ask) — instead "
            "acknowledge + suggest user-driven habit cue\n"
            "- ✅ '帮我每小时提醒喝水' → cron_expr='0 * * * *' (explicit ask)\n"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "list", "remove"],
                    "description": "Action to perform",
                },
                "message": {"type": "string", "description": "Reminder message (for add)"},
                "every_seconds": {
                    "type": "integer",
                    "description": (
                        "Interval in seconds for INFINITELY-recurring reminders. "
                        "⚠️ DO NOT use to express 'X minutes from now' — use `at` "
                        "(one-shot) for that. Only set when user explicitly wants "
                        "ongoing periodic repetition without an end."
                    ),
                },
                "cron_expr": {
                    "type": "string",
                    "description": (
                        "Cron expression like '0 9 * * *' for clock-based recurring "
                        "schedules ('every day at 9am', 'every Monday')."
                    ),
                },
                "tz": {
                    "type": "string",
                    "description": "IANA timezone for cron expressions (e.g. 'America/Vancouver')",
                },
                "at": {
                    "type": "string",
                    "description": (
                        "ISO datetime for ONE-TIME execution (e.g. '2026-02-12T10:30:00'). "
                        "Use for 'X minutes from now' / 'tomorrow at Y' / any non-repeating "
                        "reminder. Compute absolute time = current_time + duration."
                    ),
                },
                "topic_tag": {
                    "type": "string",
                    "description": (
                        "Short snake_case identifier for the subject of this "
                        "reminder (e.g. 'birthday_zhouxiaotang', "
                        "'medication_morning', 'anniversary_8year'). "
                        "STRONGLY RECOMMENDED for any recurring or topical "
                        "reminder. cron_create dedups by topic_tag: if a cron for "
                        "'medication_morning' already exists, creating a new "
                        "one with the same topic_tag updates the existing job "
                        "instead of spawning a parallel reminder. Without "
                        "topic_tag the LLM tends to create N near-duplicate "
                        "crons across the month (different schedules, same "
                        "intent), which fires N times where 1 was wanted. "
                        "Omit only for one-off arbitrary reminders with no "
                        "risk of re-asking later."
                    ),
                },
                "job_id": {"type": "string", "description": "Job ID (for remove)"},
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        message: str = "",
        every_seconds: int | None = None,
        cron_expr: str | None = None,
        tz: str | None = None,
        at: str | None = None,
        job_id: str | None = None,
        topic_tag: str | None = None,
        **kwargs: Any,
    ) -> str:
        if action == "add":
            if self._in_cron_context.get():
                return "Error: cannot schedule new jobs from within a cron job execution"
            return self._add_job(message, every_seconds, cron_expr, tz, at, topic_tag)
        elif action == "list":
            return self._list_jobs()
        elif action == "remove":
            return self._remove_job(job_id)
        return f"Unknown action: {action}"

    def _add_job(
        self,
        message: str,
        every_seconds: int | None,
        cron_expr: str | None,
        tz: str | None,
        at: str | None,
        topic_tag: str | None = None,
    ) -> str:
        if not message:
            return "Error: message is required for add"
        if not self._channel or not self._chat_id:
            return "Error: no session context (channel/chat_id)"
        # tz 为 cron 表达式的 wall-clock 周期和无时区 `at` datetime 提供锚点；
        # 对表示相对间隔的 every schedule 没有意义，因此忽略而不报错，
        # 避免 Agent 无谓重试。
        if tz and (cron_expr or at):
            from zoneinfo import ZoneInfo

            try:
                ZoneInfo(tz)
            except Exception:
                return f"Error: unknown timezone '{tz}'"

        # 构建 schedule。
        delete_after = False
        if every_seconds:
            schedule = CronSchedule(kind="every", every_ms=every_seconds * 1000)
        elif cron_expr:
            schedule = CronSchedule(kind="cron", expr=cron_expr, tz=tz)
        elif at:
            from datetime import datetime

            try:
                dt = datetime.fromisoformat(at)
            except ValueError:
                return f"Error: invalid ISO datetime format '{at}'. Expected format: YYYY-MM-DDTHH:MM:SS"
            # 无时区 `at` + tz 表示“tz 中的该 wall-clock 时间”；为它附加时区，
            # 避免 .timestamp() 静默回退到 Host 本地时区。带 offset 的字符串
            # 已自带时区，因此忽略 tz。
            if dt.tzinfo is None and tz:
                from zoneinfo import ZoneInfo

                dt = dt.replace(tzinfo=ZoneInfo(tz))
            at_ms = int(dt.timestamp() * 1000)
            schedule = CronSchedule(kind="at", at_ms=at_ms)
            delete_after = True
        else:
            return "Error: either every_seconds, cron_expr, or at is required"

        # 原样保存请求时的 channel/chat_id。投递解析（按任务直通或临时转发）
        # 在触发时由 ``codeflow.cli._cron_handler`` 通过 ``resolve_cron_delivery`` 完成。
        try:
            job = self._cron.add_job(
                name=message[:30],
                schedule=schedule,
                message=message,
                deliver=True,
                channel=self._channel,
                to=self._chat_id,
                delete_after_run=delete_after,
                topic_tag=topic_tag,
            )
        except ValueError as exc:
            # service 会拒绝不可运行的 schedule（过去的 at、every_seconds <= 0、
            # 无效 cron expr），而不是保存成永不触发的任务。把错误暴露给 Agent，
            # 使其可以重试。
            return f"Error: {exc}"
        return f"Created job '{job.name}' (id: {job.id})"

    def _list_jobs(self) -> str:
        jobs = self._cron.list_jobs()
        if not jobs:
            return "No scheduled jobs."
        lines = [f"- {j.name} (id: {j.id}, {j.schedule.kind})" for j in jobs]
        return "Scheduled jobs:\n" + "\n".join(lines)

    def _remove_job(self, job_id: str | None) -> str:
        if not job_id:
            return "Error: job_id is required for remove"
        if self._cron.remove_job(job_id):
            return f"Removed job {job_id}"
        return f"Job {job_id} not found"
