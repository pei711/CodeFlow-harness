"""Shared ``on_cron_job`` factory used by both ``gateway`` and ``agent``.

Extracted from commands.gateway() so both entry points run cron jobs with
identical semantics: a scheduled reminder fires as a CRON-origin spine turn
bound to the ``cron:<job_id>`` session, and the reply is delivered by the hub
(single target) or broadcast to the resolved targets.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Awaitable, Callable

from loguru import logger

if TYPE_CHECKING:
    from codeflow.channels.manager import ChannelManager
    from codeflow.proactive_engine.schedulers.cron.types import CronJob
    from codeflow.session.manager import SessionManager
    from codeflow.spine import TurnHandle, TurnRequest
    from codeflow.spine.delivery import DeliveryHub


def _ms_to_local_str(ms: int | None) -> str | None:
    """Render a ms-since-epoch timestamp as local HH:MM for user-facing text."""
    if not ms:
        return None
    try:
        return datetime.fromtimestamp(ms / 1000).strftime("%H:%M")
    except (OSError, ValueError):
        return None


def _format_schedule_origin(job: "CronJob") -> str:
    """Describe when the reminder was originally set, for the user.

    - 'at' jobs: "set at <HH:MM>, scheduled for <HH:MM>" (at_ms is the fire time)
    - 'every' jobs: "set at <HH:MM>, recurring every <N>s"
    - 'cron' jobs: "set at <HH:MM>, cron <expr>"
    """
    created = _ms_to_local_str(job.created_at_ms) or "?"
    kind = job.schedule.kind
    if kind == "at":
        fire_at = _ms_to_local_str(job.schedule.at_ms) or "?"
        return f"set at {created}, scheduled for {fire_at}"
    if kind == "every":
        secs = (job.schedule.every_ms or 0) // 1000
        return f"set at {created}, recurring every {secs}s"
    if kind == "cron" and job.schedule.expr:
        return f"set at {created}, cron `{job.schedule.expr}`"
    return f"set at {created}"


def make_on_cron_job(
    hub: "DeliveryHub",
    *,
    submit: "Callable[[TurnRequest], TurnHandle]",
    readback_texts: "dict[str, str] | None" = None,
    channel_manager: "ChannelManager | None" = None,
    session_manager: "SessionManager | None" = None,
    default_channel: str = "cli",
) -> Callable[["CronJob"], Awaitable[str | None]]:
    """Build the CronService.on_job callback. Every cron turn runs through the
    spine ``submit`` as a CRON-origin turn.

    ``submit`` (required) is the spine entry (build_gateway / build_repl /
    build_tui scheduler). A single-target delivering job (deliver=True, the
    user-facing reminder the cron tool creates) rides the hub to its one outlet.
    A broadcast (more than one resolved target) or a silent job (deliver=False)
    submits with the job's own (ephemeral) channel as the source so the hub drops
    the reply; a broadcast is then delivered explicitly to every target, a silent
    job delivers nothing.

    ``readback_texts`` is the host's per-conversation reply-text map. A CRON
    turn submits, then this reads its reply from
    ``readback_texts[cron:<job_id>]`` and pops it. The gateway needs the reply
    for multi-target delivery and the TUI needs it for ``cron.delivered``.

    ``default_channel`` is used when the job payload doesn't specify one —
    REPL passes "cli" so the reminder renders inline in the terminal; the
    gateway lets the payload's own channel decide.

    ``channel_manager`` / ``session_manager`` drive delivery resolution
    (``resolve_cron_delivery``). Without ``channel_manager`` the enabled
    set is empty — meaning every channel is considered ephemeral, and
    delivery falls back to forward_channels broadcast. REPL paths that
    have no real channels (agent-only mode) pass ``None`` for both and
    accept that ephemeral reminders are dropped with a warning.

    Outcomes and failures flow only through the Spine handle and CronService
    state. The callback has no additional scheduling side effects.
    """

    async def on_cron_job(job: "CronJob") -> str | None:
        from codeflow.config.loader import load_config
        from codeflow.proactive_engine.schedulers.cron.tool import (
            is_ephemeral_channel,
            resolve_cron_delivery,
        )
        from codeflow.spine import ChatType, Origin, Source, Text, TurnRequest

        # 包含最初调度时间，使提醒文本能向用户回显“设置于 17:05”；否则智能体只知道当前时间。
        reminder_note = (
            "[Scheduled Task] Timer finished.\n\n"
            f"Task '{job.name}' ({_format_schedule_origin(job)}) "
            "has been triggered.\n"
            f"Scheduled instruction: {job.payload.message}\n\n"
            "When you reply, mention when the reminder was originally set "
            '(e.g. "你在 17:05 提醒的 ...") so the user remembers the '
            "context."
        )

        # 在触发时解析投递目标（此时读取 cron 配置，使 ``cron config set`` 在下一次触发时生效）。
        # 该步骤与响应无关，可在轮次前执行；路径由 len(targets) 决定。
        cron_cfg = load_config().cron
        enabled_channels = set(channel_manager.enabled_channels) if channel_manager is not None else set()
        targets, warnings = resolve_cron_delivery(
            channel=job.payload.channel or default_channel,
            chat_id=job.payload.to or "direct",
            forward_channels=cron_cfg.forward_channels,
            enabled_channels=enabled_channels,
            session_manager=session_manager,
        )
        for w in warnings:
            logger.warning("Cron job '{}' ({}): {}", job.name, job.id, w)

        # 每个 cron 轮次都通过 spine 运行。各分支显式投递：单目标投递任务经 hub 到唯一出口；
        # 其他情况（不转发、广播或静默任务）以任务自身渠道为来源提交。真实场景下该来源是临时的，
        # 没有网关出口，hub 会丢弃回复；广播随后在下方显式投递，静默任务则不投递。run_turn
        # 在通道任务中自行设置以 origin=CRON 为键的 cron 上下文防护。
        deliver_via_hub = job.payload.deliver and len(targets) == 1
        if deliver_via_hub:
            src_channel, src_chat = targets[0].channel, targets[0].chat_id
        else:
            src_channel = job.payload.channel or default_channel
            src_chat = job.payload.to or "direct"
            # 静默任务（deliver=False）因临时来源没有网关出口而保持静默，hub 会丢弃回复。
            # 非临时渠道上的静默任务会被 hub 投递；由于所有创建路径都设置 deliver=True，
            # 只能手工编辑 jobs.json 才能到达此情况。发出警告，使该边界可见而非静默变化。
            if not job.payload.deliver and not is_ephemeral_channel(src_channel, enabled_channels):
                logger.warning(
                    "Cron job '{}' ({}): silent job on non-ephemeral channel '{}' "
                    "is delivered under the spine (no outlet-less suppression for "
                    "real channels)",
                    job.name,
                    job.id,
                    src_channel,
                )
        req = TurnRequest(
            origin=Origin.CRON,
            source=Source(channel=src_channel, chat_id=src_chat, sender_id="cron", chat_type=ChatType.DM),
            text=reminder_note,
            conversation=f"cron:{job.id}",
        )
        await submit(req).result()
        # 读回回复以执行主机特定投递，并弹出记录，避免长期运行的映射不断积累。
        response: str | None = readback_texts.pop(f"cron:{job.id}", None) if readback_texts is not None else None

        # 将多目标投递任务广播到所有已解析目标：hub 已丢弃回复（临时来源没有出口），因此这是
        # 唯一投递。message 工具自发消息时也不跳过；守护进程中发往临时来源的自发消息永远
        # 到不了用户，所以把回复广播到全部目标既更简单，也严格优于旧有自发消息防护——后者会
        # 抑制广播，使其他目标什么都收不到。
        if job.payload.deliver and len(targets) > 1 and response:
            for t in targets:
                await hub.post(
                    Text(
                        content=response,
                        source=Source(
                            channel=t.channel,
                            chat_id=t.chat_id,
                            sender_id="cron",
                            chat_type=ChatType.DM,
                        ),
                    )
                )
        return response

    return on_cron_job


__all__ = ["make_on_cron_job"]
