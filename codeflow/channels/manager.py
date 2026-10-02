"""协调 Chat Channels 的 Construction 与 Lifecycle Manager。

Manager 只负责 Discover/Construct Enabled Adapters、Start/Quiesce/Stop 与 Status。Outbound Delivery 属于
Spine `DeliveryHub`/`Outlet`，Gateway 为每个 Channel 注册 `ChannelOutletAdapter`；Inbound 数据流是各
Channel ``Intake -> scheduler.submit``。

单 Channel Dependency/Factory Failure 只禁用该 Channel，不拖垮 Gateway。Manager Start Return、Channel
Running、Turn Completion 与 Outbound Delivered 必须分别观察。
"""

from __future__ import annotations

import asyncio
import json
import sys
from importlib.metadata import PackageNotFoundError, distribution
from typing import Any

from loguru import logger

from codeflow.channels.contract import Channel
from codeflow.config.schema import Config
from codeflow.product import DISTRIBUTION_NAME
from codeflow.spine._barrier import finish_barrier

_INTAKE_DRAIN_TIMEOUT_S = 5.0
_CHANNEL_STOP_TIMEOUT_S = 5.0


def _consume_task_result(task: asyncio.Task[Any]) -> None:
    if task.cancelled():
        return
    task.exception()


def _missing_dep_hint(modname: str) -> str:
    """根据 Install Mode 返回 Channel Missing SDK 的安装提示。

    Editable Dev Checkout 使用 ``uv sync``；Wheel/Tool Install 无 Source Tree，必须 Re-run Installer。PEP 610
    ``direct_url.json`` 区分两者；Wheel 记录 ``archive_info`` 且无 ``dir_info``，因此用 ``.get`` Chain 避免
    KeyError。该函数运行在 Channel 已失败的路径，Missing/Corrupt Metadata 必须降级 Installer Hint，Never
    Raise。
    """
    editable = False
    try:
        raw = distribution(DISTRIBUTION_NAME).read_text("direct_url.json")
        if raw:
            editable = bool(json.loads(raw).get("dir_info", {}).get("editable", False))
    except (PackageNotFoundError, ValueError):
        pass

    if editable:
        return f"Run: uv sync --extra channel-{modname}"
    if sys.platform == "win32":
        return (
            "Re-run the CodeFlow installer to add channels: "
            '$h = @{Authorization="Bearer $env:CODEFLOW_GITHUB_TOKEN"}; '
            "irm https://github.com/pei711/codeflow-harness/raw/main/install.ps1 -Headers $h | iex"
        )
    return (
        "Re-run the CodeFlow installer to add channels: "
        "printf 'Authorization: Bearer %s\\n' \"$CODEFLOW_GITHUB_TOKEN\" | "
        "curl -fsSL -H @- https://github.com/pei711/codeflow-harness/raw/main/install.sh | sh"
    )


class ChannelManager:
    """管理 Chat Channels：Construct Enabled Adapters、Start/Stop、Quiesce 与 Status。

    构造时立即 Discover Specs、创建可用 Adapter 并验证 Allowlist；Runtime 生命周期由 Gateway 调用
    `start_all`、Shutdown 先 `quiesce_intake` 再 `stop_all`。`channels` Dict 是 Active Constructed Set，不含
    Disabled/Failed Specs。
    """

    def __init__(self, config: Config):
        self.config = config
        self.channels: dict[str, Channel] = {}

        self._init_channels()

    def _init_channels(self) -> None:
        """从 Declarative ``ChannelSpec`` 初始化 Enabled Channels。

        每个 Spec 读取 Matching Config Section，Factory Success 后注入 Transcription Key。Missing SDK、Factory
        Crash、Deny-all Allowlist 都只 Disable One Channel，Never Propagate，使其他 Channels/Gateway 仍启动。
        Factory Construction Success 尚未调用 Channel ``start``。
        """
        from codeflow.channels.registry import discover_specs

        groq_key = self.config.providers.groq.api_key

        for modname, spec in discover_specs().items():
            section = getattr(self.config.channels, modname, None)
            if not section or not getattr(section, "enabled", False):
                continue
            try:
                channel = spec.factory(section)
                channel.transcription_api_key = groq_key
                self.channels[modname] = channel
                logger.info("{} channel enabled", spec.display_name)
            except ImportError as e:
                logger.warning(
                    "{} channel disabled: missing dependency ({}). {}",
                    modname,
                    e,
                    _missing_dep_hint(modname),
                )
            except Exception as e:
                logger.error(
                    "{} channel disabled: adapter failed to construct ({}: {})",
                    modname,
                    type(e).__name__,
                    e,
                )

        self._validate_allow_from()

    def _validate_allow_from(self) -> None:
        """Disable Allowlist Denies Everyone 的 Channel。

        Empty ``allow_from`` 是会 Silently Drop 每条 Inbound 的 Misconfiguration，因此 Loudly 从 Active Set
        移除，避免 Running Black Hole；同时不 Abort Process，防止一条 Channel Config Error 拖垮其他 Channel。
        """
        for name in [n for n, ch in self.channels.items() if getattr(ch.config, "allow_from", None) == []]:
            logger.error(
                '{} channel disabled: empty allowFrom denies every sender. Set ["*"] to allow everyone, '
                "or add specific user IDs.",
                name,
            )
            del self.channels[name]

    async def _start_channel(self, name: str, channel: Channel) -> None:
        """Start Single Channel，并 Log/Swallow Exception。

        该隔离让 `start_all` 继续尝试其他 Adapters；失败 Channel 仍留在 Dict，但 Running State 应为 False。
        """
        try:
            await channel.start()
        except Exception as e:
            logger.error("Failed to start channel {}: {}", name, e)

    async def start_all(self) -> None:
        """并发 Start 所有 Constructed Channels，它们通常 Long-running。

        无 Channel 时 Warning + Return。Tasks Gather with Return Exceptions，单项 Start 已自行 Log。Outbound
        Delivery 属于 Spine Outlets，不是 Manager；方法返回只表示 Start Coroutines 已返回/失败。
        """
        if not self.channels:
            logger.warning("No channels enabled")
            return

        tasks = []
        for name, channel in self.channels.items():
            logger.info("Starting {} channel...", name)
            tasks.append(asyncio.create_task(self._start_channel(name, channel)))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def quiesce_intake(self) -> None:
        """Seal 每个 Intake，并为 Admitted Publishes 的 Drain 设置严格 Bound。

        先拒绝 New Inbound，再 Shield Wait-idle Task。5s Timeout 或 Caller Cancellation 时 Cancel 所有 Inflight
        Publishes，并通过 `finish_barrier` 等待清理完成后再 Raise；保证 Spine Teardown 前不残留新提交。
        正常返回表示 Intake Idle，不表示已提交 Turn Delivered。
        """
        intakes = [channel.intake for channel in self.channels.values()]
        for intake in intakes:
            intake.seal()

        async def wait_idle() -> None:
            await asyncio.gather(*(intake.wait_idle() for intake in intakes))

        # 调用方取消时不得同时取消屏障等待者：先取消已接纳的 publish，
        # 并确认系统进入 idle，之后才向上传播取消。
        drain = asyncio.create_task(wait_idle())
        try:
            await asyncio.wait_for(asyncio.shield(drain), timeout=_INTAKE_DRAIN_TIMEOUT_S)
        except TimeoutError as exc:
            cancelled = sum(intake.cancel_inflight() for intake in intakes)
            logger.error(
                "channel intake drain timed out after {}s; cancelling {} admitted publish(es)",
                _INTAKE_DRAIN_TIMEOUT_S,
                cancelled,
            )
            await finish_barrier(drain)
            raise TimeoutError(
                f"channel intake drain timed out after {_INTAKE_DRAIN_TIMEOUT_S}s; "
                f"cancelled {cancelled} admitted publish(es)"
            ) from exc
        except asyncio.CancelledError as cancellation:
            cancelled = sum(intake.cancel_inflight() for intake in intakes)
            logger.warning(
                "channel intake drain cancelled; cancelling {} admitted publish(es) before spine teardown",
                cancelled,
            )
            await finish_barrier(drain, cancellation=cancellation)
            raise cancellation

    async def _stop_channel(self, name: str, channel: Channel) -> None:
        # 若内部协程吞掉取消，wait_for 可能超过自身超时；独立任务用于严格
        # 执行该截止时间。
        task = asyncio.create_task(channel.stop())
        try:
            done, _pending = await asyncio.wait({task}, timeout=_CHANNEL_STOP_TIMEOUT_S)
        except asyncio.CancelledError:
            task.cancel()
            task.add_done_callback(_consume_task_result)
            raise
        if not done:
            task.cancel()
            task.add_done_callback(_consume_task_result)
            raise TimeoutError(f"channel {name} stop timed out after {_CHANNEL_STOP_TIMEOUT_S}s")
        await task

    async def stop_all(self) -> None:
        """对每个 Channel Transport 执行 Bounded Stop，并尝试全部项。

        每项最多 5s；即使一个失败仍继续 Stop 其他 Channels。结束后 Cancellation 优先 Re-raise，其次 First
        Error；全部成功才正常返回。Stop Return 表示 Transport Cleanup Attempt 完成，不删除 Config。
        """
        logger.info("Stopping all channels...")
        first_error: BaseException | None = None
        cancellation: asyncio.CancelledError | None = None
        for name, channel in self.channels.items():
            try:
                await self._stop_channel(name, channel)
                logger.info("Stopped {} channel", name)
            except asyncio.CancelledError as exc:
                logger.opt(exception=exc).error("Error stopping {}", name)
                if cancellation is None:
                    cancellation = exc
            except BaseException as exc:
                logger.opt(exception=exc).error("Error stopping {}", name)
                if first_error is None:
                    first_error = exc
        if cancellation is not None:
            raise cancellation
        if first_error is not None:
            raise first_error

    def get_channel(self, name: str) -> Channel | None:
        """按 Registry Name 返回 Active Constructed Channel；Absent/Disabled/Failed 时为 `None`。"""
        return self.channels.get(name)

    def get_status(self) -> dict[str, Any]:
        """返回所有 Active Channels 的 Enabled/Running Status Snapshot。

        Status 不探测 Platform Connection 或最近 Delivery，仅读取 Adapter ``is_running``。
        """
        return {name: {"enabled": True, "running": channel.is_running} for name, channel in self.channels.items()}

    @property
    def enabled_channels(self) -> list[str]:
        """返回 Active Constructed Channel Names List。

        这里的 Enabled 表示通过 Config/Construction/Allowlist Admission，不保证 Currently Running。
        """
        return list(self.channels)
