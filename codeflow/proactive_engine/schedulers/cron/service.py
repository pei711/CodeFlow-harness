"""负责 Scheduling Agent Tasks 的 Persistent Cron Service。

Service 管理 ``jobs.json`` Store、计算 At/Every/Cron 下一次触发时间、在多个 Process 间 Claim 到期
Job，并在锁外执行 `on_job` Callback。它用原子 Replace 保存 Store、Sidecar Advisory Lock 协调并发、
带 TTL 的 Claim 防止同一任务被活跃 Peer 重复执行。

Cron Trigger 只证明任务到达计划时间并被交给 Callback；实际 Agent Turn、Channel Delivery 与用户是否
收到结果属于下游证据。损坏或不兼容记录会保留成 Disabled Placeholder，避免加载时静默丢数据。
"""

import asyncio
import json
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Coroutine, Iterator

from loguru import logger

from codeflow.proactive_engine.schedulers.cron.types import CronJob, CronJobState, CronPayload, CronSchedule, CronStore
from codeflow.utils.portable_lock import file_lock

# 过期 claim 的 TTL：超过该时长后其他进程可以接管，原进程很可能在任务中途崩溃。
_CLAIM_TTL_MS = 30 * 60 * 1000

# 限制下一次唤醒前的最长睡眠时间，保证 run_due 至少按此频率运行。
# run_due 会在 mtime 变化时重载 store，从而发现同级进程写入 jobs.json 的任务。
# 没有该上限时，等待遥远唤醒时间的 gateway 会错过 REPL 新增的更早任务。
_MAX_WAKE_INTERVAL_S = 30.0


def _expect_bool(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field} must be a boolean")
    return value


def _expect_int(value: Any, field: str, *, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if type(value) is not int:
        suffix = " or null" if allow_none else ""
        raise TypeError(f"{field} must be an integer{suffix}")
    return value


def _expect_str(value: Any, field: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str):
        suffix = " or null" if allow_none else ""
        raise TypeError(f"{field} must be a string{suffix}")
    return value


def _compute_next_run(schedule: CronSchedule, now_ms: int) -> int | None:
    """根据 ``schedule`` 与参考 ``now_ms`` 计算 Next Run Time，单位为 Milliseconds。

    `at` 仅在时间仍位于未来时返回固定 Timestamp；`every` 从当前参考时间向后加正的 Interval；`cron`
    使用 `croniter` 与可选 `ZoneInfo` 计算下一次日历触发。字段缺失、表达式/时区错误或没有未来触发时
    返回 `None`，调用方必须把它视为 Non-runnable，而不是保存一个永不执行的 Job。
    """
    if schedule.kind == "at":
        return schedule.at_ms if schedule.at_ms and schedule.at_ms > now_ms else None

    if schedule.kind == "every":
        if not schedule.every_ms or schedule.every_ms <= 0:
            return None
        return now_ms + schedule.every_ms

    if schedule.kind == "cron" and schedule.expr:
        try:
            from zoneinfo import ZoneInfo

            from croniter import croniter

            # 使用调用方提供的参考时间，确保调度结果确定。
            base_time = now_ms / 1000
            tz = ZoneInfo(schedule.tz) if schedule.tz else datetime.now().astimezone().tzinfo
            base_dt = datetime.fromtimestamp(base_time, tz=tz)
            cron = croniter(schedule.expr, base_dt)
            next_dt = cron.get_next(datetime)
            return int(next_dt.timestamp() * 1000)
        except Exception:
            return None

    return None


def _validate_schedule_for_add(schedule: CronSchedule, now_ms: int) -> None:
    """验证会导致 Non-runnable Jobs 的 Schedule Fields，并在 Add 前拒绝。

    Timezone 只允许用于 Cron Schedule，未知 Zone 会抛出 `ValueError`。随后以 `_compute_next_run` 作为
    “可运行”的唯一事实来源：过去的 `at`、非正 `every`、非法 Cron Expression 或未知 Kind 都返回
    对应可诊断错误。验证与最终 Add 应共享同一个 `now_ms` Snapshot，避免临界时间先通过后失效。
    """
    if schedule.tz and schedule.kind != "cron":
        raise ValueError("tz can only be used with cron schedules")

    if schedule.kind == "cron" and schedule.tz:
        try:
            from zoneinfo import ZoneInfo

            ZoneInfo(schedule.tz)
        except Exception:
            raise ValueError(f"unknown timezone '{schedule.tz}'") from None

        # 没有下次运行时间的 schedule 若被保存，会成为永不触发的任务，向调用方
        # 造成假成功。_compute_next_run 是“可运行”的唯一事实来源，因此此处拒绝
        # 任何被它映射为 None 的类型。
    if _compute_next_run(schedule, now_ms) is None:
        if schedule.kind == "at":
            raise ValueError("at time is in the past")
        if schedule.kind == "every":
            raise ValueError("every_seconds must be positive")
        if schedule.kind == "cron":
            raise ValueError(f"invalid cron expression '{schedule.expr}'")
        raise ValueError(f"schedule kind '{schedule.kind}' is not runnable")


class CronService:
    """管理、持久化并执行 Scheduled Jobs 的 Service。

    生命周期从 `start` 加载并恢复 Store、重算 Next Runs、Arm Timer 开始，到 `stop` 取消 Timer 结束。
    Public API 支持 Add/List/Enable/Remove/Manual Run，以及 Silent-fire 衰减。多实例可共享同一 Store，
    通过锁和 Claim Ownership 避免并发覆盖与重复执行；Callback 本身在锁外运行，不能长期阻塞其他
    Scheduler 修改任务。
    """

    def __init__(
        self,
        store_path: Path,
        on_job: Callable[[CronJob], Coroutine[Any, Any, str | None]] | None = None,
        *,
        allowed_channels: set[str] | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ):
        """创建 Service，并用 ``allowed_channels`` 限制它可 Claim 的 Jobs。

        例如 REPL Mode 设置 ``{"cli"}``，可避免 REPL 抢走应由 Gateway 交付的 Feishu / Telegram
        Reminders。默认 `None` 表示 Any Channel，适合 `ChannelManager` 能路由到所有已配置 Channel 的
        Gateway。``payload.channel`` 为空或 `None` 的旧 Jobs 始终可 Claim，因为它们早于 Channel
        Attribution Field。

        `store_path` 指向共享 JSON Store，`on_job` 是到期时执行的 Async Callback，`now_fn` 可为
        Benchmark Harness 注入 Fake Clock。构造函数只准备状态并启动 Writer-independent Fields，不会
        自动加载、执行或 Arm Timer，调用方仍需 `start`。
        """
        self.store_path = store_path
        # 相邻文件用于 advisory lock；数据文件原子重命名后它仍存在，
        # 让并发进程协调 run_due。
        self.lock_path = store_path.with_suffix(store_path.suffix + ".lock")
        self.on_job = on_job
        self.allowed_channels = allowed_channels
        self._store: CronStore | None = None
        self._incompatible_records: dict[int, Any] = {}
        self._store_metadata: dict[str, Any] = {}
        # 使用纳秒精度：浮点 st_mtime 会把间隔约 238 纳秒的写入合并
        # 合并成一个值，避免外部重写后仍提供过期缓存。
        self._last_mtime: int = 0
        self._timer_task: asyncio.Task | None = None
        self._running = False
        # 为 benchmark harness（longrun）提供可选的伪时钟注入。设置后，内部
        # 所有时间读取都经过该 callable，使新任务的 next_run_at_ms 对齐模拟时间，
        # 而不是真实 wall-clock。
        self._now_fn = now_fn

    def _now_ms(self) -> int:
        """返回 Current Time 的 Milliseconds，并遵守 Fake-clock Injection。

        提供 `now_fn` 时使用其 Timestamp，使新 Job 与 Longrun Benchmark 的 Simulated Time 对齐；否则
        使用真实 `time.time()`。这是 Service 内所有调度计算的统一时钟来源。
        """
        if self._now_fn is not None:
            return int(self._now_fn().timestamp() * 1000)
        return int(time.time() * 1000)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """在 Context Block 内持有 Jobs File Sibling 上的 Cross-platform Lock。

        Lock Anchor 独立于会被原子 Replace 的 Data File，因此 Store Rename 后不同进程仍协调同一把锁。
        方法只提供互斥范围，不自动 Reload 或 Save Store。
        """
        with file_lock(self.lock_path):
            yield

    def _load_store(self) -> CronStore:
        """从 Disk 加载 Jobs；File 被 External Modification 后自动 Reload。

        已缓存 Store 会比较 Nanosecond Mtime，变化时丢弃内存 Snapshot。加载时验证顶层结构、Store
        Version 与每个 Job Field；单条 Incompatible Record 转成 Disabled Placeholder，并保留 Raw
        Record 以便再次 Save 时原样写回。整个文件无法解析则记录 Warning 并退回 Empty Store。

        该方法本身不获取 Cross-process Lock，Mutation Path 必须在 `_locked` 内先强制清空 Cache 再
        调用，避免覆盖 Peer Writes。
        """
        if self._store and self.store_path.exists():
            mtime = self.store_path.stat().st_mtime_ns
            if mtime != self._last_mtime:
                logger.info("Cron: jobs.json modified externally, reloading")
                self._store = None
        if self._store:
            return self._store

        if self.store_path.exists():
            try:
                data = json.loads(self.store_path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise TypeError("cron store must be an object")
                raw_jobs = data.get("jobs", [])
                if not isinstance(raw_jobs, list):
                    raise TypeError("cron store jobs must be an array")
                store_version = _expect_int(data.get("version", 1), "store.version")
                self._incompatible_records = {}
                self._store_metadata = {key: value for key, value in data.items() if key not in {"version", "jobs"}}
                jobs = []
                for index, raw_job in enumerate(raw_jobs):
                    try:
                        if store_version != 1:
                            raise ValueError(f"unsupported store version '{store_version}'")
                        jobs.append(self._deserialize_job(raw_job))
                    except (KeyError, TypeError, ValueError) as error:
                        logger.warning("Cron: incompatible job at index {}: {}", index, error)
                        incompatible = self._incompatible_job(raw_job, index, str(error))
                        self._incompatible_records[id(incompatible)] = raw_job
                        jobs.append(incompatible)
                self._store = CronStore(version=store_version, jobs=jobs)
            except Exception as e:
                logger.warning("Failed to load cron store: {}", e)
                self._incompatible_records = {}
                self._store_metadata = {}
                self._store = CronStore()
        else:
            self._incompatible_records = {}
            self._store_metadata = {}
            self._store = CronStore()

        return self._store

    @staticmethod
    def _deserialize_job(raw_job: Any) -> CronJob:
        if not isinstance(raw_job, dict):
            raise TypeError("job record must be an object")
        job_id = _expect_str(raw_job["id"], "job.id")
        name = _expect_str(raw_job["name"], "job.name")
        enabled = _expect_bool(raw_job.get("enabled", True), "job.enabled")
        schedule_data = raw_job["schedule"]
        payload_data = raw_job["payload"]
        if not isinstance(schedule_data, dict):
            raise TypeError("schedule must be an object")
        if not isinstance(payload_data, dict):
            raise TypeError("payload must be an object")
        schedule_kind = _expect_str(schedule_data["kind"], "schedule.kind")
        if schedule_kind not in {"at", "every", "cron"}:
            raise ValueError(f"unsupported schedule kind '{schedule_kind}'")
        at_ms = _expect_int(schedule_data.get("atMs"), "schedule.atMs", allow_none=True)
        every_ms = _expect_int(schedule_data.get("everyMs"), "schedule.everyMs", allow_none=True)
        expr = _expect_str(schedule_data.get("expr"), "schedule.expr", allow_none=True)
        tz = _expect_str(schedule_data.get("tz"), "schedule.tz", allow_none=True)
        payload_kind = _expect_str(payload_data.get("kind", "agent_turn"), "payload.kind")
        if payload_kind not in {"agent_turn", "system_event"}:
            raise ValueError(f"unsupported payload kind '{payload_kind}'")
        message = _expect_str(payload_data.get("message", ""), "payload.message")
        deliver = _expect_bool(payload_data.get("deliver", False), "payload.deliver")
        channel = _expect_str(payload_data.get("channel"), "payload.channel", allow_none=True)
        to = _expect_str(payload_data.get("to"), "payload.to", allow_none=True)
        topic_tag = _expect_str(payload_data.get("topicTag"), "payload.topicTag", allow_none=True)
        state_data = raw_job.get("state", {})
        if not isinstance(state_data, dict):
            raise TypeError("state must be an object")
        next_run_at_ms = _expect_int(state_data.get("nextRunAtMs"), "state.nextRunAtMs", allow_none=True)
        last_run_at_ms = _expect_int(state_data.get("lastRunAtMs"), "state.lastRunAtMs", allow_none=True)
        last_status = _expect_str(state_data.get("lastStatus"), "state.lastStatus", allow_none=True)
        if last_status not in {None, "ok", "error", "skipped", "expired", "cancelled", "incompatible"}:
            raise ValueError(f"unsupported job status '{last_status}'")
        last_error = _expect_str(state_data.get("lastError"), "state.lastError", allow_none=True)
        claimed_by_pid = _expect_int(state_data.get("claimedByPid"), "state.claimedByPid", allow_none=True)
        claimed_at_ms = _expect_int(state_data.get("claimedAtMs"), "state.claimedAtMs", allow_none=True)
        silent_fire_count = _expect_int(state_data.get("silentFireCount", 0), "state.silentFireCount")
        created_at_ms = _expect_int(raw_job.get("createdAtMs", 0), "job.createdAtMs")
        updated_at_ms = _expect_int(raw_job.get("updatedAtMs", 0), "job.updatedAtMs")
        delete_after_run = _expect_bool(raw_job.get("deleteAfterRun", False), "job.deleteAfterRun")
        silent_fire_limit = _expect_int(raw_job.get("silentFireLimit", 12), "job.silentFireLimit", allow_none=True)
        return CronJob(
            id=job_id,
            name=name,
            enabled=enabled,
            schedule=CronSchedule(
                kind=schedule_kind,
                at_ms=at_ms,
                every_ms=every_ms,
                expr=expr,
                tz=tz,
            ),
            payload=CronPayload(
                kind=payload_kind,
                message=message,
                deliver=deliver,
                channel=channel,
                to=to,
                topic_tag=topic_tag,
            ),
            state=CronJobState(
                next_run_at_ms=next_run_at_ms,
                last_run_at_ms=last_run_at_ms,
                last_status=last_status,
                last_error=last_error,
                claimed_by_pid=claimed_by_pid,
                claimed_at_ms=claimed_at_ms,
                silent_fire_count=silent_fire_count,
            ),
            created_at_ms=created_at_ms,
            updated_at_ms=updated_at_ms,
            delete_after_run=delete_after_run,
            silent_fire_limit=silent_fire_limit,
        )

    @staticmethod
    def _incompatible_job(raw_job: Any, index: int, error: str) -> CronJob:
        record = raw_job if isinstance(raw_job, dict) else {}
        payload = record.get("payload")
        payload_message = payload.get("message", "") if isinstance(payload, dict) else ""
        job_id = record.get("id")
        name = record.get("name")
        return CronJob(
            id=job_id if isinstance(job_id, str) and job_id else f"incompatible-{index}",
            name=name if isinstance(name, str) and name else "Incompatible cron job",
            enabled=False,
            schedule=CronSchedule(kind="at"),
            payload=CronPayload(message=payload_message if isinstance(payload_message, str) else ""),
            state=CronJobState(last_status="incompatible", last_error=error),
            created_at_ms=record.get("createdAtMs", 0),
            updated_at_ms=record.get("updatedAtMs", 0),
            delete_after_run=False,
        )

    def _save_store(self) -> None:
        """把当前 In-memory Jobs Store 原子保存到 Disk。

        方法保留未知 Top-level Metadata，并让 Incompatible Records 使用原 Raw Payload；正常 Job 则按
        Version 1 Schema 序列化。数据先写到同目录 Temp File，再用 `os.replace` 发布，避免 Reader
        看到 Partial JSON；成功后更新 Nanosecond Mtime Cache。Caller 必须在需要并发安全的路径持有
        `_locked`。
        """
        if not self._store:
            return

        self.store_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            **self._store_metadata,
            "version": self._store.version,
            "jobs": [self._serialize_job(job) for job in self._store.jobs],
        }

        # 通过临时文件 + rename 原子写入，避免并发读取方看到只刷新了一部分的文件。
        tmp = self.store_path.with_suffix(self.store_path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.store_path)
        self._last_mtime = self.store_path.stat().st_mtime_ns

    def _serialize_job(self, job: CronJob) -> Any:
        if id(job) in self._incompatible_records:
            return self._incompatible_records[id(job)]
        return {
            "id": job.id,
            "name": job.name,
            "enabled": job.enabled,
            "schedule": {
                "kind": job.schedule.kind,
                "atMs": job.schedule.at_ms,
                "everyMs": job.schedule.every_ms,
                "expr": job.schedule.expr,
                "tz": job.schedule.tz,
            },
            "payload": {
                "kind": job.payload.kind,
                "message": job.payload.message,
                "deliver": job.payload.deliver,
                "channel": job.payload.channel,
                "to": job.payload.to,
                "topicTag": job.payload.topic_tag,
            },
            "state": {
                "nextRunAtMs": job.state.next_run_at_ms,
                "lastRunAtMs": job.state.last_run_at_ms,
                "lastStatus": job.state.last_status,
                "lastError": job.state.last_error,
                "claimedByPid": job.state.claimed_by_pid,
                "claimedAtMs": job.state.claimed_at_ms,
                "silentFireCount": job.state.silent_fire_count,
            },
            "createdAtMs": job.created_at_ms,
            "updatedAtMs": job.updated_at_ms,
            "deleteAfterRun": job.delete_after_run,
            "silentFireLimit": job.silent_fire_limit,
        }

    async def start(self) -> None:
        """启动 Cron Service 并恢复 Persistent Schedule State。

        设置 Running Flag，加载 Store，重算所有 Enabled Jobs 的 Next Run，保存恢复结果并 Arm Timer。
        方法不会立即等待任务完成；到期执行由 Background Timer 驱动。重复调用会重新执行恢复流程并
        替换现有 Timer Task。
        """
        self._running = True
        self._load_store()
        self._recompute_next_runs()
        self._save_store()
        self._arm_timer()
        logger.info("Cron service started with {} jobs", len(self._store.jobs if self._store else []))

    def stop(self) -> None:
        """停止 Cron Service 并取消当前 Timer Task。

        方法清除 Running Flag，使后续 Tick 不再调用 `run_due`。它不会删除 Persistent Jobs，也不等待
        已在 `on_job` 中执行的 Callback；完整 Runtime Shutdown 仍需由上层协调正在运行的任务。
        """
        self._running = False
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None

    def _recompute_next_runs(self) -> None:
        """为所有 Enabled Jobs 重新计算 Next Run Times。

        Job 在 Restart 间保留一个 Persisted Pending Fire。错过的 Recurring Fire 恢复后只运行一次，再从
        Recovery Time 向后推进，不 Backfill 每个 Missed Interval。已完成/失败/过期的一次性 `at` Job
        会禁用；无 Future Fire 的 Schedule 标记为 `expired` 并记录原因。已有合法 Pending Timestamp 的
        Recurring Job 不被无故覆盖。
        """
        if not self._store:
            return
        now = self._now_ms()
        for job in self._store.jobs:
            if not job.enabled:
                continue
            if job.schedule.kind == "at" and job.state.last_status in {
                "ok",
                "error",
                "expired",
                "incompatible",
            }:
                job.enabled = False
                job.state.next_run_at_ms = None
                continue
            if job.schedule.kind == "at" and job.schedule.at_ms is not None and job.schedule.at_ms > 0:
                job.state.next_run_at_ms = job.schedule.at_ms
                continue
            next_run = _compute_next_run(job.schedule, now)
            if next_run is None:
                job.state.next_run_at_ms = None
                job.enabled = False
                job.state.last_status = "expired"
                job.state.last_error = "schedule has no future fire"
                job.updated_at_ms = now
                logger.warning("Cron: expired job '{}' ({}): {}", job.name, job.id, job.state.last_error)
            elif job.state.next_run_at_ms is None:
                job.state.next_run_at_ms = next_run

    def _get_next_wake_ms(self) -> int | None:
        """返回所有 Enabled Jobs 中最早的 Next Run Time。

        只考虑有非空 `next_run_at_ms` 的 Job；Store 缺失或没有待运行项时返回 `None`。结果用于安排下个
        Timer Tick，不代表该 Job 已成功 Claim。
        """
        if not self._store:
            return None
        return min(
            (j.state.next_run_at_ms for j in self._store.jobs if j.enabled and j.state.next_run_at_ms),
            default=None,
        )

    def _arm_timer(self) -> None:
        """安排下一次 Timer Tick。

         先取消旧 Timer，再按 Earliest Wake 计算 Delay；即使没有任务也最多休眠
         ``_MAX_WAKE_INTERVAL_S``。因此 Peer Process 写入 ``jobs.json``，例如 Gateway 运行时 REPL 新增
        更早 Reminder，最迟会在该 Window 内被发现，因为 `run_due` 会根据 Mtime Reload。

         Service 未 Running 时不创建 Task。Timer 只负责唤醒 `run_due`，Claim 与执行仍在后者完成。
        """
        if self._timer_task:
            self._timer_task.cancel()

        if not self._running:
            return

        next_wake = self._get_next_wake_ms()
        if next_wake:
            delay_s = max(0.0, (next_wake - self._now_ms()) / 1000)
        else:
            # 没有待执行任务时仍需轮询新写入。
            delay_s = _MAX_WAKE_INTERVAL_S
        delay_s = min(delay_s, _MAX_WAKE_INTERVAL_S)

        async def tick():
            await asyncio.sleep(delay_s)
            if self._running:
                await self.run_due()

        self._timer_task = asyncio.create_task(tick())

    async def run_due(self) -> None:
        """Claim 并运行已经 Due 的 Jobs。

        Claim Phase 在 Exclusive Lock 内强制 Reload Disk，选择到期、Channel 允许、且未被 Live Peer
        Claim 的 Jobs，写入当前 PID + Now 后保存。超过 `_CLAIM_TTL_MS` 的旧 Claim 可被接管。Execution
        Phase 释放 Lock，逐个运行已 Claim Job；每项完成后重新加锁写入 Post-run State 并清除 Claim。

        `finally` 还会释放本批残余 Claims 并重新 Arm Timer，覆盖取消与异常路径。多进程 Claim 降低重复
        执行概率，但 Callback 的外部副作用仍应自行具备 Idempotency，因为 Process Crash 可能发生在
        副作用完成而结果尚未持久化之间。
        """
        my_pid = os.getpid()
        with self._locked():
            # 强制重读：同级进程可能已在此期间修改数据。
            self._store = None
            self._load_store()
            if not self._store:
                return

            now = self._now_ms()
            my_jobs: list[CronJob] = []
            for j in self._store.jobs:
                if not (j.enabled and j.state.next_run_at_ms and now >= j.state.next_run_at_ms):
                    continue
                # 渠道路由：调用方设置 allow-list 后，只 claim 其中渠道的任务。
                # 为向后兼容，在 channel attribution 出现前创建的任务
                # （channel 为空或 None）仍可被任意进程 claim。
                if self.allowed_channels is not None and j.payload.channel:
                    if j.payload.channel not in self.allowed_channels:
                        continue
                # 已被存活的同级进程持有时跳过。
                cb = j.state.claimed_by_pid
                ca = j.state.claimed_at_ms
                if cb is not None and ca is not None and (now - ca) < _CLAIM_TTL_MS:
                    continue
                j.state.claimed_by_pid = my_pid
                j.state.claimed_at_ms = now
                my_jobs.append(j)
            if my_jobs:
                self._save_store()

        try:
            for job in my_jobs:
                try:
                    await self._execute_job(job)
                finally:
                    self._persist_job_result(job, my_pid, now)
        finally:
            self._release_claims([job.id for job in my_jobs], my_pid, now)
            self._arm_timer()

    def _persist_job_result(self, job: CronJob, claimed_by_pid: int, claimed_at_ms: int) -> None:
        with self._locked():
            self._store = None
            self._load_store()
            if self._store is None:
                return
            owns_claim = False
            for persisted in self._store.jobs:
                if (
                    persisted.id != job.id
                    or persisted.state.claimed_by_pid != claimed_by_pid
                    or persisted.state.claimed_at_ms != claimed_at_ms
                ):
                    continue
                owns_claim = True
                persisted.state.claimed_by_pid = None
                persisted.state.claimed_at_ms = None
                persisted.state.last_run_at_ms = job.state.last_run_at_ms
                persisted.state.last_status = job.state.last_status
                persisted.state.last_error = job.state.last_error
                persisted.state.next_run_at_ms = job.state.next_run_at_ms
                persisted.enabled = job.enabled
                persisted.updated_at_ms = job.updated_at_ms
                break
            if not owns_claim:
                return
            if job.schedule.kind == "at" and job.delete_after_run and job.state.last_status != "cancelled":
                self._store.jobs = [persisted for persisted in self._store.jobs if persisted.id != job.id]
            self._save_store()

    def _release_claims(self, job_ids: list[str], claimed_by_pid: int, claimed_at_ms: int) -> None:
        if not job_ids:
            return
        with self._locked():
            self._store = None
            self._load_store()
            if self._store is None:
                return
            changed = False
            for job in self._store.jobs:
                if (
                    job.id in job_ids
                    and job.state.claimed_by_pid == claimed_by_pid
                    and job.state.claimed_at_ms == claimed_at_ms
                ):
                    job.state.claimed_by_pid = None
                    job.state.claimed_at_ms = None
                    changed = True
            if changed:
                self._save_store()

    async def _execute_job(self, job: CronJob) -> None:
        """执行一个已经 Claim 的 Job，并更新其 In-memory Run State。

         有 `on_job` Callback 时 Await 执行；成功写入 `ok`，普通异常转成 `error` 与文本原因，
         `CancelledError` 则记录 `cancelled` 后重新抛出。一次性 `at` Job 完成后删除或禁用；Enabled
         Recurring Job 从当前时间计算下一次运行；被 ``cron run --force`` 强制执行的 Disabled Job 保持
        禁用且不推进 Next Run。真正持久化由 Caller 随后的 `_persist_job_result` 完成。
        """
        start_ms = self._now_ms()
        logger.info("Cron: executing job '{}' ({})", job.name, job.id)

        try:
            if self.on_job:
                await self.on_job(job)

            job.state.last_status = "ok"
            job.state.last_error = None
            logger.info("Cron: job '{}' completed", job.name)

        except asyncio.CancelledError:
            job.state.last_run_at_ms = start_ms
            job.state.last_status = "cancelled"
            job.state.last_error = "execution cancelled"
            job.updated_at_ms = self._now_ms()
            logger.warning("Cron: job '{}' cancelled", job.name)
            raise
        except Exception as e:
            job.state.last_status = "error"
            job.state.last_error = str(e)
            logger.error("Cron: job '{}' failed: {}", job.name, e)

        job.state.last_run_at_ms = start_ms
        job.updated_at_ms = self._now_ms()

        # 处理一次性任务。
        if job.schedule.kind == "at":
            if job.delete_after_run:
                job.state.next_run_at_ms = None
            else:
                job.enabled = False
                job.state.next_run_at_ms = None
        elif job.enabled:
            job.state.next_run_at_ms = _compute_next_run(job.schedule, self._now_ms())
        else:
            # 这是禁用状态下被强制触发的周期任务（CLI `cron run --force`）。
            # 不推进 next_run_at_ms：任务仍处于禁用状态，未来的 next-run 与
            # enabled=False 同时出现会误导 `cron list` 输出。
            job.state.next_run_at_ms = None

    # ========== 公共 API ==========

    def record_silent_fire(self, job_id: str) -> bool:
        """递增 Job 的 ``silent_fire_count``，达到 ``silent_fire_limit`` 时 Auto-disable。

        Harness/Dispatch Path 在一次 Cron Fire 已交付后立即调用。方法在 Lock 内 Reload Store、更新计数并
        保存；达到正数 Limit 时同时清除 Next Run，防止长期没有 User Activity 的 Reminder 无限发送。
        本次调用触发 Auto-disable 时返回 `True`，仅计数或未找到 Job 时返回 `False`。
        """
        with self._locked():
            self._store = None
            store = self._load_store()
            for j in store.jobs:
                if j.id != job_id:
                    continue
                j.state.silent_fire_count += 1
                limit = j.silent_fire_limit
                disabled = False
                if limit is not None and limit > 0 and j.state.silent_fire_count >= limit:
                    j.enabled = False
                    j.state.next_run_at_ms = None
                    disabled = True
                    logger.warning(
                        "Cron: auto-disabled job '{}' ({}) — {} silent fires without user activity (limit={})",
                        j.name,
                        j.id,
                        j.state.silent_fire_count,
                        limit,
                    )
                self._save_store()
                return disabled
            return False

    def notify_user_active(self, channel: str | None = None, to: str | None = None) -> int:
        """为匹配 ``(channel, to)`` 的 Jobs 重置 ``silent_fire_count``。

        每当 Genuine User-originated Message 到达时调用，使近期触发的 Crons 不会继续 Decay Toward
        Auto-disable。`channel` 或 `to` 为 `None` 表示该维度 Match All；只处理 Enabled 且已有非零计数的
        Jobs。返回实际 Reset 的 Job Count，只有 Count 大于零才写盘。
        """
        reset = 0
        with self._locked():
            self._store = None
            store = self._load_store()
            for j in store.jobs:
                if not j.enabled or j.state.silent_fire_count == 0:
                    continue
                if channel is not None and j.payload.channel != channel:
                    continue
                if to is not None and j.payload.to != to:
                    continue
                j.state.silent_fire_count = 0
                reset += 1
            if reset > 0:
                self._save_store()
        return reset

    def list_jobs(self, include_disabled: bool = False) -> list[CronJob]:
        """列出 Jobs，并按 Next Run Time 排序。

        默认只返回 Enabled Jobs；`include_disabled=True` 时包含 Expired、Incompatible 与用户停用项。
        没有 Next Run 的 Job 排在末尾。返回的是当前 Store 中对象列表，不是深拷贝的长期 Snapshot。
        """
        store = self._load_store()
        jobs = store.jobs if include_disabled else [j for j in store.jobs if j.enabled]
        return sorted(jobs, key=lambda j: j.state.next_run_at_ms or float("inf"))

    def add_job(
        self,
        name: str,
        schedule: CronSchedule,
        message: str,
        deliver: bool = False,
        channel: str | None = None,
        to: str | None = None,
        delete_after_run: bool = False,
        topic_tag: str | None = None,
    ) -> CronJob:
        """添加新 Job，或根据多层 Dedup 更新/复用 Existing Job。

        最基础规则是相同 ``(schedule, channel, to)`` Triple：Agent 常在不同 Conversation 中用略有差异的
        文案重复注册 Recurring Reminder；没有 Dedup，用户每个 Scheduled Tick 会收到 N 条近似消息。

        更早执行的 Cross-kind Dedup Layers 依次是：

        1. **Topic-tag Dedup**：同一 ``(channel, to, topic_tag)`` 代表同一逻辑提醒，直接原地更新 Message
           与 Schedule；
        2. **Message-equal Dedup**：同一 ``(channel, to)`` 下已有 Enabled Job 的
           ``payload.message`` *Byte-identical* 时直接返回。它捕获 LLM 在 Simulation Horizon 多次创建同一
           Reminder，例如同一 Medication String 同时作为今天 ``at`` Shot 与明天 ``cron_expr``；
        3. **Time-window Dedup**：同一 ``(channel, to)`` 的 Existing Job 在新 Schedule Next Fire 前后 15
           Minutes 内触发时直接返回，不区分 Schedule Kind。它捕获同一 Intent 同时创建 Recurring
           ``cron_expr`` **AND** Same-day ``at`` Shot，例如 “daily 8:00 take meds” 与 “today 8:00 take
           meds”。

        所有判断在 Exclusive Lock 内基于 Reloaded Store；无效 Schedule 在争锁前失败。返回值可能是新
        Job、被原地更新的 Job 或未修改的 Duplicate，因此 Caller 不应仅凭返回对象判断创建了新记录。
        15-minute Trade-off 可能合并极少数真正独立的临近提醒，这是减少垃圾重复通知的显式策略。
        """
        # 校验与存储共用同一个 ``now`` 快照。校验谓词和存储的 next_run 必须对
        # “现在”达成一致，否则临界 ``at``（at ≈ now）可能通过校验却保存为
        # next_run=None。锁外取值，使无效 schedule 无需争锁即可快速失败。
        now = self._now_ms()
        _validate_schedule_for_add(schedule, now)
        with self._locked():
            # 在锁内重载，避免覆盖并发写入方新增的任务。
            self._store = None
            store = self._load_store()
            if store.version != 1:
                raise ValueError(f"unsupported cron store version '{store.version}'")

            # L7：topic_tag 去重最严格，因此最先执行。新请求带 topic_tag 时，
            # 相同 (channel, to, topic_tag) 的任何已启用任务都视为重复。这能捕获
            # 护理场景中的失败模式：LLM 创建近乎相同的用药提醒 cron，只在时间
            # （11:20 与 11:30）或措辞上略有差异；消息全等去重和 15 分钟窗口去重
            # 都会漏掉。topic_tag 就是“提醒主题”的身份，因此相同 topic_tag 的
            # 两个 cron 按定义属于同一逻辑提醒。就地更新现有任务的消息/schedule，
            # 而不是创建并行任务。
            if topic_tag:
                for j in store.jobs:
                    if not j.enabled:
                        continue
                    if j.payload.channel != channel or j.payload.to != to:
                        continue
                    if j.payload.topic_tag != topic_tag:
                        continue
                    logger.info(
                        "Cron: topic_tag dedup — existing job '{}' ({}) "
                        "has topic_tag='{}'; updating message + schedule "
                        "in place (kinds={}/{})",
                        j.name,
                        j.id,
                        topic_tag,
                        j.schedule.kind,
                        schedule.kind,
                    )
                    j.payload.message = message
                    j.payload.deliver = deliver
                    j.name = name
                    j.schedule = schedule
                    j.state.next_run_at_ms = _compute_next_run(schedule, now)
                    j.updated_at_ms = now
                    self._save_store()
                    self._arm_timer()
                    return j

            # 消息全等去重：覆盖 LLM 跨天重复请求的同意图提醒，即使 schedule
            # 类型不同。它比时间窗口更严格：完整消息文本按字节相等，误报率约为 0。
            for j in store.jobs:
                if not j.enabled:
                    continue
                if j.payload.channel != channel or j.payload.to != to:
                    continue
                if j.payload.message != message:
                    continue
                if j.state.next_run_at_ms is None or j.state.next_run_at_ms <= now:
                    continue
                logger.info(
                    "Cron: skipped duplicate add — existing job '{}' "
                    "({}) has identical message (same channel/to, "
                    "kinds={}/{})",
                    j.name,
                    j.id,
                    j.schedule.kind,
                    schedule.kind,
                )
                self._arm_timer()
                return j

            # 跨类型时间窗口去重：覆盖护理场景中“同一意图同时 expr + at”的重复添加。
            # 窗口设为较宽的 15 分钟，因为间隔不足 15 分钟的两个真正独立提醒几乎
            # 总是 LLM 错误。少数合法情况（8:00 和 8:10 服用不同药物）会少触发
            # 一次；相比垃圾提醒，这是可接受的权衡。
            new_next = _compute_next_run(schedule, now)
            if new_next is not None:
                for j in store.jobs:
                    if not j.enabled:
                        continue
                    if j.payload.channel != channel or j.payload.to != to:
                        continue
                    existing_next = j.state.next_run_at_ms
                    if existing_next is None:
                        continue
                    if abs(existing_next - new_next) <= 15 * 60 * 1000:
                        logger.info(
                            "Cron: skipped duplicate add — existing job '{}' "
                            "({}) fires within 15min of new request "
                            "(same channel/to, kinds={}/{})",
                            j.name,
                            j.id,
                            j.schedule.kind,
                            schedule.kind,
                        )
                        self._arm_timer()
                        return j

            # 去重：周期 schedule、channel 和 recipient 都相同时，
            # 就地更新消息而不是创建重复任务。
            existing = self._find_duplicate_schedule(store.jobs, schedule, channel, to)
            if existing is not None:
                existing.payload.message = message
                existing.payload.deliver = deliver
                existing.name = name
                existing.updated_at_ms = now
                # 仅当现有任务已经触发或被禁用时才重算 next_run_at_ms，
                # 否则保留原定触发时间。
                if not existing.enabled or existing.state.next_run_at_ms is None:
                    existing.enabled = True
                    existing.state.next_run_at_ms = _compute_next_run(schedule, now)
                self._save_store()
                logger.info(
                    "Cron: updated existing job '{}' ({}) with new message (dedup on schedule+channel+to)",
                    existing.name,
                    existing.id,
                )
                self._arm_timer()
                return existing

            job = CronJob(
                id=str(uuid.uuid4())[:8],
                name=name,
                enabled=True,
                schedule=schedule,
                payload=CronPayload(
                    kind="agent_turn",
                    message=message,
                    deliver=deliver,
                    channel=channel,
                    to=to,
                    topic_tag=topic_tag,
                ),
                state=CronJobState(next_run_at_ms=_compute_next_run(schedule, now)),
                created_at_ms=now,
                updated_at_ms=now,
                delete_after_run=delete_after_run,
            )
            store.jobs.append(job)
            self._save_store()
        self._arm_timer()
        logger.info("Cron: added job '{}' ({})", name, job.id)
        return job

    @staticmethod
    def _find_duplicate_schedule(
        jobs: list[CronJob],
        schedule: CronSchedule,
        channel: str | None,
        to: str | None,
    ) -> CronJob | None:
        """返回 ``(schedule, channel, to)`` 匹配的 Existing Enabled Job。

        `add_job` 用此 Helper 执行最后一层 Dedup。Cron 要求 Expr 与 Timezone 相同，Every 要求 Interval
        相同；``at`` One-shot Jobs 只有 ``at_ms`` 完全一致、即 Same Instant 时才合并。Disabled Job 与
        不同 Recipient 不参与匹配；没有命中返回 `None`。
        """
        for j in jobs:
            if not j.enabled:
                continue
            if j.payload.channel != channel or j.payload.to != to:
                continue
            s = j.schedule
            if s.kind != schedule.kind:
                continue
            if schedule.kind == "cron" and s.expr == schedule.expr and s.tz == schedule.tz:
                return j
            if schedule.kind == "every" and s.every_ms == schedule.every_ms:
                return j
            if schedule.kind == "at" and s.at_ms == schedule.at_ms:
                return j
        return None

    def remove_job(self, job_id: str) -> bool:
        """按 Job ID 删除一个 Persistent Job。

        方法在 Lock 内 Reload Store 并过滤目标；找到并保存删除结果时返回 `True`，随后 Rearm Timer；
        未找到返回 `False` 且不写盘。它不会取消一个已在锁外执行的 Callback，上层仍可能看到当前 Fire
        完成。
        """
        with self._locked():
            self._store = None
            store = self._load_store()
            before = len(store.jobs)
            store.jobs = [j for j in store.jobs if j.id != job_id]
            removed = len(store.jobs) < before
            if removed:
                self._save_store()
        if removed:
            self._arm_timer()
            logger.info("Cron: removed job {}", job_id)
        return removed

    def enable_job(self, job_id: str, enabled: bool = True) -> CronJob | None:
        """Enable 或 Disable 指定 Job，并返回更新后的对象。

        Enable 时从当前时间重新计算 Next Run；没有 Future Fire 则保持 Disabled，并在非 Incompatible
        情况下标为 `expired`。Disable 时清除 Next Run。找到 Job 后原子保存并 Rearm Timer；未知 ID
        返回 `None`。该操作不立即执行任务。
        """
        with self._locked():
            self._store = None
            store = self._load_store()
            for job in store.jobs:
                if job.id == job_id:
                    now = self._now_ms()
                    job.updated_at_ms = now
                    if enabled:
                        next_run = _compute_next_run(job.schedule, now)
                        job.enabled = next_run is not None
                        job.state.next_run_at_ms = next_run
                        if next_run is None and job.state.last_status != "incompatible":
                            job.state.last_status = "expired"
                            job.state.last_error = "schedule has no future fire"
                    else:
                        job.enabled = False
                        job.state.next_run_at_ms = None
                    self._save_store()
                    self._arm_timer()
                    return job
        return None

    async def run_job(self, job_id: str, force: bool = False) -> bool:
        """手动 Claim 并运行一个 Job。

        默认只运行 Enabled Job；`force=True` 可运行 Disabled Recurring Job，但仍拒绝 Expired、
        Incompatible 或被 Fresh Peer Claim 的任务。成功取得 Claim 后在锁外 `_execute_job`，最后无论结果
        都持久化状态、释放 Claim 并 Rearm Timer。开始执行时返回路径最终为 `True`，无法 Claim/找到时
        返回 `False`；Callback Exception 通常已转成 Job Error State。
        """
        claimed_by_pid = os.getpid()
        claimed_at_ms = self._now_ms()
        with self._locked():
            self._store = None
            store = self._load_store()
            target = next((j for j in store.jobs if j.id == job_id), None)
            if target is None or (not force and not target.enabled):
                return False
            if target.state.last_status in {"expired", "incompatible"}:
                return False
            if (
                target.state.claimed_by_pid is not None
                and target.state.claimed_at_ms is not None
                and claimed_at_ms - target.state.claimed_at_ms < _CLAIM_TTL_MS
            ):
                return False
            target.state.claimed_by_pid = claimed_by_pid
            target.state.claimed_at_ms = claimed_at_ms
            self._save_store()

        try:
            await self._execute_job(target)
            return True
        finally:
            self._persist_job_result(target, claimed_by_pid, claimed_at_ms)
            self._release_claims([target.id], claimed_by_pid, claimed_at_ms)
            self._arm_timer()

    def status(self) -> dict:
        """返回 Cron Service 的当前 Status Snapshot。

        Dict 包含 Running ``enabled``、Store 中总 ``jobs`` 数与最早 ``next_wake_at_ms``。它反映调度器
        内部状态，不验证 Background Timer 仍健康，也不证明最近 Job 已成功交付。
        """
        store = self._load_store()
        return {
            "enabled": self._running,
            "jobs": len(store.jobs),
            "next_wake_at_ms": self._get_next_wake_ms(),
        }
