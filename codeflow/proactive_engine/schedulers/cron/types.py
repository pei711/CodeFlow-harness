"""Cron Scheduler 使用的 Persistent Data Types。

这些 Dataclasses 依次描述 Schedule、Trigger Payload、Mutable Runtime State、完整 Job 与 Versioned Store。
它们是 Service 内存模型，也对应 ``jobs.json`` Schema；字段名称或语义变化必须考虑旧 Store 的兼容
读取，不能只修改 Python 构造器。
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class CronSchedule:
    """Cron Job 的 Schedule Definition。

    `kind` 决定且只应启用一组字段：`at` 使用绝对 Millisecond Timestamp，`every` 使用 Millisecond
    Interval，`cron` 使用 Expr 与可选 Timezone。Dataclass 不自行验证组合，`CronService` Add Path 负责
    拒绝过去时间、非正间隔和非法表达式。
    """

    kind: Literal["at", "every", "cron"]
    # "at"：毫秒时间戳
    at_ms: int | None = None
    # "every"：毫秒间隔
    every_ms: int | None = None
    # "cron"：cron 表达式，如 "0 9 * * *"
    expr: str | None = None
    # cron 表达式的时区
    tz: str | None = None


@dataclass
class CronPayload:
    """描述 Job Runs 时要执行与交付的内容。

    `kind` 区分 `system_event` 与默认 `agent_turn`，`message` 是触发内容；`deliver`、`channel` 与 `to`
    保存 Request-time Delivery Binding。`topic_tag` 是稳定 Logical Subject，Service 用它更新同一提醒而
    不是创建近似重复 Schedule。Payload 是意图描述，不包含实际 Delivery Receipt。
    """

    kind: Literal["system_event", "agent_turn"] = "agent_turn"
    message: str = ""
    # 将响应投递到渠道
    deliver: bool = False
    channel: str | None = None  # 例如 "feishu"
    to: str | None = None  # 例如电话号码
    # 稳定的逻辑主题，用于更新现有任务，而不是为同一提醒创建近似重复的 schedule。
    topic_tag: str | None = None


@dataclass
class CronJobState:
    """一个 Job 的 Mutable Runtime State。

    它记录 Next/Last Run、Last Status/Error 与 Cross-process Claim Ownership。Claim 由 `run_due` 取得任务
    的 Process 写入，运行后清除，超过 TTL 才可由 Peer 接管。`silent_fire_count` 统计没有用户活动介入
    的连续 Fires，供 Auto-disable Guard 阻止 LLM 创建的失控 Recurring Task。
    """

    next_run_at_ms: int | None = None
    last_run_at_ms: int | None = None
    last_status: Literal["ok", "error", "skipped", "expired", "cancelled", "incompatible"] | None = None
    last_error: str | None = None
    # claim 字段由在 run_due 中取得任务的进程设置，运行后清除。
    # 过期 claim（早于 CLAIM_TTL_MS）可被接管。
    claimed_by_pid: int | None = None
    claimed_at_ms: int | None = None
    # 自动衰减计数：没有用户活动介入时的连续触发次数。同一 channel/to 中任意
    # 用户来源消息都会将其重置为 0。silent-fires 守卫用它禁用 LLM 创建的
    # 失控周期任务，例如永久 every_seconds=3000。
    silent_fire_count: int = 0


@dataclass
class CronJob:
    """一条完整 Scheduled Job Record。

    Job 把稳定 ID/Name、Enable Flag、`CronSchedule`、`CronPayload`、`CronJobState` 与创建更新时间组合在
    一起。`delete_after_run` 控制 One-shot 完成后是否从 Store 删除；`silent_fire_limit` 达到时自动停用，
    `None` 表示无限制。对象可在 Service 内原地更新，持久化前不构成 Durable State。
    """

    id: str
    name: str
    enabled: bool = True
    schedule: CronSchedule = field(default_factory=lambda: CronSchedule(kind="every"))
    payload: CronPayload = field(default_factory=CronPayload)
    state: CronJobState = field(default_factory=CronJobState)
    created_at_ms: int = 0
    updated_at_ms: int = 0
    delete_after_run: bool = False
    # 自动衰减上限：state.silent_fire_count 达到该值时自动禁用任务。
    # None 表示无限制，直到显式删除。默认 12 在两者间取平衡：按小时触发时
    # 约运行一天，之后判定“用户未参与”。
    silent_fire_limit: int | None = 12


@dataclass
class CronStore:
    """保存 Cron Jobs 的 Versioned Persistent Store。

    `version` 决定 JSON Schema，目前为 1；`jobs` 保持全部 Enabled、Disabled、Expired 与 Incompatible
    Records。Service 负责原子写入与兼容解析，Dataclass 本身不提供并发控制。
    """

    version: int = 1
    jobs: list[CronJob] = field(default_factory=list)
