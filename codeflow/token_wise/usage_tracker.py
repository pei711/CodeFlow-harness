"""`UsageTracker` 记录每次 LLM Call 的 Token Usage 与 Cost。

内存统计累积为 Three Tiers：

- ``per_session[session_key]``：一个 Session 内的 Cumulative Usage；
- ``per_day[date]``：Daily Roll-up，适合预算观察；
- ``total``：当前 Process Lifetime 的总计。

每次调用还会作为单行 JSON Object 追加到
``{telemetry_dir}/usage-YYYY-MM-DD.jsonl``，便于事后使用 ``jq`` 分析。只要聚合范围内任一次调用的
Pricing Unknown，Aggregate Cost 就是 `None`；Token Counts 仍照常累积，避免把未知成本误报为零。

Tracker 只是 Recorder，从不修改 Outgoing Request，所以继承 `before_llm_call` 的 Default No-op
Pass-through。记录成功也只证明计量数据被接收，不代表 LLM Reply 已满足用户任务。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from codeflow.product import get_product_home
from codeflow.token_wise.base import TokenStrategy, UsageSnapshot


def _default_telemetry_dir() -> Path:
    return get_product_home() / "telemetry"


class UsageTracker(TokenStrategy):
    """观察每次 LLM Call，持久化并 Roll Up Token 与 Cost Stats。

    实例在整个 Agent 生命周期中维护 Session、Date 与 Process 三种累计器，并按 `flush_every` 将调用
    明细批量写入 JSONL。作为 `TokenStrategy`，它只实现 Post-call 阶段；关闭 Agent 时应调用 `close`
    冲刷不足一个批次的剩余记录。持久化失败会记录 Warning 并丢弃当前 Buffer，不会中断主循环。
    """

    name = "usage_tracker"

    def __init__(
        self,
        telemetry_dir: Path | None = None,
        flush_every: int = 1,
        persist: bool = True,
    ):
        """创建 Tracker，并配置明细落盘策略。

        Args:
            telemetry_dir: ``usage-YYYY-MM-DD.jsonl`` 的写入目录；默认是
                ``~/.codeflow/telemetry``，实际路径由 Product Home 解析。
            flush_every: 写盘前在内存 Buffer 中积累 N Calls。`1` 表示每次都写，是最安全的默认值；
                更大数值可摊薄 IO，但进程异常退出时可能丢失尚未冲刷的记录。小于 1 会规范为 1。
            persist: 为 `False` 时只在内存累计，适合 Tests 或不允许写盘的运行环境。

        新实例的所有计数从零开始，不会读取历史 JSONL 恢复跨进程 Total。
        """
        self.telemetry_dir = telemetry_dir or _default_telemetry_dir()
        self.flush_every = max(1, flush_every)
        self.persist = persist

        self.per_session: dict[str, UsageSnapshot] = {}
        self.per_day: dict[date, UsageSnapshot] = {}
        self.total: UsageSnapshot = UsageSnapshot(model="__total__", estimated_cost_usd=0.0)

        self._call_count: int = 0
        self._buffer: list[dict[str, Any]] = []

    # ---- TokenStrategy 钩子 ----

    async def after_llm_call(self, response: dict[str, Any], usage: UsageSnapshot) -> None:
        """记录一次已经完成的 LLM Call。

        方法递增 Call Count，把 `usage` 累加进三个层级；启用 Persist 时再附加 UTC Timestamp 后放入
        Buffer，并在达到 `flush_every` 的整数倍时写盘。`response` 是 Hook 协议的一部分，但本策略
        不读取或保存回复正文，避免 Telemetry 文件意外复制对话内容。
        """
        self._call_count += 1
        self._accumulate(usage)

        if self.persist:
            self._buffer.append(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    **asdict(usage),
                }
            )
            if self._call_count % self.flush_every == 0:
                self._flush()

    # ---- 公共内省 ----

    def snapshot(self, session_key: str | None = None) -> UsageSnapshot:
        """返回 Session Accumulator 或 Lifetime Total 的 *Copy*。

        提供 `session_key` 时返回该 Session 的累计值；未知 Key 返回 Token 与 Cost 均为零、Model 为
        ``__empty__`` 的快照。不提供时返回 Process Total。返回对象与内部累计器分离，调用方修改它
        不会污染后续统计；但快照只覆盖当前进程已经观察到的调用，不会读取磁盘历史。
        """
        if session_key is not None:
            src = self.per_session.get(session_key) or UsageSnapshot(
                model="__empty__",
                estimated_cost_usd=0.0,
                session_key=session_key,
            )
        else:
            src = self.total
        return self._copy(src)

    def close(self) -> None:
        """把 Buffer 中所有剩余 Rows Flush 到 Disk。

        Agent 正常关闭时调用本方法，保证不足 `flush_every` 的尾批记录也尝试持久化。禁用 Persist 或
        Buffer 为空时只清理内存；写盘异常由 `_flush` 记录并吞掉，因此 `close` 不会因为 Telemetry
        失败阻止其他 Shutdown 步骤。
        """
        self._flush()

    # ---- 内部实现 ----

    def _accumulate(self, u: UsageSnapshot) -> None:
        key = u.session_key or "__no_session__"
        session_acc = self.per_session.get(key)
        if session_acc is None:
            session_acc = UsageSnapshot(model=u.model, estimated_cost_usd=0.0, session_key=key)
            self.per_session[key] = session_acc
        self._add_into(session_acc, u)

        today = date.today()
        day_acc = self.per_day.get(today)
        if day_acc is None:
            day_acc = UsageSnapshot(model="__day__", estimated_cost_usd=0.0)
            self.per_day[today] = day_acc
        self._add_into(day_acc, u)

        self._add_into(self.total, u)

    @staticmethod
    def _add_into(acc: UsageSnapshot, add: UsageSnapshot) -> None:
        acc.input_tokens += add.input_tokens
        acc.output_tokens += add.output_tokens
        acc.cache_read_tokens += add.cache_read_tokens
        acc.cache_write_tokens += add.cache_write_tokens
        acc.reasoning_tokens += add.reasoning_tokens
        if acc.estimated_cost_usd is None or add.estimated_cost_usd is None:
            acc.estimated_cost_usd = None
        else:
            acc.estimated_cost_usd += add.estimated_cost_usd

    @staticmethod
    def _copy(src: UsageSnapshot) -> UsageSnapshot:
        return UsageSnapshot(
            model=src.model,
            input_tokens=src.input_tokens,
            output_tokens=src.output_tokens,
            cache_read_tokens=src.cache_read_tokens,
            cache_write_tokens=src.cache_write_tokens,
            reasoning_tokens=src.reasoning_tokens,
            estimated_cost_usd=src.estimated_cost_usd,
            session_key=src.session_key,
        )

    def _flush(self) -> None:
        if not self._buffer or not self.persist:
            self._buffer.clear()
            return
        try:
            self.telemetry_dir.mkdir(parents=True, exist_ok=True)
            path = self.telemetry_dir / f"usage-{date.today().isoformat()}.jsonl"
            with path.open("a", encoding="utf-8") as f:
                for row in self._buffer:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._buffer.clear()
        except Exception as e:
            logger.warning("UsageTracker flush failed ({}); dropping {} rows", e, len(self._buffer))
            self._buffer.clear()
