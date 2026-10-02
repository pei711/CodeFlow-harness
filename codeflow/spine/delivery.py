"""定义 Channel 投递能力、流式 opt-in、Outlet 发送接缝与 DeliveryHub 路由。

Agent Runner 发出的 Deliverable 先按 ``source.channel`` 进入对应 Outlet 的有界队列，再由该
Outlet 唯一的串行 Worker 真正发送。队列是背压点：某 Channel 队列满只阻塞该 Channel 的
sender，不会造成跨 Outlet head-of-line blocking；同一 Channel 的事件顺序则由单 Worker
保持。这是执行侧 conversation Lane 模型在 Delivery 侧的对应结构。

`Capabilities` 显式声明 Channel 能力，`SupportsStreaming` 让支持 edit-in-place 的实现选择
接收增量，`Outlet.deliver` 负责一次可投递事件，`DeliveryHub` 拥有注册表、队列、Worker、
重试和关闭屏障。Spine 永不 import channels；Channel 经 `channels.contract` re-export 导入
这里的词汇，依赖方向不能反转。
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from loguru import logger

from codeflow.spine._barrier import finish_barrier
from codeflow.spine.events import (
    Deliverable,
    MediaOut,
    Notice,
    NoticeKind,
    StreamDelta,
    Text,
    TurnEnded,
    TurnEvent,
    TurnFailed,
    TurnStarted,
)
from codeflow.tracing import semconv, trace

DeliveryFailureSink = Callable[[Notice], Awaitable[None]]


class TerminalDeliveryError(RuntimeError):
    """表示平台以重复发送也无法修复的方式拒绝了一次投递。

    Outlet 用该异常明确区分永久失败与普通临时异常。DeliveryHub 捕获后立即把事件标记为
    dropped，不执行指数退避；权限拒绝、无效目标等不可重试原因适合走此路径。异常本身不
    发送用户通知，失败 Notice 由 Hub 的带外 failure sink 负责，避免再次进入故障 Outlet。
    """


@dataclass(frozen=True)
class Capabilities:
    """显式声明一个 Channel 可被上层消费的能力，而不是从方法存在性猜测。

    ``interactive_login`` 供 CLI `channel login` 判断是否支持 QR/扫码登录；``streaming`` 是
    `SupportsStreaming` 的能力槽，只有同时满足 Protocol 与该值为真时 Hub 才发送增量。
    当前只保留真实消费者会读取的字段。``media``/``reactions`` 仍是 adapter-internal，尚无
    路由决策依赖它们；未来必须与消费者一起加入，不能提前制造无效 capability。
    """

    interactive_login: bool = False  # QR/扫码登录；由 CLI `channel login` 读取
    streaming: bool = False  # SupportsStreaming 槽位；在 B 阶段启用


@runtime_checkable
class SupportsStreaming(Protocol):
    """让 Outlet opt-in 接收可原地更新的增量回复。

    实现提供 `send_stream_chunk(chat_id, stream_id, delta, done=...)`，Hub 才能把 StreamDelta
    逐块发送，并在最后补一个 ``done=True`` 空块关闭流。仅实现方法还不够，Capabilities 的
    ``streaming`` 也必须开启。若 Agent Loop 没有产生 stream chunks（scope B 未接线），该
    Protocol 保持 inert，不会自行把完整 Text 拆块。
    """

    async def send_stream_chunk(self, chat_id: str, stream_id: str, delta: str, *, done: bool = False) -> None: ...


@runtime_checkable
class Outlet(Protocol):
    """定义单个 Channel 把 Deliverable 送到平台的发送接缝。

    ``deliver`` 能表达事件时负责渲染；Channel 无法表达某类事件时可以记录 skip 后正常返回，
    这种“吃掉”是能力选择，不是失败。真实失败必须抛异常：``TerminalDeliveryError`` 会立即
    dropped，其他异常由 Hub 按配置重试。`name` 用于 source.channel 路由，`capabilities`
    供上层作显式能力判断。

    connect/teardown 等生命周期仍由 Channel 本体拥有，Outlet 只暴露 send seam。它不持有
    Turn state，也不决定 lifecycle event、重试次数或跨 Channel 顺序。
    """

    name: str
    capabilities: Capabilities

    async def deliver(self, out: Deliverable) -> None: ...


_SEND_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # 秒；每次重试翻倍（1、2、4）
_OUTLET_QUEUE_MAXSIZE = 100  # 每个 Outlet 的背压上限；配置项与消费者放在一起


def _is_terminal_deliverable(out: Deliverable) -> bool:
    """判断 Deliverable 是否值得建立一次终端投递 span。

    只有 Text 与 MediaOut 是用户最终可见的 Turn 输出，返回 ``True``；stream deltas、
    reasoning、notices 和 tool events 属于高频 progress surface，若逐条建立终端 span 会淹没
    Turn 主 trace。该分类只控制 tracing 与失败报告，不决定事件是否实际投递。
    """
    return isinstance(out, (Text, MediaOut))


@dataclass(frozen=True)
class _StreamClose:
    """在 Outlet 队列中表示“前序增量之后关闭该 conversation stream”的内部标记。

    TurnEnded/TurnFailed lifecycle event 没有 Source，不能直接走 deliverable-only 队列；Hub
    根据先前记录的 Channel 改为入队 `_StreamClose`。单 Worker 按序消费后发送
    ``done=True`` chunk，因此关闭一定排在仍在 flight 的最后一个 StreamDelta 后。标记只带
    ``conversation_id``，不是对外协议事件。
    """

    conversation_id: str


@dataclass(frozen=True)
class _Routed:
    """把排队 Deliverable 与 enqueue 当下捕获的 trace ids 绑定在一起。

    Outlet Worker 是每 Channel 一个、跨越多个 Turn 的常驻任务，它自身的 contextvars 快照只
    属于最初启动它的调用现场。若只传事件，之后的 delivery span 可能错误加入旧 Turn。
    `_Routed` 同时携带 ``trace_id`` 与 ``parent_span_id``，让 Worker 处理每项时 attach 回正确
    trace；这些字段只修复观测归属，不改变事件 Source 或路由。
    """

    out: Deliverable
    trace_id: str | None
    parent_span_id: str | None


class DeliveryHub:
    """按 Source Channel 把 Deliverable 路由到有界队列，并由对应 Worker 串行发送。

    Hub 拥有 Outlet registry、每 Outlet queue 和 Worker，但不拥有 Turn state。普通临时异常按
    exponential backoff 重试，TerminalDeliveryError 立即丢弃；队列满形成仅限当前 Channel 的
    backpressure。事件一旦入队，调用方得到的只是“已提交给投递所有者”，不是“用户已看到”。

    Streaming 使用同一队列：StreamDelta 交给 `send_stream_chunk`，`close_stream` 入队内部标记，
    保证 done chunk 排在所有增量后。``_open_streams`` 只由单 Worker 读写，是 stream 是否真实
    打开的唯一表；``_stream_channel`` 则在 enqueue 同步记录 conversation 所属 Channel，使无
    Source 的 lifecycle close 能找到队列。`aclose` 封住新工作并共享 cleanup barrier。
    """

    def __init__(
        self,
        send_max_retries: int = _SEND_MAX_RETRIES,
        *,
        on_delivery_failure: DeliveryFailureSink | None = None,
    ) -> None:
        self._send_max_retries = send_max_retries
        # 有意采用带外通知，不再经过 dispatch：刚耗尽重试的 channel 正是需要承载
        # 报告的 channel，重新 dispatch 会再次经过故障 Outlet 并形成循环。
        self._on_delivery_failure = on_delivery_failure
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._outlets: dict[str, Outlet] = {}
        self._queues: dict[str, asyncio.Queue[_Routed | _StreamClose]] = {}
        self._workers: dict[str, asyncio.Task[None]] = {}
        # conversation_id -> channel，在 enqueue（sink 路径）时写入，供
        # close_stream 路由标记；下方 open-stream 表由 worker 持有，映射
        # conversation_id -> chat_id，且仅在 stream 已打开时存在。
        self._stream_channel: dict[str, str] = {}
        self._open_streams: dict[str, str] = {}

    def register(self, outlet: Outlet) -> None:
        # 仅在启动时注册一次：运行中的 worker 会在启动时捕获 Outlet，因此为活跃
        # channel 重新注册其他 Outlet 不会热替换现有实例。
        if self._closed:
            raise RuntimeError("delivery hub is closed")
        if outlet.name in self._outlets:
            raise ValueError(f"outlet {outlet.name!r} is already registered")
        self._outlets[outlet.name] = outlet

    async def dispatch(self, out: Deliverable) -> None:
        await self._enqueue(out)

    async def post(self, out: Deliverable) -> None:
        """提交一个并非来自 Turn 的 Deliverable，并复用正常 Channel 路由。

        调用方必须先在 ``source.channel`` 标明目标；方法随后与 `dispatch` 一样调用 `_enqueue`。
        返回只表示事件已经入队，不表示 Outlet 已发送或用户已看到，因为 delivery ownership
        属于常驻 Worker。若该 Channel 的有界队列已满，当前调用会承受背压，但其他 Channel
        的 sender 不受影响。
        """
        await self._enqueue(out)

    async def close_stream(self, conversation_id: str) -> None:
        """结束指定 conversation 的流，并让 ``done=True`` 严格排在最后一个增量之后。

        TurnEnded / TurnFailed lifecycle event 驱动本方法；它从 `_stream_channel` 取出此前在
        StreamDelta enqueue 时记录的 Channel，再把 `_StreamClose` 放进同一 Outlet queue。
        conversation 从未产生 stream、非流式 Outlet 或映射已消费时是 no-op。

        Hub 已关闭时抛 `RuntimeError`。若入队在关闭竞争中失败，只要 Hub 尚未真正 closed 就
        把 Channel 映射恢复，允许调用方重试；成功入队不代表 close chunk 已发送，仍需 Worker
        按序消费。
        """
        if self._closed:
            raise RuntimeError("delivery hub is closed")
        channel = self._stream_channel.pop(conversation_id, None)
        if channel is None:
            return
        queue = self._queues.get(channel)
        if queue is not None:
            try:
                await self._put(queue, _StreamClose(conversation_id))
            except BaseException:
                if not self._closed:
                    self._stream_channel.setdefault(conversation_id, channel)
                raise

    async def _put(self, queue: asyncio.Queue[_Routed | _StreamClose], item: _Routed | _StreamClose) -> None:
        """把一个路由项提交给队列，并处理“满队列等待期间 Hub 被关闭”的竞争。

        `queue.put` 可能因背压挂起；恢复后若 `_closed` 已为真，说明 close 赢得竞争，方法会调用
        `drain` 清除仍排队事件并抛 `RuntimeError`，不能向调用方假报提交成功。正常路径没有
        返回业务值，队列项的完成责任转移给 Outlet Worker。
        """
        await queue.put(item)
        if self._closed:
            self.drain()
            raise RuntimeError("delivery hub is closed")

    async def _enqueue(self, out: Deliverable) -> None:
        if self._closed:
            raise RuntimeError("delivery hub is closed")
        if out.source is None:
            raise ValueError(f"cannot route a {type(out).__name__} with no source")
        channel = out.source.channel
        if channel not in self._outlets:
            logger.warning("no outlet for channel {!r}; dropping {}", channel, type(out).__name__)
            if _is_terminal_deliverable(out):
                self._deliver_span(out, channel, semconv.CHANNEL_NO_OUTLET, attempts=0)
            return
        if isinstance(out, StreamDelta):
            # 记住该 stream 所属 channel，使之后由无来源 lifecycle event 驱动的
            # close_stream 能把标记路由到这里。
            self._stream_channel.setdefault(out.conversation_id, channel)
        queue = self._queues.get(channel)
        if queue is None:
            queue = asyncio.Queue(maxsize=_OUTLET_QUEUE_MAXSIZE)
            self._queues[channel] = queue
        worker = self._workers.get(channel)
        if worker is None or worker.done():
            # 不仅在 worker 缺失时启动，done() 时也要重启。worker 常驻并阻塞在 get；
            # 已死亡的 worker 会让队列无人消费，静默阻塞该 channel 的发送方。
            self._workers[channel] = asyncio.create_task(self._run_outlet(channel))
        ctx = trace.current()
        routed = _Routed(
            out=out,
            trace_id=getattr(ctx, "trace_id", None),
            parent_span_id=getattr(ctx, "parent_span_id", None),
        )
        await self._put(queue, routed)  # 队列满时只阻塞当前 channel（每 Outlet 独立背压）

    async def _run_outlet(self, channel: str) -> None:
        queue = self._queues[channel]
        outlet = self._outlets[channel]
        while not self._closed:
            item = await queue.get()
            try:
                if isinstance(item, _StreamClose):
                    await self._close_stream_chunk(outlet, item.conversation_id)
                elif isinstance(item.out, StreamDelta):
                    await self._stream_chunk(outlet, item.out)
                else:
                    await self._deliver_routed(outlet, item)
            except Exception:
                # 渲染错误不得杀死常驻 worker，否则下一次 enqueue 重启它之前
                # 队列都无人消费，期间该 channel 会静默停滞。
                logger.exception("outlet worker recovered from a delivery error: channel={!r}", channel)
            finally:
                # 无论事件被消费还是重试耗尽，都必须标记 done，使 wait_idle 的
                # join() 覆盖每个已出队条目，永不悬挂。
                queue.task_done()

    async def _stream_chunk(self, outlet: Outlet, ev: StreamDelta) -> None:
        # 非流式 Outlet 会消费 delta，完整文本会通过其他路径到达；只有既具备能力
        # 又声明 streaming 的 Outlet 才接收分块。
        if not (isinstance(outlet, SupportsStreaming) and outlet.capabilities.streaming):
            return
        chat_id = ev.source.chat_id
        self._open_streams.setdefault(ev.conversation_id, chat_id)  # 第一个 delta 打开 stream
        await outlet.send_stream_chunk(chat_id, ev.conversation_id, ev.delta, done=False)

    async def _close_stream_chunk(self, outlet: Outlet, conversation_id: str) -> None:
        chat_id = self._open_streams.pop(conversation_id, None)
        if chat_id is None:
            return  # 没有打开的 stream（空 Turn 或非流式 Outlet），无需操作
        if isinstance(outlet, SupportsStreaming) and outlet.capabilities.streaming:
            await outlet.send_stream_chunk(chat_id, conversation_id, "", done=True)

    def _deliver_span(
        self,
        out: Deliverable,
        channel: str,
        outcome: str,
        *,
        attempts: int,
        error: str | None = None,
    ) -> None:
        chat_id = getattr(out.source, "chat_id", None) if out.source is not None else None
        conversation_id = out.conversation_id or (f"{channel}:{chat_id}" if chat_id else None)
        with trace.span(
            "channel.deliver",
            semconv.channel_deliver(
                channel=channel,
                event=type(out).__name__,
                conversation_id=conversation_id,
                outcome=outcome,
                attempts=attempts,
                error=error,
            ),
            kind="channel",
            session_key=conversation_id,
            channel=channel,
            chat_id=chat_id,
        ) as span:
            if outcome != semconv.CHANNEL_DELIVERED:
                span.error(outcome)

    async def _deliver_routed(self, outlet: Outlet, item: _Routed) -> None:
        outcome, attempts, error = await self._deliver_with_retry(outlet, item.out)
        if not _is_terminal_deliverable(item.out):
            return
        with trace.attach(item.trace_id, item.parent_span_id):
            self._deliver_span(item.out, outlet.name, outcome, attempts=attempts, error=error)
        if outcome == semconv.CHANNEL_DROPPED:
            await self._report_delivery_failure(item.out, outlet.name, error)

    async def _report_delivery_failure(self, out: Deliverable, channel: str, error: str | None) -> None:
        if self._on_delivery_failure is None:
            return
        notice = Notice(
            kind=NoticeKind.DELIVERY_FAILED,
            source=out.source,
            detail=f"{channel}:{type(out).__name__}:{error or 'unknown'}",
            conversation_id=out.conversation_id,
        )
        try:
            await self._on_delivery_failure(notice)
        except Exception:
            logger.exception("delivery-failure sink raised: channel={!r}", channel)

    async def _deliver_with_retry(self, outlet: Outlet, out: Deliverable) -> tuple[str, int, str | None]:
        """向 Outlet 投递事件，并只对非终止异常执行指数退避重试。

        首次调用加最多 `_send_max_retries` 次重试，等待从 `_RETRY_BASE_DELAY` 开始每次翻倍。
        正常返回得到 ``(CHANNEL_DELIVERED, attempts, None)``；`TerminalDeliveryError` 立即得到
        ``CHANNEL_DROPPED``；普通异常耗尽预算后也 dropped。第三个字段只保存异常类型名，
        避免把平台原始细节写入统一 trace。Outlet 正常“吃掉”不支持的事件视为 delivered，
        因为能力缺失不是发送失败。
        """
        delay = _RETRY_BASE_DELAY
        for attempt in range(self._send_max_retries + 1):
            try:
                await outlet.deliver(out)
                return semconv.CHANNEL_DELIVERED, attempt + 1, None
            except TerminalDeliveryError as exc:
                logger.error(
                    "terminal delivery failure: channel={!r} event={} reason={}",
                    outlet.name,
                    type(out).__name__,
                    exc,
                )
                return semconv.CHANNEL_DROPPED, attempt + 1, type(exc).__name__
            except Exception as exc:
                if attempt == self._send_max_retries:
                    logger.error(
                        "delivery failed after {} retries: channel={!r} event={} reason={}",
                        self._send_max_retries,
                        outlet.name,
                        type(out).__name__,
                        exc,
                    )
                    return semconv.CHANNEL_DROPPED, attempt + 1, type(exc).__name__
                await asyncio.sleep(delay)
                delay *= 2

    def drain(self) -> int:
        """丢弃所有仍在队列、尚未投递的事件，并返回数量。

        方法同步且没有 ``await``，因此不会在单 event loop 中与活跃 Worker 的取项动作交错。
        每个 `get_nowait` 都配对 `task_done`，保持 unfinished 计数一致，避免之后 `join()` 永久
        悬挂。它只处理 queued events，不取消已出队正在发送的项；带 shutdown window 的
        best-effort flush 尚未实现，所以 close 选择明确丢弃而非声称已经送达。
        """
        dropped = 0
        for queue in self._queues.values():
            while not queue.empty():
                queue.get_nowait()
                queue.task_done()  # 保持 unfinished 计数一致，避免 join() 悬挂
                dropped += 1
        if dropped:
            logger.warning("delivery hub drained {} undelivered events on shutdown", dropped)
        return dropped

    async def wait_idle(self, channel: str) -> None:
        """等待指定 Channel 的 Outlet 处理完所有已排队事件，形成 render barrier。

        `TurnHandle.result()` 只表示 Runner 不会再产生事件，并不表示已有事件 delivered 或已经
        on-screen；需要可见性边界的调用方应在 result 后 await 本方法。它使用该 Channel queue
        的 `join()`，覆盖成功、吞掉与重试耗尽后都正确 `task_done` 的项。Channel 从未创建队列
        时已经 idle，立即返回；其他 Channel 的积压不会阻塞这里。
        """
        queue = self._queues.get(channel)
        if queue is None:
            return
        await queue.join()

    async def _finish_close(self) -> None:
        workers = tuple(self._workers.values())
        for worker in workers:
            worker.cancel()
        try:
            await asyncio.gather(*workers, return_exceptions=True)
        finally:
            self._workers.clear()
            self.drain()
            self._stream_channel.clear()
            self._open_streams.clear()

    async def aclose(self) -> None:
        """封住 DeliveryHub，并在传播调用方取消前完成共享 cleanup barrier。

        首次调用把 `_closed` 设为真并创建 `_finish_close`：取消全部常驻 Worker、等待退出、
        清空 Worker 表、drain 队列并删除 stream 映射。并发调用复用同一 `_close_task`，不会
        重复清理。`finish_barrier` 即使遇到 caller cancellation 也先让 cleanup 完成，再把取消
        传播出去，防止投递任务和队列状态遗留在半关闭阶段。
        """
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._finish_close())
        await finish_barrier(self._close_task)


def make_hub_sink(hub: DeliveryHub) -> Callable[[TurnEvent], Awaitable[None]]:
    """把 DeliveryHub 适配成 Scheduler 的 `EventSink`，同时守住事件职责边界。

    Runner deliverables 会调用 `hub.dispatch`，TurnStarted、TurnFailed、TurnEnded lifecycle
    events 因没有 Source 而在这里丢弃，绝不会误入 deliverable-only enqueue 路径；未来
    lifecycle -> taps 应由独立观察链承担。返回的异步 closure 不修改事件。

    REPL 与 Gateway 共享该 sink。TUI 保留自己的 sink，因为它必须在 render barrier 之后再
    触发 message.complete / error；将 TUI 也塞进此适配器会把“Turn 已结束”和“画面已渲染”
    两个事实混为一谈。
    """

    async def sink(event: TurnEvent) -> None:
        if isinstance(event, (TurnStarted, TurnFailed, TurnEnded)):
            return
        await hub.dispatch(event)

    return sink
