"""装配 TUI RPC Turn 在 CodeFlow Spine 中经过的 runner、hub、outlet 与 sink。

TUI 的 happy path 是 ``submit -> lane -> run_turn -> hub -> outlet``。
``TuiTurnRunner`` 驱动 AgentLoop native ``run_turn``；``TuiOutlet`` 把 token、reasoning、
ToolEvent 与 Text 映射成 ``token.delta``、``thinking.delta``、``tool.*`` wire event；sink
在 render barrier 后发送 ``message.complete`` 或 ``error``。所有 deliverable 都经过 hub，
共享同一个 per-outlet FIFO，不存在第二条流式路径。依赖方向固定为 spine 不 import
``tui_rpc``，而 ``tui_rpc`` import spine。

``message.complete`` 必须由 sink 无条件发送，而不能依赖 stream-close：frontend 用它清除
turn slot，即使 Turn 没有 stream 内容（empty reply 或 tool-only）也必须发，否则 UI 会卡住。
sink 先等待 ``wait_idle``，保证 terminal event 位于最后一个 ``token.delta`` 之后；empty Turn
从未创建 queue，barrier 会立即返回。这是 REPL 的 ``result() -> wait_idle`` render barrier
移入 sink。

``message.complete`` 证明 Runtime Turn 到达成功终态并完成 event 排序，但不自动证明外部
用户已看到回复；Tool result 也不能单独作为整体任务完成或正向结论可用的证据。
"""

from collections.abc import Awaitable, Callable
from typing import Any

from codeflow.agent.spine_runner import AgentTurnRunner
from codeflow.agent.tools.message import MessageTool
from codeflow.spine import (
    Deliverable,
    Origin,
    OriginPools,
    Reasoning,
    RunnerEvent,
    Scheduler,
    Text,
    ToolEvent,
    ToolPhase,
    TurnEnded,
    TurnFailed,
    TurnOutcome,
    TurnRequest,
    TurnStarted,
)
from codeflow.spine.delivery import Capabilities, DeliveryHub
from codeflow.spine.events import TurnEvent
from codeflow.spine.runner import Drain, Emit
from codeflow.spine.teardown import teardown_spine
from codeflow.tui_rpc.errors import RpcError
from codeflow.tui_rpc.subscriptions import SubscriptionEmitter

_TURN_FAILED_CODE = -32099


def _conversation_id(req: TurnRequest) -> str:
    return req.conversation or f"{req.source.channel}:{req.source.chat_id}"


class TuiTurnRunner(AgentTurnRunner):
    """通过 AgentLoop native ``run_turn`` 执行 TUI Turn。

    User Turn 的 token/reasoning/ToolEvent/Text 全部经 hub 到 ``TuiOutlet``，使用单一
    per-outlet FIFO。相对 generic runner，本类增加三项 TUI 语义：只在 exact bound TUI
    request 开始时发 ``message.start``，并抑制同 lane 中 unbound system request 的
    deliverable；传入独立 ``usage_sink``，让 sink 把包含 cost/context、比三字段
    ``TurnOutcome.usage`` 更丰富的 usage 放入 ``message.complete``，rich usage 只留在
    TUI-internal map；当回复走 ``message`` Tool 时合成 ``tool.complete``，因为通用 Tool
    path 会跳过 MessageTool，但 UI 仍需记录 Agent 已采取动作。

    实例与一套 TUI Spine 生命周期一致，拥有 request correlation、usage、RPC error 与 cron
    readback maps 的引用，不拥有 Session persistence。CRON/SUBAGENT 使用 non-streaming
    readback 分支，USER 使用 streaming 分支。
    """

    def __init__(
        self,
        agent_loop: Any,
        emitter: SubscriptionEmitter,
        usages: dict[int, dict[str, Any]],
        turn_ids: dict[int, str],
        readback_texts: dict[str, str],
        running_requests: dict[str, int] | None = None,
        submission_ids: dict[int, str] | None = None,
        rpc_errors: dict[int, RpcError] | None = None,
        await_runtime_ready: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__(agent_loop, stream=True)
        self._emitter = emitter
        self._usages = usages
        self._turn_ids = turn_ids
        self._running_requests = running_requests if running_requests is not None else {}
        self._submission_ids = submission_ids if submission_ids is not None else {}
        self._rpc_errors = rpc_errors if rpc_errors is not None else {}
        self._await_runtime_ready = await_runtime_ready
        self._readback_texts = readback_texts

    async def run(self, req: TurnRequest, emit: Emit, drain: Drain) -> TurnOutcome:
        """执行一个 ``TurnRequest``，并返回 AgentLoop 的 ``TurnOutcome``。

        方法按 ``id(req)`` 建立 request identity，只有 ``Origin.USER`` 且同时绑定 turn_id 与
        submission_id 才视为 TUI-bound。bound request 先发 ``message.start``；可选
        ``await_runtime_ready`` 失败时保存 typed ``RpcError`` 供 sink 形成正确 error code。

        ``Origin.CRON`` 以 non-streaming 方式运行，并把 reply 写入
        ``readback_texts[conversation]``；``Origin.SUBAGENT`` 同样 non-streaming，非空结果转为
        ``subagent.delivered``；其他 Turn streaming 执行并收集 usage。返回 outcome 只交给
        Scheduler/sink 形成终态，调用本方法完成并不等于 frontend 已收到 terminal event。
        """
        cid = _conversation_id(req)
        request_key = id(req)
        self._running_requests[cid] = request_key
        turn_id = self._turn_ids.get(request_key)
        submission_id = self._submission_ids.get(request_key)
        bound = req.origin is Origin.USER and turn_id is not None and submission_id is not None

        async def emit_bound(event: RunnerEvent) -> None:
            if bound:
                await emit(event)

        if bound:
            await self._emitter.emit(
                cid,
                {
                    "type": "message.start",
                    "payload": {
                        "submission_id": submission_id,
                        "turn_id": turn_id,
                    },
                },
            )

        if self._await_runtime_ready is not None:
            try:
                await self._await_runtime_ready()
            except RpcError as exc:
                self._rpc_errors[request_key] = exc
                raise

        # CRON Turn 不是用户 Turn：它以非流式方式运行（一次回复，而非令牌流），
        # 并捕获回复文本供 cron 扇出，后者向每个会话交付 cron.delivered 事件。
        # 未绑定的可交付项在此处被抑制，不进入用户聊天流。该路径与网关的
        # GatewayTurnRunner 读回路径一致。
        if req.origin is Origin.CRON:
            text_sink: dict[str, str] = {}
            outcome = await self._loop.run_turn(req, emit_bound, drain, stream=False, text_sink=text_sink)
            if req.conversation is not None and (text := text_sink.get("text")) is not None:
                self._readback_texts[req.conversation] = text
            return outcome
        if req.origin is Origin.SUBAGENT:
            text_sink = {}
            outcome = await self._loop.run_turn(req, emit_bound, drain, stream=False, text_sink=text_sink)
            if text := text_sink.get("text"):
                await self._emitter.emit(
                    cid,
                    {
                        "type": "subagent.delivered",
                        "payload": {"text": text},
                    },
                )
            return outcome
        usage_sink: dict[str, Any] = {}
        outcome = await self._loop.run_turn(req, emit_bound, drain, stream=True, usage_sink=usage_sink)

        # 消息工具触发时生成一个合成 tool.complete（Agent Loop 在通用工具路径中会跳过它），
        # 让 UI 记录 Agent 已采取行动；回复已作为令牌增量流式输出。返回前发出，
        # 确保它在 Turn 事件流中位于 TurnEnded 之前。
        message_tool = self._loop.tools.get("message")
        if bound and isinstance(message_tool, MessageTool) and message_tool.sent_in_turn:
            await emit(
                ToolEvent(
                    phase=ToolPhase.COMPLETE,
                    tool_call_id=f"msg-{turn_id}",
                    result_preview="(message sent via tool)",
                )
            )

        self._usages[request_key] = dict(usage_sink)
        return outcome


class TuiOutlet:
    """把 Spine deliverable 转成 TUI wire event 的发送面。

    streamed token 通过 ``send_stream_chunk`` 变成 ``token.delta``；discrete
    ``Reasoning`` 变成 ``thinking.delta``，``ToolEvent`` 变成 ``tool.start`` /
    ``tool.complete``，non-streamed ``Text`` 也归一化为 ``token.delta``。Turn 成功终态
    ``message.complete`` 与失败 ``error`` 由 sink 在 render barrier 后调用专用方法发送。

    ``Notice`` 和 ``MediaOut`` 当前被消费，因为 wire protocol 没有对应 event，TUI 也尚不
    展示 per-turn progress 或 Tool media；这是已知 deferred gap。实例拥有 channel 名、
    streaming capability 与 emitter，不拥有 queue，排序由 ``DeliveryHub`` 负责。
    """

    def __init__(self, channel: str, emitter: SubscriptionEmitter) -> None:
        self.name = channel
        self.capabilities = Capabilities(streaming=True)
        self._emitter = emitter

    async def deliver(self, out: Deliverable) -> None:
        """把一个 discrete ``Deliverable`` 发到对应 conversation subscription。

        空 Reasoning/Text 不发送 event；Tool START 保留 call id、name、arguments，complete
        保留 result preview、truncated 与可选 failed。未知 deliverable 当前无 wire 映射并被
        消费。方法返回表示 emitter 已接纳 event，不保证 frontend 已渲染。
        """
        cid = out.conversation_id
        if isinstance(out, Reasoning):
            if out.content:
                await self._emitter.emit(cid, {"type": "thinking.delta", "payload": {"text": out.content}})
        elif isinstance(out, ToolEvent):
            if out.phase is ToolPhase.START:
                await self._emitter.emit(
                    cid,
                    {
                        "type": "tool.start",
                        "payload": {
                            "tool_call_id": out.tool_call_id,
                            "name": out.name,
                            "arguments": out.arguments or {},
                        },
                    },
                )
            else:
                payload = {
                    "tool_call_id": out.tool_call_id,
                    "result_preview": out.result_preview,
                    "truncated": out.truncated,
                }
                if out.failed:
                    payload["failed"] = True
                await self._emitter.emit(
                    cid,
                    {
                        "type": "tool.complete",
                        "payload": payload,
                    },
                )
        elif isinstance(out, Text):
            # 非流式回复（澄清、Hook 短路或空回退）通过一个 token.delta 进入流式回复使用的
            # 同一缓冲区，使 message.complete 能像处理其他文本一样完成它。
            if out.content:
                await self._emitter.emit(cid, {"type": "token.delta", "payload": {"text": out.content}})
            # Notice 和 MediaOut 当前没有线上事件，因此被消耗。

    async def send_stream_chunk(self, chat_id: str, stream_id: str, delta: str, *, done: bool = False) -> None:
        """把一个 stream delta 映射为 ``token.delta``。

        ``stream_id`` 作为 subscription conversation key；``chat_id`` 仅为 Outlet protocol
        兼容参数。``done=True`` 时不发 event，因为 frontend 没有 stream-done，最终收敛由
        sink 的 ``message.complete`` 完成；空 delta 同样 no-op。
        """
        if done:
            # 前端没有 stream-done 事件；Turn 由 sink 发出的 message.complete 完结。
            # done=True 只让 hub 关闭其流状态。
            return
        if not delta:
            return
        await self._emitter.emit(stream_id, {"type": "token.delta", "payload": {"text": delta}})

    async def emit_complete(
        self,
        conversation_id: str,
        submission_id: str,
        turn_id: str,
        usage: dict[str, Any],
    ) -> None:
        """发送 Turn 成功终态 ``message.complete``。

        payload 保留 ``submission_id``、``turn_id`` 与 rich ``usage``。调用方必须先通过
        render barrier，保证该 event 位于所有 token delta 之后。它表示 Turn Runtime 成功
        终止，不证明用户已读或任务的业务目标达成。
        """
        await self._emitter.emit(
            conversation_id,
            {
                "type": "message.complete",
                "payload": {
                    "submission_id": submission_id,
                    "turn_id": turn_id,
                    "usage": usage,
                },
            },
        )

    async def emit_error(
        self,
        conversation_id: str,
        submission_id: str,
        turn_id: str,
        code: int,
        message: str,
        reason: str,
    ) -> None:
        """发送 Turn 失败终态 ``error``。

        payload 包含稳定 ``code``、public ``message``、``reason`` 及 correlation IDs。sink
        负责避免 cancelled Turn double error。发送成功表示 error event 进入 emitter，不保证
        client 已收到。
        """
        await self._emitter.emit(
            conversation_id,
            {
                "type": "error",
                "payload": {
                    "code": code,
                    "message": message,
                    "reason": reason,
                    "submission_id": submission_id,
                    "turn_id": turn_id,
                },
            },
        )


def _make_tui_sink(
    hub: DeliveryHub,
    outlet: TuiOutlet,
    channel: str,
    turn_ids: dict[int, str],
    submission_ids: dict[int, str],
    usages: dict[int, dict[str, Any]],
    running_requests: dict[str, int],
    rpc_errors: dict[int, RpcError],
    on_turn_end: Callable[[str, int], None] | None,
) -> Callable[[TurnEvent], Awaitable[None]]:
    """把 ``DeliveryHub`` 适配成 Scheduler 使用的 TUI ``EventSink``。

    普通 deliverable 经 hub 路由；``TurnEnded``/``TurnFailed`` 先 ``close_stream``，再
    ``wait_idle`` 通过 render barrier，确保 ``message.complete``/``error`` 位于最后一个
    ``token.delta`` 之后。每次 Turn 退出时，在发送 terminal event 前调用 ``on_turn_end``，
    让 turn.send active slot 先释放，frontend 随后即可安全提交下一 Turn。

    callback 接收 exact request identity，避免共享 lane 的 unrelated Turn 清错 slot。cancelled
    ``TurnFailed`` 不再发 error，因为 ``turn.cancel`` 已发送唯一取消事件。该 sink 只属于
    ``build_tui``；CLI 保持自己的 lifecycle-dropping sink。
    """

    async def _finish(conversation_id: str) -> None:
        # close_stream 清除 hub 按流保存的状态，使该会话的下一个 Turn 能干净重开；
        # 随后 wait_idle 阻塞，直到所有排队的 token.delta 都已交付。空 Turn 从未创建队列，
        # 因此会立即返回。
        await hub.close_stream(conversation_id)
        await hub.wait_idle(channel)

    def _correlation(conversation_id: str) -> tuple[int | None, str | None, str | None]:
        request_key = running_requests.get(conversation_id)
        if request_key is None:
            return None, None, None
        return request_key, turn_ids.get(request_key), submission_ids.get(request_key)

    def _drop(conversation_id: str, request_key: int | None) -> None:
        if request_key is None:
            return
        if running_requests.get(conversation_id) == request_key:
            running_requests.pop(conversation_id, None)
        turn_ids.pop(request_key, None)
        submission_ids.pop(request_key, None)
        usages.pop(request_key, None)
        rpc_errors.pop(request_key, None)
        if on_turn_end is not None:
            on_turn_end(conversation_id, request_key)

    async def sink(event: TurnEvent) -> None:
        if isinstance(event, TurnEnded):
            await _finish(event.conversation_id)
            request_key, turn_id, submission_id = _correlation(event.conversation_id)
            bound = turn_id is not None and submission_id is not None
            usage = usages.get(request_key) or {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            _drop(event.conversation_id, request_key)
            if not bound:
                return
            await outlet.emit_complete(event.conversation_id, submission_id, turn_id, usage)
            return
        if isinstance(event, TurnFailed):
            await _finish(event.conversation_id)
            request_key, turn_id, submission_id = _correlation(event.conversation_id)
            bound = turn_id is not None and submission_id is not None
            rpc_error = rpc_errors.get(request_key) if request_key is not None else None
            _drop(event.conversation_id, request_key)
            # 已取消 Turn 的错误由 turn.cancel 发出，此处不再发送，避免重复错误事件。
            if bound and not event.cancelled:
                code = rpc_error.code if rpc_error is not None else _TURN_FAILED_CODE
                message = rpc_error.public_message if rpc_error is not None else "turn_failed"
                await outlet.emit_error(
                    event.conversation_id,
                    submission_id,
                    turn_id,
                    code,
                    message,
                    "internal",
                )
            return
        if isinstance(event, TurnStarted):
            # TuiTurnRunner 将精确请求与对应 turn.send 绑定关联后，再发出 message.start。
            return
        await hub.dispatch(event)

    return sink


def build_tui(
    agent_loop: Any,
    emitter: SubscriptionEmitter,
    *,
    channel: str = "tui",
    on_turn_end: Callable[[str, int], None] | None = None,
    readback_texts: dict[str, str] | None = None,
    user_pool: int = 1,
    system_pool: int = 1,
    await_runtime_ready: Callable[[], Awaitable[None]] | None = None,
    turn_ids: dict[int, str] | None = None,
    submission_ids: dict[int, str] | None = None,
) -> tuple[Scheduler, DeliveryHub, dict[int, str], dict[int, str], Callable[[], Awaitable[None]]]:
    """构建 TUI Turn 经过的完整 Spine bundle。

    创建注册了 ``TuiOutlet`` 的 ``DeliveryHub``，以及使用 ``TuiTurnRunner`` 与 TUI sink 的
    ``Scheduler``。``user_pool``/``system_pool`` 控制 OriginPools；``await_runtime_ready``
    允许 runner 在真正调用 Agent 前等待 Runtime 初始化。返回 Scheduler、Hub、按 request
    identity 关联 terminal event 的 ``turn_ids``/``submission_ids`` maps，以及调用方退出时
    必须 await 的 ``teardown``。sink 发送 ``message.complete`` 或 ``error``；``on_turn_end``
    使 turn.send 只删除匹配 active slot。

    ``readback_texts`` 是 cron read-back map（conversation -> reply text）：runner 把 CRON
    reply 写入其中，cron fan-out 再发送 ``cron.delivered``。调用方必须传入 cron callback
    读取的同一 dict；未装配 cron（如 test）时使用 private map。构建成功只表示组件已连线，
    尚未提交 Turn；teardown 通过 ``teardown_spine(..., grace=0.0)`` 关闭 Scheduler 与 Hub。
    """
    hub = DeliveryHub()
    outlet = TuiOutlet(channel, emitter)
    hub.register(outlet)
    turn_ids = turn_ids if turn_ids is not None else {}
    submission_ids = submission_ids if submission_ids is not None else {}
    usages: dict[int, dict[str, Any]] = {}
    running_requests: dict[str, int] = {}
    rpc_errors: dict[int, RpcError] = {}
    if readback_texts is None:
        readback_texts = {}
    scheduler = Scheduler(
        TuiTurnRunner(
            agent_loop,
            emitter,
            usages,
            turn_ids,
            readback_texts,
            running_requests,
            submission_ids,
            rpc_errors,
            await_runtime_ready,
        ),
        OriginPools(user=user_pool, system=system_pool),
        _make_tui_sink(
            hub,
            outlet,
            channel,
            turn_ids,
            submission_ids,
            usages,
            running_requests,
            rpc_errors,
            on_turn_end,
        ),
    )

    async def teardown() -> None:
        await teardown_spine(scheduler, hub, grace=0.0)

    return scheduler, hub, turn_ids, submission_ids, teardown
