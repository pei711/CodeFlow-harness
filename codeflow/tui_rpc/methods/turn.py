"""实现把 TUI request 接入 CodeFlow Spine 的 ``turn.*`` handlers。

* ``turn.send`` 向 Spine 提交 Turn，每个 ``session_key`` 同时最多一个，并同步返回
  ``{turn_id, accepted: True}``；streaming output 通过 build_tui runner/hub/sink 变成
  ``SubscriptionEmitter`` notification；
* ``turn.subscribe`` 包装 ``SubscriptionEmitter.register``；
* ``turn.unsubscribe`` 幂等包装 ``SubscriptionEmitter.unregister``；
* ``turn.cancel`` 取消 in-flight ``TurnHandle``，并发出唯一一条
  ``reason="cancelled_by_client"`` error；sink 对 cancelled ``TurnFailed`` 保持静默，以免
  double error。

module-level handler 让 test 可以 patch ``_resolve_model`` seam。
``register_turn_methods`` 把 ``emitter`` 与 build_tui bundle（``scheduler``、``turn_ids``、
``submission_ids``、``build_error``）闭包为 Dispatcher 所需的单参数 handler。

最重要的状态边界是：``accepted=True`` 只表示请求已被接纳或已生成可见失败事件，不等于
Turn 已开始、Agent 已完成、Session 已持久化或回复已交付。真正终态来自后续 subscription
的 ``message.complete`` 或 ``error``。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import ValidationError

from codeflow.spine import ChatType, Origin, Source, TurnHandle, TurnRequest
from codeflow.spine.scheduler import Scheduler, SchedulerDrainingError
from codeflow.tui_rpc.errors import RpcError, TurnInProgressError
from codeflow.tui_rpc.methods.image import consume_pending_images, pending_images
from codeflow.tui_rpc.models import (
    TurnCancelParams,
    TurnSendParams,
    TurnSubscribeParams,
    TurnUnsubscribeParams,
)
from codeflow.tui_rpc.subscriptions import SubscriptionEmitter

if TYPE_CHECKING:
    from codeflow.tui_rpc.dispatcher import Dispatcher

_TURN_FAILED_CODE = -32099
SchedulerFactory = Callable[[], Awaitable[Scheduler]]

# ---------------------------------------------------------------------------
# 模块级状态
# ---------------------------------------------------------------------------

# 以 session_key 为键保存正在执行的 Turn 句柄。每个会话同一时刻只能有一个 Turn：
# 存在句柄时 ``turn.send`` 以 -32003 拒绝，``turn.cancel`` 则取消该句柄。build_tui sink
# 会在每个 Turn 结束时通过 ``clear_active`` 回调清空槽位，因此此处存在记录就表示正在执行。
_active_turns: dict[str, TurnHandle] = {}
_active_request_keys: dict[str, int] = {}


def is_turn_active(session_key: str) -> bool:
    """判断 ``session_key`` 是否存在 in-flight Turn。

    build_tui sink 会在 Turn 终止时删除 active slot，因此映射中存在即表示 liveness。该值是
    当前进程 snapshot，只覆盖 TUI handler 管理的 Turn，不是持久化任务状态。
    """
    return session_key in _active_turns


def has_active_turns() -> bool:
    """返回当前是否至少有一个 TUI Turn 处于 active slot。

    该全局 gate 用于禁止 live model switch 等跨 Session 操作；返回 ``False`` 不代表 Spine
    没有其他来源的 pending work。
    """
    return bool(_active_turns)


def clear_active(session_key: str, request_key: int | None = None) -> None:
    """删除匹配的 Session active-turn slot。

    ``build_tui`` 提供 completed request identity，只有与 ``_active_request_keys`` 相同才会
    清理，防止同一 Spine lane 上更早的 system Turn 错误删除 queued user Turn。独立生命周期
    之间做 unconditional teardown 时可省略 ``request_key``。未知或 identity 不匹配时 no-op。
    """
    if request_key is not None and _active_request_keys.get(session_key) != request_key:
        return
    _active_turns.pop(session_key, None)
    _active_request_keys.pop(session_key, None)


# ---------------------------------------------------------------------------
# 可模拟边界
# ---------------------------------------------------------------------------


def _resolve_model(parsed: TurnSendParams) -> str:
    """在生成 AgentLoop 前解析本 Turn 的 model id。

    若没有 routable provider/model，应抛出 ``ModelNotAvailableError``（``-32008``）。当前
    default implementation 是 no-op pass-through，real model selection 由 AgentLoop 拥有；
    test 会 patch 此 seam 断言 ``-32008`` 路径。``parsed`` 已由 ``TurnSendParams`` 校验。
    """
    return "default"


# ---------------------------------------------------------------------------
# 处理器
# ---------------------------------------------------------------------------


async def _emit_start_then_error(
    emitter: SubscriptionEmitter,
    session_key: str,
    submission_id: str,
    turn_id: str,
    code: int,
    message: str,
    *,
    attachments_discarded: bool = False,
) -> None:
    # 先发 message.start，让前端有可清理的 Turn；随后错误会通过 onError 重置 turnId，
    # 与旧的单 Turn 任务保持相同形状。
    await emitter.emit(
        session_key,
        {
            "type": "message.start",
            "payload": {"submission_id": submission_id, "turn_id": turn_id},
        },
    )
    await emitter.emit(
        session_key,
        {
            "type": "error",
            "payload": {
                "attachments_discarded": attachments_discarded,
                "code": code,
                "message": message,
                "reason": "internal",
                "submission_id": submission_id,
                "turn_id": turn_id,
            },
        },
    )


async def turn_send(
    params: dict[str, Any],
    *,
    emitter: SubscriptionEmitter | None = None,
    scheduler: Scheduler | None = None,
    scheduler_factory: SchedulerFactory | None = None,
    turn_ids: dict[int, str] | None = None,
    submission_ids: dict[int, str] | None = None,
    build_error: RpcError | None = None,
) -> dict[str, Any]:
    """执行 ``turn.send``：向 Spine 提交 Turn，返回 ``{turn_id, accepted}``。

    ``params`` 先由 ``TurnSendParams`` 校验。可选 ``scheduler_factory`` 只在 scheduler 缺失时
    lazy 构建；构建 ``RpcError`` 被保存为 ``build_error``。model seam 在占用 active slot 前
    快速检查，不可路由时抛出 ``ModelNotAvailableError``。每次请求生成 ``turn_id``，并复用
    或生成 ``submission_id``。

    没有 scheduler 时不执行 Turn：pending image 会被消费，并通过 emitter 发送
    ``message.start`` 后紧跟 build error 或 ``-32008 model_not_available``，仍返回
    ``accepted=True`` 让前端按统一 event 流收尾。已有 active TUI Turn 或 Scheduler Lane
    pending/running work 时抛出 ``-32003 TurnInProgressError``。``SchedulerDrainingError``
    同样转换为 start + ``turn_failed`` event。

    submit 成功后，函数在无 await 间隙内按 request identity 绑定 ``turn_ids``、
    ``submission_ids`` 与 active handle，再消费附件。runner 只在 exact request 真正开始时
    发 ``message.start`` 和 token delta，``build_tui`` sink 发 ``message.complete`` /
    ``error``。返回
    ``accepted=True`` 不是任务完成判定。
    """
    try:
        parsed = TurnSendParams.model_validate(params)
    except ValidationError as exc:
        # 原样重新抛出；分派器会捕获并发出 -32603 internal_error。
        raise exc

    if scheduler is None and scheduler_factory is not None:
        try:
            scheduler = await scheduler_factory()
        except RpcError as exc:
            build_error = exc

    # 快速失败：在占用活动 Turn 槽位之前检查模型可用性，避免 -32008 拒绝后
    # 会话无法继续发送请求。
    _resolve_model(parsed)

    turn_id = uuid4().hex
    submission_id = parsed.submission_id or uuid4().hex

    if scheduler is None:
        # 未连接 Agent Loop（构建失败或无提供商）。对当前 Turn 优先暴露构建错误自身的错误码，
        # 否则使用 -32008；不执行 Turn。
        attachments_discarded = bool(pending_images(parsed.session_key))
        consume_pending_images(parsed.session_key)
        if emitter is not None:
            if build_error is not None:
                await _emit_start_then_error(
                    emitter,
                    parsed.session_key,
                    submission_id,
                    turn_id,
                    build_error.code,
                    build_error.public_message,
                    attachments_discarded=attachments_discarded,
                )
            else:
                await _emit_start_then_error(
                    emitter,
                    parsed.session_key,
                    submission_id,
                    turn_id,
                    -32008,
                    "model_not_available",
                    attachments_discarded=attachments_discarded,
                )
        return {"turn_id": turn_id, "accepted": True}

    if is_turn_active(parsed.session_key) or scheduler.has_pending_or_running(parsed.session_key):
        raise TurnInProgressError(
            f"session {parsed.session_key!r} already has pending or running work",
        )

    req = TurnRequest(
        origin=Origin.USER,
        source=Source(
            channel=parsed.channel or "tui",
            chat_id=parsed.chat_id or "default",
            sender_id=parsed.sender_id or "user",
            chat_type=ChatType.DM,
        ),
        text=parsed.content,
        media=pending_images(parsed.session_key),
        # conversation 等于前端订阅键，使运行器的流和 sink 的 message.complete
        # 到达正确订阅。
        conversation=parsed.session_key,
    )
    request_key = id(req)
    try:
        handle = scheduler.submit(req)
    except SchedulerDrainingError:
        # 服务器正在关闭：发出 turn_failed 让前端清理槽位；此时未绑定任何内容，不会泄漏。
        attachments_discarded = bool(pending_images(parsed.session_key))
        consume_pending_images(parsed.session_key)
        if emitter is not None:
            await _emit_start_then_error(
                emitter,
                parsed.session_key,
                submission_id,
                turn_id,
                _TURN_FAILED_CODE,
                "turn_failed",
                attachments_discarded=attachments_discarded,
            )
        return {"turn_id": turn_id, "accepted": True}
    consume_pending_images(parsed.session_key)

    # submit 后立即绑定，中间不出现 await。worker 虽已排程，但在这些按请求存储的记录
    # 建立前无法运行。以请求身份为键，可防止同一会话通道中的先前系统 Turn
    # 消费当前用户 Turn 的关联信息。
    if turn_ids is not None:
        turn_ids[request_key] = turn_id
    if submission_ids is not None:
        submission_ids[request_key] = submission_id
    _active_turns[parsed.session_key] = handle
    _active_request_keys[parsed.session_key] = request_key

    return {"turn_id": turn_id, "accepted": True}


async def turn_subscribe(
    params: dict[str, Any],
    *,
    emitter: SubscriptionEmitter | None = None,
) -> dict[str, Any]:
    """执行 ``turn.subscribe``，为 Session 打开 event subscription。

    参数按 ``TurnSubscribeParams`` 校验。缺少 ``SubscriptionEmitter`` 表示装配错误，会抛出
    ``RuntimeError``；成功时调用 ``emitter.register(session_key)`` 并返回
    ``{subscription_id}``。订阅建立不启动 Turn，也不重放此前事件。
    """
    parsed = TurnSubscribeParams.model_validate(params)
    if emitter is None:
        raise RuntimeError(
            "turn.subscribe requires a SubscriptionEmitter; register_turn_methods must be called with emitter=...",
        )
    sub_id = await emitter.register(parsed.session_key)
    return {"subscription_id": sub_id}


async def turn_unsubscribe(
    params: dict[str, Any],
    *,
    emitter: SubscriptionEmitter | None = None,
) -> dict[str, Any]:
    """执行 ``turn.unsubscribe``，幂等关闭指定 subscription。

    参数按 ``TurnUnsubscribeParams`` 校验。缺少 emitter 时抛出 ``RuntimeError``；返回
    ``{"unsubscribed": bool}``，unknown/已关闭 ID 得到 ``False``。关闭订阅不会取消正在
    运行的 Turn。
    """
    parsed = TurnUnsubscribeParams.model_validate(params)
    if emitter is None:
        raise RuntimeError(
            "turn.unsubscribe requires a SubscriptionEmitter; register_turn_methods must be called with emitter=...",
        )
    unsubscribed = await emitter.unregister(parsed.subscription_id)
    return {"unsubscribed": unsubscribed}


async def turn_cancel(
    params: dict[str, Any],
    *,
    emitter: SubscriptionEmitter | None = None,
    turn_ids: dict[int, str] | None = None,
    submission_ids: dict[int, str] | None = None,
) -> dict[str, Any]:
    """执行 ``turn.cancel``：取消 in-flight Turn，并通知 subscribers。

    顺序不能改变：先查 active handle，缺失返回 ``{cancelled: False}``；再调用
    ``handle.cancel()``；若有 emitter，执行
    ``emitter.emit(session_key, error(reason="cancelled_by_client"))`` 的等价 wire 发送。
    这是 ONLY cancelled-turn error，也是 MUST 发送的 UI 清理信号；sink 对 cancelled
    ``TurnFailed`` 静默，避免 double error。``session_key`` 是对应 subscription key。

    随后等待 ``handle.result()``，证明 Turn 已 unwind、sink 有机会释放 active slot，再清理
    request identity maps 并返回 ``{cancelled: True}``。这样下一次 ``turn.send`` 不会与
    half-unwound Turn 竞争并产生 phantom ``-32003``。

    subscription 是 SESSION-scoped，不是 turn-scoped；取消一个 Turn 必须保留 Session
    subscriptions，使下一次 Turn event 仍能到达 client。返回 ``True`` 表示取消与 drain
    完成，不等于之前已发生的外部 Tool 副作用被回滚。
    """
    parsed = TurnCancelParams.model_validate(params)

    handle = _active_turns.get(parsed.session_key)
    if handle is None:
        return {"cancelled": False}
    request_key = _active_request_keys.get(parsed.session_key)

    handle.cancel()

    if emitter is not None:
        turn_id = turn_ids.get(request_key) if turn_ids is not None else None
        submission_id = submission_ids.get(request_key) if submission_ids is not None else None
        payload: dict[str, Any] = {
            "code": _TURN_FAILED_CODE,
            "message": "turn_cancelled",
            "reason": "cancelled_by_client",
        }
        if turn_id is not None:
            payload["turn_id"] = turn_id
        if submission_id is not None:
            payload["submission_id"] = submission_id
        await emitter.emit(
            parsed.session_key,
            {
                "type": "error",
                "payload": payload,
            },
        )

    # 返回前排空，确保 sink 已释放活动 Turn 槽位。
    # handle.result() 在取消时返回 None，不抛异常。
    await handle.result()
    clear_active(parsed.session_key, request_key)
    if request_key is not None:
        if turn_ids is not None:
            turn_ids.pop(request_key, None)
        if submission_ids is not None:
            submission_ids.pop(request_key, None)

    return {"cancelled": True}


# ---------------------------------------------------------------------------
# 分派器注册
# ---------------------------------------------------------------------------


def register_turn_methods(
    dispatcher: "Dispatcher",
    *,
    emitter: SubscriptionEmitter | None = None,
    scheduler: Scheduler | None = None,
    scheduler_factory: SchedulerFactory | None = None,
    turn_ids: dict[int, str] | None = None,
    submission_ids: dict[int, str] | None = None,
    build_error: RpcError | None = None,
) -> None:
    """在 Dispatcher 注册 ``turn.{send,subscribe,unsubscribe,cancel}``。

    四个 module-level handler 被包装为单参数 closure，预绑定 ``emitter``、build_tui Spine
    bundle（``scheduler``、``scheduler_factory``、``turn_ids``、``submission_ids``）与
    latched ``build_error``，满足 Dispatcher single-argument handler contract。

    注册不会创建 subscription 或提交 Turn；若 ``emitter`` 缺失，subscribe/unsubscribe 会在
    调用时明确失败。重复注册由 Dispatcher 抛出 ``ValueError``。
    """

    async def _send(params: dict[str, Any]) -> dict[str, Any]:
        return await turn_send(
            params,
            emitter=emitter,
            scheduler=scheduler,
            scheduler_factory=scheduler_factory,
            turn_ids=turn_ids,
            submission_ids=submission_ids,
            build_error=build_error,
        )

    async def _subscribe(params: dict[str, Any]) -> dict[str, Any]:
        return await turn_subscribe(params, emitter=emitter)

    async def _unsubscribe(params: dict[str, Any]) -> dict[str, Any]:
        return await turn_unsubscribe(params, emitter=emitter)

    async def _cancel(params: dict[str, Any]) -> dict[str, Any]:
        return await turn_cancel(
            params,
            emitter=emitter,
            turn_ids=turn_ids,
            submission_ids=submission_ids,
        )

    dispatcher.register("turn.send", _send)
    dispatcher.register("turn.subscribe", _subscribe)
    dispatcher.register("turn.unsubscribe", _unsubscribe)
    dispatcher.register("turn.cancel", _cancel)


__all__ = [
    "has_active_turns",
    "register_turn_methods",
    "turn_send",
    "turn_subscribe",
    "turn_unsubscribe",
    "turn_cancel",
]
