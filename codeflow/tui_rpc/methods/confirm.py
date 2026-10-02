"""实现确认往返的 answer sink：``confirm.respond`` RPC handler。

前端收到 ``confirm.request`` 后调用
``confirm.respond {request_id, answer}``；handler 把回答交给 ``ConfirmBroker``，由 Broker
完成匹配的 pending Future。注册时用 closure 预绑定 Broker，这与
``register_turn_methods`` 绑定 ``emitter`` 的方式一致。

umbrella registration 只在 Broker 非 ``None`` 时开放此 method，因此没有构建 Broker 的
demo/test 路径不会注册它，也能保持 umbrella-vs-production drift test 平衡。该 method 是
OpenRPC contract 的一部分。``ok=True`` 仅表示回答匹配到等待项，不表示确认后的业务动作
已经成功执行。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from codeflow.tui_rpc.confirm_broker import ConfirmBroker
    from codeflow.tui_rpc.dispatcher import Dispatcher


async def confirm_respond(params: dict[str, Any], *, confirm_broker: "ConfirmBroker") -> dict:
    """解析前端回答，并完成匹配的 pending confirm。

    从 ``params`` 读取 ``request_id`` 与 ``answer``，缺失时分别归一化为空字符串和
    ``False``，再调用 ``confirm_broker.resolve()``。返回 ``{"ok": True}`` 表示 Broker
    接纳回答；unknown、expired 或已完成的 ``request_id`` 返回 ``{ok: False}``（JSON 为
    ``{"ok": false}``）。
    重复调用不会再次改变原 Future。
    """
    request_id = str(params.get("request_id", ""))
    answer = bool(params.get("answer", False))
    ok = confirm_broker.resolve(request_id, answer)
    return {"ok": ok}


def register_confirm_methods(dispatcher: "Dispatcher", *, confirm_broker: "ConfirmBroker") -> None:
    """把预绑定 ``confirm_broker`` 的 ``confirm.respond`` 注册到 Dispatcher。

    函数只建立 closure 和 method mapping，不发送 ``confirm.request``，也不创建 pending
    Future。重复注册同名 method 会由 Dispatcher 抛出 ``ValueError``。
    """

    async def _respond(params: dict[str, Any]) -> dict:
        return await confirm_respond(params, confirm_broker=confirm_broker)

    dispatcher.register("confirm.respond", _respond)


__all__ = ["confirm_respond", "register_confirm_methods"]
