"""实现 ask-user 往返的 answer sink：``clarify.respond`` RPC handler。

暂停中的 ``ask_user`` Tool call 通过 :class:`QuestionBroker` 发出
``clarify.request``，前端随后调用 ``clarify.respond {request_id, answer}``；handler 把
回答交回 Broker，完成匹配的 pending Future。``clarify.request`` / ``clarify.respond``
沿用 ui-tui 已有的 multi-choice prompt contract（``ClarifyPrompt``），不新增 frontend card。

注册使用预绑定 Broker 的 closure，与 ``register_confirm_methods`` 一致。umbrella 只在
Broker 非 ``None`` 时注册，因此未构建 Broker 的路径不会暴露该 method。它属于 OpenRPC
contract。``ok=True`` 只说明答案送达等待中的 Tool call，不代表后续 Agent 任务完成。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from codeflow.tui_rpc.dispatcher import Dispatcher
    from codeflow.tui_rpc.question_broker import QuestionBroker


async def question_respond(params: dict[str, Any], *, question_broker: "QuestionBroker") -> dict:
    """用前端答案完成匹配的 pending question。

    handle 可来自 ``conversation_id`` 或 ``request_id``，前者优先；``answer`` 被转换为
    string 后交给 ``question_broker.reply()``。匹配成功返回 ``{"ok": True}``；unknown、
    expired 或已完成 key 返回 ``{ok: False}``（JSON 为 ``{"ok": false}``）。重复回答不会
    再次唤醒 Tool call。
    """
    key = str(params.get("conversation_id") or params.get("request_id") or "")
    answer = str(params.get("answer", ""))
    ok = question_broker.reply(key, answer)
    return {"ok": ok}


def register_question_methods(dispatcher: "Dispatcher", *, question_broker: "QuestionBroker") -> None:
    """把预绑定 ``question_broker`` 的 ``clarify.respond`` 注册到 Dispatcher。

    注册不会创建问题或 pending Future；重复注册由 Dispatcher 抛出 ``ValueError``。
    """

    async def _respond(params: dict[str, Any]) -> dict:
        return await question_respond(params, question_broker=question_broker)

    dispatcher.register("clarify.respond", _respond)


__all__ = ["question_respond", "register_question_methods"]
