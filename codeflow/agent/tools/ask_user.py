"""实现可在 Turn 中暂停执行、向用户提问并等待回答的 ``ask_user`` Tool。

这是 blocking interaction：Registry 不为它包装统一 Tool timeout，等待的 fail-safe 由
QuestionBroker 自己管理。执行时 Tool 把本 Turn 的 conversation_id、问题和选项交给 Broker；
Broker 发出 ``clarify.request`` notification，直到入站答案到达或 fail-safe default 触发。

回答会转成自然语言 Tool Result，再进入下一轮 LLM Context；Agent Loop 不需要捕获 Broker
异常。共享 Tool 实例用 ContextVar 隔离并发 Turn 的 conversation key，Broker 则是传输层单例，
两者生命周期不能颠倒。
"""

from contextvars import ContextVar
from typing import Any

from codeflow.agent.tools.base import Tool
from codeflow.tui_rpc.question_broker import QuestionBroker


class AskUserTool(Tool):
    """在 Turn 中途向用户提出一个或多个问题，并等待对应回答。

    构建 Tool 集合的层必须通过 constructor 或 :meth:`set_broker` 注入共享
    :class:`QuestionBroker`，并用 :meth:`set_context` 写入本 Turn conversation_id；该 key 必须与
    Scheduler 的 ``req.conversation or f"{channel}:{chat_id}"`` 完全一致，Broker 才能把入站回答
    路由回正确等待者。

    Tool 标记 ``blocking_interaction=True``，所以等待人类不受 Registry 兜底超时影响。缺 Broker、
    缺 Context 或没有非空问题时返回明确 Error 字符串；无回答时返回“按最佳判断继续”的结果，
    不让模型误以为用户选择了某个选项。
    """

    blocking_interaction = True

    def __init__(
        self,
        broker: QuestionBroker | None = None,
        conversation_id: str = "",
    ) -> None:
        # Broker 是共享的传输层单例，而非每个 Turn 一个。conversation_id 按 Turn 区分，
        # 因此存放在 ContextVar 中。每个 Turn 在自己的通道任务内运行，并发 Turn 无法覆盖它。
        # str 不可变，普通 set/get 已能按任务隔离，无需写时复制。
        self._broker = broker
        self._cid: ContextVar[str] = ContextVar("ask_user_cid", default=conversation_id)

    def set_broker(self, broker: QuestionBroker | None) -> None:
        """设置共享 QuestionBroker；传 ``None`` 会禁用问答 round-trip。

        Broker 属于传输层而非单 Turn，因此直接保存在 Tool 实例。禁用后 `execute` 返回
        ``not configured`` Error，不会在没有接收方时永久等待。方法不迁移已经在旧 Broker 上
        等待的问题。
        """
        self._broker = broker

    def set_context(self, conversation_id: str) -> None:
        """设置当前 Turn 的 ``conversation_id``，作为 Broker 的 turn-local routing key。

        值写入 ContextVar，同一 AskUserTool 被 USER 与 System Turn 并发共享时互不覆盖。调用方
        应在实际运行 Turn 的 asyncio task 内设置；空字符串会让 execute 返回 context Error，
        而不是把问题发送到未知会话。
        """
        self._cid.set(conversation_id)

    @property
    def name(self) -> str:
        return "ask_user"

    @property
    def description(self) -> str:
        return (
            "Ask the user one or more questions and wait for their answer — to gather "
            "a preference, clarify an ambiguous request, or decide a choice with real "
            "trade-offs. Reach for it when the answer genuinely depends on the user; "
            "for low-stakes or reversible choices, pick a sensible default instead. "
            "When you can name a few likely answers, pass them as 'options' (the user "
            "can always type a free-form answer instead); if you recommend one, list "
            "it first with '(Recommended)'. Batch related questions into one call."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": (
                                    "The full, self-contained question to ask. "
                                    "Phrase it so it stands alone — do not repeat "
                                    "it in a separate title."
                                ),
                            },
                            "options": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional list of suggested answers",
                            },
                            "multiple": {
                                "type": "boolean",
                                "description": "Whether multiple options may be chosen",
                            },
                            "custom": {
                                "type": "boolean",
                                "description": "Whether a free-form answer is allowed",
                            },
                        },
                        "required": ["question"],
                    },
                    "description": "One or more questions to ask the user",
                }
            },
            "required": ["questions"],
        }

    async def execute(self, questions: list[dict[str, Any]], **kwargs: Any) -> str:
        cid = self._cid.get()
        if not self._broker:
            return "Error: ask_user not configured (no question broker)"
        if not cid:
            return "Error: ask_user has no conversation context"
        if not questions:
            return "Error: ask_user requires at least one question"

        results: list[str] = []
        for entry in questions:
            question = str(entry.get("question", "")).strip()
            if not question:
                continue
            options = entry.get("options") or []

            answer = await self._broker.await_question(
                cid,
                prompt=question,
                choices=[str(o) for o in options],
            )
            if answer:
                results.append(f'User answered: "{question}" -> "{answer}".')
            else:
                results.append(f'For "{question}": (user did not answer; proceed with best judgment).')

        if not results:
            return "Error: ask_user requires at least one non-empty question"
        return " ".join(results) + " Continue."
