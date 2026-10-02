"""定义唯一受限信任 driver model 的 SemanticNode contract。

A weak driver (Qwen / Kimi) follows instructions loosely and gives up early, so
every semantic step is reduced to a single call whose output must parse into a
fixed schema. When it doesn't, the node feeds the parse error back and retries a
bounded number of times before giving up — the same "retry, fix this" loop the
judge parser docstring anticipates, made explicit and reusable.

This is deliberately transport-agnostic: ``call_fn`` takes chat messages and
returns the assistant text. In production it wraps a
``codeflow.evolver.judge.llm_client`` backend (which already routes self-hosted
Qwen, Claude, and OpenRouter); in tests it is a plain function returning scripted
strings. ``parse_fn`` turns raw text into the target schema object and raises on
any defect; ``SemanticNode`` catches that, appends a repair turn, and retries.

The node is synchronous to match the orchestrator FSM. An async backend is
adapted by the caller (``asyncio.run`` in the production ``call_fn``); keeping the
retry logic sync，避免把 event loop 穿透整个 FSM。parse success 只表示 schema object 合法。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Sequence, TypeVar

T = TypeVar("T")

Messages = list[dict[str, str]]
CallFn = Callable[[Messages], str]
ParseFn = Callable[[str], T]


class SemanticNodeError(RuntimeError):
    """SemanticNode 在 retry budget 内仍无法生成 valid object 时抛出的异常。"""

    def __init__(self, name: str, attempts: int, last_error: Exception) -> None:
        super().__init__(
            f"semantic node {name!r} failed to parse after {attempts} attempt(s); last error: {last_error!r}"
        )
        self.name = name
        self.attempts = attempts
        self.last_error = last_error


def default_repair_prompt(error: Exception) -> str:
    """parse failure 后追加的 user repair turn，要求只输出 valid object。"""
    return (
        "Your previous response could not be parsed into the required format. "
        f"Error: {error}. Respond again with ONLY the valid object, no prose, "
        "no code fences."
    )


@dataclass
class SemanticNode(Generic[T]):
    """一次 schema-validated driver call 及其 bounded repair-retry 生命周期。"""

    name: str
    call_fn: CallFn
    parse_fn: ParseFn
    max_retries: int = 3
    parse_error_types: tuple[type[Exception], ...] = (Exception,)
    repair_prompt: Callable[[Exception], str] = default_repair_prompt

    def run(self, messages: Sequence[dict[str, str]]) -> T:
        """调用 driver 并 parse schema，最多 repair ``max_retries`` 次。

        Returns the parsed object on the first success. Raises
        :class:`SemanticNodeError` if every attempt (initial + retries) fails to
        parse. The raw text of each attempt is preserved in the conversation so
        model 看到自己的 bad output 与 error。耗尽后抛 ``SemanticNodeError``。
        """
        convo: Messages = [dict(m) for m in messages]
        last_error: Exception | None = None
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            raw = self.call_fn(convo)
            try:
                return self.parse_fn(raw)
            except self.parse_error_types as exc:
                last_error = exc
                if attempt == attempts - 1:
                    break
                convo = convo + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": self.repair_prompt(exc)},
                ]
        assert last_error is not None  # 循环至少运行一次
        raise SemanticNodeError(self.name, attempts, last_error)


__all__ = [
    "SemanticNode",
    "SemanticNodeError",
    "default_repair_prompt",
    "CallFn",
    "ParseFn",
    "Messages",
]
