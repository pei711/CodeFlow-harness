"""负责 Curator 对 ``*history`` 的唯一选择、结构闭包和预算裁剪路径。

Session 消息不能按单条任意删除：Assistant 的 ``tool_calls`` 与对应 Tool result 构成协议原子组，
缺一边都会让 Provider 看到 dangling call 或 orphan result。本模块从 :class:`CuratorAssembler`
抽出，让 Curator 与统一 Context Engine 共享同一实现：:meth:`canonical_ids` 补全相邻调用组，
:meth:`history_from_ids` 只保留 Provider-safe keys，:meth:`structural_errors` 验证双向配对，
:meth:`trim` 再按整 Turn 组删除最低优先级且未保护的历史直到预算允许。

这是选择 ``*history`` 的 *only* code path。Segment 6 ``# Curator Working State`` 由
:class:`ContextBuilder` 根据 plan 的 working-state text 渲染，不属于本模块；Trimmer 只通过
``build_messages`` 看到完整固定开销并决定 History，不拥有 System/User Prompt 组成。
最终包含哪些索引、为何删除某组消息以及是否仍然超限，都会作为 Outcome 返回供上层复核。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from codeflow.providers.base import LLMProvider
from codeflow.utils.helpers import estimate_prompt_tokens_chain

# Provider 安全的消息字段。会话消息上的其他字段
# （时间戳、内部 ID、清单标注）会在此前丢弃
# 必须在字典到达 LLM 前移除。reasoning_content / thinking_blocks 必须保留，
# 才能维持多 Turn 推理契约（如 DeepSeek thinking mode）；下游 Provider 门禁
# 会针对非 Anthropic 目标移除 thinking_blocks。
_ALLOWED_KEYS = {
    "role",
    "content",
    "tool_calls",
    "tool_call_id",
    "name",
    "reasoning_content",
    "thinking_blocks",
}


@dataclass
class TrimOutcome:
    """记录一次 :meth:`HistoryTrimmer.trim` 的选择结果与预算证据。

    ``history`` 是清理后的 Provider-safe 消息，``included_ids`` 对应原 Session 索引；
    ``estimated_tokens``、``max_prompt_tokens`` 与 ``source`` 说明估算值、允许上限和估算来源，
    ``warnings`` 记录为适配预算而删除的 Turn 组。`ok` 只判断 Token 是否落在上限内，`over_by`
    给出仍超出的非负数量；二者不证明语义选择正确。
    """

    history: list[dict[str, Any]]
    included_ids: list[int]
    estimated_tokens: int
    max_prompt_tokens: int
    source: str
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.estimated_tokens <= self.max_prompt_tokens

    @property
    def over_by(self) -> int:
        return max(0, self.estimated_tokens - self.max_prompt_tokens)


class HistoryTrimmer:
    """把 Session 候选消息整形成结构合法且尽量符合预算的 ``*history``。

    实例持有 Provider、模型、延迟 Tool definitions 与 Context window，用统一 Token estimate
    评估完整 Prompt。纯整形方法不执行 I/O；`trim` 才反复调用传入的 `build_messages`，以真实
    System、User、Tools 固定开销为准删除历史。Protected Turn 不会被预算算法主动丢弃，因此
    若固定开销和保护内容本身超限，Outcome 会明确 ``ok=False`` 而不是破坏保护边界。
    """

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        context_window_tokens: int,
    ) -> None:
        self.provider = provider
        self.model = model
        self.get_tool_definitions = get_tool_definitions
        self.context_window_tokens = context_window_tokens

    # ------------------------------------------------------------------
    # 纯历史整形辅助函数（不估算 token，也不执行 I/O）
    # ------------------------------------------------------------------

    @staticmethod
    def canonical_ids(messages: list[dict[str, Any]], ids: list[int]) -> list[int]:
        """对 ``ids`` 执行 Tool call/result adjacency closure，并规范起始边界。

        输入中的非法、越界索引先丢弃；选择 Assistant ``tool_calls`` 会自动带上相同 call id 的
        Tool results，选择任一 result 也会补回 Parent Assistant，直到集合稳定。结果按 Session
        原顺序返回，并裁掉第一个 ``role="user"`` 之前的项，确保 History 不从 Tool exchange
        中间开始。没有 User message 存活时返回 ``[]``，不伪造起点。
        """
        selected = {mid for mid in ids if isinstance(mid, int) and 0 <= mid < len(messages)}
        tool_parent_by_call: dict[str, int] = {}
        tool_result_by_call: dict[str, list[int]] = {}
        for idx, message in enumerate(messages):
            if message.get("role") == "assistant" and message.get("tool_calls"):
                for tc in message.get("tool_calls") or []:
                    if isinstance(tc, dict) and tc.get("id"):
                        tool_parent_by_call[str(tc["id"])] = idx
            if message.get("role") == "tool" and message.get("tool_call_id"):
                tool_result_by_call.setdefault(str(message["tool_call_id"]), []).append(idx)

        changed = True
        while changed:
            changed = False
            for call_id, parent_idx in tool_parent_by_call.items():
                result_ids = tool_result_by_call.get(call_id, [])
                if parent_idx in selected:
                    for rid in result_ids:
                        if rid not in selected:
                            selected.add(rid)
                            changed = True
                if any(rid in selected for rid in result_ids) and parent_idx not in selected:
                    selected.add(parent_idx)
                    changed = True

        ordered = sorted(selected)
        for pos, mid in enumerate(ordered):
            if messages[mid].get("role") == "user":
                return ordered[pos:]
        return []

    @staticmethod
    def history_from_ids(messages: list[dict[str, Any]], ids: list[int]) -> list[dict[str, Any]]:
        """把选中 Session 消息投影到 Provider-safe key 集合。

        对每个索引只保留 `_ALLOWED_KEYS` 中的 role、content、Tool 配对与受支持推理字段，移除
        timestamp、内部 ID 和 Manifest 标注；没有 role 的结果不进入 History。返回新字典列表，
        不修改 append-only Session。`thinking_blocks` 是否适合具体目标 Provider 由下游门禁再
        判断，本层必须先保留多 Turn reasoning contract。
        """
        history: list[dict[str, Any]] = []
        for mid in ids:
            clean = {k: v for k, v in messages[mid].items() if k in _ALLOWED_KEYS}
            if clean.get("role"):
                history.append(clean)
        return history

    @staticmethod
    def structural_errors(messages: list[dict[str, Any]]) -> list[str]:
        """验证已组装消息中的 Tool-call closure，并返回全部结构错误。

        Assistant 声明的每个 Tool call id 会进入 open set；Tool result 必须引用其中一个 id，
        否则报告 no parent assistant tool_call，配对成功则关闭该 id。遍历结束仍开放的调用会
        报 missing results。函数不修复顺序、不抛首个异常，调用方可据错误列表拒绝 Curator
        candidate 并允许重试。
        """
        errors: list[str] = []
        open_calls: set[str] = set()
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict) and tc.get("id"):
                        open_calls.add(str(tc["id"]))
            if msg.get("role") == "tool":
                call_id = str(msg.get("tool_call_id", ""))
                if call_id not in open_calls:
                    errors.append(f"tool result {call_id} has no parent assistant tool_call")
                else:
                    open_calls.remove(call_id)
        if open_calls:
            errors.append(f"assistant tool_calls missing results: {sorted(open_calls)}")
        return errors

    @staticmethod
    def _turn_groups(messages: list[dict[str, Any]], ids: list[int]) -> list[list[int]]:
        groups: list[list[int]] = []
        current: list[int] = []
        for mid in ids:
            if messages[mid].get("role") == "user" and current:
                groups.append(current)
                current = []
            current.append(mid)
        if current:
            groups.append(current)
        return groups

    # ------------------------------------------------------------------
    # 预算驱动的裁剪
    # ------------------------------------------------------------------

    def trim(
        self,
        *,
        session_messages: list[dict[str, Any]],
        ids: list[int],
        protected_ids: set[int],
        reserved_output: int,
        build_messages: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
        priority_scores: dict[int, float] | None = None,
    ) -> tuple[list[dict[str, Any]], TrimOutcome]:
        """闭包 ``ids``、构建完整 Prompt，并按整 Turn 组删除直至预算允许或无法再删。

        ``build_messages`` 把 History 映射为 system + history + user 完整列表；Caller 仍拥有
        segments、working state、router skills 等 Prompt composition，Trimmer 只拥有 History
        selection。每轮用 Provider/模型/Tool definitions 估算 Token，允许上限是 Context window
        减 ``reserved_output``，且至少为 1。

        超限时先按 User 边界分 Turn group，排除包含 ``protected_ids`` 的组，再按组内最高
        priority 和新旧顺序选择最低项删除；之后重新执行 canonical closure、构建和估算。
        没有可删组时保留超限事实。返回最终 ``messages`` 与 :class:`TrimOutcome`，warnings
        精确列出被删消息索引。
        """
        canon = self.canonical_ids(session_messages, ids)
        history = self.history_from_ids(session_messages, canon)
        messages = build_messages(history)

        estimated, source = estimate_prompt_tokens_chain(
            self.provider,
            self.model,
            messages,
            self.get_tool_definitions(),
        )
        max_prompt = max(1, self.context_window_tokens - reserved_output)
        warnings: list[str] = []
        trimmed_ids = list(canon)
        while estimated > max_prompt and trimmed_ids:
            groups = [
                group
                for group in self._turn_groups(session_messages, trimmed_ids)
                if not any(mid in protected_ids for mid in group)
            ]
            if not groups:
                break
            scores = priority_scores or {}
            dropped = min(
                groups,
                key=lambda group: (
                    max((scores.get(mid, 0.0) for mid in group), default=0.0),
                    max(group),
                ),
            )
            dropped_set = set(dropped)
            trimmed_ids = [mid for mid in trimmed_ids if mid not in dropped_set]
            trimmed_ids = self.canonical_ids(session_messages, trimmed_ids)
            warnings.append(f"dropped turn messages {dropped} to fit budget")
            history = self.history_from_ids(session_messages, trimmed_ids)
            messages = build_messages(history)
            estimated, source = estimate_prompt_tokens_chain(
                self.provider,
                self.model,
                messages,
                self.get_tool_definitions(),
            )

        return messages, TrimOutcome(
            history=history,
            included_ids=trimmed_ids,
            estimated_tokens=estimated,
            max_prompt_tokens=max_prompt,
            source=source,
            warnings=warnings,
        )


__all__ = ["HistoryTrimmer", "TrimOutcome"]
