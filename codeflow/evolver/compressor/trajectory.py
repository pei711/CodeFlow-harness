"""把 ``session.jsonl`` trajectory 压缩为约 10K-token diagnostic summary。

judge LLM，尤其 mix mode 的 cheap L1-detection backend，无法为每次 analysis 都吞入完整
80-200K-token SWE-bench trajectory。本模块把 raw event stream 转成 structured ``agent
debugger`` overview，保留 high-signal，省略 bulky Tool output。

verbatim 保留 first user task description（external scorer 的 ``WRAPPER_PATH`` + STRICT RULES +
issue text）、每条 assistant text（model reasoning chain），以及 Tool name + truncated arguments。
Tool result 只保留前 ``head_chars`` 与后 ``tail_chars``，中间加 marker，主要影响 long file dump、
stack trace 与 ``cat`` output。重复 ``(tool_name, args-prefix)`` run 用 ``x N times`` marker
折叠，244-paired analysis 中 pager-stuck、tmux-poll、re-read pathology 会显著压缩。

末尾 anomaly section 标记 empty-content Turn count、syntax error、Docker error、repetition
density，正是 L1 detector 关注的 signal。compression 纯 rule-based，不调用 LLM；future v2
可在 rule-compressed output 上再做 LLM summary，但 v1 default 已通常把 150K trajectory 压至
约 5-15K token。token estimate 使用粗略 ``chars / 4``，只供 budget shaping，不引入 tokenizer。

压缩结果不是无损 transcript；它适合 diagnosis prompt，但不能替代原始 evidence、task verdict
或 Tool side-effect receipt。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Union

# ---------------------------------------------------------------------------
# 事件模型
# ---------------------------------------------------------------------------


@dataclass
class Event:
    """external scorer ``session.jsonl`` 中一行解析后的 loose event。

    scorer 会产生 metadata、chat message、Tool result 等多种 shape，因此字段保持宽松，调用方
    按 ``role``/``event_type`` 分支。``raw`` 保存原 object，``tool_calls`` 与可选 Tool name
    保留行为信息。

    assistant row 若携带 LiteLLM/OpenAI-shape ``finish_reason`` 就保留。它是 judge prompt 中
    L1/L2 calibration 的主要 discriminator（spec §22.5 + r4 Fix A1）：empty content 且
    ``stop``/``content_filter`` -> L1；``length`` -> L2，通常是 max_tokens config issue。
    """

    event_type: str  # 取值："metadata" | "system" | "user" | "assistant" | "tool" | "other"
    content: Optional[str] = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_name: Optional[str] = None
    finish_reason: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)


def _parse_event(obj: dict[str, Any]) -> Event:
    """把 ``session.jsonl`` 的 raw JSON object 分类为 ``Event``。

    规则来自 244-paired SWE-bench Session：含 ``_type`` -> metadata；role 为 ``user``、
    ``assistant``、``system``、``tool`` -> chat；其他 -> ``other``，下游忽略。multimodal list
    content 只展平 text block，不压入 image；assistant finish_reason 同时兼容 top-level 与
    ``choices[0].finish_reason``。
    """
    if "_type" in obj:
        return Event(event_type="metadata", raw=obj)
    role = obj.get("role")
    if role in ("system", "user", "assistant", "tool"):
        content = obj.get("content")
        # 一些 scorer/litellm 行把内容存为块列表（多模态）；压缩时将其展平为单个字符串，
        # 但不压缩图片。
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    parts.append(part)
            content = "\n".join(parts) if parts else None
        elif content is not None and not isinstance(content, str):
            content = str(content)
            # 按 OpenAI/LiteLLM 结构，finish_reason 仅对助手行有意义；其他角色即使原始数据中
            # 存在该字段也保留为 None。兼容顶层键（scorer 的 session.jsonl 会展平它）以及
            # 嵌套的 ``choices[0].finish_reason``（某些适配器保留的原始 LiteLLM 响应结构）。
        finish_reason: Optional[str] = None
        if role == "assistant":
            fr = obj.get("finish_reason")
            if isinstance(fr, str) and fr:
                finish_reason = fr
            else:
                choices = obj.get("choices")
                if isinstance(choices, list) and choices:
                    first = choices[0]
                    if isinstance(first, dict):
                        nested = first.get("finish_reason")
                        if isinstance(nested, str) and nested:
                            finish_reason = nested
        return Event(
            event_type=role,
            content=content,
            tool_calls=list(obj.get("tool_calls") or []),
            tool_name=obj.get("name"),
            finish_reason=finish_reason,
            raw=obj,
        )
    return Event(event_type="other", raw=obj)


def load_session_jsonl(path: Union[str, Path]) -> list[Event]:
    """把 ``session.jsonl`` 读取为 :class:`Event` list。

    blank、JSON parse failure 或 non-object line 静默跳过；scorer crash 偶尔会产生 malformed
    line，不能让单行损坏中止 200-Turn trajectory。file open error 仍向上抛出。返回顺序与
    成功解析的原始行一致。
    """
    p = Path(path)
    events: list[Event] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            events.append(_parse_event(obj))
    return events


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass
class CompressorConfig:
    """rule-based compression pass 的参数集合。

    default 调优目标是把 80-200K-token SWE-bench trajectory 压至 5-15K token，低于
    L1-detection backend context；Qwen-397B 为 128K，Claude Haiku 为 200K。``target_tokens``
    是 soft target，不是 hard truncation；其他字段控制 args/result/task 长度、repetition
    threshold 与 anomaly detection。
    """

    target_tokens: int = 10000  # 软目标，而非硬上限
    tool_args_chars: int = 200  # 工具调用参数超过该长度时截断
    tool_result_head_chars: int = 200  # 长工具结果保留开头 N 个字符
    tool_result_tail_chars: int = 200  # 长工具结果保留结尾 N 个字符
    user_message_max_chars: int = 4000  # 任务描述通常不需要更长
    repetition_run_threshold: int = 3  # 连续至少 N 个相同调用时折叠
    detect_anomalies: bool = True


# ---------------------------------------------------------------------------
# 异常标记（压缩正文和最终总结共用）
# ---------------------------------------------------------------------------


_SYNTAX_ERROR_PATTERNS = [
    re.compile(r"\bSyntax(\s+)?Error\b", re.IGNORECASE),
    re.compile(r"Unterminated quoted string", re.IGNORECASE),
    re.compile(r"unexpected EOF", re.IGNORECASE),
]

_DOCKER_ERROR_PATTERNS = [
    re.compile(r"docker daemon", re.IGNORECASE),
    re.compile(r"connection refused", re.IGNORECASE),
    re.compile(r"container .* not running", re.IGNORECASE),
    re.compile(r"\bOOM\b", re.IGNORECASE),
]

_NETWORK_ERROR_PATTERNS = [
    re.compile(r"read timeout", re.IGNORECASE),
    re.compile(r"pool exhausted", re.IGNORECASE),
    re.compile(r"URLError", re.IGNORECASE),
]


def _matches_any(text: str, patterns: list[re.Pattern]) -> bool:
    return any(p.search(text) for p in patterns)


# ---------------------------------------------------------------------------
# 压缩器
# ---------------------------------------------------------------------------


class TrajectoryCompressor:
    """把 Event sequence 转为单一 structured text blob 的 rule-based compressor。

    实例拥有 immutable-use config reference，不保存 trajectory state；每次 ``compress`` 独立
    处理输入。输出面向 judge diagnosis，不是 JSON/Markdown contract。
    """

    def __init__(self, config: Optional[CompressorConfig] = None) -> None:
        self._cfg = config or CompressorConfig()

    def compress(self, events: Iterable[Event]) -> str:
        """为一段 trajectory 生成单个 compressed text block。

        输入先 materialize 为 list。输出是 judge prompt 预期的 plain text layout：header -> task
        -> Turns -> anomaly summary；不是 JSON 或 Markdown，只使用 heading 与 indented body。
        first user message 作为 task，assistant Turn 依次编号，Tool call/result 跟随其 owner。

        完全无 content 且无 Tool call 才计 empty；tool-only Turn 是正常行为。finish_reason 用于
        anomaly calibration。返回字符串不保证达到 ``target_tokens`` hard cap。
        """
        events_list = list(events)
        cfg = self._cfg

        out: list[str] = []
        # 标题——最后再填入轮次和异常计数。
        header_placeholder_idx = len(out)
        out.append("")  # 标题占位符

        # 任务描述：取第一条非元数据用户消息。
        task = self._find_task_description(events_list)
        if task:
            out.append("--- TASK ---")
            truncated = task[: cfg.user_message_max_chars]
            if len(task) > cfg.user_message_max_chars:
                truncated += f"\n... [truncated, {len(task) - cfg.user_message_max_chars} more chars]"
            out.append(truncated)

        # 轮次（助手消息与工具结果配对）
        out.append("\n--- TURNS ---")
        turn_idx = 0
        empty_content_count = 0
        i = 0
        while i < len(events_list):
            ev = events_list[i]
            if ev.event_type != "assistant":
                i += 1
                continue
            turn_idx += 1
            # 空内容检测——L1 信号。``finish_reason`` 用于区分 L1（stop/content_filter）和
            # L2（长度预算对模型过小）；参见评判提示中的修复 A1。
            fr_tag = f", finish_reason={ev.finish_reason}" if ev.finish_reason else ""
            asst_text = (ev.content or "").strip()
            if not asst_text and not ev.tool_calls:
                empty_content_count += 1
                out.append(f"\nTurn {turn_idx} (assistant): [EMPTY content + no tool_calls{fr_tag}]")
                i += 1
                continue
            header_suffix = f" [finish_reason={ev.finish_reason}]" if ev.finish_reason else ""
            out.append(f"\nTurn {turn_idx} (assistant){header_suffix}:")
            if asst_text:
                out.append(self._indent(asst_text, "  "))
            elif ev.tool_calls:
                # 使用工具但没有叙述文本是正常智能体行为（模型选择行动而非解释），不是空内容异常。
                # 不要增加 empty_content_count；L1 触发器必须只用于完全静默的轮次（既无内容也无
                # tool_calls），与 EmptyResponseAlertHook.is_empty_response 保持一致。
                out.append(f"  [tool-only turn, no narrative{fr_tag}]")
            # 查找紧随其后的工具调用和结果，并折叠连续相同调用。
            i += 1
            i = self._emit_tool_section(events_list, i, out, ev)

        # 异常总结
        anomaly_block = []
        if cfg.detect_anomalies:
            anomaly_block = self._compute_anomaly_summary(events_list, empty_content_count)
            if anomaly_block:
                out.append("\n--- ANOMALIES DETECTED ---")
                out.extend(anomaly_block)

        # 已知轮次数后，在顶部补上最终总结行。
        out[header_placeholder_idx] = (
            f"=== TRAJECTORY SUMMARY ===\n"
            f"Total turns: {turn_idx} | Empty-content turns: {empty_content_count} "
            f"| Anomalies flagged: {len(anomaly_block)}"
        )

        return "\n".join(out)

    # -- 辅助方法 -----------------------------------------------------------

    def _find_task_description(self, events: list[Event]) -> Optional[str]:
        """选择 first non-empty user message content 作为 task text。

        找不到时返回 ``None``，compressor 省略 TASK section。
        """
        for ev in events:
            if ev.event_type == "user" and ev.content:
                return ev.content
        return None

    def _emit_tool_section(
        self,
        events: list[Event],
        start_idx: int,
        out: list[str],
        assistant: Event,
    ) -> int:
        """输出 assistant Turn 后的 pending Tool call 与 result section。

        从 ``events[start_idx:]`` 消费连续 ``role=tool`` event，它们是前一 assistant Tool call
        的 result。每个 declared call 输出 name 与 truncated args，每个 result 使用 head/tail
        summary。assistant 未声明 Tool call 时跳过紧随的 orphan Tool event。

        返回 Tool section 结束 index，供 outer loop 继续。设计目标包含对相同
        ``(name, args-prefix)`` run 的 ``x N repetitions`` annotation；当前实现逐 call 输出，
        repetition density 仍在 anomaly summary 汇总。
        """
        cfg = self._cfg
        i = start_idx

        if not assistant.tool_calls:
            # 未声明工具调用——跳过游离的工具事件（少见）。
            while i < len(events) and events[i].event_type == "tool":
                i += 1
            return i

        # 输出每次工具调用，并截断参数。
        for tc in assistant.tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name", "?")
            args_raw = fn.get("arguments", "") or ""
            if not isinstance(args_raw, str):
                args_raw = json.dumps(args_raw, ensure_ascii=False)
            args_trunc = args_raw[: cfg.tool_args_chars]
            if len(args_raw) > cfg.tool_args_chars:
                args_trunc += f" [...truncated {len(args_raw) - cfg.tool_args_chars} chars]"
            out.append(f"  → call {name}({args_trunc})")

        # 接着消费对应的工具结果事件。
        while i < len(events) and events[i].event_type == "tool":
            result = events[i].content or ""
            out.append(f"  ← result: {self._summarize_tool_result(result)}")
            i += 1

        return i

    def _summarize_tool_result(self, content: str) -> str:
        cfg = self._cfg
        if not content:
            return "[empty result]"
        total = len(content)
        head_n = cfg.tool_result_head_chars
        tail_n = cfg.tool_result_tail_chars
        if total <= head_n + tail_n:
            return self._oneline(content)
        head = content[:head_n]
        tail = content[-tail_n:]
        return f"{self._oneline(head)} ... [ELIDED {total - head_n - tail_n} chars] ... {self._oneline(tail)}"

    @staticmethod
    def _oneline(text: str) -> str:
        """折叠 newline 与 repeated whitespace，使 Tool result 落在单行。"""
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _indent(text: str, prefix: str) -> str:
        return "\n".join(prefix + line for line in text.splitlines())

    def _compute_anomaly_summary(self, events: list[Event], empty_content_count: int) -> list[str]:
        cfg = self._cfg
        out: list[str] = []

        # 1. 空内容比例（区分 L1 与 L2 的关键信号）。finish_reason 分布可消除歧义：
        # ``stop`` 表示真正的 L1（模型静默），``length`` 表示 L2（max_tokens 预算对模型过小，
        # 推理模型耗尽预算却未产生输出）。
        n_asst = sum(1 for e in events if e.event_type == "assistant")
        if n_asst > 0 and empty_content_count > 0:
            pct = empty_content_count / n_asst * 100
            severity = "HIGH" if pct > 30 else "MEDIUM" if pct > 10 else "LOW"
            fr_breakdown = Counter(
                (e.finish_reason or "unknown")
                for e in events
                if e.event_type == "assistant" and not (e.content or "").strip() and not e.tool_calls
            )
            fr_tag = ""
            if fr_breakdown:
                parts = [f"{k}={v}" for k, v in sorted(fr_breakdown.items())]
                fr_tag = f" (by finish_reason: {', '.join(parts)})"
            out.append(
                f"  [{severity}] empty-content assistant turns: {empty_content_count}/{n_asst}"
                f" ({pct:.0f}%){fr_tag} — L1 if finish_reason=stop/content_filter dominates,"
                f" L2 if finish_reason=length dominates (max_tokens budget too small)"
            )

        # 2. 工具输出中的语法、Docker 或网络错误。
        syntax_errs = 0
        docker_errs = 0
        net_errs = 0
        for ev in events:
            if ev.event_type != "tool" or not ev.content:
                continue
            if _matches_any(ev.content, _SYNTAX_ERROR_PATTERNS):
                syntax_errs += 1
            if _matches_any(ev.content, _DOCKER_ERROR_PATTERNS):
                docker_errs += 1
            if _matches_any(ev.content, _NETWORK_ERROR_PATTERNS):
                net_errs += 1
        if syntax_errs:
            out.append(f"  [LOW] {syntax_errs} tool result(s) contained syntax-error markers")
        if docker_errs:
            out.append(f"  [HIGH] {docker_errs} tool result(s) contained docker/container errors — L1 signal")
        if net_errs:
            out.append(f"  [HIGH] {net_errs} tool result(s) contained network errors — L1 signal")

        # 3. 工具调用重复密度。
        tool_call_sigs: list[tuple[str, str]] = []
        for ev in events:
            if ev.event_type != "assistant":
                continue
            for tc in ev.tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name", "?")
                args = fn.get("arguments", "") or ""
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)
                sig = (name, args[: cfg.tool_args_chars])
                tool_call_sigs.append(sig)
        if tool_call_sigs:
            c = Counter(tool_call_sigs)
            most_common_sig, most_common_n = c.most_common(1)[0]
            if most_common_n >= cfg.repetition_run_threshold:
                args_preview = most_common_sig[1][:80].replace("\n", " ")
                out.append(
                    f"  [MEDIUM] tool call repeated {most_common_n}× "
                    f"({most_common_sig[0]}, args~{args_preview!r}) "
                    f"— possible repetition_breaker pathology"
                )

        return out


# ---------------------------------------------------------------------------
# 快速令牌估算辅助方法
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """用 ``len(text) / 4`` 粗估 token count。

    精度只适合 1K-100K scale 的 budget shaping，do NOT 用于 billing/SLA decision。使用
    ``ceil`` semantics，所以 5-character string 报告 2 而非 1；empty string 返回 0。
    """
    if not text:
        return 0
    return -(-len(text) // 4)


__all__ = [
    "CompressorConfig",
    "Event",
    "TrajectoryCompressor",
    "estimate_tokens",
    "load_session_jsonl",
]
