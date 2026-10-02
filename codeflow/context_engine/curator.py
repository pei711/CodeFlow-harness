"""实现为主 Agent 准备下一轮 Context window 的内部 Curator。

Curator 是有迭代上限的内部 Agent Loop，但它既不执行面向用户的 Tool，也不回答用户。它只
能查看压缩 manifest、归档或取回历史消息、调整 relevance 与 working state，最后提交结构化
`ContextPlan`。因此它拥有“选择哪些历史进入窗口”的提案权，不拥有主请求执行与交付权。

`CuratorArchiveStore` 保存 manifest、archives、working state 和 trace；`CuratorAssembler`
再以确定性规则验证计划、维护 Tool call 配对并按精确预算裁剪。只有该 Assembler 产出的
provider-safe messages 会交给主 Agent，Curator 的自然语言判断不能越过预算和结构门禁。
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from codeflow.agent.tools.base import Tool
from codeflow.config.codeflow import ContextConfig
from codeflow.context_engine.base import AssembledPrefix
from codeflow.context_engine.history_trimmer import HistoryTrimmer
from codeflow.memory_engine.base import AssembledContext, TokenBudget
from codeflow.memory_engine.consolidate.consolidator import MemoryStore
from codeflow.providers.base import LLMProvider
from codeflow.spine.message import Media
from codeflow.utils.helpers import (
    ensure_dir,
    estimate_message_tokens,
    safe_filename,
)
from codeflow.utils.persisted_payload import sanitize_persisted_payload, sanitize_persisted_text


def _persisted_json(value: Any, *, indent: int | None = None) -> str:
    serialized = json.dumps(
        sanitize_persisted_payload(value),
        ensure_ascii=False,
        indent=indent,
        default=str,
    )
    return sanitize_persisted_text(serialized)


@dataclass
class TurnContext:
    """汇集构建主 Agent 本轮 Context 所需的当前请求输入。

    ``current_message`` 是本轮用户文本，``media`` 携带路径或已验证 Media，``channel`` 与
    ``chat_id`` 用于运行时上下文，``selected_skills`` 保留兼容的预选 Skill 槽位。它不包含
    Session History 或 TokenBudget；前者由 Engine 单独传入，后者由 Host 计算，避免把当前
    请求载荷与历史所有权混在一个对象里。
    """

    current_message: str
    media: list[str | Media] | None = None
    channel: str | None = None
    chat_id: str | None = None
    selected_skills: list[Any] | None = None


@dataclass
class ManifestItem:
    id: int
    role: str
    ts: str | None
    tokens: int
    turn_id: int
    group_id: str | None
    snippet: str
    summary: str
    keywords: list[str] = field(default_factory=list)
    relevance: float = 0.5
    protected: bool = False
    archived: bool = False
    archive_ref: str | None = None


@dataclass
class ContextPlan:
    include_message_ids: list[int] = field(default_factory=list)
    include_archive_refs: list[str] = field(default_factory=list)
    memory_sections: list[str] = field(default_factory=list)
    working_state_injection: str = ""
    drop_message_ids: list[int] = field(default_factory=list)
    notes: str = ""


@dataclass
class CuratorState:
    session_key: str
    session_messages: list[dict[str, Any]]
    budget: TokenBudget
    turn: TurnContext
    manifest: list[ManifestItem]
    prefix: AssembledPrefix
    final_plan: ContextPlan | None = None
    final_validation: dict[str, Any] | None = None


class CuratorArchiveStore:
    """在磁盘上管理 Curator manifest、archives、working state 与 traces。

    每个 ``session_key`` 经安全文件名映射到独立 manifest 和 working-state 文件，Archive 按
    日期写 JSONL，Trace 按 Turn 追加。Store 负责序列化清理、路径组织和 relevance 持久化，
    不决定哪些消息应进入最终 Context；选择由 Curator plan 提出，结构与预算由 Assembler
    验证。

    读取缺失或损坏状态时返回受限空结构，写入时限制 goals、open_threads、decisions 数量并
    清理 payload。归档会降低对应 ManifestItem relevance 并保存 archive_ref，但原始 Session
    append-only log 仍由 Session 所有，本 Store 不把归档文件冒充会话事实源。
    """

    def __init__(self, workspace: Path, config: ContextConfig, now_fn: Callable[[], datetime] | None = None):
        self.workspace = workspace
        self.config = config
        self.root = ensure_dir(workspace / config.archive_dir).parent
        self.archive_dir = ensure_dir(workspace / config.archive_dir)
        self.manifest_dir = ensure_dir(self.root / "manifest")
        self.state_dir = ensure_dir(self.root / "working_state")
        self.trace_dir = ensure_dir(self.root / "traces")
        self._now_fn = now_fn or datetime.now

    def _safe_session(self, session_key: str) -> str:
        return safe_filename(session_key.replace(":", "_")) or "session"

    def manifest_path(self, session_key: str) -> Path:
        return self.manifest_dir / f"{self._safe_session(session_key)}.json"

    def state_path(self, session_key: str) -> Path:
        return self.state_dir / f"{self._safe_session(session_key)}.json"

    def trace_path(self, session_key: str, turn_id: str) -> Path:
        session_dir = ensure_dir(self.trace_dir / self._safe_session(session_key))
        return session_dir / f"{turn_id}.jsonl"

    def append_trace(self, session_key: str, turn_id: str, event: str, payload: dict[str, Any]) -> None:
        record = {
            "ts": self._now_fn().isoformat(),
            "event": event,
            "payload": payload,
        }
        path = self.trace_path(session_key, turn_id)
        with path.open("a", encoding="utf-8") as f:
            f.write(_persisted_json(record) + "\n")

    def read_working_state(self, session_key: str) -> dict[str, Any]:
        path = self.state_path(session_key)
        if not path.exists():
            return {"goals": [], "open_threads": [], "decisions": []}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"goals": [], "open_threads": [], "decisions": []}
        return data if isinstance(data, dict) else {"goals": [], "open_threads": [], "decisions": []}

    def write_working_state(self, session_key: str, state: dict[str, Any]) -> dict[str, Any]:
        compact = sanitize_persisted_payload(
            {
                "goals": _limit_str_list(state.get("goals", []), 8),
                "open_threads": _limit_str_list(state.get("open_threads", []), 12),
                "decisions": _limit_str_list(state.get("decisions", []), 20),
                "updated_at": self._now_fn().isoformat(),
            }
        )
        self.state_path(session_key).write_text(_persisted_json(compact, indent=2), encoding="utf-8")
        return compact

    def load_relevance(self, session_key: str) -> dict[int, dict[str, Any]]:
        path = self.manifest_path(session_key)
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        items = raw.get("items", []) if isinstance(raw, dict) else []
        out: dict[int, dict[str, Any]] = {}
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("id"), int):
                out[item["id"]] = item
        return out

    def save_manifest(self, session_key: str, manifest: list[ManifestItem]) -> None:
        payload = {
            "session_key": session_key,
            "updated_at": self._now_fn().isoformat(),
            "items": [asdict(item) for item in manifest],
        }
        self.manifest_path(session_key).write_text(_persisted_json(payload, indent=2), encoding="utf-8")

    def build_manifest(self, session_key: str, messages: list[dict[str, Any]]) -> list[ManifestItem]:
        previous = self.load_relevance(session_key)
        manifest: list[ManifestItem] = []
        turn_id = 0
        for idx, message in enumerate(messages):
            if message.get("role") == "user":
                turn_id += 1
            group_id: str | None = None
            if message.get("role") == "assistant" and message.get("tool_calls"):
                call_ids = []
                for tc in message.get("tool_calls") or []:
                    call_id = tc.get("id") if isinstance(tc, dict) else None
                    if call_id:
                        call_ids.append(str(call_id))
                group_id = "tool:" + ",".join(call_ids) if call_ids else None
            elif message.get("role") == "tool" and message.get("tool_call_id"):
                group_id = f"tool:{message.get('tool_call_id')}"

            content_text = _message_text(message)
            prior = previous.get(idx, {})
            archived = bool(prior.get("archived", False))
            archive_ref = prior.get("archive_ref") if isinstance(prior.get("archive_ref"), str) else None
            relevance = float(prior.get("relevance", 0.5))
            if idx >= max(0, len(messages) - 12):
                relevance = max(relevance, 0.75)
            if idx < self.config.protect_first_n * 2:
                relevance = max(relevance, 0.8)

            manifest.append(
                ManifestItem(
                    id=idx,
                    role=str(message.get("role", "")),
                    ts=message.get("timestamp") if isinstance(message.get("timestamp"), str) else None,
                    tokens=estimate_message_tokens(message),
                    turn_id=turn_id,
                    group_id=group_id,
                    snippet=_snippet(content_text, 240),
                    summary=_snippet(content_text, 420),
                    keywords=_keywords(content_text),
                    relevance=min(1.0, max(0.0, relevance)),
                    protected=idx < self.config.protect_first_n * 2,
                    archived=archived,
                    archive_ref=archive_ref,
                )
            )
        self.save_manifest(session_key, manifest)
        return manifest

    def archive_messages(
        self,
        session_key: str,
        manifest: list[ManifestItem],
        messages: list[dict[str, Any]],
        message_ids: list[int],
        reason: str = "",
        tags: list[str] | None = None,
        summary: str = "",
    ) -> dict[str, Any]:
        valid_ids = sorted({mid for mid in message_ids if 0 <= mid < len(messages)})
        if not valid_ids:
            return {"archived_message_ids": [], "archive_refs": [], "bytes_written": 0, "manifest_updated": False}

        date_dir = ensure_dir(self.archive_dir / self._now_fn().strftime("%Y-%m-%d"))
        first, last = valid_ids[0], valid_ids[-1]
        archive_path = date_dir / f"{self._safe_session(session_key)}_{first}_{last}_{uuid.uuid4().hex[:8]}.jsonl"
        with archive_path.open("w", encoding="utf-8") as f:
            header = {
                "_type": "curator_archive",
                "session_key": session_key,
                "reason": reason,
                "tags": tags or [],
                "summary": summary,
                "created_at": self._now_fn().isoformat(),
            }
            f.write(_persisted_json(header) + "\n")
            for mid in valid_ids:
                f.write(_persisted_json({"id": mid, "message": messages[mid]}) + "\n")

        rel = archive_path.relative_to(self.workspace)
        ref = f"{rel}#{first}-{last}"
        archived_set = set(valid_ids)
        for item in manifest:
            if item.id in archived_set:
                item.archived = True
                item.archive_ref = ref
                item.relevance = min(item.relevance, 0.45)
        self.save_manifest(session_key, manifest)
        return {
            "archived_message_ids": valid_ids,
            "archive_refs": [ref],
            "bytes_written": archive_path.stat().st_size,
            "manifest_updated": True,
        }

    def retrieve(self, refs: list[str], max_tokens: int = 2000, mode: str = "snippet") -> dict[str, Any]:
        results = []
        budget = max(1, max_tokens)
        for ref in refs:
            path_part = ref.split("#", 1)[0]
            path = self.workspace / path_part
            if not path.exists():
                results.append({"archive_ref": ref, "error": "not found", "messages": []})
                continue
            returned_tokens = 0
            messages = []
            try:
                lines = path.read_text(encoding="utf-8").splitlines()[1:]
                for line in lines:
                    record = json.loads(line)
                    msg = record.get("message", {})
                    tokens = estimate_message_tokens(msg)
                    if returned_tokens + tokens > budget:
                        break
                    payload = {
                        "id": record.get("id"),
                        "role": msg.get("role"),
                        "tokens": tokens,
                        "content": msg.get("content") if mode == "exact" else _snippet(_message_text(msg), 600),
                    }
                    messages.append(payload)
                    returned_tokens += tokens
            except (OSError, json.JSONDecodeError) as exc:
                results.append({"archive_ref": ref, "error": str(exc), "messages": []})
                continue
            results.append(
                {
                    "archive_ref": ref,
                    "messages": messages,
                    "returned_tokens": returned_tokens,
                    "truncated": returned_tokens >= budget,
                }
            )
        return {"results": results}

    def set_relevance(
        self, session_key: str, manifest: list[ManifestItem], updates: list[dict[str, Any]]
    ) -> dict[str, Any]:
        by_id = {item.id: item for item in manifest}
        accepted: list[int] = []
        rejected: list[dict[str, Any]] = []
        for update in updates:
            mid = update.get("message_id")
            if not isinstance(mid, int) or mid not in by_id:
                rejected.append({"message_id": mid, "reason": "unknown id"})
                continue
            value = update.get("relevance", by_id[mid].relevance)
            try:
                by_id[mid].relevance = min(1.0, max(0.0, float(value)))
            except (TypeError, ValueError):
                rejected.append({"message_id": mid, "reason": "invalid relevance"})
                continue
            tags = update.get("tags")
            if isinstance(tags, list):
                by_id[mid].keywords = sorted(set(by_id[mid].keywords + [str(t) for t in tags[:8]]))
            accepted.append(mid)
        self.save_manifest(session_key, manifest)
        return {"accepted": accepted, "rejected": rejected}


class CuratorAssembler:
    """验证 Curator plan，并构建 Provider 可接受且不超预算的消息列表。

    这是 prefix-based Assembler：System Prompt 的 segments 1–5 已由 :class:`ContextAssembler`
    完成，并作为每 Turn :class:`CuratorState` 上的 :class:`AssembledPrefix` 传入。本对象只从
    plan 追加 segment 6 ``# Curator Working State``、选择 ``*history``，再让 `HistoryTrimmer`
    对完整 fixed overhead（prefix + seg6 + user + tools）做预算裁剪。

    `build` 返回 `AssembledContext` 与 validation 证据；`validate_candidate` 还检查 Provider
    消息结构，失败时允许 Curator 重试。若内部 Agent 没有给出可用计划，`fallback_plan` 用
    protected、relevant、recent 三组消息产生 deterministic fallback，而不是跳过门禁。
    """

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        context_window_tokens: int,
    ):
        self.provider = provider
        self.model = model
        self.get_tool_definitions = get_tool_definitions
        self.context_window_tokens = context_window_tokens
        self.trimmer = HistoryTrimmer(
            provider,
            model,
            get_tool_definitions,
            context_window_tokens,
        )

    def replace_model(self, model: str) -> None:
        self.model = model
        self.trimmer.model = model

    @staticmethod
    def working_state_segment(working_state: str | None) -> str:
        """把非空 working state 渲染为 segment 6 ``# Curator Working State``。

        输入先去除首尾空白；有内容时加固定标题并返回 System Segment 文本，无内容时精确返回
        ``""``。该函数不解释或补全 state，也不把空标题加入 Prompt，因此是否注入完全由
        ContextPlan 的实际内容决定。
        """
        ws = (working_state or "").strip()
        return f"# Curator Working State\n\n{ws}" if ws else ""

    def _full_messages(
        self,
        prefix: AssembledPrefix,
        history: list[dict[str, Any]],
        working_state: str | None,
    ) -> list[dict[str, Any]]:
        system = prefix.system_prefix
        seg6 = self.working_state_segment(working_state)
        if seg6:
            system = system + "\n\n---\n\n" + seg6
        return [{"role": "system", "content": system}, *history, prefix.user_message]

    def build(
        self,
        state: CuratorState,
        plan: ContextPlan,
    ) -> tuple[AssembledContext, dict[str, Any]]:
        working_state = plan.working_state_injection or None
        protected_ids = {item.id for item in state.manifest if item.protected}

        messages, outcome = self.trimmer.trim(
            session_messages=state.session_messages,
            ids=plan.include_message_ids,
            protected_ids=protected_ids,
            priority_scores={item.id: item.relevance for item in state.manifest},
            reserved_output=state.budget.reserved_output,
            build_messages=lambda h: self._full_messages(state.prefix, h, working_state),
        )

        validation = {
            "ok": outcome.ok,
            "total_tokens": outcome.estimated_tokens,
            "max_prompt_tokens": outcome.max_prompt_tokens,
            "over_by": outcome.over_by,
            "source": outcome.source,
            "included_message_ids": outcome.included_ids,
            "assembler_warnings": outcome.warnings,
        }
        return AssembledContext(
            messages=messages,
            include_indices=outcome.included_ids,
            metadata={
                "engine": "context_assembler",
                "plan": asdict(plan),
                "validation": validation,
            },
        ), validation

    def validate_candidate(self, state: CuratorState, plan: ContextPlan) -> dict[str, Any]:
        assembled, validation = self.build(state, plan)
        errors = self.trimmer.structural_errors(assembled.messages)
        validation["ok"] = bool(validation["ok"] and not errors)
        validation["errors"] = errors
        validation["retry_allowed"] = not validation["ok"]
        return validation

    def fallback_plan(self, state: CuratorState) -> ContextPlan:
        protected = [item.id for item in state.manifest if item.protected]
        recent = [item.id for item in state.manifest[-16:]]
        ranked = sorted(
            state.manifest,
            key=lambda item: (item.relevance, item.id),
            reverse=True,
        )[:12]
        ranked_ids = [item.id for item in ranked]
        ids = sorted(set(protected + ranked_ids + recent))
        return ContextPlan(
            include_message_ids=ids,
            memory_sections=["Long-term Memory", "Working State"],
            working_state_injection="Curator fallback selected protected, relevant, and recent messages.",
            notes="deterministic fallback",
        )


class _CuratorTool(Tool):
    def _json(self, payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, default=str)


class CuratorCheckBudgetTool(_CuratorTool):
    def __init__(self, state: CuratorState, assembler: CuratorAssembler):
        self.state = state
        self.assembler = assembler

    @property
    def name(self) -> str:
        return "curator_check_budget"

    @property
    def description(self) -> str:
        return "Check whether a proposed context plan fits the token budget and message-structure rules."

    @property
    def parameters(self) -> dict[str, Any]:
        return _plan_schema(required=False)

    async def execute(self, **kwargs: Any) -> str:
        plan = _plan_from_kwargs(kwargs)
        return self._json(self.assembler.validate_candidate(self.state, plan))


class CuratorArchiveMessagesTool(_CuratorTool):
    def __init__(self, state: CuratorState, archive: CuratorArchiveStore):
        self.state = state
        self.archive = archive

    @property
    def name(self) -> str:
        return "curator_archive_messages"

    @property
    def description(self) -> str:
        return "Losslessly archive selected old session messages to disk and update the manifest."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message_ids": {"type": "array", "items": {"type": "integer"}},
                "reason": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "summary": {"type": "string"},
            },
            "required": ["message_ids"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return self._json(
            self.archive.archive_messages(
                self.state.session_key,
                self.state.manifest,
                self.state.session_messages,
                kwargs.get("message_ids", []),
                kwargs.get("reason", ""),
                kwargs.get("tags", []),
                kwargs.get("summary", ""),
            )
        )


class CuratorRetrieveArchivedTool(_CuratorTool):
    def __init__(self, archive: CuratorArchiveStore):
        self.archive = archive

    @property
    def name(self) -> str:
        return "curator_retrieve_archived"

    @property
    def description(self) -> str:
        return "Retrieve archived messages by archive_ref with an output token cap."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "archive_refs": {"type": "array", "items": {"type": "string"}},
                "mode": {"type": "string", "enum": ["snippet", "exact"]},
                "max_tokens": {"type": "integer", "minimum": 1, "maximum": 12000},
            },
            "required": ["archive_refs"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return self._json(
            self.archive.retrieve(
                kwargs.get("archive_refs", []),
                max_tokens=kwargs.get("max_tokens", 2000),
                mode=kwargs.get("mode", "snippet"),
            )
        )


class CuratorSearchHistoryTool(_CuratorTool):
    def __init__(self, state: CuratorState):
        self.state = state

    @property
    def name(self) -> str:
        return "curator_search_history"

    @property
    def description(self) -> str:
        return "Search the compact manifest by keywords and relevance without loading full history."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
                "include_archived": {"type": "boolean"},
                "min_relevance": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        terms = set(_keywords(str(kwargs.get("query", ""))))
        top_k = int(kwargs.get("top_k", 8))
        include_archived = bool(kwargs.get("include_archived", True))
        min_relevance = float(kwargs.get("min_relevance", 0.0))
        hits = []
        for item in self.state.manifest:
            if item.archived and not include_archived:
                continue
            if item.relevance < min_relevance:
                continue
            haystack = set(item.keywords) | set(_keywords(item.snippet + " " + item.summary))
            overlap = len(terms & haystack)
            if not overlap and terms:
                continue
            score = overlap + item.relevance + (0.25 if item.id >= len(self.state.manifest) - 12 else 0)
            hits.append(
                {
                    "message_id": item.id,
                    "archive_ref": item.archive_ref,
                    "score": round(score, 4),
                    "tokens": item.tokens,
                    "snippet": item.snippet,
                    "reason": "keyword/relevance match",
                }
            )
        hits.sort(key=lambda h: (h["score"], h["message_id"]), reverse=True)
        return self._json({"hits": hits[:top_k]})


class CuratorReadMemoryTool(_CuratorTool):
    def __init__(self, state: CuratorState, archive: CuratorArchiveStore, memory: MemoryStore):
        self.state = state
        self.archive = archive
        self.memory = memory

    @property
    def name(self) -> str:
        return "curator_read_memory"

    @property
    def description(self) -> str:
        return "Read bounded long-term memory and Curator working state."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "sections": {"type": "array", "items": {"type": "string"}},
                "max_tokens": {"type": "integer", "minimum": 1, "maximum": 12000},
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        max_chars = int(kwargs.get("max_tokens", 2500)) * 4
        memory = self.memory.read_long_term()
        working_state = self.archive.read_working_state(self.state.session_key)
        return self._json(
            {
                "sections": [
                    {
                        "name": "Long-term Memory",
                        "content": memory[:max_chars],
                        "truncated": len(memory) > max_chars,
                    },
                    {
                        "name": "Working State",
                        "content": json.dumps(working_state, ensure_ascii=False),
                        "truncated": False,
                    },
                ]
            }
        )


class CuratorSetRelevanceTool(_CuratorTool):
    def __init__(self, state: CuratorState, archive: CuratorArchiveStore):
        self.state = state
        self.archive = archive

    @property
    def name(self) -> str:
        return "curator_set_relevance"

    @property
    def description(self) -> str:
        return "Update compact relevance metadata for messages in this session."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "updates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "message_id": {"type": "integer"},
                            "relevance": {"type": "number", "minimum": 0, "maximum": 1},
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "reason": {"type": "string"},
                        },
                        "required": ["message_id", "relevance"],
                    },
                }
            },
            "required": ["updates"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return self._json(
            self.archive.set_relevance(
                self.state.session_key,
                self.state.manifest,
                kwargs.get("updates", []),
            )
        )


class CuratorUpdateWorkingStateTool(_CuratorTool):
    def __init__(self, state: CuratorState, archive: CuratorArchiveStore):
        self.state = state
        self.archive = archive

    @property
    def name(self) -> str:
        return "curator_update_working_state"

    @property
    def description(self) -> str:
        return "Update compact goals, open threads, and decisions for this session."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "goals": {"type": "array", "items": {"type": "string"}},
                "open_threads": {"type": "array", "items": {"type": "string"}},
                "decisions": {"type": "array", "items": {"type": "string"}},
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        return self._json(
            {
                "working_state": self.archive.write_working_state(self.state.session_key, kwargs),
            }
        )


class CuratorBuildContextTool(_CuratorTool):
    def __init__(self, state: CuratorState, assembler: CuratorAssembler):
        self.state = state
        self.assembler = assembler

    @property
    def name(self) -> str:
        return "curator_build_context"

    @property
    def description(self) -> str:
        return "Submit the final context plan. This must be the last Curator action when accepted."

    @property
    def parameters(self) -> dict[str, Any]:
        return _plan_schema(required=True)

    async def execute(self, **kwargs: Any) -> str:
        plan = _plan_from_kwargs(kwargs)
        validation = self.assembler.validate_candidate(self.state, plan)
        if validation.get("ok"):
            self.state.final_plan = plan
            self.state.final_validation = validation
            return self._json(
                {
                    "accepted": True,
                    "final_total_tokens": validation.get("total_tokens"),
                    "included_message_ids": validation.get("included_message_ids"),
                    "assembler_warnings": validation.get("assembler_warnings", []),
                }
            )
        return self._json(
            {
                "accepted": False,
                "errors": validation.get("errors", [])
                + ([f"budget exceeded by {validation.get('over_by')} tokens"] if validation.get("over_by") else []),
                "validation": validation,
                "retry_allowed": True,
            }
        )


def _plan_schema(required: bool) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "include_message_ids": {"type": "array", "items": {"type": "integer"}},
            "include_archive_refs": {"type": "array", "items": {"type": "string"}},
            "memory_sections": {"type": "array", "items": {"type": "string"}},
            "working_state_injection": {"type": "string"},
            "drop_message_ids": {"type": "array", "items": {"type": "integer"}},
            "notes": {"type": "string"},
        },
        "required": ["include_message_ids"] if required else [],
    }


def _plan_from_kwargs(kwargs: dict[str, Any]) -> ContextPlan:
    return ContextPlan(
        include_message_ids=[
            int(x)
            for x in kwargs.get("include_message_ids", [])
            if isinstance(x, (int, str)) and str(x).lstrip("-").isdigit()
        ],
        include_archive_refs=[str(x) for x in kwargs.get("include_archive_refs", [])],
        memory_sections=[str(x) for x in kwargs.get("memory_sections", [])],
        working_state_injection=str(kwargs.get("working_state_injection", "") or ""),
        drop_message_ids=[
            int(x)
            for x in kwargs.get("drop_message_ids", [])
            if isinstance(x, (int, str)) and str(x).lstrip("-").isdigit()
        ],
        notes=str(kwargs.get("notes", "") or ""),
    )


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, dict) and item.get("type") == "image_url":
                parts.append("[image]")
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(parts)
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def _snippet(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _keywords(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9_\-]{3,}", text.lower())
    seen: list[str] = []
    stop = {"the", "and", "for", "with", "that", "this", "from", "into", "you", "are"}
    for word in words:
        if word in stop or word in seen:
            continue
        seen.append(word)
        if len(seen) >= 16:
            break
    return seen


def _limit_str_list(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_snippet(str(item), 400) for item in value[:limit]]


def _curator_input_payload(state: CuratorState, archive: CuratorArchiveStore) -> dict[str, Any]:
    """构造 slow-path LLM 用来理解 Manifest 与预算的 JSON payload。

    载荷包含截断后的当前 User message、完整 TokenBudget、消息/Token/归档统计、最近 12 条
    ManifestItem、按 relevance 排序的最多 40 个候选，以及磁盘 working state。Instructions
    明确要求最终调用 ``curator_build_context``，并优先 protected、recent、relevant、unresolved
    消息。

    该 payload 只是 Curator 的决策输入，不直接进入主 Agent Context；文本通过 `_snippet`
    限长，ArchiveStore 的读取失败边界也已收敛。最终计划仍必须由 CuratorAssembler 再验证。
    """
    recent = [asdict(item) for item in state.manifest[-12:]]
    candidates = sorted(state.manifest, key=lambda item: (item.relevance, item.id), reverse=True)[:40]
    return {
        "session_key": state.session_key,
        "current_user_message": _snippet(state.turn.current_message, 1200),
        "budget": asdict(state.budget),
        "manifest_summary": {
            "message_count": len(state.manifest),
            "history_tokens": sum(item.tokens for item in state.manifest),
            "archived_count": sum(1 for item in state.manifest if item.archived),
        },
        "recent_messages": recent,
        "candidate_manifest": [asdict(item) for item in candidates],
        "working_state": archive.read_working_state(state.session_key),
        "instructions": (
            "Use tools to inspect budget and build a final context. "
            "End by calling curator_build_context. Prefer protected, recent, relevant, and unresolved messages."
        ),
    }


def _trace_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    traced = []
    for msg in messages:
        clean = dict(msg)
        clean.pop("reasoning_content", None)
        clean.pop("thinking_blocks", None)
        if isinstance(clean.get("content"), str):
            clean["content"] = _snippet(clean["content"], 1600)
        traced.append(clean)
    return traced
