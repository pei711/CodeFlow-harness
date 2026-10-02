"""管理 Conversation Session 的 Append-only History、JSONL Persistence 与并发代际边界。

Session 保存消息事实与少量交互状态；SessionManager 负责 Channel/Chat Key 到嵌套文件的映射、
跨进程锁、Append/Atomic Rewrite、Storage Epoch、Resolve/List/Fork/Delete。Consolidation 只在 Memory
文件生成摘要，不回写消息；Session Transcript 也不能恢复 Tool Side Effect 或 Agent Runtime。
"""

import copy
import errno
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from codeflow.utils.atomic_io import (
    StorageCorruptionError,
    atomic_replace,
    epoch_is_known,
    locked_append,
    locked_delete,
    locked_read,
    read_epoch,
    read_utf8_with_incomplete_tail,
)
from codeflow.utils.helpers import ensure_dir, safe_filename


def new_chat_id(now: datetime | None = None) -> str:
    """生成 Opaque、Sortable 的 Per-session Chat ID：``YYYYMMDD_HHMMSS_xxxxxx``。

    Timestamp Prefix 使 Value 可按创建时间排序，6-char UUID Suffix 降低同秒 Collision；格式与
    Channel 无关。该值成为 ``channel:chat_id`` Key 的 Chat Segment 与 JSONL Filename Stem，不编码
    User Identity，也不保证全局 Cryptographic Uniqueness。
    """
    ts = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:6]}"


@dataclass(frozen=True)
class SessionResolution:
    """记录 User-supplied Session ID 解析为 Full Key 的 Outcome。

    ``status`` 只能是 ``"resolved"``、``"ambiguous"``、``"not_found"``。Resolved 时 ``key`` 携带
    ``channel:chat_id``；Ambiguous 时 ``candidates`` 给出全部 Full Keys。No-match 保持 not_found，
    Caller 自己决定 Tail：Agent ``--session`` 可 Mint ``cli:<value>``，Read-only Export 必须 Error，
    Resolver 不替不同 Workflow 猜策略。
    """

    status: str
    key: str | None = None
    candidates: tuple[str, ...] = ()


@dataclass
class Session:
    """保存一段 Conversation 的 Message Fact、Metadata 与 Persistence State。

    Messages 以 JSONL 易读持久化，正常写入为 Append-only，以保持 Ordering 与 LLM Cache Efficiency。
    ``last_consolidated`` 只标明哪些消息已总结进 MEMORY.md/HISTORY.md；Consolidation does NOT 修改
    messages list，也不改变 ``get_history()`` 输出的 Tail Fact。Clear/Undo 是显式 History Rewrite，
    必须经 Manager Fence Concurrent Writer 后落盘。

    pending_clarification 是当前等待偏好答案的 Interaction State，不是普通 History；Storage Epoch、
    Persisted Snapshot 与 Rewrite Flag 属于 Manager 并发控制。Session 本身不执行 I/O，Caller 必须
    调用 SessionManager.save/commit_history_rewrite。
    """

    key: str  # channel:chat_id 格式的键
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_consolidated: int = 0  # 已归并到文件的消息数量
    # ── 个性化状态 ───────────────────────────────────────────────────────────
    # Agent 提出澄清问题并等待回答时设置。
    # 结构：{"original_message": str, "question": str, "domain": str}
    # 用户回答处理完毕后立即清除。
    pending_clarification: dict | None = field(default=None)
    # 已落盘的消息数量；save() 只追加该索引之后的消息。
    _persisted_count: int = field(default=0, repr=False)
    # 用于区分已保存的空会话和从未持久化的惰性会话。
    _persisted: bool = field(default=False, repr=False, compare=False)
    _storage_epoch: int = field(default=0, repr=False, compare=False)
    _persisted_snapshot: dict[str, Any] | None = field(default=None, repr=False, compare=False)
    _requires_rewrite: bool = field(default=False, repr=False, compare=False)

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """用 ``role``、``content`` 与扩展 Field 向 Session 追加一条 Message。

         方法只组装 Dict 并委托 `record`，因此 Timestamp、Append Order 与 updated_at 规则保持单一
        入口。它不立即写 Disk，也不校验 Provider Role Sequence；Persistence 由 Manager 负责。
        """
        self.record({"role": role, "content": content, **kwargs})

    def record(self, msg: dict[str, Any]) -> None:
        """向内存 Tail 追加 Message Dict，并在缺失时 Stamp Wall-clock Timestamp。

        这是 Session Write 的 Single Choke Point：``add_message``、AgentLoop ``_save_turn``、
        Clarification Append 都必须经过这里，确保没有 Unstamped Message。Caller-set ``timestamp``
        保留；Per-message Order 与 Turn Group 直接来自 Append Order 和 ``role`` Boundary，不维护另一套
        received_at/turn_id。

        方法原地接纳 Dict、更新 updated_at，但不 Save；Caller 后续修改同 Dict 会影响 Session，因而
        应把传入对象视为已转移所有权。
        """
        msg.setdefault("timestamp", datetime.now().isoformat())
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
        """返回供 LLM 使用的 Unconsolidated Message View，并对齐到 User Turn 起点。

         先取 ``messages[last_consolidated:]``，再保留最后 ``max_messages`` 项；若切片从 Assistant/
         Tool 中间开始，丢弃首个 User 之前内容，避免 Orphan Tool Result。输出只保留 Provider 所需
         Role/Content、Tool Pair 与 Reasoning/Thinking Field，移除 Timestamp/Metadata。

         返回新 Dict List，不修改 Append-only Messages。``max_messages=0`` 的 Python Slice 语义会取
        完整 Tail，供 Legacy Context Path 使用；此方法不执行 Token Budget Trimming。
        """
        unconsolidated = self.messages[self.last_consolidated :]
        sliced = unconsolidated[-max_messages:]

        # 丢弃开头的非用户消息，避免产生孤立的 tool_result 块。
        for i, m in enumerate(sliced):
            if m.get("role") == "user":
                sliced = sliced[i:]
                break

        out: list[dict[str, Any]] = []
        for m in sliced:
            entry: dict[str, Any] = {"role": m["role"], "content": m.get("content", "")}
            for k in ("tool_calls", "tool_call_id", "name", "reasoning_content", "thinking_blocks"):
                if k in m:
                    entry[k] = m[k]
            out.append(entry)
        return out

    def clear(self) -> None:
        """清空全部 Message，并把 Session Interaction State 重置为 Initial State。

        Messages 变空、last_consolidated 归零、pending_clarification 清除、updated_at 刷新。该操作只
        修改内存对象，不自动删除/改写 JSONL；Caller 必须用 `commit_history_rewrite` Fence Old
        Writer。Metadata/Created_at 保留，Session Identity 不变。
        """
        self.messages = []
        self.last_consolidated = 0
        self.pending_clarification = None
        self.updated_at = datetime.now()

    def undo_last_turn(self, n: int = 1) -> int:
        """从 Unconsolidated Tail 删除最后 ``n`` 个 User-turn Blocks。

        Turn 从 ``role == "user"`` 开始，到下一 User Message 前结束，后续 Assistant/Tool 都继承该
        Block。只允许修改 ``messages[last_consolidated:]``；已经总结进 MEMORY.md 的内容绝不跨越。
        ``n < 1`` 或 Tail 无 User 时返回 0，否则返回实际 Removed Message Count，并清 Waiting
        Clarification。

        这是内存 Rewrite，Persistence 仍是 Caller 责任，应经 ``SessionManager.save`` 的 Rewrite/Fence
        Path，而不是普通 Append。
        """
        if n < 1:
            return 0
        start = self.last_consolidated
        user_starts = [i for i in range(start, len(self.messages)) if self.messages[i].get("role") == "user"]
        if not user_starts:
            return 0
        cut_index = user_starts[-n] if n <= len(user_starts) else user_starts[0]
        removed = len(self.messages) - cut_index
        self.messages = self.messages[:cut_index]
        self.pending_clarification = None
        self.updated_at = datetime.now()
        return removed

    def _persistence_snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "messages": self.messages,
                "metadata": self.metadata,
                "last_consolidated": self.last_consolidated,
                "pending_clarification": self.pending_clarification,
            }
        )

    def _mark_persisted(self) -> None:
        self._persisted_snapshot = self._persistence_snapshot()

    def _is_dirty(self) -> bool:
        if self._requires_rewrite:
            return True
        if not self._persisted:
            return bool(
                self.messages or self.metadata or self.last_consolidated or self.pending_clarification is not None
            )
        return self._persisted_snapshot != self._persistence_snapshot()


class SessionManager:
    """管理 Conversation Session Cache 与 Sessions Directory 中的 JSONL Storage。

    每个 Key 映射 ``sessions/<safe channel>/<safe chat_id>.jsonl``。Manager 用 Metadata Record 保存
    Authoritative Key/Time/State，用 Message Record 保存 Append Fact；Cross-process Lock 与 Epoch 防止
    Concurrent Writer Lost Update，Atomic Rewrite 处理 Undo/Clear/Partial Tail。

    Cache 只优化对象复用，不是 Persistence Truth。Read-only Caller 应用 `peek`，未知 Key 不会因查询
    创建 Lazy Session；Delete/Fork/Resolve/List 都维护 Logical Generation Boundary。
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.sessions_dir = ensure_dir(self.workspace / "sessions")
        self._cache: dict[str, Session] = {}

    def _get_session_path(self, key: str) -> Path:
        """把 Full Session Key 映射为 ``sessions/{channel}/{chat_id}.jsonl`` Path。

        Key 以首个 Colon 分成 Channel 与 Chat ID，两段都经 safe_filename，形成嵌套路径。函数不创建
        Parent、不检查存在，也无法从 Sanitized Name 恢复被折叠字符；Authoritative Identity 仍在
        Metadata Record。
        """
        channel, _, chat_id = key.partition(":")
        return self.sessions_dir / safe_filename(channel) / f"{safe_filename(chat_id)}.jsonl"

    @staticmethod
    def key_from_path(path: Path) -> str:
        """Best-effort 反向解析 Nested Filename：Parent 是 Channel，Stem 是 ``chat_id``。

        Disk 上 ``_type:metadata`` 的 Key 存在时是 Authoritative 并优先；本函数只服务 Metadata-less
        File Fallback。``safe_filename`` Non-invertible，折叠成 ``_`` 的 ``/``、``:`` 等 Character
        无法恢复，所以返回值不是任意原 Key 的可靠逆函数。
        """
        return f"{path.parent.name}:{path.stem}"

    @staticmethod
    def _validate_metadata_identity(record_key: Any, requested_key: str) -> bool:
        if not isinstance(record_key, str):
            raise StorageCorruptionError(f"session {requested_key} metadata key must be a string")
        if record_key != requested_key:
            raise StorageCorruptionError(
                f"session metadata key {record_key!r} does not match requested key {requested_key!r}"
            )
        return True

    @classmethod
    def _validate_fallback_identity(
        cls,
        path: Path,
        requested_key: str,
        *,
        has_metadata_key: bool,
    ) -> None:
        if has_metadata_key:
            return
        canonical_key = cls.key_from_path(path)
        if requested_key != canonical_key:
            raise StorageCorruptionError(
                f"requested key {requested_key!r} does not match canonical path key {canonical_key!r}"
            )

    @classmethod
    def _decode_payload(cls, path: Path, requested_key: str, raw: str) -> dict[str, Any]:
        records = [
            (line_number, line.strip()) for line_number, line in enumerate(raw.splitlines(), start=1) if line.strip()
        ]
        messages: list[dict[str, Any]] = []
        metadata: dict[str, Any] = {}
        created_at: datetime | None = None
        updated_at: datetime | None = None
        last_consolidated = 0
        pending_clarification: dict[str, Any] | None = None
        has_metadata_key = False
        partial_tail_found = False
        for index, (line_number, line) in enumerate(records):
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                partial_tail = index == len(records) - 1 and not raw.endswith(("\n", "\r"))
                if partial_tail:
                    logger.debug("Skipping partial trailing line in session {}", requested_key)
                    partial_tail_found = True
                    continue
                raise StorageCorruptionError(
                    f"session {requested_key} has invalid JSON record at line {line_number}"
                ) from exc
            if not isinstance(data, dict):
                raise StorageCorruptionError(f"session {requested_key} record must be an object (line {line_number})")
            if data.get("_type") == "metadata":
                record_key = data.get("key")
                record_metadata = data.get("metadata", {})
                record_created_at = data.get("created_at")
                record_updated_at = data.get("updated_at")
                record_last_consolidated = data.get("last_consolidated", 0)
                record_pending_clarification = data.get("pending_clarification")
                has_metadata_key = cls._validate_metadata_identity(record_key, requested_key) or has_metadata_key
                if not isinstance(record_metadata, dict):
                    raise StorageCorruptionError(f"session {requested_key} metadata payload must be an object")
                if record_created_at is not None and not isinstance(record_created_at, str):
                    raise StorageCorruptionError(f"session {requested_key} created_at must be a string")
                if record_updated_at is not None and not isinstance(record_updated_at, str):
                    raise StorageCorruptionError(f"session {requested_key} updated_at must be a string")
                if (
                    isinstance(record_last_consolidated, bool)
                    or not isinstance(record_last_consolidated, int)
                    or record_last_consolidated < 0
                ):
                    raise StorageCorruptionError(
                        f"session {requested_key} last_consolidated must be a non-negative integer"
                    )
                if record_pending_clarification is not None and not isinstance(record_pending_clarification, dict):
                    raise StorageCorruptionError(f"session {requested_key} pending_clarification must be an object")
                metadata = record_metadata
                try:
                    created_at = datetime.fromisoformat(record_created_at) if record_created_at else None
                    updated_at = datetime.fromisoformat(record_updated_at) if record_updated_at else None
                except ValueError as exc:
                    raise StorageCorruptionError(f"session {requested_key} timestamp is invalid") from exc
                last_consolidated = record_last_consolidated
                pending_clarification = record_pending_clarification
            else:
                messages.append(data)
        cls._validate_fallback_identity(
            path,
            requested_key,
            has_metadata_key=has_metadata_key,
        )
        return {
            "messages": messages,
            "metadata": metadata,
            "created_at": created_at,
            "updated_at": updated_at,
            "last_consolidated": last_consolidated,
            "pending_clarification": pending_clarification,
            "partial_tail_found": partial_tail_found,
        }

    @staticmethod
    def _durable_state(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": payload["messages"],
            "metadata": payload["metadata"],
            "last_consolidated": payload["last_consolidated"],
            "pending_clarification": payload["pending_clarification"],
        }

    @classmethod
    def _validate_append_base(
        cls,
        path: Path,
        requested_key: str,
        raw: str,
        expected: dict[str, Any],
    ) -> None:
        payload = cls._decode_payload(path, requested_key, raw)
        if payload["partial_tail_found"]:
            raise FileNotFoundError(f"session has a partial trailing record before append: {path}")
        actual = cls._durable_state(payload)
        for state_field in ("metadata", "last_consolidated", "pending_clarification"):
            if actual[state_field] != expected[state_field]:
                raise FileNotFoundError(f"session metadata changed before append: {path}")

    @classmethod
    def _validate_rewrite_base(
        cls,
        path: Path,
        requested_key: str,
        raw: str,
        expected: dict[str, Any],
    ) -> None:
        actual = cls._durable_state(cls._decode_payload(path, requested_key, raw))
        if actual != expected:
            raise FileNotFoundError(f"session changed before rewrite: {path}")

    def resolve_key(self, value: str) -> SessionResolution:
        """跨 Channel 把 Session ID 解析为 Full ``channel:chat_id`` Key。

        Agent ``--session`` 与 Export 共用此 Core：含 ``:`` 直接视为 Full Key；跨 Channel 精确 Chat ID
        唯一命中优先，其次唯一 Prefix；多个 Match 返回 Ambiguous Candidate，零 Match 返回 not_found。

        No-match 不在此 Mint 或 Error，Caller 决定 Agent Path 创建 ``cli:<value>``，还是 Read-only
        Export 拒绝。Resolver 读取 List Snapshot，不锁定之后 Session 仍存在。
        """
        if ":" in value:
            return SessionResolution("resolved", key=value)
        sessions = self.list_sessions(channel=None)
        exact = [s for s in sessions if s["key"].partition(":")[2] == value]
        matches = exact or [s for s in sessions if s["key"].partition(":")[2].startswith(value)]
        if len(matches) > 1:
            return SessionResolution("ambiguous", candidates=tuple(s["key"] for s in matches))
        if matches:
            return SessionResolution("resolved", key=matches[0]["key"])
        return SessionResolution("not_found")

    def find_most_recent_chat_id(self, channel: str) -> str | None:
        """返回 Channel 中最近更新 Session 的 Chat ID，无有效 Session 时返回 None。

        Cron Delivery 在 Trigger Time 用它自动选择 Ephemeral CLI/TUI Reminder 目标，User 无需知道
        Target Channel 的 open_id/chat_id。方法扫描 Candidate JSONL 的 Last Metadata，验证
        Authoritative ``<channel>:<chat_id>``，按 ``updated_at`` 比较；缺时间时回退 File mtime。

        Corrupt/Identity-mismatch File 经 `_scan_file` 跳过。结果只是一刻 Snapshot，不创建 Session 或
        持久化 Cron Binding。
        """
        channel_dir = self.sessions_dir / safe_filename(channel)
        if not channel_dir.is_dir():
            return None

        best_chat_id: str | None = None
        best_updated = ""
        for p in channel_dir.glob("*.jsonl"):
            meta, _count = self._scan_file(p)
            if meta is None:
                continue
            key_val = meta.get("key", "")
            if ":" not in key_val:
                continue
            ch, chat_id = key_val.split(":", 1)
            if ch != channel or not chat_id:
                continue
            updated = meta.get("updated_at")
            if not isinstance(updated, str) or not updated:
                try:
                    updated = datetime.fromtimestamp(p.stat().st_mtime).isoformat()
                except OSError:
                    continue
            if updated > best_updated:
                best_chat_id = chat_id
                best_updated = updated
        return best_chat_id

    @staticmethod
    def _scan_file(path: Path) -> tuple[dict[str, Any] | None, int]:
        """单 Pass 验证 Session File，并返回 Last Metadata 与 Message Count。

        每次 Save 追加 Metadata Record，所以最后一条代表 Current State；普通 Message 只计数不保留，
        降低 List Scan Memory。函数验证 JSON Object、Metadata Key 与 Path Identity、Field Types，并
        容忍最后一个未换行 Partial Record；中间损坏或 Identity Drift 返回 ``(None, 0)``。
        """
        meta: dict[str, Any] | None = None
        identity: str | None = None
        count = 0
        try:
            raw = read_utf8_with_incomplete_tail(path)
            records = [line.strip() for line in raw.splitlines() if line.strip()]
            for index, line in enumerate(records):
                line = line.strip()
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    if index == len(records) - 1 and not raw.endswith(("\n", "\r")):
                        break
                    return None, 0
                if not isinstance(data, dict):
                    return None, 0
                if data.get("_type") == "metadata":
                    key = data.get("key")
                    if not isinstance(key, str):
                        return None, 0
                    channel, separator, chat_id = key.partition(":")
                    if (
                        not separator
                        or path.parent.name != safe_filename(channel)
                        or path.stem != safe_filename(chat_id)
                        or (identity is not None and key != identity)
                    ):
                        return None, 0
                    record_metadata = data.get("metadata", {})
                    created_at = data.get("created_at")
                    updated_at = data.get("updated_at")
                    last_consolidated = data.get("last_consolidated", 0)
                    pending_clarification = data.get("pending_clarification")
                    if not isinstance(record_metadata, dict):
                        return None, 0
                    if created_at is not None and not isinstance(created_at, str):
                        return None, 0
                    if updated_at is not None and not isinstance(updated_at, str):
                        return None, 0
                    if (
                        isinstance(last_consolidated, bool)
                        or not isinstance(last_consolidated, int)
                        or last_consolidated < 0
                    ):
                        return None, 0
                    if pending_clarification is not None and not isinstance(pending_clarification, dict):
                        return None, 0
                    identity = key
                    meta = data
                else:
                    count += 1
        except (OSError, UnicodeError):
            return None, 0
        return meta, count

    def get_or_create(self, key: str) -> Session:
        """取得 ``key`` 的 Existing Session，Disk 不存在时创建 Lazy Empty Session。

        Cache Hit 直接返回同一 Instance；否则 `_load_state` 读取稳定 Snapshot。没有 File 时建立未
        Persist 的 Session 并记录 Storage Epoch，只有后续 Dirty Save 才创建 JSONL。这里的 ``key``
        通常采用 ``channel:chat_id``，因此它同时标识消息来源与一次对话。Read-only Caller 不应使用
        本方法查询未知 Key，应使用 `peek`，避免 Cache 中产生空逻辑 Session。
        """
        if key in self._cache:
            return self._cache[key]

        session, storage_epoch = self._load_state(key)
        if session is None:
            session = Session(key=key, _storage_epoch=storage_epoch)

        self._cache[key] = session
        return session

    def _load_state(self, key: str) -> tuple[Session | None, int]:
        """从一个稳定 Storage Snapshot 加载 Session 与 Missing-path Generation。

        首选 locked_read；只在 Permission/Read-only Lock 不可用时，用 Epoch-before/after + Exists 一致
        的三次尝试读取。Known Generation Primary File 缺失或无法取得 Stable Snapshot 视为
        StorageCorruptionError。成功 Decode Metadata/Messages，标记 Persisted Count/Snapshot 与 Partial
        Tail Rewrite Requirement；其他异常包装为 Corruption，不静默返回空 Session。
        """
        path = self._get_session_path(key)

        try:
            try:
                loaded = locked_read(path)
            except OSError as exc:
                if not isinstance(exc, PermissionError) and exc.errno not in {errno.EACCES, errno.EROFS}:
                    raise
                loaded = None
                for _ in range(3):
                    epoch_before = read_epoch(path)
                    known_before = epoch_is_known(path)
                    try:
                        raw = read_utf8_with_incomplete_tail(path)
                    except FileNotFoundError:
                        raw = None
                    epoch_after = read_epoch(path)
                    known_after = epoch_is_known(path)
                    exists_after = path.exists()
                    if (
                        epoch_before == epoch_after
                        and known_before == known_after
                        and exists_after == (raw is not None)
                    ):
                        loaded = (raw, epoch_after, known_after)
                        break
            if loaded is None:
                raise StorageCorruptionError(f"failed to read a stable snapshot for session {key}")
            raw, storage_epoch, known_epoch = loaded
            if raw is None:
                if known_epoch and storage_epoch == 0:
                    raise StorageCorruptionError(f"session {key} is missing its primary file for a known generation")
                return None, storage_epoch
            payload = self._decode_payload(path, key, raw)

            session = Session(
                key=key,
                messages=payload["messages"],
                created_at=payload["created_at"] or datetime.now(),
                updated_at=payload["updated_at"] or payload["created_at"] or datetime.now(),
                metadata=payload["metadata"],
                last_consolidated=payload["last_consolidated"],
                pending_clarification=payload["pending_clarification"],
                _storage_epoch=storage_epoch,
                _requires_rewrite=payload["partial_tail_found"],
            )
            session._persisted_count = len(session.messages)
            session._persisted = True
            session._mark_persisted()
            return session, storage_epoch
        except StorageCorruptionError:
            raise
        except Exception as exc:
            logger.warning("Failed to load session {}: {}", key, exc)
            raise StorageCorruptionError(f"failed to load session {key}") from exc

    def _load(self, key: str) -> Session | None:
        """从 Disk 加载 Session，并丢弃 `_load_state` 同时返回的 Epoch 辅助值。

        File 不存在且 Generation 合法时返回 None；Corruption 继续抛出。方法不写 Cache，供 `peek`
        Read-only Path 使用。
        """
        session, _storage_epoch = self._load_state(key)
        return session

    def save(self, session: Session, *, force_rewrite: bool = False) -> None:
        """在 Cross-process Concurrency Fence 下把 Session 保存为 JSONL。

        正常 Append Path 写一条 Fresh Metadata 与 `_persisted_count` 后的新 Messages，Cross-process Lock
        保证 Concurrent Writer 不丢 Turn，且单 Turn Messages 连续。Message List 缩短、Partial Tail、
        Forced History Fence 或非 Append-only Drift 走 Atomic Rewrite，先验证 Expected Snapshot/Epoch，
        防止覆盖别人已提交状态。

        Metadata 每次保存都包含 Reserved Source/Channel/Chat/Title/Parent Slot、Consolidation 与 Pending
        Clarification。成功后更新 Persisted Count、Epoch、Snapshot 与 Cache；Conflict/Corruption 向外
        抛，Caller 不能把 Failed Save 当成功。
        """
        path = self._get_session_path(session.key)

        channel, _, chat_id = session.key.partition(":")
        reserved = {
            "source": None,
            "channel": channel,
            "chat_id": chat_id,
            "title": None,
            "parent_session_id": None,
        }
        session.metadata = {**reserved, **session.metadata}

        metadata_line = json.dumps(
            {
                "_type": "metadata",
                "key": session.key,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "metadata": session.metadata,
                "last_consolidated": session.last_consolidated,
                # 个性化状态：跨重启保留澄清问题的等待状态。
                "pending_clarification": session.pending_clarification,
            },
            ensure_ascii=False,
        )

        persisted_messages = (
            session._persisted_snapshot.get("messages") if session._persisted_snapshot is not None else None
        )
        append_only = (
            persisted_messages is not None
            and len(session.messages) >= len(persisted_messages)
            and session.messages[: len(persisted_messages)] == persisted_messages
        )

        rewrite = force_rewrite or session._requires_rewrite or (session._persisted and not append_only)
        if rewrite:
            lines = [metadata_line]
            lines += [json.dumps(m, ensure_ascii=False) for m in session.messages]
            expected_state = session._persisted_snapshot
            storage_epoch = atomic_replace(
                path,
                "".join(line + "\n" for line in lines),
                expected_epoch=session._storage_epoch,
                expected_exists=session._persisted,
                require_existing=session._persisted,
                increment_epoch=True,
                validate_existing=(
                    None
                    if expected_state is None
                    else lambda raw: self._validate_rewrite_base(
                        path,
                        session.key,
                        raw,
                        expected_state,
                    )
                ),
            )
        else:
            new_messages = session.messages[session._persisted_count :]
            lines = [metadata_line]
            lines += [json.dumps(m, ensure_ascii=False) for m in new_messages]
            expected_state = (
                session._persisted_snapshot
                if session._persisted_snapshot is not None
                else session._persistence_snapshot()
            )
            storage_epoch = locked_append(
                path,
                lines,
                expected_epoch=session._storage_epoch,
                require_existing=session._persisted,
                validate_existing=lambda raw: self._validate_append_base(
                    path,
                    session.key,
                    raw,
                    expected_state,
                ),
            )

        session._persisted_count = len(session.messages)
        session._persisted = True
        session._storage_epoch = storage_epoch
        session._requires_rewrite = False
        session._mark_persisted()
        self._cache[session.key] = session

    def commit_history_rewrite(self, session: Session) -> None:
        """提交 Clear/Undo History Rewrite，并 Fence 仍引用 Old Generation 的 Writer。

        Persisted 或 Dirty Session 强制 `save(..., force_rewrite=True)`；从未 Persist 的空 Session 则对
        Missing Path 执行 locked_delete Fence 并递增 Epoch，使 Stale Reference 不能稍后复活旧历史。
        成功后 Cache 指向当前对象，方法不改变 Session Key。
        """
        if session._persisted or session._is_dirty():
            self.save(session, force_rewrite=True)
            return

        path = self._get_session_path(session.key)
        locked_delete(
            path,
            expected_epoch=session._storage_epoch,
            expected_exists=False,
            fence_missing=True,
            increment_epoch=True,
        )
        session._storage_epoch += 1
        self._cache[session.key] = session

    def invalidate(self, key: str) -> None:
        """按 Key 从 In-memory Cache 移除 Session，缺失时 No-op。

        该操作不删除 JSONL、不改变 Epoch，也不使 Caller 已持有的 Session Reference 失效；需要逻辑
        Delete 与 Stale-writer Fence 时必须调用 `delete`。
        """
        self._cache.pop(key, None)

    def delete(
        self,
        key: str,
        *,
        allow_cached_missing: bool = False,
        expected_epoch: int | None = None,
        expected_exists: bool | None = None,
    ) -> bool:
        """删除/Invalidate 当前 Logical Session Generation，并 Fence Stale References。

        locked_delete 可校验 Expected Epoch/Existence，Cached Lazy Session 即使无 File 也用 fence_missing
        推进 Generation。成功后 Invalidate Cache。匹配 Persisted File 被移除时 True；
        ``allow_cached_missing`` 可把 Known Lazy Session 的 Logical Delete 也视为 True；Unknown Key Safe
        No-op。OSError 记录并返回 False，不假报删除。
        """
        path = self._get_session_path(key)
        cached = key in self._cache
        try:
            removed = locked_delete(
                path,
                expected_epoch=expected_epoch,
                expected_exists=expected_exists,
                fence_missing=cached,
                increment_epoch=True,
            )
        except OSError:
            logger.warning("session.delete: failed to remove file for {}", key)
            return False
        self.invalidate(key)
        return removed or (allow_cached_missing and cached and expected_exists is False)

    def exists(self, key: str) -> bool:
        """返回 Session Primary JSONL 是否存在于 Disk。

        Lazy Cached Session 在 First Save 前返回 False；结果不验证 File 内容或 Cache，也不说明 Session
        当前 Generation 可安全写入。需要读取请使用 `peek`。
        """
        return self._get_session_path(key).exists()

    def peek(self, key: str) -> "Session | None":
        """返回 Cached Session；否则从 Disk Load，但不把结果加入 Cache。

        Read-only Caller 应使用本方法而非 get_or_create，因为未知 Key 会返回 None，不会创建/缓存 Fresh
        Empty Session。Cache Hit 返回 Mutable Original Instance，Caller 若只读不应修改；Disk Corruption
        继续抛出。
        """
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        return self._load(key)

    def fork(self, source_key: str, *, title: str | None = None) -> "Session | None":
        """在 ``source_key`` 当前 Head 创建可独立 Diverge 的 Child Session。

        Full-copy Semantics：Child 沿 Source Channel Mint Fresh ``chat_id``，Deep-copy 全部 Messages，Reserved
        ``parent_session_id`` 记录 Lineage。继承 ``last_consolidated``，使 Fork Point Active Context 与
        Parent 一致；``pending_clarification`` 不继承，因为 Interaction Wait-state 不是 History。
        Explicit Title 优先，否则 Parent Title 加 ``(fork)``。

        Child Eager Persist。Source 不存在、Flush Failed、Zero Messages 时返回 None；成功返回 Persisted
        Child。Fork 不建立后续同步，Parent/Child 从此独立。
        """
        source = self.peek(source_key)
        if source is None:
            return None
        if not self.flush(source_key):
            return None
        if not source.messages:
            return None

        channel = source_key.partition(":")[0]
        child = Session(
            key=f"{channel}:{new_chat_id()}",
            messages=copy.deepcopy(source.messages),
            last_consolidated=source.last_consolidated,
        )
        if title is not None:
            child.metadata["title"] = title
        else:
            parent_title = (source.metadata or {}).get("title")
            if parent_title:
                child.metadata["title"] = f"{parent_title} (fork)"
        child.metadata["parent_session_id"] = source_key
        self.save(child)
        return child

    def flush(self, key: str) -> bool:
        """仅在 Cached Session 有 Unpersisted State 时 Save，并用 Boolean 汇报 Attempt Outcome。

        Key 未 Cache 或 Snapshot 无变化是 No-op True；Dirty 时调用 Save，任何 Failure 记录后吞掉并
        返回 False。该 Contract 让 Fork/Shutdown 决定是否继续，而不把“无需保存”误判失败；False
        不包含具体 Error，诊断看 Log。
        """
        cached = self._cache.get(key)
        if cached is None:
            return True
        if cached._is_dirty():
            try:
                self.save(cached)
            except Exception:
                logger.warning("flush: failed to persist session {}", key)
                return False
        return True

    def list_sessions(self, channel: str | None = None) -> list[dict[str, Any]]:
        """列出 Valid Persisted Sessions，并可按 Channel Filter。

        每个 JSONL Single-pass Scan，Corrupt File 跳过；Entry 带 key、created_at、updated_at、path、
        message_count 与 metadata，按 updated_at Descending。Filter 比较 Sanitized Channel Directory；
        Lazy Cache-only Session 不出现，因为它尚无 Durable Fact。
        """
        sessions = []

        for path in self.sessions_dir.glob("*/*.jsonl"):
            if channel is not None and path.parent.name != channel:
                continue
            data, message_count = self._scan_file(path)
            if data is None:
                continue
            key = data.get("key") or self.key_from_path(path)
            sessions.append(
                {
                    "key": key,
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "path": str(path),
                    "message_count": message_count,
                    "metadata": data.get("metadata", {}),
                }
            )

        return sorted(
            sessions,
            key=lambda item: item["updated_at"] if isinstance(item.get("updated_at"), str) else "",
            reverse=True,
        )
