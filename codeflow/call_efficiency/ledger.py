"""为标准化 Provider Calls 提供 Append-only Evidence Ledger。

`CallLedger` 先在内存接收 `CallRecord`，再由单独 Writer Thread 按批追加到每日 JSONL，避免模型调用
主路径等待磁盘 IO。每个 Ledger Instance 还有独立 Health Entry，记录 Accepted、Persisted 与 Lost
Records；进程关闭时汇总到带 Schema 的健康文件，让使用者能够区分“生成了调用证据”和“证据已经
可靠落盘”。

Append-only 保护调用记录不被日常更新覆盖，但不等于绝对不丢数据：Queue Full、Writer Failure 或
Shutdown Timeout 会使 Ledger 进入 Degraded，并通过 `CallLedgerError` 与 Health File 暴露证据
缺口。调用方必须把这个状态纳入验收，不能只看内存中的 Recent Records。
"""

from __future__ import annotations

import json
import os
import queue
import threading
import uuid
from collections import deque
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Final

from loguru import logger

from codeflow.call_efficiency.models import CallRecord
from codeflow.utils.atomic_io import locked_append
from codeflow.utils.portable_lock import file_lock

_STOP: Final = object()
HEALTH_SCHEMA = "codeflow.call-efficiency.ledger-health.v1"


class CallLedgerError(RuntimeError):
    pass


class CallLedger:
    def __init__(
        self,
        telemetry_dir: Path,
        *,
        persist: bool,
        queue_capacity: int = 1024,
        recent_capacity: int = 256,
    ) -> None:
        self.telemetry_dir = telemetry_dir
        self.persist = persist
        self.records: deque[CallRecord] = deque(maxlen=recent_capacity)
        self._state_lock = threading.Lock()
        self._closed = False
        self._close_requested = threading.Event()
        self._failure: BaseException | None = None
        self._ledger_id = uuid.uuid4().hex
        self._accepted = 0
        self._persisted = 0
        self._queue: queue.Queue[str | object] | None = None
        self._writer: threading.Thread | None = None
        if persist:
            self._queue = queue.Queue(maxsize=queue_capacity)
            self._writer = threading.Thread(
                target=self._write_loop,
                name="codeflow-call-efficiency-ledger",
                daemon=True,
            )
            self._writer.start()

    def append(self, record: CallRecord) -> None:
        line = json.dumps(asdict(record), ensure_ascii=False)
        with self._state_lock:
            if self._closed:
                raise CallLedgerError("CallEfficiency ledger is closed")
            self.records.append(record)
            self._accepted += 1
            if self._failure is not None:
                raise CallLedgerError("CallEfficiency ledger writer failed") from self._failure
            if self._queue is not None:
                try:
                    self._queue.put_nowait(line)
                except queue.Full as exc:
                    self._failure = exc
                    raise CallLedgerError("CallEfficiency ledger writer is not keeping up") from exc

    def close(self) -> None:
        with self._state_lock:
            if not self._closed:
                self._closed = True
                self._close_requested.set()
                if self._queue is not None:
                    try:
                        self._queue.put(_STOP, timeout=1.0)
                    except queue.Full:
                        pass
            writer = self._writer
        if writer is not None:
            writer.join(timeout=5.0)
            if writer.is_alive():
                with self._state_lock:
                    self._failure = self._failure or TimeoutError("CallEfficiency ledger writer did not stop")
        with self._state_lock:
            status = "degraded" if self._failure is not None or self._persisted != self._accepted else "healthy"
        if self.persist:
            try:
                self._write_health(status)
            except Exception as exc:
                with self._state_lock:
                    self._failure = self._failure or exc
                logger.error("CallEfficiency ledger health write failed: {}", exc)
        with self._state_lock:
            self._raise_if_failed()

    def _write_loop(self) -> None:
        work_queue = self._queue
        if work_queue is None:
            return
        while True:
            try:
                first = work_queue.get(timeout=0.25)
            except queue.Empty:
                if self._close_requested.is_set():
                    return
                continue
            if first is _STOP:
                return
            lines = [first]
            stop_after_batch = False
            while len(lines) < 64:
                try:
                    item = work_queue.get_nowait()
                except queue.Empty:
                    break
                if item is _STOP:
                    stop_after_batch = True
                    break
                lines.append(item)
            try:
                path = self.telemetry_dir / f"call-efficiency-{date.today().isoformat()}.jsonl"
                locked_append(path, lines)
                with self._state_lock:
                    self._persisted += len(lines)
            except BaseException as exc:
                with self._state_lock:
                    self._failure = exc
                try:
                    self._write_health("degraded")
                except Exception as health_exc:
                    logger.error("CallEfficiency ledger health write failed: {}", health_exc)
                logger.error("CallEfficiency ledger write failed: {}", exc)
                return
            if stop_after_batch:
                return

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise CallLedgerError("CallEfficiency ledger writer failed") from self._failure

    def _write_health(self, status: str) -> None:
        with self._state_lock:
            ledger_id = self._ledger_id
            accepted_records = self._accepted
            persisted_records = self._persisted
            error_type = type(self._failure).__name__ if self._failure is not None else None
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        path = self.telemetry_dir / "call-efficiency-ledger-health.json"
        lock_path = path.parent / ".lock" / f"{path.name}.lock"
        with file_lock(lock_path):
            entries = _read_health_entries(path)
            previous = entries.get(ledger_id, {})
            accepted = max(accepted_records, _health_count(previous, "accepted_records", default=0))
            persisted = max(persisted_records, _health_count(previous, "persisted_records", default=0))
            entry_status = "degraded" if previous.get("status") == "degraded" or status == "degraded" else "healthy"
            entries[ledger_id] = {
                "status": entry_status,
                "accepted_records": accepted,
                "persisted_records": persisted,
                "lost_records": accepted - persisted,
                "error_type": error_type or previous.get("error_type"),
            }
            payload = _aggregate_health(entries)
            temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            try:
                with temp.open("w", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp, path)
            finally:
                temp.unlink(missing_ok=True)


def _health_count(payload: dict, key: str, *, default: int | None = None) -> int:
    if key not in payload and default is not None:
        return default
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid CallEfficiency health field: {key}")
    return value


def _validate_health_entry(entry: object) -> dict:
    if not isinstance(entry, dict) or entry.get("status") not in {"healthy", "degraded"}:
        raise ValueError("invalid CallEfficiency health entry")
    accepted = _health_count(entry, "accepted_records")
    persisted = _health_count(entry, "persisted_records")
    lost = _health_count(entry, "lost_records")
    if persisted > accepted or lost != accepted - persisted:
        raise ValueError("inconsistent CallEfficiency health counts")
    if entry["status"] == "healthy" and lost:
        raise ValueError("healthy CallEfficiency entry reports loss")
    return entry


def _read_health_entries(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != HEALTH_SCHEMA:
        raise ValueError("unsupported CallEfficiency health schema")
    _validate_health_entry(payload)
    entries = payload.get("ledgers")
    if entries is None:
        return {"legacy": payload}
    if not isinstance(entries, dict):
        raise ValueError("invalid CallEfficiency health ledgers")
    validated = {str(key): _validate_health_entry(value) for key, value in entries.items()}
    aggregate = _aggregate_health(validated)
    for key in ("status", "accepted_records", "persisted_records", "lost_records"):
        if payload.get(key) != aggregate[key]:
            raise ValueError("CallEfficiency health aggregate does not match its ledger entries")
    return validated


def _aggregate_health(entries: dict[str, dict]) -> dict:
    accepted = sum(_health_count(entry, "accepted_records") for entry in entries.values())
    persisted = sum(_health_count(entry, "persisted_records") for entry in entries.values())
    lost = accepted - persisted
    degraded = lost > 0 or any(entry["status"] == "degraded" for entry in entries.values())
    error_type = next((entry.get("error_type") for entry in entries.values() if entry.get("error_type")), None)
    return {
        "schema": HEALTH_SCHEMA,
        "status": "degraded" if degraded else "healthy",
        "accepted_records": accepted,
        "persisted_records": persisted,
        "lost_records": lost,
        "error_type": error_type,
        "ledgers": entries,
    }


__all__ = ["HEALTH_SCHEMA", "CallLedger", "CallLedgerError"]
