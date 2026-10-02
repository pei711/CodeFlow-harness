"""管理 atomic JSON persistence 与 ``run_meta.json`` durable lifecycle。

resume 依赖两项 invariant：本层所有 JSON 都经过 :func:`atomic_write_json` 的 tmp + rename，
crash 不会留下 half-written file 污染下次 resume；``run_meta.json`` 保存 config snapshot，按
SOP §0 禁止 mid-run drift，并保存 one-way ``unsealed_at``。test number 一旦揭示，继续 evolution
会让 decision 泄漏 sealed information，因此 default 必须拒绝 resume。

durable write 成功只说明 lifecycle metadata 落盘，不说明 benchmark artifact 完整或 run 成功。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

META_FILENAME = "run_meta.json"


def atomic_write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_json_or(path: Path, default: Any) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def config_fingerprint(snapshot: dict) -> str:
    """计算 effective run config 的 order-insensitive stable fingerprint。

    dict 以 sorted JSON canonical-like encoding 序列化，返回 SHA-256 前 16 hex。它用于 drift
    detection，不是 security signature。
    """
    canon = json.dumps(snapshot, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class RunMeta:
    """位于 ``work_dir/run_meta.json`` 的 per-run lifecycle record。

    拥有 creation time、config snapshot/hash、optional unseal time 与 finalize reason。对象可由
    ``load`` 恢复或 ``create`` 原子生成；``stamp_unsealed`` 是 one-way 状态写入。
    """

    work_dir: Path
    created_at: str = ""
    config_snapshot: dict = field(default_factory=dict)
    config_hash: str = ""
    unsealed_at: Optional[str] = None
    finalize_reason: Optional[str] = None

    @property
    def path(self) -> Path:
        return Path(self.work_dir) / META_FILENAME

    @classmethod
    def load(cls, work_dir: Path) -> Optional["RunMeta"]:
        data = load_json_or(Path(work_dir) / META_FILENAME, None)
        if not isinstance(data, dict):
            return None
        return cls(
            work_dir=Path(work_dir),
            created_at=data.get("created_at", ""),
            config_snapshot=data.get("config_snapshot", {}),
            config_hash=data.get("config_hash", ""),
            unsealed_at=data.get("unsealed_at"),
            finalize_reason=data.get("finalize_reason"),
        )

    @classmethod
    def create(cls, work_dir: Path, snapshot: dict) -> "RunMeta":
        meta = cls(
            work_dir=Path(work_dir),
            created_at=_utcnow(),
            config_snapshot=snapshot,
            config_hash=config_fingerprint(snapshot),
        )
        meta.save()
        return meta

    def save(self) -> None:
        atomic_write_json(
            self.path,
            {
                "created_at": self.created_at,
                "config_snapshot": self.config_snapshot,
                "config_hash": self.config_hash,
                "unsealed_at": self.unsealed_at,
                "finalize_reason": self.finalize_reason,
            },
        )

    def check_config(self, snapshot: dict) -> bool:
        """仅当 ``snapshot`` fingerprint 匹配 recorded config 时返回 ``True``。"""
        return config_fingerprint(snapshot) == self.config_hash

    def stamp_unsealed(self, reason: str) -> None:
        """写入 one-way unseal stamp；之后 ``run`` 无 ``--force`` 必须拒绝 resume。

        同时保存 ``finalize_reason`` 并 atomic write meta。该方法不执行 sealed evaluator。
        """
        self.unsealed_at = _utcnow()
        self.finalize_reason = reason
        self.save()


__all__ = [
    "RunMeta",
    "atomic_write_json",
    "config_fingerprint",
    "load_json_or",
    "META_FILENAME",
]
