"""把 Model Catalog 持久化到磁盘，内容包括 Per-model Pricing 与 Context Window。

缓存是 ``~/.codeflow/cache/`` 下一个带 Version 的 JSON 文件，通过 Temp-file + ``os.replace`` 原子写入，
保证并发 Multi-process Readers 不会看到 Torn File。Catalog 是可丢弃、可重新 Fetch 的 Whole-blob
Cache，Authoritative Source 在网络端；即使两个 Writer 发生 Lost Write Race，代价也只是下一次多做
一次 Refetch，不会丢失唯一业务数据，所以无需 File Lock。

模块按它持久化的对象 Model Catalog 命名，而不是按当前 Source 命名。Storage Layer 保持
Source-agnostic，未来更换 Catalog Source 仍可复用。这里仅负责磁盘存储；Freshness TTL、In-process
Tier 与真实 Fetch 都由 Caller 管理，参见 ``pricing._fetch_openrouter_models``。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from loguru import logger

from codeflow.config.paths import get_cache_dir

# schema 变更后递增此值，以强制失效所有磁盘缓存文件。
CACHE_VERSION = 1
CACHE_FILENAME = "model-catalog.json"

# 测试接缝：设置后覆盖磁盘路径，确保测试不会触碰真实的 ~/.codeflow/cache/。
# None 表示稍后通过 get_cache_dir() 惰性计算。
_CACHE_PATH: Path | None = None


def cache_path() -> Path:
    """解析 On-disk Catalog Path，并遵守 Test Override。

    `_CACHE_PATH` 非 `None` 时原样返回测试指定路径，避免测试触碰真实用户缓存；否则惰性组合
    `get_cache_dir()` 与 `CACHE_FILENAME`。函数只计算路径，不创建目录或检查文件存在。
    """
    if _CACHE_PATH is not None:
        return _CACHE_PATH
    return get_cache_dir() / CACHE_FILENAME


def load() -> tuple[dict[str, dict], float] | None:
    """从磁盘返回 ``(models, fetched_at)``，无法使用时返回 `None`。

    Missing、Unparseable、Malformed 或 Wrong-version File 都视为 Cache Miss；读取缓存绝不能把异常
    传播进 Cost Path。成功返回前会验证 Version、Models 必须是 Dict、Fetched Timestamp 必须可转成
    Float，但不深度校验每个模型字段，具体定价解析仍由上层负责。
    """
    try:
        path = cache_path()
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if raw.get("version") != CACHE_VERSION:
        return None
    models = raw.get("models")
    if not isinstance(models, dict):
        return None
    try:
        fetched_at = float(raw.get("fetched_at", 0.0))
    except (TypeError, ValueError):
        return None
    return models, fetched_at


def save(models: dict[str, dict]) -> None:
    """以 Temp File + ``os.replace`` 原子持久化 Catalog，采用 Best-effort 语义。

    Temp File 位于目标的 Same Directory，因而同一 Filesystem 上 Rename 具有 POSIX-atomic 保证；
    文件名带 PID，避免 Racing Writers 互相覆盖临时文件。Lost Rename Race 最多导致额外一次 Fetch，
    不会丢失唯一数据。目录创建、序列化或替换失败时仅记录 Debug 并跳过，定价主路径不会因此失败。
    """
    try:
        path = cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CACHE_VERSION,
            "fetched_at": time.time(),
            "models": models,
        }
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except Exception as exc:
        logger.debug("model_catalog_cache: failed to persist disk cache ({}), skipping", exc)
