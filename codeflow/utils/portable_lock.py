"""Cross-platform Advisory File Lock。

通过 ``portalocker`` 在相邻 ``.lock`` Anchor 上取得 Exclusive Lock，串行化写入同一 Shared File 的
Cross-process 与 Cross-thread Writers。底层在 POSIX 使用 ``fcntl.flock``，在 Windows 使用
``LockFileEx``。

它替代了过去只支持 ``fcntl`` 的 Lock Paths；旧实现会在 Windows 因 ``import fcntl`` 触发
`ImportError` 或 ``sys.platform == "win32"`` No-op Branch 而静默退化成 *Unlocked*，导致 Concurrent
Writes 丢失。Advisory 表示所有参与者都必须自愿使用同一 Anchor，未加锁进程仍可直接修改目标文件。
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import portalocker


class LockTimeoutError(RuntimeError):
    """Non-blocking Acquire 遇到另一个 Holder 已持锁时抛出。

    该异常把底层 `portalocker` 差异统一成项目协议。它表示当前没有取得互斥权，调用方不得继续执行受
    保护的 Mutation；可以选择稍后重试或向上报告冲突。
    """


@contextmanager
def file_lock(lock_path: Path, *, blocking: bool = True) -> Iterator[None]:
    """在 Context Block 生命周期内持有 ``lock_path`` 的 Exclusive Advisory Lock。

    Anchor File 在首次使用时创建，父目录也会自动建立。`blocking=True` 会等待其他 Holder 释放；
    `blocking=False` 时若锁已被其他 Process 持有，立即抛出 :class:`LockTimeoutError`。离开 Block 时无论
    正常返回还是内部抛错都会尝试 Unlock 并关闭文件；Unlock 异常被吞掉，避免遮蔽 Block 原始异常。
    """
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = portalocker.LOCK_EX
    if not blocking:
        flags |= portalocker.LOCK_NB

    with open(lock_path, "a+") as fh:
        try:
            portalocker.lock(fh, flags)
        except portalocker.exceptions.LockException as exc:
            raise LockTimeoutError(f"lock already held: {lock_path}") from exc
        try:
            yield
        finally:
            try:
                portalocker.unlock(fh)
            except Exception:
                pass


__all__ = ["file_lock", "LockTimeoutError"]
