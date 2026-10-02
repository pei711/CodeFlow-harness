"""Crash-safe JSONL File Primitives：Locked Append、Replace 与 Delete。

这些 Helpers 使用 Advisory Lock 串行化 Cross-process Mutations。Sidecar Lock 放在目标 Parent 下隐藏的
``.lock/`` 子目录；Process Death 时自动释放，因此无需 Stale-lock Cleanup。底层锁跨平台，
``portalocker`` 在 POSIX 使用 ``fcntl``、在 Windows 使用 ``LockFileEx``，所以 Windows Concurrent
Writers 也会被正确串行化。

除互斥外，模块用持久化 Epoch 为删除和重建建立 Generation Fence，防止持有旧 Snapshot 的 Writer
把已删除文件悄悄复活。调用方必须携带读取时取得的 Expected Epoch，才能得到这一并发保护。
"""

import os
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from codeflow.utils.portable_lock import file_lock


class StorageCorruptionError(ValueError):
    """表示 Persistent Generation Metadata 缺失或 Invalid。

    这不是普通的“文件尚未创建”，而是系统已经知道该 Path 受 Epoch 管理，却无法可信读取其代际。
    调用方应停止写入并报告损坏，不能把它当成零代继续覆盖。
    """


def _epoch_path(path: Path) -> Path:
    return path.parent / ".generation" / f"{path.name}.epoch"


def _epoch_known_path(path: Path) -> Path:
    return path.parent / ".generation" / f"{path.name}.known"


def read_epoch(path: Path) -> int:
    """读取 Deletion Epoch，过程中不创建 Lock 或 Metadata Files。

    Epoch File 不存在且 Path 从未被标记为 Known 时返回 0，表示合法的初始代；Known Marker 已存在却
    缺 Epoch、编码非法、内容不是非负整数时抛出 `StorageCorruptionError`。该读取是只读 Snapshot，
    需要与 Mutation 原子协调时应通过 `locked_read`。
    """
    epoch_path = _epoch_path(path)
    try:
        raw = epoch_path.read_text(encoding="ascii")
    except FileNotFoundError:
        if _epoch_known_path(path).exists():
            raise StorageCorruptionError(f"missing deletion epoch: {epoch_path}")
        return 0
    except UnicodeError as exc:
        raise StorageCorruptionError(f"invalid deletion epoch: {epoch_path}") from exc

    try:
        epoch = int(raw.strip())
    except ValueError as exc:
        raise StorageCorruptionError(f"invalid deletion epoch: {epoch_path}") from exc
    if epoch < 0:
        raise StorageCorruptionError(f"invalid deletion epoch: {epoch_path}")
    return epoch


def epoch_is_known(path: Path) -> bool:
    """返回 Path 是否已有 Durable Generation Metadata。

    Known Marker 或 Epoch File 任一个存在即返回 `True`。它只说明该路径进入过代际管理，不验证两者
    是否一致；完整一致性由 `read_epoch` 与 Mutation Checks 负责。
    """
    return _epoch_known_path(path).exists() or _epoch_path(path).exists()


def read_utf8_with_incomplete_tail(path: Path) -> str:
    """解码 UTF-8，并把 Incomplete Final Code Point 保留为 Junk Marker。

    完整 Payload 正常解码；只有文件末尾因 Crash 截断一个 UTF-8 Code Point 时，才保留此前合法文本并
    追加替换字符 ``\ufffd``。中间位置或其他原因的 `UnicodeDecodeError` 仍向上抛出，避免把真正损坏
    静默伪装成可恢复 Tail。
    """
    payload = path.read_bytes()
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        if exc.reason != "unexpected end of data" or exc.end != len(payload):
            raise
        return payload[: exc.start].decode("utf-8") + "\ufffd"


def _mark_epoch_known(path: Path) -> None:
    known_path = _epoch_known_path(path)
    if known_path.exists():
        return
    known_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = known_path.with_name(known_path.name + ".tmp")
    with open(tmp_path, "w", encoding="ascii") as f:
        f.write("1")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, known_path)


def _write_epoch(path: Path, epoch: int) -> None:
    epoch_path = _epoch_path(path)
    _mark_epoch_known(path)
    tmp_path = epoch_path.with_name(epoch_path.name + ".tmp")
    with open(tmp_path, "w", encoding="ascii") as f:
        f.write(str(epoch))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, epoch_path)


def _ensure_epoch(path: Path, epoch: int) -> None:
    if not _epoch_path(path).exists():
        _write_epoch(path, epoch)


def _restore_epoch(
    path: Path,
    epoch: int,
    *,
    epoch_existed: bool,
    known_existed: bool,
    epoch_dir_existed: bool,
) -> None:
    epoch_path = _epoch_path(path)
    if epoch_existed:
        _write_epoch(path, epoch)
    else:
        epoch_path.unlink(missing_ok=True)
    if not known_existed:
        _epoch_known_path(path).unlink(missing_ok=True)
    if not epoch_dir_existed:
        try:
            epoch_path.parent.rmdir()
        except OSError:
            pass


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / ".lock" / (path.name + ".lock")
    with file_lock(lock_path):
        yield


def _check_epoch(
    path: Path,
    *,
    expected_epoch: int | None,
    operation: str,
) -> int:
    current = read_epoch(path)
    if not path.exists() and epoch_is_known(path) and current == 0:
        raise StorageCorruptionError(f"known generation is missing primary file: {path}")
    if expected_epoch is not None and current != expected_epoch:
        raise FileNotFoundError(f"file was deleted or replaced before {operation}: {path}")
    return current


def locked_read(path: Path) -> tuple[str | None, int, bool]:
    """在 Mutation Lock 下读取文本与 Generation State。

    返回 ``(raw, epoch, known)``：主文件缺失时 `raw` 为 `None`，`epoch` 是当前代，`known` 表示是否已
    有持久化代际元数据。完全未知且不存在的 Path 可直接返回合法 Epoch-zero Snapshot，不实际创建
    Lock；其他情况与 Writer 使用同一 Sidecar Lock，保证三个值来自一致观察点。
    """
    # 写入方会先发布已知标记，再发布主文件，因此主文件缺失代表合法的
    # epoch-zero 快照，无需实际创建锁。
    if not path.exists() and not epoch_is_known(path):
        return None, 0, False
    with _locked(path):
        raw = read_utf8_with_incomplete_tail(path) if path.exists() else None
        return raw, read_epoch(path), epoch_is_known(path)


def locked_append(
    path: Path,
    lines: list[str],
    *,
    expected_epoch: int | None = None,
    require_existing: bool = False,
    validate_existing: Callable[[str], None] | None = None,
) -> int:
    """在 Lock 内可选验证 Existing Bytes 后，Append 一个完整 Block。

    空 `lines` 不写盘，只返回当前 Epoch。非空时可要求文件已经存在、校验 Expected Epoch，并用
    `validate_existing` 检查锁内读到的旧内容；任何条件变化都在写入前失败。写入会 Flush + Fsync，
    如果 Crash 曾留下无换行 Partial Line，会先补换行避免新旧 Records 粘连。返回值是写入所在 Epoch，
    本操作不递增 Generation。
    """
    if not lines:
        return read_epoch(path)
    with _locked(path):
        exists = path.exists()
        if require_existing and not exists:
            raise FileNotFoundError(f"file was deleted before append: {path}")
        epoch = _check_epoch(
            path,
            expected_epoch=expected_epoch,
            operation="append",
        )
        if exists and validate_existing is not None:
            validate_existing(read_utf8_with_incomplete_tail(path))
        _ensure_epoch(path, epoch)
        with open(path, "a+b") as f:
            payload = "".join(line + "\n" for line in lines).encode("utf-8")
            # 写入方崩溃后可能留下没有换行符的残缺行；从新行开始写入，
            # 避免两条记录粘连。
            if f.tell() > 0:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    payload = b"\n" + payload
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        return epoch


def atomic_replace(
    path: Path,
    data: str,
    *,
    expected_epoch: int | None = None,
    expected_exists: bool | None = None,
    require_existing: bool = False,
    increment_epoch: bool = False,
    validate_existing: Callable[[str], None] | None = None,
) -> int:
    """原子 Replace ``path``，并可选择拒绝 Recreate 已删除文件。

     函数在同一 Lock 内验证 Expected Existence、Expected Epoch 与可选 Existing-content Validator，先把
    新数据写入并 Fsync Temp File，再用 `os.replace` 发布。`increment_epoch=True` 可同时推进
     Generation；若最终 Replace 失败，会尽力恢复先前 Epoch Metadata，避免 Fence 与 Primary File
     状态分离。返回发布后的 Epoch。
    """
    with _locked(path):
        exists = path.exists()
        if expected_exists is not None and exists != expected_exists:
            raise FileNotFoundError(f"file existence changed before replace: {path}")
        if require_existing and not exists:
            raise FileNotFoundError(f"file was deleted before replace: {path}")
        epoch = _check_epoch(
            path,
            expected_epoch=expected_epoch,
            operation="replace",
        )
        if exists and validate_existing is not None:
            validate_existing(read_utf8_with_incomplete_tail(path))
        tmp_path = path.with_name(path.name + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

        epoch_path = _epoch_path(path)
        epoch_existed = epoch_path.exists()
        known_existed = _epoch_known_path(path).exists()
        epoch_dir_existed = epoch_path.parent.exists()
        next_epoch = epoch + 1 if increment_epoch else epoch
        _write_epoch(path, next_epoch)
        try:
            os.replace(tmp_path, path)
        except OSError:
            _restore_epoch(
                path,
                epoch,
                epoch_existed=epoch_existed,
                known_existed=known_existed,
                epoch_dir_existed=epoch_dir_existed,
            )
            raise
        return next_epoch


def locked_delete(
    path: Path,
    *,
    expected_epoch: int | None = None,
    expected_exists: bool | None = None,
    fence_missing: bool = False,
    increment_epoch: bool = False,
) -> bool:
    """持有与 Writers 相同的 Lock 删除 ``path``。

    可校验 Expected Existence 与 Epoch；`fence_missing=True` 允许即使 Primary File 已缺失也建立
    Generation Fence，`increment_epoch=True` 则使旧 Writer 的 Expected Epoch 失效。真实删除成功
    返回 `True`，文件本来不存在或竞态中已消失返回 `False`。若 Unlink 发生其他 OS Error，会在需要
    时恢复原 Epoch 后重新抛出，避免失败删除留下虚假新代。
    """
    with _locked(path):
        exists = path.exists()
        if expected_exists is not None and exists != expected_exists:
            raise FileNotFoundError(f"file existence changed before delete: {path}")
        prior_epoch = _check_epoch(
            path,
            expected_epoch=expected_epoch,
            operation="delete",
        )
        if not exists and not fence_missing:
            return False

        epoch_path = _epoch_path(path)
        epoch_existed = epoch_path.exists()
        known_existed = _epoch_known_path(path).exists()
        epoch_dir_existed = epoch_path.parent.exists()
        if increment_epoch:
            _write_epoch(path, prior_epoch + 1)
        if not exists:
            return False

        try:
            path.unlink()
        except FileNotFoundError:
            return False
        except OSError:
            if increment_epoch:
                _restore_epoch(
                    path,
                    prior_epoch,
                    epoch_existed=epoch_existed,
                    known_existed=known_existed,
                    epoch_dir_existed=epoch_dir_existed,
                )
            raise
        return True
