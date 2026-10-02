"""为 :class:`SkillRegistry` 提供 Background Filesystem Watcher。

Watcher 监控 Workspace Skill Tree；``SKILL.md`` Add/Change/Disappear 时按 Source Invalidate Registry Cache，
使手工编辑 ``<workspace>/skills/foo/SKILL.md`` 无需 Process Restart 即进入 In-process Selector。它补充
其他 In-process Writers 历史调用 :meth:`SkillService.invalidate_skill_cache`、当前调用 Catalog
Invalidation Hook。Watcher Roots 通过 ``__init__`` 传入。

Design Notes：使用运行 ``watchfiles.watch()`` 的 **Daemon Thread**，Rust Iterator 默认约 1.6s Debounce；
Daemon 在 Process Exit 自动清理，显式 :meth:`stop` 用于 Tests/Clean Shutdown。Scope 刻意 Workspace-only，
Builtin/External 是 Read-only Mirrors，Builtin 约可达 80K Files，Recursive Watch 会超过 Linux
``fs.inotify.max_user_watches``。``watchfiles`` 缺失时 Defensive `ImportError` 降级 Manual Invalidation。
Start/Stop 与 Thread 内 Error 都是 **Best-effort**，失败返回 ``False`` 或记录后不让 Runtime 崩溃。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable
from pathlib import Path, PurePath

log = logging.getLogger(__name__)


def _is_skill_md(_change, path: str) -> bool:
    """``watchfiles`` Filter，只保留 ``SKILL.md`` Events。

    使用 :class:`PurePath` 做 Basename Check，避免依赖特定 OS/Version 返回 ``/`` 或 ``\\`` Separator。
    ``PurePath`` 在 Linux/macOS 选择 ``PurePosixPath``，在 Windows 选择 ``PureWindowsPath``。Change Kind 不参与过滤，Create、
    Modify、Delete 都需要 Invalidate。
    """
    return PurePath(path).name == "SKILL.md"


class SkillFileWatcher:
    """把 File Events 送入 Registry Invalidation 的 Daemon-thread Watcher。

    Lifecycle：:meth:`start` Idempotent 且 Never Raises，只有 New Daemon Thread Running 才返回 `True`；
    :meth:`stop` Signal Thread Exit 并 Best-effort Join，可重复调用。实例持有 Existing Roots、Source
    Resolver、On-change Callback 与 Stop Event；一个 Callback Failure 不终止后续 Batch。
    """

    def __init__(
        self,
        roots: Iterable[Path],
        on_change: Callable[[str], None],
        resolve_source: Callable[[Path], str | None],
    ):
        # 过滤 None 和不存在的项，让调用方可以直接传入可选的分层根目录
        # （如 ``external_skills``），无需预先检查。
        self._roots = [r for r in roots if r is not None and r.exists()]
        self._on_change = on_change
        self._resolve_source = resolve_source
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        """启动 Daemon Watcher Thread。

        Just Started 时返回 `True`。``watchfiles`` 未安装、Disk 无 Watchable Roots、或 Thread Already Running
        时返回 `False` 并保持 No-op。Start 会 Clear Stop Event、创建 Named Daemon Thread 后立即返回，不
        等待首次 Watch Batch。
        """
        if self._thread is not None and self._thread.is_alive():
            return False
        if not self._roots:
            log.debug("SkillFileWatcher: no existing roots, not starting")
            return False
        try:
            import watchfiles  # noqa: F401
        except ImportError:
            log.info(
                "watchfiles not installed — SkillRegistry auto-refresh "
                "disabled. Reinstall codeflow-harness with the official CodeFlow "
                "installer to enable it."
            )
            return False

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="SkillFileWatcher",
            daemon=True,
        )
        self._thread.start()
        log.debug(
            "SkillFileWatcher started on %d root(s): %s",
            len(self._roots),
            [str(r) for r in self._roots],
        )
        return True

    def stop(self, timeout: float = 1.0) -> None:
        """Signal Watcher Exit，并在 ``timeout`` 内 Best-effort Join。

        Never Started 或重复调用都安全。无论 Join 是否在时限内完成都会清空 `_thread` Reference；Daemon
        Thread 最终仍可在 Process Exit 被系统回收。
        """
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None

    # ------------------------------------------------------------------

    def _run(self) -> None:
        # 在线程内导入，使 ``start()`` 的 ImportError 检查成为“依赖是否存在”的唯一事实来源。
        from watchfiles import watch

        try:
            for changes in watch(
                *self._roots,
                watch_filter=_is_skill_md,
                stop_event=self._stop,
                # 这是大型应用内的守护线程，不让主线程中的 Ctrl+C 在此处合成 KeyboardInterrupt。
                raise_interrupt=False,
            ):
                # 触发回调前先将批次收敛为不重复的数据源，使多文件保存（如 git checkout
                # 一次恢复多个 SKILL.md）每个数据源只需一次失效，而不是每个文件一次。
                dirty: set[str] = set()
                for _change, raw_path in changes:
                    source = self._resolve_source(Path(raw_path))
                    if source is not None:
                        dirty.add(source)
                for source in dirty:
                    try:
                        self._on_change(source)
                    except Exception:
                        # 单个异常回调不能终止监视器，下一批可能针对另一个数据源。
                        log.exception(
                            "SkillFileWatcher on_change failed for source=%s",
                            source,
                        )
        except Exception:
            # 其他任何失败（watchfiles 内部错误、文件系统消失等）都会让线程终止，
            # 但注册表仍可以在手动失效模式下使用。
            log.exception("SkillFileWatcher crashed; auto-refresh disabled until restart")
