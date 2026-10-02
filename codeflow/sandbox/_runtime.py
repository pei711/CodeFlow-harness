"""Internal Helper：构造 Rooted at CodeFlow Data Dir 的 BoxLite Runtime。

CodeFlow 内所有 BoxLite Usage，包括 `BoxliteExecutor` 与 `SandboxDebugServer`，都通过这里取得 Runtime，
使其 ``home_dir`` 中的 DB、Images、Layers 位于 ``<data_dir>/sandbox/boxlite``，而不是 BoxLite 默认
``~/.boxlite``。

Runtime 按 ``(Boxlite class, home_dir)`` Memoised。BoxLite Rust Core 会为每个 Home Dir 取得
Process-wide Filesystem Lock，且只有 ``Boxlite`` Instance 被 Drop 才释放；每次调用都新建 Instance 会
与仍存活的前一个 Runtime 冲突，并 Panic：
``Another BoxliteRuntime is already using directory: …``。

Cache Key 不只包含 Home Dir，还包含 Class Object。这样 Unit Tests 用
``mock.patch("boxlite.Boxlite")`` 时，每次 Patch 都取得新的 Mocked Runtime，而不会复用先前测试缓存的
Real BoxLite。Helper 只创建 Runtime，不启动具体 VM。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import boxlite as _boxlite_t

_runtime_cache: dict[tuple[int, str], Any] = {}


def get_boxlite_runtime() -> "_boxlite_t.Boxlite":
    import boxlite

    from codeflow.config.paths import get_sandbox_dir

    home = str(get_sandbox_dir("boxlite"))
    key = (id(boxlite.Boxlite), home)
    rt = _runtime_cache.get(key)
    if rt is None:
        rt = boxlite.Boxlite(boxlite.Options(home_dir=home))
        _runtime_cache[key] = rt
    return rt
