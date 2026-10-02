"""处理 ``terminal.resize`` RPC：记录最新 cols/rows，并始终返回 ok。

ui-tui 的 ``useMainApp.ts:426`` 在 Ink 观察到 SIGWINCH 时，以 fire-and-forget 方式调用
``terminal.resize``，payload 为新的 ``{cols, rows}``。handler 有两个约束：第一，永不抛出，
否则 SIGWINCH burst 会产生大量 error frame；第二，为 active TUI Session 记录最近尺寸。

状态保存在 module level，因为单个 TUI subprocess 恰好对应一个 terminal，singleton 在此
是正确所有权。resize 成功只表示尺寸 snapshot 已更新，不表示终端绘制完成或 Agent 状态
发生变化。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from codeflow.tui_rpc.dispatcher import Dispatcher


# 模块级的最新已知终端尺寸。``None`` 表示尚未观察到 resize 事件；
# 此时调用方应回退到 ``shutil.get_terminal_size()`` 或合理默认值（80 列）。
_LATEST_COLS: int | None = None
_LATEST_ROWS: int | None = None


def get_latest_cols() -> int | None:
    """返回最近上报的 terminal column count，尚未观察到时返回 ``None``。

    TUI Runtime 可用该值计算 terminal output 宽度；``None`` 时应回退到
    ``shutil.get_terminal_size()`` 或 80 列等合理默认值。函数只读取 snapshot。
    """
    return _LATEST_COLS


def get_latest_rows() -> int | None:
    """返回最近上报的 terminal row count，尚未观察到时返回 ``None``。

    返回值是进程内 snapshot，可能在下一次 SIGWINCH 后立即变化。
    """
    return _LATEST_ROWS


def _coerce_dim(value: Any) -> int | None:
    """把 ``value`` 接纳为 positive int，不合规时返回 ``None``。

    bool 虽是 int 子类也必须拒绝；zero、negative 与其他类型同样返回 ``None``。该约定让
    resize handler 可以静默忽略坏字段，而不抛出协议错误。
    """
    if isinstance(value, bool):
        # bool 是 int 的子类，需显式拒绝。
        return None
    if isinstance(value, int) and value > 0:
        return value
    return None


async def terminal_resize(params: dict) -> dict:
    """执行 ``terminal.resize``，记录尺寸并返回 ``{ok: true}``。

    接受 ``{cols, rows}``，两项都是可选 positive int。合法字段独立更新，所以一项无效不
    影响另一项；非 dict 或其他输入静默退化为 no-op。这里有意永不抛出，因为 upstream
    SIGWINCH burst 否则会淹没 error frame。``ok`` 表示事件已安全处理，不保证字段有效。
    """
    global _LATEST_COLS, _LATEST_ROWS
    if isinstance(params, dict):
        cols = _coerce_dim(params.get("cols"))
        rows = _coerce_dim(params.get("rows"))
        if cols is not None:
            _LATEST_COLS = cols
        if rows is not None:
            _LATEST_ROWS = rows
    return {"ok": True}


def register_terminal_methods(dispatcher: "Dispatcher") -> None:
    """在 Dispatcher 上注册 ``terminal.resize``。

    注册不读取或修改当前尺寸；重复注册由 Dispatcher 抛出 ``ValueError``。
    """
    dispatcher.register("terminal.resize", terminal_resize)


__all__ = [
    "terminal_resize",
    "register_terminal_methods",
    "get_latest_cols",
    "get_latest_rows",
]
