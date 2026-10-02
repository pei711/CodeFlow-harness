"""实现 CodeFlow TUI 所需的轻量异步 JSON-RPC 2.0 分派器。

模块接收已经解析为 Python object 的 request frame，校验 ``jsonrpc``、``method`` 和
``params``，查找异步 handler，再把返回值或异常包装成 JSON-RPC response。newline-
delimited JSON framing 属于 :mod:`codeflow.tui_rpc.server`，Pydantic v2 参数校验属于各
``methods`` handler，subscription registry 与 16ms throttle 则属于订阅层。

design.md §3 D8 的决定是使用自写约 30 行核心分派逻辑，而不是引入 ``ajsonrpc`` 或
``jsonrpcserver``：Pydantic v2 已覆盖 schema validation，framing 很简单，自定义订阅
side-channel 仍需单独实现，且更少依赖意味着更小的 ``pip-audit`` surface。若未来
cancellation/error edge cases 超出可维护预算，fallback 是切换到 ``ajsonrpc``，复用其
dispatch，并手工补上 notification side-channel。

成功 response 只表示 handler 返回了合规 dict；它不自动证明 handler 的副作用已
持久化、Agent 任务完成或最终回复已交付。
"""

from __future__ import annotations

import inspect
import traceback
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from codeflow.tui_rpc.errors import (
    INTERNAL_ERROR,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    RpcError,
)

Handler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class Dispatcher:
    """把 JSON-RPC 2.0 request frame 路由到已注册的异步 handler。

    实例拥有 ``method -> Handler`` 映射；方法注册通常发生在 ``RpcServer`` 启动阶段，
    随后 ``dispatch()`` 只读该映射。所有 handler 必须是 async，并返回可放入
    ``result`` 的 dict。``RpcError`` 会保留协议错误码，未处理异常则收敛成
    ``INTERNAL_ERROR``，避免单个方法使 server 主循环退出。

    使用示例::

        d = Dispatcher()
        d.register("system.hello", system_hello)
        response = await d.dispatch(request_frame)

    Dispatcher 不解析 stdin 字节、不管理 subscription，也不拥有 Runtime 状态；它只对
    单帧调用的协议形状负责。
    """

    def __init__(self) -> None:
        self._handlers: dict[str, Handler] = {}

    def register(self, method: str, handler: Handler) -> None:
        """注册一个 JSON-RPC ``method`` 对应的异步 ``handler``。

        ``handler`` 必须是 coroutine function，否则抛出 ``TypeError``；同名 method
        已存在时抛出 ``ValueError``，防止后注册代码静默覆盖协议行为。注册成功无返回值，
        也不会立即调用 handler 或验证它未来的 result schema。
        """
        if not inspect.iscoroutinefunction(handler):
            raise TypeError(f"handler for '{method}' must be async; got {type(handler).__name__}")
        if method in self._handlers:
            raise ValueError(f"method '{method}' already registered")
        self._handlers[method] = handler

    def methods(self) -> list[str]:
        """返回当前已注册 method 名称的排序副本。

        排序使 capability 握手和测试结果稳定；调用方修改返回 list 不会改变内部 handler
        映射。列表只能证明方法已注册，不能证明相关 Runtime 依赖已经就绪。
        """
        return sorted(self._handlers)

    async def dispatch(self, frame: dict[str, Any]) -> dict[str, Any]:
        """校验并分派一帧请求，按 JSON-RPC 2.0 包装 result 或 error。

        ``frame`` 应是解析后的 object。非 object、版本错误、method 缺失、params 非 object
        或 method 未注册时不会调用 handler，而是返回相应协议错误。``params=None`` 被
        归一化为空 dict。handler 的 ``RpcError`` 保留 code/message/data；``SystemExit``
        和其他异常转成 ``INTERNAL_ERROR``，并只暴露截断后的 traceback tail。

        每条路径都返回 response dict。按照规范，parse error 与无法恢复 ``id`` 的帧仍返回
        ``id: null``，上层调用者可以选择丢弃。handler 返回非 dict 也视为内部错误。正常
        result 仅表示该调用完成，不能单独推出状态已持久化或任务已完成。
        """
        # ----- 帧校验 -----------------------------------------------------
        if not isinstance(frame, dict):
            return _err_frame(None, PARSE_ERROR, "parse_error", data={"reason": "frame is not an object"})

        frame_id = frame.get("id")  # 通知帧的 ID 可以为 None

        if frame.get("jsonrpc") != "2.0":
            return _err_frame(
                frame_id,
                INVALID_REQUEST,
                "invalid_request",
                data={"reason": "missing or wrong jsonrpc version"},
            )

        method = frame.get("method")
        if not isinstance(method, str) or not method:
            return _err_frame(
                frame_id,
                INVALID_REQUEST,
                "invalid_request",
                data={"reason": "missing or non-string method"},
            )

        params = frame.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return _err_frame(
                frame_id,
                INVALID_REQUEST,
                "invalid_request",
                data={"reason": "params must be an object"},
            )

        # ----- 方法路由 -------------------------------------------------------
        handler = self._handlers.get(method)
        if handler is None:
            return _err_frame(frame_id, METHOD_NOT_FOUND, "method_not_found", data={"method": method})

        # ----- 分派 -------------------------------------------------------------
        try:
            result = await handler(params)
        except RpcError as exc:
            err_payload: dict[str, Any] = {
                "code": exc.code,
                "message": exc.message,
            }
            if exc.data is not None:
                err_payload["data"] = exc.data
            elif exc.detail:
                err_payload["data"] = {"detail": exc.detail}
            return {"jsonrpc": "2.0", "id": frame_id, "error": err_payload}
        except SystemExit as exc:
            # 即使 standalone_mode=False，Click/Typer 仍可能泄漏 SystemExit；
            # 将其视为内部错误，不让分派器崩溃。
            tb_tail = _truncate_traceback(traceback.format_exc())
            logger.warning("tui_rpc: SystemExit in handler {}: code={}", method, exc.code)
            return _err_frame(
                frame_id,
                INTERNAL_ERROR,
                "internal_error",
                data={"reason": "SystemExit from handler", "traceback_tail": tb_tail},
            )
        except Exception:
            tb_tail = _truncate_traceback(traceback.format_exc())
            logger.exception("tui_rpc: unhandled exception in handler {}", method)
            return _err_frame(
                frame_id,
                INTERNAL_ERROR,
                "internal_error",
                data={"traceback_tail": tb_tail},
            )

        # ----- 结果校验 ---------------------------------------------------
        if not isinstance(result, dict):
            logger.error("tui_rpc: handler {} returned non-dict result", method)
            return _err_frame(
                frame_id,
                INTERNAL_ERROR,
                "internal_error",
                data={"reason": f"handler returned {type(result).__name__}, expected dict"},
            )

        return {"jsonrpc": "2.0", "id": frame_id, "result": result}


def _err_frame(
    frame_id: Any,
    code: int,
    message: str,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": frame_id, "error": err}


def _truncate_traceback(tb: str, max_lines: int = 12) -> str:
    """把格式化 traceback 截断为末尾 ``max_lines`` 行。

    traceback 未超过上限时保持全部行；超过时以前缀 ``...`` 表示省略。保留尾部是因为
    通常包含最终异常类型和最接近失败点的 stack frame。该字符串会进入 RPC error data，
    只用于诊断，不应被调用方解析为稳定协议字段。
    """
    lines = tb.strip().splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "...\n" + "\n".join(lines[-max_lines:])


__all__ = ["Dispatcher", "Handler"]
