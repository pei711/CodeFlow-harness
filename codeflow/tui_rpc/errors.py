"""定义映射到 JSON-RPC 2.0 error code 的 Runtime 异常类型。

handler 不应把 Python 异常细节随意暴露给 TUI，而应抛出这里的 ``RpcError`` 子类；
``Dispatcher`` 随后读取稳定的 ``CODE``、``MESSAGE`` 和可选 ``data`` 形成协议 error。
code table 冻结在 ``specs/tui-ipc.md`` §4，server-defined range 为
``-32000..-32099``：

| code   | message                       | 含义                             |
|--------|-------------------------------|----------------------------------|
| -32001 | session_not_found             | session_key unknown              |
| -32002 | session_locked                | session held by another client   |
| -32003 | turn_in_progress              | session currently running a turn |
| -32008 | model_not_available           | model_id not routable            |
| -32009 | model_switch_in_turn          | switch attempt while turn live   |
| -32010 | config_field_readonly         | not on hot-changeable whitelist  |
| -32011 | config_validation_error       | Pydantic / semver validation     |
| -32012 | not_supported_in_v01          | unsupported provider auth action |

JSON-RPC pre-defined codes ``-32700/-32600/-32601/-32602`` 由 dispatcher 直接
产生，没有专用 exception class。``-32603 internal_error`` 也用于未捕获的 handler
exception，但仍提供 ``InternalError``，使 dispatcher 之外的路径——尤其是由
``_spawn_agent_loop_task`` 调用的 ``_build_tui_agent_loop`` factory——可以跨模块抛出
有类型的 ``-32603``。没有它时，初始化 crash 会被错误混同为 ``-32008
model_not_available``。

协议 error 只说明调用失败的分类；它本身不证明 Runtime 是否已经回滚副作用，也不能
用来推断 Session 持久化、任务完成或最终交付状态。
"""

from __future__ import annotations

from typing import Any, ClassVar


class RpcError(Exception):
    """所有可序列化为 JSON-RPC error frame 的异常基类。

    子类通过 class-level ``CODE`` 和 ``MESSAGE`` 声明稳定协议身份，dispatcher 在构造
    error frame 时读取它们。初始化参数 ``detail`` 是开发者诊断文本；``data`` 是可选的
    structured context，会原样放入 JSON-RPC ``error.data``，因此调用方必须确保其中不
    含 secrets 或不应暴露的内部状态。

    实例只携带一次 RPC 失败的信息，不拥有错误恢复生命周期。``public_message`` 优先读取
    非空 ``data["public_message"]``，否则回退到稳定 ``MESSAGE``；它不保证适合日志之外
    的所有终端，也不表示失败动作没有产生部分副作用。
    """

    CODE: ClassVar[int] = -32099  # 兜底错误
    MESSAGE: ClassVar[str] = "rpc_error"

    def __init__(self, detail: str = "", data: dict[str, Any] | None = None):
        self.detail = detail
        self.data = data
        # 默认 str() 展示错误码、消息和可选详情
        super().__init__(f"{self.MESSAGE}: {detail}" if detail else self.MESSAGE)

    @property
    def code(self) -> int:
        return self.CODE

    @property
    def message(self) -> str:
        return self.MESSAGE

    @property
    def public_message(self) -> str:
        if isinstance(self.data, dict):
            value = self.data.get("public_message")
            if isinstance(value, str) and value.strip():
                return value
        return self.message


class SessionNotFoundError(RpcError):
    CODE = -32001
    MESSAGE = "session_not_found"


class SessionLockedError(RpcError):
    CODE = -32002
    MESSAGE = "session_locked"


class TurnInProgressError(RpcError):
    CODE = -32003
    MESSAGE = "turn_in_progress"


class ModelNotAvailableError(RpcError):
    CODE = -32008
    MESSAGE = "model_not_available"


class ModelSwitchInTurnError(RpcError):
    CODE = -32009
    MESSAGE = "model_switch_in_turn"


class ConfigFieldReadonlyError(RpcError):
    CODE = -32010
    MESSAGE = "config_field_readonly"


class ConfigValidationError(RpcError):
    CODE = -32011
    MESSAGE = "config_validation_error"


class NotSupportedInV01Error(RpcError):
    CODE = -32012
    MESSAGE = "not_supported_in_v01"


# 后续扩展区间（-32016..-32049）。-32016 表示订阅溢出；早期草案曾误用 -32010，
# 但 -32010 已属于 ConfigFieldReadonlyError。
class SubscriptionCapacityExceededError(RpcError):
    CODE = -32016
    MESSAGE = "subscription_capacity_exceeded"


# JSON-RPC 预定义的 ``internal_error``（-32603）。添加此类后，非分派器路径也能
# 跨模块抛出带类型的 -32603。例如 ``_build_tui_agent_loop`` 运行在处理器上下文之外，
# 却需通过 ``_spawn_agent_loop_task`` 使用的工厂闭包暴露初始化崩溃。
class InternalError(RpcError):
    CODE = -32603
    MESSAGE = "internal_error"


# JSON-RPC 预定义错误码（见 specs §2.3 / RFC）。
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


# 反向查找：JSON-RPC 错误码 → 异常类，用于客户端重建异常或测试断言。
JSONRPC_ERROR_REGISTRY: dict[int, type[RpcError]] = {
    cls.CODE: cls
    for cls in (
        SessionNotFoundError,
        SessionLockedError,
        TurnInProgressError,
        ModelNotAvailableError,
        ModelSwitchInTurnError,
        ConfigFieldReadonlyError,
        ConfigValidationError,
        NotSupportedInV01Error,
        SubscriptionCapacityExceededError,
        InternalError,
    )
}


__all__ = [
    "RpcError",
    "SessionNotFoundError",
    "SessionLockedError",
    "TurnInProgressError",
    "ModelNotAvailableError",
    "ModelSwitchInTurnError",
    "ConfigFieldReadonlyError",
    "ConfigValidationError",
    "NotSupportedInV01Error",
    "SubscriptionCapacityExceededError",
    "InternalError",
    "JSONRPC_ERROR_REGISTRY",
    "PARSE_ERROR",
    "INVALID_REQUEST",
    "METHOD_NOT_FOUND",
    "INVALID_PARAMS",
    "INTERNAL_ERROR",
]
