"""实现 handshake、ping 与 version 查询的 ``system.*`` RPC handlers。

Dispatcher 使用 plain ``params: dict`` 调用这些 handler，handler 必须返回 plain
``result: dict``。输入 validation 在可用时使用 ``codeflow/tui_rpc/models.py`` 的 Pydantic v2
model；handshake 仍内联 lightweight semver guard，使 Dispatcher 可以 standalone 测试。

``system.hello`` 建立协议能力认知，``system.ping`` 提供 RTT 时间点，``system.version``
报告实现/schema/package 三组版本。它们不创建 Session、不启动 Agent，也不证明其他
Runtime capability 已准备就绪。
"""

from __future__ import annotations

import importlib.metadata as _md
import os
import re
import time
from typing import TYPE_CHECKING

from loguru import logger

from codeflow.tui_rpc.errors import ConfigValidationError

if TYPE_CHECKING:
    from codeflow.tui_rpc.dispatcher import Dispatcher


# ----------------------------------------------------------------------------
# 版本定义
# ----------------------------------------------------------------------------
# server_version：IPC 桥接协议的实现版本，发布新的线上协议兼容版本时递增。
# schema_version：与 `ui-tui/rpc-schema/openrpc.json` 中的 OpenRPC `info.version` 一致。
# codeflow_version：从已安装元数据读取的 CodeFlow 发行版本。
SERVER_VERSION = "0.1.0"
SCHEMA_VERSION = "0.1.0"
SERVER_CAPABILITIES = ["jsonrpc-2.0", "subscriptions", "attachments", "sessions", "confirm"]

# 宽松的语义化版本：<major>.<minor>.<patch>，可选 `-prerelease` 或 `+build`。
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")


def _codeflow_version() -> str:
    try:
        return _md.version("codeflow-harness")
    except _md.PackageNotFoundError:
        # CI 中的可编辑安装可能未注册元数据；回退到约定的哨兵值，避免握手崩溃。
        return "0.0.0+unknown"


# ----------------------------------------------------------------------------
# 处理器
# ----------------------------------------------------------------------------


async def system_hello(params: dict) -> dict:
    """执行 ``system.hello`` initial handshake，并校验 ``client_version`` semver。

    Spec §3.7 ``system.hello`` 规定：client version 缺失或不符合
    ``<major>.<minor>.<patch>``（可带 prerelease/build）时抛出 ``-32011``
    ``ConfigValidationError``。成功时记录 pid、client version/capabilities，并返回
    ``server_version``、``server_capabilities`` 与默认 TUI Session 描述。

    handshake 成功只表示基础协议形状兼容；这里不会逐项协商 capability，也不保证 provider
    或 AgentLoop 已构建。
    """
    client_version = params.get("client_version")
    if not isinstance(client_version, str) or not client_version:
        raise ConfigValidationError(
            "client_version is required",
            data={"field": "client_version", "reason": "missing"},
        )
    if not _SEMVER_RE.match(client_version):
        raise ConfigValidationError(
            f"client_version '{client_version}' is not a valid semver",
            data={"field": "client_version", "value": client_version},
        )

    client_capabilities = params.get("client_capabilities", []) or []
    # pid 用于区分共享同一日志文件的并发 `codeflow` 进程。
    logger.info(
        "tui_rpc: handshake — pid={} client_version={} client_capabilities={}",
        os.getpid(),
        client_version,
        client_capabilities,
    )
    return {
        "server_version": SERVER_VERSION,
        "server_capabilities": list(SERVER_CAPABILITIES),
        "session": {
            "default_channel": "tui",
            "default_session_key": "tui:default",
        },
    }


async def system_ping(params: dict) -> dict:
    """执行 ``system.ping`` RTT probe，返回 server Unix timestamp（ms）。

    返回 ``{"pong": True, "server_time_ms": ...}``，``params`` 当前被忽略。client 可用本地
    发收时间估算 round-trip time，但该时间戳不是单调时钟，也不表示 Runtime 空闲或健康。
    """
    return {
        "pong": True,
        "server_time_ms": int(time.time() * 1000),
    }


async def system_version(params: dict) -> dict:
    """执行 ``system.version``，返回 diagnostics/compatibility 使用的 version triple。

    ``server_version`` 是 IPC bridge 实现版本，``schema_version`` 对齐 OpenRPC
    ``info.version``，``codeflow_version`` 来自 installed package metadata；editable/源码环境
    缺少 metadata 时后者为 ``0.0.0+unknown``。返回值不执行 upgrade 或兼容性决策。
    """
    return {
        "server_version": SERVER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "codeflow_version": _codeflow_version(),
    }


def register_system_methods(dispatcher: "Dispatcher") -> None:
    """在 Dispatcher 上注册三个 ``system.*`` method。

    注册 ``system.hello``、``system.ping`` 与 ``system.version``；函数不执行 handshake，
    重复注册由 Dispatcher 抛出 ``ValueError``。
    """
    dispatcher.register("system.hello", system_hello)
    dispatcher.register("system.ping", system_ping)
    dispatcher.register("system.version", system_version)


__all__ = [
    "system_hello",
    "system_ping",
    "system_version",
    "register_system_methods",
    "SERVER_VERSION",
    "SCHEMA_VERSION",
    "SERVER_CAPABILITIES",
    "_codeflow_version",
]
