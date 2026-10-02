"""Channel Sender Allowlist，是 Deny-by-default Check 的 Canonical Home。

``codeflow.channels.base.ChannelBase.is_allowed``、各 Channel Override 与 ``Intake`` 都复用这里。Semantics
刻意保持 Conservative：

- Empty Allowlist：DENY Everything，避免 Misconfigured Channel 意外接收整个 Internet 的消息；
- 列表包含 ``"*"``：ALLOW Everything，这是明确的 Opt-in；
- 其他情况：把 Sender ID 转成字符串后做 Exact Match。

某个 Channel 因 Empty Allowlist 首次拒绝时只记录一条 Warning；每个 Inbound 都重复日志会掩盖真正
异常。这个模块只判断 Sender Identity 是否在名单内，不完成平台签名验证或更细粒度权限授权。
"""

from __future__ import annotations

import logging
from typing import Iterable

logger = logging.getLogger(__name__)


# 记录本进程中已经输出过“空白名单”警告的渠道名称。放在模块级，
# 以便同一渠道适配器的所有实例和多次调用共享。
_warned_empty: set[str] = set()


def is_allowed(
    channel_name: str,
    sender_id: str | int | None,
    allow_list: Iterable[str] | None,
) -> bool:
    """判断 ``sender_id`` 是否获准从 ``channel_name`` 进入。

    Args:
        channel_name: Human-readable Channel Identifier，只用于 Logging 与去重 Warning，不参与授权匹配。
        sender_id: Platform-specific Sender ID。比较前统一 Stringify，使 `int`、``open_id`` 与 Phone-number
            等不同表示都能稳定匹配。
        allow_list: 获准 Sender IDs 的 Iterable；包含 ``"*"`` 表示 Allow-all。`None` 或 Empty Iterable
            都表示 Deny。

    Returns:
        Sender 获准时返回 `True`，否则返回 `False`。返回 `True` 只说明通过这一层 Allowlist，不代表
        消息签名、Capability 或业务权限也已经验证。
    """
    if allow_list is None:
        _warn_empty_once(channel_name)
        return False

    # 只展开一次，既能检查成员关系也能判断是否为空，
    # 同时不强制调用方传入序列。
    allow_set = set(map(str, allow_list))
    if not allow_set:
        _warn_empty_once(channel_name)
        return False
    if "*" in allow_set:
        return True
    return str(sender_id) in allow_set


def _warn_empty_once(channel_name: str) -> None:
    if channel_name in _warned_empty:
        return
    _warned_empty.add(channel_name)
    logger.warning(
        "%s: allow_from is empty — all access denied",
        channel_name,
    )


def reset_warning_state() -> None:
    """Test-only Helper，清空 ``already warned`` Tracker。

    Fixture 可借此把某个 Channel 恢复成 Fresh 状态，并断言 Empty Allowlist Warning 确实触发。Production
    Code 不应周期性调用，否则会让本应去重的 Warning 再次刷屏；函数不修改任何 Allowlist 数据。
    """
    _warned_empty.clear()


__all__ = ["is_allowed", "reset_warning_state"]
