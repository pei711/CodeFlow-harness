"""Evolver candidate patch 的 commit-side application machinery。

输入是 judge 产生的 candidate ``AppliedPatch``；本 Package 判断它能否安全作用于 working tree，
并为后续写入提供 gate。:mod:`path_guard` 是 first-line immutability check，按 spec §22 与
§22.7 拒绝 kernel path；:mod:`beacon_guard` 验证 code-class patch，在 eval 前拒绝缺少
activation beacon 的不可观测改动（design section 3）。

Future C2 将加入 Git-backed atomic apply，在创建任何 commit 前内部调用 ``path_guard``
short-circuit。当前 guard 通过只表示路径/可观测性合规，不表示 patch 可编译、task 完成、
benchmark 改善或可以 promote。
"""

from .beacon_guard import (
    MissingBeaconError,
    assert_beacon_present,
)
from .path_guard import (
    IMMUTABLE_PATTERNS,
    MUTABLE_OVERRIDES,
    ImmutablePathError,
    UnsafePathError,
    assert_patch_allowed,
    check_patch_paths,
    is_immutable,
)

__all__ = [
    "IMMUTABLE_PATTERNS",
    "MUTABLE_OVERRIDES",
    "ImmutablePathError",
    "UnsafePathError",
    "assert_patch_allowed",
    "check_patch_paths",
    "is_immutable",
    "MissingBeaconError",
    "assert_beacon_present",
]
