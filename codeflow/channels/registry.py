"""Channel Adapters 的 Auto-discovery，无 Hardcoded Registry。

Registry 通过 Package Scan 发现 Adapter Names，并只 Import Cheap ``spec.py``。Heavy SDK Import 延迟到
Factory Construction，使 CLI Listing/Onboarding 在缺可选依赖时仍可运行。
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeflow.channels.contract import ChannelSpec

_ADAPTERS_PKG = "codeflow.channels.adapters"


def discover_specs() -> dict[str, ChannelSpec]:
    """返回按 Package Name Keyed 的 ``{name: ChannelSpec}``。

    只 Import 每个 ``<name>/spec.py``；Heavy SDK 延迟到 ``factory``。没有 ``spec.py`` 的未迁移 Adapter Skip。
    Spec Import 内其他 ModuleNotFoundError 当前也会 Skip，Factory/Capability 验证在后续完成。
    """
    import codeflow.channels.adapters as pkg

    specs: dict[str, ChannelSpec] = {}
    for _, name, ispkg in pkgutil.iter_modules(pkg.__path__):
        if not ispkg:
            continue
        try:
            mod = importlib.import_module(f"{_ADAPTERS_PKG}.{name}.spec")
        except ModuleNotFoundError:
            continue  # 尚未迁移
        if (spec := getattr(mod, "SPEC", None)) is not None:
            specs[name] = spec
    return specs


def discover_channel_names() -> list[str]:
    """Zero Imports 扫描 Adapters Package，返回 Sorted Adapter Names。

    枚举 Flat Modules 与 Adapter Subpackages；Scan One Level Deep，因此 Nested Helper Modules 不会被误认为
    Channel。Name Discovery 不保证存在 Migrated Spec。
    """
    import codeflow.channels.adapters as pkg

    return sorted(name for _, name, _ in pkgutil.iter_modules(pkg.__path__))
