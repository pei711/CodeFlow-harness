"""Plugin Runtime Context 与 Host Capability Grant。

Manifest 中 ``module.path:callable`` 指向的 Factory 只接收一个 :class:`PluginContext`，并从中取得：

- ``config``：来自 `CodeFlowConfig` 的 Plugin-own Config Slice。Registry 原样传递 Dict，每个 Factory 自行
  验证所消费 Fields；Manifest ``config_schema`` 目前只是 Descriptive Metadata；
- ``services``：:class:`ServiceLocator` 只暴露 Backend 获准触碰的 Host Services。Locator 刻意 Narrow，
  防止 Plugin 对任意 Host Internals 形成 Ambient Dependencies；每个 Field 都是 Deliberate Capability
  Grant；
- ``logger``：预绑定 ``plugin=<id>`` 的 Logger，使 Plugin Output 能在 Mixed Logs 中 Grep。

Locator 是 Frozen Dataclass：Factory 只能 Read，不能 Mutate Host 对 Available Services 的 View。这是
对象接口边界，不是恶意 Python Code 的进程级 Sandbox；一旦 Factory 被 Import，它仍在 Host Process。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ServiceLocator:
    """向 Plugin Factory 提供 Host Services 的 Narrow Grant。

    只有 Host 明确需要暴露的 Field 才进入这里。PG-1 从 Bare Minimum 建立 Seam；未来只有 Concrete
    Backend 证明需要时，才可添加 ``provider``、``bus`` 等。Dataclass 刻意 Frozen，因为每个 Field 都是
    Capability，新增必须成为本文件的 Explicit Edit，不能通过 Ambient ``setattr`` 悄悄扩权。
    """

    workspace: Path
    """Plugin Operates On 的 Workspace Root Path；是否可写仍取决于 Host Filesystem 权限。"""


@dataclass(frozen=True)
class PluginContext:
    """Plugin Factory 在 Activation Time 可见的完整 Context。

    它把 Plugin-specific Config、受限 `ServiceLocator` 与 Logger 组合成单一稳定参数。Frozen 只防止
    Factory 重新绑定这些字段，不会深度冻结 Config Dict 或阻止 Python 模块访问全局资源；Registry 的
    Lazy Import 仍是首次执行插件代码的关键 Trust Boundary。
    """

    config: dict[str, Any]
    services: ServiceLocator
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger("codeflow.plugin"),
    )


__all__ = ["PluginContext", "ServiceLocator"]
