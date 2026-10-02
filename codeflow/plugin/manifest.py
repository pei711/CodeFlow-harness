"""Plugin Manifest Schema。

Manifest 是与 Plugin Python Package 一起发布的 TOML File ``codeflow-plugin.toml``。它声明 Plugin Identity、
Contribution Points 与 Config Schema，也就是 Registry 在 **不 Import Plugin Code** 的情况下需要知道的
全部数据。

唯一 Root Table 是 ``[plugin]``，Contribution Arrays 使用 ``[[plugin.contributes.<kind>]]``。PG-1 最初
只有 ``memory_backends``，当前也支持 Tools；Model 会 Silently 接受 Unknown Extras，使 Future Kinds
不会破坏 Older Hosts。

Important Validation Rules：``id`` 与 ``version`` Required 且 ``min_length=1``；``factory`` 必须符合
``module.path:callable_name``，让 Typo 在 Startup 而非 First Activation 才失败；Contribution Name 在
*单个 Manifest 内*必须唯一，Registry 另行约束 *跨 Manifests* 冲突。
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# 工厂引用格式为 ``module.path:callable``。正则有意保持宽松：
# 非空的模块式路径、一个冒号以及非空的近似标识符后缀即可。
_FACTORY_REF_RE = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")


class _ManifestBase(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class MemoryBackendContribution(_ManifestBase):
    """一条 ``[[plugin.contributes.memory_backends]]`` Entry。

    `name` 是 Registry Slot，`factory` 是符合 ``module.path:callable`` 的 Lazy Reference。Model 只验证
    引用格式，不 Import Factory，也不证明返回对象满足 Memory Backend Protocol。
    """

    name: str = Field(min_length=1)
    factory: str = Field(min_length=1)

    @field_validator("factory")
    @classmethod
    def _factory_is_module_path(cls, v: str) -> str:
        if not _FACTORY_REF_RE.match(v):
            raise ValueError(
                f"factory must be 'module.path:callable', got {v!r}",
            )
        return v


class ToolContribution(_ManifestBase):
    """一条 ``[[plugin.contributes.tools]]`` Entry。

    ``factory`` 是解析为 ``Callable[[PluginContext], Tool]`` 的 ``module.path:callable``，返回一个 Host
    在 Boot 时注册进 Agent Tool Set 的 :class:`~codeflow.agent.tools.base.Tool`。每条 Entry 对应 One Tool；
    Plugin 暴露多个 Tools 时必须列出多个 Entries。Manifest Parse 不执行该 Callable。
    """

    name: str = Field(min_length=1)
    factory: str = Field(min_length=1)

    @field_validator("factory")
    @classmethod
    def _factory_is_module_path(cls, v: str) -> str:
        if not _FACTORY_REF_RE.match(v):
            raise ValueError(
                f"factory must be 'module.path:callable', got {v!r}",
            )
        return v


class Contributes(_ManifestBase):
    """单个 Manifest 的全部 Contribution Arrays。

    当前消费 ``memory_backends`` 与 ``tools``；Model Silently 保留兼容策略、忽略 Extra Fields，使 Older
    Hosts 读取带 Future Contribution Types 的 Newer Manifests 时不会直接失败。忽略也意味着旧 Host 不会
    激活它不理解的新贡献。
    """

    memory_backends: list[MemoryBackendContribution] = Field(default_factory=list)
    tools: list[ToolContribution] = Field(default_factory=list)


class PluginManifest(_ManifestBase):
    """解析后的 ``codeflow-plugin.toml`` Pure-data Model。

    通常通过 :meth:`from_toml_path` / :meth:`from_toml_str` 构造，Programmatic Tests 也可直接使用 Raw
    ``__init__``。Frozen Pydantic Model 保存 Identity、CodeFlow Compatibility、Default Enablement、
    Contributions 与 Descriptive Config Schema；构造成功不触发 Plugin Import。
    """

    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    display_name: str | None = None
    codeflow: str | None = None
    bundled: bool = False
    enabled_by_default: bool = False
    contributes: Contributes = Field(default_factory=Contributes)
    config_schema: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _contribution_names_unique(self) -> "PluginManifest":
        # 唯一性只在同一 kind 内约束；后端和工具可以同名，因为占用不同槽位。
        # registry 另行约束跨 manifest 的唯一性。
        for kind, items in (
            ("memory_backend", self.contributes.memory_backends),
            ("tool", self.contributes.tools),
        ):
            names = [c.name for c in items]
            if len(names) != len(set(names)):
                dupes = sorted({n for n in names if names.count(n) > 1})
                raise ValueError(
                    f"duplicate {kind} name(s) in manifest {self.id!r}: {dupes}",
                )
        return self

    # ── 构造方法 ───────────────────────────────────────────────────

    @classmethod
    def from_toml_str(cls, data: str) -> "PluginManifest":
        """从 Raw TOML String 解析 `PluginManifest`。

        先由 `tomllib.loads` 解码，再要求 Top-level ``[plugin]`` Table 并执行 Pydantic Validation。Malformed
        TOML 或 Schema Error 原样向上传播；方法不访问 Filesystem 或 Import Factory。
        """
        raw = tomllib.loads(data)
        return cls._from_raw(raw)

    @classmethod
    def from_toml_path(cls, path: Path) -> "PluginManifest":
        """从 Disk File 解析 `PluginManifest`。

        Missing File 抛出 ``FileNotFoundError``，Malformed TOML 抛出 ``tomllib.TOMLDecodeError``，Schema
        Mismatch 抛出 ``pydantic.ValidationError``。文件以 Binary 打开交给 `tomllib`；成功只表示声明
        合法，不执行或验证 Factory Code。
        """
        with path.open("rb") as f:
            raw = tomllib.load(f)
        return cls._from_raw(raw)

    @classmethod
    def _from_raw(cls, raw: dict[str, Any]) -> "PluginManifest":
        # manifest 把所有内容嵌套在 [plugin] 下。交给 pydantic 前先解包，
        # 使 schema 使用相对于 plugin 的字段。
        if "plugin" not in raw:
            raise ValueError(
                "manifest missing top-level [plugin] table",
            )
        return cls.model_validate(raw["plugin"])


__all__ = [
    "Contributes",
    "MemoryBackendContribution",
    "PluginManifest",
    "ToolContribution",
]
