"""Multi-source Plugin Discovery。

一次 Discovery Pass 扫描 Host 已知的所有 Sources：Bundled Sub-tree、User-level Dir、Project-level Dir
与 Pip Entry Points；随后按 Plugin ID Deduplicate，返回稳定的 :class:`DiscoveredPlugin` Records。
Discovery **只读 Manifests**，这里不 Import Plugin Python Code。:class:`Source` Enum 同时承担 Conflict
Resolution Priority，数值越高越优先。

Priority Order 是 ``bundled > user > project > entry_points``，遵守设计中的 Builtin Shadow Rule：同名
Local/Pip Plugin 永远不能 Shadow Bundled Plugin。Non-bundled Sources 中 User-level 胜出，使 Developer
迭代时能用 Locally Edited Copy 替代 Pip-installed Version。

Manifest Admission 仍会验证 Distribution Identity 与 CodeFlow Version Compatibility；Stable Discovery List
不代表 Factory 已 Import 或 Plugin 行为可信。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from importlib import metadata
from pathlib import Path, PurePosixPath

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from codeflow import __version__ as _installed_codeflow_version
from codeflow.plugin.manifest import PluginManifest

logger = logging.getLogger(__name__)


_MANIFEST_FILENAME = "codeflow-plugin.toml"


class PluginCompatibilityError(RuntimeError):
    """Plugin Manifest 排除了 Installed CodeFlow Version 时抛出。

    这表示声明的 Compatibility Specifier 与当前 Runtime 不相交，Plugin 不应进入 Registry。修复方式是
    安装兼容版本或禁用对应 Backend，不能忽略后继续 Lazy Import。
    """


class PluginIdentityError(RuntimeError):
    """Installed Distribution Metadata 与其 Plugin Manifest 不一致时抛出。

    Entry Point 的 Owning Distribution Name、Version 或 File Inventory 与 Manifest 冲突可能表示错误打包、
    陈旧安装或身份混淆。Discovery Fail Closed，并要求重新安装 Official Distribution。
    """


class Source(IntEnum):
    """Manifest 来源；Numeric Value 同时表示 Conflict Priority，Higher Wins。

    Lower-priority Duplicate 不进入最终列表，但会记录日志而非完全 Silent，便于诊断 Shadowing。枚举顺序
    是安全策略的一部分，修改数值会改变同名 Plugin 的实际选择。
    """

    ENTRY_POINTS = 1
    PROJECT = 2
    USER = 3
    BUNDLED = 4


@dataclass(frozen=True)
class DiscoveredPlugin:
    """从 Specific Source 读取、Awaiting Activation 的 Manifest Record。

    它同时保留 Parsed `PluginManifest`、`Source` 与可选 Location。对象只代表发现事实，Factory 尚未
    Import，Contributions 也尚未进入 Active Registry。
    """

    manifest: PluginManifest
    source: Source
    location: Path | None
    """Manifest File Path。Entry-points-discovered Plugin 的 Manifest 位于 Wheel Package Data 中，
    因此这里为 `None`；Discovery 仍通过 Distribution File Inventory 定位并读取真实文件。"""


class PluginDiscovery:
    """扫描每个 Configured Source，并返回 Deduplicated Plugins。

    Constructor Params 默认 ``off``；Caller 传入 Concrete Paths 或 Entry-point Group Name，逐个 Opt In Source。
    这使 Tests 保持 Hermetic：测试只指向 Temp Dirs，不会意外捡到 Developer Machine 上的 Real Plugins。
    实例还持有待验证的 CodeFlow Version，所有 Scan 共享同一兼容标准。
    """

    def __init__(
        self,
        *,
        bundled_dir: Path | None = None,
        user_dir: Path | None = None,
        project_dir: Path | None = None,
        entry_points_group: str | None = None,
        codeflow_version: str = _installed_codeflow_version,
    ) -> None:
        self._bundled_dir = bundled_dir
        self._user_dir = user_dir
        self._project_dir = project_dir
        self._entry_points_group = entry_points_group
        self._codeflow_version = codeflow_version

    def discover(self) -> list[DiscoveredPlugin]:
        """运行所有 Enabled Sources，并 Resolve Conflicts。

        各 Source 只在构造时显式启用后扫描，结果合并再按 Priority Dedup。返回 List 按 Plugin ID Stable
        Order，使 Caller 能 Deterministically Log / Display。Compatibility 或 Identity Error 会 Fail
        Discovery；普通单 Manifest Parse Error 记录 Warning 后跳过。
        """
        all_found: list[DiscoveredPlugin] = []
        if self._bundled_dir is not None:
            all_found.extend(
                self._scan_dir(self._bundled_dir, Source.BUNDLED),
            )
        if self._user_dir is not None:
            all_found.extend(self._scan_dir(self._user_dir, Source.USER))
        if self._project_dir is not None:
            all_found.extend(
                self._scan_dir(self._project_dir, Source.PROJECT),
            )
        if self._entry_points_group is not None:
            all_found.extend(self._scan_entry_points(self._entry_points_group))

        return self._resolve_conflicts(all_found)

    # ── 基于文件的来源 ───────────────────────────────────────────

    def _scan_dir(
        self,
        root: Path,
        source: Source,
    ) -> list[DiscoveredPlugin]:
        """在 ``<root>/<plugin_id>/codeflow-plugin.toml`` 查找 File-based Plugins。

        Subdir Name 只是 Informational，Canonical Plugin ID 来自 Manifest。两者 Mismatch 会记录日志但仍
        加载；解析失败则 Warning + Skip。每个成功 Manifest 都立即验证 CodeFlow Compatibility，返回的
        `location` 指向真实 TOML 文件。扫描不会递归到更深任意路径。
        """
        out: list[DiscoveredPlugin] = []
        if not root.is_dir():
            return out
        for sub in sorted(root.iterdir()):
            if not sub.is_dir():
                continue
            manifest_path = sub / _MANIFEST_FILENAME
            if not manifest_path.is_file():
                continue
            try:
                mf = PluginManifest.from_toml_path(manifest_path)
            except Exception as e:
                logger.warning(
                    "plugin manifest %s failed to parse (%s); skipping",
                    manifest_path,
                    e,
                )
                continue
            self._validate_compatibility(mf)
            if mf.id != sub.name:
                logger.info(
                    "plugin %s lives in directory %s — id and dir name differ",
                    mf.id,
                    sub.name,
                )
            out.append(
                DiscoveredPlugin(
                    manifest=mf,
                    source=source,
                    location=manifest_path,
                ),
            )
        return out

    # ── 入口点来源 ───────────────────────────────────────────────

    def _scan_entry_points(self, group: str) -> list[DiscoveredPlugin]:
        """解析 ``group`` 中所有 Entry Points，并读取其 Package 内 Manifest。

        Entry-point Value 指出拥有 ``codeflow-plugin.toml`` 的 Package。Discovery 通过 Owning Distribution 的
        Installed File Inventory 定位 Manifest，**不 Import Plugin Package**。每项还验证 Distribution
        Name/Version 与 Manifest Identity、以及 CodeFlow Compatibility；Identity/Compatibility Error 向上
        抛出，普通定位或 Parse Failure 则 Warning + Skip。
        """
        out: list[DiscoveredPlugin] = []
        try:
            eps = metadata.entry_points(group=group)
        except Exception as e:
            logger.warning("entry_points discovery failed (%s); skipping", e)
            return out

        for ep in eps:
            package_name = ep.value.split(":", 1)[0]
            try:
                manifest_path = self._entry_point_manifest_path(ep, package_name)
                if manifest_path is None:
                    logger.warning(
                        "entry-point %s points at package %s but no %s found; skipping",
                        ep.name,
                        package_name,
                        _MANIFEST_FILENAME,
                    )
                    continue
                mf = PluginManifest.from_toml_path(manifest_path)
            except Exception as e:
                if isinstance(e, (PluginCompatibilityError, PluginIdentityError)):
                    raise
                logger.warning(
                    "failed to load manifest for entry-point %s (%s); skipping",
                    ep.name,
                    e,
                )
                continue
            self._validate_entry_point_identity(ep, mf)
            self._validate_compatibility(mf)
            out.append(
                DiscoveredPlugin(
                    manifest=mf,
                    source=Source.ENTRY_POINTS,
                    location=None,
                ),
            )
        return out

    @staticmethod
    def _entry_point_manifest_path(ep, package_name: str) -> Path | None:
        distribution = getattr(ep, "dist", None)
        distribution_files = getattr(distribution, "files", None)
        if distribution is None or distribution_files is None:
            raise PluginIdentityError(
                f"entry-point {ep.name!r} has no owning distribution file inventory; reinstall the plugin"
            )
        expected = PurePosixPath(*package_name.split("."), _MANIFEST_FILENAME)
        for item in distribution_files:
            if PurePosixPath(str(item)) == expected:
                path = Path(distribution.locate_file(item))
                return path if path.is_file() else None
        return None

    @staticmethod
    def _validate_entry_point_identity(ep, manifest: PluginManifest) -> None:
        distribution = getattr(ep, "dist", None)
        if distribution is None:
            raise PluginIdentityError(
                f"entry-point {ep.name!r} has no owning distribution metadata; reinstall the plugin"
            )
        distribution_name = getattr(distribution, "name", None)
        distribution_version = getattr(distribution, "version", None)
        if not distribution_name or canonicalize_name(distribution_name) != canonicalize_name(manifest.id):
            raise PluginIdentityError(
                f"entry-point {ep.name!r} distribution {distribution_name!r} does not match "
                f"manifest id {manifest.id!r}; reinstall the plugin from its official distribution"
            )
        try:
            version_matches = Version(str(distribution_version)) == Version(manifest.version)
        except InvalidVersion as exc:
            raise PluginIdentityError(
                f"entry-point {ep.name!r} has invalid distribution or manifest version metadata"
            ) from exc
        if not version_matches:
            raise PluginIdentityError(
                f"entry-point {ep.name!r} distribution version {distribution_version} does not match "
                f"manifest version {manifest.version}; reinstall the plugin"
            )

    def _validate_compatibility(self, manifest: PluginManifest) -> None:
        if not manifest.codeflow:
            return
        try:
            compatible = Version(self._codeflow_version) in SpecifierSet(manifest.codeflow)
        except (InvalidSpecifier, InvalidVersion) as exc:
            raise PluginCompatibilityError(
                f"plugin {manifest.id!r} has invalid CodeFlow compatibility metadata; reinstall a compatible plugin"
            ) from exc
        if not compatible:
            raise PluginCompatibilityError(
                f"plugin {manifest.id!r} {manifest.version} requires CodeFlow {manifest.codeflow}, but installed CodeFlow is "
                f"{self._codeflow_version}; install a compatible plugin version or set memory.backend to null"
            )

    # ── 冲突解决 ─────────────────────────────────────────────────

    @staticmethod
    def _resolve_conflicts(
        found: list[DiscoveredPlugin],
    ) -> list[DiscoveredPlugin]:
        """按 Plugin ID Group，并保留 Highest-priority Source。

        Higher Source 替换 Current 时记录 Shadows 日志，Lower Duplicate 也各记录一次 Shadowed By，使
        Misconfigured Setup 可诊断而不是 Silently Dropping Plugins。同一 Priority 的重复项保持第一个；
        最终按 Manifest ID 排序，确保展示稳定。
        """
        by_id: dict[str, DiscoveredPlugin] = {}
        for d in found:
            current = by_id.get(d.manifest.id)
            if current is None or d.source > current.source:
                if current is not None:
                    logger.info(
                        "plugin %s: %s shadows %s",
                        d.manifest.id,
                        d.source.name,
                        current.source.name,
                    )
                by_id[d.manifest.id] = d
            elif d.source < current.source:
                logger.info(
                    "plugin %s: %s shadowed by %s",
                    d.manifest.id,
                    d.source.name,
                    current.source.name,
                )
        # 按 id 稳定排序，确保调用方展示顺序确定。
        return sorted(by_id.values(), key=lambda p: p.manifest.id)


__all__ = [
    "DiscoveredPlugin",
    "PluginCompatibilityError",
    "PluginDiscovery",
    "PluginIdentityError",
    "Source",
]
