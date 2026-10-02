"""`LocalSkillCatalog`，Local Skill Pool 的 Single Owner。

它吸收旧 ``SkillService`` 的文件侧职责：构造 :class:`SkillRegistry` + :class:`LocalPool`，运行
``SKILL.md`` File Watcher，并为 Prompt 渲染 Always-skills、Selected Injection 与 XML Summary。

Retrieval **不在这里**；:class:`LocalSkillSource` 复用 Catalog 的 ``pool`` + ``registry`` 执行 Search，再
由 :class:`SkillForgeRouter` 与其他 Sources 融合。旧 ``SkillService.select`` / LLM-gate / Query-rewriter
Path 已在 Router 替代后 Retired。Catalog 读到 Skill 不等于 Router 选中或 Prompt 已注入。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeflow.memory_engine.skill_forge.refs import resolve_refs
from codeflow.memory_engine.skill_local.local_pool import LocalPool
from codeflow.memory_engine.skill_local.registry import SkillRegistry
from codeflow.memory_engine.skill_local.types import SkillMeta

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from codeflow.memory_engine.skill_local.watcher import SkillFileWatcher
    from codeflow.providers.base import LLMProvider


class LocalSkillCatalog:
    """负责 Local Skill Pool 的 File Access、Watcher 与 Directory Rendering。

    初始化时根据 Workspace、Builtin Dir 与 Configured `local_dirs` 建立分层 Registry 和 BM25 Pool，并可
    启动 Background Watcher。Catalog 拥有这些对象的生命周期与缓存失效；Retriever 只借用它们 Search。
    短生命周期 Caller 应禁用 Watcher，避免 Daemon Thread/Handle 泄漏。
    """

    def __init__(
        self,
        workspace: Path,
        config: Any = None,
        builtin_skills_dir: Path | None = None,
        llm_provider: "LLMProvider | None" = None,  # 为兼容调用方而保留，当前未使用
        *,
        start_watcher: bool = True,
    ):
        # R1：从 config.local_dirs 构建 extra_dirs。列表顺序即优先级，名称冲突时后者覆盖前者。
        # 每个元组格式为 (path, display_name, always_enabled)。
        extra_dirs: list[tuple[Path, str, bool]] = []
        if config is not None and getattr(config, "enabled", False):
            seen_names: dict[str, int] = {}
            for entry in getattr(config, "local_dirs", []) or []:
                entry_path = Path(os.path.expanduser(getattr(entry, "path", "") or "")).resolve()
                entry_enabled = getattr(entry, "enabled", True)
                if not entry_enabled or not entry_path or not entry_path.is_dir():
                    if entry_enabled and getattr(entry, "path", ""):
                        log.warning(
                            "local_dirs: path does not exist or is not a directory: %s",
                            entry_path,
                        )
                    continue
                entry_name = getattr(entry, "name", None) or entry_path.name
                seen_names[entry_name] = seen_names.get(entry_name, 0) + 1
                if seen_names[entry_name] > 1:
                    entry_name = f"{entry_name}_{seen_names[entry_name]}"
                entry_always_enabled = getattr(entry, "always_enabled", True)
                extra_dirs.append((entry_path, entry_name, entry_always_enabled))

        self._registry = SkillRegistry(
            workspace,
            builtin_skills_dir=builtin_skills_dir,
            extra_dirs=extra_dirs,
            scan_max_depth=int(getattr(config, "scan_max_depth", 5) if config else 5),
        )

        self._config = config

        # 对基于文件的 Skill（工作区和内置）执行本地池 BM25 检索，不需模型或 GPU，始终可用。
        self._local_pool = LocalPool(self._registry)

        # 后台 SKILL.md 监视器默认自动启动，使长生命周期消费方 ContextBuilder 能自动获取
        # 对 ``<workspace>/skills/**/SKILL.md`` 的手动编辑。监视器运行在守护线程中，进程退出时自动清理；
        # 缺少 ``watchfiles`` 时退化为空操作并记录一条 INFO 日志。
        #
        # 短生命周期消费方（单个 CLI 命令、为子 Agent 执行一次的 ``build_skills_summary()``）
        # 应传入 ``start_watcher=False``。启动约耗时 25 毫秒，且监视器的守护线程会通过
        # ``on_change`` 绑定方法持有反向强引用，使目录在单次使用后仍存活。对会启动
        # 许多子 Agent 的长生命周期父进程而言，这是真实的线程和句柄泄漏。
        self._file_watcher: "SkillFileWatcher | None" = None
        if start_watcher:
            self.start_file_watcher()

    @property
    def registry(self) -> SkillRegistry:
        return self._registry

    @property
    def pool(self) -> LocalPool:
        return self._local_pool

    def invalidate_skill_cache(self, source: str | None = None) -> None:
        """Invalidate File-registry Cache，并 Refresh Local BM25 Index。

        提供 ``source`` 时只重建该 Source Slice，再 Merge 回 Registry，避免 Re-walk 未变化的 Builtin /
        Workspace / External Trees。传 `None` 执行 Hard Reset，只适合 Off-band 同时改写多个 Sources。

        Registry 更新后 Eagerly Rebuild BM25，使 File-watcher Event 直接流入 Retrieval，``search`` Caller
        不在 Hot Path 支付 Index Build。方法完成表示索引反映当前扫描结果，不保证 Skill Body 有效。
        """
        if source is None:
            self._registry.invalidate_cache()
        else:
            self._registry.invalidate_source(source)
        self._local_pool.rebuild_index()

    def start_file_watcher(self) -> bool:
        """启动 Background ``SKILL.md`` Filesystem Watcher。

        :meth:`__init__` 默认自动调用；Public Method 主要供 Tests 或 `stop_file_watcher` 后 Rare Restart。
        第一次成功会建立 ``watchfiles``-backed Daemon Thread，Workspace Skills 下 Add/Edit/Remove 时把
        Per-source Invalidation 送入 :meth:`invalidate_skill_cache`；监控根是 ``<workspace>/skills``。后续
        调用 Idempotent No-op。

        只有新 Watcher Running 时返回 `True`。``watchfiles`` Missing、Workspace Skills Dir 不存在或已有
        Watcher 时返回 `False`，Catalog 仍可 Manual-invalidation。Builtin/External Layers 刻意 **不 Watch**：
        它们是 Read-only Mirrors，Builtin 可有约 80K Files，Recursive Watch 会超过 Linux Default Inotify
        Limit。方法 Never Raises。
        """
        if self._file_watcher is not None:
            return False
        from codeflow.memory_engine.skill_local.watcher import SkillFileWatcher

        watcher = SkillFileWatcher(
            roots=[self._registry.workspace_skills],
            on_change=self.invalidate_skill_cache,
            resolve_source=self._registry.resolve_source_for_path,
        )
        if not watcher.start():
            # 启动失败（缺少依赖或根目录）时不设置 ``_file_watcher``，使后续调用能在
            # 前置条件修复后重试，例如工作区已实体化。
            return False
        self._file_watcher = watcher
        return True

    def stop_file_watcher(self) -> None:
        """通知 Watcher Thread Exit，并 Best-effort Join。

        从未启动时安全 No-op；有实例时调用其 `stop` 并清空引用，使后续 Start 可重试。返回不携带线程
        退出证明，具体 Join 行为由 `SkillFileWatcher` 实现。
        """
        watcher = self._file_watcher
        if watcher is None:
            return
        watcher.stop()
        self._file_watcher = None

    # ------------------------------------------------------------------
    # 旧版 ``SkillsLoader`` API（签名兼容的直接替代）
    # ------------------------------------------------------------------

    def list_skills(self, filter_unavailable: bool = True) -> list[dict[str, str]]:
        """以 Legacy-shape Dicts ``{name, path, source}`` 返回所有 Skills。

        默认过滤 Requirement 不满足的 Unavailable Skills；关闭过滤时反映完整 Registry。返回仅含目录字段，
        不读取完整 Body，也不按当前 Query 排名。
        """
        metas = self._registry.list_all()
        if filter_unavailable:
            metas = [m for m in metas if self._registry.check_available(m.name, source=m.source)]
        return [{"name": m.name, "path": str(m.path), "source": m.source} for m in metas]

    def load_skill(self, name: str) -> str | None:
        """按 Name 返回完整 ``SKILL.md`` Content；Absent 时返回 `None`。

        查找遵循 Registry 的 Source Priority。内容已由 Registry 加载，不表示 Requirement 可用或 Skill 被
        Router 选中。
        """
        return self._registry.get_body(name)

    def get_always_skills(self) -> list[SkillMeta]:
        """返回标记 ``always: true`` 且 Requirements Met 的 Skills。

        R3 Truncation Order 按 `local_dirs` List Order（映射 Registry Source Priority）+ Source 内 Alphabetical。
        `disable_always` 返回空；超过 `always_max` 时保留前项并以 WARN/Warning 列出 Dropped Names。返回 Skills 会
        默认注入，但仍受后续 Context Budget 影响。
        """
        if getattr(self._config, "disable_always", False):
            return []
        # 注册表 list_all 已按分层迭代顺序返回 Skill：工作区 → 按顺序的 extra_dirs → 内置；
        # 每层内按发现顺序。再按（来源优先级，名称）稳定排序，使截断结果可预测。
        all_always = [
            m for m in self._registry.list_all() if m.always and self._registry.check_available(m.name, source=m.source)
        ]
        all_always.sort(key=lambda m: (m.source, m.name))
        cap = int(getattr(self._config, "always_max", 5) or 5)
        if len(all_always) > cap:
            kept = all_always[:cap]
            dropped = all_always[cap:]
            log.warning(
                "always skills (%d) exceed always_max (%d), dropped: %s",
                len(all_always),
                cap,
                ", ".join(m.name for m in dropped),
            )
            return kept
        return all_always

    def load_skills_for_context(
        self,
        skills: "list[SkillMeta] | list[str]",
        max_inject: int | None = None,
    ) -> str:
        """把给定 Skill Bodies 渲染成一个可注入 Blob，Frontmatter 已移除。

        Body 直接来自 Registry-loaded ``meta.content``，其来源是 ``SKILL.md``。``max_inject`` 限制 Inline Skill 数量，每个通常
        1-5K Tokens；`None` 回退到 ``config.inject_max``，0/None 表示不设数量 Cap。为 Backward
        Compatibility 也接受 Name List，并通过 Registry Resolve。

        Filesystem-backed Skill 的 ``{baseDir}`` 替换为 ``meta.path.parent``（OpenSpace Convention），使
        ``{baseDir}/scripts/foo.py`` 等 Relative Ref 可运行；DB-only/无 Physical Path Skill 保持 Placeholder
        或按无 Base Resolver 处理。返回 Blob 表示内容已渲染，不表示已放入本轮 Prompt。
        """
        if max_inject is None:
            max_inject = getattr(self._config, "inject_max", 0) or 0
        # 向后兼容：接受名称列表，并通过注册表解析。
        if skills and isinstance(skills[0], str):
            resolved: list[SkillMeta] = []
            for name in skills:
                meta = self._lookup_meta(name, None)
                if meta is not None:
                    resolved.append(meta)
            skills = resolved
        parts: list[str] = []
        for m in skills:
            if not m.content:
                continue
            body = m.content
            path_obj = getattr(m, "path", None)
            path_str = str(path_obj) if path_obj is not None else ""
            has_real_path = path_obj is not None and not path_str.startswith("sqlite:")
            header = f"### Skill: {m.name}\n\n"
            if has_real_path and path_obj.parent.exists():
                base_dir = str(path_obj.parent)
                base_dir_refs = body.count("{baseDir}")
                body, _ = resolve_refs(body, path_obj.parent)
                resolved_base_dir = body.count("{baseDir}") < base_dir_refs
                if base_dir_refs == 0 or resolved_base_dir:
                    header = (
                        f"### Skill: {m.name}\n"
                        f"**Skill directory**: `{base_dir}`\n"
                        "Relative refs (e.g. `references/x.md`, `./scripts/y.sh`) "
                        "resolve under this directory — use the absolute form for "
                        "read_file / exec.\n\n"
                    )
            elif not has_real_path:
                body, _ = resolve_refs(body, None)
            parts.append(f"{header}{body}")
            if max_inject and len(parts) >= max_inject:
                break
        return "\n\n---\n\n".join(parts) if parts else ""

    def get_skill_metadata(self, name: str) -> dict | None:
        """返回 Top-level Frontmatter Dict；Skill Absent 时返回 `None`。

        Metadata 是 Registry Parse Result，Caller 修改返回对象是否影响缓存取决于 Registry 实现，应按只读
        使用。该接口不返回 Body。
        """
        return self._registry.get_raw_metadata(name)

    def build_skills_summary(self, only: "list[SkillMeta] | list[str] | None" = None) -> str:
        """构建 XML-formatted Skill Directory。

        Args:
            only: 提供时只包含这些 Skills。Canonical 输入是 ``list[SkillMeta]``；Older Callers 可传
                ``list[str]`` Names。`None` 保留 Legacy Full Local Directory，供 Selector 未命中时 Fallback。

        Local Skills 通过 On-disk ``meta.path`` 呈现，LLM 后续用 ``read_file`` 读取 Full Body；``available``
        来自 Registry ``requires`` Check，Unavailable 项附 Missing Requirements。XML 会 Escape Name/Desc，
        但 Location 原样输出。Summary 是可发现目录，不等于 Skill 已注入。
        """
        if only is None:
            metas: list[SkillMeta] = self._registry.list_all()
        else:
            # 向后兼容：接受 Skill 名称列表，通过注册表解析为 SkillMeta，并丢弃未知项。
            if only and isinstance(only[0], str):
                resolved: list[SkillMeta] = []
                for name in only:
                    meta = self._lookup_meta(name, None)
                    if meta is not None:
                        resolved.append(meta)
                only = resolved
            metas = list(only)
        if not metas:
            return ""

        lines: list[str] = ["<skills>"]
        for m in metas:
            name_x = _escape_xml(m.name)
            desc_x = _escape_xml(m.description or m.name)
            available = self._registry.check_available(m.name, source=m.source)
            lines.append(f'  <skill available="{str(available).lower()}">')
            lines.append(f"    <name>{name_x}</name>")
            lines.append(f"    <description>{desc_x}</description>")
            lines.append(f"    <location>{m.path}</location>")
            if not available:
                missing = self._registry.get_missing_requirements(
                    m.name,
                    source=m.source,
                )
                if missing:
                    lines.append(f"    <requires>{_escape_xml(missing)}</requires>")
            lines.append("  </skill>")
        lines.append("</skills>")
        return "\n".join(lines)

    def gather_all_skills(self) -> list[SkillMeta]:
        """返回 Catalog 可见的全部 Skills：Workspace、Builtin、External Mirrors。

        CLI ``skill list`` / Inspection Helpers 用它收集 Everything Unranked；Hot-path Retrieval 必须走
        :class:`SkillForgeRouter`。结果包含 Unavailable/未选中项，不能直接当作本轮 Injected Set。
        """
        return self._registry.list_all()

    def _lookup_meta(
        self,
        name: str,
        source: str | None,
    ) -> SkillMeta | None:
        """通过 Local Registry 把 ``(name, source)`` Resolve 为 `SkillMeta`。

        `source=None` 使用 Registry Priority；无命中返回 `None`。Helper 不检查 Requirements，也不读取
        External Remote Source。
        """
        return self._registry.get(name, source=source)


# ----------------------------------------------------------------------
# 渲染辅助函数（模块级、无状态）
# ----------------------------------------------------------------------


def _escape_xml(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


__all__ = ["LocalSkillCatalog"]
