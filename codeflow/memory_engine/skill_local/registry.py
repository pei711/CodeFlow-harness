"""`SkillRegistry`：Skills 的 Internal Data Layer。

职责仅是 Pure IO + Frontmatter Parsing + Dependency Checking，**不**做 Rendering/Retrieval。代码从
Pre-refactor ``agent/skills.py`` Ported，保留 Layer Priority 与 Three-namespace Metadata Lookup
``codeflow > nanobot > openclaw``。当前优先级实际为 Workspace > 按配置顺序的 External/Extra Dirs > Builtin，
其中后挂载 Extra 可覆盖先挂载项，但 Workspace 始终最高。

具体 Layers 是 ``workspace`` 的 ``<workspace>/skills/``、``external`` 的 ``<skills_dir>/``（历史由
``config.skill_forge.skills_dir`` 挂载 ``skill_library`` Output）、以及 ``builtin`` 的 Packaged
``codeflow/memory_engine/skills/``；Builtin Markdown 与 Code 同在 ``memory_engine`` Package。

每层自动识别两种 Disk Layout：Legacy Flat ``<root>/<skill>/SKILL.md`` 使用 Layer Label Source；Mirror
Nested ``<root>/<source>/<skill>/SKILL.md`` 使用首级目录为 Source，例如 ``anthropics``、``antigravity``、
``awesome:foo/bar``。Registry 缓存所有 Physical Skills，并同时提供 Compound ``(source, name)`` 与
Priority Winner 两种索引。

External Runtime Caller 历史上通过 :class:`SkillService`，当前应通过 :class:`LocalSkillCatalog`，而不是把 Registry 当完整 Skill Service。文件
被扫描到不代表 Requirements 满足或本轮被检索。
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
from pathlib import Path

log = logging.getLogger(__name__)

from codeflow.memory_engine.skill_local.types import SkillMeta

# 默认内置 Skill 目录与旧版 ``SkillsLoader`` 使用的路径一致，使替换可直接完成。
#
# 解析为 ``codeflow/memory_engine/skills/``；内置 Markdown 库与 Skill 代码一起位于 memory_engine 包内。
_DEFAULT_BUILTIN_SKILLS_DIR = Path(__file__).parent.parent / "skills"


class SkillRegistry:
    """On-disk Skill Pool 的 Cached Read-only View。

    Discovery 覆盖 ``workspace/skills/``、Configured Extra/External 与 Packaged ``builtin``。跨 Source 同名 Physical Skills
    在 Full List 中保留；Older Name-only Lookup 依据 Priority 选 Winner。Background Watcher 可标记单个
    Source Dirty，下一次 `list_all` 只 Re-scan 该 Slice。

    Read-only 指 Registry 不编辑 ``SKILL.md``；它仍维护 Mutable Cache/Indices，并以 `RLock` 协调 Watcher
    Thread 与读取方。
    """

    def __init__(
        self,
        workspace: Path,
        builtin_skills_dir: Path | None = None,
        external_skills_dir: Path | None = None,
        extra_dirs: "list[tuple[Path, str, bool]] | None" = None,
        scan_max_depth: int = 5,
    ):
        """构造 Registry，并配置扫描 Layers。

        Args:
            extra_dirs: R1 Multi-directory Support。每项为 ``(path, name, always_enabled)``；Name Collision
                时 Later Entry 覆盖 Earlier。``external_skills_dir`` 是 Legacy Parameter，会 Prepend 到
                ``extra_dirs``。
            scan_max_depth: R2 扫描 ``SKILL.md`` 的 Maximum Recursion Depth，防止 Huge Mirror 无界遍历。

        构造时尽力创建 Workspace Skills Dir，但不立即扫描；First `list_all` 才构建 Cache。
        """
        self.workspace = workspace
        self.workspace_skills = workspace / "skills"
        self._scan_max_depth = scan_max_depth
        try:
            self.workspace_skills.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        self._extra_dirs: list[tuple[Path, str, bool]] = []
        if external_skills_dir is not None:
            self._extra_dirs.append((external_skills_dir, "external", False))
        if extra_dirs:
            self._extra_dirs.extend(extra_dirs)
        self.builtin_skills = builtin_skills_dir or _DEFAULT_BUILTIN_SKILLS_DIR
        self._metas_cache: list[SkillMeta] | None = None
        # 主键为 (source, name)。每个物理 Skill 占一项，跨来源同名的镜像项在此保持分离。
        self._by_full_key: dict[tuple[str, str], SkillMeta] | None = None
        # 次级索引：名称 → 最高优先级元数据（工作区 > 外部 > 内置 > 其他来源字母序），
        # 供不携带来源的旧调用方使用。
        self._by_name: dict[str, SkillMeta] | None = None
        # 下次 ``list_all`` 需部分重新扫描的数据源。空集合且 ``_metas_cache`` 不为 None
        # 表示缓存已是最新。由 :meth:`invalidate_source` 填充，使小范围变更不会丢弃整个缓存。
        self._dirty_sources: set[str] = set()
        # 将缓存读写与后台 SkillFileWatcher 线程串行化。否则在重建中途由监视器触发的
        # ``invalidate_source`` 会设置一个标志，随后被 ``list_all`` 尾部清除，丢失本次更新。
        # 使用 RLock，使未来调用方可在持锁区域内组合 ``invalidate_*`` 而不死锁。
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def invalidate_cache(self) -> None:
        """Drop Entire Cache，使下一次 ``list_all`` Re-scan Disk。

        这是 Hard Reset，所有 Sources 重建。只有一个 Logical Source 变化时优先
        :meth:`invalidate_source`，可跳过 Larger Builtin/Workspace/External Sets 的重复扫描。方法在 Lock 内
        同时清空两个 Indices 与 Dirty Flags。
        """
        with self._lock:
            self._metas_cache = None
            self._by_full_key = None
            self._by_name = None
            self._dirty_sources.clear()

    def invalidate_source(self, source: str) -> None:
        """标记 Single Source Dirty；下次 ``list_all`` 只重建该 Slice 并 Merge 回 Cache。

        历史触发点是 :meth:`SkillService.invalidate_skill_cache(source)`，当前 Catalog/Watcher 用等价路径让 Newly Materialized ``SKILL.md`` 无需 Restart 即 Surface，同时避免扫描其他
        Sources。尚无 Cache 时 No-op，因为 First `list_all` 本来会 Full Scan。Source Label 必须与 Discovery
        规则一致，否则不会替换任何 Existing Slice。
        """
        with self._lock:
            if self._metas_cache is None:
                return
            self._dirty_sources.add(source)

    def resolve_source_for_path(self, path: Path) -> str | None:
        """把 ``SKILL.md`` Path 映射回 Source Label。

        规则 Mirrors :meth:`_iter_skill_dirs`，输入 ``path`` 让 Filesystem Watcher Event 无需 Re-walk Tree 就能路由到
        :meth:`invalidate_source`。Flat ``<workspace_skills>/<skill>/SKILL.md`` 映射 ``workspace``；Nested
        ``<workspace_skills>/<src>/.../SKILL.md`` 映射 ``<src>``；Default Literal 是 ``"workspace"``，
        Extra/Builtin 使用各自 Label。

        规则字段名也对应 ``external_skills`` / ``builtin_skills``。Path 在所有 Known Layer 外或直接位于
        Root 时返回 `None`。双方先 ``Path.resolve(strict=False)``，再用 ``relative_to`` 比较，处理 macOS
        ``/var/...``、``/tmp/...`` 与 ``/private/var/...``、``/private/tmp/...`` Symlink 差异；
        `strict=False` 也支持 Target 已消失的 Delete Event。
        """
        try:
            resolved_path = path.resolve(strict=False)
        except OSError:
            resolved_path = path
        layers: list[tuple[Path | None, str]] = [
            (self.workspace_skills, "workspace"),
        ]
        for ed_path, ed_label, _ in self._extra_dirs:
            layers.append((ed_path, ed_label))
        layers.append((self.builtin_skills, "builtin"))
        for root, default in layers:
            if root is None:
                continue
            try:
                resolved_root = root.resolve(strict=False)
            except OSError:
                resolved_root = root
            try:
                rel = resolved_path.parent.relative_to(resolved_root)
            except ValueError:
                continue
            parts = rel.parts
            if not parts:
                return None  # 迭代器会跳过直接位于根目录的 SKILL.md
            return default if len(parts) == 1 else parts[0]
        return None

    def list_all(self) -> list[SkillMeta]:
        """返回所有 Visible Skills，以 Compound ``(source, name)`` Identity 保留并 Cached。

        Cross-source Name Collision 两个 Metas 都出现在 List；同一 ``(source, name)`` 内 First Physical Path
        Wins，后续 Disk-redundant Copy 跳过。Layer Priority 只影响 Caller 省略 Source 时的 Secondary
        ``_by_name`` Lookup，Full List 始终保留每项。

        ``_dirty_sources`` 非空时只 Re-scan 那些 Sources；Flag 由 ``invalidate_source`` 设置，并 Merge Existing Cache，节省 Unchanged Slices
        成本。Full Rebuild 全程持 :attr:`_lock`，防止 Watcher-thread Invalidation 与 Tail-clear Race 丢 Flag。
        返回的是 Cache List Reference，Caller 应只读。
        """
        with self._lock:
            return self._list_all_locked()

    def _list_all_locked(self) -> list[SkillMeta]:
        if self._metas_cache is not None and not self._dirty_sources:
            return self._metas_cache

        # 保留仍有效的缓存项，即来源不在 ``_dirty_sources`` 中的项。完整重建路径
        # 将每个来源视为脏数据，会被下方 ``filter_in`` 拒绝，因此从空集合开始。
        if self._metas_cache is not None:
            kept = [m for m in self._metas_cache if m.source not in self._dirty_sources]
            wanted_dirty = self._dirty_sources
        else:
            kept = []
            wanted_dirty = None  # 完整扫描时保留发现的每个来源

        metas: list[SkillMeta] = list(kept)
        full_key: dict[tuple[str, str], SkillMeta] = {(m.source, m.name): m for m in kept}
        # 物理目录去重按目录名执行；``full_key`` 以（来源，显示名称）为键，使显示名称
        # 与目录名不同时，按名查找仍可用。
        seen_dirs: set[tuple[str, str]] = {(m.source, m.path.parent.name) for m in kept}

        # 遍历分层：工作区 → 按列表顺序的 extra_dirs → 内置。
        # R1：extra_dirs 中后出现的项在名称冲突时覆盖先出现的项。
        layers: list[tuple[Path, str, bool]] = [
            (self.workspace_skills, "workspace", True),
        ]
        for path, name, always_enabled in self._extra_dirs:
            layers.append((path, name, always_enabled))
        layers.append((self.builtin_skills, "builtin", True))

        for root, layer_label, layer_always_enabled in layers:
            if root is None or not root.exists():
                continue
            for skill_dir, source in self._iter_skill_dirs(
                root,
                default_source=layer_label,
                max_depth=self._scan_max_depth,
            ):
                if wanted_dirty is not None and source not in wanted_dirty:
                    continue
                dir_key = (source, skill_dir.name)
                if dir_key in seen_dirs:
                    continue
                seen_dirs.add(dir_key)
                meta = self._build_meta(
                    skill_dir,
                    source,
                    always_enabled=layer_always_enabled,
                )
                if meta is None:
                    continue
                full_key.setdefault((source, meta.name), meta)
                metas.append(meta)

        # 构建 _by_name：后挂载的分层覆盖先挂载的分层（R1）。metas 已按工作区 → 外部 → 内置
        # 的分层顺序迭代。extra_dirs 中最后写入者获胜，但工作区必须始终获胜，
        # 因为它是用户自己的 Skill，应具有最高优先级。
        #
        # 策略：按优先级从低到高迭代，使最后写入获胜。
        # 优先级：内置 < extra_dirs[0] < ... < extra_dirs[-1] < 工作区。
        n_extra = len(self._extra_dirs)
        prio: dict[str, int] = {"builtin": 0}
        for i, (_, label, _) in enumerate(self._extra_dirs):
            prio[label] = i + 1
        prio["workspace"] = n_extra + 1

        by_name: dict[str, SkillMeta] = {}
        for m in sorted(metas, key=lambda x: (prio.get(x.source, 0), x.source)):
            prev = by_name.get(m.name)
            if prev is not None and prev.source != m.source:
                log.warning(
                    "Skill '%s' from '%s' shadowed by '%s'",
                    m.name,
                    prev.source,
                    m.source,
                )
            by_name[m.name] = m  # 最后写入获胜

        # R2：启动日志
        source_counts: dict[str, int] = {}
        for m in metas:
            source_counts[m.source] = source_counts.get(m.source, 0) + 1
        parts_str = " ".join(f"{s}={c}" for s, c in sorted(source_counts.items()))
        log.info(
            "LocalPool loaded: %s (total=%d)",
            parts_str,
            len(metas),
        )

        self._metas_cache = metas
        self._by_full_key = full_key
        self._by_name = by_name
        self._dirty_sources.clear()
        return metas

    def get(self, name: str, source: str | None = None) -> SkillMeta | None:
        """返回 Single Skill Metadata；First ``list_all`` 后为 O(1)。

        ``source=None`` 返回 Priority Winner，当前为 Workspace > Later Extra > Earlier Extra > Builtin；传
        ``source=`` 执行 Exact Compound-key Lookup。无命中返回 `None`。方法会在需要时 Lazy Build 两个
        Indices。
        """
        if self._by_name is None:
            self.list_all()  # 同时填充两个索引
        if source is not None:
            return (self._by_full_key or {}).get((source, name))
        return self._by_name.get(name) if self._by_name else None

    def get_body(self, name: str, source: str | None = None) -> str | None:
        """返回 Full ``SKILL.md`` Content。

        先通过 ``list_all`` Index Resolve Nested Layout；省略 ``source`` 时使用 Layer Priority。File Read OS
        Error 或 Skill Missing 返回 `None`。与 `SkillMeta.content` 不同，这里包含原始 Frontmatter。
        """
        meta = self.get(name, source=source)
        if meta is None:
            return None
        try:
            return meta.path.read_text(encoding="utf-8")
        except OSError:
            return None

    def get_raw_metadata(
        self,
        name: str,
        source: str | None = None,
    ) -> dict | None:
        """返回 YAML-lite Parsed Top-level Frontmatter Dict。

        Skill/Body Missing 或无合法 Frontmatter 时返回 `None`。Parser 只支持 Flat ``key: value``，不应把
        结果当完整 YAML Semantics。
        """
        body = self.get_body(name, source=source)
        if not body:
            return None
        return _parse_frontmatter(body)

    def check_available(
        self,
        name: str,
        source: str | None = None,
    ) -> bool:
        """所有 Declared ``requires`` 中 Binaries 与 Environment Variables 满足时返回 `True`。

        Skill Missing 返回 `False`。检查只探测 ``shutil.which`` 与 Non-empty Env，不执行外部 API Login、
        Version Constraint 或 Workflow Smoke。
        """
        meta = self.get(name, source=source)
        if meta is None:
            return False
        return _check_requirements(meta.requires)

    def get_missing_requirements(
        self,
        name: str,
        source: str | None = None,
    ) -> str:
        """返回 Human-readable Unmet Requirements；Satisfied 或 Skill Missing 时为空。

        Binary 格式为 ``CLI: name``，环境变量为 ``ENV: name``，多项用 Comma Join。空字符串不能区分
        “Skill 不存在”与“全部满足”，需要时应先调用 `get`。
        """
        meta = self.get(name, source=source)
        if meta is None:
            return ""
        return _missing_requirements(meta.requires)

    # ------------------------------------------------------------------
    # 内部辅助函数
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_skill_dirs(
        root: Path,
        default_source: str = "workspace",
        max_depth: int = 5,
    ):
        """通过 Recursive ``SKILL.md`` Glob Yield ``(skill_dir, source)`` Pairs。

        R2 ``max_depth`` 限制 ``root`` 下搜索层级，更深文件 Silently Skip，防止 Huge Mirrors 无界 Walk。
        ``/workspaces/`` Path 排除；若父 Skill Dir 已保留，Nested Skill 被跳过，避免同一 Tree 重复解释。
        Flat Layout 用 Default Source，Nested 用 First Path Part。
        """
        if not root.exists():
            return

        all_md = list(root.rglob("SKILL.md"))
        all_md = [p for p in all_md if "/workspaces/" not in str(p)]
        all_md.sort(key=lambda p: (len(p.parts), str(p)))

        kept_parents: set[str] = set()
        root_str = str(root)
        for skill_md in all_md:
            try:
                rel = skill_md.parent.relative_to(root)
            except ValueError:
                continue
            parts = rel.parts
            if not parts:
                continue
            if len(parts) > max_depth:
                continue
            parent = skill_md.parent
            is_sub = False
            for ancestor in parent.parents:
                if str(ancestor) == root_str:
                    break
                if str(ancestor) in kept_parents:
                    is_sub = True
                    break
            if is_sub:
                continue
            kept_parents.add(str(parent))
            if len(parts) == 1:
                yield parent, default_source
            else:
                yield parent, parts[0]

    def _build_meta(
        self,
        skill_dir: Path,
        source: str,
        *,
        always_enabled: bool = True,
    ) -> SkillMeta | None:
        skill_file = skill_dir / "SKILL.md"
        try:
            body = skill_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

        frontmatter = _parse_frontmatter(body) or {}
        nested = _parse_nested_metadata(frontmatter.get("metadata", ""))
        requires = nested.get("requires", {}) if isinstance(nested.get("requires"), dict) else {}
        always = _resolve_always(frontmatter, nested)
        if not always_enabled:
            always = False
        content = _strip_frontmatter(body)

        # ``stable_key`` 使用目录名。
        stable_key = skill_dir.name
        # 显示名称优先使用前置元数据的 ``name``，否则回退到目录名。
        display_name = (frontmatter.get("name") or "").strip() or skill_dir.name
        description = frontmatter.get("description", "") or display_name

        return SkillMeta(
            id=f"{source}/{stable_key}",
            name=display_name,
            description=description,
            path=skill_file,
            content=content,
            source=source,  # type: ignore[arg-type]
            always=always,
            requires=requires,
            raw_frontmatter=frontmatter,
        )


# ----------------------------------------------------------------------
# 模块级辅助函数（纯函数，可独立测试）
# ----------------------------------------------------------------------


def _parse_frontmatter(content: str) -> dict | None:
    """Minimal YAML-lite Parser，与 Legacy `SkillsLoader` Behavior 一致。

    Expected Format：

        ---
        name: foo
        description: "bar"
        metadata: '{"codeflow": {...}}'
        ---

    Values 会移除 Surrounding Quotes；Nested Keys 不支持。无 Frontmatter 或 Closing Fence 不匹配时返回
    `None`。这不是通用 YAML Parser，复杂值应放在 JSON ``metadata`` String。
    """
    if not content.startswith("---"):
        return None
    m = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not m:
        return None
    metadata: dict = {}
    for line in m.group(1).split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip().strip("\"'")
    return metadata


def _strip_frontmatter(content: str) -> str:
    """返回移除 Leading ``---...---`` Frontmatter 的 Markdown Body。

    Content 不以 Fence 开头或 Pattern 不闭合时原样返回，避免误删正文。成功时从 Closing Fence 后第一
    字符开始，不额外 Strip Body Whitespace。
    """
    if not content.startswith("---"):
        return content
    m = re.match(r"^---\n.*?\n---\n?", content, re.DOTALL)
    if not m:
        return content
    return content[m.end() :]


def _parse_nested_metadata(raw: str) -> dict:
    """从 ``metadata`` JSON Blob 提取 CodeFlow-compatible Namespaced Metadata。

    Lookup Order First Match Wins：``codeflow`` > ``nanobot`` > ``openclaw``。Empty、Invalid JSON、Non-dict 或
    Namespace Value 非 Dict 时返回 ``{}``。函数不合并多个 Namespace，最高优先项完整胜出。
    """
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    for key in ("codeflow", "nanobot", "openclaw"):
        if key in data and isinstance(data[key], dict):
            return data[key]
    return {}


_ALWAYS_TRUTHY = {"true", "1", "yes"}
_ALWAYS_KNOWN = _ALWAYS_TRUTHY | {"false", "0", "no", ""}


def _parse_always_value(raw: object) -> bool:
    """R3 Strict ``always`` Parser，只有 Explicit Truthy Values 计为 True。

    支持 Bool、Numeric 与 ``true/1/yes`` Strings；Unknown String Warning 后按 False，避免任意非空文本被
    Python Truthiness 误判为 Always-enabled。
    """
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        lower = raw.strip().lower()
        if lower not in _ALWAYS_KNOWN:
            log.warning(
                "Unrecognized 'always' value '%s' — treated as false. Expected true/false/yes/no/1/0.",
                raw,
            )
        return lower in _ALWAYS_TRUTHY
    return False


def _resolve_always(frontmatter: dict, nested: dict) -> bool:
    """从 Nested Metadata（Priority）或 Frontmatter Resolve ``always`` Flag。

    Nested 明确提供时优先。R3 Fix 让 ``"false"`` 正确成为 `False`，不再沿用
    ``bool("false") == True`` 的旧错误。两处都缺失时返回 False。
    """
    nested_val = nested.get("always")
    if nested_val is not None:
        return _parse_always_value(nested_val)
    return _parse_always_value(frontmatter.get("always"))


def _check_requirements(requires: dict) -> bool:
    for b in requires.get("bins", []):
        if not isinstance(b, str):
            continue
        if not shutil.which(b):
            return False
    for env in requires.get("env", []):
        if not isinstance(env, str):
            continue
        if not os.environ.get(env):
            return False
    return True


def _missing_requirements(requires: dict) -> str:
    missing: list[str] = []
    for b in requires.get("bins", []):
        if not isinstance(b, str):
            continue
        if not shutil.which(b):
            missing.append(f"CLI: {b}")
    for env in requires.get("env", []):
        if not isinstance(env, str):
            continue
        if not os.environ.get(env):
            missing.append(f"ENV: {env}")
    return ", ".join(missing)
