"""为每个 Turn 创建 Workspace 的 out-of-band shadow-git checkpoint（Bug2 safety net）。

Turn 结束时，Service 用独立 ``--git-dir``、真实 Workspace 作为 work-tree 提交一份 Snapshot，
never 触碰 User 自有 ``.git``。因此多文件编辑在迭代上限处被截断时仍有可恢复现场，Changed
Files 与 Commit id 可进入 Next Turn recovery prompt。Granularity 是 per-turn one commit，与
Claude Code/Cursor 类似。

Scope 有明确上限：只 Snapshot Filesystem state，不保存 Conversation state；Shell Tool 的
``rm``/``mv``/``sed -i`` 会被下一次 ``add -A`` 捕获，却无法归因到具体 Tool call。这是 Working
Tree 的 *undo stack*，not full crash recovery。

Safety 采用 Defense in Depth：Shadow Repo ``info/exclude`` 默认排除 Build、Virtualenv、IDE、OS
Junk 与 likely-credential path；Work-tree 自身 ``.gitignore`` 继续按 Git standard semantics 生效；
``gc.auto`` 与周期 ``git gc --auto`` 控制 loose objects。Every Git invocation 都是 best-effort，
失败记录后退化为 ``None``/空结果，Checkpoint 绝不能打断 Turn。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger

from codeflow.product import WORKSPACE_STATE_DIRNAME

# 将提交者身份写入影子仓库，使提交不依赖也不修改用户的全局 Git 配置。
_GIT_IDENT = (
    "-c",
    "user.name=CodeFlow",
    "-c",
    "user.email=checkpoint@codeflow.local",
    "-c",
    "commit.gpgsign=false",
)


# 将临时或高风险模式写入影子仓库的 ``info/exclude``。通过纵深防御，即使工作区
# 没有 ``.gitignore``，它们也不会进入快照。分类与真实项目通常忽略的内容保持一致：
#
# - 影子仓库自身和 Python 缓存：避免递归收录仓库和常见 Python 构建噪声。
# - 构建和打包产物：常见多语言输出目录，可达 GB 级且没有恢复价值。
# - 虚拟环境：体积大，且可从锁文件重建。
# - 凭据和 dotenv：高影响泄漏载体。用户的 ``.gitignore`` 通常会覆盖，但仍需兜底排除。
# - 日志、操作系统垃圾和 IDE 状态：不是密钥，只会使仓库膨胀。
_DEFAULT_EXCLUDES = """\
# CodeFlow shadow-git default excludes (see checkpoint.py).
# Layered on top of any .gitignore files in the work-tree.

# Self + Python caches
.codeflow/
__pycache__/
*.pyc
*.pyo

# Build / package artifacts
dist/
build/
target/
*.egg-info/
.eggs/
node_modules/
.next/
.nuxt/
out/

# Virtualenvs
# (``env/`` deliberately omitted — too easily collides with a legitimate
# project source dir; users whose env IS a virtualenv typically have it
# in their own .gitignore, which S4-A honors automatically.)
venv/
.venv/
.tox/

# Credentials & dotenv (defense in depth — usually in user's .gitignore too)
.env
.env.*
*.key
*.pem
*.crt
*.p12
.aws/credentials
secrets.yaml
secrets.yml

# Logs
*.log
logs/

# OS junk
.DS_Store
Thumbs.db

# IDE state
.idea/
.vscode/
"""


# 对影子仓库触发 ``git gc --auto`` 的成功提交间隔。``--auto`` 让 Git 根据内部启发式
# 自行判断是否需要 GC，此处只提供心跳。0 完全禁用周期调用。
_GC_EVERY_N_COMMITS = 50


# 单个 Git 子进程的上限。如果没有此限制，NFS 锁、被占用的 ``.git/index.lock`` 或磁盘已满
# 都可能让 ``communicate()`` 永久挂起并锁死 Agent Loop。该值宽裕到可容纳正常冷启动，
# 又足够严格，能在一个 Turn 内识别真正挂起。
_GIT_TIMEOUT_SECONDS = 30.0


class CheckpointService:
    """管理每 Turn 一个 Commit 的 Shadow-git Working-tree Snapshot。

    Workspace 与可选 State Root 在构造时解析；Shadow dir 必须严格位于对应 Root 下，``..``、
    Absolute、空或 ``.`` 等 Escape 立即 `ValueError`，避免不同 Workspace 共享 Shadow Repo 并污染
    ``edited_files``。Repo Lazy Initialize，Commit/GC 计数按 Service instance 维护。

    所有 Git 子进程使用独立 Identity、禁用 GPG、设置 ``core.quotePath=false`` 以保留 CJK Path，
    并受 30 秒 Timeout。Service 只提供恢复证据，不修改 User Git 配置或分支。
    """

    def __init__(
        self,
        workspace: Path,
        shadow_dir: str = f"{WORKSPACE_STATE_DIRNAME}/shadow.git",
        *,
        state: Path | None = None,
    ) -> None:
        self._workspace = Path(workspace).expanduser().resolve()
        state_root = Path(state).expanduser().resolve() if state is not None else self._workspace
        shadow_path = Path(shadow_dir)
        if state is not None and shadow_path.parts[:1] == (WORKSPACE_STATE_DIRNAME,):
            shadow_path = Path(*shadow_path.parts[1:])
        candidate = (state_root / shadow_path).resolve()
        root_label = "workspace" if state is None else "state root"
        # 包容性是关键不变式：如果影子 Git 落到状态根目录之外，按工作区的恢复隔离就会失效。
        # 不同工作区中配置了类似逃逸路径的另一个 AgentLoop 可能共享仓库，交叉污染
        # ``edited_files``。``..``、绝对路径、``""`` 和 ``"."`` 都会触发该问题；
        # 应以清晰错误拒绝，而不是让解析后的路径静默漂移。
        if candidate == state_root or not candidate.is_relative_to(state_root):
            raise ValueError(
                f"shadow_dir={shadow_dir!r} must resolve to a path strictly "
                f"under the {root_label} ({state_root}); got {candidate}"
            )
        self._git_dir = candidate
        self._ready = False
        self._commit_count = 0

    async def _git(self, *args: str) -> tuple[int, str, str]:
        """针对 Shadow Repo 运行 Git Command，并返回 ``(rc, out, err)``。

        Command 固定带独立 git-dir/work-tree 与 ``core.quotePath=false``，使 CJK/Japanese/Emoji
        等 non-ASCII Path 保持真实 UTF-8，而不是 Git 默认 Octal Escape；否则 ``edited_files`` 会以
        ``"\\346\\265\\213"`` gibberish 进入 Recovery Prompt。Subprocess cwd 是真实 Workspace。

        communicate 超过 `_GIT_TIMEOUT_SECONDS` 会 Kill/Wait Process，并合成 ``(-1, "", "timeout")``；
        正常输出用 replacement decoding，调用方据 rc 决定 best-effort degradation。
        """
        cmd = (
            "git",
            f"--git-dir={self._git_dir}",
            f"--work-tree={self._workspace}",
            "-c",
            "core.quotePath=false",
            *args,
        )
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._workspace),
        )
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(),
                timeout=_GIT_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            # NFS、索引锁或磁盘已满时，不泄漏僵尸进程，也不让 Turn 挂起。
            # 合成非零返回码，触发调用方的失败降级路径。
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
            logger.debug(
                "checkpoint git timed out after {}s: {}",
                _GIT_TIMEOUT_SECONDS,
                " ".join(args[:2]),
            )
            return -1, "", "timeout"
        return proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")

    async def _ensure_init(self) -> bool:
        """Lazy Initialize Shadow Repo，幂等返回是否 Ready。

        HEAD 不存在时创建 Parent 并 ``git init``，同时 best-effort 写 NOTICE.txt 解释目录用途、删除
        与禁用方式。每次 Ready 前都会刷新 ``info/exclude``，并配置 gc.auto=256、
        gc.autoDetach=false；前台 GC 避免与 Test/Shutdown rmtree 竞争，且受统一 Git timeout。

        成功设置 `_ready=True`；Init/Git/OSError 记录 Debug 并返回 ``False``，不让 Checkpoint
        Configuration 或 Disk Failure 破坏主 Turn。
        """
        if self._ready:
            return True
        try:
            if not (self._git_dir / "HEAD").exists():
                self._git_dir.parent.mkdir(parents=True, exist_ok=True)
                rc, _, err = await self._git("init")
                if rc != 0:
                    logger.debug("checkpoint init failed: {}", err.strip())
                    return False
                # 在影子 Git 旁写入可发现性提示，让注意到 ``.codeflow/`` 的用户无需搜索代码库即可识别它。
                # 该操作尽力而为，写入失败不影响主流程。
                try:
                    notice = self._git_dir.parent / "NOTICE.txt"
                    notice.write_text(
                        "This directory is created by CodeFlow's runtime "
                        "checkpoint feature (a per-turn safety net). It is "
                        "an out-of-band shadow git repo; your own .git is "
                        "untouched.\n\n"
                        "Safe to delete; will be recreated on next agent run. "
                        'Disable via `runtime.checkpoint.policy = "never"` '
                        "in your CodeFlow config (typically "
                        "~/.codeflow/config.json, or whichever file you "
                        "passed via --config).\n",
                        encoding="utf-8",
                    )
                except OSError:
                    pass
            # 分层忽略：影子仓库默认规则加用户自己的 .gitignore，后者由 Git 在工作树中自动遍历。
            # 两者共同将构建产物、临时状态和用户标记为私密的文件排除在所有快照之外。
            exclude = self._git_dir / "info" / "exclude"
            exclude.parent.mkdir(parents=True, exist_ok=True)
            exclude.write_text(_DEFAULT_EXCLUDES, encoding="utf-8")
            # gc.auto 是 Git 判断对象和引用是否已累积到需要真正 GC 的阈值。初始化时设置一次，
            # 后续每次 ``git gc --auto`` 都可查询同一阈值，无需每次传入 ``-c``。
            await self._git("config", "gc.auto", "256")
            # gc.autoDetach=false 让 Git 决定自动 GC 时在前台运行，而不分离后台守护进程。
            # 分离的 GC 会与工作区清理（测试临时目录、Agent 关闭）竞态；rmtree 遇到仍在写
            # ``objects/`` 的 GC 时会报“目录非空”。同步 GC 受 _GIT_TIMEOUT_SECONDS 约束，同样不会挂起 Turn。
            await self._git("config", "gc.autoDetach", "false")
            self._ready = True
            return True
        except OSError as exc:
            logger.debug("checkpoint init error: {}", exc)
            return False

    async def commit_turn(self, label: str) -> tuple[str | None, list[str]]:
        """把当前 Worktree Snapshot 为一个带 ``label`` 的 Shadow Commit。

        Ready 后依次执行 ``add -A``、cached name-only diff、Commit、short HEAD。Changed Files 必须
        在 Commit 前捕获，才能精确表示本 Turn staged state；无变化不创建空 Commit。成功返回
        ``(checkpoint_id, changed_files)``，增加实例 Commit count 并触发 `_maybe_gc`。

        Init、Add、Commit、Rev-parse 或 OS 任一步失败均 best-effort 返回 ``(None, [])``。这意味着
        “没有 Checkpoint”可能是无变化也可能是 Git Failure，Caller 不能把它当作 Workspace 未改。
        """
        if not await self._ensure_init():
            return None, []
        try:
            rc, _, err = await self._git("add", "-A")
            if rc != 0:
                logger.debug("checkpoint add failed: {}", err.strip())
                return None, []
            # 当前 Turn 暂存的文件就是本 Turn 的变更，需在提交前捕获。
            _, out, _ = await self._git("diff", "--cached", "--name-only")
            changed = [ln for ln in out.splitlines() if ln.strip()]
            if not changed:
                return None, []  # 没有可快照的内容
            rc, _, err = await self._git(*_GIT_IDENT, "commit", "-m", label)
            if rc != 0:
                logger.debug("checkpoint commit failed: {}", err.strip())
                return None, []
            rc, out, _ = await self._git("rev-parse", "--short", "HEAD")
            cid = out.strip() or None
            self._commit_count += 1
            await self._maybe_gc()
            return cid, changed
        except OSError as exc:
            logger.debug("checkpoint commit error: {}", exc)
            return None, []

    async def _maybe_gc(self) -> None:
        """周期运行 ``git gc --auto``，避免 Long-lived Session 无限累积 Loose Objects。

        ``--auto`` 在低于 ``gc.auto``（默认 256 Loose Objects）时是 no-op，steady state 成本只是
        Cheap rev-list count，不会每 50 次都 Repack。``_commit_count`` 是 per-instance，重建
        CheckpointService（例如 Fresh AgentLoop Start）会重置，所以 50-commit heartbeat 只是 Hint。

        Load-bearing safety net 是 Init 时设置且跨 Process Restart 仍由 Repo 保存的 ``gc.auto=256``。
        Heartbeat 禁用或 GC failure 都不影响 Turn，失败只记录 Debug。
        """
        if _GC_EVERY_N_COMMITS <= 0:
            return
        if self._commit_count % _GC_EVERY_N_COMMITS != 0:
            return
        rc, _, err = await self._git("gc", "--auto")
        if rc != 0:
            logger.debug("checkpoint gc failed: {}", err.strip())


__all__ = ["CheckpointService"]
