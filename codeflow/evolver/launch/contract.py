"""定义 benchmark 接入 Evolver 所需的 bench plugin contract。

bench module 暴露 ``build(ctx: LaunchContext) -> BenchBundle``。bundle 只做 pure wiring，runner
真正调用 closure 前不执行 expensive work，因此 ``status`` 可以只 build bundle 统计 artifact。

按 ``docs/specs/evolve-bench-contract.md``，新 bench 必须提供 scorer subprocess、带 infra marker
的 result file、result -> ``TaskEval`` reader、per-attempt trajectory、train/test split，以及
subject repo editable-path whitelist。funnel、gates、sealed test 与 resume 都由 shared loop 提供。
plugin build 成功只证明接口已装配，不证明环境 precheck 或 trial 可运行。
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from codeflow.evolver.launch.config import RunSpec
from codeflow.evolver.launch.models import CallFn


@dataclass(frozen=True)
class LaunchContext:
    """构建 BenchBundle 所需的 effective RunSpec 与 role model callables。

    ``smoke`` property 直接反映 spec，context 不拥有 run lifecycle。
    """

    spec: RunSpec
    models: dict[str, Optional[CallFn]]

    @property
    def smoke(self) -> bool:
        return self.spec.smoke


@dataclass
class BenchBundle:
    """runner state machine 所需的全部 lazy-evaluated benchmark wiring。

    ``cold_start_done``/``run_cold_start`` 必须在 trial granularity 幂等，重复调用只填 missing
    trial。runner 每次 run 都调用 ``run_cold_start``，即使 base trial 已齐；bench 欠下的 SOP §0
    infra-salvage rerun 必须放在 closure 内，不能藏在 done-count 后。

    ``unseal`` 接收 journal records + built orchestrator，返回 plain dict report；``None`` 表示
    bench 未配置 sealed test。optional ``precheck`` 在 dead endpoint、bound port、missing install
    等环境不支持时抛出 actionable ``RuntimeError``；check 在付费 trial 前调用它。
    """

    root_node_id: str
    root_node: Any
    journal_path: Path
    cold_start_total: int
    cold_start_done: Callable[[], int]
    run_cold_start: Callable[[], None]
    build_orchestrator: Callable[[], Any]
    unseal: Optional[Callable[[list[dict], Any], dict]] = None
    precheck: Optional[Callable[[], None]] = None


def validate_whitelist(repo_root: Path, base_sha: str, prefixes: tuple[str, ...]) -> None:
    """whitelist 任一 entry 在 ``base_sha`` 匹配不到文件时 fail loudly。

    empty whitelist 直接拒绝。函数用 ``git ls-tree -r --name-only`` 列出 base paths；trailing
    slash prefix 按 subtree，其他按 exact file。dead prefix 在 Runtime 不会自然报错，只会让
    designer edit 被当作 out-of-whitelist 静默 revert、所有 candidate 为空；这曾浪费完整 run，
    因此唯一 honest behavior 是启动前拒绝。
    """
    if not prefixes:
        raise ValueError("whitelist is empty: the designer would have no editable surface")
    proc = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", base_sha],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise ValueError(f"cannot list {base_sha} in {repo_root}: {proc.stderr.strip()}")
    paths = proc.stdout.splitlines()
    dead = [
        pattern
        for pattern in prefixes
        if not any(path.startswith(pattern) if pattern.endswith("/") else path == pattern for path in paths)
    ]
    if dead:
        raise ValueError(
            f"whitelist entries match no files at {base_sha[:12]}: {dead} — "
            "designer edits would be silently dropped; fix the entries "
            "(or the base_sha) before running"
        )


__all__ = ["BenchBundle", "LaunchContext", "validate_whitelist"]
