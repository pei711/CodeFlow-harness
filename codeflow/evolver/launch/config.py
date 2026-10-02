"""把 YAML run spec 加载为 validated ``RunSpec``，并应用 ``--smoke`` overlay。

The YAML shape:

    bench: appworld
    repo_root: /path/to/subject          # the repo being evolved
    base_sha: <commit>                   # optional; omitted -> repo_root HEAD at launch
    work_dir: ./evo_work

    models:                              # optional; omitted -> CodeFlow's own model
      driver:  {provider: claude_cli, model: claude-haiku-4-5}
      design:  {provider: claude_cli, model: claude-opus-4-8}
      verdict: {provider: openai_compat, base_url: ..., model: ...}

    funnel:                              # optional; SOP-aligned defaults
      k_screen: 1
      k_confirm: 3
      budget:      {max_why_per_round: 2, candidates_per_why: 3}
      termination: {patience: 10, max_rounds: 20}
      anchor:      {n_sentinel: 12, cull_sigma_mult: 1.5}

    bench_config: {...}                  # schema owned by the bench entry

    smoke: {...}                         # optional deep-merge overlay for --smoke

``--smoke`` 先应用 built-in shrink default：1 WHY x 1 candidate x 1 round、K=1，再 deep-merge
用户 ``smoke:``，并给 work_dir 加 ``_smoke`` suffix，确保 smoke state 不触碰 real run。

loader 还固定 subject commit、验证 role/model shape 与 funnel bounds，并确保 work_dir 在 repo
外、由 ``run_meta.json`` 标记 ownership 且可写。配置加载成功只证明静态 contract 合法，不
执行 provider/benchmark precheck。
"""

from __future__ import annotations

import copy
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from codeflow.evolver.launch.state import META_FILENAME
from codeflow.evolver.orchestrator.config import (
    AnchorParams,
    Budget,
    OrchestratorConfig,
    Termination,
)

SMOKE_BUILTIN: dict = {
    "funnel": {
        "k_confirm": 1,
        "budget": {"max_why_per_round": 1, "candidates_per_why": 1, "recombinations_per_round": 0},
        "termination": {"patience": 1, "max_rounds": 1},
    },
}


class RunSpecError(ValueError):
    """用户必须修复的 run spec 问题，message 提供具体字段或路径原因。"""


def _redact_secrets(models: dict) -> dict:
    if not isinstance(models, dict):
        return models
    out = {}
    for role, spec in models.items():
        if isinstance(spec, dict):
            out[role] = {k: ("<redacted>" if "key" in k.lower() and k != "api_key_env" else v) for k, v in spec.items()}
        else:
            out[role] = spec
    return out


def deep_merge(base: dict, overlay: dict) -> dict:
    """递归合并 mapping，并返回不共享 mutable value 的 deep copy。

    两侧同 key 都为 dict 时递归；其他情况 overlay 完全替换 base。输入不被修改。
    """
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@dataclass
class RunSpec:
    """一次 Evolution Run 的 effective、已解析配置。

    对象拥有 subject repo/base commit、owned work_dir、orchestrator funnel、role model config、
    bench-specific config、smoke 标记与原始 merged mapping。``base_sha_defaulted`` 记录 YAML
    是否省略 commit，便于提示 provenance。
    """

    bench: str
    repo_root: Path
    base_sha: str
    work_dir: Path
    funnel: OrchestratorConfig
    models: dict = field(default_factory=dict)
    bench_config: dict = field(default_factory=dict)
    smoke: bool = False
    base_sha_defaulted: bool = False
    config_dir: Path = field(default_factory=Path.cwd)
    raw: dict = field(default_factory=dict)

    def snapshot(self) -> dict:
        """生成写入 run_meta、用于 drift guard 的 effective config snapshot。

        secret 在落盘前用 constant ``<redacted>`` 替换，因为 work_dir 可能共享；constant
        redaction 还确保 key rotation 不触发 config drift。snapshot 包含 bench、repo/base、
        models、funnel、bench_config 与 smoke，不包含 runtime callable。
        """
        return {
            "bench": self.bench,
            "repo_root": str(self.repo_root),
            "base_sha": self.base_sha,
            "models": _redact_secrets(self.raw.get("models", {})),
            "funnel": self.raw.get("funnel", {}),
            "bench_config": self.raw.get("bench_config", {}),
            "smoke": self.smoke,
        }


def _build_funnel(repo_root: Path, work_dir: Path, funnel: dict) -> OrchestratorConfig:
    if not isinstance(funnel, dict):
        raise RunSpecError(f"funnel: must be a mapping, got {type(funnel).__name__}")
    known = {"k_screen", "k_confirm", "anchor", "budget", "termination", "sealed_test_split"}
    unknown = set(funnel) - known
    if unknown:
        raise RunSpecError(f"funnel: unknown keys {sorted(unknown)}")
    try:
        cfg = OrchestratorConfig(
            repo_root=repo_root,
            work_dir=work_dir,
            driver_llm_spec={},
            k_screen=int(funnel.get("k_screen", 1)),
            k_confirm=int(funnel.get("k_confirm", 3)),
            anchor=AnchorParams(**(funnel.get("anchor") or {})),
            budget=Budget(**(funnel.get("budget") or {})),
            termination=Termination(**(funnel.get("termination") or {})),
            sealed_test_split=funnel.get("sealed_test_split", "test"),
            sealed_output_dir=work_dir / "sealed",
        )
    except (TypeError, ValueError) as exc:
        raise RunSpecError(f"funnel: {exc}") from exc
    if cfg.k_screen < 1 or cfg.k_confirm < 1:
        raise RunSpecError("funnel: k_screen and k_confirm must be >= 1")
    if cfg.budget.max_why_per_round < 1 or cfg.budget.candidates_per_why < 1:
        raise RunSpecError("funnel: budget.max_why_per_round and budget.candidates_per_why must be >= 1")
    if cfg.termination.patience < 1 or cfg.termination.max_rounds < 1:
        raise RunSpecError("funnel: termination.patience and termination.max_rounds must be >= 1")
    return cfg


def load_run_spec(config_path: str | Path, *, smoke: bool = False) -> RunSpec:
    """读取、合并并验证一份 run-spec YAML。

    file missing、YAML/top-level shape、required key、Git checkout/commit、work_dir ownership、
    model role 或 funnel value 不合规时抛出 ``RunSpecError``。relative path 以 config file 目录
    为基准，使从不同 cwd resume 仍定位同一 state。省略 base_sha 时立即解析当前 HEAD 为 full
    commit；smoke 使用独立 suffixed work_dir。

    返回 ``RunSpec``，并确保 work_dir 已存在且可写；不会创建 ``RunMeta`` 或运行 trial。
    """
    path = Path(config_path)
    if not path.is_file():
        raise RunSpecError(f"config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise RunSpecError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise RunSpecError(f"{path}: top level must be a mapping")

    if smoke:
        overlay = data.pop("smoke", {}) or {}
        data = deep_merge(deep_merge(data, SMOKE_BUILTIN), overlay)
    else:
        data.pop("smoke", None)

    missing = [k for k in ("bench", "repo_root", "work_dir") if not data.get(k)]
    if missing:
        raise RunSpecError(f"{path}: missing required keys: {missing}")

    def _abs(value: str) -> Path:
        # 相对路径以配置文件而非当前工作目录为基准；从其他目录恢复时必须找到同一 work_dir，
        # 而不是静默开始新运行。
        p = Path(value).expanduser()
        return p.resolve() if p.is_absolute() else (path.parent / p).resolve()

    repo_root = _abs(data["repo_root"])
    if not (repo_root / ".git").exists():
        raise RunSpecError(f"repo_root is not a git checkout: {repo_root}")

    base_sha = str(data.get("base_sha") or "").strip()
    base_sha_defaulted = not base_sha
    if base_sha_defaulted:
        base_sha = _resolve_head(repo_root)
    base_sha = _resolve_commit(repo_root, base_sha)
    data["base_sha"] = base_sha
    work_dir = _abs(data["work_dir"])
    if smoke:
        work_dir = work_dir.with_name(work_dir.name + "_smoke")
    _validate_work_dir(repo_root, work_dir)

    models = data.get("models") or {}
    if not isinstance(models, dict):
        raise RunSpecError("models: must be a mapping of role -> provider spec")
    unknown_roles = set(models) - {"driver", "design", "verdict"}
    if unknown_roles:
        raise RunSpecError(f"models: unknown roles {sorted(unknown_roles)}")
    for role, spec_val in models.items():
        if not isinstance(spec_val, dict):
            raise RunSpecError(f"models.{role}: must be a mapping (provider/model/...), got {type(spec_val).__name__}")

    return RunSpec(
        bench=str(data["bench"]),
        repo_root=repo_root,
        base_sha=base_sha,
        work_dir=work_dir,
        funnel=_build_funnel(repo_root, work_dir, data.get("funnel") or {}),
        models=models,
        bench_config=data.get("bench_config") or {},
        smoke=smoke,
        base_sha_defaulted=base_sha_defaulted,
        config_dir=path.parent.resolve(),
        raw=data,
    )


def _resolve_head(repo_root: Path) -> str:
    """YAML 省略 ``base_sha`` 时解析 subject repo 当前 ``HEAD``。

    load time 固定为 full SHA 并写入 config snapshot，使 run 始终 pinned 到启动时 commit；repo
    后续新增 commit 时 resume 会触发 drift guard，而不会 silently move root。git 不可执行或
    rev-parse 失败抛出 ``RunSpecError``。
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise RunSpecError(f"git is not runnable ({exc}) — install git") from exc
    if proc.returncode != 0:
        raise RunSpecError(f"base_sha omitted and resolving HEAD of {repo_root} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _resolve_commit(repo_root: Path, revision: str) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--verify", f"{revision}^{{commit}}"],
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise RunSpecError(f"git is not runnable ({exc}) — install git") from exc
    if proc.returncode != 0:
        raise RunSpecError(f"base_sha {revision!r} does not resolve to a commit in {repo_root}: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _validate_work_dir(repo_root: Path, work_dir: Path) -> None:
    repo_root = repo_root.resolve()
    work_dir = work_dir.resolve()
    if work_dir == repo_root or repo_root in work_dir.parents:
        raise RunSpecError(
            f"work_dir must be outside repo_root so cleanup and candidate worktrees "
            f"cannot mutate the subject checkout: {work_dir}"
        )
    if work_dir in repo_root.parents:
        raise RunSpecError(
            f"work_dir must not contain repo_root because cleanup is scoped to an owned run directory: {work_dir}"
        )
    if work_dir.exists() and not work_dir.is_dir():
        raise RunSpecError(f"work_dir exists but is not a directory: {work_dir}")
    marker = work_dir / META_FILENAME
    if marker.is_symlink():
        raise RunSpecError(f"work_dir ownership marker must not be a symlink: {marker}")
    if work_dir.is_dir() and any(work_dir.iterdir()) and not marker.is_file():
        raise RunSpecError(
            f"work_dir is nonempty but is not an owned Evolution Run (missing {META_FILENAME}): {work_dir}"
        )
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".codeflow-evolver-write-", dir=work_dir):
            pass
    except OSError as exc:
        raise RunSpecError(f"work_dir is not writable: {work_dir}: {exc}") from exc


__all__ = ["RunSpec", "RunSpecError", "load_run_spec", "deep_merge", "SMOKE_BUILTIN"]
