"""Scorer driver and result readers (the parent side of the grading process).

The scorer never checks a candidate out as a working tree. It reads the single
declared mutable file out of the candidate commit with ``git show``, drops that
copy into a throwaway directory, and grades it there. A candidate therefore
cannot influence the run through any file other than the one its Candidate
Manifest declares, and the live subject checkout is never written to.

``grade.py`` is always the copy that sits next to this module in the *live*
checkout, never a version taken from the candidate commit, so the measurement
code is fixed for the whole run.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from benchmarks.appworld.evolve import tasks as task_defs
from codeflow.evolver.orchestrator.scoring import TaskEval, prefer_rerun_measurement
from codeflow.evolver.tree import git_ops

MODULE_PATH = "benchmarks/appworld/agent_cli.py"
GRADE_SCRIPT = Path(__file__).resolve().parent / "grade.py"
DEFAULT_TIMEOUT = 120.0


def _read_record(path: Path) -> dict | None:
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def _is_done(path: Path) -> bool:
    """A trial counts as done only when a complete, non-infra result exists."""
    record = _read_record(path)
    return bool(record) and not record.get("infra_error")


def _write_marker(out_dir: Path, task_id: str, k: int, *, infra: str | None, detail: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{task_id}_k{k}.json"
    path.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "k": k,
                "success": False,
                "infra_error": infra,
                "detail": detail,
                "cases": [],
            },
            indent=2,
            sort_keys=True,
        )
    )


def score_trials(
    repo_root: str | Path,
    commit_sha: str,
    out_dir: str | Path,
    task_ids: list[str],
    k: int,
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> None:
    """Fill the missing trials of ``task_ids`` at K under ``out_dir``.

    Idempotent at trial granularity: an existing complete non-infra result file
    is never recomputed, so an interrupted run resumes by re-invocation.
    """
    out_dir = Path(out_dir)
    pending = [(tid, i) for tid in task_ids for i in range(k) if not _is_done(out_dir / f"{tid}_k{i}.json")]
    if not pending:
        return
    try:
        source = git_ops.read_file_at(Path(repo_root), commit_sha, MODULE_PATH)
    except git_ops.GitOpError as exc:
        for tid, i in pending:
            _write_marker(out_dir, tid, i, infra=f"cannot read {MODULE_PATH} at {commit_sha[:12]}: {exc}", detail="")
        return
    with tempfile.TemporaryDirectory(prefix="small-real-subject-") as tmp:
        module_path = Path(tmp) / "agent_cli.py"
        module_path.write_bytes(source)
        job = {
            "module_path": str(module_path),
            "out_dir": str(out_dir),
            "trials": [
                {
                    "task_id": tid,
                    "k": i,
                    "cases": task_defs.task_for(tid).to_payload()["cases"],
                }
                for tid, i in pending
            ],
        }
        try:
            proc = subprocess.run(
                [sys.executable, str(GRADE_SCRIPT)],
                input=json.dumps(job),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            # 目标模块是确定性的标准库文本处理，因此超时表示候选挂起。这是候选失败，
            # 而不是测量失败；若标记为基础设施故障，会允许无限重跑，把坏候选变成
            # 无法测量的运行。
            for tid, i in pending:
                if not (out_dir / f"{tid}_k{i}.json").is_file():
                    _write_marker(out_dir, tid, i, infra=None, detail=f"grading timed out after {timeout:.0f}s")
            return
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()
        reason = tail[-1] if tail else f"grading process exited {proc.returncode}"
        for tid, i in pending:
            if not (out_dir / f"{tid}_k{i}.json").is_file():
                _write_marker(out_dir, tid, i, infra=reason, detail="")
        return
    for tid, i in pending:
        if not (out_dir / f"{tid}_k{i}.json").is_file():
            _write_marker(out_dir, tid, i, infra="grading process wrote no result file", detail="")


def read_out_dir(out_dir: str | Path) -> dict[str, TaskEval]:
    """Aggregate one out-dir into per-task :class:`TaskEval`."""
    out_dir = Path(out_dir)
    files = sorted(out_dir.glob("*_k*.json"))
    if not files:
        raise FileNotFoundError(f"no per-trial *_k*.json files in out-dir: {out_dir}")
    passes: dict[str, int] = {}
    attempts: dict[str, int] = {}
    infra: dict[str, int] = {}
    for path in files:
        record = _read_record(path)
        if record is None:
            continue
        tid = str(record.get("task_id") or path.name.rsplit("_k", 1)[0])
        attempts[tid] = attempts.get(tid, 0) + 1
        if record.get("infra_error"):
            infra[tid] = infra.get(tid, 0) + 1
        if record.get("success"):
            passes[tid] = passes.get(tid, 0) + 1
    return {
        tid: TaskEval(
            task_id=tid,
            passes=passes.get(tid, 0),
            attempts=count,
            infra_attempts=infra.get(tid, 0),
        )
        for tid, count in attempts.items()
    }


def ladder_out_dirs(out_dir: str | Path) -> list[Path]:
    """The out-dir plus the infra-rerun ladder siblings that exist."""
    out_dir = Path(out_dir)
    dirs = []
    for name in (out_dir.name, f"{out_dir.name}_infra_rerun1", f"{out_dir.name}_infra_rerun2"):
        path = out_dir.parent / name
        if path.is_dir() and any(path.glob("*_k*.json")):
            dirs.append(path)
    return dirs


def read_kept_out_dir(out_dir: str | Path, *, expected_attempts: int) -> dict[str, TaskEval]:
    """Per-task KEPT measurement across the infra-rerun ladder.

    Uses the same keep rule as ``eval_with_infra_rerun``, so a control arm read
    back from disk sees the measurement the candidate arm was scored with.
    """
    if expected_attempts < 1:
        raise ValueError("expected_attempts must be >= 1")
    dirs = ladder_out_dirs(out_dir)
    if not dirs:
        raise FileNotFoundError(f"no per-trial *_k*.json files in out-dir: {out_dir}")
    kept: dict[str, TaskEval] = {}
    for path in dirs:
        for tid, ev in read_out_dir(path).items():
            if prefer_rerun_measurement(kept.get(tid), ev, expected_attempts=expected_attempts):
                kept[tid] = ev
    return kept


def read_case_failures(out_dir: str | Path, task_ids: list[str]) -> dict[str, list[dict]]:
    """The first observed failing-case detail per task, for design evidence."""
    out_dir = Path(out_dir)
    failures: dict[str, list[dict]] = {}
    for tid in task_ids:
        for path in sorted(out_dir.glob(f"{tid}_k*.json")):
            record = _read_record(path)
            if record is None or record.get("success") or record.get("infra_error"):
                continue
            bad = [case for case in record.get("cases") or [] if not case.get("ok")]
            failures[tid] = bad or [{"detail": record.get("detail") or "no case detail recorded"}]
            break
    return failures


def make_eval_fn(repo_root: str | Path, runs_root: str | Path, *, timeout: float = DEFAULT_TIMEOUT):
    """Build the bench :class:`~codeflow.evolver.orchestrator.scoring.EvalFn`."""
    repo_root = Path(repo_root)
    runs_root = Path(runs_root)

    def eval_fn(node, task_ids, k, job_name, *, split="train"):
        ids = list(task_ids)
        if not ids:
            return {}
        out_dir = runs_root / job_name
        score_trials(repo_root, node.git_commit_sha, out_dir, ids, k, timeout=timeout)
        evals = read_out_dir(out_dir)
        return {tid: ev for tid, ev in evals.items() if tid in set(ids)}

    return eval_fn


__all__ = [
    "DEFAULT_TIMEOUT",
    "GRADE_SCRIPT",
    "MODULE_PATH",
    "ladder_out_dirs",
    "make_eval_fn",
    "read_case_failures",
    "read_kept_out_dir",
    "read_out_dir",
    "score_trials",
]
