"""为 paired baseline trial dir 计算 per-task k-attempt stability bucket。

模块读取 legacy-runner 目录，例如 ``data/v7_k3_baseline/<dated>/``，其中每个 task 最多有
``k`` 个 independent attempt；输出供 cold-start bandit task-cohort stratification 使用。

``k=3`` 时：``STABLE_PASS`` 是 3/3，``BORDERLINE_2_3`` 是 2/3，
``BORDERLINE_1_3`` 是 1/3，``STABLE_FAIL`` 是 0/3。pass criterion 仅为
``verifier/reward.txt`` 存在且值 ``>= 1.0``；无 reward 的 ``RewardFileNotFoundError``、
wall-clock ``AgentTimeoutError``、``VerifierTimeoutError`` 等都按 FAIL。

k 从实际数据推断。pre-trial failure 可能使 task 少于 nominal k 个 attempt，bucket 按 observed
attempt fraction 计算；例如 nominal k=3 但只有 1/2 pass，仍在该 grouping 中归为
``BORDERLINE_1_3``。bucket 描述重复试验稳定性，不解释失败原因，也不是模型能力结论。
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class StabilityBucket(str, Enum):
    """按 observed pass fraction 划分的四类 task stability。

    两个 borderline 名称沿用 k=3 语义；arbitrary k 时由 pass count 相对 midpoint 映射。
    """

    STABLE_PASS = "stable_pass"
    BORDERLINE_2_3 = "borderline_2_3"
    BORDERLINE_1_3 = "borderline_1_3"
    STABLE_FAIL = "stable_fail"


@dataclass(frozen=True)
class TaskStability:
    """一个 task 的 observed attempt count、pass count 与 bucket snapshot。"""

    task_id: str
    attempts: int
    passes: int
    bucket: StabilityBucket


def _trial_passed(trial_dir: Path) -> bool:
    """仅当 ``verifier/reward.txt`` 可读且值 ``>= 1.0`` 时返回 ``True``。

    missing、invalid 或 I/O error 都保守返回 ``False``，不尝试从 result status 推断 pass。
    """
    reward = trial_dir / "verifier" / "reward.txt"
    if not reward.exists():
        return False
    try:
        return float(reward.read_text().strip()) >= 1.0
    except (ValueError, OSError):
        return False


def _bucket_for(passes: int, attempts: int) -> StabilityBucket:
    """把 ``(passes, attempts)`` 映射为 ``StabilityBucket``。

    规则主要为 k in ``{2, 3}`` 设计。arbitrary k 时，``passes == 0`` 永远
    ``STABLE_FAIL``，``passes == attempts`` 永远 ``STABLE_PASS``；中间值按 pass count 位于
    midpoint 哪一侧分成 ``BORDERLINE_2_3`` / ``BORDERLINE_1_3``，恰好一半归后者。
    """
    if passes == 0:
        return StabilityBucket.STABLE_FAIL
    if passes == attempts:
        return StabilityBucket.STABLE_PASS
    # 混合情况：区分“多数通过”和“多数失败”。
    if passes * 2 > attempts:
        return StabilityBucket.BORDERLINE_2_3
    return StabilityBucket.BORDERLINE_1_3


def _task_id(trial_name: str) -> str:
    """从 legacy trial dir name 提取 canonical task id。

    layout 为 ``{task-id}__{8-char-suffix}``，删除最后一个 ``__`` suffix；无 separator 时
    原样返回。
    """
    sep = "__"
    if sep in trial_name:
        return trial_name.rsplit(sep, 1)[0]
    return trial_name


def _looks_like_trial_dir(p: Path) -> bool:
    """判断 path 是否符合 legacy trial dir shape。

    trial dir 要求 name 含 ``__``、top-level ``result.json`` AND ``verifier/`` subdir。
    job-level dated dir 也有 result.json，但自身没有 verifier，因此后者是关键 discriminator。
    """
    return p.is_dir() and "__" in p.name and (p / "result.json").exists() and (p / "verifier").is_dir()


def _find_attempt_root(trial_dir: Path) -> Path:
    """定位 children 为 per-trial dirs 的 attempt root。

    输入可为包含 dated subdir 的 legacy ``jobs_dir``，如 ``data/v7_k3_baseline/``，也可直接
    是 dated subdir。两类 name 都可能含 ``__``，因此使用 ``_looks_like_trial_dir`` 的
    result/verifier shape 判断。输入非目录抛出 ``NotADirectoryError``。
    """
    if not trial_dir.is_dir():
        raise NotADirectoryError(trial_dir)
    if any(_looks_like_trial_dir(p) for p in trial_dir.iterdir()):
        return trial_dir
    nested = [p for p in trial_dir.iterdir() if p.is_dir()]
    if len(nested) == 1:
        return nested[0]
    return trial_dir


def compute_stability(trial_dir: str | Path) -> dict[str, TaskStability]:
    """聚合每个 task 的 k-attempt pass count，并分配 bucket。

    返回 ``{task_id: TaskStability}``。遍历 attempt root 中 name 含 ``__`` 的 directory，按
    ``_trial_passed`` 记录 bool；不验证 nominal k 是否一致。函数只读 trial data。
    """
    root = _find_attempt_root(Path(trial_dir))
    per_task_passes: dict[str, list[bool]] = defaultdict(list)
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if "__" not in d.name:
            continue
        task_id = _task_id(d.name)
        per_task_passes[task_id].append(_trial_passed(d))

    result: dict[str, TaskStability] = {}
    for task_id, passes in per_task_passes.items():
        n = len(passes)
        k = sum(passes)
        result[task_id] = TaskStability(
            task_id=task_id,
            attempts=n,
            passes=k,
            bucket=_bucket_for(k, n),
        )
    return result


def bucket_counts(stability: Iterable[TaskStability]) -> dict[StabilityBucket, int]:
    """统计 iterable 中每个 ``StabilityBucket`` 的 task 数量。

    返回包含所有 Enum key 的 dict，即使某类 count 为 0。
    """
    counts = {b: 0 for b in StabilityBucket}
    for ts in stability:
        counts[ts.bucket] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trial-dir", required=True, help="legacy jobs_dir or dated subdir")
    ap.add_argument("--json", default=None, help="optional JSON dump path")
    args = ap.parse_args(argv)

    stab = compute_stability(args.trial_dir)
    counts = bucket_counts(stab.values())

    print(f"trial_dir: {args.trial_dir}")
    print(f"tasks observed: {len(stab)}")
    for b in StabilityBucket:
        print(f"  {b.value:18s} {counts[b]}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    tid: {"attempts": ts.attempts, "passes": ts.passes, "bucket": ts.bucket.value}
                    for tid, ts in sorted(stab.items())
                },
                f,
                indent=2,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
