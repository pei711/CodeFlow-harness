"""为 downstream stratification 低成本提取 per-trial proxy metadata。

模块读取 legacy-runner trial dir，为每个 trial 生成 :class:`ProxyFeatures`。计算只解析一个
``session.jsonl`` 和一个 ``result.json``，不 replay、不调用 LLM、不启动 container，因此可用于
cold-start bandit 在 ``stable_fail`` 0/3 bucket 上做 K-means sub-strata；v7 下约 70% task
落在该 bucket，bandit 需要进一步切分 population 选择 cohort。

所有 feature 都是 per-trial、O(session.jsonl)：``turn_count`` 是至少包含一个 Tool call 的
assistant Turn 数；``final_exit_status`` 是 :class:`ExitStatus` category；
``has_tool_calls_ever`` 表示是否曾调用 Tool；``assistant_text_length_avg`` 是全部 assistant
content 的 mean character length；``docker_error_count`` 统计 Tool response 与 exception
traceback 中的 docker-error pattern。

docker error 是 container-side noise indicator，pattern 有意宽松，因为正常 per-trial occurrence
应很少，spike 提示 bandit stratification 前应过滤 infra issue。Proxy 只用于分层，不是 task
正确性、Agent 能力或性能提升的直接证据。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

__all__ = [
    "ExitStatus",
    "ProxyFeatures",
    "extract_features",
    "extract_trial_dir",
]


class ExitStatus(str, Enum):
    """Harness level 观察到的 trial exit category。

    ``PASSED`` / ``FAILED_VERIFIER`` 是正常 exit mode；其余值区分 Agent timeout、Verifier
    timeout、reward file missing、Runtime error、no Session 与 other failure，供 bandit 过滤。
    例如 wall-cap-bound task 不应与 agent-decision-bound task 落入同一 K-means cluster。
    """

    PASSED = "passed"
    FAILED_VERIFIER = "failed_verifier"
    AGENT_TIMEOUT = "agent_timeout"
    VERIFIER_TIMEOUT = "verifier_timeout"
    REWARD_FILE_NOT_FOUND = "reward_file_not_found"
    RUNTIME_ERROR = "runtime_error"
    NO_SESSION = "no_session"
    OTHER = "other"


@dataclass(frozen=True)
class ProxyFeatures:
    """一个 trial 的不可变 cheap feature snapshot。

    ``trial_id`` 保留 attempt directory name，``task_id`` 去掉 ``__`` 后的 attempt suffix；其余
    字段只来自本地 result/session files。对象不持有原始 trajectory。
    """

    trial_id: str
    task_id: str
    turn_count: int
    final_exit_status: ExitStatus
    has_tool_calls_ever: bool
    assistant_text_length_avg: float
    docker_error_count: int


# Docker/容器守护进程错误最常见于 exec 工具输出，或基础设施层故障时运行器的异常回溯中。
_DOCKER_ERROR_PATTERNS = re.compile(
    r"(?:"
    r"docker:\s+error\b"
    r"|cannot\s+connect\s+to\s+the\s+docker\s+daemon"
    r"|error\s+response\s+from\s+daemon"
    r"|no\s+such\s+container"
    r"|container\s+not\s+running"
    r"|docker\.errors\."
    r")",
    re.IGNORECASE,
)


_EXCEPTION_TO_STATUS = {
    "AgentTimeoutError": ExitStatus.AGENT_TIMEOUT,
    "VerifierTimeoutError": ExitStatus.VERIFIER_TIMEOUT,
    "RewardFileNotFoundError": ExitStatus.REWARD_FILE_NOT_FOUND,
    "RuntimeError": ExitStatus.RUNTIME_ERROR,
}


def _classify_exit(result_json: dict, reward_passed: bool | None) -> ExitStatus:
    """把 ``result.json`` exception_info 与 ``reward.txt`` 映射为 ``ExitStatus``。

    reward pass 优先得到 ``PASSED``；known exception type 映射专属状态；明确 reward failure
    得到 ``FAILED_VERIFIER``；unknown exception 为 ``OTHER``；两类证据都缺失为 ``NO_SESSION``。
    """
    if reward_passed is True:
        return ExitStatus.PASSED
    exc = (result_json.get("exception_info") or {}).get("exception_type")
    if exc in _EXCEPTION_TO_STATUS:
        return _EXCEPTION_TO_STATUS[exc]
    if reward_passed is False:
        return ExitStatus.FAILED_VERIFIER
        # 没有奖励且没有已识别异常时归入 OTHER。
    if exc:
        return ExitStatus.OTHER
    return ExitStatus.NO_SESSION


def _read_reward(trial_dir: Path) -> bool | None:
    rt = trial_dir / "verifier" / "reward.txt"
    if not rt.exists():
        return None
    try:
        return float(rt.read_text().strip()) >= 1.0
    except (ValueError, OSError):
        return None


def _read_result_json(trial_dir: Path) -> dict:
    rj = trial_dir / "result.json"
    if not rj.exists():
        return {}
    try:
        return json.loads(rj.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _session_path(trial_dir: Path) -> Path | None:
    """定位 trial dir 下 Agent ``session.jsonl``。

    优先 ``agent/workspace/sessions/tb2-task.jsonl``；不存在时取排序后的首个 ``*.jsonl``。
    sessions dir 缺失或无候选时返回 ``None``。
    """
    sessions_dir = trial_dir / "agent" / "workspace" / "sessions"
    if not sessions_dir.is_dir():
        return None
    # 旧运行器按约定写入单个 tb2-task.jsonl；名称不同时回退到任意 .jsonl 文件。
    preferred = sessions_dir / "tb2-task.jsonl"
    if preferred.exists():
        return preferred
    candidates = sorted(sessions_dir.glob("*.jsonl"))
    return candidates[0] if candidates else None


def _scan_session(session_path: Path) -> tuple[int, bool, float, int]:
    """单次扫描 Session，提取四个 proxy feature。

    返回 ``(turn_count, has_tool_calls_ever, assistant_text_length_avg,
    docker_error_count)``。每行只解析一次；malformed line 跳过。assistant record 更新 content
    length 与 Tool-call Turn 数，Tool record 扫描 Docker error。empty/unreadable Session 返回
    ``0/False/0.0/0``，因此调用方不能把零与完整的 honest zero 自动等同。
    """
    turn_count = 0
    has_tool_calls_ever = False
    assistant_lengths: list[int] = []
    docker_errors = 0

    try:
        with session_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                role = r.get("role")
                if role == "assistant":
                    content = r.get("content") or ""
                    assistant_lengths.append(len(content))
                    if r.get("tool_calls"):
                        turn_count += 1
                        has_tool_calls_ever = True
                elif role == "tool":
                    content = r.get("content") or ""
                    docker_errors += len(_DOCKER_ERROR_PATTERNS.findall(content))
    except OSError:
        pass

    avg_len = (sum(assistant_lengths) / len(assistant_lengths)) if assistant_lengths else 0.0
    return turn_count, has_tool_calls_ever, avg_len, docker_errors


def _trial_task_id(trial_name: str) -> str:
    return trial_name.rsplit("__", 1)[0] if "__" in trial_name else trial_name


def extract_features(trial_dir: str | Path) -> ProxyFeatures:
    """从单个 trial dir 提取 :class:`ProxyFeatures`。

    ``trial_dir`` 不存在或不是 directory 时抛出 :class:`FileNotFoundError`。函数读取 result、
    reward 与可选 Session，再把 exception traceback 中的 Docker error 加入 Session count。
    返回 snapshot，不修改 trial files。
    """
    p = Path(trial_dir)
    if not p.is_dir():
        raise FileNotFoundError(p)

    result = _read_result_json(p)
    reward_passed = _read_reward(p)

    session = _session_path(p)
    if session is None:
        turn_count = 0
        has_tool_calls_ever = False
        avg_len = 0.0
        docker_errors_session = 0
    else:
        turn_count, has_tool_calls_ever, avg_len, docker_errors_session = _scan_session(session)

        # 同时扫描异常回溯中的 Docker 错误。
    exc_tb = (result.get("exception_info") or {}).get("exception_traceback") or ""
    docker_errors = docker_errors_session + len(_DOCKER_ERROR_PATTERNS.findall(exc_tb))

    final = _classify_exit(result, reward_passed)

    return ProxyFeatures(
        trial_id=p.name,
        task_id=_trial_task_id(p.name),
        turn_count=turn_count,
        final_exit_status=final,
        has_tool_calls_ever=has_tool_calls_ever,
        assistant_text_length_avg=avg_len,
        docker_error_count=docker_errors,
    )


def _find_attempt_root(trial_dir: Path) -> Path:
    """按与 ``stability_bucket`` 相同逻辑定位 attempt root。

    直接包含带 ``__`` 且有 ``verifier/`` 的 child 时使用当前目录；仅有一个 nested dir 时
    进入该目录；否则保留当前目录。输入不是 directory 时抛出 ``NotADirectoryError``。
    """
    if not trial_dir.is_dir():
        raise NotADirectoryError(trial_dir)
    has_trial_children = any(p.is_dir() and "__" in p.name and (p / "verifier").is_dir() for p in trial_dir.iterdir())
    if has_trial_children:
        return trial_dir
    nested = [p for p in trial_dir.iterdir() if p.is_dir()]
    if len(nested) == 1:
        return nested[0]
    return trial_dir


def extract_trial_dir(trial_dir: str | Path) -> dict[str, ProxyFeatures]:
    """为 ``trial_dir`` 下每个合法 trial 提取 :class:`ProxyFeatures`。

    输入可为 legacy ``jobs_dir`` 或 dated subdir。只处理 name 含 ``__``、有 ``result.json``
    且存在 ``verifier`` directory 的 child；返回 ``{trial_id: ProxyFeatures}``。不合 shape 的
    entry 静默跳过。
    """
    root = _find_attempt_root(Path(trial_dir))
    out: dict[str, ProxyFeatures] = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir() or "__" not in d.name:
            continue
        if not (d / "result.json").exists():
            continue
        if not (d / "verifier").is_dir():
            continue
        out[d.name] = extract_features(d)
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    from collections import Counter

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trial-dir", required=True, help="legacy jobs_dir or dated subdir")
    ap.add_argument("--json", default=None, help="optional JSON dump path")
    args = ap.parse_args(argv)

    feats = extract_trial_dir(args.trial_dir)
    by_status = Counter(f.final_exit_status.value for f in feats.values())

    print(f"trial_dir: {args.trial_dir}")
    print(f"trials observed: {len(feats)}")
    print("\nExit status breakdown:")
    for status, n in sorted(by_status.items(), key=lambda x: -x[1]):
        print(f"  {status:24s} {n}")

    print("\nTurn count quantiles (incl. wall-cap-zero):")
    turns = sorted(f.turn_count for f in feats.values())
    if turns:
        for q, p in (("min", 0.0), ("p25", 0.25), ("p50", 0.5), ("p75", 0.75), ("max", 1.0)):
            idx = min(len(turns) - 1, int(p * len(turns)))
            print(f"  {q}: {turns[idx]}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    tid: {
                        "task_id": x.task_id,
                        "turn_count": x.turn_count,
                        "final_exit_status": x.final_exit_status.value,
                        "has_tool_calls_ever": x.has_tool_calls_ever,
                        "assistant_text_length_avg": x.assistant_text_length_avg,
                        "docker_error_count": x.docker_error_count,
                    }
                    for tid, x in sorted(feats.items())
                },
                f,
                indent=2,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
