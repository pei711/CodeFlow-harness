"""记录每个 trial 的 activation ledger 与 code beacon。

每次 Runtime activation event——hook fire、skill injection mirror、code beacon 或 presence
assert——都向 ``<workspace>/activation_ledger.jsonl`` 追加一行 JSON。写入是 best-effort：
ledger failure 绝不能改变 trial 行为或结果，否则测量工具本身会污染 benchmark。

``activation_beacon`` 是每个 evolved code path 必须包含的一行 instrumentation（design
section 3 的 code class）。它优先从当前 asyncio context 获取 workspace，再回退到
``CODEFLOW_ACTIVATION_WORKSPACE``；未设置时静默 no-op，使 product Runtime 不承担 I/O 成本。

ledger 非空只证明 instrumentation 留下记录，不证明机制产生了预期效果、task 完成或
candidate 可归因；collection marker 还用于区分 honest zero 与根本未接线。
"""

from __future__ import annotations

import json
import os
import time
from contextvars import ContextVar
from pathlib import Path

LEDGER_FILENAME = "activation_ledger.jsonl"
WORKSPACE_ENV = "CODEFLOW_ACTIVATION_WORKSPACE"

_workspace_var: ContextVar[str | None] = ContextVar("codeflow_activation_workspace", default=None)


def set_activation_workspace(workspace: "Path | str"):
    """把 activation workspace 绑定到当前 asyncio context。

    benchmark Harness 每个 trial 调用一次。child task 会继承 ``ContextVar`` binding，因此
    同一进程并发 trial 不会 cross-write；process-global env var 无法提供这一保证。返回值是
    ``ContextVar.set`` token，调用方可用于恢复之前绑定。
    """
    return _workspace_var.set(str(workspace))


class ActivationLedger:
    """向单个 workspace 的 ``activation_ledger.jsonl`` 追加事件。

    实例只持有目标 path，不缓存 file handle。每次 ``record`` 独立 append，使生命周期可短至
    一个 beacon 调用；所有异常都被吞掉，以维持 instrumentation 不影响 trial 的约束。
    """

    def __init__(self, workspace: Path | str):
        self._path = Path(workspace) / LEDGER_FILENAME

    def record(self, *, kind: str, source: str, detail: dict | None = None) -> None:
        """best-effort 追加一条 activation record。

        record 包含 Unix ``ts``、``kind``、``source`` 与 ``detail``；detail 缺失时写空 object。
        函数无返回值，写入失败静默忽略，因此调用成功不能证明记录已经持久化，后续 gate
        必须从 ledger 实际回读证据。
        """
        try:
            with open(self._path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "ts": time.time(),
                            "kind": kind,
                            "source": source,
                            "detail": detail or {},
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass


def activation_beacon(node_id: str, site: str = "", **detail: object) -> None:
    """为 evolved code path 记录一次 ``kind="beacon"`` activation。

    workspace 优先来自 ``ContextVar``，再读 ``CODEFLOW_ACTIVATION_WORKSPACE``；两者都不存在时
    no-op。``node_id`` 写入 source，非空 ``site`` 合并进 detail。该函数必须足够安全，不能
    因 instrumentation 异常改变 product Runtime。
    """
    workspace = _workspace_var.get() or os.environ.get(WORKSPACE_ENV)
    if not workspace:
        return
    d: dict = dict(detail)
    if site:
        d["site"] = site
    ActivationLedger(workspace).record(kind="beacon", source=node_id, detail=d)

    # ---- 逐任务收集（门控 b 回读侧）-------------------------------------------


BEACONS_DIRNAME = "beacons"
ENABLED_MARKER = ".enabled"


def beacon_workspace(out_dir: "Path | str", task_id: str, k: int) -> Path:
    """返回 eval out-dir 下 canonical per-attempt beacon workspace。

    batch runner 为每个 task-attempt subprocess 把 ``WORKSPACE_ENV`` 指向这里，使 beacon line
    按 task 预先分区；``read_fired_tasks`` 按同一 layout glob。writer 和 reader 必须共同使用
    本函数，路径形状为 ``beacons/<task_id>_k<k>``。
    """
    return Path(out_dir) / BEACONS_DIRNAME / f"{task_id}_k{k}"


def mark_beacons_enabled(out_dir: "Path | str") -> None:
    """写入 beacon collection marker，区分 honest zero 与 no data。

    marker 存在且无 ledger 表示 ``instrumentation ran, nothing fired``；marker 缺失表示
    ``collection was never wired``。目录创建或 touch 失败会被吞掉，因此 gate 必须按缺失
    instrumentation 处理，不能伪造零值。
    """
    root = Path(out_dir) / BEACONS_DIRNAME
    try:
        root.mkdir(parents=True, exist_ok=True)
        (root / ENABLED_MARKER).touch()
    except OSError:
        pass


def read_fired_tasks(out_dirs: "list[Path | str]", task_ids: "list[str]") -> "set[str] | None":
    """回读哪些 ``task_ids`` 在任一 eval out-dir 下至少有一行 beacon。

    ``out_dirs`` 通常包含 confirm dir 与 infra-rerun ladder sibling。若所有目录都没有
    collection marker，返回 ``None``：no instrumentation data 要求 Gate-b fail OPEN、跳过
    attribution，而不是拒绝全部 candidate。只要任一 marker 存在，返回 set；空 set 是 honest
    ``the mechanism never fired anywhere``，Gate-b 应正确地不给任何 task 归因。

    判定依据是对应 ledger file 存在且 size > 0，不解析内容语义；I/O error 被跳过。
    """
    roots = [Path(d) / BEACONS_DIRNAME for d in out_dirs]
    if not any((r / ENABLED_MARKER).exists() for r in roots):
        return None
    fired: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for tid in task_ids:
            if tid in fired:
                continue
            for ws in root.glob(f"{tid}_k*"):
                lf = ws / LEDGER_FILENAME
                try:
                    if lf.is_file() and lf.stat().st_size > 0:
                        fired.add(tid)
                        break
                except OSError:
                    continue
    return fired
