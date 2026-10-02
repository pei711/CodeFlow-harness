"""用一份 config 与一条命令启动任意 registered benchmark 的 unified layer。

``codeflow evolve run --config <yaml>`` 把完整 SOP 驱动为 resumable state machine：cold-start
thick ledger -> evolution rounds -> terminate -> unseal。任意位置 interruption 后，重复同一命令
会从最后一个 durable artifact（trial files、round journal、meta stamp）恢复。

Launch 只装配 benchmark 与 shared loop；run 返回成功不等于 candidate promoted，unseal 后也必须
按 sealed report 的 evidence boundary 解读。
"""

from codeflow.evolver.launch.contract import BenchBundle, LaunchContext, validate_whitelist
from codeflow.evolver.launch.registry import load_bench

__all__ = ["BenchBundle", "LaunchContext", "load_bench", "validate_whitelist"]
