"""把 ``never stop early`` 的 Loop termination discipline 固化为代码。

The SOP stops on the first of these conditions, and the exhaustion signal is
always measured against VANILLA (the fixed cold-start baseline) on train, never
against the previous parent and never against the sealed test set:

- ``patience`` consecutive rounds in which no candidate beat vanilla on train
  (exploration exhausted — the primary signal), or
- ``max_rounds`` reached (a hard cap backstop), or
- ``max_consecutive_errors`` rounds in a row that produced no real decision
  (failed or inconclusive evidence). A no-decision round is NOT evidence about
  exploration, so it must not burn patience — but an endless outage must not
  loop either, so it gets its own counter and an honest ``errors_exhausted``
  stop reason.

``record_round(promoted=...)`` 接收 vanilla-comparison signal：本轮至少一个 candidate full-train
confirm beat vanilla 才为 True，不管它是否同时 beat ratcheted parent 并 bank。small unit-tested
tracker 让 weak driver 也能运行 Loop；stop decision 属于 Harness，不依赖 model 记忆。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TerminationTracker:
    """跨 round 跟踪 promotion/no-decision counter，并决定是否停止。

    tracker 是进程内状态，resume 时应由 journal replay 恢复；它不读取 sealed test。
    """

    patience: int = 10
    max_rounds: int = 20
    max_consecutive_errors: int = 5
    rounds_completed: int = 0
    consecutive_no_promotion: int = 0
    consecutive_errors: int = 0

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("patience must be >= 1")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        if self.max_consecutive_errors < 1:
            raise ValueError("max_consecutive_errors must be >= 1")

    def record_round(self, promoted: bool, *, errored: bool = False) -> None:
        """记录一个 completed round outcome。

        ``promoted`` 是 SOP exhaustion signal：至少一个 candidate full-train beat VANILLA。
        ``errored`` 表示全部 candidate failed/inconclusive、无真实 decision；它增加 round/error
        counter，但保持 patience untouched。正常 decision 会 reset consecutive_errors。
        """
        self.rounds_completed += 1
        if errored:
            self.consecutive_errors += 1
            return
        self.consecutive_errors = 0
        if promoted:
            self.consecutive_no_promotion = 0
        else:
            self.consecutive_no_promotion += 1

    def should_stop(self) -> tuple[bool, str | None]:
        """根据已记录 round 返回 ``(stop, reason)``。

        continue 时 reason 为 ``None``。检查顺序为 max_rounds、errors_exhausted、
        patience_exhausted，因此同轮同时到 cap/patience 时报告 hard cap。返回 stop 不是 run
        finalize/unseal 已完成的证明。
        """
        if self.rounds_completed >= self.max_rounds:
            return True, "max_rounds"
        if self.consecutive_errors >= self.max_consecutive_errors:
            return True, "errors_exhausted"
        if self.consecutive_no_promotion >= self.patience:
            return True, "patience_exhausted"
        return False, None


__all__ = ["TerminationTracker"]
