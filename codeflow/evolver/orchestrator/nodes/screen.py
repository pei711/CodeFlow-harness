"""执行 Step ⑤a：使用 wide-pass verdict 的 K=1 anchor screen。

The screen is deliberately wide (wide-pass): a candidate advances to the full-set
confirm unless its anchor-mean pass@1 falls *clearly* below vanilla's. The SOP
describes three buckets, but operationally there is a single cut:

- clear win        (margin >= +cull_threshold)  -> confirm
- within the band  (|margin| <  cull_threshold)  -> confirm  (a slightly-low
                                                    anchor mean is not the
                                                    full-set mean; don't cull)
- clear loss       (margin <= -cull_threshold)  -> pruned_at_screen

``cull_threshold`` is ``cull_sigma_mult * sigma_screen`` from ``select_anchor``,
i.e. sized off the *same* vanilla thick ledger the anchor was drawn from, so the
band reflects genuine K=1 anchor-mean sampling noise rather than a guess.

Vanilla's anchor mean is read from the cold-start thick-ledger per-task rates
(K=3, the fixed baseline the funnel always compares against); the candidate's is
the K=1 screen run. Comparing a noisy K=1 mean against a tighter K=3 mean is
正是 band 必须宽的原因。screen pass 只允许进入 full confirm，不是 promotion evidence。
"""

from __future__ import annotations

from dataclasses import dataclass

from codeflow.evolver.orchestrator.scoring import (
    TaskEval,
    anchor_mean_pass_rate,
)
from codeflow.evolver.scheduler.anchor_selection import AnchorSelection


@dataclass(frozen=True)
class ScreenResult:
    """一个 candidate anchor screen 的 means、noise band、bucket 与 advancement verdict。"""

    candidate_mean: float
    vanilla_mean: float
    cull_threshold: float
    sigma_screen: float
    passes_to_confirm: bool
    bucket: str  # "clear_win" | "within_band" | "cull"，用于日志/账本

    @property
    def margin(self) -> float:
        """返回 candidate mean 减 vanilla mean，以 fraction 表示 pp margin。"""
        return self.candidate_mean - self.vanilla_mean


def screen_candidate(
    *,
    candidate_evals: dict[str, TaskEval],
    anchor: AnchorSelection,
    vanilla_evals: dict[str, TaskEval],
) -> ScreenResult:
    """对 candidate K=1 anchor eval 应用 wide-pass cut。

    clear win/within band 进入 confirm，只有 clear loss 才 cull。
    """
    candidate_mean = anchor_mean_pass_rate(candidate_evals, anchor.task_ids)
    vanilla_mean = anchor_mean_pass_rate(vanilla_evals, anchor.task_ids)
    margin = candidate_mean - vanilla_mean
    cull = anchor.cull_threshold

    if margin >= cull:
        bucket = "clear_win"
    elif margin > -cull:
        bucket = "within_band"
    else:
        bucket = "cull"

    return ScreenResult(
        candidate_mean=candidate_mean,
        vanilla_mean=vanilla_mean,
        cull_threshold=cull,
        sigma_screen=anchor.sigma_screen,
        passes_to_confirm=(bucket != "cull"),
        bucket=bucket,
    )


__all__ = ["ScreenResult", "screen_candidate"]
