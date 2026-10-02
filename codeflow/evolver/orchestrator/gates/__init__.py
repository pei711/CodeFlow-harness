"""实现 post-confirm gates（SOP §2 ⑥）：infra health、activation 与 paired lift。

Also home to the pluggable decision policy (``policy``/``strategies``) and the
focused-Fisher statistics (``fisher``) that let the two benchmark lines share
one round loop。Package 只汇总 gate/policy public surface；任一单 gate 成功都不等于 candidate
已 promote 或 sealed generalisation 成立。
"""

from __future__ import annotations

from codeflow.evolver.orchestrator.gates.policy import (
    Baseline,
    BaselineProvider,
    CandidateOutcome,
    DecisionContext,
    FrozenColdStartBaseline,
    GatePolicy,
    PerParentFrozenBaseline,
    SameSessionPairedBaseline,
)
from codeflow.evolver.orchestrator.gates.strategies import (
    FocusedFisherGate,
    PairedTwoSigmaGate,
    confirm_job_name,
)

__all__ = [
    "Baseline",
    "BaselineProvider",
    "CandidateOutcome",
    "DecisionContext",
    "GatePolicy",
    "FrozenColdStartBaseline",
    "PerParentFrozenBaseline",
    "SameSessionPairedBaseline",
    "FocusedFisherGate",
    "PairedTwoSigmaGate",
    "confirm_job_name",
]
