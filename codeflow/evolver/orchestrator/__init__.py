"""在 seven-step funnel 之上运行 deterministic self-evolution Harness。

The canonical evolver package (``codeflow.evolver.{scheduler,analysis,judge,tree}``)
ships every deterministic operator the self-evolution SOP needs — anchor
selection, stability bucketing, the tree-aware bandit, the failure-map
builder, the node ledger, git ops, and the LLM judge. What it never had is a
faithful *driver* of the SOP's seven-step funnel.

This package supplies that missing layer. It is a small finite-state machine
that owns the control flow the SOP used to delegate to a long, high-compliance
Claude session:

- the round loop and per-candidate fork,
- the "never stop early" discipline (a code counter, not a prompt),
- the termination conditions (10 rounds with no vanilla-beating candidate,
  or a hard cap of 20 rounds),
- and state persistence so no model has to hold cross-round context.

The driver model (diagnose / design / verdict) only ever makes small,
schema-validated calls through the existing judge backends
(``codeflow.evolver.judge.llm_client``), which already route self-hosted Qwen,
Claude, and OpenRouter models. That is what lets a weaker model drive the loop:
Harness 承担 model 无法可靠记忆的 control-flow burden。Orchestrator 只根据 train artifacts 推进
round；sealed test 属于终止后的独立边界，中间状态不得推断为最终正向结论。
"""

from __future__ import annotations
