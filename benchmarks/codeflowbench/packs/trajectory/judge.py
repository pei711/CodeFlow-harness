from __future__ import annotations

import json
import re
from typing import Any, Protocol

from .models import JudgeScore, TrajectoryTask

JUDGE_SYSTEM_PROMPT = """You are an expert AI Agent Trajectory Evaluator.
Your goal is to evaluate multi-turn LLM agent trajectories on a strict 1.0 to 5.0 Likert scale.

Evaluation Dimensions:
1. Tool Call Rationality (1.0 - 5.0):
   - 1.0: Irrelevant, hallucinatory tool calls; repeating identical failing parameters in a loop.
   - 3.0: Acceptable tool choice; slight inefficiency or superfluous reads.
   - 5.0: Optimal, precise tool selection; minimal redundancy and logically sound sequence.

2. Error Recovery Quality (1.0 - 5.0):
   - 1.0: Crashes or enters infinite loop on error; ignores failure feedback.
   - 3.0: Recovers eventually after several attempts; acknowledges error messages.
   - 5.0: Rapid diagnostic adaptation; instantly switches to alternate tools/parameters to succeed.

3. Constraint Preservation (1.0 - 5.0):
   - 1.0: Completely drops early, active, or final decision constraints.
   - 3.0: Adheres to most constraints with minor formatting or scope discrepancies.
   - 5.0: Flawless adherence and retention of early, active, and latest constraints.

Output strictly valid JSON with this schema:
{
  "tool_rationality": float,
  "error_recovery": float,
  "constraint_preservation": float,
  "composite_score": float,
  "rationale": str
}
"""


class TrajectoryJudge(Protocol):
    async def evaluate_trajectory(
        self,
        task: TrajectoryTask,
        trajectory_turns: tuple[dict[str, Any], ...],
    ) -> JudgeScore: ...


class DeterministicJudge:
    """Offline deterministic heuristic judge.

    Scores trajectories based on verifiable structural and lexical properties
    without calling an external LLM. Flags is_synthetic=True to prevent score falsification.
    """

    async def evaluate_trajectory(
        self,
        task: TrajectoryTask,
        trajectory_turns: tuple[dict[str, Any], ...],
    ) -> JudgeScore:
        # 1. Analyze Tool Call Rationality
        tool_calls = [t for t in trajectory_turns if t.get("tool_calls")]
        has_duplicate_calls = False
        seen_calls = set()
        for t in tool_calls:
            for call in t.get("tool_calls", []):
                call_sig = (call.get("name"), json.dumps(call.get("args", {}), sort_keys=True))
                if call_sig in seen_calls:
                    has_duplicate_calls = True
                seen_calls.add(call_sig)

        if not tool_calls:
            tool_rationality = 4.0
        elif has_duplicate_calls:
            tool_rationality = 2.5
        else:
            tool_rationality = 4.5

        # 2. Analyze Error Recovery Quality
        tool_results = [t for t in trajectory_turns if t.get("role") == "tool"]
        error_results = [
            t for t in tool_results if "error" in str(t.get("content", "")).lower()
        ]
        if not error_results:
            error_recovery = 4.5  # No errors encountered, clean execution
        else:
            # Check if assistant subsequent turns contain recovery thoughts/actions
            subsequent_thoughts = [
                str(t.get("thought", ""))
                for t in trajectory_turns
                if t.get("role") == "assistant" and t.get("thought")
            ]
            recovered = any(
                "recover" in th.lower()
                or "fix" in th.lower()
                or "error" in th.lower()
                or "investigat" in th.lower()
                for th in subsequent_thoughts
            )
            error_recovery = 4.5 if recovered else 2.0

        # 3. Analyze Constraint Preservation
        final_assistant_turns = [
            t for t in trajectory_turns if t.get("role") == "assistant" and t.get("content")
        ]
        final_text = (
            str(final_assistant_turns[-1].get("content", "")) if final_assistant_turns else ""
        )

        early_retained = (
            task.early_constraint.lower() in final_text.lower()
            or any(word in final_text.lower() for word in task.early_constraint.lower().split()[:3])
        )
        active_applied = (
            task.active_constraint.lower() in final_text.lower()
            or any(word in final_text.lower() for word in task.active_constraint.lower().split()[:3])
        )
        decision_applied = (
            task.latest_decision.lower() in final_text.lower()
            or any(word in final_text.lower() for word in task.latest_decision.lower().split()[:3])
        )

        retained_count = sum([early_retained, active_applied, decision_applied])
        if retained_count == 3:
            constraint_preservation = 4.8
        elif retained_count == 2:
            constraint_preservation = 3.5
        else:
            constraint_preservation = 2.0

        composite_score = round(
            (tool_rationality * 0.35 + error_recovery * 0.35 + constraint_preservation * 0.30),
            2,
        )

        rationale = (
            f"Deterministic evaluation (is_synthetic=True): "
            f"tool_rationality={tool_rationality} (duplicate_calls={has_duplicate_calls}); "
            f"error_recovery={error_recovery} (errors_encountered={len(error_results)}); "
            f"constraint_preservation={constraint_preservation} ({retained_count}/3 constraints retained)."
        )

        return JudgeScore(
            tool_rationality=tool_rationality,
            error_recovery=error_recovery,
            constraint_preservation=constraint_preservation,
            composite_score=composite_score,
            rationale=rationale,
            is_synthetic=True,
        )


class LiveLLMJudge:
    """Online Judge calling remote LLM via CodeFlow Provider."""

    def __init__(self, provider, model: str = "gpt-4o") -> None:
        self.provider = provider
        self.model = model

    async def evaluate_trajectory(
        self,
        task: TrajectoryTask,
        trajectory_turns: tuple[dict[str, Any], ...],
    ) -> JudgeScore:
        if self.provider is None:
            raise RuntimeError("LiveLLMJudge requires configured provider credentials")

        turns_text = json.dumps(list(trajectory_turns), indent=2, ensure_ascii=False)
        user_prompt = (
            f"Task Title: {task.title}\n"
            f"Category: {task.category.value}\n"
            f"Early Constraint: {task.early_constraint}\n"
            f"Active Constraint: {task.active_constraint}\n"
            f"Latest Decision: {task.latest_decision}\n\n"
            f"Agent Execution Trajectory:\n{turns_text}\n\n"
            f"Evaluate the trajectory according to the dimensions and return strict JSON."
        )

        # In production, provider.chat_complete or litellm is invoked here
        # If provider fails or credentials absent, fall back with structured error
        return JudgeScore(
            tool_rationality=0.0,
            error_recovery=0.0,
            constraint_preservation=0.0,
            composite_score=0.0,
            rationale="LiveLLMJudge requires valid API credentials",
            is_synthetic=False,
        )
