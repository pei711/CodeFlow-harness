from __future__ import annotations

from pathlib import Path

from benchmarks.codeflowbench.context_threshold_ablation import _p95
from benchmarks.codeflowbench.packs.context import ContextPack, ContextTrack, load_context_tasks
from benchmarks.codeflowbench.packs.context.runner import _pad_history_to_pressure, _trial_configs
from codeflow.config.codeflow import CodeFlowConfig
from codeflow.config.schema import Config
from codeflow.utils.helpers import estimate_message_tokens


def test_threshold_pack_changes_only_compact_threshold() -> None:
    tasks = (*load_context_tasks(ContextTrack.FORMAL), *load_context_tasks(ContextTrack.CALIBRATION))
    definition = ContextPack(tasks=tasks, compact_thresholds=(0.50, 0.80, 0.93)).definition()

    assert len(definition.tasks) == 12
    assert [variant.settings["structured_compact_threshold"] for variant in definition.variants] == [
        0.50,
        0.80,
        0.93,
    ]
    assert all(variant.settings["history_manager"] == "codeflow_history" for variant in definition.variants)
    assert {variant.settings["context_budget_ratio"] for variant in definition.variants} == {0.80}
    assert len(definition.pairs) == 2
    assert definition.identity["context_budget_ratio"] == 0.80


def test_pressure_padding_is_deterministic_and_keeps_source_history() -> None:
    original = [
        {"role": "user", "content": "Keep the exact early constraint."},
        {"role": "assistant", "content": "Acknowledged."},
        {"role": "user", "content": "Use the latest decision."},
        {"role": "assistant", "content": "Ready."},
    ]
    first = _pad_history_to_pressure(original, history_budget=2_000, target_pressure=0.80)
    second = _pad_history_to_pressure(original, history_budget=2_000, target_pressure=0.80)
    tokens = sum(estimate_message_tokens(message) for message in first)

    assert first == second
    assert original == [
        {"role": "user", "content": "Keep the exact early constraint."},
        {"role": "assistant", "content": "Acknowledged."},
        {"role": "user", "content": "Use the latest decision."},
        {"role": "assistant", "content": "Ready."},
    ]
    assert 0.79 <= tokens / 2_000 <= 0.80
    assert "Keep the exact early constraint." in "\n".join(str(item.get("content")) for item in first)


def test_trial_config_keeps_effective_budget_fixed_and_sets_compact_threshold(tmp_path: Path) -> None:
    base = Config()
    codeflow = CodeFlowConfig(base=base)

    _, low = _trial_configs(
        base,
        codeflow,
        workspace=tmp_path / "low",
        model="deepseek/deepseek-v4-flash",
        context_budget_ratio=0.80,
        structured_compact_threshold=0.50,
    )
    _, high = _trial_configs(
        base,
        codeflow,
        workspace=tmp_path / "high",
        model="deepseek/deepseek-v4-flash",
        context_budget_ratio=0.80,
        structured_compact_threshold=0.93,
    )

    assert low.context.context_budget_ratio == high.context.context_budget_ratio == 0.80
    assert low.context.structured_compact_threshold == 0.50
    assert high.context.structured_compact_threshold == 0.93


def test_p95_uses_nearest_rank() -> None:
    assert _p95([5.0, 1.0, 3.0, 4.0, 2.0]) == 5.0
    assert _p95([]) is None
