from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import statistics
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.codeflowbench.artifacts import ArtifactStore
from benchmarks.codeflowbench.budget import (
    BudgetGuardedProvider,
    ProviderBudgetConfig,
    ProviderBudgetLedger,
    provider_call_budget_scope,
)
from benchmarks.codeflowbench.canonical import canonical_digest
from benchmarks.codeflowbench.harness import run
from benchmarks.codeflowbench.packs.context import (
    CONTEXT_BENCHMARK_OUTPUT_TOKENS,
    CONTEXT_BENCHMARK_WINDOW_TOKENS,
    ContextPack,
    ContextTrack,
    RuntimeContextTrialRunner,
    load_context_tasks,
)
from benchmarks.codeflowbench.protocol import TrialKey
from benchmarks.codeflowbench.registry import PackRegistry
from benchmarks.codeflowbench.report import rebuild_full_report
from benchmarks.codeflowbench.schema import (
    ExecutionPolicy,
    ExperimentRef,
    ExperimentSpec,
    ProviderTrialBudget,
)
from codeflow.cli._helpers import make_provider
from codeflow.config.codeflow import load_codeflow_config
from codeflow.config.loader import load_config

THRESHOLDS = (0.50, 0.80, 0.93)
PRESSURE_BANDS = (0.48, 0.58, 0.68, 0.76, 0.82, 0.88, 0.92, 0.97, 0.62, 0.72, 0.85, 0.95)
REPETITIONS = 3
MAX_COST_CNY = 40.32
INPUT_PRICE_USD_PER_MILLION = 0.14
OUTPUT_PRICE_USD_PER_MILLION = 0.28
USD_TO_CNY = 7.5
MAX_CALLS_PER_TRIAL = 8
# The budget guard estimates the serialized provider payload, which can be
# larger than the model's tokenized prompt. Leave room for JSON/tool overhead.
MAX_ATTEMPTS_PER_CALL = 1
MAX_INPUT_TOKENS_PER_CALL = 42_000
PACK_ID = "context-threshold-ablation"
RAW_FIELDS = (
    "experiment_id",
    "task_id",
    "scenario",
    "threshold",
    "repetition",
    "model",
    "provider_reported_models",
    "context_window_tokens",
    "effective_history_budget",
    "pressure_target",
    "history_pressure",
    "trigger_pressure",
    "agent_input_tokens",
    "agent_output_tokens",
    "summary_input_tokens",
    "summary_output_tokens",
    "total_input_tokens",
    "total_output_tokens",
    "total_tokens",
    "estimated_cost_cny",
    "normal_trim_count",
    "compact_count",
    "summary_model_call_count",
    "summary_fallback_count",
    "over_budget_count",
    "end_to_end_latency_ms",
    "summary_call_latency_ms",
    "agent_steps",
    "verifier_state",
    "task_status",
    "early_constraint_retained",
    "active_constraint_applied",
    "latest_decision_applied",
    "artifact_exact",
    "error_info",
)


def _source_digest() -> str:
    root = Path(__file__).resolve().parents[2]
    paths = (
        "codeflow/config/codeflow.py",
        "codeflow/context_engine/assembler.py",
        "codeflow/context_engine/segments/codeflow_history.py",
        "benchmarks/codeflowbench/packs/context/history.py",
        "benchmarks/codeflowbench/packs/context/pack.py",
        "benchmarks/codeflowbench/packs/context/runner.py",
        "benchmarks/codeflowbench/context_threshold_ablation.py",
    )
    digest = hashlib.sha256()
    for relative in paths:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


async def _preflight(provider: BudgetGuardedProvider, model: str) -> None:
    tool = {
        "type": "function",
        "function": {
            "name": "context_threshold_preflight",
            "description": "Return the fixed readiness marker for this benchmark preflight.",
            "parameters": {
                "type": "object",
                "properties": {"ready": {"type": "boolean"}},
                "required": ["ready"],
                "additionalProperties": False,
            },
        },
    }
    with provider_call_budget_scope(
        trial_id="context-threshold-ablation-preflight-v1",
        max_logical_calls=1,
        max_attempts_per_call=1,
        max_input_tokens_per_call=1_000,
        max_output_tokens_per_call=100,
    ):
        response = await asyncio.wait_for(
            provider.chat(
                messages=[
                    {"role": "system", "content": "Call the supplied tool with ready=true."},
                    {"role": "user", "content": "Run the context threshold experiment preflight."},
                ],
                tools=[tool],
                model=model,
                max_tokens=100,
                temperature=0,
                tool_choice="auto",
            ),
            timeout=30,
        )
    usage = response.usage or {}
    if response.finish_reason == "error" or response.error_classification is not None:
        raise RuntimeError("Provider preflight returned an API error")
    if not {"prompt_tokens", "completion_tokens", "total_tokens"}.issubset(usage):
        raise RuntimeError("Provider preflight returned incomplete usage")
    if not response.tool_calls:
        raise RuntimeError("Provider preflight did not confirm tool calling")


def _task_scenario(task_id: str) -> str:
    if "incident" in task_id or "runbook" in task_id or "alert" in task_id:
        return "long_session"
    if "release" in task_id or "migration" in task_id or "retention" in task_id:
        return "cross_turn_dependency"
    return "pressure_compact_degradation"


def _estimate_cost(input_tokens: Any, output_tokens: Any) -> float | None:
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return None
    usd = (
        input_tokens / 1_000_000 * INPUT_PRICE_USD_PER_MILLION
        + output_tokens / 1_000_000 * OUTPUT_PRICE_USD_PER_MILLION
    )
    return usd * USD_TO_CNY


def _read_trial(store: ArtifactStore, experiment_id: str, task_id: str, threshold: float, repetition: int):
    key = TrialKey(
        experiment_id=experiment_id,
        pack_id=PACK_ID,
        task_id=task_id,
        variant_id=f"compact-{round(threshold * 100)}",
        repetition=repetition,
    )
    return store.read_json(store.trial_path(key))


def _trial_row(
    experiment_id: str,
    task_id: str,
    threshold: float,
    repetition: int,
    trial: dict,
    model: str,
) -> dict[str, Any]:
    metrics = trial.get("metrics", {})
    verification = trial.get("verification", {})
    total_input = metrics.get("trial_total_input_tokens")
    total_output = metrics.get("trial_total_output_tokens")
    return {
        "experiment_id": experiment_id,
        "task_id": task_id,
        "scenario": _task_scenario(task_id),
        "threshold": threshold,
        "repetition": repetition + 1,
        "model": model,
        "provider_reported_models": json.dumps(metrics.get("provider_reported_models", []), ensure_ascii=False),
        "context_window_tokens": CONTEXT_BENCHMARK_WINDOW_TOKENS,
        "effective_history_budget": metrics.get("effective_history_budget"),
        "pressure_target": metrics.get("pressure_target"),
        "history_pressure": json.dumps(metrics.get("history_pressure"), ensure_ascii=False),
        "trigger_pressure": json.dumps(metrics.get("trigger_pressure"), ensure_ascii=False),
        "agent_input_tokens": metrics.get("main_agent_input_tokens"),
        "agent_output_tokens": metrics.get("main_agent_output_tokens"),
        "summary_input_tokens": metrics.get("context_auxiliary_input_tokens"),
        "summary_output_tokens": metrics.get("context_auxiliary_output_tokens"),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_input + total_output
        if isinstance(total_input, int) and isinstance(total_output, int)
        else None,
        "estimated_cost_cny": _estimate_cost(total_input, total_output),
        "normal_trim_count": metrics.get("normal_trim_count"),
        "compact_count": metrics.get("compact_count"),
        "summary_model_call_count": metrics.get("summary_model_call_count"),
        "summary_fallback_count": metrics.get("summary_fallback_count"),
        "over_budget_count": metrics.get("over_budget_count"),
        "end_to_end_latency_ms": metrics.get("end_to_end_latency_ms"),
        "summary_call_latency_ms": metrics.get("summary_call_latency_ms"),
        "agent_steps": metrics.get("agent_steps"),
        "verifier_state": verification.get("state"),
        "task_status": trial.get("status"),
        "early_constraint_retained": metrics.get("early_constraint_retained"),
        "active_constraint_applied": metrics.get("active_constraint_applied"),
        "latest_decision_applied": metrics.get("latest_decision_applied"),
        "artifact_exact": metrics.get("artifact_exact"),
        "error_info": " | ".join(str(item) for item in trial.get("findings", [])),
    }


def _rows_for(
    ref: ExperimentRef,
    tasks,
    *,
    model: str,
    repetitions: int,
    thresholds=THRESHOLDS,
) -> list[dict[str, Any]]:
    store = ArtifactStore(ref)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        for threshold in thresholds:
            for repetition in range(repetitions):
                try:
                    trial = _read_trial(store, ref.experiment_id, task.task_id, threshold, repetition)
                except (OSError, KeyError, ValueError) as exc:
                    rows.append(
                        {
                            **{field: None for field in RAW_FIELDS},
                            "experiment_id": ref.experiment_id,
                            "task_id": task.task_id,
                            "scenario": _task_scenario(task.task_id),
                            "threshold": threshold,
                            "repetition": repetition + 1,
                            "task_status": "missing_trial_record",
                            "verifier_state": "not_run",
                            "error_info": f"missing_trial_record:{type(exc).__name__}:{exc}",
                        }
                    )
                else:
                    rows.append(_trial_row(ref.experiment_id, task.task_id, threshold, repetition, trial, model))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _group_summary(rows: list[dict[str, Any]]) -> dict[float, dict[str, Any]]:
    summary: dict[float, dict[str, Any]] = {}
    for threshold in THRESHOLDS:
        records = [row for row in rows if row.get("threshold") == threshold]

        def valid(metric: str) -> list[float]:
            return [v for row in records if (v := _number(row.get(metric))) is not None]

        costs = valid("estimated_cost_cny")
        verifier_pass = sum(row.get("verifier_state") == "passed" for row in records)
        summary[threshold] = {
            "n": len(records),
            "verifier_pass_rate": verifier_pass / len(records) if records else None,
            "task_success_rate": sum(row.get("task_status") == "passed" for row in records) / len(records)
            if records
            else None,
            "early_constraint_retention_rate": sum(row.get("early_constraint_retained") is True for row in records)
            / len(records)
            if records
            else None,
            "agent_input_tokens": sum(valid("agent_input_tokens")) if valid("agent_input_tokens") else None,
            "agent_output_tokens": sum(valid("agent_output_tokens")) if valid("agent_output_tokens") else None,
            "summary_input_tokens": sum(valid("summary_input_tokens")) if valid("summary_input_tokens") else None,
            "summary_output_tokens": sum(valid("summary_output_tokens")) if valid("summary_output_tokens") else None,
            "total_tokens": sum(valid("total_tokens")) if valid("total_tokens") else None,
            "compact_count": sum(valid("compact_count")) if valid("compact_count") else None,
            "normal_trim_count": sum(valid("normal_trim_count")) if valid("normal_trim_count") else None,
            "summary_model_call_count": sum(valid("summary_model_call_count"))
            if valid("summary_model_call_count")
            else None,
            "summary_fallback_count": sum(valid("summary_fallback_count")) if valid("summary_fallback_count") else None,
            "over_budget_count": sum(valid("over_budget_count")) if valid("over_budget_count") else None,
            "mean_latency_ms": statistics.mean(valid("end_to_end_latency_ms"))
            if valid("end_to_end_latency_ms")
            else None,
            "p50_latency_ms": statistics.median(valid("end_to_end_latency_ms"))
            if valid("end_to_end_latency_ms")
            else None,
            "p95_latency_ms": _p95(valid("end_to_end_latency_ms")),
            "summary_latency_ms": sum(valid("summary_call_latency_ms")) if valid("summary_call_latency_ms") else None,
            "agent_steps": sum(valid("agent_steps")) if valid("agent_steps") else None,
            "estimated_cost_cny": sum(costs) if costs else None,
            "failures": [
                {
                    "task_id": row.get("task_id"),
                    "repetition": row.get("repetition"),
                    "status": row.get("task_status"),
                    "error": row.get("error_info"),
                }
                for row in records
                if row.get("task_status") != "passed"
            ],
        }
    return summary


def _fmt(value: Any, *, percent: bool = False, digits: int = 2) -> str:
    if value is None:
        return "未测"
    return f"{value * 100:.{digits}f}%" if percent else f"{value:,.{digits}f}"


def _write_reports(output_dir: Path, rows: list[dict[str, Any]], *, experiment_id: str, model: str) -> None:
    groups = _group_summary(rows)
    columns = [groups[ratio] for ratio in THRESHOLDS]
    report_rows = [
        ("Verifier 通过率", "verifier_pass_rate", True),
        ("任务成功率", "task_success_rate", True),
        ("关键早期约束保留率", "early_constraint_retention_rate", True),
        ("Agent 输入 Token（总计）", "agent_input_tokens", False),
        ("Agent 输出 Token（总计）", "agent_output_tokens", False),
        ("摘要模型输入 Token（总计）", "summary_input_tokens", False),
        ("摘要模型输出 Token（总计）", "summary_output_tokens", False),
        ("总 Token（含摘要调用）", "total_tokens", False),
        ("普通裁剪次数", "normal_trim_count", False),
        ("Compact 次数", "compact_count", False),
        ("摘要模型调用次数", "summary_model_call_count", False),
        ("规则摘要兜底次数", "summary_fallback_count", False),
        ("超预算次数", "over_budget_count", False),
        ("端到端平均耗时 ms", "mean_latency_ms", False),
        ("端到端 P50 耗时 ms", "p50_latency_ms", False),
        ("端到端 P95 耗时 ms", "p95_latency_ms", False),
        ("摘要调用累计耗时 ms", "summary_latency_ms", False),
        ("Agent 工具执行步数", "agent_steps", False),
        ("API 估算成本 CNY", "estimated_cost_cny", False),
    ]
    lines = [
        "# Context Compact 阈值消融汇总",
        "",
        f"- Run ID：`{experiment_id}`",
        f"- 模型：`{model}`（Provider 返回的模型标识另见原始 CodeFlowBench trial artifact）",
        f"- 每组样本：{groups[0.50]['n']}；总样本：{len(rows)}",
        "- 预算比例固定为 80%；Compact 压力阈值为唯一处理变量。有效可用历史预算按 ContextAssembler 实际计算，输出预留与工具配置不变。",
        "- Token 为 Provider Usage；总 Token 含 Agent 与 Context Auxiliary（结构化摘要）调用。费用按实验脚本冻结单价估算，不等同账单。",
        "- P95 使用 nearest-rank：排序后取 `ceil(0.95 × n)` 项；P50 为中位数。缺失值不补零；成功率分母为全部计划 Trial。",
        "",
        "| 指标 | 50% | 80% | 93% |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key, is_percent in report_rows:
        lines.append(
            "| " + label + " | " + " | ".join(_fmt(group[key], percent=is_percent) for group in columns) + " |"
        )

    for comparator in (0.50, 0.93):
        baseline = groups[comparator]
        target = groups[0.80]
        lines.extend(["", f"## 80% 相对 {int(comparator * 100)}% 的差异", ""])
        for label, key, _ in report_rows:
            left, right = _number(target.get(key)), _number(baseline.get(key))
            if left is None or right in (None, 0):
                delta = "未测"
            else:
                delta = f"{(left - right) / abs(right) * 100:+.2f}%"
            lines.append(f"- {label}：{delta}")
    lines.extend(["", "## 异常和失败", ""])
    for threshold, group in groups.items():
        lines.append(f"- {int(threshold * 100)}%：{len(group['failures'])}/{group['n']} 条未通过/失败记录")
        for failure in group["failures"]:
            lines.append(
                f"  - `{failure['task_id']}` repeat {failure['repetition']}: "
                f"{failure['status']} — {failure['error'] or '无额外错误信息'}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    best = sorted(
        ((ratio, group) for ratio, group in groups.items()),
        key=lambda item: (
            item[1]["verifier_pass_rate"] if item[1]["verifier_pass_rate"] is not None else -1,
            -(item[1]["total_tokens"] or float("inf")),
        ),
        reverse=True,
    )
    lines = [
        "# Context Compact 阈值面试报告",
        "",
        f"本次实验运行 `{experiment_id}`，固定同一模型配置 `{model}`、8,192-token 窗口、输出预留、工具集和任务输入；每个阈值 36 次，三组共 108 次。阈值仅控制 CodeFlowHistory 的结构化 Compact；80% ContextAssembler 工作预算比例保持固定。",
        "",
        "## 结果解读",
        "",
        f"- 50% 是否过于保守：Compact 次数 {groups[0.50]['compact_count']}，Agent+摘要总 Token {groups[0.50]['total_tokens']}，平均耗时 {_fmt(groups[0.50]['mean_latency_ms'])} ms；依据这些实测值判断提前摘要的实际开销。",
        f"- 93% 是否偏晚：Compact 次数 {groups[0.93]['compact_count']}，超预算 {groups[0.93]['over_budget_count']}，Verifier 通过率 {_fmt(groups[0.93]['verifier_pass_rate'], percent=True)}，P95 {_fmt(groups[0.93]['p95_latency_ms'])} ms。只有观察到溢出/治理失败样本时才称为已验证风险。",
        f"- 80% 实测指标：Verifier {_fmt(groups[0.80]['verifier_pass_rate'], percent=True)}；总 Token {groups[0.80]['total_tokens']}；Compact {groups[0.80]['compact_count']}；平均/P95 {_fmt(groups[0.80]['mean_latency_ms'])}/{_fmt(groups[0.80]['p95_latency_ms'])} ms；成本 {_fmt(groups[0.80]['estimated_cost_cny'], digits=4)} CNY。",
        "- 80% 优势与无优势应以 `summary.md` 的配对任务原始记录为准；没有胜出的指标会明确写为无优势，不从总体历史 Token 降幅推断阈值效果。",
        f"- 本次三组按 Verifier 通过率排序的首位是 {int(best[0][0] * 100)}%（描述性排序，样本量有限，不是统计显著性结论）。",
        "",
        "## 局限",
        "",
        "- 每个条件 36 次（12 个任务 × 3 次重复）；远程模型不接受/不保证固定随机种子，temperature=0 仍不能消除服务端非确定性和模型别名漂移。",
        "- 12 个任务历史中加入确定性、可追溯的中性对话填充，将压力分布覆盖到 Compact 阈值附近；这能测治理时机和早期约束保留，但不代表真实用户对话的自然分布。",
        "- Verifier 是 sealed JSON artifact 精确校验，不评价开放式答案的语义质量。早期约束保留率基于最后一次模型输入窗口中是否出现原始约束文本。",
        "- 规则摘要兜底只在 Provider 摘要异常时观测；本试验不会人为篡改 Provider 结果来制造异常。异常恢复另用确定性单元测试覆盖。",
        "- 成本是根据 Provider token usage 和脚本冻结单价计算的估算值；平台缓存折扣、价格变更和账单差异可能改变实际费用。",
        "",
        "## 40–60 秒面试回答",
        "",
    ]
    lines.append(
        f"我没有先假设 80% 最好，而是把 CodeFlow 的结构化 Compact 阈值单独参数化，固定 ContextAssembler 的 80% 工作预算、模型、窗口、任务和工具配置，做了 50%、80%、93% 三组各 36 次的配对实验。三组 Verifier 通过率分别是 {_fmt(groups[0.50]['verifier_pass_rate'], percent=True)}、{_fmt(groups[0.80]['verifier_pass_rate'], percent=True)}、{_fmt(groups[0.93]['verifier_pass_rate'], percent=True)}；总 Token 分别是 {groups[0.50]['total_tokens']}、{groups[0.80]['total_tokens']}、{groups[0.93]['total_tokens']}，平均耗时分别是 {_fmt(groups[0.50]['mean_latency_ms'])}、{_fmt(groups[0.80]['mean_latency_ms'])}、{_fmt(groups[0.93]['mean_latency_ms'])} 毫秒。按这组数据，80% 在 [填入实际领先/未领先指标] 上 [有/没有]优势；由于只有 36 个任务样本/组且远程模型存在非确定性，我把结论限定为当前任务集上的测量结果，而不声称普遍最优。"
    )
    (output_dir / "interview_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prepare_budget(output_root: Path, *, approval_digest: str, max_attempts: int):
    ledger_path = output_root / "provider-budget.jsonl"
    base_config = ProviderBudgetConfig(
        hard_cap_cny=MAX_COST_CNY,
        external_service_reserve_cny=0,
        max_total_request_attempts=max_attempts,
        max_additional_request_attempts=0,
        max_input_tokens_per_call=MAX_INPUT_TOKENS_PER_CALL,
        max_output_tokens_per_call=CONTEXT_BENCHMARK_OUTPUT_TOKENS,
        input_cache_miss_usd_per_million=INPUT_PRICE_USD_PER_MILLION,
        output_usd_per_million=OUTPUT_PRICE_USD_PER_MILLION,
        conservative_usd_to_cny_multiplier=USD_TO_CNY,
    )
    prefix = ProviderBudgetLedger(ledger_path, base_config).snapshot()
    config = ProviderBudgetConfig(
        **{
            **base_config.__dict__,
            "approval_digest": approval_digest,
            "ledger_prefix_event_count": prefix.ledger_event_count,
            "ledger_prefix_digest": prefix.ledger_digest,
            "ledger_prefix_charged_cny": prefix.provider_charged_cny,
        }
    )
    return BudgetGuardedProvider(make_provider(load_config()), ledger=ProviderBudgetLedger(ledger_path, config))


async def _run_stage(
    *,
    output_root: Path,
    tasks,
    repetitions: int,
    provider: BudgetGuardedProvider,
    config,
    codeflow_config,
    pressure_profile: dict[str, float],
    run_id: str,
    label: str,
) -> ExperimentRef:
    runner = RuntimeContextTrialRunner(
        config=config,
        codeflow_config=codeflow_config,
        provider=provider,
        pressure_profile=pressure_profile,
    )
    pack = ContextPack(
        ContextTrack.FORMAL,
        tasks=tuple(tasks),
        compact_thresholds=THRESHOLDS,
        runner=runner,
    )
    definition = pack.definition()
    registry = PackRegistry()
    registry.register(pack)
    spec = ExperimentSpec(
        suite=f"context-threshold-ablation-{label}",
        repetitions=repetitions,
        pack_ids=(PACK_ID,),
        output_root=output_root,
        identity={
            "run_id": run_id,
            "stage": label,
            "provider": "deepseek",
            "model": config.agents.defaults.model,
            "temperature": 0,
            "seed": None,
            "context_window_tokens": CONTEXT_BENCHMARK_WINDOW_TOKENS,
            "output_tokens": CONTEXT_BENCHMARK_OUTPUT_TOKENS,
            "context_budget_ratio": 0.80,
            "compact_thresholds": list(THRESHOLDS),
            "pressure_profile": pressure_profile,
            "task_set_digest": definition.identity["task_set_digest"],
            "source_digest": _source_digest(),
            "budget_cap_cny": MAX_COST_CNY,
        },
        execution=ExecutionPolicy(
            retry_policy="symmetric",
            provider_call_max_attempts=MAX_ATTEMPTS_PER_CALL,
            max_comparison_block_attempts=1,
            max_comparison_block_retries_total=0,
            max_provider_calls_per_trial=MAX_CALLS_PER_TRIAL,
            provider_trial_budgets=(
                ProviderTrialBudget(
                    pack_id=PACK_ID,
                    max_provider_calls_per_trial=MAX_CALLS_PER_TRIAL,
                    max_input_tokens_per_call=MAX_INPUT_TOKENS_PER_CALL,
                    max_output_tokens_per_call=CONTEXT_BENCHMARK_OUTPUT_TOKENS,
                ),
            ),
        ),
    )
    return await run(spec, registry=registry)


async def run_ablation(output_dir: Path) -> dict[str, Any]:
    config = load_config()
    codeflow_config = load_codeflow_config()
    provider_name = config.get_provider_name(config.agents.defaults.model)
    model = config.agents.defaults.model
    if provider_name != "deepseek" or model.rsplit("/", 1)[-1] != "deepseek-v4-flash":
        raise RuntimeError("The threshold experiment requires the configured deepseek-v4-flash model")
    config.agents.defaults.temperature = 0
    tasks = (*load_context_tasks(ContextTrack.FORMAL), *load_context_tasks(ContextTrack.CALIBRATION))
    pressure_profile = {task.task_id: PRESSURE_BANDS[index % len(PRESSURE_BANDS)] for index, task in enumerate(tasks)}
    run_id = f"ct-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    evidence_root = Path(".codeflow/e") / uuid.uuid4().hex[:8]
    evidence_root.mkdir(parents=True, exist_ok=False)
    max_trials = 1 * len(THRESHOLDS) + len(tasks) * len(THRESHOLDS) * REPETITIONS
    max_attempts = 1 + max_trials * MAX_CALLS_PER_TRIAL * MAX_ATTEMPTS_PER_CALL
    approval_digest = canonical_digest(
        {
            "run_id": run_id,
            "task_count": len(tasks),
            "repetitions": REPETITIONS,
            "thresholds": list(THRESHOLDS),
            "hard_cap_cny": MAX_COST_CNY,
            "source_digest": _source_digest(),
        }
    )
    provider = _prepare_budget(evidence_root, approval_digest=approval_digest, max_attempts=max_attempts)
    provider.generation = provider._delegate.generation
    await _preflight(provider, model)

    # One real trial per arm, on the same fixed-pressure task, checks that the
    # threshold knob changes provider and compression behavior before 108 runs.
    pilot_task = next(task for task in tasks if task.task_id == "ctx-incident-handoff")
    pilot_profile = {pilot_task.task_id: 0.70}
    pilot_ref = await _run_stage(
        output_root=evidence_root,
        tasks=(pilot_task,),
        repetitions=1,
        provider=provider,
        config=config,
        codeflow_config=codeflow_config,
        pressure_profile=pilot_profile,
        run_id=run_id,
        label="pilot",
    )
    pilot_rows = _rows_for(pilot_ref, (pilot_task,), model=model, repetitions=1)
    _write_csv(evidence_root / "pilot.csv", pilot_rows)
    pilot_cost = sum(row["estimated_cost_cny"] or 0 for row in pilot_rows)
    pilot_total_tokens = sum(row["total_tokens"] or 0 for row in pilot_rows)
    pilot_summary = {
        "run_id": run_id,
        "experiment_id": pilot_ref.experiment_id,
        "trials": len(pilot_rows),
        "cost_cny": pilot_cost,
        "total_tokens": pilot_total_tokens,
        "compact_count_by_threshold": {
            str(int(threshold * 100)): next(
                (row["compact_count"] for row in pilot_rows if row["threshold"] == threshold),
                None,
            )
            for threshold in THRESHOLDS
        },
        "extrapolated_108_trial_cost_cny": pilot_cost / max(1, len(pilot_rows)) * 108,
        "hard_cap_cny": MAX_COST_CNY,
    }
    (evidence_root / "pilot-summary.json").write_text(
        json.dumps(pilot_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not all(isinstance(row.get("total_tokens"), int) for row in pilot_rows):
        raise RuntimeError("Pilot usage is incomplete; full matrix was not started")
    if len({row.get("compact_count") for row in pilot_rows}) < 2:
        raise RuntimeError("Pilot did not distinguish Compact behavior; full matrix was not started")
    if pilot_summary["extrapolated_108_trial_cost_cny"] >= MAX_COST_CNY:
        raise RuntimeError("Pilot cost projection reached the authorized hard cap; full matrix was not started")

    full_ref = await _run_stage(
        output_root=evidence_root,
        tasks=tasks,
        repetitions=REPETITIONS,
        provider=provider,
        config=config,
        codeflow_config=codeflow_config,
        pressure_profile=pressure_profile,
        run_id=run_id,
        label="formal",
    )
    rows = _rows_for(full_ref, tasks, model=model, repetitions=REPETITIONS)
    _write_csv(output_dir / "raw_results.csv", rows)
    report = rebuild_full_report(full_ref)
    _write_reports(output_dir, rows, experiment_id=full_ref.experiment_id, model=model)
    result = {
        "run_id": run_id,
        "experiment_id": full_ref.experiment_id,
        "planned_trials": len(tasks) * len(THRESHOLDS) * REPETITIONS,
        "raw_rows": len(rows),
        "codeflowbench_measurement_valid": report.measurement_valid,
        "report_digest": report.report_digest,
        "pilot": pilot_summary,
        "provider_budget": provider.ledger.snapshot().__dict__,
        "source_digest": _source_digest(),
        "task_ids": [task.task_id for task in tasks],
    }
    (evidence_root / "run-summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the real 50% / 80% / 93% Compact threshold ablation.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/context_threshold_ablation"),
    )
    args = parser.parse_args()
    result = asyncio.run(run_ablation(args.output_dir.resolve()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
