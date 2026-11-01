"""15 preregistered paired real-runtime context trials; no outcome-based retries."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from context_study_evidence import diagnostics, identity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.codeflowbench.budget import (
    BudgetGuardedProvider,
    ProviderBudgetConfig,
    ProviderBudgetLedger,
    provider_call_budget_scope,
)
from benchmarks.codeflowbench.canonical import to_primitive
from benchmarks.codeflowbench.packs.context.history import context_engine_factory_for
from benchmarks.codeflowbench.packs.context.models import ContextTrack
from benchmarks.codeflowbench.packs.context.runner import RuntimeContextTrialRunner
from benchmarks.codeflowbench.packs.context.tasks import load_context_tasks
from benchmarks.codeflowbench.protocol import TrialContext
from benchmarks.codeflowbench.records import TrialKey
from benchmarks.codeflowbench.schema import ExperimentSpec, VariantSpec
from codeflow.cli._helpers import make_provider
from codeflow.config.codeflow import load_codeflow_config
from codeflow.config.loader import load_config

SCENARIOS = {
    "long_session": {"messages": 60, "pressure": 0.72, "fault": False},
    "cross_turn_dependency": {"messages": 44, "pressure": 0.97, "fault": False},
    "compression_degradation": {"messages": 44, "pressure": 0.97, "fault": True},
}
ARMS = ("baseline", "governed")
SUMMARY_OUTPUT_TOKENS = 2048


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def utcnow():
    return datetime.now(timezone.utc).isoformat()


class SummaryUnavailable:
    """Fail before the actual summary request; ordinary Agent calls remain live."""

    def __init__(self, delegate):
        self.delegate = delegate

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    async def chat_with_retry(self, **kwargs):
        raise RuntimeError("Injected context-summary service unavailable before dispatch")


def degraded_factory(**kwargs):
    kwargs["provider"] = SummaryUnavailable(kwargs["provider"])
    return context_engine_factory_for("codeflow_history")(**kwargs)


def summarize(rows):
    def group(selected):
        output = {}
        for arm in ARMS:
            cells = [row for row in selected if row["arm"] == arm]
            complete = bool(cells) and all(row.get("usage_complete") is True for row in cells)

            def total(key):
                return (
                    sum(row[key] for row in cells)
                    if complete and all(isinstance(row.get(key), int) for row in cells)
                    else None
                )

            output[arm] = {
                "n": len(cells),
                "planned_n": len(selected) // 2,
                "verifier_passed": sum(row["verifier_state"] == "passed" for row in cells),
                "task_passed": sum(row["status"] == "passed" for row in cells),
                "usage_complete_n": sum(row.get("usage_complete") is True for row in cells),
                "input_tokens": total("trial_total_input_tokens"),
                "output_tokens": total("trial_total_output_tokens"),
                "auxiliary_input_tokens": total("context_auxiliary_input_tokens"),
                "mean_latency_ms": statistics.mean(row["wall_latency_ms"] for row in cells) if cells else None,
                "compact_count": sum(row.get("compact_count", 0) for row in cells),
                "fallback_count": sum(row.get("summary_fallback_count", 0) for row in cells),
            }
        left, right = output["baseline"]["input_tokens"], output["governed"]["input_tokens"]
        output["input_token_reduction_pct"] = (1 - right / left) * 100 if left and right is not None else None
        pairs = {}
        for row in selected:
            pairs.setdefault(row["pair_id"], {})[row["arm"]] = row
        both_pass = [
            pair
            for pair in pairs.values()
            if all(
                a in pair and pair[a]["verifier_state"] == "passed" and pair[a].get("usage_complete") is True
                for a in ARMS
            )
        ]
        bp_input = {a: sum(pair[a]["trial_total_input_tokens"] for pair in both_pass) for a in ARMS}
        output["both_pass_pairs"] = len(both_pass)
        output["both_pass_input_tokens"] = bp_input
        output["both_pass_input_reduction_pct"] = (
            (1 - bp_input["governed"] / bp_input["baseline"]) * 100 if bp_input["baseline"] else None
        )
        return output

    return {"overall": group(rows), "scenarios": {s: group([r for r in rows if r["scenario"] == s]) for s in SCENARIOS}}


def reports(root, rows):
    result = summarize(rows)
    save(root / "summary.json", result)
    fields = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (dict, list))})
    with (root / "raw_results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# CodeFlow 上下文管理成对实测",
        "",
        "每类 5 个任务对；共 15 对、30 次真实 Runtime 执行。",
        "",
        "| 场景 | 对照 Verifier | 治理 Verifier | 对照输入 Token | 治理输入 Token | 输入减少 | 双通过任务对 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, g in [*result["scenarios"].items(), ("overall", result["overall"])]:
        a, b = g["baseline"], g["governed"]
        reduction = g["input_token_reduction_pct"]
        percent = f"{reduction:.2f}%" if reduction is not None else "证据不完整"
        lines.append(
            f"| {name} | {a['verifier_passed']}/{a['n']} | {b['verifier_passed']}/{b['n']} | {a['input_tokens']} | {b['input_tokens']} | {percent} | {g['both_pass_pairs']} |"
        )
    lines.extend(
        [
            "",
            "口径与边界：",
            "",
            "- baseline 保留全部历史；governed 使用当前 CodeFlowHistorySegmentBuilder。其余模型、工具、预算和历史固定，AB/BA 交替顺序。",
            "- 长会话：60 条预置消息，目标历史压力 72%；跨轮次：44 条预置消息，目标压力 97%，早期约束与新旧决策共同决定产物。",
            "- 压缩降级：44 条预置消息，目标压力 97%，摘要服务在发出网络请求前故障注入，实际 Agent 调用仍访问真实模型。",
            "- 预置历史是确定性合成的有效对话，并非把 60/44 个历史轮次全部现场重新生成。",
            "- 输入 Token 为实际 Provider usage，包含主 Agent 与真实摘要请求；未发出的故障注入请求不虚构 Token。",
            f"- 主 Agent 输出仍为每次 500 Token；摘要输出独立为每次 {SUMMARY_OUTPUT_TOKENS} Token。两组其余工具与窗口设置一致。",
            "- 本轮向两组公开完全相同的 JSON 对象字段契约（additionalProperties=false），不提供预期答案值。Verifier 与预期文件保持不变；与旧轮次相比，任务输出契约更明确。",
            "- 全样本输入减少不等于保持质量的收益；另报双方 Verifier 均通过任务对的 Token 减少。失败和超时保留。",
            "- 每个任务/条件一次，无结果筛选和质量重试；这是探索性小样本评测，不建立统计显著性或生产可靠性结论。",
            "- 耗时为实测 trial wall time（含 Runtime 初始化/结束），不包含 Langfuse 上传；远程调度、缓存和执行顺序会影响耗时。",
            "- 独立 verifier 检查完整 JSON、禁止文件和预期文件 seal；预期答案不注入 Agent 工作区。",
            "",
        ]
    )
    for row in rows:
        if row["status"] != "passed":
            detail = diagnostics(row)
            lines.append(
                f"- 失败：{row['pair_id']} / {row['arm']}；{row['status']}；{detail['failure_categories']}；{detail['findings']}"
            )
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result


async def execute(args):
    from loguru import logger

    logger.disable("codeflow")
    config = load_config()
    cf = load_codeflow_config()
    cf.context.compact_max_output_tokens = SUMMARY_OUTPUT_TOKENS
    config.agents.defaults.temperature = 0
    model = config.agents.defaults.model
    if config.get_provider_name(model) != "deepseek" or model.rsplit("/", 1)[-1] != "deepseek-v4-flash":
        raise ValueError("This study requires the configured deepseek-v4-flash model")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_id = "c-" + root.name
    tasks = []
    base_tasks = load_context_tasks(ContextTrack.FORMAL)[:5]
    for scenario, settings in SCENARIOS.items():
        for index, base in enumerate(base_tasks, 1):
            task = replace(base, task_id=f"{scenario}-{index}", message_count=settings["messages"])
            tasks.append((scenario, index, task))
    relevant = [
        *ROOT.glob("codeflow/context_engine/**/*.py"),
        *ROOT.glob("benchmarks/codeflowbench/packs/context/*.py"),
        *ROOT.glob("codeflow/templates/**/*.md"),
        ROOT / "benchmarks/codeflowbench/verifier.py",
        ROOT / "benchmarks/codeflowbench/canonical.py",
        ROOT / "scripts/context_study_evidence.py",
        Path(__file__),
    ]
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(relevant)}
    manifest = dict(
        schema="codeflow.context-paired-study.v1",
        created_at=utcnow(),
        run_id=run_id,
        model=model,
        scenarios=SCENARIOS,
        task_pairs=15,
        planned_trials=30,
        output_contract_version="public-json-exact-keys-and-enums-v2",
        variants={"baseline": "full_history", "governed": "codeflow_history"},
        context_window_tokens=8192,
        context_budget_ratio=0.8,
        structured_compact_threshold=0.95,
        max_calls_per_trial=8,
        max_attempts_per_call=1,
        timeout_seconds=120,
        max_output_tokens_per_call=SUMMARY_OUTPUT_TOKENS,
        main_agent_output_tokens=500,
        summary_output_tokens=SUMMARY_OUTPUT_TOKENS,
        priced_budget_cap_cny=12,
        runtime_config={
            "context": cf.context.model_dump(mode="json"),
            "agent": {
                k: getattr(config.agents.defaults, k)
                for k in (
                    "model",
                    "temperature",
                    "max_tokens",
                    "max_tool_iterations",
                    "context_window_tokens",
                    "reasoning_effort",
                )
            },
        },
        source_hashes=source_hashes,
        commit=subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip(),
        working_tree_dirty=True,
        tasks=[
            dict(
                scenario=s,
                case=i,
                spec=to_primitive(t.to_task_spec()),
                history=t.materialize_history(),
                expected_digest=t.expected_digest,
                expected_output=json.loads(t.expected_path.read_text(encoding="utf-8")),
            )
            for s, i, t in tasks
        ],
    )
    manifest["versions"] = identity(manifest)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if identity(old)["experiment_version"] != manifest["versions"]["experiment_version"]:
            raise ValueError("Study source/model changed; use a fresh output directory")
    else:
        save(manifest_path, manifest)
    budget = ProviderBudgetConfig(
        hard_cap_cny=12,
        external_service_reserve_cny=0,
        max_total_request_attempts=240,
        max_input_tokens_per_call=42000,
        max_output_tokens_per_call=SUMMARY_OUTPUT_TOKENS,
        input_cache_miss_usd_per_million=0.14,
        output_usd_per_million=0.28,
        conservative_usd_to_cny_multiplier=7.5,
    )
    provider = BudgetGuardedProvider(
        make_provider(config), ledger=ProviderBudgetLedger(root / "provider-budget.jsonl", budget)
    )
    provider.generation = provider._delegate.generation
    runner = RuntimeContextTrialRunner(
        config=config,
        codeflow_config=cf,
        provider=provider,
        pressure_profile={t.task_id: SCENARIOS[s]["pressure"] for s, _, t in tasks},
    )
    spec = ExperimentSpec(
        suite="langfuse-context-paired",
        repetitions=1,
        pack_ids=("context-paired",),
        output_root=root,
        identity={"run_id": run_id},
    )
    rows = []
    for position, (scenario, case, task) in enumerate(tasks):
        order = ARMS if position % 2 == 0 else tuple(reversed(ARMS))
        for arm in order:
            pair_id = f"{scenario}-{case}"
            row_path = root / f"{pair_id}--{arm}.json"
            if row_path.is_file():
                rows.append(json.loads(row_path.read_text(encoding="utf-8")))
                continue
            variant = VariantSpec(
                variant_id=arm,
                settings={
                    "history_manager": "full_history" if arm == "baseline" else "codeflow_history",
                    "context_budget_ratio": 0.8,
                    "structured_compact_threshold": 0.95,
                },
            )
            context = TrialContext(
                experiment_id=run_id,
                plan_digest=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                key=TrialKey(run_id, "context-paired", task.task_id, arm, 0),
                block_attempt=1,
                experiment=spec,
                task=task.to_task_spec(),
                variant=variant,
            )
            factory = context_engine_factory_for(variant.settings["history_manager"])
            if arm == "governed" and SCENARIOS[scenario]["fault"]:
                factory = degraded_factory
            # The local tracing store caches its root; every sequential trial gets a fresh root.
            import codeflow.tracing.spans as trace_spans

            trace_spans._store = None
            started_at = utcnow()
            started = time.perf_counter()
            print(f"START {len(rows) + 1}/30 {pair_id} {arm}", flush=True)
            try:
                with provider_call_budget_scope(
                    trial_id=pair_id + ":" + arm,
                    max_logical_calls=8,
                    max_attempts_per_call=1,
                    max_input_tokens_per_call=42000,
                    max_output_tokens_per_call=SUMMARY_OUTPUT_TOKENS,
                ):
                    execution = await asyncio.wait_for(
                        runner(context=context, task=task, context_engine_factory=factory), timeout=120
                    )
                row = dict(
                    pair_id=pair_id,
                    scenario=scenario,
                    case=case,
                    arm=arm,
                    status=execution.status.value,
                    verifier_state=execution.verification.state.value,
                    findings=list(execution.findings),
                    verifier_findings=list(execution.verification.findings),
                    artifact_refs=list(execution.artifact_refs),
                    **dict(execution.metrics),
                )
            except Exception as exc:
                row = dict(
                    pair_id=pair_id,
                    scenario=scenario,
                    case=case,
                    arm=arm,
                    status="task_timeout" if isinstance(exc, TimeoutError) else "infrastructure_failure",
                    verifier_state="not_run",
                    usage_complete=False,
                    findings=[type(exc).__name__ + ": " + str(exc)],
                )
            row.update(
                started_at=started_at,
                finished_at=utcnow(),
                wall_latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            save(row_path, row)
            rows.append(row)
            reports(root, rows)
            print(
                f"DONE {len(rows)}/30 {pair_id} {arm} verifier={row['verifier_state']} input={row.get('trial_total_input_tokens')} wall={row['wall_latency_ms'] / 1000:.2f}s",
                flush=True,
            )
    summary = reports(root, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".codeflow" / "lc" / uuid.uuid4().hex[:8])
    args = parser.parse_args()
    os.environ["CODEFLOW_TRACING"] = "1"
    asyncio.run(execute(args))
