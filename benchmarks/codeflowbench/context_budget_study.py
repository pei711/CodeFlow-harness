from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
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
    load_context_budget_study_tasks,
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

RATIOS = (0.50, 0.80, 0.93)
MAX_COST_CNY = 11.2
INPUT_PRICE_USD_PER_MILLION = 0.14
OUTPUT_PRICE_USD_PER_MILLION = 0.28
USD_TO_CNY = 7.5
STUDY_PACK_ID = "context-budget-study"
MAX_CALLS_PER_TRIAL = 8
MAX_ATTEMPTS_PER_CALL = 2


def _source_digest() -> str:
    root = Path(__file__).resolve().parents[2]
    paths = (
        "codeflow/config/codeflow.py",
        "codeflow/context_engine/assembler.py",
        "codeflow/context_engine/factory.py",
        "benchmarks/codeflowbench/packs/context/pack.py",
        "benchmarks/codeflowbench/packs/context/runner.py",
        "benchmarks/codeflowbench/packs/context/history.py",
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
            "name": "context_budget_preflight",
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
        trial_id="context-budget-study-preflight-3",
        max_logical_calls=1,
        max_attempts_per_call=1,
        max_input_tokens_per_call=1_000,
        max_output_tokens_per_call=100,
    ):
        response = await asyncio.wait_for(
            provider.chat(
                messages=[
                    {"role": "system", "content": "Call the supplied tool with ready=true."},
                    {"role": "user", "content": "Run the benchmark readiness check."},
                ],
                tools=[tool],
                model=model,
                max_tokens=100,
                temperature=0,
                tool_choice="auto",
            ),
            timeout=30,
        )
    if response.finish_reason == "error" or response.error_classification is not None:
        category = response.error_classification.category if response.error_classification is not None else "unknown"
        detail = (response.content or "").replace("\n", " ")[:300]
        raise RuntimeError(
            f"provider preflight failed ({category}): {detail or 'no provider detail'}",
        )
    usage = response.usage or {}
    required_usage = {"prompt_tokens", "completion_tokens", "total_tokens"}
    if not required_usage.issubset(usage):
        raise RuntimeError("provider preflight returned incomplete token usage")
    if not response.tool_calls:
        raise RuntimeError("provider preflight did not confirm tool calling")


def _read_trial(store: ArtifactStore, task_id: str, variant_id: str) -> dict[str, Any]:
    key = TrialKey(
        experiment_id=store.ref.experiment_id,
        pack_id=STUDY_PACK_ID,
        task_id=task_id,
        variant_id=variant_id,
        repetition=0,
    )
    return store.read_json(store.trial_path(key))


def _summary(ref: ExperimentRef, task_ids: tuple[str, ...]) -> dict[str, Any]:
    store = ArtifactStore(ref)
    report = rebuild_full_report(ref)
    groups: dict[str, list[dict[str, Any]]] = {f"budget-{round(ratio * 100)}": [] for ratio in RATIOS}
    details: list[dict[str, Any]] = []
    for task_id in task_ids:
        for variant_id in groups:
            trial = _read_trial(store, task_id, variant_id)
            metrics = trial.get("metrics", {})
            verification = trial.get("verification", {})
            item = {
                "task_id": task_id,
                "variant_id": variant_id,
                "status": trial.get("status"),
                "verifier_state": verification.get("state"),
                "input_tokens": metrics.get("trial_total_input_tokens"),
                "output_tokens": metrics.get("trial_total_output_tokens"),
                "total_tokens": (
                    metrics.get("trial_total_input_tokens") + metrics.get("trial_total_output_tokens")
                    if isinstance(metrics.get("trial_total_input_tokens"), int)
                    and isinstance(metrics.get("trial_total_output_tokens"), int)
                    else None
                ),
                "repeated_tool_calls": metrics.get("repeated_tool_calls"),
                "usage_complete": metrics.get("usage_complete"),
            }
            details.append(item)
            groups[variant_id].append(item)

    conditions: dict[str, Any] = {}
    for variant_id, records in groups.items():
        passed = sum(record["verifier_state"] == "passed" for record in records)
        usage_complete = sum(record["usage_complete"] is True for record in records)
        token_fields = ("input_tokens", "output_tokens", "total_tokens")
        totals = {
            field: (
                sum(record[field] for record in records)
                if all(isinstance(record[field], int) for record in records)
                else None
            )
            for field in token_fields
        }
        conditions[variant_id] = {
            "context_budget_ratio": int(variant_id.removeprefix("budget-")) / 100,
            "trials": len(records),
            "verifier_passed": passed,
            "verifier_pass_rate": passed / len(records) if records else 0,
            "usage_complete_trials": usage_complete,
            **totals,
            "repeated_tool_calls": sum(
                record["repeated_tool_calls"] for record in records if isinstance(record["repeated_tool_calls"], int)
            ),
        }
    return {
        "schema": "codeflow.context-budget-study.v1",
        "experiment_id": ref.experiment_id,
        "report_digest": report.report_digest,
        "planned_trials": report.planned_trials,
        "terminal_trials": report.terminal_trials,
        "measurement_valid": report.measurement_valid,
        "task_ids": list(task_ids),
        "condition_summary": conditions,
        "trial_details": details,
        "interpretation": {
            "repeated_tool_call_definition": (
                "an additional exact repeat of the same registered tool name and canonical arguments "
                "within one task run"
            ),
            "verifier_pass_rate_denominator": "all 10 planned trials in each budget condition",
            "token_scope": "actual Provider usage across main Agent and Curator calls",
        },
    }


async def run_study(
    output_root: Path,
    provider_budget_ledger: Path | None = None,
    *,
    skip_preflight: bool = False,
) -> dict[str, Any]:
    print("Loading the frozen DeepSeek configuration.", flush=True)
    config = load_config()
    codeflow_config = load_codeflow_config()
    model = config.agents.defaults.model
    provider_name = config.get_provider_name(model)
    if provider_name != "deepseek" or model.rsplit("/", 1)[-1] != "deepseek-v4-flash":
        raise RuntimeError(
            "the context-budget study is frozen to deepseek / deepseek-v4-flash; configure that model before running",
        )
    delegate = make_provider(config)
    print("Provider initialized; preparing the capped usage ledger.", flush=True)
    # The ledger's UTF-8-byte fallback is intentionally conservative and can
    # exceed provider-reported prompt tokens for serialized tool-heavy turns.
    # Keep the Agent's context window fixed, while allowing a 2x accounting
    # ceiling for that estimator's headroom.
    max_input_tokens = CONTEXT_BENCHMARK_WINDOW_TOKENS * 2
    max_output_tokens = CONTEXT_BENCHMARK_OUTPUT_TOKENS
    max_trials = len(load_context_budget_study_tasks()) * len(RATIOS)
    max_request_attempts = 1 + max_trials * MAX_CALLS_PER_TRIAL * MAX_ATTEMPTS_PER_CALL
    approval_digest = canonical_digest(
        {
            "suite": "context-budget-study-10-distinct-tasks",
            "max_cost_cny": MAX_COST_CNY,
            "codeflow_source_digest": _source_digest(),
        }
    )
    ledger_path = provider_budget_ledger or output_root / "provider-budget.jsonl"
    ledger_config = ProviderBudgetConfig(
        hard_cap_cny=MAX_COST_CNY,
        external_service_reserve_cny=0,
        max_total_request_attempts=max_request_attempts,
        max_additional_request_attempts=max_trials * MAX_CALLS_PER_TRIAL,
        max_input_tokens_per_call=max_input_tokens,
        max_output_tokens_per_call=max_output_tokens,
        input_cache_miss_usd_per_million=INPUT_PRICE_USD_PER_MILLION,
        output_usd_per_million=OUTPUT_PRICE_USD_PER_MILLION,
        conservative_usd_to_cny_multiplier=USD_TO_CNY,
    )
    # Bootstrap the empty, task-scoped ledger's high-water file before binding
    # the experiment approval digest. This never dispatches a provider call.
    ledger_prefix = ProviderBudgetLedger(ledger_path, ledger_config).snapshot()
    ledger = ProviderBudgetLedger(
        ledger_path,
        ProviderBudgetConfig(
            **{
                **ledger_config.__dict__,
                "approval_digest": approval_digest,
                "ledger_prefix_event_count": ledger_prefix.ledger_event_count,
                "ledger_prefix_digest": ledger_prefix.ledger_digest,
                "ledger_prefix_charged_cny": ledger_prefix.provider_charged_cny,
            }
        ),
    )
    provider = BudgetGuardedProvider(delegate, ledger=ledger)
    provider.generation = delegate.generation
    if skip_preflight:
        print("Using the previously successful preflight recorded in the shared budget ledger.", flush=True)
    else:
        print(f"Running provider/tool/usage preflight under a {MAX_COST_CNY:.2f} CNY hard cap.", flush=True)
        await _preflight(provider, model)
    print("Preflight passed; starting the 30-trial three-ratio matrix.", flush=True)

    tasks = load_context_budget_study_tasks()
    pack = ContextPack(
        ContextTrack.FORMAL,
        tasks=tasks,
        budget_ratios=RATIOS,
        runner=RuntimeContextTrialRunner(
            config=config,
            codeflow_config=codeflow_config,
            provider=provider,
        ),
    )
    pack_definition = pack.definition()
    registry = PackRegistry()
    registry.register(pack)
    execution = ExecutionPolicy(
        retry_policy="symmetric",
        provider_call_max_attempts=MAX_ATTEMPTS_PER_CALL,
        max_comparison_block_attempts=1,
        max_comparison_block_retries_total=0,
        max_provider_calls_per_trial=MAX_CALLS_PER_TRIAL,
        provider_trial_budgets=(
            ProviderTrialBudget(
                pack_id=STUDY_PACK_ID,
                max_provider_calls_per_trial=MAX_CALLS_PER_TRIAL,
                max_input_tokens_per_call=max_input_tokens,
                max_output_tokens_per_call=max_output_tokens,
            ),
        ),
    )
    spec = ExperimentSpec(
        suite="context-budget-study-10",
        repetitions=1,
        pack_ids=(STUDY_PACK_ID,),
        output_root=output_root,
        identity={
            "provider": provider_name,
            "model": model,
            "codeflow_source_digest": _source_digest(),
            "context_window_tokens": CONTEXT_BENCHMARK_WINDOW_TOKENS,
            "max_input_tokens_per_call": max_input_tokens,
            "reserved_output_tokens": CONTEXT_BENCHMARK_OUTPUT_TOKENS,
            "context_budget_ratios": list(RATIOS),
            "task_set_digest": pack_definition.identity["task_set_digest"],
            "budget_cap_cny": MAX_COST_CNY,
            "pricing": {
                "input_cache_miss_usd_per_million": INPUT_PRICE_USD_PER_MILLION,
                "output_usd_per_million": OUTPUT_PRICE_USD_PER_MILLION,
                "usd_to_cny_multiplier": USD_TO_CNY,
            },
        },
        execution=execution,
    )
    ref = await run(spec, registry=registry)
    print("All planned Trials have terminal records; rebuilding the report.", flush=True)
    result = _summary(ref, tuple(task.task_id for task in tasks))
    result["provider_budget"] = ledger.snapshot().__dict__
    summary_path = ref.root / "context-budget-summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the 50% / 80% / 93% CodeFlow context budget study.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(".codeflow/evidence/context-budget-study"),
    )
    parser.add_argument(
        "--provider-budget-ledger",
        type=Path,
        help="reuse a prior ledger so retries stay within the same total spend cap",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="reuse a successful preflight already recorded in the shared budget ledger",
    )
    args = parser.parse_args()
    result = asyncio.run(
        run_study(
            args.output_root.resolve(),
            args.provider_budget_ledger.resolve() if args.provider_budget_ledger else None,
            skip_preflight=args.skip_preflight,
        )
    )
    print(json.dumps(result["condition_summary"], ensure_ascii=False, indent=2))
    print(f"Artifacts: {args.output_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
