from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.codeflowbench.canonical import canonical_bytes, canonical_digest
from codeflow.agent.loop import AgentLoop
from codeflow.agent.tools.base import Tool
from codeflow.call_efficiency.models import CallRecord
from codeflow.config.paths import RuntimePaths
from codeflow.config.codeflow import CodeFlowConfig
from codeflow.config.schema import Config
from codeflow.providers.base import GenerationSettings, LLMResponse
from codeflow.providers.litellm_provider import LiteLLMProvider
from codeflow.spine import ChatType, Origin, Source, Text, TurnRequest

from .live import (
    PRICE_SNAPSHOT,
    ArmConfig,
    CampaignBudget,
    CampaignConfig,
    CampaignError,
    LiveTask,
    LiveTrialResult,
    TaskCorpus,
    build_arm,
    build_current_campaign_report,
    rotated_cache_policies,
)


@dataclass(frozen=True)
class ProviderCallRecord:
    request_digest: str
    requested_model: str
    actual_model: str | None
    usage_complete: bool
    fresh_input_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    output_tokens: int
    cost_usd: float
    finish_reason: str
    latency_ms: int
    outcome: str = "success"
    error_type: str | None = None


@dataclass(frozen=True)
class TrialArtifact:
    result: LiveTrialResult
    provider_calls: tuple[ProviderCallRecord, ...]
    outputs: tuple[str, ...]
    call_efficiency_records: tuple[dict[str, Any], ...] = ()


TrialExecutor = Callable[..., Awaitable[TrialArtifact]]


def load_deepseek_key(*, config_path: Path | None = None) -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    path = config_path or Path.home() / ".codeflow" / "config.json"
    if not key and path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            key = str(raw.get("providers", {}).get("deepseek", {}).get("apiKey", "")).strip()
        except (OSError, json.JSONDecodeError, AttributeError):
            key = ""
    if not key:
        raise CampaignError("DeepSeek credential is not configured")
    return key


async def run_formal_campaign(
    *,
    config: CampaignConfig,
    corpus: TaskCorpus,
    api_key: str,
    codeflow_commit: str,
    trial_executor: TrialExecutor | None = None,
    preflight_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not api_key:
        raise CampaignError("DeepSeek credential is not configured")
    if len(codeflow_commit) != 40:
        raise CampaignError("codeflow_commit must be a full Git commit")
    executor = trial_executor or execute_live_trial
    if executor is execute_live_trial and (preflight_report is None or preflight_report.get("passed") is not True):
        raise CampaignError("a passing live cache preflight is required")
    manifest = {
        "schema": "codeflow.codeflowbench.call-efficiency-cost.manifest.v2",
        "codeflow_commit": codeflow_commit,
        "model": config.model,
        "repetitions": config.repetitions,
        "hard_cap_usd": config.hard_cap_usd,
        "max_provider_calls": config.max_provider_calls,
        "bootstrap_samples": config.bootstrap_samples,
        "bootstrap_seed": config.bootstrap_seed,
        "task_corpus_digest": corpus.digest,
        "price_snapshot": PRICE_SNAPSHOT,
        "runtime_boundary": "shared_runtime_assembly",
        "call_efficiency_mode": "observe",
    }
    plan_digest = canonical_digest(manifest)
    manifest["plan_digest"] = plan_digest
    config.output_root.mkdir(parents=True, exist_ok=True)
    _freeze_json(config.output_root / "manifest.json", manifest)

    artifacts = _load_retained_trials(config.output_root / "trials", plan_digest=plan_digest)
    budget = CampaignBudget(
        hard_cap_usd=config.hard_cap_usd,
        max_provider_calls=config.max_provider_calls,
        spent_usd=sum(artifact.result.cost_usd for artifact in artifacts.values()),
        provider_calls=sum(artifact.result.provider_calls for artifact in artifacts.values()),
    )
    block_index = 0
    for task in corpus.tasks:
        for repetition in range(config.repetitions):
            for cache_policy in rotated_cache_policies(plan_digest, block_index):
                key = _trial_key(task.task_id, repetition, cache_policy)
                if key in artifacts:
                    continue
                before_calls = budget.provider_calls
                before_cost = budget.spent_usd
                artifact = await executor(
                    task,
                    repetition,
                    build_arm(cache_policy),
                    api_key=api_key,
                    config=config,
                    plan_digest=plan_digest,
                    budget=budget,
                )
                if budget.provider_calls == before_calls and budget.spent_usd == before_cost:
                    budget.provider_calls += artifact.result.provider_calls
                    budget.spent_usd += artifact.result.cost_usd
                path = config.output_root / "trials" / f"{key}.json"
                _freeze_json(path, _trial_payload(artifact, plan_digest=plan_digest))
                artifacts[key] = artifact
            block_index += 1

    trials = tuple(
        artifact.result
        for _, artifact in sorted(
            artifacts.items(),
            key=lambda item: (
                item[1].result.task_id,
                item[1].result.repetition,
                item[1].result.cache_policy,
            ),
        )
    )
    report = build_current_campaign_report(config=config, corpus=corpus, trials=trials)
    report["campaign"]["codeflow_commit"] = codeflow_commit
    report["campaign"]["plan_digest"] = plan_digest
    report["campaign"]["observed_cost_usd"] = sum(trial.cost_usd for trial in trials)
    report["campaign"]["observed_provider_calls"] = sum(trial.provider_calls for trial in trials)
    if preflight_report is not None:
        report["campaign"]["preflight_digest"] = canonical_digest(preflight_report)
    report["report_digest"] = canonical_digest({key: value for key, value in report.items() if key != "report_digest"})
    _write_json(config.output_root / "report.json", report)
    return report


def verify_retained_campaign(
    *,
    output_root: Path,
    corpus: TaskCorpus,
    codeflow_commit: str,
) -> dict[str, Any]:
    manifest_path = output_root / "manifest.json"
    report_path = output_root / "report.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        retained_report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError("retained campaign manifest or report is unreadable") from exc
    expected_plan_digest = str(manifest.get("plan_digest") or "")
    manifest_payload = {key: value for key, value in manifest.items() if key != "plan_digest"}
    checks = {
        "manifest_schema": manifest.get("schema") == "codeflow.codeflowbench.call-efficiency-cost.manifest.v2",
        "manifest_digest": expected_plan_digest == canonical_digest(manifest_payload),
        "codeflow_commit": manifest.get("codeflow_commit") == codeflow_commit,
        "task_corpus": manifest.get("task_corpus_digest") == corpus.digest,
        "runtime_boundary": manifest.get("runtime_boundary") == "shared_runtime_assembly",
        "call_efficiency_mode": manifest.get("call_efficiency_mode") == "observe",
    }
    try:
        config = CampaignConfig(
            model=str(manifest.get("model") or ""),
            repetitions=int(manifest.get("repetitions", 0)),
            hard_cap_usd=float(manifest.get("hard_cap_usd", 0)),
            max_provider_calls=int(manifest.get("max_provider_calls", 0)),
            bootstrap_samples=int(manifest.get("bootstrap_samples", 0)),
            bootstrap_seed=int(manifest.get("bootstrap_seed", 0)),
            output_root=output_root,
        )
    except (TypeError, ValueError) as exc:
        raise CampaignError("retained campaign manifest is invalid") from exc
    trial_paths = sorted((output_root / "trials").glob("*.json"))
    try:
        raw_rows = [json.loads(path.read_text(encoding="utf-8")) for path in trial_paths]
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError("retained trial artifact is unreadable") from exc
    artifacts = _load_retained_trials(output_root / "trials", plan_digest=expected_plan_digest)
    expected_trials = len(corpus.tasks) * config.repetitions * 2
    checks["trial_count"] = len(artifacts) == expected_trials
    checks["raw_trial_file_count"] = len(trial_paths) == expected_trials
    checks["raw_trial_schema"] = all(
        row.get("schema") == "codeflow.codeflowbench.call-efficiency-cost.trial.v2" for row in raw_rows
    )
    checks["raw_plan_binding"] = all(row.get("plan_digest") == expected_plan_digest for row in raw_rows)
    checks["raw_call_receipts"] = all(_serialized_call_records_complete(artifact) for artifact in artifacts.values())
    trials = tuple(
        artifact.result
        for _, artifact in sorted(
            artifacts.items(),
            key=lambda item: (
                item[1].result.task_id,
                item[1].result.repetition,
                item[1].result.cache_policy,
            ),
        )
    )
    rebuilt = build_current_campaign_report(config=config, corpus=corpus, trials=trials)
    rebuilt["campaign"]["codeflow_commit"] = codeflow_commit
    rebuilt["campaign"]["plan_digest"] = expected_plan_digest
    rebuilt["campaign"]["observed_cost_usd"] = sum(trial.cost_usd for trial in trials)
    rebuilt["campaign"]["observed_provider_calls"] = sum(trial.provider_calls for trial in trials)
    preflight_path = output_root / "preflight.json"
    try:
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError("retained preflight is unreadable") from exc
    checks["preflight_passed"] = preflight.get("passed") is True
    rebuilt["campaign"]["preflight_digest"] = canonical_digest(preflight)
    rebuilt["report_digest"] = canonical_digest(
        {key: value for key, value in rebuilt.items() if key != "report_digest"}
    )
    checks["report_rebuilt"] = canonical_bytes(rebuilt) == canonical_bytes(retained_report)
    verifier = {
        "schema": "codeflow.codeflowbench.call-efficiency-cost.verifier.v1",
        "checks": checks,
        "trial_count": len(trials),
        "provider_call_count": sum(trial.provider_calls for trial in trials),
        "report_digest": rebuilt["report_digest"],
        "passed": all(checks.values()),
    }
    verifier["verifier_digest"] = canonical_digest(verifier)
    _write_bytes(
        output_root / "raw-outcomes.jsonl",
        b"".join(canonical_bytes(row) + b"\n" for row in raw_rows),
    )
    _write_json(output_root / "aggregate.json", rebuilt)
    _write_json(output_root / "verifier-report.json", verifier)
    _write_json(
        output_root / "claim-eligibility.json",
        {
            "schema": "codeflow.codeflowbench.call-efficiency-cost.claim.v1",
            "measurement_valid": verifier["passed"],
            "positive_claim_eligible": verifier["passed"] and rebuilt["claim"]["claim_eligible"],
            "gates": rebuilt["gates"],
            "paired_cost_reduction_95_ci": rebuilt["claim"]["paired_cost_reduction_95_ci"],
            "report_digest": rebuilt["report_digest"],
        },
    )
    inventory = _build_inventory(output_root)
    _write_json(output_root / "inventory.json", inventory)
    if not verifier["passed"]:
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise CampaignError(f"offline verification failed: {failed}")
    return verifier


async def run_cache_preflight(
    *,
    api_key: str,
    config: CampaignConfig,
) -> dict[str, Any]:
    if not api_key:
        raise CampaignError("DeepSeek credential is not configured")
    budget = CampaignBudget(hard_cap_usd=0.05, max_provider_calls=8)
    namespace = canonical_digest(
        {"model": config.model, "probe": "deepseek-tokenwise-cache-preflight-v2", "time": time.time_ns()}
    )[:24]
    stable_provider = _RecordingDeepSeekProvider(
        api_key=api_key,
        model=config.model,
        session_id=namespace,
        budget=budget,
    )
    disrupted_namespace = canonical_digest({"namespace": namespace, "arm": "disrupted"})[:24]
    disrupted_provider = _RecordingDeepSeekProvider(
        api_key=api_key,
        model=config.model,
        session_id=disrupted_namespace,
        budget=budget,
    )
    long_system = _preflight_system(namespace)
    stable_messages = [
        {"role": "system", "content": long_system},
        {"role": "user", "content": "Reply only CACHE_OK."},
    ]
    for _ in range(2):
        await stable_provider.chat(
            messages=stable_messages,
            model=config.model,
            max_tokens=16,
            temperature=0.0,
        )
    disruptor = build_arm("prefix_disrupted").strategy
    if disruptor is None:
        raise CampaignError("prefix disruption control is unavailable")
    for _ in range(2):
        messages, tools, model = await disruptor.before_llm_call(stable_messages, None, config.model)
        await disrupted_provider.chat(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=16,
            temperature=0.0,
        )

    cold, warm = stable_provider.records
    disrupted_cold, disrupted_warm = disrupted_provider.records
    records = tuple(stable_provider.records + disrupted_provider.records)
    checks = {
        "warm_call_reads_cache": warm.cache_read_tokens > 0,
        "stable_prefix_improves_cache_hits": warm.cache_read_tokens > disrupted_warm.cache_read_tokens,
        "cold_namespaces_are_isolated": cold.cache_read_tokens == 0 and disrupted_cold.cache_read_tokens == 0,
        "usage_complete": all(record.usage_complete for record in records),
        "model_exact": {record.actual_model for record in records} == {config.model},
    }
    report = {
        "schema": "codeflow.codeflowbench.tokenwise-cost.preflight.v1",
        "model": config.model,
        "price_snapshot": PRICE_SNAPSHOT,
        "checks": checks,
        "calls": [asdict(record) for record in records],
        "spent_usd": budget.spent_usd,
        "passed": all(checks.values()),
    }
    report["report_digest"] = canonical_digest(report)
    _write_json(config.output_root / "preflight.json", report)
    if not report["passed"]:
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise CampaignError(f"live cache preflight failed: {failed}")
    return report


async def execute_live_trial(
    task: LiveTask,
    repetition: int,
    arm: ArmConfig,
    *,
    api_key: str,
    config: CampaignConfig,
    plan_digest: str,
    budget: CampaignBudget,
) -> TrialArtifact:
    namespace = canonical_digest(
        {
            "plan_digest": plan_digest,
            "task_id": task.task_id,
            "repetition": repetition,
            "cache_policy": arm.cache_policy,
        }
    )[:24]
    workspace = config.output_root / "workspaces" / _trial_key(task.task_id, repetition, arm.cache_policy)
    workspace.mkdir(parents=True, exist_ok=True)
    _seed_workspace(workspace, namespace=namespace, workload_class=task.workload_class)
    provider = _RecordingDeepSeekProvider(
        api_key=api_key,
        model=config.model,
        session_id=namespace,
        budget=budget,
    )
    provider.generation = GenerationSettings(temperature=0.0, max_tokens=96)
    runtime_config = Config()
    runtime_config.agents.defaults.workspace = str(workspace)
    runtime_config.agents.defaults.model = config.model
    runtime_config.agents.defaults.max_tool_iterations = 6
    runtime_config.agents.defaults.context_window_tokens = 200_000
    runtime_config.agents.defaults.temperature = 0.0
    runtime_config.agents.defaults.max_tokens = 96
    runtime_config.agents.defaults.enable_personalization = False
    runtime_config.tools.mcp_servers = {}
    runtime_config.tools.sandbox.backend = "none"
    codeflow_config = CodeFlowConfig(base=runtime_config)
    codeflow_config.memory.backend = None
    codeflow_config.skill_forge.enabled = False
    codeflow_config.skill_forge.router.enabled = False
    codeflow_config.skill_forge.rewrite_enabled = False
    codeflow_config.skill_forge.llm_gate_enabled = False
    codeflow_config.call_efficiency.mode = "observe"
    codeflow_config.call_efficiency.usage_tracking = True
    codeflow_config.context.fast_path_threshold = 1.0
    codeflow_config.runtime.checkpoint.policy = "never"
    from codeflow.cli._runtime_assembly import assemble_runtime

    runtime = assemble_runtime(
        runtime_config,
        codeflow_config,
        provider=provider,
        cron_service=None,
        interactive=False,
        paths=RuntimePaths(workspace=workspace, state=workspace / "state"),
    )
    loop = runtime.agent_loop
    loop.now_fn = lambda: datetime(2026, 8, 6, tzinfo=timezone.utc)
    if arm.strategy is not None:
        loop.strategies.register(arm.strategy)
    loop.tools._tools.clear()
    if task.workload_class == "tool_accumulation":
        loop.tools.register(_LookupTool())
    elif task.workload_class == "intra_turn_tool_chain":
        for tool in (_ChainAlpha(), _ChainBeta(), _ChainGamma()):
            loop.tools.register(tool)

    session_key = f"tokenwise:{namespace}"
    if task.seed_history_turns:
        _seed_history(loop, session_key=session_key, turns=task.seed_history_turns)

    outputs: list[str] = []
    findings: list[str] = []
    started = time.perf_counter()
    total_tool_calls = 0
    total_tool_failures = 0
    try:
        for turn_index, prompt in enumerate(task.user_prompts):
            events: list[Any] = []

            async def emit(event: Any) -> None:
                events.append(event)

            outcome = await loop.run_turn(
                TurnRequest(
                    origin=Origin.USER,
                    source=Source(
                        channel="cli",
                        chat_id=namespace,
                        sender_id="tokenwise-benchmark",
                        chat_type=ChatType.DM,
                    ),
                    text=prompt,
                    conversation=session_key,
                ),
                emit,
                lambda: [],
                stream=False,
            )
            text = next((event.content for event in reversed(events) if isinstance(event, Text)), "")
            normalized = text.strip()
            outputs.append(normalized)
            if normalized != task.expected_outputs[turn_index]:
                findings.append(f"output_mismatch_turn_{turn_index + 1}")
            total_tool_calls += outcome.tool_calls
            total_tool_failures += outcome.tool_failures
    except Exception as exc:
        findings.append(f"runtime_error:{type(exc).__name__}")
    finally:
        await runtime.close()

    expected_tool_calls = task.turn_count * task.expected_tool_calls_per_turn
    if total_tool_calls != expected_tool_calls:
        findings.append("tool_call_count_mismatch")
    if total_tool_failures:
        findings.append("tool_failure")
    records = tuple(provider.records)
    call_efficiency_records = runtime.call_efficiency.records if runtime.call_efficiency is not None else ()
    serialized_call_efficiency_records = tuple(asdict(record) for record in call_efficiency_records)
    ledger_id = getattr(getattr(runtime.call_efficiency, "ledger", None), "_ledger_id", None)
    health = _load_call_efficiency_health(
        workspace / "state" / "telemetry",
        ledger_id=ledger_id,
    )
    call_records_complete = _call_efficiency_records_complete(records, call_efficiency_records)
    actual_models = {record.actual_model for record in records if record.actual_model}
    usage_complete = bool(records) and all(record.usage_complete for record in records)
    cost_complete = usage_complete and all(record.cost_usd >= 0 for record in records)
    actual_model = next(iter(actual_models)) if len(actual_models) == 1 else None
    fallback_used = actual_models != {config.model}
    if fallback_used:
        findings.append("model_or_route_drift")
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    result = LiveTrialResult(
        task_id=task.task_id,
        workload_class=task.workload_class,
        repetition=repetition,
        cache_policy=arm.cache_policy,
        task_passed=not findings,
        usage_complete=usage_complete,
        cost_complete=cost_complete,
        requested_model=config.model,
        actual_model=actual_model,
        fallback_used=fallback_used,
        fresh_input_tokens=sum(record.fresh_input_tokens for record in records),
        cache_write_tokens=sum(record.cache_write_tokens for record in records),
        cache_read_tokens=sum(record.cache_read_tokens for record in records),
        output_tokens=sum(record.output_tokens for record in records),
        cost_usd=sum(record.cost_usd for record in records),
        provider_calls=len(records),
        latency_ms=elapsed_ms,
        findings=tuple(findings),
        runtime_assembly=True,
        call_efficiency_mode=runtime.call_efficiency.mode if runtime.call_efficiency is not None else None,
        call_efficiency_records=len(call_efficiency_records),
        call_efficiency_records_complete=call_records_complete,
        call_efficiency_ledger_status=health.get("status"),
        call_efficiency_accepted_records=int(health.get("accepted_records", 0)),
        call_efficiency_persisted_records=int(health.get("persisted_records", 0)),
        call_efficiency_lost_records=int(health.get("lost_records", 0)),
    )
    return TrialArtifact(
        result=result,
        provider_calls=records,
        outputs=tuple(outputs),
        call_efficiency_records=serialized_call_efficiency_records,
    )


class _RecordingDeepSeekProvider(LiteLLMProvider):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        session_id: str,
        budget: CampaignBudget,
    ) -> None:
        super().__init__(
            api_key=api_key,
            default_model=model,
            provider_name="deepseek",
            disable_auto_cache_control=True,
            extra_body={"user_id": session_id, "thinking": {"type": "disabled"}},
        )
        self.set_transport_num_retries(0)
        self._budget = budget
        self.records: list[ProviderCallRecord] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        self._budget.reserve_call()
        started = time.perf_counter()
        requested_model = model or self.default_model
        try:
            response = await super().chat(
                messages=messages,
                tools=tools,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                tool_choice=tool_choice,
            )
        except BaseException as exc:
            self.records.append(
                ProviderCallRecord(
                    request_digest=canonical_digest(
                        {
                            "model": requested_model,
                            "messages": messages,
                            "tools": tools or [],
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                        }
                    ),
                    requested_model=requested_model,
                    actual_model=None,
                    usage_complete=False,
                    fresh_input_tokens=0,
                    cache_write_tokens=0,
                    cache_read_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    finish_reason="error",
                    latency_ms=round((time.perf_counter() - started) * 1000),
                    outcome="error",
                    error_type=type(exc).__name__,
                )
            )
            raise
        usage = response.usage or {}
        complete = all(
            field in usage and usage[field] is not None
            for field in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "cache_read_input_tokens",
                "cache_miss_input_tokens",
            )
        )
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        output = int(usage.get("completion_tokens", 0) or 0)
        cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
        fresh = int(usage.get("cache_miss_input_tokens", 0) or 0)
        cache_write = 0
        complete = complete and prompt == fresh + cache_read
        cost = (
            fresh * float(PRICE_SNAPSHOT["cache_miss_usd_per_token"])
            + output * float(PRICE_SNAPSHOT["output_usd_per_token"])
            + cache_read * float(PRICE_SNAPSHOT["cache_hit_usd_per_token"])
        )
        self._budget.record_call(cost_usd=cost)
        actual_model = _canonical_model(response.model)
        record = ProviderCallRecord(
            request_digest=canonical_digest(
                {
                    "model": requested_model,
                    "messages": messages,
                    "tools": tools or [],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
            ),
            requested_model=requested_model,
            actual_model=actual_model,
            usage_complete=complete,
            fresh_input_tokens=fresh,
            cache_write_tokens=cache_write,
            cache_read_tokens=cache_read,
            output_tokens=output,
            cost_usd=cost,
            finish_reason=response.finish_reason,
            latency_ms=round((time.perf_counter() - started) * 1000),
        )
        self.records.append(record)
        return response


_TOOL_BLOB = "\n".join(f"field_{index:03d}=stable_value_{index:03d}_aaaaaaaa_bbbbbbbb_cccccccc" for index in range(80))


class _LookupTool(Tool):
    name = "lookup_record"
    description = "Look up one benchmark record by code. Call exactly once for each user request."
    parameters = {
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
    }

    async def execute(self, code: str = "", **_kwargs: Any) -> str:
        return f"Record {code}\n{_TOOL_BLOB}\nReply only {code}."


class _ChainAlpha(Tool):
    name = "chain_alpha"
    description = "First step of the benchmark chain."
    parameters = {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}

    async def execute(self, code: str = "", **_kwargs: Any) -> str:
        return f"Alpha complete for {code}. Call chain_beta next.\n{_TOOL_BLOB}"


class _ChainBeta(Tool):
    name = "chain_beta"
    description = "Second step of the benchmark chain."
    parameters = {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}

    async def execute(self, code: str = "", **_kwargs: Any) -> str:
        return f"Beta complete for {code}. Call chain_gamma next.\n{_TOOL_BLOB}"


class _ChainGamma(Tool):
    name = "chain_gamma"
    description = "Final step of the benchmark chain."
    parameters = {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}

    async def execute(self, code: str = "", **_kwargs: Any) -> str:
        return f"Gamma complete for {code}. Reply only {code}.\n{_TOOL_BLOB}"


def _seed_workspace(workspace: Path, *, namespace: str, workload_class: str) -> None:
    rules = "\n".join(
        f"Rule {index:03d}: preserve the benchmark protocol and return only the requested token."
        for index in range(180)
    )
    tool_protocol = ""
    if workload_class == "tool_accumulation":
        tool_protocol = "For every request, call lookup_record exactly once, then return only the requested code."
    elif workload_class == "intra_turn_tool_chain":
        tool_protocol = (
            "Call exactly one tool per response in this order: chain_alpha, chain_beta, chain_gamma. "
            "After chain_gamma, return only the requested code."
        )
    state = workspace / "state"
    agent_profile = state / "agent_memory" / "profile"
    user_profile = state / "user_memory" / "profile"
    agent_profile.mkdir(parents=True, exist_ok=True)
    user_profile.mkdir(parents=True, exist_ok=True)
    (agent_profile / "soul.md").write_text(
        f"# TokenWise benchmark\nNamespace: {namespace}\n{tool_protocol}\n{rules}\n",
        encoding="utf-8",
    )
    (agent_profile / "agent.md").write_text(
        "# Agent\nFollow the benchmark protocol exactly.\n",
        encoding="utf-8",
    )
    (user_profile / "user.md").write_text("# User\nBenchmark operator.\n", encoding="utf-8")
    (state / "TOOLS.md").write_text(
        "# Tools\nUse only the registered benchmark tools.\n",
        encoding="utf-8",
    )


def _seed_history(loop: AgentLoop, *, session_key: str, turns: int) -> None:
    session = loop.sessions.get_or_create(session_key)
    body = " ".join(f"history_token_{index:03d}" for index in range(80))
    for index in range(turns):
        session.add_message("user", f"Background {index + 1}: {body}")
        session.add_message("assistant", f"Recorded background {index + 1}: {body}")


def _canonical_model(model: str | None) -> str | None:
    if not model:
        return None
    normalized = model.removeprefix("openrouter/")
    if normalized.startswith("deepseek-v4-"):
        return f"deepseek/{normalized}"
    return normalized


def _preflight_system(namespace: str) -> str:
    rules = "\n".join(
        f"stable rule {index:03d}: preserve this exact prefix for the prompt cache verification."
        for index in range(220)
    )
    return f"Namespace {namespace}. Reply only with the requested token.\n{rules}"


def _preflight_tool(version: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "preflight_lookup",
            "description": f"Prompt cache schema probe {version}.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    }


def _trial_key(task_id: str, repetition: int, cache_policy: str) -> str:
    return f"{task_id}--r{repetition}--{cache_policy}"


def _trial_payload(artifact: TrialArtifact, *, plan_digest: str) -> dict[str, Any]:
    return {
        "schema": "codeflow.codeflowbench.call-efficiency-cost.trial.v2",
        "plan_digest": plan_digest,
        "result": asdict(artifact.result),
        "provider_calls": [asdict(record) for record in artifact.provider_calls],
        "call_efficiency_records": list(artifact.call_efficiency_records),
        "outputs": list(artifact.outputs),
    }


def _load_retained_trials(root: Path, *, plan_digest: str) -> dict[str, TrialArtifact]:
    retained: dict[str, TrialArtifact] = {}
    if not root.is_dir():
        return retained
    for path in root.glob("*.json"):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw.get("plan_digest") != plan_digest:
                continue
            result = LiveTrialResult(**{**raw["result"], "findings": tuple(raw["result"]["findings"])})
            calls = tuple(ProviderCallRecord(**record) for record in raw["provider_calls"])
            artifact = TrialArtifact(
                result=result,
                provider_calls=calls,
                outputs=tuple(raw["outputs"]),
                call_efficiency_records=tuple(raw.get("call_efficiency_records", ())),
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        retained[_trial_key(result.task_id, result.repetition, result.cache_policy)] = artifact
    return retained


def _load_call_efficiency_health(
    telemetry_dir: Path,
    *,
    ledger_id: str | None,
) -> dict[str, Any]:
    path = telemetry_dir / "call-efficiency-ledger-health.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    ledgers = payload.get("ledgers")
    if ledger_id is not None and isinstance(ledgers, dict):
        entry = ledgers.get(ledger_id)
        return entry if isinstance(entry, dict) else {}
    return payload


def _call_efficiency_records_complete(
    provider_records: tuple[ProviderCallRecord, ...],
    call_records: tuple[CallRecord, ...],
) -> bool:
    if len(provider_records) != len(call_records):
        return False
    for provider_record, call_record in zip(provider_records, call_records, strict=True):
        if provider_record.outcome != call_record.outcome:
            return False
        if provider_record.requested_model != call_record.requested_model:
            return False
        if provider_record.usage_complete != call_record.usage.complete:
            return False
        if provider_record.fresh_input_tokens != call_record.usage.input_tokens:
            return False
        if provider_record.cache_read_tokens != call_record.usage.cache_read_tokens:
            return False
        if provider_record.cache_write_tokens != call_record.usage.cache_write_tokens:
            return False
        if provider_record.output_tokens != call_record.usage.output_tokens:
            return False
        if provider_record.cost_usd != call_record.estimated_cost_usd:
            return False
    return True


def _serialized_call_records_complete(artifact: TrialArtifact) -> bool:
    if artifact.result.provider_calls != len(artifact.provider_calls):
        return False
    if artifact.result.call_efficiency_records != len(artifact.call_efficiency_records):
        return False
    if len(artifact.provider_calls) != len(artifact.call_efficiency_records):
        return False
    for provider_record, raw_call_record in zip(
        artifact.provider_calls,
        artifact.call_efficiency_records,
        strict=True,
    ):
        usage = raw_call_record.get("usage")
        if not isinstance(usage, dict):
            return False
        expected = {
            "outcome": provider_record.outcome,
            "requested_model": provider_record.requested_model,
            "actual_model": _canonical_model(provider_record.actual_model) or "",
            "usage_complete": provider_record.usage_complete,
            "fresh_input_tokens": provider_record.fresh_input_tokens,
            "cache_read_tokens": provider_record.cache_read_tokens,
            "cache_write_tokens": provider_record.cache_write_tokens,
            "output_tokens": provider_record.output_tokens,
            "cost_usd": provider_record.cost_usd,
        }
        observed = {
            "outcome": raw_call_record.get("outcome"),
            "requested_model": raw_call_record.get("requested_model"),
            "actual_model": _canonical_model(raw_call_record.get("actual_model")),
            "usage_complete": usage.get("complete"),
            "fresh_input_tokens": usage.get("input_tokens"),
            "cache_read_tokens": usage.get("cache_read_tokens"),
            "cache_write_tokens": usage.get("cache_write_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "cost_usd": raw_call_record.get("estimated_cost_usd"),
        }
        if observed != expected:
            return False
    return True


def _freeze_json(path: Path, value: dict[str, Any]) -> None:
    payload = canonical_bytes(value)
    if path.exists():
        try:
            existing = canonical_bytes(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            raise CampaignError(f"existing artifact is corrupt: {path}") from exc
        if existing != payload:
            raise CampaignError(f"existing artifact does not match campaign plan: {path}")
        return
    _write_bytes(path, payload)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    _write_bytes(path, canonical_bytes(value))


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _build_inventory(root: Path) -> dict[str, Any]:
    paths = [
        root / "manifest.json",
        root / "preflight.json",
        root / "report.json",
        root / "raw-outcomes.jsonl",
        root / "aggregate.json",
        root / "verifier-report.json",
        root / "claim-eligibility.json",
        *sorted((root / "trials").glob("*.json")),
    ]
    artifacts = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        artifacts.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": digest,
                "bytes": path.stat().st_size,
            }
        )
    payload = {
        "schema": "codeflow.codeflowbench.call-efficiency-cost.inventory.v1",
        "artifacts": artifacts,
    }
    payload["inventory_digest"] = canonical_digest(payload)
    return payload


__all__ = [
    "ProviderCallRecord",
    "TrialArtifact",
    "execute_live_trial",
    "load_deepseek_key",
    "run_cache_preflight",
    "run_formal_campaign",
    "verify_retained_campaign",
]
