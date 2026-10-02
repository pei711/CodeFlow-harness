from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.codeflowbench.canonical import canonical_bytes, canonical_digest
from codeflow.agent.tools.base import Tool
from codeflow.config.paths import RuntimePaths
from codeflow.config.codeflow import CodeFlowConfig
from codeflow.config.schema import Config
from codeflow.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from codeflow.spine import ChatType, Origin, Source, Text, TurnRequest
from codeflow.tracing import spans as tracing_spans

MANIFEST_SCHEMA = "codeflow.codeflowbench.tracing-overhead.manifest.v1"
BLOCK_SCHEMA = "codeflow.codeflowbench.tracing-overhead.block.v1"
AGGREGATE_SCHEMA = "codeflow.codeflowbench.tracing-overhead.aggregate.v1"
MODEL = "benchmark/tracing-local-v1"


class TracingExperimentError(RuntimeError):
    pass


@dataclass(frozen=True)
class TracingExperimentConfig:
    blocks: int = 20
    turns_per_block: int = 50
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20_260_813

    def __post_init__(self) -> None:
        if min(self.blocks, self.turns_per_block, self.bootstrap_samples) < 1:
            raise ValueError("tracing experiment counts must be positive")

    @property
    def pairs(self) -> int:
        return self.blocks * self.turns_per_block


class _TracingProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def get_default_model(self) -> str:
        return MODEL

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        **_kwargs: Any,
    ) -> LLMResponse:
        self.calls += 1
        usage = {
            "prompt_tokens": 128,
            "completion_tokens": 8,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 128,
        }
        if messages and messages[-1].get("role") == "tool":
            return LLMResponse(content="TRACE_OK", usage=usage, model=model or MODEL)
        return LLMResponse(
            content=None,
            finish_reason="tool_calls",
            tool_calls=[
                ToolCallRequest(
                    id=f"trace-call-{self.calls}",
                    name="trace_lookup",
                    arguments={"key": "stable"},
                )
            ],
            usage=usage,
            model=model or MODEL,
        )


class _TraceLookup(Tool):
    name = "trace_lookup"
    description = "Return the stable tracing benchmark value."
    parameters = {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    }

    async def execute(self, key: str, **_kwargs: Any) -> str:
        return f"value:{key}"


async def run_campaign(
    *,
    output_root: Path,
    codeflow_commit: str,
    config: TracingExperimentConfig,
) -> dict[str, Any]:
    manifest = _manifest(codeflow_commit, config)
    plan_digest = manifest["plan_digest"]
    output_root.mkdir(parents=True, exist_ok=True)
    _freeze_json(output_root / "manifest.json", manifest)
    blocks_dir = output_root / "blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)

    blocks: list[dict[str, Any]] = []
    for block_index in range(config.blocks):
        block_path = blocks_dir / f"block-{block_index:03d}.json"
        if block_path.exists():
            block = _read_json(block_path, "retained tracing block is unreadable")
            if block.get("plan_digest") != plan_digest:
                raise TracingExperimentError("retained tracing block has a different plan")
        else:
            block = await _run_block(
                output_root=output_root,
                block_index=block_index,
                turns=config.turns_per_block,
                plan_digest=plan_digest,
            )
            _freeze_json(block_path, block)
        blocks.append(block)

    aggregate = build_aggregate(config=config, blocks=blocks)
    aggregate["codeflow_commit"] = codeflow_commit
    aggregate["plan_digest"] = plan_digest
    aggregate["aggregate_digest"] = canonical_digest(
        {key: value for key, value in aggregate.items() if key != "aggregate_digest"}
    )
    _write_json(output_root / "aggregate.json", aggregate)
    return verify_campaign(
        output_root=output_root,
        expected_codeflow_commit=codeflow_commit,
    )


def verify_campaign(*, output_root: Path, expected_codeflow_commit: str) -> dict[str, Any]:
    manifest = _read_json(output_root / "manifest.json", "tracing manifest is unreadable")
    retained_aggregate = _read_json(output_root / "aggregate.json", "tracing aggregate is unreadable")
    try:
        config = TracingExperimentConfig(**manifest["config"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TracingExperimentError("tracing manifest config is invalid") from exc
    plan_digest = str(manifest.get("plan_digest") or "")
    manifest_payload = {key: value for key, value in manifest.items() if key != "plan_digest"}
    block_paths = sorted((output_root / "blocks").glob("block-*.json"))
    blocks = [_read_json(path, "retained tracing block is unreadable") for path in block_paths]
    trace_receipts = [_verify_trace_receipts(output_root, block) for block in blocks]
    checks = {
        "manifest_schema": manifest.get("schema") == MANIFEST_SCHEMA,
        "manifest_digest": plan_digest == canonical_digest(manifest_payload),
        "codeflow_commit": manifest.get("codeflow_commit") == expected_codeflow_commit,
        "block_count": len(blocks) == config.blocks,
        "block_indices": [block.get("block_index") for block in blocks] == list(range(config.blocks)),
        "block_schema": all(block.get("schema") == BLOCK_SCHEMA for block in blocks),
        "block_plan_binding": all(block.get("plan_digest") == plan_digest for block in blocks),
        "paired_turn_receipts": all(_block_pairs_complete(block) for block in blocks),
        "trace_receipts": all(trace_receipts),
    }
    rebuilt = build_aggregate(config=config, blocks=blocks)
    rebuilt["codeflow_commit"] = expected_codeflow_commit
    rebuilt["plan_digest"] = plan_digest
    rebuilt["aggregate_digest"] = canonical_digest(
        {key: value for key, value in rebuilt.items() if key != "aggregate_digest"}
    )
    checks["aggregate_rebuilt"] = canonical_bytes(rebuilt) == canonical_bytes(retained_aggregate)
    verifier = {
        "schema": "codeflow.codeflowbench.tracing-overhead.verifier.v1",
        "checks": checks,
        "pairs": rebuilt["summary"]["pairs"],
        "aggregate_digest": rebuilt["aggregate_digest"],
        "passed": all(checks.values()) and rebuilt["measurement_valid"],
    }
    verifier["verifier_digest"] = canonical_digest(verifier)
    raw_rows = [pair for block in blocks for pair in block["pairs"]]
    _write_bytes(
        output_root / "raw-outcomes.jsonl",
        b"".join(canonical_bytes(row) + b"\n" for row in raw_rows),
    )
    _write_json(output_root / "aggregate.json", rebuilt)
    _write_json(output_root / "verifier-report.json", verifier)
    _write_json(
        output_root / "claim-eligibility.json",
        {
            "schema": "codeflow.codeflowbench.tracing-overhead.claim.v1",
            "measurement_valid": verifier["passed"],
            "cv_metrics_eligible": verifier["passed"],
            "gates": rebuilt["gates"],
            "p95_overhead_percent": rebuilt["summary"]["p95_overhead_percent"],
            "p95_overhead_95_ci": rebuilt["summary"]["p95_overhead_95_ci"],
            "bytes_per_traced_turn": rebuilt["summary"]["bytes_per_traced_turn"],
            "aggregate_digest": rebuilt["aggregate_digest"],
        },
    )
    _write_json(output_root / "inventory.json", _inventory(output_root))
    if not verifier["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        if not rebuilt["measurement_valid"]:
            failed.extend(name for name, passed in rebuilt["gates"].items() if not passed)
        raise TracingExperimentError(f"tracing verification failed: {', '.join(failed)}")
    return verifier


def build_aggregate(
    *,
    config: TracingExperimentConfig,
    blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    pairs = [pair for block in blocks for pair in block.get("pairs", [])]
    off_ms = [pair["tracing_off"]["latency_ns"] / 1_000_000 for pair in pairs]
    on_ms = [pair["tracing_on"]["latency_ns"] / 1_000_000 for pair in pairs]
    off_p50 = _nearest_rank(off_ms, 0.50)
    on_p50 = _nearest_rank(on_ms, 0.50)
    off_p95 = _nearest_rank(off_ms, 0.95)
    on_p95 = _nearest_rank(on_ms, 0.95)
    interval = _clustered_p95_overhead_interval(blocks, config)
    on_trace_summaries = [block["arms"]["tracing_on"]["trace_summary"] for block in blocks]
    total_trace_bytes = sum(summary["bytes"] for summary in on_trace_summaries)
    gates = {
        "exact_pair_count": len(pairs) == config.pairs,
        "balanced_arm_order": abs(
            sum(block.get("arm_order", [None])[0] == "tracing_on" for block in blocks)
            - sum(block.get("arm_order", [None])[0] == "tracing_off" for block in blocks)
        )
        <= 1,
        "both_arms_correct": bool(pairs)
        and all(
            pair[arm]["status"] == "completed" and pair[arm]["reply"] == "TRACE_OK" and pair[arm]["provider_calls"] == 2
            for pair in pairs
            for arm in ("tracing_off", "tracing_on")
        ),
        "disabled_arm_emits_no_trace": all(
            block["arms"]["tracing_off"]["trace_summary"]["bytes"] == 0 for block in blocks
        ),
        "enabled_turn_correlation_complete": bool(blocks)
        and all(summary["correlation_complete"] for summary in on_trace_summaries),
        "one_trace_per_enabled_turn": bool(blocks)
        and all(summary["root_spans"] == config.turns_per_block for summary in on_trace_summaries),
    }
    return {
        "schema": AGGREGATE_SCHEMA,
        "evidence_scope": "deterministic_real_runtime_tracing_on_off",
        "config": asdict(config),
        "gates": gates,
        "measurement_valid": all(gates.values()),
        "summary": {
            "pairs": len(pairs),
            "turns": len(pairs) * 2,
            "tracing_off_p50_ms": off_p50,
            "tracing_on_p50_ms": on_p50,
            "p50_overhead_percent": _overhead_percent(on_p50, off_p50),
            "tracing_off_p95_ms": off_p95,
            "tracing_on_p95_ms": on_p95,
            "p95_overhead_percent": _overhead_percent(on_p95, off_p95),
            "p95_overhead_95_ci": interval,
            "trace_bytes": total_trace_bytes,
            "bytes_per_traced_turn": total_trace_bytes / len(on_ms) if on_ms else 0,
            "root_spans": sum(summary["root_spans"] for summary in on_trace_summaries),
            "span_count": sum(summary["span_count"] for summary in on_trace_summaries),
        },
    }


async def _run_block(
    *,
    output_root: Path,
    block_index: int,
    turns: int,
    plan_digest: str,
) -> dict[str, Any]:
    arm_order = ("tracing_off", "tracing_on") if block_index % 2 == 0 else ("tracing_on", "tracing_off")
    arms: dict[str, dict[str, Any]] = {}
    for arm in arm_order:
        trace_dir = output_root / "traces" / f"block-{block_index:03d}" / arm
        if trace_dir.exists():
            raise TracingExperimentError(f"incomplete tracing evidence root must not be overwritten: {trace_dir}")
        arms[arm] = await _run_arm(
            enabled=arm == "tracing_on",
            trace_dir=trace_dir,
            block_index=block_index,
            turns=turns,
        )
    pairs = []
    by_arm = {arm: {turn["turn_id"]: turn for turn in arms[arm]["turns"]} for arm in arm_order}
    for turn_index in range(turns):
        turn_id = f"block-{block_index:03d}-turn-{turn_index:03d}"
        pairs.append(
            {
                "turn_id": turn_id,
                "tracing_off": by_arm["tracing_off"][turn_id],
                "tracing_on": by_arm["tracing_on"][turn_id],
            }
        )
    return {
        "schema": BLOCK_SCHEMA,
        "plan_digest": plan_digest,
        "block_index": block_index,
        "arm_order": list(arm_order),
        "arms": arms,
        "pairs": pairs,
    }


async def _run_arm(
    *,
    enabled: bool,
    trace_dir: Path,
    block_index: int,
    turns: int,
) -> dict[str, Any]:
    previous_tracing = os.environ.get("CODEFLOW_TRACING")
    previous_dir = os.environ.get("CODEFLOW_TRACING_DIR")
    os.environ["CODEFLOW_TRACING"] = "1" if enabled else "0"
    os.environ["CODEFLOW_TRACING_DIR"] = str(trace_dir)
    tracing_spans._store = None
    try:
        with tempfile.TemporaryDirectory(prefix="codeflow-tracing-benchmark-") as tmp:
            root = Path(tmp)
            _seed_workspace(root)
            provider = _TracingProvider()
            runtime = _assemble_runtime(root, provider)
            runtime.agent_loop.tools._tools.clear()
            runtime.agent_loop.tools.register(_TraceLookup())
            turns_out: list[dict[str, Any]] = []
            try:
                for turn_index in range(turns):
                    turn_id = f"block-{block_index:03d}-turn-{turn_index:03d}"
                    conversation = f"trace:{turn_id}"
                    request = TurnRequest(
                        text="Call trace_lookup once, then reply only TRACE_OK.",
                        source=Source(
                            channel="codeflowbench",
                            sender_id="tracing-benchmark",
                            chat_id=conversation,
                            chat_type=ChatType.DM,
                        ),
                        message_id=turn_id,
                        conversation=conversation,
                        origin=Origin.USER,
                    )
                    emitted: list[Any] = []

                    async def emit(event: Any) -> None:
                        emitted.append(event)

                    text_sink: dict[str, Any] = {}
                    before_calls = provider.calls
                    started = time.perf_counter_ns()
                    outcome = await runtime.agent_loop.run_turn(
                        request,
                        emit,
                        lambda: [],
                        stream=False,
                        text_sink=text_sink,
                    )
                    latency_ns = time.perf_counter_ns() - started
                    reply = text_sink.get("text")
                    if reply is None:
                        reply = next((event.content for event in emitted if isinstance(event, Text)), None)
                    turns_out.append(
                        {
                            "turn_id": turn_id,
                            "conversation": conversation,
                            "latency_ns": latency_ns,
                            "status": (
                                "completed"
                                if outcome.explicit_reply and outcome.tool_calls == 1 and outcome.tool_failures == 0
                                else "failed"
                            ),
                            "reply": reply,
                            "provider_calls": provider.calls - before_calls,
                        }
                    )
            finally:
                await runtime.close()
        trace_summary = _trace_summary(trace_dir, expected={turn["conversation"] for turn in turns_out})
        return {
            "enabled": enabled,
            "turns": turns_out,
            "trace_summary": trace_summary,
            "trace_files": _file_receipts(trace_dir),
        }
    finally:
        tracing_spans._store = None
        _restore_env("CODEFLOW_TRACING", previous_tracing)
        _restore_env("CODEFLOW_TRACING_DIR", previous_dir)


def _assemble_runtime(root: Path, provider: _TracingProvider):
    from codeflow.cli._runtime_assembly import assemble_runtime

    config = Config()
    config.agents.defaults.workspace = str(root)
    config.agents.defaults.model = MODEL
    config.agents.defaults.max_tool_iterations = 3
    config.agents.defaults.enable_personalization = False
    config.tools.mcp_servers = {}
    config.tools.sandbox.backend = "none"
    codeflow_config = CodeFlowConfig(base=config)
    codeflow_config.memory.backend = None
    codeflow_config.skill_forge.enabled = False
    codeflow_config.skill_forge.router.enabled = False
    codeflow_config.skill_forge.rewrite_enabled = False
    codeflow_config.skill_forge.llm_gate_enabled = False
    codeflow_config.call_efficiency.mode = "off"
    codeflow_config.call_efficiency.enabled = False
    codeflow_config.call_efficiency.usage_tracking = False
    codeflow_config.runtime.checkpoint.policy = "never"
    return assemble_runtime(
        config,
        codeflow_config,
        provider=provider,
        cron_service=None,
        interactive=False,
        paths=RuntimePaths(workspace=root, state=root / "state"),
    )


def _seed_workspace(root: Path) -> None:
    agent_profile = root / "state" / "agent_memory" / "profile"
    user_profile = root / "state" / "user_memory" / "profile"
    agent_profile.mkdir(parents=True, exist_ok=True)
    user_profile.mkdir(parents=True, exist_ok=True)
    (agent_profile / "soul.md").write_text(
        "# Tracing benchmark\nCall trace_lookup exactly once and follow the requested reply format.\n",
        encoding="utf-8",
    )
    (agent_profile / "agent.md").write_text("# Agent\nFollow the benchmark protocol.\n", encoding="utf-8")
    (user_profile / "user.md").write_text("# User\nBenchmark operator.\n", encoding="utf-8")
    (root / "state" / "TOOLS.md").write_text("# Tools\nUse trace_lookup only.\n", encoding="utf-8")


def _trace_summary(trace_dir: Path, *, expected: set[str]) -> dict[str, Any]:
    log_path = trace_dir / "logs" / "audit-spans.log"
    if not log_path.is_file():
        return {
            "bytes": 0,
            "span_count": 0,
            "root_spans": 0,
            "unique_traces": 0,
            "correlation_complete": not expected,
        }
    spans = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    roots = [
        span
        for span in spans
        if span.get("name") == "session.turn"
        and span.get("parentSpanId") is None
        and span.get("attributes", {}).get("turn.in_progress") is False
    ]
    roots_by_session: dict[str, list[dict[str, Any]]] = {}
    for root in roots:
        session = root.get("attributes", {}).get("session.id")
        if isinstance(session, str):
            roots_by_session.setdefault(session, []).append(root)
    names_by_trace: dict[str, list[str]] = {}
    for span in spans:
        trace_id = span.get("traceId")
        name = span.get("name")
        if isinstance(trace_id, str) and isinstance(name, str):
            names_by_trace.setdefault(trace_id, []).append(name)
    correlation_complete = set(roots_by_session) == expected
    for session in expected:
        session_roots = roots_by_session.get(session, [])
        if len(session_roots) != 1:
            correlation_complete = False
            continue
        names = names_by_trace.get(session_roots[0].get("traceId"), [])
        if names.count("llm.call") != 2 or names.count("tool.call") != 1:
            correlation_complete = False
    return {
        "bytes": sum(path.stat().st_size for path in trace_dir.rglob("*") if path.is_file()),
        "span_count": len(spans),
        "root_spans": len(roots),
        "unique_traces": len({root.get("traceId") for root in roots}),
        "correlation_complete": correlation_complete,
    }


def _clustered_p95_overhead_interval(
    blocks: list[dict[str, Any]],
    config: TracingExperimentConfig,
) -> dict[str, Any]:
    if not blocks:
        return {"lower": 0.0, "upper": 0.0, "samples": config.bootstrap_samples, "seed": config.bootstrap_seed}
    per_block = [block.get("pairs", []) for block in blocks]
    rng = random.Random(config.bootstrap_seed)
    distribution: list[float] = []
    for _ in range(config.bootstrap_samples):
        sampled = [rng.choice(per_block) for _ in per_block]
        pairs = [pair for cluster in sampled for pair in cluster]
        off = [pair["tracing_off"]["latency_ns"] / 1_000_000 for pair in pairs]
        on = [pair["tracing_on"]["latency_ns"] / 1_000_000 for pair in pairs]
        distribution.append(_overhead_percent(_nearest_rank(on, 0.95), _nearest_rank(off, 0.95)))
    distribution.sort()
    return {
        "lower": _nearest_rank(distribution, 0.025),
        "upper": _nearest_rank(distribution, 0.975),
        "samples": config.bootstrap_samples,
        "seed": config.bootstrap_seed,
        "unit": "block_clustered_p95_ratio",
        "blocks": len(blocks),
    }


def _nearest_rank(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _overhead_percent(on: float, off: float) -> float:
    return (on - off) / off * 100 if off > 0 else 0.0


def _manifest(codeflow_commit: str, config: TracingExperimentConfig) -> dict[str, Any]:
    if len(codeflow_commit) != 40:
        raise TracingExperimentError("codeflow_commit must be a full Git commit")
    payload = {
        "schema": MANIFEST_SCHEMA,
        "codeflow_commit": codeflow_commit,
        "config": asdict(config),
        "model": MODEL,
        "runtime_boundary": "shared_runtime_assembly_agent_loop",
        "treatment": "CODEFLOW_TRACING",
        "workload": "one_tool_two_provider_attempt_turn",
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    payload["plan_digest"] = canonical_digest(payload)
    return payload


def _verify_trace_receipts(output_root: Path, block: dict[str, Any]) -> bool:
    try:
        block_index = int(block.get("block_index", -1))
    except (TypeError, ValueError):
        return False
    for arm in ("tracing_off", "tracing_on"):
        arm_payload = block.get("arms", {}).get(arm, {})
        expected = arm_payload.get("trace_files", {})
        trace_dir = output_root / "traces" / f"block-{block_index:03d}" / arm
        if _file_receipts(trace_dir) != expected:
            return False
        conversations = {
            turn.get("conversation")
            for turn in arm_payload.get("turns", [])
            if isinstance(turn, dict) and isinstance(turn.get("conversation"), str)
        }
        try:
            rebuilt_summary = _trace_summary(trace_dir, expected=conversations)
        except (OSError, json.JSONDecodeError):
            return False
        if rebuilt_summary != arm_payload.get("trace_summary"):
            return False
    return True


def _block_pairs_complete(block: dict[str, Any]) -> bool:
    arms = block.get("arms")
    if not isinstance(arms, dict):
        return False
    by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ("tracing_off", "tracing_on"):
        turns = arms.get(arm, {}).get("turns", [])
        if not isinstance(turns, list):
            return False
        by_arm[arm] = {
            turn["turn_id"]: turn for turn in turns if isinstance(turn, dict) and isinstance(turn.get("turn_id"), str)
        }
        if len(by_arm[arm]) != len(turns):
            return False
    if set(by_arm["tracing_off"]) != set(by_arm["tracing_on"]):
        return False
    rebuilt = [
        {
            "turn_id": turn_id,
            "tracing_off": by_arm["tracing_off"][turn_id],
            "tracing_on": by_arm["tracing_on"][turn_id],
        }
        for turn_id in sorted(by_arm["tracing_off"])
    ]
    return canonical_bytes(rebuilt) == canonical_bytes(block.get("pairs"))


def _file_receipts(root: Path) -> dict[str, dict[str, Any]]:
    if not root.is_dir():
        return {}
    receipts = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        receipts[path.relative_to(root).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return receipts


def _inventory(output_root: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(item for item in output_root.rglob("*") if item.is_file()):
        if path.name == "inventory.json":
            continue
        files[path.relative_to(output_root).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    payload = {
        "schema": "codeflow.codeflowbench.tracing-overhead.inventory.v1",
        "files": files,
    }
    payload["inventory_digest"] = canonical_digest(payload)
    return payload


def _freeze_json(path: Path, value: dict[str, Any]) -> None:
    payload = canonical_bytes(value)
    if path.exists():
        existing = _read_json(path, f"existing artifact is corrupt: {path}")
        if canonical_bytes(existing) != payload:
            raise TracingExperimentError(f"existing artifact does not match the campaign: {path}")
        return
    _write_bytes(path, payload)


def _read_json(path: Path, message: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TracingExperimentError(message) from exc
    if not isinstance(value, dict):
        raise TracingExperimentError(message)
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    _write_bytes(path, canonical_bytes(value))


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def _current_commit() -> str:
    git = shutil.which("git")
    if git is None:
        raise TracingExperimentError("git is required to freeze the CodeFlow commit")
    return subprocess.run(
        [git, "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[4],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the deterministic Runtime tracing overhead campaign.")
    parser.add_argument("mode", choices=("plan", "run", "verify"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(".codeflow/evidence/tracing-overhead-current"),
    )
    parser.add_argument("--codeflow-commit")
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--turns-per-block", type=int, default=50)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


async def _main() -> int:
    args = _parse_args()
    config = TracingExperimentConfig(
        blocks=args.blocks,
        turns_per_block=args.turns_per_block,
        bootstrap_samples=args.bootstrap_samples,
    )
    commit = args.codeflow_commit or _current_commit()
    if args.mode == "plan":
        result = _manifest(commit, config)
        result["pairs"] = config.pairs
        result["turns"] = config.pairs * 2
    elif args.mode == "run":
        result = await run_campaign(output_root=args.output_root, codeflow_commit=commit, config=config)
    else:
        result = verify_campaign(output_root=args.output_root, expected_codeflow_commit=commit)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(_main()))
    except TracingExperimentError as exc:
        raise SystemExit(f"Tracing campaign aborted: {exc}") from exc
