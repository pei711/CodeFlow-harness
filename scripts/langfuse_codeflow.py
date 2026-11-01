"""Time a real CodeFlow CLI task and export its historical spans to Langfuse.

Run this script with the separate Langfuse SDK environment. Credentials are
read only by the exporter, never injected into the measured Agent process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / ".codeflow" / "langfuse"


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sanitize(value):
    """Remove credential-like values before uploading local attributes/previews."""
    if isinstance(value, dict):
        return {
            k: "[REDACTED]" if re.search(r"secret|api.?key|authorization|password", k, re.I) else sanitize(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        return re.sub(r"\b(?:sk|pk)-[A-Za-z0-9_-]{8,}", "[REDACTED]", value)
    return value


def load_spans(path):
    # A turn checkpoint and its terminal event reuse the same span ID.
    # Import only the last event, preserving original IDs and timestamps.
    unique = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        span = json.loads(line)
        unique[span["spanId"]] = span
    return sorted(unique.values(), key=lambda s: (s["startTime"], s["spanId"]))


def stable_id(value, length=32):
    return hashlib.sha256(value.encode()).hexdigest()[:length]


def make_events(result, spans):
    """Use the SDK ingestion API so import time never replaces execution time."""
    from langfuse.api import (
        CreateGenerationBody,
        CreateSpanBody,
        IngestionEvent_GenerationCreate,
        IngestionEvent_ScoreCreate,
        IngestionEvent_SpanCreate,
        IngestionEvent_TraceCreate,
        ScoreBody,
        TraceBody,
    )

    run_id = result["run_id"]
    trace_id = stable_id("codeflow:" + run_id)
    wall_id = stable_id(run_id + ":wall", 16)
    stamp = now()
    events = [
        IngestionEvent_TraceCreate(
            id=stable_id(run_id + ":trace-event"),
            timestamp=stamp,
            body=TraceBody(
                id=trace_id,
                timestamp=result["started_at"],
                name="CodeFlow task runtime",
                input=result["task"],
                output=sanitize(result.get("actual_artifact")),
                environment="development",
                tags=["codeflow", "runtime-measurement"],
                metadata={
                    **{
                        k: result[k]
                        for k in [
                            "run_id",
                            "task_id",
                            "commit",
                            "model",
                            "task_latency_ms",
                            "return_code",
                            "timed_out",
                            "verifier_passed",
                        ]
                    },
                    "working_tree_dirty": result.get("working_tree_dirty", "unknown"),
                    **{
                        k: result[k]
                        for k in ["scenario", "pair_id", "arm", "study_id", "usage_complete"]
                        if k in result
                    },
                    **sanitize(result.get("evidence_metadata", {})),
                },
            ),
        )
    ]
    events.append(
        IngestionEvent_SpanCreate(
            id=stable_id(run_id + ":wall-event"),
            timestamp=stamp,
            body=CreateSpanBody(
                id=wall_id,
                trace_id=trace_id,
                name="codeflow.cli.end_to_end",
                start_time=result["started_at"],
                end_time=result["finished_at"],
                input=result["task"],
                output=sanitize(result.get("actual_artifact")),
                level="DEFAULT" if result["return_code"] == 0 else "ERROR",
                metadata={
                    "includes": "CLI startup, runtime, model/tool calls, process exit",
                    "excludes": "fixture preparation, external verification, Langfuse upload",
                },
            ),
        )
    )
    ids = {s["spanId"] for s in spans}
    for span in spans:
        attrs = sanitize(span.get("attributes", {}))
        kind = attrs.get("span.type")
        prefix = "llm" if kind == "model" else "tool" if kind == "tool" else "turn"
        parent = span.get("parentSpanId")
        common = dict(
            id=stable_id(run_id + ":" + span["spanId"], 16),
            trace_id=trace_id,
            parent_observation_id=stable_id(run_id + ":" + parent, 16) if parent in ids else wall_id,
            name=span["name"],
            start_time=span["startTime"],
            end_time=span["endTime"],
            input=attrs.get(prefix + ".input_preview"),
            output=attrs.get(prefix + ".output_preview"),
            metadata=attrs,
            level="ERROR" if span.get("status", {}).get("code") == "ERROR" else "DEFAULT",
            status_message=sanitize(span.get("status", {}).get("message")),
        )
        event_id = stable_id(run_id + ":span-event:" + span["spanId"])
        if kind == "model":
            usage = {k: attrs.get("llm.usage." + k + "_tokens") for k in ["input", "output"]}
            # CodeFlow normalizes input into uncached + cache-read/write parts;
            # Langfuse's input total must include all actual prompt tokens.
            if isinstance(usage["input"], int):
                usage["input"] += sum(
                    attrs.get("llm.usage." + key) or 0 for key in ["cache_read_tokens", "cache_write_tokens"]
                )
            usage = {k: v for k, v in usage.items() if isinstance(v, int)}
            events.append(
                IngestionEvent_GenerationCreate(
                    id=event_id,
                    timestamp=stamp,
                    body=CreateGenerationBody(**common, model=attrs.get("llm.model"), usage_details=usage or None),
                )
            )
        else:
            events.append(
                IngestionEvent_SpanCreate(
                    id=event_id,
                    timestamp=stamp,
                    body=CreateSpanBody(**common),
                )
            )
    scores = [("task_latency_ms", result["task_latency_ms"], "NUMERIC")]
    if result["verifier_passed"] is not None:
        scores.append(("artifact_exact_match", int(result["verifier_passed"]), "BOOLEAN"))
    for name, value in result.get("evidence_scores", {}).items():
        if isinstance(value, bool):
            scores.append((name, int(value), "BOOLEAN"))
        elif isinstance(value, (int, float)):
            import math

            if math.isfinite(value):
                scores.append((name, value, "NUMERIC"))
    for name, value, data_type in scores:
        events.append(
            IngestionEvent_ScoreCreate(
                id=stable_id(run_id + ":score-event:" + name),
                timestamp=stamp,
                body=ScoreBody(
                    id=stable_id(run_id + ":score:" + name),
                    trace_id=trace_id,
                    name=name,
                    value=value,
                    data_type=data_type,
                    comment=(
                        "Recorded diagnostic; cost fields are estimates, not invoices."
                        if name in result.get("evidence_scores", {})
                        else "Measured externally; independent of Agent self-report."
                    ),
                ),
            )
        )
    return trace_id, events


def credentials(path):
    values = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    for key in ["LANGFUSE_BASE_URL", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]:
        values[key] = os.environ.get(key) or values.get(key)
        if not values[key]:
            raise ValueError("Missing configuration: " + key)
    return values


def send_historical_spans(client, tracer, generator, events, *, flush=True, experiment_attributes=None):
    """OTel transports trace writes; public SDK wrappers encode Langfuse fields."""
    from langfuse import LangfuseGeneration, LangfuseSpan
    from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags, set_span_in_context

    trace_record = events[0].body
    trace_id = trace_record.id
    for event in events:
        if event.type not in {"span-create", "generation-create"}:
            continue
        body = event.body
        generator.next_span_id = int(body.id, 16)
        context = None
        if body.parent_observation_id:
            parent = SpanContext(
                int(trace_id, 16),
                int(body.parent_observation_id, 16),
                is_remote=False,
                trace_flags=TraceFlags(TraceFlags.SAMPLED),
            )
            context = set_span_in_context(NonRecordingSpan(parent))
        start_ns = int(body.start_time.timestamp() * 1_000_000_000)
        end_ns = int(body.end_time.timestamp() * 1_000_000_000)
        otel_span = tracer.start_span(body.name, context=context, start_time=start_ns)
        if not body.parent_observation_id:
            otel_span.set_attribute("langfuse.trace.name", trace_record.name)
        common = dict(
            otel_span=otel_span,
            langfuse_client=client,
            input=body.input,
            output=body.output,
            metadata=body.metadata,
            level=body.level,
            status_message=body.status_message,
            environment="sdk-experiment" if experiment_attributes else "development",
        )
        if event.type == "generation-create":
            wrapper = LangfuseGeneration(**common, model=body.model, usage_details=body.usage_details)
        else:
            if not body.parent_observation_id:
                common["metadata"] = {**(body.metadata or {}), **trace_record.metadata}
            wrapper = LangfuseSpan(**common)
        if experiment_attributes:
            otel_span.set_attributes(experiment_attributes)
        wrapper.end(end_time=end_ns)
    for event in events:
        if event.type == "score-create":
            body = event.body
            client.create_score(
                name=body.name,
                value=body.value,
                trace_id=trace_id,
                score_id=body.id,
                data_type=body.data_type,
                comment=body.comment,
            )
    if flush:
        client.flush()


def upload(result_path, env_file, verify_seconds=35):
    from langfuse import Langfuse
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.id_generator import IdGenerator

    class HistoricalIds(IdGenerator):
        next_span_id = 1

        def generate_trace_id(self):
            return int(trace_id, 16)

        def generate_span_id(self):
            return self.next_span_id

    result = json.loads(result_path.read_text(encoding="utf-8"))
    trace_path = result_path.parent / "trace" / "logs" / "audit-spans.log"
    spans = load_spans(trace_path) if trace_path.is_file() else []
    trace_id, events = make_events(result, spans)
    cfg = credentials(env_file)
    generator = HistoricalIds()
    provider = TracerProvider(id_generator=generator)
    client = Langfuse(
        public_key=cfg["LANGFUSE_PUBLIC_KEY"],
        secret_key=cfg["LANGFUSE_SECRET_KEY"],
        base_url=cfg["LANGFUSE_BASE_URL"],
        tracer_provider=provider,
        timeout=20,
    )
    tracer = provider.get_tracer("langfuse-sdk", "4.18.0", attributes={"public_key": cfg["LANGFUSE_PUBLIC_KEY"]})
    started = time.perf_counter()
    receipt = {
        "trace_id": trace_id,
        "observation_count": len(spans) + 1,
        "event_count": len(events),
        "sdk_version": "4.18.0",
        "server_verified": False,
    }
    try:
        # Auth check is explicit; SDK transport success alone is not proof of ingestion.
        project_data = client.api.projects.get()
        project = project_data.data[0]
        receipt["trace_url"] = cfg["LANGFUSE_BASE_URL"].rstrip("/") + f"/project/{project.id}/traces/{trace_id}"
        send_historical_spans(client, tracer, generator, events)
        receipt["upload_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        deadline = time.monotonic() + verify_seconds
        while time.monotonic() < deadline:
            try:
                remote = client.api.observations.get_many(trace_id=trace_id, limit=100, fields="core,basic,usage")
                receipt["remote_observation_count"] = len(remote.data)
                scores = client.api.scores_v3.get_many_v3(trace_id=trace_id, limit=100)
                receipt["remote_score_count"] = len(scores.data)
                if receipt["remote_observation_count"] >= receipt["observation_count"] and receipt[
                    "remote_score_count"
                ] >= (2 if result["verifier_passed"] is not None else 1):
                    receipt["server_verified"] = True
                    break
            except Exception:
                pass  # Ingestion is eventually consistent; verify with bounded polling.
            time.sleep(2)
    except Exception as exc:
        receipt["error"] = sanitize(type(exc).__name__ + ": " + str(exc))
    finally:
        client.shutdown()
        receipt["upload_and_verification_ms"] = round((time.perf_counter() - started) * 1000, 2)
        save(result_path.parent / "langfuse-receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["server_verified"] else 2


def run_task(args):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    directory = LOCAL / "runs" / run_id
    workspace = directory / "workspace"
    workspace.mkdir(parents=True)
    # This synthetic fixture intentionally contains no user/repository data.
    save(workspace / "numbers.json", {"numbers": [1, 2, 3, 4, 5, 6]})
    task = (
        "Read numbers.json. Count its even numbers and calculate their sum. "
        'Write result.json containing exactly {"count": <integer>, "sum": <integer>}. '
        "Use the file tools. Then briefly report completion. Do not use subagents or network tools."
    )
    original = json.loads(args.config.read_text(encoding="utf-8"))
    defaults = original.setdefault("agents", {}).setdefault("defaults", {})
    defaults["workspace"] = str(workspace)
    defaults["maxTokens"] = 1024
    defaults["maxToolIterations"] = 6
    tools = original.setdefault("tools", {})
    tools["mcpServers"] = {}
    tools["restrictToWorkspace"] = True
    tools["disabledTools"] = ["spawn", "exec", "web_search", "web_fetch", "message", "cron"]
    original["memory"] = {"backend": None}
    config = directory / "home" / "config.json"
    config.parent.mkdir()
    save(config, original)
    env = {k: v for k, v in os.environ.items() if not k.startswith("LANGFUSE_")}
    env.update(
        CODEFLOW_HOME=str(config.parent),
        CODEFLOW_TRACING="1",
        CODEFLOW_TRACING_DIR=str(directory / "trace"),
        PYTHONIOENCODING="utf-8",
    )
    command = [
        str(args.runtime_python),
        "-m",
        "codeflow",
        "run",
        "--config",
        str(config),
        "--workspace",
        str(workspace),
        "--message",
        task,
        "--no-markdown",
        "--no-logs",
    ]
    started_at = now()
    started = time.perf_counter()
    timed_out = False
    with (
        (directory / "stdout.txt").open("w", encoding="utf-8") as stdout,
        (directory / "stderr.txt").open("w", encoding="utf-8") as stderr,
    ):
        try:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=args.timeout)
            return_code = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            return_code = None
    elapsed = round((time.perf_counter() - started) * 1000, 2)
    finished_at = now()
    actual = None
    try:
        actual = json.loads((workspace / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    result = dict(
        schema="codeflow.langfuse-runtime.v1",
        run_id=run_id,
        task_id="even-number-artifact-v1",
        task=task,
        commit=commit,
        model=defaults.get("model"),
        started_at=started_at,
        finished_at=finished_at,
        task_latency_ms=elapsed,
        return_code=return_code,
        timed_out=timed_out,
        verifier_passed=actual == {"count": 3, "sum": 12},
        actual_artifact=actual,
        working_tree_dirty=dirty,
        note="Single real-provider task; timing includes CLI startup/exit, excludes upload and external verifier.",
    )
    save(directory / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("result_path=" + str(directory / "result.json"))
    if args.upload:
        upload_code = upload(directory / "result.json", args.env_file)
        return upload_code or (0 if result["verifier_passed"] and return_code == 0 else 1)
    return 0 if result["verifier_passed"] and return_code == 0 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=LOCAL / "langfuse.env")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run a tiny real-provider file task and measure wall time")
    run.add_argument("--runtime-python", type=Path, default=ROOT / ".venv" / "Scripts" / "python.exe")
    run.add_argument("--config", type=Path, default=Path.home() / ".codeflow" / "config.json")
    run.add_argument("--timeout", type=int, default=180)
    run.add_argument("--upload", action="store_true")
    export = sub.add_parser("upload", help="Upload or retry a previously measured run")
    export.add_argument("result", type=Path)
    args = parser.parse_args()
    return run_task(args) if args.command == "run" else upload(args.result.resolve(), args.env_file)


if __name__ == "__main__":
    raise SystemExit(main())
