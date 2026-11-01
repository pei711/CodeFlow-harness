"""Upload the frozen paired study to a Langfuse dataset and two experiment runs."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from context_study_evidence import CRITERIA, digest, prepare, safe_path, write_report
from langfuse import Langfuse
from langfuse.api import IngestionEvent_ScoreCreate, ScoreBody
from langfuse_codeflow import (
    LOCAL,
    ROOT,
    credentials,
    load_spans,
    make_events,
    now,
    sanitize,
    save,
    send_historical_spans,
    stable_id,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.id_generator import IdGenerator


class HistoricalIds(IdGenerator):
    trace_id = 1
    next_span_id = 1

    def generate_trace_id(self):
        return self.trace_id

    def generate_span_id(self):
        return self.next_span_id


def expected_outputs(manifest):
    """Resolve by frozen digest, never by the current formal.json ordering."""
    expected = {}
    fixture_root = ROOT / "benchmarks/codeflowbench/tasks/context/expected"
    candidates = {
        digest(json.loads(p.read_text(encoding="utf-8"))): json.loads(p.read_text(encoding="utf-8"))
        for p in fixture_root.glob("*.json")
    }
    for task in manifest["tasks"]:
        checksum = task["expected_digest"]
        value = task.get("expected_output", candidates.get(checksum))
        if value is None or digest(value) != checksum:
            raise ValueError("Frozen expected output missing or digest mismatch: " + task["spec"]["task_id"])
        expected[task["spec"]["task_id"]] = value
    return expected


def export(root, revision="evidence-v1"):
    manifest, cells, evidence = prepare(root)
    expected = expected_outputs(manifest)  # Fail before any external writes.
    versions = evidence["versions"]
    export_version = digest(
        {
            "versions": versions,
            "rows": cells,
            "revision": revision,
            "exporter": digest(Path(__file__).read_text(encoding="utf-8")),
            "adapter": digest((ROOT / "scripts/langfuse_codeflow.py").read_text(encoding="utf-8")),
            "diagnostics": digest((ROOT / "scripts/context_study_evidence.py").read_text(encoding="utf-8")),
        }
    )
    evidence_by_key = {(t["pair_id"], t["arm"]): t for t in evidence["trials"]}
    cfg = credentials(LOCAL / "langfuse.env")
    ids = HistoricalIds()
    provider = TracerProvider(id_generator=ids)
    client = Langfuse(
        public_key=cfg["LANGFUSE_PUBLIC_KEY"],
        secret_key=cfg["LANGFUSE_SECRET_KEY"],
        base_url=cfg["LANGFUSE_BASE_URL"],
        tracer_provider=provider,
        timeout=25,
    )
    tracer = provider.get_tracer("langfuse-sdk", "4.18.0", attributes={"public_key": cfg["LANGFUSE_PUBLIC_KEY"]})
    records = []
    receipt = {
        "study_id": manifest["run_id"],
        "export_revision": revision,
        "export_version": export_version,
        "versions": versions,
        "traces": records,
        "server_verified": False,
    }
    receipt_path = root / ("langfuse-export-" + export_version[:16] + ".json")
    started = time.perf_counter()
    try:
        project = client.api.projects.get().data[0]
        project_base = cfg["LANGFUSE_BASE_URL"].rstrip("/") + "/project/" + project.id
        dataset_name = "CodeFlow-context-" + root.name + "-" + versions["dataset_digest"][:12] + "-" + revision
        dataset = client.api.datasets.create(
            name=dataset_name,
            description="15 paired tasks: long session, cross-turn dependency, forced compression degradation; one execution per condition.",
            metadata={
                "study_id": manifest["run_id"],
                "model": manifest["model"],
                "planned_trials": len(cells),
                "dataset_digest": versions["dataset_digest"],
            },
        )
        receipt.update(dataset_name=dataset_name, dataset_url=project_base + "/datasets/" + dataset.id)
        item_ids = {}
        for task in manifest["tasks"]:
            task_id = task["spec"]["task_id"]
            item = client.api.dataset_items.create(
                dataset_name=dataset_name,
                id=stable_id(dataset_name + ":" + task_id),
                input={
                    "history": task["history"],
                    "final_prompt": task["spec"]["payload"]["final_prompt"],
                    "output_schema": task["spec"]["payload"].get("output_schema"),
                    "output_contract": task["spec"]["payload"].get("output_contract"),
                },
                expected_output=expected[task_id],
                metadata={
                    "scenario": task["scenario"],
                    "case": task["case"],
                    "history_digest": task["spec"]["payload"]["history_digest"],
                    "pressure": manifest["scenarios"][task["scenario"]]["pressure"],
                },
            )
            item_ids[task_id] = item.id
        for index, row in enumerate(cells, 1):
            task = next(t for t in manifest["tasks"] if t["spec"]["task_id"] == row["pair_id"])
            detail = evidence_by_key[(row["pair_id"], row["arm"])]
            artifact_root = safe_path(root, row["artifact_refs"][0]) if row.get("artifact_refs") else None
            spans = []
            actual = None
            if artifact_root:
                trace_path = artifact_root / "traces/logs/audit-spans.log"
                if trace_path.is_file():
                    spans = load_spans(trace_path)
                try:
                    actual = json.loads(
                        safe_path(artifact_root / "workspace", task["spec"]["payload"]["artifact_path"]).read_text(
                            encoding="utf-8"
                        )
                    )
                except (OSError, ValueError):
                    pass
            result = dict(
                run_id=manifest["run_id"] + ":" + row["pair_id"] + ":" + row["arm"] + ":" + export_version,
                task_id=row["pair_id"],
                task=task["spec"]["payload"]["final_prompt"]
                + "\n\n"
                + task["spec"]["payload"].get("output_contract", ""),
                commit=manifest["commit"],
                working_tree_dirty=manifest["working_tree_dirty"],
                model=manifest["model"],
                task_latency_ms=row["wall_latency_ms"],
                started_at=row["started_at"],
                finished_at=row["finished_at"],
                return_code=0 if row["status"] == "passed" else 1,
                timed_out=row["status"] == "task_timeout",
                verifier_passed=None if row["verifier_state"] == "not_run" else row["verifier_state"] == "passed",
                actual_artifact=actual,
                scenario=row["scenario"],
                pair_id=row["pair_id"],
                arm=row["arm"],
                study_id=manifest["run_id"],
                usage_complete=row.get("usage_complete"),
                evidence_metadata={
                    **versions,
                    **detail["diagnostics"],
                    "accounting": detail["accounting"],
                    "export_version": export_version,
                },
                evidence_scores={
                    **{k: row[k] for k in CRITERIA if isinstance(row.get(k), bool)},
                    **{k: v for k, v in detail["accounting"].items() if isinstance(v, (int, float)) and v is not None},
                },
            )
            trace_id, events = make_events(result, spans)
            ids.trace_id = int(trace_id, 16)
            score_count = sum(e.type == "score-create" for e in events)
            for metric in [
                "trial_total_input_tokens",
                "trial_total_output_tokens",
                "context_auxiliary_input_tokens",
                "compact_count",
                "summary_fallback_count",
            ]:
                value = row.get(metric)
                if isinstance(value, (int, float)):
                    events.append(
                        IngestionEvent_ScoreCreate(
                            id=stable_id(result["run_id"] + ":event:" + metric),
                            timestamp=now(),
                            body=ScoreBody(
                                id=stable_id(result["run_id"] + ":score:" + metric),
                                trace_id=trace_id,
                                name=metric,
                                value=value,
                                data_type="NUMERIC",
                            ),
                        )
                    )
                    score_count += 1
            # v4 experiments are created from OTel context, not legacy dataset-run writes.
            experiment_attributes = {
                "langfuse.experiment.id": stable_id(dataset_name + ":" + row["arm"] + ":" + export_version),
                "langfuse.experiment.name": row["arm"]
                + "-"
                + versions["experiment_version"][:12]
                + "-"
                + export_version[:8],
                "langfuse.experiment.dataset.id": dataset.id,
                "langfuse.experiment.item.id": item_ids[row["pair_id"]],
                "langfuse.experiment.item.root_observation_id": stable_id(result["run_id"] + ":wall", 16),
                "langfuse.experiment.description": "Historical paired CodeFlow trial; original timestamps preserved",
            }
            send_historical_spans(client, tracer, ids, events, flush=False, experiment_attributes=experiment_attributes)
            records.append(
                {
                    "pair_id": row["pair_id"],
                    "arm": row["arm"],
                    "trace_id": trace_id,
                    "observation_count": len(spans) + 1,
                    "score_count": score_count,
                    "expected_input_tokens": row.get("trial_total_input_tokens"),
                    "trace_url": project_base + "/traces/" + trace_id,
                }
            )
            save(receipt_path, receipt)
            records[-1]["expected_scores"] = {e.body.id: e.body.value for e in events if e.type == "score-create"}
            records[-1]["expected_observation_ids"] = [
                e.body.id for e in events if e.type in {"span-create", "generation-create"}
            ]
            print(
                f"UPLOADED {index}/{len(cells)} {row['pair_id']} {row['arm']} observations={len(spans) + 1}", flush=True
            )
        client.flush()

        def verify_record(record):
            observations = client.api.observations.get_many(
                trace_id=record["trace_id"], limit=100, fields="core,basic,usage,metadata"
            )
            scores = client.api.scores_v3.get_many_v3(trace_id=record["trace_id"], limit=100)
            record["remote_observation_count"] = len(observations.data)
            record["remote_score_count"] = len(scores.data)
            remote_input = 0
            for observation in observations.data:
                data = observation.model_dump(mode="json")
                usage = data.get("usage_details") or data.get("usageDetails") or {}
                remote_input += usage.get("input", 0)
            record["remote_input_tokens"] = remote_input
            remote_scores = {s.id: s.value for s in scores.data}
            remote_ids = {o.id for o in observations.data}
            root_span = next((o for o in observations.data if o.id == record["expected_observation_ids"][0]), None)
            record["metadata_verified"] = bool(
                root_span and root_span.metadata and root_span.metadata.get("export_version") == export_version
            )
            record["verified"] = (
                set(record["expected_observation_ids"]) <= remote_ids
                and all(remote_scores.get(k) == v for k, v in record["expected_scores"].items())
                and record["metadata_verified"]
                and remote_input == record["expected_input_tokens"]
            )

        # A slow first read sweep may itself exceed a wall-clock polling window.
        # Allow up to three complete sweeps over the still-unverified records.
        for verification_attempt in range(3):
            with ThreadPoolExecutor(max_workers=5) as pool:
                list(pool.map(verify_record, [r for r in records if not r.get("verified")]))
            save(receipt_path, receipt)
            verified = sum(r.get("verified", False) for r in records)
            print(f"REMOTE VERIFIED {verified}/{len(cells)}", flush=True)
            if verified == len(cells):
                receipt["server_verified"] = True
                break
            if verification_attempt == 2:
                break
            time.sleep(2)
    except Exception as exc:
        receipt["error"] = sanitize(type(exc).__name__ + ": " + str(exc))
        print(receipt["error"], flush=True)
    finally:
        client.shutdown()
        receipt["export_and_verification_seconds"] = round(time.perf_counter() - started, 2)
        save(receipt_path, receipt)
    print("dataset_url=" + receipt.get("dataset_url", ""), flush=True)
    return 0 if receipt["server_verified"] else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--revision", default="evidence-v1")
    parser.add_argument("--dry-run", action="store_true", help="Only build local evidence; no credentials or API calls")
    parser.add_argument("--output", type=Path, help="Fresh output directory for --dry-run")
    args = parser.parse_args()
    if args.dry_run:
        if not args.output or args.output.exists():
            parser.error("--dry-run requires a new --output directory")
        expected_outputs(json.loads((args.root / "manifest.json").read_text(encoding="utf-8")))
        write_report(args.root.resolve(), args.output.resolve())
        print(str(args.output.resolve() / "REPORT.md"))
    else:
        raise SystemExit(export(args.root.resolve(), args.revision))
