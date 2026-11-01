"""Offline, evidence-only diagnostics for paired context studies (no model calls)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from langfuse_codeflow import load_spans, sanitize, save

DIAGNOSTIC_VERSION = "context-evidence-v1"
CRITERIA = (
    "early_constraint_retained",
    "active_constraint_applied",
    "latest_decision_applied",
    "artifact_valid_json",
    "forbidden_paths_clean",
)
CONFIG_KEYS = (
    "model",
    "variants",
    "scenarios",
    "context_window_tokens",
    "context_budget_ratio",
    "structured_compact_threshold",
    "max_calls_per_trial",
    "max_attempts_per_call",
    "timeout_seconds",
    "main_agent_output_tokens",
    "summary_output_tokens",
    "max_output_tokens_per_call",
    "output_contract_version",
    "runtime_config",
)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def identity(manifest):
    """Only hash frozen evidence, never attribute today's source to a historical run."""
    sources = {k.replace("\\", "/"): v for k, v in manifest.get("source_hashes", {}).items()}
    verifier = {k: v for k, v in sources.items() if "verifier" in k or k.endswith("canonical.py")}
    tasks = manifest.get("tasks", [])
    versions = dict(
        source_digest=digest(sources) if sources else "unknown",
        config_digest=digest({k: manifest[k] for k in CONFIG_KEYS if k in manifest}),
        prompt_digest=digest([{"history": t.get("history"), "payload": t["spec"]["payload"]} for t in tasks]),
        dataset_digest=digest(tasks),
        verifier_digest=digest(verifier) if verifier else "unknown",
        diagnostic_version=DIAGNOSTIC_VERSION,
        runtime_config_complete="runtime_config" in manifest,
    )
    versions["experiment_version"] = digest(versions)
    return versions


def validate_cells(manifest, rows):
    expected = {(t["spec"]["task_id"], a) for t in manifest["tasks"] for a in manifest["variants"]}
    actual = [(r["pair_id"], r["arm"]) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Study must contain each planned task/arm exactly once")
    if len(actual) != manifest["planned_trials"]:
        raise ValueError("planned_trials disagrees with task/arm matrix")
    if any(
        r.get("status")
        not in {"passed", "task_failed", "task_timeout", "infrastructure_failure", "budget_exceeded", "cancelled"}
        for r in rows
    ):
        raise ValueError("Study contains non-terminal or unsupported trial status")


def safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact reference escapes study directory")
    return path


def trial_spans(root, row):
    refs = row.get("artifact_refs") or []
    path = safe_path(root, refs[0]) / "traces/logs/audit-spans.log" if refs else None
    return load_spans(path) if path and path.is_file() else []


def diagnostics(row):
    findings = list(dict.fromkeys([*row.get("findings", []), *row.get("verifier_findings", [])]))
    status = row.get("status")
    categories = []
    if status in {"task_timeout", "infrastructure_failure", "budget_exceeded", "cancelled"}:
        categories.append(status)
    for finding in findings:
        text = finding.lower()
        if "missing" in text and "artifact" in text:
            categories.append("artifact_missing")
        elif "json" in text:
            categories.append("invalid_json")
        elif "artifact_mismatch" in text:
            categories.append("artifact_mismatch")
        elif "forbidden" in text:
            categories.append("forbidden_path_violation")
    for key in CRITERIA:
        if row.get(key) is False:
            categories.append(key + "_failed")
    if row.get("verifier_state") == "failed" and not categories:
        categories.append("verifier_failed_unclassified")
    if status != "passed" and not categories:
        categories.append("unknown")
    return dict(
        failure_category=categories[0] if categories else "none",
        failure_categories=list(dict.fromkeys(categories)),
        findings=findings,
        verifier_findings=row.get("verifier_findings", []),
        **{k: row.get(k) for k in CRITERIA},
        evidence_chain={
            "original_history": "manifest.tasks.history",
            "summary_retention": "unknown",
            "final_context_retention": "unknown",
            "artifact_checks": {k: row.get(k) for k in CRITERIA},
        },
        root_cause="unknown",
        next_check="Compare original history, saved summary, final model input and artifact; change one factor per experiment.",
    )


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def accounting(row, spans, ledger):
    models = [s for s in spans if s.get("attributes", {}).get("span.type") == "model"]
    totals = {}
    for key in ("input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens"):
        values = [s["attributes"].get("llm.usage." + key) for s in models]
        totals[key] = sum(values) if values and all(number(v) for v in values) else None
    prompt = (
        sum(totals[k] for k in ("input_tokens", "cache_read_tokens", "cache_write_tokens"))
        if all(totals[k] is not None for k in ("input_tokens", "cache_read_tokens", "cache_write_tokens"))
        else None
    )
    complete = (
        row.get("usage_complete") is True
        and prompt is not None
        and prompt == row.get("trial_total_input_tokens")
        and totals["output_tokens"] == row.get("trial_total_output_tokens")
    )
    costs = [s["attributes"].get("llm.usage.cost_total") for s in models]
    runtime_cost = sum(costs) if complete and costs and all(number(c) for c in costs) else None
    trial_id = row["pair_id"] + ":" + row["arm"]
    reserved = {e["request_id"]: e for e in ledger if e.get("kind") == "reserved" and e.get("trial_id") == trial_id}
    terminals = [e for e in ledger if e.get("request_id") in reserved and e.get("kind") in {"settled", "failed"}]
    settled = [e for e in terminals if e["kind"] == "settled"]
    ledger_complete = bool(reserved) and len({e["request_id"] for e in terminals}) == len(reserved)
    return dict(
        **totals,
        usage_reconciled=complete,
        recorded_runtime_estimate_usd=runtime_cost,
        main_agent_estimate_usd=None,
        summary_estimate_usd=None,
        cost_role_split="unknown: historical model spans do not record main/summary roles",
        cost_basis="recorded_runtime_estimate_not_invoice",
        pricing_version="unknown",
        actual_billed_cost_usd=None,
        ledger_budget_charge_cny=sum(e["charged_cny"] for e in terminals) if ledger_complete else None,
        ledger_request_attempts=len(reserved),
        ledger_terminal_attempts=len(terminals),
        ledger_failed_attempts=sum(e["kind"] == "failed" for e in terminals),
        ledger_usage_reconciled=(
            ledger_complete
            and not any(e["kind"] == "failed" for e in terminals)
            and sum(e.get("input_tokens", 0) for e in settled) == row.get("trial_total_input_tokens")
            and sum(e.get("output_tokens", 0) for e in settled) == row.get("trial_total_output_tokens")
        ),
        summary_input_tokens=row.get("context_auxiliary_input_tokens"),
        summary_output_tokens=row.get("context_auxiliary_output_tokens"),
        summary_latency_ms=row.get("summary_call_latency_ms"),
        trial_wall_latency_ms=row.get("wall_latency_ms"),
        human_wait_ms=None,
    )


def prepare(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("*--*.json"))]
    validate_cells(manifest, rows)
    ledger_path = root / "provider-budget.jsonl"
    ledger = (
        [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if ledger_path.exists()
        else []
    )
    evidence = []
    for row in rows:
        evidence.append(
            dict(
                pair_id=row["pair_id"],
                arm=row["arm"],
                scenario=row["scenario"],
                verifier_state=row["verifier_state"],
                status=row["status"],
                artifact_refs=row.get("artifact_refs", []),
                diagnostics=diagnostics(row),
                accounting=accounting(row, trial_spans(root, row), ledger),
            )
        )
    return manifest, rows, dict(schema=DIAGNOSTIC_VERSION, versions=identity(manifest), trials=evidence)


def comparisons(trials):
    groups = {}
    for scenario in ["overall", *sorted({r["scenario"] for r in trials})]:
        selected = [r for r in trials if scenario == "overall" or r["scenario"] == scenario]
        arms = {}
        for arm in sorted({r["arm"] for r in selected}):
            cells = [r for r in selected if r["arm"] == arm]

            def total(key):
                values = [r["accounting"].get(key) for r in cells]
                return sum(values) if values and all(number(v) for v in values) else None

            passed = sum(r["verifier_state"] == "passed" for r in cells)
            cost = total("recorded_runtime_estimate_usd")
            arms[arm] = dict(
                n=len(cells),
                passed=passed,
                failure_counts=dict(Counter(c for r in cells for c in r["diagnostics"]["failure_categories"])),
                input_tokens=total("input_tokens"),
                cache_read_tokens=total("cache_read_tokens"),
                cache_write_tokens=total("cache_write_tokens"),
                output_tokens=total("output_tokens"),
                recorded_runtime_estimate_usd=cost,
                estimate_usd_per_verified_task_including_failures=cost / passed
                if cost is not None and passed
                else None,
                ledger_budget_charge_cny=total("ledger_budget_charge_cny"),
                mean_wall_latency_ms=statistics.mean(r["accounting"]["trial_wall_latency_ms"] for r in cells),
                total_summary_latency_ms=total("summary_latency_ms"),
            )
        groups[scenario] = arms
    return groups


def write_report(root, output):
    manifest, rows, bundle = prepare(root)
    bundle["comparisons"] = comparisons(bundle["trials"])
    output.mkdir(parents=True, exist_ok=True)
    save(output / "evidence.json", bundle)
    lines = [
        "# Langfuse 评测证据复盘",
        "",
        f"实验：{manifest['run_id']}",
        f"版本：{bundle['versions']['experiment_version']}",
        "",
        "费用为历史运行时记录的估算，不是实际账单；价格版本未知。预算账本单独列示，失败尝试保留。",
        "未知或不完整的费用不按零计算。每任务一次的小样本不能证明质量等价或统计显著性。",
        "",
        "历史 Span 未记录调用角色，主调用与摘要的费用拆分未知；分别保留摘要 Token 与耗时，不能按 Token 比例分摊费用。",
        "",
        "| 场景 | 条件 | 通过/总数 | 未缓存输入 | 缓存读取 | 输出 | 均耗时 ms | 估算 USD | 预算记账 CNY |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario, arms in bundle["comparisons"].items():
        for arm, g in arms.items():
            values = [
                g[k]
                for k in (
                    "input_tokens",
                    "cache_read_tokens",
                    "output_tokens",
                    "mean_wall_latency_ms",
                    "recorded_runtime_estimate_usd",
                    "ledger_budget_charge_cny",
                )
            ]
            lines.append(
                f"| {scenario} | {arm} | {g['passed']}/{g['n']} | "
                + " | ".join("unknown" if v is None else f"{v:.6f}" if isinstance(v, float) else str(v) for v in values)
                + " |"
            )
    lines += ["", "## 失败排查", "", "判定来自独立产物检查，不代表已证明摘要导致失败。人工等待时间未采集。", ""]
    for trial in bundle["trials"]:
        if trial["status"] == "passed":
            continue
        d = trial["diagnostics"]
        lines += [
            f"### {trial['pair_id']} / {trial['arm']}",
            f"- 分类：{', '.join(d['failure_categories'])}",
            f"- 验证依据：{json.dumps(sanitize(d['findings']), ensure_ascii=False)}",
            f"- 产物检查：{json.dumps(d['evidence_chain']['artifact_checks'], ensure_ascii=False)}",
            f"- 本地证据：{trial['artifact_refs']}",
            "- 排查顺序：manifest 原始历史 → 保存的摘要 → 最终模型输入 → 实际产物。缺失环节标 unknown。",
            "- 下一步：核对关键决策在各环节的保留情况；固定数据与评分规则，每次只改一个因素再评测。",
            "",
        ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--output", type=Path, required=True, help="New report directory; original study is never rewritten"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new directory to preserve previous reports")
    write_report(args.root.resolve(), args.output.resolve())
    print(str(args.output.resolve() / "REPORT.md"))
