# CodeFlow task timing with Langfuse

## Evidence review and versioned export

Build a new offline report without credentials, model calls or uploads:

```powershell
.codeflow/langfuse/sdk-env/Scripts/python.exe scripts/export_context_study.py .codeflow/lc/<study-id> --dry-run --output .codeflow/langfuse/reviews/<new-review-id>
```

The report and `evidence.json` include failure categories, verifier findings,
individual artifact checks, cache-read/write and uncached input usage, output
usage, wall time, summary time, and per-scenario comparisons. The original
manifest, terminal records, and old reports are retained. Unknown values remain
unknown rather than becoming zero or an invented root cause. Failure costs are
included in total cost and cost per verified task.

Costs have separate meanings: `recorded_runtime_estimate_usd` is the historical
runtime estimate, with historical price version unknown; `ledger_budget_charge_cny`
is conservative budget accounting, including terminal failed requests. Neither
is a provider invoice. Missing usage or a mismatch with trace usage suppresses
the runtime cost estimate. Historical spans lack main/summary role labels, so
their cost split remains unknown; summary token counts and time are still shown.
Human waiting time was not collected.

New studies freeze held-out expected outputs in the manifest (never in the
Agent workspace), context/agent configuration and relevant source/template
hashes. Historical exports resolve expected output by frozen digest, never by
the current task ordering, and fail before API writes if no matching answer
exists. Partial, duplicate or unplanned trial matrices cannot be exported.

To export a completed study to the configured Langfuse project:

```powershell
.codeflow/langfuse/sdk-env/Scripts/python.exe scripts/export_context_study.py .codeflow/lc/<study-id>
```

The evidence export uses a separate dataset name from legacy exports. Source,
configuration, prompt, dataset and verifier digests are root-observation metadata;
diagnostic and export versions are separate from the measured execution version.
Changed exporter/source evidence creates new trace and experiment identities;
an identical export retries the same IDs. Existing experiment traces are retained.
Remote verification checks observation IDs, score IDs and values, export metadata
and input totals, not just counts. Missing evidence is not scored as false.

For a failed latest-decision check, compare frozen original history, saved summary,
final model input and actual artifact. The diagnostic reports the failed check;
summary retention and causal attribution remain unknown until inspected. Keep
the task/Verifier fixed and change one factor for the next paired experiment.

Offline regression checks:

```powershell
.codeflow/langfuse/sdk-env/Scripts/python.exe -m unittest discover -s scripts -p test_context_study_evidence.py -v
.codeflow/langfuse/sdk-env/Scripts/python.exe -m unittest discover -s scripts -p test_langfuse_codeflow.py -v
```

This optional adapter measures one real-provider file task, independently
checks its JSON artifact, and uploads the measured process span plus CodeFlow's
raw audit spans. It does not change CodeFlow's runtime dependencies or tracing.
The current fixture is a small arithmetic/file task, not a repository bugfix
benchmark or a representative latency study.

## Local configuration

Use a separate Python 3.12 environment because Langfuse brings OpenTelemetry
dependencies while CodeFlow's main environment deliberately excludes them:

```powershell
uv venv --python .venv/Scripts/python.exe .codeflow/langfuse/sdk-env
uv pip install --python .codeflow/langfuse/sdk-env/Scripts/python.exe -r scripts/requirements-langfuse.txt
```

Write these variables to `.codeflow/langfuse/langfuse.env` (already Git-ignored),
or set them in the exporter's environment:

```text
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
LANGFUSE_PUBLIC_KEY=<project-public-key>
LANGFUSE_SECRET_KEY=<project-secret-key>
```

Keys are read by the exporter and are not passed into the measured Agent's
environment. Runtime provider credentials come from the existing CodeFlow
config; a local copy is stored only in the ignored run directory. These
directories contain credentials and should not be shared wholesale.

## Measure and upload

From the repository root:

```powershell
.codeflow/langfuse/sdk-env/Scripts/python.exe scripts/langfuse_codeflow.py run --upload
```

The task reads `numbers.json`, counts the even numbers, and writes `result.json`.
The external verifier requires exactly `{"count": 3, "sum": 12}`. Execution uses
the checkout's `.venv/Scripts/python.exe`, the configured model, an isolated
workspace/state directory, at most six tool iterations, 1,024 output tokens per
request and a 180-second process timeout. These are task limits, not a certified
provider spending cap. External MCP and selected unrelated tools are disabled.

Output is under `.codeflow/langfuse/runs/<run-id>/`:

- `result.json`: measured task time, original timestamps, model and verdict.
- `stdout.txt`, `stderr.txt`: local CLI output.
- `trace/logs/audit-spans.log`: original execution spans.
- `langfuse-receipt.json`: upload time, trace URL and remote verification status.

`task_latency_ms` includes CLI startup, model/tool work and process exit. Fixture
preparation, independent JSON verification and Langfuse export are excluded.
The Langfuse parent span uses these original timestamps; importing historical
spans does not replace their timing with export time. The turn span separately
shows the Agent's runtime interval. A single trial is a smoke measurement,
not P50/P95 or evidence of an optimization.

Retry an upload without another paid model call:

```powershell
.codeflow/langfuse/sdk-env/Scripts/python.exe scripts/langfuse_codeflow.py upload .codeflow/langfuse/runs/<run-id>/result.json
```

The adapter preserves parent relations, uses deterministic trace/observation/score
IDs, and merges repeated checkpoints by original span ID. Only recorded previews
and sanitized attributes are exported; full local payload artifacts are not
automatically uploaded. Model calls become generations with recorded input and
output token usage. Missing verifier evidence creates no success/failure score.

Trace writes use the SDK's OpenTelemetry exporter, not the retired legacy trace
ingestion API. The adapter uses the public OTel tracer interface and Langfuse
observation wrappers to retain historical start/end times. SDK project scope
attributes are required for its project-isolation filter. It verifies observation
and score counts via the current observation and score APIs; a successful flush
alone does not mark a run as remotely verified.

## Offline checks

```powershell
.codeflow/langfuse/sdk-env/Scripts/python.exe -m unittest discover -s scripts -p test_langfuse_codeflow.py -v
.venv/Scripts/python.exe -m pytest tests/test_no_otel_tracing.py -q
```

The timing adapter accepts its own run schema. Importing every existing
CodeFlowBench result schema and configuring LLM judge rules remain separate work.

## Paired context study

`scripts/context_langfuse_study.py` runs three scenario groups with five task
pairs each: long session, cross-turn dependency and explicit summary-service
failure. It reuses the real Context trial runner, sealed artifact verifier and
provider budget ledger. Its baseline keeps full history; treatment uses the
shipping history manager. Source hashes and all 30 planned conditions are frozen
in a manifest before execution. A failed task is retained without quality retries.

```powershell
.venv/Scripts/python.exe scripts/context_langfuse_study.py --output .codeflow/lc/<new-study-id>
.codeflow/langfuse/sdk-env/Scripts/python.exe scripts/export_context_study.py .codeflow/lc/<new-study-id>
```

These commands make real model calls. The study records all actual prompt
tokens, including cache-read and cache-write input, plus any live summary calls.
The exporter creates a 15-item Langfuse dataset and associates already completed
traces with `baseline` and `governed` experiment runs; export does not rerun tasks.
Experiment membership uses v4 OTel experiment attributes. The old dataset-run
read/write APIs are not used for this association.

The initial `study15a` did not reproduce 100% quality in both conditions:
baseline passed 14/15 and treatment 5/15. All-sample input decreased 14.31%, while
the five both-pass pairs decreased 3.02%. Under the fixed 500-token output cap,
all five high-pressure summary requests were truncated with empty text and
entered fallback. See the retained `ANALYSIS.md`, `manifest.json`,
`evidence-audit.json`, raw traces and provider ledger for limitations and failure
examples. This is exploratory evidence, not a production reliability claim.
