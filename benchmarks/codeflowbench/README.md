# CodeFlowBench Ship-1

CodeFlowBench is CodeFlow's checkout-only evaluation campaign. It runs the real CodeFlow
Runtime through frozen, single-axis comparisons and rebuilds every result from
terminal artifacts. It is not part of the `codeflow` wheel and does not add a
public `codeflow bench` command.

## Entry points

Run the credential-free gate first:

```bash
make codeflowbench-smoke
```

It executes the 2,000-request Scheduler track, the 100-Turn full Runtime track,
the local stdio MCP transport smoke, and an artifact-only report rebuild. It
does not resolve a Provider or make a paid model call.

Run the frozen calibration and formal campaign with:

```bash
make codeflowbench
```

The paid command uses the normal CodeFlow configuration, but only proceeds when it
resolves exactly `deepseek / deepseek-v4-flash`, proves Tool Calling and
complete usage fields with one live preflight, and freezes the tokenizer
identity. Fallback models are forbidden.

## Current Scorecard campaign

The current Scorecard campaign runs the Context and Tool/MCP Packs. CallEfficiency
and Runtime remain independent evidence tracks and are composed only after their
own Claim Gates are evaluated. No external Memory backend experiment is included
in this release.

Run or resume the complete multidimensional workflow with:

```bash
CODEFLOW_BENCH_EXECUTE_PAID=1 make codeflowbench-reproduce
```

The command performs a full preflight before any paid call, then runs Runtime,
TokenWise, and the Context plus Tool/MCP Scorecard tracks. It
composes the multidimensional score, renders a terminal table, and writes
`report.json`, `REPORT.md`, `score.json`, and stage logs under
`.codeflow/evidence/codeflowbench-reproduction/<codeflow-commit>/`. Repeating the command
validates and resumes retained stage artifacts instead of rerunning completed
work.

Existing current-commit evidence can be reused without paid execution:

```bash
CODEFLOW_SCORECARD_FORMAL_SUMMARY=/absolute/path/to/summary.json \
CODEFLOW_SCORECARD_RUNTIME_EVIDENCE=/absolute/path/to/runtime-evidence.json \
CODEFLOW_SCORECARD_TOKENWISE_REPORT=/absolute/path/to/tokenwise-report.json \
  make codeflowbench-reproduce
```

The runner copies these compact inputs into its output directory and verifies
their commit identities and digests before composing the score. The formal
summary must remain beside the `manifest.json` from the same experiment so its
Scorecard identity can be rebuilt. A diagnostic score is always labeled
separately from a certified score, and the terminal and Markdown reports list
every failed certification check in plain language.

Print the frozen worst-case budget with:

```bash
make codeflowbench-scorecard-estimate
```

Run the paid campaign with:

```bash
make codeflowbench-scorecard-ship
```

Runtime evidence is not required to measure the independent Context and
Tool/MCP Packs. When `CODEFLOW_SCORECARD_RUNTIME_EVIDENCE` is supplied, the
campaign verifies that its CodeFlow product code, dependency identity, and Runtime
Pack are byte-equivalent before making a paid call. It never silently reruns
the Runtime experiment.

The v1 multidimensional score uses Capability 50, Reliability 20, Efficiency
20, and Process 10. Capability averages current treatment rates for Context,
Tool/MCP, and Memory. Context capability is the average of four frozen checks:
the early constraint is still present, the artifact applies it, the latest
decision is applied, and the artifact is exactly correct. The strict external
verifier pass rate is still reported separately and is not relaxed. Context
reserves 500 output tokens and protects its first constraint turn, leaving
enough of the 2,400-token window for the latest decision. Empty Provider
responses may be retried up to four times, symmetrically in both arms.
Context capability validity is independent from token-usage completeness.
Missing usage invalidates only the Context efficiency claim; it is never
coerced to zero and never blocks verifier-backed capability diagnostics.
Efficiency assigns five points to each eligible TokenWise, Context, Tool/MCP,
and Turn-efficiency claim. Missing, ineligible, or commit-incompatible evidence
contributes zero. Process checks MCP disclosure, transport, invalid-target,
and exact-repeat gates. Safety and full evidence coverage are certification
gates, not bonus points.

Compute the score from immutable formal and Runtime artifacts with:

```bash
CODEFLOW_SCORECARD_FORMAL_SUMMARY=/absolute/path/to/summary.json \
CODEFLOW_SCORECARD_RUNTIME_EVIDENCE=/absolute/path/to/runtime-evidence.json \
CODEFLOW_SCORECARD_TOKENWISE_REPORT=/absolute/path/to/tokenwise-report.json \
CODEFLOW_SCORECARD_MEMORY_SUMMARY=/absolute/path/to/memory-summary.json \
CODEFLOW_SCORECARD_MEMORY_HANDOFF=/absolute/path/to/memory-handoff.json \
CODEFLOW_SCORECARD_PREREGISTERED=1 \
  make codeflowbench-scorecard-score
```

Runtime, Memory, and preregistration inputs are optional. The Memory summary
is accepted only with a digest-bound handoff that names the current CodeFlow
commit. The scorer always emits a diagnostic score and assigns zero to missing
dimensions. It emits a certified score only when the scoring specification was
preregistered and every evidence and safety gate is complete.

## Frozen scale and budget

The suite at
`benchmarks/codeflowbench/suites/agent_application_ship_1.yaml` freezes:

- calibration: 64 E2E Trials and 34 Retrieval Cases;
- formal: 216 E2E Trials and 260 Retrieval Cases;
- at most two Provider attempts and two whole-block attempts;
- a warning threshold of 80 CNY and a hard ceiling of 100 CNY.

The frozen Provider budget is pack-specific:

- Context Trials allow at most eight logical calls with 42,000 input tokens
  and 1,200 output tokens per call. The benchmark-only Curator is capped at
  four steps and the main Agent Loop at four calls.
- Memory / Skill Trials allow at most four logical calls with 15,000 input
  tokens and 1,500 output tokens per call.
- Tool / MCP Trials allow at most four logical calls with 40,000 input tokens
  and 1,500 output tokens per call.

Pricing all DeepSeek input at the cache-miss rate, using the frozen USD-to-CNY
multiplier, and reserving 5 CNY for external services gives a 62.86592 CNY
fresh-campaign worst case for the current suite.

Before any paid call, CodeFlowBench includes existing ledger spend and two live
preflight attempts in its projection. It freezes the current request count and
the newly authorized attempt count in a digest-bound approval record. Resumed
runs reuse that lifetime ceiling rather than opening a new budget. Every
request reserves its pack-specific maximum before dispatch and settles from
complete Provider usage; incomplete accounting is an infrastructure failure,
not permission to continue spending.

## Formal result

The final-history Ship-1 campaign ran from clean source commit
`e6c790e37d707f74c44896dbcba9de9ee4ad8327` with
`deepseek / deepseek-v4-flash`. The Provider did not expose a seed, so the
matrix records three repetitions rather than three seeds.

The formal experiment id is
`abffb7d2fe6a76f1102741cacd3cff1ed02697be0fb37fe2fa7a910dbeb11b4d`:

- `ship_complete=true`, but `measurement_valid=false`;
- all 216 planned E2E Trials and all 260 Retrieval Cases have terminal records;
- selected E2E outcomes are 82 passed, 72 task failures, 61 task timeouts,
  and one infrastructure failure;
- 119 of 120 Pairs are valid. One Context comparison exhausted its symmetric
  retry after incomplete treatment usage evidence;
- repeated raw-artifact rebuilds produce report digest
  `25ca985d2f80560fba789d14fc76acc07d0a45ea3b6a0b4aa7a3d4cfadf19eb1`;
- the cumulative main-campaign ledger recorded a 25.26335175 CNY Provider
  high-water mark, or 30.26335175 CNY committed including the fixed 5 CNY
  external-service reserve. This is a budget-control estimate, not a Provider
  invoice.

Across the final-history main campaign, semantic v1 replay, and semantic v2
campaign, the three Provider high-water marks are 25.26335175, 0.13745664,
and 0.21468672 CNY, totaling 25.61549511 CNY with zero open reservations.
This cross-ledger total is also a budget-control estimate, not a Provider
invoice.

Global `positive_claim_eligible` is false and `cv-metrics.json` exports no
metrics. Context has only 23 valid Pairs, so the declared pack invalidates the
aggregate measurement. Tool/MCP remains independently ineligible even though
progressive disclosure reduced equal-task macro estimated visible Tool Schema
tokens by 93.4513% across six measurable tasks: treatment passed 20 of 24
Trials versus 23 of 24 control Trials, including one task that lost at least
two of three passes. Semantic Memory end-to-end produced zero passes in both
arms and no verifier gain. These retained failures are product findings, not
positive resume claims.

## Evidence boundary

Calibration and formal task and query IDs are disjoint. Claim Rules are hashed
before the live preflight. Every planned Trial and Retrieval Case must have one
terminal record, and the report is rebuilt only from those records. Provider,
infrastructure, timeout, cancellation, and task failures remain in the
artifacts.

`ship_complete` means the campaign and evidence chain completed.
`measurement_valid` means the retained measurements are interpretable.
`positive_claim_eligible` is reported globally and per capability; only metrics
that pass every rule in their capability group enter `cv-metrics.json`. A
valid negative result completes Ship-1 without becoming a positive resume
claim.

Deterministic R0 and R1 Runtime metrics are exported separately from a clean
source commit into an immutable, self-digested Runtime evidence artifact. The
current artifact is bound to source commit
`e6c790e37d707f74c44896dbcba9de9ee4ad8327` with evidence digest
`1c9fc1c4882ff09e3cf44140d84206bfb5ce923344cf7e482615a36e4f0f6006`.
Those metrics may be quoted only with their explicit Scheduler or full-path
deterministic scope; they are not live throughput, task-effect, or production
SLO claims.

Raw Trials, traces, Memory content, receipts, and generated reports stay under
`.codeflow/evidence/codeflowbench/` and are not committed.
