# Runtime scheduler experiments

The Runtime scheduler benchmark isolates three claims that cannot be supported
by the original dispatch-overhead probe alone. It uses deterministic local
workloads so Provider and network variance cannot dominate queueing behavior.
The results are scheduler evidence, not a production service-level objective.

Run the frozen defaults from a clean checkout:

```bash
make codeflowbench-runtime-scheduler
```

The command writes immutable evidence under
`.codeflow/evidence/codeflowbench-runtime-scheduler/`. Generated evidence remains
outside Git. A result is claim-eligible only when every correctness Gate passes
and the commit, dependency lock, environment identity, and clean worktree stay
stable for the full run.

## Experiment 1: session head-of-line isolation

The control is one strict global FIFO with a fixed worker limit and
per-session serialization. If the queue head belongs to an already-running
session, later sessions wait. The treatment is CodeFlow's session Lane scheduler
with the same USER concurrency limit. Both arms receive the same ordered trace:
a long hot-session burst interleaved with short foreground Turns from other
sessions.

The primary metric is foreground P95 queue wait. Each paired repetition
alternates arm order. The reported improvement is the median of the paired P95
reductions, rather than a percentile calculated from pooled repetitions.

## Experiment 2: foreground and background bulkheads

The control maps USER, CRON, and SUBAGENT work to one shared semaphore. The
treatment uses independent USER and Runtime-origin pools. Both policies have
the same total nominal capacity and run the same idle and background-saturated
foreground trace.

The primary metric is the ratio of loaded foreground P95 queue wait to idle
foreground P95 queue wait. Reporting a ratio controls for host-level timing
differences between repetitions. Policy and load order alternate across paired
repetitions.

## Experiment 3: accepted-request fate accounting

The request-fate experiment repeats the existing R0 Runtime conformance track.
It exercises normal execution, queued and running cancellation, injection,
interrupt, shutdown, origin limits, and rejection after draining. It aggregates
all accepted requests and requires zero lost requests, unexpected duplicate
executions, unresolved Handles, lifecycle contradictions, and pool-limit
violations.

This is an in-process guarantee from scheduler acceptance to terminal Handle
resolution. It does not claim exactly-once execution across process crashes or
external side effects.

## Live Agent response experiment

The live extension keeps the head-of-line comparison but replaces scripted
delays with `AgentTurnRunner -> AgentLoop -> configured Provider` calls. Every
foreground prompt owns a unique marker. A Turn is measurable only when the
reply contains that marker, Provider usage is complete, no Tool is called, and
the Runtime reaches a terminal result.

The control and treatment receive the same prompts in the same order, use the
same Provider and model, and use a balanced alternating arm order across 80
paired repetitions. Each arm contains two consecutive Turns from one hot
conversation followed by 24 foreground conversations, for 4,160 real Agent
Turns in total. The primary metric is foreground accept-to-terminal P95
latency. The reported effect is the median paired P95 reduction with a
deterministic 10,000-resample bootstrap 95% confidence interval.

Every Turn is retained as a raw record with its submission index, conversation,
accept/start/terminal offsets, queue wait, execution time, end-to-end latency,
usage, invocation count, and Verifier failures. The result is eligible only if
all Turns pass their task Verifiers, the confidence interval lower bound is
positive, and every Provider, usage, budget, checkout, and performance Gate
passes. This is an adversarial head-of-line burst benchmark, not a production
traffic model or service-level objective.

Live execution is a separately approved paid action. First freeze the clean
commit, exact manifest, maximum request attempts, and CNY hard cap:

```bash
make codeflowbench-runtime-live-plan
```

Then pass the exact approved manifest digest and amount:

```bash
CODEFLOW_LIVE_PERF_APPROVAL_DIGEST=<digest> \
CODEFLOW_LIVE_PERF_APPROVED_CNY=<amount> \
make codeflowbench-runtime-live-run
```

Rebuild the summaries and confidence interval from the retained raw Turn
records without calling the Provider:

```bash
CODEFLOW_LIVE_PERF_EVIDENCE=<run-artifact.json> \
make codeflowbench-runtime-live-verify
```

## Formal live campaign result

The formal campaign was bound to source commit
`00fd757f1b55cb05f67bde099de2b1196d690db6`, plan digest
`d45e5ba88b8f07a82276612170bdbf2b9205f15d3c3b73ac1534378e7c1b4e8c`,
and evidence digest
`07f8c955347f50797e8f4e4aafb02ee85cd9e3ab06547ecaa75cc0fd9fd8bf59`.
The source SHA is retained as experiment identity. Its development branch is
not published in this release repository.
It completed 80 paired repetitions and retained 4,160 raw Turn records.

The median foreground P95 fell from 13,567.997 ms under strict global FIFO
to 11,663.193 ms with session Lanes, a 13.53% reduction. The deterministic
10,000-resample paired bootstrap 95% confidence interval was 11.36% to
15.26%. All 4,160 requests executed with zero missing executions, duplicate
executions, runner exceptions, or Provider failures.

The task Verifier passed 4,158 of 4,160 Turns, or 99.95%. The two failures
were `marker_missing`: each request reached a normal terminal result with
complete usage and an explicit reply, but the model reply omitted its unique
marker. One reply reached the 512 completion-token limit. These were model
response-quality failures, not scheduler loss or Provider errors.

The preregistered `all_tasks_verified` Gate required 4,160 of 4,160 marker
passes, so the aggregate artifact records `claim_eligible: false`. Resume and
public wording must report the 99.95% Verifier pass rate and must not say that
every task passed. The raw-record verifier independently reproduces the P95
reduction, confidence interval, usage totals, and request-integrity counts.
