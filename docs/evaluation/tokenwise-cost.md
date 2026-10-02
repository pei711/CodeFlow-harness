# DeepSeek TokenWise cost experiment

> **Status: rerun through current CallEfficiency on 2026-08-13.** The frozen
> DeepSeek V4 Flash campaign completed 36 valid Comparison Blocks, 72 Trials,
> and 504 real Provider calls. All Claim Gates passed.

This page keeps the historical TokenWise experiment name for continuity. The
active Runtime subsystem is CallEfficiency, and the current campaign crossed
the shared Runtime Assembly and retained one Call Record per physical attempt.

## Question

How much does DeepSeek's automatic disk context cache reduce the estimated API
cost of a verified successful CodeFlow Agent task when the request prefix remains
stable?

DeepSeek ignores Anthropic `cache_control` markers. TokenWise therefore does
not place explicit breakpoints for this Provider. It normalizes DeepSeek's
`prompt_cache_hit_tokens` and `prompt_cache_miss_tokens`, rebuilds cost from
the Provider's cache-hit, cache-miss, and output rates, and measures the value
of prefix stability.

## Treatment axis

Every Comparison Block executes the same task under two policies. DeepSeek's
automatic cache remains enabled in both arms.

| Policy | Request behavior | Role |
| --- | --- | --- |
| `prefix_disrupted` | Change the leading system and Tool Schema bytes before every Provider call | Negative control |
| `prefix_stable` | Preserve CodeFlow's ordinary request prefix | Treatment |

The disrupted arm is an experimental counterfactual, not an earlier product
version and not a deployable configuration. Each Trial uses a separate
DeepSeek `user_id` so one arm cannot warm another arm's KV cache.

The Provider, exact model, non-thinking generation mode, Tool set, Context
budget, workspace fixture, prompts, and retry limits remain fixed. Fallbacks
are forbidden.

## Frozen workload matrix

The campaign has four workload classes, three cases per class, and three
repetitions:

| Workload class | Observable pressure | Shape per Trial |
| --- | --- | --- |
| `stable_dialogue` | Repeated stable system instructions | Six short Turns, no Tools |
| `long_history` | Growing conversation prefix | Six Turns after sixteen seeded history Turns |
| `tool_accumulation` | Tool schemas and results accumulate across Turns | Six Turns, one verified Tool call per Turn |
| `intra_turn_tool_chain` | Tool results extend the prefix within one Turn | One Turn with a verified three-step Tool chain |

This produces 36 Comparison Blocks and 72 Trials. A Trial is one policy
executing one case once. The campaign made 504 real Provider calls.

## Metrics and gates

The primary metric is estimated cost per verified success:

```text
cost_per_verified_success = sum(all valid Trial cost) / verified successes
```

Failed tasks remain in the numerator. The conservative cache hit rate is:

```text
cache_read / (cache_miss + cache_read)
```

The report exports CV metrics only when all planned blocks are valid, every
usage record satisfies `prompt = cache_hit + cache_miss`, the exact requested
model served every call, all four workload classes are present, treatment task
success does not regress, and stable prefixes improve both cache hit rate and
cost per verified success.

## Result

The campaign was pinned to `deepseek/deepseek-v4-flash`. The frozen pricing
snapshot was USD 0.14 per million cache-miss input tokens, USD 0.0028 per
million cache-hit input tokens, and USD 0.28 per million output tokens.

| Metric | Prefix disrupted | Prefix stable |
| --- | ---: | ---: |
| Valid Trials | 36 | 36 |
| Verified task pass rate | 100% | 100% |
| Conservative cache hit rate | 0% | 74.0478% |
| Estimated cost per verified success | $0.008356 | $0.002311 |

Stable prefixes reduced estimated cost per verified success by **72.3413%**
in the aggregate. The task-clustered paired estimate was **72.0750%**, with a
95 percent interval of **68.8471% to 75.0961%**. All 36 Comparison Blocks were
valid, no fallback or model drift occurred, and the full campaign used an
estimated USD 0.384031.

Per-workload treatment hit rates were 65.12% for stable dialogue, 75.72% for
long history, 79.31% for Tool accumulation, and 65.47% for the intra-Turn Tool
chain.

## Evidence boundary

The result proves that CodeFlow's stable request prefixes benefit from DeepSeek's
automatic cache under the frozen workload and that TokenWise reconstructs
DeepSeek cache usage and estimated cost. It does not prove that CodeFlow created
DeepSeek's cache, that every production workload will achieve a 75.19% hit
rate, or that the estimate has been reconciled against a Provider invoice.

The current report is retained outside Git under
`.codeflow/evidence/call-efficiency-cost/1df7029-formal/`. Its SHA-256 is
`b905ec833231236a53959cf78b05c89ca9b72b4066055aa5b6e3c327df3e4337`.
Raw manifests, immutable inputs, and standalone report artifacts are intentionally
not published in this repository.

## Reproduction

The CallEfficiency replay path makes no Provider calls. It validates the
historical report digest, recalculates every Trial from the embedded frozen
price snapshot, and runs the original reducer again:

```bash
uv run python -m benchmarks.codeflowbench.packs.tokenwise_cost.replay \
  --source-report .codeflow/evidence/tokenwise-cost-deepseek-rebased/report.json \
  --expected-source-digest fcde99b98c8bc46d0852015d7a92c01a0de6a4e4216f773045375f2f06e75aec \
  --output .codeflow/evidence/call-efficiency-replay/report.json
```

`--expected-source-digest` is the external lineage binding. Obtain it from a
separately trusted manifest or frozen evidence record; copying a digest from the
same report being checked does not establish provenance. The replay refuses to
claim equivalence without that binding and refuses to overwrite its source
artifact.

An `equivalent: true` result establishes artifact and reducer equivalence only.
It is not a new live Runtime result.

The current paid runner crosses the shared Runtime Assembly and observes every
physical Provider attempt through CallEfficiency. It retains raw Provider and
CallEfficiency receipts, applies a task-clustered paired bootstrap interval,
and rebuilds the result offline. Paid modes remain behind an explicit flag:

```bash
uv run python -m benchmarks.codeflowbench.tokenwise_cost_campaign \
  --mode preflight \
  --output-root .codeflow/evidence/call-efficiency-cost-current \
  --execute-paid-campaign

uv run python -m benchmarks.codeflowbench.tokenwise_cost_campaign \
  --mode formal \
  --output-root .codeflow/evidence/call-efficiency-cost-current \
  --execute-paid-campaign

uv run python -m benchmarks.codeflowbench.tokenwise_cost_campaign \
  --mode verify \
  --output-root .codeflow/evidence/call-efficiency-cost-current
```

The runner reads `DEEPSEEK_API_KEY`, then falls back to
`providers.deepseek.apiKey` in CodeFlow's config. It never writes credentials into
artifacts. It stops before a new call at either 1,200 Provider calls or USD 2
of observed estimated spend.

The formal campaign remains 12 frozen tasks times three repetitions times two
arms: 36 pairs and 72 Trials. A positive claim additionally requires every task
to pass, complete Usage and cost data, exact-model execution without fallback,
one persisted CallEfficiency record per physical attempt, healthy ledgers, and
a paired cost-reduction confidence interval whose lower bound is above zero.
The verifier writes raw outcomes, the rebuilt aggregate, claim eligibility,
verifier status, and a SHA-256 inventory without making Provider calls.
