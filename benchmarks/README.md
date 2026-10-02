# CodeFlow Benchmarks

This directory holds **evaluation harnesses** that are deliberately decoupled
from the runtime package. They are not imported by `codeflow/` and are
excluded from the wheel build — keep it that way.

The Markdown task cards under `pinchbench/tasks/` are executable benchmark
fixtures, not CodeFlow product documentation. Some intentionally probe capabilities
removed from CodeFlow, including image generation or remote Skill discovery. A
task card's presence does not mean the current Runtime supports that Tool.

Use this area for reproducible evaluation work: capability suites, agent
comparisons, and context stress tests that should not ship as part of the
end-user CLI package.

## CodeFlowBench

CodeFlowBench Ship-1 is the checkout-only Agent application evaluation harness
under `benchmarks/codeflowbench/`. Its contract is documented under
[public evaluation notes](../docs/evaluation/README.md). It evaluates the existing
Runtime through frozen, single-axis paired tasks and parent-owned deterministic
Verifiers.

The implementation does not create a result claim by itself. Generated
CodeFlowBench evidence remains outside Git. Ship Completeness and Measurement
Validity govern the campaign, while each capability separately applies its
Positive Claim Eligibility rules. PinchBench, EvalEngine, Evolver evidence,
CodeFlowBench, and V-R0 remain distinct scopes.

The final-history Ship-1 campaign completed all 216 planned E2E Trials and 260
Retrieval Cases, but one Context Pair lacked complete usage evidence and made
the aggregate measurement invalid. Tool disclosure also regressed task pass
count, so the retained main-campaign material exports no positive CV metric.
See [CodeFlowBench](codeflowbench/README.md) for the published experiment boundary.

## Layout

```
benchmarks/
├── appworld/           AppWorld agent benchmark + evolver plugin
│   ├── agent_cli.py       One-task subject agent (drives AgentLoop)
│   ├── batch.py           Batch scorer: N tasks x K trials, resumable
│   └── evolve/            codeflow.evolver BenchBundle plugin (entry.py)
│                          + designer/diagnosis/sandbox/precheck glue
│
├── evolver/            Deterministic Evolution Run subject
│   ├── small_real.yaml     One-round run spec (+ --smoke overlay)
│   └── subject_template/   Disposable subject: a defective agent_cli.py plus
│                           its own bench plugin; materialized to subject/
│                           (gitignored) by scripts/setup_small_real_subject.py
│
├── pinchbench/         Context / AgentLoop capability benchmark
│   ├── tasks/             23 task_*.md cards (YAML frontmatter + sections)
│   ├── direct/            Drives AgentLoop.run_turn() per task
│   ├── bot_runner/        Drives full gateway + channel path per task
│   ├── assets/            Task-specific workspace files
│   └── results/           Run outputs (gitignored)
│
├── codeflowbench/          Agent application evaluation harness
│   ├── packs/             Runtime, Context, Memory/Skill, and Tool/MCP tracks
│   ├── suites/            Frozen experiment plans and claim rules
│   └── README.md          Smoke, campaign, rebuild, and evidence boundaries
│
├── clawbench/          ClawBench streaming benchmark adapter
│   ├── stream.py          Drives AgentLoop.run_turn() across one session
│   ├── run.sh             Shell wrapper
│   └── README.md          Setup and run instructions
│
├── skill_evals/        Query corpus for the retained SkillForge evaluation
│   └── queries.jsonl      Used by scripts/skill_forge_retrieval_eval.py
│
└── README.md           This file
```

Run `uv run python scripts/skill_forge_retrieval_eval.py` for the
self-contained, offline SkillForge retrieval evaluation. The obsolete
SQLite mass-library runner was removed with the retired remote skill
retrieval architecture.

## Running

### Model and tool configuration

The benchmark runners can use the normal `~/.codeflow/config.json`, or the
environment overrides below. Never commit real keys.

For an OpenAI-compatible gateway using OpenRouter-style environment names:

```bash
export OPENROUTER_API_KEY="..."
export OPENROUTER_API_BASE="https://openrouter.ai/api/v1"
export CODEFLOW_BENCH_PROVIDER="custom"
export CODEFLOW_BENCH_MODEL="deepseek-v4-flash"
```

Optional web tools:

```bash
export SERPER_API_KEY="..."
export JINA_API_KEY="..."
```

Equivalent `~/.codeflow/config.json`:

```json
{
  "agents": {
    "defaults": {
      "provider": "custom",
      "model": "deepseek-v4-flash",
      "maxToolIterations": 40,
      "contextWindowTokens": 65536
    }
  },
  "providers": {
    "custom": {
      "apiKey": "YOUR_API_KEY",
      "apiBase": "YOUR_OPENAI_COMPATIBLE_API_BASE"
    }
  },
  "tools": {
    "web": {
      "jinaApiKey": "YOUR_JINA_KEY",
      "search": {
        "apiKey": "YOUR_SERPER_KEY"
      }
    }
  }
}
```

PinchBench (Direct mode):
```bash
./benchmarks/pinchbench/direct/run.sh \
    --model deepseek-v4-flash \
    --provider custom \
    --api-base "$OPENROUTER_API_BASE" \
    --api-key "$OPENROUTER_API_KEY" \
    --suite task_00_sanity
```

PinchBench (Bot mode):
```bash
./benchmarks/pinchbench/bot_runner/run.sh --suite automated-only
```

ClawBench (first 80 tasks, one streaming session):
```bash
git clone https://github.com/claw-bench/claw-bench ../claw-bench
export CLAW_BENCH_ROOT="$PWD/../claw-bench"

./benchmarks/clawbench/run.sh \
    --clawbench-root "$CLAW_BENCH_ROOT" \
    --limit 80 \
    --session-id clawbench-stream-codeflow-80 \
    --max-iterations 40
```

ClawBench with Curator context engine:
```bash
./benchmarks/clawbench/run.sh \
    --clawbench-root "$CLAW_BENCH_ROOT" \
    --limit 80 \
    --session-id clawbench-stream-codeflow-curator-80 \
    --context-engine curator \
    --curator-model deepseek-v4-flash \
    --max-iterations 40
```

## Relation to runtime

The runtime (`codeflow/`) **never statically imports from `benchmarks/`** — this
is the "independent eval track" principle. The reverse is allowed and
expected: benchmarks import `codeflow.agent`, `codeflow.providers`, etc. directly.

One scoped exception: `codeflow.evolver` loads its bench *plugins* from here by
registry name at launch (`benchmarks.appworld.evolve.entry:build`), inserting
the subject repo root on `sys.path` first. It is lazy, opt-in, and only works
from a repo checkout — evolution needs the git repo as its subject anyway, so
nothing in the installed wheel depends on this directory.

AppWorld is the checkout example. The tracked small-real template materializes
a disposable subject repository that owns its own registered benchmark plugin
and immutable grader. References to EvoAgentBench or other benchmark lines in
methodology/design notes remain planned or historical unless corresponding
code exists.
