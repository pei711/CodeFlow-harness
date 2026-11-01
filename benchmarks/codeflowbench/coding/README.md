# Repository Coding Task Track

This checkout-only track adds real repository issue regression beside the
existing synthetic Context and Tool/MCP packs. It is deliberately outside the
`codeflow` wheel. The first version is a task runner and artifact format, not a
curated SWE-bench-sized dataset. Add 10–20 reviewed, commit-pinned tasks before
making broad capability claims.

## Task definition

Save a JSON file with this shape:

```json
{
  "schema": "codeflow.coding-task.v1",
  "task_id": "bugfix-example",
  "repository": "C:/work/example-repo",
  "commit": "0123456789abcdef0123456789abcdef01234567",
  "prompt": "Fix the reported bug. Preserve the public API and add a regression test.",
  "agent_command": ["codeflow", "run", "--message", "{task}", "--workspace", "{workspace}", "--no-logs"],
  "fail_to_pass": [["python", "-m", "pytest", "tests/test_bugfix.py", "-q"]],
  "pass_to_pass": [["python", "-m", "pytest", "tests/test_stable_api.py", "-q"]],
  "protected_paths": ["tests/test_bugfix.py", "tests/test_stable_api.py"],
  "expected_initial_failures": 1,
  "timeout_seconds": 900,
  "token_budget": 50000,
  "environment": {}
}
```

Commands are argv arrays, not shell strings. The runner substitutes `{task}`,
`{workspace}`, and `{run_dir}`. Fail-to-pass checks must fail on the pinned
baseline and pass after the Agent run. Pass-to-pass checks must pass both before
and after. A task with a broken baseline is reported `invalid_baseline` and is
excluded from version success rates. `expected_initial_failures` counts the
fail-to-pass commands that should fail on the baseline. List every test/verifier
source file in `protected_paths`; the runner hashes them before the Agent run
and invalidates a run if the Agent changes them.

Run one repetition:

```bash
python -m benchmarks.codeflowbench.coding run path/to/task.json --output .codeflow/evidence/coding-v1 --repetition 1
```

When CodeFlow credentials are stored in a config file, include its path in the
`agent_command`, for example `"--config", "C:/Users/me/.codeflow/config.json"`.
Each run receives a fresh CodeFlow home and does not reuse user session or trace
state.

Run the same task three times with repetition values 1, 2, and 3. Each run
creates a fresh local Git clone and isolated `CODEFLOW_HOME` / trace directory.
Each run emits `result.json`, Agent stdout/stderr, verifier stdout/stderr, the
workspace and raw trace. The repository commit and task-definition digest are
recorded in the result.

## Trajectory scorer

`score-trace` reads CodeFlow's raw `logs/audit-spans.log`, not the lossy
Evolver-compressed trajectory. It reports tool failure rate, repeated identical
file reads when both input/output evidence exists, complete usage totals,
verification-command calls, and available compaction/verifier/resume signals.
It emits five separate dimensions (tool efficiency, tool reliability, context
efficiency, execution completeness, and recovery) with evidence states; there
is no blended score or arbitrary weighting.
Repeated reads require matching path, range, and output digest; a changed file
does not count as a duplicate. This signal is diagnostic and is not by itself a
quality penalty.

The current trace schema does not consistently expose a structured verifier
result, compaction event, or successful checkpoint resume. Those fields remain
`null` and are called out as missing evidence. The task runner's independent
fail-to-pass/pass-to-pass command results remain the correctness authority and
are attached to the execution-completeness dimension. Token budgets are currently a
post-run check; the generic CLI adapter cannot enforce a provider-side budget
before spending.

## Compare agent versions

Put each version's `result.json` files under separate roots. Keep task IDs and
repetition numbers aligned:

```bash
python -m benchmarks.codeflowbench.coding compare artifacts/v1 artifacts/v2 --output artifacts/comparison.json
```

The report includes unpaired and invalid-baseline denominators, task pass rate,
paired regressions/recoveries, paired latency/token deltas, and empirical
pass@1/pass@3 when the run counts support them. These are descriptive measures;
they do not imply statistical significance unless the study separately
preregisters and analyzes adequate independent repetitions.

## Execution boundary

Task test commands and Agent commands execute as subprocesses. The Git clone and
CodeFlow state directories isolate ordinary writes, but this runner is not an
OS sandbox and does not prevent commands from accessing the host or network.
Use trusted tasks and a disposable OS/container environment for untrusted code.
Keep task verifiers outside the writable Agent workspace when the task requires
strong tamper resistance. No external repository tasks are included yet; their
commit, license, reproducible environment and independent verifier need review
before they become a formal suite.
