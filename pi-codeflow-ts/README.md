# pi-codeflow-ts

An independent TypeScript port of selected CodeFlow runtime boundaries, hosted
on Pi Coding Agent 1.4.0. It replaces the model/tool loop behind the shape of
CodeFlow's `TurnRunner` signature, streams Pi lifecycle events through a CodeFlow-shaped
event contract, applies bounded transcript pruning before each provider request,
and enforces a fail-closed tool-effect policy before tool execution.

This first slice is not a complete rewrite of CodeFlow. Spine lane scheduling,
memory retrieval and consolidation, CodeFlow's full ContextEngine, SRT, MCP
lifecycle, resource locks, checkpoint recovery, channels, subagents and the
existing CodeFlowBench campaign remain in the Python project.

## Run

Requires Node.js 22.19 or newer, npm access to Pi 1.4.0, and a configured Pi
provider/model:

```bash
npm install --ignore-scripts
npm run type-check
npm start -- C:/work/my-repo "Inspect the project and explain the test layout"
```

The runner keeps one in-memory Pi session per conversation ID until
`closeConversation()` is called. It allows `read` tools by default; `write`,
`edit`, `bash`, and undeclared tools are blocked. The `drain` callback is
accepted for interface compatibility, but injected requests are not merged into
an active Pi run yet. Use only in
a trusted workspace until CodeFlow's sandbox and write-tool policy are ported.
The context budget uses serialized transcript length divided by four as a
rough token estimate; it is not provider tokenization. It removes older
assistant messages and complete assistant-tool-result groups while preserving
system/user messages, orphan or incomplete tool results, and the recent window.
It reports when this protected context still exceeds the budget.

The runner can be embedded behind the CodeFlow `TurnRunner` boundary:

```ts
import { PiTurnRunner } from "pi-codeflow-ts";

const runner = new PiTurnRunner({
  allowedEffects: ["read", "write"],
  toolEffects: { my_search: "read", deploy: "external" },
  contextTokenEstimate: 12_000,
});
```

Custom Pi tools default to `unknown` and are blocked until assigned an effect.
The allowed effect set remains the final gate, so registering `deploy` as
`external` does not enable it unless `external` is also allowed.

## Port map

| New TypeScript module | CodeFlow source concept |
|---|---|
| `src/protocol.ts` | `codeflow/spine/runner.py`: request, event and outcome contracts |
| `src/runner.ts` | `codeflow/agent/spine_runner.py`: execution backend at the TurnRunner boundary |
| `src/context.ts` | bounded pruning at the `prepareRequest` boundary |
| `src/policy.ts` | `codeflow/agent/tools/execution.py`: tool effect classification |

Pi's `prepareRequest` hook calls the context budget manager before every model
request. Pi's `beforeToolCall` hook calls the policy registry after parameter
validation and before execution, then preserves the hook installed by Pi's
session extension runtime. The first version sets an allowlist by effect with
read-only as its default.

## Migration boundary

The Python implementation includes considerably more behavior than this first
port. In particular, the current project does not claim parity for dynamic
memory and skills, adaptive context pressure, session persistence across turns,
read/write concurrency barriers, sandboxing, checkpoint restoration, or MCP.
These should be migrated as separate slices and compared against the existing
CodeFlowBench tasks before switching production hosts.

## Pi source basis

This project was aligned against the supplied `pi-main.zip`. Its
`packages/coding-agent/package.json` and `packages/agent/package.json` identify
version `1.4.0`; the SDK docs describe `createAgentSession`, session events and
the Pi agent's `prepareRequest` / `beforeToolCall` hooks. Dependencies are
exact-pinned to `1.4.0`. The ZIP contains no Git metadata, so an upstream commit
SHA could not be recorded.
