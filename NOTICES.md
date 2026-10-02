# Third-Party Notices

CodeFlow is an Apache-2.0-licensed derivative of Pico. It incorporates code from the
following MIT-licensed projects. Their copyright notices and license texts are
retained in `LICENSES/`.


## nanobot (base agent runtime)
- Source: https://github.com/HKUDS/nanobot
- Copyright (c) 2025 nanobot contributors
- License: MIT — see `LICENSES/MIT-nanobot.txt`
- Scope: forked at v0.1.5.post3 and modified throughout the corresponding
  `codeflow/` Runtime packages, including agent, bus, channels, CLI, config, cron,
  providers, sessions, skills, templates, and utilities.

## hermes-agent (TUI layer)
- Source: https://github.com/NousResearch/hermes-agent
- Copyright (c) 2025 Nous Research
- License: MIT — see `LICENSES/MIT-hermes-agent.txt`
- Vendored at commit: `dd0923bb89ed2dd56f82cb63656a1323f6f42e6f` (2026-05-12)
- Scope: the imported TUI remains under `ui-tui/`, including the vendored
  `@hermes/ink` fork. CodeFlow modifications include product branding, CodeFlow
  environment variables, the JSON-RPC client and Runtime bridge, and
  repository SPDX/Copyright headers.

## ink (vendored via `@hermes/ink`)
- Upstream source: https://github.com/vadimdemedes/ink
- Copyright (c) Vadym Demedes, Sindre Sorhus, and ink contributors
- License: MIT — see `LICENSES/MIT-ink.txt`
- Scope: hermes-agent ships its own fork of community ink at
  `ui-tui/packages/hermes-ink/`. CodeFlow inherits this
  vendor verbatim (package name `@hermes/ink` preserved for attribution).
  Triple attribution chain (ink contributors → Nous Research hermes-ink
  → EverMind modifications) is encoded in the 5-line SPDX header of
  every substantial file under `ui-tui/packages/hermes-ink/src/`.
  The vendored package must keep its original package name and notices unless
  a future replacement performs a fresh license and compatibility review.

# External Runtime Tools (not vendored)

The following tools are invoked by CodeFlow via `subprocess` calls but are
**not bundled or redistributed** as part of any CodeFlow release artifact.
Their attribution here is supply-chain hygiene, not a license requirement.
Users install them separately through their respective package managers.

## tui-use (TUI autotest harness Tier 1 backend)
- Source: https://github.com/onesuper/tui-use
- Copyright (c) 2026 Wei Hong (onesuper)
- License: MIT
- Install: `npm install -g tui-use` (npm package `tui-use`)
- Scope: invoked by `tests/tui/autotest/runner.py::Harness` for PTY-driven
  TUI subprocess control. Selected as Tier 1 backend per Day 0 spike
  (2026-05-20) — all 5 acceptance gates S1-S5 passed. CodeFlow does NOT
  vendor, redistribute, or modify `tui-use` source.
- If a future change vendors or modifies `tui-use`, it must add the applicable
  license text to `LICENSES/` and update this notice before release.
