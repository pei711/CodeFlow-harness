# SRT Shell sandbox

CodeFlow-harness supports `none`, `boxlite`, and `srt` Shell execution backends. `srt` uses Anthropic Sandbox Runtime
to isolate each Shell command with operating-system controls. It does not create a persistent VM and cannot start MCP
stdio child processes. Filesystem, MCP, and messaging tools remain governed by their existing permissions and approvals.

## Install

Node.js 20.11 or newer is required. From the repository root, install the pinned SRT release:

```bash
npm ci
```

On Windows, initialize SRT's system isolation components once from an elevated PowerShell:

```powershell
npx @anthropic-ai/sandbox-runtime windows-install
```

macOS and Linux require the platform dependencies listed in the [upstream SRT platform guide](https://github.com/anthropics/sandbox-runtime#platform-support).

## Enable

Set the backend in the CodeFlow configuration file:

```json
{
  "tools": {
    "sandbox": {
      "backend": "srt"
    }
  }
}
```

The default policy allows Shell commands to read and write the current workspace, denies reads from `.env` and `.codeflow`,
blocks writes to `.git`, and disables network access. The policy is in `codeflow/sandbox/srt-settings.json`. Configure
`srtSettingsPath` to use another SRT JSON settings file, or `srtCliPath` to point at an installed SRT CLI. If Node.js,
the CLI, or operating-system sandbox initialization is unavailable, Agent startup fails instead of falling back to a
host Shell.

SRT isolates only commands sent through `exec`. The `exec` working directory must stay within the configured workspace.
SRT does not support MCP stdio servers; use `boxlite` when those servers need to run in isolation.
