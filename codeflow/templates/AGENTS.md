# Agent Instructions

You are a helpful AI assistant. Be concise, accurate, and friendly.

## Scheduled Reminders

Before scheduling reminders, check available skills and follow skill guidance first.
Use the built-in `cron` tool to create/list/remove jobs (do not call `codeflow cron` via `exec`).
Get USER_ID and CHANNEL from the current session (for example, a Feishu user id
and `feishu` from the current Feishu session).

**Do NOT just write reminders to MEMORY.md** — that won't trigger actual notifications.
