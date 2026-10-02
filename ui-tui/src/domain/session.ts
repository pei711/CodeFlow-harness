// SPDX-License-Identifier: MIT
// Copyright (c) 2026 EverMind.
// See NOTICES.md.

export const toTuiSessionKey = (sessionId: string) => (sessionId.includes(':') ? sessionId : `tui:${sessionId}`)
