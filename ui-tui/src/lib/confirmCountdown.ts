// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

export const CONFIRM_COUNTDOWN_SECONDS = 30

export interface CountdownTick {
  autoCancel: boolean
  remaining: number
}

/**
 * 确认倒计时每秒执行一次的纯函数步骤。返回下一个 `remaining` 值（下限为 0）
 * 以及是否触底；触底时调用方必须自动取消，即回答 false。
 */
export const tickCountdown = (remaining: number): CountdownTick => {
  const next = Math.max(0, remaining - 1)

  return { autoCancel: next <= 0, remaining: next }
}

export const buildConfirmRespond = (requestId: string, answer: boolean) => ({
  answer,
  request_id: requestId
})
