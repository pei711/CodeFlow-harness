// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

// 滚轮滚动加速状态机。
//
// 每事件一行在触控板（每秒 200 多个事件）和持续滚轮上显得迟钝；每事件六行又像瞬移，破坏精度。
// 根据事件间隔和方向翻转使用启发式：
//
//   间隔 < 5ms                  → 同批突发 → 每事件 1 行
//   间隔 < 40ms（原生）         → 增幅 +0.3，上限 6
//   间隔 80-500ms（xterm.js）   → mult = 1 + (mult-1)·0.5^(gap/150) + 5·decay
//                                 慢速上限 3，快速上限 6
//   间隔 > 500ms                → 重置，保持有意单击的响应性
//   翻转后 200ms 内翻回          → 编码器回弹 → 启用滚轮模式（粘性上限）
//   连续 5 个 <5ms 事件          → 触控板轻扫 → 退出滚轮模式
//
// 原生终端（Ghostty、iTerm2）与 xterm.js 宿主（VS Code、Cursor）的滚轮事件节奏不同，
// 因此使用两条路径。

import { isXtermJs } from '@hermes/ink'

// ── 原生终端（Ghostty、iTerm2、WezTerm 等）────────────────────────────
const WHEEL_ACCEL_WINDOW_MS = 40
const WHEEL_ACCEL_STEP = 0.3
const WHEEL_ACCEL_MAX = 6

// ── 编码器回弹/滚轮模式（机械滚轮）───────────────────────────────────
const WHEEL_BOUNCE_GAP_MAX_MS = 200
const WHEEL_MODE_STEP = 15
const WHEEL_MODE_CAP = 15
const WHEEL_MODE_RAMP = 3
const WHEEL_MODE_IDLE_DISENGAGE_MS = 1500

// ── xterm.js（VS Code / Cursor / 浏览器终端）───────────────────────────
const WHEEL_DECAY_HALFLIFE_MS = 150
const WHEEL_DECAY_STEP = 5
const WHEEL_BURST_MS = 5
const WHEEL_DECAY_GAP_MS = 80
const WHEEL_DECAY_CAP_SLOW = 3
const WHEEL_DECAY_CAP_FAST = 6
const WHEEL_DECAY_IDLE_MS = 500

export type WheelAccelState = {
  time: number
  mult: number
  dir: 0 | 1 | -1
  xtermJs: boolean
  /** 累积的小数滚动量（xterm.js）。scrollBy 会向下取整，因此不累积时 1.5 倍
   *  始终只滚动一行；保留余数后会得到 1、2、1、2，长期吞吐量才正确。 */
  frac: number
  /** 原生路径每事件的基准行数。空闲或反向时重置，并在此基础上递增；xterm.js
   *  路径忽略该值。 */
  base: number
  /** 原生路径延迟处理的方向翻转；是抖动还是反向由下一个事件决定。 */
  pendingFlip: boolean
  /** 在抖动窗口内发生翻转后又翻回时锁定；空闲解除或触控板突发事件会清除。 */
  wheelMode: boolean
  /** 连续小于 5 毫秒的事件数；达到 5 个即视为触控板快速滑动并解除加速。 */
  burstCount: number
}

export function initWheelAccel(xtermJs = false, base = 1): WheelAccelState {
  return { burstCount: 0, base, dir: 0, frac: 0, mult: base, pendingFlip: false, time: 0, wheelMode: false, xtermJs }
}

/** CODEFLOW_TUI_SCROLL_SPEED（也可用 CLAUDE_CODE_SCROLL_SPEED 保持兼容），
 *  默认值为 1，并限制在 (0, 20]。 */
export function readScrollSpeedBase(): number {
  const n = parseFloat(process.env.CODEFLOW_TUI_SCROLL_SPEED ?? process.env.CLAUDE_CODE_SCROLL_SPEED ?? '')

  return Number.isFinite(n) && n > 0 ? Math.min(n, 20) : 1
}

export function initWheelAccelForHost(): WheelAccelState {
  return initWheelAccel(isXtermJs(), readScrollSpeedBase())
}

/** 计算一次滚轮事件的行数并修改 `state`。因抖动检测而延迟方向翻转时返回 0，
 *  调用点遇到 0 应不执行操作。 */
export function computeWheelStep(state: WheelAccelState, dir: -1 | 1, now: number): number {
  return state.xtermJs ? xtermJsStep(state, dir, now) : nativeStep(state, dir, now)
}

function nativeStep(state: WheelAccelState, dir: -1 | 1, now: number): number {
  // 先执行空闲退出，避免待处理回弹把“用户暂停 1.5 秒后单击鼠标”掩盖为真实反转。
  if (state.wheelMode && now - state.time > WHEEL_MODE_IDLE_DISENGAGE_MS) {
    state.wheelMode = false
    state.burstCount = 0
    state.mult = state.base
  }

  if (state.pendingFlip) {
    state.pendingFlip = false

    if (dir !== state.dir || now - state.time > WHEEL_BOUNCE_GAP_MAX_MS) {
      // 真实反转（翻转持续或翻回太晚），提交。延迟事件的一行会丢失，此延迟可以接受。
      state.dir = dir
      state.time = now
      state.mult = state.base

      return Math.floor(state.mult)
    }

    state.wheelMode = true
  }

  const gap = now - state.time

  if (dir !== state.dir && state.dir !== 0) {
    state.pendingFlip = true
    state.time = now

    return 0
  }

  state.dir = dir
  state.time = now

  if (state.wheelMode) {
    if (gap < WHEEL_BURST_MS) {
      // 同批突发（SGR 比例式）或触控板轻扫。每事件一行；触控板轻扫会触发突发计数退出。
      if (++state.burstCount >= 5) {
        state.wheelMode = false
        state.burstCount = 0
        state.mult = state.base
      } else {
        return 1
      }
    } else {
      state.burstCount = 0
    }
  }

  if (state.wheelMode) {
    const m = Math.pow(0.5, gap / WHEEL_DECAY_HALFLIFE_MS)
    const cap = Math.max(WHEEL_MODE_CAP, state.base * 2)
    const next = 1 + (state.mult - 1) * m + WHEEL_MODE_STEP * m

    state.mult = Math.min(cap, next, state.mult + WHEEL_MODE_RAMP)

    return Math.floor(state.mult)
  }

  // 触控板/高分辨率原生输入使用严格 40 毫秒窗口；窗口内递增，更慢则重置到基线。
  if (gap > WHEEL_ACCEL_WINDOW_MS) {
    state.mult = state.base
  } else {
    const cap = Math.max(WHEEL_ACCEL_MAX, state.base * 2)

    state.mult = Math.min(cap, state.mult + WHEEL_ACCEL_STEP)
  }

  return Math.floor(state.mult)
}

function xtermJsStep(state: WheelAccelState, dir: -1 | 1, now: number): number {
  const gap = now - state.time
  const sameDir = dir === state.dir

  state.time = now
  state.dir = dir

  if (sameDir && gap < WHEEL_BURST_MS) {
    return 1
  }

  if (!sameDir || gap > WHEEL_DECAY_IDLE_MS) {
    // 反转或长时间空闲后从 2 开始，使暂停后的第一次滚动有明显移动。
    state.mult = 2
    state.frac = 0
  } else {
    const m = Math.pow(0.5, gap / WHEEL_DECAY_HALFLIFE_MS)
    const cap = gap >= WHEEL_DECAY_GAP_MS ? WHEEL_DECAY_CAP_SLOW : WHEEL_DECAY_CAP_FAST

    state.mult = Math.min(cap, 1 + (state.mult - 1) * m + WHEEL_DECAY_STEP * m)
  }

  const total = state.mult + state.frac
  const rows = Math.floor(total)

  state.frac = total - rows

  return rows
}
