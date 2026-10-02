// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

// 由 Ink 的 onFrame 回调供数的轻量 FPS 跟踪器。每个条目都是一个 Ink 帧
// （React 提交帧加仅排空帧），与用户感知的运动更一致。
//
// 未设置 CODEFLOW_TUI_FPS 时没有额外成本：trackFrame 为 undefined，onFrame
// 回调会在可选链处短路。

import { atom } from 'nanostores'

import { SHOW_FPS } from '../config/env.js'

const WINDOW_SIZE = 30

export type FpsState = {
  fps: number
  /** 在 JavaScript 安全整数处回绕，使调试浮层可安全计算相邻差值。 */
  totalFrames: number
  /** 上一帧 Ink 渲染阶段的总次数。 */
  lastDurationMs: number
}

export const $fpsState = atom<FpsState>({ fps: 0, lastDurationMs: 0, totalFrames: 0 })

const timestamps: number[] = []
let totalFrames = 0

export const trackFrame = SHOW_FPS
  ? (durationMs: number) => {
      timestamps.push(performance.now())

      if (timestamps.length > WINDOW_SIZE) {
        timestamps.shift()
      }

      totalFrames++

      if (timestamps.length < 2) {
        return
      }

      const elapsed = (timestamps[timestamps.length - 1]! - timestamps[0]!) / 1000

      if (elapsed > 0) {
        $fpsState.set({
          fps: Math.round(((timestamps.length - 1) / elapsed) * 10) / 10,
          lastDurationMs: Math.round(durationMs * 100) / 100,
          totalFrames
        })
      }
    }
  : undefined
