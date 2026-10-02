// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

// 完整渲染流水线的性能检测。
//
//   PerfPane（React.Profiler）  → 各面板提交耗时
//   logFrameEvent（ink.onFrame）→ yoga、renderer、diff、optimize、write
//                                 各阶段，以及 yoga 计数器和滚动快速路径
//
// 两者均由 CODEFLOW_DEV_PERF=1 启用，并输出 JSON Lines（默认 ~/.codeflow/perf.log，
// 可用 CODEFLOW_DEV_PERF_LOG 覆盖）。通过 { src: 'react' | 'frame' } 标记供 jq
// 使用。CODEFLOW_DEV_PERF_MS 默认为 2，会跳过低于阈值的空闲帧；设为 0 可全部捕获。
//
// 未启用时没有额外成本：PerfPane 直接返回子节点，logFrameEvent 为 undefined，
// 因而 Ink 不承担计时开销。

import type { FrameEvent } from '@hermes/ink'

import { scrollFastPathStats } from '@hermes/ink'
import { appendFileSync, mkdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { Profiler, type ProfilerOnRenderCallback, type ReactNode } from 'react'

import { getCodeFlowHome } from '../config/paths.js'

const ENABLED = /^(?:1|true|yes|on)$/i.test((process.env.CODEFLOW_DEV_PERF ?? '').trim())
const THRESHOLD_MS = Number(process.env.CODEFLOW_DEV_PERF_MS ?? '2') || 0
const LOG_PATH = process.env.CODEFLOW_DEV_PERF_LOG?.trim() || join(getCodeFlowHome(), 'perf.log')

let logReady = false

const writeRow = (row: Record<string, unknown>) => {
  if (!logReady) {
    logReady = true

    try {
      mkdirSync(dirname(LOG_PATH), { recursive: true })
    } catch {
      // 尽力而为，不能因记录样本而使 TUI 崩溃。
    }
  }

  try {
    appendFileSync(LOG_PATH, `${JSON.stringify(row)}\n`)
  } catch {
    /* 尽力而为。 */
  }
}

const round2 = (n: number) => Math.round(n * 100) / 100

const onRender: ProfilerOnRenderCallback = (id, phase, actualMs, baseMs, startTime, commitTime) => {
  if (actualMs < THRESHOLD_MS) {
    return
  }

  writeRow({
    actualMs: round2(actualMs),
    baseMs: round2(baseMs),
    commitTimeMs: round2(commitTime),
    id,
    phase,
    src: 'react',
    startTimeMs: round2(startTime),
    ts: Date.now()
  })
}

export function PerfPane({ children, id }: { children: ReactNode; id: string }) {
  if (!ENABLED) {
    return children
  }

  return (
    <Profiler id={id} onRender={onRender}>
      {children}
    </Profiler>
  )
}

export const logFrameEvent = ENABLED
  ? (event: FrameEvent) => {
      if (event.durationMs < THRESHOLD_MS) {
        return
      }

      writeRow({
        durationMs: round2(event.durationMs),
        // 这是累计计数器，消费者需对相邻值求差得到每帧增量。
        fastPath: { ...scrollFastPathStats, declined: { ...scrollFastPathStats.declined } },
        flickers: event.flickers.length ? event.flickers : undefined,
        phases: event.phases
          ? {
              ...event.phases,
              commit: round2(event.phases.commit),
              diff: round2(event.phases.diff),
              optimize: round2(event.phases.optimize),
              prevFrameDrainMs: round2(event.phases.prevFrameDrainMs),
              renderer: round2(event.phases.renderer),
              write: round2(event.phases.write),
              yoga: round2(event.phases.yoga)
            }
          : undefined,
        src: 'frame',
        ts: Date.now()
      })
    }
  : undefined

export const PERF_ENABLED = ENABLED
export const PERF_LOG_PATH = LOG_PATH
