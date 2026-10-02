// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { type HeapDumpResult, performHeapDump } from './memory.js'

export type MemoryLevel = 'critical' | 'high' | 'normal'

export interface MemorySnapshot {
  heapUsed: number
  level: MemoryLevel
  rss: number
}

export interface MemoryMonitorOptions {
  criticalBytes?: number
  highBytes?: number
  intervalMs?: number
  onCritical?: (snap: MemorySnapshot, dump: HeapDumpResult | null) => void
  onHigh?: (snap: MemorySnapshot, dump: HeapDumpResult | null) => void
}

const GB = 1024 ** 3

// 延迟导入 @hermes/ink：若在模块顶层加载 `@hermes/ink`，完整约 414KB 的 Ink
// 包（React、渲染器、组件和钩子）会在 Python 网关生成前进入关键路径，导致每次
// 冷启动 `codeflow --tui` 时，gw.start() 前串行增加约 150 毫秒的 Node 工作。
//
// evictInkCaches 只在 `tick()` 内运行；后者每 10 秒触发一次，且仅在堆压力越过
// 高水位时执行，此时应用入口早已加载 Ink。因此动态导入在热路径上不会产生额外
// 工作（模块已在 ESM 缓存中）。若启动尖峰在应用登记自身 Ink 导入前意外触发
// 阈值，也只会在需要它的那次 tick 中支付一次加载成本。
let _evictInkCaches: ((level: 'all' | 'half') => unknown) | null = null
let _evictInkCachesPromise: Promise<(level: 'all' | 'half') => unknown> | null = null

async function _ensureEvictInkCaches(): Promise<(level: 'all' | 'half') => unknown> {
  if (_evictInkCaches) {
    return _evictInkCaches
  }

  _evictInkCachesPromise ??= import('@hermes/ink')
    .then(mod => {
      _evictInkCaches = mod.evictInkCaches as (level: 'all' | 'half') => unknown

      return _evictInkCaches
    })
    .catch(err => {
      _evictInkCachesPromise = null
      throw err
    })

  return _evictInkCachesPromise
}

export function startMemoryMonitor({
  criticalBytes = 2.5 * GB,
  highBytes = 1.5 * GB,
  intervalMs = 10_000,
  onCritical,
  onHigh
}: MemoryMonitorOptions = {}): () => void {
  const dumped = new Set<Exclude<MemoryLevel, 'normal'>>()
  const inFlight = new Set<Exclude<MemoryLevel, 'normal'>>()

  const tick = async () => {
    const { heapUsed, rss } = process.memoryUsage()
    const level: MemoryLevel = heapUsed >= criticalBytes ? 'critical' : heapUsed >= highBytes ? 'high' : 'normal'

    if (level === 'normal') {
      dumped.clear()

      return
    }

    if (dumped.has(level) || inFlight.has(level)) {
      return
    }

    inFlight.add(level)

    // 转储或退出前裁剪 Ink 内容缓存：`high`（可恢复）时裁剪一半，`critical`
    // 时全部裁剪，以在转储后降低 RSS 并让用户继续运行。延迟导入使
    // `@hermes/ink` 不进入冷启动关键路径；启动 10 秒后 tick 触发时，应用已经
    // 加载同一模块，因此会直接从 ESM 缓存解析。
    try {
      try {
        const evictInkCaches = await _ensureEvictInkCaches()
        evictInkCaches(level === 'critical' ? 'all' : 'half')
      } catch {
        // 尽力而为：动态导入无论因何失败，都继续执行下方堆转储以提供诊断信息。
      }

      dumped.add(level)
      const dump = await performHeapDump(level === 'critical' ? 'auto-critical' : 'auto-high').catch(() => null)

      const snap: MemorySnapshot = { heapUsed, level, rss }

      ;(level === 'critical' ? onCritical : onHigh)?.(snap, dump)
    } finally {
      inFlight.delete(level)
    }
  }

  const handle = setInterval(() => void tick(), intervalMs)

  handle.unref?.()

  return () => clearInterval(handle)
}
