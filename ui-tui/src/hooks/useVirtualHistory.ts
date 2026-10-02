// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import type { ScrollBoxHandle } from '@hermes/ink'

import {
  type RefObject,
  useCallback,
  useDeferredValue,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  useSyncExternalStore
} from 'react'

const ESTIMATE = 4
// 高度估算准确时，原本等于视口的 40 行超扫描远超所需。减半后每个滚动边缘少挂载约 20 项，
// Fiber 树更小，每帧缓冲组合工作更少。HN/CC 开发者确认，重写后大型 JSX 树造成的垃圾回收
// 压力是其主要性能问题：https://news.ycombinator.com/item?id=46699072
const OVERSCAN = 20
// 已挂载项硬上限。原值 260；持续 PageUp 追赶时约有 2.3 万个存活 Yoga 节点，渲染器 p99
// 为 106 毫秒。视口加两侧超扫描共需覆盖 80 行，按每项平均 3 行约为 25 项，因此 120 留出
// 超过 4 倍余量，即使项目很小也不会让视口空白。
const MAX_MOUNTED = 120
const COLD_START = 30
// 计算覆盖范围时使用的未测量行高下限，保证无论项目实际多小，已挂载区间都能到达视口底部。
// 项目较大时会过度挂载，但由超扫描吸收。
const PESSIMISTIC = 1
// useSyncExternalStore 快照可安全使用的最细 scrollTop 分箱。未跨箱的小滚轮刻度会完全短路 React
// 提交；Ink 仍通过 ScrollBox.forceRender 和直接读取 scrollTop 绘制。OVERSCAN 的一半可在
// 已挂载范围真正需要移动前保留至少 20 行缓冲。
const QUANTUM = OVERSCAN >> 1
// 宽度变化后保持挂载范围冻结的渲染次数，此时高度已缩放但尚未重测。第 1 次渲染跳过测量，
// 避免缩放前 Yoga 污染缩放缓存；第 2 次的 useLayoutEffect 捕获缩放后高度；第 3 次用准确数据
// 重新计算范围。
const FREEZE_RENDERS = 2
// 快速滚动时每次提交新挂载项的上限。若无此限制，单次 PageUp 进入未测量区域会按
// PESSIMISTIC=1 覆盖挂载约 190 行；每行运行 marked 词法分析和语法高亮约 3 毫秒，造成约
// 600 毫秒同步阻塞。分多次提交滑向目标可限制单次挂载成本。上限从 25 收紧到 12：每个新项
// 增加约 100 个 Fiber/Yoga 节点，25 项提交是 p99 超过 100 毫秒的主要来源。
const SLIDE_STEP = 12

const NOOP = () => {}

const upperBound = (arr: ArrayLike<number>, target: number, length = arr.length) => {
  let lo = 0
  let hi = length

  while (lo < hi) {
    const mid = (lo + hi) >> 1

    arr[mid]! <= target ? (lo = mid + 1) : (hi = mid)
  }

  return lo
}

export const shouldSetVirtualClamp = ({
  itemCount,
  liveTailActive = false,
  sticky,
  viewportHeight
}: {
  itemCount: number
  liveTailActive?: boolean
  sticky: boolean
  viewportHeight: number
}) => itemCount > 0 && viewportHeight > 0 && !sticky && !liveTailActive

export const ensureVirtualItemHeight = (
  heights: Map<string, number>,
  key: string,
  index: number,
  estimate: number,
  estimateHeight?: (index: number, key: string) => number
) => {
  const cached = heights.get(key)

  if (cached !== undefined) {
    return Math.max(1, Math.floor(cached))
  }

  const seeded = Math.max(1, Math.floor(estimateHeight?.(index, key) ?? estimate))
  heights.set(key, seeded)

  return seeded
}

export function useVirtualHistory(
  scrollRef: RefObject<ScrollBoxHandle | null>,
  items: readonly { key: string }[],
  columns: number,
  {
    estimate = ESTIMATE,
    estimateHeight,
    initialHeights,
    liveTailActive = false,
    onHeightsChange,
    overscan = OVERSCAN,
    maxMounted = MAX_MOUNTED,
    coldStartCount = COLD_START
  }: VirtualHistoryOptions = {}
) {
  const nodes = useRef(new Map<string, unknown>())
  const heights = useRef(new Map(initialHeights))
  const initialHeightsRef = useRef(initialHeights)
  const refs = useRef(new Map<string, (el: unknown) => void>())
  const onHeightsChangeRef = useRef(onHeightsChange)
  // heightCache 每次变化时递增，使偏移在下次读取时重建。使用 Ref 而非状态，在渲染阶段检查，
  // 不产生额外提交。
  const offsetVersion = useRef(0)

  // 缓存偏移：复用以（项目数、版本）为键的 Float64Array，仅在实际变化时重建。旧方案每次
  // 渲染都分配新的 Array(n+1)；n=1 万时，流式期间每次渲染产生约 80KB 垃圾回收压力。
  const offsetsCache = useRef<{ arr: Float64Array; n: number; version: number }>({
    arr: new Float64Array(0),
    n: -1,
    version: -1
  })

  const [hasScrollRef, setHasScrollRef] = useState(false)
  // 高度缓存写入发生在布局副作用中；递增一次，使偏移和钳制边界无需等待下次滚动/输入便重建。
  const [measuredHeightVersion, bumpMeasuredHeightVersion] = useState(0)
  const metrics = useRef({ sticky: true, top: 0, vp: 0 })
  const lastScrollTopRef = useRef(0)

  // 宽度变化时按 oldCols/newCols 缩放缓存高度，而不是清空。清空会强制悲观回溯，一次挂载约
  // 190 行，每行重新运行 marked.lexer 和语法高亮约 3 毫秒。冻结挂载范围两次渲染以保留热
  // 记忆；跳过一次测量，避免 useLayoutEffect 用缩放前 Yoga 高度污染缩放缓存。
  const prevColumns = useRef(columns)
  const skipMeasurement = useRef(false)
  const prevRange = useRef<null | readonly [number, number]>(null)
  const freezeRenders = useRef(0)

  onHeightsChangeRef.current = onHeightsChange

  if (initialHeightsRef.current !== initialHeights) {
    initialHeightsRef.current = initialHeights
    heights.current = new Map(initialHeights)
    offsetVersion.current++
  }

  if (prevColumns.current !== columns && prevColumns.current > 0 && columns > 0) {
    const ratio = prevColumns.current / columns

    prevColumns.current = columns

    for (const [k, h] of heights.current) {
      heights.current.set(k, Math.max(1, Math.round(h * ratio)))
    }

    offsetVersion.current++
    skipMeasurement.current = true
    freezeRenders.current = FREEZE_RENDERS
  }

  useLayoutEffect(() => {
    setHasScrollRef(Boolean(scrollRef.current))
  }, [scrollRef])

  // 量化快照：同一分箱内的滚动（多数滚轮刻度）产生相同数字，使 React.Object.is 完全短路
  // 提交。sticky 状态通过符号位折叠，因此从 sticky 到 broken 的转换也会触发。使用目标值
  // （已提交值加 pendingDelta）而非已提交 scrollTop，使 scrollBy 通知在 Ink 排空帧需要子项前
  // 立即为目标位置重新挂载。
  const subscribe = useCallback(
    (cb: () => void) => (hasScrollRef ? scrollRef.current?.subscribe(cb) : null) ?? NOOP,
    [hasScrollRef, scrollRef]
  )

  useSyncExternalStore(
    subscribe,
    () => {
      const s = scrollRef.current

      if (!s) {
        return NaN
      }

      const target = s.getScrollTop() + s.getPendingDelta()
      const bin = Math.floor(target / QUANTUM)

      return s.isSticky() ? ~bin : bin
    },
    () => NaN
  )

  useEffect(() => {
    const keep = new Set(items.map(i => i.key))
    let dirty = false

    for (const k of heights.current.keys()) {
      if (!keep.has(k)) {
        heights.current.delete(k)
        nodes.current.delete(k)
        refs.current.delete(k)
        dirty = true
      }
    }

    if (dirty) {
      offsetVersion.current++
    }
  }, [items])

  // 偏移：跨渲染复用 Float64Array；heightCache 写入方（measureRef、缩放、垃圾回收）递增
  // offsetVersion 时使其失效。二分搜索兼容任一单调来源，因此仅在发生变化时重建。
  const n = items.length

  if (offsetsCache.current.version !== offsetVersion.current || offsetsCache.current.n !== n) {
    const arr = offsetsCache.current.arr.length >= n + 1 ? offsetsCache.current.arr : new Float64Array(n + 1)

    arr[0] = 0

    for (let i = 0; i < n; i++) {
      arr[i + 1] = arr[i]! + ensureVirtualItemHeight(heights.current, items[i]!.key, i, estimate, estimateHeight)
    }

    offsetsCache.current = { arr, n, version: offsetVersion.current }
  }

  const offsets = offsetsCache.current.arr
  const total = offsets[n] ?? 0
  const top = Math.max(0, scrollRef.current?.getScrollTop() ?? 0)
  const pendingDelta = scrollRef.current?.getPendingDelta() ?? 0
  const target = Math.max(0, top + pendingDelta)
  const vp = Math.max(0, scrollRef.current?.getViewportHeight() ?? 0)
  const sticky = scrollRef.current?.isSticky() ?? true
  const recentManual = Date.now() - (scrollRef.current?.getLastManualScrollAt() ?? 0) < 1200

  // 冻结期间若项目收缩越过冻结范围起点（/clear、压缩），则丢弃冻结范围；钳制会收缩成
  // 空挂载并闪白。此时进入正常路径。
  const frozenRange =
    freezeRenders.current > 0 && prevRange.current && prevRange.current[0] < n ? prevRange.current : null

  let start = 0
  let end = n

  if (frozenRange) {
    start = frozenRange[0]
    end = Math.min(frozenRange[1], n)
  } else if (n > 0) {
    if (vp <= 0) {
      start = Math.max(0, n - coldStartCount)
    } else if (sticky && !recentManual) {
      const budget = vp + overscan
      start = n

      while (start > 0 && total - offsets[start - 1]! < budget) {
        start--
      }
    } else {
      // 用户向上滚动。覆盖 [committed..target]，使每个排空帧都在范围内。Claude Code 将区间
      // 限制为视口的 3 倍，避免无限增长的 pendingDelta（如 MX Master 自由滚动）耗尽挂载预算；
      // 追赶期间由 setClampBounds 显示已挂载内容边缘。
      const MAX_SPAN = vp * 3
      const rawLo = Math.min(top, target)
      const rawHi = Math.max(top, target)
      const span = rawHi - rawLo
      const clampedLo = span > MAX_SPAN ? (pendingDelta < 0 ? rawHi - MAX_SPAN : rawLo) : rawLo
      const clampedHi = clampedLo + Math.min(span, MAX_SPAN)
      const lo = Math.max(0, clampedLo - overscan)
      const hi = clampedHi + vp + overscan

      // 二分搜索——offsets 单调。n 超过 1 万时，线性遍历为 O(n)，滚动期间每次渲染约 2 毫秒。
      start = Math.max(0, Math.min(n - 1, upperBound(offsets, lo, n + 1) - 1))
      end = Math.max(start + 1, Math.min(n, upperBound(offsets, hi, n + 1)))
    }
  }

  if (end - start > maxMounted) {
    sticky ? (start = Math.max(0, end - maxMounted)) : (end = Math.min(n, start + maxMounted))
  }

  // 覆盖保证：确保真实或悲观高度之和至少为 viewportH + 2*overscan，使项目很小时视口仍被
  // 实际覆盖。未缓存项目使用下限 1，因而是悲观估计；项目较大时会过度挂载，但绝不会露出空白。
  if (n > 0 && vp > 0 && !frozenRange) {
    const needed = vp + 2 * overscan
    let coverage = 0

    for (let i = start; i < end; i++) {
      coverage += ensureVirtualItemHeight(heights.current, items[i]!.key, i, PESSIMISTIC, estimateHeight)
    }

    if (sticky) {
      const minStart = Math.max(0, end - maxMounted)

      while (start > minStart && coverage < needed) {
        start--
        coverage += ensureVirtualItemHeight(heights.current, items[start]!.key, start, PESSIMISTIC, estimateHeight)
      }
    } else {
      const maxEnd = Math.min(n, start + maxMounted)

      while (end < maxEnd && coverage < needed) {
        coverage += ensureVirtualItemHeight(heights.current, items[end]!.key, end, PESSIMISTIC, estimateHeight)
        end++
      }
    }
  }

  // 滑动上限：限制本次提交新挂载项数量。按滚动速度启用，即自上次提交后的 scrollTop 差值绝对值
  // 加 pendingDelta 绝对值超过两倍视口；按键重复 PageUp 每次移动约半个视口。兼容 scrollBy
  // （pendingDelta）和 scrollTo（直接写入）。普通单次 PageUp 跳过；追赶时钳制将视口固定在
  // 已挂载边缘，避免空白。只限制范围增长，收缩不受限。
  if (!frozenRange && prevRange.current && vp > 0) {
    const velocity = Math.abs(top - lastScrollTopRef.current) + Math.abs(pendingDelta)

    if (velocity > vp * 2) {
      const [pS, pE] = prevRange.current

      start = Math.max(start, pS - SLIDE_STEP)
      end = Math.min(end, pE + SLIDE_STEP)

      // 大幅跳过受限终点时范围可能倒置（start > end）；从新起点挂载 SLIDE_STEP 项，
      // 避免追赶期间视口空白。
      if (start > end) {
        end = Math.min(start + SLIDE_STEP, n)
      }
    }
  }

  lastScrollTopRef.current = top

  if (freezeRenders.current > 0) {
    freezeRenders.current--
  } else {
    prevRange.current = [start, end]
  }

  // 通过 useDeferredValue 对范围增长做时间切片。紧急渲染让 Ink 用旧范围继续绘制，全部命中缓存；
  // 延迟渲染在非阻塞后台提交中切换到新范围，新挂载会运行 Markdown 和语法高亮。钳制
  // setClampBounds 将视口固定在已挂载边缘，避免延迟范围短暂落后产生视觉伪影。只延迟范围增长；
  // 收缩成本很低，因为卸载只移除 Fiber，无需解析。
  const dStart = useDeferredValue(start)
  const dEnd = useDeferredValue(end)
  let effStart = start < dStart ? dStart : start
  let effEnd = end > dEnd ? dEnd : end

  // 范围倒置（大幅跳转且延迟值落后）或粘性吸底时跳过延迟；scrollToBottom 必须立即挂载尾部，
  // 让 maxScroll 落在内容而非底部占位上。
  if (effStart > effEnd || sticky) {
    effStart = start
    effEnd = end
  }

  // 向下滚动时绕过 effEnd 延迟，使尾部立即挂载。否则钳制会让 scrollTop 停在真实底部之前，
  // 用户会感觉“卡在底部前”。effStart 仍延迟，使向上滚动保持时间切片，因为旧消息会在挂载时解析。
  if (pendingDelta > 0) {
    effEnd = end
  }

  // 最终强制保持 O(viewport)。上方延迟与绕过组合可能泄漏：持续 PageUp 时，并发模式会跨提交交错
  // dStart 更新与 effEnd=end 绕过，使有效窗口比任一单独边界更宽。按视口位置裁剪远端边缘，
  // 不按 pendingDelta 方向；后者会在并发调度稳定途中翻转并猛拉 scrollTop。
  if (effEnd - effStart > maxMounted && vp > 0) {
    const mid = (offsets[effStart]! + offsets[effEnd]!) / 2

    if (top < mid) {
      effEnd = effStart + maxMounted
    } else {
      effStart = effEnd - maxMounted
    }
  }

  const measureRef = useCallback((key: string) => {
    let fn = refs.current.get(key)

    if (!fn) {
      fn = (el: unknown) => {
        if (el) {
          nodes.current.set(key, el)

          return
        }

        // 卸载时测量：yogaNode 此时仍有效，协调器会在 removeChild -> freeRecursive 前调用
        // ref(null)，因此在 WASM 释放前取得最终高度。否则快速平移时滚出视口的项目会在
        // heightCache 中保留陈旧估计，偏移计算持续漂移到下次挂载/重挂载周期。
        const existing = nodes.current.get(key) as MeasuredNode | undefined
        const h = Math.ceil(existing?.yogaNode?.getComputedHeight?.() ?? 0)

        if (h > 0 && heights.current.get(key) !== h) {
          heights.current.set(key, h)
          offsetVersion.current++
          onHeightsChangeRef.current?.(heights.current)
        }

        nodes.current.delete(key)
      }

      refs.current.set(key, fn)
    }

    return fn
  }, [])

  useLayoutEffect(() => {
    const s = scrollRef.current
    let dirty = false
    let heightDirty = false

    // 向渲染器提供已挂载行覆盖范围，用于被动滚动钳制。钳制必须使用有效的延迟范围，而非即时范围。
    // 快速滚动时，即时 [start,end] 可能已覆盖新 scrollTop，但子项仍按延迟范围渲染。若钳制使用
    // 即时边界，render-node-to-output 的排空门会越过延迟子项范围，使视口落入占位并闪白。
    if (s && shouldSetVirtualClamp({ itemCount: n, liveTailActive, sticky, viewportHeight: vp })) {
      const effTopSpacer = offsets[effStart] ?? 0
      const effBottom = offsets[effEnd] ?? total
      // effEnd=n 时没有 bottomSpacer；使用 Infinity，让 render-node-to-output 自身的
      // Math.min(cur, maxScroll) 控制。若在此使用 offsets[n]，会固化落后 Yoga 一次渲染的
      // heightCache；流式期间尾项缓存高度落后真实高度，sticky-break 会钳制到真实最大值下方，
      // 把流式文本推离视口。
      const clampMin = effStart === 0 ? 0 : effTopSpacer
      const clampMax = effEnd === n ? Infinity : Math.max(effTopSpacer, effBottom - vp)

      s.setClampBounds(clampMin, clampMax)
    } else {
      s?.setClampBounds(undefined, undefined)
    }

    if (skipMeasurement.current) {
      skipMeasurement.current = false
    } else {
      for (let i = effStart; i < effEnd; i++) {
        const k = items[i]?.key

        if (!k) {
          continue
        }

        const h = Math.ceil((nodes.current.get(k) as MeasuredNode | undefined)?.yogaNode?.getComputedHeight?.() ?? 0)

        if (h > 0 && heights.current.get(k) !== h) {
          heights.current.set(k, h)
          dirty = true
          heightDirty = true
        }
      }
    }

    if (s) {
      const next = {
        sticky: s.isSticky(),
        top: Math.max(0, s.getScrollTop() + s.getPendingDelta()),
        vp: Math.max(0, s.getViewportHeight())
      }

      if (
        next.sticky !== metrics.current.sticky ||
        next.top !== metrics.current.top ||
        next.vp !== metrics.current.vp
      ) {
        metrics.current = next
        dirty = true
      }
    }

    if (dirty) {
      offsetVersion.current++
      onHeightsChangeRef.current?.(heights.current)
    }

    if (heightDirty) {
      bumpMeasuredHeightVersion(n => n + 1)
    }
  }, [effEnd, effStart, items, liveTailActive, measuredHeightVersion, n, offsets, scrollRef, sticky, total, vp])

  return {
    bottomSpacer: Math.max(0, total - (offsets[effEnd] ?? total)),
    end: effEnd,
    measureRef,
    offsets,
    start: effStart,
    topSpacer: offsets[effStart] ?? 0
  }
}

interface MeasuredNode {
  yogaNode?: { getComputedHeight?: () => number } | null
}

interface VirtualHistoryOptions {
  coldStartCount?: number
  estimate?: number
  estimateHeight?: (index: number, key: string) => number
  initialHeights?: ReadonlyMap<string, number>
  liveTailActive?: boolean
  maxMounted?: number
  onHeightsChange?: (heights: ReadonlyMap<string, number>) => void
  overscan?: number
}
