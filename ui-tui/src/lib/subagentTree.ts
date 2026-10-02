// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import type { SubagentAggregate, SubagentNode, SubagentProgress } from '../types.js'

const ROOT_KEY = '__root__'

/**
 * 从按事件顺序排列的扁平列表重建子智能体生成树。
 *
 * 按 `parentId` 分组；缺少 `parentId` 或指向未知子智能体时，视为当前轮次的
 * 顶层生成。父节点内的子节点先按 `depth`、再按 `index` 排序，与
 * `turnController.upsertSubagent` 使用同一组键，因此无论网关事件如何在网络中
 * 重排，渲染顺序都与生成顺序一致。
 *
 * 旧网关会省略 `parentId`，此时每个子智能体都是顶层节点，树按扁平形式渲染，
 * 与可观测性功能加入前的行为一致。
 */
export function buildSubagentTree(items: readonly SubagentProgress[]): SubagentNode[] {
  if (!items.length) {
    return []
  }

  const byParent = new Map<string, SubagentProgress[]>()
  const known = new Set<string>()

  for (const item of items) {
    known.add(item.id)
  }

  for (const item of items) {
    const parentKey = item.parentId && known.has(item.parentId) ? item.parentId : ROOT_KEY
    const bucket = byParent.get(parentKey) ?? []
    bucket.push(item)
    byParent.set(parentKey, bucket)
  }

  for (const bucket of byParent.values()) {
    bucket.sort((a, b) => a.depth - b.depth || a.index - b.index)
  }

  const build = (item: SubagentProgress): SubagentNode => {
    const kids = byParent.get(item.id) ?? []
    const children = kids.map(build)

    return { aggregate: aggregate(item, children), children, item }
  }

  return (byParent.get(ROOT_KEY) ?? []).map(build)
}

/**
 * 汇总一个节点完整子树的计数。保持纯函数，使实时视图与事后回放无需改动即可
 * 共享同一渲染器。
 *
 * `hotness` 表示整棵子树每秒调用工具的次数，粗略反映该分支的工作量。浮层和
 * 行内视图用它为树形轨道着色，便于快速识别开销较大的分支。
 */
export function aggregate(item: SubagentProgress, children: readonly SubagentNode[]): SubagentAggregate {
  let totalTools = item.toolCount ?? 0
  let totalDuration = item.durationSeconds ?? 0
  let descendantCount = 0
  let activeCount = isRunning(item) ? 1 : 0
  let maxDepthFromHere = 0
  let inputTokens = item.inputTokens ?? 0
  let outputTokens = item.outputTokens ?? 0
  let costUsd = item.costUsd ?? 0
  let filesTouched = (item.filesRead?.length ?? 0) + (item.filesWritten?.length ?? 0)

  for (const child of children) {
    totalTools += child.aggregate.totalTools
    totalDuration += child.aggregate.totalDuration
    descendantCount += child.aggregate.descendantCount + 1
    activeCount += child.aggregate.activeCount
    maxDepthFromHere = Math.max(maxDepthFromHere, child.aggregate.maxDepthFromHere + 1)
    inputTokens += child.aggregate.inputTokens
    outputTokens += child.aggregate.outputTokens
    costUsd += child.aggregate.costUsd
    filesTouched += child.aggregate.filesTouched
  }

  const hotness = totalDuration > 0 ? totalTools / totalDuration : 0

  return {
    activeCount,
    costUsd,
    descendantCount,
    filesTouched,
    hotness,
    inputTokens,
    maxDepthFromHere,
    outputTokens,
    totalDuration,
    totalTools
  }
}

/**
 * 各深度层级的子智能体数量，以深度为索引（0 表示顶层）。用于驱动行内迷你图
 * （`▁▃▇▅`）和状态栏 HUD。
 */
export function widthByDepth(tree: readonly SubagentNode[]): number[] {
  const widths: number[] = []

  const walk = (nodes: readonly SubagentNode[], depth: number) => {
    if (!nodes.length) {
      return
    }

    widths[depth] = (widths[depth] ?? 0) + nodes.length

    for (const node of nodes) {
      walk(node.children, depth + 1)
    }
  }

  walk(tree, 0)

  return widths
}

/**
 * 整棵树的扁平汇总数据，用于摘要标签标题。
 */
export function treeTotals(tree: readonly SubagentNode[]): SubagentAggregate {
  let totalTools = 0
  let totalDuration = 0
  let descendantCount = 0
  let activeCount = 0
  let maxDepthFromHere = 0
  let inputTokens = 0
  let outputTokens = 0
  let costUsd = 0
  let filesTouched = 0

  for (const node of tree) {
    totalTools += node.aggregate.totalTools
    totalDuration += node.aggregate.totalDuration
    descendantCount += node.aggregate.descendantCount + 1
    activeCount += node.aggregate.activeCount
    maxDepthFromHere = Math.max(maxDepthFromHere, node.aggregate.maxDepthFromHere + 1)
    inputTokens += node.aggregate.inputTokens
    outputTokens += node.aggregate.outputTokens
    costUsd += node.aggregate.costUsd
    filesTouched += node.aggregate.filesTouched
  }

  const hotness = totalDuration > 0 ? totalTools / totalDuration : 0

  return {
    activeCount,
    costUsd,
    descendantCount,
    filesTouched,
    hotness,
    inputTokens,
    maxDepthFromHere,
    outputTokens,
    totalDuration,
    totalTools
  }
}

/**
 * 按访问顺序将树展平，供键盘导航以及“终止子树”时逐个后代发起 RPC 使用。
 */
export function flattenTree(tree: readonly SubagentNode[]): SubagentNode[] {
  const out: SubagentNode[] = []

  const walk = (nodes: readonly SubagentNode[]) => {
    for (const node of nodes) {
      out.push(node)
      walk(node.children)
    }
  }

  walk(tree)

  return out
}

/**
 * 收集指定节点所有后代的标识，不包括节点自身。
 */
export function descendantIds(node: SubagentNode): string[] {
  const ids: string[] = []

  const walk = (children: readonly SubagentNode[]) => {
    for (const child of children) {
      ids.push(child.item.id)
      walk(child.children)
    }
  }

  walk(node.children)

  return ids
}

export function isRunning(item: Pick<SubagentProgress, 'status'>): boolean {
  return item.status === 'running' || item.status === 'queued'
}

const SPARK_RAMP = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'] as const

/**
 * 根据正整数数组生成八级 Unicode 柱状迷你图。零值渲染为空格，避免稀疏树看似
 * 在每个深度都有相同活动量。
 */
export function sparkline(values: readonly number[]): string {
  if (!values.length) {
    return ''
  }

  const max = Math.max(...values)

  if (max <= 0) {
    return ' '.repeat(values.length)
  }

  return values
    .map(v => {
      if (v <= 0) {
        return ' '
      }

      const idx = Math.min(SPARK_RAMP.length - 1, Math.max(0, Math.ceil((v / max) * (SPARK_RAMP.length - 1))))

      return SPARK_RAMP[idx]
    })
    .join('')
}

/**
 * 将汇总数据格式化为紧凑的单行摘要：`d2 · 7 agents · 124 tools · 2m 14s`。
 */
export function formatSummary(totals: SubagentAggregate): string {
  const pieces = [`d${Math.max(0, totals.maxDepthFromHere)}`]
  pieces.push(`${totals.descendantCount} agent${totals.descendantCount === 1 ? '' : 's'}`)

  if (totals.totalTools > 0) {
    pieces.push(`${totals.totalTools} tool${totals.totalTools === 1 ? '' : 's'}`)
  }

  if (totals.totalDuration > 0) {
    pieces.push(fmtDuration(totals.totalDuration))
  }

  const tokens = totals.inputTokens + totals.outputTokens

  if (tokens > 0) {
    pieces.push(`${fmtTokens(tokens)} tok`)
  }

  if (totals.costUsd > 0) {
    pieces.push(fmtCost(totals.costUsd))
  }

  if (totals.activeCount > 0) {
    pieces.push(`⚡${totals.activeCount}`)
  }

  return pieces.join(' · ')
}

/** 紧凑美元金额，如 `$0.02`、`$1.34`、`$12.4`，美元符号后不超过五个字符。 */
export function fmtCost(usd: number): string {
  if (!Number.isFinite(usd) || usd <= 0) {
    return ''
  }

  if (usd < 0.01) {
    return '<$0.01'
  }

  if (usd < 10) {
    return `$${usd.toFixed(2)}`
  }

  return `$${usd.toFixed(1)}`
}

/** 紧凑 token 数量，如 `12k`、`1.2k`、`542`。 */
export function fmtTokens(n: number): string {
  if (!Number.isFinite(n) || n <= 0) {
    return '0'
  }

  if (n < 1000) {
    return String(Math.round(n))
  }

  if (n < 10_000) {
    return `${(n / 1000).toFixed(1)}k`
  }

  return `${Math.round(n / 1000)}k`
}

/**
 * 将秒数格式化为 `Ns`、`Nm` 或 `Nm Ss`。与智能体浮层共享，使时间线、列表和
 * 摘要采用同一种表达。
 */
export function fmtDuration(seconds: number): string {
  if (seconds < 60) {
    return `${Math.max(0, Math.round(seconds))}s`
  }

  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds - m * 60)

  return s === 0 ? `${m}m` : `${m}m ${s}s`
}

/**
 * 子智能体没有 `parentId`，或其父节点不在同一快照中（运行中根节点被裁剪而成
 * 为孤儿）时视为顶层节点。此规则与 `buildSubagentTree` 一致，使实时视图、
 * 磁盘标签和差异面板中的调用点保持统一。
 */
export function topLevelSubagents(items: readonly SubagentProgress[]): SubagentProgress[] {
  const ids = new Set(items.map(s => s.id))

  return items.filter(s => !s.parentId || !ids.has(s.parentId))
}

/**
 * 将节点热度归一化为调色板索引 0..N-1，其中 N 为色阶数。热度越高，颜色越
 * “热”。以整棵树的峰值热度为基准，使整体较慢的树仍能在最繁忙分支间呈现渐变。
 */
export function hotnessBucket(hotness: number, peakHotness: number, buckets: number): number {
  if (!Number.isFinite(hotness) || hotness <= 0 || peakHotness <= 0 || buckets <= 1) {
    return 0
  }

  const ratio = Math.min(1, hotness / peakHotness)

  return Math.min(buckets - 1, Math.max(0, Math.round(ratio * (buckets - 1))))
}

export function peakHotness(tree: readonly SubagentNode[]): number {
  let peak = 0

  const walk = (nodes: readonly SubagentNode[]) => {
    for (const node of nodes) {
      peak = Math.max(peak, node.aggregate.hotness)
      walk(node.children)
    }
  }

  walk(tree)

  return peak
}
