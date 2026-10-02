// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { atom } from 'nanostores'

import type { SubagentProgress } from '../types.js'

export interface SpawnSnapshot {
  finishedAt: number
  id: string
  label: string
  sessionId: null | string
  startedAt: number
  subagents: SubagentProgress[]
}

export interface SpawnDiffPair {
  baseline: SpawnSnapshot
  candidate: SpawnSnapshot
}

const HISTORY_LIMIT = 10

export const $spawnHistory = atom<SpawnSnapshot[]>([])
export const $spawnDiff = atom<null | SpawnDiffPair>(null)

export const getSpawnHistory = () => $spawnHistory.get()
export const getSpawnDiff = () => $spawnDiff.get()

export const clearSpawnHistory = () => $spawnHistory.set([])
export const clearDiffPair = () => $spawnDiff.set(null)
export const setDiffPair = (pair: SpawnDiffPair) => $spawnDiff.set(pair)

/**
 * 将已完成轮次的生成树写入历史。保留最近 10 个非空快照；没有子智能体的空轮次
 * 会被丢弃。
 *
 * 之所以保存在内存，是因为主要排查闭环发生在同一会话：刚执行扇出后行为异常，
 * 随即查看发生了什么。跨进程重启的磁盘持久化虽可自然扩展，但会为较少使用的
 * 路径增加 RPC 接口面。
 */
export const pushSnapshot = (
  subagents: readonly SubagentProgress[],
  meta: { sessionId?: null | string; startedAt?: null | number }
) => {
  if (!subagents.length) {
    return
  }

  const now = Date.now()
  const started = meta.startedAt ?? Math.min(...subagents.map(s => s.startedAt ?? now))

  const snap: SpawnSnapshot = {
    finishedAt: now,
    id: `snap-${now.toString(36)}`,
    label: summarizeLabel(subagents),
    sessionId: meta.sessionId ?? null,
    startedAt: Number.isFinite(started) ? started : now,
    subagents: subagents.map(item => ({ ...item }))
  }

  const next = [snap, ...$spawnHistory.get()].slice(0, HISTORY_LIMIT)
  $spawnHistory.set(next)
}

function summarizeLabel(subagents: readonly SubagentProgress[]): string {
  const top = subagents
    .filter(s => s.parentId == null || subagents.every(o => o.id !== s.parentId))
    .slice(0, 2)
    .map(s => s.goal || 'subagent')
    .join(' · ')

  return top || `${subagents.length} agent${subagents.length === 1 ? '' : 's'}`
}
