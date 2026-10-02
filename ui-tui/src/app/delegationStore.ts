// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { atom } from 'nanostores'

export interface DelegationState {
  maxConcurrentChildren: null | number
  maxSpawnDepth: null | number
  paused: boolean
  updatedAt: null | number
}

const buildState = (): DelegationState => ({
  maxConcurrentChildren: null,
  maxSpawnDepth: null,
  paused: false,
  updatedAt: null
})

export const $delegationState = atom<DelegationState>(buildState())

export const getDelegationState = () => $delegationState.get()

export const resetDelegationState = () => $delegationState.set(buildState())

// ── 浮层手风琴展开状态 ──────────────────────────────────────
//
// 将状态从 OverlaySection 的局部 useState 中提升，使折叠选择在以下操作后保留：
//   - 导航到其他子智能体（Detail 重新挂载）；
//   - 在列表与详情模式之间切换（列表模式会卸载 Detail）；
//   - 使用 ←/→ 浏览历史。
// 以分区标题为键；缺失项回退到分区的 `defaultOpen` 属性。

export const $overlaySectionsOpen = atom<Record<string, boolean>>({})

export const toggleOverlaySection = (title: string, defaultOpen: boolean) => {
  const state = $overlaySectionsOpen.get()
  const current = title in state ? state[title]! : defaultOpen

  $overlaySectionsOpen.set({ ...state, [title]: !current })
}

export const getOverlaySectionOpen = (title: string, defaultOpen: boolean): boolean => {
  const state = $overlaySectionsOpen.get()

  return title in state ? state[title]! : defaultOpen
}
