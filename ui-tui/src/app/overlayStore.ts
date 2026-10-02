// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { atom, computed } from 'nanostores'

import type { OverlayState } from './interfaces.js'

const buildOverlayState = (): OverlayState => ({
  agents: false,
  agentsInitialHistoryIndex: 0,
  clarify: null,
  confirm: null,
  modelPicker: false,
  pager: null,
  picker: false
})

export const $overlayState = atom<OverlayState>(buildOverlayState())

export const $isBlocked = computed($overlayState, ({ agents, clarify, confirm, modelPicker, pager, picker }) =>
  Boolean(agents || clarify || confirm || modelPicker || pager || picker)
)

export const getOverlayState = () => $overlayState.get()

export const patchOverlayState = (next: Partial<OverlayState> | ((state: OverlayState) => OverlayState)) =>
  $overlayState.set(typeof next === 'function' ? next($overlayState.get()) : { ...$overlayState.get(), ...next })

/** 完整重置，供会话或轮次拆除以及测试使用。 */
export const resetOverlayState = () => $overlayState.set(buildOverlayState())

/**
 * 软重置：移除流程范围的浮层（clarify、confirm、pager），但保留用户主动切换的
 * 智能体仪表板、模型选择器和会话选择器。它们由用户明确打开，不应在轮次结束时
 * 消失。每次轮次完成或中断时由 turnController.idle() 调用；旧版“全部重置”行为
 * 会在委派完成瞬间静默关闭 /agents。
 */
export const resetFlowOverlays = () =>
  $overlayState.set({
    ...buildOverlayState(),
    agents: $overlayState.get().agents,
    agentsInitialHistoryIndex: $overlayState.get().agentsInitialHistoryIndex,
    modelPicker: $overlayState.get().modelPicker,
    picker: $overlayState.get().picker
  })
