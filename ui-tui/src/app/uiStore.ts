import { atom, computed } from 'nanostores'

import type { GatewaySkin } from '../gatewayTypes.js'

import { MOUSE_TRACKING } from '../config/env.js'
import { ZERO } from '../domain/usage.js'
import { applyDetectedBackground, DEFAULT_THEME, fromSkin, resolveCurrentDefaultTheme } from '../theme.js'
import { DEFAULT_INDICATOR_STYLE, type UiState } from './interfaces.js'

const buildUiState = (): UiState => ({
  bgTasks: new Set(),
  busy: false,
  busyInputMode: 'queue',
  compact: false,
  escapeArmed: false,
  detailsMode: 'collapsed',
  detailsModeCommandOverride: false,
  indicatorStyle: DEFAULT_INDICATOR_STYLE,
  info: null,
  inlineDiffs: true,
  mouseTracking: MOUSE_TRACKING,
  sections: {},
  sessionMutating: false,
  sessionSwitching: false,
  showCost: false,
  // 默认开启，使推理模型（deepseek-v4-pro、qwen、o-series）开箱即显示思考流，
  // 避免用户面对无响应界面等待 1 到 4 分钟；可通过 /thinking 切换。
  showReasoning: true,
  sid: null,
  status: 'starting codeflow…',
  statusBar: 'top',
  streaming: true,
  theme: DEFAULT_THEME,
  usage: ZERO
})

export const $uiState = atom<UiState>(buildUiState())

export const $uiTheme = computed($uiState, state => state.theme)
export const $uiSessionId = computed($uiState, state => state.sid)

export const getUiState = () => $uiState.get()

export const patchUiState = (next: Partial<UiState> | ((state: UiState) => UiState)) =>
  $uiState.set(typeof next === 'function' ? next($uiState.get()) : { ...$uiState.get(), ...next })

export const resetUiState = () => $uiState.set(buildUiState())

// 保留网关最后推送的皮肤，使延迟返回的终端背景探测能按纠正后的明暗模式重建主题。
let lastSkin: GatewaySkin | null = null

const buildSkinTheme = (s: GatewaySkin) =>
  fromSkin(
    s.colors ?? {},
    s.branding ?? {},
    s.banner_logo ?? '',
    s.banner_hero ?? '',
    s.tool_prefix ?? '',
    s.help_header ?? ''
  )

export const applySkinTheme = (s: GatewaySkin) => {
  lastSkin = s
  patchUiState({ theme: buildSkinTheme(s) })
}

/**
 * 将 OSC 11 背景色响应纳入主题。若它改变探测到的明暗模式，则用最近到达的皮肤
 * 重建活动主题；尚无皮肤时使用按模式精选的调色板，使整个界面同步换色。模式
 * 未变化或响应无法解析时不执行操作。
 */
export const applyTerminalBackground = (oscData: string) => {
  const res = applyDetectedBackground(oscData)

  if (res?.changed) {
    patchUiState({ theme: lastSkin ? buildSkinTheme(lastSkin) : resolveCurrentDefaultTheme() })
  }

  return res
}
