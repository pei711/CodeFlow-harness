// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { activeColorTier } from '@hermes/ink'

export interface ThemeColors {
  primary: string
  accent: string
  border: string
  text: string
  muted: string
  surface: string
  surfaceRaised: string
  heading: string
  info: string
  path: string
  thinking: string
  completionBg: string
  completionCurrentBg: string
  completionMetaBg: string
  completionMetaCurrentBg: string

  label: string
  ok: string
  error: string
  warn: string

  prompt: string
  sessionLabel: string
  sessionBorder: string

  statusBg: string
  statusFg: string
  statusGood: string
  statusWarn: string
  statusBad: string
  statusCritical: string
  selectionBg: string

  diffAdded: string
  diffRemoved: string
  diffAddedWord: string
  diffRemovedWord: string

  shellDollar: string
}

export interface ThemeBrand {
  name: string
  icon: string
  prompt: string
  welcome: string
  goodbye: string
  tool: string
  helpHeader: string
}

export interface Theme {
  color: ThemeColors
  brand: ThemeBrand
  bannerLogo: string
  bannerHero: string
  // 品牌黄色色阶（从亮到暗），按当前颜色层级解析，用于渐变横幅图案。
  yellow: readonly string[]
}

export type ColorScheme = 'dark' | 'light'

// ── 色彩计算 ─────────────────────────────────────────────────────────
//
// 这里只保留真彩调色板自身所需的辅助方法。运行时不做 RGB 到 ANSI 转换：下方低层级调色板
// 是预先推导的字面值（见 scripts/gen-color-palettes.mjs），因此二级终端会获得精选
// `ansi256(N)` 值，而不是 Chalk 有损的十六进制降采样；后者曾把深绿色边框压到橄榄色色块。

function parseHex(h: string): [number, number, number] | null {
  const m = /^#?([0-9a-f]{6})$/i.exec(h)

  if (!m) {
    return null
  }

  const n = parseInt(m[1]!, 16)

  return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff]
}

function mix(a: string, b: string, t: number) {
  const pa = parseHex(a)
  const pb = parseHex(b)

  if (!pa || !pb) {
    return a
  }

  const lerp = (i: 0 | 1 | 2) => Math.round(pa[i] + (pb[i] - pa[i]) * t)

  return '#' + ((1 << 24) | (lerp(0) << 16) | (lerp(1) << 8) | lerp(2)).toString(16).slice(1)
}

// ── 品牌色 ───────────────────────────────────────────────────────────

const BRAND: ThemeBrand = {
  name: 'CodeFlow',
  icon: '◆',
  prompt: '❯',
  welcome: 'Type your message or /help for commands.',
  goodbye: 'Goodbye!',
  tool: '·',
  helpHeader: 'Commands'
}

const cleanPromptSymbol = (s: string | undefined, fallback: string) => {
  const cleaned = String(s ?? '')
    .replace(/\s+/g, ' ')
    .trim()

  return cleaned || fallback
}

// ── 品牌黄色色阶（渐变标志/立体阴影）────────────────────────────────
//
// 品牌资产（codeflow-tui-design-system 的“品牌色阶”），按从亮到暗排序，用于渐变横幅图案。
// .50/.300/.500/.700/.900 是文档定义的标题色带
// （docs/tui-color-problem/title-gradient-table.md），其他节点为插值。横幅只读取前几个主色带，
// 并回退到最后一个，因此色阶长度不影响正确性。
//
// 真彩额外包含 .600 节点，使主视觉渐变更平滑，共 9 项：
// [.50,.100,.300,.500,.600,.700,.900,.950,.990]。256/16 色层级保留文档定义的 8 项集合，
// 不含 .600。真彩和 256 色的深色/浅色色阶彼此独立；浅色围绕 #B87900 重新推导，不是暗色色阶
// 降亮度。16 色统一使用 `yellow`。

const YELLOW_RAMP_TC_DARK: readonly string[] = [
  '#fff7c2', // 50
  '#fff0a4', // 100
  '#FFE573', // 300
  '#fbe23f', // 500
  '#e1c405', // 600
  '#c8a900', // 700
  '#8a6d00', // 900
  '#594600', // 950
  '#2d2300' // 990
]

const YELLOW_RAMP_TC_LIGHT: readonly string[] = [
  '#F6DA8B',
  '#EBC76C',
  '#D9A83A',
  '#B87900', // 500
  '#935F00', // 600
  '#935F00', // 700
  '#684300',
  '#432B00',
  '#221600'
]

const YELLOW_RAMP_256_DARK: readonly string[] = [
  'ansi256(229)',
  'ansi256(229)',
  'ansi256(228)',
  'ansi256(220)', // 500
  'ansi256(178)',
  'ansi256(94)',
  'ansi256(58)',
  'ansi256(234)'
]

const YELLOW_RAMP_256_LIGHT: readonly string[] = [
  'ansi256(222)',
  'ansi256(222)',
  'ansi256(179)',
  'ansi256(136)',
  'ansi256(94)',
  'ansi256(58)',
  'ansi256(58)',
  'ansi256(234)'
]

const YELLOW_RAMP_16: readonly string[] = [
  'ansi:yellow',
  'ansi:yellow',
  'ansi:yellow',
  'ansi:yellow',
  'ansi:yellow',
  'ansi:yellow',
  'ansi:yellow',
  'ansi:yellow'
]

function yellowRamp(tier: 0 | 1 | 2 | 3, scheme: ColorScheme): readonly string[] {
  if (tier === 2) {
    return scheme === 'light' ? YELLOW_RAMP_256_LIGHT : YELLOW_RAMP_256_DARK
  }
  if (tier === 1) {
    return YELLOW_RAMP_16
  }
  return scheme === 'light' ? YELLOW_RAMP_TC_LIGHT : YELLOW_RAMP_TC_DARK
}

// ── 第 3 层：真彩（权威来源）────────────────────────────────────────

export const DARK_THEME: Theme = {
  color: {
    primary: '#c8c8c8',
    accent: '#7aa2f7',
    border: '#323237',
    text: '#e1e1e1',
    muted: '#6c6c6c',
    surface: '#141414',
    surfaceRaised: '#242424',
    heading: '#73daca',
    info: '#7aa2f7',
    path: '#ff9e64',
    thinking: '#bb9af7',
    completionBg: '#242424',
    completionCurrentBg: '#363636',
    completionMetaBg: '#242424',
    completionMetaCurrentBg: '#363636',

    label: '#c8c8c8',
    ok: '#9ece6a',
    error: '#f7768e',
    warn: '#e0af68',

    prompt: '#c8c8c8',
    sessionLabel: '#6c6c6c',
    sessionBorder: '#505058',

    statusBg: '#141414',
    statusFg: '#6c6c6c',
    statusGood: '#9ece6a',
    statusWarn: '#e0af68',
    statusBad: '#f7768e',
    statusCritical: '#f7768e',
    selectionBg: '#363636',

    diffAdded: '#063806',
    diffRemoved: '#420e14',
    diffAddedWord: '#9ece6a',
    diffRemovedWord: '#f7768e',
    shellDollar: '#e0af68'
  },

  brand: BRAND,

  bannerLogo: '',
  bannerHero: '',
  yellow: YELLOW_RAMP_TC_DARK
}

// 浅色终端调色板：使用更深、对比更高且在白底上清晰的颜色。结构与 DARK_THEME 相同，
// 使 `fromSkin` 仍能干净叠加（#11300）。
export const LIGHT_THEME: Theme = {
  color: {
    primary: '#444444',
    accent: '#2f64d2',
    border: '#c8c8cd',
    text: '#262626',
    muted: '#767676',
    surface: '#eeeeee',
    surfaceRaised: '#dedede',
    heading: '#0c947c',
    info: '#2f64d2',
    path: '#c3691e',
    thinking: '#7d4bc6',
    completionBg: '#dedede',
    completionCurrentBg: '#c6c6c6',
    completionMetaBg: '#dedede',
    completionMetaCurrentBg: '#c6c6c6',

    label: '#444444',
    ok: '#378e23',
    error: '#cd3048',
    warn: '#a27612',

    prompt: '#444444',
    sessionLabel: '#767676',
    sessionBorder: '#a5a5af',

    statusBg: '#eeeeee',
    statusFg: '#767676',
    statusGood: '#378e23',
    statusWarn: '#a27612',
    statusBad: '#cd3048',
    statusCritical: '#cd3048',
    selectionBg: '#c6c6c6',

    diffAdded: '#daf2dc',
    diffRemoved: '#f5dade',
    diffAddedWord: '#378e23',
    diffRemovedWord: '#cd3048',
    shellDollar: '#a27612'
  },

  brand: BRAND,

  bannerLogo: '',
  bannerHero: '',
  yellow: YELLOW_RAMP_TC_LIGHT
}

// ── 第 2 层：256 色（按设计令牌 docs/tui-color-problem/tokens.md）────────

const DARK_256_COLORS: ThemeColors = {
  primary: 'ansi256(251)',
  accent: 'ansi256(111)',
  border: 'ansi256(236)',
  text: 'ansi256(254)',
  muted: 'ansi256(242)',
  surface: 'ansi256(233)',
  surfaceRaised: 'ansi256(235)',
  heading: 'ansi256(79)',
  info: 'ansi256(111)',
  path: 'ansi256(209)',
  thinking: 'ansi256(141)',
  completionBg: 'ansi256(235)',
  completionCurrentBg: 'ansi256(237)',
  completionMetaBg: 'ansi256(235)',
  completionMetaCurrentBg: 'ansi256(237)',
  label: 'ansi256(251)',
  ok: 'ansi256(149)',
  error: 'ansi256(210)',
  warn: 'ansi256(179)',
  prompt: 'ansi256(251)',
  sessionLabel: 'ansi256(242)',
  sessionBorder: 'ansi256(240)',
  statusBg: 'ansi256(233)',
  statusFg: 'ansi256(242)',
  statusGood: 'ansi256(149)',
  statusWarn: 'ansi256(179)',
  statusBad: 'ansi256(210)',
  statusCritical: 'ansi256(210)',
  selectionBg: 'ansi256(237)',
  diffAdded: 'ansi256(22)',
  diffRemoved: 'ansi256(52)',
  diffAddedWord: 'ansi256(149)',
  diffRemovedWord: 'ansi256(210)',
  shellDollar: 'ansi256(179)'
}

const LIGHT_256_COLORS: ThemeColors = {
  primary: 'ansi256(238)',
  accent: 'ansi256(26)',
  border: 'ansi256(251)',
  text: 'ansi256(234)',
  muted: 'ansi256(243)',
  surface: 'ansi256(255)',
  surfaceRaised: 'ansi256(253)',
  heading: 'ansi256(29)',
  info: 'ansi256(26)',
  path: 'ansi256(166)',
  thinking: 'ansi256(98)',
  completionBg: 'ansi256(253)',
  completionCurrentBg: 'ansi256(251)',
  completionMetaBg: 'ansi256(253)',
  completionMetaCurrentBg: 'ansi256(251)',
  label: 'ansi256(238)',
  ok: 'ansi256(28)',
  error: 'ansi256(161)',
  warn: 'ansi256(136)',
  prompt: 'ansi256(238)',
  sessionLabel: 'ansi256(243)',
  sessionBorder: 'ansi256(248)',
  statusBg: 'ansi256(255)',
  statusFg: 'ansi256(243)',
  statusGood: 'ansi256(28)',
  statusWarn: 'ansi256(136)',
  statusBad: 'ansi256(161)',
  statusCritical: 'ansi256(161)',
  selectionBg: 'ansi256(251)',
  diffAdded: 'ansi256(194)',
  diffRemoved: 'ansi256(224)',
  diffAddedWord: 'ansi256(28)',
  diffRemovedWord: 'ansi256(161)',
  shellDollar: 'ansi256(136)'
}

// ── 第 1 层：16 色（按设计令牌 docs/tui-color-problem/tokens.md）─────────
//
// 与令牌规范相比有两点限制，单个颜色字符串无法编码：
//   - `reverse` 高亮（completionCurrentBg/completionMetaCurrentBg/selectionBg）回退到
//     brightBlack，这是规范指定的替代方案。
//   - 丢弃 statusCritical 的 `+ bold`，因为粗体是文本样式而非颜色，保留规范基础色 `red`。

const DARK_16_COLORS: ThemeColors = {
  primary: 'ansi:white',
  accent: 'ansi:blueBright',
  border: 'ansi:blackBright',
  text: 'ansi:white',
  muted: 'ansi:blackBright',
  surface: 'ansi:black',
  surfaceRaised: 'ansi:blackBright',
  heading: 'ansi:cyanBright',
  info: 'ansi:blueBright',
  path: 'ansi:yellow',
  thinking: 'ansi:magentaBright',
  completionBg: 'ansi:black',
  completionCurrentBg: 'ansi:blackBright',
  completionMetaBg: 'ansi:black',
  completionMetaCurrentBg: 'ansi:blackBright',
  label: 'ansi:blackBright',
  ok: 'ansi:greenBright',
  error: 'ansi:redBright',
  warn: 'ansi:yellow',
  prompt: 'ansi:white',
  sessionLabel: 'ansi:blackBright',
  sessionBorder: 'ansi:blackBright',
  statusBg: 'ansi:black',
  statusFg: 'ansi:blackBright',
  statusGood: 'ansi:greenBright',
  statusWarn: 'ansi:yellow',
  statusBad: 'ansi:redBright',
  statusCritical: 'ansi:red',
  selectionBg: 'ansi:blackBright',
  diffAdded: 'ansi:blackBright',
  diffRemoved: 'ansi:blackBright',
  diffAddedWord: 'ansi:green',
  diffRemovedWord: 'ansi:red',
  shellDollar: 'ansi:yellow'
}

const LIGHT_16_COLORS: ThemeColors = {
  primary: 'ansi:black',
  accent: 'ansi:blue',
  border: 'ansi:blackBright',
  text: 'ansi:black',
  muted: 'ansi:blackBright',
  surface: 'ansi:white',
  surfaceRaised: 'ansi:whiteBright',
  heading: 'ansi:cyan',
  info: 'ansi:blue',
  path: 'ansi:yellow',
  thinking: 'ansi:magenta',
  completionBg: 'ansi:white',
  completionCurrentBg: 'ansi:blackBright',
  completionMetaBg: 'ansi:white',
  completionMetaCurrentBg: 'ansi:blackBright',
  label: 'ansi:blackBright',
  ok: 'ansi:green',
  error: 'ansi:red',
  warn: 'ansi:yellow',
  prompt: 'ansi:black',
  sessionLabel: 'ansi:blackBright',
  sessionBorder: 'ansi:blackBright',
  statusBg: 'ansi:white',
  statusFg: 'ansi:blackBright',
  statusGood: 'ansi:green',
  statusWarn: 'ansi:yellow',
  statusBad: 'ansi:red',
  statusCritical: 'ansi:red',
  selectionBg: 'ansi:blackBright',
  diffAdded: 'ansi:blackBright',
  diffRemoved: 'ansi:blackBright',
  diffAddedWord: 'ansi:green',
  diffRemovedWord: 'ansi:red',
  shellDollar: 'ansi:yellow'
}

const DARK_256: Theme = { ...DARK_THEME, color: DARK_256_COLORS, yellow: YELLOW_RAMP_256_DARK }
const DARK_16: Theme = { ...DARK_THEME, color: DARK_16_COLORS, yellow: YELLOW_RAMP_16 }
const LIGHT_256: Theme = { ...LIGHT_THEME, color: LIGHT_256_COLORS, yellow: YELLOW_RAMP_256_LIGHT }
const LIGHT_16: Theme = { ...LIGHT_THEME, color: LIGHT_16_COLORS, yellow: YELLOW_RAMP_16 }

/**
 * 根据配色模式和颜色级别选择调色板。第 3 级（真彩色）与第 0 级（无颜色，
 * Chalk 最终会移除颜色码）都使用十六进制调色板，因此原样返回真彩色 `Theme`
 * 引用，便于做同一性检查。
 */
export function resolveTheme(scheme: ColorScheme, tier: 0 | 1 | 2 | 3): Theme {
  if (scheme === 'light') {
    if (tier === 2) {
      return LIGHT_256
    }
    if (tier === 1) {
      return LIGHT_16
    }
    return LIGHT_THEME
  }

  if (tier === 2) {
    return DARK_256
  }
  if (tier === 1) {
    return DARK_16
  }
  return DARK_THEME
}

// ── 明暗检测 ─────────────────────────────────────────────────────────

const TRUE_RE = /^(?:1|true|yes|on)$/
const FALSE_RE = /^(?:0|false|no|off)$/

// 默认配置为浅色但可能不暴露 COLORFGBG 的终端所用 TERM_PROGRAM 回退白名单。默认留空：
// 单凭 TERM_PROGRAM 无法区分明暗配置，Terminal.app 两者都有且都不输出 COLORFGBG；暗色配置
// 又很常见，因此无法检测时保持暗色，除非 CODEFLOW_TUI_THEME、CODEFLOW_TUI_LIGHT、
// CODEFLOW_TUI_BACKGROUND 或 COLORFGBG 明确表示浅色。仍允许注入，供测试验证优先级规则。
const LIGHT_DEFAULT_TERM_PROGRAMS = new Set<string>([])

// 尽力进行 RGB 到亮度检查。目前只接受 3 位或 6 位十六进制值，可带或不带前导 `#`。
// 环境变量名 `CODEFLOW_TUI_BACKGROUND` 刻意保持通用，未来 OSC11 查询辅助方法也可在其中缓存结果；
// 其他格式（rgb()/hsl()/具名颜色）需先在此显式解析。
const LUMA_LIGHT_THRESHOLD = 0.6

// 严格白名单：parseInt(..., 16) 会在首个非十六进制字符处静默截断，例如 `fffgff` 会解析为
// `fff` 并误判为白色，因此预先拒绝不符合规范 3 位或 6 位形态的值。
const HEX_3_RE = /^[0-9a-f]{3}$/
const HEX_6_RE = /^[0-9a-f]{6}$/

function backgroundLuminance(raw: string): null | number {
  const v = raw.trim().toLowerCase()

  if (!v) {
    return null
  }

  const hex = v.startsWith('#') ? v.slice(1) : v

  const rgb = HEX_6_RE.test(hex)
    ? [parseInt(hex.slice(0, 2), 16), parseInt(hex.slice(2, 4), 16), parseInt(hex.slice(4, 6), 16)]
    : HEX_3_RE.test(hex)
      ? [parseInt(hex[0]! + hex[0]!, 16), parseInt(hex[1]! + hex[1]!, 16), parseInt(hex[2]! + hex[2]!, 16)]
      : null

  if (!rgb) {
    return null
  }

  // 使用 Rec. 709 亮度，足以判断背景是否明亮。
  return (0.2126 * rgb[0]! + 0.7152 * rgb[1]! + 0.0722 * rgb[2]!) / 255
}

// 使用有序、可解释的信号选择浅色或暗色（#11300）：
//
//   1. `CODEFLOW_TUI_LIGHT` 布尔值：`1`/`true`/`yes`/`on` 表示浅色，
//      `0`/`false`/`no`/`off` 表示暗色；任一显式值都优先于后续信号。
//   2. `CODEFLOW_TUI_THEME` 具名覆盖：`light` / `dark` 优先于下方所有信号。
//   3. `CODEFLOW_TUI_BACKGROUND` 的 3 位或 6 位十六进制提示：亮度不低于阈值时为浅色。
//   4. `COLORFGBG` 最后字段：XFCE、rxvt、Terminal.app 的浅色配置输出槽位 7 或 15；
//      0 到 15 的其他值视为权威暗色，避免下方 TERM_PROGRAM 白名单覆盖显式暗色配置。
//   5. `TERM_PROGRAM` 默认浅色白名单，默认留空，见 LIGHT_DEFAULT_TERM_PROGRAMS。
//
// 无法判断时保持暗色，因为 CodeFlow 默认调色板为暗色。
export function detectLightMode(
  env: NodeJS.ProcessEnv = process.env,
  // 允许注入，使测试即使在生产白名单为空时也能验证 COLORFGBG 优先于 TERM_PROGRAM 的规则。
  lightDefaultTermPrograms: ReadonlySet<string> = LIGHT_DEFAULT_TERM_PROGRAMS
): boolean {
  const lightFlag = (env.CODEFLOW_TUI_LIGHT ?? '').trim().toLowerCase()

  if (TRUE_RE.test(lightFlag)) {
    return true
  }

  if (FALSE_RE.test(lightFlag)) {
    return false
  }

  const themeFlag = (env.CODEFLOW_TUI_THEME ?? '').trim().toLowerCase()

  if (themeFlag === 'light') {
    return true
  }

  if (themeFlag === 'dark') {
    return false
  }

  const bgHint = backgroundLuminance(env.CODEFLOW_TUI_BACKGROUND ?? '')

  if (bgHint !== null) {
    return bgHint >= LUMA_LIGHT_THRESHOLD
  }

  const colorfgbg = (env.COLORFGBG ?? '').trim()

  if (colorfgbg) {
    // 强制转换前先按十进制整数校验。`Number('')` 为 0，因此格式错误的 `COLORFGBG='15;'`
    // 原本会像权威暗色槽位一样错误阻止 TERM_PROGRAM 白名单。非纯数字值均继续回退。
    const lastField = colorfgbg.split(';').at(-1) ?? ''

    if (/^\d+$/.test(lastField)) {
      const bg = Number(lastField)

      if (bg === 7 || bg === 15) {
        return true
      }

      // 槽位 0 到 6、8 到 14 是 ANSI 0 到 15 范围中的暗色半区。设置 COLORFGBG 后将其视为
      // 权威信号；此处非浅色值不应被 TERM_PROGRAM 白名单覆盖。
      if (bg >= 0 && bg < 16) {
        return false
      }
    }
  }

  const termProgram = (env.TERM_PROGRAM ?? '').trim()

  return lightDefaultTermPrograms.has(termProgram)
}

const DEFAULT_LIGHT_MODE = detectLightMode()
const DEFAULT_SCHEME: ColorScheme = DEFAULT_LIGHT_MODE ? 'light' : 'dark'

export const DEFAULT_THEME: Theme = resolveTheme(DEFAULT_SCHEME, activeColorTier())

// 运行时由 OSC 11 背景色探针检测的配色方案，见 applyDetectedBackground。终端回答查询前或始终
// 不回答时为 null。设置后，它会在之后构建的所有主题中优先于环境探测的 DEFAULT_SCHEME，
// 因此迟到的回复也会重新设置整个应用主题。
let detectedScheme: ColorScheme | null = null

/** 实际生效的明暗配色：有 OSC 11 探测结果时使用该结果，否则采用环境探测默认值。
 *  主题构建器读取此值而不是 DEFAULT_SCHEME，使迟到的探测响应能在重建主题时
 *  生效。 */
export function currentScheme(): ColorScheme {
  return detectedScheme ?? DEFAULT_SCHEME
}

/** 当前配色模式和颜色级别对应的精选调色板，不应用皮肤。网关皮肤尚未到达而探针
 *  已切换配色模式时，用它重建主题。 */
export function resolveCurrentDefaultTheme(): Theme {
  return resolveTheme(currentScheme(), activeColorTier())
}

/** OSC 12 硬件光标颜色使用的真彩色十六进制值。OSC 12 接收 RGB 值，因此无论
 *  文本颜色级别如何都使用十六进制主色；256 色或 16 色终端仍以真彩色渲染光标。
 *  若皮肤提供第 3 级十六进制主色则采用它，否则使用按配色精选的标题颜色。 */
export function cursorColorHex(theme: Theme): string {
  const p = theme.color.primary

  return p.startsWith('#') ? p : currentScheme() === 'light' ? LIGHT_THEME.color.primary : DARK_THEME.color.primary
}

// 将 OSC 11 背景回复载荷解析为 #rrggbb 十六进制字符串。xterm 类终端返回
// `rgb:RRRR/GGGG/BBBB`，每通道 1 到 4 位十六进制并按该通道最大值缩放；少数返回
// `#RRGGBB`。其他格式返回 null，使调用方保留基于环境的方案。
function oscColorToHex(data: string): null | string {
  const s = data.trim().toLowerCase()
  const m = /^rgba?:([0-9a-f]{1,4})\/([0-9a-f]{1,4})\/([0-9a-f]{1,4})/.exec(s)

  if (m) {
    const scale = (h: string) => Math.round((parseInt(h, 16) / (16 ** h.length - 1)) * 255)

    return '#' + [m[1]!, m[2]!, m[3]!].map(h => scale(h).toString(16).padStart(2, '0')).join('')
  }

  const hex = s.startsWith('#') ? s.slice(1) : s

  if (HEX_6_RE.test(hex)) {
    return '#' + hex
  }

  if (HEX_3_RE.test(hex)) {
    return '#' + [...hex].map(c => c + c).join('')
  }

  return null
}

/**
 * 将 OSC 11 背景色响应纳入明暗模式探测。
 *
 * 将解析后的颜色缓存到 CODEFLOW_TUI_BACKGROUND，再次运行 detectLightMode()，使
 * 现有优先级规则保持不变；显式 CODEFLOW_TUI_THEME/CODEFLOW_TUI_LIGHT 仍优先于实测
 * 背景色。返回解析后的配色模式及其是否不同于当前模式，供调用方判断是否重设
 * 主题；响应无法解析为颜色时返回 null。
 */
export function applyDetectedBackground(oscData: string): { changed: boolean; scheme: ColorScheme } | null {
  const hex = oscColorToHex(oscData)

  if (!hex) {
    return null
  }

  process.env.CODEFLOW_TUI_BACKGROUND = hex
  const scheme: ColorScheme = detectLightMode() ? 'light' : 'dark'
  const changed = scheme !== currentScheme()
  detectedScheme = scheme

  return { changed, scheme }
}

// ── 皮肤转换为主题 ───────────────────────────────────────────────────

function skinColors(colors: Record<string, string>): ThemeColors {
  const base = (currentScheme() === 'light' ? LIGHT_THEME : DARK_THEME).color
  const c = (k: string) => colors[k]
  const hasSkinColors = Object.keys(colors).length > 0

  const accent = c('ui_accent') ?? c('banner_accent') ?? base.accent
  const bannerAccent = c('banner_accent') ?? c('banner_title') ?? base.accent
  const muted = c('banner_dim') ?? base.muted
  const completionBg = c('completion_menu_bg') ?? base.completionBg

  const completionCurrentBg =
    c('completion_menu_current_bg') ??
    (hasSkinColors ? mix(completionBg, bannerAccent, 0.25) : base.completionCurrentBg)

  // 元数据列先继承匹配的皮肤键，再回退到调色板自身独立的元数据值。因此空皮肤能精确复现默认
  // 主题，而只设置主补全背景的皮肤仍会把它带入元数据列。
  const completionMetaBg = c('completion_menu_meta_bg') ?? c('completion_menu_bg') ?? base.completionMetaBg
  const completionMetaCurrentBg =
    c('completion_menu_meta_current_bg') ??
    c('completion_menu_current_bg') ??
    (hasSkinColors ? completionCurrentBg : base.completionMetaCurrentBg)

  return {
    primary: c('ui_primary') ?? c('banner_title') ?? base.primary,
    accent,
    border: c('ui_border') ?? c('banner_border') ?? base.border,
    text: c('ui_text') ?? c('banner_text') ?? base.text,
    muted,
    surface: base.surface,
    surfaceRaised: base.surfaceRaised,
    heading: base.heading,
    info: base.info,
    path: base.path,
    thinking: base.thinking,
    completionBg,
    completionCurrentBg,
    completionMetaBg,
    completionMetaCurrentBg,

    label: c('ui_label') ?? base.label,
    ok: c('ui_ok') ?? base.ok,
    error: c('ui_error') ?? base.error,
    warn: c('ui_warn') ?? base.warn,

    prompt: c('prompt') ?? c('banner_text') ?? base.prompt,
    sessionLabel: c('session_label') ?? base.sessionLabel,
    sessionBorder: c('session_border') ?? base.sessionBorder,

    statusBg: base.statusBg,
    statusFg: base.statusFg,
    statusGood: c('ui_ok') ?? base.statusGood,
    statusWarn: c('ui_warn') ?? base.statusWarn,
    statusBad: base.statusBad,
    statusCritical: base.statusCritical,
    selectionBg:
      c('selection_bg') ?? c('completion_menu_current_bg') ?? (hasSkinColors ? completionCurrentBg : base.selectionBg),

    diffAdded: base.diffAdded,
    diffRemoved: base.diffRemoved,
    diffAddedWord: base.diffAddedWord,
    diffRemovedWord: base.diffRemovedWord,
    shellDollar: c('shell_dollar') ?? base.shellDollar
  }
}

export function fromSkin(
  colors: Record<string, string>,
  branding: Record<string, string>,
  bannerLogo = '',
  bannerHero = '',
  toolPrefix = '',
  helpHeader = ''
): Theme {
  const d = DEFAULT_THEME

  const brand: ThemeBrand = {
    name: branding.agent_name ?? d.brand.name,
    icon: d.brand.icon,
    prompt: cleanPromptSymbol(branding.prompt_symbol, d.brand.prompt),
    welcome: branding.welcome ?? d.brand.welcome,
    goodbye: branding.goodbye ?? d.brand.goodbye,
    tool: toolPrefix || d.brand.tool,
    helpHeader: branding.help_header ?? (helpHeader || d.brand.helpHeader)
  }

  // 皮肤以真彩十六进制编写。低颜色层级无法表示任意十六进制，因此按产品决策回退到该层级的
  // 精选内置调色板，只继承品牌色和横幅图案。十六进制路径覆盖第 3 层，也无害地覆盖第 0 层，
  // 因为颜色代码最终会被移除。
  const tier = activeColorTier()
  const color = tier === 1 || tier === 2 ? resolveTheme(currentScheme(), tier).color : skinColors(colors)

  return { color, brand, bannerLogo, bannerHero, yellow: yellowRamp(tier, currentScheme()) }
}
