// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

const RICH_RE = /\[(?:bold\s+)?(?:dim\s+)?(#(?:[0-9a-fA-F]{3,8}))\]([\s\S]*?)(\[\/\])/g

export function parseRichMarkup(markup: string): Line[] {
  const lines: Line[] = []

  for (const raw of markup.split('\n')) {
    const trimmed = raw.trimEnd()

    if (!trimmed) {
      lines.push(['', ' '])

      continue
    }

    const matches = [...trimmed.matchAll(RICH_RE)]

    if (!matches.length) {
      lines.push(['', trimmed])

      continue
    }

    let cursor = 0

    for (const m of matches) {
      const before = trimmed.slice(cursor, m.index)

      if (before) {
        lines.push(['', before])
      }

      lines.push([m[1]!, m[2]!])
      cursor = m.index! + m[0].length
    }

    if (cursor < trimmed.length) {
      lines.push(['', trimmed.slice(cursor)])
    }
  }

  return lines
}

const CODEFLOW_WORD_ART = [
  '██████╗ ██╗ ██████╗  ██████╗ ',
  '██╔══██╗██║██╔════╝ ██╔═══██╗',
  '██████╔╝██║██║      ██║   ██║',
  '██╔═══╝ ██║██║      ██║   ██║',
  '██║     ██║╚██████╗ ╚██████╔╝',
  '╚═╝     ╚═╝ ╚═════╝  ╚═════╝ '
] as const

const AGENT_WORD_ART = [
  ' █████╗  ██████╗ ███████╗███╗   ██╗████████╗',
  '██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝',
  '███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ',
  '██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ',
  '██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ',
  '╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   '
] as const

const CODEFLOW_LOGO_ART = CODEFLOW_WORD_ART.map((row, index) => `${row}   ${AGENT_WORD_ART[index]}`)

const CODEFLOW_HERO_ART = [
  '         ╱╲         ',
  '        ╱  ╲        ',
  '       ╱ ◆  ╲       ',
  '      ╱      ╲      ',
  '     ╱  CODEFLOW  ╲     ',
  '    ╰──────────╯    '
] as const

export const CODEFLOW_LOGO_WIDTH = CODEFLOW_LOGO_ART.reduce((width, row) => Math.max(width, [...row].length), 0)
export const CODEFLOW_HERO_WIDTH = CODEFLOW_HERO_ART.reduce((width, row) => Math.max(width, [...row].length), 0)

// 标题中每个渐变色覆盖的字符画行数（垂直色带）。
const LOGO_ROWS_PER_BAND = 2
// 主图横向渐变所划分的垂直色带数量。
const HERO_BANDS = 5

const bandColor = (ramp: readonly string[], row: number) =>
  ramp[Math.floor(row / LOGO_ROWS_PER_BAND)] ?? ramp[ramp.length - 1]!

// 标题字标从上到下每 LOGO_ROWS_PER_BAND 行使用一种渐变色，取前
// ceil(rows / band) 个渐变项。
export const codeflowLogo = (ramp: readonly string[], customLogo?: string): Line[] =>
  customLogo ? parseRichMarkup(customLogo) : CODEFLOW_LOGO_ART.map((text, i) => [bandColor(ramp, i), text])

// “CODEFLOW”字标的宽度，也是短版所需的最小宽度。
export const CODEFLOW_WORD_WIDTH = CODEFLOW_WORD_ART.reduce((width, row) => Math.max(width, [...row].length), 0)

// 只渲染“CODEFLOW”字标，并采用从上到下的渐变。
export const codeflowLogoWord = (ramp: readonly string[]): Line[] =>
  CODEFLOW_WORD_ART.map((text, i) => [bandColor(ramp, i), text])

// CodeFlow 主图使用由 HERO_BANDS 个列色带构成的横向渐变。颜色从右向左加深，
// 使高亮色 ramp[0] 位于右侧，最深色 ramp[HERO_BANDS-1] 位于左侧。
// 每行返回一组用于行内渲染的 `[color, segment]` 对。
export const codeflowHero = (ramp: readonly string[], customHero?: string): Line[][] => {
  if (customHero) {
    return parseRichMarkup(customHero).map(line => [line])
  }

  return CODEFLOW_HERO_ART.map(row => {
    const chars = [...row]
    const bandWidth = Math.ceil(chars.length / HERO_BANDS)
    const segments: Line[] = []

    for (let band = 0; band < HERO_BANDS; band++) {
      const text = chars.slice(band * bandWidth, (band + 1) * bandWidth).join('')

      if (!text) {
        continue
      }

      const color = ramp[HERO_BANDS - 1 - band] ?? ramp[ramp.length - 1]!
      segments.push([color, text])
    }

    return segments
  })
}

export const artWidth = (lines: Line[]) => lines.reduce((m, [, t]) => Math.max(m, [...t].length), 0)

// 分段字符画（主图）的宽度，即各行分段总宽度的最大值。
export const rowsWidth = (rows: Line[][]) =>
  rows.reduce(
    (m, segs) =>
      Math.max(
        m,
        segs.reduce((a, [, t]) => a + [...t].length, 0)
      ),
    0
  )

type Line = [string, string]
