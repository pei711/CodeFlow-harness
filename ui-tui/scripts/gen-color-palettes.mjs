#!/usr/bin/env node
// src/theme.ts 中降级调色板的一次性生成器。
//
// 真彩色（第 3 级）调色板是事实来源，以十六进制形式手工写在 theme.ts 中。
// 256 色（第 2 级）和 16 色（第 1 级）调色板由本脚本派生，并以字面量冻结到
// theme.ts，避免应用启动时执行 RGB 到 ANSI 的转换。真彩色调色板变化时重新运行
// 本脚本并粘贴输出：
//
//   node scripts/gen-color-palettes.mjs
//
// 256 色映射采用保留色相的算法，与 hermes-ink 曾用于旧版 Apple Terminal 的
// 算法相同，使深绿色保持绿色，而不是折叠到 Chalk 朴素 rgbToAnsi256 选择的
// 橄榄色色块。16 色映射按 RGB 距离选取最接近的颜色。

// --- 真彩色源调色板（与 theme.ts 保持同步）----------------------------------

const DARK = {
  primary: '#7CC950',
  accent: '#9ED66E',
  border: '#4F7A45',
  text: '#E6F2DD',
  muted: '#6FA05C',
  completionBg: '#273328',
  completionCurrentBg: '#3C5A3D',
  completionMetaBg: '#273328',
  completionMetaCurrentBg: '#3C5A3D',
  label: '#8FBF6B',
  ok: '#4caf50',
  error: '#ef5350',
  warn: '#ffa726',
  prompt: '#E6F2DD',
  sessionLabel: '#6FA05C',
  sessionBorder: '#6FA05C',
  statusBg: '#273328',
  statusFg: '#C8D6BF',
  statusGood: '#8FBC8F',
  statusWarn: '#FFD700',
  statusBad: '#FF8C00',
  statusCritical: '#FF6B6B',
  selectionBg: '#3C5A3D',
  diffAdded: 'rgb(220,255,220)',
  diffRemoved: 'rgb(255,220,220)',
  diffAddedWord: 'rgb(36,138,61)',
  diffRemovedWord: 'rgb(207,34,46)',
  shellDollar: '#4dabf7'
}

const LIGHT = {
  primary: '#2E6B2E',
  accent: '#3A7D3A',
  border: '#2F5F2F',
  text: '#273328',
  muted: '#3F6B3F',
  completionBg: '#EDF3EA',
  completionCurrentBg: '#C9DCC1',
  completionMetaBg: '#EDF3EA',
  completionMetaCurrentBg: '#C9DCC1',
  label: '#3F6B3F',
  ok: '#2E7D32',
  error: '#C62828',
  warn: '#E65100',
  prompt: '#273328',
  sessionLabel: '#3F6B3F',
  sessionBorder: '#3F6B3F',
  statusBg: '#EDF3EA',
  statusFg: '#2A3A2A',
  statusGood: '#2E7D32',
  statusWarn: '#8B6914',
  statusBad: '#D84315',
  statusCritical: '#B71C1C',
  selectionBg: '#CBE3C0',
  diffAdded: 'rgb(200,240,200)',
  diffRemoved: 'rgb(240,200,200)',
  diffAddedWord: 'rgb(27,94,32)',
  diffRemovedWord: 'rgb(183,28,28)',
  shellDollar: '#1565C0'
}

// --- 解析 ------------------------------------------------------------------

function toRgb(value) {
  const hex = /^#([0-9a-f]{6})$/i.exec(value)
  if (hex) {
    const n = Number.parseInt(hex[1], 16)
    return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff]
  }
  const rgb = /^rgb\(\s?(\d+),\s?(\d+),\s?(\d+)\s?\)$/.exec(value)
  if (rgb) {
    return [Number(rgb[1]), Number(rgb[2]), Number(rgb[3])]
  }
  throw new Error(`unparseable color: ${value}`)
}

// --- 256 色（保留色相）-----------------------------------------------------

function ansi256(red, green, blue) {
  const rn = red / 255
  const gn = green / 255
  const bn = blue / 255
  const max = Math.max(rn, gn, bn)
  const min = Math.min(rn, gn, bn)
  const lightness = (max + min) / 2
  const saturation =
    max === min ? 0 : lightness > 0.5 ? (max - min) / (2 - max - min) : (max - min) / (max + min)

  if (saturation < 0.15) {
    const gray = Math.round(lightness * 25)
    return gray === 0 ? 16 : gray === 25 ? 231 : 231 + gray
  }

  const sixRed = red < 95 ? red / 95 : 1 + (red - 95) / 40
  const sixGreen = green < 95 ? green / 95 : 1 + (green - 95) / 40
  const sixBlue = blue < 95 ? blue / 95 : 1 + (blue - 95) / 40

  return 16 + 36 * Math.round(sixRed) + 6 * Math.round(sixGreen) + Math.round(sixBlue)
}

// --- 16 色（按 RGB 距离取最近值）--------------------------------------------

const ANSI16 = [
  ['ansi:black', [0, 0, 0]],
  ['ansi:red', [128, 0, 0]],
  ['ansi:green', [0, 128, 0]],
  ['ansi:yellow', [128, 128, 0]],
  ['ansi:blue', [0, 0, 128]],
  ['ansi:magenta', [128, 0, 128]],
  ['ansi:cyan', [0, 128, 128]],
  ['ansi:white', [192, 192, 192]],
  ['ansi:blackBright', [128, 128, 128]],
  ['ansi:redBright', [255, 0, 0]],
  ['ansi:greenBright', [0, 255, 0]],
  ['ansi:yellowBright', [255, 255, 0]],
  ['ansi:blueBright', [0, 0, 255]],
  ['ansi:magentaBright', [255, 0, 255]],
  ['ansi:cyanBright', [0, 255, 255]],
  ['ansi:whiteBright', [255, 255, 255]]
]

function nearest16(red, green, blue) {
  let best = ANSI16[0][0]
  let bestScore = Number.POSITIVE_INFINITY
  for (const [name, [r, g, b]] of ANSI16) {
    const score = (r - red) ** 2 + (g - green) ** 2 + (b - blue) ** 2
    if (score < bestScore) {
      bestScore = score
      best = name
    }
  }
  return best
}

// --- 输出 ------------------------------------------------------------------

function block(palette, mode) {
  const lines = []
  for (const [key, value] of Object.entries(palette)) {
    const [r, g, b] = toRgb(value)
    const out = mode === '256' ? `ansi256(${ansi256(r, g, b)})` : nearest16(r, g, b)
    lines.push(`    ${key}: '${out}',`)
  }
  return lines.join('\n')
}

for (const [name, palette] of [['DARK', DARK], ['LIGHT', LIGHT]]) {
  for (const mode of ['256', '16']) {
    console.log(`// ${name}_${mode}`)
    console.log('  {')
    console.log(block(palette, mode))
    console.log('  },\n')
  }
}
