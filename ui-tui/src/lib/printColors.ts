// `codeflow --print-colors`（平面色块）与 `--preview-colors`（在真实界面上下文中
// 展示 token）所用的诊断颜色渲染器。
//
// 两者使用与界面完全相同的着色路径，因此输出准确反映当前颜色级别或通过
// --color 强制指定的级别。可换用不同的 `--color <tier>` 重新运行以比较级别。

import { colorize } from '@hermes/ink'

import type { Theme, ThemeColors } from '../theme.js'

const TIER_NAMES: Record<number, string> = {
  0: 'none',
  1: '16-color',
  2: '256-color',
  3: 'truecolor'
}

const fg = (s: string, color: string) => colorize(s, color, 'foreground')
// 在一个单元格中组合背景色与前景色，与真实组件中的补全行、状态栏、选择项和
// 差异行一致。
const fgbg = (s: string, fgColor: string, bgColor: string) =>
  colorize(colorize(s, bgColor, 'background'), fgColor, 'foreground')

export function renderColorSwatches(theme: Theme, tier: 0 | 1 | 2 | 3): string {
  const roles = Object.keys(theme.color) as (keyof ThemeColors)[]
  const labelWidth = Math.max(...roles.map(r => r.length))

  const lines: string[] = []
  lines.push(`CodeFlow TUI palette — tier ${tier} (${TIER_NAMES[tier] ?? 'unknown'})`)
  lines.push('')

  for (const role of roles) {
    const value = theme.color[role]
    const label = role.padEnd(labelWidth)
    const swatch = colorize('████████', value, 'foreground')
    lines.push(`  ${label}  ${swatch}  ${value}`)
  }

  lines.push('')
  lines.push('Force a tier to compare:  codeflow --color <truecolor|256|16|none> --print-colors')

  return lines.join('\n') + '\n'
}

/**
 * 在 token 实际使用的上下文中渲染，使设计者看到真实的前景色与背景色组合，
 * 而不是孤立色块。
 */
export function renderColorPreview(theme: Theme, tier: 0 | 1 | 2 | 3): string {
  const c = theme.color
  const out: string[] = []
  const section = (title: string) => {
    out.push('')
    out.push(fg(`── ${title} `.padEnd(60, '─'), c.muted))
  }

  out.push(`CodeFlow TUI color usage preview — tier ${tier} (${TIER_NAMES[tier] ?? 'unknown'})`)

  section('Transcript surfaces')
  out.push('  ' + fgbg(' base transcript ', c.text, c.surface))
  out.push('  ' + fgbg(' ❯ user prompt ', c.text, c.surfaceRaised))

  section('Prompt & input')
  out.push('  ' + fg('❯', c.prompt) + ' ' + fg('ask me something…', c.muted))
  out.push('  ' + fg('$', c.shellDollar) + ' ' + fg('git status', c.text))

  section('Text roles')
  out.push(
    '  ' +
      [
        fg('primary', c.primary),
        fg('accent / link', c.accent),
        fg('heading', c.heading),
        fg('path', c.path),
        fg('thinking', c.thinking),
        fg('body text', c.text),
        fg('muted', c.muted),
        fg('label', c.label)
      ].join('   ')
  )

  section('Semantic')
  out.push('  ' + [fg('✓ ok', c.ok), fg('⚠ warn', c.warn), fg('✗ error', c.error)].join('   '))

  section('Status bar (fg on statusBg)')
  out.push(
    '  ' +
      fgbg(' ● READY ', c.statusGood, c.statusBg) +
      fgbg(' main ', c.statusFg, c.statusBg) +
      fgbg(' ⚠ 2 ', c.statusWarn, c.statusBg) +
      fgbg(' ✗ 1 ', c.statusBad, c.statusBg) +
      fgbg(' ‼ FATAL ', c.statusCritical, c.statusBg)
  )

  section('Completion menu (row bg + meta column)')
  out.push('  ' + fgbg(' /help    ', c.text, c.completionBg) + fgbg(' show commands ', c.muted, c.completionMetaBg))
  out.push(
    '  ' +
      fgbg(' /model   ', c.text, c.completionCurrentBg) +
      fgbg(' switch model  ', c.muted, c.completionMetaCurrentBg) +
      fg('  ← current', c.muted)
  )
  out.push('  ' + fgbg(' /clear   ', c.text, c.completionBg) + fgbg(' reset session ', c.muted, c.completionMetaBg))

  section('Selection')
  out.push('  ' + fg('normal ', c.text) + fgbg('selected text', c.text, c.selectionBg) + fg(' normal', c.text))

  section('Session box (border / sessionLabel / sessionBorder)')
  out.push('  ' + fg('╭────────────────────────────╮', c.sessionBorder))
  out.push(
    '  ' +
      fg('│ ', c.sessionBorder) +
      fg('Session ', c.sessionLabel) +
      fg('a1b2c3d4', c.accent) +
      fg('           │', c.sessionBorder)
  )
  out.push('  ' + fg('╰────────────────────────────╯', c.sessionBorder))

  section('Diff (line bg = diffAdded/Removed, word fg = diff*Word)')
  out.push('  ' + fgbg('+ added line of code', c.diffAddedWord, c.diffAdded))
  out.push('  ' + fgbg('- removed line of code', c.diffRemovedWord, c.diffRemoved))

  out.push('')
  out.push(fg('Force a tier:  codeflow --color <truecolor|256|16|none> --preview-colors', c.muted))

  return out.join('\n') + '\n'
}
