// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { afterEach, describe, expect, it, vi } from 'vitest'

// `theme.js` 在模块加载时读取 `process.env` 计算 DEFAULT_THEME，而 `fromSkin`
// 会闭包捕获 DEFAULT_THEME。若开发者 shell 设置 CODEFLOW_TUI_THEME=light，或将
// CODEFLOW_TUI_BACKGROUND 设为亮色，基准主题会翻转，使这些断言只在本地失败。
// 因此先清理相关环境变量，再动态导入全新模块，确保所有闭包捕获环境的符号
// （DEFAULT_THEME、DARK_THEME、LIGHT_THEME、fromSkin）都基于已知的空环境加载。
//
// `detectLightMode` 显式接收 env，静态导入也安全；为保持一致仍采用动态导入。
const RELEVANT_ENV = [
  'CODEFLOW_TUI_LIGHT',
  'CODEFLOW_TUI_THEME',
  'CODEFLOW_TUI_BACKGROUND',
  'COLORFGBG',
  'COLORTERM',
  'TERM_PROGRAM'
] as const

async function importThemeWithEnv(env: Partial<Record<(typeof RELEVANT_ENV)[number], string>> = {}) {
  for (const key of RELEVANT_ENV) {
    vi.stubEnv(key, env[key] ?? '')
  }

  vi.resetModules()

  return import('../theme.js')
}

async function importThemeWithCleanEnv() {
  return importThemeWithEnv()
}

afterEach(() => {
  vi.unstubAllEnvs()
  vi.resetModules()
})

describe('DEFAULT_THEME', () => {
  it('has brand defaults', async () => {
    const { DEFAULT_THEME } = await importThemeWithCleanEnv()

    expect(DEFAULT_THEME.brand.name).toBe('CodeFlow')
    expect(DEFAULT_THEME.brand.icon).toBe('◆')
    expect(DEFAULT_THEME.brand.prompt).toBe('❯')
    expect(DEFAULT_THEME.brand.tool).toBe('·')
  })

  it('has color palette', async () => {
    const { DEFAULT_THEME } = await importThemeWithCleanEnv()

    expect(DEFAULT_THEME.color.primary).toBe('#c8c8c8')
    expect(DEFAULT_THEME.color.accent).toBe('#7aa2f7')
    expect(DEFAULT_THEME.color.surface).toBe('#141414')
    expect(DEFAULT_THEME.color.surfaceRaised).toBe('#242424')
    expect(DEFAULT_THEME.color.thinking).toBe('#bb9af7')
    expect(DEFAULT_THEME.color.error).toBe('#f7768e')
  })
})

describe('LIGHT_THEME', () => {
  it('avoids bright-yellow accents unreadable on white backgrounds (#11300)', async () => {
    const { LIGHT_THEME } = await importThemeWithCleanEnv()

    expect(LIGHT_THEME.color.primary).not.toBe('#FFD700')
    expect(LIGHT_THEME.color.accent).not.toBe('#FFBF00')
    expect(LIGHT_THEME.color.muted).not.toBe('#B8860B')
    expect(LIGHT_THEME.color.statusWarn).not.toBe('#FFD700')
  })

  it('keeps the same shape as DARK_THEME', async () => {
    const { DARK_THEME, LIGHT_THEME } = await importThemeWithCleanEnv()

    expect(Object.keys(LIGHT_THEME.color).sort()).toEqual(Object.keys(DARK_THEME.color).sort())
    expect(LIGHT_THEME.brand).toEqual(DARK_THEME.brand)
  })
})

describe('brand yellow ramp (title-gradient-table.md)', () => {
  // 真彩色顺序为 [.50,.100,.300,.500,.600,.700,.900,.950,.990]，额外的 .600
  // 使 .700 位于索引 5、.900 位于索引 6。256 色级别保留八阶集合
  // [.50,.100,.300,.500,.700,.900,.950,.990]，其中 .700 位于索引 4。
  it('keeps the documented dark title bands', async () => {
    const { DARK_THEME } = await importThemeWithCleanEnv()

    expect(DARK_THEME.yellow[0]).toBe('#fff7c2') // .50
    expect(DARK_THEME.yellow[2]).toBe('#FFE573') // .300
    expect(DARK_THEME.yellow[3]).toBe('#fbe23f') // .500
    expect(DARK_THEME.yellow[5]).toBe('#c8a900') // .700
    expect(DARK_THEME.yellow[6]).toBe('#8a6d00') // .900
  })

  it('gives light its own gold ramp re-derived around #B87900, not the dark scale', async () => {
    const { DARK_THEME, LIGHT_THEME } = await importThemeWithCleanEnv()

    expect(LIGHT_THEME.yellow[0]).toBe('#F6DA8B') // .50
    expect(LIGHT_THEME.yellow[2]).toBe('#D9A83A') // .300
    expect(LIGHT_THEME.yellow[3]).toBe('#B87900') // .500
    expect(LIGHT_THEME.yellow[5]).toBe('#935F00') // .700
    expect(LIGHT_THEME.yellow[6]).toBe('#684300') // .900

    expect(LIGHT_THEME.yellow).not.toEqual(DARK_THEME.yellow)
  })

  it('carries scheme-specific 256-color ramps for the documented title bands', async () => {
    const { resolveTheme } = await importThemeWithCleanEnv()

    const dark = resolveTheme('dark', 2).yellow
    const light = resolveTheme('light', 2).yellow

    // 256 色采用八阶集合：.50/.300/.500/.700/.900 对应索引 0/2/3/4/5。
    expect([dark[0], dark[2], dark[3], dark[4], dark[5]]).toEqual([
      'ansi256(229)',
      'ansi256(228)',
      'ansi256(220)',
      'ansi256(178)',
      'ansi256(94)'
    ])
    // yellow.300 与 yellow.500 不能折叠到同一个 256 色索引。
    expect(dark[2]).not.toBe(dark[3])
    expect([light[0], light[2], light[3], light[4], light[5]]).toEqual([
      'ansi256(222)',
      'ansi256(179)',
      'ansi256(136)',
      'ansi256(94)',
      'ansi256(58)'
    ])

    expect(light).not.toEqual(dark)
  })

  it('keeps 16-color yellow for both schemes', async () => {
    const { resolveTheme } = await importThemeWithCleanEnv()

    expect(resolveTheme('light', 1).yellow).toEqual(resolveTheme('dark', 1).yellow)
    expect(resolveTheme('light', 1).yellow[0]).toBe('ansi:yellow')
  })
})

describe('cursorColorHex (OSC 12 hardware cursor)', () => {
  it('returns a truecolor hex even when the theme tier is 256/16 (default dark)', async () => {
    const { cursorColorHex, resolveTheme } = await importThemeWithCleanEnv()

    // 第 1/2 级保存 ANSI 索引，但 OSC 12 需要 RGB 颜色。
    expect(cursorColorHex(resolveTheme('dark', 3))).toBe('#c8c8c8')
    expect(cursorColorHex(resolveTheme('dark', 2))).toBe('#c8c8c8')
    expect(cursorColorHex(resolveTheme('dark', 1))).toBe('#c8c8c8')
  })

  it('uses the light title color across all tiers when the scheme is light', async () => {
    const { cursorColorHex, resolveTheme } = await importThemeWithEnv({ CODEFLOW_TUI_THEME: 'light' })

    expect(cursorColorHex(resolveTheme('light', 3))).toBe('#444444')
    expect(cursorColorHex(resolveTheme('light', 2))).toBe('#444444')
    expect(cursorColorHex(resolveTheme('light', 1))).toBe('#444444')
  })
})

describe('DEFAULT_THEME aliasing', () => {
  it('defaults to DARK_THEME when nothing signals light', async () => {
    const { DEFAULT_THEME, DARK_THEME: DARK } = await importThemeWithCleanEnv()

    expect(DEFAULT_THEME).toBe(DARK)
  })
})

describe('detectLightMode', () => {
  it('returns false on empty env', async () => {
    const { detectLightMode } = await importThemeWithCleanEnv()

    expect(detectLightMode({})).toBe(false)
  })

  it('stays dark on Apple Terminal when no stronger signal is present', async () => {
    const { detectLightMode } = await importThemeWithCleanEnv()

    // TERM_PROGRAM 本身不再表示亮色：Terminal.app 同时提供明暗配置且不发送
    // COLORFGBG，因此默认按暗色处理。
    expect(detectLightMode({ TERM_PROGRAM: 'Apple_Terminal' })).toBe(false)
  })

  it('honors CODEFLOW_TUI_LIGHT on/off', async () => {
    const { detectLightMode } = await importThemeWithCleanEnv()

    expect(detectLightMode({ CODEFLOW_TUI_LIGHT: '1' })).toBe(true)
    expect(detectLightMode({ CODEFLOW_TUI_LIGHT: 'true' })).toBe(true)
    expect(detectLightMode({ CODEFLOW_TUI_LIGHT: 'on' })).toBe(true)
    expect(detectLightMode({ CODEFLOW_TUI_LIGHT: '0' })).toBe(false)
    expect(detectLightMode({ CODEFLOW_TUI_LIGHT: 'off' })).toBe(false)
  })

  it('sniffs COLORFGBG bg slots 7 and 15 as light (#11300)', async () => {
    const { detectLightMode } = await importThemeWithCleanEnv()

    expect(detectLightMode({ COLORFGBG: '0;15' })).toBe(true)
    expect(detectLightMode({ COLORFGBG: '0;default;15' })).toBe(true)
    expect(detectLightMode({ COLORFGBG: '0;7' })).toBe(true)
    expect(detectLightMode({ COLORFGBG: '15;0' })).toBe(false)
    expect(detectLightMode({ COLORFGBG: '7;default;0' })).toBe(false)
  })

  it('falls through on malformed COLORFGBG with empty/non-numeric trailing field', async () => {
    const { detectLightMode } = await importThemeWithCleanEnv()
    // `Number('')` 为 0，因此 `'15;'` 曾会被读成 bg=0（确定暗色）并错误阻止
    // TERM_PROGRAM。严格的 /^\d+$/ 保护会让这类值改走回退路径。
    const allowList = new Set(['Apple_Terminal'])

    expect(detectLightMode({ COLORFGBG: '15;', TERM_PROGRAM: 'Apple_Terminal' }, allowList)).toBe(true)
    expect(detectLightMode({ COLORFGBG: 'default;default', TERM_PROGRAM: 'Apple_Terminal' }, allowList)).toBe(true)
    // 未匹配允许列表时，回退路径仍默认为暗色。
    expect(detectLightMode({ COLORFGBG: '15;' })).toBe(false)
  })

  it('lets CODEFLOW_TUI_LIGHT=0 override a light COLORFGBG', async () => {
    const { detectLightMode } = await importThemeWithCleanEnv()

    expect(detectLightMode({ COLORFGBG: '0;15', CODEFLOW_TUI_LIGHT: '0' })).toBe(false)
  })

  it('honors CODEFLOW_TUI_THEME=light/dark as a symmetric explicit override', async () => {
    const { detectLightMode } = await importThemeWithCleanEnv()

    expect(detectLightMode({ CODEFLOW_TUI_THEME: 'light' })).toBe(true)
    expect(detectLightMode({ CODEFLOW_TUI_THEME: 'dark' })).toBe(false)
    expect(detectLightMode({ COLORFGBG: '0;15', CODEFLOW_TUI_THEME: 'dark' })).toBe(false)
    expect(detectLightMode({ COLORFGBG: '15;0', CODEFLOW_TUI_THEME: 'light' })).toBe(true)
  })

  it('uses CODEFLOW_TUI_BACKGROUND luminance when COLORFGBG is missing', async () => {
    const { detectLightMode } = await importThemeWithCleanEnv()

    expect(detectLightMode({ CODEFLOW_TUI_BACKGROUND: '#ffffff' })).toBe(true)
    expect(detectLightMode({ CODEFLOW_TUI_BACKGROUND: '#000000' })).toBe(false)
    expect(detectLightMode({ CODEFLOW_TUI_BACKGROUND: '#1e1e1e' })).toBe(false)
    // 三位十六进制颜色按 CSS 规则归一化。
    expect(detectLightMode({ CODEFLOW_TUI_BACKGROUND: '#fff' })).toBe(true)
    // 无效内容回退到默认暗色路径。
    expect(detectLightMode({ CODEFLOW_TUI_BACKGROUND: 'not-a-colour' })).toBe(false)
  })

  it('rejects partially-invalid hex instead of silently truncating', async () => {
    const { detectLightMode } = await importThemeWithCleanEnv()
    // `parseInt('fffgff'.slice(2,4), 16)` 会返回 15，因此严格正则必须拒绝这类
    // 输入，使其回退到默认暗色，而不是产生误报的亮色判断。
    expect(detectLightMode({ CODEFLOW_TUI_BACKGROUND: '#fffgff' })).toBe(false)
    expect(detectLightMode({ CODEFLOW_TUI_BACKGROUND: 'ffggff' })).toBe(false)
    expect(detectLightMode({ CODEFLOW_TUI_BACKGROUND: '#xyz' })).toBe(false)
    // 长度错误同样应被拒绝，不能隐式补齐或截断。
    expect(detectLightMode({ CODEFLOW_TUI_BACKGROUND: '#fffff' })).toBe(false)
    expect(detectLightMode({ CODEFLOW_TUI_BACKGROUND: '#fffffff' })).toBe(false)
  })

  it('treats COLORFGBG as authoritative when present so it dominates the TERM_PROGRAM allow-list', async () => {
    const { detectLightMode } = await importThemeWithCleanEnv()
    // 显式注入允许列表，使生产默认值变化后此优先级规则仍清晰可见。
    const allowList = new Set(['Apple_Terminal'])

    // 基线确认：仅允许列表本身会把该终端判断为亮色。
    expect(detectLightMode({ TERM_PROGRAM: 'Apple_Terminal' }, allowList)).toBe(true)

    // 暗色 COLORFGBG 必须优先于允许列表。
    expect(detectLightMode({ COLORFGBG: '15;0', TERM_PROGRAM: 'Apple_Terminal' }, allowList)).toBe(false)
  })
})

describe('applyDetectedBackground (OSC 11 reply)', () => {
  it('parses 16-bit rgb: replies and flips the scheme to light on a bright bg', async () => {
    const { applyDetectedBackground, currentScheme } = await importThemeWithCleanEnv()

    expect(currentScheme()).toBe('dark')

    const res = applyDetectedBackground('rgb:ffff/ffff/ffff')

    expect(res).toEqual({ changed: true, scheme: 'light' })
    expect(currentScheme()).toBe('light')
  })

  it('parses 8-bit rgb: and #rrggbb dark replies as dark', async () => {
    const { applyDetectedBackground } = await importThemeWithCleanEnv()

    expect(applyDetectedBackground('rgb:1e/1e/2e')).toEqual({ changed: false, scheme: 'dark' })
    expect(applyDetectedBackground('#1e1e2e')).toEqual({ changed: false, scheme: 'dark' })
  })

  it('caches the measured color into CODEFLOW_TUI_BACKGROUND', async () => {
    const { applyDetectedBackground } = await importThemeWithCleanEnv()

    applyDetectedBackground('rgb:eaea/eaea/eaea')

    expect(process.env.CODEFLOW_TUI_BACKGROUND).toBe('#eaeaea')
  })

  it('lets an explicit CODEFLOW_TUI_THEME override the measured background', async () => {
    const { applyDetectedBackground, currentScheme } = await importThemeWithEnv({ CODEFLOW_TUI_THEME: 'dark' })

    // 实测背景为亮色，但显式暗色覆盖优先；这里复用 detectLightMode 的优先级。
    expect(applyDetectedBackground('rgb:ffff/ffff/ffff')).toEqual({ changed: false, scheme: 'dark' })
    expect(currentScheme()).toBe('dark')
  })

  it('returns null and leaves the scheme untouched on an unparseable reply', async () => {
    const { applyDetectedBackground, currentScheme } = await importThemeWithCleanEnv()

    expect(applyDetectedBackground('not-a-color')).toBeNull()
    expect(currentScheme()).toBe('dark')
  })
})

describe('fromSkin', () => {
  // `fromSkin` 闭包捕获由环境派生的 DEFAULT_THEME，因此必须清理环境后再动态导入；
  // 否则外部 CODEFLOW_TUI_THEME=light 会翻转基准调色板，使断言结果依赖开发者 shell。

  it('overrides banner colors', async () => {
    const { fromSkin } = await importThemeWithCleanEnv()

    expect(fromSkin({ banner_title: '#FF0000' }, {}).color.primary).toBe('#FF0000')
  })

  it('preserves unset colors', async () => {
    const { DEFAULT_THEME, fromSkin } = await importThemeWithCleanEnv()

    expect(fromSkin({ banner_title: '#FF0000' }, {}).color.accent).toBe(DEFAULT_THEME.color.accent)
  })

  it('derives completion current background from resolved completion background', async () => {
    const { fromSkin } = await importThemeWithCleanEnv()

    const theme = fromSkin({ banner_accent: '#000000', completion_menu_bg: '#ffffff' }, {})

    expect(theme.color.completionBg).toBe('#ffffff')
    expect(theme.color.completionCurrentBg).toBe('#bfbfbf')
  })

  it('uses active completion color as the selection highlight fallback', async () => {
    const { fromSkin } = await importThemeWithCleanEnv()

    const theme = fromSkin({ completion_menu_current_bg: '#123456' }, {})

    expect(theme.color.selectionBg).toBe('#123456')
  })

  it('maps completion meta background colors from skins', async () => {
    const { fromSkin } = await importThemeWithCleanEnv()

    const theme = fromSkin(
      {
        completion_menu_meta_bg: '#111111',
        completion_menu_meta_current_bg: '#222222'
      },
      {}
    )

    expect(theme.color.completionMetaBg).toBe('#111111')
    expect(theme.color.completionMetaCurrentBg).toBe('#222222')
  })

  it('lets selection_bg override completion highlight colors', async () => {
    const { fromSkin } = await importThemeWithCleanEnv()

    const theme = fromSkin({ completion_menu_current_bg: '#123456', selection_bg: '#654321' }, {})

    expect(theme.color.selectionBg).toBe('#654321')
  })

  it('overrides branding', async () => {
    const { fromSkin } = await importThemeWithCleanEnv()
    const { brand } = fromSkin({}, { agent_name: 'TestBot', prompt_symbol: '$' })

    expect(brand.name).toBe('TestBot')
    expect(brand.prompt).toBe('$')
  })

  it('normalizes skin prompt symbols to trimmed single-line text', async () => {
    const { DEFAULT_THEME, fromSkin } = await importThemeWithCleanEnv()

    expect(fromSkin({}, { prompt_symbol: ' ⚔ ❯ \n' }).brand.prompt).toBe('⚔ ❯')
    expect(fromSkin({}, { prompt_symbol: ' Ψ > \n' }).brand.prompt).toBe('Ψ >')
    expect(fromSkin({}, { prompt_symbol: '\n\t' }).brand.prompt).toBe(DEFAULT_THEME.brand.prompt)
  })

  it('defaults for empty skin', async () => {
    const { DEFAULT_THEME, fromSkin } = await importThemeWithCleanEnv()

    expect(fromSkin({}, {}).color).toEqual(DEFAULT_THEME.color)
    expect(fromSkin({}, {}).brand.icon).toBe(DEFAULT_THEME.brand.icon)
  })

  it('passes banner logo/hero', async () => {
    const { fromSkin } = await importThemeWithCleanEnv()

    expect(fromSkin({}, {}, 'LOGO', 'HERO').bannerLogo).toBe('LOGO')
    expect(fromSkin({}, {}, 'LOGO', 'HERO').bannerHero).toBe('HERO')
  })

  it('maps ui_ color keys + cascades to status', async () => {
    const { fromSkin } = await importThemeWithCleanEnv()
    const { color } = fromSkin({ ui_ok: '#008000' }, {})

    expect(color.ok).toBe('#008000')
    expect(color.statusGood).toBe('#008000')
  })
})

describe('resolveTheme', () => {
  it('returns the hex palette for truecolor and no-color tiers', async () => {
    const { resolveTheme, DARK_THEME, LIGHT_THEME } = await importThemeWithCleanEnv()

    expect(resolveTheme('dark', 3)).toBe(DARK_THEME)
    expect(resolveTheme('dark', 0)).toBe(DARK_THEME)
    expect(resolveTheme('light', 3)).toBe(LIGHT_THEME)
  })

  it('uses curated ansi256 values at tier 2', async () => {
    const { resolveTheme } = await importThemeWithCleanEnv()

    expect(resolveTheme('dark', 2).color.primary).toBe('ansi256(251)')
    expect(resolveTheme('dark', 2).color.thinking).toBe('ansi256(141)')
    expect(resolveTheme('light', 2).color.primary).toBe('ansi256(238)')
  })

  it('uses named 16-color values at tier 1', async () => {
    const { resolveTheme } = await importThemeWithCleanEnv()

    expect(resolveTheme('dark', 1).color.primary).toBe('ansi:white')
    expect(resolveTheme('dark', 1).color.accent).toBe('ansi:blueBright')
    expect(resolveTheme('dark', 1).color.thinking).toBe('ansi:magentaBright')
  })

  it('keeps the same color-role shape across every tier', async () => {
    const { resolveTheme, DARK_THEME } = await importThemeWithCleanEnv()
    const roles = Object.keys(DARK_THEME.color).sort()

    for (const tier of [0, 1, 2, 3] as const) {
      expect(Object.keys(resolveTheme('dark', tier).color).sort()).toEqual(roles)
      expect(Object.keys(resolveTheme('light', tier).color).sort()).toEqual(roles)
    }
  })
})
