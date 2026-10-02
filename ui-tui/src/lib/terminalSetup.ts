// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join } from 'node:path'

export type SupportedTerminal = 'cursor' | 'vscode' | 'windsurf'

export type FileOps = {
  copyFile: typeof copyFile
  mkdir: typeof mkdir
  readFile: typeof readFile
  writeFile: typeof writeFile
}

type Keybinding = {
  args?: { text?: string }
  command?: string
  key?: string
  when?: string
}

export type TerminalSetupResult = {
  message: string
  requiresRestart?: boolean
  success: boolean
}

const DEFAULT_FILE_OPS: FileOps = { copyFile, mkdir, readFile, writeFile }
const COPY_SEQUENCE = '\u001b[99;13u'
const MULTILINE_SEQUENCE = '\\\r\n'

const TERMINAL_META: Record<SupportedTerminal, { appName: string; label: string }> = {
  vscode: { appName: 'Code', label: 'VS Code' },
  cursor: { appName: 'Cursor', label: 'Cursor' },
  windsurf: { appName: 'Windsurf', label: 'Windsurf' }
}

const MAC_COPY_BINDING: Keybinding = {
  key: 'cmd+c',
  command: 'workbench.action.terminal.sendSequence',
  when: 'terminalFocus && terminalTextSelected',
  args: { text: COPY_SEQUENCE }
}

const BASE_BINDINGS: Keybinding[] = [
  {
    key: 'shift+enter',
    command: 'workbench.action.terminal.sendSequence',
    when: 'terminalFocus',
    args: { text: MULTILINE_SEQUENCE }
  },
  {
    key: 'ctrl+enter',
    command: 'workbench.action.terminal.sendSequence',
    when: 'terminalFocus',
    args: { text: MULTILINE_SEQUENCE }
  },
  {
    key: 'cmd+enter',
    command: 'workbench.action.terminal.sendSequence',
    when: 'terminalFocus',
    args: { text: MULTILINE_SEQUENCE }
  },
  {
    key: 'cmd+z',
    command: 'workbench.action.terminal.sendSequence',
    when: 'terminalFocus',
    args: { text: '\u001b[122;9u' }
  },
  {
    key: 'shift+cmd+z',
    command: 'workbench.action.terminal.sendSequence',
    when: 'terminalFocus',
    args: { text: '\u001b[122;10u' }
  }
]

const targetBindings = (platform: NodeJS.Platform): Keybinding[] =>
  platform === 'darwin' ? [MAC_COPY_BINDING, ...BASE_BINDINGS] : BASE_BINDINGS

export function detectVSCodeLikeTerminal(env: NodeJS.ProcessEnv = process.env): null | SupportedTerminal {
  const askpass = env['VSCODE_GIT_ASKPASS_MAIN']?.toLowerCase() ?? ''

  if (env['CURSOR_TRACE_ID'] || askpass.includes('cursor')) {
    return 'cursor'
  }

  if (askpass.includes('windsurf')) {
    return 'windsurf'
  }

  if (env['TERM_PROGRAM'] === 'vscode' || env['VSCODE_GIT_IPC_HANDLE']) {
    return 'vscode'
  }

  return null
}

/**
 * 去除 JSONC 特性（// 行注释、/* 块注释 *\/ 和尾随逗号）
 * 使结果成为可由 JSON.parse() 解析的有效 JSON。
 * 字符串内部的注释形态会被正确识别并保留。
 */
export function stripJsonComments(content: string): string {
  let result = ''
  let i = 0
  const len = content.length

  while (i < len) {
    const ch = content[i]!

    // 字符串字面量保持原样复制，包括内部形似注释的字符。
    if (ch === '"') {
      let j = i + 1

      while (j < len) {
        if (content[j] === '\\') {
          j += 2 // 跳过转义字符。
        } else if (content[j] === '"') {
          j++

          break
        } else {
          j++
        }
      }

      result += content.slice(i, j)
      i = j

      continue
    }

    // 行注释。
    if (ch === '/' && content[i + 1] === '/') {
      const eol = content.indexOf('\n', i)
      i = eol === -1 ? len : eol

      continue
    }

    // 块注释。
    if (ch === '/' && content[i + 1] === '*') {
      const end = content.indexOf('*/', i + 2)
      i = end === -1 ? len : end + 2

      continue
    }

    result += ch
    i++
  }

  // 删除 ] 或 } 前的尾随逗号。
  return result.replace(/,(\s*[}\]])/g, '$1')
}

export function isRemoteShellSession(env: NodeJS.ProcessEnv): boolean {
  return Boolean(env['SSH_CONNECTION'] || env['SSH_TTY'] || env['SSH_CLIENT'])
}

export function getVSCodeStyleConfigDir(
  appName: string,
  platform: NodeJS.Platform = process.platform,
  env: NodeJS.ProcessEnv = process.env,
  homeDir: string = homedir()
): null | string {
  if (platform === 'darwin') {
    return join(homeDir, 'Library', 'Application Support', appName, 'User')
  }

  if (platform === 'win32') {
    return env['APPDATA'] ? join(env['APPDATA'], appName, 'User') : null
  }

  return join(homeDir, '.config', appName, 'User')
}

function isKeybinding(value: unknown): value is Keybinding {
  return typeof value === 'object' && value !== null
}

function sameBinding(a: Keybinding, b: Keybinding): boolean {
  return a.key === b.key && a.command === b.command && a.when === b.when && a.args?.text === b.args?.text
}

type WhenRequirements = {
  forbidden: Set<string>
  required: Set<string>
}

const WHEN_TOKEN_RE = /!?[A-Za-z_][\w.]*/g

function parseWhenRequirements(when: string): WhenRequirements {
  const required = new Set<string>()
  const forbidden = new Set<string>()

  for (const [token] of when.matchAll(WHEN_TOKEN_RE)) {
    if (token.startsWith('!')) {
      forbidden.add(token.slice(1))
    } else {
      required.add(token)
    }
  }

  return { forbidden, required }
}

function requirementsContradict(a: WhenRequirements, b: WhenRequirements): boolean {
  for (const token of a.required) {
    if (b.forbidden.has(token)) {
      return true
    }
  }

  for (const token of b.required) {
    if (a.forbidden.has(token)) {
      return true
    }
  }

  return false
}

function whensOverlap(a: string, b: string): boolean {
  if (a === b) {
    return true
  }

  // 空 when 表示全局生效，会与所有上下文重叠。
  if (!a || !b) {
    return true
  }

  const left = parseWhenRequirements(a)
  const right = parseWhenRequirements(b)

  if (requirementsContradict(left, right)) {
    return false
  }

  // 这里刻意不实现完整的 VS Code when 子句解析器。若同键绑定共享一个正向
  // 上下文 token 且没有显式互斥，就可能在该上下文中同时触发。
  for (const token of left.required) {
    if (right.required.has(token)) {
      return true
    }
  }

  return false
}

// VS Code 允许同一按键存在多个绑定，前提是 `when` 子句不重叠。上下文重叠而
// 绑定不同即视为冲突；例如已有的 `terminalFocus` cmd+c 与我们的
// `terminalFocus && terminalTextSelected` 重叠，未选中文本时前者会遮蔽后者。
function bindingsConflict(existing: Keybinding, target: Keybinding): boolean {
  if (existing.key !== target.key) {
    return false
  }

  if (!whensOverlap(existing.when ?? '', target.when ?? '')) {
    return false
  }

  return !sameBinding(existing, target)
}

async function backupFile(filePath: string, ops: FileOps): Promise<void> {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-')
  await ops.copyFile(filePath, `${filePath}.backup.${stamp}`)
}

export async function configureTerminalKeybindings(
  terminal: SupportedTerminal,
  options?: {
    env?: NodeJS.ProcessEnv
    fileOps?: Partial<FileOps>
    homeDir?: string
    platform?: NodeJS.Platform
  }
): Promise<TerminalSetupResult> {
  const env = options?.env ?? process.env
  const platform = options?.platform ?? process.platform
  const homeDir = options?.homeDir ?? homedir()
  const ops: FileOps = { ...DEFAULT_FILE_OPS, ...(options?.fileOps ?? {}) }
  const meta = TERMINAL_META[terminal]

  if (isRemoteShellSession(env)) {
    return {
      success: false,
      message: `${meta.label} terminal setup must be run on the local machine, not inside an SSH session.`
    }
  }

  const configDir = getVSCodeStyleConfigDir(meta.appName, platform, env, homeDir)

  if (!configDir) {
    return {
      success: false,
      message: `Could not determine ${meta.label} settings path on this platform.`
    }
  }

  const keybindingsFile = join(configDir, 'keybindings.json')

  try {
    await ops.mkdir(configDir, { recursive: true })

    let keybindings: unknown[] = []
    let hasExistingFile = false

    try {
      const content = await ops.readFile(keybindingsFile, 'utf8')
      hasExistingFile = true
      const parsed: unknown = JSON.parse(stripJsonComments(content))

      if (!Array.isArray(parsed)) {
        return {
          success: false,
          message: `${meta.label} keybindings.json is not a JSON array: ${keybindingsFile}`
        }
      }

      keybindings = parsed
    } catch (error) {
      const code = (error as NodeJS.ErrnoException | undefined)?.code

      if (code !== 'ENOENT') {
        return {
          success: false,
          message: `Failed to read ${meta.label} keybindings: ${error}`
        }
      }
    }

    const targets = targetBindings(platform)

    const conflicts = targets.filter(target =>
      keybindings.some(existing => isKeybinding(existing) && bindingsConflict(existing, target))
    )

    if (conflicts.length) {
      return {
        success: false,
        message:
          `Existing terminal keybindings would conflict in ${keybindingsFile}: ` + conflicts.map(c => c.key).join(', ')
      }
    }

    let added = 0

    for (const target of targets.slice().reverse()) {
      const exists = keybindings.some(existing => isKeybinding(existing) && sameBinding(existing, target))

      if (!exists) {
        keybindings.unshift(target)
        added += 1
      }
    }

    if (!added) {
      return {
        success: true,
        message: `${meta.label} terminal keybindings already configured.`
      }
    }

    if (hasExistingFile) {
      await backupFile(keybindingsFile, ops)
    }

    await ops.writeFile(keybindingsFile, `${JSON.stringify(keybindings, null, 2)}\n`, 'utf8')

    return {
      success: true,
      requiresRestart: true,
      message: `Added ${added} ${meta.label} terminal keybinding${added === 1 ? '' : 's'} in ${keybindingsFile}`
    }
  } catch (error) {
    return {
      success: false,
      message: `Failed to configure ${meta.label} terminal shortcuts: ${error}`
    }
  }
}

export async function configureDetectedTerminalKeybindings(options?: {
  env?: NodeJS.ProcessEnv
  fileOps?: Partial<FileOps>
  homeDir?: string
  platform?: NodeJS.Platform
}): Promise<TerminalSetupResult> {
  const detected = detectVSCodeLikeTerminal(options?.env ?? process.env)

  if (!detected) {
    return {
      success: false,
      message: 'No supported IDE terminal detected. Supported: VS Code, Cursor, Windsurf.'
    }
  }

  return configureTerminalKeybindings(detected, options)
}

export async function shouldPromptForTerminalSetup(options?: {
  env?: NodeJS.ProcessEnv
  fileOps?: Partial<FileOps>
  homeDir?: string
  platform?: NodeJS.Platform
}): Promise<boolean> {
  const env = options?.env ?? process.env
  const detected = detectVSCodeLikeTerminal(env)

  if (!detected || isRemoteShellSession(env)) {
    return false
  }

  const platform = options?.platform ?? process.platform
  const homeDir = options?.homeDir ?? homedir()
  const ops: FileOps = { ...DEFAULT_FILE_OPS, ...(options?.fileOps ?? {}) }
  const meta = TERMINAL_META[detected]
  const configDir = getVSCodeStyleConfigDir(meta.appName, platform, env, homeDir)

  if (!configDir) {
    return false
  }

  try {
    const content = await ops.readFile(join(configDir, 'keybindings.json'), 'utf8')
    const parsed: unknown = JSON.parse(stripJsonComments(content))

    if (!Array.isArray(parsed)) {
      return true
    }

    return targetBindings(platform).some(
      target => !parsed.some(existing => isKeybinding(existing) && sameBinding(existing, target))
    )
  } catch {
    return true
  }
}
