// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { describe, expect, it, vi } from 'vitest'

import {
  configureDetectedTerminalKeybindings,
  configureTerminalKeybindings,
  detectVSCodeLikeTerminal,
  getVSCodeStyleConfigDir,
  shouldPromptForTerminalSetup,
  stripJsonComments
} from '../lib/terminalSetup.js'

describe('terminalSetup helpers', () => {
  it('detects VS Code family terminals from environment', () => {
    expect(detectVSCodeLikeTerminal({ CURSOR_TRACE_ID: 'x' } as NodeJS.ProcessEnv)).toBe('cursor')
    expect(detectVSCodeLikeTerminal({ VSCODE_GIT_ASKPASS_MAIN: '/tmp/windsurf' } as NodeJS.ProcessEnv)).toBe('windsurf')
    expect(detectVSCodeLikeTerminal({ TERM_PROGRAM: 'vscode' } as NodeJS.ProcessEnv)).toBe('vscode')
    expect(detectVSCodeLikeTerminal({} as NodeJS.ProcessEnv)).toBeNull()
  })

  it('computes VS Code style config dirs cross-platform', () => {
    expect(getVSCodeStyleConfigDir('Code', 'darwin', {} as NodeJS.ProcessEnv, '/home/me')).toBe(
      '/home/me/Library/Application Support/Code/User'
    )
    expect(getVSCodeStyleConfigDir('Code', 'linux', {} as NodeJS.ProcessEnv, '/home/me')).toBe(
      '/home/me/.config/Code/User'
    )
    expect(
      getVSCodeStyleConfigDir(
        'Code',
        'win32',
        { APPDATA: 'C:/Users/me/AppData/Roaming' } as NodeJS.ProcessEnv,
        '/home/me'
      )
    ).toBe('C:/Users/me/AppData/Roaming/Code/User')
  })

  it('strips line comments from keybindings JSON', () => {
    expect(stripJsonComments('// comment\n[{"key":"shift+enter"}]')).toBe('\n[{"key":"shift+enter"}]')
  })

  it('strips inline comments and block comments', () => {
    expect(stripJsonComments('[{"key":"a"} // inline\n]')).toBe('[{"key":"a"} \n]')
    expect(stripJsonComments('[/* block */{"key":"a"}]')).toBe('[{"key":"a"}]')
  })

  it('removes trailing commas before ] or }', () => {
    expect(JSON.parse(stripJsonComments('[{"key":"a"},]'))).toEqual([{ key: 'a' }])
    expect(JSON.parse(stripJsonComments('[{"key":"a",}]'))).toEqual([{ key: 'a' }])
  })

  it('preserves comment-like sequences inside strings', () => {
    const input = '[{"key":"a","args":{"text":"// not a comment"}}]'
    expect(JSON.parse(stripJsonComments(input))).toEqual([{ key: 'a', args: { text: '// not a comment' } }])
  })

  it('handles unterminated block comments gracefully', () => {
    const input = '[{"key":"a"} /* never closed'
    const stripped = stripJsonComments(input)
    // 未终止注释会一直消费到文件末尾，其余内容仍可解析。
    expect(stripped).toBe('[{"key":"a"} ')
  })
})

// src/lib/terminalSetup.ts 中 configureTerminalKeybindings 的实现刻意加入 SSH
// 环境保护，要求在本机而非 SSH 会话中运行。以下测试假设宿主机不是 SSH 环境，
// 并验证其他错误路径。通过 SSH 访问开发机时跳过它们，使远程开发机和经 SSH
// 代理的 CI 运行器保持通过；本地检出以及未暴露 SSH_* 环境变量的 CI 仍会执行。
const _ssh = Boolean(process.env.SSH_CONNECTION) || Boolean(process.env.SSH_TTY) || Boolean(process.env.SSH_CLIENT)

describe.skipIf(_ssh)('configureTerminalKeybindings', () => {
  it('writes missing bindings into a VS Code style keybindings file', async () => {
    const mkdir = vi.fn().mockResolvedValue(undefined)
    const readFile = vi.fn().mockRejectedValue(Object.assign(new Error('missing'), { code: 'ENOENT' }))
    const writeFile = vi.fn().mockResolvedValue(undefined)
    const copyFile = vi.fn().mockResolvedValue(undefined)

    const result = await configureTerminalKeybindings('vscode', {
      fileOps: { copyFile, mkdir, readFile, writeFile },
      homeDir: '/Users/me',
      platform: 'darwin'
    })

    expect(result.success).toBe(true)
    expect(result.requiresRestart).toBe(true)
    expect(writeFile).toHaveBeenCalledTimes(1)
    expect(copyFile).not.toHaveBeenCalled() // 没有现有文件可备份。
    const written = writeFile.mock.calls[0]?.[1] as string
    expect(written).toContain('cmd+c')
    expect(written).toContain('terminalTextSelected')
    expect(written).toContain('\\u001b[99;13u')
    expect(written).toContain('shift+enter')
    expect(written).toContain('cmd+enter')
    expect(written).toContain('cmd+z')
  })

  it('only adds the Cmd+C forwarding binding on macOS', async () => {
    const mkdir = vi.fn().mockResolvedValue(undefined)
    const readFile = vi.fn().mockRejectedValue(Object.assign(new Error('missing'), { code: 'ENOENT' }))
    const writeFile = vi.fn().mockResolvedValue(undefined)
    const copyFile = vi.fn().mockResolvedValue(undefined)

    const result = await configureTerminalKeybindings('vscode', {
      fileOps: { copyFile, mkdir, readFile, writeFile },
      homeDir: '/home/me',
      platform: 'linux'
    })

    expect(result.success).toBe(true)
    const written = writeFile.mock.calls[0]?.[1] as string
    expect(written).not.toContain('cmd+c')
    expect(written).not.toContain('terminalTextSelected')
    expect(written).not.toContain('\\u001b[99;13u')
    expect(written).toContain('shift+enter')
  })

  it('reports conflicts without overwriting existing bindings', async () => {
    const mkdir = vi.fn().mockResolvedValue(undefined)

    const readFile = vi.fn().mockResolvedValue(
      JSON.stringify([
        {
          key: 'cmd+z',
          command: 'something.else',
          when: 'terminalFocus',
          args: { text: 'noop' }
        }
      ])
    )

    const writeFile = vi.fn().mockResolvedValue(undefined)
    const copyFile = vi.fn().mockResolvedValue(undefined)

    const result = await configureTerminalKeybindings('cursor', {
      fileOps: { copyFile, mkdir, readFile, writeFile },
      homeDir: '/Users/me',
      platform: 'darwin'
    })

    expect(result.success).toBe(false)
    expect(result.message).toContain('cmd+z')
    expect(writeFile).not.toHaveBeenCalled()
    expect(copyFile).not.toHaveBeenCalled() // 未写入时不创建备份。
  })

  it('flags a global (when-less) binding on the same key as a conflict', async () => {
    // 用户 keybindings.json 中没有 `when` 子句的 `cmd+c` 是全局绑定，会与包括
    // 终端范围在内的任意上下文重叠。不能静默添加会遮蔽它的终端范围 cmd+c。
    const mkdir = vi.fn().mockResolvedValue(undefined)

    const readFile = vi.fn().mockResolvedValue(
      JSON.stringify([
        {
          key: 'cmd+c',
          command: 'myExtension.smartCopy'
        }
      ])
    )

    const writeFile = vi.fn().mockResolvedValue(undefined)
    const copyFile = vi.fn().mockResolvedValue(undefined)

    const result = await configureTerminalKeybindings('vscode', {
      fileOps: { copyFile, mkdir, readFile, writeFile },
      homeDir: '/Users/me',
      platform: 'darwin'
    })

    expect(result.success).toBe(false)
    expect(result.message).toContain('cmd+c')
    expect(writeFile).not.toHaveBeenCalled()
  })

  it('flags an overlapping terminal-context binding as a conflict', async () => {
    // 已有、仅限 `terminalFocus` 的 `cmd+c` 与我们的
    // `terminalFocus && terminalTextSelected` 重叠；终端聚焦且选中文本时两者都会
    // 触发，已有绑定会遮蔽我们的绑定。即使字符串不同也应视为冲突。
    const mkdir = vi.fn().mockResolvedValue(undefined)

    const readFile = vi.fn().mockResolvedValue(
      JSON.stringify([
        {
          key: 'cmd+c',
          command: 'workbench.action.terminal.copySelection',
          when: 'terminalFocus'
        }
      ])
    )

    const writeFile = vi.fn().mockResolvedValue(undefined)
    const copyFile = vi.fn().mockResolvedValue(undefined)

    const result = await configureTerminalKeybindings('vscode', {
      fileOps: { copyFile, mkdir, readFile, writeFile },
      homeDir: '/Users/me',
      platform: 'darwin'
    })

    expect(result.success).toBe(false)
    expect(result.message).toContain('cmd+c')
    expect(writeFile).not.toHaveBeenCalled()
  })

  it('does not flag a negated terminalTextSelected binding as a conflict', async () => {
    // “终端聚焦但未选中文本”范围的绑定与要求 terminalTextSelected 的复制转发
    // 绑定在逻辑上互斥。
    const mkdir = vi.fn().mockResolvedValue(undefined)

    const readFile = vi.fn().mockResolvedValue(
      JSON.stringify([
        {
          key: 'cmd+c',
          command: 'workbench.action.terminal.sendSequence',
          when: 'terminalFocus && !terminalTextSelected',
          args: { text: '\u0003' }
        }
      ])
    )

    const writeFile = vi.fn().mockResolvedValue(undefined)
    const copyFile = vi.fn().mockResolvedValue(undefined)

    const result = await configureTerminalKeybindings('vscode', {
      fileOps: { copyFile, mkdir, readFile, writeFile },
      homeDir: '/Users/me',
      platform: 'darwin'
    })

    expect(result.success).toBe(true)
    expect(writeFile).toHaveBeenCalledTimes(1)
  })

  it('does not flag a disjoint-when binding on the same key as a conflict', async () => {
    // VS Code 允许同键绑定在 `when` 子句不重叠时并存。用户已有、仅限编辑器聚焦
    // 的 cmd+c 不应阻止我们添加终端范围的 cmd+c。
    const mkdir = vi.fn().mockResolvedValue(undefined)

    const readFile = vi.fn().mockResolvedValue(
      JSON.stringify([
        {
          key: 'cmd+c',
          command: 'editor.action.clipboardCopyAction',
          when: 'editorFocus'
        }
      ])
    )

    const writeFile = vi.fn().mockResolvedValue(undefined)
    const copyFile = vi.fn().mockResolvedValue(undefined)

    const result = await configureTerminalKeybindings('vscode', {
      fileOps: { copyFile, mkdir, readFile, writeFile },
      homeDir: '/Users/me',
      platform: 'darwin'
    })

    expect(result.success).toBe(true)
    expect(writeFile).toHaveBeenCalledTimes(1)
  })

  it('backs up existing keybindings.json only when writing changes', async () => {
    const mkdir = vi.fn().mockResolvedValue(undefined)
    const readFile = vi.fn().mockResolvedValue(JSON.stringify([]))
    const writeFile = vi.fn().mockResolvedValue(undefined)
    const copyFile = vi.fn().mockResolvedValue(undefined)

    const result = await configureTerminalKeybindings('vscode', {
      fileOps: { copyFile, mkdir, readFile, writeFile },
      homeDir: '/Users/me',
      platform: 'darwin'
    })

    expect(result.success).toBe(true)
    expect(writeFile).toHaveBeenCalledTimes(1)
    expect(copyFile).toHaveBeenCalledTimes(1) // 写入前创建备份。
  })

  it('reports error when keybindings.json is not readable (EACCES)', async () => {
    const mkdir = vi.fn().mockResolvedValue(undefined)
    const readFile = vi.fn().mockRejectedValue(Object.assign(new Error('permission denied'), { code: 'EACCES' }))
    const writeFile = vi.fn().mockResolvedValue(undefined)
    const copyFile = vi.fn().mockResolvedValue(undefined)

    const result = await configureTerminalKeybindings('vscode', {
      fileOps: { copyFile, mkdir, readFile, writeFile },
      homeDir: '/Users/me',
      platform: 'darwin'
    })

    expect(result.success).toBe(false)
    expect(result.message).toContain('Failed to read')
    expect(writeFile).not.toHaveBeenCalled()
  })

  it('auto-detects the current IDE terminal', async () => {
    const mkdir = vi.fn().mockResolvedValue(undefined)
    const readFile = vi.fn().mockRejectedValue(Object.assign(new Error('missing'), { code: 'ENOENT' }))
    const writeFile = vi.fn().mockResolvedValue(undefined)
    const copyFile = vi.fn().mockResolvedValue(undefined)

    const result = await configureDetectedTerminalKeybindings({
      env: { CURSOR_TRACE_ID: 'trace' } as NodeJS.ProcessEnv,
      fileOps: { copyFile, mkdir, readFile, writeFile },
      homeDir: '/Users/me',
      platform: 'darwin'
    })

    expect(result.success).toBe(true)
    expect(writeFile).toHaveBeenCalled()
  })

  it('refuses to configure IDE bindings from an SSH session', async () => {
    const result = await configureDetectedTerminalKeybindings({
      env: { SSH_CONNECTION: '1 2 3 4', TERM_PROGRAM: 'vscode' } as NodeJS.ProcessEnv,
      homeDir: '/Users/me',
      platform: 'darwin'
    })

    expect(result.success).toBe(false)
    expect(result.message).toContain('local machine')
  })

  it('prompts for setup when bindings are missing and suppresses prompt when complete', async () => {
    const readMissing = vi.fn().mockRejectedValue(Object.assign(new Error('missing'), { code: 'ENOENT' }))
    await expect(
      shouldPromptForTerminalSetup({
        env: { TERM_PROGRAM: 'vscode' } as NodeJS.ProcessEnv,
        fileOps: { readFile: readMissing }
      })
    ).resolves.toBe(true)

    const readComplete = vi.fn().mockResolvedValue(
      JSON.stringify([
        {
          key: 'cmd+c',
          command: 'workbench.action.terminal.sendSequence',
          when: 'terminalFocus && terminalTextSelected',
          args: { text: '\u001b[99;13u' }
        },
        {
          key: 'shift+enter',
          command: 'workbench.action.terminal.sendSequence',
          when: 'terminalFocus',
          args: { text: '\\\r\n' }
        },
        {
          key: 'ctrl+enter',
          command: 'workbench.action.terminal.sendSequence',
          when: 'terminalFocus',
          args: { text: '\\\r\n' }
        },
        {
          key: 'cmd+enter',
          command: 'workbench.action.terminal.sendSequence',
          when: 'terminalFocus',
          args: { text: '\\\r\n' }
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
      ])
    )

    await expect(
      shouldPromptForTerminalSetup({
        env: { TERM_PROGRAM: 'vscode' } as NodeJS.ProcessEnv,
        fileOps: { readFile: readComplete }
      })
    ).resolves.toBe(false)
  })

  it('suppresses terminal setup prompts inside SSH sessions', async () => {
    await expect(
      shouldPromptForTerminalSetup({
        env: { SSH_CONNECTION: '1 2 3 4', TERM_PROGRAM: 'vscode' } as NodeJS.ProcessEnv
      })
    ).resolves.toBe(false)
  })
})
