import { accessSync, constants } from 'node:fs'
import { delimiter, join } from 'node:path'

/**
 * 未设置 $VISUAL 和 $EDITOR 时的编辑器回退链。与 prompt_toolkit 的
 * `Buffer.open_in_editor()` 选择逻辑一致，使传统 CLI 和 TUI 在同一机器上启动
 * 相同编辑器。
 */
const FALLBACKS = ['editor', 'nano', 'codeflow', 'vi', 'emacs']

const isExecutable = (path: string): boolean => {
  try {
    accessSync(path, constants.X_OK)

    return true
  } catch {
    return false
  }
}

/**
 * 解析编辑器调用参数，不包含文件参数。
 *
 *   1. $VISUAL/$EDITOR，按 shell 规则分词以支持 `EDITOR="code --wait"`；
 *   2. POSIX：$PATH 中首个可解析的 FALLBACKS 条目；
 *   3. Windows：`notepad.exe`；
 *   4. POSIX 最终回退值 `['vi']`。
 */
export const resolveEditor = (
  env: NodeJS.ProcessEnv = process.env,
  platform: NodeJS.Platform = process.platform
): string[] => {
  const explicit = env.VISUAL ?? env.EDITOR

  if (explicit?.trim()) {
    return explicit.trim().split(/\s+/)
  }

  if (platform === 'win32') {
    return ['notepad.exe']
  }

  const dirs = (env.PATH ?? '').split(delimiter).filter(Boolean)
  const found = FALLBACKS.flatMap(name => dirs.map(d => join(d, name))).find(isExecutable)

  return [found ?? 'vi']
}
