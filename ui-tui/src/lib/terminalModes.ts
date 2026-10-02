// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { writeSync } from 'node:fs'

export const TERMINAL_MODE_RESET =
  "\x1b[0'z" + // DEC 定位器报告
  "\x1b[0'{" + // 可选择的定位器事件
  '\x1b[?2029l' + // 被动鼠标模式
  '\x1b[?1016l' + // SGR 像素鼠标模式
  '\x1b[?1015l' + // urxvt 十进制鼠标模式
  '\x1b[?1006l' + // SGR 鼠标模式
  '\x1b[?1005l' + // UTF-8 扩展鼠标模式
  '\x1b[?1003l' + // 任意移动鼠标模式
  '\x1b[?1002l' + // 按钮移动鼠标模式
  '\x1b[?1001l' + // 高亮鼠标模式
  '\x1b[?1000l' + // 点击鼠标模式
  '\x1b[?9l' + // X10 鼠标模式
  '\x1b[?1004l' + // 焦点事件
  '\x1b[?2004l' + // 括号粘贴模式
  '\x1b[?1049l' + // 备用屏幕
  '\x1b[<u' + // Kitty 键盘模式
  '\x1b[>4m' + // modifyOtherKeys 模式
  '\x1b[0m' + // 文本属性
  '\x1b]112\x07' + // 重置光标颜色（OSC 112），参见 useHardwareCursorColor
  '\x1b[?25h' // 显示光标

type ResettableStream = Pick<NodeJS.WriteStream, 'isTTY' | 'write'> & {
  fd?: number
}

export function resetTerminalModes(stream: ResettableStream = process.stdout): boolean {
  if (!stream.isTTY) {
    return false
  }

  const fd = typeof stream.fd === 'number' ? stream.fd : stream === process.stdout ? 1 : undefined

  if (fd !== undefined) {
    try {
      writeSync(fd, TERMINAL_MODE_RESET)

      return true
    } catch {
      // 对模拟或非常规 TTY 流回退到 stream.write。
    }
  }

  try {
    stream.write(TERMINAL_MODE_RESET)

    return true
  } catch {
    return false
  }
}
