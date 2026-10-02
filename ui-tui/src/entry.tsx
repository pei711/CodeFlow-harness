// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

// 此处删除原 shebang。我们从 Python 通过子进程运行 entry.js（见 codeflow/cli/tui_commands.py），
// 而不是直接调用 ./entry.js。若 TUI Shell 下出现 Node 22 内存溢出，按
// 04-node-version-impact.md 的首选回退重新加入
// `#!/usr/bin/env -S node --max-old-space-size=8192 --expose-gc`；Node 24 无法修复该问题。

// 必须首先导入。在 Chalk/supports-color 初始化前设置 FORCE_COLOR，应用所有
// CODEFLOW_TUI_COLOR / NO_COLOR 覆盖。
import './lib/colorTier.js'

import type { FrameEvent } from '@hermes/ink'

import { activeColorTier } from '@hermes/ink'

import { cleanupClipboardImages, cleanupStaleClipboardImages } from './lib/clipboard.js'
import { setupGracefulExit } from './lib/gracefulExit.js'
import { formatBytes, type HeapDumpResult, performHeapDump } from './lib/memory.js'
import { type MemorySnapshot, startMemoryMonitor } from './lib/memoryMonitor.js'
import { renderColorPreview, renderColorSwatches } from './lib/printColors.js'
import { resetTerminalModes } from './lib/terminalModes.js'
import { DEFAULT_THEME } from './theme.js'
import { TuiRpcClient } from './tuiRpcClient.js'

cleanupStaleClipboardImages()
process.on('exit', cleanupClipboardImages)

// `codeflow --print-colors` 是无 IPC 诊断：以色块输出解析后的调色板并退出。在 TTY 防护前运行，
// 因此管道中也能工作，并遵守所有 --color / CODEFLOW_TUI_COLOR 覆盖。
if (process.env.CODEFLOW_TUI_PRINT_COLORS === '1') {
  process.stdout.write(renderColorSwatches(DEFAULT_THEME, activeColorTier()))
  process.exit(0)
}

// `codeflow --preview-colors` 在真实 UI 上下文（横幅、状态栏、补全行、差异行）中渲染令牌，
// 便于直观检查前景/背景配对和横幅渐变，而不只是孤立色块。
if (process.env.CODEFLOW_TUI_COLOR_PREVIEW === '1') {
  process.stdout.write(renderColorPreview(DEFAULT_THEME, activeColorTier()))
  process.exit(0)
}

if (!process.stdin.isTTY) {
  console.log('codeflow-tui: no TTY')
  process.exit(0)
}

// 从干净状态开始。若先前 TUI 崩溃或被 kill -9，终端标签页可能仍启用鼠标、焦点或粘贴模式。
resetTerminalModes()

// `codeflow --check` 是无 IPC 冒烟路径；我们要验证的是导入链和终端重置成功。套接字传输使
// `gw.start()`（约第 53 行）通过 Unix 套接字发送真实 `system.hello` RPC，因此文件底部旧
// CODEFLOW_TUI_CHECK 处理器对检查路径不可达：无套接字时 gw.start() 先拒绝，而下方
// CODEFLOW_RPC_SOCKET 防护原本会以状态 2 退出，并对实际上由父进程在 --check 模式启动的子进程
// 给出误导性“通过父进程启动”消息。此处短路，使 --check 保持纯导入冒烟。
if (process.env.CODEFLOW_TUI_CHECK === '1') {
  process.exit(0)
}

const socketPath = process.env.CODEFLOW_RPC_SOCKET

if (!socketPath) {
  process.stderr.write('codeflow-tui: CODEFLOW_RPC_SOCKET env var required; spawn via `codeflow` parent\n')
  process.exit(2)
}

const gw = new TuiRpcClient({ socketPath })

// 握手 `system.hello` 必须在第 2 阶段 RpcServer 的 5 秒超时内完成；任何失败都会拒绝并冒泡到
// setupGracefulExit 错误路径。
await gw.start()

const dumpNotice = (snap: MemorySnapshot, dump: HeapDumpResult | null) =>
  `codeflow-tui: ${snap.level} memory (${formatBytes(snap.heapUsed)}) — auto heap dump → ${dump?.heapPath ?? '(failed)'}\n`

setupGracefulExit({
  cleanups: [
    () => {
      resetTerminalModes()

      return gw.kill()
    }
  ],
  onError: (scope, err) => {
    const message = err instanceof Error ? `${err.name}: ${err.message}` : String(err)

    process.stderr.write(`codeflow-tui ${scope}: ${message.slice(0, 2000)}\n`)
  },
  onSignal: signal => {
    resetTerminalModes()
    process.stderr.write(`codeflow-tui: received ${signal}\n`)
  }
})

const stopMemoryMonitor = startMemoryMonitor({
  onCritical: (snap, dump) => {
    resetTerminalModes()
    process.stderr.write(dumpNotice(snap, dump))
    process.stderr.write('codeflow-tui: exiting to avoid OOM; restart to recover\n')
    process.exit(137)
  },
  onHigh: (snap, dump) => process.stderr.write(dumpNotice(snap, dump))
})

if (process.env.CODEFLOW_HEAPDUMP_ON_START === '1') {
  void performHeapDump('manual')
}

process.on('beforeExit', () => stopMemoryMonitor())

const [ink, { App }, { logFrameEvent }, { trackFrame }] = await Promise.all([
  import('@hermes/ink'),
  import('./app.js'),
  import('./lib/perfPane.js'),
  import('./lib/fpsStore.js')
])

// 两个使用方的环境开关关闭时均为 undefined；至少一个开启时才挂接 onFrame，使 Ink 默认跳过计时。
const onFrame =
  logFrameEvent || trackFrame
    ? (event: FrameEvent) => {
        logFrameEvent?.(event)
        trackFrame?.(event.durationMs)
      }
    : undefined

// `codeflow --check` 设置 CODEFLOW_TUI_CHECK=1，要求启动子进程，证明导入和桩初始化不抛错，然后在
// 不显示 Ink 的情况下以 0 退出，无需 TTY 交互。短暂等待，使桩网关通过 setTimeout(0) 延迟的
// `gateway.ready` 事件有机会触发，再退出。若导入链或桩初始化中任何内容同步抛错，Node 会在
// 到达此处前以非零状态退出；这正是 `codeflow --check` 旨在显示的冒烟失败信号。
if (process.env.CODEFLOW_TUI_CHECK === '1') {
  setTimeout(() => {
    try {
      void gw.kill()
    } finally {
      resetTerminalModes()
      process.exit(0)
    }
  }, 100)
} else {
  ink.render(<App gw={gw} rpcClient={gw.rpcClient} />, { exitOnCtrlC: false, onFrame })
}
