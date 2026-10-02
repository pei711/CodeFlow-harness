import { activeColorTier, oscColor, type StdinProps, useStdin } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import type { ChatStreamRpcClient } from './app/chatStream.js'
import type { TuiRpcClient } from './tuiRpcClient.js'

import { GatewayProvider } from './app/gatewayContext.js'
import { $uiState, $uiTheme, applyTerminalBackground } from './app/uiStore.js'
import { useMainApp } from './app/useMainApp.js'
import { AppLayout } from './components/appLayout.js'
import { cursorColorHex } from './theme.js'

// 向终端查询真实背景色（OSC 11），避免仅根据 TERM_PROGRAM 猜测明暗模式。
// 响应会从 stdin 返回，但 Ink 的按键解析器会识别 OSC 响应并交给查询器，
// 不会将其作为输入发出，因此 `rgb:...` 内容不会泄漏到编辑框。若终端忽略
// 查询，flush() 发出的 DA1 哨兵会限制等待时间，并继续采用环境变量决定的配色。
function useTerminalBackgroundProbe() {
  // 这里需要 `as StdinProps`：useStdin() 的推导返回类型跨包后会将 `querier`
  // 收窄为 `unknown`，因为其内部 `.js` 对 StdinContext 的导入路径与公开的
  // `.ts` 导出路径解析结果不同；公开的 StdinProps 才包含正确的
  // `TerminalQuerier | null` 类型。
  const { querier } = useStdin() as StdinProps

  useEffect(() => {
    if (!querier) {
      return
    }

    let cancelled = false

    void Promise.all([querier.send(oscColor(11)), querier.flush()]).then(([reply]) => {
      if (!cancelled && reply) {
        applyTerminalBackground(reply.data)
      }
    })

    return () => {
      cancelled = true
    }
  }, [querier])
}

// 使用主题主色绘制终端硬件光标。TTY 聚焦时，输入光标是由
// useDeclaredCursor 定位的终端自身光标，只能通过 OSC 12 改色，文本 SGR
// 无法影响它。OSC 11 探测导致配色切换时会重新发送颜色（亮色 #B87900，
// 暗色 #fbe23f）；卸载时通过 OSC 112 重置，resetTerminalModes() 也会重置，
// 确保信号或崩溃退出后恢复用户原有的光标颜色。
const setCursorColorSeq = (hex: string) => `]12;${hex}`
const RESET_CURSOR_COLOR_SEQ = ']112'

function useHardwareCursorColor() {
  const hex = cursorColorHex(useStore($uiTheme))

  useEffect(() => {
    // 第 0 级表示已禁用颜色（NO_COLOR 或 FORCE_COLOR=0），不修改光标。
    if (activeColorTier() === 0) {
      return
    }

    process.stdout.write(setCursorColorSeq(hex))
  }, [hex])

  // 仅在卸载时重置，而不是每次重新发送前都重置，避免配色切换期间闪现默认光标色。
  useEffect(
    () => () => {
      if (activeColorTier() !== 0) {
        process.stdout.write(RESET_CURSOR_COLOR_SEQ)
      }
    },
    []
  )
}

export function App({ gw, rpcClient }: { gw: TuiRpcClient; rpcClient?: ChatStreamRpcClient }) {
  const { appActions, appComposer, appProgress, appStatus, appTranscript, gateway } = useMainApp(gw, rpcClient)
  const { mouseTracking } = useStore($uiState)

  useTerminalBackgroundProbe()
  useHardwareCursorColor()

  return (
    <GatewayProvider value={gateway}>
      <AppLayout
        actions={appActions}
        composer={appComposer}
        mouseTracking={mouseTracking}
        progress={appProgress}
        status={appStatus}
        transcript={appTranscript}
      />
    </GatewayProvider>
  )
}
