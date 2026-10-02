// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

// StreamingMd——用于正在生成的助手文本的增量 Markdown 渲染器。
//
// 朴素方案（渲染 <Md text={full}/>）会在每次流增量时重新标记整条消息。3KB 响应按每批
// 20 字符计算，需要完整重解析 150 次。
//
// 该方案在最后一个稳定顶层块边界（围栏代码块外的空行）处分割 `text`：
//   stablePrefix——传给内部 <Md>，按精确文本值记忆化。轮次期间前缀只单调增长，因此记忆键
//                  与上次渲染一致，React 复用缓存子树，无需重新标记。
//   unstableSuffix——正在生成的块。独立 <Md> 在每次增量时只重解析尾部，复杂度从总长度降为
//                    不稳定部分长度。
//
// 边界存入 Ref，只能前进，因此在 StrictMode 双重渲染下仍幂等。组件会在轮次间卸载：
// isStreaming 关闭后消息进入历史并直接由 <Md> 渲染，Ref 会自然重置。
//
// 布局：两个 <Md> 子树必须按列堆叠。messageLine.tsx 的父容器是默认
// `flexDirection: 'row'` 的 Box（Ink 默认值），直接返回两个 <Md> 同级项的 Fragment 会让它们
// 并排，造成“流式时两列文字混杂”缺陷。此处用 flexDirection="column" 的 Box 包装，将修复
// 限定在流式路径；非流式 <Md> 已返回自己的列 Box，单子项情况从未受影响。

import { Box } from '@hermes/ink'
import { memo, useRef } from 'react'

import type { Theme } from '../theme.js'

import { Md } from './markdown.js'

// 统计 `s` 到 `end` 范围内 ``` / ~~~ 及 `$$` / `\[…\]` 围栏切换次数。奇数表示当前位于
// 围栏块内；在此处分割前缀会使围栏孤立，让不稳定后缀渲染成损坏的 Markdown。数学围栏仅在
// 代码围栏关闭时切换，避免代码块中的数学示例 ` ```\n$$x$$\n``` ` 重复计数。自行开闭的
// `$$x$$` 行净切换为零，对应 `len >= 4` 加 `endsDollar`。
//
// 注意：这里刻意比 `markdown.tsx` 解析器更保守；后者遇到没有匹配关闭符的 `$$` 起始符会回退到
// 段落渲染。渲染器每次都看到完整文本，因此这样做安全；流式分块器不能。一旦块提交到单调稳定
// 前缀便会冻结，过早认定“这个 `$$` 只是正文”会永久提交段落渲染，而关闭符一流入就变错。
// 将所有未匹配 `$$` 起始符视为仍开启，可让边界停在其后，直到关闭符到达；若流结束，则由
// 非流式 `<Md>` 接管，此时渲染器回退会正确生效。
const fenceOpenAt = (s: string, end: number) => {
  let codeOpen = false
  let mathOpen = false
  let mathOpener: '$$' | '\\[' | null = null
  let i = 0

  while (i < end) {
    const nl = s.indexOf('\n', i)
    const lineEnd = nl < 0 || nl > end ? end : nl
    const line = s.slice(i, lineEnd).trim()

    if (/^(?:`{3,}|~{3,})/.test(line)) {
      codeOpen = !codeOpen
    } else if (!codeOpen) {
      if (!mathOpen && /^\$\$/.test(line)) {
        const isSingleLine = line.length >= 4 && /\$\$$/.test(line)

        if (!isSingleLine) {
          mathOpen = true
          mathOpener = '$$'
        }
      } else if (!mathOpen && /^\\\[/.test(line)) {
        const isSingleLine = /\\\]$/.test(line)

        if (!isSingleLine) {
          mathOpen = true
          mathOpener = '\\['
        }
      } else if (mathOpen && mathOpener === '$$' && /\$\$$/.test(line)) {
        mathOpen = false
        mathOpener = null
      } else if (mathOpen && mathOpener === '\\[' && /\\\]$/.test(line)) {
        mathOpen = false
        mathOpener = null
      }
    }

    if (nl < 0 || nl >= end) {
      break
    }

    i = nl + 1
  }

  return codeOpen || mathOpen
}

// 查找 `end` 前围栏代码块外最后一个 "\n\n" 边界。返回第二个换行后的索引，即下一块起点；
// 尚无安全边界时返回 -1。
export const findStableBoundary = (text: string) => {
  let idx = text.length

  while (idx > 0) {
    const boundary = text.lastIndexOf('\n\n', idx - 1)

    if (boundary < 0) {
      return -1
    }

    // 候选边界：稳定前缀终点为 boundary + 2，即下一块起点；检查到该位置的围栏平衡。
    const splitAt = boundary + 2

    if (!fenceOpenAt(text, splitAt)) {
      return splitAt
    }

    idx = boundary
  }

  return -1
}

export const StreamingMd = memo(function StreamingMd({ compact, t, text }: StreamingMdProps) {
  const stablePrefixRef = useRef('')

  // 若文本不再以已记录前缀开头则重置。此为防御性处理；正常情况下组件会在轮次间卸载，不会触发。
  if (!text.startsWith(stablePrefixRef.current)) {
    stablePrefixRef.current = ''
  }

  const boundary = findStableBoundary(text)

  // 前缀只前进不后退。边界计算每次查看完整文本；若返回索引大于此前值，则增长缓存前缀。
  // 单调增长使记忆键在各增量间稳定：相同字符串对应同一 <Md> 子树，无需重新渲染。
  if (boundary > stablePrefixRef.current.length) {
    stablePrefixRef.current = text.slice(0, boundary)
  }

  const stablePrefix = stablePrefixRef.current
  const unstableSuffix = text.slice(stablePrefix.length)

  if (!stablePrefix) {
    return <Md compact={compact} t={t} text={unstableSuffix} />
  }

  if (!unstableSuffix) {
    return <Md compact={compact} t={t} text={stablePrefix} />
  }

  return (
    <Box flexDirection="column">
      <Md compact={compact} t={t} text={stablePrefix} />
      <Md compact={compact} t={t} text={unstableSuffix} />
    </Box>
  )
})

interface StreamingMdProps {
  compact?: boolean
  t: Theme
  text: string
}
