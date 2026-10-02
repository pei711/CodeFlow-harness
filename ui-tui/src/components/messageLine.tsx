// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { Ansi, Box, NoSelect, Text } from '@hermes/ink'
import { memo, useState } from 'react'

import type { Theme } from '../theme.js'
import type { ActiveTool, DetailsMode, Msg, SectionVisibility } from '../types.js'

import { LONG_MSG } from '../config/limits.js'
import { sectionMode } from '../domain/details.js'
import { userDisplay } from '../domain/messages.js'
import { ROLE } from '../domain/roles.js'
import { transcriptBodyWidth, transcriptGutterWidth } from '../lib/inputMetrics.js'
import {
  boundedHistoryRenderText,
  boundedLiveRenderText,
  compactPreview,
  hasAnsi,
  isPasteBackedText,
  stripAnsi
} from '../lib/text.js'
import { Md } from './markdown.js'
import { StreamingMd } from './streamingMarkdown.js'
import { ToolTrail } from './thinking.js'
import { TodoPanel } from './todoPanel.js'

// 较长系统消息（如系统提示词）的折叠阈值。
const SYSTEM_COLLAPSE_CHARS = 400

export const MessageLine = memo(function MessageLine({
  cols,
  compact,
  detailsMode = 'collapsed',
  detailsModeCommandOverride = false,
  isStreaming = false,
  limitHistoryRender = false,
  msg,
  sections,
  t,
  tools = []
}: MessageLineProps) {
  // 分区级设置优先于全局模式，因此只解析一次这里可能使用的各分区，并且仅按
  // 实际承载内容的分区控制可见性，不能依赖全局模式。`trail` 消息向工具调用和
  // 活动区提供内容；带 thinking/tools 元数据的助手消息向思考区和工具调用区
  // 提供内容。若对所有分区统一设门槛，默认展开的 `thinking` 会在仅隐藏
  // `tools` 时保留空壳，这正是此前出现空 Box 的原因。
  const thinkingMode = sectionMode('thinking', detailsMode, sections, detailsModeCommandOverride)
  const toolsMode = sectionMode('tools', detailsMode, sections, detailsModeCommandOverride)
  const activityMode = sectionMode('activity', detailsMode, sections, detailsModeCommandOverride)
  const thinking = msg.thinking?.trim() ?? ''

  // 较长系统消息的折叠开关。
  const systemIsLong = msg.role === 'system' && msg.text.length > SYSTEM_COLLAPSE_CHARS
  const [systemOpen, setSystemOpen] = useState(false)

  if (msg.kind === 'trail' && msg.todos?.length) {
    return (
      <TodoPanel
        defaultCollapsed={msg.todoCollapsedByDefault}
        incomplete={msg.todoIncomplete}
        t={t}
        todos={msg.todos}
      />
    )
  }

  if (msg.kind === 'trail' && (msg.tools?.length || tools.length || thinking)) {
    return thinkingMode !== 'hidden' || toolsMode !== 'hidden' || activityMode !== 'hidden' ? (
      <Box flexDirection="column">
        <ToolTrail
          commandOverride={detailsModeCommandOverride}
          detailsMode={detailsMode}
          reasoning={thinking}
          reasoningTokens={msg.thinkingTokens}
          sections={sections}
          t={t}
          tools={tools}
          toolTokens={msg.toolTokens}
          trail={msg.tools ?? []}
        />
      </Box>
    ) : null
  }

  if (msg.role === 'tool') {
    const maxChars = Math.max(24, cols - 14)
    const stripped = hasAnsi(msg.text) ? stripAnsi(msg.text) : msg.text
    const preview = compactPreview(stripped, maxChars) || '(empty tool result)'

    return (
      <Box alignSelf="flex-start" marginLeft={3}>
        <Text color={t.color.muted}>· </Text>
        {hasAnsi(msg.text) ? (
          <Text wrap="truncate-end">
            <Ansi>{msg.text}</Ansi>
          </Text>
        ) : (
          <Text color={t.color.muted} wrap="truncate-end">
            {preview}
          </Text>
        )}
      </Box>
    )
  }

  const { body, glyph, prefix } = ROLE[msg.role](t)
  const gutterWidth = transcriptGutterWidth(msg.role, t.brand.prompt)

  const showDetails =
    (toolsMode !== 'hidden' && Boolean(msg.tools?.length)) || (thinkingMode !== 'hidden' && Boolean(thinking))

  const content = (() => {
    if (msg.kind === 'slash') {
      return <Text color={t.color.muted}>{msg.text}</Text>
    }

    // ── 可折叠的较长系统消息（系统提示词、AGENTS.md 等）──
    // 必须先于 hasAnsi 检查：后端系统消息含有 Rich 标记转义码，否则会进入
    // <Ansi> 的完整渲染路径。
    if (systemIsLong) {
      const firstLine = (msg.text.split('\n')[0] ?? '').trim().slice(0, 120) || '(system message)'

      return (
        <Box flexDirection="column">
          <Box onClick={() => setSystemOpen(v => !v)}>
            <Text color={t.color.accent}>{systemOpen ? '▾ ' : '▸ '}</Text>
            <Text color={t.color.muted}>{firstLine}</Text>
            <Text color={t.color.muted} dimColor>
              {' — '}
              {msg.text.length.toLocaleString()} chars
            </Text>
          </Box>
          {systemOpen && <Ansi>{msg.text}</Ansi>}
        </Box>
      )
    }

    if (msg.role !== 'user' && hasAnsi(msg.text)) {
      return <Ansi>{msg.text}</Ansi>
    }

    if (msg.role === 'assistant') {
      return isStreaming ? (
        // 增量 Markdown 在最后一个稳定块边界切分，因此每次增量仅重新词法分析
        // 尚未完成的尾部；成本模型见 streamingMarkdown.tsx。
        <StreamingMd compact={compact} t={t} text={boundedLiveRenderText(msg.text)} />
      ) : (
        <Md compact={compact} t={t} text={limitHistoryRender ? boundedHistoryRenderText(msg.text) : msg.text} />
      )
    }

    if (msg.role === 'user' && msg.text.length > LONG_MSG && isPasteBackedText(msg.text)) {
      const [head, ...rest] = userDisplay(msg.text).split('[long message]')

      return (
        <Text color={body}>
          {head}
          <Text color={t.color.muted} dimColor>
            [long message]
          </Text>
          {rest.join('')}
        </Text>
      )
    }

    return <Text {...(body ? { color: body } : {})}>{msg.text}</Text>
  })()

  // pushInlineDiffSegment 在叙述片段之间产生的差异片段两侧都需要空行，
  // 避免补丁紧贴前后的正文。
  const isDiffSegment = msg.kind === 'diff'

  return (
    <Box
      backgroundColor={msg.role === 'user' ? t.color.surfaceRaised : undefined}
      flexDirection="column"
      marginBottom={msg.role === 'user' || isDiffSegment ? 1 : 0}
      marginTop={msg.role === 'user' || msg.kind === 'slash' || isDiffSegment ? 1 : 0}
      paddingX={msg.role === 'user' ? 1 : 0}
      width={msg.role === 'user' ? '100%' : undefined}
    >
      {showDetails && (
        <Box flexDirection="column" marginBottom={1}>
          <ToolTrail
            commandOverride={detailsModeCommandOverride}
            detailsMode={detailsMode}
            reasoning={thinking}
            reasoningTokens={msg.thinkingTokens}
            sections={sections}
            t={t}
            toolTokens={msg.toolTokens}
            trail={msg.tools}
          />
        </Box>
      )}

      <Box>
        <NoSelect flexShrink={0} fromLeftEdge width={gutterWidth}>
          <Text bold={msg.role === 'user'} color={prefix}>
            {glyph}{' '}
          </Text>
        </NoSelect>

        <Box width={transcriptBodyWidth(cols, msg.role, t.brand.prompt)}>{content}</Box>
      </Box>
    </Box>
  )
})

interface MessageLineProps {
  cols: number
  compact?: boolean
  detailsMode?: DetailsMode
  detailsModeCommandOverride?: boolean
  isStreaming?: boolean
  limitHistoryRender?: boolean
  msg: Msg
  sections?: SectionVisibility
  t: Theme
  tools?: ActiveTool[]
}
