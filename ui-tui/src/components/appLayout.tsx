// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { AlternateScreen, Box, NoSelect, ScrollBox, Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { Fragment, memo, useMemo, useRef } from 'react'

import type { AppLayoutProps } from '../app/interfaces.js'

import { $isBlocked, $overlayState, patchOverlayState } from '../app/overlayStore.js'
import { $uiState } from '../app/uiStore.js'
import { INLINE_MODE, SHOW_FPS } from '../config/env.js'
import { FULL_RENDER_TAIL_ITEMS } from '../config/limits.js'
import { PLACEHOLDER } from '../content/placeholders.js'
import {
  COMPOSER_PROMPT_GAP_WIDTH,
  composerPromptWidth,
  inputVisualHeight,
  stableComposerColumns
} from '../lib/inputMetrics.js'
import { PerfPane } from '../lib/perfPane.js'
import { fmtK } from '../lib/text.js'
import { AgentsOverlay } from './agentsOverlay.js'
import { GoodVibesHeart, StatusRule, StickyPromptTracker, TranscriptScrollbar } from './appChrome.js'
import { FloatingOverlays, PromptZone } from './appOverlays.js'
import { Panel, SessionPanel, StartupLoader } from './branding.js'
import { FpsOverlay } from './fpsOverlay.js'
import { HelpHint } from './helpHint.js'
import { MessageLine } from './messageLine.js'
import { QueuedMessages } from './queuedMessages.js'
import { LiveTodoPanel, StreamingAssistant } from './streamingAssistant.js'
import { TextInput, type TextInputMouseApi } from './textInput.js'

const PromptPrefix = memo(function PromptPrefix({
  bold = false,
  color,
  promptText,
  width
}: {
  bold?: boolean
  color: string
  promptText: string
  width: number
}) {
  const glyphWidth = Math.max(1, width - COMPOSER_PROMPT_GAP_WIDTH)

  return (
    <Box width={width}>
      <Box width={glyphWidth}>
        <Text bold={bold} color={color}>
          {promptText}
        </Text>
      </Box>
      <Box width={COMPOSER_PROMPT_GAP_WIDTH} />
    </Box>
  )
})

const TranscriptPane = memo(function TranscriptPane({
  actions,
  composer,
  progress,
  transcript
}: Pick<AppLayoutProps, 'actions' | 'composer' | 'progress' | 'transcript'>) {
  const ui = useStore($uiState)

  // LiveTodoPanel 作为最新用户消息行的子节点，使其在视觉上属于提示词并随滚动
  // 移动。为空时取 -1，此时 row.index === -1 恒为假，因此不会渲染。
  const lastUserIdx = useMemo(() => {
    const items = transcript.historyItems

    for (let i = items.length - 1; i >= 0; i--) {
      if (items[i].role === 'user') {
        return i
      }
    }

    return -1
  }, [transcript.historyItems])

  return (
    <>
      <ScrollBox
        flexDirection="column"
        flexGrow={1}
        flexShrink={1}
        onClick={(e: { cellIsBlank?: boolean }) => {
          if (e.cellIsBlank) {
            actions.clearSelection()
          }
        }}
        ref={transcript.scrollRef}
        stickyScroll
      >
        <Box flexDirection="column" paddingX={1}>
          {transcript.virtualHistory.topSpacer > 0 ? <Box height={transcript.virtualHistory.topSpacer} /> : null}

          {transcript.virtualRows.slice(transcript.virtualHistory.start, transcript.virtualHistory.end).map(row => (
            <Box flexDirection="column" key={row.key} ref={transcript.virtualHistory.measureRef(row.key)}>
              {row.msg.kind === 'intro' ? (
                <Box flexDirection="column" paddingTop={1}>
                  {row.msg.info ? (
                    <SessionPanel info={row.msg.info} sid={ui.sid} t={ui.theme} />
                  ) : (
                    <StartupLoader t={ui.theme} />
                  )}
                </Box>
              ) : row.msg.kind === 'panel' && row.msg.panelData ? (
                <Panel sections={row.msg.panelData.sections} t={ui.theme} title={row.msg.panelData.title} />
              ) : (
                <MessageLine
                  cols={composer.cols}
                  compact={ui.compact}
                  detailsMode={ui.detailsMode}
                  detailsModeCommandOverride={ui.detailsModeCommandOverride}
                  limitHistoryRender={row.index < transcript.historyItems.length - FULL_RENDER_TAIL_ITEMS}
                  msg={row.msg}
                  sections={ui.sections}
                  t={ui.theme}
                />
              )}

              {row.index === lastUserIdx && <LiveTodoPanel />}
            </Box>
          ))}

          {transcript.virtualHistory.bottomSpacer > 0 ? <Box height={transcript.virtualHistory.bottomSpacer} /> : null}

          <StreamingAssistant
            cols={composer.cols}
            compact={ui.compact}
            detailsMode={ui.detailsMode}
            detailsModeCommandOverride={ui.detailsModeCommandOverride}
            progress={progress}
            sections={ui.sections}
          />
        </Box>
      </ScrollBox>

      <NoSelect flexShrink={0} marginLeft={1}>
        <TranscriptScrollbar scrollRef={transcript.scrollRef} t={ui.theme} />
      </NoSelect>

      <StickyPromptTracker
        messages={transcript.historyItems}
        offsets={transcript.virtualHistory.offsets}
        onChange={actions.setStickyPrompt}
        scrollRef={transcript.scrollRef}
      />
    </>
  )
})

const ComposerPane = memo(function ComposerPane({
  actions,
  composer,
  status
}: Pick<AppLayoutProps, 'actions' | 'composer' | 'status'>) {
  const ui = useStore($uiState)
  const isBlocked = useStore($isBlocked)
  const sh = (composer.inputBuf[0] ?? composer.input).startsWith('!')
  const promptText = sh ? '$' : ui.theme.brand.prompt
  const promptWidth = composerPromptWidth(promptText)
  const promptBlank = ' '.repeat(promptWidth)
  const inputColumns = stableComposerColumns(composer.cols, promptWidth)
  const inputHeight = inputVisualHeight(composer.input, inputColumns)
  const inputMouseRef = useRef<null | TextInputMouseApi>(null)

  const captureInputDrag = (e: GutterMouseEvent) => {
    if (e.button !== 0) {
      return
    }

    e.stopImmediatePropagation?.()
    inputMouseRef.current?.startAtBeginning()
  }

  // 拖动原点与输入框左上角一致，因此扣除提示词单元格后，localRow/localCol
  // 可直接映射为 TextInput 坐标。
  const dragFromPromptRow = (e: GutterMouseEvent) => {
    if (e.button !== 0) {
      return
    }

    e.stopImmediatePropagation?.()
    inputMouseRef.current?.dragAt(e.localRow ?? 0, (e.localCol ?? 0) - promptWidth)
  }

  // 间隔行使用不同的垂直原点，只有列与父级输入框对齐。强制 row=0，避免垂直
  // 拖动将光标跳到错误的换行位置。
  const dragFromSpacer = (e: GutterMouseEvent) => {
    if (e.button !== 0) {
      return
    }

    e.stopImmediatePropagation?.()
    inputMouseRef.current?.dragAt(0, (e.localCol ?? 0) - promptWidth)
  }

  const endInputDrag = () => inputMouseRef.current?.end()

  return (
    <NoSelect
      flexDirection="column"
      flexShrink={0}
      fromLeftEdge
      onClick={(e: { cellIsBlank?: boolean }) => {
        if (e.cellIsBlank) {
          actions.clearSelection()
        }
      }}
      paddingX={1}
    >
      <QueuedMessages
        cols={composer.cols}
        queued={composer.queuedDisplay}
        queueEditIdx={composer.queueEditIdx}
        t={ui.theme}
      />

      {ui.bgTasks.size > 0 && (
        <Text color={ui.theme.color.muted}>
          {ui.bgTasks.size} background {ui.bgTasks.size === 1 ? 'task' : 'tasks'} running
        </Text>
      )}

      {status.showStickyPrompt ? (
        <Text color={ui.theme.color.muted} wrap="truncate-end">
          <Text color={ui.theme.color.label}>↳ </Text>

          {status.stickyPrompt}
        </Text>
      ) : (
        <Box height={1} onMouseDown={captureInputDrag} onMouseDrag={dragFromSpacer} onMouseUp={endInputDrag} />
      )}

      <StatusRulePane at="top" composer={composer} status={status} />

      {/* 阻塞浮层打开时输入行会卸载，使此容器高度收缩为 0。statusBar='bottom'
          时，StatusRule 兄弟节点会共享计算后的顶部位置，触发渲染器的零高度跳过
          逻辑（render-node-to-output siblingSharesY），导致容器及其绝对定位的
          FloatingOverlays 子节点一起被丢弃，弹窗随之消失。因此该布局至少保留
          一行高度，确保锚点容器始终渲染。 */}
      <Box
        flexDirection="column"
        marginTop={ui.statusBar === 'top' ? 0 : 1}
        minHeight={ui.statusBar === 'bottom' ? 1 : undefined}
        position="relative"
      >
        <FloatingOverlays
          cols={composer.cols}
          compIdx={composer.compIdx}
          completions={composer.completions}
          onModelSelect={actions.onModelSelect}
          onPickerDeleteActive={actions.deleteSessionWithFallback}
          onPickerSelect={actions.resumeById}
          pagerPageSize={composer.pagerPageSize}
        />

        {composer.input === '?' && !composer.inputBuf.length && <HelpHint t={ui.theme} />}

        {!isBlocked && (
          <>
            {composer.inputBuf.map((line, i) => (
              <Box key={i}>
                <Box width={promptWidth}>
                  {i === 0 ? (
                    <PromptPrefix color={ui.theme.color.muted} promptText={promptText} width={promptWidth} />
                  ) : (
                    <Text color={ui.theme.color.muted}>{promptBlank}</Text>
                  )}
                </Box>

                <Text color={ui.theme.color.text}>{line || ' '}</Text>
              </Box>
            ))}

            <Box
              backgroundColor={ui.theme.color.surface}
              borderColor={ui.theme.color.sessionBorder}
              borderStyle={composer.cols - promptWidth >= 24 ? 'round' : undefined}
              flexDirection="column"
              onMouseDown={captureInputDrag}
              onMouseDrag={dragFromPromptRow}
              onMouseUp={endInputDrag}
              paddingX={composer.cols - promptWidth >= 24 ? 1 : 0}
              position="relative"
              width={Math.max(1, composer.cols - 2)}
            >
              <Box>
                <Box width={promptWidth}>
                  {sh ? (
                    <PromptPrefix color={ui.theme.color.shellDollar} promptText={promptText} width={promptWidth} />
                  ) : composer.inputBuf.length ? (
                    <Text color={ui.theme.color.prompt}>{promptBlank}</Text>
                  ) : (
                    <PromptPrefix bold color={ui.theme.color.prompt} promptText={promptText} width={promptWidth} />
                  )}
                </Box>

                <Box flexGrow={0} flexShrink={0} height={inputHeight} width={inputColumns}>
                  <TextInput
                    columns={inputColumns}
                    mouseApiRef={inputMouseRef}
                    onChange={composer.updateInput}
                    onPaste={composer.handleTextPaste}
                    onSubmit={composer.submit}
                    placeholder={
                      composer.empty
                        ? PLACEHOLDER
                        : ui.busy
                          ? ui.escapeArmed
                            ? 'In Progress, press Ctrl+C again to force quit'
                            : 'Ctrl+C to interrupt…'
                          : ''
                    }
                    value={composer.input}
                  />
                </Box>
              </Box>

              <Box position="absolute" right={0}>
                <GoodVibesHeart t={ui.theme} tick={status.goodVibesTick} />
              </Box>
            </Box>
          </>
        )}
      </Box>

      {!composer.empty && !ui.sid && (
        <Text color={ui.theme.color.muted}>
          {ui.theme.brand.icon} {ui.status}
        </Text>
      )}

      <StatusRulePane at="bottom" composer={composer} status={status} />

      {!isBlocked && (
        <Box justifyContent="space-between" width={Math.max(1, composer.cols - 2)}>
          <Text color={ui.theme.color.muted}>
            {composer.cols >= 44 ? (
              <>
                <Text bold color={ui.theme.color.label}>
                  tab
                </Text>{' '}
                complete <Text color={ui.theme.color.border}>│</Text>{' '}
                <Text bold color={ui.theme.color.label}>
                  ctrl+c
                </Text>{' '}
                cancel <Text color={ui.theme.color.border}>│</Text>{' '}
              </>
            ) : null}
            <Text bold color={ui.theme.color.label}>
              /help
            </Text>{' '}
            commands
          </Text>
          {ui.busy && composer.cols >= 60 ? <Text color={ui.theme.color.muted}>ctrl+c stop</Text> : null}
        </Box>
      )}
    </NoSelect>
  )
})

const SessionHeader = memo(function SessionHeader({ cwdLabel }: { cwdLabel: string }) {
  const ui = useStore($uiState)
  const context = ui.usage.context_max
    ? `${fmtK(ui.usage.context_used ?? 0)} / ${fmtK(ui.usage.context_max)}`
    : ui.usage.total > 0
      ? `${fmtK(ui.usage.total)} tokens`
      : ''

  return (
    <NoSelect flexShrink={0} fromLeftEdge paddingX={1}>
      <Box flexGrow={1}>
        <Text color={ui.theme.color.muted} wrap="truncate-end">
          {cwdLabel}
        </Text>
      </Box>
      {context ? <Text color={ui.theme.color.label}>{context}</Text> : null}
    </NoSelect>
  )
})

const AgentsOverlayPane = memo(function AgentsOverlayPane() {
  const ui = useStore($uiState)
  const overlay = useStore($overlayState)

  return (
    <AgentsOverlay
      initialHistoryIndex={overlay.agentsInitialHistoryIndex}
      onClose={() => patchOverlayState({ agents: false, agentsInitialHistoryIndex: 0 })}
      t={ui.theme}
    />
  )
})

const StatusRulePane = memo(function StatusRulePane({
  at,
  composer,
  status
}: Pick<AppLayoutProps, 'composer' | 'status'> & { at: 'bottom' | 'top' }) {
  const ui = useStore($uiState)

  if (ui.statusBar !== at) {
    return null
  }

  return (
    <Box marginTop={at === 'top' ? 1 : 0}>
      <StatusRule
        bgCount={ui.bgTasks.size}
        busy={ui.busy}
        cols={composer.cols}
        cwdLabel={status.cwdLabel}
        model={ui.info?.model ?? ''}
        modelFast={ui.info?.fast || ui.info?.service_tier === 'priority'}
        modelReasoningEffort={ui.info?.reasoning_effort}
        sessionStartedAt={status.sessionStartedAt}
        showCost={ui.showCost}
        status={ui.status}
        statusColor={status.statusColor}
        t={ui.theme}
        turnStartedAt={status.turnStartedAt}
        usage={ui.usage}
      />
    </Box>
  )
})

export const AppLayout = memo(function AppLayout({
  actions,
  composer,
  mouseTracking,
  progress,
  status,
  transcript
}: AppLayoutProps) {
  const overlay = useStore($overlayState)
  const ui = useStore($uiState)

  // 行内模式跳过 AlternateScreen，使宿主终端原生回滚区能保留滚出顶部的行；
  // 编辑框和进度条通过普通纵向弹性布局保持锚定。
  const Shell = INLINE_MODE ? Fragment : AlternateScreen
  const shellProps = INLINE_MODE ? {} : { mouseTracking }

  return (
    <Shell {...shellProps}>
      <Box backgroundColor={ui.theme.color.surface} flexDirection="column" flexGrow={1}>
        {!overlay.agents && <SessionHeader cwdLabel={status.cwdLabel} />}

        <Box flexDirection="row" flexGrow={1}>
          {overlay.agents ? (
            <PerfPane id="agents">
              <AgentsOverlayPane />
            </PerfPane>
          ) : (
            <PerfPane id="transcript">
              <TranscriptPane actions={actions} composer={composer} progress={progress} transcript={transcript} />
            </PerfPane>
          )}
        </Box>

        {!overlay.agents && (
          <>
            <PerfPane id="prompt">
              <PromptZone
                cols={composer.cols}
                onClarifyAnswer={actions.answerClarify}
                onConfirmAnswer={actions.answerConfirm}
              />
            </PerfPane>

            <PerfPane id="composer">
              <ComposerPane actions={actions} composer={composer} status={status} />
            </PerfPane>

            {SHOW_FPS && (
              <Box flexShrink={0} justifyContent="flex-end" paddingRight={1}>
                <FpsOverlay t={ui.theme} />
              </Box>
            )}
          </>
        )}
      </Box>
    </Shell>
  )
})

type GutterMouseEvent = {
  button: number
  localCol?: number
  localRow?: number
  stopImmediatePropagation?: () => void
}
