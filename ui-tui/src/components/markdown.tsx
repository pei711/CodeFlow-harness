// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { Box, Link, stringWidth, Text } from '@hermes/ink'
import { Fragment, memo, type ReactNode, useMemo } from 'react'

import type { Theme } from '../theme.js'

import { ensureEmojiPresentation } from '../lib/emoji.js'
import { normalizeExternalUrl, urlSlugTitleLabel, useLinkTitle } from '../lib/externalLink.js'
import { BOX_CLOSE, BOX_OPEN, texToUnicode } from '../lib/mathUnicode.js'
import { highlightLine, isHighlightable } from '../lib/syntax.js'

// `texToUnicode` 输出中的 `\boxed{X}` 区域由不可打印的 U+0001 / U+0002 哨兵标记。
// 按哨兵切分，并以 `inverse + bold` 渲染框内片段，使其像荧光笔一样叠加在父 `<Text>` 使用的
// 颜色上，即数学主题强调色。高亮内部首尾空格提供一个单元格的视觉边距，使高亮看起来像色块，
// 而不是紧贴文字。
const renderMath = (text: string): ReactNode => {
  if (!text.includes(BOX_OPEN)) {
    return text
  }

  const out: ReactNode[] = []
  let i = 0
  let key = 0

  while (i < text.length) {
    const start = text.indexOf(BOX_OPEN, i)

    if (start < 0) {
      out.push(text.slice(i))

      break
    }

    if (start > i) {
      out.push(text.slice(i, start))
    }

    const end = text.indexOf(BOX_CLOSE, start + 1)

    if (end < 0) {
      out.push(text.slice(start))

      break
    }

    out.push(
      <Text bold inverse key={key++}>
        {' '}
        {text.slice(start + 1, end)}{' '}
      </Text>
    )

    i = end + 1
  }

  return out
}

const FENCE_RE = /^\s*(`{3,}|~{3,})(.*)$/
const FENCE_CLOSE_RE = /^\s*(`{3,}|~{3,})\s*$/
const HR_RE = /^ {0,3}([-*_])(?:\s*\1){2,}\s*$/
const HEADING_RE = /^\s{0,3}(#{1,6})\s+(.*?)(?:\s+#+\s*)?$/
const SETEXT_RE = /^\s{0,3}(=+|-+)\s*$/
const FOOTNOTE_RE = /^\[\^([^\]]+)\]:\s*(.*)$/
const DEF_RE = /^\s*:\s+(.+)$/
const BULLET_RE = /^(\s*)[-+*]\s+(.*)$/
const TASK_RE = /^\[( |x|X)\]\s+(.*)$/
const NUMBERED_RE = /^(\s*)(\d+)[.)]\s+(.*)$/
const QUOTE_RE = /^\s*(?:>\s*)+/
const TABLE_DIVIDER_CELL_RE = /^:?-{3,}:?$/
const MD_URL_RE = '((?:[^\\s()]|\\([^\\s()]*\\))+?)'

// 块级数学起始符：TeX 的 `$$ ... $$` 和 LaTeX 的 `\[ ... \]`。只有 `$$` / `\[` 位于
// 去空白后行首时才匹配。过去 startsWith('$$') 会在 `$$x+y$$ followed by more` 等正文上
// 触发，打开永不关闭的块，因为关闭扫描循环看不到同一行末尾的 `$$`。
const MATH_BLOCK_OPEN_RE = /^\s*(\$\$|\\\[)(.*)$/
const MATH_BLOCK_CLOSE_DOLLAR_RE = /^(.*?)\$\$\s*$/
const MATH_BLOCK_CLOSE_BRACKET_RE = /^(.*?)\\\]\s*$/

export const MEDIA_LINE_RE = /^\s*[`"']?MEDIA:\s*(\S+?)[`"']?\s*$/
export const AUDIO_DIRECTIVE_RE = /^\s*\[\[audio_as_voice\]\]\s*$/

// 行内 Markdown 令牌按优先级排列。外层正则在每个位置选择最左匹配，同位时优先较早分支，
// 因此 `**` 必须先于 `*`，`__` 先于 `_`。每个模式拥有独立捕获组，MdInline 按命中组分发。
//
// 下标（`~x~`）限制为短字母数字串，避免 Kimi/Qwen/GLM 输出的颜文字式正文
// `thing ~! more ~?` 把第一个 `~` 与行内下一个配对，并将中间文本吞成暗色 `_` 前缀片段。
//
// 行内数学（`$x$` 和 `\(x\)`）在同一起点优先于强调，因为正则分支最左优先；第 N 列以美元符
// 包围的片段优先于第 N+1 列的 `*`，使 `$P=a*b*c$` 渲染为数学，而不是把 `*b*` 误作斜体。
// 最少一个字符且分隔符旁不得有空格的规则，可避免吞掉 `$5 to $10` 等货币正文。
export const INLINE_RE = new RegExp(
  [
    `!\\[(.*?)\\]\\(${MD_URL_RE}\\)`, // 1、2：图片
    `\\[(.+?)\\]\\(${MD_URL_RE}\\)`, // 3、4：链接
    `<((?:https?:\\/\\/|mailto:)[^>\\s]+|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,})>`, // 5：自动链接
    `~~(.+?)~~`, // 6：删除线
    `\`([^\\\`]+)\``, // 7：代码
    `\\*\\*(.+?)\\*\\*`, // 8：星号粗体
    `(?<!\\w)__(.+?)__(?!\\w)`, // 9：下划线粗体
    `\\*(.+?)\\*`, // 10：星号斜体
    `(?<!\\w)_(.+?)_(?!\\w)`, // 11：下划线斜体
    `==(.+?)==`, // 12：高亮
    `\\[\\^([^\\]]+)\\]`, // 13：脚注引用
    `\\^([^^\\s][^^]*?)\\^`, // 14：上标
    `~([A-Za-z0-9]{1,8})~`, // 15：下标
    `(https?:\\/\\/[^\\s<]+)`, // 16：裸 URL；单独包装以拥有自己的捕获组，
    // 否则下方数学片段会落入 m[16]，MdInline 分发器会将其当作裸 URL 并渲染为自动链接。
    `(?<!\\$)\\$([^\\s$](?:[^$\\n]*?[^\\s$])?)\\$(?!\\$)`, // 17：行内数学 $...$
    `\\\\\\(([^\\n]+?)\\\\\\)` // 18：行内数学 \(...\)
  ].join('|'),
  'g'
)

const indentDepth = (s: string) => Math.floor(s.replace(/\t/g, '  ').length / 2)

const splitRow = (row: string) =>
  row
    .trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map(c => c.trim())

const isTableDivider = (row: string) => {
  const cells = splitRow(row)

  return cells.length > 1 && cells.every(c => TABLE_DIVIDER_CELL_RE.test(c))
}

const autolinkUrl = (raw: string) =>
  raw.startsWith('mailto:') || raw.startsWith('http') || !raw.includes('@') ? raw : `mailto:${raw}`

const defaultLinkLabel = (url: string) =>
  url.startsWith('mailto:') ? url.replace(/^mailto:/, '') : /^https?:\/\//i.test(url) ? urlSlugTitleLabel(url) : url

const pickFallbackLabel = (label: string | undefined, target: string): string | undefined => {
  const trimmed = label?.trim()

  if (!trimmed) {
    return undefined
  }

  return normalizeExternalUrl(trimmed) === target ? undefined : trimmed
}

interface ResolvedLinkProps {
  fallbackLabel?: string
  t: Theme
  url: string
}

function ResolvedLink({ fallbackLabel, t, url }: ResolvedLinkProps) {
  const fetched = useLinkTitle(url)
  const display = fetched || fallbackLabel || defaultLinkLabel(url)

  return (
    <Link url={url}>
      <Text color={t.color.accent} underline>
        {display}
      </Text>
    </Link>
  )
}

const renderResolvedLink = (k: number, t: Theme, rawUrl: string, label?: string) => {
  const target = normalizeExternalUrl(rawUrl)

  return <ResolvedLink fallbackLabel={pickFallbackLabel(label, target)} key={k} t={t} url={target} />
}

export const stripInlineMarkup = (v: string) =>
  v
    .replace(/!\[(.*?)\]\(((?:[^\s()]|\([^\s()]*\))+?)\)/g, '[image: $1] $2')
    .replace(/\[(.+?)\]\(((?:[^\s()]|\([^\s()]*\))+?)\)/g, '$1')
    .replace(/<((?:https?:\/\/|mailto:)[^>\s]+|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})>/g, '$1')
    .replace(/~~(.+?)~~/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*(.+?)\*\*/g, '$1')
    .replace(/(?<!\w)__(.+?)__(?!\w)/g, '$1')
    .replace(/\*(.+?)\*/g, '$1')
    .replace(/(?<!\w)_(.+?)_(?!\w)/g, '$1')
    .replace(/==(.+?)==/g, '$1')
    .replace(/\[\^([^\]]+)\]/g, '[$1]')
    .replace(/\^([^^\s][^^]*?)\^/g, '^$1')
    .replace(/~([A-Za-z0-9]{1,8})~/g, '_$1')
    .replace(/(?<!\$)\$([^\s$](?:[^$\n]*?[^\s$])?)\$(?!\$)/g, '$1')
    .replace(/\\\(([^\n]+?)\\\)/g, '$1')

const renderTable = (k: number, rows: string[][], t: Theme) => {
  // 列宽按显示单元格而非 UTF-16 代码单元计算。中日韩字形和多数表情符号渲染为两个单元格，
  // 但 `String#length` 只计为一个，会使中日韩表格逐行错位。`stringWidth` 使用
  // Bun.stringWidth 快速路径和感知东亚字符宽度的回退，并在 @hermes/ink 中记忆化，返回真实单元格数。
  const cellWidth = (raw: string) => stringWidth(stripInlineMarkup(raw))

  const widths = rows[0]!.map((_, ci) => Math.max(...rows.map(r => cellWidth(r[ci] ?? ''))))

  // 标题下方使用细分隔线。若没有它，标题只是强调色文本，表格看起来像增加了间距的正文
  // （#15534）。刻意不用完整边框；列宽来自 `stringWidth(...)`，因此中日韩/表情符号表格中的
  // 分隔线与行内容保持同步。制表符式列间距在无方框外观时仍然清晰。
  const sep = widths.map(w => '─'.repeat(Math.max(1, w))).join('  ')

  return (
    <Box flexDirection="column" key={k} paddingLeft={2}>
      {rows.map((row, ri) => (
        <Fragment key={ri}>
          <Box>
            {widths.map((w, ci) => (
              <Text bold={ri === 0} color={ri === 0 ? t.color.accent : undefined} key={ci}>
                <MdInline t={t} text={row[ci] ?? ''} />
                {' '.repeat(Math.max(0, w - cellWidth(row[ci] ?? '')))}
                {ci < widths.length - 1 ? '  ' : ''}
              </Text>
            ))}
          </Box>
          {ri === 0 && rows.length > 1 ? (
            <Text color={t.color.muted} dimColor>
              {sep}
            </Text>
          ) : null}
        </Fragment>
      ))}
    </Box>
  )
}

function MdInline({ t, text }: { t: Theme; text: string }) {
  const parts: ReactNode[] = []

  let last = 0

  for (const m of text.matchAll(INLINE_RE)) {
    const i = m.index ?? 0
    const k = parts.length

    if (i > last) {
      parts.push(<Text key={k}>{text.slice(last, i)}</Text>)
    }

    if (m[1] && m[2]) {
      parts.push(
        <Text color={t.color.muted} key={parts.length}>
          [image: {m[1]}] {m[2]}
        </Text>
      )
    } else if (m[3] && m[4]) {
      parts.push(renderResolvedLink(parts.length, t, m[4], m[3]))
    } else if (m[5]) {
      parts.push(renderResolvedLink(parts.length, t, autolinkUrl(m[5]), m[5].replace(/^mailto:/, '')))
    } else if (m[6]) {
      parts.push(
        <Text key={parts.length} strikethrough>
          <MdInline t={t} text={m[6]} />
        </Text>
      )
    } else if (m[7]) {
      // 代码是唯一不递归的包装；行内 `code` 按定义应原样显示。让 MdInline 再处理会破坏正则示例和
      // Shell 片段。
      parts.push(
        <Text color={t.color.accent} dimColor key={parts.length}>
          {m[7]}
        </Text>
      )
    } else if (m[8] ?? m[9]) {
      // 递归粗体、斜体、删除线和高亮，使 `**bolded statement with $\mathbb{Z}$ math**` 内嵌的
      // `$...$` 数学及其他行内令牌真正渲染。否则内部内容会原样放入单个 `<Text bold>`，
      // 数学渲染器永远看不到它。
      parts.push(
        <Text bold key={parts.length}>
          <MdInline t={t} text={m[8] ?? m[9]!} />
        </Text>
      )
    } else if (m[10] ?? m[11]) {
      parts.push(
        <Text italic key={parts.length}>
          <MdInline t={t} text={m[10] ?? m[11]!} />
        </Text>
      )
    } else if (m[12]) {
      parts.push(
        <Text backgroundColor={t.color.diffAdded} color={t.color.diffAddedWord} key={parts.length}>
          <MdInline t={t} text={m[12]} />
        </Text>
      )
    } else if (m[13]) {
      parts.push(
        <Text color={t.color.muted} key={parts.length}>
          [{m[13]}]
        </Text>
      )
    } else if (m[14]) {
      parts.push(
        <Text color={t.color.muted} key={parts.length}>
          ^{m[14]}
        </Text>
      )
    } else if (m[15]) {
      parts.push(
        <Text color={t.color.muted} key={parts.length}>
          _{m[15]}
        </Text>
      )
    } else if (m[16]) {
      // 裸 URL：把尾随正文标点裁成同级文本节点，使 `see https://x.com/, which…` 中的逗号
      // 保持在链接之外。
      const url = m[16].replace(/[),.;:!?]+$/g, '')

      parts.push(renderResolvedLink(parts.length, t, url))

      if (url.length < m[16].length) {
        parts.push(<Text key={parts.length}>{m[16].slice(url.length)}</Text>)
      }
    } else if (m[17] ?? m[18]) {
      // 行内数学通过 `texToUnicode` 处理希腊字母、ℕℤℚℝ、运算符、上下标和分数，并以斜体强调色
      // 渲染。斜体用于消除歧义；链接使用强调色加下划线，若无斜体，读者无法区分数学
      // `\mathbb{R}` 和超链接词。`texToUnicode` 无法识别的内容原样保留，使陌生命令显示为原始
      // LaTeX 而不是消失。
      parts.push(
        <Text color={t.color.accent} italic key={parts.length}>
          {renderMath(texToUnicode(m[17] ?? m[18]!))}
        </Text>
      )
    }

    last = i + m[0].length
  }

  if (last < text.length) {
    parts.push(<Text key={parts.length}>{text.slice(last)}</Text>)
  }

  return <Text wrap="wrap-trim">{parts.length ? parts : text}</Text>
}

// 跨实例的已解析子项缓存：useMemo 的逐实例缓存会在重新挂载时消失，导致虚拟化重新解析每个
// 滚回视口的行。以主题为键的 WeakMap 丢弃过期调色板，内部 Map 受 LRU 上限约束。
const MD_CACHE_LIMIT = 512
const mdCache = new WeakMap<Theme, Map<string, ReactNode[]>>()

const cacheBucket = (t: Theme) => {
  const b = mdCache.get(t)

  if (b) {
    return b
  }

  const fresh = new Map<string, ReactNode[]>()
  mdCache.set(t, fresh)

  return fresh
}

const cacheGet = (b: Map<string, ReactNode[]>, key: string) => {
  const v = b.get(key)

  if (v) {
    b.delete(key)
    b.set(key, v)
  }

  return v
}

const cacheSet = (b: Map<string, ReactNode[]>, key: string, v: ReactNode[]) => {
  b.set(key, v)

  if (b.size > MD_CACHE_LIMIT) {
    b.delete(b.keys().next().value!)
  }
}

function MdImpl({ compact, t, text }: MdProps) {
  const nodes = useMemo(() => {
    const bucket = cacheBucket(t)
    const cacheKey = `${compact ? '1' : '0'}|${text}`
    const cached = cacheGet(bucket, cacheKey)

    if (cached) {
      return cached
    }

    const lines = ensureEmojiPresentation(text).split('\n')
    const nodes: ReactNode[] = []

    let prevKind: Kind = null
    let i = 0

    const gap = () => {
      if (nodes.length && prevKind !== 'blank') {
        nodes.push(<Text key={`gap-${nodes.length}`}> </Text>)
        prevKind = 'blank'
      }
    }

    const start = (kind: Exclude<Kind, null | 'blank'>) => {
      if (prevKind && prevKind !== 'blank' && prevKind !== kind) {
        gap()
      }

      prevKind = kind
    }

    while (i < lines.length) {
      const line = lines[i]!
      const key = nodes.length

      if (!line.trim()) {
        if (!compact) {
          gap()
        }

        i++

        continue
      }

      if (AUDIO_DIRECTIVE_RE.test(line)) {
        i++

        continue
      }

      const media = line.match(MEDIA_LINE_RE)?.[1]

      if (media) {
        start('paragraph')
        nodes.push(
          <Text color={t.color.muted} key={key} wrap="wrap-trim">
            {'▸ '}

            <Link url={/^(?:\/|[a-z]:[\\/])/i.test(media) ? `file://${media}` : media}>
              <Text color={t.color.accent} underline>
                {media}
              </Text>
            </Link>
          </Text>
        )
        i++

        continue
      }

      const fence = line.match(FENCE_RE)

      if (fence) {
        const char = fence[1]![0] as '`' | '~'
        const len = fence[1]!.length
        const lang = fence[2]!.trim().toLowerCase()
        const block: string[] = []

        for (i++; i < lines.length; i++) {
          const close = lines[i]!.match(FENCE_CLOSE_RE)?.[1]

          if (close && close[0] === char && close.length >= len) {
            break
          }

          block.push(lines[i]!)
        }

        if (i < lines.length) {
          i++
        }

        if (['md', 'markdown'].includes(lang)) {
          start('paragraph')
          nodes.push(<Md compact={compact} key={key} t={t} text={block.join('\n')} />)

          continue
        }

        start('code')

        const isDiff = lang === 'diff'
        const highlighted = !isDiff && isHighlightable(lang)

        nodes.push(
          <Box flexDirection="column" key={key} paddingLeft={2}>
            {lang && !isDiff && <Text color={t.color.muted}>{'─ ' + lang}</Text>}

            {block.map((l, j) => {
              if (highlighted) {
                return (
                  <Text key={j}>
                    {highlightLine(l, lang, t).map(([color, text], kk) =>
                      color ? (
                        <Text color={color} key={kk}>
                          {text}
                        </Text>
                      ) : (
                        <Text key={kk}>{text}</Text>
                      )
                    )}
                  </Text>
                )
              }

              const add = isDiff && l.startsWith('+')
              const del = isDiff && l.startsWith('-')
              const hunk = isDiff && l.startsWith('@@')

              return (
                <Text
                  backgroundColor={add ? t.color.diffAdded : del ? t.color.diffRemoved : undefined}
                  color={add ? t.color.diffAddedWord : del ? t.color.diffRemovedWord : hunk ? t.color.muted : undefined}
                  dimColor={isDiff && !add && !del && !hunk && l.startsWith(' ')}
                  key={j}
                >
                  {l}
                </Text>
              )
            })}
          </Box>
        )

        continue
      }

      const mathOpen = line.match(MATH_BLOCK_OPEN_RE)

      if (mathOpen) {
        const opener = mathOpen[1]!
        const closeRe = opener === '$$' ? MATH_BLOCK_CLOSE_DOLLAR_RE : MATH_BLOCK_CLOSE_BRACKET_RE
        const headRest = mathOpen[2] ?? ''
        const block: string[] = []

        // 单行块：`$$x + y = z$$` 或 `\[x\]`。捕获内部内容并立即输出块。否则关闭扫描循环会
        // 跳过第 `i` 行，把下一个起始符当成当前关闭符，吞掉中间所有段落。
        const sameLineClose = headRest.match(closeRe)

        if (sameLineClose) {
          const inner = sameLineClose[1]!.trim()

          start('code')
          nodes.push(
            <Box flexDirection="column" key={key} paddingLeft={2}>
              {inner ? <Text color={t.color.accent}>{renderMath(texToUnicode(inner))}</Text> : null}
            </Box>
          )
          i++

          continue
        }

        // 多行块：提交前向前扫描真实关闭符。若文档余下部分不存在关闭符，则把当前行渲染为段落，
        // 而不是吞掉后续全部内容。
        let closeIdx = -1

        for (let j = i + 1; j < lines.length; j++) {
          if (closeRe.test(lines[j]!)) {
            closeIdx = j

            break
          }
        }

        if (closeIdx < 0) {
          start('paragraph')
          nodes.push(<MdInline key={key} t={t} text={line} />)
          i++

          continue
        }

        if (headRest.trim()) {
          block.push(headRest)
        }

        for (let j = i + 1; j < closeIdx; j++) {
          block.push(lines[j]!)
        }

        const tail = lines[closeIdx]!.match(closeRe)![1]!.trimEnd()

        if (tail.trim()) {
          block.push(tail)
        }

        start('code')
        nodes.push(
          <Box flexDirection="column" key={key} paddingLeft={2}>
            {block.map((l, j) => (
              <Text color={t.color.accent} key={j}>
                {renderMath(texToUnicode(l))}
              </Text>
            ))}
          </Box>
        )
        i = closeIdx + 1

        continue
      }

      const heading = line.match(HEADING_RE)?.[2]

      if (heading) {
        start('heading')
        nodes.push(
          <Text bold color={t.color.heading} key={key} wrap="wrap-trim">
            <MdInline t={t} text={heading} />
          </Text>
        )
        i++

        continue
      }

      if (i + 1 < lines.length && SETEXT_RE.test(lines[i + 1]!)) {
        start('heading')
        nodes.push(
          <Text bold color={t.color.heading} key={key} wrap="wrap-trim">
            <MdInline t={t} text={line.trim()} />
          </Text>
        )
        i += 2

        continue
      }

      if (HR_RE.test(line)) {
        start('rule')
        nodes.push(
          <Text color={t.color.muted} key={key}>
            {'─'.repeat(36)}
          </Text>
        )
        i++

        continue
      }

      const footnote = line.match(FOOTNOTE_RE)

      if (footnote) {
        start('list')
        nodes.push(
          <Text color={t.color.muted} key={key} wrap="wrap-trim">
            [{footnote[1]}] <MdInline t={t} text={footnote[2] ?? ''} />
          </Text>
        )
        i++

        while (i < lines.length && /^\s{2,}\S/.test(lines[i]!)) {
          nodes.push(
            <Box key={`${key}-cont-${i}`} paddingLeft={2}>
              <Text color={t.color.muted} wrap="wrap-trim">
                <MdInline t={t} text={lines[i]!.trim()} />
              </Text>
            </Box>
          )
          i++
        }

        continue
      }

      if (i + 1 < lines.length && DEF_RE.test(lines[i + 1]!)) {
        start('list')
        nodes.push(
          <Text bold key={key} wrap="wrap-trim">
            {line.trim()}
          </Text>
        )
        i++

        while (i < lines.length) {
          const def = lines[i]!.match(DEF_RE)?.[1]

          if (!def) {
            break
          }

          nodes.push(
            <Text key={`${key}-def-${i}`} wrap="wrap-trim">
              <Text color={t.color.muted}> · </Text>
              <MdInline t={t} text={def} />
            </Text>
          )
          i++
        }

        continue
      }

      const bullet = line.match(BULLET_RE)

      if (bullet) {
        start('list')

        const task = bullet[2]!.match(TASK_RE)
        const marker = task ? (task[1]!.toLowerCase() === 'x' ? '☑' : '☐') : '•'

        nodes.push(
          <Box key={key} paddingLeft={indentDepth(bullet[1]!) * 2}>
            <Text wrap="wrap-trim">
              <Text color={t.color.muted}>{marker} </Text>
              <MdInline t={t} text={task ? task[2]! : bullet[2]!} />
            </Text>
          </Box>
        )
        i++

        continue
      }

      const numbered = line.match(NUMBERED_RE)

      if (numbered) {
        start('list')
        nodes.push(
          <Box key={key} paddingLeft={indentDepth(numbered[1]!) * 2}>
            <Text wrap="wrap-trim">
              <Text color={t.color.muted}>{numbered[2]}. </Text>
              <MdInline t={t} text={numbered[3]!} />
            </Text>
          </Box>
        )
        i++

        continue
      }

      if (QUOTE_RE.test(line)) {
        start('quote')

        const quoteLines: Array<{ depth: number; text: string }> = []

        while (i < lines.length && QUOTE_RE.test(lines[i]!)) {
          const prefix = lines[i]!.match(QUOTE_RE)?.[0] ?? ''

          quoteLines.push({ depth: (prefix.match(/>/g) ?? []).length, text: lines[i]!.slice(prefix.length) })
          i++
        }

        nodes.push(
          <Box flexDirection="column" key={key}>
            {quoteLines.map((ql, qi) => (
              <Box key={qi} paddingLeft={Math.max(0, ql.depth - 1) * 2}>
                <Text color={t.color.muted} wrap="wrap-trim">
                  │ <MdInline t={t} text={ql.text} />
                </Text>
              </Box>
            ))}
          </Box>
        )

        continue
      }

      if (line.includes('|') && i + 1 < lines.length && isTableDivider(lines[i + 1]!)) {
        start('table')

        const rows: string[][] = [splitRow(line)]

        for (i += 2; i < lines.length && lines[i]!.includes('|') && lines[i]!.trim(); i++) {
          rows.push(splitRow(lines[i]!))
        }

        nodes.push(renderTable(key, rows, t))

        continue
      }

      if (/^<\/?details\b/i.test(line)) {
        i++

        continue
      }

      const summary = line.match(/^<summary>(.*?)<\/summary>$/i)?.[1]

      if (summary) {
        start('paragraph')
        nodes.push(
          <Text color={t.color.muted} key={key} wrap="wrap-trim">
            ▶ {summary}
          </Text>
        )
        i++

        continue
      }

      if (/^<\/?[^>]+>$/.test(line.trim())) {
        start('paragraph')
        nodes.push(
          <Text color={t.color.muted} key={key} wrap="wrap-trim">
            {line.trim()}
          </Text>
        )
        i++

        continue
      }

      if (line.includes('|') && line.trim().startsWith('|')) {
        start('table')

        const rows: string[][] = []

        while (i < lines.length && lines[i]!.trim().startsWith('|')) {
          const row = lines[i]!.trim()

          if (!/^[|\s:-]+$/.test(row)) {
            rows.push(splitRow(row))
          }

          i++
        }

        if (rows.length) {
          nodes.push(renderTable(key, rows, t))
        }

        continue
      }

      start('paragraph')
      nodes.push(<MdInline key={key} t={t} text={line} />)
      i++
    }

    cacheSet(bucket, cacheKey, nodes)

    return nodes
  }, [compact, t, text])

  return <Box flexDirection="column">{nodes}</Box>
}

export const Md = memo(MdImpl)

type Kind = 'blank' | 'code' | 'heading' | 'list' | 'paragraph' | 'quote' | 'rule' | 'table' | null

interface MdProps {
  compact?: boolean
  t: Theme
  text: string
}
