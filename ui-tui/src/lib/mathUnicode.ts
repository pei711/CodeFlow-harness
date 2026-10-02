// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

// 尽力把 Markdown 渲染器捕获的行内/块级数学 LaTeX 转为 Unicode。终端无法排版 LaTeX，
// 但 Unicode 能覆盖模型实际输出的大部分内容：希腊字母、黑板粗体/哥特体/花体大写字母、
// 集合论和逻辑运算符、常用箭头、上下标，以及把 `\frac{a}{b}` 折叠为 `a/b`。
//
// 设计规则：
//   • 使用纯正则流水线。无法识别的内容原样保留，因此未知的 `\foo{bar}` 仍不会丢失。
//     真正的 LaTeX 解析器虽更准确，却会在不完整输入上抛错；终端用户更愿意看到原始命令，
//     而不是解析错误占位符。
//   • 命令按最长匹配优先排序，避免 `\le` 遮蔽 `\leq`、`\sub` 遮蔽 `\subseteq` 等。
//   • 每个命令后使用词边界前瞻 `(?![A-Za-z])`，避免虚构命令 `\pix` 被部分替换为 `π`。
//   • `\mathbb{X}`、`\mathcal{X}`、`\mathfrak{X}` 仅处理单字母参数；多字母
//     `\mathbb{NN}` 很少见，正确处理需要真正的解析器。
//   • 只有每个字符都有 Unicode 对应字符时才转换上下标。`^{n+1}` 等混合内容回退到原始
//     LaTeX，避免输出 `ⁿ+¹`；某些字体没有上标 `+`，显示效果反而比源码更差。

const SYMBOLS: Record<string, string> = {
  // 小写希腊字母
  '\\alpha': 'α',
  '\\beta': 'β',
  '\\gamma': 'γ',
  '\\delta': 'δ',
  '\\epsilon': 'ε',
  '\\varepsilon': 'ε',
  '\\zeta': 'ζ',
  '\\eta': 'η',
  '\\theta': 'θ',
  '\\vartheta': 'ϑ',
  '\\iota': 'ι',
  '\\kappa': 'κ',
  '\\lambda': 'λ',
  '\\mu': 'μ',
  '\\nu': 'ν',
  '\\xi': 'ξ',
  '\\pi': 'π',
  '\\varpi': 'ϖ',
  '\\rho': 'ρ',
  '\\varrho': 'ϱ',
  '\\sigma': 'σ',
  '\\varsigma': 'ς',
  '\\tau': 'τ',
  '\\upsilon': 'υ',
  '\\phi': 'φ',
  '\\varphi': 'φ',
  '\\chi': 'χ',
  '\\psi': 'ψ',
  '\\omega': 'ω',

  // 大写希腊字母
  '\\Gamma': 'Γ',
  '\\Delta': 'Δ',
  '\\Theta': 'Θ',
  '\\Lambda': 'Λ',
  '\\Xi': 'Ξ',
  '\\Pi': 'Π',
  '\\Sigma': 'Σ',
  '\\Upsilon': 'Υ',
  '\\Phi': 'Φ',
  '\\Psi': 'Ψ',
  '\\Omega': 'Ω',

  // 大型运算符
  '\\sum': '∑',
  '\\prod': '∏',
  '\\coprod': '∐',
  '\\int': '∫',
  '\\iint': '∬',
  '\\iiint': '∭',
  '\\oint': '∮',
  '\\bigcup': '⋃',
  '\\bigcap': '⋂',
  '\\bigvee': '⋁',
  '\\bigwedge': '⋀',
  '\\bigoplus': '⨁',
  '\\bigotimes': '⨂',

  // 微积分
  '\\partial': '∂',
  '\\nabla': '∇',
  '\\sqrt': '√',

  // 集合
  '\\emptyset': '∅',
  '\\varnothing': '∅',
  '\\infty': '∞',
  '\\in': '∈',
  '\\notin': '∉',
  '\\ni': '∋',
  '\\subset': '⊂',
  '\\supset': '⊃',
  '\\subseteq': '⊆',
  '\\supseteq': '⊇',
  '\\subsetneq': '⊊',
  '\\supsetneq': '⊋',
  '\\cup': '∪',
  '\\cap': '∩',
  '\\setminus': '∖',
  '\\complement': '∁',

  // 逻辑
  '\\forall': '∀',
  '\\exists': '∃',
  '\\nexists': '∄',
  '\\land': '∧',
  '\\lor': '∨',
  '\\lnot': '¬',
  '\\neg': '¬',
  '\\therefore': '∴',
  '\\because': '∵',

  // 关系运算符
  '\\le': '≤',
  '\\leq': '≤',
  '\\ge': '≥',
  '\\geq': '≥',
  '\\ne': '≠',
  '\\neq': '≠',
  '\\ll': '≪',
  '\\gg': '≫',
  '\\approx': '≈',
  '\\equiv': '≡',
  '\\cong': '≅',
  '\\sim': '∼',
  '\\simeq': '≃',
  '\\propto': '∝',
  '\\perp': '⊥',
  '\\parallel': '∥',
  '\\models': '⊨',
  '\\vdash': '⊢',
  '\\mid': '∣',
  '\\nmid': '∤',
  '\\divides': '∣',

  // 常用独立符号
  '\\blacksquare': '■',
  '\\square': '□',
  '\\Box': '□',
  '\\qed': '∎',
  '\\bigstar': '★',

  // 模运算——带参数的 `\pmod{p}` 形式在下方处理；裸 `\bmod` / `\mod` 命令只做文本替换。
  '\\bmod': 'mod',
  '\\mod': 'mod',

  // 括号/围栏（具名分隔符命令）；下方展开 `\left\X` / `\right\X` 后将它们留给符号阶段解析。
  '\\langle': '⟨',
  '\\rangle': '⟩',
  '\\lceil': '⌈',
  '\\rceil': '⌉',
  '\\lfloor': '⌊',
  '\\rfloor': '⌋',
  '\\|': '‖',

  // 箭头
  '\\to': '→',
  '\\rightarrow': '→',
  '\\leftarrow': '←',
  '\\leftrightarrow': '↔',
  '\\Rightarrow': '⇒',
  '\\Leftarrow': '⇐',
  '\\Leftrightarrow': '⇔',
  '\\implies': '⟹',
  '\\impliedby': '⟸',
  '\\iff': '⟺',
  '\\mapsto': '↦',
  '\\hookrightarrow': '↪',
  '\\hookleftarrow': '↩',
  '\\uparrow': '↑',
  '\\downarrow': '↓',
  '\\updownarrow': '↕',

  // 二元运算符
  '\\cdot': '⋅',
  '\\cdots': '⋯',
  '\\ldots': '…',
  '\\dots': '…',
  '\\dotsb': '…',
  '\\dotsc': '…',
  '\\vdots': '⋮',
  '\\ddots': '⋱',
  '\\times': '×',
  '\\div': '÷',
  '\\pm': '±',
  '\\mp': '∓',
  '\\circ': '∘',
  '\\bullet': '•',
  '\\star': '⋆',
  '\\ast': '∗',
  '\\oplus': '⊕',
  '\\ominus': '⊖',
  '\\otimes': '⊗',
  '\\odot': '⊙',
  '\\diamond': '⋄',
  '\\angle': '∠',
  '\\triangle': '△',

  // 间距——折叠为不同宽度的普通空格
  '\\,': ' ',
  '\\;': ' ',
  '\\:': ' ',
  '\\!': '',
  '\\ ': ' ',
  '\\quad': '  ',
  '\\qquad': '    ',

  // 函数（LaTeX 使用罗马体渲染；这里只保留名称）
  '\\sin': 'sin',
  '\\cos': 'cos',
  '\\tan': 'tan',
  '\\cot': 'cot',
  '\\sec': 'sec',
  '\\csc': 'csc',
  '\\arcsin': 'arcsin',
  '\\arccos': 'arccos',
  '\\arctan': 'arctan',
  '\\sinh': 'sinh',
  '\\cosh': 'cosh',
  '\\tanh': 'tanh',
  '\\log': 'log',
  '\\ln': 'ln',
  '\\exp': 'exp',
  '\\det': 'det',
  '\\dim': 'dim',
  '\\ker': 'ker',
  '\\lim': 'lim',
  '\\liminf': 'liminf',
  '\\limsup': 'limsup',
  '\\sup': 'sup',
  '\\inf': 'inf',
  '\\max': 'max',
  '\\min': 'min',
  '\\arg': 'arg',
  '\\gcd': 'gcd',

  // 转义字面量——模型偶尔会为了显示而输出这些内容
  '\\&': '&',
  '\\%': '%',
  '\\$': '$',
  '\\#': '#',
  '\\_': '_',
  '\\{': '{',
  '\\}': '}'
}

const BB: Record<string, string> = {
  A: '𝔸',
  B: '𝔹',
  C: 'ℂ',
  D: '𝔻',
  E: '𝔼',
  F: '𝔽',
  G: '𝔾',
  H: 'ℍ',
  I: '𝕀',
  J: '𝕁',
  K: '𝕂',
  L: '𝕃',
  M: '𝕄',
  N: 'ℕ',
  O: '𝕆',
  P: 'ℙ',
  Q: 'ℚ',
  R: 'ℝ',
  S: '𝕊',
  T: '𝕋',
  U: '𝕌',
  V: '𝕍',
  W: '𝕎',
  X: '𝕏',
  Y: '𝕐',
  Z: 'ℤ'
}

const CAL: Record<string, string> = {
  A: '𝒜',
  B: 'ℬ',
  C: '𝒞',
  D: '𝒟',
  E: 'ℰ',
  F: 'ℱ',
  G: '𝒢',
  H: 'ℋ',
  I: 'ℐ',
  J: '𝒥',
  K: '𝒦',
  L: 'ℒ',
  M: 'ℳ',
  N: '𝒩',
  O: '𝒪',
  P: '𝒫',
  Q: '𝒬',
  R: 'ℛ',
  S: '𝒮',
  T: '𝒯',
  U: '𝒰',
  V: '𝒱',
  W: '𝒲',
  X: '𝒳',
  Y: '𝒴',
  Z: '𝒵'
}

const FRAK: Record<string, string> = {
  A: '𝔄',
  B: '𝔅',
  C: 'ℭ',
  D: '𝔇',
  E: '𝔈',
  F: '𝔉',
  G: '𝔊',
  H: 'ℌ',
  I: 'ℑ',
  J: '𝔍',
  K: '𝔎',
  L: '𝔏',
  M: '𝔐',
  N: '𝔑',
  O: '𝔒',
  P: '𝔓',
  Q: '𝔔',
  R: 'ℜ',
  S: '𝔖',
  T: '𝔗',
  U: '𝔘',
  V: '𝔙',
  W: '𝔚',
  X: '𝔛',
  Y: '𝔜',
  Z: 'ℨ'
}

const SUPERSCRIPT: Record<string, string> = {
  '0': '⁰',
  '1': '¹',
  '2': '²',
  '3': '³',
  '4': '⁴',
  '5': '⁵',
  '6': '⁶',
  '7': '⁷',
  '8': '⁸',
  '9': '⁹',
  '+': '⁺',
  '-': '⁻',
  '=': '⁼',
  '(': '⁽',
  ')': '⁾',
  a: 'ᵃ',
  b: 'ᵇ',
  c: 'ᶜ',
  d: 'ᵈ',
  e: 'ᵉ',
  f: 'ᶠ',
  g: 'ᵍ',
  h: 'ʰ',
  i: 'ⁱ',
  j: 'ʲ',
  k: 'ᵏ',
  l: 'ˡ',
  m: 'ᵐ',
  n: 'ⁿ',
  o: 'ᵒ',
  p: 'ᵖ',
  r: 'ʳ',
  s: 'ˢ',
  t: 'ᵗ',
  u: 'ᵘ',
  v: 'ᵛ',
  w: 'ʷ',
  x: 'ˣ',
  y: 'ʸ',
  z: 'ᶻ'
}

const SUBSCRIPT: Record<string, string> = {
  '0': '₀',
  '1': '₁',
  '2': '₂',
  '3': '₃',
  '4': '₄',
  '5': '₅',
  '6': '₆',
  '7': '₇',
  '8': '₈',
  '9': '₉',
  '+': '₊',
  '-': '₋',
  '=': '₌',
  '(': '₍',
  ')': '₎',
  a: 'ₐ',
  e: 'ₑ',
  h: 'ₕ',
  i: 'ᵢ',
  j: 'ⱼ',
  k: 'ₖ',
  l: 'ₗ',
  m: 'ₘ',
  n: 'ₙ',
  o: 'ₒ',
  p: 'ₚ',
  r: 'ᵣ',
  s: 'ₛ',
  t: 'ₜ',
  u: 'ᵤ',
  v: 'ᵥ',
  x: 'ₓ'
}

// 哨兵控制字符用于在转换结果中标记 `\boxed` / `\fbox` 区域。渲染器按它们切分并应用
// 高亮样式；不需要高亮的使用方可用下方导出的 `BOX_RE` 移除。
export const BOX_OPEN = '\u0001'
export const BOX_CLOSE = '\u0002'
// eslint-disable-next-line no-control-regex -- BOX_OPEN/BOX_CLOSE 是数学 Unicode 渲染器使用的 U+0001/U+0002 哨兵；正则必须按字面匹配。
export const BOX_RE = /\u0001([^\u0001\u0002]*)\u0002/g

const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

// 预编译两类符号正则：以字母结尾的命令（`\pi`、`\sum`）需要 `(?![A-Za-z])` 前瞻，
// 避免部分匹配 `\pix` 或 `\summa`；以标点结尾的命令（`\{`、`\,`、`\|`）绝不能使用
// 该前瞻，否则 `\{p` 会因 `p` 是字母而拒绝替换。
//
// 每组内最长命令优先，使 `\leq` 优先于 `\le`。
const splitByEnding = (keys: string[]) => {
  const letter: string[] = []
  const punct: string[] = []

  for (const k of keys) {
    if (/[A-Za-z]$/.test(k)) {
      letter.push(k)
    } else {
      punct.push(k)
    }
  }

  return { letter, punct }
}

const buildAlt = (cmds: string[]) =>
  cmds
    .sort((a, b) => b.length - a.length)
    .map(escapeRe)
    .join('|')

const { letter: LETTER_CMDS, punct: PUNCT_CMDS } = splitByEnding(Object.keys(SYMBOLS))

const SYMBOL_LETTER_RE = new RegExp('(?:' + buildAlt(LETTER_CMDS) + ')(?![A-Za-z])', 'g')
const SYMBOL_PUNCT_RE = new RegExp('(?:' + buildAlt(PUNCT_CMDS) + ')', 'g')

const convertScript = (input: string, table: Record<string, string>, sigil: '^' | '_'): string => {
  let out = ''
  let allMapped = true

  for (const ch of input) {
    const mapped = table[ch]

    if (!mapped) {
      allMapped = false

      break
    }

    out += mapped
  }

  if (allMapped) {
    return out
  }

  // 回退：若正文是单个可见字符（如早先符号替换后的 `∞`），则不带花括号渲染；终端中的
  // `^∞` 远比 `^{∞}` 易读。未完全转换的多字符正文使用圆括号（`e^(iπ)`）而非花括号
  // （`e^{iπ}`），因为圆括号是普通标点，花括号看起来像未渲染的 LaTeX。
  const trimmed = input.trim()

  if ([...trimmed].length === 1) {
    return `${sigil}${trimmed}`
  }

  return `${sigil}(${trimmed})`
}

// 遍历字符串并解析 `{...}`，正确处理嵌套花括号。与 `\{[^{}]*\}` 正则不同，它能处理
// `\frac{|t|^{p-1}|P(t)|^p}{...}` 这种分子因上标自带花括号的情况。返回不含外层花括号的
// 内部内容，以及结束 `}` 后的偏移量；`start` 处没有配对花括号时返回 null。
const readBraced = (s: string, start: number): { content: string; end: number } | null => {
  if (s[start] !== '{') {
    return null
  }

  let depth = 1
  let i = start + 1

  while (i < s.length && depth > 0) {
    const c = s[i]

    // 跳过转义；正文中的 `\{` 和 `\}` 是字面花括号，不应改变花括号计数。
    if (c === '\\' && i + 1 < s.length) {
      i += 2

      continue
    }

    if (c === '{') {
      depth++
    } else if (c === '}') {
      depth--
    }

    if (depth > 0) {
      i++
    }
  }

  if (depth !== 0) {
    return null
  }

  return { content: s.slice(start + 1, i), end: i + 1 }
}

// 使用配对花括号解析替换每个 `\command{arg}`，使 `\boxed{x^{n+1}}` 在 `[^{}]*` 正则
// 失败的情况下仍可工作。`render` 回调接收已递归处理的内部内容，因此
// `\boxed{\boxed{x}}` 可从外到内干净解析。无后续 `{...}` 的未匹配 `\command` 原样保留。
const replaceBracedCommand = (input: string, command: string, render: (content: string) => string): string => {
  const cmdLen = command.length
  let out = ''
  let i = 0

  while (i < input.length) {
    const idx = input.indexOf(command, i)

    if (idx < 0) {
      out += input.slice(i)

      return out
    }

    const after = input[idx + cmdLen]

    if (after && /[A-Za-z]/.test(after)) {
      out += input.slice(i, idx + cmdLen)
      i = idx + cmdLen

      continue
    }

    out += input.slice(i, idx)

    let p = idx + cmdLen

    while (input[p] === ' ' || input[p] === '\t') {
      p++
    }

    const arg = readBraced(input, p)

    if (!arg) {
      out += input.slice(idx, p + 1)
      i = p + 1

      continue
    }

    out += render(replaceBracedCommand(arg.content, command, render))
    i = arg.end
  }

  return out
}

// 将每个 `\frac{num}{den}` 替换为 `num/den`，优先级需要时在任一侧加圆括号。递归天然处理
// 嵌套分数：`\frac{1}{\frac{1}{x}}` 会折叠为 `1/(1/x)`，因为决定是否加括号前先递归分母。
const replaceFracs = (input: string): string => {
  let out = ''
  let i = 0

  while (i < input.length) {
    const idx = input.indexOf('\\frac', i)

    if (idx < 0) {
      out += input.slice(i)

      return out
    }

    const after = input[idx + 5]

    // `(?![A-Za-z])` 用于保护 `\fraction` 等假设命令。
    if (after && /[A-Za-z]/.test(after)) {
      out += input.slice(i, idx + 5)
      i = idx + 5

      continue
    }

    out += input.slice(i, idx)

    let p = idx + 5

    while (input[p] === ' ' || input[p] === '\t') {
      p++
    }

    const num = readBraced(input, p)

    if (!num) {
      out += input.slice(idx, p + 1)
      i = p + 1

      continue
    }

    p = num.end

    while (input[p] === ' ' || input[p] === '\t') {
      p++
    }

    const den = readBraced(input, p)

    if (!den) {
      out += input.slice(idx, p + 1)
      i = p + 1

      continue
    }

    out += `${wrapForFrac(replaceFracs(num.content))}/${wrapForFrac(replaceFracs(den.content))}`
    i = den.end
  }

  return out
}

// 用圆括号包住多令牌表达式，使 `\frac{a+b}{c}` 变为 `(a+b)/c` 而非 `a+b/c`。只要行内
// `/` 会改变含义就加括号，包括任何二元运算符（`+`、`-`、`*`、`/`）或分隔令牌的空格。
// `*` 和 `/` 很重要，否则 `\frac{a*b}{c}` 与 `\frac{1/x}{y}` 会显示成有结合歧义的
// `a*b/c` 和 `1/x/y`。`n!`、`x^2`、`\sin x` 等原子因子不触发规则并保持无括号，
// 否则只会让输出更杂乱。
const wrapForFrac = (expr: string) => {
  const trimmed = expr.trim()

  if (!trimmed) {
    return trimmed
  }

  if (/^\(.*\)$/.test(trimmed)) {
    return trimmed
  }

  if (/[+\-/*]|\s/.test(trimmed)) {
    return `(${trimmed})`
  }

  return trimmed
}

export function texToUnicode(input: string): string {
  let s = input

  s = s.replace(/\\mathbb\s*\{([A-Za-z])\}/g, (raw, c: string) => BB[c] ?? raw)
  s = s.replace(/\\mathcal\s*\{([A-Za-z])\}/g, (raw, c: string) => CAL[c] ?? raw)
  s = s.replace(/\\mathfrak\s*\{([A-Za-z])\}/g, (raw, c: string) => FRAK[c] ?? raw)
  s = s.replace(/\\mathbf\s*\{([^{}]+)\}/g, (_, c: string) => c)
  s = s.replace(/\\mathit\s*\{([^{}]+)\}/g, (_, c: string) => c)
  s = s.replace(/\\mathrm\s*\{([^{}]+)\}/g, (_, c: string) => c)
  s = s.replace(/\\text\s*\{([^{}]+)\}/g, (_, c: string) => c)
  s = s.replace(/\\operatorname\s*\{([^{}]+)\}/g, (_, c: string) => c)

  s = s.replace(/\\overline\s*\{([^{}]+)\}/g, (_, c: string) => `${c}\u0305`)
  s = s.replace(/\\hat\s*\{([^{}]+)\}/g, (_, c: string) => `${c}\u0302`)
  s = s.replace(/\\bar\s*\{([^{}]+)\}/g, (_, c: string) => `${c}\u0304`)
  s = s.replace(/\\tilde\s*\{([^{}]+)\}/g, (_, c: string) => `${c}\u0303`)
  s = s.replace(/\\vec\s*\{([^{}]+)\}/g, (_, c: string) => `${c}\u20D7`)
  s = s.replace(/\\dot\s*\{([^{}]+)\}/g, (_, c: string) => `${c}\u0307`)
  s = s.replace(/\\ddot\s*\{([^{}]+)\}/g, (_, c: string) => `${c}\u0308`)

  s = replaceFracs(s)

  // `\boxed{X}` / `\fbox{X}` 用于高亮最终答案。终端无法绘制真正的框，因此用不可打印且
  // 不会出现在真实文本中的 U+0001 / U+0002 控制字符包住内容，让 Markdown 渲染器按它们
  // 切分并对框内区域应用高亮样式（反色）。这样 `texToUnicode` 仍是纯字符串转换，实际视觉
  // 强调由 React 层完成。参数使用配对花括号解析，保留框内上下标/分数中的嵌套 `{...}`。
  s = replaceBracedCommand(s, '\\boxed', body => `${BOX_OPEN}${body.trim()}${BOX_CLOSE}`)
  s = replaceBracedCommand(s, '\\fbox', body => `${BOX_OPEN}${body.trim()}${BOX_CLOSE}`)

  // `\xrightarrow{label}` / `\xleftarrow{label}` 折叠为带行内标签的箭头。LaTeX 将标签渲染
  // 在箭头上方；等宽终端中将它相邻放置，`─label→` 是最接近且可读的近似。需在符号阶段前
  // 运行，使标签随后仍能替换希腊字母和运算符。
  s = s.replace(/\\xrightarrow\s*\{([^{}]*)\}/g, (_, label: string) => `─${label.trim()}→`)
  s = s.replace(/\\xleftarrow\s*\{([^{}]*)\}/g, (_, label: string) => `←${label.trim()}─`)
  s = s.replace(/\\Longrightarrow/g, '⟹')
  s = s.replace(/\\Longleftarrow/g, '⟸')
  s = s.replace(/\\Longleftrightarrow/g, '⟺')

  // `\pmod{p}` 转为 ` (mod p)`（LaTeX 自动加括号）；`\pod{p}` 是无括号变体；`\tag{n}`
  // 是显示在公式右侧的编号标注。统一折叠为前导单空格的括号形式。模式开头的 `\s*` 吸收
  // 源码已有空白，避免用户写 `b \pmod{p}` 时得到双空格的 `b  (mod p)`。
  s = s.replace(/\s*\\pmod\s*\{([^{}]*)\}/g, (_, p: string) => ` (mod ${p.trim()})`)
  s = s.replace(/\s*\\pod\s*\{([^{}]*)\}/g, (_, p: string) => ` (${p.trim()})`)
  s = s.replace(/\s*\\tag\s*\{([^{}]*)\}/g, (_, n: string) => ` (${n.trim()})`)

  // `\big`、`\Big`、`\bigg`、`\Bigg`（可带 `l`/`r`/`m` 后缀）是类似
  // `\left`/`\right` 但无自动配对语义的尺寸包装器。移除包装器并保留后续分隔符；末尾
  // `(?![A-Za-z])` 防止削掉 `\bigtriangleup` 等继续以字母组成的命令。
  s = s.replace(/\\(?:Bigg|bigg|Big|big)[lrm]?(?![A-Za-z])/g, '')

  // 样式/尺寸提示不排版任何字形，只影响真实 LaTeX 引擎中的尺寸。终端中每个字形都占一个
  // 等宽单元，因此无需处理；将其连同尾随空白删除，避免原始 `\displaystyle` 泄漏到输出。
  s = s.replace(/\\(?:scriptscriptstyle|displaystyle|scriptstyle|textstyle|nolimits|limits)(?![A-Za-z])\s*/g, '')

  // `\left` 和 `\right` 是任意分隔符的尺寸包装器，可为裸分隔符（`\left(`）、转义分隔符
  // （`\left\{`）或具名分隔符（`\left\langle`）。无条件移除包装器，由后续流水线或符号
  // 阶段处理分隔符。可选 `.?` 会消费表示“无分隔符”的 `\left.` / `\right.`；前瞻
  // `(?![A-Za-z])` 保护 `\leftarrow` / `\leftrightarrow`。
  s = s.replace(/\\left(?![A-Za-z])\.?/g, '')
  s = s.replace(/\\right(?![A-Za-z])\.?/g, '')

  // 在上下标前运行符号替换，使 `^{\infty}` 先变为 `^{∞}`；convertScript 随后尝试把 ∞
  // 映射为上标（Unicode 无此字符），或移除已变成单字符正文外的花括号并回退为 `^∞`。
  //
  // 标点阶段先运行；标点后可以跟字母（`\{p` 表示左花括号后接 p），字母阶段的
  // `(?![A-Za-z])` 规则会错误阻止它们。
  s = s.replace(SYMBOL_PUNCT_RE, m => SYMBOLS[m] ?? m)
  s = s.replace(SYMBOL_LETTER_RE, m => SYMBOLS[m] ?? m)

  // 裸 `^c` / `_c` 仅处理字母数字和 `+`/`-`/`=`。刻意排除圆括号，因为上方花括号回退
  // 可能输出 `(...)`；不应让第二遍贪婪地把左括号转为 `⁽`，留下孤立的右括号。
  s = s.replace(/\^\s*\{([^{}]+)\}/g, (_, body: string) => convertScript(body, SUPERSCRIPT, '^'))
  s = s.replace(/\^([A-Za-z0-9+\-=])/g, (raw, ch: string) => SUPERSCRIPT[ch] ?? raw)
  s = s.replace(/_\s*\{([^{}]+)\}/g, (_, body: string) => convertScript(body, SUBSCRIPT, '_'))
  s = s.replace(/_([A-Za-z0-9+\-=])/g, (raw, ch: string) => SUBSCRIPT[ch] ?? raw)

  return s
}
