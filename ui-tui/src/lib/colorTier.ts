/**
 * 颜色能力覆盖设置，必须在 Chalk 和 supports-color 初始化前应用。
 *
 * 实际终端能力交由 Chalk（主流 `supports-color`）探测，再由 hermes-ink 针对
 * VS Code/tmux 修正级别。本模块只处理用户覆盖值，将其转换为 hermes-ink 读取的
 * `HERMES_TUI_LEVEL` 环境变量，以精确固定 Chalk 级别。FORCE_COLOR 只能设置
 * 下限；当 COLORTERM=truecolor 时 supports-color 仍返回 3，无法强制降级为
 * 256 色或 16 色。
 *
 * 配置渠道：
 *   - `CODEFLOW_TUI_COLOR` = auto | truecolor | 256 | 16 | none（`--color`
 *     参数会转发到这里）；
 *   - `CODEFLOW_TUI_TRUECOLOR` = 1/true/...，是
 *     `CODEFLOW_TUI_COLOR=truecolor` 的旧版别名。
 *
 * 优先级从高到低为 NO_COLOR、CODEFLOW_TUI_COLOR、旧版 CODEFLOW_TUI_TRUECOLOR、
 * Chalk 自动探测。根据产品约定，显式 `--color` 也不能覆盖 `NO_COLOR`。
 */

export type ColorTier = 0 | 1 | 2 | 3

const TRUE_RE = /^(?:1|true|yes|on)$/i

/**
 * 从 CodeFlow 颜色环境变量解析请求的级别。返回强制级别；“auto”则返回 `null`，
 * 交由 Chalk 探测。
 */
export function parseColorOverride(env: NodeJS.ProcessEnv = process.env): ColorTier | null {
  const raw = (env.CODEFLOW_TUI_COLOR ?? '').trim().toLowerCase()

  switch (raw) {
    case 'none':
    case 'off':
    case '0':
      return 0
    case '16':
    case 'ansi':
    case '1':
      return 1
    case '256':
    case 'ansi256':
    case '2':
      return 2
    case 'truecolor':
    case '24bit':
    case 'rgb':
    case '3':
      return 3
    case '':
    case 'auto':
      break
    default:
      break
  }

  // 兼容旧别名：CODEFLOW_TUI_TRUECOLOR=1 等价于 CODEFLOW_TUI_COLOR=truecolor。
  if (TRUE_RE.test((env.CODEFLOW_TUI_TRUECOLOR ?? '').trim())) {
    return 3
  }

  return null
}

/**
 * 通过设置 `HERMES_TUI_LEVEL` 将覆盖值原地应用到 `env`。必须在导入 Chalk 或
 * hermes-ink 前执行。返回固定的级别；保持自动探测时返回 `null`。
 */
export function applyColorOverride(env: NodeJS.ProcessEnv = process.env): ColorTier | null {
  // 按 no-color.org 约定，NO_COLOR 设置为任意值都优先于其他覆盖项，包括显式
  // --color，此时关闭颜色。
  if ('NO_COLOR' in env) {
    env.HERMES_TUI_LEVEL = '0'
    env.FORCE_COLOR = '0'
    return 0
  }

  const tier = parseColorOverride(env)

  if (tier === null) {
    return null
  }

  // 精确固定 Chalk 颜色级别，hermes-ink 会读取 HERMES_TUI_LEVEL。
  env.HERMES_TUI_LEVEL = String(tier)
  // 选择 `none` 时也关闭 FORCE_COLOR，使非 Hermes 的 Chalk 实例同样停止着色。
  if (tier === 0) {
    env.FORCE_COLOR = '0'
  }

  return tier
}

applyColorOverride()

export {}
