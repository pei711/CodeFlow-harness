const truthy = (v?: string) => /^(?:1|true|yes|on)$/i.test((v ?? '').trim())

export const STARTUP_RESUME_ID = (process.env.CODEFLOW_TUI_RESUME ?? '').trim()
export const STARTUP_QUERY = (process.env.CODEFLOW_TUI_QUERY ?? '').trim()
export const STARTUP_IMAGE = (process.env.CODEFLOW_TUI_IMAGE ?? '').trim()
export const MOUSE_TRACKING = !truthy(process.env.CODEFLOW_TUI_DISABLE_MOUSE)
export const NO_CONFIRM_DESTRUCTIVE = truthy(process.env.CODEFLOW_TUI_NO_CONFIRM)

// 跳过 AlternateScreen，让 TUI 渲染到主缓冲区，使宿主终端原生回滚区能保留
// 滚出顶部的内容。该实验开关用于在同一流水线上比较原生滚动与虚拟化滚动。
export const INLINE_MODE = truthy(process.env.CODEFLOW_TUI_INLINE)

// 实时 FPS 计数浮层由 Ink 的 onFrame 提供数据，反映真实渲染速率而非合成计时器。
export const SHOW_FPS = truthy(process.env.CODEFLOW_TUI_FPS)
