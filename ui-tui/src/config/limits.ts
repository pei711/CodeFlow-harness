export const LARGE_PASTE = { chars: 8000, lines: 80 }

export const LIVE_RENDER_MAX_CHARS = 16_000
export const LIVE_RENDER_MAX_LINES = 240

// 对 FULL_RENDER_TAIL 之外消息的历史渲染限制。每个渲染行约等于一个
// Yoga/Text 节点加行内 span，因此这是 PageUp 追赶时控制冷挂载成本的主要
// 手段。16 行乘 25 个挂载项约为 400 个节点，能稳定控制在每帧 16 毫秒预算内。
// 用户向前翻页主要用于辨认而非精读；消息进入尾部范围后再完整重绘。
export const HISTORY_RENDER_MAX_CHARS = 800
export const HISTORY_RENDER_MAX_LINES = 16
export const FULL_RENDER_TAIL_ITEMS = 8

export const LONG_MSG = 300
export const MAX_HISTORY = 800
export const THINKING_COT_MAX = 160

// 加速前每次滚轮事件滚动的行数。设为 1 可保持 Ink 的 DECSTBM 快速路径有效
// （每次滚动小于视口高度减一）并产生平滑移动；wheelAccel.ts 会在持续滚动时
// 逐步提升该值。
export const WHEEL_SCROLL_STEP = 1
