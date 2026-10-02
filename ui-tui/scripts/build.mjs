#!/usr/bin/env node
// 将 src/entry.tsx 打包为单个自包含 dist/entry.js，无需运行时 node_modules。
import { build } from 'esbuild'
import { readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const here = dirname(fileURLToPath(import.meta.url))
const root = resolve(here, '..')
const out = resolve(root, 'dist/entry.js')

// 仅在运行时 DEV=true（Ink 开发模式）时导入 `react-devtools-core`。用桩替代，
// 避免包内携带该依赖。
const stubDevtools = {
  name: 'stub-react-devtools-core',
  setup(b) {
    b.onResolve({ filter: /^react-devtools-core$/ }, args => ({
      path: args.path,
      namespace: 'stub-devtools'
    }))
    b.onLoad({ filter: /.*/, namespace: 'stub-devtools' }, () => ({
      contents: 'export default { initialize() {}, connectToDevTools() {} }',
      loader: 'js'
    }))
  }
}

await build({
  entryPoints: [resolve(root, 'src/entry.tsx')],
  bundle: true,
  platform: 'node',
  format: 'esm',
  target: 'node20',
  outfile: out,
  jsx: 'automatic',
  jsxImportSource: 'react',
  // 跳过预构建 @hermes/ink 包，因为 esbuild 的 __esm 辅助函数不会等待嵌套异步
  // 初始化，会破坏 `render` 等延迟初始化导出；从源码打包可绕过此问题。
  alias: { '@hermes/ink': resolve(root, 'packages/hermes-ink/src/entry-exports.ts') },
  plugins: [stubDevtools],
  // 部分传递依赖在运行时使用 CommonJS `require(...)`。ESM 包不会自动获得
  // `require` 绑定，因此需要注入。
  banner: {
    js: "import { createRequire as __cr } from 'node:module'; const require = __cr(import.meta.url);"
  },
  logLevel: 'info'
})

// esbuild 会把 src/entry.tsx 的 shebang 保留到包中，但 Nix 的 patchShebangs
// 阶段会破坏 `/usr/bin/env -S node --foo --bar`，移除 `node` 后留下无效解释器。
// hermes_cli 启动器始终通过 `node dist/entry.js` 调用本文件，因此 shebang 冗余，
// 应将其移除。
const body = readFileSync(out, 'utf8')
if (body.startsWith('#!')) {
  writeFileSync(out, body.slice(body.indexOf('\n') + 1))
}

console.log(`built ${out}`)
