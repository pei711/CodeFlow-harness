// 仓库 TypeScript/Node 包共享的扁平配置基线，不包含 React。
//
// 本文件位于没有 node_modules 的仓库根目录。ESM 从模块自身位置向上解析裸导入，
// 因此根文件导入 'eslint-plugin-*' 会解析失败。为避免这一点，基线采用工厂形式：
// 每个包从自身 node_modules 解析并导入插件，再将其传入；基线本身不导入插件。

export default function base({ js, tsPlugin, tsParser, unusedImports, perfectionist }) {
  return [
    js.configs.recommended,
    {
      files: ['**/*.{ts,tsx}'],
      languageOptions: {
        ecmaVersion: 'latest',
        sourceType: 'module',
        parser: tsParser,
        parserOptions: {
          ecmaFeatures: { jsx: true },
        },
      },
      plugins: {
        '@typescript-eslint': tsPlugin,
        'unused-imports': unusedImports,
        perfectionist,
      },
      rules: {
        ...tsPlugin.configs['flat/recommended'].reduce(
          (acc, cfg) => ({ ...acc, ...(cfg.rules ?? {}) }),
          {},
        ),
        '@typescript-eslint/consistent-type-imports': 'error',
        '@typescript-eslint/no-explicit-any': 'warn',
        '@typescript-eslint/no-unused-vars': 'off',
        'unused-imports/no-unused-imports': 'error',
        'perfectionist/sort-imports': 'error',
        curly: ['error', 'all'],
        'no-fallthrough': 'error',
        'no-unused-expressions': 'off',
        '@typescript-eslint/no-unused-expressions': 'warn',
        'no-undef': 'off',
        'no-unused-vars': 'off',
      },
    },
  ]
}
