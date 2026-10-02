# 参与 CodeFlow 开源贡献

感谢你愿意帮助改进 CodeFlow。第一次贡献不需要先理解整个 Runtime；请选择一个已经完成
需求澄清、带有明确验收命令的 Issue，从一个可验证的小改动开始。

## 从哪里开始

1. 在 [GitHub Issues](https://github.com/pei711/codeflow-harness/issues) 中选择带有
   `good-first-issue` 或 `help-wanted` 标签的任务。
2. 阅读 Issue 中的目标分支、范围、相关文件、非目标和验收命令。
3. 在 Issue 下留言：`我来认领，预计在 7 天内提交 PR。`
4. 等待维护者确认认领后再开始修改，避免多人重复实现。

`good-first-issue` 表示任务已经完成方案决策，维护者会为第一次贡献提供额外帮助。
如果认领后 3 天没有任何进展，维护者会提醒一次；7 天没有更新时，任务会重新开放。

## 选择正确的目标分支

| 目标分支 | 用途 |
| --- | --- |
| `main` | CodeFlow 的公开发布基线、通用 Runtime、文档和贡献基础设施 |
| `release/maintainer-profile` | 飞书开源社区 Maintainer Agent 业务场景 |

每个可认领 Issue 都必须写明目标分支。Fork 后从对应的最新远端分支创建功能分支，
Pull Request 也提交回同一个目标分支。不要根据修改文件的位置猜测目标分支。

## 本地开发

CodeFlow 使用 Python 3.12、Node.js 22 和 `uv`。依赖安装与常用验证命令：

```bash
make install
make ci
```

针对小改动，先运行 Issue 指定的聚焦测试；提交前再运行与改动范围相称的完整检查。
所有 Python 测试都通过 `uv run pytest` 执行，不要使用裸 `pytest` 或 `pip`。

常用命令：

```bash
make lint-python
make lint-tui
make test-python
make test-tui
make check-large-files
```

## 修改要求

- 一个 Issue 对应一个 Pull Request；不要顺手处理无关问题。
- 行为变化需要更新或增加可观察的回归测试。
- 用户可见行为变化需要同步更新文档。
- 不提交 Token、密码、私钥、个人数据、真实运行日志或本地路径。
- 不提交报告图片、视频、PDF、HTML 制品或超过 1 MiB 的文件。
- 不新增依赖，除非 Issue 明确要求并得到维护者确认。
- 分支使用 `<type>/<snake_case_desc>`，提交和 PR 标题使用 Conventional Commits。

## 提交与 Pull Request

提交信息和 PR 标题使用 Conventional Commits，并保持全英文 ASCII：

```text
docs: add a first contribution example
fix(maintenance): localize command errors
test(channels): cover topic reply routing
```

PR 描述至少包含：

- 修改内容和原因；
- 关联的 GitHub Issue；
- 精确验证命令及结果；
- 风险和回滚方式；
- 目标分支。

## 问题、Bug 与安全报告

普通使用问题可以先在 CodeFlow 飞书社区中提问。能够稳定复现的 Bug 再转为 GitHub Issue，
并填写环境、复现步骤、预期行为、实际行为和已脱敏日志。

不要在公开 Issue 中报告安全漏洞，也不要粘贴 Token、App Secret、SSH 私钥、内部地址
或其他敏感数据。安全问题请按照 [SECURITY.md](SECURITY.md) 中的方式私下报告。
