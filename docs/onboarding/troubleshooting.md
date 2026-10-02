# Onboarding 故障排查

## `configured memory plugin is unavailable`

CodeFlow 选择了一个当前工具环境中不存在的外部 Memory Backend。

先检查安装与 Plugin 状态：

```bash
uv tool list
codeflow plugins
codeflow doctor --json
```

当前 GitHub 发布的受支持处理方式是显式关闭 Memory：

```bash
codeflow onboard --skip-memory --reset
```

`--reset` 会重新进入配置流程。执行前记录现有 Provider、Sandbox 和渠道选择。

## 安装器无法读取 Private Release

确认当前账号可以访问仓库，并通过环境变量提供 Token：

```bash
export CODEFLOW_GITHUB_TOKEN="<your-github-token>"
./install.sh
```

不要把 Token 放进远程 URL、仓库文件或故障截图。需要固定制品时，使用维护者提供并
经过校验的 `CODEFLOW_WHEEL_URL`。

## Provider 预检通过，第一条 Turn 失败

`GET /models` 成功只证明凭证与网络的一部分。真实 Turn 还会检查模型 ID、账户额度、
Runtime 和当前工具环境。

```bash
codeflow provider test <provider-name>
codeflow doctor --probe
```

检查默认模型是否属于当前 Provider，以及代理或 VPN 是否能访问对应 Endpoint。
`--skip-test` 只跳过付费调用，不应被记录成 Provider 已验证。

## `codeflow` 无法打开原生 TUI

先检查安装版本与 Runtime：

```bash
codeflow --version
codeflow --check
```

原生 TUI 需要 Node.js 22。安装器可以配置私有 Node Runtime；如果安装被中断，重新
运行同一版本的安装器，再执行 `codeflow --check`。

## 飞书收不到消息

按顺序检查：

1. 应用是企业自建应用，机器人能力已经启用。
2. 事件模式是 WebSocket 长连接，已经添加 `im.message.receive_v1`。
3. 包含新权限与新事件的应用版本已经发布。
4. `codeflow channels get feishu` 显示 enabled，App ID 正确，密钥已经设置。
5. `codeflow gateway --workspace "$PWD" --verbose` 仍在运行。
6. 群聊中已经 @ 机器人；默认 `group-policy=mention`。
7. `allow-from` 包含当前用户 `open_id`，或者调试期间设置为 `['*']`。

配置保存成功不等于真实收发成功。最终验收需要一个真实用户发送入站消息，并在同一
会话里收到 CodeFlow 回复。

## Gateway 在运行，但操作了错误仓库

一个 Gateway 进程只服务一个固定 Workspace。停止旧进程，再使用目标路径重启：

```bash
codeflow gateway --workspace /absolute/path/to/project --verbose
```

不要根据启动终端当前目录推测 Gateway 的 Workspace，以显式参数和启动日志为准。

## 向导检测到已有配置

默认情况下，CodeFlow 会保护已有配置。自动化场景可以使用 `--yes` 复用；只有明确需要
重做配置时才使用 `--reset`。

```bash
codeflow onboard --skip-memory --yes
codeflow onboard --skip-memory --reset
```

如果不确定配置属于谁，先运行 `codeflow doctor --json` 和 `codeflow channels list`，不要
直接重置。
