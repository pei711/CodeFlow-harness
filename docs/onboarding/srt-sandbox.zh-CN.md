# 使用 SRT Shell 沙箱

CodeFlow-harness 支持 `none`、`boxlite` 和 `srt` 三种 Shell 执行后端。`srt` 使用 Anthropic Sandbox Runtime
按操作系统隔离每条 Shell 命令；它不会建立持久虚拟机，也不支持在沙箱内启动 MCP stdio 子进程。文件、MCP
及消息工具各自仍受其已有的权限和确认机制控制。

## 安装

需要 Node.js 20.11 或更高版本。在仓库根目录安装锁定的 SRT 版本：

```bash
npm ci
```

Windows 首次使用前，以管理员权限初始化 SRT 的系统隔离组件：

```powershell
npx @anthropic-ai/sandbox-runtime windows-install
```

macOS 和 Linux 用户应按 Anthropic Sandbox Runtime 对应平台的要求准备操作系统沙箱依赖。

## 启用

在 CodeFlow 配置文件中将后端设为 `srt`：

```json
{
  "tools": {
    "sandbox": {
      "backend": "srt"
    }
  }
}
```

默认策略允许 Shell 命令在当前工作区读写，拒绝访问 `.env` 和 `.codeflow`，禁止写入 `.git`，并关闭网络。
策略文件位于 `codeflow/sandbox/srt-settings.json`；也可以通过 `srtSettingsPath` 指定另一份 SRT JSON 设置，
或通过 `srtCliPath` 指定安装好的 SRT CLI。Node.js、SRT CLI 或操作系统隔离初始化不可用时，Agent 启动会失败，
不会退回到宿主机 Shell。

`srt` 只隔离 `exec` 工具发出的命令。`exec` 的工作目录必须位于配置的工作区内。SRT 后端不支持 MCP stdio
服务器；需要运行此类服务器时请使用 `boxlite`。
