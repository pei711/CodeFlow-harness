# 把 CodeFlow 接到飞书

目标：在飞书里给机器人发一条消息，由正在运行的 CodeFlow Gateway 返回真实回复。

CodeFlow 使用飞书 WebSocket 长连接，不需要公网 IP 或 webhook 域名。

## 1. 创建企业自建应用

1. 打开[飞书开放平台](https://open.feishu.cn/)，创建“企业自建应用”。
2. 在应用能力中启用“机器人”。
3. 在“凭证与基础信息”中复制 App ID 和 App Secret。App Secret 只应输入本地 CodeFlow 配置，不要发到群聊、Issue 或截图里。

## 2. 配置权限

在“权限管理”中申请：

- `im:message:send_as_bot`：以机器人身份回复。
- `im:message.p2p_msg:readonly`：接收单聊消息。
- `im:message.group_at_msg:readonly`：接收群里 @ 机器人的消息。
- `im:resource`：可选，发送或读取图片、文件等资源时使用。

权限名以飞书开放平台当前界面为准，可在[权限列表](https://open.feishu.cn/document/server-docs/application-scope/scope-list?lang=zh-CN)中复核。

## 3. 配置事件

1. 进入“事件与回调”。
2. 选择“使用长连接接收事件”。
3. 添加事件 `im.message.receive_v1`。
4. 创建并发布应用版本，然后等待管理员审批。权限或事件变更后，需要再次发布才会生效。

对照飞书官方文档：[长连接接收事件](https://open.feishu.cn/document/server-docs/event-subscription-guide/event-subscription-configure-/request-url-configuration-case?lang=zh-CN)、[添加事件](https://open.feishu.cn/document/server-docs/event-subscription-guide/event-subscription-configure-/subscription-event-case?lang=zh-CN)和[接收消息](https://open.feishu.cn/document/server-docs/im-v1/message/events/receive?lang=zh-CN)。

## 4. 在 CodeFlow 中启用飞书

可以在 `codeflow onboard` 的第 4 步填写，也可以直接运行：

```bash
codeflow channels enable feishu \
  --app-id "cli_xxxxxxxxxxxxxxxx" \
  --app-secret "$FEISHU_APP_SECRET"
```

如果飞书后台开启了 Encrypt Key 或 Verification Token，把它们一起填入：

```bash
codeflow channels set feishu \
  --encrypt-key "$FEISHU_ENCRYPT_KEY" \
  --verification-token "$FEISHU_VERIFICATION_TOKEN"
```

检查字段与脱敏后的配置：

```bash
codeflow channels show feishu
codeflow channels get feishu
```

## 5. 启动 Gateway，发送第一条消息

Gateway 只处理它启动时选定的 Workspace。先进入想让 CodeFlow 工作的仓库：

```bash
cd /path/to/your-project
codeflow gateway --workspace "$PWD" --verbose
```

保持这个进程运行，然后：

- 单聊：直接给机器人发消息。
- 群聊：默认 `group-policy=mention`，需要 @ 机器人。

看到 Gateway 日志中出现 inbound accepted，并在飞书里收到回复，才算通路成功。[发送消息 API](https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/im-v1/message/create) 是 CodeFlow 回复时使用的飞书能力。

## 安全收口

Feishu 的 `allow-from` 当前默认是 `['*']`，表示允许任意发件人。公开机器人上线前，应收紧到明确的用户 `open_id`：

```bash
codeflow channels set feishu --allow-from ou_xxxxxxxxxxxxxxxx
```

修改后重启 Gateway。不要把 App Secret、Encrypt Key 或 Verification Token 写入文档和 Git。
