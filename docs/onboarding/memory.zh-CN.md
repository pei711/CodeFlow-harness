# Memory 发布边界

CodeFlow 内置项目级的已完成轮次摘要和两路混合召回。它将最近 200 轮保存为短摘要，按新问题执行语义检索
与 BM25 关键词检索，再用 RRF 合并排名并最多注入三条。FastEmbed 是可选依赖；未安装或模型不可用时，
语义一路降级为 TF-IDF 字符向量，关键词检索仍可工作。

Memory Backend 协议仍允许外部长期记忆接入；当前 GitHub 发布不包含外部 Memory 实现、安装地址或配套制品。

## 当前受支持的配置

首次配置时显式关闭 Memory：

```bash
codeflow onboard --skip-memory
```

有效配置是 `memory.backend = null`。这只关闭外部 Memory Backend，不会关闭内置轮次摘要召回、
Host 用户资料、Local Skills、Session、Context、Tool 或其他 CodeFlow Runtime 能力。

向导仍会通过完整 Runtime 执行第一条 Turn。Memory 关闭不应阻止 Provider、工具或
Session 的正常工作。

## 为什么必须显式关闭

CodeFlow 不会把缺失的 Memory Plugin 当作可用状态。如果配置选择了一个未安装的
Backend，启动与诊断会 fail closed，避免用户以为 Recall 已经生效。

检查当前状态：

```bash
codeflow plugins
codeflow doctor --json
```

如果已有配置指向未安装的 Backend，重新进入向导并明确关闭：

```bash
codeflow onboard --skip-memory --reset
```

`--reset` 会重新进入配置流程。执行前先确认现有 Provider、Sandbox 和渠道配置可以
被重新选择。

## 外部 Memory 的信任边界

源码中的 Adapter 名称、Plugin identity 校验和兼容测试，只能证明 CodeFlow 预留了集成
边界，不能证明公开用户已经获得可安装制品。

在维护者发布兼容制品和安装说明前：

- 不要根据源码名称猜测仓库或下载地址。
- 不要从开发 Checkout 或未固定版本的包替代正式制品。
- 不要把 Plugin 被发现写成 Backend 已经健康。
- 不要把静态测试写成真实 Recall 或任务效果。

未来接入外部 Memory 时，验收至少要区分 Plugin 发现、Backend 启动、仓库绑定、
写入、Recall 和真实 Turn 注入。任一层没有证据，都应标记为未验证。
