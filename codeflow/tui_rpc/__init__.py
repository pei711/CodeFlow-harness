"""连接 CodeFlow Python Runtime 与 Node TUI 的 JSON-RPC 协议边界。

没有 RPC 经验的读者可以把本 Package 理解为 Runtime 的远程控制面：Node TUI 发送
request，``Dispatcher`` 校验并路由到 ``methods``；Runtime 再用 notification 把 Turn、
Tool、Session、确认问题和终端状态推送回前端。``server`` 管理 stdio frame，``spine``
把 Agent 事件翻译成 RPC 事件，``models`` 定义线上的 Pydantic v2 数据结构。

契约的 single source of truth 位于 ``ui-tui/rpc-schema/openrpc.json``；
:mod:`codeflow.tui_rpc.models` 是手写对应模型，并由 ``tests/test_rpc_schema_match.py`` 保持
同步。RPC response 成功只表示方法调用完成，不自动等于 Session 已持久化、Agent 任务
完成或用户已收到最终交付。本 Package 不负责 React/TypeScript UI 布局，也不定义
Agent 推理策略。
"""
