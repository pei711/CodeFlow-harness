"""用户创建的 Persistent Cron Schedules 命名空间。

Scheduler 子包保存 Cron Job 的类型、持久化 Service 与 Agent-facing Tool。公开命名空间本身不启动
Background Runner；具体 Runtime 必须显式装配并 Start Service。
"""
