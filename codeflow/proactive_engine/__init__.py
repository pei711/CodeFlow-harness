"""为 CodeFlow Runtime 提供由用户创建、可持久化的 Cron Scheduling。

Proactive Engine 让 Agent 不必等待新的 Inbound Message，也能在约定时间重新触发任务。当前实现围绕
Cron Scheduler：把 Job 定义写入持久化 Store，在 Runtime 启动时恢复，并把到期 Trigger 转换成普通
Agent Turn。它负责“何时发起”，不改变 Agent Loop 对工具、权限和最终交付的判断。
"""
