"""在 benchmark 约束下改进 Agent Harness 的 self-evolution 子系统。

初学者可以把 Evolver 理解为一个有预算上限的实验流水线，而不是让 Agent 随意改自己：它
先诊断失败 trajectory，把 candidate patch 做成真实 git commit，进行低成本 screen，再对
survivor 做 ``K=3`` confirm；只有相对 baseline 通过三道 verification gate 的 candidate
才允许 promote，最后只用一次 sealed test set 估计 honest generalisation。方法见
``docs/specs/self-evolution-loop-sop.md``；用户入口与 quickstart 位于
``codeflow/evolver/README.md``，命令是 ``codeflow evolve``。

主要数据流是 run spec -> benchmark trial/evidence -> diagnosis/design -> candidate commit ->
screen/confirm/gates -> archive/promotion -> sealed evaluation。各子 Package 的所有权如下：

* ``launch`` 是 unified entry，定义 run spec、bench plugin contract 与
  run/check/status/finalize state machine；
* ``orchestrator`` 执行 SOP round loop，包括 diagnose、design、screen、confirm、gates 与
  sealed test，设计背景见 ``orchestrator/DESIGN.md``；
* ``tree`` 保存 git-backed Harness version tree，每个 candidate 对应 commit；
* ``activation`` 用 beacon ledger 证明 candidate code 是否实际触发；
* ``applier`` 对 candidate edit 执行 path whitelist 与 beacon guard；
* ``judge`` 提供 LLM judge 的 L1/L2/L3 与 ``(WHERE, WHY)`` 输出；
* ``scheduler`` 负责 task/anchor selection 与 bandit utility；
* ``analysis`` 读取 trial ledger 并做 stability bucketing；
* ``compressor`` 为 diagnosis prompt 压缩 trajectory。

证据边界必须保持清楚：candidate code 调用成功不等于 beacon 已记录；beacon 已记录不等于
benchmark task 完成；单次 task 成功不等于 confirm 稳定；gate 通过不等于 sealed
generalisation；只有满足各自数据与统计条件时，正向结论才可用。

docstring 中 ``spec §NN`` 引用 upstream project 未随仓库发布的 internal design document；
shipped methodology spec 是 ``docs/specs/self-evolution-loop-sop.md``。编号保留为 constant 和
Enum 的 provenance。``TB2`` 是本系统最初面向的 upstream terminal-agent benchmark line，
随仓示例 benchmark 是 AppWorld。``GSME``（Gated Semantic MAP-Elites）指
``orchestrator/archive.py`` 中的 quality-diversity elite archive。
"""
