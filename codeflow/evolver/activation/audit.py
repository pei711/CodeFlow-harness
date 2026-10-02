"""执行 Gate 1 post-run audit：被测机制是否真的运行过。

模块合并 ``activation_ledger.jsonl`` 中的 hook fire、beacon、presence assert，与既有
``skill_injections.jsonl`` telemetry，使一次调用可以统计四类 mechanism。``audit_trials``
按 expected source 计算实际 activation count 和仅 wiring 的 count，并区分完全 inert 与
inert-but-wired。

``PASS`` 只证明每个 expected source 至少留下一个运行证据，不证明 task 完成、输出质量提高
或 candidate 可 promote；malformed JSON line 与不可读文件会被跳过，因此缺失证据按未观察到
处理，而不是推断机制一定没有执行。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def _iter_jsonl(path: Path):
    """逐行解析 ``path`` 中的 JSON object，并跳过 malformed line。

    文件打不开时生成器静默结束；单行 ``JSONDecodeError`` 只丢弃该行。该容错用于审计聚合，
    不代表损坏 ledger 合法，也不会修复原文件。
    """
    try:
        for line in path.open():
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
    except OSError:
        return


def audit_trials(roots: list[Path], expected_sources: list[str]) -> dict:
    """审计 ``roots`` 下所有 trial 的 mechanism activation。

    函数递归读取 ``activation_ledger.jsonl``：``kind="hook_active"`` 计入 wired，其余记录计入
    activation count；再从 ``skill_injections.jsonl`` 累加注入 Skill name。``expected_sources``
    决定最终逐项输出与 inert 判断。

    任一 expected source count 为 0 时 verdict 为 ``FAIL``，否则为 ``PASS``；同时返回
    ``inert_sources``、``inert_but_wired``、counts、wired、all_observed 与 ledger 数量。该
    verdict 是 activation gate，不是 benchmark correctness 或效果提升结论。
    """
    counts: Counter = Counter()
    wired: Counter = Counter()
    n_ledgers = 0

    for root in roots:
        for ledger in Path(root).glob("**/activation_ledger.jsonl"):
            n_ledgers += 1
            for rec in _iter_jsonl(ledger):
                src = rec.get("source", "?")
                if rec.get("kind") == "hook_active":
                    wired[src] += 1
                else:
                    counts[src] += 1

        # 技能只记录到 skill_injections.jsonl，钩子只记录到账本，两者不重叠，合并时不会重复计算。
        for inj in Path(root).glob("**/skill_injections.jsonl"):
            for rec in _iter_jsonl(inj):
                for s in rec.get("skills", []):
                    name = s.get("name")
                    if name:
                        counts[name] += 1

    inert = [s for s in expected_sources if counts[s] == 0]
    inert_but_wired = [s for s in inert if wired[s] > 0]
    return {
        "verdict": "FAIL" if inert else "PASS",
        "inert_sources": inert,
        "inert_but_wired": inert_but_wired,
        "counts": {s: counts[s] for s in expected_sources},
        "wired": {s: wired[s] for s in expected_sources},
        "all_observed": dict(counts),
        "n_ledgers": n_ledgers,
    }
