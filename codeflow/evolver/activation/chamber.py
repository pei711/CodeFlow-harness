"""在正式实验前，用 recorded trajectory 重放 activation spec。

Chamber 的问题是 candidate 机制在已有行为分布中是否至少可达。``load_corpus`` 从传入 roots
收集 Session trajectory 并保留 provenance，``run_chamber`` 调用 ``evaluate_spec`` 统计可达
数量。design section 8 的 corpus 约定是：uv-contaminated run 的 Agent trajectory 仍是
VALID behavioral data，因为 contamination 发生在 verifier-side；default corpus 是 roots
中所有 trial dir。

``PASS`` 只表示 spec 在 corpus 中至少命中一次，不能证明新 candidate 实际激活、任务成功、
稳定性或正向效果；``BLOCK`` 表示当前 corpus 没有可达证据，不等价于逻辑上永远不可达。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from codeflow.evolver.activation.spec import ActivationSpec, evaluate_spec

SESSION_GLOB = "**/sessions/tb2-task.jsonl"


@dataclass
class Corpus:
    """Chamber 使用的 trajectory 集合及其来源记录。

    ``trajectories`` 的每项是一段 JSON event sequence，``provenance`` 保存扫描 root 字符串。
    对象只拥有内存 snapshot，不验证每段 trajectory 是否完整。
    """

    trajectories: list[list[dict]]
    provenance: list[str] = field(default_factory=list)


@dataclass
class ChamberReport:
    """一个 candidate activation spec 在 corpus 上的可达性报告。

    报告记录 node、spec kind、reachable count、corpus size 与 provenance；``verdict`` 是由
    reachable count 派生的 ``PASS``/``BLOCK``，没有额外统计推断。
    """

    node_id: str
    spec_kind: str
    reachable_count: int
    corpus_size: int
    provenance: list[str]

    @property
    def verdict(self) -> str:
        return "PASS" if self.reachable_count > 0 else "BLOCK"

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "spec_kind": self.spec_kind,
            "reachable_count": self.reachable_count,
            "corpus_size": self.corpus_size,
            "provenance": self.provenance,
            "verdict": self.verdict,
        }


def load_corpus(roots: list[Path]) -> Corpus:
    """从 ``roots`` 递归加载 ``SESSION_GLOB`` 匹配的非空 trajectories。

    每个 root 都写入 provenance。malformed JSON line 与不可读 Session file 被跳过；完全空的
    trajectory 不加入 corpus。返回值保留已成功解析的数据，不把缺失文件当作成功证据。
    """
    trajectories: list[list[dict]] = []
    provenance: list[str] = []
    for root in roots:
        provenance.append(str(root))
        for session in sorted(Path(root).glob(SESSION_GLOB)):
            traj = []
            try:
                for line in session.open():
                    try:
                        traj.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            except OSError:
                continue
            if traj:
                trajectories.append(traj)
    return Corpus(trajectories=trajectories, provenance=provenance)


def run_chamber(node_id: str, spec: ActivationSpec, corpus: Corpus) -> ChamberReport:
    """在 ``corpus`` 上评估 ``spec``，生成 ``ChamberReport``。

    ``evaluate_spec`` 返回满足 activation predicate 的 trajectory 数量。函数不修改 corpus
    或 candidate，也不运行真实 benchmark；返回报告供 pre-flight gate 决策。
    """
    count = evaluate_spec(spec, corpus.trajectories)
    return ChamberReport(
        node_id=node_id,
        spec_kind=spec.kind,
        reachable_count=count,
        corpus_size=len(corpus.trajectories),
        provenance=corpus.provenance,
    )
