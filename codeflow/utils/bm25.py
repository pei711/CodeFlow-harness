"""面向 Small In-memory Corpora 的 Dependency-free BM25 Keyword Retrieval。

模块提供 Self-contained Okapi BM25，不依赖 ``rank_bm25``、``jieba`` 或 ``nltk``，并附带 CJK-aware
Tokenizer。它适合在几百篇短 Documents 上做低成本 Keyword Ranking，例如 File-based Skills 与 Tool
Catalogs；不是面向大规模全文索引的搜索引擎。

Tokenization 按 Word Boundaries 提取长度至少为二的字母数字串，并把每个 Chinese Ideograph 视为单独
Token，因此中文 Query 能产生匹配，不会 Collapse to Empty。逐字切分简单可预测，但不理解中文词语
边界或语义同义关系。
"""

from __future__ import annotations

import math
import re

# 匹配长度至少为 2 的字母数字串，或单个 CJK 表意字符。
# 在模块级预编译 ``re``，避免每次调用都初始化正则表达式。
_TOKEN_RE = re.compile(r"[a-z0-9]{2,}|[一-鿿]")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Okapi:
    """Minimal Okapi BM25，采用与 `rank_bm25` / Lucene Defaults 相同的 Formula。

    ``score(D, Q) = Σ idf(q_i) * f(q_i, D) * (k1 + 1)
                          / (f(q_i, D) + k1 * (1 - b + b * |D| / avgdl))``

    初始化时从 Tokenized Corpus 计算 Document Length、Term Frequency 与 IDF；`get_scores` 再为每个
    Query Token 累加所有文档得分。返回列表与输入 Corpus 保持同一顺序，数值越大表示关键词相关性
    越强。空 Corpus 或空 Query 全部返回零；分数是词项排序依据，不证明文档能完成 Agent Task。
    """

    def __init__(
        self,
        tokenized_corpus: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = len(tokenized_corpus)
        self.doc_lens = [len(d) for d in tokenized_corpus]
        self.avgdl = sum(self.doc_lens) / self.corpus_size if self.corpus_size else 0.0

        self.doc_freqs: list[dict[str, int]] = []
        df: dict[str, int] = {}
        for doc in tokenized_corpus:
            freqs: dict[str, int] = {}
            for tok in doc:
                freqs[tok] = freqs.get(tok, 0) + 1
            self.doc_freqs.append(freqs)
            for tok in freqs:
                df[tok] = df.get(tok, 0) + 1

        n = self.corpus_size
        # ``log(1 + (N - n + 0.5) / (n + 0.5))``——Robertson-Spärck-Jones 公式
        # ``1 +`` 保护项可在 n ≈ N 时确保权重不为负数。
        self.idf = {term: math.log(1 + (n - count + 0.5) / (count + 0.5)) for term, count in df.items()}

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.corpus_size
        if not query_tokens or self.corpus_size == 0:
            return scores
        for term in query_tokens:
            idf = self.idf.get(term, 0.0)
            if idf <= 0.0:
                continue
            for i, freqs in enumerate(self.doc_freqs):
                f = freqs.get(term, 0)
                if f == 0:
                    continue
                dl = self.doc_lens[i]
                norm = self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                scores[i] += idf * f * (self.k1 + 1) / (f + norm)
        return scores
