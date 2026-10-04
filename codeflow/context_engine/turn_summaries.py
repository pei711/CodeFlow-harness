"""Bounded, project-local summaries and hybrid retrieval for completed turns.

The retrieval path intentionally has no required embedding dependency: BM25 is
always available, while FastEmbed supplies the semantic lane when installed.
Both ranked lists are fused by reciprocal rank fusion (RRF).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from codeflow.utils.atomic_io import locked_append, locked_read
from codeflow.utils.helpers import ensure_dir
from codeflow.utils.persisted_payload import sanitize_persisted_text

SUMMARY_LIMIT = 200
SUMMARY_CHARS = 800
RETRIEVAL_CANDIDATES = 10
RRF_K = 60
_MODEL = None
_MODEL_NAME = None
_MODEL_LOCK = threading.Lock()
_EMBEDDING_CACHE: dict[tuple[str, str], tuple[float, ...]] = {}
_EMBEDDING_CACHE_LIMIT = 512


def _clip(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def summarize_turn(user_message: Any, final_answer: Any) -> str:
    """Create a bounded extractive summary without an extra model call."""
    parts = []
    user = _clip(user_message, 260)
    answer = _clip(final_answer, 520)
    if user:
        parts.append(f"User goal: {user}")
    if answer:
        parts.append(f"Outcome: {answer}")
    return sanitize_persisted_text(_clip("\n".join(parts), SUMMARY_CHARS))


def _terms(text: str) -> list[str]:
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z0-9_]+|[\u3400-\u9fff]+", str(text).lower()):
        if re.fullmatch(r"[\u3400-\u9fff]+", token):
            if len(token) == 1:
                terms.append(f"c1:{token}")
            else:
                terms.extend(f"c2:{token[i:i + 2]}" for i in range(len(token) - 1))
        else:
            terms.append(f"w:{token}")
    return terms


def _bm25(documents: list[str], query: str) -> list[tuple[int, float]]:
    query_terms = _terms(query)
    docs = [_terms(doc) for doc in documents]
    if not query_terms or not docs:
        return []
    df: dict[str, int] = {}
    for tokens in docs:
        for token in set(tokens):
            df[token] = df.get(token, 0) + 1
    avg_len = sum(map(len, docs)) / len(docs) or 1.0
    ranked = []
    for index, tokens in enumerate(docs):
        frequencies: dict[str, int] = {}
        for token in tokens:
            frequencies[token] = frequencies.get(token, 0) + 1
        score = 0.0
        for token in set(query_terms):
            tf = frequencies.get(token, 0)
            if not tf:
                continue
            idf = math.log(1 + (len(docs) - df[token] + 0.5) / (df[token] + 0.5))
            score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * len(tokens) / avg_len))
        if score > 0:
            ranked.append((index, score))
    return sorted(ranked, key=lambda item: (item[1], item[0]), reverse=True)[:RETRIEVAL_CANDIDATES]


def _vector_terms(text: str) -> list[str]:
    return _terms(text)


def _tfidf_vector(tokens: list[str], df: dict[str, int], total: int) -> dict[str, float]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    length = len(tokens) or 1
    return {
        token: (count / length) * (math.log(1 + total / (1 + df.get(token, 0))) + 1)
        for token, count in counts.items()
    }


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return sum(value * right.get(key, 0.0) for key, value in left.items()) / (left_norm * right_norm)


def _semantic_rank(documents: list[str], query: str) -> list[tuple[int, float]]:
    global _MODEL, _MODEL_NAME
    import os

    model_name = os.environ.get("CODEFLOW_MEMORY_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    if _MODEL is None or _MODEL_NAME != model_name:
        with _MODEL_LOCK:
            if _MODEL is None or _MODEL_NAME != model_name:
                from fastembed import TextEmbedding

                _MODEL = TextEmbedding(model_name=model_name)
                _MODEL_NAME = model_name
    query_vector = _cached_embedding(_MODEL, model_name, query)
    ranked = []
    for index, document in enumerate(documents):
        vector = _cached_embedding(_MODEL, model_name, document)
        dot = sum(float(a) * float(b) for a, b in zip(query_vector, vector))
        left = math.sqrt(sum(float(a) ** 2 for a in query_vector))
        right = math.sqrt(sum(float(a) ** 2 for a in vector))
        score = dot / (left * right) if left and right else 0.0
        if score > 0:
            ranked.append((index, score))
    return sorted(ranked, key=lambda item: item[1], reverse=True)[:RETRIEVAL_CANDIDATES]


def _cached_embedding(model: Any, model_name: str, text: str) -> tuple[float, ...]:
    cache_key = (model_name, hashlib.sha256(text.encode("utf-8")).hexdigest())
    cached = _EMBEDDING_CACHE.get(cache_key)
    if cached is not None:
        return cached
    vector = next(iter(model.embed([text])), None)
    if vector is None:
        raise RuntimeError("embedding model returned no vector")
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    result = tuple(float(value) for value in vector)
    _EMBEDDING_CACHE[cache_key] = result
    while len(_EMBEDDING_CACHE) > _EMBEDDING_CACHE_LIMIT:
        _EMBEDDING_CACHE.pop(next(iter(_EMBEDDING_CACHE)))
    return result


def _fuse(summaries: list[dict[str, Any]], vector_hits, keyword_hits, limit: int) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, Any]] = {}
    representatives: dict[str, int] = {}
    for source, hits in (("semantic", vector_hits), ("bm25", keyword_hits)):
        seen_in_source: set[str] = set()
        for rank, (index, score) in enumerate(hits, 1):
            identity = " ".join(str(summaries[index].get("text", "")).split()).casefold()
            if not identity or identity in seen_in_source:
                continue
            seen_in_source.add(identity)
            representatives.setdefault(identity, index)
            scores[identity] = scores.get(identity, 0.0) + 1.0 / (RRF_K + rank)
            ranks.setdefault(identity, {})[source] = {"rank": rank, "score": round(float(score), 6)}
    ordered = sorted(
        scores,
        key=lambda identity: (scores[identity], summaries[representatives[identity]].get("created_at", "")),
        reverse=True,
    )
    return [
        {
            "text": summaries[representatives[identity]]["text"],
            "summary_id": summaries[representatives[identity]]["summary_id"],
            "retrieval": {
                "mode": "hybrid_rrf",
                "rrf_score": round(scores[identity], 6),
                **ranks[identity],
            },
        }
        for identity in ordered[: max(0, limit)]
    ]


class TurnSummaryStore:
    """Append-only project-scoped turn summaries with bounded hybrid recall."""

    def __init__(self, state_root: Path):
        self.path = ensure_dir(state_root / "context_memory") / "turn_summaries.jsonl"
        self._turn_ids: dict[str, str] = {}

    def add(self, session_key: str, turn_id: str, user_message: str, answer: str) -> None:
        text = summarize_turn(user_message, answer)
        if not text:
            return
        created_at = datetime.now().isoformat()
        turn_id = str(turn_id or created_at)
        identity = hashlib.sha256(f"{session_key}\0{turn_id}\0{text}".encode()).hexdigest()[:20]
        raw, _, _ = locked_read(self.path)
        if raw:
            for line in raw.splitlines():
                try:
                    if json.loads(line).get("summary_id") == identity:
                        return
                except (json.JSONDecodeError, AttributeError):
                    continue
        record = {
            "summary_id": identity,
            "session_key": session_key,
            "turn_id": turn_id,
            "created_at": created_at,
            "text": text,
        }
        locked_append(self.path, [json.dumps(record, ensure_ascii=False)])

    def recent(self) -> list[dict[str, Any]]:
        raw, _, _ = locked_read(self.path)
        if not raw:
            return []
        records = []
        for line in raw.splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and isinstance(item.get("text"), str) and item.get("summary_id"):
                records.append(item)
        return records[-SUMMARY_LIMIT:]

    def retrieve(self, query: str, limit: int = 3) -> tuple[list[dict[str, Any]], str]:
        summaries = self.recent()
        if not summaries or not _terms(query):
            return [], "none"
        docs = [item["text"] for item in summaries]
        keywords = _bm25(docs, query)
        try:
            vectors = _semantic_rank(docs, query)
            mode = "fastembed"
        except Exception:
            tokenized = [_vector_terms(doc) for doc in docs]
            df: dict[str, int] = {}
            for tokens in tokenized:
                for token in set(tokens):
                    df[token] = df.get(token, 0) + 1
            qv = _tfidf_vector(_vector_terms(query), df, len(docs))
            vectors = [
                (i, _cosine(qv, _tfidf_vector(tokens, df, len(docs)))) for i, tokens in enumerate(tokenized)
            ]
            vectors = sorted((item for item in vectors if item[1] > 0), key=lambda item: item[1], reverse=True)[
                :RETRIEVAL_CANDIDATES
            ]
            mode = "tfidf_fallback"
        return _fuse(summaries, vectors, keywords, limit), mode


__all__ = ["TurnSummaryStore", "summarize_turn"]
