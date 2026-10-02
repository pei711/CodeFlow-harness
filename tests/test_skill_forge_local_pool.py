"""Tests for the BM25-backed local pool."""

from __future__ import annotations

import threading
from pathlib import Path

from codeflow.memory_engine.skill_local.local_pool import LocalPool
from codeflow.memory_engine.skill_local.registry import SkillRegistry
from codeflow.memory_engine.skill_local.types import SkillMeta


def _meta(
    name: str,
    description: str = "",
    body: str = "",
    source: str = "workspace",
) -> SkillMeta:
    return SkillMeta(
        id=f"{source}/{name}",
        name=name,
        description=description,
        path=Path(f"/tmp/{name}/SKILL.md"),
        content=body,
        source=source,
    )


class _StubRegistry:
    """Just enough of SkillRegistry to drive LocalPool from tests, without
    actually walking disk."""

    def __init__(self, metas: list[SkillMeta]) -> None:
        self._metas = list(metas)

    def list_all(self) -> list[SkillMeta]:
        return list(self._metas)

    def set_metas(self, metas: list[SkillMeta]) -> None:
        self._metas = list(metas)


# ----------------------------------------------------------------------


def test_localpool_empty_registry() -> None:
    pool = LocalPool(_StubRegistry([]))
    assert pool.search("anything", top_k=10) == []


def test_localpool_keyword_match_beats_unrelated() -> None:
    metas = [
        _meta("weather", description="get current weather and forecasts"),
        _meta("github", description="interact with GitHub via gh CLI"),
        _meta("pdf-gen", description="generate PDF reports with reportlab"),
    ]
    pool = LocalPool(_StubRegistry(metas))
    hits = pool.search("generate pdf", top_k=10)
    assert hits, "expected at least one BM25 hit"
    assert hits[0].name == "pdf-gen"


def test_localpool_zero_score_skills_are_dropped() -> None:
    metas = [
        _meta("a", description="alpha alpha"),
        _meta("b", description="beta beta"),
    ]
    pool = LocalPool(_StubRegistry(metas))
    assert pool.search("nothingmatches", top_k=10) == []


def test_localpool_top_k_truncation() -> None:
    metas = [_meta(f"skill_{i:02d}", description=f"skill number {i}", body="generate") for i in range(20)]
    pool = LocalPool(_StubRegistry(metas))
    hits = pool.search("generate", top_k=5)
    assert len(hits) == 5


def test_localpool_score_ordering_descending() -> None:
    metas = [
        _meta("foo", description="foo"),
        _meta("foo-bar", description="foo bar"),
        _meta("foo-bar-baz", description="foo bar baz"),
    ]
    pool = LocalPool(_StubRegistry(metas))
    hits = pool.search("bar", top_k=10)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_localpool_name_outweighs_buried_body_match() -> None:
    """A skill whose NAME matches should rank above one with the same term
    only buried deep in the body — that's the ``_format_skill_text`` weight
    decision."""
    metas = [
        _meta("pdf", description="generate pdf reports"),
        _meta("misc", description="utilities", body="word " * 500 + "pdf"),
    ]
    pool = LocalPool(_StubRegistry(metas))
    hits = pool.search("pdf", top_k=10)
    assert hits[0].name == "pdf"


def test_localpool_initial_build_is_eager() -> None:
    """``__init__`` builds the BM25 index from the current registry
    contents — first ``search`` must not pay the index-build cost."""
    metas = [_meta("weather", description="get current weather")]
    pool = LocalPool(_StubRegistry(metas))
    assert pool._bm25 is not None
    assert [m.name for m in pool._metas] == ["weather"]


def test_localpool_rebuild_index_picks_up_new_skills() -> None:
    """After ``rebuild_index()``, new skills must be searchable and the
    underlying ``_BM25Okapi`` must be a fresh instance."""
    metas = [_meta("weather", description="get current weather")]
    registry = _StubRegistry(metas)
    pool = LocalPool(registry)
    first_index = pool._bm25

    registry.set_metas(
        [
            _meta("weather", description="get current weather"),
            _meta("pdf-gen", description="generate pdf reports"),
        ]
    )
    pool.rebuild_index()
    hits = pool.search("pdf", top_k=10)

    assert pool._bm25 is not first_index, "rebuild_index left the old _BM25Okapi in place"
    assert hits and hits[0].name == "pdf-gen"


def test_localpool_concurrent_search_safe_during_rebuild() -> None:
    """Searches running concurrently with ``rebuild_index`` must never
    crash or return torn results. Each search either sees the pre-rebuild
    index or the post-rebuild one — never a half-swapped state.
    """
    metas = [_meta(f"skill_{i:02d}", description=f"alpha beta gamma {i}") for i in range(30)]
    pool = LocalPool(_StubRegistry(metas))

    n_threads = 16
    barrier = threading.Barrier(n_threads + 1)
    errors: list[BaseException] = []

    def searcher() -> None:
        try:
            barrier.wait()
            for _ in range(50):
                pool.search("alpha", top_k=10)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    def rebuilder() -> None:
        try:
            barrier.wait()
            for _ in range(50):
                pool.rebuild_index()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=searcher) for _ in range(n_threads)]
    threads.append(threading.Thread(target=rebuilder))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent search/rebuild crashed: {errors}"


def test_localpool_excludes_always_skills() -> None:
    """``always: true`` skills are auto-injected via ``get_always_skills`` →
    ``# Active Skills`` block. Letting BM25 ALSO surface them would
    double-inject the same body. Verify they're filtered out of search
    results even when the query matches their name.
    """
    metas = [
        _meta("memory", description="grep-based recall", body="memory recall"),
        _meta("self-improving", description="learning capture", body="capture learnings"),
        _meta("pdf-gen", description="generate pdf files", body="generate pdf with reportlab"),
    ]
    metas[0].always = True
    metas[1].always = True
    pool = LocalPool(_StubRegistry(metas))

    hits = pool.search("memory", top_k=10)
    assert all(h.name != "memory" for h in hits), (
        f"always-true skill 'memory' leaked into BM25 results: {[h.name for h in hits]}"
    )
    assert all(h.name != "self-improving" for h in hits)

    hits = pool.search("pdf", top_k=10)
    assert hits and hits[0].name == "pdf-gen"


def test_localpool_against_real_skill_registry(tmp_path: Path) -> None:
    """Smoke test against a tiny on-disk SKILL.md tree."""
    workspace = tmp_path / "ws"
    skills_root = workspace / "skills"
    skills_root.mkdir(parents=True)

    for name, body in [
        ("pdf-gen", "generate pdf with reportlab"),
        ("weather-query", "get current weather"),
    ]:
        d = skills_root / name
        d.mkdir()
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {body}\n---\n\n# {name}\n\n{body}",
            encoding="utf-8",
        )

    builtin = tmp_path / "builtin"
    builtin.mkdir()
    registry = SkillRegistry(workspace, builtin_skills_dir=builtin)
    pool = LocalPool(registry)

    hits = pool.search("pdf", top_k=10)
    assert hits and hits[0].name == "pdf-gen"
    hits = pool.search("weather", top_k=10)
    assert hits and hits[0].name == "weather-query"
