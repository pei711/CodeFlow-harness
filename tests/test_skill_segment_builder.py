"""End-to-end tests for the local-only SkillsSegmentBuilder path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from codeflow.context_engine.base import AssemblyContext, TokenBudget
from codeflow.context_engine.segments.skills import SkillsSegmentBuilder
from codeflow.memory_engine.skill_forge import SkillForgeRouter
from codeflow.memory_engine.skill_forge.types import RouterHit

# ----------------------------------------------------------------------

# ----------------------------------------------------------------------


class _StubSource:
    """SkillSource that returns a hard-coded hit list."""

    def __init__(self, name: str, hits: list[RouterHit], weight: float = 1.0) -> None:
        self.name = name
        self.weight = weight
        self._hits = hits

    async def search(
        self,
        query: str,
        history: list[dict[str, Any]],
        k: int,
    ) -> list[RouterHit]:
        return list(self._hits[:k])


def _hit(qid: str, name: str, body: str = "", **meta: Any) -> RouterHit:
    source = qid.split("/", 1)[0]
    meta.setdefault("source", source)
    return RouterHit(
        qualified_id=qid,
        name=name,
        content=body,
        score=0.5,
        meta=meta,
    )


def _ctx(message: str) -> AssemblyContext:
    return AssemblyContext(
        session_key="s",
        current_message=message,
        media=None,
        channel=None,
        chat_id=None,
        session_messages=[],
        budget=TokenBudget(
            context_length=200_000,
            reserved_output=8000,
            reserved_tools=4000,
            reserved_system=4000,
            available_history=184_000,
        ),
    )


# ----------------------------------------------------------------------

# ----------------------------------------------------------------------


async def test_baseline_renders_local_hits() -> None:
    src = _StubSource(
        "local",
        [
            _hit("local/foo", "foo", body="body foo"),
            _hit("local/bar", "bar", body="body bar"),
        ],
    )
    router = SkillForgeRouter([src])
    builder = SkillsSegmentBuilder(router, skill_top_k=2)
    seg = await builder.build(_ctx("anything"))
    assert seg is not None
    assert "# Skills" in seg.text
    assert "foo" in seg.text and "bar" in seg.text
    assert seg.meta["injected_skill_ids"] == ["local/foo", "local/bar"]


async def test_no_router_returns_empty_segment() -> None:
    builder = SkillsSegmentBuilder(None)
    seg = await builder.build(_ctx("q"))
    assert seg.text == ""
    assert seg.meta["injected_skill_ids"] == []


async def test_ambiguous_local_hit_is_referenced_without_body(tmp_path: Path) -> None:
    skill_dir = tmp_path / "release-helper"
    skill_dir.mkdir()
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text("release body", encoding="utf-8")
    src = _StubSource(
        "local",
        [
            _hit(
                "local/release-helper",
                "release-helper",
                body="SECRET FULL BODY",
                description="Prepare and verify a production release",
                skill_dir=str(skill_dir),
            )
        ],
    )
    builder = SkillsSegmentBuilder(SkillForgeRouter([src]))

    seg = await builder.build(_ctx("verify production readiness"))

    assert "SECRET FULL BODY" not in seg.text
    assert "skill_read" in seg.text
    assert "release-helper" in seg.text
    assert seg.meta["injected_skill_ids"] == []
    assert seg.meta["referenced_skill_ids"] == ["local/release-helper"]


async def test_unrelated_local_hit_is_omitted(tmp_path: Path) -> None:
    skill_dir = tmp_path / "weather-helper"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("weather body", encoding="utf-8")
    src = _StubSource(
        "local",
        [
            _hit(
                "local/weather-helper",
                "weather-helper",
                body="weather forecasts and temperature",
                description="Look up the weather forecast",
                skill_dir=str(skill_dir),
            )
        ],
    )
    builder = SkillsSegmentBuilder(SkillForgeRouter([src]))

    seg = await builder.build(_ctx("fix the database migration"))

    assert seg.text == ""
    assert seg.meta["injected_skill_ids"] == []
    assert seg.meta["referenced_skill_ids"] == []


async def test_activation_limit_keeps_overflow_hits_as_references(tmp_path: Path) -> None:
    hits = []
    for name in ("pdf-tool", "chart-tool"):
        skill_dir = tmp_path / name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(f"{name} body", encoding="utf-8")
        hits.append(
            _hit(
                f"local/{name}",
                name,
                body=f"{name} body",
                description=f"Use {name}",
                skill_dir=str(skill_dir),
            )
        )
    builder = SkillsSegmentBuilder(
        SkillForgeRouter([_StubSource("local", hits)]),
        activation_max=1,
    )

    seg = await builder.build(_ctx("use pdf-tool and chart-tool"))

    assert seg.meta["injected_skill_ids"] == ["local/pdf-tool"]
    assert seg.meta["referenced_skill_ids"] == ["local/chart-tool"]


# ----------------------------------------------------------------------

# ----------------------------------------------------------------------


async def test_activated_skill_resolves_local_refs(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skill"
    (skill_dir / "references").mkdir(parents=True)
    (skill_dir / "references" / "x.md").write_text("ref body")

    body = "Read {baseDir}/references/x.md."
    src = _StubSource(
        "local",
        [
            _hit(
                "local/foo",
                "foo",
                body=body,
                skill_dir=str(skill_dir),
            )
        ],
    )
    builder = SkillsSegmentBuilder(SkillForgeRouter([src]), skill_top_k=1)
    seg = await builder.build(_ctx("use foo"))
    assert f"{skill_dir}/references/x.md" in seg.text
    assert "{baseDir}" not in seg.text
