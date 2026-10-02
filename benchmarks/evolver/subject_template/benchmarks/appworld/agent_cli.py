"""Text normalisation helpers - the evolvable surface of this subject repo.

This module lives at ``benchmarks/appworld/agent_cli.py`` because that path is a
hard contract, not a naming preference. ``codeflow.evolver.candidate_manifest``
ships exactly one supported Candidate Label, ``runtime``, and its
``LabelPolicy.mutable_paths`` allowlist is
``("benchmarks/appworld/agent_cli.py", "benchmarks/appworld/tool.py")``. G5 runs
before a candidate commit exists and rejects any target outside that allowlist,
so a subject module at any other path would make every candidate fail G5 with no
commit to inspect.

The code that grades this module lives under ``benchmarks/appworld/evolve/``,
which the evolver immutable kernel forbids any candidate from touching.

Everything below is deterministic and stdlib-only: the same input always yields
the same output, so a K-trial evaluation measures the code, not the weather.
"""

from __future__ import annotations

_UNIT_SECONDS = {"s": 1, "m": 60}


def slugify(text: str) -> str:
    """Return a URL-safe slug for ``text``.

    Letters and digits are kept and lowercased; every other character becomes a
    separator.
    """
    lowered = text.strip().lower()
    return "".join(ch if ch.isalnum() else "-" for ch in lowered)


def parse_duration(text: str) -> int:
    """Return the total number of seconds described by ``text``.

    ``text`` is a whitespace- or comma-separated list of ``<value><unit>``
    chunks, for example ``"1h 30m"`` or ``"45s"``.
    """
    total = 0
    for chunk in text.replace(",", " ").split():
        value, unit = chunk[:-1], chunk[-1:]
        seconds = _UNIT_SECONDS.get(unit)
        if seconds is None:
            continue
        try:
            total += int(value) * seconds
        except ValueError:
            continue
    return total


def normalize_number(text: str) -> float:
    """Return the numeric value described by ``text``.

    Handles a leading currency marker and a trailing percent sign. Returns
    ``0.0`` when ``text`` carries no parseable number.
    """
    cleaned = text.strip().lstrip("$").rstrip("%")
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


__all__ = ["normalize_number", "parse_duration", "slugify"]
