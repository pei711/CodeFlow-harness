"""Deterministic gate for the small-real Evolution Run subject.

Everything here runs offline and spends no model call: the setup script, the
subject's own bench plugin, the subprocess grader, the G5 manifest binding, and
one complete funnel round driven by a scripted design response. A real run of
the same config is separate evidence and is never implied by this file.

Import isolation matters: the host repo ships its own ``benchmarks.appworld``
package, and the subject checkout ships a different one at the same import
path. The subject modules are therefore imported exactly the way the launcher
imports them (subject root first on ``sys.path``), with the host's
``benchmarks`` entries removed from ``sys.modules`` for the duration and put
back afterwards, so no other test in the same process sees the subject's copy.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from codeflow.evolver.candidate_manifest import LABEL_POLICIES, CandidateLabel, ManifestGateError
from codeflow.evolver.launch.config import load_run_spec
from codeflow.evolver.launch.contract import LaunchContext
from codeflow.evolver.launch.registry import load_bench
from codeflow.evolver.orchestrator.scoring import EvaluationVerdict
from codeflow.evolver.orchestrator.state.journal import RoundJournal
from codeflow.evolver.tree import git_ops

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = REPO_ROOT / "benchmarks" / "evolver" / "subject_template"
SETUP_SCRIPT = REPO_ROOT / "scripts" / "setup_small_real_subject.py"

MODULE_PATH = "benchmarks/appworld/agent_cli.py"

VANILLA_TRAIN_PASSES = ("duration-minutes", "number-decimal", "number-percent", "slug-basic")
VANILLA_TRAIN_FAILURES = (
    "duration-hours",
    "duration-mixed",
    "number-grouped",
    "number-grouped-int",
    "slug-collapse",
    "slug-trim",
)
VANILLA_TEST_PASSES = ("sealed-slug-plain",)

CORRECTED_MODULE = '''"""Corrected subject module: all three seeded defects repaired."""

from __future__ import annotations

_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600}


def slugify(text: str) -> str:
    slug = "".join(ch if ch.isalnum() else "-" for ch in text.strip().lower())
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def parse_duration(text: str) -> int:
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
    cleaned = text.strip().lstrip("$").rstrip("%").replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


__all__ = ["normalize_number", "parse_duration", "slugify"]
'''


def _run_setup(subject: Path, *flags: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SETUP_SCRIPT),
            "--template",
            str(TEMPLATE),
            "--subject",
            str(subject),
            *flags,
        ],
        capture_output=True,
        text=True,
    )


def _git(subject: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(subject), *args], capture_output=True, text=True, check=True)
    return proc.stdout.strip()


@dataclass
class SubjectBench:
    """A materialized subject plus its import-isolated bench modules."""

    root: Path
    head: str
    build: Any
    adapter: Any
    candidate: Any
    designer: Any
    tasks: Any


@contextlib.contextmanager
def _subject_bench_modules(subject: Path):
    saved = {name: mod for name, mod in sys.modules.items() if name == "benchmarks" or name.startswith("benchmarks.")}
    for name in saved:
        del sys.modules[name]
    root = str(subject.resolve())
    try:
        build = load_bench("appworld", repo_root=subject)
        yield {
            "build": build,
            "adapter": importlib.import_module("benchmarks.appworld.evolve.adapter"),
            "candidate": importlib.import_module("benchmarks.appworld.evolve.candidate"),
            "designer": importlib.import_module("benchmarks.appworld.evolve.designer"),
            "tasks": importlib.import_module("benchmarks.appworld.evolve.tasks"),
        }
    finally:
        for name in [n for n in sys.modules if n == "benchmarks" or n.startswith("benchmarks.")]:
            del sys.modules[name]
        sys.modules.update(saved)
        if root in sys.path:
            sys.path.remove(root)


@pytest.fixture(scope="module")
def bench(tmp_path_factory) -> SubjectBench:
    subject = tmp_path_factory.mktemp("small-real") / "subject"
    proc = _run_setup(subject)
    assert proc.returncode == 0, proc.stderr
    with _subject_bench_modules(subject) as modules:
        yield SubjectBench(root=subject, head=_git(subject, "rev-parse", "HEAD"), **modules)


def test_setup_script_materializes_a_deterministic_subject(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"

    assert _run_setup(first).returncode == 0
    assert _run_setup(second).returncode == 0

    assert _git(first, "rev-parse", "HEAD") == _git(second, "rev-parse", "HEAD")
    assert _git(first, "status", "--porcelain", "-uall") == ""
    tracked = sorted(_git(first, "ls-tree", "-r", "--name-only", "HEAD").splitlines())
    expected = sorted(
        str(path.relative_to(TEMPLATE))
        for path in TEMPLATE.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    assert tracked == expected
    assert MODULE_PATH in tracked


def test_setup_script_rerun_needs_an_explicit_mode(tmp_path: Path) -> None:
    subject = tmp_path / "subject"
    assert _run_setup(subject).returncode == 0
    head = _git(subject, "rev-parse", "HEAD")

    bare = _run_setup(subject)
    assert bare.returncode == 2
    assert "--reuse" in bare.stderr and "--recreate" in bare.stderr

    reuse = _run_setup(subject, "--reuse")
    assert reuse.returncode == 0
    assert _git(subject, "rev-parse", "HEAD") == head

    recreate = _run_setup(subject, "--recreate")
    assert recreate.returncode == 0
    assert _git(subject, "rev-parse", "HEAD") == head


def test_setup_script_refuses_a_dirty_or_foreign_subject(tmp_path: Path) -> None:
    subject = tmp_path / "subject"
    assert _run_setup(subject).returncode == 0
    edited = subject / MODULE_PATH
    edited.write_text(edited.read_text() + "\n# local edit\n")

    for flags in ((), ("--reuse",), ("--recreate",)):
        proc = _run_setup(subject, *flags)
        assert proc.returncode == 1, flags
        assert "refusing" in proc.stderr
    assert edited.read_text().endswith("# local edit\n")

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "notes.txt").write_text("not a checkout\n")
    proc = _run_setup(foreign)
    assert proc.returncode == 1
    assert "not a git repo" in proc.stderr
    assert (foreign / "notes.txt").is_file()


def test_setup_script_recreate_refuses_clean_non_template_repo(tmp_path: Path) -> None:
    subject = tmp_path / "foreign-clean"
    subject.mkdir()
    subprocess.run(["git", "-C", str(subject), "init", "-q", "-b", "main"], check=True)
    (subject / "notes.txt").write_text("keep me\n")
    subprocess.run(["git", "-C", str(subject), "add", "notes.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(subject),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-q",
            "-m",
            "test: foreign repo",
        ],
        check=True,
    )

    proc = _run_setup(subject, "--recreate")

    assert proc.returncode == 1
    assert "not an exact checkout" in proc.stderr
    assert (subject / "notes.txt").read_text() == "keep me\n"


def test_registry_evicts_cached_host_benchmark_modules(bench: SubjectBench) -> None:
    names = [name for name in sys.modules if name == "benchmarks" or name.startswith("benchmarks.")]
    saved = {name: sys.modules[name] for name in names}
    try:
        for name in names:
            del sys.modules[name]
        package = types.ModuleType("benchmarks")
        package.__path__ = [str(REPO_ROOT / "benchmarks")]
        package.__file__ = str(REPO_ROOT / "benchmarks" / "__init__.py")
        sys.modules["benchmarks"] = package

        build = load_bench("appworld", repo_root=bench.root)

        loaded = sys.modules[build.__module__]
        assert Path(loaded.__file__).resolve().is_relative_to(bench.root.resolve())
    finally:
        for name in [name for name in sys.modules if name == "benchmarks" or name.startswith("benchmarks.")]:
            del sys.modules[name]
        sys.modules.update(saved)


def test_parse_module_block_accepts_exactly_one_valid_block(bench: SubjectBench) -> None:
    parse = bench.designer.parse_module_block
    surface = bench.tasks.PUBLIC_SURFACE

    assert parse(f"here you go\n\n```python\n{CORRECTED_MODULE}```\n", required_names=surface) == CORRECTED_MODULE

    with pytest.raises(bench.designer.DesignParseError, match="found 0"):
        parse(CORRECTED_MODULE, required_names=surface)
    with pytest.raises(bench.designer.DesignParseError, match="found 2"):
        parse(f"```python\n{CORRECTED_MODULE}```\nand also\n```python\n{CORRECTED_MODULE}```\n", required_names=surface)
    with pytest.raises(bench.designer.DesignParseError, match="empty"):
        parse("```python\n\n```\n", required_names=surface)

    oversized = "X = 1\n" + "# pad\n" * bench.designer.MAX_MODULE_BYTES
    with pytest.raises(bench.designer.DesignParseError, match="byte cap"):
        parse(f"```python\n{oversized}```")
    with pytest.raises(bench.designer.DesignParseError, match="not valid Python"):
        parse("```\nthis is prose, not a module\n```")
    with pytest.raises(bench.designer.DesignParseError, match="must still define"):
        parse("```python\ndef slugify(text):\n    return text\n```", required_names=surface)


def test_vanilla_scoring_fails_exactly_the_seeded_defect_tasks(bench: SubjectBench, tmp_path: Path) -> None:
    out_dir = tmp_path / "vanilla"
    train_ids = bench.tasks.train_task_ids()
    test_ids = bench.tasks.test_task_ids()
    bench.adapter.score_trials(bench.root, bench.head, out_dir, train_ids + test_ids, 2)

    evals = bench.adapter.read_out_dir(out_dir)
    assert sorted(evals) == sorted(train_ids + test_ids)
    assert all(ev.attempts == 2 and ev.infra_attempts == 0 for ev in evals.values())
    passing = sorted(tid for tid, ev in evals.items() if ev.passes == ev.attempts)
    failing = sorted(tid for tid, ev in evals.items() if ev.passes == 0)
    assert passing == sorted(VANILLA_TRAIN_PASSES + VANILLA_TEST_PASSES)
    assert failing == sorted(VANILLA_TRAIN_FAILURES + tuple(t for t in test_ids if t not in VANILLA_TEST_PASSES))

    failures = bench.adapter.read_case_failures(out_dir, train_ids)
    assert sorted(failures) == sorted(VANILLA_TRAIN_FAILURES)
    assert all(case.get("ok") is False for cases in failures.values() for case in cases)


def test_corrected_module_passes_every_task(bench: SubjectBench, tmp_path: Path) -> None:
    child_sha, changed = git_ops.commit_files_as_child(
        bench.root,
        bench.head,
        {MODULE_PATH: CORRECTED_MODULE.encode("utf-8")},
        "test: corrected subject module",
    )
    assert changed == [MODULE_PATH]

    out_dir = tmp_path / "corrected"
    task_ids = bench.tasks.train_task_ids() + bench.tasks.test_task_ids()
    bench.adapter.score_trials(bench.root, child_sha, out_dir, task_ids, 2)

    evals = bench.adapter.read_out_dir(out_dir)
    assert sorted(evals) == sorted(task_ids)
    assert all(ev.passes == 2 and ev.infra_attempts == 0 for ev in evals.values())
    assert _git(bench.root, "rev-parse", "HEAD") == bench.head
    assert _git(bench.root, "status", "--porcelain", "-uall") == ""


@pytest.mark.parametrize(
    "malicious",
    [
        b"""import __main__\n__main__._matches = lambda actual, expect: True\n\ndef slugify(text):\n    return 'wrong'\n\ndef parse_duration(text):\n    return -1\n\ndef normalize_number(text):\n    return -1.0\n""",
        b"""def slugify(text):\n    open('/tmp/small-real-escape', 'w').write(text)\n\ndef parse_duration(text):\n    return 0\n\ndef normalize_number(text):\n    return 0.0\n""",
        b"""raise SystemExit(0)\n\ndef slugify(text):\n    return text\n\ndef parse_duration(text):\n    return 0\n\ndef normalize_number(text):\n    return 0.0\n""",
    ],
)
def test_grader_rejects_candidate_escape_attempts(bench: SubjectBench, tmp_path: Path, malicious: bytes) -> None:
    child_sha, _ = git_ops.commit_files_as_child(
        bench.root,
        bench.head,
        {MODULE_PATH: malicious},
        "test: malicious subject module",
    )
    out_dir = tmp_path / "malicious"

    bench.adapter.score_trials(bench.root, child_sha, out_dir, ["duration-hours"], 1)

    record = json.loads((out_dir / "duration-hours_k0.json").read_text())
    assert record["success"] is False
    assert record["infra_error"] is None
    assert "module import failed" in record["detail"]


def test_score_trials_resumes_instead_of_rescoring(bench: SubjectBench, tmp_path: Path) -> None:
    out_dir = tmp_path / "resume"
    train_ids = bench.tasks.train_task_ids()
    bench.adapter.score_trials(bench.root, bench.head, out_dir, train_ids, 1)
    assert len(list(out_dir.glob("*_k*.json"))) == len(train_ids)

    kept = out_dir / f"{train_ids[0]}_k0.json"
    record = json.loads(kept.read_text())
    record["marker"] = "not recomputed"
    kept.write_text(json.dumps(record))
    infra = out_dir / f"{train_ids[1]}_k0.json"
    infra.write_text(json.dumps({"task_id": train_ids[1], "k": 0, "success": False, "infra_error": "boom"}))

    bench.adapter.score_trials(bench.root, bench.head, out_dir, train_ids, 1)

    assert json.loads(kept.read_text())["marker"] == "not recomputed"
    assert json.loads(infra.read_text())["infra_error"] is None
    assert len(list(out_dir.glob("*_k*.json"))) == len(train_ids)
    assert bench.adapter.read_out_dir(out_dir)[train_ids[1]].infra_attempts == 0


def test_unreadable_module_is_marked_infrastructure_not_failure(bench: SubjectBench, tmp_path: Path) -> None:
    out_dir = tmp_path / "infra"
    bench.adapter.score_trials(bench.root, "0" * 40, out_dir, ["slug-basic"], 2)

    records = [json.loads(path.read_text()) for path in sorted(out_dir.glob("*_k*.json"))]
    assert len(records) == 2
    assert all(record["infra_error"] and record["success"] is False for record in records)
    ev = bench.adapter.read_out_dir(out_dir)["slug-basic"]
    assert (ev.attempts, ev.passes, ev.infra_attempts) == (2, 0, 2)


def test_manifest_pins_the_runtime_label_to_the_subject_surface(bench: SubjectBench) -> None:
    assert MODULE_PATH in LABEL_POLICIES[CandidateLabel.runtime].mutable_paths

    candidate = bench.candidate.Candidate(
        files={MODULE_PATH: CORRECTED_MODULE.encode("utf-8")},
        why="duration_hour_unit",
        summary="repair the hour unit",
    )
    manifest = bench.candidate.prepare_candidate_manifest("cand-surface", bench.head, candidate, repo_root=bench.root)

    assert manifest.target_files == (MODULE_PATH,)
    assert manifest.label is CandidateLabel.runtime
    assert candidate.applied_patch.components[0].target_file == MODULE_PATH


@pytest.mark.parametrize("path", ["benchmarks/appworld/tool.py", "benchmarks/appworld/evolve/grade.py"])
def test_manifest_gate_rejects_a_target_outside_the_subject_surface(bench: SubjectBench, path: str) -> None:
    before = _git(bench.root, "rev-list", "--all", "--count")
    candidate = bench.candidate.Candidate(files={path: b"VALUE = 1\n"}, why="duration_hour_unit")

    with pytest.raises(ManifestGateError, match="G5 manifest gate failed"):
        bench.candidate.prepare_candidate_manifest("cand-outside", bench.head, candidate, repo_root=bench.root)

    assert candidate.manifest is None
    assert _git(bench.root, "rev-list", "--all", "--count") == before


def test_scripted_round_accepts_the_corrected_module(bench: SubjectBench, tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    config = tmp_path / "small_real_test.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "bench": "appworld",
                "repo_root": str(bench.root),
                "base_sha": bench.head,
                "work_dir": str(work_dir),
                "funnel": {
                    "k_screen": 1,
                    "k_confirm": 2,
                    "budget": {"max_why_per_round": 1, "candidates_per_why": 1},
                    "termination": {"patience": 1, "max_rounds": 1},
                },
                "bench_config": {"timeout": 120},
            }
        )
    )
    prompts: list[list[dict]] = []

    def design_call_fn(messages: list) -> str:
        prompts.append(messages)
        return f"Here is the corrected module.\n\n```python\n{CORRECTED_MODULE}```\n"

    spec = load_run_spec(config)
    bundle = bench.build(LaunchContext(spec=spec, models={"design": design_call_fn, "driver": None, "verdict": None}))
    assert bundle.unseal is not None
    assert bundle.cold_start_total == len(bench.tasks.train_task_ids()) * 2

    bundle.run_cold_start()
    assert bundle.cold_start_done() == bundle.cold_start_total

    journal = RoundJournal(bundle.journal_path)
    orchestrator = bundle.build_orchestrator()
    result = orchestrator.run("C0", journal, root_node=bundle.root_node)

    assert len(result.rounds) == 1
    outcome = result.rounds[0].outcomes[0]
    assert outcome.verdict is EvaluationVerdict.accepted
    assert result.rounds[0].promoted is True
    assert outcome.stats["full_lift"] == pytest.approx(0.6)

    assert len(prompts) == 1
    assert MODULE_PATH in prompts[0][1]["content"]

    node_record = json.loads((work_dir / "nodes" / f"{outcome.node_id}.json").read_text())
    assert node_record["verdict"] == "accepted"
    assert node_record["candidate"]["manifest"]["label"] == "runtime"
    assert node_record["candidate"]["manifest"]["target_files"] == [MODULE_PATH]

    activation = work_dir / "activation" / outcome.node_id
    assert (activation / "before.json").read_bytes() == (activation / "rollback.json").read_bytes()

    assert _git(bench.root, "rev-parse", "HEAD") == bench.head
    assert _git(bench.root, "status", "--porcelain", "-uall") == ""


def test_orchestrator_builds_without_model_roles_for_finalize(bench: SubjectBench, tmp_path: Path) -> None:
    config = tmp_path / "small_real_finalize.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "bench": "appworld",
                "repo_root": str(bench.root),
                "base_sha": bench.head,
                "work_dir": str(tmp_path / "work"),
                "funnel": {
                    "k_screen": 1,
                    "k_confirm": 2,
                    "budget": {"max_why_per_round": 1, "candidates_per_why": 1},
                    "termination": {"patience": 1, "max_rounds": 1},
                },
            }
        )
    )
    spec = load_run_spec(config)
    bundle = bench.build(LaunchContext(spec=spec, models={"design": None, "driver": None, "verdict": None}))

    bundle.run_cold_start()
    orchestrator = bundle.build_orchestrator()

    assert orchestrator.vanilla_train_mean == pytest.approx(0.4)
