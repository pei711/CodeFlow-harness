from pathlib import Path

from scripts.check_public_tree import check_public_tree


def test_current_tracked_tree_is_publication_safe() -> None:
    assert check_public_tree(Path(__file__).resolve().parents[1]) == []


def test_internal_document_and_private_repository_link_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("internal policy", encoding="utf-8")
    onboarding = tmp_path / "docs" / "onboarding"
    onboarding.mkdir(parents=True)
    (onboarding / "README.md").write_text(
        "https://github.com/" + "Hackerismydream/" + "myna",
        encoding="utf-8",
    )

    findings = check_public_tree(
        tmp_path,
        tracked_paths=["AGENTS.md", "docs/onboarding/README.md"],
    )

    assert "forbidden path: AGENTS.md" in findings
    assert any("private repository reference" in finding for finding in findings)


def test_docs_are_limited_to_onboarding_evaluation_and_public_fixtures(tmp_path: Path) -> None:
    internal = tmp_path / "docs" / "plan" / "roadmap.md"
    internal.parent.mkdir(parents=True)
    internal.write_text("internal", encoding="utf-8")

    assert check_public_tree(tmp_path, tracked_paths=["docs/plan/roadmap.md"]) == [
        "forbidden documentation path: docs/plan/roadmap.md"
    ]


def test_only_public_github_contribution_templates_are_allowed(tmp_path: Path) -> None:
    github = tmp_path / ".github"
    github.mkdir()
    issue_template = github / "ISSUE_TEMPLATE.zh-CN.md"
    issue_template.write_text("public issue template", encoding="utf-8")
    private_note = github / "maintainer-notes.md"
    private_note.write_text("internal", encoding="utf-8")

    assert check_public_tree(
        tmp_path,
        tracked_paths=[
            ".github/ISSUE_TEMPLATE.zh-CN.md",
            ".github/maintainer-notes.md",
        ],
    ) == ["forbidden GitHub metadata path: .github/maintainer-notes.md"]


def test_secret_bearing_file_extensions_are_rejected(tmp_path: Path) -> None:
    key = tmp_path / "tests" / "fixtures" / "live.pem"
    key.parent.mkdir(parents=True)
    key.write_text("not a real key", encoding="utf-8")

    assert check_public_tree(tmp_path, tracked_paths=["tests/fixtures/live.pem"]) == [
        "forbidden secret-bearing file: tests/fixtures/live.pem"
    ]


def test_raw_benchmark_results_and_real_environment_tests_are_rejected(tmp_path: Path) -> None:
    result = tmp_path / "benchmarks" / "new_pack" / "results" / "raw-run.json"
    result.parent.mkdir(parents=True)
    result.write_text("{}", encoding="utf-8")
    real_test = tmp_path / "tests" / "integration" / "test_private_real_provider.py"
    real_test.parent.mkdir(parents=True)
    real_test.write_text("", encoding="utf-8")

    assert check_public_tree(
        tmp_path,
        tracked_paths=[
            "benchmarks/new_pack/results/raw-run.json",
            "tests/integration/test_private_real_provider.py",
        ],
    ) == [
        "forbidden benchmark artifact path: benchmarks/new_pack/results/raw-run.json",
        "forbidden real-environment test path: tests/integration/test_private_real_provider.py",
    ]
