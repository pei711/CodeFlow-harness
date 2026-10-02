.PHONY: help install install-deps format check check-public-tree lint lint-python lint-tui test test-python test-retained test-tui codeflowbench-smoke codeflowbench codeflowbench-reproduce codeflowbench-scorecard-estimate codeflowbench-scorecard-ship codeflowbench-scorecard-score codeflowbench-runtime-scheduler codeflowbench-runtime-tools codeflowbench-runtime-live-plan codeflowbench-runtime-live-run codeflowbench-runtime-live-verify codeflowbench-call-efficiency-plan codeflowbench-call-efficiency-preflight codeflowbench-call-efficiency-run codeflowbench-call-efficiency-verify codeflowbench-tracing-plan codeflowbench-tracing-run codeflowbench-tracing-verify verify-channels verify-evolver verify-turn-evidence build build-tui release-dist check-commits check-pr-title check-large-files ci clean

PYTHON ?= python3
PYTHON_LINT_TARGETS ?= scripts/check_commit_file.py scripts/check_commit_messages.py scripts/check_pr_title.py scripts/check_large_files.py scripts/commit_lint.py tests/test_commit_lint.py tests/test_large_file_check.py
COMMIT_RANGE ?= origin/main..HEAD
CODEFLOW_CALL_EFFICIENCY_OUTPUT ?= .codeflow/evidence/call-efficiency-cost-current
CODEFLOW_TRACING_OUTPUT ?= .codeflow/evidence/tracing-overhead-current

help:
	@echo "Targets:"
	@echo "  install        Install Python deps, Node deps, and git hooks"
	@echo "  install-deps   Install Python deps only (CI uses this)"
	@echo "  format         Format Python sources"
	@echo "  check          Run the complete local deterministic acceptance gate"
	@echo "  check-public-tree Fail if internal material crosses the publication boundary"
	@echo "  lint           Run Python and TUI lint gates"
	@echo "  lint-python    Ruff-check the current lint target set"
	@echo "  lint-tui       TypeScript lint + RPC drift check"
	@echo "  test           Run focused Python checks and TUI tests"
	@echo "  test-retained  Run the deterministic Python suite without opt-in tests"
	@echo "  codeflowbench-smoke Run the credential-free CodeFlowBench gate"
	@echo "  codeflowbench-runtime-scheduler Run deterministic scheduler A/B experiments"
	@echo "  codeflowbench-runtime-tools Run the Tool scheduler A/B microbenchmark"
	@echo "  codeflowbench-runtime-live-plan Freeze the real-Agent scheduler plan and spend ceiling"
	@echo "  codeflowbench-runtime-live-run Run the approved real-Agent scheduler experiment"
	@echo "  codeflowbench-runtime-live-verify Rebuild live scheduler metrics from raw Turn records"
	@echo "  codeflowbench-call-efficiency-plan Freeze the current integrated cost campaign plan"
	@echo "  codeflowbench-call-efficiency-preflight Verify live DeepSeek prompt-cache behavior"
	@echo "  codeflowbench-call-efficiency-run Run or resume the approved 72-Trial cost campaign"
	@echo "  codeflowbench-call-efficiency-verify Rebuild cost evidence without Provider calls"
	@echo "  codeflowbench-tracing-plan Print the 1,000-pair Runtime tracing plan"
	@echo "  codeflowbench-tracing-run Run or resume the deterministic tracing on/off campaign"
	@echo "  codeflowbench-tracing-verify Rebuild tracing metrics and verify raw trace receipts"
	@echo "  codeflowbench      Run the frozen CodeFlowBench calibration and formal campaign"
	@echo "  codeflowbench-reproduce Run or reuse every Scorecard track and render one report"
	@echo "  codeflowbench-scorecard-estimate Print the current Scorecard worst-case budget"
	@echo "  codeflowbench-scorecard-ship Run the current Context and Tool/MCP Scorecard campaign"
	@echo "  codeflowbench-scorecard-score Compute the multidimensional diagnostic score"
	@echo "  verify-channels Run the deterministic V-C0 contract and V-S0 security Channel gates"
	@echo "  verify-evolver Run the deterministic V-E0 Evolver gate"
	@echo "  verify-turn-evidence Run the deterministic V-TE0 turn-correlation gate"
	@echo "  release-dist   Build and verify a publishable wheel and source archive"
	@echo "  check-commits  Validate Conventional Commit subjects"
	@echo "  check-pr-title Validate the PR title in PR_TITLE"
	@echo "  check-large-files Validate PR files avoid blocked assets and size bloat"
	@echo "  ci             Run the local CI gate"
	@echo "  clean          Remove generated caches and build output"

install-deps:
	uv sync --frozen --extra dev --dev

install: install-deps
	uv run pre-commit install
	uv run pre-commit install --hook-type commit-msg
	npm ci
	npm ci --prefix ui-tui

format:
	uv run --extra dev ruff format codeflow scripts tests

check: check-public-tree lint build test-retained test-tui

check-public-tree:
	uv run python scripts/check_public_tree.py

lint: lint-python lint-tui

lint-python:
	uv run --extra dev ruff check $(PYTHON_LINT_TARGETS)
	uv run --extra dev ruff format --check $(PYTHON_LINT_TARGETS)

lint-tui:
	npm run lint --prefix ui-tui
	npm run lint:rpc --prefix ui-tui
	npm run lint:rpc-surface --prefix ui-tui
	npm run type-check --prefix ui-tui

test: test-python test-tui

test-python:
	uv run --extra dev pytest tests/test_commit_lint.py tests/test_large_file_check.py tests/test_cli_smoke.py tests/test_litellm_setup.py -q

test-retained: build-tui
	uv run --frozen --all-extras --exact pytest tests -q --strict-markers -m 'not (real_llm or llm_judge or real_vm or real_channel or external_runtime or e2e)'

codeflowbench-smoke:
	uv run --frozen --all-extras --exact python -m benchmarks.codeflowbench --mode smoke

codeflowbench:
	uv run --frozen --all-extras --exact python -m benchmarks.codeflowbench --mode ship

codeflowbench-reproduce:
	uv run --frozen --all-extras --exact python -m benchmarks.codeflowbench.reproduce

codeflowbench-scorecard-estimate:
	uv run --frozen --all-extras --exact python -m benchmarks.codeflowbench.scorecard_campaign estimate

codeflowbench-scorecard-ship:
	uv run --frozen --all-extras --exact python -m benchmarks.codeflowbench.scorecard_campaign ship \
		$(if $(CODEFLOW_SCORECARD_RUNTIME_EVIDENCE),--runtime-evidence "$(CODEFLOW_SCORECARD_RUNTIME_EVIDENCE)",)

codeflowbench-scorecard-score:
	@test -n "$$CODEFLOW_SCORECARD_FORMAL_SUMMARY" || (echo "CODEFLOW_SCORECARD_FORMAL_SUMMARY is required" >&2; exit 2)
	uv run --frozen --all-extras --exact python -m benchmarks.codeflowbench.scorecard \
		--formal-summary "$$CODEFLOW_SCORECARD_FORMAL_SUMMARY" \
		$(if $(CODEFLOW_SCORECARD_RUNTIME_EVIDENCE),--runtime-evidence "$(CODEFLOW_SCORECARD_RUNTIME_EVIDENCE)",) \
		$(if $(CODEFLOW_SCORECARD_TOKENWISE_REPORT),--tokenwise-report "$(CODEFLOW_SCORECARD_TOKENWISE_REPORT)",) \
		$(if $(CODEFLOW_SCORECARD_PREREGISTERED),--scoring-spec-preregistered,)

codeflowbench-runtime-scheduler:
	uv run --frozen --all-extras --exact python -m benchmarks.codeflowbench.packs.runtime.scheduler_experiments

codeflowbench-runtime-tools:
	uv run --frozen --all-extras --exact python -m benchmarks.codeflowbench.packs.runtime.tool_execution_experiments

codeflowbench-runtime-live-plan:
	uv run --frozen --all-extras --exact python -m benchmarks.codeflowbench.packs.runtime.live_scheduler_experiment plan

codeflowbench-runtime-live-run:
	@test -n "$$CODEFLOW_LIVE_PERF_APPROVAL_DIGEST" || (echo "CODEFLOW_LIVE_PERF_APPROVAL_DIGEST is required" >&2; exit 2)
	@test -n "$$CODEFLOW_LIVE_PERF_APPROVED_CNY" || (echo "CODEFLOW_LIVE_PERF_APPROVED_CNY is required" >&2; exit 2)
	uv run --frozen --all-extras --exact python -m benchmarks.codeflowbench.packs.runtime.live_scheduler_experiment run \
			--approval-digest "$$CODEFLOW_LIVE_PERF_APPROVAL_DIGEST" \
			--approved-cny "$$CODEFLOW_LIVE_PERF_APPROVED_CNY"

codeflowbench-runtime-live-verify:
	@test -n "$$CODEFLOW_LIVE_PERF_EVIDENCE" || (echo "CODEFLOW_LIVE_PERF_EVIDENCE is required" >&2; exit 2)
	uv run --frozen --all-extras --exact python -m benchmarks.codeflowbench.packs.runtime.live_scheduler_experiment verify \
			--evidence "$$CODEFLOW_LIVE_PERF_EVIDENCE"

codeflowbench-call-efficiency-plan codeflowbench-call-efficiency-preflight codeflowbench-call-efficiency-run codeflowbench-call-efficiency-verify:
	uv run --frozen --all-extras --exact python -m benchmarks.codeflowbench.tokenwise_cost_campaign \
		--mode $(if $(filter codeflowbench-call-efficiency-run,$@),formal,$(patsubst codeflowbench-call-efficiency-%,%,$@)) \
		--output-root "$(CODEFLOW_CALL_EFFICIENCY_OUTPUT)" \
		$(if $(filter codeflowbench-call-efficiency-preflight codeflowbench-call-efficiency-run,$@),--execute-paid-campaign,)

codeflowbench-tracing-plan codeflowbench-tracing-run codeflowbench-tracing-verify:
	uv run --frozen --all-extras --exact python -m benchmarks.codeflowbench.packs.tracing.overhead_experiment \
		$(patsubst codeflowbench-tracing-%,%,$@) \
		--output-root "$(CODEFLOW_TRACING_OUTPUT)" \
		$(if $(CODEFLOW_TRACING_COMMIT),--codeflow-commit "$(CODEFLOW_TRACING_COMMIT)",)

test-tui:
	npm test --prefix ui-tui

verify-channels:
	uv run --frozen --all-extras --exact python scripts/verify_channels.py \
		--output-root .codeflow/evidence/channels

verify-turn-evidence:
	uv run --frozen --all-extras --exact python scripts/verify_turn_evidence.py \
		--output-root .codeflow/evidence/turns

verify-evolver:
	uv run --frozen --all-extras --exact pytest \
		tests/test_cli_evolve_commands.py \
		tests/test_evolver_*.py \
		tests/test_appworld_precheck.py \
		tests/test_appworld_sandbox.py \
		tests/integration/test_evolver_lifecycle_e2e.py \
		-q --strict-markers

build: build-tui

build-tui:
	npm run build --prefix ui-tui

release-dist:
	@test -n "$$CODEFLOW_RELEASE_OUTPUT" || (echo "CODEFLOW_RELEASE_OUTPUT is required and must name an empty directory outside the checkout" >&2; exit 2)
	uv run --frozen --all-extras --exact python scripts/verify_distribution.py \
		--output-root "$$CODEFLOW_RELEASE_OUTPUT" \
		--entrypoint codeflow \
		--extras base,channel-feishu,channel-qq,channel-wecom,channels,sandbox

check-commits:
	npx commitlint --from origin/main --to HEAD --config commitlint.config.cjs
	PYTHONPATH=. uv run --extra dev python scripts/check_commit_messages.py $(COMMIT_RANGE)

check-pr-title:
	PYTHONPATH=. uv run --extra dev python scripts/check_pr_title.py

check-large-files:
	PYTHONPATH=. uv run --extra dev python scripts/check_large_files.py $(COMMIT_RANGE)

ci: lint test build

clean:
	rm -rf .pytest_cache .ruff_cache .uv-cache .mypy_cache htmlcov dist build
	rm -rf ui-tui/dist ui-tui/coverage ui-tui/.vitest-cache ui-tui/packages/hermes-ink/dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
