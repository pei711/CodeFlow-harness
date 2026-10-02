from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.codeflowbench import ExperimentSpec, PackRegistry, rebuild_report, run
from benchmarks.codeflowbench.artifacts import (
    ArtifactError,
    ArtifactStore,
)
from benchmarks.codeflowbench.protocol import RetrievalContext, RetrievalExecution, TrialContext, TrialExecution
from benchmarks.codeflowbench.records import (
    ComparisonBlockKey,
    DeliveryOutcome,
    RetrievalStatus,
    TrialStatus,
    TurnTerminalState,
    VerificationState,
    VerifierResult,
)
from benchmarks.codeflowbench.schema import (
    ExecutionPolicy,
    ExperimentRef,
    PackDefinition,
    PairSpec,
    RetrievalConfigurationSpec,
    RetrievalQuerySpec,
    RetrievalSuiteSpec,
    TaskSpec,
    VariantSpec,
)


class _PassingPack:
    def definition(self) -> PackDefinition:
        return PackDefinition(
            pack_id="dummy-v1",
            tasks=(TaskSpec(task_id="task-1"),),
            variants=(
                VariantSpec(variant_id="control", settings={"axis": "off"}),
                VariantSpec(variant_id="treatment", settings={"axis": "on"}),
            ),
            pairs=(
                PairSpec(
                    treatment_axis="axis",
                    control_variant_id="control",
                    treatment_variant_id="treatment",
                ),
            ),
        )

    async def run_trial(self, context: TrialContext) -> TrialExecution:
        return TrialExecution(
            status=TrialStatus.PASSED,
            runtime_state=TurnTerminalState.COMPLETED,
            delivery_state=DeliveryOutcome.DELIVERED,
            verification=VerifierResult(state=VerificationState.PASSED),
            observed_variant_settings=context.variant.settings,
            metrics={"dummy.score": 1},
        )


class _CountingPack(_PassingPack):
    def __init__(self) -> None:
        self.calls = 0

    async def run_trial(self, context: TrialContext) -> TrialExecution:
        self.calls += 1
        return await super().run_trial(context)


class _MeasurableFailurePack(_CountingPack):
    async def run_trial(self, context: TrialContext) -> TrialExecution:
        self.calls += 1
        return TrialExecution(
            status=TrialStatus.TASK_FAILED,
            runtime_state=TurnTerminalState.COMPLETED,
            delivery_state=DeliveryOutcome.DELIVERED,
            verification=VerifierResult(state=VerificationState.FAILED),
            observed_variant_settings=context.variant.settings,
        )


class _TwoTaskRetryPack(_CountingPack):
    def definition(self) -> PackDefinition:
        base = super().definition()
        return replace(
            base,
            tasks=(
                TaskSpec(task_id="task-1"),
                TaskSpec(task_id="task-2"),
            ),
        )

    async def run_trial(self, context: TrialContext) -> TrialExecution:
        self.calls += 1
        if context.block_attempt == 1:
            return TrialExecution(
                status=TrialStatus.PROVIDER_FAILURE,
                runtime_state=TurnTerminalState.PROVIDER_FAILED,
                delivery_state=DeliveryOutcome.NO_OUTLET,
                verification=VerifierResult(
                    state=VerificationState.NOT_RUN,
                ),
                observed_variant_settings=context.variant.settings,
            )
        return await _PassingPack.run_trial(self, context)


class _InjectedCrash(BaseException):
    pass


class _PartialRetryPack(_CountingPack):
    def __init__(self) -> None:
        super().__init__()
        self.crash = True
        self.retry_calls = 0

    async def run_trial(self, context: TrialContext) -> TrialExecution:
        self.calls += 1
        if context.block_attempt == 1:
            return TrialExecution(
                status=TrialStatus.PROVIDER_FAILURE,
                runtime_state=TurnTerminalState.PROVIDER_FAILED,
                delivery_state=DeliveryOutcome.NO_OUTLET,
                verification=VerifierResult(
                    state=VerificationState.NOT_RUN,
                ),
                observed_variant_settings=context.variant.settings,
            )
        self.retry_calls += 1
        if self.retry_calls == 2 and self.crash:
            raise _InjectedCrash
        return await _PassingPack.run_trial(self, context)


@pytest.mark.asyncio
async def test_run_resume_and_rebuild_use_the_public_interface(tmp_path: Path) -> None:
    registry = PackRegistry()
    registry.register(_PassingPack())
    spec = ExperimentSpec(
        suite="dummy-suite",
        repetitions=1,
        pack_ids=("dummy-v1",),
        output_root=tmp_path / "first",
        identity={"codeflow_commit": "a" * 40, "model": "scripted/dummy"},
    )

    ref = await run(spec, registry=registry)
    resumed = await run(spec, registry=registry)
    report = rebuild_report(ref)

    assert ref == resumed
    assert ref.root == tmp_path / "first" / ref.experiment_id
    assert report.experiment_id == ref.experiment_id
    assert report.planned_trials == 2
    assert report.terminal_trials == 2
    assert report.status_counts == {"passed": 2}

    relocated = replace(spec, output_root=tmp_path / "second")
    relocated_ref = await run(relocated, registry=registry)
    assert relocated_ref.experiment_id == ref.experiment_id


@pytest.mark.asyncio
async def test_resume_rebuilds_pair_and_trial_summaries_without_new_calls(
    tmp_path: Path,
) -> None:
    pack = _CountingPack()
    registry = PackRegistry()
    registry.register(pack)
    spec = ExperimentSpec(
        suite="summary-rebuild-suite",
        repetitions=1,
        pack_ids=("dummy-v1",),
        output_root=tmp_path,
        identity={"codeflow_commit": "a" * 40, "model": "scripted/dummy"},
    )

    ref = await run(spec, registry=registry)
    pair_path = next(ref.root.glob("pairs/**/pair-result.json"))
    trial_path = next(ref.root.glob("trials/**/trial-record.json"))
    pair_path.write_text(
        json.dumps(
            {
                "plan_digest": ref.experiment_id,
                "key": {"pack_id": "wrong-pack"},
            }
        ),
        encoding="utf-8",
    )
    trial_path.unlink()

    resumed = await run(spec, registry=registry)

    assert resumed == ref
    assert pack.calls == 2
    rebuilt_pair = json.loads(pair_path.read_text(encoding="utf-8"))
    assert rebuilt_pair["key"]["pack_id"] == "dummy-v1"
    assert rebuilt_pair["valid"] is True
    assert trial_path.is_file()


@pytest.mark.asyncio
async def test_resume_rejects_mismatched_block_attempt_refs_and_rebuilds(
    tmp_path: Path,
) -> None:
    pack = _CountingPack()
    registry = PackRegistry()
    registry.register(pack)
    spec = ExperimentSpec(
        suite="attempt-ref-rebuild-suite",
        repetitions=1,
        pack_ids=("dummy-v1",),
        output_root=tmp_path,
        identity={"codeflow_commit": "a" * 40, "model": "scripted/dummy"},
    )

    ref = await run(spec, registry=registry)
    block_path = next(ref.root.glob("blocks/**/block-result.json"))
    block = json.loads(block_path.read_text(encoding="utf-8"))
    block["variant_attempt_refs"] = ["trials/wrong/attempt-record.json"]
    block_path.write_text(json.dumps(block), encoding="utf-8")

    await run(spec, registry=registry)

    assert pack.calls == 2
    rebuilt = json.loads(block_path.read_text(encoding="utf-8"))
    assert rebuilt["variant_attempt_refs"] != ["trials/wrong/attempt-record.json"]
    assert len(rebuilt["variant_attempt_refs"]) == 2


class _RetryingPack(_PassingPack):
    def __init__(self) -> None:
        self.retrieval_calls = 0

    def definition(self) -> PackDefinition:
        base = super().definition()
        return PackDefinition(
            pack_id=base.pack_id,
            tasks=base.tasks,
            variants=base.variants,
            pairs=base.pairs,
            retrieval_suites=(
                RetrievalSuiteSpec(
                    retrieval_suite_id="retry-retrieval",
                    queries=(
                        RetrievalQuerySpec(
                            query_id="query-1",
                            label="positive",
                            expected_item_ids=("item-1",),
                        ),
                    ),
                    configurations=(
                        RetrievalConfigurationSpec(
                            configuration_id="left",
                            settings={"source": "left"},
                        ),
                        RetrievalConfigurationSpec(
                            configuration_id="right",
                            settings={"source": "right"},
                        ),
                    ),
                    corpus_digest="corpus",
                    query_labels_digest="labels",
                ),
            ),
        )

    async def run_trial(self, context: TrialContext) -> TrialExecution:
        execution = await super().run_trial(context)
        if context.block_attempt == 1 and context.variant.variant_id == "treatment":
            return TrialExecution(
                status=TrialStatus.PROVIDER_FAILURE,
                runtime_state=TurnTerminalState.PROVIDER_FAILED,
                delivery_state=DeliveryOutcome.NO_OUTLET,
                verification=VerifierResult(state=VerificationState.NOT_RUN),
                observed_variant_settings=context.variant.settings,
            )
        return execution

    async def run_retrieval_case(
        self,
        context: RetrievalContext,
    ) -> RetrievalExecution:
        self.retrieval_calls += 1
        if context.query_block_attempt == 1 and context.configuration.configuration_id == "right":
            return RetrievalExecution(
                status=RetrievalStatus.INFRASTRUCTURE_FAILURE,
                findings=("injected_failure",),
            )
        item = {"item_id": "item-1", "rank": 1, "injected": True}
        return RetrievalExecution(
            status=RetrievalStatus.MEASURABLE,
            ranked_results=(item,),
            injected_results=(item,),
        )


@pytest.mark.asyncio
async def test_retry_summaries_reference_every_immutable_attempt(
    tmp_path: Path,
) -> None:
    registry = PackRegistry()
    registry.register(_RetryingPack())
    spec = ExperimentSpec(
        suite="retry-suite",
        repetitions=1,
        pack_ids=("dummy-v1",),
        output_root=tmp_path,
        identity={"codeflow_commit": "b" * 40, "model": "scripted/retry"},
    )

    ref = await run(spec, registry=registry)
    trial_records = sorted(ref.root.glob("trials/**/trial-record.json"))
    retrieval_records = sorted(ref.root.glob("retrieval/**/retrieval-case-record.json"))

    assert len(trial_records) == 2
    assert all(len(json.loads(path.read_text())["attempt_refs"]) == 2 for path in trial_records)
    assert len(retrieval_records) == 2
    assert all(len(json.loads(path.read_text())["attempt_refs"]) == 2 for path in retrieval_records)

    report = rebuild_report(ref)
    assert report.retrieval_first_attempt_status_counts == {
        "infrastructure_failure": 1,
        "measurable": 1,
    }
    assert report.retrieval_all_attempt_status_counts == {
        "infrastructure_failure": 1,
        "measurable": 3,
    }


@pytest.mark.asyncio
async def test_measurable_task_failure_does_not_consume_block_retry(
    tmp_path: Path,
) -> None:
    pack = _MeasurableFailurePack()
    registry = PackRegistry()
    registry.register(pack)
    spec = ExperimentSpec(
        suite="measurable-failure-suite",
        repetitions=1,
        pack_ids=("dummy-v1",),
        output_root=tmp_path,
        identity={"codeflow_commit": "e" * 40, "model": "scripted/failure"},
        execution=ExecutionPolicy(
            max_comparison_block_attempts=2,
            max_comparison_block_retries_total=1,
        ),
    )

    ref = await run(spec, registry=registry)

    assert pack.calls == 2
    assert not tuple(ref.root.glob("blocks/**/retry-claims/*.json"))
    block = json.loads(
        next(ref.root.glob("blocks/**/block-result.json")).read_text(
            encoding="utf-8",
        )
    )
    assert block["resolved"] is True
    assert block["selected_block_attempt"] == 1


@pytest.mark.asyncio
async def test_experiment_retry_quota_is_shared_across_comparison_blocks(
    tmp_path: Path,
) -> None:
    pack = _TwoTaskRetryPack()
    registry = PackRegistry()
    registry.register(pack)
    spec = ExperimentSpec(
        suite="bounded-retry-suite",
        repetitions=1,
        pack_ids=("dummy-v1",),
        output_root=tmp_path,
        identity={"codeflow_commit": "f" * 40, "model": "scripted/retry"},
        execution=ExecutionPolicy(
            max_comparison_block_attempts=2,
            max_comparison_block_retries_total=1,
        ),
    )

    ref = await run(spec, registry=registry)

    assert pack.calls == 6
    assert len(tuple(ref.root.glob("blocks/**/retry-claims/*.json"))) == 1
    blocks = [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(ref.root.glob("blocks/**/block-result.json"))
    ]
    assert [(block["resolved"], block["selected_block_attempt"]) for block in blocks] == [
        (True, 2),
        (False, 1),
    ]


@pytest.mark.asyncio
async def test_partial_retry_attempt_is_not_persisted_or_spliced_on_resume(
    tmp_path: Path,
) -> None:
    pack = _PartialRetryPack()
    registry = PackRegistry()
    registry.register(pack)
    spec = ExperimentSpec(
        suite="partial-retry-suite",
        repetitions=1,
        pack_ids=("dummy-v1",),
        output_root=tmp_path,
        identity={"codeflow_commit": "1" * 40, "model": "scripted/retry"},
        execution=ExecutionPolicy(
            max_comparison_block_attempts=2,
            max_comparison_block_retries_total=1,
        ),
    )

    with pytest.raises(_InjectedCrash):
        await run(spec, registry=registry)

    experiment_root = next(path for path in tmp_path.iterdir() if path.is_dir())
    assert pack.calls == 4
    assert len(tuple(experiment_root.glob("blocks/**/retry-claims/*.json"))) == 1
    assert len(tuple(experiment_root.glob("trials/**/attempt-record.json"))) == 2

    pack.crash = False
    ref = await run(spec, registry=registry)

    assert pack.calls == 6
    assert len(tuple(ref.root.glob("blocks/**/retry-claims/*.json"))) == 1
    assert len(tuple(ref.root.glob("trials/**/attempt-record.json"))) == 4
    block = json.loads(
        next(ref.root.glob("blocks/**/block-result.json")).read_text(
            encoding="utf-8",
        )
    )
    assert block["resolved"] is True
    assert block["selected_block_attempt"] == 2


class _BlockingPack(_PassingPack):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def run_trial(self, context: TrialContext) -> TrialExecution:
        self.started.set()
        await self.release.wait()
        return await super().run_trial(context)


@pytest.mark.asyncio
async def test_run_rejects_a_concurrent_writer_for_the_same_experiment(
    tmp_path: Path,
) -> None:
    pack = _BlockingPack()
    registry = PackRegistry()
    registry.register(pack)
    spec = ExperimentSpec(
        suite="single-writer-suite",
        repetitions=1,
        pack_ids=("dummy-v1",),
        output_root=tmp_path,
        identity={"codeflow_commit": "2" * 40, "model": "scripted/lock"},
    )

    first = asyncio.create_task(run(spec, registry=registry))
    await pack.started.wait()
    with pytest.raises(ArtifactError, match="active writer"):
        await run(spec, registry=registry)
    pack.release.set()
    await first


def test_corrupt_comparison_retry_claim_fails_closed(
    tmp_path: Path,
) -> None:
    ref = ExperimentRef(experiment_id="experiment", root=tmp_path)
    store = ArtifactStore(ref)
    key = ComparisonBlockKey(
        experiment_id="experiment",
        pack_id="dummy-v1",
        task_id="task-1",
        repetition=1,
    )
    assert store.claim_comparison_block_retry(
        key=key,
        block_attempt=2,
        plan_digest="a" * 64,
        maximum_claims=1,
    )
    claim_path = store.comparison_block_retry_claim_path(key, 2)
    claim_path.write_text("{", encoding="utf-8")

    with pytest.raises(ArtifactError):
        store.claim_comparison_block_retry(
            key=key,
            block_attempt=2,
            plan_digest="a" * 64,
            maximum_claims=1,
        )


@pytest.mark.asyncio
async def test_resume_rebuilds_corrupt_retrieval_case_from_selected_attempt(
    tmp_path: Path,
) -> None:
    pack = _RetryingPack()
    registry = PackRegistry()
    registry.register(pack)
    spec = ExperimentSpec(
        suite="retrieval-case-rebuild-suite",
        repetitions=1,
        pack_ids=("dummy-v1",),
        output_root=tmp_path,
        identity={"codeflow_commit": "c" * 40, "model": "scripted/retry"},
    )

    ref = await run(spec, registry=registry)
    case_path = next(ref.root.glob("retrieval/**/retrieval-case-record.json"))
    case = json.loads(case_path.read_text(encoding="utf-8"))
    selected_attempt = json.loads((ref.root / case["attempt_refs"][-1]).read_text(encoding="utf-8"))
    case["ranked_results"] = []
    case_path.write_text(json.dumps(case), encoding="utf-8")

    await run(spec, registry=registry)

    rebuilt = json.loads(case_path.read_text(encoding="utf-8"))
    assert pack.retrieval_calls == 4
    assert rebuilt["ranked_results"] == selected_attempt["ranked_results"]
    block = json.loads(
        next(ref.root.glob("retrieval/query-blocks/**/retrieval-query-block-result.json")).read_text(encoding="utf-8")
    )
    assert block["resolved"] is True
    assert block["exhausted"] is False


@pytest.mark.asyncio
async def test_resume_rebuilds_corrupt_retrieval_block_attempt_refs(
    tmp_path: Path,
) -> None:
    pack = _RetryingPack()
    registry = PackRegistry()
    registry.register(pack)
    spec = ExperimentSpec(
        suite="retrieval-block-rebuild-suite",
        repetitions=1,
        pack_ids=("dummy-v1",),
        output_root=tmp_path,
        identity={"codeflow_commit": "d" * 40, "model": "scripted/retry"},
    )

    ref = await run(spec, registry=registry)
    block_path = next(ref.root.glob("retrieval/query-blocks/**/retrieval-query-block-result.json"))
    block = json.loads(block_path.read_text(encoding="utf-8"))
    expected_refs = block["configuration_attempt_refs"]
    block["configuration_attempt_refs"] = ["retrieval/wrong/retrieval-attempt-record.json"]
    block_path.write_text(json.dumps(block), encoding="utf-8")

    await run(spec, registry=registry)

    rebuilt = json.loads(block_path.read_text(encoding="utf-8"))
    assert pack.retrieval_calls == 4
    assert rebuilt["configuration_attempt_refs"] == expected_refs
    assert rebuilt["resolved"] is True
    assert rebuilt["exhausted"] is False
