from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from .artifacts import ArtifactStore, artifact_dict
from .plan import ExperimentPlan, compile_plan, variant_diff
from .protocol import (
    Pack,
    RetrievalContext,
    RetrievalExecution,
    TrialContext,
    TrialExecution,
)
from .records import (
    MEASURABLE_RETRIEVAL_STATUSES,
    MEASURABLE_TRIAL_STATUSES,
    AttemptKey,
    AttemptRecord,
    ComparisonBlockKey,
    ComparisonBlockResult,
    DeliveryOutcome,
    PairResult,
    RetrievalAttemptKey,
    RetrievalAttemptRecord,
    RetrievalCaseRecord,
    RetrievalQueryBlockKey,
    RetrievalQueryBlockResult,
    RetrievalStatus,
    TrialRecord,
    TrialStatus,
    TurnTerminalState,
    VerificationState,
    VerifierResult,
)
from .registry import PackRegistry, default_registry
from .schema import ExperimentRef, ExperimentSpec, PackDefinition


async def run(
    spec: ExperimentSpec,
    *,
    registry: PackRegistry | None = None,
) -> ExperimentRef:
    selected_registry = registry or default_registry()
    packs = selected_registry.resolve(spec.pack_ids)
    plan = compile_plan(spec, packs)
    ref = ExperimentRef(
        experiment_id=plan.experiment_id,
        root=spec.output_root / plan.experiment_id,
    )
    store = ArtifactStore(ref)
    with store.exclusive_run_lock():
        store.freeze_manifest(plan.manifest())
        pack_map = {pack.definition().pack_id: pack for pack in packs}
        definition_map = {definition.pack_id: definition for definition in plan.pack_definitions}

        for block in plan.comparison_blocks:
            if _comparison_block_is_resumable(store, plan, block.key):
                continue
            if _rebuild_comparison_block_from_attempts(
                store=store,
                plan=plan,
                definition=definition_map[block.key.pack_id],
                block=block,
            ):
                continue
            await _run_comparison_block(
                spec=spec,
                plan=plan,
                store=store,
                pack=pack_map[block.key.pack_id],
                definition=definition_map[block.key.pack_id],
                block=block,
            )

        for block in plan.retrieval_query_blocks:
            if _retrieval_block_is_resumable(store, plan, block.key):
                continue
            pack, definition = _retrieval_owner(
                block.key.retrieval_suite_id,
                packs=packs,
                definitions=plan.pack_definitions,
            )
            await _run_retrieval_block(
                spec=spec,
                plan=plan,
                store=store,
                pack=pack,
                definition=definition,
                block=block,
            )

        rebuild_report(ref)
    return ref


def rebuild_report(ref: ExperimentRef):
    from .report import rebuild_full_report

    return rebuild_full_report(ref)


async def _run_comparison_block(
    *,
    spec: ExperimentSpec,
    plan: ExperimentPlan,
    store: ArtifactStore,
    pack: Pack,
    definition: PackDefinition,
    block: Any,
) -> None:
    variants = {variant.variant_id: variant for variant in definition.variants}
    tasks = {task.task_id: task for task in definition.tasks}
    maximum = spec.execution.max_comparison_block_attempts
    selected_records: dict[str, AttemptRecord] = {}
    selected_attempt = 1
    resolved = False

    for attempt_number in range(1, maximum + 1):
        if attempt_number > 1 and not store.claim_comparison_block_retry(
            key=block.key,
            block_attempt=attempt_number,
            plan_digest=plan.plan_digest,
            maximum_claims=(spec.execution.max_comparison_block_retries_total),
        ):
            store.append_journal(
                {
                    "kind": "comparison_block_retry_quota_exhausted",
                    "plan_digest": plan.plan_digest,
                    "block_key": artifact_dict(block.key),
                    "requested_block_attempt": attempt_number,
                }
            )
            break
        selected_attempt = attempt_number
        existing_records: dict[str, AttemptRecord] = {}
        for trial_key in block.trial_keys:
            variant = variants[trial_key.variant_id]
            existing = _read_exact_attempt(
                store=store,
                plan=plan,
                key=block.key,
                variant_id=variant.variant_id,
                selected_attempt=attempt_number,
            )
            if existing is not None:
                existing_records[variant.variant_id] = _attempt_record_from_dict(existing)
        if existing_records:
            selected_records = existing_records
            if len(existing_records) != len(block.trial_keys):
                store.append_journal(
                    {
                        "kind": "comparison_block_partial_attempt_rejected",
                        "plan_digest": plan.plan_digest,
                        "block_key": artifact_dict(block.key),
                        "block_attempt": attempt_number,
                        "present_variants": sorted(existing_records),
                    }
                )
                continue
            if all(record.status in MEASURABLE_TRIAL_STATUSES for record in selected_records.values()):
                resolved = True
                break
            continue

        pending_records: dict[str, AttemptRecord] = {}
        for trial_key in block.trial_keys:
            variant = variants[trial_key.variant_id]
            attempt_key = AttemptKey(
                block=block.key,
                variant_id=variant.variant_id,
                block_attempt=attempt_number,
            )
            execution = await _execute_trial(
                pack,
                TrialContext(
                    experiment_id=plan.experiment_id,
                    plan_digest=plan.plan_digest,
                    key=trial_key,
                    block_attempt=attempt_number,
                    experiment=spec,
                    task=tasks[trial_key.task_id],
                    variant=variant,
                ),
            )
            if dict(execution.observed_variant_settings) != dict(variant.settings):
                execution = TrialExecution(
                    status=TrialStatus.INFRASTRUCTURE_FAILURE,
                    runtime_state=execution.runtime_state,
                    delivery_state=execution.delivery_state,
                    verification=VerifierResult(
                        state=VerificationState.NOT_RUN,
                        findings=("variant_drift",),
                    ),
                    observed_variant_settings=dict(execution.observed_variant_settings),
                    metrics=dict(execution.metrics),
                    findings=(*execution.findings, "variant_drift"),
                    artifact_refs=execution.artifact_refs,
                )
            record = AttemptRecord(
                key=attempt_key,
                plan_digest=plan.plan_digest,
                status=execution.status,
                runtime_state=execution.runtime_state,
                delivery_state=execution.delivery_state,
                verification=execution.verification,
                declared_variant_settings=dict(variant.settings),
                observed_variant_settings=dict(execution.observed_variant_settings),
                metrics=dict(execution.metrics),
                findings=execution.findings,
                artifact_refs=execution.artifact_refs,
            )
            pending_records[variant.variant_id] = record

        for variant_id, record in pending_records.items():
            attempt_key = AttemptKey(
                block=block.key,
                variant_id=variant_id,
                block_attempt=attempt_number,
            )
            path = store.attempt_path(attempt_key)
            store.append_immutable(path, artifact_dict(record))
            store.append_journal(
                {
                    "kind": "trial_attempt_terminal",
                    "plan_digest": plan.plan_digest,
                    "attempt_ref": store.relative(path),
                    "status": record.status,
                }
            )
        selected_records = pending_records

        if all(record.status in MEASURABLE_TRIAL_STATUSES for record in selected_records.values()):
            resolved = True
            break

    if not selected_records:
        selected_records = _load_latest_trial_attempts(
            store=store,
            plan=plan,
            block=block,
            selected_attempt=selected_attempt,
            variants=variants,
        )

    attempt_refs = _trial_attempt_refs(
        store=store,
        plan=plan,
        block=block,
        variant_ids=tuple(variants),
        selected_attempt=selected_attempt,
    )
    block_result = ComparisonBlockResult(
        key=block.key,
        plan_digest=plan.plan_digest,
        resolved=resolved,
        exhausted=not resolved,
        selected_block_attempt=selected_attempt,
        variant_attempt_refs=attempt_refs,
        findings=(() if resolved else ("comparison_block_attempts_exhausted",)),
    )
    store.write_summary(store.block_path(block.key), artifact_dict(block_result))
    _write_trial_and_pair_summaries(
        plan=plan,
        store=store,
        definition=definition,
        block=block,
        block_result=block_result,
        records=selected_records,
    )


async def _execute_trial(pack: Pack, context: TrialContext) -> TrialExecution:
    from .budget import provider_call_budget_scope

    pack_budget = context.experiment.execution.provider_trial_budget_for(
        context.key.pack_id,
    )
    maximum_calls = (
        pack_budget.max_provider_calls_per_trial
        if pack_budget is not None
        else context.experiment.execution.max_provider_calls_per_trial
    )
    maximum_input_tokens = pack_budget.max_input_tokens_per_call if pack_budget is not None else None
    maximum_output_tokens = pack_budget.max_output_tokens_per_call if pack_budget is not None else None
    try:
        scope = (
            provider_call_budget_scope(
                trial_id=(
                    f"{context.experiment_id}/"
                    f"{context.key.pack_id}/{context.key.task_id}/"
                    f"{context.key.variant_id}/"
                    f"{context.key.repetition}/"
                    f"{context.block_attempt}"
                ),
                max_logical_calls=maximum_calls,
                max_attempts_per_call=(context.experiment.execution.provider_call_max_attempts),
                max_input_tokens_per_call=maximum_input_tokens,
                max_output_tokens_per_call=maximum_output_tokens,
            )
            if maximum_calls is not None
            else contextlib.nullcontext()
        )
        with scope:
            return await asyncio.wait_for(
                pack.run_trial(context),
                timeout=context.experiment.execution.timeout_seconds,
            )
    except TimeoutError:
        return TrialExecution(
            status=TrialStatus.TASK_TIMEOUT,
            runtime_state=None,
            delivery_state=None,
            verification=VerifierResult(state=VerificationState.NOT_RUN),
            observed_variant_settings=dict(context.variant.settings),
            findings=("trial_timeout",),
        )
    except Exception as exc:
        return TrialExecution(
            status=TrialStatus.INFRASTRUCTURE_FAILURE,
            runtime_state=None,
            delivery_state=None,
            verification=VerifierResult(
                state=VerificationState.NOT_RUN,
                findings=(f"pack_exception:{type(exc).__name__}",),
            ),
            observed_variant_settings=dict(context.variant.settings),
            findings=(f"pack_exception:{type(exc).__name__}",),
        )


def _write_trial_and_pair_summaries(
    *,
    plan: ExperimentPlan,
    store: ArtifactStore,
    definition: PackDefinition,
    block: Any,
    block_result: ComparisonBlockResult,
    records: dict[str, AttemptRecord],
) -> None:
    planned_trials = {
        planned.key.variant_id: planned
        for planned in plan.trials
        if planned.key.pack_id == block.key.pack_id
        and planned.key.task_id == block.key.task_id
        and planned.key.repetition == block.key.repetition
    }
    for variant_id, planned in planned_trials.items():
        record = records.get(variant_id)
        if record is None:
            record = _missing_attempt_record(
                plan=plan,
                block=block,
                variant_id=variant_id,
                selected_attempt=block_result.selected_block_attempt,
                declared_settings=dict(
                    next(variant.settings for variant in definition.variants if variant.variant_id == variant_id)
                ),
            )
        summary = TrialRecord(
            key=planned.key,
            plan_digest=plan.plan_digest,
            pair_memberships=planned.pair_memberships,
            status=record.status,
            runtime_state=record.runtime_state,
            delivery_state=record.delivery_state,
            verification=record.verification,
            selected_block_attempt=block_result.selected_block_attempt,
            attempt_refs=_trial_attempt_refs(
                store=store,
                plan=plan,
                block=block,
                variant_ids=(variant_id,),
                selected_attempt=block_result.selected_block_attempt,
            ),
            declared_variant_settings=record.declared_variant_settings,
            observed_variant_settings=record.observed_variant_settings,
            metrics=record.metrics,
            findings=record.findings,
            artifact_refs=record.artifact_refs,
        )
        store.write_summary(store.trial_path(planned.key), artifact_dict(summary))

    for pair_key in (
        pair
        for pair in plan.pairs
        if pair.pack_id == block.key.pack_id
        and pair.task_id == block.key.task_id
        and pair.repetition == block.key.repetition
    ):
        control = records.get(pair_key.control_variant_id)
        treatment = records.get(pair_key.treatment_variant_id)
        actual_diff = (
            variant_diff(
                dict(control.observed_variant_settings),
                dict(treatment.observed_variant_settings),
            )
            if control is not None and treatment is not None
            else {}
        )
        valid = (
            block_result.resolved
            and control is not None
            and treatment is not None
            and set(actual_diff) == {pair_key.treatment_axis}
            and control.status in MEASURABLE_TRIAL_STATUSES
            and treatment.status in MEASURABLE_TRIAL_STATUSES
        )
        findings = () if valid else ("pair_invalid_or_variant_drift",)
        result = PairResult(
            key=pair_key,
            plan_digest=plan.plan_digest,
            selected_block_attempt=(block_result.selected_block_attempt if block_result.resolved else None),
            valid=valid,
            actual_variant_diff=actual_diff,
            findings=findings,
        )
        store.write_summary(store.pair_path(pair_key), artifact_dict(result))


async def _run_retrieval_block(
    *,
    spec: ExperimentSpec,
    plan: ExperimentPlan,
    store: ArtifactStore,
    pack: Pack,
    definition: PackDefinition,
    block: Any,
) -> None:
    suite = next(
        suite for suite in definition.retrieval_suites if suite.retrieval_suite_id == block.key.retrieval_suite_id
    )
    query = next(query for query in suite.queries if query.query_id == block.key.query_id)
    configurations = {configuration.configuration_id: configuration for configuration in suite.configurations}
    attempt_number = store.next_retrieval_attempt(block.key)
    maximum = spec.execution.max_retrieval_query_block_attempts
    selected_attempt = min(max(attempt_number, 1), maximum)
    selected_records: dict[str, RetrievalAttemptRecord] = {}
    resolved = False

    while attempt_number <= maximum:
        selected_attempt = attempt_number
        selected_records = {}
        for case_key in block.case_keys:
            configuration = configurations[case_key.configuration_id]
            attempt_key = RetrievalAttemptKey(
                block=block.key,
                configuration_id=configuration.configuration_id,
                query_block_attempt=attempt_number,
            )
            execution = await _execute_retrieval(
                pack,
                RetrievalContext(
                    experiment_id=plan.experiment_id,
                    plan_digest=plan.plan_digest,
                    key=case_key,
                    query_block_attempt=attempt_number,
                    experiment=spec,
                    query=query,
                    configuration=configuration,
                ),
            )
            record = RetrievalAttemptRecord(
                key=attempt_key,
                plan_digest=plan.plan_digest,
                status=execution.status,
                label=query.label,
                expected_item_ids=query.expected_item_ids,
                ranked_results=execution.ranked_results,
                injected_results=execution.injected_results,
                usage=dict(execution.usage),
                metadata=dict(execution.metadata),
                findings=execution.findings,
            )
            path = store.retrieval_attempt_path(attempt_key)
            store.append_immutable(
                path,
                _retrieval_artifact_dict(record),
            )
            store.append_journal(
                {
                    "kind": "retrieval_attempt_terminal",
                    "plan_digest": plan.plan_digest,
                    "attempt_ref": store.relative(path),
                    "status": record.status,
                }
            )
            selected_records[configuration.configuration_id] = record
        if all(record.status in MEASURABLE_RETRIEVAL_STATUSES for record in selected_records.values()):
            resolved = True
            break
        attempt_number += 1

    if not selected_records:
        selected_records = _load_latest_retrieval_attempts(
            store=store,
            plan=plan,
            block=block,
            selected_attempt=selected_attempt,
            configurations=configurations,
        )
        resolved = len(selected_records) == len(configurations) and all(
            record.status in MEASURABLE_RETRIEVAL_STATUSES for record in selected_records.values()
        )

    refs = _retrieval_attempt_refs(
        store=store,
        plan=plan,
        block=block,
        configuration_ids=tuple(configurations),
        selected_attempt=selected_attempt,
    )
    block_result = RetrievalQueryBlockResult(
        key=block.key,
        plan_digest=plan.plan_digest,
        resolved=resolved,
        exhausted=not resolved,
        selected_query_block_attempt=selected_attempt,
        configuration_attempt_refs=refs,
        findings=() if resolved else ("retrieval_query_block_attempts_exhausted",),
    )
    store.write_summary(store.retrieval_block_path(block.key), artifact_dict(block_result))
    for case_key in block.case_keys:
        record = selected_records.get(case_key.configuration_id)
        if record is None:
            status = RetrievalStatus.INFRASTRUCTURE_FAILURE
            ranked_results: tuple[dict[str, Any], ...] = ()
            injected_results: tuple[dict[str, Any], ...] = ()
            usage: dict[str, Any] = {}
            metadata: dict[str, Any] = {}
            findings = ("missing_retrieval_attempt",)
            attempt_refs: tuple[str, ...] = ()
        else:
            status = record.status
            ranked_results = record.ranked_results
            injected_results = record.injected_results
            usage = record.usage
            metadata = record.metadata
            findings = record.findings
            attempt_refs = _retrieval_attempt_refs(
                store=store,
                plan=plan,
                block=block,
                configuration_ids=(case_key.configuration_id,),
                selected_attempt=selected_attempt,
            )
        case_record = RetrievalCaseRecord(
            key=case_key,
            plan_digest=plan.plan_digest,
            query_block=block.key,
            status=status,
            label=query.label,
            expected_item_ids=query.expected_item_ids,
            selected_query_block_attempt=selected_attempt,
            attempt_refs=attempt_refs,
            ranked_results=ranked_results,
            injected_results=injected_results,
            usage=usage,
            metadata=metadata,
            findings=findings,
        )
        store.write_summary(
            store.retrieval_case_path(case_key),
            _retrieval_artifact_dict(case_record),
        )


async def _execute_retrieval(
    pack: Pack,
    context: RetrievalContext,
) -> RetrievalExecution:
    from .budget import provider_call_budget_scope

    runner = getattr(pack, "run_retrieval_case", None)
    if runner is None:
        return RetrievalExecution(
            status=RetrievalStatus.INFRASTRUCTURE_FAILURE,
            findings=("pack_missing_retrieval_runner",),
        )
    try:
        with provider_call_budget_scope(
            trial_id=(
                f"{context.experiment_id}/retrieval/"
                f"{context.key.retrieval_suite_id}/"
                f"{context.key.query_id}/"
                f"{context.key.configuration_id}/"
                f"{context.query_block_attempt}"
            ),
            max_logical_calls=0,
            max_attempts_per_call=(context.experiment.execution.provider_call_max_attempts),
        ):
            return await asyncio.wait_for(
                runner(context),
                timeout=context.experiment.execution.timeout_seconds,
            )
    except TimeoutError:
        return RetrievalExecution(
            status=RetrievalStatus.INFRASTRUCTURE_FAILURE,
            findings=("retrieval_timeout",),
        )
    except Exception as exc:
        return RetrievalExecution(
            status=RetrievalStatus.INFRASTRUCTURE_FAILURE,
            findings=(f"pack_exception:{type(exc).__name__}",),
        )


def _comparison_block_is_resumable(
    store: ArtifactStore,
    plan: ExperimentPlan,
    key: Any,
) -> bool:
    block = store.read_if_valid(store.block_path(key), plan_digest=plan.plan_digest)
    if block is None or not (block.get("resolved") or block.get("exhausted")):
        return False
    if block.get("key") != artifact_dict(key):
        return False
    selected = block.get("selected_block_attempt")
    if not isinstance(selected, int) or isinstance(selected, bool) or selected < 1:
        return False
    for block_attempt in range(2, selected + 1):
        claim = store.read_if_valid(
            store.comparison_block_retry_claim_path(
                key,
                block_attempt,
            ),
            plan_digest=plan.plan_digest,
        )
        if (
            claim is None
            or claim.get("kind") != "comparison_block_retry_claim"
            or claim.get("key") != artifact_dict(key)
            or claim.get("block_attempt") != block_attempt
        ):
            return False
    resolved = block.get("resolved") is True
    exhausted = block.get("exhausted") is True
    if resolved == exhausted:
        return False
    planned_trials = [
        trial
        for trial in plan.trials
        if trial.key.pack_id == key.pack_id
        and trial.key.task_id == key.task_id
        and trial.key.repetition == key.repetition
    ]
    if not planned_trials:
        return False
    records: dict[str, dict[str, Any]] = {}
    for planned in planned_trials:
        trial = store.read_if_valid(
            store.trial_path(planned.key),
            plan_digest=plan.plan_digest,
        )
        if (
            trial is None
            or trial.get("key") != artifact_dict(planned.key)
            or trial.get("pair_memberships")
            != artifact_dict({"pair_memberships": planned.pair_memberships})["pair_memberships"]
            or trial.get("selected_block_attempt") != selected
        ):
            return False
        expected_refs = _trial_attempt_refs(
            store=store,
            plan=plan,
            block=_BlockRef(key),
            variant_ids=(planned.key.variant_id,),
            selected_attempt=selected,
        )
        if len(expected_refs) != selected or tuple(trial.get("attempt_refs", ())) != expected_refs:
            return False
        selected_record = _read_exact_attempt(
            store=store,
            plan=plan,
            key=key,
            variant_id=planned.key.variant_id,
            selected_attempt=selected,
        )
        if selected_record is None:
            return False
        if not _trial_matches_selected_attempt(trial, selected_record):
            return False
        records[planned.key.variant_id] = selected_record

    variant_ids = tuple(planned.key.variant_id for planned in planned_trials)
    expected_block_refs = _trial_attempt_refs(
        store=store,
        plan=plan,
        block=_BlockRef(key),
        variant_ids=variant_ids,
        selected_attempt=selected,
    )
    if (
        len(expected_block_refs) != selected * len(variant_ids)
        or tuple(block.get("variant_attempt_refs", ())) != expected_block_refs
    ):
        return False
    measurable = all(_is_measurable_trial_status(record.get("status")) for record in records.values())
    if resolved != measurable:
        return False

    for pair_key in _planned_pairs_for_block(plan, key):
        pair = store.read_if_valid(
            store.pair_path(pair_key),
            plan_digest=plan.plan_digest,
        )
        if pair is None or pair.get("key") != artifact_dict(pair_key):
            return False
        expected_pair_attempt = selected if resolved else None
        if pair.get("selected_block_attempt") != expected_pair_attempt:
            return False
        control = records.get(pair_key.control_variant_id)
        treatment = records.get(pair_key.treatment_variant_id)
        if control is None or treatment is None:
            return False
        actual_diff = variant_diff(
            dict(control.get("observed_variant_settings", {})),
            dict(treatment.get("observed_variant_settings", {})),
        )
        expected_valid = (
            resolved
            and set(actual_diff) == {pair_key.treatment_axis}
            and _is_measurable_trial_status(control.get("status"))
            and _is_measurable_trial_status(treatment.get("status"))
        )
        if pair.get("actual_variant_diff") != actual_diff or pair.get("valid") is not expected_valid:
            return False
    return True


class _BlockRef:
    def __init__(self, key: Any) -> None:
        self.key = key


def _planned_pairs_for_block(
    plan: ExperimentPlan,
    key: Any,
) -> tuple[Any, ...]:
    return tuple(
        pair
        for pair in plan.pairs
        if pair.pack_id == key.pack_id and pair.task_id == key.task_id and pair.repetition == key.repetition
    )


def _read_exact_attempt(
    *,
    store: ArtifactStore,
    plan: ExperimentPlan,
    key: Any,
    variant_id: str,
    selected_attempt: int,
) -> dict[str, Any] | None:
    attempt_key = AttemptKey(
        block=key,
        variant_id=variant_id,
        block_attempt=selected_attempt,
    )
    record = store.read_if_valid(
        store.attempt_path(attempt_key),
        plan_digest=plan.plan_digest,
    )
    if record is None or record.get("key") != artifact_dict(attempt_key):
        return None
    return record


def _read_exact_retrieval_attempt(
    *,
    store: ArtifactStore,
    plan: ExperimentPlan,
    key: Any,
    configuration_id: str,
    selected_attempt: int,
) -> dict[str, Any] | None:
    attempt_key = RetrievalAttemptKey(
        block=key,
        configuration_id=configuration_id,
        query_block_attempt=selected_attempt,
    )
    record = store.read_if_valid(
        store.retrieval_attempt_path(attempt_key),
        plan_digest=plan.plan_digest,
    )
    if record is None or record.get("key") != artifact_dict(attempt_key):
        return None
    return record


def _trial_matches_selected_attempt(
    trial: dict[str, Any],
    attempt: dict[str, Any],
) -> bool:
    return all(
        trial.get(field) == attempt.get(field)
        for field in (
            "status",
            "runtime_state",
            "delivery_state",
            "verification",
            "declared_variant_settings",
            "observed_variant_settings",
            "metrics",
            "findings",
            "artifact_refs",
        )
    )


def _retrieval_case_matches_selected_attempt(
    case: dict[str, Any],
    attempt: dict[str, Any],
) -> bool:
    return all(
        case.get(field) == attempt.get(field)
        for field in (
            "status",
            "label",
            "expected_item_ids",
            "ranked_results",
            "injected_results",
            "usage",
            "findings",
        )
    ) and _optional_metadata_matches(case, attempt)


def _is_measurable_trial_status(value: Any) -> bool:
    try:
        status = TrialStatus(str(value))
    except ValueError:
        return False
    return status in MEASURABLE_TRIAL_STATUSES


def _is_measurable_retrieval_status(value: Any) -> bool:
    try:
        status = RetrievalStatus(str(value))
    except ValueError:
        return False
    return status in MEASURABLE_RETRIEVAL_STATUSES


def _rebuild_comparison_block_from_attempts(
    *,
    store: ArtifactStore,
    plan: ExperimentPlan,
    definition: PackDefinition,
    block: Any,
) -> bool:
    block_payload = store.read_if_valid(
        store.block_path(block.key),
        plan_digest=plan.plan_digest,
    )
    selected_candidates: set[int] = set()
    if (
        block_payload is not None
        and block_payload.get("key") == artifact_dict(block.key)
        and isinstance(block_payload.get("selected_block_attempt"), int)
        and not isinstance(block_payload.get("selected_block_attempt"), bool)
    ):
        selected_candidates.add(int(block_payload["selected_block_attempt"]))
    for planned in (
        planned
        for planned in plan.trials
        if planned.key.pack_id == block.key.pack_id
        and planned.key.task_id == block.key.task_id
        and planned.key.repetition == block.key.repetition
    ):
        trial = store.read_if_valid(
            store.trial_path(planned.key),
            plan_digest=plan.plan_digest,
        )
        if (
            trial is not None
            and trial.get("key") == artifact_dict(planned.key)
            and isinstance(trial.get("selected_block_attempt"), int)
            and not isinstance(trial.get("selected_block_attempt"), bool)
        ):
            selected_candidates.add(int(trial["selected_block_attempt"]))
    if len(selected_candidates) != 1:
        return False
    selected = next(iter(selected_candidates))
    maximum = int(plan.spec_payload["execution"]["max_comparison_block_attempts"])
    if selected < 1 or selected > maximum:
        return False

    records: dict[str, AttemptRecord] = {}
    for trial_key in block.trial_keys:
        payload = _read_exact_attempt(
            store=store,
            plan=plan,
            key=block.key,
            variant_id=trial_key.variant_id,
            selected_attempt=selected,
        )
        if payload is None:
            return False
        records[trial_key.variant_id] = _attempt_record_from_dict(payload)
    resolved = all(record.status in MEASURABLE_TRIAL_STATUSES for record in records.values())
    if not resolved and selected != maximum:
        return False
    if block_payload is not None:
        block_resolved = block_payload.get("resolved") is True
        block_exhausted = block_payload.get("exhausted") is True
        if block_resolved == block_exhausted:
            return False
        if block_resolved != resolved:
            return False

    block_result = ComparisonBlockResult(
        key=block.key,
        plan_digest=plan.plan_digest,
        resolved=resolved,
        exhausted=not resolved,
        selected_block_attempt=selected,
        variant_attempt_refs=_trial_attempt_refs(
            store=store,
            plan=plan,
            block=block,
            variant_ids=tuple(variant.variant_id for variant in definition.variants),
            selected_attempt=selected,
        ),
        findings=() if resolved else ("comparison_block_attempts_exhausted",),
    )
    store.write_summary(store.block_path(block.key), artifact_dict(block_result))
    _write_trial_and_pair_summaries(
        plan=plan,
        store=store,
        definition=definition,
        block=block,
        block_result=block_result,
        records=records,
    )
    return _comparison_block_is_resumable(store, plan, block.key)


def _retrieval_block_is_resumable(
    store: ArtifactStore,
    plan: ExperimentPlan,
    key: Any,
) -> bool:
    block = store.read_if_valid(
        store.retrieval_block_path(key),
        plan_digest=plan.plan_digest,
    )
    if block is None or block.get("key") != artifact_dict(key):
        return False
    selected = block.get("selected_query_block_attempt")
    if (
        not isinstance(selected, int)
        or isinstance(selected, bool)
        or selected < 1
        or selected > int(plan.spec_payload["execution"]["max_retrieval_query_block_attempts"])
    ):
        return False
    resolved = block.get("resolved") is True
    exhausted = block.get("exhausted") is True
    if resolved == exhausted:
        return False
    cases = [case for case in plan.retrieval_cases if case.query_block == key]
    if not cases:
        return False
    records: dict[str, dict[str, Any]] = {}
    for case in cases:
        record = store.read_if_valid(
            store.retrieval_case_path(case.key),
            plan_digest=plan.plan_digest,
        )
        if (
            record is None
            or record.get("key") != artifact_dict(case.key)
            or record.get("query_block") != artifact_dict(key)
            or record.get("selected_query_block_attempt") != selected
        ):
            return False
        expected_refs = _retrieval_attempt_refs(
            store=store,
            plan=plan,
            block=_BlockRef(key),
            configuration_ids=(case.key.configuration_id,),
            selected_attempt=selected,
        )
        if len(expected_refs) != selected or tuple(record.get("attempt_refs", ())) != expected_refs:
            return False
        selected_record = _read_exact_retrieval_attempt(
            store=store,
            plan=plan,
            key=key,
            configuration_id=case.key.configuration_id,
            selected_attempt=selected,
        )
        if selected_record is None or not _retrieval_case_matches_selected_attempt(
            record,
            selected_record,
        ):
            return False
        records[case.key.configuration_id] = selected_record

    configuration_ids = tuple(case.key.configuration_id for case in cases)
    expected_block_refs = _retrieval_attempt_refs(
        store=store,
        plan=plan,
        block=_BlockRef(key),
        configuration_ids=configuration_ids,
        selected_attempt=selected,
    )
    if (
        len(expected_block_refs) != selected * len(configuration_ids)
        or tuple(block.get("configuration_attempt_refs", ())) != expected_block_refs
    ):
        return False
    measurable = all(_is_measurable_retrieval_status(record.get("status")) for record in records.values())
    if resolved != measurable:
        return False
    return True


def _trial_attempt_refs(
    *,
    store: ArtifactStore,
    plan: ExperimentPlan,
    block: Any,
    variant_ids: tuple[str, ...],
    selected_attempt: int,
) -> tuple[str, ...]:
    refs: list[str] = []
    for attempt_number in range(1, selected_attempt + 1):
        for variant_id in variant_ids:
            key = AttemptKey(
                block=block.key,
                variant_id=variant_id,
                block_attempt=attempt_number,
            )
            path = store.attempt_path(key)
            if store.read_if_valid(path, plan_digest=plan.plan_digest) is not None:
                refs.append(store.relative(path))
    return tuple(refs)


def _retrieval_attempt_refs(
    *,
    store: ArtifactStore,
    plan: ExperimentPlan,
    block: Any,
    configuration_ids: tuple[str, ...],
    selected_attempt: int,
) -> tuple[str, ...]:
    refs: list[str] = []
    for attempt_number in range(1, selected_attempt + 1):
        for configuration_id in configuration_ids:
            key = RetrievalAttemptKey(
                block=block.key,
                configuration_id=configuration_id,
                query_block_attempt=attempt_number,
            )
            path = store.retrieval_attempt_path(key)
            if store.read_if_valid(path, plan_digest=plan.plan_digest) is not None:
                refs.append(store.relative(path))
    return tuple(refs)


def _load_latest_trial_attempts(
    *,
    store: ArtifactStore,
    plan: ExperimentPlan,
    block: Any,
    selected_attempt: int,
    variants: dict[str, Any],
) -> dict[str, AttemptRecord]:
    records: dict[str, AttemptRecord] = {}
    for variant_id, variant in variants.items():
        key = AttemptKey(
            block=block.key,
            variant_id=variant_id,
            block_attempt=selected_attempt,
        )
        payload = store.read_if_valid(store.attempt_path(key), plan_digest=plan.plan_digest)
        if payload is None:
            continue
        records[variant_id] = _attempt_record_from_dict(payload)
    return records


def _load_latest_retrieval_attempts(
    *,
    store: ArtifactStore,
    plan: ExperimentPlan,
    block: Any,
    selected_attempt: int,
    configurations: dict[str, Any],
) -> dict[str, RetrievalAttemptRecord]:
    records: dict[str, RetrievalAttemptRecord] = {}
    for configuration_id in configurations:
        key = RetrievalAttemptKey(
            block=block.key,
            configuration_id=configuration_id,
            query_block_attempt=selected_attempt,
        )
        payload = store.read_if_valid(
            store.retrieval_attempt_path(key),
            plan_digest=plan.plan_digest,
        )
        if payload is None:
            continue
        records[configuration_id] = _retrieval_attempt_record_from_dict(payload)
    return records


def _attempt_record_from_dict(value: dict[str, Any]) -> AttemptRecord:
    key = value["key"]
    block = key["block"]
    return AttemptRecord(
        key=AttemptKey(
            block=ComparisonBlockKey(
                experiment_id=block["experiment_id"],
                pack_id=block["pack_id"],
                task_id=block["task_id"],
                repetition=block["repetition"],
            ),
            variant_id=key["variant_id"],
            block_attempt=key["block_attempt"],
        ),
        plan_digest=value["plan_digest"],
        status=TrialStatus(value["status"]),
        runtime_state=(None if value["runtime_state"] is None else TurnTerminalState(value["runtime_state"])),
        delivery_state=(None if value["delivery_state"] is None else DeliveryOutcome(value["delivery_state"])),
        verification=VerifierResult(
            state=VerificationState(value["verification"]["state"]),
            findings=tuple(value["verification"].get("findings", [])),
            metrics=value["verification"].get("metrics", {}),
        ),
        declared_variant_settings=value["declared_variant_settings"],
        observed_variant_settings=value["observed_variant_settings"],
        metrics=value.get("metrics", {}),
        findings=tuple(value.get("findings", [])),
        artifact_refs=tuple(value.get("artifact_refs", [])),
    )


def _retrieval_attempt_record_from_dict(
    value: dict[str, Any],
) -> RetrievalAttemptRecord:
    key = value["key"]
    block = key["block"]
    return RetrievalAttemptRecord(
        key=RetrievalAttemptKey(
            block=RetrievalQueryBlockKey(
                experiment_id=block["experiment_id"],
                retrieval_suite_id=block["retrieval_suite_id"],
                query_id=block["query_id"],
            ),
            configuration_id=key["configuration_id"],
            query_block_attempt=key["query_block_attempt"],
        ),
        plan_digest=value["plan_digest"],
        status=RetrievalStatus(value["status"]),
        label=value["label"],
        expected_item_ids=tuple(value.get("expected_item_ids", ())),
        ranked_results=tuple(value.get("ranked_results", ())),
        injected_results=tuple(value.get("injected_results", ())),
        usage=value.get("usage", {}),
        metadata=value.get("metadata", {}),
        findings=tuple(value.get("findings", ())),
    )


def _retrieval_artifact_dict(
    value: RetrievalAttemptRecord | RetrievalCaseRecord,
) -> dict[str, Any]:
    payload = artifact_dict(value)
    if not payload["metadata"]:
        payload.pop("metadata")
    return payload


def _optional_metadata_matches(
    left: dict[str, Any],
    right: dict[str, Any],
) -> bool:
    left_metadata = left.get("metadata", {})
    right_metadata = right.get("metadata", {})
    return isinstance(left_metadata, dict) and isinstance(right_metadata, dict) and left_metadata == right_metadata


def _missing_attempt_record(
    *,
    plan: ExperimentPlan,
    block: Any,
    variant_id: str,
    selected_attempt: int,
    declared_settings: dict[str, Any],
) -> AttemptRecord:
    return AttemptRecord(
        key=AttemptKey(
            block=block.key,
            variant_id=variant_id,
            block_attempt=selected_attempt,
        ),
        plan_digest=plan.plan_digest,
        status=TrialStatus.INFRASTRUCTURE_FAILURE,
        runtime_state=None,
        delivery_state=None,
        verification=VerifierResult(state=VerificationState.NOT_RUN),
        declared_variant_settings=declared_settings,
        observed_variant_settings=declared_settings,
        findings=("missing_attempt_record",),
    )


def _retrieval_owner(
    retrieval_suite_id: str,
    *,
    packs: tuple[Pack, ...],
    definitions: tuple[PackDefinition, ...],
) -> tuple[Pack, PackDefinition]:
    for pack, definition in zip(packs, definitions, strict=True):
        if any(suite.retrieval_suite_id == retrieval_suite_id for suite in definition.retrieval_suites):
            return pack, definition
    raise KeyError(f"retrieval suite has no owning Pack: {retrieval_suite_id}")
