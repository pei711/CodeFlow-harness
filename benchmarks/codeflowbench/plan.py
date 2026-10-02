from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .canonical import canonical_digest, to_primitive, validate_plan_value
from .protocol import Pack
from .records import (
    ComparisonBlockKey,
    PairKey,
    RetrievalCaseKey,
    RetrievalQueryBlockKey,
    TrialKey,
)
from .schema import ExperimentSpec, PackDefinition, VariantSpec


class PlanError(ValueError):
    pass


@dataclass(frozen=True)
class PlannedTrial:
    key: TrialKey
    pair_memberships: tuple[PairKey, ...]


@dataclass(frozen=True)
class PlannedComparisonBlock:
    key: ComparisonBlockKey
    trial_keys: tuple[TrialKey, ...]


@dataclass(frozen=True)
class PlannedRetrievalCase:
    key: RetrievalCaseKey
    query_block: RetrievalQueryBlockKey


@dataclass(frozen=True)
class PlannedRetrievalQueryBlock:
    key: RetrievalQueryBlockKey
    case_keys: tuple[RetrievalCaseKey, ...]


@dataclass(frozen=True)
class ExperimentPlan:
    experiment_id: str
    spec_payload: dict[str, Any]
    pack_definitions: tuple[PackDefinition, ...]
    trials: tuple[PlannedTrial, ...]
    comparison_blocks: tuple[PlannedComparisonBlock, ...]
    pairs: tuple[PairKey, ...]
    retrieval_cases: tuple[PlannedRetrievalCase, ...]
    retrieval_query_blocks: tuple[PlannedRetrievalQueryBlock, ...]

    @property
    def plan_digest(self) -> str:
        return self.experiment_id

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.spec_payload["schema"],
            "evidence_schema": self.spec_payload["evidence_schema"],
            "experiment_id": self.experiment_id,
            "plan_digest": self.plan_digest,
            "spec": self.spec_payload,
            "pack_definitions": to_primitive(self.pack_definitions),
            "trials": to_primitive(self.trials),
            "comparison_blocks": to_primitive(self.comparison_blocks),
            "pairs": to_primitive(self.pairs),
            "retrieval_cases": to_primitive(self.retrieval_cases),
            "retrieval_query_blocks": to_primitive(self.retrieval_query_blocks),
        }


def compile_plan(spec: ExperimentSpec, packs: tuple[Pack, ...]) -> ExperimentPlan:
    definitions = tuple(pack.definition() for pack in packs)
    actual_pack_ids = tuple(definition.pack_id for definition in definitions)
    if actual_pack_ids != spec.pack_ids:
        raise PlanError(f"registry returned Packs {actual_pack_ids!r}, expected {spec.pack_ids!r}")
    spec_payload = spec.canonical_payload()
    identity_payload = {
        "spec": spec_payload,
        "pack_definitions": to_primitive(definitions),
    }
    validate_plan_value(identity_payload)
    experiment_id = canonical_digest(identity_payload)

    trials: list[PlannedTrial] = []
    blocks: list[PlannedComparisonBlock] = []
    pairs: list[PairKey] = []
    retrieval_cases: list[PlannedRetrievalCase] = []
    retrieval_blocks: list[PlannedRetrievalQueryBlock] = []

    for definition in definitions:
        variants = {variant.variant_id: variant for variant in definition.variants}
        for pair in definition.pairs:
            try:
                control = variants[pair.control_variant_id]
                treatment = variants[pair.treatment_variant_id]
            except KeyError as exc:
                raise PlanError(f"Pair references unknown variant in {definition.pack_id}") from exc
            diff = variant_diff(control, treatment)
            if set(diff) != {pair.treatment_axis}:
                raise PlanError(
                    f"Pair axis {pair.treatment_axis!r} does not match variant diff "
                    f"{sorted(diff)!r} in {definition.pack_id}"
                )

        for task in definition.tasks:
            for repetition in range(spec.repetitions):
                block_key = ComparisonBlockKey(
                    experiment_id=experiment_id,
                    pack_id=definition.pack_id,
                    task_id=task.task_id,
                    repetition=repetition,
                )
                pair_keys = tuple(
                    PairKey(
                        experiment_id=experiment_id,
                        pack_id=definition.pack_id,
                        treatment_axis=pair.treatment_axis,
                        task_id=task.task_id,
                        repetition=repetition,
                        control_variant_id=pair.control_variant_id,
                        treatment_variant_id=pair.treatment_variant_id,
                    )
                    for pair in definition.pairs
                )
                pairs.extend(pair_keys)
                trial_keys: list[TrialKey] = []
                for variant in definition.variants:
                    key = TrialKey(
                        experiment_id=experiment_id,
                        pack_id=definition.pack_id,
                        task_id=task.task_id,
                        variant_id=variant.variant_id,
                        repetition=repetition,
                    )
                    memberships = tuple(
                        pair_key
                        for pair_key in pair_keys
                        if variant.variant_id
                        in {
                            pair_key.control_variant_id,
                            pair_key.treatment_variant_id,
                        }
                    )
                    trials.append(PlannedTrial(key=key, pair_memberships=memberships))
                    trial_keys.append(key)
                ordered = tuple(
                    sorted(
                        trial_keys,
                        key=lambda key: canonical_digest(
                            {
                                "experiment_id": experiment_id,
                                "block": to_primitive(block_key),
                                "variant_id": key.variant_id,
                            }
                        ),
                    )
                )
                blocks.append(PlannedComparisonBlock(key=block_key, trial_keys=ordered))

        for suite in definition.retrieval_suites:
            for query in suite.queries:
                block_key = RetrievalQueryBlockKey(
                    experiment_id=experiment_id,
                    retrieval_suite_id=suite.retrieval_suite_id,
                    query_id=query.query_id,
                )
                case_keys = tuple(
                    RetrievalCaseKey(
                        experiment_id=experiment_id,
                        retrieval_suite_id=suite.retrieval_suite_id,
                        query_id=query.query_id,
                        configuration_id=configuration.configuration_id,
                    )
                    for configuration in suite.configurations
                )
                retrieval_blocks.append(PlannedRetrievalQueryBlock(key=block_key, case_keys=case_keys))
                retrieval_cases.extend(PlannedRetrievalCase(key=key, query_block=block_key) for key in case_keys)

    return ExperimentPlan(
        experiment_id=experiment_id,
        spec_payload=spec_payload,
        pack_definitions=definitions,
        trials=tuple(trials),
        comparison_blocks=tuple(blocks),
        pairs=tuple(pairs),
        retrieval_cases=tuple(retrieval_cases),
        retrieval_query_blocks=tuple(retrieval_blocks),
    )


def validate_manifest_identity(
    manifest: Mapping[str, Any],
    *,
    experiment_id: str,
) -> None:
    if manifest.get("experiment_id") != experiment_id or manifest.get("plan_digest") != experiment_id:
        raise PlanError(
            "manifest identity does not match the experiment reference",
        )
    spec = manifest.get("spec")
    pack_definitions = manifest.get("pack_definitions")
    if not isinstance(spec, Mapping) or not isinstance(
        pack_definitions,
        list | tuple,
    ):
        raise PlanError(
            "manifest identity payload is malformed",
        )
    if (
        "schema" not in manifest
        or "evidence_schema" not in manifest
        or "schema" not in spec
        or "evidence_schema" not in spec
        or manifest["schema"] != spec["schema"]
        or manifest["evidence_schema"] != spec["evidence_schema"]
    ):
        raise PlanError(
            "manifest identity schema does not match the frozen specification",
        )
    actual_digest = canonical_digest(
        {
            "spec": spec,
            "pack_definitions": pack_definitions,
        }
    )
    if actual_digest != experiment_id:
        raise PlanError(
            "manifest identity payload digest does not match the experiment reference",
        )


def manifest_derived_plan_matches_identity(
    manifest: Mapping[str, Any],
) -> bool:
    try:
        expected = _derive_manifest_plan(
            experiment_id=manifest["experiment_id"],
            spec=manifest["spec"],
            definitions=manifest["pack_definitions"],
        )
    except (KeyError, TypeError, ValueError):
        return False
    return all(
        manifest.get(field) == expected[field]
        for field in (
            "trials",
            "comparison_blocks",
            "pairs",
            "retrieval_cases",
            "retrieval_query_blocks",
        )
    )


def _derive_manifest_plan(
    *,
    experiment_id: Any,
    spec: Any,
    definitions: Any,
) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(experiment_id, str) or not isinstance(spec, Mapping) or not isinstance(definitions, list | tuple):
        raise TypeError("manifest identity inputs are malformed")
    repetitions = spec["repetitions"]
    pack_ids = spec["pack_ids"]
    if (
        not isinstance(repetitions, int)
        or isinstance(repetitions, bool)
        or repetitions < 1
        or not isinstance(pack_ids, list | tuple)
    ):
        raise ValueError("manifest experiment specification is malformed")

    trials: list[dict[str, Any]] = []
    comparison_blocks: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    retrieval_cases: list[dict[str, Any]] = []
    retrieval_query_blocks: list[dict[str, Any]] = []
    actual_pack_ids: list[str] = []

    for definition in definitions:
        if not isinstance(definition, Mapping):
            raise TypeError("manifest Pack definition is malformed")
        pack_id = definition["pack_id"]
        tasks = definition["tasks"]
        variants = definition["variants"]
        pair_specs = definition["pairs"]
        retrieval_suites = definition["retrieval_suites"]
        if (
            not isinstance(pack_id, str)
            or not isinstance(tasks, list | tuple)
            or not isinstance(variants, list | tuple)
            or not isinstance(pair_specs, list | tuple)
            or not isinstance(retrieval_suites, list | tuple)
        ):
            raise TypeError("manifest Pack definition fields are malformed")
        actual_pack_ids.append(pack_id)

        for task in tasks:
            if not isinstance(task, Mapping):
                raise TypeError("manifest Task is malformed")
            task_id = task["task_id"]
            if not isinstance(task_id, str):
                raise TypeError("manifest task_id is malformed")
            for repetition in range(repetitions):
                block_key = {
                    "experiment_id": experiment_id,
                    "pack_id": pack_id,
                    "task_id": task_id,
                    "repetition": repetition,
                }
                pair_keys: list[dict[str, Any]] = []
                for pair in pair_specs:
                    if not isinstance(pair, Mapping):
                        raise TypeError("manifest Pair is malformed")
                    pair_key = {
                        "experiment_id": experiment_id,
                        "pack_id": pack_id,
                        "treatment_axis": pair["treatment_axis"],
                        "task_id": task_id,
                        "repetition": repetition,
                        "control_variant_id": pair["control_variant_id"],
                        "treatment_variant_id": pair["treatment_variant_id"],
                    }
                    pair_keys.append(pair_key)
                    pairs.append(pair_key)

                trial_keys: list[dict[str, Any]] = []
                for variant in variants:
                    if not isinstance(variant, Mapping):
                        raise TypeError(
                            "manifest Variant is malformed",
                        )
                    variant_id = variant["variant_id"]
                    if not isinstance(variant_id, str):
                        raise TypeError(
                            "manifest variant_id is malformed",
                        )
                    trial_key = {
                        "experiment_id": experiment_id,
                        "pack_id": pack_id,
                        "task_id": task_id,
                        "variant_id": variant_id,
                        "repetition": repetition,
                    }
                    memberships = [
                        pair_key
                        for pair_key in pair_keys
                        if variant_id
                        in {
                            pair_key["control_variant_id"],
                            pair_key["treatment_variant_id"],
                        }
                    ]
                    trials.append(
                        {
                            "key": trial_key,
                            "pair_memberships": memberships,
                        }
                    )
                    trial_keys.append(trial_key)
                ordered_trial_keys = sorted(
                    trial_keys,
                    key=lambda key: canonical_digest(
                        {
                            "experiment_id": experiment_id,
                            "block": block_key,
                            "variant_id": key["variant_id"],
                        }
                    ),
                )
                comparison_blocks.append(
                    {
                        "key": block_key,
                        "trial_keys": ordered_trial_keys,
                    }
                )

        for suite in retrieval_suites:
            if not isinstance(suite, Mapping):
                raise TypeError(
                    "manifest Retrieval Suite is malformed",
                )
            retrieval_suite_id = suite["retrieval_suite_id"]
            queries = suite["queries"]
            configurations = suite["configurations"]
            if (
                not isinstance(retrieval_suite_id, str)
                or not isinstance(queries, list | tuple)
                or not isinstance(configurations, list | tuple)
            ):
                raise TypeError(
                    "manifest Retrieval Suite fields are malformed",
                )
            for query in queries:
                if not isinstance(query, Mapping):
                    raise TypeError(
                        "manifest Retrieval Query is malformed",
                    )
                query_id = query["query_id"]
                if not isinstance(query_id, str):
                    raise TypeError(
                        "manifest query_id is malformed",
                    )
                query_block = {
                    "experiment_id": experiment_id,
                    "retrieval_suite_id": retrieval_suite_id,
                    "query_id": query_id,
                }
                case_keys: list[dict[str, Any]] = []
                for configuration in configurations:
                    if not isinstance(configuration, Mapping):
                        raise TypeError(
                            "manifest Retrieval Configuration is malformed",
                        )
                    configuration_id = configuration["configuration_id"]
                    if not isinstance(configuration_id, str):
                        raise TypeError(
                            "manifest configuration_id is malformed",
                        )
                    case_key = {
                        "experiment_id": experiment_id,
                        "retrieval_suite_id": retrieval_suite_id,
                        "query_id": query_id,
                        "configuration_id": configuration_id,
                    }
                    case_keys.append(case_key)
                    retrieval_cases.append(
                        {
                            "key": case_key,
                            "query_block": query_block,
                        }
                    )
                retrieval_query_blocks.append(
                    {
                        "key": query_block,
                        "case_keys": case_keys,
                    }
                )
    if list(pack_ids) != actual_pack_ids:
        raise ValueError(
            "manifest Pack definitions do not match the specification",
        )
    return {
        "trials": trials,
        "comparison_blocks": comparison_blocks,
        "pairs": pairs,
        "retrieval_cases": retrieval_cases,
        "retrieval_query_blocks": retrieval_query_blocks,
    }


def variant_diff(
    control: VariantSpec | dict[str, Any],
    treatment: VariantSpec | dict[str, Any],
) -> dict[str, Any]:
    control_settings = dict(control.settings) if isinstance(control, VariantSpec) else dict(control)
    treatment_settings = dict(treatment.settings) if isinstance(treatment, VariantSpec) else dict(treatment)
    return {
        key: {
            "control": to_primitive(control_settings.get(key)),
            "treatment": to_primitive(treatment_settings.get(key)),
        }
        for key in sorted(control_settings.keys() | treatment_settings.keys())
        if control_settings.get(key) != treatment_settings.get(key)
    }
