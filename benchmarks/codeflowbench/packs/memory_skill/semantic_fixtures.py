"""Frozen historical semantic fixtures; labels must not be relabeled."""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.codeflowbench.schema import RetrievalQuerySpec

from .fixtures import anonymous_item_id
from .models import MemoryFact, SkillItem

_CUSTOMERS = (
    "Northwind",
    "Contoso",
    "Adventure Works",
    "Fabrikam",
    "Tailspin Toys",
    "Woodgrove Bank",
    "Alpine Ski House",
    "Blue Yonder Airlines",
    "Coho Winery",
    "Wide World Importers",
)
_SERVICES = (
    "inventory API",
    "billing worker",
    "shipment scheduler",
    "fraud detector",
    "support portal",
)
_REGIONS = (
    "Singapore",
    "Sydney",
    "Tokyo",
    "Mumbai",
    "Frankfurt",
    "Dublin",
    "Montreal",
    "Sao Paulo",
    "Cape Town",
    "Seoul",
)
_OLD_REGIONS = (
    "Virginia",
    "London",
    "Paris",
    "Oregon",
    "Zurich",
    "Stockholm",
    "Osaka",
    "Dubai",
    "Madrid",
    "Milan",
)
_RETENTIONS = (
    "thirty days",
    "forty-five days",
    "sixty days",
    "ninety days",
    "one hundred and twenty days",
)
_FAILURES = (
    "a deployment that is stuck during traffic cutover",
    "a queue consumer that repeatedly loses its lease",
    "an API that returns stale data after a schema migration",
    "a scheduled job that runs twice after a restart",
    "a cache cluster with an uneven shard distribution",
    "a webhook whose delivery acknowledgements arrive late",
    "a worker pool that exhausts database connections",
    "a service whose health checks fail only in one region",
)
_ACTIONS = (
    "inspect the health probes, drain traffic, and roll back the release",
    "compare lease timestamps, stop duplicate workers, and renew ownership",
    "verify the migration version, invalidate stale entries, and warm the cache",
    "check the claim ledger, suppress the duplicate run, and reconcile receipts",
    "measure per-shard load, move the hottest keys, and rebalance replicas",
    "deduplicate by event identifier, verify the receipt, and replay missing events",
    "trace connection ownership, cap concurrency, and recycle leaked sessions",
    "compare regional dependencies, isolate the failing zone, and reroute traffic",
)
_ENVIRONMENTS = (
    "the production environment",
    "the disaster-recovery environment",
    "the staging environment",
    "the regulated customer environment",
    "the overnight batch environment",
)

_V2_ORGANIZATIONS = (
    "Meridian Labs",
    "Red Cedar Health",
    "Harborline Logistics",
    "Juniper Media",
    "Atlas Field Services",
    "Nimbus Retail",
    "Solstice Energy",
    "Pinecone Legal",
    "Kestrel Manufacturing",
    "Ember Education",
)
_V2_COMPONENTS = (
    "entitlement service",
    "catalog reconciler",
    "telemetry collector",
    "invoice notifier",
    "audit exporter",
    "event ingestor",
    "ledger writer",
    "metrics relay",
    "access broker",
    "policy compiler",
)
_V2_KEY_RINGS = (
    "Quartz",
    "Amber",
    "Cobalt",
    "Jade",
    "Onyx",
    "Pearl",
    "Saffron",
    "Indigo",
    "Silver",
    "Violet",
)
_V2_ROTATION_DAYS = (17, 23, 29, 31, 37, 41, 43, 47, 53, 59)
_V2_RETENTION_WEEKS = (11, 13, 17, 19, 23)
_V2_INCIDENTS = (
    "a dead-letter backlog growing after a consumer rollout",
    "a stale leader epoch rejecting newly committed writes",
    "a certificate rollover leaving half the workers disconnected",
    "a replay gap after the offset ledger was compacted",
    "clock skew causing signed requests to expire immediately",
    "a poison record cycling through the retry queue",
    "a lease handoff leaving two coordinators active",
    "a schema fence blocking only the canary partition",
    "a checksum mismatch after snapshot restoration",
    "a rate limiter retaining permits after client disconnects",
)
_V2_RESPONSES = (
    "quarantine the failing batch, inspect the dead-letter age, then replay it in bounded slices",
    "compare the committed epoch, fence the stale leader, then re-elect from the latest replica",
    "verify the trust bundle, drain disconnected workers, then restart them with the new certificate",
    "diff the offset ledger, restore the missing range, then replay from the last verified checkpoint",
    "compare monotonic and wall clocks, resynchronize the hosts, then refresh the signed requests",
    "isolate the record fingerprint, cap retry attempts, then route it to manual review",
    "inspect lease generations, revoke the older coordinator, then reconcile outstanding ownership",
    "compare schema versions, remove the stale fence, then advance the canary partition",
    "validate chunk hashes, replace the damaged segment, then rerun the snapshot verification",
    "audit permit ownership, reclaim disconnected leases, then lower concurrency during recovery",
)
_V2_OPERATING_CONTEXTS = (
    "the regulated production cell",
    "the regional failover cell",
    "the overnight settlement cell",
    "the partner integration cell",
    "the canary verification cell",
)
_V2_CALIBRATION_ORGANIZATIONS = (
    "Lumen Research",
    "Copper Ridge Foods",
    "Northstar Transit",
    "Willow Publishing",
    "Granite Repair",
    "Cloudberry Markets",
    "Aurora Utilities",
    "Blue Heron Counsel",
    "Forgepoint Robotics",
    "Cedar Grove Schools",
)
_V2_CALIBRATION_COMPONENTS = (
    "token refresher",
    "shipment matcher",
    "signal aggregator",
    "receipt dispatcher",
    "compliance archiver",
    "message normalizer",
    "balance projector",
    "health sampler",
    "identity mediator",
    "rules translator",
)
_V2_CALIBRATION_INCIDENTS = (
    "an acknowledgement window closing before consumers commit",
    "a replica generation lagging behind the elected coordinator",
    "a trust-store refresh leaving warm workers on the old chain",
    "a compaction pass omitting a narrow sequence range",
    "host drift invalidating newly issued access assertions",
    "a malformed envelope returning repeatedly from quarantine",
    "an ownership transfer retaining the previous lease holder",
    "a compatibility lock stopping only the preview shard",
    "a restored archive failing its segment digest check",
    "a concurrency guard leaking slots after aborted requests",
)
_V2_CALIBRATION_RESPONSES = (
    "pause the affected consumers, compare acknowledgement age, then replay bounded windows",
    "read the durable generation, revoke the lagging replica, then elect from the newest copy",
    "compare trust anchors, drain the stale workers, then restart them against the refreshed chain",
    "reconcile the sequence ledger, restore the omitted range, then resume from a verified cursor",
    "compare clock sources, correct the drifting hosts, then reissue the access assertions",
    "fingerprint the envelope, stop automatic retries, then send it for operator inspection",
    "compare lease versions, fence the prior holder, then reconcile uncommitted ownership",
    "inspect compatibility versions, clear the obsolete lock, then advance the preview shard",
    "verify segment digests, replace the corrupt archive part, then rerun restoration checks",
    "trace slot ownership, reclaim aborted requests, then resume with reduced concurrency",
)
_V2_CALIBRATION_CONTEXTS = (
    "the validation production cell",
    "the rehearsal failover cell",
    "the preflight settlement cell",
    "the sandbox partner cell",
    "the preview verification cell",
)


@dataclass(frozen=True)
class SemanticFixture:
    track: str
    memory_suite_id: str
    skill_suite_id: str
    memory_corpus: tuple[MemoryFact, ...]
    skill_corpus: tuple[SkillItem, ...]
    memory_queries: tuple[RetrievalQuerySpec, ...]
    skill_queries: tuple[RetrievalQuerySpec, ...]

    @property
    def planned_cases(self) -> int:
        return len(self.memory_queries) + len(self.skill_queries) * 3


def semantic_fixture(
    track: str,
    *,
    version: str = "v1",
) -> SemanticFixture:
    if version == "v1":
        return _semantic_fixture_v1(track)
    if version == "v2":
        return _semantic_fixture_v2(track)
    raise ValueError(f"unknown semantic fixture version: {version}")


def _semantic_fixture_v1(track: str) -> SemanticFixture:
    if track == "calibration":
        active_count = 6
        cross_count = 4
        noise_count = 0
        memory_negative_count = 4
        skill_count = 5
        skill_cross_count = 4
        skill_noise_count = 0
        skill_negative_count = 3
        width = 2
    elif track == "formal":
        active_count = 50
        cross_count = 30
        noise_count = 30
        memory_negative_count = 30
        skill_count = 40
        skill_cross_count = 20
        skill_noise_count = 20
        skill_negative_count = 20
        width = 3
    else:
        raise ValueError(f"unknown semantic fixture track: {track}")

    prefix = "cal-" if track == "calibration" else ""
    memory_suite_id = f"user-memory-semantic-{track}-v1"
    skill_suite_id = f"skill-source-fusion-semantic-{track}-v1"
    active = tuple(
        MemoryFact(
            item_id=f"{prefix}memory-active-{index:0{width}d}",
            workspace_id=_workspace(track, index),
            text=_memory_fact(index),
        )
        for index in range(active_count)
    )
    stale = tuple(
        MemoryFact(
            item_id=f"{prefix}memory-stale-{index:0{width}d}",
            workspace_id=_workspace(track, index),
            text=_stale_memory_fact(index),
            active=False,
            superseded=True,
        )
        for index in range(active_count)
    )
    cross = tuple(
        MemoryFact(
            item_id=f"{prefix}memory-cross-{index:0{width}d}",
            workspace_id=_workspace(track, index + 1),
            text=_memory_fact(index),
        )
        for index in range(cross_count)
    )
    noise = tuple(
        MemoryFact(
            item_id=f"memory-noise-{index:03d}",
            workspace_id=_workspace(track, index),
            text=_memory_noise(index),
        )
        for index in range(noise_count)
    )
    memory_queries = tuple(
        RetrievalQuerySpec(
            query_id=f"{prefix}memory-positive-{index:0{width}d}",
            label="positive",
            expected_item_ids=(
                anonymous_item_id(
                    memory_suite_id,
                    f"{prefix}memory-active-{index:0{width}d}",
                ),
            ),
            payload={
                "query_text": _memory_query(index),
                "workspace_id": _workspace(track, index),
                "consuming_turn": f"{track}-memory-turn-{index:03d}",
            },
        )
        for index in range(active_count)
    ) + tuple(
        RetrievalQuerySpec(
            query_id=f"{prefix}memory-negative-{index:0{width}d}",
            label="hard_negative",
            payload={
                "query_text": _memory_negative_query(index),
                "workspace_id": _workspace(track, index),
                "consuming_turn": f"{track}-memory-negative-turn-{index:03d}",
            },
        )
        for index in range(memory_negative_count)
    )

    skill_items: list[SkillItem] = []
    for index in range(skill_count):
        if index < (3 if track == "calibration" else 25):
            skill_items.append(
                SkillItem(
                    item_id=f"{prefix}skill-local-{index:0{width}d}",
                    logical_id=f"{prefix}skill-{index:0{width}d}",
                    workspace_id=_workspace(track, index),
                    source="local",
                    text=_local_skill(index),
                )
            )
        if index >= (2 if track == "calibration" else 15):
            skill_items.append(
                SkillItem(
                    item_id=f"{prefix}skill-everos-{index:0{width}d}",
                    logical_id=f"{prefix}skill-{index:0{width}d}",
                    workspace_id=_workspace(track, index),
                    source="everos",
                    text=_semantic_skill(index),
                )
            )
    skill_items.extend(
        SkillItem(
            item_id=f"{prefix}skill-cross-{index:0{width}d}",
            logical_id=f"{prefix}skill-cross-{index:0{width}d}",
            workspace_id=_workspace(track, index + 1),
            source="everos" if index % 2 else "local",
            text=_semantic_skill(index),
        )
        for index in range(skill_cross_count)
    )
    skill_items.extend(
        SkillItem(
            item_id=f"skill-noise-{index:03d}",
            logical_id=f"skill-noise-{index:03d}",
            workspace_id=_workspace(track, index),
            source="everos" if index % 2 else "local",
            text=_skill_noise(index),
        )
        for index in range(skill_noise_count)
    )
    skill_queries = tuple(
        RetrievalQuerySpec(
            query_id=f"{prefix}skill-positive-{index:0{width}d}",
            label="positive",
            expected_item_ids=(
                anonymous_item_id(
                    skill_suite_id,
                    f"{prefix}skill-{index:0{width}d}",
                ),
            ),
            payload={
                "query_text": _skill_query(index),
                "workspace_id": _workspace(track, index),
                "consuming_turn": f"{track}-skill-turn-{index:03d}",
            },
        )
        for index in range(skill_count)
    ) + tuple(
        RetrievalQuerySpec(
            query_id=f"{prefix}skill-negative-{index:0{width}d}",
            label="hard_negative",
            payload={
                "query_text": _skill_negative_query(index),
                "workspace_id": _workspace(track, index),
                "consuming_turn": f"{track}-skill-negative-turn-{index:03d}",
            },
        )
        for index in range(skill_negative_count)
    )
    return SemanticFixture(
        track=track,
        memory_suite_id=memory_suite_id,
        skill_suite_id=skill_suite_id,
        memory_corpus=(*active, *stale, *cross, *noise),
        skill_corpus=tuple(skill_items),
        memory_queries=memory_queries,
        skill_queries=skill_queries,
    )


def _semantic_fixture_v2(track: str) -> SemanticFixture:
    if track == "calibration":
        memory_active_count = 8
        memory_cross_count = 4
        memory_negative_count = 4
        memory_noise_per_workspace = 10
        skill_count = 8
        skill_negative_count = 4
        workspace_count = 2
        width = 2
    elif track == "formal":
        memory_active_count = 50
        memory_cross_count = 30
        memory_negative_count = 30
        memory_noise_per_workspace = 5
        skill_count = 40
        skill_negative_count = 20
        workspace_count = 8
        width = 3
    else:
        raise ValueError(f"unknown semantic fixture track: {track}")

    prefix = "v2-cal-" if track == "calibration" else "v2-"
    memory_suite_id = f"user-memory-semantic-{track}-v2"
    skill_suite_id = f"skill-source-fusion-semantic-{track}-v2"
    active = tuple(
        MemoryFact(
            item_id=f"{prefix}memory-active-{index:0{width}d}",
            workspace_id=_v2_workspace(track, index),
            text=_v2_memory_fact(track, index),
        )
        for index in range(memory_active_count)
    )
    stale = tuple(
        MemoryFact(
            item_id=f"{prefix}memory-stale-{index:0{width}d}",
            workspace_id=_v2_workspace(track, index),
            text=_v2_stale_memory_fact(track, index),
            active=False,
            superseded=True,
        )
        for index in range(memory_active_count)
    )
    cross = tuple(
        MemoryFact(
            item_id=f"{prefix}memory-cross-{index:0{width}d}",
            workspace_id=_v2_workspace(track, index + 1),
            text=_v2_memory_fact(track, index),
        )
        for index in range(memory_cross_count)
    )
    memory_noise = tuple(
        MemoryFact(
            item_id=f"{prefix}memory-distractor-{index:03d}",
            workspace_id=_v2_workspace(track, index),
            text=_v2_memory_noise(track, index),
        )
        for index in range(memory_noise_per_workspace * workspace_count)
    )
    memory_queries = tuple(
        RetrievalQuerySpec(
            query_id=f"{prefix}memory-positive-{index:0{width}d}",
            label="positive",
            expected_item_ids=(
                anonymous_item_id(
                    memory_suite_id,
                    f"{prefix}memory-active-{index:0{width}d}",
                ),
            ),
            payload={
                "query_text": _v2_memory_query(track, index),
                "workspace_id": _v2_workspace(track, index),
                "consuming_turn": f"{prefix}memory-turn-{index:03d}",
            },
        )
        for index in range(memory_active_count)
    ) + tuple(
        RetrievalQuerySpec(
            query_id=f"{prefix}memory-negative-{index:0{width}d}",
            label="hard_negative",
            payload={
                "query_text": _v2_memory_negative_query(track, index),
                "workspace_id": _v2_workspace(track, index),
                "consuming_turn": (f"{prefix}memory-negative-turn-{index:03d}"),
            },
        )
        for index in range(memory_negative_count)
    )

    local_cutoff = 5 if track == "calibration" else 25
    everos_start = 3 if track == "calibration" else 15
    skill_items: list[SkillItem] = []
    for index in range(skill_count):
        logical_id = f"{prefix}skill-{index:0{width}d}"
        if index < local_cutoff:
            skill_items.append(
                SkillItem(
                    item_id=f"{prefix}skill-local-{index:0{width}d}",
                    logical_id=logical_id,
                    workspace_id=_v2_workspace(track, index),
                    source="local",
                    text=_v2_local_skill(track, index),
                )
            )
        if index >= everos_start:
            skill_items.append(
                SkillItem(
                    item_id=f"{prefix}skill-everos-{index:0{width}d}",
                    logical_id=logical_id,
                    workspace_id=_v2_workspace(track, index),
                    source="everos",
                    text=_v2_semantic_skill(track, index),
                )
            )
    for source in ("local", "everos"):
        for index in range(10 * workspace_count):
            skill_items.append(
                SkillItem(
                    item_id=(f"{prefix}skill-{source}-distractor-{index:03d}"),
                    logical_id=(f"{prefix}skill-{source}-distractor-{index:03d}"),
                    workspace_id=_v2_workspace(track, index),
                    source=source,
                    text=_v2_skill_noise(track, index, source),
                )
            )
    skill_queries = tuple(
        RetrievalQuerySpec(
            query_id=f"{prefix}skill-positive-{index:0{width}d}",
            label="positive",
            expected_item_ids=(
                anonymous_item_id(
                    skill_suite_id,
                    f"{prefix}skill-{index:0{width}d}",
                ),
            ),
            payload={
                "query_text": _v2_skill_query(track, index),
                "workspace_id": _v2_workspace(track, index),
                "consuming_turn": f"{prefix}skill-turn-{index:03d}",
            },
        )
        for index in range(skill_count)
    ) + tuple(
        RetrievalQuerySpec(
            query_id=f"{prefix}skill-negative-{index:0{width}d}",
            label="hard_negative",
            payload={
                "query_text": _v2_skill_negative_query(track, index),
                "workspace_id": _v2_workspace(track, index),
                "consuming_turn": (f"{prefix}skill-negative-turn-{index:03d}"),
            },
        )
        for index in range(skill_negative_count)
    )
    return SemanticFixture(
        track=track,
        memory_suite_id=memory_suite_id,
        skill_suite_id=skill_suite_id,
        memory_corpus=(*active, *stale, *cross, *memory_noise),
        skill_corpus=tuple(skill_items),
        memory_queries=memory_queries,
        skill_queries=skill_queries,
    )


def _workspace(track: str, index: int) -> str:
    count = 4 if track == "calibration" else 8
    return f"semantic-{track}-workspace-{index % count}"


def _memory_fact(index: int) -> str:
    customer = _CUSTOMERS[index % len(_CUSTOMERS)]
    service = _SERVICES[(index // len(_CUSTOMERS)) % len(_SERVICES)]
    region = _REGIONS[index % len(_REGIONS)]
    retention = _RETENTIONS[(index // len(_SERVICES)) % len(_RETENTIONS)]
    return (
        f"For {customer}'s {service}, disaster-recovery backups must be "
        f"stored in {region} and retained for {retention}."
    )


def _stale_memory_fact(index: int) -> str:
    customer = _CUSTOMERS[index % len(_CUSTOMERS)]
    service = _SERVICES[(index // len(_CUSTOMERS)) % len(_SERVICES)]
    region = _OLD_REGIONS[index % len(_OLD_REGIONS)]
    return f"An obsolete runbook said that {customer}'s {service} backups were stored in {region} for seven days."


def _memory_query(index: int) -> str:
    customer = _CUSTOMERS[index % len(_CUSTOMERS)]
    service = _SERVICES[(index // len(_CUSTOMERS)) % len(_SERVICES)]
    return (
        f"Where should we keep disaster-recovery backups for "
        f"{customer}'s {service}, and how long should we retain them?"
    )


def _memory_negative_query(index: int) -> str:
    customer = _CUSTOMERS[index % len(_CUSTOMERS)]
    return f"Who must approve international travel expenses for contractors working with {customer}?"


def _memory_noise(index: int) -> str:
    customer = _CUSTOMERS[index % len(_CUSTOMERS)]
    return f"{customer} prefers monthly accessibility reviews for the public documentation portal."


def _local_skill(index: int) -> str:
    environment = _ENVIRONMENTS[(index // len(_FAILURES)) % len(_ENVIRONMENTS)]
    service = _SERVICES[index % len(_SERVICES)]
    return (
        f"When {service} has {_FAILURES[index % len(_FAILURES)]} in {environment}, {_ACTIONS[index % len(_ACTIONS)]}."
    )


def _semantic_skill(index: int) -> str:
    environment = _ENVIRONMENTS[(index // len(_FAILURES)) % len(_ENVIRONMENTS)]
    service = _SERVICES[index % len(_SERVICES)]
    return (
        f"This procedure restores {service} when it has "
        f"{_FAILURES[index % len(_FAILURES)]} in {environment}. "
        f"First {_ACTIONS[index % len(_ACTIONS)]}, then confirm recovery "
        "with the service-level checks."
    )


def _skill_query(index: int) -> str:
    environment = _ENVIRONMENTS[(index // len(_FAILURES)) % len(_ENVIRONMENTS)]
    service = _SERVICES[index % len(_SERVICES)]
    return f"How should I recover {service} from {_FAILURES[index % len(_FAILURES)]} in {environment}?"


def _skill_negative_query(index: int) -> str:
    customer = _CUSTOMERS[index % len(_CUSTOMERS)]
    return f"How should I prepare a quarterly brand photography brief for {customer}?"


def _skill_noise(index: int) -> str:
    return (
        f"Prepare a customer newsletter by selecting photographs, reviewing "
        f"captions, and scheduling the edition for week {index + 1}."
    )


def _v2_workspace(track: str, index: int) -> str:
    count = 2 if track == "calibration" else 8
    return f"semantic-v2-{track}-workspace-{index % count}"


def _v2_organizations(track: str) -> tuple[str, ...]:
    if track == "calibration":
        return _V2_CALIBRATION_ORGANIZATIONS
    return _V2_ORGANIZATIONS


def _v2_components(track: str) -> tuple[str, ...]:
    if track == "calibration":
        return _V2_CALIBRATION_COMPONENTS
    return _V2_COMPONENTS


def _v2_incidents(track: str) -> tuple[str, ...]:
    if track == "calibration":
        return _V2_CALIBRATION_INCIDENTS
    return _V2_INCIDENTS


def _v2_responses(track: str) -> tuple[str, ...]:
    if track == "calibration":
        return _V2_CALIBRATION_RESPONSES
    return _V2_RESPONSES


def _v2_contexts(track: str) -> tuple[str, ...]:
    if track == "calibration":
        return _V2_CALIBRATION_CONTEXTS
    return _V2_OPERATING_CONTEXTS


def _v2_memory_fact(track: str, index: int) -> str:
    organizations = _v2_organizations(track)
    components = _v2_components(track)
    organization = organizations[index % len(organizations)]
    component = components[(index // len(organizations)) % len(components)]
    key_ring = _V2_KEY_RINGS[index % len(_V2_KEY_RINGS)]
    rotation = _V2_ROTATION_DAYS[index % len(_V2_ROTATION_DAYS)]
    retention = _V2_RETENTION_WEEKS[(index // len(components)) % len(_V2_RETENTION_WEEKS)]
    templates = (
        (
            f"The current continuity rule for {organization}'s {component} "
            f"uses the {key_ring} key ring, rotates escrow keys every "
            f"{rotation} days, and retains snapshots for {retention} weeks."
        ),
        (
            f"For {organization}, protect {component} recovery snapshots "
            f"with the {key_ring} key ring. Rekey after {rotation} days and "
            f"keep each snapshot for {retention} weeks."
        ),
        (
            f"{organization}'s active recovery policy names {key_ring} as "
            f"the key ring for {component}; the rotation interval is "
            f"{rotation} days and the retention window is {retention} weeks."
        ),
        (
            f"Audited settings for {organization}'s {component}: escrow "
            f"encryption ring {key_ring}, key renewal cadence {rotation} "
            f"days, snapshot lifetime {retention} weeks."
        ),
    )
    return templates[index % len(templates)]


def _v2_stale_memory_fact(track: str, index: int) -> str:
    organizations = _v2_organizations(track)
    components = _v2_components(track)
    organization = organizations[index % len(organizations)]
    component = components[(index // len(organizations)) % len(components)]
    key_ring = _V2_KEY_RINGS[(index + 3) % len(_V2_KEY_RINGS)]
    return (
        f"A retired continuity memo for {organization}'s {component} used "
        f"the {key_ring} key ring, renewed keys every 90 days, and deleted "
        "snapshots after four weeks."
    )


def _v2_memory_query(track: str, index: int) -> str:
    organizations = _v2_organizations(track)
    components = _v2_components(track)
    organization = organizations[index % len(organizations)]
    component = components[(index // len(organizations)) % len(components)]
    templates = (
        (
            f"A recovery audit needs the present escrow policy for "
            f"{organization}'s {component}. Which key ring, renewal "
            "interval, and retention window apply?"
        ),
        (
            f"State the active snapshot-protection settings for "
            f"{organization}'s {component}: encryption ring, rekey cadence, "
            "and how long snapshots remain available."
        ),
        (
            f"Before restoring {organization}'s {component}, confirm the "
            "approved key ring, number of days between key rotations, and "
            "snapshot lifetime."
        ),
        (
            f"What are the audited recovery-snapshot controls for "
            f"{organization}'s {component}, including the escrow ring, "
            "renewal schedule, and retention period?"
        ),
    )
    return templates[index % len(templates)]


def _v2_memory_negative_query(track: str, index: int) -> str:
    organizations = _v2_organizations(track)
    organization = organizations[index % len(organizations)]
    return (
        f"Which color palette, typeface, and booth layout should {organization} use for its annual partner conference?"
    )


def _v2_memory_noise(track: str, index: int) -> str:
    organizations = _v2_organizations(track)
    organization = organizations[index % len(organizations)]
    return (
        f"{organization} schedules accessibility office hours for product "
        f"writers on the {index % 4 + 1}th Wednesday of each quarter."
    )


def _v2_local_skill(track: str, index: int) -> str:
    components = _v2_components(track)
    incidents = _v2_incidents(track)
    responses = _v2_responses(track)
    contexts = _v2_contexts(track)
    component = components[index % len(components)]
    incident = incidents[index % len(incidents)]
    response = responses[index % len(responses)]
    context = contexts[(index // len(incidents)) % len(contexts)]
    return (
        f"When {component} has {incident} in {context}, {response}. "
        "Verify recovery against the incident ledger before restoring load."
    )


def _v2_semantic_skill(track: str, index: int) -> str:
    components = _v2_components(track)
    incidents = _v2_incidents(track)
    responses = _v2_responses(track)
    contexts = _v2_contexts(track)
    component = components[index % len(components)]
    incident = incidents[index % len(incidents)]
    response = responses[index % len(responses)]
    context = contexts[(index // len(incidents)) % len(contexts)]
    return (
        f"This runbook recovers {component} from {incident} inside "
        f"{context}. Begin by checking the incident ledger, then "
        f"{response}, and finish with a bounded-load verification."
    )


def _v2_skill_query(track: str, index: int) -> str:
    components = _v2_components(track)
    incidents = _v2_incidents(track)
    contexts = _v2_contexts(track)
    component = components[index % len(components)]
    incident = incidents[index % len(incidents)]
    context = contexts[(index // len(incidents)) % len(contexts)]
    templates = (
        f"How do I restore {component} after {incident} in {context}?",
        (f"Find the recovery procedure for {component}: we observed {incident} in {context}."),
        (f"Which runbook should handle {incident} affecting {component} inside {context}?"),
        (f"Give me the verified remediation steps when {component} experiences {incident} in {context}."),
    )
    return templates[index % len(templates)]


def _v2_skill_negative_query(track: str, index: int) -> str:
    organizations = _v2_organizations(track)
    organization = organizations[index % len(organizations)]
    return (
        f"Draft a partner-launch newsletter for {organization}, including "
        "photo captions, a publication calendar, and brand-review owners."
    )


def _v2_skill_noise(track: str, index: int, source: str) -> str:
    organizations = _v2_organizations(track)
    organization = organizations[index % len(organizations)]
    topics = (
        "quarterly dashboard ownership and review dates",
        "new-hire equipment requests and approval routing",
        "customer advisory board invitations and attendance tracking",
        "office accessibility inspections and follow-up notes",
        "vendor invoice coding and cost-center reconciliation",
        "conference booth shipments and signage inventory",
        "training calendar publication and facilitator assignments",
        "partner newsletter approvals and caption reviews",
        "facilities access badges and visitor desk coverage",
        "documentation style reviews and terminology updates",
    )
    return (
        f"{organization} procedure {source}-{index:03d}: document "
        f"{topics[index % len(topics)]}, assign an owner, and archive the "
        "completed checklist."
    )


__all__ = ["SemanticFixture", "semantic_fixture"]
