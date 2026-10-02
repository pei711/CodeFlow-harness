"""Historical EverOS semantic harness boundary after Runtime removal."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.codeflowbench.packs.memory_skill.semantic_fixtures import semantic_fixture
from benchmarks.codeflowbench.packs.memory_skill.semantic_runtime import (
    CountingEmbeddingProvider,
    SemanticRuntimeConfig,
    run_semantic_runtime,
)


class _UnusedEmbedding:
    dim = 1

    async def embed(self, text: str) -> list[float]:
        del text
        return [1.0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


async def test_historical_semantic_runtime_requires_removed_dependency(
    tmp_path: Path,
) -> None:
    embedder = CountingEmbeddingProvider(
        _UnusedEmbedding(),
        provider_identity="historical/offline",
        provider_config_digest="a" * 64,
        real_provider=False,
    )

    with pytest.raises(ModuleNotFoundError, match="everos"):
        await run_semantic_runtime(
            fixture=semantic_fixture("calibration"),
            config=SemanticRuntimeConfig(
                top_k=5,
                user_min_score=0.0,
                agent_radius=0.0,
                local_weight=1.0,
                everos_weight=0.9,
            ),
            isolated_root=tmp_path / "historical-everos",
            embedder=embedder,
        )
