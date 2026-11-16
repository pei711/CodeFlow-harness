#!/usr/bin/env python3
"""SWE-bench Verified Dataset Fetching Utility.

Provides programmatic data acquisition for SWE-bench Verified regression benchmarking.
Supports:
1. `datasets` library (`load_dataset('princeton-nlp/SWE-bench_Verified')`) if installed.
2. Direct Hugging Face Datasets Server API with chunked retrieval.
3. Verification of fetched instances against the frozen 50-task manifest.

Large dataset files (>5MB) are excluded from repository version control.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("swebench.fetch")

DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
DEFAULT_MANIFEST_DIR = (
    Path(__file__).resolve().parents[3] / "tasks" / "swebench" / "manifest"
)


def _load_manifest(manifest_dir: Path) -> dict:
    meta_path = manifest_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_dir}")
    manifest = json.loads(meta_path.read_text(encoding="utf-8"))
    tasks: list[dict] = []
    for chunk_path in sorted(manifest_dir.glob("tasks-*.json")):
        tasks.extend(json.loads(chunk_path.read_text(encoding="utf-8")).get("tasks", []))
    manifest["tasks"] = tasks
    return manifest


def fetch_via_datasets_lib(split: str = "test") -> list[dict]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        logger.info("datasets library not installed, falling back to Hugging Face Server API")
        return []

    logger.info("Fetching %s via datasets library...", DATASET_NAME)
    ds = load_dataset(DATASET_NAME, split=split)
    return [dict(row) for row in ds]


def fetch_via_hf_api(limit_total: int = 500, chunk_size: int = 100) -> list[dict]:
    logger.info("Fetching %s via Hugging Face Datasets API...", DATASET_NAME)
    all_rows: list[dict] = []
    offset = 0

    while offset < limit_total:
        url = (
            f"https://datasets-server.huggingface.co/rows"
            f"?dataset=princeton-nlp%2FSWE-bench_Verified&config=default&split=test"
            f"&offset={offset}&limit={chunk_size}"
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "CodeFlowBench-SWE-bench-Runner/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                rows = [r["row"] for r in data.get("rows", [])]
                if not rows:
                    break
                all_rows.extend(rows)
                logger.info("Fetched offset %d: %d rows (total: %d)", offset, len(rows), len(all_rows))
                offset += len(rows)
        except Exception as exc:
            logger.error("Failed to fetch chunk at offset %d: %s", offset, exc)
            break

    return all_rows


def verify_manifest(manifest_dir: Path, dataset_rows: list[dict]) -> dict:
    manifest = _load_manifest(manifest_dir)

    manifest_tasks = manifest.get("tasks", [])
    manifest_ids = {t["instance_id"] for t in manifest_tasks}
    dataset_ids = {r["instance_id"] for r in dataset_rows}

    matched = manifest_ids.intersection(dataset_ids)
    missing = manifest_ids - dataset_ids

    logger.info("Manifest verification: %d / %d tasks matched", len(matched), len(manifest_ids))
    if missing:
        logger.warning("Missing from downloaded dataset: %s", sorted(missing))

    return {
        "manifest_count": len(manifest_ids),
        "dataset_count": len(dataset_ids),
        "matched_count": len(matched),
        "missing_count": len(missing),
        "missing_ids": sorted(missing),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch SWE-bench Verified dataset rows.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_DIR,
        help="Directory holding the frozen 50-task manifest (meta.json + tasks-*.json chunks).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save downloaded dataset (must be gitignored if >5MB).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify that manifest file exists and has 50 valid tasks.",
    )
    args = parser.parse_args()

    if args.verify_only:
        if not args.manifest.is_dir():
            logger.error("Manifest missing: %s", args.manifest)
            return 1
        data = _load_manifest(args.manifest)
        count = len(data.get("tasks", []))
        logger.info("Manifest %s validated: %d tasks present", args.manifest.name, count)
        return 0 if count == 50 else 2

    # Try datasets lib first, then fall back to API
    rows = fetch_via_datasets_lib()
    if not rows:
        rows = fetch_via_hf_api()

    if not rows:
        logger.error("Failed to acquire SWE-bench Verified dataset")
        return 1

    stats = verify_manifest(args.manifest, rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        logger.info("Saved %d rows to %s", len(rows), args.output)

    return 0 if stats["matched_count"] == stats["manifest_count"] else 2


if __name__ == "__main__":
    sys.exit(main())
