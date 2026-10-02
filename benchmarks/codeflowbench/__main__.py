from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .campaign import (
    DEFAULT_OUTPUT_ROOT,
    CampaignError,
    CampaignMode,
    run_campaign,
)
from .canonical import to_primitive

_CLI_MODES = (
    CampaignMode.SMOKE,
    CampaignMode.SHIP,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the checkout-only CodeFlowBench Ship-1 campaign.",
    )
    parser.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in _CLI_MODES),
        default=CampaignMode.SHIP.value,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    args = parser.parse_args()
    try:
        outcome = asyncio.run(
            run_campaign(
                CampaignMode(args.mode),
                output_root=args.output_root,
            )
        )
    except CampaignError as exc:
        parser.exit(2, f"CodeFlowBench aborted: {exc}\n")
    print(
        json.dumps(
            to_primitive(outcome),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
