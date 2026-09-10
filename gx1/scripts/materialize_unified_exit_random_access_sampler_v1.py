#!/usr/bin/env python3
"""Materialize an explicit bounded random-access Exit sampler contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from gx1.contracts.unified_exit_random_access_sampler_v1 import (
    build_random_access_sampler_contract,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--source-lineage-sha256", required=True)
    parser.add_argument(
        "--transition-budget-per-epoch", type=_positive_int, required=True
    )
    parser.add_argument("--transitions-per-entry", type=_positive_int, required=True)
    parser.add_argument("--entry-pair-population", type=_positive_int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_OUTPUT_EXISTS")
    contract = build_random_access_sampler_contract(
        split=args.split,
        source_lineage_sha256=args.source_lineage_sha256,
        transition_budget_per_epoch=args.transition_budget_per_epoch,
        transitions_per_entry=args.transitions_per_entry,
        entry_pair_population=args.entry_pair_population,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_TEMP_EXISTS")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(contract, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
