#!/usr/bin/env python3
"""Print a read-only exact dependency readiness report for a GX1 rebuild."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from gx1.contracts.rebuild_dependency_readiness_v1 import (
    build_rebuild_dependency_readiness,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_rebuild_dependency_readiness(repo=Path(args.repo))
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
