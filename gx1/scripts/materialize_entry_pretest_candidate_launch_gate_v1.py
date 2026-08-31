#!/usr/bin/env python3
"""Materialize an immutable readiness gate for one pre-TEST candidate recipe."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

from gx1.contracts.entry_model_native_pretest_technical_recipe_v1 import (
    require_pretest_technical_recipe_metadata,
)
from gx1.contracts.entry_pretest_candidate_launch_gate_v1 import (
    EVENT_PREFIX,
    READY_DECISION,
    SCHEMA_VERSION,
    AUTHORITY,
    _candidate_readiness,
    _smoke_audit,
    artifact_binding,
)
from gx1.contracts.immutable_event_authority_v1 import (
    write_immutable_json_event,
)


def run(args: argparse.Namespace) -> tuple[Path, dict]:
    recipe_path = Path(args.recipe_json).expanduser().resolve(strict=True)
    readiness_path = Path(args.candidate_readiness_json).expanduser().resolve(strict=True)
    smoke_path = Path(args.smoke_bundle_audit_json).expanduser().resolve(strict=True)
    recipe = require_pretest_technical_recipe_metadata(
        json.loads(recipe_path.read_text(encoding="utf-8")),
        expected_profile="candidate",
    )
    recipe_binding = artifact_binding(recipe_path)
    if recipe_binding["sha256"] != str(args.recipe_sha256):
        raise RuntimeError("candidate recipe SHA-256 mismatch")
    readiness_binding = artifact_binding(readiness_path)
    if readiness_binding["sha256"] != str(args.candidate_readiness_sha256):
        raise RuntimeError("candidate readiness SHA-256 mismatch")
    smoke_binding = artifact_binding(smoke_path)
    if smoke_binding["sha256"] != str(args.smoke_bundle_audit_sha256):
        raise RuntimeError("smoke bundle audit SHA-256 mismatch")
    identity = {
        key: str(recipe[key])
        for key in ("run_id", "dataset_run_id", "dataset_dir", "out_bundle_dir")
    }
    _smoke_audit(smoke_path, recipe=identity)
    _candidate_readiness(
        readiness_path,
        recipe=identity,
        smoke_audit=smoke_binding,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": READY_DECISION,
        "failures": [],
        "authority": dict(AUTHORITY),
        "activation_authority": False,
        **identity,
        "recipe": recipe_binding,
        "candidate_readiness": readiness_binding,
        "smoke_bundle_audit": smoke_binding,
    }
    return write_immutable_json_event(Path(args.out_dir), EVENT_PREFIX, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe-json", required=True)
    parser.add_argument("--recipe-sha256", required=True)
    parser.add_argument("--candidate-readiness-json", required=True)
    parser.add_argument("--candidate-readiness-sha256", required=True)
    parser.add_argument("--smoke-bundle-audit-json", required=True)
    parser.add_argument("--smoke-bundle-audit-sha256", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    path, _ = run(build_parser().parse_args(argv))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
