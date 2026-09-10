"""Build a TRAIN/VAL-only no-cap economic lifecycle authority."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from gx1.contracts.unified_exit_fitted_q_v1 import (
    require_unified_exit_unbounded_training_readiness,
)
from gx1.contracts.unified_exit_lifecycle_v2 import terminal_state_counts_sha256
from gx1.contracts.unified_exit_no_cap_economic_authority_v1 import (
    NO_CAP_AUTHORITY_SCHEMA_VERSION,
    NO_CAP_TERMINAL_VERIFIER_SCHEMA_VERSION,
    canonical_sha256,
    file_sha256,
    require_economics_fact_manifest,
    seal_no_cap_authority,
)
from gx1.scripts.materialize_unified_exit_lifecycle_v2 import (
    ECONOMIC_COUNTS_SCHEMA_VERSION,
)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file():
        raise RuntimeError(f"NO_CAP_AUTHORITY_{label}_PATH_INVALID")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"NO_CAP_AUTHORITY_{label}_INVALID") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"NO_CAP_AUTHORITY_{label}_INVALID")
    return value


def _bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(_bytes(value))
    temporary.replace(path)


def build_no_cap_authority(
    *,
    output_dir: Path,
    dataset_run_id: str,
    split: str,
    entry_rows: int,
    coverage_start_utc: str,
    coverage_end_utc: str,
    economics_readiness_path: Path,
    economics_fact_manifest_path: Path,
    publish: bool,
) -> dict[str, Any]:
    """Verify source economics and emit all-null economic terminal counts."""

    if (
        split not in {"train", "val"}
        or not dataset_run_id
        or isinstance(entry_rows, bool)
        or not isinstance(entry_rows, int)
        or entry_rows < 1
        or type(publish) is not bool
    ):
        raise RuntimeError("NO_CAP_AUTHORITY_INVOCATION_INVALID")
    start = pd.Timestamp(coverage_start_utc)
    end = pd.Timestamp(coverage_end_utc)
    if (
        pd.isna(start)
        or pd.isna(end)
        or start.tz is None
        or end.tz is None
        or start.utcoffset() != pd.Timedelta(0)
        or end.utcoffset() != pd.Timedelta(0)
        or end <= start
    ):
        raise RuntimeError("NO_CAP_AUTHORITY_COVERAGE_INVALID")
    readiness_path = economics_readiness_path.expanduser().resolve()
    facts_path = economics_fact_manifest_path.expanduser().resolve()
    readiness = require_unified_exit_unbounded_training_readiness(
        _read_json(readiness_path, "ECONOMICS_READINESS"),
        context="NO_CAP_AUTHORITY",
    )
    require_economics_fact_manifest(
        _read_json(facts_path, "ECONOMICS_FACT_MANIFEST"),
        expected_coverage_start_utc=start,
        expected_coverage_end_utc=end,
    )
    rho = float(readiness["validated_annual_continuous_hurdle_rate"])
    if rho <= 0.0:
        raise RuntimeError("NO_CAP_AUTHORITY_POSITIVE_RHO_REQUIRED")

    mapping = {
        (entry_row, side_index): None
        for entry_row in range(entry_rows)
        for side_index in (0, 1)
    }
    terminal_hash = terminal_state_counts_sha256(mapping)
    counts = {
        "schema_version": ECONOMIC_COUNTS_SCHEMA_VERSION,
        "decision": "PASS",
        "dataset_run_id": dataset_run_id,
        "split": split,
        "entry_rows": entry_rows,
        "test_data_used": False,
        "terminal_state_counts": [
            {
                "entry_row_index": entry_row,
                "side_index": side_index,
                "terminal_state_count": None,
            }
            for entry_row in range(entry_rows)
            for side_index in (0, 1)
        ],
    }
    output = output_dir.expanduser().resolve()
    counts_path = output / f"{split}.economic_counts.no_cap.v1.json"
    authority_path = output / f"{split}.economic_authority.no_cap.v1.json"
    counts_sha256 = hashlib.sha256(_bytes(counts)).hexdigest()
    terminal_definition = {
        "schema_version": "gx1_rho_contractive_no_cap_terminal_definition_v1",
        "annual_continuous_hurdle_rate": rho,
        "discount_factor": "exp(-rho*elapsed_wall_clock_seconds/seconds_per_year)",
        "observed_economic_terminal_events": 0,
        "maximum_trade_duration_bars": None,
        "split_end_is_right_censor": True,
        "gap_is_right_censor": True,
        "capacity_forces_exit": False,
    }
    authority = seal_no_cap_authority(
        {
            "schema_version": NO_CAP_AUTHORITY_SCHEMA_VERSION,
            "decision": "PASS",
            "mode": "rho_contractive_no_duration_cap",
            "dataset_run_id": dataset_run_id,
            "split": split,
            "coverage_start_utc": start.isoformat(),
            "coverage_end_utc": end.isoformat(),
            "authority_artifact_path": str(counts_path),
            "authority_artifact_sha256": counts_sha256,
            "terminal_state_counts_sha256": terminal_hash,
            "economic_terminal_definition_sha256": canonical_sha256(
                terminal_definition
            ),
            "terminal_event_verifier_schema_version": (
                NO_CAP_TERMINAL_VERIFIER_SCHEMA_VERSION
            ),
            "no_observed_economic_terminals": True,
            "economics_readiness_path": str(readiness_path),
            "economics_readiness_sha256": file_sha256(readiness_path),
            "economics_objective_contract_sha256": readiness[
                "economics_objective_contract"
            ]["contract_sha256"],
            "annual_continuous_hurdle_rate": rho,
            "economics_fact_manifest_path": str(facts_path),
            "economics_fact_manifest_sha256": file_sha256(facts_path),
            "split_end_is_right_censor": True,
            "gap_is_right_censor": True,
            "capacity_forces_exit": False,
            "terminal_events_recomputed_from_train_val_only": True,
            "test_data_used": False,
        }
    )
    if publish:
        _write_atomic(counts_path, counts)
        _write_atomic(authority_path, authority)
        if file_sha256(counts_path) != counts_sha256:
            raise RuntimeError("NO_CAP_AUTHORITY_COUNTS_WRITE_INVALID")
    return {
        "decision": "PASS" if publish else "PASS_NOT_PUBLISHED",
        "counts_path": str(counts_path),
        "counts_sha256": counts_sha256,
        "authority_path": str(authority_path),
        "authority": authority,
        "terminal_definition": terminal_definition,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-run-id", required=True)
    parser.add_argument("--split", required=True, choices=("train", "val"))
    parser.add_argument("--entry-rows", required=True, type=int)
    parser.add_argument("--coverage-start-utc", required=True)
    parser.add_argument("--coverage-end-utc", required=True)
    parser.add_argument("--economics-readiness", required=True, type=Path)
    parser.add_argument("--economics-fact-manifest", required=True, type=Path)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    result = build_no_cap_authority(
        output_dir=args.output_dir,
        dataset_run_id=args.dataset_run_id,
        split=args.split,
        entry_rows=args.entry_rows,
        coverage_start_utc=args.coverage_start_utc,
        coverage_end_utc=args.coverage_end_utc,
        economics_readiness_path=args.economics_readiness,
        economics_fact_manifest_path=args.economics_fact_manifest,
        publish=args.publish,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
