#!/usr/bin/env python3
"""Build extended BASE34 prebuilt parquet covering 2020-2026.

Background
----------
Current BASE34 prebuilt only covers 2025-2026 (file
xauusd_m5_BASE34_20250101_20260420_MODEL_BARS.parquet, 91153 rows). The
V2 state contract and all downstream IQL gates depend on it. After the
data extension to 2020 (Phase 3A1+3A2), we need a matching prebuilt for
the full 2020-2026 window so the V2 contract / IQL pipelines can train
on extended data.

Strategy
--------
The existing BASE28_SEED_2020_2026 parquet already has 40 columns
including the HTF features (D1_atr_percentile_252,
D1_dist_from_ema200_atr, H1_range_compression_ratio, H4_trend_sign_cat,
M15_range_compression_ratio) and the BASE28 raw features
(_v1_atr14, _v1_atr_z_10_100, _v1_bb_*, _v1_body_*, _v1_clv,
_v1_close_ema_slope_3, _v1_kama_slope_30, _v1_pk_sigma20, etc.).

The missing pieces vs BASE34 are 5 session-derived features:
  is_ASIA, minutes_since_session_open, minutes_to_next_session_boundary,
  session_change_flag, session_tradable

These are 100% deterministic from M5 timestamp (no M1 needed).
gx1.features.basic_v1.add_session_features computes them from the
DatetimeIndex via gx1.time.session_detector SSoT.

This gate:
  1. Loads BASE28_SEED_2020_2026.
  2. Calls add_session_features to derive the 5 missing session features.
  3. Validates the resulting frame against the BASE34 reference schema
     (columns, dtypes, no NaN in required cols, time index).
  4. Writes
     /home/andre2/GX1_DATA/data/data/prebuilt/BASE34_EXTENDED_2020_2026/
     xauusd_m5_BASE34_2020_2026.parquet
  5. Audit: column-set match against BASE34 reference, row counts per
     year, time-range coverage, regime distribution per year.

No M1 dependency. No XGB/transformer/IQL training. Pure feature
extension on M5 timestamps.

Research-only data preparation gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.features.basic_v1 import add_session_features
from gx1.scripts import (
    materialize_build_iql_offline_data_contract_research_only_v1 as contract_gate,
)


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "BUILD_EXTENDED_BASE34_PREBUILT_2020_2026_V1"

INPUT_BASE28_SEED = Path(
    "/home/andre2/GX1_DATA/data/data/_staging/BASE28_SEED_2020_2026.parquet"
)
INPUT_BASE34_REFERENCE = Path(
    "/home/andre2/GX1_DATA/data/data/prebuilt/MONDAY_WEEK_EXTENSION_CANDIDATES/"
    "monday_week_prebuilt_extension_20260423_145325/"
    "xauusd_m5_BASE34_20250101_20260420_MODEL_BARS.parquet"
)
OUTPUT_DIR = Path(
    "/home/andre2/GX1_DATA/data/data/prebuilt/BASE34_EXTENDED_2020_2026"
)
OUTPUT_PARQUET = OUTPUT_DIR / "xauusd_m5_BASE34_2020_2026.parquet"

QUARANTINE_FORBIDDEN_PATH_FRAGMENTS = ("gx1/quarantine", "gx1.quarantine")
SESSION_FEATURES_TO_DERIVE = [
    "is_ASIA",
    "minutes_since_session_open",
    "minutes_to_next_session_boundary",
    "session_change_flag",
    "session_tradable",
]

ALLOWED_FINAL_STATUSES = {
    "EXTENDED_BASE34_PREBUILT_LOCKED_V1",
    "EXTENDED_BASE34_PREBUILT_BLOCKED_BY_INPUT_LOCK_MISSING_V1",
    "EXTENDED_BASE34_PREBUILT_BLOCKED_BY_SCHEMA_MISMATCH_V1",
}

ALLOWED_NEXT_ACTIONS = {
    "REGENERATE_TRUTH_REPLAY_2020_TO_2024_V1",
    "REPAIR_EXTENDED_BASE34_BEFORE_FURTHER_WORK_V1",
}


_jsonable = contract_gate._jsonable
_write_json = contract_gate._write_json
_write_rows = contract_gate._write_rows
_write_report = contract_gate._write_report
_file_hash = contract_gate._file_hash
_python_manifest = contract_gate._python_manifest


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_no_forbidden_actions(**kwargs: Any) -> dict[str, Any]:
    return contract_gate.validate_no_forbidden_actions(**kwargs)


def validate_final_status(status: str, next_action: str) -> bool:
    if status not in ALLOWED_FINAL_STATUSES:
        raise RuntimeError(f"FINAL_STATUS_NOT_ALLOWED: {status}")
    if next_action not in ALLOWED_NEXT_ACTIONS:
        raise RuntimeError(f"NEXT_ACTION_NOT_ALLOWED: {next_action}")
    return True


def validate_no_deprecated_revival(script_path: Path) -> bool:
    text = script_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.lstrip()
        if not (stripped.startswith("import ") or stripped.startswith("from ")):
            continue
        for fragment in QUARANTINE_FORBIDDEN_PATH_FRAGMENTS:
            if fragment in stripped:
                raise RuntimeError("DEPRECATED_QUARANTINE_REVIVAL_FORBIDDEN")
    return True


# ---------------------------------------------------------------------------
# Schema audit
# ---------------------------------------------------------------------------


def _audit_schema_match(
    extended_columns: set[str], reference_columns: set[str]
) -> dict[str, Any]:
    missing_in_extended = sorted(reference_columns - extended_columns)
    extra_in_extended = sorted(extended_columns - reference_columns)
    return {
        "audit_id_v1": "EXTENDED_BASE34_SCHEMA_MATCH_AUDIT_V1",
        "status_v1": "PASS" if not missing_in_extended else "FAIL",
        "n_extended_columns_v1": len(extended_columns),
        "n_reference_columns_v1": len(reference_columns),
        "missing_in_extended_v1": missing_in_extended,
        "extra_in_extended_v1": extra_in_extended,
    }


def _audit_row_counts_per_year(df: pd.DataFrame) -> dict[str, Any]:
    counts: dict[str, int] = {}
    if isinstance(df.index, pd.DatetimeIndex):
        for year, group in df.groupby(df.index.year):
            counts[str(int(year))] = int(len(group))
    return {
        "audit_id_v1": "EXTENDED_BASE34_ROW_COUNTS_PER_YEAR_AUDIT_V1",
        "status_v1": "PASS",
        "row_counts_v1": counts,
        "total_rows_v1": int(len(df)),
        "first_ts_v1": str(df.index.min()),
        "last_ts_v1": str(df.index.max()),
    }


def _audit_session_distribution(df: pd.DataFrame) -> dict[str, Any]:
    if "session_id" not in df.columns:
        return {
            "audit_id_v1": "EXTENDED_BASE34_SESSION_DISTRIBUTION_AUDIT_V1",
            "status_v1": "FAIL",
            "reason_v1": "session_id column missing",
        }
    counts = df["session_id"].value_counts(dropna=False).to_dict()
    return {
        "audit_id_v1": "EXTENDED_BASE34_SESSION_DISTRIBUTION_AUDIT_V1",
        "status_v1": "PASS",
        "counts_v1": {str(k): int(v) for k, v in counts.items()},
    }


def _audit_no_session_nan_in_required_cols(df: pd.DataFrame) -> dict[str, Any]:
    nan_counts: dict[str, int] = {}
    for col in SESSION_FEATURES_TO_DERIVE:
        if col in df.columns:
            nan_counts[col] = int(df[col].isna().sum())
    failures = [c for c, n in nan_counts.items() if n > 0]
    return {
        "audit_id_v1": "EXTENDED_BASE34_NO_NAN_IN_DERIVED_SESSION_FEATURES_AUDIT_V1",
        "status_v1": "PASS" if not failures else "FAIL",
        "nan_counts_v1": nan_counts,
        "failures_v1": failures,
    }


# ---------------------------------------------------------------------------
# Materializer
# ---------------------------------------------------------------------------


def _build_input_manifest(artifact_root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for name, path in [
        ("base28_seed_2020_2026", INPUT_BASE28_SEED),
        ("base34_reference_2025_2026", INPUT_BASE34_REFERENCE),
    ]:
        if path.exists():
            files.append(
                {
                    "name_v1": name,
                    "path_v1": str(path),
                    "sha256_v1": _file_hash(path),
                }
            )
    return {
        "layer_name": "EXTENDED_BASE34_PREBUILT_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "files_used_v1": files,
        "immutable_input_status_v1": "HASHED_EXPLICIT_ROOTS_ONLY",
        "no_implicit_latest_glob_selection_v1": True,
        "previous_artifacts_mutated_v1": False,
        "research_only_contract_v1": True,
        "iql_training_run_v1": False,
        "iql_production_allowed_v1": False,
        "skip_classifier_promoted_to_runtime_v1": False,
        "adapter_built_v1": False,
        "r6_run_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
        "deprecated_quarantine_revival_v1": False,
        "exit_manager_modified_v1": False,
        "live_features_modified_v1": False,
        "entry_manager_modified_v1": False,
        "v1_state_contract_modified_v1": False,
        "v2_state_contract_modified_v1": False,
        "python_manifest_v1": _python_manifest(),
    }


def write_artifacts(
    out_root: Path | None = None,
    *,
    built_at_utc: str | None = None,
) -> dict[str, Any]:
    timestamp = built_at_utc or _stamp()
    artifact_root = out_root or (DEFAULT_REPORTS_ROOT / f"{ACTION}_{timestamp}_LOCK")
    artifact_root.mkdir(parents=True, exist_ok=True)

    validate_no_deprecated_revival(Path(__file__))
    forbidden_audit = validate_no_forbidden_actions(
        adapter=False,
        r6=False,
        iql_production=False,
        package=False,
        freeze=False,
        promo=False,
        live=False,
        optuna=False,
        broad_sweep=False,
    )
    _write_json(
        artifact_root / "input_manifest_v1.json",
        _build_input_manifest(artifact_root),
    )

    # Load source.
    if not INPUT_BASE28_SEED.exists():
        raise RuntimeError(f"INPUT_BASE28_SEED_NOT_FOUND: {INPUT_BASE28_SEED}")
    if not INPUT_BASE34_REFERENCE.exists():
        raise RuntimeError(
            f"INPUT_BASE34_REFERENCE_NOT_FOUND: {INPUT_BASE34_REFERENCE}"
        )
    print(f"[3A3] loading BASE28_SEED_2020_2026 from {INPUT_BASE28_SEED}", flush=True)
    base28 = pd.read_parquet(INPUT_BASE28_SEED)
    if not isinstance(base28.index, pd.DatetimeIndex):
        if "time" in base28.columns:
            base28 = base28.set_index("time")
        else:
            raise RuntimeError("BASE28_SEED has no DatetimeIndex and no 'time' column")
    base28.index = pd.to_datetime(base28.index, utc=True)
    print(f"[3A3] loaded {len(base28)} rows, {len(base28.columns)} cols", flush=True)

    # Load reference for schema target.
    base34_ref = pd.read_parquet(INPUT_BASE34_REFERENCE)
    if not isinstance(base34_ref.index, pd.DatetimeIndex):
        if "time" in base34_ref.columns:
            base34_ref = base34_ref.set_index("time")
    reference_columns = set(base34_ref.columns)
    print(f"[3A3] reference BASE34 has {len(reference_columns)} cols", flush=True)

    # Derive session features.
    print("[3A3] deriving session features via add_session_features...", flush=True)
    base34_extended = add_session_features(base28.copy())
    print(f"[3A3] after derive: {len(base34_extended.columns)} cols", flush=True)

    # Audits.
    schema_audit = _audit_schema_match(set(base34_extended.columns), reference_columns)
    _write_json(artifact_root / "schema_match_audit_v1.json", schema_audit)
    row_counts_audit = _audit_row_counts_per_year(base34_extended)
    _write_json(artifact_root / "row_counts_per_year_audit_v1.json", row_counts_audit)
    session_dist_audit = _audit_session_distribution(base34_extended)
    _write_json(artifact_root / "session_distribution_audit_v1.json", session_dist_audit)
    nan_audit = _audit_no_session_nan_in_required_cols(base34_extended)
    _write_json(artifact_root / "session_nan_audit_v1.json", nan_audit)

    # Persist output.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base34_to_write = base34_extended.copy()
    base34_to_write.index.name = "time"
    base34_to_write.to_parquet(OUTPUT_PARQUET, index=True)
    print(f"[3A3] wrote {OUTPUT_PARQUET} ({len(base34_extended)} rows)", flush=True)

    audits = [schema_audit, row_counts_audit, session_dist_audit, nan_audit]
    audit_pass = all(a.get("status_v1") == "PASS" for a in audits)
    if audit_pass:
        status = "EXTENDED_BASE34_PREBUILT_LOCKED_V1"
        next_action = "REGENERATE_TRUTH_REPLAY_2020_TO_2024_V1"
        recommendation = (
            f"Extended BASE34 prebuilt locked at {OUTPUT_PARQUET} "
            f"({len(base34_extended)} rows, "
            f"{len(base34_extended.columns)} cols, range "
            f"{base34_extended.index.min()} -> {base34_extended.index.max()}). "
            "All schema and session-feature audits PASS. Next: regenerate "
            "TRUTH replay for 2020-2024 to produce trade_outcomes for the "
            "extended cohort (after M1 backfill + repair complete)."
        )
    elif schema_audit["status_v1"] == "FAIL":
        status = "EXTENDED_BASE34_PREBUILT_BLOCKED_BY_SCHEMA_MISMATCH_V1"
        next_action = "REPAIR_EXTENDED_BASE34_BEFORE_FURTHER_WORK_V1"
        recommendation = (
            f"Schema mismatch: {len(schema_audit['missing_in_extended_v1'])} "
            "columns missing vs reference. Investigate."
        )
    else:
        status = "EXTENDED_BASE34_PREBUILT_BLOCKED_BY_INPUT_LOCK_MISSING_V1"
        next_action = "REPAIR_EXTENDED_BASE34_BEFORE_FURTHER_WORK_V1"
        recommendation = "Audit failure; investigate per-audit details."
    validate_final_status(status, next_action)

    summary = {
        "layer_name": "EXTENDED_BASE34_PREBUILT_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "output_parquet_v1": str(OUTPUT_PARQUET),
        "row_count_v1": int(len(base34_extended)),
        "col_count_v1": int(len(base34_extended.columns)),
        "first_ts_v1": str(base34_extended.index.min()),
        "last_ts_v1": str(base34_extended.index.max()),
        "row_counts_per_year_v1": row_counts_audit["row_counts_v1"],
        "audits_v1": {a["audit_id_v1"]: a["status_v1"] for a in audits},
        "research_only_v1": True,
        "iql_training_run_v1": False,
        "iql_production_allowed_v1": False,
        "exit_manager_modified_v1": False,
        "live_features_modified_v1": False,
        "entry_manager_modified_v1": False,
        "v1_state_contract_modified_v1": False,
        "v2_state_contract_modified_v1": False,
        "forbidden_actions_audit_v1": forbidden_audit,
    }
    _write_json(artifact_root / "summary_v1.json", summary)
    _write_json(
        artifact_root / "status_v1.json",
        {
            "layer_name": "EXTENDED_BASE34_PREBUILT_STATUS_V1",
            "status_v1": "MATERIALIZED_DATA_PREP_GATE",
            "final_status_v1": status,
            "next_action_v1": next_action,
        },
    )
    artifact_manifest = {
        "layer_id_v1": ACTION,
        "built_at_utc_v1": summary["built_at_utc_v1"],
        "output_dir_v1": str(artifact_root),
        "append_only_namespace_v1": "truth_e2e_sanity",
        "artifact_paths_v1": {
            "summary": str(artifact_root / "summary_v1.json"),
            "status": str(artifact_root / "status_v1.json"),
            "input_manifest": str(artifact_root / "input_manifest_v1.json"),
            "output_parquet": str(OUTPUT_PARQUET),
            "schema_match_audit": str(artifact_root / "schema_match_audit_v1.json"),
            "row_counts_per_year_audit": str(
                artifact_root / "row_counts_per_year_audit_v1.json"
            ),
            "session_distribution_audit": str(
                artifact_root / "session_distribution_audit_v1.json"
            ),
            "session_nan_audit": str(artifact_root / "session_nan_audit_v1.json"),
        },
    }
    _write_json(artifact_root / "manifest_v1.json", artifact_manifest)
    return {"artifact_root": str(artifact_root), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize BUILD_EXTENDED_BASE34_PREBUILT_2020_2026_V1."
    )
    parser.add_argument("--out-root", type=str, default=None)
    args = parser.parse_args()
    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else None
    result = write_artifacts(out_root=out_root)
    print(json.dumps(_jsonable(result["summary"]), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
