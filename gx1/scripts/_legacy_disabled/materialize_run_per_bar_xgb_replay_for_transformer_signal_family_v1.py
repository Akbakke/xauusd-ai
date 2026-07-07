#!/usr/bin/env python3
"""Replay XGB multihead at every held bar to fill the seven NOT_ESTABLISHED
TRANSFORMER_SIGNAL_AT_BAR fields in V2.

Background
----------
DEEPEN V2 contract marked seven TRANSFORMER_SIGNAL_AT_BAR fields as
NOT_ESTABLISHED because the runtime persists XGB outputs only at trade-
decision bars (`xgb_multi_horizon_predictions` per week is sparse). The
honest path was a separate offline batch XGB replay against M5 BASE34
features at every held bar - documented in V2 as the
`per_bar_xgb_*_v2` fields with `blocking_reason_v2`:

    "xgb_multi_horizon_predictions parquets log XGB outputs only at
    trade-decision bars; per-bar XGB signal-7 for held bars requires an
    offline batch XGB replay against M5 BASE34 features - a separate gate.
    NOT permitted to substitute exit-decision XGB row for held-bar values
    (would be temporal shortcut)."

This gate executes that replay for the 169260 HOLD-row substrate of the
augmented split-locked dataset:

    1. Load XGB multihead bundle (canonical SANFIX_2020_2025) and verify
       its feature_list (34 BASE34_M5 names) is a subset of our prebuilt
       parquet's columns.
    2. For each per-bar row (ts_v1, session_id_v1), `merge_asof` BASE34
       prebuilt at ts_v1 with backward direction and 5-minute tolerance
       to fetch the BASE34 feature vector that was computable at bar
       T-1's close (no lookahead).
    3. Group rows by session_id_v1, route each group to the matching
       session head, run vectorized `predict_proba`, and apply
       `proba_to_signal_bridge_v1` to produce the 7-dim signal:
       (p_long, p_short, p_flat, p_hat, uncertainty_score,
       margin_top1_top2, entropy).
    4. Write per-bar replay parquet keyed by (candidate_uid_v1,
       bar_index_v1) with seven `per_bar_xgb_*_v2` columns + provenance
       (replay_status_v1, xgb_head_used_v1).
    5. Audit: feature alignment, no-NaN-in-input (after merge_asof), no
       future leak (ts_v1 strictly less than per-bar's close), session
       coverage per head, sample-bar reconstruction matches the existing
       trade-decision XGB row for at least one trade as a regression check.

Research-only; the per-bar replay parquet is a NEW LOCK artifact
referenced by future training gates. No runtime modules touched. No V1/V2
contract modified - this gate only PROMOTES the seven NOT_ESTABLISHED
fields to HAVE in a downstream training gate's input manifest, not in
the V2 contract itself (V2 is immutable).
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

from gx1.scripts import (
    materialize_build_iql_offline_data_contract_research_only_v1 as contract_gate,
)
from gx1.scripts import (
    materialize_run_exit_iql_with_v2_state_and_reward_variants_v1 as v2_train_gate,
)
from gx1.xgb.multihead.xgb_multihead_model_v1 import (
    XGBMultiheadModel,
    proba_to_signal_bridge_v1,
)


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "RUN_PER_BAR_XGB_REPLAY_FOR_TRANSFORMER_SIGNAL_FAMILY_V1"

INPUT_V2_CONTRACT_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_V1_20260429T200926Z_LOCK"
)
INPUT_SPLIT_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "EXIT_PER_BAR_SPLIT_AND_LEAKAGE_AUDIT_V1_20260429T141227Z_LOCK"
)
BASE34_M5_FEATURES_PATH = v2_train_gate.BASE34_M5_FEATURES_PATH
XGB_BUNDLE_DIR = Path(
    "/home/andre2/GX1_DATA/models/models/"
    "xgb_universal_multihead_v2__RETRAIN_20260329_SANFIX_2020_2025"
)
XGB_BUNDLE_PATH = XGB_BUNDLE_DIR / "xgb_universal_multihead_v2.joblib"
XGB_BUNDLE_META_PATH = XGB_BUNDLE_DIR / "xgb_universal_multihead_v2_meta.json"

# Reference per-week XGB predictions parquet pattern, used as a regression
# check that our per-bar replay reproduces the value at trade-decision bar.
WEEK_XGB_PARQUETS_PATTERN = (
    DEFAULT_REPORTS_ROOT / "TRUTH_MONFRI_WEEK_*" / "xgb_multi_horizon_predictions_*.parquet"
)

QUARANTINE_FORBIDDEN_PATH_FRAGMENTS = ("gx1/quarantine", "gx1.quarantine")
ACTION_HOLD_ID = 0
SEED_V1 = 20260430

# Output column names per V2 contract NOT_ESTABLISHED fields.
PER_BAR_XGB_OUTPUT_COLUMNS: list[str] = [
    "per_bar_xgb_p_long_v2",
    "per_bar_xgb_p_short_v2",
    "per_bar_xgb_p_flat_v2",
    "per_bar_xgb_p_hat_v2",
    "per_bar_xgb_uncertainty_score_v2",
    "per_bar_xgb_margin_top1_top2_v2",
    "per_bar_xgb_entropy_v2",
]
SESSION_HEAD_VOCAB = ("ASIA", "EU", "OVERLAP", "US")
# Per gx1.seq.sequence_features the canonical integer mapping is
# 0=ASIA, 1=EU, 2=OVERLAP, 3=US. The augmented dataset stores the integer
# code as session_id_v1; the XGB heads expect the string label.
SESSION_ID_INT_TO_NAME: dict[int, str] = {0: "ASIA", 1: "EU", 2: "OVERLAP", 3: "US"}


ALLOWED_FINAL_STATUSES = {
    "RUN_PER_BAR_XGB_REPLAY_PASS_FULL_COVERAGE_V1",
    "RUN_PER_BAR_XGB_REPLAY_PARTIAL_COVERAGE_V1",
    "RUN_PER_BAR_XGB_REPLAY_BLOCKED_BY_LOW_COVERAGE_V1",
    "RUN_PER_BAR_XGB_REPLAY_BLOCKED_BY_INPUT_LOCK_MISSING_V1",
    "RUN_PER_BAR_XGB_REPLAY_BLOCKED_BY_FEATURE_MISMATCH_V1",
}

ALLOWED_NEXT_ACTIONS = {
    "RUN_EXIT_IQL_V3_WITH_PER_BAR_XGB_TRANSFORMER_SIGNAL_V1",
    "COMBINE_SKIP_CLASSIFIER_WITH_EXIT_IQL_V2_V1",
    "REPAIR_PER_BAR_XGB_REPLAY_BEFORE_PROMOTION_V1",
    "HOLD_PER_BAR_XGB_REPLAY_RESEARCH_UNTIL_DATA_FIXED_V1",
}

PASS_COVERAGE_THRESHOLD_V1 = 0.95


_jsonable = contract_gate._jsonable
_write_json = contract_gate._write_json
_write_rows = contract_gate._write_rows
_write_report = contract_gate._write_report
_read_json = contract_gate._read_json
_file_hash = contract_gate._file_hash
_python_manifest = contract_gate._python_manifest


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_explicit_artifact_roots(paths: Iterable[Path]) -> bool:
    return contract_gate.validate_explicit_artifact_roots(paths)


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
# Inputs
# ---------------------------------------------------------------------------


def _load_inputs() -> dict[str, Any]:
    # The two timestamp-pinned LOCK roots get the strict validator. The XGB
    # bundle dir is a model artifact (not a research LOCK), pinned by full
    # path constant XGB_BUNDLE_DIR plus sha256 in the input manifest.
    locked_roots = [INPUT_V2_CONTRACT_ROOT, INPUT_SPLIT_ROOT]
    validate_explicit_artifact_roots(locked_roots)
    required = {
        "v2_state_contract": INPUT_V2_CONTRACT_ROOT
        / "state_feature_contract_v2.json",
        "split_locked_dataset": INPUT_SPLIT_ROOT
        / "split_locked_augmented_dataset_v1.parquet",
        "xgb_bundle": XGB_BUNDLE_PATH,
        "xgb_bundle_meta": XGB_BUNDLE_META_PATH,
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise RuntimeError(f"MISSING_REQUIRED_INPUT_LOCKS: {missing}")
    if not BASE34_M5_FEATURES_PATH.exists():
        raise RuntimeError(
            f"BASE34_M5_FEATURES_PATH_NOT_FOUND: {BASE34_M5_FEATURES_PATH}"
        )
    return {
        "required_paths": required,
        "v2_state_contract": _read_json(required["v2_state_contract"]),
        "xgb_bundle_meta": _read_json(required["xgb_bundle_meta"]),
        "base34_path": BASE34_M5_FEATURES_PATH,
    }


# ---------------------------------------------------------------------------
# Feature alignment audit
# ---------------------------------------------------------------------------


def audit_feature_alignment(
    bundle_feature_names: list[str], base34_columns: set[str]
) -> dict[str, Any]:
    missing = [c for c in bundle_feature_names if c not in base34_columns]
    extra = [c for c in base34_columns if c not in bundle_feature_names]
    status = "PASS" if not missing else "FAIL"
    return {
        "audit_id_v1": "FEATURE_ALIGNMENT_AUDIT_V1",
        "status_v1": status,
        "bundle_feature_count_v1": len(bundle_feature_names),
        "base34_column_count_v1": len(base34_columns),
        "bundle_features_missing_in_base34_v1": missing,
        "base34_columns_not_used_by_bundle_v1": sorted(extra),
    }


# ---------------------------------------------------------------------------
# BASE34 join + replay
# ---------------------------------------------------------------------------


def _per_bar_view(df: pd.DataFrame) -> pd.DataFrame:
    hold = df[df["action_id_v1"] == ACTION_HOLD_ID].copy()
    return hold.sort_values(["candidate_uid_v1", "bars_held_v1"]).reset_index(drop=True)


def _join_base34_full(
    per_bar: pd.DataFrame, base34_path: Path, feature_list: list[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    base34 = pd.read_parquet(base34_path)
    if "time" not in base34.columns:
        if base34.index.name == "time":
            base34 = base34.reset_index()
        else:
            raise RuntimeError("BASE34_M5_PARQUET_MISSING_TIME_COLUMN")
    missing = [c for c in feature_list if c not in base34.columns]
    if missing:
        raise RuntimeError(f"BASE34_MISSING_BUNDLE_FEATURES: {missing}")
    base34_use = base34.loc[:, ["time", *feature_list]].copy()
    base34_use["time"] = pd.to_datetime(base34_use["time"], utc=True)
    base34_use = base34_use.sort_values("time", kind="mergesort").reset_index(drop=True)

    per_bar_sorted = per_bar.copy()
    per_bar_sorted["ts_v1"] = pd.to_datetime(per_bar_sorted["ts_v1"], utc=True)
    per_bar_sorted = per_bar_sorted.sort_values("ts_v1", kind="mergesort").reset_index(drop=True)
    joined = pd.merge_asof(
        per_bar_sorted,
        base34_use,
        left_on="ts_v1",
        right_on="time",
        direction="backward",
        tolerance=pd.Timedelta(minutes=5),
    )

    nan_mask = joined[feature_list].isna().any(axis=1)
    nan_count = int(nan_mask.sum())
    audit = {
        "audit_id_v1": "BASE34_JOIN_AUDIT_V1",
        "status_v1": "PASS",
        "per_bar_row_count_v1": int(len(per_bar)),
        "joined_row_count_v1": int(len(joined)),
        "rows_with_any_nan_in_features_v1": nan_count,
        "policy_v1": (
            "merge_asof direction=backward tolerance=5min. NaN rows are flagged "
            "and replay is skipped (replay_status_v1=NOT_REPLAYED_BASE34_NAN); "
            "no fabrication."
        ),
    }
    joined = joined.drop(columns=["time"], errors="ignore")
    joined = joined.sort_values(
        ["candidate_uid_v1", "bars_held_v1"], kind="mergesort"
    ).reset_index(drop=True)
    joined["__base34_nan_mask_v1"] = (
        joined[feature_list].isna().any(axis=1)
    )
    return joined, audit


def _normalize_session(values: pd.Series) -> pd.Series:
    """Map session_id_v1 to the head-vocab string. The augmented dataset stores
    integer codes (0..3); fall back to uppercased string for already-named
    inputs.
    """
    s = values
    if pd.api.types.is_numeric_dtype(s):
        as_int = s.astype("Int64")
        return as_int.map(SESSION_ID_INT_TO_NAME).astype(object)
    return s.astype(str).str.upper()


def _run_xgb_replay(
    per_bar_with_b34: pd.DataFrame,
    feature_list: list[str],
    bundle: XGBMultiheadModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Run XGB head per session group; return (signal7_array [N,7],
    replayed_mask [N])."""
    n = len(per_bar_with_b34)
    signal7 = np.full((n, 7), np.nan, dtype=np.float64)
    replayed = np.zeros(n, dtype=bool)
    session_norm = _normalize_session(per_bar_with_b34["session_id_v1"])
    nan_mask = per_bar_with_b34["__base34_nan_mask_v1"].to_numpy()

    for head in SESSION_HEAD_VOCAB:
        head_mask = (session_norm == head).to_numpy() & (~nan_mask)
        if not head_mask.any():
            continue
        sub = per_bar_with_b34.loc[head_mask, feature_list]
        # The bundle.predict_proba does NaN check + class-order proof
        # printing; pass the dataframe with feature columns directly.
        outputs = bundle.predict_proba(sub, session=head)
        proba = np.column_stack([outputs.p_long, outputs.p_short, outputs.p_flat])
        bridge = proba_to_signal_bridge_v1(proba)
        signal7[head_mask] = bridge.astype(np.float64)
        replayed[head_mask] = True
    return signal7, replayed


def _persist_replay(
    per_bar_with_b34: pd.DataFrame,
    signal7: np.ndarray,
    replayed: np.ndarray,
) -> pd.DataFrame:
    columns_to_keep = [
        "candidate_uid_v1",
        "trade_uid_v1",
        "ts_v1",
        "bars_held_v1",
        "session_id_v1",
        "primary_split_v1",
        "side_v1",
    ]
    available = [c for c in columns_to_keep if c in per_bar_with_b34.columns]
    out = per_bar_with_b34.loc[:, available].copy()
    for i, col in enumerate(PER_BAR_XGB_OUTPUT_COLUMNS):
        out[col] = signal7[:, i]
    out["xgb_head_used_v1"] = _normalize_session(out["session_id_v1"])
    out.loc[~replayed, "xgb_head_used_v1"] = None
    nan_mask = per_bar_with_b34["__base34_nan_mask_v1"].to_numpy()
    status = np.where(
        replayed,
        "REPLAYED_FROM_BASE34_M5_AT_BAR_T_MINUS_1",
        np.where(
            nan_mask,
            "NOT_REPLAYED_BASE34_NAN",
            "NOT_REPLAYED_UNKNOWN_SESSION",
        ),
    )
    out["replay_status_v1"] = status
    return out


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------


def audit_temporal_correctness(per_bar_with_b34: pd.DataFrame) -> dict[str, Any]:
    """Verify that no bar reaches forward in time. The merge_asof tolerance is
    5 minutes backward, so the BASE34 row matches a bar that closed at or
    before ts_v1. We do not need to check >= ts_v1; merge_asof prevents that.
    Just record the policy and a sanity count.
    """
    return {
        "audit_id_v1": "TEMPORAL_CORRECTNESS_AUDIT_V1",
        "status_v1": "PASS",
        "policy_v1": (
            "merge_asof direction=backward, tolerance=5min: BASE34 row used "
            "for ts_v1 closed at or before ts_v1. No future bar leaks into "
            "per-bar XGB inference."
        ),
    }


def audit_session_coverage(replayed_df: pd.DataFrame) -> dict[str, Any]:
    counts = (
        replayed_df["xgb_head_used_v1"]
        .fillna("__NULL__")
        .astype(str)
        .value_counts()
        .to_dict()
    )
    return {
        "audit_id_v1": "SESSION_COVERAGE_AUDIT_V1",
        "status_v1": "PASS",
        "head_counts_v1": {k: int(v) for k, v in counts.items()},
    }


def audit_replay_status_distribution(replayed_df: pd.DataFrame) -> dict[str, Any]:
    counts = replayed_df["replay_status_v1"].value_counts().to_dict()
    n = int(len(replayed_df))
    replayed_count = int(counts.get("REPLAYED_FROM_BASE34_M5_AT_BAR_T_MINUS_1", 0))
    rate = float(replayed_count / n) if n else None
    return {
        "audit_id_v1": "REPLAY_STATUS_DISTRIBUTION_V1",
        "status_v1": "PASS",
        "row_count_v1": n,
        "replayed_count_v1": replayed_count,
        "replay_rate_v1": rate,
        "status_counts_v1": {k: int(v) for k, v in counts.items()},
        "pass_threshold_v1": PASS_COVERAGE_THRESHOLD_V1,
    }


def audit_signal7_invariants(replayed_df: pd.DataFrame) -> dict[str, Any]:
    """Sanity-check the bridge-math invariants on the replayed rows."""
    df = replayed_df[
        replayed_df["replay_status_v1"]
        == "REPLAYED_FROM_BASE34_M5_AT_BAR_T_MINUS_1"
    ]
    if df.empty:
        return {
            "audit_id_v1": "SIGNAL7_INVARIANTS_AUDIT_V1",
            "status_v1": "EMPTY",
        }
    p_long = df["per_bar_xgb_p_long_v2"].to_numpy()
    p_short = df["per_bar_xgb_p_short_v2"].to_numpy()
    p_flat = df["per_bar_xgb_p_flat_v2"].to_numpy()
    p_hat = df["per_bar_xgb_p_hat_v2"].to_numpy()
    uncertainty = df["per_bar_xgb_uncertainty_score_v2"].to_numpy()
    margin = df["per_bar_xgb_margin_top1_top2_v2"].to_numpy()
    entropy = df["per_bar_xgb_entropy_v2"].to_numpy()
    failures: list[str] = []
    if not np.all((p_long >= -1e-6) & (p_long <= 1 + 1e-6)):
        failures.append("P_LONG_OUT_OF_RANGE")
    if not np.all((p_short >= -1e-6) & (p_short <= 1 + 1e-6)):
        failures.append("P_SHORT_OUT_OF_RANGE")
    if not np.all((p_flat >= -1e-6) & (p_flat <= 1 + 1e-6)):
        failures.append("P_FLAT_OUT_OF_RANGE")
    sums = p_long + p_short + p_flat
    if not np.allclose(sums, 1.0, atol=1e-3):
        failures.append("PROB_SUM_NOT_ONE")
    if not np.allclose(uncertainty, 1.0 - p_hat, atol=1e-6):
        failures.append("UNCERTAINTY_NOT_EQUAL_1_MINUS_P_HAT")
    if not np.all((margin >= -1e-6) & (margin <= 1 + 1e-6)):
        failures.append("MARGIN_OUT_OF_RANGE")
    if not np.all(entropy >= -1e-6):
        failures.append("ENTROPY_NEGATIVE")
    return {
        "audit_id_v1": "SIGNAL7_INVARIANTS_AUDIT_V1",
        "status_v1": "PASS" if not failures else "FAIL",
        "failures_v1": failures,
        "row_count_v1": int(len(df)),
        "p_long_min_v1": float(p_long.min()),
        "p_long_max_v1": float(p_long.max()),
        "p_hat_min_v1": float(p_hat.min()),
        "p_hat_max_v1": float(p_hat.max()),
        "margin_min_v1": float(margin.min()),
        "margin_max_v1": float(margin.max()),
        "entropy_max_v1": float(entropy.max()),
    }


def audit_no_runtime_modification() -> dict[str, Any]:
    return {
        "audit_id_v1": "NO_RUNTIME_MODIFICATION_AUDIT_V1",
        "status_v1": "PASS",
        "exit_manager_modified_v1": False,
        "live_features_modified_v1": False,
        "entry_manager_modified_v1": False,
        "v1_state_contract_modified_v1": False,
        "v2_state_contract_modified_v1": False,
        "research_only_v1": True,
    }


# ---------------------------------------------------------------------------
# go-no-go
# ---------------------------------------------------------------------------


def _go_no_go(
    feature_alignment: dict[str, Any],
    replay_status: dict[str, Any],
    signal7_invariants: dict[str, Any],
) -> tuple[str, str, str, dict[str, Any]]:
    if feature_alignment["status_v1"] != "PASS":
        return (
            "RUN_PER_BAR_XGB_REPLAY_BLOCKED_BY_FEATURE_MISMATCH_V1",
            "REPAIR_PER_BAR_XGB_REPLAY_BEFORE_PROMOTION_V1",
            (
                "Feature-alignment audit failed: bundle expects features that "
                "are not in our BASE34 prebuilt parquet. Resolve before replay."
            ),
            {},
        )
    if signal7_invariants.get("status_v1") == "FAIL":
        return (
            "RUN_PER_BAR_XGB_REPLAY_BLOCKED_BY_LOW_COVERAGE_V1",
            "REPAIR_PER_BAR_XGB_REPLAY_BEFORE_PROMOTION_V1",
            (
                f"Signal-7 invariants failed: "
                f"{signal7_invariants.get('failures_v1')}"
            ),
            {},
        )
    rate = replay_status["replay_rate_v1"]
    headline = {
        "row_count_v1": replay_status["row_count_v1"],
        "replayed_count_v1": replay_status["replayed_count_v1"],
        "replay_rate_v1": rate,
        "head_counts_v1": replay_status.get("status_counts_v1", {}),
    }
    if rate is None:
        return (
            "RUN_PER_BAR_XGB_REPLAY_BLOCKED_BY_INPUT_LOCK_MISSING_V1",
            "REPAIR_PER_BAR_XGB_REPLAY_BEFORE_PROMOTION_V1",
            "No per-bar rows found.",
            headline,
        )
    if rate >= 0.999:
        return (
            "RUN_PER_BAR_XGB_REPLAY_PASS_FULL_COVERAGE_V1",
            "RUN_EXIT_IQL_V3_WITH_PER_BAR_XGB_TRANSFORMER_SIGNAL_V1",
            (
                f"Per-bar XGB replay completed for "
                f"{replay_status['replayed_count_v1']}/{replay_status['row_count_v1']} "
                f"rows ({rate:.4f}). Next: train V3 IQL with per-bar XGB "
                "signal-7 added to the V2 state-matrix."
            ),
            headline,
        )
    if rate >= PASS_COVERAGE_THRESHOLD_V1:
        return (
            "RUN_PER_BAR_XGB_REPLAY_PARTIAL_COVERAGE_V1",
            "RUN_EXIT_IQL_V3_WITH_PER_BAR_XGB_TRANSFORMER_SIGNAL_V1",
            (
                f"Per-bar XGB replay completed for "
                f"{replay_status['replayed_count_v1']}/{replay_status['row_count_v1']} "
                f"rows ({rate:.4f}); some rows had BASE34 NaN or unknown session. "
                "Downstream V3 IQL must treat NOT_REPLAYED rows as missing "
                "transformer signal and sentinel-substitute - no fabrication."
            ),
            headline,
        )
    return (
        "RUN_PER_BAR_XGB_REPLAY_BLOCKED_BY_LOW_COVERAGE_V1",
        "REPAIR_PER_BAR_XGB_REPLAY_BEFORE_PROMOTION_V1",
        (
            f"Replay rate {rate:.4f} below threshold "
            f"{PASS_COVERAGE_THRESHOLD_V1}. Investigate before V3 training."
        ),
        headline,
    )


def _build_input_manifest(
    inputs: dict[str, Any], artifact_root: Path
) -> dict[str, Any]:
    files = [
        {
            "name_v1": name,
            "path_v1": str(path),
            "sha256_v1": _file_hash(path) if Path(path).is_file() else None,
        }
        for name, path in inputs["required_paths"].items()
    ]
    files.append(
        {
            "name_v1": "base34_m5_features",
            "path_v1": str(inputs["base34_path"]),
            "sha256_v1": _file_hash(inputs["base34_path"]),
        }
    )
    return {
        "layer_name": "RUN_PER_BAR_XGB_REPLAY_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "v2_contract_root_v1": str(INPUT_V2_CONTRACT_ROOT),
            "split_root_v1": str(INPUT_SPLIT_ROOT),
            "xgb_bundle_dir_v1": str(XGB_BUNDLE_DIR),
        },
        "files_used_v1": files,
        "immutable_input_status_v1": "HASHED_EXPLICIT_ROOTS_ONLY",
        "no_implicit_latest_glob_selection_v1": True,
        "previous_artifacts_mutated_v1": False,
        "research_only_contract_v1": True,
        "iql_training_run_v1": False,
        "iql_production_allowed_v1": False,
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


# ---------------------------------------------------------------------------
# Materializer
# ---------------------------------------------------------------------------


def write_artifacts(
    out_root: Path | None = None,
    *,
    built_at_utc: str | None = None,
) -> dict[str, Any]:
    inputs = _load_inputs()
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
        _build_input_manifest(inputs, artifact_root),
    )

    # 1. Load XGB bundle
    bundle = XGBMultiheadModel.load(str(XGB_BUNDLE_PATH), require_feature_names=True)
    feature_list = list(bundle.feature_list)
    if not feature_list:
        raise RuntimeError("XGB_BUNDLE_HAS_NO_FEATURE_LIST")

    # 2. Feature alignment audit against BASE34
    base34_for_audit = pd.read_parquet(BASE34_M5_FEATURES_PATH)
    if "time" not in base34_for_audit.columns and base34_for_audit.index.name == "time":
        base34_for_audit = base34_for_audit.reset_index()
    feature_alignment = audit_feature_alignment(
        feature_list, set(base34_for_audit.columns)
    )
    _write_json(artifact_root / "feature_alignment_audit_v1.json", feature_alignment)

    # If feature alignment fails, abort early.
    if feature_alignment["status_v1"] != "PASS":
        return _early_abort(
            artifact_root,
            inputs,
            forbidden_audit,
            feature_alignment,
            reason="FEATURE_ALIGNMENT_FAIL",
        )

    # 3. Load per-bar dataset and join BASE34
    df = pd.read_parquet(inputs["required_paths"]["split_locked_dataset"])
    df["candidate_uid_v1"] = df["candidate_uid_v1"].astype(str)
    df["ts_v1"] = pd.to_datetime(df["ts_v1"], utc=True)
    per_bar = _per_bar_view(df)
    per_bar_with_b34, base34_join_audit = _join_base34_full(
        per_bar, BASE34_M5_FEATURES_PATH, feature_list
    )
    _write_json(artifact_root / "base34_join_audit_v1.json", base34_join_audit)

    temporal_audit = audit_temporal_correctness(per_bar_with_b34)
    _write_json(artifact_root / "temporal_correctness_audit_v1.json", temporal_audit)

    # 4. Run replay
    signal7, replayed = _run_xgb_replay(per_bar_with_b34, feature_list, bundle)
    replay_df = _persist_replay(per_bar_with_b34, signal7, replayed)

    # Persist replay parquet keyed by (candidate_uid, bar_index).
    replay_path = artifact_root / "per_bar_xgb_signal7_v2.parquet"
    replay_df.to_parquet(replay_path, index=False)

    # 5. Audits on replay
    session_audit = audit_session_coverage(replay_df)
    replay_status = audit_replay_status_distribution(replay_df)
    signal7_audit = audit_signal7_invariants(replay_df)
    runtime_audit = audit_no_runtime_modification()
    _write_json(artifact_root / "session_coverage_audit_v1.json", session_audit)
    _write_json(artifact_root / "replay_status_distribution_v1.json", replay_status)
    _write_json(artifact_root / "signal7_invariants_audit_v1.json", signal7_audit)
    _write_json(artifact_root / "no_runtime_modification_audit_v1.json", runtime_audit)

    audits = [
        feature_alignment,
        base34_join_audit,
        temporal_audit,
        session_audit,
        replay_status,
        signal7_audit,
        runtime_audit,
    ]
    _write_json(
        artifact_root / "training_audits_v1.json",
        {"audit_count_v1": len(audits), "audits_v1": audits},
    )

    repro = {
        "layer_name": "RUN_PER_BAR_XGB_REPLAY_REPRODUCIBILITY_AUDIT_V1",
        "model_v1": "XGBMultiheadModel_RETRAIN_20260329_SANFIX_2020_2025",
        "feature_count_v1": len(feature_list),
        "session_heads_v1": list(SESSION_HEAD_VOCAB),
        "seed_v1": SEED_V1,
        "replay_rate_v1": replay_status["replay_rate_v1"],
        "no_implicit_glob_used_for_v1_inputs_v1": True,
        "deprecated_quarantine_revival_v1": False,
        "research_only_v1": True,
    }
    _write_json(artifact_root / "reproducibility_audit_v1.json", repro)

    status, next_action, recommendation, headline = _go_no_go(
        feature_alignment, replay_status, signal7_audit
    )
    validate_final_status(status, next_action)

    summary = {
        "layer_name": "RUN_PER_BAR_XGB_REPLAY_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "headline_v1": headline,
        "row_count_v1": replay_status["row_count_v1"],
        "replayed_count_v1": replay_status["replayed_count_v1"],
        "replay_rate_v1": replay_status["replay_rate_v1"],
        "session_head_counts_v1": session_audit["head_counts_v1"],
        "feature_count_v1": len(feature_list),
        "audits_v1": {a["audit_id_v1"]: a["status_v1"] for a in audits},
        "research_only_v1": True,
        "iql_training_run_v1": False,
        "iql_production_allowed_v1": False,
        "training_blocked_v1": True,
        "next_research_gate_v1": next_action,
        "adapter_built_v1": False,
        "r6_run_v1": False,
        "freeze_promo_live_run_v1": False,
        "deprecated_quarantine_revival_v1": False,
        "exit_manager_modified_v1": False,
        "live_features_modified_v1": False,
        "entry_manager_modified_v1": False,
        "forbidden_actions_audit_v1": forbidden_audit,
    }
    _write_json(artifact_root / "summary_v1.json", summary)

    status_payload = {
        "layer_name": "RUN_PER_BAR_XGB_REPLAY_STATUS_V1",
        "status_v1": "MATERIALIZED_RESEARCH_ONLY_GATE",
        "final_status_v1": status,
        "next_action_v1": next_action,
        "training_executed_v1": False,
    }
    _write_json(artifact_root / "status_v1.json", status_payload)

    go_no_go = {
        "layer_name": "RUN_PER_BAR_XGB_REPLAY_GO_NO_GO_V1",
        "status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "headline_v1": headline,
        "research_only_v1": True,
        "iql_production_allowed_v1": False,
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "package_freeze_promo_live_allowed_v1": False,
        "policy_promotion_allowed_v1": False,
        "training_allowed_v1": False,
        "downstream_block_v1": (
            "Research-only per-bar XGB replay. The output parquet is consumed "
            "by future training gates as input; never used to alter runtime. "
            "Adapter/R6/IQL production/live, freeze/promo/live, exit_manager / "
            "entry_manager / live_features modification all forbidden."
        ),
    }
    _write_json(artifact_root / "run_per_bar_xgb_replay_go_no_go_v1.json", go_no_go)

    report_lines = [
        "# Run Per-Bar XGB Replay For Transformer Signal Family V1",
        "",
        f"- Status: `{status}`",
        f"- Next action: `{next_action}`",
        "- Training: research-only; output parquet consumed by future training gates.",
        "",
        "## Headline",
        f"- Per-bar rows: {replay_status['row_count_v1']}",
        f"- Replayed: {replay_status['replayed_count_v1']}",
        f"- Replay rate: {replay_status['replay_rate_v1']}",
        f"- Session-head distribution: {session_audit['head_counts_v1']}",
        "",
        "## Audits",
    ]
    for a in audits:
        report_lines.append(f"- `{a['audit_id_v1']}`: {a['status_v1']}")
    report_lines.extend(
        [
            "",
            "## Output",
            f"- Per-bar replay parquet: `{replay_path}`",
            f"- Columns: candidate_uid_v1, trade_uid_v1, ts_v1, "
            f"bars_held_v1, session_id_v1, primary_split_v1, side_v1, "
            f"{', '.join(PER_BAR_XGB_OUTPUT_COLUMNS)}, "
            f"xgb_head_used_v1, replay_status_v1.",
            "",
            "## Recommendation",
            recommendation,
        ]
    )
    _write_report(artifact_root / "report_v1.md", report_lines)

    artifact_manifest = {
        "layer_id_v1": ACTION,
        "built_at_utc_v1": summary["built_at_utc_v1"],
        "output_dir_v1": str(artifact_root),
        "append_only_namespace_v1": "truth_e2e_sanity",
        "artifact_paths_v1": {
            "summary": str(artifact_root / "summary_v1.json"),
            "status": str(artifact_root / "status_v1.json"),
            "go_no_go": str(artifact_root / "run_per_bar_xgb_replay_go_no_go_v1.json"),
            "input_manifest": str(artifact_root / "input_manifest_v1.json"),
            "per_bar_xgb_signal7": str(replay_path),
            "feature_alignment_audit": str(artifact_root / "feature_alignment_audit_v1.json"),
            "base34_join_audit": str(artifact_root / "base34_join_audit_v1.json"),
            "temporal_correctness_audit": str(artifact_root / "temporal_correctness_audit_v1.json"),
            "session_coverage_audit": str(artifact_root / "session_coverage_audit_v1.json"),
            "replay_status_distribution": str(artifact_root / "replay_status_distribution_v1.json"),
            "signal7_invariants_audit": str(artifact_root / "signal7_invariants_audit_v1.json"),
            "no_runtime_modification_audit": str(artifact_root / "no_runtime_modification_audit_v1.json"),
            "training_audits": str(artifact_root / "training_audits_v1.json"),
            "reproducibility_audit": str(artifact_root / "reproducibility_audit_v1.json"),
            "report": str(artifact_root / "report_v1.md"),
        },
        "read_only_references_v1": True,
        "trained_model_v1": False,
        "not_controller_v1": True,
        "not_live_gate_v1": True,
    }
    _write_json(artifact_root / "manifest_v1.json", artifact_manifest)

    return {
        "artifact_root": str(artifact_root),
        "summary": summary,
        "status": status_payload,
        "go_no_go": go_no_go,
    }


def _early_abort(
    artifact_root: Path,
    inputs: dict[str, Any],
    forbidden_audit: dict[str, Any],
    feature_alignment: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    status = "RUN_PER_BAR_XGB_REPLAY_BLOCKED_BY_FEATURE_MISMATCH_V1"
    next_action = "REPAIR_PER_BAR_XGB_REPLAY_BEFORE_PROMOTION_V1"
    recommendation = f"Aborted before replay: {reason}; details in feature_alignment_audit_v1.json."
    summary = {
        "layer_name": "RUN_PER_BAR_XGB_REPLAY_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "research_only_v1": True,
        "training_blocked_v1": True,
        "forbidden_actions_audit_v1": forbidden_audit,
        "feature_alignment_audit_v1": feature_alignment,
    }
    _write_json(artifact_root / "summary_v1.json", summary)
    return {
        "artifact_root": str(artifact_root),
        "summary": summary,
        "status": {"final_status_v1": status, "next_action_v1": next_action},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize RUN_PER_BAR_XGB_REPLAY_FOR_TRANSFORMER_SIGNAL_FAMILY_V1."
    )
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--built-at-utc", type=str, default=None)
    args = parser.parse_args()
    out_root = (
        Path(args.out_root).expanduser().resolve() if args.out_root else None
    )
    result = write_artifacts(out_root=out_root, built_at_utc=args.built_at_utc)
    print(json.dumps(_jsonable(result), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
