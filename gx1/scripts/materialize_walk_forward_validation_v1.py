#!/usr/bin/env python3
"""Walk-forward validation of skip-V2 + V2 IQL stack across rolling time folds.

Background
----------
The combined-stack gate showed that on the SINGLE test split (Feb-Mar 2026,
258 trades) skip-V2 + V2 IQL is SUBADDITIVE: the combined +643 bps
underperforms skip-only +1842 bps. Skip dominates exit-IQL on this
cohort.

But this was measured on ONE month of test data. Risk: results are
period-specific and don't generalize. Walk-forward validation re-runs
the full training + evaluation pipeline on multiple non-overlapping test
windows and reports per-fold stability.

Folding scheme
--------------
Three EXPANDING-WINDOW folds, time-ordered per trade by open_ts_utc:

  - Fold 1: train trades 1..1000, val 1001..1200, test 1201..1400
  - Fold 2: train trades 1..1200, val 1201..1400, test 1401..1600
  - Fold 3: train trades 1..1400, val 1401..1600, test 1601..1724

Total test coverage = 524 trades across 3 non-overlapping windows
(vs the single-fold gate's 258-trade test). Each fold's val window
becomes the next fold's test window plus a buffer, so val/test rolls
forward together.

For each fold:
  1. Override primary_split_v1 on per_trade and per_bar datasets based
     on the fold's trade-rank ranges.
  2. Re-run the full skip-V2 (logistic balanced) + V2 IQL (5 reward
     variants) training, deterministic seed.
  3. Evaluate four PNL stacks on the fold's test trades:
     - NO_SKIP_REALIZED, NO_SKIP_V2_IQL, SKIP_V2_THEN_REALIZED,
       SKIP_V2_THEN_V2_IQL.
  4. Report per-policy stability across folds (mean, std, min, max).

Decision rule:
  - If skip-only's per-fold PNL is positive in ALL 3 folds AND mean lift
    > 200 bps: STABLE (the +1842 bps was not a one-off).
  - If positive in 2/3 with mean lift > 100 bps: PARTIAL_STABLE.
  - Otherwise: NOT_STABLE.
  - Same classification for IQL-only and combined.

Research-only; no policy promotion; no runtime modification; no V1/V2
contracts modified.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.scripts import (
    materialize_build_iql_offline_data_contract_research_only_v1 as contract_gate,
)
from gx1.scripts import (
    materialize_combine_skip_v2_with_exit_iql_v2_v1 as combined_gate,
)
from gx1.scripts import (
    materialize_learn_trade_skip_meta_classifier_at_trade_open_v1 as skip_v1_gate,
)
from gx1.scripts import (
    materialize_learn_trade_skip_meta_classifier_v2_logistic_balanced_v1 as skip_v2_gate,
)
from gx1.scripts import (
    materialize_run_exit_iql_with_v2_state_and_reward_variants_v1 as v2_train_gate,
)


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "WALK_FORWARD_VALIDATION_V1"

INPUT_RECOVERY_ROOT = v2_train_gate.INPUT_RECOVERY_ROOT
INPUT_SPLIT_ROOT = v2_train_gate.INPUT_SPLIT_ROOT
INPUT_V2_CONTRACT_ROOT = v2_train_gate.INPUT_V2_CONTRACT_ROOT
INPUT_COMBINED_GATE_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "COMBINE_SKIP_V2_WITH_EXIT_IQL_V2_V1_20260430T064914Z_LOCK"
)
BASE34_M5_FEATURES_PATH = v2_train_gate.BASE34_M5_FEATURES_PATH

QUARANTINE_FORBIDDEN_PATH_FRAGMENTS = ("gx1/quarantine", "gx1.quarantine")
SEED_V1 = 20260430

# 3 expanding-window folds (defined by trade rank, 1-indexed).
FOLD_DEFINITIONS: list[dict[str, int]] = [
    {"fold_id_v1": "FOLD_1", "train_start_v1": 1, "train_end_v1": 1000, "val_end_v1": 1200, "test_end_v1": 1400},
    {"fold_id_v1": "FOLD_2", "train_start_v1": 1, "train_end_v1": 1200, "val_end_v1": 1400, "test_end_v1": 1600},
    {"fold_id_v1": "FOLD_3", "train_start_v1": 1, "train_end_v1": 1400, "val_end_v1": 1600, "test_end_v1": 1724},
]

ALLOWED_FINAL_STATUSES = {
    "WALK_FORWARD_VALIDATION_PASS_SKIP_STABLE_ALL_FOLDS",
    "WALK_FORWARD_VALIDATION_PASS_SKIP_PARTIAL_STABLE",
    "WALK_FORWARD_VALIDATION_PARTIAL_SKIP_NOT_STABLE",
    "WALK_FORWARD_VALIDATION_BLOCKED_BY_INPUT_LOCK_MISSING",
}

ALLOWED_NEXT_ACTIONS = {
    "DEFINE_PROMOTION_CRITERIA_BEFORE_PAPER_TRADING_V1",
    "EXIT_PER_BAR_PROPER_IQL_WITH_PESSIMISM_V1",
    "REPAIR_RESEARCH_STACK_BEFORE_FURTHER_WORK_V1",
    "HOLD_RESEARCH_STACK_UNTIL_DATA_FIXED_V1",
}


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


def validate_fold_definitions(folds: list[dict[str, int]]) -> dict[str, Any]:
    if not folds:
        raise RuntimeError("FOLD_DEFINITIONS_EMPTY")
    for f in folds:
        if not (
            f["train_start_v1"]
            <= f["train_end_v1"]
            <= f["val_end_v1"]
            <= f["test_end_v1"]
        ):
            raise RuntimeError(f"FOLD_RANGE_INVALID: {f}")
    return {
        "audit_id_v1": "FOLD_DEFINITIONS_AUDIT_V1",
        "status_v1": "PASS",
        "fold_count_v1": len(folds),
        "folds_v1": folds,
    }


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _load_inputs() -> dict[str, Any]:
    roots = [
        INPUT_RECOVERY_ROOT,
        INPUT_SPLIT_ROOT,
        INPUT_V2_CONTRACT_ROOT,
        INPUT_COMBINED_GATE_ROOT,
    ]
    validate_explicit_artifact_roots(roots)
    required = {
        "recovery_per_trade": INPUT_RECOVERY_ROOT
        / "entry_snapshot_signals_per_trade_v1.parquet",
        "split_locked_dataset": INPUT_SPLIT_ROOT
        / "split_locked_augmented_dataset_v1.parquet",
        "v2_state_contract": INPUT_V2_CONTRACT_ROOT / "state_feature_contract_v2.json",
        "combined_gate_summary": INPUT_COMBINED_GATE_ROOT / "summary_v1.json",
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
        "combined_gate_summary": _read_json(required["combined_gate_summary"]),
        "base34_path": BASE34_M5_FEATURES_PATH,
    }


# ---------------------------------------------------------------------------
# Per-fold split assignment
# ---------------------------------------------------------------------------


def _assign_fold_split(
    candidate_uids_in_time_order: list[str],
    fold: dict[str, int],
) -> dict[str, str]:
    """Map each candidate_uid to one of {train, val, test, hold_out}
    based on the trade's 1-indexed rank in time order."""
    out: dict[str, str] = {}
    for rank, uid in enumerate(candidate_uids_in_time_order, start=1):
        if rank < fold["train_start_v1"]:
            out[uid] = "hold_out"
        elif rank <= fold["train_end_v1"]:
            out[uid] = "train"
        elif rank <= fold["val_end_v1"]:
            out[uid] = "val"
        elif rank <= fold["test_end_v1"]:
            out[uid] = "test"
        else:
            out[uid] = "hold_out"
    return out


def _override_split_column(
    df: pd.DataFrame, uid_to_split: dict[str, str], uid_col: str
) -> pd.DataFrame:
    out = df.copy()
    out[uid_col] = out[uid_col].astype(str)
    out["primary_split_v1"] = out[uid_col].map(uid_to_split).fillna("hold_out")
    # Drop hold_out rows so downstream training pipelines see only train/val/test.
    return out[out["primary_split_v1"] != "hold_out"].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-fold training + eval
# ---------------------------------------------------------------------------


def _build_per_trade_for_fold(
    inputs: dict[str, Any], uid_to_split: dict[str, str]
) -> pd.DataFrame:
    trades = skip_v1_gate._load_trade_outcomes_concat()
    recovery = pd.read_parquet(inputs["required_paths"]["recovery_per_trade"])
    per_trade, _ = skip_v1_gate._project_per_trade_features(
        trades, recovery, BASE34_M5_FEATURES_PATH
    )
    # Override split assignment by candidate_uid_v1.
    return _override_split_column(per_trade, uid_to_split, "candidate_uid_v1")


def _build_per_bar_for_fold(
    inputs: dict[str, Any], uid_to_split: dict[str, str]
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df = pd.read_parquet(inputs["required_paths"]["split_locked_dataset"])
    df["candidate_uid_v1"] = df["candidate_uid_v1"].astype(str)
    df["ts_v1"] = pd.to_datetime(df["ts_v1"], utc=True)
    per_bar_v1 = v2_train_gate._per_bar_view(df)
    per_bar_with_b34 = v2_train_gate._join_base34(
        per_bar_v1, BASE34_M5_FEATURES_PATH
    )
    per_bar_with_deriv = v2_train_gate._compute_derivatives(per_bar_with_b34)
    per_bar_full, _ = v2_train_gate._join_recovery(
        per_bar_with_deriv,
        INPUT_RECOVERY_ROOT / "entry_snapshot_signals_per_trade_v1.parquet",
    )
    per_bar_full = _override_split_column(
        per_bar_full, uid_to_split, "candidate_uid_v1"
    )
    per_bar_train = per_bar_full[per_bar_full["primary_split_v1"] == "train"]
    norm = v2_train_gate._fit_train_normalization(per_bar_train)
    X_full, feature_names = v2_train_gate._build_state_matrix_v2(per_bar_full, norm)
    return per_bar_full, X_full, feature_names


def _train_skip_v2_for_fold(
    per_trade: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    per_trade_train = per_trade[per_trade["primary_split_v1"] == "train"]
    norm = skip_v1_gate._fit_train_normalization(per_trade_train)
    X_full, _ = skip_v1_gate._build_state_matrix(per_trade, norm)
    train_mask = (per_trade["primary_split_v1"] == "train").to_numpy()
    val_mask = (per_trade["primary_split_v1"] == "val").to_numpy()
    y_full = per_trade["should_skip_v1"].astype(int).to_numpy()
    if y_full[train_mask].sum() == 0:
        # No positive examples in train -> classifier cannot learn; mark all
        # predictions as 0 (never skip) and threshold tuning trivially picks 0.5.
        return np.zeros(len(per_trade), dtype=float), 0.5
    logreg = skip_v2_gate._train_logistic(X_full[train_mask], y_full[train_mask])
    p_skip_full = skip_v2_gate._predict_p_skip(logreg, X_full)

    per_trade_val = per_trade[val_mask].reset_index(drop=True)
    p_skip_val = p_skip_full[val_mask]
    best_thr = None
    best_pnl = -np.inf
    for thr in skip_v2_gate.THRESHOLD_GRID:
        m = skip_v1_gate._evaluate_threshold(per_trade_val, p_skip_val, thr)
        if m["pnl_taken_v1"] > best_pnl:
            best_pnl = m["pnl_taken_v1"]
            best_thr = thr
    return p_skip_full, float(best_thr if best_thr is not None else 0.5)


def _train_v2_iql_for_fold(
    per_bar_full: pd.DataFrame, X_full: np.ndarray
) -> dict[str, dict[str, np.ndarray]]:
    train_mask = (per_bar_full["primary_split_v1"] == "train").to_numpy()
    per_bar_train = per_bar_full[per_bar_full["primary_split_v1"] == "train"]
    models: dict[str, dict[str, np.ndarray]] = {}
    for variant in v2_train_gate.REWARD_VARIANTS_V2:
        v_id = variant["reward_id_v1"]
        reward_col = variant["reward_column_v1"]
        targets = v2_train_gate._compute_targets_for_variant(
            per_bar_train, reward_col
        )
        target_hold = targets["__target_hold_v1"].astype(float).to_numpy()
        target_exit_now = targets["__target_exit_now_v1"].astype(float).to_numpy()
        coef_hold = v2_train_gate._ridge_fit(X_full[train_mask], target_hold)
        coef_exit_now = v2_train_gate._ridge_fit(
            X_full[train_mask], target_exit_now
        )
        models[v_id] = {"coef_hold": coef_hold, "coef_exit_now": coef_exit_now}
    return models


def _evaluate_fold_test(
    per_trade: pd.DataFrame,
    p_skip_full: np.ndarray,
    threshold: float,
    per_bar_full: pd.DataFrame,
    X_full: np.ndarray,
    models: dict[str, dict[str, np.ndarray]],
    fold_id: str,
) -> dict[str, Any]:
    eval_out = combined_gate._evaluate_combined(
        per_trade, p_skip_full, threshold, per_bar_full, X_full, models, split="test"
    )
    eval_out["fold_id_v1"] = fold_id
    return eval_out


# ---------------------------------------------------------------------------
# Stability classification
# ---------------------------------------------------------------------------


def _stability_metrics(per_fold_values: list[float]) -> dict[str, Any]:
    if not per_fold_values:
        return {
            "mean_v1": None,
            "std_v1": None,
            "min_v1": None,
            "max_v1": None,
            "n_v1": 0,
            "n_positive_v1": 0,
        }
    arr = np.array(per_fold_values, dtype=float)
    return {
        "mean_v1": float(arr.mean()),
        "std_v1": float(arr.std(ddof=0)),
        "min_v1": float(arr.min()),
        "max_v1": float(arr.max()),
        "n_v1": int(len(arr)),
        "n_positive_v1": int((arr > 0).sum()),
    }


def _classify_stability(stats: dict[str, Any]) -> str:
    if stats["n_v1"] == 0:
        return "EMPTY"
    n = stats["n_v1"]
    n_pos = stats["n_positive_v1"]
    mean = stats["mean_v1"]
    if n_pos == n and mean > 200.0:
        return "STABLE_ALL_FOLDS_POSITIVE_MEAN_LIFT_OVER_200"
    if n_pos >= n - 1 and mean > 100.0:
        return "PARTIAL_STABLE_MOSTLY_POSITIVE_MEAN_LIFT_OVER_100"
    return "NOT_STABLE"


# ---------------------------------------------------------------------------
# go-no-go
# ---------------------------------------------------------------------------


def _go_no_go(
    per_fold_results: list[dict[str, Any]],
) -> tuple[str, str, str, dict[str, Any]]:
    if not per_fold_results:
        raise RuntimeError("WALK_FORWARD_PER_FOLD_RESULTS_MISSING")

    # Best variant per fold (by pnl_skip_then_iql_v1) for combined stack;
    # plus per-policy lifts vs the no-skip-realized floor.
    skip_only_lifts: list[float] = []
    iql_only_lifts: list[float] = []
    combined_lifts: list[float] = []
    for r in per_fold_results:
        best = max(r["per_variant_v1"], key=lambda v: v["pnl_skip_then_iql_v1"])
        floor = best["pnl_no_skip_realized_v1"]
        skip_only_lifts.append(best["pnl_skip_then_realized_v1"] - floor)
        iql_only_lifts.append(best["pnl_no_skip_iql_v1"] - floor)
        combined_lifts.append(best["pnl_skip_then_iql_v1"] - floor)

    skip_stats = _stability_metrics(skip_only_lifts)
    iql_stats = _stability_metrics(iql_only_lifts)
    combined_stats = _stability_metrics(combined_lifts)
    skip_classification = _classify_stability(skip_stats)
    iql_classification = _classify_stability(iql_stats)
    combined_classification = _classify_stability(combined_stats)

    headline = {
        "fold_count_v1": len(per_fold_results),
        "skip_only_lift_stats_v1": skip_stats,
        "iql_only_lift_stats_v1": iql_stats,
        "combined_lift_stats_v1": combined_stats,
        "skip_only_classification_v1": skip_classification,
        "iql_only_classification_v1": iql_classification,
        "combined_classification_v1": combined_classification,
    }

    if skip_classification == "STABLE_ALL_FOLDS_POSITIVE_MEAN_LIFT_OVER_200":
        return (
            "WALK_FORWARD_VALIDATION_PASS_SKIP_STABLE_ALL_FOLDS",
            "DEFINE_PROMOTION_CRITERIA_BEFORE_PAPER_TRADING_V1",
            (
                f"Skip-V2 lift is STABLE across all {skip_stats['n_v1']} folds "
                f"(mean {skip_stats['mean_v1']:+.0f}, "
                f"min {skip_stats['min_v1']:+.0f}, "
                f"max {skip_stats['max_v1']:+.0f}). The +1842 bps single-fold "
                "result is a real, robust effect. Next: define promotion "
                "criteria before paper trading."
            ),
            headline,
        )
    if skip_classification == "PARTIAL_STABLE_MOSTLY_POSITIVE_MEAN_LIFT_OVER_100":
        return (
            "WALK_FORWARD_VALIDATION_PASS_SKIP_PARTIAL_STABLE",
            "DEFINE_PROMOTION_CRITERIA_BEFORE_PAPER_TRADING_V1",
            (
                f"Skip-V2 lift is PARTIAL-STABLE: positive in "
                f"{skip_stats['n_positive_v1']}/{skip_stats['n_v1']} folds; "
                f"mean {skip_stats['mean_v1']:+.0f}, range "
                f"[{skip_stats['min_v1']:+.0f}, {skip_stats['max_v1']:+.0f}]. "
                "Real but variable. Next: define promotion criteria; consider "
                "fold-conditioned models before paper trading."
            ),
            headline,
        )
    return (
        "WALK_FORWARD_VALIDATION_PARTIAL_SKIP_NOT_STABLE",
        "REPAIR_RESEARCH_STACK_BEFORE_FURTHER_WORK_V1",
        (
            f"Skip-V2 is NOT stable: positive in only "
            f"{skip_stats['n_positive_v1']}/{skip_stats['n_v1']} folds; "
            f"mean {skip_stats['mean_v1']:+.0f}, range "
            f"[{skip_stats['min_v1']:+.0f}, {skip_stats['max_v1']:+.0f}]. "
            "The single-fold result was period-specific. Investigate before "
            "further work."
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
            "sha256_v1": _file_hash(path),
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
        "layer_name": "WALK_FORWARD_VALIDATION_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "recovery_root_v1": str(INPUT_RECOVERY_ROOT),
            "split_root_v1": str(INPUT_SPLIT_ROOT),
            "v2_contract_root_v1": str(INPUT_V2_CONTRACT_ROOT),
            "combined_gate_root_v1": str(INPUT_COMBINED_GATE_ROOT),
        },
        "files_used_v1": files,
        "immutable_input_status_v1": "HASHED_EXPLICIT_ROOTS_ONLY",
        "no_implicit_latest_glob_selection_v1": True,
        "previous_artifacts_mutated_v1": False,
        "research_only_contract_v1": True,
        "iql_training_run_v1": True,
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
    fold_audit = validate_fold_definitions(FOLD_DEFINITIONS)
    _write_json(
        artifact_root / "input_manifest_v1.json",
        _build_input_manifest(inputs, artifact_root),
    )
    _write_json(artifact_root / "fold_definitions_v1.json", fold_audit)

    # Build the time-ordered candidate_uid list once.
    trades_all = skip_v1_gate._load_trade_outcomes_concat()
    trades_all["candidate_uid_v1"] = trades_all["candidate_uid"].astype(str)
    trades_all["open_ts_utc"] = pd.to_datetime(
        trades_all["open_ts_utc"], utc=True
    )
    # Filter to only trades that exist in gate-4 split (i.e. accepted trades).
    split_df = pd.read_parquet(
        inputs["required_paths"]["split_locked_dataset"],
        columns=["candidate_uid_v1"],
    )
    accepted_uids = set(split_df["candidate_uid_v1"].astype(str).unique())
    trades_accepted = trades_all[trades_all["candidate_uid_v1"].isin(accepted_uids)].copy()
    trades_accepted = trades_accepted.sort_values(
        ["open_ts_utc", "candidate_uid_v1"], kind="mergesort"
    ).reset_index(drop=True)
    candidate_uid_order = trades_accepted["candidate_uid_v1"].astype(str).tolist()
    if len(candidate_uid_order) != len(accepted_uids):
        raise RuntimeError(
            f"CANDIDATE_UID_ORDER_MISMATCH: order has {len(candidate_uid_order)} "
            f"vs split set {len(accepted_uids)}"
        )

    # Run each fold.
    per_fold_results: list[dict[str, Any]] = []
    per_fold_audit_rows: list[dict[str, Any]] = []
    for fold in FOLD_DEFINITIONS:
        fold_id = fold["fold_id_v1"]
        uid_to_split = _assign_fold_split(candidate_uid_order, fold)
        per_trade = _build_per_trade_for_fold(inputs, uid_to_split)
        per_bar, X_full, feature_names = _build_per_bar_for_fold(inputs, uid_to_split)
        # Sanity: trade count per split
        per_trade_split_counts = (
            per_trade["primary_split_v1"].value_counts().to_dict()
        )
        per_bar_split_counts = (
            per_bar["primary_split_v1"].value_counts().to_dict()
        )

        p_skip_full, threshold = _train_skip_v2_for_fold(per_trade)
        models = _train_v2_iql_for_fold(per_bar, X_full)

        # Get test trade open_ts range for traceability.
        test_uids = [
            uid
            for uid, sp in uid_to_split.items()
            if sp == "test" and uid in accepted_uids
        ]
        test_ts = trades_accepted[
            trades_accepted["candidate_uid_v1"].isin(test_uids)
        ]["open_ts_utc"]
        test_ts_min = (
            str(test_ts.min()) if len(test_ts) else None
        )
        test_ts_max = (
            str(test_ts.max()) if len(test_ts) else None
        )
        per_fold_audit_rows.append(
            {
                "fold_id_v1": fold_id,
                "train_count_v1": int(per_trade_split_counts.get("train", 0)),
                "val_count_v1": int(per_trade_split_counts.get("val", 0)),
                "test_count_v1": int(per_trade_split_counts.get("test", 0)),
                "per_bar_train_v1": int(per_bar_split_counts.get("train", 0)),
                "per_bar_val_v1": int(per_bar_split_counts.get("val", 0)),
                "per_bar_test_v1": int(per_bar_split_counts.get("test", 0)),
                "tuned_skip_threshold_v1": threshold,
                "test_open_ts_min_v1": test_ts_min,
                "test_open_ts_max_v1": test_ts_max,
            }
        )

        eval_out = _evaluate_fold_test(
            per_trade, p_skip_full, threshold, per_bar, X_full, models, fold_id
        )
        per_fold_results.append(eval_out)

    _write_rows(artifact_root / "per_fold_audit_v1.csv", per_fold_audit_rows)
    _write_json(
        artifact_root / "per_fold_audit_v1.json",
        {"row_count_v1": len(per_fold_audit_rows), "rows_v1": per_fold_audit_rows},
    )

    # Flatten per-variant per-fold results.
    flat_per_variant_rows: list[dict[str, Any]] = []
    for r in per_fold_results:
        for v in r["per_variant_v1"]:
            flat_per_variant_rows.append({**v, "fold_id_v1": r["fold_id_v1"]})
    _write_rows(
        artifact_root / "per_fold_per_variant_test_v1.csv", flat_per_variant_rows
    )
    _write_json(
        artifact_root / "per_fold_per_variant_test_v1.json",
        {"row_count_v1": len(flat_per_variant_rows), "rows_v1": flat_per_variant_rows},
    )

    repro = {
        "layer_name": "WALK_FORWARD_VALIDATION_REPRODUCIBILITY_AUDIT_V1",
        "fold_count_v1": len(FOLD_DEFINITIONS),
        "seed_v1": SEED_V1,
        "no_implicit_glob_used_for_v1_inputs_v1": True,
        "deprecated_quarantine_revival_v1": False,
        "research_only_v1": True,
    }
    _write_json(artifact_root / "reproducibility_audit_v1.json", repro)

    status, next_action, recommendation, headline = _go_no_go(per_fold_results)
    validate_final_status(status, next_action)

    summary = {
        "layer_name": "WALK_FORWARD_VALIDATION_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "headline_v1": headline,
        "fold_count_v1": len(FOLD_DEFINITIONS),
        "fold_definitions_v1": FOLD_DEFINITIONS,
        "per_fold_audit_v1": per_fold_audit_rows,
        "per_fold_results_v1": per_fold_results,
        "research_only_v1": True,
        "iql_training_run_v1": True,
        "iql_production_allowed_v1": False,
        "skip_classifier_promoted_to_runtime_v1": False,
        "training_blocked_v1": False,
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
        "layer_name": "WALK_FORWARD_VALIDATION_STATUS_V1",
        "status_v1": "MATERIALIZED_RESEARCH_ONLY_GATE",
        "final_status_v1": status,
        "next_action_v1": next_action,
        "training_executed_v1": True,
    }
    _write_json(artifact_root / "status_v1.json", status_payload)

    go_no_go = {
        "layer_name": "WALK_FORWARD_VALIDATION_GO_NO_GO_V1",
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
        "skip_classifier_promotion_allowed_v1": False,
        "downstream_block_v1": (
            "Research-only walk-forward validation. NOT promoted to runtime; "
            "entry_manager / exit_manager / live_features / V1 / V2 contracts "
            "all unmodified."
        ),
    }
    _write_json(
        artifact_root / "walk_forward_validation_go_no_go_v1.json", go_no_go
    )

    # Build report.
    report_lines = [
        "# Walk-Forward Validation V1",
        "",
        f"- Status: `{status}`",
        f"- Next action: `{next_action}`",
        "- Training: research-only; results NOT promoted to runtime.",
        "",
        "## Headline (cross-fold stability)",
        f"- Folds evaluated: {headline['fold_count_v1']}",
        f"- Skip-only lift: mean {headline['skip_only_lift_stats_v1']['mean_v1']:+.0f} bps, "
        f"range [{headline['skip_only_lift_stats_v1']['min_v1']:+.0f}, "
        f"{headline['skip_only_lift_stats_v1']['max_v1']:+.0f}], "
        f"positive in {headline['skip_only_lift_stats_v1']['n_positive_v1']}/"
        f"{headline['skip_only_lift_stats_v1']['n_v1']} folds. "
        f"Classification: **{headline['skip_only_classification_v1']}**",
        f"- IQL-only lift: mean {headline['iql_only_lift_stats_v1']['mean_v1']:+.0f} bps, "
        f"range [{headline['iql_only_lift_stats_v1']['min_v1']:+.0f}, "
        f"{headline['iql_only_lift_stats_v1']['max_v1']:+.0f}], "
        f"positive in {headline['iql_only_lift_stats_v1']['n_positive_v1']}/"
        f"{headline['iql_only_lift_stats_v1']['n_v1']} folds. "
        f"Classification: **{headline['iql_only_classification_v1']}**",
        f"- Combined lift: mean {headline['combined_lift_stats_v1']['mean_v1']:+.0f} bps, "
        f"range [{headline['combined_lift_stats_v1']['min_v1']:+.0f}, "
        f"{headline['combined_lift_stats_v1']['max_v1']:+.0f}], "
        f"positive in {headline['combined_lift_stats_v1']['n_positive_v1']}/"
        f"{headline['combined_lift_stats_v1']['n_v1']} folds. "
        f"Classification: **{headline['combined_classification_v1']}**",
        "",
        "## Per-fold breakdown",
        "",
        "| Fold | Test trades | Test ts range | Tuned thr | No-skip+real | Skip+real | No-skip+IQL | Skip+IQL (best) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r, audit in zip(per_fold_results, per_fold_audit_rows):
        best = max(r["per_variant_v1"], key=lambda v: v["pnl_skip_then_iql_v1"])
        report_lines.append(
            f"| `{r['fold_id_v1']}` | {audit['test_count_v1']} | "
            f"{audit['test_open_ts_min_v1']} ... {audit['test_open_ts_max_v1']} | "
            f"{audit['tuned_skip_threshold_v1']} | "
            f"{best['pnl_no_skip_realized_v1']:.0f} | "
            f"{best['pnl_skip_then_realized_v1']:.0f} | "
            f"{best['pnl_no_skip_iql_v1']:.0f} | "
            f"**{best['pnl_skip_then_iql_v1']:.0f}** ({best['reward_variant_v1']}) |"
        )
    report_lines.extend(["", "## Recommendation", recommendation])
    _write_report(artifact_root / "report_v1.md", report_lines)

    artifact_manifest = {
        "layer_id_v1": ACTION,
        "built_at_utc_v1": summary["built_at_utc_v1"],
        "output_dir_v1": str(artifact_root),
        "append_only_namespace_v1": "truth_e2e_sanity",
        "artifact_paths_v1": {
            "summary": str(artifact_root / "summary_v1.json"),
            "status": str(artifact_root / "status_v1.json"),
            "go_no_go": str(artifact_root / "walk_forward_validation_go_no_go_v1.json"),
            "input_manifest": str(artifact_root / "input_manifest_v1.json"),
            "fold_definitions": str(artifact_root / "fold_definitions_v1.json"),
            "per_fold_audit_csv": str(artifact_root / "per_fold_audit_v1.csv"),
            "per_fold_audit_json": str(artifact_root / "per_fold_audit_v1.json"),
            "per_fold_per_variant_test_csv": str(
                artifact_root / "per_fold_per_variant_test_v1.csv"
            ),
            "per_fold_per_variant_test_json": str(
                artifact_root / "per_fold_per_variant_test_v1.json"
            ),
            "reproducibility_audit": str(artifact_root / "reproducibility_audit_v1.json"),
            "report": str(artifact_root / "report_v1.md"),
        },
        "read_only_references_v1": True,
        "trained_model_v1": True,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize WALK_FORWARD_VALIDATION_V1.")
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
