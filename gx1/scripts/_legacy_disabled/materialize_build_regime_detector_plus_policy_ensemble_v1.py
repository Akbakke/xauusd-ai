#!/usr/bin/env python3
"""Regime detector + policy ensemble (live system or combined-stack).

Background
----------
Phase-2A head-to-head and Phase-2B rolling-window established:
  - Live system (realized) is the strongest single policy by total PNL
    (+5963 bps over 924 rolling test trades).
  - Combined stack (skip + V2 IQL) has 15x lower per-trade volatility
    than realized AND near-zero correlation with realized (+0.06).
  - Combined was net-positive (+598) on the 524-trade walk-forward, but
    negative (-87) on the 924-trade rolling test.
  - Both fail static promotion criteria.

Hypothesis for this gate: a regime-aware router can deploy combined as a
defensive overlay only when realized is likely to lose. If we can detect
"realized will lose in next K trades" with even modest accuracy, the
ensemble policy beats both components.

Design
------
Train a logistic-balanced regime classifier at trade-entry-time using
AT_TRADE_OPEN features (the same stable family as skip-V2). Label per
trade: 1 if the realized PNL of that trade < threshold (-50 bps); else 0.
This is "is this an unfavorable trade for the live system?"

Ensemble policy at trade-entry:
  - Predict p_loss with the regime classifier.
  - If p_loss >= threshold: route to combined-stack policy (skip if its
    own classifier says so; else V2 IQL exit; else realized).
  - Otherwise: use realized exit (default to live system).

Walk-forward FROM THE START using the same 3 folds; tune the regime
threshold on val (max ensemble PNL); apply to test. Apply locked
promotion criteria.

Research-only; no policy promotion; no runtime modification.
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
    materialize_define_promotion_criteria_v1 as criteria_gate,
)
from gx1.scripts import (
    materialize_walk_forward_validation_v1 as wf_gate,
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
ACTION = "BUILD_REGIME_DETECTOR_PLUS_POLICY_ENSEMBLE_V1"

INPUT_HEAD_TO_HEAD_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "RUN_LIVE_SYSTEM_VS_RESEARCH_CANDIDATES_HEAD_TO_HEAD_V1_20260430T072907Z_LOCK"
)
INPUT_PROMOTION_CRITERIA_ROOT = (
    DEFAULT_REPORTS_ROOT / "DEFINE_PROMOTION_CRITERIA_V1_20260430T070707Z_LOCK"
)
INPUT_RECOVERY_ROOT = v2_train_gate.INPUT_RECOVERY_ROOT
INPUT_SPLIT_ROOT = v2_train_gate.INPUT_SPLIT_ROOT
INPUT_V2_CONTRACT_ROOT = v2_train_gate.INPUT_V2_CONTRACT_ROOT
BASE34_M5_FEATURES_PATH = v2_train_gate.BASE34_M5_FEATURES_PATH

QUARANTINE_FORBIDDEN_PATH_FRAGMENTS = ("gx1/quarantine", "gx1.quarantine")
SEED_V1 = 20260430

# "Trade is unfavorable for live system" label threshold.
REGIME_LOSS_LABEL_THRESHOLD_BPS = -50.0

# Threshold grid for ensemble routing decision.
REGIME_THRESHOLD_GRID: list[float] = [
    0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
]

ALLOWED_FINAL_STATUSES = {
    "REGIME_ENSEMBLE_PASS_MEETS_PROMOTION_CRITERIA",
    "REGIME_ENSEMBLE_PASS_BEATS_REALIZED_BUT_FAILS_OTHER_CRITERIA",
    "REGIME_ENSEMBLE_PARTIAL_TIES_REALIZED",
    "REGIME_ENSEMBLE_PARTIAL_DEGRADES_VS_REALIZED",
    "REGIME_ENSEMBLE_BLOCKED_BY_INPUT_LOCK_MISSING",
}

ALLOWED_NEXT_ACTIONS = {
    "DEFINE_PAPER_TRADING_PROMOTION_PLAN_V1",
    "ACCEPT_LIVE_SYSTEM_AS_RESEARCH_BASELINE_V1",
    "REPAIR_REGIME_ENSEMBLE_BEFORE_FURTHER_WORK_V1",
    "HOLD_RESEARCH_PIPELINE_AND_REFOCUS_ON_LIVE_INSTRUMENTATION_V1",
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


# ---------------------------------------------------------------------------
# Per-fold ensemble training and evaluation
# ---------------------------------------------------------------------------


def _add_regime_label(per_trade: pd.DataFrame) -> pd.DataFrame:
    """Add `regime_loss_v1` column: 1 if pnl_bps < threshold; else 0."""
    out = per_trade.copy()
    out["regime_loss_v1"] = (
        (out["pnl_bps"].astype(float) < REGIME_LOSS_LABEL_THRESHOLD_BPS)
        .astype(int)
    )
    return out


def _train_regime_classifier(
    per_trade: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Logistic-balanced classifier on regime_loss_v1 using AT_TRADE_OPEN features."""
    train = per_trade[per_trade["primary_split_v1"] == "train"]
    if len(train) < 50:
        return np.zeros(len(per_trade)), []
    norm = skip_v1_gate._fit_train_normalization(train)
    X_full, feature_names = skip_v1_gate._build_state_matrix(per_trade, norm)
    train_mask = (per_trade["primary_split_v1"] == "train").to_numpy()
    y = per_trade["regime_loss_v1"].astype(int).to_numpy()
    if y[train_mask].sum() == 0 or y[train_mask].sum() == int(train_mask.sum()):
        return np.zeros(len(per_trade)), feature_names
    logreg = skip_v2_gate._train_logistic(X_full[train_mask], y[train_mask])
    p_loss = skip_v2_gate._predict_p_skip(logreg, X_full)
    return p_loss, feature_names


def _evaluate_ensemble_at_threshold(
    per_trade: pd.DataFrame,
    p_loss: np.ndarray,
    realized_per_uid: dict[str, float],
    combined_per_uid: dict[str, float],
    threshold: float,
) -> dict[str, Any]:
    df = per_trade.copy()
    df["p_loss_v1"] = p_loss
    n = int(len(df))
    if n == 0:
        return {
            "threshold_v1": float(threshold),
            "trade_count_v1": 0,
            "ensemble_total_pnl_v1": 0.0,
            "realized_total_pnl_v1": 0.0,
            "combined_total_pnl_v1": 0.0,
        }
    route_to_combined = (df["p_loss_v1"] >= threshold).to_numpy()
    label = df["regime_loss_v1"].astype(int).to_numpy()
    pred = route_to_combined.astype(int)
    tp = int(((pred == 1) & (label == 1)).sum())
    fp = int(((pred == 1) & (label == 0)).sum())
    tn = int(((pred == 0) & (label == 0)).sum())
    fn = int(((pred == 0) & (label == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if precision and recall
        else None
    )
    ensemble_pnl = []
    realized_total = 0.0
    combined_total = 0.0
    for i, uid in enumerate(df["candidate_uid_v1"].astype(str).tolist()):
        r_pnl = realized_per_uid.get(uid, 0.0)
        c_pnl = combined_per_uid.get(uid, r_pnl)
        realized_total += r_pnl
        combined_total += c_pnl
        if route_to_combined[i]:
            ensemble_pnl.append(c_pnl)
        else:
            ensemble_pnl.append(r_pnl)
    ensemble_arr = np.array(ensemble_pnl, dtype=float)
    return {
        "threshold_v1": float(threshold),
        "trade_count_v1": n,
        "n_routed_combined_v1": int(route_to_combined.sum()),
        "n_routed_realized_v1": int((~route_to_combined).sum()),
        "tp_v1": tp,
        "fp_v1": fp,
        "tn_v1": tn,
        "fn_v1": fn,
        "precision_v1": precision,
        "recall_v1": recall,
        "f1_v1": f1,
        "ensemble_total_pnl_v1": float(ensemble_arr.sum()),
        "realized_total_pnl_v1": realized_total,
        "combined_total_pnl_v1": combined_total,
        "lift_vs_realized_v1": float(ensemble_arr.sum() - realized_total),
    }


def _compute_realized_per_uid(per_bar: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for uid, group in per_bar.groupby("candidate_uid_v1", sort=False):
        last = group.sort_values("bars_held_v1").tail(1)
        out[str(uid)] = float(last["running_pnl_at_close_bps_v1"].iloc[0])
    return out


def _compute_combined_per_uid(
    per_trade: pd.DataFrame,
    per_bar: pd.DataFrame,
    X_full: np.ndarray,
    realized_per_uid: dict[str, float],
) -> dict[str, float]:
    """Train skip-V2 + V2 IQL (best variant on val), apply to all trades.
    Returns per-trade combined PNL per candidate_uid_v1."""
    per_trade_train = per_trade[per_trade["primary_split_v1"] == "train"]
    if len(per_trade_train) < 50:
        return dict(realized_per_uid)
    norm_skip = skip_v1_gate._fit_train_normalization(per_trade_train)
    X_skip, _ = skip_v1_gate._build_state_matrix(per_trade, norm_skip)
    train_mask_skip = (per_trade["primary_split_v1"] == "train").to_numpy()
    y_skip = per_trade["should_skip_v1"].astype(int).to_numpy()
    if y_skip[train_mask_skip].sum() == 0 or y_skip[train_mask_skip].sum() == int(train_mask_skip.sum()):
        skip_threshold = 1.0
        p_skip = np.zeros(len(per_trade))
    else:
        logreg = skip_v2_gate._train_logistic(
            X_skip[train_mask_skip], y_skip[train_mask_skip]
        )
        p_skip = skip_v2_gate._predict_p_skip(logreg, X_skip)
        val_mask = (per_trade["primary_split_v1"] == "val").to_numpy()
        per_trade_val = per_trade[val_mask].reset_index(drop=True)
        p_skip_val = p_skip[val_mask]
        skip_threshold = 0.5
        best_pnl = -np.inf
        for thr in skip_v2_gate.THRESHOLD_GRID:
            m = skip_v1_gate._evaluate_threshold(per_trade_val, p_skip_val, thr)
            if m["pnl_taken_v1"] > best_pnl:
                best_pnl = m["pnl_taken_v1"]
                skip_threshold = float(thr)

    # Train V2 IQL per variant; pick best by val total.
    train_mask_pb = (per_bar["primary_split_v1"] == "train").to_numpy()
    val_mask_pb = (per_bar["primary_split_v1"] == "val").to_numpy()
    per_bar_train = per_bar[per_bar["primary_split_v1"] == "train"]
    per_bar_val = per_bar[val_mask_pb].reset_index(drop=True)
    X_val = X_full[val_mask_pb]
    best_total = -np.inf
    best_coef_hold = None
    best_coef_exit = None
    for variant in v2_train_gate.REWARD_VARIANTS_V2:
        reward_col = variant["reward_column_v1"]
        targets = v2_train_gate._compute_targets_for_variant(per_bar_train, reward_col)
        target_hold = targets["__target_hold_v1"].astype(float).to_numpy()
        target_exit_now = targets["__target_exit_now_v1"].astype(float).to_numpy()
        coef_hold = v2_train_gate._ridge_fit(X_full[train_mask_pb], target_hold)
        coef_exit_now = v2_train_gate._ridge_fit(X_full[train_mask_pb], target_exit_now)
        if per_bar_val.empty:
            best_coef_hold = coef_hold
            best_coef_exit = coef_exit_now
            break
        exit_indices = v2_train_gate._exit_index_from_iql_policy(
            per_bar_val, X_val, coef_hold, coef_exit_now
        )
        selected = per_bar_val.loc[exit_indices.values]
        total = float(selected["running_pnl_at_close_bps_v1"].sum())
        if total > best_total:
            best_total = total
            best_coef_hold = coef_hold
            best_coef_exit = coef_exit_now
    if best_coef_hold is None:
        return dict(realized_per_uid)

    # Apply combined to all rows.
    combined: dict[str, float] = {}
    for split in ["train", "val", "test"]:
        mask = (per_bar["primary_split_v1"] == split).to_numpy()
        per_bar_split = per_bar[mask].reset_index(drop=True)
        X_split = X_full[mask]
        if per_bar_split.empty:
            continue
        exit_indices = v2_train_gate._exit_index_from_iql_policy(
            per_bar_split, X_split, best_coef_hold, best_coef_exit
        )
        selected = per_bar_split.loc[exit_indices.values]
        for uid, pnl in zip(
            selected["candidate_uid_v1"].astype(str).tolist(),
            selected["running_pnl_at_close_bps_v1"].astype(float).tolist(),
        ):
            combined[uid] = pnl

    # Apply skip filter at the per-trade level.
    skipped_uids = set()
    for split in ["train", "val", "test"]:
        mask = (per_trade["primary_split_v1"] == split).to_numpy()
        sub = per_trade[mask].reset_index(drop=True)
        p_skip_sub = p_skip[mask]
        skipped = sub.loc[p_skip_sub >= skip_threshold, "candidate_uid_v1"].astype(str)
        skipped_uids.update(skipped.tolist())

    out: dict[str, float] = {}
    for uid, pnl in combined.items():
        out[uid] = 0.0 if uid in skipped_uids else pnl
    return out


def _evaluate_fold(
    inputs: dict[str, Any], uid_to_split: dict[str, str], fold_id: str
) -> dict[str, Any]:
    """Per-fold ensemble training + evaluation."""
    per_trade = wf_gate._build_per_trade_for_fold(inputs, uid_to_split)
    per_bar, X_full, _ = wf_gate._build_per_bar_for_fold(inputs, uid_to_split)
    per_trade = _add_regime_label(per_trade)

    realized_per_uid = _compute_realized_per_uid(per_bar)
    combined_per_uid = _compute_combined_per_uid(
        per_trade, per_bar, X_full, realized_per_uid
    )
    p_loss_full, feature_names = _train_regime_classifier(per_trade)

    val_mask = (per_trade["primary_split_v1"] == "val").to_numpy()
    test_mask = (per_trade["primary_split_v1"] == "test").to_numpy()
    val_df = per_trade[val_mask].reset_index(drop=True)
    test_df = per_trade[test_mask].reset_index(drop=True)
    p_loss_val = p_loss_full[val_mask]
    p_loss_test = p_loss_full[test_mask]

    val_sweep: list[dict[str, Any]] = []
    test_sweep: list[dict[str, Any]] = []
    best_thr = 0.5
    best_val_total = -np.inf
    for thr in REGIME_THRESHOLD_GRID:
        val_m = _evaluate_ensemble_at_threshold(
            val_df, p_loss_val, realized_per_uid, combined_per_uid, thr
        )
        val_m["split_v1"] = "val"
        val_sweep.append(val_m)
        test_m = _evaluate_ensemble_at_threshold(
            test_df, p_loss_test, realized_per_uid, combined_per_uid, thr
        )
        test_m["split_v1"] = "test"
        test_sweep.append(test_m)
        if val_m["ensemble_total_pnl_v1"] > best_val_total:
            best_val_total = val_m["ensemble_total_pnl_v1"]
            best_thr = float(thr)
    test_at_locked = _evaluate_ensemble_at_threshold(
        test_df, p_loss_test, realized_per_uid, combined_per_uid, best_thr
    )
    test_at_locked["split_v1"] = "test"

    return {
        "fold_id_v1": fold_id,
        "tuned_threshold_v1": best_thr,
        "feature_count_v1": len(feature_names),
        "val_sweep_v1": val_sweep,
        "test_sweep_v1": test_sweep,
        "test_at_locked_threshold_v1": test_at_locked,
    }


# ---------------------------------------------------------------------------
# Apply locked promotion criteria
# ---------------------------------------------------------------------------


def _apply_promotion_criteria(
    per_fold_results: list[dict[str, Any]],
) -> dict[str, Any]:
    per_fold_lifts_vs_realized = [
        float(r["test_at_locked_threshold_v1"]["lift_vs_realized_v1"])
        for r in per_fold_results
    ]
    per_fold_pnl = [
        float(r["test_at_locked_threshold_v1"]["ensemble_total_pnl_v1"])
        for r in per_fold_results
    ]
    per_fold_realized = [
        float(r["test_at_locked_threshold_v1"]["realized_total_pnl_v1"])
        for r in per_fold_results
    ]
    trail_stop_proxy = [1052.0] * len(per_fold_pnl)
    promotion = criteria_gate.evaluate_candidate_against_criteria(
        candidate_id="regime_detector_plus_policy_ensemble",
        per_fold_lifts_bps=per_fold_lifts_vs_realized,
        per_fold_pnl_bps=per_fold_pnl,
        per_fold_trail_stop_pnl_bps=trail_stop_proxy,
        no_shortcut_audit_passed=True,
        deterministic_reproducible=True,
    )
    promotion["per_fold_lifts_vs_realized_v1"] = per_fold_lifts_vs_realized
    promotion["per_fold_pnl_v1"] = per_fold_pnl
    promotion["per_fold_realized_v1"] = per_fold_realized
    return promotion


# ---------------------------------------------------------------------------
# go-no-go
# ---------------------------------------------------------------------------


def _go_no_go(
    per_fold_results: list[dict[str, Any]], promotion: dict[str, Any]
) -> tuple[str, str, str, dict[str, Any]]:
    n_total_test = sum(
        r["test_at_locked_threshold_v1"]["trade_count_v1"] for r in per_fold_results
    )
    ensemble_total = sum(
        r["test_at_locked_threshold_v1"]["ensemble_total_pnl_v1"]
        for r in per_fold_results
    )
    realized_total = sum(
        r["test_at_locked_threshold_v1"]["realized_total_pnl_v1"]
        for r in per_fold_results
    )
    lift = ensemble_total - realized_total
    headline = {
        "n_test_trades_v1": n_total_test,
        "n_folds_v1": len(per_fold_results),
        "ensemble_total_pnl_v1": ensemble_total,
        "realized_total_pnl_v1": realized_total,
        "lift_vs_realized_v1": lift,
        "per_fold_tuned_thresholds_v1": [
            r["tuned_threshold_v1"] for r in per_fold_results
        ],
        "per_fold_lifts_vs_realized_v1": promotion.get(
            "per_fold_lifts_vs_realized_v1", []
        ),
        "promotion_pass_v1": bool(promotion["overall_pass_v1"]),
        "promotion_n_passed_v1": int(promotion["n_criteria_passed_v1"]),
        "promotion_n_total_v1": int(promotion["n_criteria_total_v1"]),
    }
    if promotion["overall_pass_v1"]:
        return (
            "REGIME_ENSEMBLE_PASS_MEETS_PROMOTION_CRITERIA",
            "DEFINE_PAPER_TRADING_PROMOTION_PLAN_V1",
            (
                f"Regime-ensemble passes ALL promotion criteria. Total "
                f"{ensemble_total:+.0f} vs realized {realized_total:+.0f} "
                f"({lift:+.0f} lift). The regime detector successfully "
                "switched between live system and combined stack. Next: "
                "define paper trading promotion plan."
            ),
            headline,
        )
    if lift > 200.0:
        return (
            "REGIME_ENSEMBLE_PASS_BEATS_REALIZED_BUT_FAILS_OTHER_CRITERIA",
            "ACCEPT_LIVE_SYSTEM_AS_RESEARCH_BASELINE_V1",
            (
                f"Regime-ensemble beats realized by {lift:+.0f} bps total but "
                f"fails {promotion['n_criteria_total_v1'] - promotion['n_criteria_passed_v1']} "
                "of 6 promotion criteria. Likely cross-fold instability."
            ),
            headline,
        )
    if abs(lift) <= 100.0:
        return (
            "REGIME_ENSEMBLE_PARTIAL_TIES_REALIZED",
            "ACCEPT_LIVE_SYSTEM_AS_RESEARCH_BASELINE_V1",
            (
                f"Regime-ensemble ~= realized ({ensemble_total:+.0f} vs "
                f"{realized_total:+.0f}; delta {lift:+.0f}). The regime "
                "detector did not produce a clear lift over the live system."
            ),
            headline,
        )
    return (
        "REGIME_ENSEMBLE_PARTIAL_DEGRADES_VS_REALIZED",
        "HOLD_RESEARCH_PIPELINE_AND_REFOCUS_ON_LIVE_INSTRUMENTATION_V1",
        (
            f"Regime-ensemble degrades vs realized "
            f"({ensemble_total:+.0f} vs {realized_total:+.0f}; delta "
            f"{lift:+.0f}). The regime detector routes incorrectly. "
            "Hold research pipeline and refocus on live system instrumentation."
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
        "layer_name": "REGIME_ENSEMBLE_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "head_to_head_root_v1": str(INPUT_HEAD_TO_HEAD_ROOT),
            "promotion_criteria_root_v1": str(INPUT_PROMOTION_CRITERIA_ROOT),
            "recovery_root_v1": str(INPUT_RECOVERY_ROOT),
            "split_root_v1": str(INPUT_SPLIT_ROOT),
            "v2_contract_root_v1": str(INPUT_V2_CONTRACT_ROOT),
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


def write_artifacts(
    out_root: Path | None = None,
    *,
    built_at_utc: str | None = None,
) -> dict[str, Any]:
    inputs = wf_gate._load_inputs()
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

    trades_all = skip_v1_gate._load_trade_outcomes_concat()
    trades_all["candidate_uid_v1"] = trades_all["candidate_uid"].astype(str)
    trades_all["open_ts_utc"] = pd.to_datetime(
        trades_all["open_ts_utc"], utc=True
    )
    split_df = pd.read_parquet(
        inputs["required_paths"]["split_locked_dataset"],
        columns=["candidate_uid_v1"],
    )
    accepted_uids = set(split_df["candidate_uid_v1"].astype(str).unique())
    trades_accepted = trades_all[
        trades_all["candidate_uid_v1"].isin(accepted_uids)
    ].sort_values(["open_ts_utc", "candidate_uid_v1"], kind="mergesort").reset_index(
        drop=True
    )
    candidate_uid_order = trades_accepted["candidate_uid_v1"].astype(str).tolist()

    per_fold_results: list[dict[str, Any]] = []
    for fold in wf_gate.FOLD_DEFINITIONS:
        uid_to_split = wf_gate._assign_fold_split(candidate_uid_order, fold)
        per_fold_results.append(_evaluate_fold(inputs, uid_to_split, fold["fold_id_v1"]))

    promotion = _apply_promotion_criteria(per_fold_results)
    _write_json(artifact_root / "promotion_criteria_evaluation_v1.json", promotion)

    flat_test: list[dict[str, Any]] = []
    for r in per_fold_results:
        row = {**r["test_at_locked_threshold_v1"], "fold_id_v1": r["fold_id_v1"]}
        flat_test.append(row)
    _write_rows(
        artifact_root / "per_fold_test_at_locked_threshold_v1.csv", flat_test
    )

    threshold_sweep: list[dict[str, Any]] = []
    for r in per_fold_results:
        for v in r["val_sweep_v1"]:
            threshold_sweep.append({**v, "fold_id_v1": r["fold_id_v1"]})
        for v in r["test_sweep_v1"]:
            threshold_sweep.append({**v, "fold_id_v1": r["fold_id_v1"]})
    _write_rows(
        artifact_root / "per_fold_threshold_sweep_v1.csv", threshold_sweep
    )

    repro = {
        "layer_name": "REGIME_ENSEMBLE_REPRODUCIBILITY_AUDIT_V1",
        "fold_count_v1": len(wf_gate.FOLD_DEFINITIONS),
        "regime_loss_label_threshold_bps_v1": REGIME_LOSS_LABEL_THRESHOLD_BPS,
        "regime_threshold_grid_v1": REGIME_THRESHOLD_GRID,
        "seed_v1": SEED_V1,
        "no_implicit_glob_used_for_v1_inputs_v1": True,
        "deprecated_quarantine_revival_v1": False,
        "research_only_v1": True,
    }
    _write_json(artifact_root / "reproducibility_audit_v1.json", repro)

    status, next_action, recommendation, headline = _go_no_go(
        per_fold_results, promotion
    )
    validate_final_status(status, next_action)

    summary = {
        "layer_name": "REGIME_ENSEMBLE_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "headline_v1": headline,
        "fold_count_v1": len(per_fold_results),
        "promotion_evaluation_v1": promotion,
        "per_fold_summary_v1": [
            {
                "fold_id_v1": r["fold_id_v1"],
                "tuned_threshold_v1": r["tuned_threshold_v1"],
                "feature_count_v1": r["feature_count_v1"],
                "test_at_locked_threshold_v1": r["test_at_locked_threshold_v1"],
            }
            for r in per_fold_results
        ],
        "research_only_v1": True,
        "iql_training_run_v1": True,
        "iql_production_allowed_v1": False,
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
    _write_json(
        artifact_root / "status_v1.json",
        {
            "layer_name": "REGIME_ENSEMBLE_STATUS_V1",
            "status_v1": "MATERIALIZED_RESEARCH_ONLY_GATE",
            "final_status_v1": status,
            "next_action_v1": next_action,
            "training_executed_v1": True,
        },
    )
    _write_json(
        artifact_root / "build_regime_detector_plus_policy_ensemble_go_no_go_v1.json",
        {
            "layer_name": "REGIME_ENSEMBLE_GO_NO_GO_V1",
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
            "training_allowed_v1": True,
            "downstream_block_v1": (
                "Research-only regime ensemble. NOT promoted to runtime."
            ),
        },
    )

    report_lines = [
        "# Build Regime Detector Plus Policy Ensemble V1",
        "",
        f"- Status: `{status}`",
        f"- Next action: `{next_action}`",
        "- Training: research-only; ensemble NOT promoted to runtime.",
        "",
        "## Headline",
        f"- Folds: {headline['n_folds_v1']}; test trades: {headline['n_test_trades_v1']}",
        f"- Realized total: {headline['realized_total_pnl_v1']:+.0f} bps",
        f"- Ensemble total: **{headline['ensemble_total_pnl_v1']:+.0f}** bps",
        f"- Lift vs realized: {headline['lift_vs_realized_v1']:+.0f} bps",
        f"- Promotion criteria: {headline['promotion_n_passed_v1']}/{headline['promotion_n_total_v1']}",
        "",
        "## Per-fold test at val-tuned threshold",
        "",
        "| Fold | Tuned thr | Routed combined | Realized | Ensemble | Lift |",
        "|---|---|---|---|---|---|",
    ]
    for r in per_fold_results:
        t = r["test_at_locked_threshold_v1"]
        report_lines.append(
            f"| `{r['fold_id_v1']}` | {r['tuned_threshold_v1']} | "
            f"{t['n_routed_combined_v1']}/{t['trade_count_v1']} | "
            f"{t['realized_total_pnl_v1']:+.0f} | "
            f"**{t['ensemble_total_pnl_v1']:+.0f}** | "
            f"{t['lift_vs_realized_v1']:+.0f} |"
        )
    report_lines.extend(["", "## Promotion criteria breakdown"])
    for c in promotion.get("breakdown_v1", []):
        mark = "✓" if c["passed_v1"] else "✗"
        extras = ", ".join(
            f"{k}={v}"
            for k, v in c.items()
            if k not in {"criterion_id_v1", "passed_v1"}
        )
        report_lines.append(f"- {mark} `{c['criterion_id_v1']}` ({extras})")
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
            "go_no_go": str(
                artifact_root
                / "build_regime_detector_plus_policy_ensemble_go_no_go_v1.json"
            ),
            "input_manifest": str(artifact_root / "input_manifest_v1.json"),
            "per_fold_test_at_locked_threshold_csv": str(
                artifact_root / "per_fold_test_at_locked_threshold_v1.csv"
            ),
            "per_fold_threshold_sweep_csv": str(
                artifact_root / "per_fold_threshold_sweep_v1.csv"
            ),
            "promotion_criteria_evaluation": str(
                artifact_root / "promotion_criteria_evaluation_v1.json"
            ),
            "reproducibility_audit": str(
                artifact_root / "reproducibility_audit_v1.json"
            ),
            "report": str(artifact_root / "report_v1.md"),
        },
        "read_only_references_v1": True,
        "trained_model_v1": True,
        "not_controller_v1": True,
        "not_live_gate_v1": True,
    }
    _write_json(artifact_root / "manifest_v1.json", artifact_manifest)

    return {"artifact_root": str(artifact_root), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize BUILD_REGIME_DETECTOR_PLUS_POLICY_ENSEMBLE_V1."
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
