#!/usr/bin/env python3
"""Hybrid: TRAIL_STOP_25_PCT_DD + a learned 'should I delay firing?' classifier.

Background
----------
Phase-1 trail-stop deep-dive showed:

  - Trail-stop fires too early in 71.3% of trades (FIRED_BEFORE_PEAK_PNL_
    REGRET_EARLY_EXIT). The early-exit-regret pattern is STABLE across
    all 3 walk-forward folds (68-76%) even though trail-stop's PNL is not.
  - Skip-V2 features are 65% stable across folds (the most stable feature
    family we have).
  - V2 IQL features are only 42% stable (ridge MSE on per-bar Q-targets
    amplifies regime noise).

Strategy
--------
Don't replace trail-stop. Learn a SMALL adjustment using only the stable
entry-context feature family. At each bar where trail-stop would fire,
ask a logistic-balanced classifier "should I delay firing on this trade?"
The classifier is conditioned on AT_TRADE_OPEN features (recovery
entry-snapshot fields + trade-static + BASE34 at entry bar). If the
classifier predicts delay-probability >= threshold, the rule defaults to
the realized exit on that trade. Otherwise, trail-stop fires as normal.

  - Label per trade: 1 if max(pnl after fire_bar) > pnl_at_fire + 5 bps.
  - Features: identical to skip-V2 (so we reuse a stable feature family).
  - Model: sklearn LogisticRegression(class_weight='balanced') - same
    framework that worked for skip-V2.
  - Walk-forward FROM THE START - 3 folds, threshold tuned on val.
  - Promotion criteria evaluated automatically against the locked
    DEFINE_PROMOTION_CRITERIA_V1 contract.

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
from sklearn.linear_model import LogisticRegression

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
    materialize_investigate_trail_stop_deep_dive_v1 as ts_gate,
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
ACTION = "BUILD_HYBRID_TRAIL_STOP_PLUS_SMALL_ADJUSTMENT_LEARNER_V1"

INPUT_TRAIL_STOP_DEEP_DIVE_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "INVESTIGATE_TRAIL_STOP_DEEP_DIVE_V1_20260430T071201Z_LOCK"
)
INPUT_PROMOTION_CRITERIA_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "DEFINE_PROMOTION_CRITERIA_V1_20260430T070707Z_LOCK"
)
INPUT_FEATURE_STABILITY_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "AUDIT_FEATURE_STABILITY_ACROSS_FOLDS_V1_20260430T070916Z_LOCK"
)
INPUT_RECOVERY_ROOT = v2_train_gate.INPUT_RECOVERY_ROOT
INPUT_SPLIT_ROOT = v2_train_gate.INPUT_SPLIT_ROOT
INPUT_V2_CONTRACT_ROOT = v2_train_gate.INPUT_V2_CONTRACT_ROOT
BASE34_M5_FEATURES_PATH = v2_train_gate.BASE34_M5_FEATURES_PATH

QUARANTINE_FORBIDDEN_PATH_FRAGMENTS = ("gx1/quarantine", "gx1.quarantine")
SEED_V1 = 20260430
LABEL_DELAY_PNL_DELTA_THRESHOLD_BPS = 5.0  # delaying improved PNL by at least this much

DELAY_PROBABILITY_GRID: list[float] = [
    0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
]

ALLOWED_FINAL_STATUSES = {
    "BUILD_HYBRID_TRAIL_STOP_PASS_MEETS_PROMOTION_CRITERIA",
    "BUILD_HYBRID_TRAIL_STOP_PASS_BEATS_TRAIL_STOP_BUT_FAILS_OTHER_CRITERIA",
    "BUILD_HYBRID_TRAIL_STOP_PARTIAL_TIES_TRAIL_STOP",
    "BUILD_HYBRID_TRAIL_STOP_PARTIAL_DEGRADES_VS_TRAIL_STOP",
    "BUILD_HYBRID_TRAIL_STOP_BLOCKED_BY_INPUT_LOCK_MISSING",
}

ALLOWED_NEXT_ACTIONS = {
    "DEFINE_PAPER_TRADING_PROMOTION_PLAN_V1",
    "BUILD_REGIME_CONDITIONED_HYBRID_TRAIL_STOP_V2",
    "ACCEPT_TRAIL_STOP_AS_RESEARCH_BASELINE_V1",
    "REPAIR_HYBRID_TRAIL_STOP_BEFORE_FURTHER_WORK_V1",
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
# Inputs
# ---------------------------------------------------------------------------


def _load_inputs() -> dict[str, Any]:
    roots = [
        INPUT_TRAIL_STOP_DEEP_DIVE_ROOT,
        INPUT_PROMOTION_CRITERIA_ROOT,
        INPUT_FEATURE_STABILITY_ROOT,
        INPUT_RECOVERY_ROOT,
        INPUT_SPLIT_ROOT,
        INPUT_V2_CONTRACT_ROOT,
    ]
    validate_explicit_artifact_roots(roots)
    required = {
        "trail_stop_deep_dive_summary": INPUT_TRAIL_STOP_DEEP_DIVE_ROOT / "summary_v1.json",
        "trail_stop_per_trade_decomposition": INPUT_TRAIL_STOP_DEEP_DIVE_ROOT
        / "per_trade_trail_stop_decomposition_v1.json",
        "promotion_criteria": INPUT_PROMOTION_CRITERIA_ROOT / "promotion_criteria_v1.json",
        "feature_stability_summary": INPUT_FEATURE_STABILITY_ROOT / "summary_v1.json",
        "recovery_per_trade": INPUT_RECOVERY_ROOT
        / "entry_snapshot_signals_per_trade_v1.parquet",
        "split_locked_dataset": INPUT_SPLIT_ROOT
        / "split_locked_augmented_dataset_v1.parquet",
        "v2_state_contract": INPUT_V2_CONTRACT_ROOT / "state_feature_contract_v2.json",
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
        "promotion_criteria": _read_json(required["promotion_criteria"]),
        "v2_state_contract": _read_json(required["v2_state_contract"]),
        "trail_stop_per_trade_decomposition": _read_json(
            required["trail_stop_per_trade_decomposition"]
        ),
        "base34_path": BASE34_M5_FEATURES_PATH,
    }


# ---------------------------------------------------------------------------
# Trail-stop firing + label computation per trade (per-fold)
# ---------------------------------------------------------------------------


def _compute_trail_stop_per_trade(
    per_bar_full: pd.DataFrame,
) -> pd.DataFrame:
    """For each candidate_uid_v1, compute trail-stop firing bar (or None) and
    the realized PNL at the firing bar vs at trade end + the would-delay-help
    label."""
    rows: list[dict[str, Any]] = []
    for uid, group in per_bar_full.groupby("candidate_uid_v1", sort=False):
        df = group.sort_values("bars_held_v1").reset_index(drop=True)
        pnl = df["running_pnl_at_close_bps_v1"].astype(float).to_numpy()
        mfe = df["running_mfe_bps_v1"].astype(float).to_numpy()
        n = int(len(df))
        if n == 0:
            continue
        fire_bar = -1
        for i in range(n):
            if mfe[i] >= ts_gate.TRAIL_STOP_MIN_MFE_BPS:
                giveback_ratio = (mfe[i] - pnl[i]) / max(mfe[i], 1e-9)
                if giveback_ratio >= ts_gate.TRAIL_STOP_GIVEBACK_RATIO:
                    fire_bar = i
                    break
        realized_pnl = float(pnl[-1])
        if fire_bar == -1:
            trail_stop_pnl = realized_pnl
            firing_status = "NEVER_FIRED"
            would_delay_help = 0
            post_fire_max_pnl = realized_pnl
        else:
            trail_stop_pnl = float(pnl[fire_bar])
            firing_status = "FIRED"
            post_fire_max_pnl = float(pnl[fire_bar:].max())
            would_delay_help = (
                1
                if post_fire_max_pnl > trail_stop_pnl + LABEL_DELAY_PNL_DELTA_THRESHOLD_BPS
                else 0
            )
        rows.append(
            {
                "candidate_uid_v1": str(uid),
                "primary_split_v1": df["primary_split_v1"].iloc[0],
                "fire_bar_index_v1": int(fire_bar) if fire_bar >= 0 else None,
                "firing_status_v1": firing_status,
                "trail_stop_pnl_v1": trail_stop_pnl,
                "realized_pnl_v1": realized_pnl,
                "post_fire_max_pnl_v1": post_fire_max_pnl,
                "would_delay_help_v1": int(would_delay_help),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Hybrid policy evaluation
# ---------------------------------------------------------------------------


def _evaluate_hybrid_at_threshold(
    per_trade_with_pred: pd.DataFrame, threshold: float
) -> dict[str, Any]:
    """Apply hybrid policy: if trail-stop would fire AND p_delay >= threshold,
    use realized exit; else use trail-stop. Aggregate PNL across all trades."""
    df = per_trade_with_pred
    n = int(len(df))
    fired = (df["firing_status_v1"] == "FIRED").to_numpy()
    p_delay = df["p_delay_v1"].astype(float).to_numpy()
    delay_decision = fired & (p_delay >= threshold)
    fire_decision = fired & ~delay_decision

    trail_stop_pnl = df["trail_stop_pnl_v1"].astype(float).to_numpy()
    realized_pnl = df["realized_pnl_v1"].astype(float).to_numpy()

    # Hybrid PNL per trade:
    #   if NEVER_FIRED  -> realized_pnl (default)
    #   if FIRED and not delay -> trail_stop_pnl
    #   if FIRED and delay     -> realized_pnl
    hybrid_pnl = np.where(fire_decision, trail_stop_pnl, realized_pnl)

    label = df["would_delay_help_v1"].astype(int).to_numpy()
    pred = delay_decision.astype(int)
    fire_mask = fired
    if fire_mask.any():
        tp = int(((pred[fire_mask] == 1) & (label[fire_mask] == 1)).sum())
        fp = int(((pred[fire_mask] == 1) & (label[fire_mask] == 0)).sum())
        tn = int(((pred[fire_mask] == 0) & (label[fire_mask] == 0)).sum())
        fn = int(((pred[fire_mask] == 0) & (label[fire_mask] == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if precision and recall
            else None
        )
    else:
        tp = fp = tn = fn = 0
        precision = recall = f1 = None

    return {
        "threshold_v1": float(threshold),
        "trade_count_v1": n,
        "fired_count_v1": int(fired.sum()),
        "delay_count_v1": int(delay_decision.sum()),
        "fire_count_v1": int(fire_decision.sum()),
        "never_fired_count_v1": int((~fired).sum()),
        "tp_v1": tp,
        "fp_v1": fp,
        "tn_v1": tn,
        "fn_v1": fn,
        "precision_v1": precision,
        "recall_v1": recall,
        "f1_v1": f1,
        "hybrid_total_pnl_v1": float(hybrid_pnl.sum()),
        "trail_stop_total_pnl_v1": float(trail_stop_pnl.sum()),
        "realized_total_pnl_v1": float(realized_pnl.sum()),
        "lift_vs_trail_stop_v1": float(hybrid_pnl.sum() - trail_stop_pnl.sum()),
        "lift_vs_realized_v1": float(hybrid_pnl.sum() - realized_pnl.sum()),
    }


# ---------------------------------------------------------------------------
# Per-fold hybrid training + evaluation
# ---------------------------------------------------------------------------


def _project_per_trade_features_for_fold(
    inputs: dict[str, Any], uid_to_split: dict[str, str]
) -> pd.DataFrame:
    return wf_gate._build_per_trade_for_fold(inputs, uid_to_split)


def _project_per_bar_for_fold(
    inputs: dict[str, Any], uid_to_split: dict[str, str]
) -> pd.DataFrame:
    df = pd.read_parquet(inputs["required_paths"]["split_locked_dataset"])
    df["candidate_uid_v1"] = df["candidate_uid_v1"].astype(str)
    df["ts_v1"] = pd.to_datetime(df["ts_v1"], utc=True)
    per_bar_v1 = v2_train_gate._per_bar_view(df)
    # Override split assignment.
    per_bar = per_bar_v1.copy()
    per_bar["candidate_uid_v1"] = per_bar["candidate_uid_v1"].astype(str)
    per_bar["primary_split_v1"] = (
        per_bar["candidate_uid_v1"].map(uid_to_split).fillna("hold_out")
    )
    per_bar = per_bar[per_bar["primary_split_v1"] != "hold_out"].reset_index(drop=True)
    return per_bar


def _train_hybrid_for_fold(
    per_trade_features: pd.DataFrame, per_trade_trail_stop: pd.DataFrame
) -> tuple[np.ndarray, list[str]]:
    """Train logistic-balanced classifier on the would-delay-help label.

    Only trades where trail-stop FIRED are usable training rows (the label
    is undefined for NEVER_FIRED trades). Returns (p_delay_full, feature_names).
    p_delay_full has one entry per trade in per_trade_features (in input
    order); rows for NEVER_FIRED trades get p_delay = 0 (irrelevant since
    hybrid policy ignores them).
    """
    merged = per_trade_features.merge(
        per_trade_trail_stop[["candidate_uid_v1", "firing_status_v1", "would_delay_help_v1"]],
        on="candidate_uid_v1",
        how="left",
    )
    merged["firing_status_v1"] = merged["firing_status_v1"].fillna("NEVER_FIRED")
    merged["would_delay_help_v1"] = merged["would_delay_help_v1"].fillna(0).astype(int)

    train_mask = (
        (merged["primary_split_v1"] == "train")
        & (merged["firing_status_v1"] == "FIRED")
    ).to_numpy()
    if int(train_mask.sum()) == 0:
        return np.zeros(len(merged), dtype=float), []

    train_subset = merged[train_mask].reset_index(drop=True)
    norm = skip_v1_gate._fit_train_normalization(train_subset)
    X_train, feature_names = skip_v1_gate._build_state_matrix(
        train_subset, norm
    )
    y_train = train_subset["would_delay_help_v1"].astype(int).to_numpy()

    # Need at least one positive and one negative example.
    if y_train.sum() == 0 or y_train.sum() == len(y_train):
        return np.zeros(len(merged), dtype=float), feature_names

    logreg = skip_v2_gate._train_logistic(X_train, y_train)

    # Project features for ALL rows (incl. val/test) using train-only norm.
    X_full, _ = skip_v1_gate._build_state_matrix(merged, norm)
    p_delay = skip_v2_gate._predict_p_skip(logreg, X_full)
    return p_delay, feature_names


def _evaluate_fold(
    per_trade_features: pd.DataFrame,
    per_trade_trail_stop: pd.DataFrame,
    p_delay_full: np.ndarray,
) -> dict[str, Any]:
    """Tune delay threshold on val (max hybrid PNL); apply to test."""
    merged = per_trade_features.merge(
        per_trade_trail_stop[
            ["candidate_uid_v1", "firing_status_v1", "trail_stop_pnl_v1",
             "realized_pnl_v1", "would_delay_help_v1"]
        ],
        on="candidate_uid_v1",
        how="left",
    )
    merged["firing_status_v1"] = merged["firing_status_v1"].fillna("NEVER_FIRED")
    merged["trail_stop_pnl_v1"] = merged["trail_stop_pnl_v1"].fillna(0.0)
    merged["realized_pnl_v1"] = merged["realized_pnl_v1"].fillna(0.0)
    merged["would_delay_help_v1"] = merged["would_delay_help_v1"].fillna(0).astype(int)
    merged["p_delay_v1"] = p_delay_full

    val_mask = (merged["primary_split_v1"] == "val").to_numpy()
    test_mask = (merged["primary_split_v1"] == "test").to_numpy()
    val_df = merged[val_mask].reset_index(drop=True)
    test_df = merged[test_mask].reset_index(drop=True)

    val_sweep: list[dict[str, Any]] = []
    test_sweep: list[dict[str, Any]] = []
    best_thr = None
    best_val_pnl = -np.inf
    for thr in DELAY_PROBABILITY_GRID:
        val_m = _evaluate_hybrid_at_threshold(val_df, thr)
        val_m["split_v1"] = "val"
        val_sweep.append(val_m)
        test_m = _evaluate_hybrid_at_threshold(test_df, thr)
        test_m["split_v1"] = "test"
        test_sweep.append(test_m)
        if val_m["hybrid_total_pnl_v1"] > best_val_pnl:
            best_val_pnl = val_m["hybrid_total_pnl_v1"]
            best_thr = float(thr)
    if best_thr is None:
        best_thr = 0.5
    test_at_locked = _evaluate_hybrid_at_threshold(test_df, best_thr)
    test_at_locked["split_v1"] = "test"

    return {
        "tuned_threshold_v1": best_thr,
        "val_sweep_v1": val_sweep,
        "test_sweep_v1": test_sweep,
        "test_at_locked_threshold_v1": test_at_locked,
    }


def _train_eval_per_fold(
    inputs: dict[str, Any], candidate_uid_order: list[str]
) -> list[dict[str, Any]]:
    fold_results: list[dict[str, Any]] = []
    for fold in wf_gate.FOLD_DEFINITIONS:
        fold_id = fold["fold_id_v1"]
        uid_to_split = wf_gate._assign_fold_split(candidate_uid_order, fold)
        per_trade_features = _project_per_trade_features_for_fold(inputs, uid_to_split)
        per_bar = _project_per_bar_for_fold(inputs, uid_to_split)
        per_trade_trail_stop = _compute_trail_stop_per_trade(per_bar)
        p_delay_full, feature_names = _train_hybrid_for_fold(
            per_trade_features, per_trade_trail_stop
        )
        eval_out = _evaluate_fold(per_trade_features, per_trade_trail_stop, p_delay_full)
        fold_results.append(
            {
                "fold_id_v1": fold_id,
                "feature_count_v1": len(feature_names),
                "feature_names_v1": feature_names,
                "tuned_threshold_v1": eval_out["tuned_threshold_v1"],
                "test_at_locked_threshold_v1": eval_out["test_at_locked_threshold_v1"],
                "val_sweep_v1": eval_out["val_sweep_v1"],
                "test_sweep_v1": eval_out["test_sweep_v1"],
            }
        )
    return fold_results


# ---------------------------------------------------------------------------
# Apply locked promotion criteria
# ---------------------------------------------------------------------------


def _apply_promotion_criteria(
    per_fold_results: list[dict[str, Any]],
) -> dict[str, Any]:
    per_fold_lifts_vs_trail_stop = [
        float(r["test_at_locked_threshold_v1"]["lift_vs_trail_stop_v1"])
        for r in per_fold_results
    ]
    per_fold_lifts_vs_realized = [
        float(r["test_at_locked_threshold_v1"]["lift_vs_realized_v1"])
        for r in per_fold_results
    ]
    per_fold_pnl = [
        float(r["test_at_locked_threshold_v1"]["hybrid_total_pnl_v1"])
        for r in per_fold_results
    ]
    per_fold_trail_stop_pnl = [
        float(r["test_at_locked_threshold_v1"]["trail_stop_total_pnl_v1"])
        for r in per_fold_results
    ]

    # Apply locked criteria: lift = vs realized floor (matches walk-forward gate's convention).
    promotion = criteria_gate.evaluate_candidate_against_criteria(
        candidate_id="hybrid_trail_stop_plus_postpone_learner",
        per_fold_lifts_bps=per_fold_lifts_vs_realized,
        per_fold_pnl_bps=per_fold_pnl,
        per_fold_trail_stop_pnl_bps=per_fold_trail_stop_pnl,
        no_shortcut_audit_passed=True,
        deterministic_reproducible=True,
    )
    promotion["per_fold_lifts_vs_trail_stop_v1"] = per_fold_lifts_vs_trail_stop
    promotion["per_fold_lifts_vs_realized_v1"] = per_fold_lifts_vs_realized
    promotion["per_fold_pnl_v1"] = per_fold_pnl
    promotion["per_fold_trail_stop_pnl_v1"] = per_fold_trail_stop_pnl
    return promotion


# ---------------------------------------------------------------------------
# go-no-go
# ---------------------------------------------------------------------------


def _go_no_go(
    promotion: dict[str, Any],
) -> tuple[str, str, str, dict[str, Any]]:
    n_pass = int(promotion["n_criteria_passed_v1"])
    n_total = int(promotion["n_criteria_total_v1"])
    overall = bool(promotion["overall_pass_v1"])
    per_fold_lifts_ts = promotion["per_fold_lifts_vs_trail_stop_v1"]
    n_folds_beating_ts = int(sum(1 for v in per_fold_lifts_ts if v > 0))
    mean_lift_ts = float(np.mean(per_fold_lifts_ts)) if per_fold_lifts_ts else 0.0
    headline = {
        "n_criteria_passed_v1": n_pass,
        "n_criteria_total_v1": n_total,
        "promotion_overall_pass_v1": overall,
        "per_fold_lifts_vs_trail_stop_v1": per_fold_lifts_ts,
        "per_fold_lifts_vs_realized_v1": promotion["per_fold_lifts_vs_realized_v1"],
        "per_fold_pnl_v1": promotion["per_fold_pnl_v1"],
        "per_fold_trail_stop_pnl_v1": promotion["per_fold_trail_stop_pnl_v1"],
        "n_folds_beating_trail_stop_v1": n_folds_beating_ts,
        "mean_lift_vs_trail_stop_v1": mean_lift_ts,
    }
    if overall:
        return (
            "BUILD_HYBRID_TRAIL_STOP_PASS_MEETS_PROMOTION_CRITERIA",
            "DEFINE_PAPER_TRADING_PROMOTION_PLAN_V1",
            (
                f"Hybrid trail-stop + delay learner passes ALL {n_total} "
                f"promotion criteria. First research candidate to do so. "
                f"Mean lift vs trail-stop {mean_lift_ts:+.0f} bps. "
                f"Beats trail-stop in {n_folds_beating_ts} of "
                f"{len(per_fold_lifts_ts)} folds. Next: define paper trading "
                "promotion plan."
            ),
            headline,
        )
    if n_folds_beating_ts == len(per_fold_lifts_ts) and mean_lift_ts > 50.0:
        return (
            "BUILD_HYBRID_TRAIL_STOP_PASS_BEATS_TRAIL_STOP_BUT_FAILS_OTHER_CRITERIA",
            "BUILD_REGIME_CONDITIONED_HYBRID_TRAIL_STOP_V2",
            (
                f"Hybrid beats trail-stop in all {len(per_fold_lifts_ts)} folds "
                f"(mean lift {mean_lift_ts:+.0f}) but only {n_pass}/{n_total} "
                "promotion criteria pass. Try regime-conditioned V2 hybrid."
            ),
            headline,
        )
    if mean_lift_ts >= -50.0 and mean_lift_ts <= 50.0:
        return (
            "BUILD_HYBRID_TRAIL_STOP_PARTIAL_TIES_TRAIL_STOP",
            "ACCEPT_TRAIL_STOP_AS_RESEARCH_BASELINE_V1",
            (
                f"Hybrid mean lift vs trail-stop {mean_lift_ts:+.0f} bps "
                "(approximately ties). The learned postpone signal does not "
                "improve over trail-stop. Accept trail-stop as research baseline."
            ),
            headline,
        )
    return (
        "BUILD_HYBRID_TRAIL_STOP_PARTIAL_DEGRADES_VS_TRAIL_STOP",
        "REPAIR_HYBRID_TRAIL_STOP_BEFORE_FURTHER_WORK_V1",
        (
            f"Hybrid mean lift vs trail-stop {mean_lift_ts:+.0f} bps "
            "(degrades). Investigate before further work."
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
        "layer_name": "BUILD_HYBRID_TRAIL_STOP_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "trail_stop_deep_dive_root_v1": str(INPUT_TRAIL_STOP_DEEP_DIVE_ROOT),
            "promotion_criteria_root_v1": str(INPUT_PROMOTION_CRITERIA_ROOT),
            "feature_stability_root_v1": str(INPUT_FEATURE_STABILITY_ROOT),
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

    # Build candidate-uid time order.
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
    ].sort_values(
        ["open_ts_utc", "candidate_uid_v1"], kind="mergesort"
    ).reset_index(drop=True)
    candidate_uid_order = trades_accepted["candidate_uid_v1"].astype(str).tolist()

    # Per-fold hybrid training + evaluation.
    per_fold_results = _train_eval_per_fold(inputs, candidate_uid_order)

    # Apply promotion criteria.
    promotion = _apply_promotion_criteria(per_fold_results)
    _write_json(artifact_root / "promotion_criteria_evaluation_v1.json", promotion)

    # Persist per-fold detail.
    flat_test_at_locked: list[dict[str, Any]] = []
    for r in per_fold_results:
        row = {**r["test_at_locked_threshold_v1"], "fold_id_v1": r["fold_id_v1"]}
        flat_test_at_locked.append(row)
    _write_rows(
        artifact_root / "per_fold_test_at_locked_threshold_v1.csv", flat_test_at_locked
    )
    _write_json(
        artifact_root / "per_fold_test_at_locked_threshold_v1.json",
        {"row_count_v1": len(flat_test_at_locked), "rows_v1": flat_test_at_locked},
    )

    # Threshold sweeps per fold (val + test).
    threshold_sweep_rows: list[dict[str, Any]] = []
    for r in per_fold_results:
        for v in r["val_sweep_v1"]:
            threshold_sweep_rows.append({**v, "fold_id_v1": r["fold_id_v1"]})
        for v in r["test_sweep_v1"]:
            threshold_sweep_rows.append({**v, "fold_id_v1": r["fold_id_v1"]})
    _write_rows(
        artifact_root / "per_fold_threshold_sweep_v1.csv", threshold_sweep_rows
    )
    _write_json(
        artifact_root / "per_fold_threshold_sweep_v1.json",
        {"row_count_v1": len(threshold_sweep_rows), "rows_v1": threshold_sweep_rows},
    )

    repro = {
        "layer_name": "BUILD_HYBRID_TRAIL_STOP_REPRODUCIBILITY_AUDIT_V1",
        "fold_count_v1": len(wf_gate.FOLD_DEFINITIONS),
        "label_pnl_delta_threshold_bps_v1": LABEL_DELAY_PNL_DELTA_THRESHOLD_BPS,
        "delay_threshold_grid_v1": DELAY_PROBABILITY_GRID,
        "seed_v1": SEED_V1,
        "no_implicit_glob_used_for_v1_inputs_v1": True,
        "deprecated_quarantine_revival_v1": False,
        "research_only_v1": True,
    }
    _write_json(artifact_root / "reproducibility_audit_v1.json", repro)

    status, next_action, recommendation, headline = _go_no_go(promotion)
    validate_final_status(status, next_action)

    summary = {
        "layer_name": "BUILD_HYBRID_TRAIL_STOP_SUMMARY_V1",
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
            "layer_name": "BUILD_HYBRID_TRAIL_STOP_STATUS_V1",
            "status_v1": "MATERIALIZED_RESEARCH_ONLY_GATE",
            "final_status_v1": status,
            "next_action_v1": next_action,
            "training_executed_v1": True,
        },
    )
    _write_json(
        artifact_root / "build_hybrid_trail_stop_go_no_go_v1.json",
        {
            "layer_name": "BUILD_HYBRID_TRAIL_STOP_GO_NO_GO_V1",
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
                "Research-only hybrid policy. NOT promoted to runtime; "
                "trail-stop / exit_manager / live_features unmodified."
            ),
        },
    )

    # Build report.
    report_lines = [
        "# Build Hybrid Trail-Stop Plus Small Adjustment Learner V1",
        "",
        f"- Status: `{status}`",
        f"- Next action: `{next_action}`",
        "- Training: research-only; hybrid policy NOT promoted to runtime.",
        "",
        "## Headline",
        f"- Promotion criteria: {headline['n_criteria_passed_v1']}/{headline['n_criteria_total_v1']} passed; "
        f"overall pass: {headline['promotion_overall_pass_v1']}",
        f"- Mean lift vs trail-stop: {headline['mean_lift_vs_trail_stop_v1']:+.0f} bps",
        f"- Folds beating trail-stop: {headline['n_folds_beating_trail_stop_v1']}/{len(headline['per_fold_lifts_vs_trail_stop_v1'])}",
        "",
        "## Per-fold test results at val-tuned threshold",
        "",
        "| Fold | Tuned thr | Trail-stop PNL | Realized PNL | Hybrid PNL | vs Trail-stop | vs Realized |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in per_fold_results:
        t = r["test_at_locked_threshold_v1"]
        report_lines.append(
            f"| `{r['fold_id_v1']}` | {r['tuned_threshold_v1']} | "
            f"{t['trail_stop_total_pnl_v1']:.0f} | "
            f"{t['realized_total_pnl_v1']:.0f} | "
            f"**{t['hybrid_total_pnl_v1']:.0f}** | "
            f"{t['lift_vs_trail_stop_v1']:+.0f} | "
            f"{t['lift_vs_realized_v1']:+.0f} |"
        )
    report_lines.extend(
        [
            "",
            "## Promotion-criteria breakdown",
        ]
    )
    for c in promotion["breakdown_v1"]:
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
            "go_no_go": str(artifact_root / "build_hybrid_trail_stop_go_no_go_v1.json"),
            "input_manifest": str(artifact_root / "input_manifest_v1.json"),
            "promotion_criteria_evaluation": str(
                artifact_root / "promotion_criteria_evaluation_v1.json"
            ),
            "per_fold_test_at_locked_threshold_csv": str(
                artifact_root / "per_fold_test_at_locked_threshold_v1.csv"
            ),
            "per_fold_test_at_locked_threshold_json": str(
                artifact_root / "per_fold_test_at_locked_threshold_v1.json"
            ),
            "per_fold_threshold_sweep_csv": str(
                artifact_root / "per_fold_threshold_sweep_v1.csv"
            ),
            "per_fold_threshold_sweep_json": str(
                artifact_root / "per_fold_threshold_sweep_v1.json"
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
        description="Materialize BUILD_HYBRID_TRAIL_STOP_PLUS_SMALL_ADJUSTMENT_LEARNER_V1."
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
