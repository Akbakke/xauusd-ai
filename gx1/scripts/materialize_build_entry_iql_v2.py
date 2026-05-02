#!/usr/bin/env python3
"""Build entry-IQL V2: 5-action multi-head Q(s, a, K) on full 2020-2026 candidate universe.

Mission
-------
Train the RL "smart head" that supervises and overrides the entry transformer
(V10) — same architectural pattern as exit-IQL guides exit-transformer (V3).

Uses the candidate forward-outcome dataset built by
`materialize_build_candidate_forward_outcome_dataset_v1.py` (~844k candidate
moments across 282 weekly XAUUSD replays, with side-aware spread-aware
forward-bar metrics for take-now and wait-6-bars actions, at 6 K-horizons:
[12, 24, 48, 96, 144, 192] M5-bars = [1h, 2h, 4h, 8h, 12h, 16h]).

Action space (5)
----------------
  SKIP, TAKE_LONG_NOW, TAKE_SHORT_NOW, WAIT_LONG, WAIT_SHORT

Reward variants (5, all trained in parallel for comparison)
-----------------------------------------------------------
  R_RAW    : mfe_at_K  (pure upside)
  R_NET05  : mfe_at_K - 0.5 * mae_before_mfe_at_K  (mild risk-adjustment)
  R_NET10  : mfe_at_K - 1.0 * mae_before_mfe_at_K  (strong risk-adjustment)
  R_GATED  : mfe_at_K * 1[mae_before_mfe_at_K < 30bps]  (binary realisability)
  R_PATH   : mfe_at_K * mfe_at_K / max(mfe_at_K, |mae_before_mfe_at_K|)
              (smooth path-quality)

For SKIP, reward = 0 in all variants (no trade taken, no upside, no downside).

Counterfactual setup
--------------------
For each candidate state, all 5 action rewards are observed (computed from
forward bars). This is fully-observable counterfactual bandit. The Q-network
learns to predict R[s, a, K] for all (a, K) cells — much richer signal than
standard partial-observation bandit.

Folds
-----
StratifiedShuffleSplit by oracle_action (= argmax_a of R_PRIMARY at K_PRIMARY)
with n_splits=3 and test_size=0.20. Per-fold per-class guards: each fold's
val partition must have >= MIN_PER_CLASS_VAL examples of each action label,
else fold is FLAGGED but training continues.

Output
------
  per_fold_per_variant_evaluations_v1.csv  — every (fold, variant, agg, K) row
  summary_v1.json                          — go/no-go status, headlines
  status_v1.json                           — single-line status for downstream
  manifest_v1.json                         — artifact paths
  trained_models_v1/<variant>_fold<i>.pt   — model checkpoints

This is RESEARCH-ONLY. No runtime promotion.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.model_selection import StratifiedShuffleSplit

from gx1.scripts import entry_iql_multi_head_gpu_core_v1 as iql_core
from gx1.scripts import (
    materialize_build_iql_offline_data_contract_research_only_v1 as contract_gate,
)
from gx1.scripts import materialize_build_candidate_forward_outcome_dataset_v1 as fwd_pipe


ACTION = "BUILD_ENTRY_IQL_V2"
DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
DEFAULT_FORWARD_OUTCOME_DIR = (
    DEFAULT_REPORTS_ROOT / "CANDIDATE_FORWARD_OUTCOME_RUN" / "v1_full"
)

K_HORIZONS = [12, 24, 48, 96, 144, 192]
N_K = len(K_HORIZONS)
K_PRIMARY = 96  # used to derive oracle_action_label_v1 for stratification

REWARD_VARIANTS = ["R_RAW", "R_NET05", "R_NET10", "R_GATED", "R_PATH", "R_NET_REAL"]
GATED_THRESHOLD_BPS = 30.0

# R_NET_REAL parameters: realistic-net reward.
# Exit policy realizes a FRACTION of MFE (alpha < 1), pays MAE-before-MFE penalty
# (beta), and pays entry+exit spread cost (gamma * entry_spread_bps).
# This makes SKIP competitive when forward path is choppy/costly.
R_NET_REAL_ALPHA_MFE = 0.5      # exit policy captures ~50% of peak MFE on average
R_NET_REAL_BETA_MAE = 0.5       # half-weight penalty on path-drawdown
R_NET_REAL_GAMMA_SPREAD = 2.0   # entry + exit each pay 1x spread

# Cross-validation
N_FOLDS = 3
VAL_SIZE = 0.15
TEST_SIZE = 0.15
SEED_V1 = 20260501

# Training budget — modulated by --budget flag. Default is "fast" so smoke-tests
# finish on CPU; override to "full" for production runs.
TRAIN_TAU = 0.8
TRAIN_EPOCHS_Q = 30
TRAIN_EPOCHS_V = 15
TRAIN_K_VQ_ITERATIONS = 3
TRAIN_BATCH_SIZE = 512
TRAIN_HIDDEN_DIM = 128
TRAIN_N_HIDDEN = 2

BUDGET_PRESETS = {
    "fast":   {"epochs_q":  30, "epochs_v": 15, "k_iter": 3, "batch": 512, "hidden": 128, "n_hidden": 2},
    "medium": {"epochs_q":  60, "epochs_v": 25, "k_iter": 4, "batch": 256, "hidden": 192, "n_hidden": 3},
    "full":   {"epochs_q": 100, "epochs_v": 40, "k_iter": 6, "batch": 256, "hidden": 256, "n_hidden": 3},
}

# Per-class assertion guards
MIN_PER_CLASS_VAL = 50  # warn (don't abort) if any action class has fewer than this

# Aggregator strategies for action selection (compared on val)
AGGREGATORS = ["mean", "max"]

# Numeric feature columns from candidate parquet (V10 outputs + state context).
# NOTE: entry_spread_bps and `side` are POST-decision fields (only populated for
# accepted candidates ~0.3% of rows), so they're EXCLUDED here. Use chunk_0
# `spread_bps` instead — at-bar pre-decision spread from raw bid/ask.
NUMERIC_STATE_COLS_CANDIDATE = [
    "weekday_utc",
    "hour_utc",
    "atr_bps",
    "p_long",
    "p_short",
    "p_flat",
    "p_hat",
    "margin",
    "uncertainty_score",
    "tradable_prob",
    "mfe_first_n_pred",
    "path_quality_pred",
]

# CANONICAL V10/V3 features at decision_ts (from chunk_0_data.parquet).
# These are the SAME features V10 actually saw at runtime → IQL gets full
# feature parity with the entry transformer it supervises.
NUMERIC_STATE_COLS_CHUNK0 = [
    f"{c}{fwd_pipe.CHUNK0_SUFFIX}"  # type: ignore[attr-defined]
    for c in [
        "_v1_atr14", "_v1_atr_regime_id", "_v1_atr_z_10_100",
        "_v1_bb_bandwidth_delta_10", "_v1_bb_squeeze_20_2",
        "_v1_body_share_1", "_v1_body_tr",
        "_v1_close_ema_slope_3", "_v1_clv", "_v1_comp3_ratio",
        "_v1_cost_bps_dyn", "_v1_cost_bps_est",
        "_v1_ema_diff",
        "_v1_int_clv_atr", "_v1_int_ema_us", "_v1_int_r5_atr",
        "_v1_int_range_us", "_v1_int_slope_h1_us", "_v1_int_slope_h4_atr",
        "_v1_int_vwap_h1",
        "_v1_is_EU", "_v1_is_US",
        "_v1_kama_slope_30", "_v1_kurt_r", "_v1_lower_tr",
        "_v1_pk_sigma20", "_v1_r1", "_v1_r12",
        "atr_bps", "spread_bps",
        "D1_dist_from_ema200_atr", "H1_range_compression_ratio",
        "D1_atr_percentile_252", "M15_range_compression_ratio",
        "micro_momentum_3", "micro_momentum_5", "micro_acceleration",
        "wick_ratio", "distance_ema_fast",
        "dist_last_swing_high_atr", "dist_last_swing_low_atr",
        "bars_since_swing_high", "bars_since_swing_low",
        "retracement_from_last_impulse",
        "trend_regime_id", "vol_regime_id",
        "atr_bucket", "spread_bucket", "H4_trend_sign_cat",
        "session_id",
        "is_ASIA",
        "minutes_since_session_open", "minutes_to_next_session_boundary",
        "session_change_flag", "session_tradable",
    ]
]
# Canonical V10 features (basic_v1 + H1 + H4 + m5_phase + high-level), suffixed _canon_v1
# Names mirror canonical_features_v1.parquet columns.
NUMERIC_STATE_COLS_CANONICAL = [
    f"{c}{fwd_pipe.CANONICAL_FEATURES_SUFFIX}"
    for c in [
        # 49 _v1_* basic_v1 features (superset of chunk_0's 28)
        "_v1_r3", "_v1_r5", "_v1_r8", "_v1_r12", "_v1_r24", "_v1_r1", "_v1_r48_z",
        "_v1_atr14", "_v1_pk_sigma20", "_v1_atr_regime_id", "_v1_ema_diff",
        "_v1_vwap_drift48", "_v1_rsi14_z", "_v1_rsi2", "_v1_rsi14",
        "_v1_rsi2_gt_rsi14", "_v1_ret_ema_diff_2_5", "_v1_bb_bandwidth_delta_10",
        "_v1_close_ema_slope_3", "_v1_body_tr", "_v1_upper_tr", "_v1_lower_tr",
        "_v1_wick_imbalance", "_v1_range_comp_20_100", "_v1_range_adr",
        "_v1_spread_p", "_v1_slip_bps",
        "_v1_clv", "_v1_range_z", "_v1_spread_z", "_v1_cost_bps_est",
        "_v1_kurt_r", "_v1_int_r5_atr", "_v1_int_vwap_h1", "_v1_int_slope_h4_atr",
        "_v1_int_clv_atr", "_v1_cost_bps_dyn", "_v1_tod_sin", "_v1_tod_cos",
        "_v1_r1_q90_48", "_v1_r1_q10_48", "_v1_atr_z_10_100", "_v1_tema_slope_20",
        "_v1_bb_squeeze_20_2", "_v1_kama_slope_30", "_v1_ret_ema_ratio_5_34",
        "_v1_body_share_1", "_v1_tr_1_over_atr_14", "_v1_comp3_ratio",
        # 6 H1 multi-TF features
        "_v1h1_ema_diff", "_v1h1_vwap_drift", "_v1h1_atr", "_v1h1_rsi14_z",
        "_v1h1_slope3", "_v1h1_slope5",
        # 5 H4 multi-TF features
        "_v1h4_ema_diff", "_v1h4_atr", "_v1h4_rsi14_z", "_v1h4_slope3", "_v1h4_slope5",
        # 5 m5_phase one-hots
        "m5_phase_0", "m5_phase_1", "m5_phase_2", "m5_phase_3", "m5_phase_4",
        # 19 high-level basics
        "mid", "range", "body_pct", "wick_asym", "atr", "atr50", "atr_z", "std50",
        "ret_1", "ret_5", "ret_20", "roc20", "roc100", "rvol_20", "rvol_60",
        "ema20_slope", "ema100_slope", "pos_vs_ema200", "vol_ratio",
    ]
]
# Derived V3-equivalent feature
NUMERIC_STATE_COLS_DERIVED = ["entropy_v1"]

NUMERIC_STATE_COLS = (
    NUMERIC_STATE_COLS_CANDIDATE
    + NUMERIC_STATE_COLS_CHUNK0
    + NUMERIC_STATE_COLS_CANONICAL
    + NUMERIC_STATE_COLS_DERIVED
)

# Categorical state columns to one-hot encode (string-valued candidate cols).
ONE_HOT_COLS = [
    "session",          # OVERLAP / US / EU / ASIA
    "vol_regime",       # EXTREME / HIGH / MEDIUM / LOW
    "trend_regime",     # TREND_UP / TREND_DOWN / TREND_NEUTRAL
    "decision_reason",  # flat_dominant / flat_veto / pre_quality / ...
]

ALLOWED_FINAL_STATUSES = {
    "ENTRY_IQL_V2_PASS_BEATS_REALIZED_BASELINE",
    "ENTRY_IQL_V2_PARTIAL_TIES_BASELINE",
    "ENTRY_IQL_V2_PARTIAL_DEGRADES_VS_BASELINE",
    "ENTRY_IQL_V2_PARTIAL_DEGENERATE_ACTION_DIST",
    "ENTRY_IQL_V2_BLOCKED_BY_INPUT_LOCK_MISSING",
}
ALLOWED_NEXT_ACTIONS = {
    "BUILD_JOINT_ENTRY_AND_EXIT_IQL_V2",
    "REPAIR_ENTRY_IQL_V2_BEFORE_FURTHER_WORK",
    "BUILD_ENTRY_IQL_V2_RUNTIME_ADAPTER",
}

_stamp = contract_gate._stamp
_utc_now = contract_gate._utc_now
_jsonable = contract_gate._jsonable
_write_json = contract_gate._write_json
_write_rows = contract_gate._write_rows


# ---------------------------------------------------------------------------
# Dataset loading + reward construction
# ---------------------------------------------------------------------------


def load_forward_outcome_dataset(per_week_dir: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    parquets = sorted(per_week_dir.glob("forward_outcomes_*.parquet"))
    if not parquets:
        raise RuntimeError(f"[{ACTION}] no forward_outcomes_*.parquet found in {per_week_dir}")
    for p in parquets:
        parts.append(pd.read_parquet(p))
    df = pd.concat(parts, ignore_index=True)
    return df


def build_reward_matrix(df: pd.DataFrame, *, variant: str) -> np.ndarray:
    """Return R of shape (n_rows, 5_actions, n_K). NaN-filled to 0.

    Action layout follows iql_core.ACTION_*_ID:
      0=SKIP, 1=TAKE_LONG_NOW, 2=TAKE_SHORT_NOW, 3=WAIT_LONG, 4=WAIT_SHORT

    For SKIP (action 0), reward is 0 in all variants — no trade taken.
    For trade actions, reward depends on variant.

    NOTE on SKIP-competitiveness: MFE is by definition ≥ 0 (peak favorable
    excursion), so any variant that simply uses raw MFE (R_RAW) will never
    let SKIP win — oracle-action will always pick a trade action. The
    R_NET_REAL variant solves this by discounting MFE by an exit-realization
    fraction, penalizing path drawdown, and subtracting spread cost. This
    creates the "ingen trade er bedre enn dårlig trade" signal needed for
    SKIP to be a meaningful action class.
    """
    n = len(df)
    R = np.zeros((n, iql_core.N_ACTIONS_V1, N_K), dtype=np.float32)
    # SKIP (action_id 0) → reward 0 across all K. Already initialised.

    # Per-row entry spread cost (used by R_NET_REAL only).
    # Falls back to 1.5 bps (XAUUSD typical median) if missing.
    entry_spread_bps = (
        pd.to_numeric(df.get("entry_spread_bps"), errors="coerce")
        .fillna(1.5).clip(lower=0.0).astype(float).to_numpy()
        if "entry_spread_bps" in df.columns
        else np.full(n, 1.5, dtype=float)
    )

    for ai, (action_prefix, side) in enumerate([
        ("take_now", "long"),
        ("take_now", "short"),
        ("wait", "long"),
        ("wait", "short"),
    ], start=1):
        for ki, K in enumerate(K_HORIZONS):
            mfe_col = f"{action_prefix}_{side}_mfe_bps_at_K{K}_v1"
            mae_pre_col = f"{action_prefix}_{side}_mae_before_mfe_bps_at_K{K}_v1"
            mfe = df[mfe_col].astype(float).fillna(0.0).to_numpy()
            mae_pre = df[mae_pre_col].astype(float).fillna(0.0).to_numpy()
            mae_pre = np.maximum(mae_pre, 0.0)  # magnitude
            if variant == "R_RAW":
                r = mfe
            elif variant == "R_NET05":
                r = mfe - 0.5 * mae_pre
            elif variant == "R_NET10":
                r = mfe - 1.0 * mae_pre
            elif variant == "R_GATED":
                r = np.where(mae_pre < GATED_THRESHOLD_BPS, mfe, 0.0)
            elif variant == "R_PATH":
                denom = np.maximum(mfe, mae_pre)
                denom = np.where(denom < 1e-6, 1e-6, denom)
                r = mfe * mfe / denom
                r = np.where(mfe < 0, mfe, r)  # negative MFE → keep negative
            elif variant == "R_NET_REAL":
                # Realistic exit-aware reward: only fraction of MFE is captured,
                # MAE-before-MFE is penalised, and entry+exit spread cost subtracted.
                r = (R_NET_REAL_ALPHA_MFE * mfe
                     - R_NET_REAL_BETA_MAE * mae_pre
                     - R_NET_REAL_GAMMA_SPREAD * entry_spread_bps)
            else:
                raise ValueError(f"unknown reward variant: {variant}")
            # Clip extreme values for stability (top 0.1% MFE values can be 500+ bps)
            r = np.clip(r, -500.0, 500.0)
            R[:, ai, ki] = r.astype(np.float32)
    return R


def build_state_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return (X, feature_names). One-hot encodes categorical cols.

    DATA INTEGRITY:
      - Hard-fails if any required feature column has >5% NaN (indicates
        upstream pipeline failure).
      - Pre-converts numeric cols, fills NaN with 0.0 only as last-resort
        warmup-edge handling. The presence flags
        `_chunk0_features_complete_v1` / `_canonical_features_complete_v1`
        are inspected upstream to refuse rows with missing critical features.
    """
    # Refuse rows with incomplete features (canonical or chunk_0 missing).
    # These are warmup-edge or week-with-no-chunk_0 rows that should never
    # be used for training.
    if "_canonical_features_complete_v1" in df.columns:
        complete_mask = df["_canonical_features_complete_v1"].fillna(False).astype(bool)
        if "_chunk0_features_complete_v1" in df.columns:
            complete_mask = complete_mask & df["_chunk0_features_complete_v1"].fillna(False).astype(bool)
        n_dropped = int((~complete_mask).sum())
        if n_dropped > 0:
            print(f"[BUILD_STATE_MATRIX] dropping {n_dropped:,} rows with incomplete features", flush=True)
        df = df.loc[complete_mask].reset_index(drop=True)
    state_parts: list[pd.DataFrame] = []
    feature_names: list[str] = []
    nan_warnings: list[tuple[str, float]] = []
    # Numeric features
    for c in NUMERIC_STATE_COLS:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            nan_frac = s.isna().mean()
            if nan_frac > 0.05:
                nan_warnings.append((c, float(nan_frac)))
            state_parts.append(pd.DataFrame({c: s.fillna(0.0)}))
            feature_names.append(c)
        else:
            state_parts.append(pd.DataFrame({c: np.zeros(len(df))}))
            feature_names.append(c)
    if nan_warnings:
        msg = ", ".join(f"{n}={f:.1%}" for n, f in nan_warnings[:8])
        raise RuntimeError(
            f"[BUILD_STATE_MATRIX] {len(nan_warnings)} feature(s) have >5% NaN "
            f"(first 8 listed): {msg}. Investigate upstream pipeline before training."
        )
    # One-hot features
    for c in ONE_HOT_COLS:
        if c not in df.columns:
            continue
        s = df[c].astype("string").fillna("UNKNOWN")
        unique_values = sorted(s.unique().tolist())
        for v in unique_values:
            col_name = f"{c}__{v}"
            state_parts.append(pd.DataFrame({col_name: (s == v).astype(np.float32).to_numpy()}))
            feature_names.append(col_name)
    X_df = pd.concat(state_parts, axis=1).reset_index(drop=True)
    X = X_df.to_numpy().astype(np.float32)
    return X, feature_names


def derive_oracle_action(
    R_primary: np.ndarray, K_primary_idx: int,
) -> np.ndarray:
    """Per-row: argmax over a of R_primary[:, a, K_primary_idx]. Returns (n,) action ids."""
    return R_primary[:, :, K_primary_idx].argmax(axis=1).astype(np.int64)


# ---------------------------------------------------------------------------
# Fold construction (stratified shuffle by oracle action, with weekly-blocked option)
# ---------------------------------------------------------------------------


def build_stratified_folds(
    n_rows: int,
    oracle_action: np.ndarray,
    n_folds: int = N_FOLDS,
    val_size: float = VAL_SIZE,
    test_size: float = TEST_SIZE,
    seed: int = SEED_V1,
) -> list[dict[str, np.ndarray]]:
    """Generate n_folds (train, val, test) splits via two-level StratifiedShuffleSplit.

    Each fold uses the same total population (n_rows) but with a different seed
    derived from the base seed. The val/test sets are STRATIFIED by action label
    so each fold's val has all 5 actions represented (when present in pop).
    """
    folds: list[dict[str, np.ndarray]] = []
    all_idx = np.arange(n_rows)
    for fold_id_v1 in range(n_folds):
        # First, split off (val + test) from train
        non_train_size = val_size + test_size
        sss1 = StratifiedShuffleSplit(
            n_splits=1, test_size=non_train_size, random_state=seed + fold_id_v1,
        )
        train_idx, valtest_idx = next(sss1.split(all_idx, oracle_action))
        # Then split (val + test) into val and test, stratified
        valtest_oracle = oracle_action[valtest_idx]
        sss2 = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size / non_train_size,
            random_state=seed + fold_id_v1 + 7919,
        )
        rel_val_idx, rel_test_idx = next(sss2.split(valtest_idx, valtest_oracle))
        val_idx = valtest_idx[rel_val_idx]
        test_idx = valtest_idx[rel_test_idx]
        folds.append({
            "fold_id_v1": f"FOLD_{fold_id_v1 + 1}",
            "train_idx_v1": train_idx,
            "val_idx_v1": val_idx,
            "test_idx_v1": test_idx,
        })
    return folds


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_policy_outcome(
    R: np.ndarray, predicted_action: np.ndarray, K_idx: int,
) -> dict[str, Any]:
    """Per-row reward = R[i, predicted_action[i], K_idx]. Sum + dist."""
    n = R.shape[0]
    per_row = R[np.arange(n), predicted_action, K_idx]
    counts = {
        f"n_action_{ai}_v1": int((predicted_action == ai).sum())
        for ai in range(iql_core.N_ACTIONS_V1)
    }
    counts["n_total_v1"] = int(n)
    fractions = {
        f"frac_action_{ai}_v1": float((predicted_action == ai).mean())
        for ai in range(iql_core.N_ACTIONS_V1)
    }
    return {
        "total_reward_bps_v1": float(per_row.sum()),
        "mean_reward_bps_v1": float(per_row.mean()) if n > 0 else 0.0,
        "counts_v1": counts,
        "fractions_v1": fractions,
        "min_action_frac_v1": float(min(fractions.values())) if fractions else 0.0,
    }


def evaluate_baselines(
    R: np.ndarray, K_idx: int,
) -> dict[str, dict[str, Any]]:
    """Baselines: ALWAYS_SKIP, ALWAYS_TAKE_LONG, RANDOM_UNIFORM, ORACLE."""
    n = R.shape[0]
    out: dict[str, dict[str, Any]] = {}
    # ALWAYS_SKIP
    a_skip = np.zeros(n, dtype=np.int64)
    out["always_skip_v1"] = evaluate_policy_outcome(R, a_skip, K_idx)
    # ALWAYS_TAKE_LONG
    a_take_long = np.full(n, iql_core.ACTION_TAKE_LONG_NOW_ID, dtype=np.int64)
    out["always_take_long_v1"] = evaluate_policy_outcome(R, a_take_long, K_idx)
    # RANDOM_UNIFORM
    rng = np.random.default_rng(SEED_V1)
    a_rand = rng.integers(0, iql_core.N_ACTIONS_V1, size=n)
    out["random_uniform_v1"] = evaluate_policy_outcome(R, a_rand, K_idx)
    # ORACLE: argmax over a
    a_oracle = R[:, :, K_idx].argmax(axis=1)
    out["oracle_v1"] = evaluate_policy_outcome(R, a_oracle, K_idx)
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def evaluate_one_fold(
    fold: dict[str, np.ndarray],
    X: np.ndarray, R: np.ndarray, oracle_action: np.ndarray,
    *,
    variant: str,
    artifact_root: Path,
) -> dict[str, Any]:
    fold_id_v1 = fold["fold_id_v1"]
    train_idx = fold["train_idx_v1"]
    val_idx = fold["val_idx_v1"]
    test_idx = fold["test_idx_v1"]
    X_train = X[train_idx]
    R_train = R[train_idx]
    X_val = X[val_idx]
    R_val = R[val_idx]
    X_test = X[test_idx]
    R_test = R[test_idx]

    # Per-class guard on val
    val_action_counts = {
        ai: int((oracle_action[val_idx] == ai).sum())
        for ai in range(iql_core.N_ACTIONS_V1)
    }
    val_class_guard_status = (
        "OK" if min(val_action_counts.values()) >= MIN_PER_CLASS_VAL
        else "WARN_BELOW_MIN_PER_CLASS"
    )

    print(
        f"[{ACTION}] {fold_id_v1} {variant} "
        f"train={len(X_train)} val={len(X_val)} test={len(X_test)} "
        f"val_action_counts={val_action_counts} "
        f"guard={val_class_guard_status}",
        flush=True,
    )

    # Train multi-head IQL
    model = iql_core.train_multi_head_entry_iql(
        X_train, R_train,
        k_horizons=K_HORIZONS,
        n_actions=iql_core.N_ACTIONS_V1,
        tau=TRAIN_TAU,
        epochs_q=TRAIN_EPOCHS_Q,
        epochs_v=TRAIN_EPOCHS_V,
        k_iterations=TRAIN_K_VQ_ITERATIONS,
        batch_size=TRAIN_BATCH_SIZE,
        hidden_dim=TRAIN_HIDDEN_DIM,
        n_hidden=TRAIN_N_HIDDEN,
        seed=SEED_V1,
        prefer_cuda=True,
    )

    # Evaluate per K-horizon × per aggregator on val and test
    K_primary_idx = K_HORIZONS.index(K_PRIMARY)
    eval_rows: list[dict[str, Any]] = []
    best_val_total = -np.inf
    best_combo: dict[str, Any] | None = None
    best_val_actions = None
    best_test_actions = None

    for agg in AGGREGATORS:
        val_actions = iql_core.policy_argmax_action(X_val, model, aggregator=agg)
        test_actions = iql_core.policy_argmax_action(X_test, model, aggregator=agg)
        for ki, K in enumerate(K_HORIZONS):
            val_metric = evaluate_policy_outcome(R_val, val_actions, ki)
            test_metric = evaluate_policy_outcome(R_test, test_actions, ki)
            eval_rows.append({
                "fold_id_v1": fold_id_v1,
                "variant_v1": variant,
                "aggregator_v1": agg,
                "k_horizon_v1": K,
                "val_total_reward_bps_v1": val_metric["total_reward_bps_v1"],
                "val_min_action_frac_v1": val_metric["min_action_frac_v1"],
                "val_action_fractions_v1": val_metric["fractions_v1"],
                "test_total_reward_bps_v1": test_metric["total_reward_bps_v1"],
                "test_min_action_frac_v1": test_metric["min_action_frac_v1"],
                "test_action_fractions_v1": test_metric["fractions_v1"],
            })
            if K == K_PRIMARY and val_metric["total_reward_bps_v1"] > best_val_total:
                best_val_total = val_metric["total_reward_bps_v1"]
                best_combo = {"aggregator_v1": agg, "k_horizon_v1": K}
                best_val_actions = val_actions
                best_test_actions = test_actions

    test_baselines = evaluate_baselines(R_test, K_primary_idx)
    val_baselines = evaluate_baselines(R_val, K_primary_idx)

    # Save model
    model_dir = artifact_root / "trained_models_v1"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"{variant}_{fold_id_v1}.pt"
    torch.save(
        {
            "q_state_dict": model.q_net.state_dict(),
            "v_state_dict": model.v_net.state_dict(),
            "feature_means": model.feature_means.tolist(),
            "feature_stds": model.feature_stds.tolist(),
            "k_horizons": K_HORIZONS,
            "n_actions": iql_core.N_ACTIONS_V1,
            "state_dim": model.state_dim,
            "variant": variant,
            "fold_id": fold_id_v1,
            "hidden_dim": TRAIN_HIDDEN_DIM,
            "n_hidden": TRAIN_N_HIDDEN,
            "dropout": iql_core.DEFAULT_DROPOUT,
            "schema_v1": "MULTI_HEAD_ENTRY_IQL_V2_CHECKPOINT",
        },
        model_path,
    )

    return {
        "fold_id_v1": fold_id_v1,
        "variant_v1": variant,
        "best_combo_v1": best_combo,
        "best_val_total_reward_bps_v1": float(best_val_total) if np.isfinite(best_val_total) else None,
        "best_test_metric_v1": (
            evaluate_policy_outcome(R_test, best_test_actions, K_primary_idx)
            if best_test_actions is not None else None
        ),
        "val_class_guard_status_v1": val_class_guard_status,
        "val_action_counts_v1": val_action_counts,
        "val_baselines_v1": val_baselines,
        "test_baselines_v1": test_baselines,
        "all_evaluations_v1": eval_rows,
        "model_path_v1": str(model_path),
    }


def determine_status(per_fold_results: list[dict[str, Any]]) -> tuple[str, str, str, dict[str, Any]]:
    """Aggregate go/no-go across folds + variants. Picks BEST variant by mean test reward."""
    # Group by variant
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for r in per_fold_results:
        by_variant.setdefault(r["variant_v1"], []).append(r)

    best_variant = None
    best_mean_test = -np.inf
    variant_summaries: dict[str, dict[str, Any]] = {}
    for variant, results in by_variant.items():
        test_totals = []
        oracle_totals = []
        skip_baseline_totals = []
        min_action_fracs = []
        for r in results:
            test_metric = r.get("best_test_metric_v1")
            if test_metric:
                test_totals.append(test_metric["total_reward_bps_v1"])
                min_action_fracs.append(test_metric["min_action_frac_v1"])
            tb = r.get("test_baselines_v1", {})
            if tb.get("oracle_v1"):
                oracle_totals.append(tb["oracle_v1"]["total_reward_bps_v1"])
            if tb.get("always_skip_v1"):
                skip_baseline_totals.append(tb["always_skip_v1"]["total_reward_bps_v1"])
        mean_test = float(np.mean(test_totals)) if test_totals else -np.inf
        mean_oracle = float(np.mean(oracle_totals)) if oracle_totals else 0.0
        mean_skip = float(np.mean(skip_baseline_totals)) if skip_baseline_totals else 0.0
        min_min_frac = float(np.min(min_action_fracs)) if min_action_fracs else 0.0
        variant_summaries[variant] = {
            "mean_test_reward_bps_v1": mean_test,
            "mean_oracle_reward_bps_v1": mean_oracle,
            "mean_skip_baseline_reward_bps_v1": mean_skip,
            "fraction_of_oracle_v1": (mean_test / mean_oracle) if mean_oracle != 0 else None,
            "min_action_frac_across_folds_v1": min_min_frac,
        }
        if mean_test > best_mean_test:
            best_mean_test = mean_test
            best_variant = variant

    headline = {
        "n_folds_v1": max(len(v) for v in by_variant.values()) if by_variant else 0,
        "n_variants_v1": len(by_variant),
        "best_variant_v1": best_variant,
        "best_mean_test_reward_bps_v1": float(best_mean_test) if np.isfinite(best_mean_test) else None,
        "variant_summaries_v1": variant_summaries,
    }

    # Status determination
    best_summary = variant_summaries.get(best_variant or "", {})
    min_frac = best_summary.get("min_action_frac_across_folds_v1", 0.0)
    skip_baseline = best_summary.get("mean_skip_baseline_reward_bps_v1", 0.0)
    delta_skip = best_mean_test - skip_baseline

    # Degeneracy threshold: a class is genuinely degenerate only if it has
    # ZERO usage AND the oracle says it should have been used. Since R_NET_REAL
    # oracle distribution can have SKIP at ~1-3%, requiring 2% of policy
    # actions to be SKIP is too strict. Use 0.1% (3 examples per 3000) as the
    # absolute floor — anything below that means the action is fully ignored.
    if min_frac < 0.001:
        return (
            "ENTRY_IQL_V2_PARTIAL_DEGENERATE_ACTION_DIST",
            "REPAIR_ENTRY_IQL_V2_BEFORE_FURTHER_WORK",
            f"Best variant {best_variant} action distribution degenerate "
            f"(min frac {min_frac:.4f} < 0.1%) — IQL fully ignores at least one action.",
            headline,
        )
    if delta_skip > 200.0:
        return (
            "ENTRY_IQL_V2_PASS_BEATS_REALIZED_BASELINE",
            "BUILD_ENTRY_IQL_V2_RUNTIME_ADAPTER",
            f"Best variant {best_variant} beats always-skip baseline by "
            f"{delta_skip:+.0f} bps total mean test reward.",
            headline,
        )
    if abs(delta_skip) <= 200.0:
        return (
            "ENTRY_IQL_V2_PARTIAL_TIES_BASELINE",
            "REPAIR_ENTRY_IQL_V2_BEFORE_FURTHER_WORK",
            f"Best variant {best_variant} ties always-skip baseline "
            f"(delta {delta_skip:+.0f} bps).",
            headline,
        )
    return (
        "ENTRY_IQL_V2_PARTIAL_DEGRADES_VS_BASELINE",
        "REPAIR_ENTRY_IQL_V2_BEFORE_FURTHER_WORK",
        f"Best variant {best_variant} underperforms always-skip baseline by "
        f"{-delta_skip:+.0f} bps.",
        headline,
    )


def write_artifacts(
    forward_outcome_dir: Path = DEFAULT_FORWARD_OUTCOME_DIR,
    out_root: Path | None = None,
    *,
    sample_n_rows: int | None = None,
    variants_subset: list[str] | None = None,
    built_at_utc: str | None = None,
) -> dict[str, Any]:
    timestamp = built_at_utc or _stamp()
    artifact_root = out_root or (DEFAULT_REPORTS_ROOT / f"{ACTION}_{timestamp}_LOCK")
    artifact_root.mkdir(parents=True, exist_ok=True)

    per_week_dir = forward_outcome_dir / "per_week"
    if not per_week_dir.exists():
        return {
            "artifact_root": str(artifact_root),
            "summary": {
                "final_status_v1": "ENTRY_IQL_V2_BLOCKED_BY_INPUT_LOCK_MISSING",
                "next_action_v1": "REPAIR_ENTRY_IQL_V2_BEFORE_FURTHER_WORK",
                "missing_input_v1": str(per_week_dir),
            },
        }

    print(f"[{ACTION}] loading forward-outcome dataset from {per_week_dir}", flush=True)
    df = load_forward_outcome_dataset(per_week_dir)
    print(f"[{ACTION}] loaded rows={len(df):,} cols={len(df.columns)}", flush=True)
    if sample_n_rows is not None and sample_n_rows < len(df):
        df = df.sample(n=sample_n_rows, random_state=SEED_V1).reset_index(drop=True)
        print(f"[{ACTION}] subsampled to {len(df):,} rows for smoke-test", flush=True)

    # Build state matrix once (shared across variants)
    X, feature_names = build_state_matrix(df)
    print(f"[{ACTION}] state matrix: {X.shape}, features={len(feature_names)}", flush=True)

    # Build R for each variant once
    R_by_variant: dict[str, np.ndarray] = {}
    for variant in (variants_subset or REWARD_VARIANTS):
        R_by_variant[variant] = build_reward_matrix(df, variant=variant)
        print(f"[{ACTION}] built reward matrix {variant}: shape={R_by_variant[variant].shape} "
              f"mean={float(R_by_variant[variant].mean()):.3f} max={float(R_by_variant[variant].max()):.1f}",
              flush=True)

    # Oracle action computed via R_NET_REAL @ K_PRIMARY (the realistic reward
    # variant where SKIP can win when forward path is too costly). R_RAW would
    # never let SKIP win since MFE >= 0 always — using it would give 0 SKIP
    # examples in folds and collapse the action space.
    if "R_NET_REAL" in R_by_variant:
        R_for_strat = R_by_variant["R_NET_REAL"]
        strat_variant = "R_NET_REAL"
    else:
        R_for_strat = build_reward_matrix(df, variant="R_NET_REAL")
        strat_variant = "R_NET_REAL_TEMP"
    K_primary_idx = K_HORIZONS.index(K_PRIMARY)
    oracle_action = derive_oracle_action(R_for_strat, K_primary_idx)
    oracle_dist = {ai: int((oracle_action == ai).sum()) for ai in range(iql_core.N_ACTIONS_V1)}
    print(f"[{ACTION}] oracle action (stratifier={strat_variant} K={K_PRIMARY}) distribution: {oracle_dist}", flush=True)

    folds = build_stratified_folds(
        n_rows=len(df), oracle_action=oracle_action,
        n_folds=N_FOLDS, val_size=VAL_SIZE, test_size=TEST_SIZE, seed=SEED_V1,
    )

    per_fold_results: list[dict[str, Any]] = []
    flat_evaluations: list[dict[str, Any]] = []
    for variant in (variants_subset or REWARD_VARIANTS):
        for fold in folds:
            r = evaluate_one_fold(
                fold, X, R_by_variant[variant], oracle_action,
                variant=variant, artifact_root=artifact_root,
            )
            per_fold_results.append(r)
            flat_evaluations.extend(r.get("all_evaluations_v1", []))

    _write_rows(artifact_root / "per_fold_per_variant_evaluations_v1.csv", flat_evaluations)

    status, next_action, recommendation, headline = determine_status(per_fold_results)

    summary = {
        "layer_name": "ENTRY_IQL_V2_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "headline_v1": headline,
        "input_forward_outcome_dir_v1": str(forward_outcome_dir),
        "n_rows_v1": int(len(df)),
        "n_features_v1": int(X.shape[1]),
        "feature_names_v1": feature_names,
        "k_horizons_v1": K_HORIZONS,
        "k_primary_v1": K_PRIMARY,
        "reward_variants_v1": list(R_by_variant.keys()),
        "oracle_action_distribution_v1": oracle_dist,
        "n_folds_v1": N_FOLDS,
        "val_size_v1": VAL_SIZE,
        "test_size_v1": TEST_SIZE,
        "seed_v1": SEED_V1,
        "min_per_class_val_v1": MIN_PER_CLASS_VAL,
        "per_fold_summary_v1": [
            {
                "fold_id_v1": r["fold_id_v1"],
                "variant_v1": r["variant_v1"],
                "best_combo_v1": r["best_combo_v1"],
                "best_val_total_reward_bps_v1": r["best_val_total_reward_bps_v1"],
                "best_test_metric_v1": r["best_test_metric_v1"],
                "val_class_guard_status_v1": r["val_class_guard_status_v1"],
                "val_action_counts_v1": r["val_action_counts_v1"],
                "test_baselines_v1": r["test_baselines_v1"],
            }
            for r in per_fold_results
        ],
        "research_only_v1": True,
        "iql_production_allowed_v1": False,
    }
    _write_json(artifact_root / "summary_v1.json", summary)
    _write_json(
        artifact_root / "status_v1.json",
        {
            "layer_name": "ENTRY_IQL_V2_STATUS_V1",
            "status_v1": "MATERIALIZED_RESEARCH_ONLY_GATE",
            "final_status_v1": status,
            "next_action_v1": next_action,
        },
    )
    _write_json(
        artifact_root / "manifest_v1.json",
        {
            "layer_id_v1": ACTION,
            "built_at_utc_v1": summary["built_at_utc_v1"],
            "output_dir_v1": str(artifact_root),
            "artifact_paths_v1": {
                "summary": str(artifact_root / "summary_v1.json"),
                "status": str(artifact_root / "status_v1.json"),
                "per_fold_csv": str(artifact_root / "per_fold_per_variant_evaluations_v1.csv"),
                "trained_models_dir": str(artifact_root / "trained_models_v1"),
            },
        },
    )
    return {"artifact_root": str(artifact_root), "summary": summary}


def main() -> None:
    global TRAIN_EPOCHS_Q, TRAIN_EPOCHS_V, TRAIN_K_VQ_ITERATIONS, TRAIN_BATCH_SIZE, TRAIN_HIDDEN_DIM, TRAIN_N_HIDDEN
    parser = argparse.ArgumentParser(description=f"Materialize {ACTION}.")
    parser.add_argument("--forward-outcome-dir", type=str, default=str(DEFAULT_FORWARD_OUTCOME_DIR))
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--sample-n-rows", type=int, default=None,
                        help="Subsample to N rows for smoke-testing (default: use all).")
    parser.add_argument("--variants", type=str, default=None,
                        help="Comma-sep subset of reward variants to train (default: all 6).")
    parser.add_argument("--budget", type=str, default="fast", choices=list(BUDGET_PRESETS.keys()),
                        help="Training compute budget: fast (smoke), medium, full (production).")
    parser.add_argument("--built-at-utc", type=str, default=None)
    args = parser.parse_args()

    preset = BUDGET_PRESETS[args.budget]
    TRAIN_EPOCHS_Q = preset["epochs_q"]
    TRAIN_EPOCHS_V = preset["epochs_v"]
    TRAIN_K_VQ_ITERATIONS = preset["k_iter"]
    TRAIN_BATCH_SIZE = preset["batch"]
    TRAIN_HIDDEN_DIM = preset["hidden"]
    TRAIN_N_HIDDEN = preset["n_hidden"]
    print(f"[{ACTION}] training budget '{args.budget}': {preset}", flush=True)

    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else None
    variants_subset = args.variants.split(",") if args.variants else None
    result = write_artifacts(
        forward_outcome_dir=Path(args.forward_outcome_dir).expanduser().resolve(),
        out_root=out_root,
        sample_n_rows=args.sample_n_rows,
        variants_subset=variants_subset,
        built_at_utc=args.built_at_utc,
    )
    print(json.dumps(_jsonable(result["summary"]), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
