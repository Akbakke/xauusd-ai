#!/usr/bin/env python3
"""Build entry-side IQL: 3-action contextual bandit (SKIP/TAKE_NOW/WAIT).

Mission
-------
Learn an entry-quality policy from the existing pre-RL training data
(`entry_skipability_probe_v1.parquet` + `decision_moment_teacher_bridge_v1.parquet`)
that decides — at each XGB+V10 trigger moment — whether the right action is
SKIP, TAKE_NOW, or WAIT.

State: ALL available trigger-time features + direction signals (long & short)
       from XGB so the IQL is direction-aware even when V10 is long-only.

Action: 3-way categorical {SKIP, TAKE_NOW, WAIT}.

Reward (three variants for comparison):
  1. REALIZED_PNL_REWARD     — actual realized pnl for trades taken; 0 for SKIP;
                               counterfactual_value_bps for WAIT.
  2. PRIORITY_WEIGHTED_REWARD — hindsight_teacher_priority_bps weighted by
                               whether action matches teacher's recommendation.
  3. TEACHER_AGREE_REWARD    — +1 if action matches hindsight_entry_action_refined,
                               -1 otherwise. Pure supervised-classification reward.

Walk-forward
------------
3 expanding-window folds on time-ordered trades. Train on TRAIN split, lock
hyperparameters on VAL, report final on TEST.

Baselines compared
------------------
  - Realized policy (what V10 + skip-V2 actually did)
  - Skipability-teacher-direct (uses action_label_v1 as oracle)
  - Hindsight-optimal (upper-bound, picks action that maximizes value per row)
  - Random uniform 3-way
  - Always-TAKE_NOW (current default)

Promotion criteria
------------------
Entry-IQL passes if:
  - Total test pnl > realized policy by ≥ +200 bps across folds
  - Cross-fold-stability: no fold has IQL < realized by > 300 bps
  - Action distribution non-degenerate (each action chosen ≥ 5% of rows)

This is RESEARCH-ONLY. The trained policy is NOT promoted to runtime.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.scripts import entry_iql_gpu_core_v1 as entry_iql
from gx1.scripts import (
    materialize_build_iql_offline_data_contract_research_only_v1 as contract_gate,
)


ACTION = "BUILD_ENTRY_IQL_V1"
DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
DEFAULT_LEDGER_ROOT = (
    DEFAULT_REPORTS_ROOT / "ALL_TRADE_REVIEW_LEDGER_20260411"
)

# Input parquets (probed entry features + hindsight labels).
INPUT_SKIPABILITY_PROBE = (
    DEFAULT_LEDGER_ROOT
    / "shadow_meta_all_trade_review_entry_skipability_probe_v1.parquet"
)
INPUT_TEACHER_BRIDGE = (
    DEFAULT_LEDGER_ROOT
    / "shadow_meta_all_trade_review_decision_moment_teacher_bridge_v1.parquet"
)
INPUT_DECISION_LEDGER = (
    DEFAULT_LEDGER_ROOT
    / "shadow_meta_all_trade_review_as_of_decision_moment_ledger_v1.parquet"
)
INPUT_CLOSED_TRADES = (
    DEFAULT_LEDGER_ROOT
    / "shadow_meta_all_trade_review_ledger_closed_trades.parquet"
)
INPUT_WAIT_VS_TAKE_PROBE = (
    DEFAULT_LEDGER_ROOT
    / "shadow_meta_all_trade_review_entry_wait_vs_take_probe_v1.parquet"
)
INPUT_R3_PREDICTIONS = (
    DEFAULT_REPORTS_ROOT
    / "ALL_TRADE_REVIEW_LEDGER_20260421T_R3_ENTRY_LABEL_FEATURE_RETRAIN_V1"
    / "shadow_meta_all_trade_review_r3_entry_label_feature_prediction_view_v1.parquet"
)

# Categorical columns to one-hot-encode at feature-build time.
# Only columns with >1 distinct value are useful — trend_regime is uniformly
# TREND_NEUTRAL on this cohort so it's skipped.
ONE_HOT_CATEGORICAL_COLS = [
    "as_of_vol_regime_v1",   # EXTREME / HIGH / MEDIUM / LOW
    "as_of_session_v1",      # OVERLAP / US / EU / ASIA
]

# Hyperparameter grid (full production budget — no shortcuts).
TAU_GRID: list[float] = [0.7, 0.8, 0.9]
BETA_GRID: list[float] = [1.0, 3.0, 10.0]
SEED_V1 = 20260501

REWARD_VARIANTS_V1 = [
    {"reward_id_v1": "REALIZED_PNL_REWARD"},
    {"reward_id_v1": "PRIORITY_WEIGHTED_REWARD"},
    {"reward_id_v1": "TEACHER_AGREE_REWARD"},
    {"reward_id_v1": "AUDIT_SHOULD_SKIP_REWARD"},
]
USE_CLASS_BALANCED_WEIGHTS = True  # invert class frequency for sample weights

ALLOWED_FINAL_STATUSES = {
    "ENTRY_IQL_PASS_BEATS_REALIZED",
    "ENTRY_IQL_PARTIAL_TIES_REALIZED",
    "ENTRY_IQL_PARTIAL_DEGRADES_VS_REALIZED",
    "ENTRY_IQL_PARTIAL_DEGENERATE_ACTION_DIST",
    "ENTRY_IQL_BLOCKED_BY_INPUT_LOCK_MISSING",
}
ALLOWED_NEXT_ACTIONS = {
    "BUILD_ENTRY_TIMING_IQL_WITH_WAIT_N_BARS_V1",
    "BUILD_JOINT_ENTRY_AND_EXIT_IQL_V1",
    "REPAIR_ENTRY_IQL_BEFORE_FURTHER_WORK_V1",
}

_jsonable = contract_gate._jsonable
_write_json = contract_gate._write_json
_write_rows = contract_gate._write_rows
_write_report = contract_gate._write_report


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_final_status(status: str, next_action: str) -> bool:
    if status not in ALLOWED_FINAL_STATUSES:
        raise RuntimeError(f"FINAL_STATUS_NOT_ALLOWED: {status}")
    if next_action not in ALLOWED_NEXT_ACTIONS:
        raise RuntimeError(f"NEXT_ACTION_NOT_ALLOWED: {next_action}")
    return True


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _candidate_feature_columns(df: pd.DataFrame) -> list[str]:
    """Pick the maximum-extract set of feature columns from the probe parquet.

    Includes BOTH long and short direction signals so the IQL is direction-
    aware even when V10 is long-only. Excludes any column that contains
    `hindsight_*`, `policy_action_v1`, `*_label_v1` or other LEAK-bearing
    fields that are post-decision.

    Note: this returns NUMERIC feature column names only. Categorical
    columns (regime, session) are handled separately via one-hot encoding
    in `_one_hot_encode_categoricals` and appended at feature-build time.
    """
    leak_substrs = (
        "hindsight_", "_label_v1", "_outcome_", "realized_", "exit_", "trade_id",
        "candidate_uid", "trade_uid", "_priority_", "decision_anchor", "_target_",
        "_split_", "used_for_", "as_of_split_", "_action_", "_take_v1",
        "as_of_row_uid", "run_id", "_timestamp_", "_anchor_",
    )
    feat_cols: list[str] = []
    for c in df.columns:
        if df[c].dtype.kind not in "fib":  # float/int/bool only
            continue
        if any(s in c for s in leak_substrs):
            continue
        feat_cols.append(c)
    return sorted(feat_cols)


def _one_hot_encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode known string-categorical columns and return augmented df.

    Adds columns named `<col>_<value>` for each unique value. Columns with
    a single distinct value (no variance) are skipped since they carry no
    information.
    """
    out = df.copy()
    for col in ONE_HOT_CATEGORICAL_COLS:
        if col not in out.columns:
            continue
        s = out[col].astype(str).fillna("UNKNOWN")
        if s.nunique() <= 1:
            continue
        for val in sorted(s.unique()):
            out[f"{col}__{val}"] = (s == val).astype(np.float32)
    return out


def _extract_action_id(label: str) -> int | None:
    """Map action_label_v1 string to action id."""
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return None
    s = str(label).upper()
    if "SKIP" in s:
        return entry_iql.ACTION_SKIP_ID
    if "WAIT" in s:
        return entry_iql.ACTION_WAIT_ID
    if "TAKE" in s:
        return entry_iql.ACTION_TAKE_NOW_ID
    return None


def _build_realized_pnl_reward(
    df: pd.DataFrame, actions_taken: np.ndarray, ground_truth_actions: np.ndarray
) -> np.ndarray:
    """REALIZED_PNL reward: per-row reward depends on which action the row was taken with.

    For each row, reward[a] tells us the value of action a if we'd taken it.
    But we only OBSERVE one outcome per row (the action that was realized).
    So we use the ground truth action and its observed reward; for other
    actions, we use 0 (training will use only the action that was taken).
    """
    n = len(df)
    realized_pnl = np.zeros(n, dtype=np.float32)
    if "realized_pnl_bps" in df.columns:
        realized_pnl = (
            pd.to_numeric(df["realized_pnl_bps"], errors="coerce")
            .fillna(0.0).astype(np.float32).to_numpy()
        )
    # Counterfactual value coverage is sparse (~25%). When missing we use
    # 0 as the WAIT-reward (no signal). DO NOT fall back to realized_pnl
    # here — that conflates WAIT and TAKE_NOW rewards and causes IQL to
    # collapse to all-WAIT.
    cf_value = np.zeros(n, dtype=np.float32)
    if "hindsight_policy_counterfactual_value_bps_v1" in df.columns:
        cf_value = (
            pd.to_numeric(df["hindsight_policy_counterfactual_value_bps_v1"], errors="coerce")
            .fillna(0.0).astype(np.float32).to_numpy()
        )
    rewards = np.zeros(n, dtype=np.float32)
    for i, gt in enumerate(ground_truth_actions):
        if gt == entry_iql.ACTION_TAKE_NOW_ID:
            rewards[i] = realized_pnl[i]
        elif gt == entry_iql.ACTION_WAIT_ID:
            rewards[i] = cf_value[i]
        else:
            rewards[i] = 0.0
    return rewards


def _build_priority_weighted_reward(
    df: pd.DataFrame, actions_taken: np.ndarray, ground_truth_actions: np.ndarray
) -> np.ndarray:
    """PRIORITY_WEIGHTED reward: |hindsight_teacher_priority_bps| signed by agreement.

    Falls back to hindsight_policy_priority_abs_bps_v1 if teacher_priority is missing.
    """
    n = len(df)
    priority = np.zeros(n, dtype=np.float32)
    for col in ["hindsight_teacher_priority_bps_v1", "hindsight_policy_priority_abs_bps_v1"]:
        if col in df.columns:
            priority = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(0.0).astype(np.float32).to_numpy()
            )
            break
    rewards = np.zeros(len(df), dtype=np.float32)
    for i, gt in enumerate(ground_truth_actions):
        if actions_taken[i] == gt:
            rewards[i] = float(abs(priority[i]))
        else:
            rewards[i] = -float(abs(priority[i]))
    return rewards


def _build_teacher_agree_reward(
    df: pd.DataFrame, actions_taken: np.ndarray, ground_truth_actions: np.ndarray
) -> np.ndarray:
    rewards = np.where(actions_taken == ground_truth_actions, 1.0, -1.0).astype(np.float32)
    return rewards


def _build_audit_should_skip_reward(
    df: pd.DataFrame, actions_taken: np.ndarray, ground_truth_actions: np.ndarray
) -> np.ndarray:
    """AUDIT_SHOULD_SKIP_REWARD: directly uses hindsight_should_skip_trade_v1 boolean.

    For each row:
      - hindsight_should_skip_trade_v1 == True → SKIP is right action
        - SKIP reward = +|skip_avoided_loss_bps| (or default magnitude)
        - TAKE_NOW reward = -|skip_avoided_loss_bps|
        - WAIT reward = 0 (neutral; we didn't have the chance to discover wait was wrong)
      - hindsight_should_skip_trade_v1 == False → NOT_SKIP is right
        - SKIP reward = -|priority_or_pnl| (penalize for skipping good)
        - TAKE_NOW reward = +realized_pnl_bps (capture realized)
        - WAIT reward = +counterfactual_value (capture cf)

    Action semantics: actions_taken = ground_truth_actions in our offline setting.
    We compute reward FOR THE TAKEN action only.
    """
    n = len(df)
    rewards = np.zeros(n, dtype=np.float32)
    should_skip = np.zeros(n, dtype=bool)
    if "hindsight_should_skip_trade_v1" in df.columns:
        should_skip = (
            df["hindsight_should_skip_trade_v1"].fillna(False).astype(bool).to_numpy()
        )
    avoided_loss = np.zeros(n, dtype=np.float32)
    if "hindsight_skip_trade_avoided_loss_bps_v1" in df.columns:
        avoided_loss = (
            pd.to_numeric(df["hindsight_skip_trade_avoided_loss_bps_v1"], errors="coerce")
            .fillna(0.0).abs().astype(np.float32).to_numpy()
        )
    priority = np.zeros(n, dtype=np.float32)
    if "hindsight_teacher_priority_bps_v1" in df.columns:
        priority = (
            pd.to_numeric(df["hindsight_teacher_priority_bps_v1"], errors="coerce")
            .fillna(0.0).abs().astype(np.float32).to_numpy()
        )
    realized_pnl = np.zeros(n, dtype=np.float32)
    if "realized_pnl_bps" in df.columns:
        realized_pnl = (
            pd.to_numeric(df["realized_pnl_bps"], errors="coerce")
            .fillna(0.0).astype(np.float32).to_numpy()
        )
    cf_value = np.zeros(n, dtype=np.float32)
    if "hindsight_policy_counterfactual_value_bps_v1" in df.columns:
        cf_value = (
            pd.to_numeric(df["hindsight_policy_counterfactual_value_bps_v1"], errors="coerce")
            .fillna(0.0).astype(np.float32).to_numpy()
        )
    for i in range(n):
        a = ground_truth_actions[i]
        if should_skip[i]:
            magnitude = max(avoided_loss[i], priority[i], 1.0)
            if a == entry_iql.ACTION_SKIP_ID:
                rewards[i] = magnitude
            elif a == entry_iql.ACTION_TAKE_NOW_ID:
                rewards[i] = -magnitude
            else:  # WAIT
                rewards[i] = 0.0
        else:
            if a == entry_iql.ACTION_SKIP_ID:
                rewards[i] = -max(priority[i], 1.0)  # penalty for skipping good
            elif a == entry_iql.ACTION_TAKE_NOW_ID:
                rewards[i] = realized_pnl[i] if realized_pnl[i] != 0 else max(priority[i], 1.0)
            else:  # WAIT
                rewards[i] = cf_value[i] if cf_value[i] != 0 else 0.0
    return rewards


def _build_class_balanced_weights(
    actions: np.ndarray, n_actions: int = entry_iql.N_ACTIONS
) -> np.ndarray:
    """Sample weight = inverse class frequency, normalized so mean weight = 1."""
    n = len(actions)
    counts = np.array([(actions == k).sum() for k in range(n_actions)], dtype=np.float64)
    counts = np.maximum(counts, 1.0)  # avoid div by zero
    inv_freq = 1.0 / counts
    inv_freq = inv_freq / inv_freq.sum() * n_actions  # normalize so mean = 1 across classes
    weights = np.array([inv_freq[a] for a in actions], dtype=np.float32)
    return weights


REWARD_BUILDERS = {
    "REALIZED_PNL_REWARD": _build_realized_pnl_reward,
    "PRIORITY_WEIGHTED_REWARD": _build_priority_weighted_reward,
    "TEACHER_AGREE_REWARD": _build_teacher_agree_reward,
    "AUDIT_SHOULD_SKIP_REWARD": _build_audit_should_skip_reward,
}


# ---------------------------------------------------------------------------
# Walk-forward folds
# ---------------------------------------------------------------------------


def _build_walk_forward_splits(df: pd.DataFrame) -> list[dict[str, np.ndarray]]:
    """3 expanding-window folds on time-ordered rows.

    Use `decision_timestamp` if present, else fallback to row order.
    """
    if "decision_timestamp" in df.columns:
        ts = pd.to_datetime(df["decision_timestamp"], utc=True, errors="coerce")
        order = np.argsort(ts.values.astype("int64"))
    else:
        order = np.arange(len(df))
    n = len(df)

    folds = []
    # Fold k: train = first k/4 of data, val = next 1/8, test = next 1/8
    fold_specs = [
        ("FOLD_1", 0.50, 0.625, 0.75),  # train [0, 0.50), val [0.50, 0.625), test [0.625, 0.75)
        ("FOLD_2", 0.625, 0.75, 0.875),
        ("FOLD_3", 0.75, 0.875, 1.0),
    ]
    for fold_id, t_end, v_end, te_end in fold_specs:
        train_idx = order[: int(n * t_end)]
        val_idx = order[int(n * t_end) : int(n * v_end)]
        test_idx = order[int(n * v_end) : int(n * te_end)]
        folds.append(
            {
                "fold_id_v1": fold_id,
                "train_idx_v1": train_idx,
                "val_idx_v1": val_idx,
                "test_idx_v1": test_idx,
            }
        )
    return folds


# ---------------------------------------------------------------------------
# Per-fold training + evaluation
# ---------------------------------------------------------------------------


def _evaluate_policy_pnl(
    df: pd.DataFrame, actions: np.ndarray
) -> dict[str, Any]:
    """Compute total realized pnl achievable by following `actions` on this cohort.

    Per-row pnl:
      - SKIP → 0
      - TAKE_NOW → realized_pnl_bps
      - WAIT → hindsight_policy_counterfactual_value_bps_v1
              (if positive — assumes waiting actually helped)
    """
    n = len(df)
    realized_pnl = np.zeros(n, dtype=np.float32)
    if "realized_pnl_bps" in df.columns:
        realized_pnl = (
            pd.to_numeric(df["realized_pnl_bps"], errors="coerce")
            .fillna(0.0).astype(np.float32).to_numpy()
        )
    cf_value = np.zeros(n, dtype=np.float32)
    if "hindsight_policy_counterfactual_value_bps_v1" in df.columns:
        cf_value = (
            pd.to_numeric(df["hindsight_policy_counterfactual_value_bps_v1"], errors="coerce")
            .fillna(0.0).astype(np.float32).to_numpy()
        )
    per_row = np.where(
        actions == entry_iql.ACTION_TAKE_NOW_ID, realized_pnl,
        np.where(actions == entry_iql.ACTION_WAIT_ID, cf_value, 0.0)
    )
    counts = {
        "n_skip_v1": int((actions == entry_iql.ACTION_SKIP_ID).sum()),
        "n_take_now_v1": int((actions == entry_iql.ACTION_TAKE_NOW_ID).sum()),
        "n_wait_v1": int((actions == entry_iql.ACTION_WAIT_ID).sum()),
    }
    counts["n_total_v1"] = int(len(df))
    return {
        "total_pnl_bps_v1": float(per_row.sum()),
        "mean_pnl_bps_v1": float(per_row.mean()) if len(per_row) > 0 else 0.0,
        "action_dist_v1": counts,
    }


def _hindsight_optimal_actions(df: pd.DataFrame) -> np.ndarray:
    """Oracle: per row, pick the action that maximizes value (pnl, cf, 0)."""
    n = len(df)
    realized_pnl = np.zeros(n, dtype=np.float32)
    if "realized_pnl_bps" in df.columns:
        realized_pnl = (
            pd.to_numeric(df["realized_pnl_bps"], errors="coerce")
            .fillna(0.0).astype(np.float32).to_numpy()
        )
    cf_value = np.zeros(n, dtype=np.float32)
    if "hindsight_policy_counterfactual_value_bps_v1" in df.columns:
        cf_value = (
            pd.to_numeric(df["hindsight_policy_counterfactual_value_bps_v1"], errors="coerce")
            .fillna(0.0).astype(np.float32).to_numpy()
        )
    actions = np.zeros(n, dtype=np.int64)  # default SKIP
    take_better = realized_pnl > 0
    wait_better = (cf_value > realized_pnl) & (cf_value > 0)
    actions[take_better] = entry_iql.ACTION_TAKE_NOW_ID
    actions[wait_better] = entry_iql.ACTION_WAIT_ID
    return actions


def _evaluate_fold(
    df: pd.DataFrame, feature_cols: list[str], fold: dict[str, np.ndarray]
) -> dict[str, Any]:
    fold_id = fold["fold_id_v1"]
    train_idx = fold["train_idx_v1"]
    val_idx = fold["val_idx_v1"]
    test_idx = fold["test_idx_v1"]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # Build action labels (ground truth from action_label_v1)
    gt_train = np.array(
        [_extract_action_id(x) for x in df_train["action_label_v1"]],
        dtype=object,
    )
    valid_train = np.array([x is not None for x in gt_train])
    df_train = df_train.iloc[valid_train].reset_index(drop=True)
    gt_train = np.array([_extract_action_id(x) for x in df_train["action_label_v1"]], dtype=np.int64)

    print(
        f"[ENTRY_IQL] {fold_id} train_n={len(df_train)} val_n={len(df_val)} test_n={len(df_test)}",
        flush=True,
    )
    print(
        f"[ENTRY_IQL] {fold_id} train action dist: "
        f"SKIP={int((gt_train==0).sum())} TAKE_NOW={int((gt_train==1).sum())} "
        f"WAIT={int((gt_train==2).sum())}",
        flush=True,
    )

    X_train = df_train[feature_cols].astype(np.float32).fillna(0.0).to_numpy()
    X_val = df_val[feature_cols].astype(np.float32).fillna(0.0).to_numpy()
    X_test = df_test[feature_cols].astype(np.float32).fillna(0.0).to_numpy()

    # Z-score normalize on TRAIN, apply same to VAL+TEST
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-6
    X_train = ((X_train - mu) / sigma).astype(np.float32)
    X_val = ((X_val - mu) / sigma).astype(np.float32)
    X_test = ((X_test - mu) / sigma).astype(np.float32)

    actions_taken_train = gt_train  # for bandit, taken == ground_truth-recommended

    all_evals: list[dict[str, Any]] = []

    # Train per (variant, tau, beta) and pick best on VAL — but reject
    # combos with degenerate val action distribution (any action < 5%).
    # If all combos are degenerate, fall back to highest-pnl regardless.
    best_val_total = -np.inf
    best_combo: dict[str, Any] | None = None
    best_test_actions: np.ndarray | None = None
    best_val_actions: np.ndarray | None = None
    best_val_total_nondegen = -np.inf
    best_combo_nondegen: dict[str, Any] | None = None
    best_test_actions_nondegen: np.ndarray | None = None
    best_val_actions_nondegen: np.ndarray | None = None
    MIN_ACTION_FRAC = 0.05

    for variant in REWARD_VARIANTS_V1:
        v_id = variant["reward_id_v1"]
        builder = REWARD_BUILDERS[v_id]
        rewards = builder(df_train, actions_taken_train, gt_train).astype(np.float32)
        for tau in TAU_GRID:
            sample_weights = (
                _build_class_balanced_weights(actions_taken_train)
                if USE_CLASS_BALANCED_WEIGHTS else None
            )
            try:
                model = entry_iql.train_entry_iql_gpu(
                    X_train, actions_taken_train, rewards,
                    sample_weights=sample_weights,
                    tau=tau, k_iterations=entry_iql.DEFAULT_K_VQ_ITERATIONS,
                    epochs=entry_iql.DEFAULT_EPOCHS,
                    lr=entry_iql.DEFAULT_LR,
                    weight_decay=entry_iql.DEFAULT_WEIGHT_DECAY,
                    hidden_dim=entry_iql.DEFAULT_HIDDEN_DIM,
                    n_hidden=entry_iql.DEFAULT_N_HIDDEN,
                    dropout=entry_iql.DEFAULT_DROPOUT,
                    batch_size=entry_iql.DEFAULT_BATCH_SIZE,
                    seed=SEED_V1, prefer_cuda=True,
                )
            except Exception as exc:  # noqa: BLE001
                all_evals.append({
                    "fold_id_v1": fold_id, "variant_v1": v_id,
                    "tau_v1": tau, "status_v1": "TRAIN_ERROR",
                    "error_v1": str(exc)[:200],
                })
                continue
            for beta in BETA_GRID:
                val_actions = entry_iql.entry_iql_policy_action_probs(
                    X_val, model, beta=beta
                ).argmax(axis=1)
                val_metric = _evaluate_policy_pnl(df_val, val_actions)
                test_actions = entry_iql.entry_iql_policy_action_probs(
                    X_test, model, beta=beta
                ).argmax(axis=1)
                test_metric = _evaluate_policy_pnl(df_test, test_actions)
                row = {
                    "fold_id_v1": fold_id, "variant_v1": v_id,
                    "tau_v1": float(tau), "beta_v1": float(beta),
                    "val_total_pnl_bps_v1": val_metric["total_pnl_bps_v1"],
                    "test_total_pnl_bps_v1": test_metric["total_pnl_bps_v1"],
                    "val_action_dist_v1": val_metric["action_dist_v1"],
                    "test_action_dist_v1": test_metric["action_dist_v1"],
                }
                all_evals.append(row)
                if val_metric["total_pnl_bps_v1"] > best_val_total:
                    best_val_total = val_metric["total_pnl_bps_v1"]
                    best_combo = {"variant": v_id, "tau": float(tau), "beta": float(beta)}
                    best_test_actions = test_actions
                    best_val_actions = val_actions
                # Non-degenerate selection (preferred when available)
                val_dist = val_metric["action_dist_v1"]
                val_n = max(val_dist.get("n_total_v1", 1), 1)
                val_skip_frac = val_dist.get("n_skip_v1", 0) / val_n
                val_take_frac = val_dist.get("n_take_now_v1", 0) / val_n
                val_wait_frac = val_dist.get("n_wait_v1", 0) / val_n
                non_degen = (
                    val_skip_frac >= MIN_ACTION_FRAC
                    and val_take_frac >= MIN_ACTION_FRAC
                    and val_wait_frac >= MIN_ACTION_FRAC
                )
                if non_degen and val_metric["total_pnl_bps_v1"] > best_val_total_nondegen:
                    best_val_total_nondegen = val_metric["total_pnl_bps_v1"]
                    best_combo_nondegen = {"variant": v_id, "tau": float(tau), "beta": float(beta)}
                    best_test_actions_nondegen = test_actions
                    best_val_actions_nondegen = val_actions
                marker = " *" if non_degen else ""
                print(
                    f"[ENTRY_IQL]   {fold_id} {v_id} tau={tau} beta={beta} "
                    f"val={val_metric['total_pnl_bps_v1']:+.0f} "
                    f"test={test_metric['total_pnl_bps_v1']:+.0f} "
                    f"dist=S{val_skip_frac:.0%}/T{val_take_frac:.0%}/W{val_wait_frac:.0%}{marker}",
                    flush=True,
                )

    # Prefer non-degenerate combo if any exists
    if best_combo_nondegen is not None:
        best_val_total = best_val_total_nondegen
        best_combo = best_combo_nondegen
        best_test_actions = best_test_actions_nondegen
        best_val_actions = best_val_actions_nondegen
        print(
            f"[ENTRY_IQL] {fold_id} non-degenerate best selected: {best_combo}",
            flush=True,
        )
    else:
        print(
            f"[ENTRY_IQL] {fold_id} ALL combos degenerate on val — falling back to "
            f"highest-pnl combo: {best_combo}",
            flush=True,
        )

    # Sentinel block to maintain control flow indentation

    # Baselines on TEST cohort
    realized_test_actions = np.array(
        [_extract_action_id(x) or entry_iql.ACTION_TAKE_NOW_ID
         for x in df_test.get("action_label_v1", [None] * len(df_test))],
        dtype=np.int64,
    )
    hindsight_test_actions = _hindsight_optimal_actions(df_test)
    always_take_test_actions = np.full(len(df_test), entry_iql.ACTION_TAKE_NOW_ID, dtype=np.int64)
    rng = np.random.default_rng(SEED_V1)
    random_test_actions = rng.integers(0, 3, size=len(df_test))

    test_baselines = {
        "realized_v1": _evaluate_policy_pnl(df_test, realized_test_actions),
        "hindsight_optimal_v1": _evaluate_policy_pnl(df_test, hindsight_test_actions),
        "always_take_v1": _evaluate_policy_pnl(df_test, always_take_test_actions),
        "random_uniform_v1": _evaluate_policy_pnl(df_test, random_test_actions),
    }

    iql_test_metric = (
        _evaluate_policy_pnl(df_test, best_test_actions)
        if best_test_actions is not None else None
    )

    iql_test_str = (
        f"{iql_test_metric['total_pnl_bps_v1']:+.0f}"
        if iql_test_metric is not None else "None"
    )
    real_test = test_baselines['realized_v1']['total_pnl_bps_v1']
    hind_test = test_baselines['hindsight_optimal_v1']['total_pnl_bps_v1']
    print(
        f"[ENTRY_IQL] {fold_id} BEST val={best_val_total:+.0f} "
        f"test={iql_test_str} realized={real_test:+.0f} "
        f"hindsight={hind_test:+.0f} combo={best_combo}",
        flush=True,
    )

    return {
        "fold_id_v1": fold_id,
        "best_combo_v1": best_combo,
        "iql_test_metric_v1": iql_test_metric,
        "iql_val_metric_v1": (
            _evaluate_policy_pnl(df_val, best_val_actions)
            if best_val_actions is not None else None
        ),
        "test_baselines_v1": test_baselines,
        "all_evaluations_v1": all_evals,
    }


# ---------------------------------------------------------------------------
# Go/no-go
# ---------------------------------------------------------------------------


def _go_no_go(
    per_fold: list[dict[str, Any]]
) -> tuple[str, str, str, dict[str, Any]]:
    iql_total = sum(
        (r["iql_test_metric_v1"]["total_pnl_bps_v1"] if r["iql_test_metric_v1"] else 0.0)
        for r in per_fold
    )
    realized_total = sum(
        r["test_baselines_v1"]["realized_v1"]["total_pnl_bps_v1"] for r in per_fold
    )
    hindsight_total = sum(
        r["test_baselines_v1"]["hindsight_optimal_v1"]["total_pnl_bps_v1"] for r in per_fold
    )
    delta_realized = iql_total - realized_total
    delta_hindsight = iql_total - hindsight_total

    # Action distribution check (degenerate?)
    fractions: list[dict[str, float]] = []
    for r in per_fold:
        dist = (r["iql_test_metric_v1"] or {}).get("action_dist_v1", {}) or {}
        n = max(int(dist.get("n_total_v1", 1)), 1)
        fractions.append({
            "skip_frac": dist.get("n_skip_v1", 0) / n,
            "take_frac": dist.get("n_take_now_v1", 0) / n,
            "wait_frac": dist.get("n_wait_v1", 0) / n,
        })
    min_action_frac = min(
        min(f.values()) for f in fractions
    ) if fractions else 0.0
    degenerate = min_action_frac < 0.05

    # Cross-fold loss check
    per_fold_delta_realized = [
        (r["iql_test_metric_v1"]["total_pnl_bps_v1"] if r["iql_test_metric_v1"] else 0.0)
        - r["test_baselines_v1"]["realized_v1"]["total_pnl_bps_v1"]
        for r in per_fold
    ]
    worst_fold_loss = min(per_fold_delta_realized)

    headline = {
        "n_folds_v1": len(per_fold),
        "iql_total_pnl_bps_v1": float(iql_total),
        "realized_total_pnl_bps_v1": float(realized_total),
        "hindsight_optimal_total_pnl_bps_v1": float(hindsight_total),
        "iql_minus_realized_v1": float(delta_realized),
        "iql_minus_hindsight_v1": float(delta_hindsight),
        "min_action_frac_v1": float(min_action_frac),
        "worst_fold_delta_v1": float(worst_fold_loss),
        "per_fold_delta_realized_v1": [float(x) for x in per_fold_delta_realized],
        "per_fold_action_fractions_v1": fractions,
        "best_combo_per_fold_v1": [r["best_combo_v1"] for r in per_fold],
    }

    if degenerate:
        return (
            "ENTRY_IQL_PARTIAL_DEGENERATE_ACTION_DIST",
            "REPAIR_ENTRY_IQL_BEFORE_FURTHER_WORK_V1",
            f"Entry IQL test action distribution is degenerate "
            f"(min frac {min_action_frac:.2%} < 5%). Policy collapsed to ≤2 actions; "
            "investigate reward shaping and feature scaling.",
            headline,
        )
    if delta_realized > 200.0 and worst_fold_loss > -300.0:
        return (
            "ENTRY_IQL_PASS_BEATS_REALIZED",
            "BUILD_ENTRY_TIMING_IQL_WITH_WAIT_N_BARS_V1",
            f"Entry IQL beats realized policy by {delta_realized:+.0f} bps total "
            f"({iql_total:+.0f} vs {realized_total:+.0f}). Cross-fold-stable "
            f"(worst fold delta {worst_fold_loss:+.0f}). Next: extend action space "
            "to multi-step WAIT_N for explicit timing optimization.",
            headline,
        )
    if abs(delta_realized) <= 200.0:
        return (
            "ENTRY_IQL_PARTIAL_TIES_REALIZED",
            "BUILD_JOINT_ENTRY_AND_EXIT_IQL_V1",
            f"Entry IQL ties realized policy "
            f"({iql_total:+.0f} vs {realized_total:+.0f}; delta {delta_realized:+.0f}). "
            "Consider joint entry+exit IQL — entry alone may not suffice.",
            headline,
        )
    return (
        "ENTRY_IQL_PARTIAL_DEGRADES_VS_REALIZED",
        "REPAIR_ENTRY_IQL_BEFORE_FURTHER_WORK_V1",
        f"Entry IQL underperforms realized by {-delta_realized:+.0f} bps "
        f"({iql_total:+.0f} vs {realized_total:+.0f}). Investigate.",
        headline,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def write_artifacts(
    out_root: Path | None = None, *, built_at_utc: str | None = None,
) -> dict[str, Any]:
    timestamp = built_at_utc or _stamp()
    artifact_root = out_root or (DEFAULT_REPORTS_ROOT / f"{ACTION}_{timestamp}_LOCK")
    artifact_root.mkdir(parents=True, exist_ok=True)

    # Verify inputs
    for path in [INPUT_SKIPABILITY_PROBE, INPUT_TEACHER_BRIDGE]:
        if not path.is_file():
            return {
                "artifact_root": str(artifact_root),
                "summary": {
                    "final_status_v1": "ENTRY_IQL_BLOCKED_BY_INPUT_LOCK_MISSING",
                    "next_action_v1": "REPAIR_ENTRY_IQL_BEFORE_FURTHER_WORK_V1",
                    "missing_input_v1": str(path),
                },
            }

    print(f"[ENTRY_IQL] reading skipability probe: {INPUT_SKIPABILITY_PROBE}", flush=True)
    skipability_df = pd.read_parquet(INPUT_SKIPABILITY_PROBE)
    n_probe_initial = len(skipability_df)
    n_probe_uids = skipability_df["candidate_uid"].nunique()
    print(f"[ENTRY_IQL] skipability rows={n_probe_initial} unique_uids={n_probe_uids} "
          f"cols={len(skipability_df.columns)}", flush=True)

    # MERGE FIX: teacher_bridge has 2-4 rows per candidate_uid (different
    # decision anchors per trade). Filter to ENTRY_DECISION_ANCHOR only
    # (1 row per uid) BEFORE merging to prevent cartesian product explosion.
    teacher_df_full = pd.read_parquet(INPUT_TEACHER_BRIDGE)
    n_tb_full = len(teacher_df_full)
    teacher_df = teacher_df_full[
        teacher_df_full["hindsight_decision_anchor_type_v1"] == "ENTRY_DECISION_ANCHOR"
    ].copy()
    n_tb_entry = len(teacher_df)
    n_tb_uids = teacher_df["candidate_uid"].nunique()
    print(f"[ENTRY_IQL] teacher_bridge full={n_tb_full} -> ENTRY_DECISION_ANCHOR-filtered="
          f"{n_tb_entry} unique_uids={n_tb_uids}", flush=True)
    if n_tb_entry != n_tb_uids:
        # Defensive: dedupe if multiple ENTRY anchors per uid (shouldn't happen)
        teacher_df = teacher_df.drop_duplicates(subset=["candidate_uid"], keep="first")
        print(f"[ENTRY_IQL] deduped teacher_df to {len(teacher_df)} rows", flush=True)

    df = skipability_df.merge(
        teacher_df[
            [c for c in teacher_df.columns
             if c.startswith("hindsight_") and c not in skipability_df.columns]
            + ["candidate_uid"]
        ],
        on="candidate_uid", how="left", suffixes=("", "_tb"),
    )
    n_after_merge = len(df)
    print(f"[ENTRY_IQL] joined rows={n_after_merge} cols={len(df.columns)}", flush=True)
    # GUARD: row count must equal probe row count (1:1 merge invariant)
    if n_after_merge != n_probe_initial:
        raise RuntimeError(
            f"MERGE_INVARIANT_VIOLATED: probe_rows={n_probe_initial} -> after_merge="
            f"{n_after_merge}. Teacher bridge join not 1:1 — debug filter."
        )

    # Pull realized_pnl_bps + audit hindsight booleans from closed_trades.
    n_pre_closed = len(df)
    if INPUT_CLOSED_TRADES.is_file():
        closed = pd.read_parquet(INPUT_CLOSED_TRADES)
        merge_cols = ["candidate_uid"]
        for c in [
            "realized_pnl_bps",
            "hindsight_should_skip_trade_v1",
            "hindsight_skip_trade_avoided_loss_bps_v1",
            "hindsight_should_hold_longer_v1",
            "hindsight_should_exit_earlier_v1",
        ]:
            if c in closed.columns and c not in df.columns:
                merge_cols.append(c)
        if len(merge_cols) > 1 and "candidate_uid" in closed.columns:
            df = df.merge(
                closed[merge_cols].drop_duplicates("candidate_uid"),
                on="candidate_uid", how="left",
            )
        if len(df) != n_pre_closed:
            raise RuntimeError(
                f"CLOSED_TRADES_MERGE_NOT_1_TO_1: {n_pre_closed} -> {len(df)}"
            )
        print(f"[ENTRY_IQL] merged {len(merge_cols)-1} cols from closed_trades "
              f"rows={len(df)} (1:1 invariant preserved)", flush=True)

    # Merge R3 entry-label model predictions (research models trained on
    # similar audit labels). Provides extra direction signals for IQL.
    n_pre_r3 = len(df)
    if INPUT_R3_PREDICTIONS.is_file():
        r3 = pd.read_parquet(INPUT_R3_PREDICTIONS)
        if "candidate_uid" in r3.columns:
            r3_pred_cols = [
                c for c in r3.columns
                if c.startswith("entry_r3_") and "pred" in c.lower()
            ]
            r3_subset = r3[["candidate_uid"] + r3_pred_cols].drop_duplicates("candidate_uid")
            df = df.merge(r3_subset, on="candidate_uid", how="left")
            print(f"[ENTRY_IQL] merged {len(r3_pred_cols)} R3 prediction cols", flush=True)
            if len(df) != n_pre_r3:
                raise RuntimeError(
                    f"R3_MERGE_NOT_1_TO_1: {n_pre_r3} -> {len(df)}"
                )

    # One-hot encode categorical columns (regime, session) so IQL can use them.
    df = _one_hot_encode_categoricals(df)
    print(f"[ENTRY_IQL] post one-hot rows={len(df)} cols={len(df.columns)}", flush=True)

    feature_cols = _candidate_feature_columns(df)
    print(f"[ENTRY_IQL] selected {len(feature_cols)} numeric feature columns "
          f"(direction-aware, leak-free)", flush=True)

    folds = _build_walk_forward_splits(df)
    per_fold_results = [_evaluate_fold(df, feature_cols, fold) for fold in folds]

    flat_evals: list[dict[str, Any]] = []
    for r in per_fold_results:
        flat_evals.extend(r["all_evaluations_v1"])
    _write_rows(
        artifact_root / "per_fold_per_variant_evaluations_v1.csv", flat_evals
    )

    status, next_action, recommendation, headline = _go_no_go(per_fold_results)
    validate_final_status(status, next_action)

    summary = {
        "layer_name": "ENTRY_IQL_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "headline_v1": headline,
        "feature_count_v1": len(feature_cols),
        "feature_cols_v1": feature_cols,
        "tau_grid_v1": TAU_GRID,
        "beta_grid_v1": BETA_GRID,
        "reward_variants_v1": REWARD_VARIANTS_V1,
        "per_fold_summary_v1": [
            {
                "fold_id_v1": r["fold_id_v1"],
                "best_combo_v1": r["best_combo_v1"],
                "iql_test_pnl_bps_v1": (
                    r["iql_test_metric_v1"]["total_pnl_bps_v1"] if r["iql_test_metric_v1"] else None
                ),
                "iql_test_action_dist_v1": (
                    r["iql_test_metric_v1"]["action_dist_v1"] if r["iql_test_metric_v1"] else None
                ),
                "test_baselines_v1": r["test_baselines_v1"],
            }
            for r in per_fold_results
        ],
        "research_only_v1": True,
        "iql_production_allowed_v1": False,
        "training_blocked_v1": False,
    }
    _write_json(artifact_root / "summary_v1.json", summary)
    _write_json(
        artifact_root / "status_v1.json",
        {
            "layer_name": "ENTRY_IQL_STATUS_V1",
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
            },
        },
    )
    return {"artifact_root": str(artifact_root), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Materialize {ACTION}.")
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--built-at-utc", type=str, default=None)
    args = parser.parse_args()
    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else None
    result = write_artifacts(out_root=out_root, built_at_utc=args.built_at_utc)
    print(json.dumps(_jsonable(result["summary"]), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
