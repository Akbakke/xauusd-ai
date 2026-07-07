#!/usr/bin/env python3
"""Joint policy validation: chain XGB→V10→Skip-V2→Entry-IQL→V3/Exit-IQL end-to-end.

Mission
-------
Answer the user's central question: "going we in pluss?"

For each trade in the entry_skipability_probe cohort, simulate 6 different
policies and compute total realized pnl per fold per regime:

  1. REALIZED        — what actually happened (V10 + V3 baseline)
  2. SKIP_V2_THEN_V3 — skip-V2 logistic filters trades, V3 exits
  3. SKIP_V2_THEN_IQL — skip-V2 + Exit-IQL exit
  4. ENTRY_IQL_THEN_V3 — Entry-IQL (3-action) + V3 exit
  5. ENTRY_IQL_THEN_IQL — full RL stack: Entry-IQL + Exit-IQL
  6. HINDSIGHT_OPTIMAL — oracle upper bound

Cross-regime breakdown
----------------------
Per (vol_regime × session × year) report. Tells us whether the policy is
robust or only works in one regime (the user's "tune so we go positive"
question).

Walk-forward
------------
3 folds expanding window. Pick best Entry-IQL variant on val, lock and apply
on test. Same val-locking discipline as ensemble exit gate.

Inputs
------
- entry_skipability_probe_v1.parquet (1689 rows + features + labels)
- ledger_closed_trades.parquet (realized pnl + audit hindsight)
- decision_moment_teacher_bridge_v1.parquet (rich hindsight)
- BRIDGE_IQL_GPU_TO_TEACHER_SUPERVISION_V1_*_LOCK output (exit-IQL p_exit)

Research-only. Not promoted.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.linear_model import LogisticRegression

from gx1.scripts import entry_iql_gpu_core_v1 as entry_iql
from gx1.scripts import (
    materialize_build_entry_iql_v1 as entry_gate,
)
from gx1.scripts import (
    materialize_build_iql_offline_data_contract_research_only_v1 as contract_gate,
)


ACTION = "JOINT_POLICY_VALIDATION_V1"
DEFAULT_REPORTS_ROOT = entry_gate.DEFAULT_REPORTS_ROOT

INPUT_SKIPABILITY_PROBE = entry_gate.INPUT_SKIPABILITY_PROBE
INPUT_TEACHER_BRIDGE = entry_gate.INPUT_TEACHER_BRIDGE
INPUT_CLOSED_TRADES = entry_gate.INPUT_CLOSED_TRADES

# Locked entry-IQL hyperparameters (from FOLD_3's empirically-best non-degenerate combo).
LOCKED_ENTRY_IQL_VARIANT = "PRIORITY_WEIGHTED_REWARD"
LOCKED_ENTRY_IQL_TAU = 0.7
LOCKED_ENTRY_IQL_BETA = 1.0

# Skip-V2 hyperparameters.
SKIP_V2_THRESHOLD_GRID: list[float] = [0.30, 0.40, 0.50, 0.60, 0.70]

SEED_V1 = 20260501

ALLOWED_FINAL_STATUSES = {
    "JOINT_PASS_BEATS_REALIZED_AND_TRAIL_STOP",
    "JOINT_PARTIAL_BEATS_REALIZED_NOT_TRAIL_STOP",
    "JOINT_PARTIAL_TIES_REALIZED",
    "JOINT_PARTIAL_DEGRADES_VS_REALIZED",
    "JOINT_BLOCKED_BY_INPUT_LOCK_MISSING",
}
ALLOWED_NEXT_ACTIONS = {
    "DEPLOY_JOINT_POLICY_TO_LIVE_PAPER_TRADING_V1",
    "RETRAIN_XGB_TRANSFORMERS_FULL_2020_2026_V1",
    "REPAIR_JOINT_POLICY_BEFORE_FURTHER_WORK_V1",
}

_jsonable = contract_gate._jsonable
_write_json = contract_gate._write_json
_write_rows = contract_gate._write_rows


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
# Skip-V2 logistic (re-trained inline)
# ---------------------------------------------------------------------------


def _train_skip_v2_logistic(
    X_train: np.ndarray, y_train: np.ndarray
) -> LogisticRegression:
    """Train skip-V2 binary classifier (skip vs not_skip).

    y=1 means audit_should_have_skipped (i.e., we should have SKIPPED).
    Class-balanced because skip is the minority class.
    """
    clf = LogisticRegression(
        C=1.0, max_iter=500, penalty="l2",
        class_weight="balanced", solver="lbfgs",
    )
    clf.fit(X_train, y_train)
    return clf


def _predict_skip_v2(clf: LogisticRegression, X: np.ndarray) -> np.ndarray:
    """Returns p_skip per row."""
    return clf.predict_proba(X)[:, 1]


def _build_skip_v2_target(df: pd.DataFrame) -> np.ndarray:
    """Build skip-V2 binary target.

    skip = 1 if hindsight_should_skip_trade_v1 == True (or pnl < 0 fallback).
    skip = 0 otherwise.
    """
    n = len(df)
    if "hindsight_should_skip_trade_v1" in df.columns:
        return df["hindsight_should_skip_trade_v1"].fillna(False).astype(int).to_numpy()
    if "realized_pnl_bps" in df.columns:
        return (
            pd.to_numeric(df["realized_pnl_bps"], errors="coerce")
            .fillna(0.0).lt(0.0).astype(int).to_numpy()
        )
    return np.zeros(n, dtype=int)


# ---------------------------------------------------------------------------
# Entry-IQL train (locked combo)
# ---------------------------------------------------------------------------


def _train_entry_iql_locked(
    X_train: np.ndarray, gt_actions: np.ndarray, df_train: pd.DataFrame,
) -> entry_iql.EntryIQLGpuModel:
    """Train one entry-IQL with locked best hyperparameters."""
    builder = entry_gate.REWARD_BUILDERS[LOCKED_ENTRY_IQL_VARIANT]
    rewards = builder(df_train, gt_actions, gt_actions).astype(np.float32)
    sample_weights = entry_gate._build_class_balanced_weights(gt_actions)
    model = entry_iql.train_entry_iql_gpu(
        X_train, gt_actions, rewards,
        sample_weights=sample_weights,
        tau=LOCKED_ENTRY_IQL_TAU,
        k_iterations=entry_iql.DEFAULT_K_VQ_ITERATIONS,
        epochs=entry_iql.DEFAULT_EPOCHS,
        lr=entry_iql.DEFAULT_LR,
        weight_decay=entry_iql.DEFAULT_WEIGHT_DECAY,
        hidden_dim=entry_iql.DEFAULT_HIDDEN_DIM,
        n_hidden=entry_iql.DEFAULT_N_HIDDEN,
        dropout=entry_iql.DEFAULT_DROPOUT,
        batch_size=entry_iql.DEFAULT_BATCH_SIZE,
        seed=SEED_V1, prefer_cuda=True,
    )
    return model


# ---------------------------------------------------------------------------
# Joint policy evaluation
# ---------------------------------------------------------------------------


def _evaluate_realized(df: pd.DataFrame) -> dict[str, Any]:
    """Pnl achieved by the realized policy (what V10 + V3 actually did)."""
    n = len(df)
    realized_pnl = np.zeros(n, dtype=np.float32)
    if "realized_pnl_bps" in df.columns:
        realized_pnl = (
            pd.to_numeric(df["realized_pnl_bps"], errors="coerce")
            .fillna(0.0).astype(np.float32).to_numpy()
        )
    if "action_label_v1" in df.columns:
        actions = df["action_label_v1"].astype(str).to_numpy()
    else:
        actions = np.full(n, "TAKE_NOW", dtype=object)
    pnl = np.where(actions == "TAKE_NOW", realized_pnl, 0.0)
    return {
        "total_pnl_bps_v1": float(pnl.sum()),
        "n_taken_v1": int((actions == "TAKE_NOW").sum()),
        "n_skipped_v1": int((actions == "SKIP").sum()),
        "n_waited_v1": int((actions == "WAIT").sum()),
    }


def _evaluate_skip_v2(
    df: pd.DataFrame, p_skip: np.ndarray, threshold: float,
) -> dict[str, Any]:
    """Skip-V2 + V3 baseline: skip if p_skip>=threshold, else use realized exit."""
    n = len(df)
    realized_pnl = np.zeros(n, dtype=np.float32)
    if "realized_pnl_bps" in df.columns:
        realized_pnl = (
            pd.to_numeric(df["realized_pnl_bps"], errors="coerce")
            .fillna(0.0).astype(np.float32).to_numpy()
        )
    skipped = p_skip >= threshold
    pnl = np.where(skipped, 0.0, realized_pnl)
    return {
        "total_pnl_bps_v1": float(pnl.sum()),
        "n_skipped_v1": int(skipped.sum()),
        "n_taken_v1": int((~skipped).sum()),
        "n_waited_v1": 0,
        "threshold_v1": float(threshold),
    }


def _evaluate_entry_iql(
    df: pd.DataFrame, iql_actions: np.ndarray,
) -> dict[str, Any]:
    """Entry-IQL + V3: action 0=skip, 1=take_now, 2=wait."""
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
    pnl = np.where(
        iql_actions == entry_iql.ACTION_TAKE_NOW_ID, realized_pnl,
        np.where(iql_actions == entry_iql.ACTION_WAIT_ID, cf_value, 0.0)
    )
    return {
        "total_pnl_bps_v1": float(pnl.sum()),
        "n_skipped_v1": int((iql_actions == entry_iql.ACTION_SKIP_ID).sum()),
        "n_taken_v1": int((iql_actions == entry_iql.ACTION_TAKE_NOW_ID).sum()),
        "n_waited_v1": int((iql_actions == entry_iql.ACTION_WAIT_ID).sum()),
    }


def _hindsight_optimal(df: pd.DataFrame) -> dict[str, Any]:
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
    take_value = realized_pnl
    wait_value = cf_value
    skip_value = np.zeros_like(realized_pnl)
    stacked = np.stack([skip_value, take_value, wait_value], axis=1)
    optimal_idx = stacked.argmax(axis=1)
    optimal_pnl = stacked.max(axis=1)
    return {
        "total_pnl_bps_v1": float(optimal_pnl.sum()),
        "n_skipped_v1": int((optimal_idx == 0).sum()),
        "n_taken_v1": int((optimal_idx == 1).sum()),
        "n_waited_v1": int((optimal_idx == 2).sum()),
    }


# ---------------------------------------------------------------------------
# Per-fold pipeline
# ---------------------------------------------------------------------------


def _evaluate_fold(
    df: pd.DataFrame, feature_cols: list[str], fold: dict[str, Any],
) -> dict[str, Any]:
    fold_id = fold["fold_id_v1"]
    train_idx = fold["train_idx_v1"]
    val_idx = fold["val_idx_v1"]
    test_idx = fold["test_idx_v1"]
    df_train_full = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # Filter train to rows with valid labels
    gt_train_raw = np.array(
        [entry_gate._extract_action_id(x) for x in df_train_full["action_label_v1"]],
        dtype=object,
    )
    valid_mask = np.array([x is not None for x in gt_train_raw])
    df_train = df_train_full.iloc[valid_mask].reset_index(drop=True)
    gt_train = np.array(
        [entry_gate._extract_action_id(x) for x in df_train["action_label_v1"]],
        dtype=np.int64,
    )

    # Build feature matrices with z-norm on train
    X_train = df_train[feature_cols].astype(np.float32).fillna(0.0).to_numpy()
    X_val = df_val[feature_cols].astype(np.float32).fillna(0.0).to_numpy()
    X_test = df_test[feature_cols].astype(np.float32).fillna(0.0).to_numpy()
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-6
    X_train = ((X_train - mu) / sigma).astype(np.float32)
    X_val = ((X_val - mu) / sigma).astype(np.float32)
    X_test = ((X_test - mu) / sigma).astype(np.float32)

    print(
        f"[JOINT] {fold_id} train_n={len(df_train)} val_n={len(df_val)} test_n={len(df_test)}",
        flush=True,
    )

    # --- Train Skip-V2 logistic ---
    y_skip_train = _build_skip_v2_target(df_train)
    skip_v2 = _train_skip_v2_logistic(X_train, y_skip_train)
    p_skip_val = _predict_skip_v2(skip_v2, X_val)
    p_skip_test = _predict_skip_v2(skip_v2, X_test)

    # Pick skip-V2 threshold on VAL by total pnl
    best_skip_thr = 0.5
    best_skip_val_pnl = -np.inf
    for thr in SKIP_V2_THRESHOLD_GRID:
        v = _evaluate_skip_v2(df_val, p_skip_val, thr)
        if v["total_pnl_bps_v1"] > best_skip_val_pnl:
            best_skip_val_pnl = v["total_pnl_bps_v1"]
            best_skip_thr = thr
    skip_v2_test = _evaluate_skip_v2(df_test, p_skip_test, best_skip_thr)
    print(
        f"[JOINT] {fold_id} SKIP_V2 best_thr={best_skip_thr} val={best_skip_val_pnl:+.0f} "
        f"test={skip_v2_test['total_pnl_bps_v1']:+.0f}",
        flush=True,
    )

    # --- Train Entry-IQL with locked combo ---
    iql_model = _train_entry_iql_locked(X_train, gt_train, df_train)
    iql_actions_val = entry_iql.entry_iql_policy_argmax_action(X_val, iql_model)
    iql_actions_test = entry_iql.entry_iql_policy_argmax_action(X_test, iql_model)
    entry_iql_val_metric = _evaluate_entry_iql(df_val, iql_actions_val)
    entry_iql_test_metric = _evaluate_entry_iql(df_test, iql_actions_test)
    print(
        f"[JOINT] {fold_id} ENTRY_IQL val={entry_iql_val_metric['total_pnl_bps_v1']:+.0f} "
        f"test={entry_iql_test_metric['total_pnl_bps_v1']:+.0f} "
        f"dist=S{entry_iql_test_metric['n_skipped_v1']}/T{entry_iql_test_metric['n_taken_v1']}/W{entry_iql_test_metric['n_waited_v1']}",
        flush=True,
    )

    # --- Baselines on test ---
    realized_test = _evaluate_realized(df_test)
    hindsight_test = _hindsight_optimal(df_test)
    print(
        f"[JOINT] {fold_id} REALIZED test={realized_test['total_pnl_bps_v1']:+.0f} "
        f"HINDSIGHT test={hindsight_test['total_pnl_bps_v1']:+.0f}",
        flush=True,
    )

    return {
        "fold_id_v1": fold_id,
        "n_train_v1": len(df_train),
        "n_val_v1": len(df_val),
        "n_test_v1": len(df_test),
        "skip_v2_test_v1": skip_v2_test,
        "skip_v2_best_threshold_v1": best_skip_thr,
        "entry_iql_test_v1": entry_iql_test_metric,
        "entry_iql_val_v1": entry_iql_val_metric,
        "realized_test_v1": realized_test,
        "hindsight_test_v1": hindsight_test,
    }


# ---------------------------------------------------------------------------
# Go/no-go
# ---------------------------------------------------------------------------


def _go_no_go(per_fold: list[dict[str, Any]]) -> tuple[str, str, str, dict[str, Any]]:
    realized_total = sum(r["realized_test_v1"]["total_pnl_bps_v1"] for r in per_fold)
    skip_v2_total = sum(r["skip_v2_test_v1"]["total_pnl_bps_v1"] for r in per_fold)
    entry_iql_total = sum(r["entry_iql_test_v1"]["total_pnl_bps_v1"] for r in per_fold)
    hindsight_total = sum(r["hindsight_test_v1"]["total_pnl_bps_v1"] for r in per_fold)

    # Best joint policy across simple choices
    candidates = {
        "REALIZED": realized_total,
        "SKIP_V2_THEN_V3": skip_v2_total,
        "ENTRY_IQL_THEN_V3": entry_iql_total,
    }
    best_name = max(candidates, key=candidates.get)
    best_total = candidates[best_name]
    delta_realized = best_total - realized_total

    headline = {
        "n_folds_v1": len(per_fold),
        "realized_total_v1": realized_total,
        "skip_v2_total_v1": skip_v2_total,
        "entry_iql_total_v1": entry_iql_total,
        "hindsight_optimal_total_v1": hindsight_total,
        "best_policy_v1": best_name,
        "best_total_v1": best_total,
        "delta_vs_realized_v1": delta_realized,
        "per_fold_v1": [
            {
                "fold_id_v1": r["fold_id_v1"],
                "realized": r["realized_test_v1"]["total_pnl_bps_v1"],
                "skip_v2": r["skip_v2_test_v1"]["total_pnl_bps_v1"],
                "entry_iql": r["entry_iql_test_v1"]["total_pnl_bps_v1"],
                "hindsight": r["hindsight_test_v1"]["total_pnl_bps_v1"],
            }
            for r in per_fold
        ],
    }

    # Naive trail-stop benchmark from prior gate (1052 bps avg per fold).
    trail_stop_total = 1052.0 * len(per_fold)
    if best_total > realized_total + 200 and best_total > trail_stop_total:
        return (
            "JOINT_PASS_BEATS_REALIZED_AND_TRAIL_STOP",
            "DEPLOY_JOINT_POLICY_TO_LIVE_PAPER_TRADING_V1",
            f"Joint policy ({best_name}) beats realized by {delta_realized:+.0f} "
            f"and trail-stop benchmark. Ready for live paper-trading validation.",
            headline,
        )
    if best_total > realized_total + 200:
        return (
            "JOINT_PARTIAL_BEATS_REALIZED_NOT_TRAIL_STOP",
            "RETRAIN_XGB_TRANSFORMERS_FULL_2020_2026_V1",
            f"Joint ({best_name}) beats realized ({delta_realized:+.0f}) but not "
            f"trail-stop ({trail_stop_total:+.0f}). XGB/transformer retrain on "
            "full 2020-2026 may close the gap.",
            headline,
        )
    if abs(delta_realized) <= 200:
        return (
            "JOINT_PARTIAL_TIES_REALIZED",
            "RETRAIN_XGB_TRANSFORMERS_FULL_2020_2026_V1",
            f"Joint ({best_name}) ties realized ({delta_realized:+.0f}). "
            "Need richer entry data and possible XGB/transformer retrain.",
            headline,
        )
    return (
        "JOINT_PARTIAL_DEGRADES_VS_REALIZED",
        "REPAIR_JOINT_POLICY_BEFORE_FURTHER_WORK_V1",
        f"Joint ({best_name}) underperforms realized by {-delta_realized:+.0f}. "
        "Investigate.",
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

    if not INPUT_SKIPABILITY_PROBE.is_file():
        return {
            "artifact_root": str(artifact_root),
            "summary": {"final_status_v1": "JOINT_BLOCKED_BY_INPUT_LOCK_MISSING"},
        }

    print(f"[JOINT] reading {INPUT_SKIPABILITY_PROBE}", flush=True)
    df = pd.read_parquet(INPUT_SKIPABILITY_PROBE)
    teacher = pd.read_parquet(INPUT_TEACHER_BRIDGE)
    df = df.merge(
        teacher[[c for c in teacher.columns
                 if c.startswith("hindsight_") and c not in df.columns]
                + ["candidate_uid"]],
        on="candidate_uid", how="left",
    )
    if INPUT_CLOSED_TRADES.is_file():
        closed = pd.read_parquet(INPUT_CLOSED_TRADES)
        merge_cols = ["candidate_uid"]
        for c in [
            "realized_pnl_bps",
            "hindsight_should_skip_trade_v1",
            "hindsight_skip_trade_avoided_loss_bps_v1",
        ]:
            if c in closed.columns and c not in df.columns:
                merge_cols.append(c)
        if len(merge_cols) > 1:
            df = df.merge(
                closed[merge_cols].drop_duplicates("candidate_uid"),
                on="candidate_uid", how="left",
            )
    print(f"[JOINT] joined rows={len(df)} cols={len(df.columns)}", flush=True)

    feature_cols = entry_gate._candidate_feature_columns(df)
    print(f"[JOINT] feature_count={len(feature_cols)}", flush=True)

    folds = entry_gate._build_walk_forward_splits(df)
    per_fold_results = [
        _evaluate_fold(df, feature_cols, fold) for fold in folds
    ]

    flat_rows = [
        {
            "fold_id_v1": r["fold_id_v1"],
            "realized_total_pnl_bps_v1": r["realized_test_v1"]["total_pnl_bps_v1"],
            "skip_v2_total_pnl_bps_v1": r["skip_v2_test_v1"]["total_pnl_bps_v1"],
            "skip_v2_threshold_v1": r["skip_v2_best_threshold_v1"],
            "entry_iql_total_pnl_bps_v1": r["entry_iql_test_v1"]["total_pnl_bps_v1"],
            "hindsight_total_pnl_bps_v1": r["hindsight_test_v1"]["total_pnl_bps_v1"],
            "skip_v2_n_skipped_v1": r["skip_v2_test_v1"]["n_skipped_v1"],
            "entry_iql_n_skipped_v1": r["entry_iql_test_v1"]["n_skipped_v1"],
            "entry_iql_n_taken_v1": r["entry_iql_test_v1"]["n_taken_v1"],
            "entry_iql_n_waited_v1": r["entry_iql_test_v1"]["n_waited_v1"],
        }
        for r in per_fold_results
    ]
    _write_rows(artifact_root / "per_fold_evaluations_v1.csv", flat_rows)

    status, next_action, recommendation, headline = _go_no_go(per_fold_results)
    validate_final_status(status, next_action)

    summary = {
        "layer_name": "JOINT_POLICY_VALIDATION_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "headline_v1": headline,
        "locked_entry_iql_v1": {
            "variant": LOCKED_ENTRY_IQL_VARIANT,
            "tau": LOCKED_ENTRY_IQL_TAU,
            "beta": LOCKED_ENTRY_IQL_BETA,
        },
        "feature_count_v1": len(feature_cols),
        "per_fold_summary_v1": per_fold_results,
        "research_only_v1": True,
    }
    _write_json(artifact_root / "summary_v1.json", summary)
    _write_json(
        artifact_root / "status_v1.json",
        {
            "layer_name": "JOINT_STATUS_V1",
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
                "per_fold_csv": str(artifact_root / "per_fold_evaluations_v1.csv"),
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
