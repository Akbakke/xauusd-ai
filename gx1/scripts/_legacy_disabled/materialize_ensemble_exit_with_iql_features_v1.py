#!/usr/bin/env python3
"""Ensemble exit meta-classifier: combine V3 transformer + IQL_GPU + state features.

Approach (Vei A from advanced-RL discussion)
--------------------------------------------
Don't retrain V3 from scratch. Instead, learn a small **tabular meta-policy**
that takes V3's per-bar `exit_prob_v1` AND IQL_GPU's per-bar
`iql_gpu_p_exit_v1` (plus a handful of state features), and outputs an
ensemble exit-decision.

Pros:
  - Cheap (logistic regression / tiny MLP on tabular features)
  - V3 stays frozen (canonical-bundle contract intact)
  - Ensemble strictly cannot do WORSE than V3 alone (in expectation, with
    proper regularization)
  - Captures complementary signal between transformer-attention-output and
    IQL-Q-advantage-output

Inputs
------
- Per-bar augmented dataset (used by V2 IQL pipeline)
- Bridge output parquet:
  shadow_meta_with_iql_gpu_recommendations_v1.parquet (LATEST LOCK)

Target
------
Binary: "would_exiting_at_this_bar_be_profitable_v1"
    = 1 if running_pnl_at_close_bps > realized_pnl_at_trade_end_bps + spread_buffer
    = 0 otherwise

This labels each bar as a potentially-better-than-realized exit point.

Walk-forward
------------
Reuse FOLD_DEFINITIONS from wf_gate. Train per-fold, eval per-fold.
Output meta-policy p_exit per row.

Compare three policies on test cohort PnL:
  1. V3-alone: exit when exit_prob_v1 > threshold
  2. IQL-alone: exit when iql_gpu_p_exit_v1 > threshold
  3. Meta-ensemble: exit when meta_p_exit > threshold

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

from gx1.scripts import (
    materialize_build_true_implicit_q_learning_v1 as v1_gate,
)
from gx1.scripts import true_iql_gpu_core_v1 as iql_core
from gx1.scripts import (
    materialize_run_exit_iql_with_v2_state_and_reward_variants_v1 as v2_train_gate,
)
from gx1.scripts import (
    materialize_walk_forward_validation_v1 as wf_gate,
)
from gx1.scripts import (
    materialize_learn_trade_skip_meta_classifier_at_trade_open_v1 as skip_v1_gate,
)
from gx1.scripts import (
    materialize_build_iql_offline_data_contract_research_only_v1 as contract_gate,
)
from gx1.scripts import (
    materialize_bridge_iql_gpu_to_teacher_supervision_v1 as bridge_gate,
)
from gx1.scripts import materialize_exit_off_policy_eval_harness_v1 as eval_gate

import torch


ACTION = "ENSEMBLE_EXIT_WITH_IQL_FEATURES_V1"
DEFAULT_REPORTS_ROOT = v1_gate.DEFAULT_REPORTS_ROOT

# Meta-policy hyperparameters (tabular; lightweight).
LR = 1e-2
WEIGHT_DECAY = 1e-3
EPOCHS = 50
BATCH_SIZE = 1024
HIDDEN_DIM = 32
N_HIDDEN = 1
SEED_V1 = v1_gate.SEED_V1
EXIT_THRESHOLD_GRID: list[float] = [0.30, 0.40, 0.50, 0.60, 0.70]

# Meta-feature columns (built per-bar inline).
# v3_p_exit and iql_p_exit + (optionally) advantage + a few key state features.
META_FEATURE_COLS = [
    "v3_p_exit_v1",
    "iql_p_exit_v1",
    "iql_advantage_v1",
    "running_pnl_at_close_bps_v1",
    "running_mfe_bps_v1",
    "running_mae_bps_v1",
    "bars_held_v1",
    "atr_bps_now_v1",
    "exit_prob_v1",  # alias of v3_p_exit_v1 (retained for compatibility)
]

ALLOWED_FINAL_STATUSES = {
    "ENSEMBLE_PASS_BEATS_V3_ALONE",
    "ENSEMBLE_PARTIAL_TIES_V3_ALONE",
    "ENSEMBLE_PARTIAL_DEGRADES_VS_V3_ALONE",
    "ENSEMBLE_BLOCKED_BY_BRIDGE_OUTPUT_MISSING",
}
ALLOWED_NEXT_ACTIONS = {
    "FULL_BUDGET_IQL_RETRAIN_V1",
    "DISTILL_V3_AS_TARGET_FROM_IQL_LABELS_V1",
    "REPAIR_ENSEMBLE_BEFORE_FURTHER_WORK_V1",
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


def _find_latest_bridge_output() -> Path | None:
    locks = sorted(
        DEFAULT_REPORTS_ROOT.glob(
            "BRIDGE_IQL_GPU_TO_TEACHER_SUPERVISION_V1_*_LOCK"
        )
    )
    for lock in reversed(locks):
        candidate = lock / "shadow_meta_with_iql_gpu_recommendations_v1.parquet"
        if candidate.exists():
            return candidate
    return None


def _train_iql_per_fold(
    inputs: dict[str, Any], candidate_uid_order: list[str], fold: dict[str, Any]
) -> tuple[iql_core.IQLGpuModel, pd.DataFrame, np.ndarray]:
    """Train one IQL_GPU model on this fold's train split + return per_bar+X for inference."""
    uid_to_split = wf_gate._assign_fold_split(candidate_uid_order, fold)
    per_bar, X_full, _ = wf_gate._build_per_bar_for_fold(inputs, uid_to_split)
    train_mask = (per_bar["primary_split_v1"] == "train").to_numpy()
    per_bar_train = per_bar[train_mask].reset_index(drop=True)
    X_train = X_full[train_mask].astype(np.float32)
    variant = next(
        v for v in v2_train_gate.REWARD_VARIANTS_V2
        if v["reward_id_v1"] == bridge_gate.LOCKED_VARIANT
    )
    reward_col = variant["reward_column_v1"]
    if reward_col not in per_bar_train.columns:
        raise RuntimeError(f"REWARD_COLUMN_MISSING: {reward_col}")
    trans = v1_gate._build_transition_tuples(per_bar_train, X_train, reward_col)
    a_t = trans["a_v1"].astype(np.int64)
    r_t = trans["r_v1"].astype(np.float32)
    next_idx_t = trans["next_idx_v1"].astype(np.int64)
    done_t = trans["done_v1"].astype(bool)
    model = iql_core.train_true_iql_gpu(
        X_train, a_t, r_t, next_idx_t, done_t,
        tau=bridge_gate.LOCKED_TAU, gamma=bridge_gate.GAMMA_LOCKED,
        k_iterations=bridge_gate.K_VQ_ITERATIONS,
        inner_epochs=bridge_gate.INNER_EPOCHS,
        lr=bridge_gate.LR, weight_decay=bridge_gate.WEIGHT_DECAY,
        hidden_dim=bridge_gate.HIDDEN_DIM, n_hidden=bridge_gate.N_HIDDEN,
        batch_size=bridge_gate.BATCH_SIZE, seed=SEED_V1, prefer_cuda=True,
    )
    return model, per_bar, X_full


def _build_meta_features_inline(
    per_bar: pd.DataFrame, X_full: np.ndarray, model: iql_core.IQLGpuModel
) -> pd.DataFrame:
    """Compute per-bar meta-features inline: V3 p_exit + IQL p_exit + state."""
    X = X_full.astype(np.float32)
    p_exit_iql = iql_core.true_iql_policy_exit_prob_gpu(
        X, model, beta=bridge_gate.LOCKED_BETA, clip=bridge_gate.ADVANTAGE_CLIP,
    )
    q_hold = model.predict_q(X, action_id=iql_core.ACTION_HOLD_ID)
    q_exit = model.predict_q(X, action_id=iql_core.ACTION_EXIT_NOW_ID)
    df = per_bar.copy()
    df["v3_p_exit_v1"] = df["exit_prob_v1"].astype(float)
    df["iql_p_exit_v1"] = p_exit_iql
    df["iql_advantage_v1"] = (q_exit - q_hold).astype(float)
    return df


def _build_better_than_realized_label(per_bar: pd.DataFrame) -> np.ndarray:
    """Per-bar binary: is exiting at this bar profitable vs trade end?

    Compares running_pnl_at_close_bps_v1 at this bar to the trade's final
    realized pnl. Positive => exiting now would beat realized; treat as 1.
    """
    df = per_bar.reset_index(drop=True)
    final_pnl_per_uid = (
        df.sort_values(["candidate_uid_v1", "bars_held_v1"])
        .groupby("candidate_uid_v1", sort=False)
        .tail(1)[["candidate_uid_v1", "running_pnl_at_close_bps_v1"]]
        .rename(columns={"running_pnl_at_close_bps_v1": "trade_final_pnl_bps_v1"})
    )
    j = df.merge(final_pnl_per_uid, on="candidate_uid_v1", how="left")
    label = (
        j["running_pnl_at_close_bps_v1"] > j["trade_final_pnl_bps_v1"]
    ).astype(int).to_numpy()
    return label


def _train_meta_policy_torch(
    X_train: np.ndarray, y_train: np.ndarray,
) -> torch.nn.Module:
    """Tiny MLP meta-policy (1 hidden layer, 32 units) trained with BCE."""
    torch.manual_seed(SEED_V1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = X_train.shape[1]
    layers = []
    d = in_dim
    for _ in range(N_HIDDEN):
        layers += [torch.nn.Linear(d, HIDDEN_DIM), torch.nn.ReLU()]
        d = HIDDEN_DIM
    layers.append(torch.nn.Linear(d, 1))
    model = torch.nn.Sequential(*layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    bce = torch.nn.BCEWithLogitsLoss()
    Xt = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    n = Xt.shape[0]
    g = torch.Generator(device="cpu").manual_seed(SEED_V1)
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            logits = model(Xt[idx]).squeeze(-1)
            loss = bce(logits, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def _predict_meta(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    with torch.no_grad():
        Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
        return torch.sigmoid(model(Xt).squeeze(-1)).cpu().numpy()


def _exit_index_threshold(
    per_bar: pd.DataFrame, p_exit: np.ndarray, threshold: float
) -> pd.Series:
    """First bar per trade where p_exit >= threshold; else realized last bar."""
    realized_idx_map = eval_gate._exit_index_realized_exit(per_bar)
    p_series = pd.Series(p_exit, index=per_bar.index)
    out: list[tuple[str, int]] = []
    for uid, group in per_bar.groupby("candidate_uid_v1", sort=False):
        triggered = group[p_series.loc[group.index] >= threshold]
        if not triggered.empty:
            out.append((uid, int(triggered.index[0])))
        else:
            out.append((uid, int(realized_idx_map.loc[uid])))
    return pd.Series({uid: idx for uid, idx in out})


def _evaluate_three_policies(
    per_bar: pd.DataFrame, p_v3: np.ndarray, p_iql: np.ndarray, p_meta: np.ndarray,
    split: str, locked_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate three policies on this split.

    If locked_thresholds is None, sweeps EXIT_THRESHOLD_GRID and picks the
    best per-policy threshold (use this on the VAL split for selection).
    If locked_thresholds is provided (from prior VAL selection), applies
    those thresholds without re-sweeping (use this on the TEST split for
    honest walk-forward evaluation).
    """
    out: dict[str, Any] = {"split_v1": split}
    for name, p in [("v3_alone", p_v3), ("iql_alone", p_iql), ("meta", p_meta)]:
        if locked_thresholds is not None:
            thr = locked_thresholds[name]
            exit_idx = _exit_index_threshold(per_bar, p, thr)
            metric = eval_gate.evaluate_policy(
                per_bar, exit_idx, policy_id=f"{name}_locked_thr{thr}", split=split
            )
            best_total = metric["total_realized_pnl_bps_v1"] if metric else None
            out[f"{name}_best_threshold_v1"] = float(thr)
            out[f"{name}_best_total_pnl_v1"] = float(best_total) if best_total is not None else None
            out[f"{name}_best_metric_v1"] = metric
        else:
            best_total = -np.inf
            best_threshold = None
            best_metric = None
            for thr in EXIT_THRESHOLD_GRID:
                exit_idx = _exit_index_threshold(per_bar, p, thr)
                metric = eval_gate.evaluate_policy(
                    per_bar, exit_idx, policy_id=f"{name}_thr{thr}", split=split
                )
                if metric and metric["total_realized_pnl_bps_v1"] > best_total:
                    best_total = metric["total_realized_pnl_bps_v1"]
                    best_threshold = thr
                    best_metric = metric
            out[f"{name}_best_threshold_v1"] = float(best_threshold) if best_threshold is not None else None
            out[f"{name}_best_total_pnl_v1"] = float(best_total) if best_threshold is not None else None
            out[f"{name}_best_metric_v1"] = best_metric
    out["meta_minus_v3_v1"] = (
        out["meta_best_total_pnl_v1"] - out["v3_alone_best_total_pnl_v1"]
        if out["meta_best_total_pnl_v1"] is not None
        and out["v3_alone_best_total_pnl_v1"] is not None
        else None
    )
    out["meta_minus_iql_v1"] = (
        out["meta_best_total_pnl_v1"] - out["iql_alone_best_total_pnl_v1"]
        if out["meta_best_total_pnl_v1"] is not None
        and out["iql_alone_best_total_pnl_v1"] is not None
        else None
    )
    return out


def _go_no_go(per_fold_results: list[dict[str, Any]]) -> tuple[str, str, str, dict[str, Any]]:
    test_meta_total = sum(
        (r["test"]["meta_best_total_pnl_v1"] or 0.0) for r in per_fold_results
    )
    test_v3_total = sum(
        (r["test"]["v3_alone_best_total_pnl_v1"] or 0.0) for r in per_fold_results
    )
    delta = test_meta_total - test_v3_total
    headline = {
        "n_folds_v1": len(per_fold_results),
        "test_meta_total_v1": test_meta_total,
        "test_v3_alone_total_v1": test_v3_total,
        "delta_meta_minus_v3_v1": delta,
        "per_fold_v1": [
            {
                "fold_id_v1": r["fold_id_v1"],
                "meta": r["test"]["meta_best_total_pnl_v1"],
                "v3": r["test"]["v3_alone_best_total_pnl_v1"],
                "iql": r["test"]["iql_alone_best_total_pnl_v1"],
                "meta_minus_v3": r["test"]["meta_minus_v3_v1"],
            }
            for r in per_fold_results
        ],
    }
    if delta > 100.0:
        return (
            "ENSEMBLE_PASS_BEATS_V3_ALONE",
            "DISTILL_V3_AS_TARGET_FROM_IQL_LABELS_V1",
            f"Ensemble beats V3 alone by {delta:+.0f} bps total. Meta-policy "
            "captures complementary signal between V3 and IQL. Next: try full IQL retrain "
            "and see if delta grows further.",
            headline,
        )
    if abs(delta) <= 100.0:
        return (
            "ENSEMBLE_PARTIAL_TIES_V3_ALONE",
            "FULL_BUDGET_IQL_RETRAIN_V1",
            f"Ensemble approximately ties V3 alone (delta {delta:+.0f} bps). "
            "Reduced-budget IQL doesn't add much signal. Full-budget retrain should help.",
            headline,
        )
    return (
        "ENSEMBLE_PARTIAL_DEGRADES_VS_V3_ALONE",
        "REPAIR_ENSEMBLE_BEFORE_FURTHER_WORK_V1",
        f"Ensemble underperforms V3 alone by {-delta:+.0f} bps. Investigate.",
        headline,
    )


def _evaluate_fold(
    inputs: dict[str, Any], candidate_uid_order: list[str], fold: dict[str, Any],
) -> dict[str, Any]:
    fold_id = fold["fold_id_v1"]
    print(f"[ENSEMBLE] FOLD {fold_id} training IQL_GPU on FOLD train split", flush=True)
    model, per_bar, X_full = _train_iql_per_fold(inputs, candidate_uid_order, fold)
    df = _build_meta_features_inline(per_bar, X_full, model)
    label = _build_better_than_realized_label(df)
    train_mask = (df["primary_split_v1"] == "train").to_numpy()
    val_mask = (df["primary_split_v1"] == "val").to_numpy()
    test_mask = (df["primary_split_v1"] == "test").to_numpy()
    feat_cols = [c for c in META_FEATURE_COLS if c in df.columns]
    print(f"[ENSEMBLE] {fold_id} feat_cols={feat_cols}", flush=True)
    feats = df[feat_cols].astype(np.float32).fillna(0.0).to_numpy()
    Xtr, ytr = feats[train_mask], label[train_mask]
    print(f"[ENSEMBLE] {fold_id} training meta-MLP n_train={len(Xtr)} n_pos={ytr.sum()}", flush=True)
    meta_model = _train_meta_policy_torch(Xtr, ytr)
    p_meta_full = _predict_meta(meta_model, feats)
    p_v3 = df["v3_p_exit_v1"].astype(float).to_numpy()
    p_iql = df["iql_p_exit_v1"].astype(float).to_numpy()

    # Walk-forward methodology: sweep threshold on VAL, lock it, apply on TEST.
    val_eval = _evaluate_three_policies(
        df[val_mask].reset_index(drop=True),
        p_v3[val_mask], p_iql[val_mask], p_meta_full[val_mask],
        split="val",
    )
    locked = {
        "v3_alone": val_eval["v3_alone_best_threshold_v1"],
        "iql_alone": val_eval["iql_alone_best_threshold_v1"],
        "meta": val_eval["meta_best_threshold_v1"],
    }
    test_eval = _evaluate_three_policies(
        df[test_mask].reset_index(drop=True),
        p_v3[test_mask], p_iql[test_mask], p_meta_full[test_mask],
        split="test",
        locked_thresholds=locked,
    )
    print(f"[ENSEMBLE] {fold_id} test: meta={test_eval['meta_best_total_pnl_v1']:+.0f} "
          f"v3={test_eval['v3_alone_best_total_pnl_v1']:+.0f} "
          f"iql={test_eval['iql_alone_best_total_pnl_v1']:+.0f} "
          f"meta_minus_v3={test_eval['meta_minus_v3_v1']:+.0f}",
          flush=True)
    return {"fold_id_v1": fold_id, "val": val_eval, "test": test_eval}


def write_artifacts(
    out_root: Path | None = None, *, built_at_utc: str | None = None,
) -> dict[str, Any]:
    timestamp = built_at_utc or _stamp()
    artifact_root = out_root or (DEFAULT_REPORTS_ROOT / f"{ACTION}_{timestamp}_LOCK")
    artifact_root.mkdir(parents=True, exist_ok=True)

    inputs = wf_gate._load_inputs()
    trades_all = skip_v1_gate._load_trade_outcomes_concat()
    trades_all["candidate_uid_v1"] = trades_all["candidate_uid"].astype(str)
    trades_all["open_ts_utc"] = pd.to_datetime(trades_all["open_ts_utc"], utc=True)
    split_df = pd.read_parquet(
        inputs["required_paths"]["split_locked_dataset"],
        columns=["candidate_uid_v1"],
    )
    accepted = set(split_df["candidate_uid_v1"].astype(str).unique())
    trades_acc = (
        trades_all[trades_all["candidate_uid_v1"].isin(accepted)]
        .sort_values(["open_ts_utc", "candidate_uid_v1"], kind="mergesort")
        .reset_index(drop=True)
    )
    candidate_uid_order = trades_acc["candidate_uid_v1"].astype(str).tolist()

    print(f"[ENSEMBLE] cuda_available={torch.cuda.is_available()}", flush=True)
    per_fold_results = [
        _evaluate_fold(inputs, candidate_uid_order, fold)
        for fold in wf_gate.FOLD_DEFINITIONS
    ]

    flat_rows: list[dict[str, Any]] = []
    for r in per_fold_results:
        flat_rows.append({
            "fold_id_v1": r["fold_id_v1"],
            "split_v1": "val",
            **{k: v for k, v in r["val"].items() if not k.endswith("_metric_v1")},
        })
        flat_rows.append({
            "fold_id_v1": r["fold_id_v1"],
            "split_v1": "test",
            **{k: v for k, v in r["test"].items() if not k.endswith("_metric_v1")},
        })
    _write_rows(artifact_root / "per_fold_per_split_evaluations_v1.csv", flat_rows)

    status, next_action, recommendation, headline = _go_no_go(per_fold_results)
    validate_final_status(status, next_action)

    summary = {
        "layer_name": "ENSEMBLE_EXIT_WITH_IQL_FEATURES_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "headline_v1": headline,
        "per_fold_summary_v1": per_fold_results,
        "meta_feature_cols_v1": META_FEATURE_COLS,
        "exit_threshold_grid_v1": EXIT_THRESHOLD_GRID,
        "research_only_v1": True,
    }
    _write_json(artifact_root / "summary_v1.json", summary)
    _write_json(
        artifact_root / "status_v1.json",
        {
            "layer_name": "ENSEMBLE_STATUS_V1",
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
                "per_fold_csv": str(artifact_root / "per_fold_per_split_evaluations_v1.csv"),
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
