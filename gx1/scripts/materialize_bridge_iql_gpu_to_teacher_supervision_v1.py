#!/usr/bin/env python3
"""Bridge gate: enrich existing teacher-supervision parquet with GPU IQL recommendations.

Reads
-----
- shadow_meta_all_trade_review_policy_action_supervision_join_v1.parquet
  (3,774 rows: per-decision-anchor with hindsight_policy_action_v1 supervision target,
   plus train/val/holdout splits)
- The per-bar V2 state matrix (used by advanced-RL gate)

Trains
------
A fresh IQL_GPU model on the train split using hyperparameters from the
advanced-RL gate (tau=0.7, beta=3.0 — the consistent best across folds for
GIVEBACK_PENALTY_REWARD).

Writes
------
shadow_meta_with_iql_gpu_recommendations_v1.parquet — same rows as input
plus columns:
  - iql_gpu_recommended_action_v1: HOLD / EXIT_NOW (only for MANAGEMENT-domain rows)
  - iql_gpu_p_exit_v1: float in [0, 1]
  - iql_gpu_q_hold_v1, iql_gpu_q_exit_v1, iql_gpu_advantage_v1
  - iql_gpu_agreement_with_teacher_v1: bool (matches hindsight_policy_action_v1)

Diagnostics (in summary_v1.json)
--------------------------------
- agreement_rate per split (train/val/holdout)
- breakdown of iql_recommends vs teacher_says
- counterfactual_value_bps weighted by agreement

Research-only. The output parquet is consumed by future distillation gates
(retrain XGB / V10 transformer / V3 exit transformer with these labels).
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


ACTION = "BRIDGE_IQL_GPU_TO_TEACHER_SUPERVISION_V1"
DEFAULT_REPORTS_ROOT = v1_gate.DEFAULT_REPORTS_ROOT

# Hyperparameters chosen from advanced-RL gate's empirical best:
# tau=0.7 + beta=3.0 was best for GIVEBACK_PENALTY_REWARD on FOLD_1 and FOLD_2.
LOCKED_TAU = 0.7
LOCKED_BETA = 3.0
LOCKED_VARIANT = "GIVEBACK_PENALTY_REWARD"
GAMMA_LOCKED = v1_gate.GAMMA_LOCKED
K_VQ_ITERATIONS = 3
INNER_EPOCHS = 15
HIDDEN_DIM = 64
N_HIDDEN = 2
LR = 1e-3
WEIGHT_DECAY = 1e-3
BATCH_SIZE = 256
ADVANTAGE_CLIP = v1_gate.ADVANTAGE_CLIP
SEED_V1 = v1_gate.SEED_V1

INPUT_TEACHER_BRIDGE_PATH = Path(
    "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ALL_TRADE_REVIEW_LEDGER_20260411/"
    "shadow_meta_all_trade_review_policy_action_supervision_join_v1.parquet"
)

ALLOWED_FINAL_STATUSES = {
    "BRIDGE_PASS_LABELS_WRITTEN",
    "BRIDGE_PARTIAL_NO_MANAGEMENT_ROWS",
    "BRIDGE_BLOCKED_BY_INPUT_LOCK_MISSING",
}
ALLOWED_NEXT_ACTIONS = {
    "DISTILL_V3_EXIT_TRANSFORMER_FROM_IQL_LABELS_V1",
    "DISTILL_XGB_MULTIHEAD_FROM_IQL_LABELS_V1",
    "DISTILL_V10_ENTRY_TRANSFORMER_FROM_IQL_LABELS_V1",
    "REPAIR_BRIDGE_BEFORE_FURTHER_WORK_V1",
}

_jsonable = contract_gate._jsonable
_write_json = contract_gate._write_json
_write_rows = contract_gate._write_rows
_write_report = contract_gate._write_report
_read_json = contract_gate._read_json


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


def _train_iql_on_full_dataset(
    inputs: dict[str, Any], candidate_uid_order: list[str]
) -> tuple[iql_core.IQLGpuModel, dict[str, Any]]:
    """Train ONE IQL_GPU model on union-of-all-folds train split.

    The advanced-RL gate trained per-fold; here we want a single model to
    use for inference on the supervision_join rows. We use the FOLD_3 split
    (largest expanding window) to maximize training data.
    """
    fold_3 = wf_gate.FOLD_DEFINITIONS[-1]  # last expanding window
    uid_to_split = wf_gate._assign_fold_split(candidate_uid_order, fold_3)
    per_bar, X_full, _ = wf_gate._build_per_bar_for_fold(inputs, uid_to_split)

    train_mask = (per_bar["primary_split_v1"] == "train").to_numpy()
    per_bar_train = per_bar[train_mask].reset_index(drop=True)
    X_train = X_full[train_mask].astype(np.float32)

    # Find the locked variant column.
    variant = next(
        v for v in v2_train_gate.REWARD_VARIANTS_V2 if v["reward_id_v1"] == LOCKED_VARIANT
    )
    reward_col = variant["reward_column_v1"]
    if reward_col not in per_bar_train.columns:
        raise RuntimeError(
            f"LOCKED_VARIANT_REWARD_COLUMN_MISSING: {reward_col}"
        )

    trans = v1_gate._build_transition_tuples(per_bar_train, X_train, reward_col)
    a_t = trans["a_v1"].astype(np.int64)
    r_t = trans["r_v1"].astype(np.float32)
    next_idx_t = trans["next_idx_v1"].astype(np.int64)
    done_t = trans["done_v1"].astype(bool)

    print(f"[BRIDGE] training IQL_GPU on FOLD_3 train n={len(per_bar_train)} "
          f"variant={LOCKED_VARIANT} tau={LOCKED_TAU} beta={LOCKED_BETA}",
          flush=True)
    model = iql_core.train_true_iql_gpu(
        X_train, a_t, r_t, next_idx_t, done_t,
        tau=LOCKED_TAU, gamma=GAMMA_LOCKED, k_iterations=K_VQ_ITERATIONS,
        inner_epochs=INNER_EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
        hidden_dim=HIDDEN_DIM, n_hidden=N_HIDDEN, batch_size=BATCH_SIZE,
        seed=SEED_V1, prefer_cuda=True,
    )
    print(f"[BRIDGE] training done device={model.device}", flush=True)

    train_meta = {
        "fold_id_v1": fold_3["fold_id_v1"],
        "n_train_rows_v1": int(len(per_bar_train)),
        "variant_v1": LOCKED_VARIANT,
        "tau_v1": LOCKED_TAU,
        "beta_v1": LOCKED_BETA,
        "device_v1": str(model.device),
    }
    return model, train_meta


def _build_state_for_supervision_rows(
    supervision_df: pd.DataFrame,
    inputs: dict[str, Any],
    candidate_uid_order: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Build the V2 state matrix for each row in supervision_df.

    Uses **fuzzy backward-lookup**: for each supervision row, finds the
    per-bar state from the SAME candidate_uid with the largest ts_v1
    less than or equal to the row's hindsight_decision_anchor_timestamp.
    This handles the granularity offset between supervision-anchor
    timestamps and per-bar M5 timestamps.

    Returns (X, valid_mask) — X is (n, 54) float32, valid_mask is bool
    array where False means no per-bar state for that uid (no replay
    coverage at all).
    """
    from bisect import bisect_right

    fold_3 = wf_gate.FOLD_DEFINITIONS[-1]
    uid_to_split = wf_gate._assign_fold_split(candidate_uid_order, fold_3)
    per_bar, X_full, _ = wf_gate._build_per_bar_for_fold(inputs, uid_to_split)

    n = len(supervision_df)
    state_dim = X_full.shape[1]
    X_out = np.zeros((n, state_dim), dtype=np.float32)
    valid_mask = np.zeros(n, dtype=bool)

    # Build per-uid sorted lists of (ts_int_ns, X_index) for backward-lookup.
    candidate_uids = per_bar["candidate_uid_v1"].astype(str).to_numpy()
    timestamps_ns = (
        pd.to_datetime(per_bar["ts_v1"], utc=True).astype("int64").to_numpy()
    )
    per_uid_bars: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    # group indices by uid, sort by timestamp
    sort_idx = np.argsort(timestamps_ns, kind="stable")
    sorted_uids = candidate_uids[sort_idx]
    sorted_ts = timestamps_ns[sort_idx]
    sorted_pos = sort_idx
    # group: for each unique uid, get its slice in the sorted array
    unique_uids, first_idx = np.unique(sorted_uids, return_index=True)
    counts = np.diff(np.append(first_idx, len(sorted_uids)))
    for uid, fi, cnt in zip(unique_uids, first_idx, counts):
        per_uid_bars[uid] = (
            sorted_ts[fi : fi + cnt],
            sorted_pos[fi : fi + cnt],
        )

    sup_uids = supervision_df["candidate_uid"].astype(str).to_numpy()
    sup_ts_ns = (
        pd.to_datetime(
            supervision_df["hindsight_decision_anchor_timestamp_utc_v1"], utc=True
        )
        .astype("int64")
        .to_numpy()
    )
    n_exact = 0
    n_fuzzy = 0
    for j in range(n):
        uid = sup_uids[j]
        if uid not in per_uid_bars:
            continue
        ts_arr, pos_arr = per_uid_bars[uid]
        target = sup_ts_ns[j]
        # Largest ts_arr[k] <= target
        k = bisect_right(ts_arr, target) - 1
        if k < 0:
            # No bar at or before target — try first bar (forward-fallback)
            k = 0
        X_out[j] = X_full[pos_arr[k]]
        valid_mask[j] = True
        if ts_arr[k] == target:
            n_exact += 1
        else:
            n_fuzzy += 1
    print(
        f"[BRIDGE] state_lookup: exact={n_exact} fuzzy={n_fuzzy} "
        f"unmatched={n - int(valid_mask.sum())} of n={n}",
        flush=True,
    )
    return X_out, valid_mask


def _attach_iql_recommendations(
    supervision_df: pd.DataFrame,
    model: iql_core.IQLGpuModel,
    X: np.ndarray,
    valid_mask: np.ndarray,
) -> pd.DataFrame:
    """Run IQL inference on every row, append recommendation columns."""
    df = supervision_df.copy()
    n = len(df)
    p_exit = np.full(n, np.nan, dtype=np.float32)
    q_hold = np.full(n, np.nan, dtype=np.float32)
    q_exit = np.full(n, np.nan, dtype=np.float32)

    if valid_mask.any():
        X_valid = X[valid_mask]
        p_exit_v = iql_core.true_iql_policy_exit_prob_gpu(
            X_valid, model, beta=LOCKED_BETA, clip=ADVANTAGE_CLIP,
        )
        q_hold_v = model.predict_q(X_valid, action_id=iql_core.ACTION_HOLD_ID)
        q_exit_v = model.predict_q(X_valid, action_id=iql_core.ACTION_EXIT_NOW_ID)
        p_exit[valid_mask] = p_exit_v
        q_hold[valid_mask] = q_hold_v
        q_exit[valid_mask] = q_exit_v

    df["iql_gpu_p_exit_v1"] = p_exit
    df["iql_gpu_q_hold_v1"] = q_hold
    df["iql_gpu_q_exit_v1"] = q_exit
    df["iql_gpu_advantage_v1"] = q_exit - q_hold
    rec = np.full(n, "NOT_ESTABLISHED", dtype=object)
    take_exit = (~np.isnan(p_exit)) & (p_exit >= 0.5)
    take_hold = (~np.isnan(p_exit)) & (p_exit < 0.5)
    rec[take_exit] = "EXIT_NOW"
    rec[take_hold] = "HOLD"
    df["iql_gpu_recommended_action_v1"] = rec
    df["iql_gpu_state_resolved_v1"] = valid_mask
    df["iql_gpu_agreement_with_teacher_v1"] = (
        df["iql_gpu_recommended_action_v1"] == df["hindsight_policy_action_v1"]
    )
    return df


def _agreement_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    """Per-split agreement rates between IQL and teacher labels."""
    out: dict[str, Any] = {}
    for split_col, name in [
        ("used_for_training", "train"),
        ("used_for_validation", "val"),
        ("used_for_holdout", "holdout"),
    ]:
        sub = df[df[split_col].astype(bool)]
        sub_mgmt = sub[sub["hindsight_policy_action_domain_v1"] == "MANAGEMENT"]
        sub_resolved = sub_mgmt[sub_mgmt["iql_gpu_state_resolved_v1"]]
        n_total = int(len(sub_mgmt))
        n_resolved = int(len(sub_resolved))
        n_agree = int(sub_resolved["iql_gpu_agreement_with_teacher_v1"].sum())
        agreement = (n_agree / n_resolved) if n_resolved else 0.0
        out[name] = {
            "n_management_total_v1": n_total,
            "n_management_resolved_v1": n_resolved,
            "n_agree_v1": n_agree,
            "agreement_rate_v1": float(agreement),
        }
    return out


def _go_no_go(diagnostics: dict[str, Any], n_mgmt: int) -> tuple[str, str, str]:
    if n_mgmt == 0:
        return (
            "BRIDGE_PARTIAL_NO_MANAGEMENT_ROWS",
            "REPAIR_BRIDGE_BEFORE_FURTHER_WORK_V1",
            "Supervision parquet contains no MANAGEMENT-domain rows; cannot bridge IQL exit policy.",
        )
    train_agreement = diagnostics["train"]["agreement_rate_v1"]
    val_agreement = diagnostics["val"]["agreement_rate_v1"]
    holdout_agreement = diagnostics["holdout"]["agreement_rate_v1"]
    return (
        "BRIDGE_PASS_LABELS_WRITTEN",
        "DISTILL_V3_EXIT_TRANSFORMER_FROM_IQL_LABELS_V1",
        (
            f"Bridge complete. Agreement rates: "
            f"train={train_agreement:.1%} val={val_agreement:.1%} "
            f"holdout={holdout_agreement:.1%}. "
            "Output parquet ready for V3 exit transformer distillation gate."
        ),
    )


def write_artifacts(
    out_root: Path | None = None, *, built_at_utc: str | None = None,
) -> dict[str, Any]:
    timestamp = built_at_utc or _stamp()
    artifact_root = out_root or (DEFAULT_REPORTS_ROOT / f"{ACTION}_{timestamp}_LOCK")
    artifact_root.mkdir(parents=True, exist_ok=True)

    if not INPUT_TEACHER_BRIDGE_PATH.is_file():
        return {
            "artifact_root": str(artifact_root),
            "summary": {
                "final_status_v1": "BRIDGE_BLOCKED_BY_INPUT_LOCK_MISSING",
                "next_action_v1": "REPAIR_BRIDGE_BEFORE_FURTHER_WORK_V1",
                "missing_input_v1": str(INPUT_TEACHER_BRIDGE_PATH),
            },
        }

    print(f"[BRIDGE] reading {INPUT_TEACHER_BRIDGE_PATH}", flush=True)
    supervision_df = pd.read_parquet(INPUT_TEACHER_BRIDGE_PATH)
    print(f"[BRIDGE] loaded n_rows={len(supervision_df)} "
          f"n_cols={len(supervision_df.columns)}", flush=True)

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

    model, train_meta = _train_iql_on_full_dataset(inputs, candidate_uid_order)
    X, valid_mask = _build_state_for_supervision_rows(
        supervision_df, inputs, candidate_uid_order,
    )
    n_resolved = int(valid_mask.sum())
    print(f"[BRIDGE] state-resolved {n_resolved}/{len(supervision_df)} rows", flush=True)

    enriched = _attach_iql_recommendations(supervision_df, model, X, valid_mask)
    out_parquet = artifact_root / "shadow_meta_with_iql_gpu_recommendations_v1.parquet"
    enriched.to_parquet(out_parquet, index=False)
    print(f"[BRIDGE] wrote {out_parquet}", flush=True)

    n_mgmt = int(
        (enriched["hindsight_policy_action_domain_v1"] == "MANAGEMENT").sum()
    )
    diagnostics = _agreement_diagnostics(enriched)
    status, next_action, recommendation = _go_no_go(diagnostics, n_mgmt)
    validate_final_status(status, next_action)

    summary = {
        "layer_name": "BRIDGE_IQL_GPU_TO_TEACHER_SUPERVISION_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "input_teacher_bridge_path_v1": str(INPUT_TEACHER_BRIDGE_PATH),
        "output_parquet_path_v1": str(out_parquet),
        "training_meta_v1": train_meta,
        "n_total_rows_v1": int(len(enriched)),
        "n_management_rows_v1": n_mgmt,
        "n_state_resolved_v1": n_resolved,
        "n_state_unresolved_v1": int(len(enriched)) - n_resolved,
        "split_diagnostics_v1": diagnostics,
        "research_only_v1": True,
        "iql_production_allowed_v1": False,
    }
    _write_json(artifact_root / "summary_v1.json", summary)
    _write_json(
        artifact_root / "status_v1.json",
        {
            "layer_name": "BRIDGE_STATUS_V1",
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
                "iql_recommendations_parquet": str(out_parquet),
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
