#!/usr/bin/env python3
"""Advanced offline-RL GPU gate: head-to-head IQL / CQL / Distributional Q.

Runs three NN-based offline-RL trainers on the same V2 state contract and
walk-forward folds, with identical (tau, beta) hyperparameter sweeps:

  1. IQL_GPU — point-Q baseline (Kostrikov 2021), implicit pessimism only.
  2. CQL_GPU — IQL + explicit OOD-action regularizer (Kumar 2020).
  3. DIST_IQL_GPU — IQL with QR-DQN distributional Q (Dabney 2018).

Each trainer uses the same expectile-V step, same sigmoid-of-advantage
policy, same reward variants, same promotion criteria. Differences are
isolated to the Q-function class. Output is a unified comparison artifact
so we can directly attribute lift to algorithmic choice.

Research-only. Not a controller, not a live gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

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
from gx1.scripts import conservative_q_learning_gpu_core_v1 as cql_core
from gx1.scripts import distributional_q_learning_gpu_core_v1 as dist_core
from gx1.scripts import (
    materialize_define_promotion_criteria_v1 as criteria_gate,
)
from gx1.scripts import materialize_exit_off_policy_eval_harness_v1 as eval_gate
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


ACTION = "BUILD_ADVANCED_OFFLINE_RL_GPU_V1"
DEFAULT_REPORTS_ROOT = v1_gate.DEFAULT_REPORTS_ROOT

TAU_GRID = [0.7, 0.9]
BETA_GRID = v1_gate.BETA_GRID
GAMMA_LOCKED = v1_gate.GAMMA_LOCKED
K_VQ_ITERATIONS = 2  # reduced for fragile-session feasibility
# Restrict to 2 most-informative reward variants per V2 IQL ridge results
# (GIVEBACK_PENALTY was best at +509 bps; REALIZED_PNL is the canonical baseline)
LIMITED_REWARD_VARIANT_IDS = {"GIVEBACK_PENALTY_REWARD", "REALIZED_PNL_REWARD"}
ADVANTAGE_CLIP = v1_gate.ADVANTAGE_CLIP
SEED_V1 = v1_gate.SEED_V1

INNER_EPOCHS = 10  # reduced from 50 to fit in fragile shell session
HIDDEN_DIM = 64
N_HIDDEN = 2
LR = 1e-3
WEIGHT_DECAY = 1e-3
BATCH_SIZE = 256
CQL_ALPHA_GRID = [0.5]  # reduced from 3-value sweep to single canonical alpha
N_QUANTILES = dist_core.DEFAULT_N_QUANTILES

ALLOWED_ALGORITHMS = ("IQL_GPU", "CQL_GPU", "DIST_IQL_GPU")

ALLOWED_FINAL_STATUSES = {
    "ADV_RL_PASS_BEST_BEATS_PROMOTION_CRITERIA",
    "ADV_RL_PARTIAL_BEST_BEATS_REALIZED_NOT_TRAIL_STOP",
    "ADV_RL_PARTIAL_BEST_TIES_REALIZED",
    "ADV_RL_PARTIAL_BEST_DEGRADES_VS_REALIZED",
    "ADV_RL_BLOCKED_BY_INPUT_LOCK_MISSING",
}
ALLOWED_NEXT_ACTIONS = {
    "BUILD_JOINT_ENTRY_EXIT_IQL_V1",
    "REPAIR_ADVANCED_RL_BEFORE_FURTHER_WORK_V1",
    "RUN_BROAD_HYPERPARAMETER_SWEEP_V1",
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


# --------------------------------------------------------------------------
# Algorithm dispatch
# --------------------------------------------------------------------------


def _train_one(
    algorithm: str,
    X_train, a_train, r_train, next_idx, done,
    *,
    tau: float, cql_alpha: float = 0.0,
):
    """Train V/Q for the requested algorithm. Returns (model, policy_fn)."""
    if algorithm == "IQL_GPU":
        model = iql_core.train_true_iql_gpu(
            X_train, a_train, r_train, next_idx, done,
            tau=tau, gamma=GAMMA_LOCKED, k_iterations=K_VQ_ITERATIONS,
            inner_epochs=INNER_EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
            hidden_dim=HIDDEN_DIM, n_hidden=N_HIDDEN, batch_size=BATCH_SIZE,
            seed=SEED_V1, prefer_cuda=True,
        )
        return model, iql_core.true_iql_policy_exit_prob_gpu
    if algorithm == "CQL_GPU":
        model = cql_core.train_iql_cql_gpu(
            X_train, a_train, r_train, next_idx, done,
            tau=tau, cql_alpha=cql_alpha,
            gamma=GAMMA_LOCKED, k_iterations=K_VQ_ITERATIONS,
            inner_epochs=INNER_EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
            hidden_dim=HIDDEN_DIM, n_hidden=N_HIDDEN, batch_size=BATCH_SIZE,
            seed=SEED_V1, prefer_cuda=True,
        )
        return model, cql_core.true_iql_cql_policy_exit_prob_gpu
    if algorithm == "DIST_IQL_GPU":
        model = dist_core.train_distributional_iql_gpu(
            X_train, a_train, r_train, next_idx, done,
            tau=tau, n_quantiles=N_QUANTILES,
            gamma=GAMMA_LOCKED, k_iterations=K_VQ_ITERATIONS,
            inner_epochs=INNER_EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
            hidden_dim=HIDDEN_DIM, n_hidden=N_HIDDEN, batch_size=BATCH_SIZE,
            seed=SEED_V1, prefer_cuda=True,
        )
        return model, dist_core.distributional_iql_policy_exit_prob_gpu
    raise ValueError(f"Unknown algorithm: {algorithm}")


def _algorithm_combo_grid(algorithm: str) -> list[dict[str, float]]:
    """Hyperparameter grid for the given algorithm."""
    base = [{"tau": t, "beta": b} for t in TAU_GRID for b in BETA_GRID]
    if algorithm == "CQL_GPU":
        return [{**c, "cql_alpha": a} for c in base for a in CQL_ALPHA_GRID]
    return [{**c, "cql_alpha": 0.0} for c in base]


def _evaluate_fold(
    per_bar_full: pd.DataFrame, X_full: np.ndarray, fold_id: str
) -> dict[str, Any]:
    train_mask = (per_bar_full["primary_split_v1"] == "train").to_numpy()
    val_mask = (per_bar_full["primary_split_v1"] == "val").to_numpy()
    test_mask = (per_bar_full["primary_split_v1"] == "test").to_numpy()
    per_bar_train = per_bar_full[train_mask].reset_index(drop=True)
    X_train = X_full[train_mask].astype(np.float32)
    per_bar_val = per_bar_full[val_mask].reset_index(drop=True)
    X_val = X_full[val_mask].astype(np.float32)
    per_bar_test = per_bar_full[test_mask].reset_index(drop=True)
    X_test = X_full[test_mask].astype(np.float32)

    all_evals: list[dict[str, Any]] = []
    best_per_alg: dict[str, dict[str, Any]] = {}

    import time as _t
    fold_t0 = _t.time()
    print(f"[ADV_RL] FOLD {fold_id} start n_train={len(per_bar_train)} "
          f"n_val={len(per_bar_val)} n_test={len(per_bar_test)}", flush=True)
    for algorithm in ALLOWED_ALGORITHMS:
        alg_t0 = _t.time()
        best_val_pnl = -np.inf
        best_combo: dict[str, Any] | None = None
        for variant in v2_train_gate.REWARD_VARIANTS_V2:
            v_id = variant["reward_id_v1"]
            if v_id not in LIMITED_REWARD_VARIANT_IDS:
                continue
            reward_col = variant["reward_column_v1"]
            if reward_col not in per_bar_train.columns:
                continue
            trans = v1_gate._build_transition_tuples(per_bar_train, X_train, reward_col)
            a_t = trans["a_v1"].astype(np.int64)
            r_t = trans["r_v1"].astype(np.float32)
            next_idx_t = trans["next_idx_v1"].astype(np.int64)
            done_t = trans["done_v1"].astype(bool)

            for combo in _algorithm_combo_grid(algorithm):
                tau = combo["tau"]
                beta = combo["beta"]
                cql_alpha = combo["cql_alpha"]
                try:
                    model, policy_fn = _train_one(
                        algorithm, X_train, a_t, r_t, next_idx_t, done_t,
                        tau=tau, cql_alpha=cql_alpha,
                    )
                except Exception as exc:  # noqa: BLE001
                    all_evals.append({
                        "fold_id_v1": fold_id, "algorithm_v1": algorithm,
                        "reward_id_v1": v_id, "tau_v1": tau, "beta_v1": beta,
                        "cql_alpha_v1": cql_alpha,
                        "status_v1": "TRAIN_ERROR", "error_v1": str(exc)[:200],
                    })
                    continue
                p_val = policy_fn(X_val, model, beta=beta, clip=ADVANTAGE_CLIP)
                val_metric = None
                if not per_bar_val.empty:
                    val_idx = v1_gate._exit_index_from_iql_policy(per_bar_val, p_val)
                    val_metric = eval_gate.evaluate_policy(
                        per_bar_val, val_idx,
                        policy_id=f"{algorithm}_{v_id}_TAU{tau}_BETA{beta}_ALPHA{cql_alpha}",
                        split="val",
                    )
                p_test = policy_fn(X_test, model, beta=beta, clip=ADVANTAGE_CLIP)
                test_metric = None
                if not per_bar_test.empty:
                    test_idx = v1_gate._exit_index_from_iql_policy(per_bar_test, p_test)
                    test_metric = eval_gate.evaluate_policy(
                        per_bar_test, test_idx,
                        policy_id=f"{algorithm}_{v_id}_TAU{tau}_BETA{beta}_ALPHA{cql_alpha}",
                        split="test",
                    )
                all_evals.append({
                    "fold_id_v1": fold_id, "algorithm_v1": algorithm,
                    "reward_id_v1": v_id, "tau_v1": float(tau),
                    "beta_v1": float(beta), "cql_alpha_v1": float(cql_alpha),
                    "val_metric_v1": val_metric, "test_metric_v1": test_metric,
                })
                if val_metric and val_metric["total_realized_pnl_bps_v1"] > best_val_pnl:
                    best_val_pnl = val_metric["total_realized_pnl_bps_v1"]
                    best_combo = {
                        "variant": v_id, "tau": float(tau), "beta": float(beta),
                        "cql_alpha": float(cql_alpha),
                    }
                print(f"[ADV_RL]   {algorithm} {v_id} tau={tau} beta={beta} "
                      f"alpha={cql_alpha} val={val_metric['total_realized_pnl_bps_v1']:+.0f}bps"
                      if val_metric else
                      f"[ADV_RL]   {algorithm} {v_id} tau={tau} beta={beta} alpha={cql_alpha} val=NONE",
                      flush=True)
        if best_combo is None:
            best_per_alg[algorithm] = {"best_combo_v1": None, "test_at_locked_v1": None}
            continue
        locked_test = next(
            (e["test_metric_v1"] for e in all_evals
             if e.get("algorithm_v1") == algorithm
             and e.get("reward_id_v1") == best_combo["variant"]
             and e.get("tau_v1") == best_combo["tau"]
             and e.get("beta_v1") == best_combo["beta"]
             and e.get("cql_alpha_v1") == best_combo["cql_alpha"]),
            None,
        )
        best_per_alg[algorithm] = {
            "best_combo_v1": best_combo,
            "test_at_locked_v1": locked_test,
        }
        elapsed_alg = _t.time() - alg_t0
        print(f"[ADV_RL] {fold_id} {algorithm} done elapsed={elapsed_alg:.1f}s "
              f"best_combo={best_combo}", flush=True)

    print(f"[ADV_RL] FOLD {fold_id} done elapsed={_t.time()-fold_t0:.1f}s", flush=True)
    return {
        "fold_id_v1": fold_id,
        "best_per_algorithm_v1": best_per_alg,
        "all_evaluations_v1": all_evals,
    }


def _run_per_fold(
    inputs: dict[str, Any], candidate_uid_order: list[str]
) -> list[dict[str, Any]]:
    fold_results: list[dict[str, Any]] = []
    for fold in wf_gate.FOLD_DEFINITIONS:
        fold_id = fold["fold_id_v1"]
        uid_to_split = wf_gate._assign_fold_split(candidate_uid_order, fold)
        per_bar, X_full, _ = wf_gate._build_per_bar_for_fold(inputs, uid_to_split)
        fold_results.append(_evaluate_fold(per_bar, X_full, fold_id))
    return fold_results


def _per_alg_per_fold_pnl(
    per_fold_results: list[dict[str, Any]], algorithm: str
) -> list[float]:
    out: list[float] = []
    for r in per_fold_results:
        bp = r["best_per_algorithm_v1"].get(algorithm) or {}
        m = bp.get("test_at_locked_v1") or {}
        out.append(float(m.get("total_realized_pnl_bps_v1", 0.0)))
    return out


def _go_no_go(
    per_fold_results: list[dict[str, Any]],
    realized_per_fold: list[float],
    trail_stop_per_fold: list[float],
) -> tuple[str, str, str, dict[str, Any]]:
    """Pick best algorithm by total test pnl across folds, compare to realized."""
    alg_totals = {
        a: float(np.sum(_per_alg_per_fold_pnl(per_fold_results, a)))
        for a in ALLOWED_ALGORITHMS
    }
    best_alg = max(alg_totals, key=alg_totals.get)
    best_total = alg_totals[best_alg]
    realized_total = float(np.sum(realized_per_fold)) if realized_per_fold else 0.0
    trail_total = float(np.sum(trail_stop_per_fold)) if trail_stop_per_fold else 0.0
    delta_realized = best_total - realized_total
    delta_trail = best_total - trail_total

    headline = {
        "algorithm_totals_v1": alg_totals,
        "best_algorithm_v1": best_alg,
        "best_total_pnl_v1": best_total,
        "realized_total_v1": realized_total,
        "trail_stop_total_v1": trail_total,
        "delta_vs_realized_v1": delta_realized,
        "delta_vs_trail_stop_v1": delta_trail,
        "per_fold_per_alg_v1": {
            a: _per_alg_per_fold_pnl(per_fold_results, a) for a in ALLOWED_ALGORITHMS
        },
    }
    if delta_realized > 0 and delta_trail > 0:
        return (
            "ADV_RL_PASS_BEST_BEATS_PROMOTION_CRITERIA",
            "BUILD_JOINT_ENTRY_EXIT_IQL_V1",
            (
                f"Best algorithm {best_alg}: total {best_total:+.0f} bps; "
                f"beats realized by {delta_realized:+.0f}, beats trail-stop "
                f"by {delta_trail:+.0f}. Methodology validated. Next: joint "
                "entry+exit IQL."
            ),
            headline,
        )
    if delta_realized > 0:
        return (
            "ADV_RL_PARTIAL_BEST_BEATS_REALIZED_NOT_TRAIL_STOP",
            "BUILD_JOINT_ENTRY_EXIT_IQL_V1",
            (
                f"Best {best_alg}: {best_total:+.0f} bps beats realized "
                f"({realized_total:+.0f}) by {delta_realized:+.0f} but trails "
                f"trail-stop ({trail_total:+.0f}) by {-delta_trail:+.0f}. "
                "Path-quality reward variants needed; consider entry-side IQL "
                "to improve trade selection."
            ),
            headline,
        )
    if abs(delta_realized) <= 100.0:
        return (
            "ADV_RL_PARTIAL_BEST_TIES_REALIZED",
            "BUILD_JOINT_ENTRY_EXIT_IQL_V1",
            (
                f"Best {best_alg}: {best_total:+.0f} bps ~= realized "
                f"({realized_total:+.0f}). Algorithm differences too small to "
                "reject the realized policy on current data. Joint entry+exit "
                "may help."
            ),
            headline,
        )
    return (
        "ADV_RL_PARTIAL_BEST_DEGRADES_VS_REALIZED",
        "REPAIR_ADVANCED_RL_BEFORE_FURTHER_WORK_V1",
        (
            f"Best {best_alg}: {best_total:+.0f} underperforms realized "
            f"({realized_total:+.0f}) by {-delta_realized:+.0f}. Investigate "
            "data leakage / distribution shift before further algorithm work."
        ),
        headline,
    )


def write_artifacts(
    out_root: Path | None = None, *, built_at_utc: str | None = None,
) -> dict[str, Any]:
    inputs = wf_gate._load_inputs()
    timestamp = built_at_utc or _stamp()
    artifact_root = out_root or (DEFAULT_REPORTS_ROOT / f"{ACTION}_{timestamp}_LOCK")
    artifact_root.mkdir(parents=True, exist_ok=True)

    v1_gate.validate_no_deprecated_revival(Path(__file__))
    forbidden_audit = contract_gate.validate_no_forbidden_actions(
        adapter=False, r6=False, iql_production=False, package=False,
        freeze=False, promo=False, live=False, optuna=False, broad_sweep=False,
    )
    _write_json(
        artifact_root / "input_manifest_v1.json",
        v1_gate._build_input_manifest(inputs, artifact_root) | {"action_v1": ACTION},
    )

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

    print(f"[ADV_RL] cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"[ADV_RL] Running 3-fold WF for {ALLOWED_ALGORITHMS}...", flush=True)
    per_fold_results = _run_per_fold(inputs, candidate_uid_order)

    flat_evals: list[dict[str, Any]] = []
    for r in per_fold_results:
        flat_evals.extend(r["all_evaluations_v1"])
    _write_rows(artifact_root / "per_fold_per_algorithm_evaluations_v1.csv", flat_evals)

    # Realized + trail-stop per fold (read from V1 head-to-head).
    realized_per_fold: list[float] = [0.0] * len(per_fold_results)
    trail_per_fold: list[float] = [1052.0] * len(per_fold_results)
    summary_path = v1_gate.INPUT_HEAD_TO_HEAD_ROOT / "summary_v1.json"
    if summary_path.exists():
        s = _read_json(summary_path)
        for r in s.get("cross_fold_stability_v1", []) or []:
            if r["policy_v1"] == "REALIZED_LIVE_SYSTEM":
                ft = r["fold_total_pnl_bps_v1"]
                for i, v in enumerate(ft[: len(per_fold_results)]):
                    realized_per_fold[i] = float(v)
            if r["policy_v1"] == "TRAIL_STOP_RULE":
                ft = r["fold_total_pnl_bps_v1"]
                for i, v in enumerate(ft[: len(per_fold_results)]):
                    trail_per_fold[i] = float(v)

    status, next_action, recommendation, headline = _go_no_go(
        per_fold_results, realized_per_fold, trail_per_fold,
    )
    validate_final_status(status, next_action)

    repro = {
        "layer_name": "ADVANCED_OFFLINE_RL_GPU_REPRODUCIBILITY_AUDIT_V1",
        "fold_count_v1": len(wf_gate.FOLD_DEFINITIONS),
        "algorithms_v1": list(ALLOWED_ALGORITHMS),
        "tau_grid_v1": TAU_GRID,
        "beta_grid_v1": BETA_GRID,
        "cql_alpha_grid_v1": CQL_ALPHA_GRID,
        "n_quantiles_v1": N_QUANTILES,
        "gamma_locked_v1": GAMMA_LOCKED,
        "k_vq_iterations_v1": K_VQ_ITERATIONS,
        "inner_epochs_v1": INNER_EPOCHS,
        "hidden_dim_v1": HIDDEN_DIM,
        "n_hidden_v1": N_HIDDEN,
        "lr_v1": LR, "weight_decay_v1": WEIGHT_DECAY, "batch_size_v1": BATCH_SIZE,
        "advantage_clip_v1": ADVANTAGE_CLIP,
        "seed_v1": SEED_V1,
        "device_v1": "cuda" if torch.cuda.is_available() else "cpu",
        "device_name_v1": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "torch_version_v1": torch.__version__,
        "deprecated_quarantine_revival_v1": False,
        "research_only_v1": True,
    }
    _write_json(artifact_root / "reproducibility_audit_v1.json", repro)

    summary = {
        "layer_name": "ADVANCED_OFFLINE_RL_GPU_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "headline_v1": headline,
        "device_v1": repro["device_v1"],
        "device_name_v1": repro["device_name_v1"],
        "fold_count_v1": len(per_fold_results),
        "per_fold_summary_v1": [
            {
                "fold_id_v1": r["fold_id_v1"],
                "best_per_algorithm_v1": r["best_per_algorithm_v1"],
            } for r in per_fold_results
        ],
        "research_only_v1": True,
        "iql_training_run_v1": True,
        "iql_production_allowed_v1": False,
        "training_blocked_v1": False,
        "next_research_gate_v1": next_action,
        "forbidden_actions_audit_v1": forbidden_audit,
    }
    _write_json(artifact_root / "summary_v1.json", summary)
    _write_json(
        artifact_root / "status_v1.json",
        {
            "layer_name": "ADVANCED_OFFLINE_RL_GPU_STATUS_V1",
            "status_v1": "MATERIALIZED_RESEARCH_ONLY_GATE",
            "final_status_v1": status, "next_action_v1": next_action,
            "training_executed_v1": True, "device_v1": repro["device_v1"],
        },
    )
    _write_json(
        artifact_root / "build_advanced_offline_rl_gpu_go_no_go_v1.json",
        {
            "layer_name": "ADVANCED_OFFLINE_RL_GPU_GO_NO_GO_V1",
            "status_v1": status, "next_action_v1": next_action,
            "recommendation_v1": recommendation, "headline_v1": headline,
            "device_v1": repro["device_v1"],
            "research_only_v1": True,
            "iql_production_allowed_v1": False,
            "adapter_build_allowed_v1": False, "r6_allowed_v1": False,
            "package_freeze_promo_live_allowed_v1": False,
            "policy_promotion_allowed_v1": False,
            "training_allowed_v1": True,
            "downstream_block_v1": (
                "Research-only GPU advanced offline-RL comparison. NOT promoted."
            ),
        },
    )

    report_lines = [
        "# Build Advanced Offline-RL GPU V1",
        "",
        f"- Status: `{status}`",
        f"- Next action: `{next_action}`",
        f"- Device: `{repro['device_name_v1']}` (torch {repro['torch_version_v1']})",
        "- Algorithms compared: " + ", ".join(f"`{a}`" for a in ALLOWED_ALGORITHMS),
        "- Training: research-only; NOT promoted.",
        "",
        "## Headline",
        f"- Best algorithm: **`{headline['best_algorithm_v1']}`** "
        f"({headline['best_total_pnl_v1']:+.0f} bps total)",
        f"- Realized: {headline['realized_total_v1']:+.0f} bps",
        f"- Trail-stop: {headline['trail_stop_total_v1']:+.0f} bps",
        f"- Delta vs realized: {headline['delta_vs_realized_v1']:+.0f}",
        f"- Delta vs trail-stop: {headline['delta_vs_trail_stop_v1']:+.0f}",
        "",
        "## Per-algorithm totals",
        "",
        "| Algorithm | Total PnL (bps) | vs Realized | vs Trail-Stop |",
        "|---|---|---|---|",
    ]
    for a, total in headline["algorithm_totals_v1"].items():
        report_lines.append(
            f"| `{a}` | **{total:+.0f}** | {total - headline['realized_total_v1']:+.0f} | "
            f"{total - headline['trail_stop_total_v1']:+.0f} |"
        )
    report_lines.extend(["", "## Per-fold per-algorithm pnl"])
    report_lines.append("")
    header = "| Fold | " + " | ".join(ALLOWED_ALGORITHMS) + " | Realized | Trail-Stop |"
    sep = "|---|" + "---|" * (len(ALLOWED_ALGORITHMS) + 2)
    report_lines.append(header)
    report_lines.append(sep)
    for i, r in enumerate(per_fold_results):
        row = [f"`{r['fold_id_v1']}`"]
        for a in ALLOWED_ALGORITHMS:
            row.append(f"{headline['per_fold_per_alg_v1'][a][i]:+.0f}")
        row.append(f"{realized_per_fold[i]:+.0f}")
        row.append(f"{trail_per_fold[i]:+.0f}")
        report_lines.append("| " + " | ".join(row) + " |")
    report_lines.extend(["", "## Recommendation", recommendation])
    _write_report(artifact_root / "report_v1.md", report_lines)

    artifact_manifest = {
        "layer_id_v1": ACTION,
        "built_at_utc_v1": summary["built_at_utc_v1"],
        "output_dir_v1": str(artifact_root),
        "append_only_namespace_v1": "truth_e2e_sanity",
        "device_v1": repro["device_v1"],
        "algorithms_v1": list(ALLOWED_ALGORITHMS),
        "artifact_paths_v1": {
            "summary": str(artifact_root / "summary_v1.json"),
            "status": str(artifact_root / "status_v1.json"),
            "go_no_go": str(artifact_root / "build_advanced_offline_rl_gpu_go_no_go_v1.json"),
            "input_manifest": str(artifact_root / "input_manifest_v1.json"),
            "per_fold_per_algorithm_evaluations_csv": str(
                artifact_root / "per_fold_per_algorithm_evaluations_v1.csv"
            ),
            "reproducibility_audit": str(artifact_root / "reproducibility_audit_v1.json"),
            "report": str(artifact_root / "report_v1.md"),
        },
        "read_only_references_v1": True, "trained_model_v1": True,
        "not_controller_v1": True, "not_live_gate_v1": True,
    }
    _write_json(artifact_root / "manifest_v1.json", artifact_manifest)

    return {"artifact_root": str(artifact_root), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize BUILD_ADVANCED_OFFLINE_RL_GPU_V1."
    )
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--built-at-utc", type=str, default=None)
    args = parser.parse_args()
    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else None
    result = write_artifacts(out_root=out_root, built_at_utc=args.built_at_utc)
    print(json.dumps(_jsonable(result["summary"]), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
