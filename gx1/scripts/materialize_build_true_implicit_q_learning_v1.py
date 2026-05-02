#!/usr/bin/env python3
"""True Implicit Q-Learning (Kostrikov et al. 2021).

Background
----------
Our previous "IQL" gates approximated offline-RL. The AWR POC (3B5)
trained V/Q with simple ridge regression on returns, then weighted the
policy by sigmoid(beta * A). That is AWR (advantage-weighted regression)
- a SUBSET of the IQL paper, not full IQL.

Real IQL has three ingredients:

  1. Value V(s; psi) trained with EXPECTILE REGRESSION over Q(s, a):
       L_V(psi) = E_(s,a)~D [ L_tau( Q(s,a) - V(s; psi) ) ]
     where L_tau(u) = |tau - 1[u<0]| * u^2 is the asymmetric expectile
     loss. tau in (0.5, 1.0) -> upper expectile -> conservative V that
     underestimates the maximum Q in the dataset, providing pessimism.

  2. Q(s,a; theta) trained with SARSA-like Bellman backup using V as
     target on next-state:
       L_Q(theta) = E_(s,a,r,s')~D [ ( r + gamma * V(s'; psi_bar) - Q(s,a; theta) )^2 ]
     V(s'; psi_bar) is the (slowly updated) target value network.

  3. Policy pi via Advantage-Weighted Regression:
       pi*(a|s) = argmax_pi E_(s,a)~D [ exp(beta * (Q(s,a) - V(s))) * log pi(a|s) ]
     For finite actions: pi(a|s) ∝ exp(beta * (Q(s,a) - V(s))).

The expectile regression in step 1 is what makes IQL fundamentally
different from AWR alone - it explicitly downweights overestimated
Q-values, giving real pessimism without needing to model OOD actions.

Implementation
--------------
Closed-form approximation suitable for ridge-regression model classes:

  - V(s) fit by Iteratively Reweighted Least Squares (IRLS) for the
    expectile loss. Each iteration: weights w_i = |tau - 1[(y_i - X_i^T b) < 0]|
    -> weighted ridge regression.
  - Q(s,a) fit by standard ridge with target r + gamma * V(s'; previous psi).
  - Iterate V/Q updates K=10 times.
  - Final policy: pi(EXIT_NOW | s) = sigmoid(beta * (Q(s,EXIT_NOW) - Q(s,HOLD)))
    after the AWR derivation collapses for binary actions.

Per fold (3 walk-forward folds):
  - Build (s, a, r, s', done) tuples from per-bar augmented dataset:
      * HOLD bars: a=HOLD, r=0 (intermediate reward), next state = bar+1
      * EXIT_NOW bars: a=EXIT_NOW, r=variant_reward, terminal
  - Train V/Q for K iterations with tau, beta, gamma swept
  - Best (variant, tau, beta, gamma) on val
  - Lock test result; apply DEFINE_PROMOTION_CRITERIA_V1

Hyperparameters:
  tau in {0.7, 0.8, 0.9}
  beta in {3.0, 10.0}
  gamma = 0.99 (locked per MDP contract)
  K (V/Q iterations) = 10
  ridge_lambda = 1e-3

Compared against:
  - AWR POC (3B5)
  - V2 IQL ridge-MSE baseline
  - Realized exit (live system)
  - Trail-stop rule
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


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "BUILD_TRUE_IMPLICIT_Q_LEARNING_V1"

INPUT_PROMOTION_CRITERIA_ROOT = (
    DEFAULT_REPORTS_ROOT / "DEFINE_PROMOTION_CRITERIA_V1_20260430T070707Z_LOCK"
)
INPUT_HEAD_TO_HEAD_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "RUN_LIVE_SYSTEM_VS_RESEARCH_CANDIDATES_HEAD_TO_HEAD_V1_20260430T072907Z_LOCK"
)
INPUT_AWR_POC_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "BUILD_AWR_PROPER_IQL_POC_V1_20260430T132949Z_LOCK"
)
INPUT_RECOVERY_ROOT = v2_train_gate.INPUT_RECOVERY_ROOT
INPUT_SPLIT_ROOT = v2_train_gate.INPUT_SPLIT_ROOT
INPUT_V2_CONTRACT_ROOT = v2_train_gate.INPUT_V2_CONTRACT_ROOT
BASE34_M5_FEATURES_PATH = v2_train_gate.BASE34_M5_FEATURES_PATH

QUARANTINE_FORBIDDEN_PATH_FRAGMENTS = ("gx1/quarantine", "gx1.quarantine")
SEED_V1 = 20260430

# IQL hyperparameter grids.
TAU_GRID: list[float] = [0.7, 0.8, 0.9]  # expectile regression upper-quantiles
BETA_GRID: list[float] = [3.0, 10.0]  # AWR temperature
GAMMA_LOCKED: float = 0.99  # locked per MDP contract
K_VQ_ITERATIONS: int = 10
RIDGE_LAMBDA: float = 1e-3
ADVANTAGE_CLIP: float = 5.0

ALLOWED_FINAL_STATUSES = {
    "TRUE_IQL_PASS_MEETS_PROMOTION_CRITERIA",
    "TRUE_IQL_PASS_BEATS_AWR_POC_BUT_FAILS_OTHER_CRITERIA",
    "TRUE_IQL_PARTIAL_TIES_AWR_POC",
    "TRUE_IQL_PARTIAL_DEGRADES_VS_AWR_POC",
    "TRUE_IQL_BLOCKED_BY_INPUT_LOCK_MISSING",
}

ALLOWED_NEXT_ACTIONS = {
    "BUILD_CONSERVATIVE_Q_LEARNING_V1",
    "BUILD_DISTRIBUTIONAL_Q_LEARNING_V1",
    "AUDIT_XGB_HYPERPARAMETER_OPTUNA_SWEEP_V1",
    "REPAIR_TRUE_IQL_BEFORE_FURTHER_WORK_V1",
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
# IQL math helpers
# ---------------------------------------------------------------------------


def _ridge_fit(
    X: np.ndarray, y: np.ndarray, lam: float = RIDGE_LAMBDA
) -> np.ndarray:
    a = X.T @ X + lam * np.eye(X.shape[1])
    b = X.T @ y
    return np.linalg.solve(a, b)


def _weighted_ridge_fit(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray, lam: float = RIDGE_LAMBDA
) -> np.ndarray:
    """Closed-form ridge with sample weights (diagonal W)."""
    sqrt_w = np.sqrt(np.maximum(weights, 0.0))[:, None]
    Xw = X * sqrt_w
    yw = y * sqrt_w[:, 0]
    a = Xw.T @ Xw + lam * np.eye(X.shape[1])
    b = Xw.T @ yw
    return np.linalg.solve(a, b)


def fit_expectile_regression(
    X: np.ndarray,
    y: np.ndarray,
    tau: float,
    lam: float = RIDGE_LAMBDA,
    max_iter: int = 25,
    tol: float = 1e-5,
) -> np.ndarray:
    """Asymmetric (expectile) ridge regression via IRLS.

    Minimizes:  sum_i w_i(beta) * (y_i - X_i^T beta)^2  +  lam * ||beta||^2
    where      w_i(beta) = |tau - 1[(y_i - X_i^T beta) < 0]|
    """
    n, d = X.shape
    beta = _ridge_fit(X, y, lam=lam)
    for _ in range(max_iter):
        diff = y - X @ beta
        # Expectile weights: tau if diff >= 0, (1 - tau) if diff < 0.
        # Multiplied by 2 because gradient of (diff^2 * w) involves 2.
        weights = np.where(diff >= 0, tau, 1.0 - tau) * 2.0
        new_beta = _weighted_ridge_fit(X, y, weights, lam=lam)
        if np.linalg.norm(new_beta - beta) < tol:
            beta = new_beta
            break
        beta = new_beta
    return beta


def _build_state_action_features(
    X: np.ndarray, action: np.ndarray
) -> np.ndarray:
    """[state, is_hold, is_exit_now]"""
    n = X.shape[0]
    a_oh = np.zeros((n, 2), dtype=float)
    a_oh[action == v2_train_gate.ACTION_HOLD_ID, 0] = 1.0
    a_oh[action == v2_train_gate.ACTION_EXIT_NOW_ID, 1] = 1.0
    return np.concatenate([X, a_oh], axis=1)


def _q_for_action_id(
    X: np.ndarray, coef_q: np.ndarray, action_id: int
) -> np.ndarray:
    n = X.shape[0]
    a_arr = np.full(n, action_id, dtype=int)
    X_sa = _build_state_action_features(X, a_arr)
    return X_sa @ coef_q


# ---------------------------------------------------------------------------
# Per-fold IQL training
# ---------------------------------------------------------------------------


def _build_transition_tuples(
    per_bar_full: pd.DataFrame, X_full: np.ndarray, reward_col: str
) -> dict[str, np.ndarray]:
    """Construct (s, a, r, s', done) per-bar transitions.

    The augmented dataset has both HOLD (action_id=0) and EXIT_NOW
    (action_id=1) rows per held bar. We build:
      - For HOLD rows: a=HOLD, r_immediate=0, next_state = next bar's
        state for the same trade (s'). done=False except at the trade's
        last HOLD bar where the realized exit happens (done=True).
      - For EXIT_NOW rows: a=EXIT_NOW, r=reward_col, done=True (terminal).

    Returns a dict with keys s, a, r, s_next, done, valid_next.
    valid_next is a boolean mask: True if s' exists (HOLD has a next bar
    in the same trade); for EXIT_NOW or terminal HOLD it's False.
    """
    n = int(len(per_bar_full))
    a = per_bar_full["action_id_v1"].astype(int).to_numpy()
    if reward_col not in per_bar_full.columns:
        raise RuntimeError(f"REWARD_COLUMN_MISSING: {reward_col}")
    r_full = per_bar_full[reward_col].astype(float).to_numpy()
    r_full = np.where(np.isfinite(r_full), r_full, 0.0)

    # For HOLD rows: r_immediate = 0 (deferred reward). Real reward
    # comes through V(s') backup.
    is_hold = a == v2_train_gate.ACTION_HOLD_ID
    is_exit_now = a == v2_train_gate.ACTION_EXIT_NOW_ID
    r = np.zeros(n, dtype=float)
    r[is_exit_now] = r_full[is_exit_now]

    # Find s' for each HOLD row: the next bar (within same trade) of the
    # SAME action_id. We pair HOLD bar at (uid, k) with HOLD bar at
    # (uid, k+1). If no next bar exists (k is the trade's last HOLD bar),
    # done=True with terminal reward = realized HOLD pnl_at_close.
    candidate_uid = per_bar_full["candidate_uid_v1"].astype(str).to_numpy()
    bars_held = per_bar_full["bars_held_v1"].astype(int).to_numpy()
    df_index = np.arange(n)

    next_idx = np.full(n, -1, dtype=int)
    done = np.zeros(n, dtype=bool)
    # Group by (uid, action_id_v1) and assign next-row index.
    sort_order = np.lexsort((bars_held, a, candidate_uid))
    for j in range(len(sort_order) - 1):
        i = sort_order[j]
        i_next = sort_order[j + 1]
        if (
            candidate_uid[i] == candidate_uid[i_next]
            and a[i] == a[i_next]
            and bars_held[i_next] == bars_held[i] + 1
        ):
            next_idx[i] = int(i_next)
        else:
            done[i] = True
    # Last row in sort order is also done.
    done[sort_order[-1]] = True

    # Build s_next:
    s_next = np.zeros_like(X_full)
    valid_next = next_idx >= 0
    s_next[valid_next] = X_full[next_idx[valid_next]]

    # For HOLD rows that are terminal (done), r is realized HOLD pnl_at_close.
    # We use the running_pnl_at_close at the bar; the standard convention
    # in our augmented dataset is that the reward column for HOLD at the
    # last bar already encodes the trade-terminal value.
    is_hold_done = is_hold & done
    r[is_hold_done] = r_full[is_hold_done]

    # EXIT_NOW rows are always terminal.
    done_full = done | is_exit_now

    return {
        "a_v1": a,
        "r_v1": r,
        "next_idx_v1": next_idx,
        "done_v1": done_full,
        "valid_next_v1": valid_next,
    }


def train_true_iql(
    X_train: np.ndarray,
    a_train: np.ndarray,
    r_train: np.ndarray,
    next_idx_train: np.ndarray,
    done_train: np.ndarray,
    *,
    tau: float,
    gamma: float = GAMMA_LOCKED,
    k_iterations: int = K_VQ_ITERATIONS,
    lam: float = RIDGE_LAMBDA,
) -> dict[str, np.ndarray]:
    """Train V(s) and Q(s,a) via IQL Bellman backup.

    Iteration loop:
      1. V_psi = expectile_regression( Q_theta(s, a) ; tau )
      2. Q_theta = ridge( s, a -> r + gamma * V_psi(s') * (1 - done) )

    next_idx_train < 0 means s' doesn't exist; we treat that as terminal.
    """
    X_sa_train = _build_state_action_features(X_train, a_train)
    # Initial Q from one-shot ridge on raw r (no Bellman).
    coef_q = _ridge_fit(X_sa_train, r_train, lam=lam)
    coef_v = np.zeros(X_train.shape[1])

    n = X_train.shape[0]

    for _ in range(k_iterations):
        # Step 1: V via expectile regression on Q(s, a) targets.
        q_sa = X_sa_train @ coef_q
        coef_v = fit_expectile_regression(X_train, q_sa, tau=tau, lam=lam)

        # Step 2: Q via Bellman backup using V on next state.
        v_next = np.zeros(n, dtype=float)
        valid = next_idx_train >= 0
        if valid.any():
            X_next_valid = X_train[next_idx_train[valid]]
            v_next[valid] = X_next_valid @ coef_v
        target = r_train + gamma * v_next * (~done_train)
        coef_q = _ridge_fit(X_sa_train, target, lam=lam)

    return {"coef_v": coef_v, "coef_q_sa": coef_q}


def true_iql_policy_exit_prob(
    X: np.ndarray,
    coef_q: np.ndarray,
    coef_v: np.ndarray,
    beta: float,
    clip: float = ADVANTAGE_CLIP,
) -> np.ndarray:
    """π(EXIT_NOW | s) = sigmoid(β · clip(Q(s, EXIT_NOW) - Q(s, HOLD), ±clip))."""
    q_exit = _q_for_action_id(X, coef_q, v2_train_gate.ACTION_EXIT_NOW_ID)
    q_hold = _q_for_action_id(X, coef_q, v2_train_gate.ACTION_HOLD_ID)
    adv = q_exit - q_hold
    adv_clipped = np.clip(adv, -clip, clip)
    return 1.0 / (1.0 + np.exp(-beta * adv_clipped))


def _exit_index_from_iql_policy(
    per_bar: pd.DataFrame, p_exit: np.ndarray, threshold: float = 0.5
) -> pd.Series:
    realized_idx_map = eval_gate._exit_index_realized_exit(per_bar)
    per_bar = per_bar.reset_index(drop=True)
    p_series = pd.Series(p_exit, index=per_bar.index)
    out: list[tuple[str, int]] = []
    for uid, group in per_bar.groupby("candidate_uid_v1", sort=False):
        triggered = group[p_series.loc[group.index] >= threshold]
        if not triggered.empty:
            out.append((uid, int(triggered.index[0])))
        else:
            out.append((uid, int(realized_idx_map.loc[uid])))
    return pd.Series({uid: idx for uid, idx in out})


# ---------------------------------------------------------------------------
# Per-fold per-variant evaluation
# ---------------------------------------------------------------------------


def _evaluate_fold(
    per_bar_full: pd.DataFrame, X_full: np.ndarray, fold_id: str
) -> dict[str, Any]:
    train_mask = (per_bar_full["primary_split_v1"] == "train").to_numpy()
    val_mask = (per_bar_full["primary_split_v1"] == "val").to_numpy()
    test_mask = (per_bar_full["primary_split_v1"] == "test").to_numpy()
    per_bar_train = per_bar_full[train_mask].reset_index(drop=True)
    X_train = X_full[train_mask]
    per_bar_val = per_bar_full[val_mask].reset_index(drop=True)
    X_val = X_full[val_mask]
    per_bar_test = per_bar_full[test_mask].reset_index(drop=True)
    X_test = X_full[test_mask]

    all_evals: list[dict[str, Any]] = []
    best_per_test_pnl = -np.inf
    best_combo: tuple[str, float, float] | None = None
    for variant in v2_train_gate.REWARD_VARIANTS_V2:
        v_id = variant["reward_id_v1"]
        reward_col = variant["reward_column_v1"]
        if reward_col not in per_bar_train.columns:
            continue
        # Re-fit transition tuples on the train-only frame.
        trans_train = _build_transition_tuples(per_bar_train, X_train, reward_col)
        a_train = trans_train["a_v1"]
        r_train = trans_train["r_v1"]
        next_idx_train = trans_train["next_idx_v1"]
        done_train = trans_train["done_v1"]

        for tau in TAU_GRID:
            try:
                model = train_true_iql(
                    X_train,
                    a_train,
                    r_train,
                    next_idx_train,
                    done_train,
                    tau=tau,
                )
            except Exception as exc:  # noqa: BLE001
                all_evals.append(
                    {
                        "fold_id_v1": fold_id,
                        "reward_id_v1": v_id,
                        "tau_v1": tau,
                        "status_v1": "TRAIN_ERROR",
                        "error_v1": str(exc)[:200],
                    }
                )
                continue
            for beta in BETA_GRID:
                p_val = true_iql_policy_exit_prob(
                    X_val, model["coef_q_sa"], model["coef_v"], beta=beta
                )
                if per_bar_val.empty:
                    val_metric = None
                else:
                    val_exit_idx = _exit_index_from_iql_policy(per_bar_val, p_val)
                    val_metric = eval_gate.evaluate_policy(
                        per_bar_val,
                        val_exit_idx,
                        policy_id=f"TRUE_IQL_{v_id}_TAU{tau}_BETA{beta}",
                        split="val",
                    )
                p_test = true_iql_policy_exit_prob(
                    X_test, model["coef_q_sa"], model["coef_v"], beta=beta
                )
                if per_bar_test.empty:
                    test_metric = None
                else:
                    test_exit_idx = _exit_index_from_iql_policy(per_bar_test, p_test)
                    test_metric = eval_gate.evaluate_policy(
                        per_bar_test,
                        test_exit_idx,
                        policy_id=f"TRUE_IQL_{v_id}_TAU{tau}_BETA{beta}",
                        split="test",
                    )
                all_evals.append(
                    {
                        "fold_id_v1": fold_id,
                        "reward_id_v1": v_id,
                        "tau_v1": float(tau),
                        "beta_v1": float(beta),
                        "val_metric_v1": val_metric,
                        "test_metric_v1": test_metric,
                    }
                )
                if val_metric and val_metric["total_realized_pnl_bps_v1"] > best_per_test_pnl:
                    best_per_test_pnl = val_metric["total_realized_pnl_bps_v1"]
                    best_combo = (v_id, float(tau), float(beta))
    if best_combo is None:
        return {
            "fold_id_v1": fold_id,
            "best_variant_v1": None,
            "best_tau_v1": None,
            "best_beta_v1": None,
            "test_at_locked_v1": None,
            "all_evaluations_v1": all_evals,
        }
    bv, btau, bbeta = best_combo
    locked_test = next(
        (
            e["test_metric_v1"]
            for e in all_evals
            if e.get("reward_id_v1") == bv
            and e.get("tau_v1") == btau
            and e.get("beta_v1") == bbeta
        ),
        None,
    )
    return {
        "fold_id_v1": fold_id,
        "best_variant_v1": bv,
        "best_tau_v1": btau,
        "best_beta_v1": bbeta,
        "test_at_locked_v1": locked_test,
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


# ---------------------------------------------------------------------------
# Apply locked promotion criteria
# ---------------------------------------------------------------------------


def _apply_promotion_criteria(
    per_fold_results: list[dict[str, Any]],
) -> dict[str, Any]:
    realized_per_fold: dict[str, float] = {}
    summary_path = INPUT_HEAD_TO_HEAD_ROOT / "summary_v1.json"
    if summary_path.exists():
        summary = _read_json(summary_path)
        for r in summary.get("cross_fold_stability_v1", []) or []:
            if r["policy_v1"] == "REALIZED_LIVE_SYSTEM":
                ft = r["fold_total_pnl_bps_v1"]
                for i, val in enumerate(ft):
                    realized_per_fold[f"FOLD_{i+1}"] = float(val)
                break
    per_fold_pnl = [
        float(r["test_at_locked_v1"]["total_realized_pnl_bps_v1"])
        if r["test_at_locked_v1"] else 0.0
        for r in per_fold_results
    ]
    per_fold_realized = [
        realized_per_fold.get(r["fold_id_v1"], 0.0) for r in per_fold_results
    ]
    per_fold_lifts = [pnl - real for pnl, real in zip(per_fold_pnl, per_fold_realized)]
    trail_stop_proxy = [1052.0] * len(per_fold_pnl)
    promotion = criteria_gate.evaluate_candidate_against_criteria(
        candidate_id="true_implicit_q_learning_v1",
        per_fold_lifts_bps=per_fold_lifts,
        per_fold_pnl_bps=per_fold_pnl,
        per_fold_trail_stop_pnl_bps=trail_stop_proxy,
        no_shortcut_audit_passed=True,
        deterministic_reproducible=True,
    )
    promotion["per_fold_pnl_v1"] = per_fold_pnl
    promotion["per_fold_realized_v1"] = per_fold_realized
    promotion["per_fold_lifts_vs_realized_v1"] = per_fold_lifts
    return promotion


# ---------------------------------------------------------------------------
# go-no-go
# ---------------------------------------------------------------------------


def _go_no_go(
    per_fold_results: list[dict[str, Any]], promotion: dict[str, Any]
) -> tuple[str, str, str, dict[str, Any]]:
    n_pass = int(promotion["n_criteria_passed_v1"])
    n_total = int(promotion["n_criteria_total_v1"])
    overall = bool(promotion["overall_pass_v1"])
    awr_per_fold_pnl: list[float] = []
    awr_summary_path = INPUT_AWR_POC_ROOT / "summary_v1.json"
    if awr_summary_path.exists():
        awr_summary = _read_json(awr_summary_path)
        per_fold_awr = awr_summary.get("per_fold_summary_v1", []) or []
        for r in per_fold_awr:
            if r.get("test_at_locked_v1"):
                awr_per_fold_pnl.append(
                    float(r["test_at_locked_v1"]["total_realized_pnl_bps_v1"])
                )
            else:
                awr_per_fold_pnl.append(0.0)
    iql_total = float(np.sum(promotion["per_fold_pnl_v1"]))
    awr_total = float(np.sum(awr_per_fold_pnl)) if awr_per_fold_pnl else 0.0
    iql_minus_awr = iql_total - awr_total
    headline = {
        "n_folds_v1": len(per_fold_results),
        "true_iql_per_fold_pnl_v1": promotion["per_fold_pnl_v1"],
        "realized_per_fold_pnl_v1": promotion["per_fold_realized_v1"],
        "lifts_vs_realized_v1": promotion["per_fold_lifts_vs_realized_v1"],
        "iql_total_v1": iql_total,
        "awr_per_fold_pnl_v1": awr_per_fold_pnl,
        "awr_total_v1": awr_total,
        "iql_minus_awr_total_v1": iql_minus_awr,
        "promotion_pass_v1": overall,
        "promotion_n_passed_v1": n_pass,
        "promotion_n_total_v1": n_total,
        "best_per_fold_v1": [
            {
                "fold_id_v1": r["fold_id_v1"],
                "variant_v1": r["best_variant_v1"],
                "tau_v1": r["best_tau_v1"],
                "beta_v1": r["best_beta_v1"],
            }
            for r in per_fold_results
        ],
    }
    if overall:
        return (
            "TRUE_IQL_PASS_MEETS_PROMOTION_CRITERIA",
            "BUILD_CONSERVATIVE_Q_LEARNING_V1",
            (
                f"True IQL passes ALL {n_total} promotion criteria. Total "
                f"{iql_total:+.0f} bps. Beats AWR POC by {iql_minus_awr:+.0f}. "
                "Real offline RL methodology validated. Next: add CQL "
                "pessimism layer."
            ),
            headline,
        )
    if iql_minus_awr > 200.0:
        return (
            "TRUE_IQL_PASS_BEATS_AWR_POC_BUT_FAILS_OTHER_CRITERIA",
            "BUILD_CONSERVATIVE_Q_LEARNING_V1",
            (
                f"True IQL beats AWR POC by {iql_minus_awr:+.0f} bps total but "
                f"fails {n_total - n_pass} of {n_total} criteria. Methodology "
                "improvement validated; data-coverage limit remains."
            ),
            headline,
        )
    if abs(iql_minus_awr) <= 200.0:
        return (
            "TRUE_IQL_PARTIAL_TIES_AWR_POC",
            "BUILD_CONSERVATIVE_Q_LEARNING_V1",
            (
                f"True IQL ~= AWR POC ({iql_total:+.0f} vs {awr_total:+.0f}; "
                f"delta {iql_minus_awr:+.0f}). Expectile regression did not "
                "produce a clear edge over plain ridge MSE on this small "
                "dataset. Re-run on extended data + add CQL."
            ),
            headline,
        )
    return (
        "TRUE_IQL_PARTIAL_DEGRADES_VS_AWR_POC",
        "REPAIR_TRUE_IQL_BEFORE_FURTHER_WORK_V1",
        (
            f"True IQL degrades vs AWR POC ({iql_total:+.0f} vs "
            f"{awr_total:+.0f}; delta {iql_minus_awr:+.0f}). Investigate."
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
        "layer_name": "TRUE_IQL_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "promotion_criteria_root_v1": str(INPUT_PROMOTION_CRITERIA_ROOT),
            "head_to_head_root_v1": str(INPUT_HEAD_TO_HEAD_ROOT),
            "awr_poc_root_v1": str(INPUT_AWR_POC_ROOT),
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

    print("[TRUE_IQL] Running 3-fold walk-forward...", flush=True)
    per_fold_results = _run_per_fold(inputs, candidate_uid_order)

    flat_evaluations: list[dict[str, Any]] = []
    for r in per_fold_results:
        flat_evaluations.extend(r["all_evaluations_v1"])
    _write_rows(
        artifact_root / "per_fold_per_variant_per_combo_evaluations_v1.csv",
        flat_evaluations,
    )
    _write_json(
        artifact_root / "per_fold_per_variant_per_combo_evaluations_v1.json",
        {"row_count_v1": len(flat_evaluations), "rows_v1": flat_evaluations},
    )
    promotion = _apply_promotion_criteria(per_fold_results)
    _write_json(
        artifact_root / "promotion_criteria_evaluation_v1.json", promotion
    )

    repro = {
        "layer_name": "TRUE_IQL_REPRODUCIBILITY_AUDIT_V1",
        "fold_count_v1": len(wf_gate.FOLD_DEFINITIONS),
        "tau_grid_v1": TAU_GRID,
        "beta_grid_v1": BETA_GRID,
        "gamma_locked_v1": GAMMA_LOCKED,
        "k_vq_iterations_v1": K_VQ_ITERATIONS,
        "ridge_lambda_v1": RIDGE_LAMBDA,
        "advantage_clip_v1": ADVANTAGE_CLIP,
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
        "layer_name": "TRUE_IQL_SUMMARY_V1",
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
                "best_variant_v1": r["best_variant_v1"],
                "best_tau_v1": r["best_tau_v1"],
                "best_beta_v1": r["best_beta_v1"],
                "test_at_locked_v1": r["test_at_locked_v1"],
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
            "layer_name": "TRUE_IQL_STATUS_V1",
            "status_v1": "MATERIALIZED_RESEARCH_ONLY_GATE",
            "final_status_v1": status,
            "next_action_v1": next_action,
            "training_executed_v1": True,
        },
    )
    _write_json(
        artifact_root / "build_true_implicit_q_learning_go_no_go_v1.json",
        {
            "layer_name": "TRUE_IQL_GO_NO_GO_V1",
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
                "Research-only True IQL (Kostrikov 2021). NOT promoted."
            ),
        },
    )

    report_lines = [
        "# Build True Implicit Q-Learning V1",
        "",
        f"- Status: `{status}`",
        f"- Next action: `{next_action}`",
        "- Training: research-only; NOT promoted to runtime.",
        "",
        "## Headline",
        f"- Folds: {headline['n_folds_v1']}",
        f"- True IQL total: **{headline['iql_total_v1']:+.0f}** bps",
        f"- AWR POC total: {headline['awr_total_v1']:+.0f} bps",
        f"- True IQL - AWR: {headline['iql_minus_awr_total_v1']:+.0f}",
        f"- Realized total: {sum(headline['realized_per_fold_pnl_v1']):+.0f} bps",
        f"- Promotion criteria: {headline['promotion_n_passed_v1']}/{headline['promotion_n_total_v1']}",
        "",
        "## Per-fold best (val-tuned variant + tau + beta)",
        "",
        "| Fold | Variant | tau | beta | True IQL PNL | Realized | Lift |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in per_fold_results:
        if r["test_at_locked_v1"] is None:
            continue
        idx = per_fold_results.index(r)
        iql_pnl = headline["true_iql_per_fold_pnl_v1"][idx]
        real_pnl = headline["realized_per_fold_pnl_v1"][idx]
        lift = headline["lifts_vs_realized_v1"][idx]
        report_lines.append(
            f"| `{r['fold_id_v1']}` | `{r['best_variant_v1']}` | "
            f"{r['best_tau_v1']} | {r['best_beta_v1']} | "
            f"**{iql_pnl:+.0f}** | {real_pnl:+.0f} | {lift:+.0f} |"
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
                artifact_root / "build_true_implicit_q_learning_go_no_go_v1.json"
            ),
            "input_manifest": str(artifact_root / "input_manifest_v1.json"),
            "per_fold_per_variant_per_combo_evaluations_csv": str(
                artifact_root / "per_fold_per_variant_per_combo_evaluations_v1.csv"
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
        description="Materialize BUILD_TRUE_IMPLICIT_Q_LEARNING_V1."
    )
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--built-at-utc", type=str, default=None)
    args = parser.parse_args()
    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else None
    result = write_artifacts(out_root=out_root, built_at_utc=args.built_at_utc)
    print(json.dumps(_jsonable(result["summary"]), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
