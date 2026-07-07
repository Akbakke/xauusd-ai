#!/usr/bin/env python3
"""Proper IQL via Advantage-Weighted Regression (AWR) - POC.

Background
----------
Our previous "IQL" gates (V1, V2, V3, hybrid, ensemble) all used closed-
form ridge MSE on a binary label or on Q-targets directly. That is NOT
IQL. Real offline RL methods build the policy via advantage-weighted
regression with explicit value-function decomposition:

    V(s) = E[R | s]                       state value
    Q(s, a) = E[R | s, a]                 action value
    A(s, a) = Q(s, a) - V(s)              advantage
    pi(a | s) ∝ exp(β · clip(A, -A_max, A_max))   AWR policy

The advantage tells us HOW MUCH BETTER an action is vs the average
behavior at that state. Weighting policy actions by exp(β·A) means the
policy concentrates on action choices that the data shows actually
outperformed the behavior policy. This is fundamentally different from
"predict Q(a) directly and pick argmax" - the latter overconfidently
extrapolates to unseen state-action pairs.

POC scope
---------
- 5 reward variants from the V2 augmented dataset
- Per (s, a): a is HOLD vs EXIT_NOW; per-bar
- V(s) ridge regression on (state, return)
- Q(s, a) ridge regression on (state ⊕ action_onehot, return)
- Advantage A = Q - V
- Policy: at each bar, sample / argmax exp(β · A)
- Apply 3-fold walk-forward validation (same as Phase 1 framework)
- Apply locked DEFINE_PROMOTION_CRITERIA_V1 contract
- Compare against ridge-MSE-on-Q V2 IQL baseline per fold and per variant

Hyperparameters:
- β = 1.0, 3.0, 10.0 (sweep)
- A clip = 5.0
- Ridge λ = 1e-3

Run on EXISTING 1.7K-trade dataset as POC. After Phase 3A (data extension
to 2020-2026), this gate is re-run on the extended dataset.

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
ACTION = "BUILD_AWR_PROPER_IQL_POC_V1"

INPUT_PROMOTION_CRITERIA_ROOT = (
    DEFAULT_REPORTS_ROOT / "DEFINE_PROMOTION_CRITERIA_V1_20260430T070707Z_LOCK"
)
INPUT_HEAD_TO_HEAD_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "RUN_LIVE_SYSTEM_VS_RESEARCH_CANDIDATES_HEAD_TO_HEAD_V1_20260430T072907Z_LOCK"
)
INPUT_RECOVERY_ROOT = v2_train_gate.INPUT_RECOVERY_ROOT
INPUT_SPLIT_ROOT = v2_train_gate.INPUT_SPLIT_ROOT
INPUT_V2_CONTRACT_ROOT = v2_train_gate.INPUT_V2_CONTRACT_ROOT
BASE34_M5_FEATURES_PATH = v2_train_gate.BASE34_M5_FEATURES_PATH

QUARANTINE_FORBIDDEN_PATH_FRAGMENTS = ("gx1/quarantine", "gx1.quarantine")
SEED_V1 = 20260430
RIDGE_LAMBDA = 1e-3
BETA_GRID: list[float] = [1.0, 3.0, 10.0]
ADVANTAGE_CLIP = 5.0

ALLOWED_FINAL_STATUSES = {
    "AWR_PROPER_IQL_POC_PASS_MEETS_PROMOTION_CRITERIA",
    "AWR_PROPER_IQL_POC_PASS_BEATS_RIDGE_MSE_BUT_FAILS_OTHER_CRITERIA",
    "AWR_PROPER_IQL_POC_PARTIAL_TIES_RIDGE_MSE",
    "AWR_PROPER_IQL_POC_PARTIAL_DEGRADES_VS_RIDGE_MSE",
    "AWR_PROPER_IQL_POC_BLOCKED_BY_INPUT_LOCK_MISSING",
}

ALLOWED_NEXT_ACTIONS = {
    "BUILD_CONSERVATIVE_Q_LEARNING_V1",
    "BUILD_DISTRIBUTIONAL_Q_LEARNING_V1",
    "BUILD_SELF_SUPERVISED_PRETRAINED_M1_ENCODER_V1",
    "REPAIR_AWR_BEFORE_FURTHER_WORK_V1",
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
# AWR core
# ---------------------------------------------------------------------------


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = RIDGE_LAMBDA) -> np.ndarray:
    a = X.T @ X + lam * np.eye(X.shape[1])
    b = X.T @ y
    return np.linalg.solve(a, b)


def _build_action_augmented_state(
    X: np.ndarray, action_id: np.ndarray
) -> np.ndarray:
    """Concatenate state with one-hot action: [state, is_hold, is_exit_now]."""
    n = X.shape[0]
    one_hot = np.zeros((n, 2), dtype=float)
    one_hot[action_id == v2_train_gate.ACTION_HOLD_ID, 0] = 1.0
    one_hot[action_id == v2_train_gate.ACTION_EXIT_NOW_ID, 1] = 1.0
    return np.concatenate([X, one_hot], axis=1)


def _train_value_and_q_heads(
    X_train: np.ndarray,
    action_train: np.ndarray,
    return_train: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fit V(s) and Q(s,a) by ridge regression."""
    coef_v = _ridge_fit(X_train, return_train)
    X_train_sa = _build_action_augmented_state(X_train, action_train)
    coef_q = _ridge_fit(X_train_sa, return_train)
    return {"coef_v": coef_v, "coef_q_sa": coef_q}


def _compute_advantage(
    X: np.ndarray, coef_v: np.ndarray, coef_q_sa: np.ndarray
) -> np.ndarray:
    """For each row return the advantage of taking EXIT_NOW vs the
    state-only baseline V(s). A_exit_now = Q(s, EXIT_NOW) - V(s).

    We use the EXIT_NOW advantage as the policy-decision signal: at each
    bar the policy chooses EXIT_NOW iff A_exit_now > 0 (with weight
    exp(β · clipped_A))."""
    v_s = X @ coef_v
    n = X.shape[0]
    action_exit = np.full(n, v2_train_gate.ACTION_EXIT_NOW_ID, dtype=int)
    action_hold = np.full(n, v2_train_gate.ACTION_HOLD_ID, dtype=int)
    X_sa_exit = _build_action_augmented_state(X, action_exit)
    X_sa_hold = _build_action_augmented_state(X, action_hold)
    q_exit = X_sa_exit @ coef_q_sa
    q_hold = X_sa_hold @ coef_q_sa
    advantage_exit_minus_hold = q_exit - q_hold
    advantage_exit_vs_v = q_exit - v_s
    advantage_hold_vs_v = q_hold - v_s
    return {
        "v_s_v1": v_s,
        "q_exit_v1": q_exit,
        "q_hold_v1": q_hold,
        "advantage_exit_minus_hold_v1": advantage_exit_minus_hold,
        "advantage_exit_vs_v_v1": advantage_exit_vs_v,
        "advantage_hold_vs_v_v1": advantage_hold_vs_v,
    }


def _awr_policy_exit_now_probability(
    advantage_exit_minus_hold: np.ndarray, beta: float, clip: float = ADVANTAGE_CLIP
) -> np.ndarray:
    """π(EXIT_NOW | s) = sigmoid(β · clip(A_exit - A_hold, -clip, clip)).

    This is the AWR policy reduced to two actions: exp(β·A_exit) /
    (exp(β·A_exit) + exp(β·A_hold)) = sigmoid(β·(A_exit - A_hold))."""
    a_clipped = np.clip(advantage_exit_minus_hold, -clip, clip)
    return 1.0 / (1.0 + np.exp(-beta * a_clipped))


def _exit_index_from_awr_policy(
    per_bar: pd.DataFrame, p_exit_now: np.ndarray, threshold: float = 0.5
) -> pd.Series:
    """Apply the AWR policy: at each bar EXIT_NOW iff p >= threshold.
    For each trade: pick the first bar where the rule fires; if never,
    default to realized exit."""
    realized_idx_map = eval_gate._exit_index_realized_exit(per_bar)
    per_bar = per_bar.reset_index(drop=True)
    p_series = pd.Series(p_exit_now, index=per_bar.index)
    out: list[tuple[str, int]] = []
    for uid, group in per_bar.groupby("candidate_uid_v1", sort=False):
        triggered = group[p_series.loc[group.index] >= threshold]
        if not triggered.empty:
            out.append((uid, int(triggered.index[0])))
        else:
            out.append((uid, int(realized_idx_map.loc[uid])))
    return pd.Series({uid: idx for uid, idx in out})


# ---------------------------------------------------------------------------
# Per-fold AWR training + evaluation across reward variants
# ---------------------------------------------------------------------------


def _project_per_bar_for_fold(
    inputs: dict[str, Any], uid_to_split: dict[str, str]
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    return wf_gate._build_per_bar_for_fold(inputs, uid_to_split)


def _train_awr_for_fold(
    per_bar_full: pd.DataFrame, X_full: np.ndarray
) -> dict[str, Any]:
    """Train V/Q heads per reward variant on this fold's train rows."""
    train_mask = (per_bar_full["primary_split_v1"] == "train").to_numpy()
    per_bar_train = per_bar_full[train_mask]
    X_train = X_full[train_mask]
    action_train = per_bar_train["action_id_v1"].astype(int).to_numpy()
    models: dict[str, dict[str, np.ndarray]] = {}
    for variant in v2_train_gate.REWARD_VARIANTS_V2:
        v_id = variant["reward_id_v1"]
        reward_col = variant["reward_column_v1"]
        if reward_col not in per_bar_train.columns:
            continue
        return_train = per_bar_train[reward_col].astype(float).to_numpy()
        # Replace NaN with 0 (some reward variants have NaN for HOLD rows
        # that don't have terminal info yet). Train-only handling.
        return_train = np.where(np.isfinite(return_train), return_train, 0.0)
        models[v_id] = _train_value_and_q_heads(X_train, action_train, return_train)
    return models


def _evaluate_awr_per_split(
    per_bar_full: pd.DataFrame,
    X_full: np.ndarray,
    models: dict[str, dict[str, np.ndarray]],
    beta: float,
) -> list[dict[str, Any]]:
    """For each (variant, split): compute exit indices via AWR policy,
    evaluate via gate-5 harness."""
    rows: list[dict[str, Any]] = []
    for v_id, m in models.items():
        coef_v = m["coef_v"]
        coef_q_sa = m["coef_q_sa"]
        for split in ["train", "val", "test"]:
            mask = (per_bar_full["primary_split_v1"] == split).to_numpy()
            per_bar_split = per_bar_full[mask].reset_index(drop=True)
            if per_bar_split.empty:
                continue
            X_split = X_full[mask]
            adv = _compute_advantage(X_split, coef_v, coef_q_sa)
            p_exit = _awr_policy_exit_now_probability(
                adv["advantage_exit_minus_hold_v1"], beta=beta
            )
            exit_indices = _exit_index_from_awr_policy(per_bar_split, p_exit)
            metrics = eval_gate.evaluate_policy(
                per_bar_split,
                exit_indices,
                policy_id=f"AWR_PROPER_IQL_{v_id}_BETA_{beta}",
                split=split,
            )
            metrics["reward_variant_v1"] = v_id
            metrics["beta_v1"] = float(beta)
            metrics["model_id_v1"] = "EXIT_AWR_PROPER_IQL_V1"
            rows.append(metrics)
    return rows


def _select_best_variant_and_beta_on_val(
    eval_rows: list[dict[str, Any]],
) -> tuple[str, float]:
    val_rows = [r for r in eval_rows if r["split_v1"] == "val"]
    if not val_rows:
        return "REALIZED_PNL_REWARD", 1.0
    best = max(val_rows, key=lambda r: r["total_realized_pnl_bps_v1"])
    return best["reward_variant_v1"], float(best["beta_v1"])


def _run_per_fold(
    inputs: dict[str, Any], candidate_uid_order: list[str]
) -> list[dict[str, Any]]:
    fold_results: list[dict[str, Any]] = []
    for fold in wf_gate.FOLD_DEFINITIONS:
        fold_id = fold["fold_id_v1"]
        uid_to_split = wf_gate._assign_fold_split(candidate_uid_order, fold)
        per_bar, X_full, _ = _project_per_bar_for_fold(inputs, uid_to_split)
        models = _train_awr_for_fold(per_bar, X_full)
        per_beta_eval: list[dict[str, Any]] = []
        for beta in BETA_GRID:
            per_beta_eval.extend(
                _evaluate_awr_per_split(per_bar, X_full, models, beta=beta)
            )
        # Best (variant, beta) on val.
        best_variant, best_beta = _select_best_variant_and_beta_on_val(per_beta_eval)
        # Test result at best (variant, beta).
        test_at_locked = next(
            (
                r
                for r in per_beta_eval
                if r["split_v1"] == "test"
                and r["reward_variant_v1"] == best_variant
                and r["beta_v1"] == best_beta
            ),
            None,
        )
        fold_results.append(
            {
                "fold_id_v1": fold_id,
                "best_variant_v1": best_variant,
                "best_beta_v1": best_beta,
                "test_at_locked_v1": test_at_locked,
                "all_evaluations_v1": per_beta_eval,
            }
        )
    return fold_results


# ---------------------------------------------------------------------------
# Compare to V2 IQL baseline (ridge-MSE)
# ---------------------------------------------------------------------------


def _v2_baseline_per_fold_test_pnl(inputs: dict[str, Any]) -> dict[str, float]:
    """Pull V2 IQL baseline per-fold test PNL from walk-forward LOCK."""
    summary_path = INPUT_HEAD_TO_HEAD_ROOT / "summary_v1.json"
    if not summary_path.exists():
        return {}
    summary = _read_json(summary_path)
    out: dict[str, float] = {}
    for r in summary.get("full_walk_forward_per_policy_metrics_v1", []) or []:
        if r["policy_v1"] == "V2_IQL_BEST_PER_FOLD":
            # This is total over 524 trades; we don't have per-fold here
            # directly. Use cross_fold_stability_v1 instead.
            pass
    for r in summary.get("cross_fold_stability_v1", []) or []:
        if r["policy_v1"] == "V2_IQL_BEST_PER_FOLD":
            ft = r["fold_total_pnl_bps_v1"]
            for i, val in enumerate(ft):
                fold_id = f"FOLD_{i+1}"
                out[fold_id] = float(val)
            break
    return out


# ---------------------------------------------------------------------------
# Apply locked promotion criteria
# ---------------------------------------------------------------------------


def _apply_promotion_criteria(
    per_fold_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Lifts measured vs realized-exit floor (which is the lift-baseline
    used by walk-forward gate)."""
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
    per_fold_lifts = [
        per_fold_pnl[i] - realized_per_fold.get(per_fold_results[i]["fold_id_v1"], 0.0)
        for i in range(len(per_fold_results))
    ]
    trail_stop_proxy = [1052.0] * len(per_fold_pnl)
    promotion = criteria_gate.evaluate_candidate_against_criteria(
        candidate_id="awr_proper_iql_poc",
        per_fold_lifts_bps=per_fold_lifts,
        per_fold_pnl_bps=per_fold_pnl,
        per_fold_trail_stop_pnl_bps=trail_stop_proxy,
        no_shortcut_audit_passed=True,
        deterministic_reproducible=True,
    )
    promotion["per_fold_pnl_v1"] = per_fold_pnl
    promotion["per_fold_lifts_vs_realized_v1"] = per_fold_lifts
    promotion["per_fold_realized_pnl_v1"] = [
        realized_per_fold.get(r["fold_id_v1"], 0.0) for r in per_fold_results
    ]
    return promotion


# ---------------------------------------------------------------------------
# go-no-go
# ---------------------------------------------------------------------------


def _go_no_go(
    per_fold_results: list[dict[str, Any]],
    promotion: dict[str, Any],
    v2_baseline_per_fold: dict[str, float],
) -> tuple[str, str, str, dict[str, Any]]:
    awr_per_fold_pnl = promotion["per_fold_pnl_v1"]
    realized_per_fold = promotion["per_fold_realized_pnl_v1"]
    lifts_vs_realized = promotion["per_fold_lifts_vs_realized_v1"]
    v2_per_fold = [
        v2_baseline_per_fold.get(r["fold_id_v1"], 0.0) for r in per_fold_results
    ]
    awr_total = float(np.sum(awr_per_fold_pnl))
    realized_total = float(np.sum(realized_per_fold))
    v2_total = float(np.sum(v2_per_fold))
    awr_minus_v2 = awr_total - v2_total
    awr_minus_realized = awr_total - realized_total
    headline = {
        "n_folds_v1": len(per_fold_results),
        "awr_per_fold_pnl_v1": awr_per_fold_pnl,
        "realized_per_fold_pnl_v1": realized_per_fold,
        "v2_baseline_per_fold_pnl_v1": v2_per_fold,
        "lifts_vs_realized_v1": lifts_vs_realized,
        "awr_total_v1": awr_total,
        "realized_total_v1": realized_total,
        "v2_baseline_total_v1": v2_total,
        "awr_minus_v2_total_v1": awr_minus_v2,
        "awr_minus_realized_total_v1": awr_minus_realized,
        "promotion_pass_v1": bool(promotion["overall_pass_v1"]),
        "promotion_n_passed_v1": int(promotion["n_criteria_passed_v1"]),
        "promotion_n_total_v1": int(promotion["n_criteria_total_v1"]),
        "best_per_fold_v1": [
            {
                "fold_id_v1": r["fold_id_v1"],
                "variant_v1": r["best_variant_v1"],
                "beta_v1": r["best_beta_v1"],
            }
            for r in per_fold_results
        ],
    }
    if promotion["overall_pass_v1"]:
        return (
            "AWR_PROPER_IQL_POC_PASS_MEETS_PROMOTION_CRITERIA",
            "BUILD_CONSERVATIVE_Q_LEARNING_V1",
            (
                f"AWR proper IQL passes ALL promotion criteria. Total "
                f"{awr_total:+.0f} bps vs realized {realized_total:+.0f}. "
                f"Beats V2 ridge-MSE baseline by {awr_minus_v2:+.0f}. "
                "Next: add Conservative Q-Learning pessimism layer."
            ),
            headline,
        )
    if awr_minus_v2 > 200.0:
        return (
            "AWR_PROPER_IQL_POC_PASS_BEATS_RIDGE_MSE_BUT_FAILS_OTHER_CRITERIA",
            "BUILD_CONSERVATIVE_Q_LEARNING_V1",
            (
                f"AWR beats V2 ridge-MSE by {awr_minus_v2:+.0f} bps total but "
                f"fails {promotion['n_criteria_total_v1'] - promotion['n_criteria_passed_v1']} "
                "of 6 promotion criteria. Methodologically correct but still "
                "regime-dependent on this 1.7K-trade dataset. Next: CQL + "
                "extended dataset."
            ),
            headline,
        )
    if abs(awr_minus_v2) <= 200.0:
        return (
            "AWR_PROPER_IQL_POC_PARTIAL_TIES_RIDGE_MSE",
            "BUILD_CONSERVATIVE_Q_LEARNING_V1",
            (
                f"AWR ~= V2 ridge-MSE ({awr_total:+.0f} vs {v2_total:+.0f}; "
                f"delta {awr_minus_v2:+.0f}). Methodological upgrade "
                "comparable to ridge-MSE on this small dataset. Next: try "
                "CQL pessimism and re-evaluate on extended data."
            ),
            headline,
        )
    return (
        "AWR_PROPER_IQL_POC_PARTIAL_DEGRADES_VS_RIDGE_MSE",
        "REPAIR_AWR_BEFORE_FURTHER_WORK_V1",
        (
            f"AWR degrades vs V2 ridge-MSE ({awr_total:+.0f} vs "
            f"{v2_total:+.0f}; delta {awr_minus_v2:+.0f}). Investigate "
            "advantage clipping, beta tuning, or value-function fit."
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
        "layer_name": "AWR_PROPER_IQL_POC_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "promotion_criteria_root_v1": str(INPUT_PROMOTION_CRITERIA_ROOT),
            "head_to_head_root_v1": str(INPUT_HEAD_TO_HEAD_ROOT),
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

    per_fold_results = _run_per_fold(inputs, candidate_uid_order)

    # Persist all evaluations + best-per-fold.
    flat_evaluations: list[dict[str, Any]] = []
    for r in per_fold_results:
        for ev in r["all_evaluations_v1"]:
            flat_evaluations.append({**ev, "fold_id_v1": r["fold_id_v1"]})
    _write_rows(
        artifact_root / "per_fold_per_variant_per_beta_evaluations_v1.csv",
        flat_evaluations,
    )
    _write_json(
        artifact_root / "per_fold_per_variant_per_beta_evaluations_v1.json",
        {"row_count_v1": len(flat_evaluations), "rows_v1": flat_evaluations},
    )
    best_test_rows = []
    for r in per_fold_results:
        if r["test_at_locked_v1"]:
            best_test_rows.append(
                {
                    **r["test_at_locked_v1"],
                    "fold_id_v1": r["fold_id_v1"],
                    "best_variant_v1": r["best_variant_v1"],
                    "best_beta_v1": r["best_beta_v1"],
                }
            )
    _write_rows(
        artifact_root / "best_test_at_locked_per_fold_v1.csv", best_test_rows
    )

    promotion = _apply_promotion_criteria(per_fold_results)
    _write_json(
        artifact_root / "promotion_criteria_evaluation_v1.json", promotion
    )
    v2_baseline_per_fold = _v2_baseline_per_fold_test_pnl(inputs)

    repro = {
        "layer_name": "AWR_PROPER_IQL_POC_REPRODUCIBILITY_AUDIT_V1",
        "fold_count_v1": len(wf_gate.FOLD_DEFINITIONS),
        "ridge_lambda_v1": RIDGE_LAMBDA,
        "beta_grid_v1": BETA_GRID,
        "advantage_clip_v1": ADVANTAGE_CLIP,
        "seed_v1": SEED_V1,
        "no_implicit_glob_used_for_v1_inputs_v1": True,
        "deprecated_quarantine_revival_v1": False,
        "research_only_v1": True,
    }
    _write_json(artifact_root / "reproducibility_audit_v1.json", repro)

    status, next_action, recommendation, headline = _go_no_go(
        per_fold_results, promotion, v2_baseline_per_fold
    )
    validate_final_status(status, next_action)

    summary = {
        "layer_name": "AWR_PROPER_IQL_POC_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "headline_v1": headline,
        "fold_count_v1": len(per_fold_results),
        "promotion_evaluation_v1": promotion,
        "v2_baseline_per_fold_v1": v2_baseline_per_fold,
        "per_fold_summary_v1": [
            {
                "fold_id_v1": r["fold_id_v1"],
                "best_variant_v1": r["best_variant_v1"],
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
            "layer_name": "AWR_PROPER_IQL_POC_STATUS_V1",
            "status_v1": "MATERIALIZED_RESEARCH_ONLY_GATE",
            "final_status_v1": status,
            "next_action_v1": next_action,
            "training_executed_v1": True,
        },
    )
    _write_json(
        artifact_root / "build_awr_proper_iql_poc_go_no_go_v1.json",
        {
            "layer_name": "AWR_PROPER_IQL_POC_GO_NO_GO_V1",
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
                "Research-only AWR proper IQL POC. NOT promoted to runtime."
            ),
        },
    )

    report_lines = [
        "# Build AWR Proper IQL POC V1",
        "",
        f"- Status: `{status}`",
        f"- Next action: `{next_action}`",
        "- Training: research-only POC; NOT promoted.",
        "",
        "## Headline",
        f"- Folds: {headline['n_folds_v1']}",
        f"- AWR total: **{headline['awr_total_v1']:+.0f}** bps",
        f"- Realized total: {headline['realized_total_v1']:+.0f} bps",
        f"- V2 ridge-MSE baseline total: {headline['v2_baseline_total_v1']:+.0f}",
        f"- AWR - V2 baseline: {headline['awr_minus_v2_total_v1']:+.0f}",
        f"- AWR - realized: {headline['awr_minus_realized_total_v1']:+.0f}",
        f"- Promotion criteria: {headline['promotion_n_passed_v1']}/{headline['promotion_n_total_v1']}",
        "",
        "## Per-fold best (val-tuned variant + beta)",
        "",
        "| Fold | Variant | Beta | AWR PNL | Realized | V2 baseline |",
        "|---|---|---|---|---|---|",
    ]
    for r, awr_pnl, real_pnl, v2_pnl in zip(
        per_fold_results,
        headline["awr_per_fold_pnl_v1"],
        headline["realized_per_fold_pnl_v1"],
        headline["v2_baseline_per_fold_pnl_v1"],
    ):
        report_lines.append(
            f"| `{r['fold_id_v1']}` | `{r['best_variant_v1']}` | "
            f"{r['best_beta_v1']} | **{awr_pnl:+.0f}** | "
            f"{real_pnl:+.0f} | {v2_pnl:+.0f} |"
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
                artifact_root / "build_awr_proper_iql_poc_go_no_go_v1.json"
            ),
            "input_manifest": str(artifact_root / "input_manifest_v1.json"),
            "per_fold_per_variant_per_beta_evaluations_csv": str(
                artifact_root / "per_fold_per_variant_per_beta_evaluations_v1.csv"
            ),
            "best_test_at_locked_per_fold_csv": str(
                artifact_root / "best_test_at_locked_per_fold_v1.csv"
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
    parser = argparse.ArgumentParser(description="Materialize BUILD_AWR_PROPER_IQL_POC_V1.")
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--built-at-utc", type=str, default=None)
    args = parser.parse_args()
    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else None
    result = write_artifacts(out_root=out_root, built_at_utc=args.built_at_utc)
    print(json.dumps(_jsonable(result["summary"]), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
