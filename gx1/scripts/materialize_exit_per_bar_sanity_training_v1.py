#!/usr/bin/env python3
"""First contextual-IQL sanity training gate for exit-side HOLD/EXIT_NOW.

This is gate 6 of 6 in the exit-IQL pre-train dependency graph. It is the
first gate that actually trains a policy on the augmented dataset.

Training approach (deliberately conservative for first sanity):

  - Closed-form ridge regression of Q-values per action. Two heads:
    Q(state, HOLD) and Q(state, EXIT_NOW).
  - Targets: for HOLD samples, target = the realized terminal pnl of the
    trade (Monte-Carlo backup, simpler than full Bellman expectile). For
    EXIT_NOW samples, target = the bar's pnl-at-close (counterfactual exit
    pnl already locked in the augmented dataset).
  - State vector: 9 normalized features chosen from the 18 HAVE features in
    the gate-2 schema. Train-only normalization via z-score and log1p.
  - Reward variant: REALIZED_PNL_REWARD locked here as the primary; the
    other four trainable variants are reserved for a follow-up sensitivity
    gate.
  - Train split only for fitting. Val and test for off-policy evaluation
    via the gate-5 harness.
  - Policy execution: argmax(Q_HOLD, Q_EXIT_NOW) at each bar. First
    EXIT_NOW per trade determines the exit; otherwise realized exit fires.

The gate then runs the locked baselines through the same harness on the
same splits, plus the trained IQL policy, and computes a comparator table.

Audits include training-no-shortcut, train-only-fit, val/test off-policy
isolation, and standard bookkeeping.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.scripts import materialize_build_iql_offline_data_contract_research_only_v1 as contract_gate
from gx1.scripts import materialize_exit_hold_exit_now_mdp_reward_contract_v1 as mdp_gate
from gx1.scripts import materialize_exit_off_policy_eval_harness_v1 as eval_gate


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "EXIT_PER_BAR_SANITY_TRAINING_V1"

INPUT_EVAL_HARNESS_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "EXIT_OFF_POLICY_EVAL_HARNESS_V1_20260429T154407Z_LOCK"
)
INPUT_SPLIT_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "EXIT_PER_BAR_SPLIT_AND_LEAKAGE_AUDIT_V1_20260429T141227Z_LOCK"
)
INPUT_MDP_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "EXIT_HOLD_EXIT_NOW_MDP_REWARD_CONTRACT_V1_20260429T103326Z_LOCK"
)

QUARANTINE_FORBIDDEN_PATH_FRAGMENTS = ("gx1/quarantine", "gx1.quarantine")

ACTION_HOLD_ID = 0
ACTION_EXIT_NOW_ID = 1

REWARD_VARIANT_LOCKED_FOR_PRIMARY = "REALIZED_PNL_REWARD"
RIDGE_LAMBDA = 1e-3
SEED_V1 = 20260429

STATE_FEATURE_NAMES_V1 = [
    "intercept",
    "running_pnl_at_close_z",
    "running_mfe_z",
    "running_mae_z",
    "running_giveback_z",
    "bars_held_log1p_z",
    "atr_bps_now_z",
    "exit_prob_v1_or_sentinel",
    "side_long_indicator",
]

ALLOWED_FINAL_STATUSES = {
    "EXIT_PER_BAR_SANITY_TRAINING_PASS_POLICY_BEATS_REALIZED_AND_TRAIL_STOP",
    "EXIT_PER_BAR_SANITY_TRAINING_PASS_POLICY_BEATS_REALIZED_NOT_TRAIL_STOP",
    "EXIT_PER_BAR_SANITY_TRAINING_PARTIAL_POLICY_TIES_REALIZED",
    "EXIT_PER_BAR_SANITY_TRAINING_PARTIAL_POLICY_UNDERPERFORMS_REALIZED",
    "EXIT_PER_BAR_SANITY_TRAINING_BLOCKED_BY_SAFETY_FAIL",
    "EXIT_PER_BAR_SANITY_TRAINING_BLOCKED_BY_NO_SHORTCUT_FAIL",
}

ALLOWED_NEXT_ACTIONS = {
    "EXIT_PER_BAR_REWARD_VARIANT_SENSITIVITY_V1",
    "DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_V1",
    "REPAIR_EXIT_IQL_TRAINING_BEFORE_VARIANT_SENSITIVITY_V1",
    "HOLD_EXIT_IQL_RESEARCH_UNTIL_DATA_FIXED_V1",
}


# Reuse helpers
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
    roots = [INPUT_EVAL_HARNESS_ROOT, INPUT_SPLIT_ROOT, INPUT_MDP_ROOT]
    validate_explicit_artifact_roots(roots)
    required = {
        "eval_harness_summary": INPUT_EVAL_HARNESS_ROOT / "summary_v1.json",
        "eval_harness_baseline_metrics": INPUT_EVAL_HARNESS_ROOT
        / "baseline_metrics_per_split_v1.json",
        "eval_harness_baselines": INPUT_EVAL_HARNESS_ROOT / "baseline_definitions_v1.json",
        "split_locked_dataset": INPUT_SPLIT_ROOT
        / "split_locked_augmented_dataset_v1.parquet",
        "mdp_summary": INPUT_MDP_ROOT / "summary_v1.json",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise RuntimeError(f"MISSING_REQUIRED_INPUT_LOCKS: {missing}")
    return {
        "required_paths": required,
        "eval_harness_summary": _read_json(required["eval_harness_summary"]),
        "eval_harness_baseline_metrics": _read_json(
            required["eval_harness_baseline_metrics"]
        ),
        "eval_harness_baselines": _read_json(required["eval_harness_baselines"]),
        "mdp_summary": _read_json(required["mdp_summary"]),
    }


# ---------------------------------------------------------------------------
# State matrix
# ---------------------------------------------------------------------------


def _per_bar_view(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (candidate_uid_v1, bars_held_v1) using HOLD action sample."""
    hold = df[df["action_id_v1"] == ACTION_HOLD_ID].copy()
    return hold.sort_values(["candidate_uid_v1", "bars_held_v1"]).reset_index(drop=True)


def _fit_train_normalization(per_bar_train: pd.DataFrame) -> dict[str, Any]:
    norm = {}
    for col, transform in [
        ("running_pnl_at_close_bps_v1", "z"),
        ("running_mfe_bps_v1", "z"),
        ("running_mae_bps_v1", "z"),
        ("running_giveback_from_peak_bps_v1", "z"),
        ("atr_bps_now_v1", "z"),
    ]:
        s = per_bar_train[col].astype(float)
        if transform == "z":
            mean = float(s.mean())
            std = float(s.std(ddof=0)) or 1.0
            norm[col] = {"transform": "z", "mean": mean, "std": std}
    s = per_bar_train["bars_held_v1"].astype(float)
    norm["bars_held_v1"] = {
        "transform": "log1p_z",
        "mean": float(np.log1p(s).mean()),
        "std": float(np.log1p(s).std(ddof=0)) or 1.0,
    }
    norm["atr_bps_now_v1_median_v1"] = float(
        per_bar_train["atr_bps_now_v1"].astype(float).median()
    )
    return norm


def _build_state_matrix(per_bar: pd.DataFrame, norm: dict[str, Any]) -> np.ndarray:
    n = len(per_bar)
    X = np.zeros((n, len(STATE_FEATURE_NAMES_V1)), dtype=float)
    X[:, 0] = 1.0  # intercept

    def zscore(series: pd.Series, col: str) -> np.ndarray:
        cfg = norm[col]
        return ((series.astype(float) - cfg["mean"]) / cfg["std"]).clip(-5.0, 5.0).to_numpy()

    X[:, 1] = zscore(per_bar["running_pnl_at_close_bps_v1"], "running_pnl_at_close_bps_v1")
    X[:, 2] = zscore(per_bar["running_mfe_bps_v1"], "running_mfe_bps_v1")
    X[:, 3] = zscore(per_bar["running_mae_bps_v1"], "running_mae_bps_v1")
    X[:, 4] = zscore(
        per_bar["running_giveback_from_peak_bps_v1"], "running_giveback_from_peak_bps_v1"
    )
    bars_held_log = np.log1p(per_bar["bars_held_v1"].astype(float))
    X[:, 5] = (
        (bars_held_log - norm["bars_held_v1"]["mean"]) / norm["bars_held_v1"]["std"]
    ).clip(-5.0, 5.0).to_numpy()
    atr = per_bar["atr_bps_now_v1"].astype(float).fillna(norm["atr_bps_now_v1_median_v1"])
    X[:, 6] = (
        (atr - norm["atr_bps_now_v1"]["mean"]) / norm["atr_bps_now_v1"]["std"]
    ).clip(-5.0, 5.0).to_numpy()
    # exit_prob: 42% NaN. Sentinel -1.0 marks missing.
    X[:, 7] = per_bar["exit_prob_v1"].astype(float).fillna(-1.0).clip(-1.0, 1.0).to_numpy()
    side_long = (per_bar["side_v1"].astype(str) == "long").astype(float).to_numpy()
    X[:, 8] = side_long

    if not np.isfinite(X).all():
        raise RuntimeError("STATE_MATRIX_HAS_NON_FINITE_VALUES")
    return X


# ---------------------------------------------------------------------------
# Training targets
# ---------------------------------------------------------------------------


def _compute_targets(per_bar: pd.DataFrame, augmented_full: pd.DataFrame) -> pd.DataFrame:
    """For each per-bar (HOLD-row), compute target_HOLD and target_EXIT_NOW.

    target_EXIT_NOW = pnl_at_close at this bar (= EXIT_NOW reward already in
                      augmented dataset for the EXIT_NOW twin row).
    target_HOLD     = the trade's terminal pnl_at_close (Monte-Carlo backup).
    """
    out = per_bar.copy()
    # EXIT_NOW reward at this bar comes from EXIT_NOW twin row of same
    # (candidate_uid, bars_held). We can equivalently use the per-bar HOLD row's
    # running_pnl_at_close_bps_v1 because EXIT_NOW reward formula for
    # REALIZED_PNL_REWARD is exactly running_pnl_at_close at this bar.
    out["target_exit_now_v1"] = out["running_pnl_at_close_bps_v1"].astype(float)

    # Trade terminal pnl = last-bar pnl_at_close per trade
    last_bar_idx = out.groupby("candidate_uid_v1")["bars_held_v1"].idxmax()
    last_bar = out.loc[last_bar_idx, ["candidate_uid_v1", "running_pnl_at_close_bps_v1"]].copy()
    last_bar.rename(
        columns={"running_pnl_at_close_bps_v1": "target_hold_v1"}, inplace=True
    )
    out = out.merge(last_bar, on="candidate_uid_v1", how="left")
    return out


# ---------------------------------------------------------------------------
# Closed-form ridge fit
# ---------------------------------------------------------------------------


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = RIDGE_LAMBDA) -> np.ndarray:
    n_features = X.shape[1]
    a = X.T @ X + lam * np.eye(n_features)
    b = X.T @ y
    return np.linalg.solve(a, b)


def _train_q_heads(
    X_train: np.ndarray, target_hold: np.ndarray, target_exit_now: np.ndarray
) -> dict[str, Any]:
    coef_hold = _ridge_fit(X_train, target_hold)
    coef_exit_now = _ridge_fit(X_train, target_exit_now)
    return {
        "coef_hold_v1": coef_hold.tolist(),
        "coef_exit_now_v1": coef_exit_now.tolist(),
        "feature_names_v1": list(STATE_FEATURE_NAMES_V1),
        "ridge_lambda_v1": RIDGE_LAMBDA,
        "seed_v1": SEED_V1,
        "train_row_count_v1": int(X_train.shape[0]),
    }


# ---------------------------------------------------------------------------
# Policy execution
# ---------------------------------------------------------------------------


def _exit_index_from_iql_policy(
    per_bar: pd.DataFrame, X: np.ndarray, coef_hold: np.ndarray, coef_exit_now: np.ndarray
) -> pd.Series:
    """Apply the trained IQL policy to a per-bar dataframe. For each trade,
    return the row index of the first bar where Q(EXIT_NOW) > Q(HOLD), else
    fall back to realized exit (last bar)."""
    q_hold = X @ coef_hold
    q_exit = X @ coef_exit_now
    pick_exit = q_exit > q_hold
    out = []
    realized_idx_map = eval_gate._exit_index_realized_exit(per_bar)
    per_bar = per_bar.reset_index(drop=True)
    pick_exit = pd.Series(pick_exit, index=per_bar.index)
    for uid, group in per_bar.groupby("candidate_uid_v1", sort=False):
        triggered = group[pick_exit.loc[group.index]]
        if not triggered.empty:
            out.append((uid, triggered.index[0]))
        else:
            out.append((uid, realized_idx_map.loc[uid]))
    return pd.Series({uid: idx for uid, idx in out})


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------


def audit_no_shortcut_at_training_time(state_columns: Sequence[str]) -> dict[str, Any]:
    forbidden = set(mdp_gate.FORBIDDEN_STATE_FIELDS_V1)
    used_raw_cols = {
        "running_pnl_at_close_bps_v1",
        "running_mfe_bps_v1",
        "running_mae_bps_v1",
        "running_giveback_from_peak_bps_v1",
        "bars_held_v1",
        "atr_bps_now_v1",
        "exit_prob_v1",
        "side_v1",
    }
    leak = sorted(used_raw_cols & forbidden)
    if leak:
        raise RuntimeError(f"TRAINING_USES_FORBIDDEN_FIELDS: {leak}")
    return {
        "audit_id_v1": "TRAINING_NO_SHORTCUT_AUDIT_V1",
        "status_v1": "PASS",
        "model_state_columns_v1": list(state_columns),
        "raw_columns_used_v1": sorted(used_raw_cols),
        "forbidden_intersection_v1": leak,
    }


def audit_train_only_normalization(per_bar_full: pd.DataFrame, norm: dict[str, Any]) -> dict[str, Any]:
    """Verify normalization stats are computed on train rows only."""
    train = per_bar_full[per_bar_full["primary_split_v1"] == "train"]
    if len(train) == 0:
        raise RuntimeError("EMPTY_TRAIN_SPLIT_FOR_NORMALIZATION_AUDIT")
    expected_mean = float(train["running_pnl_at_close_bps_v1"].astype(float).mean())
    actual_mean = norm["running_pnl_at_close_bps_v1"]["mean"]
    if abs(actual_mean - expected_mean) > 1e-6:
        raise RuntimeError(
            f"NORMALIZATION_FIT_NOT_TRAIN_ONLY: expected {expected_mean} got {actual_mean}"
        )
    return {
        "audit_id_v1": "TRAIN_ONLY_NORMALIZATION_AUDIT_V1",
        "status_v1": "PASS",
        "checked_field_v1": "running_pnl_at_close_bps_v1",
        "train_mean_v1": expected_mean,
    }


def audit_split_isolation(per_bar: pd.DataFrame) -> dict[str, Any]:
    bad = (
        per_bar.groupby("candidate_uid_v1")["primary_split_v1"]
        .nunique()
        .gt(1)
        .sum()
    )
    if int(bad) > 0:
        raise RuntimeError(
            f"SPLIT_ISOLATION_VIOLATION: {bad} trades span multiple splits"
        )
    return {
        "audit_id_v1": "TRAINING_SPLIT_ISOLATION_AUDIT_V1",
        "status_v1": "PASS",
        "spanning_trade_count_v1": int(bad),
    }


def audit_policy_safety_at_inference(
    per_bar: pd.DataFrame, exit_indices: pd.Series
) -> dict[str, Any]:
    """No selected exit row may have a forbidden state field set in any
    suspicious way - here we just check that the exit-bar indices fall within
    each trade's actual bar range."""
    selected = per_bar.loc[exit_indices.values]
    bars_held_max_per_trade = per_bar.groupby("candidate_uid_v1")["bars_held_v1"].max()
    selected_grouped = selected.set_index("candidate_uid_v1")["bars_held_v1"]
    bad_trades = []
    for uid, sel_bar in selected_grouped.items():
        if sel_bar > bars_held_max_per_trade.loc[uid]:
            bad_trades.append(uid)
    if bad_trades:
        raise RuntimeError(
            f"POLICY_SAFETY_VIOLATION: exit bar exceeds trade length for {len(bad_trades)} trades"
        )
    return {
        "audit_id_v1": "POLICY_SAFETY_AUDIT_V1",
        "status_v1": "PASS",
        "out_of_range_trade_count_v1": 0,
    }


# ---------------------------------------------------------------------------
# Reproducibility / go-no-go
# ---------------------------------------------------------------------------


def _reproducibility_audit(model: dict[str, Any], iql_results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "layer_name": "EXIT_PER_BAR_SANITY_TRAINING_REPRODUCIBILITY_AUDIT_V1",
        "model_v1": "CLOSED_FORM_RIDGE_TWO_HEADS",
        "feature_count_v1": len(model["feature_names_v1"]),
        "ridge_lambda_v1": model["ridge_lambda_v1"],
        "seed_v1": model["seed_v1"],
        "splits_evaluated_v1": [r["split_v1"] for r in iql_results],
        "no_implicit_glob_used_for_v1_inputs_v1": True,
        "deprecated_quarantine_revival_v1": False,
        "research_only_v1": True,
    }


def _go_no_go(
    iql_results: list[dict[str, Any]], baseline_metrics_per_split: dict[str, list[dict[str, Any]]]
) -> tuple[str, str, str]:
    """Decide gate status by comparing IQL test-split PNL to the four key
    baselines: REALIZED_EXIT, ALWAYS_EXIT_NOW_AT_BAR_0, TRAIL_STOP_25_PCT_DD."""
    by_split = {r["split_v1"]: r for r in iql_results}
    iql_test = by_split.get("test")
    if iql_test is None:
        raise RuntimeError("IQL_TEST_RESULT_MISSING")
    baseline_test = {
        b["policy_id_v1"]: b for b in baseline_metrics_per_split.get("test", [])
    }
    realized = baseline_test["REALIZED_EXIT_BASELINE"]["total_realized_pnl_bps_v1"]
    trail_stop = baseline_test["TRAIL_STOP_25_PCT_DD"]["total_realized_pnl_bps_v1"]
    iql_total = iql_test["total_realized_pnl_bps_v1"]
    if iql_total >= trail_stop:
        return (
            "EXIT_PER_BAR_SANITY_TRAINING_PASS_POLICY_BEATS_REALIZED_AND_TRAIL_STOP",
            "EXIT_PER_BAR_REWARD_VARIANT_SENSITIVITY_V1",
            (
                f"IQL test PNL {iql_total:.0f} >= TRAIL_STOP {trail_stop:.0f} >= "
                f"REALIZED {realized:.0f}. Policy adds value above the implementable "
                "rule baseline. Next: reward-variant sensitivity to localize which "
                "objective best captures economic edge."
            ),
        )
    if iql_total > realized:
        return (
            "EXIT_PER_BAR_SANITY_TRAINING_PASS_POLICY_BEATS_REALIZED_NOT_TRAIL_STOP",
            "DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_V1",
            (
                f"IQL test PNL {iql_total:.0f} > REALIZED {realized:.0f} but < "
                f"TRAIL_STOP {trail_stop:.0f}. Policy beats the realized floor but "
                "underperforms the simple trail-stop rule. Next: state-feature "
                "deepening to give IQL more information than the rule has."
            ),
        )
    if abs(iql_total - realized) <= 50.0:
        return (
            "EXIT_PER_BAR_SANITY_TRAINING_PARTIAL_POLICY_TIES_REALIZED",
            "DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_V1",
            (
                f"IQL test PNL {iql_total:.0f} ~= REALIZED {realized:.0f}. Policy "
                "essentially reproduces the realized floor. State features may be "
                "too weak. Next: state-feature deepening."
            ),
        )
    return (
        "EXIT_PER_BAR_SANITY_TRAINING_PARTIAL_POLICY_UNDERPERFORMS_REALIZED",
        "REPAIR_EXIT_IQL_TRAINING_BEFORE_VARIANT_SENSITIVITY_V1",
        (
            f"IQL test PNL {iql_total:.0f} < REALIZED {realized:.0f}. Policy is "
            "actively worse than doing nothing. Investigate state representation, "
            "training target, or reward-variant before any further escalation."
        ),
    )


def _build_input_manifest(inputs: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    files = [
        {"name_v1": name, "path_v1": str(path), "sha256_v1": _file_hash(path)}
        for name, path in inputs["required_paths"].items()
    ]
    return {
        "layer_name": "EXIT_PER_BAR_SANITY_TRAINING_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "eval_harness_root_v1": str(INPUT_EVAL_HARNESS_ROOT),
            "split_root_v1": str(INPUT_SPLIT_ROOT),
            "mdp_root_v1": str(INPUT_MDP_ROOT),
        },
        "files_used_v1": files,
        "immutable_input_status_v1": "HASHED_EXPLICIT_ROOTS_ONLY",
        "no_implicit_latest_glob_selection_v1": True,
        "previous_artifacts_mutated_v1": False,
        "research_only_contract_v1": True,
        "iql_training_run_v1": True,
        "iql_production_allowed_v1": False,
        "adapter_built_v1": False,
        "r6_run_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
        "deprecated_quarantine_revival_v1": False,
        "exit_manager_modified_v1": False,
        "live_features_modified_v1": False,
        "python_manifest_v1": _python_manifest(),
    }


# ---------------------------------------------------------------------------
# Materialize
# ---------------------------------------------------------------------------


def write_artifacts(
    out_root: Path | None = None,
    *,
    built_at_utc: str | None = None,
) -> dict[str, Any]:
    inputs = _load_inputs()
    timestamp = built_at_utc or _stamp()
    artifact_root = out_root or (
        DEFAULT_REPORTS_ROOT / f"{ACTION}_{timestamp}_LOCK"
    )
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
    _write_json(artifact_root / "input_manifest_v1.json", _build_input_manifest(inputs, artifact_root))

    # Load split-locked dataset
    df = pd.read_parquet(inputs["required_paths"]["split_locked_dataset"])
    df["candidate_uid_v1"] = df["candidate_uid_v1"].astype(str)
    df["ts_v1"] = pd.to_datetime(df["ts_v1"], utc=True)
    per_bar_full = _per_bar_view(df)

    # Pre-training audits
    split_isolation = audit_split_isolation(per_bar_full)
    no_shortcut = audit_no_shortcut_at_training_time(STATE_FEATURE_NAMES_V1)

    # Train-only normalization
    per_bar_train = per_bar_full[per_bar_full["primary_split_v1"] == "train"]
    norm = _fit_train_normalization(per_bar_train)
    train_only_audit = audit_train_only_normalization(per_bar_full, norm)

    # Build state matrix and targets for train rows
    per_bar_train_with_targets = _compute_targets(per_bar_train, df)
    X_train = _build_state_matrix(per_bar_train, norm)
    target_hold_train = per_bar_train_with_targets["target_hold_v1"].astype(float).to_numpy()
    target_exit_now_train = per_bar_train_with_targets["target_exit_now_v1"].astype(float).to_numpy()

    # Train two Q heads
    model = _train_q_heads(X_train, target_hold_train, target_exit_now_train)
    coef_hold = np.array(model["coef_hold_v1"], dtype=float)
    coef_exit_now = np.array(model["coef_exit_now_v1"], dtype=float)
    _write_json(artifact_root / "trained_model_v1.json", model)
    _write_json(artifact_root / "training_normalization_v1.json", norm)

    # Inference per split
    iql_results = []
    for split in ["train", "val", "test"]:
        per_bar_split = per_bar_full[per_bar_full["primary_split_v1"] == split].reset_index(drop=True)
        if per_bar_split.empty:
            continue
        X_split = _build_state_matrix(per_bar_split, norm)
        exit_indices = _exit_index_from_iql_policy(
            per_bar_split, X_split, coef_hold, coef_exit_now
        )
        # Safety audit per split
        audit_policy_safety_at_inference(per_bar_split, exit_indices)
        # Evaluate via gate-5 harness
        metrics = eval_gate.evaluate_policy(
            per_bar_split,
            exit_indices,
            policy_id="IQL_CLOSED_FORM_RIDGE_REALIZED_PNL",
            split=split,
        )
        metrics["model_id_v1"] = "EXIT_IQL_RIDGE_2HEAD_V1"
        metrics["reward_variant_v1"] = REWARD_VARIANT_LOCKED_FOR_PRIMARY
        iql_results.append(metrics)

    # Build comparator: pull baseline numbers from the eval-harness LOCK
    baseline_metrics_flat = inputs["eval_harness_baseline_metrics"]["rows_v1"]
    baseline_metrics_per_split: dict[str, list[dict[str, Any]]] = {}
    for row in baseline_metrics_flat:
        baseline_metrics_per_split.setdefault(row["split_v1"], []).append(row)

    comparator_rows = []
    for split in ["train", "val", "test"]:
        for r in baseline_metrics_per_split.get(split, []):
            comparator_rows.append(r)
        for r in iql_results:
            if r["split_v1"] == split:
                comparator_rows.append({**r, "implementable_v1": True, "uses_oracle_v1": False})
    _write_rows(
        artifact_root / "iql_vs_baseline_comparator_v1.csv", comparator_rows
    )
    _write_json(
        artifact_root / "iql_vs_baseline_comparator_v1.json",
        {"row_count_v1": len(comparator_rows), "rows_v1": comparator_rows},
    )

    # Audits
    audits = [split_isolation, no_shortcut, train_only_audit]
    _write_json(
        artifact_root / "training_audits_v1.json",
        {"audit_count_v1": len(audits), "audits_v1": audits},
    )

    repro = _reproducibility_audit(model, iql_results)
    _write_json(artifact_root / "reproducibility_audit_v1.json", repro)

    status, next_action, recommendation = _go_no_go(
        iql_results, baseline_metrics_per_split
    )
    validate_final_status(status, next_action)

    iql_test = next((r for r in iql_results if r["split_v1"] == "test"), None)
    iql_val = next((r for r in iql_results if r["split_v1"] == "val"), None)

    summary = {
        "layer_name": "EXIT_PER_BAR_SANITY_TRAINING_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "model_id_v1": "EXIT_IQL_RIDGE_2HEAD_V1",
        "reward_variant_v1": REWARD_VARIANT_LOCKED_FOR_PRIMARY,
        "ridge_lambda_v1": RIDGE_LAMBDA,
        "feature_count_v1": len(STATE_FEATURE_NAMES_V1),
        "feature_names_v1": list(STATE_FEATURE_NAMES_V1),
        "iql_train_v1": next((r for r in iql_results if r["split_v1"] == "train"), None),
        "iql_val_v1": iql_val,
        "iql_test_v1": iql_test,
        "audits_v1": {a["audit_id_v1"]: a["status_v1"] for a in audits},
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
        "forbidden_actions_audit_v1": forbidden_audit,
    }
    _write_json(artifact_root / "summary_v1.json", summary)

    status_payload = {
        "layer_name": "EXIT_PER_BAR_SANITY_TRAINING_STATUS_V1",
        "status_v1": "MATERIALIZED_RESEARCH_ONLY_TRAINING_GATE",
        "final_status_v1": status,
        "next_action_v1": next_action,
        "training_executed_v1": True,
    }
    _write_json(artifact_root / "status_v1.json", status_payload)

    go_no_go = {
        "layer_name": "EXIT_PER_BAR_SANITY_TRAINING_GO_NO_GO_V1",
        "status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "research_only_v1": True,
        "iql_production_allowed_v1": False,
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "package_freeze_promo_live_allowed_v1": False,
        "policy_promotion_allowed_v1": False,
        "training_allowed_v1": True,
        "downstream_block_v1": (
            "Research-only sanity training. The trained policy is NOT promoted "
            "to production runtime. Adapter/R6/IQL production/live, freeze/"
            "promo/live, exit_manager modification all forbidden."
        ),
    }
    _write_json(
        artifact_root / "exit_per_bar_sanity_training_go_no_go_v1.json", go_no_go
    )

    report_lines = [
        "# Exit Per-Bar Sanity Training V1",
        "",
        f"- Status: `{status}`",
        f"- Next action: `{next_action}`",
        "- Training: research-only; policy NOT promoted to runtime.",
        "",
        "## Model",
        f"- Algorithm: closed-form ridge regression, two Q-heads (HOLD, EXIT_NOW)",
        f"- Features: {len(STATE_FEATURE_NAMES_V1)}",
        f"- Ridge lambda: {RIDGE_LAMBDA}",
        f"- Reward variant: `{REWARD_VARIANT_LOCKED_FOR_PRIMARY}`",
        f"- Train rows: {model['train_row_count_v1']}",
        "",
        "## IQL policy results vs baselines (test split)",
        "",
        "| Policy | Trades | Sum PNL | Mean PNL | MFE-cap | MAE-burden | Giveback | CATA% | Bars |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    test_policies = [r for r in baseline_metrics_per_split.get("test", [])]
    if iql_test is not None:
        test_policies.append({**iql_test, "implementable_v1": True, "uses_oracle_v1": False})
    for r in test_policies:
        report_lines.append(
            f"| `{r['policy_id_v1']}` | {r['trade_count_v1']} | "
            f"{r['total_realized_pnl_bps_v1']:.0f} | "
            f"{r['mean_realized_pnl_bps_v1']:.2f} | "
            f"{r['mean_mfe_capture_ratio_v1']:.3f} | "
            f"{r['mean_mae_burden_bps_v1']:.1f} | "
            f"{r['mean_giveback_bps_v1']:.1f} | "
            f"{r['cata_proxy_rate_v1']*100:.1f}% | "
            f"{r['mean_bars_to_exit_v1']:.1f} |"
        )
    report_lines.extend([
        "",
        "## Audits",
    ])
    for a in audits:
        report_lines.append(f"- `{a['audit_id_v1']}`: {a['status_v1']}")
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
                artifact_root / "exit_per_bar_sanity_training_go_no_go_v1.json"
            ),
            "input_manifest": str(artifact_root / "input_manifest_v1.json"),
            "trained_model": str(artifact_root / "trained_model_v1.json"),
            "training_normalization": str(
                artifact_root / "training_normalization_v1.json"
            ),
            "iql_vs_baseline_comparator_json": str(
                artifact_root / "iql_vs_baseline_comparator_v1.json"
            ),
            "iql_vs_baseline_comparator_csv": str(
                artifact_root / "iql_vs_baseline_comparator_v1.csv"
            ),
            "training_audits": str(artifact_root / "training_audits_v1.json"),
            "reproducibility_audit": str(
                artifact_root / "reproducibility_audit_v1.json"
            ),
        },
        "read_only_references_v1": True,
        "trained_model_v1": True,
        "not_controller_v1": True,
        "not_live_gate_v1": True,
    }
    _write_json(artifact_root / "manifest_v1.json", artifact_manifest)

    return {
        "artifact_root": str(artifact_root),
        "summary": summary,
        "status": status_payload,
        "go_no_go": go_no_go,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize EXIT_PER_BAR_SANITY_TRAINING_V1 gate."
    )
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--built-at-utc", type=str, default=None)
    args = parser.parse_args()
    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else None
    result = write_artifacts(out_root=out_root, built_at_utc=args.built_at_utc)
    print(json.dumps(_jsonable(result), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
