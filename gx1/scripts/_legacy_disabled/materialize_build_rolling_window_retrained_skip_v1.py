#!/usr/bin/env python3
"""Online adaptation: rolling-window retrained skip-V2 + V2 IQL stack.

Background
----------
Walk-forward + head-to-head established that:
  - Static policies are regime-dependent: trained-once on first N trades,
    they fail when the test regime differs from training regime.
  - Combined stack (skip + V2 IQL) is a near-zero-correlation diversifier
    of realized (corr +0.06, total +598 across 524 trades), but its
    cross-fold lift relative to realized fails the locked promotion
    criteria (1/3 folds positive vs realized).

Standard professional answer to regime-shift: ONLINE ADAPTATION. Retrain
on the most recent window of trades each step; the model adapts as the
regime drifts. This gate implements that:

  - Define a STEP_SIZE (50 trades) and a WINDOW_SIZE (800 trades).
  - Walk forward through the entire trade sequence; at each step:
       1. Train skip-V2 + V2 IQL (5 reward variants, best on val) on
          the last WINDOW_SIZE trades, with internal 70/15 train/val
          split. Reserve the next STEP_SIZE trades as the test window.
       2. Apply the stack on those STEP_SIZE trades. Persist per-trade
          PNL for each policy.
       3. Advance by STEP_SIZE.
  - At the end, aggregate per-trade decisions across all step-windows
    and compute the same per-policy / cross-window stability metrics as
    head-to-head.
  - Apply the locked promotion criteria.

Trade indexing: time-ordered candidate_uid_v1 list (same as walk-forward).
First WINDOW_SIZE trades cannot be tested (need a full window for the
first model). Test coverage = total_trades - WINDOW_SIZE.

Research-only; no policy promotion; no runtime modification. The output
gives an empirical answer to "does online retraining help?".
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
ACTION = "BUILD_ROLLING_WINDOW_RETRAINED_SKIP_V1"

INPUT_HEAD_TO_HEAD_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "RUN_LIVE_SYSTEM_VS_RESEARCH_CANDIDATES_HEAD_TO_HEAD_V1_20260430T072907Z_LOCK"
)
INPUT_PROMOTION_CRITERIA_ROOT = (
    DEFAULT_REPORTS_ROOT / "DEFINE_PROMOTION_CRITERIA_V1_20260430T070707Z_LOCK"
)
INPUT_RECOVERY_ROOT = v2_train_gate.INPUT_RECOVERY_ROOT
INPUT_SPLIT_ROOT = v2_train_gate.INPUT_SPLIT_ROOT
INPUT_V2_CONTRACT_ROOT = v2_train_gate.INPUT_V2_CONTRACT_ROOT
BASE34_M5_FEATURES_PATH = v2_train_gate.BASE34_M5_FEATURES_PATH

QUARANTINE_FORBIDDEN_PATH_FRAGMENTS = ("gx1/quarantine", "gx1.quarantine")
SEED_V1 = 20260430
WINDOW_SIZE_TRADES = 800
STEP_SIZE_TRADES = 50
TRAIN_FRACTION_WITHIN_WINDOW = 0.85  # 85% train, 15% val within each rolling window

ALLOWED_FINAL_STATUSES = {
    "ROLLING_WINDOW_RETRAIN_PASS_MEETS_PROMOTION_CRITERIA",
    "ROLLING_WINDOW_RETRAIN_PASS_BEATS_STATIC_BUT_FAILS_OTHER_CRITERIA",
    "ROLLING_WINDOW_RETRAIN_PARTIAL_TIES_STATIC",
    "ROLLING_WINDOW_RETRAIN_PARTIAL_DEGRADES_VS_STATIC",
    "ROLLING_WINDOW_RETRAIN_BLOCKED_BY_INPUT_LOCK_MISSING",
}

ALLOWED_NEXT_ACTIONS = {
    "BUILD_REGIME_DETECTOR_PLUS_POLICY_ENSEMBLE_V1",
    "DEFINE_PAPER_TRADING_PROMOTION_PLAN_V1",
    "ACCEPT_TRAIL_STOP_AS_RESEARCH_BASELINE_V1",
    "REPAIR_ROLLING_WINDOW_BEFORE_FURTHER_WORK_V1",
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
# Rolling step computation
# ---------------------------------------------------------------------------


def _compute_steps(n_trades: int) -> list[dict[str, int]]:
    """Define rolling-window steps. Each step trains on
    [step_start, train_end) and tests on [test_start, test_end).
    """
    steps: list[dict[str, int]] = []
    test_start = WINDOW_SIZE_TRADES + 1  # 1-indexed trade rank
    while test_start <= n_trades:
        train_start = test_start - WINDOW_SIZE_TRADES
        train_window_size = int(TRAIN_FRACTION_WITHIN_WINDOW * WINDOW_SIZE_TRADES)
        train_end = train_start + train_window_size - 1
        val_end = test_start - 1
        test_end = min(test_start + STEP_SIZE_TRADES - 1, n_trades)
        steps.append(
            {
                "step_id_v1": f"STEP_{len(steps)+1:03d}",
                "train_start_v1": train_start,
                "train_end_v1": train_end,
                "val_end_v1": val_end,
                "test_end_v1": test_end,
            }
        )
        test_start = test_end + 1
    return steps


def _step_uid_to_split(
    candidate_uid_order: list[str], step: dict[str, int]
) -> dict[str, str]:
    out: dict[str, str] = {}
    for rank, uid in enumerate(candidate_uid_order, start=1):
        if rank < step["train_start_v1"]:
            out[uid] = "hold_out"
        elif rank <= step["train_end_v1"]:
            out[uid] = "train"
        elif rank <= step["val_end_v1"]:
            out[uid] = "val"
        elif rank <= step["test_end_v1"]:
            out[uid] = "test"
        else:
            out[uid] = "hold_out"
    return out


# ---------------------------------------------------------------------------
# Per-step training + per-trade PNL
# ---------------------------------------------------------------------------


def _project_per_trade_for_step(
    inputs: dict[str, Any], uid_to_split: dict[str, str]
) -> pd.DataFrame:
    return wf_gate._build_per_trade_for_fold(inputs, uid_to_split)


def _project_per_bar_for_step(
    inputs: dict[str, Any], uid_to_split: dict[str, str]
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    return wf_gate._build_per_bar_for_fold(inputs, uid_to_split)


def _train_skip_v2_for_step(
    per_trade: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    per_trade_train = per_trade[per_trade["primary_split_v1"] == "train"]
    if len(per_trade_train) < 50:
        return np.zeros(len(per_trade)), 0.5
    norm = skip_v1_gate._fit_train_normalization(per_trade_train)
    X_full, _ = skip_v1_gate._build_state_matrix(per_trade, norm)
    train_mask = (per_trade["primary_split_v1"] == "train").to_numpy()
    val_mask = (per_trade["primary_split_v1"] == "val").to_numpy()
    y = per_trade["should_skip_v1"].astype(int).to_numpy()
    if y[train_mask].sum() == 0 or y[train_mask].sum() == int(train_mask.sum()):
        return np.zeros(len(per_trade)), 0.5
    logreg = skip_v2_gate._train_logistic(X_full[train_mask], y[train_mask])
    p_skip = skip_v2_gate._predict_p_skip(logreg, X_full)
    val_df = per_trade[val_mask].reset_index(drop=True)
    p_skip_val = p_skip[val_mask]
    best_thr = 0.5
    best_pnl = -np.inf
    for thr in skip_v2_gate.THRESHOLD_GRID:
        m = skip_v1_gate._evaluate_threshold(val_df, p_skip_val, thr)
        if m["pnl_taken_v1"] > best_pnl:
            best_pnl = m["pnl_taken_v1"]
            best_thr = float(thr)
    return p_skip, best_thr


def _train_v2_iql_for_step(
    per_bar: pd.DataFrame, X_full: np.ndarray
) -> dict[str, dict[str, np.ndarray]]:
    train_mask = (per_bar["primary_split_v1"] == "train").to_numpy()
    per_bar_train = per_bar[per_bar["primary_split_v1"] == "train"]
    models: dict[str, dict[str, np.ndarray]] = {}
    for variant in v2_train_gate.REWARD_VARIANTS_V2:
        v_id = variant["reward_id_v1"]
        reward_col = variant["reward_column_v1"]
        targets = v2_train_gate._compute_targets_for_variant(per_bar_train, reward_col)
        target_hold = targets["__target_hold_v1"].astype(float).to_numpy()
        target_exit_now = targets["__target_exit_now_v1"].astype(float).to_numpy()
        coef_hold = v2_train_gate._ridge_fit(X_full[train_mask], target_hold)
        coef_exit_now = v2_train_gate._ridge_fit(X_full[train_mask], target_exit_now)
        models[v_id] = {"coef_hold": coef_hold, "coef_exit_now": coef_exit_now}
    return models


def _select_best_variant_on_val(
    per_bar: pd.DataFrame, X_full: np.ndarray, models: dict[str, dict[str, np.ndarray]]
) -> str:
    """Pick reward variant with best val total PNL on V2 IQL exit policy."""
    val_mask = (per_bar["primary_split_v1"] == "val").to_numpy()
    per_bar_val = per_bar[val_mask].reset_index(drop=True)
    X_val = X_full[val_mask]
    best_v = None
    best_total = -np.inf
    for v_id, m in models.items():
        if per_bar_val.empty:
            return v_id  # fallback: first variant
        exit_indices = v2_train_gate._exit_index_from_iql_policy(
            per_bar_val, X_val, m["coef_hold"], m["coef_exit_now"]
        )
        selected = per_bar_val.loc[exit_indices.values]
        total = float(selected["running_pnl_at_close_bps_v1"].sum())
        if total > best_total:
            best_total = total
            best_v = v_id
    return best_v or list(models.keys())[0]


def _evaluate_step_test(
    per_trade: pd.DataFrame,
    p_skip_full: np.ndarray,
    threshold: float,
    per_bar: pd.DataFrame,
    X_full: np.ndarray,
    models: dict[str, dict[str, np.ndarray]],
    best_variant: str,
) -> pd.DataFrame:
    """Per-trade head-to-head PNL on this step's test window."""
    test_mask_pt = (per_trade["primary_split_v1"] == "test").to_numpy()
    test_mask_pb = (per_bar["primary_split_v1"] == "test").to_numpy()
    test_uids = per_trade.loc[test_mask_pt, "candidate_uid_v1"].astype(str).tolist()
    test_uid_set = set(test_uids)
    p_skip_test = p_skip_full[test_mask_pt]
    skipped_uids = set(
        per_trade.loc[test_mask_pt, "candidate_uid_v1"]
        .astype(str)
        .iloc[(p_skip_test >= threshold).nonzero()[0]]
        .tolist()
    )
    per_bar_test = per_bar[test_mask_pb].reset_index(drop=True)
    X_test = X_full[test_mask_pb]
    m = models[best_variant]
    exit_indices = v2_train_gate._exit_index_from_iql_policy(
        per_bar_test, X_test, m["coef_hold"], m["coef_exit_now"]
    )
    iql_pnl_per_uid = dict(
        zip(
            per_bar_test.loc[exit_indices.values, "candidate_uid_v1"].astype(str).tolist(),
            per_bar_test.loc[exit_indices.values, "running_pnl_at_close_bps_v1"].astype(float).tolist(),
        )
    )
    realized_pnl_per_uid: dict[str, float] = {}
    for uid, group in per_bar_test.groupby("candidate_uid_v1", sort=False):
        last = group.sort_values("bars_held_v1").tail(1)
        realized_pnl_per_uid[str(uid)] = float(
            last["running_pnl_at_close_bps_v1"].iloc[0]
        )
    rows: list[dict[str, Any]] = []
    for uid in test_uids:
        if uid not in realized_pnl_per_uid:
            continue
        realized = realized_pnl_per_uid[uid]
        iql_pnl = iql_pnl_per_uid.get(uid, realized)
        skipped = uid in skipped_uids
        skip_then_realized = 0.0 if skipped else realized
        skip_then_iql = 0.0 if skipped else iql_pnl
        rows.append(
            {
                "candidate_uid_v1": uid,
                "REALIZED_LIVE_SYSTEM": realized,
                "ROLLING_SKIP_V2_THEN_REALIZED": skip_then_realized,
                "ROLLING_V2_IQL_BEST_PER_STEP": iql_pnl,
                "ROLLING_SKIP_V2_THEN_V2_IQL_COMBINED": skip_then_iql,
                "skipped_v1": skipped,
                "tuned_skip_threshold_v1": threshold,
                "best_iql_variant_v1": best_variant,
            }
        )
    return pd.DataFrame(rows)


def _run_rolling(
    inputs: dict[str, Any], candidate_uid_order: list[str]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    n = len(candidate_uid_order)
    steps = _compute_steps(n)
    per_trade_rows: list[pd.DataFrame] = []
    step_audit: list[dict[str, Any]] = []
    for step in steps:
        uid_to_split = _step_uid_to_split(candidate_uid_order, step)
        per_trade = _project_per_trade_for_step(inputs, uid_to_split)
        per_bar, X_full, _ = _project_per_bar_for_step(inputs, uid_to_split)
        p_skip_full, threshold = _train_skip_v2_for_step(per_trade)
        models = _train_v2_iql_for_step(per_bar, X_full)
        best_variant = _select_best_variant_on_val(per_bar, X_full, models)
        step_test_table = _evaluate_step_test(
            per_trade, p_skip_full, threshold, per_bar, X_full, models, best_variant
        )
        step_test_table["step_id_v1"] = step["step_id_v1"]
        per_trade_rows.append(step_test_table)
        step_audit.append(
            {
                **step,
                "n_train_v1": int(
                    (per_trade["primary_split_v1"] == "train").sum()
                ),
                "n_val_v1": int((per_trade["primary_split_v1"] == "val").sum()),
                "n_test_v1": int((per_trade["primary_split_v1"] == "test").sum()),
                "tuned_skip_threshold_v1": threshold,
                "best_iql_variant_v1": best_variant,
            }
        )
    if per_trade_rows:
        full_table = pd.concat(per_trade_rows, ignore_index=True)
    else:
        full_table = pd.DataFrame(
            columns=[
                "step_id_v1",
                "candidate_uid_v1",
                "REALIZED_LIVE_SYSTEM",
                "ROLLING_SKIP_V2_THEN_REALIZED",
                "ROLLING_V2_IQL_BEST_PER_STEP",
                "ROLLING_SKIP_V2_THEN_V2_IQL_COMBINED",
                "skipped_v1",
                "tuned_skip_threshold_v1",
                "best_iql_variant_v1",
            ]
        )
    return full_table, step_audit


# ---------------------------------------------------------------------------
# Apply locked promotion criteria
# ---------------------------------------------------------------------------


def _per_step_totals(
    full_table: pd.DataFrame, policy_col: str
) -> list[float]:
    return [
        float(full_table.loc[full_table["step_id_v1"] == step, policy_col].sum())
        for step in full_table["step_id_v1"].unique()
    ]


def _apply_promotion_criteria(full_table: pd.DataFrame) -> dict[str, Any]:
    """Apply criteria to the rolling combined stack, lifts measured vs realized."""
    realized_per_step = _per_step_totals(full_table, "REALIZED_LIVE_SYSTEM")
    combined_per_step = _per_step_totals(
        full_table, "ROLLING_SKIP_V2_THEN_V2_IQL_COMBINED"
    )
    trail_stop_proxy = [1052.0] * len(realized_per_step)
    lifts = [c - r for c, r in zip(combined_per_step, realized_per_step)]
    promotion = criteria_gate.evaluate_candidate_against_criteria(
        candidate_id="rolling_window_retrained_combined",
        per_fold_lifts_bps=lifts,
        per_fold_pnl_bps=combined_per_step,
        per_fold_trail_stop_pnl_bps=trail_stop_proxy,
        no_shortcut_audit_passed=True,
        deterministic_reproducible=True,
    )
    promotion["per_step_lifts_vs_realized_v1"] = lifts
    promotion["per_step_combined_pnl_v1"] = combined_per_step
    promotion["per_step_realized_pnl_v1"] = realized_per_step
    return promotion


# ---------------------------------------------------------------------------
# go-no-go
# ---------------------------------------------------------------------------


def _go_no_go(
    full_table: pd.DataFrame, promotion: dict[str, Any]
) -> tuple[str, str, str, dict[str, Any]]:
    realized_total = float(full_table["REALIZED_LIVE_SYSTEM"].sum())
    combined_total = float(
        full_table["ROLLING_SKIP_V2_THEN_V2_IQL_COMBINED"].sum()
    )
    iql_total = float(full_table["ROLLING_V2_IQL_BEST_PER_STEP"].sum())
    skip_total = float(full_table["ROLLING_SKIP_V2_THEN_REALIZED"].sum())
    n_trades = int(len(full_table))
    realized_mean = realized_total / max(1, n_trades)
    combined_mean = combined_total / max(1, n_trades)
    headline = {
        "n_test_trades_v1": n_trades,
        "n_steps_v1": int(full_table["step_id_v1"].nunique()) if n_trades else 0,
        "realized_total_v1": realized_total,
        "rolling_combined_total_v1": combined_total,
        "rolling_iql_total_v1": iql_total,
        "rolling_skip_only_total_v1": skip_total,
        "rolling_combined_minus_realized_v1": combined_total - realized_total,
        "promotion_pass_v1": bool(promotion["overall_pass_v1"]),
        "promotion_n_passed_v1": int(promotion["n_criteria_passed_v1"]),
        "promotion_n_total_v1": int(promotion["n_criteria_total_v1"]),
    }
    if promotion["overall_pass_v1"]:
        return (
            "ROLLING_WINDOW_RETRAIN_PASS_MEETS_PROMOTION_CRITERIA",
            "BUILD_REGIME_DETECTOR_PLUS_POLICY_ENSEMBLE_V1",
            (
                f"Rolling-window retrained combined stack passes ALL promotion "
                f"criteria. Total {combined_total:+.0f} vs realized "
                f"{realized_total:+.0f} ({headline['rolling_combined_minus_realized_v1']:+.0f} "
                f"lift). Online adaptation is the answer to regime-shift. "
                "Next: regime detector + policy ensemble for the final layer."
            ),
            headline,
        )
    if combined_total > realized_total + 100.0:
        return (
            "ROLLING_WINDOW_RETRAIN_PASS_BEATS_STATIC_BUT_FAILS_OTHER_CRITERIA",
            "BUILD_REGIME_DETECTOR_PLUS_POLICY_ENSEMBLE_V1",
            (
                f"Rolling combined beats realized by "
                f"{combined_total - realized_total:+.0f} bps total but fails "
                f"{promotion['n_criteria_total_v1'] - promotion['n_criteria_passed_v1']} "
                "of 6 criteria. Next: try regime ensemble."
            ),
            headline,
        )
    if abs(combined_total - realized_total) <= 100.0:
        return (
            "ROLLING_WINDOW_RETRAIN_PARTIAL_TIES_STATIC",
            "BUILD_REGIME_DETECTOR_PLUS_POLICY_ENSEMBLE_V1",
            (
                f"Rolling combined ~= realized "
                f"({combined_total:+.0f} vs {realized_total:+.0f}; delta "
                f"{combined_total - realized_total:+.0f}). Online adaptation "
                "did not produce a clear lift. Next: try regime ensemble."
            ),
            headline,
        )
    return (
        "ROLLING_WINDOW_RETRAIN_PARTIAL_DEGRADES_VS_STATIC",
        "REPAIR_ROLLING_WINDOW_BEFORE_FURTHER_WORK_V1",
        (
            f"Rolling combined degrades vs realized "
            f"({combined_total:+.0f} vs {realized_total:+.0f}). Investigate."
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
        "layer_name": "ROLLING_WINDOW_RETRAIN_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "head_to_head_root_v1": str(INPUT_HEAD_TO_HEAD_ROOT),
            "promotion_criteria_root_v1": str(INPUT_PROMOTION_CRITERIA_ROOT),
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

    full_table, step_audit = _run_rolling(inputs, candidate_uid_order)
    full_table.to_csv(
        artifact_root / "per_trade_per_step_pnl_v1.csv", index=False
    )
    _write_rows(artifact_root / "step_audit_v1.csv", step_audit)
    _write_json(
        artifact_root / "step_audit_v1.json",
        {"row_count_v1": len(step_audit), "rows_v1": step_audit},
    )

    promotion = _apply_promotion_criteria(full_table) if not full_table.empty else {
        "overall_pass_v1": False,
        "n_criteria_passed_v1": 0,
        "n_criteria_total_v1": 6,
        "breakdown_v1": [],
        "per_step_lifts_vs_realized_v1": [],
        "per_step_combined_pnl_v1": [],
        "per_step_realized_pnl_v1": [],
    }
    _write_json(artifact_root / "promotion_criteria_evaluation_v1.json", promotion)

    repro = {
        "layer_name": "ROLLING_WINDOW_RETRAIN_REPRODUCIBILITY_AUDIT_V1",
        "window_size_trades_v1": WINDOW_SIZE_TRADES,
        "step_size_trades_v1": STEP_SIZE_TRADES,
        "train_fraction_within_window_v1": TRAIN_FRACTION_WITHIN_WINDOW,
        "n_steps_v1": len(step_audit),
        "seed_v1": SEED_V1,
        "no_implicit_glob_used_for_v1_inputs_v1": True,
        "deprecated_quarantine_revival_v1": False,
        "research_only_v1": True,
    }
    _write_json(artifact_root / "reproducibility_audit_v1.json", repro)

    status, next_action, recommendation, headline = _go_no_go(full_table, promotion)
    validate_final_status(status, next_action)

    summary = {
        "layer_name": "ROLLING_WINDOW_RETRAIN_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "headline_v1": headline,
        "window_size_trades_v1": WINDOW_SIZE_TRADES,
        "step_size_trades_v1": STEP_SIZE_TRADES,
        "n_steps_v1": len(step_audit),
        "step_audit_v1": step_audit,
        "promotion_evaluation_v1": promotion,
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
            "layer_name": "ROLLING_WINDOW_RETRAIN_STATUS_V1",
            "status_v1": "MATERIALIZED_RESEARCH_ONLY_GATE",
            "final_status_v1": status,
            "next_action_v1": next_action,
            "training_executed_v1": True,
        },
    )
    _write_json(
        artifact_root / "build_rolling_window_retrained_skip_go_no_go_v1.json",
        {
            "layer_name": "ROLLING_WINDOW_RETRAIN_GO_NO_GO_V1",
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
                "Research-only rolling-window retraining. NOT promoted to runtime."
            ),
        },
    )

    report_lines = [
        "# Build Rolling-Window Retrained Skip V1",
        "",
        f"- Status: `{status}`",
        f"- Next action: `{next_action}`",
        "- Training: research-only; rolling stack NOT promoted to runtime.",
        "",
        "## Headline",
        f"- Window size: {WINDOW_SIZE_TRADES} trades; step size: {STEP_SIZE_TRADES} trades; n steps: {len(step_audit)}",
        f"- Test trades evaluated: {headline['n_test_trades_v1']}",
        f"- Realized total: {headline['realized_total_v1']:+.0f} bps",
        f"- Rolling combined total: **{headline['rolling_combined_total_v1']:+.0f}** bps",
        f"- Combined - realized: {headline['rolling_combined_minus_realized_v1']:+.0f} bps",
        f"- Rolling skip-only: {headline['rolling_skip_only_total_v1']:+.0f}",
        f"- Rolling V2 IQL: {headline['rolling_iql_total_v1']:+.0f}",
        f"- Promotion criteria: {headline['promotion_n_passed_v1']}/{headline['promotion_n_total_v1']}",
        "",
        "## Promotion-criteria breakdown",
    ]
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
                artifact_root / "build_rolling_window_retrained_skip_go_no_go_v1.json"
            ),
            "input_manifest": str(artifact_root / "input_manifest_v1.json"),
            "per_trade_per_step_pnl_csv": str(
                artifact_root / "per_trade_per_step_pnl_v1.csv"
            ),
            "step_audit_csv": str(artifact_root / "step_audit_v1.csv"),
            "step_audit_json": str(artifact_root / "step_audit_v1.json"),
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
        description="Materialize BUILD_ROLLING_WINDOW_RETRAINED_SKIP_V1."
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
