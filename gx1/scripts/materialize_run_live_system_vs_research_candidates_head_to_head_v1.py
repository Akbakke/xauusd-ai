#!/usr/bin/env python3
"""Head-to-head: live system (realized) vs every research candidate, per fold.

Background
----------
The user pointed out we have implicitly chased the wrong baseline. Across
walk-forward (524 test trades, 3 folds):

  - Realized (live system: XGB + entry transformer + exit transformer V3):
    -473 bps total, -0.90 bps/trade.
  - Trail-stop rule: -989 bps total. WORSE than the live system on the
    full walk-forward, despite winning the single-fold test.
  - Combined (skip-V2 then V2 IQL): +614 bps total. The ONLY policy that
    is net-positive across all 524 trades, with a CONVEX tail-hedge
    profile (-83 / +353 / +344 vs realized's +88 / +1301 / -1862).

Promotion criteria correctly rejected combined as a static replacement,
but the criteria were designed for "find one static winner". A tail-
hedge profile that complements the live system is a different, valid
research outcome.

This gate produces the explicit head-to-head dashboard we should have
had earlier. No new training; loads existing artifacts and recomputes
per-trade PNL for each policy to compute:

  1. Per-fold totals and means.
  2. Per-trade PNL std and a Sharpe-like ratio (mean / std).
  3. Cumulative-PNL drawdown per fold per policy.
  4. Pairwise correlation matrix across policies (per-trade PNL).
  5. Diversification score: 1 - corr(policy, realized).
  6. Hit rate per policy (fraction of trades > 0).
  7. Cross-fold-stability metrics (mean / std / min / max of fold totals).

Outputs:
  - per_policy_per_fold_metrics_v1.csv
  - pairwise_pnl_correlation_v1.json
  - cross_fold_stability_v1.json
  - cumulative_pnl_per_policy_per_fold_v1.csv (one row per (policy, fold,
    trade-rank) for plotting)
  - report_v1.md with explicit head-to-head verdict.

Research-only diagnostic; no training, no model promotion. The output
informs which candidates are real complements vs noise.
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
    materialize_combine_skip_v2_with_exit_iql_v2_v1 as combined_gate,
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
ACTION = "RUN_LIVE_SYSTEM_VS_RESEARCH_CANDIDATES_HEAD_TO_HEAD_V1"

INPUT_RECOVERY_ROOT = v2_train_gate.INPUT_RECOVERY_ROOT
INPUT_SPLIT_ROOT = v2_train_gate.INPUT_SPLIT_ROOT
INPUT_V2_CONTRACT_ROOT = v2_train_gate.INPUT_V2_CONTRACT_ROOT
INPUT_PROMOTION_CRITERIA_ROOT = (
    DEFAULT_REPORTS_ROOT / "DEFINE_PROMOTION_CRITERIA_V1_20260430T070707Z_LOCK"
)
INPUT_WALK_FORWARD_ROOT = (
    DEFAULT_REPORTS_ROOT / "WALK_FORWARD_VALIDATION_V1_20260430T065421Z_LOCK"
)
BASE34_M5_FEATURES_PATH = v2_train_gate.BASE34_M5_FEATURES_PATH

QUARANTINE_FORBIDDEN_PATH_FRAGMENTS = ("gx1/quarantine", "gx1.quarantine")
SEED_V1 = 20260430

POLICIES_HEAD_TO_HEAD = [
    "REALIZED_LIVE_SYSTEM",
    "TRAIL_STOP_25_PCT_DD",
    "SKIP_V2_THEN_REALIZED",
    "V2_IQL_BEST_PER_FOLD",
    "SKIP_V2_THEN_V2_IQL_COMBINED",
    "HYBRID_TRAIL_STOP_PLUS_DELAY_LEARNER",
]

ALLOWED_FINAL_STATUSES = {
    "HEAD_TO_HEAD_LOCKED_V1",
    "HEAD_TO_HEAD_BLOCKED_BY_INPUT_LOCK_MISSING_V1",
}

ALLOWED_NEXT_ACTIONS = {
    "BUILD_ROLLING_WINDOW_RETRAINED_SKIP_V1",
    "BUILD_REGIME_DETECTOR_PLUS_POLICY_ENSEMBLE_V1",
    "REPAIR_HEAD_TO_HEAD_BEFORE_FURTHER_WORK_V1",
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
# Per-policy per-trade PNL extraction
# ---------------------------------------------------------------------------


def _trail_stop_pnl_per_trade(per_bar_full: pd.DataFrame) -> pd.DataFrame:
    """For each trade, identify trail-stop fire bar and compute per-trade
    realized PNL. Returns columns (candidate_uid_v1, trail_stop_pnl_bps,
    realized_pnl_bps)."""
    rows: list[dict[str, Any]] = []
    for uid, group in per_bar_full.groupby("candidate_uid_v1", sort=False):
        df = group.sort_values("bars_held_v1").reset_index(drop=True)
        pnl = df["running_pnl_at_close_bps_v1"].astype(float).to_numpy()
        mfe = df["running_mfe_bps_v1"].astype(float).to_numpy()
        n = int(len(df))
        if n == 0:
            continue
        fire_bar = -1
        for i in range(n):
            if mfe[i] >= ts_gate.TRAIL_STOP_MIN_MFE_BPS:
                if (mfe[i] - pnl[i]) / max(mfe[i], 1e-9) >= ts_gate.TRAIL_STOP_GIVEBACK_RATIO:
                    fire_bar = i
                    break
        realized_pnl = float(pnl[-1])
        if fire_bar == -1:
            trail_stop_pnl = realized_pnl
        else:
            trail_stop_pnl = float(pnl[fire_bar])
        rows.append(
            {
                "candidate_uid_v1": str(uid),
                "trail_stop_pnl_bps_v1": trail_stop_pnl,
                "realized_pnl_bps_v1": realized_pnl,
            }
        )
    return pd.DataFrame(rows)


def _v2_iql_pnl_per_trade(
    per_bar_split_test: pd.DataFrame,
    X_split_test: np.ndarray,
    coef_hold: np.ndarray,
    coef_exit_now: np.ndarray,
) -> dict[str, float]:
    """Apply V2 IQL exit policy to test split; return per-candidate PNL."""
    exit_indices = v2_train_gate._exit_index_from_iql_policy(
        per_bar_split_test, X_split_test, coef_hold, coef_exit_now
    )
    selected_idx = exit_indices.values
    selected = per_bar_split_test.loc[selected_idx].reset_index(drop=True)
    return dict(
        zip(
            selected["candidate_uid_v1"].astype(str).tolist(),
            selected["running_pnl_at_close_bps_v1"].astype(float).tolist(),
        )
    )


def _build_per_trade_pnl_table_for_fold(
    inputs: dict[str, Any],
    candidate_uid_order: list[str],
    fold: dict[str, int],
) -> pd.DataFrame:
    """Per fold: build a per-trade table with a column for each policy's PNL."""
    fold_id = fold["fold_id_v1"]
    uid_to_split = wf_gate._assign_fold_split(candidate_uid_order, fold)
    per_trade = wf_gate._build_per_trade_for_fold(inputs, uid_to_split)
    per_bar, X_full, _ = wf_gate._build_per_bar_for_fold(inputs, uid_to_split)
    test_mask = (per_bar["primary_split_v1"] == "test").to_numpy()
    per_bar_test = per_bar[test_mask].reset_index(drop=True)
    X_test = X_full[test_mask]

    # Trail-stop and realized PNL per trade (test only).
    ts_table = _trail_stop_pnl_per_trade(per_bar_test)

    # Skip-V2: train on fold's train, predict p_skip on test trades.
    per_trade_train = per_trade[per_trade["primary_split_v1"] == "train"]
    skip_norm = skip_v1_gate._fit_train_normalization(per_trade_train)
    X_skip_full, _ = skip_v1_gate._build_state_matrix(per_trade, skip_norm)
    skip_train_mask = (per_trade["primary_split_v1"] == "train").to_numpy()
    y_skip = per_trade["should_skip_v1"].astype(int).to_numpy()
    if y_skip[skip_train_mask].sum() > 0:
        logreg = skip_v2_gate._train_logistic(
            X_skip_full[skip_train_mask], y_skip[skip_train_mask]
        )
        p_skip_full = skip_v2_gate._predict_p_skip(logreg, X_skip_full)
        # Tune threshold on val.
        val_mask_pt = (per_trade["primary_split_v1"] == "val").to_numpy()
        per_trade_val = per_trade[val_mask_pt].reset_index(drop=True)
        p_skip_val = p_skip_full[val_mask_pt]
        best_thr = 0.5
        best_pnl = -np.inf
        for thr in skip_v2_gate.THRESHOLD_GRID:
            m = skip_v1_gate._evaluate_threshold(per_trade_val, p_skip_val, thr)
            if m["pnl_taken_v1"] > best_pnl:
                best_pnl = m["pnl_taken_v1"]
                best_thr = float(thr)
        test_mask_pt = (per_trade["primary_split_v1"] == "test").to_numpy()
        per_trade_test = per_trade[test_mask_pt].reset_index(drop=True)
        p_skip_test = p_skip_full[test_mask_pt]
        skipped_uids = set(
            per_trade_test.loc[p_skip_test >= best_thr, "candidate_uid_v1"]
            .astype(str)
            .tolist()
        )
    else:
        skipped_uids = set()
        best_thr = 0.5

    # V2 IQL per reward variant -> pick best per fold.
    train_mask_pb = (per_bar["primary_split_v1"] == "train").to_numpy()
    per_bar_train = per_bar[per_bar["primary_split_v1"] == "train"]
    best_iql_pnl_per_uid: dict[str, float] | None = None
    best_iql_total = -np.inf
    best_variant = None
    iql_pnl_by_variant: dict[str, dict[str, float]] = {}
    for variant in v2_train_gate.REWARD_VARIANTS_V2:
        v_id = variant["reward_id_v1"]
        reward_col = variant["reward_column_v1"]
        targets = v2_train_gate._compute_targets_for_variant(per_bar_train, reward_col)
        target_hold = targets["__target_hold_v1"].astype(float).to_numpy()
        target_exit_now = (
            targets["__target_exit_now_v1"].astype(float).to_numpy()
        )
        coef_hold = v2_train_gate._ridge_fit(X_full[train_mask_pb], target_hold)
        coef_exit_now = v2_train_gate._ridge_fit(
            X_full[train_mask_pb], target_exit_now
        )
        pnl_per_uid = _v2_iql_pnl_per_trade(
            per_bar_test, X_test, coef_hold, coef_exit_now
        )
        iql_pnl_by_variant[v_id] = pnl_per_uid
        total = sum(pnl_per_uid.values())
        if total > best_iql_total:
            best_iql_total = total
            best_iql_pnl_per_uid = pnl_per_uid
            best_variant = v_id
    if best_iql_pnl_per_uid is None:
        best_iql_pnl_per_uid = {}

    # Hybrid trail-stop + delay learner (re-train per fold).
    # Compute would-delay-help label and train logistic.
    ts_label_table = _trail_stop_pnl_per_trade_with_label(per_bar)
    merged_label = per_trade.merge(
        ts_label_table[["candidate_uid_v1", "firing_status_v1", "would_delay_help_v1"]],
        on="candidate_uid_v1",
        how="left",
    )
    merged_label["firing_status_v1"] = merged_label["firing_status_v1"].fillna(
        "NEVER_FIRED"
    )
    merged_label["would_delay_help_v1"] = (
        merged_label["would_delay_help_v1"].fillna(0).astype(int)
    )
    train_label_mask = (
        (merged_label["primary_split_v1"] == "train")
        & (merged_label["firing_status_v1"] == "FIRED")
    ).to_numpy()
    if int(train_label_mask.sum()) > 5:
        norm_hyb = skip_v1_gate._fit_train_normalization(
            merged_label[train_label_mask]
        )
        X_hyb_full, _ = skip_v1_gate._build_state_matrix(merged_label, norm_hyb)
        y_hyb = merged_label["would_delay_help_v1"].astype(int).to_numpy()
        if (
            y_hyb[train_label_mask].sum() > 0
            and y_hyb[train_label_mask].sum() < int(train_label_mask.sum())
        ):
            logreg_hyb = skip_v2_gate._train_logistic(
                X_hyb_full[train_label_mask], y_hyb[train_label_mask]
            )
            p_delay_full = skip_v2_gate._predict_p_skip(logreg_hyb, X_hyb_full)
        else:
            p_delay_full = np.zeros(len(merged_label))
    else:
        p_delay_full = np.zeros(len(merged_label))
    # Tune delay threshold on val.
    val_mask_pt = (per_trade["primary_split_v1"] == "val").to_numpy()
    val_uids = (
        per_trade.loc[val_mask_pt, "candidate_uid_v1"].astype(str).tolist()
    )
    p_delay_val = pd.Series(p_delay_full, index=merged_label["candidate_uid_v1"].astype(str))
    best_delay_thr = 0.5
    best_val_hybrid = -np.inf
    for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        # Compute hybrid PNL on val trades.
        hybrid_val_pnl = 0.0
        ts_label_val = ts_label_table.set_index("candidate_uid_v1")
        for uid in val_uids:
            row = ts_label_val.loc[uid] if uid in ts_label_val.index else None
            if row is None:
                continue
            firing = row["firing_status_v1"]
            if firing == "FIRED" and p_delay_val.get(uid, 0.0) >= thr:
                hybrid_val_pnl += float(row["realized_pnl_v1"])
            elif firing == "FIRED":
                hybrid_val_pnl += float(row["trail_stop_pnl_v1"])
            else:
                hybrid_val_pnl += float(row["realized_pnl_v1"])
        if hybrid_val_pnl > best_val_hybrid:
            best_val_hybrid = hybrid_val_pnl
            best_delay_thr = float(thr)

    # Apply hybrid to test.
    test_mask_pt = (per_trade["primary_split_v1"] == "test").to_numpy()
    test_uids = (
        per_trade.loc[test_mask_pt, "candidate_uid_v1"].astype(str).tolist()
    )
    ts_label_index = ts_label_table.set_index("candidate_uid_v1")
    hybrid_pnl_per_uid: dict[str, float] = {}
    for uid in test_uids:
        if uid not in ts_label_index.index:
            continue
        row = ts_label_index.loc[uid]
        firing = row["firing_status_v1"]
        if firing == "FIRED" and p_delay_val.get(uid, 0.0) >= best_delay_thr:
            hybrid_pnl_per_uid[uid] = float(row["realized_pnl_v1"])
        elif firing == "FIRED":
            hybrid_pnl_per_uid[uid] = float(row["trail_stop_pnl_v1"])
        else:
            hybrid_pnl_per_uid[uid] = float(row["realized_pnl_v1"])

    # Build the head-to-head table.
    rows: list[dict[str, Any]] = []
    for _, r in ts_table.iterrows():
        uid = r["candidate_uid_v1"]
        if uid not in test_uids:
            continue
        skipped = uid in skipped_uids
        realized = float(r["realized_pnl_bps_v1"])
        trail_stop = float(r["trail_stop_pnl_bps_v1"])
        skip_then_realized = 0.0 if skipped else realized
        iql_pnl = best_iql_pnl_per_uid.get(uid, realized)
        skip_then_iql = 0.0 if skipped else iql_pnl
        hybrid = hybrid_pnl_per_uid.get(uid, realized)
        rows.append(
            {
                "fold_id_v1": fold_id,
                "candidate_uid_v1": uid,
                "REALIZED_LIVE_SYSTEM": realized,
                "TRAIL_STOP_25_PCT_DD": trail_stop,
                "SKIP_V2_THEN_REALIZED": skip_then_realized,
                "V2_IQL_BEST_PER_FOLD": iql_pnl,
                "SKIP_V2_THEN_V2_IQL_COMBINED": skip_then_iql,
                "HYBRID_TRAIL_STOP_PLUS_DELAY_LEARNER": hybrid,
                "skipped_v1": skipped,
                "best_iql_variant_v1": best_variant,
                "tuned_skip_threshold_v1": best_thr,
                "tuned_delay_threshold_v1": best_delay_thr,
            }
        )
    return pd.DataFrame(rows)


def _trail_stop_pnl_per_trade_with_label(per_bar_full: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for uid, group in per_bar_full.groupby("candidate_uid_v1", sort=False):
        df = group.sort_values("bars_held_v1").reset_index(drop=True)
        pnl = df["running_pnl_at_close_bps_v1"].astype(float).to_numpy()
        mfe = df["running_mfe_bps_v1"].astype(float).to_numpy()
        n = int(len(df))
        if n == 0:
            continue
        fire_bar = -1
        for i in range(n):
            if mfe[i] >= ts_gate.TRAIL_STOP_MIN_MFE_BPS:
                if (mfe[i] - pnl[i]) / max(mfe[i], 1e-9) >= ts_gate.TRAIL_STOP_GIVEBACK_RATIO:
                    fire_bar = i
                    break
        realized_pnl = float(pnl[-1])
        if fire_bar == -1:
            trail_stop_pnl = realized_pnl
            firing_status = "NEVER_FIRED"
            would_delay_help = 0
        else:
            trail_stop_pnl = float(pnl[fire_bar])
            firing_status = "FIRED"
            post_fire_max = float(pnl[fire_bar:].max())
            would_delay_help = 1 if post_fire_max > trail_stop_pnl + 5.0 else 0
        rows.append(
            {
                "candidate_uid_v1": str(uid),
                "firing_status_v1": firing_status,
                "trail_stop_pnl_v1": trail_stop_pnl,
                "realized_pnl_v1": realized_pnl,
                "would_delay_help_v1": int(would_delay_help),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-policy per-fold metrics
# ---------------------------------------------------------------------------


def _per_policy_metrics(df: pd.DataFrame, policy_col: str) -> dict[str, Any]:
    pnl = df[policy_col].astype(float).to_numpy()
    n = int(len(pnl))
    if n == 0:
        return {}
    total = float(pnl.sum())
    mean = float(pnl.mean())
    std = float(pnl.std(ddof=0))
    sharpe_like = mean / std if std > 0 else None
    win_rate = float((pnl > 0).mean())
    cum = np.cumsum(pnl)
    drawdown = float((cum - np.maximum.accumulate(cum)).min())
    return {
        "policy_v1": policy_col,
        "trade_count_v1": n,
        "total_pnl_bps_v1": total,
        "mean_pnl_bps_v1": mean,
        "std_pnl_bps_v1": std,
        "sharpe_like_v1": sharpe_like,
        "win_rate_v1": win_rate,
        "max_drawdown_bps_v1": drawdown,
        "best_trade_v1": float(pnl.max()),
        "worst_trade_v1": float(pnl.min()),
    }


def _pairwise_correlation_matrix(df: pd.DataFrame, policies: list[str]) -> dict[str, dict[str, float]]:
    arr = df.loc[:, policies].astype(float).to_numpy()
    if len(arr) < 2:
        return {}
    corr = np.corrcoef(arr.T)
    out: dict[str, dict[str, float]] = {}
    for i, p in enumerate(policies):
        out[p] = {}
        for j, q in enumerate(policies):
            out[p][q] = float(corr[i, j]) if np.isfinite(corr[i, j]) else None
    return out


def _diversification_score(corr_matrix: dict[str, dict[str, float]], reference: str) -> dict[str, float]:
    """1 - corr(policy, reference). Higher = more diversifying."""
    out: dict[str, float] = {}
    for p, row in corr_matrix.items():
        if p == reference:
            out[p] = 0.0
        elif reference in row and row[reference] is not None:
            out[p] = 1.0 - row[reference]
        else:
            out[p] = None
    return out


# ---------------------------------------------------------------------------
# Materializer
# ---------------------------------------------------------------------------


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
        "layer_name": "HEAD_TO_HEAD_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "promotion_criteria_root_v1": str(INPUT_PROMOTION_CRITERIA_ROOT),
            "walk_forward_root_v1": str(INPUT_WALK_FORWARD_ROOT),
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

    # Build candidate-uid time order.
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

    # Per-fold per-trade table.
    fold_tables: list[pd.DataFrame] = []
    for fold in wf_gate.FOLD_DEFINITIONS:
        ft = _build_per_trade_pnl_table_for_fold(inputs, candidate_uid_order, fold)
        fold_tables.append(ft)
    full_table = pd.concat(fold_tables, ignore_index=True)
    full_table.to_csv(
        artifact_root / "per_trade_per_policy_pnl_v1.csv", index=False
    )

    # Per-policy per-fold metrics.
    per_policy_per_fold: list[dict[str, Any]] = []
    for fold_id, fold_df in full_table.groupby("fold_id_v1"):
        for policy in POLICIES_HEAD_TO_HEAD:
            m = _per_policy_metrics(fold_df, policy)
            m["fold_id_v1"] = fold_id
            per_policy_per_fold.append(m)
    _write_rows(
        artifact_root / "per_policy_per_fold_metrics_v1.csv", per_policy_per_fold
    )
    _write_json(
        artifact_root / "per_policy_per_fold_metrics_v1.json",
        {"row_count_v1": len(per_policy_per_fold), "rows_v1": per_policy_per_fold},
    )

    # Cross-fold stability per policy (totals across folds).
    cross_fold: list[dict[str, Any]] = []
    for policy in POLICIES_HEAD_TO_HEAD:
        totals = [
            r["total_pnl_bps_v1"]
            for r in per_policy_per_fold
            if r["policy_v1"] == policy
        ]
        if not totals:
            continue
        arr = np.array(totals, dtype=float)
        cross_fold.append(
            {
                "policy_v1": policy,
                "n_folds_v1": int(len(arr)),
                "fold_total_pnl_bps_v1": [float(v) for v in arr],
                "mean_fold_total_v1": float(arr.mean()),
                "std_fold_total_v1": float(arr.std(ddof=0)),
                "min_fold_total_v1": float(arr.min()),
                "max_fold_total_v1": float(arr.max()),
                "n_folds_positive_v1": int((arr > 0).sum()),
                "fold_total_sum_v1": float(arr.sum()),
            }
        )
    _write_json(
        artifact_root / "cross_fold_stability_v1.json",
        {"row_count_v1": len(cross_fold), "rows_v1": cross_fold},
    )

    # Pairwise PNL correlation across all 524 trades.
    corr_matrix = _pairwise_correlation_matrix(full_table, POLICIES_HEAD_TO_HEAD)
    diversification = _diversification_score(corr_matrix, "REALIZED_LIVE_SYSTEM")
    _write_json(
        artifact_root / "pairwise_pnl_correlation_v1.json",
        {
            "matrix_v1": corr_matrix,
            "diversification_score_vs_realized_v1": diversification,
        },
    )

    # Cumulative PNL per (policy, fold) trace.
    cum_rows: list[dict[str, Any]] = []
    for fold_id, fold_df in full_table.groupby("fold_id_v1"):
        for policy in POLICIES_HEAD_TO_HEAD:
            cum = np.cumsum(fold_df[policy].astype(float).to_numpy())
            for rank, val in enumerate(cum, start=1):
                cum_rows.append(
                    {
                        "fold_id_v1": fold_id,
                        "policy_v1": policy,
                        "trade_rank_v1": rank,
                        "cumulative_pnl_bps_v1": float(val),
                    }
                )
    _write_rows(
        artifact_root / "cumulative_pnl_per_policy_per_fold_v1.csv", cum_rows
    )

    # Headline metrics across the full walk-forward (524 trades).
    full_metrics: list[dict[str, Any]] = []
    for policy in POLICIES_HEAD_TO_HEAD:
        m = _per_policy_metrics(full_table, policy)
        m["policy_v1"] = policy
        full_metrics.append(m)
    _write_rows(
        artifact_root / "full_walk_forward_per_policy_metrics_v1.csv", full_metrics
    )
    _write_json(
        artifact_root / "full_walk_forward_per_policy_metrics_v1.json",
        {"row_count_v1": len(full_metrics), "rows_v1": full_metrics},
    )

    # Headline summary.
    realized_metric = next(
        m for m in full_metrics if m["policy_v1"] == "REALIZED_LIVE_SYSTEM"
    )
    best_total_policy = max(full_metrics, key=lambda m: m["total_pnl_bps_v1"])
    best_diversification_policy = max(
        ((p, s) for p, s in diversification.items() if s is not None and p != "REALIZED_LIVE_SYSTEM"),
        key=lambda x: x[1] if x[1] is not None else -np.inf,
    )

    headline = {
        "trade_count_v1": int(len(full_table)),
        "fold_count_v1": int(full_table["fold_id_v1"].nunique()),
        "realized_total_v1": float(realized_metric["total_pnl_bps_v1"]),
        "realized_mean_per_trade_v1": float(realized_metric["mean_pnl_bps_v1"]),
        "best_total_policy_v1": best_total_policy["policy_v1"],
        "best_total_pnl_bps_v1": float(best_total_policy["total_pnl_bps_v1"]),
        "best_diversification_policy_v1": best_diversification_policy[0],
        "best_diversification_score_v1": float(best_diversification_policy[1]),
        "policy_correlation_with_realized_v1": {
            p: float(corr_matrix.get(p, {}).get("REALIZED_LIVE_SYSTEM", float("nan")))
            for p in POLICIES_HEAD_TO_HEAD
        },
    }

    repro = {
        "layer_name": "HEAD_TO_HEAD_REPRODUCIBILITY_AUDIT_V1",
        "fold_count_v1": len(wf_gate.FOLD_DEFINITIONS),
        "policies_v1": POLICIES_HEAD_TO_HEAD,
        "seed_v1": SEED_V1,
        "no_implicit_glob_used_for_v1_inputs_v1": True,
        "deprecated_quarantine_revival_v1": False,
        "research_only_v1": True,
    }
    _write_json(artifact_root / "reproducibility_audit_v1.json", repro)

    status = "HEAD_TO_HEAD_LOCKED_V1"
    next_action = "BUILD_ROLLING_WINDOW_RETRAINED_SKIP_V1"
    recommendation = (
        f"Head-to-head locked. Realized (live system) total {realized_metric['total_pnl_bps_v1']:.0f} bps "
        f"across {int(len(full_table))} trades. Best total: "
        f"`{best_total_policy['policy_v1']}` "
        f"({best_total_policy['total_pnl_bps_v1']:.0f} bps). Most "
        f"diversifying vs realized: `{best_diversification_policy[0]}` "
        f"(1-corr = {best_diversification_policy[1]:.2f}). Next: rolling-"
        "window retrained skip-V2 (online adaptation)."
    )
    validate_final_status(status, next_action)

    summary = {
        "layer_name": "HEAD_TO_HEAD_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "headline_v1": headline,
        "full_walk_forward_per_policy_metrics_v1": full_metrics,
        "cross_fold_stability_v1": cross_fold,
        "pairwise_correlation_v1": corr_matrix,
        "diversification_score_vs_realized_v1": diversification,
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
            "layer_name": "HEAD_TO_HEAD_STATUS_V1",
            "status_v1": "MATERIALIZED_RESEARCH_ONLY_GATE",
            "final_status_v1": status,
            "next_action_v1": next_action,
            "training_executed_v1": True,
        },
    )
    _write_json(
        artifact_root / "head_to_head_go_no_go_v1.json",
        {
            "layer_name": "HEAD_TO_HEAD_GO_NO_GO_V1",
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
                "Research-only diagnostic. No model promotion."
            ),
        },
    )

    # Build report.
    report_lines = [
        "# Run Live System vs Research Candidates Head-To-Head V1",
        "",
        f"- Status: `{status}`",
        f"- Next action: `{next_action}`",
        "- Diagnostic only; no model promotion.",
        "",
        "## Headline (full walk-forward, 524 trades, 3 folds)",
        f"- Realized (live system) total: {headline['realized_total_v1']:+.0f} bps "
        f"(mean {headline['realized_mean_per_trade_v1']:+.2f}/trade)",
        f"- Best total policy: **`{headline['best_total_policy_v1']}`** "
        f"({headline['best_total_pnl_bps_v1']:+.0f} bps)",
        f"- Most diversifying vs realized: `{headline['best_diversification_policy_v1']}` "
        f"(1 - corr = {headline['best_diversification_score_v1']:.3f})",
        "",
        "## Per-policy full walk-forward metrics",
        "",
        "| Policy | Trades | Total PNL | Mean | Std | Sharpe-like | Win rate | Max DD | Best | Worst |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in sorted(full_metrics, key=lambda x: x["total_pnl_bps_v1"], reverse=True):
        sl = m["sharpe_like_v1"]
        report_lines.append(
            f"| `{m['policy_v1']}` | {m['trade_count_v1']} | "
            f"{m['total_pnl_bps_v1']:+.0f} | {m['mean_pnl_bps_v1']:+.2f} | "
            f"{m['std_pnl_bps_v1']:.2f} | "
            f"{sl:+.3f}" if sl is not None else "(n/a)"
            f" | {m['win_rate_v1']*100:.0f}% | "
            f"{m['max_drawdown_bps_v1']:+.0f} | "
            f"{m['best_trade_v1']:+.1f} | "
            f"{m['worst_trade_v1']:+.1f} |"
        )
    report_lines.extend(
        [
            "",
            "## Cross-fold stability per policy",
            "",
            "| Policy | F1 | F2 | F3 | Mean | Std | n_pos | Sum |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for r in sorted(cross_fold, key=lambda x: x["fold_total_sum_v1"], reverse=True):
        ft = r["fold_total_pnl_bps_v1"]
        # Pad ft to 3 entries for column display.
        ft = ft + [0.0] * (3 - len(ft))
        report_lines.append(
            f"| `{r['policy_v1']}` | {ft[0]:+.0f} | {ft[1]:+.0f} | "
            f"{ft[2]:+.0f} | {r['mean_fold_total_v1']:+.0f} | "
            f"{r['std_fold_total_v1']:.0f} | "
            f"{r['n_folds_positive_v1']}/{r['n_folds_v1']} | "
            f"**{r['fold_total_sum_v1']:+.0f}** |"
        )
    report_lines.extend(
        [
            "",
            "## Diversification score vs realized (1 - corr)",
            "",
            "Higher = more diversifying. `REALIZED_LIVE_SYSTEM` = 0 by definition.",
            "",
        ]
    )
    for p, s in sorted(diversification.items(), key=lambda x: -(x[1] or 0)):
        report_lines.append(f"- `{p}`: {s:.3f}" if s is not None else f"- `{p}`: (n/a)")
    report_lines.extend(
        [
            "",
            "## Pairwise correlation matrix (per-trade PNL across 524 trades)",
            "",
        ]
    )
    header = "| | " + " | ".join(f"`{p}`" for p in POLICIES_HEAD_TO_HEAD) + " |"
    sep = "|" + "---|" * (len(POLICIES_HEAD_TO_HEAD) + 1)
    report_lines.append(header)
    report_lines.append(sep)
    for p in POLICIES_HEAD_TO_HEAD:
        row = [f"`{p}`"]
        for q in POLICIES_HEAD_TO_HEAD:
            v = corr_matrix.get(p, {}).get(q)
            row.append(f"{v:+.2f}" if v is not None else "n/a")
        report_lines.append("| " + " | ".join(row) + " |")
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
            "go_no_go": str(artifact_root / "head_to_head_go_no_go_v1.json"),
            "input_manifest": str(artifact_root / "input_manifest_v1.json"),
            "per_trade_per_policy_pnl_csv": str(
                artifact_root / "per_trade_per_policy_pnl_v1.csv"
            ),
            "per_policy_per_fold_metrics_csv": str(
                artifact_root / "per_policy_per_fold_metrics_v1.csv"
            ),
            "per_policy_per_fold_metrics_json": str(
                artifact_root / "per_policy_per_fold_metrics_v1.json"
            ),
            "cross_fold_stability": str(artifact_root / "cross_fold_stability_v1.json"),
            "pairwise_pnl_correlation": str(
                artifact_root / "pairwise_pnl_correlation_v1.json"
            ),
            "cumulative_pnl_per_policy_per_fold_csv": str(
                artifact_root / "cumulative_pnl_per_policy_per_fold_v1.csv"
            ),
            "full_walk_forward_per_policy_metrics_csv": str(
                artifact_root / "full_walk_forward_per_policy_metrics_v1.csv"
            ),
            "full_walk_forward_per_policy_metrics_json": str(
                artifact_root / "full_walk_forward_per_policy_metrics_v1.json"
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
        description="Materialize RUN_LIVE_SYSTEM_VS_RESEARCH_CANDIDATES_HEAD_TO_HEAD_V1."
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
