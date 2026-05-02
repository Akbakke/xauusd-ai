#!/usr/bin/env python3
"""Deep-dive analysis of why TRAIL_STOP_25_PCT_DD beats our learned models.

Background
----------
Across all gates the simple TRAIL_STOP_25_PCT_DD rule (+1052 bps single-
fold test PNL) consistently outperformed our learned models. Walk-forward
validation showed our models are NOT_STABLE while trail-stop's rule-based
nature should be regime-invariant by construction. This gate quantifies
that hypothesis and identifies trail-stop's own failure modes that any
learned model would need to fix to genuinely beat it.

Per-fold analysis on the 3 walk-forward folds:

  1. Apply trail-stop on each fold's test set.
  2. Per trade record: when did the rule fire (bar index), what was
     running_mfe and running_giveback at firing, what was realized PNL,
     was the trade ultimately a winner / loser / breakeven.
  3. Compare to PEAK_MFE_ORACLE (perfect foresight): for each trade,
     what was the best possible exit?
  4. Decompose trail-stop errors into:
     - FIRED_BEFORE_PEAK_MFE: rule fired but a higher MFE would come later
       in the same trade (regret of early exit)
     - FIRED_AFTER_PEAK_GIVEBACK: rule fired after MFE was already given
       back too much (failure to lock peak)
     - NEVER_FIRED: trade never had MFE > 25%-of-peak-giveback threshold;
       fell to trade-end (defaulted to realized exit)
  5. Per-fold: the per-trade firing distribution (giveback% at firing).

Output:
  - per_trade_trail_stop_decomposition_v1.csv (one row per test trade per fold)
  - failure_mode_breakdown_per_fold_v1.csv
  - hybrid_recommendation_v1.json: recommendations for "learn small
    adjustments to trail-stop" rather than "replace trail-stop entirely".

Research-only diagnostic; no trail-stop modification, no model promotion.
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
    materialize_exit_off_policy_eval_harness_v1 as eval_gate,
)
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
ACTION = "INVESTIGATE_TRAIL_STOP_DEEP_DIVE_V1"

INPUT_RECOVERY_ROOT = v2_train_gate.INPUT_RECOVERY_ROOT
INPUT_SPLIT_ROOT = v2_train_gate.INPUT_SPLIT_ROOT
INPUT_V2_CONTRACT_ROOT = v2_train_gate.INPUT_V2_CONTRACT_ROOT
INPUT_EVAL_HARNESS_ROOT = v2_train_gate.INPUT_EVAL_HARNESS_ROOT
BASE34_M5_FEATURES_PATH = v2_train_gate.BASE34_M5_FEATURES_PATH

QUARANTINE_FORBIDDEN_PATH_FRAGMENTS = ("gx1/quarantine", "gx1.quarantine")
SEED_V1 = 20260430

# The eval-harness's TRAIL_STOP_25_PCT_DD rule fires when running_giveback /
# running_mfe >= 0.25, after running_mfe has reached at least some positive
# value. We re-implement the firing-bar identification here for diagnostic
# purposes (the harness already does this internally).
TRAIL_STOP_GIVEBACK_RATIO = 0.25
TRAIL_STOP_MIN_MFE_BPS = 5.0  # rule needs MFE > 5 bps to be meaningful

ALLOWED_FINAL_STATUSES = {
    "INVESTIGATE_TRAIL_STOP_LOCKED_V1",
    "INVESTIGATE_TRAIL_STOP_BLOCKED_BY_INPUT_LOCK_MISSING_V1",
}

ALLOWED_NEXT_ACTIONS = {
    "BUILD_HYBRID_TRAIL_STOP_PLUS_SMALL_ADJUSTMENT_LEARNER_V1",
    "BUILD_REGIME_CONDITIONED_SKIP_V1",
    "ACCEPT_TRAIL_STOP_AS_RESEARCH_BASELINE_V1",
    "REPAIR_RESEARCH_STACK_BEFORE_FURTHER_WORK_V1",
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
# Per-trade trail-stop decomposition
# ---------------------------------------------------------------------------


def _decompose_trade(
    per_bar_trade: pd.DataFrame, fold_id: str
) -> dict[str, Any]:
    """For one trade's per-bar trajectory, identify trail-stop firing bar
    and classify the failure mode vs peak-MFE oracle."""
    df = per_bar_trade.sort_values("bars_held_v1").reset_index(drop=True)
    candidate_uid = str(df["candidate_uid_v1"].iloc[0])

    pnl = df["running_pnl_at_close_bps_v1"].astype(float).to_numpy()
    mfe = df["running_mfe_bps_v1"].astype(float).to_numpy()
    n = int(len(df))
    if n == 0:
        return {}

    # Trail-stop fires at the first bar where:
    #   running_mfe >= TRAIL_STOP_MIN_MFE_BPS AND
    #   (running_mfe - running_pnl_at_close) / running_mfe >= 0.25
    # i.e. giveback ratio >= 0.25.
    fire_bar = -1
    for i in range(n):
        if mfe[i] >= TRAIL_STOP_MIN_MFE_BPS:
            giveback_ratio = (mfe[i] - pnl[i]) / max(mfe[i], 1e-9)
            if giveback_ratio >= TRAIL_STOP_GIVEBACK_RATIO:
                fire_bar = i
                break

    realized_exit_bar = n - 1
    realized_pnl = float(pnl[realized_exit_bar])
    peak_mfe_bar = int(np.argmax(mfe))
    peak_mfe = float(mfe[peak_mfe_bar])
    peak_pnl_at_close = float(pnl[int(np.argmax(pnl))])  # best closing PnL
    best_pnl_bar = int(np.argmax(pnl))

    if fire_bar == -1:
        trail_stop_pnl = realized_pnl
        firing_status = "NEVER_FIRED"
        giveback_at_fire = None
        mfe_at_fire = None
        bars_to_fire = None
    else:
        trail_stop_pnl = float(pnl[fire_bar])
        firing_status = "FIRED"
        giveback_at_fire = float(mfe[fire_bar] - pnl[fire_bar])
        mfe_at_fire = float(mfe[fire_bar])
        bars_to_fire = int(df["bars_held_v1"].iloc[fire_bar])

    # Classification of trail-stop's outcome vs ORACLE peak-pnl.
    if firing_status == "NEVER_FIRED":
        if realized_pnl > 0:
            failure_mode = "NEVER_FIRED_TRADE_WON_AT_REALIZED"
        elif realized_pnl < 0:
            failure_mode = "NEVER_FIRED_TRADE_LOST_AT_REALIZED"
        else:
            failure_mode = "NEVER_FIRED_TRADE_FLAT"
    else:
        # Was there a higher PNL after the firing bar?
        post_fire_pnl_max = float(pnl[fire_bar:].max())
        if post_fire_pnl_max > trail_stop_pnl + 5.0:
            failure_mode = "FIRED_BEFORE_PEAK_PNL_REGRET_EARLY_EXIT"
        elif trail_stop_pnl < peak_pnl_at_close - 5.0:
            failure_mode = "FIRED_AFTER_PEAK_PNL_REGRET_LATE_EXIT"
        else:
            failure_mode = "FIRED_AT_OR_NEAR_PEAK_PNL_OK"

    return {
        "fold_id_v1": fold_id,
        "candidate_uid_v1": candidate_uid,
        "n_bars_v1": n,
        "fire_bar_index_v1": int(fire_bar) if fire_bar >= 0 else None,
        "bars_to_fire_v1": bars_to_fire,
        "firing_status_v1": firing_status,
        "trail_stop_pnl_v1": trail_stop_pnl,
        "realized_pnl_v1": realized_pnl,
        "peak_mfe_bps_v1": peak_mfe,
        "peak_pnl_at_close_bps_v1": peak_pnl_at_close,
        "peak_pnl_bar_v1": best_pnl_bar,
        "delta_trail_stop_minus_realized_v1": trail_stop_pnl - realized_pnl,
        "delta_peak_pnl_minus_trail_stop_v1": peak_pnl_at_close - trail_stop_pnl,
        "giveback_at_fire_bps_v1": giveback_at_fire,
        "mfe_at_fire_bps_v1": mfe_at_fire,
        "failure_mode_v1": failure_mode,
    }


def _decompose_per_fold(
    per_bar_full: pd.DataFrame, fold_id: str
) -> list[dict[str, Any]]:
    test_mask = per_bar_full["primary_split_v1"] == "test"
    test_df = per_bar_full[test_mask]
    rows: list[dict[str, Any]] = []
    for uid, group in test_df.groupby("candidate_uid_v1", sort=False):
        rows.append(_decompose_trade(group, fold_id))
    return [r for r in rows if r]


def _summarize_failure_modes(
    decomp_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not decomp_rows:
        return {}
    df = pd.DataFrame(decomp_rows)
    n = int(len(df))
    failure_counts = df["failure_mode_v1"].value_counts().to_dict()
    failure_pcts = {k: 100.0 * v / n for k, v in failure_counts.items()}
    pnl_by_mode = (
        df.groupby("failure_mode_v1")["trail_stop_pnl_v1"]
        .agg(["count", "mean", "sum"])
        .to_dict("index")
    )
    return {
        "trade_count_v1": n,
        "failure_mode_counts_v1": {k: int(v) for k, v in failure_counts.items()},
        "failure_mode_pcts_v1": {k: float(v) for k, v in failure_pcts.items()},
        "pnl_by_mode_v1": {
            mode: {
                "count_v1": int(stats["count"]),
                "mean_pnl_v1": float(stats["mean"]),
                "sum_pnl_v1": float(stats["sum"]),
            }
            for mode, stats in pnl_by_mode.items()
        },
        "trail_stop_total_pnl_v1": float(df["trail_stop_pnl_v1"].sum()),
        "realized_total_pnl_v1": float(df["realized_pnl_v1"].sum()),
        "peak_pnl_oracle_total_v1": float(df["peak_pnl_at_close_bps_v1"].sum()),
        "delta_trail_stop_vs_realized_v1": float(
            df["delta_trail_stop_minus_realized_v1"].sum()
        ),
        "delta_peak_oracle_vs_trail_stop_v1": float(
            df["delta_peak_pnl_minus_trail_stop_v1"].sum()
        ),
        "giveback_at_fire_p25_v1": float(
            df["giveback_at_fire_bps_v1"].dropna().quantile(0.25)
        )
        if df["giveback_at_fire_bps_v1"].notna().any()
        else None,
        "giveback_at_fire_p50_v1": float(
            df["giveback_at_fire_bps_v1"].dropna().quantile(0.50)
        )
        if df["giveback_at_fire_bps_v1"].notna().any()
        else None,
        "giveback_at_fire_p75_v1": float(
            df["giveback_at_fire_bps_v1"].dropna().quantile(0.75)
        )
        if df["giveback_at_fire_bps_v1"].notna().any()
        else None,
        "bars_to_fire_p50_v1": float(
            df["bars_to_fire_v1"].dropna().quantile(0.50)
        )
        if df["bars_to_fire_v1"].notna().any()
        else None,
    }


# ---------------------------------------------------------------------------
# Hybrid-recommendation analysis
# ---------------------------------------------------------------------------


def _recommend_hybrid(
    per_fold_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Synthesize a high-level recommendation for hybrid trail-stop +
    learned-adjustment design."""
    failure_pcts_all_folds: dict[str, list[float]] = {}
    for fold_summary in per_fold_summaries.values():
        for mode, pct in fold_summary.get("failure_mode_pcts_v1", {}).items():
            failure_pcts_all_folds.setdefault(mode, []).append(float(pct))
    failure_pct_means = {
        mode: float(np.mean(vals)) for mode, vals in failure_pcts_all_folds.items()
    }
    # Sort failure modes by importance (highest mean pct).
    ranked = sorted(failure_pct_means.items(), key=lambda x: -x[1])
    primary_mode = ranked[0][0] if ranked else None

    notes: list[str] = []
    if primary_mode and "FIRED_BEFORE_PEAK_PNL" in primary_mode:
        notes.append(
            "Trail-stop's largest residual error is firing too early (before peak). "
            "A learned adjustment that POSTPONES firing in conditions where the "
            "trade is likely to keep moving favorably could lift PNL further."
        )
    if primary_mode and "FIRED_AFTER_PEAK_PNL" in primary_mode:
        notes.append(
            "Trail-stop's largest residual error is firing too late (after peak). "
            "A learned adjustment that ACCELERATES firing on volatile / regime-shifted "
            "trades could capture more peak."
        )
    if primary_mode and "NEVER_FIRED_TRADE_LOST" in primary_mode:
        notes.append(
            "Trail-stop frequently fails to fire on losing trades (no MFE peak to "
            "trail). A complementary 'cut-loss' rule (stop-loss based on MAE / time-out) "
            "or skip-classifier at entry would address this gap."
        )
    if primary_mode and "FIRED_AT_OR_NEAR_PEAK" in primary_mode:
        notes.append(
            "Trail-stop fires well most of the time. Marginal gains likely require "
            "a different model class (proper IQL with pessimism, gradient boosting) "
            "or expanded data, not a simple hybrid."
        )

    return {
        "primary_failure_mode_v1": primary_mode,
        "failure_pct_means_v1": failure_pct_means,
        "ranked_failure_modes_v1": ranked,
        "design_notes_v1": notes,
    }


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
        "layer_name": "INVESTIGATE_TRAIL_STOP_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "recovery_root_v1": str(INPUT_RECOVERY_ROOT),
            "split_root_v1": str(INPUT_SPLIT_ROOT),
            "v2_contract_root_v1": str(INPUT_V2_CONTRACT_ROOT),
            "eval_harness_root_v1": str(INPUT_EVAL_HARNESS_ROOT),
        },
        "files_used_v1": files,
        "immutable_input_status_v1": "HASHED_EXPLICIT_ROOTS_ONLY",
        "no_implicit_latest_glob_selection_v1": True,
        "previous_artifacts_mutated_v1": False,
        "research_only_contract_v1": True,
        "iql_training_run_v1": False,
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

    inputs = wf_gate._load_inputs()
    _write_json(
        artifact_root / "input_manifest_v1.json",
        _build_input_manifest(inputs, artifact_root),
    )

    # Build candidate-uid time order and per-fold split assignment.
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

    # For each fold, decompose trail-stop on the fold's test set.
    all_decomp_rows: list[dict[str, Any]] = []
    per_fold_summaries: dict[str, dict[str, Any]] = {}
    for fold in wf_gate.FOLD_DEFINITIONS:
        fold_id = fold["fold_id_v1"]
        uid_to_split = wf_gate._assign_fold_split(candidate_uid_order, fold)
        per_bar, _, _ = wf_gate._build_per_bar_for_fold(inputs, uid_to_split)
        decomp = _decompose_per_fold(per_bar, fold_id)
        all_decomp_rows.extend(decomp)
        per_fold_summaries[fold_id] = _summarize_failure_modes(decomp)

    _write_rows(
        artifact_root / "per_trade_trail_stop_decomposition_v1.csv", all_decomp_rows
    )
    _write_json(
        artifact_root / "per_trade_trail_stop_decomposition_v1.json",
        {"row_count_v1": len(all_decomp_rows), "rows_v1": all_decomp_rows},
    )
    _write_json(
        artifact_root / "failure_mode_breakdown_per_fold_v1.json",
        {"per_fold_v1": per_fold_summaries},
    )

    # Hybrid recommendation.
    hybrid = _recommend_hybrid(per_fold_summaries)
    _write_json(artifact_root / "hybrid_recommendation_v1.json", hybrid)

    # Headline.
    total_decomp = pd.DataFrame(all_decomp_rows)
    headline = {
        "fold_count_v1": len(per_fold_summaries),
        "total_test_trades_decomposed_v1": int(len(total_decomp)),
        "primary_failure_mode_v1": hybrid.get("primary_failure_mode_v1"),
        "failure_pct_means_across_folds_v1": hybrid.get("failure_pct_means_v1", {}),
        "trail_stop_total_pnl_per_fold_v1": {
            fold_id: float(s["trail_stop_total_pnl_v1"])
            for fold_id, s in per_fold_summaries.items()
        },
        "realized_total_pnl_per_fold_v1": {
            fold_id: float(s["realized_total_pnl_v1"])
            for fold_id, s in per_fold_summaries.items()
        },
        "peak_pnl_oracle_total_per_fold_v1": {
            fold_id: float(s["peak_pnl_oracle_total_v1"])
            for fold_id, s in per_fold_summaries.items()
        },
    }

    repro = {
        "layer_name": "INVESTIGATE_TRAIL_STOP_REPRODUCIBILITY_AUDIT_V1",
        "fold_count_v1": len(wf_gate.FOLD_DEFINITIONS),
        "trail_stop_giveback_ratio_v1": TRAIL_STOP_GIVEBACK_RATIO,
        "trail_stop_min_mfe_bps_v1": TRAIL_STOP_MIN_MFE_BPS,
        "no_implicit_glob_used_for_v1_inputs_v1": True,
        "deprecated_quarantine_revival_v1": False,
        "research_only_v1": True,
    }
    _write_json(artifact_root / "reproducibility_audit_v1.json", repro)

    status = "INVESTIGATE_TRAIL_STOP_LOCKED_V1"
    primary_mode = hybrid.get("primary_failure_mode_v1") or ""
    # Pick next action based on primary failure mode.
    if "NEVER_FIRED_TRADE_LOST" in primary_mode:
        next_action = "BUILD_REGIME_CONDITIONED_SKIP_V1"
        recommendation = (
            "Trail-stop's primary residual error is failing to fire on losing "
            "trades (no MFE peak to trail). A skip-classifier at entry is the "
            "right complement, not a hybrid trail-stop adjustment. Build a "
            "regime-conditioned skip-classifier as the next gate."
        )
    elif "FIRED_BEFORE_PEAK_PNL" in primary_mode:
        next_action = "BUILD_HYBRID_TRAIL_STOP_PLUS_SMALL_ADJUSTMENT_LEARNER_V1"
        recommendation = (
            "Trail-stop's primary residual error is firing too early. A "
            "learned 'postpone' adjustment that uses entry-context features "
            "to selectively delay firing on momentum-rich trades could lift "
            "PNL further."
        )
    elif "FIRED_AFTER_PEAK_PNL" in primary_mode:
        next_action = "BUILD_HYBRID_TRAIL_STOP_PLUS_SMALL_ADJUSTMENT_LEARNER_V1"
        recommendation = (
            "Trail-stop's primary residual error is firing too late. A "
            "learned 'accelerate' adjustment based on volatility / regime "
            "features could capture more peak."
        )
    else:
        next_action = "ACCEPT_TRAIL_STOP_AS_RESEARCH_BASELINE_V1"
        recommendation = (
            "Trail-stop fires near-optimally on most trades. Marginal gains "
            "likely require a different model class or expanded data, not a "
            "simple hybrid adjustment. Accept trail-stop as the research baseline."
        )
    validate_final_status(status, next_action)

    summary = {
        "layer_name": "INVESTIGATE_TRAIL_STOP_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "headline_v1": headline,
        "hybrid_recommendation_v1": hybrid,
        "per_fold_summaries_v1": per_fold_summaries,
        "research_only_v1": True,
        "iql_training_run_v1": False,
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
            "layer_name": "INVESTIGATE_TRAIL_STOP_STATUS_V1",
            "status_v1": "MATERIALIZED_RESEARCH_ONLY_GATE",
            "final_status_v1": status,
            "next_action_v1": next_action,
            "training_executed_v1": False,
        },
    )
    _write_json(
        artifact_root / "investigate_trail_stop_deep_dive_go_no_go_v1.json",
        {
            "layer_name": "INVESTIGATE_TRAIL_STOP_GO_NO_GO_V1",
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
            "training_allowed_v1": False,
            "downstream_block_v1": (
                "Research-only diagnostic; no model promotion, no trail-stop "
                "modification."
            ),
        },
    )

    # Build the report.
    report_lines = [
        "# Investigate Trail-Stop Deep-Dive V1",
        "",
        f"- Status: `{status}`",
        f"- Next action: `{next_action}`",
        "- Training: research-only diagnostic.",
        "",
        "## Headline",
        f"- Folds analyzed: {headline['fold_count_v1']}",
        f"- Total test trades decomposed: {headline['total_test_trades_decomposed_v1']}",
        f"- Primary failure mode (across folds): `{headline['primary_failure_mode_v1']}`",
        "",
        "### Per-fold trail-stop PNL vs realized vs peak-pnl-oracle",
        "",
        "| Fold | Trail-stop PNL | Realized PNL | Peak-PNL oracle | Trail-stop vs realized | Oracle vs trail-stop |",
        "|---|---|---|---|---|---|",
    ]
    for fold_id, s in per_fold_summaries.items():
        report_lines.append(
            f"| `{fold_id}` | {s['trail_stop_total_pnl_v1']:.0f} | "
            f"{s['realized_total_pnl_v1']:.0f} | "
            f"{s['peak_pnl_oracle_total_v1']:.0f} | "
            f"{s['delta_trail_stop_vs_realized_v1']:+.0f} | "
            f"{s['delta_peak_oracle_vs_trail_stop_v1']:+.0f} |"
        )
    report_lines.extend(
        [
            "",
            "## Failure mode breakdown per fold",
        ]
    )
    for fold_id, s in per_fold_summaries.items():
        report_lines.append(f"### `{fold_id}` (n={s['trade_count_v1']})")
        for mode, count in sorted(
            s["failure_mode_counts_v1"].items(), key=lambda x: -x[1]
        ):
            pct = s["failure_mode_pcts_v1"][mode]
            mean_pnl = s["pnl_by_mode_v1"][mode]["mean_pnl_v1"]
            report_lines.append(
                f"- `{mode}`: {count} trades ({pct:.1f}%), mean PNL "
                f"{mean_pnl:+.1f} bps"
            )
        report_lines.append("")
    report_lines.extend(
        [
            "## Hybrid recommendation",
            "",
        ]
    )
    if hybrid.get("ranked_failure_modes_v1"):
        report_lines.append("Ranked failure modes (mean % across folds):")
        for mode, pct in hybrid["ranked_failure_modes_v1"]:
            report_lines.append(f"- `{mode}`: {pct:.1f}%")
        report_lines.append("")
    for note in hybrid.get("design_notes_v1", []):
        report_lines.append(f"- {note}")
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
                artifact_root / "investigate_trail_stop_deep_dive_go_no_go_v1.json"
            ),
            "input_manifest": str(artifact_root / "input_manifest_v1.json"),
            "per_trade_trail_stop_decomposition_csv": str(
                artifact_root / "per_trade_trail_stop_decomposition_v1.csv"
            ),
            "per_trade_trail_stop_decomposition_json": str(
                artifact_root / "per_trade_trail_stop_decomposition_v1.json"
            ),
            "failure_mode_breakdown_per_fold": str(
                artifact_root / "failure_mode_breakdown_per_fold_v1.json"
            ),
            "hybrid_recommendation": str(artifact_root / "hybrid_recommendation_v1.json"),
            "reproducibility_audit": str(
                artifact_root / "reproducibility_audit_v1.json"
            ),
            "report": str(artifact_root / "report_v1.md"),
        },
        "read_only_references_v1": True,
        "trained_model_v1": False,
        "not_controller_v1": True,
        "not_live_gate_v1": True,
    }
    _write_json(artifact_root / "manifest_v1.json", artifact_manifest)

    return {"artifact_root": str(artifact_root), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize INVESTIGATE_TRAIL_STOP_DEEP_DIVE_V1.")
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
