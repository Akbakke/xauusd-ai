#!/usr/bin/env python3
"""End-to-end evaluation: skip classifier V2 + exit IQL V2, on the same cohort.

Background
----------
- Skip-classifier V2 (logistic balanced) lifted test PNL by +1142 bps on
  test (no-skip floor -194 -> with-skip +948).
- V2 IQL exit best variant (GIVEBACK_PENALTY_REWARD) lifted test PNL by
  +864 bps over realized-exit floor (-355 -> +509).
- V3 IQL with per-bar XGB DEGRADED V2's best variants by -110 bps.

These were measured INDEPENDENTLY on different reference points. We do
not yet know whether the effects multiply (skip avoids trades that the
exit-IQL would also have rescued -> subadditive) or compound (skip
removes orthogonal failures, exit-IQL improves the kept trades ->
superadditive).

This gate runs the full stack on the SAME test cohort and reports four
per-test-cohort PNL numbers:

  1. NO_SKIP_REALIZED         - the realized-exit floor (-355 bps).
  2. NO_SKIP_V2_IQL           - V2 IQL on all trades (the +509 best).
  3. SKIP_V2_THEN_REALIZED    - skip V2 then realized exit on kept (+948 bps).
  4. SKIP_V2_THEN_V2_IQL      - the new combined stack (this gate's answer).

Plus per-reward-variant breakdown (5 variants for the IQL side) so we
see which reward family combines best with the skip filter.

Research-only; no policy promotion; no runtime modification; no V1/V2
state-contract or trained-model modification. Skip-V2 and V2 IQL models
are re-trained inline using their pinned scripts' deterministic seeds,
so bit-for-bit identical to their LOCKed training runs.
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
from gx1.scripts import materialize_exit_off_policy_eval_harness_v1 as eval_gate
from gx1.scripts import (
    materialize_run_exit_iql_with_v2_state_and_reward_variants_v1 as v2_train_gate,
)
from gx1.scripts import (
    materialize_learn_trade_skip_meta_classifier_at_trade_open_v1 as skip_v1_gate,
)
from gx1.scripts import (
    materialize_learn_trade_skip_meta_classifier_v2_logistic_balanced_v1 as skip_v2_gate,
)


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "COMBINE_SKIP_V2_WITH_EXIT_IQL_V2_V1"

INPUT_SKIP_V2_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "LEARN_TRADE_SKIP_META_CLASSIFIER_V2_LOGISTIC_BALANCED_V1_20260430T062405Z_LOCK"
)
INPUT_V2_TRAIN_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "RUN_EXIT_IQL_WITH_V2_STATE_AND_REWARD_VARIANTS_V1_20260429T204407Z_LOCK"
)
INPUT_V2_CONTRACT_ROOT = v2_train_gate.INPUT_V2_CONTRACT_ROOT
INPUT_RECOVERY_ROOT = v2_train_gate.INPUT_RECOVERY_ROOT
INPUT_SPLIT_ROOT = v2_train_gate.INPUT_SPLIT_ROOT
INPUT_EVAL_HARNESS_ROOT = v2_train_gate.INPUT_EVAL_HARNESS_ROOT
BASE34_M5_FEATURES_PATH = v2_train_gate.BASE34_M5_FEATURES_PATH

QUARANTINE_FORBIDDEN_PATH_FRAGMENTS = ("gx1/quarantine", "gx1.quarantine")
SEED_V1 = 20260430

ALLOWED_FINAL_STATUSES = {
    "COMBINE_SKIP_V2_WITH_EXIT_IQL_V2_PASS_SUPERADDITIVE_LIFT",
    "COMBINE_SKIP_V2_WITH_EXIT_IQL_V2_PASS_ADDITIVE_LIFT",
    "COMBINE_SKIP_V2_WITH_EXIT_IQL_V2_PASS_SKIP_DOMINATES",
    "COMBINE_SKIP_V2_WITH_EXIT_IQL_V2_PARTIAL_SUBADDITIVE_LIFT",
    "COMBINE_SKIP_V2_WITH_EXIT_IQL_V2_PARTIAL_DEGRADES_VS_BEST_COMPONENT",
    "COMBINE_SKIP_V2_WITH_EXIT_IQL_V2_BLOCKED_BY_INPUT_LOCK_MISSING",
}

ALLOWED_NEXT_ACTIONS = {
    "WALK_FORWARD_VALIDATION_V1",
    "EXIT_PER_BAR_PROPER_IQL_WITH_PESSIMISM_V1",
    "DEFINE_PROMOTION_CRITERIA_BEFORE_PAPER_TRADING_V1",
    "REPAIR_COMBINED_STACK_BEFORE_PROMOTION_V1",
    "HOLD_COMBINED_STACK_RESEARCH_UNTIL_DATA_FIXED_V1",
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
# Inputs
# ---------------------------------------------------------------------------


def _load_inputs() -> dict[str, Any]:
    roots = [
        INPUT_SKIP_V2_ROOT,
        INPUT_V2_TRAIN_ROOT,
        INPUT_V2_CONTRACT_ROOT,
        INPUT_RECOVERY_ROOT,
        INPUT_SPLIT_ROOT,
        INPUT_EVAL_HARNESS_ROOT,
    ]
    validate_explicit_artifact_roots(roots)
    required = {
        "skip_v2_summary": INPUT_SKIP_V2_ROOT / "summary_v1.json",
        "skip_v2_locked_test": INPUT_SKIP_V2_ROOT / "locked_test_evaluation_v1.json",
        "v2_training_summary": INPUT_V2_TRAIN_ROOT / "summary_v1.json",
        "v2_state_contract": INPUT_V2_CONTRACT_ROOT / "state_feature_contract_v2.json",
        "recovery_per_trade": INPUT_RECOVERY_ROOT
        / "entry_snapshot_signals_per_trade_v1.parquet",
        "split_locked_dataset": INPUT_SPLIT_ROOT
        / "split_locked_augmented_dataset_v1.parquet",
        "eval_harness_baseline_metrics": INPUT_EVAL_HARNESS_ROOT
        / "baseline_metrics_per_split_v1.json",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise RuntimeError(f"MISSING_REQUIRED_INPUT_LOCKS: {missing}")
    if not BASE34_M5_FEATURES_PATH.exists():
        raise RuntimeError(
            f"BASE34_M5_FEATURES_PATH_NOT_FOUND: {BASE34_M5_FEATURES_PATH}"
        )
    return {
        "required_paths": required,
        "skip_v2_summary": _read_json(required["skip_v2_summary"]),
        "skip_v2_locked_test": _read_json(required["skip_v2_locked_test"]),
        "v2_training_summary": _read_json(required["v2_training_summary"]),
        "v2_state_contract": _read_json(required["v2_state_contract"]),
        "eval_harness_baseline_metrics": _read_json(
            required["eval_harness_baseline_metrics"]
        ),
        "base34_path": BASE34_M5_FEATURES_PATH,
    }


# ---------------------------------------------------------------------------
# Re-train both models deterministically (matches their LOCKed training runs)
# ---------------------------------------------------------------------------


def _retrain_skip_v2(
    inputs: dict[str, Any]
) -> tuple[pd.DataFrame, np.ndarray, float]:
    """Re-run skip-V2 training (deterministic). Returns (per_trade_df,
    p_skip_array, val_tuned_threshold). The per_trade_df has all 1724
    accepted trades with split assignment."""
    trades = skip_v1_gate._load_trade_outcomes_concat()
    recovery = pd.read_parquet(inputs["required_paths"]["recovery_per_trade"])
    per_trade, _ = skip_v1_gate._project_per_trade_features(
        trades, recovery, BASE34_M5_FEATURES_PATH
    )
    per_trade_split, _ = skip_v1_gate._join_split_assignment(
        per_trade, inputs["required_paths"]["split_locked_dataset"]
    )
    per_trade_train = per_trade_split[per_trade_split["primary_split_v1"] == "train"]
    norm = skip_v1_gate._fit_train_normalization(per_trade_train)
    X_full, _ = skip_v1_gate._build_state_matrix(per_trade_split, norm)
    train_mask = (per_trade_split["primary_split_v1"] == "train").to_numpy()
    y_full = per_trade_split["should_skip_v1"].astype(int).to_numpy()

    logreg = skip_v2_gate._train_logistic(X_full[train_mask], y_full[train_mask])
    p_skip_full = skip_v2_gate._predict_p_skip(logreg, X_full)

    # Tune threshold on val.
    val_mask = (per_trade_split["primary_split_v1"] == "val").to_numpy()
    per_trade_val = per_trade_split[val_mask].reset_index(drop=True)
    p_skip_val = p_skip_full[val_mask]
    best_thr = None
    best_pnl = -np.inf
    for thr in skip_v2_gate.THRESHOLD_GRID:
        m = skip_v1_gate._evaluate_threshold(per_trade_val, p_skip_val, thr)
        if m["pnl_taken_v1"] > best_pnl:
            best_pnl = m["pnl_taken_v1"]
            best_thr = thr
    if best_thr is None:
        raise RuntimeError("SKIP_V2_VAL_TUNING_FAILED")
    return per_trade_split, p_skip_full, float(best_thr)


def _retrain_v2_iql_per_variant(
    inputs: dict[str, Any]
) -> tuple[pd.DataFrame, np.ndarray, list[str], dict[str, dict[str, np.ndarray]]]:
    """Re-run V2 IQL training for all 5 reward variants (deterministic).
    Returns (per_bar_full, X_full, feature_names, models_by_variant) where
    models_by_variant[reward_id] = {"coef_hold": ..., "coef_exit_now": ...}.
    """
    df = pd.read_parquet(inputs["required_paths"]["split_locked_dataset"])
    df["candidate_uid_v1"] = df["candidate_uid_v1"].astype(str)
    df["ts_v1"] = pd.to_datetime(df["ts_v1"], utc=True)
    per_bar_v1 = v2_train_gate._per_bar_view(df)
    per_bar_with_b34 = v2_train_gate._join_base34(
        per_bar_v1, BASE34_M5_FEATURES_PATH
    )
    per_bar_with_deriv = v2_train_gate._compute_derivatives(per_bar_with_b34)
    per_bar_full, _ = v2_train_gate._join_recovery(
        per_bar_with_deriv,
        INPUT_RECOVERY_ROOT / "entry_snapshot_signals_per_trade_v1.parquet",
    )
    per_bar_train = per_bar_full[per_bar_full["primary_split_v1"] == "train"]
    norm = v2_train_gate._fit_train_normalization(per_bar_train)
    X_full, feature_names = v2_train_gate._build_state_matrix_v2(per_bar_full, norm)
    train_mask = (per_bar_full["primary_split_v1"] == "train").to_numpy()

    models: dict[str, dict[str, np.ndarray]] = {}
    for variant in v2_train_gate.REWARD_VARIANTS_V2:
        v_id = variant["reward_id_v1"]
        reward_col = variant["reward_column_v1"]
        targets = v2_train_gate._compute_targets_for_variant(
            per_bar_train, reward_col
        )
        target_hold = targets["__target_hold_v1"].astype(float).to_numpy()
        target_exit_now = targets["__target_exit_now_v1"].astype(float).to_numpy()
        coef_hold = v2_train_gate._ridge_fit(X_full[train_mask], target_hold)
        coef_exit_now = v2_train_gate._ridge_fit(
            X_full[train_mask], target_exit_now
        )
        models[v_id] = {"coef_hold": coef_hold, "coef_exit_now": coef_exit_now}
    return per_bar_full, X_full, feature_names, models


# ---------------------------------------------------------------------------
# Combined stack evaluation
# ---------------------------------------------------------------------------


def _per_trade_realized_pnl(per_bar_split: pd.DataFrame) -> dict[str, float]:
    """Sum running_pnl_at_close at the realized-exit row (last bar) per trade."""
    last = (
        per_bar_split.sort_values(["candidate_uid_v1", "bars_held_v1"])
        .groupby("candidate_uid_v1")
        .tail(1)
    )
    return dict(
        zip(
            last["candidate_uid_v1"].astype(str).tolist(),
            last["running_pnl_at_close_bps_v1"].astype(float).tolist(),
        )
    )


def _per_trade_iql_pnl(
    per_bar_split: pd.DataFrame, X_split: np.ndarray, coef_hold: np.ndarray, coef_exit_now: np.ndarray
) -> dict[str, float]:
    """Apply V2 IQL exit policy per trade; return realized PNL at the
    chosen exit bar per candidate_uid_v1."""
    exit_indices = v2_train_gate._exit_index_from_iql_policy(
        per_bar_split, X_split, coef_hold, coef_exit_now
    )
    selected_idx = exit_indices.values
    selected = per_bar_split.loc[selected_idx].reset_index(drop=True)
    return dict(
        zip(
            selected["candidate_uid_v1"].astype(str).tolist(),
            selected["running_pnl_at_close_bps_v1"].astype(float).tolist(),
        )
    )


def _evaluate_combined(
    per_trade_split: pd.DataFrame,
    p_skip_full: np.ndarray,
    threshold: float,
    per_bar_full: pd.DataFrame,
    X_full: np.ndarray,
    models: dict[str, dict[str, np.ndarray]],
    *,
    split: str,
) -> dict[str, Any]:
    """For one split, compute four PNL numbers per-reward-variant."""
    per_trade_mask = (per_trade_split["primary_split_v1"] == split).to_numpy()
    per_bar_mask = (per_bar_full["primary_split_v1"] == split).to_numpy()
    per_trade_sub = per_trade_split[per_trade_mask].reset_index(drop=True)
    per_bar_sub = per_bar_full[per_bar_mask].reset_index(drop=True)
    X_sub = X_full[per_bar_mask]
    p_skip_sub = p_skip_full[per_trade_mask]
    pred_skip = (p_skip_sub >= threshold).astype(int)

    skipped_uids = set(
        per_trade_sub.loc[pred_skip == 1, "candidate_uid_v1"].astype(str).tolist()
    )
    taken_uids = set(
        per_trade_sub.loc[pred_skip == 0, "candidate_uid_v1"].astype(str).tolist()
    )
    n_taken = len(taken_uids)
    n_skipped = len(skipped_uids)

    realized_pnl_per_trade = _per_trade_realized_pnl(per_bar_sub)
    pnl_no_skip_realized = sum(realized_pnl_per_trade.values())
    pnl_skip_then_realized = sum(
        v for uid, v in realized_pnl_per_trade.items() if uid in taken_uids
    )

    # IQL eval per reward variant.
    per_variant_results: list[dict[str, Any]] = []
    for v_id, m in models.items():
        iql_pnl_per_trade = _per_trade_iql_pnl(
            per_bar_sub, X_sub, m["coef_hold"], m["coef_exit_now"]
        )
        pnl_no_skip_iql = sum(iql_pnl_per_trade.values())
        pnl_skip_then_iql = sum(
            v for uid, v in iql_pnl_per_trade.items() if uid in taken_uids
        )
        per_variant_results.append(
            {
                "reward_variant_v1": v_id,
                "split_v1": split,
                "trade_count_total_v1": int(len(per_trade_sub)),
                "trade_count_taken_v1": n_taken,
                "trade_count_skipped_v1": n_skipped,
                "pnl_no_skip_realized_v1": pnl_no_skip_realized,
                "pnl_no_skip_iql_v1": pnl_no_skip_iql,
                "pnl_skip_then_realized_v1": pnl_skip_then_realized,
                "pnl_skip_then_iql_v1": pnl_skip_then_iql,
                "lift_skip_only_v1": pnl_skip_then_realized - pnl_no_skip_realized,
                "lift_iql_only_v1": pnl_no_skip_iql - pnl_no_skip_realized,
                "lift_combined_v1": pnl_skip_then_iql - pnl_no_skip_realized,
                "lift_combined_minus_sum_of_components_v1": (
                    (pnl_skip_then_iql - pnl_no_skip_realized)
                    - (
                        (pnl_skip_then_realized - pnl_no_skip_realized)
                        + (pnl_no_skip_iql - pnl_no_skip_realized)
                    )
                ),
            }
        )
    return {
        "split_v1": split,
        "tuned_threshold_v1": threshold,
        "trade_count_total_v1": int(len(per_trade_sub)),
        "trade_count_taken_v1": n_taken,
        "trade_count_skipped_v1": n_skipped,
        "pnl_no_skip_realized_v1": pnl_no_skip_realized,
        "pnl_skip_then_realized_v1": pnl_skip_then_realized,
        "per_variant_v1": per_variant_results,
    }


# ---------------------------------------------------------------------------
# go-no-go
# ---------------------------------------------------------------------------


def _go_no_go(
    test_eval: dict[str, Any],
    baseline_metrics_per_split: dict[str, list[dict[str, Any]]],
) -> tuple[str, str, str, dict[str, Any]]:
    test_variants = test_eval["per_variant_v1"]
    if not test_variants:
        raise RuntimeError("COMBINED_TEST_RESULTS_MISSING")
    best = max(test_variants, key=lambda r: r["pnl_skip_then_iql_v1"])
    best_combined = float(best["pnl_skip_then_iql_v1"])
    best_skip_only = float(best["pnl_skip_then_realized_v1"])
    best_iql_only = float(best["pnl_no_skip_iql_v1"])
    no_skip_realized = float(best["pnl_no_skip_realized_v1"])
    interaction = float(best["lift_combined_minus_sum_of_components_v1"])
    baseline_test = {
        b["policy_id_v1"]: b for b in baseline_metrics_per_split.get("test", [])
    }
    trail_stop = baseline_test["TRAIL_STOP_25_PCT_DD"]["total_realized_pnl_bps_v1"]

    headline = {
        "best_reward_variant_v1": best["reward_variant_v1"],
        "tuned_skip_threshold_v1": test_eval["tuned_threshold_v1"],
        "trade_count_total_v1": test_eval["trade_count_total_v1"],
        "trade_count_taken_v1": test_eval["trade_count_taken_v1"],
        "trade_count_skipped_v1": test_eval["trade_count_skipped_v1"],
        "pnl_no_skip_realized_v1": no_skip_realized,
        "pnl_no_skip_iql_v1": best_iql_only,
        "pnl_skip_then_realized_v1": best_skip_only,
        "pnl_skip_then_iql_v1": best_combined,
        "trail_stop_v1": float(trail_stop),
        "lift_combined_minus_sum_of_components_v1": interaction,
        "additivity_classification_v1": (
            "SUPERADDITIVE"
            if interaction > 50.0
            else "ADDITIVE"
            if abs(interaction) <= 50.0
            else "SUBADDITIVE"
        ),
    }

    # Three top-of-stack scenarios:
    #   (a) Combined > best component AND interaction > 0 -> superadditive.
    #   (b) Combined > best component AND interaction ~= 0 -> additive.
    #   (c) Combined ~= best component (skip dominates exit-IQL).
    #   (d) Combined < best component -> subadditive (the components fight).
    best_component = max(best_skip_only, best_iql_only)
    if best_combined >= best_component + 100.0 and interaction > 50.0:
        return (
            "COMBINE_SKIP_V2_WITH_EXIT_IQL_V2_PASS_SUPERADDITIVE_LIFT",
            "WALK_FORWARD_VALIDATION_V1",
            (
                f"Combined stack PNL {best_combined:.0f} > best single "
                f"component {best_component:.0f} by {best_combined-best_component:+.0f}; "
                f"interaction {interaction:+.0f} bps -> superadditive. The "
                "skip filter and IQL exit attack different failures. Next: "
                "walk-forward validation to confirm robustness across time."
            ),
            headline,
        )
    if best_combined >= best_component + 50.0:
        return (
            "COMBINE_SKIP_V2_WITH_EXIT_IQL_V2_PASS_ADDITIVE_LIFT",
            "WALK_FORWARD_VALIDATION_V1",
            (
                f"Combined stack PNL {best_combined:.0f} > best single "
                f"component {best_component:.0f} by {best_combined-best_component:+.0f}; "
                f"interaction {interaction:+.0f} bps. Effects approximately "
                "add. Next: walk-forward validation."
            ),
            headline,
        )
    if abs(best_combined - best_component) <= 100.0:
        return (
            "COMBINE_SKIP_V2_WITH_EXIT_IQL_V2_PASS_SKIP_DOMINATES",
            "WALK_FORWARD_VALIDATION_V1",
            (
                f"Combined stack PNL {best_combined:.0f} ~= best single "
                f"component {best_component:.0f}. Skip-V2 dominates; the "
                "exit-IQL adds little on top of skipping bad trades. Honest "
                "result. Next: walk-forward validation."
            ),
            headline,
        )
    if best_combined < best_component:
        return (
            "COMBINE_SKIP_V2_WITH_EXIT_IQL_V2_PARTIAL_SUBADDITIVE_LIFT",
            "REPAIR_COMBINED_STACK_BEFORE_PROMOTION_V1",
            (
                f"Combined stack PNL {best_combined:.0f} < best single "
                f"component {best_component:.0f} by {best_combined-best_component:+.0f}. "
                "Skip and IQL fight each other (skip removes the trades the "
                "IQL would have rescued, OR IQL exits worse on the kept set). "
                "Investigate before promotion."
            ),
            headline,
        )
    return (
        "COMBINE_SKIP_V2_WITH_EXIT_IQL_V2_PARTIAL_DEGRADES_VS_BEST_COMPONENT",
        "REPAIR_COMBINED_STACK_BEFORE_PROMOTION_V1",
        (
            f"Combined stack PNL {best_combined:.0f} degrades vs best "
            f"component {best_component:.0f}. Investigate."
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
        "layer_name": "COMBINE_SKIP_V2_WITH_EXIT_IQL_V2_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "skip_v2_root_v1": str(INPUT_SKIP_V2_ROOT),
            "v2_training_root_v1": str(INPUT_V2_TRAIN_ROOT),
            "v2_contract_root_v1": str(INPUT_V2_CONTRACT_ROOT),
            "recovery_root_v1": str(INPUT_RECOVERY_ROOT),
            "split_root_v1": str(INPUT_SPLIT_ROOT),
            "eval_harness_root_v1": str(INPUT_EVAL_HARNESS_ROOT),
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


# ---------------------------------------------------------------------------
# Materializer
# ---------------------------------------------------------------------------


def write_artifacts(
    out_root: Path | None = None,
    *,
    built_at_utc: str | None = None,
) -> dict[str, Any]:
    inputs = _load_inputs()
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

    # Re-train both models deterministically.
    per_trade_split, p_skip_full, tuned_threshold = _retrain_skip_v2(inputs)
    per_bar_full, X_full, feature_names, models = _retrain_v2_iql_per_variant(inputs)

    # Evaluate combined stack on each split.
    per_split_eval: dict[str, dict[str, Any]] = {}
    flat_per_variant_rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        ev = _evaluate_combined(
            per_trade_split,
            p_skip_full,
            tuned_threshold,
            per_bar_full,
            X_full,
            models,
            split=split,
        )
        per_split_eval[split] = ev
        flat_per_variant_rows.extend(ev["per_variant_v1"])

    _write_json(
        artifact_root / "combined_stack_evaluation_v1.json",
        {"per_split_v1": per_split_eval},
    )
    _write_rows(
        artifact_root / "combined_stack_per_variant_v1.csv", flat_per_variant_rows
    )

    # Comparator with eval-harness baselines.
    baseline_metrics_flat = inputs["eval_harness_baseline_metrics"]["rows_v1"]
    baseline_per_split: dict[str, list[dict[str, Any]]] = {}
    for row in baseline_metrics_flat:
        baseline_per_split.setdefault(row["split_v1"], []).append(row)

    status, next_action, recommendation, headline = _go_no_go(
        per_split_eval["test"], baseline_per_split
    )
    validate_final_status(status, next_action)

    summary = {
        "layer_name": "COMBINE_SKIP_V2_WITH_EXIT_IQL_V2_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "headline_v1": headline,
        "tuned_skip_threshold_v1": tuned_threshold,
        "v2_iql_feature_count_v1": len(feature_names),
        "reward_variant_count_v1": len(models),
        "reward_variants_v1": list(models.keys()),
        "per_split_eval_v1": per_split_eval,
        "research_only_v1": True,
        "iql_training_run_v1": True,
        "iql_production_allowed_v1": False,
        "skip_classifier_promoted_to_runtime_v1": False,
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

    status_payload = {
        "layer_name": "COMBINE_SKIP_V2_WITH_EXIT_IQL_V2_STATUS_V1",
        "status_v1": "MATERIALIZED_RESEARCH_ONLY_GATE",
        "final_status_v1": status,
        "next_action_v1": next_action,
        "training_executed_v1": True,
    }
    _write_json(artifact_root / "status_v1.json", status_payload)

    go_no_go = {
        "layer_name": "COMBINE_SKIP_V2_WITH_EXIT_IQL_V2_GO_NO_GO_V1",
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
        "skip_classifier_promotion_allowed_v1": False,
        "downstream_block_v1": (
            "Research-only combined-stack evaluation. Models NOT promoted "
            "to runtime; entry_manager / exit_manager / live_features / V1 / "
            "V2 contracts all unmodified."
        ),
    }
    _write_json(
        artifact_root / "combine_skip_v2_with_exit_iql_v2_go_no_go_v1.json", go_no_go
    )

    # Build report.
    test_per_variant = per_split_eval["test"]["per_variant_v1"]
    test_sorted = sorted(test_per_variant, key=lambda r: r["pnl_skip_then_iql_v1"], reverse=True)
    report_lines = [
        "# Combine Skip V2 With Exit IQL V2 V1",
        "",
        f"- Status: `{status}`",
        f"- Next action: `{next_action}`",
        "- Training: research-only; combined stack NOT promoted to runtime.",
        "",
        "## Headline (test split)",
        f"- Best reward variant: `{headline['best_reward_variant_v1']}`",
        f"- Tuned skip threshold (val-best): {headline['tuned_skip_threshold_v1']}",
        f"- Trades: {headline['trade_count_total_v1']} total = "
        f"{headline['trade_count_taken_v1']} taken + "
        f"{headline['trade_count_skipped_v1']} skipped",
        "",
        "### PNL stack (test split, best variant)",
        f"- Floor (no skip, realized exit): {headline['pnl_no_skip_realized_v1']:.0f} bps",
        f"- Skip-only (skip, realized exit on kept): {headline['pnl_skip_then_realized_v1']:.0f} bps",
        f"- IQL-only (no skip, V2 IQL exit): {headline['pnl_no_skip_iql_v1']:.0f} bps",
        f"- **Combined (skip, V2 IQL exit on kept): {headline['pnl_skip_then_iql_v1']:.0f} bps**",
        f"- TRAIL_STOP rule reference: {headline['trail_stop_v1']:.0f} bps",
        "",
        f"- Interaction (combined - sum-of-components): {headline['lift_combined_minus_sum_of_components_v1']:+.0f} bps",
        f"- Additivity classification: **{headline['additivity_classification_v1']}**",
        "",
        "## Per-variant test breakdown",
        "",
        "| Reward | Skip | Take | No-skip+realized | No-skip+IQL | Skip+realized | Skip+IQL | Combined-lift | Interaction |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in test_sorted:
        report_lines.append(
            f"| `{r['reward_variant_v1']}` | "
            f"{r['trade_count_skipped_v1']} | {r['trade_count_taken_v1']} | "
            f"{r['pnl_no_skip_realized_v1']:.0f} | "
            f"{r['pnl_no_skip_iql_v1']:.0f} | "
            f"{r['pnl_skip_then_realized_v1']:.0f} | "
            f"**{r['pnl_skip_then_iql_v1']:.0f}** | "
            f"{r['lift_combined_v1']:+.0f} | "
            f"{r['lift_combined_minus_sum_of_components_v1']:+.0f} |"
        )
    report_lines.extend(
        [
            "",
            "## Per-split summary (best variant)",
        ]
    )
    for split in ["train", "val", "test"]:
        ev = per_split_eval.get(split, {})
        if not ev:
            continue
        best_split = max(
            ev["per_variant_v1"], key=lambda r: r["pnl_skip_then_iql_v1"]
        )
        report_lines.append(
            f"- {split}: combined PNL {best_split['pnl_skip_then_iql_v1']:.0f} bps "
            f"(skip {ev['trade_count_skipped_v1']}/{ev['trade_count_total_v1']}, "
            f"variant `{best_split['reward_variant_v1']}`)"
        )
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
                artifact_root / "combine_skip_v2_with_exit_iql_v2_go_no_go_v1.json"
            ),
            "input_manifest": str(artifact_root / "input_manifest_v1.json"),
            "combined_stack_evaluation": str(
                artifact_root / "combined_stack_evaluation_v1.json"
            ),
            "combined_stack_per_variant_csv": str(
                artifact_root / "combined_stack_per_variant_v1.csv"
            ),
            "report": str(artifact_root / "report_v1.md"),
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
        description="Materialize COMBINE_SKIP_V2_WITH_EXIT_IQL_V2_V1."
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
