#!/usr/bin/env python3
"""Train ridge IQL on V3 state = V2 state + 7 per-bar XGB transformer signals.

Background
----------
V2 training (`RUN_EXIT_IQL_WITH_V2_STATE_AND_REWARD_VARIANTS_V1`) gave
+509 bps test PNL with GIVEBACK_PENALTY_REWARD on V2 state, above the
realized floor (-355) but below the TRAIL_STOP rule (+1052). The V2
parameter sweep showed that hyperparameter tuning alone could not close
the gap; the recommended next step was per-bar XGB replay to fill the
seven NOT_ESTABLISHED TRANSFORMER_SIGNAL_AT_BAR fields.

`RUN_PER_BAR_XGB_REPLAY_FOR_TRANSFORMER_SIGNAL_FAMILY_V1` produced
`per_bar_xgb_signal7_v2.parquet` with 168904 / 169260 rows (99.79%)
replayed (356 NOT_REPLAYED_BASE34_NAN due to weekend-gap edge cases).

This V3 gate trains ridge IQL with V2's 54-feature state EXTENDED by
seven new per-bar XGB transformer signal features (one per signal-bridge
field at the per-bar held-bar level). NOT_REPLAYED rows get a sentinel
value -1.0 for the seven new fields so the model can learn "missing"
explicitly rather than fabricating values.

Five reward variants are trained as in V2 (REALIZED_PNL, MFE_CAPTURE,
MAE_PENALTY, GIVEBACK_PENALTY, TRANSPARENT_COMBINED). Each is evaluated
on train, val, test via the gate-5 harness and compared against V2's
result on the same reward variant (delta_v3_minus_v2 lift). The headline
question is: does adding the seven per-bar XGB signals close the gap
between V2 (+509) and TRAIL_STOP (+1052)?

Research-only; no policy promotion; no runtime modification; no V1/V2
state-contract modification.
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

from gx1.scripts import (
    materialize_build_iql_offline_data_contract_research_only_v1 as contract_gate,
)
from gx1.scripts import materialize_exit_off_policy_eval_harness_v1 as eval_gate
from gx1.scripts import (
    materialize_run_exit_iql_with_v2_state_and_reward_variants_v1 as v2_train_gate,
)


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "RUN_EXIT_IQL_V3_WITH_PER_BAR_XGB_TRANSFORMER_SIGNAL_V1"

INPUT_V2_TRAIN_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "RUN_EXIT_IQL_WITH_V2_STATE_AND_REWARD_VARIANTS_V1_20260429T204407Z_LOCK"
)
INPUT_PER_BAR_XGB_REPLAY_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "RUN_PER_BAR_XGB_REPLAY_FOR_TRANSFORMER_SIGNAL_FAMILY_V1_20260430T060120Z_LOCK"
)
INPUT_V2_CONTRACT_ROOT = v2_train_gate.INPUT_V2_CONTRACT_ROOT
INPUT_RECOVERY_ROOT = v2_train_gate.INPUT_RECOVERY_ROOT
INPUT_SPLIT_ROOT = v2_train_gate.INPUT_SPLIT_ROOT
INPUT_EVAL_HARNESS_ROOT = v2_train_gate.INPUT_EVAL_HARNESS_ROOT
INPUT_V1_TRAINING_ROOT = v2_train_gate.INPUT_V1_TRAINING_ROOT
INPUT_MDP_ROOT = v2_train_gate.INPUT_MDP_ROOT
BASE34_M5_FEATURES_PATH = v2_train_gate.BASE34_M5_FEATURES_PATH

QUARANTINE_FORBIDDEN_PATH_FRAGMENTS = ("gx1/quarantine", "gx1.quarantine")
SEED_V1 = 20260430
RIDGE_LAMBDA = 1e-3

REWARD_VARIANTS_V3 = list(v2_train_gate.REWARD_VARIANTS_V2)

# Per-bar XGB signal-7 fields (added on top of V2 state).
PER_BAR_XGB_FIELDS: list[str] = [
    "per_bar_xgb_p_long_v2",
    "per_bar_xgb_p_short_v2",
    "per_bar_xgb_p_flat_v2",
    "per_bar_xgb_p_hat_v2",
    "per_bar_xgb_uncertainty_score_v2",
    "per_bar_xgb_margin_top1_top2_v2",
    "per_bar_xgb_entropy_v2",
]
PER_BAR_XGB_SENTINEL_VALUE = -1.0

ALLOWED_FINAL_STATUSES = {
    "RUN_EXIT_IQL_V3_PASS_BEST_VARIANT_BEATS_TRAIL_STOP",
    "RUN_EXIT_IQL_V3_PASS_BEST_VARIANT_BEATS_REALIZED_NOT_TRAIL_STOP_LIFTS_V2",
    "RUN_EXIT_IQL_V3_PASS_BEST_VARIANT_BEATS_REALIZED_NOT_TRAIL_STOP_TIES_V2",
    "RUN_EXIT_IQL_V3_PARTIAL_BEST_VARIANT_TIES_REALIZED",
    "RUN_EXIT_IQL_V3_PARTIAL_BEST_VARIANT_UNDERPERFORMS_REALIZED",
    "RUN_EXIT_IQL_V3_BLOCKED_BY_INPUT_LOCK_MISSING",
}

ALLOWED_NEXT_ACTIONS = {
    "COMBINE_SKIP_CLASSIFIER_V2_WITH_EXIT_IQL_V3_V1",
    "EXIT_PER_BAR_PROPER_IQL_WITH_PESSIMISM_V1",
    "REPAIR_EXIT_IQL_TRAINING_BEFORE_VARIANT_SENSITIVITY_V1",
    "HOLD_EXIT_IQL_RESEARCH_UNTIL_DATA_FIXED_V1",
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
        INPUT_V2_TRAIN_ROOT,
        INPUT_PER_BAR_XGB_REPLAY_ROOT,
        INPUT_V2_CONTRACT_ROOT,
        INPUT_RECOVERY_ROOT,
        INPUT_SPLIT_ROOT,
        INPUT_EVAL_HARNESS_ROOT,
        INPUT_V1_TRAINING_ROOT,
        INPUT_MDP_ROOT,
    ]
    validate_explicit_artifact_roots(roots)
    required = {
        "v2_training_summary": INPUT_V2_TRAIN_ROOT / "summary_v1.json",
        "per_bar_xgb_replay_parquet": INPUT_PER_BAR_XGB_REPLAY_ROOT
        / "per_bar_xgb_signal7_v2.parquet",
        "per_bar_xgb_replay_summary": INPUT_PER_BAR_XGB_REPLAY_ROOT / "summary_v1.json",
        "v2_state_contract": INPUT_V2_CONTRACT_ROOT / "state_feature_contract_v2.json",
        "split_locked_dataset": INPUT_SPLIT_ROOT
        / "split_locked_augmented_dataset_v1.parquet",
        "eval_harness_baseline_metrics": INPUT_EVAL_HARNESS_ROOT
        / "baseline_metrics_per_split_v1.json",
        "v1_training_summary": INPUT_V1_TRAINING_ROOT / "summary_v1.json",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise RuntimeError(f"MISSING_REQUIRED_INPUT_LOCKS: {missing}")
    return {
        "required_paths": required,
        "v2_training_summary": _read_json(required["v2_training_summary"]),
        "per_bar_xgb_replay_summary": _read_json(required["per_bar_xgb_replay_summary"]),
        "v2_state_contract": _read_json(required["v2_state_contract"]),
        "eval_harness_baseline_metrics": _read_json(
            required["eval_harness_baseline_metrics"]
        ),
        "v1_training_summary": _read_json(required["v1_training_summary"]),
        "base34_path": BASE34_M5_FEATURES_PATH,
    }


# ---------------------------------------------------------------------------
# Per-bar XGB join + V3 state matrix
# ---------------------------------------------------------------------------


def _join_per_bar_xgb(
    per_bar_full: pd.DataFrame, replay_path: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join the replay parquet on (candidate_uid_v1, bars_held_v1).
    NOT_REPLAYED rows get NaN, sentinel-substituted at state-matrix build."""
    replay = pd.read_parquet(replay_path)
    needed = ["candidate_uid_v1", "bars_held_v1", *PER_BAR_XGB_FIELDS, "replay_status_v1"]
    missing = [c for c in needed if c not in replay.columns]
    if missing:
        raise RuntimeError(f"REPLAY_PARQUET_MISSING_COLUMNS: {missing}")
    replay_use = replay.loc[:, needed].copy()
    replay_use["candidate_uid_v1"] = replay_use["candidate_uid_v1"].astype(str)
    # Deduplicate by (candidate_uid, bars_held) - the parquet should already be
    # unique per per-bar row but we are defensive.
    replay_use = replay_use.drop_duplicates(
        subset=["candidate_uid_v1", "bars_held_v1"], keep="first"
    )
    out = per_bar_full.merge(
        replay_use, on=["candidate_uid_v1", "bars_held_v1"], how="left"
    )
    not_replayed_mask = (
        out["replay_status_v1"].fillna("MISSING")
        != "REPLAYED_FROM_BASE34_M5_AT_BAR_T_MINUS_1"
    )
    not_replayed_count = int(not_replayed_mask.sum())
    for c in PER_BAR_XGB_FIELDS:
        out.loc[not_replayed_mask, c] = np.nan
    audit = {
        "audit_id_v1": "PER_BAR_XGB_JOIN_AUDIT_V1",
        "status_v1": "PASS",
        "row_count_v1": int(len(out)),
        "rows_with_replay_v1": int(len(out) - not_replayed_count),
        "rows_without_replay_v1": not_replayed_count,
        "policy_v1": (
            "Rows without REPLAYED_FROM_BASE34_M5_AT_BAR_T_MINUS_1 status get "
            "NaN for the seven per_bar_xgb_*_v2 fields; sentinel "
            f"({PER_BAR_XGB_SENTINEL_VALUE}) substitution applied at "
            "state-matrix build time. No fabrication."
        ),
    }
    return out, audit


def _per_bar_xgb_passthrough_or_sentinel(values: pd.Series) -> np.ndarray:
    """Per-bar XGB fields are in [0, 1] for probs and small bounded ranges
    for entropy/margin. NaN -> sentinel -1.0 to flag missing."""
    s = values.astype(float)
    return np.where(
        s.notna(),
        s.clip(-2.0, 5.0).to_numpy(),  # generous clip; entropy can be > 1
        PER_BAR_XGB_SENTINEL_VALUE,
    ).astype(float)


def _build_state_matrix_v3(
    per_bar: pd.DataFrame, norm: dict[str, Any]
) -> tuple[np.ndarray, list[str]]:
    """V2 state matrix + 7 per-bar XGB transformer signal columns."""
    X_v2, names_v2 = v2_train_gate._build_state_matrix_v2(per_bar, norm)
    extra_blocks: list[np.ndarray] = []
    extra_names: list[str] = []
    for col in PER_BAR_XGB_FIELDS:
        extra_names.append(f"{col}__pass_or_sentinel")
        extra_blocks.append(
            _per_bar_xgb_passthrough_or_sentinel(per_bar[col]).reshape(-1, 1)
        )
    X = np.concatenate([X_v2, *extra_blocks], axis=1)
    if not np.isfinite(X).all():
        bad = int((~np.isfinite(X)).sum())
        raise RuntimeError(f"V3_STATE_MATRIX_NON_FINITE: count={bad}")
    return X, names_v2 + extra_names


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------


def audit_no_shortcut_at_training_time_v3(
    feature_names: Sequence[str], raw_columns_used: set[str]
) -> dict[str, Any]:
    base = v2_train_gate.audit_no_shortcut_at_training_time(
        feature_names, raw_columns_used
    )
    base["audit_id_v1"] = "TRAINING_NO_SHORTCUT_AUDIT_V3"
    return base


# ---------------------------------------------------------------------------
# go-no-go
# ---------------------------------------------------------------------------


def _go_no_go(
    iql_results: list[dict[str, Any]],
    baseline_metrics_per_split: dict[str, list[dict[str, Any]]],
    v2_baseline_test_pnl: float | None,
    v1_test_total_pnl: float | None,
) -> tuple[str, str, str, dict[str, Any]]:
    test_results = [r for r in iql_results if r["split_v1"] == "test"]
    if not test_results:
        raise RuntimeError("V3_TEST_RESULTS_MISSING")
    best = max(test_results, key=lambda r: r["total_realized_pnl_bps_v1"])
    baseline_test = {
        b["policy_id_v1"]: b for b in baseline_metrics_per_split.get("test", [])
    }
    realized = baseline_test["REALIZED_EXIT_BASELINE"]["total_realized_pnl_bps_v1"]
    trail_stop = baseline_test["TRAIL_STOP_25_PCT_DD"]["total_realized_pnl_bps_v1"]
    best_total = best["total_realized_pnl_bps_v1"]
    delta_v2 = (
        best_total - v2_baseline_test_pnl if v2_baseline_test_pnl is not None else None
    )
    delta_v1 = (
        best_total - v1_test_total_pnl if v1_test_total_pnl is not None else None
    )
    headline = {
        "best_variant_v1": best["reward_variant_v1"],
        "best_test_pnl_v1": float(best_total),
        "realized_v1": float(realized),
        "trail_stop_v1": float(trail_stop),
        "v2_baseline_test_pnl_v1": v2_baseline_test_pnl,
        "delta_v3_minus_v2_v1": delta_v2,
        "v1_iql_test_pnl_v1": v1_test_total_pnl,
        "delta_v3_minus_v1_v1": delta_v1,
        "best_test_mean_bars_to_exit_v1": float(best["mean_bars_to_exit_v1"]),
    }

    if best_total >= trail_stop:
        return (
            "RUN_EXIT_IQL_V3_PASS_BEST_VARIANT_BEATS_TRAIL_STOP",
            "COMBINE_SKIP_CLASSIFIER_V2_WITH_EXIT_IQL_V3_V1",
            (
                f"Best V3 variant `{best['reward_variant_v1']}` test PNL "
                f"{best_total:.0f} >= TRAIL_STOP {trail_stop:.0f}. Per-bar XGB "
                f"transformer signals closed the gap. Delta vs V2: "
                f"{delta_v2:+.0f} bps. Next: combine V2 skip classifier with "
                "V3 exit IQL."
            ),
            headline,
        )
    if best_total > realized:
        if delta_v2 is not None and delta_v2 > 50.0:
            status = "RUN_EXIT_IQL_V3_PASS_BEST_VARIANT_BEATS_REALIZED_NOT_TRAIL_STOP_LIFTS_V2"
            recommendation = (
                f"Best V3 variant `{best['reward_variant_v1']}` test PNL "
                f"{best_total:.0f} lifts V2 by {delta_v2:+.0f} bps but still "
                f"below TRAIL_STOP {trail_stop:.0f}. Per-bar XGB helps but "
                "doesn't fully close the gap. Next: combine with V2 skip."
            )
        else:
            status = "RUN_EXIT_IQL_V3_PASS_BEST_VARIANT_BEATS_REALIZED_NOT_TRAIL_STOP_TIES_V2"
            delta_str = f"{delta_v2:+.0f}" if delta_v2 is not None else "None"
            recommendation = (
                f"Best V3 variant `{best['reward_variant_v1']}` test PNL "
                f"{best_total:.0f} > REALIZED {realized:.0f} but ~ V2 "
                f"({v2_baseline_test_pnl}; delta {delta_str}). "
                "Per-bar XGB did not produce expected lift. Next: combine "
                "with V2 skip or try proper IQL."
            )
        return (
            status,
            "COMBINE_SKIP_CLASSIFIER_V2_WITH_EXIT_IQL_V3_V1",
            recommendation,
            headline,
        )
    if abs(best_total - realized) <= 50.0:
        return (
            "RUN_EXIT_IQL_V3_PARTIAL_BEST_VARIANT_TIES_REALIZED",
            "EXIT_PER_BAR_PROPER_IQL_WITH_PESSIMISM_V1",
            (
                f"Best V3 variant `{best['reward_variant_v1']}` test PNL "
                f"{best_total:.0f} ~= REALIZED {realized:.0f}. Per-bar XGB "
                "did not lift V3 above the realized floor. Next: proper IQL "
                "with pessimism."
            ),
            headline,
        )
    return (
        "RUN_EXIT_IQL_V3_PARTIAL_BEST_VARIANT_UNDERPERFORMS_REALIZED",
        "REPAIR_EXIT_IQL_TRAINING_BEFORE_VARIANT_SENSITIVITY_V1",
        (
            f"Best V3 variant `{best['reward_variant_v1']}` test PNL "
            f"{best_total:.0f} < REALIZED {realized:.0f}. V3 ridge IQL "
            "actively underperforms. Investigate before any further escalation."
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
        "layer_name": "RUN_EXIT_IQL_V3_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "v2_training_root_v1": str(INPUT_V2_TRAIN_ROOT),
            "per_bar_xgb_replay_root_v1": str(INPUT_PER_BAR_XGB_REPLAY_ROOT),
            "v2_contract_root_v1": str(INPUT_V2_CONTRACT_ROOT),
            "recovery_root_v1": str(INPUT_RECOVERY_ROOT),
            "split_root_v1": str(INPUT_SPLIT_ROOT),
            "eval_harness_root_v1": str(INPUT_EVAL_HARNESS_ROOT),
            "v1_training_root_v1": str(INPUT_V1_TRAINING_ROOT),
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

    # 1. Project per-bar with V2 pipeline
    df = pd.read_parquet(inputs["required_paths"]["split_locked_dataset"])
    df["candidate_uid_v1"] = df["candidate_uid_v1"].astype(str)
    df["ts_v1"] = pd.to_datetime(df["ts_v1"], utc=True)
    per_bar_v1 = v2_train_gate._per_bar_view(df)
    per_bar_with_b34 = v2_train_gate._join_base34(
        per_bar_v1, BASE34_M5_FEATURES_PATH
    )
    per_bar_with_deriv = v2_train_gate._compute_derivatives(per_bar_with_b34)
    per_bar_full, recovery_audit = v2_train_gate._join_recovery(
        per_bar_with_deriv,
        INPUT_RECOVERY_ROOT / "entry_snapshot_signals_per_trade_v1.parquet",
    )

    # 2. Add per-bar XGB replay
    per_bar_full, per_bar_xgb_audit = _join_per_bar_xgb(
        per_bar_full, INPUT_PER_BAR_XGB_REPLAY_ROOT / "per_bar_xgb_signal7_v2.parquet"
    )

    split_isolation = v2_train_gate.audit_split_isolation(per_bar_full)
    per_bar_train = per_bar_full[per_bar_full["primary_split_v1"] == "train"]
    norm = v2_train_gate._fit_train_normalization(per_bar_train)
    train_only_audit = v2_train_gate.audit_train_only_normalization(per_bar_full, norm)
    _write_json(artifact_root / "training_normalization_v1.json", norm)

    # 3. Build V3 state matrix
    X_full, feature_names = _build_state_matrix_v3(per_bar_full, norm)
    train_mask = (per_bar_full["primary_split_v1"] == "train").to_numpy()
    X_train_only = X_full[train_mask]

    raw_cols_used: set[str] = set(
        v2_train_gate.V1_CONTINUOUS_FROM_AUGMENTED
        + v2_train_gate.V1_LOG1P_FROM_AUGMENTED
        + v2_train_gate.V1_PASSTHROUGH_FROM_AUGMENTED
        + list(v2_train_gate.V1_ONEHOT_FROM_AUGMENTED.keys())
        + v2_train_gate.NEW_BASE34_CONTINUOUS
        + v2_train_gate.NEW_BASE34_BINARY
        + v2_train_gate.DERIVED_CONTINUOUS
        + v2_train_gate.RECOVERED_PASSTHROUGH
        + PER_BAR_XGB_FIELDS
    )
    no_shortcut = audit_no_shortcut_at_training_time_v3(feature_names, raw_cols_used)

    # 4. Train per reward variant
    iql_results: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    safety_audits: list[dict[str, Any]] = []
    for variant in REWARD_VARIANTS_V3:
        variant_id = variant["reward_id_v1"]
        reward_col = variant["reward_column_v1"]
        targets = v2_train_gate._compute_targets_for_variant(per_bar_train, reward_col)
        target_hold = targets["__target_hold_v1"].astype(float).to_numpy()
        target_exit_now = targets["__target_exit_now_v1"].astype(float).to_numpy()

        coef_hold = v2_train_gate._ridge_fit(X_train_only, target_hold, lam=RIDGE_LAMBDA)
        coef_exit_now = v2_train_gate._ridge_fit(
            X_train_only, target_exit_now, lam=RIDGE_LAMBDA
        )

        models.append(
            {
                "reward_id_v1": variant_id,
                "feature_count_v1": len(feature_names),
                "coef_hold_l2_norm_v1": float(np.linalg.norm(coef_hold)),
                "coef_exit_now_l2_norm_v1": float(np.linalg.norm(coef_exit_now)),
                "coef_hold_v1": coef_hold.tolist(),
                "coef_exit_now_v1": coef_exit_now.tolist(),
                "feature_names_v1": list(feature_names),
                "ridge_lambda_v1": RIDGE_LAMBDA,
                "seed_v1": SEED_V1,
            }
        )

        for split in ["train", "val", "test"]:
            mask = (per_bar_full["primary_split_v1"] == split).to_numpy()
            per_bar_split = per_bar_full[mask].reset_index(drop=True)
            if per_bar_split.empty:
                continue
            X_split = X_full[mask]
            exit_indices = v2_train_gate._exit_index_from_iql_policy(
                per_bar_split, X_split, coef_hold, coef_exit_now
            )
            safety_audits.append(
                v2_train_gate.audit_policy_safety_at_inference(
                    per_bar_split,
                    exit_indices,
                    variant_id=f"V3_{variant_id}_{split}",
                )
            )
            metrics = eval_gate.evaluate_policy(
                per_bar_split,
                exit_indices,
                policy_id=f"IQL_V3_RIDGE_2HEAD_{variant_id}",
                split=split,
            )
            metrics["model_id_v1"] = "EXIT_IQL_V3_RIDGE_2HEAD"
            metrics["reward_variant_v1"] = variant_id
            iql_results.append(metrics)

    _write_json(
        artifact_root / "trained_models_per_variant_v1.json",
        {"variant_count_v1": len(models), "models_v1": models},
    )

    # 5. Comparator: V3 + V2 baseline + V1 reference + 6 baselines
    baseline_metrics_flat = inputs["eval_harness_baseline_metrics"]["rows_v1"]
    baseline_per_split: dict[str, list[dict[str, Any]]] = {}
    for row in baseline_metrics_flat:
        baseline_per_split.setdefault(row["split_v1"], []).append(row)

    v2_results = inputs["v2_training_summary"].get("iql_results_v1", []) or []
    v2_baseline_test = None
    v2_test_per_variant: dict[str, float] = {}
    for r in v2_results:
        if r["split_v1"] == "test":
            v2_test_per_variant[r["reward_variant_v1"]] = float(
                r["total_realized_pnl_bps_v1"]
            )
            if r["reward_variant_v1"] == "GIVEBACK_PENALTY_REWARD":
                v2_baseline_test = float(r["total_realized_pnl_bps_v1"])

    v1_summary = inputs["v1_training_summary"]
    v1_test = v1_summary.get("iql_test_v1") or {}
    v1_test_total = (
        float(v1_test["total_realized_pnl_bps_v1"]) if v1_test else None
    )

    # Per-variant V3 vs V2 delta table.
    v3_vs_v2_table: list[dict[str, Any]] = []
    for r in iql_results:
        if r["split_v1"] != "test":
            continue
        v_id = r["reward_variant_v1"]
        v3_pnl = float(r["total_realized_pnl_bps_v1"])
        v2_pnl = v2_test_per_variant.get(v_id)
        v3_vs_v2_table.append(
            {
                "reward_variant_v1": v_id,
                "v3_test_pnl_bps_v1": v3_pnl,
                "v2_test_pnl_bps_v1": v2_pnl,
                "delta_v3_minus_v2_v1": (
                    v3_pnl - v2_pnl if v2_pnl is not None else None
                ),
                "v3_test_mean_bars_v1": float(r["mean_bars_to_exit_v1"]),
            }
        )
    _write_rows(artifact_root / "v3_vs_v2_per_variant_v1.csv", v3_vs_v2_table)
    _write_json(
        artifact_root / "v3_vs_v2_per_variant_v1.json",
        {"row_count_v1": len(v3_vs_v2_table), "rows_v1": v3_vs_v2_table},
    )

    comparator_rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        for r in baseline_per_split.get(split, []):
            comparator_rows.append({**r, "row_kind_v1": "BASELINE"})
        for r in iql_results:
            if r["split_v1"] == split:
                comparator_rows.append(
                    {
                        **r,
                        "implementable_v1": True,
                        "uses_oracle_v1": False,
                        "row_kind_v1": "IQL_V3",
                    }
                )
    if v2_baseline_test is not None:
        comparator_rows.append(
            {
                "split_v1": "test",
                "policy_id_v1": "IQL_V2_BASELINE_GIVEBACK_PENALTY_REFERENCE",
                "row_kind_v1": "IQL_V2_BASELINE_REFERENCE",
                "total_realized_pnl_bps_v1": v2_baseline_test,
            }
        )
    if v1_test:
        comparator_rows.append(
            {
                **v1_test,
                "row_kind_v1": "IQL_V1_REFERENCE",
                "policy_id_v1": "IQL_V1_RIDGE_REALIZED_PNL_REFERENCE",
                "implementable_v1": True,
                "uses_oracle_v1": False,
                "split_v1": "test",
            }
        )
    _write_rows(
        artifact_root / "iql_v3_vs_baseline_comparator_v1.csv", comparator_rows
    )
    _write_json(
        artifact_root / "iql_v3_vs_baseline_comparator_v1.json",
        {"row_count_v1": len(comparator_rows), "rows_v1": comparator_rows},
    )

    audits = [
        split_isolation,
        no_shortcut,
        train_only_audit,
        recovery_audit,
        per_bar_xgb_audit,
    ]
    _write_json(
        artifact_root / "training_audits_v1.json",
        {"audit_count_v1": len(audits), "audits_v1": audits},
    )
    _write_json(
        artifact_root / "policy_safety_audits_v1.json",
        {"audit_count_v1": len(safety_audits), "audits_v1": safety_audits},
    )
    repro = {
        "layer_name": "RUN_EXIT_IQL_V3_REPRODUCIBILITY_AUDIT_V1",
        "model_v1": "CLOSED_FORM_RIDGE_TWO_HEADS_PER_REWARD_VARIANT",
        "feature_count_v1": len(feature_names),
        "ridge_lambda_v1": RIDGE_LAMBDA,
        "seed_v1": SEED_V1,
        "reward_variant_count_v1": len(models),
        "splits_evaluated_v1": sorted({r["split_v1"] for r in iql_results}),
        "no_implicit_glob_used_for_v1_inputs_v1": True,
        "deprecated_quarantine_revival_v1": False,
        "research_only_v1": True,
    }
    _write_json(artifact_root / "reproducibility_audit_v1.json", repro)

    status, next_action, recommendation, headline = _go_no_go(
        iql_results, baseline_per_split, v2_baseline_test, v1_test_total
    )
    validate_final_status(status, next_action)

    summary = {
        "layer_name": "RUN_EXIT_IQL_V3_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "headline_v1": headline,
        "model_id_v1": "EXIT_IQL_V3_RIDGE_2HEAD",
        "reward_variant_count_v1": len(REWARD_VARIANTS_V3),
        "reward_variants_v1": [v["reward_id_v1"] for v in REWARD_VARIANTS_V3],
        "ridge_lambda_v1": RIDGE_LAMBDA,
        "feature_count_v1": len(feature_names),
        "v3_vs_v2_per_variant_v1": v3_vs_v2_table,
        "iql_results_v1": iql_results,
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
        "entry_manager_modified_v1": False,
        "forbidden_actions_audit_v1": forbidden_audit,
    }
    _write_json(artifact_root / "summary_v1.json", summary)

    status_payload = {
        "layer_name": "RUN_EXIT_IQL_V3_STATUS_V1",
        "status_v1": "MATERIALIZED_RESEARCH_ONLY_TRAINING_GATE",
        "final_status_v1": status,
        "next_action_v1": next_action,
        "training_executed_v1": True,
    }
    _write_json(artifact_root / "status_v1.json", status_payload)

    go_no_go = {
        "layer_name": "RUN_EXIT_IQL_V3_GO_NO_GO_V1",
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
            "Research-only V3 training. Policies NOT promoted to runtime; "
            "exit_manager / live_features / entry_manager / V1 / V2 contracts "
            "all unmodified."
        ),
    }
    _write_json(artifact_root / "run_exit_iql_v3_go_no_go_v1.json", go_no_go)

    test_rows_sorted = sorted(
        [r for r in iql_results if r["split_v1"] == "test"],
        key=lambda r: r["total_realized_pnl_bps_v1"],
        reverse=True,
    )
    report_lines = [
        "# Run Exit IQL V3 With Per-Bar XGB Transformer Signal V1",
        "",
        f"- Status: `{status}`",
        f"- Next action: `{next_action}`",
        "- Training: research-only; policies NOT promoted to runtime.",
        "",
        "## Headline",
        f"- Best V3 variant: `{headline['best_variant_v1']}`",
        f"- Best test PNL: {headline['best_test_pnl_v1']:.0f} bps",
        f"- REALIZED floor: {headline['realized_v1']:.0f} bps",
        f"- TRAIL_STOP rule: {headline['trail_stop_v1']:.0f} bps",
        f"- V2 baseline (GIVEBACK L1E3 FULL): {headline['v2_baseline_test_pnl_v1']} bps",
        f"- Delta V3 - V2: {headline['delta_v3_minus_v2_v1']}",
        f"- V1 IQL reference: {headline['v1_iql_test_pnl_v1']} bps",
        f"- Delta V3 - V1: {headline['delta_v3_minus_v1_v1']}",
        f"- Best mean bars to exit: {headline['best_test_mean_bars_to_exit_v1']:.1f}",
        "",
        "## V3 IQL test results sorted by total PNL (descending)",
        "",
        "| Reward variant | Trades | Sum PNL | Mean PNL | MFE-cap | MAE-burden | Giveback | CATA% | Bars |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in test_rows_sorted:
        report_lines.append(
            f"| `{r['reward_variant_v1']}` | {r['trade_count_v1']} | "
            f"{r['total_realized_pnl_bps_v1']:.0f} | "
            f"{r['mean_realized_pnl_bps_v1']:.2f} | "
            f"{r['mean_mfe_capture_ratio_v1']:.3f} | "
            f"{r['mean_mae_burden_bps_v1']:.1f} | "
            f"{r['mean_giveback_bps_v1']:.1f} | "
            f"{r['cata_proxy_rate_v1']*100:.1f}% | "
            f"{r['mean_bars_to_exit_v1']:.1f} |"
        )
    report_lines.extend(
        [
            "",
            "## V3 vs V2 per-variant delta (test split)",
            "",
            "| Reward variant | V2 PNL | V3 PNL | Delta V3-V2 | V3 mean bars |",
            "|---|---|---|---|---|",
        ]
    )
    for r in v3_vs_v2_table:
        delta = r["delta_v3_minus_v2_v1"]
        report_lines.append(
            f"| `{r['reward_variant_v1']}` | "
            f"{r['v2_test_pnl_bps_v1']:.0f} | "
            f"{r['v3_test_pnl_bps_v1']:.0f} | "
            f"{delta:+.0f} | "
            f"{r['v3_test_mean_bars_v1']:.1f} |"
        )
    report_lines.extend(["", "## Audits"])
    for a in audits:
        report_lines.append(f"- `{a['audit_id_v1']}`: {a['status_v1']}")
    safety_pass = all(a["status_v1"] == "PASS" for a in safety_audits)
    report_lines.append(
        f"- Per-(variant, split) policy safety audits: {len(safety_audits)} "
        f"({'all PASS' if safety_pass else 'FAILURES'})"
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
            "go_no_go": str(artifact_root / "run_exit_iql_v3_go_no_go_v1.json"),
            "input_manifest": str(artifact_root / "input_manifest_v1.json"),
            "trained_models_per_variant": str(
                artifact_root / "trained_models_per_variant_v1.json"
            ),
            "training_normalization": str(artifact_root / "training_normalization_v1.json"),
            "iql_v3_vs_baseline_comparator_csv": str(
                artifact_root / "iql_v3_vs_baseline_comparator_v1.csv"
            ),
            "iql_v3_vs_baseline_comparator_json": str(
                artifact_root / "iql_v3_vs_baseline_comparator_v1.json"
            ),
            "v3_vs_v2_per_variant_csv": str(
                artifact_root / "v3_vs_v2_per_variant_v1.csv"
            ),
            "v3_vs_v2_per_variant_json": str(
                artifact_root / "v3_vs_v2_per_variant_v1.json"
            ),
            "training_audits": str(artifact_root / "training_audits_v1.json"),
            "policy_safety_audits": str(
                artifact_root / "policy_safety_audits_v1.json"
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

    return {
        "artifact_root": str(artifact_root),
        "summary": summary,
        "status": status_payload,
        "go_no_go": go_no_go,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize RUN_EXIT_IQL_V3_WITH_PER_BAR_XGB_TRANSFORMER_SIGNAL_V1."
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
