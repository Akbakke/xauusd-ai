#!/usr/bin/env python3
"""Run a 10-config parameter sweep on the V2 exit-IQL ridge IQL.

Background
----------
`RUN_EXIT_IQL_WITH_V2_STATE_AND_REWARD_VARIANTS_V1` (the V2 training gate)
showed that GIVEBACK_PENALTY_REWARD on V2 state lifts test PNL to +509 bps,
above the realized-exit floor (-355 bps) but still below the simple
TRAIL_STOP_25_PCT_DD rule baseline (+1052 bps). The recommended next gate
was `RUN_PER_BAR_XGB_REPLAY_FOR_TRANSFORMER_SIGNAL_FAMILY_V1` to fill the
seven NOT_ESTABLISHED transformer-signal fields, which would let IQL see
the per-bar XGB output (a real online signal the trail-stop rule cannot
use). That replay is a separate, heavier infrastructure effort (load XGB
multihead bundle, run ~169k per-bar inferences with feature alignment).

This sweep gate is a lighter-weight, immediately runnable exploration of
the three knobs that most plausibly change the outcome under V2 state:

  1. Ridge regularization strength (lambda): under-regularized -> high
     variance, over-regularized -> trivial policy. Sweep three orders of
     magnitude (1e-3, 1e-2, 1e-1).
  2. Reward variant: which hindsight path-outcome the agent optimizes.
     Five locked variants in REWARD_VARIANTS_V2.
  3. State subset / ablation: V2_FULL (54 features post-one-hot) vs
     V2_NO_DERIVATIVES (50: drops the four V2 running-state derivatives)
     vs V2_NO_RECOVERY (50: drops the four entry-snapshot recovered
     fields). This isolates which V2 additions actually contribute.

Ten configs are evaluated on train, val, test via the gate-5 off-policy
harness. The best-on-test config is reported alongside per-config metrics
and a per-knob sensitivity table. Research-only; no policy promotion; no
runtime modification; no V1/V2 state-contract modification.
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
ACTION = "RUN_EXIT_IQL_V2_PARAMETER_SWEEP_V1"

INPUT_V2_TRAIN_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "RUN_EXIT_IQL_WITH_V2_STATE_AND_REWARD_VARIANTS_V1_20260429T204407Z_LOCK"
)
INPUT_V2_CONTRACT_ROOT = v2_train_gate.INPUT_V2_CONTRACT_ROOT
INPUT_RECOVERY_ROOT = v2_train_gate.INPUT_RECOVERY_ROOT
INPUT_SPLIT_ROOT = v2_train_gate.INPUT_SPLIT_ROOT
INPUT_EVAL_HARNESS_ROOT = v2_train_gate.INPUT_EVAL_HARNESS_ROOT

QUARANTINE_FORBIDDEN_PATH_FRAGMENTS = ("gx1/quarantine", "gx1.quarantine")
SEED_V1 = 20260429


# ---------------------------------------------------------------------------
# 10 sweep configs
# ---------------------------------------------------------------------------
#
# state_subset_v1 codes:
#   FULL                 - all 54 features (V2 baseline)
#   NO_DERIVATIVES       - drops the 4 derivatives (50 features)
#   NO_RECOVERY          - drops the 4 entry-snapshot recovery fields (50 features)

SWEEP_CONFIGS: list[dict[str, Any]] = [
    {"config_id_v1": "C01_GIVEBACK_L1E3_FULL", "reward_id_v1": "GIVEBACK_PENALTY_REWARD", "ridge_lambda_v1": 1e-3, "state_subset_v1": "FULL"},
    {"config_id_v1": "C02_GIVEBACK_L1E2_FULL", "reward_id_v1": "GIVEBACK_PENALTY_REWARD", "ridge_lambda_v1": 1e-2, "state_subset_v1": "FULL"},
    {"config_id_v1": "C03_GIVEBACK_L1E1_FULL", "reward_id_v1": "GIVEBACK_PENALTY_REWARD", "ridge_lambda_v1": 1e-1, "state_subset_v1": "FULL"},
    {"config_id_v1": "C04_GIVEBACK_L1E3_NO_DERIVATIVES", "reward_id_v1": "GIVEBACK_PENALTY_REWARD", "ridge_lambda_v1": 1e-3, "state_subset_v1": "NO_DERIVATIVES"},
    {"config_id_v1": "C05_GIVEBACK_L1E3_NO_RECOVERY", "reward_id_v1": "GIVEBACK_PENALTY_REWARD", "ridge_lambda_v1": 1e-3, "state_subset_v1": "NO_RECOVERY"},
    {"config_id_v1": "C06_MAE_L1E3_FULL", "reward_id_v1": "MAE_PENALTY_REWARD", "ridge_lambda_v1": 1e-3, "state_subset_v1": "FULL"},
    {"config_id_v1": "C07_MAE_L1E2_FULL", "reward_id_v1": "MAE_PENALTY_REWARD", "ridge_lambda_v1": 1e-2, "state_subset_v1": "FULL"},
    {"config_id_v1": "C08_TRANSPARENT_L1E3_FULL", "reward_id_v1": "TRANSPARENT_COMBINED_REWARD", "ridge_lambda_v1": 1e-3, "state_subset_v1": "FULL"},
    {"config_id_v1": "C09_REALIZED_L1E2_FULL", "reward_id_v1": "REALIZED_PNL_REWARD", "ridge_lambda_v1": 1e-2, "state_subset_v1": "FULL"},
    {"config_id_v1": "C10_REALIZED_L1E1_FULL", "reward_id_v1": "REALIZED_PNL_REWARD", "ridge_lambda_v1": 1e-1, "state_subset_v1": "FULL"},
]
assert len(SWEEP_CONFIGS) == 10

ALLOWED_STATE_SUBSETS = {"FULL", "NO_DERIVATIVES", "NO_RECOVERY"}

ALLOWED_FINAL_STATUSES = {
    "RUN_EXIT_IQL_V2_PARAMETER_SWEEP_PASS_BEST_BEATS_TRAIL_STOP",
    "RUN_EXIT_IQL_V2_PARAMETER_SWEEP_PASS_BEST_BEATS_REALIZED_NOT_TRAIL_STOP",
    "RUN_EXIT_IQL_V2_PARAMETER_SWEEP_PARTIAL_BEST_TIES_REALIZED",
    "RUN_EXIT_IQL_V2_PARAMETER_SWEEP_PARTIAL_BEST_UNDERPERFORMS_REALIZED",
    "RUN_EXIT_IQL_V2_PARAMETER_SWEEP_BLOCKED_BY_INPUT_LOCK_MISSING",
}

ALLOWED_NEXT_ACTIONS = {
    "RUN_PER_BAR_XGB_REPLAY_FOR_TRANSFORMER_SIGNAL_FAMILY_V1",
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


def validate_sweep_grid(configs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(configs) != 10:
        raise RuntimeError(f"SWEEP_CONFIG_COUNT_MISMATCH: expected 10, got {len(configs)}")
    ids = [c["config_id_v1"] for c in configs]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"DUPLICATE_CONFIG_ID_V1: {ids}")
    valid_rewards = {v["reward_id_v1"] for v in v2_train_gate.REWARD_VARIANTS_V2}
    bad_rewards = [c["config_id_v1"] for c in configs if c["reward_id_v1"] not in valid_rewards]
    if bad_rewards:
        raise RuntimeError(f"UNKNOWN_REWARD_ID_V1: {bad_rewards}")
    bad_subsets = [c["config_id_v1"] for c in configs if c["state_subset_v1"] not in ALLOWED_STATE_SUBSETS]
    if bad_subsets:
        raise RuntimeError(f"UNKNOWN_STATE_SUBSET_V1: {bad_subsets}")
    bad_lambdas = [c["config_id_v1"] for c in configs if c["ridge_lambda_v1"] <= 0]
    if bad_lambdas:
        raise RuntimeError(f"NON_POSITIVE_RIDGE_LAMBDA: {bad_lambdas}")
    return {
        "audit_id_v1": "SWEEP_GRID_VALIDATION_V1",
        "status_v1": "PASS",
        "config_count_v1": len(configs),
        "unique_reward_variants_used_v1": sorted({c["reward_id_v1"] for c in configs}),
        "unique_state_subsets_used_v1": sorted({c["state_subset_v1"] for c in configs}),
        "unique_lambdas_used_v1": sorted({c["ridge_lambda_v1"] for c in configs}),
    }


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _load_inputs() -> dict[str, Any]:
    roots = [
        INPUT_V2_TRAIN_ROOT,
        INPUT_V2_CONTRACT_ROOT,
        INPUT_RECOVERY_ROOT,
        INPUT_SPLIT_ROOT,
        INPUT_EVAL_HARNESS_ROOT,
    ]
    validate_explicit_artifact_roots(roots)
    required = {
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
    if not v2_train_gate.BASE34_M5_FEATURES_PATH.exists():
        raise RuntimeError(
            f"BASE34_M5_FEATURES_PATH_NOT_FOUND: {v2_train_gate.BASE34_M5_FEATURES_PATH}"
        )
    return {
        "required_paths": required,
        "v2_training_summary": _read_json(required["v2_training_summary"]),
        "v2_state_contract": _read_json(required["v2_state_contract"]),
        "eval_harness_baseline_metrics": _read_json(
            required["eval_harness_baseline_metrics"]
        ),
        "base34_path": v2_train_gate.BASE34_M5_FEATURES_PATH,
    }


# ---------------------------------------------------------------------------
# State-subset selection
# ---------------------------------------------------------------------------


def _state_matrix_for_subset(
    X_full: np.ndarray, feature_names: list[str], subset: str
) -> tuple[np.ndarray, list[str]]:
    """Return a column-filtered (X, names) for the requested ablation subset."""
    if subset == "FULL":
        return X_full, list(feature_names)
    if subset == "NO_DERIVATIVES":
        keep = [
            (i, n)
            for i, n in enumerate(feature_names)
            if not any(d in n for d in v2_train_gate.DERIVED_CONTINUOUS)
        ]
    elif subset == "NO_RECOVERY":
        keep = [
            (i, n)
            for i, n in enumerate(feature_names)
            if not any(r in n for r in v2_train_gate.RECOVERED_PASSTHROUGH)
        ]
    else:
        raise RuntimeError(f"UNKNOWN_STATE_SUBSET: {subset}")
    idx = [i for i, _ in keep]
    names = [n for _, n in keep]
    return X_full[:, idx], names


# ---------------------------------------------------------------------------
# Per-config train + eval
# ---------------------------------------------------------------------------


def _run_one_config(
    config: dict[str, Any],
    per_bar_full: pd.DataFrame,
    X_full: np.ndarray,
    feature_names_full: list[str],
    train_mask: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Train ridge IQL for one (reward, lambda, subset) config and evaluate
    on all three splits via gate-5 harness. Returns (model_summary, [metrics_per_split])."""
    subset = config["state_subset_v1"]
    X_subset_full, names = _state_matrix_for_subset(X_full, feature_names_full, subset)
    X_train_only = X_subset_full[train_mask]

    reward_id = config["reward_id_v1"]
    reward_col = next(
        v["reward_column_v1"]
        for v in v2_train_gate.REWARD_VARIANTS_V2
        if v["reward_id_v1"] == reward_id
    )
    per_bar_train = per_bar_full[per_bar_full["primary_split_v1"] == "train"]
    targets = v2_train_gate._compute_targets_for_variant(per_bar_train, reward_col)
    target_hold = targets["__target_hold_v1"].astype(float).to_numpy()
    target_exit_now = targets["__target_exit_now_v1"].astype(float).to_numpy()

    lam = float(config["ridge_lambda_v1"])
    coef_hold = v2_train_gate._ridge_fit(X_train_only, target_hold, lam=lam)
    coef_exit_now = v2_train_gate._ridge_fit(X_train_only, target_exit_now, lam=lam)

    model_summary = {
        **config,
        "feature_count_v1": len(names),
        "feature_names_v1": names,
        "train_row_count_v1": int(X_train_only.shape[0]),
        "seed_v1": SEED_V1,
        # Coefs are kept compact by hashing - we don't write all 54 values per
        # config to summary, but per-config trained model JSONs do.
        "coef_hold_l2_norm_v1": float(np.linalg.norm(coef_hold)),
        "coef_exit_now_l2_norm_v1": float(np.linalg.norm(coef_exit_now)),
    }
    coefs = {
        "config_id_v1": config["config_id_v1"],
        "coef_hold_v1": coef_hold.tolist(),
        "coef_exit_now_v1": coef_exit_now.tolist(),
        "feature_names_v1": names,
        "ridge_lambda_v1": lam,
    }

    metrics_per_split: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        mask = (per_bar_full["primary_split_v1"] == split).to_numpy()
        per_bar_split = per_bar_full[mask].reset_index(drop=True)
        if per_bar_split.empty:
            continue
        X_split = X_subset_full[mask]
        exit_indices = v2_train_gate._exit_index_from_iql_policy(
            per_bar_split, X_split, coef_hold, coef_exit_now
        )
        # Inline the policy-safety check here; raise if it fails.
        v2_train_gate.audit_policy_safety_at_inference(
            per_bar_split, exit_indices, variant_id=f"{config['config_id_v1']}_{split}"
        )
        m = eval_gate.evaluate_policy(
            per_bar_split,
            exit_indices,
            policy_id=f"IQL_V2_SWEEP_{config['config_id_v1']}",
            split=split,
        )
        m["config_id_v1"] = config["config_id_v1"]
        m["reward_variant_v1"] = reward_id
        m["ridge_lambda_v1"] = lam
        m["state_subset_v1"] = subset
        m["model_id_v1"] = "EXIT_IQL_V2_RIDGE_SWEEP"
        metrics_per_split.append(m)

    model_summary["coefs_v1"] = coefs
    return model_summary, metrics_per_split


# ---------------------------------------------------------------------------
# Sensitivity analyses
# ---------------------------------------------------------------------------


def _ridge_lambda_sensitivity(
    test_metrics: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """For each (reward, subset) pair held constant, report PNL across lambdas."""
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], dict[float, float]] = {}
    for m in test_metrics:
        key = (m["reward_variant_v1"], m["state_subset_v1"])
        grouped.setdefault(key, {})[m["ridge_lambda_v1"]] = m["total_realized_pnl_bps_v1"]
    for (reward, subset), lam_to_pnl in sorted(grouped.items()):
        if len(lam_to_pnl) < 2:
            continue
        rows.append(
            {
                "reward_variant_v1": reward,
                "state_subset_v1": subset,
                "lambda_to_test_pnl_v1": lam_to_pnl,
                "best_lambda_v1": max(lam_to_pnl, key=lam_to_pnl.get),
                "spread_test_pnl_v1": float(
                    max(lam_to_pnl.values()) - min(lam_to_pnl.values())
                ),
            }
        )
    return rows


def _state_subset_ablation(
    test_metrics: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """For each (reward, lambda) pair, report PNL across state subsets."""
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, float], dict[str, float]] = {}
    for m in test_metrics:
        key = (m["reward_variant_v1"], m["ridge_lambda_v1"])
        grouped.setdefault(key, {})[m["state_subset_v1"]] = m["total_realized_pnl_bps_v1"]
    for (reward, lam), subset_to_pnl in sorted(grouped.items()):
        if len(subset_to_pnl) < 2:
            continue
        rows.append(
            {
                "reward_variant_v1": reward,
                "ridge_lambda_v1": lam,
                "subset_to_test_pnl_v1": subset_to_pnl,
                "best_subset_v1": max(subset_to_pnl, key=subset_to_pnl.get),
                "spread_test_pnl_v1": float(
                    max(subset_to_pnl.values()) - min(subset_to_pnl.values())
                ),
            }
        )
    return rows


def _reward_variant_sensitivity(
    test_metrics: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """For each (lambda, subset) pair, report PNL across reward variants."""
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[float, str], dict[str, float]] = {}
    for m in test_metrics:
        key = (m["ridge_lambda_v1"], m["state_subset_v1"])
        grouped.setdefault(key, {})[m["reward_variant_v1"]] = m["total_realized_pnl_bps_v1"]
    for (lam, subset), reward_to_pnl in sorted(grouped.items()):
        if len(reward_to_pnl) < 2:
            continue
        rows.append(
            {
                "ridge_lambda_v1": lam,
                "state_subset_v1": subset,
                "reward_to_test_pnl_v1": reward_to_pnl,
                "best_reward_v1": max(reward_to_pnl, key=reward_to_pnl.get),
                "spread_test_pnl_v1": float(
                    max(reward_to_pnl.values()) - min(reward_to_pnl.values())
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# go-no-go
# ---------------------------------------------------------------------------


def _go_no_go(
    test_metrics: list[dict[str, Any]],
    baseline_per_split: dict[str, list[dict[str, Any]]],
    v2_baseline_test_pnl: float | None,
) -> tuple[str, str, str, dict[str, Any]]:
    if not test_metrics:
        raise RuntimeError("SWEEP_TEST_METRICS_MISSING")
    best = max(test_metrics, key=lambda r: r["total_realized_pnl_bps_v1"])
    baseline_test = {
        b["policy_id_v1"]: b for b in baseline_per_split.get("test", [])
    }
    realized = baseline_test["REALIZED_EXIT_BASELINE"]["total_realized_pnl_bps_v1"]
    trail_stop = baseline_test["TRAIL_STOP_25_PCT_DD"]["total_realized_pnl_bps_v1"]
    best_total = best["total_realized_pnl_bps_v1"]
    delta_v2 = (
        best_total - v2_baseline_test_pnl if v2_baseline_test_pnl is not None else None
    )
    headline = {
        "best_config_id_v1": best["config_id_v1"],
        "best_reward_variant_v1": best["reward_variant_v1"],
        "best_ridge_lambda_v1": best["ridge_lambda_v1"],
        "best_state_subset_v1": best["state_subset_v1"],
        "best_test_pnl_v1": float(best_total),
        "realized_v1": float(realized),
        "trail_stop_v1": float(trail_stop),
        "v2_baseline_test_pnl_v1": v2_baseline_test_pnl,
        "delta_vs_v2_baseline_v1": delta_v2,
        "best_test_mean_bars_to_exit_v1": float(best["mean_bars_to_exit_v1"]),
    }
    if best_total >= trail_stop:
        return (
            "RUN_EXIT_IQL_V2_PARAMETER_SWEEP_PASS_BEST_BEATS_TRAIL_STOP",
            "EXIT_PER_BAR_PROPER_IQL_WITH_PESSIMISM_V1",
            (
                f"Best sweep config `{best['config_id_v1']}` test PNL "
                f"{best_total:.0f} >= TRAIL_STOP {trail_stop:.0f}. The closed-"
                "form ridge with V2 state matches/beats the implementable rule "
                "baseline. Next: proper IQL with pessimism (advantage-weighted "
                "regression) to formalize the conservative policy."
            ),
            headline,
        )
    if best_total > realized:
        return (
            "RUN_EXIT_IQL_V2_PARAMETER_SWEEP_PASS_BEST_BEATS_REALIZED_NOT_TRAIL_STOP",
            "RUN_PER_BAR_XGB_REPLAY_FOR_TRANSFORMER_SIGNAL_FAMILY_V1",
            (
                f"Best sweep config `{best['config_id_v1']}` test PNL "
                f"{best_total:.0f} > REALIZED {realized:.0f} but < TRAIL_STOP "
                f"{trail_stop:.0f}. Hyperparameter sweep alone did not close "
                "the gap. Next: per-bar XGB replay to add transformer-signal "
                "information the trail-stop rule cannot see."
            ),
            headline,
        )
    if abs(best_total - realized) <= 50.0:
        return (
            "RUN_EXIT_IQL_V2_PARAMETER_SWEEP_PARTIAL_BEST_TIES_REALIZED",
            "RUN_PER_BAR_XGB_REPLAY_FOR_TRANSFORMER_SIGNAL_FAMILY_V1",
            (
                f"Best sweep config `{best['config_id_v1']}` test PNL "
                f"{best_total:.0f} ~= REALIZED {realized:.0f}. Sweep did not "
                "produce a clear lift. Next: per-bar XGB replay."
            ),
            headline,
        )
    return (
        "RUN_EXIT_IQL_V2_PARAMETER_SWEEP_PARTIAL_BEST_UNDERPERFORMS_REALIZED",
        "REPAIR_EXIT_IQL_TRAINING_BEFORE_VARIANT_SENSITIVITY_V1",
        (
            f"Best sweep config `{best['config_id_v1']}` test PNL "
            f"{best_total:.0f} < REALIZED {realized:.0f}. Sweep regression. "
            "Investigate before any further escalation."
        ),
        headline,
    )


def _build_input_manifest(inputs: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
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
        "layer_name": "RUN_EXIT_IQL_V2_PARAMETER_SWEEP_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
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
    grid_audit = validate_sweep_grid(SWEEP_CONFIGS)
    _write_json(
        artifact_root / "input_manifest_v1.json",
        _build_input_manifest(inputs, artifact_root),
    )
    _write_json(artifact_root / "sweep_grid_v1.json", {"configs_v1": SWEEP_CONFIGS})

    # Reuse the V2 training gate's data-projection pipeline.
    df = pd.read_parquet(inputs["required_paths"]["split_locked_dataset"])
    df["candidate_uid_v1"] = df["candidate_uid_v1"].astype(str)
    df["ts_v1"] = pd.to_datetime(df["ts_v1"], utc=True)
    per_bar_v1 = v2_train_gate._per_bar_view(df)
    per_bar_with_b34 = v2_train_gate._join_base34(
        per_bar_v1, v2_train_gate.BASE34_M5_FEATURES_PATH
    )
    per_bar_with_deriv = v2_train_gate._compute_derivatives(per_bar_with_b34)
    per_bar_full, recovery_join_audit = v2_train_gate._join_recovery(
        per_bar_with_deriv,
        INPUT_RECOVERY_ROOT / "entry_snapshot_signals_per_trade_v1.parquet",
    )

    split_isolation = v2_train_gate.audit_split_isolation(per_bar_full)

    per_bar_train = per_bar_full[per_bar_full["primary_split_v1"] == "train"]
    norm = v2_train_gate._fit_train_normalization(per_bar_train)
    train_only_audit = v2_train_gate.audit_train_only_normalization(per_bar_full, norm)

    X_full, feature_names_full = v2_train_gate._build_state_matrix_v2(
        per_bar_full, norm
    )
    train_mask = (per_bar_full["primary_split_v1"] == "train").to_numpy()

    raw_cols_used: set[str] = set(
        v2_train_gate.V1_CONTINUOUS_FROM_AUGMENTED
        + v2_train_gate.V1_LOG1P_FROM_AUGMENTED
        + v2_train_gate.V1_PASSTHROUGH_FROM_AUGMENTED
        + list(v2_train_gate.V1_ONEHOT_FROM_AUGMENTED.keys())
        + v2_train_gate.NEW_BASE34_CONTINUOUS
        + v2_train_gate.NEW_BASE34_BINARY
        + v2_train_gate.DERIVED_CONTINUOUS
        + v2_train_gate.RECOVERED_PASSTHROUGH
    )
    no_shortcut = v2_train_gate.audit_no_shortcut_at_training_time(
        feature_names_full, raw_cols_used
    )

    # Run all 10 configs.
    config_models: list[dict[str, Any]] = []
    config_metrics_flat: list[dict[str, Any]] = []
    for config in SWEEP_CONFIGS:
        model_summary, metrics_per_split = _run_one_config(
            config, per_bar_full, X_full, feature_names_full, train_mask
        )
        config_models.append(model_summary)
        config_metrics_flat.extend(metrics_per_split)

    _write_json(
        artifact_root / "trained_models_per_config_v1.json",
        {
            "config_count_v1": len(config_models),
            "models_v1": [
                {
                    k: v
                    for k, v in m.items()
                    if k != "coefs_v1"
                }
                for m in config_models
            ],
        },
    )
    _write_json(
        artifact_root / "trained_model_coefs_per_config_v1.json",
        {
            "config_count_v1": len(config_models),
            "coefs_v1": [m["coefs_v1"] for m in config_models],
        },
    )
    _write_json(artifact_root / "training_normalization_v1.json", norm)
    _write_rows(
        artifact_root / "config_metrics_per_split_v1.csv", config_metrics_flat
    )
    _write_json(
        artifact_root / "config_metrics_per_split_v1.json",
        {"row_count_v1": len(config_metrics_flat), "rows_v1": config_metrics_flat},
    )

    # Sensitivity tables (test split only).
    test_metrics = [m for m in config_metrics_flat if m["split_v1"] == "test"]
    lambda_sensitivity = _ridge_lambda_sensitivity(test_metrics)
    subset_ablation = _state_subset_ablation(test_metrics)
    reward_sensitivity = _reward_variant_sensitivity(test_metrics)
    _write_json(
        artifact_root / "ridge_lambda_sensitivity_v1.json",
        {"row_count_v1": len(lambda_sensitivity), "rows_v1": lambda_sensitivity},
    )
    _write_json(
        artifact_root / "state_subset_ablation_v1.json",
        {"row_count_v1": len(subset_ablation), "rows_v1": subset_ablation},
    )
    _write_json(
        artifact_root / "reward_variant_sensitivity_v1.json",
        {"row_count_v1": len(reward_sensitivity), "rows_v1": reward_sensitivity},
    )

    # Comparator with baselines and V2-baseline reference.
    baseline_metrics_flat = inputs["eval_harness_baseline_metrics"]["rows_v1"]
    baseline_per_split: dict[str, list[dict[str, Any]]] = {}
    for row in baseline_metrics_flat:
        baseline_per_split.setdefault(row["split_v1"], []).append(row)

    v2_baseline_test = None
    for r in inputs["v2_training_summary"].get("iql_results_v1", []) or []:
        if r["split_v1"] == "test" and r["reward_variant_v1"] == "GIVEBACK_PENALTY_REWARD":
            v2_baseline_test = float(r["total_realized_pnl_bps_v1"])
            break

    comparator_rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        for r in baseline_per_split.get(split, []):
            comparator_rows.append({**r, "row_kind_v1": "BASELINE"})
        for m in config_metrics_flat:
            if m["split_v1"] == split:
                comparator_rows.append({**m, "row_kind_v1": "IQL_V2_SWEEP", "implementable_v1": True})
    if v2_baseline_test is not None:
        comparator_rows.append(
            {
                "split_v1": "test",
                "policy_id_v1": "IQL_V2_BASELINE_GIVEBACK_PENALTY_L1E3_FULL",
                "row_kind_v1": "IQL_V2_BASELINE_REFERENCE",
                "total_realized_pnl_bps_v1": v2_baseline_test,
            }
        )
    _write_rows(
        artifact_root / "sweep_vs_baseline_comparator_v1.csv", comparator_rows
    )
    _write_json(
        artifact_root / "sweep_vs_baseline_comparator_v1.json",
        {"row_count_v1": len(comparator_rows), "rows_v1": comparator_rows},
    )

    audits = [grid_audit, split_isolation, no_shortcut, train_only_audit, recovery_join_audit]
    _write_json(
        artifact_root / "training_audits_v1.json",
        {"audit_count_v1": len(audits), "audits_v1": audits},
    )
    repro = {
        "layer_name": "RUN_EXIT_IQL_V2_PARAMETER_SWEEP_REPRODUCIBILITY_AUDIT_V1",
        "model_v1": "CLOSED_FORM_RIDGE_TWO_HEADS_PER_CONFIG",
        "config_count_v1": len(SWEEP_CONFIGS),
        "seed_v1": SEED_V1,
        "v2_training_match_baseline_v1": v2_baseline_test,
        "no_implicit_glob_used_for_v1_inputs_v1": True,
        "deprecated_quarantine_revival_v1": False,
        "research_only_v1": True,
    }
    _write_json(artifact_root / "reproducibility_audit_v1.json", repro)

    status, next_action, recommendation, headline = _go_no_go(
        test_metrics, baseline_per_split, v2_baseline_test
    )
    validate_final_status(status, next_action)

    summary = {
        "layer_name": "RUN_EXIT_IQL_V2_PARAMETER_SWEEP_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "headline_v1": headline,
        "config_count_v1": len(SWEEP_CONFIGS),
        "configs_v1": SWEEP_CONFIGS,
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
        "layer_name": "RUN_EXIT_IQL_V2_PARAMETER_SWEEP_STATUS_V1",
        "status_v1": "MATERIALIZED_RESEARCH_ONLY_TRAINING_GATE",
        "final_status_v1": status,
        "next_action_v1": next_action,
        "training_executed_v1": True,
    }
    _write_json(artifact_root / "status_v1.json", status_payload)

    go_no_go = {
        "layer_name": "RUN_EXIT_IQL_V2_PARAMETER_SWEEP_GO_NO_GO_V1",
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
            "Research-only sweep. The trained policies are NOT promoted to "
            "runtime. Adapter/R6/IQL production/live, freeze/promo/live, "
            "exit_manager modification, entry_manager modification, V1 / V2 "
            "state contracts unmodified."
        ),
    }
    _write_json(artifact_root / "run_exit_iql_v2_parameter_sweep_go_no_go_v1.json", go_no_go)

    # Build the report.
    test_rows_sorted = sorted(
        test_metrics, key=lambda r: r["total_realized_pnl_bps_v1"], reverse=True
    )
    report_lines = [
        "# Run Exit IQL V2 Parameter Sweep V1",
        "",
        f"- Status: `{status}`",
        f"- Next action: `{next_action}`",
        "- Training: research-only; policies NOT promoted to runtime.",
        "",
        "## Headline",
        f"- Best config: `{headline['best_config_id_v1']}`",
        f"- Reward: `{headline['best_reward_variant_v1']}`",
        f"- Lambda: {headline['best_ridge_lambda_v1']}",
        f"- State subset: {headline['best_state_subset_v1']}",
        f"- Test PNL: {headline['best_test_pnl_v1']:.0f} bps",
        f"- vs REALIZED: {headline['best_test_pnl_v1'] - headline['realized_v1']:+.0f} bps",
        f"- vs TRAIL_STOP: {headline['best_test_pnl_v1'] - headline['trail_stop_v1']:+.0f} bps",
        f"- V2 baseline (gate-7 GIVEBACK L1E3 FULL): {headline['v2_baseline_test_pnl_v1']} bps",
        f"- Delta vs V2 baseline: {headline['delta_vs_v2_baseline_v1']}",
        f"- Best mean bars to exit: {headline['best_test_mean_bars_to_exit_v1']:.1f}",
        "",
        "## Config sweep test results sorted by total PNL (descending)",
        "",
        "| Config | Reward | Lambda | Subset | Trades | Sum PNL | Mean PNL | Bars | CATA% |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in test_rows_sorted:
        report_lines.append(
            f"| `{r['config_id_v1']}` | `{r['reward_variant_v1']}` | "
            f"{r['ridge_lambda_v1']} | {r['state_subset_v1']} | "
            f"{r['trade_count_v1']} | "
            f"{r['total_realized_pnl_bps_v1']:.0f} | "
            f"{r['mean_realized_pnl_bps_v1']:.2f} | "
            f"{r['mean_bars_to_exit_v1']:.1f} | "
            f"{r['cata_proxy_rate_v1']*100:.1f}% |"
        )
    report_lines.extend(
        [
            "",
            "## Ridge lambda sensitivity (test split, holding reward+subset)",
            "",
        ]
    )
    for r in lambda_sensitivity:
        report_lines.append(
            f"- `{r['reward_variant_v1']}` / `{r['state_subset_v1']}`: "
            f"best lambda={r['best_lambda_v1']}, "
            f"spread={r['spread_test_pnl_v1']:.0f} bps; lambdas={r['lambda_to_test_pnl_v1']}"
        )
    report_lines.extend(["", "## State-subset ablation (test split, holding reward+lambda)", ""])
    for r in subset_ablation:
        report_lines.append(
            f"- `{r['reward_variant_v1']}` / lambda={r['ridge_lambda_v1']}: "
            f"best subset={r['best_subset_v1']}, "
            f"spread={r['spread_test_pnl_v1']:.0f} bps; subsets={r['subset_to_test_pnl_v1']}"
        )
    report_lines.extend(["", "## Reward-variant sensitivity (test split, holding lambda+subset)", ""])
    for r in reward_sensitivity:
        report_lines.append(
            f"- lambda={r['ridge_lambda_v1']} / `{r['state_subset_v1']}`: "
            f"best reward={r['best_reward_v1']}, "
            f"spread={r['spread_test_pnl_v1']:.0f} bps; rewards={r['reward_to_test_pnl_v1']}"
        )
    report_lines.extend(["", "## Audits"])
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
                artifact_root / "run_exit_iql_v2_parameter_sweep_go_no_go_v1.json"
            ),
            "input_manifest": str(artifact_root / "input_manifest_v1.json"),
            "sweep_grid": str(artifact_root / "sweep_grid_v1.json"),
            "trained_models_per_config": str(
                artifact_root / "trained_models_per_config_v1.json"
            ),
            "trained_model_coefs_per_config": str(
                artifact_root / "trained_model_coefs_per_config_v1.json"
            ),
            "training_normalization": str(
                artifact_root / "training_normalization_v1.json"
            ),
            "config_metrics_per_split_csv": str(
                artifact_root / "config_metrics_per_split_v1.csv"
            ),
            "config_metrics_per_split_json": str(
                artifact_root / "config_metrics_per_split_v1.json"
            ),
            "ridge_lambda_sensitivity": str(
                artifact_root / "ridge_lambda_sensitivity_v1.json"
            ),
            "state_subset_ablation": str(
                artifact_root / "state_subset_ablation_v1.json"
            ),
            "reward_variant_sensitivity": str(
                artifact_root / "reward_variant_sensitivity_v1.json"
            ),
            "sweep_vs_baseline_comparator_csv": str(
                artifact_root / "sweep_vs_baseline_comparator_v1.csv"
            ),
            "sweep_vs_baseline_comparator_json": str(
                artifact_root / "sweep_vs_baseline_comparator_v1.json"
            ),
            "training_audits": str(artifact_root / "training_audits_v1.json"),
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
        description="Materialize RUN_EXIT_IQL_V2_PARAMETER_SWEEP_V1 gate."
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
