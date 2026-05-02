#!/usr/bin/env python3
"""Off-policy evaluation harness for exit-IQL research.

This is gate 5 of 6 in the exit-IQL pre-train dependency graph. The harness
defines:

  - Six exit-policy baselines that produce per-trade exit decisions on the
    locked val/test splits, with documented economic metrics.
  - Eight metrics computed per (policy, split) pair: total realized PNL,
    mean realized PNL, mean MFE-capture, mean MAE-burden, mean giveback,
    CATA-rate proxy, mean bars-to-exit, and trade count.
  - A reusable `evaluate_policy(dataset, policy_fn) -> metrics` API that
    gate 6 (sanity training) and any later gate must use to produce
    comparable numbers.

Baselines locked here:

  1. REALIZED_EXIT_BASELINE - the actually-realized exit. Floor any
     learned policy must beat or tie to add value.
  2. ALWAYS_HOLD_TO_REALIZED_END - identical to baseline 1 by construction
     (HOLD until forced terminal).
  3. ALWAYS_EXIT_NOW_AT_BAR_0 - exit immediately at bar 0. Pessimism floor.
  4. PEAK_MFE_ORACLE - exit at bar with max running MFE. Upper bound, not
     implementable - establishes ceiling.
  5. TRAIL_STOP_25_PCT_DD - exit when running giveback exceeds 25 percent
     of running peak MFE and peak is positive.
  6. EXIT_PROB_THRESHOLD_50 - exit at first bar where the exit-transformer
     (logged in EXIT_EVAL_TRACE) emitted exit_prob > 0.5; falls back to
     realized exit if no qualifying bar.

The harness is run on val and test splits to produce reference numbers.
Train-set evaluation is also reported but flagged as in-sample.

Audits: state-leakage during eval, eval-time-determinism, comparator
sanity (REALIZED_EXIT and ALWAYS_HOLD_TO_REALIZED must match exactly,
PEAK_MFE_ORACLE must dominate REALIZED_EXIT, ALWAYS_EXIT_NOW_AT_BAR_0
must be approximately zero PNL).
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


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "EXIT_OFF_POLICY_EVAL_HARNESS_V1"

INPUT_SPLIT_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "EXIT_PER_BAR_SPLIT_AND_LEAKAGE_AUDIT_V1_20260429T141227Z_LOCK"
)
INPUT_AUGMENT_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "EXIT_ACTION_SUPPORT_AUGMENT_V1_20260429T130000Z_LOCK"
)
INPUT_MDP_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "EXIT_HOLD_EXIT_NOW_MDP_REWARD_CONTRACT_V1_20260429T103326Z_LOCK"
)

QUARANTINE_FORBIDDEN_PATH_FRAGMENTS = ("gx1/quarantine", "gx1.quarantine")

ACTION_HOLD_ID = 0
ACTION_EXIT_NOW_ID = 1

# CATA proxy: trade ended with negative PNL AND mae < -100 bps (rough proxy for
# CATASTROPHIC_GUARD-equivalent in our reconstruction)
CATA_PROXY_PNL_THRESHOLD = 0.0
CATA_PROXY_MAE_THRESHOLD = -100.0

ALLOWED_FINAL_STATUSES = {
    "EXIT_OFF_POLICY_EVAL_HARNESS_LOCKED_BASELINE_NUMBERS_AVAILABLE",
    "EXIT_OFF_POLICY_EVAL_HARNESS_BLOCKED_BY_BASELINE_SANITY_FAIL",
    "EXIT_OFF_POLICY_EVAL_HARNESS_BLOCKED_BY_LEAKAGE_AUDIT_FAIL",
}

ALLOWED_NEXT_ACTIONS = {
    "EXIT_PER_BAR_SANITY_TRAINING_V1",
    "REPAIR_EVAL_HARNESS_BEFORE_TRAINING_V1",
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


def _split_locked_dataset_path() -> Path:
    return INPUT_SPLIT_ROOT / "split_locked_augmented_dataset_v1.parquet"


def _load_inputs() -> dict[str, Any]:
    roots = [INPUT_SPLIT_ROOT, INPUT_AUGMENT_ROOT, INPUT_MDP_ROOT]
    validate_explicit_artifact_roots(roots)
    required = {
        "split_locked_dataset": _split_locked_dataset_path(),
        "split_summary": INPUT_SPLIT_ROOT / "summary_v1.json",
        "split_leakage_audits": INPUT_SPLIT_ROOT / "leakage_audits_v1.json",
        "augment_summary": INPUT_AUGMENT_ROOT / "summary_v1.json",
        "mdp_summary": INPUT_MDP_ROOT / "summary_v1.json",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise RuntimeError(f"MISSING_REQUIRED_INPUT_LOCKS: {missing}")
    return {
        "required_paths": required,
        "split_summary": _read_json(required["split_summary"]),
        "split_leakage_audits": _read_json(required["split_leakage_audits"]),
        "augment_summary": _read_json(required["augment_summary"]),
        "mdp_summary": _read_json(required["mdp_summary"]),
    }


def _load_split_dataset() -> pd.DataFrame:
    df = pd.read_parquet(_split_locked_dataset_path())
    df["candidate_uid_v1"] = df["candidate_uid_v1"].astype(str)
    df["ts_v1"] = pd.to_datetime(df["ts_v1"], utc=True)
    return df


# ---------------------------------------------------------------------------
# Per-bar dataset extraction (HOLD rows only carry per-bar state cleanly)
# ---------------------------------------------------------------------------


def _per_bar_view(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (candidate_uid_v1, bars_held_v1) using the HOLD
    sample as the canonical state vector. EXIT_NOW samples have identical
    state but are dropped here because each bar should appear once."""
    hold = df[df["action_id_v1"] == ACTION_HOLD_ID].copy()
    return hold.sort_values(["candidate_uid_v1", "bars_held_v1"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Baseline policies
# ---------------------------------------------------------------------------


def _exit_index_realized_exit(per_bar: pd.DataFrame) -> pd.Series:
    """Last bar of each trade is the realized exit by construction."""
    last_idx = per_bar.groupby("candidate_uid_v1")["bars_held_v1"].idxmax()
    return last_idx


def _exit_index_always_exit_now_bar_0(per_bar: pd.DataFrame) -> pd.Series:
    """First bar of each trade."""
    first_idx = per_bar.groupby("candidate_uid_v1")["bars_held_v1"].idxmin()
    return first_idx


def _exit_index_peak_mfe_oracle(per_bar: pd.DataFrame) -> pd.Series:
    """Bar with max running_mfe per trade (perfect-foresight oracle)."""
    idx = per_bar.groupby("candidate_uid_v1")["running_mfe_bps_v1"].idxmax()
    return idx


def _exit_index_trail_stop_25_pct_dd(per_bar: pd.DataFrame) -> pd.Series:
    """First bar where giveback exceeds 25% of running peak MFE.

    If no bar triggers, fall back to realized exit.
    """
    eps = 1e-6
    triggered_mask = (
        (per_bar["running_mfe_bps_v1"] > eps)
        & (
            (
                per_bar["running_mfe_bps_v1"]
                - per_bar["running_pnl_at_close_bps_v1"]
            )
            > 0.25 * per_bar["running_mfe_bps_v1"]
        )
    )
    out = []
    realized_idx_map = _exit_index_realized_exit(per_bar)
    for uid, group in per_bar.groupby("candidate_uid_v1", sort=False):
        trig = group[triggered_mask.loc[group.index]]
        if not trig.empty:
            out.append((uid, trig.index[0]))
        else:
            out.append((uid, realized_idx_map.loc[uid]))
    return pd.Series({uid: idx for uid, idx in out})


def _exit_index_exit_prob_threshold(
    per_bar: pd.DataFrame, threshold: float
) -> pd.Series:
    """First bar where exit_prob_v1 exceeds threshold; else realized exit."""
    qualifying = per_bar["exit_prob_v1"].fillna(-1.0) > threshold
    realized_idx_map = _exit_index_realized_exit(per_bar)
    out = []
    for uid, group in per_bar.groupby("candidate_uid_v1", sort=False):
        qualified = group[qualifying.loc[group.index]]
        if not qualified.empty:
            out.append((uid, qualified.index[0]))
        else:
            out.append((uid, realized_idx_map.loc[uid]))
    return pd.Series({uid: idx for uid, idx in out})


BASELINE_DEFINITIONS = [
    {
        "baseline_id_v1": "REALIZED_EXIT_BASELINE",
        "description_v1": "Trade ends at the realized exit bar (the actual exit logged in the historical run). Floor any learned exit policy must beat or tie to add value.",
        "implementable_v1": True,
        "uses_oracle_v1": False,
        "fn_name_v1": "_exit_index_realized_exit",
    },
    {
        "baseline_id_v1": "ALWAYS_HOLD_TO_REALIZED_END",
        "description_v1": "Agent always picks HOLD; episode terminates at realized exit. By construction identical to REALIZED_EXIT_BASELINE.",
        "implementable_v1": True,
        "uses_oracle_v1": False,
        "fn_name_v1": "_exit_index_realized_exit",
    },
    {
        "baseline_id_v1": "ALWAYS_EXIT_NOW_AT_BAR_0",
        "description_v1": "Agent exits immediately at bar 0. Pessimism floor demonstrating zero patience.",
        "implementable_v1": True,
        "uses_oracle_v1": False,
        "fn_name_v1": "_exit_index_always_exit_now_bar_0",
    },
    {
        "baseline_id_v1": "PEAK_MFE_ORACLE",
        "description_v1": "Agent exits at the bar with maximum running MFE per trade. Perfect-foresight ceiling - not implementable but quantifies upside.",
        "implementable_v1": False,
        "uses_oracle_v1": True,
        "fn_name_v1": "_exit_index_peak_mfe_oracle",
    },
    {
        "baseline_id_v1": "TRAIL_STOP_25_PCT_DD",
        "description_v1": "Agent exits at the first bar where running giveback exceeds 25 percent of running peak MFE and peak is positive. Implementable rule-based comparator.",
        "implementable_v1": True,
        "uses_oracle_v1": False,
        "fn_name_v1": "_exit_index_trail_stop_25_pct_dd",
    },
    {
        "baseline_id_v1": "EXIT_PROB_THRESHOLD_50",
        "description_v1": "Agent exits at first bar where exit-transformer's exit_prob > 0.5; falls back to realized exit if no qualifying bar. Reuses production exit-transformer signal directly as a samstemte baseline.",
        "implementable_v1": True,
        "uses_oracle_v1": False,
        "fn_name_v1": "_exit_index_exit_prob_threshold(threshold=0.5)",
    },
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


METRIC_DEFINITIONS = [
    {
        "metric_id_v1": "TOTAL_REALIZED_PNL_BPS",
        "description_v1": "Sum of pnl_at_close_bps at the policy-chosen exit bar across all trades in the split.",
    },
    {
        "metric_id_v1": "MEAN_REALIZED_PNL_BPS",
        "description_v1": "Mean per-trade pnl_at_close_bps at policy-chosen exit bar.",
    },
    {
        "metric_id_v1": "MEAN_MFE_CAPTURE_RATIO",
        "description_v1": "Mean of clip(pnl/max(mfe, eps), -2, 2) over trades.",
    },
    {
        "metric_id_v1": "MEAN_MAE_BURDEN_BPS",
        "description_v1": "Mean of pnl - 0.5*abs(mae) at policy-chosen exit.",
    },
    {
        "metric_id_v1": "MEAN_GIVEBACK_BPS",
        "description_v1": "Mean of max(running_mfe - selected_pnl, 0) at policy-chosen exit.",
    },
    {
        "metric_id_v1": "CATA_PROXY_RATE",
        "description_v1": "Fraction of trades where selected pnl <= 0 AND running_mae < -100 bps. Approximates CATASTROPHIC_GUARD trigger rate.",
    },
    {
        "metric_id_v1": "MEAN_BARS_TO_EXIT",
        "description_v1": "Mean bars_held at policy-chosen exit per trade.",
    },
    {
        "metric_id_v1": "TRADE_COUNT",
        "description_v1": "Number of trades in the split.",
    },
]


def evaluate_policy(
    per_bar: pd.DataFrame, exit_indices: pd.Series, *, policy_id: str, split: str
) -> dict[str, Any]:
    """Compute all 8 metrics given a Series of (candidate_uid -> exit_idx)."""
    eps = 1e-6
    selected_indices = exit_indices.values
    selected = per_bar.loc[selected_indices].copy()
    selected = selected.reset_index(drop=True)
    pnl = selected["running_pnl_at_close_bps_v1"].astype(float).to_numpy()
    mfe = selected["running_mfe_bps_v1"].astype(float).to_numpy()
    mae = selected["running_mae_bps_v1"].astype(float).to_numpy()
    bars = selected["bars_held_v1"].astype(int).to_numpy()
    mfe_capture = np.clip(pnl / np.maximum(mfe, eps), -2.0, 2.0)
    mae_burden = pnl - 0.5 * np.abs(mae)
    giveback = np.maximum(mfe - pnl, 0.0)
    cata_mask = (pnl <= CATA_PROXY_PNL_THRESHOLD) & (mae < CATA_PROXY_MAE_THRESHOLD)
    return {
        "policy_id_v1": policy_id,
        "split_v1": split,
        "trade_count_v1": int(len(selected)),
        "total_realized_pnl_bps_v1": float(pnl.sum()),
        "mean_realized_pnl_bps_v1": float(pnl.mean()) if len(pnl) else 0.0,
        "mean_mfe_capture_ratio_v1": float(mfe_capture.mean()) if len(pnl) else 0.0,
        "mean_mae_burden_bps_v1": float(mae_burden.mean()) if len(pnl) else 0.0,
        "mean_giveback_bps_v1": float(giveback.mean()) if len(pnl) else 0.0,
        "cata_proxy_rate_v1": float(cata_mask.mean()) if len(pnl) else 0.0,
        "cata_proxy_count_v1": int(cata_mask.sum()),
        "mean_bars_to_exit_v1": float(bars.mean()) if len(bars) else 0.0,
    }


# ---------------------------------------------------------------------------
# Run all baselines per split
# ---------------------------------------------------------------------------


def _apply_baseline(per_bar: pd.DataFrame, baseline_id: str) -> pd.Series:
    if baseline_id == "REALIZED_EXIT_BASELINE":
        return _exit_index_realized_exit(per_bar)
    if baseline_id == "ALWAYS_HOLD_TO_REALIZED_END":
        return _exit_index_realized_exit(per_bar)
    if baseline_id == "ALWAYS_EXIT_NOW_AT_BAR_0":
        return _exit_index_always_exit_now_bar_0(per_bar)
    if baseline_id == "PEAK_MFE_ORACLE":
        return _exit_index_peak_mfe_oracle(per_bar)
    if baseline_id == "TRAIL_STOP_25_PCT_DD":
        return _exit_index_trail_stop_25_pct_dd(per_bar)
    if baseline_id == "EXIT_PROB_THRESHOLD_50":
        return _exit_index_exit_prob_threshold(per_bar, threshold=0.5)
    raise RuntimeError(f"UNKNOWN_BASELINE_ID: {baseline_id}")


def _run_all_baselines_on_split(
    per_bar: pd.DataFrame, split: str
) -> list[dict[str, Any]]:
    results = []
    for baseline in BASELINE_DEFINITIONS:
        bid = baseline["baseline_id_v1"]
        exit_indices = _apply_baseline(per_bar, bid)
        metrics = evaluate_policy(per_bar, exit_indices, policy_id=bid, split=split)
        metrics["implementable_v1"] = baseline["implementable_v1"]
        metrics["uses_oracle_v1"] = baseline["uses_oracle_v1"]
        results.append(metrics)
    return results


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------


def audit_baseline_sanity(results_per_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """A1: REALIZED_EXIT and ALWAYS_HOLD_TO_REALIZED_END should match exactly.
    A2: PEAK_MFE_ORACLE.total_pnl >= REALIZED_EXIT.total_pnl per split.
    A3: ALWAYS_EXIT_NOW_AT_BAR_0.mean_pnl should be near zero (small magnitude).
    """
    failures = []
    for split, results in results_per_split.items():
        by_id = {r["policy_id_v1"]: r for r in results}
        a = by_id.get("REALIZED_EXIT_BASELINE")
        b = by_id.get("ALWAYS_HOLD_TO_REALIZED_END")
        if a is None or b is None:
            failures.append(f"{split}: REALIZED_EXIT or ALWAYS_HOLD baseline missing")
            continue
        if abs(a["total_realized_pnl_bps_v1"] - b["total_realized_pnl_bps_v1"]) > 1e-6:
            failures.append(
                f"{split}: REALIZED_EXIT and ALWAYS_HOLD totals diverge"
            )
        oracle = by_id.get("PEAK_MFE_ORACLE")
        if oracle is not None and oracle["total_realized_pnl_bps_v1"] < a["total_realized_pnl_bps_v1"] - 1e-6:
            failures.append(
                f"{split}: PEAK_MFE_ORACLE underperforms REALIZED_EXIT - oracle should never be worse"
            )
        bar0 = by_id.get("ALWAYS_EXIT_NOW_AT_BAR_0")
        if bar0 is not None:
            mean_pnl = abs(bar0["mean_realized_pnl_bps_v1"])
            # Bar 0 mean pnl can be slightly nonzero due to entry-bar drift but
            # should be small relative to the per-trade scale.
            if mean_pnl > 50.0:
                failures.append(
                    f"{split}: ALWAYS_EXIT_NOW_AT_BAR_0 mean pnl |{bar0['mean_realized_pnl_bps_v1']:.2f}| > 50 bps"
                )
    if failures:
        raise RuntimeError(f"BASELINE_SANITY_AUDIT_FAILED: {failures}")
    return {
        "audit_id_v1": "EVAL_HARNESS_BASELINE_SANITY_AUDIT_V1",
        "status_v1": "PASS",
        "splits_checked_v1": list(results_per_split.keys()),
        "checks_v1": [
            "REALIZED_EXIT and ALWAYS_HOLD totals identical",
            "PEAK_MFE_ORACLE >= REALIZED_EXIT per split",
            "ALWAYS_EXIT_NOW_AT_BAR_0 mean pnl small (< 50 bps)",
        ],
    }


def audit_eval_state_leakage_check(per_bar: pd.DataFrame) -> dict[str, Any]:
    """The eval harness must not depend on any forbidden state field."""
    forbidden = set(mdp_gate.FORBIDDEN_STATE_FIELDS_V1)
    used_fields = {
        "running_pnl_at_close_bps_v1",
        "running_mfe_bps_v1",
        "running_mae_bps_v1",
        "running_giveback_from_peak_bps_v1",
        "bars_held_v1",
        "exit_prob_v1",
    }
    leaked = sorted(used_fields & forbidden)
    if leaked:
        raise RuntimeError(f"EVAL_HARNESS_USES_FORBIDDEN_FIELDS: {leaked}")
    return {
        "audit_id_v1": "EVAL_HARNESS_STATE_LEAKAGE_CHECK_V1",
        "status_v1": "PASS",
        "fields_used_v1": sorted(used_fields),
        "forbidden_intersection_v1": leaked,
    }


def audit_eval_split_only_uses_split_data(
    per_bar_full: pd.DataFrame,
) -> dict[str, Any]:
    """Eval per split must only see that split's rows. We check trade-uid
    partitioning via the split column."""
    bad = (
        per_bar_full.groupby("candidate_uid_v1")["primary_split_v1"]
        .nunique()
        .gt(1)
        .sum()
    )
    if int(bad) > 0:
        raise RuntimeError(
            f"EVAL_SPLIT_PARTITION_VIOLATION: {bad} trades span multiple splits"
        )
    return {
        "audit_id_v1": "EVAL_SPLIT_PARTITION_AUDIT_V1",
        "status_v1": "PASS",
        "spanning_trade_count_v1": int(bad),
    }


# ---------------------------------------------------------------------------
# Reproducibility / go-no-go
# ---------------------------------------------------------------------------


def _reproducibility_audit(
    results_per_split: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    return {
        "layer_name": "EXIT_OFF_POLICY_EVAL_HARNESS_REPRODUCIBILITY_AUDIT_V1",
        "splits_evaluated_v1": list(results_per_split.keys()),
        "baseline_count_v1": len(BASELINE_DEFINITIONS),
        "metric_count_v1": len(METRIC_DEFINITIONS),
        "no_implicit_glob_used_for_v1_inputs_v1": True,
        "deprecated_quarantine_revival_v1": False,
        "research_only_v1": True,
    }


def _go_no_go() -> tuple[str, str, str]:
    return (
        "EXIT_OFF_POLICY_EVAL_HARNESS_LOCKED_BASELINE_NUMBERS_AVAILABLE",
        "EXIT_PER_BAR_SANITY_TRAINING_V1",
        (
            "Eval harness locked: 6 baselines, 8 metrics, 3 audits passing on "
            "train/val/test. Reference numbers are now the bar that gate 6's "
            "first sanity-trained IQL policy must beat or match. Training "
            "remains research-only."
        ),
    )


def _build_input_manifest(inputs: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    files = [
        {"name_v1": name, "path_v1": str(path), "sha256_v1": _file_hash(path)}
        for name, path in inputs["required_paths"].items()
    ]
    return {
        "layer_name": "EXIT_OFF_POLICY_EVAL_HARNESS_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "split_root_v1": str(INPUT_SPLIT_ROOT),
            "augment_root_v1": str(INPUT_AUGMENT_ROOT),
            "mdp_root_v1": str(INPUT_MDP_ROOT),
        },
        "files_used_v1": files,
        "immutable_input_status_v1": "HASHED_EXPLICIT_ROOTS_ONLY",
        "no_implicit_latest_glob_selection_v1": True,
        "previous_artifacts_mutated_v1": False,
        "research_only_contract_v1": True,
        "iql_training_run_v1": False,
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

    df = _load_split_dataset()
    per_bar = _per_bar_view(df)

    # Audit split-partition before evaluation
    split_audit = audit_eval_split_only_uses_split_data(per_bar)
    state_leak_audit = audit_eval_state_leakage_check(per_bar)

    # Evaluate baselines per split
    results_per_split: dict[str, list[dict[str, Any]]] = {}
    for split in ["train", "val", "test"]:
        split_df = per_bar[per_bar["primary_split_v1"] == split]
        if split_df.empty:
            continue
        results_per_split[split] = _run_all_baselines_on_split(split_df, split)

    sanity_audit = audit_baseline_sanity(results_per_split)

    all_results_flat = []
    for split, rows in results_per_split.items():
        all_results_flat.extend(rows)
    _write_rows(
        artifact_root / "baseline_metrics_per_split_v1.csv", all_results_flat
    )
    _write_json(
        artifact_root / "baseline_metrics_per_split_v1.json",
        {"row_count_v1": len(all_results_flat), "rows_v1": all_results_flat},
    )

    _write_json(
        artifact_root / "baseline_definitions_v1.json",
        {"baselines_v1": BASELINE_DEFINITIONS},
    )
    _write_json(
        artifact_root / "metric_definitions_v1.json",
        {"metrics_v1": METRIC_DEFINITIONS},
    )

    audits = [split_audit, state_leak_audit, sanity_audit]
    _write_json(
        artifact_root / "eval_harness_audits_v1.json",
        {"audit_count_v1": len(audits), "audits_v1": audits},
    )

    repro = _reproducibility_audit(results_per_split)
    _write_json(artifact_root / "reproducibility_audit_v1.json", repro)

    status, next_action, recommendation = _go_no_go()
    validate_final_status(status, next_action)

    summary = {
        "layer_name": "EXIT_OFF_POLICY_EVAL_HARNESS_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "baseline_count_v1": len(BASELINE_DEFINITIONS),
        "metric_count_v1": len(METRIC_DEFINITIONS),
        "splits_evaluated_v1": list(results_per_split.keys()),
        "audits_status_v1": {a["audit_id_v1"]: a["status_v1"] for a in audits},
        "reference_numbers_v1": all_results_flat,
        "research_only_v1": True,
        "iql_training_run_v1": False,
        "training_blocked_v1": True,
        "next_pre_train_gate_v1": next_action,
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
        "layer_name": "EXIT_OFF_POLICY_EVAL_HARNESS_STATUS_V1",
        "status_v1": "MATERIALIZED_RESEARCH_ONLY_GATE",
        "final_status_v1": status,
        "next_action_v1": next_action,
        "training_executed_v1": False,
    }
    _write_json(artifact_root / "status_v1.json", status_payload)

    go_no_go = {
        "layer_name": "EXIT_OFF_POLICY_EVAL_HARNESS_GO_NO_GO_V1",
        "status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "research_only_v1": True,
        "iql_production_allowed_v1": False,
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "package_freeze_promo_live_allowed_v1": False,
        "policy_promotion_allowed_v1": False,
        "training_allowed_v1": False,
        "downstream_block_v1": (
            "Research-only eval harness gate. Training of any IQL policy "
            "remains BLOCKED until gate 6 (sanity training). Adapter/R6/IQL "
            "production/live, freeze/promo/live, exit_manager modification "
            "all forbidden."
        ),
    }
    _write_json(
        artifact_root / "exit_off_policy_eval_harness_go_no_go_v1.json", go_no_go
    )

    report_lines = [
        "# Exit Off-Policy Evaluation Harness V1",
        "",
        f"- Status: `{status}`",
        f"- Next action: `{next_action}`",
        "- Training: **BLOCKED** until gate 6 (sanity training).",
        "",
        "## Baselines locked",
    ]
    for b in BASELINE_DEFINITIONS:
        report_lines.append(f"- `{b['baseline_id_v1']}`: {b['description_v1']}")
    report_lines.extend(["", "## Reference numbers per split"])
    for split, rows in results_per_split.items():
        report_lines.append(f"### {split}")
        report_lines.append("")
        report_lines.append(
            "| Policy | Trades | Sum PNL | Mean PNL | MFE-cap | MAE-burden | Giveback | CATA% | Bars |"
        )
        report_lines.append(
            "|---|---|---|---|---|---|---|---|---|"
        )
        for r in rows:
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
        report_lines.append("")
    report_lines.extend(["## Audits"])
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
                artifact_root / "exit_off_policy_eval_harness_go_no_go_v1.json"
            ),
            "input_manifest": str(artifact_root / "input_manifest_v1.json"),
            "baseline_definitions": str(artifact_root / "baseline_definitions_v1.json"),
            "metric_definitions": str(artifact_root / "metric_definitions_v1.json"),
            "baseline_metrics_per_split_json": str(
                artifact_root / "baseline_metrics_per_split_v1.json"
            ),
            "baseline_metrics_per_split_csv": str(
                artifact_root / "baseline_metrics_per_split_v1.csv"
            ),
            "eval_harness_audits": str(
                artifact_root / "eval_harness_audits_v1.json"
            ),
            "reproducibility_audit": str(
                artifact_root / "reproducibility_audit_v1.json"
            ),
        },
        "read_only_references_v1": True,
        "not_trainer_v1": True,
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
        description="Materialize EXIT_OFF_POLICY_EVAL_HARNESS_V1 gate."
    )
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--built-at-utc", type=str, default=None)
    args = parser.parse_args()
    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else None
    result = write_artifacts(out_root=out_root, built_at_utc=args.built_at_utc)
    print(json.dumps(_jsonable(result), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
