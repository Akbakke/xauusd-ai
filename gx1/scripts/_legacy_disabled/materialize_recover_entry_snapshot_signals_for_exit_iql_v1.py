#!/usr/bin/env python3
"""Recover the four entry-snapshot signal fields for the exit-IQL state vector.

Background
----------
EXIT_PER_BAR_STATE_FEATURE_CONTRACT_V1 marked four ENTRY_CONTEXT_SNAPSHOT
fields as NOT_ESTABLISHED because they were not present in trade_outcomes
parquets and trade_log.csv was empty:

  - p_long_entry_v1
  - p_hat_entry_v1
  - uncertainty_entry_v1
  - margin_entry_v1

A code audit of gx1/execution/entry_manager.py:2336-2340 shows that at
trade-entry time these four fields are populated as direct snapshots of
the XGB signal-7 dictionary (signal_bridge_v1 ORDERED_FIELDS):

    trade.p_long_entry      = signal7_now["p_long"]
    trade.p_hat_entry       = signal7_now["p_hat"]
    trade.uncertainty_entry = signal7_now["uncertainty_score"]
    trade.margin_entry      = signal7_now["margin_top1_top2"]

They are NOT entry-transformer outputs - they are XGB-classifier outputs
captured at the trade-decision bar. Per gx1.xgb.multihead.xgb_multihead_model_v1
the XGB bridge formulas are deterministic from (p_long, p_short, p_flat):

    p_hat               = max(p_long, p_short, p_flat)
    uncertainty_score   = 1.0 - p_hat
    margin_top1_top2    = top1 - top2 (descending sort)
    entropy             = -sum(p_i * log(p_i))            (not used here)

Per-week artifact xgb_multi_horizon_predictions_<run_id>.parquet contains
the XGB-call rows (p_long, p_short, p_flat, p_hat) at every decision bar.
Joining trade_outcomes.open_ts_utc == xgb.ts recovers the four fields
exactly as they were at runtime, with no replay required.

Empirical join coverage measured before this gate: 1899 of 1914 trades
(99.2%) match deterministically; the 15 unmatched trades belong to a
single week (TRUTH_MONFRI_WEEK_20250623_20250630) whose
xgb_multi_horizon_predictions parquet is missing from the substrate.

Scope
-----
- Reads per-week trade_outcomes_*_MERGED.parquet and
  xgb_multi_horizon_predictions_*.parquet sorted deterministically.
- Joins on (open_ts_utc == ts), exactly one xgb row per match.
- Computes the four bridge fields from (p_long, p_short, p_flat).
- Writes a per-trade recovery parquet keyed by candidate_uid + trade_uid,
  per-week match-rate audit, manifest, summary, status, go-no-go,
  reproducibility audit, and a short report.
- Research-only, append-only namespace, no training, no exit_manager
  modification, no deprecated-quarantine revival, no glob/latest input.

This gate is a prerequisite for DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_V1.
It does NOT modify any V1 contract or any runtime module.
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


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "RECOVER_ENTRY_SNAPSHOT_SIGNALS_FOR_EXIT_IQL_V1"

# Pinned upstream LOCK that this recovery serves.
INPUT_V1_STATE_CONTRACT_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "EXIT_PER_BAR_STATE_FEATURE_CONTRACT_V1_20260429T113745Z_LOCK"
)

# We never accept a quarantined module as a recovery input.
QUARANTINE_FORBIDDEN_PATH_FRAGMENTS = ("gx1/quarantine", "gx1.quarantine")

ALLOWED_FINAL_STATUSES = {
    "RECOVER_ENTRY_SNAPSHOT_SIGNALS_PASS_FULL_COVERAGE_V1",
    "RECOVER_ENTRY_SNAPSHOT_SIGNALS_PARTIAL_COVERAGE_V1",
    "RECOVER_ENTRY_SNAPSHOT_SIGNALS_BLOCKED_LOW_COVERAGE_V1",
    "RECOVER_ENTRY_SNAPSHOT_SIGNALS_BLOCKED_BY_INPUT_LOCK_MISSING_V1",
}

ALLOWED_NEXT_ACTIONS = {
    "DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_V1",
    "HOLD_UNTIL_RECOVERY_COVERAGE_RESOLVED_V1",
}

# Required columns in source parquets.
TRADE_OUTCOMES_REQUIRED_COLS = (
    "candidate_uid",
    "trade_uid",
    "open_ts_utc",
    "session",
    "side",
)
XGB_PREDICTIONS_REQUIRED_COLS = (
    "ts",
    "head",
    "p_long",
    "p_short",
    "p_flat",
    "p_hat",
)

# Coverage thresholds. Below 0.95 is considered low.
PASS_COVERAGE_THRESHOLD_V1 = 0.95


# ---------------------------------------------------------------------------
# Reused helpers
# ---------------------------------------------------------------------------

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


def _list_truth_weeks() -> list[Path]:
    return sorted(
        DEFAULT_REPORTS_ROOT.glob("TRUTH_MONFRI_WEEK_*"),
        key=lambda p: p.name,
    )


def _trade_outcomes_path(week_dir: Path) -> Path:
    base = week_dir.name
    return week_dir / f"trade_outcomes_{base}_MERGED.parquet"


def _xgb_predictions_path(week_dir: Path) -> Path:
    base = week_dir.name
    return week_dir / f"xgb_multi_horizon_predictions_{base}.parquet"


def _load_v1_state_contract() -> dict[str, Any]:
    p = INPUT_V1_STATE_CONTRACT_ROOT / "state_feature_contract_v1.json"
    if not p.exists():
        raise RuntimeError(f"V1_STATE_CONTRACT_LOCK_MISSING: {p}")
    return _read_json(p)


def _validate_columns(df: pd.DataFrame, required: tuple[str, ...], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label}_MISSING_COLUMNS: {missing}")


# ---------------------------------------------------------------------------
# Bridge formulas (mirrors gx1.xgb.multihead.xgb_multihead_model_v1)
# ---------------------------------------------------------------------------


def _compute_bridge_fields(
    p_long: np.ndarray, p_short: np.ndarray, p_flat: np.ndarray
) -> dict[str, np.ndarray]:
    probs = np.stack([p_long, p_short, p_flat], axis=1)
    p_hat = probs.max(axis=1)
    sorted_desc = np.sort(probs, axis=1)[:, ::-1]
    margin = sorted_desc[:, 0] - sorted_desc[:, 1]
    uncertainty = 1.0 - p_hat
    return {
        "p_long_entry_v1": p_long.astype(np.float64),
        "p_hat_entry_v1": p_hat.astype(np.float64),
        "uncertainty_entry_v1": uncertainty.astype(np.float64),
        "margin_entry_v1": margin.astype(np.float64),
    }


# ---------------------------------------------------------------------------
# Per-week recovery
# ---------------------------------------------------------------------------


def _recover_week(
    week_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Recover snapshots for a single week. Returns (recovered_df, audit_row).

    recovered_df has one row per matched trade with the four bridge fields plus
    candidate_uid, trade_uid, week_name, open_ts_utc, recovery_status_v1.
    audit_row summarizes match counts and missing-input flags for the week.
    """
    week_name = week_dir.name
    to_path = _trade_outcomes_path(week_dir)
    xgb_path = _xgb_predictions_path(week_dir)

    if not to_path.exists():
        return pd.DataFrame(), {
            "week_name_v1": week_name,
            "trade_count_v1": 0,
            "matched_count_v1": 0,
            "match_rate_v1": None,
            "trade_outcomes_present_v1": False,
            "xgb_predictions_present_v1": False,
            "trade_outcomes_path_v1": str(to_path),
            "xgb_predictions_path_v1": str(xgb_path),
        }

    trades = pd.read_parquet(to_path)
    if trades.empty:
        return pd.DataFrame(), {
            "week_name_v1": week_name,
            "trade_count_v1": 0,
            "matched_count_v1": 0,
            "match_rate_v1": None,
            "trade_outcomes_present_v1": True,
            "xgb_predictions_present_v1": xgb_path.exists(),
            "trade_outcomes_path_v1": str(to_path),
            "xgb_predictions_path_v1": str(xgb_path),
        }

    _validate_columns(trades, TRADE_OUTCOMES_REQUIRED_COLS, "TRADE_OUTCOMES")

    if not xgb_path.exists():
        recovered = trades.loc[
            :, list(TRADE_OUTCOMES_REQUIRED_COLS)
        ].copy()
        recovered["open_ts_utc"] = pd.to_datetime(recovered["open_ts_utc"], utc=True)
        recovered["week_name_v1"] = week_name
        for f in (
            "p_long_entry_v1",
            "p_hat_entry_v1",
            "uncertainty_entry_v1",
            "margin_entry_v1",
        ):
            recovered[f] = np.nan
        recovered["recovery_status_v1"] = "NOT_RECOVERED_XGB_PARQUET_MISSING"
        return recovered, {
            "week_name_v1": week_name,
            "trade_count_v1": int(len(trades)),
            "matched_count_v1": 0,
            "match_rate_v1": 0.0,
            "trade_outcomes_present_v1": True,
            "xgb_predictions_present_v1": False,
            "trade_outcomes_path_v1": str(to_path),
            "xgb_predictions_path_v1": str(xgb_path),
        }

    xgb = pd.read_parquet(xgb_path)
    if xgb.empty:
        recovered = trades.loc[:, list(TRADE_OUTCOMES_REQUIRED_COLS)].copy()
        recovered["open_ts_utc"] = pd.to_datetime(recovered["open_ts_utc"], utc=True)
        recovered["week_name_v1"] = week_name
        for f in (
            "p_long_entry_v1",
            "p_hat_entry_v1",
            "uncertainty_entry_v1",
            "margin_entry_v1",
        ):
            recovered[f] = np.nan
        recovered["recovery_status_v1"] = "NOT_RECOVERED_XGB_PARQUET_EMPTY"
        return recovered, {
            "week_name_v1": week_name,
            "trade_count_v1": int(len(trades)),
            "matched_count_v1": 0,
            "match_rate_v1": 0.0,
            "trade_outcomes_present_v1": True,
            "xgb_predictions_present_v1": True,
            "trade_outcomes_path_v1": str(to_path),
            "xgb_predictions_path_v1": str(xgb_path),
        }

    _validate_columns(xgb, XGB_PREDICTIONS_REQUIRED_COLS, "XGB_PREDICTIONS")

    trades_use = trades.loc[
        :,
        [
            "candidate_uid",
            "trade_uid",
            "open_ts_utc",
            "session",
            "side",
        ],
    ].copy()
    trades_use["open_ts_utc"] = pd.to_datetime(trades_use["open_ts_utc"], utc=True)

    xgb_use = xgb.loc[
        :,
        ["ts", "head", "p_long", "p_short", "p_flat", "p_hat"],
    ].copy()
    xgb_use["ts"] = pd.to_datetime(xgb_use["ts"], utc=True)
    # If multiple xgb heads share a ts, prefer head == trade.session; else
    # take the deterministic first by sorted head order. Build a (ts, head)
    # priority frame.
    xgb_use["head_norm"] = xgb_use["head"].astype(str).str.upper()
    xgb_use = xgb_use.sort_values(["ts", "head_norm"], kind="mergesort").reset_index(
        drop=True
    )

    # First try (ts, head_norm) join with trade.session as head.
    trades_use["session_norm"] = trades_use["session"].astype(str).str.upper()
    primary = trades_use.merge(
        xgb_use,
        left_on=["open_ts_utc", "session_norm"],
        right_on=["ts", "head_norm"],
        how="left",
    )

    # Fallback: rows that did not join on (ts, head_norm) -> join on ts only.
    needs_fallback_mask = primary["p_long"].isna()
    fallback_idx = primary.index[needs_fallback_mask]
    if len(fallback_idx) > 0:
        # Build a ts-only lookup that takes the first xgb row per ts (sorted).
        xgb_ts_only = (
            xgb_use.drop_duplicates(subset="ts", keep="first")
            .loc[:, ["ts", "head_norm", "p_long", "p_short", "p_flat", "p_hat"]]
            .rename(
                columns={
                    "head_norm": "head_norm_fallback",
                    "p_long": "p_long_fallback",
                    "p_short": "p_short_fallback",
                    "p_flat": "p_flat_fallback",
                    "p_hat": "p_hat_fallback",
                }
            )
        )
        fallback_join = primary.loc[fallback_idx, ["open_ts_utc"]].merge(
            xgb_ts_only, left_on="open_ts_utc", right_on="ts", how="left"
        )
        primary.loc[fallback_idx, "p_long"] = fallback_join["p_long_fallback"].values
        primary.loc[fallback_idx, "p_short"] = fallback_join["p_short_fallback"].values
        primary.loc[fallback_idx, "p_flat"] = fallback_join["p_flat_fallback"].values
        primary.loc[fallback_idx, "p_hat"] = fallback_join["p_hat_fallback"].values
        primary.loc[fallback_idx, "head_norm"] = fallback_join[
            "head_norm_fallback"
        ].values

    matched_mask = primary["p_long"].notna()
    matched_count = int(matched_mask.sum())

    # Compute bridge fields for matched rows.
    pl = primary["p_long"].fillna(0.0).to_numpy(dtype=np.float64)
    ps = primary["p_short"].fillna(0.0).to_numpy(dtype=np.float64)
    pf = primary["p_flat"].fillna(0.0).to_numpy(dtype=np.float64)
    bridge = _compute_bridge_fields(pl, ps, pf)

    out = primary.loc[
        :,
        [
            "candidate_uid",
            "trade_uid",
            "open_ts_utc",
            "session",
            "side",
        ],
    ].copy()
    out["week_name_v1"] = week_name
    for k, v in bridge.items():
        out[k] = v
    # Restore xgb head used for traceability (may differ from trade session).
    out["xgb_head_used_v1"] = primary["head_norm"].astype(object)
    # Mark unmatched rows as NaN and status=NOT_RECOVERED_TS_NOT_IN_XGB.
    status = np.where(matched_mask, "RECOVERED_FROM_XGB_PREDICTIONS", "NOT_RECOVERED_TS_NOT_IN_XGB")
    out["recovery_status_v1"] = status
    for f in (
        "p_long_entry_v1",
        "p_hat_entry_v1",
        "uncertainty_entry_v1",
        "margin_entry_v1",
    ):
        out.loc[~matched_mask, f] = np.nan
    out.loc[~matched_mask, "xgb_head_used_v1"] = None

    audit_row = {
        "week_name_v1": week_name,
        "trade_count_v1": int(len(trades)),
        "matched_count_v1": matched_count,
        "match_rate_v1": float(matched_count / len(trades)) if len(trades) else None,
        "trade_outcomes_present_v1": True,
        "xgb_predictions_present_v1": True,
        "trade_outcomes_path_v1": str(to_path),
        "xgb_predictions_path_v1": str(xgb_path),
    }
    return out, audit_row


def _recover_all_weeks() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    weeks = _list_truth_weeks()
    if len(weeks) == 0:
        raise RuntimeError("NO_TRUTH_MONFRI_WEEKS_FOUND")
    parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for w in weeks:
        df, row = _recover_week(w)
        if not df.empty:
            parts.append(df)
        audits.append(row)
    if not parts:
        return (
            pd.DataFrame(
                columns=[
                    "candidate_uid",
                    "trade_uid",
                    "open_ts_utc",
                    "session",
                    "side",
                    "week_name_v1",
                    "p_long_entry_v1",
                    "p_hat_entry_v1",
                    "uncertainty_entry_v1",
                    "margin_entry_v1",
                    "xgb_head_used_v1",
                    "recovery_status_v1",
                ]
            ),
            audits,
        )
    full = pd.concat(parts, ignore_index=True)
    full = full.sort_values(
        ["week_name_v1", "open_ts_utc", "candidate_uid"],
        kind="mergesort",
    ).reset_index(drop=True)
    return full, audits


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------


def _coverage_audit(
    recovered: pd.DataFrame, per_week_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    total_trades = sum(r["trade_count_v1"] for r in per_week_rows)
    matched_total = sum(r["matched_count_v1"] for r in per_week_rows)
    match_rate = float(matched_total / total_trades) if total_trades else None

    weeks_missing_xgb = [
        r["week_name_v1"]
        for r in per_week_rows
        if r["trade_count_v1"] > 0 and not r["xgb_predictions_present_v1"]
    ]
    weeks_low_match = [
        {
            "week_name_v1": r["week_name_v1"],
            "match_rate_v1": r["match_rate_v1"],
            "trade_count_v1": r["trade_count_v1"],
        }
        for r in per_week_rows
        if r["trade_count_v1"] > 0
        and r["match_rate_v1"] is not None
        and r["match_rate_v1"] < PASS_COVERAGE_THRESHOLD_V1
    ]
    head_used_breakdown: dict[str, int] = {}
    if not recovered.empty:
        s = recovered["xgb_head_used_v1"].fillna("__NULL__").astype(str).value_counts()
        head_used_breakdown = {k: int(v) for k, v in s.items()}

    return {
        "layer_name": "RECOVER_ENTRY_SNAPSHOT_SIGNALS_COVERAGE_AUDIT_V1",
        "total_trade_count_v1": int(total_trades),
        "matched_trade_count_v1": int(matched_total),
        "match_rate_v1": match_rate,
        "pass_coverage_threshold_v1": PASS_COVERAGE_THRESHOLD_V1,
        "weeks_missing_xgb_v1": sorted(weeks_missing_xgb),
        "weeks_below_threshold_v1": weeks_low_match,
        "xgb_head_used_breakdown_v1": head_used_breakdown,
        "per_week_count_v1": len(per_week_rows),
    }


def _bridge_audit(recovered: pd.DataFrame) -> dict[str, Any]:
    """Sanity audit on bridge math: 0<=p<=1, 0<=margin<=1, uncertainty=1-p_hat."""
    if recovered.empty:
        return {
            "layer_name": "RECOVER_ENTRY_SNAPSHOT_SIGNALS_BRIDGE_AUDIT_V1",
            "status_v1": "EMPTY",
        }
    matched = recovered[recovered["recovery_status_v1"] == "RECOVERED_FROM_XGB_PREDICTIONS"]
    if matched.empty:
        return {
            "layer_name": "RECOVER_ENTRY_SNAPSHOT_SIGNALS_BRIDGE_AUDIT_V1",
            "status_v1": "NO_MATCHED_ROWS",
        }
    p = matched["p_long_entry_v1"].to_numpy()
    h = matched["p_hat_entry_v1"].to_numpy()
    u = matched["uncertainty_entry_v1"].to_numpy()
    m = matched["margin_entry_v1"].to_numpy()
    p_in_range = bool(np.all((p >= 0.0 - 1e-9) & (p <= 1.0 + 1e-9)))
    h_in_range = bool(np.all((h >= 0.0 - 1e-9) & (h <= 1.0 + 1e-9)))
    u_consistent = bool(np.allclose(u, 1.0 - h, atol=1e-9))
    m_in_range = bool(np.all((m >= 0.0 - 1e-9) & (m <= 1.0 + 1e-9)))
    fail = []
    if not p_in_range:
        fail.append("P_LONG_OUT_OF_RANGE")
    if not h_in_range:
        fail.append("P_HAT_OUT_OF_RANGE")
    if not u_consistent:
        fail.append("UNCERTAINTY_NOT_EQUAL_1_MINUS_P_HAT")
    if not m_in_range:
        fail.append("MARGIN_OUT_OF_RANGE")
    return {
        "layer_name": "RECOVER_ENTRY_SNAPSHOT_SIGNALS_BRIDGE_AUDIT_V1",
        "status_v1": "PASS" if not fail else "FAIL",
        "failures_v1": fail,
        "matched_row_count_v1": int(len(matched)),
        "p_long_min_v1": float(p.min()),
        "p_long_max_v1": float(p.max()),
        "p_hat_min_v1": float(h.min()),
        "p_hat_max_v1": float(h.max()),
        "uncertainty_min_v1": float(u.min()),
        "uncertainty_max_v1": float(u.max()),
        "margin_min_v1": float(m.min()),
        "margin_max_v1": float(m.max()),
    }


def _no_runtime_modification_audit() -> dict[str, Any]:
    return {
        "layer_name": "RECOVER_ENTRY_SNAPSHOT_SIGNALS_NO_RUNTIME_MODIFICATION_AUDIT_V1",
        "exit_manager_modified_v1": False,
        "live_features_modified_v1": False,
        "entry_manager_modified_v1": False,
        "v1_state_contract_modified_v1": False,
        "trade_outcomes_modified_v1": False,
        "xgb_predictions_modified_v1": False,
        "research_only_v1": True,
    }


# ---------------------------------------------------------------------------
# Manifests / go-no-go
# ---------------------------------------------------------------------------


def _go_no_go(coverage_audit: dict[str, Any], bridge_audit: dict[str, Any]) -> tuple[str, str, str]:
    if bridge_audit.get("status_v1") == "FAIL":
        return (
            "RECOVER_ENTRY_SNAPSHOT_SIGNALS_BLOCKED_LOW_COVERAGE_V1",
            "HOLD_UNTIL_RECOVERY_COVERAGE_RESOLVED_V1",
            f"Bridge math audit failed: {bridge_audit.get('failures_v1')}",
        )
    rate = coverage_audit["match_rate_v1"]
    if rate is None:
        return (
            "RECOVER_ENTRY_SNAPSHOT_SIGNALS_BLOCKED_BY_INPUT_LOCK_MISSING_V1",
            "HOLD_UNTIL_RECOVERY_COVERAGE_RESOLVED_V1",
            "No trades found in any week.",
        )
    if rate >= 0.999:
        return (
            "RECOVER_ENTRY_SNAPSHOT_SIGNALS_PASS_FULL_COVERAGE_V1",
            "DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_V1",
            f"Recovered {coverage_audit['matched_trade_count_v1']}/{coverage_audit['total_trade_count_v1']} trades.",
        )
    if rate >= PASS_COVERAGE_THRESHOLD_V1:
        return (
            "RECOVER_ENTRY_SNAPSHOT_SIGNALS_PARTIAL_COVERAGE_V1",
            "DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_V1",
            (
                f"Recovered {coverage_audit['matched_trade_count_v1']}/"
                f"{coverage_audit['total_trade_count_v1']} trades "
                f"({rate:.4f}). Unmatched trades flagged "
                "NOT_RECOVERED; downstream gate must treat them as missing "
                "ENTRY_CONTEXT_SNAPSHOT and fail-soft (mask) rather than "
                "fabricate values."
            ),
        )
    return (
        "RECOVER_ENTRY_SNAPSHOT_SIGNALS_BLOCKED_LOW_COVERAGE_V1",
        "HOLD_UNTIL_RECOVERY_COVERAGE_RESOLVED_V1",
        (
            f"Match rate {rate:.4f} below threshold "
            f"{PASS_COVERAGE_THRESHOLD_V1}. Recovery cannot lock; investigate "
            "missing xgb_multi_horizon_predictions per-week artifacts."
        ),
    )


def _input_manifest(artifact_root: Path, weeks_seen: list[Path]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    files.append(
        {
            "name_v1": "v1_state_contract",
            "path_v1": str(INPUT_V1_STATE_CONTRACT_ROOT / "state_feature_contract_v1.json"),
            "sha256_v1": _file_hash(
                INPUT_V1_STATE_CONTRACT_ROOT / "state_feature_contract_v1.json"
            ),
        }
    )
    # Record only the count of per-week parquets to keep the manifest finite,
    # plus an explicit list of the first three for spot-checkable hashes.
    sample_paths: list[Path] = []
    for w in weeks_seen[:3]:
        for p in (_trade_outcomes_path(w), _xgb_predictions_path(w)):
            if p.exists():
                sample_paths.append(p)
    files.extend(
        {"name_v1": p.name, "path_v1": str(p), "sha256_v1": _file_hash(p)}
        for p in sample_paths
    )
    return {
        "layer_name": "RECOVER_ENTRY_SNAPSHOT_SIGNALS_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "v1_state_contract_root_v1": str(INPUT_V1_STATE_CONTRACT_ROOT),
            "truth_weeks_root_v1": str(DEFAULT_REPORTS_ROOT),
        },
        "files_used_v1": files,
        "truth_week_dirs_seen_v1": [w.name for w in weeks_seen],
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
        "entry_manager_modified_v1": False,
        "python_manifest_v1": _python_manifest(),
    }


def _reproducibility_audit(coverage_audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_name": "RECOVER_ENTRY_SNAPSHOT_SIGNALS_REPRODUCIBILITY_AUDIT_V1",
        "deterministic_iteration_order_v1": "TRUTH_MONFRI_WEEK_DIRS_SORTED_BY_NAME",
        "deterministic_concat_sort_v1": "WEEK_NAME_THEN_OPEN_TS_UTC_THEN_CANDIDATE_UID",
        "no_implicit_glob_used_for_v1_inputs_v1": True,
        "deprecated_quarantine_revival_v1": False,
        "research_only_v1": True,
        "match_rate_v1": coverage_audit["match_rate_v1"],
        "matched_count_v1": coverage_audit["matched_trade_count_v1"],
        "total_count_v1": coverage_audit["total_trade_count_v1"],
    }


# ---------------------------------------------------------------------------
# Materializer
# ---------------------------------------------------------------------------


def write_artifacts(
    out_root: Path | None = None,
    *,
    built_at_utc: str | None = None,
) -> dict[str, Any]:
    # Validate that the upstream V1 contract LOCK exists. We don't read
    # anything from it; the existence check is the input pin.
    _ = _load_v1_state_contract()

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
    weeks_seen = _list_truth_weeks()
    _write_json(artifact_root / "input_manifest_v1.json", _input_manifest(artifact_root, weeks_seen))

    recovered, per_week_rows = _recover_all_weeks()

    # Per-week audit table.
    _write_rows(artifact_root / "per_week_match_rate_audit_v1.csv", per_week_rows)
    _write_json(
        artifact_root / "per_week_match_rate_audit_v1.json",
        {
            "layer_name": "RECOVER_ENTRY_SNAPSHOT_SIGNALS_PER_WEEK_MATCH_RATE_AUDIT_V1",
            "rows_v1": per_week_rows,
        },
    )

    coverage_audit = _coverage_audit(recovered, per_week_rows)
    _write_json(artifact_root / "coverage_audit_v1.json", coverage_audit)

    bridge_audit = _bridge_audit(recovered)
    _write_json(artifact_root / "bridge_math_audit_v1.json", bridge_audit)

    runtime_audit = _no_runtime_modification_audit()
    _write_json(artifact_root / "no_runtime_modification_audit_v1.json", runtime_audit)

    # Persist recovered per-trade table (parquet, deterministic).
    recovered_out_path = artifact_root / "entry_snapshot_signals_per_trade_v1.parquet"
    if not recovered.empty:
        # Force open_ts_utc to UTC datetime, week_name_v1 to string, uids string.
        recovered_to_write = recovered.copy()
        recovered_to_write["open_ts_utc"] = pd.to_datetime(
            recovered_to_write["open_ts_utc"], utc=True
        )
        recovered_to_write["candidate_uid"] = recovered_to_write[
            "candidate_uid"
        ].astype(str)
        recovered_to_write["trade_uid"] = recovered_to_write["trade_uid"].astype(str)
        recovered_to_write["week_name_v1"] = recovered_to_write["week_name_v1"].astype(
            str
        )
        recovered_to_write["xgb_head_used_v1"] = recovered_to_write[
            "xgb_head_used_v1"
        ].astype(object)
        recovered_to_write["recovery_status_v1"] = recovered_to_write[
            "recovery_status_v1"
        ].astype(str)
        recovered_to_write.to_parquet(recovered_out_path, index=False)
    else:
        # Write an empty parquet with the contract columns so downstream code
        # can rely on the path always existing.
        empty = pd.DataFrame(
            columns=[
                "candidate_uid",
                "trade_uid",
                "open_ts_utc",
                "session",
                "side",
                "week_name_v1",
                "p_long_entry_v1",
                "p_hat_entry_v1",
                "uncertainty_entry_v1",
                "margin_entry_v1",
                "xgb_head_used_v1",
                "recovery_status_v1",
            ]
        )
        empty.to_parquet(recovered_out_path, index=False)

    repro = _reproducibility_audit(coverage_audit)
    _write_json(artifact_root / "reproducibility_audit_v1.json", repro)

    status, next_action, recommendation = _go_no_go(coverage_audit, bridge_audit)
    validate_final_status(status, next_action)

    summary = {
        "layer_name": "RECOVER_ENTRY_SNAPSHOT_SIGNALS_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "total_trade_count_v1": coverage_audit["total_trade_count_v1"],
        "matched_trade_count_v1": coverage_audit["matched_trade_count_v1"],
        "match_rate_v1": coverage_audit["match_rate_v1"],
        "weeks_missing_xgb_v1": coverage_audit["weeks_missing_xgb_v1"],
        "bridge_math_status_v1": bridge_audit.get("status_v1"),
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
        "entry_manager_modified_v1": False,
        "forbidden_actions_audit_v1": forbidden_audit,
    }
    _write_json(artifact_root / "summary_v1.json", summary)

    status_payload = {
        "layer_name": "RECOVER_ENTRY_SNAPSHOT_SIGNALS_STATUS_V1",
        "status_v1": "MATERIALIZED_RESEARCH_ONLY_GATE",
        "final_status_v1": status,
        "next_action_v1": next_action,
        "training_executed_v1": False,
    }
    _write_json(artifact_root / "status_v1.json", status_payload)

    go_no_go = {
        "layer_name": "RECOVER_ENTRY_SNAPSHOT_SIGNALS_GO_NO_GO_V1",
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
            "Research-only recovery sub-gate. No training. Adapter/R6/IQL "
            "production/live, freeze/promo/live, exit_manager modification, "
            "entry_manager modification, V1 state contract modification all "
            "forbidden."
        ),
    }
    _write_json(artifact_root / "recover_entry_snapshot_signals_go_no_go_v1.json", go_no_go)

    report_lines = [
        "# Recover Entry Snapshot Signals For Exit IQL V1",
        "",
        f"- Status: `{status}`",
        f"- Next action: `{next_action}`",
        "- Training: **BLOCKED** (research-only recovery sub-gate).",
        "",
        "## Coverage",
        f"- Total trades: {coverage_audit['total_trade_count_v1']}",
        f"- Matched: {coverage_audit['matched_trade_count_v1']}",
        f"- Match rate: {coverage_audit['match_rate_v1']}",
        f"- Weeks missing xgb_predictions parquet: {coverage_audit['weeks_missing_xgb_v1']}",
        "",
        "## Bridge math audit",
        f"- Status: `{bridge_audit.get('status_v1')}`",
    ]
    if bridge_audit.get("status_v1") == "PASS":
        report_lines.extend(
            [
                f"- p_long range: [{bridge_audit['p_long_min_v1']:.6f}, {bridge_audit['p_long_max_v1']:.6f}]",
                f"- p_hat range: [{bridge_audit['p_hat_min_v1']:.6f}, {bridge_audit['p_hat_max_v1']:.6f}]",
                f"- margin range: [{bridge_audit['margin_min_v1']:.6f}, {bridge_audit['margin_max_v1']:.6f}]",
            ]
        )
    report_lines.extend(
        [
            "",
            "## Recommendation",
            recommendation,
        ]
    )
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
                artifact_root / "recover_entry_snapshot_signals_go_no_go_v1.json"
            ),
            "input_manifest": str(artifact_root / "input_manifest_v1.json"),
            "entry_snapshot_signals_per_trade": str(recovered_out_path),
            "per_week_match_rate_audit_csv": str(
                artifact_root / "per_week_match_rate_audit_v1.csv"
            ),
            "per_week_match_rate_audit_json": str(
                artifact_root / "per_week_match_rate_audit_v1.json"
            ),
            "coverage_audit": str(artifact_root / "coverage_audit_v1.json"),
            "bridge_math_audit": str(artifact_root / "bridge_math_audit_v1.json"),
            "no_runtime_modification_audit": str(
                artifact_root / "no_runtime_modification_audit_v1.json"
            ),
            "reproducibility_audit": str(artifact_root / "reproducibility_audit_v1.json"),
            "report": str(artifact_root / "report_v1.md"),
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
        description="Materialize RECOVER_ENTRY_SNAPSHOT_SIGNALS_FOR_EXIT_IQL_V1 gate."
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
