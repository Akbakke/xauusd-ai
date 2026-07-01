#!/usr/bin/env python3
"""Materialize Entry candidate offline replay evidence.

This script consumes an explicit trade-level replay log and writes the
`replay_policy_metrics.csv` and `replay_policy_monthly.csv` artifacts required
by Entry replay-readiness. It does not run replay, train, promote, shadow, live,
or select implicit latest/legacy artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.features.entry_specialist_feature_groups_v1 import required_training_specialists_for_mode
from gx1.scripts.verify_entry_foundation_state_v1 import REPORTS_ROOT


DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_candidate_replay_20260628_v1"
DEFAULT_CANDIDATE_BUNDLE_AUDIT = (
    REPORTS_ROOT / "entry_candidate_bundle_audit_20260628_v1/ENTRY_FOUNDATION_SMOKE_BUNDLE_AUDIT_latest.json"
)
DEFAULT_SELECTIVE_EDGE_SUMMARY = REPORTS_ROOT / "entry_candidate_selective_edge_20260628_v1/summary.json"
CONTRACT_INPUT_DIMS = {
    "foundation_seq146": 146,
    "challenger_seq215": 215,
    "smart_seq520_candidate": 520,
}

IQL_TRANSITION_REQUIRED_COLUMNS = (
    "entry_time",
    "policy_id",
    "session",
    "side",
    "score",
    "p_long",
    "p_short",
    "p_flat",
    "net_pnl_bps",
    "mfe_bps",
    "mae_bps",
    "held_bars",
)

IQL_TRANSITION_NUMERIC_COLUMNS = (
    "score",
    "p_long",
    "p_short",
    "p_flat",
    "net_pnl_bps",
    "mfe_bps",
    "mae_bps",
    "held_bars",
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        out = float(obj)
        return out if np.isfinite(out) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing replay trades file: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise RuntimeError(f"unsupported replay trades extension: {path.suffix}")


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_specialists(contract_mode: str) -> list[str]:
    try:
        return sorted(required_training_specialists_for_mode(contract_mode))
    except ValueError:
        return []


def _candidate_bundle_specialist_contract(candidate_audit: dict[str, Any], contract_mode: str) -> dict[str, Any]:
    bundle = candidate_audit.get("bundle_summary") if isinstance(candidate_audit.get("bundle_summary"), dict) else {}
    specialist_contract = (
        candidate_audit.get("bundle_specialist_model_contract")
        if isinstance(candidate_audit.get("bundle_specialist_model_contract"), dict)
        else {}
    )
    expected = _expected_specialists(contract_mode)
    required = sorted(str(x) for x in candidate_audit.get("required_training_specialists", []) if str(x))
    groups = sorted(str(x) for x in bundle.get("specialist_groups", []) if str(x))
    failures: list[str] = []

    if not expected:
        failures.append(f"unknown specialist contract mode: {contract_mode}")
    if required != expected:
        failures.append(f"candidate audit required specialist set mismatch: observed={required} expected={expected}")
    if groups != expected:
        failures.append(f"candidate bundle specialist group mismatch: observed={groups} expected={expected}")
    if not bool(bundle.get("specialist_fusion_enabled")):
        failures.append("candidate bundle summary specialist_fusion_enabled is not true")
    for flag in (
        "specialist_model_contract_valid",
        "specialist_model_contract_set_exact",
        "specialist_model_contract_owned_objectives_match",
        "specialist_model_contract_support_heads_match",
        "specialist_model_contract_signal_families_match",
        "specialist_model_contract_model_roles_match",
    ):
        if not bool(bundle.get(flag)):
            failures.append(f"candidate bundle summary {flag} is not true")
    for flag in (
        "valid",
        "set_exact",
        "owned_objectives_match",
        "support_heads_match",
        "signal_families_match",
        "model_roles_match",
    ):
        if not bool(specialist_contract.get(flag)):
            failures.append(f"candidate bundle specialist contract {flag} is not true")
    if specialist_contract.get("failures"):
        failures.append(f"candidate bundle specialist contract failures: {specialist_contract.get('failures')}")
    if contract_mode != "foundation_seq146":
        for required_name in ("chart_geometry_encoder", "price_action_candle_encoder"):
            if required_name not in groups:
                failures.append(f"candidate {contract_mode} bundle missing specialist group: {required_name}")
            if required_name not in required:
                failures.append(f"candidate {contract_mode} audit missing required specialist: {required_name}")

    return {
        "ready": not failures,
        "contract_mode": contract_mode,
        "expected_specialists": expected,
        "required_training_specialists": required,
        "bundle_specialist_groups": groups,
        "bundle_specialist_model_contract": specialist_contract,
        "failures": failures,
    }


def _selective_specialist_snapshot_checks(snapshot: dict[str, Any], contract_mode: str, *, label: str) -> list[str]:
    expected = _expected_specialists(contract_mode)
    expected_dim = CONTRACT_INPUT_DIMS.get(contract_mode)
    failures: list[str] = []
    if not snapshot:
        return [f"{label} missing specialist contract snapshot"]
    if snapshot.get("failures"):
        failures.append(f"{label} specialist contract snapshot failures: {snapshot.get('failures')}")
    if str(snapshot.get("requested_contract_mode") or "") != contract_mode:
        failures.append(
            f"{label} specialist contract mode mismatch: "
            f"{snapshot.get('requested_contract_mode')} != {contract_mode}"
        )
    if expected_dim is not None and int(snapshot.get("expected_signal_dim") or 0) != int(expected_dim):
        failures.append(
            f"{label} expected signal dim mismatch: {snapshot.get('expected_signal_dim')} != {expected_dim}"
        )
    if expected_dim is not None and int(snapshot.get("bundle_seq_input_dim") or 0) != int(expected_dim):
        failures.append(
            f"{label} seq_input_dim mismatch: {snapshot.get('bundle_seq_input_dim')} != {expected_dim}"
        )
    if expected_dim is not None and int(snapshot.get("bundle_snap_input_dim") or 0) != int(expected_dim):
        failures.append(
            f"{label} snap_input_dim mismatch: {snapshot.get('bundle_snap_input_dim')} != {expected_dim}"
        )
    if sorted(str(x) for x in snapshot.get("expected_specialists", []) if str(x)) != expected:
        failures.append(f"{label} expected specialist set mismatch")
    if sorted(str(x) for x in snapshot.get("observed_specialists", []) if str(x)) != expected:
        failures.append(f"{label} observed specialist set mismatch")
    if not bool(snapshot.get("specialist_fusion_enabled")):
        failures.append(f"{label} specialist fusion is not enabled")
    if not bool(snapshot.get("required_specialists_exact")):
        failures.append(f"{label} required specialist set is not exact")
    for flag in (
        "specialist_model_contract_valid",
        "specialist_model_contract_set_exact",
        "specialist_model_contract_owned_objectives_match",
        "specialist_model_contract_signal_families_match",
        "specialist_model_contract_support_heads_match",
        "specialist_model_contract_model_roles_match",
    ):
        if not bool(snapshot.get(flag)):
            failures.append(f"{label} {flag} is not true")
    if contract_mode != "foundation_seq146":
        if not bool(snapshot.get("chart_geometry_present")):
            failures.append(f"{label} missing chart_geometry_encoder")
        if not bool(snapshot.get("price_action_candle_present")):
            failures.append(f"{label} missing price_action_candle_encoder")
    return failures


def _selective_edge_specialist_contract(selective_summary: dict[str, Any], contract_mode: str) -> dict[str, Any]:
    candidate_snapshot = (
        selective_summary.get("bundle_specialist_contract")
        if isinstance(selective_summary.get("bundle_specialist_contract"), dict)
        else {}
    )
    no_xgb_snapshot = (
        selective_summary.get("no_xgb_bundle_specialist_contract")
        if isinstance(selective_summary.get("no_xgb_bundle_specialist_contract"), dict)
        else {}
    )
    no_xgb_bundle_dir = str(selective_summary.get("no_xgb_bundle_dir") or "")
    failures = _selective_specialist_snapshot_checks(candidate_snapshot, contract_mode, label="selective-edge candidate")
    if no_xgb_bundle_dir:
        failures.extend(
            _selective_specialist_snapshot_checks(no_xgb_snapshot, contract_mode, label="selective-edge no-XGB")
        )
    return {
        "ready": not failures,
        "contract_mode": contract_mode,
        "candidate_bundle_specialist_contract": candidate_snapshot,
        "no_xgb_bundle_specialist_contract": no_xgb_snapshot,
        "no_xgb_bundle_dir": no_xgb_bundle_dir,
        "failures": failures,
    }


def _identity_contract(
    *,
    candidate_bundle_audit_path: Path,
    selective_edge_summary_path: Path,
    require_identity_artifacts: bool,
    requested_contract_mode: str | None = None,
) -> dict[str, Any]:
    candidate_audit = _read_json_if_exists(candidate_bundle_audit_path)
    selective_summary = _read_json_if_exists(selective_edge_summary_path)
    bundle = candidate_audit.get("bundle_summary") if isinstance(candidate_audit.get("bundle_summary"), dict) else {}
    contract_mode = str(
        requested_contract_mode
        or candidate_audit.get("specialist_contract_mode")
        or candidate_audit.get("contract_mode")
        or bundle.get("specialist_contract_mode")
        or bundle.get("contract_mode")
        or bundle.get("audit_contract_mode")
        or ""
    ).strip()
    if not contract_mode:
        contract_mode = "foundation_seq146"
    selective_contract_mode = str(selective_summary.get("contract_mode") or "foundation_seq146").strip()
    expected_input_dim = CONTRACT_INPUT_DIMS.get(contract_mode)
    bundle_seq_input_dim = int(bundle.get("seq_input_dim") or 0)
    bundle_snap_input_dim = int(bundle.get("snap_input_dim") or 0)
    selective_seq_input_dim = int(selective_summary.get("bundle_seq_input_dim") or 0)
    selective_snap_input_dim = int(selective_summary.get("bundle_snap_input_dim") or 0)
    failures: list[str] = []
    if require_identity_artifacts and not candidate_bundle_audit_path.exists():
        failures.append(f"missing candidate bundle audit: {candidate_bundle_audit_path}")
    if require_identity_artifacts and not selective_edge_summary_path.exists():
        failures.append(f"missing selective-edge summary: {selective_edge_summary_path}")
    if candidate_audit and str(candidate_audit.get("decision")) != "PASS":
        failures.append(f"candidate bundle audit decision is not PASS: {candidate_audit.get('decision')}")
    if selective_summary and str(selective_summary.get("decision")) != "PASS":
        failures.append(f"selective-edge summary decision is not PASS: {selective_summary.get('decision')}")

    candidate_bundle_dir = str(candidate_audit.get("bundle_dir") or "")
    selective_bundle_dir = str(selective_summary.get("bundle_dir") or "")
    if require_identity_artifacts and not candidate_bundle_dir:
        failures.append("candidate bundle audit does not declare bundle_dir")
    if require_identity_artifacts and not selective_bundle_dir:
        failures.append("selective-edge summary does not declare bundle_dir")
    if candidate_bundle_dir and selective_bundle_dir and candidate_bundle_dir != selective_bundle_dir:
        failures.append(
            "selective-edge bundle_dir does not match candidate bundle audit: "
            f"{selective_bundle_dir} != {candidate_bundle_dir}"
        )
    if selective_contract_mode != contract_mode:
        failures.append(
            "selective-edge contract_mode does not match candidate bundle audit: "
            f"{selective_contract_mode} != {contract_mode}"
        )
    if expected_input_dim is None:
        failures.append(f"unknown replay evidence contract_mode: {contract_mode}")
    elif (
        bundle_seq_input_dim
        and bundle_snap_input_dim
        and (bundle_seq_input_dim != expected_input_dim or bundle_snap_input_dim != expected_input_dim)
    ):
        failures.append(
            "candidate bundle input dimensions do not match contract mode: "
            f"seq={bundle_seq_input_dim} snap={bundle_snap_input_dim} expected={expected_input_dim}"
        )
    if expected_input_dim is not None and (
        selective_seq_input_dim
        and selective_snap_input_dim
        and (selective_seq_input_dim != expected_input_dim or selective_snap_input_dim != expected_input_dim)
    ):
        failures.append(
            "selective-edge input dimensions do not match contract mode: "
            f"seq={selective_seq_input_dim} snap={selective_snap_input_dim} expected={expected_input_dim}"
        )

    candidate_specialist_contract = _candidate_bundle_specialist_contract(candidate_audit, contract_mode)
    selective_specialist_contract = _selective_edge_specialist_contract(selective_summary, contract_mode)
    failures.extend(candidate_specialist_contract["failures"])
    failures.extend(selective_specialist_contract["failures"])

    return {
        "ready": not failures,
        "contract_mode": contract_mode,
        "selective_edge_contract_mode": selective_contract_mode,
        "expected_input_dim": expected_input_dim,
        "candidate_bundle_audit_sha256": _sha256_file(candidate_bundle_audit_path)
        if candidate_bundle_audit_path.exists()
        else "",
        "selective_edge_summary_sha256": _sha256_file(selective_edge_summary_path)
        if selective_edge_summary_path.exists()
        else "",
        "candidate_bundle_seq_input_dim": bundle_seq_input_dim,
        "candidate_bundle_snap_input_dim": bundle_snap_input_dim,
        "selective_edge_seq_input_dim": selective_seq_input_dim,
        "selective_edge_snap_input_dim": selective_snap_input_dim,
        "candidate_bundle_audit_json": str(candidate_bundle_audit_path),
        "selective_edge_summary_json": str(selective_edge_summary_path),
        "candidate_bundle_dir": candidate_bundle_dir,
        "selective_edge_bundle_dir": selective_bundle_dir,
        "no_xgb_bundle_dir": str(selective_summary.get("no_xgb_bundle_dir") or ""),
        "selective_edge_feature_mask_ablation": selective_summary.get("feature_mask_ablation")
        if isinstance(selective_summary.get("feature_mask_ablation"), dict)
        else {},
        "candidate_audit_decision": str(candidate_audit.get("decision") or ""),
        "selective_edge_decision": str(selective_summary.get("decision") or ""),
        "candidate_specialist_contract": candidate_specialist_contract,
        "selective_edge_specialist_contract": selective_specialist_contract,
        "require_identity_artifacts": bool(require_identity_artifacts),
        "failures": failures,
    }


def _first_present(frame: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _safe_mean(values: pd.Series) -> float | None:
    vals = pd.to_numeric(values, errors="coerce").to_numpy(np.float64)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else None


def _safe_percentile(values: pd.Series, q: float) -> float | None:
    vals = pd.to_numeric(values, errors="coerce").to_numpy(np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, q)) if vals.size else None


def _profit_factor(values: pd.Series) -> float | None:
    vals = pd.to_numeric(values, errors="coerce").to_numpy(np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    gains = float(vals[vals > 0.0].sum())
    losses = float(vals[vals < 0.0].sum())
    if losses == 0.0:
        return None
    return float(gains / abs(losses))


def _max_drawdown(values: pd.Series) -> tuple[float, float]:
    vals = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(np.float64)
    if vals.size == 0:
        return 0.0, 0.0
    equity = np.concatenate([[0.0], np.cumsum(vals)])
    dd = equity - np.maximum.accumulate(equity)
    signed = float(np.min(dd))
    return abs(signed), signed


def normalize_trades(raw: pd.DataFrame, *, policy_id: str, require_year: int | None, allow_non_2026: bool) -> tuple[pd.DataFrame, list[str]]:
    failures: list[str] = []
    if raw.empty:
        raise RuntimeError("replay trades input is empty")

    time_col = _first_present(raw, ["entry_time", "entry_ts", "time", "open_time", "timestamp", "decision_time"])
    pnl_col = _first_present(raw, ["net_pnl_bps", "realized_pnl_bps", "pnl_bps", "gross_pnl_bps"])
    if time_col is None:
        raise RuntimeError("replay trades input needs an entry time column")
    if pnl_col is None:
        raise RuntimeError("replay trades input needs a PnL bps column")

    out = raw.copy()
    out["entry_time"] = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    out["net_pnl_bps"] = pd.to_numeric(out[pnl_col], errors="coerce")
    if "gross_pnl_bps" not in out.columns:
        out["gross_pnl_bps"] = out["net_pnl_bps"]
    else:
        out["gross_pnl_bps"] = pd.to_numeric(out["gross_pnl_bps"], errors="coerce")

    out = out.dropna(subset=["entry_time", "net_pnl_bps"]).reset_index(drop=True)
    if out.empty:
        raise RuntimeError("replay trades input has no valid entry_time/net_pnl_bps rows")

    if "policy_id" not in out.columns:
        out["policy_id"] = str(policy_id)
    out["policy_id"] = out["policy_id"].fillna(str(policy_id)).astype(str)
    observed_policies = sorted({str(value).strip() for value in out["policy_id"].dropna().astype(str) if str(value).strip()})
    if observed_policies != [str(policy_id)]:
        failures.append(
            "replay trades policy_id mismatch: "
            f"expected={[str(policy_id)]} observed={observed_policies}"
        )
    if "fold" not in out.columns:
        out["fold"] = "2026"
    out["fold"] = out["fold"].fillna("2026").astype(str)
    if "side" not in out.columns:
        out["side"] = "UNKNOWN"
    out["side"] = out["side"].fillna("UNKNOWN").astype(str).str.upper()
    if "session" not in out.columns:
        out["session"] = "UNKNOWN"
    out["session"] = out["session"].fillna("UNKNOWN").astype(str).str.upper()
    if "score" not in out.columns:
        out["score"] = np.nan
    if "mfe_bps" not in out.columns:
        out["mfe_bps"] = np.nan
    if "mae_bps" not in out.columns:
        out["mae_bps"] = np.nan
    if "direction_correct" not in out.columns:
        label_col = _first_present(out, ["label", "y_direction", "target_direction"])
        if label_col is not None:
            out["direction_correct"] = out["side"].astype(str).str.upper() == out[label_col].astype(str).str.upper()
        else:
            out["direction_correct"] = np.nan
    out["direction_correct"] = pd.Series(out["direction_correct"]).map(
        lambda x: (str(x).strip().lower() == "true") if str(x).strip().lower() in {"true", "false"} else x
    )
    out["entry_day"] = out["entry_time"].dt.strftime("%Y-%m-%d")
    out["entry_month"] = out["entry_time"].dt.strftime("%Y-%m")
    if "tail_bucket" not in out.columns:
        pnl_values = pd.to_numeric(out["net_pnl_bps"], errors="coerce")
        p05 = float(pnl_values.quantile(0.05)) if len(pnl_values) else 0.0
        p10 = float(pnl_values.quantile(0.10)) if len(pnl_values) else 0.0
        out["tail_bucket"] = np.select(
            [pnl_values <= p05, pnl_values <= p10],
            ["tail_loss_p05", "tail_loss_p10"],
            default="normal",
        )
    if "bad_path_bucket" not in out.columns:
        if "bad_path_prob" in out.columns:
            bad_path = pd.to_numeric(out["bad_path_prob"], errors="coerce")
            p75 = float(bad_path.quantile(0.75)) if bad_path.notna().any() else 0.0
            p90 = float(bad_path.quantile(0.90)) if bad_path.notna().any() else 0.0
            out["bad_path_bucket"] = np.select(
                [bad_path >= p90, bad_path >= p75],
                ["bad_path_p90", "bad_path_p75"],
                default="normal",
            )
        else:
            out["bad_path_bucket"] = "unknown"

    if require_year is not None:
        years = set(int(x) for x in out["entry_time"].dt.year.dropna().astype(int).unique())
        if int(require_year) not in years:
            failures.append(f"no trades in required replay year {require_year}")
        if not allow_non_2026 and years != {int(require_year)}:
            failures.append(f"replay trades contain years outside {require_year}: {sorted(years)}")

    keep = [
        "fold",
        "policy_id",
        "session",
        "entry_day",
        "entry_month",
        "entry_time",
        "side",
        "direction_correct",
        "score",
        "gross_pnl_bps",
        "net_pnl_bps",
        "mfe_bps",
        "mae_bps",
    ]
    optional_prefixes = (
        "foundation_",
        "specialist_",
        "ctx_",
        "state_",
        "teacher_",
        "candidate_",
    )
    for optional in (
        "candidate_uid",
        "trade_uid",
        "exit_time",
        "exit_reason",
        "held_bars",
        "horizon_bars",
        "vol_regime",
        "tail_bucket",
        "bad_path_bucket",
        "threshold_top_frac",
        "score_threshold",
        "cost_stress_bps",
        "policy_config_hash",
        "p_long",
        "p_short",
        "p_flat",
        "path_quality_pred",
        "bad_path_prob",
        "tradable_prob",
    ):
        if optional in out.columns and optional not in keep:
            keep.append(optional)
    for col in out.columns:
        if any(str(col).startswith(prefix) for prefix in optional_prefixes) and col not in keep:
            keep.append(str(col))
    return out[[c for c in keep if c in out.columns]].copy(), failures


def _safe_value_counts(frame: pd.DataFrame, col: str) -> dict[str, int]:
    if col not in frame.columns:
        return {}
    return {
        str(k): int(v)
        for k, v in frame[col].fillna("UNKNOWN").astype(str).str.upper().value_counts(dropna=False).to_dict().items()
    }


def audit_iql_transition_trades(trades: pd.DataFrame) -> dict[str, Any]:
    missing = [col for col in IQL_TRANSITION_REQUIRED_COLUMNS if col not in trades.columns]
    failures: list[str] = []
    if missing:
        failures.append(f"IQL transition trade log missing required columns: {missing}")

    numeric_status: dict[str, dict[str, Any]] = {}
    for col in IQL_TRANSITION_NUMERIC_COLUMNS:
        if col not in trades.columns:
            continue
        values = pd.to_numeric(trades[col], errors="coerce")
        arr = values.to_numpy(dtype=np.float64)
        finite = bool(np.isfinite(arr).all()) if len(arr) else False
        null_count = int(values.isna().sum())
        numeric_status[col] = {
            "finite": finite,
            "null_count": null_count,
            "min": float(values.min()) if len(values) and values.notna().any() else None,
            "max": float(values.max()) if len(values) and values.notna().any() else None,
        }
        if not finite or null_count > 0:
            failures.append(f"IQL transition numeric column not fully finite: {col}")

    for prob_col in ("p_long", "p_short", "p_flat"):
        if prob_col in trades.columns:
            values = pd.to_numeric(trades[prob_col], errors="coerce")
            if bool(((values < 0.0) | (values > 1.0)).any()):
                failures.append(f"IQL transition probability column outside [0,1]: {prob_col}")

    if {"p_long", "p_short", "p_flat"}.issubset(trades.columns):
        prob_sum = (
            pd.to_numeric(trades["p_long"], errors="coerce")
            + pd.to_numeric(trades["p_short"], errors="coerce")
            + pd.to_numeric(trades["p_flat"], errors="coerce")
        )
        probability_sum_max_abs_error = float((prob_sum - 1.0).abs().max()) if len(prob_sum) else float("inf")
        if not np.isfinite(probability_sum_max_abs_error) or probability_sum_max_abs_error > 0.05:
            failures.append(
                "IQL transition probability sum drifts from 1.0: "
                f"max_abs_error={probability_sum_max_abs_error}"
            )
    else:
        probability_sum_max_abs_error = None

    session_counts = _safe_value_counts(trades, "session")
    if session_counts and set(session_counts) <= {"UNKNOWN"}:
        failures.append("IQL transition session state is all UNKNOWN")

    side_counts = _safe_value_counts(trades, "side")
    valid_side_rows = 0
    if "side" in trades.columns:
        valid_side_rows = int(trades["side"].astype(str).str.upper().isin(["LONG", "SHORT"]).sum())
        if valid_side_rows <= 0:
            failures.append("IQL transition action side has no LONG/SHORT rows")

    return {
        "ready": not failures,
        "required_columns": list(IQL_TRANSITION_REQUIRED_COLUMNS),
        "missing_columns": missing,
        "numeric_status": numeric_status,
        "probability_sum_max_abs_error": probability_sum_max_abs_error,
        "session_counts": session_counts,
        "side_counts": side_counts,
        "valid_side_rows": valid_side_rows,
        "failures": failures,
    }


def _metrics_row(scope: str, fold: str, policy_id: str, frame: pd.DataFrame) -> dict[str, Any]:
    pnl = pd.to_numeric(frame["net_pnl_bps"], errors="coerce")
    gross = pd.to_numeric(frame["gross_pnl_bps"], errors="coerce")
    dd_abs, dd_signed = _max_drawdown(pnl)
    direction = pd.to_numeric(frame.get("direction_correct", pd.Series(dtype=float)), errors="coerce")
    return {
        "scope": scope,
        "fold": fold,
        "policy_id": policy_id,
        "n_trades": int(len(frame)),
        "n_days": int(frame["entry_day"].nunique()),
        "n_months": int(frame["entry_month"].nunique()),
        "net_sum_bps": float(pnl.sum()),
        "net_mean_bps": _safe_mean(pnl),
        "net_median_bps": _safe_percentile(pnl, 50),
        "net_p10_bps": _safe_percentile(pnl, 10),
        "net_p90_bps": _safe_percentile(pnl, 90),
        "gross_mean_bps": _safe_mean(gross),
        "win_rate": float((pnl > 0.0).mean()) if len(pnl) else None,
        "profit_factor": _profit_factor(pnl),
        "max_win_bps": float(pnl.max()) if len(pnl) else None,
        "max_loss_bps": float(pnl.min()) if len(pnl) else None,
        "max_drawdown_bps": dd_abs,
        "max_drawdown_signed_bps": dd_signed,
        "mean_score": _safe_mean(frame["score"]),
        "mean_mfe_bps": _safe_mean(frame["mfe_bps"]),
        "mean_mae_bps": _safe_mean(frame["mae_bps"]),
        "long_rate": float((frame["side"].astype(str).str.upper() == "LONG").mean()),
        "short_rate": float((frame["side"].astype(str).str.upper() == "SHORT").mean()),
        "direction_precision": _safe_mean(direction),
        "avg_trades_per_day": float(len(frame) / max(frame["entry_day"].nunique(), 1)),
    }


def build_replay_tables(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    for (policy_id, fold), frame in trades.groupby(["policy_id", "fold"], sort=True):
        metric_rows.append(_metrics_row("fold", str(fold), str(policy_id), frame))
    for policy_id, frame in trades.groupby("policy_id", sort=True):
        metric_rows.append(_metrics_row("aggregate", "ALL", str(policy_id), frame))
    metrics = pd.DataFrame(metric_rows)

    daily = (
        trades.groupby(["policy_id", "entry_day"], as_index=False)
        .agg(
            n_trades=("net_pnl_bps", "size"),
            net_sum_bps=("net_pnl_bps", "sum"),
            net_mean_bps=("net_pnl_bps", "mean"),
            wins=("net_pnl_bps", lambda s: int((s > 0.0).sum())),
        )
    )
    daily["win_rate"] = daily["wins"] / daily["n_trades"].clip(lower=1)

    monthly = (
        trades.groupby(["policy_id", "entry_month"], as_index=False)
        .agg(
            n_trades=("net_pnl_bps", "size"),
            net_sum_bps=("net_pnl_bps", "sum"),
            net_mean_bps=("net_pnl_bps", "mean"),
            wins=("net_pnl_bps", lambda s: int((s > 0.0).sum())),
        )
    )
    monthly["month"] = monthly["entry_month"]
    monthly["win_rate"] = monthly["wins"] / monthly["n_trades"].clip(lower=1)
    return metrics, daily, monthly


def _slice_metrics_row(slice_dimension: str, slice_value: str, policy_id: str, frame: pd.DataFrame) -> dict[str, Any]:
    row = _metrics_row("slice", f"{slice_dimension}={slice_value}", policy_id, frame)
    row["slice_family"] = slice_dimension
    row["slice_dimension"] = slice_dimension
    row["slice_value"] = slice_value
    if "bad_path_prob" in frame.columns:
        bad_path = pd.to_numeric(frame["bad_path_prob"], errors="coerce")
        row["mean_bad_path_prob"] = _safe_mean(bad_path)
        row["bad_path_rate"] = float((bad_path >= 0.5).mean()) if len(bad_path) else None
    if "path_quality_pred" in frame.columns:
        row["mean_path_quality_pred"] = _safe_mean(frame["path_quality_pred"])
        row["path_quality_p10"] = _safe_percentile(frame["path_quality_pred"], 10)
    if "mae_bps" in frame.columns:
        row["mae_p95_bps"] = _safe_percentile(frame["mae_bps"], 95)
    return row


def build_replay_slices(trades: pd.DataFrame) -> pd.DataFrame:
    dimensions = [
        ("session", "session"),
        ("regime", "vol_regime"),
        ("direction", "side"),
        ("tail", "tail_bucket"),
        ("bad_path", "bad_path_bucket"),
    ]
    rows: list[dict[str, Any]] = []
    for slice_dimension, column in dimensions:
        if column not in trades.columns:
            continue
        work = trades.copy()
        work[column] = work[column].fillna("UNKNOWN").astype(str)
        for (policy_id, value), frame in work.groupby(["policy_id", column], sort=True):
            rows.append(_slice_metrics_row(slice_dimension, str(value), str(policy_id), frame))
    return pd.DataFrame(rows)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Candidate Replay Evidence",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Trades: `{report['n_trades']}`",
        f"- Out dir: `{report['out_dir']}`",
        f"- Promotion/shadow/live allowed: `{report['promotion_shadow_live_allowed']}`",
        "",
        "## Failures",
        "",
    ]
    if report["failures"]:
        lines.extend(f"- {failure}" for failure in report["failures"])
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    trades_path = Path(args.trades_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    candidate_bundle_audit_path = Path(args.candidate_bundle_audit_json).expanduser().resolve()
    selective_edge_summary_path = Path(args.selective_edge_summary_json).expanduser().resolve()
    ablation_id = str(args.ablation_id or "").strip()
    policy_id = str(args.policy_id or "").strip()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = _read_table(trades_path)
    identity = _identity_contract(
        candidate_bundle_audit_path=candidate_bundle_audit_path,
        selective_edge_summary_path=selective_edge_summary_path,
        require_identity_artifacts=bool(args.require_identity_artifacts),
        requested_contract_mode=getattr(args, "contract_mode", None),
    )
    require_year = None if args.require_year <= 0 else int(args.require_year)
    trades, failures = normalize_trades(
        raw,
        policy_id=str(args.policy_id),
        require_year=require_year,
        allow_non_2026=bool(args.allow_non_2026),
    )
    failures.extend(identity["failures"])
    metrics, daily, monthly = build_replay_tables(trades)
    slices = build_replay_slices(trades)
    iql_transition_audit = audit_iql_transition_trades(trades)
    if bool(args.require_iql_transition_fields):
        failures.extend(iql_transition_audit["failures"])

    best = metrics[metrics["scope"].astype(str).isin(["aggregate", "all", "ALL"])]
    if best.empty:
        failures.append("no aggregate replay metrics were produced")
    else:
        row = best.sort_values("net_sum_bps", ascending=False).iloc[0]
        if int(row.get("n_trades") or 0) <= 0:
            failures.append("aggregate replay metrics have zero trades")

    trades_out = out_dir / "replay_policy_trades.csv"
    metrics_out = out_dir / "replay_policy_metrics.csv"
    daily_out = out_dir / "replay_policy_daily.csv"
    monthly_out = out_dir / "replay_policy_monthly.csv"
    slices_out = out_dir / "replay_policy_slices.csv"
    summary_out = out_dir / "summary.json"
    manifest_out = out_dir / "REPLAY_EVIDENCE_MANIFEST.json"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_json = out_dir / f"ENTRY_CANDIDATE_REPLAY_EVIDENCE_{timestamp}.json"
    report_md = out_dir / f"ENTRY_CANDIDATE_REPLAY_EVIDENCE_{timestamp}.md"

    trades.to_csv(trades_out, index=False)
    metrics.to_csv(metrics_out, index=False)
    daily.to_csv(daily_out, index=False)
    monthly.to_csv(monthly_out, index=False)
    slices.to_csv(slices_out, index=False)
    artifact_hashes = {
        "replay_policy_trades.csv": _sha256_file(trades_out),
        "replay_policy_metrics.csv": _sha256_file(metrics_out),
        "replay_policy_daily.csv": _sha256_file(daily_out),
        "replay_policy_monthly.csv": _sha256_file(monthly_out),
        "replay_policy_slices.csv": _sha256_file(slices_out),
    }

    best_row = best.sort_values("net_sum_bps", ascending=False).iloc[0].to_dict() if not best.empty else {}
    report = {
        "schema_version": "entry_candidate_replay_evidence_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "PASS" if not failures else "FAIL",
        "contract_mode": identity["contract_mode"],
        "ablation_id": ablation_id,
        "policy_id": policy_id,
        "trades_path": str(trades_path),
        "out_dir": str(out_dir),
        "candidate_bundle_audit_json": str(candidate_bundle_audit_path),
        "selective_edge_summary_json": str(selective_edge_summary_path),
        "candidate_bundle_dir": identity["candidate_bundle_dir"],
        "no_xgb_bundle_dir": identity["no_xgb_bundle_dir"],
        "feature_mask_ablation": identity["selective_edge_feature_mask_ablation"],
        "replay_identity_contract": identity,
        "required_year": require_year,
        "n_trades": int(len(trades)),
        "policies": sorted(str(x) for x in trades["policy_id"].unique()),
        "best_aggregate_row": best_row,
        "trades_csv": str(trades_out),
        "metrics_csv": str(metrics_out),
        "daily_csv": str(daily_out),
        "monthly_csv": str(monthly_out),
        "slices_csv": str(slices_out),
        "summary_json": str(summary_out),
        "manifest_json": str(manifest_out),
        "artifact_hashes": artifact_hashes,
        "iql_transition_dataset_ready": bool(iql_transition_audit["ready"]),
        "iql_transition_contract": iql_transition_audit,
        "json_path": str(report_json),
        "md_path": str(report_md),
        "trainer_started": False,
        "replay_started": False,
        "promotion_shadow_live_allowed": False,
        "failures": failures,
    }
    summary_out.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    manifest_out.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(report_md, report)
    (out_dir / "ENTRY_CANDIDATE_REPLAY_EVIDENCE_latest.json").write_text(
        report_json.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out_dir / "ENTRY_CANDIDATE_REPLAY_EVIDENCE_latest.md").write_text(
        report_md.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "failures": failures,
                    "metrics_csv": str(metrics_out),
                    "monthly_csv": str(monthly_out),
                    "json_path": str(report_json),
                },
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
        )
    if args.fail_on_audit_fail and failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades-path", required=True)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--candidate-bundle-audit-json", default=str(DEFAULT_CANDIDATE_BUNDLE_AUDIT))
    ap.add_argument("--selective-edge-summary-json", default=str(DEFAULT_SELECTIVE_EDGE_SUMMARY))
    ap.add_argument("--policy-id", default="candidate_replay")
    ap.add_argument("--ablation-id", default="")
    ap.add_argument("--require-year", type=int, default=2026)
    ap.add_argument("--allow-non-2026", action="store_true")
    ap.add_argument("--contract-mode", choices=tuple(CONTRACT_INPUT_DIMS), default=None)
    ap.add_argument("--challenger-seq215", action="store_const", const="challenger_seq215", dest="contract_mode")
    ap.add_argument("--smart-seq520", action="store_const", const="smart_seq520_candidate", dest="contract_mode")
    ap.add_argument("--require-iql-transition-fields", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--require-identity-artifacts", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--fail-on-audit-fail", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
