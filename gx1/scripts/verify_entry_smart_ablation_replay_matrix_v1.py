#!/usr/bin/env python3
"""Verify report-only smart Entry ablation replay matrix evidence.

This gate only reads the approved ablation plan and already-materialized replay
evidence. It never starts training, replay, IQL distillation, shadow, live, or
promotion paths.
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

from gx1.scripts.materialize_entry_smart_ablation_replay_plan_gate_v1 import (
    SMART_VARIANT,
    build_required_ablation_plan,
)
from gx1.scripts.verify_entry_foundation_state_v1 import REPORTS_ROOT


PLAN_READY_DECISION = "READY_FOR_SMART_ABLATION_REPLAY_PLAN_REVIEW"
REQUIRED_ABLATION_COUNT = 14
CONTRACT_SEQ_SNAP_WIDTH = 520
REQUIRED_NO_XGB_BRIDGE_FIELDS = (
    "p_long",
    "p_short",
    "p_flat",
    "p_hat",
    "uncertainty_score",
    "margin_top1_top2",
    "entropy",
)
DEFAULT_PLAN_JSON = (
    REPORTS_ROOT
    / "entry_smart_ablation_replay_plan_gate_20260630_v1/ENTRY_SMART_ABLATION_REPLAY_PLAN_GATE_latest.json"
)
DEFAULT_REPLAY_ROOT = REPORTS_ROOT / "entry_smart_ablation_replay_matrix_20260701_v1"
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_smart_ablation_replay_matrix_gate_20260701_v1"

REQUIRED_REPLAY_FILES = (
    "REPLAY_EVIDENCE_MANIFEST.json",
    "replay_policy_metrics.csv",
    "replay_policy_monthly.csv",
    "replay_policy_trades.csv",
    "replay_policy_slices.csv",
)
REQUIRED_METRIC_COLUMNS = (
    "net_sum_bps",
    "profit_factor",
    "max_drawdown_bps",
    "n_trades",
)
REQUIRED_SLICE_SCOPE_ALIASES = {
    "session": ("session", "session_id", "session_name"),
    "regime": ("regime", "trend_regime", "vol_regime", "regime_id", "vol_regime_id"),
    "direction": ("direction", "side", "trade_side", "pred_direction"),
    "tail": ("tail", "tail_bucket", "tail_risk_bucket", "tail_event", "tail_loss_bucket"),
}
REQUIRED_SLICE_SCOPES = tuple(REQUIRED_SLICE_SCOPE_ALIASES)
SLICE_SCOPE_COLUMNS = ("slice_family", "slice_dimension", "dimension", "group", "scope")
SLICE_VALUE_COLUMNS = ("slice_value", "value")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _check(name: str, ok: bool, details: Any = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "details": details}


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_meta(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "sha256": _sha256_file(path),
    }


def _read_json_report(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, f"missing JSON artifact: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, f"failed to parse JSON artifact {path}: {exc}"
    if not isinstance(payload, dict):
        return {}, f"JSON artifact is not an object: {path}"
    return payload, None


def _read_csv_report(path: Path) -> tuple[pd.DataFrame, str | None]:
    if not path.exists():
        return pd.DataFrame(), f"missing CSV artifact: {path}"
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return pd.DataFrame(), f"failed to read CSV artifact {path}: {exc}"
    return frame, None


def _nested_dicts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    stack = [payload]
    while stack:
        current = stack.pop()
        out.append(current)
        for value in current.values():
            if isinstance(value, dict):
                stack.append(value)
    return out


def _dict_value(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _first_text_from_sources(payload: dict[str, Any], keys: tuple[str, ...]) -> str:
    for source in _nested_dicts(payload):
        for key in keys:
            value = source.get(key)
            text = str(value or "").strip()
            if text:
                return text
    return ""


def _manifest_variant(manifest: dict[str, Any]) -> str:
    return _first_text_from_sources(
        manifest,
        (
            "manifest_variant",
            "candidate_variant",
            "smart_variant",
            "contract_mode",
            "specialist_contract_mode",
            "replay_identity_contract_mode",
        ),
    )


def _manifest_ablation_id(manifest: dict[str, Any]) -> str:
    return _first_text_from_sources(
        manifest,
        (
            "ablation_id",
            "required_ablation_id",
            "replay_ablation_id",
            "smart_ablation_id",
        ),
    )


def _manifest_feature_mask(manifest: dict[str, Any]) -> dict[str, Any]:
    value = manifest.get("feature_mask_ablation")
    return value if isinstance(value, dict) else {}


def _canonical_arm_signature(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "ablation_id": str(row.get("ablation_id") or ""),
        "ablation_type": str(row.get("ablation_type") or ""),
        "manifest_variant": str(row.get("manifest_variant") or ""),
        "expected_seq_snap_width": int(row.get("expected_seq_snap_width") or 0),
        "xgb_bridge_mode": str(row.get("xgb_bridge_mode") or ""),
        "included_feature_blocks": list(row.get("included_feature_blocks") or []),
        "excluded_feature_blocks": list(row.get("excluded_feature_blocks") or []),
    }


def _manifest_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    value = manifest.get("replay_identity_contract")
    return value if isinstance(value, dict) else {}


def _policy_values(frame: pd.DataFrame) -> list[str]:
    if "policy_id" not in frame.columns:
        return []
    values = pd.Series(frame["policy_id"]).dropna().astype(str).str.strip()
    return sorted({value for value in values if value})


def _summary_path_from_manifest(manifest: dict[str, Any], identity: dict[str, Any]) -> Path | None:
    raw = manifest.get("selective_edge_summary_json") or identity.get("selective_edge_summary_json")
    return Path(str(raw)).expanduser() if raw else None


def _identity_expected_contract(
    *,
    plan: dict[str, Any],
    identity: dict[str, Any],
    manifest: dict[str, Any],
    selective_summary: dict[str, Any],
    selective_summary_path: Path | None,
    selective_summary_sha: str | None,
) -> dict[str, Any]:
    candidate_identity = _dict_value(plan, "candidate_identity")
    expected_bundle_dir = str(candidate_identity.get("candidate_bundle_dir") or "")
    expected_audit_json = str(candidate_identity.get("candidate_bundle_audit_json") or "")
    expected_audit_sha = _sha256_file(Path(expected_audit_json).expanduser()) if expected_audit_json else None
    expected_seq_dim = int(candidate_identity.get("seq_input_dim") or CONTRACT_SEQ_SNAP_WIDTH)
    expected_snap_dim = int(candidate_identity.get("snap_input_dim") or CONTRACT_SEQ_SNAP_WIDTH)
    expected_variant = str(candidate_identity.get("manifest_variant") or SMART_VARIANT)
    observed_summary_sha = str(identity.get("selective_edge_summary_sha256") or "")

    return {
        "candidate_identity": candidate_identity,
        "expected_bundle_dir": expected_bundle_dir,
        "expected_audit_json": expected_audit_json,
        "expected_audit_sha256": expected_audit_sha,
        "expected_seq_input_dim": expected_seq_dim,
        "expected_snap_input_dim": expected_snap_dim,
        "expected_variant": expected_variant,
        "manifest_contract_mode": str(manifest.get("contract_mode") or ""),
        "identity_contract_mode": str(identity.get("contract_mode") or ""),
        "selective_edge_contract_mode": str(identity.get("selective_edge_contract_mode") or ""),
        "summary_contract_mode": str(selective_summary.get("contract_mode") or ""),
        "candidate_bundle_dir": str(identity.get("candidate_bundle_dir") or manifest.get("candidate_bundle_dir") or ""),
        "candidate_bundle_audit_json": str(identity.get("candidate_bundle_audit_json") or manifest.get("candidate_bundle_audit_json") or ""),
        "candidate_bundle_audit_sha256": str(identity.get("candidate_bundle_audit_sha256") or ""),
        "candidate_bundle_seq_input_dim": identity.get("candidate_bundle_seq_input_dim"),
        "candidate_bundle_snap_input_dim": identity.get("candidate_bundle_snap_input_dim"),
        "expected_input_dim": identity.get("expected_input_dim"),
        "selective_edge_seq_input_dim": identity.get("selective_edge_seq_input_dim"),
        "selective_edge_snap_input_dim": identity.get("selective_edge_snap_input_dim"),
        "summary_bundle_seq_input_dim": selective_summary.get("bundle_seq_input_dim"),
        "summary_bundle_snap_input_dim": selective_summary.get("bundle_snap_input_dim"),
        "selective_edge_summary_json": str(selective_summary_path) if selective_summary_path else "",
        "selective_edge_summary_sha256": observed_summary_sha,
        "selective_edge_summary_observed_sha256": selective_summary_sha,
    }


def _valid_seq520_identity(contract: dict[str, Any]) -> bool:
    return (
        contract["manifest_contract_mode"] == SMART_VARIANT
        and contract["identity_contract_mode"] == SMART_VARIANT
        and contract["selective_edge_contract_mode"] == SMART_VARIANT
        and contract["summary_contract_mode"] == SMART_VARIANT
        and contract["candidate_bundle_dir"] == contract["expected_bundle_dir"]
        and bool(contract["candidate_bundle_audit_sha256"])
        and contract["candidate_bundle_audit_sha256"] == contract["expected_audit_sha256"]
        and int(contract["expected_input_dim"] or 0) == CONTRACT_SEQ_SNAP_WIDTH
        and int(contract["candidate_bundle_seq_input_dim"] or 0) == contract["expected_seq_input_dim"]
        and int(contract["candidate_bundle_snap_input_dim"] or 0) == contract["expected_snap_input_dim"]
        and int(contract["selective_edge_seq_input_dim"] or 0) == CONTRACT_SEQ_SNAP_WIDTH
        and int(contract["selective_edge_snap_input_dim"] or 0) == CONTRACT_SEQ_SNAP_WIDTH
        and int(contract["summary_bundle_seq_input_dim"] or 0) == CONTRACT_SEQ_SNAP_WIDTH
        and int(contract["summary_bundle_snap_input_dim"] or 0) == CONTRACT_SEQ_SNAP_WIDTH
        and bool(contract["selective_edge_summary_sha256"])
        and contract["selective_edge_summary_sha256"] == contract["selective_edge_summary_observed_sha256"]
    )


def _no_xgb_contract(selective_summary: dict[str, Any]) -> dict[str, Any]:
    no_xgb = _dict_value(selective_summary, "no_xgb_ablation")
    diagnostics = _dict_value(selective_summary, "no_xgb_ablation_diagnostics")
    bridge_contract = _dict_value(selective_summary, "input_bridge_contract")
    bridge_splits = _dict_value(bridge_contract, "splits")
    split_reviews: dict[str, dict[str, Any]] = {}
    for split in ("val", "test"):
        diag = _dict_value(_dict_value(diagnostics, "splits"), split)
        bridge = _dict_value(bridge_splits, split)
        split_reviews[split] = {
            "diagnostics": {
                "comparable": bool(diag.get("comparable")),
                "time_match": bool(diag.get("time_match")),
                "identical_predictions": bool(diag.get("identical_predictions")),
                "max_abs_prob_delta": diag.get("max_abs_prob_delta"),
                "max_abs_edge_score_delta": diag.get("max_abs_edge_score_delta"),
                "pred_direction_diff_count": diag.get("pred_direction_diff_count"),
                "trade_side_diff_count": diag.get("trade_side_diff_count"),
            },
            "bridge": {
                "neutral_xgb_bridge": bool(bridge.get("neutral_xgb_bridge")),
                "bridge_source": str(bridge.get("bridge_source") or ""),
                "bridge_fields": list(bridge.get("bridge_fields") or []),
                "seq_input_dim": bridge.get("seq_input_dim"),
                "snap_input_dim": bridge.get("snap_input_dim"),
                "field_count": len(bridge.get("fields") or []),
            },
        }
    return {
        "no_xgb_ablation": no_xgb,
        "diagnostics": diagnostics,
        "split_reviews": split_reviews,
    }


def _valid_no_xgb_contract(review: dict[str, Any]) -> bool:
    no_xgb = review["no_xgb_ablation"]
    fields = list(no_xgb.get("neutralized_fields") or [])
    values = list(no_xgb.get("neutral_values") or [])
    if (
        str(no_xgb.get("mode") or "") != "neutralize_signal_bridge"
        or bool(no_xgb.get("neutralize_signal_bridge")) is not True
        or bool(no_xgb.get("required")) is not True
        or fields != list(REQUIRED_NO_XGB_BRIDGE_FIELDS)
        or len(values) != len(REQUIRED_NO_XGB_BRIDGE_FIELDS)
    ):
        return False
    diagnostics = review["diagnostics"]
    if bool(diagnostics.get("available")) is not True:
        return False
    for split, split_review in review["split_reviews"].items():
        diag = split_review["diagnostics"]
        bridge = split_review["bridge"]
        if not (
            diag["comparable"]
            and diag["time_match"]
            and diag["identical_predictions"]
            and float(diag["max_abs_prob_delta"] or 0.0) == 0.0
            and float(diag["max_abs_edge_score_delta"] or 0.0) == 0.0
            and int(diag["pred_direction_diff_count"] or 0) == 0
            and int(diag["trade_side_diff_count"] or 0) == 0
            and bridge["neutral_xgb_bridge"]
            and bridge["bridge_source"] == "neutral_uniform_proba"
            and bridge["bridge_fields"] == list(REQUIRED_NO_XGB_BRIDGE_FIELDS)
            and int(bridge["seq_input_dim"] or 0) == CONTRACT_SEQ_SNAP_WIDTH
            and int(bridge["snap_input_dim"] or 0) == CONTRACT_SEQ_SNAP_WIDTH
            and int(bridge["field_count"] or 0) == CONTRACT_SEQ_SNAP_WIDTH
        ):
            return False
    return True


def _required_ablation_rows(plan: dict[str, Any]) -> list[dict[str, Any]]:
    required_plan = _dict_value(plan, "required_ablation_plan")
    rows = required_plan.get("required_ablations")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _plan_freshness_checks(plan_path: Path, plan: dict[str, Any], plan_error: str | None) -> list[dict[str, Any]]:
    required_plan = _dict_value(plan, "required_ablation_plan")
    side_effects_started = _dict_value(plan, "side_effects_started")
    rows = _required_ablation_rows(plan)
    row_ids = [str(row.get("ablation_id") or "") for row in rows]
    duplicate_ids = sorted({ablation_id for ablation_id in row_ids if row_ids.count(ablation_id) > 1})
    canonical_plan = build_required_ablation_plan()
    canonical_rows = canonical_plan.get("required_ablations", [])
    canonical_by_id = {
        str(row.get("ablation_id") or ""): _canonical_arm_signature(row)
        for row in canonical_rows
        if isinstance(row, dict)
    }
    plan_by_id = {
        str(row.get("ablation_id") or ""): _canonical_arm_signature(row)
        for row in rows
        if isinstance(row, dict)
    }
    missing_canonical = sorted(set(canonical_by_id) - set(plan_by_id))
    unexpected_plan_ids = sorted(set(plan_by_id) - set(canonical_by_id))
    changed_arms = {
        ablation_id: {
            "expected": canonical_by_id[ablation_id],
            "observed": plan_by_id[ablation_id],
        }
        for ablation_id in sorted(set(canonical_by_id) & set(plan_by_id))
        if canonical_by_id[ablation_id] != plan_by_id[ablation_id]
    }
    payload_json_path = Path(str(plan.get("json_path") or "")).expanduser() if plan.get("json_path") else None
    payload_json_hash = _sha256_file(payload_json_path) if payload_json_path else None
    plan_hash = _sha256_file(plan_path)
    sibling_latest = plan_path.with_name("ENTRY_SMART_ABLATION_REPLAY_PLAN_GATE_latest.json")
    sibling_latest_hash = _sha256_file(sibling_latest)
    compare_latest = sibling_latest.exists() and plan_path.resolve(strict=False) != sibling_latest.resolve(strict=False)

    return [
        _check("plan JSON exists and parses as object", plan_error is None, {"error": plan_error, **_artifact_meta(plan_path)}),
        _check(
            "plan decision is ready",
            str(plan.get("decision") or "") == PLAN_READY_DECISION,
            {"expected": PLAN_READY_DECISION, "observed": plan.get("decision")},
        ),
        _check(
            "plan gate schema matches smart ablation replay plan gate",
            str(plan.get("schema_version") or "") == "entry_smart_ablation_replay_plan_gate_v1",
            {"schema_version": plan.get("schema_version")},
        ),
        _check(
            "plan blocks training/replay/IQL/shadow/live",
            bool(plan.get("training_allowed")) is False
            and bool(plan.get("replay_allowed_by_this_gate")) is False
            and bool(plan.get("iql_allowed_by_this_gate")) is False
            and bool(plan.get("shadow_live_promotion_allowed")) is False
            and all(not bool(value) for value in side_effects_started.values()),
            {
                "training_allowed": plan.get("training_allowed"),
                "replay_allowed_by_this_gate": plan.get("replay_allowed_by_this_gate"),
                "iql_allowed_by_this_gate": plan.get("iql_allowed_by_this_gate"),
                "shadow_live_promotion_allowed": plan.get("shadow_live_promotion_allowed"),
                "side_effects_started": plan.get("side_effects_started"),
            },
        ),
        _check(
            "plan required_ablation_plan schema matches matrix v1",
            str(required_plan.get("schema_version") or "") == "entry_smart_ablation_matrix_v1",
            {"schema_version": required_plan.get("schema_version")},
        ),
        _check(
            "plan requires exactly 14 ablations",
            len(rows) == REQUIRED_ABLATION_COUNT
            and int(required_plan.get("ablation_count") or 0) == REQUIRED_ABLATION_COUNT,
            {
                "required_ablation_count": REQUIRED_ABLATION_COUNT,
                "declared": required_plan.get("ablation_count"),
                "observed": len(rows),
            },
        ),
        _check(
            "plan ablation IDs are present and unique",
            len(row_ids) == len(rows) and all(row_ids) and not duplicate_ids,
            {"ablation_ids": row_ids, "duplicates": duplicate_ids},
        ),
        _check(
            "plan smart variant matches expected smart candidate",
            str(required_plan.get("smart_variant") or plan.get("smart_variant") or "") == SMART_VARIANT,
            {
                "expected": SMART_VARIANT,
                "observed": required_plan.get("smart_variant") or plan.get("smart_variant"),
            },
        ),
        _check(
            "plan is current against canonical ablation matrix",
            not missing_canonical and not unexpected_plan_ids and not changed_arms,
            {
                "missing_canonical_ids": missing_canonical,
                "unexpected_plan_ids": unexpected_plan_ids,
                "changed_arms": changed_arms,
            },
        ),
        _check(
            "plan latest/timestamp artifact is not stale",
            bool(plan_hash)
            and bool(payload_json_path)
            and bool(payload_json_hash)
            and plan_hash == payload_json_hash
            and (not compare_latest or sibling_latest_hash == plan_hash),
            {
                "plan_json": str(plan_path),
                "plan_json_sha256": plan_hash,
                "payload_json_path": str(payload_json_path) if payload_json_path else "",
                "payload_json_sha256": payload_json_hash,
                "sibling_latest_json": str(sibling_latest),
                "sibling_latest_sha256": sibling_latest_hash,
                "selected_matches_sibling_latest_required": compare_latest,
            },
        ),
    ]


def _finite_required_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, Any]:
    missing = [col for col in columns if col not in frame.columns]
    nonfinite: dict[str, int] = {}
    for col in columns:
        if col not in frame.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        bad_count = int((~np.isfinite(values.to_numpy(dtype=float))).sum())
        if bad_count:
            nonfinite[col] = bad_count
    return {"missing": missing, "nonfinite": nonfinite}


def _slice_scope_values(slices: pd.DataFrame, scope: str) -> dict[str, Any]:
    aliases = tuple(str(value).lower() for value in REQUIRED_SLICE_SCOPE_ALIASES[scope])
    matches: list[dict[str, Any]] = []
    for col in SLICE_SCOPE_COLUMNS:
        if col not in slices.columns:
            continue
        series = slices[col].dropna().astype(str).str.strip()
        lowered = series.str.lower()
        mask = lowered.isin(aliases)
        for alias in aliases:
            mask = mask | lowered.str.startswith(f"{alias}=")
        if bool(mask.any()):
            matches.append({"column": col, "count": int(mask.sum())})
    for col in SLICE_VALUE_COLUMNS:
        if col not in slices.columns:
            continue
        series = slices[col].dropna().astype(str).str.strip()
        for scope_col in SLICE_SCOPE_COLUMNS:
            if scope_col not in slices.columns:
                continue
            scope_series = slices[scope_col].dropna().astype(str).str.strip()
            mask = scope_series.str.lower().isin(aliases) & series.ne("")
            if bool(mask.any()):
                matches.append({"column": f"{scope_col}+{col}", "count": int(mask.sum())})
    return {"scope": scope, "accepted_aliases": list(aliases), "present": bool(matches), "matches": matches}


def _slice_nonfinite_summary(slices: pd.DataFrame) -> dict[str, Any]:
    required_numeric = [
        col for col in REQUIRED_METRIC_COLUMNS if col in slices.columns
    ]
    optional_numeric = [
        col
        for col in (
            "net_mean_bps",
            "win_rate",
            "mean_score",
            "mean_mae_bps",
            "mean_mfe_bps",
            "mean_bad_path_prob",
            "bad_path_rate",
            "mean_path_quality_pred",
            "path_quality_p10",
            "mae_p95_bps",
        )
        if col in slices.columns
    ]
    checked = tuple(dict.fromkeys(required_numeric + optional_numeric))
    summary = _finite_required_columns(slices, checked)
    summary["checked_columns"] = list(checked)
    profit_factor_bad = int(summary["nonfinite"].get("profit_factor", 0))
    if profit_factor_bad and {"profit_factor", "win_rate", "max_drawdown_bps", "net_sum_bps"}.issubset(slices.columns):
        pf = pd.to_numeric(slices["profit_factor"], errors="coerce")
        bad_mask = ~np.isfinite(pf.to_numpy(dtype=float))
        win_rate = pd.to_numeric(slices["win_rate"], errors="coerce")
        drawdown = pd.to_numeric(slices["max_drawdown_bps"], errors="coerce")
        net = pd.to_numeric(slices["net_sum_bps"], errors="coerce")
        no_loss_mask = bad_mask & (win_rate >= 1.0) & (drawdown <= 0.0) & (net > 0.0)
        if int(no_loss_mask.sum()) == profit_factor_bad:
            summary["nullable_profit_factor_no_loss_rows"] = int(no_loss_mask.sum())
            summary["nonfinite"].pop("profit_factor", None)
    return summary


def _metric_row(metrics: pd.DataFrame) -> dict[str, Any]:
    if metrics.empty:
        return {}
    frame = metrics.copy()
    if "scope" in frame.columns:
        aggregate = frame[frame["scope"].astype(str).str.lower().eq("aggregate")]
        if not aggregate.empty:
            frame = aggregate
    if "fold" in frame.columns:
        all_fold = frame[frame["fold"].astype(str).str.upper().eq("ALL")]
        if not all_fold.empty:
            frame = all_fold
    if "net_sum_bps" in frame.columns:
        values = pd.to_numeric(frame["net_sum_bps"], errors="coerce")
        if bool(values.notna().any()):
            return frame.iloc[int(values.fillna(-np.inf).to_numpy().argmax())].to_dict()
    return frame.iloc[0].to_dict()


def _num(row: dict[str, Any], key: str) -> float | None:
    value = pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
    if pd.isna(value) or not np.isfinite(float(value)):
        return None
    return float(value)


def _metrics_summary(metrics: pd.DataFrame) -> dict[str, Any]:
    row = _metric_row(metrics)
    if not row:
        return {}
    out: dict[str, Any] = {}
    for key in (
        "policy_id",
        "n_trades",
        "net_sum_bps",
        "profit_factor",
        "max_drawdown_bps",
        "win_rate",
        "direction_precision",
        "mean_mae_bps",
        "mean_mfe_bps",
        "avg_trades_per_day",
    ):
        if key not in row:
            continue
        if key == "policy_id":
            out[key] = str(row.get(key) or "")
            continue
        value = _num(row, key)
        out[key] = int(value) if key == "n_trades" and value is not None else value
    return out


def _monthly_summary(monthly: pd.DataFrame) -> dict[str, Any]:
    if monthly.empty or "net_sum_bps" not in monthly.columns:
        return {"months": int(len(monthly)), "negative_months": None, "min_month_net_bps": None}
    values = pd.to_numeric(monthly["net_sum_bps"], errors="coerce")
    valid = monthly.loc[values.notna()].copy()
    if valid.empty:
        return {"months": int(len(monthly)), "negative_months": None, "min_month_net_bps": None}
    valid["_net_sum_bps_num"] = pd.to_numeric(valid["net_sum_bps"], errors="coerce")
    worst = valid.sort_values("_net_sum_bps_num", ascending=True).iloc[0].to_dict()
    month_label = str(worst.get("entry_month") or worst.get("month") or "")
    min_month = _num(worst, "_net_sum_bps_num")
    negative_months = int((valid["_net_sum_bps_num"] < 0).sum())
    return {
        "months": int(len(valid)),
        "negative_months": negative_months,
        "all_months_positive": negative_months == 0,
        "min_month_net_bps": min_month,
        "worst_month": month_label,
    }


def _slice_edge_summary(slices: pd.DataFrame) -> dict[str, Any]:
    if slices.empty or "net_sum_bps" not in slices.columns:
        return {"slices": int(len(slices)), "negative_slices": None, "worst_slice": {}}
    frame = slices.copy()
    frame["_net_sum_bps_num"] = pd.to_numeric(frame["net_sum_bps"], errors="coerce")
    valid = frame.loc[frame["_net_sum_bps_num"].notna()]
    if valid.empty:
        return {"slices": int(len(slices)), "negative_slices": None, "worst_slice": {}}
    worst = valid.sort_values("_net_sum_bps_num", ascending=True).iloc[0].to_dict()
    family_col = "slice_family" if "slice_family" in valid.columns else "scope" if "scope" in valid.columns else ""
    value_col = "slice_value" if "slice_value" in valid.columns else "value" if "value" in valid.columns else ""
    by_family: dict[str, Any] = {}
    if family_col:
        for family, group in valid.groupby(family_col, dropna=False):
            values = pd.to_numeric(group["_net_sum_bps_num"], errors="coerce")
            by_family[str(family)] = {
                "slice_count": int(len(group)),
                "min_net_sum_bps": float(values.min()) if len(values) else None,
                "negative_slices": int((values < 0).sum()),
            }
    return {
        "slices": int(len(valid)),
        "negative_slices": int((valid["_net_sum_bps_num"] < 0).sum()),
        "worst_slice": {
            "family": str(worst.get(family_col) or "") if family_col else "",
            "value": str(worst.get(value_col) or "") if value_col else "",
            "net_sum_bps": _num(worst, "_net_sum_bps_num"),
        },
        "by_family": by_family,
    }


def _artifact_hashes_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    for source in _nested_dicts(manifest):
        artifact_hashes = source.get("artifact_hashes")
        if isinstance(artifact_hashes, dict):
            return artifact_hashes
    return {}


def _validate_replay_arm(
    *,
    arm: dict[str, Any],
    plan: dict[str, Any],
    replay_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ablation_id = str(arm.get("ablation_id") or "")
    expected_variant = str(arm.get("manifest_variant") or SMART_VARIANT)
    replay_dir = replay_root / ablation_id
    paths = {name: replay_dir / name for name in REQUIRED_REPLAY_FILES}
    manifest, manifest_error = _read_json_report(paths["REPLAY_EVIDENCE_MANIFEST.json"])
    csvs: dict[str, pd.DataFrame] = {}
    csv_errors: dict[str, str | None] = {}
    for name in REQUIRED_REPLAY_FILES:
        if not name.endswith(".csv"):
            continue
        csvs[name], csv_errors[name] = _read_csv_report(paths[name])

    metrics = csvs.get("replay_policy_metrics.csv", pd.DataFrame())
    monthly = csvs.get("replay_policy_monthly.csv", pd.DataFrame())
    trades = csvs.get("replay_policy_trades.csv", pd.DataFrame())
    slices = csvs.get("replay_policy_slices.csv", pd.DataFrame())
    observed_variant = _manifest_variant(manifest)
    observed_ablation_id = _manifest_ablation_id(manifest)
    replay_identity = _manifest_identity(manifest)
    artifact_hashes = _artifact_hashes_from_manifest(manifest)
    feature_mask = _manifest_feature_mask(manifest)
    feature_mask_path = Path(str(feature_mask.get("path") or "")).expanduser() if feature_mask.get("path") else None
    feature_mask_sha = _sha256_file(feature_mask_path) if feature_mask_path else None
    feature_mask_spec, feature_mask_spec_error = (
        _read_json_report(feature_mask_path) if feature_mask_path else ({}, "feature-mask path is missing")
    )
    selective_summary_path = _summary_path_from_manifest(manifest, replay_identity)
    selective_summary, selective_summary_error = (
        _read_json_report(selective_summary_path)
        if selective_summary_path
        else ({}, "selective-edge summary path is missing")
    )
    selective_summary_sha = _sha256_file(selective_summary_path) if selective_summary_path else None
    identity_contract = _identity_expected_contract(
        plan=plan,
        identity=replay_identity,
        manifest=manifest,
        selective_summary=selective_summary,
        selective_summary_path=selective_summary_path,
        selective_summary_sha=selective_summary_sha,
    )
    no_xgb_contract = _no_xgb_contract(selective_summary)
    ablation_type = str(arm.get("ablation_type") or "")
    xgb_bridge_mode = str(arm.get("xgb_bridge_mode") or "")
    requires_feature_mask = ablation_type in {"feature_set_ablation", "drop_smart_family"}
    requires_no_xgb = xgb_bridge_mode == "neutralize_signal_bridge"
    expected_mask_count = max(0, CONTRACT_SEQ_SNAP_WIDTH - int(arm.get("expected_seq_snap_width") or 0))
    observed_mask_indices = [int(value) for value in feature_mask.get("zero_indices", [])] if isinstance(feature_mask.get("zero_indices"), list) else []
    observed_mask_names = [str(value) for value in feature_mask.get("zero_feature_names", [])] if isinstance(feature_mask.get("zero_feature_names"), list) else []
    spec_mask_indices = [int(value) for value in feature_mask_spec.get("zero_indices", [])] if isinstance(feature_mask_spec.get("zero_indices"), list) else []
    spec_mask_names = [str(value) for value in feature_mask_spec.get("zero_feature_names", [])] if isinstance(feature_mask_spec.get("zero_feature_names"), list) else []
    mask_zero_value = (
        feature_mask.get("zero_value")
        if feature_mask.get("zero_value") is not None
        else feature_mask_spec.get("zero_value")
    )
    expected_arm_signature = _canonical_arm_signature(arm)
    mask_plan_signature = _canonical_arm_signature(feature_mask_spec.get("plan_arm") if isinstance(feature_mask_spec.get("plan_arm"), dict) else {})
    expected_policy_id = str(manifest.get("policy_id") or "")
    manifest_policies = sorted({str(value).strip() for value in manifest.get("policies", []) if str(value).strip()}) if isinstance(manifest.get("policies"), list) else []
    best_policy_id = str(_dict_value(manifest, "best_aggregate_row").get("policy_id") or "")
    policy_review = {
        "manifest_policy_id": expected_policy_id,
        "manifest_policies": manifest_policies,
        "best_aggregate_policy_id": best_policy_id,
        "metrics_policy_ids": _policy_values(metrics),
        "monthly_policy_ids": _policy_values(monthly),
        "trades_policy_ids": _policy_values(trades),
        "slices_policy_ids": _policy_values(slices),
    }
    artifact_hash_review = {
        name: {
            "expected": artifact_hashes.get(name),
            "observed": _sha256_file(path),
        }
        for name, path in paths.items()
        if name.endswith(".csv")
    }
    metric_finite = _finite_required_columns(metrics, REQUIRED_METRIC_COLUMNS)
    slice_scopes = {scope: _slice_scope_values(slices, scope) for scope in REQUIRED_SLICE_SCOPES}
    slice_nonfinite = _slice_nonfinite_summary(slices)

    checks = [
        _check(f"{ablation_id} replay dir exists", replay_dir.exists() and replay_dir.is_dir(), {"replay_dir": str(replay_dir)}),
        *[
            _check(
                f"{ablation_id} required file exists: {name}",
                path.exists() and path.is_file() and path.stat().st_size > 0,
                _artifact_meta(path),
            )
            for name, path in paths.items()
        ],
        _check(
            f"{ablation_id} replay manifest parses",
            manifest_error is None,
            {"error": manifest_error, **_artifact_meta(paths["REPLAY_EVIDENCE_MANIFEST.json"])},
        ),
        _check(
            f"{ablation_id} replay manifest decision PASS",
            str(manifest.get("decision") or "") == "PASS",
            {"decision": manifest.get("decision"), "failures": manifest.get("failures")},
        ),
        _check(
            f"{ablation_id} replay manifest declares variant",
            bool(observed_variant),
            {"observed_variant": observed_variant},
        ),
        _check(
            f"{ablation_id} replay manifest variant matches plan",
            bool(observed_variant) and observed_variant == expected_variant,
            {"expected_variant": expected_variant, "observed_variant": observed_variant},
        ),
        _check(
            f"{ablation_id} replay manifest declares ablation_id",
            bool(observed_ablation_id),
            {"observed_ablation_id": observed_ablation_id},
        ),
        _check(
            f"{ablation_id} replay manifest ablation_id matches plan",
            bool(observed_ablation_id) and observed_ablation_id == ablation_id,
            {"expected_ablation_id": ablation_id, "observed_ablation_id": observed_ablation_id},
        ),
        _check(
            f"{ablation_id} replay identity binds to active smart seq520 candidate",
            _valid_seq520_identity(identity_contract),
            identity_contract,
        ),
        _check(
            f"{ablation_id} selective-edge summary exists and matches replay identity hash",
            selective_summary_error is None
            and bool(identity_contract["selective_edge_summary_sha256"])
            and identity_contract["selective_edge_summary_sha256"] == selective_summary_sha,
            {
                "error": selective_summary_error,
                "selective_edge_summary_json": str(selective_summary_path) if selective_summary_path else "",
                "expected_sha256": identity_contract["selective_edge_summary_sha256"],
                "observed_sha256": selective_summary_sha,
            },
        ),
        _check(
            f"{ablation_id} no-XGB provenance matches neutralized bridge contract when required",
            (not requires_no_xgb) or _valid_no_xgb_contract(no_xgb_contract),
            {
                "requires_no_xgb": requires_no_xgb,
                "xgb_bridge_mode": xgb_bridge_mode,
                "no_xgb_contract": no_xgb_contract,
            },
        ),
        _check(
            f"{ablation_id} feature-mask provenance required when plan masks features",
            (not requires_feature_mask) or bool(feature_mask),
            {
                "requires_feature_mask": requires_feature_mask,
                "ablation_type": ablation_type,
                "feature_mask_present": bool(feature_mask),
            },
        ),
        _check(
            f"{ablation_id} feature-mask ablation_id matches plan",
            (not requires_feature_mask)
            or (
                bool(feature_mask)
                and bool(feature_mask.get("enabled"))
                and str(feature_mask.get("ablation_id") or "") == ablation_id
            ),
            {
                "requires_feature_mask": requires_feature_mask,
                "enabled": feature_mask.get("enabled"),
                "expected_ablation_id": ablation_id,
                "observed_ablation_id": feature_mask.get("ablation_id"),
            },
        ),
        _check(
            f"{ablation_id} feature-mask count matches plan width delta",
            (not requires_feature_mask) or (len(observed_mask_indices) == expected_mask_count),
            {
                "requires_feature_mask": requires_feature_mask,
                "contract_width": CONTRACT_SEQ_SNAP_WIDTH,
                "expected_seq_snap_width": arm.get("expected_seq_snap_width"),
                "expected_mask_count": expected_mask_count,
                "observed_mask_count": len(observed_mask_indices),
            },
        ),
        _check(
            f"{ablation_id} feature-mask spec parses and canonical plan arm matches",
            (not requires_feature_mask)
            or (
                feature_mask_spec_error is None
                and bool(feature_mask_spec)
                and mask_plan_signature == expected_arm_signature
            ),
            {
                "requires_feature_mask": requires_feature_mask,
                "error": feature_mask_spec_error,
                "expected_plan_arm": expected_arm_signature,
                "observed_plan_arm": mask_plan_signature,
            },
        ),
        _check(
            f"{ablation_id} feature-mask zero contract matches seq520 mask spec",
            (not requires_feature_mask)
            or (
                str(feature_mask.get("mask_mode") or feature_mask_spec.get("mask_mode") or "") == "zero_seq_snap_features"
                and mask_zero_value is not None
                and float(mask_zero_value) == 0.0
                and int(feature_mask.get("signal_field_count") or feature_mask_spec.get("signal_field_count") or 0)
                == CONTRACT_SEQ_SNAP_WIDTH
                and int(feature_mask_spec.get("zero_feature_count") or 0) == expected_mask_count
                and observed_mask_indices == spec_mask_indices
                and observed_mask_names == spec_mask_names
                and len(set(observed_mask_indices)) == len(observed_mask_indices)
                and all(0 <= idx < CONTRACT_SEQ_SNAP_WIDTH for idx in observed_mask_indices)
            ),
            {
                "requires_feature_mask": requires_feature_mask,
                "mask_mode": feature_mask.get("mask_mode") or feature_mask_spec.get("mask_mode"),
                "zero_value": mask_zero_value,
                "signal_field_count": feature_mask.get("signal_field_count") or feature_mask_spec.get("signal_field_count"),
                "expected_mask_count": expected_mask_count,
                "spec_zero_feature_count": feature_mask_spec.get("zero_feature_count"),
                "summary_indices_match_spec": observed_mask_indices == spec_mask_indices,
                "summary_names_match_spec": observed_mask_names == spec_mask_names,
                "unique_indices": len(set(observed_mask_indices)) == len(observed_mask_indices),
                "out_of_range_indices": [idx for idx in observed_mask_indices if idx < 0 or idx >= CONTRACT_SEQ_SNAP_WIDTH],
            },
        ),
        _check(
            f"{ablation_id} feature-mask hash matches artifact",
            (not requires_feature_mask)
            or (
                bool(feature_mask.get("path"))
                and bool(feature_mask.get("sha256"))
                and feature_mask_sha == feature_mask.get("sha256")
            ),
            {
                "requires_feature_mask": requires_feature_mask,
                "path": feature_mask.get("path"),
                "expected_sha256": feature_mask.get("sha256"),
                "observed_sha256": feature_mask_sha,
            },
        ),
        _check(
            f"{ablation_id} policy_id is non-empty and consistent across manifest and replay artifacts",
            bool(expected_policy_id)
            and manifest_policies == [expected_policy_id]
            and best_policy_id == expected_policy_id
            and policy_review["metrics_policy_ids"] == [expected_policy_id]
            and policy_review["monthly_policy_ids"] == [expected_policy_id]
            and policy_review["trades_policy_ids"] == [expected_policy_id]
            and policy_review["slices_policy_ids"] == [expected_policy_id],
            policy_review,
        ),
        _check(
            f"{ablation_id} replay metrics reads with rows",
            csv_errors.get("replay_policy_metrics.csv") is None and not metrics.empty,
            {"error": csv_errors.get("replay_policy_metrics.csv"), "rows": int(len(metrics)), "columns": list(metrics.columns)},
        ),
        _check(
            f"{ablation_id} replay monthly reads with rows",
            csv_errors.get("replay_policy_monthly.csv") is None and not monthly.empty,
            {"error": csv_errors.get("replay_policy_monthly.csv"), "rows": int(len(monthly)), "columns": list(monthly.columns)},
        ),
        _check(
            f"{ablation_id} replay trades reads with rows",
            csv_errors.get("replay_policy_trades.csv") is None and not trades.empty,
            {"error": csv_errors.get("replay_policy_trades.csv"), "rows": int(len(trades)), "columns": list(trades.columns)},
        ),
        _check(
            f"{ablation_id} replay slices reads with rows",
            csv_errors.get("replay_policy_slices.csv") is None and not slices.empty,
            {"error": csv_errors.get("replay_policy_slices.csv"), "rows": int(len(slices)), "columns": list(slices.columns)},
        ),
        _check(
            f"{ablation_id} replay metrics contain required columns",
            not metric_finite["missing"],
            {"required_columns": list(REQUIRED_METRIC_COLUMNS), **metric_finite},
        ),
        _check(
            f"{ablation_id} replay metrics required columns are finite",
            not metric_finite["missing"] and not metric_finite["nonfinite"],
            {"required_columns": list(REQUIRED_METRIC_COLUMNS), **metric_finite},
        ),
        _check(
            f"{ablation_id} replay metrics declare positive trades",
            "n_trades" in metrics.columns
            and not metrics.empty
            and bool((pd.to_numeric(metrics["n_trades"], errors="coerce") > 0).all()),
            {"n_trades": metrics["n_trades"].tolist() if "n_trades" in metrics.columns else []},
        ),
        _check(
            f"{ablation_id} replay slices contain required scopes",
            all(row["present"] for row in slice_scopes.values()),
            slice_scopes,
        ),
        _check(
            f"{ablation_id} replay slices numeric fields are finite",
            not slice_nonfinite["missing"] and not slice_nonfinite["nonfinite"],
            slice_nonfinite,
        ),
        _check(
            f"{ablation_id} replay manifest artifact hashes match CSVs",
            all(row["expected"] and row["expected"] == row["observed"] for row in artifact_hash_review.values()),
            artifact_hash_review,
        ),
    ]
    identity = {
        "ablation_id": ablation_id,
        "expected_variant": expected_variant,
        "observed_variant": observed_variant,
        "observed_ablation_id": observed_ablation_id,
        "replay_dir": str(replay_dir),
        "files": {name: _artifact_meta(path) for name, path in paths.items()},
        "metrics_rows": int(len(metrics)),
        "monthly_rows": int(len(monthly)),
        "trades_rows": int(len(trades)),
        "slices_rows": int(len(slices)),
        "metrics_columns": list(metrics.columns),
        "metrics_summary": _metrics_summary(metrics),
        "monthly_summary": _monthly_summary(monthly),
        "slice_edge_summary": _slice_edge_summary(slices),
        "feature_mask": feature_mask,
        "feature_mask_spec": {
            "path": str(feature_mask_path) if feature_mask_path else "",
            "sha256": feature_mask_sha,
            "plan_arm": mask_plan_signature,
            "zero_feature_count": feature_mask_spec.get("zero_feature_count"),
        },
        "replay_identity_contract": identity_contract,
        "no_xgb_contract": no_xgb_contract if requires_no_xgb else {},
        "policy_review": policy_review,
        "slice_scopes": slice_scopes,
        "artifact_hashes": artifact_hash_review,
        "plan_arm": _canonical_arm_signature(arm),
    }
    return identity, checks


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Smart Ablation Replay Matrix Gate",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Plan decision: `{report['plan_decision']}`",
        f"- Replay root: `{report['replay_root']}`",
        f"- Ablations checked: `{report['ablation_count_observed']}`",
        f"- Required ablations: `{report['ablation_count_required']}`",
        f"- Failures: `{len(report['failures'])}`",
        f"- Training/replay/IQL/shadow/live started: `{False}`",
        "",
        "## Ablation Arms",
        "",
    ]
    for arm in report["ablation_results"]:
        identity = arm.get("identity") if isinstance(arm.get("identity"), dict) else {}
        metrics = identity.get("metrics_summary") if isinstance(identity.get("metrics_summary"), dict) else {}
        monthly = identity.get("monthly_summary") if isinstance(identity.get("monthly_summary"), dict) else {}
        slices = identity.get("slice_edge_summary") if isinstance(identity.get("slice_edge_summary"), dict) else {}
        edge_bits = []
        if metrics:
            edge_bits.extend(
                [
                    f"n={metrics.get('n_trades')}",
                    f"net={metrics.get('net_sum_bps')}",
                    f"PF={metrics.get('profit_factor')}",
                    f"DD={metrics.get('max_drawdown_bps')}",
                ]
            )
        if monthly:
            edge_bits.append(f"neg_months={monthly.get('negative_months')}")
        if slices:
            edge_bits.append(f"neg_slices={slices.get('negative_slices')}")
        suffix = f"; {'; '.join(edge_bits)}" if edge_bits else ""
        lines.append(
            f"- `{arm['ablation_id']}`: `{arm['decision']}` "
            f"({arm['passed']}/{arm['total']} checks{suffix})"
        )
    lines.extend(["", "## Failures", ""])
    if report["failures"]:
        for failure in report["failures"]:
            lines.append(f"- `{failure['gate']}` / `{failure['check']}`")
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = Path(args.plan_json).expanduser().resolve()
    replay_root = Path(args.replay_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plan, plan_error = _read_json_report(plan_path)
    plan_checks = _plan_freshness_checks(plan_path, plan, plan_error)
    arms = _required_ablation_rows(plan)
    ablation_results: list[dict[str, Any]] = []
    gate_checks: dict[str, list[dict[str, Any]]] = {
        "plan": plan_checks,
        "side_effect_guard": [
            _check("gate never starts training", True),
            _check("gate never starts replay", True),
            _check("gate never starts IQL", True),
            _check("gate never starts shadow", True),
            _check("gate never starts live", True),
            _check("gate never promotes", True),
        ],
    }
    if len(arms) == REQUIRED_ABLATION_COUNT:
        for arm in arms:
            ablation_id = str(arm.get("ablation_id") or "")
            identity, checks = _validate_replay_arm(arm=arm, plan=plan, replay_root=replay_root)
            gate_name = f"ablation:{ablation_id or 'missing_ablation_id'}"
            gate_checks[gate_name] = checks
            passed = int(sum(1 for check in checks if check["ok"]))
            ablation_results.append(
                {
                    "ablation_id": ablation_id,
                    "decision": "PASS" if passed == len(checks) else "FAIL",
                    "passed": passed,
                    "total": int(len(checks)),
                    "identity": identity,
                    "checks": checks,
                }
            )
    else:
        gate_checks["ablation_matrix"] = [
            _check(
                "ablation matrix validation skipped because plan does not expose exactly 14 arms",
                False,
                {"observed": len(arms), "required": REQUIRED_ABLATION_COUNT},
            )
        ]

    gates = []
    for name, checks in gate_checks.items():
        passed = int(sum(1 for check in checks if check["ok"]))
        gates.append(
            {
                "name": name,
                "decision": "PASS" if passed == len(checks) else "FAIL",
                "passed": passed,
                "total": int(len(checks)),
                "checks": checks,
            }
        )
    failures = [
        {"gate": gate["name"], "check": check["name"], "details": check.get("details")}
        for gate in gates
        for check in gate["checks"]
        if not check["ok"]
    ]
    ready = not failures
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"ENTRY_SMART_ABLATION_REPLAY_MATRIX_GATE_{timestamp}.json"
    md_path = out_dir / f"ENTRY_SMART_ABLATION_REPLAY_MATRIX_GATE_{timestamp}.md"
    report = {
        "schema_version": "entry_smart_ablation_replay_matrix_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "READY_FOR_SMART_ABLATION_REPLAY_MATRIX_REVIEW" if ready else "BLOCKED_SMART_ABLATION_REPLAY_MATRIX_GATE",
        "report_only": True,
        "training_allowed": False,
        "replay_allowed_by_this_gate": False,
        "iql_allowed_by_this_gate": False,
        "shadow_live_promotion_allowed": False,
        "side_effects_started": {
            "training": False,
            "replay": False,
            "iql_distillation": False,
            "shadow": False,
            "live": False,
            "promotion": False,
        },
        "plan_json": str(plan_path),
        "plan_decision": plan.get("decision"),
        "replay_root": str(replay_root),
        "ablation_count_required": REQUIRED_ABLATION_COUNT,
        "ablation_count_observed": int(len(arms)),
        "required_replay_files": list(REQUIRED_REPLAY_FILES),
        "required_metric_columns": list(REQUIRED_METRIC_COLUMNS),
        "required_slice_scopes": list(REQUIRED_SLICE_SCOPES),
        "gates": gates,
        "ablation_results": ablation_results,
        "failures": failures,
        "next_required_gate": (
            "manual review of ablation matrix evidence; promotion, shadow, live, IQL and replay remain blocked"
            if ready
            else "materialize missing PASS replay evidence for every required ablation arm, then rerun this report-only gate"
        ),
        "json_path": str(json_path),
        "md_path": str(md_path),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    (out_dir / "ENTRY_SMART_ABLATION_REPLAY_MATRIX_GATE_latest.json").write_text(
        json_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out_dir / "ENTRY_SMART_ABLATION_REPLAY_MATRIX_GATE_latest.md").write_text(
        md_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "failures": failures,
                    "json_path": report["json_path"],
                    "md_path": report["md_path"],
                    "next_required_gate": report["next_required_gate"],
                },
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
        )
    if args.fail_on_not_ready and failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan-json", default=str(DEFAULT_PLAN_JSON))
    ap.add_argument("--replay-root", default=str(DEFAULT_REPLAY_ROOT))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
