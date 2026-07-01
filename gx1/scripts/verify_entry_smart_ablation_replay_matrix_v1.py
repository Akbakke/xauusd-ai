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
    artifact_hashes = _artifact_hashes_from_manifest(manifest)
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
            identity, checks = _validate_replay_arm(arm=arm, replay_root=replay_root)
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
