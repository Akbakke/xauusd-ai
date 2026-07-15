#!/usr/bin/env python3
"""Diagnose failed XAU direction slices against emitted smart520 features.

Report-only. This audit reads fail-closed direction-slice evidence and the
matching XAU validation parquet, then measures whether existing rail/SR/wick/
regime features separate LONG vs SHORT labels inside the red ctx slices.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.audit.entry_transformer_feature_audit import _stack_list_column


DEFAULT_RUN_ROOT = Path(
    "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/"
    "v10_6yr_rebuild_20260628_foundation_seq146"
)
DEFAULT_OUT_DIR = Path(
    "/home/andre2/GX1_DATA/reports/xau_red_slice_separability_audit_20260715_v1"
)
XGB_SIGNAL_FIELD_COUNT = 7
SIDE_NAMES = {0: "LONG", 1: "SHORT", 2: "FLAT"}
REQUIRED_XAU_DIRECTION_FEATURES = (
    "chart.geometry_rising_support_rail_long_pressure",
    "chart.geometry_rising_support_rail_short_trap_pressure",
    "chart.geometry_falling_resistance_rail_short_pressure",
    "chart.geometry_falling_resistance_rail_long_trap_pressure",
)
DOMAIN_KEYWORDS = (
    "support",
    "resistance",
    "rail",
    "wick",
    "sweep",
    "rejection",
    "trap",
    "sr_memory",
    "confluence",
    "h4",
    "d1",
    "trend",
    "regime",
    "liquidity",
)
TARGET_COLUMNS = (
    "y_direction",
    "y_long_path_utility_bps",
    "y_short_path_utility_bps",
    "y_long_bad_path",
    "y_short_bad_path",
    "y_rising_channel_support_touch",
    "y_falling_channel_resistance_touch",
    "y_countertrend_short_trap",
    "y_countertrend_long_trap",
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj) if np.isfinite(obj) else None
    return str(obj)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"missing json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_failure_evidence(run_root: Path) -> Path:
    matches = sorted(
        run_root.glob("v10_entry_smart_seq520_smoke_*__direction_slice_failure_evidence.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not matches:
        raise RuntimeError(f"no XAU smart smoke direction-slice failure evidence under {run_root}")
    return matches[-1]


def _manifest_for_parquet(parquet_path: Path) -> Path:
    manifest = parquet_path.with_suffix(".manifest.json")
    if not manifest.exists():
        raise RuntimeError(f"missing split manifest for {parquet_path}: {manifest}")
    return manifest


def _manifest_fields(manifest: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    extra = manifest.get("extra") if isinstance(manifest.get("extra"), dict) else {}
    bridge = extra.get("signal_bridge") if isinstance(extra.get("signal_bridge"), dict) else {}
    ctx = extra.get("ctx_contract") if isinstance(extra.get("ctx_contract"), dict) else {}
    fields = bridge.get("fields") or bridge.get("snap_fields")
    if not isinstance(fields, list) or not all(isinstance(item, str) for item in fields):
        raise RuntimeError("split manifest lacks signal_bridge.fields/snap_fields")
    ctx_cont_names = list(ctx.get("ctx_cont_names") or [])
    ctx_cat_names = list(ctx.get("ctx_cat_names") or [])
    return list(fields), ctx_cont_names, ctx_cat_names


def _require_xau_path(path: Path, *, role: str) -> None:
    text = str(path)
    if "xau_direction_repair" not in text:
        raise RuntimeError(f"{role} must be an XAU direction-repair path, got {path}")


def _rate_dict(labels: np.ndarray) -> dict[str, float]:
    counts = np.bincount(labels.astype(np.int64), minlength=3)
    total = int(counts.sum())
    if total <= 0:
        return {SIDE_NAMES[i]: 0.0 for i in range(3)}
    return {SIDE_NAMES[i]: float(counts[i] / total) for i in range(3)}


def _safe_mean(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(finite.mean())


def _feature_summary(values: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    labels = labels.astype(np.int64)
    means: dict[str, float | None] = {}
    counts: dict[str, int] = {}
    for klass, name in SIDE_NAMES.items():
        mask = labels == klass
        counts[name] = int(mask.sum())
        means[name] = _safe_mean(values[mask])
    long_mean = means["LONG"]
    short_mean = means["SHORT"]
    delta = None if long_mean is None or short_mean is None else float(long_mean - short_mean)
    finite = values[np.isfinite(values)]
    pooled = float(finite.std()) if finite.size > 1 else 0.0
    standardized = None if delta is None or pooled <= 1e-12 else float(delta / pooled)
    return {
        "counts": counts,
        "means": means,
        "long_minus_short_mean": delta,
        "abs_long_short_mean": None if delta is None else abs(delta),
        "pooled_std": pooled,
        "standardized_long_minus_short": standardized,
        "abs_standardized_long_short": None if standardized is None else abs(standardized),
        "overall_mean": _safe_mean(values),
    }


def _target_summary(frame: pd.DataFrame, mask: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in TARGET_COLUMNS:
        if col in frame.columns:
            out[f"{col}_mean"] = _safe_mean(pd.to_numeric(frame.loc[mask, col], errors="coerce").to_numpy())
    if "y_long_path_utility_bps" in frame.columns and "y_short_path_utility_bps" in frame.columns:
        long_u = pd.to_numeric(frame.loc[mask, "y_long_path_utility_bps"], errors="coerce").to_numpy()
        short_u = pd.to_numeric(frame.loc[mask, "y_short_path_utility_bps"], errors="coerce").to_numpy()
        out["long_minus_short_path_utility_bps_mean"] = _safe_mean(long_u - short_u)
    if "y_long_bad_path" in frame.columns and "y_short_bad_path" in frame.columns:
        long_bad = pd.to_numeric(frame.loc[mask, "y_long_bad_path"], errors="coerce").to_numpy()
        short_bad = pd.to_numeric(frame.loc[mask, "y_short_bad_path"], errors="coerce").to_numpy()
        out["short_minus_long_bad_path_mean"] = _safe_mean(short_bad - long_bad)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    evidence_path = Path(args.evidence_json).expanduser().resolve() if args.evidence_json else _latest_failure_evidence(DEFAULT_RUN_ROOT)
    evidence = _read_json(evidence_path)
    val_data = Path(str(evidence.get("val_data") or "")).expanduser().resolve()
    _require_xau_path(val_data, role="val_data")
    manifest_path = _manifest_for_parquet(val_data)
    manifest = _read_json(manifest_path)
    signal_fields, ctx_cont_names, ctx_cat_names = _manifest_fields(manifest)

    details = (
        ((evidence.get("best_direction_slice_stats") or {}).get("direction_slice_failure_details"))
        if isinstance(evidence.get("best_direction_slice_stats"), dict)
        else None
    )
    if not isinstance(details, list) or not details:
        raise RuntimeError(f"evidence has no best direction slice failure details: {evidence_path}")

    parquet_cols = ["time", "snap", "ctx_cont", "ctx_cat", *[c for c in TARGET_COLUMNS if c != "y_direction"], "y_direction"]
    frame = pd.read_parquet(val_data, columns=parquet_cols)
    snap = _stack_list_column(frame["snap"], np.float32)
    ctx_cat = _stack_list_column(frame["ctx_cat"], np.int64)
    if snap.shape[1] != len(signal_fields):
        raise RuntimeError(f"snap width mismatch: parquet={snap.shape[1]} manifest={len(signal_fields)}")
    if ctx_cat.shape[1] != len(ctx_cat_names):
        raise RuntimeError(f"ctx_cat width mismatch: parquet={ctx_cat.shape[1]} manifest={len(ctx_cat_names)}")
    labels = pd.to_numeric(frame["y_direction"], errors="raise").to_numpy(dtype=np.int64)
    if not set(np.unique(labels)).issubset({0, 1, 2}):
        raise RuntimeError(f"unexpected y_direction classes: {sorted(set(np.unique(labels)))}")

    missing_required = [name for name in REQUIRED_XAU_DIRECTION_FEATURES if name not in signal_fields]
    domain_features = [
        name
        for name in signal_fields[XGB_SIGNAL_FIELD_COUNT:]
        if any(keyword in name.lower() for keyword in DOMAIN_KEYWORDS)
    ]
    for name in REQUIRED_XAU_DIRECTION_FEATURES:
        if name in signal_fields and name not in domain_features:
            domain_features.append(name)
    feature_index = {name: int(signal_fields.index(name)) for name in domain_features if name in signal_fields}

    checks: list[dict[str, Any]] = [
        {
            "name": "evidence decision is fail-closed direction slice guard",
            "ok": evidence.get("decision") == "FAIL_DIRECTION_SLICE_GUARD",
            "details": evidence.get("decision"),
        },
        {
            "name": "val data path is XAU direction repair",
            "ok": "xau_direction_repair" in str(val_data),
            "details": str(val_data),
        },
        {
            "name": "required XAU rail direction features exist",
            "ok": not missing_required,
            "details": {"missing": missing_required, "required": list(REQUIRED_XAU_DIRECTION_FEATURES)},
        },
        {
            "name": "domain feature set is non-empty",
            "ok": bool(domain_features),
            "details": {"count": len(domain_features)},
        },
    ]

    slice_reports: list[dict[str, Any]] = []
    weak_required_feature_slice_count = 0
    for detail in details:
        if not isinstance(detail, dict):
            continue
        ctx_idx = int(detail.get("ctx_cat_index"))
        ctx_value = int(detail.get("ctx_cat_value"))
        if ctx_idx < 0 or ctx_idx >= ctx_cat.shape[1]:
            raise RuntimeError(f"ctx_cat_index {ctx_idx} outside width {ctx_cat.shape[1]}")
        mask = ctx_cat[:, ctx_idx] == ctx_value
        if not bool(mask.any()):
            raise RuntimeError(f"red slice has no matching rows: ctx_cat[{ctx_idx}]={ctx_value}")
        slice_labels = labels[mask]
        summaries: list[dict[str, Any]] = []
        required_summaries: dict[str, Any] = {}
        for name, idx in feature_index.items():
            summary = _feature_summary(snap[mask, idx], slice_labels)
            row = {"feature": name, **summary}
            summaries.append(row)
            if name in REQUIRED_XAU_DIRECTION_FEATURES:
                required_summaries[name] = summary
        summaries.sort(
            key=lambda row: (
                -float(row.get("abs_standardized_long_short") or 0.0),
                row["feature"],
            )
        )
        max_required_abs_std = max(
            [float(row.get("abs_standardized_long_short") or 0.0) for row in required_summaries.values()] or [0.0]
        )
        if max_required_abs_std < float(args.weak_required_feature_std_delta):
            weak_required_feature_slice_count += 1
        slice_reports.append(
            {
                "ctx_cat_index": ctx_idx,
                "ctx_cat_name": ctx_cat_names[ctx_idx],
                "ctx_cat_value": ctx_value,
                "rows": int(mask.sum()),
                "evidence_rows": int(detail.get("rows") or 0),
                "label_rates": _rate_dict(slice_labels),
                "evidence_label_rates": detail.get("label_rates"),
                "evidence_pred_rates": detail.get("pred_rates"),
                "evidence_pred_rate_failed_classes": detail.get("pred_rate_failed_classes"),
                "target_summary": _target_summary(frame, mask),
                "max_required_abs_standardized_long_short": max_required_abs_std,
                "required_feature_summaries": required_summaries,
                "top_domain_feature_summaries": summaries[: int(args.top_features_per_slice)],
            }
        )

    failures = [row for row in checks if not row["ok"]]
    decision = (
        "XAU_RED_SLICE_SEPARABILITY_AUDIT_COMPLETE"
        if not failures
        else "BLOCKED_XAU_RED_SLICE_SEPARABILITY_AUDIT"
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"XAU_RED_SLICE_SEPARABILITY_AUDIT_{created}.json"
    latest_json = out_dir / "XAU_RED_SLICE_SEPARABILITY_AUDIT_latest.json"
    report = {
        "schema_version": "xau_red_slice_separability_audit_v1",
        "created_utc": _utc_now(),
        "decision": decision,
        "report_only": True,
        "training_allowed": False,
        "candidate_training_allowed": False,
        "replay_allowed": False,
        "iql_allowed": False,
        "shadow_live_promotion_allowed": False,
        "side_effects_started": {
            "dataset_rebuild": False,
            "training": False,
            "candidate_training": False,
            "replay": False,
            "iql_distillation": False,
            "shadow": False,
            "live": False,
            "promotion": False,
        },
        "inputs": {
            "evidence_json": str(evidence_path),
            "val_data": str(val_data),
            "manifest_json": str(manifest_path),
        },
        "evidence": {
            "decision": evidence.get("decision"),
            "failure_code": evidence.get("failure_code"),
            "best_epoch": evidence.get("best_epoch"),
            "last_epoch": evidence.get("last_epoch"),
            "best_direction_balance_guard_ok": evidence.get("best_direction_balance_guard_ok"),
            "best_direction_slice_contract_ok": evidence.get("best_direction_slice_contract_ok"),
            "direction_slice_failure_count": (evidence.get("best_direction_slice_stats") or {}).get("direction_slice_failure_count")
            if isinstance(evidence.get("best_direction_slice_stats"), dict)
            else None,
        },
        "feature_policy": {
            "included": ["snap[7:] domain XAU structure features"],
            "excluded_xgb_anchor_count": XGB_SIGNAL_FIELD_COUNT,
            "required_xau_direction_features": list(REQUIRED_XAU_DIRECTION_FEATURES),
            "missing_required_xau_direction_features": missing_required,
            "domain_feature_count": len(domain_features),
            "domain_keywords": list(DOMAIN_KEYWORDS),
            "weak_required_feature_std_delta": float(args.weak_required_feature_std_delta),
        },
        "aggregate": {
            "red_slice_detail_count": len(slice_reports),
            "weak_required_feature_slice_count": weak_required_feature_slice_count,
            "weak_required_feature_slice_rate": (
                float(weak_required_feature_slice_count / len(slice_reports)) if slice_reports else None
            ),
        },
        "checks": checks,
        "failures": failures,
        "slice_reports": slice_reports,
        "next_required_action": (
            "Use this report to decide whether the next repair is label/feature separability "
            "or transformer objective/cap hardening. This audit opens no training, replay, IQL, shadow or live path."
        ),
        "json_path": str(json_path),
    }
    text = json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n"
    json_path.write_text(text, encoding="utf-8")
    latest_json.write_text(text, encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Report-only XAU red-slice separability audit.")
    ap.add_argument("--evidence-json", default="")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--top-features-per-slice", type=int, default=12)
    ap.add_argument("--weak-required-feature-std-delta", type=float, default=0.10)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-fail-on-audit-fail", action="store_true")
    args = ap.parse_args()
    report = run(args)
    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    if report["decision"] != "XAU_RED_SLICE_SEPARABILITY_AUDIT_COMPLETE" and not args.no_fail_on_audit_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
