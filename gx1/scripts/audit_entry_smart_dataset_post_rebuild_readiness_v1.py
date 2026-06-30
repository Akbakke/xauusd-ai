#!/usr/bin/env python3
"""Audit the dormant smart Entry dataset after a future rebuild.

Report-only. This gate is intentionally downstream of the smart rebuild
preflight and upstream of any smoke/candidate/replay/IQL path. It validates the
rebuilt smart_seq520_candidate dataset contract, writes an audit report, and
keeps every train/replay/IQL/shadow/live side effect closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq

from gx1.features.entry_specialist_feature_groups_v1 import (
    SPECIALIST_AUDIT_CONTRACT_MODES,
    classify_entry_specialist_feature,
    group_features_by_specialist,
    required_training_specialists_for_mode,
)
from gx1.scripts.verify_entry_foundation_state_v1 import FOUNDATION_DATASET_DIR, REPORTS_ROOT


DEFAULT_SMART_MANIFEST = (
    REPORTS_ROOT
    / "entry_specialist_challenger_extension_manifest_20260630_v1"
    / "ENTRY_SPECIALIST_CHALLENGER_SMART_EXTENSION_MANIFEST_latest.json"
)
DEFAULT_SMART_REPORT = (
    REPORTS_ROOT
    / "entry_specialist_challenger_extension_manifest_20260630_v1"
    / "ENTRY_SPECIALIST_CHALLENGER_SMART_EXTENSION_REPORT_latest.json"
)
DEFAULT_DATASET_DIR = FOUNDATION_DATASET_DIR.parent / "v10_dataset_smart_candidate_20260630"
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_smart_dataset_post_rebuild_readiness_20260630_v1"

SPLITS = ("train", "val", "test")
EXPECTED_MANIFEST_VARIANT = "smart_seq520_candidate"
AUDIT_CONTRACT_MODES = tuple(dict.fromkeys((*SPECIALIST_AUDIT_CONTRACT_MODES, EXPECTED_MANIFEST_VARIANT)))
DEFAULT_SEQ_LEN = 96
READY_DECISION = "ENTRY_SMART_DATASET_READY_FOR_TRAIN_READINESS_REVIEW"
BLOCKED_DECISION = "BLOCKED_BY_ENTRY_SMART_DATASET_POST_REBUILD_AUDIT"
NEAR_CONSTANT_STD = 1e-9
MIN_ACTIVE_RATE = 0.01
SIDE_EFFECTS_CLOSED = {
    "dataset_rebuild": False,
    "training": False,
    "replay": False,
    "iql_distillation": False,
    "shadow": False,
    "live": False,
}


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


def _check(name: str, condition: bool, details: Any = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(condition), "details": details if details is not None else {}}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_meta(path: Path, *, compute_hash: bool = True) -> dict[str, Any]:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": bool(exists),
        "size_bytes": int(path.stat().st_size) if exists else None,
        "sha256": _sha256_file(path) if exists and compute_hash else "",
    }


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _is_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _parse_variant_width(variant: str) -> int:
    text = str(variant or "")
    if not text.startswith("smart_seq") or not text.endswith("_candidate"):
        return 0
    raw = text.removeprefix("smart_seq").removesuffix("_candidate")
    return int(raw) if raw.isdigit() else 0


def _selected_features(manifest: dict[str, Any]) -> list[str]:
    return [str(x) for x in manifest.get("selected_features", []) if str(x).strip()]


def _expected_width(smart_manifest: dict[str, Any], smart_report: dict[str, Any]) -> dict[str, Any]:
    counts = smart_report.get("counts") if isinstance(smart_report.get("counts"), dict) else {}
    variant = str(
        smart_manifest.get("manifest_variant")
        or counts.get("manifest_variant")
        or smart_report.get("manifest_variant")
        or ""
    )
    selected = _selected_features(smart_manifest)
    base_count = int(smart_manifest.get("base_signal_feature_count") or counts.get("base_signal_features") or 41)
    selected_count = int(
        smart_manifest.get("selected_feature_count")
        or counts.get("combined_selected_features")
        or len(selected)
    )
    explicit = int(smart_manifest.get("expected_seq_snap_width") or counts.get("expected_seq_snap_width") or 0)
    variant_width = _parse_variant_width(variant)
    expected = explicit or variant_width or base_count + selected_count
    return {
        "manifest_variant": variant,
        "expected_seq_snap_width": int(expected),
        "variant_width": int(variant_width),
        "base_signal_feature_count": int(base_count),
        "selected_feature_count": int(selected_count),
        "selected_feature_count_observed": int(len(selected)),
    }


def _required_specialists_for_audit_mode(mode: str) -> tuple[str, ...]:
    normalized = str(mode or EXPECTED_MANIFEST_VARIANT).strip()
    try:
        return tuple(required_training_specialists_for_mode(normalized))
    except ValueError:
        if normalized == EXPECTED_MANIFEST_VARIANT:
            return tuple(required_training_specialists_for_mode("challenger_seq215"))
        raise


def _walk_source_entries(obj: Any, *, prefix: str = "") -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(obj, dict):
        if "path" in obj and "sha256" in obj:
            yield prefix.rstrip(".") or str(obj.get("label") or "source"), obj
        for key, value in obj.items():
            if isinstance(value, dict):
                child = f"{prefix}{key}."
                yield from _walk_source_entries(value, prefix=child)


def _source_manifest_hash_review(smart_manifest: dict[str, Any]) -> dict[str, Any]:
    source_manifests = smart_manifest.get("source_manifests")
    rows: list[dict[str, Any]] = []
    if not isinstance(source_manifests, dict):
        return {"present": False, "all_recorded_hashes_present": False, "all_observed_hashes_match": False, "rows": rows}
    for label, entry in _walk_source_entries(source_manifests):
        raw_path = str(entry.get("path") or "").strip()
        path = Path(raw_path).expanduser().resolve() if raw_path else None
        recorded = str(entry.get("sha256") or "")
        observed = _sha256_file(path) if path is not None and path.exists() else ""
        rows.append(
            {
                "label": label,
                "path": str(path) if path is not None else "",
                "exists": bool(path is not None and path.exists()),
                "recorded_sha256": recorded,
                "recorded_sha256_present": _is_sha256(recorded),
                "observed_sha256": observed,
                "hash_matches": bool(_is_sha256(recorded) and observed == recorded),
            }
        )
    return {
        "present": True,
        "all_recorded_hashes_present": bool(rows and all(row["recorded_sha256_present"] for row in rows)),
        "all_observed_hashes_match": bool(rows and all(row["hash_matches"] for row in rows)),
        "rows": rows,
    }


def _split_manifest_candidates(dataset_dir: Path, split: str) -> list[Path]:
    return sorted(dataset_dir.glob(f"*_{split}.manifest.json")) if dataset_dir.exists() else []


def _split_parquet_candidates(dataset_dir: Path, split: str) -> list[Path]:
    return sorted(dataset_dir.glob(f"*_{split}.parquet")) if dataset_dir.exists() else []


def _split_manifest_summary(
    dataset_dir: Path,
    split: str,
    *,
    verify_source_parquet_hashes: bool,
) -> dict[str, Any]:
    manifests = _split_manifest_candidates(dataset_dir, split)
    glob_parquets = _split_parquet_candidates(dataset_dir, split)
    manifest_path = manifests[0].resolve() if len(manifests) == 1 else None
    manifest = _read_json_or_empty(manifest_path) if manifest_path is not None else {}
    output_raw = str(manifest.get("output_data_path") or "").strip()
    output_path = (
        Path(output_raw).expanduser().resolve()
        if output_raw
        else (glob_parquets[0].resolve() if len(glob_parquets) == 1 else None)
    )
    extra = manifest.get("extra") if isinstance(manifest.get("extra"), dict) else {}
    signal = extra.get("signal_bridge") if isinstance(extra.get("signal_bridge"), dict) else {}
    extension = (
        signal.get("seq_structure_extension_v1")
        if isinstance(signal.get("seq_structure_extension_v1"), dict)
        else {}
    )
    ctx = extra.get("ctx_contract") if isinstance(extra.get("ctx_contract"), dict) else {}
    base28 = extra.get("base28_manifest") if isinstance(extra.get("base28_manifest"), dict) else {}
    source_raw = str(base28.get("parquet_path") or extension.get("source_parquet_for_price_features") or "").strip()
    source_path = Path(source_raw).expanduser().resolve() if source_raw else None
    recorded_source_sha = str(base28.get("parquet_sha256") or "")
    observed_source_sha = (
        _sha256_file(source_path)
        if verify_source_parquet_hashes and source_path is not None and source_path.exists()
        else ""
    )
    fields = [str(x) for x in signal.get("fields", []) if str(x).strip()]
    extension_features = [str(x) for x in extension.get("features", []) if str(x).strip()]
    splits_meta = manifest.get("splits") if isinstance(manifest.get("splits"), dict) else {}
    ts_meta = manifest.get("ts_min_max_by_split") if isinstance(manifest.get("ts_min_max_by_split"), dict) else {}
    return {
        "split": split,
        "manifest_candidates": [str(path) for path in manifests],
        "parquet_candidates": [str(path) for path in glob_parquets],
        "manifest_count": int(len(manifests)),
        "parquet_candidate_count": int(len(glob_parquets)),
        "manifest_path": str(manifest_path) if manifest_path is not None else "",
        "manifest_exists": bool(manifest_path is not None and manifest_path.exists()),
        "manifest_sha256": _sha256_file(manifest_path) if manifest_path is not None and manifest_path.exists() else "",
        "output_data_path": str(output_path) if output_path is not None else "",
        "output_data_exists": bool(output_path is not None and output_path.exists()),
        "output_data_sha256": "",
        "manifest_output_matches_split_parquet": bool(
            output_path is not None
            and len(glob_parquets) == 1
            and output_path == glob_parquets[0].resolve()
        ),
        "fields": fields,
        "field_count": int(len(fields)),
        "seq_input_dim": int(signal.get("seq_input_dim") or 0),
        "snap_input_dim": int(signal.get("snap_input_dim") or 0),
        "base_seq_input_dim": int(signal.get("base_seq_input_dim") or 0),
        "seq_structure_extension_dim": int(signal.get("seq_structure_extension_dim") or 0),
        "neutral_xgb_bridge": bool(signal.get("neutral_xgb_bridge")),
        "seq_structure_extension": {
            "enabled": bool(extension.get("enabled")),
            "mode": extension.get("mode"),
            "feature_count": int(extension.get("feature_count") or len(extension_features) or 0),
            "manifest_path": str(extension.get("manifest_path") or ""),
            "manifest_selected_feature_count": int(extension.get("manifest_selected_feature_count") or 0),
            "source_parquet_for_price_features": str(extension.get("source_parquet_for_price_features") or ""),
            "features": extension_features,
        },
        "ctx_contract": {
            "tag": str(ctx.get("tag") or ""),
            "ctx_cont_dim": int(ctx.get("ctx_cont_dim") or 0),
            "ctx_cat_dim": int(ctx.get("ctx_cat_dim") or 0),
            "ctx_cont_names": [str(x) for x in ctx.get("ctx_cont_names", []) if str(x).strip()],
            "ctx_cat_names": [str(x) for x in ctx.get("ctx_cat_names", []) if str(x).strip()],
        },
        "source_parquet": str(source_path) if source_path is not None else "",
        "source_parquet_exists": bool(source_path is not None and source_path.exists()),
        "source_parquet_recorded_sha256": recorded_source_sha,
        "source_parquet_recorded_sha256_present": _is_sha256(recorded_source_sha),
        "source_parquet_observed_sha256": observed_source_sha,
        "source_parquet_hash_verified": (
            observed_source_sha == recorded_source_sha
            if verify_source_parquet_hashes and observed_source_sha
            else None
        ),
        "split_window_declared": split in splits_meta,
        "all_split_windows_declared": all(name in splits_meta for name in SPLITS),
        "ts_min_max_declared_for_split": split in ts_meta,
        "raw_manifest": manifest,
    }


def _stack_list_column(values: Iterable[Any], dtype: np.dtype) -> np.ndarray:
    items = list(values)
    if not items:
        return np.asarray([], dtype=dtype)
    try:
        return np.stack(items).astype(dtype, copy=False)
    except ValueError:
        return np.stack([np.stack(item) for item in items]).astype(dtype, copy=False)


class _FeatureStats:
    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        self.rows = 0
        self.finite = np.zeros(self.dim, dtype=np.int64)
        self.sum = np.zeros(self.dim, dtype=np.float64)
        self.sumsq = np.zeros(self.dim, dtype=np.float64)
        self.active = np.zeros(self.dim, dtype=np.int64)

    def add(self, arr: np.ndarray) -> None:
        values = np.asarray(arr, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.dim:
            raise RuntimeError(f"feature stats shape mismatch got={values.shape} expected=(*,{self.dim})")
        finite = np.isfinite(values)
        clean = np.where(finite, values, 0.0)
        self.rows += int(values.shape[0])
        self.finite += finite.sum(axis=0).astype(np.int64)
        self.sum += clean.sum(axis=0)
        self.sumsq += (clean * clean).sum(axis=0)
        self.active += ((np.abs(clean) > 1e-7) & finite).sum(axis=0).astype(np.int64)

    def summarize(self, names: list[str], groups: list[str]) -> dict[str, Any]:
        if self.rows <= 0:
            return {
                "rows": 0,
                "all_finite": False,
                "live_feature_count": 0,
                "live_feature_fraction": 0.0,
                "collapsed_feature_count": int(self.dim),
                "collapsed_features_sample": names[:50],
                "by_specialist": {},
            }
        mean = self.sum / max(self.rows, 1)
        var = np.maximum(self.sumsq / max(self.rows, 1) - mean * mean, 0.0)
        std = np.sqrt(var)
        active_rate = self.active / max(self.rows, 1)
        finite_all = self.finite == self.rows
        live = finite_all & (std > NEAR_CONSTANT_STD) & (active_rate >= MIN_ACTIVE_RATE)
        collapsed = [name for name, ok in zip(names, live, strict=False) if not bool(ok)]
        by_specialist: dict[str, dict[str, Any]] = {}
        for group in sorted(set(groups)):
            idx = [i for i, owner in enumerate(groups) if owner == group]
            live_count = int(np.sum(live[idx])) if idx else 0
            by_specialist[group] = {
                "feature_count": int(len(idx)),
                "live_feature_count": live_count,
                "nonfinite_feature_count": int(np.sum(~finite_all[idx])) if idx else 0,
                "collapsed_feature_count": int(len(idx) - live_count),
            }
        return {
            "rows": int(self.rows),
            "all_finite": bool(np.all(finite_all)),
            "live_feature_count": int(np.sum(live)),
            "live_feature_fraction": float(np.sum(live) / max(self.dim, 1)),
            "collapsed_feature_count": int(len(collapsed)),
            "collapsed_features_sample": collapsed[:50],
            "by_specialist": by_specialist,
        }


def _scan_split_parquet(
    path: Path,
    split: str,
    *,
    expected_width: int,
    expected_seq_len: int,
    expected_ctx_cont_dim: int,
    expected_ctx_cat_dim: int,
    signal_fields: list[str],
    fullscan: bool,
    sample_rows: int,
    batch_size: int,
) -> dict[str, Any]:
    if not path.exists():
        return {"split": split, "path": str(path), "exists": False, "ready": False, "errors": ["missing parquet"]}
    errors: list[str] = []
    schema_names = set(pq.ParquetFile(path).schema_arrow.names)
    required_cols = ["seq", "snap", "ctx_cont", "ctx_cat"]
    missing_cols = [col for col in required_cols if col not in schema_names]
    if missing_cols:
        return {
            "split": split,
            "path": str(path),
            "exists": True,
            "ready": False,
            "errors": [f"missing columns: {missing_cols}"],
            "schema_columns": sorted(schema_names),
        }

    pf = pq.ParquetFile(path)
    total_rows = int(pf.metadata.num_rows or 0)
    scan_limit = total_rows if fullscan else min(total_rows, int(sample_rows))
    stats = _FeatureStats(expected_width)
    groups = [classify_entry_specialist_feature(field) for field in signal_fields]
    scanned = 0
    nonfinite = {"seq": 0, "snap": 0, "ctx_cont": 0, "ctx_cat": 0}
    seq_last_snap_mismatch_rows = 0
    shape_examples: dict[str, Any] = {}
    for batch in pf.iter_batches(batch_size=int(batch_size), columns=required_cols):
        if not fullscan and scanned >= scan_limit:
            break
        pdf = batch.to_pandas()
        if not fullscan:
            remaining = scan_limit - scanned
            pdf = pdf.iloc[:remaining].copy()
        if pdf.empty:
            continue
        seq = _stack_list_column(pdf["seq"], np.float32)
        snap = _stack_list_column(pdf["snap"], np.float32)
        ctx_cont = _stack_list_column(pdf["ctx_cont"], np.float32)
        ctx_cat = _stack_list_column(pdf["ctx_cat"], np.float64)
        if seq.ndim != 3 or seq.shape[1:] != (expected_seq_len, expected_width):
            errors.append(f"seq shape mismatch got={list(seq.shape)} expected=(*,{expected_seq_len},{expected_width})")
        if snap.ndim != 2 or snap.shape[1] != expected_width:
            errors.append(f"snap shape mismatch got={list(snap.shape)} expected=(*,{expected_width})")
        if ctx_cont.ndim != 2 or ctx_cont.shape[1] != expected_ctx_cont_dim:
            errors.append(f"ctx_cont shape mismatch got={list(ctx_cont.shape)} expected=(*,{expected_ctx_cont_dim})")
        if ctx_cat.ndim != 2 or ctx_cat.shape[1] != expected_ctx_cat_dim:
            errors.append(f"ctx_cat shape mismatch got={list(ctx_cat.shape)} expected=(*,{expected_ctx_cat_dim})")
        shape_examples = {
            "seq": list(seq.shape),
            "snap": list(snap.shape),
            "ctx_cont": list(ctx_cont.shape),
            "ctx_cat": list(ctx_cat.shape),
        }
        if errors:
            scanned += int(len(pdf))
            continue
        nonfinite["seq"] += int((~np.isfinite(seq)).sum())
        nonfinite["snap"] += int((~np.isfinite(snap)).sum())
        nonfinite["ctx_cont"] += int((~np.isfinite(ctx_cont)).sum())
        nonfinite["ctx_cat"] += int((~np.isfinite(ctx_cat)).sum())
        seq_last_snap_mismatch_rows += int(np.any(np.abs(seq[:, -1, :] - snap) > 1e-6, axis=1).sum())
        stats.add(snap)
        scanned += int(len(pdf))
    stats_summary = stats.summarize(signal_fields, groups)
    return {
        "split": split,
        "path": str(path),
        "exists": True,
        "total_rows": total_rows,
        "scanned_rows": int(scanned),
        "fullscan": bool(fullscan),
        "scan_complete": bool(scanned == total_rows),
        "shape_examples": shape_examples,
        "nonfinite_counts": nonfinite,
        "all_scanned_values_finite": bool(all(count == 0 for count in nonfinite.values())),
        "seq_last_snap_mismatch_rows": int(seq_last_snap_mismatch_rows),
        "liveness": stats_summary,
        "errors": errors,
        "ready": bool(
            total_rows > 0
            and scanned > 0
            and not errors
            and all(count == 0 for count in nonfinite.values())
            and stats_summary["all_finite"]
        ),
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Smart Dataset Post-Rebuild Readiness",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Dataset dir: `{report['dataset_dir']}`",
        f"- Expected seq/snap width: `{report['expected_contract']['expected_seq_snap_width']}`",
        f"- Scan mode: `{'fullscan' if report['scan_policy']['fullscan'] else 'sample'}`",
        f"- Failures: `{len(report['failures'])}`",
        f"- Training allowed: `{report['training_allowed']}`",
        "- Replay/IQL/shadow/live allowed: `false`",
        "",
        "## Failures",
        "",
    ]
    if report["failures"]:
        lines.extend([f"- `{failure['check']}`" for failure in report["failures"]])
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    smart_manifest_path = Path(args.smart_manifest).expanduser().resolve()
    smart_report_path = Path(args.smart_report).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    smart_manifest = _read_json_or_empty(smart_manifest_path)
    smart_report = _read_json_or_empty(smart_report_path)
    expected = _expected_width(smart_manifest, smart_report)
    expected_width = int(expected["expected_seq_snap_width"])
    selected_features = _selected_features(smart_manifest)
    contract_mode = str(args.contract_mode or EXPECTED_MANIFEST_VARIANT)
    required_specialists = _required_specialists_for_audit_mode(contract_mode)

    split_manifests = {
        split: _split_manifest_summary(
            dataset_dir,
            split,
            verify_source_parquet_hashes=bool(args.verify_source_parquet_hashes),
        )
        for split in SPLITS
    }
    first_fields = next((row["fields"] for row in split_manifests.values() if row["fields"]), [])
    groups = group_features_by_specialist(first_fields)
    signal_group_counts = {group: int(len(features)) for group, features in groups.items()}
    unmapped_fields = list(groups.get("unmapped", []))
    selected_missing_by_split = {
        split: [feature for feature in selected_features if feature not in set(row["fields"])]
        for split, row in split_manifests.items()
    }
    source_manifest_review = _source_manifest_hash_review(smart_manifest)

    scans: dict[str, dict[str, Any]] = {}
    for split, row in split_manifests.items():
        output = Path(row["output_data_path"]).expanduser() if row["output_data_path"] else Path("")
        ctx = row["ctx_contract"]
        scans[split] = _scan_split_parquet(
            output,
            split,
            expected_width=expected_width,
            expected_seq_len=int(args.expected_seq_len),
            expected_ctx_cont_dim=int(ctx.get("ctx_cont_dim") or 0),
            expected_ctx_cat_dim=int(ctx.get("ctx_cat_dim") or 0),
            signal_fields=row["fields"],
            fullscan=bool(args.fullscan),
            sample_rows=int(args.sample_rows),
            batch_size=int(args.batch_size),
        ) if row["output_data_path"] else {
            "split": split,
            "path": "",
            "exists": False,
            "ready": False,
            "errors": ["missing output_data_path"],
        }

    specialist_liveness: dict[str, dict[str, Any]] = {}
    for split, scan in scans.items():
        by_specialist = ((scan.get("liveness") or {}).get("by_specialist") or {})
        specialist_liveness[split] = {
            specialist: by_specialist.get(
                specialist,
                {
                    "feature_count": signal_group_counts.get(specialist, 0),
                    "live_feature_count": 0,
                    "nonfinite_feature_count": 0,
                    "collapsed_feature_count": signal_group_counts.get(specialist, 0),
                },
            )
            for specialist in required_specialists
        }

    checks: list[dict[str, Any]] = [
        _check("smart dataset directory exists", dataset_dir.exists(), {"dataset_dir": str(dataset_dir)}),
        _check("smart manifest exists", smart_manifest_path.exists(), _artifact_meta(smart_manifest_path)),
        _check(
            "smart manifest variant is smart_seq520_candidate",
            expected["manifest_variant"] == EXPECTED_MANIFEST_VARIANT,
            expected,
        ),
        _check(
            "smart manifest expected width is 520",
            expected_width == 520 and expected["variant_width"] == 520,
            expected,
        ),
        _check(
            "smart manifest selected feature count matches width formula",
            expected["base_signal_feature_count"] + expected["selected_feature_count_observed"] == expected_width,
            expected,
        ),
        _check(
            "smart source manifest hashes are present and match observed files",
            bool(source_manifest_review["all_recorded_hashes_present"] and source_manifest_review["all_observed_hashes_match"]),
            source_manifest_review,
        ),
        _check("exact train/val/test split manifests exist", all(row["manifest_count"] == 1 for row in split_manifests.values()), split_manifests),
        _check("exact train/val/test split parquet candidates exist", all(row["parquet_candidate_count"] == 1 for row in split_manifests.values()), split_manifests),
        _check("split manifest output_data_path exists", all(row["output_data_exists"] for row in split_manifests.values()), split_manifests),
        _check(
            "split manifest output_data_path matches split parquet",
            all(row["manifest_output_matches_split_parquet"] for row in split_manifests.values()),
            split_manifests,
        ),
        _check(
            "split manifests declare train/val/test windows and per-split ts bounds",
            all(row["all_split_windows_declared"] and row["ts_min_max_declared_for_split"] for row in split_manifests.values()),
            split_manifests,
        ),
    ]
    for split, row in split_manifests.items():
        ext = row["seq_structure_extension"]
        checks.extend(
            [
                _check(f"{split} seq/snap dims equal smart manifest expected width", row["seq_input_dim"] == expected_width and row["snap_input_dim"] == expected_width, row),
                _check(f"{split} signal field count equals expected width", row["field_count"] == expected_width, row),
                _check(f"{split} smart extension dim equals selected feature count", row["seq_structure_extension_dim"] == expected["selected_feature_count_observed"], row),
                _check(f"{split} smart extension selected features are present", not selected_missing_by_split[split], {"missing": selected_missing_by_split[split][:50], "missing_count": len(selected_missing_by_split[split])}),
                _check(f"{split} smart extension records source manifest count", ext["manifest_selected_feature_count"] == expected["selected_feature_count_observed"], ext),
                _check(f"{split} source parquet recorded sha256 is present", row["source_parquet_recorded_sha256_present"], row),
                _check(f"{split} source parquet exists", row["source_parquet_exists"], row),
                _check(f"{split} neutral xgb bridge remains declared", row["neutral_xgb_bridge"] is True, row),
            ]
        )
        if args.verify_source_parquet_hashes:
            checks.append(
                _check(f"{split} source parquet observed hash matches recorded", row["source_parquet_hash_verified"] is True, row)
            )
    checks.extend(
        [
            _check("all splits share identical signal field order", all(row["fields"] == first_fields for row in split_manifests.values()), {"field_counts": {split: row["field_count"] for split, row in split_manifests.items()}}),
            _check("specialist routing maps every signal field", not unmapped_fields, {"unmapped_fields": unmapped_fields[:50], "unmapped_count": len(unmapped_fields)}),
            _check("specialist routing covers required challenger specialists", all(signal_group_counts.get(group, 0) > 0 for group in required_specialists), {"required_specialists": list(required_specialists), "signal_group_counts": signal_group_counts}),
            _check("parquet scan loaded finite seq/snap/ctx sample", all(scan.get("ready") for scan in scans.values()), scans),
            _check(
                "liveness/non-collapse covers required specialists in every split",
                all(
                    int(specialist_liveness[split][group].get("live_feature_count") or 0) >= int(args.min_live_features_per_specialist)
                    for split in SPLITS
                    for group in required_specialists
                ),
                specialist_liveness,
            ),
            _check(
                "overall live signal fraction meets minimum",
                all(float((scan.get("liveness") or {}).get("live_feature_fraction") or 0.0) >= float(args.min_live_feature_fraction) for scan in scans.values()),
                {split: (scan.get("liveness") or {}) for split, scan in scans.items()},
            ),
            _check("side effects remain closed", all(value is False for value in SIDE_EFFECTS_CLOSED.values()), SIDE_EFFECTS_CLOSED),
        ]
    )

    failures = [{"check": check["name"], "details": check.get("details") or {}} for check in checks if not check["ok"]]
    ready = not failures
    decision = READY_DECISION if ready else BLOCKED_DECISION
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"ENTRY_SMART_DATASET_POST_REBUILD_READINESS_{timestamp}.json"
    md_path = out_dir / f"ENTRY_SMART_DATASET_POST_REBUILD_READINESS_{timestamp}.md"
    report = {
        "schema_version": "entry_smart_dataset_post_rebuild_readiness_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "report_only": True,
        "dataset_dir": str(dataset_dir),
        "smart_manifest": _artifact_meta(smart_manifest_path),
        "smart_report": _artifact_meta(smart_report_path) if smart_report_path.exists() else _artifact_meta(smart_report_path, compute_hash=False),
        "expected_contract": expected,
        "contract_mode": contract_mode,
        "required_training_specialists": list(required_specialists),
        "split_manifests": {
            split: {key: value for key, value in row.items() if key not in {"raw_manifest", "fields"}}
            for split, row in split_manifests.items()
        },
        "source_manifest_hash_review": source_manifest_review,
        "signal_routing": {
            "field_count": int(len(first_fields)),
            "unmapped_count": int(len(unmapped_fields)),
            "unmapped_fields": unmapped_fields,
            "group_counts": signal_group_counts,
        },
        "scan_policy": {
            "fullscan": bool(args.fullscan),
            "sample_rows": int(args.sample_rows),
            "batch_size": int(args.batch_size),
            "verify_source_parquet_hashes": bool(args.verify_source_parquet_hashes),
        },
        "split_scans": scans,
        "specialist_liveness": specialist_liveness,
        "checks": checks,
        "failures": failures,
        "training_allowed": False,
        "candidate_training_allowed": False,
        "replay_allowed": False,
        "iql_allowed": False,
        "shadow_live_promotion_allowed": False,
        "side_effects_started": dict(SIDE_EFFECTS_CLOSED),
        "next_required_gate": (
            "separate smart train-readiness review; no smoke/candidate/replay/IQL/shadow/live is opened by this audit"
            if ready
            else "run the explicit smart dataset rebuild and repair this post-rebuild audit before any train-readiness review"
        ),
        "json_path": str(json_path),
        "md_path": str(md_path),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    (out_dir / "ENTRY_SMART_DATASET_POST_REBUILD_READINESS_latest.json").write_text(
        json_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out_dir / "ENTRY_SMART_DATASET_POST_REBUILD_READINESS_latest.md").write_text(
        md_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    if not args.quiet:
        print(json.dumps({"decision": decision, "failures": failures, "json_path": str(json_path)}, indent=2, sort_keys=True, default=_json_default))
    if args.fail_on_not_ready and failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    ap.add_argument("--smart-manifest", default=str(DEFAULT_SMART_MANIFEST))
    ap.add_argument("--smart-report", default=str(DEFAULT_SMART_REPORT))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--contract-mode", choices=AUDIT_CONTRACT_MODES, default=EXPECTED_MANIFEST_VARIANT)
    ap.add_argument("--expected-seq-len", type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--sample-rows", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--fullscan", action="store_true")
    ap.add_argument("--verify-source-parquet-hashes", action="store_true")
    ap.add_argument("--min-live-features-per-specialist", type=int, default=1)
    ap.add_argument("--min-live-feature-fraction", type=float, default=0.05)
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
