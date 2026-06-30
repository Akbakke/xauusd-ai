#!/usr/bin/env python3
"""Audit Entry candlestick-pattern challenger features before any training."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gx1.features.entry_candlestick_patterns_v1 import (
    CANDLESTICK_PATTERN_FEATURE_NAMES,
    CANDLESTICK_PATTERN_FEATURE_VERSION,
    CANDLESTICK_PATTERN_SOURCE_FIELDS,
    build_entry_candlestick_pattern_layer,
    missing_candlestick_source_fields,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    REQUIRED_TRAINING_SPECIALISTS,
    classify_entry_specialist_feature,
)
from gx1.scripts.audit_entry_chart_geometry_challenger_v1 import NumericAccumulator, _liveness_failures
from gx1.scripts.evaluate_entry_selective_edge_v1 import _split_files
from gx1.scripts.experiment_entry_chart_structure_ablation_v1 import DEFAULT_SOURCE_PARQUET
from gx1.scripts.verify_entry_foundation_state_v1 import FOUNDATION_DATASET_DIR, REPORTS_ROOT


DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_candlestick_pattern_challenger_audit_20260630_v1"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    return str(obj)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _load_source_features(source_parquet: Path) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    schema = pq.read_schema(source_parquet)
    columns = list(schema.names)
    missing = missing_candlestick_source_fields(columns)
    if missing:
        raise RuntimeError(f"candlestick source fields missing: {missing} source={source_parquet}")
    source = pd.read_parquet(source_parquet, columns=list(CANDLESTICK_PATTERN_SOURCE_FIELDS), engine="pyarrow")
    source["time"] = pd.to_datetime(source["time"], utc=True)
    source = source.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    features, names = build_entry_candlestick_pattern_layer(source)
    feature_df = pd.DataFrame(features, columns=names)
    feature_df["time"] = source["time"].to_numpy()
    feature_df = feature_df.set_index("time")
    meta = {
        "file": _file_metadata(source_parquet),
        "row_count": int(len(source)),
        "schema_column_count": int(len(columns)),
        "schema_columns_sha256": _sha256_text(json.dumps(columns, sort_keys=True)),
        "source_fields": list(CANDLESTICK_PATTERN_SOURCE_FIELDS),
        "time_min": str(source["time"].min()) if len(source) else None,
        "time_max": str(source["time"].max()) if len(source) else None,
    }
    return feature_df, names, meta


def _audit_split(
    *,
    split: str,
    parquet_path: Path,
    source_features: pd.DataFrame,
    feature_names: list[str],
    max_rows: int,
    liveness_epsilon: float,
    near_constant_std: float,
    min_active_rate: float,
    min_active_count: int,
) -> dict[str, Any]:
    columns = ["time"]
    schema_names = pq.read_schema(parquet_path).names
    if "y_direction" in schema_names:
        columns.append("y_direction")
    df = pd.read_parquet(parquet_path, columns=columns, engine="pyarrow")
    if int(max_rows) > 0:
        df = df.iloc[: int(max_rows)].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    aligned = source_features.reindex(df["time"]).fillna(0.0)
    matrix = aligned[feature_names].to_numpy(np.float32)
    acc = NumericAccumulator(feature_names, liveness_epsilon=float(liveness_epsilon))
    acc.add(matrix)
    rows = acc.rows(
        split=split,
        near_constant_std=float(near_constant_std),
        min_active_rate=float(min_active_rate),
        min_active_count=int(min_active_count),
        name_key="feature",
    )
    label_counts: dict[str, int] = {}
    if "y_direction" in df.columns:
        labels = df["y_direction"].astype(int).to_numpy()
        uniq, counts = np.unique(labels, return_counts=True)
        label_counts = {str(int(k)): int(v) for k, v in zip(uniq, counts, strict=False)}
    return {
        "split": split,
        "path": str(parquet_path),
        "rows": int(len(df)),
        "max_rows_cap": int(max_rows),
        "time_min": str(df["time"].min()) if len(df) else None,
        "time_max": str(df["time"].max()) if len(df) else None,
        "label_counts": label_counts,
        "feature_liveness": rows,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Candlestick Pattern Challenger Audit",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Dataset: `{report['dataset_dir']}`",
        f"- Source parquet: `{report['source_parquet']}`",
        f"- Features: `{report['feature_count']}`",
        f"- Failure count: `{len(report['failures'])}`",
        "",
        "## Failures",
        "",
    ]
    lines.extend([f"- {failure}" for failure in report["failures"]] or ["- None"])
    lines.extend(["", "## Features", ""])
    lines.extend(f"- `{name}`" for name in report["features"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    source_parquet = Path(args.source_parquet).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = [part.strip() for part in str(args.data_splits).split(",") if part.strip()]
    files = _split_files(dataset_dir, splits)

    failures: list[str] = []
    source_features, feature_names, source_meta = _load_source_features(source_parquet)
    if tuple(feature_names) != tuple(CANDLESTICK_PATTERN_FEATURE_NAMES):
        failures.append("candlestick feature order/count differs from repo constant")
    routing = {name: classify_entry_specialist_feature(name) for name in feature_names}
    misrouted = {name: group for name, group in routing.items() if group != "price_action_candle_encoder"}
    if misrouted:
        failures.append(f"candlestick features misrouted: {dict(list(misrouted.items())[:20])} total={len(misrouted)}")

    split_reports: dict[str, Any] = {}
    liveness_rows: list[dict[str, Any]] = []
    for split in splits:
        try:
            row = _audit_split(
                split=split,
                parquet_path=Path(files[split]),
                source_features=source_features,
                feature_names=feature_names,
                max_rows=int(args.max_rows_per_split),
                liveness_epsilon=float(args.liveness_epsilon),
                near_constant_std=float(args.near_constant_std),
                min_active_rate=float(args.min_active_rate),
                min_active_count=int(args.min_active_count),
            )
            split_reports[split] = {k: v for k, v in row.items() if k != "feature_liveness"}
            liveness_rows.extend(row["feature_liveness"])
        except Exception as exc:
            failures.append(f"{split}: candlestick audit failed: {type(exc).__name__}: {exc}")
    failures.extend(_liveness_failures(liveness_rows, id_key="feature", row_type="candlestick pattern feature"))

    trainable_now = "price_action_candle_encoder" in set(REQUIRED_TRAINING_SPECIALISTS)
    decision = "READY_FOR_CHALLENGER_DATASET_REBUILD" if not failures else "FAIL"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "schema_version": "entry_candlestick_pattern_challenger_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "dataset_dir": str(dataset_dir),
        "source_parquet": str(source_parquet),
        "candlestick_pattern_feature_version": CANDLESTICK_PATTERN_FEATURE_VERSION,
        "selected_features": list(feature_names),
        "source_fields": list(CANDLESTICK_PATTERN_SOURCE_FIELDS),
        "specialist": "price_action_candle_encoder",
        "trainable_in_current_contract": bool(trainable_now),
        "current_required_training_specialists": list(REQUIRED_TRAINING_SPECIALISTS),
        "dataset_rebuild_required_before_training": True,
        "activation_or_training_allowed": False,
        "shadow_live_promotion_allowed": False,
    }
    report = {
        "schema_version": "entry_candlestick_pattern_challenger_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "dataset_dir": str(dataset_dir),
        "source_parquet": str(source_parquet),
        "source_provenance": source_meta,
        "data_splits": splits,
        "feature_version": CANDLESTICK_PATTERN_FEATURE_VERSION,
        "feature_count": int(len(feature_names)),
        "features": list(feature_names),
        "source_fields": list(CANDLESTICK_PATTERN_SOURCE_FIELDS),
        "specialist": "price_action_candle_encoder",
        "trainable_in_current_contract": bool(trainable_now),
        "dataset_rebuild_required_before_training": True,
        "activation_or_training_allowed": False,
        "shadow_live_promotion_allowed": False,
        "split_summaries": split_reports,
        "feature_liveness": liveness_rows,
        "routing": routing,
        "failures": failures,
        "challenger_manifest": manifest,
    }
    json_path = out_dir / f"ENTRY_CANDLESTICK_PATTERN_CHALLENGER_AUDIT_{timestamp}.json"
    md_path = out_dir / f"ENTRY_CANDLESTICK_PATTERN_CHALLENGER_AUDIT_{timestamp}.md"
    manifest_path = out_dir / f"ENTRY_CANDLESTICK_PATTERN_CHALLENGER_MANIFEST_{timestamp}.json"
    report["json_path"] = str(json_path)
    report["md_path"] = str(md_path)
    report["manifest_path"] = str(manifest_path)
    manifest["manifest_path"] = str(manifest_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    (out_dir / "ENTRY_CANDLESTICK_PATTERN_CHALLENGER_AUDIT_latest.json").write_text(
        json_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (out_dir / "ENTRY_CANDLESTICK_PATTERN_CHALLENGER_AUDIT_latest.md").write_text(
        md_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (out_dir / "ENTRY_CANDLESTICK_PATTERN_CHALLENGER_MANIFEST_latest.json").write_text(
        manifest_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": decision,
                    "feature_count": len(feature_names),
                    "failures": failures,
                    "json_path": str(json_path),
                    "manifest_path": str(manifest_path),
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
    ap.add_argument("--dataset-dir", default=str(FOUNDATION_DATASET_DIR))
    ap.add_argument("--source-parquet", default=str(DEFAULT_SOURCE_PARQUET))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--data-splits", default="train,val,test")
    ap.add_argument("--max-rows-per-split", type=int, default=0, help="0 means all split rows.")
    ap.add_argument("--liveness-epsilon", type=float, default=1e-7)
    ap.add_argument("--near-constant-std", type=float, default=1e-10)
    ap.add_argument("--min-active-rate", type=float, default=0.0)
    ap.add_argument("--min-active-count", type=int, default=1)
    ap.add_argument("--fail-on-audit-fail", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
