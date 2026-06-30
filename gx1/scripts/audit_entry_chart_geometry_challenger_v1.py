#!/usr/bin/env python3
"""Audit Entry chart-geometry challenger features before any training.

This is report-only. It proves that the numeric support/resistance,
Fibonacci, trendline, EMA-cross and chart-pattern proxy layer can be built from
the active foundation dataset, has live source fields, routes to the challenger
specialist, and carries enough provenance to justify a later dataset rebuild.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq

from gx1.audit.entry_transformer_feature_audit import _stack_list_column
from gx1.features.entry_chart_geometry_v1 import (
    CHART_GEOMETRY_FEATURE_NAMES,
    CHART_GEOMETRY_FEATURE_PREFIX,
    CHART_GEOMETRY_FEATURE_VERSION,
    CHART_GEOMETRY_SOURCE_FIELDS,
    build_entry_chart_geometry_layer,
    missing_chart_geometry_source_fields,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    REQUIRED_TRAINING_SPECIALISTS,
    classify_entry_specialist_feature,
)
from gx1.scripts.evaluate_entry_selective_edge_v1 import _split_files
from gx1.scripts.verify_entry_foundation_state_v1 import FOUNDATION_DATASET_DIR, REPORTS_ROOT


DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_chart_geometry_challenger_audit_20260630_v1"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_manifest_path(parquet_path: Path) -> Path:
    return parquet_path.with_suffix(".manifest.json")


def _load_emitted_contract(parquet_path: Path) -> dict[str, Any]:
    manifest_path = _split_manifest_path(parquet_path)
    if not manifest_path.exists():
        raise RuntimeError(f"split manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    extra = manifest.get("extra") or {}
    signal_bridge = extra.get("signal_bridge") or {}
    ctx_contract = extra.get("ctx_contract") or {}
    signal_fields = [str(x) for x in signal_bridge.get("fields", [])]
    ctx_cont_names = [str(x) for x in ctx_contract.get("ctx_cont_names", [])]
    if not signal_fields:
        raise RuntimeError(f"signal fields missing from split manifest: {manifest_path}")
    if not ctx_cont_names:
        raise RuntimeError(f"ctx_cont names missing from split manifest: {manifest_path}")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "signal_fields": signal_fields,
        "ctx_cont_names": ctx_cont_names,
        "seq_input_dim": int(signal_bridge.get("seq_input_dim") or len(signal_fields)),
        "seq_structure_extension_dim": int(signal_bridge.get("seq_structure_extension_dim") or 0),
        "neutral_xgb_bridge": bool(signal_bridge.get("neutral_xgb_bridge") or extra.get("neutral_xgb_bridge")),
        "seq_structure_extension_v1": signal_bridge.get("seq_structure_extension_v1") or {},
    }


@dataclass
class NumericAccumulator:
    names: list[str]
    liveness_epsilon: float

    def __post_init__(self) -> None:
        dim = len(self.names)
        self.n = 0
        self.finite = np.zeros(dim, dtype=np.int64)
        self.nonfinite = np.zeros(dim, dtype=np.int64)
        self.zero = np.zeros(dim, dtype=np.int64)
        self.active = np.zeros(dim, dtype=np.int64)
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sumsq = np.zeros(dim, dtype=np.float64)
        self.min = np.full(dim, np.inf, dtype=np.float64)
        self.max = np.full(dim, -np.inf, dtype=np.float64)

    def add(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != len(self.names):
            raise RuntimeError(f"numeric shape mismatch: got={arr.shape} expected=(*,{len(self.names)})")
        finite = np.isfinite(arr)
        clean = np.where(finite, arr, 0.0)
        self.n += int(arr.shape[0])
        self.finite += finite.sum(axis=0).astype(np.int64)
        self.nonfinite += (~finite).sum(axis=0).astype(np.int64)
        self.zero += ((clean == 0.0) & finite).sum(axis=0).astype(np.int64)
        self.active += ((np.abs(clean) > float(self.liveness_epsilon)) & finite).sum(axis=0).astype(np.int64)
        self.sum += clean.sum(axis=0)
        self.sumsq += (clean * clean).sum(axis=0)
        if arr.shape[0]:
            finite_or_nan = np.where(finite, arr, np.nan)
            with np.errstate(all="ignore"):
                batch_min = np.nanmin(finite_or_nan, axis=0)
                batch_max = np.nanmax(finite_or_nan, axis=0)
            self.min = np.minimum(self.min, np.where(np.isfinite(batch_min), batch_min, self.min))
            self.max = np.maximum(self.max, np.where(np.isfinite(batch_max), batch_max, self.max))

    def rows(
        self,
        *,
        split: str,
        near_constant_std: float,
        min_active_rate: float,
        min_active_count: int,
        name_key: str = "feature",
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        n = max(int(self.n), 1)
        mean = self.sum / n
        var = np.maximum(self.sumsq / n - mean * mean, 0.0)
        std = np.sqrt(var)
        for i, name in enumerate(self.names):
            finite_rate = float(self.finite[i]) / n
            active_rate = float(self.active[i]) / n
            near_constant = bool(std[i] <= float(near_constant_std))
            out.append(
                {
                    "split": split,
                    name_key: str(name),
                    "n": int(self.n),
                    "finite_rate": finite_rate,
                    "nonfinite_count": int(self.nonfinite[i]),
                    "zero_rate": float(self.zero[i]) / n,
                    "active_count": int(self.active[i]),
                    "active_rate": active_rate,
                    "mean": float(mean[i]),
                    "std": float(std[i]),
                    "min": float(self.min[i]) if np.isfinite(self.min[i]) else 0.0,
                    "max": float(self.max[i]) if np.isfinite(self.max[i]) else 0.0,
                    "near_constant": near_constant,
                    "live": bool(
                        self.n > 0
                        and int(self.nonfinite[i]) == 0
                        and not near_constant
                        and int(self.active[i]) >= int(min_active_count)
                        and active_rate >= float(min_active_rate)
                    ),
                }
            )
        return out


def _batch_column(batch: Any, name: str) -> list[Any]:
    idx = batch.schema.get_field_index(name)
    if idx < 0:
        raise RuntimeError(f"batch lacks required column: {name}")
    return batch.column(idx).to_pylist()


def _source_matrix(
    *,
    snap: np.ndarray,
    ctx_cont: np.ndarray,
    signal_fields: list[str],
    ctx_cont_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    if snap.ndim != 2 or snap.shape[1] != len(signal_fields):
        raise RuntimeError(f"snap shape {list(snap.shape)} incompatible with {len(signal_fields)} signal fields")
    if ctx_cont.ndim != 2 or ctx_cont.shape[1] != len(ctx_cont_names):
        raise RuntimeError(f"ctx_cont shape {list(ctx_cont.shape)} incompatible with {len(ctx_cont_names)} ctx fields")
    x = np.concatenate([snap, ctx_cont], axis=1).astype(np.float32, copy=False)
    names = [f"snap.{name}" for name in signal_fields] + [f"ctx_cont.{name}" for name in ctx_cont_names]
    return x, names


def _select_source_columns(x: np.ndarray, names: list[str], source_fields: Iterable[str]) -> np.ndarray:
    idx = {name: i for i, name in enumerate(names)}
    cols = []
    missing = []
    for source_field in source_fields:
        i = idx.get(str(source_field))
        if i is None:
            missing.append(str(source_field))
        else:
            cols.append(i)
    if missing:
        raise RuntimeError(f"source columns missing from matrix: {missing[:30]} total={len(missing)}")
    return x[:, cols].astype(np.float32, copy=False)


def _label_counts(values: list[Any]) -> dict[str, int]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.int64)
    uniq, counts = np.unique(arr, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(uniq, counts, strict=False)}


def _merge_counts(dst: dict[str, int], src: dict[str, int]) -> None:
    for key, value in src.items():
        dst[key] = int(dst.get(key, 0)) + int(value)


def _audit_split(
    *,
    split: str,
    parquet_path: Path,
    contract: dict[str, Any],
    batch_size: int,
    max_rows: int,
    liveness_epsilon: float,
    near_constant_std: float,
    min_generated_active_rate: float,
    min_generated_active_count: int,
    min_source_active_rate: float,
    min_source_active_count: int,
) -> dict[str, Any]:
    pf = pq.ParquetFile(parquet_path)
    available_columns = set(pf.schema_arrow.names)
    required_columns = ["snap", "ctx_cont"]
    optional_columns = [name for name in ("time", "y_direction") if name in available_columns]
    columns = required_columns + optional_columns

    feature_acc = NumericAccumulator(list(CHART_GEOMETRY_FEATURE_NAMES), liveness_epsilon=liveness_epsilon)
    source_acc = NumericAccumulator(list(CHART_GEOMETRY_SOURCE_FIELDS), liveness_epsilon=liveness_epsilon)
    feature_names_seen: list[str] | None = None
    previous_x: np.ndarray | None = None
    rows_seen = 0
    label_counts: dict[str, int] = {}
    time_min: str | None = None
    time_max: str | None = None

    signal_fields = [str(x) for x in contract["signal_fields"]]
    ctx_cont_names = [str(x) for x in contract["ctx_cont_names"]]

    for batch in pf.iter_batches(batch_size=int(batch_size), columns=columns):
        if int(max_rows) > 0 and rows_seen >= int(max_rows):
            break
        snap = _stack_list_column(_batch_column(batch, "snap"), np.float32)
        ctx_cont = _stack_list_column(_batch_column(batch, "ctx_cont"), np.float32)
        if int(max_rows) > 0:
            remaining = int(max_rows) - rows_seen
            snap = snap[:remaining]
            ctx_cont = ctx_cont[:remaining]
        if snap.size == 0:
            continue
        x, names = _source_matrix(
            snap=snap,
            ctx_cont=ctx_cont,
            signal_fields=signal_fields,
            ctx_cont_names=ctx_cont_names,
        )
        source_acc.add(_select_source_columns(x, names, CHART_GEOMETRY_SOURCE_FIELDS))

        if previous_x is not None:
            build_x = np.vstack([previous_x, x])
            out, out_names = build_entry_chart_geometry_layer(build_x, names)
            out = out[1:]
        else:
            out, out_names = build_entry_chart_geometry_layer(x, names)
        previous_x = x[-1:].copy()
        if feature_names_seen is None:
            feature_names_seen = list(out_names)
        elif list(out_names) != feature_names_seen:
            raise RuntimeError(f"{split}: generated feature order changed within stream")
        feature_acc.add(out)

        if "y_direction" in optional_columns:
            raw_labels = _batch_column(batch, "y_direction")
            if int(max_rows) > 0:
                raw_labels = raw_labels[: x.shape[0]]
            _merge_counts(label_counts, _label_counts(raw_labels))
        if "time" in optional_columns:
            raw_times = [str(x) for x in _batch_column(batch, "time")[: x.shape[0]]]
            if raw_times:
                batch_min = min(raw_times)
                batch_max = max(raw_times)
                time_min = batch_min if time_min is None else min(time_min, batch_min)
                time_max = batch_max if time_max is None else max(time_max, batch_max)

        rows_seen += int(x.shape[0])

    if rows_seen <= 0:
        raise RuntimeError(f"{split}: no rows audited from {parquet_path}")

    generated_rows = feature_acc.rows(
        split=split,
        near_constant_std=near_constant_std,
        min_active_rate=min_generated_active_rate,
        min_active_count=min_generated_active_count,
        name_key="feature",
    )
    source_rows = source_acc.rows(
        split=split,
        near_constant_std=near_constant_std,
        min_active_rate=min_source_active_rate,
        min_active_count=min_source_active_count,
        name_key="source_field",
    )
    return {
        "split": split,
        "path": str(parquet_path),
        "parquet_sha256": _sha256_file(parquet_path),
        "source_rows": int(pf.metadata.num_rows),
        "audited_rows": int(rows_seen),
        "max_rows_cap": int(max_rows),
        "time_min": time_min,
        "time_max": time_max,
        "label_counts": label_counts,
        "generated_feature_names": feature_names_seen or [],
        "generated_feature_liveness": generated_rows,
        "source_field_liveness": source_rows,
    }


def _liveness_failures(
    rows: list[dict[str, Any]],
    *,
    id_key: str,
    row_type: str,
) -> list[str]:
    failures: list[str] = []
    for row in rows:
        name = str(row.get(id_key))
        split = str(row.get("split"))
        if int(row.get("nonfinite_count") or 0) > 0:
            failures.append(f"{split}: {row_type} has non-finite values: {name} nonfinite={row.get('nonfinite_count')}")
        if bool(row.get("near_constant")):
            failures.append(f"{split}: {row_type} is near-constant: {name} std={row.get('std')}")
        if not bool(row.get("live")):
            failures.append(
                f"{split}: {row_type} is not live: {name} "
                f"active_count={row.get('active_count')} active_rate={row.get('active_rate')}"
            )
    return failures


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Chart Geometry Challenger Audit",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Dataset: `{report['dataset_dir']}`",
        f"- Generated features: `{report['generated_feature_count']}`",
        f"- Source fields: `{report['source_field_count']}`",
        f"- Failure count: `{len(report['failures'])}`",
        "",
        "## Failures",
        "",
    ]
    if report["failures"]:
        lines.extend(f"- {failure}" for failure in report["failures"])
    else:
        lines.append("- None")
    lines.extend(["", "## Split Summary", ""])
    for row in report["split_summaries"]:
        lines.append(
            f"- `{row['split']}`: audited_rows={row['audited_rows']} "
            f"source_rows={row['source_rows']} time={row.get('time_min')}..{row.get('time_max')}"
        )
    lines.extend(["", "## Contract", ""])
    lines.append(f"- Feature version: `{report['chart_geometry_feature_version']}`")
    lines.append(f"- Specialist: `{report['specialist']}`")
    lines.append(f"- Current trainable specialist: `{report['trainable_in_current_contract']}`")
    lines.append(f"- Dataset rebuild required before training: `{report['dataset_rebuild_required_before_training']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = [part.strip() for part in str(args.data_splits).split(",") if part.strip()]
    files = _split_files(dataset_dir, splits)

    failures: list[str] = []
    contracts: dict[str, Any] = {}
    split_reports: dict[str, Any] = {}
    generated_liveness: list[dict[str, Any]] = []
    source_liveness: list[dict[str, Any]] = []
    split_summaries: list[dict[str, Any]] = []

    for split in splits:
        parquet_path = Path(files[split])
        try:
            contract = _load_emitted_contract(parquet_path)
            contracts[split] = {
                "manifest_path": contract["manifest_path"],
                "manifest_sha256": contract["manifest_sha256"],
                "signal_field_count": int(len(contract["signal_fields"])),
                "ctx_cont_field_count": int(len(contract["ctx_cont_names"])),
                "seq_input_dim": int(contract["seq_input_dim"]),
                "seq_structure_extension_dim": int(contract["seq_structure_extension_dim"]),
                "neutral_xgb_bridge": bool(contract["neutral_xgb_bridge"]),
            }
            source_universe = [f"snap.{name}" for name in contract["signal_fields"]] + [
                f"ctx_cont.{name}" for name in contract["ctx_cont_names"]
            ]
            missing_sources = missing_chart_geometry_source_fields(source_universe)
            contracts[split]["chart_geometry_source_missing"] = missing_sources
            contracts[split]["chart_geometry_source_missing_count"] = int(len(missing_sources))
            if missing_sources:
                failures.append(f"{split}: chart geometry source fields missing: {missing_sources[:30]} total={len(missing_sources)}")
                continue
            split_report = _audit_split(
                split=split,
                parquet_path=parquet_path,
                contract=contract,
                batch_size=int(args.batch_size),
                max_rows=int(args.max_rows_per_split),
                liveness_epsilon=float(args.liveness_epsilon),
                near_constant_std=float(args.near_constant_std),
                min_generated_active_rate=float(args.min_generated_active_rate),
                min_generated_active_count=int(args.min_generated_active_count),
                min_source_active_rate=float(args.min_source_active_rate),
                min_source_active_count=int(args.min_source_active_count),
            )
            split_reports[split] = split_report
            generated_liveness.extend(split_report["generated_feature_liveness"])
            source_liveness.extend(split_report["source_field_liveness"])
            split_summaries.append(
                {
                    key: split_report.get(key)
                    for key in (
                        "split",
                        "path",
                        "parquet_sha256",
                        "source_rows",
                        "audited_rows",
                        "max_rows_cap",
                        "time_min",
                        "time_max",
                        "label_counts",
                    )
                }
            )
            if tuple(split_report["generated_feature_names"]) != tuple(CHART_GEOMETRY_FEATURE_NAMES):
                failures.append(f"{split}: generated chart geometry feature order/count mismatch")
        except Exception as exc:
            failures.append(f"{split}: chart geometry audit failed: {type(exc).__name__}: {exc}")

    routed = {
        name: classify_entry_specialist_feature(name)
        for name in CHART_GEOMETRY_FEATURE_NAMES
    }
    misrouted = {name: group for name, group in routed.items() if group != "chart_geometry_encoder"}
    if misrouted:
        failures.append(f"chart geometry features misrouted: {dict(list(misrouted.items())[:20])} total={len(misrouted)}")
    if not all(str(name).startswith(CHART_GEOMETRY_FEATURE_PREFIX) for name in CHART_GEOMETRY_FEATURE_NAMES):
        failures.append("chart geometry feature prefix contract failed")

    failures.extend(_liveness_failures(generated_liveness, id_key="feature", row_type="generated chart geometry feature"))
    failures.extend(_liveness_failures(source_liveness, id_key="source_field", row_type="chart geometry source field"))

    trainable_now = "chart_geometry_encoder" in set(REQUIRED_TRAINING_SPECIALISTS)
    decision = "READY_FOR_CHALLENGER_DATASET_REBUILD" if not failures else "FAIL"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "schema_version": "entry_chart_geometry_challenger_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "dataset_dir": str(dataset_dir),
        "chart_geometry_feature_version": CHART_GEOMETRY_FEATURE_VERSION,
        "selected_features": list(CHART_GEOMETRY_FEATURE_NAMES),
        "source_fields": list(CHART_GEOMETRY_SOURCE_FIELDS),
        "specialist": "chart_geometry_encoder",
        "trainable_in_current_contract": bool(trainable_now),
        "current_required_training_specialists": list(REQUIRED_TRAINING_SPECIALISTS),
        "dataset_rebuild_required_before_training": True,
        "activation_or_training_allowed": False,
        "shadow_live_promotion_allowed": False,
    }

    report = {
        "schema_version": "entry_chart_geometry_challenger_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "dataset_dir": str(dataset_dir),
        "data_splits": splits,
        "chart_geometry_feature_version": CHART_GEOMETRY_FEATURE_VERSION,
        "generated_feature_count": int(len(CHART_GEOMETRY_FEATURE_NAMES)),
        "generated_features": list(CHART_GEOMETRY_FEATURE_NAMES),
        "source_field_count": int(len(CHART_GEOMETRY_SOURCE_FIELDS)),
        "source_fields": list(CHART_GEOMETRY_SOURCE_FIELDS),
        "specialist": "chart_geometry_encoder",
        "trainable_in_current_contract": bool(trainable_now),
        "dataset_rebuild_required_before_training": True,
        "activation_or_training_allowed": False,
        "shadow_live_promotion_allowed": False,
        "contracts": contracts,
        "split_summaries": split_summaries,
        "generated_feature_liveness": generated_liveness,
        "source_field_liveness": source_liveness,
        "routing": routed,
        "failures": failures,
        "challenger_manifest": manifest,
    }

    json_path = out_dir / f"ENTRY_CHART_GEOMETRY_CHALLENGER_AUDIT_{timestamp}.json"
    md_path = out_dir / f"ENTRY_CHART_GEOMETRY_CHALLENGER_AUDIT_{timestamp}.md"
    manifest_path = out_dir / f"ENTRY_CHART_GEOMETRY_CHALLENGER_MANIFEST_{timestamp}.json"
    latest_json = out_dir / "ENTRY_CHART_GEOMETRY_CHALLENGER_AUDIT_latest.json"
    latest_md = out_dir / "ENTRY_CHART_GEOMETRY_CHALLENGER_AUDIT_latest.md"
    latest_manifest = out_dir / "ENTRY_CHART_GEOMETRY_CHALLENGER_MANIFEST_latest.json"
    report["json_path"] = str(json_path)
    report["md_path"] = str(md_path)
    report["manifest_path"] = str(manifest_path)
    manifest["manifest_path"] = str(manifest_path)

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_manifest.write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")

    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "generated_feature_count": report["generated_feature_count"],
                    "source_field_count": report["source_field_count"],
                    "failures": report["failures"],
                    "json_path": report["json_path"],
                    "manifest_path": report["manifest_path"],
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
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--data-splits", default="train,val,test")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--max-rows-per-split", type=int, default=0, help="0 means stream the full split.")
    ap.add_argument("--liveness-epsilon", type=float, default=1e-7)
    ap.add_argument("--near-constant-std", type=float, default=1e-10)
    ap.add_argument("--min-generated-active-rate", type=float, default=0.00001)
    ap.add_argument("--min-generated-active-count", type=int, default=1)
    ap.add_argument("--min-source-active-rate", type=float, default=0.0001)
    ap.add_argument("--min-source-active-count", type=int, default=1)
    ap.add_argument("--fail-on-audit-fail", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
