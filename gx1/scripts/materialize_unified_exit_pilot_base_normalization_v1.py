#!/usr/bin/env python3
"""Fit canonical base feature normalization on physical child TRAIN rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gx1.contracts.entry_model_native_input_normalization_v1 import (
    CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS,
    EXPECTED_SURFACES,
    EXPECTED_TFS,
    MTF_SEMANTIC_CATEGORICAL_DOMAINS,
    SIGNAL_SEMANTIC_CATEGORICAL_DOMAINS,
    MatrixPopulationPart,
    build_input_normalization_contract,
    fit_ctx_cat_contract,
    fit_surface_normalization,
    share_temporal_alias_stats_from_signal,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
)
from gx1.features.htf_features import (
    MULTI_TF_PER_BAR_FEATURES_V4,
    MULTI_TF_SHIFT,
    load_multi_tf_v4_cache,
)
from gx1.models.entry_v10.entry_v10_input_normalization import (
    _derive_temporal_aliases,
    select_shared_causal_mtf_fit_population,
)

from gx1.scripts.materialize_unified_exit_pilot_normalization_inputs_v1 import (
    _canonical_sha256, _clock_hash, _prefix_entry_rows,
)


SCHEMA_VERSION = "gx1_unified_exit_pilot_base_normalization_v1"
PER_TF_SEQ_LENS = {"M5": 16, "M15": 64, "H1": 96, "H4": 96, "D1": 252}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError("PILOT_BASE_NORMALIZATION_JSON_INVALID")
    return value


def _matrix(table: Any, name: str, dtype: str) -> np.ndarray:
    column = table[name].combine_chunks()
    flat = column.values.to_numpy(zero_copy_only=False)
    if len(column) < 1 or len(flat) % len(column) != 0:
        raise RuntimeError("PILOT_BASE_NORMALIZATION_LIST_SHAPE_INVALID")
    width = len(flat) // len(column)
    return np.ascontiguousarray(flat.reshape(len(column), width), dtype=dtype)


def fit_base(
    *,
    child_admission_path: Path,
    normalization_view_path: Path,
    population_witness_path: Path,
    m5_feature_path: Path,
    m5_prebuilt_path: Path,
    m1_feature_path: Path,
    mtf_cache_dir: Path,
    output_path: Path,
    publish: bool,
    fit_entry_rows_path: Path | None = None,
    fit_cutoff_time_ns: int | None = None,
) -> dict[str, Any]:
    admission = _json(child_admission_path)
    view = _json(normalization_view_path)
    population = _json(population_witness_path)
    prefix = fit_entry_rows_path is not None
    scope = population.get("normalization_fit_population")
    if (prefix != (fit_cutoff_time_ns is not None) or prefix != (scope is not None)
            or scope != view.get("normalization_fit_population")
            or (prefix and (type(fit_cutoff_time_ns) is not int or fit_cutoff_time_ns <= 0
                            or scope.get("cutoff_time_ns") != fit_cutoff_time_ns))):
        raise RuntimeError("PILOT_BASE_NORMALIZATION_PREFIX_SCOPE_INVALID")
    if prefix and (
            population["contract_sha256"] != _canonical_sha256({k:v for k,v in population.items() if k != "contract_sha256"})
            or view["train_normalization_population_witness"]["file_sha256"] != _sha(population_witness_path)):
        raise RuntimeError("PILOT_BASE_NORMALIZATION_PREFIX_WITNESS_INVALID")
    child = admission["splits"]["train"]
    parent_manifest_path = Path(view["parent_train_manifest"]["path"])
    if _sha(parent_manifest_path) != view["parent_train_manifest"]["sha256"]:
        raise RuntimeError("PILOT_BASE_NORMALIZATION_PARENT_MANIFEST_INVALID")
    parent_manifest = _json(parent_manifest_path)
    signal_fields = tuple(parent_manifest["feature_contract"]["signal_bridge_fields"])
    if (
        admission.get("decision") != "PASS"
        or view.get("decision") != "PASS"
        or population.get("decision") != "PASS"
        or view["train_normalization_population_witness"]["contract_sha256"] != population["contract_sha256"]
        or population["val_fit_rows"] != 0
        or population["test_fit_rows"] != 0
    ):
        raise RuntimeError("PILOT_BASE_NORMALIZATION_PARENT_INVALID")
    entry_table = pq.read_table(child["parquet_path"], columns=["time", "snap", "ctx_cont", "ctx_cat"])
    entry_times = pd.DatetimeIndex(entry_table["time"].to_pandas()).as_unit("ns")
    entry_snap = _matrix(entry_table, "snap", "<f4")
    entry_ctx = _matrix(entry_table, "ctx_cont", "<f4")
    entry_cat = _matrix(entry_table, "ctx_cat", "<i8")
    if prefix:
        rows, rows_binding = _prefix_entry_rows(fit_entry_rows_path, len(entry_times))
        if scope["entry_rows"] != rows_binding:
            raise RuntimeError("PILOT_BASE_NORMALIZATION_PREFIX_ROWS_BINDING_INVALID")
        entry_times = entry_times[rows]
        entry_snap, entry_ctx, entry_cat = entry_snap[rows], entry_ctx[rows], entry_cat[rows]
        if (population["train_entry_decision_rows"] != len(rows)
                or population["train_entry_clock_sha256"] != _clock_hash(entry_times.asi8)):
            raise RuntimeError("PILOT_BASE_NORMALIZATION_PREFIX_ENTRY_CLOCK_INVALID")
    m5_table = pq.read_table(m5_feature_path, columns=["time", "signal"])
    m5_times = pd.DatetimeIndex(m5_table["time"].to_pandas()).as_unit("ns")
    m5_signal = _matrix(m5_table, "signal", "<f4")
    m1_table = pq.read_table(m1_feature_path, columns=["time", "signal", "ctx_cont", "ctx_cat"])
    m1_times = pd.DatetimeIndex(m1_table["time"].to_pandas()).as_unit("ns")
    m1_signal = _matrix(m1_table, "signal", "<f4")
    m1_ctx = _matrix(m1_table, "ctx_cont", "<f4")
    m1_cat = _matrix(m1_table, "ctx_cat", "<i8")
    entry_positions = np.searchsorted(m5_times.asi8, entry_times.asi8)
    if (
        np.any(entry_positions >= len(m5_times))
        or not np.array_equal(m5_times.asi8[entry_positions], entry_times.asi8)
        or not np.array_equal(m5_signal[entry_positions], entry_snap)
    ):
        raise RuntimeError("PILOT_BASE_NORMALIZATION_ENTRY_MAPPING_INVALID")
    entry_indices = np.concatenate([np.arange(item["start_row"], item["end_row_exclusive"], dtype=np.int64) for item in population["entry_m5_local_intervals"]])
    child_m1_times = pd.DatetimeIndex(pq.read_table(population["m1_source"]["path"], columns=["time"])["time"].to_pandas()).as_unit("ns")
    child_current = np.concatenate([np.arange(item["start_row"], item["end_row_exclusive"], dtype=np.int64) for item in population["exit_m1_current_intervals"]])
    current = np.searchsorted(m1_times.asi8, child_m1_times.asi8[child_current])
    if np.any(current >= len(m1_times)) or not np.array_equal(m1_times.asi8[current], child_m1_times.asi8[child_current]):
        raise RuntimeError("PILOT_BASE_NORMALIZATION_M1_MAPPING_INVALID")
    if prefix and (not len(current) or not len(entry_indices)
            or int(entry_times.asi8.max()) + 300_000_000_000 > fit_cutoff_time_ns
            or int(m5_times.asi8[entry_indices].max()) + 300_000_000_000 > fit_cutoff_time_ns
            or int(m1_times.asi8[current].max()) + 60_000_000_000 > fit_cutoff_time_ns):
        raise RuntimeError("PILOT_BASE_NORMALIZATION_PREFIX_CUTOFF_EXCEEDED")
    expanded = [(max(0, int(left) - 479), int(right)) for left, right in zip(current, current + 1)]
    expanded.sort()
    merged: list[list[int]] = []
    for left, right in expanded:
        if not merged or left > merged[-1][1]:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    local = np.concatenate([np.arange(left, right, dtype=np.int64) for left, right in merged])
    aliases = _derive_temporal_aliases(signal_fields)
    signal_parts = [MatrixPopulationPart(m5_signal, row_indices=entry_indices, source="entry_m5"), MatrixPopulationPart(m1_signal, row_indices=local, source="exit_m1")]
    ctx_parts = [MatrixPopulationPart(entry_ctx, source="entry"), MatrixPopulationPart(m1_ctx, row_indices=current, source="exit")]
    cat_parts = [MatrixPopulationPart(entry_cat, source="entry"), MatrixPopulationPart(m1_cat, row_indices=current, source="exit")]
    signal_surface = fit_surface_normalization(signal_parts, surface="signal", field_names=signal_fields, row_count=len(entry_indices) + len(local), semantic_categorical_domains=SIGNAL_SEMANTIC_CATEGORICAL_DOMAINS, allow_constant_train_fields=prefix)
    ctx_raw = fit_surface_normalization(ctx_parts, surface="ctx_cont", field_names=MODEL_NATIVE_CTX_CONT_FIELDS, row_count=len(entry_times) + len(current), semantic_categorical_domains=CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS, allow_constant_train_fields=prefix)
    ctx_surface = share_temporal_alias_stats_from_signal(ctx_raw, signal_surface, temporal_aliases=aliases, ctx_cont_values=ctx_parts)
    surfaces: dict[str, Any] = {"signal": signal_surface, "ctx_cont": ctx_surface}
    cache = load_multi_tf_v4_cache(mtf_cache_dir)
    windows: dict[str, Any] = {}
    for tf in EXPECTED_TFS:
        source = cache[tf]
        selected, window, _proof = select_shared_causal_mtf_fit_population(tf=tf, source=source, entry_train_times_ns=np.asarray(entry_times.asi8), exit_train_times_ns=np.asarray(m1_times.asi8[current]), seq_len=PER_TF_SEQ_LENS[tf])
        surfaces[f"mtf_{tf.lower()}"] = fit_surface_normalization(selected, surface=f"mtf_{tf.lower()}", field_names=MULTI_TF_PER_BAR_FEATURES_V4, row_count=window["selected_unique_row_count"], semantic_categorical_domains=MTF_SEMANTIC_CATEGORICAL_DOMAINS, allow_constant_train_fields=prefix)
        windows[tf] = window
    if tuple(surfaces) != EXPECTED_SURFACES:
        raise RuntimeError("PILOT_BASE_NORMALIZATION_SURFACE_ORDER_INVALID")
    cache_manifest = mtf_cache_dir / "manifest.json"
    cache_raw = _json(cache_manifest)
    train_min = min(int(entry_times.asi8[0]), int(m1_times.asi8[current[0]]))
    train_max = max(int(entry_times.asi8[-1]), int(m1_times.asi8[current[-1]]))
    lineage = {
        "dataset_run_id": admission["child_dataset_run_id"],
        "train_parquet_path": child["parquet_path"], "train_parquet_sha256": child["parquet_sha256"],
        "train_manifest_path": child["manifest_path"], "train_manifest_sha256": child["manifest_file_sha256"],
        "train_row_count": len(entry_times) + len(current), "entry_train_decision_row_count": len(entry_times), "exit_train_decision_row_count": len(current),
        "local_fit_row_count": len(entry_indices) + len(local), "context_fit_row_count": len(entry_times) + len(current),
        "val_fit_row_count": 0, "test_fit_row_count": 0,
        "train_time_min_utc": pd.Timestamp(train_min, tz="UTC").isoformat(), "train_time_max_utc": pd.Timestamp(train_max, tz="UTC").isoformat(),
        "m5_prebuilt_path": str(m5_prebuilt_path), "m5_prebuilt_sha256": _sha(m5_prebuilt_path),
        "mtf_cache_manifest_path": str(cache_manifest), "mtf_cache_manifest_sha256": _sha(cache_manifest),
        "mtf_builder_version": cache_raw["builder_version"],
        "mtf_feature_names_sha256": hashlib.sha256(json.dumps(list(cache_raw["feature_names"]), sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "per_tf_seq_lens": PER_TF_SEQ_LENS,
        "per_tf_shift_seconds": {tf: int(MULTI_TF_SHIFT[tf].total_seconds()) for tf in EXPECTED_TFS},
        "per_tf_fit_windows": windows,
    }
    contract = build_input_normalization_contract(fit_start_utc=lineage["train_time_min_utc"], fit_end_utc=lineage["train_time_max_utc"], surfaces=surfaces, ctx_cat=fit_ctx_cat_contract(cat_parts, field_names=MODEL_NATIVE_CTX_CAT_FIELDS), lineage=lineage, temporal_aliases=aliases)
    result = {"schema_version": SCHEMA_VERSION, "decision": "PASS", "contract": contract, "contract_sha256": contract["contract_sha256"], "population_witness_sha256": population["contract_sha256"], "val_fit_rows": 0, "test_fit_rows": 0, "test_accessed": False}
    if prefix:
        result["normalization_fit_population"] = scope
    if publish:
        if output_path.exists():
            raise RuntimeError("PILOT_BASE_NORMALIZATION_OUTPUT_EXISTS")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
        with os.fdopen(fd, "w") as handle:
            json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(tmp, output_path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("child-admission", "normalization-view", "population-witness", "m5-feature", "m5-prebuilt", "m1-feature", "mtf-cache-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--fit-entry-rows", type=Path)
    parser.add_argument("--fit-cutoff-time-ns", type=int)
    args = parser.parse_args(argv)
    result = fit_base(child_admission_path=args.child_admission.resolve(), normalization_view_path=args.normalization_view.resolve(), population_witness_path=args.population_witness.resolve(), m5_feature_path=args.m5_feature.resolve(), m5_prebuilt_path=args.m5_prebuilt.resolve(), m1_feature_path=args.m1_feature.resolve(), mtf_cache_dir=args.mtf_cache_dir.resolve(), output_path=args.output.resolve(), publish=args.publish, fit_entry_rows_path=args.fit_entry_rows, fit_cutoff_time_ns=args.fit_cutoff_time_ns)
    print(json.dumps({"decision": result["decision"], "contract_sha256": result["contract_sha256"], "published": args.publish}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
