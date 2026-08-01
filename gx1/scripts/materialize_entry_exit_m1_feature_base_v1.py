"""Materialize the canonical row-level shared feature base at M1.

This producer accepts only an already enriched, causal M1 frame.  It reuses
the exact model-native signal-layer owner used by the Entry dataset builder;
it never copies M5 rows, synthesizes missing fields, or fills absent context.
The upstream enrichment therefore remains a hard input contract, not a hidden
fallback inside this materializer.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.contracts.entry_exit_feature_base_v1 import (
    ENTRY_DECISION_BAR_SECONDS,
    ENTRY_FEATURE_SEQUENCE_BARS,
    EXIT_DECISION_BAR_SECONDS,
    EXIT_FEATURE_SEQUENCE_BARS,
    ENTRY_EXIT_FEATURE_BASE_SCHEMA_VERSION,
    entry_exit_shared_feature_base_contract,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_SIGNAL_DIM,
    require_model_native_manifest,
)
from gx1.contracts.entry_exit_feature_surface_v1 import (
    ENTRY_EXIT_FEATURE_SURFACE_COLUMNS,
    ENTRY_EXIT_FEATURE_SURFACE_SCHEMA_VERSION,
    load_m1_feature_surface,
)
from gx1.contracts.gx1_scope_v1 import require_offline_scope


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _fixed_size_list_column(values: np.ndarray, width: int, *, dtype: pa.DataType) -> pa.Array:
    matrix = np.ascontiguousarray(values)
    if matrix.ndim != 2 or matrix.shape[1] != width:
        raise RuntimeError(
            "M1_FEATURE_BASE_FIXED_LIST_SHAPE_INVALID: "
            f"shape={matrix.shape} width={width}"
        )
    return pa.FixedSizeListArray.from_arrays(
        pa.array(matrix.reshape(-1), type=dtype),
        width,
    )


def _materialize_feature_base(
    *,
    source_parquet: Path,
    seq_structure_manifest: Path,
    output_parquet: Path,
    dataset_run_id: str,
    pair_generation_id: str,
    timeframe: str,
) -> dict[str, Any]:
    if timeframe not in {"M1", "M5"}:
        raise RuntimeError("ENTRY_EXIT_FEATURE_BASE_TIMEFRAME_INVALID")
    bar_seconds = (
        EXIT_DECISION_BAR_SECONDS if timeframe == "M1" else ENTRY_DECISION_BAR_SECONDS
    )
    sequence_bars = (
        EXIT_FEATURE_SEQUENCE_BARS
        if timeframe == "M1"
        else ENTRY_FEATURE_SEQUENCE_BARS
    )
    schema_version = (
        ENTRY_EXIT_FEATURE_SURFACE_SCHEMA_VERSION
        if timeframe == "M1"
        else "gx1_entry_exit_m5_feature_surface_v1"
    )
    require_offline_scope("featurebase_build")
    source = Path(source_parquet).expanduser().resolve()
    manifest_path = Path(seq_structure_manifest).expanduser().resolve()
    output = Path(output_parquet).expanduser().resolve()
    if (
        not source.is_file()
        or source.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.is_symlink()
        or output.exists()
        or output.is_symlink()
        or not output.parent.is_dir()
    ):
        raise RuntimeError("M1_FEATURE_BASE_INPUT_OR_OUTPUT_INVALID")
    if not isinstance(dataset_run_id, str) or not dataset_run_id:
        raise RuntimeError("M1_FEATURE_BASE_DATASET_RUN_ID_INVALID")
    if not isinstance(pair_generation_id, str) or not pair_generation_id:
        raise RuntimeError("M1_FEATURE_BASE_PAIR_GENERATION_ID_INVALID")

    contract = require_model_native_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        context="M1_FEATURE_BASE",
    )
    if tuple(contract["base_fields"]) != MODEL_NATIVE_BASE_FIELDS:
        raise RuntimeError("M1_FEATURE_BASE_BASE_FIELD_ORDER_INVALID")
    if int(contract["seq_input_dim"]) != MODEL_NATIVE_SIGNAL_DIM:
        raise RuntimeError("M1_FEATURE_BASE_SIGNAL_DIM_INVALID")

    frame = pd.read_parquet(source)
    required = {
        "time",
        "open",
        "high",
        "low",
        "close",
        "atr",
        *MODEL_NATIVE_BASE_FIELDS,
        *MODEL_NATIVE_CTX_CONT_FIELDS,
        *MODEL_NATIVE_CTX_CAT_FIELDS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            "M1_FEATURE_BASE_ENRICHED_CAUSAL_FRAME_FIELDS_MISSING: "
            f"{missing[:30]}"
        )
    times = pd.DatetimeIndex(
        pd.to_datetime(frame["time"], utc=True, errors="coerce")
    ).as_unit("ns")
    if (
        len(times) == 0
        or times.hasnans
        or not times.is_unique
        or not times.is_monotonic_increasing
        or not times.floor(f"{bar_seconds}s").equals(times)
    ):
        raise RuntimeError("M1_FEATURE_BASE_TIME_GEOMETRY_INVALID")

    for name in (*MODEL_NATIVE_BASE_FIELDS, *MODEL_NATIVE_CTX_CONT_FIELDS):
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(
            dtype=np.float32
        )
        if not np.isfinite(values).all():
            raise RuntimeError(f"M1_FEATURE_BASE_NONFINITE_FIELD: {name}")
    ctx_cat = frame[list(MODEL_NATIVE_CTX_CAT_FIELDS)].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    if (
        not np.isfinite(ctx_cat).all()
        or not np.equal(ctx_cat, np.rint(ctx_cat)).all()
    ):
        raise RuntimeError("M1_FEATURE_BASE_CTX_CAT_NONINTEGER")
    ctx_cat = ctx_cat.astype(np.int64)

    from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import (
        _build_inline_seq_structure_extension,
    )

    extension, extension_names, extension_meta = (
        _build_inline_seq_structure_extension(
            frame,
            requested_features=list(contract["selected_fields"]),
            ctx_cont_names=list(MODEL_NATIVE_CTX_CONT_FIELDS),
            ctx_cat_names=list(MODEL_NATIVE_CTX_CAT_FIELDS),
            source_parquet=source,
            source_contract_label=f"causal_enriched_{timeframe.lower()}_frame_v1",
            base_signal_fields=list(MODEL_NATIVE_BASE_FIELDS),
        )
    )
    base_signal = frame[list(MODEL_NATIVE_BASE_FIELDS)].astype(
        np.float32
    ).to_numpy()
    signal = np.empty((len(frame), MODEL_NATIVE_SIGNAL_DIM), dtype=np.float32)
    signal[:, : len(MODEL_NATIVE_BASE_FIELDS)] = base_signal
    signal[:, len(MODEL_NATIVE_BASE_FIELDS) :] = extension
    if signal.shape != (len(frame), MODEL_NATIVE_SIGNAL_DIM):
        raise RuntimeError(
            "M1_FEATURE_BASE_SIGNAL_SHAPE_INVALID: "
            f"observed={signal.shape} expected=({len(frame)},{MODEL_NATIVE_SIGNAL_DIM})"
        )
    if tuple([*MODEL_NATIVE_BASE_FIELDS, *extension_names]) != tuple(
        contract["fields"]
    ):
        raise RuntimeError("M1_FEATURE_BASE_SIGNAL_FIELD_ORDER_MISMATCH")
    if not np.isfinite(signal).all():
        raise RuntimeError("M1_FEATURE_BASE_SIGNAL_NONFINITE")

    ctx_cont = frame[list(MODEL_NATIVE_CTX_CONT_FIELDS)].astype(
        np.float32
    ).to_numpy()
    output_table = pa.Table.from_arrays(
        [
            pa.array(times),
            _fixed_size_list_column(signal, MODEL_NATIVE_SIGNAL_DIM, dtype=pa.float32()),
            _fixed_size_list_column(ctx_cont, len(MODEL_NATIVE_CTX_CONT_FIELDS), dtype=pa.float32()),
            _fixed_size_list_column(ctx_cat, len(MODEL_NATIVE_CTX_CAT_FIELDS), dtype=pa.int64()),
        ],
        names=list(ENTRY_EXIT_FEATURE_SURFACE_COLUMNS),
    )
    # Keep live Exit reads bounded: the provider reads complete parquet
    # rowgroups, so one rowgroup is exactly one causal M1 window.
    pq.write_table(
        output_table,
        output,
        row_group_size=sequence_bars,
    )
    manifest = {
        "schema_version": schema_version,
        "decision": "PASS",
        "feature_base_contract_schema_version": ENTRY_EXIT_FEATURE_BASE_SCHEMA_VERSION,
        "shared_feature_base_contract": entry_exit_shared_feature_base_contract(),
        "dataset_run_id": dataset_run_id,
        "pair_generation_id": pair_generation_id,
        "anchor_timeframe": timeframe,
        "source_parquet": str(source),
        "source_sha256": _sha256_file(source),
        "seq_structure_manifest": str(manifest_path),
        "seq_structure_manifest_sha256": _sha256_file(manifest_path),
        "output_parquet": str(output),
        "output_parquet_sha256": _sha256_file(output),
        "rows": int(len(frame)),
        "signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "ctx_cont_dim": len(MODEL_NATIVE_CTX_CONT_FIELDS),
        "ctx_cat_dim": len(MODEL_NATIVE_CTX_CAT_FIELDS),
        "feature_field_order": list(contract["fields"]),
        "extension": extension_meta,
        "causal_contract": {
            "future_rows_used": False,
            "closed_decision_bar_required": True,
            "m1_closed_bar_required": timeframe == "M1",
            "m5_row_reuse": False,
            "duplicate_feature_implementation": False,
        },
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    manifest_path_out = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path_out.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    del output_table, ctx_cont, ctx_cat, signal, extension, base_signal, frame
    gc.collect()
    _loaded_times, _loaded = load_m1_feature_surface(
        output,
        context="M1_FEATURE_BASE_POST_WRITE",
    )
    return manifest


def materialize_m1_feature_base(**kwargs: Any) -> dict[str, Any]:
    return _materialize_feature_base(timeframe="M1", **kwargs)


def materialize_m5_feature_base(**kwargs: Any) -> dict[str, Any]:
    return _materialize_feature_base(timeframe="M5", **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-parquet", required=True, type=Path)
    parser.add_argument("--seq-structure-manifest", required=True, type=Path)
    parser.add_argument("--output-parquet", required=True, type=Path)
    parser.add_argument("--dataset-run-id", required=True)
    parser.add_argument("--pair-generation-id", required=True)
    parser.add_argument("--timeframe", choices=("M1", "M5"), default="M1")
    args = parser.parse_args()
    manifest = _materialize_feature_base(
        source_parquet=args.source_parquet,
        seq_structure_manifest=args.seq_structure_manifest,
        output_parquet=args.output_parquet,
        dataset_run_id=args.dataset_run_id,
        pair_generation_id=args.pair_generation_id,
        timeframe=args.timeframe,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
