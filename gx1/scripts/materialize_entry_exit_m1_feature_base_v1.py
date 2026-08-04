"""Materialize native M1/M5 surfaces through one feature-owner implementation.

This producer accepts an already enriched causal frame at one explicit native
clock. It reuses the exact model-native signal-layer owner used by the Entry
dataset builder; it never copies values across M1/M5, synthesizes missing
fields, fills absent context, or resamples computed M1 features upward. M1 also
receives the exact closed-M1 source timeline used by the Exit lifecycle.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.contracts.entry_exit_feature_base_v1 import (  # noqa: E402
    ENTRY_DECISION_BAR_SECONDS,
    ENTRY_FEATURE_SEQUENCE_BARS,
    EXIT_DECISION_BAR_SECONDS,
    EXIT_FEATURE_SEQUENCE_BARS,
    ENTRY_EXIT_ENRICHED_CAUSAL_FRAME_SCHEMA_VERSION,
    ENTRY_EXIT_FEATURE_BASE_SCHEMA_VERSION,
    entry_exit_shared_feature_base_contract,
    require_entry_exit_shared_feature_base_contract,
)
from gx1.contracts.entry_model_native_signal_v1 import (  # noqa: E402
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_SIGNAL_DIM,
    require_model_native_manifest,
)
from gx1.contracts.entry_exit_feature_surface_v1 import (  # noqa: E402
    ENTRY_EXIT_M5_FEATURE_SURFACE_SCHEMA_VERSION,
    ENTRY_EXIT_FEATURE_SURFACE_COLUMNS,
    ENTRY_EXIT_FEATURE_SURFACE_SCHEMA_VERSION,
    load_m1_feature_surface,
)
from gx1.contracts.gx1_scope_v1 import require_offline_scope  # noqa: E402


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


def _load_enriched_source_binding(
    source: Path,
    *,
    dataset_run_id: str,
    pair_generation_id: str,
    timeframe: str,
) -> dict[str, Any]:
    """Require the exact producer sidecar; caller labels cannot relabel bytes."""

    source_manifest = source.with_suffix(source.suffix + ".manifest.json")
    if source_manifest.is_symlink() or not source_manifest.is_file():
        raise RuntimeError("ENTRY_EXIT_FEATURE_BASE_SOURCE_MANIFEST_MISSING")
    try:
        payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "ENTRY_EXIT_FEATURE_BASE_SOURCE_MANIFEST_INVALID"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("ENTRY_EXIT_FEATURE_BASE_SOURCE_MANIFEST_INVALID")

    declared_manifest_sha = payload.get("manifest_sha256")
    unhashed = dict(payload)
    unhashed.pop("manifest_sha256", None)
    source_sha = _sha256_file(source)
    expected_bar_seconds = (
        EXIT_DECISION_BAR_SECONDS
        if timeframe == "M1"
        else ENTRY_DECISION_BAR_SECONDS
    )
    if (
        payload.get("schema_version")
        != ENTRY_EXIT_ENRICHED_CAUSAL_FRAME_SCHEMA_VERSION
        or payload.get("decision") != "PASS"
        or payload.get("dataset_run_id") != dataset_run_id
        or payload.get("pair_generation_id") != pair_generation_id
        or payload.get("timeframe") != timeframe
        or payload.get("base_bar_seconds") != expected_bar_seconds
        or payload.get("output_parquet") != str(source)
        or payload.get("output_parquet_sha256") != source_sha
        or declared_manifest_sha != _canonical_sha256(unhashed)
    ):
        raise RuntimeError("ENTRY_EXIT_FEATURE_BASE_SOURCE_LINEAGE_INVALID")
    require_entry_exit_shared_feature_base_contract(
        payload.get("shared_feature_base_contract"),
        context="ENTRY_EXIT_FEATURE_BASE_SOURCE",
    )
    rank_path = Path(str(payload.get("rank_reference_npz") or ""))
    rank_sha = str(payload.get("rank_reference_sha256") or "")
    if (
        not rank_path.is_absolute()
        or rank_path.is_symlink()
        or not rank_path.is_file()
        or len(rank_sha) != 64
        or _sha256_file(rank_path) != rank_sha
    ):
        raise RuntimeError(
            "ENTRY_EXIT_FEATURE_BASE_SOURCE_RANK_LINEAGE_INVALID"
        )
    return {
        "manifest_path": str(source_manifest),
        "manifest_sha256": _sha256_file(source_manifest),
        "schema_version": ENTRY_EXIT_ENRICHED_CAUSAL_FRAME_SCHEMA_VERSION,
        "source_sha256": source_sha,
        "rank_reference_npz": str(rank_path),
        "rank_reference_sha256": rank_sha,
    }


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
    alignment_parquet: Path | None,
    seq_structure_manifest: Path,
    output_parquet: Path,
    dataset_run_id: str,
    pair_generation_id: str,
    timeframe: str,
) -> dict[str, Any]:
    if timeframe not in {"M1", "M5"}:
        raise RuntimeError("ENTRY_EXIT_FEATURE_BASE_TIMEFRAME_INVALID")
    if timeframe == "M1" and alignment_parquet is None:
        raise RuntimeError("M1_FEATURE_BASE_ALIGNMENT_SOURCE_REQUIRED")
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
        else ENTRY_EXIT_M5_FEATURE_SURFACE_SCHEMA_VERSION
    )
    require_offline_scope("featurebase_build")
    source = Path(source_parquet).expanduser().resolve()
    alignment = (
        None
        if alignment_parquet is None
        else Path(alignment_parquet).expanduser().resolve()
    )
    manifest_path = Path(seq_structure_manifest).expanduser().resolve()
    output = Path(output_parquet).expanduser().resolve()
    if (
        not source.is_file()
        or source.is_symlink()
        or (
            alignment is not None
            and (
                not alignment.is_file()
                or alignment.is_symlink()
            )
        )
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

    source_binding = _load_enriched_source_binding(
        source,
        dataset_run_id=dataset_run_id,
        pair_generation_id=pair_generation_id,
        timeframe=timeframe,
    )

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

    alignment_times: pd.DatetimeIndex | None = None
    if alignment is not None:
        alignment_times = pd.DatetimeIndex(
            pd.to_datetime(
                pd.read_parquet(alignment, columns=["time"])["time"],
                utc=True,
                errors="coerce",
            )
        ).as_unit("ns")
        if (
            len(alignment_times) == 0
            or alignment_times.hasnans
            or not alignment_times.is_unique
            or not alignment_times.is_monotonic_increasing
            or not alignment_times.floor(f"{bar_seconds}s").equals(
                alignment_times
            )
        ):
            raise RuntimeError("M1_FEATURE_BASE_ALIGNMENT_TIME_GEOMETRY_INVALID")
        positions = times.get_indexer(alignment_times)
        if np.any(positions < 0):
            missing = int(np.count_nonzero(positions < 0))
            raise RuntimeError(
                "M1_FEATURE_BASE_ALIGNMENT_TIME_NOT_SUBSET: "
                f"missing_rows={missing}"
            )
        # Keep the exact source timeline.  The enriched producer may retain a
        # causal warmup/history superset, but Exit must consume the same closed
        # M1 rows as its pair-bound lifecycle source.
        frame = frame.iloc[positions].reset_index(drop=True)
        times = alignment_times

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
        "source_sha256": source_binding["source_sha256"],
        "source_manifest": source_binding["manifest_path"],
        "source_manifest_sha256": source_binding["manifest_sha256"],
        "source_manifest_schema_version": source_binding["schema_version"],
        "rank_reference_npz": source_binding["rank_reference_npz"],
        "rank_reference_sha256": source_binding["rank_reference_sha256"],
        "alignment_parquet": None if alignment is None else str(alignment),
        "alignment_sha256": (
            None if alignment is None else _sha256_file(alignment)
        ),
        "seq_structure_manifest": str(manifest_path),
        "seq_structure_manifest_sha256": _sha256_file(manifest_path),
        "output_parquet": str(output),
        "output_parquet_sha256": _sha256_file(output),
        "rows": int(len(frame)),
        "signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "ctx_cont_dim": len(MODEL_NATIVE_CTX_CONT_FIELDS),
        "ctx_cat_dim": len(MODEL_NATIVE_CTX_CAT_FIELDS),
        "feature_field_order": list(contract["fields"]),
        "feature_field_order_sha256": _canonical_sha256(
            list(contract["fields"])
        ),
        "extension": extension_meta,
        "causal_contract": {
            "future_rows_used": False,
            "closed_decision_bar_required": True,
            "m1_closed_bar_required": timeframe == "M1",
            "m5_row_reuse": False,
            "native_resolution_values": True,
            "cross_resolution_value_copy": False,
            "computed_m1_feature_resampling": False,
            "duplicate_feature_implementation": False,
            "exact_closed_source_timestamp_subset": alignment is not None,
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
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.validation.",
        dir=str(output.parent),
    ) as validation_storage:
        _loaded_times, _loaded = load_m1_feature_surface(
            output,
            context=f"{timeframe}_FEATURE_BASE_POST_WRITE",
            storage_dir=Path(validation_storage),
            expected_bar_seconds=bar_seconds,
        )
        del _loaded_times, _loaded
        gc.collect()
    return manifest


def materialize_m1_feature_base(**kwargs: Any) -> dict[str, Any]:
    return _materialize_feature_base(timeframe="M1", **kwargs)


def materialize_m5_feature_base(**kwargs: Any) -> dict[str, Any]:
    return _materialize_feature_base(timeframe="M5", **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-parquet", required=True, type=Path)
    parser.add_argument("--alignment-parquet", type=Path)
    parser.add_argument("--seq-structure-manifest", required=True, type=Path)
    parser.add_argument("--output-parquet", required=True, type=Path)
    parser.add_argument("--dataset-run-id", required=True)
    parser.add_argument("--pair-generation-id", required=True)
    parser.add_argument("--timeframe", choices=("M1", "M5"), default="M1")
    args = parser.parse_args()
    manifest = _materialize_feature_base(
        source_parquet=args.source_parquet,
        alignment_parquet=args.alignment_parquet,
        seq_structure_manifest=args.seq_structure_manifest,
        output_parquet=args.output_parquet,
        dataset_run_id=args.dataset_run_id,
        pair_generation_id=args.pair_generation_id,
        timeframe=args.timeframe,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
