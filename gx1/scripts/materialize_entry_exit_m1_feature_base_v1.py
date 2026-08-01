"""Materialize the canonical row-level shared feature base at M1.

This producer accepts only an already enriched, causal M1 frame.  It reuses
the exact model-native signal-layer owner used by the Entry dataset builder;
it never copies M5 rows, synthesizes missing fields, or fills absent context.
The upstream enrichment therefore remains a hard input contract, not a hidden
fallback inside this materializer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_exit_feature_base_v1 import (
    EXIT_DECISION_BAR_SECONDS,
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


def materialize_m1_feature_base(
    *,
    source_parquet: Path,
    seq_structure_manifest: Path,
    output_parquet: Path,
    dataset_run_id: str,
) -> dict[str, Any]:
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
        or not times.floor(f"{EXIT_DECISION_BAR_SECONDS}s").equals(times)
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
            source_contract_label="causal_enriched_m1_frame_v1",
            base_signal_fields=list(MODEL_NATIVE_BASE_FIELDS),
        )
    )
    base_signal = frame[list(MODEL_NATIVE_BASE_FIELDS)].astype(
        np.float32
    ).to_numpy()
    signal = np.concatenate([base_signal, extension], axis=1).astype(
        np.float32,
        copy=False,
    )
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

    output_frame = pd.DataFrame(
        {
            "time": times,
            "signal": [row.tolist() for row in signal],
            "ctx_cont": [
                row.tolist()
                for row in frame[list(MODEL_NATIVE_CTX_CONT_FIELDS)]
                .astype(np.float32)
                .to_numpy()
            ],
            "ctx_cat": [row.tolist() for row in ctx_cat],
        },
        columns=list(ENTRY_EXIT_FEATURE_SURFACE_COLUMNS),
    )
    output_frame.to_parquet(output, index=False)
    manifest = {
        "schema_version": ENTRY_EXIT_FEATURE_SURFACE_SCHEMA_VERSION,
        "decision": "PASS",
        "feature_base_contract_schema_version": ENTRY_EXIT_FEATURE_BASE_SCHEMA_VERSION,
        "shared_feature_base_contract": entry_exit_shared_feature_base_contract(),
        "dataset_run_id": dataset_run_id,
        "anchor_timeframe": "M1",
        "source_parquet": str(source),
        "source_sha256": _sha256_file(source),
        "seq_structure_manifest": str(manifest_path),
        "seq_structure_manifest_sha256": _sha256_file(manifest_path),
        "output_parquet": str(output),
        "output_parquet_sha256": _sha256_file(output),
        "rows": int(len(output_frame)),
        "signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "ctx_cont_dim": len(MODEL_NATIVE_CTX_CONT_FIELDS),
        "ctx_cat_dim": len(MODEL_NATIVE_CTX_CAT_FIELDS),
        "feature_field_order": list(contract["fields"]),
        "extension": extension_meta,
        "causal_contract": {
            "future_rows_used": False,
            "m1_closed_bar_required": True,
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
    _loaded_times, _loaded = load_m1_feature_surface(
        output,
        context="M1_FEATURE_BASE_POST_WRITE",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-parquet", required=True, type=Path)
    parser.add_argument("--seq-structure-manifest", required=True, type=Path)
    parser.add_argument("--output-parquet", required=True, type=Path)
    parser.add_argument("--dataset-run-id", required=True)
    args = parser.parse_args()
    manifest = materialize_m1_feature_base(
        source_parquet=args.source_parquet,
        seq_structure_manifest=args.seq_structure_manifest,
        output_parquet=args.output_parquet,
        dataset_run_id=args.dataset_run_id,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
