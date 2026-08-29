#!/usr/bin/env python3
"""Re-attest an unchanged Entry/Exit feature surface to a new signal manifest.

This narrowly scoped repair is permitted only when the old and replacement
model-native signal contracts are byte-for-byte equal.  It revalidates the
old immutable surface manifest against all of its sources, copies the already
validated parquet bytes without recomputing any feature, and writes a new
immutable sidecar bound to the replacement signal manifest.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

from gx1.contracts.entry_exit_feature_surface_v1 import (
    ENTRY_EXIT_FEATURE_SURFACE_COLUMNS,
    build_entry_exit_feature_surface_manifest,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_SIGNAL_DIM,
    require_model_native_manifest,
)
from gx1.utils.artifact_primitives_v1 import canonical_json_sha256, sha256_file


def _regular(path: Path, *, context: str) -> Path:
    resolved = Path(path).expanduser().absolute()
    if (
        not resolved.is_absolute()
        or resolved.is_symlink()
        or not resolved.is_file()
        or resolved.resolve(strict=True) != resolved
    ):
        raise RuntimeError(f"{context}_PATH_INVALID: {resolved}")
    return resolved


def _read_json(path: Path, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{context}_JSON_INVALID") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{context}_JSON_OBJECT_REQUIRED")
    return value


def _load_signal_contract(path: Path, *, context: str) -> tuple[Path, dict[str, Any]]:
    signal = _regular(path, context=context)
    return signal, require_model_native_manifest(
        _read_json(signal, context=context), context=context
    )


def _require_equivalent_signal_contract(
    old: Mapping[str, Any], new: Mapping[str, Any]
) -> None:
    """The policy lineage may change; the feature-surface contract may not."""

    if dict(old) != dict(new):
        raise RuntimeError("FEATURE_SURFACE_REATTEST_SIGNAL_CONTRACT_CHANGED")


def _surface_manifest(
    *,
    payload: Mapping[str, Any],
    signal_manifest: Path,
    signal_contract: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    timeframe = payload.get("anchor_timeframe")
    if timeframe not in {"M1", "M5"}:
        raise RuntimeError("FEATURE_SURFACE_REATTEST_TIMEFRAME_INVALID")
    source = payload.get("source_parquet")
    source_manifest = payload.get("source_manifest")
    source_sha = payload.get("source_sha256")
    source_manifest_sha = payload.get("source_manifest_sha256")
    source_manifest_schema = payload.get("source_manifest_schema_version")
    if not all(
        isinstance(value, str) and value
        for value in (
            source,
            source_manifest,
            source_sha,
            source_manifest_sha,
            source_manifest_schema,
        )
    ):
        raise RuntimeError("FEATURE_SURFACE_REATTEST_SOURCE_BINDING_INVALID")
    dataset_run_id = payload.get("dataset_run_id")
    pair_generation_id = payload.get("pair_generation_id")
    rows = payload.get("rows")
    if (
        not isinstance(dataset_run_id, str)
        or not dataset_run_id
        or not isinstance(pair_generation_id, str)
        or not pair_generation_id
        or isinstance(rows, bool)
        or not isinstance(rows, int)
        or rows <= 0
    ):
        raise RuntimeError("FEATURE_SURFACE_REATTEST_IDENTITY_INVALID")
    alignment_raw = payload.get("alignment_parquet")
    alignment = None if alignment_raw is None else Path(str(alignment_raw))
    materialization = payload.get("materialization")
    if timeframe == "M1" and not isinstance(materialization, Mapping):
        raise RuntimeError("FEATURE_SURFACE_REATTEST_M1_MATERIALIZATION_INVALID")
    if timeframe == "M5" and materialization is not None:
        raise RuntimeError("FEATURE_SURFACE_REATTEST_M5_MATERIALIZATION_INVALID")
    extension = payload.get("extension")
    registry_fit_binding = payload.get("registry_fit_binding")
    squeeze = payload.get("volatility_squeeze_artifact_set")
    if not all(
        isinstance(value, Mapping)
        for value in (extension, registry_fit_binding, squeeze)
    ):
        raise RuntimeError("FEATURE_SURFACE_REATTEST_COMPONENT_INVALID")
    causal_warmup = payload.get("causal_warmup")
    if causal_warmup is not None and not isinstance(causal_warmup, Mapping):
        raise RuntimeError("FEATURE_SURFACE_REATTEST_WARMUP_INVALID")
    return build_entry_exit_feature_surface_manifest(
        timeframe=timeframe,
        dataset_run_id=dataset_run_id,
        pair_generation_id=pair_generation_id,
        source=Path(source),
        source_binding={
            "source_sha256": source_sha,
            "manifest_path": source_manifest,
            "manifest_sha256": source_manifest_sha,
            "schema_version": source_manifest_schema,
        },
        alignment=alignment,
        seq_structure_manifest=signal_manifest,
        output=output,
        rows=rows,
        signal_contract=signal_contract,
        extension=extension,
        registry_fit_binding=registry_fit_binding,
        volatility_squeeze_artifact_binding=squeeze,
        materialization=(None if materialization is None else dict(materialization)),
        causal_warmup=(None if causal_warmup is None else dict(causal_warmup)),
    )


def _validate_surface_schema(path: Path, *, rows: int) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
    except Exception as exc:
        raise RuntimeError("FEATURE_SURFACE_REATTEST_PARQUET_INVALID") from exc
    expected_types = {
        "signal": pa.list_(pa.float32(), MODEL_NATIVE_SIGNAL_DIM),
        "ctx_cont": pa.list_(pa.float32(), MODEL_NATIVE_CTX_CONT_DIM),
        "ctx_cat": pa.list_(pa.int64(), MODEL_NATIVE_CTX_CAT_DIM),
    }
    if (
        parquet.metadata is None
        or parquet.metadata.num_rows != rows
        or tuple(parquet.schema_arrow.names) != ENTRY_EXIT_FEATURE_SURFACE_COLUMNS
        or any(
            parquet.schema_arrow.field(name).type != expected
            for name, expected in expected_types.items()
        )
    ):
        raise RuntimeError("FEATURE_SURFACE_REATTEST_PARQUET_SCHEMA_INVALID")


def _admit_stage(stage: Path, destination: Path) -> None:
    with stage.open("rb") as handle:
        os.fsync(handle.fileno())
    try:
        os.link(stage, destination, follow_symlinks=False)
    except FileExistsError as exc:
        raise RuntimeError(
            f"FEATURE_SURFACE_REATTEST_OUTPUT_EXISTS: {destination}"
        ) from exc
    finally:
        stage.unlink(missing_ok=True)


def _copy_file_noreplace(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"FEATURE_SURFACE_REATTEST_OUTPUT_EXISTS: {destination}")
    stage = destination.with_name(f".{destination.name}.partial-{uuid.uuid4().hex}")
    try:
        with source.open("rb") as src, stage.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        _admit_stage(stage, destination)
    finally:
        stage.unlink(missing_ok=True)


def _write_json_noreplace(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"FEATURE_SURFACE_REATTEST_OUTPUT_EXISTS: {path}")
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    stage = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    try:
        with stage.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _admit_stage(stage, path)
    finally:
        stage.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    old_surface = _regular(args.source_surface_parquet, context="FEATURE_SURFACE_REATTEST_SOURCE")
    old_sidecar = _regular(
        Path(f"{old_surface}.manifest.json"), context="FEATURE_SURFACE_REATTEST_SOURCE_MANIFEST"
    )
    old_payload = _read_json(old_sidecar, context="FEATURE_SURFACE_REATTEST_SOURCE_MANIFEST")
    old_signal, old_contract = _load_signal_contract(
        args.source_signal_manifest, context="FEATURE_SURFACE_REATTEST_OLD_SIGNAL"
    )
    new_signal, new_contract = _load_signal_contract(
        args.replacement_signal_manifest,
        context="FEATURE_SURFACE_REATTEST_NEW_SIGNAL",
    )
    _require_equivalent_signal_contract(old_contract, new_contract)
    if (
        old_payload.get("seq_structure_manifest") != str(old_signal)
        or old_payload.get("seq_structure_manifest_sha256") != sha256_file(old_signal)
    ):
        raise RuntimeError("FEATURE_SURFACE_REATTEST_OLD_SIGNAL_BINDING_INVALID")
    rows = old_payload.get("rows")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise RuntimeError("FEATURE_SURFACE_REATTEST_ROWS_INVALID")
    _validate_surface_schema(old_surface, rows=rows)
    old_expected = _surface_manifest(
        payload=old_payload,
        signal_manifest=old_signal,
        signal_contract=old_contract,
        output=old_surface,
    )
    if old_payload != old_expected:
        raise RuntimeError("FEATURE_SURFACE_REATTEST_SOURCE_MANIFEST_INVALID")

    output = Path(args.output_parquet).expanduser().absolute()
    sidecar = Path(f"{output}.manifest.json")
    if (
        output.suffix != ".parquet"
        or output.is_symlink()
        or sidecar.is_symlink()
        or not output.parent.is_dir()
        or output.parent.is_symlink()
        or output.exists()
        or sidecar.exists()
    ):
        raise RuntimeError("FEATURE_SURFACE_REATTEST_OUTPUT_PATH_INVALID")

    report = {
        "schema_version": "entry_exit_feature_surface_reattest_v1",
        "decision": "PASS",
        "source_surface": str(old_surface),
        "source_surface_sha256": sha256_file(old_surface),
        "source_manifest": str(old_sidecar),
        "source_manifest_sha256": sha256_file(old_sidecar),
        "replacement_signal_manifest": str(new_signal),
        "replacement_signal_manifest_sha256": sha256_file(new_signal),
        "feature_contract_sha256": canonical_json_sha256(old_contract),
        "output_surface": str(output),
        "output_manifest": str(sidecar),
        "rows": rows,
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        return report

    _copy_file_noreplace(old_surface, output)
    if sha256_file(output) != report["source_surface_sha256"]:
        raise RuntimeError("FEATURE_SURFACE_REATTEST_COPY_HASH_MISMATCH")
    _validate_surface_schema(output, rows=rows)
    new_manifest = _surface_manifest(
        payload=old_payload,
        signal_manifest=new_signal,
        signal_contract=new_contract,
        output=output,
    )
    _write_json_noreplace(sidecar, new_manifest)
    report["output_surface_sha256"] = sha256_file(output)
    report["output_manifest_sha256"] = sha256_file(sidecar)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-surface-parquet", type=Path, required=True)
    parser.add_argument("--source-signal-manifest", type=Path, required=True)
    parser.add_argument("--replacement-signal-manifest", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
