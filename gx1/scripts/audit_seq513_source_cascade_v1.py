"""Audit one fresh seq513 source cascade and emit immutable PASS proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_run_lineage_v1 import require_entry_run_id
from gx1.contracts.xau_tape_provenance_v1 import (
    CANONICAL_NATIVE_SOURCE_SCHEMA,
    CURRENT_SNAPSHOT_SCHEMA,
    validate_xau_tape_provenance_v1,
)
from gx1.features.htf_features import HTF_V2_CACHE_BUILDER_VERSION
from gx1.scripts.materialize_cv3_modelrange_v1 import (
    ENTRY_DEAD_CONSTANT_COLUMNS,
    EXTRA_COLUMNS_FROM_CANONICAL_V2,
    SCHEMA_VERSION as MODELRANGE_SCHEMA,
)


SCHEMA_VERSION = "seq513_source_cascade_proof_v5"
EXPECTED_CV2_COLUMNS = 118
# canonical-v3 computes this manifest count before surfacing the DatetimeIndex
# as the plain `time` column.  The parquet is 113 wide; the manifest's
# `cols_total` is therefore the exact 112 feature-column count.
EXPECTED_CV3_MANIFEST_COLUMNS = 112
EXPECTED_MODELRANGE_COLUMNS = 109
EXPECTED_FULL_COLUMNS = 188
EXPECTED_TFS = ("M5", "M15", "H1", "H4", "D1")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"SEQ513_SOURCE_{label}_MISSING_OR_SYMLINK: {path}")
    return path.resolve()


def _json(path: Path, *, label: str) -> dict[str, Any]:
    resolved = _regular(path, label=label)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"SEQ513_SOURCE_{label}_JSON_INVALID: {resolved}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"SEQ513_SOURCE_{label}_JSON_OBJECT_REQUIRED")
    return value


def _same_path(raw: Any, expected: Path, *, label: str) -> None:
    actual = Path(str(raw or "")).expanduser().resolve()
    if actual != expected.resolve():
        raise RuntimeError(
            f"SEQ513_SOURCE_{label}_PATH_MISMATCH: actual={actual} expected={expected.resolve()}"
        )


def _same(raw: Any, expected: Any, *, label: str) -> None:
    if raw != expected:
        raise RuntimeError(
            f"SEQ513_SOURCE_{label}_MISMATCH: actual={raw!r} expected={expected!r}"
        )


def _utc(raw: Any, *, label: str) -> pd.Timestamp:
    try:
        parsed = pd.to_datetime(raw, utc=True, errors="raise")
    except Exception as exc:
        raise RuntimeError(f"SEQ513_SOURCE_{label}_TIMESTAMP_INVALID: {raw!r}") from exc
    if pd.isna(parsed):
        raise RuntimeError(f"SEQ513_SOURCE_{label}_TIMESTAMP_INVALID: {raw!r}")
    return pd.Timestamp(parsed)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short write: {temporary}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _full_numeric_liveness(path: Path) -> dict[str, Any]:
    """Reject every non-finite, constant, or exact duplicate source field."""

    frame = pd.read_parquet(path)
    field_sha256: dict[str, str] = {}
    groups: dict[str, list[str]] = {}
    nonfinite: list[str] = []
    constants: list[str] = []
    for name in frame.columns:
        if name == "time":
            continue
        try:
            values = pd.to_numeric(frame[name], errors="raise").to_numpy(
                dtype=np.float64
            )
        except Exception as exc:
            raise RuntimeError(
                f"SEQ513_SOURCE_FULL_COLUMN_NONNUMERIC: {name}"
            ) from exc
        if not np.isfinite(values).all():
            nonfinite.append(name)
            continue
        if float(np.ptp(values)) == 0.0:
            constants.append(name)
        digest = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
        field_sha256[name] = digest
        groups.setdefault(digest, []).append(name)
    duplicates = sorted(
        (sorted(names) for names in groups.values() if len(names) > 1),
        key=lambda names: tuple(names),
    )
    if nonfinite:
        raise RuntimeError(f"SEQ513_SOURCE_FULL_NUMERIC_NONFINITE: {nonfinite}")
    if constants:
        raise RuntimeError(f"SEQ513_SOURCE_FULL_NUMERIC_CONSTANT: {constants}")
    if duplicates:
        raise RuntimeError(f"SEQ513_SOURCE_FULL_NUMERIC_DUPLICATES: {duplicates}")
    return {
        "decision": "PASS",
        "rows": int(len(frame)),
        "audited_numeric_fields": len(field_sha256),
        "nonfinite_fields": [],
        "constant_fields": [],
        "exact_duplicate_groups": [],
        "field_float64_sha256": field_sha256,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_id = require_entry_run_id(getattr(args, "run_id", ""))
    required_history_start = _utc(
        getattr(args, "required_history_start", None), label="REQUIRED_HISTORY_START"
    )
    expected_full_time_min = _utc(
        getattr(args, "expected_full_time_min", None), label="EXPECTED_FULL_TIME_MIN"
    )
    expected_full_time_max = _utc(
        getattr(args, "expected_full_time_max", None), label="EXPECTED_FULL_TIME_MAX"
    )
    if expected_full_time_max < expected_full_time_min:
        raise RuntimeError("SEQ513_SOURCE_EXPECTED_FULL_TIME_WINDOW_INVALID")
    if not expected_full_time_min <= required_history_start <= expected_full_time_max:
        raise RuntimeError(
            "SEQ513_SOURCE_REQUIRED_HISTORY_NOT_COVERED: "
            f"full_min={expected_full_time_min.isoformat()} "
            f"history_start={required_history_start.isoformat()} "
            f"full_max={expected_full_time_max.isoformat()}"
        )
    root_arg = Path(args.event_root).expanduser()
    if root_arg.is_symlink() or not root_arg.is_dir():
        raise RuntimeError(f"SEQ513_SOURCE_EVENT_ROOT_MISSING_OR_SYMLINK: {root_arg}")
    root = root_arg.resolve()
    out = Path(args.out).expanduser().resolve()
    if out.parent != root or out.exists() or out.is_symlink():
        raise RuntimeError("SEQ513_SOURCE_PROOF_TARGET_NOT_FRESH_EVENT_LOCAL")

    tape_native = root / "m5_tape_native_v3"
    tape_repaired = root / "m5_tape_repaired_dec2024"
    if tape_native.exists() and tape_repaired.exists():
        raise RuntimeError(
            "SEQ513_SOURCE_TAPE_IDENTITY_AMBIGUOUS: both m5_tape_native_v3 "
            "and m5_tape_repaired_dec2024 exist in the event root"
        )
    tape = tape_native if tape_native.exists() else tape_repaired
    tape_provenance = validate_xau_tape_provenance_v1(
        tape,
        expected_run_id=run_id,
        require_current=True,
    )
    if tape_provenance.get("schema_version") == CANONICAL_NATIVE_SOURCE_SCHEMA:
        _same(
            _utc(
                tape_provenance.get("time_max_utc"),
                label="NATIVE_TAPE_LAST_COMPLETE_M5",
            ),
            expected_full_time_max,
            label="NATIVE_TAPE_LAST_COMPLETE_M5",
        )
    else:
        repair = _json(tape / "REPAIR_MANIFEST.json", label="REPAIR_MANIFEST")
        _same(
            repair.get("schema_version"),
            CURRENT_SNAPSHOT_SCHEMA,
            label="REPAIR_SCHEMA",
        )
        _same(
            _utc(
                repair.get("last_complete_m5_utc"),
                label="REPAIR_LAST_COMPLETE_M5",
            ),
            expected_full_time_max,
            label="REPAIR_LAST_COMPLETE_M5",
        )
    tape_hashes = dict(tape_provenance["year_sha256"])
    year_numbers = sorted(int(key.split("=", 1)[1]) for key in tape_hashes)

    cv2 = _regular(root / "canonical_features_v2.parquet", label="CV2")
    cv2_sha = _sha256_file(cv2)
    cv2_rows = len(pd.read_parquet(cv2, columns=["time"]))
    cv2_summary = _json(root / "canonical_features_v2_summary.json", label="CV2_SUMMARY")
    _same_path(cv2_summary.get("out_path_v1"), cv2, label="CV2_OUTPUT")
    _same_path(cv2_summary.get("m5_tape_root_v1"), tape, label="CV2_TAPE")
    _same(cv2_summary.get("m5_bars_loaded_v1"), cv2_rows, label="CV2_ROWS")
    _same(cv2_summary.get("total_columns_v1"), EXPECTED_CV2_COLUMNS, label="CV2_COLUMNS")
    _same(
        (cv2_summary.get("htf_alignment_contract_v1") or {}).get("no_lookahead"),
        True,
        label="CV2_NO_LOOKAHEAD",
    )

    cv3 = _regular(
        root / "cv3" / "xauusd_m5_CANONICAL_V3_2020_2026.parquet", label="CV3"
    )
    cv3_sha = _sha256_file(cv3)
    cv3_rows = len(pd.read_parquet(cv3, columns=["time"]))
    cv3_manifest = _json(root / "cv3" / "CURRENT_MANIFEST.json", label="CV3_MANIFEST")
    _same_path(cv3_manifest.get("parquet_path"), cv3, label="CV3_OUTPUT")
    _same(cv3_manifest.get("parquet_sha256"), cv3_sha, label="CV3_HASH")
    _same_path(cv3_manifest.get("source_v2_parquet"), cv2, label="CV3_SOURCE")
    _same(cv3_manifest.get("source_v2_parquet_sha256"), cv2_sha, label="CV3_SOURCE_HASH")
    _same(cv3_manifest.get("rows"), cv3_rows, label="CV3_ROWS")
    _same(cv3_rows, cv2_rows, label="CV3_CV2_ROW_PARITY")
    _same(
        cv3_manifest.get("cols_total"),
        EXPECTED_CV3_MANIFEST_COLUMNS,
        label="CV3_COLUMNS",
    )
    _same(cv3_manifest.get("source_v2_no_lookahead"), True, label="CV3_NO_LOOKAHEAD")

    modelrange = _regular(root / "cv3_modelrange.parquet", label="MODELRANGE")
    modelrange_sha = _sha256_file(modelrange)
    modelrange_time = pd.to_datetime(
        pd.read_parquet(modelrange, columns=["time"])["time"], utc=True, errors="coerce"
    )
    modelrange_rows = len(modelrange_time)
    modelrange_manifest = _json(
        root / "cv3_modelrange.provenance.json", label="MODELRANGE_MANIFEST"
    )
    _same(modelrange_manifest.get("schema_version"), MODELRANGE_SCHEMA, label="MODELRANGE_SCHEMA")
    _same(modelrange_manifest.get("entry_run_id"), run_id, label="MODELRANGE_RUN_ID")
    modelrange_inputs = modelrange_manifest.get("inputs") or {}
    _same_path(modelrange_inputs.get("cv3"), cv3, label="MODELRANGE_CV3")
    _same(modelrange_inputs.get("cv3_sha256"), cv3_sha, label="MODELRANGE_CV3_HASH")
    _same_path(modelrange_inputs.get("canonical_v2"), cv2, label="MODELRANGE_CV2")
    _same(modelrange_inputs.get("canonical_v2_sha256"), cv2_sha, label="MODELRANGE_CV2_HASH")
    _same_path(modelrange_manifest.get("output"), modelrange, label="MODELRANGE_OUTPUT")
    _same(modelrange_manifest.get("output_sha256"), modelrange_sha, label="MODELRANGE_HASH")
    _same(modelrange_manifest.get("rows"), modelrange_rows, label="MODELRANGE_ROWS")
    _same(
        _utc(modelrange_manifest.get("time_max_utc"), label="MODELRANGE_TIME_MAX"),
        expected_full_time_max,
        label="MODELRANGE_TIME_MAX",
    )
    _same(
        modelrange_manifest.get("columns"), EXPECTED_MODELRANGE_COLUMNS, label="MODELRANGE_COLUMNS"
    )
    _same(
        modelrange_manifest.get("extra_columns_from_canonical_v2"),
        list(EXTRA_COLUMNS_FROM_CANONICAL_V2),
        label="MODELRANGE_REQUIRED_EXTRA_COLUMNS",
    )
    _same(
        modelrange_manifest.get("entry_dead_constant_columns_removed"),
        list(ENTRY_DEAD_CONSTANT_COLUMNS),
        label="MODELRANGE_DEAD_CONSTANT_COLUMNS_REMOVED",
    )

    mtf_root = root / "MULTI_TF_V2_CACHE"
    mtf = _json(mtf_root / "manifest.json", label="MTF_MANIFEST")
    _same(mtf.get("builder_version"), HTF_V2_CACHE_BUILDER_VERSION, label="MTF_BUILDER")
    _same_path(mtf.get("m5_prebuilt_source"), cv3, label="MTF_SOURCE")
    _same(mtf.get("m5_prebuilt_source_sha256"), cv3_sha, label="MTF_SOURCE_HASH")
    mtf_meta = mtf.get("tfs")
    if not isinstance(mtf_meta, dict) or tuple(mtf_meta) != EXPECTED_TFS:
        raise RuntimeError("SEQ513_SOURCE_MTF_SET_OR_ORDER_INVALID")
    mtf_hashes: dict[str, dict[str, str]] = {}
    for tf in EXPECTED_TFS:
        row = mtf_meta[tf]
        feats = _regular(mtf_root / str(row.get("feats_npy")), label=f"MTF_{tf}_FEATS")
        timestamps = _regular(mtf_root / str(row.get("ts_npy")), label=f"MTF_{tf}_TS")
        mtf_hashes[tf] = {
            "feats_sha256": _sha256_file(feats),
            "timestamps_sha256": _sha256_file(timestamps),
        }

    full = _regular(root / "FULL_PLUS_CTX_v3src.parquet", label="FULL_PLUS")
    full_sha = _sha256_file(full)
    full_manifest = _json(root / "FULL_PLUS_CTX_v3src.manifest.json", label="FULL_MANIFEST")
    _same(full_manifest.get("kind"), "entry_model_native_prebuilt_manifest_v2", label="FULL_SCHEMA")
    _same_path(full_manifest.get("prebuilt_path"), full, label="FULL_OUTPUT")
    _same(full_manifest.get("prebuilt_sha256"), full_sha, label="FULL_HASH")
    _same(full_manifest.get("no_fallback_enforced"), True, label="FULL_NO_FALLBACK")
    diagnostics = _json(
        root / "FULL_PLUS_CTX_v3src.ctx_diagnostics.json", label="FULL_DIAGNOSTICS"
    )
    _same_path(diagnostics.get("prebuilt_path"), modelrange, label="FULL_SOURCE")
    _same_path(diagnostics.get("output_path"), full, label="FULL_DIAGNOSTICS_OUTPUT")
    _same_path(diagnostics.get("tape_root"), tape, label="FULL_TAPE")
    expected_raw = [
        str((tape / f"year={year}" / "part-000.parquet").resolve())
        for year in year_numbers
    ]
    _same(diagnostics.get("raw_m5_paths"), expected_raw, label="FULL_RAW_M5_PATHS")
    schema = _json(root / "FULL_PLUS_CTX_v3src.schema_manifest.json", label="FULL_SCHEMA_MANIFEST")
    required_features = schema.get("required_all_features")
    if not isinstance(required_features, list) or len(required_features) != EXPECTED_FULL_COLUMNS:
        raise RuntimeError("SEQ513_SOURCE_FULL_SCHEMA_WIDTH_INVALID")

    full_time = pd.read_parquet(full, columns=["time"])["time"]
    parsed_time = pd.to_datetime(full_time, utc=True, errors="coerce")
    full_rows = len(parsed_time)
    _same(diagnostics.get("n_rows"), full_rows, label="FULL_ROWS_DIAGNOSTICS")
    if (
        parsed_time.isna().any()
        or parsed_time.duplicated().any()
        or not parsed_time.is_monotonic_increasing
        or pd.Timestamp(parsed_time.iloc[0]) != expected_full_time_min
        or pd.Timestamp(parsed_time.iloc[-1]) != expected_full_time_max
    ):
        raise RuntimeError("SEQ513_SOURCE_FULL_TIME_CONTRACT_INVALID")
    if (
        modelrange_time.isna().any()
        or modelrange_time.duplicated().any()
        or not modelrange_time.is_monotonic_increasing
        or pd.Timestamp(modelrange_time.iloc[-1]) != expected_full_time_max
    ):
        raise RuntimeError("SEQ513_SOURCE_MODELRANGE_TIME_CONTRACT_INVALID")
    full_numeric_liveness = _full_numeric_liveness(full)

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "PASS",
        "entry_run_id": run_id,
        "event_root": str(root),
        "artifacts": {
            "repair_manifest_sha256": _sha256_file(tape / "REPAIR_MANIFEST.json"),
            "tape_year_sha256": tape_hashes,
            "canonical_v2_sha256": cv2_sha,
            "canonical_v2_summary_sha256": _sha256_file(root / "canonical_features_v2_summary.json"),
            "canonical_v3_sha256": cv3_sha,
            "canonical_v3_manifest_sha256": _sha256_file(root / "cv3" / "CURRENT_MANIFEST.json"),
            "modelrange_sha256": modelrange_sha,
            "modelrange_manifest_sha256": _sha256_file(root / "cv3_modelrange.provenance.json"),
            "multi_tf_manifest_sha256": _sha256_file(mtf_root / "manifest.json"),
            "multi_tf_arrays": mtf_hashes,
            "full_plus_sha256": full_sha,
            "full_plus_manifest_sha256": _sha256_file(root / "FULL_PLUS_CTX_v3src.manifest.json"),
            "full_plus_diagnostics_sha256": _sha256_file(
                root / "FULL_PLUS_CTX_v3src.ctx_diagnostics.json"
            ),
            "full_plus_schema_sha256": _sha256_file(
                root / "FULL_PLUS_CTX_v3src.schema_manifest.json"
            ),
        },
        "contracts": {
            "xau_tape_provenance": tape_provenance,
            "no_stale_self_paths": True,
            "no_symlink_artifacts": True,
            "exact_run_lineage": True,
            "no_fallback": True,
            "full_rows": full_rows,
            "full_columns": EXPECTED_FULL_COLUMNS,
            "full_time_min_utc": expected_full_time_min.isoformat(),
            "full_time_max_utc": expected_full_time_max.isoformat(),
            "required_history_start_utc": required_history_start.isoformat(),
            "required_history_start_covered": True,
            "full_numeric_feature_liveness": full_numeric_liveness,
        },
    }
    _atomic_json(out, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--event-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--required-history-start", required=True)
    parser.add_argument("--expected-full-time-min", required=True)
    parser.add_argument("--expected-full-time-max", required=True)
    return parser


def main() -> int:
    report = run(build_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
