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
    CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA,
    validate_xau_tape_provenance_v1,
)
from gx1.features.htf_features import (
    HTF_V4_CACHE_BUILDER_VERSION,
    HTF_V4_CACHE_SCHEMA_VERSION,
    MULTI_TF_FEATURE_COUNT_V4,
    MULTI_TF_PER_BAR_FEATURES_V4,
    MULTI_TF_TIMEFRAMES,
    load_multi_tf_cache,
    require_multi_tf_v4_liveness_contract,
)
from gx1.scripts.materialize_cv3_modelrange_v1 import (
    CTX_OWNED_SESSION_COLUMNS,
    ENTRY_DEAD_CONSTANT_COLUMNS,
    EXTRA_COLUMNS_FROM_CANONICAL_V2,
    SCHEMA_VERSION as MODELRANGE_SCHEMA,
)


SCHEMA_VERSION = "seq513_source_cascade_proof_v7"
CURRENT_PAIR_SCHEMA_VERSION = "seq513_source_cascade_pair_proof_v1"
# 2026-07-24 source decisions changed the canonical surface: the three
# non-causal slippage/cost fields are removed, the session evidence block is
# mandatory (nine add_session_features fields plus _v1_is_EU/_v1_is_US and
# three _v1_int_*_us interactions), _v1_session_volatility_pressure replaces
# _v1_cost_bps_dyn and observed spread keeps its honest name (spread_pct).
EXPECTED_CV2_COLUMNS = 131
# canonical-v3 computes this manifest count before surfacing the DatetimeIndex
# as the plain `time` column.  The parquet is 126 wide; the manifest's
# `cols_total` is therefore the exact 125 feature-column count.
EXPECTED_CV3_MANIFEST_COLUMNS = 125
# Entry projection: 126 cv3 columns − 3 dead constants − 9 ctx-owned session
# columns + exact canonical-v2 `atr` = 115.
EXPECTED_MODELRANGE_COLUMNS = 115
# FULL_PLUS = 115 modelrange columns + the exact 79-column Entry context set.
EXPECTED_FULL_COLUMNS = 194
EXPECTED_TFS = MULTI_TF_TIMEFRAMES
NATIVE_SOURCE_SCHEMAS = {
    CANONICAL_NATIVE_SOURCE_SCHEMA,
    CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA,
}


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

    tape = root / "m5_tape_native_v3"
    if tape.is_symlink() or not tape.is_dir():
        raise RuntimeError("SEQ513_SOURCE_NATIVE_M5_TAPE_REQUIRED")
    tape_provenance = validate_xau_tape_provenance_v1(
        tape,
        expected_run_id=run_id,
        require_current=True,
    )
    if tape_provenance.get("schema_version") not in NATIVE_SOURCE_SCHEMAS:
        raise RuntimeError("SEQ513_SOURCE_NATIVE_M5_PROVENANCE_REQUIRED")
    _same(
        _utc(
            tape_provenance.get("time_max_utc"),
            label="NATIVE_TAPE_LAST_COMPLETE_M5",
        ),
        expected_full_time_max,
        label="NATIVE_TAPE_LAST_COMPLETE_M5",
    )
    tape_hashes = dict(tape_provenance["year_sha256"])
    year_numbers = sorted(int(key.split("=", 1)[1]) for key in tape_hashes)
    tape_manifest_name = "MANIFEST.json"

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
    _same(
        modelrange_manifest.get("ctx_owned_session_columns_removed"),
        list(CTX_OWNED_SESSION_COLUMNS),
        label="MODELRANGE_CTX_OWNED_SESSION_COLUMNS_REMOVED",
    )

    mtf_root = root / "MULTI_TF_V4_CACHE"
    mtf = _json(mtf_root / "manifest.json", label="MTF_MANIFEST")
    _same(
        mtf.get("builder_version"),
        HTF_V4_CACHE_BUILDER_VERSION,
        label="MTF_BUILDER",
    )
    _same(
        mtf.get("schema_version"),
        HTF_V4_CACHE_SCHEMA_VERSION,
        label="MTF_SCHEMA",
    )
    _same(
        mtf.get("feature_count"),
        MULTI_TF_FEATURE_COUNT_V4,
        label="MTF_FEATURE_COUNT",
    )
    _same(
        mtf.get("feature_names"),
        list(MULTI_TF_PER_BAR_FEATURES_V4),
        label="MTF_FEATURE_NAMES",
    )
    _same_path(mtf.get("m5_prebuilt_source"), cv3, label="MTF_SOURCE")
    _same(mtf.get("m5_prebuilt_source_sha256"), cv3_sha, label="MTF_SOURCE_HASH")
    require_multi_tf_v4_liveness_contract(mtf.get("full_input_liveness"))
    verified_mtf = load_multi_tf_cache(mtf_root)
    _same(
        getattr(verified_mtf, "cache_identity_sha256", None),
        mtf.get("cache_identity_sha256"),
        label="MTF_VERIFIED_CACHE_IDENTITY",
    )
    mtf_meta = mtf.get("tfs")
    if not isinstance(mtf_meta, dict) or set(mtf_meta) != set(EXPECTED_TFS):
        raise RuntimeError("SEQ513_SOURCE_MTF_SET_INVALID")
    mtf_hashes: dict[str, dict[str, str]] = {}
    for tf in EXPECTED_TFS:
        row = mtf_meta[tf]
        feats = _regular(mtf_root / str(row.get("feats_npy")), label=f"MTF_{tf}_FEATS")
        timestamps = _regular(mtf_root / str(row.get("ts_npy")), label=f"MTF_{tf}_TS")
        mtf_hashes[tf] = {
            "feats_sha256": _sha256_file(feats),
            "timestamps_sha256": _sha256_file(timestamps),
        }
        _same(
            mtf_hashes[tf]["feats_sha256"],
            row.get("feats_npy_sha256"),
            label=f"MTF_{tf}_FEATS_HASH",
        )
        _same(
            mtf_hashes[tf]["timestamps_sha256"],
            row.get("ts_npy_sha256"),
            label=f"MTF_{tf}_TS_HASH",
        )

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
            "tape_manifest_sha256": _sha256_file(tape / tape_manifest_name),
            "tape_year_sha256": tape_hashes,
            "canonical_v2_sha256": cv2_sha,
            "canonical_v2_summary_sha256": _sha256_file(root / "canonical_features_v2_summary.json"),
            "canonical_v3_sha256": cv3_sha,
            "canonical_v3_manifest_sha256": _sha256_file(root / "cv3" / "CURRENT_MANIFEST.json"),
            "modelrange_sha256": modelrange_sha,
            "modelrange_manifest_sha256": _sha256_file(root / "cv3_modelrange.provenance.json"),
            "multi_tf_manifest_sha256": _sha256_file(mtf_root / "manifest.json"),
            "multi_tf_cache_identity_sha256": str(
                mtf["cache_identity_sha256"]
            ),
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
            "multi_tf_v4_liveness_contract_sha256": mtf[
                "full_input_liveness"
            ]["contract_sha256"],
        },
    }
    _atomic_json(out, report)
    return report


def validate_seq513_source_cascade_proof(
    proof_path: Path,
    *,
    expected_run_id: str,
    expected_source_parquet: Path,
    expected_canonical_v2_parquet: Path,
    expected_mtf_cache_dir: Path,
    expected_history_start_utc: object,
    expected_time_max_utc: object,
) -> dict[str, Any]:
    """Revalidate the immutable source→V4-cache proof at a consumer boundary."""

    run_id = require_entry_run_id(expected_run_id)
    source = _regular(
        expected_source_parquet.expanduser().resolve(),
        label="BOUND_FULL_PLUS",
    )
    canonical_v2 = _regular(
        expected_canonical_v2_parquet.expanduser().resolve(),
        label="BOUND_CV2",
    )
    cache_dir = expected_mtf_cache_dir.expanduser().resolve()
    if cache_dir.is_symlink() or not cache_dir.is_dir():
        raise RuntimeError(
            f"SEQ513_SOURCE_BOUND_MTF_CACHE_MISSING_OR_SYMLINK: {cache_dir}"
        )
    event_root = source.parent.resolve()
    proof = _json(proof_path.expanduser().resolve(), label="CASCADE_PROOF")
    resolved_proof = proof_path.expanduser().resolve()
    if resolved_proof.parent != event_root:
        raise RuntimeError("SEQ513_SOURCE_CASCADE_PROOF_NOT_EVENT_LOCAL")
    if proof.get("schema_version") == CURRENT_PAIR_SCHEMA_VERSION:
        return _validate_current_pair_source_cascade_proof(
            proof_path=resolved_proof,
            proof=proof,
            expected_run_id=run_id,
            expected_source_parquet=source,
            expected_canonical_v2_parquet=canonical_v2,
            expected_mtf_cache_dir=cache_dir,
            expected_history=expected_history_start_utc,
            expected_time_max=expected_time_max_utc,
        )
    if canonical_v2.parent.resolve() != event_root or cache_dir.parent.resolve() != event_root:
        raise RuntimeError("SEQ513_SOURCE_BOUND_ARTIFACTS_NOT_ONE_EVENT_ROOT")
    if set(proof) != {
        "schema_version",
        "created_utc",
        "decision",
        "entry_run_id",
        "event_root",
        "artifacts",
        "contracts",
    }:
        raise RuntimeError("SEQ513_SOURCE_CASCADE_PROOF_KEYS_INVALID")
    _same(proof.get("schema_version"), SCHEMA_VERSION, label="CASCADE_SCHEMA")
    _same(proof.get("decision"), "PASS", label="CASCADE_DECISION")
    _same(proof.get("entry_run_id"), run_id, label="CASCADE_RUN_ID")
    _same_path(proof.get("event_root"), event_root, label="CASCADE_EVENT_ROOT")

    artifacts = proof.get("artifacts")
    contracts = proof.get("contracts")
    if not isinstance(artifacts, dict) or not isinstance(contracts, dict):
        raise RuntimeError("SEQ513_SOURCE_CASCADE_PROOF_SECTIONS_INVALID")
    cache = load_multi_tf_cache(cache_dir)
    manifest_path = cache_dir / "manifest.json"
    expected_history = _utc(
        expected_history_start_utc,
        label="BOUND_HISTORY_START",
    )
    expected_time_max = _utc(
        expected_time_max_utc,
        label="BOUND_TIME_MAX",
    )
    _same(
        artifacts.get("full_plus_sha256"),
        _sha256_file(source),
        label="CASCADE_FULL_PLUS_HASH",
    )
    _same(
        artifacts.get("canonical_v2_sha256"),
        _sha256_file(canonical_v2),
        label="CASCADE_CV2_HASH",
    )
    _same(
        artifacts.get("multi_tf_manifest_sha256"),
        _sha256_file(manifest_path),
        label="CASCADE_MTF_MANIFEST_HASH",
    )
    _same(
        artifacts.get("multi_tf_cache_identity_sha256"),
        cache.cache_identity_sha256,
        label="CASCADE_MTF_CACHE_IDENTITY",
    )
    _same(
        _utc(
            contracts.get("required_history_start_utc"),
            label="CASCADE_HISTORY_START",
        ),
        expected_history,
        label="CASCADE_HISTORY_START",
    )
    _same(
        _utc(contracts.get("full_time_max_utc"), label="CASCADE_TIME_MAX"),
        expected_time_max,
        label="CASCADE_TIME_MAX",
    )
    _same(
        contracts.get("required_history_start_covered"),
        True,
        label="CASCADE_HISTORY_COVERED",
    )
    _same(contracts.get("no_fallback"), True, label="CASCADE_NO_FALLBACK")
    return {
        "path": str(resolved_proof),
        "sha256": _sha256_file(resolved_proof),
        "schema_version": SCHEMA_VERSION,
        "entry_run_id": run_id,
        "event_root": str(event_root),
        "source_parquet_sha256": str(artifacts["full_plus_sha256"]),
        "canonical_v2_sha256": str(artifacts["canonical_v2_sha256"]),
        "multi_tf_manifest_sha256": str(
            artifacts["multi_tf_manifest_sha256"]
        ),
        "multi_tf_cache_identity_sha256": str(
            artifacts["multi_tf_cache_identity_sha256"]
        ),
        "history_start_utc": expected_history.isoformat(),
        "time_max_utc": expected_time_max.isoformat(),
    }


def _validate_current_pair_source_cascade_proof(
    *,
    proof_path: Path,
    proof: dict[str, Any],
    expected_run_id: str,
    expected_source_parquet: Path,
    expected_canonical_v2_parquet: Path,
    expected_mtf_cache_dir: Path,
    expected_history: object,
    expected_time_max: object,
) -> dict[str, Any]:
    """Validate the compact V3 pair lineage without reviving legacy FULL_PLUS."""

    required_keys = {
        "schema_version",
        "created_utc",
        "decision",
        "entry_run_id",
        "event_root",
        "artifacts",
        "contracts",
    }
    if set(proof) != required_keys:
        raise RuntimeError("SEQ513_CURRENT_PAIR_PROOF_KEYS_INVALID")
    event_root = expected_source_parquet.parent.resolve()
    _same(proof.get("decision"), "PASS", label="CURRENT_PAIR_DECISION")
    _same(proof.get("entry_run_id"), expected_run_id, label="CURRENT_PAIR_RUN_ID")
    _same_path(proof.get("event_root"), event_root, label="CURRENT_PAIR_EVENT_ROOT")
    artifacts = proof.get("artifacts")
    contracts = proof.get("contracts")
    if not isinstance(artifacts, dict) or not isinstance(contracts, dict):
        raise RuntimeError("SEQ513_CURRENT_PAIR_PROOF_SECTIONS_INVALID")
    if set(artifacts) != {
        "source_parquet_path",
        "source_parquet_sha256",
        "canonical_v2_path",
        "canonical_v2_sha256",
        "multi_tf_manifest_sha256",
        "multi_tf_cache_identity_sha256",
        "pair_manifest_path",
        "pair_manifest_sha256",
        "pair_generation_id",
    }:
        raise RuntimeError("SEQ513_CURRENT_PAIR_PROOF_ARTIFACT_KEYS_INVALID")
    if set(contracts) != {
        "required_history_start_utc",
        "required_history_start_covered",
        "time_min_utc",
        "time_max_utc",
        "no_fallback",
        "future_rows_used",
    }:
        raise RuntimeError("SEQ513_CURRENT_PAIR_PROOF_CONTRACT_KEYS_INVALID")
    _same_path(artifacts.get("source_parquet_path"), expected_source_parquet, label="CURRENT_PAIR_SOURCE")
    _same(artifacts.get("source_parquet_sha256"), _sha256_file(expected_source_parquet), label="CURRENT_PAIR_SOURCE_HASH")
    _same_path(artifacts.get("canonical_v2_path"), expected_canonical_v2_parquet, label="CURRENT_PAIR_CANONICAL")
    _same(artifacts.get("canonical_v2_sha256"), _sha256_file(expected_canonical_v2_parquet), label="CURRENT_PAIR_CANONICAL_HASH")
    cache = load_multi_tf_cache(expected_mtf_cache_dir)
    _same(artifacts.get("multi_tf_manifest_sha256"), _sha256_file(expected_mtf_cache_dir / "manifest.json"), label="CURRENT_PAIR_MTF_MANIFEST_HASH")
    _same(artifacts.get("multi_tf_cache_identity_sha256"), cache.cache_identity_sha256, label="CURRENT_PAIR_MTF_IDENTITY")
    pair_manifest = Path(str(artifacts.get("pair_manifest_path") or "")).expanduser().resolve()
    _same(artifacts.get("pair_manifest_sha256"), _sha256_file(pair_manifest), label="CURRENT_PAIR_PAIR_MANIFEST_HASH")
    pair = _json(pair_manifest, label="CURRENT_PAIR_MANIFEST")
    _same(artifacts.get("pair_generation_id"), pair.get("pair_generation_id"), label="CURRENT_PAIR_GENERATION_ID")
    expected_history_ts = _utc(expected_history, label="CURRENT_PAIR_EXPECTED_HISTORY")
    expected_max_ts = _utc(expected_time_max, label="CURRENT_PAIR_EXPECTED_TIME_MAX")
    _same(_utc(contracts.get("required_history_start_utc"), label="CURRENT_PAIR_HISTORY"), expected_history_ts, label="CURRENT_PAIR_HISTORY")
    _same(_utc(contracts.get("time_max_utc"), label="CURRENT_PAIR_TIME_MAX"), expected_max_ts, label="CURRENT_PAIR_TIME_MAX")
    _same(contracts.get("required_history_start_covered"), True, label="CURRENT_PAIR_HISTORY_COVERED")
    _same(contracts.get("no_fallback"), True, label="CURRENT_PAIR_NO_FALLBACK")
    _same(contracts.get("future_rows_used"), False, label="CURRENT_PAIR_FUTURE_ROWS")
    return {
        "path": str(proof_path),
        "sha256": _sha256_file(proof_path),
        "schema_version": CURRENT_PAIR_SCHEMA_VERSION,
        "entry_run_id": expected_run_id,
        "event_root": str(event_root),
        "source_parquet_path": str(expected_source_parquet),
        "source_parquet_sha256": str(artifacts["source_parquet_sha256"]),
        "canonical_v2_path": str(expected_canonical_v2_parquet),
        "canonical_v2_sha256": str(artifacts["canonical_v2_sha256"]),
        "multi_tf_cache_dir": str(expected_mtf_cache_dir),
        "multi_tf_manifest_sha256": str(artifacts["multi_tf_manifest_sha256"]),
        "multi_tf_cache_identity_sha256": str(artifacts["multi_tf_cache_identity_sha256"]),
        "pair_manifest_path": str(pair_manifest),
        "pair_manifest_sha256": str(artifacts["pair_manifest_sha256"]),
        "pair_generation_id": str(artifacts["pair_generation_id"]),
        "history_start_utc": expected_history_ts.isoformat(),
        "time_max_utc": expected_max_ts.isoformat(),
    }


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
