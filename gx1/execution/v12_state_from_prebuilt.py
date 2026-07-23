#!/usr/bin/env python3
"""V12 state loader — reads features DIRECTLY from disk prebuilts.

This module replaces v12_canonical_live.py + v12_ctx_augment_live.py.
Those tried to RECOMPUTE features from raw M1 data, which subtly diverged
from training-time distributions (different M5 aggregation, different
rolling-window percentiles for buckets, different bid/ask handling).

The correct architecture: use the EXACT same prebuilt files the cemented
V12.1.1 models were trained on. By construction, live state = batch state,
so XGB / V10 / Entry-IQL / Exit-IQL inference reproduces Phase 6 backtest
output exactly.

Architecture:
    The updater computes both candidate artifacts without touching active state,
    then publishes one immutable generation by atomically replacing:
        CANONICAL_V3_BASE28_CURRENT_PAIR_MANIFEST.json

    Live (every M1 tick):
        - Look up the latest closed M5 bar in canonical_v3 prebuilt
        - Join with BASE28 augmented features at the same timestamp
        - Result: 92-feature XGB input identical to training distribution

The two artifacts in every admitted pair generation:

  canonical_v3 prebuilt (M5 cadence, 112 cols):
    Path: CANONICAL_V3_BASE28_GENERATIONS/<pair_generation_id>/
            canonical_v3.parquet
    Contains: M5 OHLC + 84 canonical_v2 features + 4 cyclic + smc_premium_state
              + m5h1_momentum  (108 features + time + open/high/low/close/volume)
    Built by: materialize_build_canonical_features_v2.py
              + materialize_canonical_v3_augment.py

  BASE28 / BASE34 artifact (M1 cadence, including augmented context):
    Path: CANONICAL_V3_BASE28_GENERATIONS/<pair_generation_id>/
            base28.parquet
    Contains: 32 augmented features (atr_bps, session_id, regime/bucket,
              HTF, micro_momentum, swing, _v1_is_EU/US, _v1_int_*_us, etc.)
    Built by: add_ctx_cont_columns_to_prebuilt.py

Together they provide all 92 features the cemented XGB v5 needs.

The pair pointer, artifact hashes, row counts and physical Arrow schemas are
one fail-closed serving contract. Individual legacy manifests and direct
mutable parquet paths are not admissible fallbacks.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ONE-TRUTH REGIME_V4 per-TF V2 mtf scalar projection (5-TF/9-feat). Shared with the
# canonical_incremental daemon + build so the live serve, the daemon-maintained BASE34,
# and the training BASE34 all project the SAME per-TF regime inputs (no train≠serve).
from gx1.features.htf_features import (
    REGIME_V4_V2_MTF_PER_TF,
    REGIME_V4_V2_MTF_TFS,
    REGIME_V4_V2_MTF_SKIP,
)
from gx1.features.basic_v1 import compute_plus5_features

LOG = logging.getLogger("v12_state_from_prebuilt")


# ── Top-level subprocess workers for parallel augment (2026-06-01) ──
# These run in subprocesses spawned by _async_full_refresh so V2 mtf and
# GROUP-A — the two heavy augmenters — execute in true parallel rather than
# serializing behind the GIL. Each worker takes a pickled cv3, runs ONE
# augmenter via an ephemeral PrebuiltStateLoader instance, returns only the
# new columns (smaller payload back). On Linux fork() is used so cv3 is
# inherited via copy-on-write — actual pickle cost is only the result.
def _mp_v2_mtf_worker(cv3: pd.DataFrame) -> pd.DataFrame:
    """Run V2 multi-TF augmenter in a subprocess. Returns DataFrame with ONLY
    the new V2 mtf columns (suffix '_v2'), indexed identically to cv3 input.

    NB: the augmenter mutates cv3 in-place and returns the same object, so we
    must snapshot the column set BEFORE the call to compute the new-cols delta.
    """
    loader = PrebuiltStateLoader()
    augmented = loader._augment_cv3_with_v2_mtf_scalars(cv3)
    expected = loader._expected_v2_mtf_columns()
    missing = [name for name in expected if name not in augmented.columns]
    if missing:
        raise RuntimeError(f"parallel V2 MTF worker output missing: {missing}")
    return augmented[expected].copy()


def _mp_group_a_worker(cv3: pd.DataFrame) -> pd.DataFrame:
    """Run GROUP-A + DIP/STRUCT augmenter in a subprocess. Returns DataFrame
    with ONLY the new ctx_cont columns, indexed identically to cv3 input.
    """
    from gx1.contracts.entry_model_native_signal_v1 import (
        MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS as _GROUP_A,
        MODEL_NATIVE_CTX_CONT_DIP_STRUCT_FIELDS as _DIP_STRUCT,
    )
    loader = PrebuiltStateLoader()
    augmented = loader._augment_cv3_with_group_a_and_dip_struct(cv3)
    expected = list(_GROUP_A) + list(_DIP_STRUCT)
    missing = [name for name in expected if name not in augmented.columns]
    if missing:
        raise RuntimeError(f"parallel group-A worker output missing: {missing}")
    return augmented[expected].copy()

PREBUILT_PAIR_SCHEMA_VERSION = "gx1_canonical_v3_base28_pair_generation_v1"
PREBUILT_PAIR_ROOT = Path(
    "/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_BASE28_GENERATIONS"
)
PREBUILT_PAIR_MANIFEST_PATH = Path(
    "/home/andre2/GX1_DATA/data/data/prebuilt/"
    "CANONICAL_V3_BASE28_CURRENT_PAIR_MANIFEST.json"
)


class PrebuiltIdentityError(RuntimeError):
    """The atomic pair pointer or either exact parquet identity is invalid."""


@dataclass(frozen=True)
class _PrebuiltManifestBinding:
    label: str
    pair_generation_id: str
    parquet_path: Path
    parquet_sha256: str
    rows: int
    cols_total: int
    arrow_schema: tuple[tuple[str, str, bool], ...]


@dataclass(frozen=True)
class _PrebuiltPairBinding:
    manifest_path: Path
    manifest_sha256: str
    pair_generation_id: str
    canonical_v3: _PrebuiltManifestBinding
    base28: _PrebuiltManifestBinding


@dataclass(frozen=True)
class _VerifiedPrebuilt:
    binding: _PrebuiltManifestBinding
    file_stamp: tuple[int, int, int, int]
    arrow_schema: tuple[tuple[str, str, bool], ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_stamp(path: Path, *, label: str) -> tuple[int, int, int, int]:
    if path.is_symlink() or not path.is_file():
        raise PrebuiltIdentityError(
            f"{label}_PARQUET_NOT_REGULAR_FILE: {path}"
        )
    stat = path.stat()
    return (
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _strict_positive_int(value: object, *, field_name: str, label: str) -> int:
    if isinstance(value, bool):
        raise PrebuiltIdentityError(f"{label}_MANIFEST_{field_name.upper()}_INVALID")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PrebuiltIdentityError(
            f"{label}_MANIFEST_{field_name.upper()}_INVALID"
        ) from exc
    if parsed <= 0 or parsed != value:
        raise PrebuiltIdentityError(f"{label}_MANIFEST_{field_name.upper()}_INVALID")
    return parsed


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_generation_identity(
    artifacts: dict[str, dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": PREBUILT_PAIR_SCHEMA_VERSION,
        "artifacts": {
            label: {
                "parquet_sha256": artifacts[label]["parquet_sha256"],
                "rows": artifacts[label]["rows"],
                "cols_total": artifacts[label]["cols_total"],
                "arrow_schema": artifacts[label]["arrow_schema"],
            }
            for label in ("canonical_v3", "base28")
        },
    }


def pair_generation_id_for_artifacts(
    artifacts: dict[str, dict[str, object]],
) -> str:
    """Return the only pair generation id.

    It is SHA-256 over canonical JSON containing the schema version and, in
    fixed canonical-v3/BASE28 order, each artifact's SHA-256, row count,
    feature-column count and full Arrow schema. Paths and wall-clock time are
    excluded, so identity means content contract rather than publication site.
    """
    return _sha256_json(_artifact_generation_identity(artifacts))


def inspect_prebuilt_artifact(path: Path, *, label: str) -> dict[str, object]:
    """Return the exact manifest contract for one stable parquet candidate."""
    path = Path(path).resolve(strict=True)
    before = _file_stamp(path, label=label.upper())
    parquet_sha256 = _sha256_file(path)
    try:
        parquet = pq.ParquetFile(path)
        rows = int(parquet.metadata.num_rows)
        schema = tuple(
            {
                "name": str(item.name),
                "type": str(item.type),
                "nullable": bool(item.nullable),
            }
            for item in parquet.schema_arrow
        )
    except Exception as exc:  # noqa: BLE001 - normalize format/schema failure
        raise PrebuiltIdentityError(f"{label.upper()}_PARQUET_METADATA_INVALID") from exc
    if rows <= 0 or len(schema) <= 1:
        raise PrebuiltIdentityError(f"{label.upper()}_PARQUET_SHAPE_INVALID")
    names = [str(item["name"]) for item in schema]
    if len(names) != len(set(names)):
        raise PrebuiltIdentityError(
            f"{label.upper()}_PARQUET_SCHEMA_DUPLICATE_COLUMNS"
        )
    after = _file_stamp(path, label=label.upper())
    if before != after:
        raise PrebuiltIdentityError(
            f"{label.upper()}_PARQUET_CHANGED_DURING_INSPECTION"
        )
    return {
        "parquet_path": str(path),
        "parquet_sha256": parquet_sha256,
        "rows": rows,
        "cols_total": len(schema) - 1,
        "arrow_schema": list(schema),
    }


def _parse_pair_artifact(
    raw: object,
    *,
    label: str,
    pair_generation_id: str,
    generation_root: Path,
) -> _PrebuiltManifestBinding:
    if not isinstance(raw, dict):
        raise PrebuiltIdentityError(f"{label.upper()}_PAIR_ARTIFACT_INVALID")
    required = (
        "parquet_path",
        "parquet_sha256",
        "rows",
        "cols_total",
        "arrow_schema",
    )
    if set(raw) != set(required):
        raise PrebuiltIdentityError(
            f"{label.upper()}_PAIR_ARTIFACT_FIELDS_INVALID"
        )

    raw_path = raw["parquet_path"]
    if not isinstance(raw_path, str) or not raw_path or raw_path.strip() != raw_path:
        raise PrebuiltIdentityError(f"{label.upper()}_PAIR_PARQUET_PATH_INVALID")
    declared_path = Path(raw_path)
    if not declared_path.is_absolute():
        raise PrebuiltIdentityError(
            f"{label.upper()}_PAIR_PARQUET_PATH_NOT_ABSOLUTE"
        )
    try:
        resolved_path = declared_path.resolve(strict=True)
    except OSError as exc:
        raise PrebuiltIdentityError(
            f"{label.upper()}_PAIR_PARQUET_PATH_MISSING: {declared_path}"
        ) from exc
    expected_parent = (
        Path(generation_root).resolve(strict=True) / pair_generation_id
    )
    expected_name = (
        "canonical_v3.parquet" if label == "canonical_v3" else "base28.parquet"
    )
    if (
        declared_path != resolved_path
        or declared_path.is_symlink()
        or resolved_path.parent != expected_parent
        or resolved_path.name != expected_name
    ):
        raise PrebuiltIdentityError(
            f"{label.upper()}_PAIR_PARQUET_PATH_NOT_GENERATION_EXACT: "
            f"path={resolved_path} expected={expected_parent / expected_name}"
        )

    parquet_sha256 = raw["parquet_sha256"]
    if (
        not isinstance(parquet_sha256, str)
        or len(parquet_sha256) != 64
        or parquet_sha256.lower() != parquet_sha256
        or any(char not in "0123456789abcdef" for char in parquet_sha256)
    ):
        raise PrebuiltIdentityError(
            f"{label.upper()}_PAIR_PARQUET_SHA256_INVALID"
        )
    raw_schema = raw["arrow_schema"]
    if not isinstance(raw_schema, list) or len(raw_schema) <= 1:
        raise PrebuiltIdentityError(f"{label.upper()}_PAIR_ARROW_SCHEMA_INVALID")
    parsed_schema: list[tuple[str, str, bool]] = []
    for item in raw_schema:
        if not isinstance(item, dict) or set(item) != {"name", "type", "nullable"}:
            raise PrebuiltIdentityError(
                f"{label.upper()}_PAIR_ARROW_SCHEMA_INVALID"
            )
        name = item["name"]
        dtype = item["type"]
        nullable = item["nullable"]
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(dtype, str)
            or not dtype
            or not isinstance(nullable, bool)
        ):
            raise PrebuiltIdentityError(
                f"{label.upper()}_PAIR_ARROW_SCHEMA_INVALID"
            )
        parsed_schema.append((name, dtype, nullable))
    if len({item[0] for item in parsed_schema}) != len(parsed_schema):
        raise PrebuiltIdentityError(
            f"{label.upper()}_PAIR_ARROW_SCHEMA_DUPLICATE_COLUMNS"
        )
    cols_total = _strict_positive_int(
        raw["cols_total"],
        field_name="cols_total",
        label=label.upper(),
    )
    if len(parsed_schema) != cols_total + 1:
        raise PrebuiltIdentityError(
            f"{label.upper()}_PAIR_ARROW_SCHEMA_COUNT_MISMATCH"
        )
    return _PrebuiltManifestBinding(
        label=label.upper(),
        pair_generation_id=pair_generation_id,
        parquet_path=resolved_path,
        parquet_sha256=parquet_sha256,
        rows=_strict_positive_int(
            raw["rows"],
            field_name="rows",
            label=label.upper(),
        ),
        cols_total=cols_total,
        arrow_schema=tuple(parsed_schema),
    )


def read_prebuilt_pair_manifest(
    manifest_path: Path,
    *,
    generation_root: Path,
) -> _PrebuiltPairBinding:
    """Read one stable atomic pair pointer; individual manifests are inadmissible."""
    manifest_path = Path(manifest_path)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise PrebuiltIdentityError(
            f"PREBUILT_PAIR_MANIFEST_NOT_REGULAR_FILE: {manifest_path}"
        )
    before = _file_stamp(manifest_path, label="PREBUILT_PAIR_MANIFEST")
    raw = manifest_path.read_bytes()
    after = _file_stamp(manifest_path, label="PREBUILT_PAIR_MANIFEST")
    if before != after:
        raise PrebuiltIdentityError("PREBUILT_PAIR_MANIFEST_CHANGED_DURING_READ")
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PrebuiltIdentityError("PREBUILT_PAIR_MANIFEST_JSON_INVALID") from exc
    if not isinstance(manifest, dict):
        raise PrebuiltIdentityError("PREBUILT_PAIR_MANIFEST_ROOT_INVALID")
    if set(manifest) != {
        "schema_version",
        "pair_generation_id",
        "created_utc",
        "artifacts",
    }:
        raise PrebuiltIdentityError("PREBUILT_PAIR_MANIFEST_FIELDS_INVALID")
    if manifest["schema_version"] != PREBUILT_PAIR_SCHEMA_VERSION:
        raise PrebuiltIdentityError("PREBUILT_PAIR_SCHEMA_VERSION_INVALID")
    created_utc = manifest["created_utc"]
    if (
        not isinstance(created_utc, str)
        or not created_utc
        or created_utc.strip() != created_utc
    ):
        raise PrebuiltIdentityError("PREBUILT_PAIR_CREATED_UTC_INVALID")
    try:
        parsed_created_utc = datetime.strptime(
            created_utc,
            "%Y-%m-%dT%H:%M:%SZ",
        )
    except ValueError as exc:
        raise PrebuiltIdentityError("PREBUILT_PAIR_CREATED_UTC_INVALID") from exc
    if parsed_created_utc.strftime("%Y-%m-%dT%H:%M:%SZ") != created_utc:
        raise PrebuiltIdentityError("PREBUILT_PAIR_CREATED_UTC_INVALID")
    generation_root = Path(generation_root)
    if generation_root.is_symlink() or not generation_root.is_dir():
        raise PrebuiltIdentityError(
            f"PREBUILT_PAIR_GENERATION_ROOT_INVALID: {generation_root}"
        )
    generation_root = generation_root.resolve(strict=True)
    pair_generation_id = manifest["pair_generation_id"]
    if (
        not isinstance(pair_generation_id, str)
        or len(pair_generation_id) != 64
        or pair_generation_id.lower() != pair_generation_id
        or any(char not in "0123456789abcdef" for char in pair_generation_id)
    ):
        raise PrebuiltIdentityError("PREBUILT_PAIR_GENERATION_ID_INVALID")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "canonical_v3",
        "base28",
    }:
        raise PrebuiltIdentityError("PREBUILT_PAIR_ARTIFACT_SET_INVALID")
    try:
        observed_generation_id = pair_generation_id_for_artifacts(artifacts)
    except (KeyError, TypeError, ValueError) as exc:
        raise PrebuiltIdentityError(
            "PREBUILT_PAIR_ARTIFACT_IDENTITY_INVALID"
        ) from exc
    if observed_generation_id != pair_generation_id:
        raise PrebuiltIdentityError(
            "PREBUILT_PAIR_GENERATION_ID_CONTENT_MISMATCH"
        )
    canonical = _parse_pair_artifact(
        artifacts["canonical_v3"],
        label="canonical_v3",
        pair_generation_id=pair_generation_id,
        generation_root=generation_root,
    )
    base28 = _parse_pair_artifact(
        artifacts["base28"],
        label="base28",
        pair_generation_id=pair_generation_id,
        generation_root=generation_root,
    )
    return _PrebuiltPairBinding(
        manifest_path=manifest_path.resolve(strict=True),
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        pair_generation_id=pair_generation_id,
        canonical_v3=canonical,
        base28=base28,
    )


def _verify_prebuilt(
    binding: _PrebuiltManifestBinding,
    *,
    expected_schema: tuple[tuple[str, str, bool], ...] | None = None,
) -> _VerifiedPrebuilt:
    """Verify exact bytes, Parquet row count and immutable physical schema."""
    before = _file_stamp(binding.parquet_path, label=binding.label)
    observed_sha = _sha256_file(binding.parquet_path)
    if observed_sha != binding.parquet_sha256:
        raise PrebuiltIdentityError(
            f"{binding.label}_PARQUET_SHA256_MISMATCH: "
            f"observed={observed_sha} expected={binding.parquet_sha256}"
        )
    try:
        parquet = pq.ParquetFile(binding.parquet_path)
        observed_rows = int(parquet.metadata.num_rows)
        schema = parquet.schema_arrow
    except Exception as exc:  # noqa: BLE001 - normalize format/schema failure
        raise PrebuiltIdentityError(
            f"{binding.label}_PARQUET_METADATA_INVALID"
        ) from exc
    arrow_schema = tuple(
        (str(item.name), str(item.type), bool(item.nullable))
        for item in schema
    )
    names = [name for name, _dtype, _nullable in arrow_schema]
    if len(names) != len(set(names)):
        raise PrebuiltIdentityError(
            f"{binding.label}_PARQUET_SCHEMA_DUPLICATE_COLUMNS"
        )
    # Both admitted producers persist one timestamp carrier in addition to the
    # manifest's feature-column count: canonical-v3 writes `time` as a column;
    # BASE28 writes its DatetimeIndex into the parquet schema.
    if len(arrow_schema) != binding.cols_total + 1:
        raise PrebuiltIdentityError(
            f"{binding.label}_PARQUET_SCHEMA_COUNT_MISMATCH: "
            f"physical={len(arrow_schema)} manifest_features={binding.cols_total}"
        )
    if observed_rows != binding.rows:
        raise PrebuiltIdentityError(
            f"{binding.label}_PARQUET_ROWS_MISMATCH: "
            f"observed={observed_rows} expected={binding.rows}"
        )
    if arrow_schema != binding.arrow_schema:
        raise PrebuiltIdentityError(
            f"{binding.label}_PARQUET_SCHEMA_MANIFEST_MISMATCH"
        )
    if expected_schema is not None and arrow_schema != expected_schema:
        raise PrebuiltIdentityError(
            f"{binding.label}_PARQUET_SCHEMA_IDENTITY_MISMATCH"
        )
    after = _file_stamp(binding.parquet_path, label=binding.label)
    if before != after:
        raise PrebuiltIdentityError(
            f"{binding.label}_PARQUET_CHANGED_DURING_VERIFICATION"
        )
    return _VerifiedPrebuilt(
        binding=binding,
        file_stamp=after,
        arrow_schema=arrow_schema,
    )


def verify_prebuilt_pair(
    binding: _PrebuiltPairBinding,
    *,
    expected_canonical_schema: tuple[tuple[str, str, bool], ...] | None = None,
    expected_base28_schema: tuple[tuple[str, str, bool], ...] | None = None,
) -> tuple[_VerifiedPrebuilt, _VerifiedPrebuilt]:
    """Verify both immutable artifacts before a pair is computed from or served."""
    canonical = _verify_prebuilt(
        binding.canonical_v3,
        expected_schema=expected_canonical_schema,
    )
    base28 = _verify_prebuilt(
        binding.base28,
        expected_schema=expected_base28_schema,
    )
    return canonical, base28


def _load_verified_prebuilt(
    binding: _PrebuiltManifestBinding,
    *,
    expected_schema: tuple[tuple[str, str, bool], ...] | None = None,
) -> tuple[pd.DataFrame, _VerifiedPrebuilt]:
    verified = _verify_prebuilt(binding, expected_schema=expected_schema)
    try:
        frame = pd.read_parquet(binding.parquet_path)
    except Exception as exc:  # noqa: BLE001 - normalize read failure
        raise PrebuiltIdentityError(f"{binding.label}_PARQUET_READ_FAILED") from exc
    if _file_stamp(binding.parquet_path, label=binding.label) != verified.file_stamp:
        raise PrebuiltIdentityError(
            f"{binding.label}_PARQUET_CHANGED_DURING_LOAD"
        )
    if "time" in frame.columns:
        if isinstance(frame.index, pd.DatetimeIndex):
            raise PrebuiltIdentityError(
                f"{binding.label}_PARQUET_TIMESTAMP_AMBIGUOUS"
            )
        frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
        frame = frame.set_index("time")
    elif isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    else:
        raise PrebuiltIdentityError(
            f"{binding.label}_PARQUET_TIMESTAMP_SCHEMA_MISSING"
        )
    if (
        frame.index.hasnans
        or not frame.index.is_monotonic_increasing
        or not frame.index.is_unique
    ):
        raise PrebuiltIdentityError(
            f"{binding.label}_PARQUET_TIMESTAMP_INDEX_INVALID"
        )
    if len(frame) != binding.rows or len(frame.columns) != binding.cols_total:
        raise PrebuiltIdentityError(
            f"{binding.label}_LOADED_SHAPE_MISMATCH: "
            f"observed={frame.shape} "
            f"expected=({binding.rows}, {binding.cols_total})"
        )
    return frame, verified


@dataclass
class PrebuiltStateLoader:
    pair_manifest_path: Path = PREBUILT_PAIR_MANIFEST_PATH
    generation_root: Path = PREBUILT_PAIR_ROOT
    canonical_v3_path: Path | None = field(default=None, init=False)
    base28_path: Path | None = field(default=None, init=False)
    _cv3: pd.DataFrame | None = field(default=None, init=False)
    _base28: pd.DataFrame | None = field(default=None, init=False)
    _last_ts: pd.Timestamp | None = field(default=None, init=False)
    _pair_binding: _PrebuiltPairBinding | None = field(default=None, init=False)
    _cv3_binding: _PrebuiltManifestBinding | None = field(default=None, init=False)
    _base28_binding: _PrebuiltManifestBinding | None = field(default=None, init=False)
    _cv3_source_schema: tuple[tuple[str, str, bool], ...] | None = field(
        default=None,
        init=False,
    )
    _base28_source_schema: tuple[tuple[str, str, bool], ...] | None = field(
        default=None,
        init=False,
    )
    _cv3_file_stamp: tuple[int, int, int, int] | None = field(
        default=None,
        init=False,
    )
    _base28_file_stamp: tuple[int, int, int, int] | None = field(
        default=None,
        init=False,
    )
    # Retained for diagnostics used elsewhere; identity decisions use the
    # nanosecond/inode/size/ctime stamps above.
    _cv3_mtime: float = field(default=0.0, init=False)
    _base28_mtime: float = field(default=0.0, init=False)
    # 2026-06-01: async refresh — background thread runs the full 5-augmenter
    # pipeline on a LOCAL copy so the M1 exit-path never blocks on cv3 reload.
    # When the thread finishes, it atomically swaps self._cv3 / self._base28 /
    # self._last_ts in. Main loop reads them between calls (Python attribute
    # assignment is atomic under the GIL, so reader sees old or new — never half).
    _refresh_thread: threading.Thread | None = field(default=None, init=False)
    _refresh_enabled: bool = field(default=True, init=False)
    # An async error invalidates the old snapshot. Every public read and future
    # refresh raises until a fresh process completes load() successfully.
    _refresh_error: RuntimeError | None = field(default=None, init=False)

    def _raise_refresh_error(self) -> None:
        if self._refresh_error is not None:
            raise PrebuiltIdentityError(
                f"PREBUILT_REFRESH_LATCHED: {self._refresh_error}"
            ) from self._refresh_error

    def _admit_current_refresh_pair(
        self,
    ) -> tuple[
        _PrebuiltPairBinding,
        _VerifiedPrebuilt,
        _VerifiedPrebuilt,
        bool,
        bool,
    ] | None:
        pair_binding = read_prebuilt_pair_manifest(
            self.pair_manifest_path,
            generation_root=self.generation_root,
        )
        cv3_binding = pair_binding.canonical_v3
        base28_binding = pair_binding.base28
        cv3_stamp = _file_stamp(cv3_binding.parquet_path, label="CANONICAL_V3")
        base28_stamp = _file_stamp(base28_binding.parquet_path, label="BASE28")
        cv3_advanced = (
            pair_binding != self._pair_binding
            or cv3_binding != self._cv3_binding
            or cv3_stamp != self._cv3_file_stamp
        )
        b28_advanced = (
            pair_binding != self._pair_binding
            or base28_binding != self._base28_binding
            or base28_stamp != self._base28_file_stamp
        )
        if not (cv3_advanced or b28_advanced):
            return None

        # Validate the complete pair before starting expensive asynchronous
        # augmentation. Serving identity is the pair, never two independently
        # moving files or individual legacy manifests.
        cv3_verified, base28_verified = verify_prebuilt_pair(
            pair_binding,
            expected_canonical_schema=self._cv3_source_schema,
            expected_base28_schema=self._base28_source_schema,
        )
        return (
            pair_binding,
            cv3_verified,
            base28_verified,
            cv3_advanced,
            b28_advanced,
        )

    def refresh_if_changed(self) -> bool:
        """Verify the atomic pair identity and schedule an exact joint refresh.

        A missing/invalid manifest, path, hash, row count, or schema raises
        synchronously. Background failures are latched and invalidate reads;
        stale in-memory state is never treated as an admissible fallback.
        """
        self._raise_refresh_error()
        if self._cv3_binding is None or self._base28_binding is None:
            raise PrebuiltIdentityError(
                "PREBUILT_BINDINGS_UNINITIALIZED: call load() first"
            )

        # If a verified refresh is already in flight, let it finish. Its final
        # identity recheck either swaps the pair or latches an error.
        if self._refresh_thread is not None and self._refresh_thread.is_alive():
            return False
        if not self._refresh_enabled:
            return False

        try:
            admission = self._admit_current_refresh_pair()
        except PrebuiltIdentityError as exc:
            self._refresh_error = RuntimeError(
                f"PREBUILT_REFRESH_IDENTITY_FAILED: {exc}"
            )
            raise
        if admission is None:
            return False
        (
            pair_binding,
            cv3_verified,
            base28_verified,
            cv3_advanced,
            b28_advanced,
        ) = admission

        # Schedule background refresh.
        t = threading.Thread(
            target=self._async_full_refresh,
            args=(
                cv3_verified,
                base28_verified,
                pair_binding,
                cv3_advanced,
                b28_advanced,
            ),
            daemon=True,
            name="cv3_async_refresh",
        )
        self._refresh_thread = t
        t.start()
        return True

    def _async_full_refresh(
        self,
        cv3_verified: _VerifiedPrebuilt,
        base28_verified: _VerifiedPrebuilt,
        pair_binding: _PrebuiltPairBinding,
        cv3_advanced: bool,
        b28_advanced: bool,
    ) -> None:
        """Background-thread worker: read new cv3 / BASE28 from disk, run all
        5 augmenters on a LOCAL DataFrame, then atomically swap into self.

        Bit-parity with synchronous path is guaranteed because exactly the same
        augmenter methods are called — only the orchestration is threaded.
        """
        try:
            t_start = pd.Timestamp.utcnow()
            # --- cv3 ---
            new_cv3 = self._cv3
            if cv3_advanced:
                new_cv3, loaded_cv3 = _load_verified_prebuilt(
                    cv3_verified.binding,
                    expected_schema=self._cv3_source_schema,
                )
                if loaded_cv3 != cv3_verified:
                    raise PrebuiltIdentityError(
                        "CANONICAL_V3_CHANGED_AFTER_REFRESH_ADMISSION"
                    )
            # --- base28 ---
            new_b28 = self._base28
            if b28_advanced:
                new_b28, loaded_base28 = _load_verified_prebuilt(
                    base28_verified.binding,
                    expected_schema=self._base28_source_schema,
                )
                if loaded_base28 != base28_verified:
                    raise PrebuiltIdentityError(
                        "BASE28_CHANGED_AFTER_REFRESH_ADMISSION"
                    )

            # --- augment local cv3 (does NOT touch self._cv3 yet) ---
            # 2026-06-01: PARALLEL via multiprocessing for V2 mtf + GROUP-A.
            # Threading first attempt gave 0% speedup (GIL serialized the two
            # heavy augmenters which spend their time in Python-level pandas
            # glue). Multiprocessing with fork() gives true parallelism — cv3
            # is inherited via copy-on-write (zero IPC for input), only the
            # new columns are pickled back (small payload). Any multiprocessing
            # failure aborts the refresh; a different execution path is not an
            # admissible train≠serve substitute.
            if new_cv3 is not None and (cv3_advanced or b28_advanced):
                t_aug = pd.Timestamp.utcnow()
                # Fast augmenters in main thread (~3s total)
                new_cv3 = self._augment_cv3_with_volume_features(new_cv3)
                new_cv3 = self._augment_cv3_with_v1_legacy(new_cv3)

                # Heavy augmenters in 2 subprocesses (~95s wall, was 180s sequential)
                try:
                    import multiprocessing as _mp
                    from concurrent.futures import ProcessPoolExecutor
                    fork_ctx = _mp.get_context("fork")
                    with ProcessPoolExecutor(max_workers=2, mp_context=fork_ctx) as pool:
                        f_mtf = pool.submit(_mp_v2_mtf_worker, new_cv3)
                        f_grp = pool.submit(_mp_group_a_worker, new_cv3)
                        mtf_new = f_mtf.result(timeout=600)
                        grp_new = f_grp.result(timeout=600)
                    # Merge new cols back
                    for col in mtf_new.columns:
                        new_cv3[col] = mtf_new[col].values
                    for col in grp_new.columns:
                        new_cv3[col] = grp_new[col].values
                    # V1/R10 completion: run regime_v4 only after the exact
                    # multiprocessing outputs have been merged. At
                    # the live async hot-refresh would otherwise drop the active regime columns ->
                    # v12_v3_live.build_window fail-closed guard crashes the regime exit serve on first refresh.
                    # Runs AFTER the mtf merge above (supplies the 12 {tf}_*_v2 sources); joins D1_dist + is a
                    # no-op at flag=0 (cement bit-identical).
                    new_cv3 = self._augment_cv3_with_regime_v4(new_cv3)
                except Exception as exc:
                    raise RuntimeError(
                        "parallel augment failed; refusing a second transform path"
                    ) from exc

                aug_took = (pd.Timestamp.utcnow() - t_aug).total_seconds()
                LOG.info(
                    f"[parallel-augment] full pipeline finished in {aug_took:.1f}s "
                    "(method=multiprocessing)"
                )

            # Joint cutoff
            if new_cv3 is not None and new_b28 is not None:
                new_last_ts = min(new_cv3.index[-1], new_b28.index[-1])
            elif new_cv3 is not None:
                new_last_ts = new_cv3.index[-1]
            else:
                new_last_ts = self._last_ts

            # --- build multi_tf BEFORE the swap so they stay in lock-step ---
            # 2026-06-01 CRITICAL fix: previously we did atomic swap of self._cv3
            # then called build_multi_tf_features() which mutates self._multi_tf_feats
            # separately. Between those two writes, a reader saw new cv3 with OLD
            # multi_tf — silent V10 feature mismatch. Now we build mtf on the LOCAL
            # new_cv3 and swap cv3 + mtf together (single sequential write block
            # under the GIL).
            new_mtf_bundle = None
            if self._multi_tf_feats is not None and new_cv3 is not None:
                try:
                    new_mtf_bundle = self.build_multi_tf_features(new_cv3)
                except Exception as exc:
                    raise RuntimeError(
                        "multi-TF refresh failed; aborting cv3 swap to avoid new-cv3/stale-mtf split-brain"
                    ) from exc

            # The pair pointer and both parquet files must still be the exact
            # pair admitted before the thread began. A producer advance during
            # augmentation cannot be combined with this older local snapshot.
            final_pair_binding = read_prebuilt_pair_manifest(
                self.pair_manifest_path,
                generation_root=self.generation_root,
            )
            if final_pair_binding != pair_binding:
                raise PrebuiltIdentityError(
                    "PREBUILT_PAIR_MANIFEST_CHANGED_DURING_REFRESH"
                )
            if (
                _file_stamp(
                    final_pair_binding.canonical_v3.parquet_path,
                    label="CANONICAL_V3",
                )
                != cv3_verified.file_stamp
            ):
                raise PrebuiltIdentityError(
                    "CANONICAL_V3_PARQUET_CHANGED_DURING_REFRESH"
                )
            if (
                _file_stamp(
                    final_pair_binding.base28.parquet_path,
                    label="BASE28",
                )
                != base28_verified.file_stamp
            ):
                raise PrebuiltIdentityError(
                    "BASE28_PARQUET_CHANGED_DURING_REFRESH"
                )

            # --- atomic swap (single attribute assignments are GIL-atomic) ---
            old_cutoff = self._cv3.index[-1] if self._cv3 is not None else None
            self._cv3 = new_cv3
            self._base28 = new_b28
            self.canonical_v3_path = cv3_verified.binding.parquet_path
            self.base28_path = base28_verified.binding.parquet_path
            self._cv3_binding = cv3_verified.binding
            self._base28_binding = base28_verified.binding
            self._pair_binding = pair_binding
            self._cv3_file_stamp = cv3_verified.file_stamp
            self._base28_file_stamp = base28_verified.file_stamp
            self._cv3_mtime = cv3_verified.file_stamp[2] / 1_000_000_000.0
            self._base28_mtime = base28_verified.file_stamp[2] / 1_000_000_000.0
            self._last_ts = new_last_ts
            if new_mtf_bundle is not None:
                self._multi_tf_feats = new_mtf_bundle["feats"]
                self._multi_tf_shift = new_mtf_bundle["shift"]
                self._multi_tf_feat_count = new_mtf_bundle["feat_count"]

            took_sec = (pd.Timestamp.utcnow() - t_start).total_seconds()
            new_cutoff = self._cv3.index[-1] if self._cv3 is not None else None
            LOG.info(f"[async-refresh] cv3 cutoff {old_cutoff} → {new_cutoff} "
                     f"(took {took_sec:.1f}s, main loop never blocked)")
        except Exception as exc:
            failure = RuntimeError(
                f"PREBUILT_ASYNC_REFRESH_FAILED: "
                f"{type(exc).__name__}: {exc}"
            )
            self._refresh_error = failure
            LOG.exception("async prebuilt refresh failed closed; old state invalidated")

    def load(self) -> None:
        """Load the one exact atomic canonical-v3 + BASE28 pair generation."""
        pair_binding = read_prebuilt_pair_manifest(
            self.pair_manifest_path,
            generation_root=self.generation_root,
        )
        cv3_binding = pair_binding.canonical_v3
        base28_binding = pair_binding.base28
        cv3, cv3_verified = _load_verified_prebuilt(cv3_binding)
        base28, base28_verified = _load_verified_prebuilt(base28_binding)
        if read_prebuilt_pair_manifest(
            self.pair_manifest_path,
            generation_root=self.generation_root,
        ) != pair_binding:
            raise PrebuiltIdentityError(
                "PREBUILT_PAIR_MANIFEST_CHANGED_DURING_INITIAL_LOAD"
            )
        if (
            _file_stamp(cv3_binding.parquet_path, label="CANONICAL_V3")
            != cv3_verified.file_stamp
            or _file_stamp(base28_binding.parquet_path, label="BASE28")
            != base28_verified.file_stamp
        ):
            raise PrebuiltIdentityError(
                "PREBUILT_PARQUET_CHANGED_DURING_INITIAL_LOAD"
            )

        LOG.info(f"loading canonical_v3 prebuilt: {cv3_binding.parquet_path.name}")
        self._cv3 = cv3
        self.canonical_v3_path = cv3_binding.parquet_path
        self._cv3_binding = cv3_binding
        self._pair_binding = pair_binding
        self._cv3_source_schema = cv3_verified.arrow_schema
        self._cv3_file_stamp = cv3_verified.file_stamp
        self._cv3_mtime = cv3_verified.file_stamp[2] / 1_000_000_000.0
        LOG.info(f"  canonical_v3: {len(cv3):,} rows × {len(cv3.columns)} cols  "
                  f"range {cv3.index[0]} → {cv3.index[-1]}")

        LOG.info(f"loading BASE28 prebuilt: {base28_binding.parquet_path.name}")
        self._base28 = base28
        self.base28_path = base28_binding.parquet_path
        self._base28_binding = base28_binding
        self._base28_source_schema = base28_verified.arrow_schema
        self._base28_file_stamp = base28_verified.file_stamp
        self._base28_mtime = base28_verified.file_stamp[2] / 1_000_000_000.0
        LOG.info(f"  BASE28: {len(base28):,} rows × {len(base28.columns)} cols  "
                  f"range {base28.index[0]} → {base28.index[-1]}")

        self._last_ts = min(cv3.index[-1], base28.index[-1])
        LOG.info(f"  effective live cutoff (min of both): {self._last_ts}")
        self._augment_cv3_with_volume_features()
        self._augment_cv3_with_v2_mtf_scalars()
        self._augment_cv3_with_group_a_and_dip_struct()
        self._augment_cv3_with_v1_legacy()
        self._augment_cv3_with_regime_v4()  # after v2_mtf sources; immutable active transform
        self._refresh_error = None

    # ── Exit-XGB V2 multi-TF scalar augmentation ──────────────────────
    # The exact bundle contract consumes 31 V2-suffixed columns per row.
    # ONE TRUTH (2026-06-13): the per-TF V2 mtf projection now lives in
    # gx1.features.htf_features.REGIME_V4_V2_MTF_* and is imported at module top, so the serve,
    # the canonical_incremental daemon (BASE34 ctx recompute), and the build all use the IDENTICAL
    # 5-TF/9-feat regime projection. (m5 ADDED 2026-06-05 — regime ALL-5; trend_age_bars_norm = R2
    # for gx1.features.regime_v4_features.) Was inline here; promoted to kill the duplication.
    _V2_MTF_PER_TF = REGIME_V4_V2_MTF_PER_TF
    _V2_MTF_TFS = REGIME_V4_V2_MTF_TFS
    _V2_MTF_SKIP = REGIME_V4_V2_MTF_SKIP

    @classmethod
    def _expected_v2_mtf_columns(cls) -> list[str]:
        return [
            f"{tf}_{live_frag}_v2"
            for tf in cls._V2_MTF_TFS
            for live_frag, _source_col in cls._V2_MTF_PER_TF
            if (tf, live_frag) not in cls._V2_MTF_SKIP
        ]

    # ── V1 / R10 (2026-06-04): REGIME_V4 exit-context augmentation ────────
    def _augment_cv3_with_regime_v4(self, cv3: pd.DataFrame | None = None) -> pd.DataFrame | None:
        """Compute the 16 REGIME_V4 EXIT_IO_V8 features on the FULL cv3 through
        the canonical `add_regime_v4_features` owner. Any future V3 dataset
        builder must call the same owner and prove parity. This immutable active
        transform MUST run AFTER
        _augment_cv3_with_v2_mtf_scalars (supplies the 12 {tf}_*_v2 source cols) and on the FULL frame
        (F4 d1_dist_roc_288 / F9 bars_since_d1_regime_change need >=288-bar D1 history; a 96-bar window
        would clip them to 0 = a second-order skew). Fail-closed: raises on a missing source col.
        """
        in_place = cv3 is None
        target = cv3 if cv3 is not None else self._cv3
        if target is None:
            raise RuntimeError("REGIME_V4 augmentation requires a loaded canonical_v3 frame")
        from gx1.features.regime_v4_features import (
            add_regime_v4_features as _add_rv4,
        )
        # D1_dist_from_ema200_atr is the ONLY REGIME_V4 source not emitted by _augment_cv3_with_v2_mtf_scalars
        # (which supplies the 12 {tf}_*_v2). Join it from base28 (capital-D1) by timestamp before computing.
        if self._base28 is None or "D1_dist_from_ema200_atr" not in self._base28.columns:
            raise RuntimeError(
                "[REGIME_V4_SERVE] D1_dist_from_ema200_atr missing from base28 — cannot compute the "
                "EXIT_IO_V8 REGIME_V4 tail at serve (would be train!=serve). Provide it on the prebuilt."
            )
        source = self._base28["D1_dist_from_ema200_atr"]
        missing_index = target.index[~target.index.isin(source.index)]
        if len(missing_index):
            raise RuntimeError(
                "[REGIME_V4_SERVE] exact D1 distance timestamps missing: "
                f"count={len(missing_index)} first={missing_index[0]}"
            )
        target["D1_dist_from_ema200_atr"] = source.loc[target.index]
        # add_regime_v4_features needs time-ascending order (shift/run-length); cv3 is sorted at load.
        _add_rv4(target)  # in-place; one-truth; raises on any missing {tf}_*_v2 source col
        if in_place:
            self._cv3 = target
        return target

    def _augment_cv3_with_group_a_and_dip_struct(self, cv3: pd.DataFrame | None = None) -> pd.DataFrame | None:
        """Add 24 GROUP-A parity + 36 DIP/STRUCT ctx_cont columns to cv3
        (defaults to self._cv3). Returns the augmented DataFrame.

        One-truth: re-uses `attach_group_a_dip_struct_ctx_columns` from
        `gx1.scripts.augment_forward_outcome_v2` — same function the V10 builder
        + V3 exit builder + inference-batch candidate gen use. Train/serve skew
        impossible by construction.

        Cost: ~10-20 min at first load (456K cv3 rows × per-bar augment_candidate).
        Cached in cv3 thereafter; re-runs only on cv3 reload (refresh_if_changed).
        """
        in_place = cv3 is None
        target = cv3 if cv3 is not None else self._cv3
        if target is None:
            raise RuntimeError("group-A augmentation requires a loaded canonical_v3 frame")
        from gx1.contracts.entry_model_native_signal_v1 import (
            MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS as _GROUP_A,
            MODEL_NATIVE_CTX_CONT_DIP_STRUCT_FIELDS as _DIP_STRUCT,
        )
        missing_ohlc = [c for c in ("open", "high", "low", "close", "volume") if c not in target.columns]
        if missing_ohlc:
            raise RuntimeError(
                f"group-A augmentation requires exact canonical OHLCV sources: {missing_ohlc}"
            )
        LOG.info(f"augmenting canonical_v3 with {len(_GROUP_A)} GROUP-A + "
                 f"{len(_DIP_STRUCT)} DIP/STRUCT ctx_cont cols "
                 f"(~10-20 min for {len(target):,} M5 rows)...")
        from gx1.scripts.augment_forward_outcome_v2 import (
            attach_group_a_dip_struct_ctx_columns,
        )
        from gx1.features.htf_features import build_multi_tf_per_bar_features_v2
        # LIVE serve: build the multi-TF bundle IN-MEMORY from THIS cv3 and pass it, instead
        # of reading the on-disk MULTI_TF_V2_CACHE. The disk cache is only as fresh as its
        # last rebuild — nothing in the live loop advances it, so it eventually trips the
        # 2-day stale guard (live breaks) or, worse, would serve stale HTF context. Building
        # from the live cv3 makes the group-A/dip-struct ctx ALWAYS current and removes the
        # last live disk-cache dependency. Same builder + same bars as the disk cache (which
        # prebuild_multi_tf_cache_v2 also makes via build_multi_tf_per_bar_features_v2), and
        # the features are causal/asof → bit-identical at every decision ts (train==serve).
        mtf_in_mem = build_multi_tf_per_bar_features_v2(target)
        # The helper mutates in place via the returned DataFrame; rebind to be safe.
        target = attach_group_a_dip_struct_ctx_columns(
            target, journal_label="live_costfix", multi_tf=mtf_in_mem,
        )
        need = list(_GROUP_A) + list(_DIP_STRUCT)
        missing = [name for name in need if name not in target.columns]
        if missing:
            raise RuntimeError(f"group-A augmentation output contract incomplete: {missing}")
        if in_place:
            self._cv3 = target
        LOG.info(f"  group_a/dip_struct augment done: cv3 now {len(target.columns)} cols")
        return target

    # ── Legacy canonical_v1 features Entry-IQL trained on ────────────────
    _V1_LEGACY_FEATURES = (
        "atr",                  # M5 ATR14 (price units)
        "std50",                # M5 50-bar close-return std
        "roc20",                # M5 20-bar rate-of-change
        "_v1_vwap_drift48",     # M5 48-bar VWAP drift fraction
        "_v1h1_vwap_drift",     # H1 VWAP drift fraction (last H1 bar)
    )

    def _augment_cv3_with_v1_legacy(self, cv3: pd.DataFrame | None = None) -> pd.DataFrame | None:
        """Compute 5 legacy canonical_v1 features Entry-IQL was trained on but
        which the canonical_v3 prebuilt does not carry. Same compute as the
        v1 canonical augmenter — values match training within FP noise.
        Idempotent.
        """
        in_place = cv3 is None
        target = cv3 if cv3 is not None else self._cv3
        if target is None:
            raise RuntimeError("legacy feature augmentation requires a loaded canonical_v3 frame")
        cv3 = target
        missing_ohlcv = [
            c for c in ("open", "high", "low", "close", "volume") if c not in cv3.columns
        ]
        if missing_ohlcv:
            raise RuntimeError(f"legacy feature augmentation missing exact OHLCV: {missing_ohlcv}")
        LOG.info(f"augmenting canonical_v3 with {len(self._V1_LEGACY_FEATURES)} "
                 f"legacy canonical_v1 features (atr/std50/roc20/vwap_drifts)")

        plus5 = compute_plus5_features(cv3)
        new_cols = plus5[list(self._V1_LEGACY_FEATURES)].copy()
        existing = [c for c in new_cols.columns if c in cv3.columns]
        if existing:
            cv3 = cv3.drop(columns=existing)
        cv3 = pd.concat([cv3, new_cols], axis=1)
        if in_place:
            self._cv3 = cv3

        LOG.info(f"  v1 legacy augment done: cv3 now {len(cv3.columns)} cols")
        return cv3

    def _augment_cv3_with_volume_features(self, cv3: pd.DataFrame | None = None) -> pd.DataFrame | None:
        """Add 4 volume features (VOLUME_FEATURE_NAMES) to cv3 (defaults to self._cv3).
        Pure-ish: mutates the cv3 you pass in (or self._cv3 by default) and returns it.
        Idempotent — if all are present, skip.
        """
        target = cv3 if cv3 is not None else self._cv3
        if target is None:
            raise RuntimeError("volume augmentation requires a loaded canonical_v3 frame")
        from gx1.features.volume_features import (
            compute_volume_features, VOLUME_FEATURE_NAMES,
        )
        if "volume" not in target.columns or "close" not in target.columns:
            raise RuntimeError("volume augmentation requires exact volume and close sources")
        LOG.info(f"augmenting canonical_v3 with {len(VOLUME_FEATURE_NAMES)} "
                 f"volume features (vol_z_20, vol_ratio_5_20, ...)")
        feats = compute_volume_features(target[["volume", "close"]])
        missing = [name for name in VOLUME_FEATURE_NAMES if name not in feats]
        if missing:
            raise RuntimeError(f"volume feature owner output incomplete: {missing}")
        for name in VOLUME_FEATURE_NAMES:
            target[name] = feats[name]
        LOG.info(f"  volume features added: cv3 now {len(target.columns)} cols")
        return target

    def _augment_cv3_with_v2_mtf_scalars(self, cv3: pd.DataFrame | None = None) -> pd.DataFrame | None:
        """Add 31 V2 multi-TF scalar columns to cv3 (defaults to self._cv3).
        Returns the augmented DataFrame. Idempotent — if columns are already
        present (e.g. from a rebuilt cv3 parquet), skip.
        """
        target = cv3 if cv3 is not None else self._cv3
        if target is None:
            raise RuntimeError("V2 MTF augmentation requires a loaded canonical_v3 frame")
        from gx1.features.htf_features import attach_v2_mtf_per_bar_scalars
        cv3 = target
        ohlc = ["open", "high", "low", "close"]
        missing_ohlcv = [c for c in (*ohlc, "volume") if c not in cv3.columns]
        if missing_ohlcv:
            raise RuntimeError(f"V2 MTF augmentation missing exact OHLCV: {missing_ohlcv}")
        m5_df = cv3[ohlc].copy()
        for c in ohlc:
            m5_df[c] = m5_df[c].astype(np.float64)
        m5_df["volume"] = cv3["volume"].astype(np.float64)
        LOG.info(f"augmenting canonical_v3 with V2 multi-TF scalar features "
                 f"(31 cols expected) — {len(m5_df):,} M5 rows...")
        # V2 (2026-06-04 one-truth): per-bar projection now lives in the SHARED
        # htf_features.attach_v2_mtf_per_bar_scalars (build_multi_tf_per_bar_features_v2 +
        # MULTI_TF_SHIFT only-closed-bars searchsorted) so the V3 exit builder produces the
        # IDENTICAL {tf}_*_v2 cols (was an inline loop here -> the builder's stale join drifted to
        # 88-95% match). Byte-identical to the prior loop.
        cv3_ts_ns = cv3.index.values.astype("datetime64[ns]").astype(np.int64)
        _v2cols = attach_v2_mtf_per_bar_scalars(
            m5_df, cv3_ts_ns, self._V2_MTF_PER_TF, self._V2_MTF_TFS, self._V2_MTF_SKIP,
        )
        expected = self._expected_v2_mtf_columns()
        missing = [name for name in expected if name not in _v2cols]
        if missing:
            raise RuntimeError(f"V2 MTF feature owner output incomplete: {missing}")
        for _col, _vals in _v2cols.items():
            cv3[_col] = _vals
        LOG.info(f"  V2 mtf augment done: +{len(_v2cols)} cols  "
                 f"(cv3 now {len(cv3.columns)} cols total)")
        return cv3

    # ── public API ────────────────────────────────────────────────────

    @property
    def cutoff_ts(self) -> pd.Timestamp:
        """Latest M5 bar timestamp available in BOTH prebuilts (joint coverage)."""
        self._raise_refresh_error()
        if self._last_ts is None:
            raise RuntimeError("loader not initialized — call .load() first")
        return self._last_ts

    @property
    def pair_generation_id(self) -> str:
        """Content identity shared by the two currently loaded artifacts."""
        self._raise_refresh_error()
        if self._pair_binding is None:
            raise RuntimeError("loader not initialized — call .load() first")
        return self._pair_binding.pair_generation_id

    def get_window(self, end_ts: pd.Timestamp, n_bars: int = 96) -> pd.DataFrame:
        """Return the last `n_bars` M5 bars up to and including `end_ts`,
        joined from canonical_v3 + BASE28 prebuilts.

        Latest row is the decision bar (at end_ts.floor('5min')).
        Empty DataFrame if `end_ts` is past the prebuilt coverage.
        """
        self._raise_refresh_error()
        if self._cv3 is None or self._base28 is None:
            raise RuntimeError("loader not initialized — call .load() first")

        end_bucket = end_ts.floor("5min")
        if end_bucket > self.cutoff_ts:
            LOG.warning(f"decision_ts {end_bucket} is past prebuilt cutoff {self.cutoff_ts} — "
                         f"run canonical rebuild cron")
            return pd.DataFrame()

        cv3_win = self._cv3.loc[:end_bucket].tail(n_bars)
        if cv3_win.empty:
            return pd.DataFrame()

        # Join BASE28 augmented columns at exact timestamps. Missing timestamps
        # are unavailable evidence; they may not be reconstructed from a later
        # M1 row inside the bucket.
        b28_cols = [c for c in self._base28.columns if c not in cv3_win.columns]
        if b28_cols:
            missing_index = cv3_win.index[~cv3_win.index.isin(self._base28.index)]
            if len(missing_index):
                raise RuntimeError(
                    "BASE28 exact timestamp contract missing rows: "
                    f"count={len(missing_index)} first={missing_index[0]}"
                )
            b28_slice = self._base28.loc[cv3_win.index, b28_cols]
            joined = pd.concat([cv3_win, b28_slice], axis=1)
        else:
            joined = cv3_win.copy()
        return joined

    # ── Multi-TF (V12.2) ──────────────────────────────────────────────
    _multi_tf_feats: dict | None = field(default=None, init=False)
    _multi_tf_shift: dict | None = field(default=None, init=False)
    _multi_tf_feat_count: int = field(default=0, init=False)

    def build_multi_tf_features(self, cv3: pd.DataFrame | None = None) -> dict | None:
        """Build M5/M15/H1/H4/D1 per-bar feature tables from cv3 (defaults to
        self._cv3). Returns a dict {feats, shift, feat_count} that the caller
        can swap atomically into self.* — required by _async_full_refresh so
        the multi_tf cache is never inconsistent with the cv3 it was built from.

        Called once at runner startup (post-load) AND on every async refresh
        cycle. When `cv3` is None (legacy callers) the result is also assigned
        back to self.* for backward-compat.

        2026-05-29: switched to V2 helper (25 features per TF, includes the
        feature set the COSTFIX V10 + V3 v7 multi-TF were trained on). V1
        helper (17 features per TF) was retired with V12.2 cement.
        2026-06-01: refactored to return dict so async refresh can atomic-swap
        cv3 + multi_tf together (fixes CRITICAL race that produced silent V10
        feature mismatch during cv3 swap window).
        """
        self._raise_refresh_error()
        source = cv3 if cv3 is not None else self._cv3
        if source is None:
            raise RuntimeError("call .load() before .build_multi_tf_features()")
        from gx1.features.htf_features import (
            build_multi_tf_per_bar_features_v2, MULTI_TF_SHIFT,
        )
        ohlc_cols = ["open", "high", "low", "close"]
        missing = [c for c in (*ohlc_cols, "volume") if c not in source.columns]
        if missing:
            raise RuntimeError(f"canonical_v3 missing exact OHLCV cols: {missing}")
        m5_df = source[ohlc_cols].copy()
        m5_df["volume"] = source["volume"].astype(np.float64)
        LOG.info(f"building multi-TF V2 features (M5/M15/H1/H4/D1, 25 dim each) "
                 f"from {len(m5_df):,} M5 bars...")
        feats_dict = build_multi_tf_per_bar_features_v2(m5_df)
        feat_count = 25
        for tf, feats in feats_dict.items():
            if len(feats) > 0:
                feat_count = int(feats.shape[1])
                break
        for tf, feats in feats_dict.items():
            LOG.info(f"  multi-TF V2 {tf}: {len(feats):,} bars × {feats.shape[1]} feats")
        result = {
            "feats": feats_dict,
            "shift": MULTI_TF_SHIFT,
            "feat_count": feat_count,
        }
        # Backward-compat: legacy callers (load() and direct user calls without
        # cv3 arg) expect side-effect on self.*. When cv3 IS provided the caller
        # is responsible for the atomic swap.
        if cv3 is None:
            self._multi_tf_feats = result["feats"]
            self._multi_tf_shift = result["shift"]
            self._multi_tf_feat_count = result["feat_count"]
        return result

    def get_multi_tf_windows(self, end_ts: pd.Timestamp, n_bars: int = 96
                              ) -> dict[str, np.ndarray]:
        """Return {seq_m5, seq_m15, seq_h1, seq_h4, seq_d1} arrays at-or-before end_ts.
        Each is (n_bars, n_features) float32. Zero-padded at start if warmup unmet.

        Empty dict if multi-TF features aren't built (caller falls back to v3 path)."""
        self._raise_refresh_error()
        if self._multi_tf_feats is None:
            return {}
        from gx1.features.htf_features import get_last_n_at_or_before
        return {
            f"seq_{tf.lower()}": get_last_n_at_or_before(
                feats, end_ts, n=n_bars, tf_shift=self._multi_tf_shift[tf]
            )
            for tf, feats in self._multi_tf_feats.items()
        }
