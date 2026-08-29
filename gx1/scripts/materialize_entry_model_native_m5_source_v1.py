"""Materialize the exact offline Entry M5 model source with bounded memory.

The enriched M5 surface is immutable and already owns all code-derived local
features.  This producer only reattaches the literal native bid/ask fields and
the cache-owned raw trend-age scalars by exact timestamp. Inputs are validated
before data pages are read, parquet is consumed in bounded record batches, and
the parquet/manifest pair is published atomically with no replacement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gx1.contracts.entry_exit_feature_base_v1 import (
    ENTRY_DECISION_BAR_SECONDS,
    ENTRY_EXIT_ENRICHED_CAUSAL_FRAME_SCHEMA_VERSION,
    entry_exit_shared_feature_base_contract,
    require_entry_exit_shared_feature_base_contract,
)
from gx1.contracts.entry_exit_production_architecture_v1 import (
    current_entry_exit_architecture_observation,
    require_entry_exit_production_architecture,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS,
    MODEL_NATIVE_CTX_CONT_REGIME_FIELDS,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.contracts.gx1_scope_v1 import require_offline_scope
from gx1.contracts.xau_tape_provenance_v1 import (
    CANONICAL_NATIVE_SOURCE_SCHEMA,
    CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA,
)
from gx1.features.htf_features import (
    HTF_V4_CACHE_SCHEMA_VERSION,
    MULTI_TF_TIMEFRAMES_LOWER_M5_LAST,
    compute_htf_v4_cache_identity,
    load_multi_tf_v4_cache,
    project_multi_tf_v4_scalars,
)
from gx1.scripts.materialize_pretest_native_pair_lineage_v1 import (
    PAIR_LINEAGE_SCHEMA_VERSION as PRETEST_NATIVE_PAIR_LINEAGE_SCHEMA_VERSION,
    TEST_BOUNDARY_UTC as PRETEST_TEST_BOUNDARY_UTC,
)
MAX_BATCH_ROWS = 32_768
MAX_NATIVE_COMPACT_BYTES = 512 * 1024 * 1024
M5_NS = ENTRY_DECISION_BAR_SECONDS * 1_000_000_000
M5_SOURCE_SCHEMA_VERSION = "gx1_entry_model_native_m5_source_surface_v4"
PAIR_MANIFEST_SCHEMA_VERSION = "gx1_canonical_v3_raw_base28_pair_generation_v3"
PAIR_LINEAGE_SCHEMA_VERSION = "gx1_native_pair_lineage_v2"
# A sealed pre-TEST history is a canonical V3 native source.  A later sealed
# successor can be V4.  Both are the one native OHLCV contract; the pair
# manifest, exact file hashes, row/time bounds and Arrow schema below remain
# mandatory.  Do not accept an arbitrary version string just because its
# column layout happens to match.
NATIVE_SOURCE_SCHEMA_VERSIONS = frozenset(
    (
        CANONICAL_NATIVE_SOURCE_SCHEMA,
        CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA,
    )
)

RAW_MARKET_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "ask_open",
    "ask_high",
    "ask_low",
    "ask_close",
    "volume",
)
NATIVE_SOURCE_COLUMNS = ("time", *RAW_MARKET_COLUMNS)
NATIVE_FLOAT_COLUMNS = tuple(
    name for name in RAW_MARKET_COLUMNS if name != "volume"
)
NATIVE_FLOAT_INDEX = {
    name: index for index, name in enumerate(NATIVE_FLOAT_COLUMNS)
}
MARKET_IDENTITY_COLUMNS = (
    "time", "high", "low", "close", "bid_close", "ask_close"
)
ENRICHED_COLUMNS = tuple(
    dict.fromkeys(
        (
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "atr",
            # Canonical-v3 cross-timeframe momentum owner, carried on both
            # native clocks. The M1 enriched builder declares it in the same
            # position; the two lanes must stay resolution-symmetric.
            "m5h1_momentum",
            *MODEL_NATIVE_BASE_FIELDS,
            *MODEL_NATIVE_CTX_CONT_FIELDS,
            *MODEL_NATIVE_CTX_CAT_FIELDS,
        )
    )
)
SOURCE_OWNED_FIELDS = tuple(
    dict.fromkeys((*MODEL_NATIVE_CTX_CONT_REGIME_FIELDS, "volume"))
)
RANKER_OWNED_DERIVATIONS = tuple(
    dict.fromkeys(MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS)
)
# 2026-08-18 (V30 wave 2) — renamed from TREND_AGE_* because the block no
# longer contains only trend ages: the contract added the five
# `{tf}_ema_stack_aligned_v2` companions that make the five
# `{tf}_trend_state_age_bars_v2` readable at all, and both are produced by the
# same compact projection. Keeping the old name would have been a restated
# claim about the contents (rule 13).
#
# The field list is DERIVED from the contract's regime tuple rather than
# matched by name suffix: `d1_dist_change_1bar_atr_v4` is the single member of
# that tuple which is not a per-TF projection output (it is the D1 distance
# owner's own first difference, computed upstream), so it is named as the one
# exclusion. A suffix filter would silently drop any future companion whose
# spelling nobody thought to add here — which is exactly how the ema-stack
# columns would have skipped the cross-check below.
REGIME_COMPACT_PROJECTION = (
    ("trend_state_age_bars", "trend_state_age_bars"),
    ("ema_stack_aligned", "ema_stack_aligned_v2"),
)
REGIME_PROJECTED_FIELDS = tuple(
    name
    for name in MODEL_NATIVE_CTX_CONT_REGIME_FIELDS
    if name != "d1_dist_change_1bar_atr_v4"
)
OUTPUT_COLUMNS = tuple(
    dict.fromkeys(
        (
            "time",
            *RAW_MARKET_COLUMNS,
            *(
                name
                for name in ENRICHED_COLUMNS
                if name not in {"time", *RAW_MARKET_COLUMNS}
                and name not in RANKER_OWNED_DERIVATIONS
            ),
            *REGIME_PROJECTED_FIELDS,
        )
    )
)


@dataclass(frozen=True)
class _FileSeal:
    path: Path
    device: int
    inode: int
    size_bytes: int
    mtime_ns: int
    sha256: str


@dataclass(frozen=True)
class _NativePart:
    year_key: str
    path: Path
    rows: int
    seal: _FileSeal


@dataclass
class _NativeArrays:
    times_ns: np.ndarray
    floats: np.ndarray
    volume: np.ndarray

    @property
    def nbytes(self) -> int:
        return int(self.times_ns.nbytes + self.floats.nbytes + self.volume.nbytes)


def _sha256_file(path: Path) -> str:
    return _seal_file(path).sha256


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


def _exact_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise RuntimeError(f"M5_SOURCE_{label}_SHA256_INVALID")
    return value


def _exact_positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"M5_SOURCE_{label}_INVALID")
    return int(value)


def _require_lineage_id(value: Any, *, label: str, sha256: bool = False) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RuntimeError(f"M5_SOURCE_{label}_INVALID")
    if any(char.isspace() or ord(char) < 33 for char in value):
        raise RuntimeError(f"M5_SOURCE_{label}_INVALID")
    if sha256:
        _exact_sha256(value, label=label)
    return value


def _path_has_symlink_component(path: Path) -> bool:
    return any(component.is_symlink() for component in (path, *path.parents))


def _absolute_path(path: Path, *, label: str) -> Path:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute():
        raise RuntimeError(f"M5_SOURCE_{label}_PATH_NOT_ABSOLUTE: {supplied}")
    normalized = Path(os.path.abspath(os.fspath(supplied)))
    if normalized != supplied or _path_has_symlink_component(normalized):
        raise RuntimeError(f"M5_SOURCE_{label}_PATH_INVALID: {supplied}")
    return normalized


def _require_regular_file(path: Path, *, label: str) -> Path:
    resolved = _absolute_path(path, label=label)
    try:
        mode = os.lstat(resolved).st_mode
    except OSError as exc:
        raise RuntimeError(f"M5_SOURCE_{label}_FILE_INVALID: {resolved}") from exc
    if not stat.S_ISREG(mode):
        raise RuntimeError(f"M5_SOURCE_{label}_FILE_INVALID: {resolved}")
    return resolved


def _require_directory(path: Path, *, label: str) -> Path:
    resolved = _absolute_path(path, label=label)
    try:
        mode = os.lstat(resolved).st_mode
    except OSError as exc:
        raise RuntimeError(f"M5_SOURCE_{label}_DIRECTORY_INVALID: {resolved}") from exc
    if not stat.S_ISDIR(mode):
        raise RuntimeError(f"M5_SOURCE_{label}_DIRECTORY_INVALID: {resolved}")
    return resolved


def _require_new_outputs(output: Path) -> tuple[Path, Path]:
    output_path = _absolute_path(output, label="OUTPUT")
    if output_path.suffix != ".parquet":
        raise RuntimeError("M5_SOURCE_OUTPUT_SUFFIX_INVALID")
    _require_directory(output_path.parent, label="OUTPUT_PARENT")
    manifest_path = Path(f"{output_path}.manifest.json")
    for label, path in (("OUTPUT", output_path), ("OUTPUT_MANIFEST", manifest_path)):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"M5_SOURCE_{label}_ALREADY_EXISTS")
    return output_path, manifest_path


def _seal_file(path: Path) -> _FileSeal:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"M5_SOURCE_FILE_SEAL_OPEN_FAILED: {path}") from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"M5_SOURCE_FILE_SEAL_NOT_REGULAR: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise RuntimeError(f"M5_SOURCE_FILE_CHANGED_WHILE_HASHING: {path}")
    return _FileSeal(
        path=Path(path),
        device=int(after.st_dev),
        inode=int(after.st_ino),
        size_bytes=int(after.st_size),
        mtime_ns=int(after.st_mtime_ns),
        sha256=digest.hexdigest(),
    )


def _require_seal_unchanged(seal: _FileSeal) -> None:
    if _seal_file(seal.path) != seal:
        raise RuntimeError(f"M5_SOURCE_INPUT_CHANGED_DURING_BUILD: {seal.path}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json_sealed(
    path: Path,
    *,
    label: str,
    # Bound on a *malformed* file, not a contract on a legitimate one: this guard
    # exists so a corrupt or adversarial JSON cannot be read into memory unbounded.
    # It was 16 MiB with no stated origin and became smaller than real content on
    # 2026-08-20, failing the chain at m5-model-source on a perfectly valid cache
    # manifest.  MEASURED, two points on the declared tape: a V4 cache manifest is
    # 9.00 MB for a 1-year TRAIN fit (70,668 M5 rows, V31 chains) and 20.04 MB for
    # a 4-year fit (283,883 rows, V32_CHAIN_20260820T084951Z).  All of the growth
    # is v29_registry_constants.provenance.level_recurrence_threshold, the
    # competing-risk fit observation stream, which scales with TRAIN rows.  Linear
    # in rows, the tape's own ceiling on train_start (2020-10-29, 5.77 years,
    # ~409k TRAIN M5 rows) extrapolates to ~26.5 MB.  64 MiB is the next power of
    # two that clears that ceiling with margin, so no window this tape can declare
    # can trip it; anything larger is not a bigger fit, it is a malformed file.
    max_bytes: int = 64 * 1024 * 1024,
) -> tuple[dict[str, Any], _FileSeal]:
    regular = _require_regular_file(path, label=label)
    seal = _seal_file(regular)
    if seal.size_bytes <= 0 or seal.size_bytes > max_bytes:
        raise RuntimeError(f"M5_SOURCE_{label}_JSON_SIZE_INVALID")
    try:
        payload = json.loads(
            regular.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError(f"M5_SOURCE_{label}_JSON_INVALID") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"M5_SOURCE_{label}_JSON_INVALID")
    _require_seal_unchanged(seal)
    return payload, seal


def _timestamp_ns(value: Any, *, label: str) -> int:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"M5_SOURCE_{label}_TIME_INVALID") from exc
    if timestamp.tzinfo is None:
        raise RuntimeError(f"M5_SOURCE_{label}_TIME_INVALID")
    timestamp = timestamp.tz_convert("UTC")
    if timestamp.value % M5_NS != 0:
        raise RuntimeError(f"M5_SOURCE_{label}_TIME_OFF_M5_GRID")
    return int(timestamp.value)


def _native_schema_valid(schema: pa.Schema) -> bool:
    expected_types = {
        "time": pa.timestamp("ns", tz="UTC"),
        **{name: pa.float64() for name in NATIVE_FLOAT_COLUMNS},
        "volume": pa.int64(),
    }
    return tuple(schema.names) == NATIVE_SOURCE_COLUMNS and all(
        schema.field(name).type == expected_types[name] for name in NATIVE_SOURCE_COLUMNS
    )


def _require_pair_manifest_native_sources(
    pair_payload: Mapping[str, Any],
    *,
    pair_generation_id: str,
) -> Mapping[str, Any]:
    """Admit only a legacy native pair or the sealed pre-TEST native pair.

    The pre-TEST lineage is intentionally a distinct schema: it binds direct,
    physically pre-TEST M1/M5 materialisations and must never be coerced into
    the old raw/base28 pair format.  Both routes return the exact native-source
    mapping which the caller then compares field-for-field with the supplied
    M5 root and its sealed manifest.
    """

    pair_schema = pair_payload.get("schema_version")
    lineage = pair_payload.get("lineage")
    native_sources = (
        lineage.get("native_sources") if isinstance(lineage, Mapping) else None
    )
    bound_m5 = (
        native_sources.get("m5") if isinstance(native_sources, Mapping) else None
    )
    if pair_schema == PAIR_MANIFEST_SCHEMA_VERSION:
        if (
            pair_payload.get("pair_generation_id") != pair_generation_id
            or not isinstance(lineage, Mapping)
            or lineage.get("schema_version") != PAIR_LINEAGE_SCHEMA_VERSION
            or not isinstance(bound_m5, Mapping)
        ):
            raise RuntimeError("M5_SOURCE_PAIR_MANIFEST_CONTRACT_MISMATCH")
        return bound_m5

    if pair_schema != PRETEST_NATIVE_PAIR_LINEAGE_SCHEMA_VERSION:
        raise RuntimeError("M5_SOURCE_PAIR_MANIFEST_CONTRACT_MISMATCH")
    declared_payload_sha256 = _exact_sha256(
        pair_payload.get("manifest_payload_sha256"),
        label="PRETEST_PAIR_MANIFEST_PAYLOAD",
    )
    pair_without_hash = dict(pair_payload)
    pair_without_hash.pop("manifest_payload_sha256", None)
    direct_m5 = pair_payload.get("m5")
    if (
        declared_payload_sha256 != _canonical_sha256(pair_without_hash)
        or set(pair_payload)
        != {
            "schema_version",
            "pair_generation_id",
            "pair_symbol",
            "test_boundary_utc",
            "test_accessed",
            "m1",
            "m5",
            "lineage",
            "manifest_payload_sha256",
        }
        or pair_payload.get("pair_generation_id") != pair_generation_id
        or pair_payload.get("pair_symbol") != "XAUUSD"
        or pair_payload.get("test_boundary_utc") != PRETEST_TEST_BOUNDARY_UTC
        or pair_payload.get("test_accessed") is not False
        or not isinstance(lineage, Mapping)
        or set(lineage) != {"native_sources"}
        or not isinstance(native_sources, Mapping)
        or set(native_sources) != {"m1", "m5"}
        or not isinstance(bound_m5, Mapping)
        or not isinstance(pair_payload.get("m1"), Mapping)
        or not isinstance(direct_m5, Mapping)
        or direct_m5.get("native_source") != dict(bound_m5)
    ):
        raise RuntimeError("M5_SOURCE_PRETEST_PAIR_MANIFEST_CONTRACT_MISMATCH")
    return bound_m5


def _preflight_native_source(
    *,
    native_root: Path,
    pair_manifest_path: Path,
    pair_generation_id: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[_NativePart],
    list[_FileSeal],
]:
    root = _require_directory(native_root, label="NATIVE_M5_ROOT")
    source_manifest_path = _require_regular_file(
        root / "MANIFEST.json",
        label="NATIVE_M5_MANIFEST",
    )
    source_manifest, source_manifest_seal = _read_json_sealed(
        source_manifest_path,
        label="NATIVE_M5_MANIFEST",
    )
    source_without_hash = dict(source_manifest)
    source_payload_sha256 = _exact_sha256(
        source_without_hash.pop("manifest_payload_sha256", None),
        label="NATIVE_M5_MANIFEST_PAYLOAD",
    )
    if source_payload_sha256 != _canonical_sha256(source_without_hash):
        raise RuntimeError("M5_SOURCE_NATIVE_M5_MANIFEST_PAYLOAD_MISMATCH")
    if (
        source_manifest.get("schema_version") not in NATIVE_SOURCE_SCHEMA_VERSIONS
        or source_manifest.get("instrument") != "XAU_USD"
        or source_manifest.get("timeframe") != "M5"
        or source_manifest.get("out_root") != str(root)
        or source_manifest.get("bar_duration_seconds") != ENTRY_DECISION_BAR_SECONDS
        or source_manifest.get("decision_available_offset_seconds")
        != ENTRY_DECISION_BAR_SECONDS
        or source_manifest.get("schema_required_cols")
        != list(NATIVE_SOURCE_COLUMNS)
        or source_manifest.get("schema_optional_cols") != []
    ):
        raise RuntimeError("M5_SOURCE_NATIVE_M5_MANIFEST_CONTRACT_MISMATCH")

    pair_path = _require_regular_file(pair_manifest_path, label="PAIR_MANIFEST")
    pair_payload, pair_seal = _read_json_sealed(pair_path, label="PAIR_MANIFEST")
    bound_m5 = _require_pair_manifest_native_sources(
        pair_payload,
        pair_generation_id=pair_generation_id,
    )

    row_count = _exact_positive_int(
        source_manifest.get("row_count"),
        label="NATIVE_M5_ROW_COUNT",
    )
    time_min_ns = _timestamp_ns(
        source_manifest.get("time_min_utc"),
        label="NATIVE_M5_MIN",
    )
    time_max_ns = _timestamp_ns(
        source_manifest.get("time_max_utc"),
        label="NATIVE_M5_MAX",
    )
    if time_max_ns <= time_min_ns:
        raise RuntimeError("M5_SOURCE_NATIVE_M5_TIME_RANGE_INVALID")
    year_rows = source_manifest.get("year_rows")
    year_sha256 = source_manifest.get("year_sha256")
    if (
        not isinstance(year_rows, dict)
        or not isinstance(year_sha256, dict)
        or not year_rows
        or set(year_rows) != set(year_sha256)
        or sum(
            _exact_positive_int(value, label="NATIVE_M5_YEAR_ROWS")
            for value in year_rows.values()
        )
        != row_count
    ):
        raise RuntimeError("M5_SOURCE_NATIVE_M5_YEAR_MANIFEST_INVALID")
    expected_pair_binding = {
        "root": str(root),
        "manifest_path": str(source_manifest_path),
        "manifest_sha256": source_manifest_seal.sha256,
        "row_count": row_count,
        "time_min_utc": pd.Timestamp(time_min_ns, tz="UTC").isoformat(),
        "time_max_utc": pd.Timestamp(time_max_ns, tz="UTC").isoformat(),
    }
    for name, expected in {
        **expected_pair_binding,
        "instrument": "XAU_USD",
        "timeframe": "M5",
        "canonical_rows_sha256": source_manifest.get("canonical_rows_sha256"),
        "manifest_payload_sha256": source_payload_sha256,
        "year_rows": year_rows,
        "year_sha256": year_sha256,
    }.items():
        if bound_m5.get(name) != expected:
            raise RuntimeError(
                f"M5_SOURCE_PAIR_NATIVE_M5_BINDING_MISMATCH: field={name}"
            )

    expected_year_keys = sorted(year_rows)
    if any(
        not isinstance(key, str)
        or len(key) != 9
        or not key.startswith("year=")
        or not key[5:].isdigit()
        for key in expected_year_keys
    ):
        raise RuntimeError("M5_SOURCE_NATIVE_M5_YEAR_KEYS_INVALID")
    actual_year_paths = sorted(
        path for path in root.iterdir() if path.name.startswith("year=")
    )
    if [path.name for path in actual_year_paths] != expected_year_keys:
        raise RuntimeError("M5_SOURCE_NATIVE_M5_YEAR_SET_MISMATCH")

    parts: list[_NativePart] = []
    part_sha_by_path: dict[str, str] = {}
    for year_key in expected_year_keys:
        year_dir = _require_directory(root / year_key, label="NATIVE_M5_YEAR")
        if set(os.listdir(year_dir)) != {"part-000.parquet"}:
            raise RuntimeError(f"M5_SOURCE_NATIVE_M5_YEAR_SURFACE_INVALID: {year_key}")
        part_path = _require_regular_file(
            year_dir / "part-000.parquet",
            label="NATIVE_M5_PART",
        )
        part_seal = _seal_file(part_path)
        expected_hash = _exact_sha256(
            year_sha256.get(year_key),
            label="NATIVE_M5_YEAR",
        )
        if part_seal.sha256 != expected_hash:
            raise RuntimeError(f"M5_SOURCE_NATIVE_M5_YEAR_HASH_MISMATCH: {year_key}")
        try:
            parquet = pq.ParquetFile(part_path)
        except Exception as exc:
            raise RuntimeError(f"M5_SOURCE_NATIVE_M5_PARQUET_INVALID: {part_path}") from exc
        expected_rows = _exact_positive_int(
            year_rows.get(year_key),
            label="NATIVE_M5_YEAR_ROWS",
        )
        if (
            not _native_schema_valid(parquet.schema_arrow)
            or parquet.metadata.num_rows != expected_rows
        ):
            raise RuntimeError(
                f"M5_SOURCE_NATIVE_M5_PARQUET_CONTRACT_MISMATCH: {year_key}"
            )
        parts.append(
            _NativePart(
                year_key=year_key,
                path=part_path,
                rows=expected_rows,
                seal=part_seal,
            )
        )
        part_sha_by_path[str(part_path)] = part_seal.sha256

    compact_bytes = row_count * (len(NATIVE_FLOAT_COLUMNS) + 2) * 8
    if compact_bytes <= 0 or compact_bytes > MAX_NATIVE_COMPACT_BYTES:
        raise RuntimeError(
            "M5_SOURCE_NATIVE_COMPACT_CAP_EXCEEDED: "
            f"required={compact_bytes} cap={MAX_NATIVE_COMPACT_BYTES}"
        )
    identity = {
        "root": str(root),
        "manifest_path": str(source_manifest_path),
        "manifest_sha256": source_manifest_seal.sha256,
        "part_paths": [str(part.path) for part in parts],
        "part_sha256": part_sha_by_path,
    }
    pair_binding = {
        "manifest_path": str(pair_path),
        "manifest_sha256": pair_seal.sha256,
        "pair_generation_id": pair_generation_id,
        "native_m5": expected_pair_binding,
    }
    seals = [source_manifest_seal, pair_seal, *(part.seal for part in parts)]
    return identity, pair_binding, parts, seals


def _preflight_enriched_and_cache(
    *,
    enriched: Path,
    cache_dir: Path,
    dataset_run_id: str,
    pair_generation_id: str,
    native_identity: Mapping[str, Any],
    pair_binding: Mapping[str, Any],
) -> tuple[pq.ParquetFile, _FileSeal, dict[str, Any], _FileSeal]:
    enriched_path = _require_regular_file(enriched, label="ENRICHED")
    enriched_manifest_path = _require_regular_file(
        Path(f"{enriched_path}.manifest.json"),
        label="ENRICHED_MANIFEST",
    )
    enriched_manifest, enriched_manifest_seal = _read_json_sealed(
        enriched_manifest_path,
        label="ENRICHED_MANIFEST",
    )
    declared_manifest_hash = _exact_sha256(
        enriched_manifest.get("manifest_sha256"),
        label="ENRICHED_MANIFEST",
    )
    manifest_without_hash = dict(enriched_manifest)
    manifest_without_hash.pop("manifest_sha256", None)
    if declared_manifest_hash != _canonical_sha256(manifest_without_hash):
        raise RuntimeError("M5_SOURCE_ENRICHED_MANIFEST_HASH_MISMATCH")
    require_entry_exit_shared_feature_base_contract(
        enriched_manifest.get("shared_feature_base_contract"),
        context="M5_SOURCE_ENRICHED",
    )
    if (
        enriched_manifest.get("schema_version")
        != ENTRY_EXIT_ENRICHED_CAUSAL_FRAME_SCHEMA_VERSION
        or enriched_manifest.get("decision") != "PASS"
        or enriched_manifest.get("dataset_run_id") != dataset_run_id
        or enriched_manifest.get("pair_generation_id") != pair_generation_id
        or enriched_manifest.get("timeframe") != "M5"
        or enriched_manifest.get("base_bar_seconds") != ENTRY_DECISION_BAR_SECONDS
        or enriched_manifest.get("output_parquet") != str(enriched_path)
        or enriched_manifest.get("columns") != list(ENRICHED_COLUMNS)
        or enriched_manifest.get("required_base_fields") != list(MODEL_NATIVE_BASE_FIELDS)
        or enriched_manifest.get("required_context_cont_fields")
        != list(MODEL_NATIVE_CTX_CONT_FIELDS)
        or enriched_manifest.get("required_context_cat_fields")
        != list(MODEL_NATIVE_CTX_CAT_FIELDS)
        or enriched_manifest.get("native_m5_source") != dict(native_identity)
        or enriched_manifest.get("pair_binding") != dict(pair_binding)
    ):
        raise RuntimeError("M5_SOURCE_ENRICHED_MANIFEST_CONTRACT_MISMATCH")

    enriched_seal = _seal_file(enriched_path)
    if enriched_manifest.get("output_parquet_sha256") != enriched_seal.sha256:
        raise RuntimeError("M5_SOURCE_ENRICHED_HASH_MISMATCH")
    try:
        parquet = pq.ParquetFile(enriched_path)
    except Exception as exc:
        raise RuntimeError("M5_SOURCE_ENRICHED_PARQUET_INVALID") from exc
    rows = _exact_positive_int(enriched_manifest.get("rows"), label="ENRICHED_ROWS")
    if (
        tuple(parquet.schema_arrow.names) != ENRICHED_COLUMNS
        or parquet.metadata.num_rows != rows
        or parquet.schema_arrow.field("time").type != pa.timestamp("ns", tz="UTC")
    ):
        raise RuntimeError("M5_SOURCE_ENRICHED_PARQUET_CONTRACT_MISMATCH")

    cache_path = _require_directory(cache_dir, label="MULTI_TF_CACHE")
    cache_manifest_path = _require_regular_file(
        cache_path / "manifest.json",
        label="MULTI_TF_CACHE_MANIFEST",
    )
    cache_manifest, cache_manifest_seal = _read_json_sealed(
        cache_manifest_path,
        label="MULTI_TF_CACHE_MANIFEST",
    )
    cache_identity = _exact_sha256(
        cache_manifest.get("cache_identity_sha256"),
        label="MULTI_TF_CACHE_IDENTITY",
    )
    if (
        cache_manifest.get("schema_version") != HTF_V4_CACHE_SCHEMA_VERSION
        or cache_identity != compute_htf_v4_cache_identity(cache_manifest)
        or cache_manifest.get("m5_prebuilt_source") != str(enriched_path)
        or cache_manifest.get("m5_prebuilt_source_sha256") != enriched_seal.sha256
    ):
        raise RuntimeError("M5_SOURCE_MULTI_TF_CACHE_MANIFEST_MISMATCH")
    expected_cache_binding = {
        "cache_dir": str(cache_path),
        "cache_manifest_path": str(cache_manifest_path),
        "cache_manifest_sha256": cache_manifest_seal.sha256,
        "cache_identity_sha256": cache_identity,
        "m5_context_source": str(enriched_path),
        "m5_context_source_sha256": enriched_seal.sha256,
    }
    if enriched_manifest.get("multi_tf_cache_binding") != expected_cache_binding:
        raise RuntimeError("M5_SOURCE_ENRICHED_MULTI_TF_CACHE_BINDING_MISMATCH")
    return parquet, enriched_seal, expected_cache_binding, cache_manifest_seal


def _iter_parquet_batches(
    parquet: pq.ParquetFile,
    *,
    columns: Sequence[str],
    batch_rows: int,
) -> Iterator[pa.RecordBatch]:
    for batch in parquet.iter_batches(
        batch_size=batch_rows,
        columns=list(columns),
        use_threads=False,
    ):
        if batch.num_rows <= 0 or batch.num_rows > batch_rows:
            raise RuntimeError("M5_SOURCE_PARQUET_BATCH_BOUND_VIOLATION")
        yield batch


def _time_array_ns(array: pa.Array, *, label: str) -> np.ndarray:
    if array.null_count:
        raise RuntimeError(f"M5_SOURCE_{label}_TIME_NULL")
    values = array.to_numpy(zero_copy_only=False)
    try:
        result = np.asarray(values).astype("datetime64[ns]", copy=False).view(np.int64)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"M5_SOURCE_{label}_TIME_INVALID") from exc
    if result.ndim != 1 or np.any(result % M5_NS != 0):
        raise RuntimeError(f"M5_SOURCE_{label}_TIME_GEOMETRY_INVALID")
    return result


def _numeric_array(array: pa.Array, *, label: str, dtype: np.dtype) -> np.ndarray:
    if array.null_count:
        raise RuntimeError(f"M5_SOURCE_{label}_NULL")
    try:
        result = np.asarray(array.to_numpy(zero_copy_only=False), dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"M5_SOURCE_{label}_NUMERIC_INVALID") from exc
    if result.ndim != 1:
        raise RuntimeError(f"M5_SOURCE_{label}_NUMERIC_INVALID")
    return result


def _load_native_arrays(
    parts: Sequence[_NativePart],
    *,
    batch_rows: int,
    expected_time_min_ns: int,
    expected_time_max_ns: int,
) -> tuple[_NativeArrays, int]:
    rows = sum(part.rows for part in parts)
    times_ns = np.empty(rows, dtype=np.int64)
    floats = np.empty((rows, len(NATIVE_FLOAT_COLUMNS)), dtype=np.float64)
    volume = np.empty(rows, dtype=np.int64)
    offset = 0
    max_observed_batch = 0
    previous_time: int | None = None
    for part in parts:
        parquet = pq.ParquetFile(part.path)
        year = int(part.year_key[5:])
        part_start = offset
        for batch in _iter_parquet_batches(
            parquet,
            columns=NATIVE_SOURCE_COLUMNS,
            batch_rows=batch_rows,
        ):
            size = batch.num_rows
            max_observed_batch = max(max_observed_batch, size)
            end = offset + size
            if end > rows:
                raise RuntimeError("M5_SOURCE_NATIVE_M5_ROW_COUNT_OVERFLOW")
            by_name = {name: batch.column(index) for index, name in enumerate(batch.schema.names)}
            batch_times = _time_array_ns(by_name["time"], label="NATIVE_M5")
            if (
                np.any(np.diff(batch_times) <= 0)
                or (previous_time is not None and int(batch_times[0]) <= previous_time)
            ):
                raise RuntimeError("M5_SOURCE_NATIVE_M5_TIME_ORDER_INVALID")
            if np.any(pd.DatetimeIndex(batch_times).year.to_numpy() != year):
                raise RuntimeError("M5_SOURCE_NATIVE_M5_YEAR_PARTITION_MISMATCH")
            times_ns[offset:end] = batch_times
            for name, column_index in NATIVE_FLOAT_INDEX.items():
                floats[offset:end, column_index] = _numeric_array(
                    by_name[name],
                    label=f"NATIVE_M5_{name.upper()}",
                    dtype=np.dtype(np.float64),
                )
            volume[offset:end] = _numeric_array(
                by_name["volume"],
                label="NATIVE_M5_VOLUME",
                dtype=np.dtype(np.int64),
            )
            if (
                not np.isfinite(floats[offset:end]).all()
                or np.any(volume[offset:end] <= 0)
            ):
                raise RuntimeError("M5_SOURCE_NATIVE_M5_NONFINITE_OR_VOLUME_INVALID")
            open_ = floats[offset:end, NATIVE_FLOAT_INDEX["open"]]
            high = floats[offset:end, NATIVE_FLOAT_INDEX["high"]]
            low = floats[offset:end, NATIVE_FLOAT_INDEX["low"]]
            close = floats[offset:end, NATIVE_FLOAT_INDEX["close"]]
            if (
                np.any(open_ <= 0.0)
                or np.any(low <= 0.0)
                or np.any(high < low)
                or np.any(high < open_)
                or np.any(high < close)
                or np.any(low > open_)
                or np.any(low > close)
            ):
                raise RuntimeError("M5_SOURCE_NATIVE_M5_OHLC_GEOMETRY_INVALID")
            for component in ("open", "high", "low", "close"):
                bid = floats[offset:end, NATIVE_FLOAT_INDEX[f"bid_{component}"]]
                ask = floats[offset:end, NATIVE_FLOAT_INDEX[f"ask_{component}"]]
                if np.any(bid <= 0.0) or np.any(ask < bid):
                    raise RuntimeError("M5_SOURCE_NATIVE_M5_BID_ASK_GEOMETRY_INVALID")
            previous_time = int(batch_times[-1])
            offset = end
        if offset - part_start != part.rows:
            raise RuntimeError("M5_SOURCE_NATIVE_M5_YEAR_ROW_COUNT_MISMATCH")
    if (
        offset != rows
        or rows == 0
        or int(times_ns[0]) != expected_time_min_ns
        or int(times_ns[-1]) != expected_time_max_ns
    ):
        raise RuntimeError("M5_SOURCE_NATIVE_M5_FULL_CLOCK_MISMATCH")
    arrays = _NativeArrays(times_ns=times_ns, floats=floats, volume=volume)
    if arrays.nbytes > MAX_NATIVE_COMPACT_BYTES:
        raise RuntimeError("M5_SOURCE_NATIVE_COMPACT_CAP_EXCEEDED")
    return arrays, max_observed_batch


def _load_enriched_times(
    parquet: pq.ParquetFile,
    *,
    batch_rows: int,
) -> tuple[np.ndarray, int]:
    rows = int(parquet.metadata.num_rows)
    times_ns = np.empty(rows, dtype=np.int64)
    offset = 0
    max_observed_batch = 0
    previous_time: int | None = None
    for batch in _iter_parquet_batches(
        parquet,
        columns=("time",),
        batch_rows=batch_rows,
    ):
        size = batch.num_rows
        end = offset + size
        batch_times = _time_array_ns(batch.column(0), label="ENRICHED")
        if (
            np.any(np.diff(batch_times) <= 0)
            or (previous_time is not None and int(batch_times[0]) <= previous_time)
            or end > rows
        ):
            raise RuntimeError("M5_SOURCE_ENRICHED_TIME_ORDER_INVALID")
        times_ns[offset:end] = batch_times
        previous_time = int(batch_times[-1])
        max_observed_batch = max(max_observed_batch, size)
        offset = end
    if offset != rows or rows == 0:
        raise RuntimeError("M5_SOURCE_ENRICHED_ROW_COUNT_MISMATCH")
    return times_ns, max_observed_batch


def _exact_contiguous_positions(
    native_times_ns: np.ndarray,
    enriched_times_ns: np.ndarray,
) -> np.ndarray:
    positions = np.searchsorted(native_times_ns, enriched_times_ns, side="left")
    if (
        len(positions) != len(enriched_times_ns)
        or np.any(positions >= len(native_times_ns))
        or not np.array_equal(native_times_ns[positions], enriched_times_ns)
        or not np.array_equal(
            positions,
            np.arange(int(positions[0]), int(positions[0]) + len(positions)),
        )
    ):
        raise RuntimeError("M5_SOURCE_NATIVE_EXACT_CONTIGUOUS_JOIN_INCOMPLETE")
    return positions.astype(np.int64, copy=False)


def _compact_regime_projection(
    *,
    multi_tf: Mapping[str, pd.DataFrame],
    enriched_times_ns: np.ndarray,
) -> np.ndarray:
    projected = project_multi_tf_v4_scalars(
        multi_tf,
        enriched_times_ns,
        REGIME_COMPACT_PROJECTION,
        MULTI_TF_TIMEFRAMES_LOWER_M5_LAST,
        frozenset(),
        decision_bar_duration=pd.Timedelta(minutes=5),
    )
    if set(projected) != set(REGIME_PROJECTED_FIELDS):
        raise RuntimeError("M5_SOURCE_REGIME_PROJECTION_FIELDS_MISMATCH")
    compact = np.empty(
        (len(enriched_times_ns), len(REGIME_PROJECTED_FIELDS)),
        dtype=np.float64,
    )
    # A decision row that precedes the first closed bar of a declared context
    # timeframe has no value for that context: it is undefined by causality,
    # not missing. Those rows are excluded from the model source rather than
    # filled. Anything non-finite AFTER that leading warmup is real corruption
    # and still fails closed.
    warmup_rows = 0
    for index, name in enumerate(REGIME_PROJECTED_FIELDS):
        values = np.asarray(projected[name], dtype=np.float64)
        if values.shape != (len(enriched_times_ns),):
            raise RuntimeError(f"M5_SOURCE_REGIME_SHAPE_INVALID: {name}")
        finite = np.isfinite(values)
        if not finite.all():
            first_finite = int(np.argmax(finite)) if finite.any() else len(values)
            if not finite[first_finite:].all():
                raise RuntimeError(
                    f"M5_SOURCE_REGIME_NONFINITE_AFTER_WARMUP: {name}"
                )
            warmup_rows = max(warmup_rows, first_finite)
        compact[:, index] = values
    if warmup_rows >= len(enriched_times_ns):
        raise RuntimeError("M5_SOURCE_REGIME_WARMUP_CONSUMES_WHOLE_SOURCE")
    return compact, warmup_rows


def _new_partial(final_path: Path) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{final_path.name}.partial-",
        dir=final_path.parent,
    )
    os.close(descriptor)
    return Path(raw_path)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError(f"M5_SOURCE_FSYNC_NOT_REGULAR: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_identity(path: Path) -> tuple[int, int]:
    observed = os.lstat(path)
    if not stat.S_ISREG(observed.st_mode):
        raise RuntimeError(f"M5_SOURCE_ARTIFACT_NOT_REGULAR: {path}")
    return int(observed.st_dev), int(observed.st_ino)


def _unlink_if_owned(path: Path, identity: tuple[int, int]) -> None:
    try:
        if _file_identity(path) != identity:
            return
        os.unlink(path)
    except FileNotFoundError:
        return


def _publish_no_replace(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination, follow_symlinks=False)
    except FileExistsError as exc:
        raise RuntimeError(f"M5_SOURCE_PUBLISH_TARGET_RACE: {destination}") from exc
    except OSError as exc:
        raise RuntimeError(f"M5_SOURCE_PUBLISH_FAILED: {destination}") from exc


def _publish_artifact_pair(
    *,
    parquet_partial: Path,
    parquet_final: Path,
    parquet_sha256: str,
    manifest_partial: Path,
    manifest_final: Path,
    manifest_file_sha256: str,
) -> None:
    parquet_identity = _file_identity(parquet_partial)
    manifest_identity = _file_identity(manifest_partial)
    parquet_published = False
    manifest_published = False
    try:
        if (
            parquet_final.exists()
            or parquet_final.is_symlink()
            or manifest_final.exists()
            or manifest_final.is_symlink()
        ):
            raise RuntimeError("M5_SOURCE_PUBLISH_TARGET_EXISTS")
        _publish_no_replace(parquet_partial, parquet_final)
        parquet_published = True
        _fsync_directory(parquet_final.parent)
        if _sha256_file(parquet_final) != parquet_sha256:
            raise RuntimeError("M5_SOURCE_PUBLISHED_PARQUET_HASH_MISMATCH")
        _publish_no_replace(manifest_partial, manifest_final)
        manifest_published = True
        _fsync_directory(manifest_final.parent)
        if _sha256_file(manifest_final) != manifest_file_sha256:
            raise RuntimeError("M5_SOURCE_PUBLISHED_MANIFEST_HASH_MISMATCH")
    except BaseException:
        if manifest_published:
            _unlink_if_owned(manifest_final, manifest_identity)
        if parquet_published:
            _unlink_if_owned(parquet_final, parquet_identity)
        _fsync_directory(parquet_final.parent)
        raise


def _write_output_partial(
    *,
    enriched_parquet: pq.ParquetFile,
    enriched_times_ns: np.ndarray,
    native: _NativeArrays,
    native_positions: np.ndarray,
    regime_compact: np.ndarray,
    partial: Path,
    batch_rows: int,
) -> tuple[pa.Schema, int]:
    enriched_read_columns = tuple(
        name for name in ENRICHED_COLUMNS if name not in RANKER_OWNED_DERIVATIONS
    )
    projected_index = {
        name: index for index, name in enumerate(REGIME_PROJECTED_FIELDS)
    }
    writer: pq.ParquetWriter | None = None
    output_schema: pa.Schema | None = None
    offset = 0
    max_observed_batch = 0
    try:
        for batch in _iter_parquet_batches(
            enriched_parquet,
            columns=enriched_read_columns,
            batch_rows=batch_rows,
        ):
            size = batch.num_rows
            end = offset + size
            if end > len(enriched_times_ns):
                raise RuntimeError("M5_SOURCE_OUTPUT_ROW_COUNT_OVERFLOW")
            by_name = {
                name: batch.column(index) for index, name in enumerate(batch.schema.names)
            }
            batch_times = _time_array_ns(by_name["time"], label="OUTPUT")
            if not np.array_equal(batch_times, enriched_times_ns[offset:end]):
                raise RuntimeError("M5_SOURCE_ENRICHED_CHANGED_BETWEEN_PASSES")
            positions = native_positions[offset:end]
            for name in ("open", "high", "low", "close"):
                enriched_values = _numeric_array(
                    by_name[name],
                    label=f"ENRICHED_{name.upper()}",
                    dtype=np.dtype(np.float64),
                )
                native_values = native.floats[
                    positions,
                    NATIVE_FLOAT_INDEX[name],
                ]
                if not np.array_equal(enriched_values, native_values):
                    raise RuntimeError(
                        f"M5_SOURCE_ENRICHED_NATIVE_{name.upper()}_MISMATCH"
                    )
            bid_close = native.floats[
                positions,
                NATIVE_FLOAT_INDEX["bid_close"],
            ]
            ask_close = native.floats[
                positions,
                NATIVE_FLOAT_INDEX["ask_close"],
            ]
            if np.any(ask_close < bid_close):
                raise RuntimeError("M5_SOURCE_BID_ASK_GEOMETRY_INVALID")
            for name in REGIME_PROJECTED_FIELDS:
                if name not in by_name:
                    continue
                existing = _numeric_array(
                    by_name[name],
                    label=f"REGIME_{name.upper()}",
                    dtype=np.dtype(np.float64),
                )
                incoming = regime_compact[offset:end, projected_index[name]]
                # The verified V4 cache owns these scalars and publishes only
                # closed raw states, so it is the authority. Where it defines a
                # value the enriched frame must match it exactly. Where it does
                # not - a row before the first closed bar of that timeframe -
                # the value is undefined by causality, and any enriched value
                # there could only have come from a forming bar, so the cache's
                # undefined value is written instead.
                defined = np.isfinite(incoming)
                if not np.array_equal(existing[defined], incoming[defined]):
                    raise RuntimeError(
                        f"M5_SOURCE_REGIME_{name.upper()}_MISMATCH"
                    )

            arrays: list[pa.Array] = []
            for name in OUTPUT_COLUMNS:
                if name == "time":
                    array = by_name[name]
                elif name == "volume":
                    array = pa.array(native.volume[positions], type=pa.int64())
                elif name in NATIVE_FLOAT_INDEX and name not in by_name:
                    array = pa.array(
                        native.floats[positions, NATIVE_FLOAT_INDEX[name]],
                        type=pa.float64(),
                    )
                elif name in projected_index:
                    array = pa.array(
                        regime_compact[offset:end, projected_index[name]],
                        type=pa.float64(),
                    )
                elif name in by_name:
                    array = by_name[name]
                else:
                    raise RuntimeError(f"M5_SOURCE_OUTPUT_FIELD_UNRESOLVED: {name}")
                arrays.append(array)
            output_batch = pa.RecordBatch.from_arrays(arrays, names=list(OUTPUT_COLUMNS))
            if output_schema is None:
                output_schema = output_batch.schema
                writer = pq.ParquetWriter(
                    partial,
                    output_schema,
                    compression="snappy",
                    use_dictionary=True,
                    write_statistics=True,
                )
            elif not output_batch.schema.equals(output_schema, check_metadata=False):
                raise RuntimeError("M5_SOURCE_OUTPUT_BATCH_SCHEMA_DRIFT")
            assert writer is not None
            writer.write_batch(output_batch, row_group_size=size)
            max_observed_batch = max(max_observed_batch, size)
            offset = end
        if writer is None or output_schema is None or offset != len(enriched_times_ns):
            raise RuntimeError("M5_SOURCE_OUTPUT_INCOMPLETE")
        writer.close()
        writer = None
    finally:
        if writer is not None:
            writer.close()
    return output_schema, max_observed_batch


def _verify_output_partial(
    *,
    partial: Path,
    expected_schema: pa.Schema,
    expected_rows: int,
    batch_rows: int,
) -> tuple[str, int, str]:
    _fsync_file(partial)
    _fsync_directory(partial.parent)
    try:
        parquet = pq.ParquetFile(partial)
    except Exception as exc:
        raise RuntimeError("M5_SOURCE_OUTPUT_PARQUET_VERIFY_FAILED") from exc
    if (
        parquet.metadata.num_rows != expected_rows
        or not parquet.schema_arrow.equals(expected_schema, check_metadata=False)
        or tuple(parquet.schema_arrow.names) != OUTPUT_COLUMNS
        or any(
            parquet.metadata.row_group(index).num_rows > batch_rows
            for index in range(parquet.metadata.num_row_groups)
        )
    ):
        raise RuntimeError("M5_SOURCE_OUTPUT_PARQUET_PROOF_FAILED")
    seal = _seal_file(partial)
    schema_contract = [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": bool(field.nullable),
        }
        for field in parquet.schema_arrow
    ]
    return seal.sha256, seal.size_bytes, _canonical_sha256(schema_contract)


def _write_manifest_partial(path: Path, manifest: Mapping[str, Any]) -> str:
    encoded = (
        json.dumps(dict(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)
    try:
        observed = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError("M5_SOURCE_OUTPUT_MANIFEST_VERIFY_FAILED") from exc
    if observed != dict(manifest):
        raise RuntimeError("M5_SOURCE_OUTPUT_MANIFEST_VERIFY_FAILED")
    declared_hash = observed.get("manifest_sha256")
    without_hash = dict(observed)
    without_hash.pop("manifest_sha256", None)
    if declared_hash != _canonical_sha256(without_hash):
        raise RuntimeError("M5_SOURCE_OUTPUT_MANIFEST_HASH_MISMATCH")
    return _sha256_file(path)


def materialize_m5_source(
    *,
    enriched_parquet: Path,
    multi_tf_cache_dir: Path,
    native_m5_root: Path,
    pair_manifest: Path,
    output_parquet: Path,
    dataset_run_id: str,
    pair_generation_id: str,
    batch_rows: int = MAX_BATCH_ROWS,
) -> dict[str, Any]:
    require_offline_scope("featurebase_build")
    architecture = require_entry_exit_production_architecture(
        current_entry_exit_architecture_observation(),
        context="M5_SOURCE",
    )
    dataset_id = _require_lineage_id(dataset_run_id, label="DATASET_RUN_ID")
    pair_id = _require_lineage_id(
        pair_generation_id,
        label="PAIR_GENERATION_ID",
        sha256=True,
    )
    if (
        isinstance(batch_rows, bool)
        or not isinstance(batch_rows, int)
        or batch_rows <= 0
        or batch_rows > MAX_BATCH_ROWS
    ):
        raise RuntimeError(
            f"M5_SOURCE_BATCH_ROWS_INVALID: observed={batch_rows} max={MAX_BATCH_ROWS}"
        )
    output, output_manifest = _require_new_outputs(output_parquet)

    native_identity, pair_binding, native_parts, native_seals = (
        _preflight_native_source(
            native_root=native_m5_root,
            pair_manifest_path=pair_manifest,
            pair_generation_id=pair_id,
        )
    )
    enriched_file, enriched_seal, cache_binding, cache_manifest_seal = (
        _preflight_enriched_and_cache(
            enriched=enriched_parquet,
            cache_dir=multi_tf_cache_dir,
            dataset_run_id=dataset_id,
            pair_generation_id=pair_id,
            native_identity=native_identity,
            pair_binding=pair_binding,
        )
    )

    multi_tf = load_multi_tf_v4_cache(Path(cache_binding["cache_dir"]))
    if (
        Path(multi_tf.m5_prebuilt_source) != enriched_seal.path
        or multi_tf.m5_prebuilt_source_sha256 != enriched_seal.sha256
        or multi_tf.manifest_sha256 != cache_binding["cache_manifest_sha256"]
        or multi_tf.cache_identity_sha256 != cache_binding["cache_identity_sha256"]
    ):
        raise RuntimeError("M5_SOURCE_MULTI_TF_CACHE_BINDING_MISMATCH")

    native_manifest_payload, _native_manifest_seal = _read_json_sealed(
        Path(native_identity["manifest_path"]),
        label="NATIVE_M5_MANIFEST_RECHECK",
    )
    native, native_max_batch = _load_native_arrays(
        native_parts,
        batch_rows=batch_rows,
        expected_time_min_ns=_timestamp_ns(
            native_manifest_payload["time_min_utc"],
            label="NATIVE_M5_MIN",
        ),
        expected_time_max_ns=_timestamp_ns(
            native_manifest_payload["time_max_utc"],
            label="NATIVE_M5_MAX",
        ),
    )
    enriched_times_ns, enriched_time_max_batch = _load_enriched_times(
        enriched_file,
        batch_rows=batch_rows,
    )
    native_positions = _exact_contiguous_positions(
        native.times_ns,
        enriched_times_ns,
    )
    regime_compact, regime_warmup_rows = _compact_regime_projection(
        multi_tf=multi_tf,
        enriched_times_ns=enriched_times_ns,
    )
    del multi_tf

    parquet_partial = _new_partial(output)
    manifest_partial: Path | None = None
    parquet_identity = _file_identity(parquet_partial)
    try:
        output_schema, output_max_batch = _write_output_partial(
            enriched_parquet=enriched_file,
            enriched_times_ns=enriched_times_ns,
            native=native,
            native_positions=native_positions,
            regime_compact=regime_compact,
            partial=parquet_partial,
            batch_rows=batch_rows,
        )
        output_sha256, output_size_bytes, output_schema_sha256 = (
            _verify_output_partial(
                partial=parquet_partial,
                expected_schema=output_schema,
                expected_rows=len(enriched_times_ns),
                batch_rows=batch_rows,
            )
        )
        for seal in (*native_seals, enriched_seal, cache_manifest_seal):
            _require_seal_unchanged(seal)

        manifest: dict[str, Any] = {
            "schema_version": M5_SOURCE_SCHEMA_VERSION,
            "decision": "PASS",
            "dataset_run_id": dataset_id,
            "pair_generation_id": pair_id,
            "timeframe": "M5",
            "anchor_timeframe": "M5",
            "production_architecture": architecture,
            "enriched_source": str(enriched_seal.path),
            "enriched_source_sha256": enriched_seal.sha256,
            "multi_tf_cache_dir": cache_binding["cache_dir"],
            "multi_tf_cache_manifest_sha256": cache_binding[
                "cache_manifest_sha256"
            ],
            "multi_tf_cache_identity_sha256": cache_binding[
                "cache_identity_sha256"
            ],
            "native_m5_source": native_identity,
            "pair_manifest": pair_binding["manifest_path"],
            "pair_manifest_sha256": pair_binding["manifest_sha256"],
            "output_parquet": str(output),
            "output_parquet_sha256": output_sha256,
            "output_parquet_size_bytes": output_size_bytes,
            "output_arrow_schema_sha256": output_schema_sha256,
            "rows": int(len(enriched_times_ns)),
            "context_warmup_rows": int(regime_warmup_rows),
            "first_fully_defined_context_row_utc": str(
                pd.Timestamp(
                    int(enriched_times_ns[regime_warmup_rows]), tz="UTC"
                )
            ),
            "columns": list(OUTPUT_COLUMNS),
            "raw_market_columns": list(RAW_MARKET_COLUMNS),
            "market_identity_columns": list(MARKET_IDENTITY_COLUMNS),
            "source_owned_fields": list(SOURCE_OWNED_FIELDS),
            "regime_source_rehydrated_from_native_m5": True,
            "ranker_owned_derivations_removed": list(RANKER_OWNED_DERIVATIONS),
            "shared_feature_base_contract": entry_exit_shared_feature_base_contract(),
            "model_surface_dimensions": {
                "signal_dim": MODEL_NATIVE_SIGNAL_DIM,
                "context_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
                "context_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
            },
            "bounded_io": {
                "configured_batch_rows": batch_rows,
                "maximum_batch_rows": MAX_BATCH_ROWS,
                "native_max_observed_batch_rows": native_max_batch,
                "enriched_time_max_observed_batch_rows": enriched_time_max_batch,
                "output_max_observed_batch_rows": output_max_batch,
                "native_compact_array_bytes": native.nbytes,
                "native_compact_array_cap_bytes": MAX_NATIVE_COMPACT_BYTES,
                "regime_compact_array_bytes": int(regime_compact.nbytes),
                "use_threads": False,
            },
            "causal_contract": {
                "exact_timestamp_join": True,
                "exact_contiguous_native_view": True,
                "native_view_start_row": int(native_positions[0]),
                "native_view_end_row_inclusive": int(native_positions[-1]),
                "future_rows_used": False,
                "native_bid_ask_reused": True,
                "resample_or_fill": False,
                "same_feature_owner_as_entry_exit": True,
                "regime_projection_uses_full_cache_history": True,
            },
        }
        manifest["manifest_sha256"] = _canonical_sha256(manifest)
        manifest_partial = _new_partial(output_manifest)
        manifest_identity = _file_identity(manifest_partial)
        manifest_file_sha256 = _write_manifest_partial(manifest_partial, manifest)
        _publish_artifact_pair(
            parquet_partial=parquet_partial,
            parquet_final=output,
            parquet_sha256=output_sha256,
            manifest_partial=manifest_partial,
            manifest_final=output_manifest,
            manifest_file_sha256=manifest_file_sha256,
        )
        _unlink_if_owned(parquet_partial, parquet_identity)
        _unlink_if_owned(manifest_partial, manifest_identity)
        _fsync_directory(output.parent)
        return manifest
    finally:
        _unlink_if_owned(parquet_partial, parquet_identity)
        if manifest_partial is not None and manifest_partial.exists():
            try:
                _unlink_if_owned(manifest_partial, _file_identity(manifest_partial))
            except OSError:
                pass
        _fsync_directory(output.parent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enriched-parquet", type=Path, required=True)
    parser.add_argument("--multi-tf-cache-dir", type=Path, required=True)
    parser.add_argument("--native-m5-root", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--dataset-run-id", required=True)
    parser.add_argument("--pair-generation-id", required=True)
    parser.add_argument("--batch-rows", type=int, default=MAX_BATCH_ROWS)
    args = parser.parse_args()
    print(json.dumps(materialize_m5_source(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
