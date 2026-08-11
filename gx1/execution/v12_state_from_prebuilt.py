#!/usr/bin/env python3
"""V12 state loader for one immutable canonical-v3/raw-M1 pair generation.

The loader admits one atomic, hash-bound pair:

* canonical-v3 is the persisted model-agnostic M5 feature surface;
* BASE28 is only the 13 physical non-time native M1 fields.

It never treats an old augmented BASE file or mutable parquet path as
authority. TRAIN-fit state such as ATR/spread ranks is deliberately outside
the pair and must be supplied and bound by the consuming dataset/model
contract. A complete v2 canonical already contains every model-agnostic
derived field, so compatibility augmenters become verified no-ops instead of
recomputing full history during load/refresh.

Architecture:
    The updater computes both candidate artifacts without touching active state,
    then publishes one immutable generation by atomically replacing:
        CANONICAL_V3_BASE28_CURRENT_PAIR_MANIFEST.json

    Live (every M1 tick):
        - map M1 decision availability to the exact latest closed M5 bar;
        - read persisted canonical context and exact native M1 market data;
        - let the admitted model contract derive/bind consumer-owned state.

The two artifacts in every admitted pair generation:

  canonical_v3 prebuilt (M5 cadence):
    Path: CANONICAL_V3_BASE28_GENERATIONS/<pair_generation_id>/
            canonical_v3.parquet
    Contains: the complete model-agnostic canonical feature surface.
    Built by: materialize_build_canonical_features_v2.py
              + materialize_canonical_v3_augment.py

  BASE28 artifact (M1 cadence):
    Path: CANONICAL_V3_BASE28_GENERATIONS/<pair_generation_id>/
            base28.parquet
    Contains: exactly the 13 native M1 market fields, in source order.
    Built by: v12_canonical_incremental.py from an immutable native M1 snapshot.

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

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:  # runtime imports stay lazy to keep loader startup light
    from gx1.contracts.entry_model_native_state_v2 import TrainRankReferenceV2

# ONE-TRUTH REGIME_V4 per-TF V2 scalar projection (5-TF/9-feat). Shared with
# immutable snapshot construction and offline build so admitted serving and
# training project the same per-TF regime inputs (no train≠serve).
from gx1.features.htf_features import (
    REGIME_V4_MTF_PROJECTION,
    REGIME_V4_MTF_TIMEFRAMES,
    REGIME_V4_MTF_SKIP,
)
from gx1.features.basic_v1 import PLUS5_FEATURES, compute_plus5_features
from gx1.contracts.xau_tape_provenance_v1 import (
    CANONICAL_NATIVE_REQUIRED_COLUMNS,
)
from gx1.models.entry_v10.direction_decision_contract import (
    canonical_closed_m1_bar,
)

LOG = logging.getLogger("v12_state_from_prebuilt")


def _require_v29_registry_constants_from_bound_cache() -> dict:
    """Frozen V29 registry constants from the run's bound V4 cache manifest.

    In-memory V4 builds must use the exact constants of the immutable cache
    the run binds (GX1_V10_MULTI_TF_V4_CACHE_DIR — the same env the derived
    -context path already requires); a missing binding fails closed (no
    default exists, rule 2a/18).
    """
    import os as _os
    from pathlib import Path as _Path
    from gx1.features.htf_features import load_v29_registry_constants_manifest

    raw = _os.environ.get("GX1_V10_MULTI_TF_V4_CACHE_DIR", "").strip()
    if not raw:
        raise RuntimeError(
            "GX1_V10_MULTI_TF_V4_CACHE_DIR_REQUIRED: in-memory V4 builds "
            "need the bound cache's frozen v29_registry_constants"
        )
    return load_v29_registry_constants_manifest(
        _Path(raw).expanduser() / "manifest.json"
    )


# ── Top-level subprocess workers for parallel augment (2026-06-01) ──
# These run in subprocesses spawned by _async_full_refresh so V2 mtf and
# GROUP-A — the two heavy augmenters — execute in true parallel rather than
# serializing behind the GIL. Each worker takes a pickled cv3, runs ONE
# augmenter via an ephemeral PrebuiltStateLoader instance, returns only the
# new columns (smaller payload back). On Linux fork() is used so cv3 is
# inherited via copy-on-write — actual pickle cost is only the result.
def _mp_v4_mtf_worker(cv3: pd.DataFrame) -> pd.DataFrame:
    """Run the V4 projection in a subprocess and return persistent fields.

    NB: the augmenter mutates cv3 in-place and returns the same object, so we
    must snapshot the column set BEFORE the call to compute the new-cols delta.
    """
    loader = PrebuiltStateLoader()
    augmented = loader._augment_cv3_with_v4_mtf_scalars(cv3)
    expected = loader._expected_mtf_scalar_columns()
    missing = [name for name in expected if name not in augmented.columns]
    if missing:
        raise RuntimeError(f"parallel V4 MTF worker output missing: {missing}")
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

PREBUILT_PAIR_SCHEMA_VERSION = "gx1_canonical_v3_raw_base28_pair_generation_v3"
PREBUILT_PAIR_LEGACY_SCHEMA_VERSION = (
    "gx1_canonical_v3_raw_base28_pair_generation_v2"
)
PREBUILT_PAIR_LINEAGE_SCHEMA_VERSION = "gx1_native_pair_lineage_v2"
PREBUILT_PAIR_LEGACY_LINEAGE_SCHEMA_VERSION = "gx1_native_pair_lineage_v1"
PREBUILT_PAIR_GENERATION_MANIFEST_FILENAME = "PAIR_MANIFEST.json"
PAIR_PUBLISH_LOCK_FILENAME = ".canonical_v3_base28_pair_publish.lock"
PREBUILT_PAIR_PRODUCER_OWNER = "gx1.execution.v12_canonical_incremental"
PREBUILT_CANONICAL_BUILDER_CONTRACT = (
    "canonical_v2_plus_v3_plus5_model_agnostic_ctx_group_a_v2"
)
PREBUILT_PAIR_FORMULA_CONTRACT = {
    "rank_inputs": "shared_model_native_simple_tr14_hl_mid_bid_ask_v1",
    "train_rank_application": "consumer_boundary_only",
    "group_a": "shared_parallel_exact_checkpointed_v1",
    "canonical_m5_atr_regime": (
        "atr14_rolling5760_min2880_q333_q667_shift1_"
        "integer_write_through_v2"
    ),
    "native_m1_m5_aggregation": (
        "source_exact_dense_or_sparse_no_session_hour_rules_v1"
    ),
}
PREBUILT_PAIR_TIMING_CONTRACT = {
    "m1_label": "bar_start_utc",
    "m1_availability": "bar_start_plus_1min",
    "m5_label": "bar_start_utc",
    "m5_availability": "bar_start_plus_5min",
    "htf_join_left_key": "m5_bar_start_plus_5min",
    "session_and_cycles": "m5_bar_start_plus_5min",
}
RAW_BASE28_COLUMNS = tuple(CANONICAL_NATIVE_REQUIRED_COLUMNS[1:])
PAIR_FORBIDDEN_CANONICAL_FIELDS = frozenset(
    {
        "atr_bucket",
        "spread_bucket",
        "is_model_bar",
        "m5_phase_0",
        "m5_phase_1",
        "m5_phase_2",
        "m5_phase_3",
        "m5_phase_4",
    }
)
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
    generation_manifest_path: Path | None
    pair_generation_id: str
    lineage: dict[str, object]
    canonical_v3: _PrebuiltManifestBinding
    base28: _PrebuiltManifestBinding


@dataclass(frozen=True)
class _VerifiedPrebuilt:
    binding: _PrebuiltManifestBinding
    file_stamp: tuple[int, int, int, int]
    arrow_schema: tuple[tuple[str, str, bool], ...]


@dataclass(frozen=True)
class PrebuiltServingSnapshot:
    """One immutable-reference view of a complete admitted serving generation."""

    pair_generation_id: str
    pair_manifest_sha256: str
    pair_generation_manifest_path: Path
    base28_path: Path
    base28_sha256: str
    base28_file_stamp: tuple[int, int, int, int]
    cv3: pd.DataFrame
    base28: pd.DataFrame
    cutoff_ts: pd.Timestamp
    multi_tf_feats: dict | None
    multi_tf_shift: dict | None
    multi_tf_feat_count: int


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


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value.lower() == value
        and not any(char not in "0123456789abcdef" for char in value)
    )


def validate_prebuilt_pair_lineage(
    raw: object,
) -> dict[str, object]:
    """Validate the source/formula envelope included in pair identity."""

    if not isinstance(raw, dict):
        raise PrebuiltIdentityError("PREBUILT_PAIR_LINEAGE_INVALID")
    required = {
        "schema_version",
        "explicit_vedtak_id",
        "producer_owner",
        "producer_git_commit",
        "producer_repository_clean",
        "producer_source_files",
        "producer_source_inventory_sha256",
        "native_sources",
        "derivation_contract",
        "coverage",
        "parent_pair_generation_id",
    }
    lineage_schema_version = raw.get("schema_version")
    if lineage_schema_version == PREBUILT_PAIR_LINEAGE_SCHEMA_VERSION:
        required.add("parent_pair_manifest_sha256")
    elif lineage_schema_version != PREBUILT_PAIR_LEGACY_LINEAGE_SCHEMA_VERSION:
        raise PrebuiltIdentityError("PREBUILT_PAIR_LINEAGE_SCHEMA_INVALID")
    if set(raw) != required:
        raise PrebuiltIdentityError("PREBUILT_PAIR_LINEAGE_FIELDS_INVALID")
    for name in ("explicit_vedtak_id", "producer_owner"):
        value = raw[name]
        if not isinstance(value, str) or not value or value.strip() != value:
            raise PrebuiltIdentityError(
                f"PREBUILT_PAIR_LINEAGE_{name.upper()}_INVALID"
            )
    if (
        lineage_schema_version == PREBUILT_PAIR_LINEAGE_SCHEMA_VERSION
        and raw["producer_owner"] != PREBUILT_PAIR_PRODUCER_OWNER
    ):
        raise PrebuiltIdentityError(
            "PREBUILT_PAIR_LINEAGE_PRODUCER_OWNER_INVALID"
        )
    commit = raw["producer_git_commit"]
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(char not in "0123456789abcdef" for char in commit)
    ):
        raise PrebuiltIdentityError("PREBUILT_PAIR_LINEAGE_GIT_COMMIT_INVALID")
    if raw["producer_repository_clean"] is not True:
        raise PrebuiltIdentityError("PREBUILT_PAIR_LINEAGE_DIRTY_REPOSITORY")
    source_files = raw["producer_source_files"]
    if not isinstance(source_files, list) or not source_files:
        raise PrebuiltIdentityError(
            "PREBUILT_PAIR_LINEAGE_SOURCE_FILES_INVALID"
        )
    previous_path = ""
    for item in source_files:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise PrebuiltIdentityError(
                "PREBUILT_PAIR_LINEAGE_SOURCE_FILE_INVALID"
            )
        path = item["path"]
        if (
            not isinstance(path, str)
            or not path
            or path <= previous_path
            or not _valid_sha256(item["sha256"])
        ):
            raise PrebuiltIdentityError(
                "PREBUILT_PAIR_LINEAGE_SOURCE_FILE_INVALID"
            )
        previous_path = path
    if (
        not _valid_sha256(raw["producer_source_inventory_sha256"])
        or raw["producer_source_inventory_sha256"] != _sha256_json(source_files)
    ):
        raise PrebuiltIdentityError(
            "PREBUILT_PAIR_LINEAGE_SOURCE_INVENTORY_MISMATCH"
        )
    native = raw["native_sources"]
    if not isinstance(native, dict) or set(native) != {"m1", "m5"}:
        raise PrebuiltIdentityError("PREBUILT_PAIR_NATIVE_SOURCES_INVALID")
    mandatory_native = {
        "root",
        "manifest_sha256",
        "instrument",
        "timeframe",
        "explicit_vedtak_id",
        "source_environment",
        "source_base_url",
        "requested_start_utc",
        "requested_end_utc_exclusive",
        "row_count",
        "time_min_utc",
        "time_max_utc",
        "canonical_rows_sha256",
        "producer_git_commit",
        "producer_source_inventory_sha256",
        "manifest_payload_sha256",
    }
    exact_native_v2 = mandatory_native | {
        "manifest_path",
        "year_sha256",
        "year_rows",
    }
    for label, timeframe in (("m1", "M1"), ("m5", "M5")):
        descriptor = native[label]
        if (
            not isinstance(descriptor, dict)
            or not mandatory_native.issubset(descriptor)
            or (
                lineage_schema_version == PREBUILT_PAIR_LINEAGE_SCHEMA_VERSION
                and set(descriptor) != exact_native_v2
            )
            or descriptor.get("instrument") != "XAU_USD"
            or descriptor.get("timeframe") != timeframe
            or descriptor.get("explicit_vedtak_id") != raw["explicit_vedtak_id"]
        ):
            raise PrebuiltIdentityError(
                f"PREBUILT_PAIR_NATIVE_{timeframe}_DESCRIPTOR_INVALID"
            )
        for digest_field in (
            "manifest_sha256",
            "canonical_rows_sha256",
            "producer_source_inventory_sha256",
            "manifest_payload_sha256",
        ):
            if not _valid_sha256(descriptor.get(digest_field)):
                raise PrebuiltIdentityError(
                    f"PREBUILT_PAIR_NATIVE_{timeframe}_{digest_field.upper()}_INVALID"
                )
        if (
            not isinstance(descriptor.get("row_count"), int)
            or isinstance(descriptor.get("row_count"), bool)
            or descriptor["row_count"] <= 0
        ):
            raise PrebuiltIdentityError(
                f"PREBUILT_PAIR_NATIVE_{timeframe}_ROWS_INVALID"
            )
        if lineage_schema_version == PREBUILT_PAIR_LINEAGE_SCHEMA_VERSION:
            root = descriptor.get("root")
            manifest_path = descriptor.get("manifest_path")
            year_sha256 = descriptor.get("year_sha256")
            year_rows = descriptor.get("year_rows")
            if (
                not isinstance(root, str)
                or not Path(root).is_absolute()
                or not isinstance(manifest_path, str)
                or not Path(manifest_path).is_absolute()
                or not isinstance(year_sha256, dict)
                or not year_sha256
                or not isinstance(year_rows, dict)
                or set(year_sha256) != set(year_rows)
                or not all(_valid_sha256(value) for value in year_sha256.values())
                or not all(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value > 0
                    for value in year_rows.values()
                )
            ):
                raise PrebuiltIdentityError(
                    f"PREBUILT_PAIR_NATIVE_{timeframe}_IMMUTABLE_LAYOUT_INVALID"
                )
            try:
                requested_start = pd.Timestamp(
                    descriptor["requested_start_utc"]
                )
                requested_end = pd.Timestamp(
                    descriptor["requested_end_utc_exclusive"]
                )
                time_min = pd.Timestamp(descriptor["time_min_utc"])
                time_max = pd.Timestamp(descriptor["time_max_utc"])
            except Exception as exc:
                raise PrebuiltIdentityError(
                    f"PREBUILT_PAIR_NATIVE_{timeframe}_TIME_IDENTITY_INVALID"
                ) from exc
            timestamps = (
                requested_start,
                requested_end,
                time_min,
                time_max,
            )
            if (
                any(
                    value.tzinfo is None
                    or value.utcoffset() != pd.Timedelta(0)
                    for value in timestamps
                )
                or requested_start >= requested_end
                or time_min > time_max
                or time_min < requested_start
                or time_max >= requested_end
            ):
                raise PrebuiltIdentityError(
                    f"PREBUILT_PAIR_NATIVE_{timeframe}_TIME_IDENTITY_INVALID"
                )
    for shared_field in (
        "source_environment",
        "source_base_url",
        "requested_start_utc",
        "requested_end_utc_exclusive",
    ):
        if native["m1"].get(shared_field) != native["m5"].get(shared_field):
            raise PrebuiltIdentityError(
                f"PREBUILT_PAIR_NATIVE_SOURCE_{shared_field.upper()}_MISMATCH"
            )
    derivation = raw["derivation_contract"]
    if (
        not isinstance(derivation, dict)
        or (
            lineage_schema_version == PREBUILT_PAIR_LINEAGE_SCHEMA_VERSION
            and set(derivation)
            != {
                "canonical_builder",
                "canonical_ordered_columns_sha256",
                "raw_base28_columns",
                "raw_base28_columns_sha256",
                "rank_fit_fields_absent",
                "m5_phase_owned_by_m1_time",
                "formula_contract",
                "timing_contract",
            }
        )
        or derivation.get("raw_base28_columns") != list(RAW_BASE28_COLUMNS)
        or derivation.get("rank_fit_fields_absent") is not True
        or derivation.get("m5_phase_owned_by_m1_time") is not True
        or not isinstance(derivation.get("formula_contract"), dict)
        or not derivation["formula_contract"]
        or not isinstance(derivation.get("timing_contract"), dict)
        or not derivation["timing_contract"]
    ):
        raise PrebuiltIdentityError(
            "PREBUILT_PAIR_DERIVATION_CONTRACT_INVALID"
        )
    if lineage_schema_version == PREBUILT_PAIR_LINEAGE_SCHEMA_VERSION:
        expected_raw_sha = _sha256_json(list(RAW_BASE28_COLUMNS))
        if (
            derivation["canonical_builder"]
            != PREBUILT_CANONICAL_BUILDER_CONTRACT
            or not _valid_sha256(
                derivation["canonical_ordered_columns_sha256"]
            )
            or derivation["raw_base28_columns_sha256"] != expected_raw_sha
            or derivation["formula_contract"] != PREBUILT_PAIR_FORMULA_CONTRACT
            or derivation["timing_contract"] != PREBUILT_PAIR_TIMING_CONTRACT
        ):
            raise PrebuiltIdentityError(
                "PREBUILT_PAIR_DERIVATION_CONTRACT_IDENTITY_MISMATCH"
            )
    coverage = raw["coverage"]
    if not isinstance(coverage, dict) or not coverage:
        raise PrebuiltIdentityError("PREBUILT_PAIR_COVERAGE_INVALID")
    if lineage_schema_version == PREBUILT_PAIR_LINEAGE_SCHEMA_VERSION:
        expected_coverage_fields = {
            "native_m1_rows",
            "native_m5_rows",
            "canonical_rows",
            "base28_rows",
            "canonical_time_min_utc",
            "canonical_time_max_utc",
            "base28_time_min_utc",
            "base28_time_max_utc",
            "canonical_warmup_prefix_rows_trimmed",
        }
        if set(coverage) != expected_coverage_fields:
            raise PrebuiltIdentityError(
                "PREBUILT_PAIR_COVERAGE_FIELDS_INVALID"
            )
        for name in (
            "native_m1_rows",
            "native_m5_rows",
            "canonical_rows",
            "base28_rows",
        ):
            if (
                not isinstance(coverage[name], int)
                or isinstance(coverage[name], bool)
                or coverage[name] <= 0
            ):
                raise PrebuiltIdentityError(
                    f"PREBUILT_PAIR_COVERAGE_{name.upper()}_INVALID"
                )
        trimmed = coverage["canonical_warmup_prefix_rows_trimmed"]
        if (
            not isinstance(trimmed, int)
            or isinstance(trimmed, bool)
            or trimmed < 0
            or coverage["native_m5_rows"] - coverage["canonical_rows"]
            != trimmed
        ):
            raise PrebuiltIdentityError(
                "PREBUILT_PAIR_COVERAGE_WARMUP_INVALID"
            )
    parent = raw["parent_pair_generation_id"]
    if parent is not None and not _valid_sha256(parent):
        raise PrebuiltIdentityError("PREBUILT_PAIR_PARENT_ID_INVALID")
    if lineage_schema_version == PREBUILT_PAIR_LINEAGE_SCHEMA_VERSION:
        parent_manifest_sha256 = raw["parent_pair_manifest_sha256"]
        if (parent is None) != (parent_manifest_sha256 is None):
            raise PrebuiltIdentityError(
                "PREBUILT_PAIR_PARENT_MANIFEST_IDENTITY_INCOMPLETE"
            )
        if parent is not None and not _valid_sha256(parent_manifest_sha256):
            raise PrebuiltIdentityError(
                "PREBUILT_PAIR_PARENT_MANIFEST_SHA256_INVALID"
            )
    # Break references to caller-owned mutable nested objects.
    return json.loads(json.dumps(raw, sort_keys=True))


def _artifact_generation_identity(
    artifacts: dict[str, dict[str, object]],
    lineage: dict[str, object],
    *,
    schema_version: str = PREBUILT_PAIR_SCHEMA_VERSION,
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "lineage": lineage,
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
    *,
    lineage: dict[str, object],
    schema_version: str = PREBUILT_PAIR_SCHEMA_VERSION,
) -> str:
    """Return the only pair generation id.

    It is SHA-256 over canonical JSON containing the schema version and, in
    fixed canonical-v3/BASE28 order, each artifact's SHA-256, row count,
    feature-column count and full Arrow schema. Paths and wall-clock time are
    excluded, so identity means content contract rather than publication site.
    """
    validated_lineage = validate_prebuilt_pair_lineage(lineage)
    return _sha256_json(
        _artifact_generation_identity(
            artifacts,
            validated_lineage,
            schema_version=schema_version,
        )
    )


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
    physical_names = [item[0] for item in parsed_schema]
    if label == "base28":
        expected_names = ["time", *RAW_BASE28_COLUMNS]
        if physical_names != expected_names:
            raise PrebuiltIdentityError(
                "BASE28_RAW_SCHEMA_INVALID: "
                f"observed={physical_names} expected={expected_names}"
            )
    else:
        forbidden = sorted(
            PAIR_FORBIDDEN_CANONICAL_FIELDS & set(physical_names)
        )
        if forbidden:
            raise PrebuiltIdentityError(
                f"CANONICAL_V3_DATASET_FIT_OR_M1_TIME_FIELDS_FORBIDDEN: {forbidden}"
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
    schema_version = manifest.get("schema_version")
    common_fields = {
        "schema_version",
        "pair_generation_id",
        "created_utc",
        "lineage",
        "artifacts",
    }
    if schema_version == PREBUILT_PAIR_SCHEMA_VERSION:
        expected_fields = common_fields | {"generation_manifest_required"}
    elif schema_version == PREBUILT_PAIR_LEGACY_SCHEMA_VERSION:
        expected_fields = common_fields
    else:
        raise PrebuiltIdentityError("PREBUILT_PAIR_SCHEMA_VERSION_INVALID")
    if set(manifest) != expected_fields:
        raise PrebuiltIdentityError("PREBUILT_PAIR_MANIFEST_FIELDS_INVALID")
    if (
        schema_version == PREBUILT_PAIR_SCHEMA_VERSION
        and manifest["generation_manifest_required"] is not True
    ):
        raise PrebuiltIdentityError(
            "PREBUILT_PAIR_GENERATION_MANIFEST_REQUIREMENT_INVALID"
        )
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
    lineage = validate_prebuilt_pair_lineage(manifest["lineage"])
    try:
        observed_generation_id = pair_generation_id_for_artifacts(
            artifacts,
            lineage=lineage,
            schema_version=str(schema_version),
        )
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
    if lineage["schema_version"] == PREBUILT_PAIR_LINEAGE_SCHEMA_VERSION:
        coverage = lineage["coverage"]
        native_sources = lineage["native_sources"]
        canonical_names = [
            name
            for name, _dtype, _nullable in canonical.arrow_schema
            if name != "time"
        ]
        if (
            coverage["canonical_rows"] != canonical.rows
            or coverage["base28_rows"] != base28.rows
            or coverage["native_m1_rows"]
            != native_sources["m1"]["row_count"]
            or coverage["native_m5_rows"]
            != native_sources["m5"]["row_count"]
            or lineage["derivation_contract"][
                "canonical_ordered_columns_sha256"
            ]
            != _sha256_json(canonical_names)
        ):
            raise PrebuiltIdentityError(
                "PREBUILT_PAIR_LINEAGE_ARTIFACT_COVERAGE_MISMATCH"
            )
    generation_manifest_path: Path | None = None
    if schema_version == PREBUILT_PAIR_SCHEMA_VERSION:
        generation_manifest_path = (
            generation_root
            / pair_generation_id
            / PREBUILT_PAIR_GENERATION_MANIFEST_FILENAME
        )
        if (
            generation_manifest_path.is_symlink()
            or not generation_manifest_path.is_file()
        ):
            raise PrebuiltIdentityError(
                "PREBUILT_PAIR_GENERATION_MANIFEST_MISSING"
            )
        generation_before = _file_stamp(
            generation_manifest_path,
            label="PREBUILT_PAIR_GENERATION_MANIFEST",
        )
        generation_raw = generation_manifest_path.read_bytes()
        generation_after = _file_stamp(
            generation_manifest_path,
            label="PREBUILT_PAIR_GENERATION_MANIFEST",
        )
        if generation_before != generation_after:
            raise PrebuiltIdentityError(
                "PREBUILT_PAIR_GENERATION_MANIFEST_CHANGED_DURING_READ"
            )
        if generation_raw != raw:
            raise PrebuiltIdentityError(
                "PREBUILT_PAIR_POINTER_GENERATION_MANIFEST_MISMATCH"
            )
    return _PrebuiltPairBinding(
        manifest_path=manifest_path.resolve(strict=True),
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        generation_manifest_path=generation_manifest_path,
        pair_generation_id=pair_generation_id,
        lineage=lineage,
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
    if (
        binding.lineage["schema_version"]
        == PREBUILT_PAIR_LINEAGE_SCHEMA_VERSION
    ):
        coverage = binding.lineage["coverage"]
        for artifact, first_field, last_field in (
            (
                canonical,
                "canonical_time_min_utc",
                "canonical_time_max_utc",
            ),
            (base28, "base28_time_min_utc", "base28_time_max_utc"),
        ):
            try:
                time_frame = pd.read_parquet(
                    artifact.binding.parquet_path,
                    columns=["time"],
                )
                times = pd.DatetimeIndex(
                    pd.to_datetime(
                        time_frame["time"],
                        utc=True,
                        errors="raise",
                    )
                )
            except Exception as exc:
                raise PrebuiltIdentityError(
                    f"{artifact.binding.label}_COVERAGE_TIME_READ_INVALID"
                ) from exc
            if (
                len(times) != artifact.binding.rows
                or times.hasnans
                or times.has_duplicates
                or not times.is_monotonic_increasing
                or times[0].isoformat() != coverage[first_field]
                or times[-1].isoformat() != coverage[last_field]
                or _file_stamp(
                    artifact.binding.parquet_path,
                    label=artifact.binding.label,
                )
                != artifact.file_stamp
            ):
                raise PrebuiltIdentityError(
                    f"{artifact.binding.label}_LINEAGE_TIME_COVERAGE_MISMATCH"
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


def require_prebuilt_pair_parent(
    lineage: dict[str, object],
    *,
    expected_parent_pair_generation_id: str | None,
    expected_parent_manifest_sha256: str | None,
) -> None:
    """Require bootstrap ``None`` or the exact current parent ID and bytes."""

    validated = validate_prebuilt_pair_lineage(lineage)
    observed = validated["parent_pair_generation_id"]
    observed_manifest_sha256 = validated.get(
        "parent_pair_manifest_sha256"
    )
    if (
        observed != expected_parent_pair_generation_id
        or observed_manifest_sha256 != expected_parent_manifest_sha256
    ):
        raise PrebuiltIdentityError(
            "PREBUILT_PAIR_PARENT_MISMATCH: "
            f"observed={observed!r} "
            f"expected={expected_parent_pair_generation_id!r} "
            f"observed_manifest_sha256={observed_manifest_sha256!r} "
            f"expected_manifest_sha256={expected_parent_manifest_sha256!r}"
        )


def require_prebuilt_successor_frame(
    current: pd.DataFrame,
    successor: pd.DataFrame,
    *,
    label: str,
) -> None:
    """Prove one immutable artifact is a strict, bit-exact prefix successor."""

    if (
        not isinstance(current, pd.DataFrame)
        or not isinstance(successor, pd.DataFrame)
        or current.empty
        or successor.empty
        or not isinstance(current.index, pd.DatetimeIndex)
        or not isinstance(successor.index, pd.DatetimeIndex)
        or current.index.hasnans
        or successor.index.hasnans
        or not current.index.is_unique
        or not successor.index.is_unique
        or not current.index.is_monotonic_increasing
        or not successor.index.is_monotonic_increasing
    ):
        raise PrebuiltIdentityError(
            f"{label}_PREBUILT_SUCCESSOR_FRAME_INVALID"
        )
    if (
        list(successor.columns) != list(current.columns)
        or tuple(str(dtype) for dtype in successor.dtypes)
        != tuple(str(dtype) for dtype in current.dtypes)
    ):
        raise PrebuiltIdentityError(
            f"{label}_PREBUILT_SUCCESSOR_SCHEMA_MISMATCH"
        )
    if (
        len(successor) <= len(current)
        or successor.index[-1] <= current.index[-1]
    ):
        raise PrebuiltIdentityError(
            f"{label}_PREBUILT_SUCCESSOR_NOT_STRICTLY_ADVANCING"
        )
    prefix = successor.iloc[: len(current)]
    current_arrow = pa.Table.from_pandas(
        current,
        preserve_index=True,
    ).combine_chunks()
    prefix_arrow = pa.Table.from_pandas(
        prefix,
        preserve_index=True,
    ).combine_chunks()
    current_buffers = tuple(
        tuple(
            None if buffer is None else buffer.to_pybytes()
            for buffer in current_arrow.column(index).chunk(0).buffers()
        )
        for index in range(current_arrow.num_columns)
    )
    prefix_buffers = tuple(
        tuple(
            None if buffer is None else buffer.to_pybytes()
            for buffer in prefix_arrow.column(index).chunk(0).buffers()
        )
        for index in range(prefix_arrow.num_columns)
    )
    if (
        current_arrow.schema != prefix_arrow.schema
        or current_buffers != prefix_buffers
    ):
        raise PrebuiltIdentityError(
            f"{label}_PREBUILT_SUCCESSOR_PREFIX_MISMATCH"
        )


def _persisted_source_view(
    frame: pd.DataFrame,
    schema: tuple[tuple[str, str, bool], ...] | None,
    *,
    label: str,
) -> pd.DataFrame:
    """Project an augmented in-memory frame back to its persisted columns."""

    if schema is None:
        raise PrebuiltIdentityError(f"{label}_PERSISTED_SCHEMA_UNBOUND")
    columns = [name for name, _dtype, _nullable in schema if name != "time"]
    if not columns or any(name not in frame.columns for name in columns):
        raise PrebuiltIdentityError(f"{label}_PERSISTED_SOURCE_FIELDS_MISSING")
    return frame.loc[:, columns]


def _require_persisted_model_agnostic_canonical(
    frame: pd.DataFrame,
) -> None:
    """Prove that pair-v2 already contains the shared deterministic surface."""

    from gx1.contracts.entry_model_native_signal_v1 import (
        MODEL_NATIVE_CTX_CONT_DIP_STRUCT_FIELDS,
        MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS,
        MODEL_NATIVE_CTX_CONT_MICRO_FIELDS,
        MODEL_NATIVE_CTX_CONT_SESSION_FIELDS,
        MODEL_NATIVE_CTX_CONT_SWING_FIELDS,
    )
    from gx1.features.regime_v4_features import (
        REGIME_V4_DERIVED_COLS,
        REGIME_V4_SOURCE_COLS,
    )
    from gx1.features.volume_features import VOLUME_FEATURE_NAMES

    required = {
        *CANONICAL_NATIVE_REQUIRED_COLUMNS[1:],
        *PLUS5_FEATURES,
        *VOLUME_FEATURE_NAMES,
        *MODEL_NATIVE_CTX_CONT_MICRO_FIELDS,
        *MODEL_NATIVE_CTX_CONT_SWING_FIELDS,
        *MODEL_NATIVE_CTX_CONT_SESSION_FIELDS,
        *MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS,
        *MODEL_NATIVE_CTX_CONT_DIP_STRUCT_FIELDS,
        *REGIME_V4_SOURCE_COLS,
        *REGIME_V4_DERIVED_COLS,
        "session_id",
        "trend_regime_id",
        "vol_regime_id",
        "H4_trend_sign_cat",
        "atr_bps",
        "spread_bps",
        *(
            f"{timeframe}_{live_fragment}_v2"
            for timeframe in REGIME_V4_MTF_TIMEFRAMES
            for live_fragment, _source_column in REGIME_V4_MTF_PROJECTION
            if (timeframe, live_fragment) not in REGIME_V4_MTF_SKIP
        ),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PrebuiltIdentityError(
            f"CANONICAL_V3_PERSISTED_MODEL_AGNOSTIC_FIELDS_MISSING: {missing}"
        )
    forbidden = sorted(
        {"atr_bucket", "spread_bucket"} & set(frame.columns)
    )
    if forbidden:
        raise PrebuiltIdentityError(
            f"CANONICAL_V3_TRAIN_FIT_FIELDS_FORBIDDEN: {forbidden}"
        )
    numeric = frame.loc[:, sorted(required)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise PrebuiltIdentityError(
            "CANONICAL_V3_PERSISTED_MODEL_AGNOSTIC_NONFINITE"
        )


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
    # One lock owns the complete in-memory serving capsule. A sequence of GIL-
    # atomic assignments is not an atomic state transition for a concurrent
    # reader; every public read and the refresh swap therefore use this lock.
    _state_lock: threading.RLock = field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
    )
    # 2026-06-01: async refresh — background thread runs the full 5-augmenter
    # pipeline on a LOCAL copy so the M1 exit-path never blocks on cv3 reload.
    # When the thread finishes, it swaps the entire serving capsule under
    # _state_lock; readers therefore observe one old or one new generation.
    _refresh_thread: threading.Thread | None = field(default=None, init=False)
    _refresh_enabled: bool = field(default=True, init=False)
    # An async error invalidates the old snapshot. Every public read and future
    # refresh raises until a fresh process completes load() successfully.
    _refresh_error: RuntimeError | None = field(default=None, init=False)
    # One immutable TRAIN-only rank reference bound by the consuming Exit
    # contract. TRAIN-fit state is deliberately outside the persisted pair;
    # the exact bucket fields exist only after an explicit attach.
    _train_rank_reference: "TrainRankReferenceV2 | None" = field(
        default=None,
        init=False,
    )

    # ── TRAIN-rank reference binding (Exit-chain bucket ownership) ────
    def attach_train_rank_reference(self, reference: "TrainRankReferenceV2") -> None:
        """Bind one immutable TRAIN-rank reference and derive the exact
        ``atr_bucket``/``spread_bucket`` fields on the loaded canonical frame.

        The persisted pair is model-agnostic by contract (bucket fields are
        forbidden on disk and validated at admission, before this call). The
        derivation reuses the single formula owner
        ``bucket_against_train_reference``; a second attach — same or different
        bytes — is a wiring error and fails closed.
        """

        from gx1.contracts.entry_model_native_state_v2 import (
            TrainRankReferenceV2,
        )

        if not isinstance(reference, TrainRankReferenceV2):
            raise RuntimeError("PREBUILT_TRAIN_RANK_REFERENCE_TYPE_INVALID")
        with self._state_lock:
            self._raise_refresh_error()
            if self._cv3 is None:
                raise RuntimeError(
                    "PREBUILT_TRAIN_RANK_REFERENCE_REQUIRES_LOAD: "
                    "call load() or load_frozen_pair() first"
                )
            if self._train_rank_reference is not None:
                raise RuntimeError(
                    "PREBUILT_TRAIN_RANK_REFERENCE_ALREADY_ATTACHED: "
                    f"bound_sha256={self._train_rank_reference.sha256} "
                    f"offered_sha256={reference.sha256}"
                )
            self._derive_train_rank_buckets(self._cv3, reference=reference)
            self._train_rank_reference = reference

    def _derive_train_rank_buckets(
        self,
        cv3: pd.DataFrame,
        *,
        reference: "TrainRankReferenceV2",
    ) -> pd.DataFrame:
        """Derive both bucket fields in place via the one formula owner."""

        from gx1.contracts.entry_model_native_state_v2 import (
            bucket_against_train_reference,
        )

        missing = [
            name for name in ("atr_bps", "spread_bps") if name not in cv3.columns
        ]
        if missing:
            raise RuntimeError(
                f"PREBUILT_TRAIN_RANK_SOURCE_FIELDS_MISSING: {missing}"
            )
        cv3["atr_bucket"] = bucket_against_train_reference(
            cv3["atr_bps"].to_numpy(dtype=float),
            reference.atr_bps_sorted,
        )
        cv3["spread_bucket"] = bucket_against_train_reference(
            cv3["spread_bps"].to_numpy(dtype=float),
            reference.spread_bps_sorted,
        )
        return cv3

    def train_rank_reference_attached(self) -> bool:
        """True only when one immutable reference is bound to this loader."""

        with self._state_lock:
            return self._train_rank_reference is not None

    def train_rank_reference_binding(self) -> dict[str, object]:
        """Exact bindable identity of the attached reference (fail-closed)."""

        from gx1.contracts.entry_model_native_state_v2 import (
            train_rank_reference_identity_v2,
        )

        with self._state_lock:
            if self._train_rank_reference is None:
                raise RuntimeError("PREBUILT_TRAIN_RANK_REFERENCE_UNBOUND")
            return train_rank_reference_identity_v2(self._train_rank_reference)

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
        if self._pair_binding is None:
            raise PrebuiltIdentityError("PREBUILT_CURRENT_PAIR_UNBOUND")
        if pair_binding == self._pair_binding:
            if (
                _file_stamp(
                    pair_binding.canonical_v3.parquet_path,
                    label="CANONICAL_V3",
                )
                != self._cv3_file_stamp
                or _file_stamp(
                    pair_binding.base28.parquet_path,
                    label="BASE28",
                )
                != self._base28_file_stamp
            ):
                raise PrebuiltIdentityError(
                    "PREBUILT_PAIR_IMMUTABLE_ARTIFACT_REWRITTEN"
                )
            return None
        if (
            pair_binding.pair_generation_id
            == self._pair_binding.pair_generation_id
        ):
            raise PrebuiltIdentityError(
                "PREBUILT_PAIR_IDENTITY_REWRITTEN_WITHOUT_ADVANCE"
            )
        require_prebuilt_pair_parent(
            pair_binding.lineage,
            expected_parent_pair_generation_id=(
                self._pair_binding.pair_generation_id
            ),
            expected_parent_manifest_sha256=(
                self._pair_binding.manifest_sha256
            ),
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
        with self._state_lock:
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
            with self._state_lock:
                self._raise_refresh_error()
                admitted_parent = self._pair_binding
                current_cv3 = self._cv3
                current_b28 = self._base28
                cv3_source_schema = self._cv3_source_schema
                base28_source_schema = self._base28_source_schema
                train_rank_reference = self._train_rank_reference
                had_multi_tf = self._multi_tf_feats is not None
            if admitted_parent is None or current_cv3 is None or current_b28 is None:
                raise PrebuiltIdentityError(
                    "PREBUILT_REFRESH_PARENT_STATE_UNAVAILABLE"
                )
            require_prebuilt_pair_parent(
                pair_binding.lineage,
                expected_parent_pair_generation_id=(
                    admitted_parent.pair_generation_id
                ),
                expected_parent_manifest_sha256=(
                    admitted_parent.manifest_sha256
                ),
            )

            # --- cv3 ---
            new_cv3 = current_cv3
            if cv3_advanced:
                new_cv3, loaded_cv3 = _load_verified_prebuilt(
                    cv3_verified.binding,
                    expected_schema=cv3_source_schema,
                )
                if loaded_cv3 != cv3_verified:
                    raise PrebuiltIdentityError(
                        "CANONICAL_V3_CHANGED_AFTER_REFRESH_ADMISSION"
                    )
            # --- base28 ---
            new_b28 = current_b28
            if b28_advanced:
                new_b28, loaded_base28 = _load_verified_prebuilt(
                    base28_verified.binding,
                    expected_schema=base28_source_schema,
                )
                if loaded_base28 != base28_verified:
                    raise PrebuiltIdentityError(
                        "BASE28_CHANGED_AFTER_REFRESH_ADMISSION"
                    )
            require_prebuilt_successor_frame(
                _persisted_source_view(
                    current_cv3,
                    cv3_source_schema,
                    label="CANONICAL_V3",
                ),
                new_cv3,
                label="CANONICAL_V3",
            )
            require_prebuilt_successor_frame(
                _persisted_source_view(
                    current_b28,
                    base28_source_schema,
                    label="BASE28",
                ),
                new_b28,
                label="BASE28",
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
                if cv3_advanced:
                    # The persisted-surface proof applies to the freshly loaded
                    # bytes. On a BASE28-only advance, new_cv3 is the already
                    # admitted in-memory frame, which legitimately carries the
                    # attached TRAIN-rank bucket fields.
                    _require_persisted_model_agnostic_canonical(new_cv3)
                # Fast augmenters in main thread (~3s total)
                new_cv3 = self._augment_cv3_with_volume_features(new_cv3)
                new_cv3 = self._augment_cv3_with_v1_legacy(new_cv3)

                # Heavy augmenters in 2 subprocesses (~95s wall, was 180s sequential)
                try:
                    import multiprocessing as _mp
                    from concurrent.futures import ProcessPoolExecutor
                    fork_ctx = _mp.get_context("fork")
                    with ProcessPoolExecutor(max_workers=2, mp_context=fork_ctx) as pool:
                        f_mtf = pool.submit(_mp_v4_mtf_worker, new_cv3)
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
                    # Keep the transition explicit so a first refresh cannot
                    # fabricate regime state.
                    # Runs AFTER the mtf merge above (supplies the 12 {tf}_*_v2 sources); joins D1_dist + is a
                    # no-op at flag=0 (cement bit-identical).
                    new_cv3 = self._augment_cv3_with_regime_v4(new_cv3)
                except Exception as exc:
                    raise RuntimeError(
                        "parallel augment failed; refusing a second transform path"
                    ) from exc
                # A freshly loaded persisted cv3 is model-agnostic by contract.
                # When one immutable TRAIN-rank reference is attached, the
                # exact bucket fields must be re-derived BEFORE the atomic
                # swap so live never loses them on a refresh cycle.
                if cv3_advanced and train_rank_reference is not None:
                    new_cv3 = self._derive_train_rank_buckets(
                        new_cv3,
                        reference=train_rank_reference,
                    )

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
            if had_multi_tf and new_cv3 is not None:
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

            # --- one atomic serving-capsule swap under the state lock ---
            with self._state_lock:
                if self._pair_binding != admitted_parent:
                    raise PrebuiltIdentityError(
                        "PREBUILT_IN_MEMORY_PARENT_CHANGED_DURING_REFRESH"
                    )
                old_cutoff = (
                    self._cv3.index[-1] if self._cv3 is not None else None
                )
                self._cv3 = new_cv3
                self._base28 = new_b28
                self.canonical_v3_path = cv3_verified.binding.parquet_path
                self.base28_path = base28_verified.binding.parquet_path
                self._cv3_binding = cv3_verified.binding
                self._base28_binding = base28_verified.binding
                self._pair_binding = pair_binding
                self._cv3_file_stamp = cv3_verified.file_stamp
                self._base28_file_stamp = base28_verified.file_stamp
                self._cv3_mtime = (
                    cv3_verified.file_stamp[2] / 1_000_000_000.0
                )
                self._base28_mtime = (
                    base28_verified.file_stamp[2] / 1_000_000_000.0
                )
                self._last_ts = new_last_ts
                if new_mtf_bundle is not None:
                    self._multi_tf_feats = new_mtf_bundle["feats"]
                    self._multi_tf_shift = new_mtf_bundle["shift"]
                    self._multi_tf_feat_count = new_mtf_bundle["feat_count"]
                new_cutoff = (
                    self._cv3.index[-1] if self._cv3 is not None else None
                )

            took_sec = (pd.Timestamp.utcnow() - t_start).total_seconds()
            LOG.info(f"[async-refresh] cv3 cutoff {old_cutoff} → {new_cutoff} "
                     f"(took {took_sec:.1f}s, main loop never blocked)")
        except Exception as exc:
            failure = RuntimeError(
                f"PREBUILT_ASYNC_REFRESH_FAILED: "
                f"{type(exc).__name__}: {exc}"
            )
            with self._state_lock:
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

        with self._state_lock:
            LOG.info(
                f"loading canonical_v3 prebuilt: {cv3_binding.parquet_path.name}"
            )
            self._cv3 = cv3
            self.canonical_v3_path = cv3_binding.parquet_path
            self._cv3_binding = cv3_binding
            self._pair_binding = pair_binding
            self._cv3_source_schema = cv3_verified.arrow_schema
            self._cv3_file_stamp = cv3_verified.file_stamp
            self._cv3_mtime = cv3_verified.file_stamp[2] / 1_000_000_000.0
            LOG.info(
                f"  canonical_v3: {len(cv3):,} rows × "
                f"{len(cv3.columns)} cols  "
                f"range {cv3.index[0]} → {cv3.index[-1]}"
            )

            LOG.info(
                f"loading BASE28 prebuilt: {base28_binding.parquet_path.name}"
            )
            self._base28 = base28
            self.base28_path = base28_binding.parquet_path
            self._base28_binding = base28_binding
            self._base28_source_schema = base28_verified.arrow_schema
            self._base28_file_stamp = base28_verified.file_stamp
            self._base28_mtime = (
                base28_verified.file_stamp[2] / 1_000_000_000.0
            )
            LOG.info(
                f"  BASE28: {len(base28):,} rows × "
                f"{len(base28.columns)} cols  "
                f"range {base28.index[0]} → {base28.index[-1]}"
            )

            _require_persisted_model_agnostic_canonical(cv3)
            self._last_ts = min(cv3.index[-1], base28.index[-1])
            LOG.info(f"  effective live cutoff (min of both): {self._last_ts}")
            self._augment_cv3_with_volume_features()
            self._augment_cv3_with_v4_mtf_scalars()
            self._augment_cv3_with_group_a_and_dip_struct()
            self._augment_cv3_with_v1_legacy()
            # after v2_mtf sources; immutable active transform
            self._augment_cv3_with_regime_v4()
            self._refresh_error = None

    def load_frozen_pair(self) -> dict[str, object]:
        """Load one exact pair generation and disable mutable refresh.

        Historical unified-Exit replay must observe one immutable canonical-v3
        and BASE28 generation for its entire lifetime.  The owner returns the
        exact admitted identity so a producer can bind it in replay evidence.
        """

        self._refresh_enabled = False
        self.load()
        _canonical, _base, identity = self.frozen_pair_frames()
        return identity

    def frozen_pair_frames(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
        """Return the exact loaded canonical/base frames and rechecked identity.

        Dataset and replay producers may consume the same frozen pair as live
        inference without reaching through private attributes.  This method
        never reloads, sorts, refreshes or substitutes either frame.
        """

        if (
            self._refresh_enabled is not False
            or self._pair_binding is None
            or self._cv3 is None
            or self._base28 is None
            or self._cv3_file_stamp is None
            or self._base28_file_stamp is None
        ):
            raise PrebuiltIdentityError("PREBUILT_FROZEN_PAIR_NOT_LOADED")
        binding = read_prebuilt_pair_manifest(
            self.pair_manifest_path,
            generation_root=self.generation_root,
        )
        if binding != self._pair_binding:
            raise PrebuiltIdentityError(
                "PREBUILT_FROZEN_PAIR_MANIFEST_CHANGED"
            )
        if (
            _file_stamp(binding.canonical_v3.parquet_path, label="CANONICAL_V3")
            != self._cv3_file_stamp
            or _file_stamp(binding.base28.parquet_path, label="BASE28")
            != self._base28_file_stamp
        ):
            raise PrebuiltIdentityError(
                "PREBUILT_FROZEN_PAIR_FILES_CHANGED"
            )
        identity = {
            "manifest_path": str(binding.manifest_path),
            "manifest_sha256": binding.manifest_sha256,
            "pair_generation_id": binding.pair_generation_id,
            "lineage": binding.lineage,
            "lineage_sha256": _sha256_json(binding.lineage),
            "canonical_v3": {
                "path": str(binding.canonical_v3.parquet_path),
                "sha256": binding.canonical_v3.parquet_sha256,
                "rows": binding.canonical_v3.rows,
                "cols_total": binding.canonical_v3.cols_total,
            },
            "base28": {
                "path": str(binding.base28.parquet_path),
                "sha256": binding.base28.parquet_sha256,
                "rows": binding.base28.rows,
                "cols_total": binding.base28.cols_total,
            },
            "refresh_enabled": False,
        }
        return self._cv3, self._base28, identity

    _MTF_PROJECTION = REGIME_V4_MTF_PROJECTION
    _MTF_TIMEFRAMES = REGIME_V4_MTF_TIMEFRAMES
    _MTF_SKIP = REGIME_V4_MTF_SKIP

    @classmethod
    def _expected_mtf_scalar_columns(cls) -> list[str]:
        return [
            f"{tf}_{live_frag}_v2"
            for tf in cls._MTF_TIMEFRAMES
            for live_frag, _source_col in cls._MTF_PROJECTION
            if (tf, live_frag) not in cls._MTF_SKIP
        ]

    # ── V1 / R10 (2026-06-04): REGIME_V4 exit-context augmentation ────────
    def _augment_cv3_with_regime_v4(self, cv3: pd.DataFrame | None = None) -> pd.DataFrame | None:
        """Compute the 16 REGIME_V4 EXIT_IO_V8 features on the FULL cv3 through
        the canonical `add_regime_v4_features` owner. Any future V3 dataset
        builder must call the same owner and prove parity. This immutable active
        transform MUST run AFTER
        _augment_cv3_with_v4_mtf_scalars (supplies the persistent {tf}_*_v2 source cols) and on the FULL frame
        (F4 d1_dist_roc_288 / F9 bars_since_d1_regime_change need >=288-bar D1 history; a 96-bar window
        would clip them to 0 = a second-order skew). Fail-closed: raises on a missing source col.
        """
        in_place = cv3 is None
        target = cv3 if cv3 is not None else self._cv3
        if target is None:
            raise RuntimeError("REGIME_V4 augmentation requires a loaded canonical_v3 frame")
        from gx1.features.regime_v4_features import (
            REGIME_V4_DERIVED_COLS as _REGIME_V4_DERIVED,
            REGIME_V4_SOURCE_COLS as _REGIME_V4_SOURCE,
            add_regime_v4_features as _add_rv4,
        )
        if set(_REGIME_V4_DERIVED).issubset(target.columns) and set(
            _REGIME_V4_SOURCE
        ).issubset(target.columns):
            return target
        # Raw BASE28 is M1 market identity only and may never own derived D1
        # state. A v2 pair persists this field on canonical. Retained direct
        # helper callers may derive it from the same full canonical M5 history.
        if "D1_dist_from_ema200_atr" not in target.columns:
            from gx1.execution.v12_ctx_augment_live import _add_htf_features

            required = ("open", "high", "low", "close", "volume")
            missing = [name for name in required if name not in target.columns]
            if missing:
                raise RuntimeError(
                    "[REGIME_V4_SERVE] canonical D1 source unavailable: "
                    f"{missing}"
                )
            _add_htf_features(target, target.loc[:, list(required)])
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
        expected = list(_GROUP_A) + list(_DIP_STRUCT)
        if set(expected).issubset(target.columns):
            return target
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
        from gx1.features.htf_features import build_multi_tf_per_bar_features_v4
        # LIVE serve: build the multi-TF bundle IN-MEMORY from THIS cv3 and pass it, instead
        # of reading the on-disk MULTI_TF_V4_CACHE. The disk cache is only as fresh as its
        # last rebuild — nothing in the live loop advances it, so it eventually trips the
        # 2-day stale guard (live breaks) or, worse, would serve stale HTF context. Building
        # from the live cv3 makes the group-A/dip-struct ctx ALWAYS current and removes the
        # last live disk-cache dependency. Same V4 builder + same bars as the disk cache, and
        # the features are causal/asof → bit-identical at every decision ts (train==serve).
        mtf_in_mem = build_multi_tf_per_bar_features_v4(
            target,
            v29_registry_constants=_require_v29_registry_constants_from_bound_cache(),
        )
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
        if set(self._V1_LEGACY_FEATURES).issubset(cv3.columns):
            return cv3
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
        if set(VOLUME_FEATURE_NAMES).issubset(target.columns):
            return target
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

    def _augment_cv3_with_v4_mtf_scalars(self, cv3: pd.DataFrame | None = None) -> pd.DataFrame | None:
        """Project persistent scalar columns from one exact in-memory V4 cache.
        Returns the augmented DataFrame. Idempotent — if columns are already
        present (e.g. from a rebuilt cv3 parquet), skip.
        """
        target = cv3 if cv3 is not None else self._cv3
        if target is None:
            raise RuntimeError("V4 MTF augmentation requires a loaded canonical_v3 frame")
        from gx1.features.htf_features import (
            build_multi_tf_per_bar_features_v4,
            project_multi_tf_v4_scalars,
        )
        cv3 = target
        expected = self._expected_mtf_scalar_columns()
        if set(expected).issubset(cv3.columns):
            return cv3
        ohlc = ["open", "high", "low", "close"]
        missing_ohlcv = [c for c in (*ohlc, "volume") if c not in cv3.columns]
        if missing_ohlcv:
            raise RuntimeError(f"V4 MTF augmentation missing exact OHLCV: {missing_ohlcv}")
        m5_df = cv3[ohlc].copy()
        for c in ohlc:
            m5_df[c] = m5_df[c].astype(np.float64)
        m5_df["volume"] = cv3["volume"].astype(np.float64)
        LOG.info(f"augmenting canonical_v3 with V4-backed scalar features "
                 f"(31 cols expected) — {len(m5_df):,} M5 rows...")
        cv3_ts_ns = cv3.index.values.astype("datetime64[ns]").astype(np.int64)
        multi_tf = build_multi_tf_per_bar_features_v4(
            m5_df,
            v29_registry_constants=_require_v29_registry_constants_from_bound_cache(),
        )
        projected = project_multi_tf_v4_scalars(
            multi_tf,
            cv3_ts_ns,
            self._MTF_PROJECTION,
            self._MTF_TIMEFRAMES,
            self._MTF_SKIP,
            decision_bar_duration=pd.Timedelta(minutes=5),
        )
        missing = [name for name in expected if name not in projected]
        if missing:
            raise RuntimeError(f"V4 MTF feature owner output incomplete: {missing}")
        for _col, _vals in projected.items():
            cv3[_col] = _vals
        LOG.info(f"  V4 mtf projection done: +{len(projected)} cols  "
                 f"(cv3 now {len(cv3.columns)} cols total)")
        return cv3

    # ── public API ────────────────────────────────────────────────────

    def acquire_serving_snapshot(self) -> PrebuiltServingSnapshot:
        """Capture all frames/caches from one generation under one lock.

        The returned DataFrame/cache objects are no longer mutated by a refresh;
        a later swap installs new objects. Callers that need several inputs for
        one decision must retain and reuse this capsule throughout that decision.
        """

        with self._state_lock:
            self._raise_refresh_error()
            if (
                self._pair_binding is None
                or self._cv3 is None
                or self._base28 is None
                or self._last_ts is None
                or self._pair_binding.generation_manifest_path is None
                or self._base28_file_stamp is None
            ):
                raise RuntimeError("loader not initialized — call .load() first")
            return PrebuiltServingSnapshot(
                pair_generation_id=self._pair_binding.pair_generation_id,
                pair_manifest_sha256=self._pair_binding.manifest_sha256,
                pair_generation_manifest_path=(
                    self._pair_binding.generation_manifest_path
                ),
                base28_path=self._pair_binding.base28.parquet_path,
                base28_sha256=self._pair_binding.base28.parquet_sha256,
                base28_file_stamp=self._base28_file_stamp,
                cv3=self._cv3,
                base28=self._base28,
                cutoff_ts=self._last_ts,
                multi_tf_feats=self._multi_tf_feats,
                multi_tf_shift=self._multi_tf_shift,
                multi_tf_feat_count=self._multi_tf_feat_count,
            )

    def canonical_frame_view(self) -> pd.DataFrame:
        """Return the canonical frame from one captured serving capsule."""

        return self.acquire_serving_snapshot().cv3

    def get_closed_m1_bar(
        self,
        expected_m1: pd.Timestamp,
        *,
        snapshot: PrebuiltServingSnapshot | None = None,
    ) -> dict[str, object]:
        """Return one exact complete BASE28 row from one captured pair."""

        expected = pd.Timestamp(expected_m1)
        if (
            pd.isna(expected)
            or expected.tzinfo is None
            or expected.utcoffset() != pd.Timedelta(0)
            or expected != expected.floor("min")
        ):
            raise PrebuiltIdentityError(
                "PREBUILT_CLOSED_M1_TIMESTAMP_INVALID"
            )
        expected = expected.tz_convert("UTC")
        state = snapshot or self.acquire_serving_snapshot()
        if not isinstance(state, PrebuiltServingSnapshot):
            raise TypeError("snapshot must be a PrebuiltServingSnapshot")
        generation = read_prebuilt_pair_manifest(
            state.pair_generation_manifest_path,
            generation_root=self.generation_root,
        )
        if (
            generation.pair_generation_id != state.pair_generation_id
            or generation.manifest_sha256 != state.pair_manifest_sha256
            or generation.base28.parquet_path != state.base28_path
            or generation.base28.parquet_sha256 != state.base28_sha256
        ):
            raise PrebuiltIdentityError(
                "PREBUILT_CLOSED_M1_SNAPSHOT_IDENTITY_CHANGED"
            )
        if (
            _file_stamp(state.base28_path, label="BASE28")
            != state.base28_file_stamp
        ):
            raise PrebuiltIdentityError(
                "PREBUILT_CLOSED_M1_SOURCE_CHANGED"
            )
        position = int(state.base28.index.get_indexer([expected])[0])
        if position < 0:
            raise PrebuiltIdentityError(
                "PREBUILT_CLOSED_M1_EXACT_BAR_MISSING: "
                f"{expected.isoformat()}"
            )
        row = state.base28.iloc[position]
        return canonical_closed_m1_bar(
            m1_bar_ts=expected,
            complete=True,
            source_path=str(state.base28_path),
            source_sha256=state.base28_sha256,
            bid_open=float(row["bid_open"]),
            bid_high=float(row["bid_high"]),
            bid_low=float(row["bid_low"]),
            bid_close=float(row["bid_close"]),
            ask_open=float(row["ask_open"]),
            ask_high=float(row["ask_high"]),
            ask_low=float(row["ask_low"]),
            ask_close=float(row["ask_close"]),
            mid_open=float(row["open"]),
            mid_high=float(row["high"]),
            mid_low=float(row["low"]),
            mid_close=float(row["close"]),
            volume=int(row["volume"]),
        )

    @property
    def cutoff_ts(self) -> pd.Timestamp:
        """Latest M5 bar timestamp available in BOTH prebuilts (joint coverage)."""
        return self.acquire_serving_snapshot().cutoff_ts

    @property
    def pair_generation_id(self) -> str:
        """Content identity shared by the two currently loaded artifacts."""
        return self.acquire_serving_snapshot().pair_generation_id

    def get_window(
        self,
        end_ts: pd.Timestamp,
        n_bars: int = 96,
        *,
        snapshot: PrebuiltServingSnapshot | None = None,
    ) -> pd.DataFrame:
        """Return the last `n_bars` M5 bars up to and including `end_ts`,
        joined from canonical_v3 + BASE28 prebuilts.

        Latest row is the decision bar (at end_ts.floor('5min')).
        Empty DataFrame if `end_ts` is past the prebuilt coverage.
        """
        state = snapshot or self.acquire_serving_snapshot()
        if not isinstance(state, PrebuiltServingSnapshot):
            raise TypeError("snapshot must be a PrebuiltServingSnapshot")

        end_bucket = end_ts.floor("5min")
        if end_bucket > state.cutoff_ts:
            LOG.warning(f"decision_ts {end_bucket} is past prebuilt cutoff {state.cutoff_ts} — "
                         f"run canonical rebuild cron")
            return pd.DataFrame()

        cv3_win = state.cv3.loc[:end_bucket].tail(n_bars)
        if cv3_win.empty:
            return pd.DataFrame()

        # Join BASE28 augmented columns at exact timestamps. Missing timestamps
        # are unavailable evidence; they may not be reconstructed from a later
        # M1 row inside the bucket.
        b28_cols = [c for c in state.base28.columns if c not in cv3_win.columns]
        if b28_cols:
            missing_index = cv3_win.index[~cv3_win.index.isin(state.base28.index)]
            if len(missing_index):
                raise RuntimeError(
                    "BASE28 exact timestamp contract missing rows: "
                    f"count={len(missing_index)} first={missing_index[0]}"
                )
            b28_slice = state.base28.loc[cv3_win.index, b28_cols]
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
        with self._state_lock:
            self._raise_refresh_error()
            source = cv3 if cv3 is not None else self._cv3
            source_pair = self._pair_binding
        if source is None:
            raise RuntimeError("call .load() before .build_multi_tf_features()")
        from gx1.features.htf_features import (
            MULTI_TF_FEATURE_COUNT_V4,
            MULTI_TF_SHIFT,
            build_multi_tf_per_bar_features_v4,
        )
        ohlc_cols = ["open", "high", "low", "close"]
        missing = [c for c in (*ohlc_cols, "volume") if c not in source.columns]
        if missing:
            raise RuntimeError(f"canonical_v3 missing exact OHLCV cols: {missing}")
        m5_df = source[ohlc_cols].copy()
        m5_df["volume"] = source["volume"].astype(np.float64)
        LOG.info(f"building multi-TF V4 features (M5/M15/H1/H4/D1, {MULTI_TF_FEATURE_COUNT_V4} dim each) "
                 f"from {len(m5_df):,} M5 bars...")
        feats_dict = build_multi_tf_per_bar_features_v4(
            m5_df,
            v29_registry_constants=_require_v29_registry_constants_from_bound_cache(),
        )
        feat_count = MULTI_TF_FEATURE_COUNT_V4
        for tf, feats in feats_dict.items():
            if len(feats) > 0:
                feat_count = int(feats.shape[1])
                break
        for tf, feats in feats_dict.items():
            LOG.info(f"  multi-TF V4 {tf}: {len(feats):,} bars × {feats.shape[1]} feats")
        result = {
            "feats": feats_dict,
            "shift": MULTI_TF_SHIFT,
            "feat_count": feat_count,
        }
        # Backward-compat: legacy callers (load() and direct user calls without
        # cv3 arg) expect side-effect on self.*. When cv3 IS provided the caller
        # is responsible for the atomic swap.
        if cv3 is None:
            with self._state_lock:
                if self._cv3 is not source or self._pair_binding != source_pair:
                    raise PrebuiltIdentityError(
                        "PREBUILT_STATE_CHANGED_DURING_MULTI_TF_BUILD"
                    )
                self._multi_tf_feats = result["feats"]
                self._multi_tf_shift = result["shift"]
                self._multi_tf_feat_count = result["feat_count"]
        return result
