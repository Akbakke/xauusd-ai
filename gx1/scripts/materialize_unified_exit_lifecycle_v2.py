#!/usr/bin/env python3
"""Adopt immutable Entry TRAIN/VAL data into a compact lifecycle-v2 bundle.

The producer reads only Entry clocks, the authoritative M1 clock and an
explicit economic-lifecycle authority.  It stores one row per Entry pair and
hash-binds every derived chunk/successor pointer without expanding states or
storing model-dependent target Q values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gx1.contracts.entry_exit_feature_base_v1 import (
    ENTRY_DECISION_BAR_SECONDS,
    EXIT_DECISION_BAR_SECONDS,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION,
)
from gx1.contracts.unified_exit_lifecycle_v1 import (
    UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION,
    UnifiedExitLifecycleCorpus,
)
from gx1.contracts.unified_exit_lifecycle_v2 import (
    UNIFIED_EXIT_CHUNK_ROWS,
    UNIFIED_EXIT_ECONOMIC_AUTHORITY_SCHEMA_VERSION,
    require_unified_exit_economic_lifecycle_authority,
    terminal_state_counts_sha256,
    unified_exit_lifecycle_v2_contract,
)


COMPACT_LIFECYCLE_SCHEMA_VERSION = "gx1_unified_exit_compact_lifecycle_v2"
COMPACT_ROOT_SCHEMA_VERSION = "gx1_unified_exit_compact_train_val_bundle_v1"
ECONOMIC_COUNTS_SCHEMA_VERSION = "gx1_unified_exit_economic_terminal_counts_v1"
PAIR_SCHEDULE_SCHEMA_VERSION = "gx1_unified_exit_pair_chunk_schedule_v1"
PAIR_COVERAGE_SCHEMA_VERSION = "gx1_unified_exit_pair_chunk_coverage_v1"
PRODUCER_SOURCE_SCHEMA_VERSION = "gx1_unified_exit_compact_producer_source_v1"
ADOPTION_WITNESS_SCHEMA_VERSION = "gx1_unified_exit_v1_adoption_witness_v1"
COMPACT_COLUMNS = (
    "schema_version",
    "entry_row_index",
    "entry_time",
    "entry_m1_start_row",
    "first_state_row_time",
    "available_state_count",
    "long_lifecycle_state_count",
    "short_lifecycle_state_count",
    "long_chunk_count",
    "short_chunk_count",
    "pair_chunk_count",
    "long_economic_terminal",
    "short_economic_terminal",
    "long_right_censored",
    "short_right_censored",
    "long_last_state_m1_row",
    "short_last_state_m1_row",
    "long_terminal_state_m1_row",
    "short_terminal_state_m1_row",
    "long_chunk_pointer_stream_sha256",
    "short_chunk_pointer_stream_sha256",
    "entry_binding_sha256",
    "row_identity_sha256",
    "gap_classification",
    "gap_seconds",
    "gap_after_m1_row",
    "gap_binding_sha256",
    "m1_clock_sha256",
    "m1_source_sha256",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _producer_source_identity() -> dict[str, Any]:
    source = Path(__file__).resolve()
    return {
        "schema_version": PRODUCER_SOURCE_SCHEMA_VERSION,
        "module": "gx1.scripts.materialize_unified_exit_lifecycle_v2",
        "source_sha256": _sha256_file(source),
    }


def _require_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"COMPACT_LIFECYCLE_{label}_SHA_INVALID")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise RuntimeError(f"COMPACT_LIFECYCLE_{label}_FILE_INVALID")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"COMPACT_LIFECYCLE_{label}_JSON_INVALID") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"COMPACT_LIFECYCLE_{label}_JSON_INVALID")
    return value


class _EconomicAuthorityBlocked(RuntimeError):
    pass


def _require_full_v1_admission(
    *,
    entry_paths: Mapping[str, Path],
    entry_manifest_paths: Mapping[str, Path],
    dataset_run_id: str,
) -> dict[str, Any]:
    declared_roots: set[Path] = set()
    bindings: dict[str, dict[str, str]] = {}
    for split in ("train", "val"):
        manifest_path = entry_manifest_paths[split]
        manifest = _read_json(manifest_path, f"{split.upper()}_ENTRY_MANIFEST")
        extra = manifest.get("extra")
        lifecycle = extra.get("unified_exit_lifecycle") if isinstance(extra, Mapping) else None
        root_dir = Path(
            str(lifecycle.get("output_dir") or "")
            if isinstance(lifecycle, Mapping)
            else ""
        )
        declared_roots.add(root_dir / "UNIFIED_EXIT_LIFECYCLE_MANIFEST.json")
        bindings[split] = {
            "path": str(manifest_path),
            "sha256": _sha256_file(manifest_path),
        }
    if len(declared_roots) != 1:
        raise RuntimeError("COMPACT_LIFECYCLE_V1_ROOT_IDENTITY_INVALID")
    root_path = next(iter(declared_roots))
    try:
        return UnifiedExitLifecycleCorpus._require_file_admission(
            root_manifest_path=root_path,
            entry_parquets=entry_paths,
            entry_manifest_bindings=bindings,
            dataset_run_id=dataset_run_id,
            splits=("train", "val"),
        )
    except Exception as exc:
        raise RuntimeError("COMPACT_LIFECYCLE_FULL_V1_ADOPTION_INVALID") from exc


def pair_chunk_permutation(
    *,
    chunk_count: int,
    lineage_sha256: str,
    split: str,
    entry_row_index: int,
) -> tuple[int, ...]:
    """Return the deterministic pair timeline; no side/outcome enters its API."""

    lineage = _require_sha(lineage_sha256, "SCHEDULE_LINEAGE")
    if (
        isinstance(chunk_count, bool)
        or not isinstance(chunk_count, int)
        or chunk_count < 1
        or split not in {"train", "val"}
        or isinstance(entry_row_index, bool)
        or not isinstance(entry_row_index, int)
        or entry_row_index < 0
    ):
        raise RuntimeError("COMPACT_LIFECYCLE_SCHEDULE_IDENTITY_INVALID")
    seed = bytes.fromhex(
        _canonical_sha256(
            {
                "schema_version": PAIR_SCHEDULE_SCHEMA_VERSION,
                "lineage_sha256": lineage,
                "split": split,
                "entry_row_index": entry_row_index,
            }
        )
    )
    multiplier = int.from_bytes(seed[:16], "big") % chunk_count
    while math.gcd(multiplier, chunk_count) != 1:
        multiplier = (multiplier + 1) % chunk_count
    offset = int.from_bytes(seed[16:], "big") % chunk_count
    return tuple(
        (multiplier * position + offset) % chunk_count
        for position in range(chunk_count)
    )


def pair_chunk_for_epoch(
    *,
    chunk_count: int,
    epoch_index: int,
    lineage_sha256: str,
    split: str,
    entry_row_index: int,
) -> int:
    if (
        isinstance(epoch_index, bool)
        or not isinstance(epoch_index, int)
        or epoch_index < 0
    ):
        raise RuntimeError("COMPACT_LIFECYCLE_SCHEDULE_EPOCH_INVALID")
    order = pair_chunk_permutation(
        chunk_count=chunk_count,
        lineage_sha256=lineage_sha256,
        split=split,
        entry_row_index=entry_row_index,
    )
    return order[epoch_index % chunk_count]


def pair_schedule_coverage(
    *,
    chunk_count_by_entry: Mapping[int, int],
    planned_epochs: int,
    lineage_sha256: str,
    split: str,
    claim_full_coverage: bool,
) -> dict[str, Any]:
    """Bind a resumable coverage claim and fail closed if it is false."""

    lineage = _require_sha(lineage_sha256, "SCHEDULE_LINEAGE")
    if (
        not isinstance(chunk_count_by_entry, Mapping)
        or not chunk_count_by_entry
        or isinstance(planned_epochs, bool)
        or not isinstance(planned_epochs, int)
        or planned_epochs < 1
        or split not in {"train", "val"}
        or type(claim_full_coverage) is not bool
    ):
        raise RuntimeError("COMPACT_LIFECYCLE_COVERAGE_INPUT_INVALID")
    normalized: dict[int, int] = {}
    for entry_row, chunk_count in chunk_count_by_entry.items():
        if (
            isinstance(entry_row, bool)
            or not isinstance(entry_row, int)
            or entry_row < 0
            or isinstance(chunk_count, bool)
            or not isinstance(chunk_count, int)
            or chunk_count < 1
        ):
            raise RuntimeError("COMPACT_LIFECYCLE_COVERAGE_INPUT_INVALID")
        normalized[entry_row] = chunk_count
    total = int(sum(normalized.values()))
    covered = int(sum(min(planned_epochs, count) for count in normalized.values()))
    full_rows = int(sum(count <= planned_epochs for count in normalized.values()))
    full = covered == total
    if claim_full_coverage and not full:
        raise RuntimeError("COMPACT_LIFECYCLE_FALSE_FULL_COVERAGE_CLAIM")
    payload = {
        "schema_version": PAIR_COVERAGE_SCHEMA_VERSION,
        "schedule_schema_version": PAIR_SCHEDULE_SCHEMA_VERSION,
        "lineage_sha256": lineage,
        "split": split,
        "planned_epochs": planned_epochs,
        "resume_cursor": {"next_epoch_index": planned_epochs},
        "entry_rows": len(normalized),
        "total_chunks": total,
        "covered_unique_chunks": covered,
        "coverage_fraction": covered / total,
        "fully_covered_entry_rows": full_rows,
        "not_fully_covered_entry_rows": len(normalized) - full_rows,
        "max_pair_chunk_count": max(normalized.values()),
        "full_coverage": full,
        "full_coverage_claimed": claim_full_coverage,
        "both_sides_share_timeline": True,
        "selection_uses_outcome_values": False,
    }
    payload["contract_sha256"] = _canonical_sha256(payload)
    return payload


def scheduled_pair_chunk_pointer(
    *,
    compact_row: Mapping[str, Any],
    epoch_index: int,
    lineage_sha256: str,
    split: str,
) -> dict[str, Any]:
    """Derive one common LONG/SHORT chunk slot and its static successors."""

    if not isinstance(compact_row, Mapping):
        raise RuntimeError("COMPACT_LIFECYCLE_SCHEDULE_ROW_INVALID")
    if compact_row.get("schema_version") != COMPACT_LIFECYCLE_SCHEMA_VERSION:
        raise RuntimeError("COMPACT_LIFECYCLE_SCHEDULE_ROW_INVALID")
    entry_row = compact_row.get("entry_row_index")
    pair_count = compact_row.get("pair_chunk_count")
    slot = pair_chunk_for_epoch(
        chunk_count=pair_count,
        epoch_index=epoch_index,
        lineage_sha256=lineage_sha256,
        split=split,
        entry_row_index=entry_row,
    )
    chunk_start = slot * UNIFIED_EXIT_CHUNK_ROWS
    sides: dict[str, Any] = {}
    for side in ("long", "short"):
        state_count = int(compact_row[f"{side}_lifecycle_state_count"])
        active = chunk_start < state_count
        valid_count = min(UNIFIED_EXIT_CHUNK_ROWS, state_count - chunk_start) if active else 0
        after = chunk_start + valid_count
        successor = active and after < state_count
        terminal = (
            active
            and not successor
            and bool(compact_row[f"{side}_economic_terminal"])
        )
        sides[side] = {
            "active": active,
            "valid_state_count": valid_count,
            "successor_available": successor,
            "successor_m1_row": (
                int(compact_row["entry_m1_start_row"]) + after
                if successor
                else -1
            ),
            "right_censored": (
                active
                and not successor
                and bool(compact_row[f"{side}_right_censored"])
            ),
            "terminal_reason": "economic_terminal" if terminal else "none",
            "pointer_stream_sha256": compact_row[
                f"{side}_chunk_pointer_stream_sha256"
            ],
        }
    payload = {
        "schema_version": PAIR_SCHEDULE_SCHEMA_VERSION,
        "split": split,
        "lineage_sha256": lineage_sha256,
        "entry_row_index": entry_row,
        "epoch_index": epoch_index,
        "resume_cursor": {"next_epoch_index": epoch_index + 1},
        "pair_chunk_slot": slot,
        "chunk_start_bars_in_trade": chunk_start,
        "chunk_m1_start_row": int(compact_row["entry_m1_start_row"]) + chunk_start,
        "gap_classification": compact_row["gap_classification"],
        "gap_binding_sha256": compact_row["gap_binding_sha256"],
        "both_sides_share_timeline": True,
        "selection_uses_outcome_values": False,
        "sides": sides,
    }
    payload["schedule_sha256"] = _canonical_sha256(payload)
    return payload


def _chunk_pointer_stream_sha256(
    *,
    entry_row_index: int,
    side_index: int,
    entry_m1_start_row: int,
    state_count: int,
    economic_terminal: bool,
    gap_binding_sha256: str,
    m1_clock_sha256: str,
    m1_source_sha256: str,
) -> str:
    chunk_count = int(math.ceil(state_count / UNIFIED_EXIT_CHUNK_ROWS))
    return _canonical_sha256(
        {
            "schema_version": "gx1_compact_successor_pointer_formula_v1",
            "entry_row_index": entry_row_index,
            "side_index": side_index,
            "entry_m1_start_row": entry_m1_start_row,
            "state_count": state_count,
            "chunk_count": chunk_count,
            "chunk_state_capacity": UNIFIED_EXIT_CHUNK_ROWS,
            "chunk_start_formula": "entry_m1_start_row+chunk_slot*chunk_state_capacity",
            "successor_formula": "chunk_end_lt_state_count_then_entry_m1_start_row+chunk_end",
            "economic_terminal": economic_terminal,
            "right_censor_at_last_state": not economic_terminal,
            "gap_binding_sha256": _require_sha(
                gap_binding_sha256, "GAP_BINDING"
            ),
            "m1_clock_sha256": _require_sha(m1_clock_sha256, "M1_CLOCK"),
            "m1_source_sha256": m1_source_sha256,
        }
    )


def _m1_clock_sha256(clock_ns: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(clock_ns, dtype="<i8").tobytes()
    ).hexdigest()


def _row_identity_sha256(value: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {
            "schema_version": COMPACT_LIFECYCLE_SCHEMA_VERSION,
            "entry_row_index": int(value["entry_row_index"]),
            "entry_time_ns": int(pd.Timestamp(value["entry_time"]).value),
            "entry_m1_start_row": int(value["entry_m1_start_row"]),
            "first_state_row_time_ns": int(
                pd.Timestamp(value["first_state_row_time"]).value
            ),
            "entry_binding_sha256": value["entry_binding_sha256"],
            "m1_source_sha256": value["m1_source_sha256"],
            "m1_clock_sha256": value["m1_clock_sha256"],
            "gap_binding_sha256": value["gap_binding_sha256"],
        }
    )


def _compact_pointer_stream_sha256(frame: pd.DataFrame) -> str:
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        raw = row._asdict()
        item: dict[str, Any] = {}
        for name in COMPACT_COLUMNS:
            value = raw[name]
            if name in ("entry_time", "first_state_row_time"):
                item[name] = pd.Timestamp(value).isoformat()
            elif isinstance(value, (np.integer, int)) and not isinstance(
                value, (np.bool_, bool)
            ):
                item[name] = int(value)
            elif isinstance(value, (np.bool_, bool)):
                item[name] = bool(value)
            else:
                item[name] = value
        rows.append(item)
    return _canonical_sha256(
        {
            "schema_version": COMPACT_LIFECYCLE_SCHEMA_VERSION,
            "rows": rows,
        }
    )


def _load_terminal_counts(
    authority_path: Path,
    *,
    split: str,
    dataset_run_id: str,
    entry_rows: int,
) -> tuple[dict[tuple[int, int], int | None], dict[str, Any], str]:
    authority = _read_json(authority_path, f"{split.upper()}_ECONOMIC_AUTHORITY")
    if authority.get("schema_version") != UNIFIED_EXIT_ECONOMIC_AUTHORITY_SCHEMA_VERSION:
        raise RuntimeError("COMPACT_LIFECYCLE_ECONOMIC_AUTHORITY_SCHEMA_INVALID")
    counts_path = Path(str(authority.get("authority_artifact_path") or ""))
    counts = _read_json(counts_path, f"{split.upper()}_ECONOMIC_COUNTS")
    counts_sha = _sha256_file(counts_path)
    raw = counts.get("terminal_state_counts")
    if (
        counts.get("schema_version") != ECONOMIC_COUNTS_SCHEMA_VERSION
        or counts.get("decision") != "PASS"
        or counts.get("dataset_run_id") != dataset_run_id
        or counts.get("split") != split
        or counts.get("entry_rows") != entry_rows
        or counts.get("test_data_used") is not False
        or not isinstance(raw, list)
        or len(raw) != entry_rows * 2
        or authority.get("authority_artifact_sha256") != counts_sha
    ):
        raise RuntimeError("COMPACT_LIFECYCLE_ECONOMIC_COUNTS_INVALID")
    mapping: dict[tuple[int, int], int | None] = {}
    for expected_position, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise RuntimeError("COMPACT_LIFECYCLE_ECONOMIC_COUNTS_INVALID")
        entry_row = expected_position // 2
        side_index = expected_position % 2
        state_count = item.get("terminal_state_count")
        if (
            set(item) != {"entry_row_index", "side_index", "terminal_state_count"}
            or item.get("entry_row_index") != entry_row
            or item.get("side_index") != side_index
            or (
                state_count is not None
                and (
                    isinstance(state_count, bool)
                    or not isinstance(state_count, int)
                    or state_count < 1
                )
            )
        ):
            raise RuntimeError("COMPACT_LIFECYCLE_ECONOMIC_COUNTS_INVALID")
        mapping[(entry_row, side_index)] = state_count
    require_unified_exit_economic_lifecycle_authority(
        authority,
        expected_terminal_state_counts_sha256=terminal_state_counts_sha256(mapping),
    )
    raise _EconomicAuthorityBlocked(
        "COMPACT_LIFECYCLE_ECONOMIC_TERMINAL_VERIFIER_UNAVAILABLE"
    )


def _validate_entry_manifest(
    manifest_path: Path,
    *,
    entry_path: Path,
    split: str,
    dataset_run_id: str,
    full_v1_admission: Mapping[str, Any],
) -> tuple[dict[str, Any], int, dict[str, Any]]:
    manifest = _read_json(manifest_path, f"{split.upper()}_ENTRY_MANIFEST")
    extra = manifest.get("extra")
    guard = extra.get("pretest_test_guard") if isinstance(extra, Mapping) else None
    if (
        not isinstance(extra, Mapping)
        or extra.get("entry_run_id") != dataset_run_id
        or extra.get("pretest_only") is not True
        or not isinstance(guard, Mapping)
        or guard.get("test_accessed") is not False
    ):
        raise RuntimeError("COMPACT_LIFECYCLE_ENTRY_PRETEST_BINDING_INVALID")
    entry_sha = _sha256_file(entry_path)
    try:
        parquet = pq.ParquetFile(entry_path)
        entry_rows = int(parquet.metadata.num_rows)
        parquet_columns = set(parquet.schema_arrow.names)
    except Exception as exc:
        raise RuntimeError("COMPACT_LIFECYCLE_ENTRY_PARQUET_INVALID") from exc
    lifecycle = extra.get("unified_exit_lifecycle")
    lifecycle_dir = Path(
        str(lifecycle.get("output_dir") or "")
        if isinstance(lifecycle, Mapping)
        else ""
    )
    lifecycle_root_path = lifecycle_dir / "UNIFIED_EXIT_LIFECYCLE_MANIFEST.json"
    if (
        manifest.get("schema_version")
        != MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION
        or manifest.get("manifest_variant") != MODEL_NATIVE_CONTRACT_MODE
        or manifest.get("output_data_path") != str(entry_path)
        or isinstance(extra.get("rows"), bool)
        or extra.get("rows") != entry_rows
        or entry_rows < 1
        or "time" not in parquet_columns
        or not isinstance(lifecycle, Mapping)
        or lifecycle.get("schema_version")
        != UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION
        or not lifecycle_root_path.is_absolute()
        or lifecycle_root_path.resolve() != lifecycle_root_path
        or lifecycle_root_path.is_symlink()
        or not lifecycle_root_path.is_file()
    ):
        raise RuntimeError("COMPACT_LIFECYCLE_ENTRY_ARTIFACT_BINDING_INVALID")
    lifecycle_root = _read_json(
        lifecycle_root_path,
        f"{split.upper()}_ENTRY_LIFECYCLE_ROOT",
    )
    split_bindings = lifecycle_root.get("splits")
    binding = split_bindings.get(split) if isinstance(split_bindings, Mapping) else None
    admitted_splits = full_v1_admission.get("splits")
    admitted_split = (
        admitted_splits.get(split) if isinstance(admitted_splits, Mapping) else None
    )
    entry_windows = full_v1_admission.get("entry_windows")
    admitted_window = (
        entry_windows.get(split) if isinstance(entry_windows, Mapping) else None
    )
    manifest_sha = _sha256_file(manifest_path)
    root_sha = _sha256_file(lifecycle_root_path)
    if (
        lifecycle_root.get("schema_version")
        != UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION
        or lifecycle_root.get("decision") != "PASS"
        or lifecycle_root.get("entry_run_id") != dataset_run_id
        or not isinstance(binding, Mapping)
        or binding.get("entry_dataset_path") != str(entry_path)
        or binding.get("entry_dataset_sha256") != entry_sha
        or isinstance(binding.get("episode_rows"), bool)
        or binding.get("episode_rows") != entry_rows * 2
        or full_v1_admission.get("root_manifest_path") != lifecycle_root_path
        or full_v1_admission.get("root_manifest_sha256") != root_sha
        or not isinstance(admitted_split, Mapping)
        or admitted_split.get("entry_path") != entry_path
        or not isinstance(admitted_window, Mapping)
        or admitted_window.get("manifest_sha256") != manifest_sha
    ):
        raise RuntimeError("COMPACT_LIFECYCLE_ENTRY_ARTIFACT_BINDING_INVALID")
    witness = {
        "schema_version": ADOPTION_WITNESS_SCHEMA_VERSION,
        "split": split,
        "dataset_run_id": dataset_run_id,
        "entry_parquet_path": str(entry_path),
        "entry_parquet_sha256": entry_sha,
        "entry_manifest_path": str(manifest_path),
        "entry_manifest_sha256": manifest_sha,
        "lifecycle_root_path": str(lifecycle_root_path),
        "lifecycle_root_sha256": root_sha,
        "full_v1_admission_verified": True,
        "test_accessed": False,
    }
    witness["witness_sha256"] = _canonical_sha256(witness)
    return manifest, entry_rows, witness


def _validate_m1_source(
    source_path: Path, manifest_path: Path
) -> tuple[pd.DatetimeIndex, dict[str, Any], str, str]:
    manifest = _read_json(manifest_path, "M1_SOURCE_MANIFEST")
    source_sha = _sha256_file(source_path)
    if (
        manifest.get("timeframe") != "M1"
        or manifest.get("quote_complete_m1") is not True
        or manifest.get("test_accessed") is not False
        or manifest.get("output_parquet") != str(source_path)
        or manifest.get("output_parquet_sha256") != source_sha
    ):
        raise RuntimeError("COMPACT_LIFECYCLE_M1_SOURCE_INVALID")
    times = pd.DatetimeIndex(
        pd.to_datetime(pd.read_parquet(source_path, columns=["time"])["time"], utc=True)
    ).as_unit("ns")
    if (
        times.empty
        or times.hasnans
        or not times.is_unique
        or not times.is_monotonic_increasing
        or not times.floor("min").equals(times)
        or manifest.get("row_count") != len(times)
    ):
        raise RuntimeError("COMPACT_LIFECYCLE_M1_CLOCK_INVALID")
    return times, manifest, _sha256_file(manifest_path), source_sha


def build_compact_split(
    *,
    entry_times: Sequence[Any],
    m1_times: Sequence[Any],
    split: str,
    split_end: Any,
    terminal_state_count_by_entry_side: Mapping[tuple[int, int], int | None],
    m1_source_sha256: str,
    gap_classification_source_sha256: str,
    entry_binding_sha256: str,
) -> pd.DataFrame:
    """Build one compact row per Entry pair and bind all successor pointers."""

    if split not in {"train", "val"}:
        raise RuntimeError("COMPACT_LIFECYCLE_SPLIT_FORBIDDEN")
    source_sha = _require_sha(m1_source_sha256, "M1_SOURCE")
    gap_source_sha = _require_sha(
        gap_classification_source_sha256, "GAP_CLASSIFICATION_SOURCE"
    )
    entry_binding = _require_sha(entry_binding_sha256, "ENTRY_BINDING")
    entries = pd.DatetimeIndex(pd.to_datetime(entry_times, utc=True)).as_unit("ns")
    clock = pd.DatetimeIndex(pd.to_datetime(m1_times, utc=True)).as_unit("ns")
    if (
        entries.empty
        or entries.hasnans
        or not entries.is_unique
        or not entries.is_monotonic_increasing
        or clock.empty
        or clock.hasnans
        or not clock.is_unique
        or not clock.is_monotonic_increasing
        or not entries.floor(f"{ENTRY_DECISION_BAR_SECONDS}s").equals(entries)
        or not clock.floor(f"{EXIT_DECISION_BAR_SECONDS}s").equals(clock)
    ):
        raise RuntimeError("COMPACT_LIFECYCLE_CLOCK_INVALID")
    end = pd.Timestamp(split_end)
    if pd.isna(end) or end.tz is None or end.utcoffset() != pd.Timedelta(0):
        raise RuntimeError("COMPACT_LIFECYCLE_SPLIT_END_INVALID")
    end = end.as_unit("ns")
    expected = {(row, side) for row in range(len(entries)) for side in (0, 1)}
    if set(terminal_state_count_by_entry_side) != expected:
        raise RuntimeError("COMPACT_LIFECYCLE_TERMINAL_POPULATION_INVALID")
    clock_ns = np.asarray(clock.asi8, dtype=np.int64)
    clock_sha = _m1_clock_sha256(clock_ns)
    available_ns = np.asarray(
        entries.asi8 + int(pd.Timedelta(seconds=ENTRY_DECISION_BAR_SECONDS).value),
        dtype=np.int64,
    )
    starts = np.searchsorted(clock_ns, available_ns, side="left")
    exact = starts < len(clock_ns)
    positions = np.flatnonzero(exact)
    exact[positions] &= clock_ns[starts[positions]] == available_ns[positions]
    if not exact.all():
        raise RuntimeError("COMPACT_LIFECYCLE_ENTRY_M1_OPEN_MISSING")
    delta_ns = int(pd.Timedelta(seconds=EXIT_DECISION_BAR_SECONDS).value)
    available_stop = int(np.searchsorted(clock_ns, end.value - delta_ns, side="right"))
    gap_after_rows = np.flatnonzero(np.diff(clock_ns) != delta_ns)
    rows: list[dict[str, Any]] = []
    for entry_row, start_value in enumerate(starts.tolist()):
        start = int(start_value)
        gap_position = int(np.searchsorted(gap_after_rows, start, side="left"))
        gap_after = (
            int(gap_after_rows[gap_position])
            if gap_position < len(gap_after_rows)
            and int(gap_after_rows[gap_position]) < available_stop - 1
            else -1
        )
        contiguous_stop = min(
            available_stop,
            gap_after + 1 if gap_after >= 0 else available_stop,
        )
        available = contiguous_stop - start
        if available < 1:
            raise RuntimeError("COMPACT_LIFECYCLE_NO_STATE_BEFORE_SPLIT_END")
        if gap_after >= 0:
            gap_seconds = int((clock_ns[gap_after + 1] - clock_ns[gap_after]) // 1_000_000_000)
            before = clock[gap_after]
            after = clock[gap_after + 1]
            gap_classification = (
                "weekend_source_absence"
                if gap_seconds >= 24 * 60 * 60
                and (before.weekday() == 4 or after.weekday() in (6, 0))
                else "unknown_source_absence"
            )
        else:
            gap_seconds = 0
            gap_classification = "split_end"
        gap_binding_sha = _canonical_sha256(
            {
                "schema_version": "gx1_m1_gap_censor_binding_v1",
                "entry_row_index": entry_row,
                "classification_source_sha256": gap_source_sha,
                "classification": gap_classification,
                "gap_after_m1_row": gap_after,
                "gap_seconds": gap_seconds,
                "successor_across_gap_allowed": False,
            }
        )
        side_values: list[dict[str, Any]] = []
        for side_index in (0, 1):
            terminal_count = terminal_state_count_by_entry_side[(entry_row, side_index)]
            if terminal_count is not None and terminal_count > available:
                raise RuntimeError("COMPACT_LIFECYCLE_TERMINAL_OUTSIDE_SPLIT")
            state_count = terminal_count if terminal_count is not None else available
            chunk_count = int(math.ceil(state_count / UNIFIED_EXIT_CHUNK_ROWS))
            side_values.append(
                {
                    "state_count": state_count,
                    "chunk_count": chunk_count,
                    "economic_terminal": terminal_count is not None,
                    "right_censored": terminal_count is None,
                    "last_row": start + state_count - 1,
                    "terminal_row": start + state_count - 1 if terminal_count is not None else -1,
                    "pointer_sha": _chunk_pointer_stream_sha256(
                        entry_row_index=entry_row,
                        side_index=side_index,
                        entry_m1_start_row=start,
                        state_count=state_count,
                        economic_terminal=terminal_count is not None,
                        gap_binding_sha256=gap_binding_sha,
                        m1_clock_sha256=clock_sha,
                        m1_source_sha256=source_sha,
                    ),
                }
            )
        long, short = side_values
        row = {
                "schema_version": COMPACT_LIFECYCLE_SCHEMA_VERSION,
                "entry_row_index": entry_row,
                "entry_time": entries[entry_row],
                "entry_m1_start_row": start,
                "first_state_row_time": clock[start],
                "available_state_count": available,
                "long_lifecycle_state_count": long["state_count"],
                "short_lifecycle_state_count": short["state_count"],
                "long_chunk_count": long["chunk_count"],
                "short_chunk_count": short["chunk_count"],
                "pair_chunk_count": max(long["chunk_count"], short["chunk_count"]),
                "long_economic_terminal": long["economic_terminal"],
                "short_economic_terminal": short["economic_terminal"],
                "long_right_censored": long["right_censored"],
                "short_right_censored": short["right_censored"],
                "long_last_state_m1_row": long["last_row"],
                "short_last_state_m1_row": short["last_row"],
                "long_terminal_state_m1_row": long["terminal_row"],
                "short_terminal_state_m1_row": short["terminal_row"],
                "long_chunk_pointer_stream_sha256": long["pointer_sha"],
                "short_chunk_pointer_stream_sha256": short["pointer_sha"],
                "entry_binding_sha256": entry_binding,
                "row_identity_sha256": "",
                "gap_classification": gap_classification,
                "gap_seconds": gap_seconds,
                "gap_after_m1_row": gap_after,
                "gap_binding_sha256": gap_binding_sha,
                "m1_clock_sha256": clock_sha,
                "m1_source_sha256": source_sha,
            }
        row["row_identity_sha256"] = _row_identity_sha256(row)
        rows.append(row)
    frame = pd.DataFrame(rows, columns=COMPACT_COLUMNS)
    require_compact_split(
        frame,
        split_end=end,
        m1_times=clock,
        expected_m1_source_sha256=source_sha,
        expected_entry_binding_sha256=entry_binding,
        expected_gap_classification_source_sha256=gap_source_sha,
    )
    return frame


def require_compact_split(
    frame: pd.DataFrame,
    *,
    split_end: Any,
    m1_times: Sequence[Any],
    expected_m1_source_sha256: str,
    expected_entry_binding_sha256: str,
    expected_gap_classification_source_sha256: str,
) -> pd.DataFrame:
    source_sha = _require_sha(expected_m1_source_sha256, "EXPECTED_M1_SOURCE")
    entry_binding = _require_sha(
        expected_entry_binding_sha256, "EXPECTED_ENTRY_BINDING"
    )
    gap_source_sha = _require_sha(
        expected_gap_classification_source_sha256,
        "EXPECTED_GAP_CLASSIFICATION_SOURCE",
    )
    if (
        not isinstance(frame, pd.DataFrame)
        or tuple(frame.columns) != COMPACT_COLUMNS
        or frame.empty
        or frame["schema_version"].ne(COMPACT_LIFECYCLE_SCHEMA_VERSION).any()
        or frame["entry_row_index"].tolist() != list(range(len(frame)))
        or frame["m1_source_sha256"].ne(source_sha).any()
        or frame["entry_binding_sha256"].ne(entry_binding).any()
    ):
        raise RuntimeError("COMPACT_LIFECYCLE_FRAME_INVALID")
    clock = pd.DatetimeIndex(pd.to_datetime(m1_times, utc=True)).as_unit("ns")
    end = pd.Timestamp(split_end).as_unit("ns")
    clock_ns = np.asarray(clock.asi8, dtype=np.int64)
    clock_sha = _m1_clock_sha256(clock_ns)
    for row in frame.itertuples(index=False):
        if row.m1_clock_sha256 != clock_sha:
            raise RuntimeError("COMPACT_LIFECYCLE_M1_CLOCK_BINDING_INVALID")
        gap_after = int(row.gap_after_m1_row)
        if gap_after >= 0:
            if (
                gap_after < int(row.entry_m1_start_row)
                or gap_after + 1 >= len(clock)
            ):
                raise RuntimeError("COMPACT_LIFECYCLE_GAP_CENSOR_INVALID")
            observed_gap_seconds = int(
                (clock_ns[gap_after + 1] - clock_ns[gap_after]) // 1_000_000_000
            )
            expected_gap_classification = (
                "weekend_source_absence"
                if observed_gap_seconds >= 24 * 60 * 60
                and (
                    clock[gap_after].weekday() == 4
                    or clock[gap_after + 1].weekday() in (6, 0)
                )
                else "unknown_source_absence"
            )
            if (
                clock_ns[gap_after + 1] - clock_ns[gap_after]
                == int(pd.Timedelta(minutes=1).value)
                or int(row.available_state_count)
                != gap_after - int(row.entry_m1_start_row) + 1
                or int(row.gap_seconds) != observed_gap_seconds
                or row.gap_classification != expected_gap_classification
                or np.any(
                    np.diff(
                        clock_ns[int(row.entry_m1_start_row) : gap_after + 1]
                    )
                    != int(pd.Timedelta(minutes=1).value)
                )
            ):
                raise RuntimeError("COMPACT_LIFECYCLE_GAP_CENSOR_INVALID")
        elif row.gap_classification != "split_end" or int(row.gap_seconds) != 0:
            raise RuntimeError("COMPACT_LIFECYCLE_GAP_CENSOR_INVALID")
        expected_gap_binding = _canonical_sha256(
            {
                "schema_version": "gx1_m1_gap_censor_binding_v1",
                "entry_row_index": int(row.entry_row_index),
                "classification_source_sha256": gap_source_sha,
                "classification": str(row.gap_classification),
                "gap_after_m1_row": gap_after,
                "gap_seconds": int(row.gap_seconds),
                "successor_across_gap_allowed": False,
            }
        )
        if row.gap_binding_sha256 != expected_gap_binding:
            raise RuntimeError("COMPACT_LIFECYCLE_GAP_BINDING_INVALID")
        if row.row_identity_sha256 != _row_identity_sha256(row._asdict()):
            raise RuntimeError("COMPACT_LIFECYCLE_ROW_IDENTITY_INVALID")
        for side in ("long", "short"):
            count = int(getattr(row, f"{side}_lifecycle_state_count"))
            chunks = int(getattr(row, f"{side}_chunk_count"))
            terminal = bool(getattr(row, f"{side}_economic_terminal"))
            censored = bool(getattr(row, f"{side}_right_censored"))
            last = int(getattr(row, f"{side}_last_state_m1_row"))
            terminal_row = int(getattr(row, f"{side}_terminal_state_m1_row"))
            if (
                count < 1
                or chunks != math.ceil(count / UNIFIED_EXIT_CHUNK_ROWS)
                or terminal == censored
                or last != int(row.entry_m1_start_row) + count - 1
                or terminal_row != (last if terminal else -1)
                or last >= len(clock)
                or clock[last] + pd.Timedelta(seconds=EXIT_DECISION_BAR_SECONDS) > end
            ):
                raise RuntimeError("COMPACT_LIFECYCLE_FRAME_VALUE_INVALID")
            expected_pointer_sha = _chunk_pointer_stream_sha256(
                entry_row_index=int(row.entry_row_index),
                side_index=0 if side == "long" else 1,
                entry_m1_start_row=int(row.entry_m1_start_row),
                state_count=count,
                economic_terminal=terminal,
                gap_binding_sha256=str(row.gap_binding_sha256),
                m1_clock_sha256=clock_sha,
                m1_source_sha256=str(row.m1_source_sha256),
            )
            if getattr(row, f"{side}_chunk_pointer_stream_sha256") != expected_pointer_sha:
                raise RuntimeError("COMPACT_LIFECYCLE_POINTER_BINDING_INVALID")
        if int(row.pair_chunk_count) != max(
            int(row.long_chunk_count), int(row.short_chunk_count)
        ):
            raise RuntimeError("COMPACT_LIFECYCLE_PAIR_TIMELINE_INVALID")
    return frame.copy()


def _build_split_from_files(
    *,
    split: str,
    entry_path: Path,
    entry_manifest_path: Path,
    economic_authority_path: Path,
    split_end: str,
    dataset_run_id: str,
    m1_times: pd.DatetimeIndex,
    m1_source_sha256: str,
    m1_source_manifest_sha256: str,
    full_v1_admission: Mapping[str, Any],
    planned_epochs: int,
    claim_full_coverage: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _manifest, bound_entry_rows, adoption_witness = _validate_entry_manifest(
        entry_manifest_path,
        entry_path=entry_path,
        split=split,
        dataset_run_id=dataset_run_id,
        full_v1_admission=full_v1_admission,
    )
    entry_manifest_sha = adoption_witness["entry_manifest_sha256"]
    entry_sha = adoption_witness["entry_parquet_sha256"]
    entry_binding_sha = adoption_witness["witness_sha256"]
    entry_times = pd.read_parquet(entry_path, columns=["time"])["time"]
    if len(entry_times) != bound_entry_rows:
        raise RuntimeError("COMPACT_LIFECYCLE_ENTRY_ROW_COUNT_CHANGED")
    mapping, authority, authority_sha = _load_terminal_counts(
        economic_authority_path,
        split=split,
        dataset_run_id=dataset_run_id,
        entry_rows=len(entry_times),
    )
    compact = build_compact_split(
        entry_times=entry_times,
        m1_times=m1_times,
        split=split,
        split_end=split_end,
        terminal_state_count_by_entry_side=mapping,
        m1_source_sha256=m1_source_sha256,
        gap_classification_source_sha256=m1_source_manifest_sha256,
        entry_binding_sha256=entry_binding_sha,
    )
    lineage = _canonical_sha256(
        {
            "dataset_run_id": dataset_run_id,
            "split": split,
            "entry_sha256": entry_sha,
            "entry_manifest_sha256": entry_manifest_sha,
            "entry_adoption_witness_sha256": entry_binding_sha,
            "m1_source_sha256": m1_source_sha256,
            "economic_authority_sha256": authority_sha,
        }
    )
    counts = {
        int(row.entry_row_index): int(row.pair_chunk_count)
        for row in compact.itertuples(index=False)
    }
    coverage = pair_schedule_coverage(
        chunk_count_by_entry=counts,
        planned_epochs=planned_epochs,
        lineage_sha256=lineage,
        split=split,
        claim_full_coverage=claim_full_coverage,
    )
    manifest = {
        **unified_exit_lifecycle_v2_contract(),
        "compact_schema_version": COMPACT_LIFECYCLE_SCHEMA_VERSION,
        "decision": "PASS",
        "dataset_run_id": dataset_run_id,
        "split": split,
        "split_end_utc": pd.Timestamp(split_end).isoformat(),
        "test_accessed": False,
        "entry_parquet_path": str(entry_path),
        "entry_parquet_sha256": entry_sha,
        "entry_manifest_path": str(entry_manifest_path),
        "entry_manifest_sha256": entry_manifest_sha,
        "entry_adoption_witness": adoption_witness,
        "entry_binding_sha256": entry_binding_sha,
        "gap_classification_source_sha256": m1_source_manifest_sha256,
        "economic_authority_path": str(economic_authority_path),
        "economic_authority_sha256": authority_sha,
        "economic_authority": authority,
        "m1_source_sha256": m1_source_sha256,
        "compact_rows": len(compact),
        "compact_pointer_stream_sha256": _compact_pointer_stream_sha256(compact),
        "successor_pointer_binding": (
            "per_entry_side_chunk_pointer_stream_sha256"
        ),
        "target_q_stored": False,
        "producer_source": _producer_source_identity(),
        "schedule_lineage_sha256": lineage,
        "schedule_coverage": coverage,
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    return compact, manifest


def materialize_compact_train_val_bundle(
    *,
    output_dir: Path,
    dataset_run_id: str,
    train_entry_path: Path,
    train_entry_manifest_path: Path,
    train_economic_authority_path: Path,
    train_split_end: str,
    val_entry_path: Path,
    val_entry_manifest_path: Path,
    val_economic_authority_path: Path,
    val_split_end: str,
    m1_source_path: Path,
    m1_source_manifest_path: Path,
    planned_epochs: int,
    claim_full_coverage: bool,
    publish: bool,
) -> dict[str, Any]:
    """Validate or atomically publish one TRAIN/VAL-only compact bundle."""

    if not dataset_run_id or type(publish) is not bool:
        raise RuntimeError("COMPACT_LIFECYCLE_INVOCATION_INVALID")
    output_dir = output_dir.expanduser().resolve()
    paths = [
        train_entry_path,
        train_entry_manifest_path,
        train_economic_authority_path,
        val_entry_path,
        val_entry_manifest_path,
        val_economic_authority_path,
        m1_source_path,
        m1_source_manifest_path,
    ]
    paths = [path.expanduser().resolve() for path in paths]
    (
        train_entry_path,
        train_entry_manifest_path,
        train_economic_authority_path,
        val_entry_path,
        val_entry_manifest_path,
        val_economic_authority_path,
        m1_source_path,
        m1_source_manifest_path,
    ) = paths
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise RuntimeError("COMPACT_LIFECYCLE_INPUT_FILE_INVALID")
    full_v1_admission = _require_full_v1_admission(
        entry_paths={"train": train_entry_path, "val": val_entry_path},
        entry_manifest_paths={
            "train": train_entry_manifest_path,
            "val": val_entry_manifest_path,
        },
        dataset_run_id=dataset_run_id,
    )
    m1_times, _m1_manifest, m1_manifest_sha, m1_source_sha = _validate_m1_source(
        m1_source_path, m1_source_manifest_path
    )
    split_specs = {
        "train": (
            train_entry_path,
            train_entry_manifest_path,
            train_economic_authority_path,
            train_split_end,
        ),
        "val": (
            val_entry_path,
            val_entry_manifest_path,
            val_economic_authority_path,
            val_split_end,
        ),
    }
    built: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}
    try:
        for split, (entry, manifest, authority, split_end) in split_specs.items():
            built[split] = _build_split_from_files(
                split=split,
                entry_path=entry,
                entry_manifest_path=manifest,
                economic_authority_path=authority,
                split_end=split_end,
                dataset_run_id=dataset_run_id,
                m1_times=m1_times,
                m1_source_sha256=m1_source_sha,
                m1_source_manifest_sha256=m1_manifest_sha,
                full_v1_admission=full_v1_admission,
                planned_epochs=planned_epochs,
                claim_full_coverage=claim_full_coverage,
            )
    except _EconomicAuthorityBlocked as exc:
        if publish:
            raise RuntimeError("COMPACT_LIFECYCLE_PUBLISH_BLOCKED") from exc
        return {
            "mode": "validate_no_publish",
            "decision": "BLOCKED",
            "published": False,
            "output_dir": str(output_dir),
            "blockers": [str(exc)],
            "full_v1_admission_verified": True,
            "test_accessed": False,
        }
    root = {
        "schema_version": COMPACT_ROOT_SCHEMA_VERSION,
        "decision": "PASS",
        "dataset_run_id": dataset_run_id,
        "allowed_splits": ["train", "val"],
        "test_accessed": False,
        "m1_source_path": str(m1_source_path),
        "m1_source_sha256": m1_source_sha,
        "m1_source_manifest_path": str(m1_source_manifest_path),
        "m1_source_manifest_sha256": m1_manifest_sha,
        "chunk_state_capacity": UNIFIED_EXIT_CHUNK_ROWS,
        "compact_storage_complexity": "one_row_per_entry_pair_O_entry",
        "successor_pointer_binding": "hashed_static_pointer_stream_per_side",
        "target_q_stored": False,
        "producer_source": _producer_source_identity(),
        "split_manifests": {
            split: {
                "manifest_sha256": manifest["manifest_sha256"],
                "compact_rows": len(frame),
                "schedule_coverage_sha256": manifest["schedule_coverage"][
                    "contract_sha256"
                ],
            }
            for split, (frame, manifest) in built.items()
        },
        "publish_requested": publish,
    }
    root["manifest_sha256"] = _canonical_sha256(root)
    if not publish:
        return {
            "mode": "validate_no_publish",
            "published": False,
            "output_dir": str(output_dir),
            "root_manifest": root,
            "splits": {
                split: {
                    "compact_rows": len(frame),
                    "compact_pointer_stream_sha256": manifest[
                        "compact_pointer_stream_sha256"
                    ],
                    "schedule_coverage": manifest["schedule_coverage"],
                }
                for split, (frame, manifest) in built.items()
            },
        }
    if output_dir.exists() or output_dir.is_symlink():
        raise RuntimeError("COMPACT_LIFECYCLE_OUTPUT_ALREADY_EXISTS")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging.", dir=output_dir.parent)
    )
    try:
        for split, (frame, manifest) in built.items():
            parquet = staging / f"{split}_unified_exit_lifecycle_v2.parquet"
            frame.to_parquet(parquet, index=False)
            manifest = {
                **manifest,
                "compact_parquet": parquet.name,
                "compact_parquet_sha256": _sha256_file(parquet),
            }
            manifest["manifest_sha256"] = _canonical_sha256(
                {key: value for key, value in manifest.items() if key != "manifest_sha256"}
            )
            (staging / f"{split}_unified_exit_lifecycle_v2.manifest.json").write_bytes(
                json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False).encode()
                + b"\n"
            )
            root["split_manifests"][split].update(
                {
                    "manifest": f"{split}_unified_exit_lifecycle_v2.manifest.json",
                    "manifest_sha256": manifest["manifest_sha256"],
                    "compact_parquet": parquet.name,
                    "compact_parquet_sha256": manifest["compact_parquet_sha256"],
                }
            )
        root["manifest_sha256"] = _canonical_sha256(
            {key: value for key, value in root.items() if key != "manifest_sha256"}
        )
        (staging / "UNIFIED_EXIT_LIFECYCLE_V2_MANIFEST.json").write_bytes(
            json.dumps(root, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
        )
        os.rename(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "mode": "publish",
        "published": True,
        "output_dir": str(output_dir),
        "root_manifest_sha256": root["manifest_sha256"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-run-id", required=True)
    parser.add_argument("--train-entry-parquet", required=True, type=Path)
    parser.add_argument("--train-entry-manifest", required=True, type=Path)
    parser.add_argument("--train-economic-authority", required=True, type=Path)
    parser.add_argument("--train-split-end", required=True)
    parser.add_argument("--val-entry-parquet", required=True, type=Path)
    parser.add_argument("--val-entry-manifest", required=True, type=Path)
    parser.add_argument("--val-economic-authority", required=True, type=Path)
    parser.add_argument("--val-split-end", required=True)
    parser.add_argument("--m1-source-parquet", required=True, type=Path)
    parser.add_argument("--m1-source-manifest", required=True, type=Path)
    parser.add_argument("--planned-epochs", required=True, type=int)
    parser.add_argument("--claim-full-coverage", action="store_true")
    parser.add_argument("--validate-no-publish", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = materialize_compact_train_val_bundle(
        output_dir=args.output_dir,
        dataset_run_id=args.dataset_run_id,
        train_entry_path=args.train_entry_parquet,
        train_entry_manifest_path=args.train_entry_manifest,
        train_economic_authority_path=args.train_economic_authority,
        train_split_end=args.train_split_end,
        val_entry_path=args.val_entry_parquet,
        val_entry_manifest_path=args.val_entry_manifest,
        val_economic_authority_path=args.val_economic_authority,
        val_split_end=args.val_split_end,
        m1_source_path=args.m1_source_parquet,
        m1_source_manifest_path=args.m1_source_manifest,
        planned_epochs=args.planned_epochs,
        claim_full_coverage=args.claim_full_coverage,
        publish=not args.validate_no_publish,
    )
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = (
    "COMPACT_COLUMNS",
    "COMPACT_LIFECYCLE_SCHEMA_VERSION",
    "ECONOMIC_COUNTS_SCHEMA_VERSION",
    "PAIR_COVERAGE_SCHEMA_VERSION",
    "PAIR_SCHEDULE_SCHEMA_VERSION",
    "build_compact_split",
    "materialize_compact_train_val_bundle",
    "pair_chunk_for_epoch",
    "pair_chunk_permutation",
    "pair_schedule_coverage",
    "require_compact_split",
    "scheduled_pair_chunk_pointer",
)
