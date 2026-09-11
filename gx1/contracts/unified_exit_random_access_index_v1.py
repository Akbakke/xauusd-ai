"""Minimal O(Entry) lifecycle index for sampled random-access Exit learning."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RANDOM_ACCESS_INDEX_SCHEMA_VERSION = "gx1_unified_exit_random_access_index_v1"
RANDOM_ACCESS_INDEX_ROOT_SCHEMA_VERSION = "gx1_unified_exit_random_access_index_root_v1"
RANDOM_ACCESS_INDEX_V2_SCHEMA_VERSION = "gx1_unified_exit_random_access_index_v2"
RANDOM_ACCESS_INDEX_V2_ROOT_SCHEMA_VERSION = (
    "gx1_unified_exit_random_access_index_root_v2"
)
RANDOM_ACCESS_INDEX_V1_COLUMNS = (
    "entry_row_index",
    "entry_time_ns",
    "first_state_time_ns",
    "parent_m1_start_row",
    "child_m1_start_row",
    "successor_transition_count",
    "lifecycle_state_count",
    "economic_terminal",
    "right_censored",
    "entry_bid",
    "entry_ask",
    "episode_binding_sha256",
    "entry_fill_binding_sha256",
    "row_identity_sha256",
)
RANDOM_ACCESS_INDEX_V2_COLUMNS = (
    "entry_row_index",
    "parent_entry_row_index",
    *RANDOM_ACCESS_INDEX_V1_COLUMNS[1:],
)
# Backward-compatible public name for immutable V1-V3 evidence.
RANDOM_ACCESS_INDEX_COLUMNS = RANDOM_ACCESS_INDEX_V1_COLUMNS
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise RuntimeError(f"UNIFIED_EXIT_RANDOM_ACCESS_INDEX_{label}_SHA_INVALID")
    return value


def _clock(values: Sequence[Any], label: str) -> pd.DatetimeIndex:
    result = pd.DatetimeIndex(
        pd.to_datetime(values, utc=True, errors="coerce")
    ).as_unit("ns")
    if (
        result.empty
        or result.hasnans
        or not result.is_unique
        or not result.is_monotonic_increasing
    ):
        raise RuntimeError(f"UNIFIED_EXIT_RANDOM_ACCESS_INDEX_{label}_CLOCK_INVALID")
    return result


def _row_identity(record: Mapping[str, Any]) -> str:
    columns = (
        RANDOM_ACCESS_INDEX_V2_COLUMNS
        if "parent_entry_row_index" in record
        else RANDOM_ACCESS_INDEX_V1_COLUMNS
    )
    return canonical_sha256({key: record[key] for key in columns[:-1]})


def _clock_sha256(clock: pd.DatetimeIndex) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(clock.asi8, dtype="<i8").tobytes()
    ).hexdigest()


def parent_entry_mapping_sha256(frame: pd.DataFrame) -> str:
    """Bind child id, exact parent id/time and existing episode/fill identity."""

    checked = require_random_access_index(frame)
    if "parent_entry_row_index" not in checked:
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_INDEX_PARENT_ENTRY_MISSING")
    digest = hashlib.sha256()
    for row in checked.itertuples(index=False):
        digest.update(
            json.dumps(
                {
                    "entry_row_index": int(row.entry_row_index),
                    "parent_entry_row_index": int(row.parent_entry_row_index),
                    "entry_time_ns": int(row.entry_time_ns),
                    "episode_binding_sha256": row.episode_binding_sha256,
                    "entry_fill_binding_sha256": row.entry_fill_binding_sha256,
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def build_random_access_index_v2(
    *,
    split: str,
    entry_times: Sequence[Any],
    parent_entry_times: Sequence[Any],
    child_m1_times: Sequence[Any],
    parent_m1_times: Sequence[Any],
    successor_transition_counts: Sequence[int],
    entry_bid: Sequence[float],
    entry_ask: Sequence[float],
    episode_binding_sha256_by_entry: Sequence[str],
    entry_fill_binding_sha256_by_entry: Sequence[str],
) -> tuple[pd.DataFrame, int]:
    """Build V2 with an exact child-to-parent Entry row mapping."""

    legacy, parent_m1_offset = build_random_access_index(
        split=split,
        entry_times=entry_times,
        child_m1_times=child_m1_times,
        parent_m1_times=parent_m1_times,
        successor_transition_counts=successor_transition_counts,
        entry_bid=entry_bid,
        entry_ask=entry_ask,
        episode_binding_sha256_by_entry=episode_binding_sha256_by_entry,
        entry_fill_binding_sha256_by_entry=entry_fill_binding_sha256_by_entry,
    )
    child_entry = _clock(entry_times, "ENTRY")
    parent_entry = _clock(parent_entry_times, "PARENT_ENTRY")
    parent_rows = np.searchsorted(parent_entry.asi8, child_entry.asi8).astype(
        "<i8", copy=False
    )
    if (
        np.any(parent_rows < 0)
        or np.any(parent_rows >= len(parent_entry))
        or not np.array_equal(parent_entry.asi8[parent_rows], child_entry.asi8)
        or len(np.unique(parent_rows)) != len(parent_rows)
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_INDEX_PARENT_ENTRY_INVALID")
    frame = legacy.copy()
    frame.insert(1, "parent_entry_row_index", parent_rows)
    for position, row in enumerate(frame.to_dict(orient="records")):
        frame.at[position, "row_identity_sha256"] = _row_identity(row)
    return frame.loc[:, RANDOM_ACCESS_INDEX_V2_COLUMNS], parent_m1_offset


def build_random_access_index(
    *,
    split: str,
    entry_times: Sequence[Any],
    child_m1_times: Sequence[Any],
    parent_m1_times: Sequence[Any],
    successor_transition_counts: Sequence[int],
    entry_bid: Sequence[float],
    entry_ask: Sequence[float],
    episode_binding_sha256_by_entry: Sequence[str],
    entry_fill_binding_sha256_by_entry: Sequence[str],
) -> tuple[pd.DataFrame, int]:
    """Build one immutable row per Entry; no prefix state or chunk is stored."""

    if split not in {"train", "val"}:
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_INDEX_SPLIT_INVALID")
    entry = _clock(entry_times, "ENTRY")
    child = _clock(child_m1_times, "CHILD_M1")
    parent = _clock(parent_m1_times, "PARENT_M1")
    counts = np.ascontiguousarray(successor_transition_counts, dtype="<i8")
    bids = np.ascontiguousarray(entry_bid, dtype="<f8")
    asks = np.ascontiguousarray(entry_ask, dtype="<f8")
    population = len(entry)
    if (
        counts.shape != (population,)
        or bids.shape != (population,)
        or asks.shape != (population,)
        or len(episode_binding_sha256_by_entry) != population
        or len(entry_fill_binding_sha256_by_entry) != population
        or np.any(counts < 1)
        or not np.isfinite(bids).all()
        or not np.isfinite(asks).all()
        or np.any(bids <= 0.0)
        or np.any(asks <= 0.0)
        or np.any(bids > asks)
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_INDEX_INPUT_INVALID")
    for value in (
        *episode_binding_sha256_by_entry,
        *entry_fill_binding_sha256_by_entry,
    ):
        _sha(value, "ENTRY_BINDING")
    parent_offset = int(np.searchsorted(parent.asi8, child.asi8[0]))
    parent_stop = parent_offset + len(child)
    if parent_stop > len(parent) or not np.array_equal(
        parent.asi8[parent_offset:parent_stop], child.asi8
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_INDEX_PARENT_SLICE_INVALID")
    first_state = entry.asi8 + 300_000_000_000
    child_starts = np.searchsorted(child.asi8, first_state).astype("<i8", copy=False)
    if (
        np.any(child_starts >= len(child))
        or not np.array_equal(child.asi8[child_starts], first_state)
        or np.any(child_starts + counts >= len(child))
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_INDEX_RANGE_INVALID")
    parent_starts = child_starts + parent_offset
    rows: list[dict[str, Any]] = []
    for position in range(population):
        record = {
            "entry_row_index": position,
            "entry_time_ns": int(entry.asi8[position]),
            "first_state_time_ns": int(first_state[position]),
            "parent_m1_start_row": int(parent_starts[position]),
            "child_m1_start_row": int(child_starts[position]),
            "successor_transition_count": int(counts[position]),
            "lifecycle_state_count": int(counts[position]) + 1,
            "economic_terminal": False,
            "right_censored": True,
            "entry_bid": float(bids[position]),
            "entry_ask": float(asks[position]),
            "episode_binding_sha256": episode_binding_sha256_by_entry[position],
            "entry_fill_binding_sha256": entry_fill_binding_sha256_by_entry[position],
        }
        record["row_identity_sha256"] = _row_identity(record)
        rows.append(record)
    return pd.DataFrame(rows, columns=RANDOM_ACCESS_INDEX_COLUMNS), parent_offset


def index_stream_sha256(frame: pd.DataFrame) -> str:
    checked = require_random_access_index(frame)
    digest = hashlib.sha256()
    for row in checked.itertuples(index=False, name=None):
        digest.update(
            json.dumps(
                list(row),
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def require_random_access_index(
    frame: pd.DataFrame,
    *,
    expected_split: str | None = None,
) -> pd.DataFrame:
    if (
        not isinstance(frame, pd.DataFrame)
        or tuple(frame.columns)
        not in {RANDOM_ACCESS_INDEX_V1_COLUMNS, RANDOM_ACCESS_INDEX_V2_COLUMNS}
        or frame.empty
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_INDEX_SCHEMA_INVALID")
    population = len(frame)
    parent_entry_rows = (
        frame["parent_entry_row_index"].to_numpy(dtype="<i8")
        if "parent_entry_row_index" in frame
        else None
    )
    if (
        not np.array_equal(
            frame["entry_row_index"].to_numpy(dtype="<i8"),
            np.arange(population, dtype="<i8"),
        )
        or not frame["entry_time_ns"].is_monotonic_increasing
        or not frame["first_state_time_ns"].is_monotonic_increasing
        or not np.array_equal(
            frame["first_state_time_ns"].to_numpy(dtype="<i8"),
            frame["entry_time_ns"].to_numpy(dtype="<i8") + 300_000_000_000,
        )
        or np.any(frame["parent_m1_start_row"].to_numpy(dtype="<i8") < 0)
        or np.any(frame["child_m1_start_row"].to_numpy(dtype="<i8") < 0)
        or (
            parent_entry_rows is not None
            and (
                np.any(parent_entry_rows < 0)
                or len(np.unique(parent_entry_rows)) != population
                or np.any(np.diff(parent_entry_rows) <= 0)
            )
        )
        or np.any(frame["successor_transition_count"].to_numpy(dtype="<i8") < 1)
        or not np.array_equal(
            frame["lifecycle_state_count"].to_numpy(dtype="<i8"),
            frame["successor_transition_count"].to_numpy(dtype="<i8") + 1,
        )
        or frame["economic_terminal"].astype(bool).any()
        or not frame["right_censored"].astype(bool).all()
        or not np.isfinite(frame[["entry_bid", "entry_ask"]].to_numpy()).all()
        or np.any(frame["entry_bid"].to_numpy() > frame["entry_ask"].to_numpy())
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_INDEX_VALUES_INVALID")
    for row in frame.to_dict(orient="records"):
        if (
            _sha(row["episode_binding_sha256"], "EPISODE_BINDING")
            != row["episode_binding_sha256"]
            or _sha(row["entry_fill_binding_sha256"], "FILL_BINDING")
            != row["entry_fill_binding_sha256"]
            or _sha(row["row_identity_sha256"], "ROW_IDENTITY")
            != row["row_identity_sha256"]
            or row["row_identity_sha256"] != _row_identity(row)
        ):
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_INDEX_IDENTITY_INVALID")
    if expected_split is not None and expected_split not in {"train", "val"}:
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_INDEX_SPLIT_INVALID")
    return frame


def require_random_access_index_manifest(
    value: Mapping[str, Any],
    *,
    expected_split: str,
    index_frame: pd.DataFrame | None = None,
    index_path: Path | None = None,
    verify_sources: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_INDEX_MANIFEST_INVALID")
    data = dict(value)
    claimed = data.pop("manifest_sha256", None)
    sources = value.get("source_bindings")
    if (
        value.get("schema_version")
        not in {
            RANDOM_ACCESS_INDEX_SCHEMA_VERSION,
            RANDOM_ACCESS_INDEX_V2_SCHEMA_VERSION,
        }
        or value.get("decision") != "PASS"
        or value.get("split") != expected_split
        or value.get("storage_granularity") != "one_row_per_entry"
        or value.get("full_prefix_states_stored") is not False
        or value.get("chunk_pointers_stored") is not False
        or value.get("target_q_stored") is not False
        or value.get("economic_terminal_count") != 0
        or value.get("split_end_is_right_censor") is not True
        or value.get("test_accessed") is not False
        or not isinstance(sources, Mapping)
        or claimed != canonical_sha256(data)
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_INDEX_MANIFEST_INVALID")
    is_v2 = value["schema_version"] == RANDOM_ACCESS_INDEX_V2_SCHEMA_VERSION
    if is_v2:
        required_sources = {
            "entry_parquet",
            "entry_manifest",
            "parent_entry_parquet",
            "parent_entry_manifest",
        }
        for key in (
            "child_entry_clock_sha256",
            "parent_entry_clock_sha256",
            "parent_entry_row_indices_sha256",
            "parent_entry_mapping_sha256",
            "parent_entry_source_sha256",
            "parent_entry_manifest_sha256",
        ):
            _sha(value.get(key), key.upper())
        if (
            not required_sources <= set(sources)
            or value.get("parent_entry_source_sha256")
            != sources["parent_entry_parquet"].get("sha256")
            or value.get("parent_entry_manifest_sha256")
            != sources["parent_entry_manifest"].get("sha256")
            or isinstance(value.get("parent_entry_source_rows"), bool)
            or not isinstance(value.get("parent_entry_source_rows"), int)
            or value["parent_entry_source_rows"] < value.get("entry_row_count", 0)
        ):
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_INDEX_PARENT_ENTRY_INVALID")
    for binding in sources.values():
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "sha256"}
            or not isinstance(binding.get("path"), str)
        ):
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_INDEX_SOURCE_INVALID")
        _sha(binding.get("sha256"), "SOURCE")
    if index_path is not None:
        resolved = index_path.expanduser().resolve()
        if (
            not resolved.is_file()
            or resolved.is_symlink()
            or str(resolved) != value.get("index_parquet_path")
            or hashlib.sha256(resolved.read_bytes()).hexdigest()
            != value.get("index_parquet_sha256")
        ):
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_INDEX_FILE_INVALID")
    if verify_sources:
        for binding in sources.values():
            resolved = Path(binding["path"]).expanduser().resolve()
            if (
                not resolved.is_file()
                or resolved.is_symlink()
                or hashlib.sha256(resolved.read_bytes()).hexdigest()
                != binding["sha256"]
            ):
                raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_INDEX_SOURCE_INVALID")
        if is_v2:
            child_entry = _clock(
                pd.read_parquet(sources["entry_parquet"]["path"], columns=["time"])[
                    "time"
                ],
                "ENTRY",
            )
            parent_entry = _clock(
                pd.read_parquet(
                    sources["parent_entry_parquet"]["path"], columns=["time"]
                )["time"],
                "PARENT_ENTRY",
            )
            if (
                index_frame is None
                or "parent_entry_row_index" not in index_frame
                or _clock_sha256(child_entry) != value["child_entry_clock_sha256"]
                or _clock_sha256(parent_entry) != value["parent_entry_clock_sha256"]
                or len(parent_entry) != value["parent_entry_source_rows"]
            ):
                raise RuntimeError(
                    "UNIFIED_EXIT_RANDOM_ACCESS_INDEX_PARENT_ENTRY_INVALID"
                )
            parent_rows = np.ascontiguousarray(
                index_frame["parent_entry_row_index"].to_numpy(dtype="<i8")
            )
            if (
                hashlib.sha256(parent_rows.tobytes()).hexdigest()
                != value["parent_entry_row_indices_sha256"]
                or not np.array_equal(parent_entry.asi8[parent_rows], child_entry.asi8)
                or parent_entry_mapping_sha256(index_frame)
                != value["parent_entry_mapping_sha256"]
            ):
                raise RuntimeError(
                    "UNIFIED_EXIT_RANDOM_ACCESS_INDEX_PARENT_ENTRY_INVALID"
                )
    if index_frame is not None:
        checked = require_random_access_index(
            index_frame, expected_split=expected_split
        )
        if (
            len(checked) != value.get("entry_row_count")
            or int(checked["successor_transition_count"].sum())
            != value.get("successor_transition_total")
            or index_stream_sha256(checked) != value.get("index_stream_sha256")
            or (
                is_v2
                and parent_entry_mapping_sha256(checked)
                != value.get("parent_entry_mapping_sha256")
            )
        ):
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_INDEX_MANIFEST_MISMATCH")
    return dict(value)


def require_random_access_index_root(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_INDEX_ROOT_INVALID")
    data = dict(value)
    claimed = data.pop("root_sha256", None)
    splits = value.get("splits")
    if (
        value.get("schema_version")
        not in {
            RANDOM_ACCESS_INDEX_ROOT_SCHEMA_VERSION,
            RANDOM_ACCESS_INDEX_V2_ROOT_SCHEMA_VERSION,
        }
        or value.get("decision") != "PASS"
        or value.get("allowed_splits") != ["train", "val"]
        or value.get("storage_granularity") != "one_row_per_entry"
        or value.get("full_prefix_states_stored") is not False
        or value.get("chunk_pointers_stored") is not False
        or value.get("test_accessed") is not False
        or not isinstance(splits, Mapping)
        or set(splits) != {"train", "val"}
        or claimed != canonical_sha256(data)
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_INDEX_ROOT_INVALID")
    if value["schema_version"] == RANDOM_ACCESS_INDEX_V2_ROOT_SCHEMA_VERSION:
        equivalence = value.get("predecessor_equivalence")
        if (
            not isinstance(equivalence, Mapping)
            or set(equivalence)
            != {
                "path",
                "sha256",
                "receipt_sha256",
                "benchmark_receipt_transfer_to_v4_authorized",
            }
            or not Path(str(equivalence.get("path", ""))).is_absolute()
            or equivalence.get("benchmark_receipt_transfer_to_v4_authorized")
            is not True
        ):
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_INDEX_EQUIVALENCE_INVALID")
        _sha(equivalence.get("sha256"), "EQUIVALENCE_FILE")
        _sha(equivalence.get("receipt_sha256"), "EQUIVALENCE_RECEIPT")
    return dict(value)


__all__ = (
    "RANDOM_ACCESS_INDEX_COLUMNS",
    "RANDOM_ACCESS_INDEX_V1_COLUMNS",
    "RANDOM_ACCESS_INDEX_V2_COLUMNS",
    "RANDOM_ACCESS_INDEX_V2_ROOT_SCHEMA_VERSION",
    "RANDOM_ACCESS_INDEX_V2_SCHEMA_VERSION",
    "RANDOM_ACCESS_INDEX_ROOT_SCHEMA_VERSION",
    "RANDOM_ACCESS_INDEX_SCHEMA_VERSION",
    "build_random_access_index",
    "build_random_access_index_v2",
    "canonical_sha256",
    "index_stream_sha256",
    "parent_entry_mapping_sha256",
    "require_random_access_index",
    "require_random_access_index_manifest",
    "require_random_access_index_root",
)
