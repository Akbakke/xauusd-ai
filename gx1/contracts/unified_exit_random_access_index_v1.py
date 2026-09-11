"""Minimal O(Entry) lifecycle index for sampled random-access Exit learning."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

RANDOM_ACCESS_INDEX_SCHEMA_VERSION = "gx1_unified_exit_random_access_index_v1"
RANDOM_ACCESS_INDEX_ROOT_SCHEMA_VERSION = "gx1_unified_exit_random_access_index_root_v1"
RANDOM_ACCESS_INDEX_COLUMNS = (
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
    return canonical_sha256(
        {key: record[key] for key in RANDOM_ACCESS_INDEX_COLUMNS[:-1]}
    )


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
        or tuple(frame.columns) != RANDOM_ACCESS_INDEX_COLUMNS
        or frame.empty
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_INDEX_SCHEMA_INVALID")
    population = len(frame)
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
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_INDEX_MANIFEST_INVALID")
    data = dict(value)
    claimed = data.pop("manifest_sha256", None)
    sources = value.get("source_bindings")
    if (
        value.get("schema_version") != RANDOM_ACCESS_INDEX_SCHEMA_VERSION
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
    for binding in sources.values():
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "sha256"}
            or not isinstance(binding.get("path"), str)
        ):
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_INDEX_SOURCE_INVALID")
        _sha(binding.get("sha256"), "SOURCE")
    if index_frame is not None:
        checked = require_random_access_index(
            index_frame, expected_split=expected_split
        )
        if (
            len(checked) != value.get("entry_row_count")
            or int(checked["successor_transition_count"].sum())
            != value.get("successor_transition_total")
            or index_stream_sha256(checked) != value.get("index_stream_sha256")
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
        value.get("schema_version") != RANDOM_ACCESS_INDEX_ROOT_SCHEMA_VERSION
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
    return dict(value)


__all__ = (
    "RANDOM_ACCESS_INDEX_COLUMNS",
    "RANDOM_ACCESS_INDEX_ROOT_SCHEMA_VERSION",
    "RANDOM_ACCESS_INDEX_SCHEMA_VERSION",
    "build_random_access_index",
    "canonical_sha256",
    "index_stream_sha256",
    "require_random_access_index",
    "require_random_access_index_manifest",
    "require_random_access_index_root",
)
