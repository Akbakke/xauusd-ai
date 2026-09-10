"""Chunked, unbounded Exit lifecycle pointers.

This owner deliberately stores source pointers rather than feature tensors.
The 512-row value is a compute/detail-tail capacity.  Economic termination,
right censoring and causal successor availability are separate fields.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_exit_feature_base_v1 import EXIT_DECISION_BAR_SECONDS
from gx1.models.entry_v10.direction_decision_contract import (
    UNIFIED_EXIT_DETAILED_PATH_TAIL_BARS,
    UNIFIED_EXIT_SIDE_ORDER,
)


UNIFIED_EXIT_LIFECYCLE_V2_SCHEMA_VERSION = (
    "gx1_unified_exit_chunked_lifecycle_v2"
)
UNIFIED_EXIT_ECONOMIC_AUTHORITY_SCHEMA_VERSION = (
    "gx1_unified_exit_economic_lifecycle_authority_v1"
)
UNIFIED_EXIT_CHUNK_POINTER_SCHEMA_VERSION = (
    "gx1_unified_exit_chunk_pointer_stream_v1"
)
UNIFIED_EXIT_CHUNK_ROWS = UNIFIED_EXIT_DETAILED_PATH_TAIL_BARS
UNIFIED_EXIT_CHUNK_COLUMNS = (
    "schema_version",
    "chunk_index",
    "entry_row_index",
    "side_index",
    "side",
    "entry_m1_start_row",
    "chunk_m1_start_row",
    "chunk_start_bars_in_trade",
    "valid_state_count",
    "successor_available",
    "successor_m1_row",
    "right_censored",
    "terminal_reason",
    "first_state_row_time",
    "last_state_row_time",
    "successor_state_row_time",
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _require_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"UNIFIED_EXIT_LIFECYCLE_V2_{label}_SHA_INVALID")
    return value


def terminal_state_counts_sha256(
    values: Mapping[tuple[int, int], int | None],
) -> str:
    rows: list[list[int | None]] = []
    for (entry_row, side_index), state_count in sorted(values.items()):
        if (
            isinstance(entry_row, bool)
            or not isinstance(entry_row, int)
            or entry_row < 0
            or isinstance(side_index, bool)
            or not isinstance(side_index, int)
            or side_index not in (0, 1)
            or (
                state_count is not None
                and (
                    isinstance(state_count, bool)
                    or not isinstance(state_count, int)
                    or state_count < 1
                )
            )
        ):
            raise RuntimeError("UNIFIED_EXIT_ECONOMIC_TERMINAL_MAP_INVALID")
        rows.append([entry_row, side_index, state_count])
    if not rows:
        raise RuntimeError("UNIFIED_EXIT_ECONOMIC_TERMINAL_MAP_EMPTY")
    return _canonical_sha256(rows)


def require_unified_exit_economic_lifecycle_authority(
    value: Mapping[str, Any],
    *,
    expected_terminal_state_counts_sha256: str,
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "decision",
        "authority_artifact_path",
        "authority_artifact_sha256",
        "terminal_state_counts_sha256",
        "economic_terminal_definition_sha256",
        "terminal_event_verifier_schema_version",
        "terminal_events_recomputed_from_train_val_only",
        "test_data_used",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise RuntimeError("UNIFIED_EXIT_ECONOMIC_LIFECYCLE_AUTHORITY_INVALID")
    observed = dict(value)
    path = observed["authority_artifact_path"]
    if (
        observed["schema_version"]
        != UNIFIED_EXIT_ECONOMIC_AUTHORITY_SCHEMA_VERSION
        or observed["decision"] != "PASS"
        or not isinstance(path, str)
        or not path.startswith("/")
        or observed["terminal_event_verifier_schema_version"]
        != "gx1_economic_terminal_event_verifier_v1"
        or observed["terminal_events_recomputed_from_train_val_only"] is not True
        or observed["test_data_used"] is not False
        or observed["terminal_state_counts_sha256"]
        != expected_terminal_state_counts_sha256
    ):
        raise RuntimeError("UNIFIED_EXIT_ECONOMIC_LIFECYCLE_AUTHORITY_INVALID")
    for key in (
        "authority_artifact_sha256",
        "terminal_state_counts_sha256",
        "economic_terminal_definition_sha256",
    ):
        _require_sha(observed[key], key.upper())
    return observed


def unified_exit_lifecycle_v2_contract() -> dict[str, Any]:
    payload = {
        "schema_version": UNIFIED_EXIT_LIFECYCLE_V2_SCHEMA_VERSION,
        "allowed_splits": ["train", "val"],
        "test_access": False,
        "chunk_state_capacity": UNIFIED_EXIT_CHUNK_ROWS,
        "detailed_path_tail_bars": UNIFIED_EXIT_CHUNK_ROWS,
        "maximum_trade_duration_bars": None,
        "capacity_forces_exit": False,
        "first_decision_local_history": {
            "pre_entry_rows": 479,
            "first_post_fill_rows": 1,
        },
        "later_chunk_state": "full_causal_prefix_or_exact_model_carry",
        "terminal_reason_values": ["none", "economic_terminal"],
        "right_censor_is_terminal": False,
        "nonterminal_full_chunk_successor_required": True,
        "clock_gap_policy": "right_censor_before_first_non_m1_transition",
        "economic_authority_schema_version": (
            UNIFIED_EXIT_ECONOMIC_AUTHORITY_SCHEMA_VERSION
        ),
        "chunk_schedule": {
            "mode": "outcome_blind_affine_permutation_v1",
            "unit": "entry_pair_timeline",
            "seed_fields": [
                "lineage_sha256",
                "split",
                "entry_row_index",
            ],
            "forbidden_seed_fields": ["reward", "price", "label"],
        },
    }
    payload["contract_sha256"] = _canonical_sha256(payload)
    return payload


def outcome_blind_chunk_permutation(
    *,
    chunk_count: int,
    lineage_sha256: str,
    split: str,
    entry_row_index: int,
    side_index: int,
) -> tuple[int, ...]:
    """Return a deterministic permutation whose seed cannot inspect outcomes."""

    lineage = _require_sha(lineage_sha256, "LINEAGE")
    if (
        isinstance(chunk_count, bool)
        or not isinstance(chunk_count, int)
        or chunk_count < 1
        or split not in {"train", "val"}
        or isinstance(entry_row_index, bool)
        or not isinstance(entry_row_index, int)
        or entry_row_index < 0
        or isinstance(side_index, bool)
        or not isinstance(side_index, int)
        or side_index not in (0, 1)
    ):
        raise RuntimeError("UNIFIED_EXIT_CHUNK_SCHEDULE_IDENTITY_INVALID")
    seed = bytes.fromhex(
        _canonical_sha256(
            {
                "lineage_sha256": lineage,
                "split": split,
                "entry_row_index": entry_row_index,
            }
        )
    )
    a = int.from_bytes(seed[:16], "big") % chunk_count
    while math.gcd(a, chunk_count) != 1:
        a = (a + 1) % chunk_count
    b = int.from_bytes(seed[16:], "big") % chunk_count
    return tuple((a * epoch + b) % chunk_count for epoch in range(chunk_count))


def outcome_blind_chunk_index(
    *,
    chunk_count: int,
    epoch_index: int,
    lineage_sha256: str,
    split: str,
    entry_row_index: int,
    side_index: int,
) -> int:
    """Choose one chunk for an epoch, cycling through the whole trade."""

    if isinstance(epoch_index, bool) or not isinstance(epoch_index, int) or epoch_index < 0:
        raise RuntimeError("UNIFIED_EXIT_CHUNK_SCHEDULE_EPOCH_INVALID")
    permutation = outcome_blind_chunk_permutation(
        chunk_count=chunk_count,
        lineage_sha256=lineage_sha256,
        split=split,
        entry_row_index=entry_row_index,
        side_index=side_index,
    )
    return permutation[epoch_index % chunk_count]


def _utc_m1_times(values: Sequence[Any]) -> pd.DatetimeIndex:
    times = pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors="coerce")).as_unit(
        "ns"
    )
    if (
        times.empty
        or times.hasnans
        or not times.is_unique
        or not times.is_monotonic_increasing
        or not times.floor(f"{EXIT_DECISION_BAR_SECONDS}s").equals(times)
    ):
        raise RuntimeError("UNIFIED_EXIT_LIFECYCLE_V2_M1_CLOCK_INVALID")
    return times


def _chunk_pointer_stream_sha256(
    chunks: pd.DataFrame, times: pd.DatetimeIndex
) -> str:
    clock = np.asarray(times.asi8, dtype=np.int64)
    stream_fields = [
        [
            int(row.chunk_index),
            int(row.entry_row_index),
            int(row.side_index),
            int(row.entry_m1_start_row),
            int(row.chunk_m1_start_row),
            int(row.chunk_start_bars_in_trade),
            int(row.valid_state_count),
            bool(row.successor_available),
            int(row.successor_m1_row),
            bool(row.right_censored),
            str(row.terminal_reason),
        ]
        for row in chunks.itertuples(index=False)
    ]
    return _canonical_sha256(
        {
            "schema_version": UNIFIED_EXIT_CHUNK_POINTER_SCHEMA_VERSION,
            "m1_clock_sha256": hashlib.sha256(
                np.ascontiguousarray(clock, dtype="<i8").tobytes()
            ).hexdigest(),
            "rows": stream_fields,
        }
    )


def build_unified_exit_lifecycle_chunks_v2(
    *,
    entry_m1_start_rows: Sequence[int],
    m1_times: Sequence[Any],
    split: str,
    split_end: Any,
    terminal_state_count_by_entry_side: Mapping[tuple[int, int], int | None],
    economic_lifecycle_authority: Mapping[str, Any],
    m1_source_sha256: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build all bounded pointers for explicitly owned finite/censored trades."""

    if split not in {"train", "val"}:
        raise RuntimeError("UNIFIED_EXIT_LIFECYCLE_V2_SPLIT_FORBIDDEN")
    source_sha = _require_sha(m1_source_sha256, "M1_SOURCE")
    times = _utc_m1_times(m1_times)
    end = pd.Timestamp(split_end)
    if (
        pd.isna(end)
        or end.tz is None
        or end.utcoffset() != pd.Timedelta(0)
    ):
        raise RuntimeError("UNIFIED_EXIT_LIFECYCLE_V2_SPLIT_END_INVALID")
    end = end.as_unit("ns")
    starts = np.asarray(entry_m1_start_rows)
    if (
        starts.ndim != 1
        or starts.size < 1
        or starts.dtype.kind not in "iu"
        or np.any(starts < 0)
        or np.any(starts >= len(times))
    ):
        raise RuntimeError("UNIFIED_EXIT_LIFECYCLE_V2_ENTRY_POINTER_INVALID")
    terminal_hash = terminal_state_counts_sha256(
        terminal_state_count_by_entry_side
    )
    authority = require_unified_exit_economic_lifecycle_authority(
        economic_lifecycle_authority,
        expected_terminal_state_counts_sha256=terminal_hash,
    )
    expected_keys = {
        (entry_row, side_index)
        for entry_row in range(len(starts))
        for side_index in (0, 1)
    }
    if set(terminal_state_count_by_entry_side) != expected_keys:
        raise RuntimeError("UNIFIED_EXIT_ECONOMIC_TERMINAL_POPULATION_INVALID")

    clock = np.asarray(times.asi8, dtype=np.int64)
    delta = int(pd.Timedelta(seconds=EXIT_DECISION_BAR_SECONDS).value)
    rows: list[dict[str, Any]] = []
    chunk_index = 0
    for entry_row, raw_start in enumerate(starts.tolist()):
        start = int(raw_start)
        available_stop = int(
            np.searchsorted(clock, int(end.value) - delta, side="right")
        )
        available_count = max(0, available_stop - start)
        if available_count > 1:
            local_deltas = np.diff(clock[start : start + available_count])
            gap_positions = np.flatnonzero(local_deltas != delta)
            if gap_positions.size:
                available_count = int(gap_positions[0]) + 1
        if available_count < 1:
            continue
        for side_index, side in enumerate(UNIFIED_EXIT_SIDE_ORDER):
            terminal_count = terminal_state_count_by_entry_side[
                (entry_row, side_index)
            ]
            if terminal_count is not None and terminal_count > available_count:
                raise RuntimeError(
                    "UNIFIED_EXIT_ECONOMIC_TERMINAL_OUTSIDE_SPLIT"
                )
            total = terminal_count or available_count
            offset = 0
            while offset < total:
                valid_count = min(UNIFIED_EXIT_CHUNK_ROWS, total - offset)
                after = offset + valid_count
                economic_terminal = terminal_count is not None and after == total
                successor_available = after < total
                right_censored = terminal_count is None and after == total
                successor_row = start + after if successor_available else -1
                rows.append(
                    {
                        "schema_version": UNIFIED_EXIT_LIFECYCLE_V2_SCHEMA_VERSION,
                        "chunk_index": chunk_index,
                        "entry_row_index": entry_row,
                        "side_index": side_index,
                        "side": side,
                        "entry_m1_start_row": start,
                        "chunk_m1_start_row": start + offset,
                        "chunk_start_bars_in_trade": offset,
                        "valid_state_count": valid_count,
                        "successor_available": successor_available,
                        "successor_m1_row": successor_row,
                        "right_censored": right_censored,
                        "terminal_reason": (
                            "economic_terminal" if economic_terminal else "none"
                        ),
                        "first_state_row_time": times[start + offset],
                        "last_state_row_time": times[start + after - 1],
                        "successor_state_row_time": (
                            times[successor_row] if successor_available else pd.NaT
                        ),
                    }
                )
                chunk_index += 1
                offset = after
    chunks = pd.DataFrame(rows, columns=UNIFIED_EXIT_CHUNK_COLUMNS)
    if chunks.empty:
        raise RuntimeError("UNIFIED_EXIT_LIFECYCLE_V2_EMPTY")
    stream_hash = _chunk_pointer_stream_sha256(chunks, times)
    manifest = {
        **unified_exit_lifecycle_v2_contract(),
        "split": split,
        "split_end_utc": end.isoformat(),
        "m1_source_sha256": source_sha,
        "economic_lifecycle_authority": authority,
        "economic_lifecycle_authority_sha256": _canonical_sha256(authority),
        "terminal_state_counts_sha256": terminal_hash,
        "chunk_rows": int(len(chunks)),
        "state_rows": int(chunks["valid_state_count"].sum()),
        "successor_chunk_rows": int(chunks["successor_available"].sum()),
        "right_censored_chunk_rows": int(chunks["right_censored"].sum()),
        "clock_gap_policy": "right_censor_before_first_non_m1_transition",
        "chunk_pointer_stream_schema_version": (
            UNIFIED_EXIT_CHUNK_POINTER_SCHEMA_VERSION
        ),
        "chunk_pointer_stream_sha256": stream_hash,
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    require_unified_exit_lifecycle_chunks_v2(
        chunks,
        manifest=manifest,
        m1_times=times,
    )
    return chunks, manifest


def require_unified_exit_lifecycle_chunks_v2(
    chunks: pd.DataFrame,
    *,
    manifest: Mapping[str, Any],
    m1_times: Sequence[Any],
) -> pd.DataFrame:
    """Recompute chunk continuity, successor and split-isolation invariants."""

    if (
        not isinstance(chunks, pd.DataFrame)
        or tuple(chunks.columns) != UNIFIED_EXIT_CHUNK_COLUMNS
        or chunks.empty
        or not isinstance(manifest, Mapping)
    ):
        raise RuntimeError("UNIFIED_EXIT_LIFECYCLE_V2_CHUNKS_INVALID")
    observed = dict(manifest)
    times = _utc_m1_times(m1_times)
    if (
        observed.get("schema_version") != UNIFIED_EXIT_LIFECYCLE_V2_SCHEMA_VERSION
        or observed.get("split") not in {"train", "val"}
        or observed.get("test_access") is not False
        or observed.get("capacity_forces_exit") is not False
        or observed.get("maximum_trade_duration_bars") is not None
        or observed.get("clock_gap_policy")
        != "right_censor_before_first_non_m1_transition"
        or observed.get("chunk_rows") != len(chunks)
        or observed.get("state_rows") != int(chunks["valid_state_count"].sum())
        or observed.get("chunk_pointer_stream_sha256")
        != _chunk_pointer_stream_sha256(chunks, times)
    ):
        raise RuntimeError("UNIFIED_EXIT_LIFECYCLE_V2_MANIFEST_INVALID")
    split_end = pd.Timestamp(observed["split_end_utc"]).as_unit("ns")
    if observed.get("manifest_sha256") != _canonical_sha256(
        {key: value for key, value in observed.items() if key != "manifest_sha256"}
    ):
        raise RuntimeError("UNIFIED_EXIT_LIFECYCLE_V2_MANIFEST_HASH_INVALID")
    if chunks["schema_version"].ne(UNIFIED_EXIT_LIFECYCLE_V2_SCHEMA_VERSION).any():
        raise RuntimeError("UNIFIED_EXIT_LIFECYCLE_V2_ROW_SCHEMA_INVALID")
    if chunks["chunk_index"].tolist() != list(range(len(chunks))):
        raise RuntimeError("UNIFIED_EXIT_LIFECYCLE_V2_ORDER_INVALID")
    delta = pd.Timedelta(seconds=EXIT_DECISION_BAR_SECONDS)
    for (_entry, _side), group in chunks.groupby(
        ["entry_row_index", "side_index"], sort=False
    ):
        expected_offset = 0
        terminal_seen = False
        for row in group.itertuples(index=False):
            valid_count = int(row.valid_state_count)
            if (
                terminal_seen
                or int(row.chunk_start_bars_in_trade) != expected_offset
                or int(row.chunk_m1_start_row)
                != int(row.entry_m1_start_row) + expected_offset
                or not 1 <= valid_count <= UNIFIED_EXIT_CHUNK_ROWS
                or (
                    str(row.terminal_reason) == "none"
                    and bool(row.successor_available) == bool(row.right_censored)
                )
                or str(row.terminal_reason)
                not in {"none", "economic_terminal"}
                or str(row.side) != UNIFIED_EXIT_SIDE_ORDER[int(row.side_index)]
            ):
                raise RuntimeError("UNIFIED_EXIT_LIFECYCLE_V2_CONTINUITY_INVALID")
            last = int(row.chunk_m1_start_row) + valid_count - 1
            if (
                last >= len(times)
                or times[last] + delta > split_end
                or (
                    valid_count > 1
                    and not bool(
                        np.all(
                            np.diff(
                                np.asarray(
                                    times[
                                        int(row.chunk_m1_start_row) : last + 1
                                    ].asi8,
                                    dtype=np.int64,
                                )
                            )
                            == int(delta.value)
                        )
                    )
                )
                or pd.Timestamp(row.first_state_row_time)
                != times[int(row.chunk_m1_start_row)]
                or pd.Timestamp(row.last_state_row_time) != times[last]
            ):
                raise RuntimeError("UNIFIED_EXIT_LIFECYCLE_V2_SPLIT_LEAK")
            if bool(row.successor_available):
                if (
                    valid_count != UNIFIED_EXIT_CHUNK_ROWS
                    or int(row.successor_m1_row) != last + 1
                    or pd.Timestamp(row.successor_state_row_time) != times[last + 1]
                    or times[last + 1] - times[last] != delta
                    or times[last + 1] + delta > split_end
                ):
                    raise RuntimeError("UNIFIED_EXIT_LIFECYCLE_V2_SUCCESSOR_INVALID")
            elif int(row.successor_m1_row) != -1 or not pd.isna(
                row.successor_state_row_time
            ):
                raise RuntimeError("UNIFIED_EXIT_LIFECYCLE_V2_SUCCESSOR_INVALID")
            terminal_seen = str(row.terminal_reason) == "economic_terminal"
            if terminal_seen and bool(row.right_censored):
                raise RuntimeError("UNIFIED_EXIT_LIFECYCLE_V2_TERMINAL_INVALID")
            expected_offset += valid_count
    return chunks.copy()


__all__ = (
    "UNIFIED_EXIT_CHUNK_COLUMNS",
    "UNIFIED_EXIT_CHUNK_ROWS",
    "UNIFIED_EXIT_ECONOMIC_AUTHORITY_SCHEMA_VERSION",
    "UNIFIED_EXIT_LIFECYCLE_V2_SCHEMA_VERSION",
    "build_unified_exit_lifecycle_chunks_v2",
    "outcome_blind_chunk_index",
    "outcome_blind_chunk_permutation",
    "require_unified_exit_economic_lifecycle_authority",
    "require_unified_exit_lifecycle_chunks_v2",
    "terminal_state_counts_sha256",
    "unified_exit_lifecycle_v2_contract",
)
