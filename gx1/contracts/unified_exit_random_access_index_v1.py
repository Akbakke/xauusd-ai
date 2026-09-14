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

from gx1.contracts.unified_exit_no_cap_economic_authority_v1 import file_sha256

RANDOM_ACCESS_INDEX_SCHEMA_VERSION = "gx1_unified_exit_random_access_index_v1"
RANDOM_ACCESS_INDEX_ROOT_SCHEMA_VERSION = "gx1_unified_exit_random_access_index_root_v1"
RANDOM_ACCESS_INDEX_V2_SCHEMA_VERSION = "gx1_unified_exit_random_access_index_v2"
RANDOM_ACCESS_INDEX_V2_ROOT_SCHEMA_VERSION = (
    "gx1_unified_exit_random_access_index_root_v2"
)
FULL_POPULATION_ROOT_SCHEMA_VERSION = "gx1_unified_exit_random_access_full_population_root_v1"
VAL_REVISION_ROOT_SCHEMA_VERSION = "gx1_unified_exit_random_access_val_revision_root_v1"
LATEST_YEAR_ROOT_SCHEMA_VERSION = "gx1_unified_exit_random_access_latest_year_root_v1"
LATEST_YEAR_WINDOWS = {
    "train": {"start_utc": "2025-06-01T00:00:00+00:00", "end_utc_exclusive": "2026-06-01T00:00:00+00:00"},
    "val": {"start_utc": "2026-06-01T00:00:00+00:00", "end_utc_exclusive": "2026-07-01T00:00:00+00:00"},
}

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
            or file_sha256(resolved)
            != value.get("index_parquet_sha256")
        ):
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_INDEX_FILE_INVALID")
    if verify_sources:
        for binding in sources.values():
            resolved = Path(binding["path"]).expanduser().resolve()
            if (
                not resolved.is_file()
                or resolved.is_symlink()
                or file_sha256(resolved)
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


def require_parent_entry_coordinate_equivalence(
    *, index_manifest: Mapping[str, Any], index_frame: pd.DataFrame,
    expected_split: str, parent_parquet: Mapping[str, str],
    parent_manifest: Mapping[str, str],
) -> dict[str, Any]:
    """Bind a launch-selected parent to the index's exact existing coordinates.

    This proves coordinate identity only. Model inputs and their feature/sequence
    authorities continue to be selected and verified by the immutable launch.
    A changed row count, clock, order or row mapping is never a compatible parent.
    """
    checked = require_random_access_index_manifest(
        index_manifest, expected_split=expected_split, index_frame=index_frame,
        verify_sources=True,
    )
    if checked["schema_version"] != RANDOM_ACCESS_INDEX_V2_SCHEMA_VERSION:
        raise RuntimeError("UNIFIED_EXIT_PARENT_ENTRY_COORDINATES_REQUIRE_V2")
    paths = []
    for binding in (parent_parquet, parent_manifest):
        if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
            raise RuntimeError("UNIFIED_EXIT_PARENT_ENTRY_BINDING_INVALID")
        path = Path(binding["path"])
        if (not path.is_absolute() or path.resolve() != path
                or not path.is_file() or path.is_symlink()):
            raise RuntimeError("UNIFIED_EXIT_PARENT_ENTRY_BINDING_INVALID")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != _sha(binding["sha256"], "PARENT_ENTRY_SOURCE"):
            raise RuntimeError("UNIFIED_EXIT_PARENT_ENTRY_BINDING_INVALID")
        paths.append(path)
    manifest = json.loads(paths[1].read_text())
    if (manifest.get("output_data_path") != str(paths[0])
            or manifest.get("extra", {}).get("pretest_test_guard", {}).get("test_accessed") is not False):
        raise RuntimeError("UNIFIED_EXIT_PARENT_ENTRY_MANIFEST_INVALID")
    clock = _clock(pd.read_parquet(paths[0], columns=["time"])["time"], "LAUNCH_PARENT_ENTRY")
    if (len(clock) != checked["parent_entry_source_rows"]
            or _clock_sha256(clock) != checked["parent_entry_clock_sha256"]):
        raise RuntimeError("UNIFIED_EXIT_PARENT_ENTRY_COORDINATES_DIFFER")
    evidence = {
        "schema_version": "gx1_random_access_parent_entry_coordinates_v1",
        "decision": "PASS_EXACT_COORDINATES",
        "split": expected_split,
        "index_manifest_sha256": checked["manifest_sha256"],
        "recorded_parent_parquet": dict(checked["source_bindings"]["parent_entry_parquet"]),
        "recorded_parent_manifest": dict(checked["source_bindings"]["parent_entry_manifest"]),
        "launch_parent_parquet": dict(parent_parquet),
        "launch_parent_manifest": dict(parent_manifest),
        "parent_row_count": len(clock),
        "entry_row_count": len(index_frame),
        "entire_parent_clock_sha256": _clock_sha256(clock),
        "parent_entry_mapping_sha256": checked["parent_entry_mapping_sha256"],
        "row_coordinates_changed": False,
        "model_input_authority": "separate_immutable_launch_bindings",
        "test_data_used": False,
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    return evidence


def _latest_year_binding(binding: Any, *, verify_file: bool = False) -> Path:
    if (not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}
            or not isinstance(binding.get("path"), str)):
        raise RuntimeError("UNIFIED_EXIT_LATEST_YEAR_BINDING_INVALID")
    path = Path(binding["path"])
    _sha(binding["sha256"], "LATEST_YEAR_BINDING")
    if (not path.is_absolute() or (verify_file and (
            path.is_symlink() or path.resolve() != path or not path.is_file()
            or file_sha256(path) != binding["sha256"]))):
        raise RuntimeError("UNIFIED_EXIT_LATEST_YEAR_BINDING_INVALID")
    return path


def latest_year_selected_entry_rows(frame: pd.DataFrame, *, split: str) -> np.ndarray:
    """Return original child ids in the one fixed chronological TRAIN/VAL scope."""
    if split not in LATEST_YEAR_WINDOWS:
        raise RuntimeError("UNIFIED_EXIT_LATEST_YEAR_SPLIT_INVALID")
    checked = require_random_access_index(frame, expected_split=split)
    window = LATEST_YEAR_WINDOWS[split]
    times = checked["entry_time_ns"].to_numpy(dtype="<i8")
    keep = ((times >= pd.Timestamp(window["start_utc"]).value)
            & (times < pd.Timestamp(window["end_utc_exclusive"]).value))
    return checked.loc[keep, "entry_row_index"].to_numpy(dtype="<i8", copy=True)


def build_latest_year_population(*, source_root_binding: Mapping[str, str]) -> dict[str, Any]:
    """Bind a subset of immutable full-index ids without rewriting any data.

    Full Entry/Exit indices, parent context, normalization and economic fold
    remain exactly the source root's. Only the TRAIN epoch population changes.
    """
    source_path = _latest_year_binding(source_root_binding, verify_file=True)
    source = require_random_access_index_root(json.loads(source_path.read_text(encoding="utf-8")))
    if source["schema_version"] != FULL_POPULATION_ROOT_SCHEMA_VERSION:
        raise RuntimeError("UNIFIED_EXIT_LATEST_YEAR_FULL_ROOT_REQUIRED")
    proofs = {}
    for split, window in LATEST_YEAR_WINDOWS.items():
        binding = source["splits"][split]
        manifest_path = Path(binding["manifest_path"])
        if not manifest_path.is_absolute() or manifest_path.is_symlink() or not manifest_path.is_file():
            raise RuntimeError("UNIFIED_EXIT_LATEST_YEAR_INDEX_MANIFEST_INVALID")
        index_path = _latest_year_binding({"path": binding["index_parquet_path"], "sha256": binding["index_parquet_sha256"]}, verify_file=True)
        frame = pd.read_parquet(index_path)
        manifest = require_random_access_index_manifest(
            json.loads(manifest_path.read_text(encoding="utf-8")), expected_split=split,
            index_frame=frame, index_path=index_path,
        )
        if (manifest["schema_version"] != RANDOM_ACCESS_INDEX_V2_SCHEMA_VERSION
                or manifest["manifest_sha256"] != binding["manifest_sha256"]
                or manifest["entry_row_count"] != binding["entry_row_count"]
                or manifest["successor_transition_total"] != binding["successor_transition_total"]
                or pd.Timestamp(manifest.get("split_end_utc")) != pd.Timestamp(window["end_utc_exclusive"])):
            raise RuntimeError("UNIFIED_EXIT_LATEST_YEAR_INDEX_MANIFEST_INVALID")
        sources = manifest["source_bindings"]
        parent_path = _latest_year_binding(sources["parent_entry_parquet"], verify_file=True)
        parent_manifest_path = _latest_year_binding(sources["parent_entry_manifest"], verify_file=True)
        child_path = _latest_year_binding(sources["entry_parquet"], verify_file=True)
        parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
        parent_clock = _clock(pd.read_parquet(parent_path, columns=["time"])["time"], "LATEST_YEAR_PARENT")
        child_clock = _clock(pd.read_parquet(child_path, columns=["time"])["time"], "LATEST_YEAR_CHILD")
        full_parents = frame["parent_entry_row_index"].to_numpy(dtype="<i8")
        selected_children = latest_year_selected_entry_rows(frame, split=split)
        expected_parents = np.flatnonzero(
            (parent_clock >= pd.Timestamp(window["start_utc"]))
            & (parent_clock < pd.Timestamp(window["end_utc_exclusive"]))
        ).astype("<i8", copy=False)
        selected_parents = full_parents[selected_children]
        selected_times = frame["entry_time_ns"].to_numpy(dtype="<i8")[selected_children]
        if (expected_parents.size == 0 or np.any(full_parents >= len(parent_clock))
                or not np.array_equal(selected_parents, expected_parents)
                or not np.array_equal(selected_times, parent_clock.asi8[expected_parents])
                or not np.array_equal(frame["entry_time_ns"].to_numpy(dtype="<i8"), child_clock.asi8)
                or not np.array_equal(parent_clock.asi8[full_parents], child_clock.asi8)
                or len(parent_clock) != manifest["parent_entry_source_rows"]
                or _clock_sha256(parent_clock) != manifest["parent_entry_clock_sha256"]
                or _clock_sha256(child_clock) != manifest["child_entry_clock_sha256"]
                or hashlib.sha256(np.ascontiguousarray(full_parents).tobytes()).hexdigest() != manifest["parent_entry_row_indices_sha256"]
                or parent_manifest.get("output_data_path") != str(parent_path)
                or parent_manifest.get("extra", {}).get("pretest_test_guard", {}).get("test_accessed") is not False
                or (split == "val" and len(selected_children) != len(frame))):
            raise RuntimeError("UNIFIED_EXIT_LATEST_YEAR_COMPLETE_SELECTION_INVALID")
        if split == "train":
            parent_window = parent_manifest.get("splits", {}).get("train", {})
            parent_end = pd.Timestamp(parent_window.get("end"))
            if (pd.Timestamp(parent_window.get("start")) != pd.Timestamp("2021-06-01T00:00:00Z")
                    or parent_end.tzinfo is None
                    or not pd.Timestamp("2026-05-31T00:00:00Z") <= parent_end <= pd.Timestamp("2026-06-01T00:00:00Z")
                    or not np.array_equal(np.sort(full_parents), np.arange(len(parent_clock), dtype="<i8"))
                    or len(selected_children) >= len(frame)):
                raise RuntimeError("UNIFIED_EXIT_LATEST_YEAR_FULL_PARENT_REQUIRED")
        proofs[split] = {
            **window, "selected_entry_row_count": len(selected_children),
            "index_entry_row_count": len(frame), "parent_entry_source_rows": len(parent_clock),
            "parent_entry_parquet": dict(sources["parent_entry_parquet"]),
            "parent_entry_manifest": dict(sources["parent_entry_manifest"]),
            "selected_child_entry_row_indices_sha256": hashlib.sha256(np.ascontiguousarray(selected_children).tobytes()).hexdigest(),
            "selected_parent_entry_row_indices_sha256": hashlib.sha256(np.ascontiguousarray(selected_parents).tobytes()).hexdigest(),
            "selected_entry_clock_sha256": hashlib.sha256(np.ascontiguousarray(selected_times).tobytes()).hexdigest(),
        }
    return {
        "source_root": dict(source_root_binding), "source_root_sha256": source["root_sha256"],
        "splits": proofs, "selection_uses_outcome_values": False, "test_accessed": False,
    }


def require_latest_year_index_root(
    value: Mapping[str, Any], *, expected_parent_bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recheck complete date selection and exact immutable full-root inheritance."""
    root = require_random_access_index_root(value)
    if root["schema_version"] != LATEST_YEAR_ROOT_SCHEMA_VERSION:
        raise RuntimeError("UNIFIED_EXIT_LATEST_YEAR_ROOT_REQUIRED")
    population = root["latest_year_population"]
    if expected_parent_bindings is not None and (
            set(expected_parent_bindings) != {"train", "val"}
            or any(expected_parent_bindings[split] != {
                "parquet": population["splits"][split]["parent_entry_parquet"],
                "manifest": population["splits"][split]["parent_entry_manifest"],
            } for split in ("train", "val"))):
        raise RuntimeError("UNIFIED_EXIT_LATEST_YEAR_LAUNCH_PARENT_MISMATCH")
    rebuilt = build_latest_year_population(source_root_binding=population["source_root"])
    source = json.loads(Path(population["source_root"]["path"]).read_text(encoding="utf-8"))
    old_common = {k: v for k, v in source.items() if k not in {"schema_version", "root_sha256", "full_train_population"}}
    new_common = {k: v for k, v in root.items() if k not in {"schema_version", "root_sha256", "latest_year_population"}}
    if rebuilt != population or old_common != new_common:
        raise RuntimeError("UNIFIED_EXIT_LATEST_YEAR_POPULATION_DRIFT")
    return root


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
            VAL_REVISION_ROOT_SCHEMA_VERSION,
            FULL_POPULATION_ROOT_SCHEMA_VERSION,
            LATEST_YEAR_ROOT_SCHEMA_VERSION,
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
    if value["schema_version"] == LATEST_YEAR_ROOT_SCHEMA_VERSION:
        population = value.get("latest_year_population")
        if (not isinstance(population, Mapping)
                or set(population) != {"source_root", "source_root_sha256", "splits", "selection_uses_outcome_values", "test_accessed"}
                or population.get("selection_uses_outcome_values") is not False
                or population.get("test_accessed") is not False
                or not isinstance(population.get("splits"), Mapping)
                or set(population["splits"]) != {"train", "val"}
                or any(key in value for key in ("full_train_population", "predecessor_equivalence", "val_data_revision"))):
            raise RuntimeError("UNIFIED_EXIT_LATEST_YEAR_POPULATION_INVALID")
        _latest_year_binding(population["source_root"])
        _sha(population["source_root_sha256"], "LATEST_YEAR_SOURCE_ROOT")
        for split, window in LATEST_YEAR_WINDOWS.items():
            proof = population["splits"][split]
            if (not isinstance(proof, Mapping) or not isinstance(splits[split], Mapping) or set(proof) != {
                    *window, "selected_entry_row_count", "index_entry_row_count", "parent_entry_source_rows",
                    "parent_entry_parquet", "parent_entry_manifest", "selected_child_entry_row_indices_sha256",
                    "selected_parent_entry_row_indices_sha256", "selected_entry_clock_sha256"}
                    or any(proof.get(key) != expected for key, expected in window.items())
                    or any(type(proof.get(key)) is not int or proof[key] < 1 for key in ("selected_entry_row_count", "index_entry_row_count", "parent_entry_source_rows"))
                    or not proof["selected_entry_row_count"] <= proof["index_entry_row_count"] <= proof["parent_entry_source_rows"]
                    or (split == "train" and proof["selected_entry_row_count"] == proof["index_entry_row_count"])
                    or (split == "val" and proof["selected_entry_row_count"] != proof["index_entry_row_count"])
                    or proof["index_entry_row_count"] != splits[split].get("entry_row_count")):
                raise RuntimeError("UNIFIED_EXIT_LATEST_YEAR_POPULATION_INVALID")
            for key in ("parent_entry_parquet", "parent_entry_manifest"):
                _latest_year_binding(proof[key])
            for key in ("selected_child_entry_row_indices_sha256", "selected_parent_entry_row_indices_sha256", "selected_entry_clock_sha256"):
                _sha(proof[key], "LATEST_YEAR_POPULATION")
    elif "latest_year_population" in value:
        raise RuntimeError("UNIFIED_EXIT_LATEST_YEAR_ROOT_REQUIRED")
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
    if value["schema_version"] == FULL_POPULATION_ROOT_SCHEMA_VERSION:
        population = value.get("full_train_population")
        train = splits.get("train")
        if (
            not isinstance(population, Mapping)
            or set(population) != {"entry_row_count", "parent_entry_source_rows", "train_manifest_sha256"}
            or not isinstance(train, Mapping)
            or type(population.get("entry_row_count")) is not int
            or population["entry_row_count"] < 1
            or type(population.get("parent_entry_source_rows")) is not int
            or population["parent_entry_source_rows"] != population["entry_row_count"]
            or population["entry_row_count"] != train.get("entry_row_count")
            or population["train_manifest_sha256"] != train.get("manifest_sha256")
            or "predecessor_equivalence" in value or "val_data_revision" in value
        ):
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_FULL_POPULATION_INVALID")
        _sha(population["train_manifest_sha256"], "FULL_TRAIN_MANIFEST")
    if value["schema_version"] == VAL_REVISION_ROOT_SCHEMA_VERSION:
        revision = value.get("val_data_revision")
        if (
            not isinstance(revision, Mapping)
            or set(revision) != {"predecessor_root", "predecessor_root_sha256", "changed_split"}
            or revision.get("changed_split") != "val"
            or "predecessor_equivalence" in value
        ):
            raise RuntimeError("UNIFIED_EXIT_VAL_REVISION_SCOPE_INVALID")
        predecessor = revision.get("predecessor_root")
        if (not isinstance(predecessor, Mapping)
                or set(predecessor) != {"path", "sha256"}
                or not Path(str(predecessor.get("path", ""))).is_absolute()):
            raise RuntimeError("UNIFIED_EXIT_VAL_REVISION_PREDECESSOR_INVALID")
        _sha(predecessor.get("sha256"), "VAL_REVISION_PREDECESSOR_FILE")
        _sha(revision.get("predecessor_root_sha256"), "VAL_REVISION_PREDECESSOR")
    return dict(value)


def require_val_index_revision_root(
    value: Mapping[str, Any], *, expected_predecessor: Mapping[str, str],
) -> dict[str, Any]:
    """Admit a VAL-only data correction against the immutable seed TRAIN root."""
    from gx1.contracts.unified_exit_no_cap_economic_authority_v1 import file_sha256

    root = require_random_access_index_root(value)
    revision = root.get("val_data_revision", {})
    if (root["schema_version"] != VAL_REVISION_ROOT_SCHEMA_VERSION
            or revision.get("predecessor_root") != dict(expected_predecessor)):
        raise RuntimeError("UNIFIED_EXIT_VAL_REVISION_SEED_INVALID")

    def read(binding: Mapping[str, str]) -> dict[str, Any]:
        path = Path(binding["path"])
        if (not path.is_absolute() or path.is_symlink() or not path.is_file()
                or file_sha256(path) != binding["sha256"]):
            raise RuntimeError("UNIFIED_EXIT_VAL_REVISION_SOURCE_INVALID")
        return json.loads(path.read_text(encoding="utf-8"))

    old = require_random_access_index_root(read(expected_predecessor))
    unchanged = (
        "allowed_splits", "storage_granularity", "full_prefix_states_stored",
        "chunk_pointers_stored", "composite_normalization_sha256",
        "sampler_selection_status", "selected_sampler_contract_sha256", "test_accessed",
    )
    if (old["schema_version"] != RANDOM_ACCESS_INDEX_V2_ROOT_SCHEMA_VERSION
            or revision["predecessor_root_sha256"] != old["root_sha256"]
            or root["splits"]["train"] != old["splits"]["train"]
            or any(root[key] != old[key] for key in unchanged)):
        raise RuntimeError("UNIFIED_EXIT_VAL_REVISION_TRAIN_DRIFT")
    manifests = []
    for current in (old, root):
        split = current["splits"]["val"]
        path = Path(split["manifest_path"])
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise RuntimeError("UNIFIED_EXIT_VAL_REVISION_MANIFEST_INVALID")
        manifest = require_random_access_index_manifest(
            json.loads(path.read_text(encoding="utf-8")), expected_split="val",
        )
        if (split["manifest_sha256"] != manifest["manifest_sha256"]
                or split["index_parquet_sha256"] != manifest["index_parquet_sha256"]
                or split["index_parquet_path"] != manifest["index_parquet_path"]
                or split["entry_row_count"] != manifest["entry_row_count"]
                or split["successor_transition_total"] != manifest["successor_transition_total"]):
            raise RuntimeError("UNIFIED_EXIT_VAL_REVISION_MANIFEST_INVALID")
        manifests.append(manifest)
    before, after = manifests
    changed_sources = {
        "summary_manifest", "successor_counts", "closure_authority",
        "sequence_binding", "first_state_bridge", "final_bindings_bundle",
    }
    old_sources, new_sources = before["source_bindings"], after["source_bindings"]
    if (set(old_sources) != set(new_sources)
            or any(old_sources[name]["sha256"] != new_sources[name]["sha256"]
                   for name in old_sources if name not in changed_sources)
            or after["entry_row_count"] != before["entry_row_count"]
            or any(after[key] != before[key] for key in (
                "parent_entry_row_indices_sha256", "child_entry_clock_sha256",
                "parent_entry_clock_sha256", "parent_entry_source_rows",
            ))
            or after["composite_normalization_sha256"] != old["composite_normalization_sha256"]):
        raise RuntimeError("UNIFIED_EXIT_VAL_REVISION_INPUT_DRIFT")
    old_bundle = read(old_sources["final_bindings_bundle"])
    new_bundle = read(new_sources["final_bindings_bundle"])
    for bundle, current in ((old_bundle, old), (new_bundle, root)):
        if (bundle["bundle_sha256"] != current["final_bindings_bundle_sha256"]
                or bundle["bundle_sha256"] != canonical_sha256(
                    {k: v for k, v in bundle.items() if k != "bundle_sha256"})):
            raise RuntimeError("UNIFIED_EXIT_VAL_REVISION_BUNDLE_INVALID")
    if (any(old_bundle[key] != new_bundle[key]
            for key in ("sampler_benchmark_candidates", "composite_normalization"))
            or any(old_bundle[key]["train"] != new_bundle[key]["train"]
                   for key in ("split_sequence_bindings", "first_state_entry_bridges"))):
        raise RuntimeError("UNIFIED_EXIT_VAL_REVISION_TRAIN_BUNDLE_DRIFT")
    old_recipe = read({"path": old_bundle["recipe_path"], "sha256": old_bundle["recipe_file_sha256"]})
    new_recipe = read({"path": new_bundle["recipe_path"], "sha256": new_bundle["recipe_file_sha256"]})
    def train_recipe(recipe: Mapping[str, Any]) -> dict[str, Any]:
        return {**{k: v for k, v in recipe.items() if k not in {"recipe_sha256", "splits"}},
                "splits": {"train": recipe["splits"]["train"]}}
    if train_recipe(old_recipe) != train_recipe(new_recipe):
        raise RuntimeError("UNIFIED_EXIT_VAL_REVISION_TRAIN_RECIPE_DRIFT")
    return root


__all__ = (
    "RANDOM_ACCESS_INDEX_COLUMNS",
    "RANDOM_ACCESS_INDEX_V1_COLUMNS",
    "RANDOM_ACCESS_INDEX_V2_COLUMNS",
    "RANDOM_ACCESS_INDEX_V2_ROOT_SCHEMA_VERSION",
    "FULL_POPULATION_ROOT_SCHEMA_VERSION",
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
    "require_parent_entry_coordinate_equivalence",
)
