"""Immutable causal M1 lifecycle contract for unified Entry/Exit training."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from gx1.contracts.xau_tape_provenance_v1 import (
    CANONICAL_NATIVE_CLOSURE_CONTRACT,
    CANONICAL_NATIVE_REQUIRED_COLUMNS,
    CANONICAL_NATIVE_SOURCE_SCHEMA,
    CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA,
    canonical_native_rows_sha256,
    canonical_xau_source_descriptor_v1,
)
from gx1.contracts.entry_exit_feature_base_v1 import (
    ENTRY_DECISION_BAR_SECONDS,
    EXIT_DECISION_BAR_SECONDS,
    EXIT_FEATURE_ROW_CLOCK,
    EXIT_FEATURE_SEQUENCE_BARS,
    entry_exit_shared_feature_base_contract,
    require_entry_exit_shared_feature_base_contract,
)
from gx1.contracts.entry_exit_feature_surface_v1 import (
    load_m1_feature_surface,
    require_exact_m1_feature_surface_manifest,
)
from gx1.contracts.entry_exit_production_architecture_v1 import (
    current_entry_exit_architecture_observation,
    require_entry_exit_production_architecture,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.execution.v12_state_from_prebuilt import (
    PREBUILT_PAIR_GENERATION_MANIFEST_FILENAME,
    PREBUILT_PAIR_LINEAGE_SCHEMA_VERSION,
    read_prebuilt_pair_manifest,
    verify_prebuilt_pair,
)
from gx1.models.entry_v10.direction_decision_contract import (
    UNIFIED_EXIT_ACTION_ORDER,
    UNIFIED_EXIT_MAX_PATH_BARS,
    UNIFIED_EXIT_PATH_PRICE_FIELDS,
    UNIFIED_EXIT_SIDE_ORDER,
    unified_exit_path_tensor_from_values,
)
from gx1.io.price_glitch_guard import assert_no_price_scale_glitch


UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION = (
    "gx1_unified_exit_lifecycle_episode_envelope_v10"
)
UNIFIED_EXIT_STATE_SELECTION_SCHEMA_VERSION = (
    "gx1_unified_exit_full_authoritative_state_pointer_population_v2"
)
UNIFIED_EXIT_M1_AUTHORITY_SCHEMA_VERSION = (
    "gx1_unified_exit_native_pair_m1_authority_v1"
)
PRETEST_NATIVE_PAIR_LINEAGE_SCHEMA_VERSION = "gx1_pretest_native_pair_lineage_v3"
PRETEST_DIRECT_NATIVE_SOURCE_SCHEMA_VERSION = "gx1_direct_native_pretest_source_v2"
UNIFIED_EXIT_LIFECYCLE_REQUIRED_M1_COLUMNS = (
    "time",
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
UNIFIED_EXIT_LIFECYCLE_EPISODE_COLUMNS = (
    "schema_version",
    "episode_index",
    "entry_row_index",
    "entry_time",
    "entry_available_at",
    "side_index",
    "side",
    "entry_bid",
    "entry_ask",
    "m1_start_row",
    "m1_start_time",
    "first_state_decision_time",
    "path_state_count",
)
UNIFIED_EXIT_TRAINING_SAMPLES_PER_ENTRY = (
    len(UNIFIED_EXIT_SIDE_ORDER) * UNIFIED_EXIT_MAX_PATH_BARS
)
UNIFIED_EXIT_INVALID_DECISION_TIME_NS = np.iinfo(np.int64).min


def sha256_file(path: Path) -> str:
    resolved = Path(path).expanduser().absolute()
    if resolved.is_symlink() or not resolved.is_file():
        raise RuntimeError(
            f"UNIFIED_EXIT_LIFECYCLE_ARTIFACT_INVALID: {resolved}"
        )
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def unified_exit_state_population_arrays(
    *,
    m1_times: pd.DatetimeIndex,
    m1_start_row: int,
) -> dict[str, np.ndarray]:
    """Build the exact full 512-state population for one episode.

    M1 source timestamps label bar starts.  A decision state becomes available
    only after that authoritative bar has closed, so ``decision_time_ns`` is
    one full M1 bar later than ``state_row_time_ns``.  No target value controls
    state membership. Outcome values never enter this population owner.
    """

    times = pd.DatetimeIndex(m1_times).as_unit("ns")
    count = int(UNIFIED_EXIT_MAX_PATH_BARS)
    if (
        times.empty
        or times.hasnans
        or not times.is_unique
        or not times.is_monotonic_increasing
        or times.tz is None
        or times[0].utcoffset() != pd.Timedelta(0)
        or not times.floor(f"{EXIT_DECISION_BAR_SECONDS}s").equals(times)
        or isinstance(m1_start_row, bool)
        or not isinstance(m1_start_row, (int, np.integer))
    ):
        raise RuntimeError("UNIFIED_EXIT_STATE_POPULATION_INPUT_INVALID")
    start = int(m1_start_row)
    if (
        start < 0
        or start + count > len(times)
    ):
        raise RuntimeError("UNIFIED_EXIT_STATE_POPULATION_INPUT_INVALID")
    state_indices = np.arange(count, dtype=np.int32)
    decision_rows = np.arange(start, start + count, dtype=np.int64)
    state_row_time_ns = np.asarray(times.asi8[decision_rows], dtype=np.int64)
    decision_time_ns = state_row_time_ns + np.int64(
        pd.Timedelta(seconds=EXIT_DECISION_BAR_SECONDS).value
    )
    if np.any(decision_time_ns <= state_row_time_ns):
        raise RuntimeError("UNIFIED_EXIT_STATE_DECISION_CLOCK_INVALID")
    return {
        "state_indices": state_indices,
        "decision_row_indices": decision_rows,
        "state_row_time_ns": state_row_time_ns,
        "decision_time_ns": decision_time_ns,
        "state_valid_mask": np.ones(count, dtype=np.bool_),
    }


def unified_exit_state_pointer_stream_sha256(
    *,
    episode_indices: np.ndarray,
    entry_row_indices: np.ndarray,
    side_indices: np.ndarray,
    m1_start_rows: np.ndarray,
    m1_times: pd.DatetimeIndex,
    chunk_rows: int = 65_536,
) -> str:
    """Hash the exact full-state population through compact source pointers.

    Every episode owns all ``UNIFIED_EXIT_MAX_PATH_BARS`` consecutive states.
    Those state indices, source rows, timestamps, decision clocks and validity
    masks are deterministic functions of ``m1_start_row`` and the immutable M1
    clock. Persisting five 512-element Python lists per episode duplicated that
    deterministic population and made a full TRAIN build unbounded in memory.

    V2 hashes the complete authoritative M1 clock once, then a fixed-width
    little-endian header per episode: episode/entry/side/start/count followed by
    first/last state timestamps and their first/last decision timestamps. The
    M1 artifact hash binds prices and the full clock outside this stream; the
    validator recomputes this stream from that same artifact. No target or
    outcome value participates in membership or hashing.
    """

    raw_arrays = (
        np.asarray(episode_indices),
        np.asarray(entry_row_indices),
        np.asarray(side_indices),
        np.asarray(m1_start_rows),
    )
    if any(array.ndim != 1 for array in raw_arrays):
        raise RuntimeError("UNIFIED_EXIT_STATE_POINTER_STREAM_INPUT_INVALID")
    row_count = len(raw_arrays[0])
    if row_count < 1 or any(len(array) != row_count for array in raw_arrays):
        raise RuntimeError("UNIFIED_EXIT_STATE_POINTER_STREAM_INPUT_INVALID")
    normalized: list[np.ndarray] = []
    for raw in raw_arrays:
        values = np.asarray(raw, dtype=np.int64)
        if raw.dtype.kind not in "iu" or not np.array_equal(raw, values):
            raise RuntimeError("UNIFIED_EXIT_STATE_POINTER_STREAM_INPUT_INVALID")
        normalized.append(values)
    episodes, entries, sides, starts = normalized
    times = pd.DatetimeIndex(m1_times).as_unit("ns")
    count = int(UNIFIED_EXIT_MAX_PATH_BARS)
    if (
        times.empty
        or times.hasnans
        or not times.is_unique
        or not times.is_monotonic_increasing
        or times.tz is None
        or times[0].utcoffset() != pd.Timedelta(0)
        or not times.floor(f"{EXIT_DECISION_BAR_SECONDS}s").equals(times)
        or np.any(episodes < 0)
        or np.any(entries < 0)
        or np.any((sides < 0) | (sides >= len(UNIFIED_EXIT_SIDE_ORDER)))
        or np.any(starts < 0)
        or np.any(starts + count > len(times))
        or isinstance(chunk_rows, bool)
        or not isinstance(chunk_rows, int)
        or chunk_rows < 1
    ):
        raise RuntimeError("UNIFIED_EXIT_STATE_POINTER_STREAM_INPUT_INVALID")

    clock = np.ascontiguousarray(times.asi8, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(UNIFIED_EXIT_STATE_SELECTION_SCHEMA_VERSION.encode("ascii"))
    digest.update(np.asarray([len(clock)], dtype="<i8").tobytes())
    digest.update(clock.tobytes(order="C"))
    decision_delta = np.int64(
        pd.Timedelta(seconds=EXIT_DECISION_BAR_SECONDS).value
    )
    for lo in range(0, row_count, chunk_rows):
        hi = min(lo + chunk_rows, row_count)
        chunk_starts = starts[lo:hi]
        headers = np.empty((hi - lo, 9), dtype="<i8")
        headers[:, 0] = episodes[lo:hi]
        headers[:, 1] = entries[lo:hi]
        headers[:, 2] = sides[lo:hi]
        headers[:, 3] = chunk_starts
        headers[:, 4] = count
        headers[:, 5] = clock[chunk_starts]
        headers[:, 6] = clock[chunk_starts + count - 1]
        headers[:, 7] = headers[:, 5] + decision_delta
        headers[:, 8] = headers[:, 6] + decision_delta
        digest.update(headers.tobytes(order="C"))
    return digest.hexdigest()


def _require_native_m1_subset_identity(
    *,
    source_path: Path,
    native_root: Path,
    native_years: Mapping[str, Any],
    expected_rows: int,
    source_kind: str,
    timeframe: str = "M1",
) -> dict[str, Any]:
    """Prove every declared native-resolution source row is byte-equivalent."""

    if (
        source_kind not in {"base28", "quote_complete_pretest"}
        or timeframe not in {"M1", "M5"}
    ):
        raise RuntimeError("UNIFIED_EXIT_M1_NATIVE_SOURCE_KIND_INVALID")

    base_times_frame = pd.read_parquet(source_path, columns=["time"])
    base_times = pd.DatetimeIndex(
        pd.to_datetime(
            base_times_frame["time"],
            utc=True,
            errors="coerce",
        )
    ).as_unit("ns")
    if (
        len(base_times) != expected_rows
        or base_times.hasnans
        or not base_times.is_unique
        or not base_times.is_monotonic_increasing
    ):
        raise RuntimeError("UNIFIED_EXIT_M1_BASE28_TIME_IDENTITY_INVALID")
    years = sorted(set(base_times.year.tolist()))
    missing = [year for year in years if f"year={year}" not in native_years]
    if missing:
        raise RuntimeError(
            f"UNIFIED_EXIT_M1_NATIVE_YEAR_MISSING: {missing}"
        )
    proof_by_year: dict[str, dict[str, Any]] = {}
    observed_rows = 0
    for year in years:
        start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
        end = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")
        base = pd.read_parquet(
            source_path,
            columns=list(CANONICAL_NATIVE_REQUIRED_COLUMNS),
            filters=[
                ("time", ">=", start.to_pydatetime()),
                ("time", "<", end.to_pydatetime()),
            ],
        )
        native_path = native_root / f"year={year}" / "part-000.parquet"
        native = pd.read_parquet(
            native_path,
            columns=list(CANONICAL_NATIVE_REQUIRED_COLUMNS),
        )
        base_index = pd.DatetimeIndex(
            pd.to_datetime(base["time"], utc=True, errors="coerce")
        ).as_unit("ns")
        native_index = pd.DatetimeIndex(
            pd.to_datetime(native["time"], utc=True, errors="coerce")
        ).as_unit("ns")
        if (
            len(base) == 0
            or base_index.hasnans
            or native_index.hasnans
            or not native_index.is_unique
            or not native_index.is_monotonic_increasing
            or not base_index.isin(native_index).all()
        ):
            raise RuntimeError(
                f"UNIFIED_EXIT_M1_NATIVE_SUBSET_GEOMETRY_INVALID: year={year}"
            )
        native = native.copy()
        native.index = native_index
        matched = native.loc[base_index].reset_index(drop=True)
        base = base.reset_index(drop=True)
        matched["time"] = base_index
        base["time"] = base_index
        base_hash = canonical_native_rows_sha256(base, timeframe=timeframe)
        native_hash = canonical_native_rows_sha256(
            matched,
            timeframe=timeframe,
        )
        if base_hash != native_hash:
            raise RuntimeError(
                f"UNIFIED_EXIT_M1_NATIVE_ROW_IDENTITY_MISMATCH: year={year}"
            )
        observed_rows += len(base)
        proof_by_year[f"year={year}"] = {
            "rows": len(base),
            "canonical_rows_sha256": base_hash,
        }
    if observed_rows != expected_rows:
        raise RuntimeError("UNIFIED_EXIT_M1_NATIVE_SUBSET_ROW_COUNT_MISMATCH")
    return {
        "method": (
            f"exact_{source_kind}_rows_are_native_{timeframe.lower()}_subset_v1"
        ),
        "rows": observed_rows,
        "years": proof_by_year,
        "proof_sha256": canonical_json_sha256(proof_by_year),
    }


def _require_exact_json_object(
    path: Path,
    *,
    expected_keys: set[str],
    context: str,
) -> tuple[Path, dict[str, Any], str]:
    """Load one immutable JSON object with no undeclared contract fields."""

    candidate = Path(path).expanduser()
    if (
        not candidate.is_absolute()
        or candidate.is_symlink()
        or not candidate.is_file()
        or candidate.resolve() != candidate
    ):
        raise RuntimeError(f"{context}_PATH_INVALID")
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{context}_JSON_INVALID") from exc
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise RuntimeError(f"{context}_SCHEMA_INVALID")
    observed_file_sha256 = sha256_file(candidate)
    return candidate, payload, observed_file_sha256


def _require_payload_sha256(
    payload: Mapping[str, Any],
    *,
    key: str,
    context: str,
) -> None:
    """Require the self-declared canonical payload digest without guessing."""

    declared = payload.get(key)
    stripped = dict(payload)
    stripped.pop(key, None)
    if (
        not isinstance(declared, str)
        or len(declared) != 64
        or any(char not in "0123456789abcdef" for char in declared)
        or declared != canonical_json_sha256(stripped)
    ):
        raise RuntimeError(f"{context}_PAYLOAD_SHA256_INVALID")


def _require_pretest_utc_boundary(value: Any, *, context: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value).as_unit("ns")
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{context}_TIMESTAMP_INVALID") from exc
    if (
        pd.isna(parsed)
        or parsed.tz is None
        or parsed.utcoffset() != pd.Timedelta(0)
    ):
        raise RuntimeError(f"{context}_TIMESTAMP_INVALID")
    return parsed


def require_unified_exit_pretest_m1_quote_authority(
    *,
    pair_lineage_path: Path,
    quote_source_manifest_path: Path,
) -> tuple[Path, dict[str, Any]]:
    """Resolve pre-TEST executable M1 quotes through immutable native proof.

    This is deliberately a separate admission path from the historical BASE28
    pair.  It accepts only the exact quote-complete M1 source produced from the
    same TEST-sealed native M1 root named by the V3 pre-TEST pair lineage.
    """

    pair_path, pair, pair_file_sha256 = _require_exact_json_object(
        pair_lineage_path,
        expected_keys={
            "schema_version",
            "pair_generation_id",
            "pair_symbol",
            "test_boundary_utc",
            "test_accessed",
            "lineage",
            "m1",
            "m5",
            "manifest_payload_sha256",
        },
        context="UNIFIED_EXIT_PRETEST_M1_PAIR",
    )
    _require_payload_sha256(
        pair,
        key="manifest_payload_sha256",
        context="UNIFIED_EXIT_PRETEST_M1_PAIR",
    )
    if (
        pair.get("schema_version") != PRETEST_NATIVE_PAIR_LINEAGE_SCHEMA_VERSION
        or pair.get("pair_symbol") != "XAUUSD"
        or pair.get("test_accessed") is not False
    ):
        raise RuntimeError("UNIFIED_EXIT_PRETEST_M1_PAIR_CONTRACT_INVALID")
    pair_generation_id = pair.get("pair_generation_id")
    if (
        not isinstance(pair_generation_id, str)
        or len(pair_generation_id) != 64
        or any(char not in "0123456789abcdef" for char in pair_generation_id)
    ):
        raise RuntimeError("UNIFIED_EXIT_PRETEST_M1_PAIR_GENERATION_INVALID")
    test_boundary = _require_pretest_utc_boundary(
        pair.get("test_boundary_utc"),
        context="UNIFIED_EXIT_PRETEST_M1_PAIR_BOUNDARY",
    )
    lineage = pair.get("lineage")
    m1_binding = pair.get("m1")
    if (
        not isinstance(lineage, Mapping)
        or set(lineage) != {"native_sources"}
        or not isinstance(lineage.get("native_sources"), Mapping)
        or set(lineage["native_sources"]) != {"m1", "m5"}
        or not isinstance(m1_binding, Mapping)
        or set(m1_binding)
        != {
            "native_source",
            "row_count",
            "source_manifest_path",
            "source_manifest_payload_sha256",
            "source_manifest_sha256",
            "source_parquet",
            "source_parquet_sha256",
            "time_max_utc",
            "time_min_utc",
        }
    ):
        raise RuntimeError("UNIFIED_EXIT_PRETEST_M1_PAIR_LINEAGE_INVALID")
    declared_native = lineage["native_sources"]["m1"]
    if (
        not isinstance(declared_native, Mapping)
        or dict(m1_binding["native_source"]) != dict(declared_native)
        or set(declared_native)
        != {
            "root",
            "manifest_path",
            "manifest_sha256",
            "row_count",
            "time_min_utc",
            "time_max_utc",
        }
    ):
        raise RuntimeError("UNIFIED_EXIT_PRETEST_M1_NATIVE_BINDING_INVALID")
    native_root = Path(str(declared_native["root"])).expanduser()
    if not native_root.is_absolute() or native_root.is_symlink():
        raise RuntimeError("UNIFIED_EXIT_PRETEST_M1_NATIVE_ROOT_INVALID")
    observed_native = canonical_xau_source_descriptor_v1(
        native_root,
        timeframe="M1",
    )
    for key, expected in declared_native.items():
        if observed_native.get(key) != expected:
            raise RuntimeError(
                "UNIFIED_EXIT_PRETEST_M1_NATIVE_PAIR_BINDING_MISMATCH: "
                f"field={key}"
            )
    if (
        observed_native.get("schema_version")
        not in {
            CANONICAL_NATIVE_SOURCE_SCHEMA,
            CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA,
        }
        or observed_native.get("completion_field") != "complete"
        or observed_native.get("completion_value") is not True
        or observed_native.get("market_closure_contract")
        != CANONICAL_NATIVE_CLOSURE_CONTRACT
    ):
        raise RuntimeError("UNIFIED_EXIT_PRETEST_M1_COMPLETION_PROOF_INVALID")
    if (
        _require_pretest_utc_boundary(
            observed_native.get("time_max_utc"),
            context="UNIFIED_EXIT_PRETEST_M1_NATIVE_MAX",
        )
        >= test_boundary
    ):
        raise RuntimeError("UNIFIED_EXIT_PRETEST_M1_NATIVE_TEST_BOUNDARY_BREACH")

    quote_path, quote, quote_manifest_file_sha256 = _require_exact_json_object(
        quote_source_manifest_path,
        expected_keys={
            "schema_version",
            "instrument",
            "timeframe",
            "timestamp_semantics",
            "test_boundary_utc",
            "test_accessed",
            "quote_complete_m1",
            "source_native_root",
            "source_native_manifest_path",
            "source_native_manifest_sha256",
            "source_native_manifest_payload_sha256",
            "source_requested_start_utc",
            "source_requested_end_utc_exclusive",
            "time_min_utc",
            "time_max_utc",
            "row_count",
            "output_columns",
            "output_parquet",
            "output_parquet_sha256",
            "producer_git_commit",
            "producer_repository_clean",
            "manifest_payload_sha256",
        },
        context="UNIFIED_EXIT_PRETEST_M1_QUOTES",
    )
    _require_payload_sha256(
        quote,
        key="manifest_payload_sha256",
        context="UNIFIED_EXIT_PRETEST_M1_QUOTES",
    )
    required_columns = list(UNIFIED_EXIT_LIFECYCLE_REQUIRED_M1_COLUMNS)
    if (
        quote.get("schema_version") != PRETEST_DIRECT_NATIVE_SOURCE_SCHEMA_VERSION
        or quote.get("instrument") != "XAU_USD"
        or quote.get("timeframe") != "M1"
        or quote.get("timestamp_semantics") != "bar_start_utc"
        or quote.get("test_accessed") is not False
        or quote.get("quote_complete_m1") is not True
        or quote.get("output_columns") != required_columns
        or quote.get("source_native_root") != str(native_root)
        or quote.get("source_native_manifest_path")
        != observed_native.get("manifest_path")
        or quote.get("source_native_manifest_sha256")
        != observed_native.get("manifest_sha256")
        or quote.get("source_native_manifest_payload_sha256")
        != observed_native.get("manifest_payload_sha256")
        or quote.get("row_count") != declared_native["row_count"]
        or quote.get("time_min_utc") != declared_native["time_min_utc"]
        or quote.get("time_max_utc") != declared_native["time_max_utc"]
        or _require_pretest_utc_boundary(
            quote.get("test_boundary_utc"),
            context="UNIFIED_EXIT_PRETEST_M1_QUOTES_BOUNDARY",
        )
        != test_boundary
        or _require_pretest_utc_boundary(
            quote.get("source_requested_end_utc_exclusive"),
            context="UNIFIED_EXIT_PRETEST_M1_QUOTES_END",
        )
        != test_boundary
    ):
        raise RuntimeError("UNIFIED_EXIT_PRETEST_M1_QUOTES_CONTRACT_INVALID")
    source_path = Path(str(quote["output_parquet"])).expanduser()
    if (
        not source_path.is_absolute()
        or source_path.is_symlink()
        or not source_path.is_file()
        or source_path.resolve() != source_path
        or sha256_file(source_path) != quote.get("output_parquet_sha256")
    ):
        raise RuntimeError("UNIFIED_EXIT_PRETEST_M1_QUOTES_SOURCE_INVALID")
    native_subset_proof = _require_native_m1_subset_identity(
        source_path=source_path,
        native_root=native_root,
        native_years=observed_native["year_sha256"],
        expected_rows=int(quote["row_count"]),
        source_kind="quote_complete_pretest",
    )
    authority = {
        "schema_version": UNIFIED_EXIT_M1_AUTHORITY_SCHEMA_VERSION,
        "authority_mode": "pretest_quote_complete_native_v1",
        "pair_manifest_path": str(pair_path),
        "pair_manifest_sha256": pair_file_sha256,
        "pair_generation_root": None,
        "pair_generation_id": pair_generation_id,
        "pair_lineage_schema_version": pair["schema_version"],
        "m1_source_path": str(source_path),
        "m1_source_sha256": quote["output_parquet_sha256"],
        "m1_source_rows": quote["row_count"],
        "m1_source_manifest_path": str(quote_path),
        "m1_source_manifest_sha256": quote_manifest_file_sha256,
        "native_m1_root": str(native_root),
        "native_m1_manifest_path": str(observed_native["manifest_path"]),
        "native_m1_manifest_sha256": observed_native["manifest_sha256"],
        "native_m1_canonical_rows_sha256": observed_native[
            "canonical_rows_sha256"
        ],
        "native_m1_source_chunks_sha256": observed_native[
            "source_chunks_sha256"
        ],
        "native_m1_producer_source_inventory_sha256": observed_native[
            "producer_source_inventory_sha256"
        ],
        "native_m1_completion_field": observed_native["completion_field"],
        "native_m1_completion_value": observed_native["completion_value"],
        "native_m1_market_closure_contract": observed_native[
            "market_closure_contract"
        ],
        "native_m1_requested_end_utc_exclusive": observed_native[
            "requested_end_utc_exclusive"
        ],
        "native_m1_time_max_utc": observed_native["time_max_utc"],
        "test_accessed": False,
        "test_boundary_utc": str(test_boundary.isoformat()),
        "base28_native_m1_subset_proof": native_subset_proof,
    }
    return source_path, authority


def require_pretest_m5_quote_authority(
    *,
    pair_lineage_path: Path,
    quote_source_manifest_path: Path,
    expected_pair_generation_id: str,
) -> tuple[Path, dict[str, Any]]:
    """Bind the TEST-sealed M5 executable-quote tape used by Entry labels."""

    pair_path, pair, pair_file_sha256 = _require_exact_json_object(
        pair_lineage_path,
        expected_keys={
            "schema_version",
            "pair_generation_id",
            "pair_symbol",
            "test_boundary_utc",
            "test_accessed",
            "lineage",
            "m1",
            "m5",
            "manifest_payload_sha256",
        },
        context="PRETEST_M5_PAIR",
    )
    _require_payload_sha256(
        pair,
        key="manifest_payload_sha256",
        context="PRETEST_M5_PAIR",
    )
    pair_generation_id = pair.get("pair_generation_id")
    if (
        pair.get("schema_version") != PRETEST_NATIVE_PAIR_LINEAGE_SCHEMA_VERSION
        or pair.get("pair_symbol") != "XAUUSD"
        or pair.get("test_accessed") is not False
        or pair_generation_id != expected_pair_generation_id
    ):
        raise RuntimeError("PRETEST_M5_PAIR_CONTRACT_INVALID")
    test_boundary = _require_pretest_utc_boundary(
        pair.get("test_boundary_utc"),
        context="PRETEST_M5_PAIR_BOUNDARY",
    )
    lineage = pair.get("lineage")
    m5_binding = pair.get("m5")
    if (
        not isinstance(lineage, Mapping)
        or set(lineage) != {"native_sources"}
        or not isinstance(lineage.get("native_sources"), Mapping)
        or set(lineage["native_sources"]) != {"m1", "m5"}
        or not isinstance(m5_binding, Mapping)
        or set(m5_binding)
        != {
            "native_source",
            "row_count",
            "source_manifest_path",
            "source_manifest_payload_sha256",
            "source_manifest_sha256",
            "source_parquet",
            "source_parquet_sha256",
            "time_max_utc",
            "time_min_utc",
        }
    ):
        raise RuntimeError("PRETEST_M5_PAIR_LINEAGE_INVALID")
    declared_native = lineage["native_sources"]["m5"]
    if (
        not isinstance(declared_native, Mapping)
        or dict(m5_binding["native_source"]) != dict(declared_native)
        or set(declared_native)
        != {
            "root",
            "manifest_path",
            "manifest_sha256",
            "row_count",
            "time_min_utc",
            "time_max_utc",
        }
    ):
        raise RuntimeError("PRETEST_M5_NATIVE_BINDING_INVALID")
    native_root = Path(str(declared_native["root"])).expanduser()
    if not native_root.is_absolute() or native_root.is_symlink():
        raise RuntimeError("PRETEST_M5_NATIVE_ROOT_INVALID")
    observed_native = canonical_xau_source_descriptor_v1(
        native_root,
        timeframe="M5",
    )
    for key, expected in declared_native.items():
        if observed_native.get(key) != expected:
            raise RuntimeError(
                "PRETEST_M5_NATIVE_PAIR_BINDING_MISMATCH: "
                f"field={key}"
            )
    if (
        observed_native.get("completion_field") != "complete"
        or observed_native.get("completion_value") is not True
        or observed_native.get("market_closure_contract")
        != CANONICAL_NATIVE_CLOSURE_CONTRACT
        or _require_pretest_utc_boundary(
            observed_native.get("time_max_utc"),
            context="PRETEST_M5_NATIVE_MAX",
        )
        >= test_boundary
    ):
        raise RuntimeError("PRETEST_M5_COMPLETION_PROOF_INVALID")

    quote_path, quote, quote_manifest_file_sha256 = _require_exact_json_object(
        quote_source_manifest_path,
        expected_keys={
            "schema_version",
            "instrument",
            "timeframe",
            "timestamp_semantics",
            "test_boundary_utc",
            "test_accessed",
            "quote_complete_m1",
            "source_native_root",
            "source_native_manifest_path",
            "source_native_manifest_sha256",
            "source_native_manifest_payload_sha256",
            "source_requested_start_utc",
            "source_requested_end_utc_exclusive",
            "time_min_utc",
            "time_max_utc",
            "row_count",
            "output_columns",
            "output_parquet",
            "output_parquet_sha256",
            "producer_git_commit",
            "producer_repository_clean",
            "manifest_payload_sha256",
        },
        context="PRETEST_M5_QUOTES",
    )
    _require_payload_sha256(
        quote,
        key="manifest_payload_sha256",
        context="PRETEST_M5_QUOTES",
    )
    if (
        quote.get("schema_version") != PRETEST_DIRECT_NATIVE_SOURCE_SCHEMA_VERSION
        or quote.get("instrument") != "XAU_USD"
        or quote.get("timeframe") != "M5"
        or quote.get("timestamp_semantics") != "bar_start_utc"
        or quote.get("test_accessed") is not False
        or quote.get("quote_complete_m1") is not False
        or quote.get("output_columns")
        != list(UNIFIED_EXIT_LIFECYCLE_REQUIRED_M1_COLUMNS)
        or quote.get("source_native_root") != str(native_root)
        or quote.get("source_native_manifest_path")
        != observed_native.get("manifest_path")
        or quote.get("source_native_manifest_sha256")
        != observed_native.get("manifest_sha256")
        or quote.get("source_native_manifest_payload_sha256")
        != observed_native.get("manifest_payload_sha256")
        or quote.get("row_count") != declared_native["row_count"]
        or quote.get("time_min_utc") != declared_native["time_min_utc"]
        or quote.get("time_max_utc") != declared_native["time_max_utc"]
        or _require_pretest_utc_boundary(
            quote.get("test_boundary_utc"),
            context="PRETEST_M5_QUOTES_BOUNDARY",
        )
        != test_boundary
        or _require_pretest_utc_boundary(
            quote.get("source_requested_end_utc_exclusive"),
            context="PRETEST_M5_QUOTES_END",
        )
        != test_boundary
    ):
        raise RuntimeError("PRETEST_M5_QUOTES_CONTRACT_INVALID")
    source_path = Path(str(quote["output_parquet"])).expanduser()
    if (
        not source_path.is_absolute()
        or source_path.is_symlink()
        or not source_path.is_file()
        or source_path.resolve() != source_path
        or sha256_file(source_path) != quote.get("output_parquet_sha256")
    ):
        raise RuntimeError("PRETEST_M5_QUOTES_SOURCE_INVALID")
    native_subset_proof = _require_native_m1_subset_identity(
        source_path=source_path,
        native_root=native_root,
        native_years=observed_native["year_sha256"],
        expected_rows=int(quote["row_count"]),
        source_kind="quote_complete_pretest",
        timeframe="M5",
    )
    return source_path, {
        "schema_version": "gx1_pretest_m5_quote_authority_v1",
        "pair_manifest_path": str(pair_path),
        "pair_manifest_sha256": pair_file_sha256,
        "pair_generation_id": pair_generation_id,
        "m5_source_path": str(source_path),
        "m5_source_sha256": quote["output_parquet_sha256"],
        "m5_source_rows": quote["row_count"],
        "m5_source_manifest_path": str(quote_path),
        "m5_source_manifest_sha256": quote_manifest_file_sha256,
        "test_accessed": False,
        "test_boundary_utc": str(test_boundary.isoformat()),
        "native_m5_market_closure_contract": observed_native[
            "market_closure_contract"
        ],
        "native_m5_subset_proof": native_subset_proof,
    }


def require_unified_exit_m1_pair_authority(
    *,
    pair_manifest_path: Path,
    pair_generation_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Resolve exact closed-M1 bytes only through immutable native/pair proof."""

    manifest_arg = Path(pair_manifest_path).expanduser()
    generation_arg = Path(pair_generation_root).expanduser()
    if (
        not manifest_arg.is_absolute()
        or manifest_arg.is_symlink()
        or not manifest_arg.is_file()
        or manifest_arg.resolve() != manifest_arg
        or not generation_arg.is_absolute()
        or generation_arg.is_symlink()
        or not generation_arg.is_dir()
        or generation_arg.resolve() != generation_arg
    ):
        raise RuntimeError("UNIFIED_EXIT_M1_PAIR_AUTHORITY_PATH_INVALID")
    binding = read_prebuilt_pair_manifest(
        manifest_arg,
        generation_root=generation_arg,
    )
    expected_generation_manifest = (
        generation_arg
        / binding.pair_generation_id
        / PREBUILT_PAIR_GENERATION_MANIFEST_FILENAME
    )
    if (
        binding.generation_manifest_path is None
        or binding.manifest_path != expected_generation_manifest
        or binding.generation_manifest_path != expected_generation_manifest
    ):
        raise RuntimeError(
            "UNIFIED_EXIT_M1_MUTABLE_PAIR_POINTER_FORBIDDEN: "
            "use the generation-local PAIR_MANIFEST.json"
        )
    _canonical_verified, base28_verified = verify_prebuilt_pair(binding)
    lineage = binding.lineage
    if lineage.get("schema_version") != PREBUILT_PAIR_LINEAGE_SCHEMA_VERSION:
        raise RuntimeError("UNIFIED_EXIT_M1_PAIR_LINEAGE_SCHEMA_INVALID")
    declared_native = lineage["native_sources"]["m1"]
    native_root = Path(str(declared_native["root"]))
    observed_native = canonical_xau_source_descriptor_v1(
        native_root,
        timeframe="M1",
    )
    for key, expected in declared_native.items():
        if observed_native.get(key) != expected:
            raise RuntimeError(
                "UNIFIED_EXIT_M1_NATIVE_PAIR_BINDING_MISMATCH: "
                f"field={key}"
            )
    if (
        observed_native.get("schema_version")
        not in {
            CANONICAL_NATIVE_SOURCE_SCHEMA,
            CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA,
        }
        or observed_native.get("completion_field") != "complete"
        or observed_native.get("completion_value") is not True
        or observed_native.get("market_closure_contract")
        != CANONICAL_NATIVE_CLOSURE_CONTRACT
    ):
        raise RuntimeError("UNIFIED_EXIT_M1_COMPLETION_PROOF_INVALID")
    source_path = base28_verified.binding.parquet_path
    native_subset_proof = _require_native_m1_subset_identity(
        source_path=source_path,
        native_root=native_root,
        native_years=observed_native["year_sha256"],
        expected_rows=base28_verified.binding.rows,
        source_kind="base28",
    )
    authority = {
        "schema_version": UNIFIED_EXIT_M1_AUTHORITY_SCHEMA_VERSION,
        "pair_manifest_path": str(binding.manifest_path),
        "pair_manifest_sha256": binding.manifest_sha256,
        "pair_generation_root": str(generation_arg),
        "pair_generation_id": binding.pair_generation_id,
        "pair_lineage_schema_version": lineage["schema_version"],
        "m1_source_path": str(source_path),
        "m1_source_sha256": base28_verified.binding.parquet_sha256,
        "m1_source_rows": base28_verified.binding.rows,
        "native_m1_root": str(native_root),
        "native_m1_manifest_path": str(observed_native["manifest_path"]),
        "native_m1_manifest_sha256": observed_native["manifest_sha256"],
        "native_m1_canonical_rows_sha256": observed_native[
            "canonical_rows_sha256"
        ],
        "native_m1_source_chunks_sha256": observed_native[
            "source_chunks_sha256"
        ],
        "native_m1_producer_source_inventory_sha256": observed_native[
            "producer_source_inventory_sha256"
        ],
        "native_m1_completion_field": observed_native["completion_field"],
        "native_m1_completion_value": observed_native["completion_value"],
        "native_m1_market_closure_contract": observed_native[
            "market_closure_contract"
        ],
        "native_m1_requested_end_utc_exclusive": observed_native[
            "requested_end_utc_exclusive"
        ],
        "native_m1_time_max_utc": observed_native["time_max_utc"],
        "base28_native_m1_subset_proof": native_subset_proof,
    }
    return source_path, authority


def require_unified_exit_lifecycle_authority_evidence(
    value: Any,
) -> dict[str, Any]:
    """Validate the native-M1 proof embedded in trained lifecycle evidence."""

    if not isinstance(value, Mapping):
        raise RuntimeError("UNIFIED_EXIT_LIFECYCLE_EVIDENCE_MISSING")
    authority = value.get("m1_authority")
    if (
        value.get("schema_version")
        != UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION
        or value.get("future_outcomes_used_as_model_inputs") is not False
        or value.get("sample_selection_depends_on_future_target") is not False
        or value.get("training_population")
        != UNIFIED_EXIT_STATE_SELECTION_SCHEMA_VERSION
        or value.get("validation_population") != "all_authoritative_states"
        or value.get("test_population") != "all_authoritative_states"
        or value.get("exit_supervision_authority")
        != "executable_exit_now_reward_plus_train_fitted_q"
        or not isinstance(authority, Mapping)
        or authority.get("schema_version")
        != UNIFIED_EXIT_M1_AUTHORITY_SCHEMA_VERSION
        or value.get("m1_authority_sha256")
        != canonical_json_sha256(authority)
        or value.get("m1_source_path") != authority.get("m1_source_path")
        or value.get("m1_source_sha256")
        != authority.get("m1_source_sha256")
        or value.get("extra_lookahead_beyond_trajectory") != 0
    ):
        raise RuntimeError("UNIFIED_EXIT_LIFECYCLE_AUTHORITY_EVIDENCE_INVALID")
    authority_mode = authority.get("authority_mode")
    expected_subset_method = (
        "exact_quote_complete_pretest_rows_are_native_m1_subset_v1"
        if authority_mode == "pretest_quote_complete_native_v1"
        else "exact_base28_rows_are_native_m1_subset_v1"
    )
    if authority_mode not in {None, "pretest_quote_complete_native_v1"}:
        raise RuntimeError("UNIFIED_EXIT_M1_AUTHORITY_MODE_INVALID")
    if authority_mode == "pretest_quote_complete_native_v1" and (
        authority.get("test_accessed") is not False
        or _require_pretest_utc_boundary(
            authority.get("test_boundary_utc"),
            context="UNIFIED_EXIT_PRETEST_M1_EVIDENCE_BOUNDARY",
        )
        <= _require_pretest_utc_boundary(
            authority.get("native_m1_time_max_utc"),
            context="UNIFIED_EXIT_PRETEST_M1_EVIDENCE_MAX",
        )
    ):
        raise RuntimeError("UNIFIED_EXIT_PRETEST_M1_AUTHORITY_EVIDENCE_INVALID")
    subset = authority.get("base28_native_m1_subset_proof")
    if (
        not isinstance(subset, Mapping)
        or subset.get("method")
        != expected_subset_method
        or isinstance(subset.get("rows"), bool)
        or not isinstance(subset.get("rows"), int)
        or int(subset["rows"]) <= 0
        or not isinstance(subset.get("years"), Mapping)
        or subset.get("proof_sha256")
        != canonical_json_sha256(subset["years"])
    ):
        raise RuntimeError("UNIFIED_EXIT_M1_NATIVE_SUBSET_EVIDENCE_INVALID")
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _read_exact_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"UNIFIED_EXIT_LIFECYCLE_JSON_INVALID: {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"UNIFIED_EXIT_LIFECYCLE_JSON_OBJECT_REQUIRED: {path}")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    observed = set(value)
    if observed != expected:
        raise RuntimeError(
            f"{context}_SCHEMA_MISMATCH: "
            f"missing={sorted(expected - observed)} "
            f"unexpected={sorted(observed - expected)}"
        )


def _m1_feature_surface_rows(manifest_path) -> int:
    """Row count the Exit feature surface actually publishes."""

    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise RuntimeError("UNIFIED_EXIT_M1_FEATURE_SURFACE_ROWS_INVALID")
    return rows


def _validated_m1_arrays(
    source_path: Path,
) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    frame = pd.read_parquet(
        source_path,
        columns=list(UNIFIED_EXIT_LIFECYCLE_REQUIRED_M1_COLUMNS),
    )
    if tuple(frame.columns) != UNIFIED_EXIT_LIFECYCLE_REQUIRED_M1_COLUMNS:
        raise RuntimeError("UNIFIED_EXIT_M1_SOURCE_SCHEMA_MISMATCH")
    assert_no_price_scale_glitch(
        frame,
        context="UNIFIED_EXIT_LIFECYCLE_M1_SOURCE",
    )
    times = pd.DatetimeIndex(
        pd.to_datetime(frame.pop("time"), utc=True, errors="coerce")
    ).as_unit("ns")
    if (
        len(times) == 0
        or times.hasnans
        or not times.is_unique
        or not times.is_monotonic_increasing
        or not times.floor(f"{EXIT_DECISION_BAR_SECONDS}s").equals(times)
    ):
        raise RuntimeError("UNIFIED_EXIT_M1_TIME_GEOMETRY_INVALID")
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("UNIFIED_EXIT_M1_SOURCE_NONFINITE")
    for prefix in ("", "bid_", "ask_"):
        open_values = numeric[f"{prefix}open"].to_numpy(dtype=np.float64)
        high_values = numeric[f"{prefix}high"].to_numpy(dtype=np.float64)
        low_values = numeric[f"{prefix}low"].to_numpy(dtype=np.float64)
        close_values = numeric[f"{prefix}close"].to_numpy(dtype=np.float64)
        if (
            np.any(open_values <= 0.0)
            or np.any(high_values <= 0.0)
            or np.any(low_values <= 0.0)
            or np.any(close_values <= 0.0)
            or np.any(high_values < np.maximum(open_values, close_values))
            or np.any(low_values > np.minimum(open_values, close_values))
            or np.any(low_values > high_values)
        ):
            raise RuntimeError(
                f"UNIFIED_EXIT_M1_OHLC_GEOMETRY_INVALID: {prefix or 'mid_'}"
            )
    for suffix in ("open", "high", "low", "close"):
        if np.any(
            numeric[f"ask_{suffix}"].to_numpy(dtype=np.float64)
            <= numeric[f"bid_{suffix}"].to_numpy(dtype=np.float64)
        ):
            raise RuntimeError(
                f"UNIFIED_EXIT_M1_EXECUTABLE_SPREAD_INVALID: {suffix}"
            )
    volume = numeric["volume"].to_numpy(dtype=np.float64)
    if np.any(volume < 0.0) or not np.equal(volume, np.floor(volume)).all():
        raise RuntimeError("UNIFIED_EXIT_M1_VOLUME_INVALID")
    arrays = {
        name: numeric[name].to_numpy(dtype=np.float64, copy=True)
        for name in numeric.columns
    }
    return times, arrays


class UnifiedExitLifecycleSplit:
    """Validated episodes containing every authoritative lifecycle state."""

    def __init__(
        self,
        *,
        split: str,
        entry_row_count: int,
        entry_times: pd.DatetimeIndex,
        feature_row_offset: int,
        episodes: pd.DataFrame,
        split_manifest: Mapping[str, Any],
        m1_times: pd.DatetimeIndex,
        m1_arrays: Mapping[str, np.ndarray],
        m1_feature_times: pd.DatetimeIndex,
        m1_feature_arrays: Mapping[str, np.ndarray],
    ) -> None:
        architecture = current_entry_exit_architecture_observation()
        architecture["exit"]["max_path_bars"] = UNIFIED_EXIT_MAX_PATH_BARS
        require_entry_exit_production_architecture(
            architecture,
            context="UNIFIED_EXIT_LIFECYCLE_SPLIT_CONSTRUCTION",
        )
        self.split = str(split)
        self.entry_row_count = int(entry_row_count)
        parsed_entry_times = pd.DatetimeIndex(entry_times).as_unit("ns")
        if (
            len(parsed_entry_times) != self.entry_row_count
            or parsed_entry_times.empty
            or parsed_entry_times.hasnans
            or not parsed_entry_times.is_unique
            or not parsed_entry_times.is_monotonic_increasing
            or parsed_entry_times.tz is None
            or parsed_entry_times[0].utcoffset() != pd.Timedelta(0)
            or not parsed_entry_times.floor(
                f"{ENTRY_DECISION_BAR_SECONDS}s"
            ).equals(parsed_entry_times)
        ):
            raise RuntimeError(
                f"UNIFIED_EXIT_LIFECYCLE_ENTRY_TIME_POPULATION_INVALID: "
                f"{self.split}"
            )
        self._entry_times = parsed_entry_times
        self._m1_times = m1_times
        self._m1 = dict(m1_arrays)
        # The canonical native tape names midpoint OHLC ``open/high/low/close``.
        # The path tensor uses explicit ``mid_*`` ownership to distinguish it
        # from executable bid/ask.  Bind those names to the exact same arrays;
        # never reconstruct midpoint from bid/ask.
        for suffix in ("open", "high", "low", "close"):
            source_name = suffix
            target_name = f"mid_{suffix}"
            source_values = self._m1.get(source_name)
            if not isinstance(source_values, np.ndarray):
                raise RuntimeError(
                    f"UNIFIED_EXIT_M1_MID_{suffix.upper()}_SOURCE_MISSING"
                )
            self._m1[target_name] = source_values
        # The Exit feature surface begins later than the source clock, so a
        # source row r is feature row r - feature_row_offset. The surface must
        # be exactly the tail of the source clock from that offset; anything
        # else is a mis-alignment and fails closed.
        self._feature_row_offset = int(feature_row_offset)
        if self._feature_row_offset < 0 or not m1_feature_times.equals(
            m1_times[self._feature_row_offset :]
        ):
            raise RuntimeError(
                f"UNIFIED_EXIT_M1_FEATURE_SURFACE_TIME_MISMATCH: {self.split}"
            )
        self._m1_feature_times = m1_feature_times
        self._m1_features = dict(m1_feature_arrays)
        for name, width in (
            ("signal", MODEL_NATIVE_SIGNAL_DIM),
            ("ctx_cont", MODEL_NATIVE_CTX_CONT_DIM),
            ("ctx_cat", MODEL_NATIVE_CTX_CAT_DIM),
        ):
            values = self._m1_features.get(name)
            # The surface is the tail of the source clock from the declared
            # offset, verified against m1_feature_times just above, so its row
            # count is the source row count minus that offset.
            if not isinstance(values, np.ndarray) or values.shape != (
                len(m1_times) - self._feature_row_offset,
                width,
            ):
                raise RuntimeError(
                    f"UNIFIED_EXIT_M1_FEATURE_SURFACE_{name.upper()}_SHAPE_INVALID"
                )
        self._validate_full_population(episodes, split_manifest)
        self._validate_complete_eligible_entry_population(
            episodes,
            split_manifest,
        )

    def _validate_complete_eligible_entry_population(
        self,
        episodes: pd.DataFrame,
        manifest: Mapping[str, Any],
    ) -> None:
        """Recompute the causal eligibility set; an episode list is not authority.

        A valid compact episode proves every state *inside* that episode, but
        cannot by itself prove that a producer did not silently omit another
        fully serviceable Entry row. Reconstruct that exact set from the
        immutable Entry clock, M1 clock and declared split boundary before a
        trainer is allowed to treat ``None`` as a legitimate ineligible row.
        No reward, target or model output participates in this calculation.
        """

        raw_split_end = manifest.get("split_end_utc")
        try:
            split_end = pd.Timestamp(raw_split_end).as_unit("ns")
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"UNIFIED_EXIT_LIFECYCLE_SPLIT_END_INVALID: {self.split}"
            ) from exc
        if (
            pd.isna(split_end)
            or split_end.tz is None
            or split_end.utcoffset() != pd.Timedelta(0)
        ):
            raise RuntimeError(
                f"UNIFIED_EXIT_LIFECYCLE_SPLIT_END_INVALID: {self.split}"
            )

        m1_ns = np.asarray(self._m1_times.asi8, dtype=np.int64)
        entry_available_ns = np.asarray(
            self._entry_times.asi8
            + int(pd.Timedelta(seconds=ENTRY_DECISION_BAR_SECONDS).value),
            dtype=np.int64,
        )
        start_rows = np.searchsorted(m1_ns, entry_available_ns, side="left")
        exact_open = start_rows < len(m1_ns)
        exact_positions = np.flatnonzero(exact_open)
        exact_open[exact_positions] &= (
            m1_ns[start_rows[exact_positions]]
            == entry_available_ns[exact_positions]
        )
        path_state_count = int(UNIFIED_EXIT_MAX_PATH_BARS)
        insufficient_tail = exact_open & (
            start_rows + path_state_count > len(m1_ns)
        )
        complete_tail = exact_open & ~insufficient_tail
        crosses_split_end = np.zeros(self.entry_row_count, dtype=np.bool_)
        complete_positions = np.flatnonzero(complete_tail)
        crosses_split_end[complete_positions] = (
            m1_ns[start_rows[complete_positions] + path_state_count - 1]
            + int(pd.Timedelta(seconds=EXIT_DECISION_BAR_SECONDS).value)
            > int(split_end.value)
        )
        eligible = complete_tail & ~crosses_split_end
        expected_rows = np.flatnonzero(eligible).astype(np.int64, copy=False)
        expected_starts = np.asarray(
            start_rows[expected_rows], dtype=np.int64
        )
        feature_floor = self._feature_row_offset + EXIT_FEATURE_SEQUENCE_BARS - 1
        if expected_starts.size and int(expected_starts.min()) < feature_floor:
            raise RuntimeError(
                f"UNIFIED_EXIT_LIFECYCLE_ELIGIBILITY_BEFORE_FEATURE_FLOOR: "
                f"{self.split}"
            )

        observed_rows = episodes["entry_row_index"].to_numpy(
            dtype=np.int64
        )[0::2]
        observed_starts = episodes["m1_start_row"].to_numpy(
            dtype=np.int64
        )[0::2]
        if (
            not np.array_equal(observed_rows, expected_rows)
            or not np.array_equal(observed_starts, expected_starts)
        ):
            raise RuntimeError(
                f"UNIFIED_EXIT_LIFECYCLE_ELIGIBLE_ENTRY_POPULATION_MISMATCH: "
                f"{self.split}"
            )
        expected_skipped = {
            "missing_entry_available_m1_open": int(
                np.count_nonzero(~exact_open)
            ),
            "insufficient_m1_tail": int(np.count_nonzero(insufficient_tail)),
            "crosses_split_end": int(np.count_nonzero(crosses_split_end)),
        }
        if (
            manifest.get("entry_rows") != self.entry_row_count
            or manifest.get("eligible_entry_rows") != len(expected_rows)
            or manifest.get("skipped_entry_rows") != expected_skipped
        ):
            raise RuntimeError(
                f"UNIFIED_EXIT_LIFECYCLE_ELIGIBILITY_PROOF_INVALID: "
                f"{self.split}"
            )

    def _validate_full_population(
        self,
        episodes: pd.DataFrame,
        manifest: Mapping[str, Any],
    ) -> None:
        if tuple(episodes.columns) != UNIFIED_EXIT_LIFECYCLE_EPISODE_COLUMNS:
            raise RuntimeError(
                f"UNIFIED_EXIT_LIFECYCLE_EPISODE_COLUMNS_INVALID: {self.split}"
            )
        if (
            len(episodes) == 0
            or not (
                episodes["schema_version"]
                == UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION
            ).all()
        ):
            raise RuntimeError(
                f"UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_INVALID: {self.split}"
            )
        def exact_integer_column(name: str) -> np.ndarray:
            raw = episodes[name].to_numpy()
            values = np.asarray(raw, dtype=np.int64)
            if (
                raw.ndim != 1
                or raw.dtype.kind not in "iu"
                or not np.array_equal(raw, values)
            ):
                raise RuntimeError(
                    f"UNIFIED_EXIT_LIFECYCLE_EPISODE_INTEGER_INVALID: "
                    f"{self.split}.{name}"
                )
            return values

        episode_indices = exact_integer_column("episode_index")
        entry_indices = exact_integer_column("entry_row_index")
        side_indices = exact_integer_column("side_index")
        starts = exact_integer_column("m1_start_row")
        state_counts = exact_integer_column("path_state_count")
        row_count = len(episodes)
        expected_sides = np.tile(
            np.arange(len(UNIFIED_EXIT_SIDE_ORDER), dtype=np.int64),
            row_count // len(UNIFIED_EXIT_SIDE_ORDER),
        )
        if not np.array_equal(
            episode_indices, np.arange(row_count, dtype=np.int64)
        ):
            raise RuntimeError(
                f"UNIFIED_EXIT_LIFECYCLE_EPISODE_ORDER_INVALID: {self.split}"
            )
        if (
            row_count % len(UNIFIED_EXIT_SIDE_ORDER) != 0
            or np.any((entry_indices < 0) | (entry_indices >= self.entry_row_count))
            or not np.array_equal(side_indices, expected_sides)
            or not np.array_equal(entry_indices[0::2], entry_indices[1::2])
            or np.any(np.diff(entry_indices[0::2]) <= 0)
            or not np.array_equal(starts[0::2], starts[1::2])
            or np.any(starts < 0)
            or np.any(starts + UNIFIED_EXIT_MAX_PATH_BARS > len(self._m1_times))
            or np.any(state_counts != UNIFIED_EXIT_MAX_PATH_BARS)
        ):
            raise RuntimeError(
                f"UNIFIED_EXIT_LIFECYCLE_EPISODE_VALUE_INVALID: {self.split}"
            )
        observed_sides = episodes["side"].to_numpy(dtype=object)
        expected_side_names = np.asarray(UNIFIED_EXIT_SIDE_ORDER, dtype=object)[
            side_indices
        ]
        if not np.array_equal(observed_sides, expected_side_names):
            raise RuntimeError(
                f"UNIFIED_EXIT_LIFECYCLE_ENTRY_SIDE_POPULATION_INVALID: {self.split}"
            )

        def timestamp_ns(name: str) -> np.ndarray:
            parsed = pd.DatetimeIndex(
                pd.to_datetime(episodes[name], utc=True, errors="coerce")
            ).as_unit("ns")
            if parsed.hasnans:
                raise RuntimeError(
                    f"UNIFIED_EXIT_LIFECYCLE_POINTER_TIME_MISMATCH: "
                    f"{self.split}.{name}"
                )
            return np.asarray(parsed.asi8, dtype=np.int64)

        start_time_ns = np.asarray(self._m1_times.asi8[starts], dtype=np.int64)
        decision_delta_ns = int(
            pd.Timedelta(seconds=EXIT_DECISION_BAR_SECONDS).value
        )
        if (
            not np.array_equal(timestamp_ns("m1_start_time"), start_time_ns)
            or not np.array_equal(
                timestamp_ns("entry_available_at"), start_time_ns
            )
            or not np.array_equal(
                timestamp_ns("first_state_decision_time"),
                start_time_ns + decision_delta_ns,
            )
            or not np.array_equal(
                timestamp_ns("entry_time")
                + int(pd.Timedelta(seconds=ENTRY_DECISION_BAR_SECONDS).value),
                start_time_ns,
            )
        ):
            raise RuntimeError(
                f"UNIFIED_EXIT_LIFECYCLE_POINTER_TIME_MISMATCH: {self.split}"
            )
        entry_bid = pd.to_numeric(
            episodes["entry_bid"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        entry_ask = pd.to_numeric(
            episodes["entry_ask"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        expected_bid = np.asarray(self._m1["bid_open"][starts], dtype=np.float64)
        expected_ask = np.asarray(self._m1["ask_open"][starts], dtype=np.float64)
        if (
            not np.isfinite(entry_bid).all()
            or not np.isfinite(entry_ask).all()
            or not np.array_equal(entry_bid, expected_bid)
            or not np.array_equal(entry_ask, expected_ask)
        ):
            raise RuntimeError(
                f"UNIFIED_EXIT_LIFECYCLE_ENTRY_QUOTE_MISMATCH: {self.split}"
            )
        self._episode_pointers = {
            (int(entry_index), int(side_index)): (
                int(episode_index),
                int(start),
                float(bid),
                float(ask),
            )
            for episode_index, entry_index, side_index, start, bid, ask in zip(
                episode_indices,
                entry_indices,
                side_indices,
                starts,
                expected_bid,
                expected_ask,
                strict=True,
            )
        }
        state_population_sha256 = unified_exit_state_pointer_stream_sha256(
            episode_indices=episode_indices,
            entry_row_indices=entry_indices,
            side_indices=side_indices,
            m1_start_rows=starts,
            m1_times=self._m1_times,
        )
        side_population_counts = (
            np.bincount(
                side_indices,
                minlength=len(UNIFIED_EXIT_SIDE_ORDER),
            ).astype(np.int64)
            * int(UNIFIED_EXIT_MAX_PATH_BARS)
        )
        if int(manifest.get("episode_rows", -1)) != len(episodes):
            raise RuntimeError(
                f"UNIFIED_EXIT_LIFECYCLE_EPISODE_ROWS_MISMATCH: {self.split}"
            )
        expected_population_rows = len(episodes) * UNIFIED_EXIT_MAX_PATH_BARS
        if (
            manifest.get("state_population_schema_version")
            != UNIFIED_EXIT_STATE_SELECTION_SCHEMA_VERSION
            or manifest.get("state_population")
            != "all_authoritative_states_both_sides_every_complete_episode"
            or manifest.get("state_population_per_episode")
            != UNIFIED_EXIT_MAX_PATH_BARS
            or manifest.get("state_population_rows") != expected_population_rows
            or manifest.get("state_population_stream_sha256")
            != state_population_sha256
            or manifest.get("first_state_pre_entry_history_rows")
            != EXIT_FEATURE_SEQUENCE_BARS - 1
            or manifest.get("first_state_post_fill_closed_bars") != 1
            or manifest.get("sample_selection_depends_on_future_target") is not False
            or manifest.get("path_values_duplicated_into_episode_artifact") is not False
            or manifest.get("state_vectors_duplicated_into_episode_artifact") is not False
            or int(side_population_counts.sum()) != expected_population_rows
            or np.any(side_population_counts <= 0)
        ):
            raise RuntimeError(
                f"UNIFIED_EXIT_LIFECYCLE_FULL_POPULATION_PROOF_INVALID: {self.split}"
            )
        self.state_side_counts = {
            UNIFIED_EXIT_SIDE_ORDER[index]: int(side_population_counts[index])
            for index in range(2)
        }
        self.state_population_rows = expected_population_rows
        self.state_population_sha256 = state_population_sha256

    def _full_current_indices(self) -> np.ndarray:
        intervals = sorted(
            {
                (int(pointer[1]), int(pointer[1]) + UNIFIED_EXIT_MAX_PATH_BARS)
                for pointer in self._episode_pointers.values()
            }
        )
        merged: list[tuple[int, int]] = []
        for left, right in intervals:
            if not merged or left > merged[-1][1]:
                merged.append((left, right))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        if not merged:
            raise RuntimeError("UNIFIED_EXIT_FULL_CURRENT_POPULATION_EMPTY")
        return np.concatenate(
            [np.arange(left, right, dtype=np.int64) for left, right in merged]
        )

    def authoritative_current_decision_times_ns(self) -> np.ndarray:
        """Return every unique full-population bar-close decision clock."""

        return self.authoritative_current_state_row_times_ns() + int(
            pd.Timedelta(seconds=EXIT_DECISION_BAR_SECONDS).value
        )

    def authoritative_current_state_row_times_ns(self) -> np.ndarray:
        """Return every unique full-population M1 state-bar start clock."""

        current_indices = self._full_current_indices()
        return np.asarray(
            self._m1_times.asi8[current_indices],
            dtype=np.int64,
        )

    def selected_current_decision_times_ns(self) -> np.ndarray:
        """Compatibility spelling; the returned population is no longer sampled."""

        return self.authoritative_current_decision_times_ns()

    def train_normalization_population(self) -> dict[str, Any]:
        """Expose exact unique physical M1 rows consumed by TRAIN Exit.

        The returned feature matrices remain the lifecycle-owned disk-backed
        arrays.  Only sorted int64 row selections are materialized, so the
        normalization fit can scan them repeatedly without copying the full
        full owner-declared signal/context surfaces into RAM.
        """

        if self.split != "train":
            raise RuntimeError(
                "UNIFIED_EXIT_NORMALIZATION_TRAIN_SPLIT_REQUIRED"
            )
        current_indices = self._full_current_indices()
        if current_indices.size < 1:
            raise RuntimeError(
                "UNIFIED_EXIT_NORMALIZATION_CURRENT_POPULATION_EMPTY"
            )
        local_left = current_indices - int(EXIT_FEATURE_SEQUENCE_BARS) + 1
        if int(local_left[0]) < 0:
            raise RuntimeError(
                "UNIFIED_EXIT_NORMALIZATION_M1_HISTORY_INSUFFICIENT"
            )

        merged: list[tuple[int, int]] = []
        for left, current in zip(local_left.tolist(), current_indices.tolist()):
            right = int(current) + 1
            if not merged or int(left) > merged[-1][1]:
                merged.append((int(left), right))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        local_count = sum(right - left for left, right in merged)
        local_indices = np.empty(local_count, dtype=np.int64)
        offset = 0
        for left, right in merged:
            count = right - left
            local_indices[offset : offset + count] = np.arange(
                left,
                right,
                dtype=np.int64,
            )
            offset += count
        if (
            offset != local_count
            or local_indices.size < current_indices.size
            or np.any(np.diff(local_indices) <= 0)
            or not np.isin(current_indices, local_indices).all()
        ):
            raise RuntimeError(
                "UNIFIED_EXIT_NORMALIZATION_LOCAL_POPULATION_INVALID"
            )
        # This exports the FEATURE population, so every array leaves in feature
        # coordinates. The row indices are computed against the source clock,
        # which starts _feature_row_offset rows earlier; returning them unshifted
        # beside the feature matrices would index the matrices at the wrong rows
        # while the timestamps stayed right.
        offset = self._feature_row_offset
        local_feature_indices = local_indices - offset
        current_feature_indices = current_indices - offset
        if offset and (
            local_feature_indices.size
            and int(local_feature_indices.min()) < 0
            or current_feature_indices.size
            and int(current_feature_indices.min()) < 0
        ):
            raise RuntimeError(
                "UNIFIED_EXIT_NORMALIZATION_ROW_BEFORE_FEATURE_SURFACE"
            )
        return {
            "signal": self._m1_features["signal"],
            "ctx_cont": self._m1_features["ctx_cont"],
            "ctx_cat": self._m1_features["ctx_cat"],
            "local_row_indices": local_feature_indices,
            "current_row_indices": current_feature_indices,
            "current_decision_times_ns": np.asarray(
                self._m1_times.asi8[current_indices]
                + int(pd.Timedelta(seconds=EXIT_DECISION_BAR_SECONDS).value),
                dtype=np.int64,
            ),
            "source_times_ns": np.asarray(
                self._m1_times.asi8[offset:], dtype=np.int64
            ),
            "local_merged_intervals": tuple(merged),
        }

    def sample(self, entry_row_index: int) -> dict[str, np.ndarray]:
        raise RuntimeError(
            "UNIFIED_EXIT_STATE_SAMPLE_RETIRED: consume "
            "materialize_causal_episode_core"
        )

    def materialize_causal_episode_core(
        self,
        entry_row_index: int,
    ) -> dict[str, Any] | None:
        """Materialize one non-repeated local/path/target episode pair.

        The local feature timeline contains 479 pre-entry owner rows followed
        by exactly 512 newly closed M1 rows.  Both side paths are complete and
        unpadded.  MTF unique histories are attached by the dataset cache owner.
        """

        if (
            isinstance(entry_row_index, bool)
            or not isinstance(entry_row_index, (int, np.integer))
        ):
            raise RuntimeError("UNIFIED_EXIT_EPISODE_REQUEST_INVALID")
        index = int(entry_row_index)
        if not 0 <= index < self.entry_row_count:
            raise RuntimeError("UNIFIED_EXIT_EPISODE_REQUEST_INVALID")
        pointers = tuple(
            self._episode_pointers.get((index, side_index))
            for side_index in (0, 1)
        )
        if pointers == (None, None):
            return None
        if any(pointer is None for pointer in pointers):
            raise RuntimeError("UNIFIED_EXIT_EPISODE_PAIR_MISSING")
        long_pointer = pointers[0]
        short_pointer = pointers[1]
        if long_pointer is None or short_pointer is None:
            raise RuntimeError("UNIFIED_EXIT_EPISODE_PAIR_MISSING")
        if (
            int(long_pointer[1]) != int(short_pointer[1])
            or float(long_pointer[2]) != float(short_pointer[2])
            or float(long_pointer[3]) != float(short_pointer[3])
        ):
            raise RuntimeError("UNIFIED_EXIT_EPISODE_SIDE_SOURCE_SPLIT_BRAIN")
        start = int(long_pointer[1])
        warm_rows = EXIT_FEATURE_SEQUENCE_BARS - 1
        local_source_start = start - warm_rows
        local_source_stop = start + UNIFIED_EXIT_MAX_PATH_BARS
        local_feature_start = local_source_start - self._feature_row_offset
        local_feature_stop = local_source_stop - self._feature_row_offset
        if local_feature_start < 0 or local_feature_stop > len(
            self._m1_feature_times
        ):
            raise RuntimeError(
                "UNIFIED_EXIT_EPISODE_M1_FEATURE_HISTORY_INSUFFICIENT"
            )
        source_slice = slice(start, local_source_stop)
        price_arrays = tuple(
            self._m1[name] for name in UNIFIED_EXIT_PATH_PRICE_FIELDS
        )
        if len(price_arrays) != len(UNIFIED_EXIT_PATH_PRICE_FIELDS):
            raise RuntimeError("UNIFIED_EXIT_EPISODE_PATH_LAYOUT_INVALID")
        path_by_side = []
        for side_index, pointer in enumerate((long_pointer, short_pointer)):
            del side_index
            path_by_side.append(
                unified_exit_path_tensor_from_values(
                    price_values=np.column_stack(
                        [values[source_slice] for values in price_arrays]
                    ),
                    volumes=self._m1["volume"][source_slice],
                    bars_in_trade=UNIFIED_EXIT_MAX_PATH_BARS,
                    entry_bid=float(pointer[2]),
                    entry_ask=float(pointer[3]),
                )
            )
        long_exit_reward = (
            np.asarray(self._m1["bid_close"][source_slice], dtype=np.float64)
            - float(long_pointer[3])
        ) / float(long_pointer[3]) * 10_000.0
        short_exit_reward = (
            float(short_pointer[2])
            - np.asarray(self._m1["ask_close"][source_slice], dtype=np.float64)
        ) / float(short_pointer[2]) * 10_000.0
        exit_now_reward = np.stack(
            [long_exit_reward, short_exit_reward], axis=0
        )
        state_valid = np.ones(
            (2, UNIFIED_EXIT_MAX_PATH_BARS), dtype=np.bool_
        )
        terminal = np.zeros_like(state_valid)
        terminal[:, -1] = True
        action_valid = np.repeat(state_valid[..., None], 2, axis=2)
        action_valid[..., 0] &= ~terminal
        terminal_reason = np.zeros_like(state_valid, dtype=np.int64)
        terminal_reason[:, -1] = 1
        if not np.isfinite(exit_now_reward).all():
            raise RuntimeError("UNIFIED_EXIT_EPISODE_EXIT_REWARD_NONFINITE")
        local_times = np.asarray(
            self._m1_feature_times.asi8[
                local_feature_start:local_feature_stop
            ],
            dtype=np.int64,
        )
        state_times = np.asarray(
            self._m1_times.asi8[start:local_source_stop], dtype=np.int64
        )
        return {
            "entry_row_index": index,
            "episode_index_by_side": [
                int(long_pointer[0]),
                int(short_pointer[0]),
            ],
            "m1_start_row": start,
            "lifecycle_state_population_sha256": self.state_population_sha256,
            "exit_local_history_x": np.ascontiguousarray(
                self._m1_features["signal"][
                    local_feature_start:local_feature_stop
                ],
                dtype=np.float32,
            ),
            "exit_local_history_time_ns": local_times,
            "exit_state_ctx_cont": np.ascontiguousarray(
                self._m1_features["ctx_cont"][
                    start
                    - self._feature_row_offset : local_source_stop
                    - self._feature_row_offset
                ],
                dtype=np.float32,
            ),
            "exit_state_ctx_cat": np.ascontiguousarray(
                self._m1_features["ctx_cat"][
                    start
                    - self._feature_row_offset : local_source_stop
                    - self._feature_row_offset
                ],
                dtype=np.int64,
            ),
            "exit_state_row_time_ns": state_times,
            "exit_decision_time_ns": np.ascontiguousarray(
                state_times
                + int(pd.Timedelta(seconds=EXIT_DECISION_BAR_SECONDS).value),
                dtype=np.int64,
            ),
            "exit_path_x": np.ascontiguousarray(
                np.stack(path_by_side, axis=0), dtype=np.float32
            ),
            "exit_entry_bid_ask": np.asarray(
                [
                    [float(long_pointer[2]), float(long_pointer[3])],
                    [float(short_pointer[2]), float(short_pointer[3])],
                ],
                dtype=np.float64,
            ),
            "exit_now_reward_bps": np.ascontiguousarray(
                exit_now_reward, dtype=np.float32
            ),
            "exit_action_valid_mask": np.ascontiguousarray(
                action_valid, dtype=np.bool_
            ),
            "exit_state_valid_mask": np.ascontiguousarray(
                state_valid, dtype=np.bool_
            ),
            "exit_terminal_mask": np.ascontiguousarray(
                terminal, dtype=np.bool_
            ),
            "exit_terminal_reason_index": np.ascontiguousarray(
                terminal_reason, dtype=np.int64
            ),
            "exit_episode_lengths": np.full(2, UNIFIED_EXIT_MAX_PATH_BARS, dtype=np.int64),
        }

class UnifiedExitLifecycleCorpus:
    """Load and cryptographically validate one immutable lifecycle directory."""

    def __init__(
        self,
        *,
        root_manifest_path: Path,
        entry_parquets: Mapping[str, Path],
        dataset_run_id: str,
        splits: Sequence[str] = ("train", "val"),
    ) -> None:
        architecture = current_entry_exit_architecture_observation()
        architecture["exit"]["max_path_bars"] = UNIFIED_EXIT_MAX_PATH_BARS
        require_entry_exit_production_architecture(
            architecture,
            context="UNIFIED_EXIT_LIFECYCLE_CORPUS_CONSTRUCTION",
        )
        selected_splits = tuple(splits)
        if (
            not selected_splits
            or len(selected_splits) != len(set(selected_splits))
            or any(
                split not in {"train", "val", "test"}
                for split in selected_splits
            )
        ):
            raise RuntimeError("UNIFIED_EXIT_LIFECYCLE_SELECTED_SPLITS_INVALID")
        manifest_path = Path(root_manifest_path).expanduser().absolute()
        if (
            not manifest_path.is_absolute()
            or manifest_path.is_symlink()
            or not manifest_path.is_file()
            or manifest_path.name != "UNIFIED_EXIT_LIFECYCLE_MANIFEST.json"
        ):
            raise RuntimeError(
                "UNIFIED_EXIT_LIFECYCLE_ROOT_MANIFEST_INVALID"
            )
        root = manifest_path.parent
        # A pre-freeze TRAIN/VAL consumer must not enumerate, stat, hash or
        # parse the sealed TEST artifacts.  The immutable root manifest and
        # the separately verified TEST seal bind TEST lineage; only a TEST
        # consumer may materialise the TEST lifecycle itself.  In particular,
        # do not turn this inventory check into an accidental TEST path-stat.
        root_manifest = _read_exact_json(manifest_path)
        _require_exact_keys(
            root_manifest,
            {
                "schema_version",
                "decision",
                "entry_run_id",
                "m1_source_path",
                "m1_source_sha256",
                "m1_feature_base_path",
                "m1_feature_base_sha256",
                "m1_feature_base_manifest_path",
                "m1_feature_base_manifest_sha256",
                "m1_authority",
                "m1_authority_sha256",
                "path_state_count",
                "state_population_schema_version",
                "state_population_per_episode",
                "m1_row_clock",
                "shared_feature_base_contract",
                "side_order",
                "action_order",
                "splits",
            },
            context="UNIFIED_EXIT_LIFECYCLE_ROOT",
        )
        # The pre-TEST authority deliberately materializes no TEST lifecycle
        # files.  A trainer using that authority may only consume the two
        # physical TRAIN/VAL bindings.  Legacy/post-seal authority retains the
        # complete three-split inventory.  This is a root-manifest rule, not
        # merely an iteration choice: admitting a missing TEST split under the
        # legacy authority would weaken its sealed inventory contract.
        raw_authority = root_manifest["m1_authority"]
        pretest_authority = (
            isinstance(raw_authority, Mapping)
            and raw_authority.get("authority_mode")
            == "pretest_quote_complete_native_v1"
        )
        expected_root_splits = (
            {"train", "val"}
            if pretest_authority
            else {"train", "val", "test"}
        )
        if (
            root_manifest["schema_version"]
            != UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION
            or root_manifest["decision"] != "PASS"
            or root_manifest["entry_run_id"] != dataset_run_id
            or root_manifest["path_state_count"] != UNIFIED_EXIT_MAX_PATH_BARS
            or root_manifest["state_population_schema_version"]
            != UNIFIED_EXIT_STATE_SELECTION_SCHEMA_VERSION
            or root_manifest["state_population_per_episode"]
            != UNIFIED_EXIT_MAX_PATH_BARS
            or root_manifest["m1_row_clock"] != EXIT_FEATURE_ROW_CLOCK
            or root_manifest["side_order"] != list(UNIFIED_EXIT_SIDE_ORDER)
            or root_manifest["action_order"] != list(UNIFIED_EXIT_ACTION_ORDER)
            or not isinstance(root_manifest["splits"], Mapping)
            or set(root_manifest["splits"]) != expected_root_splits
        ):
            raise RuntimeError("UNIFIED_EXIT_LIFECYCLE_ROOT_CONTRACT_INVALID")
        require_entry_exit_shared_feature_base_contract(
            root_manifest["shared_feature_base_contract"],
            context="UNIFIED_EXIT_LIFECYCLE_ROOT",
        )
        if (
            not isinstance(raw_authority, dict)
            or raw_authority.get("schema_version")
            != UNIFIED_EXIT_M1_AUTHORITY_SCHEMA_VERSION
            or canonical_json_sha256(raw_authority)
            != root_manifest["m1_authority_sha256"]
        ):
            raise RuntimeError("UNIFIED_EXIT_M1_AUTHORITY_EVIDENCE_INVALID")
        if pretest_authority:
            # The strict pre-TEST authority has no mutable generation root;
            # its M1 source is instead bound through the explicit immutable
            # quote-source manifest.  Do not coerce the intentional null to
            # a Path, and do not fall back to the legacy BASE28 admission.
            m1_path_from_authority, observed_authority = (
                require_unified_exit_pretest_m1_quote_authority(
                    pair_lineage_path=Path(
                        str(raw_authority["pair_manifest_path"])
                    ),
                    quote_source_manifest_path=Path(
                        str(raw_authority["m1_source_manifest_path"])
                    ),
                )
            )
        else:
            m1_path_from_authority, observed_authority = (
                require_unified_exit_m1_pair_authority(
                    pair_manifest_path=Path(raw_authority["pair_manifest_path"]),
                    pair_generation_root=Path(
                        raw_authority["pair_generation_root"]
                    ),
                )
            )
        if observed_authority != raw_authority:
            raise RuntimeError("UNIFIED_EXIT_M1_AUTHORITY_REVALIDATION_MISMATCH")
        m1_path = Path(root_manifest["m1_source_path"]).expanduser().absolute()
        if (
            m1_path != m1_path_from_authority
            or not m1_path.is_absolute()
            or m1_path.is_symlink()
            or not m1_path.is_file()
            or sha256_file(m1_path) != root_manifest["m1_source_sha256"]
        ):
            raise RuntimeError("UNIFIED_EXIT_LIFECYCLE_M1_IDENTITY_INVALID")
        m1_feature_path = Path(
            root_manifest["m1_feature_base_path"]
        ).expanduser().absolute()
        if (
            not m1_feature_path.is_absolute()
            or m1_feature_path.is_symlink()
            or not m1_feature_path.is_file()
            or sha256_file(m1_feature_path)
            != root_manifest["m1_feature_base_sha256"]
        ):
            raise RuntimeError("UNIFIED_EXIT_M1_FEATURE_BASE_IDENTITY_INVALID")
        m1_feature_manifest_path = Path(
            root_manifest["m1_feature_base_manifest_path"]
        ).expanduser().absolute()
        if (
            not m1_feature_manifest_path.is_absolute()
            or m1_feature_manifest_path.is_symlink()
            or not m1_feature_manifest_path.is_file()
            or sha256_file(m1_feature_manifest_path)
            != root_manifest["m1_feature_base_manifest_sha256"]
        ):
            raise RuntimeError("UNIFIED_EXIT_M1_FEATURE_BASE_MANIFEST_IDENTITY_INVALID")
        authority_pair_generation_id = raw_authority.get("pair_generation_id")
        authority_m1_source_rows = raw_authority.get("m1_source_rows")
        if (
            not isinstance(authority_pair_generation_id, str)
            or not authority_pair_generation_id
            or isinstance(authority_m1_source_rows, bool)
            or not isinstance(authority_m1_source_rows, int)
            or authority_m1_source_rows <= 0
        ):
            raise RuntimeError("UNIFIED_EXIT_M1_AUTHORITY_SURFACE_BINDING_INVALID")
        require_exact_m1_feature_surface_manifest(
            manifest_path=m1_feature_manifest_path,
            expected_manifest_sha256=root_manifest[
                "m1_feature_base_manifest_sha256"
            ],
            expected_parquet_path=m1_feature_path,
            expected_parquet_sha256=root_manifest[
                "m1_feature_base_sha256"
            ],
            expected_dataset_run_id=dataset_run_id,
            expected_pair_generation_id=authority_pair_generation_id,
            # The Exit surface begins after the D1 warmup the M1 lane trims and
            # after the price layer's causal warmup, so it carries fewer rows
            # than the pair authority declares. The rows it does carry are bound
            # by the parquet and manifest hashes checked immediately above; the
            # containment of the authority's rows over the covered window is
            # enforced below.
            expected_rows=_m1_feature_surface_rows(m1_feature_manifest_path),
            expected_m1_source_path=m1_path,
            expected_m1_source_sha256=root_manifest["m1_source_sha256"],
            context="UNIFIED_EXIT_M1_FEATURE_BASE_MANIFEST",
        )

        # No source or feature matrix is allocated before every immutable
        # feature-surface binding above has passed exactly.
        m1_times, m1_arrays = _validated_m1_arrays(m1_path)
        m1_feature_tempdir = tempfile.TemporaryDirectory(
            prefix="gx1_m1_feature_surface_"
        )
        try:
            m1_feature_times, m1_feature_arrays = load_m1_feature_surface(
                m1_feature_path,
                context="UNIFIED_EXIT_LIFECYCLE",
                storage_dir=Path(m1_feature_tempdir.name),
            )
        except Exception:
            m1_feature_tempdir.cleanup()
            raise
        self._m1_feature_tempdir = m1_feature_tempdir
        # Authority rows before the surface begins cannot be produced with valid
        # features at all. Every authority row from the surface start onward must
        # be present, and a gap inside that window is still a hard failure.
        covered_offset = (
            int(np.searchsorted(m1_times.asi8, m1_feature_times.asi8[0], "left"))
            if len(m1_feature_times)
            else len(m1_times)
        )
        m1_covered_times = m1_times[covered_offset:]
        if len(m1_covered_times) == 0 or not m1_feature_times.equals(
            m1_covered_times
        ):
            raise RuntimeError("UNIFIED_EXIT_M1_FEATURE_BASE_TIME_MISMATCH")
        # The source clock is NOT advanced. Episode pointers are absolute rows
        # written against it and sealed into the population streams, so moving it
        # would invalidate immutable evidence. The offset is carried instead and
        # applied wherever the feature surface is indexed.

        if set(entry_parquets) != set(selected_splits):
            raise RuntimeError(
                "UNIFIED_EXIT_LIFECYCLE_ENTRY_SPLIT_SET_INVALID"
            )
        self.splits: dict[str, UnifiedExitLifecycleSplit] = {}
        split_evidence: dict[str, Any] = {}
        # Validate exactly the splits this consumer is authorised to
        # materialise.  TEST semantic validation belongs to the immutable
        # build/seal step; the pre-freeze trainer has only metadata-only TEST
        # lineage and must not touch a TEST path or byte.
        for split in selected_splits:
            binding = root_manifest["splits"][split]
            _require_exact_keys(
                binding,
                {
                    "entry_dataset_path",
                    "entry_dataset_sha256",
                    "lifecycle_parquet",
                    "lifecycle_parquet_sha256",
                    "lifecycle_manifest",
                    "lifecycle_manifest_sha256",
                    "episode_rows",
                    "state_population_rows",
                    "state_population_stream_sha256",
                },
                context=f"UNIFIED_EXIT_LIFECYCLE_{split.upper()}_BINDING",
            )
            entry_path = Path(
                binding["entry_dataset_path"]
            ).expanduser().absolute()
            if (
                split in entry_parquets
                and Path(entry_parquets[split]).expanduser().absolute()
                != entry_path
            ) or (
                not entry_path.is_absolute()
                or entry_path.is_symlink()
                or not entry_path.is_file()
                or sha256_file(entry_path) != binding["entry_dataset_sha256"]
            ):
                raise RuntimeError(
                    f"UNIFIED_EXIT_LIFECYCLE_ENTRY_IDENTITY_INVALID: {split}"
                )
            lifecycle_path = root / binding["lifecycle_parquet"]
            split_manifest_path = root / binding["lifecycle_manifest"]
            if (
                lifecycle_path.name
                != f"{split}_unified_exit_lifecycle.parquet"
                or split_manifest_path.name
                != f"{split}_unified_exit_lifecycle.manifest.json"
                or sha256_file(lifecycle_path)
                != binding["lifecycle_parquet_sha256"]
                or sha256_file(split_manifest_path)
                != binding["lifecycle_manifest_sha256"]
            ):
                raise RuntimeError(
                    f"UNIFIED_EXIT_LIFECYCLE_SPLIT_IDENTITY_INVALID: {split}"
                )
            split_manifest = _read_exact_json(split_manifest_path)
            for key, expected in (
                ("schema_version", UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION),
                ("decision", "PASS"),
                ("entry_run_id", dataset_run_id),
                ("split", split),
                ("entry_dataset_path", str(entry_path)),
                ("entry_dataset_sha256", binding["entry_dataset_sha256"]),
                ("m1_source_path", str(m1_path)),
                ("m1_source_sha256", root_manifest["m1_source_sha256"]),
                (
                    "m1_authority_sha256",
                    root_manifest["m1_authority_sha256"],
                ),
                ("lifecycle_parquet", lifecycle_path.name),
                ("lifecycle_parquet_sha256", binding["lifecycle_parquet_sha256"]),
                ("lifecycle_parquet_rows", binding["episode_rows"]),
                ("state_population_rows", binding["state_population_rows"]),
                (
                    "state_population_stream_sha256",
                    binding["state_population_stream_sha256"],
                ),
                ("m1_row_clock", EXIT_FEATURE_ROW_CLOCK),
            ):
                if split_manifest.get(key) != expected:
                    raise RuntimeError(
                        f"UNIFIED_EXIT_LIFECYCLE_SPLIT_MANIFEST_MISMATCH: "
                        f"{split}.{key}"
                    )
            entry_times = pd.read_parquet(entry_path, columns=["time"])
            episodes = pd.read_parquet(lifecycle_path)
            parsed_episode_entry_times = pd.to_datetime(
                episodes["entry_time"],
                utc=True,
                errors="coerce",
            )
            entry_index = pd.to_numeric(
                episodes["entry_row_index"],
                errors="coerce",
            ).to_numpy(dtype=np.int64)
            parsed_entry_times = pd.DatetimeIndex(
                pd.to_datetime(
                    entry_times["time"],
                    utc=True,
                    errors="coerce",
                )
            )
            if (
                np.any(entry_index < 0)
                or np.any(entry_index >= len(parsed_entry_times))
                or not pd.DatetimeIndex(
                    parsed_episode_entry_times
                ).equals(parsed_entry_times[entry_index])
            ):
                raise RuntimeError(
                    f"UNIFIED_EXIT_LIFECYCLE_ENTRY_ROW_POINTER_INVALID: {split}"
                )
            split_contract = UnifiedExitLifecycleSplit(
                split=split,
                entry_row_count=len(entry_times),
                entry_times=parsed_entry_times,
                feature_row_offset=covered_offset,
                episodes=episodes,
                split_manifest=split_manifest,
                m1_times=m1_times,
                m1_arrays=m1_arrays,
                m1_feature_times=m1_feature_times,
                m1_feature_arrays=m1_feature_arrays,
            )
            if split in selected_splits:
                self.splits[split] = split_contract
            split_evidence[split] = {
                "entry_dataset_sha256": binding["entry_dataset_sha256"],
                "lifecycle_parquet_sha256": binding[
                    "lifecycle_parquet_sha256"
                ],
                "lifecycle_manifest_sha256": binding[
                    "lifecycle_manifest_sha256"
                ],
                "episode_rows": int(binding["episode_rows"]),
                "state_population_rows": int(binding["state_population_rows"]),
                "state_population_sha256": split_contract.state_population_sha256,
                "state_side_counts": dict(split_contract.state_side_counts),
            }
        self.evidence = {
            "schema_version": UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION,
            "root_manifest_path": str(manifest_path),
            "root_manifest_sha256": sha256_file(manifest_path),
            "entry_run_id": dataset_run_id,
            "m1_source_path": str(m1_path),
            "m1_source_sha256": root_manifest["m1_source_sha256"],
            "m1_feature_base_path": str(m1_feature_path),
            "m1_feature_base_sha256": root_manifest["m1_feature_base_sha256"],
            "m1_feature_base_manifest_path": str(m1_feature_manifest_path),
            "m1_feature_base_manifest_sha256": root_manifest[
                "m1_feature_base_manifest_sha256"
            ],
            "m1_authority": raw_authority,
            "m1_authority_sha256": root_manifest[
                "m1_authority_sha256"
            ],
            "path_state_count": UNIFIED_EXIT_MAX_PATH_BARS,
            "state_population_schema_version": (
                UNIFIED_EXIT_STATE_SELECTION_SCHEMA_VERSION
            ),
            "state_population_per_episode": UNIFIED_EXIT_MAX_PATH_BARS,
            "m1_row_clock": EXIT_FEATURE_ROW_CLOCK,
            "shared_feature_base_contract": (
                entry_exit_shared_feature_base_contract()
            ),
            "training_population": (
                UNIFIED_EXIT_STATE_SELECTION_SCHEMA_VERSION
            ),
            "sample_selection_depends_on_future_target": False,
            "validation_population": "all_authoritative_states",
            "test_population": "all_authoritative_states",
            "exit_supervision_authority": (
                "executable_exit_now_reward_plus_train_fitted_q"
            ),
            "extra_lookahead_beyond_trajectory": 0,
            "future_outcomes_used_as_model_inputs": False,
            "splits": split_evidence,
        }
