"""Immutable causal M1 lifecycle contract for unified Entry/Exit training."""

from __future__ import annotations

import hashlib
import json
import math
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
    UNIFIED_EXIT_PATH_FEATURE_DIM,
    UNIFIED_EXIT_PATH_PRICE_FIELDS,
    UNIFIED_EXIT_SIDE_ORDER,
    unified_exit_path_tensor_from_values,
)
from gx1.io.price_glitch_guard import assert_no_price_scale_glitch


UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION = (
    "gx1_unified_exit_lifecycle_episode_envelope_v3"
)
UNIFIED_EXIT_M1_AUTHORITY_SCHEMA_VERSION = (
    "gx1_unified_exit_native_pair_m1_authority_v1"
)
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
    "entry_row_index",
    "entry_time",
    "entry_available_at",
    "side_index",
    "side",
    "entry_bid",
    "entry_ask",
    "m1_start_row",
    "m1_start_time",
    "path_state_count",
    "target_lookahead_m1_steps",
    "non_tied_target_count",
    "hold_target_count",
    "exit_now_target_count",
    "tied_target_count",
)
UNIFIED_EXIT_TRAINING_SAMPLES_PER_ENTRY = 4
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


def _require_base28_native_m1_subset_identity(
    *,
    source_path: Path,
    native_root: Path,
    native_years: Mapping[str, Any],
    expected_rows: int,
) -> dict[str, Any]:
    """Prove every pair BASE28 row is byte-equivalent to native M1."""

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
        base_hash = canonical_native_rows_sha256(base, timeframe="M1")
        native_hash = canonical_native_rows_sha256(
            matched,
            timeframe="M1",
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
        "method": "exact_base28_rows_are_native_m1_subset_v1",
        "rows": observed_rows,
        "years": proof_by_year,
        "proof_sha256": canonical_json_sha256(proof_by_year),
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
    native_subset_proof = _require_base28_native_m1_subset_identity(
        source_path=source_path,
        native_root=native_root,
        native_years=observed_native["year_sha256"],
        expected_rows=base28_verified.binding.rows,
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
        or not isinstance(authority, Mapping)
        or authority.get("schema_version")
        != UNIFIED_EXIT_M1_AUTHORITY_SCHEMA_VERSION
        or value.get("m1_authority_sha256")
        != canonical_json_sha256(authority)
        or value.get("m1_source_path") != authority.get("m1_source_path")
        or value.get("m1_source_sha256")
        != authority.get("m1_source_sha256")
    ):
        raise RuntimeError("UNIFIED_EXIT_LIFECYCLE_AUTHORITY_EVIDENCE_INVALID")
    subset = authority.get("base28_native_m1_subset_proof")
    if (
        not isinstance(subset, Mapping)
        or subset.get("method")
        != "exact_base28_rows_are_native_m1_subset_v1"
        or isinstance(subset.get("rows"), bool)
        or not isinstance(subset.get("rows"), int)
        or int(subset["rows"]) <= 0
        or not isinstance(subset.get("years"), Mapping)
        or subset.get("proof_sha256")
        != canonical_json_sha256(subset["years"])
    ):
        raise RuntimeError("UNIFIED_EXIT_M1_NATIVE_SUBSET_EVIDENCE_INVALID")
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def unified_exit_future_extrema(
    *,
    bid: np.ndarray,
    ask: np.ndarray,
    lookahead: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return next-L executable extrema for every non-censored state."""

    if (
        isinstance(lookahead, bool)
        or not isinstance(lookahead, int)
        or lookahead <= 0
        or len(bid) != len(ask)
        or len(bid) <= lookahead
    ):
        raise RuntimeError("UNIFIED_EXIT_TARGET_RIGHT_CENSORED_EPISODE")
    future_bid = np.lib.stride_tricks.sliding_window_view(
        np.asarray(bid)[1:],
        lookahead,
    ).max(axis=1)
    future_ask = np.lib.stride_tricks.sliding_window_view(
        np.asarray(ask)[1:],
        lookahead,
    ).min(axis=1)
    return future_bid, future_ask


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
    """Validated split-local episode pointers and deterministic training samples."""

    def __init__(
        self,
        *,
        split: str,
        entry_row_count: int,
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
        self._m1_times = m1_times
        self._m1 = dict(m1_arrays)
        if not m1_feature_times.equals(m1_times):
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
            if not isinstance(values, np.ndarray) or values.shape != (
                len(m1_times),
                width,
            ):
                raise RuntimeError(
                    f"UNIFIED_EXIT_M1_FEATURE_SURFACE_{name.upper()}_SHAPE_INVALID"
                )
        self._selected_state = np.full(
            (self.entry_row_count, UNIFIED_EXIT_TRAINING_SAMPLES_PER_ENTRY),
            -1,
            dtype=np.int16,
        )
        self._selected_start = np.full_like(
            self._selected_state,
            -1,
            dtype=np.int64,
        )
        self._selected_side = np.zeros_like(self._selected_state, dtype=np.int8)
        self._selected_target = np.zeros_like(
            self._selected_state,
            dtype=np.int8,
        )
        self._entry_bid = np.zeros_like(self._selected_state, dtype=np.float64)
        self._entry_ask = np.zeros_like(self._selected_state, dtype=np.float64)
        self._validate_and_select(episodes, split_manifest)

    def _validate_and_select(
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
        lookahead = manifest.get("target_lookahead_m1_steps")
        if (
            isinstance(lookahead, bool)
            or not isinstance(lookahead, int)
            or lookahead <= 0
        ):
            raise RuntimeError(
                f"UNIFIED_EXIT_LIFECYCLE_LOOKAHEAD_INVALID: {self.split}"
            )
        future_bid, future_ask = unified_exit_future_extrema(
            bid=self._m1["bid_close"],
            ask=self._m1["ask_close"],
            lookahead=lookahead,
        )
        current_bid = self._m1["bid_close"][:-lookahead]
        current_ask = self._m1["ask_close"][:-lookahead]
        long_targets = np.where(
            future_bid > current_bid,
            0,
            np.where(future_bid < current_bid, 1, -1),
        ).astype(np.int8)
        short_targets = np.where(
            future_ask < current_ask,
            0,
            np.where(future_ask > current_ask, 1, -1),
        ).astype(np.int8)

        target_stream = hashlib.sha256()
        target_counts = {
            UNIFIED_EXIT_ACTION_ORDER[0]: 0,
            UNIFIED_EXIT_ACTION_ORDER[1]: 0,
            "TIED_OMITTED": 0,
        }
        observed_pairs: set[tuple[int, int]] = set()
        selected_counts = np.zeros(2, dtype=np.int64)
        selected_side_counts = np.zeros(2, dtype=np.int64)
        for raw in episodes.to_dict(orient="records"):
            entry_index = int(raw["entry_row_index"])
            side_index = int(raw["side_index"])
            start = int(raw["m1_start_row"])
            state_count = int(raw["path_state_count"])
            episode_lookahead = int(raw["target_lookahead_m1_steps"])
            if (
                not 0 <= entry_index < self.entry_row_count
                or side_index not in (0, 1)
                or raw["side"] != UNIFIED_EXIT_SIDE_ORDER[side_index]
                or state_count != UNIFIED_EXIT_MAX_PATH_BARS
                or episode_lookahead != lookahead
                or start < 0
                or start + state_count + lookahead > len(self._m1_times)
            ):
                raise RuntimeError(
                    f"UNIFIED_EXIT_LIFECYCLE_EPISODE_VALUE_INVALID: {self.split}"
                )
            pair = (entry_index, side_index)
            if pair in observed_pairs:
                raise RuntimeError(
                    f"UNIFIED_EXIT_LIFECYCLE_EPISODE_DUPLICATE: {self.split} {pair}"
                )
            observed_pairs.add(pair)
            if (
                pd.Timestamp(raw["m1_start_time"])
                != self._m1_times[start]
                or pd.Timestamp(raw["entry_available_at"])
                != self._m1_times[start]
            ):
                raise RuntimeError(
                    f"UNIFIED_EXIT_LIFECYCLE_POINTER_TIME_MISMATCH: {self.split}"
                )
            expected_bid = float(self._m1["bid_open"][start])
            expected_ask = float(self._m1["ask_open"][start])
            if (
                float(raw["entry_bid"]) != expected_bid
                or float(raw["entry_ask"]) != expected_ask
            ):
                raise RuntimeError(
                    f"UNIFIED_EXIT_LIFECYCLE_ENTRY_QUOTE_MISMATCH: {self.split}"
                )
            target = (
                long_targets[start : start + state_count]
                if side_index == 0
                else short_targets[start : start + state_count]
            )
            counts = {
                0: int(np.count_nonzero(target == 0)),
                1: int(np.count_nonzero(target == 1)),
                -1: int(np.count_nonzero(target == -1)),
            }
            if (
                int(raw["hold_target_count"]) != counts[0]
                or int(raw["exit_now_target_count"]) != counts[1]
                or int(raw["tied_target_count"]) != counts[-1]
                or int(raw["non_tied_target_count"]) != counts[0] + counts[1]
            ):
                raise RuntimeError(
                    f"UNIFIED_EXIT_LIFECYCLE_TARGET_COUNT_MISMATCH: {self.split}"
                )
            target_counts[UNIFIED_EXIT_ACTION_ORDER[0]] += counts[0]
            target_counts[UNIFIED_EXIT_ACTION_ORDER[1]] += counts[1]
            target_counts["TIED_OMITTED"] += counts[-1]
            target_stream.update(
                np.asarray(
                    [entry_index, side_index, start],
                    dtype="<i8",
                ).tobytes()
            )
            target_stream.update(np.ascontiguousarray(target).tobytes())

            for action_index in (0, 1):
                candidates = np.flatnonzero(target == action_index)
                if len(candidates) == 0:
                    continue
                slot = side_index * 2 + action_index
                selector = hashlib.sha256(
                    (
                        f"{self.split}|{entry_index}|{side_index}|"
                        f"{action_index}|{manifest['target_stream_sha256']}"
                    ).encode("ascii")
                ).digest()
                selected = int(
                    candidates[
                        int.from_bytes(selector[:8], "little") % len(candidates)
                    ]
                )
                self._selected_state[entry_index, slot] = selected
                self._selected_start[entry_index, slot] = start
                self._selected_side[entry_index, slot] = side_index
                self._selected_target[entry_index, slot] = action_index
                self._entry_bid[entry_index, slot] = expected_bid
                self._entry_ask[entry_index, slot] = expected_ask
                selected_counts[action_index] += 1
                selected_side_counts[side_index] += 1

        expected_target_stream = manifest.get("target_stream_sha256")
        if target_stream.hexdigest() != expected_target_stream:
            raise RuntimeError(
                f"UNIFIED_EXIT_LIFECYCLE_TARGET_STREAM_MISMATCH: {self.split}"
            )
        if manifest.get("target_counts") != target_counts:
            raise RuntimeError(
                f"UNIFIED_EXIT_LIFECYCLE_TARGET_PROOF_MISMATCH: {self.split}"
            )
        if int(manifest.get("episode_rows", -1)) != len(episodes):
            raise RuntimeError(
                f"UNIFIED_EXIT_LIFECYCLE_EPISODE_ROWS_MISMATCH: {self.split}"
            )
        if np.any(selected_counts <= 0) or np.any(selected_side_counts <= 0):
            raise RuntimeError(
                f"UNIFIED_EXIT_LIFECYCLE_SELECTED_CLASS_OR_SIDE_DEAD: "
                f"split={self.split} classes={selected_counts.tolist()} "
                f"sides={selected_side_counts.tolist()}"
            )
        self.selected_target_counts = {
            UNIFIED_EXIT_ACTION_ORDER[index]: int(selected_counts[index])
            for index in range(2)
        }
        self.selected_side_counts = {
            UNIFIED_EXIT_SIDE_ORDER[index]: int(selected_side_counts[index])
            for index in range(2)
        }

    def selected_current_decision_times_ns(self) -> np.ndarray:
        """Return unique selected decision clocks for route coverage proof."""

        valid = self._selected_state >= 0
        current_indices = np.unique(
            self._selected_start[valid].astype(np.int64, copy=False)
            + self._selected_state[valid].astype(np.int64, copy=False)
        )
        if current_indices.size < 1:
            raise RuntimeError(
                "UNIFIED_EXIT_SELECTED_CURRENT_POPULATION_EMPTY"
            )
        return np.asarray(
            self._m1_times.asi8[current_indices],
            dtype=np.int64,
        )

    def train_normalization_population(self) -> dict[str, Any]:
        """Expose exact unique physical M1 rows consumed by TRAIN Exit.

        The returned feature matrices remain the lifecycle-owned disk-backed
        arrays.  Only sorted int64 row selections are materialized, so the
        normalization fit can scan them repeatedly without copying the full
        513/142/5 surfaces into RAM.
        """

        if self.split != "train":
            raise RuntimeError(
                "UNIFIED_EXIT_NORMALIZATION_TRAIN_SPLIT_REQUIRED"
            )
        valid = self._selected_state >= 0
        selected_current = (
            self._selected_start[valid].astype(np.int64, copy=False)
            + self._selected_state[valid].astype(np.int64, copy=False)
        )
        current_indices = np.unique(selected_current)
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
        return {
            "signal": self._m1_features["signal"],
            "ctx_cont": self._m1_features["ctx_cont"],
            "ctx_cat": self._m1_features["ctx_cat"],
            "local_row_indices": local_indices,
            "current_row_indices": current_indices,
            "current_decision_times_ns": np.asarray(
                self._m1_times.asi8[current_indices],
                dtype=np.int64,
            ),
            "source_times_ns": np.asarray(self._m1_times.asi8, dtype=np.int64),
            "local_merged_intervals": tuple(merged),
        }

    def sample(self, entry_row_index: int) -> dict[str, np.ndarray]:
        index = int(entry_row_index)
        if not 0 <= index < self.entry_row_count:
            raise RuntimeError(
                f"UNIFIED_EXIT_LIFECYCLE_ENTRY_INDEX_INVALID: {index}"
            )
        paths = np.zeros(
            (
                UNIFIED_EXIT_TRAINING_SAMPLES_PER_ENTRY,
                UNIFIED_EXIT_MAX_PATH_BARS,
                UNIFIED_EXIT_PATH_FEATURE_DIM,
            ),
            dtype=np.float32,
        )
        lengths = np.zeros(
            UNIFIED_EXIT_TRAINING_SAMPLES_PER_ENTRY,
            dtype=np.int64,
        )
        feature_seq = np.zeros(
            (
                UNIFIED_EXIT_TRAINING_SAMPLES_PER_ENTRY,
                EXIT_FEATURE_SEQUENCE_BARS,
                MODEL_NATIVE_SIGNAL_DIM,
            ),
            dtype=np.float32,
        )
        feature_snap = np.zeros(
            (UNIFIED_EXIT_TRAINING_SAMPLES_PER_ENTRY, MODEL_NATIVE_SIGNAL_DIM),
            dtype=np.float32,
        )
        feature_ctx_cont = np.zeros(
            (UNIFIED_EXIT_TRAINING_SAMPLES_PER_ENTRY, MODEL_NATIVE_CTX_CONT_DIM),
            dtype=np.float32,
        )
        feature_ctx_cat = np.zeros(
            (UNIFIED_EXIT_TRAINING_SAMPLES_PER_ENTRY, MODEL_NATIVE_CTX_CAT_DIM),
            dtype=np.int64,
        )
        decision_time_ns = np.full(
            UNIFIED_EXIT_TRAINING_SAMPLES_PER_ENTRY,
            UNIFIED_EXIT_INVALID_DECISION_TIME_NS,
            dtype=np.int64,
        )
        valid = self._selected_state[index] >= 0
        for slot in np.flatnonzero(valid):
            state = int(self._selected_state[index, slot])
            start = int(self._selected_start[index, slot])
            length = state + 1
            entry_bid = float(self._entry_bid[index, slot])
            entry_ask = float(self._entry_ask[index, slot])
            if (
                not math.isfinite(entry_bid)
                or not math.isfinite(entry_ask)
                or entry_bid <= 0.0
                or entry_ask <= entry_bid
            ):
                raise RuntimeError("UNIFIED_EXIT_LIFECYCLE_ENTRY_QUOTE_INVALID")
            source_slice = slice(start, start + length)
            current_row = start + state
            decision_time_ns[slot] = int(self._m1_times[current_row].value)
            feature_start = current_row - EXIT_FEATURE_SEQUENCE_BARS + 1
            if feature_start < 0:
                raise RuntimeError(
                    "UNIFIED_EXIT_LIFECYCLE_M1_FEATURE_HISTORY_INSUFFICIENT"
                )
            feature_slice = slice(
                feature_start,
                current_row + 1,
            )
            feature_seq[slot] = self._m1_features["signal"][feature_slice]
            feature_snap[slot] = feature_seq[slot, -1]
            feature_ctx_cont[slot] = self._m1_features["ctx_cont"][current_row]
            feature_ctx_cat[slot] = self._m1_features["ctx_cat"][current_row]
            price_arrays = (
                self._m1["bid_open"],
                self._m1["bid_high"],
                self._m1["bid_low"],
                self._m1["bid_close"],
                self._m1["ask_open"],
                self._m1["ask_high"],
                self._m1["ask_low"],
                self._m1["ask_close"],
                self._m1["open"],
                self._m1["high"],
                self._m1["low"],
                self._m1["close"],
            )
            if len(price_arrays) != len(UNIFIED_EXIT_PATH_PRICE_FIELDS):
                raise RuntimeError("UNIFIED_EXIT_PATH_PRICE_LAYOUT_INVALID")
            paths[slot, :length] = unified_exit_path_tensor_from_values(
                price_values=np.column_stack(
                    [
                        values[source_slice]
                        for values in price_arrays
                    ]
                ),
                volumes=self._m1["volume"][source_slice],
                entry_bid=entry_bid,
                entry_ask=entry_ask,
            )
            lengths[slot] = length
        if not np.isfinite(paths).all():
            raise RuntimeError("UNIFIED_EXIT_LIFECYCLE_PATH_TENSOR_NONFINITE")
        return {
            "exit_feature_seq_x": feature_seq,
            "exit_feature_snap_x": feature_snap,
            "exit_feature_ctx_cat": feature_ctx_cat,
            "exit_feature_ctx_cont": feature_ctx_cont,
            "exit_decision_time_ns": decision_time_ns,
            "exit_path_x": paths,
            "exit_path_lengths": lengths,
            "exit_side_index": self._selected_side[index].astype(
                np.int64,
                copy=True,
            ),
            "exit_action_target": self._selected_target[index].astype(
                np.int64,
                copy=True,
            ),
            "exit_sample_valid": valid.astype(np.bool_, copy=True),
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
                split not in {"train", "val"}
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
        expected_inventory = {
            "UNIFIED_EXIT_LIFECYCLE_MANIFEST.json",
            *(
                f"{split}_unified_exit_lifecycle{suffix}"
                for split in ("train", "val", "test")
                for suffix in (".parquet", ".manifest.json")
            ),
        }
        observed_inventory = {path.name for path in root.iterdir()}
        if (
            observed_inventory != expected_inventory
            or any(path.is_symlink() or not path.is_file() for path in root.iterdir())
        ):
            raise RuntimeError("UNIFIED_EXIT_LIFECYCLE_INVENTORY_INVALID")
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
                "target_lookahead_m1_steps",
                "m1_row_clock",
                "shared_feature_base_contract",
                "side_order",
                "action_order",
                "splits",
            },
            context="UNIFIED_EXIT_LIFECYCLE_ROOT",
        )
        if (
            root_manifest["schema_version"]
            != UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION
            or root_manifest["decision"] != "PASS"
            or root_manifest["entry_run_id"] != dataset_run_id
            or root_manifest["path_state_count"] != UNIFIED_EXIT_MAX_PATH_BARS
            or root_manifest["m1_row_clock"] != EXIT_FEATURE_ROW_CLOCK
            or root_manifest["side_order"] != list(UNIFIED_EXIT_SIDE_ORDER)
            or root_manifest["action_order"] != list(UNIFIED_EXIT_ACTION_ORDER)
            or set(root_manifest["splits"]) != {"train", "val", "test"}
        ):
            raise RuntimeError("UNIFIED_EXIT_LIFECYCLE_ROOT_CONTRACT_INVALID")
        require_entry_exit_shared_feature_base_contract(
            root_manifest["shared_feature_base_contract"],
            context="UNIFIED_EXIT_LIFECYCLE_ROOT",
        )
        raw_authority = root_manifest["m1_authority"]
        if (
            not isinstance(raw_authority, dict)
            or raw_authority.get("schema_version")
            != UNIFIED_EXIT_M1_AUTHORITY_SCHEMA_VERSION
            or canonical_json_sha256(raw_authority)
            != root_manifest["m1_authority_sha256"]
        ):
            raise RuntimeError("UNIFIED_EXIT_M1_AUTHORITY_EVIDENCE_INVALID")
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
        # Both clocks are carried forward on the same covered window so every
        # positional lookup downstream indexes the same row on both sides. A
        # length difference here would silently mis-index the Exit path.
        if covered_offset:
            m1_times = m1_covered_times
            m1_arrays = {
                name: values[covered_offset:] for name, values in m1_arrays.items()
            }

        if set(entry_parquets) != set(selected_splits):
            raise RuntimeError(
                "UNIFIED_EXIT_LIFECYCLE_ENTRY_SPLIT_SET_INVALID"
            )
        self.splits: dict[str, UnifiedExitLifecycleSplit] = {}
        split_evidence: dict[str, Any] = {}
        for split in selected_splits:
            entry_path = Path(entry_parquets[split]).expanduser().absolute()
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
                    "target_counts",
                    "target_stream_sha256",
                },
                context=f"UNIFIED_EXIT_LIFECYCLE_{split.upper()}_BINDING",
            )
            if (
                Path(binding["entry_dataset_path"]).absolute() != entry_path
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
                ("target_counts", binding["target_counts"]),
                ("target_stream_sha256", binding["target_stream_sha256"]),
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
                episodes=episodes,
                split_manifest=split_manifest,
                m1_times=m1_times,
                m1_arrays=m1_arrays,
                m1_feature_times=m1_feature_times,
                m1_feature_arrays=m1_feature_arrays,
            )
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
                "target_counts": dict(binding["target_counts"]),
                "selected_target_counts": dict(
                    split_contract.selected_target_counts
                ),
                "selected_side_counts": dict(
                    split_contract.selected_side_counts
                ),
                "target_stream_sha256": binding["target_stream_sha256"],
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
            "target_lookahead_m1_steps": int(
                root_manifest["target_lookahead_m1_steps"]
            ),
            "m1_row_clock": EXIT_FEATURE_ROW_CLOCK,
            "shared_feature_base_contract": (
                entry_exit_shared_feature_base_contract()
            ),
            "training_sample_selection": (
                "deterministic_one_state_per_available_side_and_action_class"
            ),
            "future_outcomes_used_as_model_inputs": False,
            "splits": split_evidence,
        }
