"""TRAIN-fitted executable-PnL policy for Entry direction supervision."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_aux_targets_v3 import (
    MODEL_NATIVE_AUX_MAX_FUTURE_HORIZON_BARS,
)
from gx1.contracts.entry_exit_feature_base_v1 import (
    ENTRY_DECISION_BAR_SECONDS,
)
from gx1.io.price_glitch_guard import assert_no_price_scale_glitch


ENTRY_DIRECTION_TARGET_POLICY_SCHEMA_VERSION = (
    "gx1_entry_direction_target_policy_v2"
)
ENTRY_DIRECTION_TARGET_POLICY_FIT_METHOD = (
    "train_native_m5_executable_profit_max_chord_knee_v2"
)
ENTRY_DIRECTION_TARGET_POLICY_EDGE_FIT_METHOD = (
    "train_median_observed_executable_spread_bps_v1"
)
ENTRY_DIRECTION_TARGET_POLICY_PATH_THRESHOLD_FIT_METHOD = (
    "train_combined_long_short_empirical_quartiles_v1"
)
ENTRY_DIRECTION_TARGET_POLICY_PATH_QUANTILES = (0.25, 0.50, 0.75)
ENTRY_DIRECTION_TARGET_ACTION_RULE = (
    "trade_better_executable_pnl_side_iff_edge_and_side_margin_exceed_"
    "train_fitted_spread_floor_else_flat_v1"
)
ENTRY_DIRECTION_TARGET_POLICY_MAX_HORIZON_BARS = int(
    MODEL_NATIVE_AUX_MAX_FUTURE_HORIZON_BARS
)

_POLICY_KEYS = {
    "schema_version",
    "decision",
    "fit_split",
    "fit_scope",
    "fit_method",
    "edge_fit_method",
    "path_threshold_fit_method",
    "action_rule",
    "m5_row_clock",
    "source_parquet_sha256",
    "tape_provenance_sha256",
    "train_start_utc",
    "train_end_utc",
    "fit_first_m5_row",
    "fit_last_m5_row",
    "fit_first_m5_time_utc",
    "fit_last_m5_time_utc",
    "fit_population_rows",
    "fit_population_stream_sha256",
    "candidate_horizon_min_bars",
    "candidate_horizon_max_bars",
    "executable_spread_bps",
    "spread_source_stream_sha256",
    "long_first_material_profit_delay_counts",
    "short_first_material_profit_delay_counts",
    "combined_material_profit_discovery_counts",
    "selected_direction_horizon_bars",
    "selected_knee_distance_numerator",
    "path_quality_horizon_bars",
    "path_threshold_quantile_probabilities",
    "path_threshold_population_rows",
    "path_threshold_stream_sha256",
    "path_aux_thresholds_bps",
    "handwritten_path_thresholds",
    "tradable_edge_floor_bps",
    "side_margin_floor_bps",
    "early_move_threshold_bps",
    "side_score_formula",
    "future_outcomes_used_as_model_inputs",
    "val_test_rows_used_for_fit",
    "policy_sha256",
}


def canonical_entry_target_policy_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _utc(value: Any, *, name: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except Exception as exc:
        raise RuntimeError(f"ENTRY_TARGET_POLICY_{name}_INVALID") from exc
    if (
        pd.isna(parsed)
        or parsed.tz is None
        or parsed.utcoffset() != pd.Timedelta(0)
    ):
        raise RuntimeError(f"ENTRY_TARGET_POLICY_{name}_INVALID")
    return parsed.as_unit("ns")


def _selected_chord_knee(discovery_counts: np.ndarray) -> tuple[int, int]:
    counts = np.asarray(discovery_counts, dtype=np.int64)
    horizon = ENTRY_DIRECTION_TARGET_POLICY_MAX_HORIZON_BARS
    if (
        counts.shape != (horizon,)
        or np.any(counts < 0)
        or np.any(np.diff(counts) < 0)
    ):
        raise RuntimeError("ENTRY_TARGET_POLICY_DISCOVERY_CURVE_INVALID")
    if counts[-1] == counts[0]:
        return 1, 0
    offsets = np.arange(horizon, dtype=np.int64)
    distance = (
        (counts - counts[0]) * np.int64(horizon - 1)
        - offsets * np.int64(counts[-1] - counts[0])
    )
    selected = int(np.argmax(distance))
    return selected + 1, int(distance[selected])


def _first_material_profit_delay(
    *,
    entry: np.ndarray,
    future_exit: np.ndarray,
    state_rows: np.ndarray,
    hurdle_bps: float,
    side: str,
) -> np.ndarray:
    rows = np.asarray(state_rows, dtype=np.int64)
    entry_values = np.asarray(entry, dtype=np.float64)[rows]
    delay = np.zeros(len(rows), dtype=np.int16)
    unresolved = np.ones(len(rows), dtype=bool)
    for step in range(1, ENTRY_DIRECTION_TARGET_POLICY_MAX_HORIZON_BARS + 1):
        positions = np.flatnonzero(unresolved)
        if len(positions) == 0:
            break
        exit_values = future_exit[rows[positions] + step]
        if side == "long":
            pnl_bps = (exit_values / entry_values[positions] - 1.0) * 10_000.0
        elif side == "short":
            pnl_bps = (entry_values[positions] / exit_values - 1.0) * 10_000.0
        else:  # pragma: no cover - internal enum
            raise RuntimeError("ENTRY_TARGET_POLICY_SIDE_INVALID")
        newly = pnl_bps > hurdle_bps
        if np.any(newly):
            resolved = positions[newly]
            delay[resolved] = np.int16(step)
            unresolved[resolved] = False
    return delay


def _fit_path_aux_thresholds(
    *,
    bid: np.ndarray,
    ask: np.ndarray,
    rows: np.ndarray,
    horizon_bars: int,
) -> tuple[dict[str, float], str, int]:
    """Fit side-symmetric path thresholds from exact TRAIN outcomes.

    LONG and SHORT path observations share one empirical distribution so the
    auxiliary supervision cannot encode a directional prior.  MFE, MAE and
    path remain separate outcomes; no handwritten utility weights are fitted
    or applied here.
    """

    if horizon_bars < 1 or int(rows[-1]) + horizon_bars >= len(bid):
        raise RuntimeError("ENTRY_TARGET_POLICY_PATH_HORIZON_INVALID")
    entry_bid = bid[rows]
    entry_ask = ask[rows]
    max_bid = entry_bid.copy()
    min_bid = entry_bid.copy()
    max_ask = entry_ask.copy()
    min_ask = entry_ask.copy()
    for step in range(1, horizon_bars + 1):
        future_bid = bid[rows + step]
        future_ask = ask[rows + step]
        np.maximum(max_bid, future_bid, out=max_bid)
        np.minimum(min_bid, future_bid, out=min_bid)
        np.maximum(max_ask, future_ask, out=max_ask)
        np.minimum(min_ask, future_ask, out=min_ask)

    mfe_long = (max_bid - entry_ask) / entry_ask * 10_000.0
    mae_long = (entry_ask - min_bid) / entry_ask * 10_000.0
    mfe_short = (entry_bid - min_ask) / entry_bid * 10_000.0
    mae_short = (max_ask - entry_bid) / entry_bid * 10_000.0
    mfe = np.concatenate((mfe_long, mfe_short)).astype(np.float64, copy=False)
    mae = np.concatenate((mae_long, mae_short)).astype(np.float64, copy=False)
    path = (mfe - mae).astype(np.float64, copy=False)
    if (
        not np.isfinite(mfe).all()
        or not np.isfinite(mae).all()
        or not np.isfinite(path).all()
        or np.any(mae <= 0.0)
    ):
        raise RuntimeError("ENTRY_TARGET_POLICY_PATH_POPULATION_INVALID")

    probabilities = np.asarray(
        ENTRY_DIRECTION_TARGET_POLICY_PATH_QUANTILES,
        dtype=np.float64,
    )
    thresholds: dict[str, float] = {}
    for name, values in (("mfe", mfe), ("mae", mae), ("path", path)):
        fitted = np.quantile(values, probabilities, method="linear")
        thresholds.update(
            {
                f"{name}_low_bps": float(fitted[0]),
                f"{name}_median_bps": float(fitted[1]),
                f"{name}_high_bps": float(fitted[2]),
            }
        )
    stream = hashlib.sha256()
    stream.update(b"entry_direction_path_threshold_population_v1\0")
    for values in (mfe, mae, path):
        stream.update(np.ascontiguousarray(values, dtype="<f8").tobytes())
    return thresholds, stream.hexdigest(), int(len(mfe))


def fit_entry_direction_target_policy(
    *,
    closed_m5: pd.DataFrame,
    train_start: pd.Timestamp | str,
    train_end: pd.Timestamp | str,
    source_parquet_sha256: str,
    tape_provenance_sha256: str,
) -> dict[str, Any]:
    """Fit one direction policy on exact native TRAIN M5 rows only."""

    if not _is_sha256(source_parquet_sha256) or not _is_sha256(
        tape_provenance_sha256
    ):
        raise RuntimeError("ENTRY_TARGET_POLICY_SOURCE_BINDING_INVALID")
    if not isinstance(closed_m5, pd.DataFrame) or not {
        "time",
        "bid_close",
        "ask_close",
    }.issubset(closed_m5.columns):
        raise RuntimeError("ENTRY_TARGET_POLICY_M5_FRAME_INVALID")
    assert_no_price_scale_glitch(
        closed_m5,
        context="ENTRY_DIRECTION_TARGET_POLICY_TRAIN_SOURCE",
    )
    start = _utc(train_start, name="TRAIN_START")
    end = _utc(train_end, name="TRAIN_END")
    if end <= start:
        raise RuntimeError("ENTRY_TARGET_POLICY_TRAIN_RANGE_INVALID")
    times = pd.DatetimeIndex(
        pd.to_datetime(closed_m5["time"], utc=True, errors="coerce")
    ).as_unit("ns")
    if (
        len(times) <= ENTRY_DIRECTION_TARGET_POLICY_MAX_HORIZON_BARS
        or times.hasnans
        or not times.is_unique
        or not times.is_monotonic_increasing
        or not times.floor(f"{ENTRY_DECISION_BAR_SECONDS}s").equals(times)
    ):
        raise RuntimeError("ENTRY_TARGET_POLICY_M5_TIME_INVALID")
    bid = pd.to_numeric(closed_m5["bid_close"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    ask = pd.to_numeric(closed_m5["ask_close"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if (
        not np.isfinite(bid).all()
        or not np.isfinite(ask).all()
        or np.any(bid <= 0.0)
        or np.any(ask <= bid)
    ):
        raise RuntimeError("ENTRY_TARGET_POLICY_EXECUTABLE_QUOTES_INVALID")
    available = times + pd.Timedelta(seconds=ENTRY_DECISION_BAR_SECONDS)
    first = int(np.searchsorted(available.asi8, start.value, side="left"))
    last_future = int(np.searchsorted(available.asi8, end.value, side="right") - 1)
    last = last_future - ENTRY_DIRECTION_TARGET_POLICY_MAX_HORIZON_BARS
    if first < 0 or last < first:
        raise RuntimeError("ENTRY_TARGET_POLICY_TRAIN_POPULATION_EMPTY")
    rows = np.arange(first, last + 1, dtype=np.int64)
    mid = (ask[rows] + bid[rows]) * 0.5
    spread_bps = (ask[rows] - bid[rows]) / mid * 10_000.0
    hurdle = float(np.median(spread_bps))
    if (
        not np.isfinite(spread_bps).all()
        or np.any(spread_bps <= 0.0)
        or not math.isfinite(hurdle)
        or hurdle <= 0.0
    ):
        raise RuntimeError("ENTRY_TARGET_POLICY_SPREAD_FIT_INVALID")

    long_delay = _first_material_profit_delay(
        entry=ask,
        future_exit=bid,
        state_rows=rows,
        hurdle_bps=hurdle,
        side="long",
    )
    short_delay = _first_material_profit_delay(
        entry=bid,
        future_exit=ask,
        state_rows=rows,
        hurdle_bps=hurdle,
        side="short",
    )
    size = ENTRY_DIRECTION_TARGET_POLICY_MAX_HORIZON_BARS + 1
    long_counts = np.bincount(long_delay, minlength=size).astype(np.int64)
    short_counts = np.bincount(short_delay, minlength=size).astype(np.int64)
    discovery = np.cumsum(long_counts[1:] + short_counts[1:], dtype=np.int64)
    selected, distance = _selected_chord_knee(discovery)
    path_thresholds, path_threshold_stream, path_threshold_rows = (
        _fit_path_aux_thresholds(
            bid=bid,
            ask=ask,
            rows=rows,
            horizon_bars=selected,
        )
    )

    population_stream = hashlib.sha256()
    population_stream.update(
        np.ascontiguousarray(times.asi8[rows], dtype="<i8").tobytes()
    )
    population_stream.update(np.ascontiguousarray(bid[rows], dtype="<f8").tobytes())
    population_stream.update(np.ascontiguousarray(ask[rows], dtype="<f8").tobytes())
    spread_stream = hashlib.sha256(
        np.ascontiguousarray(spread_bps, dtype="<f8").tobytes()
    ).hexdigest()
    policy: dict[str, Any] = {
        "schema_version": ENTRY_DIRECTION_TARGET_POLICY_SCHEMA_VERSION,
        "decision": "PASS",
        "fit_split": "train",
        "fit_scope": "TRAIN_ONLY",
        "fit_method": ENTRY_DIRECTION_TARGET_POLICY_FIT_METHOD,
        "edge_fit_method": ENTRY_DIRECTION_TARGET_POLICY_EDGE_FIT_METHOD,
        "path_threshold_fit_method": (
            ENTRY_DIRECTION_TARGET_POLICY_PATH_THRESHOLD_FIT_METHOD
        ),
        "action_rule": ENTRY_DIRECTION_TARGET_ACTION_RULE,
        "m5_row_clock": "consecutive_authoritative_closed_m5_source_rows",
        "source_parquet_sha256": source_parquet_sha256,
        "tape_provenance_sha256": tape_provenance_sha256,
        "train_start_utc": start.isoformat(),
        "train_end_utc": end.isoformat(),
        "fit_first_m5_row": first,
        "fit_last_m5_row": last,
        "fit_first_m5_time_utc": times[first].isoformat(),
        "fit_last_m5_time_utc": times[last].isoformat(),
        "fit_population_rows": int(len(rows)),
        "fit_population_stream_sha256": population_stream.hexdigest(),
        "candidate_horizon_min_bars": 1,
        "candidate_horizon_max_bars": (
            ENTRY_DIRECTION_TARGET_POLICY_MAX_HORIZON_BARS
        ),
        "executable_spread_bps": hurdle,
        "spread_source_stream_sha256": spread_stream,
        "long_first_material_profit_delay_counts": long_counts.tolist(),
        "short_first_material_profit_delay_counts": short_counts.tolist(),
        "combined_material_profit_discovery_counts": discovery.tolist(),
        "selected_direction_horizon_bars": selected,
        "selected_knee_distance_numerator": distance,
        "path_quality_horizon_bars": selected,
        "path_threshold_quantile_probabilities": list(
            ENTRY_DIRECTION_TARGET_POLICY_PATH_QUANTILES
        ),
        "path_threshold_population_rows": path_threshold_rows,
        "path_threshold_stream_sha256": path_threshold_stream,
        "path_aux_thresholds_bps": path_thresholds,
        "handwritten_path_thresholds": False,
        "tradable_edge_floor_bps": hurdle,
        "side_margin_floor_bps": hurdle,
        "early_move_threshold_bps": hurdle,
        "side_score_formula": "spread_aware_executable_pnl_at_selected_horizon_bps",
        "future_outcomes_used_as_model_inputs": False,
        "val_test_rows_used_for_fit": 0,
    }
    policy["policy_sha256"] = canonical_entry_target_policy_sha256(policy)
    return require_entry_direction_target_policy(
        policy,
        expected_source_parquet_sha256=source_parquet_sha256,
        expected_tape_provenance_sha256=tape_provenance_sha256,
    )


def require_entry_direction_target_policy(
    value: Any,
    *,
    expected_source_parquet_sha256: str | None = None,
    expected_tape_provenance_sha256: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _POLICY_KEYS:
        raise RuntimeError("ENTRY_TARGET_POLICY_SCHEMA_INVALID")
    policy = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    declared_hash = policy.pop("policy_sha256", None)
    if not _is_sha256(declared_hash) or declared_hash != canonical_entry_target_policy_sha256(
        policy
    ):
        raise RuntimeError("ENTRY_TARGET_POLICY_HASH_INVALID")
    policy["policy_sha256"] = declared_hash
    if (
        policy["schema_version"] != ENTRY_DIRECTION_TARGET_POLICY_SCHEMA_VERSION
        or policy["decision"] != "PASS"
        or policy["fit_split"] != "train"
        or policy["fit_scope"] != "TRAIN_ONLY"
        or policy["fit_method"] != ENTRY_DIRECTION_TARGET_POLICY_FIT_METHOD
        or policy["edge_fit_method"]
        != ENTRY_DIRECTION_TARGET_POLICY_EDGE_FIT_METHOD
        or policy["path_threshold_fit_method"]
        != ENTRY_DIRECTION_TARGET_POLICY_PATH_THRESHOLD_FIT_METHOD
        or policy["action_rule"] != ENTRY_DIRECTION_TARGET_ACTION_RULE
        or policy["m5_row_clock"]
        != "consecutive_authoritative_closed_m5_source_rows"
        or policy["candidate_horizon_min_bars"] != 1
        or policy["candidate_horizon_max_bars"]
        != ENTRY_DIRECTION_TARGET_POLICY_MAX_HORIZON_BARS
        or policy["future_outcomes_used_as_model_inputs"] is not False
        or policy["val_test_rows_used_for_fit"] != 0
        or policy["side_score_formula"]
        != "spread_aware_executable_pnl_at_selected_horizon_bps"
    ):
        raise RuntimeError("ENTRY_TARGET_POLICY_CONTRACT_INVALID")
    if expected_source_parquet_sha256 is not None and policy[
        "source_parquet_sha256"
    ] != expected_source_parquet_sha256:
        raise RuntimeError("ENTRY_TARGET_POLICY_SOURCE_MISMATCH")
    if expected_tape_provenance_sha256 is not None and policy[
        "tape_provenance_sha256"
    ] != expected_tape_provenance_sha256:
        raise RuntimeError("ENTRY_TARGET_POLICY_TAPE_MISMATCH")
    if not _is_sha256(policy["source_parquet_sha256"]) or not _is_sha256(
        policy["tape_provenance_sha256"]
    ):
        raise RuntimeError("ENTRY_TARGET_POLICY_SOURCE_HASH_INVALID")
    start = _utc(policy["train_start_utc"], name="TRAIN_START")
    end = _utc(policy["train_end_utc"], name="TRAIN_END")
    first_time = _utc(policy["fit_first_m5_time_utc"], name="FIT_FIRST_TIME")
    last_time = _utc(policy["fit_last_m5_time_utc"], name="FIT_LAST_TIME")
    integer_fields = (
        "fit_first_m5_row",
        "fit_last_m5_row",
        "fit_population_rows",
        "selected_direction_horizon_bars",
        "selected_knee_distance_numerator",
        "path_quality_horizon_bars",
    )
    if any(
        isinstance(policy[name], bool) or not isinstance(policy[name], int)
        for name in integer_fields
    ):
        raise RuntimeError("ENTRY_TARGET_POLICY_INTEGER_INVALID")
    rows = policy["fit_population_rows"]
    if (
        end <= start
        or last_time < first_time
        or policy["fit_first_m5_row"] < 0
        or policy["fit_last_m5_row"] < policy["fit_first_m5_row"]
        or rows <= 0
        or policy["fit_last_m5_row"] - policy["fit_first_m5_row"] + 1 != rows
        or not _is_sha256(policy["fit_population_stream_sha256"])
        or not _is_sha256(policy["spread_source_stream_sha256"])
    ):
        raise RuntimeError("ENTRY_TARGET_POLICY_POPULATION_INVALID")
    numeric_fields = (
        "executable_spread_bps",
        "tradable_edge_floor_bps",
        "side_margin_floor_bps",
        "early_move_threshold_bps",
    )
    if any(
        isinstance(policy[name], bool)
        or not isinstance(policy[name], (int, float))
        or not math.isfinite(float(policy[name]))
        or float(policy[name]) <= 0.0
        for name in numeric_fields
    ) or any(
        float(policy[name]) != float(policy["executable_spread_bps"])
        for name in numeric_fields[1:]
    ):
        raise RuntimeError("ENTRY_TARGET_POLICY_EDGE_FLOOR_INVALID")
    size = ENTRY_DIRECTION_TARGET_POLICY_MAX_HORIZON_BARS + 1
    histograms: list[np.ndarray] = []
    for name in (
        "long_first_material_profit_delay_counts",
        "short_first_material_profit_delay_counts",
    ):
        raw = policy[name]
        if (
            not isinstance(raw, list)
            or len(raw) != size
            or any(isinstance(item, bool) or not isinstance(item, int) for item in raw)
        ):
            raise RuntimeError("ENTRY_TARGET_POLICY_DELAY_COUNTS_INVALID")
        values = np.asarray(raw, dtype=np.int64)
        if np.any(values < 0) or int(values.sum()) != rows:
            raise RuntimeError("ENTRY_TARGET_POLICY_DELAY_COUNTS_INVALID")
        histograms.append(values)
    discovery = np.cumsum(histograms[0][1:] + histograms[1][1:], dtype=np.int64)
    if policy["combined_material_profit_discovery_counts"] != discovery.tolist():
        raise RuntimeError("ENTRY_TARGET_POLICY_DISCOVERY_COUNTS_INVALID")
    selected, distance = _selected_chord_knee(discovery)
    if (
        policy["selected_direction_horizon_bars"] != selected
        or policy["selected_knee_distance_numerator"] != distance
        or policy["path_quality_horizon_bars"] != selected
    ):
        raise RuntimeError("ENTRY_TARGET_POLICY_KNEE_INVALID")
    if (
        policy["path_threshold_quantile_probabilities"]
        != list(ENTRY_DIRECTION_TARGET_POLICY_PATH_QUANTILES)
        or policy["handwritten_path_thresholds"] is not False
        or type(policy["path_threshold_population_rows"]) is not int
        or policy["path_threshold_population_rows"] != 2 * rows
        or not _is_sha256(policy["path_threshold_stream_sha256"])
    ):
        raise RuntimeError("ENTRY_TARGET_POLICY_PATH_THRESHOLD_PROVENANCE_INVALID")
    thresholds = policy["path_aux_thresholds_bps"]
    expected_threshold_keys = {
        f"{name}_{level}_bps"
        for name in ("mfe", "mae", "path")
        for level in ("low", "median", "high")
    }
    if not isinstance(thresholds, dict) or set(thresholds) != expected_threshold_keys:
        raise RuntimeError("ENTRY_TARGET_POLICY_PATH_THRESHOLDS_INVALID")
    for name in ("mfe", "mae", "path"):
        values = [thresholds[f"{name}_{level}_bps"] for level in ("low", "median", "high")]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ) or not float(values[0]) <= float(values[1]) <= float(values[2]):
            raise RuntimeError("ENTRY_TARGET_POLICY_PATH_THRESHOLDS_INVALID")
    if any(float(thresholds[f"mae_{level}_bps"]) <= 0.0 for level in ("low", "median", "high")):
        raise RuntimeError("ENTRY_TARGET_POLICY_PATH_THRESHOLDS_INVALID")
    return policy


def apply_entry_direction_target_rule(
    *,
    long_executable_pnl_bps: Any,
    short_executable_pnl_bps: Any,
    tradable_edge_floor_bps: Any,
    side_margin_floor_bps: Any,
) -> dict[str, np.ndarray]:
    """Apply the one immutable Entry direction rule to executable outcomes.

    The two floors are the only decision-affecting values in the rule.  They
    are taken as arguments rather than read from a policy object so that a
    consumer holding only the policy's *projection* -- a split manifest carries
    `diagnostic_tradable_edge_floor_bps` / `diagnostic_side_margin_floor_bps`
    but is forbidden by
    `build_entry_v10_ctx_training_dataset_v3` from carrying the policy payload
    itself -- can still evaluate the exact same rule instead of restating it.
    Callers that hold the full fitted policy must go through
    `entry_direction_targets_from_policy`, which supplies the floors from the
    normalized policy.
    """

    long_score = np.asarray(long_executable_pnl_bps, dtype=np.float32)
    short_score = np.asarray(short_executable_pnl_bps, dtype=np.float32)
    if (
        long_score.shape != short_score.shape
        or long_score.ndim != 1
        or not np.isfinite(long_score).all()
        or not np.isfinite(short_score).all()
    ):
        raise RuntimeError("ENTRY_TARGET_POLICY_EXECUTABLE_OUTCOMES_INVALID")

    margin = (long_score - short_score).astype(np.float32)
    edge_floor = np.float32(tradable_edge_floor_bps)
    margin_floor = np.float32(side_margin_floor_bps)
    if not np.isfinite(edge_floor) or not np.isfinite(margin_floor):
        raise RuntimeError("ENTRY_TARGET_POLICY_RULE_FLOORS_INVALID")
    tradable_long = (long_score > edge_floor) & (margin > margin_floor)
    tradable_short = (short_score > edge_floor) & ((-margin) > margin_floor)

    side = np.full(long_score.shape, -1, dtype=np.int8)
    side[tradable_long & ~tradable_short] = np.int8(0)
    side[tradable_short & ~tradable_long] = np.int8(1)
    both = tradable_long & tradable_short
    side[both & (long_score >= short_score)] = np.int8(0)
    side[both & (short_score > long_score)] = np.int8(1)

    direction = np.full(long_score.shape, 2, dtype=np.int32)
    direction[side == 0] = np.int32(0)
    direction[side == 1] = np.int32(1)
    return {
        "long_score_bps": long_score,
        "short_score_bps": short_score,
        "side_margin_bps": margin,
        "tradable_long": tradable_long,
        "tradable_short": tradable_short,
        "side": side,
        "direction": direction,
        "trade": (side != -1),
    }


def entry_direction_targets_from_policy(
    *,
    policy: Mapping[str, Any],
    long_executable_pnl_bps: Any,
    short_executable_pnl_bps: Any,
) -> dict[str, np.ndarray]:
    """Apply the immutable direction rule using a full fitted policy.

    This is deliberately owned beside the fitted policy so dataset build,
    pre-train audit and trainer validation cannot drift into separate label
    formulas.  Inputs are realized supervision outcomes only; this helper is
    never part of the serving forward path.
    """

    normalized = require_entry_direction_target_policy(policy)
    return apply_entry_direction_target_rule(
        long_executable_pnl_bps=long_executable_pnl_bps,
        short_executable_pnl_bps=short_executable_pnl_bps,
        tradable_edge_floor_bps=normalized["tradable_edge_floor_bps"],
        side_margin_floor_bps=normalized["side_margin_floor_bps"],
    )


def require_entry_direction_target_manifest_binding(
    extra: Any,
    *,
    expected_source_parquet_sha256: str | None = None,
    expected_tape_provenance_sha256: str | None = None,
    expected_train_start: Any | None = None,
    expected_train_end: Any | None = None,
) -> dict[str, Any]:
    """Validate the exact policy projection embedded in a split manifest."""

    if not isinstance(extra, Mapping):
        raise RuntimeError("ENTRY_TARGET_POLICY_MANIFEST_EXTRA_INVALID")
    policy = require_entry_direction_target_policy(
        extra.get("entry_direction_target_policy"),
        expected_source_parquet_sha256=expected_source_parquet_sha256,
        expected_tape_provenance_sha256=expected_tape_provenance_sha256,
    )
    if extra.get("entry_direction_target_policy_sha256") != policy["policy_sha256"]:
        raise RuntimeError("ENTRY_TARGET_POLICY_MANIFEST_HASH_MISMATCH")
    if expected_train_start is not None and _utc(
        expected_train_start, name="EXPECTED_TRAIN_START"
    ) != _utc(policy["train_start_utc"], name="TRAIN_START"):
        raise RuntimeError("ENTRY_TARGET_POLICY_MANIFEST_TRAIN_START_MISMATCH")
    if expected_train_end is not None and _utc(
        expected_train_end, name="EXPECTED_TRAIN_END"
    ) != _utc(policy["train_end_utc"], name="TRAIN_END"):
        raise RuntimeError("ENTRY_TARGET_POLICY_MANIFEST_TRAIN_END_MISMATCH")

    strict = extra.get("strict_entry_labels")
    if not isinstance(strict, Mapping):
        raise RuntimeError("ENTRY_TARGET_POLICY_STRICT_LABEL_CONTRACT_MISSING")
    expected_fields = {
        "direction_target_mode": "train_fitted_executable_pnl_v1",
        "direction_label_source": (
            "train_fitted_spread_aware_executable_pnl_at_selected_horizon"
        ),
        "direction_label_horizon_bars": policy[
            "selected_direction_horizon_bars"
        ],
        "direction_side_score_formula": policy["side_score_formula"],
        "direction_tradable_edge_floor_bps": policy[
            "tradable_edge_floor_bps"
        ],
        "direction_side_margin_floor_bps": policy["side_margin_floor_bps"],
        "direction_path_quality_horizon_bars": policy[
            "path_quality_horizon_bars"
        ],
        "entry_direction_target_policy_sha256": policy["policy_sha256"],
    }
    for owner_name, owner in (("extra", extra), ("strict_entry_labels", strict)):
        if owner.get("entry_direction_target_policy") != policy:
            raise RuntimeError(
                f"ENTRY_TARGET_POLICY_{owner_name.upper()}_PAYLOAD_MISMATCH"
            )
        for key, expected in expected_fields.items():
            if owner.get(key) != expected:
                raise RuntimeError(
                    "ENTRY_TARGET_POLICY_MANIFEST_PROJECTION_MISMATCH: "
                    f"owner={owner_name} key={key}"
                )
    if float(extra.get("early_move_threshold_bps", math.nan)) != float(
        policy["early_move_threshold_bps"]
    ):
        raise RuntimeError("ENTRY_TARGET_POLICY_EARLY_MOVE_PROJECTION_MISMATCH")
    return policy
