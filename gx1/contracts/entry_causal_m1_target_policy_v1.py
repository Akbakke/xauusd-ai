"""TRAIN-only Entry auxiliary policy fitted from exact M1 execution evidence.

The historic direction policy uses the M5 close as entry and exit.  This
successor is intentionally separate: a rebuild must bind this exact M1 source
in the ranker, builder and manifests before it can claim causal labels.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_causal_m1_outcomes_v1 import (
    build_entry_m1_fill_surface,
    causal_m1_outcomes_at_horizon,
    causal_m1_target_contract,
    causal_m1_terminal_outcomes_at_horizon,
    prepare_causal_m1_quote_source,
)
from gx1.contracts.entry_direction_target_policy_v1 import (
    ENTRY_DIRECTION_TARGET_POLICY_MAX_HORIZON_BARS,
    ENTRY_DIRECTION_TARGET_POLICY_PATH_QUANTILES,
    apply_entry_direction_target_rule,
)
from gx1.contracts.entry_exit_feature_base_v1 import ENTRY_DECISION_BAR_SECONDS


ENTRY_CAUSAL_M1_TARGET_POLICY_SCHEMA_VERSION = "gx1_entry_causal_m1_target_policy_v1"
ENTRY_CAUSAL_M1_TARGET_POLICY_FIT_METHOD = "train_exact_m1_fill_executable_profit_max_chord_knee_v1"
ENTRY_CAUSAL_M1_TARGET_POLICY_EDGE_FIT_METHOD = "train_median_exact_m1_entry_spread_bps_v1"
ENTRY_CAUSAL_M1_TARGET_POLICY_PATH_THRESHOLD_FIT_METHOD = "train_combined_long_short_exact_m1_path_empirical_quartiles_v1"
ENTRY_CAUSAL_M1_TARGET_POLICY_ACTION_RULE = "trade_better_exact_m1_executable_pnl_side_iff_edge_and_side_margin_exceed_train_fitted_m1_spread_floor_else_flat_v1"
ENTRY_CAUSAL_M1_DIAGNOSTIC_OUTCOME_TARGET_MODE = "train_fitted_exact_m1_execution_diagnostics_v1"
ENTRY_CAUSAL_M1_DIAGNOSTIC_OUTCOME_LABEL_SOURCE = "train_fitted_exact_m1_fill_executable_pnl_at_selected_horizon"

_POLICY_KEYS = {
    "schema_version", "decision", "fit_split", "fit_scope", "fit_method",
    "edge_fit_method", "path_threshold_fit_method", "action_rule",
    "m5_decision_clock", "source_parquet_sha256", "tape_provenance_sha256",
    "m1_source_sha256", "target_contract", "train_start_utc", "train_end_utc",
    "fit_first_m5_row", "fit_last_m5_row", "fit_first_m5_time_utc",
    "fit_last_m5_time_utc", "fit_population_rows", "fit_population_stream_sha256",
    "candidate_horizon_min_bars", "candidate_horizon_max_bars",
    "executable_spread_bps", "spread_source_stream_sha256",
    "long_first_material_profit_delay_counts", "short_first_material_profit_delay_counts",
    "combined_material_profit_discovery_counts", "selected_direction_horizon_bars",
    "selected_knee_distance_numerator", "path_quality_horizon_bars",
    "path_threshold_quantile_probabilities", "path_threshold_population_rows",
    "path_threshold_stream_sha256", "path_aux_thresholds_bps",
    "handwritten_path_thresholds", "tradable_edge_floor_bps",
    "side_margin_floor_bps", "early_move_threshold_bps", "side_score_formula",
    "future_outcomes_used_as_model_inputs", "val_test_rows_used_for_fit", "policy_sha256",
}


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
        raise RuntimeError(f"ENTRY_CAUSAL_M1_TARGET_POLICY_{name}_INVALID") from exc
    if pd.isna(parsed) or parsed.tz is None or parsed.utcoffset() != pd.Timedelta(0):
        raise RuntimeError(f"ENTRY_CAUSAL_M1_TARGET_POLICY_{name}_INVALID")
    return parsed.as_unit("ns")


def _m5_times(frame: pd.DataFrame) -> pd.DatetimeIndex:
    if not isinstance(frame, pd.DataFrame) or "time" not in frame.columns:
        raise RuntimeError("ENTRY_CAUSAL_M1_TARGET_POLICY_M5_FRAME_INVALID")
    times = pd.DatetimeIndex(pd.to_datetime(frame["time"], utc=True, errors="coerce"))
    times = times.as_unit("ns")
    if (
        len(times) <= ENTRY_DIRECTION_TARGET_POLICY_MAX_HORIZON_BARS
        or times.hasnans
        or not times.is_unique
        or not times.is_monotonic_increasing
        or not times.floor(f"{ENTRY_DECISION_BAR_SECONDS}s").equals(times)
    ):
        raise RuntimeError("ENTRY_CAUSAL_M1_TARGET_POLICY_M5_TIME_INVALID")
    return times


def canonical_causal_m1_target_policy_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _selected_chord_knee(discovery_counts: np.ndarray) -> tuple[int, int]:
    counts = np.asarray(discovery_counts, dtype=np.int64)
    horizon = ENTRY_DIRECTION_TARGET_POLICY_MAX_HORIZON_BARS
    if (
        counts.shape != (horizon,)
        or np.any(counts < 0)
        or np.any(np.diff(counts) < 0)
    ):
        raise RuntimeError("ENTRY_CAUSAL_M1_TARGET_POLICY_DISCOVERY_CURVE_INVALID")
    if counts[-1] == counts[0]:
        return 1, 0
    offsets = np.arange(horizon, dtype=np.int64)
    distance = (
        (counts - counts[0]) * np.int64(horizon - 1)
        - offsets * np.int64(counts[-1] - counts[0])
    )
    selected = int(np.argmax(distance))
    return selected + 1, int(distance[selected])


def causal_m1_direction_targets_from_policy(
    *,
    policy: Mapping[str, Any],
    long_executable_pnl_bps: Any,
    short_executable_pnl_bps: Any,
) -> dict[str, np.ndarray]:
    """Apply the frozen M1-evidence target rule; this is offline only."""

    normalized = require_causal_m1_target_policy(policy)
    long_pnl = np.asarray(long_executable_pnl_bps, dtype=np.float64)
    short_pnl = np.asarray(short_executable_pnl_bps, dtype=np.float64)
    if long_pnl.shape != short_pnl.shape or long_pnl.ndim != 1:
        raise RuntimeError("ENTRY_CAUSAL_M1_TARGET_POLICY_TARGET_SHAPE_INVALID")
    if not np.isfinite(long_pnl).all() or not np.isfinite(short_pnl).all():
        raise RuntimeError("ENTRY_CAUSAL_M1_TARGET_POLICY_TARGET_VALUES_INVALID")
    return apply_entry_direction_target_rule(
        long_executable_pnl_bps=long_pnl,
        short_executable_pnl_bps=short_pnl,
        tradable_edge_floor_bps=normalized["tradable_edge_floor_bps"],
        side_margin_floor_bps=normalized["side_margin_floor_bps"],
    )


def causal_m1_direction_diagnostic_outcome_contract(
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Projection placed in split manifests; never a serving action rule."""

    normalized = require_causal_m1_target_policy(policy)
    return {
        "diagnostic_outcome_target_mode": ENTRY_CAUSAL_M1_DIAGNOSTIC_OUTCOME_TARGET_MODE,
        "diagnostic_outcome_label_source": ENTRY_CAUSAL_M1_DIAGNOSTIC_OUTCOME_LABEL_SOURCE,
        "diagnostic_outcome_horizon_bars": int(normalized["selected_direction_horizon_bars"]),
        "diagnostic_side_score_formula": normalized["side_score_formula"],
        "diagnostic_tradable_edge_floor_bps": float(normalized["tradable_edge_floor_bps"]),
        "diagnostic_side_margin_floor_bps": float(normalized["side_margin_floor_bps"]),
        "diagnostic_path_quality_horizon_bars": int(normalized["path_quality_horizon_bars"]),
        "diagnostic_outcome_policy_sha256": normalized["policy_sha256"],
        "entry_action_authority": False,
    }


def materialize_causal_m1_auxiliary_outcomes(
    *,
    policy: Mapping[str, Any],
    m5_decision_times: Any,
    closed_m1: pd.DataFrame,
) -> pd.DataFrame:
    """Materialize exact M1 evidence used by all non-Q Entry auxiliaries.

    Invalid M1 fills/paths stay in the returned surface with
    ``outcome_valid=False`` and NaN outcomes.  Consumers must inner-join only
    valid rows; this is intentional so no source can silently substitute an
    M5 close or bridge a market-closure gap.
    """

    normalized = require_causal_m1_target_policy(policy)
    prepared_m1 = prepare_causal_m1_quote_source(closed_m1)
    surface = build_entry_m1_fill_surface(
        m5_decision_times=m5_decision_times, closed_m1=prepared_m1
    )
    horizon = int(normalized["selected_direction_horizon_bars"])
    terminal = causal_m1_terminal_outcomes_at_horizon(
        fill_surface=surface, closed_m1=prepared_m1, horizon_m5_bars=horizon
    )
    path = causal_m1_outcomes_at_horizon(
        fill_surface=surface, closed_m1=prepared_m1, horizon_m5_bars=horizon
    )
    valid = terminal["outcome_valid"].to_numpy(dtype=bool) & path["outcome_valid"].to_numpy(dtype=bool)
    if not np.array_equal(
        terminal["exit_decision_at"].notna().to_numpy(),
        path["exit_decision_at"].notna().to_numpy(),
    ):
        raise RuntimeError("ENTRY_CAUSAL_M1_TARGET_POLICY_TERMINAL_PATH_CLOCK_MISMATCH")
    valid_rows = np.flatnonzero(valid)
    side = np.full(len(surface), -1, dtype=np.int8)
    trade = np.zeros(len(surface), dtype=bool)
    margin = np.full(len(surface), np.nan, dtype=np.float64)
    if len(valid_rows):
        target = causal_m1_direction_targets_from_policy(
            policy=normalized,
            long_executable_pnl_bps=terminal["long_executable_pnl_bps"].to_numpy(dtype=np.float64)[valid_rows],
            short_executable_pnl_bps=terminal["short_executable_pnl_bps"].to_numpy(dtype=np.float64)[valid_rows],
        )
        side[valid_rows] = target["side"]
        trade[valid_rows] = target["trade"]
        margin[valid_rows] = target["side_margin_bps"]
    return pd.DataFrame(
        {
            "time": surface["time"].to_numpy(),
            "entry_decision_at": surface["entry_decision_at"].to_numpy(),
            "exit_decision_at": terminal["exit_decision_at"].to_numpy(),
            "mfe_long_first_n_bps": path["long_mfe_bps"].to_numpy(dtype=np.float64),
            "mae_long_first_n_bps": path["long_mae_bps"].to_numpy(dtype=np.float64),
            "mfe_short_first_n_bps": path["short_mfe_bps"].to_numpy(dtype=np.float64),
            "mae_short_first_n_bps": path["short_mae_bps"].to_numpy(dtype=np.float64),
            "bad_path_long_first_n": (terminal["long_executable_pnl_bps"].to_numpy(dtype=np.float64) < 0.0).astype(np.float32),
            "bad_path_short_first_n": (terminal["short_executable_pnl_bps"].to_numpy(dtype=np.float64) < 0.0).astype(np.float32),
            "v11_pnl_long_at_dir_horizon_bps": terminal["long_executable_pnl_bps"].to_numpy(dtype=np.float64),
            "v11_pnl_short_at_dir_horizon_bps": terminal["short_executable_pnl_bps"].to_numpy(dtype=np.float64),
            "path_quality_horizon_bars": np.full(len(surface), horizon, dtype=np.int32),
            "bad_path_horizon_bars": np.full(len(surface), horizon, dtype=np.int32),
            "direction_side": side,
            "direction_trade": trade,
            "direction_side_margin_bps": margin,
            "outcome_valid": valid,
        }
    )


def fit_causal_m1_target_policy(
    *,
    closed_m5: pd.DataFrame,
    closed_m1: pd.DataFrame,
    train_start: pd.Timestamp | str,
    train_end: pd.Timestamp | str,
    source_parquet_sha256: str,
    tape_provenance_sha256: str,
    m1_source_sha256: str,
) -> dict[str, Any]:
    """Fit one policy from M1 fills and paths whose exits remain in TRAIN."""

    if not all(_is_sha256(value) for value in (
        source_parquet_sha256, tape_provenance_sha256, m1_source_sha256
    )):
        raise RuntimeError("ENTRY_CAUSAL_M1_TARGET_POLICY_SOURCE_BINDING_INVALID")
    times = _m5_times(closed_m5)
    start = _utc(train_start, name="TRAIN_START")
    end = _utc(train_end, name="TRAIN_END")
    if end <= start:
        raise RuntimeError("ENTRY_CAUSAL_M1_TARGET_POLICY_TRAIN_RANGE_INVALID")
    # The policy evaluates every allowed horizon.  Parse and validate the M1
    # parquet exactly once; the prepared arrays are read-only and retain the
    # same timestamps/quotes for each candidate, avoiding 96 full conversions.
    prepared_m1 = prepare_causal_m1_quote_source(closed_m1)
    surface = build_entry_m1_fill_surface(m5_decision_times=times, closed_m1=prepared_m1)
    candidate = (surface["entry_decision_at"] >= start).to_numpy(dtype=bool)
    # A valid maximum-horizon path proves that every contained M1 minute and
    # every shorter exit horizon exists as well.  This avoids retaining 96
    # DataFrames (hundreds of MB on the real tape) merely to discover one
    # missing minute or a TRAIN-boundary crossing.
    maximum_horizon = ENTRY_DIRECTION_TARGET_POLICY_MAX_HORIZON_BARS
    terminal_maximum = causal_m1_terminal_outcomes_at_horizon(
        fill_surface=surface, closed_m1=prepared_m1, horizon_m5_bars=maximum_horizon
    )
    exit_in_train = terminal_maximum["exit_decision_at"].notna().to_numpy() & (
        terminal_maximum["exit_decision_at"] <= end
    ).to_numpy()
    valid_all = (
        candidate
        & terminal_maximum["outcome_valid"].to_numpy(dtype=bool)
        & exit_in_train
    )
    rows = np.flatnonzero(valid_all)
    if len(rows) < 2:
        raise RuntimeError("ENTRY_CAUSAL_M1_TARGET_POLICY_TRAIN_POPULATION_EMPTY")
    entry_bid = surface["entry_bid"].to_numpy(dtype=np.float64)[rows]
    entry_ask = surface["entry_ask"].to_numpy(dtype=np.float64)[rows]
    spread_bps = (entry_ask - entry_bid) / ((entry_ask + entry_bid) * 0.5) * 1e4
    hurdle = float(np.median(spread_bps))
    if (
        not np.isfinite(spread_bps).all()
        or np.any(spread_bps <= 0.0)
        or not math.isfinite(hurdle)
        or hurdle <= 0.0
    ):
        raise RuntimeError("ENTRY_CAUSAL_M1_TARGET_POLICY_SPREAD_FIT_INVALID")
    long_delay = np.zeros(len(rows), dtype=np.int16)
    short_delay = np.zeros(len(rows), dtype=np.int16)
    unresolved_long = np.ones(len(rows), dtype=bool)
    unresolved_short = np.ones(len(rows), dtype=bool)
    for horizon in range(1, maximum_horizon + 1):
        terminal = causal_m1_terminal_outcomes_at_horizon(
            fill_surface=surface, closed_m1=prepared_m1, horizon_m5_bars=horizon
        )
        long_pnl = terminal["long_executable_pnl_bps"].to_numpy(dtype=np.float64)[rows]
        short_pnl = terminal["short_executable_pnl_bps"].to_numpy(dtype=np.float64)[rows]
        new_long = unresolved_long & (long_pnl > hurdle)
        new_short = unresolved_short & (short_pnl > hurdle)
        long_delay[new_long] = np.int16(horizon)
        short_delay[new_short] = np.int16(horizon)
        unresolved_long[new_long] = False
        unresolved_short[new_short] = False
    size = ENTRY_DIRECTION_TARGET_POLICY_MAX_HORIZON_BARS + 1
    long_counts = np.bincount(long_delay, minlength=size).astype(np.int64)
    short_counts = np.bincount(short_delay, minlength=size).astype(np.int64)
    discovery = np.cumsum(long_counts[1:] + short_counts[1:], dtype=np.int64)
    selected, distance = _selected_chord_knee(discovery)
    path = causal_m1_outcomes_at_horizon(
        fill_surface=surface, closed_m1=prepared_m1, horizon_m5_bars=selected
    ).iloc[rows]
    mfe = np.concatenate((
        path["long_mfe_bps"].to_numpy(dtype=np.float64),
        path["short_mfe_bps"].to_numpy(dtype=np.float64),
    ))
    mae = np.concatenate((
        path["long_mae_bps"].to_numpy(dtype=np.float64),
        path["short_mae_bps"].to_numpy(dtype=np.float64),
    ))
    path_score = mfe - mae
    if not np.isfinite(mfe).all() or not np.isfinite(mae).all() or np.any(mae < 0.0):
        raise RuntimeError("ENTRY_CAUSAL_M1_TARGET_POLICY_PATH_POPULATION_INVALID")
    thresholds: dict[str, float] = {}
    for name, values in (("mfe", mfe), ("mae", mae), ("path", path_score)):
        fitted = np.quantile(values, np.asarray(ENTRY_DIRECTION_TARGET_POLICY_PATH_QUANTILES), method="linear")
        thresholds.update({
            f"{name}_low_bps": float(fitted[0]),
            f"{name}_median_bps": float(fitted[1]),
            f"{name}_high_bps": float(fitted[2]),
        })
    population_stream = hashlib.sha256()
    population_stream.update(np.ascontiguousarray(times.asi8[rows], dtype="<i8").tobytes())
    population_stream.update(np.ascontiguousarray(entry_bid, dtype="<f8").tobytes())
    population_stream.update(np.ascontiguousarray(entry_ask, dtype="<f8").tobytes())
    path_stream = hashlib.sha256()
    path_stream.update(b"entry_causal_m1_path_threshold_population_v1\0")
    for values in (mfe, mae, path_score):
        path_stream.update(np.ascontiguousarray(values, dtype="<f8").tobytes())
    policy: dict[str, Any] = {
        "schema_version": ENTRY_CAUSAL_M1_TARGET_POLICY_SCHEMA_VERSION,
        "decision": "PASS", "fit_split": "train", "fit_scope": "TRAIN_ONLY",
        "fit_method": ENTRY_CAUSAL_M1_TARGET_POLICY_FIT_METHOD,
        "edge_fit_method": ENTRY_CAUSAL_M1_TARGET_POLICY_EDGE_FIT_METHOD,
        "path_threshold_fit_method": ENTRY_CAUSAL_M1_TARGET_POLICY_PATH_THRESHOLD_FIT_METHOD,
        "action_rule": ENTRY_CAUSAL_M1_TARGET_POLICY_ACTION_RULE,
        "m5_decision_clock": "authoritative_closed_m5_bar_then_exact_m1_fill",
        "source_parquet_sha256": source_parquet_sha256,
        "tape_provenance_sha256": tape_provenance_sha256,
        "m1_source_sha256": m1_source_sha256,
        "target_contract": causal_m1_target_contract(),
        "train_start_utc": start.isoformat(), "train_end_utc": end.isoformat(),
        "fit_first_m5_row": int(rows[0]), "fit_last_m5_row": int(rows[-1]),
        "fit_first_m5_time_utc": times[rows[0]].isoformat(),
        "fit_last_m5_time_utc": times[rows[-1]].isoformat(),
        "fit_population_rows": int(len(rows)),
        "fit_population_stream_sha256": population_stream.hexdigest(),
        "candidate_horizon_min_bars": 1,
        "candidate_horizon_max_bars": ENTRY_DIRECTION_TARGET_POLICY_MAX_HORIZON_BARS,
        "executable_spread_bps": hurdle,
        "spread_source_stream_sha256": hashlib.sha256(np.ascontiguousarray(spread_bps, dtype="<f8").tobytes()).hexdigest(),
        "long_first_material_profit_delay_counts": long_counts.tolist(),
        "short_first_material_profit_delay_counts": short_counts.tolist(),
        "combined_material_profit_discovery_counts": discovery.tolist(),
        "selected_direction_horizon_bars": selected,
        "selected_knee_distance_numerator": distance,
        "path_quality_horizon_bars": selected,
        "path_threshold_quantile_probabilities": list(ENTRY_DIRECTION_TARGET_POLICY_PATH_QUANTILES),
        "path_threshold_population_rows": int(len(mfe)),
        "path_threshold_stream_sha256": path_stream.hexdigest(),
        "path_aux_thresholds_bps": thresholds, "handwritten_path_thresholds": False,
        "tradable_edge_floor_bps": hurdle, "side_margin_floor_bps": hurdle,
        "early_move_threshold_bps": hurdle,
        "side_score_formula": "exact_m1_fill_executable_pnl_at_selected_horizon_bps",
        "future_outcomes_used_as_model_inputs": False, "val_test_rows_used_for_fit": 0,
    }
    policy["policy_sha256"] = canonical_causal_m1_target_policy_sha256(policy)
    return require_causal_m1_target_policy(
        policy, expected_source_parquet_sha256=source_parquet_sha256,
        expected_tape_provenance_sha256=tape_provenance_sha256,
        expected_m1_source_sha256=m1_source_sha256,
    )


def require_causal_m1_target_policy(
    value: Any,
    *,
    expected_source_parquet_sha256: str | None = None,
    expected_tape_provenance_sha256: str | None = None,
    expected_m1_source_sha256: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _POLICY_KEYS:
        raise RuntimeError("ENTRY_CAUSAL_M1_TARGET_POLICY_SCHEMA_INVALID")
    policy = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    declared_hash = policy.pop("policy_sha256", None)
    if not _is_sha256(declared_hash) or declared_hash != canonical_causal_m1_target_policy_sha256(policy):
        raise RuntimeError("ENTRY_CAUSAL_M1_TARGET_POLICY_HASH_INVALID")
    policy["policy_sha256"] = declared_hash
    if (
        policy["schema_version"] != ENTRY_CAUSAL_M1_TARGET_POLICY_SCHEMA_VERSION
        or policy["decision"] != "PASS" or policy["fit_split"] != "train"
        or policy["fit_scope"] != "TRAIN_ONLY"
        or policy["fit_method"] != ENTRY_CAUSAL_M1_TARGET_POLICY_FIT_METHOD
        or policy["edge_fit_method"] != ENTRY_CAUSAL_M1_TARGET_POLICY_EDGE_FIT_METHOD
        or policy["path_threshold_fit_method"] != ENTRY_CAUSAL_M1_TARGET_POLICY_PATH_THRESHOLD_FIT_METHOD
        or policy["action_rule"] != ENTRY_CAUSAL_M1_TARGET_POLICY_ACTION_RULE
        or policy["m5_decision_clock"] != "authoritative_closed_m5_bar_then_exact_m1_fill"
        or policy["target_contract"] != causal_m1_target_contract()
        or policy["candidate_horizon_min_bars"] != 1
        or policy["candidate_horizon_max_bars"] != ENTRY_DIRECTION_TARGET_POLICY_MAX_HORIZON_BARS
        or policy["future_outcomes_used_as_model_inputs"] is not False
        or policy["val_test_rows_used_for_fit"] != 0
        or policy["side_score_formula"] != "exact_m1_fill_executable_pnl_at_selected_horizon_bps"
    ):
        raise RuntimeError("ENTRY_CAUSAL_M1_TARGET_POLICY_CONTRACT_INVALID")
    for field, expected in (
        ("source_parquet_sha256", expected_source_parquet_sha256),
        ("tape_provenance_sha256", expected_tape_provenance_sha256),
        ("m1_source_sha256", expected_m1_source_sha256),
    ):
        if not _is_sha256(policy[field]):
            raise RuntimeError("ENTRY_CAUSAL_M1_TARGET_POLICY_SOURCE_HASH_INVALID")
        if expected is not None and policy[field] != expected:
            raise RuntimeError(f"ENTRY_CAUSAL_M1_TARGET_POLICY_{field.upper()}_MISMATCH")
    start = _utc(policy["train_start_utc"], name="TRAIN_START")
    end = _utc(policy["train_end_utc"], name="TRAIN_END")
    first = _utc(policy["fit_first_m5_time_utc"], name="FIT_FIRST_TIME")
    last = _utc(policy["fit_last_m5_time_utc"], name="FIT_LAST_TIME")
    int_fields = ("fit_first_m5_row", "fit_last_m5_row", "fit_population_rows", "selected_direction_horizon_bars", "selected_knee_distance_numerator", "path_quality_horizon_bars", "path_threshold_population_rows")
    if any(isinstance(policy[name], bool) or not isinstance(policy[name], int) for name in int_fields):
        raise RuntimeError("ENTRY_CAUSAL_M1_TARGET_POLICY_INTEGER_INVALID")
    rows = policy["fit_population_rows"]
    if (
        end <= start or last < first or policy["fit_first_m5_row"] < 0
        or policy["fit_last_m5_row"] < policy["fit_first_m5_row"] or rows < 2
        or not _is_sha256(policy["fit_population_stream_sha256"])
        or not _is_sha256(policy["spread_source_stream_sha256"])
        or policy["path_threshold_population_rows"] != 2 * rows
        or not _is_sha256(policy["path_threshold_stream_sha256"])
    ):
        raise RuntimeError("ENTRY_CAUSAL_M1_TARGET_POLICY_POPULATION_INVALID")
    numeric_fields = ("executable_spread_bps", "tradable_edge_floor_bps", "side_margin_floor_bps", "early_move_threshold_bps")
    if any(
        isinstance(policy[name], bool) or not isinstance(policy[name], (int, float))
        or not math.isfinite(float(policy[name])) or float(policy[name]) <= 0.0
        for name in numeric_fields
    ) or any(float(policy[name]) != float(policy["executable_spread_bps"]) for name in numeric_fields[1:]):
        raise RuntimeError("ENTRY_CAUSAL_M1_TARGET_POLICY_EDGE_FLOOR_INVALID")
    size = ENTRY_DIRECTION_TARGET_POLICY_MAX_HORIZON_BARS + 1
    histograms: list[np.ndarray] = []
    for name in ("long_first_material_profit_delay_counts", "short_first_material_profit_delay_counts"):
        raw = policy[name]
        if not isinstance(raw, list) or len(raw) != size or any(isinstance(item, bool) or not isinstance(item, int) for item in raw):
            raise RuntimeError("ENTRY_CAUSAL_M1_TARGET_POLICY_DELAY_COUNTS_INVALID")
        values = np.asarray(raw, dtype=np.int64)
        if np.any(values < 0) or int(values.sum()) != rows:
            raise RuntimeError("ENTRY_CAUSAL_M1_TARGET_POLICY_DELAY_COUNTS_INVALID")
        histograms.append(values)
    discovery = np.cumsum(histograms[0][1:] + histograms[1][1:], dtype=np.int64)
    selected, distance = _selected_chord_knee(discovery)
    if (
        policy["combined_material_profit_discovery_counts"] != discovery.tolist()
        or policy["selected_direction_horizon_bars"] != selected
        or policy["selected_knee_distance_numerator"] != distance
        or policy["path_quality_horizon_bars"] != selected
        or policy["path_threshold_quantile_probabilities"] != list(ENTRY_DIRECTION_TARGET_POLICY_PATH_QUANTILES)
        or policy["handwritten_path_thresholds"] is not False
    ):
        raise RuntimeError("ENTRY_CAUSAL_M1_TARGET_POLICY_DISCOVERY_INVALID")
    expected_thresholds = {f"{kind}_{quantile}_bps" for kind in ("mfe", "mae", "path") for quantile in ("low", "median", "high")}
    thresholds = policy["path_aux_thresholds_bps"]
    if not isinstance(thresholds, Mapping) or set(thresholds) != expected_thresholds or any(
        isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(float(number))
        for number in thresholds.values()
    ):
        raise RuntimeError("ENTRY_CAUSAL_M1_TARGET_POLICY_THRESHOLDS_INVALID")
    return policy
