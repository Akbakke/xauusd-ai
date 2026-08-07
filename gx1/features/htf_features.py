"""V4-only causal multi-timeframe feature and immutable-cache owner.

The sole cache source is exact native M5 OHLCV. It emits the fixed ordered
111-field surface for M5/M15/H1/H4/D1, routes Entry on M15/H1/H4/D1 and Exit on
M5/M15/H1/H4/D1, and fails closed on any schema, byte, chronology, warmup, or
feature-order mismatch. No historical cache contract or computed-feature
fallback is exposed.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

# Retained shared warmup floors used by the active context owner.
D1_EMA200_MIN_BARS = 220
H1_ATR100_MIN_BARS = 120
M15_ATR100_MIN_BARS = 200
H4_EMA50_MIN_BARS = 80
D1_PCTL252_MIN_BARS = 270

def _last_valid(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()



def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate exact observed OHLCV for the V4 feature owner."""
    required = ("open", "high", "low", "close", "volume")
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise RuntimeError(
            f"HTF_V4_VOLUME_SOURCE_MISSING: exact OHLCV source required; missing={missing}"
        )
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": _last_valid,
        "volume": "sum",
    }
    return df.loc[:, list(required)].resample(rule).agg(agg).dropna(how="all")



def _validate_m5_input(
    m5_candles: pd.DataFrame,
    *,
    require_volume: bool = False,
    bar_duration: pd.Timedelta = pd.Timedelta(minutes=5),
) -> None:
    if not isinstance(m5_candles, pd.DataFrame):
        raise TypeError(
            f"HTF_INPUT_FAIL: m5_candles must be DataFrame, got {type(m5_candles).__name__}"
        )
    if m5_candles.empty:
        raise RuntimeError("HTF_INPUT_FAIL: m5_candles must be non-empty")
    if not isinstance(bar_duration, pd.Timedelta) or bar_duration <= pd.Timedelta(0):
        raise RuntimeError("HTF_INPUT_FAIL: bar_duration must be positive")
    required_cols = ["open", "high", "low", "close"]
    if require_volume:
        required_cols.append("volume")
    missing = [c for c in required_cols if c not in m5_candles.columns]
    if missing:
        raise RuntimeError(
            f"HTF_INPUT_FAIL: m5_candles missing required columns: {missing}"
        )
    if not isinstance(m5_candles.index, pd.DatetimeIndex):
        raise RuntimeError(
            "HTF_INPUT_FAIL: m5_candles index must be DatetimeIndex"
        )
    if m5_candles.index.tz is None:
        raise RuntimeError("HTF_INPUT_FAIL: m5_candles index must be timezone-aware UTC")
    if any(pd.Timestamp(ts).utcoffset() != pd.Timedelta(0) for ts in m5_candles.index[:1]):
        raise RuntimeError("HTF_INPUT_FAIL: m5_candles index must be UTC")
    if (
        m5_candles.index.hasnans
        or not m5_candles.index.is_unique
        or not m5_candles.index.is_monotonic_increasing
    ):
        raise RuntimeError(
            "HTF_INPUT_FAIL: timestamps must be finite, unique and chronological"
        )
    if np.any(m5_candles.index.asi8 % int(bar_duration.value) != 0):
        raise RuntimeError("HTF_INPUT_FAIL: timestamps are off the declared base grid")
    numeric = m5_candles.loc[:, required_cols].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("HTF_INPUT_FAIL: exact OHLCV sources must be finite")
    open_values = numeric["open"].to_numpy(dtype=np.float64)
    high_values = numeric["high"].to_numpy(dtype=np.float64)
    low_values = numeric["low"].to_numpy(dtype=np.float64)
    close_values = numeric["close"].to_numpy(dtype=np.float64)
    if (
        np.any(open_values <= 0.0)
        or np.any(high_values <= 0.0)
        or np.any(low_values <= 0.0)
        or np.any(close_values <= 0.0)
        or np.any(high_values < close_values)
        or np.any(low_values > close_values)
        or np.any(high_values < low_values)
        or (require_volume and np.any(high_values < open_values))
        or (require_volume and np.any(low_values > open_values))
    ):
        raise RuntimeError("HTF_INPUT_FAIL: OHLC geometry is invalid")
    if require_volume and np.any(numeric["volume"].to_numpy(dtype=np.float64) <= 0.0):
        raise RuntimeError(
            "HTF_V4_VOLUME_SOURCE_INVALID: observed volume must be finite and positive"
        )


# ---------------------------------------------------------------------------
# Sole V4 per-bar multi-timeframe surface.
# ---------------------------------------------------------------------------

# Exact ordered V4 feature contract. Persistent field names ending in _v2 are
# model fields and remain unchanged; they are not compatibility APIs.
from gx1.features.smc_v1 import (  # noqa: E402
    SMC_MTF_FEATURE_NAMES_V1,
    SMC_MTF_GEOMETRY_FEATURE_NAMES_V1,
)


def _candlestick_feature_names_v4() -> tuple[str, ...]:
    from gx1.features.entry_candlestick_patterns_v1 import (
        CANDLESTICK_PATTERN_FEATURE_NAMES,
    )

    return tuple(
        f"mtf_{name.split('.', 1)[1] if '.' in name else name}"
        for name in CANDLESTICK_PATTERN_FEATURE_NAMES
    )


MULTI_TF_V4_GROUP_A_BASE_FEATURES = (
    "atr_bps_14",
    "rsi14_centered",
    "mom_5_atr",
    "mom_20_atr",
    "close_open_atr",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "ema20_dist_atr",
    "ema50_dist_atr",
    "ema100_dist_atr",
    "ema200_dist_atr",
    "ema20_slope_atr",
    "ema50_slope_atr",
    "ema200_slope_atr",
    "ema_stack_aligned_v2",
    "regime_class_id",
    "vwap_local_cycle_dist_atr",
    "vwap20_dist_atr",
    "vwap96_dist_atr",
    "vwap_local_cycle_slope_atr",
    "bb_position",
    "bb_width_atr",
    "adx_centered",
    "trend_age_bars_norm",
)
MULTI_TF_V4_CANDLESTICK_FEATURES = _candlestick_feature_names_v4()
MULTI_TF_V4_SWING_FEATURES = (
    "swing_bars_since_swing_high",
    "swing_bars_since_swing_low",
    "swing_dist_last_swing_high_atr",
    "swing_dist_last_swing_low_atr",
    "swing_retracement_from_last_impulse",
)

# Persistent model inputs that historically came from three separate HTF
# implementations.  They now have one owner: the native-M5 V4 lane.  The
# fixed 111-field matrices remain unchanged; this compact scalar surface is
# computed from the same closed OHLCV bars and projected onto either local
# decision clock.  Names are persistent model fields, not compatibility APIs.
MODEL_NATIVE_MTF_SCALAR_FIELDS_BY_TIMEFRAME_V4 = {
    "M5": (),
    "M15": (
        "M15_range_compression_ratio",
        "m15_rsi14_canon_v2",
        "m15_range_z_20_canon_v2",
        "m15_trend_sign_canon_v2",
    ),
    "H1": (
        "H1_range_compression_ratio",
        "_v1h1_ema_diff",
        "_v1h1_atr",
        "_v1h1_rsi14_z",
        "_v1h1_slope3",
        "_v1h1_slope5",
    ),
    "H4": (
        "H4_trend_sign_cat",
        "_v1h4_ema_diff",
        "_v1h4_atr",
        "_v1h4_rsi14_z",
        "_v1h4_slope3",
        "_v1h4_slope5",
    ),
    "D1": (
        "D1_dist_from_ema200_atr",
        "D1_atr_percentile_252",
        "d1_atr14_canon_v2",
        "d1_rsi14_canon_v2",
        "d1_ema_slope_20_canon_v2",
        "d1_range_z_20_canon_v2",
        "d1_close_pct_in_20day_range_canon_v2",
        "d1_pct_change_5_canon_v2",
    ),
}
MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4 = (
    "D1_dist_from_ema200_atr",
    "H1_range_compression_ratio",
    "D1_atr_percentile_252",
    "M15_range_compression_ratio",
    "_v1h1_ema_diff",
    "_v1h1_atr",
    "_v1h1_rsi14_z",
    "_v1h1_slope3",
    "_v1h1_slope5",
    "_v1h4_ema_diff",
    "_v1h4_atr",
    "_v1h4_rsi14_z",
    "_v1h4_slope3",
    "_v1h4_slope5",
    "d1_atr14_canon_v2",
    "d1_rsi14_canon_v2",
    "d1_ema_slope_20_canon_v2",
    "d1_range_z_20_canon_v2",
    "d1_close_pct_in_20day_range_canon_v2",
    "d1_pct_change_5_canon_v2",
    "m15_rsi14_canon_v2",
    "m15_range_z_20_canon_v2",
    "m15_trend_sign_canon_v2",
    "H4_trend_sign_cat",
)
MODEL_NATIVE_MTF_SCALAR_CONTRACT_V4 = (
    "model_native_mtf_scalar_owner_native_m5_v4"
)
MULTI_TF_PER_BAR_FEATURES_V4 = (
    MULTI_TF_V4_GROUP_A_BASE_FEATURES
    + MULTI_TF_V4_CANDLESTICK_FEATURES
    + MULTI_TF_V4_SWING_FEATURES
    + SMC_MTF_FEATURE_NAMES_V1
    + SMC_MTF_GEOMETRY_FEATURE_NAMES_V1
)
MULTI_TF_FEATURE_COUNT_V4 = len(MULTI_TF_PER_BAR_FEATURES_V4)
MULTI_TF_FEATURE_NAMES_SHA256_V4 = hashlib.sha256(
    json.dumps(
        list(MULTI_TF_PER_BAR_FEATURES_V4),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()
HTF_V4_MATRIX_CONTRACT = "HTF_V4_EIGHT_FAMILY_CAUSAL_MATRIX_V2"
HTF_V4_CACHE_SCHEMA_VERSION = "htf_v4_disk_cache_manifest_v4"
HTF_V4_CACHE_BUILDER_VERSION = (
    "prebuild_multi_tf_cache_v4_only_closed_resample_20260804"
)
HTF_V4_FULL_INPUT_LIVENESS_SCHEMA_VERSION = "htf_v4_full_input_liveness_v2"

# These are deliberate aliases inside the fixed 111-field model surface.
HTF_V4_DECLARED_ALIAS_PAIRS = frozenset(
    {
        ("body_pct", "mtf_pattern_body_share"),
        ("upper_wick_pct", "mtf_pattern_upper_wick_share"),
        ("lower_wick_pct", "mtf_pattern_lower_wick_share"),
    }
)

MULTI_TF_RESAMPLE_RULES = {
    # Resample cadence only. Entry window lengths are explicit recipe inputs
    # and must form a strictly increasing wall-clock coverage pyramid.
    "M5": "5min",
    "M15": "15min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}
MULTI_TF_TIMEFRAMES = tuple(MULTI_TF_RESAMPLE_RULES)
MULTI_TF_TIMEFRAMES_LOWER = tuple(
    timeframe.lower() for timeframe in MULTI_TF_TIMEFRAMES
)
MULTI_TF_TIMEFRAMES_LOWER_M5_LAST = (
    *MULTI_TF_TIMEFRAMES_LOWER[1:],
    MULTI_TF_TIMEFRAMES_LOWER[0],
)
MULTI_TF_BARS_IN_M5 = {
    timeframe.lower(): int(
        pd.Timedelta(rule) / pd.Timedelta(MULTI_TF_RESAMPLE_RULES["M5"])
    )
    for timeframe, rule in MULTI_TF_RESAMPLE_RULES.items()
}

# Pandas-Timedelta shift per TF: ensures we use only CLOSED bars at-or-before t
MULTI_TF_SHIFT = {
    "M5": pd.Timedelta(minutes=5),
    "M15": pd.Timedelta(minutes=15),
    "H1": pd.Timedelta(hours=1),
    "H4": pd.Timedelta(hours=4),
    "D1": pd.Timedelta(days=1),
}
MULTI_TF_PYRAMID_SCHEMA_VERSION = "entry_multi_tf_causal_resolution_pyramid_v1"


def multi_tf_last_closed_label(
    decision_bar_start: pd.Timestamp | str,
    timeframe: str,
    *,
    base_bar_duration: pd.Timedelta = pd.Timedelta(minutes=5),
) -> pd.Timestamp:
    """Return the exact opening label of the last closed bar for one TF.

    ``decision_bar_start`` is the opening timestamp of an observed M5 candle.
    Its information becomes available five minutes later.  HTF resample labels
    are bar-opening timestamps, so the availability cutoff must be shifted by
    the full HTF duration and then floored to that timeframe's UTC grid.
    """
    if timeframe not in MULTI_TF_RESAMPLE_RULES:
        raise RuntimeError(
            f"HTF_V4_TIMEFRAME_INVALID: {timeframe!r}"
        )
    if not isinstance(base_bar_duration, pd.Timedelta) or base_bar_duration <= pd.Timedelta(0):
        raise RuntimeError("HTF_V4_BASE_BAR_DURATION_INVALID")
    timestamp = pd.Timestamp(decision_bar_start)
    if timestamp.tz is None or timestamp.utcoffset() != pd.Timedelta(0):
        raise RuntimeError(
            "HTF_V4_DECISION_TIMESTAMP_INVALID: timezone-aware UTC required"
        )
    return (
        timestamp
        + base_bar_duration
        - MULTI_TF_SHIFT[timeframe]
    ).floor(MULTI_TF_RESAMPLE_RULES[timeframe])


def build_multi_tf_v4_closed_timestamp_indices(
    m5_index: pd.DatetimeIndex,
) -> dict[str, pd.DatetimeIndex]:
    """Derive the sole V4 cache axis from an exact native-M5 source."""
    base_bar_duration = pd.Timedelta(minutes=5)
    if not isinstance(m5_index, pd.DatetimeIndex) or len(m5_index) == 0:
        raise RuntimeError(
            "HTF_V4_SOURCE_TIMESTAMP_GEOMETRY_INVALID: non-empty "
            "DatetimeIndex required"
        )
    m5_index = m5_index.as_unit("ns")
    if (
        m5_index.tz is None
        or m5_index.hasnans
        or not m5_index.is_unique
        or not m5_index.is_monotonic_increasing
        or m5_index[0].utcoffset() != pd.Timedelta(0)
    ):
        raise RuntimeError(
            "HTF_V4_SOURCE_TIMESTAMP_GEOMETRY_INVALID: exact chronological "
            "unique UTC timestamps required"
        )
    if not m5_index.floor(base_bar_duration).equals(m5_index):
        raise RuntimeError(
            "HTF_V4_SOURCE_TIMESTAMP_GEOMETRY_INVALID: source timestamps "
            "must lie on the exact M5 UTC grid"
        )

    expected: dict[str, pd.DatetimeIndex] = {}
    for timeframe, rule in MULTI_TF_RESAMPLE_RULES.items():
        labels = m5_index.floor(rule).drop_duplicates()
        last_closed = multi_tf_last_closed_label(
            m5_index[-1],
            timeframe,
            base_bar_duration=base_bar_duration,
        )
        labels = labels[labels <= last_closed]
        if len(labels) and m5_index[0] > labels[0]:
            labels = labels[1:]
        if len(labels) == 0:
            raise RuntimeError(
                f"HTF_V4_NO_COMPLETE_RESAMPLED_BARS: {timeframe}"
            )
        expected[timeframe] = labels
    return expected


def require_multi_tf_resolution_pyramid(
    per_tf_seq_lens: dict[str, int],
) -> dict[str, object]:
    """Validate explicit windows as strictly increasing wall-clock coverage."""
    expected_tfs = tuple(MULTI_TF_RESAMPLE_RULES)
    if not isinstance(per_tf_seq_lens, dict) or tuple(per_tf_seq_lens) != expected_tfs:
        raise RuntimeError(
            "MULTI_TF_RESOLUTION_PYRAMID_ORDER_INVALID: exact "
            "M5/M15/H1/H4/D1 declaration required"
        )
    lengths: dict[str, int] = {}
    coverage_seconds: dict[str, int] = {}
    for tf in expected_tfs:
        value = per_tf_seq_lens[tf]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(
                f"MULTI_TF_RESOLUTION_PYRAMID_LENGTH_INVALID: {tf}={value!r}"
            )
        lengths[tf] = int(value)
        coverage_seconds[tf] = int(
            value * MULTI_TF_SHIFT[tf].total_seconds()
        )
    spans = tuple(coverage_seconds.values())
    if any(left >= right for left, right in zip(spans, spans[1:])):
        raise RuntimeError(
            "MULTI_TF_RESOLUTION_PYRAMID_COVERAGE_INVALID: progressively "
            f"coarser timeframes must cover strictly older history; {coverage_seconds}"
        )
    payload: dict[str, object] = {
        "schema_version": MULTI_TF_PYRAMID_SCHEMA_VERSION,
        "timeframe_order": list(expected_tfs),
        "per_tf_seq_lens": lengths,
        "coverage_seconds": coverage_seconds,
        "strictly_increasing_wall_clock_coverage": True,
    }
    payload["contract_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return payload


def build_multi_tf_v4_liveness_contract(
    features: dict[str, pd.DataFrame],
) -> dict[str, object]:
    """Prove every V4 field is finite, variable and non-duplicated on every TF."""

    if tuple(features) != tuple(MULTI_TF_RESAMPLE_RULES):
        raise RuntimeError(
            "HTF_V4_LIVENESS_TIMEFRAME_ORDER_INVALID: exact "
            "M5/M15/H1/H4/D1 required"
        )
    failures: list[str] = []
    timeframe_rows: dict[str, object] = {}
    for tf_name in MULTI_TF_RESAMPLE_RULES:
        frame = features[tf_name]
        if (
            not isinstance(frame, pd.DataFrame)
            or tuple(frame.columns) != MULTI_TF_PER_BAR_FEATURES_V4
            or frame.attrs.get("htf_feature_contract")
            != HTF_V4_MATRIX_CONTRACT
        ):
            raise RuntimeError(
                f"HTF_V4_LIVENESS_SURFACE_INVALID: {tf_name}"
            )
        values = np.asarray(frame.attrs.get("feats_np"))
        warmup_rows = frame.attrs.get("causal_warmup_rows")
        if (
            values.dtype != np.dtype(np.float32)
            or values.shape
            != (len(frame), MULTI_TF_FEATURE_COUNT_V4)
            or isinstance(warmup_rows, bool)
            or not isinstance(warmup_rows, (int, np.integer))
            or not 0 <= int(warmup_rows) < len(frame)
        ):
            raise RuntimeError(
                f"HTF_V4_LIVENESS_ARRAY_INVALID: {tf_name}"
            )
        warmup = int(warmup_rows)
        live = values[warmup:].astype(np.float64, copy=False)
        if not np.isfinite(live).all():
            failures.append(f"{tf_name}:nonfinite_post_warmup")
        field_stats: dict[str, object] = {}
        column_hash_owner: dict[str, str] = {}
        duplicate_pairs: list[list[str]] = []
        constant_fields: list[str] = []
        for index, feature_name in enumerate(MULTI_TF_PER_BAR_FEATURES_V4):
            column = live[:, index]
            unique_count = int(np.unique(column).size)
            standard_deviation = float(np.std(column, dtype=np.float64))
            nonzero_fraction = float(np.mean(np.abs(column) > 1e-12))
            digest = hashlib.sha256(
                np.ascontiguousarray(column).view(np.uint8)
            ).hexdigest()
            if unique_count <= 1 or standard_deviation <= 0.0:
                constant_fields.append(feature_name)
            prior = column_hash_owner.get(digest)
            if prior is not None:
                pair = (prior, feature_name)
                if pair not in HTF_V4_DECLARED_ALIAS_PAIRS:
                    duplicate_pairs.append([prior, feature_name])
            else:
                column_hash_owner[digest] = feature_name
            field_stats[feature_name] = {
                "unique_count": unique_count,
                "mean": float(np.mean(column, dtype=np.float64)),
                "std": standard_deviation,
                "minimum": float(np.min(column)),
                "maximum": float(np.max(column)),
                "nonzero_fraction": nonzero_fraction,
                "values_sha256": digest,
            }
        if constant_fields:
            failures.append(
                f"{tf_name}:constant_fields={constant_fields}"
            )
        if duplicate_pairs:
            failures.append(
                f"{tf_name}:exact_duplicate_fields={duplicate_pairs}"
            )
        timeframe_rows[tf_name] = {
            "rows": int(len(frame)),
            "warmup_rows": warmup,
            "live_rows": int(len(live)),
            "constant_fields": constant_fields,
            "exact_duplicate_pairs": duplicate_pairs,
            "fields": field_stats,
        }
    payload: dict[str, object] = {
        "schema_version": HTF_V4_FULL_INPUT_LIVENESS_SCHEMA_VERSION,
        "matrix_contract": HTF_V4_MATRIX_CONTRACT,
        "feature_names_sha256": MULTI_TF_FEATURE_NAMES_SHA256_V4,
        "timeframe_order": list(MULTI_TF_RESAMPLE_RULES),
        "decision": "PASS" if not failures else "FAIL",
        "failures": failures,
        "timeframes": timeframe_rows,
    }
    payload["contract_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return payload


def require_multi_tf_v4_liveness_contract(
    value: object,
) -> dict[str, object]:
    """Validate the exact immutable V4 per-field/per-timeframe proof."""

    if not isinstance(value, dict):
        raise RuntimeError("HTF_V4_LIVENESS_CONTRACT_MISSING")
    expected_keys = {
        "schema_version",
        "matrix_contract",
        "feature_names_sha256",
        "timeframe_order",
        "decision",
        "failures",
        "timeframes",
        "contract_sha256",
    }
    if set(value) != expected_keys:
        raise RuntimeError("HTF_V4_LIVENESS_CONTRACT_KEYS_INVALID")
    identity_payload = {
        key: item for key, item in value.items() if key != "contract_sha256"
    }
    expected_sha = hashlib.sha256(
        json.dumps(
            identity_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if value.get("contract_sha256") != expected_sha:
        raise RuntimeError("HTF_V4_LIVENESS_CONTRACT_IDENTITY_INVALID")
    if (
        value.get("schema_version")
        != HTF_V4_FULL_INPUT_LIVENESS_SCHEMA_VERSION
        or value.get("matrix_contract") != HTF_V4_MATRIX_CONTRACT
        or value.get("feature_names_sha256")
        != MULTI_TF_FEATURE_NAMES_SHA256_V4
        or value.get("timeframe_order") != list(MULTI_TF_RESAMPLE_RULES)
        or value.get("decision") != "PASS"
        or value.get("failures") != []
    ):
        raise RuntimeError("HTF_V4_LIVENESS_CONTRACT_DECISION_INVALID")
    timeframes = value.get("timeframes")
    if not isinstance(timeframes, dict) or set(timeframes) != set(
        MULTI_TF_RESAMPLE_RULES
    ):
        raise RuntimeError("HTF_V4_LIVENESS_TIMEFRAME_ORDER_INVALID")
    expected_tf_keys = {
        "rows",
        "warmup_rows",
        "live_rows",
        "constant_fields",
        "exact_duplicate_pairs",
        "fields",
    }
    expected_stat_keys = {
        "unique_count",
        "mean",
        "std",
        "minimum",
        "maximum",
        "nonzero_fraction",
        "values_sha256",
    }
    for tf_name, row in timeframes.items():
        if not isinstance(row, dict) or set(row) != expected_tf_keys:
            raise RuntimeError(f"HTF_V4_LIVENESS_TF_KEYS_INVALID: {tf_name}")
        rows = row.get("rows")
        warmup = row.get("warmup_rows")
        live_rows = row.get("live_rows")
        if (
            isinstance(rows, bool)
            or not isinstance(rows, int)
            or rows <= 0
            or isinstance(warmup, bool)
            or not isinstance(warmup, int)
            or not 0 <= warmup < rows
            or live_rows != rows - warmup
            or row.get("constant_fields") != []
            or row.get("exact_duplicate_pairs") != []
        ):
            raise RuntimeError(f"HTF_V4_LIVENESS_TF_DECISION_INVALID: {tf_name}")
        fields = row.get("fields")
        if not isinstance(fields, dict) or set(fields) != set(
            MULTI_TF_PER_BAR_FEATURES_V4
        ):
            raise RuntimeError(f"HTF_V4_LIVENESS_FIELDS_INVALID: {tf_name}")
        for field_name, stats in fields.items():
            if not isinstance(stats, dict) or set(stats) != expected_stat_keys:
                raise RuntimeError(
                    f"HTF_V4_LIVENESS_STATS_KEYS_INVALID: {tf_name}:{field_name}"
                )
            unique_count = stats.get("unique_count")
            numeric = [
                stats.get(name)
                for name in (
                    "mean",
                    "std",
                    "minimum",
                    "maximum",
                    "nonzero_fraction",
                )
            ]
            if (
                isinstance(unique_count, bool)
                or not isinstance(unique_count, int)
                or unique_count <= 1
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    for item in numeric
                )
                or float(stats["std"]) <= 0.0
                or not 0.0 < float(stats["nonzero_fraction"]) <= 1.0
                or float(stats["minimum"]) > float(stats["maximum"])
                or not isinstance(stats.get("values_sha256"), str)
                or len(stats["values_sha256"]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in stats["values_sha256"]
                )
            ):
                raise RuntimeError(
                    f"HTF_V4_LIVENESS_STATS_INVALID: {tf_name}:{field_name}"
                )
    return value


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder-style RSI on close series. Returns Series indexed like close."""
    diff = close.diff()
    gain = diff.where(diff > 0, 0.0)
    loss = -diff.where(diff < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False).mean()
    rs = avg_gain / np.maximum(avg_loss, 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


def _rolling_vwap(close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """Rolling N-bar VWAP from observed volume only."""
    if volume.isna().any() or (~np.isfinite(volume.to_numpy(dtype=np.float64))).any():
        raise RuntimeError("HTF_V4_VOLUME_SOURCE_INVALID: rolling VWAP volume is non-finite")
    if (volume <= 0.0).any():
        raise RuntimeError("HTF_V4_VOLUME_SOURCE_INVALID: rolling VWAP volume must be positive")
    pv = close * volume
    pv_sum = pv.rolling(window, min_periods=1).sum()
    v_sum = volume.rolling(window, min_periods=1).sum()
    return pv_sum / v_sum


def _session_vwap(close: pd.Series, volume: pd.Series) -> pd.Series:
    """VWAP reset at each calendar day's midnight UTC."""
    if volume.isna().any() or (~np.isfinite(volume.to_numpy(dtype=np.float64))).any():
        raise RuntimeError("HTF_V4_VOLUME_SOURCE_INVALID: session VWAP volume is non-finite")
    if (volume <= 0.0).any():
        raise RuntimeError("HTF_V4_VOLUME_SOURCE_INVALID: session VWAP volume must be positive")
    pv = close * volume
    # Group by date — cumulative within day
    grp = close.index.normalize()
    pv_cs = pv.groupby(grp).cumsum()
    v_cs = volume.groupby(grp).cumsum()
    return pv_cs / v_cs


def _adx14(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 14,
) -> pd.Series:
    """Welles Wilder's ADX with explicit causal warmup."""
    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    plus_dm = pd.Series(plus_dm, index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)
    tr = pd.concat([
        (high - low).abs(),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/n, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0/n, adjust=False).mean() / np.maximum(atr, 1e-12)
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0/n, adjust=False).mean() / np.maximum(atr, 1e-12)
    dx = 100.0 * (plus_di - minus_di).abs() / np.maximum(plus_di + minus_di, 1e-12)
    adx = dx.ewm(alpha=1.0/n, adjust=False).mean()
    adx.iloc[: 2 * n - 1] = np.nan
    return adx


def _regime_class(stack_aligned: pd.Series, ema200_slope: pd.Series, atr_safe: pd.Series) -> pd.Series:
    """Combine EMA-stack alignment + EMA200 slope into 5-class regime enum.

    0 = range (stack=0)
    1 = uptrend_low (stack=+1, slope <= +0.3 ATR)
    2 = uptrend_high (stack=+1, slope > +0.3 ATR)
    3 = downtrend_low (stack=-1, slope >= -0.3 ATR)
    4 = downtrend_high (stack=-1, slope < -0.3 ATR)
    """
    slope_atr = ema200_slope / atr_safe
    valid = stack_aligned.notna() & slope_atr.notna()
    out = pd.Series(np.nan, index=stack_aligned.index, dtype=np.float64)
    out.loc[valid] = 0.0
    up = valid & (stack_aligned == 1)
    down = valid & (stack_aligned == -1)
    out.loc[up] = np.where(slope_atr.loc[up] > 0.3, 2.0, 1.0)
    out.loc[down] = np.where(slope_atr.loc[down] < -0.3, 4.0, 3.0)
    return out


def _trend_age_bars(stack_aligned: pd.Series) -> pd.Series:
    """Number of consecutive bars since the EMA stack last changed sign."""
    # Convert to int sign sequence; count runs
    chg = (stack_aligned != stack_aligned.shift(1)).cumsum()
    return stack_aligned.groupby(chg).cumcount().astype(float)


def validate_causal_feature_matrix(
    values,
    *,
    expected_width: int,
    context: str,
) -> int:
    """Validate an exact feature matrix and return its warmup-prefix length.

    A model feature may be unavailable only in one chronological prefix. Once a
    complete row exists, every later row must be finite. Numeric sentinels are
    deliberately not introduced here.
    """
    arr = np.asarray(values)
    if arr.ndim != 2 or arr.shape[1] != int(expected_width) or arr.shape[0] == 0:
        raise RuntimeError(
            f"[{context}] feature matrix must have non-zero shape (N, {expected_width}); "
            f"observed={arr.shape}"
        )
    try:
        numeric = arr.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"[{context}] feature matrix must be numeric") from exc
    if np.isinf(numeric).any():
        raise RuntimeError(f"[{context}] feature matrix contains infinity")
    complete = np.isfinite(numeric).all(axis=1)
    if not complete.any():
        return int(len(numeric))
    first_complete = int(np.argmax(complete))
    if not complete[first_complete:].all():
        raise RuntimeError(
            f"[{context}] non-finite feature rows are not one causal warmup prefix"
        )
    return first_complete


def compute_per_bar_features_v4(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute the exact fixed 111-field V4 surface directly from one OHLCV TF."""

    from gx1.features.entry_candlestick_patterns_v1 import (
        build_entry_candlestick_pattern_layer,
    )
    from gx1.features.swing_structure_v1 import compute_swing_structure_features
    from gx1.features.smc_v1 import compute_smc_mtf_primitives_v1

    _validate_m5_input(ohlcv, require_volume=True)
    df = ohlcv[["open", "high", "low", "close", "volume"]].astype(
        np.float64
    ).copy()
    out = pd.DataFrame(index=df.index, dtype=np.float64)

    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    atr14 = _atr(high, low, close, 14)
    atr_floor = np.maximum(close * 1e-4, 1e-3)
    atr_safe = np.maximum(atr14, atr_floor)

    out["atr_bps_14"] = (atr14 / close * 1e4).clip(0, 500)
    rsi = _rsi(close, 14)
    rsi.iloc[:14] = np.nan
    out["rsi14_centered"] = ((rsi - 50.0) / 50.0).clip(-1.0, 1.0)
    for lag in (5, 20):
        out[f"mom_{lag}_atr"] = (
            (close - close.shift(lag)) / atr_safe
        ).clip(-10.0, 10.0)
    out["close_open_atr"] = ((close - open_) / atr_safe).clip(-10.0, 10.0)

    bar_range = np.maximum(high - low, atr_floor)
    body = (close - open_).abs()
    upper_wick = high - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - low
    out["body_pct"] = (body / bar_range).clip(0.0, 1.0)
    out["upper_wick_pct"] = (upper_wick / bar_range).clip(0.0, 1.0)
    out["lower_wick_pct"] = (lower_wick / bar_range).clip(0.0, 1.0)

    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    ema100 = _ema(close, 100)
    ema200 = _ema(close, 200)
    out["ema20_dist_atr"] = ((close - ema20) / atr_safe).clip(-10.0, 10.0)
    out["ema50_dist_atr"] = ((close - ema50) / atr_safe).clip(-15.0, 15.0)
    out["ema100_dist_atr"] = ((close - ema100) / atr_safe).clip(-20.0, 20.0)
    out["ema200_dist_atr"] = ((close - ema200) / atr_safe).clip(-30.0, 30.0)
    out["ema20_slope_atr"] = (
        (ema20 - ema20.shift(5)) / atr_safe
    ).clip(-5.0, 5.0)
    out["ema50_slope_atr"] = (
        (ema50 - ema50.shift(5)) / atr_safe
    ).clip(-5.0, 5.0)
    out["ema200_slope_atr"] = (
        (ema200 - ema200.shift(5)) / atr_safe
    ).clip(-5.0, 5.0)

    bull = (ema20 > ema50) & (ema50 > ema100) & (ema100 > ema200)
    bear = (ema20 < ema50) & (ema50 < ema100) & (ema100 < ema200)
    stack = pd.Series(0, index=close.index)
    stack[bull] = 1
    stack[bear] = -1
    out["ema_stack_aligned_v2"] = stack.astype(float)
    out["regime_class_id"] = _regime_class(
        stack,
        ema200 - ema200.shift(5),
        atr_safe,
    )

    if len(close) >= 2:
        median_delta_hours = float(
            (
                close.index.to_series().diff().median() or pd.Timedelta(0)
            ).total_seconds()
            / 3600.0
        )
    else:
        median_delta_hours = 0.0
    local_cycle_vwap = (
        _rolling_vwap(close, volume, 5)
        if median_delta_hours >= 23.0
        else _session_vwap(close, volume)
    )
    out["vwap_local_cycle_dist_atr"] = (
        (close - local_cycle_vwap) / atr_safe
    ).clip(-15.0, 15.0)
    vwap20 = _rolling_vwap(close, volume, 20)
    out["vwap20_dist_atr"] = ((close - vwap20) / atr_safe).clip(-10.0, 10.0)
    vwap96 = _rolling_vwap(close, volume, 96)
    out["vwap96_dist_atr"] = ((close - vwap96) / atr_safe).clip(-15.0, 15.0)
    out["vwap_local_cycle_slope_atr"] = (
        (local_cycle_vwap - local_cycle_vwap.shift(5)) / atr_safe
    ).clip(-5.0, 5.0)

    sma20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std()
    bb_upper = sma20 + 2.0 * std20
    bb_lower = sma20 - 2.0 * std20
    bb_width = bb_upper - bb_lower
    out["bb_position"] = (
        (close - bb_lower) / np.maximum(bb_width, atr_floor)
    ).clip(0.0, 1.0)
    out["bb_width_atr"] = (bb_width / atr_safe).clip(0.0, 20.0)
    adx = _adx14(high, low, close, 14)
    out["adx_centered"] = ((adx - 25.0) / 25.0).clip(-1.0, 3.0)
    age = _trend_age_bars(stack).clip(upper=500.0)
    out["trend_age_bars_norm"] = np.log1p(age) / np.log1p(500.0)
    out = out.loc[:, list(MULTI_TF_V4_GROUP_A_BASE_FEATURES)]

    candle_source = df[["open", "high", "low", "close"]].copy()
    candle_source.index.name = "time"
    candle_values, candle_names = build_entry_candlestick_pattern_layer(
        candle_source.reset_index()
    )
    observed_candle_names = tuple(
        f"mtf_{name.split('.', 1)[1] if '.' in name else name}"
        for name in candle_names
    )
    candle_values = np.asarray(candle_values, dtype=np.float64)
    if (
        candle_values.shape
        != (len(out), len(MULTI_TF_V4_CANDLESTICK_FEATURES))
        or observed_candle_names != MULTI_TF_V4_CANDLESTICK_FEATURES
    ):
        raise RuntimeError("HTF_V4_CANDLESTICK_CONTRACT_INVALID")
    for name, values in zip(
        MULTI_TF_V4_CANDLESTICK_FEATURES,
        candle_values.T,
        strict=True,
    ):
        out[name] = values

    swing = compute_swing_structure_features(
        high.to_numpy(dtype=np.float64),
        low.to_numpy(dtype=np.float64),
        close.to_numpy(dtype=np.float64),
    )
    for name in MULTI_TF_V4_SWING_FEATURES:
        source_name = name.removeprefix("swing_")
        if source_name not in swing:
            raise RuntimeError(
                f"HTF_V4_SWING_FIELD_MISSING: {source_name}"
            )
        out[name] = np.asarray(swing[source_name], dtype=np.float64)

    smc_source = df[["high", "low", "close"]].copy()
    smc_source["atr"] = atr14
    primitives = compute_smc_mtf_primitives_v1(smc_source)
    if not primitives.index.equals(out.index):
        raise RuntimeError("HTF_V4_SMC_ROW_AXIS_MISMATCH")
    out = pd.concat((out, primitives), axis=1)
    if tuple(out.columns) != MULTI_TF_PER_BAR_FEATURES_V4:
        raise RuntimeError(
            "HTF_V4_COLUMN_ORDER_INVALID: "
            f"observed={tuple(out.columns)}"
        )
    validate_causal_feature_matrix(
        out.to_numpy(dtype=np.float64, copy=False),
        expected_width=MULTI_TF_FEATURE_COUNT_V4,
        context="HTF_V4_CAUSAL_FEATURES",
    )
    return out.astype(np.float32)


def _model_native_rsi_z48_v4(close: pd.Series) -> pd.Series:
    rsi = _rsi(close, 14)
    mean = rsi.rolling(48, min_periods=24).mean()
    std = rsi.rolling(48, min_periods=24).std(ddof=0).replace(0.0, np.nan)
    return (rsi - mean) / std


def _model_native_percentile_last_v4(values: np.ndarray) -> float:
    observed = np.asarray(values, dtype=np.float64)
    if not np.isfinite(observed).all():
        return float("nan")
    return float(np.mean(observed <= observed[-1]))


def _model_native_htf_slope_v4(
    values: pd.Series,
    *,
    order: int,
) -> pd.Series:
    """Compute the retained causal slope formula on the native HTF clock."""

    source = values.to_numpy(dtype=np.float64)
    if source.ndim != 1 or order not in {3, 5} or len(source) < order:
        raise RuntimeError("HTF_V4_MODEL_NATIVE_HTF_SLOPE_INPUT_INVALID")
    delta = np.diff(source, n=order, prepend=source[:order])
    shifted = np.roll(delta, 1)
    shifted[0] = 0.0
    return pd.Series(
        np.nan_to_num(shifted, nan=0.0),
        index=values.index,
        dtype=np.float64,
    )


def _compute_model_native_mtf_scalar_frame_v4(
    ohlcv: pd.DataFrame,
    *,
    timeframe: str,
) -> pd.DataFrame:
    """Compute the compact persistent scalar surface on one closed TF clock."""

    if timeframe not in MODEL_NATIVE_MTF_SCALAR_FIELDS_BY_TIMEFRAME_V4:
        raise RuntimeError(
            f"HTF_V4_MODEL_NATIVE_SCALAR_TIMEFRAME_INVALID: {timeframe!r}"
        )
    expected_fields = MODEL_NATIVE_MTF_SCALAR_FIELDS_BY_TIMEFRAME_V4[timeframe]
    if timeframe == "M5":
        return pd.DataFrame(index=ohlcv.index)
    _validate_m5_input(ohlcv, require_volume=True)
    source = ohlcv.loc[:, ["open", "high", "low", "close", "volume"]].astype(
        np.float64
    )
    high = source["high"]
    low = source["low"]
    close = source["close"]
    atr14 = _atr(high, low, close, 14)
    out = pd.DataFrame(index=source.index)

    if timeframe in {"H1", "H4"}:
        prefix = "_v1h1" if timeframe == "H1" else "_v1h4"
        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)
        if timeframe == "H1":
            atr100 = _atr(high, low, close, 100)
            compression = atr14 / np.maximum(atr100, 1e-9)
            compression.iloc[: H1_ATR100_MIN_BARS - 1] = np.nan
            out["H1_range_compression_ratio"] = compression
        else:
            mid = (high + low) * 0.5
            ema50 = _ema(mid, 50)
            category = np.sign(mid - ema50) + 1.0
            category.iloc[: H4_EMA50_MIN_BARS - 1] = np.nan
            out["H4_trend_sign_cat"] = category
        ema_diff = ema12 - ema26
        out[f"{prefix}_ema_diff"] = ema_diff
        out[f"{prefix}_atr"] = atr14
        out[f"{prefix}_rsi14_z"] = _model_native_rsi_z48_v4(close)
        out[f"{prefix}_slope3"] = _model_native_htf_slope_v4(
            ema_diff,
            order=3,
        )
        out[f"{prefix}_slope5"] = _model_native_htf_slope_v4(
            ema_diff,
            order=5,
        )
    elif timeframe == "M15":
        atr100 = _atr(high, low, close, 100)
        compression = atr14 / np.maximum(atr100, 1e-9)
        compression.iloc[: M15_ATR100_MIN_BARS - 1] = np.nan
        out["M15_range_compression_ratio"] = compression
        out["m15_rsi14_canon_v2"] = _rsi(close, 14)
        bar_range = high - low
        range_mean = bar_range.rolling(20, min_periods=5).mean()
        range_std = bar_range.rolling(20, min_periods=5).std().replace(
            0.0, np.nan
        )
        out["m15_range_z_20_canon_v2"] = (
            bar_range - range_mean
        ) / range_std
        out["m15_trend_sign_canon_v2"] = np.sign(
            _ema(close, 5) - _ema(close, 20)
        )
    elif timeframe == "D1":
        mid = (high + low) * 0.5
        ema200 = _ema(mid, 200)
        distance = (mid - ema200) / np.maximum(atr14, 1e-9)
        distance.iloc[: D1_EMA200_MIN_BARS - 1] = np.nan
        out["D1_dist_from_ema200_atr"] = distance
        percentile = atr14.rolling(252, min_periods=252).apply(
            _model_native_percentile_last_v4,
            raw=True,
        )
        percentile.iloc[: D1_PCTL252_MIN_BARS - 1] = np.nan
        out["D1_atr_percentile_252"] = percentile
        out["d1_atr14_canon_v2"] = atr14
        out["d1_rsi14_canon_v2"] = _rsi(close, 14)
        ema20 = _ema(close, 20)
        out["d1_ema_slope_20_canon_v2"] = ema20 - ema20.shift(5)
        bar_range = high - low
        range_mean = bar_range.rolling(20, min_periods=5).mean()
        range_std = bar_range.rolling(20, min_periods=5).std().replace(
            0.0, np.nan
        )
        out["d1_range_z_20_canon_v2"] = (
            bar_range - range_mean
        ) / range_std
        high20 = high.rolling(20, min_periods=5).max()
        low20 = low.rolling(20, min_periods=5).min()
        out["d1_close_pct_in_20day_range_canon_v2"] = (
            close - low20
        ) / (high20 - low20).replace(0.0, np.nan)
        out["d1_pct_change_5_canon_v2"] = close.pct_change(5) * 10000.0

    if tuple(out.columns) != expected_fields:
        raise RuntimeError(
            "HTF_V4_MODEL_NATIVE_SCALAR_ORDER_INVALID: "
            f"timeframe={timeframe} observed={tuple(out.columns)} "
            f"expected={expected_fields}"
        )
    values = out.to_numpy(dtype=np.float64, copy=False)
    validate_causal_feature_matrix(
        values,
        expected_width=len(expected_fields),
        context=f"HTF_V4_MODEL_NATIVE_SCALARS_{timeframe}",
    )
    return out.astype(np.float32)


def _attach_model_native_mtf_scalar_frame_v4(
    frame: pd.DataFrame,
    scalar_frame: pd.DataFrame,
    *,
    timeframe: str,
) -> None:
    expected_fields = MODEL_NATIVE_MTF_SCALAR_FIELDS_BY_TIMEFRAME_V4[timeframe]
    if not frame.index.equals(scalar_frame.index):
        raise RuntimeError(
            f"HTF_V4_MODEL_NATIVE_SCALAR_TIMESTAMP_MISMATCH: {timeframe}"
        )
    values = np.ascontiguousarray(
        scalar_frame.to_numpy(dtype=np.float32, copy=False)
    )
    if expected_fields:
        warmup_rows = validate_causal_feature_matrix(
            values,
            expected_width=len(expected_fields),
            context=f"HTF_V4_MODEL_NATIVE_SCALARS_{timeframe}",
        )
    else:
        if values.shape != (len(frame), 0):
            raise RuntimeError(
                "HTF_V4_MODEL_NATIVE_SCALAR_EMPTY_MATRIX_INVALID"
            )
        warmup_rows = 0
    frame.attrs["model_native_mtf_scalar_fields_v4"] = expected_fields
    frame.attrs["model_native_mtf_scalars_np_v4"] = values
    frame.attrs["model_native_mtf_scalar_warmup_rows_v4"] = warmup_rows
    frame.attrs["model_native_mtf_scalar_contract_v4"] = (
        MODEL_NATIVE_MTF_SCALAR_CONTRACT_V4
    )


def build_multi_tf_per_bar_features_v4(
    m5_df: pd.DataFrame,
) -> dict:
    """Build all eight causal specialist families from native M5 only."""
    base_bar_duration = pd.Timedelta(minutes=5)
    _validate_m5_input(
        m5_df,
        require_volume=True,
        bar_duration=base_bar_duration,
    )
    source = m5_df.copy(deep=False)
    source.index = source.index.as_unit("ns")
    expected_indices = build_multi_tf_v4_closed_timestamp_indices(
        source.index,
    )
    result = {}
    for tf_name, rule in MULTI_TF_RESAMPLE_RULES.items():
        resampled = _resample_ohlcv(source, rule)
        resampled = resampled.dropna(subset=["open", "high", "low", "close"])
        expected_index = expected_indices[tf_name]
        if not resampled.index.is_unique or not expected_index.isin(
            resampled.index
        ).all():
            raise RuntimeError(
                f"HTF_V4_RESAMPLED_TIMESTAMP_GEOMETRY_INVALID: {tf_name}"
            )
        resampled = resampled.loc[expected_index]
        if not resampled.index.equals(expected_index):
            raise RuntimeError(
                f"HTF_V4_RESAMPLED_TIMESTAMP_GEOMETRY_INVALID: {tf_name}"
            )
        computed = compute_per_bar_features_v4(resampled)
        # Retain the exact float32 matrix used to construct the DataFrame so
        # every in-memory V4 consumer sees the same verified bytes as attrs.
        # A fragmented pandas result may otherwise allocate a fresh matrix on
        # each ``to_numpy`` call and violate the one-cache P0 contract.
        feats_np = computed.to_numpy(dtype=np.float32, copy=False)
        feats = pd.DataFrame(
            feats_np,
            index=computed.index,
            columns=MULTI_TF_PER_BAR_FEATURES_V4,
            copy=False,
        )
        ts_int64 = feats.index.asi8.astype(np.int64, copy=True)
        # V4 is the active Entry/Exit owner surface. Keep one shared float32
        # matrix instead of duplicating it in attrs.
        warmup_rows = validate_causal_feature_matrix(
            feats_np,
            expected_width=MULTI_TF_FEATURE_COUNT_V4,
            context=f"HTF_V4_{tf_name}",
        )
        feats.attrs["ts_int64"] = ts_int64
        feats.attrs["feats_np"] = feats_np
        feats.attrs["causal_warmup_rows"] = warmup_rows
        feats.attrs["htf_feature_contract"] = HTF_V4_MATRIX_CONTRACT
        scalar_frame = _compute_model_native_mtf_scalar_frame_v4(
            resampled,
            timeframe=tf_name,
        )
        _attach_model_native_mtf_scalar_frame_v4(
            feats,
            scalar_frame,
            timeframe=tf_name,
        )
        result[tf_name] = feats
    return result



# Records, per verified cache frame object, that its full-matrix validation has
# passed. The frames are immutable for a run, so the O(frame) equality and
# finiteness checks below need to run once per frame, not once per sample. The
# token binds the frame's exact identity (the two cache-array data pointers,
# length and width); any replacement or in-place change misses the token and the
# full validation runs again. The checks themselves are unchanged.
_HTF_FRAMES_VALIDATED: dict = {}


def require_multi_tf_v4_frames(
    features: Mapping[str, pd.DataFrame],
) -> Mapping[str, pd.DataFrame]:
    """Validate the exact ordered V4/111 cache matrices and verified views."""

    expected_tfs = tuple(MULTI_TF_RESAMPLE_RULES)
    if not isinstance(features, Mapping) or tuple(features) != expected_tfs:
        raise RuntimeError(
            "HTF_V4_CACHE_SET_INVALID: exact ordered M5/M15/H1/H4/D1 required"
        )
    for timeframe in expected_tfs:
        frame = features[timeframe]
        if (
            not isinstance(frame, pd.DataFrame)
            or frame.empty
            or tuple(frame.columns) != MULTI_TF_PER_BAR_FEATURES_V4
            or frame.attrs.get("htf_feature_contract") != HTF_V4_MATRIX_CONTRACT
        ):
            raise RuntimeError(
                f"HTF_V4_CACHE_FRAME_CONTRACT_INVALID: {timeframe}"
            )
        timestamps = np.asarray(frame.attrs.get("ts_int64"))
        verified = np.asarray(frame.attrs.get("feats_np"))
        _frame_token = (
            verified.__array_interface__["data"][0]
            if verified.dtype == np.dtype(np.float32) else None,
            timestamps.__array_interface__["data"][0]
            if timestamps.dtype == np.dtype(np.int64) else None,
            len(frame),
            int(verified.shape[1]) if verified.ndim == 2 else -1,
        )
        if _HTF_FRAMES_VALIDATED.get(id(frame)) == _frame_token:
            continue
        frame_values = frame.to_numpy(dtype=np.float32, copy=False)
        if (
            timestamps.dtype != np.dtype(np.int64)
            or timestamps.shape != (len(frame),)
            or np.any(np.diff(timestamps) <= 0)
            or not np.array_equal(frame.index.asi8, timestamps)
            or verified.dtype != np.dtype(np.float32)
            or verified.shape != (len(frame), MULTI_TF_FEATURE_COUNT_V4)
            or not np.shares_memory(frame_values, verified)
            or not np.array_equal(frame_values, verified, equal_nan=True)
        ):
            raise RuntimeError(
                f"HTF_V4_CACHE_VERIFIED_MATRIX_INVALID: {timeframe}"
            )
        warmup_rows = validate_causal_feature_matrix(
            verified,
            expected_width=MULTI_TF_FEATURE_COUNT_V4,
            context=f"HTF_V4_CACHE_{timeframe}",
        )
        if (
            warmup_rows == len(frame)
            or frame.attrs.get("causal_warmup_rows") != warmup_rows
        ):
            raise RuntimeError(
                f"HTF_V4_CACHE_WARMUP_INVALID: {timeframe}"
            )
        _HTF_FRAMES_VALIDATED[id(frame)] = _frame_token
    return features


def require_model_native_mtf_scalar_owner_v4(
    features: Mapping[str, pd.DataFrame],
) -> Mapping[str, pd.DataFrame]:
    """Require the exact compact scalar surface on every verified V4 frame."""

    require_multi_tf_v4_frames(features)
    for timeframe, expected_fields in (
        MODEL_NATIVE_MTF_SCALAR_FIELDS_BY_TIMEFRAME_V4.items()
    ):
        frame = features[timeframe]
        fields = frame.attrs.get("model_native_mtf_scalar_fields_v4")
        values = np.asarray(
            frame.attrs.get("model_native_mtf_scalars_np_v4")
        )
        if (
            fields != expected_fields
            or values.dtype != np.dtype(np.float32)
            or values.shape != (len(frame), len(expected_fields))
            or frame.attrs.get("model_native_mtf_scalar_contract_v4")
            != MODEL_NATIVE_MTF_SCALAR_CONTRACT_V4
        ):
            raise RuntimeError(
                f"HTF_V4_MODEL_NATIVE_SCALAR_CONTRACT_INVALID: {timeframe}"
            )
        if expected_fields:
            warmup_rows = validate_causal_feature_matrix(
                values,
                expected_width=len(expected_fields),
                context=f"HTF_V4_MODEL_NATIVE_SCALARS_{timeframe}",
            )
        else:
            warmup_rows = 0
        if frame.attrs.get("model_native_mtf_scalar_warmup_rows_v4") != warmup_rows:
            raise RuntimeError(
                f"HTF_V4_MODEL_NATIVE_SCALAR_WARMUP_INVALID: {timeframe}"
            )
    return features


def bind_model_native_mtf_scalar_owner_v4(
    features: Mapping[str, pd.DataFrame],
    native_m5_ohlcv: pd.DataFrame,
) -> Mapping[str, pd.DataFrame]:
    """Bind deterministic scalar views to a verified cache from its exact M5 source."""

    require_multi_tf_v4_frames(features)
    _validate_m5_input(
        native_m5_ohlcv,
        require_volume=True,
        bar_duration=pd.Timedelta(minutes=5),
    )
    source = native_m5_ohlcv.copy(deep=False)
    source.index = source.index.as_unit("ns")
    expected_indices = build_multi_tf_v4_closed_timestamp_indices(source.index)
    for timeframe, rule in MULTI_TF_RESAMPLE_RULES.items():
        if not features[timeframe].index.equals(expected_indices[timeframe]):
            raise RuntimeError(
                "HTF_V4_MODEL_NATIVE_SCALAR_SOURCE_GEOMETRY_MISMATCH: "
                f"{timeframe}"
            )
        resampled = _resample_ohlcv(source, rule).dropna(
            subset=["open", "high", "low", "close"]
        )
        if not expected_indices[timeframe].isin(resampled.index).all():
            raise RuntimeError(
                "HTF_V4_MODEL_NATIVE_SCALAR_SOURCE_GEOMETRY_MISMATCH: "
                f"{timeframe}"
            )
        resampled = resampled.loc[expected_indices[timeframe]]
        scalar_frame = _compute_model_native_mtf_scalar_frame_v4(
            resampled,
            timeframe=timeframe,
        )
        _attach_model_native_mtf_scalar_frame_v4(
            features[timeframe],
            scalar_frame,
            timeframe=timeframe,
        )
    return require_model_native_mtf_scalar_owner_v4(features)


def project_model_native_mtf_scalars_v4(
    features: Mapping[str, pd.DataFrame],
    target_ts_ns,
    *,
    decision_bar_duration: pd.Timedelta,
) -> dict[str, np.ndarray]:
    """Project the one native-M5 scalar owner onto local M5 or local M1."""

    require_model_native_mtf_scalar_owner_v4(features)
    routes = {
        pd.Timedelta(minutes=5): ("M15", "H1", "H4", "D1"),
        pd.Timedelta(minutes=1): ("M5", "M15", "H1", "H4", "D1"),
    }
    if decision_bar_duration not in routes:
        raise RuntimeError(
            "HTF_V4_MODEL_NATIVE_PROJECTION_CLOCK_INVALID: exact M1 or M5 required"
        )
    target = np.asarray(target_ts_ns, dtype=np.int64)
    if (
        target.ndim != 1
        or len(target) < 5
        or np.any(np.diff(target) <= 0)
        or np.any(target % int(decision_bar_duration.value) != 0)
    ):
        raise RuntimeError("HTF_V4_MODEL_NATIVE_PROJECTION_TARGET_INVALID")

    projected: dict[str, np.ndarray] = {}
    for timeframe in routes[decision_bar_duration]:
        fields = MODEL_NATIVE_MTF_SCALAR_FIELDS_BY_TIMEFRAME_V4[timeframe]
        if not fields:
            continue
        frame = features[timeframe]
        timestamps = np.asarray(frame.attrs["ts_int64"], dtype=np.int64)
        values = np.asarray(frame.attrs["model_native_mtf_scalars_np_v4"])
        cutoff = (
            target
            + int(decision_bar_duration.value)
            - int(MULTI_TF_SHIFT[timeframe].value)
        )
        right = np.searchsorted(timestamps, cutoff, side="right") - 1
        valid = right >= 0
        safe = np.clip(right, 0, len(timestamps) - 1)
        for column, name in enumerate(fields):
            if name in projected:
                raise RuntimeError(
                    f"HTF_V4_MODEL_NATIVE_PROJECTION_DUPLICATE_FIELD: {name}"
                )
            aligned = np.full(len(target), np.nan, dtype=np.float64)
            aligned[valid] = values[safe[valid], column]
            projected[name] = aligned

    if set(projected) != set(MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4):
        raise RuntimeError(
            "HTF_V4_MODEL_NATIVE_PROJECTION_FIELDS_INVALID: "
            f"missing={sorted(set(MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4) - set(projected))} "
            f"unexpected={sorted(set(projected) - set(MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4))}"
        )
    ordered = {
        name: projected[name]
        for name in MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4
    }
    validate_causal_feature_matrix(
        np.column_stack(list(ordered.values())),
        expected_width=len(MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4),
        context="HTF_V4_MODEL_NATIVE_PROJECTION",
    )
    return ordered


def model_native_mtf_owner_marker_v4(
    *,
    decision_bar_duration: pd.Timedelta,
) -> dict[str, object]:
    routes = {
        pd.Timedelta(minutes=5): ("M15", "H1", "H4", "D1"),
        pd.Timedelta(minutes=1): ("M5", "M15", "H1", "H4", "D1"),
    }
    if decision_bar_duration not in routes:
        raise RuntimeError("HTF_V4_MODEL_NATIVE_OWNER_MARKER_CLOCK_INVALID")
    fields = list(MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4)
    return {
        "schema_version": MODEL_NATIVE_MTF_SCALAR_CONTRACT_V4,
        "source": "exact_native_m5_closed_ohlcv",
        "decision_bar_seconds": int(decision_bar_duration.total_seconds()),
        "route_timeframes": list(routes[decision_bar_duration]),
        "field_order": fields,
        "field_order_sha256": hashlib.sha256(
            json.dumps(
                fields,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    }


def attach_model_native_mtf_scalars_v4(
    frame: pd.DataFrame,
    *,
    multi_tf: Mapping[str, pd.DataFrame],
    decision_bar_duration: pd.Timedelta,
) -> pd.DataFrame:
    """Attach all persistent MTF scalars once; existing fields are an owner conflict."""

    _validate_m5_input(
        frame,
        require_volume=True,
        bar_duration=decision_bar_duration,
    )
    conflicts = sorted(
        set(MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4) & set(frame.columns)
    )
    if conflicts:
        raise RuntimeError(
            f"HTF_V4_MODEL_NATIVE_DUPLICATE_MTF_OWNER: {conflicts}"
        )
    projected = project_model_native_mtf_scalars_v4(
        multi_tf,
        frame.index.asi8.astype(np.int64, copy=False),
        decision_bar_duration=decision_bar_duration,
    )
    for name in MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4:
        frame[name] = projected[name]
    frame.attrs["model_native_mtf_owner_v4"] = model_native_mtf_owner_marker_v4(
        decision_bar_duration=decision_bar_duration
    )
    return frame


def require_model_native_mtf_owner_marker_v4(
    frame: pd.DataFrame,
    *,
    decision_bar_duration: pd.Timedelta,
) -> dict[str, object]:
    expected = model_native_mtf_owner_marker_v4(
        decision_bar_duration=decision_bar_duration
    )
    if frame.attrs.get("model_native_mtf_owner_v4") != expected:
        raise RuntimeError("HTF_V4_MODEL_NATIVE_OWNER_MARKER_MISSING")
    observed = tuple(
        name
        for name in MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4
        if name in frame.columns
    )
    if observed != MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4:
        raise RuntimeError("HTF_V4_MODEL_NATIVE_OWNER_FIELDS_MISSING")
    return expected


def project_multi_tf_v4_scalars(
    multi_tf: Mapping[str, pd.DataFrame],
    target_ts_ns,
    per_tf_map,
    tfs=("m15", "h1", "h4", "d1"),
    skip=frozenset(),
    *,
    decision_bar_duration: pd.Timedelta,
) -> dict[str, np.ndarray]:
    """Project persistent scalar fields from explicit verified V4 cache bytes."""

    require_multi_tf_v4_frames(multi_tf)
    if decision_bar_duration not in (
        pd.Timedelta(minutes=1),
        pd.Timedelta(minutes=5),
    ):
        raise RuntimeError(
            "HTF_V4_PROJECTION_DECISION_CLOCK_INVALID: exact M1 or M5 required"
        )
    target_ts_ns = np.asarray(target_ts_ns, dtype=np.int64)
    if (
        target_ts_ns.ndim != 1
        or len(target_ts_ns) == 0
        or np.any(np.diff(target_ts_ns) <= 0)
        or np.any(target_ts_ns % int(decision_bar_duration.value) != 0)
    ):
        raise RuntimeError(
            "HTF_V4_PROJECTION_TARGET_INVALID: exact chronological local grid required"
        )
    requested_tfs = tuple(str(name).lower() for name in tfs)
    if (
        any(name.upper() not in MULTI_TF_SHIFT for name in requested_tfs)
        or len(set(requested_tfs)) != len(requested_tfs)
    ):
        raise RuntimeError(
            f"HTF_V4_PROJECTION_TF_INVALID: tfs={requested_tfs}"
        )
    projection = tuple(
        (str(output_name), str(source_name))
        for output_name, source_name in per_tf_map
    )
    if not projection or len(set(projection)) != len(projection):
        raise RuntimeError(
            "HTF_V4_PROJECTION_MAP_INVALID: non-empty unique map required"
        )

    out: dict[str, np.ndarray] = {}
    for tf_lower in requested_tfs:
        tf_key = tf_lower.upper()
        frame = multi_tf[tf_key]
        timestamps = np.asarray(frame.attrs["ts_int64"], dtype=np.int64)
        verified = np.asarray(frame.attrs["feats_np"])
        positions = {
            str(name): index for index, name in enumerate(frame.columns)
        }
        decision_close_ns = target_ts_ns + int(decision_bar_duration.value)
        cutoffs = decision_close_ns - int(MULTI_TF_SHIFT[tf_key].value)
        right = np.searchsorted(timestamps, cutoffs, side="right") - 1
        valid = right >= 0
        safe = np.clip(right, 0, len(timestamps) - 1)
        for output_name, source_name in projection:
            if (tf_lower, output_name) in skip:
                continue
            if source_name not in positions:
                raise RuntimeError(
                    f"HTF_V4_PROJECTION_SOURCE_MISSING: {tf_key}.{source_name}"
                )
            projected = np.full(len(target_ts_ns), np.nan, dtype=np.float64)
            projected[valid] = verified[
                safe[valid],
                positions[source_name],
            ]
            out[f"{tf_lower}_{output_name}_v2"] = projected
    if not out:
        raise RuntimeError("HTF_V4_PROJECTION_EMPTY")
    validate_causal_feature_matrix(
        np.column_stack(list(out.values())),
        expected_width=len(out),
        context="HTF_V4_PROJECTION",
    )
    return out


# REGIME_V4 projection fields. Output names ending in _v2 are persistent model fields.
REGIME_V4_MTF_PROJECTION = (
    ("ema20_slope_atr", "ema20_slope_atr"),
    ("ema_stack_aligned", "ema_stack_aligned_v2"),
    ("regime_class_id", "regime_class_id"),
    ("trend_age_bars_norm", "trend_age_bars_norm"),
    ("mom_5_atr", "mom_5_atr"),
    ("mom_20_atr", "mom_20_atr"),
    ("rsi14_centered", "rsi14_centered"),
    ("atr_bps_14", "atr_bps_14"),
    ("lower_wick_pct", "lower_wick_pct"),
)
REGIME_V4_MTF_TIMEFRAMES = MULTI_TF_TIMEFRAMES_LOWER_M5_LAST
REGIME_V4_MTF_SKIP = frozenset({("d1", "lower_wick_pct")})


def attach_default_regime_v4_scalars(
    frame: pd.DataFrame,
    *,
    multi_tf: Mapping[str, pd.DataFrame],
    decision_bar_duration: pd.Timedelta,
) -> pd.DataFrame:
    """Overwrite persistent regime scalars from the explicit shared V4 cache."""

    _validate_m5_input(
        frame,
        require_volume=True,
        bar_duration=decision_bar_duration,
    )
    for name, values in project_multi_tf_v4_scalars(
        multi_tf,
        frame.index.asi8.astype(np.int64, copy=False),
        REGIME_V4_MTF_PROJECTION,
        REGIME_V4_MTF_TIMEFRAMES,
        REGIME_V4_MTF_SKIP,
        decision_bar_duration=decision_bar_duration,
    ).items():
        frame[name] = values
    return frame


_HTF_V4_CACHE_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "cache_identity_sha256",
        "feature_count",
        "feature_names",
        "shift_contract",
        "builder_version",
        "m5_prebuilt_source",
        "m5_prebuilt_source_sha256",
        "full_input_liveness",
        "tfs",
    }
)
_HTF_V4_CACHE_TF_KEYS = frozenset(
    {
        "n_bars",
        "feature_count",
        "feats_npy",
        "feats_npy_sha256",
        "feats_npy_size_bytes",
        "ts_npy",
        "ts_npy_sha256",
        "ts_npy_size_bytes",
        "first_ts_ns",
        "last_ts_ns",
        "causal_warmup_rows",
    }
)


class MultiTFV4DiskCache(dict):
    """Verified TF mapping with one content-bound disk-cache identity."""

    def __init__(
        self,
        *,
        cache_identity_sha256: str,
        manifest_sha256: str,
        m5_prebuilt_source: str,
        m5_prebuilt_source_sha256: str,
    ) -> None:
        super().__init__()
        self.cache_identity_sha256 = cache_identity_sha256
        self.manifest_sha256 = manifest_sha256
        self.m5_prebuilt_source = m5_prebuilt_source
        self.m5_prebuilt_source_sha256 = m5_prebuilt_source_sha256


def compute_htf_v4_cache_identity(manifest: dict) -> str:
    """Return the canonical identity for a manifest and all declared arrays."""

    identity_payload = dict(manifest)
    identity_payload.pop("cache_identity_sha256", None)
    try:
        encoded = json.dumps(
            identity_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuntimeError("HTF_V4_CACHE_MANIFEST_INVALID: non-canonical value") from exc
    return hashlib.sha256(encoded).hexdigest()


def _json_object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _cache_path_has_symlink_component(path: Path) -> bool:
    absolute = path if path.is_absolute() else Path.cwd() / path
    return any(component.is_symlink() for component in (absolute, *absolute.parents))


def _read_cache_file_bytes(
    directory_fd: int,
    name: str,
    *,
    expected_sha256: str | None,
    expected_size_bytes: int | None,
    label: str,
) -> bytes:
    """Read one regular cache file once and verify those exact bytes.

    ``dir_fd`` pins the already-opened cache directory. ``O_NOFOLLOW`` prevents
    a manifest-named symlink from being resolved between inventory validation
    and open. The returned bytes are also the bytes passed to ``numpy.load``.
    """

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise RuntimeError(f"HTF_V4_CACHE_FILE_INVALID: {label}") from exc
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise RuntimeError(f"HTF_V4_CACHE_FILE_INVALID: {label} is not regular")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        observed_size = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
            observed_size += len(chunk)
    finally:
        os.close(fd)
    if expected_size_bytes is not None and observed_size != expected_size_bytes:
        raise RuntimeError(
            f"HTF_V4_CACHE_SIZE_MISMATCH: {label} "
            f"observed={observed_size} expected={expected_size_bytes}"
        )
    observed_sha256 = digest.hexdigest()
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"HTF_V4_CACHE_SHA256_MISMATCH: {label} "
            f"observed={observed_sha256} expected={expected_sha256}"
        )
    return b"".join(chunks)


def _exact_cache_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(
            f"HTF_V4_CACHE_CONTRACT_MISMATCH: {label} must be an exact SHA-256"
        )
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise RuntimeError(
            f"HTF_V4_CACHE_CONTRACT_MISMATCH: {label} must be an exact SHA-256"
        )
    return value


def _exact_cache_int(
    value: object,
    *,
    label: str,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise RuntimeError(
            f"HTF_V4_CACHE_CONTRACT_MISMATCH: {label} must be an exact integer"
        )
    observed = int(value)
    if observed < minimum:
        raise RuntimeError(
            f"HTF_V4_CACHE_CONTRACT_MISMATCH: {label}={observed} < {minimum}"
        )
    return observed


def _load_verified_cache_npy(
    directory_fd: int,
    name: str,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    label: str,
) -> np.ndarray:
    payload = _read_cache_file_bytes(
        directory_fd,
        name,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
        label=label,
    )
    try:
        loaded = np.load(io.BytesIO(payload), allow_pickle=False)
    except Exception as exc:
        raise RuntimeError(f"HTF_V4_CACHE_NPY_INVALID: {label}") from exc
    if not isinstance(loaded, np.ndarray):
        raise RuntimeError(f"HTF_V4_CACHE_NPY_INVALID: {label} is not an ndarray")
    return loaded


def load_multi_tf_v4_cache(cache_dir) -> MultiTFV4DiskCache:
    """Load the sole immutable V4 cache after byte and contract verification."""
    supplied = Path(cache_dir).expanduser()
    absolute = supplied if supplied.is_absolute() else Path.cwd() / supplied
    if _cache_path_has_symlink_component(absolute):
        raise RuntimeError(
            f"HTF_V4_CACHE_PATH_INVALID: cache path traverses a symlink: {absolute}"
        )
    try:
        resolved_cache_dir = absolute.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"HTF_V4_CACHE_PATH_INVALID: {absolute}") from exc
    if not resolved_cache_dir.is_dir():
        raise RuntimeError(f"HTF_V4_CACHE_PATH_INVALID: {resolved_cache_dir}")

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        directory_fd = os.open(resolved_cache_dir, directory_flags)
    except OSError as exc:
        raise RuntimeError(
            f"HTF_V4_CACHE_PATH_INVALID: {resolved_cache_dir}"
        ) from exc
    try:
        initial_inventory = set(os.listdir(directory_fd))
        if "manifest.json" not in initial_inventory:
            raise RuntimeError(
                f"HTF_V4_CACHE_MANIFEST_MISSING: {resolved_cache_dir / 'manifest.json'}"
            )
        manifest_bytes = _read_cache_file_bytes(
            directory_fd,
            "manifest.json",
            expected_sha256=None,
            expected_size_bytes=None,
            label="manifest.json",
        )
        try:
            manifest = json.loads(
                manifest_bytes.decode("utf-8"),
                object_pairs_hook=_json_object_without_duplicate_keys,
            )
        except (UnicodeError, ValueError) as exc:
            raise RuntimeError(
                f"HTF_V4_CACHE_MANIFEST_INVALID: {resolved_cache_dir / 'manifest.json'}"
            ) from exc
        if not isinstance(manifest, dict):
            raise RuntimeError("HTF_V4_CACHE_MANIFEST_INVALID: root must be an object")
        schema_version = manifest.get("schema_version")
        if schema_version != HTF_V4_CACHE_SCHEMA_VERSION:
            raise RuntimeError(
                "HTF_V4_CACHE_CONTRACT_REQUIRED: reject legacy manifest "
                f"before array load; observed={schema_version!r} "
                f"expected={HTF_V4_CACHE_SCHEMA_VERSION!r}"
            )
        expected_manifest_keys = _HTF_V4_CACHE_MANIFEST_KEYS
        if set(manifest) != expected_manifest_keys:
            raise RuntimeError(
                "HTF_V4_CACHE_CONTRACT_MISMATCH: manifest exact keys differ "
                f"missing={sorted(expected_manifest_keys - set(manifest))} "
                f"unexpected={sorted(set(manifest) - expected_manifest_keys)}"
            )
        expected_shift = {
            tf: str(shift) for tf, shift in MULTI_TF_SHIFT.items()
        }
        matrix_contract = HTF_V4_MATRIX_CONTRACT
        feature_width = MULTI_TF_FEATURE_COUNT_V4
        feature_names = MULTI_TF_PER_BAR_FEATURES_V4
        builder_version = HTF_V4_CACHE_BUILDER_VERSION
        contracts = {
            "schema_version": schema_version,
            "builder_version": builder_version,
            "feature_count": feature_width,
            "feature_names": list(feature_names),
            "shift_contract": expected_shift,
        }
        for name, expected in contracts.items():
            if manifest.get(name) != expected:
                raise RuntimeError(
                    f"HTF_V4_CACHE_CONTRACT_MISMATCH: {name} observed={manifest.get(name)!r} "
                    f"expected={expected!r}"
                )
        source_path = Path(str(manifest.get("m5_prebuilt_source") or "")).expanduser()
        if not source_path.is_absolute():
            raise RuntimeError(
                "HTF_V4_CACHE_CONTRACT_MISMATCH: m5_prebuilt_source must be absolute"
            )
        m5_prebuilt_source_sha256 = _exact_cache_sha256(
            manifest["m5_prebuilt_source_sha256"],
            label="m5_prebuilt_source_sha256",
        )
        cache_identity_sha256 = _exact_cache_sha256(
            manifest["cache_identity_sha256"],
            label="cache_identity_sha256",
        )
        computed_cache_identity = compute_htf_v4_cache_identity(manifest)
        if cache_identity_sha256 != computed_cache_identity:
            raise RuntimeError(
                "HTF_V4_CACHE_IDENTITY_MISMATCH: "
                f"observed={cache_identity_sha256} expected={computed_cache_identity}"
            )
        tf_manifest = manifest.get("tfs")
        if not isinstance(tf_manifest, dict) or tuple(tf_manifest) != tuple(
            MULTI_TF_RESAMPLE_RULES
        ):
            raise RuntimeError(
                "HTF_V4_CACHE_CONTRACT_MISMATCH: ordered exact "
                "M5/M15/H1/H4/D1 entries required"
            )
        declared_inventory = {"manifest.json"}
        for tf_name in MULTI_TF_RESAMPLE_RULES:
            info = tf_manifest[tf_name]
            if not isinstance(info, dict) or set(info) != _HTF_V4_CACHE_TF_KEYS:
                observed_keys = set(info) if isinstance(info, dict) else set()
                raise RuntimeError(
                    f"HTF_V4_CACHE_CONTRACT_MISMATCH: {tf_name} exact keys differ "
                    f"missing={sorted(_HTF_V4_CACHE_TF_KEYS - observed_keys)} "
                    f"unexpected={sorted(observed_keys - _HTF_V4_CACHE_TF_KEYS)}"
                )
            feats_name = str(info["feats_npy"])
            ts_name = str(info["ts_npy"])
            expected_names = (f"{tf_name}_feats.npy", f"{tf_name}_ts.npy")
            if (feats_name, ts_name) != expected_names:
                raise RuntimeError(
                    f"HTF_V4_CACHE_CONTRACT_MISMATCH: {tf_name} filenames "
                    f"observed={(feats_name, ts_name)!r} expected={expected_names!r}"
                )
            declared_inventory.update((feats_name, ts_name))
        if initial_inventory != declared_inventory:
            raise RuntimeError(
                "HTF_V4_CACHE_INVENTORY_MISMATCH: "
                f"missing={sorted(declared_inventory - initial_inventory)} "
                f"unexpected={sorted(initial_inventory - declared_inventory)}"
            )

        out = MultiTFV4DiskCache(
            cache_identity_sha256=cache_identity_sha256,
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            m5_prebuilt_source=str(source_path),
            m5_prebuilt_source_sha256=m5_prebuilt_source_sha256,
        )
        for tf_name in MULTI_TF_RESAMPLE_RULES:
            info = tf_manifest[tf_name]
            n_bars = _exact_cache_int(
                info["n_bars"], label=f"{tf_name}.n_bars", minimum=1
            )
            feature_count = _exact_cache_int(
                info["feature_count"],
                label=f"{tf_name}.feature_count",
                minimum=1,
            )
            if feature_count != feature_width:
                raise RuntimeError(
                    f"HTF_V4_CACHE_CONTRACT_MISMATCH: {tf_name}.feature_count "
                    f"observed={feature_count} expected={feature_width}"
                )
            feats_size = _exact_cache_int(
                info["feats_npy_size_bytes"],
                label=f"{tf_name}.feats_npy_size_bytes",
                minimum=1,
            )
            ts_size = _exact_cache_int(
                info["ts_npy_size_bytes"],
                label=f"{tf_name}.ts_npy_size_bytes",
                minimum=1,
            )
            feats_np = _load_verified_cache_npy(
                directory_fd,
                str(info["feats_npy"]),
                expected_sha256=_exact_cache_sha256(
                    info["feats_npy_sha256"],
                    label=f"{tf_name}.feats_npy_sha256",
                ),
                expected_size_bytes=feats_size,
                label=f"{tf_name}.feats_npy",
            )
            ts_int64 = _load_verified_cache_npy(
                directory_fd,
                str(info["ts_npy"]),
                expected_sha256=_exact_cache_sha256(
                    info["ts_npy_sha256"],
                    label=f"{tf_name}.ts_npy_sha256",
                ),
                expected_size_bytes=ts_size,
                label=f"{tf_name}.ts_npy",
            )
            if (
                feats_np.dtype != np.dtype(np.float32)
                or ts_int64.dtype != np.dtype(np.int64)
            ):
                raise RuntimeError(
                    f"HTF_V4_CACHE_CONTRACT_MISMATCH: {tf_name} requires "
                    "float32 features/int64 timestamps"
                )
            if feats_np.shape != (n_bars, feature_width):
                raise RuntimeError(
                    f"HTF_V4_CACHE_CONTRACT_MISMATCH: {tf_name} feature shape "
                    f"observed={feats_np.shape} "
                    f"expected={(n_bars, feature_width)}"
                )
            if ts_int64.shape != (n_bars,) or np.any(np.diff(ts_int64) <= 0):
                raise RuntimeError(
                    f"HTF_V4_CACHE_CONTRACT_MISMATCH: {tf_name} timestamps invalid"
                )
            warmup_rows = validate_causal_feature_matrix(
                feats_np,
                expected_width=feature_width,
                context=f"HTF_V4_CACHE_{tf_name}",
            )
            if warmup_rows == len(feats_np):
                raise RuntimeError(
                    f"HTF_V4_CACHE_WARMUP_INCOMPLETE: {tf_name} has no complete row"
                )
            expected_meta = {
                "n_bars": n_bars,
                "feature_count": feature_width,
                "first_ts_ns": int(ts_int64[0]),
                "last_ts_ns": int(ts_int64[-1]),
                "causal_warmup_rows": warmup_rows,
            }
            for name, expected in expected_meta.items():
                observed = _exact_cache_int(
                    info[name],
                    label=f"{tf_name}.{name}",
                    minimum=0,
                )
                if observed != expected:
                    raise RuntimeError(
                        f"HTF_V4_CACHE_CONTRACT_MISMATCH: {tf_name}.{name} "
                        f"observed={observed!r} expected={expected!r}"
                    )
            # Keep one verified feature matrix. DataFrame columns and the
            # fast-path attrs must be two views of those same bytes; a separate
            # placeholder matrix would let consumers read unverified values and
            # would double the cache's resident memory.
            idx = pd.DatetimeIndex(ts_int64.astype("datetime64[ns]"), tz="UTC")
            verified_feats = np.ascontiguousarray(feats_np)
            df = pd.DataFrame(
                verified_feats,
                index=idx,
                columns=feature_names,
                copy=False,
            )
            frame_values = df.to_numpy(dtype=np.float32, copy=False)
            if (
                not np.shares_memory(frame_values, verified_feats)
                or not np.array_equal(frame_values, verified_feats, equal_nan=True)
            ):
                raise RuntimeError(
                    f"HTF_V4_CACHE_MATRIX_VIEW_INVALID: {tf_name}"
                )
            df.attrs["ts_int64"] = np.ascontiguousarray(ts_int64)
            df.attrs["feats_np"] = frame_values
            df.attrs["causal_warmup_rows"] = warmup_rows
            df.attrs["htf_feature_contract"] = matrix_contract
            out[tf_name] = df
        try:
            require_multi_tf_v4_liveness_contract(
                manifest.get("full_input_liveness")
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "HTF_V4_CACHE_FULL_INPUT_LIVENESS_INVALID"
            ) from exc
        observed_liveness = build_multi_tf_v4_liveness_contract(out)
        if (
            observed_liveness.get("decision") != "PASS"
            or manifest.get("full_input_liveness") != observed_liveness
        ):
            raise RuntimeError(
                "HTF_V4_CACHE_FULL_INPUT_LIVENESS_INVALID"
            )
        final_inventory = set(os.listdir(directory_fd))
        if final_inventory != declared_inventory:
            raise RuntimeError(
                "HTF_V4_CACHE_INVENTORY_CHANGED_DURING_LOAD: "
                f"missing={sorted(declared_inventory - final_inventory)} "
                f"unexpected={sorted(final_inventory - declared_inventory)}"
            )
        return out
    finally:
        os.close(directory_fd)



# Per-frame validation memo for slice_multi_tf_v4_window. Keyed by the frame's
# id and bound to its exact cache-array identities, so a reused immutable frame
# is validated once instead of on every window slice. Bounded to the handful of
# multi-TF frames a run holds.
_HTF_WINDOW_VALIDATED: dict = {}


def slice_multi_tf_v4_window(
    feats: pd.DataFrame, target_ts: pd.Timestamp, n: int, tf_shift: pd.Timedelta,
) -> np.ndarray:
    """Slice the last `n` per-bar feature rows whose close-time is <= (target_ts - tf_shift).

    Returns an exact finite ``(n, n_features)`` float32 array. Missing history,
    indicator warmup, malformed cache metadata, and non-finite evidence are hard
    errors; this owner never pads or substitutes a neutral value.

    `tf_shift` enforces the "only closed bars" invariant: e.g. for H1, target=12:35
    means we use H1 bars closing at-or-before 11:35 (the 11:00 H1 bar, since
    12:00 H1 bar hasn't closed yet at 12:35).

    Verified V4 fast path: when `feats.attrs["ts_int64"]` and `feats.attrs["feats_np"]`
    are present (set by build_multi_tf_per_bar_features), we use numpy
    searchsorted on int64 timestamps — ~100× faster than pandas .loc.
    """
    if not isinstance(feats, pd.DataFrame) or feats.empty:
        raise RuntimeError("HTF_WINDOW_SOURCE_MISSING: exact non-empty feature table required")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) <= 0:
        raise RuntimeError(f"HTF_WINDOW_LENGTH_INVALID: n={n!r}")
    n = int(n)
    if not isinstance(tf_shift, pd.Timedelta) or tf_shift <= pd.Timedelta(0):
        raise RuntimeError(f"HTF_WINDOW_SHIFT_INVALID: tf_shift={tf_shift!r}")
    target = pd.Timestamp(target_ts)
    if target.tzinfo is None or target.utcoffset() != pd.Timedelta(0):
        raise RuntimeError("HTF_WINDOW_TARGET_INVALID: target_ts must be timezone-aware UTC")
    declared_contract = feats.attrs.get("htf_feature_contract")
    if (
        declared_contract != HTF_V4_MATRIX_CONTRACT
        or tuple(feats.columns) != MULTI_TF_PER_BAR_FEATURES_V4
    ):
        raise RuntimeError(
            "HTF_V4_WINDOW_SOURCE_CONTRACT_INVALID: exact V4/111 required"
        )

    ts_int64 = np.asarray(feats.attrs.get("ts_int64"))
    feats_np = np.asarray(feats.attrs.get("feats_np"))
    width = int(feats.shape[1])
    # The cache-array validation compares the entire per-timeframe frame
    # (e.g. 476k x 111 for M5) with np.array_equal on every window slice. The
    # frame is immutable during a run, so the full check is run once per frame
    # object and memoised: a token bound to this frame's exact identity
    # (id, shape, and the two cache-array identities) records that it passed.
    # The check itself is unchanged; only its per-window repetition is removed.
    _seen = _HTF_WINDOW_VALIDATED.get(id(feats))
    _token = (feats_np.__array_interface__["data"][0], ts_int64.__array_interface__["data"][0], len(feats), width)
    if _seen != _token:
        if (
            ts_int64.dtype != np.dtype(np.int64)
            or ts_int64.shape != (len(feats),)
            or feats_np.dtype != np.dtype(np.float32)
            or feats_np.shape != (len(feats), width)
            or not np.shares_memory(
                feats.to_numpy(dtype=np.float32, copy=False),
                feats_np,
            )
            or not np.array_equal(
                feats.to_numpy(dtype=np.float32, copy=False),
                feats_np,
                equal_nan=True,
            )
        ):
            raise RuntimeError("HTF_WINDOW_SOURCE_INVALID: malformed exact cache arrays")
        _HTF_WINDOW_VALIDATED[id(feats)] = _token
    warmup_rows = feats.attrs.get("causal_warmup_rows")
    if (
        isinstance(warmup_rows, bool)
        or not isinstance(warmup_rows, (int, np.integer))
        or not 0 <= int(warmup_rows) <= len(feats)
    ):
        raise RuntimeError("HTF_WINDOW_SOURCE_INVALID: causal warmup metadata missing")

    cutoff_ns = int(target.value) - int(tf_shift.value)
    right = int(np.searchsorted(ts_int64, cutoff_ns, side="right"))
    if right < n:
        raise RuntimeError(
            f"HTF_WINDOW_HISTORY_INSUFFICIENT: need={n} closed_rows={right} target={target.isoformat()}"
        )
    left = right - n
    if left < int(warmup_rows):
        raise RuntimeError(
            f"HTF_WINDOW_WARMUP_INCOMPLETE: first_row={left} warmup_rows={int(warmup_rows)}"
        )
    tail = np.asarray(feats_np[left:right], dtype=np.float32)
    if tail.shape != (n, width) or not np.isfinite(tail).all():
        raise RuntimeError("HTF_WINDOW_SOURCE_INVALID: selected feature evidence is non-finite")
    return np.ascontiguousarray(tail)


def get_model_native_multi_tf_route_windows(
    features: dict[str, pd.DataFrame],
    *,
    decision_bar_start: pd.Timestamp,
    per_tf_seq_lens: dict[str, int],
    route_timeframes: tuple[str, ...],
    base_bar_duration: pd.Timedelta,
) -> dict[str, np.ndarray]:
    """Slice one canonical Entry or Exit MTF route from the shared V4 cache.

    Entry and Exit deliberately use this same owner.  Their only differences
    are the local decision clock and the exact route declared by the shared
    feature-base contract.  The cache remains M5/M15/H1/H4/D1; no route copies,
    padding, neutral values or computed-feature resampling are permitted.
    """

    from gx1.contracts.entry_exit_feature_base_v1 import (
        ENTRY_DECISION_BAR_SECONDS,
        ENTRY_MTF_CONTEXT_TIMEFRAMES,
        EXIT_DECISION_BAR_SECONDS,
        EXIT_MTF_CONTEXT_TIMEFRAMES,
    )

    expected_cache_tfs = tuple(MULTI_TF_RESAMPLE_RULES)
    require_multi_tf_v4_frames(features)
    route = tuple(route_timeframes)
    canonical_routes = {
        tuple(ENTRY_MTF_CONTEXT_TIMEFRAMES): pd.Timedelta(
            seconds=ENTRY_DECISION_BAR_SECONDS
        ),
        tuple(EXIT_MTF_CONTEXT_TIMEFRAMES): pd.Timedelta(
            seconds=EXIT_DECISION_BAR_SECONDS
        ),
    }
    if route not in canonical_routes:
        raise RuntimeError(
            f"MODEL_NATIVE_MTF_ROUTE_INVALID: observed={route!r}"
        )
    if base_bar_duration != canonical_routes[route]:
        raise RuntimeError(
            "MODEL_NATIVE_MTF_LOCAL_CLOCK_INVALID: "
            f"route={route!r} observed={base_bar_duration} "
            f"expected={canonical_routes[route]}"
        )
    if (
        not isinstance(per_tf_seq_lens, dict)
        or tuple(per_tf_seq_lens) != expected_cache_tfs
        or any(
            isinstance(per_tf_seq_lens[tf], bool)
            or not isinstance(per_tf_seq_lens[tf], (int, np.integer))
            or int(per_tf_seq_lens[tf]) <= 0
            for tf in expected_cache_tfs
        )
    ):
        raise RuntimeError(
            "MODEL_NATIVE_MTF_SEQUENCE_LENGTHS_INVALID: exact ordered positive "
            "M5/M15/H1/H4/D1 mapping required"
        )
    target = pd.Timestamp(decision_bar_start)
    if target.tz is None or target.utcoffset() != pd.Timedelta(0):
        raise RuntimeError(
            "MODEL_NATIVE_MTF_DECISION_TIMESTAMP_INVALID: timezone-aware UTC required"
        )
    availability = target + base_bar_duration
    return {
        tf: slice_multi_tf_v4_window(
            features[tf],
            availability,
            n=int(per_tf_seq_lens[tf]),
            tf_shift=MULTI_TF_SHIFT[tf],
        )
        for tf in route
    }


def require_multi_tf_decision_window_coverage(
    features: dict[str, pd.DataFrame],
    *,
    per_tf_seq_lens: dict[str, int],
    decision_times_by_route_split: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Prove Entry +5m and Exit +1m TRAIN/VAL routes on one V4 cache."""

    from gx1.contracts.entry_exit_feature_base_v1 import (
        ENTRY_DECISION_BAR_SECONDS,
        ENTRY_MTF_CONTEXT_TIMEFRAMES,
        EXIT_DECISION_BAR_SECONDS,
        EXIT_MTF_CONTEXT_TIMEFRAMES,
    )

    pyramid = require_multi_tf_resolution_pyramid(per_tf_seq_lens)
    expected_tfs = tuple(MULTI_TF_RESAMPLE_RULES)
    try:
        require_multi_tf_v4_frames(features)
    except RuntimeError as exc:
        raise RuntimeError(
            "MULTI_TF_DECISION_COVERAGE_FEATURE_SET_INVALID: exact ordered "
            "V4/111 M5/M15/H1/H4/D1 cache required"
        ) from exc
    if (
        not isinstance(decision_times_by_route_split, dict)
        or tuple(decision_times_by_route_split) != ("entry", "exit")
        or any(
            not isinstance(route_splits, dict)
            or tuple(route_splits) != ("train", "val")
            for route_splits in decision_times_by_route_split.values()
        )
    ):
        raise RuntimeError(
            "MULTI_TF_DECISION_COVERAGE_ROUTE_SPLIT_SET_INVALID: exact ordered "
            "entry/exit and train/val decision times required"
        )

    route_specs = {
        "entry": {
            "timeframes": tuple(ENTRY_MTF_CONTEXT_TIMEFRAMES),
            "base_bar_duration": pd.Timedelta(
                seconds=ENTRY_DECISION_BAR_SECONDS
            ),
        },
        "exit": {
            "timeframes": tuple(EXIT_MTF_CONTEXT_TIMEFRAMES),
            "base_bar_duration": pd.Timedelta(
                seconds=EXIT_DECISION_BAR_SECONDS
            ),
        },
    }
    route_rows: dict[str, dict[str, object]] = {}
    route_windows: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    for route, spec in route_specs.items():
        split_bounds: dict[str, dict[str, object]] = {}
        for split, raw_times in decision_times_by_route_split[route].items():
            try:
                times = pd.DatetimeIndex(
                    pd.to_datetime(raw_times, utc=True, errors="raise")
                )
            except Exception as exc:
                raise RuntimeError(
                    f"MULTI_TF_DECISION_COVERAGE_TIME_INVALID: {route}.{split}"
                ) from exc
            if (
                times.empty
                or times.hasnans
                or not times.is_monotonic_increasing
                or not times.is_unique
            ):
                raise RuntimeError(
                    "MULTI_TF_DECISION_COVERAGE_TIME_INVALID: "
                    f"{route}.{split} must be non-empty, unique and chronological"
                )
            first = pd.Timestamp(times[0])
            last = pd.Timestamp(times[-1])
            split_bounds[split] = {
                "rows": int(len(times)),
                "first_utc": first.isoformat(),
                "last_utc": last.isoformat(),
            }
            for edge, target in (("first", first), ("last", last)):
                try:
                    route_windows[(route, split, edge)] = (
                        get_model_native_multi_tf_route_windows(
                            features,
                            decision_bar_start=target,
                            per_tf_seq_lens=per_tf_seq_lens,
                            route_timeframes=spec["timeframes"],
                            base_bar_duration=spec["base_bar_duration"],
                        )
                    )
                except RuntimeError as exc:
                    raise RuntimeError(
                        "MULTI_TF_DECISION_COVERAGE_UNAVAILABLE: "
                        f"{route}.{split}.{edge} target={target.isoformat()}: {exc}"
                    ) from exc
        route_rows[route] = {
            "timeframes": list(spec["timeframes"]),
            "target_availability_shift_seconds": int(
                spec["base_bar_duration"].total_seconds()
            ),
            "split_bounds": split_bounds,
        }

    per_tf: dict[str, object] = {}
    for tf in expected_tfs:
        frame = features[tf]
        n = int(per_tf_seq_lens[tf])
        route_metadata: dict[str, object] = {}
        for route, spec in route_specs.items():
            enabled = tf in spec["timeframes"]
            boundary_rows: dict[str, object] = {}
            if enabled:
                for split in ("train", "val"):
                    bounds = route_rows[route]["split_bounds"][split]
                    for edge in ("first", "last"):
                        window = route_windows[(route, split, edge)][tf]
                        boundary_rows[f"{split}_{edge}"] = {
                            "target_utc": bounds[f"{edge}_utc"],
                            "window_sha256": hashlib.sha256(
                                np.ascontiguousarray(
                                    window,
                                    dtype="<f4",
                                ).tobytes()
                            ).hexdigest(),
                        }
            route_metadata[route] = {
                "enabled": enabled,
                "boundaries": boundary_rows,
            }
        per_tf[tf] = {
            "seq_len": n,
            "coverage_seconds": pyramid["coverage_seconds"][tf],
            "causal_warmup_rows": int(frame.attrs["causal_warmup_rows"]),
            "routes": route_metadata,
        }

    payload: dict[str, object] = {
        "schema_version": "entry_exit_multi_tf_decision_window_coverage_v2",
        "cache_contract": HTF_V4_MATRIX_CONTRACT,
        "routes": route_rows,
        "resolution_pyramid": pyramid,
        "per_tf": per_tf,
        "all_route_split_boundaries_sliceable": True,
    }
    payload["contract_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return require_multi_tf_decision_window_coverage_metadata(
        payload,
        per_tf_seq_lens=per_tf_seq_lens,
    )


def require_multi_tf_decision_window_coverage_metadata(
    value: Mapping[str, object],
    *,
    per_tf_seq_lens: dict[str, int],
) -> dict[str, object]:
    """Strictly validate the immutable split-boundary coverage proof."""

    expected_keys = {
        "schema_version",
        "cache_contract",
        "routes",
        "resolution_pyramid",
        "per_tf",
        "all_route_split_boundaries_sliceable",
        "contract_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise RuntimeError("MULTI_TF_DECISION_COVERAGE_METADATA_KEYS_INVALID")
    payload = dict(value)
    pyramid = require_multi_tf_resolution_pyramid(per_tf_seq_lens)
    if (
        payload["schema_version"]
        != "entry_exit_multi_tf_decision_window_coverage_v2"
        or payload["cache_contract"] != HTF_V4_MATRIX_CONTRACT
        or payload["resolution_pyramid"] != pyramid
        or payload["all_route_split_boundaries_sliceable"] is not True
    ):
        raise RuntimeError("MULTI_TF_DECISION_COVERAGE_METADATA_INVALID")
    from gx1.contracts.entry_exit_feature_base_v1 import (
        ENTRY_DECISION_BAR_SECONDS,
        ENTRY_MTF_CONTEXT_TIMEFRAMES,
        EXIT_DECISION_BAR_SECONDS,
        EXIT_MTF_CONTEXT_TIMEFRAMES,
    )
    expected_routes = {
        "entry": (
            tuple(ENTRY_MTF_CONTEXT_TIMEFRAMES),
            ENTRY_DECISION_BAR_SECONDS,
        ),
        "exit": (
            tuple(EXIT_MTF_CONTEXT_TIMEFRAMES),
            EXIT_DECISION_BAR_SECONDS,
        ),
    }
    routes = payload["routes"]
    if not isinstance(routes, dict) or tuple(routes) != tuple(expected_routes):
        raise RuntimeError("MULTI_TF_DECISION_COVERAGE_ROUTE_METADATA_INVALID")
    parsed_route_bounds: dict[
        str,
        dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    ] = {}
    for route, (timeframes, availability_seconds) in expected_routes.items():
        raw_route = routes[route]
        if not isinstance(raw_route, dict) or set(raw_route) != {
            "timeframes",
            "target_availability_shift_seconds",
            "split_bounds",
        }:
            raise RuntimeError(
                "MULTI_TF_DECISION_COVERAGE_ROUTE_METADATA_INVALID"
            )
        if (
            raw_route["timeframes"] != list(timeframes)
            or raw_route["target_availability_shift_seconds"]
            != availability_seconds
            or not isinstance(raw_route["split_bounds"], dict)
            or tuple(raw_route["split_bounds"]) != ("train", "val")
        ):
            raise RuntimeError(
                "MULTI_TF_DECISION_COVERAGE_ROUTE_METADATA_INVALID"
            )
        parsed_route_bounds[route] = {}
        for split, raw in raw_route["split_bounds"].items():
            if not isinstance(raw, dict) or set(raw) != {
                "rows",
                "first_utc",
                "last_utc",
            }:
                raise RuntimeError(
                    "MULTI_TF_DECISION_COVERAGE_SPLIT_METADATA_INVALID"
                )
            rows = raw["rows"]
            first = pd.Timestamp(raw["first_utc"])
            last = pd.Timestamp(raw["last_utc"])
            if (
                isinstance(rows, bool)
                or not isinstance(rows, int)
                or rows <= 0
                or first.tzinfo is None
                or last.tzinfo is None
                or first.utcoffset() != pd.Timedelta(0)
                or last.utcoffset() != pd.Timedelta(0)
                or first > last
            ):
                raise RuntimeError(
                    "MULTI_TF_DECISION_COVERAGE_SPLIT_METADATA_INVALID"
                )
            parsed_route_bounds[route][split] = (first, last)
    per_tf = payload["per_tf"]
    if not isinstance(per_tf, dict) or tuple(per_tf) != tuple(
        MULTI_TF_RESAMPLE_RULES
    ):
        raise RuntimeError("MULTI_TF_DECISION_COVERAGE_TF_METADATA_INVALID")
    expected_boundaries = tuple(
        f"{split}_{edge}"
        for split in ("train", "val")
        for edge in ("first", "last")
    )
    for tf, raw in per_tf.items():
        if not isinstance(raw, dict) or set(raw) != {
            "seq_len",
            "coverage_seconds",
            "causal_warmup_rows",
            "routes",
        }:
            raise RuntimeError(
                "MULTI_TF_DECISION_COVERAGE_TF_METADATA_INVALID"
            )
        warmup = raw["causal_warmup_rows"]
        if (
            raw["seq_len"] != per_tf_seq_lens[tf]
            or raw["coverage_seconds"] != pyramid["coverage_seconds"][tf]
            or isinstance(warmup, bool)
            or not isinstance(warmup, int)
            or warmup < 0
        ):
            raise RuntimeError(
                "MULTI_TF_DECISION_COVERAGE_TF_METADATA_INVALID"
            )
        tf_routes = raw["routes"]
        if not isinstance(tf_routes, dict) or tuple(tf_routes) != tuple(
            expected_routes
        ):
            raise RuntimeError(
                "MULTI_TF_DECISION_COVERAGE_TF_ROUTE_METADATA_INVALID"
            )
        for route, (route_tfs, _availability) in expected_routes.items():
            route_row = tf_routes[route]
            enabled = tf in route_tfs
            if (
                not isinstance(route_row, dict)
                or set(route_row) != {"enabled", "boundaries"}
                or route_row["enabled"] is not enabled
                or not isinstance(route_row["boundaries"], dict)
                or tuple(route_row["boundaries"])
                != (expected_boundaries if enabled else ())
            ):
                raise RuntimeError(
                    "MULTI_TF_DECISION_COVERAGE_TF_ROUTE_METADATA_INVALID"
                )
            for boundary, row in route_row["boundaries"].items():
                if not isinstance(row, dict) or set(row) != {
                    "target_utc",
                    "window_sha256",
                }:
                    raise RuntimeError(
                        "MULTI_TF_DECISION_COVERAGE_BOUNDARY_METADATA_INVALID"
                    )
                split, edge = boundary.rsplit("_", 1)
                expected_target = parsed_route_bounds[route][split][
                    0 if edge == "first" else 1
                ]
                if (
                    pd.Timestamp(row["target_utc"]) != expected_target
                    or not isinstance(row["window_sha256"], str)
                    or len(row["window_sha256"]) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in row["window_sha256"]
                    )
                ):
                    raise RuntimeError(
                        "MULTI_TF_DECISION_COVERAGE_BOUNDARY_METADATA_INVALID"
                    )
    observed_hash = payload.pop("contract_sha256")
    expected_hash = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if observed_hash != expected_hash:
        raise RuntimeError("MULTI_TF_DECISION_COVERAGE_HASH_INVALID")
    payload["contract_sha256"] = observed_hash
    return payload
