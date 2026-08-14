import inspect

import numpy as np
import pandas as pd
import pytest

from gx1.features import basic_v1
from gx1.features.basic_v1 import BASIC_V1_FEATURES
from gx1.scripts import materialize_build_canonical_features_v2 as canonical_v2
from gx1.scripts.materialize_canonical_v3_augment import add_cyclic_time_features


def _m5_frame(start: str, periods: int) -> pd.DataFrame:
    time = pd.date_range(start, periods=periods, freq="5min", tz="UTC")
    close = 100.0 + np.arange(periods, dtype=np.float64)
    return pd.DataFrame(
        {
            "time": time,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
        }
    )


def test_canonical_v2_and_basic_owner_are_structurally_local_only() -> None:
    canonical_source = inspect.getsource(canonical_v2)
    basic_source = inspect.getsource(basic_v1)

    assert canonical_v2.CANONICAL_V2_MTF_OWNER == "native_m5_v4_only"
    for forbidden in (
        "compute_d1_features",
        "compute_m15_features",
        "merge_asof_features",
        ".resample(",
    ):
        assert forbidden not in canonical_source
    for forbidden in (
        "build_htf_from_m5",
        "_align_htf_to_m5_numpy",
        'df["_v1h1_ema_diff"]',
        'df["_v1h4_ema_diff"]',
    ):
        assert forbidden not in basic_source


@pytest.mark.parametrize(
    ("start", "frequency", "decision_delay_seconds"),
    (
        ("2026-07-01T06:50:00Z", "5min", 300),
        ("2026-07-01T06:58:00Z", "1min", 60),
    ),
)
def test_basic_session_flags_switch_at_the_exact_m1_m5_decision_boundary(
    start: str,
    frequency: str,
    decision_delay_seconds: int,
) -> None:
    index = pd.date_range(start, periods=3, freq=frequency, tz="UTC")
    observed = basic_v1.add_session_features(
        pd.DataFrame(index=index),
        decision_delay_seconds=decision_delay_seconds,
    )

    assert observed["is_EU"].tolist() == [0, 1, 1]
    assert observed["is_EU"].tolist() == [0, 1, 1]


def test_cyclic_clock_uses_m5_decision_availability_across_hour_and_day() -> None:
    labels = pd.DatetimeIndex(
        ["2026-07-23T12:55:00Z", "2026-07-23T23:55:00Z"]
    )
    observed = add_cyclic_time_features(pd.DataFrame(index=labels))
    expected_hour = np.asarray([13.0, 0.0])
    expected_dow = np.asarray([3.0, 4.0])

    np.testing.assert_allclose(
        observed["hour_sin"],
        np.sin(2 * np.pi * expected_hour / 24),
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        observed["dow_sin"],
        np.sin(2 * np.pi * expected_dow / 7),
        rtol=0.0,
        atol=1e-7,
    )


def test_cyclic_features_are_append_stable() -> None:
    base = _m5_frame("2026-01-01T00:00:00Z", periods=8)
    prefix_cycles = add_cyclic_time_features(
        base.iloc[:5].set_index("time")
    )
    full_cycles = add_cyclic_time_features(base.set_index("time")).iloc[:5]
    pd.testing.assert_frame_equal(prefix_cycles, full_cycles)


def _native_m5_frame(start: str, periods: int) -> pd.DataFrame:
    """Exact 14-column native MBA frame as produced by the strict M5 source."""
    time = pd.date_range(start, periods=periods, freq="5min", tz="UTC")
    close = 2400.0 + np.cumsum(np.sin(np.arange(periods) / 7.0))
    half_range = 0.8 + 0.1 * np.cos(np.arange(periods) / 11.0)
    high = close + half_range
    low = close - half_range
    open_ = close + 0.1
    half_spread = 0.15
    return pd.DataFrame(
        {
            "time": time,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "bid_open": open_ - half_spread,
            "bid_high": high - half_spread,
            "bid_low": low - half_spread,
            "bid_close": close - half_spread,
            "ask_open": open_ + half_spread,
            "ask_high": high + half_spread,
            "ask_low": low + half_spread,
            "ask_close": close + half_spread,
            "volume": np.full(periods, 25, dtype=np.int64),
        }
    )


def test_canonical_v2_native_frame_emits_mandatory_session_evidence() -> None:
    # The chunked builder indexes the native frame by `time` (no `ts` column),
    # so the session owner must derive current context fields from the index.
    from gx1.scripts.materialize_build_canonical_features_v2 import (
        build_canonical_v2,
    )

    m5 = _native_m5_frame("2026-01-05T00:00:00Z", periods=500)
    v2 = build_canonical_v2(m5)
    assert {name for name in v2 if name.startswith("_v1_")} == set(
        BASIC_V1_FEATURES
    )

    for column in (
        "is_ASIA",
        "is_EU",
        "is_OVERLAP",
        "is_US",
        "session_id",
        "minutes_since_session_open",
        "minutes_to_next_session_boundary",
    ):
        assert column in v2.columns, column
        assert np.isfinite(v2[column].to_numpy(dtype=np.float64)).all(), column
    is_us = v2["is_US"].to_numpy(dtype=np.float64)
    assert np.isfinite(is_us).all()
    assert is_us.max() == 1.0 and is_us.min() == 0.0


def test_add_session_features_without_time_source_fails_closed() -> None:
    from gx1.features.basic_v1 import add_session_features

    frame = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    try:
        add_session_features(frame)
    except RuntimeError as exc:
        assert "SESSION_TIME_SOURCE_MISSING" in str(exc)
    else:
        raise AssertionError("expected SESSION_TIME_SOURCE_MISSING")
