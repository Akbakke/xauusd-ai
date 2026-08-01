import numpy as np
import pandas as pd
from pathlib import Path

from gx1.scripts.materialize_build_canonical_features_v2 import (
    compute_d1_features,
    compute_m15_features,
    merge_asof_features,
)
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


def test_canonical_v2_htf_feature_times_are_close_times_not_bucket_starts() -> None:
    m5 = _m5_frame("2026-01-01T00:00:00Z", periods=24 * 12 * 3)

    d1 = compute_d1_features(m5)
    m15 = compute_m15_features(m5)

    assert pd.Timestamp(d1["time"].iloc[0]) == pd.Timestamp("2026-01-02T00:00:00Z")
    assert pd.Timestamp(m15["time"].iloc[0]) == pd.Timestamp("2026-01-01T00:15:00Z")


def test_merge_asof_features_only_exposes_closed_htf_rows() -> None:
    base = _m5_frame("2026-01-01T00:00:00Z", periods=4)
    close_time = pd.Timestamp("2026-01-01T00:15:00Z")
    extra = pd.DataFrame(
        {
            "time": [close_time],
            "_time_ns": [int(close_time.value)],
            "m15_marker": [1.0],
        }
    )

    merged = merge_asof_features(base, extra)

    assert merged.loc[0:1, "m15_marker"].isna().all()
    # The 00:10 M5 row closes at 00:15 and can immediately use the M15 row
    # that also closed at 00:15. Waiting for the 00:15 label is one M5 stale.
    assert merged.loc[2, "m15_marker"] == 1.0
    assert merged.loc[3, "m15_marker"] == 1.0


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


def test_htf_and_cyclic_features_are_append_stable() -> None:
    base = _m5_frame("2026-01-01T00:00:00Z", periods=8)
    extra = pd.DataFrame(
        {
            "time": [pd.Timestamp("2026-01-01T00:15:00Z")],
            "_time_ns": [pd.Timestamp("2026-01-01T00:15:00Z").value],
            "m15_marker": [7.0],
        }
    )
    prefix_merge = merge_asof_features(base.iloc[:5], extra)
    full_merge = merge_asof_features(base, extra).iloc[:5]
    pd.testing.assert_series_equal(
        prefix_merge["m15_marker"],
        full_merge["m15_marker"],
    )
    prefix_cycles = add_cyclic_time_features(
        base.iloc[:5].set_index("time")
    )
    full_cycles = add_cyclic_time_features(base.set_index("time")).iloc[:5]
    pd.testing.assert_frame_equal(prefix_cycles, full_cycles)


def test_canonical_v2_summary_declares_no_lookahead_contract() -> None:
    repo = Path(__file__).resolve().parents[1]
    text = (repo / "gx1/scripts/materialize_build_canonical_features_v2.py").read_text(encoding="utf-8")

    assert "canonical_features_v2_decision_availability_20260724" in text
    assert '"no_lookahead": True' in text
    assert '"d1_feature_time": "bar_close_time"' in text
    assert '"m15_feature_time": "bar_close_time"' in text
    assert '"m5_decision_availability_shift": "5min"' in text


def test_canonical_v3_manifest_records_source_v2_no_lookahead_provenance() -> None:
    repo = Path(__file__).resolve().parents[1]
    text = (repo / "gx1/scripts/materialize_canonical_v3_augment.py").read_text(encoding="utf-8")

    assert "source_v2_parquet_sha256" in text
    assert "source_v2_no_lookahead" in text
    assert "source_v2_htf_alignment_contract" in text


def _native_m5_frame(start: str, periods: int) -> pd.DataFrame:
    """Exact 14-column native MBA frame as produced by the strict M5 source."""
    time = pd.date_range(start, periods=periods, freq="5min", tz="UTC")
    close = 2400.0 + np.cumsum(np.sin(np.arange(periods) / 7.0))
    high = close + 0.8
    low = close - 0.8
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
    # Regression for the 2026-07-24 pair-bootstrap RED: the chunked builder
    # indexes the native frame by `time` (no `ts` column), and the old
    # `if "ts" in df` guard silently skipped session features until the
    # mandatory session-volatility-pressure block failed closed.
    from gx1.scripts.materialize_build_canonical_features_v2 import (
        build_canonical_v2,
    )

    # >= 288*20 rows so ADR20-backed features get at least one finite value.
    m5 = _native_m5_frame("2026-01-05T00:00:00Z", periods=21 * 288)
    # Real quiet-tape bars have high == low == open == close (102 such bars in
    # the 2020-2026 native M5). They must yield exact zero body/wick shares,
    # not mid-series NaN gaps.
    for degenerate in (700, 1500):
        px = float(m5.loc[degenerate, "close"])
        for col in ("open", "high", "low", "close"):
            m5.loc[degenerate, col] = px
        for col in ("bid_open", "bid_high", "bid_low", "bid_close"):
            m5.loc[degenerate, col] = px - 0.15
        for col in ("ask_open", "ask_high", "ask_low", "ask_close"):
            m5.loc[degenerate, col] = px + 0.15
    v2 = build_canonical_v2(m5)
    for column in ("_v1_body_tr", "_v1_upper_tr", "_v1_lower_tr", "_v1_wick_imbalance"):
        values = v2[column].to_numpy(dtype=np.float64)
        assert np.isfinite(values).all(), column
        for degenerate in (700, 1500):
            assert values[degenerate + 1] == 0.0, (column, degenerate)

    for column in (
        "is_ASIA",
        "is_EU",
        "is_OVERLAP",
        "is_US",
        "session_id",
        "minutes_since_session_open",
        "minutes_to_next_session_boundary",
        "_v1_is_EU",
        "_v1_is_US",
    ):
        assert column in v2.columns, column
        assert np.isfinite(v2[column].to_numpy(dtype=np.float64)).all(), column
    assert "_v1_session_volatility_pressure" in v2.columns
    is_us = v2["_v1_is_US"].to_numpy(dtype=np.float64)
    assert np.isfinite(is_us).all()
    assert is_us.max() == 1.0 and is_us.min() == 0.0
    pressure = v2["_v1_session_volatility_pressure"].to_numpy(dtype=np.float64)
    assert np.isfinite(pressure[288:]).any()
    assert (pressure[np.isfinite(pressure)] > 0.0).all()


def test_add_session_features_without_time_source_fails_closed() -> None:
    from gx1.features.basic_v1 import add_session_features

    frame = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    try:
        add_session_features(frame)
    except RuntimeError as exc:
        assert "SESSION_TIME_SOURCE_MISSING" in str(exc)
    else:
        raise AssertionError("expected SESSION_TIME_SOURCE_MISSING")
