from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.features.entry_candle_primitives_v1 import (
    CANDLE_PRIMITIVE_FEATURE_NAMES,
    CandlePrimitiveCarryState,
    build_entry_candle_primitive_layer,
    compute_entry_candle_primitive_chunk,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range(
                "2026-01-01", periods=4, freq="5min", tz="UTC"
            ),
            "open": [10.0, 11.0, 12.0, 12.0],
            "high": [12.0, 13.0, 12.0, 14.0],
            "low": [9.0, 10.0, 12.0, 11.0],
            "close": [11.0, 12.0, 12.0, 13.0],
        }
    )


def test_raw_candle_geometry_has_exact_units_and_one_honest_prefix() -> None:
    values, names = build_entry_candle_primitive_layer(_frame())
    index = {name: i for i, name in enumerate(names)}

    assert tuple(names) == CANDLE_PRIMITIVE_FEATURE_NAMES
    assert values.shape == (4, len(names))
    assert all("pattern" not in name and "score" not in name for name in names)
    assert np.isfinite(values[:, :5]).all()
    relational = [
        column
        for column, name in enumerate(names)
        if column >= 5 and "body_direction_duration" not in name
    ]
    assert np.isnan(values[0, relational]).all()
    assert values[
        0,
        index["candle.raw_observed_body_direction_duration_bars"],
    ] == 1.0
    assert np.isfinite(values[1:, 5:]).all()

    assert values[0, index["candle.raw_body_signed_range"]] == pytest.approx(
        1.0 / 3.0
    )
    assert values[0, index["candle.raw_upper_wick_share"]] == pytest.approx(
        1.0 / 3.0
    )
    assert values[0, index["candle.raw_lower_wick_share"]] == pytest.approx(
        1.0 / 3.0
    )
    assert values[0, index["candle.raw_close_location"]] == pytest.approx(
        2.0 / 3.0
    )
    assert values[1, index["candle.raw_close_change_local_geometry"]] == pytest.approx(
        1.0 / 3.0
    )


def test_zero_range_is_explicit_and_not_a_false_extreme_close() -> None:
    values, names = build_entry_candle_primitive_layer(_frame())
    index = {name: i for i, name in enumerate(names)}

    assert values[2, index["candle.raw_zero_range_flag"]] == 1.0
    assert values[2, index["candle.raw_body_signed_range"]] == 0.0
    assert values[2, index["candle.raw_upper_wick_share"]] == 0.0
    assert values[2, index["candle.raw_lower_wick_share"]] == 0.0
    assert values[2, index["candle.raw_close_location"]] == 0.5


def test_candle_primitive_owner_is_prefix_invariant() -> None:
    frame = _frame()
    full, names = build_entry_candle_primitive_layer(frame)
    prefix, prefix_names = build_entry_candle_primitive_layer(frame.iloc[:3])

    assert prefix_names == names
    np.testing.assert_allclose(full[:3], prefix, equal_nan=True)

    changed = frame.copy()
    changed.loc[3, ["open", "high", "low", "close"]] = [20.0, 22.0, 19.0, 21.0]
    mutated, _ = build_entry_candle_primitive_layer(changed)
    np.testing.assert_allclose(full[:3], mutated[:3], equal_nan=True)


def test_candle_primitive_owner_rejects_bad_source() -> None:
    invalid = _frame()
    invalid.loc[1, "high"] = 9.0
    with pytest.raises(RuntimeError, match="SOURCE_GEOMETRY_INVALID"):
        build_entry_candle_primitive_layer(invalid)

    naive = _frame()
    naive["time"] = naive["time"].dt.tz_localize(None)
    with pytest.raises(RuntimeError, match="TIME_NOT_UTC"):
        build_entry_candle_primitive_layer(naive)


def _relation_frame(freq: str = "5min") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range(
                "2026-01-01T00:00:00Z", periods=9, freq=freq
            ),
            "open": [11.0, 9.5, 11.0, 11.0, 13.0, 14.0, 12.5, 13.2, 13.0],
            "high": [12.0, 12.5, 12.0, 12.0, 14.0, 15.0, 13.5, 13.5, 13.0],
            "low": [9.0, 9.0, 9.5, 9.5, 11.8, 12.5, 11.0, 12.0, 13.0],
            "close": [10.0, 11.5, 10.5, 10.5, 12.0, 13.0, 13.0, 12.3, 13.0],
        }
    )


def test_exact_relation_events_have_no_textbook_pattern_aliases() -> None:
    values, names = build_entry_candle_primitive_layer(_relation_frame())
    index = {name: i for i, name in enumerate(names)}

    assert all(
        token not in name
        for name in names
        for token in (
            "hammer",
            "doji",
            "engulf",
            "morning_star",
            "confidence",
            "quality",
            "score",
        )
    )
    # Row 1 is a bullish body whose exact interval contains row 0's bearish
    # body. This is geometry, not a trend-conditioned textbook pattern.
    assert values[1, index["candle.raw_body_contains_previous_flag"]] == 1.0
    assert (
        values[
            1,
            index["candle.raw_bull_body_covers_previous_bear_body_event"],
        ]
        == 1.0
    )
    assert values[2, index["candle.raw_range_contained_by_previous_flag"]] == 1.0
    # Exact equality is readable from zero high/low deltas; no near-dead
    # equality flag is duplicated into the mandatory surface.
    assert values[3, index["candle.raw_high_change_local_geometry"]] == 0.0
    assert values[3, index["candle.raw_low_change_local_geometry"]] == 0.0
    assert values[4, index["candle.raw_open_above_previous_high_local_geometry"]] > 0.0
    assert values[5, index["candle.raw_high_rejection_previous_high_event"]] == 1.0
    assert values[5, index["candle.raw_high_rejection_depth_local_geometry"]] > 0.0
    assert values[6, index["candle.raw_low_rejection_previous_low_event"]] == 1.0
    assert values[6, index["candle.raw_low_rejection_depth_local_geometry"]] > 0.0
    assert (
        values[
            7,
            index["candle.raw_bear_body_covers_previous_bull_body_event"],
        ]
        == 1.0
    )


def test_state_duration_identity_and_polarity_are_exact() -> None:
    values, names = build_entry_candle_primitive_layer(_relation_frame())
    index = {name: i for i, name in enumerate(names)}
    body_duration = values[
        :, index["candle.raw_observed_body_direction_duration_bars"]
    ]
    range_duration = values[
        :, index["candle.raw_observed_range_relation_duration_bars"]
    ]
    np.testing.assert_array_equal(body_duration[:4], [1.0, 1.0, 1.0, 2.0])
    assert np.isnan(range_duration[0])
    # Rows 2 and 3 are different exact relation identities (contained/equal),
    # so each duration restarts at one rather than merging "inside-ish" bars.
    np.testing.assert_array_equal(range_duration[1:4], [1.0, 1.0, 1.0])


def test_zero_range_and_gap_after_zero_range_are_finite_and_flagged() -> None:
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC"),
            "open": [10.0, 10.0, 11.0],
            "high": [10.0, 10.0, 11.0],
            "low": [10.0, 10.0, 11.0],
            "close": [10.0, 10.0, 11.0],
        }
    )
    values, names = build_entry_candle_primitive_layer(frame)
    index = {name: i for i, name in enumerate(names)}
    assert (values[:, index["candle.raw_zero_range_flag"]] == 1.0).all()
    assert values[2, index["candle.raw_open_gap_local_geometry"]] == 1.0
    assert values[2, index["candle.raw_open_above_previous_high_local_geometry"]] == 1.0
    assert np.isfinite(values[1:]).all()


@pytest.mark.parametrize("split", [1, 2, 4, 8])
def test_chunk_carry_is_exact_at_every_relation_boundary(split: int) -> None:
    frame = _relation_frame()
    full, names = build_entry_candle_primitive_layer(frame)
    left, left_names, carry = compute_entry_candle_primitive_chunk(
        frame.iloc[:split]
    )
    right, right_names, carry = compute_entry_candle_primitive_chunk(
        frame.iloc[split:],
        carry=carry,
    )
    assert left_names == right_names == names
    np.testing.assert_array_equal(np.concatenate([left, right]), full)
    np.testing.assert_array_equal(right[0], full[split])
    assert carry.rows_seen == len(frame)


def test_chunk_carry_rejects_inconsistent_or_overlapping_state() -> None:
    frame = _relation_frame()
    _, _, carry = compute_entry_candle_primitive_chunk(frame.iloc[:3])

    with pytest.raises(RuntimeError, match="CARRY_INVALID"):
        compute_entry_candle_primitive_chunk(
            frame.iloc[3:],
            carry=CandlePrimitiveCarryState(rows_seen=3),
        )
    with pytest.raises(RuntimeError, match="CARRY_TIME_OVERLAP"):
        compute_entry_candle_primitive_chunk(frame.iloc[2:], carry=carry)


@pytest.mark.parametrize("freq", ["1min", "5min", "15min", "1h", "4h", "1D"])
def test_one_owner_has_value_parity_on_each_native_clock(freq: str) -> None:
    baseline, names = build_entry_candle_primitive_layer(_relation_frame("5min"))
    observed, observed_names = build_entry_candle_primitive_layer(
        _relation_frame(freq)
    )
    assert observed_names == names
    np.testing.assert_array_equal(observed, baseline)


def test_future_ohlc_mutation_cannot_change_prefix() -> None:
    frame = _relation_frame()
    baseline, _ = build_entry_candle_primitive_layer(frame)
    changed = frame.copy()
    changed.loc[changed.index[-1], ["open", "high", "low", "close"]] = [
        20.0,
        22.0,
        19.0,
        21.0,
    ]
    observed, _ = build_entry_candle_primitive_layer(changed)
    np.testing.assert_array_equal(observed[:-1], baseline[:-1])
