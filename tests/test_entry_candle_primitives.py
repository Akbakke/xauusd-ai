from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from gx1.features.entry_candle_primitives_v1 import (
    CANDLE_PRIMITIVE_FEATURE_NAMES,
    CANDLE_PRIMITIVE_RELATIONAL_FEATURE_NAMES,
    CANDLE_PRIMITIVE_WHOLE_BAR_FEATURE_NAMES,
    CandlePrimitiveCarryState,
    build_entry_candle_primitive_layer,
    compute_entry_candle_primitive_chunk,
)

# The block boundary derives from the owner's declared tuples; restating it as
# a literal here is exactly the stale-index hazard the owner's
# ``_require_emitted_row_layout`` guard exists to remove.
_RELATIONAL_BASE = len(CANDLE_PRIMITIVE_WHOLE_BAR_FEATURE_NAMES)


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
    assert np.isfinite(values[:, :_RELATIONAL_BASE]).all()
    relational = [
        column
        for column, name in enumerate(names)
        if column >= _RELATIONAL_BASE
        and "body_direction_duration" not in name
    ]
    assert np.isnan(values[0, relational]).all()
    assert values[
        0,
        index["candle.raw_observed_body_direction_duration_bars"],
    ] == 1.0
    assert np.isfinite(values[1:, _RELATIONAL_BASE:]).all()

    assert values[0, index["candle.raw_body_signed_range"]] == pytest.approx(
        1.0 / 3.0
    )
    assert values[0, index["candle.raw_upper_wick_share"]] == pytest.approx(
        1.0 / 3.0
    )
    assert values[0, index["candle.raw_lower_wick_share"]] == pytest.approx(
        1.0 / 3.0
    )
    assert values[1, index["candle.raw_close_change_local_geometry"]] == pytest.approx(
        1.0 / 3.0
    )


def test_zero_range_takes_the_storage_zero_and_stays_exactly_recoverable() -> None:
    """A ``high == low`` bar emits the storage zero and needs no flag column.

    ``candle.raw_zero_range_flag`` was retired on 2026-08-15: it is constant
    0.0 post-warmup on the H4 and D1 lanes of the declared tape, so it can
    never reach a liveness verdict there, and a declared-constant exemption
    only moves the failure to [ENTRY_INPUT_NORMALIZATION_UNSCALEABLE].

    Nothing was lost with it.  The three range shares partition the bar range
    exactly (``upper_wick + lower_wick + |body| == high - low``), so on a
    positive-range bar they divide to sum to one and cannot all be zero, while
    a zero-range bar emits all three as the storage zero.  The biconditional
    asserted below is therefore exact, not approximate, and it is what makes
    the retirement evidence-preserving under CLAUDE.md rule 4.
    """

    values, names = build_entry_candle_primitive_layer(_frame())
    index = {name: i for i, name in enumerate(names)}

    assert "candle.raw_zero_range_flag" not in index
    assert not any("zero_range" in name for name in names)

    assert values[2, index["candle.raw_body_signed_range"]] == 0.0
    assert values[2, index["candle.raw_upper_wick_share"]] == 0.0
    assert values[2, index["candle.raw_lower_wick_share"]] == 0.0

    # A real-range doji shares the zero body, so the body alone does not
    # identify a zero-range bar — the wick shares do.
    doji = _frame()
    doji.loc[2, ["open", "high", "low", "close"]] = [12.0, 13.0, 11.0, 12.0]
    doji_values, _ = build_entry_candle_primitive_layer(doji)
    assert doji_values[2, index["candle.raw_body_signed_range"]] == 0.0
    assert doji_values[2, index["candle.raw_upper_wick_share"]] == 0.5
    assert doji_values[2, index["candle.raw_lower_wick_share"]] == 0.5

    # The exact recovery, over zero-range, doji, marubozu and both
    # extreme-touching shapes in one frame.
    shapes = pd.DataFrame(
        {
            "time": pd.date_range("2026-02-01", periods=5, freq="5min", tz="UTC"),
            "open": [12.0, 12.0, 11.0, 11.0, 13.0],
            "high": [12.0, 13.0, 13.0, 13.0, 13.0],
            "low": [12.0, 11.0, 11.0, 10.0, 11.0],
            "close": [12.0, 12.0, 13.0, 13.0, 11.0],
        }
    )
    shape_values, shape_names = build_entry_candle_primitive_layer(shapes)
    shape_index = {name: i for i, name in enumerate(shape_names)}
    body = shape_values[:, shape_index["candle.raw_body_signed_range"]]
    upper = shape_values[:, shape_index["candle.raw_upper_wick_share"]]
    lower = shape_values[:, shape_index["candle.raw_lower_wick_share"]]
    recovered = (body == 0.0) & (upper == 0.0) & (lower == 0.0)
    truth = (
        shapes["high"].to_numpy(dtype=np.float64)
        == shapes["low"].to_numpy(dtype=np.float64)
    )
    np.testing.assert_array_equal(recovered, truth)
    # And the partition identity the recovery rests on, on every other bar.
    np.testing.assert_allclose(
        np.abs(body[~truth]) + upper[~truth] + lower[~truth], 1.0, atol=1e-6
    )


def test_v30_wave2_retired_columns_are_absent_and_exactly_recoverable() -> None:
    """The four v4 retirements are gone AND their algebra survives.

    The absence assertions are the regression half: every one of them fails
    against the pre-v4 owner, where all four names are declared and written.
    The identity assertions are the CLAUDE.md rule 4 half: each retired column
    is reproduced, bit for bit or to float tolerance, out of columns that are
    still emitted by this same owner and therefore reach the same specialist
    family.
    """

    retired = (
        "candle.raw_close_location",
        "candle.raw_range_change_local_geometry",
        "candle.raw_high_rejection_depth_local_geometry",
        "candle.raw_low_rejection_depth_local_geometry",
    )
    for name in retired:
        assert name not in CANDLE_PRIMITIVE_FEATURE_NAMES
        assert name not in CANDLE_PRIMITIVE_WHOLE_BAR_FEATURE_NAMES
        assert name not in CANDLE_PRIMITIVE_RELATIONAL_FEATURE_NAMES

    frame = _relation_frame()
    values, names = build_entry_candle_primitive_layer(frame)
    index = {name: i for i, name in enumerate(names)}
    for name in retired:
        assert name not in index

    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    open_ = frame["open"].to_numpy(dtype=np.float64)
    close = frame["close"].to_numpy(dtype=np.float64)
    bar_range = high - low
    positive = bar_range > 0.0

    # 1. close_location == lower_wick_share + max(body_signed_range, 0)
    body = values[:, index["candle.raw_body_signed_range"]].astype(np.float64)
    lower = values[:, index["candle.raw_lower_wick_share"]].astype(np.float64)
    recovered_location = lower + np.maximum(body, 0.0)
    np.testing.assert_allclose(
        recovered_location[positive],
        (close[positive] - low[positive]) / bar_range[positive],
        atol=1e-6,
    )

    # 2. range_change == high_change - low_change, same denominator, exactly.
    high_change = values[
        :, index["candle.raw_high_change_local_geometry"]
    ].astype(np.float64)
    low_change = values[
        :, index["candle.raw_low_change_local_geometry"]
    ].astype(np.float64)
    body_change = values[
        :, index["candle.raw_body_change_local_geometry"]
    ].astype(np.float64)
    previous_range = bar_range[:-1]
    # The shared scalar is recovered from a sibling that divides by it, so the
    # identity is checked against the raw geometry, not against itself.
    abs_body = np.abs(close - open_)
    usable = body_change[1:] != 0.0
    scale = (abs_body[1:][usable] - abs_body[:-1][usable]) / body_change[1:][usable]
    expected_range_change = (bar_range[1:][usable] - previous_range[usable]) / scale
    assert usable.any()
    np.testing.assert_allclose(
        (high_change - low_change)[1:][usable],
        expected_range_change,
        atol=1e-6,
    )

    # 3./4. the two rejection depths are the product of a retained change
    # column and a retained exactly-binary event column.
    high_event = values[
        :, index["candle.raw_high_rejection_previous_high_event"]
    ].astype(np.float64)
    low_event = values[
        :, index["candle.raw_low_rejection_previous_low_event"]
    ].astype(np.float64)
    assert set(np.unique(high_event[1:])) <= {0.0, 1.0}
    assert set(np.unique(low_event[1:])) <= {0.0, 1.0}
    assert high_event[5] == 1.0 and low_event[6] == 1.0
    # On the firing bar the depth WAS exactly the change column (up) and its
    # negation (down); off-event it was the storage zero, which the flag
    # already announces.
    assert high_change[5] > 0.0
    assert -low_change[6] > 0.0
    np.testing.assert_allclose(
        high_change[1:] * high_event[1:],
        np.where(high_event[1:] > 0.0, high_change[1:], 0.0),
        atol=0.0,
    )
    np.testing.assert_allclose(
        -low_change[1:] * low_event[1:],
        np.where(low_event[1:] > 0.0, -low_change[1:], 0.0),
        atol=0.0,
    )


def test_upper_wick_share_is_not_recoverable_and_therefore_stays() -> None:
    """A gravestone and a zero-range bar agree on body and lower wick.

    ``upper_wick_share`` is the only column that separates them, so the
    apparent recovery ``1 - lower - |body|`` is wrong on exactly the rows the
    v3 zero-range retirement depends on.  This test is the standing block
    against a later wave retiring it by the same argument that carried the
    four v4 retirements.
    """

    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-03-01", periods=2, freq="5min", tz="UTC"),
            # bar 0: gravestone -- open == close == low < high
            # bar 1: zero range  -- open == close == low == high
            "open": [11.0, 11.0],
            "high": [13.0, 11.0],
            "low": [11.0, 11.0],
            "close": [11.0, 11.0],
        }
    )
    values, names = build_entry_candle_primitive_layer(frame)
    index = {name: i for i, name in enumerate(names)}
    body = values[:, index["candle.raw_body_signed_range"]]
    lower = values[:, index["candle.raw_lower_wick_share"]]
    upper = values[:, index["candle.raw_upper_wick_share"]]

    assert body[0] == 0.0 and lower[0] == 0.0
    assert body[1] == 0.0 and lower[1] == 0.0
    assert upper[0] == 1.0
    assert upper[1] == 0.0
    # The proposed recovery collapses the two shapes onto the gravestone.
    recovered = 1.0 - lower - np.abs(body)
    assert recovered[0] == pytest.approx(upper[0])
    assert recovered[1] != pytest.approx(upper[1])


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
    # The retired depth column WAS high_change on exactly this bar, and
    # high_change stays; the magnitude is still on the surface.
    assert values[5, index["candle.raw_high_change_local_geometry"]] > 0.0
    assert values[6, index["candle.raw_low_rejection_previous_low_event"]] == 1.0
    assert values[6, index["candle.raw_low_change_local_geometry"]] < 0.0
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


def test_zero_range_and_gap_after_zero_range_are_finite_and_recoverable() -> None:
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
    # Every bar here is zero-range, and every bar is recoverable as such from
    # the three surviving shares alone.
    assert (values[:, index["candle.raw_body_signed_range"]] == 0.0).all()
    assert (values[:, index["candle.raw_upper_wick_share"]] == 0.0).all()
    assert (values[:, index["candle.raw_lower_wick_share"]] == 0.0).all()
    # Row 2 opens a full unit above a ZERO-range previous bar, so the position
    # ratio is an undefined 0/0 and takes the storage zero.  That zero would be
    # a placeholder reading as "opened at the previous low" (rule 2e) if it
    # stood alone -- it does not: the whole magnitude is carried by the clamped
    # sibling on exactly these rows, and the rows themselves stay identifiable
    # from the previous row's three shares (v3 biconditional, asserted above).
    assert values[2, index["candle.raw_open_position_previous_range"]] == 0.0
    assert values[2, index["candle.raw_open_above_previous_high_local_geometry"]] == 1.0
    assert np.isfinite(values[1:]).all()


def test_emitted_column_layout_is_derived_from_the_declared_name_tuples() -> None:
    """The write offsets must be a bijection onto the declared column order.

    This is the hazard the 26 hand-numbered ``matrix[:, k]`` writes carried: a
    field entering or leaving the tuple silently renumbered every write after
    it, and every column kept its dtype, name and finiteness prefix, so no
    downstream gate could see two fields swap meanings.
    """

    module = importlib.import_module(
        "gx1.features.entry_candle_primitives_v1"
    )
    width = len(CANDLE_PRIMITIVE_FEATURE_NAMES)
    assert module._EMITTED_COLUMN_INDICES == tuple(range(width))
    assert (
        CANDLE_PRIMITIVE_WHOLE_BAR_FEATURE_NAMES
        + CANDLE_PRIMITIVE_RELATIONAL_FEATURE_NAMES
        == CANDLE_PRIMITIVE_FEATURE_NAMES
    )
    assert len(set(CANDLE_PRIMITIVE_FEATURE_NAMES)) == width

    # The guard is executable, not decorative: a stale offset must raise.
    original = module._EMITTED_COLUMN_INDICES
    try:
        module._EMITTED_COLUMN_INDICES = original[:-1] + (original[-1] - 1,)
        with pytest.raises(RuntimeError, match="CANDLE_PRIMITIVE_ROW_LAYOUT_INVALID"):
            module._require_emitted_row_layout()
    finally:
        module._EMITTED_COLUMN_INDICES = original
    module._require_emitted_row_layout()


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


def test_open_gap_era_proxy_is_retired_and_repaired() -> None:
    """The v5 era repair: absence, exact arithmetic, and the rule-4 recovery.

    Three independent halves, each of which fails on its own against the v4
    owner:

    * ABSENCE. ``candle.raw_open_gap_local_geometry`` is gone from the declared
      tuples and from the emitted index.
    * VALUE (the non-vacuity half). The emitted column must equal
      ``(open[t] - low[t-1]) / (high[t-1] - low[t-1])`` on every positive
      previous-range row. Reverting only the arithmetic to the retired
      ``(open[t] - close[t-1]) / local_geometry_scale`` -- keeping the new
      name, the v5 version string and every comment -- fails here, which is
      what makes the rename non-vacuous.
    * RULE 4. The retired reading survives exactly, in previous-range units, as
      ``position[t] - (lower_wick_share[t-1] + max(body_signed_range[t-1], 0))``
      over columns this same owner still emits. Measured on the complete
      declared native M5 tape the identity holds to 7.1e-15 in float64 over
      537,645 rows; here it is asserted to float32 storage tolerance.
    """

    retired = "candle.raw_open_gap_local_geometry"
    assert retired not in CANDLE_PRIMITIVE_FEATURE_NAMES
    assert retired not in CANDLE_PRIMITIVE_RELATIONAL_FEATURE_NAMES
    assert retired not in CANDLE_PRIMITIVE_WHOLE_BAR_FEATURE_NAMES
    repaired = "candle.raw_open_position_previous_range"
    assert repaired in CANDLE_PRIMITIVE_RELATIONAL_FEATURE_NAMES
    # The repaired name must not claim the local_geometry_scale convention it
    # no longer uses (rule 19: one denominator convention per spelling).
    assert "local_geometry" not in repaired

    frame = _relation_frame()
    values, names = build_entry_candle_primitive_layer(frame)
    index = {name: i for i, name in enumerate(names)}
    assert retired not in index

    open_ = frame["open"].to_numpy(dtype=np.float64)
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    close = frame["close"].to_numpy(dtype=np.float64)
    previous_range = (high - low)[:-1]
    positive = previous_range > 0.0
    assert positive.any(), "fixture must exercise the defined branch"

    observed = values[1:, index[repaired]].astype(np.float64)
    expected = (open_[1:] - low[:-1]) / previous_range
    np.testing.assert_allclose(
        observed[positive], expected[positive], rtol=0.0, atol=1e-6
    )
    # The retired arithmetic must NOT reproduce the column: this is the
    # discriminating assertion, not a restatement of the one above.
    bar_range = high - low
    body = np.abs(close - open_)
    price_scale = np.maximum.reduce(
        [np.abs(v) for v in (open_[1:], high[1:], low[1:], close[1:],
                             open_[:-1], high[:-1], low[:-1], close[:-1])]
        + [np.ones_like(open_[1:])]
    )
    local_geometry_scale = np.maximum.reduce([
        bar_range[1:], previous_range,
        np.abs(open_[1:] - close[:-1]), np.abs(close[1:] - close[:-1]),
        np.abs(high[1:] - high[:-1]), np.abs(low[1:] - low[:-1]),
        np.abs(bar_range[1:] - previous_range), np.abs(body[1:] - body[:-1]),
        price_scale * np.finfo(np.float64).eps,
    ])
    retired_values = (open_[1:] - close[:-1]) / local_geometry_scale
    assert not np.allclose(
        observed[positive], retired_values[positive], rtol=0.0, atol=1e-6
    )

    # Rule 4: the retired reading, in previous-range units, out of emitted
    # columns only.
    body_share = values[:, index["candle.raw_body_signed_range"]].astype(np.float64)
    lower_share = values[:, index["candle.raw_lower_wick_share"]].astype(np.float64)
    previous_close_location = lower_share[:-1] + np.maximum(body_share[:-1], 0.0)
    recovered = observed - previous_close_location
    target = (open_[1:] - close[:-1]) / previous_range
    np.testing.assert_allclose(
        recovered[positive], target[positive], rtol=0.0, atol=1e-6
    )


def test_open_position_is_a_bounded_location_on_a_contained_open() -> None:
    """A bar opening at the previous midpoint reads 0.5, not a seam magnitude.

    This is the fidelity statement the retired column could not make: its value
    depended on how far the feed's first tick of the bar had drifted from the
    previous bar's last tick, which is a property of the tape's sampling
    density rather than of the market.
    """

    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC"),
            "open": [100.0, 100.0, 108.0],
            "high": [110.0, 104.0, 109.0],
            "low": [90.0, 99.0, 107.0],
            "close": [104.0, 103.0, 108.5],
        }
    )
    values, names = build_entry_candle_primitive_layer(frame)
    index = {name: i for i, name in enumerate(names)}
    position = values[:, index["candle.raw_open_position_previous_range"]]
    # Row 1 opens at 100 inside the [90, 110] previous range -> 0.5 exactly.
    assert position[1] == pytest.approx(0.5)
    # Row 2 opens at 108 above the [99, 104] previous range -> above 1, and the
    # clamped sibling flags the same event.
    assert position[2] > 1.0
    assert values[2, index["candle.raw_open_above_previous_high_local_geometry"]] > 0.0
    assert values[2, index["candle.raw_open_below_previous_low_local_geometry"]] == 0.0
