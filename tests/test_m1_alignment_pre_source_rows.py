"""Alignment rows that precede the M1 surface are excluded, gaps inside are not.

Every emitted M1 row must own a complete higher-timeframe context. The widest is
the 252-bar D1 window - about 353 calendar days, built by resampling the native
M5 tape - so the first M1 row that can own a full D1 window lands roughly a year
after the M5 source begins. A pair generation built before that rule asks for
earlier rows, which cannot be produced with valid D1 context at all.

Those leading rows are excluded. Everything else about the subset requirement is
unchanged: a missing row INSIDE the covered span is still a hard failure, and
non-leading pre-source rows are rejected outright.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


WARMUP = 201


def _resolve_positions(source: pd.DatetimeIndex, surface: pd.DatetimeIndex):
    """The alignment resolution as the producer performs it."""

    warmup_floor = source[min(WARMUP, len(source) - 1)]
    covered = surface >= warmup_floor
    pre_source_rows = int(np.count_nonzero(~covered))
    if pre_source_rows:
        if not bool(covered[pre_source_rows:].all()):
            raise RuntimeError(
                "M1_FEATURE_BASE_ALIGNMENT_PRE_WARMUP_ROWS_NOT_LEADING"
            )
        surface = surface[covered]
        if len(surface) == 0:
            raise RuntimeError(
                "M1_FEATURE_BASE_ALIGNMENT_NO_ROWS_WITHIN_SOURCE_SPAN"
            )
    positions = source.get_indexer(surface)
    if np.any(positions < 0):
        raise RuntimeError(
            "M1_FEATURE_BASE_ALIGNMENT_TIME_NOT_SUBSET: "
            f"missing_rows={int(np.count_nonzero(positions < 0))}"
        )
    return positions, pre_source_rows


def _minutes(start: str, count: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=count, freq="min", tz="UTC")


def test_rows_before_the_causal_warmup_are_excluded() -> None:
    source = _minutes("2020-09-27T22:00", 500)
    surface = _minutes("2020-09-27T21:00", 560)

    positions, pre_source_rows = _resolve_positions(source, surface)

    # 60 rows precede the source, and the first WARMUP rows of the source
    # itself cannot carry a valid EMA/derivative value.
    assert pre_source_rows == 60 + WARMUP
    np.testing.assert_array_equal(positions, np.arange(WARMUP, 500))


def test_a_gap_inside_the_covered_span_still_fails_closed() -> None:
    source = _minutes("2020-09-27T22:00", 500)
    surface = source[WARMUP:].delete(50)
    surface = surface.append(
        pd.DatetimeIndex(["2020-09-28T12:34:30Z"])
    ).sort_values()

    with pytest.raises(RuntimeError, match="ALIGNMENT_TIME_NOT_SUBSET"):
        _resolve_positions(source, surface)


def test_non_leading_pre_warmup_rows_are_rejected() -> None:
    source = _minutes("2020-09-27T22:00", 400)
    surface = source[WARMUP:300].append(
        pd.DatetimeIndex(["2019-12-15T23:59:00Z"])
    )

    with pytest.raises(RuntimeError, match="PRE_WARMUP_ROWS_NOT_LEADING"):
        _resolve_positions(source, surface)


def test_an_alignment_entirely_before_the_source_is_rejected() -> None:
    source = _minutes("2020-09-27T22:00", 400)
    surface = _minutes("2019-12-15T23:59", 100)

    with pytest.raises(RuntimeError, match="NO_ROWS_WITHIN_SOURCE_SPAN"):
        _resolve_positions(source, surface)


def test_a_subset_that_starts_after_the_warmup_is_unaffected() -> None:
    source = _minutes("2020-09-27T22:00", 800)
    surface = source[WARMUP + 50 : 700]

    positions, pre_source_rows = _resolve_positions(source, surface)

    assert pre_source_rows == 0
    np.testing.assert_array_equal(positions, np.arange(WARMUP + 50, 700))
