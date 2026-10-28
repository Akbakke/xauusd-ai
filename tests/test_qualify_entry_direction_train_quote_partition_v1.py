"""Synthetic tests for the bounded TRAIN-year quote join; no real payloads."""
import copy

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_direction_price_projection_v1 import compare_direction_native_midpoint
from gx1.scripts.qualify_entry_direction_train_quote_partition_v1 import (
    TRAIN_END_NS, add_year_partition, empty_quote_buffers,
)


def ns(value):
    return int(pd.Timestamp(value).value)


def sample():
    decision = ns("2024-12-31T23:55:00Z")
    terminal = decision + np.array([1, 5, 15], np.int64) * 60_000_000_000
    bid0, ask0 = 100., 100.02
    bidt = np.array([100.1, 99.9, 100.], np.float64)
    askt = np.array([100.12, 99.92, 100.02], np.float64)
    midpoint = (bid0 + ask0) / 2
    moves = (bidt + askt) / 2 - midpoint
    saved = {"parent_rows": np.array([0], np.int64),
             "bar_start_ns": np.array([decision - 300_000_000_000], np.int64),
             "dynamic_entry_m1_row": np.array([1000], np.int64),
             "decision_ns": np.array([decision], np.int64),
             "terminal_time_ns": terminal.reshape(1, 3),
             "elapsed_wall_seconds": np.array([[60, 300, 900]], np.int64),
             "terminal_source_row": np.array([[1001, 1002, 1003]], np.int64),
             "observed": np.ones((1, 3), np.bool_), "mid_move": moves.reshape(1, 3),
             "mid_return_bps": (10000.0 * moves / midpoint).reshape(1, 3),
             "sign": np.sign(moves).astype(np.int8).reshape(1, 3)}
    years = [
        (2024, np.array([decision, decision + 60_000_000_000], np.int64),
         {"open": np.array([100.01, 100.11]), "bid_open": np.array([100., 100.1]),
          "ask_open": np.array([100.02, 100.12])}, 1000),
        (2025, np.array([terminal[1], terminal[2]], np.int64),
         {"open": np.array([99.91, 100.01]), "bid_open": np.array([99.9, 100.]),
          "ask_open": np.array([99.92, 100.02])}, 1002),
    ]
    return saved, years


def join(saved, years):
    buffers = empty_quote_buffers(len(saved["parent_rows"]))
    for year, clock, prices, offset in years:
        add_year_partition(saved, buffers, year=year, times_ns=clock,
                           prices=prices, row_offset=offset, cutoff_ns=TRAIN_END_NS)
    return buffers


def test_exact_cross_year_midpoint_and_row_parity():
    saved, years = sample()
    result = join(saved, years)
    report = compare_direction_native_midpoint(saved, entry_quotes=result["entry"],
                                               terminal_quotes=result["terminal"], cutoff_ns=TRAIN_END_NS)
    assert all(report["legacy_bit_parity"].values())
    assert result["terminal"]["source_row"].tolist() == [[1001, 1002, 1003]]
    assert [r["observed"] for r in report["per_horizon"]] == [1, 1, 1]


def test_missing_terminal_remains_missing_not_zero_price():
    saved, years = sample()
    saved["terminal_source_row"][0, 2] = -1
    saved["observed"][0, 2] = False
    for key in ("mid_move", "mid_return_bps", "sign"):
        saved[key][0, 2] = 0
    y, times, prices, offset = years[1]
    years[1] = (y, times[:1], {k: v[:1] for k, v in prices.items()}, offset)
    result = join(saved, years)
    report = compare_direction_native_midpoint(saved, entry_quotes=result["entry"],
                                               terminal_quotes=result["terminal"], cutoff_ns=TRAIN_END_NS)
    assert report["per_horizon"][2]["mature_missing"] == 1
    assert result["terminal"]["source_row"][0, 2] == -1


def test_wrong_source_offset_fails_actual_parity():
    saved, years = sample()
    y, times, prices, offset = years[1]
    years[1] = (y, times, prices, offset + 1)
    result = join(saved, years)
    with pytest.raises(RuntimeError, match="MIDPOINT_TERMINAL_JOIN"):
        compare_direction_native_midpoint(saved, entry_quotes=result["entry"],
                                          terminal_quotes=result["terminal"], cutoff_ns=TRAIN_END_NS)


def test_wrong_saved_price_fails_bit_parity():
    saved, years = sample()
    result = join(saved, years)
    saved["mid_return_bps"][0, 0] += 1.
    with pytest.raises(RuntimeError, match="MIDPOINT_LEGACY_BIT_PARITY"):
        compare_direction_native_midpoint(saved, entry_quotes=result["entry"],
                                          terminal_quotes=result["terminal"], cutoff_ns=TRAIN_END_NS)


def test_duplicate_partition_is_rejected():
    saved, years = sample()
    with pytest.raises(RuntimeError, match="DUPLICATE_QUOTE"):
        join(saved, [years[0], years[0]])


def test_wrong_year_and_out_of_year_quote_are_rejected():
    saved, years = sample()
    y, times, prices, offset = years[0]
    with pytest.raises(RuntimeError, match="YEAR_BOUNDARY"):
        add_year_partition(saved, empty_quote_buffers(1), year=2025, times_ns=times,
                           prices=prices, row_offset=offset, cutoff_ns=TRAIN_END_NS)
    with pytest.raises(RuntimeError, match="YEAR_SCOPE"):
        add_year_partition(saved, empty_quote_buffers(1), year=2026, times_ns=times,
                           prices=prices, row_offset=offset, cutoff_ns=TRAIN_END_NS)


def test_crossed_quote_is_rejected():
    saved, years = sample()
    y, times, prices, offset = years[0]
    broken = copy.deepcopy(prices)
    broken["ask_open"][0] = 99.
    with pytest.raises(RuntimeError, match="CROSSED_QUOTES"):
        add_year_partition(saved, empty_quote_buffers(1), year=y, times_ns=times,
                           prices=broken, row_offset=offset, cutoff_ns=TRAIN_END_NS)


def test_target_at_cutoff_is_not_joined():
    saved, years = sample()
    saved["terminal_time_ns"][0, 2] = TRAIN_END_NS
    result = join(saved, years)
    assert not result["terminal"]["observed"][0, 2]
    assert result["terminal"]["source_row"][0, 2] == -1


def test_unsorted_quote_clock_is_rejected():
    saved, years = sample()
    y, times, prices, offset = years[0]
    with pytest.raises(RuntimeError, match="YEAR_CLOCK"):
        add_year_partition(saved, empty_quote_buffers(1), year=y, times_ns=times[::-1],
                           prices=prices, row_offset=offset, cutoff_ns=TRAIN_END_NS)
