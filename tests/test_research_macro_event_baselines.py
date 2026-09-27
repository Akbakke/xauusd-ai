"""Mechanics of the preregistered macro-event cells (synthetic tapes prove only that code runs)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gx1.scripts.research_entry_direction_walkforward_v1 import Tape
from gx1.scripts import research_macro_event_baselines_v1 as events


def _tape(start: str, bars: int, jump_at: pd.Timestamp, jump: float) -> Tape:
    time = pd.date_range(start, periods=bars, freq="5min", tz="UTC")
    mid = np.full(bars, 100.0) + np.where(time >= jump_at, jump, 0.0)
    return Tape(time=time, mid=mid, bid=mid - 0.1, ask=mid + 0.1, manifest_sha256="0" * 64, root="synthetic")


def test_event_cells_use_the_registered_bars_and_signal():
    release = pd.Timestamp("2015-01-09 13:30", tz="UTC")
    tape = _tape("2015-01-08 12:00", 12 * 30, release, 1.0)
    out = events.event_trades(tape, pd.DatetimeIndex([release]))
    cells = out["cells"]
    t = tape.time
    assert t[cells["pre_long"]["entry"][0]] == pd.Timestamp("2015-01-08 13:30", tz="UTC")
    assert t[cells["pre_long"]["exit"][0]] == release - pd.Timedelta(minutes=5)
    assert cells["pre_short"]["side"].tolist() == [-1]
    assert t[cells["post_mom_h12"]["entry"][0]] == release + pd.Timedelta(minutes=10)
    assert t[cells["post_mom_h48"]["exit"][0]] == release + pd.Timedelta(minutes=10 + 240)
    assert cells["post_mom_h12"]["side"].tolist() == [1] and cells["post_rev_h12"]["side"].tolist() == [-1]
    assert out["skipped"] == {}


def test_event_is_skipped_when_a_required_bar_is_missing():
    release = pd.Timestamp("2015-01-09 13:30", tz="UTC")
    tape = _tape("2015-01-09 12:00", 40, release, 1.0)  # no bar 24 h earlier, no 4 h exit
    out = events.event_trades(tape, pd.DatetimeIndex([release]))
    assert "pre_long" not in out["cells"] and "post_mom_h48" not in out["cells"]
    assert out["skipped"]["pre"] == 1 and out["skipped"]["post_h48"] == 1
    assert "post_mom_h12" in out["cells"]


def test_flat_first_fifteen_minutes_trade_nothing():
    release = pd.Timestamp("2015-01-09 13:30", tz="UTC")
    tape = _tape("2015-01-08 12:00", 12 * 30, release, 0.0)
    out = events.event_trades(tape, pd.DatetimeIndex([release]))
    assert not any(name.startswith("post_") for name in out["cells"])
    assert out["skipped"]["post_h12"] == 1


def test_verdict_threshold_is_the_registration():
    assert events.verdict({"t": 2.8, "positive_full_year_share": 0.6}) == "GO"
    assert events.verdict({"t": 2.1, "positive_full_year_share": 0.6}) == "LOVENDE"
    assert events.verdict({"t": 3.5, "positive_full_year_share": 0.5}) == "NO_GO"
