"""Golden parity test for the one-truth circuit-breaker (2026-06-04).

Proves gx1.portfolio.circuit_breaker_v1 reproduces the LIVE admission logic
byte-for-byte, so the live sites can delegate to it (one truth) and the offline
harnesses can call the SAME chokepoint — making "live == Phase-6 when no breaker
fires" provable. Pins the exact reason strings (journals + Phase-6 gate diff on them).
"""
from __future__ import annotations

import itertools

import pandas as pd
import pytest

from gx1.portfolio import circuit_breaker_v1 as cb


# ── evaluate_same_opp_cap == live evaluate_entry_safety (the B6 delegation target) ──

def test_same_opp_cap_matches_live_evaluate_entry_safety():
    try:
        from gx1.execution.v12_paper_runner import evaluate_entry_safety as live_fn
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"cannot import live evaluate_entry_safety: {e}")
    sides = ("long", "short")
    for side, n_same, n_opp, hard_max in itertools.product(
        sides, range(0, 5), range(0, 4), (1, 2, 3)
    ):
        assert cb.evaluate_same_opp_cap(side, n_same, n_opp, hard_max) == \
            live_fn(side, n_same, n_opp, hard_max), \
            f"parity break: side={side} n_same={n_same} n_opp={n_opp} hard_max={hard_max}"


# ── evaluate_entry_admission: ordering + each breaker branch ──

CFG = cb.BreakerCfg(hard_max_same_side=2, drawdown_block_bps=-100.0, cluster1_enabled=True)
M5 = pd.Timestamp("2026-01-06 12:00:00")


def test_admission_ok_on_empty_book():
    ok, reason, _ = cb.evaluate_entry_admission(cb.OpenBook(), "long", M5, CFG)
    assert ok and reason == cb.REASON_OK


def test_admission_opposing_block():
    book = cb.OpenBook(n_long=0, n_short=1)
    ok, reason, detail = cb.evaluate_entry_admission(book, "long", M5, CFG)
    assert not ok and reason == cb.REASON_OPPOSING and detail == 1


def test_admission_same_side_cap():
    book = cb.OpenBook(n_long=2)
    ok, reason, detail = cb.evaluate_entry_admission(book, "long", M5, CFG)
    assert not ok and reason == cb.REASON_SAME_SIDE_CAP and detail == 2


def test_admission_drawdown_block():
    book = cb.OpenBook(n_long=1, combined_unrealized_bps=-150.0)
    ok, reason, detail = cb.evaluate_entry_admission(book, "long", M5, CFG)
    assert not ok and reason == cb.REASON_DRAWDOWN and detail == -150.0


def test_admission_cluster1_same_side():
    book = cb.OpenBook(last_entry_m5_by_side={"long": M5})
    ok, reason, _ = cb.evaluate_entry_admission(book, "long", M5, CFG)
    assert not ok and reason == cb.REASON_CLUSTER1_SAME


def test_admission_cluster1_opposite_side():
    book = cb.OpenBook(last_entry_m5_by_side={"short": M5})
    ok, reason, _ = cb.evaluate_entry_admission(book, "long", M5, CFG)
    assert not ok and reason == cb.REASON_CLUSTER1_OPP


def test_admission_cluster1_precedes_other_breakers():
    # CLUSTER1 fires first in live (inside the pipeline) even if cap/drawdown would also fire.
    book = cb.OpenBook(n_long=2, combined_unrealized_bps=-500.0,
                       last_entry_m5_by_side={"long": M5})
    ok, reason, _ = cb.evaluate_entry_admission(book, "long", M5, CFG)
    assert not ok and reason == cb.REASON_CLUSTER1_SAME


def test_admission_cluster1_disabled():
    cfg = cb.BreakerCfg(cluster1_enabled=False)
    book = cb.OpenBook(last_entry_m5_by_side={"long": M5})
    ok, reason, _ = cb.evaluate_entry_admission(book, "long", M5, cfg)
    assert ok and reason == cb.REASON_OK


def test_admission_cluster1_different_m5_allowed():
    book = cb.OpenBook(last_entry_m5_by_side={"long": M5})
    other = pd.Timestamp("2026-01-06 12:05:00")
    ok, reason, _ = cb.evaluate_entry_admission(book, "long", other, CFG)
    assert ok and reason == cb.REASON_OK


def test_admission_rejects_bad_side():
    with pytest.raises(ValueError):
        cb.evaluate_entry_admission(cb.OpenBook(), "flat", M5, CFG)


# ── book_from_open_trades ──

class _FakeTrade:
    def __init__(self, side, pnl, ts):
        self.side = side
        self.current_pnl_bps = pnl
        self.entry_ts = pd.Timestamp(ts)


def test_book_from_open_trades():
    trades = [
        _FakeTrade("long", 10.0, "2026-01-06 11:00"),
        _FakeTrade("long", -5.0, "2026-01-06 11:30"),
        _FakeTrade("short", 3.0, "2026-01-06 11:15"),
    ]
    book = cb.book_from_open_trades(trades)
    assert book.n_long == 2 and book.n_short == 1
    assert book.combined_unrealized_bps == pytest.approx(8.0)
    # most-recent same-side entry recorded
    assert book.last_entry_m5_by_side["long"] == pd.Timestamp("2026-01-06 11:30")
    assert book.last_entry_m5_by_side["short"] == pd.Timestamp("2026-01-06 11:15")


def test_book_from_open_trades_empty():
    assert cb.book_from_open_trades(None) == cb.OpenBook()
    assert cb.book_from_open_trades([]) == cb.OpenBook()
