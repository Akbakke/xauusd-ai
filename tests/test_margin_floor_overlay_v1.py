"""Parity tests for the margin-floor entry overlay (apply_margin_floor_overlay).

The overlay is a default-OFF, byte-identical serve-time selection filter (GX1_ENTRY_MARGIN_FLOOR):
it SKIPs a final TAKE whose V10 conviction `margin` is below a floor, NEVER converts a SKIP into a
TAKE, and is a pure no-op when the flag is unset. It is ONE TRUTH with the OOT gate
(v12_phase1_entry_iql_inference imports the SAME helper) so gate==serve. These tests lock the
invariants the validation gate (GX1_DATA/MARGIN_FLOOR_GATE_SPEC_20260623.md) relies on.
"""
from __future__ import annotations

import pytest

import gx1.execution.v12_entry_iql_live as live
from gx1.scripts import entry_iql_multi_head_gpu_core_v1 as iql_core

SKIP = iql_core.ACTION_SKIP_ID
LONG = iql_core.ACTION_TAKE_LONG_NOW_ID
SHORT = iql_core.ACTION_TAKE_SHORT_NOW_ID


@pytest.fixture
def floor_on(monkeypatch):
    """Arm the overlay at THR=0.47 (the spec's recommended moderate floor) for the duration of a test."""
    monkeypatch.setattr(live, "MARGIN_FLOOR_ON", True)
    monkeypatch.setattr(live, "MARGIN_FLOOR_THR", 0.47)
    yield


def test_default_off_is_byte_identical_noop(monkeypatch):
    """Flag OFF ⇒ pure no-op for EVERY (action, margin) pair, incl. missing margin → live byte-unchanged."""
    monkeypatch.setattr(live, "MARGIN_FLOOR_ON", False)
    for a in (SKIP, LONG, SHORT):
        for m in (0.0, 0.46, 0.47, 0.99, None):
            cand = {} if m is None else {"margin": m}
            nid, mk = live.apply_margin_floor_overlay(a, cand)
            assert nid == a and mk == {}


def test_on_skips_low_margin_take(floor_on):
    """A low-conviction TAKE (margin < THR) is converted to SKIP, with an attributable marker."""
    for a in (LONG, SHORT):
        nid, mk = live.apply_margin_floor_overlay(a, {"margin": 0.30})
        assert nid == SKIP
        assert mk["margin_floor_applied"] is True
        assert mk["margin_floor_margin"] == pytest.approx(0.30)
        assert mk["margin_floor_thr"] == pytest.approx(0.47)
        assert mk["margin_floor_from"] == iql_core.ACTION_LABELS_V1[a]


def test_on_keeps_high_margin_take(floor_on):
    """A high-conviction TAKE (margin >= THR) is untouched (no-op marker)."""
    for a in (LONG, SHORT):
        nid, mk = live.apply_margin_floor_overlay(a, {"margin": 0.60})
        assert nid == a and mk == {}


def test_on_boundary_is_inclusive(floor_on):
    """margin == THR is KEPT (>= passes); just-below is skipped (no off-by-one)."""
    nid, mk = live.apply_margin_floor_overlay(LONG, {"margin": 0.47})
    assert nid == LONG and mk == {}
    nid, mk = live.apply_margin_floor_overlay(LONG, {"margin": 0.4699})
    assert nid == SKIP


def test_on_never_converts_skip_to_take(floor_on):
    """The cardinal invariant: the overlay only ever removes TAKEs — a SKIP stays a SKIP at any margin."""
    for m in (0.0, 0.30, 0.60, 0.99):
        nid, mk = live.apply_margin_floor_overlay(SKIP, {"margin": m})
        assert nid == SKIP and mk == {}


def test_on_missing_margin_is_failopen_noop(floor_on):
    """Missing margin ⇒ keep the trade (never silent-drop a TAKE); it warns-once but returns a no-op."""
    nid, mk = live.apply_margin_floor_overlay(LONG, {})
    assert nid == LONG and mk == {}


def test_on_unparseable_margin_is_failopen(floor_on):
    """Unparseable margin ⇒ fail-open (keep the trade), same conservative side as missing."""
    nid, mk = live.apply_margin_floor_overlay(LONG, {"margin": "n/a"})
    assert nid == LONG and mk == {}
