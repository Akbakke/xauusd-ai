"""Contract tests for the V29 Phase A level registry owner.

Covers: causality (future-row invariance), touch dedup, break-once semantics,
reaction accounting (signed),
chunk-carry bit-exactness, name-tuple drift guards, absence/expiry encoding,
retest hold/fail/none lifecycle, pivot one-truth vs smc_v1, and the
fail-closed input/state/kind validation standard.

All series here are synthetic: per rule 2c they prove code properties
(algebraic identities, causal invariances, declared contract shapes), never
production behaviour.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.features.level_registry_v1 import (
    LEVEL_KIND_PIVOT_ANCHOR,
    LEVEL_KIND_SESSION_ANCHORED,
    LEVEL_KINDS,
    LEVEL_KINDS_IMPLEMENTED_V1,
    LEVEL_REGISTRY_M5_FEATURE_NAMES,
    LEVEL_REGISTRY_MTF_FEATURE_NAMES,
    LEVEL_REGISTRY_STATE_KEYS,
    collect_level_registry_runtime_population_v1 as _collect_runtime,
    compute_level_registry_m5_block_v1 as _compute_m5,
    compute_level_registry_mtf_block_v1 as _compute_mtf,
    fit_level_registry_hyperparameters_v1,
    require_level_kind_implemented,
)
from gx1.features.smc_v1 import SWING_LOOKBACK, _detect_swing_pivots


TOL = 0.5  # explicit test input for the engine (production uses the TRAIN fit)
SHIFT = 4000.0  # XAU-magnitude translation; pivot/zone geometry is invariant
TEST_MAX_EVIDENCE_AGE_BARS = 240


def compute_level_registry_m5_block_v1(df, **kwargs):
    kwargs.setdefault(
        "max_evidence_age_bars", TEST_MAX_EVIDENCE_AGE_BARS
    )
    return _compute_m5(df, **kwargs)


def compute_level_registry_mtf_block_v1(df, *, tf, **kwargs):
    kwargs.setdefault("max_evidence_age_bars", TEST_MAX_EVIDENCE_AGE_BARS)
    return _compute_mtf(df, tf=tf, **kwargs)


def collect_level_registry_runtime_population_v1(df, *, tf, **kwargs):
    kwargs.setdefault(
        "max_evidence_age_bars", TEST_MAX_EVIDENCE_AGE_BARS
    )
    return _collect_runtime(df, tf=tf, **kwargs)


def _mk_df(high, low, close, atr=None) -> pd.DataFrame:
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    if atr is None:
        atr = np.ones_like(high)
    return pd.DataFrame({"high": high, "low": low, "close": close, "atr": np.asarray(atr, dtype=np.float64)})


def _fit_source(tmp_path: Path, *, clock: str) -> dict:
    paths = {}
    for name in ("source", "tape", "split"):
        path = (tmp_path / f"{name}.json").resolve()
        path.write_text(json.dumps({"name": name}) + "\n", encoding="utf-8")
        paths[name] = path
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "source_artifact": str(paths["source"]),
        "source_sha256": digest(paths["source"]),
        "source_schema_version": "synthetic_closed_ohlcv_v1",
        "source_lane": clock,
        "tape_manifest_artifact": str(paths["tape"]),
        "tape_manifest_sha256": digest(paths["tape"]),
        "split_manifest_artifact": str(paths["split"]),
        "split_manifest_sha256": digest(paths["split"]),
        "train_split_id": "synthetic:TRAIN",
        "declared_train_window_start": "2020-01-01T00:00:00+00:00",
        "declared_train_window_end": "2020-01-03T10:15:00+00:00",
    }


def _col(matrix: np.ndarray, names: list[str], name: str) -> np.ndarray:
    return matrix[:, names.index(name)]


def _series_s1() -> pd.DataFrame:
    """Stable resistance anchors with same-side recurrence observations."""
    n = 28
    high = SHIFT + 10.0 + 0.01 * np.arange(n)
    high[5] = SHIFT + 12.0
    high[12:16] = SHIFT + np.array([11.6, 11.65, 11.7, 11.75])
    high[16:20] = SHIFT + np.array([11.4, 11.35, 11.3, 11.25])
    high[20] = SHIFT + 11.55
    high[21:] = SHIFT + np.array([11.0, 10.9, 10.8, 10.7, 10.6, 10.5, 10.4])
    low = SHIFT + 9.0 - 0.01 * np.arange(n)
    close = SHIFT + 9.5 - 0.005 * np.arange(n)
    return _mk_df(high, low, close)


def _series_s2() -> pd.DataFrame:
    """Support level (pivot j=5, center 7+SHIFT): touch, break down, retest hold,
    one positive and one negative completed reaction."""
    n = 30
    idx = np.arange(n)
    high = SHIFT + 14.0 + 0.01 * idx
    low = SHIFT + 9.0 - 0.01 * idx
    low[5] = SHIFT + 7.0
    close = low + 0.3
    high[12], low[12], close[12] = SHIFT + 7.6, SHIFT + 7.4, SHIFT + 7.5
    tail = idx[13:]
    high[13:] = SHIFT + 6.9 - 0.1 * (tail - 13)
    low[13:] = SHIFT + 6.5 - 0.1 * (tail - 13)
    close[13:] = low[13:] + 0.05
    close[13] = SHIFT + 6.6
    close[14] = SHIFT + 6.45  # first close below zone edge 6.5 -> break at t=14
    return _mk_df(high, low, close)


def _series_alternating_anchors(n: int = 96) -> pd.DataFrame:
    """Stable alternating extrema: later closes never cross prior wick anchors."""

    t = np.arange(n, dtype=np.float64)
    close = SHIFT + 100.0 + 3.0 * np.sin(2.0 * np.pi * t / 12.0)
    return _mk_df(close + 0.4, close - 0.4, close)


def _series_s3() -> pd.DataFrame:
    """S2 variant: after the down-break at t=14 the retest at t=15 FAILS
    (close back above center)."""
    df = _series_s2()
    high = df["high"].to_numpy().copy()
    low = df["low"].to_numpy().copy()
    close = df["close"].to_numpy().copy()
    high[15], low[15], close[15] = SHIFT + 7.3, SHIFT + 6.5, SHIFT + 7.2
    for i in range(16, len(high)):
        high[i] = SHIFT + 7.25 - 0.05 * (i - 16)
        low[i] = SHIFT + 6.6 - 0.05 * (i - 16)
        close[i] = 0.5 * (high[i] + low[i])
    return _mk_df(high, low, close)


def _series_s4_no_reentry() -> pd.DataFrame:
    """Support break at t=13 with no exact-center retest before tape end."""
    n = 42
    idx = np.arange(n)
    high = SHIFT + 14.0 + 0.01 * idx
    low = SHIFT + 9.0 - 0.01 * idx
    low[5] = SHIFT + 7.0
    close = low + 0.3
    high[12], low[12], close[12] = SHIFT + 7.6, SHIFT + 7.4, SHIFT + 7.5
    tail = idx[13:]
    high[13:] = SHIFT + 6.4 - 0.1 * (tail - 13)
    low[13:] = SHIFT + 6.0 - 0.1 * (tail - 13)
    close[13:] = 0.5 * (high[13:] + low[13:])
    return _mk_df(high, low, close)


def _series_expiry() -> pd.DataFrame:
    """One resistance level admitted at t=8 and never touched again: expiry at
    t - last_touch_bar(=5) > AGE_CAP['m5'] = 240, i.e. removal at t=246."""
    n = 260
    idx = np.arange(n)
    high = SHIFT + 10.0 + 0.001 * idx
    high[5] = SHIFT + 12.0
    low = SHIFT + 9.0 - 0.001 * idx
    close = SHIFT + 9.3 - 0.0005 * idx
    return _mk_df(high, low, close)


def _rng_df(n: int = 200, seed: int = 20260811) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mid = SHIFT + np.cumsum(rng.normal(0.0, 1.0, n))
    hi_off = np.abs(rng.normal(0.0, 1.0, n)) + 0.1
    lo_off = np.abs(rng.normal(0.0, 1.0, n)) + 0.1
    high = mid + hi_off
    low = mid - lo_off
    close = low + rng.uniform(0.0, 1.0, n) * (high - low)
    atr = 1.0 + np.abs(rng.normal(0.0, 0.2, n))
    return _mk_df(high, low, close, atr)


# ---------------------------------------------------------------------------
# Declared contracts / drift guards
# ---------------------------------------------------------------------------

def test_declared_name_tuples_match_design_doc_verbatim():
    # Raw uncapped measurements use explicit slot/break masks. Completed
    # reaction counts disambiguate real zero reactions from no reaction.
    assert LEVEL_REGISTRY_M5_FEATURE_NAMES == (
        "level_above_dist_atr",
        "level_above_present",
        "level_above2_dist_atr",
        "level_above2_present",
        "level_above_touch_count",
        "level_above_completed_reaction_count",
        "level_above_recurrence_confirmed",
        "level_above_age_bars",
        "level_above_bars_since_touch",
        "level_above_mean_reaction_atr",
        "level_above_max_reaction_atr",
        "level_above_last_reaction_atr",
        "level_below_dist_atr",
        "level_below_present",
        "level_below2_dist_atr",
        "level_below2_present",
        "level_below_touch_count",
        "level_below_completed_reaction_count",
        "level_below_recurrence_confirmed",
        "level_below_age_bars",
        "level_below_bars_since_touch",
        "level_below_mean_reaction_atr",
        "level_below_max_reaction_atr",
        "level_below_last_reaction_atr",
        "level_break_up_event",
        "level_break_down_event",
        "level_broken_touch_count",
        "level_bars_since_break",
        "level_bars_since_break_signed",
        "level_retest_hold_signed",
        "level_retest_fail_signed",
    )
    assert LEVEL_REGISTRY_MTF_FEATURE_NAMES == tuple(
        f"mtf_{name}" for name in LEVEL_REGISTRY_M5_FEATURE_NAMES
    )
    assert len(LEVEL_REGISTRY_M5_FEATURE_NAMES) == 31
    assert len(LEVEL_REGISTRY_MTF_FEATURE_NAMES) == 31
    assert all(n.startswith("level_") for n in LEVEL_REGISTRY_M5_FEATURE_NAMES)
    assert all(n.startswith("mtf_level_") for n in LEVEL_REGISTRY_MTF_FEATURE_NAMES)


def test_emitted_names_match_declared_tuples():
    df = _series_s1()
    m5, names_m5 = compute_level_registry_m5_block_v1(df, recurrence_threshold_atr=TOL)
    assert tuple(names_m5) == LEVEL_REGISTRY_M5_FEATURE_NAMES
    assert m5.shape == (len(df), len(LEVEL_REGISTRY_M5_FEATURE_NAMES))
    assert m5.dtype == np.float32
    mtf, names_mtf = compute_level_registry_mtf_block_v1(df, tf="m5", recurrence_threshold_atr=TOL)
    assert tuple(names_mtf) == LEVEL_REGISTRY_MTF_FEATURE_NAMES
    assert mtf.shape == (len(df), len(LEVEL_REGISTRY_MTF_FEATURE_NAMES))
    np.testing.assert_array_equal(mtf, m5, strict=True)
    assert tuple(name.removeprefix("mtf_") for name in names_mtf) == tuple(
        names_m5
    )


def test_kind_enum_phase_a_boundary():
    assert LEVEL_KINDS == ("pivot_anchor", "session_anchored")
    assert LEVEL_KINDS_IMPLEMENTED_V1 == ("pivot_anchor",)
    assert require_level_kind_implemented(LEVEL_KIND_PIVOT_ANCHOR) == "pivot_anchor"
    with pytest.raises(RuntimeError, match="KIND_NOT_IMPLEMENTED"):
        require_level_kind_implemented(LEVEL_KIND_SESSION_ANCHORED)
    with pytest.raises(RuntimeError, match="KIND_UNKNOWN"):
        require_level_kind_implemented("bogus")


def test_mtf_block_values_equal_m5_registry_fields_on_same_clock():
    df = _rng_df()
    m5, names_m5 = compute_level_registry_m5_block_v1(df, recurrence_threshold_atr=0.6)
    mtf, names_mtf = compute_level_registry_mtf_block_v1(df, tf="m5", recurrence_threshold_atr=0.6)
    pairs = {
        "mtf_level_above_dist_atr": "level_above_dist_atr",
        "mtf_level_below_dist_atr": "level_below_dist_atr",
        "mtf_level_above_touch_count": "level_above_touch_count",
        "mtf_level_below_touch_count": "level_below_touch_count",
        "mtf_level_above_mean_reaction_atr": "level_above_mean_reaction_atr",
        "mtf_level_below_mean_reaction_atr": "level_below_mean_reaction_atr",
        "mtf_level_break_up_event": "level_break_up_event",
        "mtf_level_break_down_event": "level_break_down_event",
        "mtf_level_bars_since_break": "level_bars_since_break",
        "mtf_level_retest_hold_signed": "level_retest_hold_signed",
        "mtf_level_retest_fail_signed": "level_retest_fail_signed",
    }
    for mtf_name, m5_name in pairs.items():
        np.testing.assert_array_equal(
            _col(mtf, names_mtf, mtf_name), _col(m5, names_m5, m5_name)
        )


# ---------------------------------------------------------------------------
# S1: warmup, creation, touch dedup, merges, positive reaction accounting
# ---------------------------------------------------------------------------

def test_s1_warmup_prefix_and_creation():
    df = _series_s1()
    m5, names = compute_level_registry_m5_block_v1(df, recurrence_threshold_atr=TOL)
    assert np.isnan(m5[:8]).all()          # NaN until first admitted level (t=8)
    break_age_columns = [
        names.index("level_bars_since_break"),
        names.index("level_bars_since_break_signed"),
    ]
    assert np.isfinite(np.delete(m5[8:], break_age_columns, axis=1)).all()
    dist = _col(m5, names, "level_above_dist_atr")
    touch = _col(m5, names, "level_above_touch_count")
    age = _col(m5, names, "level_above_age_bars")
    bst = _col(m5, names, "level_above_bars_since_touch")
    assert dist[8] == pytest.approx(12.0 - (9.5 - 0.04), abs=1e-3)  # 2.54
    assert touch[8] == 1.0
    assert age[8] == 0.0
    assert bst[8] == 3.0  # creation sets last_touch_bar = pivot bar j = 5


def test_s1_all_immutable_anchors_and_recurrence_accounting():
    df = _series_s1()
    m5, names, state = compute_level_registry_m5_block_v1(
        df, recurrence_threshold_atr=TOL, return_registry_state=True
    )
    touch = _col(m5, names, "level_above_touch_count")
    bst = _col(m5, names, "level_above_bars_since_touch")
    # Canonical identity uses exact-center touches. Every confirmed pivot is a
    # new immutable anchor; public nearest-two slots never delete older state.
    assert touch[11] == 1.0
    assert (touch[12:18] == 1.0).all()
    assert touch[18] == 1.0
    assert touch[20] == 1.0
    assert touch[23] == 1.0
    assert bst[14] == 9.0   # raw last-touch pivot remains j=5
    assert bst[18] == 3.0
    assert [lv["center_price"] for lv in state["levels"]] == pytest.approx(
        [SHIFT + 12.0, SHIFT + 11.75, SHIFT + 11.55]
    )
    assert [lv["member_pivot_bars"] for lv in state["levels"]] == [[5], [15], [20]]
    assert [lv["recurrence_confirmed"] for lv in state["levels"]] == [0, 1, 1]
    # no breaks anywhere in S1
    assert _col(m5, names, "level_break_up_event")[8:].sum() == 0.0
    assert _col(m5, names, "level_break_down_event")[8:].sum() == 0.0
    assert np.isnan(_col(m5, names, "level_bars_since_break")[8:]).all()


def test_s1_reaction_accounting_and_event_gated_zero():
    df = _series_s1()
    m5, names = compute_level_registry_m5_block_v1(df, recurrence_threshold_atr=TOL)
    mean_r = _col(m5, names, "level_above_mean_reaction_atr")
    max_r = _col(m5, names, "level_above_max_reaction_atr")
    last_r = _col(m5, names, "level_above_last_reaction_atr")
    # Recurrence never closes or drifts an anchor. The oldest anchor's
    # supersession outcome remains in the canonical event tape, while the two
    # emitted live anchors have no completed reaction yet.
    assert (mean_r[8:] == 0.0).all()
    assert (max_r[8:] == 0.0).all()
    assert (last_r[8:] == 0.0).all()


def test_alternating_high_low_pivots_keep_distinct_stable_anchor_identities():
    df = _series_alternating_anchors()
    lifecycle: list[tuple] = []
    matrix, names, state = compute_level_registry_m5_block_v1(
        df,
        recurrence_threshold_atr=TOL,
        return_registry_state=True,
        runtime_lifecycle_log=lifecycle,
    )
    creates = [event for event in lifecycle if event[0] == "create"]
    assert len(creates) >= 12
    assert {event[3] for event in creates} == {"high_pivot", "low_pivot"}
    assert len({event[4] for event in creates}) == len(creates)
    assert not ({"merge", "supersede"} & {event[0] for event in lifecycle})

    created = {int(event[4]): event for event in creates}
    assert len(state["levels"]) == len(creates)
    for level in state["levels"]:
        event = created[int(level["level_id"])]
        pivot_bar = int(event[2])
        source = "high" if event[3] == "high_pivot" else "low"
        assert level["birth_side"] == event[3]
        assert level["center_price"] == event[5] == pytest.approx(
            float(df[source].iloc[pivot_bar])
        )

    recurrence_by_new_id = {
        int(event[4]): float(event[5])
        for event in lifecycle
        if event[0] == "recurrence"
    }
    for level in state["levels"]:
        expected = int(
            int(level["level_id"]) in recurrence_by_new_id
            and recurrence_by_new_id[int(level["level_id"])] <= TOL
        )
        assert level["recurrence_confirmed"] == expected

    active = [level for level in state["levels"] if level["status"] == "active"]
    assert sum(level["side_of_origin"] == "high_pivot" for level in active) >= 5
    assert sum(level["side_of_origin"] == "low_pivot" for level in active) >= 5
    close = float(df["close"].iloc[-1])
    above = sorted(
        (level["center_price"] - close, level["level_id"])
        for level in active
        if level["center_price"] > close
    )
    below = sorted(
        (close - level["center_price"], level["level_id"])
        for level in active
        if level["center_price"] <= close
    )
    assert _col(matrix, names, "level_above_dist_atr")[-1] == pytest.approx(
        above[0][0]
    )
    assert _col(matrix, names, "level_above2_dist_atr")[-1] == pytest.approx(
        above[1][0]
    )
    assert _col(matrix, names, "level_below_dist_atr")[-1] == pytest.approx(
        below[0][0]
    )
    assert _col(matrix, names, "level_below2_dist_atr")[-1] == pytest.approx(
        below[1][0]
    )


def test_s1_absent_side_encoding():
    df = _series_s1()
    m5, names = compute_level_registry_m5_block_v1(df, recurrence_threshold_atr=TOL)
    # no low-side level ever exists: absent-side encoding
    assert (_col(m5, names, "level_below_dist_atr")[8:] == 0.0).all()
    assert (_col(m5, names, "level_below_present")[8:] == 0.0).all()
    assert (_col(m5, names, "level_below_touch_count")[8:] == 0.0).all()
    assert (_col(m5, names, "level_below_age_bars")[8:] == 0.0).all()
    assert (_col(m5, names, "level_below_bars_since_touch")[8:] == 0.0).all()
    assert (_col(m5, names, "level_below_mean_reaction_atr")[8:] == 0.0).all()


# ---------------------------------------------------------------------------
# S2/S3/S4: break-once, retest hold/fail/none, signed (negative) reactions
# ---------------------------------------------------------------------------

def test_s2_break_once_retest_hold_and_signed_reactions():
    df = _series_s2()
    m5, names, state = compute_level_registry_m5_block_v1(
        df, recurrence_threshold_atr=TOL, return_registry_state=True
    )
    down = _col(m5, names, "level_break_down_event")
    up = _col(m5, names, "level_break_up_event")
    btc = _col(m5, names, "level_broken_touch_count")
    bsb = _col(m5, names, "level_bars_since_break")
    hold = _col(m5, names, "level_retest_hold_signed")
    fail = _col(m5, names, "level_retest_fail_signed")
    below_dist = _col(m5, names, "level_below_dist_atr")
    # Exact-center break occurs at t=13; no tolerance alters identity timing.
    assert down[13] == 1.0
    assert down[8:].sum() == 1.0
    assert up[8:].sum() == 0.0
    assert btc[13] == 1.0 and btc[8:].sum() == 1.0
    assert np.isnan(bsb[8:13]).all()
    assert bsb[13] == 0.0 and bsb[14] == 1.0
    # broken level leaves the below slot at the break bar
    assert below_dist[9] == pytest.approx((9.0 - 0.09 + 0.3) - 7.0, abs=1e-3)  # 2.21
    assert below_dist[13] == 0.0
    assert _col(m5, names, "level_below_present")[13] == 0.0
    # No later bar intersects the exact center, so the raw retest stays armed.
    assert hold[8:].sum() == 0.0
    assert fail[8:].sum() == 0.0
    lv = next(l for l in state["levels"] if l["break_bar"] == 13)
    assert lv["status"] == "broken" and lv["retest_state"] == "pending"


def test_s3_retest_fail_signed_carries_break_direction():
    df = _series_s3()
    m5, names, state = compute_level_registry_m5_block_v1(
        df, recurrence_threshold_atr=TOL, return_registry_state=True
    )
    hold = _col(m5, names, "level_retest_hold_signed")
    fail = _col(m5, names, "level_retest_fail_signed")
    assert fail[15] == -1.0          # sign = break direction (down)
    assert fail[8:].sum() == -1.0
    # A later, distinct low-pivot anchor breaks and holds its own retest at 28.
    assert hold[8:].sum() == -1.0
    # Failed retest is terminal for eligibility but the immutable identity is
    # retained in the state archive.
    failed = [lv for lv in state["levels"] if lv["retest_state"] == "failed"]
    assert len(failed) == 1 and failed[0]["break_side"] == -1


def test_s4_retest_stays_pending_past_old_twenty_four_bar_window():
    df = _series_s4_no_reentry()
    m5, names, state = compute_level_registry_m5_block_v1(
        df, recurrence_threshold_atr=TOL, return_registry_state=True
    )
    assert _col(m5, names, "level_break_down_event")[8:].sum() == 1.0
    assert _col(m5, names, "level_retest_hold_signed")[8:].sum() == 0.0
    assert _col(m5, names, "level_retest_fail_signed")[8:].sum() == 0.0
    lv = next(l for l in state["levels"] if l["break_side"] == -1)
    assert lv["retest_state"] == "pending"
    assert len(df) - 1 - lv["break_bar"] > 24


def test_learned_lifetime_expires_eligibility_without_deleting_identity():
    df = _series_expiry()
    m5, names, state = compute_level_registry_m5_block_v1(
        df, recurrence_threshold_atr=TOL, return_registry_state=True
    )
    dist = _col(m5, names, "level_above_dist_atr")
    bst = _col(m5, names, "level_above_bars_since_touch")
    age = _col(m5, names, "level_above_age_bars")
    assert dist[245] == pytest.approx(12.0 - (9.3 - 0.0005 * 245), abs=1e-3)
    assert bst[245] == 240.0
    assert age[245] == 237.0
    # The selected lifetime removes event/slot eligibility, while the exact
    # immutable anchor survives in the internal identity archive.
    assert dist[248] != 0.0 and age[248] == 240.0 and bst[248] == 243.0
    assert dist[249] == 0.0 and age[249] == 0.0 and bst[249] == 0.0
    assert _col(m5, names, "level_above_present")[249] == 0.0
    assert _col(m5, names, "level_above_touch_count")[249] == 0.0
    expired = [lv for lv in state["levels"] if lv["status"] == "expired"]
    assert len(expired) == 1 and expired[0]["center_price"] == SHIFT + 12.0
    break_age_columns = [
        names.index("level_bars_since_break"),
        names.index("level_bars_since_break_signed"),
    ]
    assert np.isfinite(np.delete(m5[8:], break_age_columns, axis=1)).all()


# ---------------------------------------------------------------------------
# Causality, chunk-carry, one pivot truth
# ---------------------------------------------------------------------------

def test_causality_future_row_invariance():
    df = _rng_df(n=200)
    full, _ = compute_level_registry_m5_block_v1(df, recurrence_threshold_atr=0.6)
    for k in (23, 61, 137, 190):
        prefix, _ = compute_level_registry_m5_block_v1(df.iloc[:k], recurrence_threshold_atr=0.6)
        np.testing.assert_array_equal(full[:k], prefix)


def test_runtime_shadow_population_is_serve_identical_and_prefix_causal():
    df = _rng_df(n=400, seed=19)
    tol = TOL

    shadow_population, shadow_lifecycle = (
        collect_level_registry_runtime_population_v1(
            df,
            tf="m5",
            recurrence_threshold_atr=tol,
        )
    )
    serve_population: list[float] = []
    serve_lifecycle: list[tuple] = []
    observed, _names = compute_level_registry_m5_block_v1(
        df,
        recurrence_threshold_atr=tol,
        runtime_admission_distance_log=serve_population,
        runtime_lifecycle_log=serve_lifecycle,
    )
    baseline, _names = compute_level_registry_m5_block_v1(
        df,
        recurrence_threshold_atr=tol,
    )
    np.testing.assert_array_equal(observed, baseline, strict=True)
    np.testing.assert_array_equal(
        shadow_population,
        np.asarray(serve_population, dtype=np.float64),
        strict=True,
    )
    assert shadow_lifecycle == serve_lifecycle
    assert shadow_population.size > 0

    prefix_population, prefix_lifecycle = (
        collect_level_registry_runtime_population_v1(
            df.iloc[:300],
            tf="m5",
            recurrence_threshold_atr=tol,
        )
    )
    np.testing.assert_array_equal(
        prefix_population,
        shadow_population[: len(prefix_population)],
        strict=True,
    )
    assert prefix_lifecycle == shadow_lifecycle[: len(prefix_lifecycle)]


def test_expiry_changes_only_eligibility_not_identity_or_recurrence_population():
    df = _rng_df(n=500, seed=119)
    short_population, short_lifecycle = _collect_runtime(
        df,
        tf="m5",
        recurrence_threshold_atr=TOL,
        max_evidence_age_bars=7,
    )
    long_population, long_lifecycle = _collect_runtime(
        df,
        tf="m5",
        recurrence_threshold_atr=TOL,
        max_evidence_age_bars=200,
    )
    np.testing.assert_array_equal(short_population, long_population, strict=True)
    short_rows, _, short_state = _compute_m5(
        df,
        recurrence_threshold_atr=TOL,
        max_evidence_age_bars=7,
        return_registry_state=True,
    )
    long_rows, _, long_state = _compute_m5(
        df,
        recurrence_threshold_atr=TOL,
        max_evidence_age_bars=200,
        return_registry_state=True,
    )
    identity_fields = (
        "level_id",
        "birth_side",
        "center_price",
        "birth_bar",
        "member_pivot_bars",
        "member_pivot_prices",
    )
    assert [
        tuple(level[field] for field in identity_fields)
        for level in short_state["levels"]
    ] == [
        tuple(level[field] for field in identity_fields)
        for level in long_state["levels"]
    ]
    assert any(event[0] == "eligibility_expiry" for event in short_lifecycle)
    assert short_lifecycle != long_lifecycle
    assert not np.array_equal(short_rows, long_rows, equal_nan=True)


def test_runtime_shadow_observes_canonical_recurrence_and_break_lifecycle():
    s1_lifecycle: list[tuple] = []
    compute_level_registry_m5_block_v1(
        _series_s1(),
        recurrence_threshold_atr=TOL,
        runtime_lifecycle_log=s1_lifecycle,
    )
    assert {event[0] for event in s1_lifecycle} >= {"create", "recurrence"}
    assert "supersede" not in {event[0] for event in s1_lifecycle}

    s2_lifecycle: list[tuple] = []
    compute_level_registry_m5_block_v1(
        _series_s2(),
        recurrence_threshold_atr=TOL,
        runtime_lifecycle_log=s2_lifecycle,
    )
    assert {event[0] for event in s2_lifecycle} >= {"create", "break"}


def test_runtime_shadow_support_is_strictly_nonempty_when_required():
    n = 40
    no_pivots = _mk_df(
        SHIFT + 10.0 + 0.01 * np.arange(n),
        SHIFT + 9.0 - 0.01 * np.arange(n),
        SHIFT + 9.5 + 0.001 * np.arange(n),
    )
    population, lifecycle = collect_level_registry_runtime_population_v1(
        no_pivots,
        tf="m5",
        recurrence_threshold_atr=TOL,
        strict_nonempty=False,
    )
    assert population.size == 0
    assert lifecycle == []
    with pytest.raises(RuntimeError, match="RUNTIME_FIT_SUPPORT_EMPTY"):
        collect_level_registry_runtime_population_v1(
            no_pivots,
            tf="m5",
            recurrence_threshold_atr=TOL,
        )


def test_chunk_carry_bit_exactness_m5():
    df = _rng_df(n=200)
    full, _, full_state = compute_level_registry_m5_block_v1(
        df, recurrence_threshold_atr=0.6, return_registry_state=True
    )
    parts = []
    state = None
    for lo, hi in ((0, 7), (7, 97), (97, 141), (141, 200)):
        block, _, state = compute_level_registry_m5_block_v1(
            df.iloc[lo:hi],
            recurrence_threshold_atr=0.6,
            registry_state=state,
            return_registry_state=True,
        )
        parts.append(block)
    chunked = np.vstack(parts)
    np.testing.assert_array_equal(full, chunked)
    assert state == full_state


def test_chunk_carry_bit_exactness_mtf_and_atr_prefix():
    df = _rng_df(n=180, seed=7)
    atr = df["atr"].to_numpy().copy()
    atr[:37] = np.nan  # declared contiguous ATR warmup prefix
    df = df.assign(atr=atr)
    full, _, full_state = compute_level_registry_mtf_block_v1(
        df, tf="h1", recurrence_threshold_atr=0.7, return_registry_state=True
    )
    # warmup prefix is all-NaN; finite rows form a single suffix
    finite = np.isfinite(full).all(axis=1)
    assert finite.any()
    first = int(np.argmax(finite))
    assert first >= 37
    assert finite[first:].all() and not finite[:first].any()
    parts = []
    state = None
    for lo, hi in ((0, 20), (20, 44), (44, 180)):  # split inside the NaN prefix too
        block, _, state = compute_level_registry_mtf_block_v1(
            df.iloc[lo:hi],
            tf="h1",
            recurrence_threshold_atr=0.7,
            registry_state=state,
            return_registry_state=True,
        )
        parts.append(block)
    np.testing.assert_array_equal(full, np.vstack(parts))
    assert state == full_state


def test_one_pivot_truth_every_smc_v1_pivot_is_a_member():
    df = _rng_df(n=200)  # < AGE_CAP['m5'] so no level can expire
    lifecycle: list[tuple] = []
    compute_level_registry_m5_block_v1(
        df, recurrence_threshold_atr=0.6, runtime_lifecycle_log=lifecycle
    )
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    sh, sl = _detect_swing_pivots(high, low, SWING_LOOKBACK)
    expected = sorted(list(np.flatnonzero(sh)) + list(np.flatnonzero(sl)))
    members = sorted(int(event[2]) for event in lifecycle if event[0] == "create")
    assert members == expected


def test_emitted_value_domains_on_generic_series():
    df = _rng_df(n=200)
    m5, names = compute_level_registry_m5_block_v1(df, recurrence_threshold_atr=0.6)
    finite = np.isfinite(m5).all(axis=1)
    rows = m5[finite]
    for name in (
        "level_break_up_event",
        "level_break_down_event",
        "level_above_present",
        "level_above2_present",
        "level_below_present",
        "level_below2_present",
    ):
        assert set(np.unique(_col(rows, names, name))) <= {0.0, 1.0}
    for name in ("level_retest_hold_signed", "level_retest_fail_signed"):
        assert set(np.unique(_col(rows, names, name))) <= {-1.0, 0.0, 1.0}
    for name in ("level_above_dist_atr", "level_below_dist_atr"):
        col = _col(rows, names, name)
        assert (col >= 0.0).all()
    for name in (
        "level_above_touch_count",
        "level_below_touch_count",
        "level_above_age_bars",
        "level_below_age_bars",
        "level_above_bars_since_touch",
        "level_below_bars_since_touch",
        "level_broken_touch_count",
        "level_bars_since_break",
    ):
        col = _col(rows, names, name)
        assert (col >= 0.0).all()


def test_hyperfit_binds_exact_population_and_learned_lifetime(tmp_path: Path):
    df = _rng_df(n=700, seed=11)
    df.index = pd.date_range(
        "2020-01-01", periods=len(df), freq="5min", tz="UTC"
    )
    payload = fit_level_registry_hyperparameters_v1(
        df,
        tf="m5",
        inner_fit_end_exclusive=400,
        source_provenance=_fit_source(tmp_path, clock="M5"),
    )
    assert payload["selected_threshold_atr"] > 0.0
    assert payload["learned_expiry_bars"] > 0
    population = payload["population_configuration"]
    assert population["owner"] == (
        "level_immutable_anchor_recurrence_decomposition_v2"
    )
    assert population["runtime_owner"] == (
        "gx1.features.level_registry_v1._run_level_registry"
    )
    assert population["swing_lookback"] == SWING_LOOKBACK
    assert population["canonical_recurrence_observation_count"] > 0
    assert len(population["canonical_event_tape_sha256"]) == 64
    assert len(population["canonical_outcome_stream_sha256"]) == 64
    assert population["fit_phase"] == (
        "immutable_recurrence_canonical_outcome_decomposition"
    )
    assert population["final_fit_recurrence_population_sha256"] == (
        population["final_serve_recurrence_population_sha256"]
    )
    assert population["canonical_fit_outcome_population_sha256"] == (
        payload["outcome_stream_sha256"]
    )
    assert len(population["selected_derived_lifecycle_sha256"]) == 64
    assert len(population["selected_derived_episode_sha256"]) == 64
    assert len(population["selected_derived_state_sha256"]) == 64
    assert len(population["selected_derived_emission_sha256"]) == 64
    assert population["selected_eligibility_expiry_count"] >= 0
    assert payload["lifetime_candidate_count_total"] == 1
    assert payload["hazard_prior_fit"]["contract"] == (
        "inner_train_empirical_bayes_dirichlet_hazard_prior_v1"
    )
    rows, _ = _compute_m5(
        df,
        recurrence_threshold_atr=payload["selected_threshold_atr"],
        max_evidence_age_bars=payload["learned_expiry_bars"],
    )
    assert rows.shape == (len(df), len(LEVEL_REGISTRY_M5_FEATURE_NAMES))


def test_retired_13_14_expiry_cycle_uses_non_circular_population_decomposition(
    tmp_path: Path,
):
    # This exact TRAIN fixture made selected-lifecycle replay/refit oscillate
    # expiry 13 -> 14 -> 13. The artifact must neither hide the cycle nor pick
    # one endpoint: recurrence is now structurally independent of eligibility,
    # while canonical exact-cross outcomes remain the fit-only label tape.
    n_rows = 700
    rng = np.random.default_rng(20260814)
    mid = 2000.0 + np.cumsum(rng.normal(0.0, 1.0, n_rows))
    high = mid + np.abs(rng.normal(0.0, 0.7, n_rows)) + 0.1
    low = mid - np.abs(rng.normal(0.0, 0.7, n_rows)) - 0.1
    frame = pd.DataFrame(
        {
            "high": high,
            "low": low,
            "close": low + rng.uniform(0.0, 1.0, n_rows) * (high - low),
            "atr": 1.0 + np.abs(rng.normal(0.0, 0.2, n_rows)),
        },
        index=pd.date_range(
            "2020-01-01", periods=n_rows, freq="5min", tz="UTC"
        ),
    )
    population_13, lifecycle_13 = _collect_runtime(
        frame,
        tf="m5",
        recurrence_threshold_atr=0.2,
        max_evidence_age_bars=13,
    )
    population_14, lifecycle_14 = _collect_runtime(
        frame,
        tf="m5",
        recurrence_threshold_atr=0.2,
        max_evidence_age_bars=14,
    )
    np.testing.assert_array_equal(population_13, population_14, strict=True)
    assert lifecycle_13 != lifecycle_14
    payload = fit_level_registry_hyperparameters_v1(
        frame,
        tf="m5",
        inner_fit_end_exclusive=400,
        source_provenance=_fit_source(tmp_path, clock="M5"),
    )
    population = payload["population_configuration"]
    assert "fixed_point_contract" not in population
    assert population["final_fit_recurrence_population_sha256"] == (
        population["final_serve_recurrence_population_sha256"]
    )
    assert population["population_decomposition_contract"].startswith(
        "serve_recurrence_uses_all_prior_immutable_same_birth_side_anchors"
    )


def test_local_m1_and_m5_share_formula_but_preserve_clock_identity():
    df = _rng_df(n=200, seed=91)
    m1, m1_names, m1_state = _compute_m5(
        df,
        recurrence_threshold_atr=0.6,
        max_evidence_age_bars=80,
        decision_clock="m1",
        return_registry_state=True,
    )
    m5, m5_names, m5_state = _compute_m5(
        df,
        recurrence_threshold_atr=0.6,
        max_evidence_age_bars=80,
        decision_clock="m5",
        return_registry_state=True,
    )
    np.testing.assert_array_equal(m1, m5)
    assert m1_names == m5_names
    assert m1_state["tf"] == "m1"
    assert m5_state["tf"] == "m5"
    assert m1_state != m5_state


# ---------------------------------------------------------------------------
# Tolerance fit
# ---------------------------------------------------------------------------

def test_input_validation_fail_closed():
    df = _series_s1()
    with pytest.raises(RuntimeError, match="SOURCE_EMPTY"):
        compute_level_registry_m5_block_v1(df.iloc[0:0], recurrence_threshold_atr=TOL)
    with pytest.raises(RuntimeError, match="SOURCE_MISSING"):
        compute_level_registry_m5_block_v1(df.drop(columns=["atr"]), recurrence_threshold_atr=TOL)
    bad = df.copy()
    bad.loc[bad.index[4], "high"] = np.nan
    with pytest.raises(RuntimeError, match="SOURCE_NONFINITE"):
        compute_level_registry_m5_block_v1(bad, recurrence_threshold_atr=TOL)
    bad = df.copy()
    bad.loc[bad.index[4], "low"] = bad.loc[bad.index[4], "high"] + 1.0
    with pytest.raises(RuntimeError, match="GEOMETRY_INVALID"):
        compute_level_registry_m5_block_v1(bad, recurrence_threshold_atr=TOL)
    bad = df.copy()
    bad.loc[bad.index[4], "atr"] = 0.0
    with pytest.raises(RuntimeError, match="GEOMETRY_INVALID"):
        compute_level_registry_m5_block_v1(bad, recurrence_threshold_atr=TOL)
    bad = df.copy()  # mid-series ATR gap is not a prefix
    bad.loc[bad.index[10], "atr"] = np.nan
    with pytest.raises(RuntimeError, match="ATR_AVAILABILITY_INVALID"):
        compute_level_registry_m5_block_v1(bad, recurrence_threshold_atr=TOL)
    for bad_tol in (0.0, -1.0, float("nan"), float("inf"), None):
        with pytest.raises(RuntimeError, match="RECURRENCE_THRESHOLD_INVALID"):
            compute_level_registry_m5_block_v1(df, recurrence_threshold_atr=bad_tol)
    with pytest.raises(RuntimeError, match="TF_INVALID"):
        compute_level_registry_mtf_block_v1(df, tf="m30", recurrence_threshold_atr=TOL)


def test_state_contract_fail_closed():
    df = _rng_df(n=120)
    _, _, state = compute_level_registry_m5_block_v1(
        df.iloc[:60], recurrence_threshold_atr=0.6, return_registry_state=True
    )
    assert set(state) == set(LEVEL_REGISTRY_STATE_KEYS)
    # tf mismatch (m5 state into an h1 lane)
    with pytest.raises(RuntimeError, match="STATE_CONTRACT_MISMATCH"):
        compute_level_registry_mtf_block_v1(
            df.iloc[60:], tf="h1", recurrence_threshold_atr=0.6, registry_state=state
        )
    # tolerance mismatch: carried state must bind the exact frozen constant
    with pytest.raises(RuntimeError, match="STATE_CONTRACT_MISMATCH"):
        compute_level_registry_m5_block_v1(
            df.iloc[60:], recurrence_threshold_atr=0.61, registry_state=state
        )
    # schema drift
    broken = dict(state)
    broken.pop("levels")
    with pytest.raises(RuntimeError, match="STATE_SCHEMA_INVALID"):
        compute_level_registry_m5_block_v1(
            df.iloc[60:], recurrence_threshold_atr=0.6, registry_state=broken
        )
    broken = dict(state)
    broken["state_version"] = "level_registry_v0_state"
    with pytest.raises(RuntimeError, match="STATE_VERSION_MISMATCH"):
        compute_level_registry_m5_block_v1(
            df.iloc[60:], recurrence_threshold_atr=0.6, registry_state=broken
        )
    # a carried started-state forbids new unavailable-ATR rows
    started_state = state
    assert started_state["atr_started"] is True
    tail_df = df.iloc[60:].copy()
    tail_df.loc[tail_df.index[0], "atr"] = np.nan
    with pytest.raises(RuntimeError, match="ATR_AVAILABILITY_INVALID"):
        compute_level_registry_m5_block_v1(
            tail_df, recurrence_threshold_atr=0.6, registry_state=started_state
        )
