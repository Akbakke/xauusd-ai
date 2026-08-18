"""
Trendline/channel registry — V1 (V29 Phase A, chart_geometry family).

Bounded authority for SLOPED lines and channels, per
`docs/V29_EVENT_SURFACE_DESIGN_20260811.md` §2 and the chart_geometry report
Part B (`/home/andre2/GX1_DATA/logs/event_gap_review_20260811/chart_geometry.md`).
The level registry owns HORIZONTAL levels; this module owns two-point-anchored
lines with immutable anchors, a CANDIDATE -> ACTIVE -> BROKEN ->
(role-flipped back to ACTIVE | retired) state machine, and
parallel/converging channel pairs.

This module is stage 1 only: it declares the exact emission name tuples and
the computation, and is NOT wired into any producer, contract owner or
routing table.  Stage 2 (the V29 Phase-A wave) wires the emission through the
existing HTF matrix owner and specialist routing, exactly as the design doc
specifies.

One pivot truth (rules 13/19/21): confirmed swing pivots come from
``smc_v1._detect_swing_pivots`` with the ``SWING_LOOKBACK`` confirmation-lag
rule of ``smc_v1._track_recent_swings`` (a pivot at bar j participates only
from bar j + lookback).  The registry never re-detects pivots with a second
implementation; it calls the smc_v1 predicate on a rolling
(2*lookback+1)-bar window, which is bit-identical to the batch masks for
every confirmable bar (proven by the parity test in
``tests/test_trendline_registry_v1.py``).

Constant origins (rule 2a) — every decision-affecting number named:

- ``band_atr`` (tolerance band, ATR fraction) and identity expiry are selected
  per timeframe by :func:`fit_trendline_registry_hyperparameters_v1` using
  chronological TRAIN-only competing-risk likelihood. A two-anchor
  candidate gets exactly one validation opportunity: the next confirmed
  same-side pivot.  Before that pivot, a close through the exact projected
  line invalidates it.  Both lifecycle decisions are band-independent, so fit
  and serve observe the same population without a circular bootstrap.  The
  break margin REUSES the fitted band — one constant, no second number (rule
  2b).  This module has no default; the caller must pass the frozen value.
- ``seq_len`` (candidate window): the per-TF model sequence length —
  an explicit recipe input (``per_tf_seq_lens``, validated upstream by
  ``htf_features.require_multi_tf_resolution_pyramid``).  Line evidence never
  reaches beyond the receptive field the model actually sees.
- Retest has no fixed bar-window parameter. A BROKEN line remains armed for
  its first re-entry until learned identity expiry; raw break age is emitted.
- Counts and ages are emitted raw. Current active masks own current slot
  absence; break age is NaN before the first genuine break, so no global
  ever-seen mask or numeric sentinel is needed.
- Warmup: rows are NaN while ``bar_index < 2*swing_lookback + 2`` (the
  structural earliest bar at which an ACTIVE line can exist: three adjacent
  tie pivots at bars n, n+1, n+2, third confirmed at 2n+2 — pure arithmetic
  on the pivot contract) or while ATR is unavailable (band undefined).  The
  NaN region is a single chronological prefix; afterwards absence is emitted
  as presence-flag 0 with attributes 0, the established
  the sided SMC sweep-depth flag-disambiguated-zero pattern (design B.5,
  not a placeholder under rule 2e).

Intra-bar order (fixed, documented; design B.1/B.3):
  (0) prune both pivot stores and both CANDIDATE populations against bar t;
      no validation or channel construction may observe evidence outside the
      declared ``seq_len`` receptive field.
  (1) pivot confirmation at bar t for pivot bar j = t - lookback
      (support side first, then resistance):
      (1a) the candidate's one third-touch validation test and later evidence
           updates for same-side ACTIVE lines (deviation measured at the
           pivot's own bar with that bar's ATR — where the touch physically
           happened; the decision is taken at t, causally);
      (1b) new CANDIDATE lines pairing the pivot with each stored same-side
           pivot; then the pivot enters the store.
  (2) per-line bar update with bar t's close/extremes/ATR:
      CANDIDATE violation discard, ACTIVE first-break (edge-triggered once,
      line -> BROKEN), ACTIVE intra-band touch (first-entry-per-excursion),
      BROKEN retest hold (role flip back to ACTIVE) / fail / expiry
      (the last two retire the line).
  (3) emission of the 33-field block from the post-update population
      (V30 2026-08-13: + per-side ACTIVE counts and break memory).

Documented design decisions inside the adopted spec:

- A CANDIDATE whose line is crossed on the exact structurally wrong side is
  discarded silently (no event: it never had validated identity).  The exact
  projection is an algebraic boundary, not another tolerance.  This prevents
  later promotion of a line already broken before its identity existed while
  keeping fit/serve population membership independent of ``band_atr``.
- An ACTIVE line is retired silently once ``seq_len`` bars pass since its
  last touch: every bar of its evidence has then left the declared
  receptive-field window, the same bound B.1 applies to the pivot store
  ("line evidence never reaches beyond the receptive field the model
  actually sees"), and it is what makes the report's O(L) ~ O(P) per-bar
  cost arithmetic true (B.7).  A touch renews the line, so genuinely
  long-lived respected lines persist.  Bound origin: the existing seq_len
  recipe input — no new constant.
- Touch semantics: a pivot within the band always counts (inclusive <=);
  an intra-band bar touch requires the side-relevant extreme in the band AND
  the close on the intact side (close on the line counts as intact — the
  break requires close beyond the band).  Per-bar touches deduplicate per
  excursion (first entry only) and per bar (a bar counted as an intra-band
  touch is not re-counted when it later confirms as a pivot; the pivot's
  deviation still updates ``max_dev_atr``, the R^2 substitute).
- Retest (design B.3): after BREAK, first re-entry
  of the band from the break side resolves the line: close still beyond the
  old line => RETEST_HOLD; close back through (or exactly on it
  — ambiguous evidence fails closed to FAIL) => RETEST_FAIL.  Re-entry stays
  armed until receptive-field expiry. V30 package 8A (2026-08-13) PERFORMS the role
  flip this clause always named: a HOLD returns the line to ACTIVE with its
  ``side`` flipped and its touch history intact (the retest bar counts as
  the touch) instead of deleting it on the bar it proved itself as flipped
  resistance/support.  A FAIL still retires the line.
- Break-bar attribute fields: if several lines break on one bar, the
  reported touch_count/age belong to the line with the highest touch count
  (strongest evidence), ties to the older line (smaller line_id).  The
  break impulses themselves are binary ORs.
- Slot ties: projection exactly on the close goes to the ABOVE slot
  (keeps ``geomline_above_dist_atr >= 0`` exact); equal-distance line ties
  resolve to the smaller line_id (older identity).  Channel-route order on
  an exact slot-distance tie evaluates the above line first.  All ties are
  measure-zero and deterministic.
- Channel (design B.4): triangle first — both slots occupied, below is a
  support-side line, above a resistance-side line, strictly converging
  slopes; else a parallel return rail anchored on the most extreme
  opposite-side stored pivot with >= 2 pivots within the band of that rail.
  Only the nearest-to-price pair is emitted.  The parallel construction
  introduces no slope tolerance (the rail is parallel by construction) and
  the triangle uses a strict inequality — no new constants.  The emitted
  triangle slope is the midline slope (s_lower + s_upper)/2, an algebraic
  property of the two anchored rails, not a chosen weight.

Evidence class: everything in this file is proven from source/algebra plus
synthetic-execution tests; no claim about real-tape behaviour is made here.
Registry compute cost on the real declared tape is a pre-adoption red gate
(design §6, measurement 1).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from gx1.contracts.registry_hyperparameter_fit_v1 import (
    REGISTRY_OUTCOME_BREAK,
    REGISTRY_OUTCOME_CENSORED,
    REGISTRY_OUTCOME_REACTION,
    RegistryOutcomeStreamV1,
    fit_registry_competing_risk_threshold_v1,
    require_registry_hyperparameter_payload_v1,
)

# One pivot truth (design §2, chart report B.1): predicate + confirmation lag
# are owned by smc_v1; this registry consumes them and never re-detects.
from gx1.features.smc_v1 import SWING_LOOKBACK, _detect_swing_pivots
from gx1.features.event_age_v1 import raw_event_age_from_last_observed_row


# V4 (2026-08-15): the geomline_{above,below}_active presence masks are retired
# from the emitted block; the graded per-side ACTIVE counts remain.
# V5 (2026-08-15): the ACTIVE staleness bound returns to the declared receptive
# field ``seq_len`` and the BROKEN retest clock returns to its own named
# constant, so ``config_key`` loses the retired ``identity_expiry_bars`` member
# and every emitted value changes; chunk-carry state from a V4 run must not
# interoperate silently.
TRENDLINE_REGISTRY_CONTRACT_V1 = "TRENDLINE_REGISTRY_TWO_POINT_ANCHOR_RAW_V5"

TRENDLINE_SIDE_SUPPORT = 1
TRENDLINE_SIDE_RESISTANCE = -1

TRENDLINE_STATE_CANDIDATE = "candidate"
TRENDLINE_STATE_ACTIVE = "active"
TRENDLINE_STATE_BROKEN = "broken"

# Structural earliest bar (0-based) at which an ACTIVE line can exist for the
# default lookback — pure arithmetic on the pivot contract (module docstring).
TRENDLINE_STRUCTURAL_WARMUP_BARS_V1 = 2 * SWING_LOOKBACK + 2

# How long a BROKEN line stays armed for its single re-entry decision.
# Origin (rule 2a): the family's own named ``SWING_LOOKBACK`` — the full
# look-around span of one swing pivot is the shortest interval in which a
# re-entry can itself produce a confirmable pivot.  This is a lifecycle clock
# of the retest event and is deliberately NOT the ACTIVE staleness bound: the
# two measure different things and must not collapse into one number.
TRENDLINE_RETEST_WINDOW_BARS_V1: int = 2 * SWING_LOOKBACK + 1

# ---------------------------------------------------------------------------
# Exact emission name tuples (design B.5) — the stage-2 wiring contract.
# ---------------------------------------------------------------------------

# V30 (2026-08-13) added the graded per-side occupancy
# ``geomline_{above,below}_active_count``: the number of ACTIVE lines
# projecting on that side of the close.  It is raw and uncapped.
#
# 2026-08-15: the presence MASKS ``geomline_{above,below}_active`` are retired.
#
# PROVEN FROM SOURCE, and this alone carries the change: in :func:`_emit_row`
# the mask and the count beside it were written from the same branch over the
# same loop, so the mask was exactly the count's ``>= 1`` indicator and
# carried no evidence the count does not carry.  CLAUDE.md rule 4 permits
# retiring a derived field only while every field it consumed stays a model
# input: the ACTIVE-line population it read IS the count, and the count stays.
# Rule 22 then prefers the smaller surface.
#
# CORROBORATING, MEASURED but on a SUPERSEDED axis
# (docs/V29_EVENT_SURFACE_DESIGN_20260811.md §10, native-M5 V4 tape -> D1,
# N=2304 declared D1 TRAIN bars, measured 2026-08-11 on the midnight-UTC D1
# origin that V30 package 3 replaced with the trading-day origin two days
# later): ``geomline_below_active`` was exactly 1.0 on EVERY row and took the
# V29B chain RED on ``D1:constant_fields=['geomline_below_active']`` while all
# six sibling attributes stayed non-constant.  The saturation exemption
# written for that RED has since been retired (see
# tests/test_htf_v4_liveness_saturation_contract.py), leaving the mask with no
# route to green at D1 scale.  NOT RE-MEASURED on the trading-day axis — that
# needs a TRAIN-fitted registry-constants artifact, and none has been fitted.
#
# NOT the reason, despite looking like one: mask and count are bit-identical
# on any lane whose per-side ACTIVE population never exceeds 1, but both are
# MANDATORY selected fields and the ``exact_duplicate`` path in
# ``entry_model_native_feature_availability_v1`` only ever sees the CANDIDATE
# pool, which excludes mandatory fields by construction.  That duplicate would
# have been caught by the per-TF liveness owner, not by the availability
# owner.
TRENDLINE_REGISTRY_SLOT_FEATURE_NAMES_V1 = (
    "geomline_above_active_count",
    "geomline_above_dist_atr",
    "geomline_above_slope_atr_per_bar",
    "geomline_above_touch_count",
    "geomline_above_age_bars",
    "geomline_above_last_touch_age_bars",
    "geomline_above_max_dev_atr",
    "geomline_below_active_count",
    "geomline_below_dist_atr",
    "geomline_below_slope_atr_per_bar",
    "geomline_below_touch_count",
    "geomline_below_age_bars",
    "geomline_below_last_touch_age_bars",
    "geomline_below_max_dev_atr",
)

TRENDLINE_REGISTRY_EVENT_FEATURE_NAMES_V1 = (
    "geomline_touch_above",
    "geomline_touch_below",
    "geomline_break_up",
    "geomline_break_down",
    "geomline_break_line_touch_count",
    "geomline_break_line_age_bars",
    "geomline_retest_hold_up",
    "geomline_retest_fail_up",
    "geomline_retest_hold_down",
    "geomline_retest_fail_down",
    # Raw bars since the most recent first-break of any registry line.
    "geomline_bars_since_break",
)

TRENDLINE_REGISTRY_CHANNEL_FEATURE_NAMES_V1 = (
    "geomchan_active",
    "geomchan_width_atr",
    "geomchan_pos_0_1",
    "geomchan_slope_atr_per_bar",
    "geomchan_converging",
    "geomchan_apex_proximity",
)

TRENDLINE_REGISTRY_FEATURE_NAMES_V1 = (
    TRENDLINE_REGISTRY_SLOT_FEATURE_NAMES_V1
    + TRENDLINE_REGISTRY_EVENT_FEATURE_NAMES_V1
    + TRENDLINE_REGISTRY_CHANNEL_FEATURE_NAMES_V1
)

TRENDLINE_REGISTRY_FEATURE_COUNT_V1 = len(TRENDLINE_REGISTRY_FEATURE_NAMES_V1)


def _require_feature_name_integrity() -> None:
    names = TRENDLINE_REGISTRY_FEATURE_NAMES_V1
    if len(names) != TRENDLINE_REGISTRY_FEATURE_COUNT_V1:
        raise RuntimeError(
            "[TRENDLINE_FEATURE_NAMES_INVALID] expected "
            f"{TRENDLINE_REGISTRY_FEATURE_COUNT_V1} names, got {len(names)}"
        )
    invalid = [n for n in names if not isinstance(n, str) or not n]
    if invalid:
        raise RuntimeError(f"[TRENDLINE_FEATURE_NAMES_INVALID] {invalid[:5]}")
    if len(set(names)) != len(names):
        raise RuntimeError("[TRENDLINE_FEATURE_NAMES_INVALID] duplicate names")


_require_feature_name_integrity()

# Same derivation convention as MULTI_TF_FEATURE_NAMES_SHA256_V4
# (htf_features.py): the ordered name list is the hashed contract surface.
TRENDLINE_REGISTRY_FEATURE_NAMES_SHA256_V1 = hashlib.sha256(
    json.dumps(
        list(TRENDLINE_REGISTRY_FEATURE_NAMES_V1),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()

# Event vector layout — must mirror the per-bar impulse prefix of
# TRENDLINE_REGISTRY_EVENT_FEATURE_NAMES_V1 (geomline_bars_since_break is
# persistent carry state, emitted from the registry state, not from this
# per-bar-reset vector).
_EV_TOUCH_ABOVE = 0
_EV_TOUCH_BELOW = 1
_EV_BREAK_UP = 2
_EV_BREAK_DOWN = 3
_EV_BREAK_LINE_TOUCH_COUNT = 4
_EV_BREAK_LINE_AGE_BARS = 5
_EV_RETEST_HOLD_UP = 6
_EV_RETEST_FAIL_UP = 7
_EV_RETEST_HOLD_DOWN = 8
_EV_RETEST_FAIL_DOWN = 9
_EV_COUNT = 10

# Emitted-row layout, DERIVED from the name tuples above rather than restated
# as literals (CLAUDE.md rule 13: a repeated literal is not ownership — the
# hand-numbered ``row[0]..row[32]`` writes that this replaced had to be
# renumbered by hand every time a field entered or left the tuple).  Each side
# block is the same ordered septet, so one base index per side is enough.
_SLOT_SIDE_FIELD_COUNT = len(TRENDLINE_REGISTRY_SLOT_FEATURE_NAMES_V1) // 2
_IX_ABOVE_BASE = TRENDLINE_REGISTRY_FEATURE_NAMES_V1.index(
    "geomline_above_active_count"
)
_IX_BELOW_BASE = TRENDLINE_REGISTRY_FEATURE_NAMES_V1.index(
    "geomline_below_active_count"
)
_SLOT_ACTIVE_COUNT = 0
_SLOT_DIST_ATR = 1
_SLOT_SLOPE_ATR_PER_BAR = 2
_SLOT_TOUCH_COUNT = 3
_SLOT_AGE_BARS = 4
_SLOT_LAST_TOUCH_AGE_BARS = 5
_SLOT_MAX_DEV_ATR = 6
_IX_EVENT_BASE = TRENDLINE_REGISTRY_FEATURE_NAMES_V1.index("geomline_touch_above")
_IX_BARS_SINCE_BREAK = TRENDLINE_REGISTRY_FEATURE_NAMES_V1.index(
    "geomline_bars_since_break"
)
_IX_CHANNEL_BASE = TRENDLINE_REGISTRY_FEATURE_NAMES_V1.index("geomchan_active")


def _require_emitted_row_layout() -> None:
    """Prove the derived write offsets reproduce the declared name order."""

    if _SLOT_SIDE_FIELD_COUNT * 2 != len(TRENDLINE_REGISTRY_SLOT_FEATURE_NAMES_V1):
        raise RuntimeError("[TRENDLINE_ROW_LAYOUT_INVALID] slot block not paired")
    above = TRENDLINE_REGISTRY_SLOT_FEATURE_NAMES_V1[:_SLOT_SIDE_FIELD_COUNT]
    below = TRENDLINE_REGISTRY_SLOT_FEATURE_NAMES_V1[_SLOT_SIDE_FIELD_COUNT:]
    if tuple(
        name.replace("_above_", "_", 1) for name in above
    ) != tuple(name.replace("_below_", "_", 1) for name in below):
        raise RuntimeError("[TRENDLINE_ROW_LAYOUT_INVALID] side blocks differ")
    if (_IX_ABOVE_BASE, _IX_BELOW_BASE) != (0, _SLOT_SIDE_FIELD_COUNT):
        raise RuntimeError("[TRENDLINE_ROW_LAYOUT_INVALID] side bases moved")
    if _IX_EVENT_BASE != len(TRENDLINE_REGISTRY_SLOT_FEATURE_NAMES_V1):
        raise RuntimeError("[TRENDLINE_ROW_LAYOUT_INVALID] event block moved")
    if _IX_BARS_SINCE_BREAK != _IX_EVENT_BASE + _EV_COUNT:
        raise RuntimeError("[TRENDLINE_ROW_LAYOUT_INVALID] event vector width")
    if _IX_CHANNEL_BASE != _IX_BARS_SINCE_BREAK + 1:
        raise RuntimeError("[TRENDLINE_ROW_LAYOUT_INVALID] channel block moved")
    if (
        _IX_CHANNEL_BASE + len(TRENDLINE_REGISTRY_CHANNEL_FEATURE_NAMES_V1)
        != TRENDLINE_REGISTRY_FEATURE_COUNT_V1
    ):
        raise RuntimeError("[TRENDLINE_ROW_LAYOUT_INVALID] trailing width")


_require_emitted_row_layout()


# ---------------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------------


@dataclass
class TrendlineV1:
    """One registry line.

    Identity — ``line_id`` and the two anchors, hence the projection itself —
    is immutable from creation (design B.1: an event source must not rewrite
    its own history).  Evidence attributes (touch_count, last_touch_bar,
    max_dev_atr, touch_bars, in_band_prev) and lifecycle state evolve.

    ``side`` was part of the immutable identity until V30 package 8A
    (2026-08-13).  It is now lifecycle state: a RETEST_HOLD flips it once, in
    place, because that is precisely what the event proves — the line still
    governs price, from the other side ("old support is new resistance").  The
    anchors, and therefore every historical claim the line has made, are
    untouched by the flip.
    """

    line_id: int
    side: int
    anchor1_bar: int
    anchor1_price: float
    anchor2_bar: int
    anchor2_price: float
    state: str
    touch_count: int
    last_touch_bar: int
    max_dev_atr: float
    touch_bars: set = field(default_factory=set)
    in_band_prev: bool = False
    break_bar: int = -1
    break_dir: int = 0
    # Derived once from the immutable anchors (never re-fitted).
    slope: float = field(init=False)
    intercept: float = field(init=False)

    def __post_init__(self) -> None:
        self.slope = (self.anchor2_price - self.anchor1_price) / float(
            self.anchor2_bar - self.anchor1_bar
        )
        self.intercept = self.anchor1_price - self.slope * float(
            self.anchor1_bar
        )

    def projection(self, bar: int) -> float:
        return self.intercept + self.slope * float(bar)


@dataclass
class _CandidateArraysV1:
    """CANDIDATE lines as parallel arrays (bounded compute).

    A CANDIDATE carries exactly its two anchors and no evolving evidence —
    the next same-side pivot is its single validation opportunity; an
    in-band 3rd touch promotes it to an ACTIVE :class:`TrendlineV1`, while a
    miss retires it.  Storing the
    O(P^2) candidate population as arrays keeps the per-bar violation check
    and the per-pivot promotion test vectorized; only ACTIVE/BROKEN lines
    (few) are walked as Python objects per bar.  Semantics are identical to
    the object form: line_ids are assigned at creation, order is creation
    order."""

    line_id: np.ndarray
    anchor1_bar: np.ndarray
    anchor1_price: np.ndarray
    anchor2_bar: np.ndarray
    anchor2_price: np.ndarray
    slope: np.ndarray
    intercept: np.ndarray

    @classmethod
    def empty(cls) -> "_CandidateArraysV1":
        return cls(
            line_id=np.empty(0, dtype=np.int64),
            anchor1_bar=np.empty(0, dtype=np.int64),
            anchor1_price=np.empty(0, dtype=np.float64),
            anchor2_bar=np.empty(0, dtype=np.int64),
            anchor2_price=np.empty(0, dtype=np.float64),
            slope=np.empty(0, dtype=np.float64),
            intercept=np.empty(0, dtype=np.float64),
        )

    def __len__(self) -> int:
        return int(self.line_id.shape[0])

    def keep(self, mask: np.ndarray) -> None:
        if bool(mask.all()):
            return
        self.line_id = self.line_id[mask]
        self.anchor1_bar = self.anchor1_bar[mask]
        self.anchor1_price = self.anchor1_price[mask]
        self.anchor2_bar = self.anchor2_bar[mask]
        self.anchor2_price = self.anchor2_price[mask]
        self.slope = self.slope[mask]
        self.intercept = self.intercept[mask]

    def extend_pairs(
        self,
        ids: np.ndarray,
        anchor1_bars: np.ndarray,
        anchor1_prices: np.ndarray,
        anchor2_bar: int,
        anchor2_price: float,
    ) -> None:
        span = (anchor2_bar - anchor1_bars).astype(np.float64)
        slope = (anchor2_price - anchor1_prices) / span
        intercept = anchor1_prices - slope * anchor1_bars.astype(np.float64)
        self.line_id = np.concatenate((self.line_id, ids))
        self.anchor1_bar = np.concatenate((self.anchor1_bar, anchor1_bars))
        self.anchor1_price = np.concatenate(
            (self.anchor1_price, anchor1_prices)
        )
        self.anchor2_bar = np.concatenate(
            (self.anchor2_bar, np.full(len(ids), anchor2_bar, dtype=np.int64))
        )
        self.anchor2_price = np.concatenate(
            (
                self.anchor2_price,
                np.full(len(ids), anchor2_price, dtype=np.float64),
            )
        )
        self.slope = np.concatenate((self.slope, slope))
        self.intercept = np.concatenate((self.intercept, intercept))


@dataclass
class TrendlineRegistryStateV1:
    """Exact serializable carry state — chunked processing with this state is
    bit-identical to one-shot processing (proven by the chunk-carry test).

    ``active_lines`` holds ACTIVE and BROKEN :class:`TrendlineV1` objects;
    CANDIDATEs live in the vectorized per-side stores."""

    config_key: tuple
    bar_count: int = 0
    atr_seen: bool = False
    buf_high: list = field(default_factory=list)
    buf_low: list = field(default_factory=list)
    buf_atr: list = field(default_factory=list)
    support_pivots: list = field(default_factory=list)
    resistance_pivots: list = field(default_factory=list)
    active_lines: list = field(default_factory=list)
    cand_support: _CandidateArraysV1 = field(
        default_factory=_CandidateArraysV1.empty
    )
    cand_resistance: _CandidateArraysV1 = field(
        default_factory=_CandidateArraysV1.empty
    )
    next_line_id: int = 0
    last_index: object = None
    # V30 break memory: registry bar index of the most recent first-break of
    # any line; -1 is internal state only and emits as an honest NaN age.
    last_break_bar: int = -1


# ---------------------------------------------------------------------------
# Validation (fail-closed; mirrors the compute_smc_mtf_primitives_v1 input
# contract, including the monotone ATR-availability prefix)
# ---------------------------------------------------------------------------


def _require_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise RuntimeError(f"[TRENDLINE_{name}_INVALID] {value!r}")
    if int(value) <= 0:
        raise RuntimeError(f"[TRENDLINE_{name}_INVALID] {value!r}")
    return int(value)


def _require_band(band_atr: object) -> float:
    if isinstance(band_atr, bool) or not isinstance(
        band_atr, (int, float, np.integer, np.floating)
    ):
        raise RuntimeError(f"[TRENDLINE_BAND_INVALID] {band_atr!r}")
    band = float(band_atr)
    if not math.isfinite(band) or band <= 0.0:
        raise RuntimeError(f"[TRENDLINE_BAND_INVALID] {band_atr!r}")
    return band


def _require_timeframe(timeframe: object) -> str:
    if not isinstance(timeframe, str) or not timeframe:
        raise RuntimeError(f"[TRENDLINE_TIMEFRAME_INVALID] {timeframe!r}")
    return timeframe


def _validated_arrays(
    df: pd.DataFrame,
    high_col: str,
    low_col: str,
    close_col: str,
    atr_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(df) == 0:
        raise RuntimeError("[TRENDLINE_SOURCE_EMPTY]")
    required = (high_col, low_col, close_col, atr_col)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RuntimeError(f"[TRENDLINE_SOURCE_MISSING] {missing}")
    high = pd.to_numeric(df[high_col], errors="coerce").to_numpy(dtype=np.float64)
    low = pd.to_numeric(df[low_col], errors="coerce").to_numpy(dtype=np.float64)
    close = pd.to_numeric(df[close_col], errors="coerce").to_numpy(dtype=np.float64)
    atr = pd.to_numeric(df[atr_col], errors="coerce").to_numpy(dtype=np.float64)
    if (
        not np.isfinite(high).all()
        or not np.isfinite(low).all()
        or not np.isfinite(close).all()
        or np.isinf(atr).any()
    ):
        raise RuntimeError("[TRENDLINE_SOURCE_NONFINITE]")
    atr_available = np.isfinite(atr)
    if atr_available.any():
        first_atr = int(np.argmax(atr_available))
        if not atr_available[first_atr:].all():
            raise RuntimeError("[TRENDLINE_ATR_AVAILABILITY_INVALID]")
    if (
        np.any(high <= 0.0)
        or np.any(low <= 0.0)
        or np.any(close <= 0.0)
        or np.any(atr[atr_available] <= 0.0)
        or np.any(high < low)
        or np.any(high < close)
        or np.any(low > close)
    ):
        raise RuntimeError("[TRENDLINE_SOURCE_GEOMETRY_INVALID]")
    return high, low, close, atr


def _require_increasing_index(df: pd.DataFrame) -> None:
    index = df.index
    if len(index) > 1:
        values = index.to_numpy()
        try:
            increasing = bool(np.all(values[1:] > values[:-1]))
        except TypeError as exc:
            raise RuntimeError("[TRENDLINE_INDEX_NOT_INCREASING]") from exc
        if not increasing:
            raise RuntimeError("[TRENDLINE_INDEX_NOT_INCREASING]")


# ---------------------------------------------------------------------------
# Core per-bar mechanics
# ---------------------------------------------------------------------------


def _confirmed_center_pivots(
    buf_high: list, buf_low: list, swing_lookback: int
) -> tuple[bool, bool]:
    """Test the center bar of the full (2n+1) rolling window with the exact
    smc_v1 predicate.  Only index n is testable, which is precisely the bar
    that confirms now (one pivot truth; parity-proven in the tests)."""
    wh = np.asarray(buf_high, dtype=np.float64)
    wl = np.asarray(buf_low, dtype=np.float64)
    sh_mask, sl_mask = _detect_swing_pivots(wh, wl, swing_lookback)
    return bool(sh_mask[swing_lookback]), bool(sl_mask[swing_lookback])


def _prune_receptive_field_state(
    state: TrendlineRegistryStateV1,
    *,
    t: int,
    seq_len: int,
) -> None:
    """Remove unobservable pivot/candidate evidence before it can be used.

    The decision row at ``t`` may observe only bars ``b`` satisfying
    ``t - b < seq_len``.  Pruning once at the beginning of every bar is
    necessary for two independent consumers: candidate promotion on a newly
    confirmed pivot and the parallel-channel route, which reads the
    opposite-side pivot store even on bars with no new pivot of that side.
    """

    state.support_pivots[:] = [
        (bar, price)
        for bar, price in state.support_pivots
        if t - bar < seq_len
    ]
    state.resistance_pivots[:] = [
        (bar, price)
        for bar, price in state.resistance_pivots
        if t - bar < seq_len
    ]
    if len(state.cand_support):
        state.cand_support.keep(t - state.cand_support.anchor1_bar < seq_len)
    if len(state.cand_resistance):
        state.cand_resistance.keep(
            t - state.cand_resistance.anchor1_bar < seq_len
        )


def _ingest_confirmed_pivot(
    state: TrendlineRegistryStateV1,
    side: int,
    pivot_bar: int,
    pivot_price: float,
    pivot_atr: float,
    t: int,
    band_atr: float,
    seq_len: int,
    events: np.ndarray,
    line_log: list | None = None,
    candidate_deviation_log: list[float] | None = None,
    candidate_observation_log: list[tuple] | None = None,
) -> None:
    if side == TRENDLINE_SIDE_SUPPORT:
        store = state.support_pivots
        candidates = state.cand_support
        touch_event = _EV_TOUCH_BELOW
    else:
        store = state.resistance_pivots
        candidates = state.cand_resistance
        touch_event = _EV_TOUCH_ABOVE
    # (1a) ACTIVE-line evidence updates and the CANDIDATE's single validating
    # 3rd-touch opportunity, same side only.
    for line in state.active_lines:
        if line.side != side or line.state != TRENDLINE_STATE_ACTIVE:
            continue
        dev = abs(pivot_price - line.projection(pivot_bar)) / pivot_atr
        if dev > band_atr:
            continue
        if dev > line.max_dev_atr:
            line.max_dev_atr = dev
        if pivot_bar not in line.touch_bars:
            line.touch_count += 1
            line.touch_bars.add(pivot_bar)
            line.last_touch_bar = pivot_bar
            events[touch_event] = 1.0
            if line_log is not None:
                line_log.append(("touch", t, line.line_id, line.side, line.slope))
    if len(candidates):
        projections = candidates.intercept + candidates.slope * float(pivot_bar)
        deviations = np.abs(pivot_price - projections) / pivot_atr
        if candidate_deviation_log is not None:
            candidate_deviation_log.extend(
                float(value) for value in deviations.tolist()
            )
        if candidate_observation_log is not None:
            candidate_observation_log.extend(
                (
                    int(t),
                    int(candidates.line_id[idx]),
                    int(side),
                    float(candidates.slope[idx]),
                    float(candidates.intercept[idx]),
                    float(deviations[idx]),
                )
                for idx in range(len(candidates))
            )
        promoted = deviations <= band_atr
        if bool(promoted.any()):
            events[touch_event] = 1.0
            for idx in np.flatnonzero(promoted):
                anchor1_bar = int(candidates.anchor1_bar[idx])
                anchor2_bar = int(candidates.anchor2_bar[idx])
                promoted_line = TrendlineV1(
                    line_id=int(candidates.line_id[idx]),
                    side=side,
                    anchor1_bar=anchor1_bar,
                    anchor1_price=float(candidates.anchor1_price[idx]),
                    anchor2_bar=anchor2_bar,
                    anchor2_price=float(candidates.anchor2_price[idx]),
                    state=TRENDLINE_STATE_ACTIVE,
                    touch_count=3,
                    last_touch_bar=int(pivot_bar),
                    max_dev_atr=float(deviations[idx]),
                    touch_bars={anchor1_bar, anchor2_bar, int(pivot_bar)},
                )
                state.active_lines.append(promoted_line)
                if line_log is not None:
                    line_log.append(
                        (
                            "touch",
                            t,
                            promoted_line.line_id,
                            promoted_line.side,
                            promoted_line.slope,
                        )
                    )
        # The next same-side pivot is the candidate's one third-touch test.
        # A miss is evidence against this exact two-anchor identity, not a
        # reason to wait for an arbitrary fourth/fifth pivot and silently
        # change the fit population.
        candidates.keep(np.zeros(len(candidates), dtype=bool))
    # (1b) new candidates: pair the pivot with each stored same-side pivot.
    if store:
        anchor1_bars = np.array([b for b, _ in store], dtype=np.int64)
        anchor1_prices = np.array([p for _, p in store], dtype=np.float64)
        ids = np.arange(
            state.next_line_id,
            state.next_line_id + len(store),
            dtype=np.int64,
        )
        state.next_line_id += len(store)
        candidates.extend_pairs(
            ids, anchor1_bars, anchor1_prices, int(pivot_bar), float(pivot_price)
        )
    store.append((int(pivot_bar), float(pivot_price)))


def _discard_violated_candidates_for_bar(
    state: TrendlineRegistryStateV1,
    *,
    t: int,
    close: float,
) -> None:
    """Retire unvalidated lines crossed on the structurally wrong side.

    The exact projection is the algebraic support/resistance boundary.  Using
    it here keeps candidate population membership independent of the fitted
    tolerance while preventing a future promotion after the candidate was
    already structurally broken.
    """

    if len(state.cand_support):
        projections = (
            state.cand_support.intercept + state.cand_support.slope * float(t)
        )
        state.cand_support.keep(~(close < projections))
    if len(state.cand_resistance):
        projections = (
            state.cand_resistance.intercept
            + state.cand_resistance.slope * float(t)
        )
        state.cand_resistance.keep(~(close > projections))


def _update_lines_for_bar(
    state: TrendlineRegistryStateV1,
    t: int,
    high: float,
    low: float,
    close: float,
    atr_t: float,
    band_atr: float,
    swing_lookback: int,
    seq_len: int,
    events: np.ndarray,
    line_log: list | None = None,
) -> None:
    _discard_violated_candidates_for_bar(
        state,
        t=t,
        close=close,
    )
    margin = band_atr * atr_t
    keep: list = []
    broken_this_bar: list = []
    for line in state.active_lines:
        if (
            line.state == TRENDLINE_STATE_ACTIVE
            and t - line.last_touch_bar >= seq_len
        ):
            # Staleness retirement (no event): every bar of this line's
            # evidence has left the declared receptive-field window
            # (design B.1 — "line evidence never reaches beyond the
            # receptive field the model actually sees"); this is also what
            # makes the report's O(L) ~ O(P) per-bar cost bound true (B.7).
            #
            # Bound origin (rule 2a): the declared receptive field ``seq_len``
            # itself — the same explicit recipe input this call already
            # carries, no new constant.  This is an INFORMATION bound, not a
            # claim about how long a trendline lives in the market: a line
            # whose most recent evidence lies outside the model's own input
            # window cannot be corroborated by the model, so keeping it would
            # publish a state the model cannot see the basis for.
            #
            # It must never be replaced by a fitted event-time lifetime.  A
            # learned trendline expiry was substituted here on 2026-08-15
            # (a94f5c6e) and, being <= SWING_LOOKBACK on every clock, deleted
            # every promoted line on its own promotion bar — the promotion
            # path stamps ``last_touch_bar = t - SWING_LOOKBACK``, so the very
            # first test on bar t reads an age of SWING_LOOKBACK.  28 of the
            # 31 emitted fields went constant 0.0 and the build died on
            # HTF_V4_CACHE_WARMUP_INVALID.  ``seq_len >= 16 > SWING_LOOKBACK``
            # on every declared lane, which is what makes this bound safe, and
            # that is arithmetic, not a measurement.
            continue
        proj = line.projection(t)
        if line.state == TRENDLINE_STATE_ACTIVE:
            if line.side == TRENDLINE_SIDE_SUPPORT and close < proj - margin:
                line.state = TRENDLINE_STATE_BROKEN
                line.break_bar = t
                line.break_dir = -1
                events[_EV_BREAK_DOWN] = 1.0
                state.last_break_bar = t
                broken_this_bar.append(line)
                keep.append(line)
                if line_log is not None:
                    line_log.append(("break", t, line.line_id))
            elif (
                line.side == TRENDLINE_SIDE_RESISTANCE and close > proj + margin
            ):
                line.state = TRENDLINE_STATE_BROKEN
                line.break_bar = t
                line.break_dir = 1
                events[_EV_BREAK_UP] = 1.0
                state.last_break_bar = t
                broken_this_bar.append(line)
                keep.append(line)
                if line_log is not None:
                    line_log.append(("break", t, line.line_id))
            else:
                if line.side == TRENDLINE_SIDE_SUPPORT:
                    extreme = low
                    intact = close >= proj
                    touch_event = _EV_TOUCH_BELOW
                else:
                    extreme = high
                    intact = close <= proj
                    touch_event = _EV_TOUCH_ABOVE
                in_band = abs(extreme - proj) <= margin
                if in_band and intact and not line.in_band_prev:
                    line.touch_count += 1
                    line.touch_bars.add(t)
                    line.last_touch_bar = t
                    events[touch_event] = 1.0
                    if line_log is not None:
                        line_log.append(
                            ("touch", t, line.line_id, line.side, line.slope)
                        )
                line.in_band_prev = in_band
                keep.append(line)
        else:  # BROKEN — first re-entry lifecycle
            if t - line.break_bar > TRENDLINE_RETEST_WINDOW_BARS_V1:
                # The single re-entry opportunity has passed unresolved: this
                # IS the retest decision window and it owns its own named
                # constant (origin at the definition).  It is deliberately not
                # the ACTIVE staleness bound: widening it to seq_len would
                # silently redefine geomline_retest_{hold,fail}_{up,down},
                # whose only real measurement (the registered liveness floors
                # in entry_full_input_liveness_v1) was taken under exactly
                # this window.
                continue
            resolved = False
            flipped = False
            if line.break_dir < 0:
                if high >= proj - margin:  # re-entry from below
                    if close < proj:
                        events[_EV_RETEST_HOLD_DOWN] = 1.0
                        flipped = True
                    else:
                        # includes the exact tie: hold unproven -> FAIL
                        events[_EV_RETEST_FAIL_DOWN] = 1.0
                    resolved = True
            else:
                if low <= proj + margin:  # re-entry from above
                    if close > proj:
                        events[_EV_RETEST_HOLD_UP] = 1.0
                        flipped = True
                    else:
                        events[_EV_RETEST_FAIL_UP] = 1.0
                    resolved = True
            if flipped:
                # V30 package 8A (2026-08-13) — POLARITY FLIP.  The module
                # docstring has always called a RETEST_HOLD a "role flip", but
                # the line was DELETED on the very bar it proved itself as
                # flipped resistance/support, so the construct existed only as
                # a one-bar impulse and never as an object (fidelity audit
                # §5).  A held retest is exactly the chartist's "old support is
                # new resistance": the line still governs price, from the other
                # side.  The line therefore returns to ACTIVE with its
                # line_id, anchors, projection, touch history and max_dev_atr
                # preserved, and only its ``side`` — lifecycle state, not
                # identity — flipped.  The retest bar is counted as the touch
                # it is (same touch_bars/touch_count/last_touch_bar update the
                # intra-band touch path uses), which is also what stops the
                # staleness rule from retiring the flipped line on the next
                # bar; ``in_band_prev`` is set so the same excursion is not
                # counted twice.  The generic touch impulse is deliberately
                # NOT raised: the retest-hold event is this bar's specific
                # impulse for the same physical event.  No retest-window
                # constant exists; only the fitted band participates.
                line.side = (
                    TRENDLINE_SIDE_RESISTANCE
                    if line.side == TRENDLINE_SIDE_SUPPORT
                    else TRENDLINE_SIDE_SUPPORT
                )
                line.state = TRENDLINE_STATE_ACTIVE
                line.break_bar = -1
                line.break_dir = 0
                if t not in line.touch_bars:
                    line.touch_count += 1
                    line.touch_bars.add(t)
                line.last_touch_bar = t
                line.in_band_prev = True
                keep.append(line)
                if line_log is not None:
                    line_log.append(
                        ("touch", t, line.line_id, line.side, line.slope)
                    )
            elif not resolved:
                keep.append(line)
    if broken_this_bar:
        chosen = max(
            broken_this_bar, key=lambda ln: (ln.touch_count, -ln.line_id)
        )
        events[_EV_BREAK_LINE_TOUCH_COUNT] = float(chosen.touch_count)
        events[_EV_BREAK_LINE_AGE_BARS] = float(
            t - (chosen.anchor1_bar + swing_lookback)
        )
    state.active_lines[:] = keep


def _channel_block(
    state: TrendlineRegistryStateV1,
    above: TrendlineV1 | None,
    above_dist: float,
    below: TrendlineV1 | None,
    below_dist: float,
    t: int,
    close: float,
    atr_t: float,
    band_atr: float,
    seq_len: int,
) -> tuple[float, float, float, float, float, float]:
    margin = band_atr * atr_t
    # Route 1 — triangle/wedge: two ACTIVE opposite-side lines, strictly
    # converging (support slope > resistance slope), design B.4.
    if (
        above is not None
        and below is not None
        and below.side == TRENDLINE_SIDE_SUPPORT
        and above.side == TRENDLINE_SIDE_RESISTANCE
        and below.slope > above.slope
    ):
        proj_up = above.projection(t)
        proj_dn = below.projection(t)
        width_price = proj_up - proj_dn
        if width_price > 0.0:
            s_up = above.slope
            s_dn = below.slope
            c_up = above.anchor1_price - s_up * float(above.anchor1_bar)
            c_dn = below.anchor1_price - s_dn * float(below.anchor1_bar)
            apex_bar = (c_up - c_dn) / (s_dn - s_up)
            apex_dist = apex_bar - float(t)
            apex_proximity = min(max(1.0 - apex_dist / float(seq_len), 0.0), 1.0)
            pos = min(max((close - proj_dn) / width_price, 0.0), 1.0)
            return (
                1.0,
                width_price / atr_t,
                pos,
                ((s_up + s_dn) / 2.0) / atr_t,
                1.0,
                apex_proximity,
            )
    # Route 2 — parallel return rail through >=2 opposite-side stored pivots,
    # anchored on the most extreme opposite pivot; nearest slot line first
    # (tie: above first).
    ordered: list = []
    if above is not None:
        ordered.append((above_dist, 0, above))
    if below is not None:
        ordered.append((below_dist, 1, below))
    ordered.sort(key=lambda item: (item[0], item[1]))
    for _, _, line in ordered:
        s = line.slope
        r_line = line.anchor1_price - s * float(line.anchor1_bar)
        if line.side == TRENDLINE_SIDE_SUPPORT:
            opposite = state.resistance_pivots
        else:
            opposite = state.support_pivots
        offsets = [price - s * float(bar) for bar, price in opposite]
        if line.side == TRENDLINE_SIDE_SUPPORT:
            candidates = [r for r in offsets if r > r_line]
            if not candidates:
                continue
            r_ext = max(candidates)
            members = [r for r in candidates if (r_ext - r) <= margin]
        else:
            candidates = [r for r in offsets if r < r_line]
            if not candidates:
                continue
            r_ext = min(candidates)
            members = [r for r in candidates if (r - r_ext) <= margin]
        if len(members) < 2:
            continue
        width_price = abs(r_ext - r_line)
        if width_price <= 0.0:
            continue
        if line.side == TRENDLINE_SIDE_SUPPORT:
            lower_offset = r_line
        else:
            lower_offset = r_ext
        proj_dn = lower_offset + s * float(t)
        pos = min(max((close - proj_dn) / width_price, 0.0), 1.0)
        return (1.0, width_price / atr_t, pos, s / atr_t, 0.0, 0.0)
    return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _emit_row(
    state: TrendlineRegistryStateV1,
    t: int,
    close: float,
    atr_t: float,
    band_atr: float,
    seq_len: int,
    swing_lookback: int,
    events: np.ndarray,
) -> np.ndarray:
    row = np.zeros(TRENDLINE_REGISTRY_FEATURE_COUNT_V1, dtype=np.float64)
    above: TrendlineV1 | None = None
    below: TrendlineV1 | None = None
    above_key = (math.inf, math.inf)
    below_key = (math.inf, math.inf)
    # V30 graded occupancy: the per-side ACTIVE-line population for the
    # nearest-slot evidence below (same loop, same side rule:
    # projection-on-close ties to the ABOVE side).  A zero count is exactly
    # "no ACTIVE line on this side", which is what the retired
    # geomline_*_active masks used to say and nothing more.
    above_count = 0
    below_count = 0
    for line in state.active_lines:
        if line.state != TRENDLINE_STATE_ACTIVE:
            continue
        proj = line.projection(t)
        if proj >= close:
            above_count += 1
            key = (proj - close, float(line.line_id))
            if key < above_key:
                above_key = key
                above = line
        else:
            below_count += 1
            key = (close - proj, float(line.line_id))
            if key < below_key:
                below_key = key
                below = line
    row[_IX_ABOVE_BASE + _SLOT_ACTIVE_COUNT] = float(above_count)
    row[_IX_BELOW_BASE + _SLOT_ACTIVE_COUNT] = float(below_count)
    if above is not None:
        row[_IX_ABOVE_BASE + _SLOT_DIST_ATR] = above_key[0] / atr_t
        row[_IX_ABOVE_BASE + _SLOT_SLOPE_ATR_PER_BAR] = above.slope / atr_t
        row[_IX_ABOVE_BASE + _SLOT_TOUCH_COUNT] = float(above.touch_count)
        row[_IX_ABOVE_BASE + _SLOT_AGE_BARS] = float(
            t - (above.anchor1_bar + swing_lookback)
        )
        row[_IX_ABOVE_BASE + _SLOT_LAST_TOUCH_AGE_BARS] = float(
            t - above.last_touch_bar
        )
        row[_IX_ABOVE_BASE + _SLOT_MAX_DEV_ATR] = above.max_dev_atr
    if below is not None:
        row[_IX_BELOW_BASE + _SLOT_DIST_ATR] = below_key[0] / atr_t
        row[_IX_BELOW_BASE + _SLOT_SLOPE_ATR_PER_BAR] = below.slope / atr_t
        row[_IX_BELOW_BASE + _SLOT_TOUCH_COUNT] = float(below.touch_count)
        row[_IX_BELOW_BASE + _SLOT_AGE_BARS] = float(
            t - (below.anchor1_bar + swing_lookback)
        )
        row[_IX_BELOW_BASE + _SLOT_LAST_TOUCH_AGE_BARS] = float(
            t - below.last_touch_bar
        )
        row[_IX_BELOW_BASE + _SLOT_MAX_DEV_ATR] = below.max_dev_atr
    row[_IX_EVENT_BASE : _IX_EVENT_BASE + _EV_COUNT] = events
    row[_IX_BARS_SINCE_BREAK] = raw_event_age_from_last_observed_row(
        t, state.last_break_bar
    )
    row[
        _IX_CHANNEL_BASE : _IX_CHANNEL_BASE
        + len(TRENDLINE_REGISTRY_CHANNEL_FEATURE_NAMES_V1)
    ] = _channel_block(
        state,
        above,
        above_key[0],
        below,
        below_key[0],
        t,
        close,
        atr_t,
        band_atr,
        seq_len,
    )
    return row


# ---------------------------------------------------------------------------
# Public API — emission
# ---------------------------------------------------------------------------


def compute_trendline_registry_features_v1(
    df: pd.DataFrame,
    *,
    timeframe: str,
    seq_len: int,
    band_atr: float,
    swing_lookback: int = SWING_LOOKBACK,
    state: TrendlineRegistryStateV1 | None = None,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    atr_col: str = "atr",
    line_event_log: list | None = None,
    candidate_deviation_log: list[float] | None = None,
    candidate_observation_log: list[tuple] | None = None,
) -> tuple[pd.DataFrame, TrendlineRegistryStateV1]:
    """Compute the declared trendline/channel registry block for one TF.

    Incremental and chunk-safe: pass the returned state to the next call on
    the next contiguous chunk; chunked output is exactly equal to one-shot
    output.  ``state=None`` starts a fresh registry.  The state object is
    advanced in place and returned.

    ``line_event_log`` (optional) collects line-identified
    ``("touch", bar, line_id, side, slope)`` / ``("break", bar, line_id)``
    entries for the forward-realized label producer
    (:func:`compute_trendline_touch_hold_labels_v1`); it never changes the
    emitted features.

    ``candidate_deviation_log`` (optional) receives every candidate deviation
    evaluated by the runtime population, before the inclusive promotion test.
    It is an observation hook used by the TRAIN-only tolerance fitter and
    parity tests; it cannot alter state or emitted values.
    """
    timeframe = _require_timeframe(timeframe)
    seq_len = _require_positive_int("SEQ_LEN", seq_len)
    band_atr = _require_band(band_atr)
    swing_lookback = _require_positive_int("SWING_LOOKBACK", swing_lookback)
    high, low, close, atr = _validated_arrays(
        df, high_col, low_col, close_col, atr_col
    )
    _require_increasing_index(df)
    config_key = (
        TRENDLINE_REGISTRY_CONTRACT_V1,
        timeframe,
        seq_len,
        band_atr,
        swing_lookback,
        high_col,
        low_col,
        close_col,
        atr_col,
    )
    if state is None:
        state = TrendlineRegistryStateV1(config_key=config_key)
    else:
        if not isinstance(state, TrendlineRegistryStateV1):
            raise RuntimeError(f"[TRENDLINE_STATE_TYPE_INVALID] {type(state)!r}")
        if state.config_key != config_key:
            raise RuntimeError(
                "[TRENDLINE_STATE_CONFIG_MISMATCH] "
                f"state={state.config_key!r} call={config_key!r}"
            )
        if state.last_index is not None:
            try:
                contiguous = df.index[0] > state.last_index
            except TypeError as exc:
                raise RuntimeError(
                    "[TRENDLINE_STATE_INDEX_DISCONTINUITY]"
                ) from exc
            if not contiguous:
                raise RuntimeError("[TRENDLINE_STATE_INDEX_DISCONTINUITY]")
        if state.atr_seen and not np.isfinite(atr).all():
            raise RuntimeError("[TRENDLINE_STATE_ATR_GAP]")

    n = swing_lookback
    window = 2 * n + 1
    warmup_bars = 2 * n + 2
    n_rows = len(df)
    out = np.full(
        (n_rows, TRENDLINE_REGISTRY_FEATURE_COUNT_V1), np.nan, dtype=np.float64
    )
    for i in range(n_rows):
        t = state.bar_count
        # Receptive-field pruning precedes every possible consumer.  Doing
        # this only while ingesting a same-side pivot lets an expired
        # candidate promote before its later prune and leaves stale
        # opposite-side pivots visible to the parallel-channel route.
        _prune_receptive_field_state(state, t=t, seq_len=seq_len)
        state.buf_high.append(float(high[i]))
        state.buf_low.append(float(low[i]))
        state.buf_atr.append(float(atr[i]))
        if len(state.buf_high) > window:
            del state.buf_high[0]
            del state.buf_low[0]
            del state.buf_atr[0]
        atr_t = atr[i]
        atr_ok = math.isfinite(atr_t)
        if atr_ok:
            state.atr_seen = True
        events = np.zeros(_EV_COUNT, dtype=np.float64)
        # (1) pivot confirmation for bar j = t - n (support first).
        if len(state.buf_high) == window:
            pivot_bar = t - n
            pivot_atr = state.buf_atr[n]
            if math.isfinite(pivot_atr):
                sh_center, sl_center = _confirmed_center_pivots(
                    state.buf_high, state.buf_low, n
                )
                if sl_center:
                    _ingest_confirmed_pivot(
                        state,
                        TRENDLINE_SIDE_SUPPORT,
                        pivot_bar,
                        state.buf_low[n],
                        pivot_atr,
                        t,
                        band_atr,
                        seq_len,
                        events,
                        line_log=line_event_log,
                        candidate_deviation_log=candidate_deviation_log,
                        candidate_observation_log=candidate_observation_log,
                    )
                if sh_center:
                    _ingest_confirmed_pivot(
                        state,
                        TRENDLINE_SIDE_RESISTANCE,
                        pivot_bar,
                        state.buf_high[n],
                        pivot_atr,
                        t,
                        band_atr,
                        seq_len,
                        events,
                        line_log=line_event_log,
                        candidate_deviation_log=candidate_deviation_log,
                        candidate_observation_log=candidate_observation_log,
                    )
        # (2) per-line bar update.
        if atr_ok:
            _update_lines_for_bar(
                state,
                t,
                float(high[i]),
                float(low[i]),
                float(close[i]),
                float(atr_t),
                band_atr,
                n,
                seq_len,
                events,
                line_log=line_event_log,
            )
        # (3) emission — NaN during the declared warmup prefix.
        if atr_ok and t >= warmup_bars:
            out[i] = _emit_row(
                state,
                t,
                float(close[i]),
                float(atr_t),
                band_atr,
                seq_len,
                n,
                events,
            )
        state.bar_count += 1
        state.last_index = df.index[i]

    result = pd.DataFrame(
        out, index=df.index, columns=TRENDLINE_REGISTRY_FEATURE_NAMES_V1
    ).astype(np.float32)
    if tuple(result.columns) != TRENDLINE_REGISTRY_FEATURE_NAMES_V1:
        raise RuntimeError("[TRENDLINE_OUTPUT_ORDER_INVALID]")
    return result, state


# ---------------------------------------------------------------------------
# Public API — forward-realized line-hold LABELS (chart report B.8; design
# §5.2.8).  Labels may look forward; features never do.  This replaces the
# retired same-bar tautologies y_rising_channel_support_touch /
# y_falling_channel_resistance_touch.
# ---------------------------------------------------------------------------

TRENDLINE_TOUCH_HOLD_LABEL_COLUMNS_V1 = (
    "y_line_support_touch_mask",
    "y_line_support_touch_held",
    "y_line_resistance_touch_mask",
    "y_line_resistance_touch_held",
)


def compute_trendline_touch_hold_labels_v1(
    df: pd.DataFrame,
    *,
    seq_len: int,
    band_atr: float,
    horizon_bars: int,
    swing_lookback: int = SWING_LOOKBACK,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    atr_col: str = "atr",
) -> pd.DataFrame:
    """Forward-realized line-hold labels on real registry touch events.

    - ``y_line_support_touch_held``: defined (mask 1) on bars where a
      ``geomline_touch_below`` fired on an ACTIVE support-side line with
      slope > 0 (rising support) AND the full forward window of
      ``horizon_bars`` bars is observed; label 1 iff every such touched line
      is not BROKEN within the next ``horizon_bars`` bars (a same-bar
      multi-line touch aggregates conservatively: any break fails the hold).
    - ``y_line_resistance_touch_held``: exact mirror on
      ``geomline_touch_above`` with slope < 0 (falling resistance).
    - Off-event or undecidable rows carry mask 0 with label 0 — the
      established ``y_side_mask`` masking pattern (the mask, not the zero,
      carries the semantics; rule 2e).
    - ``horizon_bars`` must be the existing named forward first-N label
      horizon of the dataset builder (origin cited at the call site; this
      module invents no number).

    Runs one fresh full-history registry pass with the exact serving
    configuration, so the labelled lines are the same registry objects the
    emitted features describe.
    """

    if (
        isinstance(horizon_bars, bool)
        or not isinstance(horizon_bars, (int, np.integer))
        or int(horizon_bars) <= 0
    ):
        raise RuntimeError(f"[TRENDLINE_LABEL_HORIZON_INVALID] {horizon_bars!r}")
    horizon = int(horizon_bars)
    line_event_log: list = []
    _features, _state = compute_trendline_registry_features_v1(
        df,
        timeframe="M5",
        seq_len=seq_len,
        band_atr=band_atr,
        swing_lookback=swing_lookback,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        atr_col=atr_col,
        line_event_log=line_event_log,
    )
    n_rows = len(df)
    break_bars_by_line: dict[int, list[int]] = {}
    touches_support: dict[int, list[int]] = {}
    touches_resistance: dict[int, list[int]] = {}
    for entry in line_event_log:
        if entry[0] == "break":
            _kind, bar, line_id = entry
            # A held retest flips polarity and can make the same line identity
            # ACTIVE again, so every lifecycle break must remain observable.
            break_bars_by_line.setdefault(int(line_id), []).append(int(bar))
        else:
            _kind, bar, line_id, side, slope = entry
            if side == TRENDLINE_SIDE_SUPPORT and slope > 0.0:
                touches_support.setdefault(int(bar), []).append(int(line_id))
            elif side == TRENDLINE_SIDE_RESISTANCE and slope < 0.0:
                touches_resistance.setdefault(int(bar), []).append(int(line_id))
    out = {
        name: np.zeros(n_rows, dtype=np.float32)
        for name in TRENDLINE_TOUCH_HOLD_LABEL_COLUMNS_V1
    }
    for touch_map, mask_name, held_name in (
        (
            touches_support,
            "y_line_support_touch_mask",
            "y_line_support_touch_held",
        ),
        (
            touches_resistance,
            "y_line_resistance_touch_mask",
            "y_line_resistance_touch_held",
        ),
    ):
        for bar, line_ids in touch_map.items():
            if bar + horizon > n_rows - 1:
                # Outcome window not fully observed: undecidable, stays
                # masked out (rule 2e — no placeholder outcome).
                continue
            out[mask_name][bar] = 1.0
            held = all(
                not any(
                    bar < break_bar <= bar + horizon
                    for break_bar in break_bars_by_line.get(line_id, ())
                )
                for line_id in line_ids
            )
            out[held_name][bar] = 1.0 if held else 0.0
    frame = pd.DataFrame(out, index=df.index)
    return frame.loc[:, list(TRENDLINE_TOUCH_HOLD_LABEL_COLUMNS_V1)]


# ---------------------------------------------------------------------------
# Public API — TRAIN tolerance fit (design B.1)
# ---------------------------------------------------------------------------


def _collect_runtime_candidate_population(
    *,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    seq_len: int,
    swing_lookback: int,
) -> tuple[np.ndarray, dict[int, int], list[tuple]]:
    """Replay the exact, band-independent runtime CANDIDATE lifecycle.

    ACTIVE/BROKEN line evolution cannot affect candidate creation, pruning,
    promotion, or violation death, so promoted lines are discarded after each
    confirmation.  All candidate-affecting operations are delegated to the
    same helpers used by :func:`compute_trendline_registry_features_v1`.
    """

    state = TrendlineRegistryStateV1(
        config_key=("TRENDLINE_TOLERANCE_FIT_RUNTIME_POPULATION",)
    )
    n = swing_lookback
    window = 2 * n + 1
    deviations: list[float] = []
    observations: list[tuple] = []
    pivot_counts = {
        TRENDLINE_SIDE_SUPPORT: 0,
        TRENDLINE_SIDE_RESISTANCE: 0,
    }
    for t in range(len(high)):
        _prune_receptive_field_state(state, t=t, seq_len=seq_len)
        state.buf_high.append(float(high[t]))
        state.buf_low.append(float(low[t]))
        state.buf_atr.append(float(atr[t]))
        if len(state.buf_high) > window:
            del state.buf_high[0]
            del state.buf_low[0]
            del state.buf_atr[0]
        if len(state.buf_high) == window:
            pivot_bar = t - n
            pivot_atr = state.buf_atr[n]
            if math.isfinite(pivot_atr):
                sh_center, sl_center = _confirmed_center_pivots(
                    state.buf_high,
                    state.buf_low,
                    n,
                )
                events = np.zeros(_EV_COUNT, dtype=np.float64)
                if sl_center:
                    _ingest_confirmed_pivot(
                        state,
                        TRENDLINE_SIDE_SUPPORT,
                        pivot_bar,
                        state.buf_low[n],
                        pivot_atr,
                        t,
                        0.0,
                        seq_len,
                        events,
                        candidate_deviation_log=deviations,
                        candidate_observation_log=observations,
                    )
                    pivot_counts[TRENDLINE_SIDE_SUPPORT] += 1
                if sh_center:
                    _ingest_confirmed_pivot(
                        state,
                        TRENDLINE_SIDE_RESISTANCE,
                        pivot_bar,
                        state.buf_high[n],
                        pivot_atr,
                        t,
                        0.0,
                        seq_len,
                        events,
                        candidate_deviation_log=deviations,
                        candidate_observation_log=observations,
                    )
                    pivot_counts[TRENDLINE_SIDE_RESISTANCE] += 1
                # Promoted ACTIVE lines have no feedback into the candidate
                # population.  Dropping them avoids paying unrelated O(L)
                # serve cost in a TRAIN statistic owner.
                state.active_lines.clear()
        if math.isfinite(atr[t]):
            _discard_violated_candidates_for_bar(
                state,
                t=t,
                close=float(close[t]),
            )
    return np.asarray(deviations, dtype=np.float64), pivot_counts, observations


def collect_trendline_threshold_outcome_stream_v1(
    df: pd.DataFrame,
    *,
    seq_len: int,
    swing_lookback: int = SWING_LOOKBACK,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    atr_col: str = "atr",
) -> RegistryOutcomeStreamV1:
    """Build the threshold-independent TRAIN outcome population.

    Each two-anchor candidate receives exactly one observation when the next
    same-side confirmed pivot tests it.  Its distance is already available on
    that closed confirmation bar.  Starting on the following bar, the first
    exact geometric event is recorded:

    * support: ``close < projection`` is BREAK; a whole bar strictly above
      the projection (``low > projection``) is REACTION;
    * resistance: the exact mirror.

    These zero-width boundaries are algebraic properties of the candidate
    line, not fitted thresholds.  An unresolved observation is right-censored
    at the declared TRAIN end.  The future event rows are consumed only by the
    TRAIN hyperparameter fitter and never by feature computation.
    """

    seq_len = _require_positive_int("SEQ_LEN", seq_len)
    swing_lookback = _require_positive_int("SWING_LOOKBACK", swing_lookback)
    high, low, close, atr = _validated_arrays(
        df, high_col, low_col, close_col, atr_col
    )
    _require_increasing_index(df)
    _sample, _pivot_counts, observations = _collect_runtime_candidate_population(
        high=high,
        low=low,
        close=close,
        atr=atr,
        seq_len=seq_len,
        swing_lookback=swing_lookback,
    )
    resolved: list[tuple[int, float, int, str, int]] = []
    for origin, line_id, side, slope, intercept, deviation in observations:
        if int(origin) >= len(df) - 1:
            continue
        # ``deviation`` is ``|pivot_price - projection| / pivot_atr``, so it is
        # exactly 0.0 when a pivot lands on the projected line.  That is a real
        # geometric coincidence, not an error -- but it cannot enter a
        # *tolerance* fit population.  The fitter's candidate support is every
        # distinct observed distance, so a 0.0 observation is itself selectable
        # as the threshold, and a fitted trendline tolerance of 0.0 ATR would
        # admit a touch only on exact float equality, silently killing the
        # feature at serve.  ``level_registry_v1`` already excludes
        # non-positive and non-finite distances for this reason; both owners of
        # ``RegistryOutcomeStreamV1`` must agree, and the contract requires
        # strictly positive distances (rule 25: two internally consistent
        # owners disagreeing is exactly what no gate can see).
        if not math.isfinite(float(deviation)) or float(deviation) <= 0.0:
            continue
        # An observation that never resolves before the declared TRAIN end is
        # right-censored, which is exactly what the docstring above promises
        # and what the shared contract requires: ``event_row == -1`` is only
        # legal with ``REGISTRY_OUTCOME_CENSORED``.  Leaving the REACTION
        # placeholder here emitted an unresolved row as a resolved reaction at
        # row -1, which ``require_registry_outcome_stream_v1`` rejects with the
        # unrelated ``REGISTRY_OUTCOME_STREAM_INVALID``.  Measured 2026-08-15:
        # zero censored observations on all six declared clocks, so no emitted
        # value and no fitted hash moves — the defect is latent, not active,
        # and this closes it before it can fire.
        event_row = -1
        event_cause = REGISTRY_OUTCOME_CENSORED
        for row in range(int(origin) + 1, len(df)):
            projection = float(intercept) + float(slope) * float(row)
            if side == TRENDLINE_SIDE_SUPPORT:
                if float(close[row]) < projection:
                    event_row = row
                    event_cause = REGISTRY_OUTCOME_BREAK
                    break
                if float(low[row]) > projection:
                    event_row = row
                    event_cause = REGISTRY_OUTCOME_REACTION
                    break
            else:
                if float(close[row]) > projection:
                    event_row = row
                    event_cause = REGISTRY_OUTCOME_BREAK
                    break
                if float(high[row]) < projection:
                    event_row = row
                    event_cause = REGISTRY_OUTCOME_REACTION
                    break
        resolved.append(
            (
                int(origin),
                float(deviation),
                int(event_row),
                event_cause,
                int(line_id),
            )
        )
    # Generic fit population order is (origin, distance, identity).  Sorting
    # here makes chunk/replay equality independent of candidate-array layout.
    resolved.sort(key=lambda item: (item[0], item[1], item[4]))
    return RegistryOutcomeStreamV1(
        origin_row=np.asarray([item[0] for item in resolved], dtype=np.int64),
        distance_atr=np.asarray([item[1] for item in resolved], dtype=np.float64),
        event_row=np.asarray([item[2] for item in resolved], dtype=np.int64),
        event_cause=tuple(item[3] for item in resolved),
    )


def fit_trendline_registry_hyperparameters_v1(
    df: pd.DataFrame,
    *,
    timeframe: str,
    seq_len: int,
    inner_fit_end_exclusive: int,
    source_provenance: Mapping[str, Any],
    swing_lookback: int = SWING_LOOKBACK,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    atr_col: str = "atr",
) -> dict[str, Any]:
    """Fit band + event-time lifecycle on chronological inner-TRAIN.

    This owner has no median or other operator-selected quantile. It evaluates
    every exact empirical candidate
    distance through the shared competing-risk likelihood contract.  The
    returned payload is the artifact payload; callers must write/load it
    through ``registry_hyperparameter_fit_v1`` before apply.
    """

    timeframe = _require_timeframe(timeframe)
    seq_len = _require_positive_int("SEQ_LEN", seq_len)
    swing_lookback = _require_positive_int("SWING_LOOKBACK", swing_lookback)
    high, low, close, atr = _validated_arrays(
        df, high_col, low_col, close_col, atr_col
    )
    _require_increasing_index(df)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise RuntimeError("[TRENDLINE_FIT_DATETIME_INDEX_REQUIRED]")
    stream = collect_trendline_threshold_outcome_stream_v1(
        df,
        seq_len=seq_len,
        swing_lookback=swing_lookback,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        atr_col=atr_col,
    )
    atr_available = np.isfinite(atr).astype(np.float64)
    atr_bound = np.where(np.isfinite(atr), atr, 0.0)
    payload = fit_registry_competing_risk_threshold_v1(
        stream,
        registry_kind="trendline",
        clock=timeframe,
        n_rows=len(df),
        inner_fit_end_exclusive=inner_fit_end_exclusive,
        index_ns=df.index.asi8,
        frame_columns=(high, low, close, atr_bound, atr_available),
        source_provenance=source_provenance,
        population_configuration={
            "owner": "trendline_exact_runtime_candidate_population_v1",
            "seq_len": int(seq_len),
            "swing_lookback": int(swing_lookback),
        },
    )
    # The runtime population is described by the two values that actually
    # define it — the declared receptive field and the pivot look-around.
    # ``learned_expiry_bars`` stays an internally validated payload key of the
    # shared fit contract, but it is no longer injected here and no longer
    # consumed by the registry: it was re-declared as an identity lifetime on
    # 2026-08-15 (a94f5c6e) and, being <= SWING_LOOKBACK, killed every line on
    # its promotion bar.  It measures bars-to-projection-break, not bars since
    # a promoted identity was last touched (rule 2g: different quantity,
    # different population).
    return require_registry_hyperparameter_payload_v1(
        payload,
        registry_kind="trendline",
        clock=timeframe,
    )
