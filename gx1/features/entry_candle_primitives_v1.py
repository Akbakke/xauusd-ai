"""Raw causal candle geometry shared by local Entry/Exit and every MTF lane.

The owner deliberately emits no named candlestick pattern and contains no
market threshold, vote or direction decision.  A temporal model can learn
doji, rejection, engulfing, inside/outside and multi-bar patterns from these
continuous measurements without inheriting a hand-written definition.

Row ``t`` describes the bar that closes at that row's authoritative close.
Relational fields need row ``t-1`` and therefore preserve one honest NaN
prefix.  On a zero-range (``high == low``) bar every range SHARE is
mathematically undefined, so the shares are the storage zero.  The separate
``candle.raw_zero_range_flag`` that used to mark that encoding was retired on
2026-08-15; it was an exact algebraic function of three shares that remain
model inputs.  The reason, the identity and the measured event counts are in
the ``CANDLE_PRIMITIVE_FEATURE_VERSION`` note below, together with the four
V30 wave-2 retirements that followed the same rule.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


# v3 (2026-08-15): ``candle.raw_zero_range_flag`` is RETIRED, here and on
# every MTF lane (``mtf_candle_raw_zero_range_flag``).
#
# MEASURED on the complete declared native M5 tape
# ``XAU_M5_NATIVE_2019_20260804_V4`` (537,861 rows, 2019-01-01..2026-08-04),
# resampled by the declared per-timeframe owner and counted after the >=199-row
# causal warmup: 215 zero-range (``high == low``) bars on M5 (0.040%), 24 on
# M15 (0.013%), 14 on H1 (0.031%), and ZERO on H4 and D1.  Two independent
# failure modes follow, neither of them fixable:
#
# 1. Constant post-warmup on H4 and D1, so
#    ``htf_features.build_multi_tf_v4_liveness_contract`` fails closed on
#    ``unique_count <= 1`` and ``prebuild_multi_tf_cache_v4`` turns that into a
#    hard ``HTF_V4_CACHE_FULL_INPUT_LIVENESS_FAIL``.  Gold prints no zero-range
#    4-hour or daily bar; that is a market fact, not a wiring defect, and no
#    window or warmup choice changes it.
# 2. Declaring it an allowed constant does NOT rescue it.  PROVEN FROM SOURCE
#    in ``gx1/contracts/entry_model_native_input_normalization_v1.py``: the
#    ``is_binary`` identity fast path requires the column to contain BOTH 0.0
#    and 1.0, so a constant-0 column falls through to the IQR path where
#    ``q75 - q25 == 0``, the positive-absolute-deviation fallback is empty, and
#    ``[ENTRY_INPUT_NORMALIZATION_UNSCALEABLE]`` is raised.  An allowlist entry
#    only moves the failure one stage downstream.
#
# On the lanes where it is not constant the rate is an order of magnitude
# below the ``MIN_ACTIVE_RATE`` = 0.01 floor of
# ``gx1/contracts/entry_full_input_liveness_v1.py``, and the M15/H1 counts (24,
# 14) are already under the smallest floor that owner registers (32); a 9-month
# TRAIN leaves ~19 M5 events, also under it.  Registering a lower floor would
# be inventing a magnitude, which CLAUDE.md rule 2b forbids.
#
# CLAUDE.md rule 4 is satisfied twice over.  The flag consumed only ``high``
# and ``low`` of its own bar, and both remain model inputs on every lane — but
# the stronger statement is that NO evidence at all is lost, because the flag
# is an EXACT algebraic function of three shares that stay on the surface.
#
# PROVEN FROM SOURCE AND ALGEBRA, independent of any data.  For every bar,
# ``upper_wick + lower_wick + |body| == high - low == R`` identically
# (``upper_wick = high - max(open, close)``, ``lower_wick = min(open, close) -
# low``, ``|body| = |close - open|``; the three are the exact partition of the
# range).  When ``R > 0`` this owner emits those three quantities divided by
# ``R``, so ``|body_signed_range| + upper_wick_share + lower_wick_share == 1``
# and at least one of the three is >= 1/3 — none of them can round to zero
# under the float32 cast.  When ``R == 0`` the owner emits all three as the
# storage zero.  Therefore
#
#     high == low   <==>   body_signed_range == 0
#                          and upper_wick_share == 0
#                          and lower_wick_share == 0
#
# is an exact biconditional over the surviving inputs, on every lane and at
# every timeframe.  MEASURED corroboration on 200,000 synthetic rows seeded
# with zero-range, marubozu and extreme-touching bars: the recovered mask
# equals ``high == low`` on every row, and the share sum is 1.0 +/- 4.5e-8 on
# positive-range rows against exactly 0.0 on zero-range rows.  A doji is NOT
# confusable with a zero-range bar: a real-range doji has
# ``body_signed_range == 0`` but nonzero wick shares summing to 1.
# ``candle.raw_close_location``'s neutral 0.5 was likewise disambiguated by
# the same conjunction; that field is itself RETIRED in v4 below, so no
# encoding is left ambiguous and none is left unread.
# v4 (2026-08-18, V30 wave 2): four RELATIONAL/whole-bar columns are RETIRED
# because each is an EXACT function of columns that stay on this surface, in
# this same owner and therefore in the same specialist family.  Nothing is
# measured away; the model forms the same quantity if the quantity helps.
# Every statement below is PROVEN FROM SOURCE AND ALGEBRA at this commit and
# holds independent of any data (CLAUDE.md rule 2c/2d).  The denominators are
# untouched: ``local_geometry_scale`` still carries its
# ``abs(bar_range - previous_range)`` term, so every surviving relational
# column keeps a bit-identical value.
#
# 1. ``candle.raw_close_location`` = ``lower_wick_share + max(body_signed_range, 0)``
#    On a positive-range bar, with ``R = high - low``:
#      close >= open  ->  lower = (open-low)/R, max(body,0) = (close-open)/R,
#                         sum = (close-low)/R = close_location
#      close <  open  ->  lower = (close-low)/R = close_location, max(body,0)=0
#    The ONLY rows the identity misses are zero-range bars, where the retired
#    field emitted a hard-coded 0.5 for an undefined 0/0 — a placeholder that
#    reads as "close at mid-range" and carries no observation at all
#    (rule 2e).  Those rows are exactly the rows the surviving biconditional in
#    the v3 note already identifies, so removing the placeholder removes an
#    ambiguity rather than an observation.
#
# 2. ``candle.raw_range_change_local_geometry`` = ``high_change - low_change``
#    ``(R[t] - R[t-1]) = (high[t]-high[t-1]) - (low[t]-low[t-1])`` identically,
#    and all three columns divide by the SAME ``local_geometry_scale`` scalar
#    of that row and share the same NaN prefix.  This is an AFFINE recovery —
#    the strongest class — so it is formable by the per-family linear
#    projection itself, not merely by the encoder above it.
#
# 3. ``candle.raw_high_rejection_depth_local_geometry``
#      = ``high_change_local_geometry * high_rejection_previous_high_event``
#    The emitted depth IS ``(high[t]-high[t-1])/local_geometry_scale`` on an
#    event bar and 0.0 otherwise, i.e. the bit-identical product of two
#    retained columns.  The event column is exactly binary, so the two
#    branches are separated by a unit gap and the recovery is Lipschitz.  The
#    direction of the recovery matters: the event is NOT recoverable from the
#    depth (``1[depth > 0]`` would be a sign test at 0 on a dense field), which
#    is why the flag stays and the magnitude goes.
#
# 4. ``candle.raw_low_rejection_depth_local_geometry``
#      = ``-1 * low_change_local_geometry * low_rejection_previous_low_event``
#    Same statement mirrored; the emitted depth is
#    ``(low[t-1]-low[t])/local_geometry_scale``, the sign flip against
#    ``low_change`` being the only difference.  Negation is affine.
#
# WHAT IS NOT RETIRED, and why the neighbouring near-misses stay:
#   * ``candle.raw_upper_wick_share`` looks recoverable as
#     ``1 - lower_wick_share - abs(body_signed_range)``, and on a POSITIVE
#     range bar it is.  It stays because of the zero-range branch: a gravestone
#     bar with ``open == close == low < high`` emits body = 0 and lower = 0,
#     which is the identical pair a zero-range bar emits, and ONLY
#     ``upper_wick_share`` separates them (1 against 0).  The recovery
#     ``1 - lower - |body|`` would hand a zero-range bar the value 1, which is
#     the gravestone's answer, not its own.  The v3 retirement of
#     ``candle.raw_zero_range_flag`` rests on the biconditional over all THREE
#     shares, so removing one of the three would silently withdraw that
#     retirement's proof as well.
#   * ``candle.raw_range_contains_previous_flag`` and
#     ``..._range_contained_by_previous_flag`` stay for two independent
#     reasons.  (a) STATE: ``candle.raw_observed_range_relation_duration_bars``
#     is an emitted run length over the four-state label {2, 1, -1, 0} whose
#     identity lives ONLY in these two columns; retiring them orphans a
#     retained emitted field.  (b) CONDITIONING: the recovery is three sign
#     tests at exactly 0 on two continuous ratios with no integer witness, and
#     the pair already cannot separate state 2 from state 0 (both press 0/0),
#     so it is a lossy encoding already — dropping one half collapses a
#     four-state variable to two.
# v5 (2026-08-19, era-proxy repair): ``candle.raw_open_gap_local_geometry`` is
# REPLACED, name and value together, by
# ``candle.raw_open_position_previous_range``.  The retired name is spelled
# here, in the version note that exists to record it, exactly as the v3 and v4
# retirements above are; ``test_open_gap_era_proxy_is_retired_and_repaired`` in
# tests/test_entry_candle_primitives.py asserts it is absent from the declared
# tuples and from the emitted index, which is where absence is enforceable.
#
# WHAT WAS WRONG.  MEASURED on the complete declared native M5 tape
# ``XAU_M5_NATIVE_2019_20260804_V4`` (537,861 rows, 2019-01-01..2026-08-04,
# 537,860 after the one-row relational NaN prefix), IQR by calendar year of the
# emitted float32 column:
#
#   2019 0.13435 | 2020 0.06995 | 2021 0.08517 | 2022 0.04964
#   2023 0.05382 | 2024 0.02075 | 2025 0.01407 | 2026 0.00678
#
# Monotone after 2021 and 19.8x from 2019 to 2026; on the tape's first/last
# third the IQR ratio is 0.1512 and the median reaches EXACTLY 0.  It was the
# single most extreme structural break on the per-timeframe surface.  With the
# TRAIN window widening from one year to five, a fitted IQR of ~0.03 would map
# 2021 rows to roughly +/-3 and 2026 rows to roughly +/-0.2: a calendar clock
# handed to the model in a column named for candle geometry.
#
# WHY THE DENOMINATOR WAS NOT THE CAUSE — this is the part that had to be
# measured before anything was changed, and it REFUTES the obvious repair.
# The numerator ``open[t] - close[t-1]`` is the price move across the bin
# seam, and its own magnitude is set by the FEED, not by the market: median
# ``|open - close[t-1]|`` on rows where it is nonzero runs 0.042, 0.046, 0.046,
# 0.045, 0.040 USD for 2019-2023 and then steps to 0.015, 0.020, 0.020 for
# 2024-2026, while median M5 bar range over the same years runs 0.524, 1.248,
# 0.938, 1.040, 0.915, 1.335, 2.485, 5.100 USD.  A near-constant microstructure
# quantum divided by a volatility that grew 9.7x is an inverse-volatility clock
# whatever the volatility estimator is, and all three natural denominators
# measure the same collapse: last/first-year IQR ratio 0.0505 for
# ``local_geometry_scale``, 0.0488 for the bar range and 0.0503 for a Wilder-14
# ATR.  Two further hypotheses were tested and refuted on the same bytes: the
# column is not dying into a zero mass (the gap is nonzero on 92.2%-99.98% of
# rows in every year) and it is not saturating on its own denominator term
# (``|gap| >= local_geometry_scale`` on 0.02%-0.06% of rows).
#
# WHAT THE REPAIR IS.  The numerator changes from a seam DIFFERENCE to a
# POSITION, so that both sides of the ratio scale with volatility:
#
#     open_position_previous_range[t] = (open[t] - low[t-1]) / (high[t-1] - low[t-1])
#
# No magnitude is invented (rule 2b).  The denominator is a bar's own range,
# which is the convention this same owner already uses for its entire whole-bar
# block (``body_signed_range``, ``upper_wick_share``, ``lower_wick_share`` all
# divide by ``high - low``), and ``high[t-1] - low[t-1]`` is already one of the
# terms inside ``local_geometry_scale``.  On a zero-range previous bar the ratio
# is an undefined 0/0 and the column takes the STORAGE ZERO — the identical
# convention, and the identical justification, as the three shares: those rows
# are exactly the rows the v3 biconditional identifies from the previous row's
# three shares, so the zero is disambiguated rather than a placeholder that
# reads as a result (rule 2e).  MEASURED: 215 such rows on the whole M5 tape,
# and zero on the H4 and D1 lanes.
#
# MEASURED after the repair, same tape, same population, IQR by year:
#
#   2019 0.61804 | 2020 0.58103 | 2021 0.59646 | 2022 0.58781
#   2023 0.58201 | 2024 0.57287 | 2025 0.55453 | 2026 0.55411
#
# ratio 0.8966 first-to-last year, 0.9401 on the first/last third, median flat
# at 0.506-0.525.  Stated plainly: that is NOT the 0.966-1.064 band the nine
# unit repairs earlier in this wave reached.  It is the band the untouched
# members of this same family already sit in on the same tape
# (``close_change_local_geometry`` 0.9015, ``high_change_local_geometry``
# 0.8957, ``body_signed_range`` 0.8833), i.e. the residual common to the family
# rather than a defect of this column, and it is a factor 17.8 better than the
# 0.0505 it replaces.
#
# RESIDUAL, stated uninvited (rule 25a).  The repaired column's TAILS still
# carry the feed: the fraction of rows opening outside the previous bar's range
# falls monotonically from 7.6%/7.5% (above/below) in 2019 to 1.0%/0.9% in
# 2026, because a shrinking seam puts the open inside the previous envelope
# more often.  The IQR and the median — the statistics TRAIN normalization
# fits — are flat; the tail mass is not.  The same physics makes
# ``candle.raw_open_above_previous_high_local_geometry`` and
# ``candle.raw_open_below_previous_low_local_geometry`` era proxies in their
# own right (activation 0.0768 -> 0.0100 and 0.0761 -> 0.0087, nonzero median
# 0.0706 -> 0.0082 and 0.0721 -> 0.0057).  They are NOT repaired here: they are
# now exact indicator functions of this column
# (``above > 0`` <=> position > 1, ``below > 0`` <=> position < 0) and their
# retirement is a surface-width decision for the signal-contract owner, not a
# candle-owner edit.
#
# RULE 4 — no market evidence leaves the learned path.  The retired reading is
# EXACTLY recoverable from this column plus columns that remain model inputs,
# in previous-range units, using the identity this owner already proved for the
# retired ``candle.raw_close_location`` (see v4 note 1):
#
#     (open[t] - close[t-1]) / (high[t-1] - low[t-1])
#         = open_position_previous_range[t]
#           - ( lower_wick_share[t-1] + max(body_signed_range[t-1], 0) )
#
# both terms on the right being emitted columns of this owner, the second one
# read from the immediately preceding row that every sequence consumer already
# sees.  MEASURED on the 537,645 rows of that tape with a positive previous
# range: in float64 the identity holds to 7.1e-15 (102,443 rows bitwise equal),
# and carried out on the EMITTED float32 columns as a consumer would, to
# 1.575e-06 -- roughly 13 float32 ULP at unit magnitude, which is the cost of
# the three storage roundings the recovery passes through and not a gap in the
# algebra.  What is NOT
# recoverable is the retired column's own ``local_geometry_scale`` unit, since
# ``(high[t-1] - low[t-1]) / local_geometry_scale`` is not an emitted column
# (measured p1 0.2363, p50 1.0000, exactly 1.0 on 50.92% of rows) — the unit is
# gone, the observation is not, which is the same trade the nine unit repairs
# earlier in this wave made.
#
# ``local_geometry_scale`` itself is UNTOUCHED, including its
# ``abs(open[t] - close[t-1])`` term, so every surviving column that divides by
# it keeps a bit-identical value.  MEASURED, not assumed: running the pre-v5
# owner and this one over the whole declared M5 tape, all 20 shared columns are
# bit-identical (NaN prefix included) and the renamed column occupies the same
# slot, so the emitted WIDTH and the column order are unchanged and exactly one
# column moved.
CANDLE_PRIMITIVE_FEATURE_VERSION = (
    "entry_candle_primitives_v5_20260819_open_position_previous_range"
)
CANDLE_PRIMITIVE_FEATURE_PREFIX = "candle.raw_"
CANDLE_PRIMITIVE_SOURCE_FIELDS = ("time", "open", "high", "low", "close")
# Whole-bar geometry: row ``t`` alone, finite from the first row, emitted as
# whole vectorized columns.
CANDLE_PRIMITIVE_WHOLE_BAR_FEATURE_NAMES = (
    "candle.raw_body_signed_range",
    "candle.raw_upper_wick_share",
    "candle.raw_lower_wick_share",
)
# Relational/carry block: emitted from the two-bar carry loop, therefore
# preceded by the one honest NaN prefix row (or continued exactly from a carry
# state).
CANDLE_PRIMITIVE_RELATIONAL_FEATURE_NAMES = (
    # Where this bar OPENED inside the previous bar's own range: 0 at the
    # previous low, 1 at the previous high, outside [0, 1] on a genuine gap.
    # Denominator is that bar's range, the convention the whole-bar block above
    # already uses; storage zero on a zero-range previous bar.  See the v5 note
    # at CANDLE_PRIMITIVE_FEATURE_VERSION for the era measurement that retired
    # the seam-difference form and for the exact rule-4 recovery identity.
    "candle.raw_open_position_previous_range",
    # These five use the exact two-bar geometry envelope (ranges and absolute
    # OHLC/body displacements).  The retired *_prev_range spelling falsely
    # claimed an exact previous-range denominator, including after a
    # zero-range bar.
    "candle.raw_close_change_local_geometry",
    "candle.raw_high_change_local_geometry",
    "candle.raw_low_change_local_geometry",
    "candle.raw_body_change_local_geometry",
    # Exact two-bar geometry. Magnitudes are zero off-event and accompanied
    # by either an algebraically readable sign or an explicit flag.
    "candle.raw_open_above_previous_high_local_geometry",
    "candle.raw_open_below_previous_low_local_geometry",
    "candle.raw_body_overlap_previous_local_geometry",
    "candle.raw_body_contains_previous_flag",
    "candle.raw_body_contained_by_previous_flag",
    "candle.raw_range_contains_previous_flag",
    "candle.raw_range_contained_by_previous_flag",
    "candle.raw_bull_body_covers_previous_bear_body_event",
    "candle.raw_bear_body_covers_previous_bull_body_event",
    "candle.raw_high_rejection_previous_high_event",
    "candle.raw_low_rejection_previous_low_event",
    # Raw observed-bar duration of the current exact state.  No cap, tanh or
    # threshold is applied; TRAIN-only input normalization owns scale.
    "candle.raw_observed_body_direction_duration_bars",
    "candle.raw_observed_range_relation_duration_bars",
)
CANDLE_PRIMITIVE_FEATURE_NAMES = (
    CANDLE_PRIMITIVE_WHOLE_BAR_FEATURE_NAMES
    + CANDLE_PRIMITIVE_RELATIONAL_FEATURE_NAMES
)
CANDLE_PRIMITIVE_FEATURE_NAMES_SHA256 = hashlib.sha256(
    "\n".join(CANDLE_PRIMITIVE_FEATURE_NAMES).encode("utf-8")
).hexdigest()


# ---------------------------------------------------------------------------
# Emitted-column layout, DERIVED from the name tuples above rather than
# restated as literals (CLAUDE.md rule 13: a repeated literal is not
# ownership).  The hand-numbered ``matrix[:, 0] .. matrix[:, 25]`` writes this
# replaces had to be renumbered by hand every time a field entered or left the
# tuple, and a missed renumber is SILENT: every column keeps its dtype, its
# finiteness prefix and its name, so two fields swap meanings without any
# downstream gate being able to see it.  Each constant is a lookup into the
# declared tuple, so a removed name raises ``ValueError`` at import.
# ---------------------------------------------------------------------------
_IX_BODY_SIGNED_RANGE = CANDLE_PRIMITIVE_FEATURE_NAMES.index(
    "candle.raw_body_signed_range"
)
_IX_UPPER_WICK_SHARE = CANDLE_PRIMITIVE_FEATURE_NAMES.index(
    "candle.raw_upper_wick_share"
)
_IX_LOWER_WICK_SHARE = CANDLE_PRIMITIVE_FEATURE_NAMES.index(
    "candle.raw_lower_wick_share"
)
_IX_OPEN_POSITION_PREVIOUS_RANGE = CANDLE_PRIMITIVE_FEATURE_NAMES.index(
    "candle.raw_open_position_previous_range"
)
_IX_CLOSE_CHANGE = CANDLE_PRIMITIVE_FEATURE_NAMES.index(
    "candle.raw_close_change_local_geometry"
)
_IX_HIGH_CHANGE = CANDLE_PRIMITIVE_FEATURE_NAMES.index(
    "candle.raw_high_change_local_geometry"
)
_IX_LOW_CHANGE = CANDLE_PRIMITIVE_FEATURE_NAMES.index(
    "candle.raw_low_change_local_geometry"
)
_IX_BODY_CHANGE = CANDLE_PRIMITIVE_FEATURE_NAMES.index(
    "candle.raw_body_change_local_geometry"
)
_IX_OPEN_ABOVE_PREVIOUS_HIGH = CANDLE_PRIMITIVE_FEATURE_NAMES.index(
    "candle.raw_open_above_previous_high_local_geometry"
)
_IX_OPEN_BELOW_PREVIOUS_LOW = CANDLE_PRIMITIVE_FEATURE_NAMES.index(
    "candle.raw_open_below_previous_low_local_geometry"
)
_IX_BODY_OVERLAP_PREVIOUS = CANDLE_PRIMITIVE_FEATURE_NAMES.index(
    "candle.raw_body_overlap_previous_local_geometry"
)
_IX_BODY_CONTAINS_PREVIOUS = CANDLE_PRIMITIVE_FEATURE_NAMES.index(
    "candle.raw_body_contains_previous_flag"
)
_IX_BODY_CONTAINED_BY_PREVIOUS = CANDLE_PRIMITIVE_FEATURE_NAMES.index(
    "candle.raw_body_contained_by_previous_flag"
)
_IX_RANGE_CONTAINS_PREVIOUS = CANDLE_PRIMITIVE_FEATURE_NAMES.index(
    "candle.raw_range_contains_previous_flag"
)
_IX_RANGE_CONTAINED_BY_PREVIOUS = CANDLE_PRIMITIVE_FEATURE_NAMES.index(
    "candle.raw_range_contained_by_previous_flag"
)
_IX_BULL_BODY_COVERS_PREVIOUS_BEAR = CANDLE_PRIMITIVE_FEATURE_NAMES.index(
    "candle.raw_bull_body_covers_previous_bear_body_event"
)
_IX_BEAR_BODY_COVERS_PREVIOUS_BULL = CANDLE_PRIMITIVE_FEATURE_NAMES.index(
    "candle.raw_bear_body_covers_previous_bull_body_event"
)
_IX_HIGH_REJECTION_EVENT = CANDLE_PRIMITIVE_FEATURE_NAMES.index(
    "candle.raw_high_rejection_previous_high_event"
)
_IX_LOW_REJECTION_EVENT = CANDLE_PRIMITIVE_FEATURE_NAMES.index(
    "candle.raw_low_rejection_previous_low_event"
)
_IX_BODY_DIRECTION_DURATION = CANDLE_PRIMITIVE_FEATURE_NAMES.index(
    "candle.raw_observed_body_direction_duration_bars"
)
_IX_RANGE_RELATION_DURATION = CANDLE_PRIMITIVE_FEATURE_NAMES.index(
    "candle.raw_observed_range_relation_duration_bars"
)

# The complete write inventory, listed in declared-emission order.  Every
# assignment in :func:`compute_entry_candle_primitive_chunk` targets exactly
# one of these constants.
_EMITTED_COLUMN_INDICES = (
    _IX_BODY_SIGNED_RANGE,
    _IX_UPPER_WICK_SHARE,
    _IX_LOWER_WICK_SHARE,
    _IX_OPEN_POSITION_PREVIOUS_RANGE,
    _IX_CLOSE_CHANGE,
    _IX_HIGH_CHANGE,
    _IX_LOW_CHANGE,
    _IX_BODY_CHANGE,
    _IX_OPEN_ABOVE_PREVIOUS_HIGH,
    _IX_OPEN_BELOW_PREVIOUS_LOW,
    _IX_BODY_OVERLAP_PREVIOUS,
    _IX_BODY_CONTAINS_PREVIOUS,
    _IX_BODY_CONTAINED_BY_PREVIOUS,
    _IX_RANGE_CONTAINS_PREVIOUS,
    _IX_RANGE_CONTAINED_BY_PREVIOUS,
    _IX_BULL_BODY_COVERS_PREVIOUS_BEAR,
    _IX_BEAR_BODY_COVERS_PREVIOUS_BULL,
    _IX_HIGH_REJECTION_EVENT,
    _IX_LOW_REJECTION_EVENT,
    _IX_BODY_DIRECTION_DURATION,
    _IX_RANGE_RELATION_DURATION,
)


def _require_emitted_row_layout() -> None:
    """Prove the derived write offsets reproduce the declared name order.

    Listing the constants in emission order and demanding they equal
    ``range(width)`` proves a bijection onto the declared columns: no gap, no
    duplicate, no stale offset and no reordering.  A name that leaves
    :data:`CANDLE_PRIMITIVE_FEATURE_NAMES` has already raised ``ValueError`` at
    its ``.index`` lookup above; a name that joins it without a write site is
    caught here.
    """

    width = len(CANDLE_PRIMITIVE_FEATURE_NAMES)
    if _EMITTED_COLUMN_INDICES != tuple(range(width)):
        raise RuntimeError(
            "CANDLE_PRIMITIVE_ROW_LAYOUT_INVALID: write plan "
            f"{_EMITTED_COLUMN_INDICES} is not the declared column order"
        )
    if len(set(CANDLE_PRIMITIVE_FEATURE_NAMES)) != width:
        raise RuntimeError("CANDLE_PRIMITIVE_ROW_LAYOUT_INVALID: duplicate name")
    if not all(
        isinstance(name, str)
        and name.startswith(CANDLE_PRIMITIVE_FEATURE_PREFIX)
        for name in CANDLE_PRIMITIVE_FEATURE_NAMES
    ):
        raise RuntimeError("CANDLE_PRIMITIVE_ROW_LAYOUT_INVALID: name prefix")
    if CANDLE_PRIMITIVE_FEATURE_NAMES[
        : len(CANDLE_PRIMITIVE_WHOLE_BAR_FEATURE_NAMES)
    ] != CANDLE_PRIMITIVE_WHOLE_BAR_FEATURE_NAMES:
        raise RuntimeError(
            "CANDLE_PRIMITIVE_ROW_LAYOUT_INVALID: whole-bar block moved"
        )
    if CANDLE_PRIMITIVE_FEATURE_NAMES[
        len(CANDLE_PRIMITIVE_WHOLE_BAR_FEATURE_NAMES) :
    ] != CANDLE_PRIMITIVE_RELATIONAL_FEATURE_NAMES:
        raise RuntimeError(
            "CANDLE_PRIMITIVE_ROW_LAYOUT_INVALID: relational block moved"
        )


_require_emitted_row_layout()


def candle_primitive_contract_metadata() -> dict[str, object]:
    """Return the immutable identity embedded in local signal artifacts."""

    return {
        "owner": "gx1.features.entry_candle_primitives_v1",
        "feature_version": CANDLE_PRIMITIVE_FEATURE_VERSION,
        "feature_count": len(CANDLE_PRIMITIVE_FEATURE_NAMES),
        "ordered_feature_names_sha256": CANDLE_PRIMITIVE_FEATURE_NAMES_SHA256,
        "ordered_feature_names": list(CANDLE_PRIMITIVE_FEATURE_NAMES),
    }


@dataclass(frozen=True)
class CandlePrimitiveCarryState:
    """Exact continuation state for native-clock chunk evaluation."""

    feature_version: str = CANDLE_PRIMITIVE_FEATURE_VERSION
    previous_open: float | None = None
    previous_high: float | None = None
    previous_low: float | None = None
    previous_close: float | None = None
    previous_body_direction: int | None = None
    body_direction_duration_bars: int = 0
    previous_range_relation: int | None = None
    range_relation_duration_bars: int = 0
    rows_seen: int = 0
    last_timestamp_ns: int | None = None


def missing_candle_primitive_source_fields(columns: object) -> list[str]:
    available = set(columns)
    return [name for name in CANDLE_PRIMITIVE_SOURCE_FIELDS if name not in available]


def _time_index(frame: object) -> pd.DatetimeIndex:
    try:
        index = pd.DatetimeIndex(frame["time"])  # type: ignore[index]
    except Exception as exc:
        raise RuntimeError("CANDLE_PRIMITIVE_TIME_INVALID") from exc
    if index.empty or index.hasnans:
        raise RuntimeError("CANDLE_PRIMITIVE_TIME_INVALID")
    if index.tz is None or str(index.tz) != "UTC":
        raise RuntimeError("CANDLE_PRIMITIVE_TIME_NOT_UTC")
    if not index.is_monotonic_increasing or not index.is_unique:
        raise RuntimeError("CANDLE_PRIMITIVE_TIME_NOT_STRICT")
    return index


def _numeric_column(frame: object, name: str) -> np.ndarray:
    try:
        raw = frame[name]  # type: ignore[index]
    except Exception as exc:
        raise RuntimeError(f"CANDLE_PRIMITIVE_SOURCE_MISSING: {name}") from exc
    values = (
        raw.to_numpy(dtype=np.float64)
        if hasattr(raw, "to_numpy")
        else np.asarray(raw, dtype=np.float64)
    )
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise RuntimeError(f"CANDLE_PRIMITIVE_SOURCE_INVALID: {name}")
    return values


def compute_entry_candle_primitive_chunk(
    frame: object,
    *,
    carry: CandlePrimitiveCarryState | None = None,
) -> tuple[np.ndarray, list[str], CandlePrimitiveCarryState]:
    columns = getattr(frame, "columns", None)
    if columns is None:
        try:
            columns = frame.keys()  # type: ignore[union-attr]
        except Exception as exc:
            raise RuntimeError("CANDLE_PRIMITIVE_SOURCE_COLUMNS_INVALID") from exc
    missing = missing_candle_primitive_source_fields(columns)
    if missing:
        raise RuntimeError(f"CANDLE_PRIMITIVE_SOURCE_FIELDS_MISSING: {missing}")

    times = _time_index(frame)
    open_ = _numeric_column(frame, "open")
    high = _numeric_column(frame, "high")
    low = _numeric_column(frame, "low")
    close = _numeric_column(frame, "close")
    n_rows = len(close)
    if not (len(times) == len(open_) == len(high) == len(low) == n_rows):
        raise RuntimeError("CANDLE_PRIMITIVE_SOURCE_LENGTH_MISMATCH")
    if np.any(open_ <= 0.0) or np.any(high <= 0.0) or np.any(low <= 0.0) or np.any(close <= 0.0):
        raise RuntimeError("CANDLE_PRIMITIVE_SOURCE_PRICE_NONPOSITIVE")
    invalid_geometry = (
        (high < low)
        | (high < open_)
        | (high < close)
        | (low > open_)
        | (low > close)
    )
    if invalid_geometry.any():
        raise RuntimeError(
            "CANDLE_PRIMITIVE_SOURCE_GEOMETRY_INVALID: "
            f"row={int(np.flatnonzero(invalid_geometry)[0])}"
        )

    bar_range = high - low
    zero_range = bar_range == 0.0
    positive_range = ~zero_range
    body_signed = close - open_
    body = np.abs(body_signed)
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low

    body_signed_range = np.zeros(n_rows, dtype=np.float64)
    upper_wick_share = np.zeros(n_rows, dtype=np.float64)
    lower_wick_share = np.zeros(n_rows, dtype=np.float64)
    body_signed_range[positive_range] = (
        body_signed[positive_range] / bar_range[positive_range]
    )
    upper_wick_share[positive_range] = (
        upper_wick[positive_range] / bar_range[positive_range]
    )
    lower_wick_share[positive_range] = (
        lower_wick[positive_range] / bar_range[positive_range]
    )

    if carry is None:
        state = CandlePrimitiveCarryState()
    else:
        if (
            not isinstance(carry, CandlePrimitiveCarryState)
            or carry.feature_version != CANDLE_PRIMITIVE_FEATURE_VERSION
            or carry.rows_seen < 0
            or carry.body_direction_duration_bars < 0
            or carry.range_relation_duration_bars < 0
            or carry.previous_body_direction not in {None, -1, 0, 1}
            or carry.previous_range_relation not in {None, -1, 0, 1, 2}
            or any(
                value is not None and (not np.isfinite(value) or value <= 0.0)
                for value in (
                    carry.previous_open,
                    carry.previous_high,
                    carry.previous_low,
                    carry.previous_close,
                )
            )
        ):
            raise RuntimeError("CANDLE_PRIMITIVE_CARRY_INVALID")
        previous_values = (
            carry.previous_open,
            carry.previous_high,
            carry.previous_low,
            carry.previous_close,
        )
        if any(value is None for value in previous_values) != all(
            value is None for value in previous_values
        ):
            raise RuntimeError("CANDLE_PRIMITIVE_CARRY_INVALID")
        is_empty_state = carry.rows_seen == 0
        if is_empty_state != all(value is None for value in previous_values):
            raise RuntimeError("CANDLE_PRIMITIVE_CARRY_INVALID")
        if is_empty_state:
            if (
                carry.last_timestamp_ns is not None
                or carry.previous_body_direction is not None
                or carry.body_direction_duration_bars != 0
                or carry.previous_range_relation is not None
                or carry.range_relation_duration_bars != 0
            ):
                raise RuntimeError("CANDLE_PRIMITIVE_CARRY_INVALID")
        else:
            previous_open, previous_high, previous_low, previous_close = (
                float(value) for value in previous_values
            )
            if (
                carry.last_timestamp_ns is None
                or previous_high < previous_low
                or previous_high < max(previous_open, previous_close)
                or previous_low > min(previous_open, previous_close)
                or carry.previous_body_direction
                != int(np.sign(previous_close - previous_open))
                or carry.body_direction_duration_bars < 1
                or (carry.rows_seen == 1)
                != (carry.previous_range_relation is None)
                or (carry.rows_seen == 1)
                != (carry.range_relation_duration_bars == 0)
                or (
                    carry.rows_seen > 1
                    and carry.range_relation_duration_bars < 1
                )
            ):
                raise RuntimeError("CANDLE_PRIMITIVE_CARRY_INVALID")
        if (
            carry.last_timestamp_ns is not None
            and int(times[0].value) <= carry.last_timestamp_ns
        ):
            raise RuntimeError("CANDLE_PRIMITIVE_CARRY_TIME_OVERLAP")
        state = carry

    matrix = np.full(
        (n_rows, len(CANDLE_PRIMITIVE_FEATURE_NAMES)),
        np.nan,
        dtype=np.float64,
    )
    matrix[:, _IX_BODY_SIGNED_RANGE] = body_signed_range
    matrix[:, _IX_UPPER_WICK_SHARE] = upper_wick_share
    matrix[:, _IX_LOWER_WICK_SHARE] = lower_wick_share

    previous_open_value = state.previous_open
    previous_high_value = state.previous_high
    previous_low_value = state.previous_low
    previous_close_value = state.previous_close
    previous_body_direction = state.previous_body_direction
    body_direction_duration = int(state.body_direction_duration_bars)
    previous_range_relation = state.previous_range_relation
    range_relation_duration = int(state.range_relation_duration_bars)

    for row in range(n_rows):
        body_direction = int(np.sign(body_signed[row]))
        if previous_body_direction == body_direction:
            body_direction_duration += 1
        else:
            body_direction_duration = 1
        matrix[row, _IX_BODY_DIRECTION_DURATION] = float(body_direction_duration)

        if previous_close_value is not None:
            previous_range = float(previous_high_value - previous_low_value)
            previous_body = abs(previous_close_value - previous_open_value)
            price_scale = max(
                abs(open_[row]),
                abs(high[row]),
                abs(low[row]),
                abs(close[row]),
                abs(previous_open_value),
                abs(previous_high_value),
                abs(previous_low_value),
                abs(previous_close_value),
                1.0,
            )
            local_geometry_scale = max(
                float(bar_range[row]),
                previous_range,
                abs(open_[row] - previous_close_value),
                abs(close[row] - previous_close_value),
                abs(high[row] - previous_high_value),
                abs(low[row] - previous_low_value),
                abs(float(bar_range[row]) - previous_range),
                abs(float(body[row]) - previous_body),
                price_scale * np.finfo(np.float64).eps,
            )
            # Open position inside the previous bar's range.  Undefined 0/0
            # on a zero-range previous bar takes the storage zero, exactly as
            # the three whole-bar shares do, and those rows stay identifiable
            # from the previous row's three shares (v3 biconditional).
            matrix[row, _IX_OPEN_POSITION_PREVIOUS_RANGE] = (
                (open_[row] - previous_low_value) / previous_range
                if previous_range > 0.0
                else 0.0
            )
            matrix[row, _IX_CLOSE_CHANGE] = (
                close[row] - previous_close_value
            ) / local_geometry_scale
            matrix[row, _IX_HIGH_CHANGE] = (
                high[row] - previous_high_value
            ) / local_geometry_scale
            matrix[row, _IX_LOW_CHANGE] = (
                low[row] - previous_low_value
            ) / local_geometry_scale
            matrix[row, _IX_BODY_CHANGE] = (
                body[row] - previous_body
            ) / local_geometry_scale
            matrix[row, _IX_OPEN_ABOVE_PREVIOUS_HIGH] = (
                max(open_[row] - previous_high_value, 0.0) / local_geometry_scale
            )
            matrix[row, _IX_OPEN_BELOW_PREVIOUS_LOW] = (
                max(previous_low_value - open_[row], 0.0) / local_geometry_scale
            )

            body_low = min(open_[row], close[row])
            body_high = max(open_[row], close[row])
            previous_body_low = min(previous_open_value, previous_close_value)
            previous_body_high = max(previous_open_value, previous_close_value)
            body_overlap = max(
                min(body_high, previous_body_high)
                - max(body_low, previous_body_low),
                0.0,
            )
            matrix[row, _IX_BODY_OVERLAP_PREVIOUS] = (
                body_overlap / local_geometry_scale
            )
            body_equal = (
                body_low == previous_body_low
                and body_high == previous_body_high
            )
            body_contains = (
                body_low <= previous_body_low
                and body_high >= previous_body_high
                and not body_equal
            )
            body_contained = (
                body_low >= previous_body_low
                and body_high <= previous_body_high
                and not body_equal
            )
            matrix[row, _IX_BODY_CONTAINS_PREVIOUS] = float(body_contains)
            matrix[row, _IX_BODY_CONTAINED_BY_PREVIOUS] = float(body_contained)

            range_equal = (
                high[row] == previous_high_value
                and low[row] == previous_low_value
            )
            range_contains = (
                high[row] >= previous_high_value
                and low[row] <= previous_low_value
                and not range_equal
            )
            range_contained = (
                high[row] <= previous_high_value
                and low[row] >= previous_low_value
                and not range_equal
            )
            matrix[row, _IX_RANGE_CONTAINS_PREVIOUS] = float(range_contains)
            matrix[row, _IX_RANGE_CONTAINED_BY_PREVIOUS] = float(range_contained)

            previous_direction = int(previous_close_value > previous_open_value) - int(
                previous_close_value < previous_open_value
            )
            bull_cover = body_contains and body_direction == 1 and previous_direction == -1
            bear_cover = body_contains and body_direction == -1 and previous_direction == 1
            matrix[row, _IX_BULL_BODY_COVERS_PREVIOUS_BEAR] = float(bull_cover)
            matrix[row, _IX_BEAR_BODY_COVERS_PREVIOUS_BULL] = float(bear_cover)

            high_rejection = high[row] > previous_high_value and close[row] <= previous_high_value
            low_rejection = low[row] < previous_low_value and close[row] >= previous_low_value
            matrix[row, _IX_HIGH_REJECTION_EVENT] = float(high_rejection)
            matrix[row, _IX_LOW_REJECTION_EVENT] = float(low_rejection)

            range_relation = (
                2
                if range_equal
                else 1
                if range_contains
                else -1
                if range_contained
                else 0
            )
            if previous_range_relation == range_relation:
                range_relation_duration += 1
            else:
                range_relation_duration = 1
            matrix[row, _IX_RANGE_RELATION_DURATION] = float(
                range_relation_duration
            )
            previous_range_relation = range_relation

        previous_open_value = float(open_[row])
        previous_high_value = float(high[row])
        previous_low_value = float(low[row])
        previous_close_value = float(close[row])
        previous_body_direction = body_direction

    matrix = matrix.astype(np.float32, copy=False)
    if matrix.shape != (n_rows, len(CANDLE_PRIMITIVE_FEATURE_NAMES)):
        raise RuntimeError("CANDLE_PRIMITIVE_OUTPUT_SHAPE_INVALID")
    if np.isinf(matrix).any():
        raise RuntimeError("CANDLE_PRIMITIVE_OUTPUT_INFINITY")
    complete = np.isfinite(matrix).all(axis=1)
    expected_first_complete = 0 if state.previous_close is not None else 1
    if complete.any():
        first_complete = int(np.argmax(complete))
        if first_complete != expected_first_complete or not complete[first_complete:].all():
            raise RuntimeError("CANDLE_PRIMITIVE_OUTPUT_AVAILABILITY_INVALID")
    elif n_rows > expected_first_complete:
        raise RuntimeError("CANDLE_PRIMITIVE_OUTPUT_UNAVAILABLE")
    next_carry = CandlePrimitiveCarryState(
        previous_open=previous_open_value,
        previous_high=previous_high_value,
        previous_low=previous_low_value,
        previous_close=previous_close_value,
        previous_body_direction=previous_body_direction,
        body_direction_duration_bars=body_direction_duration,
        previous_range_relation=previous_range_relation,
        range_relation_duration_bars=range_relation_duration,
        rows_seen=state.rows_seen + n_rows,
        last_timestamp_ns=int(times[-1].value),
    )
    return matrix, list(CANDLE_PRIMITIVE_FEATURE_NAMES), next_carry


def build_entry_candle_primitive_layer(
    frame: object,
) -> tuple[np.ndarray, list[str]]:
    """Batch compatibility route through the exact chunk/carry owner."""

    matrix, names, _carry = compute_entry_candle_primitive_chunk(frame)
    return matrix, names


__all__ = [
    "CANDLE_PRIMITIVE_FEATURE_NAMES",
    "CANDLE_PRIMITIVE_FEATURE_NAMES_SHA256",
    "CANDLE_PRIMITIVE_FEATURE_PREFIX",
    "CANDLE_PRIMITIVE_FEATURE_VERSION",
    "CANDLE_PRIMITIVE_RELATIONAL_FEATURE_NAMES",
    "CANDLE_PRIMITIVE_SOURCE_FIELDS",
    "CANDLE_PRIMITIVE_WHOLE_BAR_FEATURE_NAMES",
    "CandlePrimitiveCarryState",
    "build_entry_candle_primitive_layer",
    "candle_primitive_contract_metadata",
    "compute_entry_candle_primitive_chunk",
    "missing_candle_primitive_source_fields",
]
