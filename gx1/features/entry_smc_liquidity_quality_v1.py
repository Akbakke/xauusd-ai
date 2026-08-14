"""Strict closed-bar SMC/liquidity quality features for model-native Entry.

FIDELITY NOTE (docs/INDICATOR_FIDELITY_AUDIT_20260813.md §1b, recorded here
2026-08-13 by V30 package 8B so the next session inherits the map instead of the
impression -- rule 25c).  These 24 fields are hand-authored SCORING RULEBOOKS
over APPROXIMATE ingredients, and a rulebook can never be more faithful than its
inputs.  Concretely, and proven from source:

* **The sweep has no reclaim.**  ``snap.smc_sweep_up/down`` marks a wick beyond a
  prior extreme.  Nothing here detects the *return* inside the range that makes a
  sweep a sweep; ``chart.foundation_sweep_low_reclaim_up_proxy`` is a continuous
  blend, hence the ``_proxy`` in its name.  Every ``reclaim_*`` and
  ``sweep_reclaim_*`` field is therefore a confidence over a sweep candidate, not
  a confirmed reclaim event.
* **The pool has no pool.**  ``liquidity_pool_proximity_*`` is a weighted blend
  of distances to swing extremes and per-TF highs/lows.  There is no object with
  an identity, a size, or a life-cycle -- nothing that can be "swept and gone".
* **The dealing range is not anchored.**  ``snap.smc_premium_discount`` splits a
  rolling window, not a BOS-anchored impulse leg, so "premium"/"discount" is
  relative to a window boundary that moves every bar.
* **The touch count was not a touch count.**  ``_count_proxy`` was the fraction
  of twelve DIFFERENT proximity rows (pivots, pivot points, per-TF highs/lows,
  swing, geometry) exceeding 0.55 on the SINGLE CURRENT bar -- a cross-sectional
  confluence share with no time axis at all -- and it was consumed under the
  names ``support_touch_count`` / ``resistance_touch_count``.  Twelve different
  things being close once is confluence; one thing being close twelve times is a
  touch count.  Repaired 2026-08-13: the genuine temporal counts
  ``level_{above,below}_touch_count`` are now consumed from the level registry
  (see ``_touch_count_strength``).

UPGRADE PATH (the named, bounded work that would raise the ceiling; none of it
is done here):

1. **A real reclaim event.**  Emit a discrete "swept the extreme at bar t-k and
   closed back inside the range by bar t" event with its own age, replacing the
   ``*_reclaim_*_proxy`` blends.  The swing/level machinery to detect it already
   exists in ``swing_structure_v1`` and ``level_registry_v1``.
2. **Pool identity from the level registry.**  ``level_registry_v1`` already
   maintains per-level identity, ``touch_count``, ``test_count``,
   ``bars_since_touch``, ``age_bars`` and reaction magnitudes, plus
   ``level_break_*`` and ``level_retest_{hold,fail}_signed``.  A liquidity pool
   should BE a registry level with an identity that can be broken, not a
   distance blend.  Step one of that migration is taken below for the touch
   counts.
3. **A BOS-anchored dealing range.**  Anchor premium/discount to the last
   ``foundation_bos_*`` impulse leg (low -> high of the leg that broke
   structure), not to a rolling window.

Evidence class: PROVEN FROM SOURCE (the structure of these expressions).  Whether
the approximations cost measurable edge is UNPROVEN -- no ablation exists.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


SMC_LIQUIDITY_QUALITY_FEATURE_VERSION = (
    "entry_smc_liquidity_quality_v4_20260813_registry_touch_counts_and_honest_confluence_name"
)
SMC_LIQUIDITY_QUALITY_FEATURE_PREFIX = "chart.smc_liquidity_"

SMC_LIQUIDITY_QUALITY_FEATURE_SUFFIXES = (
    "sweep_low_quality_long",
    "sweep_high_quality_short",
    "reclaim_confirmation_long",
    "reclaim_confirmation_short",
    "false_break_reversal_pressure_long",
    "false_break_reversal_pressure_short",
    "sweep_reclaim_strength_long",
    "sweep_reclaim_strength_short",
    "false_breakout_quality_long",
    "false_breakout_quality_short",
    "inducement_trap_risk_long",
    "inducement_trap_risk_short",
    "liquidity_pool_proximity_low",
    "liquidity_pool_proximity_high",
    "liquidity_pool_proximity_balance_low_minus_high",
    "two_sided_liquidity_pool_pressure",
    "wick_rejection_strength_long",
    "wick_rejection_strength_short",
    "premium_discount_trend_alignment_long",
    "premium_discount_trend_alignment_short",
    "premium_discount_reclaim_confluence_long",
    "premium_discount_reclaim_confluence_short",
    "continuation_pressure_long",
    "continuation_pressure_short",
)
SMC_LIQUIDITY_QUALITY_FEATURE_NAMES = tuple(
    f"{SMC_LIQUIDITY_QUALITY_FEATURE_PREFIX}{name}" for name in SMC_LIQUIDITY_QUALITY_FEATURE_SUFFIXES
)

SMC_LIQUIDITY_QUALITY_SOURCE_FIELDS = (
    "snap.smc_sweep_up",
    "snap.smc_sweep_down",
    "snap.smc_sweep_size_atr",
    "snap.smc_bars_since_sweep",
    "snap.smc_premium_discount",
    "snap.smc_choch",
    "ctx_cont.smc_sweep_bull_pressure_last12",
    "ctx_cont.smc_sweep_bull_pressure_last48",
    "ctx_cont.smc_sweep_size_recent_tau12",
    "ctx_cont.smc_sweep_recency_tau24",
    "ctx_cont.smc_choch_recent_tau12",
    "ctx_cont.smc_choch_recent_tau24",
    "ctx_cont.sr_nearest_pivot_abs_atr",
    "ctx_cont.sr_support_proximity_exp",
    "ctx_cont.sr_resistance_proximity_exp",
    "ctx_cont.liquidity_hi_nearest_abs_atr",
    "ctx_cont.liquidity_lo_nearest_abs_atr",
    "ctx_cont.dist_to_R1_atr",
    "ctx_cont.dist_to_R2_atr",
    "ctx_cont.dist_to_S1_atr",
    "ctx_cont.dist_to_S2_atr",
    "ctx_cont.dist_to_m5_hi_atr",
    "ctx_cont.dist_to_m5_lo_atr",
    "ctx_cont.dist_to_m15_hi_atr",
    "ctx_cont.dist_to_m15_lo_atr",
    "ctx_cont.dist_to_h1_hi_atr",
    "ctx_cont.dist_to_h1_lo_atr",
    "ctx_cont.dist_to_h4_hi_atr",
    "ctx_cont.dist_to_h4_lo_atr",
    "ctx_cont.dist_to_d1_hi_atr",
    "ctx_cont.dist_to_d1_lo_atr",
    "ctx_cont.dist_last_swing_high_atr",
    "ctx_cont.dist_last_swing_low_atr",
    "ctx_cont.bars_since_swing_high",
    "ctx_cont.bars_since_swing_low",
    "ctx_cont._v1h1_ema_diff",
    "ctx_cont._v1h4_ema_diff",
    "ctx_cont.d1_ema_slope_20_canon_v2",
    "ctx_cont.m15_trend_sign_canon_v2",
    "ctx_cont.regime_stack_sum_v3",
    "chart.foundation_sweep_low_reclaim_up_proxy",
    "chart.foundation_sweep_high_reclaim_down_proxy",
    "chart.foundation_false_breakout_high_followthrough_down_proxy",
    "chart.foundation_false_breakout_low_followthrough_up_proxy",
    "chart.geometry_support_line_proximity_stack",
    "chart.geometry_resistance_line_proximity_stack",
    "chart.geometry_failed_breakout_high_reversal_pressure",
    "chart.geometry_failed_breakout_low_reversal_pressure",
    "candle.pattern_upper_wick_share",
    "candle.pattern_lower_wick_share",
    "candle.pattern_close_location",
    "candle.pattern_body_direction",
    "candle.pattern_body_share",
    # V30 package 8B (2026-08-13): the level registry's GENUINE temporal touch
    # counts, adopted in place of the cross-sectional confluence share that used
    # to be consumed under the name ``*_touch_count``.  These are emitted
    # unprefixed on the entry-M5 surface by ``level_registry_m5_layer``, which
    # the inline extension materializes BEFORE this layer runs.
    "level_above_touch_count",
    "level_below_touch_count",
)


# --- Shared hand-authored constants (single owners, rule 19) -----------------
# Trend composite weights over (h1_trend, h4_trend, d1_slope, m15_trend,
# regime_stack).  Origin: the chart-layer trend_proxy weighting in
# gx1/features/entry_model_native_feature_layers_v1.py:557-563, adopted
# 2026-08-09 as the single owner of this five-input fusion.  The per-file
# variants (0.32/0.30/0.20/0.12/0.06 here, 0.30/0.30/0.20/0.12/0.08 in the
# support/resistance memory layer) were hand copies of the same composite with
# no separately stated origin and are retired in favour of this one weighting.
TREND_COMPOSITE_WEIGHTS = (0.35, 0.30, 0.20, 0.10, 0.05)

# Wick-rejection close-location base term.  Defined once here as the single
# owner 2026-08-09: this file's 0.50 variant is the earlier owner; the
# support/resistance memory layer carried a 0.55 copy of the otherwise
# identical expression with no separate origin and now imports this constant.
WICK_REJECTION_CLOSE_LOCATION_BASE = 0.50

# Algebraic maximum of the wick-rejection product
#   wick_share * (BASE + close_location + 0.25 * body_side) * (0.75 + 0.25 * body_share)
# with wick_share, close_location, body_side, body_share all in [0, 1]:
#   1.0 * (0.50 + 1.0 + 0.25) * (0.75 + 0.25 * 1.0) = 1.75.
# Dividing the product by this bound before the [0, 1] clip maps the algebraic
# range onto the declared domain instead of clipping away its top 43%.
WICK_REJECTION_STRENGTH_ALGEBRAIC_MAX = WICK_REJECTION_CLOSE_LOCATION_BASE + 1.0 + 0.25

# Algebraic maximum of the premium/discount trend-alignment product
#   side * (0.50 + trend_side) * (0.50 + stack + pool):
#   side <= 1 (_clip01); trend_side < 1 because TREND_COMPOSITE_WEIGHTS sums to
#   1.0 over tanh-bounded inputs, so (0.50 + trend_side) <= 1.5; stack <= 1.5
#   (its unclipped swing row is prox <= 1 times (0.50 + recency <= 1.0)) and
#   pool <= 1, so (0.50 + stack + pool) <= 3.0.  Bound: 1.0 * 1.5 * 3.0 = 4.5.
PREMIUM_DISCOUNT_TREND_ALIGNMENT_ALGEBRAIC_MAX = 4.5


def _name_index(names: Iterable[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(names)}


def missing_smc_liquidity_quality_source_fields(feature_names: Iterable[str]) -> list[str]:
    available = set(feature_names)
    return [name for name in SMC_LIQUIDITY_QUALITY_SOURCE_FIELDS if name not in available]


def _col(x: np.ndarray, index: dict[str, int], name: str) -> np.ndarray:
    if name not in index:
        raise RuntimeError(f"SMC_LIQUIDITY_QUALITY_SOURCE_FIELD_MISSING: {name}")
    arr = np.asarray(x[:, index[name]], dtype=np.float32)
    if not np.isfinite(arr).all():
        raise RuntimeError(f"SMC_LIQUIDITY_QUALITY_SOURCE_NONFINITE: {name}")
    return arr


def _clip(arr: np.ndarray, lo: float = -25.0, hi: float = 25.0) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError("SMC_LIQUIDITY_QUALITY_DERIVATION_NONFINITE")
    return np.clip(values, lo, hi).astype(np.float32, copy=False)


def _clip01(arr: np.ndarray) -> np.ndarray:
    return _clip(arr, 0.0, 1.0)


def _pos(arr: np.ndarray) -> np.ndarray:
    return np.maximum(arr, 0.0).astype(np.float32, copy=False)


def _neg(arr: np.ndarray) -> np.ndarray:
    return np.maximum(-arr, 0.0).astype(np.float32, copy=False)


def _prox_abs(arr: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.abs(arr))).astype(np.float32, copy=False)


def _recency(arr: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.maximum(arr, 0.0))).astype(np.float32, copy=False)


def _tanh(arr: np.ndarray, scale: float = 1.0) -> np.ndarray:
    return np.tanh(arr / max(float(scale), 1e-6)).astype(np.float32, copy=False)


def _touch_count_strength(counts: np.ndarray) -> np.ndarray:
    """Map a non-negative touch count onto [0, 1): ``n / (1 + n)``.

    This is the exact algebraic complement of this file's own ``_recency``
    owner (``1 / (1 + max(x, 0))``), reused rather than re-parameterized so no
    magnitude is introduced (rule 2b): 0 touches -> 0.0, 1 -> 0.5, 2 -> 0.667.
    The level registry encodes an absent side as ``touch_count = 0``, which maps
    to 0.0 -- "no evidence", not a fabricated middle value.
    """

    return (1.0 - _recency(counts)).astype(np.float32, copy=False)


def _add(arrays: list[np.ndarray], names: list[str], name: str, arr: np.ndarray, *, lo: float = -25.0, hi: float = 25.0) -> None:
    clean = _clip(np.asarray(arr, dtype=np.float32), lo, hi)
    if clean.ndim != 1:
        raise RuntimeError(f"SMC/liquidity quality feature {name} is not 1D: {clean.shape}")
    if not np.isfinite(clean).all():
        raise RuntimeError(f"SMC/liquidity quality feature {name} contains non-finite values")
    arrays.append(clean)
    names.append(f"{SMC_LIQUIDITY_QUALITY_FEATURE_PREFIX}{name}")


def build_entry_smc_liquidity_quality_layer(
    x: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Build deterministic SMC/liquidity quality candidates from closed-bar inputs."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise RuntimeError(f"SMC_LIQUIDITY_QUALITY_INPUT_NOT_2D: shape={x.shape}")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise RuntimeError(f"SMC_LIQUIDITY_QUALITY_INPUT_EMPTY: shape={x.shape}")
    if x.shape[1] != len(feature_names):
        raise RuntimeError(
            "SMC_LIQUIDITY_QUALITY_FEATURE_NAME_DIM_MISMATCH: "
            f"cols={x.shape[1]} names={len(feature_names)}"
        )
    if any(not isinstance(name, str) or not name for name in feature_names):
        raise RuntimeError("SMC_LIQUIDITY_QUALITY_FEATURE_NAME_INVALID")
    if len(feature_names) != len(set(feature_names)):
        seen: set[str] = set()
        duplicates = sorted(
            {name for name in feature_names if name in seen or seen.add(name)}
        )
        raise RuntimeError(
            f"SMC_LIQUIDITY_QUALITY_DUPLICATE_FEATURE_NAMES: {duplicates[:20]}"
        )
    idx = _name_index(feature_names)
    missing = missing_smc_liquidity_quality_source_fields(feature_names)
    if missing:
        raise RuntimeError(f"SMC_LIQUIDITY_QUALITY_SOURCE_FIELDS_MISSING: {missing[:20]} total={len(missing)}")
    if not np.isfinite(x).all():
        bad_rows, bad_cols = np.where(~np.isfinite(x))
        examples = [
            {"row": int(row), "feature": feature_names[int(col)]}
            for row, col in zip(bad_rows[:10], bad_cols[:10])
        ]
        raise RuntimeError(f"SMC_LIQUIDITY_QUALITY_SOURCE_NONFINITE: {examples}")
    arrays: list[np.ndarray] = []
    names: list[str] = []

    def c(name: str) -> np.ndarray:
        return _col(x, idx, name)

    h1_trend = _tanh(c("ctx_cont._v1h1_ema_diff"))
    h4_trend = _tanh(c("ctx_cont._v1h4_ema_diff"))
    d1_slope = _tanh(c("ctx_cont.d1_ema_slope_20_canon_v2"))
    m15_trend = _tanh(c("ctx_cont.m15_trend_sign_canon_v2"))
    regime_stack = _tanh(c("ctx_cont.regime_stack_sum_v3"), scale=3.0)
    w_h1, w_h4, w_d1, w_m15, w_regime = TREND_COMPOSITE_WEIGHTS
    trend = _clip(
        w_h1 * h1_trend + w_h4 * h4_trend + w_d1 * d1_slope + w_m15 * m15_trend + w_regime * regime_stack
    )
    trend_up = _pos(trend)
    trend_down = _neg(trend)

    sweep_up = _clip01(
        c("snap.smc_sweep_up")
        + 0.50 * _neg(c("ctx_cont.smc_sweep_bull_pressure_last12"))
        + 0.25 * _neg(c("ctx_cont.smc_sweep_bull_pressure_last48"))
    )
    sweep_down = _clip01(
        c("snap.smc_sweep_down")
        + 0.50 * _pos(c("ctx_cont.smc_sweep_bull_pressure_last12"))
        + 0.25 * _pos(c("ctx_cont.smc_sweep_bull_pressure_last48"))
    )
    sweep_recent = _clip01(_recency(c("snap.smc_bars_since_sweep")) + c("ctx_cont.smc_sweep_recency_tau24"))
    sweep_size = _clip01(c("snap.smc_sweep_size_atr") + c("ctx_cont.smc_sweep_size_recent_tau12"))
    choch_recent = _clip01(c("snap.smc_choch") + c("ctx_cont.smc_choch_recent_tau12") + 0.5 * c("ctx_cont.smc_choch_recent_tau24"))

    premium = _clip01(c("snap.smc_premium_discount"))
    discount = _clip01(1.0 - premium)
    body_direction = _clip(c("candle.pattern_body_direction"), -1.0, 1.0)
    body_share = _clip01(c("candle.pattern_body_share"))
    body_bull = _pos(body_direction)
    body_bear = _neg(body_direction)

    upper_wick = _clip01(c("candle.pattern_upper_wick_share"))
    lower_wick = _clip01(c("candle.pattern_lower_wick_share"))
    close_near_high = _clip01(c("candle.pattern_close_location"))
    close_near_low = _clip01(1.0 - close_near_high)
    # Normalized by the derived algebraic maximum before the [0, 1] clip so the
    # full algebraic range maps onto the declared domain (see the constant's
    # derivation comment at module level).
    wick_reject_long = _clip01(
        lower_wick
        * (WICK_REJECTION_CLOSE_LOCATION_BASE + close_near_high + 0.25 * body_bull)
        * (0.75 + 0.25 * body_share)
        / WICK_REJECTION_STRENGTH_ALGEBRAIC_MAX
    )
    wick_reject_short = _clip01(
        upper_wick
        * (WICK_REJECTION_CLOSE_LOCATION_BASE + close_near_low + 0.25 * body_bear)
        * (0.75 + 0.25 * body_share)
        / WICK_REJECTION_STRENGTH_ALGEBRAIC_MAX
    )

    recent_swing_low = _recency(c("ctx_cont.bars_since_swing_low"))
    recent_swing_high = _recency(c("ctx_cont.bars_since_swing_high"))
    near_swing_low = _prox_abs(c("ctx_cont.dist_last_swing_low_atr"))
    near_swing_high = _prox_abs(c("ctx_cont.dist_last_swing_high_atr"))
    swing_low_row = near_swing_low * (0.50 + recent_swing_low)
    swing_high_row = near_swing_high * (0.50 + recent_swing_high)
    low_liquidity_proximity = _prox_abs(c("ctx_cont.liquidity_lo_nearest_abs_atr"))
    high_liquidity_proximity = _prox_abs(c("ctx_cont.liquidity_hi_nearest_abs_atr"))
    support_geometry_row = _clip01(c("chart.geometry_support_line_proximity_stack"))
    resistance_geometry_row = _clip01(c("chart.geometry_resistance_line_proximity_stack"))

    # Support/resistance proximity stacks (2026-08-09 repair).
    #
    # The max() previously ran over 12 rows.  Producer contract
    # (gx1/features/entry_smart_context.py:140-184):
    #   s_abs = min(|dist_to_S1_atr|, |dist_to_S2_atr|)
    #   r_abs = min(|dist_to_R1_atr|, |dist_to_R2_atr|)
    #   sr_nearest_pivot_abs_atr    = min(r_abs, s_abs)
    #   sr_support_proximity_exp    = exp(-min(s_abs, 20))
    #   sr_resistance_proximity_exp = exp(-min(r_abs, 20))
    #   liquidity_lo_nearest_abs_atr = min over |dist_to_{m5,m15,h1,h4,d1}_lo_atr|
    #   liquidity_hi_nearest_abs_atr = min over |dist_to_{m5,m15,h1,h4,d1}_hi_atr|
    # Because 1/(1+|x|) is monotone decreasing and round-to-nearest is monotone,
    # _prox_abs of a min equals the componentwise max of _prox_abs bit-exactly,
    # so under that contract these max inputs were algebraically dead (each one
    # <= a row that remains) and are removed from the max:
    #   removed (dominated) row               dominator kept in the max
    #   _prox_abs(dist_to_m5_lo_atr)          _prox_abs(liquidity_lo_nearest_abs_atr)
    #   _prox_abs(dist_to_m15_lo_atr)         _prox_abs(liquidity_lo_nearest_abs_atr)
    #   _prox_abs(dist_to_h1_lo_atr)          _prox_abs(liquidity_lo_nearest_abs_atr)
    #   _prox_abs(dist_to_h4_lo_atr)          _prox_abs(liquidity_lo_nearest_abs_atr)
    #   _prox_abs(dist_to_d1_lo_atr)          _prox_abs(liquidity_lo_nearest_abs_atr)
    #   _clip01(sr_support_proximity_exp)     max(_prox_abs(dist_to_S1_atr), _prox_abs(dist_to_S2_atr))
    #                                         = 1/(1+s_abs) >= exp(-min(s_abs, 20))
    #                                         since exp(-x) <= 1/(1+x) for x >= 0
    #                                         (valid while s_abs <= e^20 - 1
    #                                         ~= 4.9e8 ATR, i.e. every real tape;
    #                                         exact up to <=1-ulp faithful
    #                                         rounding of np.exp as s_abs -> 0)
    # and the mirrored hi/R rows for the resistance stack.
    #
    # Separately, the shared row _prox_abs(sr_nearest_pivot_abs_atr) — prox of
    # min(r_abs, s_abs) — appeared in BOTH stacks, flooring support proximity by
    # resistance distance (and vice versa) and making the two stacks bit-equal
    # whenever it won.  It is removed from both maxes (VALUE CHANGE): the
    # side-specific rows dist_to_S1/S2 (support) and dist_to_R1/R2 (resistance)
    # remain and now carry the side information.
    support_stack = np.maximum.reduce(
        [
            _prox_abs(c("ctx_cont.dist_to_S1_atr")),
            _prox_abs(c("ctx_cont.dist_to_S2_atr")),
            low_liquidity_proximity,
            swing_low_row,
            support_geometry_row,
        ]
    ).astype(np.float32)
    resistance_stack = np.maximum.reduce(
        [
            _prox_abs(c("ctx_cont.dist_to_R1_atr")),
            _prox_abs(c("ctx_cont.dist_to_R2_atr")),
            high_liquidity_proximity,
            swing_high_row,
            resistance_geometry_row,
        ]
    ).astype(np.float32)

    # V30 package 8B (2026-08-13).  The two variables consumed below used to be
    # ``_count_proxy`` over a 12-row proximity stack — a CROSS-SECTIONAL
    # confluence share at the current bar, wearing a temporal "touch count"
    # name.  They now carry the GENUINE temporal counts from the level
    # registry's per-level crossing engine (level_registry_v1): how many times
    # price has come back to THIS identified level.  ``level_below_*`` is the
    # nearest level below price (support), ``level_above_*`` the nearest above
    # (resistance); an absent side is encoded as touch_count = 0 by the
    # registry, which maps to 0.0 here.
    #
    # Rule 4: the 12-row confluence stack is not evidence that was removed —
    # every one of its rows is an independent model input in its own right
    # (``ctx_cont.sr_{support,resistance}_proximity_exp``,
    # ``sr_nearest_pivot_abs_atr``, ``dist_to_{S1,S2,R1,R2}_atr``, the ten
    # per-TF ``dist_to_*_{hi,lo}_atr`` fields, ``liquidity_{hi,lo}_nearest_abs_atr``,
    # the swing-distance/age fields and
    # ``chart.geometry_{support,resistance}_line_proximity_stack``), and the
    # side-specific maxima of the same rows are still computed above as
    # ``support_stack`` / ``resistance_stack``.  What is gone is only a
    # hand-authored fraction-over-a-threshold that no longer had a consumer once
    # the real count arrived; the model can form it from the inputs if it helps.
    support_touch_count = _touch_count_strength(c("level_below_touch_count"))
    resistance_touch_count = _touch_count_strength(c("level_above_touch_count"))

    # A liquidity pool is not the same object as generic support/resistance.
    # The previous max(support_stack, liquidity_proximity) expression was
    # algebraically identical to support_stack because liquidity_proximity was
    # already one of the max inputs above.  That silently spent two model
    # slots on exact S/R duplicates.  Keep the S/R stack for confluence, while
    # the pool surface now measures a causal blend of the dedicated liquidity
    # locator, the most recent swing, and clustered lower/higher TF extremes.
    # This preserves the intended cross-family cooperation without allowing
    # either family to masquerade as the other.
    low_swing_proximity = _clip01(swing_low_row)
    high_swing_proximity = _clip01(swing_high_row)
    low_tf_cluster = _clip01(
        0.30 * _prox_abs(c("ctx_cont.dist_to_m5_lo_atr"))
        + 0.25 * _prox_abs(c("ctx_cont.dist_to_m15_lo_atr"))
        + 0.20 * _prox_abs(c("ctx_cont.dist_to_h1_lo_atr"))
        + 0.15 * _prox_abs(c("ctx_cont.dist_to_h4_lo_atr"))
        + 0.10 * _prox_abs(c("ctx_cont.dist_to_d1_lo_atr"))
    )
    high_tf_cluster = _clip01(
        0.30 * _prox_abs(c("ctx_cont.dist_to_m5_hi_atr"))
        + 0.25 * _prox_abs(c("ctx_cont.dist_to_m15_hi_atr"))
        + 0.20 * _prox_abs(c("ctx_cont.dist_to_h1_hi_atr"))
        + 0.15 * _prox_abs(c("ctx_cont.dist_to_h4_hi_atr"))
        + 0.10 * _prox_abs(c("ctx_cont.dist_to_d1_hi_atr"))
    )
    low_pool = _clip01(
        0.55 * low_liquidity_proximity
        + 0.25 * low_swing_proximity
        + 0.20 * low_tf_cluster
    )
    high_pool = _clip01(
        0.55 * high_liquidity_proximity
        + 0.25 * high_swing_proximity
        + 0.20 * high_tf_cluster
    )

    foundation_reclaim_long = _clip01(c("chart.foundation_sweep_low_reclaim_up_proxy") / 5.0)
    foundation_reclaim_short = _clip01(c("chart.foundation_sweep_high_reclaim_down_proxy") / 5.0)
    foundation_false_low = _clip01(c("chart.foundation_false_breakout_low_followthrough_up_proxy") / 5.0)
    foundation_false_high = _clip01(c("chart.foundation_false_breakout_high_followthrough_down_proxy") / 5.0)
    geometry_false_low = _clip01(c("chart.geometry_failed_breakout_low_reversal_pressure") / 5.0)
    geometry_false_high = _clip01(c("chart.geometry_failed_breakout_high_reversal_pressure") / 5.0)

    sweep_low_quality = _clip(
        sweep_down
        * sweep_recent
        * (0.50 + sweep_size)
        * (0.50 + low_pool + support_stack)
        * (0.50 + wick_reject_long)
        * (0.50 + discount + trend_up),
        0.0,
        5.0,
    )
    sweep_high_quality = _clip(
        sweep_up
        * sweep_recent
        * (0.50 + sweep_size)
        * (0.50 + high_pool + resistance_stack)
        * (0.50 + wick_reject_short)
        * (0.50 + premium + trend_down),
        0.0,
        5.0,
    )
    reclaim_long = _clip01(0.45 * foundation_reclaim_long + 0.35 * (sweep_low_quality / 5.0) + 0.20 * close_near_high)
    reclaim_short = _clip01(0.45 * foundation_reclaim_short + 0.35 * (sweep_high_quality / 5.0) + 0.20 * close_near_low)
    false_low_context = _clip01(0.40 * foundation_false_low + 0.20 * geometry_false_low + 0.40 * (sweep_low_quality / 5.0 + choch_recent * trend_up))
    false_high_context = _clip01(0.40 * foundation_false_high + 0.20 * geometry_false_high + 0.40 * (sweep_high_quality / 5.0 + choch_recent * trend_down))
    # Normalized by the derived algebraic maximum before the [0, 1] clip (see
    # the constant's derivation comment at module level).
    long_pd_alignment = _clip01(
        discount
        * (0.50 + trend_up)
        * (0.50 + support_stack + low_pool)
        / PREMIUM_DISCOUNT_TREND_ALIGNMENT_ALGEBRAIC_MAX
    )
    short_pd_alignment = _clip01(
        premium
        * (0.50 + trend_down)
        * (0.50 + resistance_stack + high_pool)
        / PREMIUM_DISCOUNT_TREND_ALIGNMENT_ALGEBRAIC_MAX
    )
    inducement_risk_long = _clip(
        high_pool
        * resistance_stack
        * premium
        * (0.50 + sweep_up + wick_reject_short + false_high_context)
        * (1.0 - reclaim_long),
        0.0,
        5.0,
    )
    inducement_risk_short = _clip(
        low_pool
        * support_stack
        * discount
        * (0.50 + sweep_down + wick_reject_long + false_low_context)
        * (1.0 - reclaim_short),
        0.0,
        5.0,
    )
    sweep_reclaim_strength_long = _clip01(
        0.30 * (sweep_low_quality / 5.0)
        + 0.25 * reclaim_long
        + 0.20 * wick_reject_long
        + 0.15 * close_near_high
        + 0.10 * foundation_reclaim_long
    )
    sweep_reclaim_strength_short = _clip01(
        0.30 * (sweep_high_quality / 5.0)
        + 0.25 * reclaim_short
        + 0.20 * wick_reject_short
        + 0.15 * close_near_low
        + 0.10 * foundation_reclaim_short
    )
    false_break_reversal_pressure_long = _clip01(
        (0.30 * false_low_context + 0.25 * wick_reject_long + 0.20 * reclaim_long + 0.15 * discount + 0.10 * low_pool)
        * (0.75 + 0.25 * choch_recent)
    )
    false_break_reversal_pressure_short = _clip01(
        (0.30 * false_high_context + 0.25 * wick_reject_short + 0.20 * reclaim_short + 0.15 * premium + 0.10 * high_pool)
        * (0.75 + 0.25 * choch_recent)
    )
    false_breakout_quality_long = _clip01(
        false_low_context
        * (0.45 + 0.35 * reclaim_long + 0.20 * wick_reject_long)
        * (0.70 + 0.30 * low_pool)
    )
    false_breakout_quality_short = _clip01(
        false_high_context
        * (0.45 + 0.35 * reclaim_short + 0.20 * wick_reject_short)
        * (0.70 + 0.30 * high_pool)
    )
    liquidity_pool_balance = _clip(low_pool - high_pool, -1.0, 1.0)
    two_sided_liquidity_pressure = _clip01(
        np.minimum(low_pool, high_pool)
        # Both sides tested repeatedly = a real two-sided pool.  The weights are
        # unchanged; only the quantity they weigh became honest (a registry
        # touch count instead of a same-bar confluence share).
        * (0.50 + 0.25 * support_touch_count + 0.25 * resistance_touch_count)
        * (0.75 + 0.25 * sweep_recent)
    )
    premium_discount_reclaim_confluence_long = _clip01(
        discount
        * reclaim_long
        * (0.45 + 0.25 * support_stack + 0.20 * low_pool + 0.10 * wick_reject_long)
    )
    premium_discount_reclaim_confluence_short = _clip01(
        premium
        * reclaim_short
        * (0.45 + 0.25 * resistance_stack + 0.20 * high_pool + 0.10 * wick_reject_short)
    )
    continuation_pressure_long = _clip01(
        (
            0.35 * trend_up
            + 0.25 * long_pd_alignment
            + 0.20 * sweep_reclaim_strength_long
            + 0.20 * reclaim_long
        )
        * (1.0 - 0.35 * (inducement_risk_long / 5.0))
    )
    continuation_pressure_short = _clip01(
        (
            0.35 * trend_down
            + 0.25 * short_pd_alignment
            + 0.20 * sweep_reclaim_strength_short
            + 0.20 * reclaim_short
        )
        * (1.0 - 0.35 * (inducement_risk_short / 5.0))
    )

    _add(arrays, names, "sweep_low_quality_long", sweep_low_quality, lo=0.0, hi=5.0)
    _add(arrays, names, "sweep_high_quality_short", sweep_high_quality, lo=0.0, hi=5.0)
    _add(arrays, names, "reclaim_confirmation_long", reclaim_long, lo=0.0, hi=1.0)
    _add(arrays, names, "reclaim_confirmation_short", reclaim_short, lo=0.0, hi=1.0)
    _add(arrays, names, "false_break_reversal_pressure_long", false_break_reversal_pressure_long, lo=0.0, hi=1.0)
    _add(arrays, names, "false_break_reversal_pressure_short", false_break_reversal_pressure_short, lo=0.0, hi=1.0)
    _add(arrays, names, "sweep_reclaim_strength_long", sweep_reclaim_strength_long, lo=0.0, hi=1.0)
    _add(arrays, names, "sweep_reclaim_strength_short", sweep_reclaim_strength_short, lo=0.0, hi=1.0)
    _add(arrays, names, "false_breakout_quality_long", false_breakout_quality_long, lo=0.0, hi=1.0)
    _add(arrays, names, "false_breakout_quality_short", false_breakout_quality_short, lo=0.0, hi=1.0)
    _add(arrays, names, "inducement_trap_risk_long", inducement_risk_long, lo=0.0, hi=5.0)
    _add(arrays, names, "inducement_trap_risk_short", inducement_risk_short, lo=0.0, hi=5.0)
    _add(arrays, names, "liquidity_pool_proximity_low", low_pool, lo=0.0, hi=1.0)
    _add(arrays, names, "liquidity_pool_proximity_high", high_pool, lo=0.0, hi=1.0)
    _add(arrays, names, "liquidity_pool_proximity_balance_low_minus_high", liquidity_pool_balance, lo=-1.0, hi=1.0)
    _add(arrays, names, "two_sided_liquidity_pool_pressure", two_sided_liquidity_pressure, lo=0.0, hi=1.0)
    _add(arrays, names, "wick_rejection_strength_long", wick_reject_long, lo=0.0, hi=1.0)
    _add(arrays, names, "wick_rejection_strength_short", wick_reject_short, lo=0.0, hi=1.0)
    _add(arrays, names, "premium_discount_trend_alignment_long", long_pd_alignment, lo=0.0, hi=1.0)
    _add(arrays, names, "premium_discount_trend_alignment_short", short_pd_alignment, lo=0.0, hi=1.0)
    _add(
        arrays,
        names,
        "premium_discount_reclaim_confluence_long",
        premium_discount_reclaim_confluence_long,
        lo=0.0,
        hi=1.0,
    )
    _add(
        arrays,
        names,
        "premium_discount_reclaim_confluence_short",
        premium_discount_reclaim_confluence_short,
        lo=0.0,
        hi=1.0,
    )
    _add(arrays, names, "continuation_pressure_long", continuation_pressure_long, lo=0.0, hi=1.0)
    _add(arrays, names, "continuation_pressure_short", continuation_pressure_short, lo=0.0, hi=1.0)

    out = np.column_stack(arrays).astype(np.float32, copy=False) if arrays else np.empty((x.shape[0], 0), dtype=np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("SMC/liquidity quality layer contains non-finite values")
    if len(set(names)) != len(names):
        dupes = sorted({name for name in names if names.count(name) > 1})
        raise RuntimeError(f"SMC/liquidity quality layer has duplicate names: {dupes[:10]}")
    if tuple(names) != SMC_LIQUIDITY_QUALITY_FEATURE_NAMES:
        raise RuntimeError("SMC_LIQUIDITY_QUALITY_FEATURE_NAME_DRIFT")
    return out, names
