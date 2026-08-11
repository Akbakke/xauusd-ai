"""V29 level registry V1 — Phase A: pivot-clustered horizontal levels + round grid.

New bounded authority (rule 21): persistent horizontal price-level identity
with touch/reaction memory, break/retest events, and the stateless
round-number grid.  Design owner: ``docs/V29_EVENT_SURFACE_DESIGN_20260811.md``
§1; exact field/lifecycle spec: the smc_liquidity event-gap report of
2026-08-11 (``/home/andre2/GX1_DATA/logs/event_gap_review_20260811/
smc_liquidity.md``, CENTERPIECE §§0-5).  Phase A implements the
``pivot_cluster`` and ``round_number`` kinds; ``session_anchored`` is declared
in the kind enum below and is produced in Phase B.

Pivot truth (rule 13): swing pivots are detected by *executing*
``smc_v1._detect_swing_pivots`` (owner code, not a copy) on the causal
``2*SWING_LOOKBACK+1`` bar window, and a pivot at bar ``j`` is confirmed at bar
``j + SWING_LOOKBACK`` — the published confirmation rule of
``smc_v1._track_recent_swings``.  No second pivot detector exists.

Constant origins (rule 2a — one sentence each, at the constant definitions
below): pivot lookback ``SWING_LOOKBACK`` imported from ``smc_v1``; the
cluster tolerance ``TOL_LEVEL_ATR`` is a TRAIN-fitted statistic produced by
:func:`fit_level_registry_tolerance` and is an *explicit input* of every
compute function here (never a module default — no legitimate named-constant
origin exists for it, per the smc report §2); reaction window = the tau-12
family convention; retest window = the tau-24 family convention; expiry caps =
the ``augment_forward_outcome_v2._liquidity_zones`` per-TF lookbacks; distance
saturation 20 = the ``exp(-min(x, 20))`` cap convention; count/age cap 999 =
the ``smc_bars_since_sweep = 999`` sentinel convention; round grid 50/100 USD
= the operator-declared XAUUSD round-number convention (design doc §1.4).

Interpretation notes — decisions the design doc leaves open; each stays inside
the doc's stated conventions and is testable:

1. Break check: a level's break direction is anchored to its
   ``side_of_origin`` (high_pivot breaks UP on ``close > center + TOL*atr``,
   low_pivot breaks DOWN on ``close < center - TOL*atr``) — the registry
   analogue of the smc_v1 BOS crossings.  Because ``status`` flips to
   ``broken`` on the first firing bar, "first bar where the condition holds"
   IS the edge trigger; a freshly created level cannot satisfy its own break
   condition at birth (the confirming bar lies inside the pivot window), and a
   merged level whose center moved adopts the smc_v1 "level changed" refire
   idiom (`smc_v1.compute_smc_mtf_primitives_v1` BOS).
2. Reaction windows open on every touch-count increment (creation, pivot
   merge, zone-entry touch), with ``t0`` = the closed bar at which the
   registry observes the touch, ``atr[t0]`` and the level's post-update
   ``center_price`` at ``t0`` frozen for the measurement (rule 2g: the
   quantity is measured against the level as tested).  Windows complete at
   ``t0 + W`` regardless of a later break (a touch that never lifts off
   scores negative — the signed-reaction requirement).
3. Retest: the break bar itself is excluded (``t > break_bar``); the first
   bar within ``RETEST_WINDOW`` whose range intersects the zone is the retest
   bar; outcome hold requires close strictly on the break side of
   ``center_price`` — an exact tie fails closed to ``failed``.
4. Absent-slot encoding (no active level on a side): distance saturates at
   the 20 cap (the design's declared side-absence representation), the two
   bars-since style fields (``age_bars``, ``bars_since_touch``) carry the 999
   "no event yet" sentinel, counts and event-gated reaction stats are 0.
5. Nearest-slot side assignment: ``center > close`` is above, ``center <=
   close`` is below (an exact tie is measure-zero and deterministic);
   equidistant slot ties resolve to the lower ``level_id``, the same declared
   tie-break as pivot-merge ties (smc report §2).
6. Same-bar multiple retest outcomes aggregate by the sign of the summed
   signed outcomes (net zero on an exact conflict — fail-closed neutral);
   multiple same-bar breaks keep event = 1.0 with ``level_broken_touch_count``
   taking the max (the report's declared aggregation).
7. Expiry (``bar - last_touch_bar > AGE_CAP[tf]``, strict) is evaluated at
   the start of each closed bar, before admissions.
8. ATR availability follows the ``compute_smc_mtf_primitives_v1`` contract:
   an optional contiguous NaN prefix.  Pivots whose confirmation bar falls in
   that prefix are dropped (their admission distance is not computable; no
   fallback), and :func:`fit_level_registry_tolerance` drops the same pivots
   so the fitted population matches the admitted population (rule 2g).

Wiring: none here.  Builders/contracts consume the declared name tuples
``LEVEL_REGISTRY_M5_FEATURE_NAMES`` / ``LEVEL_REGISTRY_MTF_FEATURE_NAMES`` in
a separate stage-2 change (design doc §5.2).
"""
from __future__ import annotations

import copy
import math
from bisect import bisect_left, insort
from typing import Any, Mapping

import numpy as np
import pandas as pd

from gx1.features.smc_v1 import SWING_LOOKBACK, _detect_swing_pivots


LEVEL_REGISTRY_FEATURE_VERSION = (
    "level_registry_v1_20260811_phase_a_pivot_cluster_round_number"
)

# ---------------------------------------------------------------------------
# Level kinds — the full V29 enum (design doc §1).  Phase A implements
# pivot_cluster + round_number; session_anchored is Phase B.
# ---------------------------------------------------------------------------
LEVEL_KIND_PIVOT_CLUSTER = "pivot_cluster"
LEVEL_KIND_SESSION_ANCHORED = "session_anchored"
LEVEL_KIND_ROUND_NUMBER = "round_number"
LEVEL_KINDS = (
    LEVEL_KIND_PIVOT_CLUSTER,
    LEVEL_KIND_SESSION_ANCHORED,
    LEVEL_KIND_ROUND_NUMBER,
)
LEVEL_KINDS_IMPLEMENTED_V1 = (
    LEVEL_KIND_PIVOT_CLUSTER,
    LEVEL_KIND_ROUND_NUMBER,
)


def require_level_kind_implemented(kind: str) -> str:
    """Fail closed on unknown kinds and on the declared-but-unbuilt Phase B kind."""
    if kind not in LEVEL_KINDS:
        raise RuntimeError(f"[LEVEL_REGISTRY_KIND_UNKNOWN] {kind!r} not in {LEVEL_KINDS}")
    if kind not in LEVEL_KINDS_IMPLEMENTED_V1:
        raise RuntimeError(
            f"[LEVEL_REGISTRY_KIND_NOT_IMPLEMENTED] {kind!r} is declared for "
            "Phase B (design doc §1/§6) and has no Phase A producer"
        )
    return kind


# ---------------------------------------------------------------------------
# Named constants with declared origins (rule 2a).
# ---------------------------------------------------------------------------

# Reaction measurement window W, closed bars.  Origin: the family's tau-12
# short-horizon evidence-window convention (smc_sweep_size_recent_tau12,
# smc_bos_pressure_last12, smc_choch_recent_tau12 in entry_smart_context.py).
# Recorded as an immutable-recipe key by the design doc (§1.4) and flagged
# there for a TRAIN-fitted upgrade at the recipe decision.
LEVEL_REGISTRY_REACTION_WINDOW_BARS = 12

# Retest-after-break window, closed bars.  Origin: the family's tau-24
# recency convention (smc_sweep_recency_tau24, smc_choch_recent_tau24 in
# entry_smart_context.py; exp(-age/24) in entry_foundation_structure_v1).
# Immutable-recipe key per the design doc (§1.4).
LEVEL_REGISTRY_RETEST_WINDOW_BARS = 24

# Per-TF expiry: a level with bar - last_touch_bar > AGE_CAP is removed.
# Origin: the per-TF liquidity-zone lookbacks declared in
# augment_forward_outcome_v2._liquidity_zones (M5 240, M15 192, H1 168,
# H4 168, D1 60 closed bars) — named constants in an existing contract owner
# (design doc §1.4).
LEVEL_REGISTRY_AGE_CAP_BARS = {
    "m5": 240,
    "m15": 192,
    "h1": 168,
    "h4": 168,
    "d1": 60,
}

# ATR-distance saturation for every emitted *_dist_atr / *_reaction_atr
# field.  Origin: the family-wide exp(-min(x, 20.0)) proximity cap convention
# (sr_support_proximity_exp / sr_resistance_proximity_exp in
# entry_smart_context.py), semantics preserved monotonically (design doc §1.4).
LEVEL_REGISTRY_DIST_SATURATION_ATR = 20.0

# Count/age cap and "no event yet" sentinel.  Origin: the
# smc_bars_since_sweep = 999 sentinel-at-cap convention (smc_v1.py).
LEVEL_REGISTRY_COUNT_AGE_CAP = 999.0

# Round-number grid steps in USD.  Origin: the instrument's operator-declared
# round-number convention for XAUUSD (design doc §1.4; smc report §4 items
# 21-22 — analogous in kind to the classic-pivot formula constants in
# _build_daily_pivots).
LEVEL_ROUND_GRID_50_USD = 50.0
LEVEL_ROUND_GRID_100_USD = 100.0

# The TRAIN-fit quantile key `q` deliberately has NO default value here: the
# design doc defers its declared value to the immutable recipe decision (§8
# item 4), so it is a required explicit argument of
# fit_level_registry_tolerance (rule 2a origin class 3; inventing a default
# would violate rule 2b).
LEVEL_REGISTRY_TOL_QUANTILE_RECIPE_KEY = "level_registry_tol_quantile_q"


# ---------------------------------------------------------------------------
# Declared output contracts (exact names + order; stage 2 consumes these).
# M5/513 lane: design doc §1.2 (22 = 14 slots + 6 events + 2 round).
# Per-TF lane: design doc §1.3 (11 per TF).
# ---------------------------------------------------------------------------
LEVEL_REGISTRY_M5_FEATURE_NAMES = (
    "level_above_dist_atr",
    "level_above_touch_count",
    "level_above_age_bars",
    "level_above_bars_since_touch",
    "level_above_mean_reaction_atr",
    "level_above_max_reaction_atr",
    "level_above_last_reaction_atr",
    "level_below_dist_atr",
    "level_below_touch_count",
    "level_below_age_bars",
    "level_below_bars_since_touch",
    "level_below_mean_reaction_atr",
    "level_below_max_reaction_atr",
    "level_below_last_reaction_atr",
    "level_break_up_event",
    "level_break_down_event",
    "level_broken_touch_count",
    "level_bars_since_break",
    "level_retest_hold_signed",
    "level_retest_fail_signed",
    "level_round_50_dist_atr",
    "level_round_100_dist_atr",
)

LEVEL_REGISTRY_MTF_FEATURE_NAMES = (
    "mtf_level_above_dist_atr",
    "mtf_level_below_dist_atr",
    "mtf_level_above_touch_count",
    "mtf_level_below_touch_count",
    "mtf_level_above_mean_reaction_atr",
    "mtf_level_below_mean_reaction_atr",
    "mtf_level_break_up_event",
    "mtf_level_break_down_event",
    "mtf_level_bars_since_break",
    "mtf_level_retest_hold_signed",
    "mtf_level_retest_fail_signed",
)

# Per-TF lane fields are exact renames of registry fields (design doc §1.3).
_LEVEL_REGISTRY_MTF_SOURCE_MAP = {
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


# ---------------------------------------------------------------------------
# Serializable incremental state (chunk-safe carry, the
# build_entry_support_resistance_memory_layer memory_state pattern).  The
# caller owns chronological continuity of the chunks, exactly as with that
# owner's memory_state.
# ---------------------------------------------------------------------------
LEVEL_REGISTRY_STATE_VERSION = "level_registry_v1_state_1"
LEVEL_REGISTRY_STATE_KEYS = (
    "state_version",
    "tf",
    "tol_level_atr",
    "bars_processed",
    "atr_started",
    "first_level_admitted",
    "next_level_id",
    "last_break_bar",
    "tail_high",
    "tail_low",
    "levels",
)
LEVEL_REGISTRY_LEVEL_STATE_KEYS = (
    "level_id",
    "side_of_origin",
    "center_price",
    "member_pivot_count",
    "member_pivot_bars",
    "member_pivot_prices",
    "touch_count",
    "birth_bar",
    "last_touch_bar",
    "reaction_sum_atr",
    "reaction_max_atr",
    "reaction_last_atr",
    "completed_reaction_count",
    "status",
    "break_bar",
    "break_side",
    "retest_state",
    "prev_bar_in_zone",
    "open_reactions",
)
_SIDE_HIGH = "high_pivot"
_SIDE_LOW = "low_pivot"
_SIDES = (_SIDE_HIGH, _SIDE_LOW)
_STATUSES = ("active", "broken")
_RETEST_STATES = ("none", "pending", "held", "failed")


def _empty_registry_state(tf: str, tol_level_atr: float) -> dict[str, Any]:
    return {
        "state_version": LEVEL_REGISTRY_STATE_VERSION,
        "tf": tf,
        "tol_level_atr": float(tol_level_atr),
        "bars_processed": 0,
        "atr_started": False,
        "first_level_admitted": False,
        "next_level_id": 0,
        "last_break_bar": -1,
        "tail_high": [],
        "tail_low": [],
        "levels": [],
    }


def _validate_tf(tf: str) -> str:
    if tf not in LEVEL_REGISTRY_AGE_CAP_BARS:
        raise RuntimeError(
            f"[LEVEL_REGISTRY_TF_INVALID] {tf!r} not in "
            f"{tuple(LEVEL_REGISTRY_AGE_CAP_BARS)}"
        )
    return tf


def _validate_tol(tol_level_atr: float) -> float:
    try:
        tol = float(tol_level_atr)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("[LEVEL_REGISTRY_TOL_INVALID] not a number") from exc
    if not math.isfinite(tol) or tol <= 0.0:
        raise RuntimeError(
            f"[LEVEL_REGISTRY_TOL_INVALID] tol_level_atr must be finite and > 0, got {tol!r}"
        )
    return tol


def _validate_registry_state(
    state: Mapping[str, Any], *, tf: str, tol_level_atr: float
) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise RuntimeError("[LEVEL_REGISTRY_STATE_SCHEMA_INVALID] state is not a mapping")
    if set(state) != set(LEVEL_REGISTRY_STATE_KEYS):
        raise RuntimeError(
            "[LEVEL_REGISTRY_STATE_SCHEMA_INVALID] keys "
            f"{sorted(state)} != {sorted(LEVEL_REGISTRY_STATE_KEYS)}"
        )
    if state["state_version"] != LEVEL_REGISTRY_STATE_VERSION:
        raise RuntimeError(
            f"[LEVEL_REGISTRY_STATE_VERSION_MISMATCH] {state['state_version']!r}"
        )
    if state["tf"] != tf:
        raise RuntimeError(
            f"[LEVEL_REGISTRY_STATE_CONTRACT_MISMATCH] state tf={state['tf']!r} != {tf!r}"
        )
    # The tolerance is frozen bundle state (rule 18): carried state must bind
    # the exact same constant, bit for bit.
    if float(state["tol_level_atr"]) != float(tol_level_atr):
        raise RuntimeError(
            "[LEVEL_REGISTRY_STATE_CONTRACT_MISMATCH] state tol_level_atr="
            f"{state['tol_level_atr']!r} != {tol_level_atr!r}"
        )
    out = copy.deepcopy(dict(state))
    bars_processed = out["bars_processed"]
    if not isinstance(bars_processed, int) or isinstance(bars_processed, bool) or bars_processed < 0:
        raise RuntimeError("[LEVEL_REGISTRY_STATE_INVALID] bars_processed")
    for key in ("atr_started", "first_level_admitted"):
        if not isinstance(out[key], bool):
            raise RuntimeError(f"[LEVEL_REGISTRY_STATE_INVALID] {key}")
    if not isinstance(out["next_level_id"], int) or out["next_level_id"] < 0:
        raise RuntimeError("[LEVEL_REGISTRY_STATE_INVALID] next_level_id")
    if not isinstance(out["last_break_bar"], int) or out["last_break_bar"] < -1:
        raise RuntimeError("[LEVEL_REGISTRY_STATE_INVALID] last_break_bar")
    tail_high, tail_low = out["tail_high"], out["tail_low"]
    if (
        not isinstance(tail_high, list)
        or not isinstance(tail_low, list)
        or len(tail_high) != len(tail_low)
        or len(tail_high) > 2 * SWING_LOOKBACK
        or len(tail_high) > bars_processed
        or any(not isinstance(v, float) or not math.isfinite(v) for v in tail_high + tail_low)
    ):
        raise RuntimeError("[LEVEL_REGISTRY_STATE_INVALID] tail buffers")
    if not isinstance(out["levels"], list):
        raise RuntimeError("[LEVEL_REGISTRY_STATE_INVALID] levels not a list")
    for lv in out["levels"]:
        if not isinstance(lv, dict) or set(lv) != set(LEVEL_REGISTRY_LEVEL_STATE_KEYS):
            raise RuntimeError("[LEVEL_REGISTRY_STATE_INVALID] level schema")
        if lv["side_of_origin"] not in _SIDES:
            raise RuntimeError("[LEVEL_REGISTRY_STATE_INVALID] side_of_origin")
        if lv["status"] not in _STATUSES or lv["retest_state"] not in _RETEST_STATES:
            raise RuntimeError("[LEVEL_REGISTRY_STATE_INVALID] status/retest_state")
        for key in ("center_price", "reaction_sum_atr", "reaction_max_atr", "reaction_last_atr"):
            if not isinstance(lv[key], float) or not math.isfinite(lv[key]):
                raise RuntimeError(f"[LEVEL_REGISTRY_STATE_INVALID] level {key}")
        for key in (
            "level_id",
            "member_pivot_count",
            "touch_count",
            "birth_bar",
            "last_touch_bar",
            "completed_reaction_count",
            "break_bar",
            "break_side",
        ):
            if not isinstance(lv[key], int) or isinstance(lv[key], bool):
                raise RuntimeError(f"[LEVEL_REGISTRY_STATE_INVALID] level {key}")
        if not isinstance(lv["prev_bar_in_zone"], bool):
            raise RuntimeError("[LEVEL_REGISTRY_STATE_INVALID] level prev_bar_in_zone")
        if not isinstance(lv["member_pivot_bars"], list) or not isinstance(
            lv["member_pivot_prices"], list
        ):
            raise RuntimeError("[LEVEL_REGISTRY_STATE_INVALID] level members")
        if not isinstance(lv["open_reactions"], list):
            raise RuntimeError("[LEVEL_REGISTRY_STATE_INVALID] level open_reactions")
        for window in lv["open_reactions"]:
            if (
                not isinstance(window, list)
                or len(window) != 4
                or not isinstance(window[0], int)
                or not isinstance(window[1], float)
                or not math.isfinite(window[1])
                or window[1] <= 0.0
                or not isinstance(window[2], float)
                or not math.isfinite(window[2])
                or not (window[3] is None or (isinstance(window[3], float) and math.isfinite(window[3])))
            ):
                raise RuntimeError("[LEVEL_REGISTRY_STATE_INVALID] level reaction window")
    return out


# ---------------------------------------------------------------------------
# Input validation — the sibling owners' fail-closed standard
# (compute_smc_features / compute_smc_mtf_primitives_v1).
# ---------------------------------------------------------------------------
def _validate_source(
    df: pd.DataFrame,
    *,
    high_col: str,
    low_col: str,
    close_col: str,
    atr_col: str,
    atr_started: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(df) == 0:
        raise RuntimeError("[LEVEL_REGISTRY_SOURCE_EMPTY]")
    required = (high_col, low_col, close_col, atr_col)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RuntimeError(f"[LEVEL_REGISTRY_SOURCE_MISSING] {missing}")
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
        raise RuntimeError("[LEVEL_REGISTRY_SOURCE_NONFINITE]")
    atr_available = np.isfinite(atr)
    if atr_started:
        if not atr_available.all():
            raise RuntimeError(
                "[LEVEL_REGISTRY_ATR_AVAILABILITY_INVALID] carried state declares "
                "ATR started but the chunk contains unavailable ATR rows"
            )
    elif atr_available.any():
        first_atr = int(np.argmax(atr_available))
        if not atr_available[first_atr:].all():
            raise RuntimeError("[LEVEL_REGISTRY_ATR_AVAILABILITY_INVALID]")
    if (
        np.any(high <= 0.0)
        or np.any(low <= 0.0)
        or np.any(close <= 0.0)
        or np.any(atr[atr_available] <= 0.0)
        or np.any(high < low)
        or np.any(high < close)
        or np.any(low > close)
    ):
        raise RuntimeError("[LEVEL_REGISTRY_SOURCE_GEOMETRY_INVALID]")
    return high, low, close, atr


# ---------------------------------------------------------------------------
# TRAIN fit of the cluster tolerance (smc report §2 — required TRAIN-fitted
# statistic; rule 18 pattern: fitted once on the complete declared TRAIN
# window, frozen as bundle state, never refitted downstream).
# ---------------------------------------------------------------------------
def fit_level_registry_tolerance(
    df: pd.DataFrame,
    *,
    q: float,
    tf: str,
    declared_train_window: str,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    atr_col: str = "atr",
) -> tuple[float, dict[str, Any]]:
    """Fit ``TOL_LEVEL_ATR[tf]`` on a declared TRAIN window.

    For every confirmed swing pivot (in confirmation order, high before low on
    a shared confirmation bar — the same order the runtime admission uses),
    the sample is the ATR-normalized absolute distance from its price to the
    nearest *earlier* confirmed pivot price; the tolerance is the ``q``
    quantile of that sample.  ``q`` is a required explicit recipe input
    (``LEVEL_REGISTRY_TOL_QUANTILE_RECIPE_KEY``); the design doc declares no
    default (§8 item 4), so none exists here.

    Returns ``(tol_level_atr, provenance)`` where provenance states the sample
    size and the quantile's sampling bound (rule 2f): ``quantile_prob_se =
    sqrt(q*(1-q)/N)`` (the distribution-free binomial SE of the empirical CDF
    at ``q``) and the empirical value bracket at ``q ± quantile_prob_se``.
    """
    _validate_tf(tf)
    try:
        q_value = float(q)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("[LEVEL_REGISTRY_TOL_FIT_Q_INVALID] not a number") from exc
    if not math.isfinite(q_value) or not (0.0 < q_value < 1.0):
        raise RuntimeError(
            f"[LEVEL_REGISTRY_TOL_FIT_Q_INVALID] q must lie strictly in (0, 1), got {q_value!r}"
        )
    if not isinstance(declared_train_window, str) or not declared_train_window:
        raise RuntimeError("[LEVEL_REGISTRY_TOL_FIT_WINDOW_UNDECLARED]")
    high, low, _close, atr = _validate_source(
        df,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        atr_col=atr_col,
        atr_started=False,
    )
    nb = len(high)
    n = SWING_LOOKBACK
    swing_high_mask, swing_low_mask = _detect_swing_pivots(high, low, n)
    distances: list[float] = []
    earlier_prices: list[float] = []
    admitted = 0
    dropped_atr_unavailable = 0
    for t in range(n, nb):
        j = t - n
        for is_pivot, price in (
            (bool(swing_high_mask[j]), float(high[j])),
            (bool(swing_low_mask[j]), float(low[j])),
        ):
            if not is_pivot:
                continue
            a = float(atr[t])
            if not math.isfinite(a):
                # Confirmation falls in the declared ATR warmup prefix: the
                # runtime drops this pivot, so the fit drops it too (rule 2g).
                dropped_atr_unavailable += 1
                continue
            if earlier_prices:
                pos = bisect_left(earlier_prices, price)
                nearest = math.inf
                if pos < len(earlier_prices):
                    nearest = min(nearest, abs(earlier_prices[pos] - price))
                if pos > 0:
                    nearest = min(nearest, abs(earlier_prices[pos - 1] - price))
                distances.append(nearest / a)
            insort(earlier_prices, price)
            admitted += 1
    if not distances:
        raise RuntimeError(
            "[LEVEL_REGISTRY_TOL_FIT_INSUFFICIENT] fewer than two admitted "
            f"pivots in the declared window (admitted={admitted})"
        )
    sample = np.asarray(distances, dtype=np.float64)
    sample_size = int(sample.size)
    tol = float(np.quantile(sample, q_value))
    if not math.isfinite(tol) or tol <= 0.0:
        raise RuntimeError(
            f"[LEVEL_REGISTRY_TOL_FIT_DEGENERATE] fitted tolerance {tol!r} is not "
            "a usable zone half-width"
        )
    prob_se = math.sqrt(q_value * (1.0 - q_value) / sample_size)
    bracket_q_lo = max(0.0, q_value - prob_se)
    bracket_q_hi = min(1.0, q_value + prob_se)
    bracket = np.quantile(sample, [bracket_q_lo, bracket_q_hi])
    provenance = {
        "fit_owner": "gx1.features.level_registry_v1.fit_level_registry_tolerance",
        "module_version": LEVEL_REGISTRY_FEATURE_VERSION,
        "level_kind": LEVEL_KIND_PIVOT_CLUSTER,
        "tf": tf,
        "declared_train_window": declared_train_window,
        "swing_lookback": int(SWING_LOOKBACK),
        "quantile_recipe_key": LEVEL_REGISTRY_TOL_QUANTILE_RECIPE_KEY,
        "quantile_q": q_value,
        "quantile_method": "linear",
        "n_bars": int(nb),
        "n_pivots_admitted": admitted,
        "n_pivots_dropped_atr_unavailable": dropped_atr_unavailable,
        "sample_size": sample_size,
        "tol_level_atr": tol,
        "quantile_prob_se": prob_se,
        "tol_bracket_q_lo": float(bracket_q_lo),
        "tol_bracket_q_hi": float(bracket_q_hi),
        "tol_bracket_lo": float(bracket[0]),
        "tol_bracket_hi": float(bracket[1]),
    }
    return tol, provenance


# ---------------------------------------------------------------------------
# Registry engine (one code path for full runs and chunked runs — chunk
# invariance is by construction: scalar float64 arithmetic per closed bar).
# ---------------------------------------------------------------------------
def _saturate(value: float) -> float:
    cap = LEVEL_REGISTRY_DIST_SATURATION_ATR
    if value > cap:
        return cap
    if value < -cap:
        return -cap
    return value


def _cap999(value: float) -> float:
    cap = LEVEL_REGISTRY_COUNT_AGE_CAP
    if value > cap:
        return cap
    if value < 0.0:
        return 0.0
    return value


def _new_level(
    level_id: int, side: str, price: float, birth_bar: int, pivot_bar: int, atr_now: float
) -> dict[str, Any]:
    return {
        "level_id": level_id,
        "side_of_origin": side,
        "center_price": price,
        "member_pivot_count": 1,
        "member_pivot_bars": [pivot_bar],
        "member_pivot_prices": [price],
        "touch_count": 1,
        "birth_bar": birth_bar,
        "last_touch_bar": pivot_bar,
        "reaction_sum_atr": 0.0,
        "reaction_max_atr": 0.0,
        "reaction_last_atr": 0.0,
        "completed_reaction_count": 0,
        "status": "active",
        "break_bar": -1,
        "break_side": 0,
        "retest_state": "none",
        "prev_bar_in_zone": False,
        "open_reactions": [[birth_bar, atr_now, price, None]],
    }


def _slot_fields(level: dict[str, Any] | None, dist: float, t: int) -> tuple[float, ...]:
    if level is None:
        # Absent-side encoding: distance saturates at the cap (the design's
        # declared side-absence representation); bars-since fields carry the
        # 999 "no event yet" sentinel; counts/reaction stats are event-gated 0.
        return (
            LEVEL_REGISTRY_DIST_SATURATION_ATR,
            0.0,
            LEVEL_REGISTRY_COUNT_AGE_CAP,
            LEVEL_REGISTRY_COUNT_AGE_CAP,
            0.0,
            0.0,
            0.0,
        )
    count = level["completed_reaction_count"]
    mean_reaction = level["reaction_sum_atr"] / count if count > 0 else 0.0
    return (
        min(dist, LEVEL_REGISTRY_DIST_SATURATION_ATR),
        _cap999(float(level["touch_count"])),
        _cap999(float(t - level["birth_bar"])),
        _cap999(float(t - level["last_touch_bar"])),
        _saturate(mean_reaction) if count > 0 else 0.0,
        _saturate(level["reaction_max_atr"]) if count > 0 else 0.0,
        _saturate(level["reaction_last_atr"]) if count > 0 else 0.0,
    )


def _round_grid_dist_atr(close: float, atr_now: float, grid_step: float) -> float:
    nearest = round(close / grid_step) * grid_step
    return _saturate((close - nearest) / atr_now)


def _run_level_registry(
    df: pd.DataFrame,
    *,
    tf: str,
    tol_level_atr: float,
    high_col: str,
    low_col: str,
    close_col: str,
    atr_col: str,
    registry_state: Mapping[str, Any] | None,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    _validate_tf(tf)
    tol = _validate_tol(tol_level_atr)
    if registry_state is None:
        state = _empty_registry_state(tf, tol)
    else:
        state = _validate_registry_state(registry_state, tf=tf, tol_level_atr=tol)
    high, low, close, atr = _validate_source(
        df,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        atr_col=atr_col,
        atr_started=state["atr_started"],
    )
    n = SWING_LOOKBACK
    window_len = 2 * n + 1
    age_cap = LEVEL_REGISTRY_AGE_CAP_BARS[tf]
    reaction_w = LEVEL_REGISTRY_REACTION_WINDOW_BARS
    retest_w = LEVEL_REGISTRY_RETEST_WINDOW_BARS

    buf_high: list[float] = list(state["tail_high"])
    buf_low: list[float] = list(state["tail_low"])
    levels: list[dict[str, Any]] = state["levels"]
    next_level_id = state["next_level_id"]
    last_break_bar = state["last_break_bar"]
    atr_started = state["atr_started"]
    first_level_admitted = state["first_level_admitted"]
    base = state["bars_processed"]

    record: dict[str, list[float]] = {name: [] for name in LEVEL_REGISTRY_M5_FEATURE_NAMES}

    def emit_nan_row() -> None:
        for values in record.values():
            values.append(math.nan)

    nb = len(high)
    for i in range(nb):
        t = base + i
        h = float(high[i])
        l = float(low[i])
        c = float(close[i])
        a = float(atr[i])

        buf_high.append(h)
        buf_low.append(l)
        if len(buf_high) > window_len:
            del buf_high[0]
            del buf_low[0]

        pivot_high = pivot_low = False
        pivot_high_price = pivot_low_price = math.nan
        if len(buf_high) == window_len:
            # One pivot truth (rule 13): execute the smc_v1 owner detector on
            # the causal window whose center is the confirming pivot bar
            # j = t - SWING_LOOKBACK.
            sh_mask, sl_mask = _detect_swing_pivots(
                np.asarray(buf_high, dtype=np.float64),
                np.asarray(buf_low, dtype=np.float64),
                n,
            )
            pivot_high = bool(sh_mask[n])
            pivot_low = bool(sl_mask[n])
            pivot_high_price = buf_high[n]
            pivot_low_price = buf_low[n]

        if not math.isfinite(a):
            # Declared ATR warmup prefix: no registry processing is possible
            # (admission distances are ATR-normalized); the row is warmup NaN.
            emit_nan_row()
            continue
        atr_started = True
        half = tol * a

        # (0) expiry — strict per the smc report §2 (bar - last_touch_bar > cap).
        if levels:
            levels[:] = [lv for lv in levels if t - lv["last_touch_bar"] <= age_cap]

        # (1) pivot admissions / merges — high before low on a shared bar.
        admitted_ids: set[int] = set()
        if pivot_high or pivot_low:
            j = t - n
            for is_pivot, side, price in (
                (pivot_high, _SIDE_HIGH, pivot_high_price),
                (pivot_low, _SIDE_LOW, pivot_low_price),
            ):
                if not is_pivot:
                    continue
                best: dict[str, Any] | None = None
                best_dist = math.inf
                for lv in levels:
                    if lv["status"] != "active":
                        continue
                    d = abs(price - lv["center_price"]) / a
                    if d > tol:
                        continue
                    if (
                        best is None
                        or d < best_dist
                        or (d == best_dist and lv["level_id"] < best["level_id"])
                    ):
                        best = lv
                        best_dist = d
                if best is not None:
                    best["touch_count"] += 1
                    best["member_pivot_count"] += 1
                    best["center_price"] += (price - best["center_price"]) / best[
                        "member_pivot_count"
                    ]
                    best["member_pivot_bars"].append(j)
                    best["member_pivot_prices"].append(price)
                    best["last_touch_bar"] = j
                    best["open_reactions"].append([t, a, best["center_price"], None])
                    admitted_ids.add(best["level_id"])
                else:
                    lv = _new_level(next_level_id, side, price, t, j, a)
                    next_level_id += 1
                    levels.append(lv)
                    admitted_ids.add(lv["level_id"])
                first_level_admitted = True

        # (2) break checks — side-of-origin anchored crossings, break-once.
        break_up_fired = False
        break_down_fired = False
        broken_touch_count = 0
        for lv in levels:
            if lv["status"] != "active":
                continue
            if lv["side_of_origin"] == _SIDE_HIGH:
                if c > lv["center_price"] + half:
                    lv["status"] = "broken"
                    lv["break_bar"] = t
                    lv["break_side"] = 1
                    lv["retest_state"] = "pending"
                    break_up_fired = True
                    broken_touch_count = max(broken_touch_count, lv["touch_count"])
                    last_break_bar = t
            else:
                if c < lv["center_price"] - half:
                    lv["status"] = "broken"
                    lv["break_bar"] = t
                    lv["break_side"] = -1
                    lv["retest_state"] = "pending"
                    break_down_fired = True
                    broken_touch_count = max(broken_touch_count, lv["touch_count"])
                    last_break_bar = t

        # (3) retest checks — first zone re-entry strictly after the break bar.
        hold_sign_sum = 0
        fail_sign_sum = 0
        for lv in levels:
            if lv["status"] != "broken" or lv["retest_state"] != "pending":
                continue
            if lv["break_bar"] == t:
                continue
            if t - lv["break_bar"] > retest_w:
                # Window expiry without re-entry: none-equivalent, no event
                # (rule 2e — no placeholder outcome).
                lv["retest_state"] = "none"
                continue
            if h >= lv["center_price"] - half and l <= lv["center_price"] + half:
                if (c - lv["center_price"]) * lv["break_side"] > 0:
                    lv["retest_state"] = "held"
                    hold_sign_sum += lv["break_side"]
                else:
                    # Includes the exact tie close == center (fail-closed).
                    lv["retest_state"] = "failed"
                    fail_sign_sum += lv["break_side"]

        # (4) touch checks — first-entry-per-excursion; a same-bar admission
        # suppresses the touch check for that level (counts once).
        for lv in levels:
            if lv["status"] != "active" or lv["level_id"] in admitted_ids:
                continue
            in_zone = h >= lv["center_price"] - half and l <= lv["center_price"] + half
            if in_zone and not lv["prev_bar_in_zone"]:
                lv["touch_count"] += 1
                lv["last_touch_bar"] = t
                lv["open_reactions"].append([t, a, lv["center_price"], None])

        # (5) reaction-window updates and completions (confirmation-lagged at
        # t0 + W; windows survive breaks — a touch that never lifts off scores
        # negative).
        for lv in levels:
            if not lv["open_reactions"]:
                continue
            is_low_side = lv["side_of_origin"] == _SIDE_LOW
            keep: list[list[Any]] = []
            for window in lv["open_reactions"]:
                t0, atr0, center0, extreme = window
                if t > t0:
                    if is_low_side:
                        extreme = h if extreme is None else max(extreme, h)
                    else:
                        extreme = l if extreme is None else min(extreme, l)
                    window[3] = extreme
                if t == t0 + reaction_w:
                    if is_low_side:
                        value = (extreme - center0) / atr0
                    else:
                        value = (center0 - extreme) / atr0
                    lv["reaction_sum_atr"] += value
                    lv["completed_reaction_count"] += 1
                    lv["reaction_last_atr"] = value
                    if lv["completed_reaction_count"] == 1:
                        lv["reaction_max_atr"] = value
                    else:
                        lv["reaction_max_atr"] = max(lv["reaction_max_atr"], value)
                else:
                    keep.append(window)
            lv["open_reactions"] = keep

        # (6) excursion memory for the next bar's touch dedup.
        for lv in levels:
            if lv["status"] != "active":
                continue
            lv["prev_bar_in_zone"] = (
                h >= lv["center_price"] - half and l <= lv["center_price"] + half
            )

        # (7) emission (post-update state; broken levels are never "nearest
        # ACTIVE").
        if not first_level_admitted:
            emit_nan_row()
            continue
        above: dict[str, Any] | None = None
        below: dict[str, Any] | None = None
        above_dist = below_dist = math.inf
        for lv in levels:
            if lv["status"] != "active":
                continue
            delta = lv["center_price"] - c
            if delta > 0.0:
                d = delta / a
                if d < above_dist or (d == above_dist and above is not None and lv["level_id"] < above["level_id"]):
                    above = lv
                    above_dist = d
            else:
                d = -delta / a
                if d < below_dist or (d == below_dist and below is not None and lv["level_id"] < below["level_id"]):
                    below = lv
                    below_dist = d
        above_vals = _slot_fields(above, above_dist, t)
        below_vals = _slot_fields(below, below_dist, t)
        record["level_above_dist_atr"].append(above_vals[0])
        record["level_above_touch_count"].append(above_vals[1])
        record["level_above_age_bars"].append(above_vals[2])
        record["level_above_bars_since_touch"].append(above_vals[3])
        record["level_above_mean_reaction_atr"].append(above_vals[4])
        record["level_above_max_reaction_atr"].append(above_vals[5])
        record["level_above_last_reaction_atr"].append(above_vals[6])
        record["level_below_dist_atr"].append(below_vals[0])
        record["level_below_touch_count"].append(below_vals[1])
        record["level_below_age_bars"].append(below_vals[2])
        record["level_below_bars_since_touch"].append(below_vals[3])
        record["level_below_mean_reaction_atr"].append(below_vals[4])
        record["level_below_max_reaction_atr"].append(below_vals[5])
        record["level_below_last_reaction_atr"].append(below_vals[6])
        record["level_break_up_event"].append(1.0 if break_up_fired else 0.0)
        record["level_break_down_event"].append(1.0 if break_down_fired else 0.0)
        record["level_broken_touch_count"].append(
            _cap999(float(broken_touch_count)) if (break_up_fired or break_down_fired) else 0.0
        )
        record["level_bars_since_break"].append(
            _cap999(float(t - last_break_bar)) if last_break_bar >= 0 else LEVEL_REGISTRY_COUNT_AGE_CAP
        )
        record["level_retest_hold_signed"].append(
            float((hold_sign_sum > 0) - (hold_sign_sum < 0))
        )
        record["level_retest_fail_signed"].append(
            float((fail_sign_sum > 0) - (fail_sign_sum < 0))
        )
        record["level_round_50_dist_atr"].append(
            _round_grid_dist_atr(c, a, LEVEL_ROUND_GRID_50_USD)
        )
        record["level_round_100_dist_atr"].append(
            _round_grid_dist_atr(c, a, LEVEL_ROUND_GRID_100_USD)
        )

    out_state = {
        "state_version": LEVEL_REGISTRY_STATE_VERSION,
        "tf": tf,
        "tol_level_atr": tol,
        "bars_processed": base + nb,
        "atr_started": atr_started,
        "first_level_admitted": first_level_admitted,
        "next_level_id": next_level_id,
        "last_break_bar": last_break_bar,
        "tail_high": buf_high[-(2 * n):] if len(buf_high) > 2 * n else list(buf_high),
        "tail_low": buf_low[-(2 * n):] if len(buf_low) > 2 * n else list(buf_low),
        "levels": levels,
    }
    return record, out_state


def _finalize_block(
    record: dict[str, list[float]],
    declared_names: tuple[str, ...],
    source_names: tuple[str, ...],
) -> tuple[np.ndarray, list[str]]:
    emitted = tuple(record)
    if emitted != LEVEL_REGISTRY_M5_FEATURE_NAMES:
        raise RuntimeError("[LEVEL_REGISTRY_FEATURE_NAME_DRIFT] engine emission order")
    columns = []
    for source in source_names:
        if source not in record:
            raise RuntimeError(f"[LEVEL_REGISTRY_FEATURE_NAME_DRIFT] missing source {source}")
        columns.append(np.asarray(record[source], dtype=np.float32))
    matrix = np.column_stack(columns) if columns else np.empty((0, 0), dtype=np.float32)
    if matrix.shape[1] != len(declared_names):
        raise RuntimeError("[LEVEL_REGISTRY_FEATURE_NAME_DRIFT] column count")
    if np.isinf(matrix).any():
        raise RuntimeError("[LEVEL_REGISTRY_OUTPUT_NONFINITE]")
    complete = np.isfinite(matrix).all(axis=1)
    incomplete_any = ~np.isfinite(matrix).any(axis=1)
    if complete.any():
        first_complete = int(np.argmax(complete))
        if not complete[first_complete:].all() or not incomplete_any[:first_complete].all():
            raise RuntimeError("[LEVEL_REGISTRY_OUTPUT_AVAILABILITY_INVALID]")
    elif not incomplete_any.all():
        raise RuntimeError("[LEVEL_REGISTRY_OUTPUT_AVAILABILITY_INVALID]")
    return matrix, list(declared_names)


def compute_level_registry_m5_block_v1(
    df: pd.DataFrame,
    *,
    tol_level_atr: float,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    atr_col: str = "atr",
    registry_state: Mapping[str, Any] | None = None,
    return_registry_state: bool = False,
) -> (
    tuple[np.ndarray, list[str]]
    | tuple[np.ndarray, list[str], dict[str, Any]]
):
    """M5/513 lane: the 22-field ``level_`` block (design doc §1.2).

    Runs on the entry M5 clock (tf is bound to ``"m5"``; the expiry cap is the
    M5 liquidity-zone lookback).  ``tol_level_atr`` is the TRAIN-fitted frozen
    constant from :func:`fit_level_registry_tolerance`.  Rows are NaN until
    the first admitted level (single chronological warmup prefix, trimmed
    downstream); afterwards no mid-series NaN is possible.
    """
    record, state = _run_level_registry(
        df,
        tf="m5",
        tol_level_atr=tol_level_atr,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        atr_col=atr_col,
        registry_state=registry_state,
    )
    matrix, names = _finalize_block(
        record, LEVEL_REGISTRY_M5_FEATURE_NAMES, LEVEL_REGISTRY_M5_FEATURE_NAMES
    )
    if return_registry_state:
        return matrix, names, state
    return matrix, names


def compute_level_registry_mtf_block_v1(
    df: pd.DataFrame,
    *,
    tf: str,
    tol_level_atr: float,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    atr_col: str = "atr",
    registry_state: Mapping[str, Any] | None = None,
    return_registry_state: bool = False,
) -> (
    tuple[np.ndarray, list[str]]
    | tuple[np.ndarray, list[str], dict[str, Any]]
):
    """Per-TF V4 lane: the 11-field ``mtf_level_`` block (design doc §1.3).

    The same registry engine runs independently on each timeframe's closed
    bars (``tf`` selects that TF's expiry cap; ``tol_level_atr`` is that TF's
    TRAIN-fitted frozen constant).  Field values are exact renames of the
    registry fields — pivot_cluster kind only; round numbers and session
    anchors do not replicate per TF.
    """
    if tuple(_LEVEL_REGISTRY_MTF_SOURCE_MAP) != LEVEL_REGISTRY_MTF_FEATURE_NAMES:
        raise RuntimeError("[LEVEL_REGISTRY_FEATURE_NAME_DRIFT] mtf source map")
    record, state = _run_level_registry(
        df,
        tf=tf,
        tol_level_atr=tol_level_atr,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        atr_col=atr_col,
        registry_state=registry_state,
    )
    matrix, names = _finalize_block(
        record,
        LEVEL_REGISTRY_MTF_FEATURE_NAMES,
        tuple(
            _LEVEL_REGISTRY_MTF_SOURCE_MAP[name]
            for name in LEVEL_REGISTRY_MTF_FEATURE_NAMES
        ),
    )
    if return_registry_state:
        return matrix, names, state
    return matrix, names
