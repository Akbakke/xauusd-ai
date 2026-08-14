"""V29 level registry V1 — causal immutable-pivot horizontal levels.

New causal authority (rule 21): persistent horizontal price-level identity
with touch/reaction memory and break/retest events.  Design owner:
``docs/V29_EVENT_SURFACE_DESIGN_20260811.md``
§1; exact field/lifecycle spec: the smc_liquidity event-gap report of
2026-08-11 (``/home/andre2/GX1_DATA/logs/event_gap_review_20260811/
smc_liquidity.md``, CENTERPIECE §§0-5).  This owner implements immutable
``pivot_anchor`` identities; ``session_anchored`` remains declared but unbuilt.

Pivot truth (rule 13): swing pivots are detected by *executing*
``smc_v1._detect_swing_pivots`` (owner code, not a copy) on the causal
``2*SWING_LOOKBACK+1`` bar window, and a pivot at bar ``j`` is confirmed at bar
``j + SWING_LOOKBACK`` — the published confirmation rule of
``smc_v1._track_recent_swings``.  No second pivot detector exists.

Constant origins (rule 2a — one sentence each, at the constant definitions
below): pivot lookback ``SWING_LOOKBACK`` imported from ``smc_v1``; recurrence
threshold and evidence lifetime are chronological TRAIN-fitted competing-risk
parameters produced by :func:`fit_level_registry_hyperparameters_v1` and are
explicit inputs of every compute function (never module defaults); distance
All distances, counts, ages, and reactions are emitted in raw units;
current-slot presence masks encode current absence while global event ages
remain NaN until their first genuine event. The former fixed
50/100 USD round-number inputs are retired: no TRAIN artifact owned those
periodicities, so they were not admissible market evidence.

Interpretation notes — decisions the design doc leaves open; each stays inside
the doc's stated conventions and is testable:

1. Break check: a level's break direction is anchored to its
   ``side_of_origin`` (high_pivot breaks UP on ``close > center`` and
   low_pivot breaks DOWN on ``close < center``) — the registry
   analogue of the smc_v1 BOS crossings.  Because ``status`` flips to
   ``broken`` on the first firing bar, "first bar where the condition holds"
   IS the edge trigger; a freshly created level cannot satisfy its own break
   condition at birth (the confirming bar lies inside the pivot window), and a
   immutable anchor never moves after confirmation.
2. Reaction lifecycle opens on a touch and completes at the first subsequent
   distinct touch or break event; the outer TRAIN boundary
   right-censors unresolved episodes.
   The raw event age is recorded without a fixed reaction window.
3. Retest: the break bar itself is excluded (``t > break_bar``); the first
   later bar whose range intersects the exact anchor price is the retest
   bar; outcome hold requires close strictly on the break side of
   ``center_price`` — an exact tie fails closed to ``failed``. A HOLD
   also FLIPS the level's polarity (V30 package 8A, 2026-08-13): broken
   resistance that is retested and holds becomes an ACTIVE support and vice
   versa, keeping its identity, member pivots, touch history and reaction
   memory. ``side_of_origin`` is therefore lifecycle state, not a birth
   constant; the level's break check and nearest-ACTIVE slot participation
   both follow the new side from the flip bar onward. A
   FAILED retest is unchanged: the break stands and the level stays broken.
4. Absent-slot encoding (no active level on a side): raw attributes are 0 and
   explicit current ``*_present`` masks are 0. Break age is NaN before the
   first genuine break and therefore needs no eventually constant seen mask.
5. Nearest-slot side assignment: ``center > close`` is above, ``center <=
   close`` is below (an exact tie is measure-zero and deterministic);
   equidistant slot ties resolve to the lower ``level_id``, the same declared
   ``level_id`` tie-break.
6. Same-bar multiple retest outcomes aggregate by the sign of the summed
   signed outcomes (net zero on an exact conflict — fail-closed neutral);
   multiple same-bar breaks keep event = 1.0 with ``level_broken_touch_count``
   taking the max (the report's declared aggregation).
7. Learned lifetime moves an anchor out of ACTIVE/pending event and slot
   eligibility when raw ``bar-last_eligibility_refresh_bar`` exceeds it. It
   never deletes or mutates immutable identity or recurrence membership.
8. ATR availability follows the ``compute_smc_mtf_primitives_v1`` contract:
   an optional contiguous NaN prefix.  Pivots whose confirmation bar falls in
   that prefix are dropped (their admission distance is not computable; no
   fallback), and the hyperparameter fit drops the same pivots
   so the fitted population matches the admitted population (rule 2g).
9. Registry identity and recurrence membership form one parameter-independent
   canonical tape: every confirmed pivot creates an immutable high- or
   low-birth-side anchor. Nearest recurrence is found among ALL prior immutable
   anchors of the same birth side through a price index; threshold and lifetime
   cannot change its origin/distance bytes. Breaks, touches and retests use exact
   anchor geometry on the eligible event set. TRAIN outcome labels come from
   the parameter-independent exact-center reaction/break/censor lifecycle;
   serving never consumes those future labels. The frozen threshold only marks
   the birth recurrence flag; lifetime only gates selected event/slot
   eligibility. Separate hashes bind recurrence fit/apply parity, canonical
   TRAIN outcome labels, selected eligibility replay and emissions.

Wiring: none here.  Builders/contracts consume the declared name tuples
``LEVEL_REGISTRY_M5_FEATURE_NAMES`` / ``LEVEL_REGISTRY_MTF_FEATURE_NAMES`` in
a separate stage-2 change (design doc §5.2).
"""
from __future__ import annotations

import copy
import hashlib
import math
from bisect import bisect_left, insort
from typing import Any, Mapping

import numpy as np
import pandas as pd

from gx1.contracts.registry_hyperparameter_fit_v1 import (
    REGISTRY_OUTCOME_BREAK,
    REGISTRY_OUTCOME_CENSORED,
    REGISTRY_OUTCOME_REACTION,
    RegistryOutcomeStreamV1,
    fit_registry_competing_risk_threshold_v1,
    require_registry_fit_source_v1,
    require_registry_hyperparameter_payload_v1,
)
from gx1.utils.artifact_primitives_v1 import canonical_json_sha256

from gx1.features.smc_v1 import SWING_LOOKBACK, _detect_swing_pivots
from gx1.features.event_age_v1 import raw_event_age_from_last_observed_row


LEVEL_REGISTRY_FEATURE_VERSION = (
    "level_registry_v13_honest_raw_break_age_no_global_seen"
)
# ---------------------------------------------------------------------------
# Level kinds — only implemented or explicitly reserved identities remain.
# ---------------------------------------------------------------------------
LEVEL_KIND_PIVOT_ANCHOR = "pivot_anchor"
LEVEL_KIND_SESSION_ANCHORED = "session_anchored"
LEVEL_KINDS = (
    LEVEL_KIND_PIVOT_ANCHOR,
    LEVEL_KIND_SESSION_ANCHORED,
)
LEVEL_KINDS_IMPLEMENTED_V1 = (
    LEVEL_KIND_PIVOT_ANCHOR,
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

LEVEL_REGISTRY_TIMEFRAMES = ("m1", "m5", "m15", "h1", "h4", "d1")

# ---------------------------------------------------------------------------
# Declared output contracts (exact names + order; stage 2 consumes these).
# Local and per-TF native-clock lanes expose the same 31 causal
# anchor/lifecycle primitives. Current slots retain presence masks; sparse
# break memory is an honest NaN prefix followed by raw age.
# ---------------------------------------------------------------------------
# V30 package 8A (2026-08-13) appends four EMISSION-ONLY fields per the
# Completed-reaction count disambiguates honest zero reaction magnitudes from
# "no completed reaction". The second-nearest distance exposes shelf density.
#   level_above2/below2_dist_atr — the SECOND-nearest ACTIVE level on that
#     side.  Today a four-level shelf and an isolated level emit identical
#     rows. Explicit presence masks carry absence, with the same side rule and
#     level_id tie-break.
LEVEL_REGISTRY_M5_FEATURE_NAMES = (
    "level_above_dist_atr",
    "level_above_present",
    "level_above2_dist_atr",
    "level_above2_present",
    "level_above_touch_count",
    "level_above_completed_reaction_count",
    # Confirmed same-side recurrence pivots inside the TRAIN-frozen distance;
    # exact-center touches remain a separate primitive.
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
    # V30 (2026-08-13): the same bars-since-break memory signed by the break
    # side of the most recent break bar (+1 up / -1 down; a same-bar
    # up+down conflict nets to 0 — the sibling signed-aggregation
    # convention of interpretation note 6). The pre-event value is NaN.
    "level_bars_since_break_signed",
    "level_retest_hold_signed",
    "level_retest_fail_signed",
)

LEVEL_REGISTRY_MTF_FEATURE_NAMES = tuple(
    f"mtf_{name}" for name in LEVEL_REGISTRY_M5_FEATURE_NAMES
)

# Per-TF lane fields are exhaustive exact renames of the local owner.  A
# native H1/H4/D1 clock must not receive a half-indicator missing recurrence,
# shelf density, lifecycle ages or reaction extrema.
_LEVEL_REGISTRY_MTF_SOURCE_MAP = {
    mtf_name: local_name
    for mtf_name, local_name in zip(
        LEVEL_REGISTRY_MTF_FEATURE_NAMES,
        LEVEL_REGISTRY_M5_FEATURE_NAMES,
        strict=True,
    )
}


# ---------------------------------------------------------------------------
# Serializable incremental state (chunk-safe carry, the
# bounded feature-owner carry-state pattern). The
# caller owns chronological continuity of the chunks, exactly as with that
# owner's memory_state.
# ---------------------------------------------------------------------------
# V30 (2026-08-13): state schema 2 adds ``last_break_side`` (the break-side
# memory of the most recent break bar) for the signed bars-since-break field.
# V30 package 8A (2026-08-13): schema 3 — the top-level key set is unchanged,
# but two per-level semantics moved, so carried schema-2 state is no longer
# interchangeable and must be rejected rather than silently reinterpreted:
# (a) ``side_of_origin`` is now lifecycle state, not a birth constant (a held
# retest flips it and returns the level to ACTIVE), and (b) each open reaction
# window carries a fifth element, the side frozen at its own t0.
LEVEL_REGISTRY_STATE_VERSION = (
    "level_registry_v1_state_9_immutable_recurrence_full_mtf"
)
LEVEL_REGISTRY_STATE_KEYS = (
    "state_version",
    "tf",
    "recurrence_threshold_atr",
    "max_evidence_age_bars",
    "bars_processed",
    "atr_started",
    "first_level_admitted",
    "next_level_id",
    "last_break_bar",
    "last_break_side",
    "tail_high",
    "tail_low",
    "levels",
)
LEVEL_REGISTRY_LEVEL_STATE_KEYS = (
    "level_id",
    "birth_side",
    "side_of_origin",
    "center_price",
    "member_pivot_bars",
    "member_pivot_prices",
    "touch_count",
    "birth_bar",
    "last_touch_bar",
    "last_eligibility_refresh_bar",
    "reaction_sum_atr",
    "reaction_max_atr",
    "reaction_last_atr",
    "completed_reaction_count",
    "status",
    "break_bar",
    "break_side",
    "retest_state",
    "prev_bar_intersects_center",
    "recurrence_confirmed",
    "open_reactions",
)
_SIDE_HIGH = "high_pivot"
_SIDE_LOW = "low_pivot"
_SIDES = (_SIDE_HIGH, _SIDE_LOW)
_STATUSES = ("active", "broken", "expired")
_RETEST_STATES = ("none", "pending", "held", "failed", "expired")


def _empty_registry_state(
    tf: str, recurrence_threshold_atr: float, max_evidence_age_bars: int
) -> dict[str, Any]:
    return {
        "state_version": LEVEL_REGISTRY_STATE_VERSION,
        "tf": tf,
        "recurrence_threshold_atr": float(recurrence_threshold_atr),
        "max_evidence_age_bars": int(max_evidence_age_bars),
        "bars_processed": 0,
        "atr_started": False,
        "first_level_admitted": False,
        "next_level_id": 0,
        "last_break_bar": -1,
        "last_break_side": 0,
        "tail_high": [],
        "tail_low": [],
        "levels": [],
    }


def _validate_tf(tf: str) -> str:
    if tf not in LEVEL_REGISTRY_TIMEFRAMES:
        raise RuntimeError(
            f"[LEVEL_REGISTRY_TF_INVALID] {tf!r} not in "
            f"{LEVEL_REGISTRY_TIMEFRAMES}"
        )
    return tf


def _validate_tol(recurrence_threshold_atr: float) -> float:
    try:
        tol = float(recurrence_threshold_atr)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "[LEVEL_REGISTRY_RECURRENCE_THRESHOLD_INVALID] not a number"
        ) from exc
    if not math.isfinite(tol) or tol <= 0.0:
        raise RuntimeError(
            "[LEVEL_REGISTRY_RECURRENCE_THRESHOLD_INVALID] "
            f"recurrence_threshold_atr must be finite and > 0, got {tol!r}"
        )
    return tol


def _validate_registry_state(
    state: Mapping[str, Any],
    *,
    tf: str,
    recurrence_threshold_atr: float,
    max_evidence_age_bars: int,
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
    # The recurrence threshold is frozen bundle state (rule 18): carried state must bind
    # the exact same constant, bit for bit.
    if float(state["recurrence_threshold_atr"]) != float(recurrence_threshold_atr):
        raise RuntimeError(
            "[LEVEL_REGISTRY_STATE_CONTRACT_MISMATCH] state recurrence_threshold_atr="
            f"{state['recurrence_threshold_atr']!r} != {recurrence_threshold_atr!r}"
        )
    if state["max_evidence_age_bars"] != max_evidence_age_bars:
        raise RuntimeError(
            "[LEVEL_REGISTRY_STATE_CONTRACT_MISMATCH] state max_evidence_age_bars="
            f"{state['max_evidence_age_bars']!r} != {max_evidence_age_bars!r}"
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
    if (
        not isinstance(out["last_break_side"], int)
        or isinstance(out["last_break_side"], bool)
        or out["last_break_side"] not in (-1, 0, 1)
    ):
        raise RuntimeError("[LEVEL_REGISTRY_STATE_INVALID] last_break_side")
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
            "touch_count",
            "birth_bar",
            "last_touch_bar",
            "last_eligibility_refresh_bar",
            "completed_reaction_count",
            "recurrence_confirmed",
            "break_bar",
            "break_side",
        ):
            if not isinstance(lv[key], int) or isinstance(lv[key], bool):
                raise RuntimeError(f"[LEVEL_REGISTRY_STATE_INVALID] level {key}")
        if lv["birth_side"] not in _SIDES:
            raise RuntimeError("[LEVEL_REGISTRY_STATE_INVALID] birth_side")
        if not isinstance(lv["prev_bar_intersects_center"], bool):
            raise RuntimeError("[LEVEL_REGISTRY_STATE_INVALID] level center memory")
        if not isinstance(lv["member_pivot_bars"], list) or not isinstance(
            lv["member_pivot_prices"], list
        ):
            raise RuntimeError("[LEVEL_REGISTRY_STATE_INVALID] level members")
        if not isinstance(lv["open_reactions"], list):
            raise RuntimeError("[LEVEL_REGISTRY_STATE_INVALID] level open_reactions")
        for window in lv["open_reactions"]:
            if (
                not isinstance(window, list)
                or len(window) != 5
                or not isinstance(window[0], int)
                or not isinstance(window[1], float)
                or not math.isfinite(window[1])
                or window[1] <= 0.0
                or not isinstance(window[2], float)
                or not math.isfinite(window[2])
                or not (window[3] is None or (isinstance(window[3], float) and math.isfinite(window[3])))
                or window[4] not in _SIDES
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
# TRAIN fit of the nearest-same-side recurrence threshold (required TRAIN-fitted
# statistic; rule 18 pattern: fitted once on the complete declared TRAIN
# window, frozen as bundle state, never refitted downstream).
#
# Fit-lane labels: the engine TFs plus "m1" — the Exit
# local M1 lane runs the same M5-block engine on the native M1 clock (the
# shared local layer in entry_model_native_feature_layers_v1), so its
# parameters must be fitted on that clock's own confirmed-pivot population.
# ---------------------------------------------------------------------------
LEVEL_REGISTRY_FIT_LANE_TFS = LEVEL_REGISTRY_TIMEFRAMES


def fit_level_registry_hyperparameters_v1(
    df: pd.DataFrame,
    *,
    tf: str,
    inner_fit_end_exclusive: int,
    source_provenance: Mapping[str, Any],
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    atr_col: str = "atr",
) -> dict[str, Any]:
    """Fit recurrence threshold/lifetime on the canonical runtime event tape.

    Identity/lifecycle is materialized once, independently of both selected
    parameters. Future outcomes are used only inside this TRAIN fit; serving
    consumes the frozen selection as derived evidence without altering tape
    control flow.
    """

    return _fit_level_registry_canonical_tape_v1(
        df,
        tf=tf,
        inner_fit_end_exclusive=inner_fit_end_exclusive,
        source_provenance=source_provenance,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        atr_col=atr_col,
    )

def _new_level(
    level_id: int, side: str, price: float, birth_bar: int, pivot_bar: int, atr_now: float
) -> dict[str, Any]:
    return {
        "level_id": level_id,
        "birth_side": side,
        "side_of_origin": side,
        "center_price": price,
        "member_pivot_bars": [pivot_bar],
        "member_pivot_prices": [price],
        "touch_count": 1,
        "birth_bar": birth_bar,
        "last_touch_bar": pivot_bar,
        "last_eligibility_refresh_bar": birth_bar,
        "reaction_sum_atr": 0.0,
        "reaction_max_atr": 0.0,
        "reaction_last_atr": 0.0,
        "completed_reaction_count": 0,
        "status": "active",
        "break_bar": -1,
        "break_side": 0,
        "retest_state": "none",
        "prev_bar_intersects_center": False,
        "recurrence_confirmed": 0,
        # Reaction window = [t0, atr0, center0, extreme, side_at_t0].  V30
        # package 8A froze the side alongside atr0/center0: the polarity flip
        # can now change a level's side WHILE a window is open, and the
        # measurement must stay "against the level as tested" (rule 2g).
        "open_reactions": [[birth_bar, atr_now, price, None, side]],
    }


def _slot_fields(
    level: dict[str, Any] | None,
    dist: float,
    t: int,
    *,
    dist2: float,
    max_evidence_age_bars: int,
) -> tuple[float, ...]:
    """Return one side's slot block in declared order.

    ``dist2`` is the distance to the SECOND-nearest ACTIVE level on the same
    side (V30 package 8A).  It carries the same absent-slot encoding as the
    nearest slot. Its explicit presence mask distinguishes "one lonely level"
    from "a shelf of four". It is a distance only: the second
    level's attributes are deliberately NOT emitted (rule 22 — the density
    question is answered by the distance, and eight more columns per side
    would be mechanism without a removed failure mode).
    """

    dist2_value = float(dist2) if math.isfinite(dist2) else 0.0
    if level is None:
        # Explicit presence masks disambiguate raw zero-valued attributes.
        return (
            0.0,
            dist2_value,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    count = level["completed_reaction_count"]
    mean_reaction = level["reaction_sum_atr"] / count if count > 0 else 0.0
    evidence_live = (
        t - level["last_eligibility_refresh_bar"] <= max_evidence_age_bars
    )
    return (
        float(dist),
        dist2_value,
        float(level["touch_count"]) if evidence_live else 0.0,
        float(count) if evidence_live else 0.0,
        # Confirmed same-side recurrence pivots inside the TRAIN-frozen
        # distance. This is separate from exact-center touches.
        float(level["recurrence_confirmed"]) if evidence_live else 0.0,
        float(t - level["birth_bar"]),
        float(t - level["last_touch_bar"]),
        mean_reaction if count > 0 and evidence_live else 0.0,
        float(level["reaction_max_atr"]) if count > 0 and evidence_live else 0.0,
        float(level["reaction_last_atr"]) if count > 0 and evidence_live else 0.0,
    )


def _update_reaction_extreme(
    window: list[Any],
    *,
    t: int,
    high: float,
    low: float,
) -> None:
    t0, _atr0, _center0, extreme, side0 = window
    if t <= t0:
        return
    if side0 == _SIDE_LOW:
        window[3] = high if extreme is None else max(float(extreme), high)
    else:
        window[3] = low if extreme is None else min(float(extreme), low)


def _finalize_reaction_lifecycle(
    level: dict[str, Any],
    *,
    t: int,
    high: float,
    low: float,
    reason: str,
    runtime_lifecycle_log: list[tuple] | None,
) -> None:
    """Finalize open reactions on an observed lifecycle boundary.

    A reaction begins at a genuine touch and ends at the next distinct
    touch, break, role flip or expiry.  There is no bar-count window.  A
    same-bar duplicate admission is not a distinct duration and therefore
    cannot manufacture a zero-length completed reaction.
    """

    keep: list[list[Any]] = []
    for window in level["open_reactions"]:
        t0, atr0, center0, _extreme, side0 = window
        if t <= t0:
            keep.append(window)
            continue
        _update_reaction_extreme(window, t=t, high=high, low=low)
        extreme = window[3]
        if extreme is None:
            raise RuntimeError("[LEVEL_REGISTRY_REACTION_EXTREME_MISSING]")
        if side0 == _SIDE_LOW:
            value = (float(extreme) - center0) / atr0
        else:
            value = (center0 - float(extreme)) / atr0
        level["reaction_sum_atr"] += value
        level["completed_reaction_count"] += 1
        level["reaction_last_atr"] = value
        if level["completed_reaction_count"] == 1:
            level["reaction_max_atr"] = value
        else:
            level["reaction_max_atr"] = max(level["reaction_max_atr"], value)
        if runtime_lifecycle_log is not None:
            runtime_lifecycle_log.append(
                (
                    "reaction_complete",
                    t,
                    int(level["level_id"]),
                    int(t - t0),
                    str(reason),
                    float(value),
                )
            )
    level["open_reactions"] = keep


def _open_reaction_lifecycle(
    level: dict[str, Any],
    *,
    t: int,
    atr_now: float,
    runtime_lifecycle_log: list[tuple] | None,
) -> None:
    if any(int(window[0]) == t for window in level["open_reactions"]):
        return
    level["open_reactions"].append(
        [
            t,
            atr_now,
            level["center_price"],
            None,
            level["side_of_origin"],
        ]
    )
    if runtime_lifecycle_log is not None:
        runtime_lifecycle_log.append(
            (
                "reaction_open",
                int(t),
                int(level["level_id"]),
                str(level["side_of_origin"]),
                float(level["center_price"]),
                float(atr_now),
            )
        )


def _run_level_registry(
    df: pd.DataFrame,
    *,
    tf: str,
    recurrence_threshold_atr: float,
    max_evidence_age_bars: int,
    high_col: str,
    low_col: str,
    close_col: str,
    atr_col: str,
    registry_state: Mapping[str, Any] | None,
    runtime_admission_distance_log: list[float] | None = None,
    runtime_threshold_breakpoint_log: list[float] | None = None,
    runtime_lifecycle_log: list[tuple] | None = None,
    runtime_expiry_age_log: list[int] | None = None,
    _allow_fit_boundary_values: bool = False,
    _apply_selected_eligibility: bool = True,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    _validate_tf(tf)
    if _allow_fit_boundary_values:
        try:
            tol = float(recurrence_threshold_atr)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("[LEVEL_REGISTRY_FIT_BOUNDARY_TOL_INVALID]") from exc
        if math.isnan(tol) or tol < 0.0:
            raise RuntimeError("[LEVEL_REGISTRY_FIT_BOUNDARY_TOL_INVALID]")
    else:
        tol = _validate_tol(recurrence_threshold_atr)
    if (
        isinstance(max_evidence_age_bars, bool)
        or not isinstance(max_evidence_age_bars, int)
        or max_evidence_age_bars < (0 if _allow_fit_boundary_values else 1)
    ):
        raise RuntimeError("[LEVEL_REGISTRY_MAX_EVIDENCE_AGE_INVALID]")
    if runtime_admission_distance_log is not None and not isinstance(
        runtime_admission_distance_log, list
    ):
        raise RuntimeError("[LEVEL_REGISTRY_RUNTIME_AUDIT_LOG_INVALID] admission")
    if runtime_threshold_breakpoint_log is not None and not isinstance(
        runtime_threshold_breakpoint_log, list
    ):
        raise RuntimeError("[LEVEL_REGISTRY_RUNTIME_AUDIT_LOG_INVALID] threshold")
    if runtime_lifecycle_log is not None and not isinstance(
        runtime_lifecycle_log, list
    ):
        raise RuntimeError("[LEVEL_REGISTRY_RUNTIME_AUDIT_LOG_INVALID] lifecycle")
    if runtime_expiry_age_log is not None and not isinstance(
        runtime_expiry_age_log, list
    ):
        raise RuntimeError("[LEVEL_REGISTRY_RUNTIME_AUDIT_LOG_INVALID] expiry_age")
    if not isinstance(_apply_selected_eligibility, bool):
        raise RuntimeError("[LEVEL_REGISTRY_ELIGIBILITY_POLICY_INVALID]")
    if registry_state is None:
        state = _empty_registry_state(tf, tol, max_evidence_age_bars)
    else:
        state = _validate_registry_state(
            registry_state,
            tf=tf,
            recurrence_threshold_atr=tol,
            max_evidence_age_bars=max_evidence_age_bars,
        )
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
    age_cap = int(max_evidence_age_bars)

    buf_high: list[float] = list(state["tail_high"])
    buf_low: list[float] = list(state["tail_low"])
    levels: list[dict[str, Any]] = state["levels"]
    active_by_id: dict[int, dict[str, Any]] = {
        int(level["level_id"]): level
        for level in levels
        if level["status"] == "active"
    }
    pending_retest_by_id: dict[int, dict[str, Any]] = {
        int(level["level_id"]): level
        for level in levels
        if level["status"] == "broken" and level["retest_state"] == "pending"
    }
    immutable_anchor_index: dict[str, list[tuple[float, int]]] = {
        _SIDE_HIGH: [],
        _SIDE_LOW: [],
    }
    for level in levels:
        insort(
            immutable_anchor_index[str(level["birth_side"])],
            (float(level["center_price"]), int(level["level_id"])),
        )
    next_level_id = state["next_level_id"]
    last_break_bar = state["last_break_bar"]
    last_break_side = state["last_break_side"]
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

        # The canonical fit tape disables this selected-policy step. Serving
        # expires only event/slot eligibility after the TRAIN-learned lifetime;
        # the immutable identity remains in ``levels`` forever.
        if _apply_selected_eligibility:
            for level_id, level in list(active_by_id.items()):
                if t - int(level["last_eligibility_refresh_bar"]) <= age_cap:
                    continue
                _finalize_reaction_lifecycle(
                    level,
                    t=t,
                    high=h,
                    low=l,
                    reason="learned_eligibility_expiry",
                    runtime_lifecycle_log=runtime_lifecycle_log,
                )
                level["status"] = "expired"
                level["retest_state"] = "expired"
                active_by_id.pop(level_id)
                if runtime_lifecycle_log is not None:
                    runtime_lifecycle_log.append(
                        (
                            "eligibility_expiry",
                            t,
                            level_id,
                            str(level["side_of_origin"]),
                            int(level["last_eligibility_refresh_bar"]),
                        )
                    )
            for level_id, level in list(pending_retest_by_id.items()):
                if t - int(level["last_eligibility_refresh_bar"]) <= age_cap:
                    continue
                level["status"] = "expired"
                level["retest_state"] = "expired"
                pending_retest_by_id.pop(level_id)
                if runtime_lifecycle_log is not None:
                    runtime_lifecycle_log.append(
                        (
                            "eligibility_expiry",
                            t,
                            level_id,
                            str(level["side_of_origin"]),
                            int(level["last_eligibility_refresh_bar"]),
                        )
                    )
        # Canonical identity/lifecycle geometry is threshold-independent.
        # The learned threshold is used only by the derived recurrence counter;
        # it never changes level creation, center updates, break/flip/touch or
        # retention control flow.
        # (0) optional evidence-age audit. Learned lifetime gates emitted
        # evidence only; canonical identities are never deleted by it.
        if active_by_id:
            if runtime_expiry_age_log is not None:
                runtime_expiry_age_log.extend(
                    int(t - lv["last_eligibility_refresh_bar"])
                    for lv in active_by_id.values()
                )

        # (1) immutable pivot identities — high before low on a shared bar.
        admitted_ids: set[int] = set()
        if pivot_high or pivot_low:
            j = t - n
            for is_pivot, side, price in (
                (pivot_high, _SIDE_HIGH, pivot_high_price),
                (pivot_low, _SIDE_LOW, pivot_low_price),
            ):
                if not is_pivot:
                    continue
                # Recurrence owns an immutable per-birth-side anchor index,
                # not the selected active/pending event indexes.  Learned
                # expiry can therefore make an identity dormant for slot and
                # event processing without changing the fit/apply recurrence
                # population.  Binary search limits work to the two adjacent
                # price anchors rather than rescanning retained history.
                anchor_index = immutable_anchor_index[side]
                position = bisect_left(anchor_index, (float(price), -1))
                nearest: tuple[int, float] | None = None
                for anchor_price, anchor_id in anchor_index[
                    max(0, position - 1) : min(len(anchor_index), position + 1)
                ]:
                    distance = abs(float(price) - anchor_price) / a
                    candidate = (int(anchor_id), float(distance))
                    if (
                        nearest is None
                        or candidate[1] < nearest[1]
                        or (
                            candidate[1] == nearest[1]
                            and candidate[0] < nearest[0]
                        )
                    ):
                        nearest = candidate
                comparisons = [] if nearest is None else [nearest]
                for _prior_id, distance in comparisons:
                    if runtime_admission_distance_log is not None:
                        runtime_admission_distance_log.append(float(distance))
                    if runtime_threshold_breakpoint_log is not None:
                        runtime_threshold_breakpoint_log.append(float(distance))

                level = _new_level(next_level_id, side, price, t, j, a)
                next_level_id += 1
                level["recurrence_confirmed"] = sum(
                    distance <= tol for _prior_id, distance in comparisons
                )
                levels.append(level)
                insort(
                    immutable_anchor_index[side],
                    (float(price), int(level["level_id"])),
                )
                active_by_id[int(level["level_id"])] = level
                admitted_ids.add(level["level_id"])
                if runtime_lifecycle_log is not None:
                    runtime_lifecycle_log.append(
                        (
                            "create",
                            t,
                            j,
                            side,
                            int(level["level_id"]),
                            float(price),
                        )
                    )
                    runtime_lifecycle_log.append(
                        (
                            "reaction_open",
                            int(t),
                            int(level["level_id"]),
                            str(side),
                            float(price),
                            float(a),
                        )
                    )
                    for prior_id, distance in comparisons:
                        runtime_lifecycle_log.append(
                            (
                                "recurrence",
                                int(t),
                                int(j),
                                str(side),
                                int(level["level_id"]),
                                float(distance),
                                int(prior_id),
                            )
                        )
                first_level_admitted = True

        # (2) break checks — side-of-origin anchored crossings, break-once.
        break_up_fired = False
        break_down_fired = False
        broken_touch_count = 0
        for lv in list(active_by_id.values()):
            if lv["side_of_origin"] == _SIDE_HIGH:
                if c > lv["center_price"]:
                    _finalize_reaction_lifecycle(
                        lv,
                        t=t,
                        high=h,
                        low=l,
                        reason="break",
                        runtime_lifecycle_log=runtime_lifecycle_log,
                    )
                    lv["status"] = "broken"
                    lv["break_bar"] = t
                    lv["break_side"] = 1
                    lv["retest_state"] = "pending"
                    active_by_id.pop(int(lv["level_id"]), None)
                    pending_retest_by_id[int(lv["level_id"])] = lv
                    break_up_fired = True
                    broken_touch_count = max(broken_touch_count, lv["touch_count"])
                    last_break_bar = t
                    if runtime_lifecycle_log is not None:
                        runtime_lifecycle_log.append(
                            ("break", t, int(lv["level_id"]), 1)
                        )
            else:
                if c < lv["center_price"]:
                    _finalize_reaction_lifecycle(
                        lv,
                        t=t,
                        high=h,
                        low=l,
                        reason="break",
                        runtime_lifecycle_log=runtime_lifecycle_log,
                    )
                    lv["status"] = "broken"
                    lv["break_bar"] = t
                    lv["break_side"] = -1
                    lv["retest_state"] = "pending"
                    active_by_id.pop(int(lv["level_id"]), None)
                    pending_retest_by_id[int(lv["level_id"])] = lv
                    break_down_fired = True
                    broken_touch_count = max(broken_touch_count, lv["touch_count"])
                    last_break_bar = t
                    if runtime_lifecycle_log is not None:
                        runtime_lifecycle_log.append(
                            ("break", t, int(lv["level_id"]), -1)
                        )
        if break_up_fired or break_down_fired:
            # V30 break-side memory for the signed bars-since-break field: a
            # same-bar up+down conflict nets to 0 (the sibling signed
            # aggregation convention of interpretation note 6).
            last_break_side = int(break_up_fired) - int(break_down_fired)

        # (3) retest checks — first exact-center intersection after the break bar.
        hold_sign_sum = 0
        fail_sign_sum = 0
        for lv in list(pending_retest_by_id.values()):
            if lv["break_bar"] == t:
                continue
            if h >= lv["center_price"] and l <= lv["center_price"]:
                _finalize_reaction_lifecycle(
                    lv,
                    t=t,
                    high=h,
                    low=l,
                    reason="retest_resolution",
                    runtime_lifecycle_log=runtime_lifecycle_log,
                )
                if (c - lv["center_price"]) * lv["break_side"] > 0:
                    lv["retest_state"] = "held"
                    hold_sign_sum += lv["break_side"]
                    # V30 package 8A — POLARITY FLIP ("old support is new
                    # resistance", fidelity audit §5).  A held retest is the
                    # proof that the level still governs price from the OTHER
                    # side: the level was broken up and price came back to it
                    # and closed above, or vice versa.  Before this repair the
                    # level stayed status="broken" forever and was excluded
                    # from the nearest-ACTIVE scan, so the construct existed
                    # only as a 1-bar impulse and never as an object.  It now
                    # returns to ACTIVE with its identity, anchors, member
                    # pivots, touch history and reaction memory intact and its
                    # side_of_origin flipped, so its future break check and
                    # slot participation both run on the new
                    # role.  No retest-window constant exists: first re-entry
                    # remains armed until the level's evidence lifetime ends,
                    # and the flip is a re-labelling of
                    # state that already exists.  The break check for THIS bar
                    # has already run (step 2 precedes step 3), so a flipped
                    # level cannot re-break on its own flip bar.
                    #
                    # The retest bar genuinely intersects the anchor, so it counts
                    # as a touch — but through the EXISTING step-4 owner, not
                    # a second touch path here. ``prev_bar_intersects_center`` is stale
                    # while a level is broken (step 6 skips broken levels), so
                    # it is cleared: step 4, which runs immediately after this
                    # step on the same bar, then sees the normal
                    # first-entry-per-excursion condition, increments
                    # touch_count, refreshes last_touch_bar (the expiry clock)
                    # and opens a reaction window measured against the level's
                    # NEW side. A level created on this bar is in
                    # ``admitted_ids`` and is correctly not counted twice.
                    lv["side_of_origin"] = (
                        _SIDE_LOW if lv["side_of_origin"] == _SIDE_HIGH else _SIDE_HIGH
                    )
                    lv["status"] = "active"
                    lv["prev_bar_intersects_center"] = False
                    pending_retest_by_id.pop(int(lv["level_id"]), None)
                    active_by_id[int(lv["level_id"])] = lv
                    if runtime_lifecycle_log is not None:
                        runtime_lifecycle_log.append(
                            (
                                "flip",
                                t,
                                int(lv["level_id"]),
                                int(lv["break_side"]),
                                lv["side_of_origin"],
                            )
                        )
                else:
                    # Includes the exact tie close == center (fail-closed).
                    lv["retest_state"] = "failed"
                    pending_retest_by_id.pop(int(lv["level_id"]), None)
                    fail_sign_sum += lv["break_side"]
                    if runtime_lifecycle_log is not None:
                        runtime_lifecycle_log.append(
                            ("retest_fail", t, int(lv["level_id"]), int(lv["break_side"]))
                        )

        # Failed retests remain immutable terminal identities in ``levels``.
        # They are absent from both active and pending indexes, so retention
        # does not make them eligible for later events or public slots.

        # (4) touch checks — first-entry-per-excursion; a same-bar admission
        # suppresses the touch check for that level (counts once).
        for lv in active_by_id.values():
            if lv["level_id"] in admitted_ids:
                continue
            intersects_center = h >= lv["center_price"] and l <= lv["center_price"]
            if intersects_center and not lv["prev_bar_intersects_center"]:
                _finalize_reaction_lifecycle(
                    lv,
                    t=t,
                    high=h,
                    low=l,
                    reason="exact_center_reentry",
                    runtime_lifecycle_log=runtime_lifecycle_log,
                )
                lv["touch_count"] += 1
                lv["last_touch_bar"] = t
                lv["last_eligibility_refresh_bar"] = t
                _open_reaction_lifecycle(
                    lv,
                    t=t,
                    atr_now=a,
                    runtime_lifecycle_log=runtime_lifecycle_log,
                )
                if runtime_lifecycle_log is not None:
                    runtime_lifecycle_log.append(
                        ("touch", t, int(lv["level_id"]), lv["side_of_origin"])
                    )

        # (5) reaction lifecycle update. Completion belongs exclusively to a
        # genuine next touch/break boundary; no bar window exists.
        for lv in active_by_id.values():
            if not lv["open_reactions"]:
                continue
            for window in lv["open_reactions"]:
                _update_reaction_extreme(window, t=t, high=h, low=l)

        # (6) excursion memory for the next bar's touch dedup.
        for lv in active_by_id.values():
            lv["prev_bar_intersects_center"] = (
                h >= lv["center_price"] and l <= lv["center_price"]
            )

        # (7) emission (post-update state; broken levels are never "nearest
        # ACTIVE").
        if not first_level_admitted:
            emit_nan_row()
            continue
        above: dict[str, Any] | None = None
        below: dict[str, Any] | None = None
        above_dist = below_dist = math.inf
        # V30 package 8A: the runner-up distance per side, tracked in the same
        # single pass with the same side rule and the same (distance,
        # level_id) ordering as the nearest slot.
        above_dist2 = below_dist2 = math.inf
        for lv in active_by_id.values():
            delta = lv["center_price"] - c
            if delta > 0.0:
                d = delta / a
                if d < above_dist or (d == above_dist and above is not None and lv["level_id"] < above["level_id"]):
                    above_dist2 = above_dist
                    above = lv
                    above_dist = d
                elif d < above_dist2:
                    above_dist2 = d
            else:
                d = -delta / a
                if d < below_dist or (d == below_dist and below is not None and lv["level_id"] < below["level_id"]):
                    below_dist2 = below_dist
                    below = lv
                    below_dist = d
                elif d < below_dist2:
                    below_dist2 = d
        above_vals = _slot_fields(
            above,
            above_dist,
            t,
            dist2=above_dist2,
            max_evidence_age_bars=age_cap,
        )
        below_vals = _slot_fields(
            below,
            below_dist,
            t,
            dist2=below_dist2,
            max_evidence_age_bars=age_cap,
        )
        record["level_above_dist_atr"].append(above_vals[0])
        record["level_above_present"].append(1.0 if above is not None else 0.0)
        record["level_above2_dist_atr"].append(above_vals[1])
        record["level_above2_present"].append(
            1.0 if math.isfinite(above_dist2) else 0.0
        )
        record["level_above_touch_count"].append(above_vals[2])
        record["level_above_completed_reaction_count"].append(above_vals[3])
        record["level_above_recurrence_confirmed"].append(above_vals[4])
        record["level_above_age_bars"].append(above_vals[5])
        record["level_above_bars_since_touch"].append(above_vals[6])
        record["level_above_mean_reaction_atr"].append(above_vals[7])
        record["level_above_max_reaction_atr"].append(above_vals[8])
        record["level_above_last_reaction_atr"].append(above_vals[9])
        record["level_below_dist_atr"].append(below_vals[0])
        record["level_below_present"].append(1.0 if below is not None else 0.0)
        record["level_below2_dist_atr"].append(below_vals[1])
        record["level_below2_present"].append(
            1.0 if math.isfinite(below_dist2) else 0.0
        )
        record["level_below_touch_count"].append(below_vals[2])
        record["level_below_completed_reaction_count"].append(below_vals[3])
        record["level_below_recurrence_confirmed"].append(below_vals[4])
        record["level_below_age_bars"].append(below_vals[5])
        record["level_below_bars_since_touch"].append(below_vals[6])
        record["level_below_mean_reaction_atr"].append(below_vals[7])
        record["level_below_max_reaction_atr"].append(below_vals[8])
        record["level_below_last_reaction_atr"].append(below_vals[9])
        record["level_break_up_event"].append(1.0 if break_up_fired else 0.0)
        record["level_break_down_event"].append(1.0 if break_down_fired else 0.0)
        record["level_broken_touch_count"].append(
            float(broken_touch_count) if (break_up_fired or break_down_fired) else 0.0
        )
        break_age = raw_event_age_from_last_observed_row(t, last_break_bar)
        record["level_bars_since_break"].append(break_age)
        # V30 signed break memory: sign = side of the most recent break bar
        # (+1 up / -1 down / 0 same-bar conflict); NaN before any observed
        # break. The event fields disambiguate the zero-age firing row.
        record["level_bars_since_break_signed"].append(
            float(last_break_side) * break_age
        )
        record["level_retest_hold_signed"].append(
            float((hold_sign_sum > 0) - (hold_sign_sum < 0))
        )
        record["level_retest_fail_signed"].append(
            float((fail_sign_sum > 0) - (fail_sign_sum < 0))
        )

    out_state = {
        "state_version": LEVEL_REGISTRY_STATE_VERSION,
        "tf": tf,
        "recurrence_threshold_atr": tol,
        "max_evidence_age_bars": age_cap,
        "bars_processed": base + nb,
        "atr_started": atr_started,
        "first_level_admitted": first_level_admitted,
        "next_level_id": next_level_id,
        "last_break_bar": last_break_bar,
        "last_break_side": last_break_side,
        "tail_high": buf_high[-(2 * n):] if len(buf_high) > 2 * n else list(buf_high),
        "tail_low": buf_low[-(2 * n):] if len(buf_low) > 2 * n else list(buf_low),
        "levels": levels,
    }
    return record, out_state


def _canonical_level_identity_state_sha256(state: Mapping[str, Any]) -> str:
    """Hash only immutable anchor identity, never selected eligibility state."""

    identities = [
        {
            "level_id": int(level["level_id"]),
            "birth_side": str(level["birth_side"]),
            "center_price": float(level["center_price"]),
            "birth_bar": int(level["birth_bar"]),
            "member_pivot_bars": list(level["member_pivot_bars"]),
            "member_pivot_prices": list(level["member_pivot_prices"]),
        }
        for level in state["levels"]
    ]
    identities.sort(key=lambda level: int(level["level_id"]))
    return canonical_json_sha256(
        {
            "tf": str(state["tf"]),
            "bars_processed": int(state["bars_processed"]),
            "next_level_id": int(state["next_level_id"]),
            "tail_high": list(state["tail_high"]),
            "tail_low": list(state["tail_low"]),
            "identities": identities,
        }
    )


def _level_record_sha256(record: Mapping[str, list[float]]) -> str:
    digest = hashlib.sha256()
    for name in LEVEL_REGISTRY_M5_FEATURE_NAMES:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            np.ascontiguousarray(record[name], dtype=np.float64).tobytes(order="C")
        )
    return digest.hexdigest()


def _level_registry_outcome_stream_sha256(
    stream: RegistryOutcomeStreamV1,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        np.ascontiguousarray(stream.origin_row, dtype=np.int64).tobytes(order="C")
    )
    digest.update(
        np.ascontiguousarray(stream.distance_atr, dtype=np.float64).tobytes(
            order="C"
        )
    )
    digest.update(
        np.ascontiguousarray(stream.event_row, dtype=np.int64).tobytes(order="C")
    )
    digest.update("\n".join(stream.event_cause).encode("ascii"))
    return digest.hexdigest()


def _level_registry_outcome_stream_from_replay(
    *,
    lifecycle: list[tuple],
    state: Mapping[str, Any],
    last_row: int,
) -> tuple[RegistryOutcomeStreamV1, list[dict[str, Any]]]:
    """Build the fit population from the exact runtime lifecycle bytes."""

    episodes = _runtime_replay_episodes(
        lifecycle=lifecycle,
        state=state,
        last_row=last_row,
    )
    episode_by_identity = {
        (int(item["level_id"]), int(item["origin_row"])): item
        for item in episodes
    }
    observations: list[tuple[int, float, int, str, int]] = []
    identity = 0
    for event in lifecycle:
        if str(event[0]) != "recurrence":
            continue
        origin = int(event[1])
        level_id = int(event[4])
        distance = float(event[5])
        # A recurrence born on the final closed TRAIN row is genuine apply
        # evidence but has zero authoritative outcome exposure. It cannot
        # enter a competing-risk fit population and is deterministically
        # excluded on both canonical and selected replay extraction.
        if origin >= last_row:
            continue
        if not math.isfinite(distance) or distance <= 0.0:
            continue
        episode = episode_by_identity.get((level_id, origin))
        if episode is None:
            raise RuntimeError("[LEVEL_REGISTRY_RUNTIME_TAPE_EPISODE_MISSING]")
        cause = str(episode["event_cause"])
        event_row = int(episode["end_row"])
        if (
            cause == REGISTRY_OUTCOME_CENSORED
            and str(episode["resolution"]) == "outer_train_end"
        ):
            event_row = -1
        observations.append((origin, distance, event_row, cause, identity))
        identity += 1
    observations.sort(key=lambda item: (item[0], item[1], item[4]))
    return (
        RegistryOutcomeStreamV1(
            origin_row=np.asarray(
                [item[0] for item in observations], dtype=np.int64
            ),
            distance_atr=np.asarray(
                [item[1] for item in observations], dtype=np.float64
            ),
            event_row=np.asarray(
                [item[2] for item in observations], dtype=np.int64
            ),
            event_cause=tuple(item[3] for item in observations),
        ),
        episodes,
    )


def _collect_level_registry_canonical_event_tape_v1(
    df: pd.DataFrame,
    *,
    tf: str,
    high_col: str,
    low_col: str,
    close_col: str,
    atr_col: str,
) -> tuple[RegistryOutcomeStreamV1, dict[str, Any]]:
    """Materialize the threshold/lifetime-independent serving event tape."""

    lifecycle: list[tuple] = []
    _record, state = _run_level_registry(
        df,
        tf=tf,
        recurrence_threshold_atr=0.0,
        max_evidence_age_bars=0,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        atr_col=atr_col,
        registry_state=None,
        runtime_lifecycle_log=lifecycle,
        _allow_fit_boundary_values=True,
        _apply_selected_eligibility=False,
    )
    stream, episodes = _level_registry_outcome_stream_from_replay(
        lifecycle=lifecycle,
        state=state,
        last_row=len(df) - 1,
    )
    tape_evidence = {
        "canonical_lifecycle_sha256": canonical_json_sha256(lifecycle),
        "canonical_episode_sha256": canonical_json_sha256(episodes),
        "canonical_identity_state_sha256": _canonical_level_identity_state_sha256(
            state
        ),
        "canonical_recurrence_observation_count": int(len(stream.origin_row)),
        "canonical_outcome_stream_sha256": (
            _level_registry_outcome_stream_sha256(stream)
        ),
    }
    tape_evidence["canonical_event_tape_sha256"] = canonical_json_sha256(
        tape_evidence
    )
    return stream, tape_evidence


def _fit_level_registry_canonical_tape_v1(
    df: pd.DataFrame,
    *,
    tf: str,
    inner_fit_end_exclusive: int,
    source_provenance: Mapping[str, Any],
    high_col: str,
    low_col: str,
    close_col: str,
    atr_col: str,
) -> dict[str, Any]:
    if tf not in LEVEL_REGISTRY_FIT_LANE_TFS:
        raise RuntimeError(f"[LEVEL_REGISTRY_TF_INVALID] {tf!r}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise RuntimeError("[LEVEL_REGISTRY_FIT_DATETIME_INDEX_REQUIRED]")
    high, low, close, atr = _validate_source(
        df,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        atr_col=atr_col,
        atr_started=False,
    )
    source = require_registry_fit_source_v1(
        source_provenance,
        clock=tf.upper(),
    )
    canonical_stream, tape = _collect_level_registry_canonical_event_tape_v1(
        df,
        tf=tf,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        atr_col=atr_col,
    )
    atr_available = np.isfinite(atr).astype(np.float64)
    base_population_configuration = {
        "owner": "level_immutable_anchor_recurrence_decomposition_v2",
        "runtime_owner": "gx1.features.level_registry_v1._run_level_registry",
        "identity_control": (
            "immutable_per_confirmed_pivot_anchor_exact_center_break_touch_"
            "and_parameter_independent_identity_retention"
        ),
        "threshold_selection": (
            "recurrence_confirmed_on_nearest_same_side_distance"
        ),
        "lifetime_selection": (
            "learned_expiry_of_event_and_slot_eligibility_without_identity_deletion"
        ),
        "swing_lookback": int(SWING_LOOKBACK),
        **tape,
    }

    def fit_with_population(
        outcome_stream: RegistryOutcomeStreamV1,
        configuration: Mapping[str, Any],
    ) -> dict[str, Any]:
        return fit_registry_competing_risk_threshold_v1(
            outcome_stream,
            registry_kind="horizontal_level",
            clock=tf.upper(),
            n_rows=len(df),
            inner_fit_end_exclusive=inner_fit_end_exclusive,
            index_ns=df.index.asi8,
            frame_columns=(
                high,
                low,
                close,
                np.where(np.isfinite(atr), atr, 0.0),
                atr_available,
            ),
            source_provenance=source,
            population_configuration=configuration,
        )

    seed = fit_with_population(
        canonical_stream,
        {
            **base_population_configuration,
            "fit_phase": "parameter_independent_identity_tape_seed",
        },
    )
    selected_threshold = float(seed["selected_threshold_atr"])
    selected_expiry = int(seed["learned_expiry_bars"])
    selected_lifecycle: list[tuple] = []
    selected_record, selected_state = _run_level_registry(
        df,
        tf=tf,
        recurrence_threshold_atr=selected_threshold,
        max_evidence_age_bars=selected_expiry,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        atr_col=atr_col,
        registry_state=None,
        runtime_lifecycle_log=selected_lifecycle,
    )
    identity_sha = _canonical_level_identity_state_sha256(selected_state)
    if identity_sha != tape["canonical_identity_state_sha256"]:
        raise RuntimeError("[LEVEL_REGISTRY_CANONICAL_SELECTION_CHANGED_IDENTITY]")
    selected_stream, selected_episodes = _level_registry_outcome_stream_from_replay(
        lifecycle=selected_lifecycle,
        state=selected_state,
        last_row=len(df) - 1,
    )

    def recurrence_population_sha(stream: RegistryOutcomeStreamV1) -> str:
        digest = hashlib.sha256()
        digest.update(
            np.ascontiguousarray(stream.origin_row, dtype=np.int64).tobytes(
                order="C"
            )
        )
        digest.update(
            np.ascontiguousarray(stream.distance_atr, dtype=np.float64).tobytes(
                order="C"
            )
        )
        return digest.hexdigest()

    fit_recurrence_sha = recurrence_population_sha(canonical_stream)
    serve_recurrence_sha = recurrence_population_sha(selected_stream)
    if (
        fit_recurrence_sha != serve_recurrence_sha
        or not np.array_equal(
            canonical_stream.origin_row,
            selected_stream.origin_row,
        )
        or not np.array_equal(
            canonical_stream.distance_atr,
            selected_stream.distance_atr,
        )
    ):
        raise RuntimeError("[LEVEL_REGISTRY_FIT_SERVE_RECURRENCE_POPULATION_DRIFT]")

    # Independent selected-policy replay proves the apply bytes and recurrence
    # population are deterministic. Outcome labels remain on the canonical
    # exact-cross TRAIN tape: they are fit-only future evidence and never
    # inputs to apply. This decomposition avoids circular expiry->population
    # refits while keeping selected expiry out of identity/recurrence control.
    verify_lifecycle: list[tuple] = []
    verify_record, verify_state = _run_level_registry(
        df,
        tf=tf,
        recurrence_threshold_atr=selected_threshold,
        max_evidence_age_bars=selected_expiry,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        atr_col=atr_col,
        registry_state=None,
        runtime_lifecycle_log=verify_lifecycle,
    )
    verify_stream, verify_episodes = _level_registry_outcome_stream_from_replay(
        lifecycle=verify_lifecycle,
        state=verify_state,
        last_row=len(df) - 1,
    )
    if (
        verify_lifecycle != selected_lifecycle
        or canonical_json_sha256(verify_episodes)
        != canonical_json_sha256(selected_episodes)
        or _level_record_sha256(verify_record)
        != _level_record_sha256(selected_record)
        or canonical_json_sha256(verify_state)
        != canonical_json_sha256(selected_state)
        or recurrence_population_sha(verify_stream) != serve_recurrence_sha
    ):
        raise RuntimeError("[LEVEL_REGISTRY_SELECTED_SERVE_REPLAY_DRIFT]")

    final_configuration = {
        **base_population_configuration,
        "fit_phase": "immutable_recurrence_canonical_outcome_decomposition",
        "population_decomposition_contract": (
            "serve_recurrence_uses_all_prior_immutable_same_birth_side_anchors_"
            "via_price_index;train_outcomes_use_parameter_independent_exact_"
            "cross_lifecycle;selected_expiry_gates_only_event_slot_eligibility"
        ),
        "selected_threshold_atr": selected_threshold,
        "selected_expiry_bars": selected_expiry,
        "selected_runtime_recurrence_observation_count": int(
            len(selected_stream.origin_row)
        ),
        "final_fit_recurrence_population_sha256": fit_recurrence_sha,
        "final_serve_recurrence_population_sha256": serve_recurrence_sha,
        "canonical_fit_outcome_population_sha256": (
            _level_registry_outcome_stream_sha256(canonical_stream)
        ),
        "selected_derived_outcome_population_sha256": (
            _level_registry_outcome_stream_sha256(selected_stream)
        ),
        "selected_derived_lifecycle_sha256": canonical_json_sha256(
            selected_lifecycle
        ),
        "selected_derived_episode_sha256": canonical_json_sha256(
            selected_episodes
        ),
        "selected_derived_state_sha256": canonical_json_sha256(selected_state),
        "selected_derived_emission_sha256": _level_record_sha256(selected_record),
        "selected_eligibility_expiry_count": sum(
            event[0] == "eligibility_expiry" for event in selected_lifecycle
        ),
    }
    payload = fit_with_population(canonical_stream, final_configuration)
    if (
        float(payload["selected_threshold_atr"]) != selected_threshold
        or int(payload["learned_expiry_bars"]) != selected_expiry
        or payload["outcome_stream_sha256"]
        != final_configuration["canonical_fit_outcome_population_sha256"]
        or final_configuration["final_fit_recurrence_population_sha256"]
        != final_configuration["final_serve_recurrence_population_sha256"]
    ):
        raise RuntimeError("[LEVEL_REGISTRY_FINAL_DECOMPOSITION_DRIFT]")
    return require_registry_hyperparameter_payload_v1(
        payload,
        registry_kind="horizontal_level",
        clock=tf.upper(),
    )


def _runtime_replay_episodes(
    *,
    lifecycle: list[tuple],
    state: Mapping[str, Any],
    last_row: int,
) -> list[dict[str, Any]]:
    open_episodes: set[tuple[int, int]] = set()
    episodes: list[dict[str, Any]] = []
    for event in lifecycle:
        kind = str(event[0])
        if kind == "reaction_open":
            key = (int(event[2]), int(event[1]))
            if key in open_episodes:
                raise RuntimeError("[LEVEL_REGISTRY_REPLAY_DUPLICATE_OPEN]")
            open_episodes.add(key)
        elif kind == "reaction_complete":
            end = int(event[1])
            level_id = int(event[2])
            duration = int(event[3])
            origin = end - duration
            key = (level_id, origin)
            if key not in open_episodes:
                raise RuntimeError("[LEVEL_REGISTRY_REPLAY_ORPHAN_COMPLETION]")
            open_episodes.remove(key)
            resolution = str(event[4])
            episodes.append(
                {
                    "origin_row": origin,
                    "end_row": end,
                    "event_cause": (
                        REGISTRY_OUTCOME_BREAK
                        if resolution == "break"
                        else (
                            REGISTRY_OUTCOME_CENSORED
                            if resolution == "learned_eligibility_expiry"
                            else REGISTRY_OUTCOME_REACTION
                        )
                    ),
                    "level_id": level_id,
                    "resolution": resolution,
                }
            )
        elif kind == "reaction_censored":
            censor_boundary = int(event[1])
            level_id = int(event[2])
            duration = int(event[3])
            end = censor_boundary - 1
            origin = end - duration
            key = (level_id, origin)
            if key not in open_episodes:
                raise RuntimeError("[LEVEL_REGISTRY_REPLAY_ORPHAN_CENSOR]")
            open_episodes.remove(key)
            if end > origin:
                episodes.append(
                    {
                        "origin_row": origin,
                        "end_row": end,
                        "event_cause": REGISTRY_OUTCOME_CENSORED,
                        "level_id": level_id,
                        "resolution": str(event[4]),
                    }
                )
    for level in state["levels"]:
        level_id = int(level["level_id"])
        for window in level["open_reactions"]:
            origin = int(window[0])
            key = (level_id, origin)
            if key not in open_episodes:
                raise RuntimeError("[LEVEL_REGISTRY_REPLAY_OPEN_STATE_MISMATCH]")
            open_episodes.remove(key)
            if last_row > origin:
                episodes.append(
                    {
                        "origin_row": origin,
                        "end_row": int(last_row),
                        "event_cause": REGISTRY_OUTCOME_CENSORED,
                        "level_id": level_id,
                        "resolution": "outer_train_end",
                    }
                )
    if open_episodes:
        raise RuntimeError("[LEVEL_REGISTRY_REPLAY_UNRESOLVED_OPEN]")
    episodes.sort(
        key=lambda item: (
            int(item["origin_row"]),
            int(item["level_id"]),
            int(item["end_row"]),
            str(item["event_cause"]),
        )
    )
    return episodes


def collect_level_registry_runtime_population_v1(
    df: pd.DataFrame,
    *,
    tf: str,
    recurrence_threshold_atr: float,
    max_evidence_age_bars: int,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    atr_col: str = "atr",
    strict_nonempty: bool = True,
) -> tuple[np.ndarray, list[tuple]]:
    """Replay the exact runtime registry and return its fit-parity evidence.

    The float64 population contains every ATR-normalized nearest same-side
    recurrence distance evaluated before a new immutable anchor is created, in
    exact chronological order. The lifecycle log is emitted by the same loop
    and exposes create/recurrence/touch/break/retest/flip events for
    source-proven tests. Both hooks are observation-only.

    ``tf`` is the exact closed-bar decision clock and is preserved in carried
    state; M1 never aliases to M5.
    """

    if not isinstance(strict_nonempty, bool):
        raise RuntimeError("[LEVEL_REGISTRY_RUNTIME_SUPPORT_POLICY_INVALID]")
    if tf not in LEVEL_REGISTRY_FIT_LANE_TFS:
        raise RuntimeError(
            f"[LEVEL_REGISTRY_TF_INVALID] {tf!r} not in "
            f"{LEVEL_REGISTRY_FIT_LANE_TFS}"
        )
    distances: list[float] = []
    lifecycle: list[tuple] = []
    _record, _state = _run_level_registry(
        df,
        tf=tf,
        recurrence_threshold_atr=recurrence_threshold_atr,
        max_evidence_age_bars=max_evidence_age_bars,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        atr_col=atr_col,
        registry_state=None,
        runtime_admission_distance_log=distances,
        runtime_lifecycle_log=lifecycle,
    )
    population = np.asarray(distances, dtype=np.float64)
    if population.ndim != 1 or not np.isfinite(population).all():
        raise RuntimeError("[LEVEL_REGISTRY_RUNTIME_POPULATION_INVALID]")
    if strict_nonempty and population.size == 0:
        raise RuntimeError(
            "[LEVEL_REGISTRY_RUNTIME_FIT_SUPPORT_EMPTY] no nearest same-side "
            "recurrence distance exists on the declared window"
        )
    return population, lifecycle


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
    for column in range(matrix.shape[1]):
        finite = np.isfinite(matrix[:, column])
        if finite.any() and not finite[int(np.argmax(finite)) :].all():
            raise RuntimeError("[LEVEL_REGISTRY_OUTPUT_AVAILABILITY_INVALID]")
    return matrix, list(declared_names)


def compute_level_registry_m5_block_v1(
    df: pd.DataFrame,
    *,
    recurrence_threshold_atr: float,
    max_evidence_age_bars: int,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    atr_col: str = "atr",
    registry_state: Mapping[str, Any] | None = None,
    return_registry_state: bool = False,
    runtime_admission_distance_log: list[float] | None = None,
    runtime_lifecycle_log: list[tuple] | None = None,
    decision_clock: str = "m5",
) -> (
    tuple[np.ndarray, list[str]]
    | tuple[np.ndarray, list[str], dict[str, Any]]
):
    """Local M5 lane: the declared ``level_`` block (design doc §1.2 + V30).

    Runs on the entry M5 clock (tf is bound to ``"m5"``; the expiry cap is the
    M5 evidence horizon). ``recurrence_threshold_atr`` is the TRAIN-fitted
    nearest-same-side recurrence threshold from
    :func:`fit_level_registry_hyperparameters_v1`. Rows are NaN until
    the first admitted level (single chronological warmup prefix, trimmed
    downstream); afterwards no mid-series NaN is possible.
    """
    if decision_clock not in ("m1", "m5"):
        raise RuntimeError("[LEVEL_REGISTRY_LOCAL_DECISION_CLOCK_INVALID]")
    record, state = _run_level_registry(
        df,
        tf=decision_clock,
        recurrence_threshold_atr=recurrence_threshold_atr,
        max_evidence_age_bars=max_evidence_age_bars,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        atr_col=atr_col,
        registry_state=registry_state,
        runtime_admission_distance_log=runtime_admission_distance_log,
        runtime_lifecycle_log=runtime_lifecycle_log,
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
    recurrence_threshold_atr: float,
    max_evidence_age_bars: int,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    atr_col: str = "atr",
    registry_state: Mapping[str, Any] | None = None,
    return_registry_state: bool = False,
    runtime_admission_distance_log: list[float] | None = None,
    runtime_lifecycle_log: list[tuple] | None = None,
) -> (
    tuple[np.ndarray, list[str]]
    | tuple[np.ndarray, list[str], dict[str, Any]]
):
    """Per-TF V4 lane: the 11-field ``mtf_level_`` block (design doc §1.3).

    The same registry engine runs independently on each timeframe's closed
    bars (``tf`` selects the clock identity;
    ``recurrence_threshold_atr`` is that TF's TRAIN-fitted frozen recurrence
    threshold). Field values are exact renames of the
    registry fields — immutable pivot anchors only; session anchors do not
    replicate per TF.
    """
    if tuple(_LEVEL_REGISTRY_MTF_SOURCE_MAP) != LEVEL_REGISTRY_MTF_FEATURE_NAMES:
        raise RuntimeError("[LEVEL_REGISTRY_FEATURE_NAME_DRIFT] mtf source map")
    record, state = _run_level_registry(
        df,
        tf=tf,
        recurrence_threshold_atr=recurrence_threshold_atr,
        max_evidence_age_bars=max_evidence_age_bars,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        atr_col=atr_col,
        registry_state=registry_state,
        runtime_admission_distance_log=runtime_admission_distance_log,
        runtime_lifecycle_log=runtime_lifecycle_log,
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
