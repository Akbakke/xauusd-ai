"""
Smart Money Concept (SMC) features — V1 implementation.

For each closed bar, compute causal SMC primitives that describe market structure,
liquidity events, and premium/discount position. Designed to feed the
canonical_v3 feature parquet and downstream model-native consumers.

Features (all per-bar, lookahead-safe):
  smc_swing_state        int8     0=HH+HL (clean up), 1=HH+LL (two-sided expansion), 2=LH+HL (contraction/inside), 3=LH+LL (clean down), 4=exact ties or warmup
  smc_bos_up             float32  1.0 only on the bar where close first crosses above the last confirmed swing high
  smc_bos_down           float32  1.0 only on the bar where close first crosses below the last confirmed swing low
  smc_choch              float32  1.0 only on an opposing BOS while a structural side is armed
  smc_sweep_up           float32  1.0 if high > last swing high but close <= it (false breakout / liquidity hunt)
  smc_sweep_down         float32  1.0 if low  < last swing low  but close >= it
  smc_sweep_event_age_bars float32 raw bars since the most recent one-shot sweep event; NaN before it
  smc_pivot_envelope_position float32 raw close position in the causal 4-pivot envelope

Lookahead safety: a swing pivot at bar j is only considered "confirmed" once
j + SWING_LOOKBACK bars have elapsed. So features at bar i only use swings
confirmed up to bar (i - SWING_LOOKBACK), no future leakage.

2026-08-13 (V30 package 8A): the M5 owner gained optional owner-parity
emissions behind ``include_v30_additions`` (``SMC_V30_ADDITION_NAMES_V1``).
The MTF owner carries the matching signed BOS displacement, two one-shot sweep
events. On 2026-08-14 the model-native M5/M1 layer enabled these raw roots.

2026-08-14: both local and MTF owners use level-identity BOS/sweep events and
define CHOCH as an opposing BOS, not a pivot-pattern label transition.

2026-08-09 backport: BOS became an event instead of a persistent state, the
premium/discount range is the 4-pivot envelope instead of the last-high/
last-low pair, and the swing-state predicates for states 1/2 (previously
self-contradictory for distinct prices) form a true partition of the generic
two-sided comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from gx1.features.event_age_v1 import raw_event_age_from_last_observed_row


SWING_LOOKBACK = 3  # bars look-around for swing pivot detection (3 → 7-bar window centered)
SMC_CAUSAL_REPLAY_SCHEMA_VERSION = "smc_causal_replay_v4"
SMC_PRIMITIVE_CONTRACT_SCHEMA_VERSION = "smc_raw_primitives_v2"


SMCLevelIdentity = tuple[str, int]


@dataclass(frozen=True)
class SMCCausalReplayStateV2:
    """Exact carry for the closed-bar SMC structure replay.

    A level identity is ``(polarity, absolute_pivot_row)``.  Price is evidence
    attached to that identity, never the identity itself: two equal-price
    pivots are two independently consumable structures.
    """

    schema_version: str
    swing_lookback: int
    bars_seen: int
    high_tail: tuple[float, ...]
    low_tail: tuple[float, ...]
    last_high_idx: int
    last_high_price: float
    prev_high_idx: int
    prev_high_price: float
    last_low_idx: int
    last_low_price: float
    prev_low_idx: int
    prev_low_price: float
    structure_regime: int
    consumed_bos_high: SMCLevelIdentity | None
    consumed_bos_low: SMCLevelIdentity | None
    consumed_sweep_high: SMCLevelIdentity | None
    consumed_sweep_low: SMCLevelIdentity | None
    last_sweep_event_idx: int


def _empty_smc_causal_replay_state(
    swing_lookback: int,
) -> SMCCausalReplayStateV2:
    return SMCCausalReplayStateV2(
        schema_version=SMC_CAUSAL_REPLAY_SCHEMA_VERSION,
        swing_lookback=swing_lookback,
        bars_seen=0,
        high_tail=(),
        low_tail=(),
        last_high_idx=-1,
        last_high_price=float("nan"),
        prev_high_idx=-1,
        prev_high_price=float("nan"),
        last_low_idx=-1,
        last_low_price=float("nan"),
        prev_low_idx=-1,
        prev_low_price=float("nan"),
        structure_regime=0,
        consumed_bos_high=None,
        consumed_bos_low=None,
        consumed_sweep_high=None,
        consumed_sweep_low=None,
        last_sweep_event_idx=-1,
    )


def _require_smc_causal_replay_state(
    state: SMCCausalReplayStateV2,
    *,
    swing_lookback: int,
) -> None:
    if not isinstance(state, SMCCausalReplayStateV2):
        raise RuntimeError("[SMC_REPLAY_STATE_TYPE_INVALID]")
    if (
        state.schema_version != SMC_CAUSAL_REPLAY_SCHEMA_VERSION
        or state.swing_lookback != swing_lookback
        or state.bars_seen < 0
        or len(state.high_tail) != len(state.low_tail)
        or len(state.high_tail) != min(2 * swing_lookback, state.bars_seen)
        or not np.isfinite(np.asarray(state.high_tail, dtype=np.float64)).all()
        or not np.isfinite(np.asarray(state.low_tail, dtype=np.float64)).all()
        or state.structure_regime not in (-1, 0, 1)
        or state.last_sweep_event_idx < -1
        or state.last_sweep_event_idx >= state.bars_seen
    ):
        raise RuntimeError("[SMC_REPLAY_STATE_CONTRACT_INVALID]")
    for last_idx, last_price, prev_idx, prev_price in (
        (
            state.last_high_idx,
            state.last_high_price,
            state.prev_high_idx,
            state.prev_high_price,
        ),
        (
            state.last_low_idx,
            state.last_low_price,
            state.prev_low_idx,
            state.prev_low_price,
        ),
    ):
        if (
            isinstance(last_idx, bool)
            or isinstance(prev_idx, bool)
            or not isinstance(last_idx, (int, np.integer))
            or not isinstance(prev_idx, (int, np.integer))
            or last_idx < -1
            or prev_idx < -1
            or last_idx >= state.bars_seen
            or prev_idx >= state.bars_seen
            or (last_idx < 0) != (not np.isfinite(last_price))
            or (prev_idx < 0) != (not np.isfinite(prev_price))
            or (prev_idx >= 0 and not prev_idx < last_idx)
        ):
            raise RuntimeError("[SMC_REPLAY_STATE_CONTRACT_INVALID]")
    for identity, polarity in (
        (state.consumed_bos_high, "high"),
        (state.consumed_bos_low, "low"),
        (state.consumed_sweep_high, "high"),
        (state.consumed_sweep_low, "low"),
    ):
        if identity is not None and (
            not isinstance(identity, tuple)
            or len(identity) != 2
            or identity[0] != polarity
            or isinstance(identity[1], bool)
            or not isinstance(identity[1], (int, np.integer))
            or int(identity[1]) < 0
            or int(identity[1]) >= state.bars_seen
        ):
            raise RuntimeError("[SMC_REPLAY_LEVEL_IDENTITY_INVALID]")


def replay_smc_causal_structure_v2(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    swing_lookback: int = SWING_LOOKBACK,
    state: SMCCausalReplayStateV2 | None = None,
) -> tuple[dict[str, np.ndarray], SMCCausalReplayStateV2]:
    """Replay confirmed structure, breaks and liquidity events once.

    This is the sole event owner used by both the native M1/M5 surface and
    every MTF surface.  Inputs are already-closed bars on their declared native
    clock; this function never resamples or fabricates a clock.  A pivot at
    absolute row ``j`` becomes visible only while processing row
    ``j + swing_lookback``.  Passing the returned state into the next chunk is
    exactly equivalent to one uninterrupted replay.

    Sweep conditions remain per-bar wick-through/close-back evidence.  Sweep
    *events* consume the exact confirmed level identity once; a false gap in
    the condition does not rearm it.  A close-through BOS invalidates that
    identity for future sweeps, and only a newly confirmed pivot identity can
    arm another sweep.
    """

    if (
        isinstance(swing_lookback, bool)
        or not isinstance(swing_lookback, int)
        or swing_lookback < 1
    ):
        raise RuntimeError(f"[SMC_REPLAY_SWING_LOOKBACK_INVALID] {swing_lookback!r}")
    high_values = np.asarray(high, dtype=np.float64)
    low_values = np.asarray(low, dtype=np.float64)
    close_values = np.asarray(close, dtype=np.float64)
    if (
        high_values.ndim != 1
        or low_values.ndim != 1
        or close_values.ndim != 1
        or not (high_values.shape == low_values.shape == close_values.shape)
    ):
        raise RuntimeError("[SMC_REPLAY_INPUT_SHAPE_INVALID]")
    if (
        not np.isfinite(high_values).all()
        or not np.isfinite(low_values).all()
        or not np.isfinite(close_values).all()
        or np.any(high_values < low_values)
        or np.any(high_values < close_values)
        or np.any(low_values > close_values)
    ):
        raise RuntimeError("[SMC_REPLAY_INPUT_GEOMETRY_INVALID]")

    current = state or _empty_smc_causal_replay_state(swing_lookback)
    _require_smc_causal_replay_state(current, swing_lookback=swing_lookback)
    n_rows = len(high_values)
    field_names = (
        "last_high_idx",
        "prev_high_idx",
        "last_low_idx",
        "prev_low_idx",
        "last_high_price",
        "prev_high_price",
        "last_low_price",
        "prev_low_price",
        "swing_state",
        "structure_bias",
        "structure_available",
        "bos_up",
        "bos_down",
        "choch_up",
        "choch_down",
        "sweep_up",
        "sweep_down",
        "sweep_up_event",
        "sweep_down_event",
        "sweep_event_age_bars",
    )
    integer_fields = {
        "last_high_idx",
        "prev_high_idx",
        "last_low_idx",
        "prev_low_idx",
        "swing_state",
    }
    values = {
        name: np.full(
            n_rows,
            -1 if name != "swing_state" else 4,
            dtype=np.int64,
        )
        if name in integer_fields
        else np.full(n_rows, np.nan, dtype=np.float64)
        for name in field_names
    }
    for name in (
        "structure_bias",
        "structure_available",
        "bos_up",
        "bos_down",
        "choch_up",
        "choch_down",
        "sweep_up",
        "sweep_down",
        "sweep_up_event",
        "sweep_down_event",
    ):
        values[name].fill(0.0)

    tail_high = np.asarray(current.high_tail, dtype=np.float64)
    tail_low = np.asarray(current.low_tail, dtype=np.float64)
    combined_high = np.concatenate((tail_high, high_values))
    combined_low = np.concatenate((tail_low, low_values))
    combined_start = current.bars_seen - len(tail_high)
    swing_high_mask, swing_low_mask = _detect_swing_pivots(
        combined_high,
        combined_low,
        swing_lookback,
    )

    last_high_idx = current.last_high_idx
    last_high_price = current.last_high_price
    prev_high_idx = current.prev_high_idx
    prev_high_price = current.prev_high_price
    last_low_idx = current.last_low_idx
    last_low_price = current.last_low_price
    prev_low_idx = current.prev_low_idx
    prev_low_price = current.prev_low_price
    regime = current.structure_regime
    consumed_bos_high = current.consumed_bos_high
    consumed_bos_low = current.consumed_bos_low
    consumed_sweep_high = current.consumed_sweep_high
    consumed_sweep_low = current.consumed_sweep_low
    last_sweep_event_idx = current.last_sweep_event_idx

    for offset in range(n_rows):
        absolute_row = current.bars_seen + offset
        confirm_idx = absolute_row - swing_lookback
        confirm_local = confirm_idx - combined_start
        if 0 <= confirm_local < len(combined_high):
            if swing_high_mask[confirm_local]:
                prev_high_idx, prev_high_price = last_high_idx, last_high_price
                last_high_idx = confirm_idx
                last_high_price = float(combined_high[confirm_local])
            if swing_low_mask[confirm_local]:
                prev_low_idx, prev_low_price = last_low_idx, last_low_price
                last_low_idx = confirm_idx
                last_low_price = float(combined_low[confirm_local])

        values["last_high_idx"][offset] = last_high_idx
        values["prev_high_idx"][offset] = prev_high_idx
        values["last_low_idx"][offset] = last_low_idx
        values["prev_low_idx"][offset] = prev_low_idx
        values["last_high_price"][offset] = last_high_price
        values["prev_high_price"][offset] = prev_high_price
        values["last_low_price"][offset] = last_low_price
        values["prev_low_price"][offset] = prev_low_price

        higher_high = last_high_idx >= 0 and prev_high_idx >= 0 and last_high_price > prev_high_price
        lower_high = last_high_idx >= 0 and prev_high_idx >= 0 and last_high_price < prev_high_price
        higher_low = last_low_idx >= 0 and prev_low_idx >= 0 and last_low_price > prev_low_price
        lower_low = last_low_idx >= 0 and prev_low_idx >= 0 and last_low_price < prev_low_price
        if higher_high and higher_low:
            swing_state = 0
        elif higher_high and lower_low:
            swing_state = 1
        elif lower_high and higher_low:
            swing_state = 2
        elif lower_high and lower_low:
            swing_state = 3
        else:
            swing_state = 4
        values["swing_state"][offset] = swing_state

        high_sign = float(np.sign(last_high_price - prev_high_price)) if last_high_idx >= 0 and prev_high_idx >= 0 else 0.0
        low_sign = float(np.sign(last_low_price - prev_low_price)) if last_low_idx >= 0 and prev_low_idx >= 0 else 0.0
        structure_bias = (high_sign + low_sign) / 2.0
        structure_available = (
            last_high_idx >= 0
            and prev_high_idx >= 0
            and last_low_idx >= 0
            and prev_low_idx >= 0
            and max(last_high_price, prev_high_price, last_low_price, prev_low_price)
            > min(last_high_price, prev_high_price, last_low_price, prev_low_price)
        )
        values["structure_bias"][offset] = structure_bias
        values["structure_available"][offset] = float(structure_available)

        high_identity: SMCLevelIdentity | None = (
            ("high", last_high_idx) if last_high_idx >= 0 else None
        )
        low_identity: SMCLevelIdentity | None = (
            ("low", last_low_idx) if last_low_idx >= 0 else None
        )
        bos_up = bool(
            high_identity is not None
            and close_values[offset] > last_high_price
            and consumed_bos_high != high_identity
        )
        bos_down = bool(
            low_identity is not None
            and close_values[offset] < last_low_price
            and consumed_bos_low != low_identity
        )
        if bos_up:
            consumed_bos_high = high_identity
        if bos_down:
            consumed_bos_low = low_identity
        values["bos_up"][offset] = float(bos_up)
        values["bos_down"][offset] = float(bos_down)

        choch_up = regime < 0 and bos_up
        choch_down = regime > 0 and bos_down
        if choch_up:
            regime = 1
        elif choch_down:
            regime = -1
        elif regime == 0 and structure_available and structure_bias != 0.0:
            regime = int(np.sign(structure_bias))
        values["choch_up"][offset] = float(choch_up)
        values["choch_down"][offset] = float(choch_down)

        # A close-through invalidates the current level for sweep semantics.
        # Merely leaving and re-entering the wick condition never rearms it.
        sweep_up = bool(
            high_identity is not None
            and consumed_bos_high != high_identity
            and high_values[offset] > last_high_price
            and close_values[offset] <= last_high_price
        )
        sweep_down = bool(
            low_identity is not None
            and consumed_bos_low != low_identity
            and low_values[offset] < last_low_price
            and close_values[offset] >= last_low_price
        )
        sweep_up_event = sweep_up and consumed_sweep_high != high_identity
        sweep_down_event = sweep_down and consumed_sweep_low != low_identity
        if sweep_up_event:
            consumed_sweep_high = high_identity
        if sweep_down_event:
            consumed_sweep_low = low_identity
        if sweep_up_event or sweep_down_event:
            last_sweep_event_idx = absolute_row
        values["sweep_up"][offset] = float(sweep_up)
        values["sweep_down"][offset] = float(sweep_down)
        values["sweep_up_event"][offset] = float(sweep_up_event)
        values["sweep_down_event"][offset] = float(sweep_down_event)
        values["sweep_event_age_bars"][offset] = (
            raw_event_age_from_last_observed_row(
                absolute_row,
                last_sweep_event_idx,
            )
        )

    tail_count = min(2 * swing_lookback, len(combined_high))
    next_state = SMCCausalReplayStateV2(
        schema_version=SMC_CAUSAL_REPLAY_SCHEMA_VERSION,
        swing_lookback=swing_lookback,
        bars_seen=current.bars_seen + n_rows,
        high_tail=tuple(float(value) for value in combined_high[-tail_count:]),
        low_tail=tuple(float(value) for value in combined_low[-tail_count:]),
        last_high_idx=last_high_idx,
        last_high_price=last_high_price,
        prev_high_idx=prev_high_idx,
        prev_high_price=prev_high_price,
        last_low_idx=last_low_idx,
        last_low_price=last_low_price,
        prev_low_idx=prev_low_idx,
        prev_low_price=prev_low_price,
        structure_regime=regime,
        consumed_bos_high=consumed_bos_high,
        consumed_bos_low=consumed_bos_low,
        consumed_sweep_high=consumed_sweep_high,
        consumed_sweep_low=consumed_sweep_low,
        last_sweep_event_idx=last_sweep_event_idx,
    )
    _require_smc_causal_replay_state(next_state, swing_lookback=swing_lookback)
    return values, next_state


def _detect_swing_pivots(high: np.ndarray, low: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (swing_high_mask, swing_low_mask) — bool arrays, True at pivot bars.

    Swing high at bar i if high[i] is the maximum over [i-n, i+n].
    Swing low at bar i if low[i] is the minimum over [i-n, i+n].
    Edges (first n + last n bars) cannot be pivots.
    """
    nb = len(high)
    sh = np.zeros(nb, dtype=bool)
    sl = np.zeros(nb, dtype=bool)
    for i in range(n, nb - n):
        wh = high[i - n : i + n + 1]
        wl = low[i - n : i + n + 1]
        if high[i] >= wh.max() - 1e-12:
            sh[i] = True
        if low[i] <= wl.min() + 1e-12:
            sl[i] = True
    return sh, sl


def _track_recent_swings(
    swing_high_mask: np.ndarray,
    swing_low_mask: np.ndarray,
    n_lookback: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project the shared confirmed pivots for momentum-divergence consumers."""

    nb = len(swing_high_mask)
    last_sh = np.full(nb, -1, dtype=np.int64)
    prev_sh = np.full(nb, -1, dtype=np.int64)
    last_sl = np.full(nb, -1, dtype=np.int64)
    prev_sl = np.full(nb, -1, dtype=np.int64)
    cur_last_sh = cur_prev_sh = cur_last_sl = cur_prev_sl = -1
    for row in range(nb):
        confirm_idx = row - n_lookback
        if confirm_idx >= 0:
            if swing_high_mask[confirm_idx]:
                cur_prev_sh, cur_last_sh = cur_last_sh, confirm_idx
            if swing_low_mask[confirm_idx]:
                cur_prev_sl, cur_last_sl = cur_last_sl, confirm_idx
        last_sh[row], prev_sh[row] = cur_last_sh, cur_prev_sh
        last_sl[row], prev_sl[row] = cur_last_sl, cur_prev_sl
    return last_sh, prev_sh, last_sl, prev_sl


def compute_smc_features(
    df: pd.DataFrame,
    *,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    atr_col: str = "atr",
    swing_lookback: int = SWING_LOOKBACK,
    include_v30_additions: bool = False,
) -> pd.DataFrame:
    """Compute the raw local ``SMC_FEATURE_NAMES`` primitives.

    Required columns on df: high, low, close and atr. All are exact observed or
    causally computed inputs; no ATR sentinel is permitted.

    ``include_v30_additions`` appends ``SMC_V30_ADDITION_NAMES_V1``. Default False
    == the accepted canonical surface, byte-identical; the flag is a call-site
    CONTRACT switch (the ``swing_structure_v1.include_v29_additions``
    precedent), never an environment gate. The native M5/M1 specialist layer
    enables it; the canonical M5 frame remains dynamically column/hash-bound
    by the current-pair canonical owner, and a raw canonical column has no
    route into the model surface anyway —
    ``materialize_entry_model_native_train_feature_ranker_v1._candidate_universe``
    scans only specialist-layer owners, and
    ``build_entry_v10_ctx_training_dataset_v3._build_inline_seq_structure_extension``
    exposes the frozen base block plus specialist-layer outputs. Per-TF
    siblings are produced by :func:`compute_smc_mtf_primitives_v1`.
    """
    if not isinstance(include_v30_additions, bool):
        raise RuntimeError("[SMC_V30_ADDITION_FLAG_INVALID]")
    nb = len(df)
    if nb == 0:
        raise RuntimeError("[SMC_SOURCE_EMPTY] cannot produce SMC features from zero rows")
    required = (high_col, low_col, close_col, atr_col)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RuntimeError(f"[SMC_SOURCE_MISSING] required columns missing: {missing}")
    high = pd.to_numeric(df[high_col], errors="coerce").to_numpy(dtype=np.float64)
    low = pd.to_numeric(df[low_col], errors="coerce").to_numpy(dtype=np.float64)
    close = pd.to_numeric(df[close_col], errors="coerce").to_numpy(dtype=np.float64)
    atr = pd.to_numeric(df[atr_col], errors="coerce").to_numpy(dtype=np.float64)
    invalid_ohlc = (
        (~np.isfinite(high))
        | (~np.isfinite(low))
        | (~np.isfinite(close))
    )
    if invalid_ohlc.any() or np.isinf(atr).any():
        raise RuntimeError(
            "[SMC_SOURCE_NONFINITE] OHLC/ATR contains unavailable values: "
            f"count={int(np.count_nonzero(invalid_ohlc) + np.count_nonzero(np.isinf(atr)))}"
        )
    # Wilder ATR has one legitimate chronological warmup prefix.  Preserve
    # that unavailability in ATR-scaled outputs; never replace it with 1.0 or
    # zero.  A gap after the first finite ATR is malformed and still fails
    # closed.
    atr_available = np.isfinite(atr)
    if atr_available.any():
        first_atr = int(np.argmax(atr_available))
        if not atr_available[first_atr:].all():
            raise RuntimeError(
                "[SMC_SOURCE_NONFINITE] ATR availability must be one causal prefix"
            )
    invalid_geometry = (
        (high < low)
        | (high < close)
        | (low > close)
        | (atr_available & (atr <= 0.0))
    )
    if invalid_geometry.any():
        raise RuntimeError(
            "[SMC_SOURCE_INVALID] OHLC geometry must be valid and ATR strictly "
            f"positive: count={int(np.count_nonzero(invalid_geometry))}"
        )

    # One replay owner supplies confirmed level identity, structure regime,
    # close-through BOS/CHOCH and consumed sweep events. Both this local
    # surface and the MTF surface below use the same state machine.
    replay, _replay_state = replay_smc_causal_structure_v2(
        high,
        low,
        close,
        swing_lookback=swing_lookback,
    )
    last_sh = replay["last_high_idx"]
    prev_sh = replay["prev_high_idx"]
    last_sl = replay["last_low_idx"]
    prev_sl = replay["prev_low_idx"]
    last_sh_price = replay["last_high_price"]
    prev_sh_price = replay["prev_high_price"]
    last_sl_price = replay["last_low_price"]
    prev_sl_price = replay["prev_low_price"]
    swing_state = replay["swing_state"].astype(np.int8)
    bos_up = replay["bos_up"].astype(np.float32)
    bos_down = replay["bos_down"].astype(np.float32)

    # 3b. BOS displacement — V30 package 8A (2026-08-13), EMISSION ONLY.
    # Signed (close - broken level)/atr AT the firing bar, 0 off-event: the
    # flag-disambiguated-zero convention the sibling break-displacement
    # geometry already uses in compute_smc_mtf_primitives_v1 below
    # ((close - last_high)/atr, (last_low - close)/atr), reused rather than
    # re-derived. The sign is not a second direction rule: a BOS up closes
    # ABOVE its level (positive) and a BOS down BELOW its level (negative) by
    # construction of cond_up/cond_down. When both fire on one bar (possible
    # when the armed low level sits above the armed high level) the more
    # recently CONFIRMED level is carried, tie to the high side — the exact
    # documented rule swing_structure_v1's G1 displacement already uses.
    # The tie-break is taken on the EVENTS, not on the raw conditions: a bar
    # where the up-event fires while the down CONDITION merely persists is a
    # one-sided event and must not be routed to the low level (which would
    # leave a firing flag with a 0 displacement).
    fired_up = bos_up > 0.0
    fired_down = bos_down > 0.0
    fire_high = fired_up & (~fired_down | (last_sh >= last_sl))
    fire_low = fired_down & ~fire_high
    bos_displacement_atr = np.zeros(nb, dtype=np.float64)
    bos_displacement_atr[fire_high] = (
        close[fire_high] - last_sh_price[fire_high]
    ) / atr[fire_high]
    bos_displacement_atr[fire_low] = (
        close[fire_low] - last_sl_price[fire_low]
    ) / atr[fire_low]
    bos_displacement_atr = bos_displacement_atr.astype(np.float32)

    # CHOCH is the opposing close-through emitted by the replay only after a
    # confirmed structural side has armed the regime.
    choch = (
        (replay["choch_up"] > 0.0) | (replay["choch_down"] > 0.0)
    ).astype(np.float32)

    # 5. Liquidity sweep — wick beyond swept level, close back inside
    sweep_up = replay["sweep_up"].astype(np.float32)
    sweep_down = replay["sweep_down"].astype(np.float32)
    # Preserve sided depth evidence. A combined max would discard which side
    # fired on a double-sided bar and is exactly recoverable from these roots.
    sweep_up_depth_atr = np.where(
        sweep_up > 0.0,
        (high - last_sh_price) / atr,
        0.0,
    ).astype(np.float32)
    sweep_down_depth_atr = np.where(
        sweep_down > 0.0,
        (last_sl_price - low) / atr,
        0.0,
    ).astype(np.float32)
    sweep_event_age_bars = replay["sweep_event_age_bars"].astype(np.float32)
    sweep_up_depth_atr[~atr_available] = np.nan
    sweep_down_depth_atr[~atr_available] = np.nan
    bos_displacement_atr[~atr_available] = np.nan

    # One-shot events consume (polarity, confirmed pivot index). A false bar
    # between two wick-through observations does not rearm the same level.
    sweep_up_event = replay["sweep_up_event"].astype(np.float32)
    sweep_down_event = replay["sweep_down_event"].astype(np.float32)

    # Raw close position in the causal four-pivot envelope. Unknown warmup is
    # NaN. Equal-width envelopes remain unavailable: a zero denominator has
    # no honest numeric position, and the historical availability mask was a
    # global-ever flag that became permanently one after the fourth pivot.
    # Values outside [0, 1] are genuine close-beyond-envelope observations.
    pivot_stack = np.vstack(
        (last_sh_price, prev_sh_price, last_sl_price, prev_sl_price)
    )
    env_high = np.max(pivot_stack, axis=0)
    env_low = np.min(pivot_stack, axis=0)
    four_pivots_observed = (
        (last_sh >= 0)
        & (prev_sh >= 0)
        & (last_sl >= 0)
        & (prev_sl >= 0)
    )
    envelope_width = env_high - env_low
    envelope_present = four_pivots_observed & (envelope_width > 0.0)
    envelope_position = np.full(nb, np.nan, dtype=np.float64)
    np.divide(
        close - env_low,
        envelope_width,
        out=envelope_position,
        where=envelope_present,
    )

    out_cols = {
        "smc_swing_state": swing_state,
        "smc_bos_up": bos_up,
        "smc_bos_down": bos_down,
        "smc_choch": choch,
        "smc_sweep_up": sweep_up,
        "smc_sweep_down": sweep_down,
        "smc_sweep_event_age_bars": sweep_event_age_bars,
        "smc_pivot_envelope_position": envelope_position.astype(np.float32),
    }
    if include_v30_additions:
        out_cols["smc_bos_displacement_atr"] = bos_displacement_atr
        out_cols["smc_sweep_up_depth_atr"] = sweep_up_depth_atr
        out_cols["smc_sweep_down_depth_atr"] = sweep_down_depth_atr
        out_cols["smc_sweep_up_event"] = sweep_up_event
        out_cols["smc_sweep_down_event"] = sweep_down_event
    expected_columns = tuple(SMC_FEATURE_NAMES) + (
        SMC_V30_ADDITION_NAMES_V1 if include_v30_additions else ()
    )
    if tuple(out_cols) != expected_columns:
        raise RuntimeError("[SMC_OUTPUT_ORDER_INVALID]")
    out = pd.DataFrame(out_cols, index=df.index)
    return out


SMC_FEATURE_NAMES = [
    "smc_swing_state",
    "smc_bos_up",
    "smc_bos_down",
    "smc_choch",
    "smc_sweep_up",
    "smc_sweep_down",
    "smc_sweep_event_age_bars",
    "smc_pivot_envelope_position",
]
# V30 package 8A (2026-08-13) — owner-parity emissions for the M5 SMC block,
# declared separately from SMC_FEATURE_NAMES because the canonical local block
# and specialist additions have distinct producers and artifact identities.
# Each name is a quantity this owner already computes
# and discards, or the sided/event form its own MTF sibling already
# emits (docs/INDICATOR_FIDELITY_AUDIT_20260813.md §4b: "a fix landed in one
# owner and not its sibling"):
#   smc_bos_displacement_atr   signed (close - broken level)/atr at the firing
#                              bar, 0 off-event (flag-disambiguated zero)
#   smc_sweep_up/down_depth_atr  raw sided depths, including double sweeps
#   smc_sweep_up/down_event    one-shot confirmed-level-identity events
SMC_V30_ADDITION_NAMES_V1 = (
    "smc_bos_displacement_atr",
    "smc_sweep_up_depth_atr",
    "smc_sweep_down_depth_atr",
    "smc_sweep_up_event",
    "smc_sweep_down_event",
)
if set(SMC_V30_ADDITION_NAMES_V1) & set(SMC_FEATURE_NAMES) or len(
    set(SMC_V30_ADDITION_NAMES_V1)
) != len(SMC_V30_ADDITION_NAMES_V1):
    raise RuntimeError("[SMC_V30_ADDITION_NAMES_INVALID]")
# Exact fixed-width primitives for the multi-resolution Entry surface.  Unlike
# the historical M5 contract above, this contract is independent of ambient
# environment flags and never emits numeric unknown/sentinel values.  Rows are
# NaN until two highs and two lows have been causally confirmed; the shared HTF
# matrix owner trims that one chronological warmup prefix before a row can
# reach training or serving.
SMC_MTF_FEATURE_NAMES_V1 = (
    "mtf_smc_structure_bias",
    "mtf_smc_bos_up",
    "mtf_smc_bos_down",
    "mtf_smc_choch_up",
    "mtf_smc_choch_down",
    "mtf_smc_sweep_up",
    "mtf_smc_sweep_down",
    "mtf_smc_sweep_up_depth_atr",
    "mtf_smc_sweep_down_depth_atr",
    "mtf_smc_pivot_envelope_position",
    "mtf_smc_range_width_atr",
    # V30 package 8A (2026-08-13), EMISSION ONLY — appended so the pre-existing
    # per-TF column order is byte-stable ahead of them:
    #   mtf_smc_bos_displacement_atr  signed (close - broken level)/atr at the
    #     BOS bar, 0 off-event.  Same construction as the
    #     mtf_geometry_*_break_displacement_atr siblings below, event-gated;
    #     the BOS flags already tell 0-the-value from 0-the-non-event.
    #   mtf_smc_sweep_up/down_event  the sweep flags above are per-bar
    #     CONDITIONS. Events consume the confirmed pivot identity once and
    #     rearm only when a new identity is confirmed.
    #     The repeating flags stay untouched (rule 25b sweeps the defect class
    #     across the sibling owner without changing an accepted field).
    "mtf_smc_bos_displacement_atr",
    "mtf_smc_sweep_up_event",
    "mtf_smc_sweep_down_event",
    "mtf_smc_sweep_event_age_bars",
)

SMC_MTF_GEOMETRY_FEATURE_NAMES_V1 = (
    "mtf_geometry_support_dist_atr",
    "mtf_geometry_resistance_dist_atr",
    "mtf_geometry_support_age_bars",
    "mtf_geometry_resistance_age_bars",
    "mtf_geometry_support_rail_slope_atr_per_bar",
    "mtf_geometry_resistance_rail_slope_atr_per_bar",
    "mtf_geometry_support_break_displacement_atr",
    "mtf_geometry_resistance_break_displacement_atr",
    "mtf_geometry_nearest_level_abs_atr",
    "mtf_geometry_range_mid_dist_atr",
)

SMC_PRIMITIVE_FORMULA_CONTRACT = (
    "identity=confirmed_pivot_polarity_and_absolute_row",
    "events=one_shot_bos_choch_and_sweep_on_closed_native_clock",
    "sweep_age=raw_uncapped_native_bars_with_nan_before_first_event",
    "sweep_depth=separate_up_and_down_atr_depths",
    "envelope=raw_close_position_in_four_confirmed_pivots_nan_if_zero_width",
    "warmup=nan_until_four_pivots_for_envelope_surfaces",
)


def _smc_sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def smc_primitive_contract_metadata() -> dict[str, object]:
    """Return the exact shared local/MTF SMC owner identity."""

    return {
        "owner": "gx1.features.smc_v1",
        "schema_version": SMC_PRIMITIVE_CONTRACT_SCHEMA_VERSION,
        "causal_replay_schema_version": SMC_CAUSAL_REPLAY_SCHEMA_VERSION,
        "formula_sha256": _smc_sha256_json(SMC_PRIMITIVE_FORMULA_CONTRACT),
        "local_base_names": list(SMC_FEATURE_NAMES),
        "local_base_names_sha256": _smc_sha256_json(SMC_FEATURE_NAMES),
        "local_addition_names": list(SMC_V30_ADDITION_NAMES_V1),
        "local_addition_names_sha256": _smc_sha256_json(
            SMC_V30_ADDITION_NAMES_V1
        ),
        "mtf_names": list(SMC_MTF_FEATURE_NAMES_V1),
        "mtf_names_sha256": _smc_sha256_json(SMC_MTF_FEATURE_NAMES_V1),
    }


def compute_smc_mtf_primitives_v1(
    df: pd.DataFrame,
    *,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    atr_col: str = "atr",
    swing_lookback: int = SWING_LOOKBACK,
) -> pd.DataFrame:
    """Return fixed, causal SMC and S/R geometry roots for one resolution.

    The same observed-bar calculation runs independently on M5, M15, H1, H4
    and D1.  It contains no direction decision, cross-timeframe proxy, ambient
    feature flag or resolution-specific weight.
    """
    if (
        isinstance(swing_lookback, bool)
        or not isinstance(swing_lookback, int)
        or swing_lookback < 1
    ):
        raise RuntimeError(
            f"[SMC_MTF_SWING_LOOKBACK_INVALID] {swing_lookback!r}"
        )
    required = (high_col, low_col, close_col, atr_col)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RuntimeError(f"[SMC_MTF_SOURCE_MISSING] {missing}")
    if len(df) == 0:
        raise RuntimeError("[SMC_MTF_SOURCE_EMPTY]")

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
        raise RuntimeError("[SMC_MTF_SOURCE_NONFINITE]")
    atr_available = np.isfinite(atr)
    if atr_available.any():
        first_atr = int(np.argmax(atr_available))
        if not atr_available[first_atr:].all():
            raise RuntimeError("[SMC_MTF_ATR_AVAILABILITY_INVALID]")
    if (
        np.any(high <= 0.0)
        or np.any(low <= 0.0)
        or np.any(close <= 0.0)
        or np.any(atr[atr_available] <= 0.0)
        or np.any(high < low)
        or np.any(high < close)
        or np.any(low > close)
    ):
        raise RuntimeError("[SMC_MTF_SOURCE_GEOMETRY_INVALID]")

    n_rows = len(df)
    replay, _replay_state = replay_smc_causal_structure_v2(
        high,
        low,
        close,
        swing_lookback=swing_lookback,
    )
    last_high_idx = replay["last_high_idx"]
    prev_high_idx = replay["prev_high_idx"]
    last_low_idx = replay["last_low_idx"]
    prev_low_idx = replay["prev_low_idx"]
    last_high = replay["last_high_price"]
    prev_high = replay["prev_high_price"]
    last_low = replay["last_low_price"]
    prev_low = replay["prev_low_price"]

    # The structural range is the causal envelope of both most-recent
    # confirmed high pivots and both most-recent confirmed low pivots.  Using
    # only the latest high/low made a perfectly valid equal-pivot transition
    # collapse to zero width on real XAU M15 data (2025-08-18).  The previous
    # confirmed pivots are already required below and are known at the same
    # decision time, so the envelope defines the geometry without an epsilon,
    # sentinel, future observation, or dropped interior row.
    pivot_stack = np.vstack((last_high, prev_high, last_low, prev_low))
    range_low = np.min(pivot_stack, axis=0)
    range_high = np.max(pivot_stack, axis=0)
    channel_width = range_high - range_low
    four_pivots_observed = (
        (last_high_idx >= 0)
        & (prev_high_idx >= 0)
        & (last_low_idx >= 0)
        & (prev_low_idx >= 0)
    )
    available = four_pivots_observed & atr_available
    if not available.any():
        return pd.DataFrame(
            np.full(
                (
                    n_rows,
                    len(SMC_MTF_FEATURE_NAMES_V1)
                    + len(SMC_MTF_GEOMETRY_FEATURE_NAMES_V1),
                ),
                np.nan,
                dtype=np.float32,
            ),
            index=df.index,
            columns=(
                SMC_MTF_FEATURE_NAMES_V1
                + SMC_MTF_GEOMETRY_FEATURE_NAMES_V1
            ),
        )

    # The mean of the independently observed high- and low-structure signs:
    # +1=HH+HL, -1=LH+LL, and 0=mixed.  This is evidence, not a direction rule.
    structure_bias = replay["structure_bias"]

    # BOS is a causal crossing event, not a persistent "price remains outside
    # the last swing" state.  The latter is already represented continuously
    # by the support/resistance break-displacement geometry fields; keeping it
    # here as well would duplicate evidence and turn one break into many
    # identical event observations.
    bos_up = replay["bos_up"].copy()
    bos_down = replay["bos_down"].copy()
    choch_up = replay["choch_up"].copy()
    choch_down = replay["choch_down"].copy()

    # V30 package 8A: signed BOS displacement at the firing bar.  Both sides
    # can fire on one bar (the causal envelope allows last_low > last_high);
    # the more recently CONFIRMED level is carried, tie to the high side — the
    # documented rule swing_structure_v1's G1 displacement already uses.
    use_high_level = (bos_up > 0.0) & (
        (bos_down <= 0.0) | (last_high_idx >= last_low_idx)
    )
    use_low_level = (bos_down > 0.0) & ~use_high_level
    bos_displacement = np.zeros(n_rows, dtype=np.float64)
    np.divide(close - last_high, atr, out=bos_displacement, where=use_high_level)
    np.divide(close - last_low, atr, out=bos_displacement, where=use_low_level)

    sweep_up = replay["sweep_up"] > 0.0
    sweep_down = replay["sweep_down"] > 0.0
    sweep_up_depth = np.where(sweep_up, (high - last_high) / atr, 0.0)
    sweep_down_depth = np.where(sweep_down, (last_low - low) / atr, 0.0)
    sweep_up_event = replay["sweep_up_event"] > 0.0
    sweep_down_event = replay["sweep_down_event"] > 0.0
    envelope_present = channel_width > 0.0
    envelope_position = np.full(n_rows, np.nan, dtype=np.float64)
    np.divide(
        close - range_low,
        channel_width,
        out=envelope_position,
        where=available & envelope_present,
    )

    row_index = np.arange(n_rows, dtype=np.int64)
    support_dist = (close - last_low) / atr
    resistance_dist = (last_high - close) / atr
    support_age = row_index - last_low_idx
    resistance_age = row_index - last_high_idx
    support_rail_span = last_low_idx - prev_low_idx
    resistance_rail_span = last_high_idx - prev_high_idx
    support_slope = np.full(n_rows, np.nan, dtype=np.float64)
    resistance_slope = np.full(n_rows, np.nan, dtype=np.float64)
    np.divide(
        last_low - prev_low,
        support_rail_span.astype(np.float64) * atr,
        out=support_slope,
        where=available & (support_rail_span > 0),
    )
    np.divide(
        last_high - prev_high,
        resistance_rail_span.astype(np.float64) * atr,
        out=resistance_slope,
        where=available & (resistance_rail_span > 0),
    )
    support_break = np.maximum(last_low - close, 0.0) / atr
    resistance_break = np.maximum(close - last_high, 0.0) / atr
    nearest_level = np.minimum(np.abs(support_dist), np.abs(resistance_dist))
    range_mid_dist = (close - ((range_high + range_low) / 2.0)) / atr

    values = {
        "mtf_smc_structure_bias": structure_bias,
        "mtf_smc_bos_up": bos_up,
        "mtf_smc_bos_down": bos_down,
        "mtf_smc_choch_up": choch_up,
        "mtf_smc_choch_down": choch_down,
        "mtf_smc_sweep_up": sweep_up.astype(np.float64),
        "mtf_smc_sweep_down": sweep_down.astype(np.float64),
        "mtf_smc_sweep_up_depth_atr": sweep_up_depth,
        "mtf_smc_sweep_down_depth_atr": sweep_down_depth,
        "mtf_smc_pivot_envelope_position": envelope_position,
        "mtf_smc_range_width_atr": channel_width / atr,
        "mtf_smc_bos_displacement_atr": bos_displacement,
        "mtf_smc_sweep_up_event": sweep_up_event.astype(np.float64),
        "mtf_smc_sweep_down_event": sweep_down_event.astype(np.float64),
        "mtf_smc_sweep_event_age_bars": replay["sweep_event_age_bars"],
        "mtf_geometry_support_dist_atr": support_dist,
        "mtf_geometry_resistance_dist_atr": resistance_dist,
        "mtf_geometry_support_age_bars": support_age,
        "mtf_geometry_resistance_age_bars": resistance_age,
        "mtf_geometry_support_rail_slope_atr_per_bar": support_slope,
        "mtf_geometry_resistance_rail_slope_atr_per_bar": resistance_slope,
        "mtf_geometry_support_break_displacement_atr": support_break,
        "mtf_geometry_resistance_break_displacement_atr": resistance_break,
        "mtf_geometry_nearest_level_abs_atr": nearest_level,
        "mtf_geometry_range_mid_dist_atr": range_mid_dist,
    }
    expected_names = (
        SMC_MTF_FEATURE_NAMES_V1 + SMC_MTF_GEOMETRY_FEATURE_NAMES_V1
    )
    if tuple(values) != expected_names:
        raise RuntimeError("[SMC_MTF_OUTPUT_ORDER_INVALID]")

    out = pd.DataFrame(index=df.index, columns=expected_names, dtype=np.float64)
    for name, raw in values.items():
        column = np.asarray(raw, dtype=np.float64)
        column[~available] = np.nan
        out[name] = column
    numeric = out.to_numpy(dtype=np.float64, copy=False)
    if np.isinf(numeric).any():
        raise RuntimeError("[SMC_MTF_OUTPUT_AVAILABILITY_INVALID]")
    for column in range(numeric.shape[1]):
        finite = np.isfinite(numeric[:, column])
        if finite.any() and not finite[int(np.argmax(finite)) :].all():
            raise RuntimeError("[SMC_MTF_OUTPUT_AVAILABILITY_INVALID]")
    return out.astype(np.float32)
