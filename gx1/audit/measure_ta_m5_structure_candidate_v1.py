"""TRAIN-only source candidate for the next confirmed M5 swing pair.

This is a target diagnostic, not a selected S/F target or model input.  The
caller supplies closed M5 bars, an issue subset and a fold target cutoff.
No source reads, model forwards, training, TEST or trading occur here.
"""
from __future__ import annotations

import numpy as np

from gx1.features.smc_v1 import SWING_LOOKBACK, replay_smc_causal_structure_v2

M5_NS = 300 * 10**9
TRAIN_START_NS = int(np.datetime64("2024-05-01T00:00:00", "ns").astype(np.int64))
TRAIN_END_NS = int(np.datetime64("2026-01-01T00:00:00", "ns").astype(np.int64))
UNKNOWN, MIXED, UP, DOWN, TIE = -1, 0, 1, 2, 3


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("TA_M5_STRUCTURE_" + reason)


def _new_pivots(last_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    confirmation = np.flatnonzero(last_index[1:] != last_index[:-1]) + 1
    confirmation = confirmation[last_index[confirmation] >= 0]
    pivot = last_index[confirmation]
    _need(bool(np.all(pivot[1:] > pivot[:-1])), "PIVOT_ORDER")
    _need(bool(np.all(confirmation == pivot + SWING_LOOKBACK)), "PIVOT_CAUSAL_LAG")
    return pivot, confirmation


def measure_candidate(
    closed_ns: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    issue_rows: np.ndarray,
    *,
    fold_target_end_ns: int,
) -> dict[str, np.ndarray]:
    """Compare the next confirmed high/low with levels frozen at each issue.

    An observed pair means that both pivot bars start strictly after the
    issue, both confirmations close by the fold boundary, and neither the
    frozen nor future seven-bar pivot windows cross an M5 source gap.
    Missing components remain masked; they are never FLAT or no-touch.
    """
    arrays = [np.asarray(x) for x in (closed_ns, high, low, close, atr)]
    time, high, low, close, atr = arrays
    issues = np.asarray(issue_rows)
    n = len(time)
    _need(time.dtype == np.int64 and time.ndim == 1 and n >= 2 * SWING_LOOKBACK + 1,
          "TIME")
    _need(all(x.ndim == 1 and len(x) == n and np.issubdtype(x.dtype, np.floating)
              for x in (high, low, close, atr)), "VALUES")
    _need(bool(np.all(np.isfinite(high)) and np.all(np.isfinite(low))
               and np.all(np.isfinite(close)) and np.all(high >= low)
               and np.all((close >= low) & (close <= high))), "OHLC")
    _need(issues.dtype == np.int64 and issues.ndim == 1 and len(issues) > 0
          and bool(np.all((issues >= 0) & (issues < n)))
          and bool(np.all(issues[1:] > issues[:-1])), "ISSUES")
    _need(type(fold_target_end_ns) is int
          and TRAIN_START_NS < fold_target_end_ns <= TRAIN_END_NS
          and bool(np.all(time[1:] > time[:-1]))
          and int(time[-1]) < fold_target_end_ns
          and int(time[-1]) < TRAIN_END_NS
          and bool(np.all((time[issues] >= TRAIN_START_NS)
                          & (time[issues] < fold_target_end_ns))), "CLOCK")

    # The sole existing SMC owner determines pivot identities and release lag.
    replay, _ = replay_smc_causal_structure_v2(high, low, close)
    gap = np.concatenate(([0], np.cumsum(np.diff(time) != M5_NS, dtype=np.int64)))
    def clear_window(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return (left >= 0) & (right < n) & (gap[np.clip(right, 0, n - 1)]
                                                 == gap[np.clip(left, 0, n - 1)])

    m = len(issues)
    result: dict[str, np.ndarray] = {
        "issue_row": issues.copy(),
        "frozen_high_idx": replay["last_high_idx"][issues].copy(),
        "frozen_low_idx": replay["last_low_idx"][issues].copy(),
        "frozen_high": replay["last_high_price"][issues].copy(),
        "frozen_low": replay["last_low_price"][issues].copy(),
        "next_high_idx": np.full(m, -1, dtype=np.int64),
        "next_low_idx": np.full(m, -1, dtype=np.int64),
        "next_high_confirm_row": np.full(m, -1, dtype=np.int64),
        "next_low_confirm_row": np.full(m, -1, dtype=np.int64),
        "high_delta_atr": np.full(m, np.nan, dtype=np.float64),
        "low_delta_atr": np.full(m, np.nan, dtype=np.float64),
        "high_observed": np.zeros(m, dtype=bool),
        "low_observed": np.zeros(m, dtype=bool),
        "high_hit": np.zeros(m, dtype=bool),
        "low_hit": np.zeros(m, dtype=bool),
        "high_wick_reject": np.zeros(m, dtype=bool),
        "low_wick_reject": np.zeros(m, dtype=bool),
        "high_close_break": np.zeros(m, dtype=bool),
        "low_close_break": np.zeros(m, dtype=bool),
        "pair_class": np.full(m, UNKNOWN, dtype=np.int8),
        "same_pivot_bar": np.zeros(m, dtype=bool),
        "both_levels_hit": np.zeros(m, dtype=bool),
        "no_frozen_level_hit": np.zeros(m, dtype=bool),
    }
    base = np.isfinite(atr[issues]) & (atr[issues] > 0)
    for side in ("high", "low"):
        frozen_idx = result[f"frozen_{side}_idx"]
        frozen_price = result[f"frozen_{side}"]
        frozen_confirm = frozen_idx + SWING_LOOKBACK
        frozen_valid = (frozen_idx >= SWING_LOOKBACK) & np.isfinite(frozen_price)
        frozen_valid &= clear_window(frozen_idx - SWING_LOOKBACK, frozen_confirm)
        frozen_valid &= gap[issues] == gap[np.clip(frozen_confirm, 0, n - 1)]
        pivot, confirmation = _new_pivots(replay[f"last_{side}_idx"])
        position = np.searchsorted(pivot, issues + 1, side="left")
        available = position < len(pivot)
        chosen = np.minimum(position, max(len(pivot) - 1, 0))
        if not len(pivot):
            continue
        next_idx, next_confirm = pivot[chosen], confirmation[chosen]
        future_valid = (available & (time[next_confirm] < fold_target_end_ns)
                        & clear_window(next_idx - SWING_LOOKBACK, next_confirm)
                        & (gap[next_confirm] == gap[issues]))
        observed = base & frozen_valid & future_valid
        result[f"{side}_observed"] = observed
        result[f"next_{side}_idx"][observed] = next_idx[observed]
        result[f"next_{side}_confirm_row"][observed] = next_confirm[observed]
        target = high if side == "high" else low
        signed = (target[next_idx] - frozen_price) / atr[issues]
        result[f"{side}_delta_atr"][observed] = signed[observed]
        if side == "high":
            hit = target[next_idx] >= frozen_price
            reject = (target[next_idx] > frozen_price) & (close[next_idx] <= frozen_price)
            break_ = close[next_idx] > frozen_price
        else:
            hit = target[next_idx] <= frozen_price
            reject = (target[next_idx] < frozen_price) & (close[next_idx] >= frozen_price)
            break_ = close[next_idx] < frozen_price
        result[f"{side}_hit"][observed] = hit[observed]
        result[f"{side}_wick_reject"][observed] = reject[observed]
        result[f"{side}_close_break"][observed] = break_[observed]
    both = result["high_observed"] & result["low_observed"]
    hi = result["high_delta_atr"]
    lo = result["low_delta_atr"]
    pair = result["pair_class"]
    pair[both & (hi > 0) & (lo > 0)] = UP
    pair[both & (hi < 0) & (lo < 0)] = DOWN
    pair[both & (((hi > 0) & (lo < 0)) | ((hi < 0) & (lo > 0)))] = MIXED
    pair[both & ((hi == 0) | (lo == 0))] = TIE
    result["same_pivot_bar"][both] = (result["next_high_idx"][both]
                                      == result["next_low_idx"][both])
    pair[result["same_pivot_bar"]] = UNKNOWN  # no high/low event order
    result["both_levels_hit"][both] = (result["high_hit"][both]
                                       & result["low_hit"][both])
    result["no_frozen_level_hit"][both] = (~result["high_hit"][both]
                                           & ~result["low_hit"][both])
    return result
