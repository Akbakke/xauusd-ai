"""Joint entry+exit IQL policy: compose skip classifier (entry) with exit policy.

For each candidate trade in the dataset, the joint policy decides:
  1. Should we TAKE the trade? (entry decision, handled by skip classifier)
  2. If TAKEN, when do we EXIT? (exit decision, handled by IQL/CQL/Dist-IQL)

The joint pnl per trade is:
  - 0.0 if SKIP
  - exit_pnl_at_chosen_index if TAKE

This module is *algorithm-agnostic*: it accepts callable hooks for both the
skip and exit policies, so the same joint logic works whether the exit
policy is ridge IQL, GPU NN IQL, CQL, or distributional Q.

Mission framing
---------------
Per the user: "tar vi de riktige tradesa, for tidlig, for sent, eller skulle
ikke tradet". This module makes that question quantifiable:
  - "skulle ikke tradet": p_skip dominates → joint sets pnl = 0
  - "for tidlig / for sent": exit policy chooses an index different from
    the realized exit, with measurable pnl delta vs realized
  - "riktige tradesa": joint pnl > realized pnl

This is a research-only compute primitive. No artifacts are written here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

# Type aliases for the policy callables.
SkipPolicy = Callable[[np.ndarray], np.ndarray]
"""Maps per-trade entry-state matrix (n, d_entry) -> p_skip (n,) in [0, 1]."""

ExitPolicy = Callable[[np.ndarray], np.ndarray]
"""Maps per-bar exit-state matrix (n_bars, d_exit) -> p_exit_per_bar (n_bars,) in [0, 1]."""


@dataclass
class JointPolicyResult:
    n_candidates_v1: int
    n_skipped_v1: int
    n_taken_v1: int
    skip_rate_v1: float
    take_pnl_total_bps_v1: float
    skip_floor_total_bps_v1: float  # what the skipped trades' realized pnl would have been
    joint_pnl_total_bps_v1: float
    realized_pnl_total_bps_v1: float
    delta_vs_realized_bps_v1: float
    per_candidate_pnl_v1: list[float]
    per_candidate_status_v1: list[str]  # 'TAKE' or 'SKIP'


def _exit_pnl_for_taken_trade(
    per_bar_trade: pd.DataFrame, exit_idx_in_trade: int
) -> float:
    """PnL at the policy-chosen exit row of a single trade.

    per_bar_trade is sorted by bars_held_v1 ascending. exit_idx_in_trade is
    the 0-based row index within this trade.
    """
    if exit_idx_in_trade >= len(per_bar_trade):
        exit_idx_in_trade = len(per_bar_trade) - 1
    row = per_bar_trade.iloc[exit_idx_in_trade]
    return float(row.get("running_pnl_at_close_bps_v1", 0.0))


def _first_exit_index(p_exit: np.ndarray, threshold: float = 0.5) -> int:
    """First bar where p_exit >= threshold; -1 if never crossed."""
    above = np.where(p_exit >= threshold)[0]
    if len(above) == 0:
        return -1
    return int(above[0])


def evaluate_joint_policy(
    per_bar: pd.DataFrame,
    entry_features_per_trade: pd.DataFrame,
    exit_features_per_bar: np.ndarray,
    skip_policy: SkipPolicy,
    exit_policy: ExitPolicy,
    *,
    skip_threshold: float = 0.5,
    exit_threshold: float = 0.5,
) -> JointPolicyResult:
    """Compose skip + exit policies and evaluate joint pnl across the cohort.

    per_bar: DataFrame with columns candidate_uid_v1, bars_held_v1,
        running_pnl_at_close_bps_v1. One row per held bar per trade.
    entry_features_per_trade: DataFrame with one row per candidate_uid_v1
        plus a numeric entry-feature matrix (preserve uid order).
    exit_features_per_bar: (n_bars, d_exit) feature matrix aligned to per_bar.
    skip_policy: returns p_skip per candidate.
    exit_policy: returns p_exit per bar.
    skip_threshold: take only if p_skip < threshold.
    exit_threshold: exit at first bar where p_exit >= threshold.
    """
    if per_bar.empty:
        return JointPolicyResult(
            n_candidates_v1=0, n_skipped_v1=0, n_taken_v1=0,
            skip_rate_v1=0.0, take_pnl_total_bps_v1=0.0,
            skip_floor_total_bps_v1=0.0, joint_pnl_total_bps_v1=0.0,
            realized_pnl_total_bps_v1=0.0, delta_vs_realized_bps_v1=0.0,
            per_candidate_pnl_v1=[], per_candidate_status_v1=[],
        )

    uid_order = entry_features_per_trade["candidate_uid_v1"].astype(str).tolist()
    entry_X = entry_features_per_trade.drop(columns=["candidate_uid_v1"]).to_numpy(dtype=np.float32)
    p_skip = skip_policy(entry_X)
    if p_skip.shape != (len(uid_order),):
        raise ValueError(
            f"skip_policy must return shape ({len(uid_order)},); got {p_skip.shape}"
        )
    p_exit = exit_policy(exit_features_per_bar.astype(np.float32))
    if p_exit.shape[0] != len(per_bar):
        raise ValueError(
            f"exit_policy must return shape ({len(per_bar)},); got {p_exit.shape}"
        )

    per_bar = per_bar.reset_index(drop=True).copy()
    per_bar["_p_exit_v1"] = p_exit

    per_candidate_pnl: list[float] = []
    per_candidate_status: list[str] = []
    take_pnl = 0.0
    skip_floor_pnl = 0.0  # would-have-been-pnl of skipped trades (realized)
    realized_total = 0.0
    n_skipped = 0
    n_taken = 0

    grouped = per_bar.groupby("candidate_uid_v1", sort=False)
    skip_lookup = dict(zip(uid_order, p_skip))

    for uid, group in grouped:
        group = group.sort_values("bars_held_v1").reset_index(drop=True)
        realized_pnl_at_trade_end = float(
            group["running_pnl_at_close_bps_v1"].iloc[-1]
            if len(group) else 0.0
        )
        realized_total += realized_pnl_at_trade_end
        skip_p = float(skip_lookup.get(str(uid), 0.0))
        if skip_p >= skip_threshold:
            per_candidate_pnl.append(0.0)
            per_candidate_status.append("SKIP")
            skip_floor_pnl += realized_pnl_at_trade_end
            n_skipped += 1
            continue
        # TAKE: choose exit by exit policy.
        exit_idx = _first_exit_index(group["_p_exit_v1"].to_numpy(), exit_threshold)
        if exit_idx < 0:
            # Exit policy never triggered → fall through to realized last bar.
            exit_idx = len(group) - 1
        pnl = _exit_pnl_for_taken_trade(group, exit_idx)
        per_candidate_pnl.append(pnl)
        per_candidate_status.append("TAKE")
        take_pnl += pnl
        n_taken += 1

    n_total = n_taken + n_skipped
    skip_rate = n_skipped / n_total if n_total > 0 else 0.0
    joint_total = take_pnl  # SKIPs contribute 0
    return JointPolicyResult(
        n_candidates_v1=n_total,
        n_skipped_v1=n_skipped,
        n_taken_v1=n_taken,
        skip_rate_v1=skip_rate,
        take_pnl_total_bps_v1=take_pnl,
        skip_floor_total_bps_v1=skip_floor_pnl,
        joint_pnl_total_bps_v1=joint_total,
        realized_pnl_total_bps_v1=realized_total,
        delta_vs_realized_bps_v1=joint_total - realized_total,
        per_candidate_pnl_v1=per_candidate_pnl,
        per_candidate_status_v1=per_candidate_status,
    )


def diagnose_trade_timing(
    per_bar: pd.DataFrame,
    exit_idx_chosen: dict[str, int],
) -> pd.DataFrame:
    """Diagnose per-trade timing labels: TOO_EARLY, TOO_LATE, ON_TIME.

    A trade is TOO_EARLY if the chosen exit_idx is BEFORE the realized
    bar that achieved peak running pnl; TOO_LATE if the chosen exit_idx
    is AFTER the peak; ON_TIME otherwise.

    Returns per-uid DataFrame with timing label and pnl deltas.
    """
    rows: list[dict[str, Any]] = []
    for uid, group in per_bar.groupby("candidate_uid_v1", sort=False):
        group = group.sort_values("bars_held_v1").reset_index(drop=True)
        if not len(group):
            continue
        pnl_series = group["running_pnl_at_close_bps_v1"].to_numpy()
        peak_idx = int(np.argmax(pnl_series))
        chosen_idx = int(exit_idx_chosen.get(str(uid), len(group) - 1))
        if chosen_idx < peak_idx:
            label = "TOO_EARLY"
        elif chosen_idx > peak_idx:
            label = "TOO_LATE"
        else:
            label = "ON_TIME"
        rows.append({
            "candidate_uid_v1": str(uid),
            "timing_label_v1": label,
            "chosen_exit_idx_v1": chosen_idx,
            "peak_pnl_idx_v1": peak_idx,
            "chosen_pnl_bps_v1": float(pnl_series[min(chosen_idx, len(pnl_series) - 1)]),
            "peak_pnl_bps_v1": float(pnl_series[peak_idx]),
            "regret_bps_v1": float(pnl_series[peak_idx] - pnl_series[
                min(chosen_idx, len(pnl_series) - 1)
            ]),
        })
    return pd.DataFrame(rows)
