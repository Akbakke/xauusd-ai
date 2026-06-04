"""One-truth entry-admission circuit-breaker (2026-06-04).

Single source of truth for the live SAFETY circuit-breaker layer, callable
identically from live execution (v12_paper_runner / v12_pipeline) and the offline
harnesses (gx1/backtest/portfolio_sim_v1, Phase-6 gate, counterfactual replay) so
that the invariant "live == Phase-6 when no breaker fires" becomes PROVABLE.

These are RISK circuit-breakers (catastrophe clamps), NOT strategy gates: each runs
AFTER the learned policy's action and can only DOWNGRADE a TAKE to a SKIP — never
flip side, resize, or read Q-values / logits. When none fires the runtime is
byte-identical to Phase-6 OOT. They stay ALWAYS-ON (never PURE_PHASE6-gated): the
06-02 -2000 pile-up was PURE_PHASE6 un-gating the same-side cap.

Extracted VERBATIM from the live sites (kept in sync by the golden parity test
tests/test_circuit_breaker_parity.py):
  - CLUSTER1 same-M5 dedup ......... gx1/execution/v12_pipeline.py:328-358
  - opposing-side + same-side cap .. gx1/execution/v12_paper_runner.py:evaluate_entry_safety (157-172)
  - portfolio drawdown block ....... gx1/execution/v12_paper_runner.py:895-908

Live admission ORDER (mirrored here): CLUSTER1 (same-M5 same-side, then opposite) ->
opposing-side-open -> same-side hard cap -> portfolio combined-drawdown.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Hashable

# Reason strings — MUST match the live sites verbatim (journals + Phase-6 gate diff
# on these exact strings). Do not rename without updating both live sites + the gate.
REASON_OK = "ok"
REASON_CLUSTER1_SAME = "CLUSTER1_SAME_SIDE_SAME_M5"
REASON_CLUSTER1_OPP = "CLUSTER1_OPPOSITE_SIDE_SAME_M5"
REASON_OPPOSING = "blocked_opposing_side_open"
REASON_SAME_SIDE_CAP = "blocked_hard_same_side_cap"
REASON_DRAWDOWN = "blocked_portfolio_drawdown"

SIDE_LONG = "long"
SIDE_SHORT = "short"
_SIDES = (SIDE_LONG, SIDE_SHORT)

DEFAULT_HARD_MAX_SAME_SIDE = 2
DEFAULT_DRAWDOWN_BLOCK_BPS = -100.0


@dataclass(frozen=True)
class BreakerCfg:
    """Safety thresholds. NOT trained state, NOT strategy knobs — risk insurance."""
    hard_max_same_side: int = DEFAULT_HARD_MAX_SAME_SIDE
    drawdown_block_bps: float = DEFAULT_DRAWDOWN_BLOCK_BPS
    cluster1_enabled: bool = True

    @classmethod
    def from_env(cls, hard_max_same_side: int = DEFAULT_HARD_MAX_SAME_SIDE) -> "BreakerCfg":
        """Build from the same env the live sites read (GX1_DRAWDOWN_BLOCK_BPS,
        GX1_CLUSTER1_DISABLE). hard_max_same_side is passed by the caller (live
        reconciles it against broker truth)."""
        return cls(
            hard_max_same_side=int(hard_max_same_side),
            drawdown_block_bps=float(os.environ.get("GX1_DRAWDOWN_BLOCK_BPS", str(DEFAULT_DRAWDOWN_BLOCK_BPS))),
            cluster1_enabled=not bool(int(os.environ.get("GX1_CLUSTER1_DISABLE", "0"))),
        )


@dataclass(frozen=True)
class OpenBook:
    """Reconstructed open-trade book — the only state the breakers read.

    last_entry_m5_by_side maps side -> the M5-bar timestamp of the most recent
    entry on that side (the live `_last_entry_m5_by_side`), used for CLUSTER1.
    """
    n_long: int = 0
    n_short: int = 0
    combined_unrealized_bps: float = 0.0
    last_entry_m5_by_side: dict[str, Hashable] = field(default_factory=dict)

    def same_opp(self, side: str) -> tuple[int, int]:
        if side == SIDE_LONG:
            return self.n_long, self.n_short
        return self.n_short, self.n_long


def evaluate_same_opp_cap(side: str, n_same_side: int, n_opposing: int,
                          hard_max_same_side: int) -> tuple[bool, str, int]:
    """Exact mirror of v12_paper_runner.evaluate_entry_safety (opposing then cap).

    Kept identical so the live function can delegate here (one truth) with zero
    behavior change. Pinned by tests/test_circuit_breaker_parity.py.
    """
    if n_opposing > 0:
        return False, REASON_OPPOSING, n_opposing
    if n_same_side >= hard_max_same_side:
        return False, REASON_SAME_SIDE_CAP, n_same_side
    return True, REASON_OK, 0


def evaluate_entry_admission(book: OpenBook, side: str, decision_m5: Hashable | None,
                             cfg: BreakerCfg) -> tuple[bool, str, float]:
    """Post-policy entry admission across the full breaker chain.

    Returns (allowed, reason, detail). Mirrors the live ordering exactly:
    CLUSTER1 (same-M5 same-side, then opposite) -> opposing-open -> same-side cap
    -> combined-drawdown. detail is the offending count / pnl for logging.
    """
    if side not in _SIDES:
        raise ValueError(f"side must be one of {_SIDES}, got {side!r}")

    # 1) CLUSTER1 same-M5 dedup (v12_pipeline.py:328-358). Mirrors live `==`.
    if cfg.cluster1_enabled and decision_m5 is not None:
        opp = SIDE_SHORT if side == SIDE_LONG else SIDE_LONG
        if book.last_entry_m5_by_side.get(side) == decision_m5:
            return False, REASON_CLUSTER1_SAME, 0.0
        if book.last_entry_m5_by_side.get(opp) == decision_m5:
            return False, REASON_CLUSTER1_OPP, 0.0

    # 2)+3) opposing-side then same-side cap (evaluate_entry_safety).
    n_same, n_opp = book.same_opp(side)
    ok, reason, detail = evaluate_same_opp_cap(side, n_same, n_opp, cfg.hard_max_same_side)
    if not ok:
        return False, reason, float(detail)

    # 4) portfolio combined-unrealized drawdown (v12_paper_runner.py:895-908).
    if book.combined_unrealized_bps < cfg.drawdown_block_bps:
        return False, REASON_DRAWDOWN, float(book.combined_unrealized_bps)

    return True, REASON_OK, 0.0


def book_from_open_trades(open_trades: list[Any] | None,
                          floor_m5: Any = None) -> OpenBook:
    """Reconstruct an OpenBook from a list of open trades (duck-typed: each has
    .side, .current_pnl_bps, .entry_ts). `floor_m5` optionally floors entry_ts to
    the M5 bucket for last_entry_m5_by_side (pass a callable ts->m5_ts; default
    uses the raw entry_ts so callers that key on the recorded M5 bar pass it in).
    """
    if not open_trades:
        return OpenBook()
    n_long = sum(1 for t in open_trades if t.side == SIDE_LONG)
    n_short = sum(1 for t in open_trades if t.side == SIDE_SHORT)
    combined = float(sum(getattr(t, "current_pnl_bps", 0.0) for t in open_trades))
    last_by_side: dict[str, Hashable] = {}
    for t in sorted(open_trades, key=lambda x: x.entry_ts):
        m5 = floor_m5(t.entry_ts) if callable(floor_m5) else t.entry_ts
        last_by_side[t.side] = m5  # later (newer) entries overwrite -> most recent
    return OpenBook(n_long=n_long, n_short=n_short,
                    combined_unrealized_bps=combined, last_entry_m5_by_side=last_by_side)
