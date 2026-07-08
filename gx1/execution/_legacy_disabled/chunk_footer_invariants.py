"""
Chunk Footer Bars Invariant.

Contract:
- Pure function, no I/O.
- When status=="ok": bars_total_input - bars_processed MUST equal total holdback.
  total_holdback = warmup_holdback_bars + tail_holdback_bars.
- When status!="ok" (stopped/failed): invariant not enforced.
"""

from __future__ import annotations


def check_bars_invariant(
    bars_total_input: int,
    bars_processed: int,
    tail_holdback_bars: int,
    status: str,
    warmup_holdback_bars: int = 0,
) -> bool:
    """
    Returns True if invariant holds.
    When status != "ok", always returns True (no enforcement).
    """
    s = (status or "").lower().strip()
    if s != "ok":
        return True

    # TRUTH-grade sanity in ok-case (avoid "accidental equality")
    if (
        bars_total_input < 0
        or bars_processed < 0
        or warmup_holdback_bars < 0
        or tail_holdback_bars < 0
    ):
        return False

    if bars_processed > bars_total_input:
        return False

    bars_gap = bars_total_input - bars_processed
    expected_gap = warmup_holdback_bars + tail_holdback_bars
    return bars_gap == expected_gap