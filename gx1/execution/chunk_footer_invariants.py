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
    Check bars invariant: bars_total_input - bars_processed == warmup_holdback_bars + tail_holdback_bars when complete.

    Returns True if invariant holds; False if violated (status==ok and gap != expected).
    When status != "ok", always returns True (no enforcement).
    """
    if status != "ok":
        return True  # Don't enforce on stopped/failed/early-abort
    bars_gap = bars_total_input - bars_processed
    expected_gap = warmup_holdback_bars + tail_holdback_bars
    return bars_gap == expected_gap
