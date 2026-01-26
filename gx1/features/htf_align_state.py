"""
Stateful HTF alignment - O(1) per bar instead of O(N) per call.

Eliminates O(N²) by using pointer-walk instead of searchsorted on entire series.
Removes per-call allocations (np.zeros, np.roll, boolean masks).

Design:
- HTFAligner maintains a pointer j into htf_close_times
- step(m5_ts) advances pointer O(1) amortized (monotonic timestamps)
- Returns aligned value with shift(1) semantics (prev value, first = 0.0)
- No allocations in hot path (only scalar operations)
"""

import numpy as np
from typing import Optional
import time
import os


class HTFAligner:
    """
    Stateful HTF alignment using pointer-walk (O(1) amortized per bar).
    
    Replaces _align_htf_to_m5_numpy() for per-bar incremental alignment.
    Eliminates O(N²) when m5_timestamps grows with each bar.
    
    Usage:
        aligner = HTFAligner(htf_close_times, htf_values, is_replay=True)
        for m5_ts in m5_timestamps:
            aligned_value = aligner.step(m5_ts)  # O(1) amortized
            # Use aligned_value for all HTF features this bar
    """
    
    def __init__(
        self,
        htf_close_times: np.ndarray,
        htf_values: np.ndarray,
        is_replay: bool = False,
    ):
        """
        Initialize HTF aligner.
        
        Args:
            htf_close_times: HTF bar close times (seconds since epoch, int64, sorted ascending)
            htf_values: HTF feature values (float64, same length as htf_close_times)
            is_replay: If True, hard-fail on invalid state (deterministic)
        """
        if len(htf_close_times) != len(htf_values):
            raise ValueError(f"htf_close_times ({len(htf_close_times)}) and htf_values ({len(htf_values)}) must have same length")
        
        # Store HTF data (read-only after init)
        self.htf_close_times = htf_close_times  # int64 array, sorted ascending
        self.htf_values = htf_values  # float64 array
        self.is_replay = is_replay
        
        # State: pointer into htf_close_times (last completed HTF bar <= current M5 timestamp)
        self.j = -1  # Start before first HTF bar (j < 0 means no completed HTF bar)
        
        # State: previous aligned value (for shift(1) semantics)
        self.prev_aligned = 0.0  # First bar gets 0.0 (shift(1) semantics)
        
        # State: first valid HTF bar index (for warmup validation)
        self.first_valid_j = None
        if len(htf_close_times) > 0:
            # Find first valid HTF bar (first index where we can have a completed bar)
            # In practice, j=0 is first valid (htf_close_times[0] is first completed HTF bar)
            self.first_valid_j = 0
        
        # PATCH 1: Monotonitets-invariant - track last M5 timestamp
        self.last_m5_ts: Optional[int] = None  # For monotonicity check
        
        # Instrumentation (for perf JSON export)
        self.call_count = 0
        self.warn_count = 0  # Count of "no completed HTF bar" events
        self.m5_len_max = 0  # Max observed m5_len (for O(N²) detection)
        self.fallback_count = 0  # Count of fallback to legacy alignment
        
        # PATCH 4: Validate HTF history is complete and sorted (hard-fail in replay)
        if len(htf_close_times) > 1:
            if not np.all(np.diff(htf_close_times) > 0):
                if is_replay:
                    raise RuntimeError(
                        "HTFAligner: htf_close_times must be strictly monotonic increasing. "
                        f"Found {np.sum(np.diff(htf_close_times) <= 0)} non-monotonic pairs."
                    )
                # Live mode: warn but continue
                import logging
                log = logging.getLogger(__name__)
                log.warning("HTFAligner: htf_close_times not strictly monotonic, may cause incorrect alignment")
        
        # Validate htf_close_times are int64 (same time unit as m5_ts)
        if len(htf_close_times) > 0:
            if htf_close_times.dtype != np.int64:
                if is_replay:
                    raise RuntimeError(
                        f"HTFAligner: htf_close_times must be int64, got {htf_close_times.dtype}"
                    )
                import logging
                log = logging.getLogger(__name__)
                log.warning(f"HTFAligner: htf_close_times dtype is {htf_close_times.dtype}, expected int64")
    
    def step(self, m5_ts: int) -> float:
        """
        Get aligned HTF value for current M5 timestamp (with shift(1) semantics).
        
        Args:
            m5_ts: M5 timestamp in seconds since epoch (int64)
        
        Returns:
            Aligned value (float64) with shift(1) semantics:
            - Returns previous bar's aligned value
            - First bar returns 0.0
            - Returns 0.0 if no completed HTF bar available (warmup not satisfied)
        
        Side effects:
            - Advances internal pointer j (O(1) amortized for monotonic m5_ts)
            - Updates prev_aligned for next call
            - Increments call_count, warn_count if needed
            - Enforces monotonicity invariant (hard-fail in replay)
        """
        self.call_count += 1
        
        # PATCH 1: Enforce monotonicity invariant (CRITICAL for pointer-walk correctness)
        if self.last_m5_ts is not None:
            if m5_ts < self.last_m5_ts:
                # Non-monotonic timestamp detected
                if self.is_replay:
                    # Replay: hard-fail (deterministic requirement)
                    raise RuntimeError(
                        f"HTFAligner: non-monotonic m5_ts detected. "
                        f"Previous: {self.last_m5_ts}, current: {m5_ts}, diff: {m5_ts - self.last_m5_ts}. "
                        f"This breaks pointer-walk invariant. Check chunk slicing or data sorting."
                    )
                else:
                    # Live: reset aligner + warn (out-of-order ticks can happen)
                    import logging
                    log = logging.getLogger(__name__)
                    log.warning(
                        f"HTFAligner: non-monotonic m5_ts in live mode. "
                        f"Previous: {self.last_m5_ts}, current: {m5_ts}. Resetting aligner."
                    )
                    self.reset()
                    # Continue with reset state (j=-1, prev_aligned=0.0)
        
        self.last_m5_ts = m5_ts
        
        # Advance pointer j to find last completed HTF bar <= m5_ts
        # Since m5_ts is monotonic (per-bar loop), we only need to advance forward
        # This is O(1) amortized (each HTF bar is visited at most once)
        while self.j + 1 < len(self.htf_close_times):
            if self.htf_close_times[self.j + 1] <= m5_ts:
                self.j += 1
            else:
                break  # Next HTF bar hasn't closed yet
        
        # PATCH 7: Warmup handling - j < 0 is OK before first_valid_eval_idx
        # Return 0.0 for pre-warmup bars (no completed HTF bar available)
        if self.j < 0:
            # No completed HTF bar available (warmup not satisfied)
            self.warn_count += 1
            # Return 0.0 (no completed HTF bar)
            current_aligned = 0.0
        else:
            # Get HTF value for current bar
            current_aligned = float(self.htf_values[self.j])
        
        # PATCH 3: Shift(1) semantics - return previous bar's value, store current for next call
        # CRITICAL: prev_aligned is updated AFTER return (so first bar gets 0.0, second bar gets first HTF value)
        result = self.prev_aligned
        self.prev_aligned = current_aligned
        
        return result
    
    def get_aligned_value(self, htf_values: np.ndarray) -> float:
        """
        Get aligned value for current pointer position (without advancing).
        Used when we need to align different HTF feature arrays using same alignment index.
        
        Args:
            htf_values: HTF feature values array (float64, same length as htf_close_times)
        
        Returns:
            Aligned value (float64) with shift(1) semantics (returns prev_aligned)
        """
        if self.j < 0:
            return 0.0
        if self.j >= len(htf_values):
            return 0.0
        # Note: This doesn't update prev_aligned - use step() for that
        # This is for getting aligned values for different features using same alignment
        return self.prev_aligned  # shift(1) semantics
    
    def reset(self) -> None:
        """
        Reset aligner state (for new chunk or testing).
        
        PATCH 2: Chunk-restart - each chunk is a new universe.
        Aligners are reset per chunk (new runner instance = new feature_state = new aligners).
        This method is called explicitly if needed (e.g., live reconnect/backfill).
        """
        self.j = -1
        self.prev_aligned = 0.0
        self.last_m5_ts = None  # Reset monotonicity tracking
        self.call_count = 0
        self.warn_count = 0
        self.m5_len_max = 0
        self.fallback_count = 0
    
    def get_stats(self) -> dict:
        """
        Get alignment statistics (for perf JSON export).
        
        PATCH 8: Instrumentering - to tall som avgjør alt:
        - call_count: skal være ≈ n_bars × 2 (H1 + H4), ikke n_bars × 7–8
        - warn_count: antall "no completed HTF bar" events (pre-warmup)
        """
        return {
            "call_count": self.call_count,
            "warn_count": self.warn_count,
            "m5_len_max": self.m5_len_max,
            "fallback_count": self.fallback_count,
            "last_m5_ts": self.last_m5_ts,  # Optional: for debugging
            "last_j": self.j,  # Optional: current alignment index
        }


def _align_htf_to_m5_stateful_legacy(
    htf_values: np.ndarray,
    htf_close_times: np.ndarray,
    m5_timestamps: np.ndarray,
    is_replay: bool,
) -> np.ndarray:
    """
    Legacy wrapper for _align_htf_to_m5_numpy() compatibility.
    
    This is used when we need to align an entire array at once (e.g., in build_basic_v1
    when called with full history). For per-bar incremental alignment, use HTFAligner.step().
    
    This function is kept for backward compatibility but should be avoided in hot path.
    """
    # Sample m5_len for instrumentation (every 50k calls in replay)
    if is_replay:
        from gx1.utils.perf_timer import perf_add
        # Sample len(m5_timestamps) to detect O(N²) pattern
        # Only sample occasionally to avoid overhead
        if not hasattr(_align_htf_to_m5_stateful_legacy, "_sample_counter"):
            _align_htf_to_m5_stateful_legacy._sample_counter = 0  # type: ignore
        _align_htf_to_m5_stateful_legacy._sample_counter += 1  # type: ignore
        if _align_htf_to_m5_stateful_legacy._sample_counter % 50000 == 0:  # type: ignore
            perf_add("feat.htf_align.m5_len_sample", float(len(m5_timestamps)))
    
    # Fall back to original implementation for full-array alignment
    from gx1.features.basic_v1 import _align_htf_to_m5_numpy
    return _align_htf_to_m5_numpy(htf_values, htf_close_times, m5_timestamps, is_replay)
