# gx1/features/feature_state.py
# -*- coding: utf-8 -*-
"""
Feature state management for persistent caching across build_basic_v1 calls.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from gx1.features.rolling_state_numba import RollingR1Quantiles48State


@dataclass
class FeatureState:
    """
    Persistent state for feature building (caching, counters, etc).
    Should persist across multiple build_basic_v1 calls within a replay run.
    """
    htf_zscore_cache: Dict = field(default_factory=dict)
    htf_cache_hits: int = 0
    htf_cache_misses: int = 0
    r1_quantiles_state: Optional['RollingR1Quantiles48State'] = None
    # PATCH: Stateful HTF aligners (O(1) per bar instead of O(N) per call)
    h1_aligner: Optional['HTFAligner'] = None  # type: ignore
    h4_aligner: Optional['HTFAligner'] = None  # type: ignore
    # PATCH: Current aligned values (gjenbrukes for alle HTF features denne baren)
    h1_aligned_values: Optional[Dict[str, float]] = None  # type: ignore  # Maps feature name -> aligned value
    h4_aligned_values: Optional[Dict[str, float]] = None  # type: ignore
    # PATCH: Instrumentation (PATCH 8: to tall som avgjør alt)
    htf_align_call_count_total: int = 0  # Should be ≈ n_bars × 2 (H1 + H4), not n_bars × 7–8
    htf_align_m5_len_max: int = 0
    htf_align_fallback_count: int = 0  # Count of fallback to legacy alignment (should be 0 in replay)
    # PATCH: Replay invariant - count bars where HTF features were actually computed
    htf_feature_compute_bars: int = 0  # Count of bars where stateful aligner.step() was called
    
    def reset(self):
        """Reset cache and counters (useful for testing or new runs)."""
        self.htf_zscore_cache.clear()
        self.htf_cache_hits = 0
        self.htf_cache_misses = 0
        if self.r1_quantiles_state is not None:
            from gx1.features.rolling_state_numba import RollingR1Quantiles48State
            self.r1_quantiles_state = RollingR1Quantiles48State()
        # PATCH: Reset HTF aligners
        self.h1_aligner = None
        self.h4_aligner = None
        self.h1_aligned_values = None
        self.h4_aligned_values = None
        self.htf_align_call_count_total = 0
        self.htf_align_m5_len_max = 0
        self.htf_align_fallback_count = 0
        self.htf_feature_compute_bars = 0

