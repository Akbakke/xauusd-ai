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

    def reset(self):
        """Reset cache and counters (useful for testing or new runs)."""
        self.htf_zscore_cache.clear()
        self.htf_cache_hits = 0
        self.htf_cache_misses = 0
        if self.r1_quantiles_state is not None:
            from gx1.features.rolling_state_numba import RollingR1Quantiles48State
            self.r1_quantiles_state = RollingR1Quantiles48State()
