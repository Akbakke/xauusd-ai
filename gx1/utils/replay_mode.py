"""
Replay Mode Enum - Strict separation between PREBUILT and BASELINE modes.

FASE 1: PREBUILT and BASELINE are two different programs (practically).
This enum enforces strict mode separation to prevent accidental mixing.
"""

from enum import Enum, auto


class ReplayMode(Enum):
    """
    Replay execution modes.
    
    PREBUILT: Uses pre-computed features from parquet file.
              - NO feature-building allowed
              - NO imports of build_basic_v1, build_live_entry_features, etc.
              - Only .loc[timestamp] on prebuilt DataFrame
    
    BASELINE: Builds features on-the-fly (normal replay).
              - Feature-building enabled
              - Prebuilt code paths are unavailable
    """
    PREBUILT = auto()
    BASELINE = auto()
    
    @classmethod
    def from_env(cls) -> "ReplayMode":
        """
        Determine replay mode from environment variables.
        
        Returns:
            ReplayMode.PREBUILT if GX1_REPLAY_USE_PREBUILT_FEATURES=1
            ReplayMode.BASELINE otherwise
        """
        import os
        prebuilt_enabled = os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES", "0") == "1"
        return cls.PREBUILT if prebuilt_enabled else cls.BASELINE
    
    def is_prebuilt(self) -> bool:
        """Check if mode is PREBUILT."""
        return self == ReplayMode.PREBUILT
    
    def is_baseline(self) -> bool:
        """Check if mode is BASELINE."""
        return self == ReplayMode.BASELINE
