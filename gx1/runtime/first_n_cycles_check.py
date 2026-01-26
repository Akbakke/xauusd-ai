"""
First-N-Cycles Echo Check — Early Detection of Feature Issues

For the first N cycles (50-200), logs min/max/NaN-count for critical input blocks (seq/snap/ctx).
Hard-fails if NaN-count > 0 (or > extremely low tolerance) after warmup.

Dependencies (explicit install line):
  (no external dependencies beyond stdlib)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class CycleEchoStats:
    """Echo check statistics for one cycle."""
    
    cycle_num: int
    seq_min: float
    seq_max: float
    seq_nan_count: int
    snap_min: float
    snap_max: float
    snap_nan_count: int
    ctx_cat_nan_count: int = 0
    ctx_cont_nan_count: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "cycle_num": self.cycle_num,
            "seq_min": float(self.seq_min),
            "seq_max": float(self.seq_max),
            "seq_nan_count": self.seq_nan_count,
            "snap_min": float(self.snap_min),
            "snap_max": float(self.snap_max),
            "snap_nan_count": self.snap_nan_count,
            "ctx_cat_nan_count": self.ctx_cat_nan_count,
            "ctx_cont_nan_count": self.ctx_cont_nan_count,
        }


class FirstNCyclesChecker:
    """First-N-Cycles echo check tracker."""
    
    def __init__(
        self,
        n_cycles: int = 200,
        warmup_bars: int = 288,
        nan_tolerance: int = 0,
    ):
        """
        Initialize first-N-cycles checker.
        
        Args:
            n_cycles: Number of cycles to track (default: 200)
            warmup_bars: Number of warmup bars before strict validation (default: 288)
            nan_tolerance: Maximum allowed NaN count after warmup (default: 0)
        """
        self.n_cycles = n_cycles
        self.warmup_bars = warmup_bars
        self.nan_tolerance = nan_tolerance
        
        self.cycle_stats: List[CycleEchoStats] = []
        self.current_cycle = 0
        self.warmup_complete = False
    
    def check_cycle(
        self,
        seq_data: np.ndarray,
        snap_data: np.ndarray,
        ctx_cat: Optional[np.ndarray] = None,
        ctx_cont: Optional[np.ndarray] = None,
        current_bar_index: int = 0,
    ) -> Optional[str]:
        """
        Check one cycle and log stats.
        
        Args:
            seq_data: Sequence features [seq_len, n_features]
            snap_data: Snapshot features [n_features]
            ctx_cat: Context categorical features [n_ctx_cat] (optional)
            ctx_cont: Context continuous features [n_ctx_cont] (optional)
            current_bar_index: Current bar index (for warmup check)
        
        Returns:
            Error message if validation fails, None otherwise
        """
        self.current_cycle += 1
        
        # Check if we're past warmup
        if current_bar_index >= self.warmup_bars:
            self.warmup_complete = True
        
        # Compute stats
        seq_min = float(np.nanmin(seq_data))
        seq_max = float(np.nanmax(seq_data))
        seq_nan_count = int(np.isnan(seq_data).sum())
        
        snap_min = float(np.nanmin(snap_data))
        snap_max = float(np.nanmax(snap_data))
        snap_nan_count = int(np.isnan(snap_data).sum())
        
        ctx_cat_nan_count = 0
        if ctx_cat is not None:
            ctx_cat_nan_count = int(np.isnan(ctx_cat).sum())
        
        ctx_cont_nan_count = 0
        if ctx_cont is not None:
            ctx_cont_nan_count = int(np.isnan(ctx_cont).sum())
        
        # Store stats
        stats = CycleEchoStats(
            cycle_num=self.current_cycle,
            seq_min=seq_min,
            seq_max=seq_max,
            seq_nan_count=seq_nan_count,
            snap_min=snap_min,
            snap_max=snap_max,
            snap_nan_count=snap_nan_count,
            ctx_cat_nan_count=ctx_cat_nan_count,
            ctx_cont_nan_count=ctx_cont_nan_count,
        )
        self.cycle_stats.append(stats)
        
        # Log stats (first 10 cycles, then every 50)
        if self.current_cycle <= 10 or self.current_cycle % 50 == 0:
            log.info(
                "[FIRST_N_CYCLES] Cycle %d: seq=[%.4f, %.4f] NaN=%d, snap=[%.4f, %.4f] NaN=%d, "
                "ctx_cat_NaN=%d, ctx_cont_NaN=%d, warmup_complete=%s",
                self.current_cycle,
                seq_min, seq_max, seq_nan_count,
                snap_min, snap_max, snap_nan_count,
                ctx_cat_nan_count, ctx_cont_nan_count,
                self.warmup_complete,
            )
        
        # Hard-fail if NaN count exceeds tolerance after warmup
        if self.warmup_complete:
            total_nan = seq_nan_count + snap_nan_count + ctx_cat_nan_count + ctx_cont_nan_count
            if total_nan > self.nan_tolerance:
                error_msg = (
                    f"[FIRST_N_CYCLES_FAIL] NaN count exceeds tolerance after warmup: "
                    f"seq_NaN={seq_nan_count}, snap_NaN={snap_nan_count}, "
                    f"ctx_cat_NaN={ctx_cat_nan_count}, ctx_cont_NaN={ctx_cont_nan_count}, "
                    f"total_NaN={total_nan}, tolerance={self.nan_tolerance}, "
                    f"cycle={self.current_cycle}, bar_index={current_bar_index}"
                )
                return error_msg
        
        return None
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.cycle_stats:
            return {"n_cycles": 0}
        
        seq_nan_counts = [s.seq_nan_count for s in self.cycle_stats]
        snap_nan_counts = [s.snap_nan_count for s in self.cycle_stats]
        
        return {
            "n_cycles": len(self.cycle_stats),
            "warmup_complete": self.warmup_complete,
            "max_seq_nan": max(seq_nan_counts),
            "max_snap_nan": max(snap_nan_counts),
            "total_nan_after_warmup": sum(
                s.seq_nan_count + s.snap_nan_count + s.ctx_cat_nan_count + s.ctx_cont_nan_count
                for s in self.cycle_stats
                if s.cycle_num > self.warmup_bars
            ),
        }
    
    def should_continue_checking(self) -> bool:
        """Check if we should continue checking cycles."""
        return self.current_cycle < self.n_cycles
