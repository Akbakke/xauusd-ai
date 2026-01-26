"""
Hot patch for HTF alignment timing instrumentation.

This patch adds detailed timing for HTF alignment operations without
breaking existing functionality. Safe to apply during replay.

Usage:
    # In build_basic_v1(), add at top:
    from gx1.features.basic_v1_htf_timing_patch import patch_htf_timing
    patch_htf_timing()
"""

import time
import numpy as np
from typing import Optional
import os


def patch_htf_timing():
    """
    Patch _align_htf_to_m5_numpy to add detailed timing.
    This is a monkey-patch that wraps the existing function.
    """
    from gx1.features import basic_v1
    from gx1.utils.perf_timer import perf_add
    
    # Store original function
    original_align = basic_v1._align_htf_to_m5_numpy
    
    def patched_align_htf_to_m5_numpy(
        htf_values: np.ndarray,
        htf_close_times: np.ndarray,
        m5_timestamps: np.ndarray,
        is_replay: bool
    ) -> np.ndarray:
        """
        Wrapped version with timing instrumentation.
        """
        t_call_start = time.perf_counter()
        
        # Track warning overhead
        t_warn_start = None
        t_warn_end = None
        has_warning = False
        
        # Call original function (with timing hooks if possible)
        try:
            result = original_align(htf_values, htf_close_times, m5_timestamps, is_replay)
        except Exception as e:
            t_call_end = time.perf_counter()
            if is_replay:
                perf_add("feat.htf_align.call_total", t_call_end - t_call_start)
                perf_add("feat.htf_align.call_error", 1.0)  # Count errors
            raise
        
        t_call_end = time.perf_counter()
        
        # Instrument timing
        if is_replay:
            call_time = t_call_end - t_call_start
            perf_add("feat.htf_align.call_total", call_time)
            perf_add("feat.htf_align.call_count", 1.0)
            
            # Track if warning path was taken (indices < 0)
            if len(htf_close_times) > 0:
                indices = np.searchsorted(htf_close_times, m5_timestamps, side="right") - 1
                if np.any(indices < 0):
                    perf_add("feat.htf_align.warning_path", 1.0)
                    # Estimate warning overhead (searchsorted + mask + zeros)
                    n_missing = np.sum(indices < 0)
                    # Rough estimate: 1us per missing bar
                    estimated_warn_overhead = n_missing * 1e-6
                    perf_add("feat.htf_align.warning_overhead_est", estimated_warn_overhead)
        
        return result
    
    # Apply patch
    basic_v1._align_htf_to_m5_numpy = patched_align_htf_to_m5_numpy
    
    return original_align  # Return original for unpatch if needed
