"""
Simple performance timer aggregator for feature timing breakdown.

Del 2A: Minimal timing utility to aggregate performance metrics by block name.
Uses contextvars for global access without signature changes.
"""

from __future__ import annotations

import contextvars
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


class PerfCollector:
    """
    Simple aggregator for timing metrics.
    
    Usage:
        collector = PerfCollector()
        set_current_perf(collector)
        perf_add("feat.basic_v1.total", 0.5)
        top = collector.top(5)  # Returns sorted list of (name, total_sec, count) tuples
    """
    
    def __init__(self):
        self._times: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)
    
    def add(self, name: str, dt_sec: float) -> None:
        """
        Add time to a named block.
        
        Parameters
        ----------
        name : str
            Block name (e.g., "feat.basic_v1.total")
        dt_sec : float
            Time in seconds to add
        """
        self._times[name] += dt_sec
        self._counts[name] += 1
    
    def inc(self, name: str, n: int = 1) -> None:
        """
        Increment call count for a named block.
        
        Parameters
        ----------
        name : str
            Block name
        n : int
            Count to increment (default 1)
        """
        self._counts[name] += n
    
    def top(self, n: int = 10) -> List[Tuple[str, float, int]]:
        """
        Get top N blocks by total time.
        
        Parameters
        ----------
        n : int
            Number of top blocks to return
            
        Returns
        -------
        List[Tuple[str, float, int]]
            List of (name, total_sec, count) tuples, sorted by total_sec descending
        """
        items = [(name, self._times[name], self._counts[name]) for name in self._times.keys()]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:n]
    
    def get_all(self) -> Dict[str, Dict[str, float]]:
        """
        Get all timing data as dict.
        
        Returns
        -------
        Dict[str, Dict[str, float]]
            Dict mapping block names to {"total_sec": float, "count": int}
        """
        return {
            name: {"total_sec": self._times[name], "count": self._counts[name]}
            for name in self._times.keys()
        }
    
    def reset(self) -> None:
        """Reset all timing data."""
        self._times.clear()
        self._counts.clear()


# Context variable for current perf collector
CURRENT_PERF: contextvars.ContextVar[Optional[PerfCollector]] = contextvars.ContextVar(
    "CURRENT_PERF", default=None
)


def set_current_perf(collector: Optional[PerfCollector]) -> contextvars.Token:
    """Set the current performance collector in context. Returns token for reset."""
    return CURRENT_PERF.set(collector)

def reset_current_perf(token: contextvars.Token) -> None:
    """Reset current perf collector to previous value."""
    CURRENT_PERF.reset(token)


def get_current_perf() -> Optional[PerfCollector]:
    """Get the current performance collector from context."""
    return CURRENT_PERF.get(None)


def perf_add(name: str, dt_sec: float) -> None:
    """
    Add time to a named block using current context collector (no-op if None).
    
    Parameters
    ----------
    name : str
        Block name (e.g., "feat.basic_v1.total")
    dt_sec : float
        Time in seconds to add
    """
    collector = get_current_perf()
    if collector is not None:
        collector.add(name, dt_sec)
    # Debug: log once per unique name (first call) to verify it's being called
    # Note: This will only log once per name to avoid spam
    if not hasattr(perf_add, '_logged_names'):
        perf_add._logged_names = set()
    if name not in perf_add._logged_names and collector is None:
        import logging
        log = logging.getLogger(__name__)
        log.debug(f"[PERF_TIMER] perf_add({name}) called but collector is None (context not set)")
        perf_add._logged_names.add(name)


def perf_inc(name: str, n: int = 1) -> None:
    """
    Increment call count for a named block using current context collector (no-op if None).
    
    Parameters
    ----------
    name : str
        Block name
    n : int
        Count to increment (default 1)
    """
    collector = get_current_perf()
    if collector is not None:
        collector.inc(name, n)


# Backward compatibility alias
PerfTimer = PerfCollector

