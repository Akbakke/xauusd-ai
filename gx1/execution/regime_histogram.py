"""
DEL 1: Regime histogram tracking for replay.

Tracks regime distribution per session during replay to understand calibration coverage needs.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict

log = logging.getLogger(__name__)


class RegimeHistogram:
    """Track regime distribution per session during replay."""
    
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.total_calls = 0
    
    def record(self, session: str, regime_bucket: str):
        """Record a regime occurrence."""
        self.counts[session][regime_bucket] += 1
        self.total_calls += 1
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "total_calls": self.total_calls,
            "by_session": {
                session: dict(regime_counts)
                for session, regime_counts in self.counts.items()
            },
            "summary": {
                session: {
                    "total": sum(regime_counts.values()),
                    "regimes": dict(regime_counts)
                }
                for session, regime_counts in self.counts.items()
            }
        }
    
    def save(self, output_dir: Path):
        """Save histogram to JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"CALIB_REGIME_HIST_{self.run_id}.json"
        
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        log.info(f"[REGIME_HIST] Saved regime histogram to {output_path}")
        return output_path


# Global instance (will be initialized in replay)
_regime_histogram: RegimeHistogram = None


def init_regime_histogram(run_id: str) -> RegimeHistogram:
    """Initialize global regime histogram."""
    global _regime_histogram
    _regime_histogram = RegimeHistogram(run_id)
    return _regime_histogram


def get_regime_histogram() -> RegimeHistogram:
    """Get global regime histogram."""
    return _regime_histogram


def record_regime(session: str, regime_bucket: str):
    """Record a regime occurrence (convenience function)."""
    if _regime_histogram is not None:
        _regime_histogram.record(session, regime_bucket)
