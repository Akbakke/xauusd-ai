"""
Test trade journal marks degraded warmup.

Tests that entry_snapshot includes warmup_degraded fields
when trade is created in CANARY mode with cached_bars < warmup_bars.
"""
import pytest
import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gx1.monitoring.trade_journal import TradeJournal


def test_trade_journal_marks_degraded_warmup():
    """Test that trade journal marks degraded warmup in entry_snapshot."""
    # Create temporary run directory
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        trade_journal_dir = run_dir / "trade_journal" / "trades"
        trade_journal_dir.mkdir(parents=True)
        
        # Create trade journal
        journal = TradeJournal(
            run_dir=run_dir,
            run_tag="TEST_DEGRADED_WARMUP",
            header={"test": True},
        )
        
        # Log entry snapshot with degraded warmup
        trade_id = "TEST-TRADE-123"
        journal.log_entry_snapshot(
            trade_id=trade_id,
            entry_time="2025-12-17T10:00:00Z",
            instrument="XAU_USD",
            side="long",
            entry_price=2000.0,
            session="EU",
            regime="MEDIUM",
            entry_model_version="ENTRY_V9",
            entry_score={"p_long": 0.6, "p_short": 0.4},
            entry_filters_passed=["spread", "atr"],
            entry_filters_blocked=[],
            test_mode=False,
            reason=None,
            warmup_degraded=True,
            cached_bars_at_entry=120,
            warmup_bars_required=288,
        )
        
        # Read trade JSON
        trade_json_path = trade_journal_dir / f"{trade_id}.json"
        assert trade_json_path.exists(), "Trade JSON should exist"
        
        with open(trade_json_path, 'r') as f:
            trade_data = json.load(f)
        
        # Verify degraded warmup fields
        entry_snapshot = trade_data.get("entry_snapshot", {})
        assert entry_snapshot.get("warmup_degraded") == True, "warmup_degraded should be True"
        assert entry_snapshot.get("cached_bars_at_entry") == 120, "cached_bars_at_entry should be 120"
        assert entry_snapshot.get("warmup_bars_required") == 288, "warmup_bars_required should be 288"


def test_trade_journal_no_degraded_warmup_when_not_degraded():
    """Test that trade journal does NOT mark degraded warmup when warmup is complete."""
    # Create temporary run directory
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        trade_journal_dir = run_dir / "trade_journal" / "trades"
        trade_journal_dir.mkdir(parents=True)
        
        # Create trade journal
        journal = TradeJournal(
            run_dir=run_dir,
            run_tag="TEST_NORMAL_WARMUP",
            header={"test": True},
        )
        
        # Log entry snapshot WITHOUT degraded warmup
        trade_id = "TEST-TRADE-456"
        journal.log_entry_snapshot(
            trade_id=trade_id,
            entry_time="2025-12-17T10:00:00Z",
            instrument="XAU_USD",
            side="long",
            entry_price=2000.0,
            session="EU",
            regime="MEDIUM",
            entry_model_version="ENTRY_V9",
            entry_score={"p_long": 0.6, "p_short": 0.4},
            entry_filters_passed=["spread", "atr"],
            entry_filters_blocked=[],
            test_mode=False,
            reason=None,
            warmup_degraded=False,
            cached_bars_at_entry=300,  # >= warmup_bars
            warmup_bars_required=288,
        )
        
        # Read trade JSON
        trade_json_path = trade_journal_dir / f"{trade_id}.json"
        assert trade_json_path.exists(), "Trade JSON should exist"
        
        with open(trade_json_path, 'r') as f:
            trade_data = json.load(f)
        
        # Verify degraded warmup fields are NOT present (or False)
        entry_snapshot = trade_data.get("entry_snapshot", {})
        assert entry_snapshot.get("warmup_degraded", False) == False, "warmup_degraded should be False or absent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

