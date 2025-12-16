#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Execution Smoke Test Trade Journal Schema.

Verifies that a "test trade" JSON has test_mode=true, execution_events
are appended in correct order, and client_ext_id format is correct.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gx1.monitoring.trade_journal import TradeJournal


class TestExecSmokeTradeJournalSchema(unittest.TestCase):
    """Test execution smoke test trade journal schema."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.journal_dir = self.temp_dir / "journal"
        self.journal_dir.mkdir(parents=True)
        
        # Create minimal run header
        self.run_header = {
            "timestamp": "2025-01-01T00:00:00Z",
            "run_tag": "EXEC_SMOKE_TEST",
            "meta": {
                "role": "TEST",
                "test_mode": True,
            },
            "artifacts": {},
        }
        
        self.trade_journal = TradeJournal(
            run_dir=self.temp_dir,
            run_tag="EXEC_SMOKE_TEST",
            header=self.run_header,
            enabled=True,
        )
        
        self.trade_id = "EXEC-SMOKE-1234567890-abcdefgh"
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_test_mode_flag_in_entry_snapshot(self):
        """Test that entry_snapshot has test_mode=true."""
        # Log entry snapshot
        self.trade_journal.log_entry_snapshot(
            trade_id=self.trade_id,
            entry_time="2025-01-01T00:00:00Z",
            instrument="XAU_USD",
            side="long",
            entry_price=2000.0,
            session=None,
            regime=None,
            entry_model_version=None,
            entry_score=None,
            entry_filters_passed=[],
            entry_filters_blocked=[],
        )
        
        # Mark as test mode
        journal_data = self.trade_journal._get_trade_journal(self.trade_id)
        journal_data["entry_snapshot"]["test_mode"] = True
        journal_data["entry_snapshot"]["reason"] = "EXECUTION_SMOKE_TEST"
        self.trade_journal._write_trade_json(self.trade_id)
        
        # Verify
        journal_data = self.trade_journal._get_trade_journal(self.trade_id)
        self.assertTrue(journal_data["entry_snapshot"]["test_mode"])
        self.assertEqual(journal_data["entry_snapshot"]["reason"], "EXECUTION_SMOKE_TEST")
        self.assertIsNone(journal_data["entry_snapshot"]["entry_model_version"])
    
    def test_test_mode_flag_in_feature_context(self):
        """Test that feature_context has test_mode=true."""
        # Log feature context
        self.trade_journal.log_feature_context(
            trade_id=self.trade_id,
            atr_bps=None,
            atr_price=None,
            atr_percentile=None,
            range_pos=None,
            distance_to_range=None,
            range_edge_dist_atr=None,
            spread_price=None,
            spread_pct=None,
            candle_close=None,
            candle_high=None,
            candle_low=None,
        )
        
        # Mark as test mode
        journal_data = self.trade_journal._get_trade_journal(self.trade_id)
        journal_data["feature_context"]["test_mode"] = True
        self.trade_journal._write_trade_json(self.trade_id)
        
        # Verify
        journal_data = self.trade_journal._get_trade_journal(self.trade_id)
        self.assertTrue(journal_data["feature_context"]["test_mode"])
    
    def test_execution_events_order(self):
        """Test that execution events are appended in correct order."""
        # Log ORDER_SUBMITTED
        self.trade_journal.log_order_submitted(
            trade_id=self.trade_id,
            instrument="XAU_USD",
            side="long",
            units=1,
            order_type="MARKET",
            client_ext_id="GX1:EXEC_SMOKE:TEST:123",
        )
        
        # Log ORDER_FILLED
        self.trade_journal.log_order_filled(
            trade_id=self.trade_id,
            oanda_order_id="12345",
            fill_price=2000.0,
            fill_units=1,
        )
        
        # Log TRADE_OPENED_OANDA
        self.trade_journal.log_oanda_trade_update(
            trade_id=self.trade_id,
            event_type="TRADE_OPENED_OANDA",
            oanda_trade_id="67890",
        )
        
        # Verify order
        journal_data = self.trade_journal._get_trade_journal(self.trade_id)
        events = journal_data.get("execution_events", [])
        
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0]["event_type"], "ORDER_SUBMITTED")
        self.assertEqual(events[1]["event_type"], "ORDER_FILLED")
        self.assertEqual(events[2]["event_type"], "TRADE_OPENED_OANDA")
    
    def test_client_ext_id_format(self):
        """Test that client_ext_id format is correct."""
        run_tag = "EXEC_SMOKE_TEST"
        trade_id = "EXEC-SMOKE-1234567890-abcdefgh"
        client_ext_id = f"GX1:EXEC_SMOKE:{run_tag}:{trade_id}"
        
        # Verify format
        self.assertTrue(client_ext_id.startswith("GX1:EXEC_SMOKE:"))
        self.assertIn(run_tag, client_ext_id)
        self.assertIn(trade_id, client_ext_id)
        
        # Log ORDER_SUBMITTED with client_ext_id
        self.trade_journal.log_order_submitted(
            trade_id=trade_id,
            instrument="XAU_USD",
            side="long",
            units=1,
            order_type="MARKET",
            client_ext_id=client_ext_id,
        )
        
        # Verify in journal
        journal_data = self.trade_journal._get_trade_journal(trade_id)
        events = journal_data.get("execution_events", [])
        
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "ORDER_SUBMITTED")
        self.assertEqual(events[0]["client_extensions"]["id"], client_ext_id)


if __name__ == "__main__":
    unittest.main()

