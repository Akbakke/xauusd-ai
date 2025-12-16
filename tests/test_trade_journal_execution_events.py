#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Trade Journal Execution Events.

Verifies that ORDER_SUBMITTED, ORDER_FILLED, and ORDER_REJECTED
events are correctly logged in per-trade JSON files.
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


class TestTradeJournalExecutionEvents(unittest.TestCase):
    """Test execution event logging in trade journal."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.journal_dir = self.temp_dir / "journal"
        self.journal_dir.mkdir(parents=True)
        
        # Mock run header
        self.mock_header = {
            "run_tag": "TEST_RUN",
            "artifacts": {
                "policy": {"sha256": "test_policy_hash"},
                "router_model": {"sha256": "test_router_hash"},
                "feature_manifest": {"sha256": "test_manifest_hash"},
            },
        }
        
        # TradeJournal expects run_dir to be the parent, and creates trade_journal/ subdirectory
        self.run_dir = self.temp_dir
        self.journal = TradeJournal(
            run_dir=self.run_dir,
            run_tag="TEST_RUN",
            header=self.mock_header,
        )
        # Ensure journal is enabled
        self.journal.enabled = True
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_order_submitted_logged(self):
        """Test that ORDER_SUBMITTED is logged correctly."""
        trade_id = "test_trade_001"
        
        self.journal.log_order_submitted(
            trade_id=trade_id,
            instrument="XAU_USD",
            side="long",
            units=100,
            order_type="MARKET",
            client_order_id="CLIENT_001",
            client_ext_id="GX1:TEST_RUN:test_trade_001",
            client_ext_tag="TEST_RUN",
            client_ext_comment="RULE5",
            requested_price=2000.0,
            stop_loss_price=1990.0,
            take_profit_price=2010.0,
            oanda_env="practice",
            account_id_masked="101-***-001",
        )
        
        # Force write (journal writes on log, but ensure it's flushed)
        self.journal.close()
        
        # Load trade journal JSON (journal creates trade_journal/trades/ subdirectory)
        trade_json_path = self.run_dir / "trade_journal" / "trades" / f"{trade_id}.json"
        self.assertTrue(trade_json_path.exists(), f"Trade journal JSON should exist at {trade_json_path}")
        
        with open(trade_json_path, "r") as f:
            trade_data = json.load(f)
        
        # Check execution_events exists
        self.assertIn("execution_events", trade_data)
        execution_events = trade_data["execution_events"]
        self.assertEqual(len(execution_events), 1)
        
        # Check ORDER_SUBMITTED event
        event = execution_events[0]
        self.assertEqual(event["event_type"], "ORDER_SUBMITTED")
        self.assertEqual(event["instrument"], "XAU_USD")
        self.assertEqual(event["side"], "long")
        self.assertEqual(event["units"], 100)
        self.assertEqual(event["order_type"], "MARKET")
        self.assertEqual(event["client_order_id"], "CLIENT_001")
        self.assertEqual(event["client_extensions"]["id"], "GX1:TEST_RUN:test_trade_001")
        self.assertEqual(event["client_extensions"]["tag"], "TEST_RUN")
        self.assertEqual(event["client_extensions"]["comment"], "RULE5")
        self.assertEqual(event["requested_price"], 2000.0)
        self.assertEqual(event["stop_loss_price"], 1990.0)
        self.assertEqual(event["take_profit_price"], 2010.0)
        self.assertEqual(event["oanda_env"], "practice")
        self.assertEqual(event["account_id_masked"], "101-***-001")
    
    def test_order_filled_logged(self):
        """Test that ORDER_FILLED is logged correctly."""
        trade_id = "test_trade_002"
        
        self.journal.log_order_filled(
            trade_id=trade_id,
            oanda_order_id="OANDA_ORDER_001",
            oanda_trade_id="OANDA_TRADE_001",
            oanda_transaction_id="OANDA_TXN_001",
            fill_price=2000.5,
            fill_units=100,
            commission=0.5,
            financing=0.0,
            pl=0.0,
            ts_oanda="2025-01-15T10:00:00.000000000Z",
        )
        
        # Force write
        self.journal.close()
        
        # Load trade journal JSON (journal creates trade_journal/trades/ subdirectory)
        trade_json_path = self.run_dir / "trade_journal" / "trades" / f"{trade_id}.json"
        self.assertTrue(trade_json_path.exists(), f"Trade journal JSON should exist at {trade_json_path}")
        
        with open(trade_json_path, "r") as f:
            trade_data = json.load(f)
        
        # Check execution_events
        execution_events = trade_data.get("execution_events", [])
        self.assertEqual(len(execution_events), 1)
        
        # Check ORDER_FILLED event
        event = execution_events[0]
        self.assertEqual(event["event_type"], "ORDER_FILLED")
        self.assertEqual(event["oanda_order_id"], "OANDA_ORDER_001")
        self.assertEqual(event["oanda_trade_id"], "OANDA_TRADE_001")
        self.assertEqual(event["oanda_transaction_id"], "OANDA_TXN_001")
        self.assertEqual(event["fill_price"], 2000.5)
        self.assertEqual(event["fill_units"], 100)
        self.assertEqual(event["commission"], 0.5)
        self.assertEqual(event["financing"], 0.0)
        self.assertEqual(event["pl"], 0.0)
        self.assertEqual(event["ts_oanda"], "2025-01-15T10:00:00.000000000Z")
    
    def test_order_rejected_logged(self):
        """Test that ORDER_REJECTED is logged correctly."""
        trade_id = "test_trade_003"
        
        self.journal.log_order_rejected(
            trade_id=trade_id,
            client_order_id="CLIENT_002",
            status_code=400,
            reject_reason="INSUFFICIENT_MARGIN",
            response_body='{"error": "Insufficient margin"}',
        )
        
        # Force write
        self.journal.close()
        
        # Load trade journal JSON (journal creates trade_journal/trades/ subdirectory)
        trade_json_path = self.run_dir / "trade_journal" / "trades" / f"{trade_id}.json"
        self.assertTrue(trade_json_path.exists(), f"Trade journal JSON should exist at {trade_json_path}")
        
        with open(trade_json_path, "r") as f:
            trade_data = json.load(f)
        
        # Check execution_events
        execution_events = trade_data.get("execution_events", [])
        self.assertEqual(len(execution_events), 1)
        
        # Check ORDER_REJECTED event
        event = execution_events[0]
        self.assertEqual(event["event_type"], "ORDER_REJECTED")
        self.assertEqual(event["client_order_id"], "CLIENT_002")
        self.assertEqual(event["status_code"], 400)
        self.assertEqual(event["reject_reason"], "INSUFFICIENT_MARGIN")
        self.assertIn("response_body", event)
    
    def test_client_ext_id_in_order_submitted(self):
        """Test that client_ext_id is present in ORDER_SUBMITTED."""
        trade_id = "test_trade_004"
        
        client_ext_id = "GX1:TEST_RUN:test_trade_004"
        
        self.journal.log_order_submitted(
            trade_id=trade_id,
            instrument="XAU_USD",
            side="long",
            units=100,
            order_type="MARKET",
            client_ext_id=client_ext_id,
            client_ext_tag="TEST_RUN",
            client_ext_comment="RULE5",
        )
        
        # Force write
        self.journal.close()
        
        # Load trade journal JSON (journal creates trade_journal/trades/ subdirectory)
        trade_json_path = self.run_dir / "trade_journal" / "trades" / f"{trade_id}.json"
        self.assertTrue(trade_json_path.exists(), f"Trade journal JSON should exist at {trade_json_path}")
        with open(trade_json_path, "r") as f:
            trade_data = json.load(f)
        
        # Check client_ext_id exists
        execution_events = trade_data.get("execution_events", [])
        self.assertEqual(len(execution_events), 1)
        
        event = execution_events[0]
        self.assertEqual(event["event_type"], "ORDER_SUBMITTED")
        self.assertIn("client_extensions", event)
        self.assertEqual(event["client_extensions"]["id"], client_ext_id)
    
    def test_multiple_events_accumulate(self):
        """Test that multiple events accumulate in execution_events array."""
        trade_id = "test_trade_005"
        
        # Log ORDER_SUBMITTED
        self.journal.log_order_submitted(
            trade_id=trade_id,
            instrument="XAU_USD",
            side="long",
            units=100,
            order_type="MARKET",
            client_ext_id="GX1:TEST_RUN:test_trade_005",
        )
        
        # Log ORDER_FILLED
        self.journal.log_order_filled(
            trade_id=trade_id,
            oanda_order_id="OANDA_ORDER_002",
            fill_price=2000.0,
        )
        
        # Force write
        self.journal.close()
        
        # Load trade journal JSON (journal creates trade_journal/trades/ subdirectory)
        trade_json_path = self.run_dir / "trade_journal" / "trades" / f"{trade_id}.json"
        self.assertTrue(trade_json_path.exists(), f"Trade journal JSON should exist at {trade_json_path}")
        with open(trade_json_path, "r") as f:
            trade_data = json.load(f)
        
        # Check both events exist
        execution_events = trade_data.get("execution_events", [])
        self.assertEqual(len(execution_events), 2)
        
        event_types = [e["event_type"] for e in execution_events]
        self.assertIn("ORDER_SUBMITTED", event_types)
        self.assertIn("ORDER_FILLED", event_types)


if __name__ == "__main__":
    unittest.main()

