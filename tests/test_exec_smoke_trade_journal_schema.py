#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Execution Smoke Test Trade Journal Schema.

Verifies that a "test trade" JSON has test_mode=true, execution_events
are appended in correct order, and client_ext_id format is correct.
"""
import tempfile
import unittest
from pathlib import Path

import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gx1.monitoring.trade_journal import TradeJournal
from tests.model_native_sizing_support import (
    model_native_runtime_evidence_fixture,
)
from gx1.contracts.entry_model_native_runtime_evidence_v1 import (
    MODEL_NATIVE_RUNTIME_POLICY,
)


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
    
    def test_execution_smoke_snapshot_has_no_direction_substitute(self):
        """A broker smoke record may be explicit, but cannot fake model evidence."""
        self.trade_journal.log_entry_snapshot(
            trade_id=self.trade_id,
            entry_time="2025-01-01T00:00:00Z",
            instrument="XAU_USD",
            side="long",
            entry_price=2000.0,
            model_evidence={},
        )

        journal_data = self.trade_journal._get_trade_journal(self.trade_id)
        entry = journal_data["entry_snapshot"]
        self.assertEqual(entry["model_evidence"], {})
        self.assertNotIn("entry_score", entry)
        self.assertNotIn("entry_filters_passed", entry)
        self.assertNotIn("test_mode", entry)
    
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

    def test_journal_rejects_unadopted_model_native_sizing_evidence(self):
        """Complete Entry evidence cannot invent executable sizing authority."""
        evidence = model_native_runtime_evidence_fixture(
            timestamp="2026-07-08T17:55:00Z",
            entry_action_q_bps=(2.0, 0.2, -1.0),
            position_size_logit=0.4,
            session="US",
            include_execution_timing=True,
        )
        with self.assertRaisesRegex(RuntimeError, "TRADE_JOURNAL_ENTRY"):
            self.trade_journal.log_entry_snapshot(
                trade_id=self.trade_id,
                entry_time="2026-07-08T18:00:00Z",
                instrument="XAU_USD",
                side="long",
                entry_price=2360.2,
                model_evidence=evidence,
                entry_bid=2360.0,
                entry_ask=2360.2,
                entry_spread_bps=0.85,
                session="US",
                model_policy=MODEL_NATIVE_RUNTIME_POLICY,
                execution_checks=["fresh_quote", "learned_sizing_proof_bound"],
                capacity_units=1,
                reference_pre_round_units=1.0,
                pre_round_units=1.0,
                units=1,
                applied_size_multiplier=1.0,
                sizing_application={},
                atr_bps=float(evidence["atr_bps"]),
            )
