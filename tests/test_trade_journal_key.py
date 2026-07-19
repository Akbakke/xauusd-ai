#!/usr/bin/env python3
"""
Unit tests for TradeJournal key normalization (COMMIT C).
"""
import unittest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from gx1.monitoring.trade_journal import TradeJournal


class TestTradeJournalKey(unittest.TestCase):
    """Test strict TradeJournal identifier normalization."""
    
    def setUp(self):
        """Create temporary journal for testing."""
        self.tmpdir = tempfile.mkdtemp()
        self.journal = TradeJournal(
            Path(self.tmpdir),
            'EXEC_SMOKE_KEY_TEST',
            header={"meta": {"role": "TEST", "test_mode": True}},
            enabled=True,
        )
    
    def test_key_prefers_uid(self):
        """Test that trade_uid is preferred over trade_id."""
        key = self.journal._key(trade_uid='run1:chunk0:000001:abc123', trade_id='SIM-123')
        self.assertEqual(key, 'run1:chunk0:000001:abc123')
    
    def test_key_namespaces_trade_id(self):
        """Broker/display IDs are explicitly namespaced."""
        key = self.journal._key(trade_id='SIM-456')
        self.assertEqual(key, 'TRADE:SIM-456')
    
    def test_key_raises_if_neither(self):
        """Test that _key raises ValueError if neither trade_uid nor trade_id provided."""
        with self.assertRaises(ValueError):
            self.journal._key()
    
    def test_log_entry_snapshot_with_uid(self):
        """Test log_entry_snapshot with trade_uid (new API)."""
        self.journal.log_entry_snapshot(
            entry_time='2025-01-01T00:00:00Z',
            instrument='XAUUSD',
            side='long',
            entry_price=2000.0,
            model_evidence={},
            trade_uid='run1:chunk0:000001:abc123',
            trade_id='SIM-123-000001'
        )
        
        # Verify journal was created with trade_uid as key
        key = self.journal._key(trade_uid='run1:chunk0:000001:abc123')
        self.assertIn(key, self.journal._trade_journals)
        trade_journal = self.journal._trade_journals[key]
        self.assertEqual(trade_journal['trade_uid'], 'run1:chunk0:000001:abc123')
        self.assertEqual(trade_journal['trade_id'], 'SIM-123-000001')
    
    def test_log_entry_snapshot_with_trade_id(self):
        """Test log_entry_snapshot with a broker/display trade ID."""
        self.journal.log_entry_snapshot(
            entry_time='2025-01-01T00:00:00Z',
            instrument='XAUUSD',
            side='long',
            entry_price=2000.0,
            model_evidence={},
            trade_id='SIM-123-000001',
        )

        key = self.journal._key(trade_id='SIM-123-000001')
        self.assertIn(key, self.journal._trade_journals)
        trade_journal = self.journal._trade_journals[key]
        self.assertIsNone(trade_journal['trade_uid'])
        self.assertEqual(trade_journal['trade_id'], 'SIM-123-000001')

    def test_empty_model_evidence_is_rejected_outside_explicit_smoke(self):
        live_journal = TradeJournal(
            Path(self.tmpdir) / "live",
            "LIVE_RUN",
            header={"meta": {"role": "LIVE", "test_mode": False}},
            enabled=True,
        )

        with self.assertRaisesRegex(RuntimeError, "ENTRY_EVIDENCE_FAILED"):
            live_journal.log_entry_snapshot(
                entry_time="2026-07-16T12:00:00Z",
                instrument="XAU_USD",
                side="long",
                entry_price=3300.0,
                model_evidence={},
                trade_id="LIVE-EMPTY-EVIDENCE",
            )


if __name__ == '__main__':
    unittest.main()
