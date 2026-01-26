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
    """Test TradeJournal key normalization and backward compatibility."""
    
    def setUp(self):
        """Create temporary journal for testing."""
        self.tmpdir = tempfile.mkdtemp()
        self.journal = TradeJournal(Path(self.tmpdir), 'test_run', enabled=True)
    
    def test_key_prefers_uid(self):
        """Test that trade_uid is preferred over trade_id."""
        key = self.journal._key(trade_uid='run1:chunk0:000001:abc123', trade_id='SIM-123')
        self.assertEqual(key, 'run1:chunk0:000001:abc123')
    
    def test_key_legacy_trade_id(self):
        """Test that legacy trade_id is wrapped with LEGACY: prefix."""
        key = self.journal._key(trade_id='SIM-456')
        self.assertTrue(key.startswith('LEGACY:'), f'Expected LEGACY: prefix, got {key}')
        self.assertEqual(key, 'LEGACY:SIM-456')
    
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
            trade_uid='run1:chunk0:000001:abc123',
            trade_id='SIM-123-000001'
        )
        
        # Verify journal was created with trade_uid as key
        key = self.journal._key(trade_uid='run1:chunk0:000001:abc123')
        self.assertIn(key, self.journal._trade_journals)
        trade_journal = self.journal._trade_journals[key]
        self.assertEqual(trade_journal['trade_uid'], 'run1:chunk0:000001:abc123')
        self.assertEqual(trade_journal['trade_id'], 'SIM-123-000001')
    
    def test_log_entry_snapshot_legacy(self):
        """Test log_entry_snapshot with trade_id only (backward compatibility)."""
        self.journal.log_entry_snapshot(
            entry_time='2025-01-01T00:00:00Z',
            instrument='XAUUSD',
            side='long',
            entry_price=2000.0,
            trade_id='SIM-123-000001'  # Old API
        )
        
        # Verify journal was created with LEGACY: prefix
        key = self.journal._key(trade_id='SIM-123-000001')
        self.assertIn(key, self.journal._trade_journals)
        trade_journal = self.journal._trade_journals[key]
        self.assertIsNone(trade_journal['trade_uid'])
        self.assertEqual(trade_journal['trade_id'], 'SIM-123-000001')


if __name__ == '__main__':
    unittest.main()

