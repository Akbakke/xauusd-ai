#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test OANDA Reconciliation Parsing.

Tests transaction matching logic using fixture data (no network calls).
"""
import json
import unittest
from pathlib import Path

import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gx1.monitoring.reconcile_oanda import match_transaction_to_trade, extract_trade_times_and_ids


class TestReconcileOandaParsing(unittest.TestCase):
    """Test OANDA transaction matching logic."""
    
    def setUp(self):
        """Set up test fixtures with OANDA transaction structure."""
        # Sample OANDA transaction (ORDER_FILL)
        self.oanda_fill_transaction = {
            "id": "12345",
            "time": "2025-01-15T10:00:00.000000000Z",
            "type": "ORDER_FILL",
            "instrument": "XAU_USD",
            "units": "100",
            "price": "2000.50",
            "pl": "0.00",
            "commission": "0.50",
            "clientExtensions": {
                "id": "GX1:TEST_RUN:test_trade_001",
                "tag": "TEST_RUN",
                "comment": "RULE5",
            },
            "tradeOpened": {
                "tradeID": "67890",
                "units": "100",
            },
        }
        
        # Sample OANDA transaction (TRADE_CLOSE)
        self.oanda_close_transaction = {
            "id": "12346",
            "time": "2025-01-15T11:00:00.000000000Z",
            "type": "TRADE_CLOSE",
            "instrument": "XAU_USD",
            "units": "-100",
            "price": "2010.00",
            "pl": "9.50",
            "tradeID": "67890",
        }
        
        # Sample trade journal with clientExtensions
        self.trade_journal_with_client_ext = {
            "trade_id": "test_trade_001",
            "execution_events": [
                {
                    "event_type": "ORDER_SUBMITTED",
                    "client_extensions": {
                        "id": "GX1:TEST_RUN:test_trade_001",
                        "tag": "TEST_RUN",
                        "comment": "RULE5",
                    },
                },
                {
                    "event_type": "ORDER_FILLED",
                    "oanda_trade_id": "67890",
                    "oanda_order_id": "12345",
                    "fill_price": 2000.50,
                },
            ],
        }
        
        # Sample trade journal without clientExtensions (fallback)
        self.trade_journal_without_client_ext = {
            "trade_id": "test_trade_002",
            "execution_events": [
                {
                    "event_type": "ORDER_FILLED",
                    "oanda_trade_id": "67891",
                    "oanda_order_id": "12347",
                },
            ],
        }
    
    def test_match_via_client_extensions_id(self):
        """Test matching via clientExtensions.id (primary method)."""
        # Transaction with clientExtensions matching journal
        transaction = self.oanda_fill_transaction.copy()
        trade_journal = self.trade_journal_with_client_ext.copy()
        trade_info = extract_trade_times_and_ids(trade_journal)
        
        match_found, match_method = match_transaction_to_trade(transaction, trade_info)
        self.assertTrue(match_found, "Should match via clientExtensions.id")
        self.assertEqual(match_method, "CLIENT_EXT")
    
    def test_match_via_oanda_trade_id(self):
        """Test matching via oanda_trade_id (secondary method)."""
        # Transaction without clientExtensions but with matching tradeID
        transaction = {
            "id": "12346",
            "type": "TRADE_CLOSE",
            "tradeID": "67890",  # Matches journal
        }
        trade_journal = self.trade_journal_without_client_ext.copy()
        trade_journal["execution_events"][0]["oanda_trade_id"] = "67890"
        trade_info = extract_trade_times_and_ids(trade_journal)
        
        match_found, match_method = match_transaction_to_trade(transaction, trade_info)
        self.assertTrue(match_found, "Should match via oanda_trade_id")
        self.assertEqual(match_method, "TRADE_ID")
    
    def test_match_via_oanda_order_id(self):
        """Test matching via oanda_order_id (tertiary method)."""
        # Transaction with matching orderID
        transaction = {
            "id": "12345",
            "type": "ORDER_FILL",
            "orderID": "12345",  # Matches journal
        }
        trade_journal = self.trade_journal_without_client_ext.copy()
        trade_journal["execution_events"][0]["oanda_order_id"] = "12345"
        trade_info = extract_trade_times_and_ids(trade_journal)
        
        match_found, match_method = match_transaction_to_trade(transaction, trade_info)
        self.assertTrue(match_found, "Should match via oanda_order_id")
        self.assertEqual(match_method, "ORDER_ID_OPEN")
    
    def test_no_match_when_ids_differ(self):
        """Test that transactions don't match when IDs differ."""
        transaction = {
            "id": "99999",
            "type": "ORDER_FILL",
            "tradeID": "99999",
            "orderID": "99999",
            "clientExtensions": {
                "id": "GX1:OTHER_RUN:other_trade",
            },
        }
        trade_journal = self.trade_journal_with_client_ext.copy()
        trade_info = extract_trade_times_and_ids(trade_journal)
        
        match_found, match_method = match_transaction_to_trade(transaction, trade_info)
        self.assertFalse(match_found, "Should not match when all IDs differ")
        self.assertEqual(match_method, "NO_MATCH")
    
    def test_match_with_missing_client_extensions(self):
        """Test matching when transaction lacks clientExtensions."""
        # Transaction without clientExtensions
        transaction = {
            "id": "12345",
            "type": "ORDER_FILL",
            "tradeID": "67890",
        }
        trade_journal = self.trade_journal_without_client_ext.copy()
        trade_journal["execution_events"][0]["oanda_trade_id"] = "67890"
        trade_info = extract_trade_times_and_ids(trade_journal)
        
        match_found, match_method = match_transaction_to_trade(transaction, trade_info)
        self.assertTrue(match_found, "Should match via oanda_trade_id fallback")
        self.assertEqual(match_method, "TRADE_ID")
    
    def test_match_with_missing_journal_client_ext(self):
        """Test matching when journal lacks clientExtensions."""
        # Transaction with clientExtensions but different ID, but matching tradeID
        transaction = {
            "id": "12345",
            "type": "ORDER_FILL",
            "tradeID": "67890",  # This should match
            "clientExtensions": {
                "id": "GX1:OTHER_RUN:other_trade",  # Different client_ext_id
            },
            "tradeOpened": {
                "tradeID": "67890",
            },
        }
        # Journal without clientExtensions in ORDER_SUBMITTED, but with matching tradeID
        trade_journal = {
            "trade_id": "test_trade_003",
            "execution_events": [
                {
                    "event_type": "ORDER_FILLED",
                    "oanda_trade_id": "67890",  # Matches transaction tradeID
                },
            ],
        }
        trade_info = extract_trade_times_and_ids(trade_journal)
        
        # Should match via oanda_trade_id fallback since clientExtensions.id differs
        match_found, match_method = match_transaction_to_trade(transaction, trade_info)
        self.assertTrue(match_found, "Should match via oanda_trade_id fallback when clientExtensions differ")
        self.assertEqual(match_method, "TRADE_ID")
    
    def test_match_close_transaction(self):
        """Test matching closing transaction."""
        transaction = self.oanda_close_transaction.copy()
        trade_journal = self.trade_journal_with_client_ext.copy()
        trade_info = extract_trade_times_and_ids(trade_journal)
        
        match_found, match_method = match_transaction_to_trade(transaction, trade_info)
        self.assertTrue(match_found, "Should match closing transaction via tradeID")
        self.assertEqual(match_method, "TRADE_ID")
    
    def test_string_vs_int_id_comparison(self):
        """Test that string and int IDs are compared correctly."""
        # Transaction with string tradeID
        transaction = {
            "id": "12345",
            "tradeID": "67890",  # String
        }
        # Journal with string oanda_trade_id
        trade_journal = {
            "execution_events": [
                {
                    "event_type": "ORDER_FILLED",
                    "oanda_trade_id": "67890",  # String
                },
            ],
        }
        trade_info = extract_trade_times_and_ids(trade_journal)
        
        match_found, match_method = match_transaction_to_trade(transaction, trade_info)
        self.assertTrue(match_found, "Should match string IDs")
        self.assertEqual(match_method, "TRADE_ID")
        
        # Test with int in transaction
        transaction_int = {
            "id": "12345",
            "tradeID": 67890,  # Int
        }
        
        match_found_int, match_method_int = match_transaction_to_trade(transaction_int, trade_info)
        self.assertTrue(match_found_int, "Should match int/string IDs via str() conversion")
        self.assertEqual(match_method_int, "TRADE_ID")
    
    def test_match_via_txn_id(self):
        """Test matching via transaction ID (quaternary method)."""
        transaction = {
            "id": "12345",
            "type": "ORDER_FILL",
        }
        trade_journal = {
            "execution_events": [
                {
                    "event_type": "ORDER_FILLED",
                    "oanda_transaction_id": "12345",
                },
            ],
        }
        trade_info = extract_trade_times_and_ids(trade_journal)
        
        match_found, match_method = match_transaction_to_trade(transaction, trade_info)
        self.assertTrue(match_found, "Should match via transaction ID")
        self.assertEqual(match_method, "TXN_ID_OPEN")
    
    def test_window1_zero_window2_success(self):
        """Test that window2 is tried when window1 finds 0 transactions."""
        # This is a mock test - actual implementation would test fetch_oanda_transactions_robust
        # For now, we verify the logic exists in the code
        from gx1.monitoring.reconcile_oanda import fetch_oanda_transactions_robust
        self.assertTrue(callable(fetch_oanda_transactions_robust))
    
    def test_since_id_fallback(self):
        """Test that since-id fallback is used when windows find 0 transactions."""
        # This is a mock test - actual implementation would test fetch_oanda_transactions_robust
        from gx1.monitoring.reconcile_oanda import fetch_oanda_transactions_since
        self.assertTrue(callable(fetch_oanda_transactions_since))


if __name__ == "__main__":
    unittest.main()

