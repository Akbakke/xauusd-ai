#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Execution Smoke Test Client Extensions Format.

Verifies format: GX1:EXEC_SMOKE:<run_tag>:<trade_id>
(Mock OANDA client - no real API calls in tests)
"""
import unittest
from unittest.mock import Mock, patch

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gx1.execution.exec_smoke_test import create_client_ext_id, create_test_trade_id


class TestExecSmokeClientExtensionsFormat(unittest.TestCase):
    """Test execution smoke test client extensions format."""
    
    def test_create_test_trade_id_format(self):
        """Test that test trade ID has correct format."""
        trade_id = create_test_trade_id()
        
        # Verify format: EXEC-SMOKE-<timestamp>-<uuid>
        self.assertTrue(trade_id.startswith("EXEC-SMOKE-"))
        parts = trade_id.split("-")
        self.assertGreaterEqual(len(parts), 3)
        
        # Verify timestamp is numeric
        self.assertTrue(parts[2].isdigit())
    
    def test_create_client_ext_id_format(self):
        """Test that client extension ID has correct format."""
        run_tag = "EXEC_SMOKE_TEST_2025"
        trade_id = "EXEC-SMOKE-1234567890-abcdefgh"
        client_ext_id = create_client_ext_id(run_tag, trade_id)
        
        # Verify format: GX1:EXEC_SMOKE:<run_tag>:<trade_id>
        expected = f"GX1:EXEC_SMOKE:{run_tag}:{trade_id}"
        self.assertEqual(client_ext_id, expected)
        
        # Verify components
        parts = client_ext_id.split(":")
        self.assertEqual(len(parts), 4)
        self.assertEqual(parts[0], "GX1")
        self.assertEqual(parts[1], "EXEC_SMOKE")
        self.assertEqual(parts[2], run_tag)
        self.assertEqual(parts[3], trade_id)
    
    def test_client_ext_id_with_different_run_tags(self):
        """Test client extension ID with different run tags."""
        test_cases = [
            ("EXEC_SMOKE_2025", "EXEC-SMOKE-123-abc"),
            ("EXEC_SMOKE_TEST", "EXEC-SMOKE-456-def"),
            ("EXEC_SMOKE_20251216_120000", "EXEC-SMOKE-789-ghi"),
        ]
        
        for run_tag, trade_id in test_cases:
            client_ext_id = create_client_ext_id(run_tag, trade_id)
            self.assertTrue(client_ext_id.startswith("GX1:EXEC_SMOKE:"))
            self.assertIn(run_tag, client_ext_id)
            self.assertIn(trade_id, client_ext_id)
    
    def test_client_ext_id_uniqueness(self):
        """Test that client extension IDs are unique for different trades."""
        run_tag = "EXEC_SMOKE_TEST"
        trade_id_1 = "EXEC-SMOKE-123-abc"
        trade_id_2 = "EXEC-SMOKE-456-def"
        
        client_ext_id_1 = create_client_ext_id(run_tag, trade_id_1)
        client_ext_id_2 = create_client_ext_id(run_tag, trade_id_2)
        
        self.assertNotEqual(client_ext_id_1, client_ext_id_2)
        self.assertIn(trade_id_1, client_ext_id_1)
        self.assertIn(trade_id_2, client_ext_id_2)


if __name__ == "__main__":
    unittest.main()

