#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test trade journal functionality.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from gx1.monitoring.trade_journal import (
    TradeJournal,
    EVENT_ENTRY_SIGNAL,
    EVENT_ROUTER_DECISION,
    EVENT_TRADE_CLOSED,
    _mask_account_id,
)


def test_trade_journal_basic():
    """Test basic trade journal functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "test_run"
        run_dir.mkdir(parents=True)
        
        header = {
            "artifacts": {
                "policy": {"sha256": "test_policy_hash"},
                "router_model": {"sha256": "test_router_hash"},
                "feature_manifest": {"sha256": "test_manifest_hash"},
            }
        }
        
        journal = TradeJournal(run_dir, "TEST_RUN", header)
        
        # Log entry signal
        journal.log(
            EVENT_ENTRY_SIGNAL,
            {
                "entry_time": "2025-01-01T00:00:00Z",
                "entry_price": 2650.0,
                "side": "long",
                "entry_model_outputs": {"p_long": 0.7, "p_short": 0.3},
            },
            trade_key={"entry_time": "2025-01-01T00:00:00Z", "entry_price": 2650.0, "side": "long"},
            trade_id="TEST-001",
        )
        
        # Log router decision
        journal.log(
            EVENT_ROUTER_DECISION,
            {
                "router_version": "V3_RANGE",
                "raw_prediction": "RULE6A",
                "final_decision": "RULE5",
            },
        )
        
        # Log trade closed
        journal.log(
            EVENT_TRADE_CLOSED,
            {
                "exit_time": "2025-01-01T01:00:00Z",
                "exit_price": 2655.0,
                "pnl_bps": 50.0,
            },
            trade_key={"entry_time": "2025-01-01T00:00:00Z", "entry_price": 2650.0, "side": "long"},
            trade_id="TEST-001",
        )
        
        journal.close()
        
        # Verify journal file exists
        journal_path = run_dir / "journal" / "trade_journal.jsonl"
        assert journal_path.exists(), "Journal file should exist"
        
        # Read and verify events
        events = []
        with open(journal_path) as f:
            for line in f:
                event = json.loads(line)
                events.append(event)
        
        assert len(events) == 3, f"Expected 3 events, got {len(events)}"
        
        # Verify first event (ENTRY_SIGNAL)
        assert events[0]["event_type"] == EVENT_ENTRY_SIGNAL
        assert events[0]["trade_id"] == "TEST-001"
        assert events[0]["run_tag"] == "TEST_RUN"
        assert events[0]["policy_sha256"] == "test_policy_hash"
        
        # Verify no secrets in payload
        payload_str = json.dumps(events[0]["payload"])
        assert "api_token" not in payload_str.lower()
        assert "api_key" not in payload_str.lower()
        
        print("✅ Trade journal test passed")


def test_account_id_masking():
    """Test account ID masking."""
    assert _mask_account_id("101-004-12345-001") == "101-***-001"
    assert _mask_account_id("101-004-31061417-001") == "101-***-001"
    assert _mask_account_id("") == "MISSING"
    print("✅ Account ID masking test passed")


if __name__ == "__main__":
    test_trade_journal_basic()
    test_account_id_masking()
    print("✅ All tests passed")

