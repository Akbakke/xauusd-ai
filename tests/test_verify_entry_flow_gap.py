#!/usr/bin/env python3
"""
Unit tests for verify_entry_flow_gap.py

Tests pure functions with synthetic data.
"""

import pytest
from gx1.scripts.verify_entry_flow_gap import decide_status, extract_fields

def test_decide_status_exception_caught():
    """Test EXCEPTION_CAUGHT status."""
    payload = {
        "control_flow_counts": {
            "AFTER_SOFT_ELIGIBILITY_PASSED": 10,
            "BEFORE_STAGE0_CHECK": 0,
            "EXCEPTION_IN_SOFT_TO_STAGE0_GAP": 5,
        },
        "exception_gap": {
            "exc_type": "KeyError",
            "exc_msg": "Missing key",
            "line": 1234,
            "ts": "2025-01-08T12:00:00",
        },
    }
    
    status, exit_code, report_data = decide_status(payload)
    
    assert status == "EXCEPTION_CAUGHT"
    assert exit_code == 10
    assert report_data["exception_present"] is True
    assert report_data["exception_gap"]["exc_type"] == "KeyError"

def test_decide_status_stage0_reached_fully_verified():
    """Test STAGE0_REACHED with all criteria met."""
    payload = {
        "control_flow_counts": {
            "AFTER_SOFT_ELIGIBILITY_PASSED": 10,
            "BEFORE_STAGE0_CHECK": 10,
        },
        "transformer_forward_calls": 10,
        "entry_routing_aggregate": {
            "selected_model_counts": {
                "v10_hybrid": 10,
            },
        },
    }
    
    status, exit_code, report_data = decide_status(payload)
    
    assert status == "STAGE0_REACHED"
    assert exit_code == 0
    assert report_data["stage0_count"] == 10
    assert report_data["transformer_forward_calls"] == 10

def test_decide_status_stage0_reached_not_fully_verified():
    """Test STAGE0_REACHED but transformer not called."""
    payload = {
        "control_flow_counts": {
            "AFTER_SOFT_ELIGIBILITY_PASSED": 10,
            "BEFORE_STAGE0_CHECK": 10,
        },
        "transformer_forward_calls": 0,
        "entry_routing_aggregate": {
            "selected_model_counts": {
                "v10_hybrid": 10,
            },
        },
    }
    
    status, exit_code, report_data = decide_status(payload)
    
    assert status == "STAGE0_REACHED_BUT_NOT_FULLY_VERIFIED"
    assert exit_code == 20
    assert "transformer_forward_calls == 0" in str(report_data["hints"])

def test_decide_status_unknown_exit():
    """Test UNKNOWN_EXIT status."""
    payload = {
        "control_flow_counts": {
            "AFTER_SOFT_ELIGIBILITY_PASSED": 10,
            "BEFORE_STAGE0_CHECK": 0,
        },
        "exception_gap": None,
    }
    
    status, exit_code, report_data = decide_status(payload)
    
    assert status == "UNKNOWN_EXIT"
    assert exit_code == 30
    assert report_data["soft_passed_count"] == 10
    assert report_data["stage0_count"] == 0

def test_decide_status_insufficient_evidence():
    """Test INSUFFICIENT_EVIDENCE status."""
    payload = {
        "control_flow_counts": {
            "AFTER_SOFT_ELIGIBILITY_PASSED": 0,
            "BEFORE_STAGE0_CHECK": 0,
        },
    }
    
    status, exit_code, report_data = decide_status(payload)
    
    assert status == "INSUFFICIENT_EVIDENCE"
    assert exit_code == 40
    assert report_data["soft_passed_count"] == 0

def test_decide_status_model_entry_dict():
    """Test model_entry with dict format."""
    payload = {
        "control_flow_counts": {
            "AFTER_SOFT_ELIGIBILITY_PASSED": 10,
            "BEFORE_STAGE0_CHECK": 10,
        },
        "transformer_forward_calls": 10,
        "entry_routing_aggregate": {
            "selected_model_counts": {
                "v10_hybrid": 10,
            },
        },
        "model_entry": {
            "model_attempt_calls": {
                "v10_hybrid": 5,
                "legacy": 2,
            },
            "model_forward_calls": {
                "v10_hybrid": 5,
            },
        },
    }
    
    status, exit_code, report_data = decide_status(payload)
    
    assert status == "STAGE0_REACHED"
    assert report_data["model_attempts"] == 7
    assert report_data["model_forwards"] == 5

def test_decide_status_model_entry_int():
    """Test model_entry with int format."""
    payload = {
        "control_flow_counts": {
            "AFTER_SOFT_ELIGIBILITY_PASSED": 10,
            "BEFORE_STAGE0_CHECK": 10,
        },
        "transformer_forward_calls": 10,
        "entry_routing_aggregate": {
            "selected_model_counts": {
                "v10_hybrid": 10,
            },
        },
        "model_entry": {
            "model_attempt_calls": 7,
            "model_forward_calls": 5,
        },
    }
    
    status, exit_code, report_data = decide_status(payload)
    
    assert status == "STAGE0_REACHED"
    assert report_data["model_attempts"] == 7
    assert report_data["model_forwards"] == 5

def test_decide_status_routing_unknown():
    """Test STAGE0_REACHED with routing data not available."""
    payload = {
        "control_flow_counts": {
            "AFTER_SOFT_ELIGIBILITY_PASSED": 10,
            "BEFORE_STAGE0_CHECK": 10,
        },
        "transformer_forward_calls": 10,
        "entry_routing_aggregate": {},  # Empty - routing unknown
    }
    
    status, exit_code, report_data = decide_status(payload)
    
    assert status == "STAGE0_REACHED"
    assert exit_code == 0  # Should not block if routing data not available
    assert "WARNING" in str(report_data["hints"])

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
