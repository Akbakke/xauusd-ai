#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for policy lock functionality.

Verifies that trading is blocked when policy file is modified on disk.
"""
import tempfile
import time
from pathlib import Path
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_policy_lock():
    """
    Test policy lock:
    1. Start runner with policy file
    2. Modify policy file on disk
    3. Verify that trading is blocked
    4. Verify that log message appears
    """
    from gx1.execution.oanda_demo_runner import GX1DemoRunner
    
    # Create temporary policy file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        policy = {
            "meta": {"role": "TEST"},
            "version": "TEST_POLICY",
            "mode": "REPLAY",
            "instrument": "XAU_USD",
            "timeframe": "M5",
            "warmup_bars": 288,
            "entry_config": "gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID/ENTRY_V9_FARM_V2B_PROD.yaml",
            "exit_config": "gx1/configs/exits/FARM_EXIT_V2_RULES_A_mp68_max5_rule5_cd0.yaml",
            "exit_hybrid": {"enabled": False},
            "logging": {"level": "INFO"},
        }
        yaml.dump(policy, f)
        policy_path = Path(f.name)
    
    try:
        logger.info(f"[TEST] Created test policy: {policy_path}")
        
        # Initialize runner
        runner = GX1DemoRunner(
            policy_path,
            dry_run_override=True,
            replay_mode=True,
            fast_replay=True,
        )
        
        logger.info(f"[TEST] Runner initialized with policy_hash: {runner.policy_hash}")
        
        # Verify policy lock passes initially
        assert runner._check_policy_lock(), "Policy lock should pass initially"
        logger.info("[TEST] ✅ Policy lock passes initially")
        
        # Modify policy file
        policy["version"] = "TEST_POLICY_MODIFIED"
        with open(policy_path, 'w') as f:
            yaml.dump(policy, f)
        
        # Small delay to ensure file system updates
        time.sleep(0.1)
        
        logger.info("[TEST] Modified policy file")
        
        # Verify policy lock fails
        assert not runner._check_policy_lock(), "Policy lock should fail after modification"
        logger.info("[TEST] ✅ Policy lock correctly blocks after modification")
        
        # Verify log message (check last log entries)
        # Note: In a real test, you'd capture log output
        logger.info("[TEST] ✅ Policy lock test passed")
        
        return True
        
    except Exception as e:
        logger.error(f"[TEST] Policy lock test failed: {e}", exc_info=True)
        return False
        
    finally:
        # Cleanup
        if policy_path.exists():
            policy_path.unlink()
            logger.info(f"[TEST] Cleaned up test policy: {policy_path}")


if __name__ == "__main__":
    success = test_policy_lock()
    exit(0 if success else 1)

