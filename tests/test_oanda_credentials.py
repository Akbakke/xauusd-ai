#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test OANDA credentials loading.

Verifies that credentials are loaded from environment variables correctly
and that PROD_BASELINE mode fails closed on missing credentials.
"""
import os
import tempfile
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_oanda_credentials_dev_mode():
    """
    Test credentials loading in dev mode (should allow missing credentials).
    """
    from gx1.execution.oanda_credentials import load_oanda_credentials
    
    # Clear env vars
    old_env = os.environ.pop("OANDA_ENV", None)
    old_token = os.environ.pop("OANDA_API_TOKEN", None)
    old_account = os.environ.pop("OANDA_ACCOUNT_ID", None)
    
    try:
        # Test dev mode (prod_baseline=False)
        logger.info("[TEST] Testing dev mode (missing credentials)...")
        try:
            creds = load_oanda_credentials(prod_baseline=False)
            logger.info("[TEST] ✅ Dev mode allows missing credentials")
            assert creds.env == "practice", "Should default to practice"
        except ValueError:
            logger.error("[TEST] ❌ Dev mode should not raise on missing credentials")
            return False
        
        # Test with valid env vars
        os.environ["OANDA_ENV"] = "practice"
        os.environ["OANDA_API_TOKEN"] = "test_token_12345"
        os.environ["OANDA_ACCOUNT_ID"] = "101-004-12345-001"
        
        logger.info("[TEST] Testing dev mode (valid credentials)...")
        creds = load_oanda_credentials(prod_baseline=False)
        assert creds.env == "practice"
        assert creds.api_token == "test_token_12345"
        assert creds.account_id == "101-004-12345-001"
        assert creds.api_url == "https://api-fxpractice.oanda.com"
        assert creds.stream_url == "https://stream-fxpractice.oanda.com"
        logger.info("[TEST] ✅ Dev mode loads credentials correctly")
        
        # Test live environment
        os.environ["OANDA_ENV"] = "live"
        creds = load_oanda_credentials(prod_baseline=False)
        assert creds.api_url == "https://api-fxtrade.oanda.com"
        assert creds.stream_url == "https://stream-fxtrade.oanda.com"
        logger.info("[TEST] ✅ Live environment URLs correct")
        
        return True
        
    finally:
        # Restore env vars
        if old_env:
            os.environ["OANDA_ENV"] = old_env
        if old_token:
            os.environ["OANDA_API_TOKEN"] = old_token
        if old_account:
            os.environ["OANDA_ACCOUNT_ID"] = old_account


def test_oanda_credentials_prod_baseline():
    """
    Test credentials loading in PROD_BASELINE mode (should fail closed).
    """
    from gx1.execution.oanda_credentials import load_oanda_credentials
    
    # Clear env vars
    old_env = os.environ.pop("OANDA_ENV", None)
    old_token = os.environ.pop("OANDA_API_TOKEN", None)
    old_account = os.environ.pop("OANDA_ACCOUNT_ID", None)
    
    try:
        # Test PROD_BASELINE mode with missing credentials
        logger.info("[TEST] Testing PROD_BASELINE mode (missing credentials)...")
        try:
            creds = load_oanda_credentials(prod_baseline=True)
            logger.error("[TEST] ❌ PROD_BASELINE should raise on missing credentials")
            return False
        except ValueError as e:
            logger.info(f"[TEST] ✅ PROD_BASELINE correctly raises ValueError: {e}")
        
        # Test PROD_BASELINE mode with valid credentials
        os.environ["OANDA_ENV"] = "practice"
        os.environ["OANDA_API_TOKEN"] = "test_token_12345"
        os.environ["OANDA_ACCOUNT_ID"] = "101-004-12345-001"
        
        logger.info("[TEST] Testing PROD_BASELINE mode (valid credentials)...")
        creds = load_oanda_credentials(prod_baseline=True)
        assert creds.env == "practice"
        assert creds.api_token == "test_token_12345"
        assert creds.account_id == "101-004-12345-001"
        logger.info("[TEST] ✅ PROD_BASELINE loads credentials correctly")
        
        return True
        
    finally:
        # Restore env vars
        if old_env:
            os.environ["OANDA_ENV"] = old_env
        if old_token:
            os.environ["OANDA_API_TOKEN"] = old_token
        if old_account:
            os.environ["OANDA_ACCOUNT_ID"] = old_account


def test_oanda_credentials_canary_mode():
    """
    Test that runner can start in CANARY mode with dummy credentials.
    """
    import yaml
    from gx1.execution.oanda_demo_runner import GX1DemoRunner
    
    # Set dummy credentials
    os.environ["OANDA_ENV"] = "practice"
    os.environ["OANDA_API_TOKEN"] = "dummy_token_for_canary_test"
    os.environ["OANDA_ACCOUNT_ID"] = "101-004-00000-001"
    
    # Create temporary canary policy
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        policy = {
            "meta": {"role": "PROD_BASELINE"},
            "version": "TEST_CANARY",
            "mode": "CANARY",
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
        logger.info("[TEST] Testing CANARY mode runner initialization...")
        runner = GX1DemoRunner(
            policy_path,
            dry_run_override=True,
            replay_mode=True,
            fast_replay=True,
        )
        
        # Verify canary mode is set
        assert runner.canary_mode, "Canary mode should be enabled"
        assert runner.exec.dry_run, "Dry run should be enabled in canary mode"
        
        # Verify credentials loaded (even if dummy)
        assert hasattr(runner, "oanda_env"), "OANDA env should be set"
        assert runner.oanda_env == "practice", "OANDA env should be practice"
        
        logger.info("[TEST] ✅ CANARY mode runner initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"[TEST] ❌ CANARY mode test failed: {e}", exc_info=True)
        return False
        
    finally:
        if policy_path.exists():
            policy_path.unlink()


if __name__ == "__main__":
    import sys
    
    success = True
    success &= test_oanda_credentials_dev_mode()
    success &= test_oanda_credentials_prod_baseline()
    success &= test_oanda_credentials_canary_mode()
    
    if success:
        logger.info("=" * 80)
        logger.info("[TEST] ✅ All OANDA credentials tests passed")
        logger.info("=" * 80)
    else:
        logger.error("=" * 80)
        logger.error("[TEST] ❌ Some tests failed")
        logger.error("=" * 80)
    
    sys.exit(0 if success else 1)

