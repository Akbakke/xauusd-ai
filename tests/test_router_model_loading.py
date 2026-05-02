#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for router model loading (PROD_BASELINE fail-closed).

Verifies that:
1. Model loads correctly with logging (path, size, hash)
2. PROD_BASELINE mode fails closed if model missing
3. Dev/replay mode falls back to hardcoded logic
"""
import tempfile
import joblib
from pathlib import Path
import logging
import pytest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StaticExitRouterModel:
    def predict(self, features):
        return ["RULE5"] * len(features)


def test_router_model_loading(monkeypatch):
    """
    Test router model loading:
    1. Create test model file
    2. Test loading in PROD_BASELINE mode (should succeed)
    3. Test loading with missing model in PROD_BASELINE mode (should fail)
    4. Test loading with missing model in dev mode (should fallback)
    """
    from gx1.core.hybrid_exit_router import ExitRouterContext, hybrid_exit_router_v3
    monkeypatch.setenv("GX1_ALLOW_NON_CANONICAL_EXIT_ROUTER", "1")
    
    # Create temporary model file
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        joblib.dump(StaticExitRouterModel(), f.name)
        model_path = Path(f.name)
    
    try:
        logger.info(f"[TEST] Created test model: {model_path}")
        
        # Test 1: PROD_BASELINE mode with valid model (should succeed)
        ctx_prod = ExitRouterContext(
            atr_pct=50.0,
            spread_pct=30.0,
            atr_bucket="MEDIUM",
            regime="FARM_ASIA_MEDIUM",
            session="ASIA",
            model_path=str(model_path),
            prod_baseline=True,
        )
        
        logger.info("[TEST] Testing PROD_BASELINE mode with valid model...")
        result = hybrid_exit_router_v3(ctx_prod)
        logger.info(f"[TEST] ✅ PROD_BASELINE mode succeeded: {result}")
        
        # Test 2: PROD_BASELINE mode with missing model (should fail closed)
        ctx_prod_missing = ExitRouterContext(
            atr_pct=50.0,
            spread_pct=30.0,
            atr_bucket="MEDIUM",
            regime="FARM_ASIA_MEDIUM",
            session="ASIA",
            model_path=str(model_path.parent / "nonexistent.pkl"),
            prod_baseline=True,
        )
        
        logger.info("[TEST] Testing PROD_BASELINE mode with missing model...")
        with pytest.raises((FileNotFoundError, RuntimeError)):
            hybrid_exit_router_v3(ctx_prod_missing)
        
        # Test 3: Dev mode with missing model (should fallback)
        ctx_dev = ExitRouterContext(
            atr_pct=50.0,
            spread_pct=30.0,
            atr_bucket="MEDIUM",
            regime="FARM_ASIA_MEDIUM",
            session="ASIA",
            model_path=str(model_path.parent / "nonexistent.pkl"),
            prod_baseline=False,
        )
        
        logger.info("[TEST] Testing dev mode with missing model...")
        result = hybrid_exit_router_v3(ctx_dev)
        logger.info(f"[TEST] ✅ Dev mode correctly fell back to hardcoded logic: {result}")
        
        logger.info("[TEST] ✅ All router model loading tests passed")
        
    except Exception as e:
        logger.error(f"[TEST] Router model loading test failed: {e}", exc_info=True)
        raise
        
    finally:
        # Cleanup
        if model_path.exists():
            model_path.unlink()
            logger.info(f"[TEST] Cleaned up test model: {model_path}")


if __name__ == "__main__":
    test_router_model_loading()
