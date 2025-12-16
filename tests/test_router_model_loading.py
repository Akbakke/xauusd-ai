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
from sklearn.tree import DecisionTreeClassifier
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_router_model_loading():
    """
    Test router model loading:
    1. Create test model file
    2. Test loading in PROD_BASELINE mode (should succeed)
    3. Test loading with missing model in PROD_BASELINE mode (should fail)
    4. Test loading with missing model in dev mode (should fallback)
    """
    from gx1.core.hybrid_exit_router import ExitRouterContext, hybrid_exit_router_v3
    
    # Create temporary model file
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        # Create minimal test model
        model = DecisionTreeClassifier(max_depth=2)
        X = np.array([[50.0, 30.0, 0.5, 1.0, 0.0, 0.0]])
        y = np.array(["RULE5"])
        model.fit(X, y)
        joblib.dump(model, f.name)
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
        try:
            result = hybrid_exit_router_v3(ctx_prod)
            logger.info(f"[TEST] ✅ PROD_BASELINE mode succeeded: {result}")
        except Exception as e:
            logger.error(f"[TEST] ❌ PROD_BASELINE mode failed unexpectedly: {e}")
            return False
        
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
        try:
            result = hybrid_exit_router_v3(ctx_prod_missing)
            logger.error("[TEST] ❌ PROD_BASELINE mode should have failed but didn't")
            return False
        except (FileNotFoundError, RuntimeError) as e:
            logger.info(f"[TEST] ✅ PROD_BASELINE mode correctly failed closed: {e}")
        except Exception as e:
            logger.error(f"[TEST] ❌ PROD_BASELINE mode failed with wrong exception: {e}")
            return False
        
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
        try:
            result = hybrid_exit_router_v3(ctx_dev)
            logger.info(f"[TEST] ✅ Dev mode correctly fell back to hardcoded logic: {result}")
        except Exception as e:
            logger.error(f"[TEST] ❌ Dev mode should have fallen back but raised exception: {e}")
            return False
        
        logger.info("[TEST] ✅ All router model loading tests passed")
        return True
        
    except Exception as e:
        logger.error(f"[TEST] Router model loading test failed: {e}", exc_info=True)
        return False
        
    finally:
        # Cleanup
        if model_path.exists():
            model_path.unlink()
            logger.info(f"[TEST] Cleaned up test model: {model_path}")


if __name__ == "__main__":
    success = test_router_model_loading()
    exit(0 if success else 1)

