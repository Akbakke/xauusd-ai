#!/usr/bin/env python3
"""
Unit tests for exit router canonical path enforcement.

Tests:
1. Non-canonical path → RuntimeError
2. Canonical path → OK
3. GX1_ALLOW_NON_CANONICAL_EXIT_ROUTER=1 → WARNING but allows
"""

import os
import sys
import tempfile
from pathlib import Path
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_exit_router_non_canonical_path_fails():
    """Test that loading from non-canonical path raises RuntimeError."""
    from gx1.core.hybrid_exit_router import ExitRouterContext, hybrid_exit_router_v3
    
    # Ensure env var is not set
    if "GX1_ALLOW_NON_CANONICAL_EXIT_ROUTER" in os.environ:
        del os.environ["GX1_ALLOW_NON_CANONICAL_EXIT_ROUTER"]
    
    # Create a non-canonical path
    with tempfile.TemporaryDirectory() as tmpdir:
        non_canonical_path = Path(tmpdir) / "exit_router_v3_tree.pkl"
        non_canonical_path.touch()  # Create empty file
        
        ctx = ExitRouterContext(
            atr_pct=10.0,
            spread_pct=0.5,
            atr_bucket="LOW",
            regime="TREND",
            session="EU",
            model_path=str(non_canonical_path),
        )
        
        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="EXIT_ROUTER_NON_CANONICAL_PATH"):
            hybrid_exit_router_v3(ctx)


def test_exit_router_canonical_path_ok():
    """Test that loading from canonical path works."""
    from gx1.core.hybrid_exit_router import ExitRouterContext, hybrid_exit_router_v3
    
    # Ensure env var is not set
    if "GX1_ALLOW_NON_CANONICAL_EXIT_ROUTER" in os.environ:
        del os.environ["GX1_ALLOW_NON_CANONICAL_EXIT_ROUTER"]
    
    # Use canonical path (should exist after canonicalization)
    project_root = Path(__file__).parent.parent.parent
    canonical_path = project_root / "gx1" / "models" / "exit_router" / "exit_router_v3_tree.pkl"
    
    if not canonical_path.exists():
        pytest.skip(f"Canonical exit router model not found: {canonical_path}")
    
    ctx = ExitRouterContext(
        atr_pct=10.0,
        spread_pct=0.5,
        atr_bucket="LOW",
        regime="TREND",
        session="EU",
        model_path=str(canonical_path),
    )
    
    # Should not raise
    result = hybrid_exit_router_v3(ctx)
    assert result in ["RULE5", "RULE6A"]


def test_exit_router_allow_non_canonical_with_flag():
    """Test that GX1_ALLOW_NON_CANONICAL_EXIT_ROUTER=1 allows non-canonical paths."""
    from gx1.core.hybrid_exit_router import ExitRouterContext, hybrid_exit_router_v3
    import logging
    
    # Set env var
    os.environ["GX1_ALLOW_NON_CANONICAL_EXIT_ROUTER"] = "1"
    
    # Create a non-canonical path
    with tempfile.TemporaryDirectory() as tmpdir:
        non_canonical_path = Path(tmpdir) / "exit_router_v3_tree.pkl"
        non_canonical_path.touch()  # Create empty file
        
        ctx = ExitRouterContext(
            atr_pct=10.0,
            spread_pct=0.5,
            atr_bucket="LOW",
            regime="TREND",
            session="EU",
            model_path=str(non_canonical_path),
        )
        
        # Should not raise (but will fail to load model, which is OK for this test)
        # The important thing is that the canonical check passes
        try:
            result = hybrid_exit_router_v3(ctx)
            # If it gets here, canonical check passed (model load may fail, but that's OK)
        except (FileNotFoundError, ValueError, KeyError):
            # Model load failed, but canonical check passed - this is expected
            pass
    
    # Clean up
    if "GX1_ALLOW_NON_CANONICAL_EXIT_ROUTER" in os.environ:
        del os.environ["GX1_ALLOW_NON_CANONICAL_EXIT_ROUTER"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
