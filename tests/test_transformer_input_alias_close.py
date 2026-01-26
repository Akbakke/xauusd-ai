#!/usr/bin/env python3
"""
Unit tests for transformer input CLOSE alias.

Tests that CLOSE is correctly aliased from candles.close when:
- CLOSE is in snap_feature_names
- CLOSE is not in feature_row
- candles has close column
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Any

# Add workspace root to path
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root))


def build_snapshot_tensor_with_alias(
    snap_feature_names: List[str],
    feature_row: pd.Series,
    candles: pd.DataFrame,
) -> tuple[np.ndarray, Dict[str, str]]:
    """
    Simulate transformer input assembly with CLOSE alias.
    
    This is a minimal version of the logic in oanda_demo_runner.py
    that builds snapshot tensor and applies CLOSE alias.
    
    Args:
        snap_feature_names: List of snapshot feature names (may include CLOSE)
        feature_row: Series with feature values (may not have CLOSE)
        candles: DataFrame with candles data (must have close column if CLOSE is needed)
    
    Returns:
        Tuple of (snapshot tensor array, input_aliases_applied dict)
    """
    snap_data = np.zeros(len(snap_feature_names), dtype=np.float32)
    input_aliases_applied = {}
    
    for i, feat_name in enumerate(snap_feature_names):
        if feat_name in feature_row.index:
            snap_data[i] = float(feature_row[feat_name])
        elif feat_name == "CLOSE":
            # DEL 2: CLOSE alias from candles.close
            if "close" in candles.columns:
                close_value = float(candles["close"].iloc[-1])
                snap_data[i] = close_value
                input_aliases_applied["CLOSE"] = "candles.close"
            else:
                raise RuntimeError(
                    "[TRANSFORMER_INPUT_FAIL] CLOSE feature required but candles.close not found. "
                    "CLOSE must be aliased from candles.close, but candles DataFrame is missing 'close' column."
                )
        else:
            # Feature missing - use 0.0 (should not happen if feature_meta is correct)
            snap_data[i] = 0.0
    
    # DEL 2: Fail-fast if CLOSE is required but aliasing failed
    if "CLOSE" in snap_feature_names:
        close_idx = snap_feature_names.index("CLOSE")
        if np.isnan(snap_data[close_idx]) or np.isinf(snap_data[close_idx]):
            raise RuntimeError(
                f"[TRANSFORMER_INPUT_FAIL] CLOSE feature has NaN/Inf value after aliasing. "
                f"This indicates a problem with candles.close or aliasing logic."
            )
    
    return snap_data, input_aliases_applied


def test_close_alias_from_candles():
    """Test that CLOSE is aliased from candles.close when missing in feature_row."""
    snap_feature_names = ["CLOSE", "_v1_atr14"]
    
    # Feature row without CLOSE (as it should be in prebuilt)
    feature_row = pd.Series({
        "_v1_atr14": 0.5,
        # CLOSE is missing (dropped from prebuilt)
    })
    
    # Candles with close column
    candles = pd.DataFrame({
        "close": [123.4, 123.5, 123.6],
        "open": [123.3, 123.4, 123.5],
        "high": [123.5, 123.6, 123.7],
        "low": [123.2, 123.3, 123.4],
    })
    
    snap_data, aliases = build_snapshot_tensor_with_alias(
        snap_feature_names=snap_feature_names,
        feature_row=feature_row,
        candles=candles,
    )
    
    # CLOSE should be aliased from candles.close (last value: 123.6)
    assert snap_data[0] == pytest.approx(123.6, abs=0.01)
    assert snap_data[1] == pytest.approx(0.5, abs=0.01)
    assert aliases["CLOSE"] == "candles.close"


def test_close_alias_fails_if_candles_missing_close():
    """Negative test: candles missing close -> hard-fail."""
    snap_feature_names = ["CLOSE", "_v1_atr14"]
    
    feature_row = pd.Series({
        "_v1_atr14": 0.5,
    })
    
    # Candles without close column
    candles = pd.DataFrame({
        "open": [123.3, 123.4, 123.5],
        "high": [123.5, 123.6, 123.7],
        "low": [123.2, 123.3, 123.4],
        # close is missing
    })
    
    with pytest.raises(RuntimeError, match="TRANSFORMER_INPUT_FAIL.*candles.close not found"):
        build_snapshot_tensor_with_alias(
            snap_feature_names=snap_feature_names,
            feature_row=feature_row,
            candles=candles,
        )


def test_close_alias_fails_if_nan_inf():
    """Negative test: candles.close has NaN/Inf -> hard-fail."""
    snap_feature_names = ["CLOSE", "_v1_atr14"]
    
    feature_row = pd.Series({
        "_v1_atr14": 0.5,
    })
    
    # Candles with NaN in close (last value is NaN)
    candles = pd.DataFrame({
        "close": [123.4, 123.5, np.nan],  # Last value is NaN
        "open": [123.3, 123.4, 123.5],
    })
    
    with pytest.raises(RuntimeError, match="TRANSFORMER_INPUT_FAIL.*NaN/Inf"):
        build_snapshot_tensor_with_alias(
            snap_feature_names=snap_feature_names,
            feature_row=feature_row,
            candles=candles,
        )
    
    # Test with Inf (last value is Inf)
    candles = pd.DataFrame({
        "close": [123.4, 123.5, np.inf],  # Last value is Inf
        "open": [123.3, 123.4, 123.5],
    })
    
    with pytest.raises(RuntimeError, match="TRANSFORMER_INPUT_FAIL.*NaN/Inf"):
        build_snapshot_tensor_with_alias(
            snap_feature_names=snap_feature_names,
            feature_row=feature_row,
            candles=candles,
        )


def test_no_alias_if_close_not_in_feature_list():
    """Test that no alias is applied if CLOSE is not in feature list."""
    snap_feature_names = ["_v1_atr14", "_v1_body_share_1"]
    
    feature_row = pd.Series({
        "_v1_atr14": 0.5,
        "_v1_body_share_1": 0.3,
    })
    
    candles = pd.DataFrame({
        "close": [123.4, 123.5, 123.6],
    })
    
    snap_data, aliases = build_snapshot_tensor_with_alias(
        snap_feature_names=snap_feature_names,
        feature_row=feature_row,
        candles=candles,
    )
    
    # No alias should be applied
    assert len(aliases) == 0
    assert snap_data[0] == pytest.approx(0.5, abs=0.01)
    assert snap_data[1] == pytest.approx(0.3, abs=0.01)


def test_telemetry_serialization():
    """Test that input_aliases_applied can be serialized to JSON."""
    import json
    
    aliases = {"CLOSE": "candles.close"}
    
    # Should serialize without error
    json_str = json.dumps(aliases)
    assert json_str == '{"CLOSE": "candles.close"}'
    
    # Should deserialize
    aliases_loaded = json.loads(json_str)
    assert aliases_loaded == aliases


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
