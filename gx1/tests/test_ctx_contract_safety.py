"""
Mini-tests for ctx contract safety: validate_bundle, session mapping, ATR proxy (no wrap), GX1_STRICT_MASK.

How to run tests (deterministic, correct interpreter):

  cd /home/andre2/src/GX1_ENGINE
  /home/andre2/venvs/gx1/bin/python -m pytest gx1/tests/test_ctx_contract_safety.py -v

Do not rely on bare `pytest ...` (wrong interpreter can make entry v10 / signal_bridge_v1 look broken).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[2]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))


def _set_truth_env() -> None:
    os.environ["GX1_TRUTH_MODE"] = "1"


def _clear_truth_env() -> None:
    os.environ.pop("GX1_TRUTH_MODE", None)
    os.environ.pop("GX1_RUN_MODE", None)


def _clear_strict_mask_env() -> None:
    os.environ.pop("GX1_STRICT_MASK", None)


# ---------------------------------------------------------------------------
# GX1_STRICT_MASK=1: wrong mask length raises without TRUTH
# ---------------------------------------------------------------------------


def test_strict_mask_wrong_cont_length_raises():
    """GX1_STRICT_MASK=1 and GX1_TRUTH_MODE off: wrong GX1_CTX_CONT_MASK length → RuntimeError."""
    _clear_truth_env()
    _clear_strict_mask_env()
    os.environ["GX1_STRICT_MASK"] = "1"
    os.environ["GX1_CTX_CONT_MASK"] = "1,1,1"
    os.environ["GX1_CTX_CAT_MASK"] = "1,1,1,1,1,1"
    try:
        from gx1.execution.ctx_feature_mask import get_ctx_feature_masks
        with pytest.raises(RuntimeError, match="CTX_MASK_FAIL.*GX1_CTX_CONT_MASK length.*must be"):
            get_ctx_feature_masks(expected_ctx_cont_dim=4, expected_ctx_cat_dim=6)
    finally:
        _clear_strict_mask_env()
        os.environ.pop("GX1_CTX_CONT_MASK", None)
        os.environ.pop("GX1_CTX_CAT_MASK", None)


def test_strict_mask_cont_prefix_length_ok_pads():
    """GX1_CTX_CONT_MASK with length == expected_ctx_cont_dim (4) is accepted and padded; index 2 can be 0."""
    _clear_truth_env()
    _clear_strict_mask_env()
    os.environ["GX1_STRICT_MASK"] = "1"
    os.environ["GX1_CTX_CONT_MASK"] = "1,1,0,1"
    os.environ["GX1_CTX_CAT_MASK"] = "1,1,1,1,1"
    try:
        from gx1.execution.ctx_feature_mask import get_ctx_feature_masks
        cont_mask, cat_mask, cont_id, cat_id = get_ctx_feature_masks(expected_ctx_cont_dim=4, expected_ctx_cat_dim=5)
        assert cont_mask.shape == (4,)
        assert cont_mask[2] == 0.0
        assert cont_mask[0] == 1.0 and cont_mask[1] == 1.0
        assert isinstance(cont_id, str) and len(cont_id) == 16
    finally:
        _clear_strict_mask_env()
        os.environ.pop("GX1_CTX_CONT_MASK", None)
        os.environ.pop("GX1_CTX_CAT_MASK", None)


def test_apply_ctx_cont_mask_zeroes_masked_dim():
    """Mask actually affects output: 0 at D1_dist (index 2) zeros that dimension (sikker kontroll E2E)."""
    _clear_truth_env()
    os.environ["GX1_STRICT_MASK"] = "1"
    try:
        from gx1.execution.ctx_feature_mask import apply_ctx_cont_mask
        ctx_cont = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=np.float32)
        cont_mask = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)  # 0 at index 2 = D1_dist
        out = apply_ctx_cont_mask(ctx_cont, cont_mask)
        expected = np.array([10.0, 20.0, 0.0, 40.0, 50.0, 60.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(out, expected)
    finally:
        _clear_strict_mask_env()


# ---------------------------------------------------------------------------
# validate_bundle_ctx_contract_in_strict
# ---------------------------------------------------------------------------


def test_validate_bundle_ctx_contract_dim4_correct_prefix_ok():
    """dim=4 + correct prefix → no error (under TRUTH)."""
    from gx1.contracts.signal_bridge_v1 import (
        ORDERED_CTX_CAT_NAMES_EXTENDED,
        ORDERED_CTX_CONT_NAMES_EXTENDED,
        validate_bundle_ctx_contract_in_strict,
    )
    _set_truth_env()
    try:
        validate_bundle_ctx_contract_in_strict(
            expected_ctx_cont_dim=4,
            expected_ctx_cat_dim=5,
            ordered_ctx_cont_names=ORDERED_CTX_CONT_NAMES_EXTENDED[:4],
            ordered_ctx_cat_names=ORDERED_CTX_CAT_NAMES_EXTENDED[:5],
            context="test",
        )
    finally:
        _clear_truth_env()


def test_validate_bundle_ctx_contract_dim7_raises():
    """dim=7 (not in allowed) → RuntimeError under TRUTH."""
    from gx1.contracts.signal_bridge_v1 import (
        ORDERED_CTX_CAT_NAMES_EXTENDED,
        ORDERED_CTX_CONT_NAMES_EXTENDED,
        validate_bundle_ctx_contract_in_strict,
    )
    _set_truth_env()
    try:
        with pytest.raises(RuntimeError, match="expected_ctx_cont_dim=7 not in"):
            validate_bundle_ctx_contract_in_strict(
                expected_ctx_cont_dim=7,
                expected_ctx_cat_dim=5,
                ordered_ctx_cont_names=ORDERED_CTX_CONT_NAMES_EXTENDED[:7],
                ordered_ctx_cat_names=ORDERED_CTX_CAT_NAMES_EXTENDED[:5],
                context="test",
            )
    finally:
        _clear_truth_env()


def test_validate_bundle_baseline_order_checked():
    """Baseline dim (2/5) is also prefix-validated: wrong order (e.g. spread_bps, atr_bps) → RuntimeError."""
    from gx1.contracts.signal_bridge_v1 import (
        ORDERED_CTX_CAT_NAMES_EXTENDED,
        validate_bundle_ctx_contract_in_strict,
    )
    _set_truth_env()
    try:
        with pytest.raises(RuntimeError, match="ordered_ctx_cont_names does not match"):
            validate_bundle_ctx_contract_in_strict(
                expected_ctx_cont_dim=2,
                expected_ctx_cat_dim=5,
                ordered_ctx_cont_names=["spread_bps", "atr_bps"],
                ordered_ctx_cat_names=ORDERED_CTX_CAT_NAMES_EXTENDED[:5],
                context="test",
            )
    finally:
        _clear_truth_env()


def test_validate_bundle_meta_cont_too_short_raises():
    """Meta ordered_ctx_cont_names shorter than expected_ctx_cont_dim → RuntimeError with meta_cont_len."""
    from gx1.contracts.signal_bridge_v1 import (
        ORDERED_CTX_CAT_NAMES_EXTENDED,
        validate_bundle_ctx_contract_in_strict,
    )
    _set_truth_env()
    try:
        with pytest.raises(RuntimeError, match="meta_cont_len=2 expected_ctx_cont_dim=4"):
            validate_bundle_ctx_contract_in_strict(
                expected_ctx_cont_dim=4,
                expected_ctx_cat_dim=5,
                ordered_ctx_cont_names=["atr_bps", "spread_bps"],
                ordered_ctx_cat_names=ORDERED_CTX_CAT_NAMES_EXTENDED[:5],
                context="test",
            )
    finally:
        _clear_truth_env()


def test_validate_bundle_overlong_meta_cont_raises():
    """TRUTH: ordered_ctx_cont_names longer than contract → RuntimeError."""
    from gx1.contracts.signal_bridge_v1 import (
        ORDERED_CTX_CAT_NAMES_EXTENDED,
        ORDERED_CTX_CONT_NAMES_EXTENDED,
        validate_bundle_ctx_contract_in_strict,
    )
    _set_truth_env()
    try:
        with pytest.raises(RuntimeError, match="ordered_ctx_cont_names longer than contract"):
            validate_bundle_ctx_contract_in_strict(
                expected_ctx_cont_dim=4,
                expected_ctx_cat_dim=5,
                ordered_ctx_cont_names=list(ORDERED_CTX_CONT_NAMES_EXTENDED) + ["EXTRA_BAD"],
                ordered_ctx_cat_names=ORDERED_CTX_CAT_NAMES_EXTENDED[:5],
                context="test",
            )
    finally:
        _clear_truth_env()


def test_validate_bundle_overlong_meta_cat_raises():
    """TRUTH: ordered_ctx_cat_names longer than contract → RuntimeError."""
    from gx1.contracts.signal_bridge_v1 import (
        ORDERED_CTX_CAT_NAMES_EXTENDED,
        ORDERED_CTX_CONT_NAMES_EXTENDED,
        validate_bundle_ctx_contract_in_strict,
    )
    _set_truth_env()
    try:
        with pytest.raises(RuntimeError, match="ordered_ctx_cat_names longer than contract"):
            validate_bundle_ctx_contract_in_strict(
                expected_ctx_cont_dim=2,
                expected_ctx_cat_dim=5,
                ordered_ctx_cont_names=ORDERED_CTX_CONT_NAMES_EXTENDED[:2],
                ordered_ctx_cat_names=list(ORDERED_CTX_CAT_NAMES_EXTENDED) + ["EXTRA_BAD"],
                context="test",
            )
    finally:
        _clear_truth_env()


def test_validate_seq_signal_dtype_guard_truth_raises_on_int():
    """TRUTH: seq_x with non-float dtype (e.g. int64) → RuntimeError invalid dtype."""
    from gx1.contracts.signal_bridge_v1 import SEQ_SIGNAL_DIM, validate_seq_signal
    _set_truth_env()
    try:
        seq_x = np.zeros((1, 2, SEQ_SIGNAL_DIM), dtype=np.int64)
        with pytest.raises(RuntimeError, match="invalid dtype"):
            validate_seq_signal(seq_x, context="test")
    finally:
        _clear_truth_env()


def test_validate_snap_signal_dtype_guard_truth_raises_on_object():
    """TRUTH: snap_x with non-float dtype (e.g. object) → RuntimeError invalid dtype."""
    from gx1.contracts.signal_bridge_v1 import SNAP_SIGNAL_DIM, validate_snap_signal
    _set_truth_env()
    try:
        snap_x = np.zeros((1, SNAP_SIGNAL_DIM), dtype=object)
        with pytest.raises(RuntimeError, match="invalid dtype"):
            validate_snap_signal(snap_x, context="test")
    finally:
        _clear_truth_env()


# ---------------------------------------------------------------------------
# session mapping: unknown session in TRUTH → RuntimeError
# ---------------------------------------------------------------------------


def test_unknown_session_in_truth_raises():
    """Unknown session tag with GX1_TRUTH_MODE=1 → RuntimeError."""
    from gx1.execution.entry_context_features import build_entry_context_features
    candles = pd.DataFrame(
        {"high": [101], "low": [99], "close": [100], "bid_close": [99.5], "ask_close": [100.5]},
        index=pd.DatetimeIndex([pd.Timestamp("2025-06-15 12:00:00", tz="UTC")]),
    )
    policy_state = {"session": "UNKNOWN"}
    _set_truth_env()
    try:
        with pytest.raises(RuntimeError, match="\\[CTX_CAT_FAIL\\].*unknown session tag"):
            build_entry_context_features(
                candles=candles,
                policy_state=policy_state,
                spread_bps=10.0,
                is_replay=True,
            )
    finally:
        _clear_truth_env()


# ---------------------------------------------------------------------------
# ATR proxy: first TR must not use wrap-around (TR[0] == high - low)
# ---------------------------------------------------------------------------


def test_atr_proxy_first_tr_no_wraparound():
    """Linear-rising close window: first bar TR must be high-low, not |high - last_close|."""
    # Build same TR logic as entry_context_features._compute_cheap_atr_proxy
    window = 14
    close_arr = np.linspace(10.0, 10.0 + (window - 1), window, dtype=np.float64)
    high_arr = close_arr + 0.5
    low_arr = close_arr - 0.5
    # tr1 = 1.0 for all bars. If we used np.roll(close, 1), first bar would get prev=last close.
    prev_close = np.concatenate([[np.nan], close_arr[:-1]])
    tr1 = high_arr - low_arr
    tr2 = np.abs(high_arr - prev_close)
    tr3 = np.abs(low_arr - prev_close)
    tr = tr1.copy()
    mask = ~np.isnan(prev_close)
    tr[mask] = np.maximum(tr1[mask], np.maximum(tr2[mask], tr3[mask]))
    assert tr[0] == high_arr[0] - low_arr[0], "first TR must be high-low (no prev close)"
    assert np.isnan(prev_close[0])
    # With wrap-around, tr[0] would be max(tr1[0], |high[0]-close[-1]|, |low[0]-close[-1]|) = max(1, 12, 13) = 13
    assert tr[0] == 1.0


def test_atr_proxy_mean_matches_explicit_tr():
    """_compute_cheap_atr_proxy result is finite and matches explicit TR series (same rules) mean."""
    from gx1.execution.entry_context_features import _compute_cheap_atr_proxy
    window = 14
    close = np.linspace(10.0, 10.0 + (window - 1), window)
    high = close + 0.5
    low = close - 0.5
    candles = pd.DataFrame(
        {"high": high, "low": low, "close": close},
        index=pd.DatetimeIndex(pd.date_range("2025-01-01", periods=window, freq="min", tz="UTC")),
    )
    atr = _compute_cheap_atr_proxy(candles, window=window)
    assert atr is not None
    assert np.isfinite(atr)
    # Explicit TR with same rules: prev_close = [nan, close[:-1]], tr = tr1; where mask, max(tr1,tr2,tr3)
    prev_close = np.concatenate([[np.nan], close[:-1]])
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = tr1.copy()
    mask = ~np.isnan(prev_close)
    tr = tr.astype(np.float64)
    tr[mask] = np.maximum(tr1[mask], np.maximum(tr2[mask], tr3[mask]))
    expected_mean = float(np.mean(tr))
    assert atr == pytest.approx(expected_mean, abs=1e-9), "atr must match explicit TR mean"
