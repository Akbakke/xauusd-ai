"""Tests for ZERO_TRADES_DIAG structure."""

import tempfile
from pathlib import Path

import pytest

from gx1.execution.zero_trades_diag import write_zero_trades_diag


def test_zero_trades_diag_written_when_zero_trades():
    """ZERO_TRADES_DIAG must be written when n_trades_closed==0."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        result = write_zero_trades_diag(
            chunk_output_dir=out,
            run_id="MODEL_COMPARE_FULLYEAR_BASE28_20260215_151602",
            chunk_idx=0,
            n_trades_closed=0,
            runner=None,
            bars_processed=70217,
            total_bars=70217,
            n_model_calls=70217,
        )
        assert result is not None
        diag_path = out / "ZERO_TRADES_DIAG.json"
        assert diag_path.exists()
        import json
        diag = json.loads(diag_path.read_text())
        assert diag.get("model_id") == "BASE28"
        assert diag.get("run_id") == "MODEL_COMPARE_FULLYEAR_BASE28_20260215_151602"
        assert diag.get("n_trades_closed") == 0
        assert "counts" in diag
        assert diag["counts"]["bars_processed"] == 70217
        assert "reject_reason_histogram" in diag


def test_zero_trades_diag_not_written_when_positive_trades():
    """ZERO_TRADES_DIAG is NOT written when n_trades_closed > 0 (only for 0-trades)."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        result = write_zero_trades_diag(
            chunk_output_dir=out,
            run_id="MODEL_COMPARE_FULLYEAR_BASE28_20260215_151602",
            chunk_idx=0,
            n_trades_closed=8,
            runner=None,
            bars_processed=70217,
            total_bars=70217,
            n_model_calls=70217,
        )
        assert result is None
        assert not (out / "ZERO_TRADES_DIAG.json").exists()
