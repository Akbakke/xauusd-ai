"""Tests for TRUTH 0-trades SSoT: empty trade_outcomes parquet schema."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from gx1.utils.empty_trade_outcomes import (
    TRADE_OUTCOMES_REQUIRED_COLUMNS,
    empty_trade_outcomes_dataframe,
    write_empty_trade_outcomes_parquet,
)


def test_empty_dataframe_has_required_columns():
    """Empty DataFrame must have all required columns."""
    df = empty_trade_outcomes_dataframe()
    assert len(df) == 0
    for col in TRADE_OUTCOMES_REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing required column: {col}"


def test_empty_parquet_roundtrip():
    """Written empty parquet can be read back with same schema."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trade_outcomes_empty.parquet"
        write_empty_trade_outcomes_parquet(path)
        assert path.exists()
        df = pd.read_parquet(path)
        assert len(df) == 0
        for col in TRADE_OUTCOMES_REQUIRED_COLUMNS:
            assert col in df.columns


def test_merge_required_cols_subset():
    """Merge expects trade_uid, entry_time, exit_time, pnl_bps - all in schema."""
    df = empty_trade_outcomes_dataframe()
    merge_required = ["trade_uid", "entry_time", "exit_time", "pnl_bps"]
    for col in merge_required:
        assert col in df.columns
