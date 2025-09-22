"""Data loading utilities for OHLCV data processing."""

import pandas as pd
from typing import Dict, List


def read_ohlcv_csv(path: str) -> pd.DataFrame:
    """Read OHLCV CSV with UTC DatetimeIndex.

    Args:
        path: Path to CSV file

    Returns:
        DataFrame with UTC DatetimeIndex and OHLCV columns
    """
    # TODO: Implement CSV reading with proper datetime parsing
    pass


def resample_to_tf(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Resample OHLCV data to target timeframe.

    Args:
        df: Input DataFrame with OHLCV data
        tf: Target timeframe (e.g., '5T', '1H', '1D')

    Returns:
        Resampled DataFrame
    """
    # TODO: Implement OHLCV resampling logic
    pass


def align_to_base(base_df: pd.DataFrame, mtf_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Align multi-timeframe data to base timeframe without look-ahead.

    Args:
        base_df: Base timeframe DataFrame
        mtf_dfs: Dict of timeframe -> DataFrame

    Returns:
        Aligned DataFrame with all timeframes
    """
    # TODO: Implement alignment logic with forward-fill, no look-ahead
    pass