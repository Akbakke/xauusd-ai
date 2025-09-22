"""Feature engineering functions for OHLCV data."""

import pandas as pd
from typing import List


def returns(df: pd.DataFrame, periods: List[int] = [1, 3, 5]) -> pd.DataFrame:
    """Calculate returns for multiple periods.

    Args:
        df: OHLCV DataFrame
        periods: List of periods for return calculation

    Returns:
        DataFrame with return features, shifted by 1 period
    """
    # TODO: Implement return calculations
    pass


def range_shape_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate range and shape features.

    Features: H-L range, |C-O| body, wick ratios

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame with range/shape features, shifted by 1 period
    """
    # TODO: Implement range/shape features
    pass


def volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volatility features.

    Features: ATR(14), RS/Parkinson/GK volatility estimators

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame with volatility features, shifted by 1 period
    """
    # TODO: Implement volatility calculations
    pass


def momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate momentum and mean reversion features.

    Features: ROC(10,20), z_ret(20,60), RSI(14), BB_pos

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame with momentum features, shifted by 1 period
    """
    # TODO: Implement momentum indicators
    pass


def volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volume-based features.

    Features: volume_z (volume z-score)

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame with volume features, shifted by 1 period
    """
    # TODO: Implement volume features
    pass


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all feature sets and combine.

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame with all features, properly shifted
    """
    # TODO: Combine all feature functions
    # TODO: Add tsfresh/catch22/miniROCKET integration
    pass
