# gx1/seq/sequence_features.py
"""
Feature engineering for TCN sequence models.
Legger til sekvens-baserte features (trend, volatilitet, momentum, microstructure)
som kan brukes over 864-bar vinduer.
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple


def build_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bygger sekvens-features for TCN-modell.
    
    Input:
      df: DataFrame med OHLCV data (må ha kolonner: open, high, low, close, volume)
          og ts (timestamp) som index eller kolonne.
    
    Output:
      df: DataFrame med tillegg av nye feature-kolonner
    
    FIX: Sanitize OHLCV dtypes before processing (avoids pandas object dtype path)
    
    Features lagt til:
      Trend/retning:
        - ema20_slope: EMA(20) slope (diff per bar)
        - ema100_slope: EMA(100) slope
        - pos_vs_ema200: close / EMA200 - 1
    
      Volatilitet:
        - std50: Rolling std(50)
        - atr50: Rolling ATR(50)
        - atr_z: ATR Z-score (ATR50 / ATR200 - 1)
    
      Momentum:
        - roc20: ROC(20) (rate of change)
        - roc100: ROC(100)
    
      Microstructure:
        - body_pct: (close-open)/(high-low+1e-8)
        - wick_asym: (upper_wick - lower_wick)/(range+1e-8)
    
      Miljøkontekst (broadcastes over hele sekvensen):
        - session_id: int (0=EU, 1=OVERLAP, 2=US)
        - atr_regime_id: int (0=LOW, 1=MID, 2=HIGH, 3=EXTREME)
        - trend_regime_tf24h: float (EMA100 slope over 24h / ATR100, normalisert)
    """
    # Sanitize OHLCV dtypes (ensure float32/float64)
    import os
    is_replay = os.getenv("GX1_REPLAY", "0") == "1"
    ohlcv_cols = ['open', 'high', 'low', 'close']
    for col in ohlcv_cols:
        if col in df.columns:
            if df[col].dtype not in [np.float32, np.float64]:
                if is_replay:
                    raise ValueError(
                        f"OHLCV dtype mismatch (replay): column '{col}' has dtype {df[col].dtype}, "
                        f"expected float32/float64. This may cause pandas timeout."
                    )
                df[col] = df[col].astype(np.float32, copy=False)
    
    df = df.copy()
    
    # Sikre at close, high, low, open, volume finnes
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Mangler kolonner for sequence features: {missing}")
    
    close = df['close'].astype(np.float32)
    high = df['high'].astype(np.float32)
    low = df['low'].astype(np.float32)
    open_price = df['open'].astype(np.float32)
    volume = df['volume'].astype(np.float32) if 'volume' in df.columns else pd.Series(0.0, index=df.index)
    
    # ============================================================
    # Trend / Retning
    # ============================================================
    
    # EMA(20) slope (diff per bar)
    ema20 = close.ewm(span=20, adjust=False).mean()
    df['ema20_slope'] = ema20.diff().fillna(0.0)
    
    # EMA(100) slope
    ema100 = close.ewm(span=100, adjust=False).mean()
    df['ema100_slope'] = ema100.diff().fillna(0.0)
    
    # Long-term position: close / EMA200 - 1
    ema200 = close.ewm(span=200, adjust=False).mean()
    df['pos_vs_ema200'] = (close / ema200) - 1.0
    df['pos_vs_ema200'] = df['pos_vs_ema200'].fillna(0.0)
    
    # ============================================================
    # Volatilitet
    # ============================================================
    
    # Rolling std(50)
    df['std50'] = close.rolling(window=50, min_periods=1).std().fillna(0.0)
    
    # Rolling ATR(50)
    # ATR = True Range rolling average
    # FIX: Replace pandas .max(axis=1) with NumPy for performance (avoids timeout)
    high_arr = high.values if hasattr(high, 'values') else np.asarray(high, dtype=np.float32)
    low_arr = low.values if hasattr(low, 'values') else np.asarray(low, dtype=np.float32)
    close_arr = close.values if hasattr(close, 'values') else np.asarray(close, dtype=np.float32)
    
    # Compute True Range components
    tr1 = high_arr - low_arr  # High - Low
    close_shifted = np.roll(close_arr, 1)
    close_shifted[0] = close_arr[0]  # First bar: no previous close, use current
    tr2 = np.abs(high_arr - close_shifted)  # |High - Previous Close|
    tr3 = np.abs(low_arr - close_shifted)   # |Low - Previous Close|
    
    # Use NumPy nanmax instead of pandas .max(axis=1)
    tr = np.nanmax(np.column_stack([tr1, tr2, tr3]), axis=1)
    tr = pd.Series(tr, index=df.index, dtype=np.float32)
    atr50 = tr.rolling(window=50, min_periods=1).mean()
    df['atr50'] = atr50.fillna(0.0)
    
    # ATR Z-score: ATR50 / ATR200 - 1
    atr200 = tr.rolling(window=200, min_periods=1).mean()
    df['atr_z'] = (atr50 / (atr200 + 1e-8)) - 1.0
    df['atr_z'] = df['atr_z'].fillna(0.0)
    
    # ============================================================
    # Momentum
    # ============================================================
    
    # ROC(20): Rate of Change over 20 bars
    df['roc20'] = ((close / close.shift(20)) - 1.0).fillna(0.0)
    
    # ROC(100): Rate of Change over 100 bars
    df['roc100'] = ((close / close.shift(100)) - 1.0).fillna(0.0)
    
    # ============================================================
    # Microstructure
    # ============================================================
    
    # Candle body percentage: (close-open)/(high-low+1e-8)
    body = close - open_price
    range_size = high - low
    df['body_pct'] = body / (range_size + 1e-8)
    df['body_pct'] = df['body_pct'].fillna(0.0)
    
    # Wick asymmetry: (upper_wick - lower_wick)/(range+1e-8)
    upper_wick = high - pd.concat([open_price, close], axis=1).max(axis=1)
    lower_wick = pd.concat([open_price, close], axis=1).min(axis=1) - low
    df['wick_asym'] = (upper_wick - lower_wick) / (range_size + 1e-8)
    df['wick_asym'] = df['wick_asym'].fillna(0.0)
    
    # ============================================================
    # Miljøkontekst (broadcastes over hele sekvensen)
    # ============================================================
    
    # session_id: 0=EU, 1=OVERLAP, 2=US
    if "session" in df.columns:
        session_map = {"EU": 0, "OVERLAP": 1, "US": 2}
        df["session_id"] = df["session"].map(lambda x: session_map.get(str(x).upper(), 1)).fillna(1).astype(np.int32)
    else:
        # Default to OVERLAP if session not available
        df["session_id"] = 1
    
    # atr_regime_id: 0=LOW, 1=MID, 2=HIGH, 3=EXTREME
    if "_v1_atr_regime_id" in df.columns:
        # _v1_atr_regime_id uses qcut with 3 bins (0, 1, 2)
        # Map to 4 bins: 0->0 (LOW), 1->1 (MID), 2->2 (HIGH), add EXTREME for top 10% of HIGH
        atr_regime_raw = df["_v1_atr_regime_id"].fillna(1.0).astype(np.float32)
        # Use ATR50 to identify EXTREME within HIGH regime (top 10% of HIGH)
        atr50_for_extreme = df['atr50'].values
        atr50_q90 = pd.Series(atr50_for_extreme).rolling(window=288, min_periods=50).quantile(0.90).fillna(pd.Series(atr50_for_extreme).quantile(0.90))
        
        atr_regime = np.zeros(len(df), dtype=np.int32)
        for i in range(len(df)):
            raw_val = float(atr_regime_raw.iloc[i]) if isinstance(atr_regime_raw, pd.Series) else float(atr_regime_raw[i])
            if pd.isna(raw_val):
                atr_regime[i] = 1  # Default to MID
            elif raw_val < 0.5:
                atr_regime[i] = 0  # LOW
            elif raw_val < 1.5:
                atr_regime[i] = 1  # MID
            elif raw_val < 2.5:
                # HIGH regime - check if EXTREME (top 10% of ATR50)
                if not pd.isna(atr50_q90.iloc[i]) and atr50_for_extreme[i] >= atr50_q90.iloc[i]:
                    atr_regime[i] = 3  # EXTREME
                else:
                    atr_regime[i] = 2  # HIGH
            else:
                # raw_val >= 2.5 (shouldn't happen with qcut(3), but handle it)
                atr_regime[i] = 3  # EXTREME
        
        df["atr_regime_id"] = atr_regime
    else:
        # Compute ATR regime from ATR50 percentiles if not available
        # Use rolling percentiles for dynamic regime classification
        atr50_vals = df['atr50'].values
        # Compute percentiles over rolling window (288 bars = 1 day)
        atr50_series = pd.Series(atr50_vals)
        q25_vals = atr50_series.rolling(window=288, min_periods=50).quantile(0.25).fillna(atr50_series.quantile(0.25))
        q50_vals = atr50_series.rolling(window=288, min_periods=50).quantile(0.50).fillna(atr50_series.quantile(0.50))
        q75_vals = atr50_series.rolling(window=288, min_periods=50).quantile(0.75).fillna(atr50_series.quantile(0.75))
        q90_vals = atr50_series.rolling(window=288, min_periods=50).quantile(0.90).fillna(atr50_series.quantile(0.90))
        
        # Classify each bar based on its ATR50 value relative to rolling percentiles
        atr_regime = np.zeros(len(df), dtype=np.int32)
        for i in range(len(df)):
            val = atr50_vals[i]
            if pd.isna(val) or pd.isna(q25_vals.iloc[i]) or pd.isna(q50_vals.iloc[i]) or pd.isna(q75_vals.iloc[i]):
                atr_regime[i] = 1  # Default to MID
            elif val < q25_vals.iloc[i]:
                atr_regime[i] = 0  # LOW
            elif val < q50_vals.iloc[i]:
                atr_regime[i] = 1  # MID
            elif val < q75_vals.iloc[i]:
                atr_regime[i] = 2  # HIGH
            elif not pd.isna(q90_vals.iloc[i]) and val >= q90_vals.iloc[i]:
                atr_regime[i] = 3  # EXTREME
            else:
                atr_regime[i] = 2  # HIGH
        
        df["atr_regime_id"] = atr_regime
    
    # trend_regime_tf24h: EMA100 slope over 24h (288 bars) / ATR100, normalisert
    # 24h = 288 bars (M5)
    # EMA100 slope over 24h = difference between current EMA100 and EMA100 288 bars ago
    ema100_24h_slope = (ema100 - ema100.shift(288)).fillna(0.0)  # Slope over 24h
    atr100 = tr.rolling(window=100, min_periods=1).mean().fillna(tr.mean() if len(tr) > 0 else 1e-8)
    # Normaliser: slope / ATR100 (for å få enhetlig skala)
    df['trend_regime_tf24h'] = (ema100_24h_slope / (atr100 + 1e-8)).fillna(0.0)
    df['trend_regime_tf24h'] = df['trend_regime_tf24h'].replace([np.inf, -np.inf], 0.0)
    
    # ============================================================
    # Handle NaN/Inf
    # ============================================================
    
    # Fyll alle NaN med 0
    seq_feat_cols = [
        'ema20_slope', 'ema100_slope', 'pos_vs_ema200',
        'std50', 'atr50', 'atr_z',
        'roc20', 'roc100',
        'body_pct', 'wick_asym',
        'session_id', 'atr_regime_id', 'trend_regime_tf24h'
    ]
    
    for col in seq_feat_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    
    return df


def get_sequence_feature_names() -> List[str]:
    """
    Returnerer liste over sekvens-feature navn.
    """
    return [
        'ema20_slope',
        'ema100_slope',
        'pos_vs_ema200',
        'std50',
        'atr50',
        'atr_z',
        'roc20',
        'roc100',
        'body_pct',
        'wick_asym',
        'session_id',
        'atr_regime_id',
        'trend_regime_tf24h'
    ]


def build_seq_features_for_v8(df: pd.DataFrame, seq_feat_names: Optional[List[str]] = None) -> np.ndarray:
    """
    Bygger sekvens-features for entry_v8 modell.
    
    Args:
        df: DataFrame med sequence features (fra build_sequence_features)
        seq_feat_names: Liste over feature-navn (default: get_sequence_feature_names())
    
    Returns:
        np.ndarray [T, n_seq_features] med sekvens-features
    """
    if seq_feat_names is None:
        seq_feat_names = get_sequence_feature_names()
    
    # Extract features
    missing = [f for f in seq_feat_names if f not in df.columns]
    if missing:
        raise KeyError(f"Missing sequence features: {missing}")
    
    X_seq = df[seq_feat_names].to_numpy(dtype=np.float32)
    X_seq = np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X_seq


def build_snap_features_for_v8(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    """
    Bygger snapshot/tabular features for entry_v8 modell.
    Samme som prepare_tabular_features, men returnerer numpy array.
    
    Args:
        df: DataFrame med features
        config: Config dict med features settings
    
    Returns:
        Tuple[np.ndarray, List[str]]: (snap_features [N, n_snap_features], feature_names)
    """
    from gx1.tuning.entry_v4_train import prepare_tabular_features
    
    X_snap, feat_names = prepare_tabular_features(df, config)
    return X_snap, feat_names

