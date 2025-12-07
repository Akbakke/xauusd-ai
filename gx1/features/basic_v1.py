# gx1/features/basic_v1.py
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import pickle
from pathlib import Path

def _roll(s, win, fn, minp=None):
    if minp is None: minp = max(1, int(win*0.8))
    return getattr(s.rolling(win, min_periods=minp), fn)()

def _ema(s, span):
    return s.ewm(span=span, adjust=False, min_periods=max(1, span//2)).mean()

def _zscore(s, win):
    r = s.rolling(win, min_periods=max(3, win//2))
    return (s - r.mean()) / (r.std(ddof=0) + 1e-12)

def _true_range(high, low, close):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def _parkinson_sigma(high, low):
    # sqrt(1/(4ln2)) * ln(high/low)
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.log(high / low.replace(0, np.nan))
    return (x**2).rolling(20, min_periods=10).mean().pow(0.5) / np.sqrt(4*np.log(2))

def add_session_features(df, tz_offset_minutes=0):
    # Deriver EU/US session fra timestamp (UTC antatt), uten å lekke
    ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    hour = ts.dt.hour
    # EU ~ 07–15 UTC, US ~ 13–20 UTC
    df["is_EU"] = ((hour >= 7) & (hour <= 15)).astype(int)
    df["is_US"] = ((hour >= 13) & (hour <= 20)).astype(int)
    return df

def build_basic_v1(df):
    """
    Forventer kolonner: ts, open, high, low, close, (valgfritt: vwap, spread_pct, slippage_bps)
    Bruker kun fortid: shift(1) og ruller bakover.
    Returnerer df med nye _v1_* features; originalkolonner beholdes.
    """
    df = df.copy()
    # Sikre typer
    for c in ["open","high","low","close"]:
        if c not in df: raise ValueError(f"Column '{c}' missing for basic_v1 features")
        df[c] = df[c].astype(float)

    # --- Momentum (laggede avkastninger) ---
    ret1 = df["close"].pct_change().fillna(0.0)
    df["_v1_r1"]  = ret1.shift(1)
    for k in [3,5,8,12,24]:
        rk = df["close"].pct_change(k)
        df[f"_v1_r{k}"] = rk.shift(1)
    df["_v1_r48_z"] = _zscore(ret1, 48).shift(1)

    # --- Volatilitet / Regime ---
    tr = _true_range(df["high"], df["low"], df["close"])
    atr14 = _roll(tr, 14, "mean")
    df["_v1_atr14"] = atr14.shift(1)
    df["_v1_pk_sigma20"] = _parkinson_sigma(df["high"], df["low"]).shift(1)

    # ATR-regime-ID (kvantiler på rullende 20 dager)
    atrq = _zscore(atr14, 288*20)  # ~20 trading-dager på M5 ~ 288 bars/dag
    df["_v1_atr_regime_id"] = pd.qcut(atr14.rank(pct=True), 3, labels=False, duplicates="drop").astype(float).fillna(1.0).shift(1)

    # --- Trend / Mean-reversion ---
    ema12 = _ema(df["close"], 12)
    ema26 = _ema(df["close"], 26)
    df["_v1_ema_diff"] = (ema12 - ema26).shift(1)

    # VWAP drift (fallback hvis vwap mangler)
    if "vwap" in df:
        vwap = df["vwap"].astype(float)
    else:
        # grov vwap-proxy med HLC3
        vwap = (df["high"]+df["low"]+df["close"])/3.0
    df["_v1_vwap_drift48"] = (df["close"] - _roll(vwap, 48, "mean")).shift(1)

    # RSI14 z-score (klassisk)
    delta = df["close"].diff()
    up = delta.clip(lower=0).rolling(14, min_periods=7).mean()
    dn = (-delta.clip(upper=0)).rolling(14, min_periods=7).mean()
    rs = up / (dn + 1e-12)
    rsi = 100 - 100/(1 + rs)
    df["_v1_rsi14_z"] = _zscore(rsi, 48).shift(1)
    
    # RSI2 and RSI14 (raw, for mini-featurepack)
    up2 = delta.clip(lower=0).rolling(2, min_periods=1).mean()
    dn2 = (-delta.clip(upper=0)).rolling(2, min_periods=1).mean()
    rs2 = up2 / (dn2 + 1e-12)
    rsi2 = 100 - 100/(1 + rs2)
    df["_v1_rsi2"] = rsi2.shift(1).fillna(50.0)
    df["_v1_rsi14"] = rsi.shift(1).fillna(50.0)
    df["_v1_rsi2_gt_rsi14"] = (rsi2 > rsi).astype(float).shift(1).fillna(0.0)

    # --- Raskere features (tempo) ---
    # 1) micro-momentum: diff mellom EMA av returns (2 vs 5)
    ret = df["close"].pct_change().fillna(0.0)
    ema_ret2 = _ema(ret, 2)
    ema_ret5 = _ema(ret, 5)
    df["_v1_ret_ema_diff_2_5"] = (ema_ret2 - ema_ret5).shift(1).fillna(0.0)
    
    # 2) BB bandwidth delta 10: endring i squeeze (opløser stramt regime)
    bb_mean10 = df["close"].rolling(10, min_periods=5).mean()
    bb_std10 = df["close"].rolling(10, min_periods=5).std(ddof=0)
    bb_upper10 = bb_mean10 + 2.0 * bb_std10
    bb_lower10 = bb_mean10 - 2.0 * bb_std10
    bb_width10 = (bb_upper10 - bb_lower10) / (bb_mean10 + 1e-12)
    df["_v1_bb_bandwidth_delta_10"] = bb_width10.diff(3).shift(1).fillna(0.0)

    # 3) Pre-trend booster: kort EMA-slope på close (3 vs 6)
    ema3 = _ema(df["close"], 3)
    ema6 = _ema(df["close"], 6)
    df["_v1_close_ema_slope_3"] = ((ema3 - ema6) / (ema6 + 1e-12)).shift(1).fillna(0.0)

    # --- Candle shape / struktur ---
    body = (df["close"] - df["open"]).abs()
    range_ = (df["high"] - df["low"]).replace(0, np.nan)
    upper = (df["high"] - df[["open","close"]].max(axis=1)).clip(lower=0)
    lower = (df[["open","close"]].min(axis=1) - df["low"]).clip(lower=0)

    df["_v1_body_tr"]   = (body / (range_ + 1e-12)).shift(1)
    df["_v1_upper_tr"]  = (upper / (range_ + 1e-12)).shift(1)
    df["_v1_lower_tr"]  = (lower / (range_ + 1e-12)).shift(1)
    
    # Wick imbalance (mini-featurepack)
    # Upper wick vs lower wick imbalance
    wick_upper = upper
    wick_lower = lower
    range_safe = range_.replace(0, np.nan)
    df["_v1_wick_imbalance"] = ((wick_upper - wick_lower) / (range_safe + 1e-12)).shift(1).fillna(0.0)
    
    # Range comparison 20 vs 100 periods (mini-featurepack)
    range_20 = range_.rolling(20, min_periods=10).mean()
    range_100 = range_.rolling(100, min_periods=50).mean()
    df["_v1_range_comp_20_100"] = ((range_20 / (range_100 + 1e-12)) - 1.0).shift(1).fillna(0.0)
    # Range relativt til ADR(20)
    adr20 = _roll(range_, 288*20, "mean")
    df["_v1_range_adr"] = (range_ / (adr20 + 1e-12)).shift(1)

    # --- Kost-proxies ---
    if "spread_pct" in df:
        df["_v1_spread_p"] = (df["spread_pct"] * 1e4).shift(1)  # til bps
    else:
        df["_v1_spread_p"] = 12.0  # konservativt
    if "slippage_bps" in df:
        df["_v1_slip_bps"] = df["slippage_bps"].astype(float).shift(1)
    else:
        df["_v1_slip_bps"] = 10.0

    # --- Session ---
    if "ts" in df:
        df = add_session_features(df)
        df["_v1_is_EU"] = df["is_EU"].shift(1).fillna(0)
        df["_v1_is_US"] = df["is_US"].shift(1).fillna(0)

    # --- HTF-kontekst (Multi-timeframe) ---
    df_temp = None
    ts_index = None
    
    if "ts" in df.columns or isinstance(df.index, pd.DatetimeIndex):
        # Resample til H1 og H4
        if isinstance(df.index, pd.DatetimeIndex):
            ts_index = df.index
            df_resample = df
        elif "ts" in df.columns:
            ts_index = pd.to_datetime(df["ts"], utc=True, errors="coerce")
            # Set temporary index for resampling
            df_temp = df.set_index(ts_index)
            df_resample = df_temp
        else:
            df_resample = None
        
        if df_resample is not None:
            
            # H1 features
            try:
                h1_df = pd.DataFrame({
                    "open": df_resample["open"].resample("1h").first(),  # Use 'h' instead of 'H'
                    "high": df_resample["high"].resample("1h").max(),
                    "low": df_resample["low"].resample("1h").min(),
                    "close": df_resample["close"].resample("1h").last(),
                }).ffill()
                
                if len(h1_df) > 0:
                    # Build same V1 features on H1
                    h1_ema12 = _ema(h1_df["close"], 12)
                    h1_ema26 = _ema(h1_df["close"], 26)
                    h1_vwap = (h1_df["high"]+h1_df["low"]+h1_df["close"])/3.0
                    h1_tr = _true_range(h1_df["high"], h1_df["low"], h1_df["close"])
                    h1_atr14 = _roll(h1_tr, 14, "mean")
                    h1_delta = h1_df["close"].diff()
                    h1_up = h1_delta.clip(lower=0).rolling(14, min_periods=7).mean()
                    h1_dn = (-h1_delta.clip(upper=0)).rolling(14, min_periods=7).mean()
                    h1_rs = h1_up / (h1_dn + 1e-12)
                    h1_rsi = 100 - 100/(1 + h1_rs)
                    
                    # Forward-fill H1 features to M5 and shift(1)
                    # Fix target_index logic: use original df.index for assignment
                    if isinstance(df.index, pd.DatetimeIndex):
                        target_index = df.index
                    elif "ts" in df.columns:
                        target_index = pd.to_datetime(df["ts"], utc=True, errors="coerce")
                    else:
                        target_index = df.index
                    
                    h1_ema_diff = (h1_ema12 - h1_ema26).reindex(target_index, method="ffill").shift(1).fillna(0.0)
                    h1_vwap_drift = (h1_df["close"] - _roll(h1_vwap, 48, "mean")).reindex(target_index, method="ffill").shift(1).fillna(0.0)
                    h1_atr = h1_atr14.reindex(target_index, method="ffill").shift(1).fillna(0.0)
                    h1_rsi_z = _zscore(h1_rsi, 48).reindex(target_index, method="ffill").shift(1).fillna(0.0)
                    
                    df["_v1h1_ema_diff"] = h1_ema_diff.values if hasattr(h1_ema_diff, 'values') else h1_ema_diff
                    df["_v1h1_vwap_drift"] = h1_vwap_drift.values if hasattr(h1_vwap_drift, 'values') else h1_vwap_drift
                    df["_v1h1_atr"] = h1_atr.values if hasattr(h1_atr, 'values') else h1_atr
                    df["_v1h1_rsi14_z"] = h1_rsi_z.values if hasattr(h1_rsi_z, 'values') else h1_rsi_z
            except Exception:
                # Fallback: ingen HTF features
                pass
            
            # H4 features (samme logikk)
            try:
                h4_df = pd.DataFrame({
                    "open": df_resample["open"].resample("4h").first(),  # Use 'h' instead of 'H'
                    "high": df_resample["high"].resample("4h").max(),
                    "low": df_resample["low"].resample("4h").min(),
                    "close": df_resample["close"].resample("4h").last(),
                }).ffill()
                
                if len(h4_df) > 0:
                    h4_ema12 = _ema(h4_df["close"], 12)
                    h4_ema26 = _ema(h4_df["close"], 26)
                    h4_tr = _true_range(h4_df["high"], h4_df["low"], h4_df["close"])
                    h4_atr14 = _roll(h4_tr, 14, "mean")
                    h4_delta = h4_df["close"].diff()
                    h4_up = h4_delta.clip(lower=0).rolling(14, min_periods=7).mean()
                    h4_dn = (-h4_delta.clip(upper=0)).rolling(14, min_periods=7).mean()
                    h4_rs = h4_up / (h4_dn + 1e-12)
                    h4_rsi = 100 - 100/(1 + h4_rs)
                    
                    # Fix target_index logic: use original df.index for assignment
                    if isinstance(df.index, pd.DatetimeIndex):
                        target_index = df.index
                    elif "ts" in df.columns:
                        target_index = pd.to_datetime(df["ts"], utc=True, errors="coerce")
                    else:
                        target_index = df.index
                    
                    h4_ema_diff = (h4_ema12 - h4_ema26).reindex(target_index, method="ffill").shift(1).fillna(0.0)
                    h4_atr = h4_atr14.reindex(target_index, method="ffill").shift(1).fillna(0.0)
                    h4_rsi_z = _zscore(h4_rsi, 48).reindex(target_index, method="ffill").shift(1).fillna(0.0)
                    
                    df["_v1h4_ema_diff"] = h4_ema_diff.values if hasattr(h4_ema_diff, 'values') else h4_ema_diff
                    df["_v1h4_atr"] = h4_atr.values if hasattr(h4_atr, 'values') else h4_atr
                    df["_v1h4_rsi14_z"] = h4_rsi_z.values if hasattr(h4_rsi_z, 'values') else h4_rsi_z
            except Exception as e:
                # Log exception for debugging
                import warnings
                warnings.warn(f"H4 features failed: {e}", RuntimeWarning)
    
    # --- Mikrostruktur ---
    # CLV (Close Location Value)
    range_safe = (df["high"] - df["low"]).replace(0, np.nan)
    df["_v1_clv"] = ((df["close"] - df["low"]) / (range_safe + 1e-12)).shift(1).fillna(0.5)
    
    # Range z-score
    df["_v1_range_z"] = _zscore(df["high"] - df["low"], 48).shift(1).fillna(0.0)
    
    # Spread z-score (rullende z for robusthet)
    if "spread_pct" in df.columns:
        sp = df["spread_pct"].astype(float)
        # Sjekk om spread varierer (ikke konstant)
        if sp.std() > 1e-6:
            sp_roll = sp.rolling(144, min_periods=72)
            sp_mean = sp_roll.mean()
            sp_std = sp_roll.std(ddof=0) + 1e-12
            df["_v1_spread_z"] = ((sp - sp_mean) / sp_std).shift(1).fillna(0.0)
        else:
            # Spread er konstant, bruk range_z i stedet eller sett til 0
            df["_v1_spread_z"] = 0.0
        
        # Kost-estimat (bruk faktisk spread + slippage hvis tilgjengelig)
        spread_bps = (sp * 1e4).shift(1).fillna(12.0)
        if "slippage_bps" in df.columns:
            slip_bps = df["slippage_bps"].astype(float).shift(1).fillna(10.0)
        else:
            slip_bps = df.get("_v1_slip_bps", 10.0)
        df["_v1_cost_bps_est"] = spread_bps + slip_bps
    else:
        df["_v1_spread_z"] = 0.0
        df["_v1_cost_bps_est"] = 22.0  # fallback
    
    # Kurtosis av returer (vol-form)
    ret1 = df["close"].pct_change().fillna(0.0)
    df["_v1_kurt_r"] = ret1.rolling(48, min_periods=12).apply(lambda x: pd.Series(x).kurtosis() if len(x) > 3 else 0.0, raw=False).shift(1).fillna(0.0)
    
    # --- HTF-momentum "slope" (trendaks) - BYGG FØRST ---
    if "_v1h1_ema_diff" in df.columns:
        # Differanse av H1 EMA-diff over 3-5 trinn
        h1_ema_diff_series = df["_v1h1_ema_diff"]
        df["_v1h1_slope3"] = h1_ema_diff_series.diff(3).shift(1).fillna(0.0)
        df["_v1h1_slope5"] = h1_ema_diff_series.diff(5).shift(1).fillna(0.0)
    
    if "_v1h4_ema_diff" in df.columns:
        h4_ema_diff_series = df["_v1h4_ema_diff"]
        df["_v1h4_slope3"] = h4_ema_diff_series.diff(3).shift(1).fillna(0.0)
        df["_v1h4_slope5"] = h4_ema_diff_series.diff(5).shift(1).fillna(0.0)
    
    # --- Interaksjoner (signal × regime) - BYGG ETTER SLOPES ---
    def _clip(x, lo=-1e3, hi=1e3):
        return x.clip(lo, hi).fillna(0.0)
    
    # Base interaksjoner
    if "_v1_r5" in df.columns and "_v1_atr_regime_id" in df.columns:
        df["_v1_int_r5_atr"] = _clip(df["_v1_r5"] * (1.0 + df["_v1_atr_regime_id"]))
    
    if "_v1_ema_diff" in df.columns and "_v1_is_US" in df.columns:
        df["_v1_int_ema_us"] = _clip(df["_v1_ema_diff"] * df["_v1_is_US"])
    
    if "_v1_vwap_drift48" in df.columns and "_v1h1_ema_diff" in df.columns:
        df["_v1_int_vwap_h1"] = _clip(df["_v1_vwap_drift48"] * df.get("_v1h1_ema_diff", 0.0))
    
    # Slope-interaksjoner (AUC-boost) - nå når slopes eksisterer
    if "_v1h1_slope3" in df.columns and "_v1_is_US" in df.columns:
        df["_v1_int_slope_h1_us"] = _clip(df.get("_v1h1_slope3", 0.0) * df["_v1_is_US"])
    
    # Sjekk at _v1h4_slope5 faktisk bygges før interaksjoner
    if "_v1h4_slope5" in df.columns and "_v1_atr_regime_id" in df.columns:
        df["_v1_int_slope_h4_atr"] = (df["_v1h4_slope5"] * (1.0 + df["_v1_atr_regime_id"])).clip(-1e3, 1e3)
    
    # Dynamisk kost-proxy (varierende kost når spread/slippage mangler/konstante)
    if "close" in df.columns and "_v1_atr14" in df.columns:
        atr_pct = (df["_v1_atr14"] / df["close"]).fillna(0.0)
        rng = (df["high"] - df["low"]).replace(0, np.nan)
        rng_z = _zscore(rng, 48).fillna(0.0)
        
        # Session factor
        if "_v1_is_US" in df.columns:
            sess_factor = 1.15 * df["_v1_is_US"] + 1.0 * (1 - df["_v1_is_US"])
        else:
            sess_factor = pd.Series(1.0, index=df.index)
        
        base_bps = 12.0
        rng_z_pos = rng_z.clip(lower=0.0)
        scale_term = (0.6 * atr_pct * 1e4 + 0.4 * rng_z_pos).clip(0.0, 3.0)
        df["_v1_cost_bps_dyn"] = (base_bps * sess_factor * (0.50 + 0.50 * scale_term)).shift(1).fillna(base_bps)
    else:
        df["_v1_cost_bps_dyn"] = 12.0  # fallback
    
    # To ekstra interaksjoner (AUC-løft) - ofte +0.01–0.02 AUC
    if "_v1_clv" in df.columns and "_v1_atr_regime_id" in df.columns:
        df["_v1_int_clv_atr"] = _clip(df.get("_v1_clv", 0.0) * (1.0 + df["_v1_atr_regime_id"]))
    
    if "_v1_range_z" in df.columns and "_v1_is_US" in df.columns:
        df["_v1_int_range_us"] = _clip(df.get("_v1_range_z", 0.0) * df["_v1_is_US"])
    
    # Time-of-day encoding (sin/cos) og rolling quantil-momenter
    if "ts" in df.columns or isinstance(df.index, pd.DatetimeIndex):
        ts_col = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df["ts"], utc=True, errors="coerce")
        if pd.api.types.is_datetime64_any_dtype(ts_col):
            if isinstance(ts_col, pd.DatetimeIndex):
                h = ts_col.hour
                m = ts_col.minute
            else:
                h = ts_col.dt.hour
                m = ts_col.dt.minute
            h_float = (h + m / 60.0).astype(float)
            tod_sin = np.sin(2 * np.pi * h_float / 24.0)
            tod_cos = np.cos(2 * np.pi * h_float / 24.0)
            df["_v1_tod_sin"] = pd.Series(tod_sin, index=df.index).shift(1).fillna(0.0)
            df["_v1_tod_cos"] = pd.Series(tod_cos, index=df.index).shift(1).fillna(0.0)
    
    # Rolling quantil-momenter
    if "close" in df.columns:
        r1 = df["close"].pct_change().fillna(0.0)
        df["_v1_r1_q90_48"] = r1.rolling(48, min_periods=24).quantile(0.90).shift(1).fillna(0.0)
        df["_v1_r1_q10_48"] = r1.rolling(48, min_periods=24).quantile(0.10).shift(1).fillna(0.0)
    
    # --- Featurepack v2 (lav risiko, høy effekt) ---
    # ATR z-score 10 vs 100 (vol-regime)
    atr10 = _roll(tr, 10, "mean")
    atr100 = _roll(tr, 100, "mean")
    df["_v1_atr_z_10_100"] = ((atr10 - atr100) / (atr100 + 1e-12)).shift(1).fillna(0.0)
    
    # TEMA slope 20 (trend-skarphet)
    tema20 = _tema(df["close"], 20)
    df["_v1_tema_slope_20"] = tema20.diff(3).shift(1).fillna(0.0)
    
    # Bollinger Band squeeze 20_2 (volatilitetssqueeze)
    bb_mean = df["close"].rolling(20, min_periods=10).mean()
    bb_std = df["close"].rolling(20, min_periods=10).std(ddof=0)
    bb_upper = bb_mean + 2.0 * bb_std
    bb_lower = bb_mean - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower) / (bb_mean + 1e-12)
    bb_width_mean = bb_width.rolling(100, min_periods=50).mean()
    df["_v1_bb_squeeze_20_2"] = (bb_width / (bb_width_mean + 1e-12) - 1.0).shift(1).fillna(0.0)
    
    # KAMA slope 30 (trend med støyfilter)
    kama30 = _kama(df["close"], 30)
    df["_v1_kama_slope_30"] = kama30.diff(5).shift(1).fillna(0.0)
    
    # Return EMA ratio 5 vs 34 (momentum vs medium trend)
    ret1 = df["close"].pct_change().fillna(0.0)
    ret_ema5 = _ema(ret1.abs(), 5)
    ret_ema34 = _ema(ret1.abs(), 34)
    df["_v1_ret_ema_ratio_5_34"] = (ret_ema5 / (ret_ema34 + 1e-12)).shift(1).fillna(1.0)
    
    # --- Raskere features (compression→expansion) ---
    # 1) body_share_1: store kropp vs range (signal på sterk momentum)
    body = (df["close"] - df["open"]).abs()
    range_safe = (df["high"] - df["low"]).replace(0, np.nan)
    df["_v1_body_share_1"] = (body / (range_safe + 1e-9)).shift(1).fillna(0.5)
    
    # 2) tr_1_over_atr_14: eksplosjon relativt ATR (breakout-styrke)
    tr1 = _true_range(df["high"], df["low"], df["close"])
    df["_v1_tr_1_over_atr_14"] = (tr1 / (atr14 + 1e-12)).shift(1).fillna(1.0)
    
    # 3) comp3_ratio: 3-bar kompresjon (lave std indikerer squeeze)
    close_std3 = df["close"].rolling(3, min_periods=2).std(ddof=0)
    close_std20 = df["close"].rolling(20, min_periods=10).std(ddof=0)
    df["_v1_comp3_ratio"] = (close_std3 / (close_std20 + 1e-12)).shift(1).fillna(1.0)
    
    # Rydd NaNs
    newcols = [c for c in df.columns if c.startswith("_v1")]
    df[newcols] = df[newcols].replace([np.inf,-np.inf], np.nan).fillna(0.0)
    return df, newcols


def _tema(series, period):
    """Triple Exponential Moving Average (TEMA)."""
    ema1 = _ema(series, period)
    ema2 = _ema(ema1, period)
    ema3 = _ema(ema2, period)
    tema = 3 * ema1 - 3 * ema2 + ema3
    return tema


def _kama(series, period):
    """Kaufman Adaptive Moving Average (KAMA)."""
    change = series.diff(period).abs()
    volatility = series.diff().abs().rolling(period, min_periods=1).sum()
    er = change / (volatility + 1e-12)
    sc = (er * (2.0 / (2 + 1) - 2.0 / (30 + 1)) + 2.0 / (30 + 1)) ** 2
    kama = pd.Series(0.0, index=series.index)
    kama.iloc[0] = series.iloc[0]
    for i in range(1, len(series)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (series.iloc[i] - kama.iloc[i-1])
    return kama


class FeaturePipeline:
    """
    Robust normalization pipeline using RobustScaler.
    Persists scaler per fold.
    """
    def __init__(self, qrange=(10.0, 90.0)):
        """
        Initialize feature pipeline with robust scaler.
        
        Args:
            qrange: Quantile range for RobustScaler (default: (10.0, 90.0))
        """
        self.scaler = RobustScaler(quantile_range=qrange)
        self.fitted = False
    
    def fit(self, X):
        """
        Fit scaler on training data.
        
        Args:
            X: Feature matrix (n_samples, n_features) or DataFrame
            
        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.scaler.fit(X)
        self.fitted = True
        return self
    
    def transform(self, X):
        """
        Transform features using fitted scaler.
        
        Args:
            X: Feature matrix (n_samples, n_features) or DataFrame
            
        Returns:
            Transformed features (numpy array or DataFrame with same index)
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")
        is_df = isinstance(X, pd.DataFrame)
        if is_df:
            index = X.index
            columns = X.columns
            X = X.values
        
        X_transformed = self.scaler.transform(X)
        
        if is_df:
            return pd.DataFrame(X_transformed, index=index, columns=columns)
        return X_transformed
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
    
    def save(self, path):
        """Save scaler to pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load(self, path):
        """Load scaler from pickle file."""
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        self.fitted = True
        return self
