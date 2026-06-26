from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


ENTRY_SMART_CTX_FEATURE_NAMES: list[str] = [
    "smc_choch_recent_tau12",
    "smc_choch_recent_tau24",
    "smc_bos_pressure_last12",
    "smc_bos_pressure_last48",
    "smc_sweep_bull_pressure_last12",
    "smc_sweep_bull_pressure_last48",
    "smc_sweep_size_recent_tau12",
    "smc_sweep_recency_tau24",
    "smc_premium_extreme_snap",
    "sr_nearest_pivot_abs_atr",
    "sr_support_proximity_exp",
    "sr_resistance_proximity_exp",
    "sr_support_minus_resistance_prox",
    "liquidity_hi_nearest_abs_atr",
    "liquidity_lo_nearest_abs_atr",
    "liquidity_lo_minus_hi_prox",
    "dip_confirmed_mean_5tf",
    "dip_confirmed_max_5tf",
    "dip_proximity_mean_h1h4d1",
]

_SOURCE_COLUMNS: tuple[str, ...] = (
    "smc_choch",
    "smc_bos_up",
    "smc_bos_down",
    "smc_sweep_up",
    "smc_sweep_down",
    "smc_sweep_size_atr",
    "smc_bars_since_sweep",
    "smc_premium_discount",
    "dist_to_R1_atr",
    "dist_to_R2_atr",
    "dist_to_S1_atr",
    "dist_to_S2_atr",
    "dist_to_m5_hi_atr",
    "dist_to_m15_hi_atr",
    "dist_to_h1_hi_atr",
    "dist_to_h4_hi_atr",
    "dist_to_d1_hi_atr",
    "dist_to_m5_lo_atr",
    "dist_to_m15_lo_atr",
    "dist_to_h1_lo_atr",
    "dist_to_h4_lo_atr",
    "dist_to_d1_lo_atr",
    "dip_confirmed_m5_v3",
    "dip_confirmed_m15_v3",
    "dip_confirmed_h1_v3",
    "dip_confirmed_h4_v3",
    "dip_confirmed_d1_v3",
    "dip_proximity_h1_v3",
    "dip_proximity_h4_v3",
    "dip_proximity_d1_v3",
)


def required_entry_smart_source_columns() -> tuple[str, ...]:
    return _SOURCE_COLUMNS


def _source_array(df: pd.DataFrame, name: str, *, strict: bool) -> np.ndarray:
    if name not in df.columns:
        if strict:
            raise RuntimeError(f"ENTRY_SMART_CONTEXT_SOURCE_MISSING: {name}")
        return np.zeros(len(df), dtype=np.float32)
    return (
        pd.to_numeric(df[name], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )


def _stack_sources(df: pd.DataFrame, names: Sequence[str], *, strict: bool) -> np.ndarray:
    return np.stack([_source_array(df, name, strict=strict) for name in names], axis=1)


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    if len(arr) == 0:
        return np.zeros(0, dtype=np.float32)
    s = pd.Series(np.asarray(arr, dtype=np.float32))
    return s.rolling(int(window), min_periods=1).mean().to_numpy(dtype=np.float32)


def _rolling_decay_weighted_max(values: np.ndarray, *, tau: float, max_window: int = 96) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    out = np.zeros(len(vals), dtype=np.float32)
    weights = np.exp(-np.arange(int(max_window), dtype=np.float32) / float(tau)).astype(np.float32)
    for i in range(len(vals)):
        start = max(0, i - int(max_window) + 1)
        window_vals = vals[start : i + 1][::-1]
        if len(window_vals):
            out[i] = float(np.max(window_vals * weights[: len(window_vals)]))
    return out


def add_entry_smart_context_features(df: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
    """Add promoted Entry Transformer context features in-place and return df.

    The formulas mirror the audit-derived candidate matrix, but operate on a
    time-ordered DataFrame so train, batch inference, and live inference can use
    the same implementation.
    """
    if len(df) == 0:
        for name in ENTRY_SMART_CTX_FEATURE_NAMES:
            df[name] = np.zeros(0, dtype=np.float32)
        return df

    choch = _source_array(df, "smc_choch", strict=strict)
    bos_up = _source_array(df, "smc_bos_up", strict=strict)
    bos_down = _source_array(df, "smc_bos_down", strict=strict)
    sweep_up = _source_array(df, "smc_sweep_up", strict=strict)
    sweep_down = _source_array(df, "smc_sweep_down", strict=strict)
    sweep_size = _source_array(df, "smc_sweep_size_atr", strict=strict)
    bars_since_sweep = np.clip(_source_array(df, "smc_bars_since_sweep", strict=strict), 0.0, 999.0)
    premium = _source_array(df, "smc_premium_discount", strict=strict)

    r = _stack_sources(df, ("dist_to_R1_atr", "dist_to_R2_atr"), strict=strict)
    s = _stack_sources(df, ("dist_to_S1_atr", "dist_to_S2_atr"), strict=strict)
    r_abs = np.min(np.abs(r), axis=1)
    s_abs = np.min(np.abs(s), axis=1)
    r_prox = np.exp(-np.minimum(r_abs, 20.0)).astype(np.float32)
    s_prox = np.exp(-np.minimum(s_abs, 20.0)).astype(np.float32)

    hi = _stack_sources(
        df,
        ("dist_to_m5_hi_atr", "dist_to_m15_hi_atr", "dist_to_h1_hi_atr", "dist_to_h4_hi_atr", "dist_to_d1_hi_atr"),
        strict=strict,
    )
    lo = _stack_sources(
        df,
        ("dist_to_m5_lo_atr", "dist_to_m15_lo_atr", "dist_to_h1_lo_atr", "dist_to_h4_lo_atr", "dist_to_d1_lo_atr"),
        strict=strict,
    )
    hi_abs = np.min(np.abs(hi), axis=1)
    lo_abs = np.min(np.abs(lo), axis=1)

    dip_confirmed = _stack_sources(
        df,
        ("dip_confirmed_m5_v3", "dip_confirmed_m15_v3", "dip_confirmed_h1_v3", "dip_confirmed_h4_v3", "dip_confirmed_d1_v3"),
        strict=strict,
    )
    dip_prox = _stack_sources(
        df,
        ("dip_proximity_h1_v3", "dip_proximity_h4_v3", "dip_proximity_d1_v3"),
        strict=strict,
    )

    df["smc_choch_recent_tau12"] = _rolling_decay_weighted_max(choch, tau=12.0)
    df["smc_choch_recent_tau24"] = _rolling_decay_weighted_max(choch, tau=24.0)
    df["smc_bos_pressure_last12"] = _rolling_mean(bos_up - bos_down, 12)
    df["smc_bos_pressure_last48"] = _rolling_mean(bos_up - bos_down, 48)
    df["smc_sweep_bull_pressure_last12"] = _rolling_mean(sweep_down - sweep_up, 12)
    df["smc_sweep_bull_pressure_last48"] = _rolling_mean(sweep_down - sweep_up, 48)
    df["smc_sweep_size_recent_tau12"] = _rolling_decay_weighted_max(sweep_size, tau=12.0)
    df["smc_sweep_recency_tau24"] = np.exp(-bars_since_sweep / 24.0).astype(np.float32)
    df["smc_premium_extreme_snap"] = (np.abs(premium - 0.5) * 2.0).astype(np.float32)
    df["sr_nearest_pivot_abs_atr"] = np.minimum(r_abs, s_abs).astype(np.float32)
    df["sr_support_proximity_exp"] = s_prox
    df["sr_resistance_proximity_exp"] = r_prox
    df["sr_support_minus_resistance_prox"] = (s_prox - r_prox).astype(np.float32)
    df["liquidity_hi_nearest_abs_atr"] = hi_abs.astype(np.float32)
    df["liquidity_lo_nearest_abs_atr"] = lo_abs.astype(np.float32)
    df["liquidity_lo_minus_hi_prox"] = (
        np.exp(-np.minimum(lo_abs, 20.0)) - np.exp(-np.minimum(hi_abs, 20.0))
    ).astype(np.float32)
    df["dip_confirmed_mean_5tf"] = dip_confirmed.mean(axis=1).astype(np.float32)
    df["dip_confirmed_max_5tf"] = dip_confirmed.max(axis=1).astype(np.float32)
    df["dip_proximity_mean_h1h4d1"] = dip_prox.mean(axis=1).astype(np.float32)
    return df
