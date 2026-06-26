from __future__ import annotations

import numpy as np
import pandas as pd

from gx1.contracts.signal_bridge_v3 import ORDERED_CTX_CONT_NAMES_V3
from gx1.features.entry_smart_context import (
    ENTRY_SMART_CTX_FEATURE_NAMES,
    add_entry_smart_context_features,
)


def _base_frame(n: int = 96) -> pd.DataFrame:
    df = pd.DataFrame(index=np.arange(n))
    for col in (
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
    ):
        df[col] = 0.0
    return df


def test_entry_smart_features_are_active_contract_features() -> None:
    names = list(ORDERED_CTX_CONT_NAMES_V3)

    assert len(names) == len(set(names))
    assert all(name in names for name in ENTRY_SMART_CTX_FEATURE_NAMES)


def test_entry_smart_context_formulas_match_audit_candidates() -> None:
    df = _base_frame()
    df.loc[95, "smc_choch"] = 1.0
    df.loc[:, "smc_bos_up"] = 1.0
    df.loc[:, "smc_sweep_down"] = 1.0
    df.loc[95, "smc_sweep_size_atr"] = 2.0
    df.loc[95, "smc_bars_since_sweep"] = 24.0
    df.loc[95, "smc_premium_discount"] = 0.75
    df.loc[95, ["dist_to_R1_atr", "dist_to_R2_atr"]] = [2.0, 4.0]
    df.loc[95, ["dist_to_S1_atr", "dist_to_S2_atr"]] = [1.0, 6.0]
    for name in ("dist_to_m5_hi_atr", "dist_to_m15_hi_atr", "dist_to_h1_hi_atr", "dist_to_h4_hi_atr", "dist_to_d1_hi_atr"):
        df.loc[95, name] = 5.0
    for name in ("dist_to_m5_lo_atr", "dist_to_m15_lo_atr", "dist_to_h1_lo_atr", "dist_to_h4_lo_atr", "dist_to_d1_lo_atr"):
        df.loc[95, name] = 2.0
    for name in ("dip_confirmed_m5_v3", "dip_confirmed_m15_v3", "dip_confirmed_h1_v3", "dip_confirmed_h4_v3", "dip_confirmed_d1_v3"):
        df.loc[95, name] = 0.5
    for name in ("dip_proximity_h1_v3", "dip_proximity_h4_v3", "dip_proximity_d1_v3"):
        df.loc[95, name] = 0.25

    add_entry_smart_context_features(df)
    row = df.iloc[95]

    assert row["smc_choch_recent_tau12"] == 1.0
    assert row["smc_bos_pressure_last12"] == 1.0
    assert row["smc_sweep_bull_pressure_last48"] == 1.0
    assert row["smc_sweep_size_recent_tau12"] == 2.0
    np.testing.assert_allclose(row["smc_sweep_recency_tau24"], np.exp(-1.0), rtol=1e-6)
    assert row["smc_premium_extreme_snap"] == 0.5
    assert row["sr_nearest_pivot_abs_atr"] == 1.0
    np.testing.assert_allclose(row["sr_support_proximity_exp"], np.exp(-1.0), rtol=1e-6)
    np.testing.assert_allclose(row["sr_resistance_proximity_exp"], np.exp(-2.0), rtol=1e-6)
    np.testing.assert_allclose(row["dip_confirmed_mean_5tf"], 0.5)
    np.testing.assert_allclose(row["dip_proximity_mean_h1h4d1"], 0.25)

