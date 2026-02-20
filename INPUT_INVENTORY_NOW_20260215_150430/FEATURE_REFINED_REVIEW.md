## FEATURE REFINED REVIEW (from inventory_inputs_now)

- schema_manifest: `/home/andre2/GX1_DATA/data/data/prebuilt/V13_REFINED3_PRUNE14/2025/xauusd_m5_2025_features_v13_refined3_prune14_20260215_150430.schema_manifest.json`
- prebuilt_parquet: `/home/andre2/GX1_DATA/data/data/prebuilt/V13_REFINED3_PRUNE14/2025/xauusd_m5_2025_features_v13_refined3_prune14_20260215_150430.parquet`
- sample_n: `20000`

### Family grouping
#### CANDLE GEOMETRY (n=1)
- `wick_asym`

#### REGIME (n=1)
- `trend_regime_tf24h`

#### RETURNS/MOMENTUM (n=4)
- `ret_1`
- `ret_5`
- `ret_20`
- `roc100`

#### SPREAD/COST (n=1)
- `spread_bps`

#### TREND (n=2)
- `ema_50`
- `price_vs_ema50_atr`

#### VOLATILITY (n=3)
- `atr`
- `std50`
- `volatility`

#### VOLUME PROXY (n=2)
- `rvol_60`
- `vol_ratio`

### RVOL_SCALE_AUDIT

```
{
  "rvol_60": {
    "unique_count": 20000,
    "std": 0.2555507895017703,
    "min": 0.0,
    "max": 1.7332030175057056,
    "percent_zero": 5e-05,
    "nan_rate": 0.0,
    "inf_rate": 0.0,
    "label": "ALIVE"
  },
  "vol_ratio": {
    "unique_count": 19995,
    "std": 0.3901646010973053,
    "min": 0.0,
    "max": 5.908074932107707,
    "percent_zero": 5e-05,
    "nan_rate": 0.0,
    "inf_rate": 0.0,
    "label": "ALIVE"
  }
}
```

### Redundant candidates

```
{
  "atr_vs_true_range": false,
  "std50_vs_volatility": true
}
```

