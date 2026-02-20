## FEATURE REFINED REVIEW (from inventory_inputs_now)

- schema_manifest: `/home/andre2/GX1_DATA/data/data/prebuilt/V13_REFINED3_PRUNE20/2025/xauusd_m5_2025_features_v13_refined3_prune20_20260215_150429.schema_manifest.json`
- prebuilt_parquet: `/home/andre2/GX1_DATA/data/data/prebuilt/V13_REFINED3_PRUNE20/2025/xauusd_m5_2025_features_v13_refined3_prune20_20260215_150429.parquet`
- sample_n: `20000`

### Family grouping
#### CANDLE GEOMETRY (n=2)
- `wick_asym`
- `high_low_ratio`

#### REGIME (n=1)
- `trend_regime_tf24h`

#### RETURNS/MOMENTUM (n=5)
- `ret_1`
- `ret_5`
- `ret_20`
- `roc100`
- `momentum`

#### SPREAD/COST (n=3)
- `spread_bps`
- `spread_atr_ratio`
- `spread_vs_rvol60`

#### TREND (n=2)
- `ema_50`
- `price_vs_ema50_atr`

#### VOLATILITY (n=4)
- `atr`
- `std50`
- `volatility`
- `atr_z_200`

#### VOLUME PROXY (n=3)
- `rvol_20`
- `rvol_60`
- `vol_ratio`

### RVOL_SCALE_AUDIT

```
{
  "rvol_20": {
    "unique_count": 20000,
    "std": 0.3745451095719997,
    "min": 0.0,
    "max": 2.7170699412888775,
    "percent_zero": 5e-05,
    "nan_rate": 0.0,
    "inf_rate": 0.0,
    "label": "ALIVE"
  },
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

