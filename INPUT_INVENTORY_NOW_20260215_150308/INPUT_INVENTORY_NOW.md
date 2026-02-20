## INPUT INVENTORY NOW

### A) SSoT — What does schema say?
- **n_schema_features**: `28`
- **sha256_pipe_join**: `2b07a0c1442bd2e76ecf66d91745328312a2d936535322cd2d8acb453b027d94`
- **schema_manifest**: `/home/andre2/GX1_DATA/data/data/prebuilt/V13_REFINED3/2025/xauusd_m5_2025_features_v13_refined3_20260215_150308.schema_manifest.json`

#### features_in_schema_ordered
- `spread_bps` (SPREAD)
- `spread_atr_ratio` (SPREAD)
- `atr` (ATR)
- `std50`
- `volatility`
- `atr_z_200`
- `ret_1`
- `ret_5`
- `ret_20`
- `roc20`
- `roc100`
- `momentum`
- `ret_1_atr`
- `rvol_20`
- `rvol_60`
- `rvol_ratio`
- `vol_ratio`
- `spread_vs_rvol60` (SPREAD)
- `ema_20` (EMA)
- `ema_50` (EMA)
- `price_vs_ema50_atr`
- `trend_regime_tf24h`
- `body_size_atr`
- `range_atr`
- `upper_shadow_atr`
- `lower_shadow_atr`
- `wick_asym`
- `high_low_ratio`

### B) Empirical — What exists in parquet right now?
- **prebuilt_parquet**: `/home/andre2/GX1_DATA/data/data/prebuilt/V13_REFINED3/2025/xauusd_m5_2025_features_v13_refined3_20260215_150308.parquet`
- **n_rows_total**: `70217`
- **n_parquet_feature_columns**: `28`
- **missing_from_parquet**: `0`
- **extra_in_parquet**: `0`

### C) Liveness on sample (parquet)
- **sample_n**: `20000` (uniform stride)
- **alive_features**: `28`
- **dead_features**: `0`
- **dead_all_zero_features**: `0`

#### Status blocks
- **ema_family_status**: `{'ema_20': {'unique_count': 20000, 'std': 381.747516130809, 'min': 2624.038041819525, 'max': 4365.683426366698, 'percent_zero': 0.0, 'nan_rate': 0.0, 'inf_rate': 0.0, 'label': 'ALIVE'}, 'ema_50': {'unique_count': 20000, 'std': 381.70957828232736, 'min': 2624.600910097967, 'max': 4362.787291259799, 'percent_zero': 0.0, 'nan_rate': 0.0, 'inf_rate': 0.0, 'label': 'ALIVE'}}`
- **oscillator_status**: `{'rsi': {'label': 'MISSING'}, 'macd': {'label': 'MISSING'}}`
- **volatility_status**: `{'atr': {'unique_count': 20000, 'std': 1.8039646402688416, 'min': 0.1500000000005457, 'max': 18.499993455281313, 'percent_zero': 0.0, 'nan_rate': 0.0, 'inf_rate': 0.0, 'label': 'ALIVE'}, 'true_range': {'label': 'MISSING'}, 'std50': {'unique_count': 20000, 'std': 0.00031932717027670826, 'min': 0.0, 'max': 0.002664454205509441, 'percent_zero': 5e-05, 'nan_rate': 0.0, 'inf_rate': 0.0, 'label': 'ALIVE'}}`
- **spread_status**: `{'spread_bps': {'unique_count': 19925, 'std': 0.597583235102278, 'min': 0.4644336142774477, 'max': 18.536643236349615, 'percent_zero': 0.0, 'nan_rate': 0.0, 'inf_rate': 0.0, 'label': 'ALIVE'}, 'spread_atr_ratio': {'unique_count': 20000, 'std': 0.14993666654223886, 'min': 0.028595965631003882, 'max': 8.26666666663817, 'percent_zero': 0.0, 'nan_rate': 0.0, 'inf_rate': 0.0, 'label': 'ALIVE'}}`

#### alive_features
- `atr`
- `atr_z_200`
- `body_size_atr`
- `ema_20`
- `ema_50`
- `high_low_ratio`
- `lower_shadow_atr`
- `momentum`
- `price_vs_ema50_atr`
- `range_atr`
- `ret_1`
- `ret_1_atr`
- `ret_20`
- `ret_5`
- `roc100`
- `roc20`
- `rvol_20`
- `rvol_60`
- `rvol_ratio`
- `spread_atr_ratio`
- `spread_bps`
- `spread_vs_rvol60`
- `std50`
- `trend_regime_tf24h`
- `upper_shadow_atr`
- `vol_ratio`
- `volatility`
- `wick_asym`

#### dead_features

### D) What does XGB actually see? (if dump exists)
- **xgb_dump_present**: `False`
- **xgb_dump_path**: `None`

### E) Conclusion — available now
- **AVAILABLE_NOW_ALIVE**: `28`
- **AVAILABLE_NOW_BUT_DEAD**: `0`
- **NOT_AVAILABLE_NOW**: `0`

### F) Strict mode
- **strict_enabled**: `True`
- PASS

