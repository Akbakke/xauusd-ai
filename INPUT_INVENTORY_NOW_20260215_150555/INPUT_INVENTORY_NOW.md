## INPUT INVENTORY NOW

### A) SSoT — What does schema say?
- **n_schema_features**: `14`
- **sha256_pipe_join**: `2157f092ddcb7d578031ee591f7a25645ff06f79cc87cddd45fe98531975ae40`
- **schema_manifest**: `/home/andre2/GX1_DATA/data/data/prebuilt/V13_REFINED3_PRUNE14/2025/xauusd_m5_2025_features_v13_refined3_prune14_20260215_150554.schema_manifest.json`

#### features_in_schema_ordered
- `spread_bps` (SPREAD)
- `atr` (ATR)
- `std50`
- `volatility`
- `ret_1`
- `ret_5`
- `ret_20`
- `roc100`
- `rvol_60`
- `vol_ratio`
- `ema_50` (EMA)
- `price_vs_ema50_atr`
- `trend_regime_tf24h`
- `wick_asym`

### B) Empirical — What exists in parquet right now?
- **prebuilt_parquet**: `/home/andre2/GX1_DATA/data/data/prebuilt/V13_REFINED3_PRUNE14/2025/xauusd_m5_2025_features_v13_refined3_prune14_20260215_150554.parquet`
- **n_rows_total**: `70217`
- **n_parquet_feature_columns**: `14`
- **missing_from_parquet**: `0`
- **extra_in_parquet**: `0`

### C) Liveness on sample (parquet)
- **sample_n**: `20000` (uniform stride)
- **alive_features**: `14`
- **dead_features**: `0`
- **dead_all_zero_features**: `0`

#### Status blocks
- **ema_family_status**: `{'ema_20': {'label': 'MISSING'}, 'ema_50': {'unique_count': 20000, 'std': 381.70957828232736, 'min': 2624.600910097967, 'max': 4362.787291259799, 'percent_zero': 0.0, 'nan_rate': 0.0, 'inf_rate': 0.0, 'label': 'ALIVE'}}`
- **oscillator_status**: `{'rsi': {'label': 'MISSING'}, 'macd': {'label': 'MISSING'}}`
- **volatility_status**: `{'atr': {'unique_count': 20000, 'std': 1.8039646402688416, 'min': 0.1500000000005457, 'max': 18.499993455281313, 'percent_zero': 0.0, 'nan_rate': 0.0, 'inf_rate': 0.0, 'label': 'ALIVE'}, 'true_range': {'label': 'MISSING'}, 'std50': {'unique_count': 20000, 'std': 0.00031932717027670826, 'min': 0.0, 'max': 0.002664454205509441, 'percent_zero': 5e-05, 'nan_rate': 0.0, 'inf_rate': 0.0, 'label': 'ALIVE'}}`
- **spread_status**: `{'spread_bps': {'unique_count': 19925, 'std': 0.597583235102278, 'min': 0.4644336142774477, 'max': 18.536643236349615, 'percent_zero': 0.0, 'nan_rate': 0.0, 'inf_rate': 0.0, 'label': 'ALIVE'}, 'spread_atr_ratio': {'label': 'MISSING'}}`

#### alive_features
- `atr`
- `ema_50`
- `price_vs_ema50_atr`
- `ret_1`
- `ret_20`
- `ret_5`
- `roc100`
- `rvol_60`
- `spread_bps`
- `std50`
- `trend_regime_tf24h`
- `vol_ratio`
- `volatility`
- `wick_asym`

#### dead_features

### D) What does XGB actually see? (if dump exists)
- **xgb_dump_present**: `False`
- **xgb_dump_path**: `None`

### E) Conclusion — available now
- **AVAILABLE_NOW_ALIVE**: `14`
- **AVAILABLE_NOW_BUT_DEAD**: `0`
- **NOT_AVAILABLE_NOW**: `0`

### F) Strict mode
- **strict_enabled**: `True`
- PASS

