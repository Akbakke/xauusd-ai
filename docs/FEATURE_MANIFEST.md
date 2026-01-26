# Feature Manifest

**Version:** 1.0  
**Generated:** 2026-01-21T15:46:52.247114  
**Prebuilt Parquet:** `data/features/xauusd_m5_2025_features_v10_ctx.parquet`  
**Total Features:** 98

## Purpose

This manifest serves as Single Source of Truth (SSoT) for feature definitions, families, and usage.
It is generated from prebuilt parquet and known feature families.

**Note:** CLOSE is NOT listed as a prebuilt feature. CLOSE is an input alias from `candles.close` 
and is applied in transformer input assembly. Prebuilt parquet schema must NOT contain CLOSE 
or any other reserved candle columns (see `DATA_CONTRACT.md` for reserved columns list).

## Feature Definitions

| Name | Family | Timeframe | Lookback | Units | Normalization | Safe in PREBUILT | Live Only | Consumers | Source Module |
|------|--------|-----------|----------|-------|---------------|------------------|-----------|-----------|---------------|
| ~~`CLOSE`~~ | ~~unknown~~ | ~~M5~~ | ~~N/A~~ | ~~unknown~~ | ~~none~~ | ~~True~~ | ~~False~~ | ~~TRANSFORMER~~ | ~~`gx1.features.runtime_sniper_core`~~ |
**Note:** CLOSE is NOT a prebuilt feature. It is an input alias from `candles.close` applied in transformer input assembly. 
If CLOSE appears in this manifest, the prebuilt parquet schema is contaminated and must be rebuilt.
| `_v1_atr14` | basic_v1 | M5 | N/A | bps | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_atr_regime_id` | basic_v1 | M5 | N/A | bps | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_atr_z_10_100` | basic_v1 | M5 | N/A | bps | ATR-normalized | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_bb_bandwidth_delta_10` | basic_v1 | M5 | N/A | int | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_bb_squeeze_20_2` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_body_share_1` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_body_tr` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_close_ema_slope_3` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_clv` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_comp3_ratio` | basic_v1 | M5 | N/A | ratio | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_cost_bps_dyn` | basic_v1 | M5 | N/A | bps | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_cost_bps_est` | basic_v1 | M5 | N/A | bps | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_ema_diff` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_int_clv_atr` | basic_v1 | M5 | N/A | bps | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_int_ema_us` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_int_r5_atr` | basic_v1 | M5 | N/A | bps | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_int_range_us` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_int_slope_h1_us` | basic_v1 | H1 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_int_slope_h4_atr` | basic_v1 | H4 | N/A | bps | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_int_vwap_h1` | basic_v1 | H1 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_is_EU` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_is_US` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_kama_slope_30` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_kurt_r` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_lower_tr` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_pk_sigma20` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_r1` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_r12` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_r1_q10_48` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_r1_q90_48` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_r24` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_r3` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_r48_z` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_r5` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_r8` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_range_adr` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_range_comp_20_100` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_range_z` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_ret_ema_diff_2_5` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_ret_ema_ratio_5_34` | basic_v1 | M5 | N/A | ratio | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_rsi14` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_rsi14_z` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_rsi2` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_rsi2_gt_rsi14` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_session_tag_EU` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_session_tag_OVERLAP` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_session_tag_US` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_slip_bps` | basic_v1 | M5 | N/A | bps | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_spread_p` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_spread_z` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_tema_slope_20` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_tod_cos` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_tod_sin` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_tr_1_over_atr_14` | basic_v1 | M5 | N/A | bps | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_upper_tr` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_vwap_drift48` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1_wick_imbalance` | basic_v1 | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.basic_v1` |
| `_v1h1_atr` | unknown | H1 | N/A | bps | none | True | False | TRANSFORMER | `gx1.features.htf_aggregator` |
| `_v1h1_ema_diff` | unknown | H1 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.htf_aggregator` |
| `_v1h1_rsi14_z` | unknown | H1 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.htf_aggregator` |
| `_v1h1_slope3` | unknown | H1 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.htf_aggregator` |
| `_v1h1_slope5` | unknown | H1 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.htf_aggregator` |
| `_v1h1_vwap_drift` | unknown | H1 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.htf_aggregator` |
| `_v1h4_atr` | unknown | H4 | N/A | bps | none | True | False | TRANSFORMER | `gx1.features.htf_aggregator` |
| `_v1h4_ema_diff` | unknown | H4 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.htf_aggregator` |
| `_v1h4_rsi14_z` | unknown | H4 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.htf_aggregator` |
| `_v1h4_slope3` | unknown | H4 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.htf_aggregator` |
| `_v1h4_slope5` | unknown | H4 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.htf_aggregator` |
| `atr` | atr | M5 | N/A | bps | none | True | False | TRANSFORMER | `gx1.features.runtime_sniper_core` |
| `atr50` | atr | M5 | 50 | bps | none | True | False | TRANSFORMER | `gx1.seq.sequence_features` |
| `atr_regime_id` | atr | M5 | 1 | bps | none | True | False | GATE_ONLY, TRANSFORMER | `gx1.seq.sequence_features` |
| `atr_z` | atr | M5 | 1 | bps | none | True | False | TRANSFORMER | `gx1.seq.sequence_features` |
| `body_pct` | microstructure | M5 | 1 | ratio | none | True | False | TRANSFORMER | `gx1.seq.sequence_features` |
| `brain_risk_score` | unknown | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.runtime_sniper_core` |
| `ema100_slope` | sequence | M5 | 100 | unknown | none | True | False | TRANSFORMER | `gx1.seq.sequence_features` |
| `ema20_slope` | sequence | M5 | 20 | unknown | none | True | False | TRANSFORMER | `gx1.seq.sequence_features` |
| `is_EU` | session | M5 | N/A | flag | none | True | False | GATE_ONLY | `gx1.features.runtime_sniper_core` |
| `is_US` | session | M5 | N/A | flag | none | True | False | GATE_ONLY | `gx1.features.runtime_sniper_core` |
| `mid` | unknown | M5 | N/A | int | none | True | False | TRANSFORMER | `gx1.features.runtime_sniper_core` |
| `pos_vs_ema200` | sequence | M5 | 20 | unknown | none | True | False | TRANSFORMER | `gx1.seq.sequence_features` |
| `prob_long` | unknown | M5 | N/A | unknown | none | True | False | LOG_ONLY | `gx1.features.runtime_sniper_core` |
| `prob_neutral` | unknown | M5 | N/A | unknown | none | True | False | LOG_ONLY | `gx1.features.runtime_sniper_core` |
| `prob_short` | unknown | M5 | N/A | unknown | none | True | False | LOG_ONLY | `gx1.features.runtime_sniper_core` |
| `range` | unknown | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.runtime_sniper_core` |
| `ret_1` | unknown | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.runtime_sniper_core` |
| `ret_20` | unknown | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.runtime_sniper_core` |
| `ret_5` | unknown | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.runtime_sniper_core` |
| `roc100` | sequence | M5 | 100 | unknown | none | True | False | TRANSFORMER | `gx1.seq.sequence_features` |
| `roc20` | sequence | M5 | 20 | unknown | none | True | False | TRANSFORMER | `gx1.seq.sequence_features` |
| `rvol_20` | unknown | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.runtime_sniper_core` |
| `rvol_60` | unknown | M5 | N/A | unknown | none | True | False | TRANSFORMER | `gx1.features.runtime_sniper_core` |
| `session_id` | session | M5 | 1 | int | none | True | False | GATE_ONLY, TRANSFORMER | `gx1.seq.sequence_features` |
| `side` | unknown | M5 | N/A | int | none | True | False | TRANSFORMER | `gx1.features.runtime_sniper_core` |
| `std50` | sequence | M5 | 50 | unknown | none | True | False | TRANSFORMER | `gx1.seq.sequence_features` |
| `trend_regime_tf24h` | sequence | M5 | 1 | unknown | none | True | False | TRANSFORMER | `gx1.seq.sequence_features` |
| `vol_ratio` | unknown | M5 | N/A | ratio | none | True | False | TRANSFORMER | `gx1.features.runtime_sniper_core` |
| `wick_asym` | microstructure | M5 | 1 | unknown | none | True | False | TRANSFORMER | `gx1.seq.sequence_features` |

## Feature Families

### Atr (4 features)

`atr`, `atr50`, `atr_regime_id`, `atr_z`

### Basic_V1 (57 features)

`_v1_atr14`, `_v1_atr_regime_id`, `_v1_atr_z_10_100`, `_v1_bb_bandwidth_delta_10`, `_v1_bb_squeeze_20_2`, `_v1_body_share_1`, `_v1_body_tr`, `_v1_close_ema_slope_3`, `_v1_clv`, `_v1_comp3_ratio`, `_v1_cost_bps_dyn`, `_v1_cost_bps_est`, `_v1_ema_diff`, `_v1_int_clv_atr`, `_v1_int_ema_us`, `_v1_int_r5_atr`, `_v1_int_range_us`, `_v1_int_slope_h1_us`, `_v1_int_slope_h4_atr`, `_v1_int_vwap_h1`, `_v1_is_EU`, `_v1_is_US`, `_v1_kama_slope_30`, `_v1_kurt_r`, `_v1_lower_tr`, `_v1_pk_sigma20`, `_v1_r1`, `_v1_r12`, `_v1_r1_q10_48`, `_v1_r1_q90_48`, `_v1_r24`, `_v1_r3`, `_v1_r48_z`, `_v1_r5`, `_v1_r8`, `_v1_range_adr`, `_v1_range_comp_20_100`, `_v1_range_z`, `_v1_ret_ema_diff_2_5`, `_v1_ret_ema_ratio_5_34`, `_v1_rsi14`, `_v1_rsi14_z`, `_v1_rsi2`, `_v1_rsi2_gt_rsi14`, `_v1_session_tag_EU`, `_v1_session_tag_OVERLAP`, `_v1_session_tag_US`, `_v1_slip_bps`, `_v1_spread_p`, `_v1_spread_z`, `_v1_tema_slope_20`, `_v1_tod_cos`, `_v1_tod_sin`, `_v1_tr_1_over_atr_14`, `_v1_upper_tr`, `_v1_vwap_drift48`, `_v1_wick_imbalance`

### Microstructure (2 features)

`body_pct`, `wick_asym`

### Sequence (7 features)

`ema100_slope`, `ema20_slope`, `pos_vs_ema200`, `roc100`, `roc20`, `std50`, `trend_regime_tf24h`

### Session (3 features)

`is_EU`, `is_US`, `session_id`

### Unknown (25 features)

`CLOSE`, `_v1h1_atr`, `_v1h1_ema_diff`, `_v1h1_rsi14_z`, `_v1h1_slope3`, `_v1h1_slope5`, `_v1h1_vwap_drift`, `_v1h4_atr`, `_v1h4_ema_diff`, `_v1h4_rsi14_z`, `_v1h4_slope3`, `_v1h4_slope5`, `brain_risk_score`, `mid`, `prob_long`, `prob_neutral`, `prob_short`, `range`, `ret_1`, `ret_20`, `ret_5`, `rvol_20`, `rvol_60`, `side`, `vol_ratio`


## Unknown Features

**Warning:** 25 unknown features found:

- `CLOSE`
- `_v1h1_atr`
- `_v1h1_ema_diff`
- `_v1h1_rsi14_z`
- `_v1h1_slope3`
- `_v1h1_slope5`
- `_v1h1_vwap_drift`
- `_v1h4_atr`
- `_v1h4_ema_diff`
- `_v1h4_rsi14_z`
- `_v1h4_slope3`
- `_v1h4_slope5`
- `brain_risk_score`
- `mid`
- `prob_long`
- `prob_neutral`
- `prob_short`
- `range`
- `ret_1`
- `ret_20`
- `ret_5`
- `rvol_20`
- `rvol_60`
- `side`
- `vol_ratio`

## References

- **Feature Map:** `reports/repo_audit/FEATURE_MAP.md`
- **Data Contract:** `docs/DATA_CONTRACT.md`
- **Prebuilt Loader:** `gx1/execution/prebuilt_features_loader.py`
