# SNIPER 2025 Regime Fields Audit (20251218_142939)

This report verifies that SNIPER trade journals contain the FARM-style
regime and microstructure fields needed for offline regime analysis.

## Audited runs

- `gx1/wf_runs/SNIPER_OBS_Q1_2025_baseline_20251218_124410` (23 columns in index)
- `gx1/wf_runs/SNIPER_OBS_Q1_2025_guarded_20251218_115831` (23 columns in index)
- `gx1/wf_runs/SNIPER_OBS_Q2_2025_baseline_20251218_095615` (23 columns in index)
- `gx1/wf_runs/SNIPER_OBS_Q2_2025_guarded_20251218_100625` (23 columns in index)
- `gx1/wf_runs/SNIPER_OBS_Q3_2025_baseline_20251218_105342` (23 columns in index)
- `gx1/wf_runs/SNIPER_OBS_Q3_2025_guarded_20251218_110643` (23 columns in index)
- `gx1/wf_runs/SNIPER_OBS_Q4_2025_baseline_20251218_111932` (23 columns in index)
- `gx1/wf_runs/SNIPER_OBS_Q4_2025_guarded_20251218_112754` (23 columns in index)

## Index columns (union across runs)

- `atr_bps`
- `distance_to_range`
- `entry_time`
- `execution_status`
- `exit_profile`
- `exit_reason`
- `exit_time`
- `guardrail_applied`
- `oanda_last_txn_id`
- `oanda_trade_id`
- `pnl_bps`
- `range_edge_dist_atr`
- `range_pos`
- `router_decision`
- `router_version`
- `session`
- `side`
- `source_chunk`
- `spread_bps`
- `trade_file`
- `trade_id`
- `trend_regime`
- `vol_regime`

## Field coverage per run

### `gx1/wf_runs/SNIPER_OBS_Q1_2025_baseline_20251218_124410`

| Field | Coverage |
| --- | --- |
| `trend_regime` | 99.9% |
| `vol_regime` | 99.9% |
| `atr_bps` | 99.9% |
| `spread_bps` | 99.8% |
| `session` | 99.9% |
| `range_pos` | 99.9% |
| `distance_to_range` | 99.9% |
| `range_edge_dist_atr` | 99.9% |
| `router_decision` | 0.0% |

### `gx1/wf_runs/SNIPER_OBS_Q1_2025_guarded_20251218_115831`

| Field | Coverage |
| --- | --- |
| `trend_regime` | 99.9% |
| `vol_regime` | 99.9% |
| `atr_bps` | 99.9% |
| `spread_bps` | 99.9% |
| `session` | 99.9% |
| `range_pos` | 99.9% |
| `distance_to_range` | 99.9% |
| `range_edge_dist_atr` | 99.9% |
| `router_decision` | 0.0% |

### `gx1/wf_runs/SNIPER_OBS_Q2_2025_baseline_20251218_095615`

| Field | Coverage |
| --- | --- |
| `trend_regime` | 100.0% |
| `vol_regime` | 100.0% |
| `atr_bps` | 100.0% |
| `spread_bps` | 99.9% |
| `session` | 100.0% |
| `range_pos` | 100.0% |
| `distance_to_range` | 100.0% |
| `range_edge_dist_atr` | 100.0% |
| `router_decision` | 0.0% |

### `gx1/wf_runs/SNIPER_OBS_Q2_2025_guarded_20251218_100625`

| Field | Coverage |
| --- | --- |
| `trend_regime` | 100.0% |
| `vol_regime` | 100.0% |
| `atr_bps` | 100.0% |
| `spread_bps` | 99.9% |
| `session` | 100.0% |
| `range_pos` | 100.0% |
| `distance_to_range` | 100.0% |
| `range_edge_dist_atr` | 100.0% |
| `router_decision` | 0.0% |

### `gx1/wf_runs/SNIPER_OBS_Q3_2025_baseline_20251218_105342`

| Field | Coverage |
| --- | --- |
| `trend_regime` | 99.9% |
| `vol_regime` | 99.9% |
| `atr_bps` | 99.9% |
| `spread_bps` | 99.9% |
| `session` | 99.9% |
| `range_pos` | 99.9% |
| `distance_to_range` | 99.9% |
| `range_edge_dist_atr` | 99.9% |
| `router_decision` | 0.0% |

### `gx1/wf_runs/SNIPER_OBS_Q3_2025_guarded_20251218_110643`

| Field | Coverage |
| --- | --- |
| `trend_regime` | 99.9% |
| `vol_regime` | 99.9% |
| `atr_bps` | 99.9% |
| `spread_bps` | 99.9% |
| `session` | 99.9% |
| `range_pos` | 99.9% |
| `distance_to_range` | 99.9% |
| `range_edge_dist_atr` | 99.9% |
| `router_decision` | 0.0% |

### `gx1/wf_runs/SNIPER_OBS_Q4_2025_baseline_20251218_111932`

| Field | Coverage |
| --- | --- |
| `trend_regime` | 99.9% |
| `vol_regime` | 99.9% |
| `atr_bps` | 99.9% |
| `spread_bps` | 99.9% |
| `session` | 99.9% |
| `range_pos` | 99.9% |
| `distance_to_range` | 99.9% |
| `range_edge_dist_atr` | 99.9% |
| `router_decision` | 0.0% |

### `gx1/wf_runs/SNIPER_OBS_Q4_2025_guarded_20251218_112754`

| Field | Coverage |
| --- | --- |
| `trend_regime` | 100.0% |
| `vol_regime` | 100.0% |
| `atr_bps` | 100.0% |
| `spread_bps` | 100.0% |
| `session` | 100.0% |
| `range_pos` | 100.0% |
| `distance_to_range` | 100.0% |
| `range_edge_dist_atr` | 100.0% |
| `router_decision` | 0.0% |

## Momentum / trend proxies

- `trend_regime` is present and can be used as a primary **momentum / trend proxy**.
- `vol_regime` and `atr_bps` are available for volatility/energy classification.

## Conclusion

**OK to classify regimes**: `trend_regime`, `vol_regime`, `atr_bps`, `spread_bps` have high coverage (>= 80%) in all audited runs.
