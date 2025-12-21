# Runtime Regime Inputs - Data Sources

This document describes where each regime classification input comes from at runtime, as used by SNIPER overlays.

## Priority Order (First Non-None Value Wins)

### session
1. `prediction.session` (if prediction object has `.session` attribute)
2. `policy_state.get("session")` (from Big Brain V1 or policy state)
3. `infer_session_tag(entry_time)` (computed from timestamp)
4. `"UNKNOWN"` (fallback)

### trend_regime
1. `prediction.trend_regime` (if prediction object has `.trend_regime` attribute)
2. `policy_state.get("brain_trend_regime")` (from Big Brain V1)
3. `policy_state.get("trend_regime")` (from policy state)
4. `feature_context.get("trend_regime")` (from feature context)
5. `None` (fallback)

### vol_regime
1. `prediction.vol_regime` (if prediction object has `.vol_regime` attribute)
2. `policy_state.get("brain_vol_regime")` (from Big Brain V1)
3. `policy_state.get("vol_regime")` (from policy state)
4. `feature_context.get("vol_regime")` (from feature context)
5. `None` (fallback)

### atr_bps
1. `current_atr_bps` (computed from `entry_bundle.atr_bps` in `evaluate_entry()`)
2. `feature_context.get("atr_bps")` (from feature context)
3. `entry_bundle.atr_bps` (from entry bundle)
4. `None` (fallback)

### spread_bps
1. Computed from `spread_pct` if present: `spread_pct * 10000.0`
2. `feature_context.get("spread_bps")` (from feature context)
3. `feature_context.get("spread_pct") * 10000.0` (if spread_pct exists in feature_context)
4. `entry_bundle.spread_bps` (from entry bundle)
5. `None` (fallback)

## Implementation

The extractor function `get_runtime_regime_inputs()` in `gx1/sniper/policy/runtime_regime_inputs.py` implements this priority order.

## Usage in Entry Manager

In `gx1/execution/entry_manager.py`, the extractor is called before overlay application:

```python
regime_inputs = get_runtime_regime_inputs(
    prediction=prediction,
    feature_context=feature_context_dict,
    spread_pct=spread_pct_for_overlay,
    current_atr_bps=current_atr_bps,
    entry_bundle=entry_bundle,
    policy_state=policy_state_snapshot,
    entry_time=now_ts,
)
```

The extracted inputs are then passed to all overlays:
- `apply_size_overlay()`
- `apply_q4_cchop_overlay()`
- `apply_q4_atrend_overlay()`

## Storage in Entry Snapshot

The same regime inputs are stored in `entry_snapshot` for consistency:
- `entry_snapshot["trend_regime"]` = `regime_inputs["trend_regime"]`
- `entry_snapshot["vol_regime"]` = `regime_inputs["vol_regime"]`
- `entry_snapshot["session"]` = `regime_inputs["session"]`
- `entry_snapshot["atr_bps"]` = `regime_inputs["atr_bps"]`
- `entry_snapshot["spread_bps"]` = `regime_inputs["spread_bps"]`

This ensures offline analysis sees the same regime data as runtime overlays.

