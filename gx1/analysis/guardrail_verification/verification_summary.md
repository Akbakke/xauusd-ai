# Guardrail Verification Summary

## DEL 1: High Precision Metrics

### Baseline
- N: 162
- sum(pnl_bps): 19062.039720
- mean(pnl_bps): 117.666912
- winrate: 0.845679

### Guardrail
- N: 162
- sum(pnl_bps): 19062.039720
- mean(pnl_bps): 117.666912
- winrate: 0.845679

### Differences
- sum(pnl_bps): 0.000000
- mean(pnl_bps): 0.000000
- winrate: 0.000000

## DEL 2: Overridden Trades

- Count: 42
- mean(delta_pnl_bps): 0.000000
- median(delta_pnl_bps): 0.000000
- delta == 0 (exact): 42
- delta == 0 (within 1e-6): 42

