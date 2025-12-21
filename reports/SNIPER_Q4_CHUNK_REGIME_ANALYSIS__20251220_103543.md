# Q4 Chunk Regime Analysis

**Generated**: 2025-12-20 10:35:43
**Run Directory**: `SNIPER_OBS_Q4_2025_baseline_overlay_20251218_152108`
**Source**: `parallel_chunks/**/trade_journal/trades/*.json`

---

## Summary

- **Total Trades**: 4810
- **EV/Trade**: 31.06 bps
- **Winrate**: 53.1%
- **Payoff**: 1.46
- **P90 Loss**: -9.69 bps

### Overlay Coverage

- **Trades with entry_snapshot**: 4780 / 4810 (99.4%)
- **Trades with sniper_overlays**: 0 / 4810 (0.0%)
- **Trades with Q4_C_CHOP_SESSION_SIZE overlay**: 0 / 4810 (0.0%)
- **Trades with overlay_applied == True**: 0 / 4810 (0.0%)

## Regime Breakdown

| Regime | Trades | EV/Trade | Winrate | Payoff | P90 Loss |
|--------|--------|----------|---------|--------|----------|
| A_TREND | 1250 | -8.40 | 47.4% | 0.98 | -15.84 |
| B_MIXED | 30 | 557.84 | 100.0% | 0.00 | 0.00 |
| C_CHOP | 3530 | 40.56 | 54.7% | 1.69 | -8.77 |

## Session Breakdown (Volume ≥ 200)

| Session | Trades | EV/Trade | Winrate | Payoff | P90 Loss |
|---------|--------|----------|---------|--------|----------|
| EU | 1711 | 16.28 | 51.2% | 1.25 | -10.38 |
| OVERLAP | 1217 | 31.21 | 55.0% | 1.40 | -18.26 |
| US | 1852 | 36.09 | 52.9% | 1.58 | -7.65 |

## Overlay Trigger Analysis

### Q4_C_CHOP_SESSION_SIZE Overlay

- **Trades with overlay metadata**: 0
- **Trades with overlay_applied == True**: 0
