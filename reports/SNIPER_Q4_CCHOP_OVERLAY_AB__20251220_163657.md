# Q4 C_CHOP Overlay A/B Evaluation

**Generated**: 2025-12-20 16:36:57
**Run A (Overlay OFF)**: `SNIPER_OBS_Q4_2025_baseline_no_cchop_20251220_152945`
**Run B (Overlay ON)**: `SNIPER_OBS_Q4_2025_baseline_20251220_151442`
**Source**: `parallel_chunks/**/trade_journal/trades/*.json`

---

## Summary

### Q4 Total

| Metric | Run A (OFF) | Run B (ON) | Delta |
|--------|-------------|------------|-------|
| Trades | 4802 | 4797 | -5 |
| EV/Trade | 30.06 bps | 29.52 bps | -0.55 bps |
| Winrate | 53.0% | 53.0% | -0.0% |
| Payoff | 1.45 | 1.44 | -0.01 |
| P90 Loss | -9.69 bps | -9.69 bps | +0.00 bps |

### Q4 × C_CHOP

| Metric | Run A (OFF) | Run B (ON) | Delta |
|--------|-------------|------------|-------|
| Trades | 3530 | 3530 | +0 |
| EV/Trade | 40.25 bps | 40.69 bps | +0.44 bps |
| Winrate | 54.7% | 54.7% | +0.0% |
| Payoff | 1.69 | 1.70 | +0.01 |
| P90 Loss | -8.77 bps | -8.77 bps | +0.00 bps |

### Q4 × C_CHOP × US (Expected Largest Effect)

| Metric | Run A (OFF) | Run B (ON) | Delta |
|--------|-------------|------------|-------|
| Trades | 1504 | 1504 | +0 |
| EV/Trade | 43.77 bps | 44.85 bps | +1.08 bps |
| Winrate | 52.7% | 52.7% | +0.0% |
| Payoff | 1.89 | 1.91 | +0.02 |
| P90 Loss | -7.08 bps | -7.08 bps | +0.00 bps |

## Overlay Activation Sanity (Run B Only)

- **Trades with overlay_applied == True**: 1118 (23.3%)

### Distribution per Session

| Session | Count |
|---------|-------|
| OVERLAP | 265 |
| US | 853 |

### Size Verification

- **Size checks**: 1118
- **Size mismatches**: 0
- **Match rate**: 100.0%

## Conclusions

✅ **EV/Trade increased** in C_CHOP×US: +1.08 bps
➡️  **P90 Loss unchanged** in C_CHOP×US
✅ **Overlay activated** for 1118 trades (23.3%)
✅ **Size calculation verified**: 100.0% match rate
