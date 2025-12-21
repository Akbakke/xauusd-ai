# Q4 A_TREND Overlay A/B Evaluation

**Generated**: 2025-12-20 17:07:33
**Run A (Overlay OFF)**: `SNIPER_OBS_Q4_2025_baseline_no_atrend_gate_20251220_155238`
**Run B (Overlay ON)**: `SNIPER_OBS_Q4_2025_baseline_atrend_gate_20251220_155932`
**Source**: `parallel_chunks/**/trade_journal/trades/*.json`

---

## Summary

### Q4 Total

| Metric | Run A (OFF) | Run B (ON) | Delta |
|--------|-------------|------------|-------|
| Trades | 4808 | 4809 | +1 |
| EV/Trade | 30.55 bps | 31.18 bps | +0.63 bps |
| Winrate | 53.1% | 53.1% | +0.0% |
| Payoff | 1.45 | 1.46 | +0.01 |
| P90 Loss | -9.69 bps | -9.69 bps | +0.00 bps |

### Q4 × A_TREND

| Metric | Run A (OFF) | Run B (ON) | Delta |
|--------|-------------|------------|-------|
| Trades | 1250 | 1250 | +0 |
| EV/Trade | -8.63 bps | -8.40 bps | +0.22 bps |
| Winrate | 47.4% | 47.4% | +0.0% |
| Payoff | 0.98 | 0.98 | +0.00 |
| P90 Loss | -15.84 bps | -15.84 bps | +0.00 bps |

### Q4 × A_TREND × Session (Min 200 Trades)

#### EU

| Metric | Run A (OFF) | Run B (ON) | Delta |
|--------|-------------|------------|-------|
| Trades | 605 | 605 | +0 |
| EV/Trade | -23.06 bps | -23.06 bps | +0.00 bps |
| P90 Loss | -14.39 bps | -14.39 bps | +0.00 bps |

#### OVERLAP

| Metric | Run A (OFF) | Run B (ON) | Delta |
|--------|-------------|------------|-------|
| Trades | 297 | 297 | +0 |
| EV/Trade | 13.37 bps | 13.37 bps | +0.00 bps |
| P90 Loss | -17.84 bps | -17.84 bps | +0.00 bps |

#### US

| Metric | Run A (OFF) | Run B (ON) | Delta |
|--------|-------------|------------|-------|
| Trades | 348 | 348 | +0 |
| EV/Trade | -2.31 bps | -1.51 bps | +0.80 bps |
| P90 Loss | -21.96 bps | -21.96 bps | +0.00 bps |

## Overlay Activation Sanity (Run B Only)

- **Trades with overlay_applied == True**: 0 (0.0%)

### Distribution per Session

| Session | Count |
|---------|-------|

### Size Verification

- **Size checks**: 0
- **Size mismatches**: 0
- **Match rate**: 0.0%

## Conclusions

✅ **EV/Trade increased** in A_TREND: +0.22 bps
➡️  **P90 Loss unchanged** in A_TREND
✅ **Total EV/Trade increased**: +0.63 bps
⚠️  **Overlay not activated** for any trades

### Critical Issue: Overlay Not Activating

**Problem**: Overlay metadata exists for 4780 trades (99.4%), but `overlay_applied == True` for 0 trades.

**Root Cause**: Overlay receives `atr_bps=None` and `spread_bps=None` at runtime (from `entry_manager.py` defensive variables), causing `classify_regime()` to classify all trades as B_MIXED or C_CHOP (default fallback when regime inputs are missing).

**Evidence**:
- Overlay reasons: `not_a_trend:B_MIXED` (3009), `not_a_trend:C_CHOP` (1771)
- All trades have `quarter=Q4` (correct)
- No trades have `regime_class=A_TREND` in overlay metadata
- Direct classification of trades shows ~1250 A_TREND trades, but overlay cannot see them due to missing `atr_bps`/`spread_bps`

**Impact**: Overlay cannot activate because it cannot correctly identify A_TREND trades at runtime.

**Note**: This is a data availability issue, not an overlay logic issue. The overlay correctly gates on `regime_class != A_TREND` when regime inputs are missing.
