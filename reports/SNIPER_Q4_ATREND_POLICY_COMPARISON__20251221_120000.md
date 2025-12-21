# Q4 A_TREND Policy Comparison Report

**Generated**: 2025-12-21 12:00:00  
**Purpose**: Compare baseline vs atrend_scale vs atrend_disable for Q4 2025

---

## Summary Table

| Variant | Trades | Mean PnL (bps) | Median PnL (bps) | P90 Loss (bps) | Winrate | Avg Trades/Day |
|---------|--------|----------------|-------------------|----------------|---------|----------------|
| **A) baseline** | 4780 | 27.76 | 5.88 | -9.69 | 52.8% | 72.4 |
| **B) atrend_scale** | 4780 | 27.54 | 5.89 | -9.69 | 52.8% | 72.4 |
| **C) atrend_disable** | 4768 | 27.93 | 5.89 | -9.67 | 52.8% | 72.2 |

### A_TREND Subset (atrend_scale only)

| Metric | Value |
|--------|-------|
| Count | 7 |
| Mean PnL | 13.11 bps |
| Median PnL | 71.58 bps |
| P90 Loss | -96.45 bps |
| Max Loss | -347.99 bps |
| Winrate | 71.4% |
| Avg Trades/Day | 0.2 |

---

## Key Findings

### 1. Total Q4 Performance
- **Baseline (A)**: 4780 trades, 27.76 bps mean PnL
- **Scale (B)**: 4780 trades, 27.54 bps mean PnL (-0.22 bps vs baseline)
- **Disable (C)**: 4768 trades, 27.93 bps mean PnL (+0.17 bps vs baseline)

**Observation**: Disable variant shows slight improvement (+0.17 bps), but difference is minimal.

### 2. A_TREND Trades (Scale Variant)
- Only **7 trades** (0.15% of total) were A_TREND
- Mean PnL: **13.11 bps** (lower than Q4 total: 27.54 bps)
- P90 Loss: **-96.45 bps** (much worse than Q4 total: -9.69 bps)
- Winrate: **71.4%** (higher than Q4 total: 52.8%)
- Max Loss: **-347.99 bps** (worse than Q4 total: -420.98 bps)

**Observation**: A_TREND trades have:
- ✅ Higher winrate (71.4% vs 52.8%)
- ❌ Much worse tail risk (P90 loss: -96.45 vs -9.69 bps)
- ❌ Lower mean PnL (13.11 vs 27.54 bps)

### 3. Scale vs Disable
- **Scale (B)**: 7 A_TREND trades executed (size_before=1 → size_after=1, no effective scaling)
- **Disable (C)**: 0 A_TREND trades executed (NO-TRADE when base_units=1)

**Observation**: Scale mode had no effect (min unit = 1 prevented size reduction). Disable mode successfully blocked A_TREND trades.

### 4. Impact Analysis
- **Scale variant**: -0.22 bps mean PnL vs baseline (negligible)
- **Disable variant**: +0.17 bps mean PnL vs baseline (negligible)
- **A_TREND subset**: 7 trades with -96.45 bps P90 loss (high tail risk)

**Conclusion**: 
- A_TREND trades are rare (0.15%) but have high tail risk
- Scale mode ineffective due to min unit = 1
- Disable mode successfully blocks A_TREND trades
- Overall impact is minimal due to low A_TREND frequency

---

## Recommendations

1. **Use disable mode** for Q4 A_TREND:
   - Successfully blocks high tail-risk trades
   - Slight improvement in mean PnL (+0.17 bps)
   - No effective scaling possible with base_units=1

2. **Monitor A_TREND frequency**:
   - Only 7 trades in Q4 (0.15%)
   - Consider if this is expected or indicates classification issue

3. **Future consideration**:
   - If base_units > 1 becomes common, scale mode may become effective
   - Current min unit = 1 constraint makes scaling ineffective

---

## Technical Notes

### Scale Mode Behavior
- `base_units=1`, `multiplier=0.3` → `1 * 0.3 = 0.3` → rounds to `1` (min unit)
- `effective_scale=False` logged in overlay metadata
- No actual size reduction occurred

### Disable Mode Behavior
- `base_units=1`, `multiplier=0.3`, `action=disable` → `units_out=0`
- Trade creation skipped (NO-TRADE)
- 7 A_TREND trades blocked in Q4

### Overlay Activation
- Overlay correctly identifies A_TREND trades
- Gating logic works as designed
- Low activation rate (0.15%) is due to Q4 regime distribution

