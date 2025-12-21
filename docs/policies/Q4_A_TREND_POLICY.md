# Q4 A_TREND Policy Decision

**Date**: 2025-12-21  
**Status**: ✅ LOCKED (Default: DISABLE)

---

## Summary

Q4 × A_TREND trades are blocked (NO-TRADE) by default due to high tail risk. Scale mode is ineffective with `base_units=1` (min unit constraint prevents size reduction).

---

## Data Foundation

### A/B Test Results (Q4 2025)

| Variant | Trades | Mean PnL (bps) | P90 Loss (bps) |
|---------|--------|----------------|----------------|
| Baseline | 4780 | 27.76 | -9.69 |
| Scale | 4780 | 27.54 | -9.69 |
| Disable | 4768 | 27.93 | -9.67 |

### A_TREND Subset (Scale Variant)

- **Count**: 7 trades (0.15% of Q4 total)
- **Mean PnL**: 13.11 bps (lower than Q4 total: 27.54 bps)
- **P90 Loss**: -96.45 bps (much worse than Q4 total: -9.69 bps)
- **Winrate**: 71.4% (higher than Q4 total: 52.8%)
- **Max Loss**: -347.99 bps

**Key Finding**: A_TREND trades have:
- ✅ Higher winrate (71.4% vs 52.8%)
- ❌ Much worse tail risk (P90 loss: -96.45 vs -9.69 bps)
- ❌ Lower mean PnL (13.11 vs 27.54 bps)

---

## Why Scale Mode Doesn't Work

Scale mode (`action="scale"`, `multiplier=0.30`) is ineffective because:
- `base_units=1` (typical for SNIPER)
- `1 * 0.30 = 0.3` → rounds to `1` (min unit constraint)
- No actual size reduction occurs
- `effective_scale=False` in all cases

**Conclusion**: Scale mode cannot reduce size when `base_units=1`.

---

## Final Decision: DISABLE (NO-TRADE)

**Action**: `action="disable"` (default)

**Behavior**: Q4 × A_TREND → `units_out=0` → trade creation skipped

**Rationale**:
1. High tail risk: P90 loss = -96.45 bps vs Q4 total = -9.69 bps
2. Scale mode ineffective: min unit = 1 prevents size reduction
3. Low frequency: Only 7 trades (0.15%), so blocking has minimal impact
4. Slight improvement: Disable variant shows +0.17 bps mean PnL vs baseline

**Policy Logging**:
- `action="disable"`
- `reason="Q4_A_TREND_high_tail_risk"`
- `size_before_units`, `size_after_units` (0), `multiplier`
- `trend_regime`, `vol_regime`, `atr_bps`, `spread_bps`
- All logged in `sniper_overlays[]` for audit trail

---

## When to Revisit

This policy should be reconsidered **ONLY** if **ALL** of the following conditions are met:

1. **A_TREND frequency**: A_TREND count in new dataset > 200 trades (vs current ~7 in Q4)
2. **Base units**: `base_units > 1` occurs frequently enough that scaling would have measurable effect
3. **Regime shift**: Q4 regime distribution changes significantly (A_TREND % increases substantially)
4. **Data validation**: New quarters show consistent A_TREND behavior (not just noise)

**Review cadence**: Quarterly (with each new quarter's OOS data)

**Review process**:
1. Run A/B test comparing disable vs scale modes
2. Verify scale mode effectiveness (check `effective_scale=True` rate)
3. Measure tail risk impact (P90/P95 loss comparison)
4. Get explicit approval before changing policy

**Current status**: Policy is locked. Do not change without meeting all criteria above.

---

## Implementation

- **Overlay**: `gx1/sniper/policy/sniper_q4_atrend_size_overlay.py`
- **Default action**: `"disable"`
- **Entry manager**: Returns `None` when `units_out=0` (NO-TRADE)
- **Logging**: Full policy decision logged in `entry_snapshot.sniper_overlays[]`

---

## References

- A/B Test Report: `reports/SNIPER_Q4_ATREND_POLICY_COMPARISON__20251221_120000.md`
- Verification Report: `reports/SNIPER_Q4_ATREND_OVERLAY_VERIFICATION__20251221_100500.md`

