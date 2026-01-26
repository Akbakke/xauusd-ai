# Truth Baseline Lock: Trial160 (2020-2025)

**Status:** LOCKED  
**Date:** 2026-01-19  
**Baseline:** Trial160, 29,710 trades across 2020-2025

---

## What Is Locked

### Trading Logic (DO NOT MODIFY WITHOUT HYPOTHESIS FLAG)

**Entry:**
- ENTRY_V10++/CTX with temperature scaling
- Pre-entry gates: WAIT_GATE (optional), OVERLAP overlay (optional)
- Threshold logic: Baseline thresholds (from policy config)

**Exit:**
- Exit Policy V2 (MAE/MFE based)
- Legacy exits (as configured)

**Gates:**
- Spread guard, ATR guard, Cost gate, Window gate
- Pre-entry wait gate (optional)

**⚠️  NO CHANGES TO GATES, THRESHOLDS, OR EXITS WITHOUT:**
1. Explicit hypothesis flag (`GX1_HYPOTHESIS_*`)
2. A/B test design
3. Monster-PC validation

---

## What Is Proven (Truth Baseline)

### Hard Facts (from `reports/truth_decomp/TRUTH_DECOMP_EXECUTIVE.md`)

1. **Total PnL:** 216,108.44 bps over 29,710 trades
2. **Overall Winrate:** 94.2%
3. **Avg PnL per Trade:** 7.27 bps
4. **Session PnL:**
   - EU: 76,303 bps (10,828 trades)
   - OVERLAP: 92,604 bps (12,099 trades)
   - US: 47,202 bps (6,783 trades)
5. **Edge Bins:** 50 (positive avg + good tail)
6. **Poison Bins:** 1 (negative avg + bad tail)
7. **Stable Edge Bins:** 130 (reliable across 2020-2025)
8. **Unstable Bins:** 0
9. **OVERLAP Pattern:** Winners holdes 2.5x lenger (7.4 vs 3.0 bars)
10. **Best Separator:** Max MAE (bps) with score 0.21

### Edge Map (from `reports/truth_decomp/TRUTH_DECOMP_SESSION_REGIME_MATRIX.md`)

- **Total EDGE BINS:** 241
- **Total POISON BINS:** 1
- **Top Edge Bin:** OVERLAP with 13.04 bps avg, 100% winrate
- **Per Session:** EU=86, OVERLAP=76, US=79 edge bins

### Stability (from `reports/truth_decomp/TRUTH_DECOMP_STABILITY_2020_vs_2025.md`)

- **Stable Edge Bins:** 130 (positive in both 2020 and 2025, small delta)
- **Unstable Bins:** 0 (no sign flips or large changes)
- **Drift Bins:** 11 (positive in both years but larger delta)

---

## What Is NOT Proven (Explicit Gaps)

### Not Proven (Requires Hypothesis Testing)

1. **Pre-entry veto for poison bins**
   - Hypothesis: Block trades matching poison bin characteristics
   - Status: NOT TESTED
   - Required: A/B test with explicit hypothesis flag

2. **Threshold modulation in poison bins**
   - Hypothesis: Higher threshold in poison bins improves winrate
   - Status: NOT TESTED
   - Required: A/B test with explicit hypothesis flag

3. **Exit policy tweaks for delayed edge bins**
   - Hypothesis: Delayed edge bins need longer hold times
   - Status: NOT TESTED
   - Required: A/B test with explicit hypothesis flag

4. **Feature engineering for separation**
   - Hypothesis: Top separators can improve entry model
   - Status: NOT TESTED
   - Required: Model retraining + A/B test

5. **Stable edge bin focus**
   - Hypothesis: Increase trade frequency in stable edge bins
   - Status: NOT TESTED
   - Required: A/B test with explicit hypothesis flag

---

## What Is Forbidden (Before Monster-PC)

### ❌ DO NOT (Mac Phase)

1. **Implement new gates on Mac**
   - Wait for monster-PC for full multiyear validation
   - Mac is for analysis only

2. **Add TCN models**
   - Not enough evidence yet
   - Requires full model retraining

3. **Modify entry thresholds without A/B testing**
   - Baseline thresholds are locked
   - All changes must be A/B tested

4. **Run new sweeps on Mac**
   - Too slow
   - Use monster-PC for sweeps

5. **Change exit policy without stability analysis**
   - Exit Policy V2 is locked
   - Changes require stability validation

6. **Consume stable_edge_bins.json or poison_bins.json at runtime**
   - These are for analysis only
   - DO NOT use for runtime gating/filtering
   - See: `gx1/hypotheses/README.md`

---

## Next Rational Steps (Later)

### Hypothesis-Driven Improvements (Monster-PC)

1. **Selective Pre-Entry Veto for Quick-Fail Regimes**
   - Target: Poison bins with high quick-fail rate
   - Test: Pre-entry gate blocking poison bin characteristics
   - Expected: Reduce losers without cutting winners

2. **Threshold Modulation in Poison Bins**
   - Target: Poison bins (1 identified)
   - Test: Increase threshold by 0.05-0.10 in poison bins
   - Expected: Better trade selection, fewer losers

3. **Exit Policy Tweaks for Delayed Edge Bins**
   - Target: Edge bins with delayed payoff (76.9% in Q40-Q60)
   - Test: Extend minimum hold time or reduce early exit probability
   - Expected: More winners reach payoff, better PnL

4. **Stable Edge Bin Focus**
   - Target: 130 stable edge bins
   - Test: Increase trade frequency in stable bins, reduce in unstable bins
   - Expected: More consistent performance across years

5. **Feature Engineering for Separation**
   - Target: Top separators (Max MAE, Bars Held, ATR)
   - Test: Add separator features to entry model training
   - Expected: Better winner/loser separation, improved winrate

---

## Registry Files (Read-Only)

### `gx1/hypotheses/stable_edge_bins.json`
- **Purpose:** Registry of 130 stable edge bins (2020 vs 2025)
- **Usage:** Analysis only, hypothesis design
- **⚠️  DO NOT CONSUME AT RUNTIME (Mac phase)**

### `gx1/hypotheses/poison_bins.json`
- **Purpose:** Registry of 1 poison bin
- **Usage:** Analysis only, hypothesis design
- **⚠️  DO NOT CONSUME AT RUNTIME (Mac phase)**

---

## Invariants (Hard Requirements)

### Analysis Scripts Must Enforce:

1. **Year Coverage:**
   - FATAL if < 2 years detected
   - FATAL if any requested year has 0 trades

2. **Unit Sanity:**
   - FATAL if spread_bps outside [0, 500]
   - FATAL if atr_bps outside [0, 2000]
   - FATAL if non-finite values found

3. **Trade Count:**
   - WARN if trades < 1000 (but don't fail)
   - FATAL if trades == 0

4. **No Silent Fallback:**
   - All errors must be explicit
   - No default values that mask problems

---

## Code Markers

### Files with LOCKED_TRUTH_BASELINE markers:

1. `gx1/scripts/run_truth_decomposition_2020_2025.py`
   - Header comment with lock notice

2. `gx1/execution/entry_manager.py`
   - Header comment with lock notice
   - Baseline configuration documented

3. `gx1/scripts/run_overlap_sanity_pack_multiyear.py`
   - Header comment with lock notice

### Analysis Scripts with Invariants:

1. `gx1/scripts/build_truth_decomp_trade_table.py`
   - Year coverage invariant
   - Unit sanity invariant
   - Trade count warning

2. `gx1/scripts/audit_truth_input_root.py`
   - Year coverage invariant (< 2 years = FATAL)

---

## References

- **Truth Baseline:** `reports/truth_decomp/`
- **Executive Summary:** `reports/truth_decomp/TRUTH_DECOMP_EXECUTIVE.md`
- **Edge Map:** `reports/truth_decomp/TRUTH_DECOMP_SESSION_REGIME_MATRIX.md`
- **Stability:** `reports/truth_decomp/TRUTH_DECOMP_STABILITY_2020_vs_2025.md`
- **Hypotheses:** `gx1/hypotheses/`

---

**Last Updated:** 2026-01-19  
**Lock Status:** ACTIVE  
**Next Review:** After monster-PC hypothesis validation
