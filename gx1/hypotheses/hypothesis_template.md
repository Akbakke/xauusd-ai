# Hypothesis: [SHORT_NAME]

**Status:** [DRAFT | DESIGNED | IMPLEMENTED | VALIDATED | REJECTED]  
**Created:** YYYY-MM-DD  
**Target Baseline:** Trial160 (2020-2025, 29,710 trades)

---

## Target Bin(s)

**Edge Bins to Enhance:**
- [List specific edge bins from stable_edge_bins.json]
- Example: `OVERLAP|ATR:Q80-Q100|SPREAD:Q0-Q50|TREND:TREND_NEUTRAL|VOL:HIGH`

**Poison Bins to Avoid:**
- [List specific poison bins from poison_bins.json]
- Example: `US|ATR:Q80-Q100|SPREAD:Q50-Q80|TREND:TREND_UP|VOL:LOW`

---

## Expected Effect

**Primary Metric:**
- [e.g., PnL/trade ↑, MaxDD ↓, Winrate ↑]

**Quantitative Target:**
- [e.g., PnL/trade +2 bps, MaxDD -10%, Winrate +2%]

**Secondary Metrics:**
- [e.g., Trade count, ExitV2 triggers, Session breakdown]

---

## Kill Criteria

**Must NOT:**
- [e.g., MaxDD > baseline, Winrate < baseline - 1%, Trade count < 0.8x baseline]

**Must:**
- [e.g., PnL/trade > baseline, P5 > baseline, Stable edge bins maintain edge]

---

## Required Metrics

**Baseline (from truth_decomp):**
- Total PnL: [from TRUTH_DECOMP_EXECUTIVE.md]
- Winrate: [from TRUTH_DECOMP_EXECUTIVE.md]
- MaxDD: [from TRUTH_DECOMP_EXECUTIVE.md]
- Target bin PnL: [from stable_edge_bins.json]

**Hypothesis Variant:**
- [Same metrics, measured after A/B test]

---

## A/B Design

**ARM_A (Baseline):**
- Config: [exact baseline config]
- Expected: [baseline metrics]

**ARM_B (Hypothesis):**
- Config: [hypothesis variant config]
- Expected: [target metrics]

**Test Period:**
- [e.g., 2020-2025 full multiyear, or specific year]

**Success Criteria:**
- [e.g., ARM_B PnL > ARM_A PnL, ARM_B MaxDD <= ARM_A MaxDD]

---

## Implementation Plan

**Phase 1: Design**
- [ ] Define hypothesis
- [ ] Identify target bins
- [ ] Design A/B test

**Phase 2: Implementation**
- [ ] Implement with `GX1_HYPOTHESIS_*` flag
- [ ] Ensure baseline remains untouched
- [ ] Add telemetry/logging

**Phase 3: Validation**
- [ ] Run A/B test on monster-PC
- [ ] Compare metrics
- [ ] Document results

**Phase 4: Decision**
- [ ] Accept hypothesis → merge to baseline
- [ ] Reject hypothesis → archive
- [ ] Iterate → refine hypothesis

---

## Notes

[Additional context, observations, risks, etc.]
