# GX1 Hypotheses Registry

**Purpose:** Structured registry for hypothesis-driven improvements to Trial160 baseline.

**Status:** Scaffold only (no runtime coupling yet)

---

## Principles

1. **All improvements must be hypothesis-driven**
   - Target specific edge/poison bins from truth baseline
   - Expected effect must be measurable
   - Kill criteria must be defined upfront

2. **No changes without A/B testing**
   - Baseline must remain untouched
   - Hypothesis variants must be isolated
   - Results must be compared against baseline

3. **Monster-PC validation required**
   - Mac is for analysis only
   - Full multiyear validation on monster-PC
   - No production changes without validation

---

## Structure

- `hypothesis_template.md` - Template for new hypotheses
- `hypotheses/` - Individual hypothesis documents (future)
- `stable_edge_bins.json` - Read-only registry (DO NOT CONSUME AT RUNTIME)
- `poison_bins.json` - Read-only registry (DO NOT CONSUME AT RUNTIME)

---

## Usage

1. Copy `hypothesis_template.md` to create new hypothesis
2. Fill in all sections (target bins, expected effect, kill criteria, etc.)
3. Design A/B test
4. Implement with explicit `GX1_HYPOTHESIS_*` flag
5. Run validation on monster-PC
6. Document results

---

## ⚠️  DO NOT CONSUME AT RUNTIME (Mac phase)

The stable_edge_bins.json and poison_bins.json files are for **analysis only**.
They must NOT be used for runtime gating or filtering until:
- Hypothesis is validated
- A/B test passes
- Monster-PC validation complete
