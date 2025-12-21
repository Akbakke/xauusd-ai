# SNIPER Golden Baseline

**Status**: ✅ LOCKED (2025-12-21)

---

## Baseline Policy

**File**: `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY.yaml`

**Commit Hash**: `8349c4b3a73088456810618fc6fb29c0cf1e6d4d`

**Policy Decisions**:
- Q4_A_TREND overlay: `action="disable"` (NO-TRADE for Q4 × A_TREND)
- See `docs/policies/Q4_A_TREND_POLICY.md` for rationale

---

## Rules

1. **Read-only**: Baseline policy is frozen. No changes without explicit approval.
2. **Experimentation**: All experiments must be done on feature branches.
3. **Production**: Only baseline policy is used in production.
4. **Changes**: Policy changes require:
   - A/B test results
   - Risk assessment
   - Explicit approval
   - Update to this baseline document

---

## Reference Metrics (Q4 2025)

- **Trades**: 4780
- **Mean PnL**: 27.51 bps
- **Median PnL**: 5.88 bps
- **P90 Loss**: -9.69 bps
- **Winrate**: 52.8%
- **A_TREND trades**: 0 (blocked by policy)

---

## Update History

- **2025-12-21**: Baseline locked with Q4_A_TREND disable policy

