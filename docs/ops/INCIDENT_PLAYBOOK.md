# SNIPER Incident Playbook

**Status**: Active  
**Last Updated**: 2025-12-21

---

## Weekly Health Report Schedule

**Frequency**: Every Monday (or first trading day of week)  
**Command**: `./scripts/run_weekly_health_report.sh <journal_dir> <start_date> <end_date>`

---

## Alarm Types

### 1. TAIL_RISK

**Trigger**: 
- P95 loss < -50 bps, OR
- Max loss < -200 bps

**Response**:
1. Inspect `Top 5 Worst Trades` in report
2. Check: `exit_reason`, `regime_class`, `session`, `atr_bps`, `spread_bps`
3. If isolated incident (< 3 trades): **No action** - monitor next week
4. If recurring pattern (> 3 trades): Review exit rules (no model changes)
5. If regime-specific: Consider overlay adjustment (policy-only)

**Policy adjustment allowed**: Yes (overlay multipliers/thresholds only)  
**Model changes**: No

---

### 2. REGIME_DRIFT

**Trigger**: 
- Regime distribution change > 10% vs baseline

**Response**:
1. Inspect `Regime Distribution` in report
2. Compare to baseline: `docs/ops/BASELINE.md`
3. If drift < 20%: **No action** - normal variation
4. If drift > 20%: Check data quality (missing bars, filter issues)
5. If persistent (> 2 weeks): Review regime classifier thresholds (no model changes)

**Policy adjustment allowed**: Yes (regime thresholds only)  
**Model changes**: No

---

### 3. COVERAGE

**Trigger**:
- Trades/day < 50, OR
- NO-TRADE rate > 5%

**Response**:
1. Inspect `Trades/day` and `Overlay Trigger Rates` in report
2. Check `Q4_A_TREND_SIZE` trigger rate (should be 0% if policy active)
3. If trades/day < 30: Check data availability, replay issues
4. If NO-TRADE > 10%: Review overlay gating logic (policy-only)
5. If isolated week: **No action** - monitor next week

**Policy adjustment allowed**: Yes (overlay enabled/disabled flags only)  
**Model changes**: No

---

## General Principles

1. **Policy changes only when data forces it**: No theoretical adjustments
2. **No model changes in production**: Models are frozen
3. **Overlay adjustments allowed**: Multipliers, thresholds, enable/disable flags
4. **Document all changes**: Update `docs/ops/BASELINE.md` if policy changes

---

## Escalation

If alarms persist > 2 weeks:
1. Review full quarter data (not just weekly)
2. Compare to baseline quarter metrics
3. Consider policy freeze (disable overlays temporarily)
4. Document decision in `docs/ops/BASELINE.md`

---

## Q4_A_TREND Policy

**Current**: DISABLE (NO-TRADE)  
**Revurdering**: Only if ALL conditions met (see `docs/policies/Q4_A_TREND_POLICY.md`)

**Do not change** unless:
- A_TREND count > 200 trades
- `base_units > 1` common enough for scaling
- Regime distribution changed significantly
- New quarters show consistent behavior

