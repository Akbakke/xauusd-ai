# SNIPER Ops Package Summary

**Date**: 2025-12-21  
**Status**: ✅ Implemented

---

## Files Created/Modified

### Baseline & Policy Lock
- ✅ `docs/ops/BASELINE.md` - Golden baseline documentation with commit hash
- ✅ `docs/policies/Q4_A_TREND_POLICY.md` - Updated with explicit revurderingsregler

### Health Reporting
- ✅ `gx1/sniper/analysis/weekly_health_report.py` - Weekly health report script
- ✅ `scripts/run_weekly_health_report.sh` - Shell wrapper for health report

---

## Usage Examples

### Weekly Health Report

```bash
# Report for last week (example dates)
./scripts/run_weekly_health_report.sh \
  gx1/wf_runs/SNIPER_OBS_Q4_2025_baseline_20251221_135105 \
  2025-10-01 \
  2025-10-07

# With CSV output
./scripts/run_weekly_health_report.sh \
  gx1/wf_runs/SNIPER_OBS_Q4_2025_baseline_20251221_135105 \
  2025-10-01 \
  2025-10-07 \
  reports/weekly_health_2025-10-01_to_2025-10-07.csv

# With baseline regime distribution for drift detection
./scripts/run_weekly_health_report.sh \
  gx1/wf_runs/SNIPER_OBS_Q4_2025_baseline_20251221_135105 \
  2025-10-01 \
  2025-10-07 \
  reports/weekly_health_2025-10-01_to_2025-10-07.csv \
  docs/ops/baseline_regime_dist_q4_2025.json
```

### Direct Python Usage

```bash
PYTHONPATH=. python gx1/sniper/analysis/weekly_health_report.py \
  --journal-root gx1/wf_runs/SNIPER_OBS_Q4_2025_baseline_20251221_135105 \
  --start-date 2025-10-01 \
  --end-date 2025-10-07 \
  --output-csv reports/weekly_health.csv
```

---

## Operational Alarms

The health report checks for three types of alarms:

1. **COVERAGE**: Trades/day < 50 or NO-TRADE rate > 5%
2. **TAIL_RISK**: P95 loss < -50 bps or max loss < -200 bps
3. **REGIME_DRIFT**: Regime distribution change > 10% vs baseline

Alarms are logged as: `ALARM: <TYPE> <details>`

Script exits with code 1 if alarms are present (for CI/cron integration).

---

## Baseline Reference

- **Commit**: `8349c4b3a73088456810618fc6fb29c0cf1e6d4d`
- **Policy**: `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY.yaml`
- **Q4 Reference Metrics**: See `docs/ops/BASELINE.md`

---

## Q4_A_TREND Revurderingsregler

Policy can only be reconsidered if **ALL** conditions are met:
1. A_TREND count > 200 trades (vs current ~7)
2. `base_units > 1` occurs frequently enough for scaling to be effective
3. Q4 regime distribution changes significantly
4. New quarters show consistent A_TREND behavior

See `docs/policies/Q4_A_TREND_POLICY.md` for full details.

---

## Run Mode: PROD vs SHADOW

### Configuration

Set `run_mode` in policy YAML:
- `run_mode: "PROD"` - Production mode (real orders)
- `run_mode: "SHADOW"` - Shadow mode (journal trades, no real orders)

### When to Use SHADOW

1. **Before policy changes**: Test new overlay settings in SHADOW before PROD
2. **Weekly validation**: Run SHADOW alongside PROD to validate behavior
3. **Regime testing**: Test overlay behavior in new regimes without risk

### Behavior

**SHADOW mode**:
- ✅ Trades are journaled (full logging)
- ✅ Weekly reports run identically
- ✅ Overlays apply normally
- ❌ No real orders sent to broker

**PROD mode**:
- ✅ Trades are journaled
- ✅ Real orders sent to broker
- ✅ Weekly reports run identically

### Switching Modes

1. Edit policy YAML: `run_mode: "SHADOW"` or `run_mode: "PROD"`
2. Restart runner
3. Verify mode in logs: `[RUN_MODE] SHADOW` or `[RUN_MODE] PROD`

---

## Incident Playbook

See `docs/ops/INCIDENT_PLAYBOOK.md` for:
- Weekly report schedule
- Alarm response procedures
- Policy adjustment guidelines
- Escalation process
