# Canary Mode - Production Testing

**Last Updated:** 2025-12-15

---

## Overview

Canary mode runs the full production pipeline with `dry_run=True` (no actual orders) and verifies all invariants. This allows testing production code paths without risking real capital.

---

## Usage

### Policy-based Canary Mode

Set `mode: CANARY` in policy YAML:

```yaml
mode: "CANARY"
meta:
  role: PROD_BASELINE
```

### CLI Flag (Future)

```bash
python gx1/execution/oanda_demo_runner.py --canary --policy <path>
```

---

## What Happens in Canary Mode

1. **Dry Run Enforced:**
   - `dry_run=True` is automatically set (even if not in config)
   - No orders are sent to broker
   - All trading logic executes normally

2. **Invariant Verification:**
   - Range features computed before router ✅
   - Guardrail active ✅
   - Router model loading verified ✅
   - Feature manifest validation passed ✅
   - Policy lock verified ✅

3. **Metrics Generation:**
   - `prod_metrics.csv` generated at end of run
   - `alerts.json` generated if thresholds triggered
   - Terminal summary printed

4. **Logging:**
   - All invariants logged with `[CANARY]` prefix
   - Full pipeline execution logged
   - No suppression of debug/info logs

---

## Example Canary Run

```bash
# Using canary policy
bash scripts/run_replay.sh \
  gx1/prod/current/policy_canary.yaml \
  2025-01-01 2025-01-07 7 \
  gx1/wf_runs/CANARY_TEST_2025_Q1

# Expected output:
# [CANARY] Canary mode enabled - dry_run=True, all invariants will be logged
# ... (full pipeline execution) ...
# [CANARY] Invariant Verification Summary
# [CANARY] ✅ Range features computed before router (EntryManager)
# [CANARY] ✅ Guardrail active (ExitModeSelector)
# [CANARY] ✅ Router model loading verified
# [CANARY] ✅ Feature manifest validation passed (if enabled)
# [CANARY] ✅ Policy lock verified (no changes detected)
# [CANARY] All invariants verified - canary run successful
# [CANARY] ✅ Prod metrics generated: gx1/wf_runs/.../prod_metrics.csv
```

---

## Artifacts Generated

- `run_header.json`: SHA256 hashes of all artifacts
- `trade_log*.csv`: Trade log (as normal)
- `prod_metrics.csv`: Production metrics
- `alerts.json`: Alerts if thresholds triggered (optional)

---

## Verification Checklist

After canary run, verify:

- ✅ All invariants passed (check `[CANARY]` logs)
- ✅ `prod_metrics.csv` generated
- ✅ `run_header.json` contains all artifact hashes
- ✅ No alerts in `alerts.json` (or alerts are expected)
- ✅ Trade log contains expected trades
- ✅ Range features present in `trade.extra`

---

## Integration with CI/CD

Canary mode can be integrated into CI/CD pipeline:

```bash
# CI/CD canary test
python gx1/execution/oanda_demo_runner.py \
  --policy gx1/prod/current/policy_canary.yaml \
  --replay-csv test_data.parquet \
  --fast-replay

# Check exit code and alerts
if [ -f gx1/wf_runs/*/alerts.json ]; then
    echo "Alerts triggered - check alerts.json"
    exit 1
fi
```

---

## Notes

- Canary mode uses same code paths as production
- All safety checks (kill switch, policy lock) are active
- Model loading uses PROD_BASELINE fail-closed behavior
- Feature manifest validation blocks trading on mismatch (in PROD_BASELINE)

