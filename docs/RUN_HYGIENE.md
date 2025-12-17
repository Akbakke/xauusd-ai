# Run Hygiene Guide

**Purpose:** Keep `gx1/wf_runs/` clean and prevent accidental use of wrong baseline

## Quick Start

### Generate Inventory

```bash
python3 gx1/scripts/audit_runs_inventory.py
```

This creates:
- `gx1/wf_runs/_inventory.json` (detailed)
- `gx1/wf_runs/_inventory.csv` (summary)

### View Current Baseline

```bash
python3 gx1/scripts/audit_runs_inventory.py | grep -A 10 "CURRENT PROD_BASELINE"
```

### Find Delete Candidates

```bash
# From CSV
python3 << 'PY'
import pandas as pd
df = pd.read_csv('gx1/wf_runs/_inventory.csv')
candidates = df[
    (~df['has_trade_journal']) & 
    (df['size_mb'] < 1.0) &
    (df['mtime'] < '2025-12-10')  # Older than 7 days
].sort_values('size_mb')
print(candidates[['run_name', 'size_mb', 'mtime']].to_string())
PY
```

## Rules

### Never Delete

1. **PROD_BASELINE runs:** Any run with `meta.role == PROD_BASELINE`
2. **Last 10 runs:** Always keep newest 10 runs
3. **Milestone runs:** FULLYEAR, DETERMINISM_GATE, OBS_REPLAY_PROD_BASELINE
4. **Runs with trades:** If `n_trades > 0`, keep it

### Safe to Delete

1. **Empty runs:** No journal, size < 1 MB, age > 1 day
2. **Test runs:** `LIVE_FORCE_*` tags, no trades, age > 1 day
3. **Duplicates:** Same fingerprint, keep newest

### Archive (Don't Delete)

1. **Old runs:** >30 days, not PROD_BASELINE, has trades
2. **Tuning sweeps:** >60 days, not referenced

## Maintenance Schedule

- **Weekly:** Run audit, review candidates
- **Monthly:** Archive old runs
- **Quarterly:** Review PROD_BASELINE runs

## See Also

- `docs/PIPELINE_AUDIT.md` - Full pipeline documentation
- `gx1/scripts/audit_runs_inventory.py` - Inventory scanner

