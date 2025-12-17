# GX1/FARM Cleanup Summary

**Generated:** 2025-12-17  
**Purpose:** Summary of files that can be safely deleted or archived

---

## Executive Summary

### Safe to Delete (Can be regenerated)
- **Python cache:** `__pycache__/` directories and `.pyc` files
- **Old logs:** Log files older than 7 days
- **Temporary files:** `*.tmp`, `*.temp`, `*.swp`, `*.bak`, `*~`

### Archive (Don't Delete)
- **Old runs:** Runs older than 30 days, not PROD_BASELINE
- **Old sweeps:** Tuning sweeps older than 60 days

### Do NOT Delete
- **Code files:** `gx1/`, `scripts/`, `tests/`
- **Config files:** `gx1/configs/`
- **Documentation:** `docs/`
- **Model files:** `gx1/models/`
- **Prod snapshots:** `gx1/configs/policies/prod_snapshot/`
- **KEEP runs:** Defined in `gx1/wf_runs/_KEEP.txt`
- **Runs with trade journal**
- **Unknown runs:** Missing `run_header.json` or role (fail-closed)

---

## Cleanup Commands

### 1. Delete Python Cache (Safe)
```bash
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete
```

### 2. Delete Old Logs (>7 days)
```bash
find gx1/live/logs -name "*.log" -mtime +7 -delete
find logs -name "*.log" -mtime +7 -delete
find gx1/wf_runs -name "*.log" -mtime +7 -delete
```

### 3. Delete Temporary Files
```bash
find . -name "*.tmp" -delete
find . -name "*.temp" -delete
find . -name "*.swp" -delete
find . -name "*.bak" -delete
find . -name "*~" -delete
```

### 4. Archive Old Runs (>30 days, not PROD_BASELINE)
```bash
mkdir -p gx1/wf_runs/_archive
find gx1/wf_runs -maxdepth 1 -type d -mtime +30 ! -name "_*" -exec sh -c '
  run_dir="$1"
  [ -f "$run_dir/run_header.json" ] || exit 0
  role=$(jq -r ".meta.role // \"unknown\"" "$run_dir/run_header.json" 2>/dev/null || echo "unknown")
  [ "$role" = "PROD_BASELINE" ] && exit 0
  echo "Archiving: $run_dir (role=$role)"
  mv "$run_dir" gx1/wf_runs/_archive/
' _ {} \;
```

### 5. Verify After Cleanup
```bash
python3 gx1/scripts/audit_runs_inventory.py --infer-role --write-inventory
```

---

## Current Status

- **Total runs:** 118
- **Unknown runs:** 94 (DO NOT DELETE - fail-closed)
- **KEEP runs:** 24
- **PROD_BASELINE runs:** 0 (0 inferred)

---

**Last Updated:** 2025-12-17

