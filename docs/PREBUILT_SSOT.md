# Prebuilt Features SSoT (Single Source of Truth)

**Last Updated:** 2026-02-03

## Overview

This document defines the canonical prebuilt features for GX1 TRUTH/SMOKE runs. The prebuilt cleanup system ensures that only verified, clean prebuilt files are used in production runs.

## Canonical Prebuilt Paths

### 2025
- **Canonical Symlink:** `/home/andre2/GX1_DATA/data/data/prebuilt/TRIAL160/2025/xauusd_m5_2025_features_v10_ctx.parquet`
- **Target:** `/home/andre2/GX1_DATA/data/data/prebuilt/TRIAL160/2025/xauusd_m5_2025_features_v10_ctx.v12ab_clean_20260203_111254.parquet`
- **SHA256:** `02e6c47a932997546e1b14c4d3559ab027eec582818b8ece0af286aba5b0f144`
- **Schema:** V12AB91 (91 features), no `prob_*` columns
- **Features:** FIXMIDATR + V12AB

### 2024
- **Canonical Symlink:** `/home/andre2/GX1_DATA/data/data/prebuilt/TRIAL160/2024/xauusd_m5_2024_features_v10_ctx.parquet`
- **Target:** `/home/andre2/GX1_DATA/data/data/prebuilt/TRIAL160/2024/xauusd_m5_2024_features_v10_ctx.v12ab_clean_20260203_162833.parquet`
- **SHA256:** `f2f20cbf6917d8a8fc4d91c8f791f93351c92c25a3f0f9d5c6e513e8dbdb387d`
- **Schema:** V12AB91 (92 features, includes extra non-required features), no `prob_*` columns
- **Features:** FIXMIDATR + V12AB

## Symlink Policy

**TRUTH/SMOKE runs MUST use canonical symlinks only.**

- The canonical symlink is the single source of truth for each year
- Direct paths to parquet files are forbidden in TRUTH/SMOKE
- The `GX1_CANONICAL_PREBUILT_PATH` environment variable can be set to override, but must still point to canonical
- The `--data` argument in replay scripts must point to the canonical symlink
- **Prebuilt Identity Gate** enforces this policy (hard-fail if non-canonical prebuilt is used)

## Prebuilt Identity Gate

The Prebuilt Identity Gate (`gx1/utils/prebuilt_identity_gate.py`) enforces:

1. **Canonical Check:** Prebuilt must be canonical symlink (or resolve to canonical target)
2. **Schema Check:** Prebuilt must NOT contain `prob_*` columns
3. **File Integrity:** Prebuilt file must exist and be readable

**Gate Execution Order (in `replay_eval_gated_parallel.py`):**
1. Syntax Gate
2. Env Identity Gate
3. Bundle Identity Gate
4. **Prebuilt Identity Gate** ← Runs before workers start

**Failure Behavior:**
- Writes `PREBUILT_IDENTITY_FATAL.json` capsule
- Exits with code 2 (hard-fail)
- Prevents replay from starting

## Cleanup System

### Generating Cleanup Plan

```bash
python3 gx1/scripts/prebuilt_cleanup_plan.py \
  --base-dir /home/andre2/GX1_DATA/data/data/prebuilt \
  --output-dir /home/andre2/GX1_DATA/reports/prebuilt_cleanup
```

**Output:**
- `PREBUILT_CLEANUP_PLAN.json` - Machine-readable plan
- `PREBUILT_CLEANUP_PLAN.md` - Human-readable report

**Tripwires:**
- HARD FAIL if more than one canonical symlink per year
- HARD FAIL if winner contains `prob_short` in schema
- HARD FAIL if canonical symlink points to non-existent file

### Applying Cleanup

```bash
# Dry-run (default)
python3 gx1/scripts/apply_prebuilt_cleanup.py \
  --plan /home/andre2/GX1_DATA/reports/prebuilt_cleanup/PREBUILT_CLEANUP_PLAN.json \
  --quarantine-root /home/andre2/GX1_DATA/quarantine/PREBUILT_CLEANUP_$(date +%Y%m%d_%H%M%S)/

# Apply (requires explicit flag)
python3 gx1/scripts/apply_prebuilt_cleanup.py \
  --plan /home/andre2/GX1_DATA/reports/prebuilt_cleanup/PREBUILT_CLEANUP_PLAN.json \
  --quarantine-root /home/andre2/GX1_DATA/quarantine/PREBUILT_CLEANUP_$(date +%Y%m%d_%H%M%S)/ \
  --apply
```

**Output:**
- `APPLY_PREBUILT_CLEANUP_REPORT.json` - Machine-readable report
- `APPLY_PREBUILT_CLEANUP_REPORT.md` - Human-readable report

**Tripwires:**
- Never uses `rm` (all moves are reversible)
- Hard-fail if `mv` would overwrite existing file in quarantine
- Hard-fail if canonical symlink changes target during apply

## Restore from Quarantine

To restore files from quarantine:

```bash
# Example: Restore a specific file
QUARANTINE_DIR="/home/andre2/GX1_DATA/quarantine/PREBUILT_CLEANUP_20260203_120000"
ORIGINAL_PATH="/home/andre2/GX1_DATA/data/data/prebuilt/TRIAL160/2025/old_file.parquet"
QUARANTINE_PATH="$QUARANTINE_DIR/data/data/prebuilt/TRIAL160/2025/old_file.parquet"

# Restore
mkdir -p "$(dirname "$ORIGINAL_PATH")"
mv "$QUARANTINE_PATH" "$ORIGINAL_PATH"
```

**Note:** Quarantine directory structure preserves original paths for easy restore.

## Tripwires Summary

### Prebuilt Cleanup Plan
1. **Max one canonical per year:** HARD FAIL if multiple canonical symlinks found
2. **Winner must be clean:** HARD FAIL if winner contains `prob_*` columns
3. **Canonical must exist:** HARD FAIL if canonical symlink points to non-existent file

### Apply Cleanup
1. **No rm:** Never uses `rm`, only `mv` (reversible)
2. **No overwrite:** HARD FAIL if quarantine destination exists
3. **Canonical stability:** HARD FAIL if canonical symlink target changes during apply

### Prebuilt Identity Gate
1. **Canonical required:** HARD FAIL if prebuilt is not canonical in TRUTH/SMOKE
2. **Schema clean:** HARD FAIL if prebuilt contains `prob_*` columns
3. **File exists:** HARD FAIL if prebuilt file does not exist

## Related Documentation

- `docs/PROOF_V1_SSOT.md` - Overall SSoT for Proof v1
- `gx1/utils/prebuilt_identity_gate.py` - Gate implementation
- `gx1/scripts/prebuilt_cleanup_plan.py` - Cleanup plan generator
- `gx1/scripts/apply_prebuilt_cleanup.py` - Cleanup executor

## Maintenance

### Adding New Year

1. Build clean prebuilt (no `prob_*` columns, V12AB schema)
2. Create canonical symlink: `xauusd_m5_<YEAR>_features_v10_ctx.parquet`
3. Point symlink to clean prebuilt file
4. Update this document with canonical path
5. Run cleanup plan to quarantine old files

### Verifying Canonical

```bash
# Check symlink
ls -l /home/andre2/GX1_DATA/data/data/prebuilt/TRIAL160/2025/xauusd_m5_2025_features_v10_ctx.parquet

# Verify target exists
readlink -f /home/andre2/GX1_DATA/data/data/prebuilt/TRIAL160/2025/xauusd_m5_2025_features_v10_ctx.parquet

# Check schema (no prob_*)
python3 -c "import pyarrow.parquet as pq; schema = pq.ParquetFile('$(readlink -f /path/to/canonical)').schema_arrow; cols = [schema.field(i).name for i in range(len(schema))]; print('prob_* columns:', [c for c in cols if c.startswith('prob_')])"
```

---

*This SSoT ensures that TRUTH/SMOKE runs use only verified, clean prebuilt features with no schema drift or contamination.*
