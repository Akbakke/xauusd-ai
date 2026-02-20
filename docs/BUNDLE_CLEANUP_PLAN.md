# Bundle Cleanup Plan

**Generated:** 2026-02-03  
**Purpose:** Plan for cleaning up deprecated bundle directories and moving them to quarantine.

## Active Canonical Bundle

**Path:** `/home/andre2/GX1_DATA/models/models/entry_v10_ctx/FULLYEAR_2024_2025_V12AB91`  
**Status:** ✅ CANONICAL (SSoT for TRUTH replay)  
**Lock:** `MASTER_MODEL_LOCK.json` present  
**Version:** v12ab91  
**Sessions:** EU, OVERLAP

## Deprecated Bundles

### FULLYEAR_2025_GATED_FUSION
- **Path:** `/home/andre2/GX1_DATA/models/models/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION`
- **Status:** ⚠️ DEPRECATED
- **Reason:** Replaced by FULLYEAR_2024_2025_V12AB91 (96 features vs 91 features, includes prob_*)
- **Lock:** `MASTER_MODEL_LOCK.json` present
- **Action:** Move to quarantine

### FULLYEAR_2025_GATED_FUSION_TEST
- **Path:** `/home/andre2/GX1_DATA/models/models/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION_TEST`
- **Status:** ⚠️ DEPRECATED
- **Reason:** Test bundle, not production
- **Lock:** None
- **Action:** Move to quarantine

### FULLYEAR_2025_BASELINE_NO_GATE
- **Path:** `/home/andre2/GX1_DATA/models/models/entry_v10_ctx/FULLYEAR_2025_BASELINE_NO_GATE`
- **Status:** ⚠️ DEPRECATED
- **Reason:** Baseline bundle, not used in production
- **Lock:** None
- **Action:** Move to quarantine

### SMOKE_20260106_ctxfusion
- **Path:** `/home/andre2/GX1_DATA/models/models/entry_v10_ctx/SMOKE_20260106_ctxfusion`
- **Status:** ⚠️ DEPRECATED
- **Reason:** Smoke test bundle, not production
- **Lock:** None
- **Action:** Move to quarantine

## Quarantine Move Commands

**⚠️ DO NOT EXECUTE WITHOUT EXPLICIT APPROVAL**

```bash
# Create quarantine directory with timestamp
QUARANTINE_DIR="/home/andre2/GX1_DATA/quarantine/BUNDLES_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$QUARANTINE_DIR"

# Move deprecated bundles
mv /home/andre2/GX1_DATA/models/models/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION "$QUARANTINE_DIR/"
mv /home/andre2/GX1_DATA/models/models/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION_TEST "$QUARANTINE_DIR/"
mv /home/andre2/GX1_DATA/models/models/entry_v10_ctx/FULLYEAR_2025_BASELINE_NO_GATE "$QUARANTINE_DIR/"
mv /home/andre2/GX1_DATA/models/models/entry_v10_ctx/SMOKE_20260106_ctxfusion "$QUARANTINE_DIR/"

# Verify canonical bundle remains
ls -la /home/andre2/GX1_DATA/models/models/entry_v10_ctx/FULLYEAR_2024_2025_V12AB91
```

## Verification After Cleanup

1. **Canonical bundle exists:**
   ```bash
   test -f /home/andre2/GX1_DATA/models/models/entry_v10_ctx/FULLYEAR_2024_2025_V12AB91/MASTER_MODEL_LOCK.json
   ```

2. **Only canonical bundle remains:**
   ```bash
   ls -1 /home/andre2/GX1_DATA/models/models/entry_v10_ctx/
   # Should show only: FULLYEAR_2024_2025_V12AB91
   ```

3. **Quarantine contains moved bundles:**
   ```bash
   ls -1 "$QUARANTINE_DIR"
   # Should show all moved bundles
   ```

## Notes

- **No deletions:** This plan only moves bundles to quarantine, never deletes them.
- **Backup:** Ensure backup exists before executing moves.
- **TRUTH/SMOKE enforcement:** After cleanup, `GX1_CANONICAL_BUNDLE_DIR` must be set to canonical bundle path in all TRUTH/SMOKE runs.

---
*This is a cleanup plan only. Execute moves only after explicit approval.*
