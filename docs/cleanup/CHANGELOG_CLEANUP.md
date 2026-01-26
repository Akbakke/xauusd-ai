# Cleanup Changelog

This document tracks all cleanup operations performed on the repository, including artifact archival and deletion.

---

## 2026-01-08 09:49:40 - Checkpoint NO-RESUME Cleanup (Final Deletion)

**Git Commit:** `2a79bcdfee56cdc6c92586d4b069bb2b15fb758b`  
**Status:** ✅ Deleted from archive

### What Was Deleted

**3 checkpoint files (25M total):**
1. `gx1/models/entry_v9/nextgen_2020_2025_full/checkpoint_epoch_10.pt` (8.54 MB)
2. `gx1/models/entry_v9/nextgen_2020_2025_full/checkpoint_epoch_20.pt` (8.54 MB)
3. `gx1/models/entry_v9/nextgen_2020_2025_optuna/checkpoint_epoch_10.pt` (8.05 MB)

### Why

**NO-RESUME Policy:**
- Checkpoints contain optimizer state and training-specific state that may not match current codebase
- Training runs are not resumed in practice
- Final models (model.pt, model_state_dict.pt) are the canonical artifacts
- Checkpoints add complexity without clear benefit
- Risk of accidental loading of outdated optimizer states

### Guardrails Implemented

**CHECKPOINT_LOAD_FORBIDDEN Guardrail:**
- **Location:** `gx1/models/entry_v10/entry_v10_bundle.py`
- **Behavior:**
  - Default: Blocks loading of `checkpoint_epoch_*.pt` files with `RuntimeError`
  - Override: Set `GX1_ALLOW_CHECKPOINT_LOAD=1` to allow (for training resume only)
  - Final models: `model.pt` and `model_state_dict.pt` are always allowed
- **Implementation:**
  - Applied in `load_entry_v10_bundle()` (line 226-238)
  - Applied in `load_entry_v10_ctx_bundle()` (line 543-554)
- **Unit Test:** `gx1/tests/test_checkpoint_guardrail.py` (3 tests, all passing)

### Archive History

- **2026-01-07 22:53:53:** Checkpoints archived to `_archive_artifacts/checkpoints_no_resume_20260107_225353/`
- **2026-01-08 09:49:40:** Archive deleted after verification

### Verification

✅ **Unit Tests:** All 3 tests passing  
✅ **Runtime Verification:** All 8 artifacts verified, no checkpoint references  
✅ **Canary Test:** Runtime does not attempt to load checkpoints

### Related Documents

- `reports/cleanup/CHECKPOINTS_NO_RESUME_DECISION_20260107_225353.md` - Full decision rationale
- `scripts/cleanup/ARCHIVE_CHECKPOINTS_NO_RESUME_20260107_225353.sh` - Archive script (executed)
- `gx1/tests/test_checkpoint_guardrail.py` - Guardrail unit tests

---
