# V9 Model Cleanup Changelog

**Date:** 2026-01-08  
**Git Commit:** `2a79bcdfee56cdc6c92586d4b069bb2b15fb758b`  
**Status:** ✅ Guardrail implemented, Archive script ready

## Rationale

V9 models are deprecated in favor of V10/V10_CTX. To ensure determinism and prevent accidental loading of outdated models:

1. **V9 models are FORBIDDEN by default** - Require explicit env flag to load
2. **Only canonical model is kept** - `nextgen_2020_2025_clean/model.pt` (40+ policy references)
3. **Non-referenced models archived** - 3 models (23.94 MB) marked for archive

## Implementation

### Guardrail

**Location (SSoT):** `gx1/models/entry_v9/entry_v9_bundle.py:load_entry_v9_bundle()`
**Also used by:** `gx1/execution/oanda_demo_runner.py:_load_entry_v9_model()` (delegates to bundle loader)

**Behavior:**
- **Default:** Blocks V9 model.pt loading with `RuntimeError("V9_MODEL_LOAD_FORBIDDEN")`
- **Override:** Set `GX1_ALLOW_V9_MODEL_LOAD=1` to allow (logs WARNING)
- **Rationale:** V9 models are deprecated, V10/V10_CTX should be used instead

**Code:**
```python
# Guardrail: Enforce V9 model.pt loading policy
allow_v9_load = os.environ.get("GX1_ALLOW_V9_MODEL_LOAD", "0")
if allow_v9_load != "1":
    raise RuntimeError(
        f"V9_MODEL_LOAD_FORBIDDEN: Attempted to load V9 model from '{model_path}'. "
        f"V9 models are deprecated in favor of V10/V10_CTX. "
        f"If you really need to load a V9 model (e.g., for legacy policy compatibility), "
        f"set GX1_ALLOW_V9_MODEL_LOAD=1. "
        f"This is a determinism and security requirement."
    )
```

### Models Status

**KEEP (2 files):**
1. `gx1/models/entry_v9/nextgen_2020_2025_clean/model.pt` (7.38 MB)
   - **SHA256:** `e28d673b49f2baaeaf3e5e1b80831a01b4e11cb8884ec00ee8f8cc7a4465015f`
   - **References:** 40 canonical policies, 19 code references
   - **Status:** Active, used by legacy policies

2. `gx1/models/entry_v9/nextgen_2020_2025_clean/checkpoint_epoch_10.pt` (7.40 MB)
   - **SHA256:** `f214fac65b697c6a72a016b325a38213480b047943787f1a662b005ff99d7db0`
   - **References:** 40 canonical policies, 13 code references
   - **Status:** Checkpoint (handled by checkpoint guardrail)

**ARCHIVE CANDIDATES (3 files, 23.94 MB):**
1. `gx1/models/entry_v9/nextgen_2020_2025_clean_shuffle/model.pt` (7.38 MB)
   - **SHA256:** `17df97e249bec47465c4b2ec43fa4ee0b266c9e3b7a9729027d01e146b526b8d`
   - **References:** 0 canonical policies, 10 code references (generic paths)
   - **Status:** NOT_REFERENCED in runnable policies

2. `gx1/models/entry_v9/nextgen_2020_2025_optuna/model.pt` (8.04 MB)
   - **SHA256:** `7f5f42b1bcf3fc313db6f1ada7fc616a5b3d92971d0921d64f54657b013b1615`
   - **References:** 0 canonical policies, 10 code references (generic paths)
   - **Status:** NOT_REFERENCED in runnable policies

3. `gx1/models/entry_v9/nextgen_2020_2025_full/model.pt` (8.52 MB)
   - **SHA256:** `53bc6d8fc2359290a8c4489dea1d56ee3ce19b51c960bf4352562ba0b5855c18`
   - **References:** 0 canonical policies, 10 code references (generic paths)
   - **Status:** NOT_REFERENCED in runnable policies

## Archive Script

**Location:** `scripts/cleanup/ARCHIVE_V9_MODELS_20260108_101428.sh`

**Usage:**
```bash
# Dry run (preview only)
./scripts/cleanup/ARCHIVE_V9_MODELS_20260108_101428.sh --dry-run

# Actual archive
./scripts/cleanup/ARCHIVE_V9_MODELS_20260108_101428.sh
```

**Archive Location:** `_archive_artifacts/v9_models_cleanup_20260108_101417/`

**Total Size:** 23.94 MB

## Verification

### Unit Tests

**File:** `gx1/tests/test_v9_model_guardrail.py`

**Tests:**
- ✅ `test_v9_model_load_forbidden_by_default` - Non-canonical path raises RuntimeError
- ✅ `test_v9_model_load_allowed_with_flag` - Override flag works
- ✅ `test_final_models_not_affected` - Entry V10/V10_CTX unaffected

### Runtime Verification

**Command:**
```bash
python gx1/scripts/verify_runtime_after_archive.py \
  --policy gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_TRAIN_V10_CTX_GATED.yaml
```

**Status:** ✅ All artifacts verified (V9 models not loaded unless GX1_ALLOW_V9_MODEL_LOAD=1)

## Migration Steps

### Before Archive

1. **Verify guardrail is active:**
   ```bash
   # Should fail without flag
   python -c "from gx1.execution.oanda_demo_runner import GX1DemoRunner; ..."
   ```

2. **Run dry-run:**
   ```bash
   ./scripts/cleanup/ARCHIVE_V9_MODELS_20260108_101428.sh --dry-run
   ```

### Archive Execution

1. **Execute archive:**
   ```bash
   ./scripts/cleanup/ARCHIVE_V9_MODELS_20260108_101428.sh
   ```

2. **Verify runtime:**
   ```bash
   python gx1/scripts/verify_runtime_after_archive.py \
     --policy gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_TRAIN_V10_CTX_GATED.yaml
   ```

3. **Run smoke tests:**
   - Verify entry_v10/entry_v10_ctx still work
   - Verify legacy V9 policies fail without GX1_ALLOW_V9_MODEL_LOAD=1

### After 7 Days

**If all tests pass and no regressions:**

1. **Verify no regressions:**
   - Run minimal smoke test/replay
   - Verify entry_v10/entry_v10_ctx still work
   - Verify legacy V9 policies work with GX1_ALLOW_V9_MODEL_LOAD=1

2. **Delete from archive:**
   ```bash
   rm -rf _archive_artifacts/v9_models_cleanup_20260108_101417/
   ```

3. **Never delete from repo directly** - only from archive

## Related Documents

- `reports/cleanup/V9_MODEL_REAL_OVERVIEW_20260108_101227.md` - Full inventory report
- `reports/cleanup/V9_MODEL_REAL_OVERVIEW_20260108_101227.json` - JSON inventory
- `reports/cleanup/V9_MODEL_REAL_OVERVIEW_20260108_101227.csv` - CSV inventory
- `gx1/scripts/audit_v9_models.py` - Audit script
- `gx1/scripts/generate_v9_archive_script.py` - Archive script generator
- `gx1/tests/test_v9_model_guardrail.py` - Unit tests

## Notes

- **V9 models are deprecated** - Use V10/V10_CTX for new policies
- **Legacy policies** - May still use V9, but require `GX1_ALLOW_V9_MODEL_LOAD=1`
- **Scalers/feature_meta** - Not affected by guardrail (only model.pt is guarded)
- **Checkpoints** - Handled separately by checkpoint guardrail

---

## Archive Tombstone / Receipt

**Archive Date:** 2026-01-08  
**Archive Location:** `_archive_artifacts/v9_models_cleanup_20260108_101428/`  
**Total Size:** 24 MB (3 files)  
**Planned Deletion Date:** 2026-01-15 (7 days after archive)

### Archived Files (with SHA256)

| Original Path | Size (MB) | SHA256 | Status |
|---------------|-----------|--------|--------|
| `gx1/models/entry_v9/nextgen_2020_2025_clean_shuffle/model.pt` | 7.38 | `17df97e249bec47465c4b2ec43fa4ee0b266c9e3b7a9729027d01e146b526b8d` | ✅ Archived |
| `gx1/models/entry_v9/nextgen_2020_2025_optuna/model.pt` | 8.04 | `7f5f42b1bcf3fc313db6f1ada7fc616a5b3d92971d0921d64f54657b013b1615` | ✅ Archived |
| `gx1/models/entry_v9/nextgen_2020_2025_full/model.pt` | 8.52 | `53bc6d8fc2359290a8c4489dea1d56ee3ce19b51c960bf4352562ba0b5855c18` | ✅ Archived |

### Execution Commands

**1. Dry-run (preview):**
```bash
./scripts/cleanup/ARCHIVE_V9_MODELS_20260108_101428.sh --dry-run
```
**Output:** Would move 3 files (23.94 MB)

**2. Execute archive:**
```bash
./scripts/cleanup/ARCHIVE_V9_MODELS_20260108_101428.sh
```
**Output:**
```
[INFO] Creating archive directory: _archive_artifacts/v9_models_cleanup_20260108_101428
[OK] Moved: gx1/models/entry_v9/nextgen_2020_2025_clean_shuffle/model.pt
[OK] Moved: gx1/models/entry_v9/nextgen_2020_2025_optuna/model.pt
[OK] Moved: gx1/models/entry_v9/nextgen_2020_2025_full/model.pt
[SUMMARY] Moved 3 files
```

**3. Runtime verification:**
```bash
python gx1/scripts/verify_runtime_after_archive.py \
  --policy gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_TRAIN_V10_CTX_GATED.yaml
```
**Output:** ✅ All artifacts verified successfully

**4. Reference check:**
```bash
grep -r "nextgen_2020_2025_full/model\.pt|nextgen_2020_2025_optuna/model\.pt|nextgen_2020_2025_clean_shuffle/model\.pt" .
```
**Output:** References found only in cleanup reports/scripts (no runtime references)

### Planned Deletion

**Date:** 2026-01-15 (7 days after archive)  
**Script:** `scripts/cleanup/DELETE_V9_MODELS_ARCHIVE_20260108_101428.sh`

**Delete-Gate Commands (MUST RUN BEFORE DELETION):**

```bash
# 1. Runtime verification (generates report for delete script)
python gx1/scripts/verify_runtime_after_archive.py \
  --policy gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_TRAIN_V10_CTX_GATED.yaml \
  --out_dir reports/cleanup

# 2. Unit tests
pytest -q gx1/tests/test_v9_model_guardrail.py

# 3. (Optional) Mini replay-smoke
# python -m gx1.execution.oanda_demo_runner --policy ... --dry-run
```

**All commands must pass before deletion is allowed.**
**The delete script will check for the verification report and enforce date/path checks.**

**Deletion Command:**
```bash
./scripts/cleanup/DELETE_V9_MODELS_ARCHIVE_20260108_101428.sh
```

**Prerequisites (enforced by script):**
- ✅ Current date >= 2026-01-15
- ✅ Runtime verification passed (report file exists with recent timestamp)
- ✅ Archive path matches expected: `_archive_artifacts/v9_models_cleanup_20260108_101428/`
- ✅ All delete-gate commands passed

---

**Mission Status:** ✅ COMPLETE - Guardrail in SSoT, Archive executed, Verification passed  
**Archive Status:** ✅ 3 models (24 MB) archived, ready for deletion after 7 days
