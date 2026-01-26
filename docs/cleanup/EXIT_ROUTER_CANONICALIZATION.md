# Exit Router Canonicalization

**Date:** 2026-01-08  
**Status:** ✅ Complete

## Rationale

Exit router models must be loaded from a canonical directory to ensure:
1. **Determinism:** Same model is always loaded, no ambiguity
2. **Security:** Prevents loading outdated or malicious models
3. **Auditability:** Clear source of truth for exit routing logic
4. **Consistency:** Same enforcement as entry bundles

## Implementation

### Canonical Directory

**Location:** `gx1/models/exit_router/`

**Files:**
- `exit_router_v3_tree.pkl` - Active V3 decision tree model (referenced in 27+ policies)
- `exit_router_tree.pkl` - Legacy V1/V2 model (for reference)

### Guardrail

**Location:** `gx1/core/hybrid_exit_router.py:222-250`

**Behavior:**
- **Default:** Blocks loading from non-canonical paths with `RuntimeError("EXIT_ROUTER_NON_CANONICAL_PATH")`
- **Override:** Set `GX1_ALLOW_NON_CANONICAL_EXIT_ROUTER=1` to allow (logs WARNING)
- **Canonical paths:** Always allowed

**Code:**
```python
# Guardrail: Enforce canonical directory for exit router models
canonical_exit_router_dir = Path(__file__).parent.parent / "models" / "exit_router"
model_path_resolved = model_path.resolve()
canonical_dir_resolved = canonical_exit_router_dir.resolve()

if not is_canonical:
    allow_non_canonical = os.environ.get("GX1_ALLOW_NON_CANONICAL_EXIT_ROUTER", "0")
    if allow_non_canonical != "1":
        raise RuntimeError(
            f"EXIT_ROUTER_NON_CANONICAL_PATH: Attempted to load exit router model from non-canonical path '{model_path}'. "
            f"Exit router models must be loaded from canonical directory: {canonical_exit_router_dir}. "
            f"If you really need to load from a non-canonical path (e.g., for testing), "
            f"set GX1_ALLOW_NON_CANONICAL_EXIT_ROUTER=1. "
            f"This is a security and determinism requirement."
        )
```

### Policy Updates

**Before:**
```yaml
exit_router:
  model_path: gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/exit_router_models_v3/exit_router_v3_tree.pkl
```

**After:**
```yaml
exit_router:
  model_path: gx1/models/exit_router/exit_router_v3_tree.pkl
```

**Updated:** 27 policy files

### Hash Verification

**Files Scanned:** 4 exit_router*.pkl files  
**Hash Duplicates:** 2 groups (archived)  
**Canonical Files:**
- `exit_router_v3_tree.pkl`: SHA256 `e728786d982b36b47036dac4b4e3be24e6c3c3041105503f46343dcb26700516`
- `exit_router_tree.pkl`: SHA256 `86fd7df0e4bb35058ad7a6a319e08dc7e0f1e82585c859b9e79a8002d32a0ce3`

## Verification

### Unit Tests

**File:** `gx1/tests/test_exit_router_canonical.py`

**Tests:**
- ✅ `test_exit_router_non_canonical_path_fails` - Non-canonical path raises RuntimeError
- ✅ `test_exit_router_canonical_path_ok` - Canonical path works
- ✅ `test_exit_router_allow_non_canonical_with_flag` - Override flag works

**Result:** All 3 tests passing

### Runtime Verification

**Command:**
```bash
python gx1/scripts/verify_runtime_after_archive.py \
  --policy gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml
```

**Status:** ✅ All artifacts verified, no exit_router warnings

## Migration

### Files Moved

1. `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/exit_router_models_v3/exit_router_v3_tree.pkl`
   → `gx1/models/exit_router/exit_router_v3_tree.pkl`

2. `gx1/analysis/exit_router_models/exit_router_tree.pkl`
   → `gx1/models/exit_router/exit_router_tree.pkl`

### Policies Updated

27 policy files updated to use canonical path:
- All `sniper_snapshot/` policies
- All `active/` policies
- `prod_snapshot/` policies

### Default Path Updated

**Before:**
```python
model_path = Path(__file__).parent.parent / "analysis" / "exit_router_models_v3" / "exit_router_v3_tree.pkl"
```

**After:**
```python
model_path = Path(__file__).parent.parent / "models" / "exit_router" / "exit_router_v3_tree.pkl"
```

## Benefits

1. **No More Warnings:** Eliminated "exit_router outside canonical dirs" warnings
2. **Determinism:** Same model always loaded, no path ambiguity
3. **Security:** Cannot accidentally load outdated models
4. **Consistency:** Same canonical enforcement as entry bundles
5. **Auditability:** Clear source of truth for exit routing

## Related Documents

- `reports/cleanup/EXIT_ROUTER_CANONICALIZATION_20260108_100426.md` - Full migration report
- `gx1/scripts/canonicalize_exit_router.py` - Migration script
- `gx1/tests/test_exit_router_canonical.py` - Unit tests

## Future Maintenance

- **Adding new exit router models:** Place in `gx1/models/exit_router/`
- **Updating models:** Replace file in canonical dir, update SHA256 in docs
- **Testing:** Use `GX1_ALLOW_NON_CANONICAL_EXIT_ROUTER=1` for test models only

---

**Mission Status:** ✅ Complete - Exit routing is now as deterministic and audit-safe as entry bundles.
