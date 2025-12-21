# FARM PROD_BASELINE Lock

**Status:** FROZEN (2025-12-15)  
**Purpose:** Anti-regression guard - FARM baseline must NEVER change behavior  
**Scope:** All files referenced by PROD_BASELINE policy snapshot

---

## FARM PROD_BASELINE Policy Path

**Primary Policy:**
```
gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml
```

**Entry Config:**
```
gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID/ENTRY_V9_FARM_V2B_PROD.yaml
```

**Exit Configs:**
```
gx1/configs/exits/FARM_EXIT_V2_RULES_A_mp68_max5_rule5_cd0.yaml
gx1/configs/exits/FARM_EXIT_V2_RULES_ADAPTIVE_v1.yaml
```

**Router Model:**
```
gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/exit_router_models_v3/exit_router_v3_tree.pkl
```

---

## Baseline Fingerprint

**Policy SHA256:** `d9da2c864eb7767840fde0b812170ae127a4f5ac6f8302afafa9e8da0cd887e4`

**Entry Config SHA256:** `fbd5b9a0187f21a0ac0a3291e621fb0faccc42a48959cda78bb2f7ea8284aa48`

**Exit Config SHA256 (RULE5):** `dcccecfd260396c2c08e5312a38e004d8eae2ef45ddeb464989b52aacd531122`

**Router Model SHA256:** `e728786d982b36b47036dac4b4e3be24e6c3c3041105503f46343dcb26700516`

**Router Version:** `HYBRID_ROUTER_V3`

**Guardrail Cutoff:** `v3_range_edge_cutoff: 1.0`

**Role:** `PROD_BASELINE`

---

## Verification Commands

### 1. Verify Run Matches FARM Baseline

```bash
python3 gx1/analysis/prod_baseline_proof.py \
  --run gx1/wf_runs/<RUN_TAG> \
  --prod-policy gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml \
  --out gx1/wf_runs/<RUN_TAG>/prod_baseline_proof.md
```

**Expected Result:** All SHA256 hashes match ✅

### 2. Check FARM Lock (Local Guard)

```bash
./scripts/check_farm_lock.sh
```

**Expected Result:** All locked files match expected hashes ✅

---

## Locked Files (Never Modify)

The following files are **LOCKED** and must NEVER be modified:

1. **Policy YAML:**
   - `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml`

2. **Entry Config:**
   - `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID/ENTRY_V9_FARM_V2B_PROD.yaml`

3. **Exit Configs:**
   - `gx1/configs/exits/FARM_EXIT_V2_RULES_A_mp68_max5_rule5_cd0.yaml`
   - `gx1/configs/exits/FARM_EXIT_V2_RULES_ADAPTIVE_v1.yaml`

4. **Router Model:**
   - `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/exit_router_models_v3/exit_router_v3_tree.pkl`

5. **Shared Engine Code (Default Behavior):**
   - `gx1/policy/exit_farm_v2_rules.py` (RULE5/RULE6A implementation)
   - `gx1/policy/exit_hybrid_controller.py` (Router implementation)
   - `gx1/policy/farm_guards.py` (FARM guard functions - `farm_brutal_guard`, `farm_brutal_guard_v2`)

**Note:** Engine code can be extended (new functions), but default behavior for FARM must not change.

---

## SNIPER Isolation

**SNIPER** has its own policy bundle under:
```
gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/
```

SNIPER can:
- ✅ Have its own entry/exit configs
- ✅ Have its own router parameters (cutoffs, thresholds)
- ✅ Tune thresholds without affecting FARM

SNIPER must NOT:
- ❌ Modify FARM PROD_BASELINE files
- ❌ Change default behavior of shared engine code
- ❌ Use FARM exit configs (must use SNIPER-specific exits)

---

## Change Process

**For FARM Changes:**
1. Create NEW snapshot directory (e.g., `2025_FARM_V2B_HYBRID_V3_RANGE_V2`)
2. Copy and modify files in NEW snapshot
3. Update `meta.role` to new version
4. NEVER modify existing `2025_FARM_V2B_HYBRID_V3_RANGE` snapshot

**For SNIPER Changes:**
1. Modify files in `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/`
2. Create new snapshot if major version change (e.g., `2025_SNIPER_V2`)
3. NEVER touch FARM PROD_BASELINE files

---

## Verification Checklist

Before any release:

- [ ] Run `./scripts/check_farm_lock.sh` - all hashes match
- [ ] Run `prod_baseline_proof.py` on latest FARM run - all checks pass
- [ ] Verify SNIPER configs are in `sniper_snapshot/` (not `prod_snapshot/`)
- [ ] Verify SNIPER exit configs are separate from FARM exits
- [ ] Verify no FARM files were modified (git diff check)

---

*Last Updated: 2025-12-17*

