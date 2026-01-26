# Repo Structure Analysis: Ideal vs Current

**Date:** 2026-01-07  
**Purpose:** Compare current repo structure with ideal state after cleanup

---

## Ideal Structure

```
gx1/
  features/      ✅ Core feature engineering
  inference/     ✅ Model inference
  runtime/       ✅ Runtime utilities
  execution/     ✅ Execution engine
  sniper/        ✅ SNIPER-specific logic
  rl/            ✅ Reinforcement learning
  tuning/        ✅ Hyperparameter tuning
  policy/        ✅ Policy definitions
  configs/       ✅ Configuration files
  models/        ✅ Model references/bundles only
  tests/         ✅ Test suite

docs/            ✅ Documentation
scripts/         ✅ Utility scripts
reports/         ✅ Audits, system docs (NOT runs)
```

**All other directories = artifacts, not system code**

---

## Current Structure Analysis

### ✅ Matches Ideal

| Directory | Status | Notes |
|-----------|--------|-------|
| `gx1/features/` | ✅ Match | Core feature engineering |
| `gx1/inference/` | ✅ Match | Model inference |
| `gx1/runtime/` | ✅ Match | Runtime utilities |
| `gx1/execution/` | ✅ Match | Execution engine |
| `gx1/sniper/` | ✅ Match | SNIPER-specific logic |
| `gx1/rl/` | ✅ Match | Reinforcement learning |
| `gx1/tuning/` | ✅ Match | Hyperparameter tuning |
| `gx1/policy/` | ✅ Match | Policy definitions |
| `gx1/configs/` | ✅ Match | Configuration files |
| `gx1/models/` | ✅ Match | Model references/bundles |
| `gx1/tests/` | ✅ Match | Test suite |
| `docs/` | ✅ Match | Documentation |
| `scripts/` | ✅ Match | Utility scripts |
| `reports/` | ✅ Match | Audits, system docs |

### ⚠️ Extra Directories (Not in Ideal)

| Directory | Type | Recommendation |
|-----------|------|-----------------|
| `gx1/analysis/` | Analysis tools | **Keep** (useful for research) |
| `gx1/backtest/` | Backtesting | **Keep** (core functionality) |
| `gx1/core/` | Core utilities | **Keep** (may be essential) |
| `gx1/data/` | Data utilities | **Review** (may be artifacts) |
| `gx1/docs/` | Internal docs | **Keep** (useful) |
| `gx1/monitoring/` | Monitoring | **Keep** (operational) |
| `gx1/orchestration/` | Orchestration | **Review** (may be unused) |
| `gx1/policies/` | Legacy policies? | **Review** (duplicate of `policy/`?) |
| `gx1/portfolio/` | Portfolio mgmt | **Keep** (core functionality) |
| `gx1/prod/` | Production configs | **Keep** (operational) |
| `gx1/regime/` | Regime detection | **Keep** (core functionality) |
| `gx1/scripts/` | Internal scripts | **Review** (may duplicate `scripts/`) |
| `gx1/seq/` | Sequence features | **Keep** (core functionality) |
| `gx1/tools/` | Tools | **Review** (may be artifacts) |
| `gx1/utils/` | Utilities | **Keep** (core utilities) |

### ❌ Artifacts (Should be Excluded)

| Directory | Status | Action |
|-----------|--------|--------|
| `gx1/wf_runs/` | ✅ Deleted | Already removed |
| `gx1/live/` | ✅ Deleted | Already removed |
| `runs/` | ⚠️ Exists | Should be empty or contain only structure |
| `data/replay/` | ✅ Deleted | Already removed |
| `data/temp/` | ✅ Deleted | Already removed |
| `logs/` | ⚠️ Exists | Should be excluded (runtime logs) |
| `$OUTPUT_DIR/` | ⚠️ Exists | Should be excluded (runtime output) |

---

## Recommendations

### 1. Keep (Core Functionality)
- `gx1/analysis/` - Analysis tools
- `gx1/backtest/` - Backtesting
- `gx1/core/` - Core utilities
- `gx1/docs/` - Internal documentation
- `gx1/monitoring/` - Monitoring
- `gx1/portfolio/` - Portfolio management
- `gx1/prod/` - Production configs
- `gx1/regime/` - Regime detection
- `gx1/seq/` - Sequence features
- `gx1/utils/` - Utilities

### 2. Review (May be Duplicates/Unused)
- `gx1/data/` - Check if contains artifacts vs utilities
- `gx1/orchestration/` - Verify if used
- `gx1/policies/` - **DEPRECATED** (duplicate of `policy/`, see analysis below)
- `gx1/scripts/` - Check if duplicates `scripts/`
- `gx1/tools/` - Review contents

#### Policy/Policies Duplicate Analysis

**Status:** `gx1/policies/` is deprecated (contains only A/B test configs)

**Evidence:**
- All imports in codebase use `gx1.policy.*` (not `gx1.policies.*`)
- `gx1/policy/` contains actual policy modules:
  - `entry_v9_policy_*.py` (SNIPER, FARM_V2, FARM_V2B)
  - `exit_*.py` (exit policies)
  - `farm_guards.py`, `sniper_risk_guard.py` (guards)
- `gx1/policies/` contains only:
  - `ab/AB_SNIPER_NY_2025W2.yaml` (A/B test config, not a policy module)

**Canonical:** `gx1/policy/` is the canonical location for policy modules

**Recommendation:**
- **Keep `gx1/policies/ab/`:** Contains A/B test configs (different purpose than policy modules)
- **Consider renaming:** `gx1/policies/` → `gx1/ab_tests/` or `gx1/configs/ab/` for clarity
- **Do not delete:** A/B test configs may be needed, but they're not policy modules
- **Action:** Review if `gx1/policies/ab/` should be moved to `gx1/configs/ab/` for better organization

### 3. Exclude (Runtime Artifacts)
- `runs/` - Should be empty or contain only structure (excluded in .gitignore)
- `logs/` - Runtime logs (should be excluded)
- `$OUTPUT_DIR/` - Runtime output (should be excluded)

---

## Action Items

1. ✅ **Completed:**
   - Deleted `gx1/wf_runs/`
   - Deleted `gx1/live/`
   - Deleted `data/replay/`
   - Deleted `data/temp/`
   - Updated `.gitignore` with permanent exclusions
   - Created `backup-exclude`

2. ⚠️ **To Review:**
   - Check `gx1/policies/` vs `gx1/policy/` (duplicate?)
   - Check `gx1/scripts/` vs `scripts/` (duplicate?)
   - Review `gx1/data/` contents (artifacts vs utilities?)
   - Review `gx1/orchestration/` usage
   - Review `gx1/tools/` contents

3. ⚠️ **To Exclude:**
   - Add `logs/` to `.gitignore` if not already
   - Add `$OUTPUT_DIR/` to `.gitignore` if not already
   - Verify `runs/` is empty or only contains structure

---

## Summary

**Current State:**
- ✅ Core ideal directories present
- ✅ Major artifacts removed
- ⚠️ Some extra directories (may be legitimate)
- ⚠️ Some runtime directories still exist (should be excluded)

**Next Steps:**
1. Review extra directories for duplicates/unused code
2. Ensure all runtime artifacts are excluded
3. Document purpose of each directory

---

**End of Document**
