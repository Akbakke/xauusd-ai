# PROD Freeze - Current Production Baseline

**Frozen Date:** 2025-12-15  
**Role:** PROD_BASELINE

This directory contains frozen production artifacts:
- `policy.yaml`: Frozen policy configuration
- `exit_router_v3_tree.pkl`: Frozen router model
- `entry_models/`: Entry model artifacts (symlinks or copies)
- `feature_manifest.json`: Feature manifest (if available)

**Usage:**
When `meta.role == PROD_BASELINE`, all paths resolve relative to `gx1/prod/current/`.
No fallback to repo default paths.

**Verification:**
Run `python gx1/prod/verify_freeze.py` to verify all artifacts are present and valid.

