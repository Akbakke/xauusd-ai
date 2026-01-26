# Depth Ladder Universe Invariants

## Overview

Depth Ladder evaluations require **identical data universes** between baseline and L+1 arms to ensure fair comparison. This document describes the universe fingerprinting and validation system.

## DataUniverseFingerprint

The `DataUniverseFingerprint` is a deterministic snapshot of all inputs that affect trade universe:

### Fingerprint Elements

- **Data paths**: Absolute paths to candles data and prebuilt features
- **Candles metadata**: Row count, first/last timestamps
- **Prebuilt metadata**: Row count, first/last timestamps
- **Policy**: SHA256 hash and policy ID
- **Bundle metadata**: SHA256, transformer layers, depth ladder delta
- **Environment**: Replay mode, temperature scaling status

### Computation

The fingerprint is computed **before replay** from input files (not from replay output) to ensure determinism:

1. **Candles stats**: Read parquet index to get rowcount and timestamp range
2. **Prebuilt stats**: Use pyarrow metadata (or pandas fallback) for rowcount and timestamp range
3. **Policy hash**: SHA256 of policy YAML file
4. **Bundle SHA256**: SHA256 of `model_state_dict.pt`
5. **Environment**: From env vars and bundle metadata

### Storage

- `smoke_stats.json`: Full smoke eval results including fingerprint
- `baseline_reference.json`: Baseline fingerprint (for L+1 validation)
- `master_early.json`: Top-level metadata including fingerprint
- `RUN_IDENTITY.json`: Runtime identity including fingerprint fields

## Hard Invariants: Universe Match

When L+1 smoke eval runs, it **must** match baseline universe in:

### Critical Keys (FATAL if mismatch)

- `candles_first_ts`
- `candles_last_ts`
- `candles_rowcount_loaded`
- `prebuilt_first_ts`
- `prebuilt_last_ts`
- `prebuilt_rowcount`
- `policy_sha256`
- `replay_mode` (must be "PREBUILT")
- `temperature_scaling_effective_enabled` (must be True)

### Error Format

```
[DEPTH_LADDER] FATAL: TRADE_UNIVERSE_MISMATCH
Arm: LPLUS1
Mismatches:
  candles_rowcount_loaded:
    baseline: 43441
    current:  14852
```

## PREBUILT and Temperature Scaling Verification

### PREBUILT Verification

**FATAL if:**
- `feature_build_disabled != True` in RUN_IDENTITY
- `feature_build_call_count > 0` in perf.json

### Temperature Scaling Verification

**FATAL if:**
- `temperature_scaling_effective_enabled != True` in RUN_IDENTITY

## Metadata Wiring

### RUN_IDENTITY.json

Must always contain:
- `transformer_layers`
- `transformer_layers_baseline`
- `depth_ladder_delta`
- `policy_id` (or `policy_sha256`)
- `replay_mode` (must be "PREBUILT")
- `temperature_scaling_effective_enabled` (must be True)
- `bundle_sha256`

### Baseline Fallback

If baseline bundle lacks metadata:
- Infer `transformer_layers=3`
- Infer `transformer_layers_baseline=3`
- Infer `depth_ladder_delta=0`
- Log source as "inferred"

## Baseline Re-run Helper

Use `--recheck-baseline-with-arm-config` to re-run baseline with same config as L+1:

```bash
python gx1/scripts/run_depth_ladder_eval_multiyear.py \
  --arm lplus1 \
  --recheck-baseline-with-arm-config \
  --bundle-dir models/entry_v10_ctx_depth_ladder/LPLUS1 \
  ...
```

This helps diagnose config/slicing mismatches.

## Acceptance Test Sequence

1. **Baseline smoke**: Generates `baseline_reference.json` + fingerprint
2. **L+1 smoke**: Validates universe match against baseline (FATAL if mismatch)
3. **When match passes**: L+1 smoke produces:
   - Different `bundle_sha256`
   - `transformer_layers=4` vs baseline=3
   - Same candles/prebuilt/policy fingerprint
   - PREBUILT verified
   - Temperature scaling verified

## Implementation

- **Fingerprint computation**: `compute_data_universe_fingerprint()` in `run_depth_ladder_eval_multiyear.py`
- **Universe validation**: `validate_universe_match()` in `run_depth_ladder_eval_multiyear.py`
- **PREBUILT/Temp scaling verification**: `verify_prebuilt_and_temp_scaling()` in `run_depth_ladder_eval_multiyear.py`
- **Compare script**: `compare_depth_ladder_smoke.py` uses fingerprints for comparison

## Principles

- **Minimal diff**: Only eval/orchestrator plumbing, no trading logic changes
- **Fail-fast**: Hard-fail on any mismatch
- **Deterministic**: No wall-clock timestamps, all from data/metadata
- **No silent fallbacks**: All missing data logged and validated
