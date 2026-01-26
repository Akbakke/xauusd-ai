# Feature Control Plane

**Version:** 2.0  
**Purpose:** Single Source of Truth (SSoT) for feature definitions, contracts, and validation.

## Overview

The Feature Control Plane provides deterministic feature classification, validation, and ablation capabilities without modifying trading logic, models, thresholds, or runtime semantics.

## Legacy Cleanup Policy

**Status:** Active  
**Purpose:** Prevent accidental execution of legacy v9/gated/sniper paths and ensure all output goes to GX1_DATA.

### Hard Fail Rules

The `gx1.runtime.legacy_guard` module enforces hard-fail rules at entrypoints:

1. **Policy Check:** Policies with `entry_models.v9` are blocked
2. **Path Check:** Output directories under engine repo (not `../GX1_DATA/reports` or `GX1_REPORTS_ROOT`) are blocked
3. **Arg Check:** Command-line arguments containing legacy references (`entry_v9`, `v9_farm`, `sniper/NY`) are blocked
4. **Env Check:** Legacy environment variables (`GX1_V9_MODE`, `GX1_ENTRY_V9_ENABLED`) are blocked

### Canonical Paths

All canonical paths use `GX1_DATA` root:

- **Data:** `../GX1_DATA/data/**/*.parquet`
- **Prebuilt:** `../GX1_DATA/data/**/*features*v10_ctx*.parquet`
- **Bundles:** `../GX1_DATA/models/**/entry_v10_ctx/FULLYEAR_*_GATED_FUSION`
- **Output:** `../GX1_DATA/reports/replay_eval/**` or `$GX1_REPORTS_ROOT/replay_eval/**`

### Legacy Audit

Run audit to find legacy references:

```bash
python3 gx1/scripts/audit_legacy_paths_and_v9.py
```

Outputs:
- `reports/repo_audit/LEGACY_SCAN_REPORT.md`
- `reports/repo_audit/LEGACY_SCAN_REPORT.json`

### Verify Entry Flow Gap Smoke

Run deterministic smoke test for entry flow verification:

```bash
python3 gx1/scripts/run_verify_entry_flow_gap_smoke.py
```

This script:
1. Runs preflight check (`preflight_prebuilt_import_check.py`)
2. Runs replay with v10_ctx, correct bundle, prebuilt, policy
3. Generates `verify_entry_flow_gap_report.json`

All paths are absolute and explicit - no guessing. Output goes to `../GX1_DATA/reports/replay_eval/VERIFY_ENTRY_FLOW_GAP/<RUN_ID>`.

### Legacy Quarantine

Legacy files should be:
- **Moved to:** `docs/legacy/` or `tools/legacy/`
- **Guarded:** Add `RuntimeError("LEGACY_DO_NOT_USE")` at import/entrypoint
- **Documented:** Add banner comment explaining why it's legacy

Files that are purely legacy (v9, gated legacy, sniper/NY) and not used should be quarantined or deleted.

## Reserved Candle Columns & Input Aliasing

**Purpose:** Prevent collisions between candle data and prebuilt features.

**Reserved Columns (Case-Insensitive):**
- `open`, `high`, `low`, `close`, `volume`
- `bid_open`, `bid_high`, `bid_low`, `bid_close`
- `ask_open`, `ask_high`, `ask_low`, `ask_close`

**Rules:**
1. **Prebuilt Features:** Must NOT contain any reserved columns
2. **CLOSE Special Case:** `CLOSE` feature is treated as an input alias, not a prebuilt feature
3. **Transformer Input Assembly:** When `CLOSE` is in `snap_feature_names`, it is aliased from `candles.close` (not from prebuilt features)
4. **Case Collisions:** Case-insensitive collisions are fatal and must be fixed at source

**CLOSE Alias Implementation:**
- **Prebuilt Builder:** Drops `CLOSE` from prebuilt schema before writing parquet
- **Transformer Input:** Maps `CLOSE` → `candles.close` when building snapshot tensor
- **Telemetry:** Records `input_aliases_applied: {"CLOSE": "candles.close"}` in `ENTRY_FEATURES_USED.json`
- **Validation:** Hard-fail if `CLOSE` is found in prebuilt schema (manifest generator, loader)

**Compat-Mode (Emergency Only):**
- `GX1_ALLOW_CLOSE_ALIAS_COMPAT=1` enables temporary workaround (drops `CLOSE` at runtime)
- **NOT for truth/baseline runs** - hard-fail if enabled in truth mode
- Default: OFF (permanent fix should be used)

## XGB Flow Ablation Tests

**Purpose:** Measure the contribution of XGBoost in the V10 hybrid entry pipeline.

### Test 1: XGB Channels → Transformer

Tests whether XGB features as input to Transformer are useful.

```bash
python3 gx1/scripts/run_xgb_flow_ablation_qsmoke.py \
  --arm test1_channels \
  --smoke-date-range "2025-01-06..2025-01-10" \
  --workers 1 \
  --data <ABS_PATH>/full_2020_2025.parquet \
  --prebuilt-parquet <ABS_PATH>/xauusd_m5_2025_features_v10_ctx.parquet \
  --bundle-dir <ABS_PATH>/FULLYEAR_2025_GATED_FUSION \
  --policy <ABS_PATH>/GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml \
  --out-root ../GX1_DATA/reports/replay_eval/XGB_FLOW_ABLATION
```

**Env Variables:**
- `GX1_DISABLE_XGB_CHANNELS_IN_TRANSFORMER=1` — Disables XGB channels in transformer input

**Invariants:**
- Baseline: `n_xgb_channels_in_transformer_input > 0`
- Ablated: `n_xgb_channels_in_transformer_input == 0`, `xgb_channel_names == []`

### Test 2: XGB Post (Calibration) — REMOVED

> **⚠️ REMOVED 2026-01-24:** XGB POST (calibration/veto) has been completely removed from the pipeline.
> XGB now only provides pre-predict channels to Transformer. No post-processing.
> The `--arm test2_post` option is no longer available.

See `POST_REMOVAL_PROOF.md` for details on the removal.

### Trading Metrics Output

Both tests produce comparison reports in `*_COMPARE.json` and `*_COMPARE.md`:

| Metric | Description |
|--------|-------------|
| `n_trades` | Number of trades |
| `total_pnl_bps` | Total PnL in basis points |
| `mean_pnl_bps` | Mean PnL per trade |
| `median_pnl_bps` | Median PnL per trade |
| `max_dd` | Maximum drawdown (bps) |
| `winrate` | Win rate (if available) |

**SSoT for Metrics:** `chunk_0/metrics_*.json`

### Test-Only Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `GX1_DISABLE_XGB_CHANNELS_IN_TRANSFORMER` | Disable ALL XGB channels | `0` |
| `GX1_XGB_CHANNEL_MASK` | Drop specific channels (comma-sep) | `""` |
| `GX1_XGB_CHANNEL_ONLY` | Keep only specific channels (comma-sep) | `""` |
| ~~`GX1_DISABLE_XGB_POST_TRANSFORMER`~~ | ~~Test 2: Disable XGB calibration~~ | REMOVED 2026-01-24 |
| ~~`GX1_REQUIRE_XGB_CALIBRATION`~~ | ~~Require calibrators~~ | REMOVED 2026-01-24 |

**Per-Channel Masking (added 2026-01-24):**
```bash
# Drop specific channels (others kept)
GX1_XGB_CHANNEL_MASK="margin_xgb,uncertainty_score"

# Keep only specific channels (others dropped)
GX1_XGB_CHANNEL_ONLY="p_long_xgb"
```

Valid channel names: `p_long_xgb`, `margin_xgb`, `uncertainty_score`, `p_hat_xgb`

**Warning:** These are for A/B testing only. Do not use for truth/baseline runs.

## Schema

### Feature Manifest v2.0

The feature manifest (`gx1/feature_manifest_v1.json`) includes the following contracts per feature:

- **stage**: `prebuilt_only` | `runtime_only` | `derived_from_model` | `gate_only` | `log_only`
- **causal**: `true` | `false` (no lookahead leakage)
- **depends_on**: List of dependencies (e.g., `["candles", "htf_h1", "htf_h4", "model_output"]`)
- **units_contract**: `bps` | `ratio` | `flag` | `zscore` | `price` | `int`
- **normalization_contract**: `none` | `atr_h1` | `z_rolling(10,100)` | `other`
- **consumer_contract**: `TRANSFORMER_SEQ` | `TRANSFORMER_SNAP` | `XGB` | `GATE` | `LOG`
- **lineage_hash**: SHA256 hash over `(name, source_module, timeframe, units_contract, normalization_contract, causal, key_params)`

### Feature Families

All features are classified into families:
- `basic_v1`: Core basic features
- `sequence`: Sequence features (lookback-based)
- `htf`: Higher timeframe features
- `htf_derived`: HTF-derived features (H1/H4)
- `microstructure`: Microstructure features
- `session`: Session features
- `price`: Raw price fields
- `returns`: Return features
- `volatility`: Volatility features
- `risk`: Risk features
- `meta`: Meta/label fields
- `model_output`: Model outputs (not features)
- `smc_pack_v1`: SMC Starter Pack v1 features

## Validation Scripts

### 1. Schema Validation

```bash
python3 gx1/scripts/validate_feature_schema.py \
    --prebuilt-parquet ../GX1_DATA/data/features/xauusd_m5_2025_features_v10_ctx.parquet \
    --manifest-json gx1/feature_manifest_v1.json \
    --out-root ../GX1_DATA/reports/feature_validation
```

**Validates:**
- Index is DatetimeIndex UTC
- Required columns from manifest exist
- Dtype matches units_contract (flag/int)
- No NaN/Inf in features (except model outputs)

### 2. Units and Ranges Validation

```bash
python3 gx1/scripts/validate_feature_units_ranges.py \
    --prebuilt-parquet ../GX1_DATA/data/features/xauusd_m5_2025_features_v10_ctx.parquet \
    --manifest-json gx1/feature_manifest_v1.json \
    --out-root ../GX1_DATA/reports/feature_validation
```

**Validates:**
- BPS within reasonable bounds (-10000 to +10000)
- Ratios within reasonable bounds (-10 to +10)
- Z-scores within reasonable range (-10 to +10)
- Flags are exactly 0 or 1
- Prices are positive

### 3. Golden Feature Checks

```bash
python3 gx1/scripts/golden_feature_checks.py \
    --prebuilt-parquet ../GX1_DATA/data/features/xauusd_m5_2025_features_v10_ctx.parquet \
    --manifest-json gx1/feature_manifest_v1.json \
    --sample-timestamps 10 \
    --out-root ../GX1_DATA/reports/feature_validation
```

**Validates:**
- Features at specific timestamps (deterministic sampling)
- No NaN/Inf at sampled timestamps
- Units contract bounds at sampled timestamps

### 4. Leakage and Causality Validation

```bash
python3 gx1/scripts/validate_feature_leakage_and_causality.py \
    --prebuilt-parquet ../GX1_DATA/data/features/xauusd_m5_2025_features_v10_ctx.parquet \
    --candles-parquet ../GX1_DATA/data/oanda/years/2025.parquet \
    --manifest-json gx1/feature_manifest_v1.json \
    --sample-timestamps 10 \
    --out-root ../GX1_DATA/reports/feature_validation
```

**Validates:**
- Features recomputed from candles match prebuilt values (within tolerance)
- No lookahead leakage (causal recomputation)

## Group Ablation

### Ablation Smoke Runner

```bash
python3 gx1/scripts/ablate_feature_groups_qsmoke.py \
    --bundle-dir ../GX1_DATA/models/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION \
    --data-root ../GX1_DATA/data/oanda/years \
    --prebuilt-parquet ../GX1_DATA/data/features/xauusd_m5_2025_features_v10_ctx.parquet \
    --policy policies/sniper_trial160_prod.json \
    --manifest-json gx1/feature_manifest_v1.json \
    --out-root ../GX1_DATA/reports/feature_ablation \
    --quarters Q1,Q2
```

**Runs:**
- Baseline (no mask)
- Mask `basic_v1`
- Mask `sequence`
- Mask `htf`
- Mask `microstructure`
- Mask `session`

**Masking Strategy:**
- Numeric features: Set to 0
- Flags: Keep as-is (0/1)
- Never NaN (default to 0)

**Output:**
- `ABLATION_SUMMARY.json`: Summary of all runs
- Per-run: `RUN_IDENTITY.json` with `feature_mask_spec`
- Per-run: `FEATURE_MASK_SPEC.json` with masked features list

## Feature Masking Implementation

Feature masking is implemented in `gx1/execution/entry_manager.py` via environment variables:

- `GX1_FEATURE_MASK_ENABLED=1`: Enable masking
- `GX1_FEATURE_MASK_FAMILIES=basic_v1,sequence`: Comma-separated families to mask
- `GX1_FEATURE_MASK_STRATEGY=zero_numeric_keep_flags`: Masking strategy
- `GX1_FEATURE_MANIFEST_JSON`: Path to feature manifest JSON

Masking is applied **before** model input, ensuring it does not affect universe-fingerprint (only completion/perf).

## Regenerating Feature Manifest

```bash
python3 gx1/scripts/generate_feature_manifest.py \
    --prebuilt-parquet ../GX1_DATA/data/features/xauusd_m5_2025_features_v10_ctx.parquet \
    --output-json gx1/feature_manifest_v1.json \
    --output-md docs/FEATURE_MANIFEST.md
```

**Note:** Use `--allowlist-unknown` only for initial generation. After v2.0, all features must be classified.

## RUNBOOK: Complete Validation Workflow

### Step 1: Regenerate Feature Manifest

```bash
python3 gx1/scripts/generate_feature_manifest.py \
    --prebuilt-parquet ../GX1_DATA/data/features/xauusd_m5_2025_features_v10_ctx.parquet \
    --output-json gx1/feature_manifest_v1.json \
    --output-md docs/FEATURE_MANIFEST.md
```

**Expected:** 0 unknown features, all features have lineage_hash.

### Step 2: Run All Validators

```bash
# Schema validation
python3 gx1/scripts/validate_feature_schema.py \
    --prebuilt-parquet ../GX1_DATA/data/features/xauusd_m5_2025_features_v10_ctx.parquet \
    --manifest-json gx1/feature_manifest_v1.json \
    --out-root ../GX1_DATA/reports/feature_validation

# Units and ranges validation
python3 gx1/scripts/validate_feature_units_ranges.py \
    --prebuilt-parquet ../GX1_DATA/data/features/xauusd_m5_2025_features_v10_ctx.parquet \
    --manifest-json gx1/feature_manifest_v1.json \
    --out-root ../GX1_DATA/reports/feature_validation

# Golden feature checks
python3 gx1/scripts/golden_feature_checks.py \
    --prebuilt-parquet ../GX1_DATA/data/features/xauusd_m5_2025_features_v10_ctx.parquet \
    --manifest-json gx1/feature_manifest_v1.json \
    --sample-timestamps 10 \
    --out-root ../GX1_DATA/reports/feature_validation

# Leakage and causality validation (if candles available)
python3 gx1/scripts/validate_feature_leakage_and_causality.py \
    --prebuilt-parquet ../GX1_DATA/data/features/xauusd_m5_2025_features_v10_ctx.parquet \
    --candles-parquet ../GX1_DATA/data/oanda/years/2025.parquet \
    --manifest-json gx1/feature_manifest_v1.json \
    --sample-timestamps 10 \
    --out-root ../GX1_DATA/reports/feature_validation
```

**Expected:** All validators PASS.

### Step 3: Run Ablation Smoke (Q1/Q2)

```bash
python3 gx1/scripts/ablate_feature_groups_qsmoke.py \
    --bundle-dir ../GX1_DATA/models/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION \
    --data-root ../GX1_DATA/data/oanda/years \
    --prebuilt-parquet ../GX1_DATA/data/features/xauusd_m5_2025_features_v10_ctx.parquet \
    --policy policies/sniper_trial160_prod.json \
    --manifest-json gx1/feature_manifest_v1.json \
    --out-root ../GX1_DATA/reports/feature_ablation \
    --quarters Q1,Q2
```

**Expected:**
- Baseline completes successfully
- All masked groups complete successfully
- `ABLATION_SUMMARY.json` generated with comparison metrics

## XGB Flow Ablation (Test 1 & 2)

The XGB Flow Ablation tests measure the value of the XGBoost↔Transformer pipeline by comparing baseline performance against ablated variants.

### Test 1: XGB Channels → Transformer

**Purpose:** Measure the impact of XGB channels in Transformer input.

**ARM_A (baseline):** Everything as it is today (XGB channels included in Transformer input).

**ARM_B (no_xgb_channels_in_transformer):** Transformer receives 0 XGB channels (but everything else identical).

**Toggle:** `GX1_DISABLE_XGB_CHANNELS_IN_TRANSFORMER=1`

**What it does:**
- Transformer input building excludes XGB channels (both sequence and snapshot).
- `n_xgb_channels_in_transformer_input` becomes 0.
- `xgb_channel_names` becomes empty.
- Base sequence/snapshot features remain unchanged.

### Test 2: XGB Post-Transformer Calibrator/Veto

**Purpose:** Measure the impact of XGB post-transformer calibration/veto.

**ARM_A (baseline):** Everything as it is today (XGB calibration/veto active).

**ARM_B (no_xgb_post):** XGB post-transformer calibration/veto deactivated (Transformer + gates otherwise identical).

**Toggle:** `GX1_DISABLE_XGB_POST_TRANSFORMER=1`

**What it does:**
- Transformer runs normally.
- XGB "post" predict/calibration is not called.
- Veto/guard coming from XGB post is not applied.
- Final decision comes from Transformer + existing gates (thresholds unchanged).

### Running the Tests

#### Preflight Check

Before running the tests, ensure PREBUILT mode is properly configured:

```bash
python gx1/scripts/preflight_prebuilt_import_check.py
```

**Expected:** No forbidden imports detected.

#### Test 1: Q1 Smoke

```bash
python gx1/scripts/run_xgb_flow_ablation_qsmoke.py \
    --arm test1_channels \
    --years 2025 \
    --data <CANDLES_PARQUET> \
    --prebuilt-parquet <PREBUILT_PARQUET> \
    --bundle-dir <BUNDLE_DIR_ABSOLUTE_PATH> \
    --policy <POLICY_YAML> \
    --out-root <OUTPUT_ROOT> \
    --smoke-date-range "2025-01-01..2025-03-31" \
    --workers 1
```

**Note:** `--bundle-dir` must be an absolute path pointing to the bundle directory (e.g., `../GX1_DATA/models/models/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION`). This overrides the `bundle_dir` specified in the policy YAML, allowing the policy to remain portable with relative paths while pointing to the actual data depot location. The override priority is: CLI `--bundle-dir` > ENV `GX1_BUNDLE_DIR` > Policy `bundle_dir`.

**Output:**
- `baseline/`: Baseline run results
- `no_xgb_channels_in_transformer/`: Ablated run results
- `XGB_FLOW_ABLATION_TEST1_COMPARE.json`: Comparison metrics
- `XGB_FLOW_ABLATION_TEST1_COMPARE.md`: Human-readable report

#### Test 2: Q1 Smoke

```bash
python gx1/scripts/run_xgb_flow_ablation_qsmoke.py \
    --arm test2_post \
    --years 2025 \
    --data <CANDLES_PARQUET> \
    --prebuilt-parquet <PREBUILT_PARQUET> \
    --bundle-dir <BUNDLE_DIR_ABSOLUTE_PATH> \
    --policy <POLICY_YAML> \
    --out-root <OUTPUT_ROOT> \
    --smoke-date-range "2025-01-01..2025-03-31" \
    --workers 1
```

**Note:** Same bundle-dir override behavior as Test 1.

**Output:**
- `baseline/`: Baseline run results (reused if Test 1 already ran)
- `no_xgb_post/`: Ablated run results
- `XGB_FLOW_ABLATION_TEST2_COMPARE.json`: Comparison metrics
- `XGB_FLOW_ABLATION_TEST2_COMPARE.md`: Human-readable report

#### Running Both Tests

```bash
python gx1/scripts/run_xgb_flow_ablation_qsmoke.py \
    --arm both \
    --years 2025 \
    --data <CANDLES_PARQUET> \
    --prebuilt-parquet <PREBUILT_PARQUET> \
    --bundle-dir <BUNDLE_DIR_ABSOLUTE_PATH> \
    --policy <POLICY_YAML> \
    --out-root <OUTPUT_ROOT> \
    --smoke-date-range "2025-01-01..2025-03-31" \
    --workers 1
```

**Note:** Same bundle-dir override behavior as Test 1.

### Reading the Compare Report

The compare report (`XGB_FLOW_ABLATION_TEST*_COMPARE.md`) includes:

1. **Trading Metrics:**
   - Trades, PnL (bps), Mean/Median PnL, MaxDD, Winrate
   - Deltas vs baseline (absolute and percentage)

2. **Telemetry Sanity:**
   - `n_xgb_channels_in_transformer_input`: Baseline vs ablated
   - `xgb_used_as`: Usage summary (none/pre/post/both)
   - `post_predict_called`: Whether post-transformer predict was called
   - `veto_applied_count`: Number of vetoes applied

3. **Invariants:**
   - **Test 1:** Verifies `n_xgb_channels_in_transformer_input == 0` in ablated arm
   - **Test 2:** Verifies `post_predict_called == False` and `veto_applied_count == 0` in ablated arm
   - **No-op detection:** If baseline has `n_xgb_channels_in_transformer_input == 0`, Test 1 is marked as no-op

### Invariants and Fail-Fast

The script enforces strict invariants:

- **Test 1:** If `GX1_DISABLE_XGB_CHANNELS_IN_TRANSFORMER=1` and `n_xgb_channels_in_transformer_input != 0` → FATAL
- **Test 2:** If `GX1_DISABLE_XGB_POST_TRANSFORMER=1` and `post_predict_called == True` → FATAL
- **Test 2:** If `GX1_DISABLE_XGB_POST_TRANSFORMER=1` and `veto_applied_count > 0` → FATAL

### Bundle Directory Override

The `--bundle-dir` argument (or `GX1_BUNDLE_DIR` environment variable) allows overriding the `bundle_dir` specified in the policy YAML. This keeps policies portable (with relative paths) while pointing to the actual data depot location.

**Priority (highest first):**
1. CLI `--bundle-dir` argument
2. ENV `GX1_BUNDLE_DIR` variable
3. Policy `entry_models.v10_ctx.bundle_dir` (resolved relative to policy file's directory, not cwd)

**Verification:**
- `RUN_IDENTITY.json` includes:
  - `bundle_dir_resolved`: Absolute path to bundle directory
  - `bundle_dir_source`: "cli" | "env" | "policy"
  - `bundle_dir_exists`: true/false
- `master_early.json` also includes these fields for early diagnostics

**Fail-Fast:**
- If resolved bundle_dir does not exist → FATAL
- If policy does not have `entry_models.v10_ctx` configured → FATAL (policy sanity check)

### Telemetry Files

Each run generates:
- `ENTRY_FEATURES_USED.json`: Contains `xgb_flow`, `toggles`, and `gate_stats`
- `ENTRY_FEATURES_TELEMETRY.json`: Detailed telemetry samples
- `FEATURE_MASK_APPLIED.json`: Feature masking state (if applicable)

These files are referenced in `RUN_SUMMARY.json` under `entry_feature_telemetry_files`.

## References

- **Feature Manifest:** `docs/FEATURE_MANIFEST.md`
- **Data Contract:** `docs/DATA_CONTRACT.md`
- **Feature Map:** `reports/repo_audit/FEATURE_MAP.md`
- **Implementation Notes:** `reports/repo_audit/FEATURE_CONTROL_PLANE_IMPLEMENTATION.md`
