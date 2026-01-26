# TRIAL 160 AUDIT CHECKLIST — Operasjonell Verifisering

**Dato:** 2026-01-16  
**Status:** 🔄 **IN PROGRESS**

## Mål

Operasjonell checklist for å verifisere at Trial 160 pipeline kjører korrekt med harde invariants. Alle sjekker må være **maskinlesbare** og **fail-fast**.

## Pre-Flight: Doctor Check

### Kommando
```bash
./scripts/doctor_trial160.sh
```

### Forventet Output
```
[RUN_CTX] root=/path/to/repo
[RUN_CTX] head=abc1234
[RUN_CTX] python=/usr/bin/python3 (3.10.13)
[DOCTOR] ✅ Repo-root verified
[DOCTOR] ✅ Git head verified
[DOCTOR] ✅ Python executable verified
[DOCTOR] ✅ Policy file exists: policies/sniper_trial160_prod.json
[DOCTOR] ✅ Policy SHA256: abc123...
[DOCTOR] ✅ Prebuilt features exist: data/features/xauusd_m5_2025_features_v10_ctx.parquet
[DOCTOR] ✅ Prebuilt SHA256: def456...
[DOCTOR] ✅ Data file exists: data/raw/xauusd_m5_2025_bid_ask.parquet
[DOCTOR] ✅ No active replays (lock check passed)
[DOCTOR] ✅ Legacy deactivated (ALLOW_LEGACY not set)
[DOCTOR] ✅ All checks passed
```

### Fail Conditions
- ❌ Not in repo-root
- ❌ Git head not found or dirty (unless ALLOW_DIRTY=1)
- ❌ Policy file missing
- ❌ Policy SHA256 mismatch
- ❌ Prebuilt features missing
- ❌ Prebuilt SHA256 mismatch
- ❌ Data file missing
- ❌ Active replay detected (lock exists)
- ❌ Legacy enabled (ALLOW_LEGACY=1 set)

---

## Input Data Verification

### 1. Data File

**Kommando:**
```bash
python3 -c "
from pathlib import Path
import pandas as pd
data_path = Path('data/raw/xauusd_m5_2025_bid_ask.parquet')
df = pd.read_parquet(data_path)
print(f'Rows: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'Date range: {df.index.min()} to {df.index.max()}')
"
```

**Forventet Output:**
```
Rows: 105120  # 365 days * 288 bars/day
Columns: ['open', 'high', 'low', 'close', 'volume', 'bid', 'ask']
Date range: 2025-01-01 00:00:00+00:00 to 2025-12-31 23:55:00+00:00
```

**Fail Conditions:**
- ❌ File missing
- ❌ Wrong date range
- ❌ Missing columns (bid/ask required)
- ❌ NaN values in critical columns

### 2. Prebuilt Features

**Kommando:**
```bash
python3 -c "
from pathlib import Path
import pandas as pd
prebuilt_path = Path('data/features/xauusd_m5_2025_features_v10_ctx.parquet')
df = pd.read_parquet(prebuilt_path)
print(f'Rows: {len(df)}')
print(f'Columns: {len(df.columns)}')
print(f'Date range: {df.index.min()} to {df.index.max()}')
print(f'Schema SHA256: ...')  # TODO: implement schema hash
"
```

**Forventet Output:**
```
Rows: 105120
Columns: 150  # V10_CTX feature count
Date range: 2025-01-01 00:00:00+00:00 to 2025-12-31 23:55:00+00:00
Schema SHA256: abc123...
```

**Fail Conditions:**
- ❌ File missing
- ❌ Wrong row count (must match data file)
- ❌ Wrong column count (must match V10_CTX schema)
- ❌ Schema mismatch (dimensions don't match model expectations)

### 3. Policy File

**Kommando:**
```bash
python3 -c "
import json
from pathlib import Path
import hashlib
policy_path = Path('policies/sniper_trial160_prod.json')
policy = json.load(open(policy_path))
policy_sha = hashlib.sha256(open(policy_path, 'rb').read()).hexdigest()
print(f'Policy ID: {policy[\"policy_id\"]}')
print(f'Policy SHA256: {policy_sha}')
print(f'Entry threshold: {policy[\"entry_threshold\"]}')
print(f'Max positions: {policy[\"max_concurrent_positions\"]}')
"
```

**Forventet Output:**
```
Policy ID: trial160_prod_v1
Policy SHA256: abc123...
Entry threshold: 0.102
Max positions: 2
```

**Fail Conditions:**
- ❌ File missing
- ❌ Missing required fields
- ❌ Wrong policy_id
- ❌ Wrong parameter values

---

## Prebuilt Pipeline Verification

### 1. Prebuilt Usage (Hard Requirement)

**Kommando:**
```bash
grep -r "GX1_REPLAY_USE_PREBUILT_FEATURES" scripts/run_fullyear_trial160_prebuilt.sh
```

**Forventet Output:**
```
GX1_REPLAY_USE_PREBUILT_FEATURES=1
GX1_FEATURE_BUILD_DISABLED=1
```

**Fail Conditions:**
- ❌ PREBUILT not enabled
- ❌ Feature building not disabled

### 2. Feature Building Forbidden

**Kommando:**
```bash
# Check that feature building functions hard-fail in PREBUILT mode
python3 -c "
import os
os.environ['GX1_REPLAY_USE_PREBUILT_FEATURES'] = '1'
os.environ['GX1_FEATURE_BUILD_DISABLED'] = '1'
# Try to call feature building function
# Should raise RuntimeError
"
```

**Forventet Output:**
```
RuntimeError: Feature building disabled in PREBUILT mode
```

**Fail Conditions:**
- ❌ Feature building succeeds (should hard-fail)

### 3. Lookup Invariants

**Kommando:**
```bash
# After running smoke test, check chunk footer
python3 -c "
import json
from pathlib import Path
footer = json.load(open('reports/replay_eval/TRIAL160_SMOKE_2DAYS/chunk_0/chunk_footer.json'))
lookup_attempts = footer['lookup_attempts']
lookup_hits = footer['lookup_hits']
lookup_misses = footer['lookup_misses']
assert lookup_attempts == lookup_hits + lookup_misses, 'SSoT violation'
print(f'✅ Lookup invariant: {lookup_attempts} == {lookup_hits} + {lookup_misses}')
"
```

**Forventet Output:**
```
✅ Lookup invariant: 1000 == 800 + 200
```

**Fail Conditions:**
- ❌ `lookup_attempts != lookup_hits + lookup_misses`
- ❌ `lookup_hits == 0` (should have hits)
- ❌ `lookup_misses > 0` with KeyError (should hard-fail)

---

## Model/Bundle Verification

### 1. Bundle Loading

**Kommando:**
```bash
# Check that bundle loads correctly
python3 -c "
from gx1.models.entry_v10_ctx import load_entry_v10_ctx_bundle
bundle_path = Path('models/entry_v10_ctx/...')  # TODO: determine path
bundle = load_entry_v10_ctx_bundle(bundle_path)
print(f'Bundle SHA256: {bundle.sha256}')
print(f'Model loaded: {bundle.model is not None}')
"
```

**Forventet Output:**
```
Bundle SHA256: abc123...
Model loaded: True
```

**Fail Conditions:**
- ❌ Bundle missing
- ❌ Bundle SHA256 mismatch
- ❌ Model not loaded

### 2. Feature Schema Match

**Kommando:**
```bash
# Check that prebuilt features match model expectations
python3 -c "
# Load prebuilt features
# Load model bundle
# Compare dimensions
# Should match exactly
"
```

**Forventet Output:**
```
✅ Feature schema matches model expectations
✅ Dimensions: 150 features
✅ Sequence length: 90
```

**Fail Conditions:**
- ❌ Schema mismatch
- ❌ Dimension mismatch
- ❌ Missing required columns
- ❌ **seq_dims mismatch**: Runtime `seq_data` must be `base_seq_features + XGB_CHANNELS` (16 total for V10_CTX)
  - `seq_feat_names` from feature_meta.json includes only base features (13)
  - Runtime `seq_data` includes XGB channels (3 additional: `["p_long_xgb", "margin_xgb", "uncertainty_score"]`)
  - Validation must account for this: `seq_data.shape[1] == len(seq_feat_names) + len(XGB_SEQ_CHANNEL_NAMES)`

---

## Policy Verification

### 1. Policy Loading

**Kommando:**
```bash
# Check that policy loads with all required fields
python3 -c "
import json
from pathlib import Path
policy = json.load(open('policies/sniper_trial160_prod.json'))
required = ['policy_id', 'entry_threshold', 'max_concurrent_positions', 
            'risk_guard_block_atr_bps_gte', 'risk_guard_block_spread_bps_gte',
            'risk_guard_cooldown_bars_after_entry']
for field in required:
    assert field in policy, f'Missing field: {field}'
print('✅ All required fields present')
"
```

**Forventet Output:**
```
✅ All required fields present
```

**Fail Conditions:**
- ❌ Missing required fields
- ❌ Wrong field types
- ❌ Wrong values

### 2. Policy ID Verification

**Kommando:**
```bash
# Check that policy_id is logged in RUN_IDENTITY.json
python3 -c "
import json
from pathlib import Path
identity = json.load(open('reports/replay_eval/TRIAL160_SMOKE_2DAYS/RUN_IDENTITY.json'))
assert identity['policy_id'] == 'trial160_prod_v1', 'Policy ID mismatch'
print(f'✅ Policy ID verified: {identity[\"policy_id\"]}')
"
```

**Forventet Output:**
```
✅ Policy ID verified: trial160_prod_v1
```

**Fail Conditions:**
- ❌ Policy ID missing
- ❌ Policy ID mismatch

---

## Guards/Gating Verification

### 1. Guard Block Rates

**Kommando:**
```bash
# After running smoke test, check guard block rates
python3 -c "
import json
from pathlib import Path
# Aggregate killchain from all chunks
# Compute guard block rates
# Verify expected ranges
"
```

**Forventet Output:**
```
Spread block rate: 0.0000 (expected: 0.0, spread never exceeds 2000 bps)
ATR block rate: 0.14-0.19 (expected range)
Threshold pass rate: 0.51-0.52 (expected range)
Max positions hit rate: 0.0 (expected: low, position limits rarely hit)
```

**Fail Conditions:**
- ❌ Guard block rates outside expected ranges
- ❌ Spread block rate > 0.0 (unexpected, should be 0.0)

### 2. Kill-Chain Stage2/Stage3

**Kommando:**
```bash
# Check kill-chain counters
python3 -c "
# Aggregate killchain from all chunks
# Verify stage2/stage3 counters
"
```

**Forventet Output:**
```
Stage2 After Vol Guard: ~42,000 (expected range)
Stage2 Pass Score Gate: ~21,000 (expected range)
Stage2 Block Threshold: ~20,000 (expected range)
Stage2 Block ATR: ~6,000-8,000 (expected range)
Stage3 Trades Created: ~13,000-15,000 (expected range)
```

**Fail Conditions:**
- ❌ Counters outside expected ranges
- ❌ Missing counters
- ❌ SSoT violations (counters don't sum correctly)

---

## Output Verification

### 1. RUN_IDENTITY.json

**Kommando:**
```bash
cat reports/replay_eval/TRIAL160_FULLYEAR/RUN_IDENTITY.json
```

**Forventet Output:**
```json
{
  "git_head": "abc1234",
  "git_dirty": false,
  "python_executable": "/usr/bin/python3",
  "python_version": "3.10.13",
  "bundle_sha256": "def456...",
  "windows_sha256": "ghi789...",
  "prebuilt_manifest_sha256": "jkl012...",
  "policy_sha256": "mno345...",
  "policy_id": "trial160_prod_v1",
  "replay_mode": "PREBUILT",
  "feature_build_disabled": true,
  "timestamp": "2026-01-16T18:00:00Z"
}
```

**Fail Conditions:**
- ❌ Missing required fields
- ❌ Wrong values
- ❌ `replay_mode != "PREBUILT"`
- ❌ `feature_build_disabled != true`

### 2. Chunk Footers

**Kommando:**
```bash
# Check all chunk footers
python3 -c "
from pathlib import Path
import json
footers = sorted(Path('reports/replay_eval/TRIAL160_FULLYEAR').glob('chunk_*/chunk_footer.json'))
for footer_path in footers:
    footer = json.load(open(footer_path))
    assert footer['status'] == 'ok', f'Chunk failed: {footer_path}'
    assert footer['prebuilt_used'] == True, f'Prebuilt not used: {footer_path}'
    assert footer['tripwire_passed'] == True, f'Tripwire failed: {footer_path}'
print(f'✅ All {len(footers)} chunks passed')
"
```

**Forventet Output:**
```
✅ All 7 chunks passed
```

**Fail Conditions:**
- ❌ Any chunk status != 'ok'
- ❌ Any chunk prebuilt_used != True
- ❌ Any chunk tripwire_passed != True

### 3. Metrics

**Kommando:**
```bash
# Check merged metrics
python3 -c "
import json
from pathlib import Path
metrics = json.load(open('reports/replay_eval/TRIAL160_FULLYEAR/metrics_*_MERGED.json'))
print(f'Trades: {metrics[\"n_trades\"]}')
print(f'Total PnL: {metrics[\"total_pnl_bps\"]} bps')
print(f'Max DD: {metrics[\"max_dd\"]} bps')
"
```

**Forventet Output:**
```
Trades: 15000-16000 (expected range)
Total PnL: 92000-93000 bps (expected range, matches promotion results)
Max DD: -200 to -250 bps (expected range)
```

**Fail Conditions:**
- ❌ Metrics missing
- ❌ Metrics outside expected ranges
- ❌ Significant deviation from promotion results

---

## Smoke Tests

### 1. 2-Day Smoke Test

**Kommando:**
```bash
./scripts/smoke_trial160_2days.sh
```

**Forventet Output:**
```
[SMOKE_2DAYS] Starting 2-day smoke test...
[SMOKE_2DAYS] ✅ All checks passed
[SMOKE_2DAYS] ✅ Trades created: 100-200 (expected range)
[SMOKE_2DAYS] ✅ PnL: positive (expected)
[SMOKE_2DAYS] ✅ Tripwires passed
```

**Fail Conditions:**
- ❌ Any check fails
- ❌ No trades created
- ❌ Negative PnL (may be acceptable, but flag)
- ❌ Tripwire failures

### 2. 7-Day Smoke Test

**Kommando:**
```bash
./scripts/smoke_trial160_7days.sh
```

**Forventet Output:**
```
[SMOKE_7DAYS] Starting 7-day smoke test...
[SMOKE_7DAYS] ✅ All checks passed
[SMOKE_7DAYS] ✅ Trades created: 500-1000 (expected range)
[SMOKE_7DAYS] ✅ PnL: positive (expected)
[SMOKE_7DAYS] ✅ Tripwires passed
```

**Fail Conditions:**
- ❌ Any check fails
- ❌ No trades created
- ❌ Negative PnL (may be acceptable, but flag)
- ❌ Tripwire failures

---

## FULLYEAR Verification

### 1. Pre-Flight

**Kommando:**
```bash
./scripts/doctor_trial160.sh && ./scripts/smoke_trial160_2days.sh && ./scripts/smoke_trial160_7days.sh
```

**Forventet Output:**
```
✅ All pre-flight checks passed
```

**Fail Conditions:**
- ❌ Any pre-flight check fails

### 2. FULLYEAR Run

**Kommando:**
```bash
./scripts/run_fullyear_trial160_prebuilt.sh
```

**Forvented Output:**
```
[FULLYEAR] Starting FULLYEAR backtest...
[FULLYEAR] ✅ All chunks completed
[FULLYEAR] ✅ All tripwires passed
[FULLYEAR] ✅ Metrics generated
```

**Fail Conditions:**
- ❌ Any chunk fails
- ❌ Any tripwire fails
- ❌ Metrics missing

### 3. Report Generation

**Kommando:**
```bash
# Report should be auto-generated
cat reports/replay_eval/TRIAL160_FULLYEAR/FULLYEAR_TRIAL160_REPORT.md
```

**Forventet Output:**
```
# FULLYEAR TRIAL 160 REPORT

## Performance Metrics
- Total PnL: 92,923.77 bps
- Trades: 15,508
- Max DD: -201.84 bps
...

## Per-Session Breakdown
- EU: ...
- US: ...
- OVERLAP: ...
```

**Fail Conditions:**
- ❌ Report missing
- ❌ Report incomplete
- ❌ Metrics don't match promotion results

---

## Summary

Alle sjekker må være **maskinlesbare** og **fail-fast**. Ingen vage punkter. Hver sjekk har:
- Eksakt kommando
- Forventet output
- Fail conditions

**Status:** 🔄 **IN PROGRESS** — Implementerer guards og scripts.
