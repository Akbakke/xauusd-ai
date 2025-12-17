# GX1/FARM Pipeline Audit - Complete Summary

**Generated:** 2025-12-17  
**Purpose:** Consolidated overview of pipeline audit findings

---

## Executive Summary

### Current State
- **Total Runs:** 118 directories
- **Total Size:** 4,467.83 MB (~4.37 GB)
- **Runs with Trade Journal:** 9 (7.6%)
- **PROD_BASELINE Runs:** 0 (identified by run_header.json)
- **CANARY Runs:** 0 (identified by run_header.json)
- **Runs with V3_RANGE Router:** Unknown (need to check guardrail params)
- **Obsolete Candidates:** 32 runs (27.1%)

### Current PROD_BASELINE

**Policy Path:** `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml`

**Fingerprint:**
- **Policy Hash:** `d9da2c864eb77678`
- **Entry Config:** `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID/ENTRY_V9_FARM_V2B_PROD.yaml`
- **Exit Config:** `gx1/configs/exits/FARM_EXIT_V2_RULES_A_mp68_max5_rule5_cd0.yaml`
- **Router Version:** `HYBRID_ROUTER_V3`
- **V3 Range Edge Cutoff:** `1.0`
- **Frozen Date:** 2025-12-15

**Guardrails:**
- RULE6A only allowed when `range_edge_dist_atr < 1.0`
- Fail-closed if router model missing
- Fail-closed if feature manifest mismatch
- PROD_BASELINE blocks trading if `cached_bars < warmup_bars` (288)

---

## Pipeline Flow (End-to-End)

### 1. Data Sources
```
Historical: data/raw/xauusd_m5_*.parquet (bid/ask OHLC)
Live:       OANDA REST API (gx1/execution/oanda_client.py)
Backfill:   gx1/execution/oanda_backfill.py (target-driven paging)
```

### 2. Signal Generation
```
Entry Features:  gx1/features/runtime_v9.py → build_v9_runtime_features()
Live Features:   gx1/execution/live_features.py → build_live_entry_features()
Regime:          gx1/regime/farm_regime.py → FARM_V2B regime detection
```

### 3. Policy Application
```
Entry Policy:    gx1/policy/entry_v9_policy_farm_v2b.py → EntryV9PolicyFarmV2B
Exit Policy:     gx1/policy/exit_farm_v2_rules.py → ExitFarmV2Rules
Hybrid Router:   gx1/core/hybrid_exit_router.py → HybridExitRouterV3
```

### 4. Execution
```
Runner:          gx1/execution/oanda_demo_runner.py → GX1DemoRunner
Entry Manager:   gx1/execution/entry_manager.py → EntryManager
Exit Manager:    gx1/execution/exit_manager.py → ExitManager
Broker Client:   gx1/execution/broker_client.py → BrokerClient
```

### 5. Journaling
```
Trade Journal:   gx1/monitoring/trade_journal.py → TradeJournal
Location:        gx1/wf_runs/<run_tag>/trade_journal/
Format:          JSON per trade + CSV index
```

### 6. Replay/Eval
```
Replay Engine:   gx1/execution/replay_engine.py → ReplayEngine
Analysis:        gx1/analysis/prod_baseline_proof.py, parity_audit.py
```

---

## Entry Points

### Live Trading

**1. PROD_BASELINE Practice-Live**
- **Script:** `scripts/go_practice_live_asia.sh`
- **Policy:** `prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_..._PROD.yaml`
- **Mode:** LIVE (dry_run=false)
- **Preflight:** Hard checks for OANDA_ENV, credentials, I_UNDERSTAND_LIVE_TRADING
- **Post-run:** Runs `prod_baseline_proof.py` and `reconcile_oanda.py`

**2. CANARY Force-One-Trade**
- **Script:** `scripts/run_live_force_one_trade_any_session.sh`
- **Policy:** `active/GX1_V11_OANDA_PRACTICE_LIVE_FORCE_ONE_TRADE.yaml`
- **Mode:** LIVE (dry_run=false, meta.role=CANARY)
- **Purpose:** Plumbing verification only
- **Preflight:** Hard checks for OANDA_ENV=practice

**3. Practice-Live Micro**
- **Script:** `scripts/run_practice_live_micro.sh`
- **Policy:** Configurable (defaults to active policy)

### Replay

**1. Standard Replay**
- **Script:** `scripts/run_replay.sh`
- **Entry Point:** `GX1DemoRunner.run_replay()`
- **Data:** `data/raw/xauusd_m5_*.parquet`

**2. Parallel Replay**
- **Script:** `scripts/active/run_replay.sh`
- **Entry Point:** `scripts/active/replay_entry_exit_parallel.py`

### Analysis

**1. Parity Audit**
- **Script:** `gx1/analysis/parity_audit.py`
- **Purpose:** Compare runs for parity (FULLYEAR/CANARY/LIVE)
- **Output:** `gx1/analysis/parity_audit_report.md`

**2. Baseline Proof**
- **Script:** `gx1/analysis/prod_baseline_proof.py`
- **Purpose:** Verify run matches PROD_BASELINE
- **Output:** `<run_dir>/prod_baseline_proof.md`

**3. Runs Inventory**
- **Script:** `gx1/scripts/audit_runs_inventory.py`
- **Purpose:** Scan all runs and generate inventory
- **Output:** `gx1/wf_runs/_inventory.json`, `_inventory.csv`

---

## Configuration Hierarchy

### Source of Truth: PROD_BASELINE

**Location:** `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/`

**Detection Logic:** `gx1/execution/oanda_demo_runner.py:1305-1310`
```python
policy_role = self.policy.get("meta", {}).get("role", "")
self.prod_baseline = (policy_role == "PROD_BASELINE")
```

**Active Policies:** `gx1/configs/policies/active/`
- Used when policy path points to `active/`
- CANARY policy: `GX1_V11_OANDA_PRACTICE_LIVE_FORCE_ONE_TRADE.yaml`

**Path Resolution:** `gx1/prod/path_resolver.py`
- PROD_BASELINE: Resolves relative to `gx1/prod/current/`
- Non-PROD: Uses paths as-is

---

## Model/Policy Artifacts

### Entry Models
- **Location:** `gx1/models/entry_v9/`
- **Loading:** `GX1DemoRunner._load_entry_v9_model()`
- **Versioning:** Tracked in `run_header.json` → `artifacts.entry_models[].sha256`

### Router Models
- **PROD_BASELINE:** `prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/exit_router_models_v3/exit_router_v3_tree.pkl`
- **Version:** HYBRID_ROUTER_V3
- **Guardrail:** `v3_range_edge_cutoff=1.0`
- **Versioning:** Tracked in `run_header.json` → `artifacts.router_model.sha256`

### Feature Manifest
- **Location:** `gx1/models/entry_v9/entry_v9_feature_meta.json`
- **Purpose:** Validates feature columns match model expectations
- **Versioning:** Tracked in `run_header.json` → `artifacts.feature_manifest.sha256`

---

## Runtime State

### Run Directories

**Location:** `gx1/wf_runs/<run_tag>/`

**Structure:**
```
gx1/wf_runs/
├── <run_tag>/
│   ├── run_header.json          # Artifact hashes, git commit, config paths
│   ├── trade_journal/
│   │   ├── trade_journal_index.csv
│   │   └── trades/
│   │       └── <trade_id>.json
│   ├── reconciliation_report.md
│   ├── prod_baseline_proof.md   # If PROD_BASELINE run
│   └── logs/
│       └── oanda_demo_runner.log
```

**Run Tag Format:** `<PREFIX>_<YYYYMMDD_HHMMSS>`

**Creation:** `GX1DemoRunner.__init__()` → `output_dir = Path("gx1/wf_runs") / self.run_id`

---

## Runs Inventory Analysis

### By Category

**LIVE_FORCE (Force-One-Trade Plumbing Tests)**
- **Count:** ~30+ runs
- **Purpose:** CANARY plumbing verification
- **Status:** Most are empty (no trades, <1 MB)
- **Action:** DELETE candidates (>1 day old)

**GO_PRACTICE (PROD_BASELINE Practice Runs)**
- **Count:** ~3 runs
- **Purpose:** PROD_BASELINE practice-live testing
- **Status:** Some have journals
- **Action:** KEEP (PROD_BASELINE runs)

**EXEC_SMOKE (Execution Smoke Tests)**
- **Count:** ~2 runs
- **Purpose:** Execution pipeline verification
- **Status:** Has journals
- **Action:** KEEP (milestone runs)

**OBS_REPLAY (Observation Replays)**
- **Count:** ~1 run
- **Purpose:** PROD_BASELINE observation replay
- **Status:** Has journal
- **Action:** KEEP (PROD_BASELINE run)

**FULLYEAR (Full Year Baselines)**
- **Count:** Multiple runs
- **Purpose:** Full year baseline verification
- **Status:** Large runs with journals
- **Action:** KEEP (milestone runs)

**DETERMINISM (Determinism Gates)**
- **Count:** ~1 run
- **Purpose:** Determinism verification
- **Status:** Has reports
- **Action:** KEEP (milestone run)

**EXIT_SWEEP (Exit Parameter Sweeps)**
- **Count:** Multiple runs
- **Purpose:** Exit tuning sweeps
- **Status:** Old tuning runs
- **Action:** ARCHIVE (>60 days old)

**OTHER**
- **Count:** Remaining runs
- **Status:** Mixed (some have journals, some don't)
- **Action:** Review individually

### Top 10 Newest Runs

1. `LIVE_FORCE_ANY_20251217_092846` - 0.00 MB, 0 trades, Role: None
2. `LIVE_FORCE_ANY_20251217_092335` - 0.00 MB, 0 trades, Role: None
3. `LIVE_FORCE_ANY_20251217_092214` - 0.00 MB, 0 trades, Role: None
4. `LIVE_FORCE_ANY_20251217_090727` - 0.00 MB, 0 trades, Role: None
5. `LIVE_FORCE_ANY_20251217_090120` - 0.00 MB, 0 trades, Role: None
6. `LIVE_FORCE_ANY_20251217_085610` - 0.00 MB, 0 trades, Role: None
7. `LIVE_FORCE_ANY_20251217_085046` - 0.00 MB, 0 trades, Role: None
8. `LIVE_FORCE_ANY_20251217_084920` - 0.00 MB, 0 trades, Role: None
9. `LIVE_FORCE_ANY_20251217_084802` - 0.00 MB, 0 trades, Role: None
10. `LIVE_FORCE_ANY_20251217_084513` - 0.00 MB, 0 trades, Role: None

**Observation:** All newest runs are empty LIVE_FORCE runs (plumbing tests that didn't complete).

---

## Prune Plan

### KEEP (Locked)

**Always Keep:**
1. **PROD_BASELINE runs:** Any run with `meta.role == PROD_BASELINE` in `run_header.json`
2. **Last 10 runs:** Top 10 newest runs (regardless of size)
3. **Milestone runs:**
   - `FULLYEAR_*` (full year baselines)
   - `DETERMINISM_GATE_*` (determinism verification)
   - `OBS_REPLAY_PROD_BASELINE_*` (observation replays)
   - `EXEC_SMOKE_*` (execution smoke tests)
4. **Referenced runs:** Runs directly referenced by active YAML configs or documentation

**Criteria:**
- Has `run_header.json` with valid fingerprints
- Has trade journal with trades
- Has reconciliation report (for LIVE runs)
- Matches current PROD_BASELINE fingerprint

### ARCHIVE (Move to Archive)

**Candidates:**
1. **Old runs (>30 days):**
   - Not referenced by active pointers
   - Not PROD_BASELINE
   - Not in top 10 newest
   - Has trade journal (historical value)

2. **Tuning sweeps:**
   - Runs with tags like `EXIT_A_SWEEP_*`, `ROUTER_THRESHOLD_*`
   - Older than 60 days
   - Not referenced in analysis reports

**Archive Location:** `gx1/wf_runs/_archive/` (or external disk)

**Action:** Move (don't delete) to preserve history

### DELETE CANDIDATES (Safe to Delete)

**Criteria:**
1. **Empty/partial runs:**
   - No `run_header.json`
   - No trade journal
   - Size < 1 MB
   - Age > 7 days

2. **Crash runs:**
   - No trades (n_trades=0)
   - No journal files
   - Size < 1 MB
   - Age > 1 day

3. **Duplicates:**
   - Same fingerprint (policy_hash + git_commit + guardrail_params)
   - Same date range
   - Keep newest, delete older

4. **Test runs:**
   - Tags like `LIVE_FORCE_*` (force-one-trade plumbing tests)
   - No trades
   - Age > 1 day

**Current Inventory:**
- **Obsolete candidates:** 32 runs
- **Estimated space recovery:** ~32 MB (mostly empty directories)

**Example Delete Candidates:**
```
LIVE_FORCE_ANY_20251217_092846  (0.00 MB, no journal, today)
LIVE_FORCE_ANY_20251217_092335  (0.00 MB, no journal, today)
... (30 more similar)
```

---

## Anti-Fail Baseline Guard

### Implementation

**Location:** `gx1/execution/oanda_demo_runner.py:2611-2635`

**Logging:** At startup, logs one line with:
```
[BASELINE_FINGERPRINT] Config=<path> PolicyHash=<hash> Role=<role> Guardrail=<cutoff> Git=<commit> RunTag=<tag>
```

**Example:**
```
[BASELINE_FINGERPRINT] Config=gx1/configs/policies/prod_snapshot/.../GX1_V11_..._PROD.yaml PolicyHash=d9da2c864eb77678 Role=PROD_BASELINE Guardrail=1.0 Git=f04f56a RunTag=GO_PRACTICE_20251217_100000
```

### Baseline Validation

**Check:** `gx1/execution/oanda_demo_runner.py:2243-2266`

**PROD_BASELINE Mode:**
1. Load policy from `prod_snapshot/`
2. Verify router model exists at expected path
3. Verify feature manifest matches
4. Fail-closed if any check fails

**CANARY Mode:**
- Allows degraded warmup
- Logs warnings but doesn't fail

### Run Header Verification

**Location:** `gx1/prod/run_header.py:generate_run_header()`

**Contents:** SHA256 hashes of:
- Policy file
- Router model
- Entry models
- Feature manifest

**Usage:** `audit_runs_inventory.py` compares run fingerprints against baseline

---

## Key Findings

### Critical Issues

1. **No PROD_BASELINE runs identified:**
   - Inventory scanner found 0 runs with `meta.role == PROD_BASELINE`
   - Possible reasons:
     - `run_header.json` doesn't exist in older runs
     - `meta.role` not set correctly in run_header
     - Need to check `OBS_REPLAY_PROD_BASELINE_*` runs manually

2. **High percentage of empty runs:**
   - 27.1% of runs are obsolete candidates
   - Most are `LIVE_FORCE_*` plumbing tests that didn't complete
   - Safe to delete after 1 day

3. **Low journal coverage:**
   - Only 7.6% of runs have trade journals
   - Suggests many runs are incomplete or test runs

### Recommendations

1. **Immediate Actions:**
   - Review `OBS_REPLAY_PROD_BASELINE_*` runs to verify PROD_BASELINE status
   - Delete empty `LIVE_FORCE_*` runs older than 1 day (~30 runs, ~0 MB)
   - Archive old tuning sweeps (>60 days)

2. **Short-term:**
   - Ensure all new runs generate `run_header.json` with `meta.role`
   - Verify baseline fingerprint logging works for all new runs
   - Set up weekly inventory scans

3. **Long-term:**
   - Automate prune plan execution
   - Set up alerts for baseline mismatches
   - Document run retention policy

---

## Commands Reference

### Generate Inventory
```bash
python3 gx1/scripts/audit_runs_inventory.py
```

### View Baseline Fingerprint
```bash
python3 gx1/scripts/audit_runs_inventory.py | grep -A 10 "CURRENT PROD_BASELINE"
```

### List Delete Candidates (Sorted by Size)
```bash
python3 << 'PY'
import pandas as pd
df = pd.read_csv('gx1/wf_runs/_inventory.csv')
candidates = df[(~df['has_trade_journal']) & (df['size_mb'] < 1.0)].sort_values('size_mb')
print(candidates[['run_name', 'size_mb', 'mtime']].to_string())
PY
```

### Archive Old Runs (>30 days, not PROD_BASELINE)
```bash
mkdir -p gx1/wf_runs/_archive
find gx1/wf_runs -maxdepth 1 -type d -mtime +30 ! -name "_*" -exec sh -c '
  run_dir="$1"
  if [ -f "$run_dir/run_header.json" ]; then
    role=$(jq -r ".meta.role // empty" "$run_dir/run_header.json")
    if [ "$role" != "PROD_BASELINE" ]; then
      echo "Archiving: $run_dir"
      mv "$run_dir" gx1/wf_runs/_archive/
    fi
  fi
' _ {} \;
```

### Verify Baseline Alignment
```bash
python3 gx1/analysis/prod_baseline_proof.py \
  --run <run_dir> \
  --prod-policy gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml
```

### Delete Empty Runs (>1 day old)
```bash
find gx1/wf_runs -maxdepth 1 -type d -name "LIVE_FORCE_*" -mtime +1 -exec rm -rf {} \;
```

---

## Documentation Files

1. **docs/PIPELINE_AUDIT.md** - Complete pipeline documentation
2. **docs/RUN_HYGIENE.md** - Quick reference for run maintenance
3. **docs/AUDIT_SUMMARY.md** - This file (consolidated summary)
4. **gx1/wf_runs/_inventory.json** - Detailed inventory (JSON)
5. **gx1/wf_runs/_inventory.csv** - Summary inventory (CSV)

---

## Next Steps

1. **Review PROD_BASELINE runs:** Check `OBS_REPLAY_PROD_BASELINE_*` runs manually
2. **Clean up empty runs:** Delete `LIVE_FORCE_*` runs older than 1 day
3. **Archive old runs:** Move tuning sweeps and old runs to archive
4. **Set up automation:** Weekly inventory scans and alerts
5. **Document retention:** Finalize run retention policy

---

**Last Updated:** 2025-12-17  
**Audit Tool:** `gx1/scripts/audit_runs_inventory.py`  
**Baseline:** `prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml`

