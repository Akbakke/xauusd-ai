# GX1/FARM Pipeline Hygiene Audit

**Last Updated:** 2025-12-17  
**Purpose:** End-to-end pipeline mapping, run inventory, and hygiene plan

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Entry Points](#entry-points)
3. [Configuration Hierarchy](#configuration-hierarchy)
4. [Model/Policy Artifacts](#modelpolicy-artifacts)
5. [Runtime State](#runtime-state)
6. [Kill-Switch / Preflight](#kill-switch--preflight)
7. [Prune Plan](#prune-plan)
8. [Anti-Fail Baseline Guard](#anti-fail-baseline-guard)

---

## Pipeline Overview

The GX1/FARM pipeline flows as follows:

```
Data → Signal → Policy → Execution → Journal → Replay/Eval
```

### Data Sources
- **Historical:** `data/raw/xauusd_m5_*.parquet` (bid/ask OHLC)
- **Live:** OANDA REST API (`gx1/execution/oanda_client.py`)
- **Backfill:** `gx1/execution/oanda_backfill.py` (target-driven paging)

### Signal Generation
- **Entry Features:** `gx1/features/runtime_v9.py` → `build_v9_runtime_features()`
- **Live Features:** `gx1/execution/live_features.py` → `build_live_entry_features()`
- **Regime Inference:** `gx1/regime/farm_regime.py` → FARM_V2B regime detection

### Policy Application
- **Entry Policy:** `gx1/policy/entry_v9_policy_farm_v2b.py` → `EntryV9PolicyFarmV2B`
- **Exit Policy:** `gx1/policy/exit_farm_v2_rules.py` → `ExitFarmV2Rules`
- **Hybrid Router:** `gx1/core/hybrid_exit_router.py` → `HybridExitRouterV3`

### Execution
- **Runner:** `gx1/execution/oanda_demo_runner.py` → `GX1DemoRunner`
- **Entry Manager:** `gx1/execution/entry_manager.py` → `EntryManager`
- **Exit Manager:** `gx1/execution/exit_manager.py` → `ExitManager`
- **Broker Client:** `gx1/execution/broker_client.py` → `BrokerClient` (wraps OandaClient)

### Journaling
- **Trade Journal:** `gx1/monitoring/trade_journal.py` → `TradeJournal`
- **Location:** `gx1/wf_runs/<run_tag>/trade_journal/`
- **Format:** JSON per trade + CSV index

### Replay/Eval
- **Replay Engine:** `gx1/execution/replay_engine.py` → `ReplayEngine`
- **Analysis:** `gx1/analysis/prod_baseline_proof.py`, `parity_audit.py`

---

## Entry Points

### Live Trading (Practice/Canary/Prod)

**Script:** `scripts/go_practice_live_asia.sh`
- **Purpose:** PROD_BASELINE practice-live start
- **Policy:** `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml`
- **Mode:** LIVE (dry_run=false)
- **Preflight:** Hard checks for OANDA_ENV, credentials, I_UNDERSTAND_LIVE_TRADING
- **Post-run:** Runs `prod_baseline_proof.py` and `reconcile_oanda.py`

**Script:** `scripts/run_live_force_one_trade_any_session.sh`
- **Purpose:** CANARY force-one-trade plumbing verification
- **Policy:** `gx1/configs/policies/active/GX1_V11_OANDA_PRACTICE_LIVE_FORCE_ONE_TRADE.yaml`
- **Mode:** LIVE (dry_run=false, meta.role=CANARY)
- **Preflight:** Hard checks for OANDA_ENV=practice
- **Post-run:** Verifies trade journal artifacts

**Script:** `scripts/run_practice_live_micro.sh`
- **Purpose:** Practice-live micro runs
- **Policy:** Configurable (defaults to active policy)

### Replay

**Script:** `scripts/run_replay.sh`
- **Purpose:** Historical replay with frozen configs
- **Entry Point:** `gx1/execution/oanda_demo_runner.py` → `run_replay()`
- **Data:** `data/raw/xauusd_m5_*.parquet`

**Script:** `scripts/active/run_replay.sh`
- **Purpose:** Parallel replay with chunking
- **Entry Point:** `scripts/active/replay_entry_exit_parallel.py`

### Analysis

**Script:** `gx1/analysis/parity_audit.py`
- **Purpose:** Compare runs for parity (FULLYEAR/CANARY/LIVE)
- **Output:** `gx1/analysis/parity_audit_report.md`

**Script:** `gx1/analysis/prod_baseline_proof.py`
- **Purpose:** Verify run matches PROD_BASELINE
- **Output:** `<run_dir>/prod_baseline_proof.md`

**Script:** `gx1/scripts/audit_runs_inventory.py`
- **Purpose:** Scan all runs and generate inventory
- **Output:** `gx1/wf_runs/_inventory.json`, `_inventory.csv`

---

## Configuration Hierarchy

### Source of Truth: PROD_BASELINE

**Location:** `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml`

**Key Fields:**
```yaml
meta:
  role: PROD_BASELINE
  router_version: V3_RANGE
  frozen_date: 2025-12-15

entry_config: gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID/ENTRY_V9_FARM_V2B_PROD.yaml
exit_config: gx1/configs/exits/FARM_EXIT_V2_RULES_A_mp68_max5_rule5_cd0.yaml

hybrid_exit_router:
  version: HYBRID_ROUTER_V3
  model_path: gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/exit_router_models_v3/exit_router_v3_tree.pkl
  v3_range_edge_cutoff: 1.0
```

**Detection Logic:** `gx1/execution/oanda_demo_runner.py:1305-1310`
```python
# Check if policy is PROD_BASELINE
policy_role = self.policy.get("meta", {}).get("role", "")
self.prod_baseline = (policy_role == "PROD_BASELINE")
```

### Active Policies

**Location:** `gx1/configs/policies/active/`

**Usage:** Loaded by `GX1DemoRunner.__init__()` when policy path points to `active/`

**CANARY Policy:** `GX1_V11_OANDA_PRACTICE_LIVE_FORCE_ONE_TRADE.yaml`
- **Role:** CANARY (not PROD_BASELINE)
- **Purpose:** Plumbing verification only
- **Force Entry:** Enabled with timeout

### Configuration Loading

**Flow:** `gx1/execution/oanda_demo_runner.py:1335-1340`
1. Load policy YAML from path
2. Resolve entry_config and exit_config paths
3. Load entry/exit configs
4. Resolve router model path (if PROD_BASELINE, use prod_snapshot path)

**Path Resolution:** `gx1/prod/path_resolver.py`
- PROD_BASELINE: Resolves relative to `gx1/prod/current/`
- Non-PROD: Uses paths as-is

---

## Model/Policy Artifacts

### Entry Models

**Location:** `gx1/models/entry_v9/`

**Loading:** `gx1/execution/oanda_demo_runner.py:_load_entry_v9_model()`
- Loads transformer model, scalers, metadata
- Feature manifest: `entry_v9_feature_meta.json`

**Versioning:** Tracked in `run_header.json` → `artifacts.entry_models[].sha256`

### Router Models

**PROD_BASELINE Router:**
- **Path:** `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/exit_router_models_v3/exit_router_v3_tree.pkl`
- **Version:** HYBRID_ROUTER_V3
- **Guardrail:** `v3_range_edge_cutoff=1.0`

**Loading:** `gx1/core/hybrid_exit_router.py:HybridExitRouterV3.load()`

**Versioning:** Tracked in `run_header.json` → `artifacts.router_model.sha256`

### Feature Manifest

**Location:** `gx1/models/entry_v9/entry_v9_feature_meta.json`

**Usage:** Validates feature columns match model expectations

**Versioning:** Tracked in `run_header.json` → `artifacts.feature_manifest.sha256`

### Baseline Selection

**Current Baseline:** `2025_FARM_V2B_HYBRID_V3_RANGE`
- **Frozen:** 2025-12-15
- **Router:** V3_RANGE with guardrail
- **Performance:** 117.67 bps EV/trade, 84.6% win rate

**Guardrails:**
- `v3_range_edge_cutoff=1.0` (RULE6A only when `range_edge_dist_atr < 1.0`)
- Fail-closed if router model missing
- Fail-closed if feature manifest mismatch

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
- Examples: `LIVE_FORCE_ANY_20251217_092846`, `GO_PRACTICE_20251216_222802`

**Creation:** `gx1/execution/oanda_demo_runner.py:3796`
```python
output_dir = Path("gx1/wf_runs") / self.run_id
```

### Trade Journal

**Implementation:** `gx1/monitoring/trade_journal.py`

**Per-Trade JSON:** `trade_journal/trades/<trade_id>.json`
- Entry snapshot (reason, test_mode, warmup_degraded)
- Feature context (ATR, spread, range)
- Router decision
- Exit lifecycle
- Execution events

**Index CSV:** `trade_journal/trade_journal_index.csv`
- Aggregated view for filtering/analysis

### Reconciliation

**Script:** `gx1/monitoring/reconcile_oanda.py`

**Output:** `<run_dir>/reconciliation_report.md`
- Matches internal trades with OANDA transactions
- Reports fill price differences (bps)
- Verifies execution events

### Run Header

**Generation:** `gx1/prod/run_header.py:generate_run_header()`

**Location:** `<run_dir>/run_header.json`

**Contents:**
- `run_tag`: Run identifier
- `git_commit`: Git commit hash at run time
- `config_path`: Policy YAML path
- `artifacts`: SHA256 hashes of policy, models, manifests
- `meta.role`: PROD_BASELINE or CANARY

**Usage:** Used by `audit_runs_inventory.py` to fingerprint runs

---

## Kill-Switch / Preflight

### Preflight Checks

**Location:** Entry point scripts (e.g., `scripts/go_practice_live_asia.sh:64-92`)

**Checks:**
1. `OANDA_ENV` must be `practice` (for practice scripts)
2. `OANDA_API_TOKEN` must be set
3. `OANDA_ACCOUNT_ID` must be set
4. `I_UNDERSTAND_LIVE_TRADING` must NOT be set (for practice)

**Failure:** Hard exit with error message

### Runtime Guards

**PROD_BASELINE Guards:** `gx1/execution/oanda_demo_runner.py:2243-2266`
- Credentials required (fail-closed)
- Feature manifest mismatch → fail-closed
- Router model missing → fail-closed

**Warmup Gate:** `gx1/execution/oanda_demo_runner.py:3618-3667`
- PROD_BASELINE: Blocks trading if `cached_bars < warmup_bars` (288)
- CANARY: Allows degraded warmup if `cached_bars >= min_start_bars` (100)

**Policy Lock:** `gx1/execution/oanda_demo_runner.py` (if implemented)
- Detects policy file changes during run
- Disables trading if hash mismatch

---

## Prune Plan

### KEEP (Locked)

**Always Keep:**
1. **PROD_BASELINE runs:** Any run with `meta.role == PROD_BASELINE` in `run_header.json`
2. **Last N runs:** Top 10 newest runs (regardless of size)
3. **Milestone runs:** Runs with unique tags:
   - `FULLYEAR_*` (full year baselines)
   - `DETERMINISM_GATE_*` (determinism verification)
   - `OBS_REPLAY_PROD_BASELINE_*` (observation replays)
   - `EXEC_SMOKE_*` (execution smoke tests)
4. **Referenced runs:** Runs directly referenced by:
   - Active YAML configs
   - `prod_snapshot/` policies
   - Documentation (e.g., `docs/` references)

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

**Current Inventory (from audit):**
- **Total runs:** 118
- **Total size:** 4467.83 MB
- **Runs with journal:** 9
- **Obsolete candidates:** 32 (empty/partial runs)

**Example Delete Candidates:**
```
LIVE_FORCE_ANY_20251217_092846  (0.00 MB, no journal, today)
LIVE_FORCE_ANY_20251217_092335  (0.00 MB, no journal, today)
... (30 more similar)
```

**Safe Delete Command:**
```bash
# Review candidates first
python3 gx1/scripts/audit_runs_inventory.py | grep "OBSOLETE CANDIDATES"

# Delete empty runs older than 1 day
find gx1/wf_runs -name "LIVE_FORCE_*" -type d -mtime +1 -exec rm -rf {} \;
```

---

## Anti-Fail Baseline Guard

### Implementation

**Location:** `gx1/execution/oanda_demo_runner.py:2611-2616`

**Logging:** At startup, logs one line with:
- Config path
- Policy bundle ID (policy_hash)
- Guardrail params (v3_range_edge_cutoff)
- Git commit
- Run tag

**Example Log:**
```
[BOOT] Config: gx1/configs/policies/prod_snapshot/.../GX1_V11_..._PROD.yaml
[BOOT] Policy Hash: d9da2c864eb77678
[BOOT] Guardrail: v3_range_edge_cutoff=1.0
[BOOT] Git Commit: f04f56a...
[BOOT] Run Tag: GO_PRACTICE_20251217_100000
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

## Commands

### Generate Inventory

```bash
# Run audit scanner
python3 gx1/scripts/audit_runs_inventory.py

# View inventory JSON
cat gx1/wf_runs/_inventory.json | jq '.summary'

# View inventory CSV
head -20 gx1/wf_runs/_inventory.csv
```

### View Active Baseline Fingerprint

```bash
# From inventory
python3 gx1/scripts/audit_runs_inventory.py | grep -A 10 "CURRENT PROD_BASELINE"

# From policy file
cat gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml | grep -E "role:|router_version:|v3_range_edge_cutoff:"
```

### List Delete Candidates (Sorted by Size)

```bash
# From CSV inventory
python3 << 'PY'
import pandas as pd
df = pd.read_csv('gx1/wf_runs/_inventory.csv')
candidates = df[
    (~df['has_trade_journal']) & 
    (df['size_mb'] < 1.0)
].sort_values('size_mb')
print(candidates[['run_name', 'size_mb', 'mtime', 'n_trades']].to_string())
PY

# Or from JSON
cat gx1/wf_runs/_inventory.json | jq '.summary.obsolete_candidates[] | {run_name, size_mb, reason}' | head -20
```

### Archive Candidates

```bash
# Create archive directory
mkdir -p gx1/wf_runs/_archive

# Move old runs (>30 days, not PROD_BASELINE)
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
# Check if run matches PROD_BASELINE
python3 gx1/analysis/prod_baseline_proof.py \
  --run gx1/wf_runs/OBS_REPLAY_PROD_BASELINE_20251215_211532 \
  --prod-policy gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml
```

---

## Maintenance

### Regular Hygiene Tasks

1. **Weekly:** Run `audit_runs_inventory.py` and review obsolete candidates
2. **Monthly:** Archive old runs (>30 days, not PROD_BASELINE)
3. **Quarterly:** Review PROD_BASELINE runs and verify they're still referenced
4. **Before major releases:** Freeze new PROD_BASELINE and archive old one

### Keeping Runs Clean

**Automation:** Add to `.gitignore`:
```
gx1/wf_runs/_inventory.json
gx1/wf_runs/_inventory.csv
gx1/wf_runs/_archive/
```

**Documentation:** Update this file when pipeline changes

**Verification:** Always verify baseline fingerprint before live runs

