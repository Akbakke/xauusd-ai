# GX1 Runbook - Operational Guidelines

**Last Updated:** 2025-12-15

---

## Parity & Canary Rule (P0 - Critical)

### Rule: Canary = SINGLE Worker

**For alle canary-runs, practice-live og beslutnings-testing skal `n_workers=1` brukes.**

**Rationale:**
- Parallel replay har chunk/state leakage som gir falske trades
- Single-worker er deterministisk og reproduserbar
- Parallel replay er et performance-verktøy, ikke et sannhets-verktøy

**When to use `n_workers=1`:**
- ✅ Canary testing
- ✅ Practice-live testing
- ✅ Decision testing (parity checks)
- ✅ Any run where reproducibility matters

**When parallel replay (`n_workers>1`) is acceptable:**
- ✅ Grov backfill / statistikk
- ✅ Initial FULLYEAR runs for model selection
- ✅ Performance testing (speed only)

**Implementation:**
```bash
# CORRECT: Canary with single worker
bash scripts/run_replay.sh <policy> <start> <end> 1 <output_dir>

# WRONG: Canary with parallel workers
bash scripts/run_replay.sh <policy> <start> <end> 7 <output_dir>
```

**Verification:**
- Use `gx1/analysis/determinism_gate.py` to verify determinism
- SINGLE vs SINGLE_REPEAT should match 100%
- SINGLE vs PARALLEL may differ (expected)

**This protects against future self-deception.**

---

## FULLYEAR Results Validity

**Important:** Chunk/state leakage does NOT disqualify FULLYEAR analysis.

**Why FULLYEAR results are still valid:**
- FULLYEAR was used for model and architecture selection
- NOT for canary-parity or decision testing
- It was aggregate direction, not microscopic reproduction

**What we learned:**
- "Don't use PARALLEL for parity"
- This is normal in stateful trading pipelines

---

## Canary Testing Workflow

### Step 1: Run Canary (Single Worker)

```bash
export OANDA_ENV=practice
export OANDA_API_TOKEN=<token>
export OANDA_ACCOUNT_ID=<account_id>

# Use n_workers=1 for canary
bash scripts/run_replay.sh \
  gx1/prod/current/policy_canary.yaml \
  2025-01-01 \
  2025-01-15 \
  1 \
  gx1/wf_runs/CANARY_TEST_<date>
```

### Step 2: Verify Determinism

```bash
python gx1/analysis/determinism_gate.py \
  --policy gx1/prod/current/policy_canary.yaml \
  --start 2025-01-01 \
  --end 2025-01-15 \
  --out gx1/wf_runs/DETERMINISM_CHECK
```

### Step 3: Compare with Baseline

```bash
python gx1/analysis/diff_runs.py \
  --baseline-run gx1/wf_runs/BASELINE_TAG \
  --canary-run gx1/wf_runs/CANARY_TEST_<date> \
  --out gx1/wf_runs/CANARY_TEST_<date>/run_diff_report.md \
  --start-date 2025-01-01 \
  --end-date 2025-01-15
```

---

## Production Deployment Checklist

1. ✅ Policy frozen in `gx1/prod/current/`
2. ✅ Router model frozen
3. ✅ Feature manifest frozen
4. ✅ Canary test passed (n_workers=1)
5. ✅ Determinism verified
6. ✅ Parity check passed
7. ✅ OANDA credentials configured
8. ✅ Live trading latch set (if live)

---

## Troubleshooting

### Issue: Trade count mismatch between runs

**Check:**
1. Are both runs using `n_workers=1`?
2. Is price data identical (check SHA256)?
3. Is policy identical (check run_header.json)?

**Solution:**
- Use `determinism_gate.py` to diagnose
- Ensure `n_workers=1` for parity testing

### Issue: Canary generates different trades than baseline

**Check:**
1. Policy differences (use `diff_runs.py`)
2. Entry config differences
3. Price data differences

**Solution:**
- Use `diff_runs.py` to identify differences
- Ensure same policy/config for both runs

---

## Trade Journal (Production-Ready)

### Overview

Production-ready structured trade journal that provides complete traceability for each trade lifecycle. Enables post-hoc analysis to understand exactly why each trade was taken, routed, and closed.

### Location

**Per-Trade JSON Files:**
- `gx1/wf_runs/<TAG>/trade_journal/trades/<trade_id>.json`

**Aggregated Index CSV:**
- `gx1/wf_runs/<TAG>/trade_journal/trade_journal_index.csv`

**JSONL Events (Backward Compatibility):**
- `gx1/wf_runs/<TAG>/trade_journal/trade_journal.jsonl`

### Structure

Each trade JSON file contains:

1. **Entry Snapshot** (why trade was taken):
   - `trade_id`, `entry_time`, `instrument`, `side`, `entry_price`
   - `session`, `regime`, `entry_model_version`
   - `entry_score` (p_long, p_short, p_hat, margin)
   - `entry_filters_passed`, `entry_filters_blocked`

2. **Feature Context** (immutable snapshot at entry):
   - ATR: `atr_bps`, `atr_price`, `atr_percentile`
   - Range: `range_pos`, `distance_to_range`, `range_edge_dist_atr`
   - Spread: `spread_price`, `spread_pct`
   - Candle: `close`, `high`, `low` (last closed bar)

3. **Router Explainability**:
   - `router_version`, `router_model_hash`
   - `router_features_used` (all features passed to router)
   - `router_raw_decision` (RULE5 or RULE6A)
   - `guardrail_applied`, `guardrail_reason`, `guardrail_cutoff`
   - `final_exit_profile` (after guardrail)

4. **Exit Configuration**:
   - `exit_profile`
   - `tp_levels`, `sl`
   - `trailing_enabled`, `be_rules`

5. **Exit Events** (append-only timeline):
   - `timestamp`, `event_type`, `price`, `pnl_bps`, `bars_held`
   - Event types: `TP1_HIT`, `BE_MOVED`, `TRAIL_ACTIVATED`, `EXIT_TRIGGERED`

6. **Exit Summary** (final closure):
   - `exit_time`, `exit_price`, `exit_reason`
   - `realized_pnl_bps`
   - `max_mfe_bps`, `max_mae_bps`, `intratrade_drawdown_bps`

7. **Execution Events** (OANDA practice-live only):
   - `ORDER_SUBMITTED`: Order sent to OANDA API
   - `ORDER_FILLED`: Order filled (immediate or later)
   - `ORDER_REJECTED`: Order rejected by OANDA
   - `TRADE_OPENED_OANDA`: Trade opened in OANDA (with OANDA trade ID)
   - `TRADE_CLOSED_OANDA`: Trade closed in OANDA (with closing transaction ID)
   
   **Execution Event Fields:**
   - `oanda_env`: Environment (practice/live)
   - `account_id_masked`: Masked account ID (e.g., `101-***-001`)
   - `instrument`, `side`, `units`
   - `order_type`, `client_ext_id` (GX1 identifier: `GX1:{run_tag}:{trade_id}`)
   - `oanda_order_id`, `oanda_trade_id`, `oanda_last_txn_id`
   - `fill_price`, `commission`, `financing`
   - `ts_oanda`, `ts_local` (timestamps)
   
   **Intratrade Metrics Definitions:**
   - **MFE (Max Favorable Excursion)**: Maximum favorable unrealized PnL during trade lifetime (in bps from entry).
     - For long: uses bar `high` prices (best case per bar)
     - For short: uses bar `low` prices (best case per bar)
   - **MAE (Max Adverse Excursion)**: Maximum adverse unrealized drawdown during trade lifetime (in bps from entry).
     - For long: uses bar `low` prices (worst case per bar)
     - For short: uses bar `high` prices (worst case per bar)
   - **Intratrade Drawdown**: Largest peak-to-trough drawdown on unrealized PnL curve (in bps).
     - Calculated from unrealized PnL curve (using bar `close` prices)
     - Measures maximum drawdown from any peak during trade lifetime
   
   **Data Source:**
   - Metrics are calculated from `_price_trace` collected per bar for open trades
   - Each trace point contains: `ts` (timestamp), `high`, `low`, `close` (from last closed bar)
   - Uses mid-price if bid/ask available, otherwise direct OHLC
   - Trace is limited to last 5000 points to prevent memory bloat
   - Trace is removed from `trade.extra` after metrics calculation (to prevent bloating trade logs)
   
   **Coverage:**
   - Metrics are **guaranteed** for all trades (100% coverage on all exit paths)
   - All exit paths (FARM_V2_RULES, FIXED_BAR_CLOSE, FARM_V1, MODEL_EXIT) log exit_summary with metrics
   - If price trace is missing, metrics will be `None` (logged but not calculated)
   
   **Invariants (Soft Warnings in Prod):**
   - **MFE >= 0**: Favorable excursion cannot be negative (logged as WARNING if violated)
   - **MAE <= 0**: Adverse excursion cannot be positive (logged as WARNING if violated)
   - **Intratrade DD <= 0**: Drawdown is negative or zero (logged as WARNING if violated)
   - **MFE >= realized_pnl** (if realized_pnl > 0): MFE is best case, should be >= final PnL (logged as WARNING if violated)
   - Invariant violations do **not** block trading (soft warnings only)
   
   **Coverage:**
   - Metrics are **guaranteed** for all trades (100% coverage on all exit paths)
   - All exit paths (FARM_V2_RULES, FIXED_BAR_CLOSE, FARM_V1, MODEL_EXIT) log exit_summary with metrics
   - If price trace is missing, metrics will be `None` (logged but not calculated)
   
   **Invariants (Soft Warnings in Prod):**
   - **MFE >= 0**: Favorable excursion cannot be negative (logged as WARNING if violated)
   - **MAE <= 0**: Adverse excursion cannot be positive (logged as WARNING if violated)
   - **Intratrade DD <= 0**: Drawdown is negative or zero (logged as WARNING if violated)
   - **MFE >= realized_pnl** (if realized_pnl > 0): MFE is best case, should be >= final PnL (logged as WARNING if violated)
   - Invariant violations do **not** block trading (soft warnings only)

### Index CSV

The index CSV provides quick filtering and analysis:

| Column | Description |
|--------|-------------|
| `trade_id` | Trade identifier |
| `entry_time` | Entry timestamp |
| `exit_time` | Exit timestamp |
| `side` | Trade side (long/short) |
| `exit_profile` | Final exit profile |
| `pnl_bps` | Realized PnL |
| `guardrail_applied` | Whether guardrail was applied |
| `range_edge_dist_atr` | Range edge distance ATR |
| `router_decision` | Raw router decision |
| `exit_reason` | Exit reason |

### How to Read a Trade

**Step 1: Find trade in index**
```bash
grep "SIM-1234567890-000001" gx1/wf_runs/<TAG>/trade_journal/trade_journal_index.csv
```

**Step 2: Open trade JSON**
```bash
cat gx1/wf_runs/<TAG>/trade_journal/trades/SIM-1234567890-000001.json | jq .
```

**Step 3: Answer key questions**

**Why did we enter?**
- Check `entry_snapshot.entry_score` (model probabilities)
- Check `entry_snapshot.entry_filters_passed` (which gates passed)
- Check `feature_context` (ATR, range, spread at entry)

**What did router choose?**
- Check `router_explainability.router_raw_decision` (RULE5 or RULE6A)
- Check `router_explainability.router_features_used` (features router saw)
- Check `router_explainability.guardrail_applied` (was guardrail triggered?)
- If guardrail applied: check `guardrail_reason` and `range_edge_dist_atr`

**What triggered exit?**
- Check `exit_events` timeline (all events in chronological order)
- Check `exit_summary.exit_reason` (TP, SL, BE, TIMEOUT, TRAIL)
- Check `exit_summary.realized_pnl_bps` (final PnL)

### Example Trade JSON

```json
{
  "trade_id": "SIM-1234567890-000001",
  "run_tag": "CANARY_TEST_2025Q1",
  "policy_sha256": "abc123...",
  "router_sha256": "def456...",
  "manifest_sha256": "ghi789...",
  "entry_snapshot": {
    "trade_id": "SIM-1234567890-000001",
    "entry_time": "2025-01-01T10:00:00Z",
    "instrument": "XAU_USD",
    "side": "long",
    "entry_price": 2650.0,
    "session": "EU",
    "regime": "MEDIUM",
    "entry_model_version": "FARM_V2B",
    "entry_score": {
      "p_long": 0.75,
      "p_short": 0.10,
      "p_hat": 0.65,
      "margin": 0.65
    },
    "entry_filters_passed": ["spread_ok", "atr_ok", "regime_ok", "session_ok"],
    "entry_filters_blocked": []
  },
  "feature_context": {
    "atr": {
      "atr_bps": 50.0,
      "atr_price": 13.25,
      "atr_percentile": 45.0
    },
    "range": {
      "range_pos": 0.3,
      "distance_to_range": 0.4,
      "range_edge_dist_atr": 0.8
    },
    "spread": {
      "spread_price": 0.15,
      "spread_pct": 12.0
    },
    "candle": {
      "close": 2650.0,
      "high": 2650.5,
      "low": 2649.5
    }
  },
  "router_explainability": {
    "router_version": "V3_RANGE",
    "router_model_hash": "def456...",
    "router_features_used": {
      "atr_pct": 45.0,
      "spread_pct": 12.0,
      "atr_bucket": "MEDIUM",
      "regime": "MEDIUM",
      "session": "EU",
      "range_pos": 0.3,
      "distance_to_range": 0.4,
      "range_edge_dist_atr": 0.8
    },
    "router_raw_decision": "RULE6A",
    "guardrail_applied": false,
    "guardrail_reason": null,
    "guardrail_cutoff": null,
    "range_edge_dist_atr": 0.8,
    "final_exit_profile": "RULE6A"
  },
  "exit_configuration": {
    "exit_profile": "RULE6A",
    "tp_levels": [6.0, 9.0],
    "sl": 4.0,
    "trailing_enabled": true,
    "be_rules": {
      "adaptive_threshold_bps": 4.0,
      "trailing_stop_bps": 2.0
    }
  },
  "exit_events": [
    {
      "timestamp": "2025-01-01T10:15:00Z",
      "event_type": "TRAIL_ACTIVATED",
      "price": 2650.4,
      "pnl_bps": 4.5,
      "bars_held": 3
    },
    {
      "timestamp": "2025-01-01T10:20:00Z",
      "event_type": "EXIT_TRIGGERED",
      "price": 2650.3,
      "pnl_bps": 3.0,
      "bars_held": 4
    }
  ],
  "exit_summary": {
    "exit_time": "2025-01-01T10:20:00Z",
    "exit_price": 2650.3,
    "exit_reason": "TRAIL",
    "realized_pnl_bps": 3.0,
    "max_mfe_bps": 4.5,
    "max_mae_bps": -1.2,
    "intratrade_drawdown_bps": -1.5
  }
}
```

### Usage

Trade journal provides complete traceability:
- **Entry**: Why did we enter? (model score + gates + features snapshot)
- **Router**: What did router choose and why? (features, raw pred, guardrail explainability)
- **Exit**: What triggered exit? (exit events timeline + final summary)

Journal is human-readable (JSON) and machine-readable (CSV index).

---

## GO Practice-Live (PROD_BASELINE) — Controlled Start

### Overview

**GO Practice-Live** is the "GO"-knappen for starting GX1 in OANDA practice mode with PROD_BASELINE policy snapshot. This is a controlled, verified start with full traceability and artifacts.

**What this does:**
- ✅ Starts GX1 in OANDA practice mode (not dry_run)
- ✅ Uses PROD_BASELINE policy snapshot from `prod_snapshot/`
- ✅ SINGLE worker (n_workers=1) for determinism/parity
- ✅ max_open_trades=1 (safety limit)
- ✅ Asia-window drift (last 24 hours warmup, then real-time)
- ✅ Full verification + artifacts (trade journal + reconciliation + baseline proof)
- ✅ Real orders sent to OANDA Practice API

**What this does NOT do:**
- ❌ Does NOT run in live environment (hard-fails if OANDA_ENV != practice)
- ❌ Does NOT allow live trading (hard-fails if I_UNDERSTAND_LIVE_TRADING=YES)
- ❌ Does NOT modify strategy parameters (uses frozen PROD_BASELINE)
- ❌ Does NOT run in parallel (always n_workers=1)

### Running GO Practice-Live

```bash
# Set environment variables
export OANDA_ENV=practice
export OANDA_API_TOKEN=<your_token>
export OANDA_ACCOUNT_ID=<your_account_id>

# IMPORTANT: Do NOT set I_UNDERSTAND_LIVE_TRADING=YES (script will fail)

# Run with default settings
./scripts/go_practice_live_asia.sh

# Run with custom run tag
./scripts/go_practice_live_asia.sh --run-tag MY_GO_TEST_2025

# Run with debug logging
./scripts/go_practice_live_asia.sh --debug
```

### Hard Requirements

**Environment Checks (script fails if not met):**
- `OANDA_ENV=practice` (hard-fail if not set)
- `OANDA_API_TOKEN` must be set (hard-fail if missing)
- `OANDA_ACCOUNT_ID` must be set (hard-fail if missing)
- `I_UNDERSTAND_LIVE_TRADING` must NOT be set (hard-fail if set to YES)

**Configuration:**
- Uses PROD_BASELINE policy: `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml`
- `dry_run=false` (REAL ORDERS)
- `n_workers=1` (deterministic)
- `max_open_trades=1` (safety)
- `logging level: INFO` (or DEBUG with `--debug` flag)

### What the Script Does

1. **Hard Checks:**
   - Verifies `OANDA_ENV=practice`
   - Verifies credentials are set
   - Verifies `I_UNDERSTAND_LIVE_TRADING` is NOT set

2. **Runs Practice-Live:**
   - Uses PROD_BASELINE policy snapshot
   - `dry_run=false` (real orders)
   - `n_workers=1` (deterministic)
   - `max_open_trades=1` (safety)
   - Last 24 hours warmup (historical data), then real-time execution

3. **Automatic Verification:**
   - Runs `prod_baseline_proof.py` (checks hashes + 100% journal coverage)
   - Runs `reconcile_oanda.py` (matches OANDA transactions via clientExtensions)
   - Verifies all required artifacts exist

4. **GO SUMMARY:**
   - Prints key metrics (trades, RULE6A_rate, fill diffs)
   - Lists all artifacts and their paths
   - Provides verification instructions

### Expected Artifacts

After a successful run, the following artifacts should exist:

```
gx1/wf_runs/<RUN_TAG>/
├── run_header.json                    # SHA256 hashes of all artifacts
├── trade_journal/
│   ├── trade_journal_index.csv        # Aggregated trade index
│   └── trades/
│       └── <trade_id>.json            # Per-trade journal (one per trade)
├── reconciliation_report.md           # OANDA transaction reconciliation
├── prod_baseline_proof.md             # Policy/artifact parity proof
├── alerts.json                         # Alerts (if any)
├── run.log                             # Full execution log
├── prod_baseline_proof.log            # Proof generation log
└── reconcile.log                       # Reconciliation log
```

### Stop Procedure (Kill Switch)

**Location:** `{project_root}/KILL_SWITCH_ON` (project root = directory containing `gx1/`)

**How to Activate:**
```bash
# Manual activation (emergency stop)
touch KILL_SWITCH_ON

# Or from project root:
touch $(pwd)/KILL_SWITCH_ON
```

**How to Verify Stop:**
```bash
# Check if flag exists
test -f KILL_SWITCH_ON && echo "KILL SWITCH ACTIVE" || echo "OK"

# Runner will log:
# [GUARD] BLOCKED ORDER (live_mode=...) reason=KILL_SWITCH_ON flag
```

**How to Clear:**
```bash
# Remove kill switch flag
rm KILL_SWITCH_ON

# Restart runner to resume trading
```

**When to Use:**
- Emergency stop (manual)
- System detects critical failures (automatic)
- Maintenance window
- Suspected bug or anomaly

### How to Verify First Trade Was Sent

**1. Check OANDA Practice Trading Platform:**
- Log into OANDA Practice account
- Check "Trades" tab for new trades
- Verify trade details match trade journal

**2. Check Trade Journal Execution Events:**
```bash
# View execution events from first trade
cat gx1/wf_runs/<RUN_TAG>/trade_journal/trades/*.json | python3 -m json.tool | grep -A 10 execution_events
```

**Expected events:**
- `ORDER_SUBMITTED`: Order sent to OANDA
- `ORDER_FILLED`: Order filled (if immediate)
- `TRADE_OPENED_OANDA`: Trade opened in OANDA (with OANDA trade ID)

**3. Check Reconciliation Report:**
```bash
cat gx1/wf_runs/<RUN_TAG>/reconciliation_report.md
```

**Expected:**
- All trades matched with OANDA transactions
- `clientExtensions.id` matches pattern `GX1:{run_tag}:{trade_id}`
- Fill prices match (within tolerance)

**4. Check Trade Journal Index:**
```bash
cat gx1/wf_runs/<RUN_TAG>/trade_journal/trade_journal_index.csv
```

**Expected:**
- Trade entries with entry_time, exit_time, pnl_bps
- Execution events logged for each trade

### Troubleshooting

**Issue: Script fails with "OANDA_ENV must be 'practice'"**
- Set: `export OANDA_ENV=practice`

**Issue: Script fails with "I_UNDERSTAND_LIVE_TRADING=YES is set"**
- Unset: `unset I_UNDERSTAND_LIVE_TRADING`
- This script is for PRACTICE only

**Issue: Missing execution events in trade journal**
- Check that `dry_run=false` was used
- Check that orders were actually sent (check `run.log`)
- Check that trade journal logging didn't fail (check `run.log` for warnings)

**Issue: Reconciliation report shows unmatched trades**
- Check that `clientExtensions.id` matches pattern `GX1:{run_tag}:{trade_id}`
- Check that OANDA transactions are available for the run period
- Check that account ID matches

**Issue: Prod baseline proof fails**
- Check that policy SHA256 matches expected PROD_BASELINE hash
- Check that router model SHA256 matches expected hash
- Check that feature manifest SHA256 matches expected hash

---

## LIVE Force-One-Trade (CANARY only - Plumbing Testing)

### Overview

**LIVE Force-One-Trade** is a CANARY-only mode for plumbing testing in OANDA practice. It guarantees at least 1 trade within 30 minutes if no trades occur naturally. This is **NOT for PROD_BASELINE** - it should never be used in production.

**What this does:**
- ✅ Runs in LIVE mode (real-time candles from OANDA)
- ✅ Uses CANARY policy with `debug_force` enabled
- ✅ Forces entry after 30 minutes if no trades occur naturally
- ✅ Ensures at least 1 trade for plumbing verification
- ✅ Full traceability (trade journal + execution events + reconciliation)

**What this does NOT do:**
- ❌ Does NOT run in PROD_BASELINE mode (uses CANARY)
- ❌ Does NOT bypass risk/guards uncritically
- ❌ Does NOT allow more than 1 trade (hard cap)
- ❌ Does NOT run in live environment (practice only)

### Running LIVE Force-One-Trade

```bash
# Set environment variables
export OANDA_ENV=practice
export OANDA_API_TOKEN=<your_token>
export OANDA_ACCOUNT_ID=<your_account_id>

# IMPORTANT: Do NOT set I_UNDERSTAND_LIVE_TRADING=YES (script will fail)

# Run with default settings (6 hours or until trade opened and closed)
./scripts/run_live_force_one_trade_asia.sh

# Run with custom duration
./scripts/run_live_force_one_trade_asia.sh --hours 3

# Run with debug logging
./scripts/run_live_force_one_trade_asia.sh --debug
```

### Hard Requirements

**Environment Checks (script fails if not met):**
- `OANDA_ENV=practice` (hard-fail if not set)
- `OANDA_API_TOKEN` must be set (hard-fail if missing)
- `OANDA_ACCOUNT_ID` must be set (hard-fail if missing)
- `I_UNDERSTAND_LIVE_TRADING` must NOT be set (hard-fail if set to YES)

**Configuration:**
- Uses CANARY policy: `gx1/configs/policies/active/GX1_V11_OANDA_PRACTICE_LIVE_FORCE_ONE_TRADE.yaml`
- `mode=LIVE` (real-time candles from OANDA)
- `dry_run=false` (REAL ORDERS)
- `max_open_trades=1` (safety)
- `debug_force.enabled=true` (force entry after 30 minutes)
- `debug_force.max_trades=1` (hard cap)

### What the Script Does

1. **Hard Checks:**
   - Verifies `OANDA_ENV=practice`
   - Verifies credentials are set
   - Verifies `I_UNDERSTAND_LIVE_TRADING` is NOT set

2. **Runs LIVE:**
   - Uses CANARY policy with debug_force enabled
   - `mode=LIVE` (real-time candles from OANDA)
   - `dry_run=false` (real orders)
   - `max_open_trades=1` (safety)
   - Runs for specified duration (default: 6 hours) or until trade opened and closed

3. **Force Entry Logic:**
   - Tracks time since start
   - If no trades after 30 minutes, forces entry (in ASIA session only)
   - Respects spread guard and warmup requirements
   - Hard cap: max 1 trade total

4. **Automatic Verification:**
   - Runs `prod_baseline_proof.py` (compares with prod_snapshot for parity)
   - Runs `reconcile_oanda.py` (matches OANDA transactions)
   - Verifies all required artifacts exist

### Expected Artifacts

After a successful run, the following artifacts should exist:

```
gx1/wf_runs/<RUN_TAG>/
├── run_header.json                    # SHA256 hashes of all artifacts
├── trade_journal/
│   ├── trade_journal_index.csv        # Aggregated trade index
│   └── trades/
│       └── <trade_id>.json            # Per-trade journal (one per trade)
├── reconciliation_report.md           # OANDA transaction reconciliation
├── prod_baseline_proof.md             # Policy/artifact parity proof
├── run.log                            # Full execution log
├── prod_baseline_proof.log            # Proof generation log
└── reconcile.log                      # Reconciliation log
```

### Force Entry Details

**When Force Entry Triggers:**
- After 30 minutes with no trades
- In ASIA session only
- If spread is acceptable (respects spread guard)
- If warmup is complete (288+ bars)
- If max_trades not reached (hard cap: 1)

**Force Entry Safety:**
- Only in CANARY mode (`meta.role=CANARY`)
- Only in practice (`OANDA_ENV=practice`)
- Does NOT bypass risk/guards uncritically
- Hard cap: max 1 trade total
- Logs clearly: `[FORCE_ENTRY] TRIGGERED: reason=...`

### How to Verify First Trade

**1. Check OANDA Practice Trading Platform:**
- Log into OANDA Practice account
- Check "Trades" tab for new trades
- Verify trade details match trade journal

**2. Check Trade Journal Execution Events:**
```bash
cat gx1/wf_runs/<RUN_TAG>/trade_journal/trades/*.json | python3 -m json.tool | grep -A 10 execution_events
```

**Expected events:**
- `ORDER_SUBMITTED`: Order sent to OANDA
- `ORDER_FILLED`: Order filled (if immediate)
- `TRADE_OPENED_OANDA`: Trade opened in OANDA (with OANDA trade ID)

**3. Check Reconciliation Report:**
```bash
cat gx1/wf_runs/<RUN_TAG>/reconciliation_report.md
```

**Expected:**
- All trades matched with OANDA transactions
- `clientExtensions.id` matches pattern `GX1:{run_tag}:{trade_id}`
- Fill prices match (within tolerance)

### Troubleshooting

**Issue: Script fails with "OANDA_ENV must be 'practice'"**
- Set: `export OANDA_ENV=practice`

**Issue: Force entry not triggering**
- Check that 30 minutes have passed
- Check that you're in ASIA session
- Check that spread is acceptable
- Check that warmup is complete (288+ bars)
- Check logs for `[FORCE_ENTRY]` messages

**Issue: No trades after force entry**
- Check that `debug_force.enabled=true` in policy
- Check that `meta.role=CANARY` in policy
- Check that `OANDA_ENV=practice`
- Check logs for force entry trigger messages

### Important Notes

**⚠️ Force-one-trade is ONLY for plumbing testing:**
- Should NEVER be used in PROD_BASELINE
- Should NEVER be used in live environment
- Should NEVER be used for performance testing
- Should be removed/deprecated after plumbing is verified

**This mode is temporary and should not live long in the codebase.**

---

## Practice-Live Micro Testing

### Overview

Practice-live micro testing runs real orders against OANDA Practice API (not dry-run) with full traceability and verification. This is a correctness and auditability test, not a performance test.

### Requirements

**Hard Requirements:**
- `OANDA_ENV=practice` (script fails if not set)
- `OANDA_API_TOKEN` and `OANDA_ACCOUNT_ID` must be set
- Uses PROD_BASELINE policy snapshot
- `dry_run=false` (real orders)
- `n_workers=1` (deterministic)
- `max_open_trades=1` (safety limit)
- Short period (default: last 24 hours)

**Safety:**
- Token never logged
- Account ID masked in logs
- `I_UNDERSTAND_LIVE_TRADING` latch only for live-env (not practice)

### Running Practice-Live Micro Test

```bash
# Set environment variables
export OANDA_ENV=practice
export OANDA_API_TOKEN=<your_token>
export OANDA_ACCOUNT_ID=<your_account_id>

# Run with default settings (last 24 hours)
./scripts/run_practice_live_micro.sh

# Run with custom hours
./scripts/run_practice_live_micro.sh --hours 12

# Run with custom run tag
./scripts/run_practice_live_micro.sh --run-tag MY_TEST_2025

# Run with custom policy
./scripts/run_practice_live_micro.sh --policy gx1/configs/policies/prod_snapshot/.../policy.yaml
```

### What the Script Does

1. **Hard Checks:**
   - Verifies `OANDA_ENV=practice`
   - Verifies credentials are set
   - Verifies policy file exists

2. **Runs Practice-Live Replay:**
   - Uses PROD_BASELINE policy
   - `dry_run=false` (real orders)
   - `n_workers=1` (deterministic)
   - `max_open_trades=1` (safety)

3. **Automatic Verification:**
   - Runs `prod_baseline_proof.py` (checks hashes + 100% journal coverage)
   - Runs `reconcile_oanda.py` (matches OANDA transactions via clientExtensions)
   - Verifies all required artifacts exist

4. **Self-Verification:**
   - Checks trade journal index CSV
   - Checks trade journal JSON files
   - Checks run_header.json
   - Checks execution events in trade journal
   - Checks reconciliation report
   - Checks prod baseline proof report

### Expected Artifacts

After a successful run, the following artifacts should exist:

```
gx1/wf_runs/<RUN_TAG>/
├── run_header.json                    # SHA256 hashes of all artifacts
├── trade_journal/
│   ├── trade_journal_index.csv        # Aggregated trade index
│   └── trades/
│       └── <trade_id>.json            # Per-trade journal (one per trade)
├── reconciliation_report.md           # OANDA transaction reconciliation
├── prod_baseline_proof.md             # Policy/artifact parity proof
├── run.log                            # Full execution log
├── prod_baseline_proof.log            # Proof generation log
└── reconcile.log                      # Reconciliation log
```

### Trade Journal Execution Events

For practice-live runs, each trade JSON should contain `execution_events` array with:

**At Entry:**
- `ORDER_SUBMITTED`: Order sent to OANDA
- `ORDER_FILLED`: Order filled (if immediate)
- `TRADE_OPENED_OANDA`: Trade opened in OANDA (with OANDA trade ID)

**At Exit:**
- `ORDER_SUBMITTED`: Close order sent
- `ORDER_FILLED`: Close order filled
- `TRADE_CLOSED_OANDA`: Trade closed in OANDA (with closing transaction ID)

**If Order Rejected:**
- `ORDER_REJECTED`: Order rejected (with reason)

### Inspecting Trade Journal

**View Trade Index:**
```bash
cat gx1/wf_runs/<RUN_TAG>/trade_journal/trade_journal_index.csv
```

**View Specific Trade:**
```bash
# Find trade ID from index
cat gx1/wf_runs/<RUN_TAG>/trade_journal/trade_journal_index.csv | grep <trade_id>

# View full trade journal
cat gx1/wf_runs/<RUN_TAG>/trade_journal/trades/<trade_id>.json | python3 -m json.tool
```

**Check Execution Events:**
```bash
# Extract execution events from trade journal
python3 << EOF
import json
with open("gx1/wf_runs/<RUN_TAG>/trade_journal/trades/<trade_id>.json", "r") as f:
    d = json.load(f)
    events = d.get("execution_events", [])
    for e in events:
        print(f"{e.get('event_type')}: {e.get('oanda_order_id', 'N/A')} @ {e.get('fill_price', 'N/A')}")
EOF
```

**View Reconciliation Report:**
```bash
cat gx1/wf_runs/<RUN_TAG>/reconciliation_report.md
```

**View Prod Baseline Proof:**
```bash
cat gx1/wf_runs/<RUN_TAG>/prod_baseline_proof.md
```

### Verification Checklist

After running practice-live micro test, verify:

- [ ] `run_header.json` exists and has correct SHA256 hashes
- [ ] `trade_journal_index.csv` exists and has trade entries
- [ ] Trade JSON files exist (one per trade)
- [ ] Each trade JSON has `entry_snapshot` (100% coverage)
- [ ] Each trade JSON has `feature_context` (100% coverage)
- [ ] Each trade JSON has `execution_events` (for practice-live runs)
- [ ] Execution events include `ORDER_SUBMITTED` and `ORDER_FILLED`
- [ ] Execution events include `TRADE_OPENED_OANDA` (if trade opened)
- [ ] Execution events include `TRADE_CLOSED_OANDA` (if trade closed)
- [ ] `reconciliation_report.md` exists and matches OANDA transactions
- [ ] `prod_baseline_proof.md` exists and shows 100% coverage
- [ ] No errors in `run.log` related to trade journal
- [ ] No errors in `run.log` related to execution events

### Troubleshooting

**Issue: Script fails with "OANDA_ENV must be 'practice'"**
- Set: `export OANDA_ENV=practice`

**Issue: Missing execution events in trade journal**
- Check that `dry_run=false` was used
- Check that orders were actually sent (check `run.log`)
- Check that trade journal logging didn't fail (check `run.log` for warnings)

**Issue: Reconciliation report shows unmatched trades**
- Check that `clientExtensions.id` matches pattern `GX1:{run_tag}:{trade_id}`
- Check that OANDA transactions are available for the run period
- Check that account ID matches

**Issue: Prod baseline proof fails**
- Check that policy SHA256 matches expected PROD_BASELINE hash
- Check that router model SHA256 matches expected hash
- Check that feature manifest SHA256 matches expected hash

---

## Emergency Stop Procedure

### Kill Switch

**Location:** `{project_root}/KILL_SWITCH_ON` (project root = directory containing `gx1/`)

**How to Activate:**
```bash
# Manual activation (emergency stop)
touch KILL_SWITCH_ON

# Or from project root:
touch $(pwd)/KILL_SWITCH_ON
```

**Automatic Activation:**
- Kill switch is automatically set when 3 consecutive order failures occur
- Location: `gx1/execution/oanda_demo_runner.py:6082-6084`
- Log message: `[HARD STOP] 3 consecutive order failures. Setting KILL_SWITCH_ON.`

**How to Clear:**
```bash
# Remove kill switch flag
rm KILL_SWITCH_ON

# Restart runner to resume trading
```

**Verification:**
- Check if flag exists: `test -f KILL_SWITCH_ON && echo "KILL SWITCH ACTIVE" || echo "OK"`
- Runner logs: `[GUARD] BLOCKED ORDER (live_mode=...) reason=KILL_SWITCH_ON flag`

**When to Use:**
- Emergency stop (manual)
- System detects critical failures (automatic)
- Maintenance window
- Suspected bug or anomaly

---

## Execution Smoke Test (Practice)

### Overview

Execution Smoke Test is a forced execution test that sends one minimal market order to OANDA Practice API, logs full execution trail in Trade Journal, and runs reconciliation. This tests plumbing (credentials → order → fill → close → journal → reconcile) without using entry/exit models or strategy logic.

**What this tests:**
- ✅ OANDA credentials loading and validation
- ✅ Order submission (market order)
- ✅ Order fill handling
- ✅ Trade opening in OANDA
- ✅ Trade closing in OANDA
- ✅ Trade Journal execution event logging
- ✅ Reconciliation with OANDA transactions

**What this does NOT test:**
- ❌ Entry/exit strategy logic
- ❌ Entry/exit models
- ❌ Router decisions
- ❌ Guardrails
- ❌ Gates/thresholds
- ❌ Performance/backtesting

**Important:** This is NOT a strategy trade. It's marked as `test_mode=true` and does not use entry/exit models.

### Requirements

**Hard Requirements:**
- `OANDA_ENV=practice` (script fails if not set)
- `OANDA_API_TOKEN` and `OANDA_ACCOUNT_ID` must be set
- Short duration (default: 180 seconds hold time)
- Minimal order size (default: 1 unit)

**Safety:**
- Token never logged
- Account ID masked in logs
- Only runs against Practice API (never live)
- Test trades are clearly marked with `test_mode=true`

### Running Execution Smoke Test

```bash
# Set environment variables
export OANDA_ENV=practice
export OANDA_API_TOKEN=<your_token>
export OANDA_ACCOUNT_ID=<your_account_id>

# Run with default settings (180 seconds hold, 1 unit)
./scripts/run_oanda_exec_smoke_test.sh

# Run with custom hold time
./scripts/run_oanda_exec_smoke_test.sh --hold-seconds 60

# Run with custom units
./scripts/run_oanda_exec_smoke_test.sh --units 1

# Run with custom instrument
./scripts/run_oanda_exec_smoke_test.sh --instrument XAU_USD

# Run with custom run tag
./scripts/run_oanda_exec_smoke_test.sh --run-tag MY_SMOKE_TEST_2025
```

**Direct Python usage:**
```bash
python gx1/execution/exec_smoke_test.py \
  --instrument XAU_USD \
  --units 1 \
  --hold-seconds 180 \
  --run-tag EXEC_SMOKE_20251216 \
  --out-dir gx1/wf_runs/EXEC_SMOKE_20251216
```

### What the Script Does

1. **Hard Checks:**
   - Verifies `OANDA_ENV=practice`
   - Verifies credentials are set
   - Verifies output directory can be created

2. **Creates Test Trade:**
   - Generates test trade ID: `EXEC-SMOKE-<timestamp>-<uuid>`
   - Creates client extension ID: `GX1:EXEC_SMOKE:<run_tag>:<trade_id>`
   - Logs entry snapshot with `test_mode=true`

3. **Sends Market Order:**
   - Gets current price from OANDA
   - Logs `ORDER_SUBMITTED` event
   - Sends market order via OANDA API
   - Logs `ORDER_FILLED` event (if successful)
   - Logs `TRADE_OPENED_OANDA` event (if trade opened)

4. **Holds Trade:**
   - Waits for specified duration (default: 180 seconds)

5. **Closes Trade:**
   - Logs `ORDER_SUBMITTED` (close) event
   - Closes trade via OANDA API
   - Logs `ORDER_FILLED` (close) event
   - Logs `TRADE_CLOSED_OANDA` event
   - Logs exit summary

6. **Runs Reconciliation:**
   - Automatically runs `reconcile_oanda.py`
   - Matches OANDA transactions via `clientExtensions.id`
   - Generates reconciliation report

### Expected Artifacts

After a successful run, the following artifacts should exist:

```
gx1/wf_runs/<RUN_TAG>/
├── run_header.json                    # Run metadata (timestamp, OANDA_ENV, masked account)
├── trade_journal/
│   ├── trade_journal_index.csv       # Aggregated trade index
│   └── trades/
│       └── <trade_id>.json           # Per-trade journal (test_mode=true)
├── exec_smoke_summary.json           # Test summary (PASS/FAIL, IDs, durations)
├── reconciliation_report.md          # OANDA transaction reconciliation
├── smoke_test.log                    # Full execution log
└── reconcile.log                     # Reconciliation log
```

### Trade Journal Schema (Test Mode)

For execution smoke test trades, the trade journal JSON has the following structure:

**Entry Snapshot:**
```json
{
  "entry_snapshot": {
    "test_mode": true,
    "reason": "EXECUTION_SMOKE_TEST",
    "entry_model_version": null,
    "entry_time": "2025-01-01T00:00:00Z",
    "instrument": "XAU_USD",
    "side": "long",
    "entry_price": 2000.0,
    ...
  }
}
```

**Feature Context:**
```json
{
  "feature_context": {
    "test_mode": true,
    "atr_bps": null,
    "atr_price": null,
    ...
  }
}
```

**Execution Events:**
```json
{
  "execution_events": [
    {
      "event_type": "ORDER_SUBMITTED",
      "client_extensions": {
        "id": "GX1:EXEC_SMOKE:<run_tag>:<trade_id>",
        "tag": "<run_tag>",
        "comment": "EXECUTION_SMOKE_TEST"
      },
      ...
    },
    {
      "event_type": "ORDER_FILLED",
      "oanda_order_id": "12345",
      "fill_price": 2000.0,
      ...
    },
    {
      "event_type": "TRADE_OPENED_OANDA",
      "oanda_trade_id": "67890",
      ...
    },
    {
      "event_type": "ORDER_SUBMITTED",
      "client_extensions": {
        "id": "GX1:EXEC_SMOKE:<run_tag>:<trade_id>:CLOSE",
        ...
      },
      ...
    },
    {
      "event_type": "ORDER_FILLED",
      "oanda_order_id": "12346",
      "fill_price": 2001.0,
      ...
    },
    {
      "event_type": "TRADE_CLOSED_OANDA",
      "oanda_trade_id": "67890",
      "pl": 1.0,
      ...
    }
  ]
}
```

### Inspecting Trade Journal

**View Trade JSON:**
```bash
cat gx1/wf_runs/<RUN_TAG>/trade_journal/trades/*.json | python3 -m json.tool
```

**Check Test Mode Flag:**
```bash
python3 << EOF
import json
with open("gx1/wf_runs/<RUN_TAG>/trade_journal/trades/*.json", "r") as f:
    d = json.load(f)
    print(f"Test mode: {d['entry_snapshot'].get('test_mode', False)}")
    print(f"Reason: {d['entry_snapshot'].get('reason', 'N/A')}")
EOF
```

**View Execution Events:**
```bash
python3 << EOF
import json
with open("gx1/wf_runs/<RUN_TAG>/trade_journal/trades/*.json", "r") as f:
    d = json.load(f)
    events = d.get("execution_events", [])
    for e in events:
        print(f"{e['event_type']}: {e.get('oanda_order_id', 'N/A')} @ {e.get('fill_price', 'N/A')}")
EOF
```

**View Reconciliation Report:**
```bash
cat gx1/wf_runs/<RUN_TAG>/reconciliation_report.md
```

### Verification Checklist

After running execution smoke test, verify:

- [ ] `run_header.json` exists and has correct OANDA_ENV and masked account ID
- [ ] `trade_journal/trades/*.json` exists (one file)
- [ ] Trade JSON has `entry_snapshot.test_mode=true`
- [ ] Trade JSON has `entry_snapshot.reason="EXECUTION_SMOKE_TEST"`
- [ ] Trade JSON has `entry_snapshot.entry_model_version=null`
- [ ] Trade JSON has `feature_context.test_mode=true`
- [ ] Trade JSON has `execution_events` array with at least:
  - `ORDER_SUBMITTED` (open)
  - `ORDER_FILLED` (open)
  - `TRADE_OPENED_OANDA`
  - `ORDER_SUBMITTED` (close)
  - `ORDER_FILLED` (close)
  - `TRADE_CLOSED_OANDA`
- [ ] `client_extensions.id` format is `GX1:EXEC_SMOKE:<run_tag>:<trade_id>`
- [ ] `exec_smoke_summary.json` exists and has `status="PASS"`
- [ ] `reconciliation_report.md` exists and matches OANDA transactions
- [ ] No errors in `smoke_test.log` related to order submission/fill/close
- [ ] No errors in `reconcile.log` related to transaction matching

### Troubleshooting

**Issue: Script fails with "OANDA_ENV must be 'practice'"**
- Set: `export OANDA_ENV=practice`

**Issue: Order submission fails**
- Check that `OANDA_API_TOKEN` is valid
- Check that `OANDA_ACCOUNT_ID` is correct
- Check that account has sufficient margin (even for 1 unit)
- Check `smoke_test.log` for detailed error messages

**Issue: Trade cannot be closed**
- Check that trade was actually opened (look for `TRADE_OPENED_OANDA` event)
- Check that `oanda_trade_id` is present in trade journal
- Check that trade is still open in OANDA account

**Issue: Reconciliation fails to match transactions**
- Check that `clientExtensions.id` matches pattern `GX1:EXEC_SMOKE:<run_tag>:<trade_id>`
- Check that OANDA transactions are available for the run period
- Check that account ID matches

**Issue: Test status is FAIL**
- Check `exec_smoke_summary.json` for details
- Check `smoke_test.log` for errors
- Verify that all execution events were logged correctly

---

## References

- Determinism Gate: `gx1/analysis/determinism_gate.py`
- Run Diff: `gx1/analysis/diff_runs.py`
- Pipeline Overview: `docs/pipeline/PIPELINE_OVERVIEW.md`
- Invariants: `docs/pipeline/INVARIANTS_AND_CHECKS.md`
- Trade Journal: `gx1/monitoring/trade_journal.py`

