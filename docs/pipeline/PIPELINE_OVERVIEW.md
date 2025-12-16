# GX1 Pipeline Overview - Single Source of Truth

**Last Updated:** 2025-12-15  
**Status:** PROD_BASELINE (V3_RANGE + Guardrail)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Runtime Pipeline (E2E Flow)](#runtime-pipeline-e2e-flow)
3. [Training Pipeline](#training-pipeline)
4. [Feature Truth (ATR + Range)](#feature-truth-atr--range)
5. [Production Safety & Invariants](#production-safety--invariants)
6. [What Remains?](#what-remains)

---

## Executive Summary

GX1 is a ML-driven trading system for XAUUSD (Gold) with:
- **Entry:** FARM_V2B transformer-based policy (ENTRY_V9)
- **Exit:** Hybrid RULE5/RULE6A routing via ML decision tree (V3_RANGE)
- **Guardrail:** RULE6A only allowed when `range_edge_dist_atr < 1.0`
- **Runtime:** Offline replay mode (parallel chunks) + live mode (OANDA API)

**Key Components:**
- Entry Manager: Evaluates entry signals, creates trades, computes range features
- Exit Manager: Executes exit logic (RULE5/RULE6A)
- Exit Router: ML decision tree routes trades to optimal exit policy
- Guardrail: Post-processing filter prevents RULE6A in non-edge regimes

---

## Runtime Pipeline (E2E Flow)

### Entry Point

**File:** `scripts/run_replay.sh`  
**Function:** Bash script → `scripts/active/replay_entry_exit_parallel.py`

**Flow:**
1. Loads policy YAML from CLI argument
2. Filters price data (M5 bars) to date range
3. Splits data into N chunks (default: 7 workers)
4. Runs parallel replays via `joblib.Parallel`
5. Merges results and generates summary

**Key Functions:**
- `run_replay_chunk()` - Runs single chunk replay
- `merge_chunk_results()` - Combines chunk trade logs

**Input:** Policy YAML, date range, price data (parquet/CSV)  
**Output:** `gx1/wf_runs/<tag>/trade_log_*.csv`, `results.json`

---

### Configuration Loading

**File:** `gx1/execution/oanda_demo_runner.py`  
**Function:** `load_yaml_config(path: Path) -> Dict[str, Any]` (line 416)

**Flow:**
1. Reads YAML file using `yaml.safe_load()`
2. Merges entry_config and exit_config into main policy dict
3. Extracts hybrid_exit_router config (including `v3_range_edge_cutoff`)
4. Creates `ExitModeSelector` with router config

**Key Config Sections:**
- `entry_config`: Entry policy config path
- `exit_config`: Exit policy config path
- `exit_hybrid`: Hybrid routing enabled/disabled
- `hybrid_exit_router`: Router version + guardrail cutoff

**Side Effects:** None (pure function)

---

### OANDA Credentials Loading

**File:** `gx1/execution/oanda_credentials.py`  
**Function:** `load_oanda_credentials(prod_baseline: bool) -> OandaCredentials`  
**Called from:** `gx1/execution/oanda_demo_runner.py` (line ~2230)

**Flow:**
1. Reads environment variables:
   - `OANDA_ENV`: "practice" or "live" (default: "practice")
   - `OANDA_API_TOKEN`: API token (preferred) or `OANDA_API_KEY` (legacy)
   - `OANDA_ACCOUNT_ID`: Account ID (required)
2. Validates environment (must be "practice" or "live")
3. **PROD_BASELINE mode**: Raises `ValueError` if credentials missing/invalid → trading disabled
4. **Dev/replay mode**: Logs warning if missing, continues (will fail later if used)
5. Determines API URLs based on environment:
   - Practice: `https://api-fxpractice.oanda.com` / `https://stream-fxpractice.oanda.com`
   - Live: `https://api-fxtrade.oanda.com` / `https://stream-fxtrade.oanda.com`
6. Returns `OandaCredentials` object with masked account ID logged

**Security:**
- Credentials NEVER in code or YAML
- Account ID masked in logs (e.g., "101-***-001")
- API token NEVER logged
- Fail-closed in PROD_BASELINE

**Input:** Environment variables  
**Output:** `OandaCredentials` object  
**Side Effects:** Credentials loaded, masked account ID logged

---

### GX1DemoRunner Initialization

**File:** `gx1/execution/oanda_demo_runner.py`  
**Class:** `GX1DemoRunner` (line 1262)  
**Method:** `__init__()` (line 1263)

**Flow:**
1. Loads policy YAML via `load_yaml_config()`
2. Initializes `EntryManager(self, exit_config_name)` (line 2385)
3. Initializes `ExitManager(self)` (line 2386)
4. Sets up `ExitModeSelector` with guardrail config (line 1346)
5. Loads entry models via `load_entry_models()` (line 2184, function at line 444)
6. Sets up logging, telemetry, broker client

**Key Attributes:**
- `self.entry_manager`: EntryManager instance
- `self.exit_manager`: ExitManager instance
- `self.exit_mode_selector`: ExitModeSelector (if hybrid enabled)
- `self.exit_hybrid_enabled`: Boolean flag
- `self.v3_range_edge_cutoff`: Guardrail cutoff (default: 1.0)

**Side Effects:** Model loading, log file creation

**Entry Model Loading:**
- **File:** `gx1/execution/oanda_demo_runner.py`
- **Function:** `load_entry_models(metadata_path, model_paths)` (line 444)
- **Called from:** `GX1DemoRunner.__init__()` (line 2184)
- **Models Loaded:**
  - EU: `gx1/models/GX1_entry_EU.joblib` (default, line 78)
  - US: `gx1/models/GX1_entry_US.joblib` (default, line 79)
  - OVERLAP: `gx1/models/GX1_entry_OVERLAP.joblib` (default, line 80)
- **Metadata:** `gx1/models/GX1_entry_session_metadata.json` (line 76)
- **Logging:**
  - Feature columns hash (MD5, first 16 chars) for drift detection (line 469)
  - Model classes per session (line 507-512)
  - `n_features_in_` verification (line 498-503)
  - Model bundle version from metadata (line 484)
- **Returns:** `EntryModelBundle` with models dict, feature_names, metadata, feature_cols_hash, model_bundle_version

---

### Data Flow: Candles → Bar Loop

**File:** `gx1/execution/oanda_demo_runner.py`  
**Method:** `_run_replay_impl()` (line ~7652)

**Flow:**
1. Loads price data (parquet/CSV) → DataFrame with time index
2. Iterates through bars chronologically
3. For each bar:
   - Calls `evaluate_entry(candles)` → `entry_manager.evaluate_entry()`
   - Calls `evaluate_and_close_trades(candles)` → `exit_manager.evaluate_and_close_trades()`
   - Updates state (candles history, feature cache)

**Input:** DataFrame with columns: `time` (index), `open`, `high`, `low`, `close`, `volume`, `bid_*`, `ask_*`  
**Output:** Trade objects created, exits executed, logs written

**Side Effects:** Trade creation, exit execution, CSV logging

---

### Feature Building

**File:** `gx1/execution/live_features.py`  
**Function:** `build_live_entry_features(candles: pd.DataFrame) -> EntryFeatureBundle`  
**Called from:** `entry_manager.evaluate_entry()` (line 197)

**Flow:**
1. Builds runtime features via `build_v9_runtime_features()` (from `gx1.features.runtime_v9`)
2. Computes ATR (via feature pipeline)
3. Computes spread (bid-ask difference)
4. Returns `EntryFeatureBundle` with:
   - `features`: DataFrame with all entry features
   - `atr_bps`: ATR in basis points (from feature pipeline)
   - Other runtime features

**Input:** DataFrame with OHLC candles (last N bars)  
**Output:** `EntryFeatureBundle` object  
**Side Effects:** Feature cache updates (if enabled)

**ATR Calculation:**
- Computed in feature pipeline (`gx1/features/runtime_v9.py`)
- Returns `atr_bps` (basis points)
- Converted to price units: `atr_value = (atr_bps / 10000.0) * entry_price` (line 1822)

---

### Entry Decision (FARM_V2B)

**File:** `gx1/execution/entry_manager.py`  
**Method:** `evaluate_entry(candles: pd.DataFrame) -> Optional[LiveTrade]` (line 196)

**Flow:**
1. Builds entry features via `build_live_entry_features()`
2. Gets entry prediction from FARM_V2B policy (`entry_v9_policy_farm_v2b.py`)
3. Checks entry filters (regime, session, risk score)
4. If entry signal → creates `LiveTrade` object
5. **Computes range features BEFORE router selection** (line 1813-1860):
   - `range_pos`, `distance_to_range` via `_compute_range_features()` (line 69)
   - `range_edge_dist_atr` via `_compute_range_edge_dist_atr()` (line 147)
6. Sets `trade.extra["range_pos"]`, `trade.extra["distance_to_range"]`, `trade.extra["range_edge_dist_atr"]`
7. Calls `exit_mode_selector.choose_exit_profile()` with range features
8. Sets `trade.extra["exit_profile"]` to selected profile

**Input:** DataFrame with candles  
**Output:** `LiveTrade` object or `None`  
**Side Effects:** Trade created, logged to CSV, `trade.extra` populated

**Key Functions Called:**
- `_compute_range_features()` (line 69): Computes `range_pos`, `distance_to_range`
- `_compute_range_edge_dist_atr()` (line 147): Computes ATR-normalized edge distance
- `exit_mode_selector.choose_exit_profile()`: Router selection

---

### Exit Routing (V3_RANGE + Guardrail)

**File:** `gx1/policy/exit_hybrid_controller.py`  
**Class:** `ExitModeSelector`  
**Method:** `choose_exit_profile()` (line 47)

**Flow:**
1. Creates `ExitRouterContext` with:
   - `atr_pct`, `spread_pct` (from entry manager)
   - `range_pos`, `distance_to_range`, `range_edge_dist_atr` (from entry manager)
   - `atr_bucket`, `regime`, `session`
2. Calls `hybrid_exit_router_v3(ctx)` → `gx1/core/hybrid_exit_router.py` (line 177)
3. Router loads decision tree model (`exit_router_v3_tree.pkl`)
4. Router predicts policy (`RULE5` or `RULE6A`)
5. **Guardrail check** (line 101-110):
   - If `policy == "RULE6A"` and `range_edge_dist_atr >= cutoff` (1.0):
     - Override to `RULE5`
     - Log debug message
6. Returns profile name (`FARM_EXIT_V2_RULES_A` or `FARM_EXIT_V2_RULES_ADAPTIVE_v1`)

**File:** `gx1/core/hybrid_exit_router.py`  
**Function:** `hybrid_exit_router_v3(ctx: ExitRouterContext) -> ExitPolicyName` (line 177)

**Flow:**
1. Lazy-loads model from `gx1/analysis/exit_router_models_v3/exit_router_v3_tree.pkl` (line 214)
2. Prepares features DataFrame matching training format
3. Calls `model.predict(X)` → returns policy string
4. Falls back to hardcoded tree logic if model load fails

**Input:** `ExitRouterContext` with all features  
**Output:** `"RULE5"` or `"RULE6A"`  
**Side Effects:** Model loaded (cached after first load), debug logging

**Guardrail Location:** `gx1/policy/exit_hybrid_controller.py`, line 101-110 (post-processing after router)

---

### Exit Execution

**File:** `gx1/execution/exit_manager.py`  
**Class:** `ExitManager`  
**Method:** `evaluate_and_close_trades(candles: pd.DataFrame)` (line ~27)

**Flow:**
1. Iterates through open trades
2. Gets exit profile from `trade.extra["exit_profile"]`
3. Loads appropriate exit policy factory:
   - `FARM_EXIT_V2_RULES_A` → `get_exit_policy_farm_v2_rules()` (RULE5)
   - `FARM_EXIT_V2_RULES_ADAPTIVE_v1` → `get_exit_policy_farm_v2_rules_adaptive()` (RULE6A)
4. Evaluates exit conditions (TP, SL, trailing stop, timeout)
5. If exit triggered → closes trade, computes PnL
6. Logs trade to CSV

**Input:** DataFrame with candles, open trades  
**Output:** Trades closed, PnL computed  
**Side Effects:** Trade closure, CSV logging, metrics updated

---

### Logging & Metrics

**File:** `gx1/execution/oanda_demo_runner.py`  
**Method:** `_log_trade()` (various locations)

**Trade Log CSV Columns:**
- `trade_id`, `entry_time`, `exit_time`
- `entry_price`, `exit_price`, `pnl_bps`
- `exit_profile` (RULE5 or RULE6A)
- `bars_held`
- `extra` (JSON string with range features, router context, etc.)

**Key Fields in `extra`:**
- `range_pos`: float [0.0, 1.0]
- `distance_to_range`: float [0.0, 1.0]
- `range_edge_dist_atr`: float [0.0, 10.0]
- `exit_profile`: string
- `exit_hybrid`: dict with router context

**Output Files:**
- `gx1/wf_runs/<tag>/trade_log_*.csv` (per chunk)
- `gx1/wf_runs/<tag>/trade_log_*_merged.csv` (merged)
- `gx1/wf_runs/<tag>/results.json` (summary metrics)

---

### Trade Journal (Production-Ready)

**File:** `gx1/monitoring/trade_journal.py`  
**Class:** `TradeJournal`

**Initialization:**
- **Location:** `gx1/execution/oanda_demo_runner.py:2137` (`_init_trade_journal()`)
- **Called from:** `GX1DemoRunner.__init__()` after `_generate_run_header()`
- **Directory:** `gx1/wf_runs/<tag>/trade_journal/`

**Per-Trade JSON Files:**
- **Path:** `gx1/wf_runs/<tag>/trade_journal/trades/<trade_id>.json`
- **Written:** On each logging call (entry snapshot, feature context, router decision, exit events, exit summary)
- **Format:** Structured JSON with complete trade lifecycle

**Aggregated Index CSV:**
- **Path:** `gx1/wf_runs/<tag>/trade_journal/trade_journal_index.csv`
- **Written:** On trade close (via `log_exit_summary()`)
- **Format:** CSV with one row per trade (for filtering/analysis)

**Integration Points:**

1. **Entry Snapshot** (`gx1/execution/entry_manager.py:1987-2055`):
   - Logged after trade creation
   - Includes: entry_time, instrument, side, entry_price, session, regime, entry_model_version, entry_score, entry_filters_passed/blocked

2. **Feature Context** (`gx1/execution/entry_manager.py:1987-2055`):
   - Logged after trade creation (immutable snapshot)
   - Includes: ATR (bps, price, percentile), Range (pos, distance, edge_dist_atr), Spread (price, pct), Candle (close, high, low)

3. **Router Decision** (`gx1/policy/exit_hybrid_controller.py:123-157`):
   - Logged after router selection
   - Includes: router_version, router_model_hash, router_features_used, router_raw_decision, guardrail_applied, guardrail_reason, final_exit_profile

4. **Exit Configuration** (`gx1/execution/oanda_demo_runner.py:2728-2758`):
   - Logged when exit policy initializes
   - Includes: exit_profile, tp_levels, sl, trailing_enabled, be_rules

5. **Exit Events** (`gx1/execution/exit_manager.py:167-186`):
   - Logged when exit events occur (TP1_HIT, BE_MOVED, TRAIL_ACTIVATED, EXIT_TRIGGERED)
   - Append-only timeline

6. **Exit Summary** (`gx1/execution/exit_manager.py:804-875`):
   - Logged on trade close via `_log_trade_close_with_metrics()` helper
   - Includes: exit_time, exit_price, exit_reason, realized_pnl_bps, max_mfe_bps, max_mae_bps, intratrade_drawdown_bps
   - **Intratrade Metrics Calculation** (`gx1/execution/exit_manager.py:877-1000`):
     - Calculated from `_price_trace` collected per bar for open trades
     - **Price Trace Collection** (`gx1/execution/exit_manager.py:740-802`):
       - Called at start of `evaluate_and_close_trades()` for all open trades
       - Stores `high`, `low`, `close` per bar in `trade.extra["_price_trace"]`
       - Limited to last 5000 points
       - Uses mid-price if bid/ask available, otherwise direct OHLC
     - **MFE/MAE/DD Calculation**:
       - MFE: max favorable excursion (uses bar `high` for long, `low` for short)
       - MAE: max adverse excursion (uses bar `low` for long, `high` for short)
       - Intratrade DD: largest peak-to-trough drawdown on unrealized PnL curve (uses bar `close`)
     - Trace removed from `trade.extra` after calculation (prevents bloating trade logs)
   - Also updates index CSV

**Production Safety:**
- All logging wrapped in try/except
- Failures log warnings but never block trading
- Enabled by default in PROD_BASELINE
- Can be disabled via `enabled=False` parameter (for testing)

**See:** `docs/RUNBOOK.md` for detailed usage guide and example trade JSON.

---

## Training Pipeline

### Entry Model Training

**Status:** ✅ IMPLEMENTERT i dette repoet

**File:** `gx1/models/entry_v9/entry_v9_train.py`  
**CLI:** `python -m gx1.models.entry_v9.entry_v9_train --config gx1/configs/entry_v9/entry_v9_train_full.yaml`

**Training Pipeline:**
- Multi-task learning with regime-conditioning
- Uses EntryV9Transformer architecture (`gx1/models/entry_v9/entry_v9_transformer.py`)
- Dataset preparation: `gx1/models/entry_v9/entry_v9_dataset.py`
- Training with PyTorch, sklearn preprocessing

**Model Artifacts:**
- Path: `gx1/models/entry_v9/nextgen_2020_2025_clean/`
- Format: Transformer models (joblib/pickle)
- Features: Runtime V9 feature set (from `gx1/features/runtime_v9.py`)

**Note:** Entry models are pre-trained and frozen for production. Training pipeline exists but models are not retrained in production.

---

### Router Model Training (V3_RANGE)

**File:** `gx1/analysis/build_exit_policy_training_dataset_v3.py`  
**Function:** `main()` (line ~368)

**Dataset Building Flow:**
1. Loads trade logs from multiple FULLYEAR runs (via `--run TAG PATH` CLI args)
2. Extracts trade data via `_extract_trade_data()` (line 92):
   - Reads `range_pos`, `distance_to_range`, `range_edge_dist_atr` from `trade.extra`
   - Extracts `atr_pct`, `spread_pct` from `exit_hybrid` or direct columns
   - Computes `policy_best` (oracle: best of RULE5 vs RULE6A based on PnL)
3. Cleans features in `build_dataset_v3()` (line ~343):
   - `range_pos`: `pd.to_numeric().fillna(0.5).clip(0.0, 1.0)`
   - `distance_to_range`: `pd.to_numeric().fillna(0.5).clip(0.0, 1.0)`
   - `range_edge_dist_atr`: `pd.to_numeric().fillna(0.0).clip(0.0, 10.0)`
4. Saves dataset: `gx1/analysis/exit_policy_training_dataset_v3.csv/parquet`
5. Saves metadata: `gx1/analysis/exit_policy_training_dataset_v3_metadata.json`

**Input:** Multiple trade log CSVs from FULLYEAR replays  
**Output:** `exit_policy_training_dataset_v3.csv/parquet`, metadata JSON  
**Side Effects:** Dataset files created, metadata logged

**File:** `gx1/analysis/router_training_v3.py`  
**Function:** `main()` (line ~200)

**Training Flow:**
1. Loads dataset via `load_dataset_v3()` (line 26)
2. Prepares features via `prepare_features_v3()` (line 65):
   - Numeric: `atr_pct`, `spread_pct`, `distance_to_range`, `range_edge_dist_atr`, `micro_volatility`, `volume_stability`
   - Categorical: `atr_bucket`, `farm_regime`, `session`
3. Builds preprocessor (OneHotEncoder for categoricals, passthrough for numeric)
4. Trains `DecisionTreeClassifier` via sklearn
5. Saves model: `gx1/analysis/exit_router_models_v3/exit_router_v3_tree.pkl`
6. Exports tree rules: `exit_router_v3_tree_rules.txt`
7. Saves metrics: `exit_router_v3_metrics.json`

**Input:** `exit_policy_training_dataset_v3.csv/parquet`  
**Output:** `exit_router_v3_tree.pkl`, `exit_router_v3_tree_rules.txt`, `exit_router_v3_metrics.json`  
**Side Effects:** Model files created, metrics logged

**Model Loading (Runtime):**
- File: `gx1/core/hybrid_exit_router.py`, line 214
- Path: `gx1/analysis/exit_router_models_v3/exit_router_v3_tree.pkl`
- Lazy-loaded and cached after first load

---

## Feature Truth (ATR + Range)

### ATR Calculation

**Source:** Feature pipeline (`gx1/features/runtime_v9.py`)  
**Unit:** Basis points (bps)  
**Runtime Access:** `entry_bundle.atr_bps` (from `build_live_entry_features()`)

**Conversion to Price Units:**
- File: `gx1/execution/entry_manager.py`, line 1821-1822
- Formula: `atr_value = (current_atr_bps / 10000.0) * entry_price`
- Used for: `range_edge_dist_atr` normalization

**ATR Percentile:**
- File: `gx1/execution/entry_manager.py`, line 212
- Function: `_percentile_from_history()` (line 51)
- History: `self.cluster_guard_history` (deque of ATR bps values)
- Returns: Percentile rank [0.0, 100.0] of current ATR vs history

---

### Range Features

**File:** `gx1/execution/entry_manager.py`  
**Method:** `_compute_range_features()` (line 69)

**Window:** 96 bars (default, configurable)

**Bar Selection:**
- If `len(candles) >= window + 1`: Uses `candles.iloc[-(window+1):-1]` (excludes last incomplete bar)
- Else: Uses `candles.tail(window)` (all available)

**Price Source Priority:**
1. Direct OHLC: `high`, `low`, `close` columns
2. Bid/Ask mid: `(bid_high + ask_high) / 2`, `(bid_low + ask_low) / 2`, `(bid_close + ask_close) / 2`

**range_pos Calculation:**
- `range_hi = max(high_vals)` (line 122)
- `range_lo = min(low_vals)` (line 123)
- `price_ref = close_vals[-1]` (last closed bar's close, line 128)
- `range_pos = (price_ref - range_lo) / (range_hi - range_lo)` (line 132)
- Clamped to `[0.0, 1.0]` (line 133)

**distance_to_range Calculation:**
- `dist_edge = min(range_pos, 1.0 - range_pos)` (line 136)
- `distance_to_range = dist_edge * 2.0` (scaled to 0..1, line 137)
- Clamped to `[0.0, 1.0]` (line 138)

**range_edge_dist_atr Calculation:**
- File: `gx1/execution/entry_manager.py`  
- Method: `_compute_range_edge_dist_atr()` (line 147)

**Flow:**
1. Uses same window and bar selection as `_compute_range_features()`
2. Computes `range_hi`, `range_lo`, `price_ref` (same as above)
3. Computes distance to nearest edge:
   - `dist_to_low = max(0.0, price_ref - range_lo)`
   - `dist_to_high = max(0.0, range_hi - price_ref)`
   - `dist_edge_price = min(dist_to_low, dist_to_high)`
4. Normalizes by ATR:
   - `atr_value = (current_atr_bps / 10000.0) * entry_price` (line 1822)
   - `range_edge_dist_atr = dist_edge_price / max(eps, atr_value)` (line 185)
5. Clamped to `[0.0, 10.0]` (line 188)

**Fallback:**
- If `atr_value` missing/invalid: returns `0.0` (line 175)
- If insufficient candles: returns `0.0` (line 194)

**Verification:**
- Debug logging: `[RANGE_FEAT]` (line 1910)
- All features set in `trade.extra` before router selection (line 1816-1860)

---

## Production Safety & Invariants

### Kill Switch

**File:** `gx1/execution/oanda_demo_runner.py`  
**Method:** `_run_once_impl()` (line 6169)  
**Check:** Lines 6170-6176 (also checked in `_execute_entry_impl()` at lines 5918-5925)

**Flow:**
1. Checks for `KILL_SWITCH_ON` file in project root (`{project_root}/KILL_SWITCH_ON`)
2. If exists → blocks all orders, logs warning, returns early
3. No entry/exit evaluation when flag is present

**File Path:** `{project_root}/KILL_SWITCH_ON` (project root = `gx1/execution/../..`)  
**Trigger:** 
- **Automatic:** Set by runtime when `max_consecutive_failures` (default: 3) order failures occur (line 6084)
- **Manual:** Can be created manually by ops team for emergency stop
- **Location:** `gx1/execution/oanda_demo_runner.py:6082-6084` (in `_execute_entry_impl()` exception handler)

**How to Clear:**
- Manual removal: `rm {project_root}/KILL_SWITCH_ON`
- After removal, restart runner to resume trading

**Side Effects:** All trading blocked, warning logged

---

### Policy Lock

**File:** `gx1/execution/oanda_demo_runner.py`  
**Method:** `_check_policy_lock()` (line 3743)  
**Called from:** `_run_once_impl()` (line 6084)

**Flow:**
1. Computes MD5 hash of policy file at startup (line 1297): `self.policy_hash = hashlib.md5(policy_content.encode("utf-8")).hexdigest()[:16]`
2. On each cycle (`_run_once_impl()`), calls `_check_policy_lock()`:
   - Reads current policy file content
   - Computes MD5 hash (first 16 chars)
   - Compares with `self.policy_hash`
   - If hash changed → logs error, returns `False` → blocks trading
3. If check fails (exception) → returns `True` (conservative: allow trading)

**Implementation:**
```python
def _check_policy_lock(self) -> bool:
    current_policy_content = self.policy_path.read_text()
    current_policy_hash = hashlib.md5(current_policy_content.encode("utf-8")).hexdigest()[:16]
    if current_policy_hash != self.policy_hash:
        log.error("[POLICY LOCK] Policy file changed on disk: hash=%s (expected %s). Trading disabled.", ...)
        return False
    return True
```

**Side Effects:** Trading blocked if policy file modified, error logged

**Test:** See `tests/test_policy_lock.py` (to be created)

---

### Preflight Cache

**Status:** ✅ IMPLEMENTERT

**File:** `gx1/execution/oanda_demo_runner.py`  
**Methods:** 
- Feature meta/scalers cache: Lines 2043-2067
- Backfill cache: Lines 2524-3741
- Feature manifest hash: Lines 3457-3493

**What is Cached:**

1. **Feature Meta/Scalers Cache** (Entry V9):
   - `self._entry_v9_seq_features`, `self._entry_v9_snap_features` (feature metadata)
   - `self._entry_v9_seq_scaler`, `self._entry_v9_snap_scaler` (scalers)
   - Pre-loaded at startup (line 2048-2052) to avoid re-reading files on every prediction
   - Cached in memory for lifetime of runner instance

2. **Backfill Cache** (`self.backfill_cache`):
   - DataFrame with historical candles (M5 bars)
   - Used for feature rehydration and warmup
   - Persisted to disk: `gx1/wf_runs/<tag>/logs/cache/*.parquet`
   - Rotated after 14 days (line 6435-6451)

3. **Feature Manifest Hash**:
   - Computed from entry model bundle: `self.entry_model_bundle.feature_cols_hash` (line 3457)
   - Used for drift detection: if hash changes, overlap_bars increased for safe rehydration (line 3472-3484)
   - Stored in backfill state for comparison

**When Triggered:**
- **Feature meta/scalers**: At startup when Entry V9 model loaded (line 2043-2067)
- **Backfill cache**: During warmup/backfill phase (line 2610-3741)
- **Feature manifest hash**: Computed during backfill initialization (line 3457)

**Side Effects:**
- Reduced I/O (feature meta/scalers loaded once)
- Faster feature rehydration (backfill cache)
- Drift detection (hash comparison triggers overlap escalation)

---

### Invariants (Must Always Be True)

1. **Range Features Available Before Router:**
   - File: `gx1/execution/entry_manager.py`, line 1813-1860
   - `range_pos`, `distance_to_range`, `range_edge_dist_atr` computed BEFORE `exit_mode_selector.choose_exit_profile()`
   - Verified: Features set in `trade.extra` before router call (line 1816-1860)

2. **Guardrail Applied After Router:**
   - File: `gx1/policy/exit_hybrid_controller.py`, line 101-110
   - Router selects policy first, then guardrail overrides if needed
   - Verified: Guardrail check happens after `hybrid_exit_router_v3()` call (line 99)

3. **ATR Conversion Correct:**
   - File: `gx1/execution/entry_manager.py`, line 1821-1822
   - Formula: `atr_value = (atr_bps / 10000.0) * entry_price`
   - Verified: Debug logging shows `atr_price`, `range_span`, `ratio` (UKJENT – må verifiseres logging location)

4. **Trade.extra Always Contains Range Features:**
   - File: `gx1/execution/entry_manager.py`, line 1816-1860
   - All trades have `range_pos`, `distance_to_range`, `range_edge_dist_atr` in `extra`
   - Fallback: `range_pos=0.5`, `distance_to_range=0.5`, `range_edge_dist_atr=0.0` if computation fails

5. **Model Loading Robust:**
   - File: `gx1/core/hybrid_exit_router.py`, line 214-223
   - Falls back to hardcoded tree logic if model load fails
   - Model cached after first load (line 213)

---

## What Remains?

### ✅ Production Ready

1. **V3_RANGE Router:**
   - Trained on FULLYEAR 2025 (1592 trades)
   - Model frozen: `exit_router_v3_tree.pkl`
   - Verified: Uses range features in splits
   - Evidence: `exit_router_v3_tree_rules.txt` shows `range_edge_dist_atr` splits

2. **Guardrail:**
   - Implemented: `gx1/policy/exit_hybrid_controller.py`, line 101-110
   - Verified: Blocks 42/57 RULE6A trades in 2025 (range_edge_dist_atr >= 1.0)
   - Evidence: All blocked trades have identical PnL (delta=0.0 exact)

3. **Range Features:**
   - Implemented: `gx1/execution/entry_manager.py`, line 69-194
   - Verified: 100% coverage in FULLYEAR replay (162/162 trades)
   - Evidence: `range_pos`, `range_edge_dist_atr` stats show variation

4. **Dataset Builder:**
   - Implemented: `gx1/analysis/build_exit_policy_training_dataset_v3.py`
   - Verified: Extracts range features from `trade.extra`
   - Evidence: Metadata shows `range_edge_dist_atr_missing_before_fill: 0`

5. **Intratrade Risk Analysis:**
   - Implemented: `gx1/analysis/compare_exit_routers_fullyear.py`, `analyze_intratrade_risk()`
   - Verified: MFE/MAE/DD computed correctly
   - Evidence: All 42 overridden trades have identical intratrade risk (delta=0.0)

---

### 🟡 In Production (Monitor)

1. **2024 Validation:**
   - Status: Configs created, replays pending (data not available)
   - Monitor: EV/trade, RULE6A allocation, delta_pnl=0 invariant
   - Metric: Compare baseline vs guardrail on 2024 data

2. **Model Performance:**
   - Status: Trained on 2025 data, used in production
   - Monitor: Router accuracy, feature importance shifts
   - Metric: Track router predictions vs actual best policy

3. **Guardrail Effectiveness:**
   - Status: Active in production
   - Monitor: RULE6A allocation rate, blocked trade PnL
   - Metric: Ensure blocked trades continue to have delta_pnl=0

---

### ❗ Remaining Work

1. **Entry Model Training Pipeline** (P2)
   - Location: `gx1/models/entry_v9/entry_v9_train.py`
   - Status: ✅ IMPLEMENTERT i dette repoet
   - Description: Training pipeline for ENTRY_V9 NEXTGEN transformer model
   - CLI: `python -m gx1.models.entry_v9.entry_v9_train --config gx1/configs/entry_v9/entry_v9_train_full.yaml`
   - Features: Multi-task learning with regime-conditioning
   - Note: Training scripts exist in repo, but models are pre-trained and frozen for production

2. **Kill Switch Documentation** (P1)
   - Location: `gx1/execution/oanda_demo_runner.py`, line 6078
   - Verify: ops_watch.py cron job creates flag correctly
   - How: Check ops_watch.py or deployment scripts

5. **Feature Mapping Verification** (P1)
   - Location: `gx1/features/runtime_v9.py`
   - Verify: Feature names match between training and runtime
   - How: Compare feature manifest with model metadata

6. **ATR Calculation Verification** (P1)
   - Location: `gx1/features/runtime_v9.py`
   - Verify: ATR period, calculation method matches expectations
   - How: Read ATR calculation code, verify period (typically 14 bars)

7. **2024 Data Availability** (P0)
   - Location: Data pipeline
   - Verify: 2024 M5 data available for validation
   - How: Check M5_DATA env var or data directory

8. **Parallel Replay Robustness** (P2)
   - Location: `scripts/active/replay_entry_exit_parallel.py`
   - Verify: Chunk merging handles edge cases (missing columns, date formats)
   - How: Test with various chunk configurations

9. **Model Versioning** (P2)
   - Location: `gx1/analysis/exit_router_models_v3/`
   - Verify: Model versioning strategy documented
   - How: Check if model files include version metadata

10. **Error Handling** (P1)
    - Location: Various (entry_manager, exit_manager, router)
    - Verify: All fallbacks work correctly (model load fails, feature computation fails)
    - How: Review exception handling in key functions

---

**Document Status:** ✅ Complete (with UKJENT markers where verification needed)  
**Next Review:** After 2024 validation completes

