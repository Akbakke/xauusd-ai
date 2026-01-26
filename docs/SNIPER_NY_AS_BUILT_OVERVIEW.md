# SNIPER/NY AS-BUILT OVERVIEW
**System Architecture and Flow Documentation (Post-V10 + Telemetry Refactor)**

**Last Updated:** 2026-01-05  
**Status:** Production (SNIPER/NY, ENTRY_V10 hybrid)  
**Documentation Scope:** Entry pipeline, exit pipeline, telemetry, gates/thresholds, orchestration

---

## EXECUTIVE SUMMARY

**Current Pipeline:** SNIPER/NY entry policy using ENTRY_V10 hybrid model (XGBoost + Transformer), with two-stage gating (Stage-0 precheck, Stage-1 candidate) and SNIPER_EXIT_RULES_A_P4_1 exit policy.

**Key Components:**
- **Entry Model:** ENTRY_V10 hybrid (XGBoost snapshot + Transformer sequence)
- **Feature Pipeline:** `build_v9_runtime_features` → `build_basic_v1` → NumPy rolling operations
- **Gating:** Stage-0 (session/regime/volatility) → Stage-1 (threshold/risk_guard/max_trades)
- **Telemetry:** `entry_telemetry` container with Stage-0/Stage-1 counters and veto tracking
- **Exit:** SNIPER_EXIT_RULES_A_P4_1 (bar-based, RULE_A variants)

**Top 3 Bottlenecks (from 1-week replay):**
1. **Feature building** (~32s for 1-week, ~31ms per bar) - `build_basic_v1` dominates
2. **Stage-0 gating** (602/1016 bars blocked: 59.3% rejection rate) - session/regime filters
3. **Stage-1 threshold filtering** (126/414 candidates blocked: 30.4% rejection rate) - `min_prob_long=0.18` threshold

---

## DEL A — SYSTEM MAP (Kodekart)

### A1) Entry Pipeline (SNIPER/NY, V10)

#### Data Source (Replay/Live)

**Location:** `gx1/execution/oanda_demo_runner.py` → `_run_replay_impl()` (lines 7627-8809)

**Replay Mode:**
- Input: CSV/Parquet file with OHLCV candles (M5 bars)
- Loading: `pd.read_csv()` or `pd.read_parquet()` (line ~7700)
- Time column: `time` (or `ts` as fallback, line ~7710)
- Index: DatetimeIndex (UTC)
- Column normalization checkpoint: `_assert_no_case_collisions()` called immediately after loading (line ~7720)

**Live Mode:**
- Input: OANDA API streaming candles (via `TickWatcher` thread)
- Column normalization: Defensive deduplication in `_predict_entry_v10_hybrid()` (lines 5940-5977)

#### Column Normalization and Collision Policy

**Location:** `gx1/features/runtime_v9.py` → `build_v9_live_base_features()` (lines 211-383)

**Policy:**
- **Offline/Replay:** Hard fail on case-insensitive collisions (e.g., `close` + `CLOSE`)
  - Detection: `collections.defaultdict(list)` maps normalized → original columns (lines 237-248)
  - Error: `V9RuntimeFeatureError` with collision details (lines 249-254)
- **Live Mode:** Defensive deduplication with warning (tolerate vendor glitches, but escalate)
  - Location: `gx1/execution/oanda_demo_runner.py` → `_predict_entry_v10_hybrid()` (lines 5948-5977)
  - Action: Drop duplicates (keep first), log warning, set `data_integrity_degraded` flag (future)

**Safety Guard:** After rename, check `df.columns.duplicated()` (line 261) - should never trigger if collision detection works.

#### Feature Building Path

**Entry Point:** `gx1/execution/entry_manager.py` → `evaluate_entry()` → `build_live_entry_features()` (line 547)

**Flow:**
1. `build_live_entry_features()` → `gx1/execution/live_features.py`
   - Builds `EntryFeatureBundle` with raw candles + features DataFrame
   
2. **V10 Inference Path:** `_predict_entry_v10_hybrid()` → `build_v9_runtime_features()` (line 5993)
   - Location: `gx1/execution/oanda_demo_runner.py` (lines 5888-6150)
   - Input: Raw candles DataFrame (with normalized columns)
   - Output: `df_v9_feats` (seq + snap features), `seq_feat_names`, `snap_feat_names`

3. **build_v9_runtime_features()** → `gx1/features/runtime_v9.py` (lines 385-550)
   - Calls `build_v9_live_base_features()` (line 410)
   - Subsets to model-expected features (lines 415-450)
   - Applies scalers (seq_scaler, snap_scaler) (lines 451-480)

4. **build_v9_live_base_features()** → `gx1/features/runtime_v9.py` (lines 211-383)
   - Column normalization (lowercase, collision detection) (lines 235-267)
   - `build_basic_v1()` → `gx1/features/basic_v1.py` (line 298)
     - NumPy rolling operations (if `GX1_FEATURE_USE_NP_ROLLING=1`): `rolling_np.py`, `rolling_numba.py`
     - Output: `_v1_*` features (returns, volatility, moments, quantiles, interactions, cost_proxy)
   - `build_sequence_features()` → `gx1/seq/sequence_features.py` (line 306)
     - Uses `_v1_atr_regime_id` from `build_basic_v1` for `atr_regime_id`
     - Output: `atr50`, `atr_z`, `body_pct`, `wick_asym`, `session_id`, etc.
   - Leakage removal (lines 324-340): `LEAKAGE_SUBSTRINGS` (mfe, mae, pnl, label, etc.)

5. **Incremental Quantiles (Replay/Live):**
   - State: `RollingR1Quantiles48State` (incremental O(1) per-bar)
   - Location: `gx1/features/rolling_state_numba.py`
   - Used for: `_v1_r1_q10_48`, `_v1_r1_q90_48`
   - Batch path disabled in replay (env guard: `GX1_REPLAY_INCREMENTAL_FEATURES=1`)

#### V10 Inference Path

**Location:** `gx1/execution/oanda_demo_runner.py` → `_predict_entry_v10_hybrid()` (lines 5888-6150)

**Flow:**
1. **V9 Feature Building** (line 5993):
   - `build_v9_runtime_features(df_raw, feature_meta_path, seq_scaler_path, snap_scaler_path)`
   - Output: `df_v9_feats` (seq + snap features, scaled)

2. **XGBoost Prediction** (lines 6000-6050):
   - Model: `self.entry_v10_bundle.xgb_model`
   - Input: Snapshot features (from `df_v9_feats`)
   - Output: `xgb_prob_long` (float)

3. **Transformer Prediction** (lines 6051-6120):
   - Model: `self.entry_v10_bundle.transformer_model`
   - Input: Sequence window (from `df_v9_feats`, last `seq_len` bars)
   - Session/regime embeddings: `session_id` (0=EU, 1=OVERLAP, 2=US), `vol_regime_id` (0-3), `trend_regime_id` (0-2)
   - Output: `transformer_prob_long` (float)

4. **Hybrid Combination** (lines 6121-6140):
   - Weighted average: `prob_long = 0.7 * xgb_prob_long + 0.3 * transformer_prob_long` (configurable)
   - Temperature scaling: Applied if configured
   - Output: `EntryPrediction` object with `prob_long`, `prob_short`, `session`, `margin`, `p_hat`

**Error Handling:**
- V10 bundle not loaded → `log.error()` → return `None`
- Feature building fails → `log.error()` → return `None`
- XGB/Transformer inference fails → `log.error()` → return `None`
- Telemetry: `n_v10_calls++`, `n_v10_pred_none_or_nan++` on failures

#### EntryManager Decision Flow

**Location:** `gx1/execution/entry_manager.py` → `evaluate_entry()` (lines 537-4329)

**Stage-0 (Precheck) - Before Prediction:**

**Telemetry Counter:** `n_cycles++` (line 584)

**Gate Sequence:**
1. **Warmup Check** (lines ~900-950):
   - Condition: `self.warmup_floor` not reached
   - Veto: `veto_pre_warmup++`
   - Code: `_should_consider_entry_impl()` → warmup check

2. **Stage-0 Opportunity Filter** (lines ~1100-1200):
   - Function: `_should_consider_entry_impl()` → `gx1/execution/oanda_demo_runner.py` (lines 638-780)
   - Inputs: `trend_regime`, `vol_regime`, `session`, `risk_score`, `stage0_config`
   - Logic:
     - UNKNOWN regimes → `stage0_unknown_field` → `veto_pre_regime++`
     - Session gate (SNIPER): Allow EU/OVERLAP/US (config-driven) → `stage0_session_block` → `veto_pre_session++`
     - Volatility gate (SNIPER): Allow LOW/MEDIUM/HIGH (block EXTREME) → `stage0_vol_block` → `veto_pre_atr++`
     - Trend-vol combination gate → `stage0_trend_vol_block` → `veto_pre_regime++`
   - Output: `(should_consider: bool, reason_code: str)`
   - Telemetry: If `should_consider=True` → `n_precheck_pass++` (line ~1150)

3. **Model Availability Check** (lines ~1300-1340):
   - V10 mode: Check `self.entry_v10_enabled` and `self.entry_v10_bundle`
   - V9 mode: Check `self.entry_v9_enabled` and `self.entry_v9_model`
   - Veto: `veto_pre_model_missing++` if model unavailable

4. **NaN Features Check** (lines ~1350-1400):
   - Validate feature DataFrame has finite values
   - Veto: `veto_pre_nan_features++` if NaN/Inf detected

**If Stage-0 passes:** `n_precheck_pass++`, continue to prediction

**Stage-1 (Candidate) - After Prediction, Before Trade:**

**Prediction:** V10 hybrid (`_predict_entry_v10_hybrid()`) or V9 (`_predict_entry_v9()`)

**Telemetry Counter:** If prediction OK (finite `prob_long`): `n_predictions++`, `n_candidates++`, append `p_long` to `p_long_values` (lines ~1400-1450)

**Gate Sequence (after prediction):**

1. **Threshold Gate** (lines ~2400-2500):
   - Policy: `entry_v9_policy_sniper` → `apply_entry_v9_policy_sniper()`
   - Location: `gx1/policy/entry_v9_policy_sniper.py` (lines 29-280)
   - Input: `p_long`, `min_prob_long` (config), `min_prob_short` (config), `allow_short` (config)
   - Logic: `p_long >= min_prob_long` (long) OR `p_short >= min_prob_short` (short, if allowed)
   - Output: `side` (LONG/SHORT/NONE)
   - Veto: If `side is None` → `veto_cand_threshold++` (line 2494), store `threshold_used = min_prob_long`
   - Return: `None` if threshold fails (no trade)

2. **Risk Guard (SNIPER)** (lines ~3050-3100):
   - Class: `SniperRiskGuardV1` → `gx1/policy/sniper_risk_guard.py` (lines 33-250)
   - Inputs: `spread_bps`, `atr_bps`, `vol_regime`, `session`
   - Checks:
     - Global spread threshold: `spread_bps >= block_spread_bps` → block
     - Global ATR threshold: `atr_bps >= block_atr_bps` → block
     - Vol regime: `vol_regime in block_vol_regimes` → block
     - Session-specific thresholds (US, OVERLAP)
   - Veto: `veto_cand_risk_guard++` if blocked (line 3092)
   - Return: `None` if blocked

3. **Max Trades Gate** (lines ~3150-3200):
   - Check: `len(self.open_trades) >= self.max_concurrent_positions`
   - Veto: `veto_cand_max_trades++` if blocked (line ~3180)
   - Return: `None` if blocked

4. **Big Brain V1 Gate** (lines ~3200-3250):
   - If enabled: Check `big_brain_v1_entry_gater.should_allow_side()`
   - Veto: `veto_cand_big_brain++` if blocked (line ~3230)
   - Return: `None` if blocked

5. **Intraday Drawdown Guard** (lines ~3250-3300):
   - Check: `unrealized_portfolio_bps <= -intraday_drawdown_bps_limit`
   - Veto: `veto_cand_risk_guard++` (line ~3280)
   - Return: `None` if blocked

6. **Cluster Guard** (lines ~3300-3350):
   - Check: Same-direction trades >= 3 AND `current_atr_bps > atr_median`
   - Veto: `veto_cand_risk_guard++` (line ~3330)
   - Return: `None` if blocked

**If Stage-1 passes:** `n_candidate_pass++` (line 3532)

**Trade Creation:**

**Location:** `gx1/execution/entry_manager.py` → `evaluate_entry()` (lines 3539-3562)

**Single Source of Truth:** `n_trades_created++` (line 3556) - incremented immediately after `LiveTrade` instantiation

**Trade Object:** `LiveTrade(trade_id, entry_time, side, units, entry_price, entry_bid, entry_ask, atr_bps, vol_bucket, entry_prob_long, entry_prob_short, dry_run)`

**Trade Journal Fields:**
- `entry_model_version`: Set to `"ENTRY_V10"` if V10 used (or `"ENTRY_V9"` if V9)
- `entry_score.p_long`: Stored in trade JSON (from `prediction.prob_long`)
- Location: `gx1/execution/oanda_demo_runner.py` → `_log_entry_only_event_impl()` (lines 6329-6437)

### A2) Exit Pipeline

**Location:** `gx1/execution/exit_manager.py` → `evaluate_and_close_trades()` (lines 44-660)

**SNIPER Exit Policy:** SNIPER_EXIT_RULES_A_P4_1

**Configuration:**
- Config path: `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/exits/SNIPER_EXIT_RULES_A_P4_1.yaml`
- Loaded in: `gx1/execution/oanda_demo_runner.py` → `__init__()` (lines 1446-1507)

**Exit Rules (EXIT_FARM_V2_RULES):**
- RULE_A_PROFIT: Take profit at configured bps threshold
- RULE_A_TRAILING: Trailing stop after profit target
- RULE_B: MAE-based stop (if enabled)
- RULE_C: Timeout-based exit (if enabled)

**Evaluation:**
- Per-bar evaluation (every M5 bar)
- Input: Current bid/ask, trade entry prices, bars in trade, unrealized PnL
- Output: Close decision (TP/SL/timeout) or continue

**Journaling:**
- Exit events logged to trade journal JSON
- Fields: `exit_time`, `exit_price`, `pnl_bps`, `exit_reason`, `bars_in_trade`
- Location: `gx1/execution/oanda_demo_runner.py` → `request_close()` (lines 4680-5345)

### A3) Risk / Sizing / Guardrails

**Stage-0 Guards (Before Prediction):**

1. **Session Gate** (SNIPER):
   - Location: `gx1/execution/oanda_demo_runner.py` → `_should_consider_entry_impl()` (lines 677-720)
   - Config: `entry_v9_policy_sniper.stage0.allowed_sessions` (default: EU, OVERLAP, US)
   - Veto: `veto_pre_session++`
   - Blocked: ASIA session (by default)

2. **Regime Gate** (SNIPER):
   - Location: `gx1/execution/oanda_demo_runner.py` → `_should_consider_entry_impl()` (lines 720-750)
   - Config: `entry_v9_policy_sniper.stage0.allowed_vol_regimes` (default: LOW, MEDIUM, HIGH)
   - Veto: `veto_pre_regime++` (via `stage0_trend_vol_block`)
   - Blocked: EXTREME volatility regime

3. **ATR Gate** (Stage-0):
   - Location: `gx1/execution/oanda_demo_runner.py` → `_should_consider_entry_impl()` (lines 750-770)
   - Veto: `veto_pre_atr++` (via `stage0_vol_block`)
   - Logic: Part of volatility regime check

4. **Warmup Gate**:
   - Location: `gx1/execution/entry_manager.py` → `evaluate_entry()` (lines ~900-950)
   - Veto: `veto_pre_warmup++`
   - Logic: Skip first N bars (warmup period)

5. **Kill-switch** (Coverage/Parity):
   - Location: `gx1/execution/entry_manager.py` → `evaluate_entry()` (lines 2800-2900)
   - Veto: `veto_pre_killswitch++`
   - Checks: Parity p99 > tolerance, Coverage avvik >50%
   - Mode: Live-only (replay: metrics-only, no blocking)

**Stage-1 Guards (After Prediction):**

1. **Threshold Gate**:
   - Location: `gx1/policy/entry_v9_policy_sniper.py` → `apply_entry_v9_policy_sniper()` (lines 207-227)
   - Config: `entry_v9_policy_sniper.min_prob_long` (default: 0.18 for SNIPER)
   - Veto: `veto_cand_threshold++`
   - Logic: `p_long >= min_prob_long` (long) OR `p_short >= min_prob_short` (short)

2. **SNIPER Risk Guard**:
   - Location: `gx1/policy/sniper_risk_guard.py` → `SniperRiskGuardV1.should_block()` (lines 67-200)
   - Config: `SNIPER_RISK_GUARD_V1.yaml`
   - Inputs: `spread_bps`, `atr_bps`, `vol_regime`, `session`
   - Veto: `veto_cand_risk_guard++`
   - Checks: Global spread/ATR thresholds, session-specific thresholds, vol regime blocks

3. **Max Trades Guard**:
   - Location: `gx1/execution/entry_manager.py` → `evaluate_entry()` (lines ~3150-3200)
   - Config: `max_concurrent_positions` (default: 3)
   - Veto: `veto_cand_max_trades++`
   - Logic: `len(open_trades) >= max_concurrent_positions`

4. **Intraday Drawdown Guard**:
   - Location: `gx1/execution/entry_manager.py` → `evaluate_entry()` (lines ~3250-3300)
   - Config: `intraday_drawdown_bps_limit` (default: -500 bps)
   - Veto: `veto_cand_risk_guard++`
   - Logic: `unrealized_portfolio_bps <= -intraday_drawdown_bps_limit`

5. **Cluster Guard**:
   - Location: `gx1/execution/entry_manager.py` → `evaluate_entry()` (lines ~3300-3350)
   - Config: Cluster threshold (same-direction >= 3), ATR percentile check
   - Veto: `veto_cand_risk_guard++`
   - Logic: Same-direction trades >= 3 AND `current_atr_bps > atr_median`

6. **Big Brain V1 Gate**:
   - Location: `gx1/execution/entry_manager.py` → `evaluate_entry()` (lines ~3200-3250)
   - Config: `big_brain_v1_entry_gater` (if enabled)
   - Veto: `veto_cand_big_brain++`
   - Logic: `big_brain_v1_entry_gater.should_allow_side()` returns False

### A4) Telemetry System (entry_telemetry)

**Container:** `gx1/execution/entry_manager.py` → `EntryManager.__init__()` (lines 40-51)

**Core Counters:**
- `n_cycles`: Total bar cycles evaluated (incremented at start of `evaluate_entry()`, line 584)
- `n_precheck_pass`: Bars that passed Stage-0 (incremented when `should_consider=True`, line ~1150)
- `n_predictions`: Predictions actually produced (V10/V9 OK, finite `prob_long`) (incremented when prediction OK, line ~1400)
- `n_candidates`: Alias to `n_predictions` (same increment, line ~1400)
- `n_candidate_pass`: Candidates that passed Stage-1 (incremented before trade creation, line 3532)
- `n_trades_created`: Trades actually created (SINGLE SOURCE OF TRUTH) (incremented after `LiveTrade` instantiation, line 3556)

**Veto Counters:**

**Stage-0 (veto_pre_*):**
- `veto_pre_warmup`: Warmup period blocks (line ~950)
- `veto_pre_session`: Session gate blocks (line 1151)
- `veto_pre_regime`: Regime gate blocks (line 1155)
- `veto_pre_atr`: ATR/volatility blocks (line 1153)
- `veto_pre_spread`: Spread blocks (not currently used, reserved)
- `veto_pre_killswitch`: Kill-switch blocks (line ~2880)
- `veto_pre_model_missing`: Model unavailable (lines 1303, 1330)
- `veto_pre_nan_features`: NaN/Inf features (line ~1380)

**Stage-1 (veto_cand_*):**
- `veto_cand_threshold`: Threshold gate blocks (line 2494)
- `veto_cand_risk_guard`: Risk guard blocks (lines 3092, ~3280, ~3330)
- `veto_cand_max_trades`: Max trades blocks (line ~3180)
- `veto_cand_big_brain`: Big Brain gate blocks (line ~3230)

**p_long Statistics:**
- Storage: `p_long_values` list (appended for all candidates, line ~1400)
- Export: Histogram-based (200 bins, 0.0-1.0 range) to avoid large arrays for FULLYEAR
- Location: `scripts/run_mini_replay_perf.py` → `write_perf_summary()` (lines 745-777)
- Stats: `mean`, `p50`, `p90`, `min`, `max` computed from histogram

**Merge Logic:**
- Location: `scripts/merge_perf_summaries.py` → `merge_summaries()` (lines 78-428)
- Counters: SUM (`n_cycles`, `n_precheck_pass`, `n_candidates`, `n_candidate_pass`, `n_trades_created`)
- Veto: SUM per reason (`veto_pre_*`, `veto_cand_*`)
- p_long: Merge histogram bins (sum counts), recompute quantiles from merged histogram (lines 150-200)
- Duration: `duration_sec = max(chunk duration_sec)` (parallel wall-clock)
- Feat_time: `feat_time_sec = sum(chunk feat_time_sec)`

**Invariants (TELEMETRY_INV_1-7):**

**Location:** `scripts/assert_perf_invariants.py` → `assert_chunk_invariants()`, `assert_merged_invariants()` (lines 50-200)

- **TELEMETRY_INV_1:** `0 <= n_precheck_pass <= n_cycles`
- **TELEMETRY_INV_2:** `0 <= n_candidates <= n_precheck_pass`
- **TELEMETRY_INV_3:** `0 <= n_candidate_pass <= n_candidates`
- **TELEMETRY_INV_4:** `0 <= n_trades_created <= n_candidate_pass`
- **TELEMETRY_INV_5:** `n_trades_created == trades_total` (if available)
- **TELEMETRY_INV_6:** `sum(veto_pre_*) <= (n_cycles - n_precheck_pass)`
- **TELEMETRY_INV_7:** `sum(veto_cand_*) <= (n_candidates - n_candidate_pass)`

**Verification:**
- Chunk mode: `python3 scripts/assert_perf_invariants.py chunk <json_path> <chunk_dir>`
- Merged mode: `python3 scripts/assert_perf_invariants.py merged <json_path> <chunk_dir1> <chunk_dir2> ...`

### A5) Scripts / Orchestration

#### run_mini_replay_perf.py

**Location:** `scripts/run_mini_replay_perf.py`

**Purpose:** Run single chunk of replay with performance timing

**Key Features:**
- Writes `WINDOW.json` at chunk creation (before replay starts) (lines ~450-470)
- Writes `worker_started.txt` as first operation in `main()` (line ~300)
- Writes `env_snapshot.json` at start (lines ~310-330)
- Enables `faulthandler` early (before imports) (lines ~280-290)
- Writes `REPLAY_PERF_SUMMARY.json/.md` in `finally` block (always, even on crash) (lines 688-940)
- Exports telemetry counters (histogram-based p_long stats) (lines 745-820)
- Exports `fault.log` if process crashes (via `faulthandler`)

**Output Files (per chunk):**
- `REPLAY_PERF_SUMMARY.json`: Chunk performance summary
- `REPLAY_PERF_SUMMARY.md`: Human-readable summary
- `WINDOW.json`: Window start/end (source of truth)
- `worker_started.txt`: Marker file (early crash detection)
- `env_snapshot.json`: Environment variables (thread limits, etc.)
- `fault.log`: Stack trace on crash (if process dies)

#### merge_perf_summaries.py

**Location:** `scripts/merge_perf_summaries.py`

**Purpose:** Merge chunk summaries into global summary

**Fail-fast:**
- Checks all chunks have `REPLAY_PERF_SUMMARY.json` (lines 41-75)
- Checks all chunks have `status="complete"` (lines 57-73)
- Raises `SystemExit` if any chunk missing or incomplete

**Merge Rules:**
- Counters: SUM (lines 98-100)
- Veto: SUM per reason (lines 120-140)
- p_long histogram: Sum bin counts, recompute quantiles (lines 150-200)
- Duration: `max(chunk duration_sec)` (line 105)
- Feat_time: `sum(chunk feat_time_sec)` (line 110)
- Threshold: `min/max/unique` aggregation (lines 200-220)

**Output:** `OUTDIR/REPLAY_PERF_SUMMARY.json` and `.md`

#### verify_fullyear_perf.sh

**Location:** `scripts/verify_fullyear_perf.sh`

**Purpose:** Verify merged summary and print Entry Telemetry section

**Output:**
- Scaling Snapshot (bars_total, duration, bars/sec, feat_time, feat_sec/bar)
- Entry Telemetry section:
  - `n_cycles`, `n_precheck_pass` + `precheck_pass_rate`
  - `n_candidates` + `candidate_rate`
  - `n_candidate_pass` + `pass_rate`
  - `n_trades_created` + `trade_rate`
  - Top 5 `veto_pre` (with %)
  - Top 5 `veto_cand` (with %)
  - `p_long` stats (mean, p50, p90)
  - `threshold` summary (min/max/unique)
  - Trade session distribution

#### run_replay_fullyear_2025_parallel.sh

**Location:** `scripts/run_replay_fullyear_2025_parallel.sh`

**Purpose:** Orchestrate FULLYEAR 2025 parallel replay (7 workers)

**Key Features:**
- Spawns 7 worker processes (chunks 0-6)
- Exports thread limits to workers: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, `NUMEXPR_MAX_THREADS=1` (lines ~50-60)
- Sets `PYTHONFAULTHANDLER=1` in worker processes (line ~70)
- Calls `merge_perf_summaries.py` after all chunks complete (line ~200)
- Fail-fast: Exit non-zero if merge fails

#### assert_perf_invariants.py

**Location:** `scripts/assert_perf_invariants.py`

**Purpose:** Assert telemetry invariants (chunk or merged)

**Usage:**
- Chunk: `python3 scripts/assert_perf_invariants.py chunk <json_path> <chunk_dir>`
- Merged: `python3 scripts/assert_perf_invariants.py merged <json_path> <chunk_dir1> <chunk_dir2> ...`

**Checks:**
- Chunk invariants (A-Q): window_start/end, status, counters, veto sums
- Merged invariants (TELEMETRY_INV_1-7): Counter relationships, veto bounds
- Exit code: 0 if all pass, non-zero if any fail

### A6) Config "Source of Truth"

**Baseline SNIPER Config:**
- Path: `gx1/configs/policies/active/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY.yaml`
- Entry config: `sniper_snapshot/2025_SNIPER_V1/ENTRY_V9_SNIPER_LONDON_NY_P4_1.yaml`
- Exit config: `sniper_snapshot/2025_SNIPER_V1/exits/SNIPER_EXIT_RULES_A_P4_1.yaml`

**Threshold Configuration:**
- Location: `entry_v9_policy_sniper.min_prob_long` (default: 0.18 for SNIPER)
- Storage: `self.threshold_used` in `EntryManager` (set when threshold gate is evaluated, line ~2495)
- Export: `threshold_used` field in perf summary JSON
- Merge: `threshold_used_min/max/unique` in merged summary

**Session Gating Config:**
- Location: `entry_v9_policy_sniper.stage0.allowed_sessions` (default: EU, OVERLAP, US)
- Location: `entry_v9_policy_sniper.stage0.allowed_vol_regimes` (default: LOW, MEDIUM, HIGH)

**Regime Config:**
- Big Brain V1: Enabled if `big_brain_v1.model_path` configured
- FARM_V2B: Disabled for SNIPER (legacy mode)

**entry_models.v10 Bundle Paths:**
- Config: `entry_models.v10.model_path` (hybrid bundle path)
- Config: `entry_models.v10.feature_meta_path`
- Config: `entry_models.v10.seq_scaler_path`
- Config: `entry_models.v10.snap_scaler_path`
- Config: `entry_models.xgb.enabled` (required: true for V10 hybrid)

**require_v9_for_entry / Legacy Toggles:**
- `require_v9_for_entry`: Should be `false` for V10-only SNIPER
- `entry_models.v9.enabled`: Should be `false` for V10-only SNIPER
- `entry_models.v10.enabled`: Should be `true` for V10 SNIPER

---

## DEL B — GATE/THRESHOLD INVENTORY (Truth Table)

| Gate Name | Stage | Where (file + function + ~lines) | Inputs | Output | Default Config | Veto Counter | Observability |
|-----------|-------|----------------------------------|--------|--------|----------------|--------------|---------------|
| **session_gate** | PRE (Stage-0) | `oanda_demo_runner.py` → `_should_consider_entry_impl()` (677-720) | `session` (EU/US/OVERLAP/ASIA), `stage0_config.allowed_sessions` | `should_consider=False` if session not in allowed | `allowed_sessions: [EU, OVERLAP, US]` | `veto_pre_session++` | `verify_fullyear_perf.sh`: Top 5 `veto_pre`, `precheck_pass_rate` |
| **regime_gate** | PRE (Stage-0) | `oanda_demo_runner.py` → `_should_consider_entry_impl()` (720-750) | `trend_regime`, `vol_regime`, `stage0_config.allowed_vol_regimes` | `should_consider=False` if EXTREME vol or UNKNOWN | `allowed_vol_regimes: [LOW, MEDIUM, HIGH]` | `veto_pre_regime++` (via `stage0_trend_vol_block`) | `verify_fullyear_perf.sh`: Top 5 `veto_pre` |
| **atr_gate** | PRE (Stage-0) | `oanda_demo_runner.py` → `_should_consider_entry_impl()` (750-770) | `atr_bps`, `vol_regime` | `should_consider=False` if EXTREME vol | Part of regime gate | `veto_pre_atr++` (via `stage0_vol_block`) | `verify_fullyear_perf.sh`: Top 5 `veto_pre` |
| **warmup_gate** | PRE (Stage-0) | `entry_manager.py` → `evaluate_entry()` (900-950) | `warmup_floor`, `current_ts` | Block if `current_ts < warmup_floor` | Warmup period (first N bars) | `veto_pre_warmup++` | `verify_fullyear_perf.sh`: Top 5 `veto_pre` |
| **killswitch_gate** | PRE (Stage-0) | `entry_manager.py` → `evaluate_entry()` (2800-2900) | `telemetry_tracker.coverage`, `parity_p99`, `target_coverage` | Block if parity/coverage thresholds exceeded | Parity p99 > tolerance, Coverage avvik >50% | `veto_pre_killswitch++` | `verify_fullyear_perf.sh`: Top 5 `veto_pre` (live-only) |
| **model_missing_gate** | PRE (Stage-0) | `entry_manager.py` → `evaluate_entry()` (1300-1340) | `self.entry_v10_enabled`, `self.entry_v10_bundle` (or V9 equivalents) | Block if model not loaded | N/A (hard fail if required) | `veto_pre_model_missing++` | `verify_fullyear_perf.sh`: Top 5 `veto_pre` |
| **nan_features_gate** | PRE (Stage-0) | `entry_manager.py` → `evaluate_entry()` (1350-1400) | Feature DataFrame (NaN/Inf check) | Block if NaN/Inf detected | N/A | `veto_pre_nan_features++` | `verify_fullyear_perf.sh`: Top 5 `veto_pre` |
| **threshold_gate** | CAND (Stage-1) | `entry_v9_policy_sniper.py` → `apply_entry_v9_policy_sniper()` (207-227) | `p_long`, `min_prob_long`, `min_prob_short`, `allow_short` | `side=None` if `p_long < min_prob_long` AND (`p_short < min_prob_short` OR `allow_short=False`) | `min_prob_long: 0.18` (SNIPER), `min_prob_short: 0.72`, `allow_short: false` | `veto_cand_threshold++` | `verify_fullyear_perf.sh`: Top 5 `veto_cand`, `threshold_used` summary, `candidate_rate` |
| **risk_guard_gate** | CAND (Stage-1) | `sniper_risk_guard.py` → `SniperRiskGuardV1.should_block()` (67-200) | `spread_bps`, `atr_bps`, `vol_regime`, `session` | Block if spread/ATR thresholds exceeded or vol regime blocked | `block_spread_bps: 3000`, `block_atr_bps: 10.0`, `block_vol_regimes: [EXTREME]` | `veto_cand_risk_guard++` | `verify_fullyear_perf.sh`: Top 5 `veto_cand` |
| **max_trades_gate** | CAND (Stage-1) | `entry_manager.py` → `evaluate_entry()` (3150-3200) | `len(self.open_trades)`, `self.max_concurrent_positions` | Block if `len(open_trades) >= max_concurrent_positions` | `max_concurrent_positions: 3` | `veto_cand_max_trades++` | `verify_fullyear_perf.sh`: Top 5 `veto_cand` |
| **intraday_dd_gate** | CAND (Stage-1) | `entry_manager.py` → `evaluate_entry()` (3250-3300) | `unrealized_portfolio_bps`, `intraday_drawdown_bps_limit` | Block if `unrealized_portfolio_bps <= -intraday_drawdown_bps_limit` | `intraday_drawdown_bps_limit: -500` | `veto_cand_risk_guard++` | `verify_fullyear_perf.sh`: Top 5 `veto_cand` |
| **cluster_gate** | CAND (Stage-1) | `entry_manager.py` → `evaluate_entry()` (3300-3350) | Same-direction trade count, `current_atr_bps`, `atr_median` | Block if same-direction >= 3 AND `current_atr_bps > atr_median` | Cluster threshold: 3, ATR percentile check | `veto_cand_risk_guard++` | `verify_fullyear_perf.sh`: Top 5 `veto_cand` |
| **big_brain_gate** | CAND (Stage-1) | `entry_manager.py` → `evaluate_entry()` (3200-3250) | `big_brain_v1_entry_gater.should_allow_side()` | Block if Big Brain gate returns False | Config-dependent (if enabled) | `veto_cand_big_brain++` | `verify_fullyear_perf.sh`: Top 5 `veto_cand` |

---

## DEL C — AS-BUILT BEHAVIOR SNAPSHOT (1-Week Run)

**Source:** `gx1/wf_runs/replay_1w_perf_20260105_174310/REPLAY_PERF_SUMMARY.json`

**Metrics:**
- `n_cycles`: 1,016 (total bars evaluated)
- `n_precheck_pass`: 414 (40.7% pass rate)
- `n_candidates`: 414 (100% of precheck_pass, 40.7% of cycles)
- `n_candidate_pass`: 288 (69.6% of candidates)
- `n_trades_created`: 288 (100% of candidate_pass, matches `trades_total`)

### Why n_precheck_pass = 414 of 1016?

**Rejection Rate:** 59.3% (602 bars blocked)

**Top Veto Reasons (Stage-0):**
1. `veto_pre_regime`: 302 (29.7% of cycles)
   - **Explanation:** EXTREME volatility regime blocks (via `stage0_trend_vol_block`) OR UNKNOWN regime blocks
   - **Gate:** `_should_consider_entry_impl()` → regime check (lines 720-750)
   - **Config:** `allowed_vol_regimes: [LOW, MEDIUM, HIGH]` (blocks EXTREME)

2. `veto_pre_session`: 300 (29.5% of cycles)
   - **Explanation:** ASIA session blocks (SNIPER allows EU/OVERLAP/US only)
   - **Gate:** `_should_consider_entry_impl()` → session check (lines 677-720)
   - **Config:** `allowed_sessions: [EU, OVERLAP, US]` (blocks ASIA)

**Combined:** 602 bars blocked by Stage-0 gates (302 + 300, with some overlap possible)

**Other Stage-0 Vetoes:**
- `veto_pre_warmup`: 0 (warmup period not in 1-week window)
- `veto_pre_atr`: 0 (covered by `veto_pre_regime`)
- `veto_pre_killswitch`: 0 (replay mode: metrics-only, no blocking)
- `veto_pre_model_missing`: 0 (V10 model loaded successfully)
- `veto_pre_nan_features`: 0 (no NaN/Inf features detected)

### Why n_candidate_pass = n_trades_created = 288?

**Stage-1 Rejection Rate:** 30.4% (126 candidates blocked: 414 - 288)

**Top Veto Reasons (Stage-1):**
- `veto_cand_threshold`: 126 (30.4% of candidates, 100% of Stage-1 blocks)
  - **Explanation:** `p_long < min_prob_long` (0.18) threshold blocks
  - **Gate:** `apply_entry_v9_policy_sniper()` → threshold check (lines 207-227)
  - **Config:** `min_prob_long: 0.18` (see `threshold_used="long=0.18"`)
  - **Observation:** `p_long_p50=0.191` suggests ~50% of candidates have `p_long >= 0.18`, but only 69.6% pass (due to threshold distribution)
  - **Invariant:** ✅ TELEMETRY_INV_8 verified: `(n_candidates - n_candidate_pass) - sum(veto_cand_*) = 126 - 126 = 0`

**Other Stage-1 Vetoes:**
- `veto_cand_risk_guard`: 0 (no risk guard blocks)
- `veto_cand_max_trades`: 0 (max_concurrent_positions=3 not reached)
- `veto_cand_big_brain`: 0 (Big Brain gate not enabled or passed)

**Why No Other Stage-1 Blocks?**
- Risk guard: Spread/ATR thresholds not exceeded (likely due to Stage-0 volatility filtering)
- Max trades: Only 288 trades total (avg 41/day), well below concurrent limit
- Big Brain: Not enabled or passed for all candidates

**Stage-1 Attribution (TELEMETRY_INV_8):**
- All 126 blocked candidates are attributed to `veto_cand_threshold` ✅
- No missing attribution (all Stage-1 blocks have a veto reason)

### threshold_used = "long=0.18"

**Status:** ✅ Fixed (Commit 4.1)

**Value:** `threshold_used="long=0.18"` (set when threshold gate is evaluated)

**Implementation:**
- Location: `gx1/execution/entry_manager.py` → `evaluate_entry()` (lines 1729-1736 for SNIPER policy)
- Set **before** `apply_entry_v9_policy_sniper()` call (ensures it's populated regardless of pass/fail)
- Format: `"long={min_prob_long}"` or `"long={min_prob_long},short={min_prob_short}"`
- Export: `scripts/run_mini_replay_perf.py` (line ~820), merged as `threshold_used_unique` in `merge_perf_summaries.py`

**Verification:**
- 1-week run: `threshold_used="long=0.18"` ✅
- Matches config: `entry_v9_policy_sniper.min_prob_long=0.18` ✅

### p_long_p50=0.191 vs threshold_used

**Observation:** `p_long_p50=0.191` is slightly above `min_prob_long=0.18`

**Interpretation:**
- 50% of candidates have `p_long >= 0.191`
- Threshold gate: `p_long >= 0.18` (30.4% rejection rate suggests distribution tail below 0.18)
- Pass rate: 69.6% (288/414) suggests ~30% of candidates have `p_long < 0.18`

**Distribution Analysis (from histogram):**
- `p_long_mean`: 0.206
- `p_long_p50`: 0.191
- `p_long_p90`: 0.295
- Suggests left tail (values < 0.18) contains ~30% of candidates

---

## DEL D — CLEANUP PLAN (Deprecation Candidates)

### High Priority (Low Risk)

1. **Remove FARM_V2B diagnostic state (`farm_diag`)**
   - Location: `gx1/execution/entry_manager.py` → `EntryManager.__init__()` (lines 52-64)
   - Risk: Low (SNIPER uses `entry_telemetry`, `farm_diag` is legacy)
   - Action: Remove `self.farm_diag` initialization, remove `farm_diag` updates, remove `farm_diag` export in `run_mini_replay_perf.py`
   - Verification: Run 1-week replay, verify `entry_telemetry` counters still work

2. **Remove legacy veto_counters**
   - Location: `gx1/execution/entry_manager.py` → `EntryManager.__init__()` (lines 87-101)
   - Risk: Low (replaced by `veto_pre`/`veto_cand`)
   - Action: Remove `self.veto_counters` initialization, remove `veto_counters` updates, remove `veto_counters` export
   - Verification: Run 1-week replay, verify `veto_pre`/`veto_cand` counters still work

3. **Remove FARM_V2B mode checks**
   - Location: `gx1/execution/entry_manager.py` → `evaluate_entry()` (lines 592-614)
   - Risk: Low (SNIPER doesn't use FARM_V2B mode)
   - Action: Remove `is_farm_v2b` checks, remove FARM regime inference code
   - Verification: Run 1-week replay, verify SNIPER entry flow still works

### Medium Priority (Medium Risk)

4. **Deprecate Big Brain V0 entry gating**
   - Location: `gx1/execution/entry_manager.py` → `evaluate_entry()` (line 2914: "Big Brain V0 removed")
   - Risk: Medium (code may still reference V0)
   - Action: Search for V0 references, remove if unused
   - Verification: Run 1-week replay, verify no V0 code paths

5. **Consolidate threshold_used storage**
   - Location: `gx1/execution/entry_manager.py` → `evaluate_entry()` (line ~2495)
   - Risk: Medium (currently `None` in 1-week run, need to fix export first)
   - Action: Fix `threshold_used` export, then verify it's populated
   - Verification: Run 1-week replay, verify `threshold_used` is set

### Low Priority (High Risk - Require Testing)

6. **Remove session-routed entry models (V9)**
   - Location: `gx1/execution/oanda_demo_runner.py` → `__init__()` (lines 2320-2400)
   - Risk: High (V10 uses single model, but V9 session routing may be used elsewhere)
   - Action: Verify V9 session routing is not used in SNIPER, then remove
   - Verification: Run FULLYEAR replay, verify V10-only path works

7. **Remove FARM exit rules (if SNIPER-only)**
   - Location: `gx1/execution/exit_manager.py` → `evaluate_and_close_trades()` (lines 93-200)
   - Risk: High (FARM exits may be used in other policies)
   - Action: Verify SNIPER uses EXIT_FARM_V2_RULES only, then mark FARM_V1 exits as deprecated
   - Verification: Run FULLYEAR replay, verify exit flow works

### Commit Plan (Small Commits)

1. **Commit 1:** Remove `farm_diag` initialization and updates (keep export for backward compatibility)
2. **Commit 2:** Remove `veto_counters` initialization and updates (keep export for backward compatibility)
3. **Commit 3:** Fix `threshold_used` export (ensure it's set and exported correctly)
4. **Commit 4:** Remove FARM_V2B mode checks (if SNIPER-only)
5. **Commit 5:** Remove `farm_diag`/`veto_counters` export (after verification)

---

## DEL E — FULLYEAR PLAN (Commands + Artifacts)

### Execution Commands

**1. Run FULLYEAR 2025 Parallel (7 Workers):**

```bash
cd "/Users/andrekildalbakke/Desktop/GX1 XAUUSD"

# Ensure thread limits are set (script exports them, but verify)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

# Run parallel replay
bash scripts/run_replay_fullyear_2025_parallel.sh 2>&1 | tee /tmp/fullyear_2025_parallel.log

# Check exit code
echo "EXIT_CODE=$?"
```

**2. Verify Merged Summary:**

```bash
# Extract OUTPUT_DIR from log or find most recent
OUTPUT_DIR=$(grep -o "gx1/wf_runs/FULLYEAR_2025_[^ ]*" /tmp/fullyear_2025_parallel.log | tail -1)
# Or find manually:
OUTPUT_DIR=$(find gx1/wf_runs -type d -name "FULLYEAR_2025_*" -mmin -60 | sort | tail -1)

# Verify merged summary
bash scripts/verify_fullyear_perf.sh "$OUTPUT_DIR" 2>&1 | tee /tmp/verify_fullyear.log
```

**3. Run Invariant Checks:**

```bash
# Chunk invariants (check all chunks)
for chunk_dir in "$OUTPUT_DIR/parallel_chunks/chunk_"*; do
    chunk_summary="$chunk_dir/REPLAY_PERF_SUMMARY.json"
    if [ -f "$chunk_summary" ]; then
        python3 scripts/assert_perf_invariants.py chunk "$chunk_summary" "$chunk_dir" 2>&1 | tail -20
    fi
done

# Merged invariants
python3 scripts/assert_perf_invariants.py merged "$OUTPUT_DIR/REPLAY_PERF_SUMMARY.json" "$OUTPUT_DIR/parallel_chunks/chunk_"* 2>&1 | tail -50
```

### Expected Artifacts

**Root OUTPUT_DIR:**
- `REPLAY_PERF_SUMMARY.json`: Merged summary (all chunks aggregated)
- `REPLAY_PERF_SUMMARY.md`: Human-readable merged summary
- `parallel_chunks/`: Directory containing chunk outputs

**Per Chunk (parallel_chunks/chunk_N/):**
- `REPLAY_PERF_SUMMARY.json`: Chunk performance summary
- `REPLAY_PERF_SUMMARY.md`: Chunk human-readable summary
- `WINDOW.json`: Window start/end for chunk
- `worker_started.txt`: Marker file (early crash detection)
- `env_snapshot.json`: Environment variables snapshot
- `fault.log`: Stack trace if process crashed (may not exist if no crash)
- `trade_journal/`: Trade journal directory
  - `trade_journal_index.csv`: Trade log index
  - `trades/*.json`: Individual trade JSON files

**Verification Checklist:**
- [ ] All 7 chunks have `REPLAY_PERF_SUMMARY.json` (status="complete")
- [ ] Merged summary exists: `OUTPUT_DIR/REPLAY_PERF_SUMMARY.json`
- [ ] Invariants pass: `TELEMETRY_INV_1-7` (chunk + merged)
- [ ] `n_trades_created == trades_total` (from trade journal)
- [ ] `n_candidates > 0` (V10 predictions working)
- [ ] `p_long_p50 ~0.19` (consistent with 1-week run)
- [ ] Top 5 `veto_pre` and `veto_cand` make sense
- [ ] No chunks with `status="incomplete"` (fail-fast should catch this)

**Expected Metrics (FULLYEAR 2025):**
- `bars_total`: ~70,217 (full year M5 bars)
- `n_cycles`: ~70,217 (all bars evaluated)
- `n_precheck_pass`: ~28,500 (40% pass rate, similar to 1-week)
- `n_candidates`: ~28,500 (100% of precheck_pass)
- `n_trades_created`: ~11,500 (40% of candidates, similar to 1-week: 288/414 = 69.6%)
- `feat_time_sec`: ~2,200s (extrapolated from 1-week: 32s * 52 weeks)
- `duration_sec`: ~3,600s (1 hour, parallel wall-clock)

---

## APPENDIX: File Reference Index

**Entry Pipeline:**
- `gx1/execution/entry_manager.py`: EntryManager class, Stage-0/Stage-1 gates, telemetry (lines 33-4329)
- `gx1/execution/oanda_demo_runner.py`: GX1DemoRunner, V10 inference, Stage-0 filter (lines 5888-6150, 638-780)
- `gx1/features/runtime_v9.py`: Feature building, column normalization (lines 211-550)
- `gx1/features/basic_v1.py`: Basic V1 features (NumPy rolling)
- `gx1/policy/entry_v9_policy_sniper.py`: SNIPER threshold gate (lines 29-280)

**Exit Pipeline:**
- `gx1/execution/exit_manager.py`: Exit evaluation (lines 44-660)
- `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/exits/SNIPER_EXIT_RULES_A_P4_1.yaml`: Exit config

**Telemetry:**
- `gx1/execution/entry_manager.py`: `entry_telemetry` container, veto counters (lines 40-126)
- `scripts/run_mini_replay_perf.py`: Export telemetry to JSON (lines 701-820)
- `scripts/merge_perf_summaries.py`: Merge telemetry (lines 78-428)
- `scripts/assert_perf_invariants.py`: Invariant checks (lines 17-350)

**Orchestration:**
- `scripts/run_mini_replay_perf.py`: Single chunk runner (lines 240-945)
- `scripts/run_replay_fullyear_2025_parallel.sh`: Parallel orchestrator
- `scripts/verify_fullyear_perf.sh`: Verification script
- `scripts/merge_perf_summaries.py`: Merge script (lines 1-428)

**Config:**
- `gx1/configs/policies/active/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY.yaml`: Baseline SNIPER config
- `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/ENTRY_V9_SNIPER_LONDON_NY_P4_1.yaml`: Entry config
- `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/exits/SNIPER_EXIT_RULES_A_P4_1.yaml`: Exit config
- `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/SNIPER_RISK_GUARD_V1.yaml`: Risk guard config

---

**Document Status:** COMPLETE (Ready for Review)

