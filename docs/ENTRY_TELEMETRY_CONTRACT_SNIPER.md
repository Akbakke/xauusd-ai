# ENTRY TELEMETRY CONTRACT (SNIPER/NY)

**Scope:** SNIPER/NY entry pipeline telemetry definitions, invariants, and verification procedures.

**Last Updated:** 2026-01-05  
**Status:** Production (Commit 4.1 - Telemetry Integrity Fixes)

---

## 1. Overview

The entry telemetry system tracks all bar-level and candidate-level decisions in the SNIPER/NY entry pipeline, providing deterministic, comparable metrics across policy modes and levels.

**Key Design Principles:**
- **Deterministic:** Same inputs → same counters (no randomness)
- **Fail-fast:** Invariants enforced at export/merge time
- **Semantic consistency:** Counter names and semantics match across single/parallel replays
- **SNIPER/NY-only:** FARM-specific telemetry is deprecated (see cleanup plan)

---

## 2. Two-Stage Gating Model

### Stage-0 (Precheck)
**Purpose:** Bar-level filtering before model prediction

**When:** Before any model inference (V10/V9)

**Gates:**
- Session gate (EU/OVERLAP/US only, blocks ASIA)
- Regime gate (volatility/trend regime filtering)
- Warmup gate (first N bars)
- Killswitch gate (parity/coverage thresholds, live-only)
- Model missing gate (hard fail if required model not loaded)
- NaN features gate (hard fail if NaN/Inf detected)

**Counter:** `n_precheck_pass` (bars that pass Stage-0, prediction allowed)

**Veto Counters:** `veto_pre_*` (all Stage-0 blocking reasons)

### Stage-1 (Candidate)
**Purpose:** Candidate-level filtering after model prediction

**When:** After V10/V9 prediction, before trade creation

**Gates:**
- Threshold gate (`p_long >= min_prob_long`)
- Risk guard gate (spread/ATR/intraday DD/cluster)
- Max trades gate (`len(open_trades) < max_concurrent_positions`)
- Big Brain gate (if enabled)

**Counter:** `n_candidate_pass` (candidates that pass all Stage-1 gates)

**Veto Counters:** `veto_cand_*` (all Stage-1 blocking reasons)

---

## 3. Counter Definitions

### Core Counters

| Counter | Definition | Increment Condition | Stage |
|---------|------------|---------------------|-------|
| `n_cycles` | Total bar cycles evaluated | Every bar evaluation (start of `evaluate_entry()`) | - |
| `n_precheck_pass` | Bars passing Stage-0 | Stage-0 passes (`should_consider=True`) | Stage-0 |
| `n_predictions` | Predictions produced | V10/V9 prediction OK (finite `prob_long`) | Stage-1 |
| `n_candidates` | Valid predictions (alias to `n_predictions`) | Same as `n_predictions` | Stage-1 |
| `n_candidate_pass` | Candidates passing Stage-1 | All Stage-1 gates pass (before trade creation) | Stage-1 |
| `n_trades_created` | Trades actually created (SINGLE SOURCE OF TRUTH) | `LiveTrade` object instantiated | Stage-1 |

**Relationships:**
- `n_precheck_pass <= n_cycles`
- `n_candidates <= n_precheck_pass`
- `n_candidate_pass <= n_candidates`
- `n_trades_created <= n_candidate_pass`
- For replay: `n_trades_created == trades_total` (from trade journal)

### Veto Counters (Stage-0)

| Counter | Gate | Trigger Condition |
|---------|------|-------------------|
| `veto_pre_session` | Session gate | Session not in `allowed_sessions` |
| `veto_pre_regime` | Regime gate | EXTREME vol or UNKNOWN regime |
| `veto_pre_atr` | ATR gate | EXTREME vol (covered by regime gate) |
| `veto_pre_warmup` | Warmup gate | `current_ts < warmup_floor` |
| `veto_pre_spread` | Spread gate | Spread > threshold (rare, usually covered by Stage-1) |
| `veto_pre_killswitch` | Killswitch gate | Parity/coverage thresholds exceeded (live-only) |
| `veto_pre_model_missing` | Model missing gate | Required model not loaded |
| `veto_pre_nan_features` | NaN features gate | NaN/Inf detected in features |

**Invariant:** `sum(veto_pre_*) <= (n_cycles - n_precheck_pass)`

### Veto Counters (Stage-1)

| Counter | Gate | Trigger Condition |
|---------|------|-------------------|
| `veto_cand_threshold` | Threshold gate | `p_long < min_prob_long` (or `p_short < min_prob_short`) |
| `veto_cand_risk_guard` | Risk guard gate | Spread/ATR/intraday DD/cluster thresholds exceeded |
| `veto_cand_max_trades` | Max trades gate | `len(open_trades) >= max_concurrent_positions` |
| `veto_cand_big_brain` | Big Brain gate | Big Brain gate returns `False` |

**Invariant:** `sum(veto_cand_*) == (n_candidates - n_candidate_pass)` (TELEMETRY_INV_8)

---

## 4. p_long Statistics (Histogram-Based)

**Storage:** 200-bin histogram (bins: 0.0 to 1.0)

**Rationale:** Avoid storing large arrays for full-year replays (O(1) memory per chunk)

**Calculation:**
1. **Per chunk:** Collect `p_long_values` array (finite values only, clipped to [0.0, 1.0])
2. **Histogram:** `np.histogram(p_long_values, bins=200, range=(0.0, 1.0))`
3. **Export:** `p_long_histogram` dict with `bins`, `min`, `max`, `counts`, `total_count`
4. **Merge:** Sum histogram counts across chunks, recompute quantiles from cumulative distribution

**Statistics:**
- `p_long_mean`: Mean of finite `p_long` values
- `p_long_p50`: 50th percentile (median)
- `p_long_p90`: 90th percentile
- `p_long_min`: Minimum finite value
- `p_long_max`: Maximum finite value

**Location:**
- Export: `scripts/run_mini_replay_perf.py` (lines ~780-810)
- Merge: `scripts/merge_perf_summaries.py` (lines ~200-250)

---

## 5. threshold_used

**Purpose:** Track threshold configuration used during threshold gate evaluation

**Format:**
- Single threshold: `"long=0.18"`
- Dual threshold: `"long=0.18,short=0.72"`

**Storage:**
- Per chunk: `threshold_used` (string)
- Merged: `threshold_used_min`, `threshold_used_max`, `threshold_used_unique` (list, capped at 20)

**Implementation:**
- Set in `EntryManager.evaluate_entry()` **before** threshold gate evaluation (lines 1729-1736)
- Ensures it's populated regardless of pass/fail
- Source: `entry_v9_policy_sniper.min_prob_long` / `min_prob_short`

**Invariant:** `threshold_used` must never be `None` when threshold gate is evaluated (Commit 4.1 fix)

---

## 6. Invariants (TELEMETRY_INV_1-8)

**Location:** `scripts/assert_perf_invariants.py`

### TELEMETRY_INV_1: Precheck Pass Bounds
```
0 <= n_precheck_pass <= n_cycles
```

**Rationale:** Cannot pass more bars than evaluated, cannot be negative

### TELEMETRY_INV_2: Candidates Bounded by Precheck Pass
```
0 <= n_candidates <= n_precheck_pass
```

**Rationale:** Cannot have more predictions than bars that passed precheck

### TELEMETRY_INV_3: Candidate Pass Bounded by Candidates
```
0 <= n_candidate_pass <= n_candidates
```

**Rationale:** Cannot pass more candidates than exist

### TELEMETRY_INV_4: Trades Bounded by Candidate Pass
```
0 <= n_trades_created <= n_candidate_pass
```

**Rationale:** Cannot create more trades than candidates that passed Stage-1

### TELEMETRY_INV_5: Trades Match Trade Journal (Replay)
```
n_trades_created == trades_total
```

**Rationale:** Replay mode: `n_trades_created` is source of truth, must match trade journal count

**Note:** Only enforced if `trades_total > 0` (available from trade journal)

### TELEMETRY_INV_6: Stage-0 Veto Sum Bounded
```
sum(veto_pre_*) <= (n_cycles - n_precheck_pass)
```

**Rationale:** Total Stage-0 vetoes cannot exceed total bars blocked at Stage-0

**Note:** Allows for overlap (one bar can be vetoed by multiple gates, but only counts once in `n_cycles - n_precheck_pass`)

### TELEMETRY_INV_7: Stage-1 Veto Sum Bounded
```
sum(veto_cand_*) <= (n_candidates - n_candidate_pass)
```

**Rationale:** Total Stage-1 vetoes cannot exceed total candidates blocked at Stage-1

### TELEMETRY_INV_8: Stage-1 Attribution Invariant (Commit 4.1)
```
(n_candidates - n_candidate_pass) - sum(veto_cand_*) == 0
```

**Rationale:** **ALL** blocked candidates must be attributed to a `veto_cand_*` reason

**Failure Mode:** If `missing > 0`, indicates missing veto attribution (counter bug)

**Location:** `scripts/assert_perf_invariants.py` (lines 174-184)

---

## 7. Verification Procedures

### 1-Week Replay Verification

**Command:**
```bash
bash scripts/run_replay_1w_perf.sh
```

**Expected Metrics (Reference):**
- `n_cycles`: ~1,016
- `n_precheck_pass`: ~414 (40.7% pass rate)
- `n_candidates`: ~414 (100% of precheck_pass)
- `n_candidate_pass`: ~288 (69.6% of candidates)
- `n_trades_created`: ~288 (100% of candidate_pass, matches `trades_total`)
- `veto_pre_session`: ~300 (29.5% of cycles)
- `veto_pre_regime`: ~302 (29.7% of cycles)
- `veto_cand_threshold`: ~126 (30.4% of candidates)
- `p_long_p50`: ~0.191
- `threshold_used`: `"long=0.18"`

**Invariant Check:**
```bash
python3 scripts/assert_perf_invariants.py chunk \
  "$OUTPUT_DIR/REPLAY_PERF_SUMMARY.json" \
  "$OUTPUT_DIR"
```

**Expected:** `PASS: All invariants satisfied`

### FULLYEAR 2025 Parallel Verification

**Command:**
```bash
bash scripts/run_replay_fullyear_2025_parallel.sh 2>&1 | tee /tmp/fullyear_2025_parallel.log
```

**After Completion:**
```bash
# Find output dir
OUTPUT_DIR=$(find gx1/wf_runs -type d -name "FULLYEAR_2025_*" -mmin -120 | sort | tail -1)

# Verify merged summary
bash scripts/verify_fullyear_perf.sh "$OUTPUT_DIR" 2>&1 | tee /tmp/verify_fullyear.log

# Chunk invariants (all chunks)
for chunk_dir in "$OUTPUT_DIR/parallel_chunks/chunk_"*; do
    chunk_summary="$chunk_dir/REPLAY_PERF_SUMMARY.json"
    if [ -f "$chunk_summary" ]; then
        python3 scripts/assert_perf_invariants.py chunk "$chunk_summary" "$chunk_dir"
    fi
done

# Merged invariants
python3 scripts/assert_perf_invariants.py merged \
  "$OUTPUT_DIR/REPLAY_PERF_SUMMARY.json" \
  "$OUTPUT_DIR/parallel_chunks/chunk_"*
```

**Key Metrics to Verify:**
- All chunks: `status="complete"` (no incomplete chunks)
- Merged: `n_trades_created == trades_total`
- All invariants: `TELEMETRY_INV_1-8 PASS`
- Top 5 `veto_pre` and `veto_cand` make sense
- `threshold_used_unique` populated (contains config value)
- Trade session distribution reasonable (EU/OVERLAP/US only)

---

## 8. Export and Merge

### Per-Chunk Export

**Location:** `scripts/run_mini_replay_perf.py` → `write_perf_summary()` (lines ~700-850)

**Fields:**
- `entry_counters`: Core counters (`n_cycles`, `n_precheck_pass`, `n_candidates`, `n_candidate_pass`, `n_trades_created`)
- `entry_counters`: Veto counters (`veto_pre_*`, `veto_cand_*`)
- `entry_counters`: p_long stats (`p_long_mean`, `p_long_p50`, `p_long_p90`, `p_long_min`, `p_long_max`)
- `entry_counters`: `p_long_histogram` (200 bins)
- `entry_counters`: `threshold_used` (string)
- `entry_counters`: Session counts (`candidate_session_counts`, `trade_session_counts`)

### Global Merge

**Location:** `scripts/merge_perf_summaries.py` (lines 78-428)

**Merge Rules:**
- Counters: SUM (`n_cycles`, `n_precheck_pass`, `n_candidates`, `n_candidate_pass`, `n_trades_created`)
- Veto: SUM per reason (`veto_pre_*`, `veto_cand_*`)
- Duration: MAX (`duration_sec` - parallel wall-clock)
- Feat_time: SUM (`feat_time_sec`)
- p_long histogram: SUM bin counts, recompute quantiles from cumulative distribution
- threshold_used: Aggregate to `threshold_used_min`, `threshold_used_max`, `threshold_used_unique`
- Session counts: SUM (`candidate_session_counts`, `trade_session_counts`)

**Fail-Fast:**
- Missing chunk summaries → raise `SystemExit`
- Incomplete chunks (`status != "complete"`) → fail merge

---

## 9. Deprecated/Legacy Fields

**Status:** Marked for removal in cleanup (Commit 5)

**Fields:**
- `farm_diag`: FARM_V2B diagnostic state (replaced by `entry_telemetry`)
- `veto_counters`: Legacy veto counters (replaced by `veto_pre`/`veto_cand`)

**Backward Compatibility:**
- Export scripts may still include legacy fields as empty dicts (during transition)
- Will be removed after verification that no external scripts depend on them

---

## 10. References

- **AS-BUILT Overview:** `docs/SNIPER_NY_AS_BUILT_OVERVIEW.md`
- **Invariant Implementation:** `scripts/assert_perf_invariants.py`
- **Export Implementation:** `scripts/run_mini_replay_perf.py`
- **Merge Implementation:** `scripts/merge_perf_summaries.py`
- **Entry Manager:** `gx1/execution/entry_manager.py` (telemetry container, counter increments)
- **Verify Script:** `scripts/verify_fullyear_perf.sh`

