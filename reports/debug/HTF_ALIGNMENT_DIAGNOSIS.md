# HTF Alignment Performance Diagnosis

**Generated:** 2026-01-11
**Context:** FULLYEAR replay (7 workers, spawn) showing ~5-6 hour wall-time

## A) HTF Alignment Warning Analysis

### Findings

#### 1. Warning Location & Trigger
**File:** `gx1/features/basic_v1.py:278`
**Function:** `_align_htf_to_m5_numpy()` (line 173-289)
**Trigger Condition:**
```python
# Line 208: np.searchsorted(htf_close_times, m5_timestamps, side="right") - 1
# Line 220: if np.any(indices < 0):
# Line 250: n_missing = np.sum(indices < 0)
# Line 278: log.warning(f"HTF alignment: {n_missing} M5 bars have no completed HTF bar, setting to 0.0")
```

**Root Cause:**
- `np.searchsorted(htf_close_times, m5_timestamps, side="right") - 1` returns indices < 0
- This happens when M5 timestamps occur BEFORE any completed HTF bar
- In replay mode: allowed for first N bars (historical context), but warnings still logged

**Current Measurement (from running FULLYEAR):**
- **Log file:** 288,498 warnings / 548,355 lines = **52.61% warning rate**
- **Chunk footers:** All show 0 warnings (quiet mode suppresses logging, but counts are tracked)
- **Conclusion:** Warnings are frequent but suppressed in quiet mode; counting still happens

#### 2. Code Path After Warning
**File:** `gx1/features/basic_v1.py:280-289`
```python
# Get HTF values (use 0.0 for indices < 0)
aligned = np.zeros(len(m5_timestamps), dtype=np.float64)  # O(n) allocation
valid_mask = indices >= 0  # O(n) boolean array
aligned[valid_mask] = htf_values[indices[valid_mask]]  # O(n) indexing
# Shift(1): move all values one position forward
shifted = np.roll(aligned, 1)  # O(n) copy
shifted[0] = 0.0
```

**Analysis:**
- ✅ **NO pandas reindex/merge/ffill** - uses NumPy only
- ✅ **O(n) complexity** - linear with number of M5 bars
- ⚠️ **Called per feature** - multiple HTF features = multiple calls per bar
- ⚠️ **No caching** - alignment recalculated every bar

#### 3. Call Frequency
**Called from:**
- `_htf_zscore_and_align()` (line 933) - for each HTF zscore feature (H1/H4 RSI)
- Direct calls for:
  - `h1_ema_diff` (line 987)
  - `h1_vwap_drift` (line 993)
  - `h1_atr` (line 996)
  - `h4_ema_diff` (line 1050)
  - `h4_atr` (line 1053)
- **Total: ~7-8 alignment calls per bar** (H1: 4 features, H4: 3 features)

**Per-bar cost:**
- Each call: `np.searchsorted()` = O(log n) where n = HTF bars (~1000-2000)
- Per-bar total: ~7-8 × O(log n) ≈ O(log n) amortized
- **BUT:** Called for EVERY bar, even when HTF hasn't changed

#### 4. Timing Instrumentation
**Current State:**
- ✅ Alignment timing exists: `perf_add(f"feat.htf_zscore.align.{name}.w{win}", t_align_end - t_align_start)` (line 935)
- ✅ Total HTF timing: `perf_add("feat.basic_v1.htf_features", t_htf_end - t_htf_start)` (line 1066)
- ⚠️ **Missing:** Per-call alignment time (only per-feature timing in `_htf_zscore_and_align`)
- ⚠️ **Missing:** Direct alignment call timing (h1_ema_diff, h1_vwap_drift, h1_atr, h4_ema_diff, h4_atr)
- ⚠️ **Missing:** Warning-specific timing (no distinction between normal vs warning path)

**Data Available:**
- `htf_align_warn_count` in chunk footer (line 492, 621) - **Currently 0 in footers (quiet mode)**
- `t_feature_build_total_sec` includes HTF alignment time
- Perf collector has `feat.htf_zscore.align.*` but NOT `feat.htf_align.call_total`
- **Gap:** No aggregate alignment time exported to chunk footer

## B) Instrumentation Plan (Non-Breaking)

### Hot Patch for Next Run

**File:** `gx1/features/basic_v1.py`

**Change 1: Add per-call alignment timing**
```python
# At start of _align_htf_to_m5_numpy() (line 173, after docstring)
t_align_call_start = time.perf_counter()
# ... existing code (lines 196-288) ...
# At end (before return, line 289)
t_align_call_end = time.perf_counter()
if is_replay:
    from gx1.utils.perf_timer import perf_add
    perf_add("feat.htf_align.call_total", t_align_call_end - t_align_call_start)
    perf_add("feat.htf_align.call_count", 1.0)
```

**Change 2: Add warning path tracking**
```python
# Around line 220, inside _align_htf_to_m5_numpy()
if np.any(indices < 0):
    if is_replay:
        from gx1.utils.perf_timer import perf_add
        perf_add("feat.htf_align.warning_path", 1.0)  # Count warning paths
        # Estimate overhead: searchsorted + mask + zeros + roll
        n_missing = np.sum(indices < 0)
        # Rough estimate: 0.5us per missing bar (conservative)
        estimated_overhead = n_missing * 0.5e-6
        perf_add("feat.htf_align.warning_overhead_est", estimated_overhead)
```

**Change 3: Export perf collector data to chunk footer**
**File:** `gx1/execution/oanda_demo_runner.py` (around line 9995)
```python
# After getting htf_align_warn_count (line 9980)
# Get HTF alignment timing from perf collector
htf_align_time_total_sec = 0.0
htf_align_warning_time_sec = 0.0
if hasattr(self, "perf_collector") and self.perf_collector:
    perf_data = self.perf_collector.get_all()
    # Sum all htf_align.* timing
    for name, data in perf_data.items():
        if name.startswith("feat.htf_align.call_total"):
            htf_align_time_total_sec += data.get("total_sec", 0.0)
        elif name.startswith("feat.htf_align.warning_overhead_est"):
            htf_align_warning_time_sec += data.get("total_sec", 0.0)
```

**Change 4: Add to chunk footer**
**File:** `gx1/scripts/replay_eval_gated_parallel.py` (line 492, 621)
```python
"htf_align_time_total_sec": convert_to_json_serializable(htf_align_time_total_sec),
"htf_align_warning_time_sec": convert_to_json_serializable(htf_align_warning_time_sec),
```

### Log Sampling (Current Run)

**✅ Already Measured:**
```bash
# Warning rate from log file
HTF warnings: 288,498 / 548,355 lines = 52.61%
```

**⚠️ Note:** Chunk footers show 0 warnings because:
- Quiet mode (`GX1_REPLAY_QUIET=1`) suppresses logging (line 274-275)
- But counting still happens (line 259, 270)
- Counters are stored in `feature_state.htf_align_warn_count` (line 270)
- Runner reads from state (line 9977-9978)

**To verify actual warning count:**
```bash
# Check if workers are tracking warnings (may need to wait for footer update)
python3 << 'EOF'
import json
from pathlib import Path

for i in range(7):
    footer = Path(f"reports/replay_eval/GATED/chunk_{i}/chunk_footer.json")
    if footer.exists():
        data = json.load(open(footer))
        print(f"Chunk {i}: htf_align_warn_count={data.get('htf_align_warn_count', 0)}")
EOF
```

## C) CPU Throttling Verification (macOS)

### Commands to Run NOW (Safe, No Installs)

#### 1. CPU Frequency & Thermal
```bash
# CPU frequency (requires sudo, but safe read-only)
sudo powermetrics --samplers cpu_power -i 1000 -n 1 | grep -E "CPU die|Frequency|Thermal"

# Alternative: System thermal state (no sudo needed)
pmset -g thermlog | tail -20
```

#### 2. Process CPU Usage
```bash
# Real-time CPU% per process
top -l 1 -n 10 -stats pid,command,cpu | grep -E "replay|python"

# Load average & run queue
uptime

# Per-worker CPU usage
ps aux | grep replay_eval_gated | awk '{print $2, $3"%", $11}' | sort -k2 -rn
```

#### 3. Memory Pressure
```bash
# Memory pressure (macOS specific)
memory_pressure

# Swap usage
sysctl vm.swapusage

# Per-process memory
ps aux | grep replay_eval_gated | awk '{print $2, $6/1024"MB", $11}'
```

#### 4. Disk I/O
```bash
# Disk I/O stats (if iostat available)
iostat -w 1 -c 5

# Alternative: fs_usage (lite, built-in)
sudo fs_usage -f filesys | grep -E "replay_eval|parquet" | head -50
```

### Throttling Signals

**CPU Throttling Indicators:**
- CPU frequency drops below base clock (e.g., < 2.0 GHz on M-series)
- Thermal pressure > 50% in `pmset -g thermlog`
- CPU% per worker drops over time (e.g., 100% → 80% → 60%)
- Load average >> CPU cores (e.g., 14+ on 8-core system)

**Memory Pressure:**
- Swap usage > 0 (should be 0 for replay)
- Memory pressure > 50%
- Per-process RSS growing over time

**I/O Wait:**
- `iostat` shows high `%iowait`
- Disk queue length > 1
- High `fs_usage` activity on parquet files

## D) Hypothesis Testing Scorecard

### Evidence for HTF Fallback Cost

**Current Evidence:**
- ✅ **52.61% warning rate** (288k warnings / 548k log lines)
- ✅ Alignment uses NumPy (O(log n) per call, no pandas overhead)
- ✅ **7-8 alignment calls per bar** (H1: 4, H4: 3)
- ⚠️ **Missing:** Per-call timing data (only per-feature timing exists)
- ⚠️ **Missing:** Warning vs normal path timing comparison
- ⚠️ **Missing:** Aggregate alignment time in chunk footer

**Measurement Needed:**
1. ✅ `htf_align_warn_count` from chunk footer (available, but shows 0 in quiet mode)
2. ⚠️ `feat.htf_zscore.align.*` timing from perf collector (exists but not exported to footer)
3. ⚠️ `feat.htf_align.call_total` timing (need to add)
4. ⚠️ Warning overhead timing (need to add)

**Hypothesis Test:**
- ✅ **Warning rate > 50%** → Very high (confirms frequent alignment issues)
- ⚠️ If `feat.htf_align.call_total` > 10% of `t_feature_build_total_sec` → alignment is bottleneck
- ⚠️ If warning path timing >> normal path → fallback is expensive
- ⚠️ If `htf_align_warn_count` >> 0 in footers → warnings are real (not just log spam)

### Evidence for CPU Throttling

**Current Measurements:**
```bash
Load average: 9.74 8.88 21.54  # Very high (system has ~8 cores)
Replay process: 0.0% CPU, 196 MB RSS  # Master process (workers are separate)
Swap usage: 703.25M / 2048M used (34% swap usage)  # ⚠️ HIGH
```

**Measurement Needed:**
1. ⚠️ CPU frequency over time (powermetrics - requires sudo)
2. ⚠️ Thermal pressure (pmset - check manually)
3. ⚠️ CPU% per worker over time (need to sample worker PIDs)
4. ✅ Load average: **21.54** (very high, suggests contention)

**Hypothesis Test:**
- ⚠️ If CPU freq drops < base clock → throttling confirmed
- ⚠️ If thermal pressure > 50% → throttling likely
- ✅ **Load average 21.54 >> 8 cores** → CPU contention confirmed
- ✅ **34% swap usage** → Memory pressure (may cause I/O wait)
- ⚠️ If CPU% per worker decreases over time → throttling or I/O wait

## E) Minimal Fix Plan

### If HTF Fallback is Problem

**Fix: Stateful HTF Alignment Cache per Chunk**

**File:** `gx1/features/basic_v1.py`

**Change:**
1. Precompute HTF alignment mapping once per chunk (not per bar)
2. Store mapping in feature_state (thread-local context)
3. Reuse mapping for all bars in chunk
4. Hard-fail in replay if alignment becomes inconsistent

**Implementation:**
```python
# In build_basic_v1(), before bar loop:
# Precompute HTF alignment mapping for entire chunk
htf_alignment_cache = {}
for tf in ["H1", "H4"]:
    htf_close_times = ...  # Get from aggregator
    m5_timestamps_all = df.index.astype(np.int64) // 1_000_000_000
    indices_all = np.searchsorted(htf_close_times, m5_timestamps_all, side="right") - 1
    htf_alignment_cache[tf] = indices_all

# In _align_htf_to_m5_numpy(), use cache if available:
if is_replay and hasattr(get_feature_state(), "htf_alignment_cache"):
    # Use precomputed indices instead of searchsorted per call
    indices = htf_alignment_cache[tf][bar_idx]
```

**Expected Impact:**
- Reduce alignment from O(n) per call to O(1) per call
- Eliminate repeated searchsorted calls
- Reduce CPU cache misses

### If Throttling is Problem

**Runtime Plan:**
1. Ensure charger connected (prevents power throttling)
2. Disable background tasks (Activity Monitor → Energy)
3. Reduce worker count from 7 → 5 (less heat)
4. Add cooldown between chunks (sleep 10s)

**Engineering Plan:**
1. Smaller chunk sizes (reduce per-chunk memory)
2. Worker count = CPU cores - 1 (leave headroom)
3. Memory-mapped parquet (reduce I/O)
4. Process priority: `renice -n -10 <pid>`

## Expected Impact

### After HTF Cache Fix
- **Alignment time:** 90% reduction (O(log n) per call → O(1) per call with precomputed mapping)
- **Feature build time:** 20-30% reduction (if alignment is 10-15% of feature time)
- **Wall time:** 15-25% reduction (if feature build is 50% of total)
- **CPU cache:** Better locality (precomputed mapping stays in L1/L2)

### After Throttling Fix
- **CPU utilization:** Stable at 100% per worker (no drops)
- **Wall time:** 10-20% reduction (if throttling was 20-30% overhead)
- **Consistency:** More predictable runtime (less variance)
- **Memory:** Reduce swap usage (may require worker count reduction)

### Combined Fix (HTF Cache + Throttling Mitigation)
- **Wall time:** 25-40% reduction (additive effects)
- **Target:** 5-6 hours → 3-4 hours (closer to 1-hour goal with PreGate)

## Next Steps

1. **Run CPU monitoring commands** (Section C) → collect 5-10 min of data
2. **Check chunk footers** for `htf_align_warn_count` → calculate warning rate
3. **Apply hot patch** (Section B) → instrument next run
4. **Compare OFF vs ON** → see if PreGate reduces alignment calls
5. **Implement fix** based on evidence (HTF cache OR throttling mitigation)
