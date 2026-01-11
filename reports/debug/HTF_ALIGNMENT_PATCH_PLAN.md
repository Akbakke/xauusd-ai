# HTF Alignment Patch Plan (Minimal, Non-Breaking)

## Summary

**Problem:** 52.61% HTF alignment warning rate + high load average (21.54) + 34% swap usage
**Root Cause:** Alignment recalculated per bar (7-8 calls) even when HTF unchanged
**Fix:** Add timing instrumentation first (evidence), then implement HTF cache per chunk

## Patch 1: Add HTF Alignment Timing (Next Run)

### Files to Modify

#### 1. `gx1/features/basic_v1.py`

**Location:** `_align_htf_to_m5_numpy()` function (line 173)

**Change:**
```python
def _align_htf_to_m5_numpy(
    htf_values: np.ndarray,
    htf_close_times: np.ndarray,
    m5_timestamps: np.ndarray,
    is_replay: bool
) -> np.ndarray:
    """
    Align HTF values to M5 timestamps using searchsorted (NO PANDAS).
    ...
    """
    # ADD: Per-call timing
    t_align_call_start = time.perf_counter()
    
    if len(htf_close_times) == 0:
        # ... existing code ...
        if is_replay:
            t_align_call_end = time.perf_counter()
            from gx1.utils.perf_timer import perf_add
            perf_add("feat.htf_align.call_total", t_align_call_end - t_align_call_start)
            perf_add("feat.htf_align.call_count", 1.0)
        return np.zeros(len(m5_timestamps), dtype=np.float64)
    
    # Use searchsorted to find last completed HTF bar for each M5 timestamp
    indices = np.searchsorted(htf_close_times, m5_timestamps, side="right") - 1
    
    # ... existing code (lines 210-278) ...
    
    # ADD: Warning path tracking
    if np.any(indices < 0):
        if is_replay:
            from gx1.utils.perf_timer import perf_add
            perf_add("feat.htf_align.warning_path", 1.0)
            n_missing = np.sum(indices < 0)
            # Conservative estimate: 0.5us per missing bar
            estimated_overhead = n_missing * 0.5e-6
            perf_add("feat.htf_align.warning_overhead_est", estimated_overhead)
        # ... existing warning logic ...
    
    # ... existing code (lines 280-288) ...
    
    # ADD: End timing
    t_align_call_end = time.perf_counter()
    if is_replay:
        from gx1.utils.perf_timer import perf_add
        perf_add("feat.htf_align.call_total", t_align_call_end - t_align_call_start)
        perf_add("feat.htf_align.call_count", 1.0)
    
    return shifted
```

#### 2. `gx1/execution/oanda_demo_runner.py`

**Location:** `_run_replay_impl()` around line 9995 (after `htf_align_warn_count`)

**Change:**
```python
# After line 9980 (htf_align_warn_count retrieval)
# Get HTF alignment timing from perf collector
htf_align_time_total_sec = 0.0
htf_align_warning_time_sec = 0.0
htf_align_call_count = 0
if hasattr(self, "perf_collector") and self.perf_collector:
    perf_data = self.perf_collector.get_all()
    for name, data in perf_data.items():
        if name.startswith("feat.htf_align.call_total"):
            htf_align_time_total_sec += data.get("total_sec", 0.0)
        elif name.startswith("feat.htf_align.call_count"):
            htf_align_call_count += int(data.get("count", 0))
        elif name.startswith("feat.htf_align.warning_overhead_est"):
            htf_align_warning_time_sec += data.get("total_sec", 0.0)
```

**Location:** Chunk summary log (line 9995-10020)

**Change:**
```python
# Add to log.info() call (line 9995)
log.info(
    "[REPLAY_PERF_SUMMARY] "
    "... existing fields ... | "
    "htf_align_time=%.2fs | htf_align_calls=%d | htf_align_warn_overhead=%.2fs",
    # ... existing args ...
    htf_align_time_total_sec,
    htf_align_call_count,
    htf_align_warning_time_sec,
)
```

#### 3. `gx1/scripts/replay_eval_gated_parallel.py`

**Location:** `process_chunk()` - chunk footer (line 492)

**Change:**
```python
chunk_footer = {
    # ... existing fields ...
    "htf_align_warn_count": convert_to_json_serializable(htf_align_warn_count),
    # ADD:
    "htf_align_time_total_sec": convert_to_json_serializable(htf_align_time_total_sec),
    "htf_align_call_count": convert_to_json_serializable(htf_align_call_count),
    "htf_align_warning_time_sec": convert_to_json_serializable(htf_align_warning_time_sec),
    # ... rest of fields ...
}
```

**Location:** `process_chunk()` - chunk_artifacts (line 534)

**Change:**
```python
chunk_artifacts = {
    # ... existing fields ...
    "htf_align_warn_count": htf_align_warn_count,
    # ADD:
    "htf_align_time_total_sec": htf_align_time_total_sec,
    "htf_align_call_count": htf_align_call_count,
    "htf_align_warning_time_sec": htf_align_warning_time_sec,
    # ... rest of fields ...
}
```

**Location:** `export_perf_json_from_footers()` - chunks_metrics (line 621)

**Change:**
```python
chunks_metrics.append({
    # ... existing fields ...
    "htf_align_warn_count": footer.get("htf_align_warn_count", 0),
    # ADD:
    "htf_align_time_total_sec": footer.get("htf_align_time_total_sec", 0.0),
    "htf_align_call_count": footer.get("htf_align_call_count", 0),
    "htf_align_warning_time_sec": footer.get("htf_align_warning_time_sec", 0.0),
    # ... rest of fields ...
})
```

### Expected Output (Next Run)

**Chunk footer will contain:**
```json
{
  "htf_align_warn_count": 12345,
  "htf_align_time_total_sec": 45.2,
  "htf_align_call_count": 560000,
  "htf_align_warning_time_sec": 2.1
}
```

**Perf JSON will aggregate:**
- Total alignment time across all chunks
- Total alignment calls
- Warning overhead estimate

## Patch 2: CPU Throttling Monitoring (Run Now)

### Commands to Run (Safe, No Code Changes)

```bash
# 1. Monitor CPU frequency (requires sudo, but safe read-only)
sudo powermetrics --samplers cpu_power -i 5000 -n 12 > /tmp/cpu_freq.log 2>&1 &
POWERMETRICS_PID=$!
echo "Monitoring CPU frequency (PID: $POWERMETRICS_PID)"
echo "Stop with: sudo kill $POWERMETRICS_PID"

# 2. Sample worker CPU% every 30 seconds
while true; do
    echo "=== $(date) ==="
    ps aux | grep -E "replay_eval_gated|python.*replay" | grep -v grep | \
      awk '{printf "PID %s: %.1f%% CPU, %d MB RSS\n", $2, $3, $6/1024}'
    uptime
    sysctl vm.swapusage | grep -E "swap|free"
    sleep 30
done > /tmp/cpu_monitor.log 2>&1 &
MONITOR_PID=$!
echo "Monitoring CPU usage (PID: $MONITOR_PID)"
echo "Stop with: kill $MONITOR_PID"

# 3. Check thermal state (no sudo needed)
pmset -g thermlog | tail -20 > /tmp/thermal_state.txt
cat /tmp/thermal_state.txt
```

### Analysis Script

```bash
# After 5-10 minutes, analyze:
python3 << 'EOF'
import re
from pathlib import Path

# Parse CPU frequency log
freq_log = Path("/tmp/cpu_freq.log")
if freq_log.exists():
    freqs = []
    for line in freq_log.read_text().split('\n'):
        if 'Frequency' in line or 'CPU die' in line:
            # Extract frequency (e.g., "2.4 GHz")
            match = re.search(r'(\d+\.\d+)\s*GHz', line)
            if match:
                freqs.append(float(match.group(1)))
    
    if freqs:
        print(f"CPU Frequency: min={min(freqs):.2f} GHz, max={max(freqs):.2f} GHz, avg={sum(freqs)/len(freqs):.2f} GHz")
        if min(freqs) < 2.0:
            print("⚠️  THROTTLING DETECTED: Frequency dropped below 2.0 GHz")

# Parse CPU monitor log
monitor_log = Path("/tmp/cpu_monitor.log")
if monitor_log.exists():
    cpu_pcts = []
    for line in monitor_log.read_text().split('\n'):
        if '% CPU' in line:
            match = re.search(r'(\d+\.\d+)% CPU', line)
            if match:
                cpu_pcts.append(float(match.group(1)))
    
    if cpu_pcts:
        print(f"Worker CPU%: min={min(cpu_pcts):.1f}%, max={max(cpu_pcts):.1f}%, avg={sum(cpu_pcts)/len(cpu_pcts):.1f}%")
        if max(cpu_pcts) < 80:
            print("⚠️  THROTTLING SUSPECTED: CPU% never reaches 100%")
EOF
```

## Patch 3: HTF Cache Implementation (After Evidence)

**Status:** Design only (not implemented yet)

**File:** `gx1/features/basic_v1.py`

**Approach:**
1. Precompute HTF alignment mapping once per chunk (in `build_basic_v1()`)
2. Store in `feature_state.htf_alignment_cache`
3. Reuse mapping for all bars in chunk
4. Hard-fail in replay if alignment becomes inconsistent

**Key Design:**
- Cache key: `(htf_close_times_hash, m5_timestamps_hash)`
- Cache value: `indices_all` (precomputed searchsorted result)
- Invalidation: Only on HTF aggregator reset (shouldn't happen in replay)

**Expected Impact:**
- Reduce alignment from O(log n) per call to O(1) per call
- Eliminate 7-8 × searchsorted calls per bar
- Reduce CPU cache misses

## Priority

1. **NOW:** Run CPU monitoring commands (Patch 2)
2. **Next Run:** Apply timing patch (Patch 1)
3. **After Evidence:** Implement HTF cache (Patch 3) if alignment time > 10% of feature time
