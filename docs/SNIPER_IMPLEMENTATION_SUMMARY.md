# SNIPER Implementation Summary

**Status:** ✅ COMPLETE  
**Date:** 2025-12-17  
**Approach:** Minimal new code, parameterized existing FARM policy

---

## Executive Summary

SNIPER v1 has been implemented as a **parameterized wrapper** around the existing FARM_V2B entry policy, with:
- ✅ New session/vol guard (`sniper_guard_v1`) for EU/OVERLAP/US + HIGH vol
- ✅ Reuse of FARM_V2B scoring, thresholds, meta-model (no copy-paste)
- ✅ Separate policy bundle (`sniper_snapshot/2025_SNIPER_V1/`)
- ✅ SNIPER-specific exit configs (can be tuned independently)
- ✅ Config-parametric router (same model, different params possible)

**Key Principle:** SNIPER changes behavior via **config parameters**, not code duplication.

---

## Files Changed/Added

### New Files

1. **`gx1/policy/entry_v9_policy_sniper.py`** (~200 lines)
   - Wrapper around `apply_entry_v9_policy_farm_v2b`
   - Applies `sniper_guard_v1` before calling FARM policy
   - Handles trend as soft gate (logs, doesn't hard-filter)

2. **`gx1/policy/farm_guards.py`** (modified)
   - Added `session_vol_guard()` - generalized guard function
   - Added `sniper_guard_v1()` - calls `session_vol_guard` with SNIPER params
   - Refactored `farm_brutal_guard_v2()` to use `session_vol_guard`

3. **`gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/ENTRY_V9_SNIPER_LONDON_NY.yaml`**
   - SNIPER entry config
   - `allowed_sessions: ["EU", "OVERLAP", "US"]`
   - `allowed_vol_regimes: ["LOW", "MEDIUM", "HIGH"]`
   - `min_prob_long: 0.67` (slightly lower than FARM's 0.68)

4. **`gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY.yaml`**
   - Full SNIPER policy
   - Points to SNIPER entry/exit configs
   - Uses same router model as FARM (can tune params)

5. **`gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/exits/SNIPER_EXIT_RULES_A.yaml`**
   - SNIPER-specific RULE5 exit config
   - Same params as FARM in v1 (can tune later)

6. **`gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/exits/SNIPER_EXIT_RULES_ADAPTIVE.yaml`**
   - SNIPER-specific RULE6A exit config
   - Same params as FARM in v1 (can tune later)

7. **`scripts/run_practice_live_sniper_london_ny.sh`**
   - Wrapper script for SNIPER live practice runs
   - Similar to `run_live_force_one_trade_any_session.sh`

8. **`docs/FARM_LOCK.md`**
   - Documents FARM PROD_BASELINE lock
   - Lists locked files and verification commands

9. **`docs/SNIPER_BACKTEST_PLAN.md`**
   - Backtest plan for SNIPER (30D, 90D, FULLYEAR)
   - Session filtering (EU/OVERLAP/US only)

10. **`docs/SNIPER_BACKTEST_REPORT.md`**
    - Report template for SNIPER backtest results
    - Metrics: trades per session, vol bucket, MFE/MAE, etc.

### Modified Files

1. **`gx1/execution/entry_manager.py`**
   - Added `use_sniper` flag detection
   - Calls `apply_entry_v9_policy_sniper` if SNIPER policy detected
   - Applies `sniper_guard_v1` before main policy

2. **`gx1/policy/farm_guards.py`**
   - Generalized `session_vol_guard()` function
   - Refactored `farm_brutal_guard_v2()` to use generalized guard
   - Added `sniper_guard_v1()` for SNIPER

---

## Key Design Decisions

### 1. Parameterization Over Duplication

**Decision:** SNIPER reuses FARM_V2B's scoring/thresholding logic via config parameters.

**Rationale:**
- Avoids code duplication
- Ensures consistency in scoring logic
- Makes it easier to maintain and update

**Implementation:**
- `apply_entry_v9_policy_sniper()` calls `apply_entry_v9_policy_farm_v2b()` after applying SNIPER guard
- SNIPER guard filters to EU/OVERLAP/US + LOW/MEDIUM/HIGH vol
- FARM_V2B then applies scoring, thresholds, meta-model on filtered data

### 2. Separate Policy Bundle

**Decision:** SNIPER has its own `sniper_snapshot/` directory, separate from `prod_snapshot/`.

**Rationale:**
- Clear separation of concerns
- Prevents accidental modification of FARM PROD_BASELINE
- Allows independent tuning of SNIPER configs

**Structure:**
```
sniper_snapshot/2025_SNIPER_V1/
├── ENTRY_V9_SNIPER_LONDON_NY.yaml
├── GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY.yaml
├── exits/
│   ├── SNIPER_EXIT_RULES_A.yaml
│   └── SNIPER_EXIT_RULES_ADAPTIVE.yaml
└── router/  (future: SNIPER-specific router params)
```

### 3. Config-Parametric Router

**Decision:** SNIPER uses same router model as FARM, but can tune parameters (e.g., `v3_range_edge_cutoff`).

**Rationale:**
- Router model is expensive to train (reuse existing)
- Parameters can be tuned per-engine without code changes
- Backward-compatible (if param not in config, use default)

**Implementation:**
- Router reads `v3_range_edge_cutoff` from policy YAML
- SNIPER can set different cutoff than FARM (currently same: 1.0)
- Future: session-specific cutoffs, vol-specific thresholds

### 4. Trend as Soft Gate

**Decision:** In SNIPER v1, trend is logged but not hard-required.

**Rationale:**
- Conservative start: don't over-constrain entries
- Can analyze trend impact in backtests
- Can add hard requirement later if needed

**Implementation:**
- `require_trend_up: false` in SNIPER config
- Trend features are logged but don't block entries
- Can be enabled later via config change

---

## Commands

### 1. Run SNIPER Replay (30D)

```bash
python3 << PYEOF
from pathlib import Path
from gx1.execution.oanda_demo_runner import GX1DemoRunner
import pandas as pd

policy = Path('gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY.yaml')
runner = GX1DemoRunner(policy, dry_run_override=True, replay_mode=True)

# Load last 30 days
df = pd.read_parquet('data/raw/xauusd_m5_2025_bid_ask.parquet')
df.index = pd.to_datetime(df.index, utc=True)
end_ts = df.index.max()
start_ts = end_ts - pd.Timedelta(days=30)
df_test = df[(df.index >= start_ts) & (df.index <= end_ts)]

# Filter out ASIA session (keep only EU/OVERLAP/US)
from gx1.execution.live_features import infer_session_tag
df_test = df_test[df_test.index.map(lambda ts: infer_session_tag(ts) in ['EU', 'OVERLAP', 'US'])]

output_dir = Path('gx1/wf_runs/SNIPER_OBS_30D_$(date +%Y%m%d_%H%M%S)')
output_dir.mkdir(parents=True, exist_ok=True)
test_data = output_dir / 'test_data.parquet'
df_test.to_parquet(test_data)

runner.run_replay(test_data)
PYEOF
```

### 2. Run SNIPER Live Practice

```bash
./scripts/run_practice_live_sniper_london_ny.sh
```

**Requirements:**
- `OANDA_ENV=practice`
- `OANDA_API_TOKEN` set
- `OANDA_ACCOUNT_ID` set
- `I_UNDERSTAND_LIVE_TRADING=YES` (if live trading)

### 3. Verify FARM Lock

```bash
./scripts/check_farm_lock.sh
```

**Expected:** All FARM PROD_BASELINE files match expected hashes ✅

### 4. Verify SNIPER Run

```bash
# Check run_header.json
jq '.meta.role' gx1/wf_runs/SNIPER_OBS_*/run_header.json

# Check trade journal
ls -lth gx1/wf_runs/SNIPER_OBS_*/trade_journal/trades/*.json | head

# Verify all entries are in EU/OVERLAP/US
python3 << PYEOF
import json
from pathlib import Path
from gx1.execution.live_features import infer_session_tag

run_dir = Path('gx1/wf_runs/SNIPER_OBS_*/')
trades = list(run_dir.glob('trade_journal/trades/*.json'))

for trade_file in trades:
    trade = json.load(open(trade_file))
    entry_ts = pd.to_datetime(trade['entry_snapshot']['timestamp'], utc=True)
    session = infer_session_tag(entry_ts)
    print(f"{trade_file.name}: session={session}, vol={trade['entry_snapshot'].get('vol_regime', 'UNKNOWN')}")
PYEOF
```

---

## Expected Differences vs FARM

### Trade Rate

**FARM (ASIA only):**
- Sessions: ASIA only
- Vol regimes: LOW, MEDIUM only
- Expected: ~3 trades/day

**SNIPER (EU/OVERLAP/US):**
- Sessions: EU, OVERLAP, US (3 sessions vs 1)
- Vol regimes: LOW, MEDIUM, HIGH (3 regimes vs 2)
- Expected: 2-3x FARM = **6-9 trades/day**

**Rationale:**
- More sessions = more opportunities
- HIGH vol allowed = more entries
- Lower threshold (0.67 vs 0.68) = slightly more entries

### Session Distribution

**Expected SNIPER distribution:**
- EU: ~30-40% of trades
- OVERLAP: ~20-30% of trades
- US: ~30-40% of trades

**Note:** Distribution depends on market conditions and volatility patterns.

### Volatility Distribution

**Expected SNIPER distribution:**
- LOW: ~40-50% of trades
- MEDIUM: ~30-40% of trades
- HIGH: ~10-20% of trades (NEW - not in FARM)

**Note:** HIGH vol trades are new for SNIPER. Performance will be analyzed in backtests.

---

## Tuning Rounds (Future)

### Round 1: Entry Threshold Sweep

**Goal:** Find optimal `min_prob_long` for SNIPER.

**Sweep:** `[0.66, 0.67, 0.68, 0.69, 0.70]`

**Metrics:**
- Trade rate (trades/day)
- EV/trade per threshold
- Win rate per threshold

**Command:**
```bash
for threshold in 0.66 0.67 0.68 0.69 0.70; do
  # Update ENTRY_V9_SNIPER_LONDON_NY.yaml: min_prob_long: $threshold
  # Run replay
  # Collect metrics
done
```

### Round 2: Session/Vol Policy

**Goal:** Test HIGH vol only in OVERLAP vs HIGH in all sessions.

**Tests:**
- Test 1: HIGH vol only in OVERLAP
- Test 2: HIGH vol in EU/US also

**Metrics:**
- Trade rate per session
- EV/trade per session
- HIGH vol performance

### Round 3: Exit Profile Tuning

**Goal:** Tune SNIPER exit profiles independently of FARM.

**Parameters:**
- `rule_a_profit_max_bps` (currently 9.0)
- `rule_a_profit_min_bps` (currently 5.0)
- `rule_a_trailing_stop_bps` (currently 2.0)

**Metrics:**
- MFE/MAE per exit profile
- Hold time distribution
- Win rate per exit profile

---

## Verification Checklist

After implementation:

- [x] SNIPER guard filters to EU/OVERLAP/US + LOW/MEDIUM/HIGH vol
- [x] SNIPER reuses FARM_V2B scoring/thresholding
- [x] SNIPER policy bundle is separate from FARM
- [x] SNIPER exit configs are separate from FARM
- [x] Router supports config-parametric `v3_range_edge_cutoff`
- [x] FARM lock check script exists
- [x] SNIPER backtest plan documented
- [x] SNIPER backtest report template created
- [ ] SNIPER replay run completed (30D)
- [ ] SNIPER live practice run completed
- [ ] Trade journal verification (all entries in EU/OVERLAP/US)
- [ ] FARM lock check passes

---

## Next Steps

1. **Run SNIPER 30D replay** to validate implementation
2. **Run SNIPER live practice** (1 trade) to verify plumbing
3. **Analyze backtest results** using report template
4. **Tune thresholds** based on Round 1 results
5. **Run FULLYEAR backtest** for comprehensive validation

---

*Last Updated: 2025-12-17*

