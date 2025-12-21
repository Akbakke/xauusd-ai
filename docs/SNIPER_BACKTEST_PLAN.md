# SNIPER Backtest Plan

**Status:** Planning  
**Purpose:** Session-aware backtests for SNIPER (EU/London/NY) without affecting FARM  
**Isolation:** SNIPER runs are completely separate from FARM runs

---

## Test Universe

**Instrument:** `XAU_USD`  
**Timeframe:** `M5`  
**Sessions:** `EU`, `OVERLAP`, `US` (UTC)  
**Excluded:** `ASIA` (FARM territory)

**Data Source:**
```
data/raw/xauusd_m5_2025_bid_ask.parquet
```

---

## Data Splits

### 1. Time-Based Splits

| Split | Duration | Purpose |
|-------|----------|---------|
| `SNIPER_OBS_30D` | Last 30 days | Quick validation |
| `SNIPER_OBS_90D` | Last 90 days | Medium-term validation |
| `SNIPER_OBS_FULLYEAR` | Full year 2025 | Comprehensive validation |

### 2. Quarterly Splits (Optional)

| Split | Period | Purpose |
|-------|--------|---------|
| `SNIPER_OBS_Q1` | Q1 2025 | Q1-specific patterns |
| `SNIPER_OBS_Q2` | Q2 2025 | Q2-specific patterns |
| `SNIPER_OBS_Q3` | Q3 2025 | Q3-specific patterns |
| `SNIPER_OBS_Q4` | Q4 2025 | Q4-specific patterns |

**Note:** Quarterly splits are optional - use if Q-specific tuning is needed.

---

## Run Tags

**Format:** `SNIPER_OBS_<SPLIT>_<YYYYMMDD_HHMMSS>`

**Examples:**
- `SNIPER_OBS_30D_20251217_120000`
- `SNIPER_OBS_90D_20251217_120000`
- `SNIPER_OBS_FULLYEAR_20251217_120000`

**Control Runs:**
- FARM runs remain unchanged (use existing `FULLYEAR`, `CANARY` tags)
- SNIPER runs are clearly tagged with `SNIPER_` prefix

---

## Policy Configuration

**SNIPER Policy Bundle:**
```
gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY.yaml
```

**Entry Config:**
```
gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/ENTRY_V9_SNIPER_LONDON_NY.yaml
```

**Exit Configs:**
```
gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/exits/SNIPER_EXIT_RULES_A.yaml
gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/exits/SNIPER_EXIT_RULES_ADAPTIVE.yaml
```

---

## Session Filtering

**Included Sessions (UTC):**
- **EU:** 07:00-16:00 UTC
- **OVERLAP:** 12:00-16:00 UTC (EU + US overlap)
- **US:** 12:00-21:00 UTC

**Excluded Sessions:**
- **ASIA:** 00:00-09:00 UTC (FARM territory)

**Implementation:**
- SNIPER guard (`sniper_guard_v1`) enforces session filtering
- Entries only allowed in EU/OVERLAP/US
- ASIA entries are blocked by guard

---

## Volatility Regimes

**Allowed Regimes:**
- `LOW` (ATR regime ID: 0)
- `MEDIUM` (ATR regime ID: 1)
- `HIGH` (ATR regime ID: 2) ← **KEY DIFFERENCE from FARM**

**Excluded:**
- `EXTREME` (ATR regime ID: 3) - not allowed in v1

**Implementation:**
- SNIPER guard enforces vol regime filtering
- `allow_high_vol: true` in SNIPER config

---

## Replay Commands

### 1. 30-Day Backtest

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

### 2. 90-Day Backtest

```bash
# Same as above, but change days=30 to days=90
# Run tag: SNIPER_OBS_90D_$(date +%Y%m%d_%H%M%S)
```

### 3. FULLYEAR Backtest

```bash
# Same as above, but use full year data
# Run tag: SNIPER_OBS_FULLYEAR_$(date +%Y%m%d_%H%M%S)
```

---

## Verification Checklist

After each backtest run:

- [ ] `run_header.json` exists and contains `meta.role: SNIPER_CANARY`
- [ ] Trade journal exists with entries
- [ ] All entries are in EU/OVERLAP/US sessions (verify via trade journal)
- [ ] No entries in ASIA session (verify via trade journal)
- [ ] Vol regimes include HIGH (verify via trade journal)
- [ ] Feature manifest matches (no mismatch errors)
- [ ] Router decisions logged (router_explainability in trade journal)

---

## Comparison with FARM

**FARM Baseline (for comparison):**
- Sessions: ASIA only
- Vol regimes: LOW, MEDIUM only
- Threshold: 0.68
- Expected trade rate: ~3 trades/day (ASIA)

**SNIPER (expected):**
- Sessions: EU, OVERLAP, US
- Vol regimes: LOW, MEDIUM, HIGH
- Threshold: 0.67
- Expected trade rate: 2-3x FARM (6-9 trades/day)

**Note:** SNIPER runs are independent - no need to compare directly with FARM unless analyzing session-specific performance.

---

*Last Updated: 2025-12-17*

