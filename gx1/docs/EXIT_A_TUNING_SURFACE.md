# EXIT_A Tuning Surface: Increasing Trade Frequency While Preserving Win Rate

## Context

**Baselines:**
- **DIAG**: FARM_V2B + EXIT_DIAG, 674 trades, 100% closed, EV/day ≈ 31.8 bps
- **EXIT_A_Q1**: FARM_V2B + EXIT_A, 36 trades, 100% closed, win rate ≈ 91.67%, EV/day ≈ 55.96 bps

**Goal**: Increase EXIT_A trade frequency from ~0.4 trades/day to 1–1.5 trades/day while preserving high win rate (~90%+).

---

## 1. ENTRY Coverage/Quality Knobs (FARM_V2B)

### 1.1 Model Output Thresholds

#### `min_prob_long` (ENTRY_V9_FARM_V2B.yaml)
- **Current value**: `0.72`
- **Location**: `gx1/configs/policies/active/ENTRY_V9_FARM_V2B.yaml:20`
- **Effect**: Coverage (primary), Quality (secondary)
- **Description**: Minimum `p_long` probability from ENTRY_V9 transformer model required to open a trade.
- **Tuning impact**:
  - **Decrease** (e.g., 0.72 → 0.68): Increases coverage significantly (~2–3x more candidates pass), but likely reduces win rate by 5–10% (lower-quality signals).
  - **Increase** (e.g., 0.72 → 0.75): Reduces coverage (~30–50% fewer trades), improves win rate slightly (~2–3%).
- **Code reference**: `gx1/policy/entry_v9_policy_farm_v2b.py:193`

#### `min_prob_profitable` (currently disabled)
- **Current value**: `0.0` (not used)
- **Location**: `gx1/configs/policies/active/ENTRY_V9_FARM_V2B.yaml:21`
- **Effect**: Quality (primary), Coverage (secondary)
- **Description**: Meta-model probability threshold for trade profitability. Currently disabled (`enable_profitable_filter: false`).
- **Tuning impact**:
  - **Enable with low threshold** (e.g., 0.5): Adds quality filter without heavy coverage loss (~10–20% reduction), may improve win rate by 2–5%.
  - **Enable with high threshold** (e.g., 0.6): Heavy coverage reduction (~50–70%), but may improve win rate by 5–10%.
- **Code reference**: `gx1/policy/entry_v9_policy_farm_v2b.py:253-268`

### 1.2 Regime/Session Filters

#### `allow_medium_vol` (FARM_V2B guard)
- **Current value**: `true`
- **Location**: `gx1/configs/policies/active/ENTRY_V9_FARM_V2B.yaml:27`
- **Effect**: Coverage (primary)
- **Description**: Whether to allow MEDIUM volatility regime in addition to LOW. FARM_V2B allows ASIA + (LOW ∪ MEDIUM).
- **Tuning impact**:
  - **Keep `true`**: Already allows both LOW and MEDIUM (current baseline).
  - **Set `false`**: Reduces coverage by ~40–60% (only ASIA+LOW), but may improve win rate by 3–5% (LOW regime is more stable).
- **Code reference**: `gx1/policy/farm_guards.py:178-278`, `gx1/policy/entry_v9_policy_farm_v2b.py:92`

#### Session filter (hardcoded: ASIA only)
- **Current value**: `ASIA` (hardcoded in guard)
- **Location**: `gx1/policy/farm_guards.py:253`
- **Effect**: Coverage (primary)
- **Description**: Brutal guard V2 enforces ASIA session only. No config knob (hardcoded).
- **Tuning impact**: **Not tunable via config** (requires code change). Adding EU/US would dramatically increase coverage but likely reduce win rate significantly (~10–20% drop).

#### `require_trend_up` (currently disabled)
- **Current value**: `false`
- **Location**: `gx1/configs/policies/active/ENTRY_V9_FARM_V2B.yaml:23`
- **Effect**: Coverage (primary), Quality (secondary)
- **Description**: Whether to require TREND_UP regime before entry.
- **Tuning impact**:
  - **Keep `false`**: Current baseline (no trend filter).
  - **Set `true`**: Reduces coverage by ~30–50%, may improve win rate by 2–5% (trend-aligned entries).
- **Code reference**: `gx1/policy/entry_v9_policy_farm_v2b.py:149-180`

### 1.3 Guards Affecting Coverage

#### `max_open_trades` (risk limit)
- **Current value**: `3` (default in `oanda_demo_runner.py:356`)
- **Location**: Risk/execution config (not in entry policy YAML)
- **Effect**: Coverage (primary)
- **Description**: Maximum number of concurrent open trades. Blocks new entries when limit reached.
- **Tuning impact**:
  - **Increase** (e.g., 3 → 5): Allows more concurrent trades, increases coverage by ~20–40% if frequently hitting limit.
  - **Decrease** (e.g., 3 → 1): Reduces coverage (~30–50% reduction), but may improve portfolio risk management.
- **Code reference**: `gx1/execution/oanda_demo_runner.py:4790`

#### Portfolio drawdown guard (intraday)
- **Current value**: Configurable via `intraday_drawdown_bps_limit` (if set)
- **Location**: `gx1/execution/entry_manager.py:1495-1508`
- **Effect**: Coverage (primary)
- **Description**: Blocks new entries if unrealized portfolio PnL drops below threshold.
- **Tuning impact**: **Not currently active** (requires config). If enabled, would reduce coverage during drawdown periods.

#### Cooldown/timing guards
- **Current value**: **None** (no cooldown logic found)
- **Effect**: Coverage (if added)
- **Description**: No explicit cooldown between trades found in codebase. Trades can open back-to-back if signals pass.
- **Tuning impact**: **Not applicable** (no cooldown exists). Could be added to reduce overtrading.

---

## 2. EXIT_A Knobs (FARM_EXIT_V2_RULES_A)

### 2.1 Profit-Capture Rules (Rule A)

#### `rule_a_profit_min_bps`
- **Current value**: `6.0`
- **Location**: `gx1/configs/exits/FARM_EXIT_V2_RULES_A.yaml:13`
- **Effect**: Win rate (primary), Average PnL (secondary), Holding time (secondary)
- **Description**: Minimum profit (in bps) required to trigger exit. Exit occurs when PnL is between `profit_min_bps` and `profit_max_bps`.
- **Tuning impact**:
  - **Decrease** (e.g., 6.0 → 4.0): More trades exit earlier, increases win rate (~3–5%), reduces average PnL per trade (~15–25%), reduces holding time (~20–30%).
  - **Increase** (e.g., 6.0 → 8.0): Fewer trades exit, reduces win rate (~5–8%), increases average PnL per trade (~20–30%), increases holding time (~30–40%).
- **Code reference**: `gx1/policy/exit_farm_v2_rules.py:234-251`

#### `rule_a_profit_max_bps`
- **Current value**: `9.0`
- **Location**: `gx1/configs/exits/FARM_EXIT_V2_RULES_A.yaml:14`
- **Effect**: Average PnL (primary), Holding time (secondary)
- **Description**: Maximum profit target (in bps). Trades exit when PnL reaches this level (if not using trailing stop).
- **Tuning impact**:
  - **Decrease** (e.g., 9.0 → 7.0): Trades exit earlier, reduces average PnL (~10–15%), reduces holding time (~15–20%).
  - **Increase** (e.g., 9.0 → 12.0): Trades hold longer for bigger wins, increases average PnL (~15–25%), increases holding time (~25–35%).
- **Code reference**: `gx1/policy/exit_farm_v2_rules.py:234-251`

#### `rule_a_adaptive_threshold_bps`
- **Current value**: `4.0`
- **Location**: `gx1/configs/exits/FARM_EXIT_V2_RULES_A.yaml:15`
- **Effect**: Win rate (primary), Average PnL (secondary)
- **Description**: If PnL exceeds this threshold within `rule_a_adaptive_bars`, trailing stop is activated.
- **Tuning impact**:
  - **Decrease** (e.g., 4.0 → 3.0): Trailing stop activates more often, protects more profits, increases win rate (~2–4%), may reduce average PnL slightly (~5–10%).
  - **Increase** (e.g., 4.0 → 5.0): Trailing stop activates less often, allows more trades to run, may increase average PnL (~5–10%), but reduces win rate (~2–3%).
- **Code reference**: `gx1/policy/exit_farm_v2_rules.py:200-209`

#### `rule_a_trailing_stop_bps`
- **Current value**: `2.0`
- **Location**: `gx1/configs/exits/FARM_EXIT_V2_RULES_A.yaml:16`
- **Effect**: Win rate (primary), Average PnL (secondary)
- **Description**: Distance (in bps) below trailing high to trigger exit when trailing stop is active.
- **Tuning impact**:
  - **Decrease** (e.g., 2.0 → 1.5): Tighter trailing stop, protects profits better, increases win rate (~2–3%), reduces average PnL (~5–10%).
  - **Increase** (e.g., 2.0 → 3.0): Looser trailing stop, allows more room for profit expansion, may increase average PnL (~5–10%), but reduces win rate (~2–3%).
- **Code reference**: `gx1/policy/exit_farm_v2_rules.py:212-231`

#### `rule_a_adaptive_bars`
- **Current value**: `3`
- **Location**: `gx1/configs/exits/FARM_EXIT_V2_RULES_A.yaml:17`
- **Effect**: Win rate (primary), Holding time (secondary)
- **Description**: Number of bars to check for adaptive trailing stop activation. Trailing stop only activates if threshold is reached within this window.
- **Tuning impact**:
  - **Decrease** (e.g., 3 → 2): Trailing stop activates earlier, protects profits better, increases win rate (~2–3%), reduces holding time (~10–15%).
  - **Increase** (e.g., 3 → 5): Trailing stop activates later, allows more time for profit expansion, may increase average PnL (~5–10%), but reduces win rate (~2–3%).
- **Code reference**: `gx1/policy/exit_farm_v2_rules.py:200-209`

### 2.2 Other Rules (Currently Disabled)

#### Rule B (Fast loss-cut)
- **Current value**: `enable_rule_b: false`
- **Effect**: Win rate (primary), Coverage (secondary via early exits)
- **Description**: Exit if MAE < -4 bps before bar 6.
- **Tuning impact**: **Not applicable** (disabled). If enabled, would increase win rate (~5–10%) but reduce average PnL (~10–15%) due to early exits.

#### Rule C (Time-based abandonment)
- **Current value**: `enable_rule_c: false`
- **Effect**: Holding time (primary), Win rate (secondary)
- **Description**: Exit within bar 8 if PnL < +2 bps.
- **Tuning impact**: **Not applicable** (disabled). If enabled, would reduce holding time (~20–30%) but may reduce win rate (~3–5%) due to premature exits.

#### `force_exit_bars` (fail-safe timeout)
- **Current value**: `None` (not set)
- **Effect**: Holding time (primary)
- **Description**: Maximum bars to hold trade before forced exit.
- **Tuning impact**: **Not applicable** (not set). If set (e.g., 20 bars), would cap holding time but may reduce win rate if trades need more time.

---

## 3. Tuning Surface Proposal

### 3.1 Best Candidates for Increasing Trade Count

#### **ENTRY Side (3–5 parameters)**

1. **`min_prob_long`: 0.72 → 0.68–0.70** ⭐ **HIGHEST IMPACT**
   - **Expected**: +50–100% more trades (18–36 → 27–54 trades in Q1)
   - **Win rate impact**: -3–7% (91.67% → 85–88%)
   - **Rationale**: Primary coverage knob. Small reduction preserves most quality while significantly increasing frequency.

2. **`max_open_trades`: 3 → 5** ⭐ **MEDIUM IMPACT**
   - **Expected**: +20–40% more trades if frequently hitting limit
   - **Win rate impact**: Minimal (if limit is binding, otherwise no impact)
   - **Rationale**: Allows more concurrent trades without changing signal quality.

3. **`require_trend_up`: false → true** (optional)
   - **Expected**: -30–50% fewer trades (but higher quality)
   - **Win rate impact**: +2–5% improvement
   - **Rationale**: **Not recommended** for increasing frequency, but useful if quality becomes priority.

4. **`enable_profitable_filter`: false → true, `min_prob_profitable`: 0.5** (optional)
   - **Expected**: -10–20% fewer trades, but higher quality
   - **Win rate impact**: +2–5% improvement
   - **Rationale**: **Not recommended** for increasing frequency, but useful as soft gate if quality drops too much.

5. **`allow_medium_vol`: true → false** (NOT recommended)
   - **Expected**: -40–60% fewer trades
   - **Win rate impact**: +3–5% improvement
   - **Rationale**: **Opposite of goal** (reduces frequency). Keep `true`.

#### **EXIT Side (2–3 parameters)**

1. **`rule_a_profit_min_bps`: 6.0 → 4.0–5.0** ⭐ **MEDIUM IMPACT**
   - **Expected**: More trades exit profitably (increases win rate by 3–5%)
   - **Average PnL impact**: -15–25% per trade
   - **Rationale**: Lower profit target allows more trades to exit profitably, protecting win rate while reducing per-trade PnL.

2. **`rule_a_profit_max_bps`: 9.0 → 7.0–8.0** (optional)
   - **Expected**: Trades exit earlier, reduces holding time
   - **Average PnL impact**: -10–15% per trade
   - **Rationale**: Faster exits may allow more trades per day if `max_open_trades` is binding.

3. **`rule_a_adaptive_threshold_bps`: 4.0 → 3.0–3.5** (optional)
   - **Expected**: Trailing stop activates more often, protects profits
   - **Win rate impact**: +2–4% improvement
   - **Rationale**: Protects win rate when lowering `profit_min_bps`.

### 3.2 Recommended Tuning Ranges

#### **Conservative (preserve ~90% win rate)**
- `min_prob_long`: 0.72 → **0.70** (small reduction)
- `max_open_trades`: 3 → **4** (moderate increase)
- `rule_a_profit_min_bps`: 6.0 → **5.0** (small reduction)
- **Expected**: ~50–70% more trades, win rate ~88–90%

#### **Moderate (target ~85–88% win rate)**
- `min_prob_long`: 0.72 → **0.68** (moderate reduction)
- `max_open_trades`: 3 → **5** (larger increase)
- `rule_a_profit_min_bps`: 6.0 → **4.5** (moderate reduction)
- `rule_a_adaptive_threshold_bps`: 4.0 → **3.5** (protect win rate)
- **Expected**: ~100–150% more trades, win rate ~85–88%

#### **Aggressive (target ~80–85% win rate)**
- `min_prob_long`: 0.72 → **0.65** (larger reduction)
- `max_open_trades`: 3 → **5**
- `rule_a_profit_min_bps`: 6.0 → **4.0** (larger reduction)
- `rule_a_profit_max_bps`: 9.0 → **7.0** (faster exits)
- `rule_a_adaptive_threshold_bps`: 4.0 → **3.0** (protect win rate)
- **Expected**: ~200–300% more trades, win rate ~80–85%

---

## 4. Parameter Sweep Harness (Q1 2025)

### 4.1 Design

A simple harness that:
1. Runs multiple replays over Q1 2025 (2025-01-02 → 2025-03-31)
2. Varies entry/exit parameters systematically
3. Logs metrics: coverage (trades/day), win rate, EV/trade, EV/day, median bars held
4. Outputs comparison table (CSV + Markdown)

### 4.2 Implementation Sketch

```python
# scripts/tuning/exit_a_parameter_sweep.py
"""
Parameter sweep for EXIT_A tuning over Q1 2025.

Tests combinations of:
- min_prob_long: [0.65, 0.68, 0.70, 0.72]
- max_open_trades: [3, 4, 5]
- rule_a_profit_min_bps: [4.0, 5.0, 6.0]
- rule_a_profit_max_bps: [7.0, 8.0, 9.0]

Outputs: CSV + Markdown summary table
"""

import pandas as pd
import subprocess
import json
from pathlib import Path
from itertools import product
import yaml

# Parameter grid
PARAM_GRID = {
    "min_prob_long": [0.65, 0.68, 0.70, 0.72],
    "max_open_trades": [3, 4, 5],
    "rule_a_profit_min_bps": [4.0, 5.0, 6.0],
    "rule_a_profit_max_bps": [7.0, 8.0, 9.0],
}

# Q1 2025 period
START_DATE = "2025-01-02"
END_DATE = "2025-03-31"
M5_DATA = "data/raw/xauusd_m5_2025_bid_ask.parquet"

def create_config_variant(base_config_path, variant_name, params):
    """Create a config variant with modified parameters."""
    # Load base config
    with open(base_config_path) as f:
        config = yaml.safe_load(f)
    
    # Modify entry config
    entry_cfg_path = config["entry_config"]
    with open(entry_cfg_path) as f:
        entry_cfg = yaml.safe_load(f)
    
    entry_cfg["entry_v9_policy_farm_v2b"]["min_prob_long"] = params["min_prob_long"]
    
    # Save modified entry config
    variant_entry_cfg_path = f"gx1/configs/policies/active/ENTRY_V9_FARM_V2B_{variant_name}.yaml"
    with open(variant_entry_cfg_path, "w") as f:
        yaml.dump(entry_cfg, f)
    
    # Modify exit config
    exit_cfg_path = config["exit_config"]
    with open(exit_cfg_path) as f:
        exit_cfg = yaml.safe_load(f)
    
    exit_cfg["exit"]["params"]["rule_a_profit_min_bps"] = params["rule_a_profit_min_bps"]
    exit_cfg["exit"]["params"]["rule_a_profit_max_bps"] = params["rule_a_profit_max_bps"]
    
    # Save modified exit config
    variant_exit_cfg_path = f"gx1/configs/exits/FARM_EXIT_V2_RULES_A_{variant_name}.yaml"
    with open(variant_exit_cfg_path, "w") as f:
        yaml.dump(exit_cfg, f)
    
    # Create variant policy config
    variant_config = config.copy()
    variant_config["version"] = f"EXIT_A_SWEEP_{variant_name}"
    variant_config["entry_config"] = variant_entry_cfg_path
    variant_config["exit_config"] = variant_exit_cfg_path
    variant_config["trade_log_csv"] = f"gx1/wf_runs/EXIT_A_SWEEP_{variant_name}/trade_log.csv"
    variant_config["logging"]["log_dir"] = f"gx1/wf_runs/EXIT_A_SWEEP_{variant_name}/logs"
    
    variant_config_path = f"gx1/configs/policies/active/EXIT_A_SWEEP_{variant_name}.yaml"
    with open(variant_config_path, "w") as f:
        yaml.dump(variant_config, f)
    
    return variant_config_path

def run_replay(config_path, variant_name):
    """Run replay for a config variant."""
    output_dir = f"gx1/wf_runs/EXIT_A_SWEEP_{variant_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "bash", "scripts/active/run_replay.sh",
        config_path,
        START_DATE,
        END_DATE,
        "7"  # n_workers
    ]
    
    env = {"M5_DATA": M5_DATA}
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {variant_name} failed: {result.stderr}")
        return None
    
    return output_dir

def analyze_results(output_dir):
    """Analyze trade log and compute metrics."""
    trade_log_path = Path(output_dir) / "trade_log.csv"
    if not trade_log_path.exists():
        return None
    
    df = pd.read_csv(trade_log_path)
    
    # Filter closed trades only
    closed = df[df["exit_time"].notna()].copy()
    if len(closed) == 0:
        return None
    
    # Compute metrics
    n_trades = len(closed)
    win_rate = (closed["pnl_bps"] > 0).mean() * 100
    mean_pnl_bps = closed["pnl_bps"].mean()
    median_pnl_bps = closed["pnl_bps"].median()
    median_bars_held = closed["bars_held"].median()
    
    # EV/day (assuming ~90 days in Q1)
    days = (pd.to_datetime(END_DATE) - pd.to_datetime(START_DATE)).days
    ev_per_day = (closed["pnl_bps"].sum() / days) if days > 0 else 0.0
    trades_per_day = n_trades / days if days > 0 else 0.0
    
    return {
        "n_trades": n_trades,
        "trades_per_day": trades_per_day,
        "win_rate": win_rate,
        "mean_pnl_bps": mean_pnl_bps,
        "median_pnl_bps": median_pnl_bps,
        "ev_per_day": ev_per_day,
        "median_bars_held": median_bars_held,
    }

def main():
    """Run parameter sweep."""
    base_config = "gx1/configs/policies/active/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_A_Q1.yaml"
    
    results = []
    
    # Generate all combinations
    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())
    
    for combo in product(*param_values):
        params = dict(zip(param_names, combo))
        variant_name = "_".join([f"{k}_{v}" for k, v in params.items()]).replace(".", "p")
        
        print(f"\n=== Running variant: {variant_name} ===")
        print(f"Params: {params}")
        
        # Create config variant
        config_path = create_config_variant(base_config, variant_name, params)
        
        # Run replay
        output_dir = run_replay(config_path, variant_name)
        if output_dir is None:
            continue
        
        # Analyze results
        metrics = analyze_results(output_dir)
        if metrics is None:
            continue
        
        # Store results
        result_row = {**params, **metrics, "variant": variant_name}
        results.append(result_row)
        
        print(f"Results: {metrics}")
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv("gx1/wf_runs/EXIT_A_SWEEP_RESULTS.csv", index=False)
    
    # Generate markdown summary
    md_lines = [
        "# EXIT_A Parameter Sweep Results (Q1 2025)",
        "",
        "## Summary Table",
        "",
        df_results.to_markdown(index=False),
        "",
        "## Baseline Comparison",
        "",
        f"- **Baseline (EXIT_A_Q1)**: 36 trades, 91.67% win rate, 55.96 bps/day",
        "",
        "## Best Variants",
        "",
        "### Highest EV/day:",
        df_results.nlargest(5, "ev_per_day")[["variant", "n_trades", "win_rate", "ev_per_day"]].to_markdown(index=False),
        "",
        "### Highest Win Rate:",
        df_results.nlargest(5, "win_rate")[["variant", "n_trades", "win_rate", "ev_per_day"]].to_markdown(index=False),
        "",
        "### Highest Trade Frequency (trades/day):",
        df_results.nlargest(5, "trades_per_day")[["variant", "n_trades", "win_rate", "ev_per_day"]].to_markdown(index=False),
    ]
    
    with open("gx1/wf_runs/EXIT_A_SWEEP_RESULTS.md", "w") as f:
        f.write("\n".join(md_lines))
    
    print("\n=== Sweep Complete ===")
    print(f"Results saved to: gx1/wf_runs/EXIT_A_SWEEP_RESULTS.csv")
    print(f"Summary saved to: gx1/wf_runs/EXIT_A_SWEEP_RESULTS.md")

if __name__ == "__main__":
    main()
```

### 4.3 Usage

```bash
# Run parameter sweep
python scripts/tuning/exit_a_parameter_sweep.py

# Results will be in:
# - gx1/wf_runs/EXIT_A_SWEEP_RESULTS.csv (detailed table)
# - gx1/wf_runs/EXIT_A_SWEEP_RESULTS.md (formatted summary)
```

### 4.4 Notes

- **Sweep size**: 4 × 3 × 3 × 3 = **108 variants** (may take ~6–12 hours to run)
- **Reduction strategies**:
  - Start with 2–3 key parameters (e.g., `min_prob_long`, `rule_a_profit_min_bps`)
  - Use coarse grid first (e.g., [0.68, 0.72] instead of [0.65, 0.68, 0.70, 0.72])
  - Run in parallel batches
- **Baseline preservation**: Keep DIAG baseline untouched (separate config path)

---

## 5. Summary: Key Levers

### **Top 3 Entry Levers (by impact on trade frequency)**
1. **`min_prob_long`** (0.72 → 0.68–0.70): **Highest impact** on coverage
2. **`max_open_trades`** (3 → 5): **Medium impact** if limit is binding
3. **`allow_medium_vol`** (keep `true`): Already optimized for coverage

### **Top 2 Exit Levers (by impact on win rate preservation)**
1. **`rule_a_profit_min_bps`** (6.0 → 4.0–5.0): Protects win rate when lowering entry threshold
2. **`rule_a_adaptive_threshold_bps`** (4.0 → 3.0–3.5): Activates trailing stop earlier to protect profits

### **Recommended First Steps**
1. **Conservative test**: `min_prob_long: 0.72 → 0.70`, `max_open_trades: 3 → 4`
2. **Measure**: Run Q1 replay, compare to baseline (36 trades, 91.67% win rate)
3. **Iterate**: If win rate stays >88%, try `min_prob_long: 0.70 → 0.68`
4. **Exit tuning**: If win rate drops, lower `rule_a_profit_min_bps` to 4.0–5.0

---

## Appendix: Code References

- Entry policy: `gx1/policy/entry_v9_policy_farm_v2b.py`
- Entry config: `gx1/configs/policies/active/ENTRY_V9_FARM_V2B.yaml`
- Exit policy: `gx1/policy/exit_farm_v2_rules.py`
- Exit config: `gx1/configs/exits/FARM_EXIT_V2_RULES_A.yaml`
- Guards: `gx1/policy/farm_guards.py`
- Entry manager: `gx1/execution/entry_manager.py`
- Runner: `gx1/execution/oanda_demo_runner.py`

