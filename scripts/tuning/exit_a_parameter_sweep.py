#!/usr/bin/env python3
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
import sys
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Parameter grid (can be reduced for faster sweeps)
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
    base_config_path = Path(base_config_path)
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")
    
    # Load base config
    with open(base_config_path) as f:
        config = yaml.safe_load(f)
    
    # Modify entry config
    entry_cfg_path = Path(config["entry_config"])
    if not entry_cfg_path.exists():
        raise FileNotFoundError(f"Entry config not found: {entry_cfg_path}")
    
    with open(entry_cfg_path) as f:
        entry_cfg = yaml.safe_load(f)
    
    entry_cfg["entry_v9_policy_farm_v2b"]["min_prob_long"] = params["min_prob_long"]
    
    # Save modified entry config
    variant_entry_cfg_path = Path(f"gx1/configs/policies/active/ENTRY_V9_FARM_V2B_{variant_name}.yaml")
    variant_entry_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(variant_entry_cfg_path, "w") as f:
        yaml.dump(entry_cfg, f, default_flow_style=False)
    
    # Modify exit config
    exit_cfg_path = Path(config["exit_config"])
    if not exit_cfg_path.exists():
        raise FileNotFoundError(f"Exit config not found: {exit_cfg_path}")
    
    with open(exit_cfg_path) as f:
        exit_cfg = yaml.safe_load(f)
    
    exit_cfg["exit"]["params"]["rule_a_profit_min_bps"] = params["rule_a_profit_min_bps"]
    exit_cfg["exit"]["params"]["rule_a_profit_max_bps"] = params["rule_a_profit_max_bps"]
    
    # Save modified exit config
    variant_exit_cfg_path = Path(f"gx1/configs/exits/FARM_EXIT_V2_RULES_A_{variant_name}.yaml")
    variant_exit_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(variant_exit_cfg_path, "w") as f:
        yaml.dump(exit_cfg, f, default_flow_style=False)
    
    # Create variant policy config
    variant_config = config.copy()
    variant_config["version"] = f"EXIT_A_SWEEP_{variant_name}"
    variant_config["entry_config"] = str(variant_entry_cfg_path)
    variant_config["exit_config"] = str(variant_exit_cfg_path)
    variant_config["trade_log_csv"] = f"gx1/wf_runs/EXIT_A_SWEEP_{variant_name}/trade_log.csv"
    variant_config["logging"]["log_dir"] = f"gx1/wf_runs/EXIT_A_SWEEP_{variant_name}/logs"
    
    variant_config_path = Path(f"gx1/configs/policies/active/EXIT_A_SWEEP_{variant_name}.yaml")
    variant_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(variant_config_path, "w") as f:
        yaml.dump(variant_config, f, default_flow_style=False)
    
    return str(variant_config_path)

def run_replay(config_path, variant_name):
    """Run replay for a config variant."""
    output_dir = Path(f"gx1/wf_runs/EXIT_A_SWEEP_{variant_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        return None
    
    cmd = [
        "bash", "scripts/active/run_replay.sh",
        str(config_path),
        START_DATE,
        END_DATE,
        "7"  # n_workers
    ]
    
    env = os.environ.copy()
    env["M5_DATA"] = M5_DATA
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=PROJECT_ROOT)
    
    if result.returncode != 0:
        print(f"ERROR: {variant_name} failed:")
        print(result.stderr)
        return None
    
    return str(output_dir)

def analyze_results(output_dir):
    """Analyze trade log and compute metrics."""
    trade_log_path = Path(output_dir) / "trade_log.csv"
    if not trade_log_path.exists():
        # Try alternative paths
        alt_paths = [
            Path(output_dir) / "trade_log_entry_v9_exit_v3_adaptive_3SEG_merged.csv",
            Path(output_dir) / "parallel_chunks" / "trade_log_chunk_0.csv",
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                trade_log_path = alt_path
                break
        
        if not trade_log_path.exists():
            print(f"WARNING: No trade log found in {output_dir}")
            return None
    
    try:
        df = pd.read_csv(trade_log_path)
    except Exception as e:
        print(f"ERROR: Failed to read trade log: {e}")
        return None
    
    # Filter closed trades only
    if "exit_time" not in df.columns:
        print(f"WARNING: exit_time column missing in {trade_log_path}")
        return None
    
    closed = df[df["exit_time"].notna()].copy()
    if len(closed) == 0:
        print(f"WARNING: No closed trades in {trade_log_path}")
        return None
    
    # Compute metrics
    n_trades = len(closed)
    
    if "pnl_bps" not in closed.columns:
        print(f"WARNING: pnl_bps column missing")
        return None
    
    win_rate = (closed["pnl_bps"] > 0).mean() * 100
    mean_pnl_bps = closed["pnl_bps"].mean()
    median_pnl_bps = closed["pnl_bps"].median()
    
    if "bars_held" in closed.columns:
        median_bars_held = closed["bars_held"].median()
    else:
        median_bars_held = 0.0
    
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
    
    if not Path(base_config).exists():
        print(f"ERROR: Base config not found: {base_config}")
        return
    
    results = []
    
    # Generate all combinations
    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())
    
    total_combos = len(list(product(*param_values)))
    print(f"Total variants to test: {total_combos}")
    print(f"Parameter grid: {PARAM_GRID}")
    
    for idx, combo in enumerate(product(*param_values), 1):
        params = dict(zip(param_names, combo))
        variant_name = "_".join([f"{k}_{v}" for k, v in params.items()]).replace(".", "p")
        
        print(f"\n=== [{idx}/{total_combos}] Running variant: {variant_name} ===")
        print(f"Params: {params}")
        
        try:
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
        except Exception as e:
            print(f"ERROR processing variant {variant_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(results) == 0:
        print("ERROR: No results collected!")
        return
    
    # Save results
    df_results = pd.DataFrame(results)
    results_path = Path("gx1/wf_runs/EXIT_A_SWEEP_RESULTS.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(results_path, index=False)
    
    # Generate markdown summary
    md_lines = [
        "# EXIT_A Parameter Sweep Results (Q1 2025)",
        "",
        f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Period**: {START_DATE} to {END_DATE}",
        f"**Total variants tested**: {len(results)}",
        "",
        "## Summary Table",
        "",
        df_results.to_markdown(index=False),
        "",
        "## Baseline Comparison",
        "",
        "- **Baseline (EXIT_A_Q1)**: 36 trades, 91.67% win rate, 55.96 bps/day",
        "",
        "## Best Variants",
        "",
        "### Highest EV/day:",
        "",
        df_results.nlargest(5, "ev_per_day")[["variant", "n_trades", "win_rate", "ev_per_day", "trades_per_day"]].to_markdown(index=False),
        "",
        "### Highest Win Rate:",
        "",
        df_results.nlargest(5, "win_rate")[["variant", "n_trades", "win_rate", "ev_per_day", "trades_per_day"]].to_markdown(index=False),
        "",
        "### Highest Trade Frequency (trades/day):",
        "",
        df_results.nlargest(5, "trades_per_day")[["variant", "n_trades", "win_rate", "ev_per_day", "trades_per_day"]].to_markdown(index=False),
        "",
        "### Best Balance (win_rate > 85%, trades_per_day > 1.0):",
        "",
    ]
    
    # Filter for balanced variants
    balanced = df_results[(df_results["win_rate"] > 85) & (df_results["trades_per_day"] > 1.0)]
    if len(balanced) > 0:
        md_lines.append(balanced.nlargest(5, "ev_per_day")[["variant", "n_trades", "win_rate", "ev_per_day", "trades_per_day"]].to_markdown(index=False))
    else:
        md_lines.append("*No variants meet criteria*")
    
    md_path = Path("gx1/wf_runs/EXIT_A_SWEEP_RESULTS.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    
    print("\n=== Sweep Complete ===")
    print(f"Results saved to: {results_path}")
    print(f"Summary saved to: {md_path}")
    print(f"\nTotal variants tested: {len(results)}")

if __name__ == "__main__":
    main()

