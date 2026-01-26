#!/usr/bin/env python3
"""
Conservative EXIT_A tuning sweep over Q1 2025.

Focuses on a curated set of variants (≤ 5) instead of a full grid:
- min_prob_long ∈ {0.72, 0.70, 0.68}
- max_open_trades ∈ {3, 4, 5}
- rule_a_profit_min_bps ∈ {6.0, 5.0, 4.0}
- cooldown/min_time_between_trades_sec ∈ {60, 0}

Outputs combined CSV + Markdown summary tables under gx1/tuning/.
"""

import pandas as pd
import subprocess
import json
from pathlib import Path
import yaml
import sys
import os
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Q1 2025 period / dataset
START_DATE = "2025-01-02"
END_DATE = "2025-03-31"
M5_DATA = "data/raw/xauusd_m5_2025_bid_ask.parquet"
N_WORKERS = "6"
RESULTS_DIR = PROJECT_ROOT / "gx1" / "tuning"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Curated variant list (baseline + variants)
VARIANTS = [
    {
        "name": "baseline",
        "min_prob_long": 0.72,
        "max_open_trades": 3,
        "rule_a_profit_min_bps": 6.0,
        "cooldown_sec": 60,
        "reuse_output": "gx1/wf_runs/FARM_V2B_EXIT_A_Q1",
    },
    {
        "name": "mp70_max4_rule5_cd60",
        "min_prob_long": 0.70,
        "max_open_trades": 4,
        "rule_a_profit_min_bps": 5.0,
        "cooldown_sec": 60,
        "reuse_output": "gx1/wf_runs/EXIT_A_SWEEP_mp70_max4_rule5_cd60",
    },
    {
        "name": "mp70_max4_rule4_cd0",
        "min_prob_long": 0.70,
        "max_open_trades": 4,
        "rule_a_profit_min_bps": 4.0,
        "cooldown_sec": 0,
        "reuse_output": "gx1/wf_runs/EXIT_A_SWEEP_mp70_max4_rule4_cd0",
    },
    {
        "name": "mp68_max5_rule5_cd60",
        "min_prob_long": 0.68,
        "max_open_trades": 5,
        "rule_a_profit_min_bps": 5.0,
        "cooldown_sec": 60,
        "reuse_output": "gx1/wf_runs/EXIT_A_SWEEP_mp68_max5_rule5_cd60",
    },
    {
        "name": "mp70_max5_rule4_cd0",
        "min_prob_long": 0.70,
        "max_open_trades": 5,
        "rule_a_profit_min_bps": 4.0,
        "cooldown_sec": 0,
        "reuse_output": "gx1/wf_runs/EXIT_A_SWEEP_mp70_max5_rule4_cd0",
    },
    {
        "name": "mp68_max5_rule5_cd0",
        "min_prob_long": 0.68,
        "max_open_trades": 5,
        "rule_a_profit_min_bps": 5.0,
        "cooldown_sec": 0,
        "reuse_output": "gx1/wf_runs/EXIT_A_SWEEP_mp68_max5_rule5_cd0",
    },
    {
        "name": "mp68_max5_rule4_cd60",
        "min_prob_long": 0.68,
        "max_open_trades": 5,
        "rule_a_profit_min_bps": 4.0,
        "cooldown_sec": 60,
    },
    {
        "name": "mp68_max4_rule4_cd0",
        "min_prob_long": 0.68,
        "max_open_trades": 4,
        "rule_a_profit_min_bps": 4.0,
        "cooldown_sec": 0,
    },
    {
        "name": "mp68_max5_rule4_cd0",
        "min_prob_long": 0.68,
        "max_open_trades": 5,
        "rule_a_profit_min_bps": 4.0,
        "cooldown_sec": 0,
        "reuse_output": "gx1/wf_runs/EXIT_A_SWEEP_mp68_max5_rule4_cd0",
    },
]

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
    variant_config.setdefault("logging", {})
    variant_config["logging"]["log_dir"] = f"gx1/wf_runs/EXIT_A_SWEEP_{variant_name}/logs"
    variant_config.setdefault("execution", {})
    variant_config["execution"]["max_open_trades"] = params["max_open_trades"]
    variant_config.setdefault("risk", {})
    variant_config["risk"]["min_time_between_trades_sec"] = params["cooldown_sec"]
    
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
        N_WORKERS  # n_workers
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

def _extract_exit_profile_series(df: pd.DataFrame) -> pd.Series | None:
    """Return exit_profile series from dataframe (parsing extra if needed)."""
    if "exit_profile" in df.columns:
        return df["exit_profile"]
    if "extra.exit_profile" in df.columns:
        return df["extra.exit_profile"]
    if "extra" not in df.columns:
        return None

    def _parse(extra):
        if isinstance(extra, dict):
            return extra.get("exit_profile")
        if isinstance(extra, str) and extra.strip():
            try:
                data = json.loads(extra)
            except Exception:
                return None
            if isinstance(data, dict):
                return data.get("exit_profile")
        return None

    return df["extra"].apply(_parse)

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
    
    total_trades = len(df)
    exit_profile_series = _extract_exit_profile_series(df)
    if exit_profile_series is None:
        missing_exit_profile = total_trades
    else:
        eps = exit_profile_series.astype("string")
        missing_exit_profile = int(eps.isna().sum() + (eps == "").sum())
    
    if "exit_time" not in df.columns:
        print(f"WARNING: exit_time column missing in {trade_log_path}")
        return None
    closed = df[df["exit_time"].notna()].copy()
    closed_trades = len(closed)
    if closed_trades == 0:
        print(f"WARNING: No closed trades in {trade_log_path}")
        return None
    
    pnl_series = pd.to_numeric(closed["pnl_bps"], errors="coerce")
    pnl_series = pnl_series.dropna()
    if pnl_series.empty:
        print(f"WARNING: No numeric pnl_bps values in {trade_log_path}")
        return None

    win_rate = float((pnl_series > 0).mean() * 100.0)
    mean_pnl_bps = float(pnl_series.mean())
    median_pnl_bps = float(pnl_series.median())
    days = max((pd.to_datetime(END_DATE) - pd.to_datetime(START_DATE)).days, 1)
    trades_per_day = closed_trades / days
    ev_per_day = float(pnl_series.sum() / days)
    ev_per_trade = mean_pnl_bps
    if "bars_held" in closed.columns:
        median_bars_held = float(pd.to_numeric(closed["bars_held"], errors="coerce").dropna().median())
    else:
        median_bars_held = 0.0
    
    return {
        "trades_total": total_trades,
        "trades_closed": closed_trades,
        "missing_exit_profile": missing_exit_profile,
        "trades_per_day": trades_per_day,
        "win_rate": win_rate,
        "ev_per_trade_bps": ev_per_trade,
        "ev_per_day_bps": ev_per_day,
        "median_pnl_bps": median_pnl_bps,
        "median_bars_held": median_bars_held,
    }

def main():
    """Run curated parameter sweep."""
    base_config = Path("gx1/configs/policies/active/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_A_Q1.yaml")
    if not base_config.exists():
        raise SystemExit(f"Base config not found: {base_config}")
    
    results = []
    for variant in VARIANTS:
        name = variant["name"]
        print(f"\n=== Variant: {name} ===")
        reuse_dir = variant.get("reuse_output")
        output_dir = None
        if reuse_dir:
            output_dir = reuse_dir
            print(f"Reusing existing output dir: {output_dir}")
        else:
            config_path = create_config_variant(base_config, name, variant)
            output_dir = run_replay(config_path, name)
            if output_dir is None:
                print(f"Skipping {name} (replay failed)")
                continue
        
        metrics = analyze_results(output_dir)
        if metrics is None:
            print(f"Skipping {name}: unable to compute metrics")
            continue
        
        metrics.update(
            {
                "variant": name,
                "min_prob_long": variant["min_prob_long"],
                "max_open_trades": variant["max_open_trades"],
                "rule_a_profit_min_bps": variant["rule_a_profit_min_bps"],
                "cooldown_sec": variant["cooldown_sec"],
                "output_dir": output_dir,
            }
        )
        results.append(metrics)
    
    if not results:
        raise SystemExit("No successful variants.")
    
    df = pd.DataFrame(results)
    df = df[
        [
            "variant",
            "min_prob_long",
            "max_open_trades",
            "rule_a_profit_min_bps",
            "cooldown_sec",
            "trades_total",
            "trades_closed",
            "missing_exit_profile",
            "trades_per_day",
            "win_rate",
            "ev_per_trade_bps",
            "ev_per_day_bps",
            "median_pnl_bps",
            "median_bars_held",
            "output_dir",
        ]
    ]
    
    csv_path = RESULTS_DIR / "exit_a_q1_sweep_results.csv"
    md_path = RESULTS_DIR / "exit_a_q1_sweep_results.md"
    df.to_csv(csv_path, index=False)
    
    md_lines = []
    md_lines.append(f"# EXIT_A Q1 Sweep ({datetime.utcnow().isoformat()}Z)\n")
    md_lines.append("| Variant | min_prob_long | max_open | rule_a_min | cooldown_s | trades | closed | miss_exit_profile | trades/day | win_rate% | EV/trade (bps) | EV/day (bps) |")
    md_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, row in df.iterrows():
        md_lines.append(
            "| {variant} | {mp:.2f} | {max_open} | {rule:.1f} | {cd:.0f} | {total} | {closed} | {miss} | {tpd:.2f} | {wr:.2f} | {evt:.2f} | {evd:.2f} |".format(
                variant=row["variant"],
                mp=row["min_prob_long"],
                max_open=int(row["max_open_trades"]),
                rule=row["rule_a_profit_min_bps"],
                cd=row["cooldown_sec"],
                total=int(row["trades_total"]),
                closed=int(row["trades_closed"]),
                miss=int(row["missing_exit_profile"]),
                tpd=row["trades_per_day"],
                wr=row["win_rate"],
                evt=row["ev_per_trade_bps"],
                evd=row["ev_per_day_bps"],
            )
        )
    
    eligible = df[(df["win_rate"] >= 80.0) & (df["trades_per_day"] >= 0.8)]
    eligible = eligible.sort_values("ev_per_day_bps", ascending=False).head(3)
    if not eligible.empty:
        md_lines.append("\n## Top Variants (win_rate ≥ 80%, trades/day ≥ 0.8, sorted by EV/day)\n")
        for _, row in eligible.iterrows():
            md_lines.append(
                f"- **{row['variant']}** → EV/day {row['ev_per_day_bps']:.2f} bps, trades/day {row['trades_per_day']:.2f}, win_rate {row['win_rate']:.2f}%, "
                f"min_prob_long {row['min_prob_long']:.2f}, max_open {int(row['max_open_trades'])}, rule_a_min {row['rule_a_profit_min_bps']:.1f}, cooldown {row['cooldown_sec']:.0f}s"
            )
    else:
        md_lines.append("\n_No variants met the EV/day filter with win_rate ≥ 80% and trades/day ≥ 0.8._\n")
    
    md_path.write_text("\n".join(md_lines))
    print(f"\nSaved results: {csv_path}, {md_path}")
    print(df)

if __name__ == "__main__":
    main()
