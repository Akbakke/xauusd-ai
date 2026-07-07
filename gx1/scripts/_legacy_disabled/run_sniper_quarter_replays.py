#!/usr/bin/env python3
"""
SNIPER Quarter Replays (Q2-Q4 2025) with Parallel Workers

Runs SNIPER replay for specified quarters, generates metrics, and creates OOS summary.
"""

import argparse
import json
import logging
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Quarter date ranges (UTC, exclusive end)
QUARTER_RANGES = {
    "Q1": ("2025-01-01T00:00:00Z", "2025-04-01T00:00:00Z"),
    "Q2": ("2025-04-01T00:00:00Z", "2025-07-01T00:00:00Z"),
    "Q3": ("2025-07-01T00:00:00Z", "2025-10-01T00:00:00Z"),
    "Q4": ("2025-10-01T00:00:00Z", "2026-01-01T00:00:00Z"),
}


def build_variant_policy(policy_path: Path, variant: str, tmp_dir: Path) -> Path:
    """
    Build variant policy overlay (runtime-only, does not modify snapshot).
    
    Args:
        policy_path: Original policy YAML path
        variant: "baseline" or "guarded"
        tmp_dir: Temporary directory for variant files
    
    Returns:
        Path to variant policy YAML
    """
    valid_base_variants = ["baseline", "guarded"]
    valid_variants = valid_base_variants + [
        "baseline_overlay", "guarded_overlay", 
        "baseline_no_cchop",
        "baseline_atrend_gate", "baseline_no_atrend_gate",
        "atrend_scale", "atrend_disable",
    ]
    if variant not in valid_variants:
        raise ValueError(
            f"Invalid variant: {variant}. Must be one of {', '.join(valid_variants)}"
        )
    
    # Determine base variant (baseline/guarded) and whether overlay is enabled
    if variant in ["atrend_scale", "atrend_disable"]:
        # Standalone A_TREND variants - use baseline as base
        base_variant = "baseline"
        overlay_enabled = False  # Not the general overlay flag
    elif variant.endswith("_no_atrend_gate"):
        base_variant = variant.replace("_no_atrend_gate", "")
        overlay_enabled = False  # A_TREND overlay disabled
    elif variant.endswith("_atrend_gate"):
        base_variant = variant.replace("_atrend_gate", "")
        overlay_enabled = False  # Not the general overlay flag, but A_TREND overlay will be enabled
    elif variant.endswith("_no_cchop"):
        base_variant = variant.replace("_no_cchop", "")
        overlay_enabled = False  # C_CHOP overlay disabled, but other overlays may be enabled
    elif variant.endswith("_overlay"):
        base_variant = variant.replace("_overlay", "")
        overlay_enabled = True
    else:
        base_variant = variant
        overlay_enabled = False
    
    if base_variant not in valid_base_variants:
        raise ValueError(
            f"Invalid base variant derived from {variant}: {base_variant}. "
            f"Must be one of {', '.join(valid_base_variants)}"
        )
    
    # Read original policy
    with open(policy_path) as f:
        policy_dict = yaml.safe_load(f)
    
    # Create variant dict (copy to avoid modifying original)
    variant_dict = yaml.safe_load(yaml.dump(policy_dict))  # Deep copy via YAML round-trip
    
    # Set meta.role
    if "meta" not in variant_dict:
        variant_dict["meta"] = {}
    variant_dict["meta"]["role"] = f"SNIPER_REPLAY_{variant.upper()}"
    
    # Set execution.max_open_trades = 1 for fair comparison
    if "execution" not in variant_dict:
        variant_dict["execution"] = {}
    variant_dict["execution"]["max_open_trades"] = 1
    
    # Handle risk_guard based on base variant (baseline vs guarded)
    if base_variant == "baseline":
        # For baseline, we want to disable guard but keep the structure
        # We'll modify the guard YAML to have enabled: false
        if "risk_guard" in variant_dict and variant_dict["risk_guard"]:
            guard_config_path = variant_dict["risk_guard"].get("config_path")
            if guard_config_path:
                # Resolve guard config path
                if not Path(guard_config_path).is_absolute():
                    project_root = Path(__file__).parent.parent.parent
                    guard_config_path = project_root / guard_config_path
                
                if guard_config_path.exists():
                    # Read guard config
                    with open(guard_config_path) as f:
                        guard_dict = yaml.safe_load(f)
                    
                    # Disable guard
                    if "sniper_risk_guard_v1" in guard_dict:
                        guard_dict["sniper_risk_guard_v1"]["enabled"] = False
                    else:
                        raise ValueError("Guard config missing 'sniper_risk_guard_v1' section")
                    
                    # Write variant guard config to tmp (disabled)
                    guard_stem = Path(guard_config_path).stem
                    variant_guard_path = tmp_dir / f"{guard_stem}__{variant}.yaml"
                    with open(variant_guard_path, "w") as f:
                        yaml.dump(guard_dict, f, default_flow_style=False, sort_keys=False)
                    
                    # Update policy to point to variant guard config
                    variant_dict["risk_guard"]["config_path"] = str(variant_guard_path)
    
    elif base_variant == "guarded":
        # For guarded, ensure guard is enabled
        if "risk_guard" not in variant_dict or variant_dict["risk_guard"] is None:
            raise ValueError("Cannot create 'guarded' variant: risk_guard not found in policy")
        
        guard_config_path = variant_dict["risk_guard"].get("config_path")
        if guard_config_path:
            # Resolve guard config path
            if not Path(guard_config_path).is_absolute():
                project_root = Path(__file__).parent.parent.parent
                guard_config_path = project_root / guard_config_path
            
            if guard_config_path.exists():
                # Read guard config
                with open(guard_config_path) as f:
                    guard_dict = yaml.safe_load(f)
                
                # Ensure guard is enabled
                if "sniper_risk_guard_v1" in guard_dict:
                    guard_dict["sniper_risk_guard_v1"]["enabled"] = True
                else:
                    raise ValueError("Guard config missing 'sniper_risk_guard_v1' section")
                
                # Write variant guard config to tmp
                guard_stem = Path(guard_config_path).stem
                variant_guard_path = tmp_dir / f"{guard_stem}__{variant}.yaml"
                with open(variant_guard_path, "w") as f:
                    yaml.dump(guard_dict, f, default_flow_style=False, sort_keys=False)
                
                # Update policy to point to variant guard config
                variant_dict["risk_guard"]["config_path"] = str(variant_guard_path)
            else:
                raise ValueError(f"Guard config not found: {guard_config_path}")
    
    # Apply SNIPER regime size overlay config for overlay variants (runtime-only)
    if overlay_enabled:
        overlay_cfg = variant_dict.get("sniper_regime_size_overlay") or {}
        overlay_cfg["enabled"] = True
        # Default multiplier for Q4 × B_MIXED size gate
        overlay_cfg.setdefault("q4_b_mixed_multiplier", 0.30)
        variant_dict["sniper_regime_size_overlay"] = overlay_cfg
    else:
        # Ensure overlay is disabled unless explicitly enabled via variant
        overlay_cfg = variant_dict.get("sniper_regime_size_overlay") or {}
        overlay_cfg["enabled"] = bool(overlay_cfg.get("enabled", False))
        variant_dict["sniper_regime_size_overlay"] = overlay_cfg
    
    # Handle Q4 C_CHOP overlay (for A/B testing)
    # If variant ends with "_no_cchop", disable C_CHOP overlay
    if variant.endswith("_no_cchop"):
        cchop_cfg = variant_dict.get("sniper_q4_cchop_overlay") or {}
        cchop_cfg["enabled"] = False
        variant_dict["sniper_q4_cchop_overlay"] = cchop_cfg
    elif overlay_enabled:
        # For overlay variants, ensure C_CHOP overlay is enabled (if config exists)
        cchop_cfg = variant_dict.get("sniper_q4_cchop_overlay") or {}
        if cchop_cfg:  # Only enable if config section exists
            cchop_cfg["enabled"] = True
            variant_dict["sniper_q4_cchop_overlay"] = cchop_cfg
    
    # Handle Q4 A_TREND overlay (for A/B testing)
    # If variant ends with "_atrend_gate", enable A_TREND overlay
    if variant.endswith("_atrend_gate"):
        atrend_cfg = variant_dict.get("sniper_q4_atrend_overlay") or {}
        atrend_cfg["enabled"] = True
        atrend_cfg.setdefault("multiplier", 0.30)
        atrend_cfg.setdefault("action", "disable")  # Default: disable (NO-TRADE)
        variant_dict["sniper_q4_atrend_overlay"] = atrend_cfg
    elif variant.endswith("_no_atrend_gate"):
        atrend_cfg = variant_dict.get("sniper_q4_atrend_overlay") or {}
        atrend_cfg["enabled"] = False
        variant_dict["sniper_q4_atrend_overlay"] = atrend_cfg
    elif variant == "atrend_scale":
        # Scale mode: apply multiplier with min unit = 1 (DEPRECATED: ineffective)
        atrend_cfg = variant_dict.get("sniper_q4_atrend_overlay") or {}
        atrend_cfg["enabled"] = True
        atrend_cfg.setdefault("multiplier", 0.30)
        atrend_cfg["action"] = "scale"
        variant_dict["sniper_q4_atrend_overlay"] = atrend_cfg
    elif variant == "atrend_disable":
        # Disable mode: NO-TRADE for Q4 × A_TREND (DEFAULT)
        atrend_cfg = variant_dict.get("sniper_q4_atrend_overlay") or {}
        atrend_cfg["enabled"] = True
        atrend_cfg.setdefault("multiplier", 0.30)
        atrend_cfg["action"] = "disable"
        variant_dict["sniper_q4_atrend_overlay"] = atrend_cfg
    
    # Write variant policy to tmp
    policy_stem = policy_path.stem
    variant_policy_path = tmp_dir / f"{policy_stem}__{variant}.yaml"
    with open(variant_policy_path, "w") as f:
        yaml.dump(variant_dict, f, default_flow_style=False, sort_keys=False)
    
    log.info(f"✅ Built variant policy: {variant_policy_path.name} (variant={variant})")
    return variant_policy_path


def filter_quarter_data(
    data_file: Path, start_ts: str, end_ts: str, output_file: Path
) -> Tuple[int, str, str]:
    """Filter dataset to quarter date range and SNIPER sessions (EU/OVERLAP/US)."""
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from gx1.execution.live_features import infer_session_tag

    df = pd.read_parquet(data_file)
    df.index = pd.to_datetime(df.index, utc=True)

    # Filter by date range
    df = df[(df.index >= start_ts) & (df.index < end_ts)].copy()

    # Filter to SNIPER sessions
    sessions = ["EU", "OVERLAP", "US"]
    mask = df.index.map(lambda ts: infer_session_tag(ts) in sessions)
    df = df[mask]

    log.info(f"Filtered data: {len(df):,} rows")
    log.info(f"Date range: {df.index.min()} → {df.index.max()}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file)

    return len(df), str(df.index.min()), str(df.index.max())


def split_into_chunks(input_file: Path, n_chunks: int, chunk_dir: Path) -> List[Path]:
    """Split filtered data into n_chunks for parallel processing."""
    df = pd.read_parquet(input_file)
    total_bars = len(df)
    chunk_size = total_bars // n_chunks
    remainder = total_bars % n_chunks

    log.info(f"Total bars: {total_bars:,}")
    log.info(f"Chunk size: ~{chunk_size:,} bars each")

    chunk_files = []
    start_idx = 0

    for i in range(n_chunks):
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size

        chunk_df = df.iloc[start_idx:end_idx].copy()
        chunk_file = chunk_dir / f"chunk_{i}.parquet"
        chunk_df.to_parquet(chunk_file)

        log.info(
            f"  Chunk {i}: {len(chunk_df):,} bars ({chunk_df.index.min()} → {chunk_df.index.max()})"
        )
        chunk_files.append(chunk_file)

        start_idx = end_idx

    return chunk_files


def run_chunk_replay(args: Tuple[int, Path, Path, Path, Path]) -> bool:
    """Run replay for a single chunk (wrapper for multiprocessing)."""
    chunk_id, policy_path, chunk_file, output_dir, project_root = args
    try:
        chunk_output_dir = output_dir / "parallel_chunks" / f"chunk_{chunk_id}"
        chunk_output_dir.mkdir(parents=True, exist_ok=True)

        # Set environment variables for this chunk
        os.environ["GX1_CHUNK_ID"] = str(chunk_id)
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["GX1_XGB_THREADS"] = "1"

        # Change to project root
        old_cwd = os.getcwd()
        os.chdir(str(project_root))

        try:
            # Import and run replay
            from gx1.execution.oanda_demo_runner import GX1DemoRunner

            runner = GX1DemoRunner(
                policy_path,
                dry_run_override=True,
                replay_mode=True,
                fast_replay=True,
            )

            log_file = chunk_output_dir / f"chunk_{chunk_id}.log"
            with open(log_file, "w") as f:
                # Redirect stdout/stderr to log file
                import sys
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = f
                sys.stderr = f

                try:
                    runner.run_replay(chunk_file)
                    print(f"[CHUNK {chunk_id}] ✅ Complete")
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr

            return True

        finally:
            os.chdir(old_cwd)

    except Exception as e:
        log.error(f"[CHUNK {chunk_id}] ❌ Exception: {e}", exc_info=True)
        return False


def run_parallel_replay(
    policy_path: Path,
    filtered_data: Path,
    quarter: str,
    n_workers: int,
    project_root: Path,
    variant: str = "baseline",
) -> Optional[Path]:
    """Run parallel replay for a quarter and variant."""
    log.info(f"Running {n_workers} parallel replays for {quarter} (variant={variant})...")

    # Create output directory with variant tag
    run_tag = f"SNIPER_OBS_{quarter}_2025_{variant}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    output_dir = project_root / "gx1" / "wf_runs" / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "parallel_chunks").mkdir(exist_ok=True)

    # Split into chunks
    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    chunk_files = split_into_chunks(filtered_data, n_workers, chunk_dir)

    # Run chunks in parallel using subprocess (same as Q1 script)
    import time
    processes = []
    for i, chunk_file in enumerate(chunk_files):
        chunk_output_dir = output_dir / "parallel_chunks" / f"chunk_{i}"
        chunk_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set environment variables
        env = os.environ.copy()
        env["GX1_CHUNK_ID"] = str(i)
        env["OMP_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["VECLIB_MAXIMUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "1"
        env["GX1_XGB_THREADS"] = "1"
        
        # Run via Python subprocess (same as Q1)
        log_file = chunk_output_dir / f"chunk_{i}.log"
        cmd = [
            sys.executable,
            "-c",
            f"""
import sys
from pathlib import Path
from gx1.execution.oanda_demo_runner import GX1DemoRunner

chunk_id = {i}
policy_path = Path(r'{policy_path}')
chunk_file = Path(r'{chunk_file}')
chunk_output_dir = Path(r'{chunk_output_dir}')

runner = GX1DemoRunner(
    policy_path,
    dry_run_override=True,
    replay_mode=True,
    fast_replay=True,
    output_dir=chunk_output_dir,
)

runner.run_replay(chunk_file)
print(f'[CHUNK {{chunk_id}}] ✅ Complete')
""",
        ]
        
        with open(log_file, "w") as f:
            proc = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append((i, proc))
    
    # Wait for all processes
    results = []
    for chunk_id, proc in processes:
        returncode = proc.wait()
        success = returncode == 0
        results.append((chunk_id, success))
        if success:
            log.info(f"[CHUNK {chunk_id}] ✅ Done")
        else:
            log.error(f"[CHUNK {chunk_id}] ❌ Failed (returncode={returncode})")

    # Check if all chunks succeeded
    failed_chunks = [cid for cid, success in results if not success]
    if failed_chunks:
        log.warning(f"Some chunks failed: {failed_chunks}")
        # Continue anyway - merge what we have

    log.info(f"✅ All chunks completed for {quarter}")
    return output_dir


def generate_metrics(run_dir: Path, quarter: str) -> Dict:
    """Generate metrics from merged trade journal.

    Fail-closed if critical columns are missing.
    """
    index_path = run_dir / "trade_journal" / "trade_journal_index.csv"
    if not index_path.exists():
        log.error(f"Trade journal index not found: {index_path}")
        # Ikke fail-closed her; bare returner tomme metrics (kan håndteres oppstrøms)
        return {}
    
    df = pd.read_csv(index_path)
    log.info(f"Loaded {len(df):,} trades from {quarter}")
    
    # Ensure required columns exist
    required_cols = ["pnl_bps", "entry_time", "exit_time", "exit_reason"]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        log.error(f"Missing required columns in trade journal for {quarter}: {missing_required}")
        raise SystemExit(2)
    
    # Backward compatibility: ensure risk_guard columns exist (may be empty for baseline/older runs)
    for col in [
        "risk_guard_blocked",
        "risk_guard_reason",
        "risk_guard_details",
        "risk_guard_min_prob_long_clamp",
    ]:
        if col not in df.columns:
            if col == "risk_guard_blocked":
                df[col] = False
            else:
                df[col] = ""
    
    # Filter complete trades
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")
    df = df[df["exit_reason"].notna() & df["exit_time"].notna()].copy()
    
    if len(df) == 0:
        log.warning(f"No complete trades found for {quarter}")
        return {}
    
    df["duration_min"] = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 60.0
    
    pnl = df["pnl_bps"].astype(float)
    wins = df[pnl > 0]
    losses = df[pnl < 0]

    metrics = {
        "quarter": quarter,
        "total_trades": len(df),
        "win_rate": len(wins) / len(df) if len(df) > 0 else 0,
        "ev_per_trade": pnl.mean(),
        "pnl_total": pnl.sum(),
        "avg_win": wins["pnl_bps"].mean() if len(wins) > 0 else None,
        "avg_loss": losses["pnl_bps"].mean() if len(losses) > 0 else None,
        "payoff_ratio": (
            wins["pnl_bps"].mean() / abs(losses["pnl_bps"].mean())
            if len(wins) > 0 and len(losses) > 0 and losses["pnl_bps"].mean() != 0
            else None
        ),
        "duration_p50": df["duration_min"].quantile(0.5),
        "duration_p90": df["duration_min"].quantile(0.9),
        "duration_p95": df["duration_min"].quantile(0.95),
        "duration_max": df["duration_min"].max(),
        "exit_reasons": df["exit_reason"].value_counts().to_dict(),
    }

    # Session breakdown
    if "session" in df.columns:
        session_stats = {}
        for sess in ["EU", "OVERLAP", "US"]:
            sess_df = df[df["session"] == sess]
            if len(sess_df) > 0:
                sess_pnl = sess_df["pnl_bps"].astype(float)
                session_stats[sess] = {
                    "trades": len(sess_df),
                    "win_rate": len(sess_pnl[sess_pnl > 0]) / len(sess_df),
                    "ev": sess_pnl.mean(),
                }
        metrics["session_stats"] = session_stats

    # Vol regime breakdown (only if coverage >= 80%)
    if "vol_regime" in df.columns:
        vol_coverage = df["vol_regime"].notna().sum() / len(df)
        if vol_coverage >= 0.8:
            vol_stats = {}
            for regime in ["LOW", "MEDIUM", "HIGH"]:
                regime_df = df[df["vol_regime"] == regime]
                if len(regime_df) > 0:
                    regime_pnl = regime_df["pnl_bps"].astype(float)
                    vol_stats[regime] = {
                        "trades": len(regime_df),
                        "win_rate": len(regime_pnl[regime_pnl > 0]) / len(regime_df),
                        "ev": regime_pnl.mean(),
                    }
            metrics["vol_stats"] = vol_stats
        else:
            metrics["vol_stats"] = "insufficient coverage"

    # Trend regime breakdown (only if coverage >= 80%)
    if "trend_regime" in df.columns:
        trend_coverage = df["trend_regime"].notna().sum() / len(df)
        if trend_coverage >= 0.8:
            trend_stats = {}
            for regime in ["TREND_UP", "TREND_DOWN", "TREND_NEUTRAL", "NEUTRAL"]:
                regime_df = df[df["trend_regime"] == regime]
                if len(regime_df) > 0:
                    regime_pnl = regime_df["pnl_bps"].astype(float)
                    trend_stats[regime] = {
                        "trades": len(regime_df),
                        "win_rate": len(regime_pnl[regime_pnl > 0]) / len(regime_df),
                        "ev": regime_pnl.mean(),
                    }
            metrics["trend_stats"] = trend_stats
        else:
            metrics["trend_stats"] = "insufficient coverage"

    # Max concurrent - global (merged across all chunks)
    events = []
    for _, row in df.iterrows():
        events.append((row["entry_time"], 1))
        events.append((row["exit_time"], -1))
    events.sort()
    current = 0
    max_concurrent_global = 0
    for _, delta in events:
        current += delta
        max_concurrent_global = max(max_concurrent_global, current)
    # Behold original felt for bakoverkompabilitet, men legg til eksplisitt navn
    metrics["max_concurrent"] = max_concurrent_global
    metrics["max_concurrent_global_merged"] = max_concurrent_global

    # Per-chunk concurrency (hvis source_chunk-kolonne finnes)
    max_concurrent_per_chunk_max = None
    max_concurrent_per_chunk_p90 = None
    n_chunks_seen = 1

    if "source_chunk" in df.columns:
        per_chunk_maxes = []
        for source_chunk, chunk_df in df.groupby("source_chunk"):
            if len(chunk_df) == 0:
                continue
            chunk_events = []
            for _, row in chunk_df.iterrows():
                chunk_events.append((row["entry_time"], 1))
                chunk_events.append((row["exit_time"], -1))
            chunk_events.sort()
            current_chunk = 0
            max_chunk = 0
            for _, delta in chunk_events:
                current_chunk += delta
                max_chunk = max(max_chunk, current_chunk)
            per_chunk_maxes.append(max_chunk)

        if per_chunk_maxes:
            n_chunks_seen = len(per_chunk_maxes)
            s = pd.Series(per_chunk_maxes, dtype=float)
            max_concurrent_per_chunk_max = float(s.max())
            max_concurrent_per_chunk_p90 = float(s.quantile(0.9))
    else:
        # Ingen source_chunk-informasjon → behandle som single-run
        n_chunks_seen = 1

    metrics["max_concurrent_per_chunk_max"] = max_concurrent_per_chunk_max
    metrics["max_concurrent_per_chunk_p90"] = max_concurrent_per_chunk_p90
    metrics["n_chunks_seen"] = n_chunks_seen
    
    # Drawdown proxy (loss magnitude, basis points)
    if len(losses) > 0:
        loss_magnitudes = losses["pnl_bps"].astype(float).abs()
        drawdown_p50_bps = float(loss_magnitudes.quantile(0.5)) if len(loss_magnitudes) > 0 else 0.0
        drawdown_p90_bps = float(loss_magnitudes.quantile(0.9)) if len(loss_magnitudes) > 0 else 0.0
    else:
        drawdown_p50_bps = 0.0
        drawdown_p90_bps = 0.0
    
    # Expose with explicit _bps suffix (for delta report)
    metrics["drawdown_p50_bps"] = drawdown_p50_bps
    metrics["drawdown_p90_bps"] = drawdown_p90_bps
    # Backward-compatible keys
    metrics["drawdown_p50"] = drawdown_p50_bps
    metrics["drawdown_p90"] = drawdown_p90_bps

    return metrics


def write_metrics_report(
    run_dir: Path,
    quarter: str,
    metrics: Dict,
    run_header: Dict,
    variant: str = "baseline",
) -> Path:
    """Write metrics report to docs/ and run_dir/."""
    artifacts = run_header.get("artifacts", {})
    policy_path = run_header.get("policy_path", "unknown")
    git_commit = run_header.get("git_commit", "unknown")
    run_tag = run_header.get("run_tag", "unknown")
    
    # Derive base policy and guard config from policy YAML
    base_policy_path = metrics.get("base_policy_path", "unknown")
    guard_config_path = "unknown"
    guard_enabled = "unknown"
    try:
        policy_yaml_path = Path(policy_path)
        if policy_yaml_path.exists():
            with open(policy_yaml_path) as f:
                policy_yaml = yaml.safe_load(f) or {}
            risk_guard_cfg = policy_yaml.get("risk_guard") or {}
            guard_config_path = risk_guard_cfg.get("config_path", "unknown")
            if guard_config_path != "unknown":
                guard_path = Path(guard_config_path)
                if not guard_path.is_absolute():
                    project_root = Path(__file__).parent.parent.parent
                    guard_path = project_root / guard_path
                if guard_path.exists():
                    with open(guard_path) as gf:
                        guard_yaml = yaml.safe_load(gf) or {}
                    guard_core = guard_yaml.get("sniper_risk_guard_v1") or {}
                    guard_enabled = bool(guard_core.get("enabled", False))
    except Exception as e:
        log.warning(f"Failed to derive guard config info for report: {e}")

    lines = [
        f"# SNIPER {quarter} 2025 Replay Metrics ({variant})",
        "",
        f"**Variant**: {variant}",
        f"**Run tag**: {run_tag}",
        f"**Quarter**: {quarter}",
        "",
        "## Policy & Guard configuration",
        f"- Policy path (variant): `{policy_path}`",
        f"- Base policy path: `{base_policy_path}`",
        f"- Guard config path: `{guard_config_path}`",
        f"- Guard enabled: `{guard_enabled}`",
        "",
        "## Executive summary",
        f"- EV per trade: {metrics.get('ev_per_trade', 0):.2f} bps",
        f"- Win rate: {metrics.get('win_rate', 0)*100:.1f}%",
        f"- Payoff ratio: {metrics.get('payoff_ratio', 0):.2f}" if metrics.get('payoff_ratio') else "- Payoff ratio: N/A",
        "",
        "## Baseline & artifact fingerprint",
        f"- Policy path: `{policy_path}`",
        f"- Entry model sha: `{artifacts.get('entry_model', {}).get('sha256', 'n/a')}`",
        f"- Feature manifest sha: `{artifacts.get('feature_manifest', {}).get('sha256', 'n/a')}`",
        f"- Router sha: `{artifacts.get('router_model', {}).get('sha256', 'n/a')}`",
        f"- Git commit: `{git_commit}`",
        "",
        "## Global performance metrics",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Total trades | {metrics.get('total_trades', 0):,} |",
        f"| Win rate | {metrics.get('win_rate', 0)*100:.1f}% |",
        f"| Avg win | {metrics.get('avg_win', 0):.2f} bps |" if metrics.get('avg_win') else "| Avg win | N/A |",
        f"| Avg loss | {metrics.get('avg_loss', 0):.2f} bps |" if metrics.get('avg_loss') else "| Avg loss | N/A |",
        f"| Payoff ratio | {metrics.get('payoff_ratio', 0):.2f} |" if metrics.get('payoff_ratio') else "| Payoff ratio | N/A |",
        f"| EV per trade | {metrics.get('ev_per_trade', 0):.2f} bps |",
        f"| PnL sum | {metrics.get('pnl_total', 0):,.2f} bps |",
        f"| Duration p50/p90/p95 | {metrics.get('duration_p50', 0):.1f}/{metrics.get('duration_p90', 0):.1f}/{metrics.get('duration_p95', 0):.1f} min |",
        f"| Max duration | {metrics.get('duration_max', 0):.1f} min |",
        f"| Max concurrent | {metrics.get('max_concurrent', 0)} |",
    ]

    # Exit reasons
    exit_reasons = metrics.get("exit_reasons", {})
    if exit_reasons:
        lines.append("")
        lines.append("## Exit reasons")
        for reason, count in sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True):
            pct = count / metrics.get("total_trades", 1) * 100
            lines.append(f"- {reason}: {count:,} ({pct:.1f}%)")

    # Session breakdown
    session_stats = metrics.get("session_stats", {})
    if session_stats:
        lines.append("")
        lines.append("## Session breakdown")
        lines.append("| Session | Trades | Win rate | EV/trade (bps) |")
        lines.append("| --- | --- | --- | --- |")
        for sess in ["EU", "OVERLAP", "US"]:
            if sess in session_stats:
                stats = session_stats[sess]
                lines.append(
                    f"| {sess} | {stats['trades']:,} | {stats['win_rate']*100:.1f}% | {stats['ev']:.2f} |"
                )
            else:
                lines.append(f"| {sess} | 0 | n/a | n/a |")

    # Vol regime breakdown
    vol_stats = metrics.get("vol_stats")
    if vol_stats and vol_stats != "insufficient coverage":
        lines.append("")
        lines.append("## Vol-regime breakdown")
        lines.append("| Regime | Trades | Win rate | EV/trade (bps) |")
        lines.append("| --- | --- | --- | --- |")
        for regime in ["LOW", "MEDIUM", "HIGH"]:
            if regime in vol_stats:
                stats = vol_stats[regime]
                lines.append(
                    f"| {regime} | {stats['trades']:,} | {stats['win_rate']*100:.1f}% | {stats['ev']:.2f} |"
                )
            else:
                lines.append(f"| {regime} | 0 | n/a | n/a |")
    elif vol_stats == "insufficient coverage":
        lines.append("")
        lines.append("## Vol-regime breakdown")
        lines.append("- Insufficient coverage (< 80%)")

    # Trend regime breakdown
    trend_stats = metrics.get("trend_stats")
    if trend_stats and trend_stats != "insufficient coverage":
        lines.append("")
        lines.append("## Trend-regime breakdown")
        lines.append("| Regime | Trades | Win rate | EV/trade (bps) |")
        lines.append("| --- | --- | --- | --- |")
        for regime in ["TREND_UP", "TREND_DOWN", "TREND_NEUTRAL", "NEUTRAL"]:
            if regime in trend_stats:
                stats = trend_stats[regime]
                lines.append(
                    f"| {regime} | {stats['trades']:,} | {stats['win_rate']*100:.1f}% | {stats['ev']:.2f} |"
                )

    # Risk & concurrency metrics
    lines.append("")
    lines.append("## Risk & duration")
    if metrics.get("drawdown_p90_bps") is not None or metrics.get("drawdown_p90") is not None:
        p50 = metrics.get("drawdown_p50_bps", metrics.get("drawdown_p50", 0)) or 0
        p90 = metrics.get("drawdown_p90_bps", metrics.get("drawdown_p90", 0)) or 0
        lines.append(
            f"- Drawdown proxy (loss magnitude) p50/p90: {p50:.2f}/{p90:.2f} bps"
        )

    # Concurrency metrics
    max_global = metrics.get("max_concurrent_global_merged", metrics.get("max_concurrent"))
    if max_global is not None:
        lines.append(f"- Max concurrent (global merged): {int(max_global)}")

    per_chunk_max = metrics.get("max_concurrent_per_chunk_max")
    per_chunk_p90 = metrics.get("max_concurrent_per_chunk_p90")
    n_chunks = metrics.get("n_chunks_seen", 1)
    if per_chunk_max is not None and per_chunk_p90 is not None:
        lines.append(f"- Max concurrent (per-chunk max): {int(per_chunk_max)}")
        lines.append(f"- Max concurrent (per-chunk p90): {per_chunk_p90:.1f}")
        lines.append(f"- Chunks: {n_chunks}")

    # Write to docs/ with variant suffix
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    doc_path = docs_dir / f"SNIPER_{quarter}_METRICS__{variant}.md"
    
    # Add timestamp if file exists
    if doc_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        doc_path = docs_dir / f"SNIPER_{quarter}_METRICS__{variant}_{timestamp}.md"
        log.warning(f"File exists, writing to: {doc_path}")

    with open(doc_path, "w") as f:
        f.write("\n".join(lines))

    # Also write to run_dir (include variant for clarity)
    run_metrics = run_dir / f"sniper_{quarter.lower()}_metrics__{variant}.md"
    with open(run_metrics, "w") as f:
        f.write("\n".join(lines))

    return doc_path


def verify_run(run_dir: Path, quarter: str) -> Tuple[bool, List[str]]:
    """Verify run completeness."""
    issues = []

    # Check run_header.json
    header_path = run_dir / "run_header.json"
    if not header_path.exists():
        issues.append(f"run_header.json missing")

    # Check trade journal index
    index_path = run_dir / "trade_journal" / "trade_journal_index.csv"
    if not index_path.exists():
        issues.append(f"trade_journal_index.csv missing")
    else:
        df = pd.read_csv(index_path)
        # Check left_open == 0
        left_open = df[df["exit_reason"].isna() | df["exit_time"].isna()].shape[0]
        if left_open > 0:
            issues.append(f"left_open = {left_open} (expected 0)")

        # Check session coverage
        if "session" in df.columns:
            session_coverage = df["session"].notna().sum() / len(df)
            if session_coverage < 0.95:
                issues.append(f"session coverage = {session_coverage*100:.1f}% (expected >= 95%)")

        # Check range features coverage
        if "range_pos" in df.columns:
            range_coverage = df["range_pos"].notna().sum() / len(df)
            if range_coverage < 0.95:
                issues.append(
                    f"range_pos coverage = {range_coverage*100:.1f}% (expected >= 95%)"
                )

    return len(issues) == 0, issues


def create_oos_summary(
    quarter_metrics: Dict[str, Dict[str, Dict]],
    quarters: List[str],
    variant: str,
) -> Path:
    """Create OOS summary comparing Q1-Q4 for a specific variant (baseline/guarded)."""
    # Determine which quarters actually have metrics for this variant
    available_quarters = [q for q in quarters if quarter_metrics.get(q, {}).get(variant)]
    missing_quarters = [q for q in quarters if q not in available_quarters]
    is_partial = len(missing_quarters) > 0
    
    available_str = ", ".join(available_quarters) if available_quarters else "none"
    missing_str = ", ".join(missing_quarters) if missing_quarters else "none"
    lines = [
        f"# SNIPER 2025 Out-of-Sample Summary (Q1-Q4, variant={variant})",
        "",
        f"**Computed from quarters**: {available_str} (missing: {missing_str})",
        "Note: partial-year / partial-OOS; averages and comparisons are computed over available quarters only.",
    ]

    # Clarify Q1 baseline availability
    has_q1_baseline = quarter_metrics.get("Q1", {}).get(variant) is not None
    if not has_q1_baseline:
        lines.append(
            "Comparisons vs Q1 are **N/A** because Q1 metrics for this variant are missing."
        )
    lines.extend(
        [
            "",
            "## Executive Summary",
            "",
            "This report compares SNIPER performance across Q1-Q4 2025 to identify",
            "trends, anomalies, and recommendations for policy adjustments.",
            "",
            "## Quarter-by-Quarter Comparison",
            "",
            "### EV per Trade",
            "| Quarter | EV (bps) | Change vs Q1 | Included in OOS avg |",
            "| --- | --- | --- | --- |",
        ]
    )

    # Use Q1 as baseline only if metrics for this variant exists
    q1_metrics = quarter_metrics.get("Q1", {}).get(variant)
    has_q1 = q1_metrics is not None
    q1_ev = q1_metrics.get("ev_per_trade", 0) if has_q1 else 0

    for q in ["Q1", "Q2", "Q3", "Q4"]:
        qm = quarter_metrics.get(q, {})
        m = qm.get(variant)
        if not m:
            lines.append(f"| {q} | missing | N/A | no |")
            continue

        ev = m.get("ev_per_trade", 0)
        included = "yes" if (q in available_quarters and q != "Q1") else "no"
        if q == "Q1":
            if has_q1:
                lines.append(f"| {q} | {ev:.2f} | baseline | {included} |")
            else:
                lines.append(f"| {q} | missing | N/A | {included} |")
        else:
            if has_q1 and q1_ev != 0:
                change_pct = ((ev - q1_ev) / abs(q1_ev) * 100)
                change_str = f"{change_pct:+.1f}%"
            else:
                change_str = "N/A"
            lines.append(f"| {q} | {ev:.2f} | {change_str} | {included} |")

    lines.append("")
    lines.append("### Win Rate")
    lines.append("| Quarter | Win Rate | Change vs Q1 | Included in OOS avg |")
    lines.append("| --- | --- | --- | --- |")
    q1_wr = q1_metrics.get("win_rate", 0) * 100 if has_q1 else 0
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        qm = quarter_metrics.get(q, {})
        m = qm.get(variant)
        if not m:
            lines.append(f"| {q} | missing | N/A | no |")
            continue

        wr = m.get("win_rate", 0) * 100
        included = "yes" if (q in available_quarters and q != "Q1") else "no"
        if q == "Q1":
            if has_q1:
                lines.append(f"| {q} | {wr:.1f}% | baseline | {included} |")
            else:
                lines.append(f"| {q} | missing | N/A | {included} |")
        else:
            if has_q1:
                change = wr - q1_wr
                change_str = f"{change:+.1f}pp"
            else:
                change_str = "N/A"
            lines.append(f"| {q} | {wr:.1f}% | {change_str} | {included} |")

    lines.append("")
    lines.append("### Payoff Ratio")
    lines.append("| Quarter | Payoff | Change vs Q1 | Included in OOS avg |")
    lines.append("| --- | --- | --- | --- |")
    q1_payoff = q1_metrics.get("payoff_ratio") if has_q1 else None
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        qm = quarter_metrics.get(q, {})
        m = qm.get(variant)
        if not m:
            lines.append(f"| {q} | missing | N/A | no |")
            continue

        payoff = m.get("payoff_ratio")
        included = "yes" if (q in available_quarters and q != "Q1") else "no"
        if payoff:
            if q == "Q1":
                if has_q1:
                    lines.append(f"| {q} | {payoff:.2f} | baseline | {included} |")
                else:
                    lines.append(f"| {q} | N/A | N/A | {included} |")
            else:
                if has_q1 and q1_payoff:
                    change = payoff - q1_payoff
                    change_str = f"{change:+.2f}"
                else:
                    change_str = "N/A"
                lines.append(f"| {q} | {payoff:.2f} | {change_str} | {included} |")
        else:
            lines.append(f"| {q} | N/A | N/A | {included} |")

    lines.append("")
    lines.append("### Drawdown Proxy (p90 Loss)")
    lines.append("| Quarter | p90 Loss (bps) |")
    lines.append("| --- | --- |")
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        qm = quarter_metrics.get(q, {})
        m = qm.get(variant)
        if not m:
            lines.append(f"| {q} | missing |")
            continue

        dd = m.get("drawdown_p90_bps", m.get("drawdown_p90"))
        if dd:
            lines.append(f"| {q} | {dd:.2f} |")
        else:
            lines.append(f"| {q} | N/A |")

    # Session EV per quarter
    lines.append("")
    lines.append("### Session EV per Quarter")
    lines.append("| Quarter | EU (bps) | OVERLAP (bps) | US (bps) |")
    lines.append("| --- | --- | --- | --- |")
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        qm = quarter_metrics.get(q, {})
        m = qm.get(variant)
        if not m:
            lines.append(f"| {q} | missing | missing | missing |")
            continue

        sess_stats = m.get("session_stats", {})
        eu_ev = sess_stats.get("EU", {}).get("ev", 0) if sess_stats.get("EU") else 0
        ov_ev = sess_stats.get("OVERLAP", {}).get("ev", 0) if sess_stats.get("OVERLAP") else 0
        us_ev = sess_stats.get("US", {}).get("ev", 0) if sess_stats.get("US") else 0
        lines.append(f"| {q} | {eu_ev:.2f} | {ov_ev:.2f} | {us_ev:.2f} |")

    # Flag anomalies
    lines.append("")
    lines.append("## Anomalies & Flags")
    flags = []
    for q in ["Q2", "Q3", "Q4"]:
        qm = quarter_metrics.get(q, {})
        m = qm.get(variant)
        if not m:
            continue
        q_ev = m.get("ev_per_trade", 0)
        q_payoff = m.get("payoff_ratio")
        q_duration_max = m.get("duration_max", 0)

        # EV drop > 30%
        if q1_ev != 0:
            ev_drop = ((q1_ev - q_ev) / abs(q1_ev)) * 100
            if ev_drop > 30:
                flags.append(f"{q}: EV dropped {ev_drop:.1f}% vs Q1")

        # Payoff drop
        if q_payoff and q1_payoff:
            if q_payoff < q1_payoff * 0.7:
                flags.append(f"{q}: Payoff ratio dropped significantly ({q_payoff:.2f} vs {q1_payoff:.2f})")

        # Duration explosion in US
        if q_duration_max > 20000:  # ~14 days
            flags.append(f"{q}: Max duration very high ({q_duration_max:.0f} min)")

    if flags:
        for flag in flags:
            lines.append(f"- ⚠️ {flag}")
    else:
        lines.append("- No significant anomalies detected")

    # Recommendations
    lines.append("")
    lines.append("## Recommendations")
    
    # Analyze trends
    evs = [quarter_metrics.get(q, {}).get("ev_per_trade", 0) for q in ["Q1", "Q2", "Q3", "Q4"]]
    evs = [e for e in evs if e != 0]
    
    if len(evs) >= 2:
        ev_trend = "declining" if evs[-1] < evs[0] * 0.7 else "stable" if abs(evs[-1] - evs[0]) < evs[0] * 0.2 else "improving"
        
        # Check session consistency
        session_evs = {}
        for sess in ["EU", "OVERLAP", "US"]:
            sess_evs = []
            for q in ["Q1", "Q2", "Q3", "Q4"]:
                if q in quarter_metrics:
                    sess_stats = quarter_metrics[q].get("session_stats", {})
                    if sess in sess_stats:
                        sess_evs.append(sess_stats[sess].get("ev", 0))
            if sess_evs:
                session_evs[sess] = sess_evs
        
        # Generate recommendation
        if ev_trend == "declining":
            lines.append("- **Trend**: EV declining across quarters")
            lines.append("- **Recommendation**: Consider tightening entry thresholds or reviewing regime filters")
        elif ev_trend == "stable":
            lines.append("- **Trend**: EV relatively stable")
            lines.append("- **Recommendation**: **Keep unified** - current policy performs consistently")
        else:
            lines.append("- **Trend**: EV improving")
            lines.append("- **Recommendation**: **Keep unified** - policy is performing well")
        
        # Session-specific recommendations
        if session_evs:
            eu_consistent = all(abs(e - session_evs["EU"][0]) < session_evs["EU"][0] * 0.3 for e in session_evs["EU"][1:]) if len(session_evs.get("EU", [])) > 1 else False
            us_consistent = all(abs(e - session_evs["US"][0]) < session_evs["US"][0] * 0.3 for e in session_evs["US"][1:]) if len(session_evs.get("US", [])) > 1 else False
            
            if not eu_consistent or not us_consistent:
                lines.append("- **Session divergence**: Consider **split EU vs US** if divergence persists")
            
            # Check OVERLAP
            if "OVERLAP" in session_evs:
                ov_evs = session_evs["OVERLAP"]
                if len(ov_evs) > 1 and any(e < ov_evs[0] * 0.7 for e in ov_evs[1:]):
                    lines.append("- **OVERLAP degradation**: Consider **tighten OVERLAP only** if trend continues")
    else:
        lines.append("- Insufficient data for recommendations")

    # Write summary
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    summary_path = docs_dir / f"SNIPER_2025_OOS_SUMMARY__{variant}.md"
    
    if summary_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = docs_dir / f"SNIPER_2025_OOS_SUMMARY__{variant}_{timestamp}.md"
        log.warning(f"File exists, writing to: {summary_path}")

    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    return summary_path


def create_delta_report(quarter_metrics: Dict[str, Dict[str, Dict]], quarters: List[str]) -> Optional[Path]:
    """
    Create delta report comparing baseline vs guarded variants.
    
    Args:
        quarter_metrics: {quarter: {variant: metrics_dict}}
        quarters: list of quarters to include
    """
    lines: List[str] = [
        "# SNIPER 2025 Delta Report: Baseline vs Guarded",
        "",
        "## Executive Summary",
        "",
        "This report compares SNIPER performance between **baseline** and **guarded** variants",
        "across all quarters to assess the impact of the SNIPER risk guard on tail risk and EV.",
        "",
    ]
    
    lines.append("## Quarter-by-Quarter Comparison")
    lines.append("")
    lines.append("| Quarter | Metric | Baseline | Guarded | Delta | % Change |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    
    for quarter in quarters:
        qm = quarter_metrics.get(quarter, {})
        baseline = qm.get("baseline")
        guarded = qm.get("guarded")
        
        if not baseline or not guarded:
            lines.append(f"| {quarter} | Status | {'missing' if not baseline else 'OK'} | {'missing' if not guarded else 'OK'} | - | - |")
            continue
        
        # Trades
        b_trades = baseline.get("total_trades", 0)
        g_trades = guarded.get("total_trades", 0)
        d_trades = g_trades - b_trades
        pct_trades = (d_trades / b_trades * 100) if b_trades > 0 else 0
        lines.append(f"| {quarter} | Trades | {b_trades:,} | {g_trades:,} | {d_trades:+,} | {pct_trades:+.1f}% |")
        
        # EV/trade
        b_ev = baseline.get("ev_per_trade", 0.0)
        g_ev = guarded.get("ev_per_trade", 0.0)
        d_ev = g_ev - b_ev
        pct_ev = (d_ev / b_ev * 100) if b_ev != 0 else 0.0
        lines.append(f"| {quarter} | EV/trade (bps) | {b_ev:.2f} | {g_ev:.2f} | {d_ev:+.2f} | {pct_ev:+.1f}% |")
        
        # Win rate
        b_wr = baseline.get("win_rate", 0.0) * 100
        g_wr = guarded.get("win_rate", 0.0) * 100
        d_wr = g_wr - b_wr
        pct_wr = (d_wr / b_wr * 100) if b_wr != 0 else 0.0
        lines.append(f"| {quarter} | Win rate (%) | {b_wr:.1f} | {g_wr:.1f} | {d_wr:+.1f}pp | {pct_wr:+.1f}% |")
        
        # Avg loss
        b_loss = baseline.get("avg_loss", 0.0) or 0.0
        g_loss = guarded.get("avg_loss", 0.0) or 0.0
        d_loss = g_loss - b_loss
        pct_loss = (d_loss / abs(b_loss) * 100) if b_loss != 0 else 0.0
        lines.append(f"| {quarter} | Avg loss (bps) | {b_loss:.2f} | {g_loss:.2f} | {d_loss:+.2f} | {pct_loss:+.1f}% |")
        
        # P90 loss (drawdown proxy)
        b_p90 = baseline.get("drawdown_p90_bps", baseline.get("drawdown_p90", 0.0)) or 0.0
        g_p90 = guarded.get("drawdown_p90_bps", guarded.get("drawdown_p90", 0.0)) or 0.0
        d_p90 = g_p90 - b_p90
        pct_p90 = (d_p90 / b_p90 * 100) if b_p90 != 0 else 0.0
        lines.append(f"| {quarter} | P90 loss (bps) | {b_p90:.2f} | {g_p90:.2f} | {d_p90:+.2f} | {pct_p90:+.1f}% |")
        
        # Payoff ratio
        b_payoff = baseline.get("payoff_ratio")
        g_payoff = guarded.get("payoff_ratio")
        if b_payoff is not None and g_payoff is not None:
            d_payoff = g_payoff - b_payoff
            pct_payoff = (d_payoff / b_payoff * 100) if b_payoff != 0 else 0.0
            lines.append(f"| {quarter} | Payoff ratio | {b_payoff:.2f} | {g_payoff:.2f} | {d_payoff:+.2f} | {pct_payoff:+.1f}% |")
        
        # Max concurrent (per-chunk)
        b_chunk_max = baseline.get("max_concurrent_per_chunk_max", baseline.get("max_concurrent_global_merged"))
        g_chunk_max = guarded.get("max_concurrent_per_chunk_max", guarded.get("max_concurrent_global_merged"))
        if b_chunk_max is not None and g_chunk_max is not None:
            d_chunk_max = g_chunk_max - b_chunk_max
            lines.append(f"| {quarter} | Max concurrent (per-chunk max) | {int(b_chunk_max)} | {int(g_chunk_max)} | {int(d_chunk_max):+} | - |")
        
        # Max concurrent (global merged) as FYI
        b_global = baseline.get("max_concurrent_global_merged", baseline.get("max_concurrent"))
        g_global = guarded.get("max_concurrent_global_merged", guarded.get("max_concurrent"))
        if b_global is not None and g_global is not None:
            d_global = g_global - b_global
            lines.append(f"| {quarter} | Max concurrent (global merged, FYI) | {int(b_global)} | {int(g_global)} | {int(d_global):+} | - |")
        
        lines.append("|  |  |  |  |  |  |")
    
    # Guard impact summary
    lines.append("")
    lines.append("## Guard Impact Summary")
    lines.append("")
    
    # Average P90 loss reduction
    p90_reductions = []
    ev_changes = []
    for quarter in quarters:
        qm = quarter_metrics.get(quarter, {})
        baseline = qm.get("baseline")
        guarded = qm.get("guarded")
        if not baseline or not guarded:
            continue
        b_p90 = baseline.get("drawdown_p90_bps", baseline.get("drawdown_p90", 0.0)) or 0.0
        g_p90 = guarded.get("drawdown_p90_bps", guarded.get("drawdown_p90", 0.0)) or 0.0
        if b_p90 != 0:
            p90_reductions.append((b_p90 - g_p90) / b_p90 * 100.0)
        b_ev = baseline.get("ev_per_trade", 0.0)
        g_ev = guarded.get("ev_per_trade", 0.0)
        if b_ev != 0:
            ev_changes.append((g_ev - b_ev) / b_ev * 100.0)
    
    if p90_reductions:
        avg_red = sum(p90_reductions) / len(p90_reductions)
        lines.append(f"- **Average P90 loss reduction**: {avg_red:.1f}%")
    if ev_changes:
        avg_ev = sum(ev_changes) / len(ev_changes)
        lines.append(f"- **Average EV impact**: {avg_ev:+.1f}%")
    
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    if p90_reductions and ev_changes:
        avg_red = sum(p90_reductions) / len(p90_reductions)
        avg_ev = sum(ev_changes) / len(ev_changes)
        if avg_red > 10 and avg_ev > -10:
            lines.append("✅ **Recommendation: Keep guard** – significant tail risk reduction with acceptable EV impact.")
        elif avg_red > 5 and avg_ev > -5:
            lines.append("⚠️ **Recommendation: Consider keeping guard** – moderate tail risk reduction with low EV cost.")
        else:
            lines.append("❌ **Recommendation: Review guard configuration** – limited tail risk benefit or high EV cost.")
    else:
        lines.append("⚠️ **Insufficient data for a robust conclusion.**")
    
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    delta_path = docs_dir / "SNIPER_2025_DELTA_BASELINE_vs_GUARDED.md"
    if delta_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        delta_path = docs_dir / f"SNIPER_2025_DELTA_BASELINE_vs_GUARDED_{timestamp}.md"
        log.warning(f"File exists, writing to: {delta_path}")
    
    with open(delta_path, "w") as f:
        f.write("\n".join(lines))
    
    return delta_path


def load_q1_metrics() -> Optional[Dict]:
    """Load Q1 metrics from existing run."""
    # Try to find Q1 run and generate metrics
    q1_runs = sorted(Path("gx1/wf_runs").glob("SNIPER_OBS_Q1_2025_*"), reverse=True)
    if q1_runs:
        log.info(f"Found Q1 run: {q1_runs[0].name}")
        # Load run header
        header_path = q1_runs[0] / "run_header.json"
        if header_path.exists():
            with open(header_path) as f:
                run_header = json.load(f)
        else:
            run_header = {"artifacts": {}, "policy_path": "unknown", "git_commit": "unknown"}
        
        # Merge journals if needed
        index_path = q1_runs[0] / "trade_journal" / "trade_journal_index.csv"
        if not index_path.exists():
            log.info("Merging Q1 trade journals...")
            merge_cmd = [sys.executable, "gx1/scripts/merge_trade_journals.py", str(q1_runs[0])]
            subprocess.run(merge_cmd, cwd=str(Path.cwd()), capture_output=True, text=True)
        
        metrics = generate_metrics(q1_runs[0], "Q1")
        if metrics:
            metrics["run_header"] = run_header
            return metrics
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Run SNIPER quarter replays (Q2-Q4)")
    parser.add_argument(
        "--policy",
        type=Path,
        required=True,
        help="Path to SNIPER policy YAML",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to parquet data file",
    )
    parser.add_argument(
        "--out-slices",
        type=Path,
        default=Path("data/raw/_sniper_slices"),
        help="Output directory for quarter slices",
    )
    parser.add_argument(
        "--quarters",
        type=str,
        default="Q2,Q3,Q4",
        help="Comma-separated list of quarters (default: Q2,Q3,Q4)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=7,
        help="Number of parallel workers (default: 7)",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="baseline,guarded",
        help="Comma-separated list of variants (default: baseline,guarded). "
             "Supported: baseline, guarded, baseline_overlay, guarded_overlay.",
    )
    args = parser.parse_args()

    project_root = Path.cwd()
    quarters = [q.strip() for q in args.quarters.split(",")]
    variants = [v.strip() for v in args.variants.split(",")]
    
    # Validate variants
    # NOTE: Overlay variants validated here; actual behavior handled in build_variant_policy()
    valid_variants = [
        "baseline",
        "guarded",
        "baseline_overlay",
        "guarded_overlay",
        "baseline_no_cchop",
        "baseline_atrend_gate",
        "baseline_no_atrend_gate",
        "atrend_scale",
        "atrend_disable",
    ]
    for variant in variants:
        if variant not in valid_variants:
            log.error(
                f"Invalid variant: {variant}. "
                f"Must be one of {valid_variants}"
            )
            return 1

    log.info("=" * 80)
    log.info("SNIPER Quarter Replays with Variants")
    log.info("=" * 80)
    log.info(f"Policy: {args.policy}")
    log.info(f"Data: {args.data}")
    log.info(f"Quarters: {', '.join(quarters)}")
    log.info(f"Variants: {', '.join(variants)}")
    log.info(f"Workers: {args.workers}")
    log.info("")
    
    # Create temp directory for variant policies
    tmp_dir = Path(tempfile.mkdtemp(prefix="sniper_variants_"))
    log.info(f"Using temp directory for variant policies: {tmp_dir}")

    # Load Q1 metrics if available
    q1_metrics = load_q1_metrics()
    quarter_metrics: Dict[str, Dict[str, Dict]] = {}
    if q1_metrics:
        quarter_metrics.setdefault("Q1", {})["baseline"] = q1_metrics
        log.info("✅ Loaded Q1 metrics from existing run (baseline)")
    
    # Process each quarter
    for quarter in quarters:
        if quarter not in QUARTER_RANGES:
            log.error(f"Unknown quarter: {quarter}")
            continue

        log.info("")
        log.info("=" * 80)
        log.info(f"Processing {quarter} 2025")
        log.info("=" * 80)

        start_ts, end_ts = QUARTER_RANGES[quarter]
        
        # Initialize quarter metrics dict
        if quarter not in quarter_metrics:
            quarter_metrics[quarter] = {}

        # Step 1: Filter quarter data (once per quarter, shared across variants)
        log.info(f"[1/6] Filtering {quarter} data (EU/OVERLAP/US only)...")
        slice_file = args.out_slices / f"{quarter.lower()}_filtered.parquet"
        n_rows, min_ts, max_ts = filter_quarter_data(args.data, start_ts, end_ts, slice_file)
        log.info(f"✅ Filtered {n_rows:,} rows ({min_ts} → {max_ts})")

        # Process each variant for this quarter
        for variant in variants:
            log.info("")
            log.info("-" * 80)
            log.info(f"Processing {quarter} 2025 - Variant: {variant}")
            log.info("-" * 80)
            
            # Build variant policy
            try:
                variant_policy_path = build_variant_policy(args.policy, variant, tmp_dir)
            except Exception as e:
                log.error(f"Failed to build variant policy for {variant}: {e}")
                return 1

            # Step 2: Run parallel replay
            log.info(f"[2/6] Running parallel replay for {quarter} ({variant})...")
            run_dir = run_parallel_replay(
                variant_policy_path, slice_file, quarter, args.workers, project_root, variant
            )

            if not run_dir:
                log.error(f"Failed to run replay for {quarter} ({variant})")
                quarter_metrics[quarter][variant] = None  # Mark as missing
                continue

            log.info(f"✅ Replay complete: {run_dir.name}")

            # Step 3: Merge trade journals (NO legacy, with quarter sanity check)
            log.info(f"[3/6] Merging trade journals for {quarter} ({variant})...")
            merge_cmd = [
                sys.executable, 
                "gx1/scripts/merge_trade_journals.py", 
                str(run_dir),
                "--expected-quarter", quarter,
            ]
            result = subprocess.run(merge_cmd, cwd=str(project_root), capture_output=True, text=True)
            if result.returncode != 0:
                log.error(f"Merge failed for {quarter} ({variant}): {result.stderr}")
                log.error(f"Merge stdout: {result.stdout}")
                quarter_metrics[quarter][variant] = None  # Mark as missing
                continue
            
            log.info("✅ Trade journals merged (legacy disabled, quarter verified)")

            # Step 4: Generate metrics
            log.info(f"[4/6] Generating metrics for {quarter} ({variant})...")
            
            # Load run header (REQUIRED - fail-closed if missing)
            header_path = run_dir / "run_header.json"
            if not header_path.exists():
                log.error(f"❌ run_header.json missing for {quarter} ({variant}): {header_path}")
                log.error("   Cannot generate metrics without fingerprint. Aborting.")
                raise SystemExit(2)
            
            with open(header_path) as f:
                run_header = json.load(f)
            
            # Verify critical fields
            if not run_header.get("policy_path") or run_header.get("policy_path") == "unknown":
                log.warning(f"⚠️  Policy path missing in run_header.json for {quarter} ({variant})")
            if not run_header.get("git_commit") or run_header.get("git_commit") == "unknown":
                log.warning(f"⚠️  Git commit missing in run_header.json for {quarter} ({variant})")

            metrics = generate_metrics(run_dir, quarter)
            if metrics:
                metrics["run_header"] = run_header
                quarter_metrics[quarter][variant] = metrics

                # Write report with variant
                report_path = write_metrics_report(run_dir, quarter, metrics, run_header, variant)
                log.info(f"✅ Metrics report: {report_path}")

                # Print summary
                log.info("")
                log.info(f"📊 {quarter} ({variant}) Summary:")
                log.info(f"   Run: {run_dir.name}")
                log.info(f"   Trades: {metrics.get('total_trades', 0):,}")
                log.info(f"   EV/trade: {metrics.get('ev_per_trade', 0):.2f} bps")
                log.info(f"   Win rate: {metrics.get('win_rate', 0)*100:.1f}%")
                if metrics.get('payoff_ratio'):
                    log.info(f"   Payoff: {metrics.get('payoff_ratio', 0):.2f}")
            else:
                quarter_metrics[quarter][variant] = None  # Mark as missing

            # Step 5: Verify
            log.info(f"[5/6] Verifying {quarter} ({variant}) run...")
            verified, issues = verify_run(run_dir, quarter)
            if verified:
                log.info(f"✅ {quarter} ({variant}) verification passed")
            else:
                log.warning(f"⚠️ {quarter} ({variant}) verification issues: {', '.join(issues)}")

    # Cleanup temp directory
    try:
        shutil.rmtree(tmp_dir)
        log.info(f"✅ Cleaned up temp directory: {tmp_dir}")
    except Exception as e:
        log.warning(f"Failed to cleanup temp directory {tmp_dir}: {e}")

    # Generate OOS summaries per variant
    log.info("")
    log.info("=" * 80)
    log.info("Generating OOS Summaries per variant")
    log.info("=" * 80)
    for variant in variants:
        try:
            summary_path = create_oos_summary(quarter_metrics, quarters, variant)
            log.info(f"✅ OOS Summary ({variant}): {summary_path}")
        except Exception as e:
            log.error(f"❌ Failed to generate OOS summary for variant={variant}: {e}")
            return 1

    # Generate delta report (baseline vs guarded) if both varianter er med
    if "baseline" in variants and "guarded" in variants:
        log.info("")
        log.info("=" * 80)
        log.info("Generating Delta Report (baseline vs guarded)")
        log.info("=" * 80)
        
        delta_path = create_delta_report(quarter_metrics, quarters)
        if delta_path:
            log.info(f"✅ Delta report: {delta_path}")
        else:
            log.error("❌ Failed to generate delta report")
            return 1

    # Final verification
    log.info("")
    log.info("=" * 80)
    log.info("Final Verification")
    log.info("=" * 80)
    
    # Check for missing quarters/variants
    missing = []
    for q in quarters:
        if q not in quarter_metrics:
            missing.append(f"{q} (all variants)")
            continue
        for variant in variants:
            if variant not in quarter_metrics[q] or quarter_metrics[q][variant] is None:
                missing.append(f"{q} ({variant})")
    
    if missing:
        log.error("❌ Missing quarters/variants:")
        for m in missing:
            log.error(f"   - {m}")
        return 1
    
    # Check reports exist
    required_reports = []
    for q in quarters:
        for variant in variants:
            report = Path(f"docs/SNIPER_{q}_METRICS__{variant}.md")
            if report.exists():
                required_reports.append(f"✅ {q} ({variant})")
            else:
                # Check for timestamped version
                timestamped = list(Path("docs").glob(f"SNIPER_{q}_METRICS__{variant}_*.md"))
                if timestamped:
                    required_reports.append(f"✅ {q} ({variant}) (timestamped)")
                else:
                    required_reports.append(f"❌ {q} ({variant})")

    for status in required_reports:
        log.info(f"   {status}")

    delta_report = Path("docs/SNIPER_2025_DELTA_BASELINE_vs_GUARDED.md")
    if delta_report.exists():
        log.info(f"   ✅ Delta Report")
    else:
        timestamped = list(Path("docs").glob("SNIPER_2025_DELTA_BASELINE_vs_GUARDED_*.md"))
        if timestamped:
            log.info(f"   ✅ Delta Report (timestamped)")
        else:
            log.error(f"   ❌ Delta Report")

    log.info("")
    log.info("=" * 80)
    log.info("✅ All quarters and variants processed!")
    log.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    main()

