#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replay Evaluation for GATED_FUSION - Pre-policy vs Post-policy Analysis

DEL 2: Produces artifacts with fixed structure:
- raw_signals_<run_id>.parquet
- policy_decisions_<run_id>.parquet
- attribution_<run_id>.json
- metrics_<run_id>.json
- summary_<run_id>.md

All files must contain:
- run_id
- git commit hash (if available)
- gated bundle path + sha256
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from gx1.execution.oanda_demo_runner import GX1DemoRunner

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_git_commit_hash() -> Optional[str]:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=workspace_root,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def flush_replay_eval_collectors(runner: GX1DemoRunner, collectors: Dict[str, Any], output_dir: Optional[Path] = None, partial: bool = False) -> None:
    """
    Flush replay eval collectors to disk.
    
    DEL 2: Writes artifacts to reports/replay_eval/GATED/ with fixed structure.
    DEL 3: If partial=True, writes checkpoint artifacts (safe to call multiple times).
    SWEEP OPTIMIZATION: Respects GX1_OUTPUT_MODE (minimal vs full).
    
    Args:
        runner: GX1DemoRunner instance
        collectors: Dict of collectors (raw_signals, policy_decisions, trade_outcomes)
        output_dir: Optional output directory (default: reports/replay_eval/GATED)
        partial: If True, this is a checkpoint flush (safe to call multiple times)
    """
    if not collectors:
        log.warning("[REPLAY_EVAL] No collectors to flush")
        return
    
    # SWEEP OPTIMIZATION: Check output mode
    import os
    output_mode_env = os.getenv("GX1_OUTPUT_MODE", "").lower()
    if output_mode_env not in ("minimal", "full"):
        # Default: check RUN_IDENTITY if available
        output_mode = "full"  # Default for backward compatibility
        if hasattr(runner, "run_identity") and runner.run_identity:
            output_mode = runner.run_identity.output_mode
    else:
        output_mode = output_mode_env
    
    # Get run_id from runner
    run_id = getattr(runner, "run_id", None) or datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory (use provided output_dir or default)
    output_dir = output_dir or Path("reports/replay_eval/GATED")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get git commit hash
    git_commit = get_git_commit_hash()
    
    # Get gated bundle path and sha256
    gated_bundle_path = None
    gated_bundle_sha256 = None
    if hasattr(runner, "entry_v10_ctx_cfg") and runner.entry_v10_ctx_cfg:
        bundle_dir = runner.entry_v10_ctx_cfg.get("bundle_dir")
        if bundle_dir:
            bundle_path = Path(bundle_dir)
            if bundle_path.exists():
                gated_bundle_path = str(bundle_path.resolve())
                # Try to find model_state_dict.pt for sha256
                model_file = bundle_path / "model_state_dict.pt"
                if model_file.exists():
                    gated_bundle_sha256 = compute_file_sha256(model_file)
    
    # Metadata for all artifacts
    metadata = {
        "run_id": run_id,
        "git_commit": git_commit,
        "gated_bundle_path": gated_bundle_path,
        "gated_bundle_sha256": gated_bundle_sha256,
        "timestamp": datetime.now().isoformat(),
    }
    
    # SWEEP OPTIMIZATION: In minimal mode, skip raw_signals, policy_decisions, trade_outcomes
    # But ALWAYS write metrics (contains PnL, trades, MaxDD - required for decision-making)
    
    # A) Raw signals (skip in minimal mode)
    if output_mode != "minimal":
        raw_collector = collectors.get("raw_signals")
        if raw_collector:
            raw_df = raw_collector.to_dataframe()
            if not raw_df.empty:
                raw_path = output_dir / f"raw_signals_{run_id}.parquet"
                raw_df.to_parquet(raw_path, index=False)
                log.info(f"[REPLAY_EVAL] Saved raw_signals: {raw_path} ({len(raw_df)} rows)")
                
                # Add metadata to parquet file (as JSON sidecar)
                metadata_path = output_dir / f"raw_signals_{run_id}.metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
            else:
                log.warning("[REPLAY_EVAL] Raw signals DataFrame is empty")
    else:
        log.debug("[REPLAY_EVAL] Skipping raw_signals (output_mode=minimal)")
    
    # B) Policy decisions (skip in minimal mode)
    if output_mode != "minimal":
        decision_collector = collectors.get("policy_decisions")
        if decision_collector:
            decisions_df = decision_collector.to_dataframe()
            if not decisions_df.empty:
                decisions_path = output_dir / f"policy_decisions_{run_id}.parquet"
                decisions_df.to_parquet(decisions_path, index=False)
                log.info(f"[REPLAY_EVAL] Saved policy_decisions: {decisions_path} ({len(decisions_df)} rows)")
                
                # Attribution summary
                attribution = decision_collector.get_attribution()
                attribution_path = output_dir / f"attribution_{run_id}.json"
                with open(attribution_path, "w") as f:
                    json.dump({**metadata, **attribution}, f, indent=2)
                log.info(f"[REPLAY_EVAL] Saved attribution: {attribution_path}")
            else:
                log.warning("[REPLAY_EVAL] Policy decisions DataFrame is empty")
    else:
        log.debug("[REPLAY_EVAL] Skipping policy_decisions (output_mode=minimal)")
    
    # C) Trade outcomes (skip in minimal mode, but ALWAYS compute metrics)
    outcome_collector = collectors.get("trade_outcomes")
    if outcome_collector:
        if output_mode != "minimal":
            outcomes_df = outcome_collector.to_dataframe()
            if not outcomes_df.empty:
                outcomes_path = output_dir / f"trade_outcomes_{run_id}.parquet"
                outcomes_df.to_parquet(outcomes_path, index=False)
                log.info(f"[REPLAY_EVAL] Saved trade_outcomes: {outcomes_path} ({len(outcomes_df)} rows)")
            else:
                log.warning("[REPLAY_EVAL] Trade outcomes DataFrame is empty")
        else:
            log.debug("[REPLAY_EVAL] Skipping trade_outcomes parquet (output_mode=minimal)")
        
        # ALWAYS write metrics (required for decision-making, even in minimal mode)
        metrics = outcome_collector.compute_metrics()
        metrics_path = output_dir / f"metrics_{run_id}.json"
        with open(metrics_path, "w") as f:
            json.dump({**metadata, **metrics}, f, indent=2)
        log.info(f"[REPLAY_EVAL] Saved metrics: {metrics_path} (output_mode={output_mode})")
    
    # Summary markdown (always write, even in minimal mode)
    summary_path = output_dir / f"summary_{run_id}.md"
    with open(summary_path, "w") as f:
        f.write(f"# Replay Evaluation Summary\n\n")
        f.write(f"**Run ID:** {run_id}\n")
        f.write(f"**Git Commit:** {git_commit or 'N/A'}\n")
        f.write(f"**Gated Bundle:** {gated_bundle_path or 'N/A'}\n")
        if gated_bundle_sha256:
            f.write(f"**Bundle SHA256:** {gated_bundle_sha256[:16]}...\n")
        f.write(f"\n## Metrics\n\n")
        if outcome_collector:
            metrics_for_summary = outcome_collector.compute_metrics()
            f.write(f"- **Trades:** {metrics_for_summary.get('n_trades', 0)}\n")
            f.write(f"- **Total PnL (bps):** {metrics_for_summary.get('total_pnl_bps', 0.0):.2f}\n")
            f.write(f"- **Mean PnL (bps):** {metrics_for_summary.get('mean_pnl_bps', 0.0):.2f}\n")
            f.write(f"- **Median PnL (bps):** {metrics_for_summary.get('median_pnl_bps', 0.0):.2f}\n")
            if "mae_bps" in metrics_for_summary:
                f.write(f"- **MAE (bps):** {metrics_for_summary['mae_bps']:.2f}\n")
            if "mfe_bps" in metrics_for_summary:
                f.write(f"- **MFE (bps):** {metrics_for_summary['mfe_bps']:.2f}\n")
            f.write(f"- **Max DD (bps):** {metrics_for_summary.get('max_dd', 0.0):.2f}\n")
            f.write(f"- **P1 Loss (bps):** {metrics_for_summary.get('p1_loss', 0.0):.2f}\n")
            f.write(f"- **P5 Loss (bps):** {metrics_for_summary.get('p5_loss', 0.0):.2f}\n")
    log.info(f"[REPLAY_EVAL] Saved summary: {summary_path}")
    
    # DEL 5: Run ghostbusters scan after flush (verify no V9 ghosts in artifacts)
    if not partial:  # Only run full scan after complete flush, not checkpoint
        try:
            from gx1.scripts.ghostbusters_scan import ghostbusters_scan
            
            chunk_id = getattr(runner, "chunk_id", None)
            scan_results = ghostbusters_scan(
                output_dir=output_dir,
                run_id=run_id,
                chunk_id=chunk_id,
                scan_log=False,  # Log scanning can be enabled separately if needed
            )
            
            if scan_results["status"] == "failed":
                error_msg = (
                    f"GHOSTBUSTERS_SCAN_FAILED: V9 ghosts detected in artifacts after flush. "
                    f"Violations: {scan_results['summary']['total_violations']}, "
                    f"Failed files: {scan_results['summary']['files_failed']}"
                )
                log.error(f"[REPLAY_EVAL] {error_msg}")
                raise RuntimeError(error_msg)
            else:
                log.info(f"[REPLAY_EVAL] ✅ Ghostbusters scan passed: {scan_results['summary']['files_passed']} files checked, no V9 references found")
        except ImportError:
            log.warning("[REPLAY_EVAL] Ghostbusters scan module not found - skipping V9 artifact scan")
        except Exception as e:
            log.error(f"[REPLAY_EVAL] Ghostbusters scan failed (non-fatal): {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description="Replay evaluation for GATED_FUSION")
    parser.add_argument("--policy", type=Path, required=True, help="Policy YAML path")
    parser.add_argument("--data", type=Path, required=True, help="Input data (CSV/parquet)")
    # DEL 4A: Use GX1_DATA env vars for default paths
    default_reports_root = Path(os.getenv("GX1_REPORTS_ROOT", "../GX1_DATA/reports"))
    parser.add_argument("--output-dir", type=Path, default=default_reports_root / "replay_eval" / "GATED", help="Output directory")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID (default: timestamp)")
    
    args = parser.parse_args()
    
    # DEL 1: Verify GX1_GATED_FUSION_ENABLED=1
    gated_fusion_enabled = os.getenv("GX1_GATED_FUSION_ENABLED", "0") == "1"
    if not gated_fusion_enabled:
        raise RuntimeError(
            "BASELINE_DISABLED: GX1_GATED_FUSION_ENABLED is not '1'. "
            "Set GX1_GATED_FUSION_ENABLED=1 to run replay eval."
        )
    
    log.info(f"[REPLAY_EVAL] Starting replay evaluation")
    log.info(f"[REPLAY_EVAL] Policy: {args.policy}")
    log.info(f"[REPLAY_EVAL] Data: {args.data}")
    log.info(f"[REPLAY_EVAL] Output: {args.output_dir}")
    
    # Run replay (collectors are automatically initialized and flushed)
    runner = GX1DemoRunner(
        args.policy,
        replay_mode=True,
        fast_replay=False,  # Full replay for evaluation
    )
    
    # Run replay (collectors will be flushed automatically at end)
    runner.run_replay(args.data)
    
    log.info(f"[REPLAY_EVAL] Completed")


if __name__ == "__main__":
    main()
