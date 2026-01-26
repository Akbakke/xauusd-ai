#!/usr/bin/env python3
"""
ENTRY_V10 Threshold Engine

Computes adaptive thresholds for V10/V10.1 predictions based on label quality analysis.
Uses quantile-based analysis to identify optimal entry bands.

Design:
    - NO dummy thresholds - if data is insufficient, returns "insufficient_data"
    - Based on quantile analysis: Q1 and Q5 typically have best PnL
    - Can compute per-regime/per-session thresholds
    - Outputs JSON and readable Markdown reports

Usage:
    from gx1.rl.entry_v10.thresholds_v10 import compute_v10_thresholds
    
    thresholds = compute_v10_thresholds(
        dataset_path="data/entry_v10/entry_v10_1_dataset_seq90.parquet",
        model_path="models/entry_v10/entry_v10_1_transformer.pt",
        meta_path="models/entry_v10/entry_v10_1_transformer_meta.json",
        output_json="reports/rl/entry_v10/ENTRY_V10_1_THRESHOLDS.json",
        output_md="reports/rl/entry_v10/ENTRY_V10_1_THRESHOLDS.md",
        per_regime=True,
        per_session=True,
    )
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from gx1.rl.entry_v10.dataset_v10 import EntryV10Dataset
from gx1.models.entry_v10.entry_v10_hybrid_transformer import EntryV10HybridTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_model_and_get_predictions(
    dataset_path: Path,
    model_path: Path,
    meta_path: Path,
    device: torch.device,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load V10 model and get predictions for threshold analysis.
    
    Returns:
        Tuple of (predictions, labels, session_ids, vol_regime_ids, trend_regime_ids)
    """
    log.info(f"Loading model from {model_path}")
    
    # Load metadata
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    variant = meta.get("variant", "v10")
    seq_len = meta.get("seq_len", 30)
    
    # Load model
    model = EntryV10HybridTransformer(
        seq_input_dim=meta.get("seq_feature_count", 16),
        snap_input_dim=meta.get("snap_feature_count", 88),
        max_seq_len=seq_len,
        variant=variant,
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    model.to(device)
    
    # Load dataset
    log.info(f"Loading dataset from {dataset_path}")
    dataset = EntryV10Dataset(dataset_path, seq_len=seq_len, device=str(device))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Get predictions
    log.info("Getting predictions...")
    all_preds = []
    all_targets = []
    session_ids = []
    vol_regime_ids = []
    trend_regime_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            seq_x = batch["seq_x"].to(device)
            snap_x = batch["snap_x"].to(device)
            session_id = batch["session_id"].to(device)
            vol_regime_id = batch["vol_regime_id"].to(device)
            trend_regime_id = batch["trend_regime_id"].to(device)
            y = batch.get("y_direction", batch.get("y")).to(device)
            
            output = model(seq_x, snap_x, session_id, vol_regime_id, trend_regime_id)
            direction_logit = output["direction_logit"].squeeze(-1)
            probs = torch.sigmoid(direction_logit)
            
            # V10.1 design: HARD FEIL on NaN/Inf - no fallback
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                raise RuntimeError(
                    "CRITICAL: NaN/Inf in threshold computation predictions. "
                    "NO FALLBACK - HARD FEIL."
                )
            
            probs_np = probs.cpu().numpy()
            # Check for constant predictions
            if len(probs_np) > 1 and np.std(probs_np) < 1e-6:
                raise RuntimeError(
                    "CRITICAL: Predictions are effectively constant "
                    f"(std={np.std(probs_np):.2e}). NO FALLBACK - HARD FEIL."
                )
            
            all_preds.extend(probs_np)
            all_targets.extend(y.cpu().numpy())
            session_ids.extend(session_id.cpu().numpy())
            vol_regime_ids.extend(vol_regime_id.cpu().numpy())
            trend_regime_ids.extend(trend_regime_id.cpu().numpy())
    
    return (
        np.array(all_preds),
        np.array(all_targets),
        np.array(session_ids),
        np.array(vol_regime_ids),
        np.array(trend_regime_ids),
    )


def compute_quantile_thresholds(
    preds: np.ndarray,
    labels: np.ndarray,
    n_quantiles: int = 5,
    min_samples: int = 100,
) -> Dict[str, Any]:
    """
    Compute quantile-based thresholds.
    
    Args:
        preds: Model predictions (p_long)
        labels: Binary labels (y_direction)
        n_quantiles: Number of quantiles (default: 5)
        min_samples: Minimum samples required per quantile
    
    Returns:
        Dictionary with threshold information, or {"status": "insufficient_data"} if data is too thin
    """
    if len(preds) < min_samples * n_quantiles:
        return {"status": "insufficient_data", "reason": f"Need at least {min_samples * n_quantiles} samples"}
    
    # Compute quantiles
    try:
        quantile_edges = np.quantile(preds, np.linspace(0, 1, n_quantiles + 1))
    except Exception as e:
        return {"status": "insufficient_data", "reason": f"Quantile computation failed: {e}"}
    
    # Create quantile labels
    quantile_labels = pd.qcut(preds, q=n_quantiles, labels=[f"Q{i+1}" for i in range(n_quantiles)], duplicates="drop")
    
    # Compute stats per quantile
    quantile_stats = []
    for q_idx in range(n_quantiles):
        q_name = f"Q{q_idx+1}"
        mask = quantile_labels == q_name
        
        if mask.sum() < min_samples:
            continue
        
        q_preds = preds[mask]
        q_labels = labels[mask]
        
        stats = {
            "quantile": q_name,
            "p_long_min": float(q_preds.min()),
            "p_long_max": float(q_preds.max()),
            "p_long_mean": float(q_preds.mean()),
            "p_long_median": float(np.median(q_preds)),
            "label_mean": float(q_labels.mean()),
            "n_samples": int(mask.sum()),
            "edge": float(q_labels.mean() - 0.5),  # Edge over 0.5 (random)
        }
        quantile_stats.append(stats)
    
    if len(quantile_stats) < 2:
        return {"status": "insufficient_data", "reason": "Too few quantiles with sufficient samples"}
    
    # Find high-band threshold (typically Q5 has best PnL)
    # Use 75th percentile as high-band entry threshold
    high_band = np.quantile(preds, 0.75)
    
    # Find low-band threshold (typically Q1 has best edge)
    low_band = np.quantile(preds, 0.25)
    
    # Mid-band blocking (Q2-Q4 often weaker)
    mid_low = np.quantile(preds, 0.4)
    mid_high = np.quantile(preds, 0.6)
    
    return {
        "status": "success",
        "quantiles": quantile_stats,
        "thresholds": {
            "high_band": float(high_band),
            "low_band": float(low_band),
            "mid_block_low": float(mid_low),
            "mid_block_high": float(mid_high),
        },
        "n_total_samples": int(len(preds)),
    }


def compute_v10_thresholds(
    dataset_path: str | Path,
    model_path: str | Path,
    meta_path: str | Path,
    output_json: Optional[str | Path] = None,
    output_md: Optional[str | Path] = None,
    per_regime: bool = False,
    per_session: bool = False,
    device: str = "cpu",
    min_samples_per_group: int = 500,
) -> Dict[str, Any]:
    """
    Compute V10/V10.1 thresholds based on label quality analysis.
    
    Args:
        dataset_path: Path to V10 dataset parquet
        model_path: Path to V10 model state_dict
        meta_path: Path to V10 metadata JSON
        output_json: Optional path to save JSON output
        output_md: Optional path to save Markdown report
        per_regime: If True, compute per-regime thresholds
        per_session: If True, compute per-session thresholds
        device: Device (cpu/cuda)
        min_samples_per_group: Minimum samples required per group (regime/session)
    
    Returns:
        Dictionary with threshold information
    
    Design:
        - NO dummy thresholds - returns "insufficient_data" if data is too thin
        - Based on quantile analysis from label quality findings
        - Q1 and Q5 typically have best PnL
        - Mid-quantiles (Q2-Q4) are often weaker
    """
    dataset_path = Path(dataset_path)
    model_path = Path(model_path)
    meta_path = Path(meta_path)
    
    device = torch.device(device)
    
    # Load predictions
    preds, labels, session_ids, vol_regime_ids, trend_regime_ids = load_model_and_get_predictions(
        dataset_path, model_path, meta_path, device
    )
    
    log.info(f"Loaded {len(preds)} predictions")
    
    # Global thresholds
    log.info("Computing global thresholds...")
    global_thresholds = compute_quantile_thresholds(preds, labels)
    
    result = {
        "global": global_thresholds,
        "per_session": {},
        "per_regime": {},
    }
    
    # Per-session thresholds
    if per_session:
        log.info("Computing per-session thresholds...")
        session_map = {0: "EU", 1: "OVERLAP", 2: "US"}
        
        for session_id, session_name in session_map.items():
            mask = session_ids == session_id
            if mask.sum() < min_samples_per_group:
                log.warning(f"Insufficient samples for session {session_name}: {mask.sum()}")
                result["per_session"][session_name] = {"status": "insufficient_data", "n_samples": int(mask.sum())}
                continue
            
            session_preds = preds[mask]
            session_labels = labels[mask]
            session_thresholds = compute_quantile_thresholds(session_preds, session_labels)
            result["per_session"][session_name] = session_thresholds
    
    # Per-regime thresholds (trend × vol)
    if per_regime:
        log.info("Computing per-regime thresholds...")
        trend_map = {0: "UP", 1: "DOWN", 2: "NEUTRAL"}
        vol_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "EXTREME"}
        
        for trend_id, trend_name in trend_map.items():
            for vol_id, vol_name in vol_map.items():
                regime_name = f"{trend_name}×{vol_name}"
                mask = (trend_regime_ids == trend_id) & (vol_regime_ids == vol_id)
                
                if mask.sum() < min_samples_per_group:
                    log.warning(f"Insufficient samples for regime {regime_name}: {mask.sum()}")
                    result["per_regime"][regime_name] = {"status": "insufficient_data", "n_samples": int(mask.sum())}
                    continue
                
                regime_preds = preds[mask]
                regime_labels = labels[mask]
                regime_thresholds = compute_quantile_thresholds(regime_preds, regime_labels)
                result["per_regime"][regime_name] = regime_thresholds
    
    # Save JSON output
    if output_json:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        log.info(f"Saved JSON thresholds to {output_json}")
    
    # Generate Markdown report
    if output_md:
        output_md = Path(output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        generate_threshold_report(result, output_md, dataset_path, model_path)
        log.info(f"Saved Markdown report to {output_md}")
    
    return result


def generate_threshold_report(
    thresholds: Dict[str, Any],
    output_path: Path,
    dataset_path: Path,
    model_path: Path,
) -> None:
    """Generate readable Markdown report from threshold results."""
    lines = [
        "# ENTRY_V10 Threshold Analysis",
        "",
        f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**Dataset:** `{dataset_path}`",
        f"**Model:** `{model_path}`",
        "",
    ]
    
    # Global thresholds
    lines.append("## Global Thresholds")
    lines.append("")
    
    global_thresh = thresholds.get("global", {})
    if global_thresh.get("status") == "success":
        thresh_vals = global_thresh.get("thresholds", {})
        lines.extend([
            "| Threshold | Value |",
            "|-----------|-------|",
            f"| High Band (75th percentile) | {thresh_vals.get('high_band', 0):.4f} |",
            f"| Low Band (25th percentile) | {thresh_vals.get('low_band', 0):.4f} |",
            f"| Mid Block Low (40th percentile) | {thresh_vals.get('mid_block_low', 0):.4f} |",
            f"| Mid Block High (60th percentile) | {thresh_vals.get('mid_block_high', 0):.4f} |",
            "",
        ])
        
        # Quantile stats table
        lines.extend([
            "### Quantile Statistics",
            "",
            "| Quantile | p_long Min | p_long Max | p_long Mean | Label Mean | Edge | N Samples |",
            "|----------|------------|------------|-------------|------------|------|-----------|",
        ])
        
        for q_stat in global_thresh.get("quantiles", []):
            lines.append(
                f"| {q_stat['quantile']} | {q_stat['p_long_min']:.4f} | {q_stat['p_long_max']:.4f} | "
                f"{q_stat['p_long_mean']:.4f} | {q_stat['label_mean']:.4f} | "
                f"{q_stat['edge']:+.4f} | {q_stat['n_samples']:,} |"
            )
    else:
        lines.append(f"**Status:** {global_thresh.get('status', 'unknown')}")
        if "reason" in global_thresh:
            lines.append(f"**Reason:** {global_thresh['reason']}")
        lines.append("")
    
    # Per-session thresholds
    if thresholds.get("per_session"):
        lines.extend([
            "",
            "## Per-Session Thresholds",
            "",
        ])
        
        for session_name, session_thresh in thresholds["per_session"].items():
            lines.append(f"### {session_name}")
            if session_thresh.get("status") == "success":
                thresh_vals = session_thresh.get("thresholds", {})
                lines.extend([
                    f"- High Band: {thresh_vals.get('high_band', 0):.4f}",
                    f"- Low Band: {thresh_vals.get('low_band', 0):.4f}",
                    f"- N Samples: {session_thresh.get('n_total_samples', 0):,}",
                    "",
                ])
            else:
                lines.append(f"- Status: {session_thresh.get('status', 'unknown')} (N={session_thresh.get('n_samples', 0)})")
                lines.append("")
    
    # Per-regime thresholds
    if thresholds.get("per_regime"):
        lines.extend([
            "",
            "## Per-Regime Thresholds",
            "",
            "| Regime | High Band | Low Band | N Samples | Status |",
            "|--------|-----------|----------|-----------|--------|",
        ])
        
        for regime_name, regime_thresh in sorted(thresholds["per_regime"].items()):
            if regime_thresh.get("status") == "success":
                thresh_vals = regime_thresh.get("thresholds", {})
                lines.append(
                    f"| {regime_name} | {thresh_vals.get('high_band', 0):.4f} | "
                    f"{thresh_vals.get('low_band', 0):.4f} | "
                    f"{regime_thresh.get('n_total_samples', 0):,} | Success |"
                )
            else:
                lines.append(
                    f"| {regime_name} | - | - | {regime_thresh.get('n_samples', 0):,} | "
                    f"{regime_thresh.get('status', 'unknown')} |"
                )
    
    lines.extend([
        "",
        "## Notes",
        "",
        "- Thresholds based on quantile analysis from label quality findings",
        "- Q1 and Q5 typically show best PnL",
        "- Mid-quantiles (Q2-Q4) are often weaker",
        "- If data is insufficient, threshold is marked as 'insufficient_data' (NO dummy values)",
        "- High band: Use for high-confidence entries (75th percentile)",
        "- Low band: Use for low-confidence entries (25th percentile)",
        "- Mid block: Range to potentially avoid (40th-60th percentile)",
        "",
    ])
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def compute_v10_1_edge_buckets(
    label_quality_json_path: Path,
    min_trades_per_bin: int = 30,
) -> Dict[str, Any]:
    """
    Compute edge-buckets for ENTRY_V10.1 based on label quality analysis.
    
    Reads JSON output from analyze_entry_v10_label_quality.py and builds edge-buckets
    per (session, regime, bin) combination.
    
    Args:
        label_quality_json_path: Path to JSON output from label quality analysis
        min_trades_per_bin: Minimum trades required per bin (default: 30)
    
    Returns:
        Dictionary with structure:
        {
            "EU": {
                "UP×LOW": [
                    {"p_min": 0.70, "p_max": 0.76, "expected_bps": 2.1, "n_trades": 85},
                    ...
                ],
                ...
            },
            "OVERLAP": { ... },
            "US": { ... }
        }
    
    Design:
        - NO smoothing or interpolation - direct use of mean PnL per bin
        - Bins with insufficient data (< min_trades_per_bin) are skipped
        - If total bins < 3 per session/regime → log warning but still write JSON
    """
    log.info(f"Loading label quality JSON from {label_quality_json_path}")
    
    if not label_quality_json_path.exists():
        raise FileNotFoundError(f"Label quality JSON not found: {label_quality_json_path}")
    
    with open(label_quality_json_path, "r", encoding="utf-8") as f:
        label_quality = json.load(f)
    
    # Extract quantile stats
    quantile_stats = label_quality.get("quantile_stats", [])
    if not quantile_stats:
        raise ValueError("No quantile_stats found in label quality JSON")
    
    log.info(f"Found {len(quantile_stats)} quantile bins")
    
    # Build edge buckets structure
    # Note: Label quality JSON may not have per-session/per-regime breakdown
    # For now, we'll build global buckets and mark them as applicable to all sessions/regimes
    # In a full implementation, we'd need per-session/per-regime quantile stats
    
    edge_buckets = {
        "EU": {},
        "OVERLAP": {},
        "US": {},
    }
    
    # Map quantiles to bins
    bins = []
    for q_stat in quantile_stats:
        if q_stat.get("status") == "insufficient_data":
            log.warning(f"Skipping quantile {q_stat.get('quantile')} - insufficient data")
            continue
        
        n_trades = q_stat.get("n_trades", 0)
        if n_trades < min_trades_per_bin:
            log.warning(
                f"Skipping quantile {q_stat.get('quantile')} - only {n_trades} trades "
                f"(< {min_trades_per_bin} required)"
            )
            continue
        
        # Extract expected_bps from mean PnL
        pnl_mean = q_stat.get("pnl_mean")
        if pnl_mean is None:
            log.warning(f"Skipping quantile {q_stat.get('quantile')} - no pnl_mean")
            continue
        
        bin_data = {
            "p_min": float(q_stat.get("p_long_min", 0.0)),
            "p_max": float(q_stat.get("p_long_max", 1.0)),
            "expected_bps": float(pnl_mean),
            "n_trades": int(n_trades),
            "quantile": q_stat.get("quantile", ""),
        }
        bins.append(bin_data)
    
    # For now, apply same buckets to all sessions/regimes
    # In a full implementation, we'd need per-session/per-regime quantile stats from label quality
    # This is a simplified version that uses global quantiles
    regime_map = {
        "UP×LOW": "UP×LOW",
        "UP×MEDIUM": "UP×MEDIUM",
        "UP×HIGH": "UP×HIGH",
        "DOWN×LOW": "DOWN×LOW",
        "DOWN×MEDIUM": "DOWN×MEDIUM",
        "DOWN×HIGH": "DOWN×HIGH",
        "NEUTRAL×LOW": "NEUTRAL×LOW",
        "NEUTRAL×MEDIUM": "NEUTRAL×MEDIUM",
        "NEUTRAL×HIGH": "NEUTRAL×HIGH",
    }
    
    for session in ["EU", "OVERLAP", "US"]:
        for regime_name in regime_map.keys():
            edge_buckets[session][regime_name] = bins.copy()
    
    # Check for insufficient bins
    total_bins = len(bins)
    if total_bins < 3:
        log.warning(
            f"Only {total_bins} bins with sufficient data (< 3 recommended). "
            "Consider lowering min_trades_per_bin or using more data."
        )
    
    log.info(f"Built edge buckets: {total_bins} bins per session/regime")
    
    return edge_buckets


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute ENTRY_V10 thresholds")
    parser.add_argument("--dataset", type=Path, help="V10 dataset path")
    parser.add_argument("--model", type=Path, help="V10 model path")
    parser.add_argument("--meta", type=Path, help="V10 metadata path")
    parser.add_argument("--output-json", type=Path, help="Output JSON path")
    parser.add_argument("--output-md", type=Path, help="Output Markdown path")
    parser.add_argument("--per-regime", action="store_true", help="Compute per-regime thresholds")
    parser.add_argument("--per-session", action="store_true", help="Compute per-session thresholds")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    
    # Edge buckets arguments
    parser.add_argument("--label-quality-json", type=Path, help="Label quality JSON path (for edge buckets)")
    parser.add_argument("--edge-buckets-output", type=Path, help="Output path for edge buckets JSON")
    parser.add_argument("--min-trades-per-bin", type=int, default=30, help="Minimum trades per bin")
    
    args = parser.parse_args()
    
    # If label quality JSON provided, compute edge buckets
    if args.label_quality_json:
        edge_buckets = compute_v10_1_edge_buckets(
            label_quality_json_path=args.label_quality_json,
            min_trades_per_bin=args.min_trades_per_bin,
        )
        
        if args.edge_buckets_output:
            output_path = Path(args.edge_buckets_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(edge_buckets, f, indent=2)
            log.info(f"Saved edge buckets to {output_path}")
        else:
            log.info("Edge buckets computed but no output path specified")
    elif args.dataset and args.model and args.meta:
        # Original threshold computation
        compute_v10_thresholds(
            dataset_path=args.dataset,
            model_path=args.model,
            meta_path=args.meta,
            output_json=args.output_json,
            output_md=args.output_md,
            per_regime=args.per_regime,
            per_session=args.per_session,
            device=args.device,
        )
    else:
        parser.print_help()

