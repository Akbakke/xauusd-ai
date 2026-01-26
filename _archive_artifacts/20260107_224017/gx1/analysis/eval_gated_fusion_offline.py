#!/usr/bin/env python3
"""
Offline evaluation of Gated Fusion on VAL + TEST datasets.

Evaluates trained checkpoint on validation and test sets, computes:
- Tail metrics (max DD, VaR95/VaR99, max loss)
- Performance by uncertainty deciles
- Performance by gate buckets
- Regime stability score
- Gate stats per session (EU/US/OVERLAP/ASIA)

Usage:
    python gx1/analysis/eval_gated_fusion_offline.py \
        --checkpoint models/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION \
        --val_data data/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION_val.parquet \
        --test_data data/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION_test.parquet \
        --policy_config gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_TRAIN_V10_CTX_GATED.yaml \
        --output_dir reports/eval
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from gx1.models.entry_v10.entry_v10_ctx_dataset import EntryV10CtxDataset
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


def load_model(checkpoint_dir: Path, device: torch.device) -> Tuple[nn.Module, Dict]:
    """Load trained model from checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)
    
    # Load metadata
    metadata_path = checkpoint_dir / "bundle_metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    # Load model state
    state_dict_path = checkpoint_dir / "model_state_dict.pt"
    state_dict = torch.load(state_dict_path, map_location=device)
    
    # Infer actual dimensions from state dict (metadata may be outdated)
    # seq_encoder.input_proj.weight has shape [d_model, seq_input_dim]
    if "seq_encoder.input_proj.weight" in state_dict:
        actual_seq_dim = state_dict["seq_encoder.input_proj.weight"].shape[1]
    else:
        actual_seq_dim = metadata.get("seq_input_dim", 13) + 3  # Default to +3 XGB channels
    
    # snap_encoder.mlp.0.weight has shape [hidden_dim, snap_input_dim]
    if "snap_encoder.mlp.0.weight" in state_dict:
        actual_snap_dim = state_dict["snap_encoder.mlp.0.weight"].shape[1]
    else:
        actual_snap_dim = metadata.get("snap_input_dim", 85) + 3  # Default to +3 XGB channels
    
    log.info(f"[EVAL] Model dimensions: seq={actual_seq_dim}, snap={actual_snap_dim} (metadata: seq={metadata.get('seq_input_dim')}, snap={metadata.get('snap_input_dim')})")
    
    # Create model with actual architecture
    model = EntryV10CtxHybridTransformer(
        seq_input_dim=actual_seq_dim,
        snap_input_dim=actual_snap_dim,
        max_seq_len=metadata.get("max_seq_len", 30),
        variant="v10_ctx",
        ctx_cat_dim=5,
        ctx_cont_dim=2,
        ctx_emb_dim=42,
        ctx_embedding_dim=8,
    ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # Update metadata with actual dimensions
    metadata["seq_input_dim"] = actual_seq_dim
    metadata["snap_input_dim"] = actual_snap_dim
    
    log.info(f"[EVAL] Loaded model from {checkpoint_dir}")
    return model, metadata


def compute_predictions(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Compute predictions and extract gate/uncertainty values."""
    all_predictions = {
        "y_direction": [],
        "y_early_move": [],
        "y_quality_score": [],
        "direction_logit": [],
        "early_move_logit": [],
        "quality_score": [],
        "gate": [],
        "session_id": [],
        "vol_regime_id": [],
        "uncertainty_score": [],
    }
    
    with torch.no_grad():
        for batch in dataloader:
            seq_x = batch["seq_x"].to(device)
            snap_x = batch["snap_x"].to(device)
            ctx_cat = batch["ctx_cat"].to(device)
            ctx_cont = batch["ctx_cont"].to(device)
            session_id = batch["session_id"].to(device)
            vol_regime_id = batch["vol_regime_id"].to(device)
            trend_regime_id = batch["trend_regime_id"].to(device)
            
            outputs = model(
                seq_x=seq_x,
                snap_x=snap_x,
                session_id=session_id,
                vol_regime_id=vol_regime_id,
                trend_regime_id=trend_regime_id,
                ctx_cat=ctx_cat,
                ctx_cont=ctx_cont,
            )
            
            # Extract predictions
            all_predictions["y_direction"].append(batch["y_direction"].numpy())
            all_predictions["y_early_move"].append(batch["y_early_move"].numpy())
            all_predictions["y_quality_score"].append(batch["y_quality_score"].numpy())
            all_predictions["direction_logit"].append(outputs["direction_logit"].squeeze(-1).cpu().numpy())
            all_predictions["early_move_logit"].append(outputs["early_move_logit"].squeeze(-1).cpu().numpy())
            all_predictions["quality_score"].append(outputs["quality_score"].squeeze(-1).cpu().numpy())
            all_predictions["gate"].append(outputs["gate"].squeeze(-1).cpu().numpy())
            all_predictions["session_id"].append(session_id.cpu().numpy())
            all_predictions["vol_regime_id"].append(vol_regime_id.cpu().numpy())
            
            # Extract uncertainty_score from seq_x[:, -1, 15]
            uncertainty = seq_x[:, -1, 15].cpu().numpy()  # Last timestep, uncertainty channel
            all_predictions["uncertainty_score"].append(uncertainty)
    
    # Concatenate all batches
    for key in all_predictions:
        all_predictions[key] = np.concatenate(all_predictions[key])
    
    return all_predictions


def compute_tail_metrics(pnl_bps: np.ndarray) -> Dict[str, float]:
    """Compute tail risk metrics."""
    if len(pnl_bps) == 0:
        return {
            "max_drawdown": 0.0,
            "var_95": 0.0,
            "var_99": 0.0,
            "max_loss": 0.0,
        }
    
    # Cumulative PnL
    cum_pnl = np.cumsum(pnl_bps)
    
    # Max drawdown
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - running_max
    max_drawdown = float(np.min(drawdown))
    
    # VaR (Value at Risk) - negative because loss
    var_95 = float(np.percentile(pnl_bps, 5))
    var_99 = float(np.percentile(pnl_bps, 1))
    
    # Max loss
    max_loss = float(np.min(pnl_bps))
    
    return {
        "max_drawdown": max_drawdown,
        "var_95": var_95,
        "var_99": var_99,
        "max_loss": max_loss,
    }


def compute_performance_by_decile(
    predictions: Dict[str, np.ndarray],
    metric_col: str,
    pnl_bps: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute performance by deciles of a metric (uncertainty, gate, etc.)."""
    values = predictions[metric_col]
    
    if pnl_bps is None:
        # Use quality_score as proxy for PnL
        pnl_bps = predictions.get("y_quality_score", np.zeros(len(values))) * 100.0
    
    # Create deciles
    deciles = np.percentile(values, np.linspace(0, 100, 11))
    
    results = {}
    for i in range(len(deciles) - 1):
        low = deciles[i]
        high = deciles[i + 1]
        mask = (values >= low) & (values < high) if i < len(deciles) - 2 else (values >= low)
        
        if mask.sum() > 0:
            bucket_pnl = pnl_bps[mask]
            results[f"decile_{i+1}"] = {
                "range_low": float(low),
                "range_high": float(high),
                "n_samples": int(mask.sum()),
                "mean_pnl": float(np.mean(bucket_pnl)),
                "median_pnl": float(np.median(bucket_pnl)),
                "win_rate": float((bucket_pnl > 0).mean()),
                "max_loss": float(np.min(bucket_pnl)),
                "std_pnl": float(np.std(bucket_pnl)),
            }
    
    return results


def compute_regime_stability(pnl_by_regime: Dict[str, np.ndarray]) -> float:
    """Compute regime stability score (0-1, higher is better)."""
    if not pnl_by_regime:
        return 0.0
    
    regime_means = {k: float(np.mean(v)) for k, v in pnl_by_regime.items() if len(v) > 0}
    
    if not regime_means:
        return 0.0
    
    mean_values = list(regime_means.values())
    if np.std(mean_values) == 0:
        return 1.0
    
    cv = np.std(mean_values) / (abs(np.mean(mean_values)) + 1e-6)
    stability_score = 1.0 / (1.0 + cv)
    
    return float(stability_score)


def evaluate_split(
    model: nn.Module,
    dataset: EntryV10CtxDataset,
    device: torch.device,
    split_name: str,
) -> Dict:
    """Evaluate model on a dataset split."""
    log.info(f"[EVAL] Evaluating {split_name} split...")
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )
    
    predictions = compute_predictions(model, dataloader, device)
    
    # Use quality_score as proxy for PnL (since we don't have actual PnL in dataset)
    # In production, this would come from actual trade outcomes
    pnl_bps = predictions["y_quality_score"] * 100.0  # Scale to bps
    
    # Tail metrics
    tail_metrics = compute_tail_metrics(pnl_bps)
    
    # Performance by uncertainty deciles
    perf_by_uncertainty = compute_performance_by_decile(
        predictions,
        "uncertainty_score",
        pnl_bps=pnl_bps,
    )
    
    # Performance by gate deciles
    perf_by_gate = compute_performance_by_decile(
        predictions,
        "gate",
        pnl_bps=pnl_bps,
    )
    
    # Regime stability
    pnl_by_regime = {}
    for regime_id in np.unique(predictions["vol_regime_id"]):
        mask = predictions["vol_regime_id"] == regime_id
        if mask.sum() > 0:
            pnl_by_regime[f"vol_{int(regime_id)}"] = pnl_bps[mask]
    
    regime_stability = compute_regime_stability(pnl_by_regime)
    
    # Gate stats per session
    session_names = {0: "EU", 1: "OVERLAP", 2: "US", 3: "ASIA"}
    gate_stats_per_session = {}
    performance_by_session = {}
    for session_id in np.unique(predictions["session_id"]):
        mask = predictions["session_id"] == session_id
        if mask.sum() > 0:
            session_gates = predictions["gate"][mask]
            session_pnl = pnl_bps[mask]
            session_name = session_names.get(int(session_id), f"UNKNOWN_{int(session_id)}")
            gate_stats_per_session[session_name] = {
                "mean": float(np.mean(session_gates)),
                "std": float(np.std(session_gates)),
                "min": float(np.min(session_gates)),
                "max": float(np.max(session_gates)),
                "p5": float(np.percentile(session_gates, 5)),
                "p95": float(np.percentile(session_gates, 95)),
                "n_samples": int(mask.sum()),
            }
            performance_by_session[session_name] = {
                "mean_pnl": float(np.mean(session_pnl)),
                "median_pnl": float(np.median(session_pnl)),
                "win_rate": float((session_pnl > 0).mean()),
                "std_pnl": float(np.std(session_pnl)),
                "max_loss": float(np.min(session_pnl)),
                "n_samples": int(mask.sum()),
            }
    
    # Gate responsiveness
    gate_corr_uncertainty = float(np.corrcoef(predictions["gate"], predictions["uncertainty_score"])[0, 1])
    
    results = {
        "split_name": split_name,
        "n_samples": int(len(pnl_bps)),
        "tail_metrics": tail_metrics,
        "performance_by_uncertainty": perf_by_uncertainty,
        "performance_by_gate": perf_by_gate,
        "regime_stability_score": regime_stability,
        "gate_stats_per_session": gate_stats_per_session,
        "performance_by_session": performance_by_session,
        "gate_responsiveness": {
            "corr_uncertainty": gate_corr_uncertainty,
            "gate_mean": float(np.mean(predictions["gate"])),
            "gate_std": float(np.std(predictions["gate"])),
            "gate_variance": float(np.var(predictions["gate"])),
        },
        "overall_metrics": {
            "mean_pnl": float(np.mean(pnl_bps)),
            "median_pnl": float(np.median(pnl_bps)),
            "win_rate": float((pnl_bps > 0).mean()),
            "std_pnl": float(np.std(pnl_bps)),
        },
    }
    
    return results


def generate_report(
    val_results: Dict,
    test_results: Dict,
    output_path: Path,
    baseline_val_results: Optional[Dict] = None,
    baseline_test_results: Optional[Dict] = None,
) -> None:
    """Generate markdown evaluation report."""
    with open(output_path, "w") as f:
        f.write("# Gated Fusion Evaluation Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # VAL split
        f.write("## VAL Split\n\n")
        _write_split_section(f, val_results)
        
        # TEST split
        f.write("\n## TEST Split\n\n")
        _write_split_section(f, test_results)
        
        # Baseline comparison (if provided)
        if baseline_val_results and baseline_test_results:
            f.write("\n## Baseline Comparison\n\n")
            f.write("### VAL Split Delta (Gated - Baseline)\n\n")
            _write_delta_section(f, val_results, baseline_val_results)
            
            f.write("\n### TEST Split Delta (Gated - Baseline)\n\n")
            _write_delta_section(f, test_results, baseline_test_results)
    
    log.info(f"[EVAL] ✅ Report saved: {output_path}")


def _write_split_section(f, results: Dict):
    """Write evaluation section for a split."""
    f.write("### Tail Risk Metrics\n\n")
    tail = results["tail_metrics"]
    f.write("| Metric | Value |\n")
    f.write("|--------|-------|\n")
    f.write(f"| Max Drawdown | {tail['max_drawdown']:.2f} bps |\n")
    f.write(f"| VaR (95th) | {tail['var_95']:.2f} bps |\n")
    f.write(f"| VaR (99th) | {tail['var_99']:.2f} bps |\n")
    f.write(f"| Max Loss | {tail['max_loss']:.2f} bps |\n")
    
    f.write("\n### Performance by Uncertainty Deciles\n\n")
    f.write("| Decile | Range | N | Mean PnL | Win Rate | Max Loss |\n")
    f.write("|--------|-------|---|----------|----------|----------|\n")
    for decile, perf in sorted(results["performance_by_uncertainty"].items()):
        f.write(f"| {decile} | [{perf['range_low']:.3f}, {perf['range_high']:.3f}] | "
               f"{perf['n_samples']} | {perf['mean_pnl']:.2f} bps | "
               f"{perf['win_rate']:.2%} | {perf['max_loss']:.2f} bps |\n")
    
    f.write("\n### Performance by Gate Deciles\n\n")
    f.write("| Decile | Range | N | Mean PnL | Win Rate | Max Loss |\n")
    f.write("|--------|-------|---|----------|----------|----------|\n")
    for decile, perf in sorted(results["performance_by_gate"].items()):
        f.write(f"| {decile} | [{perf['range_low']:.3f}, {perf['range_high']:.3f}] | "
               f"{perf['n_samples']} | {perf['mean_pnl']:.2f} bps | "
               f"{perf['win_rate']:.2%} | {perf['max_loss']:.2f} bps |\n")
    
    f.write(f"\n### Regime Stability Score\n\n")
    f.write(f"**Score:** {results['regime_stability_score']:.4f} (target: > 0.8)\n\n")
    
    f.write("\n### Gate Stats per Session\n\n")
    f.write("| Session | Mean | Std | Min | Max | P5 | P95 | N Samples |\n")
    f.write("|---------|------|-----|-----|-----|----|----|-----------|\n")
    for session, stats in sorted(results["gate_stats_per_session"].items()):
        f.write(f"| {session} | {stats['mean']:.4f} | {stats['std']:.4f} | "
               f"{stats['min']:.4f} | {stats['max']:.4f} | {stats['p5']:.4f} | "
               f"{stats['p95']:.4f} | {stats['n_samples']} |\n")
    
    f.write("\n### Performance by Session\n\n")
    f.write("| Session | Mean PnL | Win Rate | Max Loss | N Samples |\n")
    f.write("|---------|----------|----------|----------|-----------|\n")
    for session, perf in sorted(results.get("performance_by_session", {}).items()):
        f.write(f"| {session} | {perf['mean_pnl']:.2f} bps | {perf['win_rate']:.2%} | "
               f"{perf['max_loss']:.2f} bps | {perf['n_samples']} |\n")
    
    f.write("\n### Gate Responsiveness\n\n")
    gate_resp = results["gate_responsiveness"]
    f.write("| Metric | Value |\n")
    f.write("|--------|-------|\n")
    f.write(f"| Correlation (gate vs uncertainty) | {gate_resp['corr_uncertainty']:.4f} |\n")
    f.write(f"| Gate Mean | {gate_resp['gate_mean']:.4f} |\n")
    f.write(f"| Gate Std | {gate_resp['gate_std']:.4f} |\n")
    f.write(f"| Gate Variance | {gate_resp['gate_variance']:.4f} |\n")


def _write_delta_section(f, gated_results: Dict, baseline_results: Dict):
    """Write delta comparison section."""
    f.write("| Metric | Gated | Baseline | Delta |\n")
    f.write("|--------|-------|----------|-------|\n")
    
    # Tail metrics
    gated_tail = gated_results["tail_metrics"]
    baseline_tail = baseline_results["tail_metrics"]
    f.write(f"| Max Drawdown | {gated_tail['max_drawdown']:.2f} | {baseline_tail['max_drawdown']:.2f} | "
           f"{gated_tail['max_drawdown'] - baseline_tail['max_drawdown']:.2f} bps |\n")
    f.write(f"| VaR (95th) | {gated_tail['var_95']:.2f} | {baseline_tail['var_95']:.2f} | "
           f"{gated_tail['var_95'] - baseline_tail['var_95']:.2f} bps |\n")
    f.write(f"| VaR (99th) | {gated_tail['var_99']:.2f} | {baseline_tail['var_99']:.2f} | "
           f"{gated_tail['var_99'] - baseline_tail['var_99']:.2f} bps |\n")
    f.write(f"| Max Loss | {gated_tail['max_loss']:.2f} | {baseline_tail['max_loss']:.2f} | "
           f"{gated_tail['max_loss'] - baseline_tail['max_loss']:.2f} bps |\n")
    
    # Overall metrics
    gated_overall = gated_results["overall_metrics"]
    baseline_overall = baseline_results["overall_metrics"]
    f.write(f"| Mean PnL (proxy EV) | {gated_overall['mean_pnl']:.2f} | {baseline_overall['mean_pnl']:.2f} | "
           f"{gated_overall['mean_pnl'] - baseline_overall['mean_pnl']:.2f} bps |\n")
    f.write(f"| Win Rate | {gated_overall['win_rate']:.2%} | {baseline_overall['win_rate']:.2%} | "
           f"{gated_overall['win_rate'] - baseline_overall['win_rate']:.2%} |\n")
    
    # Regime stability
    f.write(f"| Regime Stability | {gated_results['regime_stability_score']:.4f} | "
           f"{baseline_results['regime_stability_score']:.4f} | "
           f"{gated_results['regime_stability_score'] - baseline_results['regime_stability_score']:.4f} |\n")
    
    # Gate responsiveness (gated only)
    if "gate_responsiveness" in gated_results:
        gated_gate = gated_results["gate_responsiveness"]
        baseline_gate = baseline_results.get("gate_responsiveness", {})
        f.write(f"| Gate Correlation (uncertainty) | {gated_gate.get('corr_uncertainty', 0.0):.4f} | "
               f"{baseline_gate.get('corr_uncertainty', 0.0):.4f} | "
               f"{gated_gate.get('corr_uncertainty', 0.0) - baseline_gate.get('corr_uncertainty', 0.0):.4f} |\n")
    
    # Performance by session (delta)
    f.write("\n#### Performance by Session (Delta: Gated - Baseline)\n\n")
    f.write("| Session | Gated Mean PnL | Baseline Mean PnL | Delta | Gated Win Rate | Baseline Win Rate |\n")
    f.write("|---------|----------------|-------------------|-------|----------------|-------------------|\n")
    
    gated_perf_by_session = gated_results.get("performance_by_session", {})
    baseline_perf_by_session = baseline_results.get("performance_by_session", {})
    
    all_sessions = set(gated_perf_by_session.keys()) | set(baseline_perf_by_session.keys())
    for session in sorted(all_sessions):
        gated_perf = gated_perf_by_session.get(session, {})
        baseline_perf = baseline_perf_by_session.get(session, {})
        
        gated_mean = gated_perf.get("mean_pnl", 0.0)
        baseline_mean = baseline_perf.get("mean_pnl", 0.0)
        gated_wr = gated_perf.get("win_rate", 0.0)
        baseline_wr = baseline_perf.get("win_rate", 0.0)
        
        f.write(f"| {session} | {gated_mean:.2f} bps | {baseline_mean:.2f} bps | "
               f"{gated_mean - baseline_mean:.2f} bps | {gated_wr:.2%} | {baseline_wr:.2%} |\n")
    
    # Gate bucket analysis (gated only - baseline has no meaningful gate)
    f.write("\n#### Gate Bucket Analysis (Gated Only)\n\n")
    f.write("| Gate Decile | Mean PnL | Win Rate | N Samples |\n")
    f.write("|-------------|----------|----------|-----------|\n")
    for decile, perf in sorted(gated_results.get("performance_by_gate", {}).items()):
        f.write(f"| {decile} | {perf['mean_pnl']:.2f} bps | {perf['win_rate']:.2%} | {perf['n_samples']} |\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate gated fusion on VAL + TEST")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--val_data", type=str, required=True, help="Validation dataset path")
    parser.add_argument("--test_data", type=str, required=True, help="Test dataset path")
    parser.add_argument("--policy_config", type=str, required=True, help="Policy config path")
    parser.add_argument("--feature_meta_path", type=str,
                       default="gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json")
    parser.add_argument("--seq_scaler_path", type=str,
                       default="gx1/models/entry_v9/nextgen_2020_2025_clean/seq_scaler.joblib")
    parser.add_argument("--snap_scaler_path", type=str,
                       default="gx1/models/entry_v9/nextgen_2020_2025_clean/snap_scaler.joblib")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--baseline_checkpoint", type=str, help="Baseline checkpoint (for comparison)")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/mps/cuda/cpu)")
    
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    log.info(f"[EVAL] Using device: {device}")
    
    # Load policy config
    with open(args.policy_config) as f:
        policy_config = yaml.safe_load(f)
    
    # Load feature meta
    with open(args.feature_meta_path) as f:
        feature_meta = json.load(f)
    
    seq_feature_names = feature_meta.get("seq_features", [])
    snap_feature_names = feature_meta.get("snap_features", [])
    
    # Load model
    model, metadata = load_model(Path(args.checkpoint), device)
    
    # Load datasets
    val_df = pd.read_parquet(args.val_data)
    test_df = pd.read_parquet(args.test_data)
    
    val_dataset = EntryV10CtxDataset(
        df=val_df,
        seq_feature_names=seq_feature_names,
        snap_feature_names=snap_feature_names,
        feature_meta_path=args.feature_meta_path,
        seq_scaler_path=Path(args.seq_scaler_path),
        snap_scaler_path=Path(args.snap_scaler_path),
        seq_len=30,
        lookback=30,
        policy_config=policy_config,
        warmup_bars=288,
        mode="train",
        allow_dummy_ctx=False,
    )
    
    test_dataset = EntryV10CtxDataset(
        df=test_df,
        seq_feature_names=seq_feature_names,
        snap_feature_names=snap_feature_names,
        feature_meta_path=args.feature_meta_path,
        seq_scaler_path=Path(args.seq_scaler_path),
        snap_scaler_path=Path(args.snap_scaler_path),
        seq_len=30,
        lookback=30,
        policy_config=policy_config,
        warmup_bars=288,
        mode="train",
        allow_dummy_ctx=False,
    )
    
    # Evaluate VAL
    val_results = evaluate_split(model, val_dataset, device, "VAL")
    
    # Evaluate TEST
    test_results = evaluate_split(model, test_dataset, device, "TEST")
    
    # Evaluate baseline (if provided)
    baseline_val_results = None
    baseline_test_results = None
    if args.baseline_checkpoint:
        log.info("[EVAL] Evaluating baseline checkpoint...")
        baseline_model, _ = load_model(Path(args.baseline_checkpoint), device)
        baseline_val_results = evaluate_split(baseline_model, val_dataset, device, "VAL")
        baseline_test_results = evaluate_split(baseline_model, test_dataset, device, "TEST")
    
    # Generate report
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"GATED_FUSION_FULLYEAR_2025_{timestamp}.md"
    
    generate_report(
        val_results,
        test_results,
        report_path,
        baseline_val_results,
        baseline_test_results,
    )
    
    # Save JSON results
    results_json = {
        "val": val_results,
        "test": test_results,
        "baseline_val": baseline_val_results,
        "baseline_test": baseline_test_results,
    }
    
    json_path = output_dir / f"GATED_FUSION_FULLYEAR_2025_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    
    log.info(f"[EVAL] ✅ Evaluation complete. Report: {report_path}")
    log.info(f"[EVAL] ✅ Results JSON: {json_path}")


if __name__ == "__main__":
    main()
