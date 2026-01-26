#!/usr/bin/env python3
"""
Evaluate ENTRY_V10 vs ENTRY_V9 offline.

Compares V9 and V10 models on test dataset and generates comparison report.

Input:
    - Test dataset: data/entry_v10/entry_v10_val.parquet
    - V10 model: models/entry_v10/entry_v10_transformer.pt
    - V10 metadata: models/entry_v10/entry_v10_transformer_meta.json
    - V9 model dir: gx1/models/entry_v9/nextgen_2020_2025_clean (optional)

Output:
    - reports/rl/entry_v10/ENTRY_V10_VS_V9_COMPARE.md (V10 vs V9 comparison)
    - reports/rl/entry_v10/ENTRY_V10_DEEP_ANALYSIS.md (deep breakdown per session/regime)

Usage:
    # Full comparison V10 vs V9
    python -m gx1.rl.entry_v10.evaluate_entry_v10_offline \
        --test-parquet data/entry_v10/entry_v10_dataset_val.parquet \
        --v10-model models/entry_v10/entry_v10_transformer.pt \
        --v10-meta models/entry_v10/entry_v10_transformer_meta.json \
        --compare-v9 \
        --output-report reports/rl/entry_v10/ENTRY_V10_VS_V9_COMPARE.md

    # Deep breakdown
    python -m gx1.rl.entry_v10.evaluate_entry_v10_offline \
        --test-parquet data/entry_v10/entry_v10_dataset_val.parquet \
        --v10-model models/entry_v10/entry_v10_transformer.pt \
        --v10-meta models/entry_v10/entry_v10_transformer_meta.json \
        --session-breakdown \
        --regime-breakdown \
        --output-report reports/rl/entry_v10/ENTRY_V10_DEEP_ANALYSIS.md
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
from joblib import load as joblib_load
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, precision_score, recall_score
from torch.utils.data import DataLoader

from gx1.rl.entry_v10.dataset_v10 import EntryV10Dataset
from gx1.models.entry_v10.entry_v10_hybrid_transformer import EntryV10HybridTransformer
from gx1.models.entry_v9.entry_v9_transformer import build_entry_v9_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# V9 model directory (baseline)
V9_MODEL_DIR = Path("gx1/models/entry_v9/nextgen_2020_2025_clean")


def load_v9_model(model_dir: Path, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any], Any, Any]:
    """Load ENTRY_V9 Transformer model with scalers."""
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"ENTRY_V9 model directory not found: {model_dir}")
    
    log.info(f"[V9] Loading model from {model_dir}")
    
    # Load metadata
    meta_path = model_dir / "meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    # Load feature metadata
    feature_meta_path = model_dir / "entry_v9_feature_meta.json"
    with open(feature_meta_path, "r") as f:
        feature_meta = json.load(f)
    
    seq_feat_names = feature_meta.get("seq_features", [])
    snap_feat_names = feature_meta.get("snap_features", [])
    
    log.info(f"[V9] Features: {len(seq_feat_names)} seq, {len(snap_feat_names)} snap")
    
    # Build model config
    model_cfg = meta.get("model_config", {})
    if not model_cfg:
        model_cfg = {
            "name": "entry_v9",
            "seq_input_dim": len(seq_feat_names),
            "snap_input_dim": len(snap_feat_names),
            "max_seq_len": 30,
            "seq_cfg": {"d_model": 128, "n_heads": 4, "num_layers": 3, "dim_feedforward": 384, "dropout": 0.1},
            "snap_cfg": {"hidden_dims": [256, 128, 64], "use_layernorm": True, "dropout": 0.0},
            "regime_cfg": {"embedding_dim": 16},
            "fusion_hidden_dim": 128,
            "fusion_dropout": 0.1,
            "head_hidden_dim": 64,
        }
    
    model = build_entry_v9_model({"model": model_cfg})
    
    # Load weights
    model_path = model_dir / "model.pt"
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    
    # Load scalers
    seq_scaler = None
    snap_scaler = None
    seq_scaler_path = model_dir / "seq_scaler.joblib"
    snap_scaler_path = model_dir / "snap_scaler.joblib"
    if seq_scaler_path.exists():
        seq_scaler = joblib_load(seq_scaler_path)
    if snap_scaler_path.exists():
        snap_scaler = joblib_load(snap_scaler_path)
    
    return model, feature_meta, seq_scaler, snap_scaler


def extract_v9_features_from_v10_dataset(
    df_v10: pd.DataFrame,
    v9_feature_meta: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract V9-compatible features from V10 dataset.
    
    V10 has 16 seq features (13 V9 + 3 XGB), 88 snap features (85 V9 + 3 XGB).
    We need to extract the first 13 seq and first 85 snap features.
    """
    # V9 expects: 13 seq features, 85 snap features
    # V10 has: 16 seq features (13 V9 + 3 XGB), 88 snap features (85 V9 + 3 XGB)
    
    # Load actual sequence and snapshot arrays
    seq_list = []
    snap_list = []
    
    for idx in range(len(df_v10)):
        row = df_v10.iloc[idx]
        
        # Extract sequence (30, 16) -> (30, 13)
        # seq can be list, array, or object array from parquet
        seq = row["seq"]
        
        # Handle object arrays (stored as lists in parquet)
        if isinstance(seq, np.ndarray) and seq.dtype == object:
            seq = seq.tolist()
        
        # Convert to list first, then to array
        if isinstance(seq, list):
            # Try to convert directly
            try:
                seq_arr = np.array(seq, dtype=np.float32)
            except (ValueError, TypeError):
                # If that fails, flatten nested lists
                seq_flat = []
                for item in seq:
                    if isinstance(item, (list, np.ndarray)):
                        seq_flat.extend(np.array(item, dtype=np.float32).flatten())
                    else:
                        seq_flat.append(float(item))
                seq_arr = np.array(seq_flat, dtype=np.float32)
            seq = seq_arr
        elif isinstance(seq, np.ndarray):
            seq = seq.astype(np.float32)
        else:
            raise ValueError(f"Unexpected seq type: {type(seq)}")
        
        # Flatten and reshape to (30, 16)
        seq_flat = seq.flatten()
        if len(seq_flat) == 30 * 16:
            seq = seq_flat.reshape(30, 16)
        else:
            raise ValueError(f"Expected 480 elements for seq, got {len(seq_flat)}")
        
        seq_v9 = seq[:, :13]  # Take first 13 features
        seq_list.append(seq_v9)
        
        # Extract snapshot (88,) -> (85,)
        # snap is stored as list (converted from array with tolist()) in parquet
        snap = row["snap"]
        # Convert list to array first
        if isinstance(snap, list):
            snap = np.array(snap, dtype=np.float32)
        elif isinstance(snap, np.ndarray):
            snap = snap.astype(np.float32)
        
        # Flatten
        snap = snap.flatten()
        if len(snap) != 88:
            if len(snap) > 88:
                snap = snap[:88]
            else:
                # Pad if needed
                snap = np.pad(snap, (0, 88 - len(snap)), 'constant', constant_values=0.0)
        
        snap_v9 = snap[:85]  # Take first 85 features
        snap_list.append(snap_v9)
    
    return np.array(seq_list), np.array(snap_list)


def evaluate_model_v9(
    model: torch.nn.Module,
    dataset: EntryV10Dataset,
    v9_feature_meta: Dict[str, Any],
    seq_scaler: Any,
    snap_scaler: Any,
    device: torch.device,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate V9 model on V10 dataset (converted to V9 format).
    
    ⚠️ BASELINE DISABLED – requires proper V9 model before enabling
    ⚠️ Current implementation produces NaN → 0.5 (dummy predictions)
    ⚠️ See reports/rl/entry_v10/V9_INVALID_NOTE.md for details
    ⚠️ DO NOT use V9 baseline until validated as working
    """
    # Load dataset as DataFrame for feature extraction
    df_v10 = pd.read_parquet(dataset.parquet_path)
    
    # Extract V9-compatible features
    seq_arrays, snap_arrays = extract_v9_features_from_v10_dataset(df_v10, v9_feature_meta)
    
    # Apply scalers if available
    if seq_scaler is not None:
        # Reshape for scaler: (n_samples, seq_len, n_features) -> (n_samples * seq_len, n_features)
        orig_shape = seq_arrays.shape
        seq_flat = seq_arrays.reshape(-1, orig_shape[-1])
        seq_scaled = seq_scaler.transform(seq_flat)
        seq_arrays = seq_scaled.reshape(orig_shape)
    
    if snap_scaler is not None:
        snap_arrays = snap_scaler.transform(snap_arrays)
    
    model.eval()
    all_preds = []
    all_targets = []
    
    # Process in batches
    n_samples = len(df_v10)
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_seq = seq_arrays[start_idx:end_idx]
        batch_snap = snap_arrays[start_idx:end_idx]
        
        # Get regime IDs from dataset
        session_ids = df_v10.iloc[start_idx:end_idx]["session_id"].values
        vol_regime_ids = df_v10.iloc[start_idx:end_idx]["vol_regime_id"].values
        trend_regime_ids = df_v10.iloc[start_idx:end_idx]["trend_regime_id"].values
        y_targets = df_v10.iloc[start_idx:end_idx]["y_direction"].values
        
        # V9 expects trend_regime_id: 0=UP, 1=DOWN, 2=RANGE/NEUTRAL
        # V10 uses: 0=UP, 1=DOWN, 2=NEUTRAL, 3=CHOP, 4=MIXED
        # Map V10 regime IDs to V9 format (clamp to 0-2)
        trend_regime_ids = np.clip(trend_regime_ids, 0, 2)
        
        # Convert to tensors
        seq_t = torch.tensor(batch_seq, dtype=torch.float32).to(device)
        snap_t = torch.tensor(batch_snap, dtype=torch.float32).to(device)
        session_t = torch.tensor(session_ids, dtype=torch.long).to(device)
        vol_t = torch.tensor(vol_regime_ids, dtype=torch.long).to(device)
        trend_t = torch.tensor(trend_regime_ids, dtype=torch.long).to(device)
        
            with torch.no_grad():
            output = model(seq_t, snap_t, session_t, vol_t, trend_t)
            direction_logit = output["direction_logit"].squeeze(-1)
            probs = torch.sigmoid(direction_logit)
            
            # V10.1 design: HARD FEIL on NaN/Inf - no fallback (no dummy 0.5)
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                raise RuntimeError(
                    "CRITICAL: NaN/Inf in V9 model predictions. "
                    "NO FALLBACK - HARD FEIL."
                )
            
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(y_targets)
    
    return np.array(all_preds), np.array(all_targets)


def evaluate_model(
    model: torch.nn.Module,
    dataset: EntryV10Dataset,
    device: torch.device,
    batch_size: int = 256,
    is_v9: bool = False,
    v9_feature_meta: Optional[Dict[str, Any]] = None,
    seq_scaler: Optional[Any] = None,
    snap_scaler: Optional[Any] = None,
) -> Dict[str, Any]:
    """Evaluate model on dataset and return metrics."""
    if is_v9:
        all_preds, all_targets = evaluate_model_v9(
            model, dataset, v9_feature_meta, seq_scaler, snap_scaler, device, batch_size
        )
    else:
        # V10 evaluation
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        model.eval()
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
                        "CRITICAL: NaN/Inf in evaluation predictions. "
                        "NO FALLBACK - HARD FEIL."
                    )
                
                probs_np = probs.cpu().numpy()
                # Check for constant predictions (std < 1e-6)
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
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        session_ids = np.array(session_ids)
        vol_regime_ids = np.array(vol_regime_ids)
        trend_regime_ids = np.array(trend_regime_ids)
    
    # Compute metrics
    auc = roc_auc_score(all_targets, all_preds)
    ap = average_precision_score(all_targets, all_preds)
    
    # Precision/Recall at optimal threshold
    precision, recall, thresholds = precision_recall_curve(all_targets, all_preds)
    # Find threshold that maximizes F1
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    pred_binary = (all_preds >= optimal_threshold).astype(int)
    prec = precision_score(all_targets, pred_binary, zero_division=0)
    rec = recall_score(all_targets, pred_binary, zero_division=0)
    
    # Binning analysis (4 bins)
    n_bins = 4
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(all_preds, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    bin_stats = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_mean_pred = all_preds[mask].mean()
            bin_mean_target = all_targets[mask].mean()
            bin_count = mask.sum()
            bin_stats.append({
                "bin": i + 1,
                "pred_mean": float(bin_mean_pred),
                "target_mean": float(bin_mean_target),
                "count": int(bin_count),
            })
    
    result = {
        "auc": float(auc),
        "ap": float(ap),
        "precision": float(prec),
        "recall": float(rec),
        "optimal_threshold": float(optimal_threshold),
        "bin_stats": bin_stats,
        "all_preds": all_preds,
        "all_targets": all_targets,
    }
    
    if not is_v9:
        result["session_ids"] = session_ids
        result["vol_regime_ids"] = vol_regime_ids
        result["trend_regime_ids"] = trend_regime_ids
    
    return result


def compute_session_breakdown(metrics: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Compute metrics per session (EU/OVERLAP/US)."""
    session_map = {0: "EU", 1: "OVERLAP", 2: "US"}
    
    breakdown = {}
    for session_id, session_name in session_map.items():
        mask = metrics["session_ids"] == session_id
        if mask.sum() == 0:
            continue
        
        preds = metrics["all_preds"][mask]
        targets = metrics["all_targets"][mask]
        
        if targets.sum() == 0 or targets.sum() == len(targets):
            continue
        
        auc = roc_auc_score(targets, preds)
        ap = average_precision_score(targets, preds)
        
        precision, recall, thresholds = precision_recall_curve(targets, preds)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        pred_binary = (preds >= optimal_threshold).astype(int)
        prec = precision_score(targets, pred_binary, zero_division=0)
        rec = recall_score(targets, pred_binary, zero_division=0)
        
        breakdown[session_name] = {
            "auc": float(auc),
            "ap": float(ap),
            "precision": float(prec),
            "recall": float(rec),
            "count": int(mask.sum()),
        }
    
    return breakdown


def compute_regime_breakdown(metrics: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Compute metrics per regime (trend × vol)."""
    trend_map = {0: "UP", 1: "DOWN", 2: "NEUTRAL"}
    vol_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "EXTREME"}
    
    breakdown = {}
    for trend_id, trend_name in trend_map.items():
        for vol_id, vol_name in vol_map.items():
            regime_name = f"{trend_name}×{vol_name}"
            mask = (metrics["trend_regime_ids"] == trend_id) & (metrics["vol_regime_ids"] == vol_id)
            
            if mask.sum() < 50:  # Skip regimes with too few samples
                continue
            
            preds = metrics["all_preds"][mask]
            targets = metrics["all_targets"][mask]
            
            if targets.sum() == 0 or targets.sum() == len(targets):
                continue
            
            try:
                auc = roc_auc_score(targets, preds)
                ap = average_precision_score(targets, preds)
                
                precision, recall, thresholds = precision_recall_curve(targets, preds)
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
                optimal_idx = np.argmax(f1_scores)
                optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
                
                pred_binary = (preds >= optimal_threshold).astype(int)
                prec = precision_score(targets, pred_binary, zero_division=0)
                rec = recall_score(targets, pred_binary, zero_division=0)
                
                breakdown[regime_name] = {
                    "auc": float(auc),
                    "ap": float(ap),
                    "precision": float(prec),
                    "recall": float(rec),
                    "count": int(mask.sum()),
                }
            except ValueError:
                continue
    
    return breakdown


def generate_comparison_report(
    v10_metrics: Dict[str, Any],
    v9_metrics: Optional[Dict[str, Any]],
    test_dataset_size: int,
    seq_len: int,
    output_path: Path,
) -> None:
    """Generate comparison report."""
    lines = [
        "# ENTRY_V10 vs ENTRY_V9 Comparison Report",
        "",
        f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Test Dataset",
        f"- Samples: {test_dataset_size:,}",
        f"- Sequence length: {seq_len}",
        "",
        "## ENTRY_V10 Performance",
        "",
        "### Overall Metrics",
        f"- **AUC:** {v10_metrics['auc']:.4f}",
        f"- **Average Precision:** {v10_metrics['ap']:.4f}",
        f"- **Precision:** {v10_metrics['precision']:.4f}",
        f"- **Recall:** {v10_metrics['recall']:.4f}",
        f"- **Optimal Threshold:** {v10_metrics['optimal_threshold']:.4f}",
        "",
        "### Binning Analysis",
        "",
        "| Bin | Pred Mean | Target Mean | Count |",
        "|-----|-----------|------------|-------|",
    ]
    
    for stat in v10_metrics["bin_stats"]:
        lines.append(
            f"| {stat['bin']} | {stat['pred_mean']:.4f} | {stat['target_mean']:.4f} | {stat['count']:,} |"
        )
    
    if v9_metrics:
        lines.extend([
            "",
            "## ENTRY_V9 Baseline Performance",
            "",
            "### Overall Metrics",
            f"- **AUC:** {v9_metrics['auc']:.4f}",
            f"- **Average Precision:** {v9_metrics['ap']:.4f}",
            f"- **Precision:** {v9_metrics['precision']:.4f}",
            f"- **Recall:** {v9_metrics['recall']:.4f}",
            f"- **Optimal Threshold:** {v9_metrics['optimal_threshold']:.4f}",
            "",
            "### Binning Analysis",
            "",
            "| Bin | Pred Mean | Target Mean | Count |",
            "|-----|-----------|------------|-------|",
        ])
        
        for stat in v9_metrics["bin_stats"]:
            lines.append(
                f"| {stat['bin']} | {stat['pred_mean']:.4f} | {stat['target_mean']:.4f} | {stat['count']:,} |"
            )
        
        # Comparison
        lines.extend([
            "",
            "## Comparison",
            "",
            "| Metric | V10 | V9 | Delta |",
            "|--------|-----|----|----|",
            f"| AUC | {v10_metrics['auc']:.4f} | {v9_metrics['auc']:.4f} | {v10_metrics['auc'] - v9_metrics['auc']:+.4f} |",
            f"| AP | {v10_metrics['ap']:.4f} | {v9_metrics['ap']:.4f} | {v10_metrics['ap'] - v9_metrics['ap']:+.4f} |",
            f"| Precision | {v10_metrics['precision']:.4f} | {v9_metrics['precision']:.4f} | {v10_metrics['precision'] - v9_metrics['precision']:+.4f} |",
            f"| Recall | {v10_metrics['recall']:.4f} | {v9_metrics['recall']:.4f} | {v10_metrics['recall'] - v9_metrics['recall']:+.4f} |",
            "",
        ])
        
        # Conclusion
        auc_delta = v10_metrics['auc'] - v9_metrics['auc']
        if auc_delta > 0.01:
            conclusion = "✅ V10 outperforms V9 baseline"
        elif auc_delta < -0.01:
            conclusion = "❌ V10 underperforms V9 baseline"
        else:
            conclusion = "➖ V10 performs similarly to V9 baseline"
        
        lines.extend([
            "## Conclusion",
            "",
            conclusion,
            "",
            f"- AUC delta: {auc_delta:+.4f}",
            f"- V10 uses XGBoost-annotated features (13 seq + 3 XGB channels, 85 snap + 3 XGB-now)",
            f"- V9 uses standard features (13 seq, 85 snap)",
            "",
        ])
    else:
        lines.extend([
            "",
            "## Notes",
            "",
            "- V9 comparison not enabled",
            "- V10 uses XGBoost-annotated sequences (13 seq + 3 XGB channels)",
            "- V10 uses XGBoost-annotated snapshots (85 snap + 3 XGB-now)",
            "",
        ])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    log.info(f"Comparison report saved to {output_path}")


def generate_deep_analysis_report(
    v10_metrics: Dict[str, Any],
    session_breakdown: Dict[str, Dict[str, float]],
    regime_breakdown: Dict[str, Dict[str, float]],
    test_dataset_size: int,
    seq_len: int,
    output_path: Path,
    variant: str = "v10",
) -> None:
    """Generate deep analysis report with session and regime breakdown."""
    # Determine title and model description based on variant
    if variant == "v10_1":
        title = "# ENTRY_V10.1 Evaluation Report (seq_len=90)"
        model_desc = "ENTRY_V10.1 HYBRID (variant=v10_1, seq_len=90)"
        arch_desc = "Transformer: num_layers=6, d_model=256, dim_feedforward=1024"
    else:
        title = "# ENTRY_V10 Deep Analysis Report"
        model_desc = "ENTRY_V10 HYBRID"
        arch_desc = "Transformer: num_layers=3, d_model=128, dim_feedforward=512"
    
    lines = [
        title,
        "",
        f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Model Configuration",
        f"- Model: {model_desc}",
        f"- Architecture: {arch_desc}",
        f"- Sequence length: {seq_len} bars",
        "",
        "## Test Dataset",
        f"- Samples: {test_dataset_size:,}",
        f"- Sequence length: {seq_len}",
        "",
        "## Overall Performance",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| AUC | {v10_metrics['auc']:.4f} |",
        f"| Average Precision | {v10_metrics['ap']:.4f} |",
        f"| Precision | {v10_metrics['precision']:.4f} |",
        f"| Recall | {v10_metrics['recall']:.4f} |",
        "",
        "## Session Breakdown",
        "",
        "| Session | AUC | AP | Precision | Recall | Count |",
        "|---------|-----|----|-----------|----|-------|",
    ]
    
    for session_name in ["EU", "OVERLAP", "US"]:
        if session_name in session_breakdown:
            stats = session_breakdown[session_name]
            lines.append(
                f"| {session_name} | {stats['auc']:.4f} | {stats['ap']:.4f} | "
                f"{stats['precision']:.4f} | {stats['recall']:.4f} | {stats['count']:,} |"
            )
    
    lines.extend([
        "",
        "## Regime Breakdown",
        "",
        "| Regime | AUC | AP | Precision | Recall | Count |",
        "|--------|-----|----|-----------|----|-------|",
    ])
    
    # Sort regimes by AUC
    sorted_regimes = sorted(regime_breakdown.items(), key=lambda x: x[1]["auc"], reverse=True)
    for regime_name, stats in sorted_regimes:
        lines.append(
            f"| {regime_name} | {stats['auc']:.4f} | {stats['ap']:.4f} | "
            f"{stats['precision']:.4f} | {stats['recall']:.4f} | {stats['count']:,} |"
        )
    
    lines.extend([
        "",
        "## Insights",
        "",
        "### Best Performing Regimes",
    ])
    
    # Top 3 regimes
    for i, (regime_name, stats) in enumerate(sorted_regimes[:3], 1):
        lines.append(f"{i}. **{regime_name}**: AUC {stats['auc']:.4f}, {stats['count']:,} samples")
    
    lines.extend([
        "",
        "### Weakest Performing Regimes",
    ])
    
    # Bottom 3 regimes
    for i, (regime_name, stats) in enumerate(sorted_regimes[-3:], 1):
        lines.append(f"{i}. **{regime_name}**: AUC {stats['auc']:.4f}, {stats['count']:,} samples")
    
        if variant == "v10_1":
            lines.extend([
                "",
                "## Notes",
                "",
                "- V10.1 uses XGBoost-annotated features (13 seq + 3 XGB channels, 85 snap + 3 XGB-now)",
                "- V10.1 uses longer sequences (seq_len=90, ~7.5 hours on M5) for improved temporal context",
                "- Session breakdown shows performance per trading session (EU/OVERLAP/US)",
                "- Regime breakdown shows performance per trend×volatility combination",
                "- NO dummy/fallback values - HARD FEIL on NaN/Inf/constant predictions",
                "",
            ])
        else:
            lines.extend([
                "",
                "## Notes",
                "",
                "- V10 uses XGBoost-annotated features (13 seq + 3 XGB channels, 85 snap + 3 XGB-now)",
                "- Session breakdown shows performance per trading session",
                "- Regime breakdown shows performance per trend×volatility combination",
                "",
            ])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    log.info(f"Deep analysis report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ENTRY_V10 vs ENTRY_V9 offline")
    parser.add_argument("--test-parquet", type=Path, required=True, help="Test dataset path")
    parser.add_argument("--v10-model", type=Path, required=True, help="V10 model state_dict path")
    parser.add_argument("--v10-meta", type=Path, required=True, help="V10 metadata JSON path")
    parser.add_argument("--v9-model-dir", type=Path, default=V9_MODEL_DIR, help="V9 model directory")
    parser.add_argument("--output-report", type=Path, required=True, help="Output report path")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--seq-len", type=int, default=30, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--compare-v9", action="store_true", help="Enable V9 comparison")
    parser.add_argument("--session-breakdown", action="store_true", help="Enable session breakdown")
    parser.add_argument("--regime-breakdown", action="store_true", help="Enable regime breakdown")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    log.info(f"Using device: {device}")
    
    # Load V10 model
    log.info(f"Loading V10 model from {args.v10_model}")
    with open(args.v10_meta, "r", encoding="utf-8") as f:
        v10_meta = json.load(f)
    
    # Get variant from metadata (default to v10 for backward compatibility)
    variant = v10_meta.get("variant", "v10")
    log.info(f"Model variant: {variant}")
    
    v10_model = EntryV10HybridTransformer(
        seq_input_dim=16,
        snap_input_dim=88,
        max_seq_len=args.seq_len,
        variant=variant,
        enable_auxiliary_heads=True,
    ).to(device)
    v10_model.load_state_dict(torch.load(args.v10_model, map_location=device))
    log.info(f"V10 model loaded (variant={variant})")
    
    # Load test dataset
    log.info(f"Loading test dataset: {args.test_parquet}")
    test_dataset = EntryV10Dataset(args.test_parquet, seq_len=args.seq_len, device=str(device))
    
    # Evaluate V10
    log.info("Evaluating V10 model")
    v10_metrics = evaluate_model(v10_model, test_dataset, device, args.batch_size, is_v9=False)
    log.info(f"V10 AUC: {v10_metrics['auc']:.4f}, AP: {v10_metrics['ap']:.4f}")
    
    # Load and evaluate V9 if requested
    # ⚠️ BASELINE DISABLED – requires proper V9 model before enabling
    # ⚠️ Current V9 predictions are constant 0.5 (dummy) due to NaN conversion
    # ⚠️ See reports/rl/entry_v10/V9_INVALID_NOTE.md for details
    # ⚠️ DO NOT use V9 baseline until validated as working
    v9_metrics = None
    if args.compare_v9:
        log.warning("⚠️  V9 BASELINE IS DISABLED - predictions are invalid (constant 0.5)")
        log.warning("⚠️  See reports/rl/entry_v10/V9_INVALID_NOTE.md for details")
        log.info("Loading V9 model (WILL PRODUCE INVALID RESULTS)")
        v9_model, v9_feature_meta, seq_scaler, snap_scaler = load_v9_model(args.v9_model_dir, device)
        log.info("Evaluating V9 model")
        v9_metrics = evaluate_model(
            v9_model, test_dataset, device, args.batch_size,
            is_v9=True, v9_feature_meta=v9_feature_meta, seq_scaler=seq_scaler, snap_scaler=snap_scaler
        )
        log.warning(f"⚠️  V9 AUC: {v9_metrics['auc']:.4f}, AP: {v9_metrics['ap']:.4f} (INVALID - constant 0.5 predictions)")
    
    # Generate reports
    if args.compare_v9:
        log.info("Generating comparison report")
        generate_comparison_report(
            v10_metrics, v9_metrics, len(test_dataset), args.seq_len, args.output_report
        )
    
    if args.session_breakdown or args.regime_breakdown:
        session_breakdown = {}
        regime_breakdown = {}
        
        if args.session_breakdown:
            log.info("Computing session breakdown")
            session_breakdown = compute_session_breakdown(v10_metrics)
        
        if args.regime_breakdown:
            log.info("Computing regime breakdown")
            regime_breakdown = compute_regime_breakdown(v10_metrics)
        
        log.info("Generating deep analysis report")
        generate_deep_analysis_report(
            v10_metrics, session_breakdown, regime_breakdown,
            len(test_dataset), args.seq_len, args.output_report, variant=variant
        )
    
    if not args.compare_v9 and not args.session_breakdown and not args.regime_breakdown:
        # Default: simple report
        generate_comparison_report(v10_metrics, None, len(test_dataset), args.seq_len, args.output_report)


if __name__ == "__main__":
    main()
