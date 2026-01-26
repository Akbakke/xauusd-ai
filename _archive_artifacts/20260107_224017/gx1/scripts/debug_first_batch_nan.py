#!/usr/bin/env python3
"""
Debug script for NaN triage in FULLYEAR training.

Identifies whether NaN originates from:
(A) Dataset tensors
(B) Model forward pass
(C) Loss computation

Exit codes:
- 0: All checks passed
- 1: NAN_IN_INPUT
- 2: NAN_IN_FORWARD
- 3: NAN_IN_LOSS:<component>
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Debug NaN in first batch")
    parser.add_argument("--data", type=str, required=True, help="Path to train parquet")
    parser.add_argument("--policy_config", type=str, required=True, help="Path to policy config")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--feature_meta_path", type=str, 
                        default="gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json")
    parser.add_argument("--seq_scaler_path", type=str,
                        default="gx1/models/entry_v9/nextgen_2020_2025_clean/seq_scaler.joblib")
    parser.add_argument("--snap_scaler_path", type=str,
                        default="gx1/models/entry_v9/nextgen_2020_2025_clean/snap_scaler.joblib")
    parser.add_argument("--seq_len", type=int, default=30)
    return parser.parse_args()


def tensor_stats(t: torch.Tensor, name: str) -> dict:
    """Compute comprehensive stats for a tensor."""
    t_np = t.detach().cpu().numpy().astype(np.float64)
    flat = t_np.flatten()
    
    nan_count = int(np.isnan(flat).sum())
    inf_count = int(np.isinf(flat).sum())
    finite_mask = np.isfinite(flat)
    
    stats = {
        "name": name,
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "finite_count": int(finite_mask.sum()),
    }
    
    if finite_mask.sum() > 0:
        finite_vals = flat[finite_mask]
        stats["min"] = float(np.min(finite_vals))
        stats["max"] = float(np.max(finite_vals))
        stats["mean"] = float(np.mean(finite_vals))
        stats["std"] = float(np.std(finite_vals))
        stats["p0"] = float(np.percentile(finite_vals, 0))
        stats["p1"] = float(np.percentile(finite_vals, 1))
        stats["p50"] = float(np.percentile(finite_vals, 50))
        stats["p99"] = float(np.percentile(finite_vals, 99))
        stats["p100"] = float(np.percentile(finite_vals, 100))
    else:
        stats["min"] = stats["max"] = stats["mean"] = stats["std"] = None
        stats["p0"] = stats["p1"] = stats["p50"] = stats["p99"] = stats["p100"] = None
    
    return stats


def print_tensor_stats(stats: dict):
    """Pretty print tensor stats."""
    print(f"\n  [{stats['name']}]")
    print(f"    shape: {stats['shape']}, dtype: {stats['dtype']}")
    print(f"    nan_count: {stats['nan_count']}, inf_count: {stats['inf_count']}, finite_count: {stats['finite_count']}")
    if stats['min'] is not None:
        print(f"    min: {stats['min']:.6f}, max: {stats['max']:.6f}, mean: {stats['mean']:.6f}, std: {stats['std']:.6f}")
        print(f"    percentiles: p0={stats['p0']:.4f}, p1={stats['p1']:.4f}, p50={stats['p50']:.4f}, p99={stats['p99']:.4f}, p100={stats['p100']:.4f}")


def check_per_column_nan_inf(t: torch.Tensor, name: str, dim_names: list = None):
    """Check nan/inf per column (last dimension)."""
    t_np = t.detach().cpu().numpy()
    
    # Reshape to (N, features)
    if t_np.ndim == 3:
        # [B, L, F] -> [B*L, F]
        n_features = t_np.shape[-1]
        t_np = t_np.reshape(-1, n_features)
    elif t_np.ndim == 2:
        # [B, F]
        n_features = t_np.shape[-1]
    else:
        print(f"    Skipping per-column check for {name} (ndim={t_np.ndim})")
        return True
    
    print(f"\n  [{name}] Per-column NaN/Inf check ({n_features} columns):")
    
    issues = []
    for col in range(n_features):
        col_data = t_np[:, col]
        nan_count = int(np.isnan(col_data).sum())
        inf_count = int(np.isinf(col_data).sum())
        
        col_name = dim_names[col] if dim_names and col < len(dim_names) else f"col_{col}"
        
        if nan_count > 0 or inf_count > 0:
            issues.append((col, col_name, nan_count, inf_count))
            print(f"    ⚠️  [{col}] {col_name}: nan={nan_count}, inf={inf_count}")
    
    if not issues:
        print(f"    ✅ All {n_features} columns finite")
        return True
    else:
        print(f"    ❌ {len(issues)} columns have NaN/Inf")
        return False


def check_expected_ranges(t: torch.Tensor, name: str, expected_min: float, expected_max: float):
    """Check if tensor values are within expected range."""
    t_np = t.detach().cpu().numpy().flatten()
    finite_mask = np.isfinite(t_np)
    
    if not finite_mask.any():
        print(f"    ⚠️  {name}: No finite values!")
        return False
    
    finite_vals = t_np[finite_mask]
    actual_min = float(np.min(finite_vals))
    actual_max = float(np.max(finite_vals))
    
    in_range = actual_min >= expected_min and actual_max <= expected_max
    status = "✅" if in_range else "❌"
    print(f"    {status} {name}: expected [{expected_min}, {expected_max}], actual [{actual_min:.4f}, {actual_max:.4f}]")
    
    return in_range


def main():
    args = parse_args()
    
    print("=" * 80)
    print("NaN TRIAGE: Debug First Batch")
    print("=" * 80)
    
    # Load policy config
    with open(args.policy_config) as f:
        policy_config = yaml.safe_load(f)
    
    # Load feature meta
    with open(args.feature_meta_path) as f:
        feature_meta = json.load(f)
    
    # Feature meta uses "seq_features" and "snap_features" keys
    seq_feature_names = feature_meta.get("seq_features", feature_meta.get("seq_feature_names", []))
    snap_feature_names = feature_meta.get("snap_features", feature_meta.get("snap_feature_names", []))
    
    print(f"\n[CONFIG]")
    print(f"  data: {args.data}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  seq_features: {len(seq_feature_names)} base + 3 XGB = {len(seq_feature_names) + 3}")
    print(f"  snap_features: {len(snap_feature_names)} base + 3 XGB = {len(snap_feature_names) + 3}")
    
    # Load DataFrame
    print(f"\n[LOADING DATA]")
    df = pd.read_parquet(args.data)
    print(f"  Loaded {len(df)} rows")
    
    # Import dataset class
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from gx1.models.entry_v10.entry_v10_ctx_dataset import EntryV10CtxDataset
    
    # Create dataset
    print(f"\n[CREATING DATASET]")
    dataset = EntryV10CtxDataset(
        df=df,
        seq_feature_names=seq_feature_names,
        snap_feature_names=snap_feature_names,
        feature_meta_path=args.feature_meta_path,
        seq_scaler_path=Path(args.seq_scaler_path),
        snap_scaler_path=Path(args.snap_scaler_path),
        seq_len=args.seq_len,
        lookback=args.seq_len,
        policy_config=policy_config,
        warmup_bars=288,
        mode="train",
        allow_dummy_ctx=False,
    )
    print(f"  Dataset length: {len(dataset)}")
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    
    # Get first batch
    print(f"\n[PULLING FIRST BATCH]")
    batch = next(iter(dataloader))
    
    # Dataset returns a dict
    if isinstance(batch, dict):
        seq_x = batch["seq_x"]
        snap_x = batch["snap_x"]
        ctx_cat = batch["ctx_cat"]
        ctx_cont = batch["ctx_cont"]
        session_id = batch["session_id"]
        vol_regime_id = batch["vol_regime_id"]
        trend_regime_id = batch["trend_regime_id"]
        y_direction = batch["y_direction"]
        y_early_move = batch["y_early_move"]
        y_quality_score = batch["y_quality_score"]
        print(f"  Batch pulled successfully (dict format)")
        print(f"  Batch keys: {list(batch.keys())}")
    else:
        # Tuple format fallback
        seq_x, snap_x, ctx_cat, ctx_cont, y_direction, y_early_move, y_quality_score = batch[:7]
        session_id = vol_regime_id = trend_regime_id = None
        print(f"  Batch pulled successfully (tuple format)")
    
    # =========================================================================
    # PHASE A: Check input tensors
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE A: INPUT TENSOR CHECKS")
    print("=" * 80)
    
    all_stats = {}
    input_has_nan = False
    
    # Check each tensor
    for name, t in [
        ("seq_x", seq_x),
        ("snap_x", snap_x),
        ("ctx_cat", ctx_cat),
        ("ctx_cont", ctx_cont),
        ("y_direction", y_direction),
        ("y_early_move", y_early_move),
        ("y_quality_score", y_quality_score),
    ]:
        stats = tensor_stats(t, name)
        all_stats[name] = stats
        print_tensor_stats(stats)
        
        if stats["nan_count"] > 0 or stats["inf_count"] > 0:
            input_has_nan = True
    
    # Per-column checks for seq_x and snap_x
    seq_names = seq_feature_names + ["p_long_xgb", "margin_xgb", "uncertainty_score"]
    snap_names = snap_feature_names + ["p_long_xgb", "margin_xgb", "p_hat_xgb"]
    
    seq_ok = check_per_column_nan_inf(seq_x, "seq_x", seq_names)
    snap_ok = check_per_column_nan_inf(snap_x, "snap_x", snap_names)
    
    # Expected range checks
    print("\n[EXPECTED RANGE CHECKS]")
    
    # Extract XGB channels from seq_x (last 3 columns)
    seq_p_long = seq_x[:, :, -3]  # p_long_xgb
    seq_margin = seq_x[:, :, -2]  # margin_xgb  
    seq_uncertainty = seq_x[:, :, -1]  # uncertainty_score
    
    # Extract XGB channels from snap_x (last 3 columns)
    snap_p_long = snap_x[:, -3]  # p_long_xgb
    snap_margin = snap_x[:, -2]  # margin_xgb
    snap_p_hat = snap_x[:, -1]  # p_hat_xgb
    
    range_checks = [
        check_expected_ranges(seq_p_long, "seq_x[:,:,-3] (p_long_xgb)", 0.0, 1.0),
        check_expected_ranges(seq_uncertainty, "seq_x[:,:,-1] (uncertainty_score)", 0.0, 1.0),
        check_expected_ranges(snap_p_long, "snap_x[:,-3] (p_long_xgb)", 0.0, 1.0),
        check_expected_ranges(snap_p_hat, "snap_x[:,-1] (p_hat_xgb)", 0.0, 1.0),
        check_expected_ranges(y_direction.float(), "y_direction", 0.0, 1.0),
        check_expected_ranges(y_early_move.float(), "y_early_move", 0.0, 1.0),
        check_expected_ranges(y_quality_score, "y_quality_score", -1.0, 1.0),
    ]
    
    # Check ctx_cat valid integers
    ctx_cat_np = ctx_cat.numpy()
    print(f"\n  ctx_cat unique values per column:")
    for col in range(ctx_cat.shape[1]):
        uniq = np.unique(ctx_cat_np[:, col])
        print(f"    col {col}: {uniq[:10]}{'...' if len(uniq) > 10 else ''}")
    
    if input_has_nan or not seq_ok or not snap_ok:
        print("\n" + "=" * 80)
        print("❌ NAN_IN_INPUT: Found NaN/Inf in input tensors")
        print("=" * 80)
        
        # Find first offending samples
        for name, t in [("seq_x", seq_x), ("snap_x", snap_x)]:
            t_np = t.numpy()
            for i in range(t_np.shape[0]):
                if not np.isfinite(t_np[i]).all():
                    print(f"  First offending sample in {name}: index {i}")
                    if t_np.ndim == 3:
                        for j in range(t_np.shape[2]):
                            col_data = t_np[i, :, j]
                            if not np.isfinite(col_data).all():
                                print(f"    Column {j}: nan={np.isnan(col_data).sum()}, inf={np.isinf(col_data).sum()}")
                    break
        
        sys.exit(1)
    
    print("\n✅ PHASE A PASSED: All input tensors are finite")
    
    # =========================================================================
    # PHASE B: Model forward pass
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE B: MODEL FORWARD PASS")
    print("=" * 80)
    
    from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer
    
    # Build model with same config as training
    actual_seq_dim = len(seq_feature_names) + 3
    actual_snap_dim = len(snap_feature_names) + 3
    
    print(f"\n[CREATING MODEL]")
    print(f"  seq_input_dim: {actual_seq_dim}")
    print(f"  snap_input_dim: {actual_snap_dim}")
    
    device = torch.device("cpu")  # Use CPU for debugging
    
    model = EntryV10CtxHybridTransformer(
        seq_input_dim=actual_seq_dim,
        snap_input_dim=actual_snap_dim,
        max_seq_len=args.seq_len,
        variant="v10_ctx",
        ctx_cat_dim=5,
        ctx_cont_dim=2,
        ctx_emb_dim=42,
        ctx_embedding_dim=8,
    ).to(device)
    
    model.eval()
    
    # Move batch to device
    seq_x = seq_x.to(device)
    snap_x = snap_x.to(device)
    ctx_cat = ctx_cat.to(device)
    ctx_cont = ctx_cont.to(device)
    if session_id is not None:
        session_id = session_id.to(device)
        vol_regime_id = vol_regime_id.to(device)
        trend_regime_id = trend_regime_id.to(device)
    
    print(f"\n[FORWARD PASS]")
    
    # Enable NaN debug mode
    os.environ["GX1_DEBUG_NAN"] = "1"
    
    with torch.no_grad():
        try:
            outputs = model(
                seq_x=seq_x,
                snap_x=snap_x,
                ctx_cat=ctx_cat,
                ctx_cont=ctx_cont,
                session_id=session_id,
                vol_regime_id=vol_regime_id,
                trend_regime_id=trend_regime_id,
            )
            
            # Unpack outputs
            if isinstance(outputs, dict):
                direction_logit = outputs.get("direction_logit")
                early_move_logit = outputs.get("early_move_logit")
                quality_score = outputs.get("quality_score")
                gate = outputs.get("gate")
            else:
                direction_logit, early_move_logit, quality_score = outputs[:3]
                gate = outputs[3] if len(outputs) > 3 else None
            
            print("\n[MODEL OUTPUTS]")
            output_has_nan = False
            
            for name, t in [
                ("direction_logit", direction_logit),
                ("early_move_logit", early_move_logit),
                ("quality_score", quality_score),
                ("gate", gate),
            ]:
                if t is None:
                    print(f"  {name}: None")
                    continue
                    
                stats = tensor_stats(t, name)
                print_tensor_stats(stats)
                
                if stats["nan_count"] > 0 or stats["inf_count"] > 0:
                    output_has_nan = True
            
            if output_has_nan:
                print("\n" + "=" * 80)
                print("❌ NAN_IN_FORWARD: Found NaN/Inf in model outputs")
                print("=" * 80)
                sys.exit(2)
            
            print("\n✅ PHASE B PASSED: All model outputs are finite")
            
        except Exception as e:
            print(f"\n❌ FORWARD PASS FAILED: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(2)
    
    # =========================================================================
    # PHASE C: Loss computation
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE C: LOSS COMPUTATION")
    print("=" * 80)
    
    y_direction = y_direction.to(device)
    y_early_move = y_early_move.to(device)
    y_quality_score = y_quality_score.to(device)
    
    print("\n[LABEL STATS]")
    print(f"  y_direction: unique={torch.unique(y_direction).tolist()}, mean={y_direction.float().mean():.4f}")
    print(f"  y_early_move: unique={torch.unique(y_early_move).tolist()}, mean={y_early_move.float().mean():.4f}")
    print(f"  y_quality_score: min={y_quality_score.min():.4f}, max={y_quality_score.max():.4f}, mean={y_quality_score.mean():.4f}")
    
    loss_has_nan = False
    loss_components = {}
    
    # Direction loss (BCE with logits)
    print("\n[COMPUTING LOSSES]")
    
    try:
        # Check logits before BCE
        print(f"  direction_logit before BCE: min={direction_logit.min():.4f}, max={direction_logit.max():.4f}")
        
        # Clamp for diagnostic (not for training)
        clamped_logit = torch.clamp(direction_logit.squeeze(), -50, 50)
        
        loss_dir = F.binary_cross_entropy_with_logits(
            clamped_logit,
            y_direction.float(),
            reduction="mean"
        )
        loss_components["direction"] = loss_dir.item()
        print(f"  ✅ loss_direction: {loss_dir.item():.6f}")
        
        if not torch.isfinite(loss_dir):
            print(f"  ❌ loss_direction is NaN/Inf!")
            loss_has_nan = True
            
    except Exception as e:
        print(f"  ❌ loss_direction computation failed: {e}")
        loss_has_nan = True
    
    # Early move loss
    try:
        clamped_early = torch.clamp(early_move_logit.squeeze(), -50, 50)
        
        loss_early = F.binary_cross_entropy_with_logits(
            clamped_early,
            y_early_move.float(),
            reduction="mean"
        )
        loss_components["early_move"] = loss_early.item()
        print(f"  ✅ loss_early_move: {loss_early.item():.6f}")
        
        if not torch.isfinite(loss_early):
            print(f"  ❌ loss_early_move is NaN/Inf!")
            loss_has_nan = True
            
    except Exception as e:
        print(f"  ❌ loss_early_move computation failed: {e}")
        loss_has_nan = True
    
    # Quality loss (MSE)
    try:
        loss_quality = F.mse_loss(
            quality_score.squeeze(),
            y_quality_score,
            reduction="mean"
        )
        loss_components["quality"] = loss_quality.item()
        print(f"  ✅ loss_quality: {loss_quality.item():.6f}")
        
        if not torch.isfinite(loss_quality):
            print(f"  ❌ loss_quality is NaN/Inf!")
            loss_has_nan = True
            
    except Exception as e:
        print(f"  ❌ loss_quality computation failed: {e}")
        loss_has_nan = True
    
    # Gate stability loss (if gate present)
    if gate is not None:
        try:
            gate_stability = torch.var(gate)
            loss_components["gate_stability"] = gate_stability.item()
            print(f"  ✅ gate_stability (var): {gate_stability.item():.6f}")
            
            if not torch.isfinite(gate_stability):
                print(f"  ❌ gate_stability is NaN/Inf!")
                loss_has_nan = True
                
        except Exception as e:
            print(f"  ❌ gate_stability computation failed: {e}")
            loss_has_nan = True
    
    # Total loss
    try:
        total_loss = sum(v for v in loss_components.values() if isinstance(v, float) and np.isfinite(v))
        print(f"\n  Total loss (sum): {total_loss:.6f}")
    except:
        pass
    
    if loss_has_nan:
        print("\n" + "=" * 80)
        print("❌ NAN_IN_LOSS: Found NaN/Inf in loss computation")
        print("=" * 80)
        sys.exit(3)
    
    print("\n✅ PHASE C PASSED: All loss components are finite")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("✅ ALL CHECKS PASSED")
    print("=" * 80)
    print("\nThe first batch is clean. NaN may occur later in training due to:")
    print("  - Gradient explosion (check grad_norm)")
    print("  - Learning rate too high")
    print("  - Specific problematic samples later in dataset")
    print("\nRecommendation: Add gradient clipping and run with smaller learning rate.")
    
    sys.exit(0)


if __name__ == "__main__":
    main()
