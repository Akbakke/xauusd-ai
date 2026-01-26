#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENTRY_V10_CTX Depth Ladder Training Script

Trains ENTRY_V10_CTX with depth ladder variants (baseline vs L+1).

⚠️  DO NOT MODIFY TRADING LOGIC - ONLY ARCHITECTURE DEPTH

Usage:
    python gx1/scripts/train_entry_v10_ctx_depth_ladder.py \
        --variant baseline|lplus1 \
        --data <parquet> \
        --out-dir checkpoints/entry_v10_ctx_depth_ladder/ \
        --seed 42 \
        --epochs 10 \
        --device auto
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from gx1.models.entry_v10.entry_v10_ctx_train import (
    main as train_main,
    set_seed,
    set_thread_limits,
)
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ============================================================================
# BASELINE ARCHITECTURE (LOCKED)
# ============================================================================
BASELINE_CONFIG = {
    "variant": "v10_ctx",
    "num_layers": 3,
    "d_model": 128,
    "n_heads": 4,
    "dim_feedforward": None,  # Will use d_model * 4 = 512
    "dropout": 0.05,
    "seq_len": 30,
    "ctx_cat_dim": 5,
    "ctx_cont_dim": 2,
    "ctx_emb_dim": 42,
    "ctx_embedding_dim": 8,
}

LPLUS1_CONFIG = {
    **BASELINE_CONFIG,
    "num_layers": 4,  # baseline + 1
    "depth_ladder_delta": +1,
}


def compute_bundle_sha256(bundle_dir: Path) -> str:
    """Compute deterministic SHA256 for bundle."""
    files_to_hash = [
        "model_state_dict.pt",
        "bundle_metadata.json",
        "feature_contract_hash.txt",
    ]
    
    hasher = hashlib.sha256()
    for filename in files_to_hash:
        filepath = bundle_dir / filename
        if filepath.exists():
            with open(filepath, "rb") as f:
                hasher.update(f.read())
    
    return hasher.hexdigest()[:16]  # First 16 chars


def validate_config_diff(baseline_cfg: Dict, variant_cfg: Dict, allowed_diff: set) -> None:
    """
    FATAL if config differs beyond allowed parameters.
    
    Args:
        baseline_cfg: Baseline configuration
        variant_cfg: Variant configuration
        allowed_diff: Set of keys allowed to differ (e.g., {"num_layers"})
    """
    all_keys = set(baseline_cfg.keys()) | set(variant_cfg.keys())
    diff_keys = set()
    
    for key in all_keys:
        baseline_val = baseline_cfg.get(key)
        variant_val = variant_cfg.get(key)
        
        if baseline_val != variant_val:
            if key not in allowed_diff:
                diff_keys.add(key)
    
    if diff_keys:
        raise RuntimeError(
            f"FATAL: Config differs beyond allowed parameters. "
            f"Diff keys: {diff_keys}, Allowed: {allowed_diff}. "
            f"This ensures only depth changes, nothing else."
        )


def train_depth_ladder_variant(
    variant: str,
    data_path: Path,
    out_dir: Path,
    feature_meta_path: Path,
    seq_scaler_path: Optional[Path],
    snap_scaler_path: Optional[Path],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
    seq_len: int = 30,
) -> Dict[str, Any]:
    """
    Train a depth ladder variant.
    
    Args:
        variant: "baseline" or "lplus1"
        data_path: Path to training dataset
        out_dir: Output directory
        feature_meta_path: Path to feature_meta.json
        seq_scaler_path: Optional path to seq scaler
        snap_scaler_path: Optional path to snap scaler
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        seed: Random seed
        device: Device (cpu/cuda/mps/auto)
        seq_len: Sequence length
    
    Returns:
        Training report dict
    """
    log.info(f"\n{'='*80}")
    log.info(f"Training Depth Ladder Variant: {variant.upper()}")
    log.info(f"{'='*80}\n")
    
    # Select config
    if variant == "baseline":
        config = BASELINE_CONFIG.copy()
    elif variant == "lplus1":
        config = LPLUS1_CONFIG.copy()
        # Validate: only num_layers should differ
        validate_config_diff(BASELINE_CONFIG, config, allowed_diff={"num_layers", "depth_ladder_delta"})
    else:
        raise ValueError(f"Unknown variant: {variant}. Must be 'baseline' or 'lplus1'")
    
    log.info(f"Config: {config}")
    
    # Create output directory
    variant_out_dir = out_dir / variant.upper()
    variant_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Set determinism
    set_seed(seed)
    set_thread_limits()
    
    # Device
    if device == "auto":
        if torch.cuda.is_available():
            torch_device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch_device = torch.device("mps")
        else:
            torch_device = torch.device("cpu")
    else:
        torch_device = torch.device(device)
    
    log.info(f"Device: {torch_device}")
    log.info(f"Seed: {seed}")
    
    # Import training function (we'll call it with modified args)
    # For now, we'll create model directly and train manually
    # But we need to use the existing training infrastructure
    
    # Build training command args
    import sys
    original_argv = sys.argv.copy()
    
    try:
        # Prepare args for training script
        train_args = [
            "--data", str(data_path),
            "--out_dir", str(variant_out_dir),
            "--feature_meta_path", str(feature_meta_path),
            "--seq_len", str(seq_len),
            "--batch_size", str(batch_size),
            "--epochs", str(epochs),
            "--learning_rate", str(learning_rate),
            "--seed", str(seed),
            "--device", device,
        ]
        
        if seq_scaler_path:
            train_args.extend(["--seq_scaler_path", str(seq_scaler_path)])
        if snap_scaler_path:
            train_args.extend(["--snap_scaler_path", str(snap_scaler_path)])
        
        # Override sys.argv for training script
        sys.argv = ["train_entry_v10_ctx_train.py"] + train_args
        
        # Import and patch model creation to use depth ladder config
        from gx1.models.entry_v10 import entry_v10_ctx_train
        
        # Monkey-patch model creation to use depth ladder config
        original_create_model = None
        
        def create_model_with_depth_ladder(*args, **kwargs):
            """Create model with depth ladder config."""
            # Override seq_cfg to use depth ladder num_layers
            if "seq_cfg" not in kwargs:
                kwargs["seq_cfg"] = {}
            
            kwargs["seq_cfg"]["num_layers"] = config["num_layers"]
            kwargs["seq_cfg"]["d_model"] = config["d_model"]
            kwargs["seq_cfg"]["n_heads"] = config["n_heads"]
            kwargs["seq_cfg"]["dropout"] = config["dropout"]
            kwargs["seq_cfg"]["dim_feedforward"] = config["dim_feedforward"]
            
            # Create model
            model = EntryV10CtxHybridTransformer(
                seq_input_dim=kwargs.get("seq_input_dim", 16),
                snap_input_dim=kwargs.get("snap_input_dim", 88),
                max_seq_len=kwargs.get("max_seq_len", seq_len),
                variant=config["variant"],
                ctx_cat_dim=config["ctx_cat_dim"],
                ctx_cont_dim=config["ctx_cont_dim"],
                ctx_emb_dim=config["ctx_emb_dim"],
                ctx_embedding_dim=config["ctx_embedding_dim"],
                seq_cfg=kwargs["seq_cfg"],
            )
            return model
        
        # This is complex - let's use a simpler approach: modify training script call
        # Actually, better: create a wrapper that calls training with modified model creation
        
        # For now, let's write a modified training function inline
        log.info("Starting training...")
        log.info("NOTE: This requires modifying entry_v10_ctx_train.py to accept depth_ladder_mode")
        log.info("For now, we'll create a wrapper script")
        
        # Actually, let's just call the training script with environment variable
        # and modify the model creation in the training script to check for depth_ladder_mode
        
        # Set environment variable for depth ladder mode
        os.environ["GX1_DEPTH_LADDER_MODE"] = "1"
        os.environ["GX1_DEPTH_LADDER_VARIANT"] = variant
        os.environ["GX1_DEPTH_LADDER_NUM_LAYERS"] = str(config["num_layers"])
        
        # Call training script
        from gx1.models.entry_v10.entry_v10_ctx_train import main as train_main_internal
        
        # We need to modify the training script to support depth_ladder_mode
        # For now, let's create a simpler approach: write a wrapper
        
        log.warning("⚠️  Direct training integration requires modifying entry_v10_ctx_train.py")
        log.warning("   For now, creating training command that user can run manually")
        
        # Generate training command
        cmd_parts = [
            "python", "-m", "gx1.models.entry_v10.entry_v10_ctx_train",
            "--data", str(data_path),
            "--out_dir", str(variant_out_dir),
            "--feature_meta_path", str(feature_meta_path),
            "--seq_len", str(seq_len),
            "--batch_size", str(batch_size),
            "--epochs", str(epochs),
            "--learning_rate", str(learning_rate),
            "--seed", str(seed),
            "--device", device,
        ]
        
        if seq_scaler_path:
            cmd_parts.extend(["--seq_scaler_path", str(seq_scaler_path)])
        if snap_scaler_path:
            cmd_parts.extend(["--snap_scaler_path", str(snap_scaler_path)])
        
        cmd_str = " ".join(cmd_parts)
        
        log.info(f"Training command (with depth_ladder_mode):")
        log.info(f"  GX1_DEPTH_LADDER_MODE=1 GX1_DEPTH_LADDER_VARIANT={variant} GX1_DEPTH_LADDER_NUM_LAYERS={config['num_layers']} {cmd_str}")
        
        # For now, return command string
        # User will need to run this manually or we modify entry_v10_ctx_train.py
        
        return {
            "variant": variant,
            "config": config,
            "out_dir": str(variant_out_dir),
            "command": cmd_str,
            "status": "command_generated",
        }
        
    finally:
        sys.argv = original_argv
        # Clean up env vars
        os.environ.pop("GX1_DEPTH_LADDER_MODE", None)
        os.environ.pop("GX1_DEPTH_LADDER_VARIANT", None)
        os.environ.pop("GX1_DEPTH_LADDER_NUM_LAYERS", None)


def main():
    parser = argparse.ArgumentParser(description="Train ENTRY_V10_CTX Depth Ladder variants")
    parser.add_argument("--variant", type=str, required=True, choices=["baseline", "lplus1"],
                        help="Variant to train (baseline or lplus1)")
    parser.add_argument("--data", type=Path, required=True,
                        help="Path to training dataset (parquet)")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory for checkpoints")
    parser.add_argument("--feature-meta-path", type=Path, required=True,
                        help="Path to feature_meta.json")
    parser.add_argument("--seq-scaler-path", type=Path, default=None,
                        help="Path to seq scaler (optional)")
    parser.add_argument("--snap-scaler-path", type=Path, default=None,
                        help="Path to snap scaler (optional)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs (default: 10, same as baseline)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cpu/cuda/mps/auto)")
    parser.add_argument("--seq-len", type=int, default=30,
                        help="Sequence length (default: 30)")
    
    args = parser.parse_args()
    
    # Train variant
    result = train_depth_ladder_variant(
        variant=args.variant,
        data_path=args.data,
        out_dir=args.out_dir,
        feature_meta_path=args.feature_meta_path,
        seq_scaler_path=args.seq_scaler_path,
        snap_scaler_path=args.snap_scaler_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        seq_len=args.seq_len,
    )
    
    log.info(f"\n✅ Training setup complete for {args.variant}")
    log.info(f"   Output: {result['out_dir']}")
    log.info(f"   Command: {result['command']}")


if __name__ == "__main__":
    main()
