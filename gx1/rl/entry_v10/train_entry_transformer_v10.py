#!/usr/bin/env python3
"""
Train ENTRY_V10 Hybrid Transformer model.

Trains the Transformer model on XGBoost-annotated sequences.

Input:
    - Training dataset: data/entry_v10/entry_v10_train.parquet
    - Validation dataset: data/entry_v10/entry_v10_val.parquet
        (Both created by dataset_v10.train_val_split)

Output:
    - models/entry_v10/entry_v10_transformer.pt (PyTorch state_dict)
    - models/entry_v10/entry_v10_transformer_meta.json (metadata)

Usage:
    python -m gx1.rl.entry_v10.train_entry_transformer_v10 \
        --train-parquet data/entry_v10/entry_v10_train.parquet \
        --val-parquet data/entry_v10/entry_v10_val.parquet \
        --model-out models/entry_v10/entry_v10_transformer.pt \
        --meta-out models/entry_v10/entry_v10_transformer_meta.json \
        --epochs 30 \
        --batch-size 256 \
        --lr 1e-4
"""

import argparse
import json
import logging
import multiprocessing
import os
import platform
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from gx1.rl.entry_v10.dataset_v10 import EntryV10Dataset
from gx1.models.entry_v10.entry_v10_hybrid_transformer import EntryV10HybridTransformer

# Set multiprocessing start method for macOS compatibility
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

# Optimize CPU usage: Aggressive configuration for maximum throughput
def get_cpu_info():
    """Get CPU information (physical and logical cores)."""
    physical = None
    logical = os.cpu_count() or 8
    
    if platform.system() == "Darwin":  # macOS
        try:
            import subprocess
            result = subprocess.run(["sysctl", "-n", "hw.physicalcpu"], capture_output=True, text=True)
            if result.returncode == 0:
                physical = int(result.stdout.strip())
        except Exception:
            pass
    
    if physical is None:
        try:
            import psutil
            physical = psutil.cpu_count(logical=False) or logical
        except ImportError:
            physical = logical  # Assume no hyperthreading
    
    return physical, logical

PHYSICAL_CORES, LOGICAL_CORES = get_cpu_info()

# Strategy: Use ALL logical cores for PyTorch BLAS (matrix ops are highly parallelizable)
# PyTorch threads handle matrix multiplication which benefits from many threads
PYTORCH_THREADS = LOGICAL_CORES
torch.set_num_threads(PYTORCH_THREADS)
torch.set_num_interop_threads(PYTORCH_THREADS)  # Use all cores for interop too

# Set OpenMP/MKL threads (for BLAS libraries) - use all logical cores
os.environ["OMP_NUM_THREADS"] = str(PYTORCH_THREADS)
os.environ["MKL_NUM_THREADS"] = str(PYTORCH_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(PYTORCH_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(PYTORCH_THREADS)  # macOS Accelerate framework

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
log.info(f"[CPU_OPT] Physical cores: {PHYSICAL_CORES}, Logical cores: {LOGICAL_CORES}")
log.info(f"[CPU_OPT] PyTorch threads: {PYTORCH_THREADS}, interop: {PYTORCH_THREADS}")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    enable_auxiliary: bool = False,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    log.info(f"Starting training loop with {len(dataloader)} batches...")
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx == 0:
            log.info(f"First batch received! Batch keys: {list(batch.keys())}")
        if batch_idx % 50 == 0:
            log.info(f"Processing batch {batch_idx}/{len(dataloader)}...")
        
        # Move to device
        seq_x = batch["seq_x"].to(device)
        snap_x = batch["snap_x"].to(device)
        session_id = batch["session_id"].to(device)
        vol_regime_id = batch["vol_regime_id"].to(device)
        trend_regime_id = batch["trend_regime_id"].to(device)
        y = batch.get("y_direction", batch.get("y")).to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(seq_x, snap_x, session_id, vol_regime_id, trend_regime_id)

        # Main loss: direction prediction
        direction_logit = output["direction_logit"].squeeze(-1)
        
        # Check for NaN/Inf in inputs
        if torch.isnan(seq_x).any() or torch.isinf(seq_x).any():
            log.error(f"NaN/Inf in seq_x! This is a data problem.")
            continue
        if torch.isnan(snap_x).any() or torch.isinf(snap_x).any():
            log.error(f"NaN/Inf in snap_x! This is a data problem.")
            continue
            
        # Check for NaN/Inf after forward pass - HARD FEIL, no fallback
        if torch.isnan(direction_logit).any() or torch.isinf(direction_logit).any():
            log.error(f"NaN/Inf in direction_logit! Model output is invalid. direction_logit stats: min={direction_logit.min():.6f}, max={direction_logit.max():.6f}, mean={direction_logit.mean():.6f}")
            raise RuntimeError(
                "CRITICAL: NaN/Inf in model predictions. "
                "This indicates a training problem. NO FALLBACK - HARD FEIL."
            )
        if torch.isnan(y).any() or torch.isinf(y).any():
            log.error(f"NaN/Inf in labels! This is a data problem.")
            continue
            
        loss = criterion(direction_logit, y)
        
        # Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            log.error(f"NaN/Inf loss detected! Loss value: {loss.item()}")
            break

        # Auxiliary losses (if enabled)
        if enable_auxiliary and "early_move_logit" in output:
            # Optional: add auxiliary losses with lower weight
            # For now, we'll skip them in the first version
            pass

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y)
        n_samples += len(y)

    avg_loss = total_loss / n_samples if n_samples > 0 else 0.0
    return {"loss": avg_loss}


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    n_samples = 0

    all_preds = []
    all_targets = []

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

            loss = criterion(direction_logit, y)
            total_loss += loss.item() * len(y)
            n_samples += len(y)

            # Collect predictions for metrics
            probs = torch.sigmoid(direction_logit)
            
            # V10.1 design: HARD FEIL on NaN/Inf - no fallback
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                raise RuntimeError(
                    "CRITICAL: NaN/Inf in validation predictions. "
                    "NO FALLBACK - HARD FEIL."
                )
            
            # Check for constant predictions (std < 1e-6)
            probs_np = probs.cpu().numpy()
            if len(probs_np) > 1 and np.std(probs_np) < 1e-6:
                raise RuntimeError(
                    "CRITICAL: Predictions are effectively constant "
                    f"(std={np.std(probs_np):.2e}). NO FALLBACK - HARD FEIL."
                )
            
            all_preds.extend(probs_np)
            all_targets.extend(y.cpu().numpy())

    avg_loss = total_loss / n_samples if n_samples > 0 else 0.0

    # Compute metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.0

    pred_binary = (all_preds > 0.5).astype(int)
    accuracy = accuracy_score(all_targets, pred_binary)

    return {
        "loss": avg_loss,
        "auc": auc,
        "accuracy": accuracy,
    }


def train_entry_v10_transformer(
    train_parquet: Path,
    val_parquet: Path,
    model_out: Path,
    meta_out: Path,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-4,
    device: str = "cpu",
    seq_len: int = 30,
    variant: str = "v10",
    early_stopping_patience: int = 7,
) -> None:
    """
    Train ENTRY_V10 Hybrid Transformer.

    Args:
        train_parquet: Path to training dataset
        val_parquet: Path to validation dataset
        model_out: Path to save model state_dict
        meta_out: Path to save metadata JSON
        epochs: Maximum number of training epochs (early stopping may stop earlier)
        batch_size: Batch size
        lr: Learning rate
        device: Device (cpu/cuda)
        seq_len: Sequence length
        variant: Model variant ("v10" or "v10_1")
        early_stopping_patience: Number of epochs without improvement before early stopping
            - v10: num_layers=3, d_model=128, dim_feedforward=512
            - v10_1: num_layers=6, d_model=256, dim_feedforward=1024, dropout=0.05
    """
    device = torch.device(device)
    log.info(f"Using device: {device}")
    
    # Note: Thread settings are already configured at module level (based on physical cores)
    # This ensures optimal BLAS/matmul performance without thread thrashing

    # Load datasets
    log.info(f"Loading training dataset: {train_parquet}")
    train_dataset = EntryV10Dataset(train_parquet, seq_len=seq_len, device=str(device))
    
    # Optimize DataLoader workers: Aggressive configuration
    # DataLoader workers run in separate processes and do I/O + preprocessing
    # They DON'T compete directly with PyTorch threads (different processes)
    # Use all physical cores for workers (they handle I/O which is I/O-bound, not CPU-bound)
    # The main training loop will use PyTorch threads (BLAS) in the main process
    NUM_WORKERS = PHYSICAL_CORES  # Use all physical cores for data loading
    log.info(f"Using {NUM_WORKERS} DataLoader workers (physical cores: {PHYSICAL_CORES}, PyTorch threads: {PYTORCH_THREADS})")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=False,  # CPU only, no pin_memory
        persistent_workers=True,  # Keep workers alive between epochs (reduces overhead)
        prefetch_factor=8,  # Aggressive prefetching - keep workers busy ahead of training
        multiprocessing_context='spawn',  # Explicit spawn for macOS
    )

    log.info(f"Loading validation dataset: {val_parquet}")
    val_dataset = EntryV10Dataset(val_parquet, seq_len=seq_len, device=str(device))
    # Use all workers for validation too (validation is still CPU-bound)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=NUM_WORKERS,  # Use all workers for validation
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=4,  # Lower prefetch for validation
        multiprocessing_context='spawn',
    )

    # Initialize model with proper weight initialization
    log.info(f"Initializing ENTRY_V10 Hybrid Transformer (variant={variant}, seq_len={seq_len})")
    model = EntryV10HybridTransformer(
        seq_input_dim=16,
        snap_input_dim=88,
        max_seq_len=seq_len,
        variant=variant,
        enable_auxiliary_heads=True,
    ).to(device)
    
    # Initialize weights properly to avoid NaN
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.1)  # Smaller gain to prevent explosion
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training loop with early stopping
    log.info(f"Starting training for up to {epochs} epochs (early stopping patience: {early_stopping_patience})")
    best_val_loss = float("inf")
    best_val_auc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Learning rate scheduling
        scheduler.step(val_metrics["loss"])

        # Logging
        log.info(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )

        # Check for improvement (using val_loss as primary metric)
        improved = val_metrics["loss"] < best_val_loss
        if improved:
            best_val_loss = val_metrics["loss"]
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            log.info(f"New best model (val_loss={best_val_loss:.4f}, val_auc={best_val_auc:.4f})")
            
            # Save model
            model_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_out)
            log.info(f"Saved model to {model_out}")
        else:
            epochs_without_improvement += 1
            log.debug(f"No improvement for {epochs_without_improvement} epochs (best: epoch {best_epoch}, val_loss={best_val_loss:.4f})")
        
        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            log.info(
                f"Early stopping triggered: no improvement for {early_stopping_patience} epochs. "
                f"Best model was at epoch {best_epoch} (val_loss={best_val_loss:.4f}, val_auc={best_val_auc:.4f})"
            )
            break

    log.info(f"Training complete. Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}, Best val AUC: {best_val_auc:.4f}")

    # Save metadata
    metadata = {
        "variant": variant,
        "seq_len": seq_len,
        "seq_feature_count": 16,
        "snap_feature_count": 88,
        "seq_features": [
            "atr50", "atr_regime_id", "atr_z", "body_pct", "ema100_slope",
            "ema20_slope", "pos_vs_ema200", "roc100", "roc20", "session_id",
            "std50", "trend_regime_tf24h", "wick_asym",
            "p_long_xgb_seq", "margin_xgb_seq", "p_long_xgb_ema_seq",
        ],
        "snap_features": [
            # 85 V9 snapshot features + 3 XGB-now
            "p_long_xgb_now", "margin_xgb_now", "p_hat_xgb_now",
        ],
        "model_path": str(model_out),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_auc": best_val_auc,
        "total_epochs_trained": epoch + 1,
        "early_stopped": epochs_without_improvement >= early_stopping_patience,
        "training_config": {
            "variant": variant,
            "max_epochs": epochs,
            "early_stopping_patience": early_stopping_patience,
            "batch_size": batch_size,
            "lr": lr,
            "device": str(device),
        },
    }

    meta_out.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Saved metadata to {meta_out}")


def main():
    parser = argparse.ArgumentParser(
        description="Train ENTRY_V10 Hybrid Transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # V10 (default)
  python -m gx1.rl.entry_v10.train_entry_transformer_v10 \\
    --train-parquet data/entry_v10/entry_v10_train.parquet \\
    --val-parquet data/entry_v10/entry_v10_val.parquet \\
    --model-out models/entry_v10/entry_v10_transformer.pt \\
    --meta-out models/entry_v10/entry_v10_transformer_meta.json \\
    --seq-len 30 \\
    --variant v10

  # V10.1 (deeper, longer sequences)
  python -m gx1.rl.entry_v10.train_entry_transformer_v10 \\
    --train-parquet data/entry_v10/entry_v10_1_train.parquet \\
    --val-parquet data/entry_v10/entry_v10_1_val.parquet \\
    --model-out models/entry_v10/entry_v10_1_transformer.pt \\
    --meta-out models/entry_v10/entry_v10_1_transformer_meta.json \\
    --seq-len 90 \\
    --variant v10_1
        """
    )
    parser.add_argument("--train-parquet", type=Path, required=True, help="Training dataset path")
    parser.add_argument("--val-parquet", type=Path, required=True, help="Validation dataset path")
    parser.add_argument("--model-out", type=Path, required=True, help="Output model path")
    parser.add_argument("--meta-out", type=Path, required=True, help="Output metadata path")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum number of epochs (early stopping may stop earlier)")
    parser.add_argument("--early-stopping-patience", type=int, default=7, help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--seq-len", type=int, default=30, help="Sequence length")
    parser.add_argument("--variant", type=str, default="v10", choices=["v10", "v10_1"], 
                       help="Model variant: v10 (default) or v10_1 (deeper)")

    args = parser.parse_args()

    train_entry_v10_transformer(
        train_parquet=args.train_parquet,
        val_parquet=args.val_parquet,
        model_out=args.model_out,
        meta_out=args.meta_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seq_len=args.seq_len,
        variant=args.variant,
        early_stopping_patience=args.early_stopping_patience,
    )


if __name__ == "__main__":
    import numpy as np
    main()

