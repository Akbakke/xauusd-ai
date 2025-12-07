#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gx1/models/entry_v9/entry_v9_train.py

Training pipeline for ENTRY_V9 NEXTGEN transformer model.
Multi-task learning with regime-conditioning.

CLI:
    python -m gx1.models.entry_v9.entry_v9_train --config gx1/configs/entry_v9/entry_v9_train_full.yaml
"""

import argparse
import json
import os
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, mean_squared_error
from joblib import dump, load

# Fix macOS multiprocessing
if hasattr(multiprocessing, 'set_start_method'):
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

# GX1 imports
from gx1.models.entry_v9.entry_v9_transformer import EntryV9Transformer, build_entry_v9_model
from gx1.models.entry_v9.entry_v9_dataset import (
    prepare_entry_v9_data,
    create_entry_v9_dataloaders,
    EntryV9Dataset,
)
from gx1.tuning.entry_v4_train import time_split_by_dates, set_seed
from torch.utils.data import DataLoader


def collate_entry_v9_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate function for EntryV9Dataset batches."""
    return {
        "seq_x": torch.stack([item["seq_x"] for item in batch]),
        "snap_x": torch.stack([item["snap_x"] for item in batch]),
        "session_id": torch.stack([item["session_id"] for item in batch]),
        "vol_regime_id": torch.stack([item["vol_regime_id"] for item in batch]),
        "trend_regime_id": torch.stack([item["trend_regime_id"] for item in batch]),
        "y_direction": torch.stack([item["y_direction"] for item in batch]),
        "y_early_move": torch.stack([item["y_early_move"] for item in batch]),
        "y_quality_score": torch.stack([item["y_quality_score"] for item in batch]),
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weights: Dict[str, float],
) -> Tuple[float, float]:
    """Train one epoch with multi-task loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    criterion_direction = nn.BCEWithLogitsLoss()
    criterion_early_move = nn.BCEWithLogitsLoss()
    criterion_quality = nn.MSELoss()

    for batch in dataloader:
        seq_x = batch["seq_x"].to(device)
        snap_x = batch["snap_x"].to(device)
        session_id = batch["session_id"].to(device)
        vol_regime_id = batch["vol_regime_id"].to(device)
        trend_regime_id = batch["trend_regime_id"].to(device)

        y_direction = batch["y_direction"].float().to(device)
        y_early_move = batch["y_early_move"].float().to(device)
        y_quality_score = batch["y_quality_score"].to(device)

        optimizer.zero_grad()

        outputs = model(
            seq_x=seq_x,
            snap_x=snap_x,
            session_id=session_id,
            vol_regime_id=vol_regime_id,
            trend_regime_id=trend_regime_id,
        )

        # Multi-task loss
        loss_direction = criterion_direction(outputs["direction_logit"], y_direction)
        loss_early_move = criterion_early_move(outputs["early_move_logit"], y_early_move)
        loss_quality = criterion_quality(outputs["quality_score"], y_quality_score)

        total_loss_batch = (
            loss_weights["direction"] * loss_direction
            + loss_weights["early_move"] * loss_early_move
            + loss_weights["quality"] * loss_quality
        )

        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += total_loss_batch.item() * seq_x.size(0)
        total_samples += seq_x.size(0)

    avg_loss = total_loss / max(1, total_samples)

    # Calculate positive rate for direction
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for batch in dataloader:
            seq_x = batch["seq_x"].to(device)
            snap_x = batch["snap_x"].to(device)
            session_id = batch["session_id"].to(device)
            vol_regime_id = batch["vol_regime_id"].to(device)
            trend_regime_id = batch["trend_regime_id"].to(device)
            y_direction = batch["y_direction"].float().to(device)

            outputs = model(
                seq_x=seq_x,
                snap_x=snap_x,
                session_id=session_id,
                vol_regime_id=vol_regime_id,
                trend_regime_id=trend_regime_id,
            )
            probs = torch.sigmoid(outputs["direction_logit"])

            all_preds.append(probs.cpu().numpy())
            all_labels.append(y_direction.cpu().numpy())

        if len(all_preds) == 0:
            pos_rate = 0.0
        else:
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            pos_rate = float(np.mean(all_preds > 0.5))

    model.train()
    return avg_loss, pos_rate


def evaluate_entry_v9(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_weights: Dict[str, float],
) -> Tuple[float, float, float, float]:
    """Evaluate model with multi-task metrics."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    criterion_direction = nn.BCEWithLogitsLoss()
    criterion_early_move = nn.BCEWithLogitsLoss()
    criterion_quality = nn.MSELoss()

    all_direction_preds = []
    all_direction_labels = []
    all_early_move_preds = []
    all_early_move_labels = []
    all_quality_preds = []
    all_quality_labels = []

    with torch.no_grad():
        for batch in dataloader:
            seq_x = batch["seq_x"].to(device)
            snap_x = batch["snap_x"].to(device)
            session_id = batch["session_id"].to(device)
            vol_regime_id = batch["vol_regime_id"].to(device)
            trend_regime_id = batch["trend_regime_id"].to(device)

            y_direction = batch["y_direction"].float().to(device)
            y_early_move = batch["y_early_move"].float().to(device)
            y_quality_score = batch["y_quality_score"].to(device)

            outputs = model(
                seq_x=seq_x,
                snap_x=snap_x,
                session_id=session_id,
                vol_regime_id=vol_regime_id,
                trend_regime_id=trend_regime_id,
            )

            loss_direction = criterion_direction(outputs["direction_logit"], y_direction)
            loss_early_move = criterion_early_move(outputs["early_move_logit"], y_early_move)
            loss_quality = criterion_quality(outputs["quality_score"], y_quality_score)

            total_loss_batch = (
                loss_weights["direction"] * loss_direction
                + loss_weights["early_move"] * loss_early_move
                + loss_weights["quality"] * loss_quality
            )

            total_loss += total_loss_batch.item() * seq_x.size(0)
            total_samples += seq_x.size(0)

            # Collect predictions for metrics
            probs_direction = torch.sigmoid(outputs["direction_logit"])
            probs_early_move = torch.sigmoid(outputs["early_move_logit"])

            all_direction_preds.append(probs_direction.cpu().numpy())
            all_direction_labels.append(y_direction.cpu().numpy())
            all_early_move_preds.append(probs_early_move.cpu().numpy())
            all_early_move_labels.append(y_early_move.cpu().numpy())
            all_quality_preds.append(outputs["quality_score"].cpu().numpy())
            all_quality_labels.append(y_quality_score.cpu().numpy())

    avg_loss = total_loss / max(1, total_samples)

    # Calculate metrics
    if len(all_direction_preds) == 0:
        direction_auc = 0.5
        early_move_auc = 0.5
        quality_mse = 0.0
        coverage = 0.0
    else:
        all_direction_preds = np.concatenate(all_direction_preds)
        all_direction_labels = np.concatenate(all_direction_labels)
        all_early_move_preds = np.concatenate(all_early_move_preds)
        all_early_move_labels = np.concatenate(all_early_move_labels)
        all_quality_preds = np.concatenate(all_quality_preds)
        all_quality_labels = np.concatenate(all_quality_labels)

        # Direction AUC
        if len(np.unique(all_direction_labels)) > 1:
            direction_auc = float(roc_auc_score(all_direction_labels, all_direction_preds))
        else:
            direction_auc = 0.5

        # Early move AUC
        if len(np.unique(all_early_move_labels)) > 1:
            early_move_auc = float(roc_auc_score(all_early_move_labels, all_early_move_preds))
        else:
            early_move_auc = 0.5

        # Quality MSE
        quality_mse = float(mean_squared_error(all_quality_labels, all_quality_preds))

        # Coverage (direction positive rate)
        coverage = float(np.mean(all_direction_preds > 0.5))

    return avg_loss, direction_auc, coverage, quality_mse


def debug_shuffle_labels_once(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    config: Dict[str, Any],
    device: torch.device,
    metadata: Dict[str, Any],
    seq_feat_names: List[str],
    snap_feat_names: List[str],
    seq_len: int,
    batch_size: int,
    num_workers: int,
) -> None:
    """
    Leakage test: Train a mini-model on shuffled val labels.
    Expected AUC ≈ 0.5 if no leakage.
    """
    print("[LEAKAGE_TEST] Creating shuffled validation set...")
    
    # Copy val set and shuffle y_direction
    df_val_shuffled = df_val.copy()
    np.random.seed(42)  # Reproducible shuffle
    df_val_shuffled["y_direction"] = np.random.permutation(df_val_shuffled["y_direction"].values)
    
    print(f"[LEAKAGE_TEST] Original val positive rate: {df_val['y_direction'].mean():.4f}")
    print(f"[LEAKAGE_TEST] Shuffled val positive rate: {df_val_shuffled['y_direction'].mean():.4f}")
    
    # Create mini model config (smaller, faster)
    mini_model_cfg = config["model"].copy()
    mini_model_cfg["name"] = "entry_v9"
    mini_model_cfg["seq_input_dim"] = len(seq_feat_names)
    mini_model_cfg["snap_input_dim"] = len(snap_feat_names)
    mini_model_cfg["seq_cfg"] = {
        "d_model": 32,  # Smaller
        "n_heads": 2,
        "num_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.1,
    }
    
    # Build mini model
    mini_model = build_entry_v9_model({"model": mini_model_cfg}).to(device)
    print(f"[LEAKAGE_TEST] Mini model built: {sum(p.numel() for p in mini_model.parameters())} parameters")
    
    # Create dataloaders
    train_dataset = EntryV9Dataset(
        df_train, seq_feat_names, snap_feat_names, seq_len=seq_len, lookback=seq_len
    )
    val_shuffled_dataset = EntryV9Dataset(
        df_val_shuffled, seq_feat_names, snap_feat_names, seq_len=seq_len, lookback=seq_len
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_shuffled_loader = DataLoader(
        val_shuffled_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    # Mini training (3 epochs)
    mini_epochs = 3
    mini_lr = 1e-3
    optimizer = torch.optim.AdamW(mini_model.parameters(), lr=mini_lr)
    loss_weights = config["training"].get("loss_weights", {"direction": 0.5, "early_move": 0.3, "quality": 0.2})
    
    print(f"[LEAKAGE_TEST] Training mini model for {mini_epochs} epochs...")
    for epoch in range(1, mini_epochs + 1):
        train_one_epoch(mini_model, train_loader, optimizer, device, loss_weights)
    
    # Evaluate on shuffled val
    _, shuffled_auc, _, _ = evaluate_entry_v9(mini_model, val_shuffled_loader, device, loss_weights)
    
    print(f"[LEAKAGE_TEST] Shuffled-label val AUC = {shuffled_auc:.4f}")
    
    if shuffled_auc > 0.55:
        raise RuntimeError(
            f"Potential leakage detected! Shuffled-label AUC = {shuffled_auc:.4f} > 0.55. "
            "This suggests the model is learning from future information."
        )
    else:
        print(f"[LEAKAGE_TEST] ✅ PASSED: Shuffled-label AUC = {shuffled_auc:.4f} ≈ 0.5 (no leakage detected)")


def train_entry_v9_model(
    config: Dict[str, Any],
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train ENTRY_V9 model."""
    print("[ENTRY_V9] Starting training...")

    # Prepare data
    df, metadata = prepare_entry_v9_data(config, generate_labels=True)

    # Time-based split: train=2020-2024, val=2025
    split_cfg = config.get("dataset", {}).get("split", {})
    if split_cfg.get("type") == "time":
        train_end = split_cfg.get("train_end", "2024-12-31")
        val_start = split_cfg.get("val_start", "2025-01-01")
        df_train, df_val = time_split_by_dates(df, train_end, val_start, ts_col="ts")
    else:
        # Fallback to ratio split
        from gx1.tuning.entry_v4_train import time_split_by_ratio
        df_train, df_val = time_split_by_ratio(df, ratio=0.8)

    print(f"[ENTRY_V9] Train: {len(df_train)} rows, Val: {len(df_val)} rows")
    
    # LEAKAGE DEBUG: Log dataset split details
    print("="*80)
    print("[LEAKAGE_DEBUG] Dataset Split Details:")
    print("="*80)
    if "ts" in df_train.columns:
        train_ts_min = df_train["ts"].min()
        train_ts_max = df_train["ts"].max()
        val_ts_min = df_val["ts"].min()
        val_ts_max = df_val["ts"].max()
        print(f"Train time range: {train_ts_min} to {train_ts_max}")
        print(f"Val time range: {val_ts_min} to {val_ts_max}")
    else:
        print("⚠️  'ts' column not found, cannot log time ranges")
    
    train_pos_rate = df_train["y_direction"].mean() if "y_direction" in df_train.columns else 0.0
    val_pos_rate = df_val["y_direction"].mean() if "y_direction" in df_val.columns else 0.0
    print(f"Train positive label rate: {train_pos_rate:.4f} ({df_train['y_direction'].sum()}/{len(df_train)})")
    print(f"Val positive label rate: {val_pos_rate:.4f} ({df_val['y_direction'].sum()}/{len(df_val)})")
    print("="*80)

    # Ensure output directory exists
    output_dir = Path(config["output"]["model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build scalers (fit on training data only)
    seq_feat_names = metadata["seq_feature_names"]
    snap_feat_names = metadata["snap_feature_names"]

    # Scale sequence features
    seq_scaler = RobustScaler()
    seq_X_train = df_train[seq_feat_names].values
    seq_X_val = df_val[seq_feat_names].values
    seq_scaler.fit(seq_X_train)
    seq_X_train_scaled = seq_scaler.transform(seq_X_train)
    seq_X_val_scaled = seq_scaler.transform(seq_X_val)

    # Update DataFrame with scaled features
    for i, feat_name in enumerate(seq_feat_names):
        df_train[feat_name] = seq_X_train_scaled[:, i]
        df_val[feat_name] = seq_X_val_scaled[:, i]

    # Scale snapshot features
    snap_scaler = RobustScaler()
    snap_X_train = df_train[snap_feat_names].values
    snap_X_val = df_val[snap_feat_names].values
    snap_scaler.fit(snap_X_train)
    snap_X_train_scaled = snap_scaler.transform(snap_X_train)
    snap_X_val_scaled = snap_scaler.transform(snap_X_val)

    # Update DataFrame with scaled features
    for i, feat_name in enumerate(snap_feat_names):
        df_train[feat_name] = snap_X_train_scaled[:, i]
        df_val[feat_name] = snap_X_val_scaled[:, i]

    # Save scalers
    seq_scaler_path = output_dir / "seq_scaler.joblib"
    snap_scaler_path = output_dir / "snap_scaler.joblib"
    dump(seq_scaler, seq_scaler_path)
    dump(snap_scaler, snap_scaler_path)
    print(f"[ENTRY_V9] Saved scalers to {seq_scaler_path}, {snap_scaler_path}")

    # Save feature metadata
    feature_meta = {
        "seq_features": sorted(seq_feat_names.copy()),
        "snap_features": sorted(snap_feat_names.copy()),
        "version": "entry_v9_v1",
        "seq_input_dim": len(seq_feat_names),
        "snap_input_dim": len(snap_feat_names),
    }
    feature_meta_path = output_dir / "entry_v9_feature_meta.json"
    with open(feature_meta_path, "w") as f:
        json.dump(feature_meta, f, indent=2)
    print(f"[ENTRY_V9] Saved feature metadata to {feature_meta_path}")

    # Create dataloaders
    seq_len = config["model"]["max_seq_len"]
    batch_size = config["training"]["batch_size"]
    # Use num_workers=0 to avoid multiprocessing issues with pandas
    num_workers = config["training"].get("num_workers", 0)

    train_loader, val_loader = create_entry_v9_dataloaders(
        df_train=df_train,
        df_val=df_val,
        seq_feature_names=seq_feat_names,
        snap_feature_names=snap_feat_names,
        seq_len=seq_len,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    print(f"[ENTRY_V9] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Build model
    model_cfg = config["model"].copy()
    # Ensure name is set (CRITICAL for build_entry_v9_model)
    model_cfg["name"] = "entry_v9"
    model_cfg["seq_input_dim"] = len(seq_feat_names)
    model_cfg["snap_input_dim"] = len(snap_feat_names)

    # build_entry_v9_model expects config with "model" key, but we pass model_cfg directly
    # So we need to wrap it or ensure it has name
    model = build_entry_v9_model({"model": model_cfg}).to(device)
    print(f"[ENTRY_V9] Model built: {sum(p.numel() for p in model.parameters())} parameters")

    # Training config
    training_cfg = config["training"]
    num_epochs = training_cfg["num_epochs"]
    learning_rate = float(training_cfg["learning_rate"])  # Ensure it's a float
    loss_weights = training_cfg.get("loss_weights", {"direction": 0.5, "early_move": 0.3, "quality": 0.2})

    # Optimizer and scheduler
    weight_decay = float(training_cfg.get("weight_decay", 1e-5))  # Ensure it's a float
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Early stopping
    best_val_auc = 0.0
    best_epoch = 0
    patience = training_cfg.get("patience", 10)
    patience_counter = 0

    # LEAKAGE DEBUG: Shuffle labels test (if enabled)
    debug_shuffle_labels = config.get("debug", {}).get("shuffle_labels", False)
    if debug_shuffle_labels:
        print("\n" + "="*80)
        print("[LEAKAGE_DEBUG] Running shuffled labels test...")
        print("="*80)
        debug_shuffle_labels_once(df_train, df_val, config, device, metadata, seq_feat_names, snap_feat_names, seq_len, batch_size, num_workers)
        print("="*80 + "\n")

    # Training loop
    print(f"[ENTRY_V9] Starting training for {num_epochs} epochs...")
    for epoch in range(1, num_epochs + 1):
        train_loss, train_coverage = train_one_epoch(
            model, train_loader, optimizer, device, loss_weights
        )
        
        # LEAKAGE DEBUG: Evaluate on both train and val
        train_loss_eval, train_auc, train_coverage_eval, train_quality_mse = evaluate_entry_v9(
            model, train_loader, device, loss_weights
        )
        val_loss, val_auc, val_coverage, val_quality_mse = evaluate_entry_v9(
            model, val_loader, device, loss_weights
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # LEAKAGE DEBUG: Log both train and val AUC
        print(
            f"[ENTRY_V9] Epoch {epoch:03d}/{num_epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"train_auc={train_auc:.4f} | val_auc={val_auc:.4f} | "
            f"coverage={val_coverage:.4f} | quality_mse={val_quality_mse:.4f} | lr={current_lr:.6f}"
        )
        
        # LEAKAGE DEBUG: Warning if val_auc is suspiciously high
        if val_auc > 0.95:
            print(f"[LEAKAGE_WARNING] ⚠️  val_auc={val_auc:.4f} > 0.95, this is suspicious – check for leakage!")

        # Early stopping on val_auc
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "val_coverage": val_coverage,
                "config": config,
            }
            checkpoint_path = output_dir / "model.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"[ENTRY_V9] ✅ New best: val_auc={val_auc:.4f} (epoch {epoch})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[ENTRY_V9] Early stopping at epoch {epoch} (patience={patience})")
                break

        # Save periodic checkpoints
        if epoch % 10 == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)

    # Load best model
    best_checkpoint = torch.load(output_dir / "model.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    print(f"[ENTRY_V9] Loaded best model from epoch {best_epoch} (val_auc={best_val_auc:.4f})")

    # Save metadata
    # Get final train AUC for metadata
    _, final_train_auc, _, _ = evaluate_entry_v9(model, train_loader, device, loss_weights)
    
    meta = {
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "final_train_auc": final_train_auc,
        "best_val_coverage": best_checkpoint.get("val_coverage", 0.0),
        "training_period": f"{train_end} to {val_start}",
        "n_train": len(df_train),
        "n_val": len(df_val),
    }
    meta_path = output_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[ENTRY_V9] Saved metadata to {meta_path}")

    return model, meta


def main():
    parser = argparse.ArgumentParser(description="Train ENTRY_V9 NEXTGEN model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ENTRY_V9] Using device: {device}")

    # Train
    model, meta = train_entry_v9_model(config, device)

    print("\n" + "=" * 80)
    print("ENTRY_V9 TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best epoch: {meta['best_epoch']}")
    print(f"Best val AUC: {meta['best_val_auc']:.4f}")
    print(f"Model saved to: {config['output']['model_dir']}")
    print("=" * 80)


if __name__ == "__main__":
    main()

