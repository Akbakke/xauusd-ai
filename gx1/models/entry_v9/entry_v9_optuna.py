#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gx1/models/entry_v9/entry_v9_optuna.py

Optuna hyperparameter tuning for ENTRY_V9 NEXTGEN.

CLI:
    python -m gx1.models.entry_v9.entry_v9_optuna --config gx1/configs/entry_v9/entry_v9_optuna.yaml --trials 50
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, Any

import optuna
import torch
import torch.nn as nn

from gx1.models.entry_v9.entry_v9_train import train_entry_v9_model
from gx1.tuning.entry_v4_train import set_seed


def objective(trial: optuna.Trial, base_config: Dict[str, Any], device: torch.device) -> float:
    """Optuna objective function for ENTRY_V9 tuning."""
    # Suggest hyperparameters
    d_model = trial.suggest_categorical("d_model", [64, 96, 128])
    n_heads = trial.suggest_categorical("n_heads", [4, 8])
    num_layers = trial.suggest_int("num_layers", 3, 5)
    dim_feedforward = trial.suggest_categorical("dim_feedforward", [256, 512, 768])
    dropout = trial.suggest_float("dropout", 0.05, 0.15, step=0.05)

    # Update config with suggested parameters (deep copy to avoid modifying base)
    import copy
    config = copy.deepcopy(base_config)
    # Ensure model section exists and name is set
    if "model" not in config:
        config["model"] = {}
    config["model"]["name"] = "entry_v9"  # CRITICAL: Must be set before build_entry_v9_model
    if "seq_cfg" not in config["model"]:
        config["model"]["seq_cfg"] = {}
    # Update seq_cfg with Optuna suggestions
    config["model"]["seq_cfg"] = {
        "d_model": d_model,
        "n_heads": n_heads,
        "num_layers": num_layers,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
    }

    # Training hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)

    config["training"]["learning_rate"] = learning_rate
    config["training"]["batch_size"] = batch_size
    config["training"]["weight_decay"] = weight_decay

    # Loss weights
    direction_weight = trial.suggest_float("direction_weight", 0.4, 0.6, step=0.1)
    early_move_weight = trial.suggest_float("early_move_weight", 0.2, 0.4, step=0.1)
    quality_weight = 1.0 - direction_weight - early_move_weight

    config["training"]["loss_weights"] = {
        "direction": direction_weight,
        "early_move": early_move_weight,
        "quality": quality_weight,
    }

    try:
        # Train model
        model, meta = train_entry_v9_model(config, device)

        # Return validation AUC (to maximize)
        val_auc = meta["best_val_auc"]
        train_auc = meta.get("final_train_auc", 0.0)

        # LEAKAGE DEBUG: Log both train and val AUC
        print(f"[OPTUNA] Trial {trial.number}: AUC_train={train_auc:.4f}, AUC_val={val_auc:.4f}")
        
        # LEAKAGE DEBUG: Warning if val_auc is suspiciously high
        if val_auc > 0.95:
            print(f"[OPTUNA WARNING] ⚠️  AUC_val={val_auc:.4f} > 0.95, this is suspicious – check leakage.")

        # Store additional metrics as user attributes
        trial.set_user_attr("val_coverage", meta.get("best_val_coverage", 0.0))
        trial.set_user_attr("best_epoch", meta.get("best_epoch", 0))
        trial.set_user_attr("train_auc", train_auc)

        return val_auc

    except Exception as e:
        import traceback
        print(f"[ENTRY_V9_OPTUNA] Trial failed: {e}")
        print(f"[ENTRY_V9_OPTUNA] Full traceback:")
        traceback.print_exc()
        return 0.0  # Return worst possible score


def main():
    parser = argparse.ArgumentParser(description="Optuna tuning for ENTRY_V9")
    parser.add_argument("--config", type=str, required=True, help="Path to base config YAML")
    parser.add_argument("--trials", type=int, default=30, help="Number of trials")
    parser.add_argument("--n-workers", type=int, default=6, help="Number of parallel workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--study-name", type=str, default="entry_v9_optuna", help="Optuna study name")
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Load base config
    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ENTRY_V9_OPTUNA] Using device: {device}")

    # Create study
    study_dir = Path(base_config["output"]["model_dir"]).parent / "optuna"
    study_dir.mkdir(parents=True, exist_ok=True)

    storage = f"sqlite:///{study_dir / 'optuna.db'}"
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
    )

    # Run optimization
    print(f"[ENTRY_V9_OPTUNA] Starting optimization with {args.trials} trials...")
    # Use n_jobs=1 for sequential trials (parallel training within each trial uses num_workers)
    study.optimize(
        lambda trial: objective(trial, base_config, device),
        n_trials=args.trials,
        n_jobs=1,  # Sequential trials (each trial uses num_workers from training config)
    )

    # Save best parameters
    best_params = study.best_params
    best_value = study.best_value

    best_params_path = study_dir / "best_params.json"
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    print("\n" + "=" * 80)
    print("ENTRY_V9 OPTUNA TUNING COMPLETE")
    print("=" * 80)
    print(f"Best value (val_auc): {best_value:.4f}")
    print(f"Best parameters saved to: {best_params_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

