"""
ENTRY_V9 NEXTGEN - Transformer-based entry model with multi-task learning and regime-conditioning.

Components:
- entry_v9_transformer: Model architecture with multi-task heads and regime embeddings
- entry_v9_dataset: Dataset loader with embeddings and labels
- entry_v9_labeler: Label generation for direction, early_move, quality_score
- entry_v9_train: Training script with multi-task loss
- entry_v9_optuna: Hyperparameter tuning with Optuna
"""

from gx1.models.entry_v9.entry_v9_transformer import (
    EntryV9Transformer,
    build_entry_v9_model,
)

__all__ = [
    "EntryV9Transformer",
    "build_entry_v9_model",
]

