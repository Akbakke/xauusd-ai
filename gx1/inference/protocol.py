#!/usr/bin/env python3
"""
Protocol definitions for ModelWorker communication.

Defines request/response structures for batch inference communication
between runner and worker process.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class InferenceRequest:
    """Request for batch inference."""
    seq_features: np.ndarray  # [batch_size, seq_len, seq_dim] float32
    snap_features: np.ndarray  # [batch_size, snap_dim] float32
    session_ids: np.ndarray  # [batch_size] int32 (0=EU, 1=OVERLAP, 2=US)
    vol_regime_ids: np.ndarray  # [batch_size] int32 (0=LOW, 1=MEDIUM, 2=HIGH, 3=EXTREME)
    trend_regime_ids: np.ndarray  # [batch_size] int32 (0=UP, 1=DOWN, 2=NEUTRAL)
    request_id: int  # Unique request ID for tracking


@dataclass
class InferenceResponse:
    """Response from batch inference."""
    p_long: np.ndarray  # [batch_size] float32 (probability of long entry)
    direction_logit: Optional[np.ndarray] = None  # [batch_size] float32 (optional)
    early_move_logit: Optional[np.ndarray] = None  # [batch_size] float32 (optional)
    quality_score: Optional[np.ndarray] = None  # [batch_size] float32 (optional)
    request_id: int = 0  # Echo of request ID
    error: Optional[str] = None  # Error message if inference failed


@dataclass
class WorkerConfig:
    """Configuration for ModelWorker."""
    checkpoint_path: str
    meta_path: str  # Transformer metadata path (not feature metadata)
    variant: str = "v10_1"
    device: str = "cpu"  # Always "cpu" for now
    timeout_seconds: float = 30.0  # Timeout for inference operations

