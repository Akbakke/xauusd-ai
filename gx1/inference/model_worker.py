#!/usr/bin/env python3
"""
ModelWorker - Isolated process for PyTorch model loading and inference.

This worker process:
1. Imports torch (only here, never in runner)
2. Loads CANONICAL checkpoint
3. Runs batch inference
4. Communicates via multiprocessing.Queue

Entry point for worker process (spawned by client).
"""

import logging
import os
import queue
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

# CRITICAL: Import torch ONLY in worker process (never in runner)
import torch

from gx1.inference.protocol import InferenceRequest, InferenceResponse, WorkerConfig

log = logging.getLogger(__name__)


def worker_main(
    config: WorkerConfig,
    request_queue: Any,  # multiprocessing.Queue
    response_queue: Any,  # multiprocessing.Queue
) -> None:
    """
    Main event loop for ModelWorker process.
    
    Parameters
    ----------
    config : WorkerConfig
        Worker configuration (checkpoint paths, device, etc.)
    request_queue : multiprocessing.Queue
        Queue for receiving inference requests
    response_queue : multiprocessing.Queue
        Queue for sending inference responses
    """
    log.info("[MODEL_WORKER] Starting worker process (PID=%d)", os.getpid())
    
    try:
        # Force CPU device (always for now)
        device = torch.device(config.device)
        log.info("[MODEL_WORKER] Using device: %s", device)
        
        # Disable MPS backend explicitly (belt and suspenders)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.backends.mps.enabled = False
            log.info("[MODEL_WORKER] MPS backend explicitly disabled")
        
        # Load Transformer model directly (XGBoost stays in runner)
        log.info("[MODEL_WORKER] Loading Transformer model...")
        import json
        from gx1.models.entry_v10.entry_v10_hybrid_transformer import EntryV10HybridTransformer
        
        # Load metadata
        meta_path = Path(config.meta_path)
        if not meta_path.exists():
            # Fallback: try to find metadata next to checkpoint
            checkpoint_path = Path(config.checkpoint_path)
            meta_path = checkpoint_path.parent / (checkpoint_path.stem.replace("_CANONICAL", "") + "_meta.json")
        if not meta_path.exists():
            # Fallback: try default names
            checkpoint_path = Path(config.checkpoint_path)
            meta_path = checkpoint_path.parent / "entry_v10_1_transformer_meta.json"
        if not meta_path.exists():
            checkpoint_path = Path(config.checkpoint_path)
            meta_path = checkpoint_path.parent / "entry_v10_transformer_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"[MODEL_WORKER] Transformer metadata not found: {config.meta_path}")
        
        with open(meta_path, "r") as f:
            transformer_meta = json.load(f)
        
        # Get variant from config or metadata
        variant = config.variant or transformer_meta.get("variant", "v10")
        
        # Build model
        seq_input_dim = transformer_meta.get("seq_feature_count", 16)
        snap_input_dim = transformer_meta.get("snap_feature_count", 88)
        max_seq_len = transformer_meta.get("seq_len", 30)
        
        log.info(f"[MODEL_WORKER] Creating model (seq_dim={seq_input_dim}, snap_dim={snap_input_dim}, seq_len={max_seq_len}, variant={variant})...")
        transformer_model = EntryV10HybridTransformer(
            seq_input_dim=seq_input_dim,
            snap_input_dim=snap_input_dim,
            max_seq_len=max_seq_len,
            variant=variant,
            enable_auxiliary_heads=True,
        )
        transformer_model = transformer_model.to(device)
        log.info(f"[MODEL_WORKER] Model created on {device}")
        
        # Load checkpoint
        checkpoint_path = Path(config.checkpoint_path)
        log.info(f"[MODEL_WORKER] Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Get state_dict
        state_dict = checkpoint["model_state_dict"] if (isinstance(checkpoint, dict) and "model_state_dict" in checkpoint) else checkpoint
        
        # Filter out causal_mask (we create it on-demand)
        filtered_state_dict = {k: v for k, v in state_dict.items() if "causal_mask" not in k}
        cpu_state_dict = {k: v.cpu() if hasattr(v, 'cpu') else v for k, v in filtered_state_dict.items()}
        
        # Load state_dict (strict=True for CANONICAL)
        missing_keys, unexpected_keys = transformer_model.load_state_dict(cpu_state_dict, strict=True)
        if missing_keys:
            log.warning(f"[MODEL_WORKER] Missing keys: {missing_keys[:5]}...")
        if unexpected_keys:
            log.warning(f"[MODEL_WORKER] Unexpected keys: {unexpected_keys[:5]}...")
        
        log.info("[MODEL_WORKER] Checkpoint loaded successfully")
        
        # Set model to eval mode
        transformer_model.eval()
        log.info("[MODEL_WORKER] Model set to eval() mode")
        
        lookback = max_seq_len
        
        # Main event loop
        log.info("[MODEL_WORKER] Entering inference loop...")
        request_id_counter = 0
        
        while True:
            try:
                # Get request (blocking)
                request: Optional[InferenceRequest] = request_queue.get(timeout=1.0)
                
                if request is None:
                    # Sentinel: shutdown
                    log.info("[MODEL_WORKER] Received shutdown signal")
                    break
                
                request_id_counter += 1
                log.debug("[MODEL_WORKER] Processing request ID=%d (batch_size=%d)", request.request_id, len(request.seq_features))
                
                # Perform inference
                start_time = time.perf_counter()
                
                try:
                    # Convert numpy arrays to torch tensors
                    batch_size = len(request.seq_features)
                    
                    seq_t = torch.FloatTensor(request.seq_features).to(device)  # [batch_size, seq_len, seq_dim]
                    snap_t = torch.FloatTensor(request.snap_features).to(device)  # [batch_size, snap_dim]
                    session_t = torch.LongTensor(request.session_ids).to(device)  # [batch_size]
                    vol_t = torch.LongTensor(request.vol_regime_ids).to(device)  # [batch_size]
                    trend_t = torch.LongTensor(request.trend_regime_ids).to(device)  # [batch_size]
                    
                    # Run inference
                    with torch.no_grad():
                        outputs = transformer_model(seq_t, snap_t, session_t, vol_t, trend_t)
                        direction_logit = outputs["direction_logit"]  # [batch_size, 1]
                    
                    # Convert to probabilities
                    p_long = torch.sigmoid(direction_logit).cpu().numpy().flatten()  # [batch_size]
                    
                    inference_time = time.perf_counter() - start_time
                    log.debug("[MODEL_WORKER] Inference completed in %.3f ms (batch_size=%d)", inference_time * 1000, batch_size)
                    
                    # Build response
                    response = InferenceResponse(
                        p_long=p_long.astype(np.float32),
                        request_id=request.request_id,
                    )
                    
                except Exception as e:
                    log.error("[MODEL_WORKER] Inference error: %s", e, exc_info=True)
                    response = InferenceResponse(
                        p_long=np.zeros(batch_size, dtype=np.float32),
                        request_id=request.request_id,
                        error=str(e),
                    )
                
                # Send response
                response_queue.put(response)
                
            except queue.Empty:
                # Normal timeout when waiting for request, just continue loop
                continue
            except Exception as e:
                log.error("[MODEL_WORKER] Error in event loop: %s", e, exc_info=True)
                # Send error response if we have a request
                if 'request' in locals():
                    error_response = InferenceResponse(
                        p_long=np.zeros(len(request.seq_features), dtype=np.float32),
                        request_id=request.request_id,
                        error=str(e),
                    )
                    response_queue.put(error_response)
    
    except Exception as e:
        log.error("[MODEL_WORKER] Fatal error in worker: %s", e, exc_info=True)
        # Send fatal error response
        fatal_response = InferenceResponse(
            p_long=np.array([], dtype=np.float32),
            request_id=0,
            error=f"Fatal worker error: {str(e)}",
        )
        try:
            response_queue.put(fatal_response)
        except Exception:
            pass
    finally:
        log.info("[MODEL_WORKER] Worker process exiting (PID=%d)", os.getpid())


if __name__ == "__main__":
    # Standalone entry point for testing (not used in production)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log.error("[MODEL_WORKER] This module should not be run directly. Use gx1.inference.client instead.")
    sys.exit(1)

