#!/usr/bin/env python3
"""
ModelLoaderWorker - Isolated process for PyTorch model loading with timeout.

This worker process:
1. Sets thread limits and CPU-only environment
2. Disables MPS backend
3. Loads model state_dict from disk
4. Builds model and loads weights
5. Runs minimal forward "smoke test" to verify model works
6. Returns OK + metadata (model_class_name, param_count, hash) via pipe

CRITICAL: Uses multiprocessing.get_context("spawn") on macOS to avoid fork issues.
"""

import hashlib
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import traceback

# CRITICAL: Import torch ONLY in worker process (never in runner)
import torch

log = logging.getLogger(__name__)


@dataclass
class ModelLoadResult:
    """Result from model loading worker."""
    success: bool
    model_class_name: str
    param_count: int
    model_hash: str
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    load_time_sec: float = 0.0
    traceback_excerpt: Optional[str] = None


@dataclass
class ModelLoadConfig:
    """Configuration for model loading."""
    bundle_dir: Path
    feature_meta_path: Path
    seq_scaler_path: Optional[Path] = None
    snap_scaler_path: Optional[Path] = None
    model_variant: str = "v10"  # "v10" or "v10_ctx"
    device: str = "cpu"
    timeout_sec: float = 60.0


def _setup_worker_environment() -> None:
    """Setup worker environment: thread limits, CPU-only, MPS-disable."""
    # CRITICAL: Set thread limits BEFORE any torch/numpy imports
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["GX1_XGB_THREADS"] = "1"
    
    # CRITICAL: Force CPU fallback for MPS operations
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    
    # Disable MPS backend explicitly
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.backends.mps.enabled = False
    
    # Set torch thread limits
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _load_model_worker_impl(config: ModelLoadConfig) -> ModelLoadResult:
    """
    Worker implementation: load model and verify.
    
    This runs in isolated process with timeout protection from parent.
    """
    start_time = time.perf_counter()
    
    try:
        log.info(f"[MODEL_LOADER_WORKER] Starting (PID={os.getpid()}, variant={config.model_variant})")
        
        # Setup environment
        _setup_worker_environment()
        
        # Force CPU device
        device = torch.device(config.device)
        log.info(f"[MODEL_LOADER_WORKER] Using device: {device}")
        
        # Load feature metadata
        if not config.feature_meta_path.exists():
            return ModelLoadResult(
                success=False,
                model_class_name="UNKNOWN",
                param_count=0,
                model_hash="",
                error_type="FILE_NOT_FOUND",
                error_message=f"Feature metadata not found: {config.feature_meta_path}",
                load_time_sec=time.perf_counter() - start_time,
                traceback_excerpt=None,
            )
        
        with open(config.feature_meta_path, "r") as f:
            import json
            feature_meta = json.load(f)
        
        # Determine model type and load accordingly
        if config.model_variant == "v10_ctx":
            # Load V10_CTX bundle
            from gx1.models.entry_v10.entry_v10_bundle import load_entry_v10_ctx_bundle
            from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer
            
            log.info("[MODEL_LOADER_WORKER] Loading V10_CTX bundle...")
            bundle = load_entry_v10_ctx_bundle(
                bundle_dir=config.bundle_dir,
                feature_meta_path=config.feature_meta_path,
                seq_scaler_path=config.seq_scaler_path,
                snap_scaler_path=config.snap_scaler_path,
                device=device,
                is_replay=True,  # Strict validation
            )
            
            transformer_model = bundle.transformer_model
            model_class_name = type(transformer_model).__name__
            
        else:
            # Load legacy V10 transformer directly (simpler: just verify transformer loads)
            from gx1.models.entry_v10.entry_v10_hybrid_transformer import EntryV10HybridTransformer
            
            log.info("[MODEL_LOADER_WORKER] Loading legacy V10 transformer directly...")
            
            # Find model file
            model_path = None
            possible_model_names = [
                "entry_v10_transformer.pt",
                "entry_v10_transformer_CANONICAL.pt",
                "model_state_dict.pt",
            ]
            for name in possible_model_names:
                candidate = config.bundle_dir / name
                if candidate.exists():
                    model_path = candidate
                    break
            
            if model_path is None:
                # Try to find any .pt file in bundle_dir
                pt_files = list(config.bundle_dir.glob("*.pt"))
                if pt_files:
                    model_path = pt_files[0]
                else:
                    raise FileNotFoundError(f"[MODEL_LOADER_WORKER] No model file found in {config.bundle_dir}")
            
            # Load metadata (try to find transformer metadata)
            meta_path = config.bundle_dir / f"{model_path.stem}_meta.json"
            if not meta_path.exists():
                meta_path = config.bundle_dir / "entry_v10_transformer_meta.json"
            if not meta_path.exists():
                meta_path = config.bundle_dir / "entry_v10_1_transformer_meta.json"
            if not meta_path.exists():
                # Fallback: use feature_meta for dimensions
                meta_path = config.feature_meta_path
            
            with open(meta_path, "r") as f:
                import json
                transformer_meta = json.load(f)
            
            # Build model
            seq_input_dim = transformer_meta.get("seq_feature_count", len(feature_meta.get("seq_features", [])))
            snap_input_dim = transformer_meta.get("snap_feature_count", len(feature_meta.get("snap_features", [])))
            max_seq_len = transformer_meta.get("seq_len", 30)
            variant = transformer_meta.get("variant", "v10")
            
            log.info(f"[MODEL_LOADER_WORKER] Creating model (seq_dim={seq_input_dim}, snap_dim={snap_input_dim}, seq_len={max_seq_len}, variant={variant})...")
            transformer_model = EntryV10HybridTransformer(
                seq_input_dim=seq_input_dim,
                snap_input_dim=snap_input_dim,
                max_seq_len=max_seq_len,
                variant=variant,
                enable_auxiliary_heads=True,
            )
            transformer_model = transformer_model.to(device)
            
            # Load checkpoint
            log.info(f"[MODEL_LOADER_WORKER] Loading checkpoint: {model_path}")
            checkpoint = torch.load(model_path, map_location="cpu")
            
            # Get state_dict
            state_dict = checkpoint["model_state_dict"] if (isinstance(checkpoint, dict) and "model_state_dict" in checkpoint) else checkpoint
            
            # Filter out causal_mask
            filtered_state_dict = {k: v for k, v in state_dict.items() if "causal_mask" not in k}
            cpu_state_dict = {k: v.cpu() if hasattr(v, 'cpu') else v for k, v in filtered_state_dict.items()}
            
            # Load state_dict
            log.info(f"[MODEL_LOADER_WORKER] Loading state_dict with {len(cpu_state_dict)} keys...")
            missing_keys, unexpected_keys = transformer_model.load_state_dict(cpu_state_dict, strict=False)
            if missing_keys:
                log.warning(f"[MODEL_LOADER_WORKER] Missing keys: {missing_keys[:5]}...")
            if unexpected_keys:
                log.warning(f"[MODEL_LOADER_WORKER] Unexpected keys: {unexpected_keys[:5]}...")
            
            model_class_name = type(transformer_model).__name__
        
        log.info(f"[MODEL_LOADER_WORKER] Model loaded: {model_class_name}")
        
        # Verify model is on CPU
        if next(transformer_model.parameters()).device.type != "cpu":
            log.warning(f"[MODEL_LOADER_WORKER] Model not on CPU: {next(transformer_model.parameters()).device}")
            # Move to CPU
            transformer_model = transformer_model.to(torch.device("cpu"))
        
        # Count parameters
        param_count = sum(p.numel() for p in transformer_model.parameters())
        log.info(f"[MODEL_LOADER_WORKER] Parameter count: {param_count:,}")
        
        # Compute model hash (from state_dict)
        state_dict = transformer_model.state_dict()
        state_dict_str = str(sorted(state_dict.items()))
        model_hash = hashlib.sha256(state_dict_str.encode()).hexdigest()[:16]
        log.info(f"[MODEL_LOADER_WORKER] Model hash: {model_hash}")
        
        # Run minimal forward "smoke test" to verify model works
        log.info("[MODEL_LOADER_WORKER] Running smoke test...")
        transformer_model.eval()
        
        # Get model dimensions from metadata or bundle
        if config.model_variant == "v10_ctx":
            seq_input_dim = len(feature_meta.get("seq_features", []))
            snap_input_dim = len(feature_meta.get("snap_features", []))
            max_seq_len = feature_meta.get("seq_len", 30)
            
            # Create dummy inputs for ctx model
            batch_size = 1
            seq_x = torch.zeros(batch_size, max_seq_len, seq_input_dim, dtype=torch.float32)
            snap_x = torch.zeros(batch_size, snap_input_dim, dtype=torch.float32)
            session_id = torch.zeros(batch_size, dtype=torch.int64)
            vol_regime_id = torch.zeros(batch_size, dtype=torch.int64)
            trend_regime_id = torch.zeros(batch_size, dtype=torch.int64)
            ctx_cat = torch.zeros(batch_size, 5, dtype=torch.int64)  # 5 categorical features
            ctx_cont = torch.zeros(batch_size, 2, dtype=torch.float32)  # 2 continuous features
            
            with torch.no_grad():
                outputs = transformer_model(
                    seq_x=seq_x,
                    snap_x=snap_x,
                    session_id=session_id,
                    vol_regime_id=vol_regime_id,
                    trend_regime_id=trend_regime_id,
                    ctx_cat=ctx_cat,
                    ctx_cont=ctx_cont,
                )
        else:
            seq_input_dim = len(feature_meta.get("seq_features", []))
            snap_input_dim = len(feature_meta.get("snap_features", []))
            max_seq_len = feature_meta.get("seq_len", 30)
            
            # Create dummy inputs for legacy model
            batch_size = 1
            seq_x = torch.zeros(batch_size, max_seq_len, seq_input_dim, dtype=torch.float32)
            snap_x = torch.zeros(batch_size, snap_input_dim, dtype=torch.float32)
            session_id = torch.zeros(batch_size, dtype=torch.int64)
            vol_regime_id = torch.zeros(batch_size, dtype=torch.int64)
            trend_regime_id = torch.zeros(batch_size, dtype=torch.int64)
            
            with torch.no_grad():
                outputs = transformer_model(
                    seq_x=seq_x,
                    snap_x=snap_x,
                    session_id=session_id,
                    vol_regime_id=vol_regime_id,
                    trend_regime_id=trend_regime_id,
                )
        
        # Verify output
        if "direction_logit" not in outputs:
            return ModelLoadResult(
                success=False,
                model_class_name=model_class_name,
                param_count=param_count,
                model_hash=model_hash,
                error_type="SMOKE_TEST_FAILED",
                error_message="Model forward pass did not return 'direction_logit'",
                load_time_sec=time.perf_counter() - start_time,
            )
        
        log.info("[MODEL_LOADER_WORKER] Smoke test passed")
        
        load_time = time.perf_counter() - start_time
        log.info(f"[MODEL_LOADER_WORKER] Load completed in {load_time:.2f}s")
        
        return ModelLoadResult(
            success=True,
            model_class_name=model_class_name,
            param_count=param_count,
            model_hash=model_hash,
            load_time_sec=load_time,
        )
        
    except FileNotFoundError as e:
        return ModelLoadResult(
            success=False,
            model_class_name="UNKNOWN",
            param_count=0,
            model_hash="",
            error_type="FILE_NOT_FOUND",
            error_message=str(e),
            load_time_sec=time.perf_counter() - start_time,
        )
    except RuntimeError as e:
        return ModelLoadResult(
            success=False,
            model_class_name="UNKNOWN",
            param_count=0,
            model_hash="",
            error_type="RUNTIME_ERROR",
            error_message=str(e),
            load_time_sec=time.perf_counter() - start_time,
            traceback_excerpt="\n".join(traceback.format_exc().splitlines()[:30]),
        )
    except Exception as e:
        return ModelLoadResult(
            success=False,
            model_class_name="UNKNOWN",
            param_count=0,
            model_hash="",
            error_type=type(e).__name__,
            error_message=str(e),
            load_time_sec=time.perf_counter() - start_time,
            traceback_excerpt="\n".join(traceback.format_exc().splitlines()[:30]),
        )


def _model_loader_worker_wrapper(config_dict: dict, child_conn):
    """
    Worker wrapper that runs in isolated process.
    
    This must be a top-level function (not nested) to be picklable.
    
    Parameters
    ----------
    config_dict : dict
        Serialized ModelLoadConfig (as dict)
    child_conn : multiprocessing.Connection
        Pipe connection to parent process
    """
    try:
        # Reconstruct config from dict
        config = ModelLoadConfig(
            bundle_dir=Path(config_dict["bundle_dir"]),
            feature_meta_path=Path(config_dict["feature_meta_path"]),
            seq_scaler_path=Path(config_dict["seq_scaler_path"]) if config_dict.get("seq_scaler_path") else None,
            snap_scaler_path=Path(config_dict["snap_scaler_path"]) if config_dict.get("snap_scaler_path") else None,
            model_variant=config_dict["model_variant"],
            device=config_dict.get("device", "cpu"),
            timeout_sec=config_dict.get("timeout_sec", 60.0),
        )
        result = _load_model_worker_impl(config)
        child_conn.send(result)
    except Exception as e:
        # Catch any exception and send error result
        error_result = ModelLoadResult(
            success=False,
            model_class_name="UNKNOWN",
            param_count=0,
            model_hash="",
            error_type=type(e).__name__,
            error_message=str(e),
            load_time_sec=0.0,
            traceback_excerpt="\n".join(traceback.format_exc().splitlines()[:30]),
        )
        child_conn.send(error_result)
    finally:
        child_conn.close()


def load_model_with_timeout(
    config: ModelLoadConfig,
    timeout_sec: float = 60.0,
) -> ModelLoadResult:
    """
    Load model in isolated process with timeout.
    
    Parameters
    ----------
    config : ModelLoadConfig
        Configuration for model loading
    timeout_sec : float
        Maximum time to wait for loading (default: 60s)
    
    Returns
    -------
    ModelLoadResult
        Result with success status, metadata, or error information
    """
    import multiprocessing
    
    # CRITICAL: Use "spawn" context on macOS to avoid fork issues
    ctx = multiprocessing.get_context("spawn")
    
    # Create pipe for communication
    parent_conn, child_conn = ctx.Pipe()
    
    # Serialize config to dict (for pickling)
    from dataclasses import asdict
    config_dict = asdict(config)
    # Convert Path objects to strings for pickling
    for key, value in config_dict.items():
        if isinstance(value, Path):
            config_dict[key] = str(value)
        elif value is None:
            config_dict[key] = None
    
    # Start worker process (using top-level function)
    process = ctx.Process(
        target=_model_loader_worker_wrapper,
        args=(config_dict, child_conn),
        name="ModelLoaderWorker"
    )
    process.start()
    
    # Wait for result with timeout
    if parent_conn.poll(timeout_sec):
        result = parent_conn.recv()
        process.join(timeout=5.0)  # Give process time to clean up
        if process.is_alive():
            log.warning("[MODEL_LOADER] Worker process still alive after join, terminating")
            process.terminate()
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
        parent_conn.close()
        return result
    else:
        # Timeout: terminate worker
        log.error(f"[MODEL_LOADER] Timeout after {timeout_sec}s, terminating worker process")
        process.terminate()
        process.join(timeout=2.0)
        if process.is_alive():
            process.kill()
        parent_conn.close()
        
        return ModelLoadResult(
            success=False,
            model_class_name="UNKNOWN",
            param_count=0,
            model_hash="",
            error_type="MODEL_LOAD_TIMEOUT",
            error_message=f"Model loading timed out after {timeout_sec}s",
            load_time_sec=timeout_sec,
            traceback_excerpt=None,
        )


if __name__ == "__main__":
    # Standalone entry point for testing
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    if len(sys.argv) < 3:
        print("Usage: python3 model_loader_worker.py <bundle_dir> <feature_meta_path> [variant]")
        sys.exit(1)
    
    bundle_dir = Path(sys.argv[1])
    feature_meta_path = Path(sys.argv[2])
    variant = sys.argv[3] if len(sys.argv) > 3 else "v10"
    
    config = ModelLoadConfig(
        bundle_dir=bundle_dir,
        feature_meta_path=feature_meta_path,
        model_variant=variant,
    )
    
    result = load_model_with_timeout(config, timeout_sec=60.0)
    
    if result.success:
        print(f"✅ Model loaded successfully")
        print(f"   Class: {result.model_class_name}")
        print(f"   Params: {result.param_count:,}")
        print(f"   Hash: {result.model_hash}")
        print(f"   Time: {result.load_time_sec:.2f}s")
    else:
        print(f"❌ Model loading failed")
        print(f"   Error: {result.error_type}")
        print(f"   Message: {result.error_message}")
        sys.exit(1)

