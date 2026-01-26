#!/usr/bin/env python3
"""
ModelWorkerClient - Client interface for ModelWorker process.

This client:
1. Spawns ModelWorker process
2. Provides predict_batch() API for runner
3. Handles timeout and restart logic
4. Manages Queue communication
"""

import logging
import multiprocessing as mp
import time
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

from gx1.inference.protocol import InferenceRequest, InferenceResponse, WorkerConfig

log = logging.getLogger(__name__)


class ModelWorkerClient:
    """
    Client for communicating with ModelWorker process.
    
    Usage:
        client = ModelWorkerClient(config)
        p_long = client.predict_batch(seq_features, snap_features, session_ids, ...)
        client.shutdown()
    """
    
    def __init__(self, config: WorkerConfig) -> None:
        """
        Initialize client and spawn worker process.
        
        Parameters
        ----------
        config : WorkerConfig
            Worker configuration
        """
        self.config = config
        self.request_queue: Optional[mp.Queue] = None
        self.response_queue: Optional[mp.Queue] = None
        self.worker_process: Optional[mp.Process] = None
        self.request_id_counter = 0
        self._shutdown = False
        
        # Spawn worker process
        self._start_worker()
    
    def _start_worker(self) -> None:
        """Spawn ModelWorker process."""
        log.info("[MODEL_CLIENT] Spawning ModelWorker process...")
        
        # Create queues
        self.request_queue = mp.Queue()
        self.response_queue = mp.Queue()
        
        # Import worker_main (must be importable for spawn)
        from gx1.inference.model_worker import worker_main
        
        # Spawn worker process
        self.worker_process = mp.Process(
            target=worker_main,
            args=(self.config, self.request_queue, self.response_queue),
            name="ModelWorker",
        )
        self.worker_process.start()
        
        log.info("[MODEL_CLIENT] Worker process spawned (PID=%d)", self.worker_process.pid)
        
        # Wait a moment to ensure worker has started
        time.sleep(0.5)
        
        # Verify worker is alive
        if not self.worker_process.is_alive():
            raise RuntimeError("[MODEL_CLIENT] Worker process died immediately after spawn")
    
    def predict_batch(
        self,
        seq_features: np.ndarray,  # [batch_size, seq_len, seq_dim]
        snap_features: np.ndarray,  # [batch_size, snap_dim]
        session_ids: np.ndarray,  # [batch_size] int32 (0=EU, 1=OVERLAP, 2=US)
        vol_regime_ids: np.ndarray,  # [batch_size] int32 (0=LOW, 1=MEDIUM, 2=HIGH, 3=EXTREME)
        trend_regime_ids: np.ndarray,  # [batch_size] int32 (0=UP, 1=DOWN, 2=NEUTRAL)
        timeout_seconds: Optional[float] = None,
    ) -> np.ndarray:
        """
        Run batch inference.
        
        Parameters
        ----------
        seq_features : np.ndarray
            Sequence features [batch_size, seq_len, seq_dim]
        snap_features : np.ndarray
            Snapshot features [batch_size, snap_dim]
        session_ids : np.ndarray
            Session IDs [batch_size] (0=EU, 1=OVERLAP, 2=US)
        vol_regime_ids : np.ndarray
            Volatility regime IDs [batch_size] (0=LOW, 1=MEDIUM, 2=HIGH, 3=EXTREME)
        trend_regime_ids : np.ndarray
            Trend regime IDs [batch_size] (0=UP, 1=DOWN, 2=NEUTRAL)
        timeout_seconds : float, optional
            Timeout for inference (default: config.timeout_seconds)
        
        Returns
        -------
        np.ndarray
            p_long probabilities [batch_size] float32
        
        Raises
        ------
        RuntimeError
            If inference fails or times out
        """
        if self._shutdown:
            raise RuntimeError("[MODEL_CLIENT] Client is shutdown")
        
        if self.worker_process is None or not self.worker_process.is_alive():
            log.warning("[MODEL_CLIENT] Worker process is dead, restarting...")
            self._start_worker()
        
        timeout = timeout_seconds if timeout_seconds is not None else self.config.timeout_seconds
        
        # Ensure numpy arrays are correct types
        seq_features = np.asarray(seq_features, dtype=np.float32)
        snap_features = np.asarray(snap_features, dtype=np.float32)
        session_ids = np.asarray(session_ids, dtype=np.int32)
        vol_regime_ids = np.asarray(vol_regime_ids, dtype=np.int32)
        trend_regime_ids = np.asarray(trend_regime_ids, dtype=np.int32)
        
        # Validate shapes
        batch_size = len(seq_features)
        if len(snap_features) != batch_size:
            raise ValueError(f"[MODEL_CLIENT] Batch size mismatch: seq={batch_size}, snap={len(snap_features)}")
        if len(session_ids) != batch_size:
            raise ValueError(f"[MODEL_CLIENT] Batch size mismatch: seq={batch_size}, session_ids={len(session_ids)}")
        if len(vol_regime_ids) != batch_size:
            raise ValueError(f"[MODEL_CLIENT] Batch size mismatch: seq={batch_size}, vol_regime_ids={len(vol_regime_ids)}")
        if len(trend_regime_ids) != batch_size:
            raise ValueError(f"[MODEL_CLIENT] Batch size mismatch: seq={batch_size}, trend_regime_ids={len(trend_regime_ids)}")
        
        # Create request
        self.request_id_counter += 1
        request = InferenceRequest(
            seq_features=seq_features,
            snap_features=snap_features,
            session_ids=session_ids,
            vol_regime_ids=vol_regime_ids,
            trend_regime_ids=trend_regime_ids,
            request_id=self.request_id_counter,
        )
        
        # Send request
        try:
            self.request_queue.put(request, timeout=5.0)
        except Exception as e:
            raise RuntimeError(f"[MODEL_CLIENT] Failed to send request: {e}")
        
        # Wait for response (with timeout)
        start_time = time.perf_counter()
        try:
            response: InferenceResponse = self.response_queue.get(timeout=timeout)
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            log.error("[MODEL_CLIENT] Inference timeout after %.2f seconds", elapsed)
            # Restart worker on timeout
            self._restart_worker()
            raise RuntimeError(f"[MODEL_CLIENT] Inference timeout after {elapsed:.2f}s: {e}")
        
        # Check for errors
        if response.error is not None:
            log.error("[MODEL_CLIENT] Worker returned error: %s", response.error)
            # Restart worker on error
            self._restart_worker()
            raise RuntimeError(f"[MODEL_CLIENT] Worker error: {response.error}")
        
        # Validate response
        if len(response.p_long) != batch_size:
            raise RuntimeError(f"[MODEL_CLIENT] Response batch size mismatch: expected {batch_size}, got {len(response.p_long)}")
        
        return response.p_long
    
    def _restart_worker(self) -> None:
        """Restart worker process after error/timeout."""
        log.warning("[MODEL_CLIENT] Restarting worker process...")
        self.shutdown()
        time.sleep(0.5)  # Brief pause before restart
        self._start_worker()
    
    def shutdown(self) -> None:
        """Shutdown worker process cleanly."""
        if self._shutdown:
            return
        
        self._shutdown = True
        log.info("[MODEL_CLIENT] Shutting down worker process...")
        
        if self.request_queue is not None:
            # Send shutdown signal
            try:
                self.request_queue.put(None, timeout=2.0)
            except Exception:
                pass
        
        if self.worker_process is not None:
            # Wait for worker to exit
            self.worker_process.join(timeout=5.0)
            if self.worker_process.is_alive():
                log.warning("[MODEL_CLIENT] Worker process did not exit cleanly, terminating...")
                self.worker_process.terminate()
                self.worker_process.join(timeout=2.0)
                if self.worker_process.is_alive():
                    log.error("[MODEL_CLIENT] Worker process did not terminate, killing...")
                    self.worker_process.kill()
                    self.worker_process.join()
        
        log.info("[MODEL_CLIENT] Worker process shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()

