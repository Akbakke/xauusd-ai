#!/usr/bin/env python3
"""
Gate 1: "Poisoned import" test (1 min)
Mål: Bevise at "runner-importkjeden" ikke kan påvirke torch lenger.
"""

import sys
import time
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


def main():
    import numpy as np
    
    print("=" * 80)
    print("GATE 1: Poisoned Import Test")
    print("=" * 80)
    print()
    print("Mål:")
    print("  - Importere runner (uten å skape den)")
    print("  - Starte worker client")
    print("  - Kjøre predict-batch")
    print("  - Shutdown")
    print()
    print("Dette er den viktigste testen for å bevise at arkitektur A faktisk løser root cause.")
    print()
    
    print("=" * 80)
    print("STEP 1: Importing runner module (poisoned import)...")
    print("=" * 80)
    
    try:
        # Import runner module (this used to "poison" torch)
        import gx1.execution.oanda_demo_runner
        print("✅ Runner module imported")
        
        # Also import other heavy modules that might affect torch
        from gx1.execution import entry_manager
        from gx1.execution import exit_manager
        print("✅ Entry/exit managers imported")
        
        print("✅ All runner-related imports completed")
        print()
        
    except Exception as e:
        print(f"❌ ERROR: Failed to import runner: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("=" * 80)
    print("STEP 2: Starting ModelWorkerClient (after poisoned import)...")
    print("=" * 80)
    
    try:
        from gx1.inference.client import ModelWorkerClient
        from gx1.inference.protocol import WorkerConfig
        
        # Checkpoint path (use CANONICAL)
        checkpoint_path = Path("models/entry_v10/entry_v10_1_transformer_CANONICAL.pt")
        if not checkpoint_path.exists():
            checkpoint_path = Path("models/entry_v10/entry_v10_1_transformer.pt")
            if not checkpoint_path.exists():
                print(f"❌ ERROR: Checkpoint not found: {checkpoint_path}")
                sys.exit(1)
        
        # Metadata path
        meta_path = checkpoint_path.parent / "entry_v10_1_transformer_meta.json"
        if not meta_path.exists():
            meta_path = checkpoint_path.parent / "entry_v10_transformer_meta.json"
        if not meta_path.exists():
            print(f"❌ ERROR: Metadata not found: {meta_path}")
            sys.exit(1)
        
        # Create worker config
        config = WorkerConfig(
            checkpoint_path=str(checkpoint_path),
            meta_path=str(meta_path),
            variant="v10_1",
            device="cpu",
            timeout_seconds=30.0,
        )
        
        print("Creating ModelWorkerClient (this should NOT hang even after runner import)...")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Metadata: {meta_path}")
        start_time = time.perf_counter()
        
        client = ModelWorkerClient(config)
        
        elapsed = time.perf_counter() - start_time
        print(f"✅ ModelWorkerClient created in {elapsed:.2f} seconds")
        
        if elapsed > 30.0:
            print(f"⚠️  WARNING: Worker creation took {elapsed:.2f}s (>30s threshold)")
        
        print(f"✅ Worker PID: {client.worker_process.pid if client.worker_process else 'N/A'}")
        print()
        
        # Give worker a moment to fully initialize
        time.sleep(1.0)
        
        # Verify worker is alive
        if client.worker_process is None or not client.worker_process.is_alive():
            print("❌ ERROR: Worker process died immediately")
            sys.exit(1)
        print("✅ Worker process is alive")
        print()
        
        print("=" * 80)
        print("STEP 3: Running predict_batch (after poisoned import)...")
        print("=" * 80)
        
        # Create dummy batch
        batch_size = 1
        seq_len = 90
        seq_dim = 16
        snap_dim = 88
        
        seq_features = np.random.randn(batch_size, seq_len, seq_dim).astype(np.float32)
        snap_features = np.random.randn(batch_size, snap_dim).astype(np.float32)
        session_ids = np.array([1], dtype=np.int32)  # OVERLAP
        vol_regime_ids = np.array([2], dtype=np.int32)  # HIGH
        trend_regime_ids = np.array([2], dtype=np.int32)  # NEUTRAL
        
        print("Sending predict_batch request...")
        start_time = time.perf_counter()
        
        p_long = client.predict_batch(
            seq_features=seq_features,
            snap_features=snap_features,
            session_ids=session_ids,
            vol_regime_ids=vol_regime_ids,
            trend_regime_ids=trend_regime_ids,
            timeout_seconds=30.0,
        )
        
        elapsed = time.perf_counter() - start_time
        print(f"✅ predict_batch completed in {elapsed*1000:.2f} ms")
        print(f"✅ Output: shape={p_long.shape}, dtype={p_long.dtype}, value={p_long[0]:.4f}")
        print()
        
        if p_long.shape[0] != batch_size:
            print(f"❌ ERROR: Output batch size mismatch: expected {batch_size}, got {p_long.shape[0]}")
            client.shutdown()
            sys.exit(1)
        
        print("=" * 80)
        print("STEP 4: Shutting down worker...")
        print("=" * 80)
        
        client.shutdown()
        
        # Verify worker is dead
        if client.worker_process is not None and client.worker_process.is_alive():
            print("❌ ERROR: Worker process still alive after shutdown")
            sys.exit(1)
        
        print("✅ Worker shutdown clean")
        print()
        
        print("=" * 80)
        print("✅ GATE 1 PASSED: Poisoned import test successful")
        print("=" * 80)
        print()
        print("This proves that architecture A successfully isolates torch from runner imports!")
        
    except Exception as e:
        print(f"❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

