#!/usr/bin/env python3
"""
Gate 0: Smoke test av worker alene (30–60 sek)
Mål: Bevise at worker kan starte, laste CANONICAL checkpoint, og returnere p_long på en dummy batch.
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
    print("GATE 0: Worker Smoke Test")
    print("=" * 80)
    print()
    print("Mål:")
    print("  - Worker kan starte")
    print("  - Laster CANONICAL checkpoint")
    print("  - Returnerer p_long på dummy batch")
    print("  - Shutdown clean")
    print()
    
    from gx1.inference.client import ModelWorkerClient
    from gx1.inference.protocol import WorkerConfig
    
    # Checkpoint path (use CANONICAL)
    checkpoint_path = Path("models/entry_v10/entry_v10_1_transformer_CANONICAL.pt")
    if not checkpoint_path.exists():
        # Fallback to non-canonical if CANONICAL doesn't exist
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
    
    print(f"✅ Using checkpoint: {checkpoint_path}")
    print(f"✅ Using metadata: {meta_path}")
    print()
    
    # Create worker config
    config = WorkerConfig(
        checkpoint_path=str(checkpoint_path),
        meta_path=str(meta_path),
        variant="v10_1",
        device="cpu",
        timeout_seconds=30.0,
    )
    
    print("=" * 80)
    print("STEP 1: Starting ModelWorkerClient...")
    print("=" * 80)
    
    try:
        client = ModelWorkerClient(config)
        print("✅ ModelWorkerClient started")
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
        print("STEP 2: Sending 3 dummy batches...")
        print("=" * 80)
        
        # Batch parameters (V10.1: seq_len=90, seq_dim=16, snap_dim=88)
        batch_size = 2
        seq_len = 90
        seq_dim = 16
        snap_dim = 88
        
        success_count = 0
        for i in range(3):
            print(f"\nBatch {i+1}/3:")
            
            # Create dummy batch
            seq_features = np.random.randn(batch_size, seq_len, seq_dim).astype(np.float32)
            snap_features = np.random.randn(batch_size, snap_dim).astype(np.float32)
            session_ids = np.random.randint(0, 3, size=batch_size, dtype=np.int32)  # 0=EU, 1=OVERLAP, 2=US
            vol_regime_ids = np.random.randint(0, 4, size=batch_size, dtype=np.int32)  # 0-3
            trend_regime_ids = np.random.randint(0, 3, size=batch_size, dtype=np.int32)  # 0-2
            
            print(f"  Input shapes: seq={seq_features.shape}, snap={snap_features.shape}")
            
            try:
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
                
                # Verify output
                print(f"  Output shape: {p_long.shape}, dtype: {p_long.dtype}")
                print(f"  Output range: [{p_long.min():.4f}, {p_long.max():.4f}]")
                print(f"  Inference time: {elapsed*1000:.2f} ms")
                
                if p_long.shape[0] != batch_size:
                    print(f"  ❌ ERROR: Output batch size mismatch: expected {batch_size}, got {p_long.shape[0]}")
                    continue
                
                if p_long.dtype != np.float32:
                    print(f"  ❌ ERROR: Output dtype mismatch: expected float32, got {p_long.dtype}")
                    continue
                
                if not (0.0 <= p_long.min() <= p_long.max() <= 1.0):
                    print(f"  ⚠️  WARNING: Output values outside [0, 1]: [{p_long.min():.4f}, {p_long.max():.4f}]")
                
                print(f"  ✅ Batch {i+1} OK")
                success_count += 1
                
            except Exception as e:
                print(f"  ❌ ERROR: Batch {i+1} failed: {e}")
                import traceback
                traceback.print_exc()
        
        print()
        print("=" * 80)
        print(f"STEP 3: Results: {success_count}/3 batches OK")
        print("=" * 80)
        
        if success_count != 3:
            print(f"❌ FAIL: Only {success_count}/3 batches succeeded")
            client.shutdown()
            sys.exit(1)
        
        print("✅ All 3 batches succeeded")
        print()
        
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
        print("✅ GATE 0 PASSED: Worker smoke test successful")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

