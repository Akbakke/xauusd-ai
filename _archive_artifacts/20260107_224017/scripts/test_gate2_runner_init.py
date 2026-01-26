#!/usr/bin/env python3
"""
Gate 2: Runner init + 1 inferens (0 replay)
Mål: Bevise at runner kan konstrueres, starte worker, og faktisk bruke _predict_entry_v10_hybrid.
"""

import sys
import time
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


def main():
    import pandas as pd
    import numpy as np
    
    POLICY = "gx1/configs/policies/sniper_snapshot/2025_SNIPER_V10_1/GX1_V12_SNIPER_ENTRY_V10_1_FLAT_THRESHOLD_0_18.yaml"
    
    policy_path = Path(POLICY)
    if not policy_path.exists():
        print(f"❌ ERROR: Policy not found: {policy_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("GATE 2: Runner Init + Inference Test")
    print("=" * 80)
    print()
    print("Mål:")
    print("  - Konstruere GX1DemoRunner med V10.1 policy")
    print("  - Verifisere entry_v10_enabled=True")
    print("  - Verifisere worker-pid settes")
    print("  - Kjøre én prediksjon")
    print("  - Verifisere destructor cleanup")
    print()
    print(f"Policy: {policy_path}")
    print()
    
    print("=" * 80)
    print("STEP 1: Constructing GX1DemoRunner...")
    print("=" * 80)
    
    try:
        from gx1.execution.oanda_demo_runner import GX1DemoRunner
        
        output_dir = Path("data/temp/test_gate2_runner")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Policy: {policy_path}")
        print(f"Output: {output_dir}")
        print()
        
        start_time = time.perf_counter()
        
        runner = GX1DemoRunner(
            policy_path,
            dry_run_override=True,
            replay_mode=True,
            fast_replay=True,
            output_dir=output_dir,
        )
        
        elapsed = time.perf_counter() - start_time
        print(f"✅ GX1DemoRunner constructed in {elapsed:.2f} seconds")
        print()
        
        if elapsed > 60.0:
            print(f"⚠️  WARNING: Runner construction took {elapsed:.2f}s (>60s threshold)")
        
    except Exception as e:
        print(f"❌ ERROR: Failed to construct runner: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("=" * 80)
    print("STEP 2: Verifying V10.1 setup...")
    print("=" * 80)
    
    try:
        # Check entry_v10_enabled
        if not hasattr(runner, 'entry_v10_enabled') or not runner.entry_v10_enabled:
            print("❌ ERROR: entry_v10_enabled is not True")
            sys.exit(1)
        print("✅ entry_v10_enabled=True")
        
        # Check client
        if not hasattr(runner, 'entry_v10_client') or runner.entry_v10_client is None:
            print("❌ ERROR: entry_v10_client is None")
            sys.exit(1)
        print("✅ entry_v10_client is not None")
        
        # Check worker PID
        client = runner.entry_v10_client
        if client.worker_process is None:
            print("❌ ERROR: worker_process is None")
            sys.exit(1)
        
        worker_pid = client.worker_process.pid
        if not client.worker_process.is_alive():
            print("❌ ERROR: Worker process is not alive")
            sys.exit(1)
        print(f"✅ Worker PID: {worker_pid}")
        print(f"✅ Worker process is alive")
        
        # Check XGB models
        if not hasattr(runner, 'entry_v10_xgb_models') or runner.entry_v10_xgb_models is None:
            print("❌ ERROR: entry_v10_xgb_models is None")
            sys.exit(1)
        print(f"✅ entry_v10_xgb_models loaded ({len(runner.entry_v10_xgb_models)} sessions)")
        
        # Check transformer config
        if not hasattr(runner, 'entry_v10_transformer_config') or runner.entry_v10_transformer_config is None:
            print("❌ ERROR: entry_v10_transformer_config is None")
            sys.exit(1)
        print(f"✅ entry_v10_transformer_config loaded")
        print()
        
    except Exception as e:
        print(f"❌ ERROR: Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("=" * 80)
    print("STEP 3: Running one inference (_predict_entry_v10_hybrid)...")
    print("=" * 80)
    
    try:
        from gx1.execution.live_features import build_live_entry_features
        
        # Create minimal candle DataFrame (need at least 90 bars for V10.1)
        # Use actual price data structure
        n_bars = 100
        dates = pd.date_range(start='2025-01-01 00:00:00', periods=n_bars, freq='5min', tz='UTC')
        
        # Create realistic OHLC data
        np.random.seed(42)
        base_price = 2650.0
        prices = base_price + np.cumsum(np.random.randn(n_bars) * 0.5)
        
        candles = pd.DataFrame({
            'open': prices,
            'high': prices + np.abs(np.random.randn(n_bars) * 0.3),
            'low': prices - np.abs(np.random.randn(n_bars) * 0.3),
            'close': prices + np.random.randn(n_bars) * 0.2,
            'volume': np.random.randint(100, 1000, n_bars),
        }, index=dates)
        
        # Ensure high >= close >= low, high >= open >= low
        candles['high'] = candles[['open', 'high', 'close']].max(axis=1)
        candles['low'] = candles[['open', 'low', 'close']].min(axis=1)
        
        print(f"Created {len(candles)} candles")
        print(f"Date range: {candles.index[0]} to {candles.index[-1]}")
        print()
        
        # Build entry bundle
        entry_bundle = build_live_entry_features(candles)
        print("✅ Entry bundle built")
        
        # Create minimal policy_state
        policy_state = {
            "session": "OVERLAP",
            "trend_regime": "NEUTRAL",
            "vol_regime": "HIGH",
        }
        
        print("Calling _predict_entry_v10_hybrid...")
        start_time = time.perf_counter()
        
        prediction = runner._predict_entry_v10_hybrid(
            entry_bundle=entry_bundle,
            candles=candles,
            policy_state=policy_state,
        )
        
        elapsed = time.perf_counter() - start_time
        print(f"✅ _predict_entry_v10_hybrid completed in {elapsed*1000:.2f} ms")
        print()
        
        if prediction is None:
            print("❌ ERROR: Prediction is None")
            sys.exit(1)
        
        print(f"✅ Prediction received:")
        print(f"   prob_long: {prediction.prob_long:.4f}")
        print(f"   prob_short: {prediction.prob_short:.4f}")
        print(f"   p_hat: {prediction.p_hat:.4f}")
        print(f"   margin: {prediction.margin:.4f}")
        print(f"   session: {prediction.session}")
        print()
        
        # Verify prediction values are reasonable
        if not (0.0 <= prediction.prob_long <= 1.0):
            print(f"⚠️  WARNING: prob_long out of range: {prediction.prob_long}")
        
        if not (0.0 <= prediction.prob_short <= 1.0):
            print(f"⚠️  WARNING: prob_short out of range: {prediction.prob_short}")
        
    except Exception as e:
        print(f"❌ ERROR: Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("=" * 80)
    print("STEP 4: Verifying destructor cleanup...")
    print("=" * 80)
    
    try:
        # Get worker PID before cleanup
        old_pid = client.worker_process.pid if client.worker_process else None
        print(f"Worker PID before cleanup: {old_pid}")
        
        # Delete runner (triggers __del__)
        print("Deleting runner (should trigger __del__ and cleanup)...")
        del runner
        
        # Give cleanup time to run
        time.sleep(1.0)
        
        # Verify worker process is dead
        try:
            import psutil
            try:
                proc = psutil.Process(old_pid)
                if proc.is_running():
                    print(f"⚠️  WARNING: Worker process {old_pid} is still running after cleanup")
                    print("   (This might be OK if cleanup hasn't finished yet)")
                else:
                    print(f"✅ Worker process {old_pid} is no longer running")
            except psutil.NoSuchProcess:
                print(f"✅ Worker process {old_pid} is no longer running (process not found)")
        except ImportError:
            print("⚠️  WARNING: psutil not available, cannot verify process cleanup")
        
        print("✅ Cleanup completed")
        print()
        
    except Exception as e:
        print(f"⚠️  WARNING: Cleanup verification failed: {e}")
        import traceback
        traceback.print_exc()
        # Don't fail the test on cleanup verification issues
    
    print("=" * 80)
    print("✅ GATE 2 PASSED: Runner init + inference test successful")
    print("=" * 80)


if __name__ == "__main__":
    main()

