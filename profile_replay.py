#!/usr/bin/env python3
"""
Profile FULLYEAR 2025 replay using cProfile.
"""
import cProfile
import pstats
import sys
from pathlib import Path
import os

# Set environment variables before importing
os.environ["GX1_DATA"] = "/home/andre2/GX1_DATA"
os.environ["GX1_OUTPUT_MODE"] = "TRUTH"
os.environ["GX1_RUN_MODE"] = "TRUTH"
os.environ["GX1_TRUTH_MODE"] = "1"
os.environ["GX1_CANONICAL_TRUTH_FILE"] = "/home/andre2/src/GX1_ENGINE/gx1/configs/canonical_truth_signal_only.json"
os.environ["GX1_CANONICAL_POLICY_PATH"] = "/home/andre2/src/GX1_ENGINE/gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml"
os.environ["GX1_REPLAY_USE_PREBUILT_FEATURES"] = "1"
os.environ["GX1_FEATURE_BUILD_DISABLED"] = "1"

# Import after setting env vars
from gx1.scripts.run_fullyear_2025_truth_proof import main

if __name__ == "__main__":
    # Profile the main function
    profiler = cProfile.Profile()
    
    # Get run_id from command line or generate
    run_id = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Run with profiling
    profiler.enable()
    try:
        main()
    finally:
        profiler.disable()
        
        # Find the run_id from the latest run directory
        if not run_id:
            reports_dir = Path("/home/andre2/GX1_DATA/reports/fullyear_truth_proof")
            if reports_dir.exists():
                run_dirs = sorted([d for d in reports_dir.glob("FULLYEAR_2025_PROOF_*") if d.is_dir()], reverse=True)
                if run_dirs:
                    run_id = run_dirs[0].name
        
        if run_id:
            # Save profile
            profile_dir = Path(f"/home/andre2/GX1_DATA/reports/profiles/{run_id}")
            profile_dir.mkdir(parents=True, exist_ok=True)
            profile_path = profile_dir / "replay.prof"
            
            profiler.dump_stats(str(profile_path))
            print(f"\n[PROFILE] Saved profile to: {profile_path}")
            
            # Generate stats
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            
            # Save stats to file
            stats_file = profile_dir / "replay_stats.txt"
            with open(stats_file, 'w') as f:
                stats.print_stats(40, file=f)
            
            print(f"[PROFILE] Saved stats to: {stats_file}")
        else:
            print("[PROFILE] Could not determine run_id, saving to /tmp/replay.prof")
            profiler.dump_stats("/tmp/replay.prof")
