#!/usr/bin/env python3
"""
Diagnose Hard Eligibility: Analyze vol_regime values and sources.

Usage:
    python3 scripts/diagnose_hard_eligibility.py <run_dir>
"""

import json
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, Any

def diagnose_hard_eligibility(run_dir: Path):
    """Diagnose hard eligibility blocking from replay run."""
    
    summary_json = run_dir / "REPLAY_PERF_SUMMARY.json"
    if not summary_json.exists():
        print(f"❌ Summary JSON not found: {summary_json}")
        return
    
    with open(summary_json) as f:
        data = json.load(f)
    
    entry_counters = data.get("entry_counters", {})
    
    print("=" * 80)
    print("HARD ELIGIBILITY DIAGNOSE")
    print("=" * 80)
    print()
    
    # 1. Breakdown of veto reasons
    print("1. HARD VETO BREAKDOWN:")
    print("-" * 80)
    n_cycles = entry_counters.get("n_cycles", 0)
    n_eligible = entry_counters.get("n_eligible_cycles", 0)
    
    veto_hard_warmup = entry_counters.get("veto_hard_warmup", 0)
    veto_hard_session = entry_counters.get("veto_hard_session", 0)
    veto_hard_vol_regime = entry_counters.get("veto_hard_vol_regime", 0)
    veto_hard_spread = entry_counters.get("veto_hard_spread", 0)
    veto_hard_killswitch = entry_counters.get("veto_hard_killswitch", 0)
    
    print(f"  n_cycles: {n_cycles}")
    print(f"  n_eligible_cycles: {n_eligible} ({n_eligible/n_cycles*100:.1f}%)")
    print(f"  veto_hard_warmup: {veto_hard_warmup} ({veto_hard_warmup/n_cycles*100:.1f}%)")
    print(f"  veto_hard_session: {veto_hard_session} ({veto_hard_session/n_cycles*100:.1f}%)")
    print(f"  veto_hard_vol_regime: {veto_hard_vol_regime} ({veto_hard_vol_regime/n_cycles*100:.1f}%)")
    print(f"  veto_hard_spread: {veto_hard_spread} ({veto_hard_spread/n_cycles*100:.1f}%)")
    print(f"  veto_hard_killswitch: {veto_hard_killswitch} ({veto_hard_killswitch/n_cycles*100:.1f}%)")
    print()
    
    # 2. Code analysis
    print("2. CODE ANALYSIS:")
    print("-" * 80)
    print("  Location: gx1/execution/entry_manager.py")
    print("  Function: _check_hard_eligibility()")
    print("  Line: 276-278")
    print()
    print("  vol_regime source in evaluate_entry():")
    print("    - Line 668-685: Tries to get from Big Brain V1")
    print("    - Fallback: 'UNKNOWN' if Big Brain not available")
    print("    - Problem: Big Brain V1 not available in replay mode")
    print()
    print("  vol_regime check in _check_hard_eligibility():")
    print("    - Line 276: vol_regime = policy_state.get('brain_vol_regime', 'UNKNOWN')")
    print("    - Line 277: if vol_regime == 'EXTREME' or vol_regime == 'UNKNOWN':")
    print("    - Line 278: return False, HARD_ELIGIBILITY_VOL_REGIME_EXTREME")
    print()
    print("  ⚠️  REKKEFØLGEFEIL IDENTIFISERT:")
    print("    - vol_regime hentes fra Big Brain V1 (ikke tilgjengelig i replay)")
    print("    - Fallback til 'UNKNOWN' → hard block")
    print("    - Dette skjer FØR feature build, så vi kan ikke beregne vol_regime")
    print("    - Men vi blokkerer på et signal vi ikke kan beregne uten feature build!")
    print()
    
    # 3. Check if Big Brain is available
    print("3. BIG BRAIN V1 STATUS:")
    print("-" * 80)
    log_file = run_dir / "replay.log"
    if log_file.exists():
        with open(log_file) as f:
            log_content = f.read()
            if "Big Brain V1 modules not available" in log_content:
                print("  ❌ Big Brain V1 NOT AVAILABLE in replay")
                print("  → vol_regime will always be 'UNKNOWN'")
            elif "Big Brain V1" in log_content:
                print("  ✅ Big Brain V1 available")
            else:
                print("  ⚠️  Big Brain V1 status unknown")
    else:
        print("  ⚠️  Log file not found")
    print()
    
    # 4. Recommendation
    print("4. RECOMMENDATION:")
    print("-" * 80)
    print("  Option 1 (PREFERRED): Split eligibility in 2 levels")
    print("    - Hard Eligibility (before feature build):")
    print("      * warmup")
    print("      * session not allowed")
    print("      * spread hard cap (if available from quotes)")
    print("      * kill-switch")
    print("      * ❌ REMOVE vol_regime check (requires feature build)")
    print()
    print("    - Soft Eligibility (after minimal cheap computation):")
    print("      * Implement ultra-cheap ATR proxy from raw candles")
    print("      * Compute ATR-like metric (14-20 bars, simple numpy)")
    print("      * Tag EXTREME and block early")
    print("      * Cost: << full feature build")
    print()
    print("  Option 2 (TEMPORARY): Allow UNKNOWN with guardrails")
    print("    - UNKNOWN → eligible=True, but increment counter")
    print("    - EXTREME → still hard block")
    print("    - ⚠️  Only if UNKNOWN doesn't mean 'regime classifier broken'")
    print()
    
    print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/diagnose_hard_eligibility.py <run_dir>")
        sys.exit(1)
    
    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"❌ Run directory not found: {run_dir}")
        sys.exit(1)
    
    diagnose_hard_eligibility(run_dir)



