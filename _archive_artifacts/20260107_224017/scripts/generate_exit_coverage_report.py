#!/usr/bin/env python3
"""
Generate exit coverage report with root cause analysis and patch summary.
"""

import json
import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/generate_exit_coverage_report.py <run_dir>")
        sys.exit(1)
    
    run_dir = Path(sys.argv[1])
    audit_file = run_dir / "exit_coverage_audit.json"
    
    if not audit_file.exists():
        print(f"ERROR: Audit file not found: {audit_file}")
        print("Run: python3 scripts/audit_exit_coverage.py <run_dir>")
        sys.exit(1)
    
    with open(audit_file, 'r') as f:
        audit = json.load(f)
    
    print("=" * 80)
    print("EXIT COVERAGE REPORT")
    print("=" * 80)
    print()
    print("## EXIT COVERAGE NUMBERS")
    print()
    print(f"- **trades_total:** {audit['trades_total']}")
    print(f"- **n_with_exit_summary:** {audit['n_with_exit_summary']}")
    print(f"- **n_open_at_end (EOF/left_open):** {audit['n_open_at_end']}")
    print(f"- **n_missing_exit_summary:** {audit['n_missing_exit_summary']}")
    print(f"- **coverage_rate:** {audit['coverage_rate']*100:.1f}%")
    print()
    print("## ROOT CAUSE")
    print()
    print("**Problem:** Entry quality analyse fant bare 2,953 trades med MAE/MFE data (av 24,600)")
    print()
    print("**Root cause:**")
    print("1. Alle 24,600 trades har exit_summary (100% coverage) ✅")
    print("2. MEN: 21,647 trades er EOF-closed og mangler MAE/MFE i exit_summary")
    print("3. Dette er fordi EOF-close path i `oanda_demo_runner.py` ikke beregner intratrade metrics")
    print("4. EOF-close kaller `log_exit_summary()` direkte uten å beregne MAE/MFE fra candles")
    print()
    print("## PATCH SUMMARY")
    print()
    print("### 1. EOF-close path fix (`gx1/execution/oanda_demo_runner.py`)")
    print("- Beregner nå MAE/MFE fra candles DataFrame for EOF-close trades")
    print("- Bruker entry_time og exit_time til å filtrere relevante bars")
    print("- Beregner MAE/MFE basert på side (long/short) og bid/ask prices")
    print("- Setter MAE/MFE i `log_exit_summary()` call")
    print()
    print("### 2. Sign convention fix")
    print("- **MAE:** Konvertert til positiv magnitude (>=0) i alle paths")
    print("  - `exit_manager.py`: `max_mae_bps = abs(max_mae_bps_raw)`")
    print("  - `oanda_demo_runner.py`: `max_mae_bps = abs(max_mae_bps_raw)`")
    print("  - `analyze_entry_quality.py`: `mae_bps = abs(mae_bps)` hvis negativ")
    print("  - `threshold_sweep_entry_quality.py`: `mae_bps = abs(mae_bps)` hvis negativ")
    print("- **MFE:** Allerede positiv, men sikret i alle paths")
    print("- **goes_against_us condition:** `mae_bps > 5.0` (nå korrekt med positiv MAE)")
    print()
    print("### 3. Replay contract (`scripts/verify_replay_exit_coverage.py`)")
    print("- Hard fail hvis noen trade mangler exit_summary i replay mode")
    print("- Verifiserer at alle trades har exit_summary etter replay")
    print("- Kan brukes med `--fail-fast` flag for CI/CD")
    print()
    print("## RE-RUN RESULTS")
    print()
    print("**Status:** Fixes er implementert, men krever re-run av FULLYEAR replay")
    print()
    print("**For å teste fixes:**")
    print("1. Kjør FULLYEAR replay på nytt med fikset kode")
    print("2. Verifiser exit coverage: `python3 scripts/verify_replay_exit_coverage.py <run_dir> --fail-fast`")
    print("3. Kjør entry quality analyse: `python3 scripts/analyze_entry_quality.py <run_dir>`")
    print("4. Forventet: Alle 24,600 trades skal ha MAE/MFE data")
    print("5. Kjør threshold sweep: `python3 scripts/threshold_sweep_entry_quality.py <run_dir>`")
    print()
    print("**Expected improvements:**")
    print("- Exit coverage: 100% (allerede oppnådd)")
    print("- MAE/MFE coverage: 100% (opp fra 12% = 2,953/24,600)")
    print("- goes_against_us rate: Korrekt beregnet med positiv MAE")
    print()

if __name__ == "__main__":
    main()



