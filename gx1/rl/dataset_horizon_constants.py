"""Unified K-horizon constants for GX1 reward/dataset builds.

Centralizes scattered K_HORIZONS definitions to prevent trainer↔dataset mismatch
(2026-05-24 Bug 4: trainer expected [1,4,12,24,48,96], V12 dataset had [12,60,240,480,1440]
which caused 5/6 K-slots to fall back to exit_now → degenerate Q-values).
"""

# Entry-IQL: K-horizons in M5 bars (for forward outcome compute).
K_HORIZONS_ENTRY_M5 = [12, 24, 48, 96, 144, 192]

# Exit-IQL: K-horizons in M1 bars (per-bar dataset cadence).
# Scalp-tuned: 1min..4h. Trainer reads hold_max_pnl_K{K}_v1 at these horizons.
K_HORIZONS_EXIT_M1_SCALP = [1, 4, 12, 48, 144, 240]

# Counterfactual replay (24h-window training).
K_HORIZONS_CF_M1 = [12, 60, 240, 480, 1440]

# Legacy V12 K-set (M5 bars). Kept for backwards compat in materialize_build_exit_iql_v2.py.
K_HORIZONS_EXIT_M5_LEGACY = [1, 4, 12, 24, 48, 96]
