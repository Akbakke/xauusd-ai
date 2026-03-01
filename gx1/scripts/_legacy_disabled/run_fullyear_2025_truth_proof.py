# ARCHIVED (NON-EXECUTABLE): kept for reference; wrapped to be syntactically parseable.
# # ARCHIVED (NON-EXECUTABLE): kept for reference; wrapped to be syntactically parseable.
# if False:
#     r"""
# #!/home/andre2/venvs/gx1/bin/python
# raise RuntimeError("LEGACY_DISABLED: Use gx1/scripts/README_TRUTH_XGB.md (TRUTH XGB lane only).")
# # -*- coding: utf-8 -*-
# """
# Thin wrapper: runs TRUTH full-year via run_truth_e2e_sanity (no legacy replay import).
# 
# This script does NOT import legacy replay or replay_eval_gated_parallel.
# It only invokes: python -m gx1.scripts.run_truth_e2e_sanity --full-year
# 
# For the full proof-pack runner (heartbeat, baseline, etc.), see:
#   gx1/_quarantine/legacy_wrappers_20260219/run_fullyear_2025_truth_proof.py
# """
# 
# from __future__ import annotations
# 
# import os
# import subprocess
# import sys
# from pathlib import Path
# 
# # Engine root for canonical venv
# _ENGINE = Path(__file__).resolve().parents[2]
# _CANONICAL_VENV = os.environ.get("GX1_CANONICAL_VENV", "/home/andre2/venvs/gx1/bin/python")
# 
# 
# def main() -> int:
#     print("[run_fullyear_2025_truth_proof] Thin wrapper → gx1.scripts.run_truth_e2e_sanity --full-year", file=sys.stderr)
#     env = os.environ.copy()
#     env.setdefault("GX1_ENGINE", str(_ENGINE))
#     env.setdefault("PYTHONPATH", str(_ENGINE))
#     result = subprocess.run(
#         [_CANONICAL_VENV, "-m", "gx1.scripts.run_truth_e2e_sanity", "--full-year"],
#         cwd=str(_ENGINE),
#         env=env,
#     )
#     return result.returncode
# 
# 
# if __name__ == "__main__":
#     sys.exit(main())
# 
#     """
