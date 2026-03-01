# ARCHIVED (NON-EXECUTABLE): kept for reference; wrapped to be syntactically parseable.
# # ARCHIVED (NON-EXECUTABLE): kept for reference; wrapped to be syntactically parseable.
# if False:
#     r"""
# #!/home/andre2/venvs/gx1/bin/python
# raise RuntimeError("LEGACY_DISABLED: Use gx1/scripts/README_TRUTH_XGB.md (TRUTH XGB lane only).")
# # -*- coding: utf-8 -*-
# """
# Smoke test for A/B trade-outcomes delta analyzer.
# 
# Accepts --run-a and --run-b, runs the analyzer with a temp out dir under
# GX1_DATA/reports/_smoke/, returns exit 0 if MD+CSV exist and matched_trades > 0.
# """
# 
# from __future__ import annotations
# 
# import argparse
# import sys
# from pathlib import Path
# 
# GX1_DATA = "/home/andre2/GX1_DATA"
# SMOKE_OUT = "/home/andre2/GX1_DATA/reports/_smoke"
# 
# 
# def main() -> int:
#     parser = argparse.ArgumentParser(description="Smoke test for ab_trade_outcomes_delta")
#     parser.add_argument("--run-a", type=str, required=True, help="Run A (baseline) root dir")
#     parser.add_argument("--run-b", type=str, required=True, help="Run B (ML-frozen) root dir")
#     args = parser.parse_args()
# 
#     run_a = Path(args.run_a).expanduser().resolve()
#     run_b = Path(args.run_b).expanduser().resolve()
#     out_dir = Path(SMOKE_OUT).resolve()
#     out_dir.mkdir(parents=True, exist_ok=True)
#     ab_run_id = "smoke_ab_trade_delta"
# 
#     from gx1.scripts.ab_trade_outcomes_delta import run_analyzer
# 
#     meta = run_analyzer(
#         run_a=run_a,
#         run_b=run_b,
#         out_dir=out_dir,
#         ab_run_id=ab_run_id,
#         max_rows=10,
#         strict=False,
#     )
# 
#     out_sub = out_dir / ab_run_id
#     md_path = out_sub / "AB_TRADE_DELTA.md"
#     csv_path = out_sub / "AB_TRADE_DELTA.csv"
# 
#     if not md_path.exists():
#         print(f"[SMOKE] FAIL: {md_path} not created", file=sys.stderr)
#         return 1
#     if not csv_path.exists():
#         print(f"[SMOKE] FAIL: {csv_path} not created", file=sys.stderr)
#         return 1
#     matched = meta.get("matched", 0)
#     if matched <= 0:
#         print(f"[SMOKE] FAIL: matched_trades={matched} (expected > 0)", file=sys.stderr)
#         return 1
# 
#     print(f"[SMOKE] OK: matched={matched} out_dir={out_sub}")
#     return 0
# 
# 
# if __name__ == "__main__":
#     sys.exit(main())
# 
#     """
