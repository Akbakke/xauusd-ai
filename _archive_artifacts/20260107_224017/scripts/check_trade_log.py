#!/usr/bin/env python3
"""
Quick trade log sanity checker.

Usage:
    PYTHONPATH=. python scripts/check_trade_log.py gx1/wf_runs/.../trade_log.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Check GX1 trade log invariants.")
    parser.add_argument("csv", type=Path, help="Path to trade_log.csv")
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"Trade log not found: {args.csv}")

    df = pd.read_csv(args.csv)
    total = len(df)

    def _extract_exit_profile(df: pd.DataFrame) -> pd.Series | None:
        """Return a Series with exit_profile values, if available."""
        # Prefer explicit column if present
        for col in ("exit_profile", "extra.exit_profile"):
            if col in df.columns:
                return df[col]

        if "extra" not in df.columns:
            return None

        def _from_extra(value):
            if isinstance(value, dict):
                return value.get("exit_profile")
            if isinstance(value, str) and value.strip():
                try:
                    data = json.loads(value)
                except json.JSONDecodeError:
                    return None
                if isinstance(data, dict):
                    return data.get("exit_profile")
            return None

        return df["extra"].apply(_from_extra)

    exit_profile_series = _extract_exit_profile(df)
    if exit_profile_series is None:
        missing_exit_profile = total
    else:
        exit_profile_series = exit_profile_series.astype("string")
        missing_exit_profile = exit_profile_series.isna().sum() + (exit_profile_series == "").sum()

    exit_time_col = df["exit_time"] if "exit_time" in df.columns else pd.Series([None] * total)
    closed_mask = exit_time_col.notna() & (exit_time_col.astype(str) != "")
    closed = int(closed_mask.sum())
    pct_closed = (closed / total * 100.0) if total else 0.0

    print(f"Total trades: {total}")
    print(f"Closed trades: {closed} ({pct_closed:.2f}%)")
    print(f"Missing exit_profile: {missing_exit_profile}")

    if closed:
        closed_df = df.loc[closed_mask].copy()
        closed_df["pnl_bps"] = pd.to_numeric(closed_df["pnl_bps"], errors="coerce")
        pnl_series = closed_df["pnl_bps"].dropna()
        if not pnl_series.empty:
            avg_pnl = pnl_series.mean()
            median_pnl = pnl_series.median()
            min_pnl = pnl_series.min()
            max_pnl = pnl_series.max()
            win_rate = (pnl_series > 0).mean() * 100.0
            print("Closed PnL summary (bps):")
            print(f"  Avg:   {avg_pnl:.2f}")
            print(f"  Med:   {median_pnl:.2f}")
            print(f"  Min:   {min_pnl:.2f}")
            print(f"  Max:   {max_pnl:.2f}")
            print(f"  Win%:  {win_rate:.2f}%")
        else:
            print("Closed PnL summary (bps): no numeric values available.")
    else:
        print("No closed trades to summarize.")


if __name__ == "__main__":
    main()
