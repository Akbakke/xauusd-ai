#!/usr/bin/env python3
"""CFTC Commitment of Traders data ingestion — XAU (gold futures).

Pulls weekly net positioning of speculators vs commercials for COMEX Gold
(contract 088691). Used as input feature to Entry-IQL after retrain (improvement
#17 from 2026-05-21 brainstorm).

Output: /home/andre2/GX1_DATA/data/external/cftc/cot_xau.parquet
Schema: week_ending_utc, large_spec_net, comm_net, total_oi, spec_pct_of_oi.

Run weekly (Tuesday 21:00 UTC after CFTC publishes Friday data):
    PYTHONPATH=/home/andre2/src/GX1_ENGINE python3 -u \\
        gx1/scripts/ingest_cftc_cot_xau.py

Sources:
  - https://www.cftc.gov/dea/futures/financial_lf.htm (Financial Futures)
  - Or via socrata API: https://publicreporting.cftc.gov/resource/6dca-aqww.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from datetime import datetime, timezone

import requests
import pandas as pd

OUT_DIR = Path("/home/andre2/GX1_DATA/data/external/cftc")
OUT_PATH = OUT_DIR / "cot_xau.parquet"
# CFTC contract code for COMEX Gold = "088691"
GOLD_CONTRACT_CODE = "088691"
SOCRATA_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"


def fetch_cot_xau(years_back: int = 5) -> pd.DataFrame:
    """Fetch CFTC COT data for COMEX Gold via Socrata API."""
    print(f"[CFTC] fetching last {years_back} years of COMEX Gold COT data", flush=True)
    # Socrata supports SoQL filtering
    params = {
        "$select": "report_date_as_yyyy_mm_dd,m_money_positions_long_all,m_money_positions_short_all,"
                   "comm_positions_long_all,comm_positions_short_all,open_interest_all",
        "$where": f"cftc_contract_market_code='{GOLD_CONTRACT_CODE}'",
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$limit": str(years_back * 53),  # weekly = ~53 per year
    }
    resp = requests.get(SOCRATA_URL, params=params, timeout=30)
    resp.raise_for_status()
    raw = resp.json()
    if not raw:
        raise RuntimeError("CFTC API returned empty result — contract code or schema may have changed")
    df = pd.DataFrame(raw)
    print(f"[CFTC] received {len(df)} weekly rows", flush=True)

    # Parse + compute net positioning
    df["report_date_as_yyyy_mm_dd"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"], utc=True)
    for col in ("m_money_positions_long_all", "m_money_positions_short_all",
                "comm_positions_long_all", "comm_positions_short_all", "open_interest_all"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    out = pd.DataFrame({
        "week_ending_utc": df["report_date_as_yyyy_mm_dd"],
        "large_spec_long": df["m_money_positions_long_all"],
        "large_spec_short": df["m_money_positions_short_all"],
        "large_spec_net": df["m_money_positions_long_all"] - df["m_money_positions_short_all"],
        "comm_long": df["comm_positions_long_all"],
        "comm_short": df["comm_positions_short_all"],
        "comm_net": df["comm_positions_long_all"] - df["comm_positions_short_all"],
        "total_oi": df["open_interest_all"],
    })
    out["spec_pct_of_oi"] = out["large_spec_net"] / out["total_oi"].replace(0, 1).abs()
    out = out.sort_values("week_ending_utc").reset_index(drop=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest CFTC COT data for COMEX Gold")
    parser.add_argument("--years-back", type=int, default=5)
    parser.add_argument("--out", type=str, default=str(OUT_PATH))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_cot_xau(years_back=args.years_back)
    df.to_parquet(args.out, index=False)
    print(f"[CFTC] wrote {len(df)} rows -> {args.out}", flush=True)
    print(f"[CFTC] latest week: {df['week_ending_utc'].max()}", flush=True)
    print(f"[CFTC] latest large_spec_net: {df['large_spec_net'].iloc[-1]:,.0f}", flush=True)
    print(f"[CFTC] latest spec_pct_of_oi: {df['spec_pct_of_oi'].iloc[-1]:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
