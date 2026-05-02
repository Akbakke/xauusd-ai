#!/usr/bin/env python3
"""Probe OANDA M1 historical availability for XAUUSD year-by-year.

Background
----------
Phase 3A requires M1 data for 2020-2024 (canonical store currently has
M1 only for 2025-2026). Before we commit to the full backfill, this
probe verifies that OANDA's API actually serves M1 candles back to 2020.
If a year is missing, we have to source it from Dukascopy ticks instead.

For each target year (2020..2024) the probe requests a single one-day
M1 window mid-year (June 15) and reports:
  - number of candles returned
  - first / last candle timestamp
  - whether the request succeeded
  - whether the candle timestamps are inside the requested window

If every year returns ~1440 M1 bars (24h * 60 min) the OANDA backfill
path is the right approach. If 2020 returns zero, the probe writes a
recommendation to fall back to Dukascopy ticks for the missing years.

Pure read-only against the OANDA REST API; no writes. Output is a small
JSON summary file under reports/.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from gx1.execution.oanda_client import OandaClient, OandaClientConfig
from gx1.execution.oanda_credentials import load_oanda_credentials
from gx1.utils.env_loader import load_dotenv_if_present


DEFAULT_OUT_DIR = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity") / "PROBE_OANDA_M1_HISTORICAL_AVAILABILITY_V1"
TARGET_YEARS = [2020, 2021, 2022, 2023, 2024]
PROBE_INSTRUMENT = "XAU_USD"
PROBE_GRANULARITY = "M1"
PROBE_DAY_OF_YEAR = (6, 16)  # June 16 (mid-year, weekday range)


def _probe_year(client: OandaClient, year: int) -> dict[str, Any]:
    """Try fetching one day of M1 candles mid-year. Returns a result dict."""
    month, day = PROBE_DAY_OF_YEAR
    try:
        target_date = datetime(year, month, day, tzinfo=timezone.utc)
    except ValueError:
        return {"year": year, "status": "INVALID_DATE"}

    # Find a weekday near target (skip Sat/Sun where forex is closed).
    while target_date.weekday() >= 5:
        target_date = target_date + timedelta(days=1)

    start = target_date.replace(hour=0, minute=0, second=0)
    end = target_date.replace(hour=23, minute=59, second=59)
    try:
        df = client.get_candles_chunked(
            instrument=PROBE_INSTRUMENT,
            granularity=PROBE_GRANULARITY,
            from_ts=pd.Timestamp(start),
            to_ts=pd.Timestamp(end),
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "year": year,
            "probe_date_v1": start.isoformat(),
            "status": "API_ERROR",
            "error_v1": str(exc)[:300],
            "candle_count_v1": 0,
        }

    if df is None or len(df) == 0:
        return {
            "year": year,
            "probe_date_v1": start.isoformat(),
            "status": "EMPTY_RESPONSE",
            "candle_count_v1": 0,
        }

    n = int(len(df))
    first_ts = str(df.index.min())
    last_ts = str(df.index.max())
    in_window = bool(
        (df.index.min() >= start.replace(tzinfo=timezone.utc))
        and (df.index.max() <= end.replace(tzinfo=timezone.utc))
    )
    return {
        "year": year,
        "probe_date_v1": start.isoformat(),
        "status": "OK" if n >= 100 and in_window else "PARTIAL",
        "candle_count_v1": n,
        "first_candle_ts_v1": first_ts,
        "last_candle_ts_v1": last_ts,
        "in_requested_window_v1": in_window,
        "expected_around_v1": 1440,
    }


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    n_ok = sum(1 for r in results if r.get("status") == "OK")
    n_total = len(results)
    missing_years = [r["year"] for r in results if r.get("status") != "OK"]
    if n_ok == n_total:
        recommendation = (
            "OANDA serves M1 candles for ALL probed years. Proceed with "
            "BACKFILL_XAUUSD_M1_2020_2024_V1 using OandaClient (3A2)."
        )
        feasible = "OANDA_DIRECT_BACKFILL"
    elif n_ok > 0:
        recommendation = (
            f"OANDA serves M1 for {n_ok}/{n_total} probed years; missing "
            f"years {missing_years}. Use OANDA for available years and "
            "Dukascopy ticks (or another source) for missing years."
        )
        feasible = "MIXED_SOURCE_BACKFILL_REQUIRED"
    else:
        recommendation = (
            "OANDA does NOT serve M1 candles for any probed year. Switch "
            "the entire 2020-2024 backfill to Dukascopy ticks -> M1 bar "
            "construction (separate gate)."
        )
        feasible = "DUKASCOPY_FALLBACK_REQUIRED"
    return {
        "n_years_probed_v1": n_total,
        "n_years_ok_v1": n_ok,
        "missing_years_v1": missing_years,
        "feasibility_v1": feasible,
        "recommendation_v1": recommendation,
        "per_year_results_v1": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe OANDA M1 historical availability for XAUUSD."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for the probe summary JSON.",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    load_dotenv_if_present()
    creds = load_oanda_credentials()
    client = OandaClient(
        OandaClientConfig(
            api_key=creds.api_token,
            account_id=creds.account_id,
            env=creds.env,
        )
    )
    print(
        f"[PROBE] env={creds.env} account={creds.account_id[:6]}...{creds.account_id[-4:]} "
        f"api_url={creds.api_url}",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    for year in TARGET_YEARS:
        print(f"[PROBE] year={year} ...", flush=True)
        res = _probe_year(client, year)
        results.append(res)
        status = res.get("status")
        n = res.get("candle_count_v1", 0)
        print(f"[PROBE] year={year} status={status} candles={n}", flush=True)

    summary = _summarize(results)
    out_path = out_dir / "probe_summary_v1.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[PROBE] summary_path={out_path}", flush=True)
    print(f"[PROBE] feasibility={summary['feasibility_v1']}", flush=True)
    print(f"[PROBE] recommendation={summary['recommendation_v1']}", flush=True)


if __name__ == "__main__":
    main()
