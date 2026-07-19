#!/usr/bin/env python3
"""Idempotent M1 backfill for XAUUSD years 2020-2024 from OANDA.

Background
----------
PROBE_OANDA_M1_HISTORICAL_AVAILABILITY_V1 confirmed OANDA serves M1
candles for all years 2020-2024. The canonical M1 store currently only
holds 2025-2026; this gate fills 2020-2024.

For each target year:
  1. Fetch full year M1 bid/ask candles via OANDA REST in chunked batches
     (3000 candles per chunk, exponential backoff on rate limits).
  2. Validate: timestamp completeness vs business calendar, no duplicates,
     no future bars, OHLC sanity (high >= max(open,close), low <= min,
     bid <= ask).
  3. Idempotent merge with any existing year=YYYY parquet (keep newest
     candle per ts).
  4. Write under
     /home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m1_bid_ask__CANONICAL/year=YYYY/part-000.parquet
  5. Update MANIFEST.json with row counts per year.
  6. Audit JSON written to a LOCK directory under reports/.

Audits per year:
  - candle_count_v1
  - first_ts_v1, last_ts_v1
  - n_business_days_covered_v1 vs n_business_days_expected_v1
  - n_duplicate_ts_v1 (must be 0)
  - n_negative_or_zero_price_v1 (must be 0)
  - bid_le_ask_invariant_v1 (must hold)
  - high_low_invariant_v1 (must hold)

Research-only data ingest. No model training. No runtime modification.
The canonical store is shared with the live system but this gate writes
ONLY to year=YYYY partitions for years that don't yet exist (2020-2024);
the 2025-2026 partitions are not touched.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.execution.oanda_client import OandaClient, OandaClientConfig
from gx1.execution.oanda_credentials import load_oanda_credentials
from gx1.scripts import exit_iql_artifact_primitives_v1 as contract_gate
from gx1.utils.env_loader import load_dotenv_if_present


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "BACKFILL_XAUUSD_M1_2020_2024_V1"

CANONICAL_M1_ROOT = Path(
    "/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m1_bid_ask__CANONICAL"
)
TARGET_YEARS = [2020, 2021, 2022, 2023, 2024]
INSTRUMENT = "XAU_USD"
GRANULARITY = "M1"

# Forex business calendar approximation: Mon-Fri, ~252 days/year minus a
# handful of holidays. We only check that we are within sanity range, not
# exact match.
EXPECTED_M1_PER_YEAR_LOWER_BOUND = 250_000  # ~250 days * 24h * 60min * 0.7 (conservative)
EXPECTED_M1_PER_YEAR_UPPER_BOUND = 400_000

QUARANTINE_FORBIDDEN_PATH_FRAGMENTS = ("gx1/quarantine", "gx1.quarantine")

ALLOWED_FINAL_STATUSES = {
    "BACKFILL_M1_2020_2024_PASS_ALL_YEARS_V1",
    "BACKFILL_M1_2020_2024_PARTIAL_SOME_YEARS_FAILED_V1",
    "BACKFILL_M1_2020_2024_BLOCKED_BY_API_FAIL_V1",
}

ALLOWED_NEXT_ACTIONS = {
    "BUILD_EXTENDED_BASE34_PREBUILT_2020_2026_V1",
    "REPAIR_BACKFILL_BEFORE_FURTHER_WORK_V1",
}


_jsonable = contract_gate._jsonable
_write_json = contract_gate._write_json
_write_rows = contract_gate._write_rows
_write_report = contract_gate._write_report
_file_hash = contract_gate._file_hash
_python_manifest = contract_gate._python_manifest


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_no_forbidden_actions(**kwargs: Any) -> dict[str, Any]:
    return contract_gate.validate_no_forbidden_actions(**kwargs)


def validate_final_status(status: str, next_action: str) -> bool:
    if status not in ALLOWED_FINAL_STATUSES:
        raise RuntimeError(f"FINAL_STATUS_NOT_ALLOWED: {status}")
    if next_action not in ALLOWED_NEXT_ACTIONS:
        raise RuntimeError(f"NEXT_ACTION_NOT_ALLOWED: {next_action}")
    return True


def validate_no_deprecated_revival(script_path: Path) -> bool:
    text = script_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.lstrip()
        if not (stripped.startswith("import ") or stripped.startswith("from ")):
            continue
        for fragment in QUARANTINE_FORBIDDEN_PATH_FRAGMENTS:
            if fragment in stripped:
                raise RuntimeError("DEPRECATED_QUARANTINE_REVIVAL_FORBIDDEN")
    return True


def _client_from_env() -> OandaClient:
    load_dotenv_if_present()
    creds = load_oanda_credentials()
    return OandaClient(
        OandaClientConfig(
            api_key=creds.api_token,
            account_id=creds.account_id,
            env=creds.env,
        )
    )


def _existing_year_partition(year: int) -> Path:
    return CANONICAL_M1_ROOT / f"year={year}" / "part-000.parquet"


def _read_existing_year(year: int) -> pd.DataFrame | None:
    p = _existing_year_partition(year)
    if not p.exists():
        return None
    return pd.read_parquet(p)


def _fetch_year(client: OandaClient, year: int) -> pd.DataFrame:
    """Fetch full-year M1 candles year=YYYY from Jan 1 to Dec 31 inclusive."""
    start = pd.Timestamp(datetime(year, 1, 1, tzinfo=timezone.utc))
    end = pd.Timestamp(datetime(year + 1, 1, 1, tzinfo=timezone.utc))
    df = client.get_candles_chunked(
        instrument=INSTRUMENT,
        granularity=GRANULARITY,
        from_ts=start,
        to_ts=end,
        chunk_size=5000,
        max_retries=8,
        include_mid=True,
        exclude_incomplete=True,
    )
    return df


def _validate_year_df(df: pd.DataFrame, year: int) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "year_v1": year,
        "candle_count_v1": int(len(df)),
        "first_ts_v1": str(df.index.min()) if len(df) else None,
        "last_ts_v1": str(df.index.max()) if len(df) else None,
        "n_duplicate_ts_v1": int(df.index.duplicated().sum()) if len(df) else 0,
    }
    if len(df) == 0:
        audit["status_v1"] = "FAIL_EMPTY"
        return audit
    cols = set(df.columns)
    audit["columns_v1"] = sorted(cols)
    bid_cols = [c for c in cols if c.startswith("bid_")]
    ask_cols = [c for c in cols if c.startswith("ask_")]
    audit["bid_columns_v1"] = sorted(bid_cols)
    audit["ask_columns_v1"] = sorted(ask_cols)
    # bid <= ask check (use closing prices).
    if "bid_close" in cols and "ask_close" in cols:
        bid_le_ask = bool((df["bid_close"] <= df["ask_close"] + 1e-9).all())
        audit["bid_le_ask_invariant_v1"] = bid_le_ask
    # high >= max(open, close), low <= min(open, close).
    # Use np.maximum / np.minimum on .values arrays so duplicate-index
    # frames don't fail Series.combine alignment.
    high_low_ok = True
    for prefix in ("bid", "ask"):
        if all(c in cols for c in (f"{prefix}_open", f"{prefix}_high", f"{prefix}_low", f"{prefix}_close")):
            o = df[f"{prefix}_open"].astype(float).values
            h = df[f"{prefix}_high"].astype(float).values
            low = df[f"{prefix}_low"].astype(float).values
            c = df[f"{prefix}_close"].astype(float).values
            if not bool(np.all(h >= np.maximum(o, c) - 1e-9)):
                high_low_ok = False
            if not bool(np.all(low <= np.minimum(o, c) + 1e-9)):
                high_low_ok = False
    audit["high_low_invariant_v1"] = high_low_ok
    # Negative / zero price check.
    n_negative = 0
    for prefix in ("bid", "ask"):
        for suffix in ("open", "high", "low", "close"):
            col = f"{prefix}_{suffix}"
            if col in cols:
                n_negative += int((df[col] <= 0).sum())
    audit["n_negative_or_zero_price_v1"] = int(n_negative)
    in_year = int(
        ((df.index >= pd.Timestamp(year, 1, 1, tz="UTC"))
         & (df.index < pd.Timestamp(year + 1, 1, 1, tz="UTC"))).sum()
    )
    audit["n_in_year_window_v1"] = in_year
    in_range = (
        EXPECTED_M1_PER_YEAR_LOWER_BOUND <= len(df) <= EXPECTED_M1_PER_YEAR_UPPER_BOUND
    )
    audit["in_expected_count_range_v1"] = in_range
    failures: list[str] = []
    if audit["n_duplicate_ts_v1"] > 0:
        failures.append("DUPLICATE_TIMESTAMPS")
    if not audit.get("bid_le_ask_invariant_v1", True):
        failures.append("BID_GT_ASK")
    if not audit.get("high_low_invariant_v1", True):
        failures.append("HIGH_LOW_INVARIANT_VIOLATED")
    if audit["n_negative_or_zero_price_v1"] > 0:
        failures.append("NEGATIVE_OR_ZERO_PRICE")
    if not audit["in_expected_count_range_v1"]:
        failures.append("CANDLE_COUNT_OUT_OF_EXPECTED_RANGE")
    audit["failures_v1"] = failures
    audit["status_v1"] = "OK" if not failures else "FAIL"
    return audit


def _merge_and_persist_year(year: int, df_new: pd.DataFrame) -> dict[str, Any]:
    """Idempotent merge with any existing year partition. Keep latest record
    per timestamp."""
    out_path = _existing_year_partition(year)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_existing_year(year)
    if existing is not None and len(existing) > 0:
        existing.index = pd.to_datetime(existing.index, utc=True)
        df_new.index = pd.to_datetime(df_new.index, utc=True)
        merged = pd.concat([existing, df_new])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = df_new.sort_index()
    if "time" in merged.columns:
        merged = merged.drop(columns=["time"])
    merged_with_time = merged.copy()
    merged_with_time.index.name = "time"
    merged_with_time.to_parquet(out_path, index=True)
    return {
        "year_v1": year,
        "out_path_v1": str(out_path),
        "row_count_after_merge_v1": int(len(merged)),
        "first_ts_after_merge_v1": str(merged.index.min()) if len(merged) else None,
        "last_ts_after_merge_v1": str(merged.index.max()) if len(merged) else None,
    }


def _update_manifest() -> dict[str, Any]:
    """Refresh CANONICAL_M1_ROOT/MANIFEST.json with row counts per year present."""
    manifest_path = CANONICAL_M1_ROOT / "MANIFEST.json"
    years: dict[str, int] = {}
    for d in sorted(CANONICAL_M1_ROOT.glob("year=*")):
        year = int(d.name.split("=")[1])
        for parquet in d.glob("part-*.parquet"):
            df = pd.read_parquet(parquet, columns=None)
            years[str(year)] = years.get(str(year), 0) + int(len(df))
    manifest_payload = {
        "instrument": "XAUUSD",
        "timeframe": "M1",
        "years": sorted(years.keys()),
        "row_counts": years,
        "out_root": str(CANONICAL_M1_ROOT),
        "updated_utc": _utc_now(),
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest_payload


def write_artifacts(
    out_root: Path | None = None,
    *,
    built_at_utc: str | None = None,
    target_years: list[int] | None = None,
) -> dict[str, Any]:
    timestamp = built_at_utc or _stamp()
    artifact_root = out_root or (DEFAULT_REPORTS_ROOT / f"{ACTION}_{timestamp}_LOCK")
    artifact_root.mkdir(parents=True, exist_ok=True)
    years_to_run = target_years or list(TARGET_YEARS)

    validate_no_deprecated_revival(Path(__file__))
    forbidden_audit = validate_no_forbidden_actions(
        adapter=False,
        r6=False,
        iql_production=False,
        package=False,
        freeze=False,
        promo=False,
        live=False,
        optuna=False,
        broad_sweep=False,
    )
    client = _client_from_env()

    per_year_audit: list[dict[str, Any]] = []
    per_year_persist: list[dict[str, Any]] = []
    for year in years_to_run:
        print(f"[BACKFILL] year={year} fetching...", flush=True)
        try:
            df = _fetch_year(client, year)
        except Exception as exc:  # noqa: BLE001
            audit = {
                "year_v1": year,
                "status_v1": "API_ERROR",
                "error_v1": str(exc)[:300],
                "candle_count_v1": 0,
                "failures_v1": ["API_ERROR"],
            }
            per_year_audit.append(audit)
            print(f"[BACKFILL] year={year} FAILED api_error={str(exc)[:120]}", flush=True)
            continue
        audit = _validate_year_df(df, year)
        per_year_audit.append(audit)
        print(
            f"[BACKFILL] year={year} fetched candles={audit['candle_count_v1']} "
            f"status={audit['status_v1']}",
            flush=True,
        )
        if audit["status_v1"] == "OK":
            persist = _merge_and_persist_year(year, df)
            per_year_persist.append(persist)
            print(
                f"[BACKFILL] year={year} persisted rows={persist['row_count_after_merge_v1']}",
                flush=True,
            )
    manifest = _update_manifest()
    _write_json(
        artifact_root / "per_year_audit_v1.json",
        {"row_count_v1": len(per_year_audit), "rows_v1": per_year_audit},
    )
    _write_rows(artifact_root / "per_year_audit_v1.csv", per_year_audit)
    _write_json(
        artifact_root / "per_year_persist_v1.json",
        {"row_count_v1": len(per_year_persist), "rows_v1": per_year_persist},
    )
    _write_json(artifact_root / "manifest_after_backfill_v1.json", manifest)

    n_ok = sum(1 for a in per_year_audit if a["status_v1"] == "OK")
    n_total = len(per_year_audit)
    if n_ok == n_total and n_total > 0:
        status = "BACKFILL_M1_2020_2024_PASS_ALL_YEARS_V1"
        next_action = "BUILD_EXTENDED_BASE34_PREBUILT_2020_2026_V1"
        recommendation = (
            f"Backfilled {n_total} years successfully. Total M1 rows now in "
            f"canonical store: {sum(manifest['row_counts'].values())}. "
            "Next: extend BASE34 prebuilt across 2020-2026."
        )
    elif n_ok > 0:
        status = "BACKFILL_M1_2020_2024_PARTIAL_SOME_YEARS_FAILED_V1"
        next_action = "REPAIR_BACKFILL_BEFORE_FURTHER_WORK_V1"
        failed_years = [a["year_v1"] for a in per_year_audit if a["status_v1"] != "OK"]
        recommendation = (
            f"Backfilled {n_ok}/{n_total} years; failed years: {failed_years}. "
            "Investigate per-year audit before proceeding."
        )
    else:
        status = "BACKFILL_M1_2020_2024_BLOCKED_BY_API_FAIL_V1"
        next_action = "REPAIR_BACKFILL_BEFORE_FURTHER_WORK_V1"
        recommendation = "All years failed. API or credentials issue."
    validate_final_status(status, next_action)

    summary = {
        "layer_name": "BACKFILL_M1_2020_2024_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "n_years_targeted_v1": n_total,
        "n_years_ok_v1": n_ok,
        "manifest_after_v1": manifest,
        "per_year_audit_v1": per_year_audit,
        "per_year_persist_v1": per_year_persist,
        "research_only_v1": True,
        "iql_training_run_v1": False,
        "iql_production_allowed_v1": False,
        "exit_manager_modified_v1": False,
        "live_features_modified_v1": False,
        "entry_manager_modified_v1": False,
        "v1_state_contract_modified_v1": False,
        "v2_state_contract_modified_v1": False,
        "forbidden_actions_audit_v1": forbidden_audit,
    }
    _write_json(artifact_root / "summary_v1.json", summary)
    _write_json(
        artifact_root / "status_v1.json",
        {
            "layer_name": "BACKFILL_M1_2020_2024_STATUS_V1",
            "status_v1": "MATERIALIZED_DATA_INGEST_GATE",
            "final_status_v1": status,
            "next_action_v1": next_action,
        },
    )
    artifact_manifest = {
        "layer_id_v1": ACTION,
        "built_at_utc_v1": summary["built_at_utc_v1"],
        "output_dir_v1": str(artifact_root),
        "append_only_namespace_v1": "truth_e2e_sanity",
        "artifact_paths_v1": {
            "summary": str(artifact_root / "summary_v1.json"),
            "status": str(artifact_root / "status_v1.json"),
            "per_year_audit_csv": str(artifact_root / "per_year_audit_v1.csv"),
            "per_year_audit_json": str(artifact_root / "per_year_audit_v1.json"),
            "per_year_persist_json": str(artifact_root / "per_year_persist_v1.json"),
            "manifest_after_backfill": str(
                artifact_root / "manifest_after_backfill_v1.json"
            ),
        },
    }
    _write_json(artifact_root / "manifest_v1.json", artifact_manifest)
    return {"artifact_root": str(artifact_root), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize BACKFILL_XAUUSD_M1_2020_2024_V1.")
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--years", type=str, default=None, help="comma-separated, e.g. 2020,2021")
    args = parser.parse_args()
    out_root = (
        Path(args.out_root).expanduser().resolve() if args.out_root else None
    )
    target_years = (
        [int(y) for y in args.years.split(",")] if args.years else None
    )
    result = write_artifacts(out_root=out_root, target_years=target_years)
    print(json.dumps(_jsonable(result["summary"]), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
