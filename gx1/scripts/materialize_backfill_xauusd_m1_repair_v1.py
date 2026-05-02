#!/usr/bin/env python3
"""Repair backfill for XAUUSD M1 years where the v1 chunked backfill failed.

Background
----------
BACKFILL_XAUUSD_M1_2020_2024_V1 succeeded for 2020 and 2021 but failed
for 2022 and 2023 (and is currently running 2024). Root cause: the
production `OandaClient.get_candles` raises `OandaAPIError` when the API
returns an empty candles array, which happens for chunks that land
entirely inside a weekend or extended holiday window. The chunked
fetcher retries 8 times with exponential backoff, then propagates the
error - aborting the entire year.

This repair gate fetches the missing years in DAY-BY-DAY pieces, with
explicit empty-day tolerance: if a day returns no candles (weekend,
holiday, market close), we log it and continue. Slower than the chunked
fetcher (one HTTP call per day) but reliable.

For each requested year:
  1. Iterate Jan 1 .. Dec 31 day by day.
  2. For each day, fetch [00:00, 24:00) M1 candles via get_candles.
  3. Catch OandaAPIError("No candles returned") and skip with audit.
  4. Concatenate all non-empty days, sort, dedup.
  5. Run same validation as v1 (OHLC invariants, bid<=ask, count range).
  6. Idempotent merge with any existing year partition.
  7. Update canonical MANIFEST.json.

Research-only data ingest. No model training. No runtime modification.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.execution.oanda_client import OandaAPIError, OandaClient, OandaClientConfig
from gx1.execution.oanda_credentials import load_oanda_credentials
from gx1.scripts import (
    materialize_backfill_xauusd_m1_2020_2024_v1 as v1_gate,
)
from gx1.scripts import (
    materialize_build_iql_offline_data_contract_research_only_v1 as contract_gate,
)
from gx1.utils.env_loader import load_dotenv_if_present


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "BACKFILL_XAUUSD_M1_REPAIR_V1"

QUARANTINE_FORBIDDEN_PATH_FRAGMENTS = ("gx1/quarantine", "gx1.quarantine")
INSTRUMENT = "XAU_USD"
GRANULARITY = "M1"


_jsonable = contract_gate._jsonable
_write_json = contract_gate._write_json
_write_rows = contract_gate._write_rows
_file_hash = contract_gate._file_hash
_python_manifest = contract_gate._python_manifest


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _fetch_year_day_by_day(
    client: OandaClient, year: int
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Day-by-day M1 fetch with empty-day tolerance.

    Returns (concatenated_df, per_day_audit). Empty days are recorded with
    status=EMPTY_OR_HOLIDAY but do not abort the year.
    """
    parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    cur = start
    while cur < end:
        next_day = cur + timedelta(days=1)
        try:
            df = client.get_candles(
                INSTRUMENT,
                GRANULARITY,
                from_ts=pd.Timestamp(cur),
                to_ts=pd.Timestamp(next_day),
                include_mid=True,
                exclude_incomplete=True,
            )
            n = int(len(df))
            if n > 0:
                parts.append(df)
                audit_rows.append(
                    {
                        "date_v1": cur.strftime("%Y-%m-%d"),
                        "status_v1": "OK",
                        "candle_count_v1": n,
                    }
                )
            else:
                audit_rows.append(
                    {
                        "date_v1": cur.strftime("%Y-%m-%d"),
                        "status_v1": "EMPTY_BUT_NOT_ERROR",
                        "candle_count_v1": 0,
                    }
                )
        except OandaAPIError as exc:
            msg = str(exc)
            if "No candles returned" in msg:
                audit_rows.append(
                    {
                        "date_v1": cur.strftime("%Y-%m-%d"),
                        "status_v1": "EMPTY_OR_HOLIDAY",
                        "candle_count_v1": 0,
                        "note_v1": "OANDA returned no candles; tolerated as weekend/holiday.",
                    }
                )
            else:
                audit_rows.append(
                    {
                        "date_v1": cur.strftime("%Y-%m-%d"),
                        "status_v1": "API_ERROR",
                        "candle_count_v1": 0,
                        "error_v1": msg[:300],
                    }
                )
        # Throttle ~10 req/s.
        time.sleep(0.1)
        cur = next_day
    if parts:
        df_concat = pd.concat(parts).sort_index()
        df_concat = df_concat[~df_concat.index.duplicated(keep="last")]
    else:
        df_concat = pd.DataFrame()
    return df_concat, audit_rows


def _ok_days(audit_rows: list[dict[str, Any]]) -> int:
    return sum(1 for a in audit_rows if a["status_v1"] == "OK")


def _empty_days(audit_rows: list[dict[str, Any]]) -> int:
    return sum(
        1 for a in audit_rows
        if a["status_v1"] in {"EMPTY_OR_HOLIDAY", "EMPTY_BUT_NOT_ERROR"}
    )


def _api_error_days(audit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [a for a in audit_rows if a["status_v1"] == "API_ERROR"]


def write_artifacts(
    target_years: list[int],
    out_root: Path | None = None,
    *,
    built_at_utc: str | None = None,
) -> dict[str, Any]:
    timestamp = built_at_utc or _stamp()
    artifact_root = out_root or (DEFAULT_REPORTS_ROOT / f"{ACTION}_{timestamp}_LOCK")
    artifact_root.mkdir(parents=True, exist_ok=True)

    v1_gate.validate_no_deprecated_revival(Path(v1_gate.__file__))
    forbidden_audit = v1_gate.validate_no_forbidden_actions(
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
    print(
        f"[REPAIR] env={client.env if hasattr(client, 'env') else '?'} "
        f"target_years={target_years}",
        flush=True,
    )

    per_year_audit: list[dict[str, Any]] = []
    per_year_persist: list[dict[str, Any]] = []
    all_per_day_audit: list[dict[str, Any]] = []
    for year in target_years:
        print(f"[REPAIR] year={year} fetching day-by-day...", flush=True)
        df, day_audit = _fetch_year_day_by_day(client, year)
        for d in day_audit:
            d["year_v1"] = year
        all_per_day_audit.extend(day_audit)
        ok = _ok_days(day_audit)
        empty = _empty_days(day_audit)
        api_errors = _api_error_days(day_audit)
        print(
            f"[REPAIR] year={year} ok_days={ok} empty_days={empty} "
            f"api_error_days={len(api_errors)} candles={int(len(df))}",
            flush=True,
        )
        audit = v1_gate._validate_year_df(df, year)
        audit["ok_days_v1"] = ok
        audit["empty_or_holiday_days_v1"] = empty
        audit["api_error_days_v1"] = len(api_errors)
        per_year_audit.append(audit)
        if audit["status_v1"] == "OK":
            persist = v1_gate._merge_and_persist_year(year, df)
            per_year_persist.append(persist)
            print(
                f"[REPAIR] year={year} persisted rows={persist['row_count_after_merge_v1']}",
                flush=True,
            )
    manifest = v1_gate._update_manifest()
    _write_json(
        artifact_root / "per_year_audit_v1.json",
        {"row_count_v1": len(per_year_audit), "rows_v1": per_year_audit},
    )
    _write_rows(artifact_root / "per_year_audit_v1.csv", per_year_audit)
    _write_json(
        artifact_root / "per_day_audit_v1.json",
        {"row_count_v1": len(all_per_day_audit), "rows_v1": all_per_day_audit},
    )
    _write_rows(artifact_root / "per_day_audit_v1.csv", all_per_day_audit)
    _write_json(
        artifact_root / "per_year_persist_v1.json",
        {"row_count_v1": len(per_year_persist), "rows_v1": per_year_persist},
    )
    _write_json(artifact_root / "manifest_after_repair_v1.json", manifest)

    n_ok = sum(1 for a in per_year_audit if a["status_v1"] == "OK")
    n_total = len(per_year_audit)
    if n_ok == n_total and n_total > 0:
        status = "REPAIR_M1_PASS_ALL_YEARS_V1"
        next_action = "BUILD_EXTENDED_BASE34_PREBUILT_2020_2026_V1"
        recommendation = (
            f"Repair succeeded for all {n_total} years. Total M1 rows in "
            f"canonical store: {sum(manifest['row_counts'].values())}. "
            "Next: extend BASE34 prebuilt across 2020-2026."
        )
    elif n_ok > 0:
        status = "REPAIR_M1_PARTIAL_SOME_YEARS_FAILED_V1"
        next_action = "REPAIR_BACKFILL_BEFORE_FURTHER_WORK_V1"
        failed_years = [a["year_v1"] for a in per_year_audit if a["status_v1"] != "OK"]
        recommendation = (
            f"Repair {n_ok}/{n_total} years; failed: {failed_years}."
        )
    else:
        status = "REPAIR_M1_BLOCKED_BY_API_FAIL_V1"
        next_action = "REPAIR_BACKFILL_BEFORE_FURTHER_WORK_V1"
        recommendation = "All years failed in repair. API or credentials issue."
    summary = {
        "layer_name": "BACKFILL_M1_REPAIR_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "target_years_v1": target_years,
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
            "layer_name": "BACKFILL_M1_REPAIR_STATUS_V1",
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
            "per_day_audit_csv": str(artifact_root / "per_day_audit_v1.csv"),
            "per_day_audit_json": str(artifact_root / "per_day_audit_v1.json"),
            "per_year_persist_json": str(artifact_root / "per_year_persist_v1.json"),
            "manifest_after_repair": str(artifact_root / "manifest_after_repair_v1.json"),
        },
    }
    _write_json(artifact_root / "manifest_v1.json", artifact_manifest)
    return {"artifact_root": str(artifact_root), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize BACKFILL_XAUUSD_M1_REPAIR_V1.")
    parser.add_argument(
        "--years",
        type=str,
        required=True,
        help="comma-separated list of years to repair, e.g. 2022,2023,2024",
    )
    parser.add_argument("--out-root", type=str, default=None)
    args = parser.parse_args()
    target_years = [int(y) for y in args.years.split(",")]
    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else None
    result = write_artifacts(target_years=target_years, out_root=out_root)
    print(json.dumps(_jsonable(result["summary"]), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
