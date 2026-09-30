#!/usr/bin/env python3
"""Bounded A/B/C research orchestration. Implements A/C, B source checks and the matched B research core.

Reuses indicator, clock, learner, portfolio and inference owners. No native
training, broker access, TEST, automatic parameter search or result promotion.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from copy import deepcopy
import hashlib
from html.parser import HTMLParser
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import date, datetime, timezone
from urllib.request import Request, urlopen
from urllib.parse import urlencode
import zipfile

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar, nearest_workday, sunday_to_monday
import pyarrow.parquet as pq

from gx1.features.htf_features import _resample_ohlc_for_model_native_scalars
from gx1.features.technical_indicators_v1 import classic_ema, wilder_atr
from gx1.time.session_detector import M5_BAR_DURATION, TRADING_SESSION_DURATION
from gx1.scripts.research_entry_direction_walkforward_v1 import RidgeGram, Tape, fit_hgb
from gx1.scripts.research_model_free_baselines_v1 import (
    ResearchFinancingCurve, causal_risk_units, max_t_inference, portfolio_path,
    portfolio_period_returns, portfolio_summary, stationary_bootstrap_indices,
)

ROOT = Path(__file__).resolve().parents[2]
FEATURES = [*(f"momentum_{n}_atr14" for n in (21, 63, 126, 252)),
            "range_position_252", "atr14_over_atr252", "ema200_distance_atr14"]

# The six-source B contract adds exactly two fields per named source.
B_MACRO_FEATURES = [
    "real10y_level", "real10y_change21D1", "usd_log_level", "usd_log_change21D1",
    "breakeven10y_level", "breakeven10y_change21D1", "gld_log_level", "gld_log_change21D1",
    "cot_noncommercial_net_over_oi", "cot_change4reports", "vix_log_level", "vix_log_change21D1",
]
B_FEATURES = FEATURES + B_MACRO_FEATURES


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def write_json(path: Path, obj: dict) -> None:
    if path.exists():
        raise RuntimeError(f"TA_OUTPUT_EXISTS: {path}")
    temp = path.with_suffix(path.suffix + ".part")
    with temp.open("x") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.rename(temp, path)


def checked_spec(path: Path, digest: str) -> dict:
    if sha(path) != digest:
        raise RuntimeError("TA_SPEC_HASH")
    if git("branch", "--show-current") != "work/gx1-current" or git("status", "--porcelain"):
        raise RuntimeError("TA_REQUIRES_CANONICAL_CLEAN_SOURCE")
    relative = path.resolve().relative_to(ROOT).as_posix()
    git("ls-files", "--error-unmatch", relative)
    spec = json.loads(path.read_text())
    for name, expected in spec["source_files"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError(f"TA_SOURCE_HASH: {name}")
    return spec


class AlfredDownloadForm(HTMLParser):
    """Read only the documented public vintage selector, never a latest alias."""

    def __init__(self) -> None:
        super().__init__()
        self.selected = None
        self.options: dict[str, list[str]] = {}

    def handle_starttag(self, tag: str, attrs: list) -> None:
        attrs = dict(attrs)
        if tag == "select":
            self.selected = attrs.get("name")
            self.options[self.selected] = []
        elif tag == "option" and self.selected is not None and "value" in attrs:
            self.options[self.selected].append(attrs["value"])

    def handle_endtag(self, tag: str) -> None:
        if tag == "select":
            self.selected = None


def alfred_form_body(raw: bytes, spec: dict) -> tuple[bytes, list[str]]:
    form = AlfredDownloadForm()
    form.feed(raw.decode("utf-8"))
    if ("lin" not in form.options.get("form[units]", [])
            or "1" not in form.options.get("form[file_type]", [])
            or "csv" not in form.options.get("form[file_format]", [])):
        raise RuntimeError("TA_ALFRED_FORM_SCHEMA")
    available = form.options["form[selected_vintage_dates][]"]
    for date in available:
        if datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d") != date:
            raise RuntimeError("TA_ALFRED_VINTAGE_FORMAT")
    selected = [date for date in available if spec["vintage_start"] <= date <= spec["vintage_end"]]
    if not selected or selected != sorted(set(selected)):
        raise RuntimeError("TA_ALFRED_VINTAGE_COVERAGE")
    values = [("form[units]", "lin"), ("form[obs_start_date]", spec["observation_start"]),
              ("form[obs_end_date]", spec["observation_end"]),
              ("form[entered_vintage_dates]", ""), ("form[file_type]", "1"),
              ("form[file_format]", "csv"), ("form[download_data]", "")]
    values += [("form[selected_vintage_dates][]", date) for date in selected]
    return urlencode(values).encode(), selected


def fetch_alfred(spec: dict, spec_path: Path) -> dict:
    """Bound source-admission fetch only. Never admits a predictor or runs a fit."""
    out = Path(spec["output_directory"])
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for series in spec["series"]:
        directory = out / series["id"]
        directory.mkdir()
        record = {"series": series["id"], "url": series["url"], "predictor_admitted": False}
        try:
            request = Request(series["url"], headers={"User-Agent": "GX1 offline research"})
            with urlopen(request, timeout=spec["timeout_seconds"]) as response:
                form = response.read(spec["maximum_form_bytes"] + 1)
            (directory / "FORM.html").write_bytes(form)
            if len(form) > spec["maximum_form_bytes"]:
                raise RuntimeError("TA_ALFRED_FORM_SIZE")
            body, dates = alfred_form_body(form, spec)
            (directory / "REQUEST.form").write_bytes(body)
            record.update(vintages=len(dates), first_vintage=dates[0], last_vintage=dates[-1],
                          form_sha256=sha(directory / "FORM.html"), request_sha256=sha(directory / "REQUEST.form"))
            request = Request(series["url"], data=body,
                              headers={"User-Agent": "GX1 offline research",
                                       "Content-Type": "application/x-www-form-urlencoded"})
            with urlopen(request, timeout=spec["timeout_seconds"]) as response:
                raw = response.read(spec["maximum_response_bytes"] + 1)
                record["final_url"] = response.url
            raw_path = directory / "RESPONSE.zip"
            raw_path.write_bytes(raw)
            record.update(raw_path=str(raw_path), raw_sha256=sha(raw_path), raw_bytes=len(raw))
            if len(raw) > spec["maximum_response_bytes"]:
                raise RuntimeError("TA_ALFRED_RESPONSE_SIZE")
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                members = archive.infolist()
                if sum(m.file_size for m in members) > spec["maximum_uncompressed_bytes"]:
                    raise RuntimeError("TA_ALFRED_UNCOMPRESSED_SIZE")
                if archive.testzip() is not None:
                    raise RuntimeError("TA_ALFRED_ZIP_CRC")
                record["members"] = {m.filename: {"bytes": m.file_size,
                    "sha256": hashlib.sha256(archive.read(m)).hexdigest()} for m in members}
            record["status"] = "RETRIEVED_NOT_ADMITTED"
        except Exception as exc:
            record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        write_json(directory / "RECEIPT.json", record)
        records.append(record)
        print(f"[TA-B-source] {series['id']} {record['status']}", flush=True)
    result = {"status": "RETRIEVAL_RECORDED", "manifest": str(spec_path),
              "manifest_sha256": sha(spec_path), "records": records,
              "predictor_admitted": False, "market_outcomes_read": False,
              "test_accessed": False, "native_training": False}
    write_json(out / "RESULT.json", result)
    write_json(out / "TERMINAL.json", {"status": "COMPLETE_SOURCE_AUDIT",
        "all_downloads_succeeded": all(r["status"] == "RETRIEVED_NOT_ADMITTED" for r in records),
        "result_sha256": sha(out / "RESULT.json")})
    return {"out": str(out), "series_status": {r["series"]: r["status"] for r in records}}



def alfred_version_summary(rows: list[tuple[str, str, str, str]]) -> dict:
    """Validate source intervals; exact chunk duplicates are audit-only deduplication."""
    unique = {}
    duplicates = 0
    for obs, value, start, end in rows:
        date.fromisoformat(obs)
        date.fromisoformat(start)
        if end:
            date.fromisoformat(end)
            if end < start:
                raise ValueError("ALFRED_REVERSED_INTERVAL")
        if value and not math.isfinite(float(value)):
            raise ValueError("ALFRED_NONFINITE_VALUE")
        key = (obs, start)
        if key in unique:
            if unique[key] != (value, end):
                raise ValueError("ALFRED_CONFLICTING_DUPLICATE")
            duplicates += 1
        else:
            unique[key] = (value, end)
    grouped = {}
    for (obs, start), (value, end) in sorted(unique.items()):
        grouped.setdefault(obs, []).append((start, end, value))
    gaps = 0
    for versions in grouped.values():
        for earlier, later in zip(versions, versions[1:]):
            if not earlier[1] or earlier[1] >= later[0]:
                raise ValueError("ALFRED_OVERLAPPING_INTERVALS")
            if (date.fromisoformat(later[0]) - date.fromisoformat(earlier[1])).days > 1:
                gaps += 1
    if not unique:
        raise ValueError("ALFRED_EMPTY_DATA")
    return {
        "input_rows": len(rows), "unique_observation_versions": len(unique),
        "identical_chunk_duplicates": duplicates, "observations": len(grouped),
        "first_observation": min(grouped), "last_observation": max(grouped),
        "first_realtime_start": min(k[1] for k in unique),
        "last_realtime_start": max(k[1] for k in unique),
        "missing_value_versions": sum(not v[0] for v in unique.values()),
        "revised_observations": sum(len(v) > 1 for v in grouped.values()),
        "gaps_between_version_intervals": gaps,
        "conflicting_duplicates": 0, "overlapping_intervals": 0,
    }


def audit_alfred_chunks(spec: dict, spec_path: Path, receipt_sha256: str) -> dict:
    """Audit already downloaded, hash-bound archives. Does not build B inputs."""
    root = Path(spec["canonical_receipt_directory"])
    receipt_path = root / "TRANSPORT_RECEIPT.json"
    if not receipt_sha256 or sha(receipt_path) != receipt_sha256:
        raise ValueError("ALFRED_TRANSPORT_RECEIPT_HASH")
    receipt = json.loads(receipt_path.read_text())
    if receipt["manifest_sha256"] != spec["download_manifest_sha256"]:
        raise ValueError("ALFRED_SOURCE_MANIFEST_HASH")
    expected_ids = [x["id"] for x in spec["series"]]
    if [x["series"] for x in receipt["series"]] != expected_ids:
        raise ValueError("ALFRED_SERIES_SET")
    if set(x["series"] for x in receipt["files"]) != set(expected_ids):
        raise ValueError("ALFRED_FILE_SERIES_SET")
    out = root / spec["audit_output_subdirectory"]
    out.mkdir(exist_ok=False)
    records = []
    for summary in receipt["series"]:
        sid = summary["series"]
        record = {"series": sid, "predictor_admitted": False, "files": []}
        try:
            chunks = [x for x in receipt["files"] if x["series"] == sid]
            if [x["chunk"] for x in chunks] != list(range(summary["chunks"])):
                raise ValueError("ALFRED_CHUNK_SET")
            rows, all_dates = [], []
            metadata = set()
            for chunk in chunks:
                path = root / chunk["path"]
                if path.parent != root or sha(path) != chunk["sha256"]:
                    raise ValueError("ALFRED_ZIP_HASH_OR_PATH")
                if path.stat().st_size != chunk["bytes"] or chunk["bytes"] > spec["maximum_download_bytes"]:
                    raise ValueError("ALFRED_ZIP_SIZE")
                with zipfile.ZipFile(path) as archive:
                    members = archive.infolist()
                    if len(members) != 2 or set(archive.namelist()) != {"README.txt", "obs._by_real-time_period.csv"}:
                        raise ValueError("ALFRED_MEMBER_SET")
                    if sum(m.file_size for m in members) > spec["maximum_uncompressed_bytes"]:
                        raise ValueError("ALFRED_UNCOMPRESSED_SIZE")
                    if archive.testzip() is not None:
                        raise ValueError("ALFRED_CRC")
                    data = {m.filename: archive.read(m) for m in members}
                text = data["README.txt"].decode("utf-8")
                if f"Series ID: {sid}\n" not in text or "Output Format: Observations by Real-Time Period\n" not in text:
                    raise ValueError("ALFRED_ID_OR_FORMAT")
                units = text.split("\nUnits\n", 1)[1].split("\nFrequency\n", 1)[0].strip()
                frequency = text.split("\nFrequency\n", 1)[1].split("\nSeasonal Adjustment\n", 1)[0].strip()
                unit_labels = [re.split(r"\s{2,}", line.strip())[0] for line in units.splitlines() if line.strip()]
                if unit_labels not in spec["expected_unit_label_variants"][sid]:
                    raise ValueError("ALFRED_UNITS")
                if [re.split(r"\s{2,}", line.strip())[0] for line in frequency.splitlines() if line.strip()] != spec["expected_frequency_labels"][sid]:
                    raise ValueError("ALFRED_FREQUENCY")
                # Preserve exact dated units metadata rather than flattening unit revisions.
                metadata.add((units, frequency))
                dates = re.findall(r"^\d{4}-\d{2}-\d{2}$", text.split("Vintage Dates Specified:", 1)[1], re.M)
                dates_digest = hashlib.sha256(("\n".join(dates) + "\n").encode()).hexdigest()
                if (dates_digest != chunk["vintage_sha256"] or len(dates) != chunk["count"]
                        or len(dates) > spec["observed_source_limit"]["daily_vintages_per_request"]
                        or dates[0] != chunk["first"] or dates[-1] != chunk["last"]):
                    raise ValueError("ALFRED_VINTAGE_SELECTION")
                all_dates.extend(dates)
                reader = csv.reader(io.StringIO(data["obs._by_real-time_period.csv"].decode("utf-8")))
                if next(reader) != ["period_start_date", sid, "realtime_start_date", "realtime_end_date"]:
                    raise ValueError("ALFRED_CSV_SCHEMA")
                chunk_rows = []
                for row in reader:
                    if len(row) != 4:
                        raise ValueError("ALFRED_CSV_WIDTH")
                    obs, value, start, end = row
                    if not (spec["observation_start"] <= obs <= spec["observation_end"]):
                        raise ValueError("ALFRED_OBSERVATION_BOUNDS")
                    if not (spec["vintage_start"] <= start <= dates[-1]):
                        raise ValueError("ALFRED_REALTIME_BOUNDS")
                    chunk_rows.append(tuple(row))
                within = alfred_version_summary(chunk_rows)
                if within["identical_chunk_duplicates"]:
                    raise ValueError("ALFRED_DUPLICATE_WITHIN_CHUNK")
                rows.extend(chunk_rows)
                record["files"].append({
                    "path": str(path), "sha256": chunk["sha256"], "rows": len(chunk_rows),
                    "members": {name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                                for name, raw in data.items()},
                })
            full_hash = hashlib.sha256(("\n".join(all_dates) + "\n").encode()).hexdigest()
            if (all_dates != sorted(set(all_dates)) or len(all_dates) != summary["count"]
                    or full_hash != summary["vintage_sha256"]
                    or all_dates[0] != summary["first"] or all_dates[-1] != summary["last"]
                    or all_dates[0] < spec["vintage_start"] or all_dates[-1] > spec["vintage_end"]):
                raise ValueError("ALFRED_FULL_VINTAGE_COVERAGE")
            record.update(alfred_version_summary(rows))
            record.update(status="ARCHIVE_CONSISTENT_NOT_PREDICTOR_ADMITTED",
                          selected_vintages=len(all_dates), first_vintage=all_dates[0],
                          last_vintage=all_dates[-1], vintage_sha256=full_hash,
                          source_metadata=[{"units": u, "frequency": f} for u, f in sorted(metadata)])
        except Exception as error:
            record.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        records.append(record)
    result = {
        "status": "COMPLETE_SOURCE_ARCHIVE_AUDIT", "records": records,
        "all_archives_consistent": all(r["status"] == "ARCHIVE_CONSISTENT_NOT_PREDICTOR_ADMITTED" for r in records),
        "manifest_sha256": sha(spec_path), "transport_receipt_sha256": receipt_sha256,
        "owner_sha256": sha(Path(__file__)), "source_commit": git("rev-parse", "HEAD"),
        "availability_policy": spec["availability"], "predictors_admitted": [],
        "fits_run": False, "market_outcomes_read": False, "test_accessed": False,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out / "RESULT.json", result)
    write_json(out / "TERMINAL.json", {"status": result["status"], "result_sha256": sha(out / "RESULT.json"),
                                     "all_archives_consistent": result["all_archives_consistent"]})
    return {"out": str(out), "result_sha256": sha(out / "RESULT.json"),
            "all_archives_consistent": result["all_archives_consistent"],
            "series_status": {r["series"]: r["status"] for r in records}}



def alfred_asof_levels(rows: list[tuple[str, str, str, str]], clocks: pd.DataFrame) -> pd.DataFrame:
    """Date-vintage publication bound + one whole canonical D1; no target access."""
    alfred_version_summary(rows)
    opens = pd.DatetimeIndex(clocks.session_open)
    decisions = pd.DatetimeIndex(clocks.decision_time)
    if (not len(opens) or opens.tz is None or decisions.tz is None or opens.has_duplicates
            or decisions.has_duplicates or not opens.is_monotonic_increasing
            or not decisions.is_monotonic_increasing
            or not (decisions == opens + TRADING_SESSION_DURATION).all()):
        raise RuntimeError("TA_B_CANONICAL_D1_CLOCK")
    # realtime_end is checked for source integrity, never used to retire a value early.
    versions = sorted(set(rows), key=lambda row: (row[2], row[0]))
    ready = []
    for obs, value, start, _end in versions:
        if obs > start:
            raise RuntimeError("TA_B_FUTURE_OBSERVATION")
        published_bound = (pd.Timestamp(start).tz_localize("America/New_York")
                           + pd.DateOffset(days=1) - pd.Timedelta(nanoseconds=1)).tz_convert("UTC")
        slot = int(opens.searchsorted(published_bound, side="left"))
        if slot < len(opens):
            ready.append((slot, obs, value, start, published_bound))
    state, index, result = {}, 0, []
    for slot in range(len(opens)):
        while index < len(ready) and ready[index][0] <= slot:
            _, obs, value, start, bound = ready[index]
            # Missing-value revisions remove that observation from the numeric state.
            state[obs] = (float(value) if value else np.nan, start, bound)
            index += 1
        numeric = [obs for obs, value in state.items() if np.isfinite(value[0])]
        if numeric:
            selected = max(numeric)
            value, start, bound = state[selected]
            result.append({"value": value, "observation_date": selected,
                           "realtime_start_date": start, "publication_upper_bound": bound,
                           "first_permitted_decision": decisions[int(opens.searchsorted(bound, side="left"))]})
        else:
            result.append({"value": np.nan, "observation_date": None,
                           "realtime_start_date": None, "publication_upper_bound": pd.NaT,
                           "first_permitted_decision": pd.NaT})
    return pd.DataFrame(result, index=clocks.index)


def prepare_b_macros(spec: dict, spec_path: Path) -> dict:
    """Prepare four validated B components; GLD/COT absence keeps full B closed."""
    out = Path(spec["output_directory"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "STARTED.json", {"source_commit": git("rev-parse", "HEAD"),
                                     "manifest_sha256": sha(spec_path)})
    try:
        audit_path = Path(spec["source_audit"]["path"])
        if sha(audit_path) != spec["source_audit"]["sha256"]:
            raise RuntimeError("TA_B_MACRO_AUDIT_HASH")
        audit = json.loads(audit_path.read_text())
        if not audit["all_archives_consistent"]:
            raise RuntimeError("TA_B_MACRO_AUDIT_FAILED")
        if [r["series"] for r in audit["records"]] != spec["macro_series"]:
            raise RuntimeError("TA_B_MACRO_SERIES_SET")
        cache_path = Path(spec["clock_cache"]["path"])
        if sha(cache_path) != spec["clock_cache"]["sha256"]:
            raise RuntimeError("TA_B_CLOCK_CACHE_HASH")
        clocks = pd.read_parquet(cache_path, columns=["session_open", "decision_time"])
        clocks = clocks.loc[(clocks.decision_time >= pd.Timestamp(spec["read_start"]))
                            & (clocks.decision_time < pd.Timestamp(spec["read_end_exclusive"]))].reset_index(drop=True)
        panel = clocks.copy()
        coverage, provenance = {}, {}
        for record in audit["records"]:
            sid = record["series"]
            rows = []
            for entry in record["files"]:
                path = Path(entry["path"])
                if sha(path) != entry["sha256"]:
                    raise RuntimeError("TA_B_MACRO_ZIP_HASH")
                with zipfile.ZipFile(path) as archive:
                    raw = archive.read("obs._by_real-time_period.csv")
                if hashlib.sha256(raw).hexdigest() != entry["members"]["obs._by_real-time_period.csv"]["sha256"]:
                    raise RuntimeError("TA_B_MACRO_MEMBER_HASH")
                reader = csv.reader(io.StringIO(raw.decode("utf-8")))
                if next(reader) != ["period_start_date", sid, "realtime_start_date", "realtime_end_date"]:
                    raise RuntimeError("TA_B_MACRO_SCHEMA")
                rows.extend(tuple(row) for row in reader)
            known = alfred_asof_levels(rows, clocks)
            level = known.value
            if spec["transforms"][sid] == "log":
                if (level.dropna() <= 0).any():
                    raise RuntimeError("TA_B_NONPOSITIVE_LOG_INPUT")
                level = np.log(level)
            elif spec["transforms"][sid] != "level":
                raise RuntimeError("TA_B_UNDECLARED_TRANSFORM")
            names = spec["feature_names"][sid]
            panel[names[0]], panel[names[1]] = level, level - level.shift(spec["change_d1"])
            for column in ["observation_date", "realtime_start_date", "publication_upper_bound", "first_permitted_decision"]:
                panel[f"{sid}__{column}"] = known[column]
            finite = panel[names].notna().all(axis=1)
            permitted = known.value.notna()
            if not (known.loc[permitted, "first_permitted_decision"].array <= clocks.loc[permitted, "decision_time"].array).all():
                raise RuntimeError("TA_B_INPUT_BEFORE_PERMITTED_DECISION")
            coverage[sid] = {"feature_complete_rows": int(finite.sum()),
                             "numeric_level_rows": int(permitted.sum()),
                             "distinct_used_observation_dates": int(known.observation_date.nunique()),
                             "distinct_used_realtime_dates": int(known.realtime_start_date.nunique())}
            if finite.any():
                coverage[sid].update(first_complete_decision=str(clocks.loc[finite, "decision_time"].iloc[0]),
                                     last_complete_decision=str(clocks.loc[finite, "decision_time"].iloc[-1]))
            provenance[sid] = {"archive_files": len(record["files"]), "source_rows": len(rows)}
        names = [name for sid in spec["macro_series"] for name in spec["feature_names"][sid]]
        common = panel[names].notna().all(axis=1)
        panel.to_parquet(out / "MACRO_COMPONENTS.parquet", index=False)
        result = {"status": "COMPLETE_FOUR_MACRO_COMPONENTS_ONLY", "source_commit": git("rev-parse", "HEAD"),
                  "manifest_sha256": sha(spec_path), "source_audit": spec["source_audit"],
                  "clock_cache": spec["clock_cache"], "clock_columns_read": ["session_open", "decision_time"],
                  "clock_rows": len(clocks), "coverage": coverage, "provenance": provenance,
                  "four_source_complete_rows": int(common.sum()), "full_b_admitted": False,
                  "missing_full_b_sources": spec["missing_full_b_sources"], "feature_names": names,
                  "fits_run": False, "new_market_outcomes_read": False, "test_accessed": False,
                  "artifact": {"path": str(out / "MACRO_COMPONENTS.parquet"),
                               "sha256": sha(out / "MACRO_COMPONENTS.parquet")}}
        if common.any():
            result["four_source_first_complete_decision"] = str(clocks.loc[common, "decision_time"].iloc[0])
            result["four_source_last_complete_decision"] = str(clocks.loc[common, "decision_time"].iloc[-1])
        write_json(out / "RESULT.json", result)
        write_json(out / "TERMINAL.json", {"status": result["status"], "result_sha256": sha(out / "RESULT.json")})
        return {"out": str(out), "result_sha256": sha(out / "RESULT.json"),
                "four_source_complete_rows": int(common.sum()), "full_b_admitted": False}
    except Exception as error:
        write_json(out / "TERMINAL.json", {"status": "FAILED", "error": str(error)})
        raise


def funding_dates(spec: dict) -> pd.DatetimeIndex:
    dates = pd.date_range(spec["start_date"], spec["end_date"], freq="D", tz="UTC")
    if spec["series"] == "DFF":
        return dates
    if spec["series"] != "EFFR":
        raise RuntimeError("TA_FUNDING_SERIES")
    # Federal Reserve Banks observe Sunday holidays on Monday; Saturday stays
    # Saturday (unlike federal-government offices). Juneteenth starts in 2021.
    calendar = USFederalHolidayCalendar()
    calendar.rules = deepcopy(calendar.rules)
    for rule in calendar.rules:
        if rule.observance is nearest_workday:
            rule.observance = sunday_to_monday
    holidays = calendar.holidays(dates[0], dates[-1])
    return dates[(dates.dayofweek < 5) & ~dates.isin(holidays)]


def parse_funding(raw: bytes, spec: dict) -> tuple[pd.DatetimeIndex, np.ndarray]:
    if spec["series"] == "DFF":
        frame = pd.read_csv(io.BytesIO(raw))
        if list(frame.columns) != ["observation_date", "DFF"]:
            raise RuntimeError(f"TA_FUNDING_SCHEMA: {list(frame.columns)}")
        dates = pd.DatetimeIndex(pd.to_datetime(frame.observation_date, utc=True))
        rates = pd.to_numeric(frame.DFF, errors="raise").to_numpy(float) / 100.0
    elif spec["series"] == "EFFR":
        frame = pd.DataFrame(json.loads(raw)["refRates"]).sort_values("effectiveDate")
        if frame.empty or not (frame["type"] == "EFFR").all():
            raise RuntimeError("TA_FUNDING_SCHEMA")
        dates = pd.DatetimeIndex(pd.to_datetime(frame.effectiveDate, utc=True))
        rates = pd.to_numeric(frame[spec["rate_field"]], errors="raise").to_numpy(float) / 100.0
    else:
        raise RuntimeError("TA_FUNDING_SERIES")
    expected = funding_dates(spec)
    if not dates.equals(expected) or not np.isfinite(rates).all():
        raise RuntimeError("TA_FUNDING_COVERAGE: " + json.dumps({
            "missing": [str(d) for d in expected.difference(dates)],
            "extra": [str(d) for d in dates.difference(expected)],
            "duplicates": bool(dates.has_duplicates), "nonfinite": bool(~np.isfinite(rates).all()),
        }))
    return dates, rates


def fetch_funding(spec: dict, spec_path: Path) -> dict:
    out = Path(spec["output_directory"])
    out.mkdir(parents=True, exist_ok=False)
    request = Request(spec["url"], headers={"User-Agent": "GX1 offline research"})
    try:
        if "reuse_download" in spec:
            raw_path = Path(spec["reuse_download"]["path"])
            if sha(raw_path) != spec["reuse_download"]["sha256"]:
                raise RuntimeError("TA_FUNDING_REUSED_BYTES_HASH")
            raw = raw_path.read_bytes()
            final_url = spec["url"]
        else:
            with urlopen(request, timeout=spec.get("timeout_seconds", 30)) as response:
                raw = response.read(spec["maximum_bytes"] + 1)
                final_url = response.url
            raw_path = out / ("EFFR.json" if spec["series"] == "EFFR" else "DFF.csv")
            raw_path.write_bytes(raw)
        if len(raw) > spec["maximum_bytes"]:
            raise RuntimeError("TA_FUNDING_RESPONSE_TOO_LARGE")
        dates, _ = parse_funding(raw, spec)
        receipt = {"status": "COMPLETE", "fetched_utc": datetime.now(timezone.utc).isoformat(),
                   "manifest": str(spec_path), "manifest_sha256": sha(spec_path),
                   "raw_path": str(raw_path), "raw_sha256": sha(raw_path), "rows": len(dates),
                   "first_date": str(dates[0]), "last_date": str(dates[-1]),
                   "final_url": final_url, "purpose": "cost_proxy_only_not_predictor"}
        write_json(out / "RECEIPT.json", receipt)
        return receipt
    except Exception as exc:
        write_json(out / "FAILED.json", {"error": str(exc), "status": "FAILED"})
        raise


def load_funding(spec: dict) -> ResearchFinancingCurve:
    fetch_manifest = Path(spec["funding"]["manifest"])
    if sha(fetch_manifest) != spec["funding"]["manifest_sha256"]:
        raise RuntimeError("TA_FUNDING_MANIFEST_HASH")
    fetch = json.loads(fetch_manifest.read_text())
    receipt = json.loads((Path(fetch["output_directory"]) / "RECEIPT.json").read_text())
    if receipt["status"] != "COMPLETE" or receipt["manifest_sha256"] != sha(fetch_manifest):
        raise RuntimeError("TA_FUNDING_RECEIPT")
    raw_path = Path(receipt["raw_path"])
    if sha(raw_path) != receipt["raw_sha256"]:
        raise RuntimeError("TA_FUNDING_BYTES_HASH")
    dates, rates = parse_funding(raw_path.read_bytes(), fetch)
    return ResearchFinancingCurve(
        effective_at=dates,
        benchmark_annual_rate=rates,
        coverage_end=pd.Timestamp(fetch["end_date"], tz="UTC") + pd.Timedelta(days=1),
        broker_markup=spec["funding"]["markup"], seconds_per_year=spec["funding"]["seconds_per_year"],
    )


def load_market(spec: dict, *, columns: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    root = Path(spec["tape"]["root"])
    if sha(root / "MANIFEST.json") != spec["tape"]["manifest_sha256"]:
        raise RuntimeError("TA_TAPE_MANIFEST_HASH")
    manifest = json.loads((root / "MANIFEST.json").read_text())
    if (manifest["timestamp_semantics"] != "bar_start_utc"
            or manifest["decision_available_offset_seconds"] != int(M5_BAR_DURATION.total_seconds())
            or manifest["market_closure_contract"] != "oanda_complete_true_source_absence_no_synthesis_v1"):
        raise RuntimeError("TA_TAPE_CLOCK")
    start, end = pd.Timestamp(spec["read_start"]), pd.Timestamp(spec["read_end_exclusive"])
    frames, bindings = [], []
    # Construct only authorized year paths; do not enumerate or stat TEST-year files.
    for year in range(start.year, (end - pd.Timedelta(nanoseconds=1)).year + 1):
        path = root / f"year={year}" / "part-000.parquet"
        digest = sha(path)
        if digest != manifest["year_sha256"][f"year={year}"]:
            raise RuntimeError(f"TA_TAPE_YEAR_HASH: {year}")
        frame = pq.read_table(
            path, columns=columns if columns is not None else ["time", "open", "high", "low", "close", "bid_open", "ask_open"],
            filters=[("time", ">=", start.to_pydatetime()), ("time", "<", end.to_pydatetime())],
        ).to_pandas()
        frames.append(frame)
        bindings.append({"path": str(path), "sha256": digest, "rows": len(frame)})
    frame = pd.concat(frames, ignore_index=True).set_index("time").sort_index()
    if frame.index.has_duplicates or not (frame.index < end).all() or not np.isfinite(frame.to_numpy()).all():
        raise RuntimeError("TA_MARKET_ROWS_INVALID")
    return frame, {"parts": bindings, "test_accessed": False}


def daily_panel(frame: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    daily = _resample_ohlc_for_model_native_scalars(frame, "D1")
    closed = daily.index + TRADING_SESSION_DURATION
    fill_index = frame.index.searchsorted(closed, side="left")
    valid = (closed < end) & (fill_index < len(frame))
    daily = daily.loc[valid].copy()
    fill_index = fill_index[valid]
    daily["decision_time"] = closed[valid]
    daily["fill_time"] = frame.index[fill_index]
    for dest, origin in [("fill_mid", "open"), ("fill_bid", "bid_open"), ("fill_ask", "ask_open")]:
        daily[dest] = frame[origin].to_numpy()[fill_index]
    if daily.fill_time.duplicated().any() or np.any(daily.fill_time < daily.decision_time):
        raise RuntimeError("TA_DAILY_FILL_CLOCK")
    a14 = wilder_atr(daily.high, daily.low, daily.close, 14)
    a252 = wilder_atr(daily.high, daily.low, daily.close, 252)
    positive = a14.where(a14 > 0)
    for n in (21, 63, 126, 252):
        daily[f"momentum_{n}_atr14"] = (daily.close - daily.close.shift(n)) / positive
    low, high = daily.low.rolling(252).min(), daily.high.rolling(252).max()
    daily["range_position_252"] = (daily.close - low) / (high - low).where(high > low)
    daily["atr14_over_atr252"] = a14 / a252.where(a252 > 0)
    daily["ema200_distance_atr14"] = (daily.close - classic_ema(daily.close, 200)) / positive
    daily["atr14"] = a14
    return daily.reset_index(names="session_open")


def join_b_components(panel: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    """Exact clock join only; does not qualify source versions or admit a B run."""
    if (panel.columns.has_duplicates or components.columns.has_duplicates
            or not set(FEATURES).issubset(panel)
            or not set(B_MACRO_FEATURES).issubset(components)
            or set(B_MACRO_FEATURES).intersection(panel)):
        raise RuntimeError("TA_B_FEATURE_CONTRACT")
    for frame in (panel, components):
        opens, decisions = pd.DatetimeIndex(frame.session_open), pd.DatetimeIndex(frame.decision_time)
        if (not len(opens) or opens.tz is None or decisions.tz is None or opens.has_duplicates
                or not opens.is_monotonic_increasing
                or not (decisions == opens + TRADING_SESSION_DURATION).all()):
            raise RuntimeError("TA_B_CANONICAL_D1_CLOCK")
    if (len(panel) != len(components)
            or not np.array_equal(panel.session_open.array, components.session_open.array)
            or not np.array_equal(panel.decision_time.array, components.decision_time.array)):
        raise RuntimeError("TA_B_COMPONENT_CLOCK_MISMATCH")
    result = panel.copy()
    result[B_MACRO_FEATURES] = components[B_MACRO_FEATURES].to_numpy(float)
    if np.isinf(result[B_FEATURES].to_numpy(float)).any():
        raise RuntimeError("TA_B_NONFINITE_FEATURE")
    return result


def fit_predictions(panel: pd.DataFrame, spec: dict) -> tuple[dict, list[dict]]:
    # Preserve A's original feature/population contract.
    population = np.isfinite(panel[FEATURES].to_numpy(float)).all(axis=1)
    return _fit_predictions(panel, spec, FEATURES, population)


def _fit_predictions(panel: pd.DataFrame, spec: dict, feature_names: list[str],
                     population: np.ndarray) -> tuple[dict, list[dict]]:
    X = panel[feature_names].to_numpy(float)
    if (population.dtype != bool or population.shape != (len(panel),)
            or not np.isfinite(X[population]).all()):
        raise RuntimeError("TA_SHARED_FIT_POPULATION_INVALID")
    mid, atr = panel.fill_mid.to_numpy(), panel.atr14.to_numpy()
    times = pd.DatetimeIndex(panel.fill_time)
    n, max_h = len(panel), max(spec["horizons"])
    positions = np.arange(n)
    eligible = population & (positions + max_h < n)
    forecasts, fits = {}, []
    for horizon in spec["horizons"]:
        target = np.full(n, np.nan)
        target[:-horizon] = (mid[horizon:] - mid[:-horizon]) / atr[:-horizon]
        forecasts[horizon] = {name: np.full(n, np.nan) for name in ["ridge", "hgb", "constant"]}
        for year in spec["fold_years"]:
            start = pd.Timestamp(f"{year}-01-01", tz="UTC")
            end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
            past = np.flatnonzero(eligible & (times < start))
            fit = past[times[past + max_h] < start]
            hold = np.flatnonzero(eligible & (times >= start) & (times < end))
            row_binding = {
                "fit_positions_sha256": hashlib.sha256(fit.astype("<i8").tobytes()).hexdigest(),
                "hold_positions_sha256": hashlib.sha256(hold.astype("<i8").tobytes()).hexdigest(),
            }
            if len(fit) < spec["min_fit_rows"] or not len(hold):
                fits.append({"year": year, "horizon": horizon, "status": "INSUFFICIENT_CAUSAL_ROWS",
                             "fit_rows": len(fit), "hold_rows": len(hold), **row_binding})
                continue
            gram = RidgeGram(X[fit], X[hold], inner_fraction=spec["inner_fraction"],
                             fit_positions=fit, purge_bars=max_h, min_inner_rows=spec["min_inner_rows"],
                             constant_alternative=False)
            ridge, ri = gram.fit_predict(target[fit])
            hgb, hi = fit_hgb(
                X[fit], target[fit], X[hold], inner_fraction=spec["inner_fraction"],
                fit_positions=fit, purge_bars=max_h, min_inner_rows=spec["min_inner_rows"],
                max_iter=spec["hgb"]["max_iter"], seed=spec["seed"],
                learning_rate=spec["hgb"]["learning_rate"], min_samples_leaf=spec["hgb"]["min_samples_leaf"],
            )
            for name, pred in [("ridge", ridge), ("hgb", hgb),
                               ("constant", np.full(len(hold), target[fit].mean()))]:
                forecasts[horizon][name][hold] = pred
            fits.append({"year": year, "horizon": horizon, "status": "FIT",
                         "fit_rows": len(fit), "hold_rows": len(hold), **row_binding,
                         "last_fit_outcome_time": str(times[fit[-1] + max_h]),
                         "first_hold_time": str(times[hold[0]]), "ridge": ri, "hgb": hi})
            arm_name = "A" if feature_names == FEATURES else "B"
            print(f"[TA-{arm_name}] fitted year={year} horizon={horizon} rows={len(fit)}/{len(hold)}", flush=True)
    return forecasts, fits


def fit_matched_b(panel: pd.DataFrame, spec: dict) -> tuple[dict, list[dict]]:
    """Fit A and full B on shared rows; source admission remains a separate prerequisite."""
    if (spec["feature_names"] != B_FEATURES or spec["horizons"] != [20, 5]
            or spec["primary_horizon"] != 20 or panel.columns.has_duplicates
            or not set(B_FEATURES).issubset(panel)):
        raise RuntimeError("TA_B_FEATURE_CONTRACT")
    values = panel[B_FEATURES].to_numpy(float)
    if np.isinf(values).any():
        raise RuntimeError("TA_B_NONFINITE_FEATURE")
    # Keep the full clock. Filtering the panel would shorten horizons across missing inputs.
    population = np.isfinite(values).all(axis=1)
    a, a_fits = _fit_predictions(panel, spec, FEATURES, population)
    b, b_fits = _fit_predictions(panel, spec, B_FEATURES, population)
    matched = []
    if len(a_fits) != len(b_fits):
        raise RuntimeError("TA_B_FOLD_MISMATCH")
    for af, bf in zip(a_fits, b_fits):
        a_rows = {k: v for k, v in af.items() if k not in ("ridge", "hgb")}
        b_rows = {k: v for k, v in bf.items() if k not in ("ridge", "hgb")}
        if a_rows != b_rows:
            raise RuntimeError("TA_B_FOLD_MISMATCH")
        row = dict(a_rows)
        if af["status"] == "FIT":
            row.update(a={"ridge": af["ridge"], "hgb": af["hgb"]},
                       b={"ridge": bf["ridge"], "hgb": bf["hgb"]})
        matched.append(row)
    forecasts = {}
    for horizon in spec["horizons"]:
        if not np.array_equal(a[horizon]["constant"], b[horizon]["constant"], equal_nan=True):
            raise RuntimeError("TA_B_CONSTANT_MISMATCH")
        forecasts[horizon] = {
            "a_ridge": a[horizon]["ridge"], "a_hgb": a[horizon]["hgb"],
            "b_ridge": b[horizon]["ridge"], "b_hgb": b[horizon]["hgb"],
            "constant": a[horizon]["constant"],
        }
        masks = [np.isfinite(v) for v in forecasts[horizon].values()]
        if any(not np.array_equal(masks[0], m) for m in masks[1:]):
            raise RuntimeError("TA_B_FORECAST_POPULATION_MISMATCH")
    return forecasts, matched


def statistics(values: dict[str, np.ndarray], comparisons: list[dict],
               normalizer: np.ndarray, annualization: float, index: np.ndarray) -> np.ndarray:
    out = []
    for c in comparisons:
        a, b = values[c["model"]][index], values[c["baseline"]][index]
        delta = a - b
        if c["metric"] == "mean_delta_bps":
            value = delta.mean()
        elif c["metric"] == "normalized_delta":
            value = (delta / normalizer[index]).mean()
        elif c["metric"] == "sharpe_delta":
            sa, sb = a.std(ddof=1), b.std(ddof=1)
            value = (a.mean() / sa - b.mean() / sb) * np.sqrt(annualization) if sa > 0 and sb > 0 else np.nan
        else:
            raise RuntimeError("TA_METRIC")
        out.append(value)
    return np.asarray(out)


def evaluate(panel: pd.DataFrame, forecasts: dict, curve: ResearchFinancingCurve, spec: dict, out: Path) -> dict:
    return _evaluate(panel, forecasts, curve, spec, out,
                     {"ridge": ["long", "constant", "trend", "buy_hold"],
                      "hgb": ["long", "constant", "trend", "buy_hold"]})


def evaluate_matched_b(panel: pd.DataFrame, forecasts: dict, curve: ResearchFinancingCurve,
                       spec: dict, out: Path) -> dict:
    """One paired portfolio/inference family, including each B learner minus its matched A."""
    if (spec["feature_names"] != B_FEATURES or spec["horizons"] != [20, 5]
            or spec["primary_horizon"] != 20 or set(forecasts) != set(spec["horizons"])):
        raise RuntimeError("TA_B_EVALUATION_CONTRACT")
    masks = []
    for arm in forecasts.values():
        if set(arm) != {"a_ridge", "a_hgb", "b_ridge", "b_hgb", "constant"}:
            raise RuntimeError("TA_B_EVALUATION_ARMS")
        for values in arm.values():
            if np.asarray(values).shape != (len(panel),) or np.isinf(values).any():
                raise RuntimeError("TA_B_FORECAST_POPULATION_MISMATCH")
            masks.append(np.isfinite(values))
    if any(not np.array_equal(masks[0], m) for m in masks[1:]):
        raise RuntimeError("TA_B_FORECAST_POPULATION_MISMATCH")
    return _evaluate(panel, forecasts, curve, spec, out,
                     {"b_ridge": ["a_ridge", "long", "constant", "trend", "buy_hold"],
                      "b_hgb": ["a_hgb", "long", "constant", "trend", "buy_hold"]})


def _evaluate(panel: pd.DataFrame, forecasts: dict, curve: ResearchFinancingCurve,
              spec: dict, out: Path, learner_baselines: dict[str, list[str]]) -> dict:
    common = np.logical_and.reduce([np.isfinite(v) for arm in forecasts.values() for v in arm.values()])
    rows = np.flatnonzero(common)
    if len(rows) < 2 or np.any(np.diff(rows) != 1):
        raise RuntimeError("TA_COMMON_EVALUATION_POPULATION_INVALID")
    sample = panel.iloc[rows]
    time = pd.DatetimeIndex(sample.fill_time)
    tape = Tape(time, sample.fill_mid.to_numpy(), sample.fill_bid.to_numpy(),
                sample.fill_ask.to_numpy(), spec["tape"]["manifest_sha256"], spec["tape"]["root"])
    full_mid = panel.fill_mid.to_numpy()
    sigma = pd.Series(full_mid).pct_change(fill_method=None).rolling(spec["risk"]["lookback"]).std(ddof=1).to_numpy()
    normalizer = sigma[rows[:-1]] * 1e4
    if not np.isfinite(normalizer).all() or np.any(normalizer <= 0):
        raise RuntimeError("TA_NORMALIZER")
    values, summaries = {}, {}
    for horizon, arm in forecasts.items():
        signals = {name: np.sign(pred) for name, pred in arm.items()}
        signals["trend"] = np.sign(np.sign(panel[FEATURES[:4]].to_numpy()).mean(axis=1))
        signals["long"] = np.ones(len(panel))
        signals["buy_hold"] = np.ones(len(panel))
        for name, sides in signals.items():
            if name == "buy_hold":
                units = np.full(len(rows) - 1, spec["initial_equity"] / tape.mid[0])
            else:
                units = causal_risk_units(
                    full_mid, sides, lookback=spec["risk"]["lookback"],
                    periods_per_year=spec["periods_per_year"], target_annual_vol=spec["risk"]["target_annual_vol"],
                    max_gross_leverage=spec["risk"]["max_gross_leverage"],
                    initial_equity=spec["initial_equity"],
                )[rows[:-1]]
            for financing in ["historical_proxy", "zero"]:
                key = f"h{horizon}:{name}:{financing}"
                path = portfolio_path(
                    tape, units, initial_equity=spec["initial_equity"],
                    slippage_bps_per_execution=spec["slippage_bps_per_execution"],
                    commission_bps_per_execution=spec["commission_bps_per_execution"],
                    financing_curve=curve if financing == "historical_proxy" else None, liquidate_at_end=True,
                )
                path.to_parquet(out / (key.replace(":", "_") + ".parquet"), index=False)
                summaries[key] = portfolio_summary(path, periods_per_year=spec["periods_per_year"])
                values[key] = portfolio_period_returns(path) * 1e4
    comparisons = []
    for horizon in spec["horizons"]:
        for learner, baselines in learner_baselines.items():
            for funding in ["historical_proxy", "zero"]:
                for baseline in baselines:
                    for metric in spec["effects"]:
                        comparisons.append({
                            "name": f"h{horizon}:{learner}:{funding}:vs_{baseline}:{metric}",
                            "model": f"h{horizon}:{learner}:{funding}",
                            "baseline": f"h{horizon}:{baseline}:{funding}", "metric": metric,
                        })
    n = len(normalizer)
    point = statistics(values, comparisons, normalizer, spec["periods_per_year"], np.arange(n))
    boot = np.empty((spec["bootstrap_draws"], len(comparisons)))
    for b, index in enumerate(stationary_bootstrap_indices(
        n, draws=spec["bootstrap_draws"], mean_block_length=spec["mean_block_length"], seed=spec["seed"],
    )):
        boot[b] = statistics(values, comparisons, normalizer, spec["periods_per_year"], index)
    # Undefined Sharpe/zero bootstrap variation remains an explicit untestable member.
    eligible = np.isfinite(point) & np.isfinite(boot).all(axis=0) & (boot.std(axis=0, ddof=1) > 0)
    inference = {}
    if eligible.any():
        chosen = [c for c, ok in zip(comparisons, eligible) if ok]
        inference = max_t_inference(
            point[eligible], boot[:, eligible], names=[c["name"] for c in chosen],
            alpha=spec["alpha"], desired_power=spec["desired_power"],
            effect_sizes=np.array([spec["effects"][c["metric"]] for c in chosen]),
            minimum_relevant_effect=np.array([spec["effects"][c["metric"]][0] for c in chosen]),
        )
    endpoints = {row["name"]: row for row in inference.get("endpoints", [])}
    untestable = []
    for c, ok in zip(comparisons, eligible):
        if not ok:
            untestable.append(c["name"])
            endpoints[c["name"]] = {"name": c["name"], "effect_verdict": "INKONKLUSIV",
                                    "reason": "undefined_statistic_or_zero_bootstrap_variation"}
    decisions = {}
    for learner in learner_baselines:
        required = [c["name"] for c in comparisons if
                    c["name"].startswith(f"h{spec['primary_horizon']}:{learner}:")
                    and ":vs_buy_hold:" not in c["name"]]
        verdicts = [endpoints[name]["effect_verdict"] for name in required]
        positive_net = all(summaries[f"h{spec['primary_horizon']}:{learner}:{f}"]["total_net_bps"] > 0
                           for f in ["historical_proxy", "zero"])
        decision = ("GO" if all(v == "GO" for v in verdicts) and positive_net else
                    "NO_GO" if "NO_GO" in verdicts else "INKONKLUSIV")
        decisions[learner] = {"decision": decision, "required_endpoints": required,
                             "positive_net_both_funding_scenarios": positive_net}
    yearly = {}
    for year in sorted(set(time[1:].year)):
        index = np.flatnonzero(time[1:].year == year)
        yearly[str(year)] = {key: {"intervals": len(index), "mean_net_bps": float(v[index].mean())}
                            for key, v in values.items()}
    pd.DataFrame({"time": time[1:], "gold_sigma_bps": normalizer, **values}).to_parquet(out / "PAIRED_RETURNS.parquet", index=False)
    np.savez_compressed(out / "BOOTSTRAP_STATISTICS.npz", estimates=point, bootstrap_estimates=boot)
    return {"evaluation": {"first_fill": str(time[0]), "last_fill": str(time[-1]), "intervals": n,
                           "reused_development_history": True},
            "portfolios": summaries, "per_year": yearly, "inference": inference,
            "declared_family": [c["name"] for c in comparisons], "untestable_endpoints": untestable,
            "endpoints": list(endpoints.values()), "decisions": decisions}


def run_a(spec: dict, spec_path: Path) -> dict:
    if spec["horizons"] != [20, 5] or spec["feature_names"] != FEATURES:
        raise RuntimeError("TA_PREREG_SCOPE")
    out = Path(spec["output_directory"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "STARTED.json", {"git_head": git("rev-parse", "HEAD"),
                                     "preregistration": str(spec_path), "preregistration_sha256": sha(spec_path)})
    try:
        curve = load_funding(spec)
        market, binding = load_market(spec)
        panel = daily_panel(market, pd.Timestamp(spec["read_end_exclusive"]))
        panel.to_parquet(out / "DAILY_PANEL.parquet", index=False)
        forecasts, fits = fit_predictions(panel, spec)
        write_json(out / "FITS.json", {"fits": fits})
        pd.DataFrame({"fill_time": panel.fill_time, **{
            f"h{h}:{name}": pred for h, arm in forecasts.items() for name, pred in arm.items()
        }}).to_parquet(out / "PREDICTIONS.parquet", index=False)
        result = evaluate(panel, forecasts, curve, spec, out)
        result["artifacts"] = {
            path.name: {"sha256": sha(path), "bytes": path.stat().st_size}
            for path in sorted(out.iterdir()) if path.is_file()
        }
        result.update(schema="gx1_ta_measurement_a_v1", git_head=git("rev-parse", "HEAD"),
                      preregistration_sha256=sha(spec_path), inputs=binding,
                      primary_horizon=spec["primary_horizon"], test_accessed=False,
                      native_training=False, evidence_class="measured_reused_development_walkforward")
        write_json(out / "RESULT.json", result)
        write_json(out / "TERMINAL.json", {"status": "COMPLETE", "result_sha256": sha(out / "RESULT.json"),
                                         "finished_utc": datetime.now(timezone.utc).isoformat()})
        return {"status": "COMPLETE", "out": str(out), "decisions": result["decisions"]}
    except Exception as exc:
        write_json(out / "TERMINAL.json", {"status": "FAILED", "error": str(exc),
                                         "finished_utc": datetime.now(timezone.utc).isoformat()})
        raise


# C is one frozen composite. The source cell catalogue is not a new search grid.
C_CELLS = (
    "rn50_cross_follow_h12", "setup_pdh_break_trend_H4_pair_h12",
    "setup_momentum_confluence_long_pair_h12",
    "setup_range_break_up_H1_trend_H4_pair_h12", "orb_london_local",
)


def c_signal_panel(market: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, dict]:
    from gx1.scripts.research_intraday_mechanisms_v1 import (
        PRIMITIVE_PARAMS, SETUP_COLUMNS, build_primitives, setup_pair_sides,
        round_number_sides, local_weekdays, local_instant, LONDON, LONDON_OPEN,
        LONDON_ORB_EXIT, ORB_RANGE_BARS,
    )
    bar = M5_BAR_DURATION
    start, end = pd.Timestamp(spec["evaluation_start"]), pd.Timestamp(spec["read_end_exclusive"])
    time = market.index
    chosen = (time + bar >= start) & (time + bar < end)
    decision_rows = time[chosen]
    primitives, _ = build_primitives(market.reset_index(names="time"), decision_rows, PRIMITIVE_PARAMS, keep_columns=SETUP_COLUMNS)
    pairs = setup_pair_sides(primitives)
    signal = pd.DataFrame(index=decision_rows)
    signal["known_at"] = decision_rows + bar
    cross, _ = round_number_sides(market.close.to_numpy(), market.high.to_numpy(),
                                  market.low.to_numpy(), spec["round_grid_usd"])
    signal[C_CELLS[0]] = cross[chosen]
    for cell, pair in zip(C_CELLS[1:4], ["pdh_break_trend_H4", "momentum_confluence_long",
                                      "range_break_up_H1_trend_H4"]):
        signal[cell] = pairs[pair]
    # Preserve the original prior-bar continuity requirement; never filter on future gaps.
    prior_contiguous = np.concatenate([[False], np.diff(time.asi8) == bar.value])
    signal.loc[~prior_contiguous[chosen], list(C_CELLS[:4])] = 0
    signal[C_CELLS[4]] = 0
    signal["orb_exit"] = pd.Series(pd.NaT, index=signal.index, dtype="datetime64[ns, UTC]")
    for day in local_weekdays(LONDON, start, end):
        opening = local_instant(day, LONDON_OPEN, LONDON)
        range_end, target = opening + ORB_RANGE_BARS * bar, local_instant(day, LONDON_ORB_EXIT, LONDON)
        expected = pd.date_range(opening, periods=ORB_RANGE_BARS, freq=bar)
        if not expected.isin(time).all():
            continue
        levels = market.loc[expected, "close"]  # original range is close extrema, not high/low
        candidates = signal.index[(signal.index >= range_end) & (signal.known_at < target)]
        for t in candidates:
            price = market.at[t, "close"]
            side = 1 if price > levels.max() else -1 if price < levels.min() else 0
            if side:
                signal.at[t, C_CELLS[4]], signal.at[t, "orb_exit"] = side, target
                break
    daily = _resample_ohlc_for_model_native_scalars(market, "D1")
    daily_units = causal_risk_units(
        daily.close.to_numpy(), np.ones(len(daily)), initial_equity=spec["initial_equity"],
        periods_per_year=spec["risk"]["periods_per_year"], lookback=spec["risk"]["lookback"],
        target_annual_vol=spec["risk"]["target_annual_vol"],
        max_gross_leverage=spec["risk"]["max_gross_leverage"],
    )
    scale = daily_units * daily.close.to_numpy() / spec["initial_equity"]
    atr_bps = wilder_atr(daily.high, daily.low, daily.close, 14).to_numpy() / daily.close.to_numpy() * 1e4
    closed = daily.index + TRADING_SESSION_DURATION
    ix = closed.searchsorted(pd.DatetimeIndex(signal.known_at), side="right") - 1
    if np.any(ix < 0):
        raise RuntimeError("TA_C_DAILY_WARMUP_CLOCK")
    signal["risk_scale"], signal["atr_bps"] = scale[ix], atr_bps[ix]
    if not np.isfinite(signal[["risk_scale", "atr_bps"]].to_numpy()).all() or (signal.atr_bps <= 0).any():
        raise RuntimeError("TA_C_RISK_WARMUP")
    return signal, {cell: {"long": int((signal[cell] > 0).sum()),
                          "short": int((signal[cell] < 0).sum())} for cell in C_CELLS}


def c_select(market: pd.DataFrame, signals: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, dict]:
    """One shared reservation cohort; fill outcomes never change later selection."""
    time, bar = market.index, M5_BAR_DURATION
    end = pd.Timestamp(spec["read_end_exclusive"])
    terminal_time = time[-1] + bar
    if terminal_time > end:
        raise RuntimeError("TA_C_TERMINAL_CLOCK")
    busy_until = pd.Timestamp(spec["evaluation_start"])
    records, counts = [], {"conflicting_rows": 0, "same_side_duplicates": 0, "overlap_rows": 0}
    for t, row in signals.iterrows():
        active = [(name, int(row[name])) for name in C_CELLS if row[name] != 0]
        if not active:
            continue
        if len({side for _, side in active}) > 1:
            counts["conflicting_rows"] += 1
            continue
        if row.known_at < busy_until:
            counts["overlap_rows"] += 1
            continue
        counts["same_side_duplicates"] += len(active) - 1
        cell, side = active[0]  # frozen catalogue order; no quality ranking
        target = row.orb_exit if cell == C_CELLS[-1] else row.known_at + spec["hold_bars"] * bar
        entry = int(time.searchsorted(row.known_at, side="left"))
        exit_ = int(time.searchsorted(target, side="left"))
        censored = exit_ == len(time)
        exit_time = terminal_time if censored else time[exit_]
        busy_until = max(target, exit_time)
        record = {"signal_bar_start": t, "known_at": row.known_at, "cell": cell, "side": side,
                  "target_time": target, "exit_time": exit_time, "censored_at_end": censored,
                  "risk_scale": row.risk_scale, "atr_bps": row.atr_bps,
                  "executable": False, "passive_placed": False, "passive_touched": False}
        if entry < len(time) and time[entry] < target and time[entry] < terminal_time:
            m = market.iloc[entry]
            close_row = market.iloc[-1] if censored else market.iloc[exit_]
            suffix = "close" if censored else "open"
            record.update(executable=True, entry_time=time[entry], decision_mid=float(m.open),
                          entry_bid=float(m.bid_open), entry_ask=float(m.ask_open),
                          exit_mid=float(close_row["close" if censored else "open"]),
                          exit_bid=float(close_row["bid_" + suffix]), exit_ask=float(close_row["ask_" + suffix]))
            window_end = time[entry] + bar
            # Place only if the whole declared one-bar window fits before the known target.
            if window_end < target and window_end <= terminal_time:
                limit = float(m.bid_open if side > 0 else m.ask_open)
                touched = bool(m.ask_low <= limit if side > 0 else m.bid_high >= limit)
                record.update(passive_placed=True, passive_touched=touched,
                              passive_limit=limit, passive_fill_time=window_end)
        records.append(record)
    if not records:
        raise RuntimeError("TA_C_NO_SELECTED_SIGNALS")
    return pd.DataFrame(records), counts


def c_quotes(market: pd.DataFrame, start: pd.Timestamp) -> Tape:
    """Actual open quotes plus actual closes at bar-end; opens win coincident timestamps."""
    opened = market[["open", "bid_open", "ask_open"]].copy()
    opened.columns = ["mid", "bid", "ask"]
    closed = market[["close", "bid_close", "ask_close"]].copy()
    closed.index += M5_BAR_DURATION
    closed.columns = opened.columns
    frame = pd.concat([opened, closed])
    frame = frame.loc[~frame.index.duplicated(keep="first")].sort_index()
    frame = frame.loc[frame.index >= start]
    if (not np.isfinite(frame.to_numpy()).all() or np.any(frame.to_numpy() <= 0)
            or np.any(frame.bid > frame.mid) or np.any(frame.mid > frame.ask)):
        raise RuntimeError("TA_C_QUOTE_SIDES")
    return Tape(frame.index, frame.mid.to_numpy(), frame.bid.to_numpy(),
                frame.ask.to_numpy(), "source-bound-in-result", "C open/close valuation grid")


def c_book(tape: Tape, cohort: pd.DataFrame, mode: str, slip: float,
           curve: ResearchFinancingCurve | None, initial: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Event cash ledger with frozen entry quantity and bar-end passive fill-time assumption.

    Each selected opportunity remains in the outcome frame, including zero PnL
    for no placement/touch. A quote touch is never called an observed execution.
    """
    if mode not in {"active", "long", "passive"} or not np.isfinite(slip) or slip < 0:
        raise RuntimeError("TA_C_BOOK_MODE")
    n = len(tape.time)
    arrays = {k: np.zeros(n) for k in ["held_units_after", "traded_units", "mid_pnl",
              "spread_cost", "slippage_cost", "commission_cost", "financing_cost"]}
    rows = []
    for i, r in cohort.iterrows():
        observed = {"opportunity": i, "known_at": r.known_at, "filled": False,
                    "mid_bps": 0., "spread_bps": 0., "slippage_bps": 0., "financing_bps": 0.,
                    "net_bps": 0., "normalized_net": 0., "risk_pnl": 0.}
        if not r.executable or (mode == "passive" and not r.passive_touched):
            rows.append(observed)
            continue
        side = 1 if mode == "long" else int(r.side)
        entry_time = r.passive_fill_time if mode == "passive" else r.entry_time
        entry_price = r.passive_limit if mode == "passive" else r.entry_ask if side > 0 else r.entry_bid
        exit_price = r.exit_bid if side > 0 else r.exit_ask
        entry_slip = 0. if mode == "passive" else slip
        a, b = tape.time.get_indexer([entry_time, r.exit_time])
        if (a < 0 or b < a or (b == a and not r.censored_at_end)
                or np.any(arrays["held_units_after"][a:b] != 0)):
            raise RuntimeError("TA_C_EVENT_CLOCK_OR_OVERLAP")
        q = side * initial * r.risk_scale / r.decision_mid
        arrays["held_units_after"][a:b] = q
        arrays["traded_units"][a] += q
        arrays["traded_units"][b] -= q
        arrays["mid_pnl"][a] += q * (tape.mid[a] - r.decision_mid)
        arrays["mid_pnl"][a + 1:b + 1] += q * np.diff(tape.mid[a:b + 1])
        arrays["spread_cost"][a] += q * (entry_price - r.decision_mid)
        arrays["spread_cost"][b] += q * (tape.mid[b] - exit_price)
        arrays["slippage_cost"][a] += abs(q) * entry_price * entry_slip / 1e4
        arrays["slippage_cost"][b] += abs(q) * exit_price * slip / 1e4
        funding_rate = 0.
        if curve is not None:
            long_rate, short_rate = curve.integrated_cost_rates(tape.time[a:b], tape.time[a + 1:b + 1])
            rates = long_rate if side > 0 else short_rate
            arrays["financing_cost"][a + 1:b + 1] += abs(q) * entry_price * rates
            funding_rate = float(rates.sum())
        mid = side * (r.exit_mid - r.decision_mid) / r.decision_mid * 1e4
        spread = side * ((entry_price - r.decision_mid) + (r.exit_mid - exit_price)) / r.decision_mid * 1e4
        slip_cost = (entry_price * entry_slip + exit_price * slip) / r.decision_mid
        funding = entry_price / r.decision_mid * funding_rate * 1e4
        net = mid - spread - slip_cost - funding
        observed.update(filled=True, mid_bps=float(mid), spread_bps=float(spread),
                        slippage_bps=float(slip_cost), financing_bps=float(funding),
                        net_bps=float(net), normalized_net=float(net / r.atr_bps),
                        risk_pnl=float(initial * r.risk_scale * net / 1e4))
        rows.append(observed)
    frame = pd.DataFrame({"time": tape.time, **arrays})
    frame["equity_mid"] = initial + np.cumsum(frame.mid_pnl - frame.spread_cost -
                                              frame.slippage_cost - frame.financing_cost)
    q = frame.held_units_after.to_numpy()
    exit_quote = np.where(q > 0, tape.bid, tape.ask)
    frame["liquidation_reserve"] = abs(q) * (abs(tape.mid - exit_quote) + exit_quote * slip / 1e4)
    frame["equity_liquidation"] = frame.equity_mid - frame.liquidation_reserve
    frame.attrs["initial_equity"] = initial
    outcomes = pd.DataFrame(rows)
    if not np.isclose(frame.equity_liquidation.iloc[-1] - initial, outcomes.risk_pnl.sum(), atol=1e-9, rtol=1e-10):
        raise RuntimeError("TA_C_LEDGER_RECONCILIATION")
    return frame, outcomes


def c_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """Calendar-day accounting; carry marks across closed days, never fabricate an input quote."""
    flow = ["mid_pnl", "spread_cost", "slippage_cost", "commission_cost", "financing_cost"]
    states = ["held_units_after", "traded_units", "equity_mid", "liquidation_reserve", "equity_liquidation"]
    grouped = frame.set_index("time").resample("D", closed="right", label="left")
    daily = pd.concat([grouped[flow].sum(), grouped[states].last().ffill()], axis=1)
    daily = daily.reset_index()
    first = {name: 0. for name in flow + states}
    first.update(time=daily.time.iloc[0] - pd.Timedelta(days=1),
                 equity_mid=frame.attrs["initial_equity"], equity_liquidation=frame.attrs["initial_equity"])
    daily = pd.concat([pd.DataFrame([first]), daily], ignore_index=True)
    daily.attrs = frame.attrs.copy()
    return daily


def c_statistics(series: dict, comparisons: list[dict], counts: np.ndarray,
                 annualization: float, ix: np.ndarray) -> np.ndarray:
    result = []
    denominator = counts[ix].sum()
    for c in comparisons:
        a, b = series[c["model"]], series.get(c["baseline"])
        if c["metric"] in {"mean_net_bps", "normalized_net"}:
            field = "raw" if c["metric"] == "mean_net_bps" else "normalized"
            numerator = a[field][ix].sum() - (b[field][ix].sum() if b is not None else 0.)
            value = numerator / denominator if denominator > 0 else np.nan
        else:
            def sharpe(x):
                sd = x.std(ddof=1)
                return x.mean() / sd * np.sqrt(annualization) if sd > 0 else np.nan
            value = sharpe(a["returns"][ix])
            if b is not None:
                value -= sharpe(b["returns"][ix])
        result.append(value)
    return np.asarray(result)


def run_c(spec: dict, spec_path: Path) -> dict:
    if (spec["cells"] != list(C_CELLS) or spec["hold_bars"] != 12
            or spec["slippage_scenarios"] != [0., .5, 1., 2.]
            or spec["read_end_exclusive"] != "2026-07-01T00:00:00Z"):
        raise RuntimeError("TA_C_PREREG_SCOPE")
    out = Path(spec["output_directory"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "STARTED.json", {"git_head": git("rev-parse", "HEAD"),
                                     "preregistration": str(spec_path), "preregistration_sha256": sha(spec_path)})
    try:
        curve = load_funding(spec)
        market, binding = load_market(spec, columns=[
            "time", "open", "high", "low", "close", "volume", "bid_open", "ask_open",
            "bid_close", "ask_close", "ask_low", "bid_high"])
        signals, signal_counts = c_signal_panel(market, spec)
        signals.to_parquet(out / "SIGNALS.parquet")
        cohort, selection = c_select(market, signals, spec)
        cohort.to_parquet(out / "COHORT.parquet", index=False)
        tape = c_quotes(market, pd.Timestamp(spec["evaluation_start"]))
        dates = pd.date_range(spec["evaluation_start"],
                              pd.Timestamp(spec["read_end_exclusive"]) - pd.Timedelta(days=1), freq="D")
        counts = cohort.groupby(cohort.known_at.dt.floor("D")).size().reindex(dates, fill_value=0).to_numpy()
        series, portfolios, components, fill_comparison, period_detail = {}, {}, {}, {}, {}
        for funding in ["historical_proxy", "zero"]:
            for slip in spec["slippage_scenarios"]:
                tag = f"s{slip:g}:{funding}"
                mode_outcomes = {}
                for mode in ["active", "long", "passive"]:
                    key = f"{mode}:{tag}"
                    book, outcomes = c_book(tape, cohort, mode, slip,
                                            curve if funding == "historical_proxy" else None, spec["initial_equity"])
                    daily = c_daily(book)
                    if not pd.DatetimeIndex(daily.time.iloc[1:]).equals(dates):
                        raise RuntimeError("TA_C_DAILY_POPULATION")
                    # One file contains all dense accounting; summary never confuses M5 and daily Sharpe.
                    daily.to_parquet(out / (key.replace(":", "_") + "_DAILY.parquet"), index=False)
                    summary = portfolio_summary(daily, periods_per_year=spec["periods_per_year"])
                    peak = np.maximum.accumulate(np.r_[spec["initial_equity"], book.equity_liquidation.to_numpy()])
                    summary["max_drawdown_quote_grid"] = float((1 - np.r_[spec["initial_equity"], book.equity_liquidation.to_numpy()] / peak).max())
                    summary["executed_round_trips"] = int(outcomes.filled.sum())
                    # Event trades may exit and re-enter at one timestamp: net traded_units is not execution count.
                    summary["execution_count"] = 2 * int(outcomes.filled.sum())
                    portfolios[key] = summary
                    outcomes.to_parquet(out / (key.replace(":", "_") + "_OUTCOMES.parquet"), index=False)
                    mode_outcomes[mode] = outcomes
                    sums = outcomes.groupby(outcomes.known_at.dt.floor("D"))[["net_bps", "normalized_net"]].sum().reindex(dates, fill_value=0)
                    series[key] = {"raw": sums.net_bps.to_numpy(), "normalized": sums.normalized_net.to_numpy(),
                                   "returns": (portfolio_period_returns(daily) if not summary["insolvent"]
                                               else np.full(len(dates), np.nan))}
                    components[key] = {"per_selected_opportunity": {
                        col: float(outcomes[col].mean()) for col in
                        ["mid_bps", "spread_bps", "slippage_bps", "financing_bps", "net_bps"]},
                        "fills": int(outcomes.filled.sum()), "nonfills_or_unexecuted": int((~outcomes.filled).sum())}
                    period_detail[key] = {str(month): {"selected": int(len(g)), "fills": int(g.filled.sum()),
                                                      "mean_net_bps_per_selected": float(g.net_bps.mean()),
                                                      "risk_pnl": float(g.risk_pnl.sum())}
                                          for month, g in outcomes.groupby(outcomes.known_at.dt.strftime("%Y-%m"))}
                touch = mode_outcomes["passive"].filled.to_numpy()
                fill_comparison[tag] = {"touched": int(touch.sum()), "not_touched_or_not_placed": int((~touch).sum())}
                for label, mask in [("touched", touch), ("not_touched_or_not_placed", ~touch)]:
                    if mask.any():
                        fill_comparison[tag][label + "_active_mean_net_bps"] = float(mode_outcomes["active"].loc[mask, "net_bps"].mean())
                if touch.any():
                    fill_comparison[tag]["touched_passive_mean_net_bps"] = float(mode_outcomes["passive"].loc[touch, "net_bps"].mean())
        comparisons = []
        for funding in ["historical_proxy", "zero"]:
            for slip in spec["slippage_scenarios"]:
                tag = f"s{slip:g}:{funding}"
                for mode, baseline in [("active", "long"), ("passive", "long"), ("passive", "flat")]:
                    for metric in spec["effects"]:
                        comparisons.append({"name": f"{mode}:{tag}:vs_{baseline}:{metric}",
                                            "model": f"{mode}:{tag}", "baseline": f"{baseline}:{tag}", "metric": metric})
        point = c_statistics(series, comparisons, counts, spec["periods_per_year"], np.arange(len(dates)))
        boot = np.array([c_statistics(series, comparisons, counts, spec["periods_per_year"], ix)
                         for ix in stationary_bootstrap_indices(len(dates), draws=spec["bootstrap_draws"],
                                   mean_block_length=spec["mean_block_length"], seed=spec["seed"])])
        eligible = np.isfinite(point) & np.isfinite(boot).all(axis=0) & (boot.std(axis=0, ddof=1) > 0)
        inference = {}
        if eligible.any():
            chosen = [c for c, ok in zip(comparisons, eligible) if ok]
            inference = max_t_inference(point[eligible], boot[:, eligible], names=[c["name"] for c in chosen],
                                       alpha=spec["alpha"], desired_power=spec["desired_power"],
                                       effect_sizes=np.array([spec["effects"][c["metric"]] for c in chosen]),
                                       minimum_relevant_effect=np.array([spec["effects"][c["metric"]][0] for c in chosen]))
        endpoints = {r["name"]: r for r in inference.get("endpoints", [])}
        for c, ok in zip(comparisons, eligible):
            if not ok:
                endpoints[c["name"]] = {"name": c["name"], "effect_verdict": "INKONKLUSIV",
                                        "reason": "undefined_statistic_or_zero_bootstrap_variation"}
        required = [f"passive:s1:{f}:vs_{b}:mean_net_bps" for f in ["historical_proxy", "zero"] for b in ["flat", "long"]]
        verdicts = [endpoints[name]["effect_verdict"] for name in required]
        positive_economics = all(not portfolios[f"passive:s1:{f}"]["insolvent"]
                                 and portfolios[f"passive:s1:{f}"]["total_net_bps"] > 0
                                 for f in ["historical_proxy", "zero"])
        decision = ("GO" if all(v == "GO" for v in verdicts) and positive_economics else
                    "NO_GO" if "NO_GO" in verdicts else "INKONKLUSIV")
        np.savez_compressed(out / "BOOTSTRAP_STATISTICS.npz", estimates=point, bootstrap_estimates=boot)
        pd.DataFrame({"day": dates, "selected": counts, **{
            f"{key}:{metric}": values for key, block in series.items() for metric, values in block.items()
        }}).to_parquet(out / "PAIRED_DAYS.parquet", index=False)
        result = {"schema": "gx1_ta_measurement_c_v1", "git_head": git("rev-parse", "HEAD"),
                  "preregistration_sha256": sha(spec_path), "inputs": binding,
                  "test_outcomes_accessed": False, "native_training": False,
                  "evidence_class": "measured_reused_development_quote_touch_simulation",
                  "selection": {**selection, "selected": len(cohort), "executable": int(cohort.executable.sum()),
                                "unexecuted": int((~cohort.executable).sum()), "censored": int(cohort.censored_at_end.sum()),
                                "passive_placed": int(cohort.passive_placed.sum()), "passive_touched": int(cohort.passive_touched.sum()),
                                "cell_counts": signal_counts, "calendar_days": len(dates)},
                  "portfolios": portfolios, "components": components, "fill_selection": fill_comparison,
                  "per_month": period_detail, "inference": inference,
                  "declared_family": [c["name"] for c in comparisons], "endpoints": list(endpoints.values()),
                  "decision": {"verdict": decision, "positive_economics": positive_economics, "required_endpoints": required,
                               "scope": "GO could propose separate prospective execution research only; no observed fills or native training"},
                  "limitations": spec["limitations"]}
        result["artifacts"] = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()}
        write_json(out / "RESULT.json", result)
        write_json(out / "TERMINAL.json", {"status": "COMPLETE", "result_sha256": sha(out / "RESULT.json"),
                                         "finished_utc": datetime.now(timezone.utc).isoformat()})
        return {"status": "COMPLETE", "out": str(out), "decision": decision}
    except Exception as exc:
        write_json(out / "TERMINAL.json", {"status": "FAILED", "error": str(exc),
                                         "finished_utc": datetime.now(timezone.utc).isoformat()})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["fetch-funding", "fetch-alfred", "audit-alfred-chunks", "prepare-b-macros", "run-a", "run-c"])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--receipt-sha256")
    args = parser.parse_args()
    spec = checked_spec(args.spec, args.spec_sha256)
    if args.mode == "audit-alfred-chunks":
        print(json.dumps(audit_alfred_chunks(spec, args.spec, args.receipt_sha256)))
        return 0
    owner = {"prepare-b-macros": prepare_b_macros, "fetch-funding": fetch_funding, "fetch-alfred": fetch_alfred, "run-a": run_a, "run-c": run_c}[args.mode]
    result = owner(spec, args.spec)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
