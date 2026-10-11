#!/usr/bin/env python3
"""Bounded A/B/C research orchestration. Implements A/C, B source checks and the matched B research core.

Reuses indicator, clock, learner, portfolio and inference owners. No native
training, broker access, TEST, automatic parameter search or result promotion.
"""
from __future__ import annotations

import argparse
import csv
import gzip
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

from gx1.contracts.immutable_event_authority_v1 import _publish_file_noreplace, _fsync_directory
from gx1.contracts.gc_order_flow_source_v1 import (
    GC_AUDIT_OUTPUT_ROOT, GC_LOCAL_SOURCE_ROOT, GC_RESEARCH_ID, GC_SOURCE_SCHEMA,
    audit_gc_records, require_gc_file_declaration,
    required_gc_fields,
)
from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution
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
ALFRED_COMPONENT_SERIES = ("DFII10", "DTWEXBGS", "T10YIE", "VIXCLS")
MACRO_CORE_SERIES = ALFRED_COMPONENT_SERIES[:3]
MACRO_CORE_FEATURES = B_MACRO_FEATURES[:6]
MACRO_CORE_ALL_FEATURES = FEATURES + MACRO_CORE_FEATURES


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
    _publish_file_noreplace(temp, path)
    _fsync_directory(path.parent)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    """Publish a complete, re-readable research table without replacing evidence."""
    temp = path.with_suffix(path.suffix + ".part")
    with temp.open("xb") as handle:
        frame.to_parquet(handle, index=False)
        handle.flush()
        os.fsync(handle.fileno())
    pd.testing.assert_frame_equal(pd.read_parquet(temp), frame.reset_index(drop=True))
    _publish_file_noreplace(temp, path)
    _fsync_directory(path.parent)


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
    return _prepare_macro_components(spec, spec_path, ALFRED_COMPONENT_SERIES, "four")


def prepare_macro_core(spec: dict, spec_path: Path) -> dict:
    """Separate approved three-source arm, never admission of full B."""
    expected_names = dict(zip(MACRO_CORE_SERIES, [MACRO_CORE_FEATURES[i:i + 2] for i in (0, 2, 4)]))
    if (spec["feature_names"] != expected_names
            or spec["transforms"] != {"DFII10": "level", "DTWEXBGS": "log", "T10YIE": "level"}
            or spec["change_d1"] != 21 or spec["fits_allowed"] is not False
            or spec["test_accessed"] is not False or spec["full_b_admitted"] is not False):
        raise RuntimeError("TA_MACRO_CORE_COMPONENT_CONTRACT")
    return _prepare_macro_components(spec, spec_path, MACRO_CORE_SERIES, "three")


def _prepare_macro_components(spec: dict, spec_path: Path, series: tuple[str, ...], count: str) -> dict:
    if spec["macro_series"] != list(series):
        raise RuntimeError("TA_B_MACRO_SERIES_SET")
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
        if [r["series"] for r in audit["records"]] != list(ALFRED_COMPONENT_SERIES):
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
            if sid not in series:
                continue
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
        write_parquet(out / "MACRO_COMPONENTS.parquet", panel)
        result = {"status": ("COMPLETE_FOUR_MACRO_COMPONENTS_ONLY" if count == "four"
                             else "COMPLETE_MACRO_CORE_COMPONENTS"), "source_commit": git("rev-parse", "HEAD"),
                  "manifest_sha256": sha(spec_path), "source_audit": spec["source_audit"],
                  "clock_cache": spec["clock_cache"], "clock_columns_read": ["session_open", "decision_time"],
                  "clock_rows": len(clocks), "coverage": coverage, "provenance": provenance,
                  f"{count}_source_complete_rows": int(common.sum()), "full_b_admitted": False,
                  "missing_full_b_sources": spec["missing_full_b_sources"], "feature_names": names,
                  "fits_run": False, "new_market_outcomes_read": False, "test_accessed": False,
                  "artifact": {"path": str(out / "MACRO_COMPONENTS.parquet"),
                               "sha256": sha(out / "MACRO_COMPONENTS.parquet")}}
        if common.any():
            result[f"{count}_source_first_complete_decision"] = str(clocks.loc[common, "decision_time"].iloc[0])
            result[f"{count}_source_last_complete_decision"] = str(clocks.loc[common, "decision_time"].iloc[-1])
        write_json(out / "RESULT.json", result)
        write_json(out / "TERMINAL.json", {"status": result["status"], "result_sha256": sha(out / "RESULT.json")})
        return {"out": str(out), "result_sha256": sha(out / "RESULT.json"),
                f"{count}_source_complete_rows": int(common.sum()), "full_b_admitted": False}
    except Exception as error:
        write_json(out / "TERMINAL.json", {"status": "FAILED", "error": str(error)})
        raise



def parse_gld_holdings_snapshot(raw: bytes) -> pd.DataFrame:
    """Parse the observed SPDR archive format, without inventing vintage dates."""
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
    if not rows or rows[0][0] != "SPDR Gold Shares (New York Stock Exchange Arca)":
        raise RuntimeError("TA_B_GLD_SOURCE_IDENTITY")
    headers = [i for i, row in enumerate(rows) if row and row[0].strip() == "Date"]
    if len(headers) != 1:
        raise RuntimeError("TA_B_GLD_HEADER")
    header = [value.strip() for value in rows[headers[0]]]
    tonnes = "Total Net Asset Value Tonnes in the Trust as at 4.15 p.m. NYT"
    if len(header) != len(set(header)) or tonnes not in header:
        raise RuntimeError("TA_B_GLD_TONNES_COLUMN")
    column, parsed = header.index(tonnes), []
    for row in rows[headers[0] + 1:]:
        if len(row) != len(header):
            raise RuntimeError("TA_B_GLD_ROW_WIDTH")
        observed = datetime.strptime(row[0].strip(), "%d-%b-%Y").date().isoformat()
        value = row[column].strip()
        if value in ("HOLIDAY", "NYSE Closed"):
            amount = np.nan
        elif re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value):
            amount = float(value)
            if not np.isfinite(amount) or amount <= 0:
                raise RuntimeError("TA_B_GLD_NONPOSITIVE_TONNES")
        else:
            raise RuntimeError("TA_B_GLD_UNKNOWN_VALUE")
        parsed.append({"observation_date": observed, "gld_tonnes": amount,
                       "source_value": value})
    dates = [row["observation_date"] for row in parsed]
    if not dates or dates != sorted(set(dates)):
        raise RuntimeError("TA_B_GLD_OBSERVATION_ORDER")
    return pd.DataFrame(parsed)


class CotPreformattedReport(HTMLParser):
    """Only the actual PRE report text; metadata/comments cannot set identity."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.inside = False
        self.count = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "pre":
            if self.inside:
                raise RuntimeError("TA_B_COT_NESTED_PRE")
            self.inside = True
            self.count += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "pre":
            self.inside = False

    def handle_data(self, data: str) -> None:
        if self.inside:
            self.parts.append(data)


def parse_cot_gold_snapshot(raw: bytes) -> pd.DataFrame:
    """Observed CFTC legacy futures-only text: exact Gold block and identities."""
    parser = CotPreformattedReport()
    parser.feed(raw.decode("iso-8859-1"))
    parser.close()
    if parser.count != 1 or parser.inside:
        raise RuntimeError("TA_B_COT_PRE")
    text = "".join(parser.parts).replace("\r\n", "\n")
    starts = list(re.finditer(r"(?m)^GOLD - COMMODITY EXCHANGE INC\.\s+Code-088691\s*$", text))
    if len(starts) != 1:
        raise RuntimeError("TA_B_COT_GOLD_IDENTITY")
    tail = text[starts[0].end():]
    next_market = re.search(r"(?m)^.*Code-[0-9]{6}\s*$", tail)
    block = tail[:next_market.start()] if next_market else tail
    position = re.search(r"(?m)^FUTURES ONLY POSITIONS AS OF ([0-9]{2}/[0-9]{2}/[0-9]{2})[ \t]*\|", block)
    category = re.search(r"(?m)^[ \t]*NON-COMMERCIAL[ \t]*\|[ \t]*COMMERCIAL[ \t]*\|[ \t]*TOTAL[ \t]*\|[ \t]*POSITIONS[ \t]*$", block)
    columns = [line for line in block.splitlines() if "SPREADS" in line]
    expected = ["LONG", "SHORT", "SPREADS", "LONG", "SHORT", "LONG", "SHORT", "LONG", "SHORT"]
    if not position or not category or len(columns) != 1 or [x.strip() for x in columns[0].split("|")] != expected:
        raise RuntimeError("TA_B_COT_LEGACY_FUTURES_ONLY_SCHEMA")
    commitments = re.search(
        r"\(CONTRACTS OF 100 TROY OUNCES\)[ \t]+OPEN INTEREST:[ \t]+([0-9,]+)\n"
        r"COMMITMENTS\n([ \t0-9,]+)\n", block)
    if not commitments:
        raise RuntimeError("TA_B_COT_COMMITMENTS_ROW")
    tokens = [commitments[1], *commitments[2].split()]
    if len(tokens) != 10 or not all(re.fullmatch(r"(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)", x) for x in tokens):
        raise RuntimeError("TA_B_COT_INTEGER_SCHEMA")
    oi, ncl, ncs, spread, cl, cs, tl, ts, nrl, nrs = [int(x.replace(",", "")) for x in tokens]
    if oi <= 0 or (ncl + spread + cl, ncs + spread + cs, tl + nrl, ts + nrs) != (tl, ts, oi, oi):
        raise RuntimeError("TA_B_COT_ACCOUNTING_IDENTITY")
    return pd.DataFrame([{
        "observation_date": datetime.strptime(position[1], "%m/%d/%y").date().isoformat(),
        "noncommercial_long": ncl, "noncommercial_short": ncs, "open_interest": oi,
        "noncommercial_net_over_oi": (ncl - ncs) / oi,
    }])


def import_b_archived_snapshots(spec: dict, spec_path: Path) -> dict:
    """Import bound observed source formats only; isolated copies never admit B."""
    out = Path(spec["output_directory"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "STARTED.json", {"source_commit": git("rev-parse", "HEAD"),
                                     "manifest_sha256": sha(spec_path)})
    try:
        parsers = {"GLD_TONNES": parse_gld_holdings_snapshot,
                   "COT_088691_LEGACY_FUTURES_ONLY": parse_cot_gold_snapshot}
        originals = {
            "GLD_TONNES": "http://www.spdrgoldshares.com/assets/dynamic/GLD/GLD_US_archive_EN.csv",
            "COT_088691_LEGACY_FUTURES_ONLY": "https://www.cftc.gov/dea/futures/deacmxsf.htm",
        }
        if {x["series"] for x in spec["snapshots"]} != set(parsers):
            raise RuntimeError("TA_B_SNAPSHOT_SERIES_SET")
        records = []
        for number, item in enumerate(spec["snapshots"]):
            source, receipt_path = Path(item["raw_path"]), Path(item["receipt_path"])
            if sha(source) != item["raw_sha256"] or sha(receipt_path) != item["receipt_sha256"]:
                raise RuntimeError("TA_B_SNAPSHOT_HASH")
            raw, receipt = source.read_bytes(), json.loads(receipt_path.read_text())
            capture = pd.Timestamp(item["capture_utc"])
            if capture.tz is None:
                raise RuntimeError("TA_B_SNAPSHOT_CAPTURE_TIMEZONE")
            capture = capture.tz_convert("UTC")
            expected_url = ("https://web.archive.org/web/" + capture.strftime("%Y%m%d%H%M%S")
                            + "id_/" + originals[item["series"]])
            if (receipt["http_status"] != 200 or receipt["final_url"] != expected_url
                    or receipt["url"] != expected_url
                    or receipt["response_sha256"] != item["raw_sha256"]
                    or receipt["response_bytes"] != len(raw)
                    or capture > pd.Timestamp(receipt["finished_utc"])):
                raise RuntimeError("TA_B_SNAPSHOT_RECEIPT")
            frame = parsers[item["series"]](raw)
            if frame.observation_date.max() > capture.tz_convert("America/New_York").date().isoformat():
                raise RuntimeError("TA_B_SNAPSHOT_FUTURE_OBSERVATION")
            frame["evidence_available_at_utc"] = capture
            frame["source_sha256"] = item["raw_sha256"]
            frame["availability_evidence"] = "archive_capture_upper_bound_not_original_publication"
            # Availability plus the registered complete D1 lag is applied downstream,
            # never by relabelling these historical rows as known on observation dates.
            path = out / f"SNAPSHOT_{number:03d}.parquet"
            temporary = path.with_suffix(".parquet.part")
            frame.to_parquet(temporary, index=False)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.rename(temporary, path)
            values = "gld_tonnes" if item["series"] == "GLD_TONNES" else "noncommercial_net_over_oi"
            record = {"series": item["series"], "raw_path": str(source), "raw_sha256": item["raw_sha256"],
                      "capture_utc": capture.isoformat(), "rows": len(frame),
                      "numeric_rows": int(frame[values].notna().sum()),
                      "missing_rows": int(frame[values].isna().sum()),
                      "first_observation": frame.observation_date.min(),
                      "last_observation": frame.observation_date.max(),
                      "artifact": {"path": str(path), "sha256": sha(path)}}
            records.append(record)
        result = {"status": "COMPLETE_BOUND_SNAPSHOT_IMPORT_ONLY",
                  "source_commit": git("rev-parse", "HEAD"), "manifest_sha256": sha(spec_path),
                  "records": records, "full_b_admitted": False, "fits_run": False,
                  "training_enabled": False, "test_accessed": False,
                  "network_requests": 0, "market_outcomes_evaluated": False,
                  "limitations": ["Isolated captures do not prove continuous GLD/COT vintage coverage.",
                                  "Original publication timestamps remain unproved.",
                                  "The additional canonical D1 lag has not been applied; these are source rows, not decision inputs.",
                                  "Four published COT report changes require additional distinct reports."]}
        write_json(out / "RESULT.json", result)
        write_json(out / "TERMINAL.json", {"status": result["status"], "result_sha256": sha(out / "RESULT.json")})
        return {"out": str(out), "result_sha256": sha(out / "RESULT.json"), "full_b_admitted": False}
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
    return _join_macro_components(panel, components, B_MACRO_FEATURES)


def join_macro_core_components(panel: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    return _join_macro_components(panel, components, MACRO_CORE_FEATURES)


def _join_macro_components(panel: pd.DataFrame, components: pd.DataFrame,
                           macro_features: list[str]) -> pd.DataFrame:
    if (panel.columns.has_duplicates or components.columns.has_duplicates
            or not set(FEATURES).issubset(panel)
            or not set(macro_features).issubset(components)
            or set(macro_features).intersection(panel)):
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
    result[macro_features] = components[macro_features].to_numpy(float)
    if np.isinf(result[FEATURES + macro_features].to_numpy(float)).any():
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
        # Holdouts are calendar years by default; ``fold_spans`` declares multi-year holdouts
        # [start_year, end_year_exclusive] for fold-structure sensitivity (same purge rule).
        spans = [tuple(int(v) for v in span) for span in spec["fold_spans"]] if spec.get("fold_spans") else [(y, y + 1) for y in spec["fold_years"]]
        for year, end_year in spans:
            start = pd.Timestamp(f"{year}-01-01", tz="UTC")
            end = pd.Timestamp(f"{end_year}-01-01", tz="UTC")
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
            arm_name = ("A" if feature_names == FEATURES else
                        "MACRO_CORE" if feature_names == MACRO_CORE_ALL_FEATURES else "B")
            print(f"[TA-{arm_name}] fitted year={year} horizon={horizon} rows={len(fit)}/{len(hold)}", flush=True)
    return forecasts, fits


def fit_matched_b(panel: pd.DataFrame, spec: dict) -> tuple[dict, list[dict]]:
    """Fit A and full B on shared rows; source admission remains a separate prerequisite."""
    return _fit_matched_macro(panel, spec, B_FEATURES, "b")


def fit_matched_macro_core(panel: pd.DataFrame, spec: dict) -> tuple[dict, list[dict]]:
    return _fit_matched_macro(panel, spec, MACRO_CORE_ALL_FEATURES, "macro_core")


def _fit_matched_macro(panel: pd.DataFrame, spec: dict, feature_names: list[str],
                       arm_name: str) -> tuple[dict, list[dict]]:
    if (spec["feature_names"] != feature_names or spec["horizons"] != [20, 5]
            or spec["primary_horizon"] != 20 or panel.columns.has_duplicates
            or not set(feature_names).issubset(panel)):
        raise RuntimeError("TA_B_FEATURE_CONTRACT")
    values = panel[feature_names].to_numpy(float)
    if np.isinf(values).any():
        raise RuntimeError("TA_B_NONFINITE_FEATURE")
    # Keep the full clock. Filtering the panel would shorten horizons across missing inputs.
    population = np.isfinite(values).all(axis=1)
    a, a_fits = _fit_predictions(panel, spec, FEATURES, population)
    b, b_fits = _fit_predictions(panel, spec, feature_names, population)
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
            row.update({"a": {"ridge": af["ridge"], "hgb": af["hgb"]},
                        arm_name: {"ridge": bf["ridge"], "hgb": bf["hgb"]}})
        matched.append(row)
    forecasts = {}
    for horizon in spec["horizons"]:
        if not np.array_equal(a[horizon]["constant"], b[horizon]["constant"], equal_nan=True):
            raise RuntimeError("TA_B_CONSTANT_MISMATCH")
        forecasts[horizon] = {
            "a_ridge": a[horizon]["ridge"], "a_hgb": a[horizon]["hgb"],
            f"{arm_name}_ridge": b[horizon]["ridge"], f"{arm_name}_hgb": b[horizon]["hgb"],
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
    return _evaluate_matched_macro(panel, forecasts, curve, spec, out, B_FEATURES, "b")


def evaluate_matched_macro_core(panel: pd.DataFrame, forecasts: dict, curve: ResearchFinancingCurve,
                                spec: dict, out: Path) -> dict:
    return _evaluate_matched_macro(panel, forecasts, curve, spec, out, MACRO_CORE_ALL_FEATURES, "macro_core")


def _evaluate_matched_macro(panel: pd.DataFrame, forecasts: dict, curve: ResearchFinancingCurve,
                            spec: dict, out: Path, feature_names: list[str], arm_name: str) -> dict:
    if (spec["feature_names"] != feature_names or spec["horizons"] != [20, 5]
            or spec["primary_horizon"] != 20 or set(forecasts) != set(spec["horizons"])):
        raise RuntimeError("TA_B_EVALUATION_CONTRACT")
    masks = []
    for arm in forecasts.values():
        if set(arm) != {"a_ridge", "a_hgb", f"{arm_name}_ridge", f"{arm_name}_hgb", "constant"}:
            raise RuntimeError("TA_B_EVALUATION_ARMS")
        for values in arm.values():
            if np.asarray(values).shape != (len(panel),) or np.isinf(values).any():
                raise RuntimeError("TA_B_FORECAST_POPULATION_MISMATCH")
            masks.append(np.isfinite(values))
    if any(not np.array_equal(masks[0], m) for m in masks[1:]):
        raise RuntimeError("TA_B_FORECAST_POPULATION_MISMATCH")
    return _evaluate(panel, forecasts, curve, spec, out,
                     {f"{arm_name}_ridge": ["a_ridge", "long", "constant", "trend", "buy_hold"],
                      f"{arm_name}_hgb": ["a_hgb", "long", "constant", "trend", "buy_hold"]})


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


def _completed_research_result(binding: dict, status: str) -> dict:
    path = Path(binding["path"])
    terminal_path = path.with_name("TERMINAL.json")
    if sha(path) != binding["sha256"] or sha(terminal_path) != binding["terminal_sha256"]:
        raise RuntimeError("TA_MACRO_CORE_RESULT_HASH")
    result = json.loads(path.read_text())
    terminal = json.loads(terminal_path.read_text())
    if terminal["status"] != status or terminal["result_sha256"] != binding["sha256"]:
        raise RuntimeError("TA_MACRO_CORE_INCOMPLETE_INPUT")
    return result


def load_macro_core_panel(spec: dict) -> tuple[pd.DataFrame, dict]:
    """Reuse complete cached bytes; admit neither a partial artifact nor a new clock."""
    component = _completed_research_result(spec["component_result"], "COMPLETE_MACRO_CORE_COMPONENTS")
    cached = _completed_research_result(spec["cache_result"], "COMPLETE")
    proof_path = Path(spec["component_verification"]["path"])
    if sha(proof_path) != spec["component_verification"]["sha256"]:
        raise RuntimeError("TA_MACRO_CORE_VERIFICATION_HASH")
    proof = json.loads(proof_path.read_text())
    if (proof["status"] != "PASS"
            or proof["component_result_sha256"] != spec["component_result"]["sha256"]
            or component["status"] != "COMPLETE_MACRO_CORE_COMPONENTS"
            or component["feature_names"] != MACRO_CORE_FEATURES
            or component["test_accessed"] is not False or component["fits_run"] is not False
            or component["full_b_admitted"] is not False
            or cached["schema"] != "gx1_ta_measurement_a_v1" or cached["test_accessed"] is not False):
        raise RuntimeError("TA_MACRO_CORE_INPUT_CONTRACT")
    cache = spec["daily_cache"]
    path = Path(cache["path"])
    if (cached["artifacts"][path.name]["sha256"] != cache["sha256"]
            or component["clock_cache"]["path"] != cache["path"]
            or component["clock_cache"]["sha256"] != cache["sha256"]
            or sha(path) != cache["sha256"]
            or sha(Path(component["artifact"]["path"])) != component["artifact"]["sha256"]):
        raise RuntimeError("TA_MACRO_CORE_CACHE_HASH")
    panel = pd.read_parquet(path)
    components = pd.read_parquet(component["artifact"]["path"])
    panel = join_macro_core_components(panel, components)
    if (pd.Timestamp(panel.decision_time.min()) < pd.Timestamp(spec["read_start"])
            or pd.Timestamp(panel.decision_time.max()) >= pd.Timestamp(spec["read_end_exclusive"])
            or pd.Timestamp(panel.fill_time.max()) >= pd.Timestamp(spec["read_end_exclusive"])):
        raise RuntimeError("TA_MACRO_CORE_TIME_BOUNDS")
    complete = np.flatnonzero(np.isfinite(panel[MACRO_CORE_ALL_FEATURES].to_numpy(float)).all(axis=1))
    expected = spec["common_input_population"]
    if (len(complete) != expected["rows"]
            or hashlib.sha256(complete.astype("<i8").tobytes()).hexdigest() != expected["positions_sha256"]):
        raise RuntimeError("TA_MACRO_CORE_POPULATION_DRIFT")
    return panel, {"daily_cache": cache, "cache_result": spec["cache_result"],
                   "component_result": spec["component_result"],
                   "component_verification": spec["component_verification"],
                   "original_market_inputs": cached["inputs"]}


def run_macro_core(spec: dict, spec_path: Path) -> dict:
    if (spec["arm"] != "MACRO_CORE" or spec["feature_names"] != MACRO_CORE_ALL_FEATURES
            or spec["macro_series"] != list(MACRO_CORE_SERIES)
            or spec["horizons"] != [20, 5] or spec["primary_horizon"] != 20
            or spec["test_accessed"] is not False or spec["full_b_admitted"] is not False):
        raise RuntimeError("TA_MACRO_CORE_PREREG_SCOPE")
    out = Path(spec["output_directory"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "STARTED.json", {"git_head": git("rev-parse", "HEAD"),
                                     "preregistration": str(spec_path), "preregistration_sha256": sha(spec_path)})
    try:
        panel, binding = load_macro_core_panel(spec)
        curve = load_funding(spec)
        forecasts, fits = fit_matched_macro_core(panel, spec)
        rows = np.flatnonzero(np.isfinite(forecasts[20]["constant"]))
        expected = spec["common_forecast_population"]
        if (len(rows) != expected["rows"]
                or hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest() != expected["positions_sha256"]):
            raise RuntimeError("TA_MACRO_CORE_FORECAST_DRIFT")
        write_json(out / "FITS.json", {"fits": fits})
        write_parquet(out / "PREDICTIONS.parquet", pd.DataFrame({"fill_time": panel.fill_time, **{
            f"h{h}:{name}": pred for h, arm in forecasts.items() for name, pred in arm.items()}}))
        result = evaluate_matched_macro_core(panel, forecasts, curve, spec, out)
        # Terminal publication follows full table readability, fsync and inventory.
        artifacts = {}
        for path in sorted(out.iterdir()):
            if path.suffix == ".parquet":
                pq.read_table(path)
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
            artifacts[path.name] = {"sha256": sha(path), "bytes": path.stat().st_size}
        result.update(schema="gx1_ta_macro_core_measurement_v1", git_head=git("rev-parse", "HEAD"),
                      preregistration_sha256=sha(spec_path), inputs=binding, artifacts=artifacts,
                      primary_horizon=spec["primary_horizon"], test_accessed=False,
                      native_training=False, full_b_admitted=False,
                      evidence_class="measured_reused_development_walkforward")
        write_json(out / "RESULT.json", result)
        write_json(out / "TERMINAL.json", {"status": "COMPLETE", "result_sha256": sha(out / "RESULT.json"),
                                         "finished_utc": datetime.now(timezone.utc).isoformat()})
        return {"status": "COMPLETE", "out": str(out), "decisions": result["decisions"]}
    except Exception as exc:
        write_json(out / "TERMINAL.json", {"status": "FAILED", "error": str(exc),
                                         "finished_utc": datetime.now(timezone.utc).isoformat()})
        raise


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


def c_select(market: pd.DataFrame, signals: pd.DataFrame, spec: dict, *,
             cells: tuple[str, ...] = C_CELLS) -> tuple[pd.DataFrame, dict]:
    """One shared reservation cohort; fill outcomes never change later selection."""
    time, bar = market.index, M5_BAR_DURATION
    end = pd.Timestamp(spec["read_end_exclusive"])
    terminal_time = time[-1] + bar
    if terminal_time > end:
        raise RuntimeError("TA_C_TERMINAL_CLOCK")
    busy_until = pd.Timestamp(spec["evaluation_start"])
    records, counts = [], {"conflicting_rows": 0, "same_side_duplicates": 0, "overlap_rows": 0}
    for t, row in signals.iterrows():
        active = [(name, int(row[name])) for name in cells if row[name] != 0]
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


# One operator-authorized technical hypothesis. These are research rules only.
SWEEP_ARMS = ("sweep", "rolling_activity", "anchored_activity", "long")


def decode_dukascopy_bi5(raw: bytes) -> np.ndarray:
    """Legacy BI5 >3i2f, milliseconds/ask/bid/quoted sizes; not executed flow.

    Layout independently checked against dukascopy-node v1.46.4 decompressor.
    Absolute hour and instrument identity still require a source receipt.
    """
    import lzma
    if not raw:
        raise RuntimeError("TA_DUKA_EMPTY_FILE")
    decoder = lzma.LZMADecompressor()
    data = decoder.decompress(raw, max_length=64 * 1024 * 1024 + 1)
    if not decoder.eof or decoder.unused_data or len(data) > 64 * 1024 * 1024 or len(data) % 20:
        raise RuntimeError("TA_DUKA_BINARY_LENGTH")
    dtype = np.dtype([("ms", ">i4"), ("ask", ">i4"), ("bid", ">i4"),
                      ("ask_size", ">f4"), ("bid_size", ">f4")])
    records = np.frombuffer(data, dtype=dtype)
    if not len(records):
        raise RuntimeError("TA_DUKA_NO_RECORDS")
    return np.column_stack([records[name] for name in dtype.names]).astype(float)


def audit_dukascopy_cache(spec: dict, spec_path: Path) -> dict:
    out = Path(spec["output_directory"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "STARTED.json", {"git_head": git("rev-parse", "HEAD"),
                                     "spec_sha256": sha(spec_path)})
    rows = []
    try:
        for item in spec["files"]:
            path = Path(item["path"])
            if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
                raise RuntimeError("TA_DUKA_SOURCE_CHANGED")
            row = {**item}
            try:
                ticks = decode_dukascopy_bi5(path.read_bytes())
                ms, ask, bid, av, bv = ticks.T
                checks = {
                    "nonfinite": int((~np.isfinite(ticks)).sum()),
                    "offset_outside_hour": int(((ms < 0) | (ms >= 3600000)).sum()),
                    "time_reversals": int((np.diff(ms) < 0).sum()),
                    "nonpositive_prices": int(((ask <= 0) | (bid <= 0)).sum()),
                    "crossed_quotes": int((ask < bid).sum()),
                    "negative_quoted_sizes": int(((av < 0) | (bv < 0)).sum()),
                }
                valid = not any(checks.values())
                row.update(status="STRUCTURALLY_VALID" if valid else "INVALID_RECORDS",
                           records=len(ticks), checks=checks,
                           same_millisecond_updates=int((np.diff(ms) == 0).sum()),
                           exact_duplicate_records=int(len(ticks) - len(np.unique(ticks, axis=0))),
                           zero_quoted_sizes=int(((av == 0) | (bv == 0)).sum()),
                           quoted_size_unique_counts=[len(np.unique(av)), len(np.unique(bv))])
                if valid:
                    spread = (ask - bid) / ((ask + bid) / 2) * 1e4
                    row.update(first_offset_ms=int(ms[0]), last_offset_ms=int(ms[-1]),
                               max_intertick_gap_ms=float(np.diff(ms).max()) if len(ms) > 1 else None,
                               spread_bps_quantiles=np.quantile(spread, [0, .5, .95, 1]).tolist())
            except (RuntimeError, ValueError, EOFError) as exc:
                row.update(status="DECODE_REJECTED", error=str(exc))
            except __import__("lzma").LZMAError as exc:
                row.update(status="DECODE_REJECTED", error=str(exc))
            rows.append(row)
        valid = [r for r in rows if r["status"] == "STRUCTURALLY_VALID"]
        result = {
            "schema": "gx1_dukascopy_existing_cache_audit_v1", "status": "COMPLETE_NOT_ADMITTED",
            "files": rows, "file_count": len(rows), "total_bytes": sum(r["bytes"] for r in rows),
            "empty_files": sum(r["bytes"] == 0 for r in rows),
            "structurally_valid_files": len(valid), "records": sum(r["records"] for r in valid),
            "invalid_or_rejected_files": len(rows) - len(valid),
            "filename_years": {str(y): sum(r.get("filename_year") == y for r in rows) for y in [2025, 2026]},
            "source_receipts_present": False, "absolute_timestamp_provenance_verified": False,
            "continuous_coverage_proven": False, "admitted_to_economic_test": False,
            "reason": "Sparse legacy cache lacks original download receipts/hour mapping; structural validity is not source qualification.",
            "new_downloads": 0, "test_outcomes_accessed": False,
            "format_reference": spec["format_reference"], "preregistration_sha256": sha(spec_path),
        }
        write_json(out / "RESULT.json", result)
        write_json(out / "TERMINAL.json", {"status": "COMPLETE", "result_sha256": sha(out / "RESULT.json")})
        return {k: v for k, v in result.items() if k != "files"}
    except Exception as exc:
        write_json(out / "TERMINAL.json", {"status": "FAILED", "error": repr(exc)})
        raise


def sweep_confirmations(market: pd.DataFrame, up: np.ndarray, down: np.ndarray,
                        confirmation_bars: int) -> pd.DataFrame:
    """Same decision cohort, one named anchor, no future-dependent confirmation."""
    from gx1.features.volume_features import compute_volume_features
    if confirmation_bars != 5:
        raise RuntimeError("TA_SWEEP_CONFIRMATION_SCOPE")
    if len(up) != len(market) or len(down) != len(market):
        raise RuntimeError("TA_SWEEP_EVENT_SHAPE")
    volume = market.volume.to_numpy(float)
    activity = compute_volume_features(market)["vol_ratio_5_20"]
    rolling = (market.close * market.volume).rolling(20).sum() / market.volume.rolling(20).sum()
    event = (up > 0) | (down > 0)
    records = []
    for i in np.flatnonzero((up > 0) ^ (down > 0)):
        j = i + confirmation_bars
        if j >= len(market):
            continue
        # All exclusions are observable when the confirmation bar closes.
        if event[i + 1:j + 1].any() or np.any(np.diff(market.index.asi8[i:j + 1]) != M5_BAR_DURATION.value):
            continue
        if not np.isfinite(rolling.iloc[j]) or not np.isfinite(activity[j]):
            continue
        close = market.close.iloc[j]
        anchored = float(np.dot(market.close.iloc[i:j + 1], volume[i:j + 1]) / volume[i:j + 1].sum())
        side = 1 if down[i] > 0 else -1
        records.append({
            "signal_bar_start": market.index[j], "known_at": market.index[j] + M5_BAR_DURATION,
            "anchor_bar_start": market.index[i], "anchor_known_at": market.index[i] + M5_BAR_DURATION,
            "sweep": side, "rolling_activity": bool(side * (close - rolling.iloc[j]) > 0 and activity[j] > 0),
            "anchored_activity": bool(side * (close - anchored) > 0 and activity[j] > 0),
            "anchored_vwap": anchored, "rolling_vwap20": float(rolling.iloc[j]),
            "activity_ratio5_20": float(activity[j]),
        })
    if not records:
        raise RuntimeError("TA_SWEEP_NO_CONFIRMATIONS")
    return pd.DataFrame(records).set_index("signal_bar_start")


def sweep_signal_panel(market: pd.DataFrame, spec: dict) -> pd.DataFrame:
    from gx1.features.smc_v1 import compute_smc_features
    features = market[["high", "low", "close"]].copy()
    features["atr"] = wilder_atr(market.high, market.low, market.close, 14)
    smc = compute_smc_features(features, include_v30_additions=True)
    signals = sweep_confirmations(market, smc.smc_sweep_up_event.to_numpy(),
                                  smc.smc_sweep_down_event.to_numpy(), spec["confirmation_bars"])
    signals = signals.loc[(signals.known_at >= pd.Timestamp(spec["evaluation_start"])) &
                          (signals.known_at < pd.Timestamp(spec["read_end_exclusive"]))].copy()
    daily = _resample_ohlc_for_model_native_scalars(market, "D1")
    units = causal_risk_units(daily.close.to_numpy(), np.ones(len(daily)),
                             initial_equity=spec["initial_equity"], **spec["risk"])
    scale = units * daily.close.to_numpy() / spec["initial_equity"]
    atr_bps = wilder_atr(daily.high, daily.low, daily.close, 14).to_numpy() / daily.close.to_numpy() * 1e4
    ix = (daily.index + TRADING_SESSION_DURATION).searchsorted(pd.DatetimeIndex(signals.known_at), side="right") - 1
    if np.any(ix < 0):
        raise RuntimeError("TA_SWEEP_RISK_CLOCK")
    signals["risk_scale"], signals["atr_bps"] = scale[ix], atr_bps[ix]
    if not np.isfinite(signals[["risk_scale", "atr_bps"]].to_numpy()).all() or (signals.atr_bps <= 0).any():
        raise RuntimeError("TA_SWEEP_RISK_WARMUP")
    return signals


def run_sweep(spec: dict, spec_path: Path) -> dict:
    if (spec["cells"] != ["sweep"] or spec["confirmation_bars"] != 5 or spec["hold_bars"] != 12
            or spec["slippage_scenarios"] != [0., .5, 1., 2.]
            or spec["read_end_exclusive"] != "2026-01-01T00:00:00Z"
            or spec["evaluation_start"] != "2011-01-01T00:00:00Z"
            or spec["inference_start"] != "2021-01-01T00:00:00Z"):
        raise RuntimeError("TA_SWEEP_PREREG_SCOPE")
    out = Path(spec["output_directory"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "STARTED.json", {"git_head": git("rev-parse", "HEAD"),
                                     "preregistration_sha256": sha(spec_path)})
    try:
        curve = load_funding(spec)
        market, binding = load_market(spec, columns=["time", "open", "high", "low", "close", "volume",
                          "bid_open", "ask_open", "bid_close", "ask_close", "ask_low", "bid_high"])
        preparation = spec.get("reuse_preparation")
        if preparation is not None:
            import inspect
            for name, expected in preparation["function_sha256"].items():
                if hashlib.sha256(inspect.getsource(globals()[name]).encode()).hexdigest() != expected:
                    raise RuntimeError("TA_SWEEP_PREPARATION_OWNER_CHANGED")
            for item in preparation["bindings"]:
                if sha(Path(item["path"])) != item["sha256"]:
                    raise RuntimeError("TA_SWEEP_PREPARATION_HASH")
            signals = pd.read_parquet(preparation["signals_path"])
        else:
            signals = sweep_signal_panel(market, spec)
        signals.to_parquet(out / "SIGNALS.parquet")
        cohort, selection = c_select(market, signals, spec, cells=("sweep",))
        for arm in ["rolling_activity", "anchored_activity"]:
            cohort[arm] = signals[arm].reindex(pd.DatetimeIndex(cohort.signal_bar_start)).to_numpy(bool)
        if preparation is not None:
            pd.testing.assert_frame_equal(cohort, pd.read_parquet(preparation["cohort_path"]))
        cohort.to_parquet(out / "COHORT.parquet", index=False)
        tape = c_quotes(market, pd.Timestamp(spec["evaluation_start"]))
        dates = pd.date_range(spec["evaluation_start"], pd.Timestamp(spec["read_end_exclusive"]) - pd.Timedelta(days=1), freq="D")
        later = np.flatnonzero(dates >= pd.Timestamp(spec["inference_start"]))
        counts = cohort.groupby(cohort.known_at.dt.floor("D")).size().reindex(dates, fill_value=0).to_numpy()
        series, portfolios, components, years = {}, {}, {}, {}
        for funding in ["historical_proxy", "zero"]:
            for slip in spec["slippage_scenarios"]:
                for arm in SWEEP_ARMS:
                    key = f"{arm}:s{slip:g}:{funding}"
                    chosen = cohort.copy()
                    if arm in ["rolling_activity", "anchored_activity"]:
                        chosen["executable"] &= chosen[arm]
                    book, outcomes = c_book(tape, chosen, "long" if arm == "long" else "active", slip,
                                           curve if funding == "historical_proxy" else None, spec["initial_equity"])
                    daily = c_daily(book).set_index("time")
                    # Calendar accounting only: before the first actual quote the
                    # book is flat at initial cash. No input quote is fabricated.
                    grid = pd.date_range(dates[0] - pd.Timedelta(days=1), dates[-1], freq="D")
                    daily = daily.reindex(grid)
                    for col in ["mid_pnl", "spread_cost", "slippage_cost", "commission_cost", "financing_cost"]:
                        daily[col] = daily[col].fillna(0.)
                    for col in ["held_units_after", "traded_units", "liquidation_reserve"]:
                        daily[col] = daily[col].ffill().fillna(0.)
                    for col in ["equity_mid", "equity_liquidation"]:
                        daily[col] = daily[col].ffill().fillna(spec["initial_equity"])
                    daily = daily.reset_index(names="time")
                    daily.attrs["initial_equity"] = spec["initial_equity"]
                    if not pd.DatetimeIndex(daily.time.iloc[1:]).equals(dates) or len(outcomes) != len(cohort):
                        raise RuntimeError("TA_SWEEP_MATCHED_POPULATION")
                    daily.to_parquet(out / (key.replace(":", "_") + "_DAILY.parquet"), index=False)
                    outcomes.to_parquet(out / (key.replace(":", "_") + "_OUTCOMES.parquet"), index=False)
                    portfolios[key] = portfolio_summary(daily, periods_per_year=spec["periods_per_year"])
                    portfolios[key]["executed_round_trips"] = int(outcomes.filled.sum())
                    sums = outcomes.groupby(outcomes.known_at.dt.floor("D"))[["net_bps", "normalized_net", "risk_pnl"]].sum().reindex(dates, fill_value=0)
                    series[key] = {"raw": sums.net_bps.to_numpy()[later], "normalized": sums.normalized_net.to_numpy()[later],
                                   "returns": (portfolio_period_returns(daily)[later] if not portfolios[key]["insolvent"]
                                               else np.full(len(later), np.nan)),
                                   "risk_pnl": sums.risk_pnl.to_numpy()[later]}
                    components[key] = {col: float(outcomes[col].mean()) for col in
                                       ["mid_bps", "spread_bps", "slippage_bps", "financing_bps", "net_bps"]}
                    years[key] = {str(y): {"opportunities": len(g), "trades": int(g.filled.sum()),
                                           "mean_net_bps_per_opportunity": float(g.net_bps.mean()),
                                           "risk_pnl": float(g.risk_pnl.sum())}
                                  for y, g in outcomes.groupby(outcomes.known_at.dt.year)}
                    del book, chosen
        comparisons = [{"name": f"anchored_activity:s{slip:g}:{funding}:vs_{baseline}:{metric}",
                        "model": f"anchored_activity:s{slip:g}:{funding}",
                        "baseline": f"{baseline}:s{slip:g}:{funding}", "metric": metric}
                       for funding in ["historical_proxy", "zero"] for slip in spec["slippage_scenarios"]
                       for baseline in ["sweep", "rolling_activity", "long", "flat"] for metric in spec["effects"]]
        later_counts = counts[later]
        point = c_statistics(series, comparisons, later_counts, spec["periods_per_year"], np.arange(len(later)))
        boot = np.array([c_statistics(series, comparisons, later_counts, spec["periods_per_year"], ix)
                         for ix in stationary_bootstrap_indices(len(later), draws=spec["bootstrap_draws"],
                                  mean_block_length=spec["mean_block_length"], seed=spec["seed"])])
        eligible = np.isfinite(point) & np.isfinite(boot).all(axis=0) & (boot.std(axis=0, ddof=1) > 0)
        selected = [c for c, ok in zip(comparisons, eligible) if ok]
        inference = (max_t_inference(point[eligible], boot[:, eligible], names=[c["name"] for c in selected],
                                    alpha=spec["alpha"], desired_power=spec["desired_power"],
                                    effect_sizes=np.array([spec["effects"][c["metric"]] for c in selected]),
                                    minimum_relevant_effect=np.array([spec["effects"][c["metric"]][0] for c in selected]))
                     if selected else {})
        endpoints = {r["name"]: r for r in inference.get("endpoints", [])}
        for c, ok in zip(comparisons, eligible):
            if not ok:
                endpoints[c["name"]] = {"name": c["name"], "effect_verdict": "INKONKLUSIV",
                                        "reason": "undefined_statistic_or_zero_bootstrap_variation"}
        required = [f"anchored_activity:s1:{f}:vs_{b}:mean_net_bps"
                    for f in ["historical_proxy", "zero"] for b in ["sweep", "rolling_activity", "long", "flat"]]
        verdicts = [endpoints[n]["effect_verdict"] for n in required]
        positive = all(series[f"anchored_activity:s1:{f}"]["risk_pnl"].sum() > 0 and
                       not portfolios[f"anchored_activity:s1:{f}"]["insolvent"] for f in ["historical_proxy", "zero"])
        decision = "GO" if all(v == "GO" for v in verdicts) and positive else "NO_GO" if "NO_GO" in verdicts else "INKONKLUSIV"
        np.savez_compressed(out / "BOOTSTRAP_STATISTICS.npz", estimates=point, bootstrap_estimates=boot)
        pd.DataFrame({"day": dates[later], "opportunities": later_counts, **{
            f"{key}:{metric}": val for key, block in series.items() for metric, val in block.items()
        }}).to_parquet(out / "PAIRED_DAYS.parquet", index=False)
        result = {"schema": "gx1_sweep_anchored_activity_research_v1", "status": "COMPLETE",
                  "decision": decision, "required_endpoints": required, "endpoints": list(endpoints.values()),
                  "declared_family_count": len(comparisons), "inference": {k:v for k,v in inference.items() if k != "endpoints"},
                  "portfolios_2011_2025": portfolios, "cost_components_per_opportunity_2011_2025": components,
                  "yearly_results": years, "selection": {**selection, "selected": len(cohort),
                  "later_selected": int(later_counts.sum()), "censored": int(cohort.censored_at_end.sum()),
                  "unexecuted": int((~cohort.executable).sum())},
                  "signal_counts": {"confirmations": len(signals), "rolling_activity": int(signals.rolling_activity.sum()),
                                    "anchored_activity": int(signals.anchored_activity.sum())},
                  "preregistration_sha256": sha(spec_path), "source_commit": git("rev-parse", "HEAD"),
                  "inputs": binding, "test_outcomes_accessed": False, "native_training": False,
                  "dukascopy_used": False, "parameters_fitted": False,
                  "evidence_class": "fixed_rule_reused_development_history_not_untouched_oos",
                  "primary_period": [spec["inference_start"], spec["read_end_exclusive"]],
                  "limitations": spec["limitations"]}
        write_json(out / "RESULT.json", result)
        write_json(out / "TERMINAL.json", {"status": "COMPLETE", "result_sha256": sha(out / "RESULT.json")})
        return {"status": "COMPLETE", "decision": decision, "selection": result["selection"],
                "result": str(out / "RESULT.json"), "sha256": sha(out / "RESULT.json")}
    except Exception as exc:
        write_json(out / "TERMINAL.json", {"status": "FAILED", "error": repr(exc)})
        raise

def _gc_bound_local_path(raw: str, source_root: Path) -> Path:
    path = Path(raw)
    path.relative_to(source_root)
    if (not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents))
            or path.resolve(strict=True) != path or not path.is_file()):
        raise RuntimeError("GC_SOURCE_PATH_NOT_CANONICAL_FILE")
    return path


def audit_gc_source(spec: dict, spec_path: Path) -> dict:
    """Audit hash-bound local vendor files; no fetch, fit or native input change."""
    if spec.get("research_id") != GC_RESEARCH_ID or spec.get("source_audit_schema") != GC_SOURCE_SCHEMA:
        raise RuntimeError("GC_RESEARCH_SPEC_REQUIRED")
    files = spec.get("files")
    if not isinstance(files, list):
        raise RuntimeError("GC_EXPLICIT_FILES_LIST_REQUIRED")
    if not files:
        return {
            "research_id": GC_RESEARCH_ID, "status": "BLOCKED_NO_BOUND_GC_FILES",
            "record_audit_performed": False, "new_downloads": 0, "fits": 0,
            "test_accessed": False, "admitted_to_model_or_economic_test": False,
            "preregistration_sha256": sha(spec_path),
        }
    root = Path(spec["source_root"])
    if (root != GC_LOCAL_SOURCE_ROOT or any(p.is_symlink() for p in (root, *root.parents))
            or root.resolve(strict=True) != root or not root.is_dir()):
        raise RuntimeError("GC_SOURCE_ROOT_NOT_CANONICAL")
    out = Path(spec["output_directory"])
    relative = out.relative_to(GC_AUDIT_OUTPUT_ROOT)
    if (not relative.parts or any(p.is_symlink() for p in (out, *out.parents))
            or out.resolve() != out):
        raise RuntimeError("GC_OUTPUT_PATH_NOT_CANONICAL")
    if out.exists():
        raise FileExistsError("GC_OUTPUT_ALREADY_EXISTS")
    # Validate every declared path before opening any source bytes.
    declared = []
    for raw in files:
        item = require_gc_file_declaration(raw)
        path = _gc_bound_local_path(item["path"], root)
        receipt = _gc_bound_local_path(item["receipt_path"], root)
        if any(path == previous_path for _, previous_path, _ in declared):
            raise RuntimeError("GC_DUPLICATE_SOURCE_FILE")
        declared.append((item, path, receipt))
    for item, path, receipt in declared:
        if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"] or sha(receipt) != item["receipt_sha256"]:
            raise RuntimeError("GC_SOURCE_OR_RECEIPT_HASH_MISMATCH")
    reports = []
    for item, path, receipt in declared:
        opener = gzip.open if item["compression"] == "gzip" else Path.open
        with opener(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            if fields is None or len(fields) != len(set(fields)) or not set(required_gc_fields(item["schema"])).issubset(fields):
                raise RuntimeError("GC_CSV_HEADER_INVALID")
            report = audit_gc_records(reader, item)
        if sha(path) != item["sha256"] or sha(receipt) != item["receipt_sha256"]:
            raise RuntimeError("GC_SOURCE_CHANGED_DURING_AUDIT")
        reports.append({**report, "path": str(path), "sha256": item["sha256"],
                        "receipt_sha256": item["receipt_sha256"]})
    out.mkdir(parents=True, exist_ok=False)
    result = {
        "research_id": GC_RESEARCH_ID, "schema_version": GC_SOURCE_SCHEMA,
        "status": "STRUCTURAL_AUDIT_COMPLETE_NOT_ADMITTED", "files": reports,
        "git_head": git("rev-parse", "HEAD"), "preregistration_sha256": sha(spec_path),
        "record_audit_performed": True, "new_downloads": 0, "fits": 0,
        "test_accessed": False, "admitted_to_model_or_economic_test": False,
    }
    write_json(out / "RESULT.json", result)
    write_json(out / "TERMINAL.json", {
        "status": "COMPLETE_NOT_ADMITTED", "result_sha256": sha(out / "RESULT.json"),
    })
    return result


def entry_edge_gradient_summary(model, weighted_losses):
    """Read-only VJPs on genuine shared and Entry-private parameters.

    Norms/cosines are pre-clipping gradients, not an AdamW update or evidence
    of generalization. Disconnected and connected-zero gradients differ.
    """
    import torch
    from gx1.contracts.entry_observed_market_v1 import is_exit_owned_parameter
    named = [(n, p) for n, p in model.named_parameters()
             if p.requires_grad and not is_exit_owned_parameter(n)
             and not n.startswith("task_log_variances.")]
    groups = {
        "shared": [i for i, (n, _) in enumerate(named)
                   if not n.startswith(("head_", "entry_q_joint_"))],
        "routing": [i for i, (n, _) in enumerate(named)
                    if n.startswith(("family_tf_context_gate.", "family_tf_token_gate."))],
        "entry_head": [i for i, (n, _) in enumerate(named)
                       if n.startswith(("head_entry_action_q.", "entry_q_joint_"))],
    }
    if not all(groups.values()) or any(p.grad is not None for _, p in named):
        raise RuntimeError("ENTRY_EDGE_GRADIENT_SURFACE_INVALID")
    vectors, report = {}, {}
    for task, loss in weighted_losses.items():
        if loss.numel() != 1 or not bool(torch.isfinite(loss)):
            raise RuntimeError("ENTRY_EDGE_GRADIENT_LOSS_INVALID")
        grads = torch.autograd.grad(loss, tuple(p for _, p in named),
                                    retain_graph=True, allow_unused=True)
        vectors[task], report[task] = {}, {}
        for group, indices in groups.items():
            vector = torch.cat([
                grads[i].detach().cpu().double().reshape(-1)
                if grads[i] is not None else torch.zeros(named[i][1].numel(), dtype=torch.float64)
                for i in indices])
            if not bool(torch.isfinite(vector).all()):
                raise RuntimeError("ENTRY_EDGE_NONFINITE_GRADIENT")
            norm = float(torch.linalg.vector_norm(vector))
            connected = sum(grads[i] is not None for i in indices)
            vectors[task][group] = vector
            report[task][group] = {
                "l2_norm": norm, "connected_tensors": connected,
                "status": "nonzero" if norm else "connected_zero" if connected else "unused"}
    alignments = {}
    auxiliaries = [k for k in weighted_losses if k.startswith("aux:")]
    for group in groups:
        entry = vectors["entry"][group]
        contrast = vectors["contrast"][group]
        auxiliary = sum((vectors[k][group] for k in auxiliaries), torch.zeros_like(entry))
        joint = entry + auxiliary
        def cosine(a, b):
            denominator = float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
            return float(torch.dot(a, b)) / denominator if denominator else None
        alignments[group] = {
            "entry_auxiliary_cosine": cosine(entry, auxiliary),
            "contrast_auxiliary_cosine": cosine(contrast, auxiliary),
            "contrast_joint_cosine": cosine(contrast, joint),
            "entry_norm": float(torch.linalg.vector_norm(entry)),
            "auxiliary_sum_norm": float(torch.linalg.vector_norm(auxiliary)),
            "contrast_dot_joint": float(torch.dot(contrast, joint)),
            "per_auxiliary_contrast_cosine": {
                k: cosine(contrast, vectors[k][group]) for k in auxiliaries},
        }
    if any(p.grad is not None for _, p in named):
        raise RuntimeError("ENTRY_EDGE_GRADIENT_ACCUMULATED")
    return {
        "groups": {k: [named[i][0] for i in indices] for k, indices in groups.items()},
        "gradients": report, "alignment": alignments,
        "routing_is_subset_of_shared": True,
        "semantics": "actual weighted task VJPs; pre-clipping, not AdamW or causal intervention"}


def _entry_edge_begin(spec, mode, spec_path):
    from gx1.contracts.gx1_capped_execution_v1 import (
        _require_capped_cgroup_limits, require_capped_lock_ancestry)
    import signal
    import socket
    if os.environ.get("GX1_CAPPED_CLASS") != "producer" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("ENTRY_EDGE_CAPPED_CPU_PRODUCER_REQUIRED")
    capped = _require_capped_cgroup_limits(
        environ=os.environ, read_text=None, max_memory_bytes=spec["memory_max_bytes"],
        error_prefix="ENTRY_EDGE_CPU")
    capped["lock"] = require_capped_lock_ancestry()
    policy = json.loads((ROOT / "NEXT_RUN_POLICY.json").read_text())
    scope_key = "entry_learning_study_20261010" if mode == "entry-learning-baseline" else "entry_edge_20261010"
    scope = policy["current_work"][scope_key]
    if (scope["authorized_mode"] != mode or scope["authorized"] is not True
            or scope["spec"] != {"path": str(spec_path.resolve()), "sha256": sha(spec_path)}):
        raise RuntimeError("ENTRY_EDGE_MODE_NOT_AUTHORIZED")
    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "CLAIM.json", {"source_commit": git("rev-parse", "HEAD"),
                                  "pid": os.getpid(), "mode": mode})
    def timeout(_sig, _frame):
        raise RuntimeError("ENTRY_EDGE_WALL_LIMIT")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(spec["max_wall_seconds"])
    def deny(*args, **kwargs):
        raise RuntimeError("ENTRY_EDGE_NETWORK_FORBIDDEN")
    socket.socket.connect = deny
    socket.socket.connect_ex = deny
    socket.create_connection = deny
    def audit(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes)):
            name = os.fsdecode(args[0]).lower()
            if name.endswith(("_test.parquet", "_test.manifest.json")) or "/exit_lifecycle/test_" in name:
                raise RuntimeError("ENTRY_EDGE_TEST_FORBIDDEN")
    sys.addaudithook(audit)
    return out, capped


def _entry_edge_bound(binding):
    path = Path(binding["path"])
    if not path.is_absolute() or not path.is_file() or path.is_symlink() or sha(path) != binding["sha256"]:
        raise RuntimeError("ENTRY_EDGE_BOUND_FILE_CHANGED: " + str(path))
    return path


def run_entry_edge_diagnostic(spec, spec_path):
    """Observe initial/final Entry and all auxiliaries on frozen TRAIN batches."""
    import time
    import resource
    import torch
    from gx1.contracts.entry_observed_market_v1 import (
        require_entry_observed_market_scope, bind_entry_observed_market_dataset,
        entry_observed_market_batch_targets)
    out, capped = _entry_edge_begin(spec, "entry-edge-diagnostic", spec_path)
    started, head = time.monotonic(), git("rev-parse", "HEAD")
    try:
        recipe = json.loads(_entry_edge_bound(spec["recipe"]).read_text())
        for key, value in recipe["recipe_env"].items():
            os.environ[key] = value
        from gx1.scripts import run_unified_exit_random_access_full_train_v1 as native
        trainer, val = native.trainer, native.val
        scope = require_entry_observed_market_scope(recipe)
        files = recipe["files"]
        meta = json.loads(_entry_edge_bound(files["source_bundle_metadata"]).read_text())
        norm = json.loads(_entry_edge_bound(meta["v38_input_sources"]["normalization"]).read_text())["contract"]
        val._bind_multi_tf_cache_from_source_bundle_metadata(meta)
        coordinates = json.loads(_entry_edge_bound(recipe["chronological_prefix"]["native_coordinates"]).read_text())
        batch_size = recipe["trainer_cli"]["batch_size"]
        if batch_size != 16:
            raise RuntimeError("ENTRY_EDGE_ORIGINAL_BATCH_REQUIRED")
        parent_rows = {
            name: np.load(_entry_edge_bound(coordinates["bindings"][binding]), allow_pickle=False)
            for name, binding in {
                "frozen_probe": "TRAIN256_PROBE_PARENT_ROWS",
                "smoke": "TRAIN_NATIVE4096_PARENT_ROWS"}.items()}
        selections = {
            "frozen_probe_first_batch": parent_rows["frozen_probe"][:batch_size],
            "smoke_first_batch": parent_rows["smoke"][:batch_size],
            "smoke_last_batch": parent_rows["smoke"][-batch_size:]}
        dataset = val.EntryV10CtxDataset(
            _entry_edge_bound(files["entry_train_parquet"]), seq_len=int(meta["seq_len"]),
            m5_prebuilt_path=_entry_edge_bound(files["m5_prebuilt"]),
            per_tf_seq_lens={tf.upper(): int(meta["multi_tf"][tf + "_seq_len"])
                            for tf in ("m5", "m15", "h1", "h4", "d1")},
            multi_tf_closed_bar=True,
            sequence_source_audit_json=_entry_edge_bound(files["sequence_source_audit"]))
        prefix = recipe["chronological_prefix"]
        dataset.bind_policy_dependent_auxiliary_targets(
            result_path=_entry_edge_bound(prefix["labels_result"]),
            expected_result_sha256=prefix["labels_result"]["sha256"],
            expected_design_sha256=prefix["design"]["sha256"], role="TRAIN")
        bind_entry_observed_market_dataset(dataset, scope)
        batches = {}
        for label, rows in selections.items():
            batches[label] = next(iter(val.DataLoader(
                dataset, batch_size=batch_size, sampler=val._ExactSampler(rows.tolist()),
                num_workers=0, generator=torch.Generator().manual_seed(recipe["trainer_cli"]["seed"]))))
            if batches[label]["entry_row_index"].tolist() != rows.tolist():
                raise RuntimeError("ENTRY_EDGE_BATCH_COORDINATE_MISMATCH")
        cache = out / "TRAIN_BATCHES.pt"
        temporary = out / "TRAIN_BATCHES.pt.part"
        with temporary.open("xb") as handle:
            torch.save({"batches": batches, "selections": selections,
                        "target_identity": dataset._entry_observed_market_binding}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_file_noreplace(temporary, cache)
        _fsync_directory(out)
        print(json.dumps({"event": "ENTRY_EDGE_BATCHES_BOUND", "path": str(cache)}), flush=True)
        del dataset
        state_path = _entry_edge_bound(spec["training_state"])
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["global_optimizer_steps"] != 256:
            raise RuntimeError("ENTRY_EDGE_ORIGINAL_STATE_MISMATCH")
        device = torch.device("cpu")
        model = val._model(meta, norm, device)
        model.requires_grad_(True)
        model.train()
        measurements = {}
        for variant, state_key in (("initial", "target_model_state"), ("final", "model_state")):
            model.load_state_dict(state[state_key], strict=True)
            before = trainer._model_state_sha256(model)
            measurements[variant] = {}
            for label, batch in batches.items():
                trainer._set_deterministic(recipe["trainer_cli"]["seed"], device, recipe["trainer_cli"]["precision_policy"])
                target, valid = entry_observed_market_batch_targets(batch, device=device)
                prediction = trainer._model_forward_fp32(
                    model, batch["seq_x"], batch["snap_x"], ctx_cat=batch["ctx_cat"],
                    ctx_cont=batch["ctx_cont"], **trainer._multi_tf_kwargs_from_batch(batch, device),
                    **spec["gradient_boundary"])
                q = prediction["entry_action_q_bps"]
                parts = native._entry_signal_losses(q, target, valid)
                precision = torch.exp(-model.task_log_variances["entry_action_q"])
                weighted = {k: precision * v for k, v in parts.items()}
                weighted["entry"] = sum(weighted.values())
                aux = trainer.dip_forecast_task_losses(prediction, batch, device)
                aux["side_mae_bps"] = trainer._side_mae_auxiliary_loss(prediction, batch, device)[0]
                aux["trendline_event"] = trainer._trendline_event_aux_loss(prediction, batch, device)[0]
                position = trainer._require_active_aux_head_prediction(
                    prediction, batch, output_name="position_size_logit",
                    target_names=("y_position_size_target", "y_position_size_mask"))
                mask = batch["y_position_size_mask"]
                if bool((mask.reshape(-1) == 1).any()):
                    aux["position_size"] = trainer._masked_position_size_mse(
                        position, batch["y_position_size_target"], mask)
                for task, value in aux.items():
                    weighted["aux:" + task] = torch.exp(-model.task_log_variances[task]) * value
                result = entry_edge_gradient_summary(model, weighted)
                result.update(
                    model_state_sha256=before, parent_rows=selections[label].tolist(),
                    raw_entry_mse_decomposition={k: float(v.detach()) for k, v in parts.items()},
                    raw_auxiliary_losses={k: float(v.detach()) for k, v in aux.items()},
                    task_precision={k: float(torch.exp(-v.detach())) for k, v in model.task_log_variances.items()},
                    target_mean_bps=target.double().mean(0).tolist(),
                    prediction_mean_bps=q.detach().double().mean(0).tolist(),
                    prediction_std_bps=q.detach().double().std(0, unbiased=False).tolist())
                measurements[variant][label] = result
                write_json(out / (variant + "_" + label + ".json"), result)
                print(json.dumps({"event": "ENTRY_EDGE_GRADIENT_COMPLETE", "variant": variant,
                                  "batch": label, "elapsed_seconds": time.monotonic()-started}), flush=True)
                del prediction, q, parts, weighted, aux, result, position, precision
                if trainer._model_state_sha256(model) != before:
                    raise RuntimeError("ENTRY_EDGE_MODEL_MUTATED")
        if git("rev-parse", "HEAD") != head or git("status", "--porcelain"):
            raise RuntimeError("ENTRY_EDGE_SOURCE_CHANGED")
        _entry_edge_bound(spec["training_state"])
        result = {
            "schema_version": "gx1_entry_edge_gradient_diagnostic_v1", "status": "COMPLETE",
            "source_commit": head, "spec": {"path": str(spec_path), "sha256": sha(spec_path)},
            "measurements": measurements, "input_cache": {"path": str(cache), "sha256": sha(cache)},
            "model_forwards": len(selections)*2, "optimizer_steps": 0, "new_fits": 0,
            "control_forwards": 0, "test_accessed": False, "original_checkpoint_preserved": True,
            "mode": "CPU deterministic FP32 train-mode with same declared seed per paired batch",
            "limitations": "Three preselected TRAIN batches, paired initial/final states. Pre-clipping task gradients; not reconstructed CUDA dropout, AdamW updates, intervention, generalization or economics.",
            "elapsed_seconds": time.monotonic()-started,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "capped_execution": capped}
        write_json(out / "RESULT.json", result)
        write_json(out / "TERMINAL.json", {"exit_code": 0, "result": {"path": str(out / "RESULT.json"), "sha256": sha(out / "RESULT.json")},
                                         "source_unchanged": True})
        return {k: v for k, v in result.items() if k != "measurements"}
    except BaseException as exc:
        write_json(out / "TERMINAL.json", {"exit_code": 1, "error_type": type(exc).__name__,
                                         "error": str(exc), "source_commit": head})
        raise


def entry_edge_features(raw):
    """One declared smooth interaction hypothesis; no fitted transform here."""
    atr = raw["ctx_cont.atr_bps"].to_numpy(float)
    if not np.isfinite(raw.to_numpy(float)).all() or np.any(atr <= 0):
        raise RuntimeError("ENTRY_EDGE_FEATURE_COVERAGE_INVALID")
    reaction = {
        "return3_atr": np.arctan(raw["ctx_cont.close_return_3_bps"].to_numpy(float) / atr),
        "acceleration1_atr": np.arctan(raw["ctx_cont.close_return_acceleration_1_bps"].to_numpy(float) / atr),
        "wick_balance": raw["candle.raw_lower_wick_share"].to_numpy(float) - raw["candle.raw_upper_wick_share"].to_numpy(float),
    }
    context = {
        name: np.arctan(raw[name].to_numpy(float)) for name in (
            "level_above_dist_atr", "level_below_dist_atr",
            "smc_sweep_up_avwap_dist_atr", "smc_sweep_down_avwap_dist_atr",
            "ctx_cont._v1h4_ema_diff", "ctx_cont.d1_ema_slope_20_canon_v2")}
    context["log_atr_bps"] = np.log(atr)
    context["spread_atr"] = np.arctan(raw["ctx_cont.spread_bps"].to_numpy(float) / atr)
    base = {**reaction, **context}
    for name in ("level_below_present", "smc_sweep_up_avwap_age_bars", "smc_sweep_down_avwap_age_bars"):
        base[name] = np.arctan(raw[name].to_numpy(float))
    expanded = {**base, **{r + "*" + c: rv * cv
                          for r, rv in reaction.items() for c, cv in context.items()}}
    return pd.DataFrame(base), pd.DataFrame(expanded)


def entry_edge_fold_rows(clock, start, end, elapsed_seconds, train_start):
    decision = clock + pd.Timedelta(minutes=5)
    outcome_end = decision + pd.Timedelta(seconds=elapsed_seconds)
    fit = np.flatnonzero((clock >= train_start) & (outcome_end < start))
    hold = np.flatnonzero((decision >= start) & (outcome_end < end))
    if not len(fit) or not len(hold) or outcome_end[fit[-1]] >= decision[hold[0]]:
        raise RuntimeError("ENTRY_EDGE_CHRONOLOGY_INVALID")
    return fit, hold


def entry_edge_actions(prediction):
    """Unique positive argmax, FLAT on zero or any LONG/SHORT tie."""
    p = np.asarray(prediction, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise RuntimeError("ENTRY_EDGE_PREDICTION_INVALID")
    return np.where((p[:, 0] > 0) & (p[:, 0] > p[:, 1]), 1,
                    np.where((p[:, 1] > 0) & (p[:, 1] > p[:, 0]), -1, 0))


def _entry_edge_portfolios(frame, spec, scope, out):
    """Single-position research cash ledgers. The 95min label is not an exit rule."""
    start = pd.Timestamp(frame.decision_time.iloc[0])
    end = pd.Timestamp(spec["train_end_exclusive"])
    m1 = pq.read_table(
        str(_entry_edge_bound(spec["m1_source"])),
        columns=["time", "open", "bid_open", "ask_open"],
        filters=[("time", ">=", start.to_pydatetime()),
                 ("time", "<", end.to_pydatetime())]).to_pandas()
    times = pd.DatetimeIndex(m1.time)
    decisions = pd.DatetimeIndex(frame.decision_time)
    if (not times.is_monotonic_increasing or times.has_duplicates or len(times) < 2
            or times[0] != start or times[-1] >= end):
        raise RuntimeError("ENTRY_EDGE_PORTFOLIO_CLOCK_INVALID")
    indices = times.get_indexer(decisions)
    if np.any(indices < 0):
        raise RuntimeError("ENTRY_EDGE_EXACT_M1_FILL_MISSING")
    tape = Tape(times, m1.open.to_numpy(float), m1.bid_open.to_numpy(float),
                m1.ask_open.to_numpy(float), spec["m1_source"]["sha256"], spec["m1_source"]["path"])
    # One fixed physical unit for every active position and always-LONG.
    # Initial capital equals the first executable ask, not later strategy risk.
    capital = float(tape.ask[0])
    rates = np.asarray(scope["economics"]["annual_financing_cost_bps"]) / 1e4
    curve = ResearchFinancingCurve(
        pd.DatetimeIndex([times[0]]), np.array([(rates[0] - rates[1]) / 2]),
        end, float((rates[0] + rates[1]) / 2), scope["economics"]["seconds_per_year"])
    authority = json.loads(_entry_edge_bound(spec["cost_authority"]).read_text())["parameters"]
    summaries, monthly = {}, {}
    previous_decision = np.searchsorted(indices, np.arange(len(times)-1), side="right") - 1
    if np.any(previous_decision < 0):
        raise RuntimeError("ENTRY_EDGE_POSITION_BEFORE_DECISION")
    for arm in ("constant", "additive", "interaction", "always_long", "flat"):
        if arm in ("always_long", "flat"):
            side = np.full(len(frame), 1 if arm == "always_long" else 0, dtype=np.int8)
        else:
            side = frame[arm + "_action"].to_numpy(np.int8)
        units = side[previous_decision].astype(float)
        path = portfolio_path(
            tape, units, initial_equity=capital,
            slippage_bps_per_execution=authority["execution_slippage"]["central_bps_per_execution"],
            commission_bps_per_execution=authority["commission"]["bps_per_execution"],
            financing_curve=curve, liquidate_at_end=False)
        summary = portfolio_summary(path, periods_per_year=(len(times)-1) * scope["economics"]["seconds_per_year"] / (times[-1]-times[0]).total_seconds())
        for statistic in ("realized_annual_vol", "sharpe_zero_cash"):
            summary.pop(statistic, None)
        equity = path.equity_liquidation.to_numpy(float)
        increments = np.diff(np.concatenate([[capital], equity])) / capital * 1e4
        months = times.strftime("%Y-%m")
        monthly[arm] = pd.Series(increments).groupby(months, sort=True).sum()
        before = np.concatenate([[0], units[:-1]])
        openings = (units != 0) & ((before == 0) | (before != units))
        summary.update(
            position_entries=int(openings.sum()),
            active_minute_fraction=float(np.mean(units != 0)),
            fixed_physical_units=1.0, initial_capital=capital,
            max_open_notional_over_initial_capital=float(path.opening_notional_after.max() / capital),
            open_position_included=True, cost_assumption="bound current-terms scenario, not historical broker truth",
            annualized_statistics_scope="Sharpe/annualized volatility omitted for irregular M1 marks",
            economics_scope="continuous M1-marked research Entry argmax policy, not native Exit")
        summaries[arm] = summary
        write_parquet(out / ("PORTFOLIO_" + arm + ".parquet"), path)
        del path
    return summaries, pd.DataFrame(monthly)


def _entry_edge_monthly_intervals(values, spec):
    """Reuse paired stationary bootstrap; intervals are conditional development estimates."""
    X = values.to_numpy(float)
    if not np.isfinite(X).all() or len(X) < spec["mean_block_months"]:
        raise RuntimeError("ENTRY_EDGE_MONTHLY_INFERENCE_INVALID")
    boot = np.array([X[i].mean(axis=0) for i in stationary_bootstrap_indices(
        len(X), draws=spec["bootstrap_draws"], mean_block_length=spec["mean_block_months"],
        seed=spec["seed"])])
    # Bonferroni percentile intervals across the explicitly declared endpoint family.
    tail = spec["alpha"] / (2 * X.shape[1])
    bounds = np.quantile(boot, [tail, 1 - tail], axis=0)
    return {name: {"equal_month_mean": float(X[:, i].mean()),
                   "family_adjusted_lower": float(bounds[0, i]),
                   "family_adjusted_upper": float(bounds[1, i]),
                   "months": len(X)}
            for i, name in enumerate(values.columns)}


def run_entry_edge_probe(spec, spec_path):
    """Two fixed ridge representations on the same purged TRAIN folds and cost targets."""
    import time
    import resource
    import torch
    from gx1.contracts.entry_observed_market_v1 import (
        GROSS_TARGET_COLUMNS, require_entry_observed_market_scope, build_entry_observed_market_targets)
    from gx1.scripts.research_entry_direction_walkforward_v1 import (
        _list_column_to_matrix, feature_fit_lineage, require_feature_fit_before)
    out, capped = _entry_edge_begin(spec, "entry-edge-probe", spec_path)
    started, head = time.monotonic(), git("rev-parse", "HEAD")
    try:
        recipe = json.loads(_entry_edge_bound(spec["recipe"]).read_text())
        scope = require_entry_observed_market_scope(recipe)
        source = scope["physical_sources"]["train"]
        if (scope["elapsed_wall_clock_seconds"] != spec["elapsed_seconds"]
                or source["parquet"] != spec["train_parquet"] or source["manifest"] != spec["train_manifest"]
                or recipe["files"]["train_cost_authority"] != spec["cost_authority"]
                or scope["target_m1_source_sha256"] != spec["m1_source"]["sha256"]):
            raise RuntimeError("ENTRY_EDGE_PROBE_TARGET_BINDING")
        manifest = json.loads(_entry_edge_bound(spec["train_manifest"]).read_text())
        lineage = feature_fit_lineage(manifest["extra"]["multi_tf_cache_binding"])
        table = pq.read_table(str(_entry_edge_bound(spec["train_parquet"])),
                              columns=["time", "snap", *GROSS_TARGET_COLUMNS])
        clock = pd.DatetimeIndex(pd.to_datetime(table.column("time").to_pandas(), utc=True)).as_unit("ns")
        if (len(clock) != source["physical_rows"] or clock.has_duplicates
                or not clock.is_monotonic_increasing or
                hashlib.sha256(np.asarray(clock.asi8, dtype="<i8").tobytes()).hexdigest() != source["clock_sha256"]):
            raise RuntimeError("ENTRY_EDGE_PROBE_INPUT_CLOCK")
        names = manifest["feature_contract"]["signal_bridge_fields"]
        snap = _list_column_to_matrix(table, "snap", len(names), np.float64)
        raw = pd.DataFrame(snap[:, [names.index(n) for n in spec["raw_feature_names"]]],
                           columns=spec["raw_feature_names"])
        gross = np.column_stack([table.column(n).to_numpy() for n in GROSS_TARGET_COLUMNS]).astype(np.float32)
        targets = build_entry_observed_market_targets(
            gross_return_bps=torch.from_numpy(gross),
            elapsed_wall_clock_seconds=torch.full((len(clock),), spec["elapsed_seconds"], dtype=torch.int64),
            economics=scope["economics"])["target_bps"].numpy()[:, :2].astype(float)
        del table, snap, gross
        additive, interaction = entry_edge_features(raw)
        if [len(additive.columns), len(interaction.columns)] != spec["feature_widths"]:
            raise RuntimeError("ENTRY_EDGE_PREREG_FEATURE_WIDTH")
        coverage = {name: {"finite_rows": int(np.isfinite(raw[name]).sum()),
                           "nonzero_rows": int((raw[name] != 0).sum())}
                    for name in raw}
        write_json(out / "INPUT_ACCEPTANCE.json", {
            "rows": len(clock), "feature_coverage": coverage, "feature_fit_lineage": lineage,
            "feature_names": {"additive": list(additive), "interaction": list(interaction)},
            "native_features_changed": False, "test_accessed": False})
        forecasts = {n: np.full((len(clock), 2), np.nan) for n in ("constant", "additive", "interaction")}
        fold_id = np.full(len(clock), -1, dtype=int)
        fits = []
        for number, bounds in enumerate(spec["folds"]):
            start, end = map(pd.Timestamp, bounds)
            require_feature_fit_before(lineage, start, context="Entry-edge outer fold")
            fit, hold = entry_edge_fold_rows(
                clock, start, end, spec["elapsed_seconds"], pd.Timestamp(spec["train_start"]))
            if len(fit) < spec["minimum_fit_rows"] or np.any(fold_id[hold] != -1):
                raise RuntimeError("ENTRY_EDGE_PROBE_FOLD_INVALID")
            row = {"fold": number, "start": str(start), "end": str(end),
                   "fit_rows": len(fit), "hold_rows": len(hold),
                   "last_fit_outcome_time": str(clock[fit[-1]] + pd.Timedelta(seconds=300+spec["elapsed_seconds"])),
                   "first_hold_decision_time": str(clock[hold[0]] + pd.Timedelta(minutes=5)),
                   "fit_rows_sha256": hashlib.sha256(fit.astype("<i8").tobytes()).hexdigest(),
                   "hold_rows_sha256": hashlib.sha256(hold.astype("<i8").tobytes()).hexdigest()}
            forecasts["constant"][hold] = targets[fit].mean(axis=0)
            for arm, features in (("additive", additive), ("interaction", interaction)):
                X = features.to_numpy(float)
                gram = RidgeGram(X[fit], X[hold], inner_fraction=spec["inner_fraction"],
                                 fit_positions=fit, purge_bars=spec["elapsed_seconds"]//300,
                                 constant_alternative=False)
                row[arm] = {}
                for side in range(2):
                    # alpha=N corresponds to unit L2 penalty in mean-squared-error units
                    prediction, info = gram.fit_predict(targets[fit, side], alpha=float(len(fit)))
                    forecasts[arm][hold, side] = prediction
                    row[arm][str(side)] = info
                del gram
            fold_id[hold] = number
            fits.append(row)
            write_json(out / ("FOLD_" + str(number) + ".json"), row)
            print(json.dumps({"event": "ENTRY_EDGE_FOLD_COMPLETE", "fold": number,
                              "elapsed_seconds": time.monotonic()-started}), flush=True)
        selected = np.flatnonzero(fold_id >= 0)
        frame = pd.DataFrame({"parent_row": selected, "decision_time": clock[selected] + pd.Timedelta(minutes=5),
                              "fold": fold_id[selected], "target_long": targets[selected, 0],
                              "target_short": targets[selected, 1]})
        metrics = {}
        for arm, full in forecasts.items():
            prediction = full[selected]
            actions = entry_edge_actions(prediction)
            frame[arm+"_long"], frame[arm+"_short"] = prediction[:, 0], prediction[:, 1]
            frame[arm+"_action"] = actions
            diff = prediction[:, 0] - prediction[:, 1]
            truth = targets[selected, 0] - targets[selected, 1]
            frame[arm+"_contrast_mse"] = (diff - truth)**2
            frame[arm+"_target_markout"] = np.where(actions == 1, targets[selected, 0],
                                                     np.where(actions == -1, targets[selected, 1], 0.0))
            metrics[arm] = {
                "rows": len(selected), "contrast_mse": float(np.mean((diff-truth)**2)),
                "side_mse": np.mean((prediction-targets[selected])**2, axis=0).tolist(),
                "prediction_contrast_std_bps": float(np.std(diff)),
                "contrast_correlation": float(np.corrcoef(diff, truth)[0, 1]) if np.std(diff) else None,
                "long_rows": int((actions == 1).sum()), "short_rows": int((actions == -1).sum()),
                "flat_rows": int((actions == 0).sum()),
                "overlapping_target_markout_mean_bps": float(frame[arm+"_target_markout"].mean()),
                "overlapping_target_markout_is_portfolio": False}
        write_parquet(out / "PREDICTIONS.parquet", frame)
        summaries, economics = _entry_edge_portfolios(frame, spec, scope, out)
        month = pd.DatetimeIndex(frame.decision_time).strftime("%Y-%m")
        monthly = pd.DataFrame(index=economics.index)
        for model, base in (("interaction", "constant"), ("interaction", "additive"), ("additive", "constant")):
            monthly[model+"_contrast_improvement_vs_"+base] = (
                frame[base+"_contrast_mse"] - frame[model+"_contrast_mse"]).groupby(month).mean()
        for model in ("additive", "interaction"):
            for base in ("flat", "always_long"):
                monthly[model+"_net_bps_vs_"+base] = economics[model] - economics[base]
        intervals = _entry_edge_monthly_intervals(monthly, spec)
        write_parquet(out / "MONTHLY_COMPARISONS.parquet", monthly.reset_index(names="month"))
        positive = {arm: all(intervals[key]["family_adjusted_lower"] > 0 for key in (
            arm+"_contrast_improvement_vs_constant", arm+"_net_bps_vs_flat", arm+"_net_bps_vs_always_long"))
                    for arm in ("additive", "interaction")}
        for arm in positive:
            positive[arm] = positive[arm] and not summaries[arm]["insolvent"]
        result = {
            "schema_version": "gx1_entry_edge_train_probe_v1", "status": "COMPLETE",
            "source_commit": head, "spec": {"path": str(spec_path), "sha256": sha(spec_path)},
            "feature_fit_lineage": lineage, "fits": fits, "prediction_metrics": metrics,
            "per_fold_prediction_metrics": {str(f): {arm: {"contrast_mse": float(g[arm+"_contrast_mse"].mean()),
                "long_rows": int((g[arm+"_action"] == 1).sum()), "short_rows": int((g[arm+"_action"] == -1).sum()),
                "flat_rows": int((g[arm+"_action"] == 0).sum())} for arm in forecasts}
                for f, g in frame.groupby("fold")},
            "portfolio_metrics": summaries, "monthly_intervals": intervals,
            "development_screen_pass": positive,
            "native_training_admitted": False, "test_accessed": False,
            "control_rows": 0, "optimizer_steps": 0, "cpu_ridge_fits": 4*len(fits),
            "target_horizon_seconds": spec["elapsed_seconds"], "economic_holding_timeout": None,
            "limitations": spec["limitations"], "elapsed_seconds": time.monotonic()-started,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "capped_execution": capped}
        if git("rev-parse", "HEAD") != head or git("status", "--porcelain"):
            raise RuntimeError("ENTRY_EDGE_SOURCE_CHANGED")
        write_json(out / "RESULT.json", result)
        write_json(out / "TERMINAL.json", {"exit_code": 0, "source_unchanged": True,
                   "result": {"path": str(out/"RESULT.json"), "sha256": sha(out/"RESULT.json")}})
        return {"status": "COMPLETE", "output": str(out), "development_screen_pass": positive}
    except BaseException as exc:
        write_json(out / "TERMINAL.json", {"exit_code": 1, "error_type": type(exc).__name__,
                                         "error": str(exc), "source_commit": head})
        raise





def entry_learning_prediction_metrics(prediction, target):
    """Prediction evidence only: signed outcomes and all rows, never portfolio PnL."""
    q,y=np.asarray(prediction,dtype=float),np.asarray(target,dtype=float)
    if (q.shape!=y.shape or q.ndim!=2 or q.shape[1]!=3 or len(q)==0
            or not np.isfinite(q).all() or not np.isfinite(y).all() or np.any(y[:,2]!=0)):
        raise RuntimeError("ENTRY_STUDY_REVIEW_ARRAY_INVALID")
    p=np.column_stack([q[:,:2],q[:,0]-q[:,1]])
    truth=np.column_stack([y[:,:2],y[:,0]-y[:,1]])
    error=p-truth
    unique=(q==q.max(axis=1,keepdims=True)).sum(axis=1)==1
    action=np.argmax(q,axis=1)
    action[~unique]=2
    correlations=[float(np.corrcoef(p[:,i],truth[:,i])[0,1])
                  if np.std(p[:,i])>0 and np.std(truth[:,i])>0 else None for i in range(3)]
    chosen=y[np.arange(len(y)),action]
    return {"rows":len(y),"mse":np.mean(error**2,axis=0).tolist(),
            "centered_mse":np.var(error,axis=0).tolist(),"bias":np.mean(error,axis=0).tolist(),
            "prediction_std":np.std(p,axis=0).tolist(),"target_std":np.std(truth,axis=0).tolist(),
            "correlation":correlations,"action_counts":np.bincount(action,minlength=3).tolist(),
            "ties":int((~unique).sum()),"nonconstant_unique_actions":len(np.unique(action[unique]))>1,
            "constant_prediction":bool(np.all(np.std(p,axis=0)==0)),
            "positive_estimated_entry_rows":int(((action!=2)&unique&(q[:,:2].max(axis=1)>0)).sum()),
            "overlapping_selected_target_mean_bps":float(chosen.mean()),
            "economic_rollout":False}


def entry_learning_paired_review(*,initial,final,target,constant,time_ns,review):
    """Fixed paired week bootstrap across the six predeclared comparisons."""
    y=np.asarray(target,dtype=float)
    constant=np.broadcast_to(np.asarray(constant,dtype=float),y.shape)
    forecasts={"initial":np.asarray(initial,dtype=float),"final":np.asarray(final,dtype=float),"constant":constant}
    metrics={k:entry_learning_prediction_metrics(q,y) for k,q in forecasts.items()}
    clock=pd.DatetimeIndex(pd.to_datetime(time_ns,unit="ns",utc=True))
    if len(clock)!=len(y) or clock.has_duplicates or not clock.is_monotonic_increasing:
        raise RuntimeError("ENTRY_STUDY_REVIEW_CLOCK")
    truth=np.column_stack([y[:,:2],y[:,0]-y[:,1]])
    squared={k:(np.column_stack([q[:,:2],q[:,0]-q[:,1]])-truth)**2 for k,q in forecasts.items()}
    diff=np.concatenate([squared["final"]-squared[k] for k in ("initial","constant")],axis=1)
    week=(clock.tz_localize(None).to_period("W-SUN").start_time).astype(str)
    weekly=pd.DataFrame(diff,columns=review["primary_learning_comparisons"]).groupby(week,sort=True).mean()
    if len(weekly)<review["expected_block_weeks"] or review["multiplicity_comparisons"]!=diff.shape[1]:
        raise RuntimeError("ENTRY_STUDY_REVIEW_INFERENCE_SUPPORT")
    boot=np.asarray([weekly.to_numpy()[indices].mean(axis=0)
        for indices in stationary_bootstrap_indices(len(weekly),
            draws=review["bootstrap_replicates"],mean_block_length=review["expected_block_weeks"],
            seed=review["bootstrap_seed"])])
    tail=review["familywise_alpha"]/(2*review["multiplicity_comparisons"])
    lower,upper=np.quantile(boot,[tail,1-tail],axis=0)
    intervals={name:{"equal_week_mean":float(weekly[name].mean()),"family_adjusted_lower":float(lower[i]),
                     "family_adjusted_upper":float(upper[i]),"weeks":len(weekly)}
               for i,name in enumerate(weekly)}
    numerical=bool(np.all(upper<0))
    centered=all(np.all(np.asarray(metrics["final"]["centered_mse"])<metrics[k]["centered_mse"])
                 for k in ("initial","constant"))
    action=bool(metrics["final"]["nonconstant_unique_actions"] and metrics["final"]["positive_estimated_entry_rows"]>0)
    month=clock.strftime("%Y-%m")
    monthly={period:{key:entry_learning_prediction_metrics(q[month==period],y[month==period])
                     for key,q in forecasts.items()} for period in sorted(set(month))}
    return {"metrics":metrics,"paired_intervals":intervals,"monthly":monthly,
            "numerical_gate":numerical,"centered_gate":centered,"action_gate":action,
            "statistical_screen_pass":numerical and centered and action,
            "period_and_operational_review_still_required":True,
            "control_role":review["control_role"],"test_data_used":False,"profitability_proven":False}

def run_entry_learning_baseline(spec, spec_path):
    """Two fixed, purged HGB fits on the native curve's identical TRAIN rows."""
    import time
    import resource
    import torch
    from gx1.contracts.entry_observed_market_v1 import (
        GROSS_TARGET_COLUMNS, require_entry_observed_market_scope, build_entry_observed_market_targets)
    from gx1.scripts.research_entry_direction_walkforward_v1 import (
        _list_column_to_matrix, feature_fit_lineage, require_feature_fit_before, _inner_split)
    baseline = spec["baseline"]
    if (spec["schema_version"] != "gx1_entry_learning_study_v1"
            or baseline["fits"] != 2 or baseline["full_fit_rows"] != 262144
            or baseline["max_iter"] != 100 or baseline["purge_bars"] != 19
            or baseline["inputs"] != "all_254_raw_snapshot_fields"
            or spec["execution"]["test_data_used"] is not False):
        raise RuntimeError("ENTRY_STUDY_BASELINE_BUDGET_INVALID")
    run = {**spec, "output":str(Path(spec["root"])/"BASELINE"),
           "memory_max_bytes":spec["execution"]["cpu_memory_max_bytes"],
           "max_wall_seconds":baseline["cpu_wall_seconds"]}
    out, capped = _entry_edge_begin(run, "entry-learning-baseline", spec_path)
    started, head = time.monotonic(), git("rev-parse","HEAD")
    try:
        scope = require_entry_observed_market_scope(spec)
        rows = {key:np.load(_entry_edge_bound(binding), allow_pickle=False)
                for key,binding in spec["row_bindings"].items()}
        fit = np.sort(rows["epoch_order"][:baseline["full_fit_rows"]])
        if (len(np.unique(fit)) != baseline["full_fit_rows"]
                or not np.array_equal(np.sort(rows["epoch_order"]),
                                      np.arange(scope["physical_sources"]["train"]["physical_rows"]))):
            raise RuntimeError("ENTRY_STUDY_BASELINE_FIT_ROWS")
        inputs, targets, clocks, manifests = {}, {}, {}, {}
        for split in ("train","val"):
            source = scope["physical_sources"][split]
            manifest = json.loads(_entry_edge_bound(source["manifest"]).read_text())
            names = manifest["feature_contract"]["signal_bridge_fields"]
            if len(names) != 254 or (split == "val" and names != manifests["train"]["feature_contract"]["signal_bridge_fields"]):
                raise RuntimeError("ENTRY_STUDY_BASELINE_FEATURES")
            table = pq.read_table(str(_entry_edge_bound(source["parquet"])),
                                  columns=["time","snap",*GROSS_TARGET_COLUMNS])
            clock = pd.DatetimeIndex(pd.to_datetime(table.column("time").to_pandas(),utc=True)).as_unit("ns")
            if (len(clock) != source["physical_rows"] or clock.has_duplicates or not clock.is_monotonic_increasing
                    or hashlib.sha256(np.asarray(clock.asi8,dtype="<i8").tobytes()).hexdigest() != source["clock_sha256"]):
                raise RuntimeError("ENTRY_STUDY_BASELINE_CLOCK")
            x = _list_column_to_matrix(table,"snap",len(names),np.float64)
            if not np.isfinite(x).all():
                raise RuntimeError("ENTRY_STUDY_BASELINE_NONFINITE_INPUT")
            gross = np.column_stack([table.column(n).to_numpy() for n in GROSS_TARGET_COLUMNS]).astype(np.float32)
            y = build_entry_observed_market_targets(
                gross_return_bps=torch.from_numpy(gross),
                elapsed_wall_clock_seconds=torch.full((len(clock),),scope["elapsed_wall_clock_seconds"],dtype=torch.int64),
                economics=scope["economics"])["target_bps"].numpy()
            selected = fit if split == "train" else rows["control"]
            boundary = scope["train_control_cutoff" if split=="train" else "development_control_entry_end_exclusive"]
            if (clock[selected[-1]] + pd.Timedelta(seconds=300+scope["elapsed_wall_clock_seconds"]) > pd.Timestamp(boundary)):
                raise RuntimeError("ENTRY_STUDY_BASELINE_OUTCOME_BOUNDARY")
            inputs[split], targets[split], clocks[split], manifests[split] = x,y,clock,manifest
            del table,gross
        if clocks["train"][fit[-1]]+pd.Timedelta(seconds=300+scope["elapsed_wall_clock_seconds"]) >= clocks["val"][rows["control"][0]]+pd.Timedelta(minutes=5):
            raise RuntimeError("ENTRY_STUDY_BASELINE_FIT_CONTROL_OVERLAP")
        lineage = feature_fit_lineage(manifests["train"]["extra"]["multi_tf_cache_binding"])
        inner_fit,inner_val = _inner_split(len(fit),baseline["inner_fraction"],
            fit_positions=fit,purge_bars=baseline["purge_bars"])
        require_feature_fit_before(lineage, clocks["train"][fit[inner_val][0]],context="Entry study HGB inner hold")
        require_feature_fit_before(lineage, clocks["val"][rows["control"][0]],context="Entry study CONTROL")
        xpred = np.concatenate([inputs["train"][rows["train_probe"]],inputs["val"][rows["control"]]])
        predictions = np.zeros((len(xpred),3),dtype=np.float64)
        fits = []
        for side in range(2):
            predictions[:,side], info = fit_hgb(
                inputs["train"][fit],targets["train"][fit,side].astype(float),xpred,
                inner_fraction=baseline["inner_fraction"],max_iter=baseline["max_iter"],
                seed=baseline["seed"],fit_positions=fit,purge_bars=baseline["purge_bars"],
                learning_rate=baseline["learning_rate"],min_samples_leaf=baseline["min_samples_leaf"])
            fits.append(info)
            write_json(out / f"SIDE_{side}.json",info)
            print(json.dumps({"event":"ENTRY_STUDY_BASELINE_SIDE","side":side,"fit":info,
                              "elapsed_seconds":time.monotonic()-started}),flush=True)
        if not np.isfinite(predictions).all():
            raise RuntimeError("ENTRY_STUDY_BASELINE_PREDICTION_INVALID")
        frames=[]
        offset=0
        for role,split,selection in (("train","train",rows["train_probe"]),("control","val",rows["control"])):
            frame=pd.DataFrame({"role":role,"parent_row":selection,"time":clocks[split][selection]})
            for side,name in enumerate(("long","short","flat")):
                frame["target_"+name]=targets[split][selection,side]
                frame["prediction_"+name]=predictions[offset:offset+len(selection),side]
            frames.append(frame)
            offset+=len(selection)
        write_parquet(out/"PREDICTIONS.parquet",pd.concat(frames,ignore_index=True))
        result={"schema_version":"gx1_entry_learning_study_baseline_v1","status":"COMPLETE",
            "plan":{"path":str(spec_path),"sha256":sha(spec_path)},"source_commit":head,
            "fit_rows":len(fit),"fit_rows_sha256":hashlib.sha256(fit.astype("<i8").tobytes()).hexdigest(),
            "fit_time_range":[str(clocks["train"][fit[0]]),str(clocks["train"][fit[-1]])],
            "all_254_snapshot_fields":names,"feature_fit_lineage":lineage,"fits":fits,
            "reference_model_kinds":[info["model_kind"] for info in fits],
            "reference_degenerate_constant":any(info["model_kind"]=="constant" for info in fits),
            "train_constant_bps":targets["train"][fit].astype(float).mean(axis=0).tolist(),
            "small_fit_constant_bps":targets["train"][rows["fit"]].astype(float).mean(axis=0).tolist(),
            "predictions":{"path":str(out/"PREDICTIONS.parquet"),"sha256":sha(out/"PREDICTIONS.parquet")},
            "elapsed_seconds":time.monotonic()-started,"peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "capped_execution":capped,"control_role":spec["review"]["control_role"],
            "test_accessed":False,"native_training":False,"economic_rollout":False,
            "sequence_equivalence_claimed":False}
        if git("rev-parse","HEAD") != head or git("status","--porcelain"):
            raise RuntimeError("ENTRY_STUDY_SOURCE_CHANGED")
        write_json(out/"RESULT.json",result)
        write_json(out/"TERMINAL.json",{"exit_code":0,"source_unchanged":True,
            "result":{"path":str(out/"RESULT.json"),"sha256":sha(out/"RESULT.json")}})
        return {"status":"COMPLETE","output":str(out)}
    except BaseException as exc:
        write_json(out/"TERMINAL.json",{"exit_code":1,"error_type":type(exc).__name__,"error":str(exc),
                                       "source_commit":head})
        raise


# ---------------------------------------------------------------------------
# Swing-program source retrieval (operator 11.10.2026). Retrieval only: no fit,
# no predictor admission, no TEST outcome access, no orders.
# ---------------------------------------------------------------------------

def _relayed_bytes(item: dict, *, maximum_bytes: int) -> tuple[bytes, str] | None:
    """Bytes relayed by the operator under an exact hash; None when the item has no relay."""
    relay = item.get("reuse_download")
    if relay is None:
        return None
    path = Path(relay["path"])
    if sha(path) != relay["sha256"]:
        raise RuntimeError("TA_RELAYED_BYTES_HASH")
    raw = path.read_bytes()
    if len(raw) > int(maximum_bytes):
        raise RuntimeError("TA_SOURCE_RESPONSE_TOO_LARGE")
    return raw, f"relay://{path}"


def _bounded_https_get(url: str, *, maximum_bytes: int, timeout_seconds: int) -> tuple[bytes, str]:
    """One GET, no retry, no redirect policy beyond urllib's default, size enforced before use."""
    request = Request(url, headers={"User-Agent": "GX1 offline research"})
    with urlopen(request, timeout=timeout_seconds) as response:
        raw = response.read(int(maximum_bytes) + 1)
        final_url = response.url
    if len(raw) > int(maximum_bytes):
        raise RuntimeError("TA_SOURCE_RESPONSE_TOO_LARGE")
    return raw, final_url


def _retrieval_scope_or_raise(spec: dict, schema: str, label: str) -> None:
    if (spec.get("schema") != schema or spec.get("fit_allowed") is not False
            or spec.get("spending_allowed") is not False or spec.get("test_read_allowed") is not False):
        raise RuntimeError(f"{label}_MANIFEST_SCOPE")


def _finish_retrieval(out: Path, spec_path: Path, records: list[dict], status: str, extra: dict) -> dict:
    result = {"status": status, "manifest": str(spec_path), "manifest_sha256": sha(spec_path),
              "records": records, "predictor_admitted": False, "fits_run": False,
              "market_outcomes_read": False, "test_accessed": False, "native_training": False,
              "orders_sent": False, "source_commit": git("rev-parse", "HEAD"),
              "finished_utc": datetime.now(timezone.utc).isoformat(), **extra}
    write_json(out / "RESULT.json", result)
    write_json(out / "TERMINAL.json", {"status": status,
        "all_items_retrieved": all(r.get("status") == "RETRIEVED_NOT_ADMITTED" for r in records),
        "result_sha256": sha(out / "RESULT.json")})
    return {"out": str(out), "status": status,
            "item_status": {r.get("item"): r.get("status") for r in records}}


def fetch_fred_current(spec: dict, spec_path: Path) -> dict:
    """Current-vintage FRED CSV per declared series; unrevised series only are later admissible."""
    _retrieval_scope_or_raise(spec, "gx1_fred_current_vintage_source_manifest_v1", "TA_FRED")
    out = Path(spec["output_directory"])
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for series in spec["series"]:
        sid = str(series["id"])
        record = {"item": sid, "series": sid, "predictor_admitted": False,
                  "revised_after_publication": series.get("revised_after_publication"),
                  "admitted_as_input_from_this_manifest": series.get("admitted_as_input_from_this_manifest", True)}
        try:
            url = spec["endpoint_pattern"].format(series_id=sid)
            relayed = _relayed_bytes(series, maximum_bytes=spec["maximum_bytes_per_series"])
            raw, final_url = relayed if relayed is not None else _bounded_https_get(
                url, maximum_bytes=spec["maximum_bytes_per_series"], timeout_seconds=spec["timeout_seconds"])
            record["provenance"] = "RELAYED_OPERATOR_BYTES" if relayed is not None else "DIRECT_HTTPS"
            raw_path = out / f"{sid}.csv"
            raw_path.write_bytes(raw)
            reader = csv.reader(io.StringIO(raw.decode("utf-8")))
            header = next(reader)
            if header != ["observation_date", sid]:
                raise RuntimeError("TA_FRED_SCHEMA")
            rows = [row for row in reader if row]
            dates = [row[0] for row in rows]
            for row in rows:
                if len(row) != 2:
                    raise RuntimeError("TA_FRED_ROW_SHAPE")
                date.fromisoformat(row[0])
                if row[1] not in ("", ".") and not math.isfinite(float(row[1])):
                    raise RuntimeError("TA_FRED_NONFINITE")
            if dates != sorted(dates) or len(set(dates)) != len(dates):
                raise RuntimeError("TA_FRED_CLOCK")
            record.update(url=url, final_url=final_url, raw_path=str(raw_path), raw_sha256=sha(raw_path),
                          raw_bytes=len(raw), rows=len(rows),
                          numeric_rows=sum(1 for row in rows if row[1] not in ("", ".")),
                          first_date=dates[0] if dates else None, last_date=dates[-1] if dates else None,
                          status="RETRIEVED_NOT_ADMITTED")
        except Exception as exc:
            record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        records.append(record)
        write_json(out / f"{sid}.receipt.json", record)
        print(f"[TA-FRED] {sid} {record['status']}", flush=True)
    return _finish_retrieval(out, spec_path, records, "COMPLETE_FRED_RETRIEVAL", {})


def fetch_cftc_cot_archives(spec: dict, spec_path: Path) -> dict:
    """Yearly COT ZIP archives stored as received with a member inventory; no row parsed."""
    _retrieval_scope_or_raise(spec, "gx1_cftc_cot_archive_source_manifest_v1", "TA_COT")
    if spec.get("rows_after_test_seal_parsed") is not False:
        raise RuntimeError("TA_COT_MANIFEST_SCOPE")
    out = Path(spec["output_directory"])
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for year in range(int(spec["first_year"]), int(spec["last_year_inclusive"]) + 1):
        record = {"item": str(year), "year": year, "predictor_admitted": False}
        try:
            url = spec["url_pattern"].format(year=year)
            relayed = _relayed_bytes({"reuse_download": spec.get("reuse_download_by_year", {}).get(str(year))},
                                     maximum_bytes=spec["maximum_bytes_per_archive"])
            raw, final_url = relayed if relayed is not None else _bounded_https_get(
                url, maximum_bytes=spec["maximum_bytes_per_archive"], timeout_seconds=spec["timeout_seconds"])
            record["provenance"] = "RELAYED_OPERATOR_BYTES" if relayed is not None else "DIRECT_HTTPS"
            raw_path = out / f"deacot{year}.zip"
            raw_path.write_bytes(raw)
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                members = archive.infolist()
                if sum(m.file_size for m in members) > int(spec["maximum_uncompressed_bytes_per_archive"]):
                    raise RuntimeError("TA_COT_UNCOMPRESSED_SIZE")
                if archive.testzip() is not None:
                    raise RuntimeError("TA_COT_ZIP_CRC")
                inventory = {m.filename: {"bytes": m.file_size,
                                          "sha256": hashlib.sha256(archive.read(m)).hexdigest()}
                             for m in members}
            record.update(url=url, final_url=final_url, raw_path=str(raw_path), raw_sha256=sha(raw_path),
                          raw_bytes=len(raw), members=inventory, status="RETRIEVED_NOT_ADMITTED")
        except Exception as exc:
            record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        records.append(record)
        write_json(out / f"deacot{year}.receipt.json", record)
        print(f"[TA-COT] {year} {record['status']}", flush=True)
    return _finish_retrieval(out, spec_path, records, "COMPLETE_COT_ARCHIVE_RETRIEVAL", {})


def fetch_oanda_panel_daily(spec: dict, spec_path: Path) -> dict:
    """Read-only OANDA daily candles for verified instrument names; no orders, no secrets written."""
    from gx1.execution.oanda_client import OandaClient
    _retrieval_scope_or_raise(spec, "gx1_oanda_panel_daily_source_manifest_v1", "TA_OANDA_PANEL")
    if spec.get("orders_allowed") is not False or spec["candles"]["include_incomplete"] is not False:
        raise RuntimeError("TA_OANDA_PANEL_MANIFEST_SCOPE")
    candles = spec["candles"]
    from_ts, to_ts = pd.Timestamp(candles["from_utc"]), pd.Timestamp(candles["to_utc_exclusive"])
    # Scope and seal are checked before any directory or network side effect.
    if to_ts > sealed_period_read_end(spec):
        raise RuntimeError("TA_OANDA_PANEL_TEST_SEAL")
    out = Path(spec["output_directory"])
    out.mkdir(parents=True, exist_ok=False)
    data_dir = Path(spec["data_directory"])
    data_dir.mkdir(parents=True, exist_ok=True)
    client = OandaClient.from_env(timeout=int(candles["timeout_seconds"]))
    inventory = client._request("GET", f"/accounts/{client.account_id}/instruments")
    listed = {str(item["name"]): {k: item.get(k) for k in ("name", "type", "displayName", "pipLocation")}
              for item in inventory.get("instruments", [])}
    write_json(out / "INSTRUMENTS.json", {"instrument_count": len(listed), "instruments": listed,
                                           "environment": client.env})
    records = []
    for candidate in spec["candidate_instruments"]:
        name = str(candidate["name"])
        record = {"item": name, "instrument": name, "role": candidate.get("role"), "predictor_admitted": False}
        if name not in listed:
            record.update(status="NOT_LISTED_DROPPED")
            records.append(record)
            write_json(out / f"{name}.receipt.json", record)
            print(f"[TA-OANDA] {name} NOT_LISTED_DROPPED", flush=True)
            continue
        try:
            frame = client.get_candles_chunked(
                name, candles["granularity"], from_ts, to_ts,
                chunk_size=int(candles["max_candles_per_request"]), max_retries=1,
                include_mid=True, exclude_incomplete=True,
            )
            frame = frame.sort_index() if not isinstance(frame.index, pd.RangeIndex) else frame
            times = pd.DatetimeIndex(frame.index if not isinstance(frame.index, pd.RangeIndex) else frame["time"])
            if times.tz is None:
                times = times.tz_localize("UTC")
            if times.has_duplicates or not times.is_monotonic_increasing or not len(times):
                raise RuntimeError("TA_OANDA_PANEL_CLOCK")
            if times[-1] >= to_ts:
                raise RuntimeError("TA_OANDA_PANEL_BEYOND_TO")
            numeric = frame.select_dtypes(include=[np.number])
            if not np.isfinite(numeric.to_numpy(float)).all():
                raise RuntimeError("TA_OANDA_PANEL_NONFINITE")
            path = data_dir / f"{name}_D.parquet"
            frame.reset_index().to_parquet(path, index=False)
            record.update(granularity=candles["granularity"], rows=int(len(frame)),
                          first_time=str(times[0]), last_time=str(times[-1]),
                          columns=[str(c) for c in frame.reset_index().columns],
                          parquet_path=str(path), parquet_sha256=sha(path),
                          alignment="client default: dailyAlignment 17 America/New_York (not parameterized)",
                          status="RETRIEVED_NOT_ADMITTED")
        except Exception as exc:
            record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        records.append(record)
        write_json(out / f"{name}.receipt.json", record)
        print(f"[TA-OANDA] {name} {record['status']}", flush=True)
    extra = {"instruments_listed": len(listed), "orders_sent": False, "account_mutations": 0}
    return _finish_retrieval(out, spec_path, records, "COMPLETE_OANDA_PANEL_RETRIEVAL", extra)


def fetch_bounded_files(spec: dict, spec_path: Path) -> dict:
    """Named public files stored as received with SHA-256 receipts; nothing parsed, nothing admitted."""
    _retrieval_scope_or_raise(spec, "gx1_bounded_files_source_manifest_v1", "TA_FILES")
    out = Path(spec["output_directory"])
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for item in spec["files"]:
        fid = str(item["id"])
        record = {"item": fid, "url": item["url"], "predictor_admitted": False,
                  "revised_after_publication": item.get("revised_after_publication")}
        try:
            relayed = _relayed_bytes(item, maximum_bytes=item["maximum_bytes"])
            raw, final_url = relayed if relayed is not None else _bounded_https_get(
                item["url"], maximum_bytes=item["maximum_bytes"], timeout_seconds=spec["timeout_seconds"])
            record["provenance"] = "RELAYED_OPERATOR_BYTES" if relayed is not None else "DIRECT_HTTPS"
            raw_path = out / str(item["filename"])
            if raw_path.parent != out:
                raise RuntimeError("TA_FILES_FILENAME")
            raw_path.write_bytes(raw)
            record.update(final_url=final_url, raw_path=str(raw_path), raw_sha256=sha(raw_path),
                          raw_bytes=len(raw), status="RETRIEVED_NOT_ADMITTED")
        except Exception as exc:
            record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        records.append(record)
        write_json(out / f"{fid}.receipt.json", record)
        print(f"[TA-FILES] {fid} {record['status']}", flush=True)
    return _finish_retrieval(out, spec_path, records, "COMPLETE_BOUNDED_FILE_RETRIEVAL", {})


# ---------------------------------------------------------------------------
# Swing D1 study, stage A (operator 11.10.2026): positioning (COT), cross-asset
# panel (OANDA H1 resampled on the canonical D1 clock) and the D1 lane of the
# multi-TF cache joined to the existing XAU daily panel; matched two-arm fit
# against the A price-state fields; selective hit-rate endpoints on taken trades.
# No FRED drivers until relayed; no native change; TEST sealed.
# ---------------------------------------------------------------------------

SWING_COT_HEADER = {
    "as_of": "As of Date in Form YYYY-MM-DD",
    "code": "CFTC Contract Market Code",
    "open_interest": "Open Interest (All)",
    "noncomm_long": "Noncommercial Positions-Long (All)",
    "noncomm_short": "Noncommercial Positions-Short (All)",
    "noncomm_spread": "Noncommercial Positions-Spreading (All)",
    "comm_long": "Commercial Positions-Long (All)",
    "comm_short": "Commercial Positions-Short (All)",
}
# Field names and panel clock are owned by the native context owner (one truth for research and native).
from gx1.features.d1_macro_context_v1 import (  # noqa: E402
    D1_MACRO_COT_FIELDS as _D1_COT, D1_MACRO_DRIVER_FIELDS as _D1_DRIVERS,
    D1_MACRO_PANEL_RETURN_DAYS as SWING_PANEL_RETURN_DAYS, D1_MACRO_PANEL_RV_DAYS as SWING_PANEL_RV_DAYS,
)
SWING_COT_FIELDS = list(_D1_COT)
SWING_D1_LANE_EXCLUDED_PREFIXES = ("mtf_candle_",)


def parse_cot_gold_archives(spec: dict, contract_market_code: str | None = None) -> pd.DataFrame:
    """Weekly legacy futures-only rows for one CFTC contract market code (gold 088691 by default)."""
    cot = dict(spec["cot"])
    if contract_market_code is not None:
        cot["contract_market_code"] = str(contract_market_code)
    result = json.loads(Path(cot["result"]["path"]).read_text())
    if sha(Path(cot["result"]["path"])) != cot["result"]["sha256"]:
        raise RuntimeError("TA_SWING_COT_RESULT_HASH")
    rows = []
    for record in result["records"]:
        if record["status"] != "RETRIEVED_NOT_ADMITTED":
            raise RuntimeError(f"TA_SWING_COT_ARCHIVE_INCOMPLETE: {record['item']}")
        raw_path = Path(record["raw_path"])
        if sha(raw_path) != record["raw_sha256"]:
            raise RuntimeError("TA_SWING_COT_ARCHIVE_HASH")
        with zipfile.ZipFile(raw_path) as archive:
            member = archive.read("annual.txt")
        if hashlib.sha256(member).hexdigest() != record["members"]["annual.txt"]["sha256"]:
            raise RuntimeError("TA_SWING_COT_MEMBER_HASH")
        reader = csv.reader(io.StringIO(member.decode("utf-8")))
        header = [h.strip() for h in next(reader)]
        index = {}
        for key, name in SWING_COT_HEADER.items():
            if name not in header:
                raise RuntimeError(f"TA_SWING_COT_HEADER: {name}")
            index[key] = header.index(name)
        for row in reader:
            if not row or row[index["code"]].strip() != str(cot["contract_market_code"]):
                continue
            rows.append({key: row[i].strip() for key, i in index.items()})
    if not rows:
        raise RuntimeError("TA_SWING_COT_EMPTY")
    frame = pd.DataFrame(rows)
    frame["as_of"] = pd.to_datetime(frame["as_of"], format="%Y-%m-%d", utc=True)
    for key in ("open_interest", "noncomm_long", "noncomm_short", "noncomm_spread", "comm_long", "comm_short"):
        frame[key] = frame[key].astype(np.int64)
    frame = frame.drop(columns="code").sort_values("as_of").reset_index(drop=True)
    if frame["as_of"].duplicated().any() or (frame["open_interest"] <= 0).any():
        raise RuntimeError("TA_SWING_COT_ROWS")
    seal = sealed_period_read_end(spec)
    frame = frame.loc[frame["as_of"] < seal].reset_index(drop=True)
    oi = frame["open_interest"].astype(float)
    frame["cot_noncommercial_net_over_oi"] = (frame["noncomm_long"] - frame["noncomm_short"]) / oi
    frame["cot_commercial_net_over_oi"] = (frame["comm_long"] - frame["comm_short"]) / oi
    frame["cot_change4reports"] = frame["cot_noncommercial_net_over_oi"].diff(4)
    frame["cot_open_interest_log_change4reports"] = np.log(oi).diff(4)
    # Availability: declared lag after the as-of Tuesday (release Friday 15:30 ET, holiday-safe).
    frame["available_at"] = frame["as_of"] + pd.Timedelta(days=int(cot["availability_lag_days"]))
    return frame


def align_asof(panel_times: pd.DatetimeIndex, available_at: pd.Series, values: pd.DataFrame,
               *, inclusive: bool = False) -> pd.DataFrame:
    """Last row available before each decision (strictly, or at the decision when inclusive); NaN before the first.

    Known values are carried between publications, as the macro components do; a
    row is never filled from a later observation.
    """
    avail = pd.DatetimeIndex(available_at)
    if not avail.is_monotonic_increasing:
        raise RuntimeError("TA_SWING_ALIGN_ORDER")
    pos = avail.searchsorted(panel_times, side="right" if inclusive else "left") - 1
    out = pd.DataFrame(index=range(len(panel_times)), columns=values.columns, dtype=float)
    ok = pos >= 0
    out.loc[ok, :] = values.to_numpy(float)[pos[ok]]
    return out


def load_oanda_panel_d1(spec: dict) -> dict[str, pd.DataFrame]:
    """Each retrieved H1 instrument resampled on the canonical D1 session clock."""
    panel = spec["oanda_panel"]
    result = json.loads(Path(panel["result"]["path"]).read_text())
    if sha(Path(panel["result"]["path"])) != panel["result"]["sha256"]:
        raise RuntimeError("TA_SWING_OANDA_RESULT_HASH")
    receipts = {r["item"]: r for r in result["records"]}
    daily = {}
    for name in panel["instruments"]:
        record = receipts.get(name)
        if record is None or record["status"] != "RETRIEVED_NOT_ADMITTED":
            raise RuntimeError(f"TA_SWING_OANDA_INSTRUMENT_MISSING: {name}")
        path = Path(record["parquet_path"])
        if sha(path) != record["parquet_sha256"]:
            raise RuntimeError("TA_SWING_OANDA_PARQUET_HASH")
        frame = pd.read_parquet(path, columns=["time", "open", "high", "low", "close"])
        frame["time"] = pd.to_datetime(frame["time"], utc=True)
        frame = frame.set_index("time").sort_index()
        if frame.index.has_duplicates or not np.isfinite(frame.to_numpy(float)).all():
            raise RuntimeError("TA_SWING_OANDA_ROWS")
        bars = _resample_ohlc_for_model_native_scalars(frame, "D1")
        bars = bars.loc[bars.index + TRADING_SESSION_DURATION < sealed_period_read_end(spec)]
        close = bars["close"]
        features = pd.DataFrame(index=bars.index)
        for days in SWING_PANEL_RETURN_DAYS:
            features[f"{name}_logret_{days}D"] = np.log(close) - np.log(close.shift(days))
        features[f"{name}_rv_{SWING_PANEL_RV_DAYS}D"] = np.log(close).diff().rolling(SWING_PANEL_RV_DAYS).std(ddof=1)
        daily[name] = features
    return daily


def load_d1_lane_fields(spec: dict) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """D1 lane of the hash-bound multi-TF cache; candle fields excluded by operator decision."""
    lane = spec["d1_lane"]
    cache_dir = Path(lane["cache_dir"])
    manifest_path = cache_dir / "manifest.json"
    if sha(manifest_path) != lane["manifest_sha256"]:
        raise RuntimeError("TA_SWING_D1_CACHE_MANIFEST_HASH")
    manifest = json.loads(manifest_path.read_text())
    info = manifest["tfs"]["D1"]
    ts_path, feats_path = cache_dir / info["ts_npy"], cache_dir / info["feats_npy"]
    if sha(ts_path) != info["ts_npy_sha256"] or sha(feats_path) != info["feats_npy_sha256"]:
        raise RuntimeError("TA_SWING_D1_CACHE_ARRAY_HASH")
    names = list(manifest["feature_names"])
    keep = [i for i, n in enumerate(names) if not n.startswith(SWING_D1_LANE_EXCLUDED_PREFIXES)]
    if lane["field_names"] != [names[i] for i in keep]:
        raise RuntimeError("TA_SWING_D1_FIELD_LIST")
    ts = pd.to_datetime(np.load(ts_path).astype(np.int64), utc=True)
    feats = np.load(feats_path)[:, keep].astype(np.float64)
    if pd.Timedelta(manifest["shift_contract"]["D1"]) != pd.Timedelta(days=1):
        raise RuntimeError("TA_SWING_D1_SHIFT_CONTRACT")
    return pd.DatetimeIndex(ts), pd.DataFrame(feats, columns=[f"d1:{names[i]}" for i in keep])


def swing_d1_lane_at_decision(panel: pd.DataFrame, labels: pd.DatetimeIndex, feats: pd.DataFrame) -> pd.DataFrame:
    """Last closed D1 bar: label <= decision_time - one day (cache shift contract)."""
    decisions = pd.DatetimeIndex(panel["decision_time"])
    pos = labels.searchsorted(decisions - pd.Timedelta(days=1), side="right") - 1
    out = pd.DataFrame(index=panel.index, columns=feats.columns, dtype=float)
    ok = pos >= 0
    out.loc[ok, :] = feats.to_numpy(float)[pos[ok]]
    return out


def build_swing_panel(spec: dict) -> tuple[pd.DataFrame, list[str], dict, dict]:
    market, binding = load_market(spec)
    panel = daily_panel(market, pd.Timestamp(spec["read_end_exclusive"]))
    groups: dict[str, list[str]] = {"price_state_A": list(FEATURES)}
    cot = parse_cot_gold_archives(spec)
    cot_fields = align_asof(pd.DatetimeIndex(panel["decision_time"]), cot["available_at"], cot[SWING_COT_FIELDS])
    for column in SWING_COT_FIELDS:
        panel[column] = cot_fields[column].to_numpy(float)
    groups["positioning_cot"] = list(SWING_COT_FIELDS)
    daily = load_oanda_panel_d1(spec)
    decisions = pd.DatetimeIndex(panel["decision_time"])
    panel_fields = []
    for name, features in daily.items():
        # Instrument bars close at label + session; the last closed bar at or before the XAU
        # decision is carried across that instrument's holidays (no fill from later bars).
        closes = pd.Series(pd.DatetimeIndex(features.index) + TRADING_SESSION_DURATION)
        joined = align_asof(decisions, closes, features, inclusive=True)
        for column in features.columns:
            panel[column] = joined[column].to_numpy(float)
            panel_fields.append(column)
    groups["panel_oanda"] = panel_fields
    group_order = ["price_state_A", "positioning_cot", "panel_oanda"]
    if spec.get("d1_lane"):
        labels, lane = load_d1_lane_fields(spec)
        lane_rows = swing_d1_lane_at_decision(panel, labels, lane)
        for column in lane.columns:
            panel[column] = lane_rows[column].to_numpy(float)
        groups["d1_lane"] = list(lane.columns)
        group_order.append("d1_lane")
    if spec.get("drivers"):
        components = build_driver_components(spec)
        driver_rows = swing_driver_fields(panel, components)
        for column in SWING_DRIVER_FIELDS:
            panel[column] = driver_rows[column].to_numpy(float)
        groups["drivers"] = list(SWING_DRIVER_FIELDS)
        group_order.append("drivers")
    if spec.get("calendar"):
        meetings = parse_fomc_meeting_dates(spec["calendar"]["fomc"])
        calendar_rows = swing_calendar_fields(panel, meetings)
        for column in SWING_CALENDAR_FIELDS:
            panel[column] = calendar_rows[column].to_numpy(float)
        groups["calendar"] = list(SWING_CALENDAR_FIELDS)
        group_order.append("calendar")
    if spec.get("dollar"):
        dollar = build_dollar_components(spec)
        dollar_rows = swing_dollar_fields(panel, dollar)
        for column in SWING_DOLLAR_FIELDS:
            panel[column] = dollar_rows[column].to_numpy(float)
        groups["dollar"] = list(SWING_DOLLAR_FIELDS)
        group_order.append("dollar")
    excluded = [str(g) for g in spec.get("exclude_groups", [])]
    if any(g not in groups or g == "price_state_A" for g in excluded):
        raise RuntimeError("TA_SWING_EXCLUDE_GROUPS")
    feature_names = [c for g in group_order if g not in excluded for c in groups[g]]
    groups = {g: cols for g, cols in groups.items() if g not in excluded}
    if panel.columns.has_duplicates or np.isinf(panel[feature_names].to_numpy(float)).any():
        raise RuntimeError("TA_SWING_FEATURE_SURFACE")
    coverage = {g: int(np.isfinite(panel[cols].to_numpy(float)).all(axis=1).sum()) for g, cols in groups.items()}
    return panel, feature_names, groups, {"market": binding, "group_complete_rows": coverage,
                                          "cot_reports": int(len(cot)), "instruments": sorted(daily)}


# ---------------------------------------------------------------------------
# Stage B drivers from primary publishers (Treasury par/real yields, Cboe VIX,
# SPDR GLD holdings). Each parser reads only hash-bound bytes from a retrieval
# RESULT.json; availability is "first XAU D1 decision strictly after the end of
# day D+1 America/New_York", declared once here, never inferred per row.
# ---------------------------------------------------------------------------

SWING_DRIVER_FIELDS = list(_D1_DRIVERS)
NY_TZ = "America/New_York"


# ---------------------------------------------------------------------------
# Forward journal authorization (operator decision 11.10.2026). The TEST seal is
# the read end for every spec unless NEXT_RUN_POLICY.json authorizes the named
# journal to read the sealed period up to a bound end. Nothing else may read it.
# ---------------------------------------------------------------------------

FORWARD_JOURNAL_POLICY_KEY = "forward_journal_20261011"


def _forward_journal_policy() -> dict:
    policy = json.loads((ROOT / "NEXT_RUN_POLICY.json").read_text())
    return policy["current_work"].get(FORWARD_JOURNAL_POLICY_KEY, {})


def sealed_period_read_end(spec: dict) -> pd.Timestamp:
    """TEST seal, or the journal's bound end when the policy authorizes exactly this journal."""
    seal = pd.Timestamp(spec["test_seal_utc"])
    construction = spec.get("native_test_construction")
    if construction is not None:
        # Native TEST split construction (never read for fitting): the policy block names the
        # exact read end for context sources; the builder passes this only for the TEST split.
        policy = json.loads((ROOT / "NEXT_RUN_POLICY.json").read_text())["current_work"].get(construction["policy_block"], {})
        end = pd.Timestamp(construction["read_end_utc"])
        if (policy.get("test_split_context_construction_authorized") is not True
                or end > pd.Timestamp(policy["test_split_context_read_end_utc"])):
            raise RuntimeError("TA_NATIVE_TEST_CONSTRUCTION_NOT_AUTHORIZED")
        return max(seal, end)
    journal = spec.get("forward_journal")
    if journal is None:
        return seal
    end = pd.Timestamp(journal["scored_end_utc"])
    block = _forward_journal_policy()
    if (block.get("authorized") is not True or block.get("journal_id") != journal.get("journal_id")
            or end > pd.Timestamp(block["scored_end_utc"]) or block.get("native_test_split_read") is not False):
        raise RuntimeError("TA_FORWARD_JOURNAL_NOT_AUTHORIZED")
    return max(seal, end)


def _retrieval_records(binding: dict, prefix: str | None = None) -> list[dict]:
    result_path = Path(binding["path"])
    if sha(result_path) != binding["sha256"]:
        raise RuntimeError("TA_SWING_DRIVER_RESULT_HASH")
    result = json.loads(result_path.read_text())
    records = [r for r in result["records"] if prefix is None or str(r["item"]).startswith(prefix)]
    for record in records:
        if record["status"] != "RETRIEVED_NOT_ADMITTED":
            raise RuntimeError(f"TA_SWING_DRIVER_ITEM_INCOMPLETE: {record['item']}")
        if sha(Path(record["raw_path"])) != record["raw_sha256"]:
            raise RuntimeError(f"TA_SWING_DRIVER_BYTES_HASH: {record['item']}")
    if not records:
        raise RuntimeError("TA_SWING_DRIVER_NO_RECORDS")
    return records


def _driver_available_at(observation_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Observation dated D (NY close) is usable strictly after the end of day D+1 America/New_York."""
    local_end = (observation_dates.tz_localize(NY_TZ) + pd.Timedelta(days=2)).normalize()
    return local_end.tz_convert("UTC")


def parse_treasury_yields(binding: dict) -> pd.DataFrame:
    """Daily 2 Yr / 10 Yr nominal and 10 YR real par yields from the yearly Treasury CSVs."""
    nominal, real = [], []
    for record in _retrieval_records(binding, "TREASURY_"):
        frame = pd.read_csv(Path(record["raw_path"]))
        frame.columns = [str(c).strip() for c in frame.columns]
        frame["Date"] = pd.to_datetime(frame["Date"], format="%m/%d/%Y")
        if record["item"].startswith("TREASURY_NOMINAL_"):
            nominal.append(frame[["Date", "2 Yr", "10 Yr"]].rename(columns={"2 Yr": "nominal2y", "10 Yr": "nominal10y"}))
        elif record["item"].startswith("TREASURY_REAL_"):
            real.append(frame[["Date", "10 YR"]].rename(columns={"10 YR": "real10y"}))
        else:
            raise RuntimeError("TA_SWING_TREASURY_ITEM")
    table = pd.concat(nominal).merge(pd.concat(real), on="Date", how="outer").sort_values("Date").reset_index(drop=True)
    if table["Date"].duplicated().any():
        raise RuntimeError("TA_SWING_TREASURY_DUPLICATE_DATES")
    for column in ("nominal2y", "nominal10y", "real10y"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    return table


def parse_vix_history(binding: dict) -> pd.DataFrame:
    record = _retrieval_records(binding, "VIX_HISTORY")[0]
    frame = pd.read_csv(Path(record["raw_path"]))
    frame.columns = [str(c).strip().upper() for c in frame.columns]
    if "DATE" not in frame.columns or "CLOSE" not in frame.columns:
        raise RuntimeError("TA_SWING_VIX_SCHEMA")
    out = pd.DataFrame({"Date": pd.to_datetime(frame["DATE"]), "vix_close": pd.to_numeric(frame["CLOSE"], errors="coerce")})
    out = out.sort_values("Date").reset_index(drop=True)
    if out["Date"].duplicated().any() or (out["vix_close"].dropna() <= 0).any():
        raise RuntimeError("TA_SWING_VIX_ROWS")
    return out


def parse_gld_archive_csv(binding: dict, item: str) -> pd.DataFrame:
    """Header-first SPDR archive variant (2026 capture); tonnes column as published."""
    record = [r for r in _retrieval_records(binding, item)][0]
    rows = list(csv.reader(io.StringIO(Path(record["raw_path"]).read_text(encoding="utf-8-sig"))))
    header = [v.strip() for v in rows[0]]
    tonnes = "Total Net Asset Value Tonnes in the Trust as at 4.15 p.m. NYT"
    if header[0] != "Date" or tonnes not in header:
        raise RuntimeError("TA_SWING_GLD_HEADER")
    column = header.index(tonnes)
    dates, values = [], []
    for row in rows[1:]:
        if not row or not row[0].strip():
            continue
        if len(row) != len(header):
            raise RuntimeError("TA_SWING_GLD_ROW_WIDTH")
        value = row[column].strip()
        dates.append(datetime.strptime(row[0].strip(), "%d-%b-%Y"))
        values.append(float(value) if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value) else np.nan)
    out = pd.DataFrame({"Date": pd.to_datetime(dates), "gld_tonnes": values}).sort_values("Date").reset_index(drop=True)
    if out["Date"].duplicated().any() or (out["gld_tonnes"].dropna() <= 0).any():
        raise RuntimeError("TA_SWING_GLD_ROWS")
    return out


def parse_gld_archive_xlsx(binding: dict, item: str) -> pd.DataFrame:
    """Live SPDR historical-archive workbook: sheet 2 carries Date and 'Tonnes of Gold'.

    Parsed with the standard library (zip + XML regex); shared strings resolved; the
    disclaimer sheet is ignored. Same output contract as parse_gld_archive_csv.
    """
    import zipfile
    import html as _html
    record = [r for r in _retrieval_records(binding, item)][0]
    with zipfile.ZipFile(Path(record["raw_path"])) as book:
        names = book.namelist()
        shared = []
        if "xl/sharedStrings.xml" in names:
            shared = [_html.unescape(re.sub(r"<[^>]+>", "", s))
                      for s in re.findall(r"<si>(.*?)</si>", book.read("xl/sharedStrings.xml").decode("utf-8", "replace"), flags=re.S)]
        if "xl/worksheets/sheet2.xml" not in names:
            raise RuntimeError("TA_SWING_GLD_XLSX_SHEET")
        xml = book.read("xl/worksheets/sheet2.xml").decode("utf-8", "replace")
    rows = []
    for row in re.findall(r"<row[^>]*>(.*?)</row>", xml, flags=re.S):
        cells = {}
        for col, attrs, inner in re.findall(r'<c r="([A-Z]+)\d+"([^>]*?)(?:/>|>(.*?)</c>)', row, flags=re.S):
            value = re.search(r"<v>(.*?)</v>", inner or "")
            value = value.group(1) if value else ""
            if 't="s"' in attrs and value:
                value = shared[int(value)]
            cells[col] = value.strip()
        rows.append(cells)
    if not rows or rows[0].get("A") != "Date":
        raise RuntimeError("TA_SWING_GLD_XLSX_HEADER")
    tonnes_col = next((c for c, v in rows[0].items() if v == "Tonnes of Gold"), None)
    if tonnes_col is None:
        raise RuntimeError("TA_SWING_GLD_XLSX_TONNES")
    dates, values = [], []
    for cells in rows[1:]:
        if not cells.get("A"):
            continue
        value = cells.get(tonnes_col, "")
        dates.append(datetime.strptime(cells["A"], "%d-%b-%Y"))
        values.append(float(value) if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value) else np.nan)
    out = pd.DataFrame({"Date": pd.to_datetime(dates), "gld_tonnes": values}).sort_values("Date").reset_index(drop=True)
    if out["Date"].duplicated().any() or (out["gld_tonnes"].dropna() <= 0).any() or not out["gld_tonnes"].notna().any():
        raise RuntimeError("TA_SWING_GLD_ROWS")
    return out


def build_driver_components(spec: dict) -> pd.DataFrame:
    """Daily driver fields on their own observation clock, with availability stamps."""
    drv = spec["drivers"]
    treasury = parse_treasury_yields(drv["treasury"])
    vix = parse_vix_history(drv["vix"])
    gld = (parse_gld_archive_xlsx(drv["gld"], drv["gld_item"]) if drv.get("gld_format") == "xlsx"
           else parse_gld_archive_csv(drv["gld"], drv["gld_item"]))
    table = treasury.merge(vix, on="Date", how="outer").merge(gld, on="Date", how="outer").sort_values("Date").reset_index(drop=True)
    table = table.loc[table["Date"] < sealed_period_read_end(spec).tz_convert(None)].reset_index(drop=True)
    # Carry the last known observation across each series' own holidays (never backward).
    for column in ("nominal2y", "nominal10y", "real10y", "vix_close", "gld_tonnes"):
        table[column] = table[column].ffill()
    out = pd.DataFrame({"Date": table["Date"]})
    out["real10y_level"] = table["real10y"]
    for days in (1, 5, 21):
        out[f"real10y_change{days}D"] = table["real10y"].diff(days)
    out["nominal10y_level"] = table["nominal10y"]
    slope = table["nominal10y"] - table["nominal2y"]
    out["slope10y2y_level"] = slope
    out["slope10y2y_change21D"] = slope.diff(21)
    breakeven = table["nominal10y"] - table["real10y"]
    out["breakeven10y_level"] = breakeven
    out["breakeven10y_change21D"] = breakeven.diff(21)
    out["vix_log_level"] = np.log(table["vix_close"])
    out["vix_log_change5D"] = out["vix_log_level"].diff(5)
    out["gld_log_level"] = np.log(table["gld_tonnes"])
    out["gld_log_change21D"] = out["gld_log_level"].diff(21)
    out["available_at"] = _driver_available_at(pd.DatetimeIndex(out["Date"]))
    if list(out.columns[1:-1]) != SWING_DRIVER_FIELDS:
        raise RuntimeError("TA_SWING_DRIVER_FIELD_ORDER")
    return out


def swing_driver_fields(panel: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    """Last driver row available strictly before each XAU decision (same rule as COT)."""
    return align_asof(pd.DatetimeIndex(panel["decision_time"]), components["available_at"],
                      components[SWING_DRIVER_FIELDS], inclusive=False)


# ---------------------------------------------------------------------------
# Dollar input: Federal Reserve H.10 nominal broad dollar index (daily) from the
# hash-bound Data Download Program package. H.10 is a weekly release: the daily
# values for the week ending Friday are posted the following Monday at 4:15 p.m.
# (Tuesday when Monday is a holiday), so observation D is usable strictly after
# the end of the Tuesday of the week after D, America/New_York.
# ---------------------------------------------------------------------------

SWING_DOLLAR_FIELDS = ["usd_broad_log_level", "usd_broad_log_change5D", "usd_broad_log_change21D"]
_H10_BROAD_SERIES = "JRXWTFB_N.B"


def parse_fed_h10_broad_index(binding: dict, item: str) -> pd.DataFrame:
    """Daily H.10 package: a 'Time Period' header row, then dates with ND where no value was published."""
    record = [r for r in _retrieval_records(binding, item)][0]
    rows = list(csv.reader(io.StringIO(Path(record["raw_path"]).read_text(encoding="utf-8-sig"))))
    header_index = next((i for i, row in enumerate(rows) if row and row[0].strip() == "Time Period"), None)
    if header_index is None:
        raise RuntimeError("TA_SWING_H10_HEADER")
    header = [v.strip() for v in rows[header_index]]
    if _H10_BROAD_SERIES not in header:
        raise RuntimeError("TA_SWING_H10_SERIES")
    column = header.index(_H10_BROAD_SERIES)
    dates, values = [], []
    for row in rows[header_index + 1:]:
        if not row or not row[0].strip():
            continue
        if len(row) != len(header):
            raise RuntimeError("TA_SWING_H10_ROW_WIDTH")
        value = row[column].strip()
        dates.append(datetime.strptime(row[0].strip(), "%Y-%m-%d"))
        values.append(float(value) if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value) else np.nan)
    out = pd.DataFrame({"Date": pd.to_datetime(dates), "usd_broad": values}).sort_values("Date").reset_index(drop=True)
    if out["Date"].duplicated().any() or (out["usd_broad"].dropna() <= 0).any() or not out["usd_broad"].notna().any():
        raise RuntimeError("TA_SWING_H10_ROWS")
    return out


def _h10_available_at(observation_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Weekly release: D is usable strictly after the end of the Tuesday of the following week (NY)."""
    week_monday = observation_dates - pd.to_timedelta(observation_dates.weekday, unit="D")
    local_end = (week_monday + pd.Timedelta(days=9)).tz_localize(NY_TZ)
    return local_end.tz_convert("UTC")


def build_dollar_components(spec: dict) -> pd.DataFrame:
    """Broad dollar index on its published observation clock (ND days dropped, changes over published days)."""
    dol = spec["dollar"]
    table = parse_fed_h10_broad_index(dol["fed_h10"], dol["fed_h10_item"])
    keep = table["usd_broad"].notna() & (table["Date"] < sealed_period_read_end(spec).tz_convert(None))
    table = table.loc[keep].reset_index(drop=True)
    out = pd.DataFrame({"Date": table["Date"]})
    out["usd_broad_log_level"] = np.log(table["usd_broad"].to_numpy(float))
    out["usd_broad_log_change5D"] = out["usd_broad_log_level"].diff(5)
    out["usd_broad_log_change21D"] = out["usd_broad_log_level"].diff(21)
    out["available_at"] = _h10_available_at(pd.DatetimeIndex(out["Date"]))
    if list(out.columns[1:-1]) != SWING_DOLLAR_FIELDS:
        raise RuntimeError("TA_SWING_DOLLAR_FIELD_ORDER")
    return out


def swing_dollar_fields(panel: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    """Last dollar-index row available strictly before each XAU decision."""
    return align_asof(pd.DatetimeIndex(panel["decision_time"]), components["available_at"],
                      components[SWING_DOLLAR_FIELDS], inclusive=False)


# ---------------------------------------------------------------------------
# Calendar input: FOMC meeting dates from the official pages (BLS schedules are
# behind a bot wall and are not included). Dates are parsed per page year.
# ---------------------------------------------------------------------------

SWING_CALENDAR_FIELDS = ["days_to_next_fomc", "days_since_last_fomc"]
_FOMC_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
                "September", "October", "November", "December")
_FOMC_MONTH_ABBREV = {"Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April", "Jun": "June",
                      "Jul": "July", "Aug": "August", "Sep": "September", "Sept": "September", "Oct": "October",
                      "Nov": "November", "Dec": "December"}


def parse_fomc_meeting_dates(binding: dict) -> pd.DataFrame:
    """Scheduled FOMC meeting end dates (second day of two-day meetings) per calendar year.

    Historical pages carry one year each; the current calendars page carries several years
    in year-labelled sections. Only month/day pairs adjacent to the words Meeting or
    Statement are accepted; unscheduled meetings are included when listed the same way.
    """
    import html as _html
    records = _retrieval_records(binding, "FOMC_")
    month_re = "|".join(list(_FOMC_MONTHS) + list(_FOMC_MONTH_ABBREV))
    pattern = re.compile(r"(?:(" + month_re + r")\s*(?:/\s*(" + month_re + r"))?)\s+(\d{1,2})(?:\s*-\s*(\d{1,2}))?[^0-9A-Za-z]{0,40}(?:Meeting|meeting|Statement|statement)")
    rows = []
    for record in records:
        page = _html.unescape(Path(record["raw_path"]).read_text(encoding="utf-8", errors="replace"))
        text = re.sub(r"<[^>]+>", " ", page)
        text = re.sub(r"\s+", " ", text)
        item = record["item"]
        if item.startswith("FOMC_HISTORICAL_"):
            segments = [(int(item.rsplit("_", 1)[1]), text)]
        else:
            # year-labelled sections on the current page: split on "YYYY FOMC Meetings"
            parts = re.split(r"(20\d{2}) FOMC Meetings", text)
            segments = [(int(parts[i]), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]
        for year, segment in segments:
            for month1, month2, day1, day2 in pattern.findall(segment):
                month = _FOMC_MONTH_ABBREV.get(month2 or month1, month2 or month1)
                day = int(day2 or day1)
                try:
                    rows.append({"meeting_end": pd.Timestamp(year=year, month=_FOMC_MONTHS.index(month) + 1, day=day), "page_item": item})
                except ValueError:
                    continue
    if not rows:
        raise RuntimeError("TA_SWING_FOMC_EMPTY")
    out = pd.DataFrame(rows).drop_duplicates("meeting_end").sort_values("meeting_end").reset_index(drop=True)
    return out


def swing_calendar_fields(panel: pd.DataFrame, meetings: pd.DataFrame) -> pd.DataFrame:
    """days_to_next_fomc / days_since_last_fomc at each decision, calendar days, NY date basis."""
    decisions = pd.DatetimeIndex(panel["decision_time"]).tz_convert(NY_TZ).tz_localize(None).normalize()
    ends = pd.DatetimeIndex(meetings["meeting_end"]).sort_values()
    nxt = ends.searchsorted(decisions, side="right")
    prev = nxt - 1
    out = pd.DataFrame(index=panel.index)
    to_next = np.full(len(decisions), np.nan)
    ok_next = nxt < len(ends)
    to_next[ok_next] = (ends[nxt[ok_next]] - decisions[ok_next]).days
    since = np.full(len(decisions), np.nan)
    ok_prev = prev >= 0
    since[ok_prev] = (decisions[ok_prev] - ends[prev[ok_prev]]).days
    out["days_to_next_fomc"] = to_next
    out["days_since_last_fomc"] = since
    return out


def _block_matched_long_null(y: np.ndarray, yrs: np.ndarray, take: np.ndarray, *, draws: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Null hit rate and captured mean of random contiguous long episodes matched per year.

    For every holdout year the number and lengths of the selected contiguous episodes are
    kept; each episode is placed uniformly at random inside that year's valid rows and
    scored as a long position. Returns (null_hit_rates, null_captured_means) over draws.
    """
    starts = np.flatnonzero(np.concatenate(([True], np.diff(take) > 1)))
    ends = np.append(starts[1:], len(take))
    lengths_by_year: dict[int, list[int]] = {}
    for s, e in zip(starts, ends):
        lengths_by_year.setdefault(int(yrs[take[s]]), []).append(int(take[e - 1] - take[s] + 1))
    members = {year: np.flatnonzero(yrs == year) for year in lengths_by_year}
    null_hit = np.empty(int(draws))
    null_mean = np.empty(int(draws))
    for d in range(int(draws)):
        rows = []
        for year, lengths in lengths_by_year.items():
            m = members[year]
            for length in lengths:
                start = rng.integers(0, max(1, len(m) - length + 1))
                rows.append(m[start:start + length])
        r = np.concatenate(rows)
        null_hit[d] = float((y[r] > 0).mean())
        null_mean[d] = float(y[r].mean())
    return null_hit, null_mean


def _select_within_fold_year(f: np.ndarray, yrs: np.ndarray, coverage: float) -> np.ndarray:
    """Top-|forecast| rows taken inside every holdout year at the declared coverage."""
    parts = []
    for year in np.unique(yrs):
        members = np.flatnonzero(yrs == year)
        k_year = max(1, int(round(coverage * len(members))))
        parts.append(members[np.argsort(-np.abs(f[members]))][:k_year])
    return np.sort(np.concatenate(parts))


def episode_economics(panel: pd.DataFrame, forecast: np.ndarray, spec: dict) -> dict:
    """One executable trade per selected episode (first selected row, fixed hold), bid/ask fills, slippage.

    Financing is not charged here; its long-side estimate from the declared annual rate is
    reported separately. Episodes whose exit lies beyond the panel are right-censored and
    excluded with their count reported. Descriptive economics, not inference.
    """
    ep = spec["episode_economics"]
    hold = int(ep["hold_days"])
    years = pd.DatetimeIndex(panel.fill_time).year.to_numpy()
    valid = np.flatnonzero(np.isfinite(forecast) & (forecast != 0.0) & (np.arange(len(panel)) + hold < len(panel)))
    if len(valid) == 0:
        return {"status": "NO_ROWS"}
    f, yrs = forecast[valid], years[valid]
    take = valid[_select_within_fold_year(f, yrs, float(ep["coverage"]))]
    starts = take[np.concatenate(([True], np.diff(take) > 1))]
    bid, ask, mid = panel.fill_bid.to_numpy(float), panel.fill_ask.to_numpy(float), panel.fill_mid.to_numpy(float)
    times = pd.DatetimeIndex(panel.fill_time)
    slip = float(spec["slippage_bps_per_execution"])
    rows = []
    censored = 0
    for i0 in starts:
        i1 = i0 + hold
        if i1 >= len(panel):
            censored += 1
            continue
        side = 1.0 if forecast[i0] > 0 else -1.0
        if side > 0:
            net = (bid[i1] / ask[i0] - 1.0) * 1e4 - 2.0 * slip
        else:
            net = (bid[i0] / ask[i1] - 1.0) * 1e4 - 2.0 * slip
        gross_mid = side * (mid[i1] / mid[i0] - 1.0) * 1e4
        rows.append({"entry_time": str(times[i0]), "exit_time": str(times[i1]), "side": int(side),
                     "gross_mid_bps": float(gross_mid), "net_bps_ex_financing": float(net),
                     "year": int(times[i0].year)})
    if not rows:
        return {"status": "NO_EPISODES", "right_censored_episodes": censored}
    frame = pd.DataFrame(rows)
    per_year = frame.groupby("year")["net_bps_ex_financing"].agg(["count", "sum", "mean"])
    long_days = int((frame["side"] > 0).sum()) * hold
    return {
        "status": "MEASURED", "coverage": float(ep["coverage"]), "hold_days": hold,
        "episodes": int(len(frame)), "right_censored_episodes": censored,
        "long_episodes": int((frame["side"] > 0).sum()),
        "hit_rate": float((frame["net_bps_ex_financing"] > 0).mean()),
        "mean_net_bps_ex_financing": float(frame["net_bps_ex_financing"].mean()),
        "median_net_bps_ex_financing": float(frame["net_bps_ex_financing"].median()),
        "sum_net_bps_ex_financing": float(frame["net_bps_ex_financing"].sum()),
        "mean_gross_mid_bps": float(frame["gross_mid_bps"].mean()),
        "worst_episode_bps": float(frame["net_bps_ex_financing"].min()),
        "per_year": {int(y): {"episodes": int(r["count"]), "sum_net_bps": float(r["sum"]), "mean_net_bps": float(r["mean"])}
                     for y, r in per_year.iterrows()},
        "long_financing_estimate_bps": float(long_days / 365.25 * float(ep["long_annual_cost_rate"]) * 1e4),
        "financing_charged": False, "inference": "none; descriptive",
        "episodes_table": rows,
    }


def selective_endpoints(panel: pd.DataFrame, forecasts: dict, spec: dict) -> dict:
    """Hit rate and captured target on the top-coverage |forecast| rows versus coin-flip and circular-shift nulls.

    ``selection`` = ``within_fold_year`` ranks |forecast| inside each holdout year so every
    fold contributes its share (forecast magnitudes are not comparable across folds);
    ``pooled_magnitude`` is the legacy pooled ranking, kept only for comparison. Episode
    and year counts are reported so clustered selections cannot pass as many trades.
    """
    sel = spec["selective"]
    mode = sel.get("selection", "pooled_magnitude")
    if mode not in ("within_fold_year", "pooled_magnitude"):
        raise RuntimeError("TA_SWING_SELECTION_MODE")
    rng = np.random.default_rng(int(sel["seed"]))
    mid, atr = panel.fill_mid.to_numpy(float), panel.atr14.to_numpy(float)
    years = pd.DatetimeIndex(panel.fill_time).year.to_numpy()
    n = len(panel)
    report = {"selection": mode}
    for horizon, arm in forecasts.items():
        target = np.full(n, np.nan)
        target[:-horizon] = (mid[horizon:] - mid[:-horizon]) / atr[:-horizon]
        for learner, forecast in arm.items():
            if learner == "constant":
                continue
            valid = np.flatnonzero(np.isfinite(forecast) & np.isfinite(target) & (forecast != 0.0))
            if len(valid) < int(sel["minimum_rows"]):
                report[f"h{horizon}:{learner}"] = {"status": "INSUFFICIENT_ROWS", "rows": int(len(valid))}
                continue
            f, y, yrs = forecast[valid], target[valid], years[valid]
            rows = {}
            for coverage in sel["coverage_grid"]:
                if mode == "within_fold_year":
                    take = _select_within_fold_year(f, yrs, float(coverage))
                else:
                    k_all = max(1, int(round(float(coverage) * len(f))))
                    take = np.sort(np.argsort(-np.abs(f))[:k_all])
                k = len(take)
                side = np.sign(f[take])
                hits = (side == np.sign(y[take]))
                captured = side * y[take]
                shifts = rng.integers(1, len(y), size=int(sel["circular_shifts"]))
                null_hit = np.array([(side == np.sign(np.roll(y, s)[take])).mean() for s in shifts])
                null_mean = np.array([(side * np.roll(y, s)[take]).mean() for s in shifts])
                hit = float(hits.mean())
                block = {}
                if sel.get("block_null_draws"):
                    block_hit, block_mean = _block_matched_long_null(y, yrs, take, draws=int(sel["block_null_draws"]), rng=rng)
                    block = {
                        "block_null_long_hit_rate_p95": float(np.quantile(block_hit, 0.95)),
                        "block_null_long_mean_p95": float(np.quantile(block_mean, 0.95)),
                        "block_null_hit_rate_p": float((block_hit >= hit).mean()),
                        "block_null_mean_p": float((block_mean >= captured.mean()).mean()),
                        "always_long_hit_rate_same_rows": float((y[take] > 0).mean()),
                    }
                rows[str(coverage)] = {
                    **block,
                    "rows": int(k),
                    "episodes": int(1 + (np.diff(take) > 1).sum()),
                    "years_covered": int(len(np.unique(yrs[take]))),
                    "long_rows": int((side > 0).sum()),
                    "hit_rate": hit,
                    "coin_flip_hit_rate": 0.5,
                    "coin_flip_se": float(np.sqrt(0.25 / k)),
                    "hit_rate_minus_coin_flip_in_se": float((hit - 0.5) / np.sqrt(0.25 / k)),
                    "mean_captured_target_atr": float(captured.mean()),
                    "circular_shift_hit_rate_p95": float(np.quantile(null_hit, 0.95)),
                    "circular_shift_mean_p95": float(np.quantile(null_mean, 0.95)),
                    "hit_rate_above_shift_p95": bool(hit > np.quantile(null_hit, 0.95)),
                    "mean_above_shift_p95": bool(captured.mean() > np.quantile(null_mean, 0.95)),
                }
            rank_ic = float(pd.Series(f).rank().corr(pd.Series(y).rank()))
            report[f"h{horizon}:{learner}"] = {"status": "MEASURED", "rows": int(len(f)),
                                                "rank_ic": rank_ic, "coverage": rows}
    if sel.get("family_inference"):
        report["family"] = selective_family_inference(panel, forecasts, spec, mode)
    return report


def _selective_cells(panel: pd.DataFrame, forecasts: dict, spec: dict, mode: str, prefix: str = "") -> list[dict]:
    sel = spec["selective"]
    mid, atr = panel.fill_mid.to_numpy(float), panel.atr14.to_numpy(float)
    years = pd.DatetimeIndex(panel.fill_time).year.to_numpy()
    n = len(panel)
    cells = []
    for horizon, arm in forecasts.items():
        target = np.full(n, np.nan)
        target[:-horizon] = (mid[horizon:] - mid[:-horizon]) / atr[:-horizon]
        for learner, forecast in arm.items():
            if learner == "constant":
                continue
            valid = np.flatnonzero(np.isfinite(forecast) & np.isfinite(target) & (forecast != 0.0))
            if len(valid) < int(sel["minimum_rows"]):
                continue
            f, y, yrs = forecast[valid], target[valid], years[valid]
            for coverage in sel["coverage_grid"]:
                take = (_select_within_fold_year(f, yrs, float(coverage)) if mode == "within_fold_year"
                        else np.sort(np.argsort(-np.abs(f))[:max(1, int(round(float(coverage) * len(f))))]))
                cells.append({"name": f"{prefix}h{horizon}:{learner}:cov{coverage}", "y": y, "take": take, "side": np.sign(f[take])})
    return cells


def selective_family_inference_over_cells(cells: list[dict], *, draws: int, seed: int) -> dict:
    """Westfall-Young max-t over an explicit cell list with one joint circular shift per draw."""
    if not cells:
        return {"status": "NO_CELLS"}
    rng = np.random.default_rng(int(seed) + 1)
    shifts = rng.integers(1, min(len(c["y"]) for c in cells), size=int(draws))
    stats = {"hit_rate": np.empty((len(shifts), len(cells))), "captured_mean": np.empty((len(shifts), len(cells)))}
    observed = {"hit_rate": np.empty(len(cells)), "captured_mean": np.empty(len(cells))}
    for j, c in enumerate(cells):
        y, take, side = c["y"], c["take"], c["side"]
        observed["hit_rate"][j] = (side == np.sign(y[take])).mean()
        observed["captured_mean"][j] = (side * y[take]).mean()
        for i, s in enumerate(shifts):
            rolled = np.roll(y, int(s))[take]
            stats["hit_rate"][i, j] = (side == np.sign(rolled)).mean()
            stats["captured_mean"][i, j] = (side * rolled).mean()
    out = {"status": "MEASURED", "draws": int(draws), "family_size": len(cells), "cells": {}}
    for metric in ("hit_rate", "captured_mean"):
        null = stats[metric]
        mu, sd = null.mean(axis=0), null.std(axis=0, ddof=1)
        usable = sd > 0
        z_null = np.where(usable, (null - mu) / np.where(usable, sd, 1.0), -np.inf)
        z_obs = np.where(usable, (observed[metric] - mu) / np.where(usable, sd, 1.0), -np.inf)
        family_max = z_null.max(axis=1)
        for j, c in enumerate(cells):
            entry = out["cells"].setdefault(c["name"], {"rows": int(len(c["take"]))})
            entry[metric] = {"observed": float(observed[metric][j]), "null_mean": float(mu[j]), "null_sd": float(sd[j]),
                             "z": float(z_obs[j]) if usable[j] else None,
                             "raw_p": float((null[:, j] >= observed[metric][j]).mean()),
                             "family_p": float((family_max >= z_obs[j]).mean()) if usable[j] else None}
    out["minimum_family_p"] = {metric: min(v[metric]["family_p"] for v in out["cells"].values() if v[metric]["family_p"] is not None)
                               for metric in ("hit_rate", "captured_mean")}
    out["cells_with_family_p_below_0_05"] = {metric: [name for name, v in out["cells"].items() if v[metric]["family_p"] is not None and v[metric]["family_p"] < 0.05]
                                             for metric in ("hit_rate", "captured_mean")}
    return out


def selective_family_inference(panel: pd.DataFrame, forecasts: dict, spec: dict, mode: str) -> dict:
    """Joint circular-shift null over every declared selective cell with a max-t family adjustment."""
    sel = spec["selective"]
    return selective_family_inference_over_cells(_selective_cells(panel, forecasts, spec, mode),
                                                 draws=int(sel["family_draws"]), seed=int(sel["seed"]))


def run_swing_d1(spec: dict, spec_path: Path) -> dict:
    if (spec["arm"] != "SWING_D1_STAGE_A" or spec["horizons"] != [20, 5] or spec["primary_horizon"] != 20
            or spec["test_accessed"] is not False or spec["native_training"] is not False):
        raise RuntimeError("TA_SWING_PREREG_SCOPE")
    out = Path(spec["output_directory"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "STARTED.json", {"git_head": git("rev-parse", "HEAD"),
                                     "preregistration": str(spec_path), "preregistration_sha256": sha(spec_path)})
    try:
        panel, feature_names, groups, binding = build_swing_panel(spec)
        if spec["feature_names"] != feature_names:
            raise RuntimeError("TA_SWING_FEATURE_NAMES_DRIFT")
        curve = load_funding(spec)
        write_parquet(out / "SWING_PANEL.parquet", panel)
        forecasts, fits = _fit_matched_macro(panel, spec, feature_names, "swing")
        write_json(out / "FITS.json", {"fits": fits})
        write_parquet(out / "PREDICTIONS.parquet", pd.DataFrame({"fill_time": panel.fill_time, **{
            f"h{h}:{name}": pred for h, arm in forecasts.items() for name, pred in arm.items()}}))
        result = _evaluate_matched_macro(panel, forecasts, curve, spec, out, feature_names, "swing")
        result["selective"] = selective_endpoints(panel, forecasts, spec)
        result["feature_groups"] = groups
        if spec.get("seeds"):
            sensitivity = {}
            population = np.isfinite(panel[feature_names].to_numpy(float)).all(axis=1)
            for seed in spec["seeds"]:
                seeded = dict(spec, seed=int(seed))
                b, _ = _fit_predictions(panel, seeded, feature_names, population)
                a, _ = _fit_predictions(panel, seeded, FEATURES, population)
                arms = {h: {"swing_hgb": b[h]["hgb"], "a_hgb": a[h]["hgb"], "constant": b[h]["constant"]} for h in spec["horizons"]}
                sensitivity[str(seed)] = selective_endpoints(panel, arms, spec)
            result["seed_sensitivity"] = sensitivity
        if spec.get("episode_economics"):
            ep = spec["episode_economics"]
            result["episode_economics"] = {
                f"h{h}:{learner}": episode_economics(panel, forecasts[h][learner], spec)
                for h in spec["horizons"] for learner in ep["learners"]
            }
        artifacts = {}
        for path in sorted(out.iterdir()):
            if path.suffix == ".parquet":
                pq.read_table(path)
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
            artifacts[path.name] = {"sha256": sha(path), "bytes": path.stat().st_size}
        result.update(schema="gx1_ta_swing_d1_stage_a_v1", git_head=git("rev-parse", "HEAD"),
                      preregistration_sha256=sha(spec_path), inputs=binding, artifacts=artifacts,
                      primary_horizon=spec["primary_horizon"], test_accessed=False, native_training=False,
                      fred_drivers_included=bool(spec.get("drivers")), evidence_class="measured_reused_development_walkforward")
        write_json(out / "RESULT.json", result)
        write_json(out / "TERMINAL.json", {"status": "COMPLETE", "result_sha256": sha(out / "RESULT.json"),
                                         "finished_utc": datetime.now(timezone.utc).isoformat()})
        return {"status": "COMPLETE", "out": str(out), "decisions": result["decisions"]}
    except Exception as exc:
        write_json(out / "TERMINAL.json", {"status": "FAILED", "error": str(exc),
                                         "finished_utc": datetime.now(timezone.utc).isoformat()})
        raise


# ---------------------------------------------------------------------------
# Pooled-instrument swing study (operator 11.10.2026 «Ja, kjør og hent inn resten»):
# the stage-B design applied to several OANDA instruments as their own targets.
# Each instrument gets its own daily panel from its hash-bound H1 candles, its own
# COT positioning, the shared drivers and the shared cross-asset panel; fits are
# per instrument; evidence is pooled at the episode level through one joint family
# inference across every instrument's selective cells.
# ---------------------------------------------------------------------------

def load_instrument_h1(spec: dict, name: str) -> pd.DataFrame:
    panel = spec["oanda_panel"]
    result = json.loads(Path(panel["result"]["path"]).read_text())
    if sha(Path(panel["result"]["path"])) != panel["result"]["sha256"]:
        raise RuntimeError("TA_POOLED_OANDA_RESULT_HASH")
    record = next((r for r in result["records"] if r["item"] == name), None)
    if record is None or record["status"] != "RETRIEVED_NOT_ADMITTED":
        raise RuntimeError(f"TA_POOLED_INSTRUMENT_MISSING: {name}")
    path = Path(record["parquet_path"])
    if sha(path) != record["parquet_sha256"]:
        raise RuntimeError("TA_POOLED_PARQUET_HASH")
    frame = pd.read_parquet(path, columns=["time", "open", "high", "low", "close", "bid_open", "ask_open"])
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame.set_index("time").sort_index()
    if frame.index.has_duplicates or not np.isfinite(frame.to_numpy(float)).all() or (frame["ask_open"] < frame["bid_open"]).any():
        raise RuntimeError("TA_POOLED_H1_ROWS")
    return frame


def build_pooled_panels(spec: dict) -> tuple[dict[str, pd.DataFrame], list[str], dict]:
    """One daily panel per target instrument on the canonical D1 session clock."""
    end = pd.Timestamp(spec["read_end_exclusive"])
    components = build_driver_components(spec)
    daily = load_oanda_panel_d1(spec)
    panels, coverage = {}, {}
    feature_names = None
    for target in spec["targets"]:
        name, code = target["instrument"], target["cot_contract_market_code"]
        panel = daily_panel(load_instrument_h1(spec, name), end)
        groups = {"price_state_A": list(FEATURES)}
        cot = parse_cot_gold_archives(spec, code)
        cot_fields = align_asof(pd.DatetimeIndex(panel["decision_time"]), cot["available_at"], cot[SWING_COT_FIELDS])
        for column in SWING_COT_FIELDS:
            panel[column] = cot_fields[column].to_numpy(float)
        groups["positioning_cot"] = list(SWING_COT_FIELDS)
        decisions = pd.DatetimeIndex(panel["decision_time"])
        panel_fields = []
        for inst, features in daily.items():
            closes = pd.Series(pd.DatetimeIndex(features.index) + TRADING_SESSION_DURATION)
            joined = align_asof(decisions, closes, features, inclusive=True)
            for column in features.columns:
                panel[column] = joined[column].to_numpy(float)
                panel_fields.append(column)
        groups["panel_oanda"] = panel_fields
        driver_rows = swing_driver_fields(panel, components)
        for column in SWING_DRIVER_FIELDS:
            panel[column] = driver_rows[column].to_numpy(float)
        groups["drivers"] = list(SWING_DRIVER_FIELDS)
        names = [c for g in ("price_state_A", "positioning_cot", "panel_oanda", "drivers") for c in groups[g]]
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise RuntimeError("TA_POOLED_FEATURE_NAMES_DIFFER")
        if panel.columns.has_duplicates or np.isinf(panel[names].to_numpy(float)).any():
            raise RuntimeError("TA_POOLED_FEATURE_SURFACE")
        panels[name] = panel
        coverage[name] = {"rows": int(len(panel)), "complete_rows": int(np.isfinite(panel[names].to_numpy(float)).all(axis=1).sum()),
                          "cot_reports": int(len(cot)), "first_decision": str(panel.decision_time.iloc[0]), "last_decision": str(panel.decision_time.iloc[-1])}
    return panels, feature_names, coverage


# ---------------------------------------------------------------------------
# Forward journal: the frozen 68-field XAU surface scored one calendar month at a
# time. A month is scored exactly once, with a ridge fit on rows whose 20-day
# outcome closed before the month; entries are immutable and hash-chained;
# outcomes are resolved later in separate files; a completed journal year is
# evaluated with the preregistered within-year selective rule.
# ---------------------------------------------------------------------------

JOURNAL_COVERAGES = (0.05, 0.1, 0.25)


def build_journal_panel(spec: dict) -> tuple[pd.DataFrame, list[str], dict]:
    """XAU_USD through the OANDA H1 route with COT, cross-asset panel and drivers (68 fields)."""
    end = sealed_period_read_end(spec)
    if pd.Timestamp(spec["read_end_exclusive"]) > end:
        raise RuntimeError("TA_JOURNAL_READ_END")
    panel = daily_panel(load_instrument_h1(spec, "XAU_USD"), pd.Timestamp(spec["read_end_exclusive"]))
    groups = {"price_state_A": list(FEATURES)}
    cot = parse_cot_gold_archives(spec, spec["cot"]["contract_market_code"])
    cot_fields = align_asof(pd.DatetimeIndex(panel["decision_time"]), cot["available_at"], cot[SWING_COT_FIELDS])
    for column in SWING_COT_FIELDS:
        panel[column] = cot_fields[column].to_numpy(float)
    groups["positioning_cot"] = list(SWING_COT_FIELDS)
    decisions = pd.DatetimeIndex(panel["decision_time"])
    panel_fields = []
    for inst, features in load_oanda_panel_d1(spec).items():
        closes = pd.Series(pd.DatetimeIndex(features.index) + TRADING_SESSION_DURATION)
        joined = align_asof(decisions, closes, features, inclusive=True)
        for column in features.columns:
            panel[column] = joined[column].to_numpy(float)
            panel_fields.append(column)
    groups["panel_oanda"] = panel_fields
    driver_rows = swing_driver_fields(panel, build_driver_components(spec))
    for column in SWING_DRIVER_FIELDS:
        panel[column] = driver_rows[column].to_numpy(float)
    groups["drivers"] = list(SWING_DRIVER_FIELDS)
    names = [c for g in ("price_state_A", "positioning_cot", "panel_oanda", "drivers") for c in groups[g]]
    if panel.columns.has_duplicates or np.isinf(panel[names].to_numpy(float)).any():
        raise RuntimeError("TA_JOURNAL_FEATURE_SURFACE")
    coverage = {g: int(np.isfinite(panel[cols].to_numpy(float)).all(axis=1).sum()) for g, cols in groups.items()}
    return panel, names, {"group_complete_rows": coverage, "cot_reports": int(len(cot)), "rows": int(len(panel)),
                          "last_fill_time": str(panel.fill_time.iloc[-1]), "sealed_period_read_end": str(end)}


def _month_bounds(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(f"{month}-01", tz="UTC")
    return start, start + pd.offsets.MonthBegin(1)


def _finite_or_none(value: float):
    return float(value) if np.isfinite(value) else None


def score_journal_month(panel: pd.DataFrame, feature_names: list[str], spec: dict, month: str) -> dict:
    """Fit on rows whose max-horizon outcome closed before the month; score every month row once."""
    X = panel[feature_names].to_numpy(float)
    population = np.isfinite(X).all(axis=1)
    mid, atr = panel.fill_mid.to_numpy(float), panel.atr14.to_numpy(float)
    times = pd.DatetimeIndex(panel.fill_time)
    n, max_h = len(panel), max(spec["horizons"])
    positions = np.arange(n)
    start, end = _month_bounds(month)
    hold = np.flatnonzero(population & (times >= start) & (times < end))
    if not len(hold):
        return {"month": month, "status": "NO_ROWS"}
    eligible = population & (positions + max_h < n)
    past = np.flatnonzero(eligible & (times < start))
    fit = past[times[past + max_h] < start]
    if len(fit) < int(spec["min_fit_rows"]):
        return {"month": month, "status": "INSUFFICIENT_CAUSAL_ROWS", "fit_rows": int(len(fit))}
    recent = fit[-int(spec["forward_journal"]["threshold_rows"]):]
    entry = {"month": month, "status": "SCORED", "fit_rows": int(len(fit)), "hold_rows": int(len(hold)),
             "fit_positions_sha256": hashlib.sha256(fit.astype("<i8").tobytes()).hexdigest(),
             "last_fit_outcome_time": str(times[fit[-1] + max_h]), "first_hold_time": str(times[hold[0]]),
             "last_hold_time": str(times[hold[-1]]), "feature_names": list(feature_names), "horizons": list(spec["horizons"]),
             "thresholds": {}, "ridge": {}, "rows": []}
    forecasts = {}
    for horizon in spec["horizons"]:
        target = np.full(n, np.nan)
        target[:-horizon] = (mid[horizon:] - mid[:-horizon]) / atr[:-horizon]
        gram = RidgeGram(X[fit], X[hold], inner_fraction=spec["inner_fraction"], fit_positions=fit,
                         purge_bars=max_h, min_inner_rows=spec["min_inner_rows"], constant_alternative=False)
        pred, info = gram.fit_predict(target[fit])
        # Thresholds: |forecast| quantiles of the same fitted model (fixed alpha) on the most recent fit rows.
        recent_gram = RidgeGram(X[fit], X[recent], inner_fraction=spec["inner_fraction"], fit_positions=fit,
                                purge_bars=max_h, min_inner_rows=spec["min_inner_rows"], constant_alternative=False)
        recent_pred, _ = recent_gram.fit_predict(target[fit], alpha=float(info["alpha"]))
        entry["thresholds"][f"h{horizon}"] = {str(c): float(np.quantile(np.abs(recent_pred), 1.0 - c)) for c in JOURNAL_COVERAGES}
        entry["ridge"][f"h{horizon}"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) and v is not None else v)
                                        for k, v in info.items() if k in ("alpha", "inner_val_mse", "alpha_selection")}
        forecasts[horizon] = (pred, target)
    bid, ask = panel.fill_bid.to_numpy(float), panel.fill_ask.to_numpy(float)
    for j, i in enumerate(hold):
        row = {"fill_time": str(times[i]), "decision_time": str(pd.Timestamp(panel.decision_time.iloc[i])),
               "fill_mid": float(mid[i]), "fill_bid": float(bid[i]), "fill_ask": float(ask[i]), "atr14": float(atr[i])}
        for horizon, (pred, target) in forecasts.items():
            f = float(pred[j])
            row[f"forecast_h{horizon}"] = f
            row[f"target_h{horizon}"] = _finite_or_none(target[i])
            for c in JOURNAL_COVERAGES:
                row[f"flag_h{horizon}_cov{c}"] = bool(abs(f) >= entry["thresholds"][f"h{horizon}"][str(c)])
        entry["rows"].append(row)
    return entry


def _journal_summary(rows: list[dict], horizon: int) -> dict:
    """Descriptive statistics over rows whose target is known (never inference)."""
    known = [r for r in rows if r.get(f"target_h{horizon}") is not None]
    if not known:
        return {"rows_with_known_target": 0}
    f = np.array([r[f"forecast_h{horizon}"] for r in known])
    y = np.array([r[f"target_h{horizon}"] for r in known])
    out = {"rows_with_known_target": int(len(known)), "rank_ic": float(pd.Series(f).rank().corr(pd.Series(y).rank())) if len(known) > 2 else None,
           "long_share": float((f > 0).mean()), "hit_rate_all_rows": float((np.sign(f) == np.sign(y)).mean()),
           "always_long_hit_rate": float((y > 0).mean())}
    for c in JOURNAL_COVERAGES:
        mask = np.array([r[f"flag_h{horizon}_cov{c}"] for r in known])
        out[f"flagged_cov{c}"] = {"rows": int(mask.sum()),
                                  "hit_rate": float((np.sign(f[mask]) == np.sign(y[mask])).mean()) if mask.any() else None,
                                  "mean_captured_target_atr": float((np.sign(f[mask]) * y[mask]).mean()) if mask.any() else None}
    return out


def resolve_journal_month(panel: pd.DataFrame, spec: dict, entry: dict) -> dict | None:
    """Targets and long economics for an entry once every row's outcome is known; None when not yet."""
    mid, bid, ask = panel.fill_mid.to_numpy(float), panel.fill_bid.to_numpy(float), panel.fill_ask.to_numpy(float)
    atr = panel.atr14.to_numpy(float)
    times = pd.DatetimeIndex(panel.fill_time)
    slip = float(spec["slippage_bps_per_execution"])
    rows = []
    for row in entry["rows"]:
        i = times.get_indexer([pd.Timestamp(row["fill_time"])])[0]
        if i < 0 or mid[i] != row["fill_mid"]:
            raise RuntimeError("TA_JOURNAL_ROW_IDENTITY")
        resolved = {"fill_time": row["fill_time"]}
        for horizon in entry["horizons"]:
            if i + horizon >= len(panel):
                return None
            resolved[f"target_h{horizon}"] = float((mid[i + horizon] - mid[i]) / atr[i])
            resolved[f"forecast_h{horizon}"] = row[f"forecast_h{horizon}"]
            for c in JOURNAL_COVERAGES:
                resolved[f"flag_h{horizon}_cov{c}"] = row[f"flag_h{horizon}_cov{c}"]
            gross = (bid[i + horizon] - ask[i]) / ask[i] * 1e4
            resolved[f"long_net_bps_h{horizon}"] = float(gross - 2.0 * slip)
        rows.append(resolved)
    return {"month": entry["month"], "status": "RESOLVED", "rows": rows,
            "summary": {f"h{h}": _journal_summary(rows, h) for h in entry["horizons"]}}


def evaluate_journal_year(entries: list[dict], resolutions: dict[str, dict], spec: dict, year: int) -> dict:
    """Preregistered within-year selective rule over a completed journal year (12 resolved months)."""
    months = [f"{year}-{m:02d}" for m in range(1, 13)]
    if any(m not in resolutions for m in months):
        return {"year": year, "status": "INCOMPLETE_YEAR", "resolved_months": sum(m in resolutions for m in months)}
    rows = [r for m in months for r in resolutions[m]["rows"]]
    fill_time = pd.DatetimeIndex([pd.Timestamp(r["fill_time"]) for r in rows])
    # selective_endpoints recomputes targets from fill_mid/atr14; feed the stored targets through an
    # identity construction: mid[i+h] - mid[i] = target * atr is not reconstructible per horizon, so
    # evaluate directly with the owner's selection and nulls per horizon.
    sel = {"selective": {**spec["selective"], "family_inference": False}}
    report = {"year": year, "status": "EVALUATED", "rows": int(len(rows))}
    rng = np.random.default_rng(int(spec["selective"]["seed"]))
    years = fill_time.year.to_numpy()
    for horizon in spec["horizons"]:
        f = np.array([r[f"forecast_h{horizon}"] for r in rows])
        y = np.array([r[f"target_h{horizon}"] for r in rows])
        valid = np.flatnonzero(np.isfinite(f) & np.isfinite(y) & (f != 0.0))
        cells = {}
        for coverage in sel["selective"]["coverage_grid"]:
            take = _select_within_fold_year(f[valid], years[valid], float(coverage))
            side = np.sign(f[valid][take])
            yy = y[valid]
            hit = float((side == np.sign(yy[take])).mean())
            captured = float((side * yy[take]).mean())
            shifts = rng.integers(1, len(yy), size=int(sel["selective"]["circular_shifts"]))
            null_hit = np.array([(side == np.sign(np.roll(yy, s)[take])).mean() for s in shifts])
            null_mean = np.array([(side * np.roll(yy, s)[take]).mean() for s in shifts])
            block_hit, block_mean = _block_matched_long_null(yy, years[valid], take, draws=int(sel["selective"].get("block_null_draws", 0) or 1), rng=rng)
            cells[str(coverage)] = {"rows": int(len(take)), "hit_rate": hit, "mean_captured_target_atr": captured,
                                    "circular_shift_hit_rate_p95": float(np.quantile(null_hit, 0.95)),
                                    "circular_shift_mean_p95": float(np.quantile(null_mean, 0.95)),
                                    "block_null_hit_rate_p": float((block_hit >= hit).mean()),
                                    "block_null_mean_p": float((block_mean >= captured).mean())}
        report[f"h{horizon}"] = cells
    return report


def run_swing_forward_journal(spec: dict, spec_path: Path) -> dict:
    if (spec["arm"] != "SWING_FORWARD_JOURNAL" or spec["horizons"] != [20, 5] or spec["primary_horizon"] != 20
            or spec["test_accessed"] is not False or spec["native_training"] is not False
            or spec["forward_journal"]["learner"] != "swing_ridge"):
        raise RuntimeError("TA_JOURNAL_PREREG_SCOPE")
    end = sealed_period_read_end(spec)  # raises unless the policy authorizes this journal
    out = Path(spec["output_directory"])
    out.mkdir(parents=True, exist_ok=True)
    run_dir = out / ("RUN_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    run_dir.mkdir(parents=False, exist_ok=False)
    write_json(run_dir / "STARTED.json", {"git_head": git("rev-parse", "HEAD"), "preregistration": str(spec_path),
                                         "preregistration_sha256": sha(spec_path), "sealed_period_read_end": str(end)})
    try:
        panel, feature_names, binding = build_journal_panel(spec)
        if spec["feature_names"] != feature_names:
            raise RuntimeError("TA_JOURNAL_FEATURE_NAMES_DRIFT")
        write_parquet(run_dir / "JOURNAL_PANEL.parquet", panel)
        scored, skipped, resolved = [], [], []
        previous = sorted(out.glob("JOURNAL_*.json"))
        chain = sha(previous[-1]) if previous else None
        for month in spec["forward_journal"]["months"]:
            path = out / f"JOURNAL_{month}.json"
            if path.exists():
                skipped.append(month)
                continue
            entry = score_journal_month(panel, feature_names, spec, month)
            entry.update({"created_utc": datetime.now(timezone.utc).isoformat(), "git_head": git("rev-parse", "HEAD"),
                          "preregistration_sha256": sha(spec_path), "previous_entry_sha256": chain,
                          "panel_identity_sha256": hashlib.sha256(pd.util.hash_pandas_object(panel[["fill_time", "fill_mid", *feature_names]], index=False).to_numpy().tobytes()).hexdigest(),
                          "sources": {k: spec[k] for k in ("cot", "oanda_panel", "drivers")}})
            if entry["status"] == "SCORED":
                entry["summary_at_scoring"] = {f"h{h}": _journal_summary(entry["rows"], h) for h in spec["horizons"]}
            write_json(path, entry)
            chain = sha(path)
            scored.append({"month": month, "status": entry["status"], "entry_sha256": chain})
        resolutions = {}
        for path in sorted(out.glob("JOURNAL_*.json")):
            entry = json.loads(path.read_text())
            if entry["status"] != "SCORED":
                continue
            rpath = out / f"RESOLUTION_{entry['month']}.json"
            if rpath.exists():
                resolutions[entry["month"]] = json.loads(rpath.read_text())
                continue
            resolution = resolve_journal_month(panel, spec, entry)
            if resolution is not None:
                resolution.update({"created_utc": datetime.now(timezone.utc).isoformat(), "entry_sha256": sha(path)})
                write_json(rpath, resolution)
                resolutions[entry["month"]] = resolution
                resolved.append(entry["month"])
        years = sorted({int(m[:4]) for m in resolutions})
        year_reports = {}
        for year in years:
            ypath = out / f"JOURNAL_YEAR_{year}.json"
            if ypath.exists():
                year_reports[str(year)] = json.loads(ypath.read_text())
                continue
            report = evaluate_journal_year([], resolutions, spec, year)
            if report["status"] == "EVALUATED":
                write_json(ypath, report)
            year_reports[str(year)] = report
        result = {"status": "COMPLETE_FORWARD_JOURNAL_RUN", "scored": scored, "skipped_already_scored": skipped,
                  "resolved_months": resolved, "year_reports": year_reports, "inputs": binding,
                  "feature_count": len(feature_names), "sealed_period_read_end": str(end),
                  "test_seal_utc": spec["test_seal_utc"], "native_test_split_read": False,
                  "preregistration_sha256": sha(spec_path), "git_head": git("rev-parse", "HEAD"),
                  "evidence_class": "forward_scored_once_after_the_fact_fit_strictly_before_each_month",
                  "predictor_admitted": False, "orders_sent": False}
        write_json(run_dir / "RESULT.json", result)
        write_json(run_dir / "TERMINAL.json", {"status": "COMPLETE", "result_sha256": sha(run_dir / "RESULT.json"),
                                              "finished_utc": datetime.now(timezone.utc).isoformat()})
        return result
    except Exception as exc:
        write_json(run_dir / "TERMINAL.json", {"status": "FAILED", "error": str(exc),
                                              "finished_utc": datetime.now(timezone.utc).isoformat()})
        raise


def audit_d1_macro_context(spec: dict, spec_path: Path) -> dict:
    """Measure the D1 macro context owner on real bytes: parity with the journal panel at D1
    decision times, the full native M5 decision clock, brute-force availability on sampled rows,
    and pre-seal equality of the refetched OANDA panel against the earlier capture."""
    from gx1.features import d1_macro_context_v1 as ctx
    if spec.get("schema") != "gx1_d1_macro_context_audit_v1" or spec.get("fit_allowed") is not False or "forward_journal" in spec:
        raise RuntimeError("TA_D1_CONTEXT_AUDIT_SCOPE")
    out = Path(spec["output_directory"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "STARTED.json", {"git_head": git("rev-parse", "HEAD"), "spec": str(spec_path), "spec_sha256": sha(spec_path)})
    try:
        seal = pd.Timestamp(spec["test_seal_utc"])
        components = ctx.d1_macro_context_components(spec)
        fields = list(ctx.D1_MACRO_CONTEXT_FIELDS)
        # (a) parity with the forward-journal panel at its D1 decision times before the seal
        jp = spec["journal_panel"]
        if sha(Path(jp["path"])) != jp["sha256"]:
            raise RuntimeError("TA_D1_CONTEXT_JOURNAL_PANEL_HASH")
        journal = pd.read_parquet(jp["path"])
        pre = journal.loc[pd.DatetimeIndex(journal["decision_time"]) < seal].reset_index(drop=True)
        built = ctx.build_d1_macro_context(pd.DatetimeIndex(pre["decision_time"]), components)
        expected = pre[fields].to_numpy(np.float64)
        got = built[fields].to_numpy(np.float64)
        parity_equal = bool(np.array_equal(expected, got, equal_nan=True))
        mismatch = {}
        if not parity_equal:
            for j, name in enumerate(fields):
                bad = ~((expected[:, j] == got[:, j]) | (np.isnan(expected[:, j]) & np.isnan(got[:, j])))
                if bad.any():
                    mismatch[name] = int(bad.sum())
        parity = {"rows": int(len(pre)), "bit_identical": parity_equal, "mismatching_fields": mismatch}
        del journal, pre, built
        # (b) the full native M5 decision clock
        market, tape_binding = load_market(spec, columns=["time", "open"])
        times = pd.DatetimeIndex(market.index) + M5_BAR_DURATION
        del market
        frame = ctx.build_d1_macro_context(times, components)
        binding = ctx.d1_macro_context_binding(spec, frame)
        values = frame[fields].to_numpy(np.float64)
        complete = np.isfinite(values).all(axis=1)
        per_group = {}
        for group, names in ctx.D1_MACRO_CONTEXT_FIELDS_BY_GROUP.items():
            idx = [fields.index(n) for n in names]
            ok = np.isfinite(values[:, idx]).all(axis=1)
            per_group[group] = {"complete_rows": int(ok.sum()), "first_complete": str(times[int(np.argmax(ok))]) if ok.any() else None}
        # (c) brute-force availability on sampled rows after the first complete row
        rng = np.random.default_rng(int(spec["seed"]))
        candidates = np.flatnonzero(complete)
        sample = np.sort(rng.choice(candidates, size=min(int(spec["sample_rows"]), len(candidates)), replace=False))
        cot = components["cot"]
        drivers = components["drivers"]
        cot_avail = pd.DatetimeIndex(cot["available_at"])
        drv_avail = pd.DatetimeIndex(drivers["available_at"])
        checks = {"cot": 0, "drivers": 0, "panel": 0}
        for i in sample:
            t = times[i]
            k = int((cot_avail < t).sum()) - 1
            if not np.array_equal(values[i, :4], cot[list(SWING_COT_FIELDS)].to_numpy(float)[k]):
                checks["cot"] += 1
            k = int((drv_avail < t).sum()) - 1
            if not np.array_equal(values[i, -13:], drivers[list(SWING_DRIVER_FIELDS)].to_numpy(float)[k], equal_nan=True):
                checks["drivers"] += 1
            for inst in ctx.D1_MACRO_PANEL_INSTRUMENTS:
                feats = components["panel"][inst]
                closes = pd.DatetimeIndex(feats.index) + TRADING_SESSION_DURATION
                k = int((closes <= t).sum()) - 1
                cols = [fields.index(c) for c in feats.columns]
                if not np.array_equal(values[i, cols], feats.to_numpy(float)[k], equal_nan=True):
                    checks["panel"] += 1
        availability = {"sampled_rows": int(len(sample)), "mismatches": checks}
        # (d) refetched OANDA panel equals the earlier capture before the seal
        previous = dict(spec, oanda_panel=spec["oanda_previous"])
        earlier = load_oanda_panel_d1(previous)
        panel_equal = {}
        for inst in ctx.D1_MACRO_PANEL_INSTRUMENTS:
            a, b = components["panel"][inst], earlier[inst]
            panel_equal[inst] = bool(len(a) == len(b) and a.index.equals(b.index) and np.array_equal(a.to_numpy(float), b.to_numpy(float), equal_nan=True))
        decision = "D1_MACRO_CONTEXT_AUDIT_PASS" if (parity_equal and checks == {"cot": 0, "drivers": 0, "panel": 0} and all(panel_equal.values())) else "D1_MACRO_CONTEXT_AUDIT_FAIL"
        result = {"decision": decision, "contract": ctx.d1_macro_context_contract(), "binding": binding, "parity_with_journal_panel": parity,
                  "m5_clock": {"rows": int(len(times)), "first": str(times[0]), "last": str(times[-1]), "complete_rows": int(complete.sum()),
                               "first_complete": str(times[int(np.argmax(complete))]) if complete.any() else None, "per_group": per_group},
                  "availability_bruteforce": availability, "refetched_panel_equals_earlier_capture_before_seal": panel_equal,
                  "tape": tape_binding, "sealed_period_read_end": str(components["read_end"]), "test_accessed": False,
                  "evidence_class": "measured_on_real_hash_bound_bytes", "git_head": git("rev-parse", "HEAD"), "spec_sha256": sha(spec_path)}
        write_json(out / "RESULT.json", result)
        write_json(out / "TERMINAL.json", {"status": "COMPLETE", "decision": decision, "result_sha256": sha(out / "RESULT.json"),
                                          "finished_utc": datetime.now(timezone.utc).isoformat()})
        return result
    except Exception as exc:
        write_json(out / "TERMINAL.json", {"status": "FAILED", "error": str(exc), "finished_utc": datetime.now(timezone.utc).isoformat()})
        raise


def audit_entry_horizon_clock(spec: dict, spec_path: Path) -> dict:
    """Measure the trading-day horizon owner on the native M1 tape: validity, elapsed wall-clock,
    weekend/holiday skipping and censoring for every M5 decision in a declared window."""
    from gx1.contracts import entry_observed_market_horizons_v1 as hz
    from gx1.contracts.entry_causal_m1_outcomes_v1 import build_entry_m1_fill_surface, prepare_causal_m1_quote_source
    if spec.get("schema") != "gx1_entry_horizon_clock_audit_v1" or spec.get("fit_allowed") is not False or "forward_journal" in spec:
        raise RuntimeError("TA_HORIZON_CLOCK_AUDIT_SCOPE")
    out = Path(spec["output_directory"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "STARTED.json", {"git_head": git("rev-parse", "HEAD"), "spec": str(spec_path), "spec_sha256": sha(spec_path)})
    try:
        seal = pd.Timestamp(spec["test_seal_utc"])
        m1_root = Path(spec["m1_tape"]["root"])
        if sha(m1_root / "MANIFEST.json") != spec["m1_tape"]["manifest_sha256"]:
            raise RuntimeError("TA_HORIZON_CLOCK_M1_MANIFEST_HASH")
        m1_manifest = json.loads((m1_root / "MANIFEST.json").read_text())
        if m1_manifest["timestamp_semantics"] != "bar_start_utc" or int(m1_manifest["bar_duration_seconds"]) != 60:
            raise RuntimeError("TA_HORIZON_CLOCK_M1_CLOCK")
        frames, m1_binding = [], []
        for year in spec["m1_years"]:
            path = m1_root / f"year={year}" / "part-000.parquet"
            digest = sha(path)
            if digest != m1_manifest["year_sha256"][f"year={year}"]:
                raise RuntimeError(f"TA_HORIZON_CLOCK_M1_YEAR_HASH: {year}")
            frame = pq.read_table(path, columns=["time", "bid_open", "bid_high", "bid_low", "ask_open", "ask_high", "ask_low"]).to_pandas()
            frames.append(frame)
            m1_binding.append({"path": str(path), "sha256": digest, "rows": int(len(frame))})
        m1 = pd.concat(frames, ignore_index=True)
        m1["time"] = pd.to_datetime(m1["time"], utc=True)
        m1 = m1.loc[m1["time"] < seal].sort_values("time").reset_index(drop=True)
        quotes = prepare_causal_m1_quote_source(m1)
        market, tape_binding = load_market(spec, columns=["time", "open"])
        window = market.loc[(market.index >= pd.Timestamp(spec["window_start"])) & (market.index < pd.Timestamp(spec["window_end_exclusive"]))]
        m5_times = pd.DatetimeIndex(window.index)
        surface = build_entry_m1_fill_surface(m5_decision_times=m5_times, closed_m1=quotes)
        bound = int(surface["entry_fill_bound"].sum())
        report = {}
        for h in hz.OBSERVATION_HORIZONS_TRADING_DAYS:
            o = hz.causal_m1_terminal_outcomes_at_trading_days(fill_surface=surface, closed_m1=quotes, horizon_trading_days=h)
            valid = o["outcome_valid"].to_numpy(bool)
            el = o[hz.ELAPSED_COLUMN_BY_HORIZON[h]].to_numpy(np.int64)[valid] / 86400.0
            entry_dow = pd.DatetimeIndex(o["entry_decision_at"]).dayofweek.to_numpy()
            exit_s = hz.session_day_index(pd.DatetimeIndex(o["exit_decision_at"][valid]))
            entry_s = hz.session_day_index(pd.DatetimeIndex(o["entry_decision_at"][valid]))
            long_bps = o[hz.GROSS_TARGET_COLUMNS_BY_HORIZON[h][0]].to_numpy(float)[valid]
            per_dow = {int(d): {"rows": int((entry_dow[valid] == d).sum()), "elapsed_days_min": float(el[entry_dow[valid] == d].min()) if (entry_dow[valid] == d).any() else None,
                                "elapsed_days_median": float(np.median(el[entry_dow[valid] == d])) if (entry_dow[valid] == d).any() else None}
                       for d in range(7)}
            report[f"h{h}d"] = {
                "rows": int(len(o)), "fill_bound_rows": bound, "valid_rows": int(valid.sum()), "censored_bound_rows": int(bound - valid.sum()),
                "elapsed_days": {"min": float(el.min()), "median": float(np.median(el)), "max": float(el.max())},
                "calendar_days_between_sessions": {"min": int((exit_s - entry_s).min()), "max": int((exit_s - entry_s).max()), "all_at_least_h": bool((exit_s - entry_s >= h).all())},
                "exit_before_seal": bool((pd.DatetimeIndex(o["exit_decision_at"][valid]) < seal).all()),
                "last_valid_entry": str(pd.DatetimeIndex(o["entry_decision_at"][valid]).max()),
                "long_gross_bps": {"mean": float(long_bps.mean()), "sd": float(long_bps.std(ddof=1)), "min": float(long_bps.min()), "max": float(long_bps.max())},
                "per_entry_weekday": per_dow,
            }
        decision = "ENTRY_HORIZON_CLOCK_AUDIT_PASS" if all(r["calendar_days_between_sessions"]["all_at_least_h"] and r["exit_before_seal"] for r in report.values()) else "ENTRY_HORIZON_CLOCK_AUDIT_FAIL"
        result = {"decision": decision, "contract": hz.entry_observed_market_horizons_contract(), "window": [spec["window_start"], spec["window_end_exclusive"]],
                  "m5_rows": int(len(m5_times)), "m1_rows_before_seal": int(len(m1)), "m1_binding": m1_binding, "m5_tape": tape_binding,
                  "horizons": report, "test_accessed": False, "evidence_class": "measured_on_real_hash_bound_bytes",
                  "git_head": git("rev-parse", "HEAD"), "spec_sha256": sha(spec_path)}
        write_json(out / "RESULT.json", result)
        write_json(out / "TERMINAL.json", {"status": "COMPLETE", "decision": decision, "result_sha256": sha(out / "RESULT.json"),
                                          "finished_utc": datetime.now(timezone.utc).isoformat()})
        return result
    except Exception as exc:
        write_json(out / "TERMINAL.json", {"status": "FAILED", "error": str(exc), "finished_utc": datetime.now(timezone.utc).isoformat()})
        raise


def run_swing_pooled(spec: dict, spec_path: Path) -> dict:
    if (spec["arm"] != "SWING_POOLED_D1" or spec["horizons"] != [20, 5] or spec["primary_horizon"] != 20
            or spec["test_accessed"] is not False or spec["native_training"] is not False):
        raise RuntimeError("TA_POOLED_PREREG_SCOPE")
    out = Path(spec["output_directory"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "STARTED.json", {"git_head": git("rev-parse", "HEAD"),
                                     "preregistration": str(spec_path), "preregistration_sha256": sha(spec_path)})
    try:
        panels, feature_names, coverage = build_pooled_panels(spec)
        if spec["feature_names"] != feature_names:
            raise RuntimeError("TA_POOLED_FEATURE_NAMES_DRIFT")
        per_instrument, all_cells, fits_all = {}, [], {}
        mode = spec["selective"].get("selection", "within_fold_year")
        for name, panel in panels.items():
            write_parquet(out / f"PANEL_{name}.parquet", panel)
            forecasts, fits = _fit_matched_macro(panel, spec, feature_names, "swing")
            fits_all[name] = fits
            write_parquet(out / f"PREDICTIONS_{name}.parquet", pd.DataFrame({"fill_time": panel.fill_time, **{
                f"h{h}:{k}": v for h, arm in forecasts.items() for k, v in arm.items()}}))
            block = {"selective": selective_endpoints(panel, forecasts, {**spec, "selective": {**spec["selective"], "family_inference": False}}),
                     "episode_economics": {f"h{h}:{learner}": episode_economics(panel, forecasts[h][learner], spec)
                                           for h in spec["horizons"] for learner in spec["episode_economics"]["learners"]}
                     if spec.get("episode_economics") else {}}
            per_instrument[name] = block
            all_cells.extend(_selective_cells(panel, forecasts, spec, mode, prefix=f"{name}:"))
        family = selective_family_inference_over_cells(all_cells, draws=int(spec["selective"]["family_draws"]), seed=int(spec["selective"]["seed"]))
        # cross-instrument consistency for each (horizon, learner, coverage) cell
        consistency = {}
        for cell in all_cells:
            key = cell["name"].split(":", 1)[1]
            hit = float((cell["side"] == np.sign(cell["y"][cell["take"]])).mean())
            consistency.setdefault(key, {})[cell["name"].split(":", 1)[0]] = round(hit, 3)
        write_json(out / "FITS.json", {"fits": fits_all})
        result = {"schema": "gx1_ta_swing_pooled_d1_v1", "git_head": git("rev-parse", "HEAD"),
                  "preregistration_sha256": sha(spec_path), "instruments": list(panels), "coverage": coverage,
                  "feature_count": len(feature_names), "per_instrument": per_instrument, "family": family,
                  "cross_instrument_hit_rates": consistency, "pooled_coefficient_fit": False,
                  "test_accessed": False, "native_training": False,
                  "evidence_class": "measured_reused_development_walkforward"}
        artifacts = {}
        for path in sorted(out.iterdir()):
            if path.suffix == ".parquet":
                pq.read_table(path)
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
            artifacts[path.name] = {"sha256": sha(path), "bytes": path.stat().st_size}
        result["artifacts"] = artifacts
        write_json(out / "RESULT.json", result)
        write_json(out / "TERMINAL.json", {"status": "COMPLETE", "result_sha256": sha(out / "RESULT.json"),
                                         "finished_utc": datetime.now(timezone.utc).isoformat()})
        return {"status": "COMPLETE", "out": str(out), "minimum_family_p": family.get("minimum_family_p")}
    except Exception as exc:
        write_json(out / "TERMINAL.json", {"status": "FAILED", "error": str(exc),
                                         "finished_utc": datetime.now(timezone.utc).isoformat()})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["fetch-funding", "fetch-alfred", "audit-alfred-chunks", "prepare-b-macros", "prepare-macro-core", "run-macro-core", "import-b-archived-snapshots", "run-a", "run-c", "audit-dukascopy-cache", "run-sweep", "audit-gc-source", "entry-edge-diagnostic", "entry-edge-probe", "entry-learning-baseline", "fetch-fred-current", "fetch-cftc-cot-archives", "fetch-oanda-panel-daily", "fetch-bounded-files", "run-swing-d1", "run-swing-pooled", "run-swing-forward-journal", "audit-d1-macro-context", "audit-entry-horizon-clock"])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--receipt-sha256")
    args = parser.parse_args()
    spec = checked_spec(args.spec, args.spec_sha256)
    if args.mode == "audit-gc-source":
        require_capped_cpu_audit_execution()
        result = audit_gc_source(spec, args.spec)
        print(json.dumps(result))
        return 2 if result["status"] == "BLOCKED_NO_BOUND_GC_FILES" else 0
    if args.mode == "audit-alfred-chunks":
        print(json.dumps(audit_alfred_chunks(spec, args.spec, args.receipt_sha256)))
        return 0
    owner = {"audit-entry-horizon-clock": audit_entry_horizon_clock, "audit-d1-macro-context": audit_d1_macro_context, "run-swing-forward-journal": run_swing_forward_journal, "run-swing-pooled": run_swing_pooled, "run-swing-d1": run_swing_d1, "fetch-bounded-files": fetch_bounded_files, "fetch-fred-current": fetch_fred_current, "fetch-cftc-cot-archives": fetch_cftc_cot_archives, "fetch-oanda-panel-daily": fetch_oanda_panel_daily, "entry-learning-baseline": run_entry_learning_baseline, "entry-edge-probe": run_entry_edge_probe, "entry-edge-diagnostic": run_entry_edge_diagnostic, "run-macro-core": run_macro_core, "prepare-macro-core": prepare_macro_core, "audit-dukascopy-cache": audit_dukascopy_cache, "run-sweep": run_sweep, "import-b-archived-snapshots": import_b_archived_snapshots, "prepare-b-macros": prepare_b_macros, "fetch-funding": fetch_funding, "fetch-alfred": fetch_alfred, "run-a": run_a, "run-c": run_c}[args.mode]
    result = owner(spec, args.spec)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
