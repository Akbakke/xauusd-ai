"""Bounded, resumable Databento batch acquisition for the GC 26-month plan.

Only submits the hash-bound MBP-1 and support scopes admitted by the GC plan.
It does not admit data, fit a model, or access TEST.
"""
from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

import pyarrow as pa


REPO_ROOT = Path("/home/andre2/src/GX1_CURRENT")
POLICY_PATH = REPO_ROOT / "NEXT_RUN_POLICY.json"
CONFIG_PATH = REPO_ROOT / "configs/research/GC_ORDER_FLOW_RESEARCH_001.json"
RAW_QUOTE_PATH = Path("/home/andre2/GX1_RUNS/M5_RECHECK_20261015_001/gc/QUOTE_RAW_26M_001.json")
SUPPORT_QUOTE_PATH = Path("/home/andre2/GX1_RUNS/M5_RECHECK_20261015_001/gc/QUOTE_SUPPORT_RAW_27M_001.json")
RUN_ROOT = Path("/home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001/DATABENTO_GC_26M_20261015_001")
RAW_REQUOTE_PATH = RUN_ROOT / "RAW_METADATA_REQUOTE_20261015.json"
SOURCE_ROOT = Path("/home/andre2/GX1_DATA/research/GC_ORDER_FLOW_RESEARCH_001/DATABENTO_GC_26M_20261015_001")
HARD_CAP = Decimal("120.00")
REPORTED_CREDITS = Decimal("125.00")
RAW_SYMBOL_RE = re.compile(r"GC[FGHJKMNQUVXZ][0-9]{1,2}\Z")
MBP_FIELDS = {
    "ts_recv", "ts_event", "rtype", "publisher_id", "instrument_id", "action",
    "side", "price", "size", "flags", "sequence", "bid_px_00", "ask_px_00",
    "bid_sz_00", "ask_sz_00",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _close_cost(left: Decimal, right: Decimal) -> bool:
    return abs(left - right) <= Decimal("0.0000000001")


def _midnight_utc(value: str) -> str:
    if re.fullmatch(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}", value) is None:
        raise ValueError("GC_DATE_SCOPE_INVALID")
    return f"{value}T00:00:00Z"


def build_scopes(raw_quote: dict[str, Any], support_quote: dict[str, Any]) -> list[dict[str, Any]]:
    if raw_quote.get("dataset") != "GLBX.MDP3" or raw_quote.get("schema") != "mbp-1":
        raise ValueError("GC_RAW_QUOTE_IDENTITY_INVALID")
    if support_quote.get("dataset") != "GLBX.MDP3":
        raise ValueError("GC_SUPPORT_QUOTE_IDENTITY_INVALID")
    raw_symbols = raw_quote.get("raw_symbols")
    if not isinstance(raw_symbols, list) or not raw_symbols or any(
        not isinstance(symbol, str) or RAW_SYMBOL_RE.fullmatch(symbol) is None for symbol in raw_symbols
    ):
        raise ValueError("GC_RAW_SYMBOL_SET_INVALID")
    segments = raw_quote.get("segment_quotes")
    if not isinstance(segments, list) or len(segments) != raw_quote.get("segment_count"):
        raise ValueError("GC_RAW_SEGMENT_COUNT_INVALID")
    scopes: list[dict[str, Any]] = []
    previous_end = "2024-05-01"
    raw_sum = Decimal("0")
    for index, segment in enumerate(segments):
        start, end = segment.get("start"), segment.get("end")
        symbols = segment.get("symbols")
        if start != previous_end or not isinstance(symbols, list) or not symbols or any(
            symbol not in raw_symbols for symbol in symbols
        ):
            raise ValueError("GC_RAW_SEGMENT_COVERAGE_INVALID")
        if Decimal(str(segment.get("cost_usd"))) < 0:
            raise ValueError("GC_RAW_SEGMENT_COST_INVALID")
        previous_end = end
        raw_sum += Decimal(str(segment["cost_usd"]))
        scopes.append({
            "scope_id": f"mbp1_{index + 1:02d}", "dataset": "GLBX.MDP3", "schema": "mbp-1",
            "symbols": sorted(symbols), "stype_in": "raw_symbol",
            "start": _midnight_utc(start), "end": _midnight_utc(end),
            "quoted_cost_usd": str(Decimal(str(segment["cost_usd"]))),
        })
    if previous_end != "2026-07-01" or not _close_cost(raw_sum, Decimal(str(raw_quote.get("estimated_cost_usd")))):
        raise ValueError("GC_RAW_QUOTE_TOTAL_OR_WINDOW_INVALID")

    support_periods = support_quote.get("per_symbol_periods")
    if not isinstance(support_periods, dict) or set(support_periods) != set(raw_symbols):
        raise ValueError("GC_SUPPORT_SYMBOL_SET_INVALID")
    definition_sum = Decimal("0")
    daily_sum = Decimal("0")
    period_count = 0
    for symbol in sorted(raw_symbols):
        if RAW_SYMBOL_RE.fullmatch(symbol) is None:
            raise ValueError("GC_SUPPORT_RAW_SYMBOL_INVALID")
        for index, period in enumerate(support_periods[symbol]):
            start, end = period.get("start"), period.get("end")
            if period.get("symbol") != symbol:
                raise ValueError("GC_SUPPORT_PERIOD_SYMBOL_INVALID")
            if not ("2024-04-01" <= start < end <= "2026-07-01"):
                raise ValueError("GC_SUPPORT_PERIOD_OUTSIDE_BOUND_WINDOW")
            for schema, cost_key, total in (
                ("definition", "definition_cost_usd", "definition"),
                ("ohlcv-1d", "ohlcv-1d_cost_usd", "ohlcv-1d"),
            ):
                quoted = Decimal(str(period.get(cost_key)))
                if quoted < 0:
                    raise ValueError("GC_SUPPORT_PERIOD_COST_INVALID")
                if total == "definition":
                    definition_sum += quoted
                else:
                    daily_sum += quoted
                scopes.append({
                    "scope_id": f"{schema.replace('-', '')}_{symbol}_{index + 1:02d}",
                    "dataset": "GLBX.MDP3", "schema": schema, "symbols": [symbol],
                    "stype_in": "raw_symbol", "start": _midnight_utc(start),
                    "end": _midnight_utc(end), "quoted_cost_usd": str(quoted),
                })
            period_count += 1
    if period_count != support_quote.get("period_count"):
        raise ValueError("GC_SUPPORT_PERIOD_COUNT_INVALID")
    if not _close_cost(definition_sum, Decimal(str(support_quote["totals_usd"]["definition"]))):
        raise ValueError("GC_SUPPORT_DEFINITION_TOTAL_INVALID")
    if not _close_cost(daily_sum, Decimal(str(support_quote["totals_usd"]["ohlcv-1d"]))):
        raise ValueError("GC_SUPPORT_DAILY_TOTAL_INVALID")
    if len({scope["scope_id"] for scope in scopes}) != len(scopes):
        raise ValueError("GC_SCOPE_ID_COLLISION")
    return scopes


def validate_budget(scopes: list[dict[str, Any]], cap: Decimal = HARD_CAP) -> Decimal:
    total = sum((Decimal(scope["quoted_cost_usd"]) for scope in scopes), Decimal("0"))
    if cap != HARD_CAP or total > cap or total > REPORTED_CREDITS:
        raise ValueError("GC_HARD_COST_CAP_EXCEEDED")
    if not scopes or any(Decimal(scope["quoted_cost_usd"]) < 0 for scope in scopes):
        raise ValueError("GC_SCOPE_LIST_INVALID")
    return total


def _env_key() -> str:
    direct = os.environ.get("DATABENTO_API_KEY")
    if direct:
        return direct.strip()
    env_path = REPO_ROOT / ".env"
    if env_path.is_symlink() or not env_path.is_file():
        raise ValueError("DATABENTO_KEY_FILE_UNAVAILABLE")
    mode = stat.S_IMODE(env_path.stat().st_mode)
    if mode & 0o077:
        raise ValueError("DATABENTO_KEY_FILE_PERMISSIONS_INVALID")
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("DATABENTO_API_KEY="):
            value = line.split("=", 1)[1].strip().strip("\"'")
            if value:
                return value
    raise ValueError("DATABENTO_KEY_MISSING")


class HistoricalAPI:
    def __init__(self, key: str, timeout: int = 120):
        self._authorization = "Basic " + base64.b64encode((key + ":").encode()).decode()
        self.timeout = timeout

    def request(self, method: str, params: dict[str, Any] | None = None, *, post: bool = False) -> Any:
        params = params or {}
        url = "https://hist.databento.com/v0/" + method
        payload = None
        headers = {"Authorization": self._authorization, "Accept": "application/json"}
        if post:
            payload = urllib.parse.urlencode(params, doseq=True).encode("utf-8")
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        else:
            url += "?" + urllib.parse.urlencode(params, doseq=True)
        request = urllib.request.Request(
            url, data=payload, method="POST" if post else "GET", headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            # Do not include response bodies: they may contain account metadata.
            raise RuntimeError(f"DATABENTO_HTTP_{exc.code}:{method}") from None
        except (TimeoutError, OSError) as exc:
            raise RuntimeError(f"DATABENTO_TRANSPORT_UNKNOWN:{method}:{type(exc).__name__}") from None
        text = body.decode("utf-8")
        if method.startswith("metadata."):
            return text.strip()
        return json.loads(text) if text else None

    def list_jobs(self) -> list[dict[str, Any]]:
        result = self.request("batch.list_jobs", {"states": "queued,processing,done,expired", "short": "false"})
        if isinstance(result, dict):
            result = result.get("jobs", result.get("data", []))
        if not isinstance(result, list):
            raise ValueError("DATABENTO_JOB_LIST_FORMAT_INVALID")
        return [job for job in result if isinstance(job, dict)]

    def submit(self, scope: dict[str, Any]) -> dict[str, Any]:
        params = {
            "dataset": scope["dataset"], "symbols": ",".join(scope["symbols"]),
            "schema": scope["schema"], "stype_in": scope["stype_in"],
            "stype_out": "instrument_id", "start": scope["start"], "end": scope["end"],
            "encoding": "csv", "compression": "zstd", "pretty_px": "false",
            "pretty_ts": "false", "map_symbols": "true", "split_symbols": "true",
            "split_duration": "none", "delivery": "download",
        }
        result = self.request("batch.submit_job", params, post=True)
        if not isinstance(result, dict) or not isinstance(result.get("id"), str):
            raise ValueError("DATABENTO_SUBMIT_RESPONSE_INVALID")
        return result

    def job_details(self, job_id: str) -> dict[str, Any]:
        result = self.request("batch.get_job_details", {"job_id": job_id})
        if not isinstance(result, dict):
            raise ValueError("DATABENTO_JOB_DETAIL_FORMAT_INVALID")
        return result

    def list_files(self, job_id: str) -> list[dict[str, Any]]:
        result = self.request("batch.list_files", {"job_id": job_id})
        if not isinstance(result, list):
            raise ValueError("DATABENTO_FILE_LIST_FORMAT_INVALID")
        return [item for item in result if isinstance(item, dict)]


def _job_symbols(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, list):
        values = value
    else:
        return ()
    return tuple(sorted(str(item).strip() for item in values if str(item).strip()))


def _same_scope(job: dict[str, Any], scope: dict[str, Any]) -> bool:
    return (
        job.get("dataset") == scope["dataset"]
        and job.get("schema") == scope["schema"]
        and _job_symbols(job.get("symbols")) == tuple(scope["symbols"])
        and str(job.get("start", ""))[:19] == scope["start"][:19]
        and str(job.get("end", ""))[:19] == scope["end"][:19]
        and job.get("stype_in", "raw_symbol") == scope["stype_in"]
        and job.get("stype_out", "instrument_id") == "instrument_id"
        and job.get("encoding", "csv") == "csv"
        and job.get("compression", "zstd") == "zstd"
        and job.get("pretty_px", False) is False
        and job.get("pretty_ts", False) is False
        and job.get("map_symbols", True) is True
        and job.get("split_symbols", True) is True
        and job.get("split_duration", "none") in {"none", None}
        and job.get("delivery", "download") == "download"
    )


def _scope_overlap(job: dict[str, Any], scope: dict[str, Any]) -> bool:
    if job.get("dataset") != scope["dataset"] or job.get("schema") != scope["schema"]:
        return False
    if not set(_job_symbols(job.get("symbols"))).intersection(scope["symbols"]):
        return False
    try:
        j0 = str(job["start"])[:10]
        j1 = str(job["end"])[:10]
        s0, s1 = scope["start"][:10], scope["end"][:10]
        return j0 < s1 and s0 < j1
    except (KeyError, TypeError):
        return True


def _public_job_fields(job: dict[str, Any]) -> dict[str, Any]:
    safe = ("id", "state", "dataset", "schema", "symbols", "start", "end", "cost_usd",
            "record_count", "billed_size", "actual_size", "package_size", "ts_received",
            "ts_process_start", "ts_process_done", "ts_expiration", "progress")
    return {name: job[name] for name in safe if name in job}


def _load_plan() -> dict[str, Any]:
    plan_path = RUN_ROOT / "SOURCE_DOWNLOAD_PLAN.json"
    if plan_path.is_symlink() or not plan_path.is_file():
        raise ValueError("GC_DOWNLOAD_PLAN_MISSING")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "gx1_gc_order_flow_download_plan_v1":
        raise ValueError("GC_DOWNLOAD_PLAN_SCHEMA_INVALID")
    expected = hashlib.sha256(json.dumps(plan["scopes"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if plan.get("scopes_sha256") != expected:
        raise ValueError("GC_DOWNLOAD_PLAN_HASH_INVALID")
    if Decimal(plan["hard_cost_cap_usd"]) != HARD_CAP:
        raise ValueError("GC_DOWNLOAD_PLAN_CAP_INVALID")
    if Decimal(plan["quoted_total_usd"]) != validate_budget(plan["scopes"]):
        raise ValueError("GC_DOWNLOAD_PLAN_TOTAL_INVALID")
    for path_key, hash_key in (("raw_quote_path", "raw_quote_sha256"),
                               ("support_quote_path", "support_quote_sha256"),
                               ("raw_metadata_requote_path", "raw_metadata_requote_sha256")):
        source = Path(plan[path_key])
        if source.is_symlink() or not source.is_file() or sha256_file(source) != plan[hash_key]:
            raise ValueError("GC_DOWNLOAD_PLAN_SOURCE_HASH_CHANGED")
    return plan


def prepare() -> dict[str, Any]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))["gc_order_flow_research_20261006"]
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    guard = config["budget_guard"]
    if not (policy.get("source_fetch_authorized") is True and policy.get("spending_authorized") is True
            and guard.get("download_authorized") is True and guard.get("spending_authorized") is True
            and config.get("scope", {}).get("source_fetch") is True
            and config.get("scope", {}).get("spending") is True):
        raise ValueError("GC_DOWNLOAD_NOT_AUTHORIZED_IN_BOUND_PLAN")
    raw_bytes = RAW_QUOTE_PATH.read_bytes()
    support_bytes = SUPPORT_QUOTE_PATH.read_bytes()
    if RAW_REQUOTE_PATH.is_symlink() or not RAW_REQUOTE_PATH.is_file():
        raise ValueError("GC_FRESH_RAW_METADATA_REQUOTE_MISSING")
    raw_requote_bytes = RAW_REQUOTE_PATH.read_bytes()
    raw_requote = json.loads(raw_requote_bytes)
    raw_quote, support_quote = json.loads(raw_bytes), json.loads(support_bytes)
    scopes = build_scopes(raw_quote, support_quote)
    total = validate_budget(scopes)
    if not _close_cost(total, Decimal("119.65511028468302")):
        raise ValueError("GC_EXPECTED_QUOTE_CHANGED")
    if Decimal(str(guard.get("hard_cost_cap_usd"))) != HARD_CAP:
        raise ValueError("GC_PLAN_HARD_CAP_NOT_120_USD")
    if raw_quote.get("account_free_balance_verified") is not False:
        raise ValueError("GC_BALANCE_EVIDENCE_CHANGED")
    if raw_quote.get("data_downloaded") is not False:
        raise ValueError("GC_SOURCE_ALREADY_MARKED_DOWNLOADED")
    scopes_json = json.dumps(scopes, sort_keys=True, separators=(",", ":"))
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if RUN_ROOT.is_symlink() or SOURCE_ROOT.is_symlink():
        raise ValueError("GC_OUTPUT_PATH_SYMLINK")
    plan = {
        "schema_version": "gx1_gc_order_flow_download_plan_v1",
        "status": "AUTHORIZED_BOUNDED_FETCH_PENDING",
        "authorized_by": "operator_request_2026-10-15_download_dataset_to_usd_120",
        "hard_cost_cap_usd": "120.00",
        "operator_reported_remaining_credits_usd": "125.00",
        "account_balance_independently_verified": False,
        "pay_as_you_go_authorized": False,
        "payment_card_charge_authorized": False,
        "quoted_total_usd": str(total),
        "quoted_remaining_under_cap_usd": str(HARD_CAP - total),
        "quoted_remaining_under_reported_credits_usd": str(REPORTED_CREDITS - total),
        "dataset": "GLBX.MDP3",
        "market_data_schemas": ["mbp-1", "definition", "ohlcv-1d"],
        "study_window_utc": {"start_inclusive": "2024-05-01T00:00:00Z", "end_exclusive": "2026-07-01T00:00:00Z"},
        "support_window_utc": {"start_inclusive": "2024-04-01T00:00:00Z", "end_exclusive": "2026-07-01T00:00:00Z"},
        "raw_quote_path": str(RAW_QUOTE_PATH), "raw_quote_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "raw_metadata_requote_path": str(RAW_REQUOTE_PATH),
        "raw_metadata_requote_sha256": hashlib.sha256(raw_requote_bytes).hexdigest(),
        "raw_billable_binary_bytes": raw_requote["total_billable_bytes"],
        "support_quote_path": str(SUPPORT_QUOTE_PATH), "support_quote_sha256": hashlib.sha256(support_bytes).hexdigest(),
        "source_root": str(SOURCE_ROOT), "run_root": str(RUN_ROOT),
        "test_accessed": False, "research_fit_authorized": False,
        "native_architecture_change_authorized": False, "trading_or_broker_authorized": False,
        "scopes": scopes, "scopes_sha256": hashlib.sha256(scopes_json.encode()).hexdigest(),
        "download_module_path": str(Path(__file__).resolve()),
        "download_module_sha256": sha256_file(Path(__file__).resolve()),
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(RUN_ROOT / "SOURCE_DOWNLOAD_PLAN.json", plan)
    return {k: plan[k] for k in ("status", "hard_cost_cap_usd", "quoted_total_usd", "quoted_remaining_under_cap_usd", "quoted_remaining_under_reported_credits_usd", "market_data_schemas", "scopes_sha256")}


def _require_capped_producer() -> None:
    if os.environ.get("GX1_CAPPED_CLASS") != "producer":
        raise ValueError("GC_DOWNLOAD_REQUIRES_GX1_CAPPED_PRODUCER")


def submit(confirm_cap: str) -> dict[str, Any]:
    _require_capped_producer()
    if Decimal(confirm_cap) != HARD_CAP:
        raise ValueError("GC_EXACT_120_USD_OPERATOR_CAP_REQUIRED")
    plan = _load_plan()
    client = HistoricalAPI(_env_key())
    jobs_dir = RUN_ROOT / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    existing = client.list_jobs()
    for job in existing:
        for scope in plan["scopes"]:
            if _scope_overlap(job, scope) and not _same_scope(job, scope):
                raise ValueError("GC_OVERLAPPING_EXISTING_DATABENTO_JOB_REQUIRES_RECONCILIATION")
    submitted = 0
    estimated_total = Decimal("0")
    for scope in plan["scopes"]:
        receipt_path = jobs_dir / f"{scope['scope_id']}.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("scope") != scope or not receipt.get("job_id"):
                raise ValueError("GC_EXISTING_JOB_RECEIPT_MISMATCH")
            estimated_total += Decimal(scope["quoted_cost_usd"])
            continue
        matches = [job for job in existing if _same_scope(job, scope)]
        if len(matches) > 1:
            raise ValueError("GC_DUPLICATE_EXISTING_DATABENTO_JOBS")
        if matches:
            job = matches[0]
            response = _public_job_fields(job)
            job_id = job.get("id")
            reused = True
        else:
            try:
                response = client.submit(scope)
            except RuntimeError as exc:
                if "TRANSPORT_UNKNOWN" in str(exc):
                    # Resolve an unknown submit outcome from the free account job list; never retry blindly.
                    found = [job for job in client.list_jobs() if _same_scope(job, scope)]
                    if len(found) == 1:
                        response = _public_job_fields(found[0])
                    else:
                        atomic_json(RUN_ROOT / "SUBMIT_OUTCOME_UNKNOWN.json", {"scope": scope, "error": str(exc), "found_exact_jobs": len(found)})
                        raise
                else:
                    raise
            job_id = response.get("id")
            reused = False
            if not isinstance(job_id, str):
                raise ValueError("GC_JOB_ID_MISSING")
        receipt = {
            "schema_version": "gx1_gc_order_flow_batch_job_receipt_v1",
            "scope": scope, "job_id": job_id, "reused_existing_exact_job": reused,
            "quoted_cost_usd": scope["quoted_cost_usd"],
            "batch_response": _public_job_fields(response),
            "submitted_or_attached_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(receipt_path, receipt)
        estimated_total += Decimal(scope["quoted_cost_usd"])
        if estimated_total > HARD_CAP:
            raise ValueError("GC_ESTIMATED_SUBMITTED_TOTAL_EXCEEDS_CAP")
        submitted += int(not reused)
        if submitted and submitted % 5 == 0:
            print(f"submitted_or_attached {len(list(jobs_dir.glob('*.json')))} / {len(plan['scopes'])}; quoted_total_usd={estimated_total}", flush=True)
        if not reused:
            time.sleep(3.1)  # Provider batch submit limit: 20 requests per minute.
    summary = {
        "schema_version": "gx1_gc_order_flow_batch_submission_summary_v1",
        "status": "ALL_SCOPES_SUBMITTED_OR_EXACT_JOB_REUSED",
        "scope_count": len(plan["scopes"]), "new_submissions": submitted,
        "quoted_total_usd": str(estimated_total), "hard_cost_cap_usd": str(HARD_CAP),
        "jobs_root": str(jobs_dir), "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(RUN_ROOT / "SUBMISSION_SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def _sha_from_vendor(value: str) -> str:
    match = re.fullmatch(r"sha256:([0-9a-fA-F]{64})", value or "")
    if not match:
        raise ValueError("GC_VENDOR_SHA256_INVALID")
    return match.group(1).lower()


def _download_verified(client: HistoricalAPI, file_item: dict[str, Any], destination: Path) -> None:
    urls = file_item.get("urls")
    url = urls.get("https") if isinstance(urls, dict) else None
    parsed = urllib.parse.urlparse(url or "")
    if parsed.scheme != "https" or parsed.hostname not in {"api.databento.com", "hist.databento.com"}:
        raise ValueError("GC_VENDOR_DOWNLOAD_URL_INVALID")
    expected_size = int(file_item.get("size", 0))
    expected_hash = _sha_from_vendor(file_item.get("hash", ""))
    if expected_size <= 0:
        raise ValueError("GC_VENDOR_FILE_SIZE_INVALID")
    free = shutil.disk_usage(destination.parent).free
    if free < expected_size * 4 + 1024 * 1024 * 1024:
        raise ValueError("GC_DISK_RESERVE_INSUFFICIENT")
    request = urllib.request.Request(url, headers={"Authorization": client._authorization})
    tmp = destination.with_name(destination.name + ".part")
    if tmp.exists() or tmp.is_symlink():
        raise ValueError("GC_PARTIAL_FILE_NEEDS_OPERATOR_REVIEW")
    digest = hashlib.sha256()
    total = 0
    try:
        with urllib.request.urlopen(request, timeout=120) as response, tmp.open("xb") as out:
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > expected_size:
                    raise ValueError("GC_VENDOR_FILE_EXCEEDS_DECLARED_SIZE")
                digest.update(block)
                out.write(block)
            out.flush()
            os.fsync(out.fileno())
        if total != expected_size or digest.hexdigest() != expected_hash:
            raise ValueError("GC_VENDOR_FILE_HASH_OR_SIZE_MISMATCH")
        os.replace(tmp, destination)
    except BaseException:
        # Preserve a partial for forensic review; never retry over it implicitly.
        raise


def _decompress_csv_zstd(source: Path, destination: Path, schema: str) -> tuple[str, list[str], int, str]:
    if not pa.Codec.is_available("zstd"):
        raise ValueError("GC_ZSTD_CODEC_UNAVAILABLE")
    raw_header = bytearray()
    digest = hashlib.sha256()
    decompressed_bytes = 0
    fd, tmp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".part", dir=destination.parent)
    os.close(fd)
    try:
        with source.open("rb") as source_stream:
            compressed = pa.CompressedInputStream(pa.input_stream(source_stream), "zstd")
            with open(tmp_name, "wb") as raw_output:
                with gzip.GzipFile(fileobj=raw_output, mode="wb", mtime=0) as gz:
                    while True:
                        block = compressed.read(8 * 1024 * 1024)
                        if not block:
                            break
                        if len(raw_header) < 1024 * 1024:
                            remainder = 1024 * 1024 - len(raw_header)
                            raw_header.extend(block[:remainder])
                        gz.write(block)
                        decompressed_bytes += len(block)
                    raw_output.flush()
                    os.fsync(raw_output.fileno())
        header_line = bytes(raw_header).split(b"\n", 1)[0].decode("utf-8-sig").strip()
        sample_lines = bytes(raw_header).decode("utf-8-sig").splitlines()
        fields = next(csv.reader([header_line])) if header_line else []
        if schema == "mbp-1" and (not MBP_FIELDS.issubset(set(fields)) or "symbol" not in fields):
            raise ValueError("GC_MBP1_CSV_HEADER_INVALID")
        if len(sample_lines) < 2 or "symbol" not in fields:
            raise ValueError("GC_SYMBOL_FIELD_OR_DATA_ROW_MISSING")
        first_row = next(csv.reader([sample_lines[1]]))
        symbol = first_row[fields.index("symbol")] if len(first_row) > fields.index("symbol") else ""
        if RAW_SYMBOL_RE.fullmatch(symbol) is None:
            raise ValueError("GC_OUTPUT_RAW_SYMBOL_INVALID")
        with open(tmp_name, "rb") as check:
            for block in iter(lambda: check.read(8 * 1024 * 1024), b""):
                digest.update(block)
        os.replace(tmp_name, destination)
        return digest.hexdigest(), fields, decompressed_bytes, symbol
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def download() -> dict[str, Any]:
    _require_capped_producer()
    plan = _load_plan()
    jobs_dir = RUN_ROOT / "jobs"
    if not (RUN_ROOT / "SUBMISSION_SUMMARY.json").is_file():
        raise ValueError("GC_SUBMISSIONS_INCOMPLETE")
    client = HistoricalAPI(_env_key())
    total_actual = Decimal("0")
    done = pending = downloaded_files = 0
    job_receipts = list(sorted(jobs_dir.glob("*.json")))
    if len(job_receipts) != len(plan["scopes"]):
        raise ValueError("GC_JOB_RECEIPT_COUNT_INVALID")
    details_by_receipt: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    projected_total = sum((Decimal(scope["quoted_cost_usd"]) for scope in plan["scopes"]), Decimal("0"))
    for receipt_path in job_receipts:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        scope, job_id = receipt["scope"], receipt["job_id"]
        details = client.job_details(job_id)
        if not _same_scope(details, scope):
            raise ValueError("GC_BATCH_JOB_DETAIL_SCOPE_MISMATCH")
        state = details.get("state")
        if state in {"failed", "expired"}:
            raise ValueError("GC_BATCH_JOB_FAILED_OR_EXPIRED")
        details_by_receipt[receipt_path.name] = (receipt, details)
        if state != "done":
            pending += 1
            continue
        done += 1
        cost = details.get("cost_usd")
        if cost is not None:
            actual = Decimal(str(cost))
            total_actual += actual
            projected_total += actual - Decimal(scope["quoted_cost_usd"])
        else:
            total_actual += Decimal(scope["quoted_cost_usd"])
    if projected_total > HARD_CAP:
        raise ValueError("GC_ACTUAL_PLUS_RESERVED_JOB_COSTS_EXCEED_120_USD")
    for receipt_path in job_receipts:
        receipt, details = details_by_receipt[receipt_path.name]
        scope, job_id = receipt["scope"], receipt["job_id"]
        if details.get("state") != "done":
            continue
        files = client.list_files(job_id)
        dest_root = SOURCE_ROOT / scope["scope_id"]
        dest_root.mkdir(parents=True, exist_ok=True)
        if dest_root.is_symlink():
            raise ValueError("GC_OUTPUT_PATH_SYMLINK")
        symbols_seen: set[str] = set()
        for item in files:
            filename = item.get("filename")
            if not isinstance(filename, str) or Path(filename).name != filename or filename in {".", ".."}:
                raise ValueError("GC_VENDOR_FILENAME_INVALID")
            if filename == "metadata.json":
                target = dest_root / filename
                receipt_file = dest_root / "metadata.receipt.json"
                if target.exists():
                    if target.stat().st_size != int(item["size"]) or sha256_file(target) != _sha_from_vendor(item["hash"]):
                        raise ValueError("GC_EXISTING_METADATA_FILE_MISMATCH")
                else:
                    _download_verified(client, item, target)
                atomic_json(receipt_file, {"filename": filename, "size": target.stat().st_size,
                                            "sha256": sha256_file(target), "job_id": job_id})
                continue
            if not filename.endswith(".csv.zst"):
                raise ValueError("GC_VENDOR_ENCODING_NOT_CSV_ZSTD")
            raw_target = dest_root / filename
            if raw_target.exists():
                if raw_target.stat().st_size != int(item["size"]) or sha256_file(raw_target) != _sha_from_vendor(item["hash"]):
                    raise ValueError("GC_EXISTING_VENDOR_FILE_MISMATCH")
            else:
                _download_verified(client, item, raw_target)
            output_name = filename[:-4] + ".gz"
            output_target = dest_root / output_name
            file_receipt = dest_root / (output_name + ".receipt.json")
            if output_target.exists() and file_receipt.exists():
                old = json.loads(file_receipt.read_text(encoding="utf-8"))
                if (old.get("sha256") != sha256_file(output_target)
                        or old.get("provider_sha256") != _sha_from_vendor(item["hash"])
                        or old.get("scope_id") != scope["scope_id"]):
                    raise ValueError("GC_EXISTING_LOCAL_FILE_MISMATCH")
                if old.get("symbol_from_first_record") not in scope["symbols"]:
                    raise ValueError("GC_EXISTING_LOCAL_SYMBOL_MISMATCH")
                symbols_seen.add(old["symbol_from_first_record"])
                continue
            if output_target.exists() or output_target.is_symlink():
                raise ValueError("GC_EXISTING_LOCAL_FILE_WITHOUT_RECEIPT")
            digest, fields, expanded, raw_symbol = _decompress_csv_zstd(raw_target, output_target, scope["schema"])
            if raw_symbol not in scope["symbols"]:
                raise ValueError("GC_DOWNLOADED_RAW_SYMBOL_OUTSIDE_SCOPE")
            symbols_seen.add(raw_symbol)
            info = {
                "schema_version": "gx1_gc_order_flow_local_file_receipt_v1",
                "provider": "databento", "dataset": scope["dataset"], "schema": scope["schema"],
                "scope_id": scope["scope_id"], "job_id": job_id,
                "symbols_requested": scope["symbols"], "start": scope["start"], "end": scope["end"],
                "provider_filename": filename, "provider_size": int(item["size"]),
                "provider_sha256": _sha_from_vendor(item["hash"]), "provider_compression": "zstd",
                "symbol_from_first_record": raw_symbol,
                "local_path": str(output_target), "bytes": output_target.stat().st_size,
                "sha256": digest, "decompressed_csv_bytes": expanded, "header_fields": fields,
                "compression": "gzip", "pretty_px": False, "pretty_ts": False,
                "batch_job_actual_cost_usd": details.get("cost_usd"),
                "batch_job_billed_size": details.get("billed_size"),
                "downloaded_utc": datetime.now(timezone.utc).isoformat(),
                "source_qualified": False, "admitted_to_model_or_economic_test": False,
            }
            atomic_json(file_receipt, info)
            downloaded_files += 1
            if downloaded_files % 5 == 0:
                print(f"verified_local_data_files {downloaded_files}; completed_jobs {done}/{len(job_receipts)}; pending_jobs {pending}", flush=True)
        if symbols_seen != set(scope["symbols"]):
            raise ValueError("GC_BATCH_JOB_SYMBOL_FILE_SET_MISMATCH")
    state = "ALL_BATCH_JOBS_COMPLETE_FILES_DOWNLOADED" if pending == 0 else "PARTIAL_BATCH_JOBS_PENDING"
    summary = {
        "schema_version": "gx1_gc_order_flow_download_summary_v1", "status": state,
        "scope_count": len(plan["scopes"]), "completed_jobs": done, "pending_jobs": pending,
        "newly_downloaded_data_files": downloaded_files, "actual_or_quoted_completed_job_cost_usd": str(total_actual),
        "hard_cost_cap_usd": str(HARD_CAP), "source_root": str(SOURCE_ROOT),
        "test_accessed": False, "research_fit_authorized": False,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(RUN_ROOT / "DOWNLOAD_SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "submit", "download"))
    parser.add_argument("--confirm-cap-usd", default="")
    args = parser.parse_args()
    try:
        if args.phase == "prepare":
            result = prepare()
        elif args.phase == "submit":
            result = submit(args.confirm_cap_usd)
        else:
            result = download()
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f"GC_DATABENTO_DOWNLOAD_BLOCKED:{exc}")
        return 75
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
