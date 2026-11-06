"""Bind full-TRAIN XAU M1 source gaps to verified published regular closures.

Only an observed gap wholly inside one published closed interval is exact.
Every other observed gap remains unknown and right-censors later target work.
For integrity it hashes the entire pre-TEST Parquet as opaque bytes, including
price and VAL bytes. It decodes and uses only TRAIN M1 time; no target or model.
"""

from __future__ import annotations

import argparse
import base64
from bisect import bisect_right
from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import signal
import stat
import subprocess
from zoneinfo import ZoneInfo


STAGE = "full_train_xau_closure_authority_20261107_001"
PLAN_SCHEMA = "gx1_full_train_xau_closure_authority_plan_v1"
CLAIM_SCHEMA = "gx1_full_train_xau_closure_authority_claim_v1"
BUDGET = {"memory_max": 4294967296, "swap": 536870912, "pids": 64,
          "numerical_threads": 1, "max_wall_seconds": 1800,
          "max_invocations": 1, "max_json_bytes": 16777216}
CALENDAR_ROOT = Path("/home/andre2/GX1_RUNS/XAU_2025_PUBLISHED_CALENDAR_EXTENSION_20261106_003")
M1_ROOT = Path("/home/andre2/GX1_DATA/data/data/prebuilt/HISTORY2009W_BOOTSTRAP_20260927/"
               "artifacts/pretest_direct_m1_quotes_source")
CALENDAR_REFS = {
    "published_calendar": {"path": str(CALENDAR_ROOT / "PUBLISHED_CALENDAR.json"),
                           "sha256": "6911f30d3c7aa013c8ad2a5c31b5e2535a872cd54613ced70b4b812d4226e3a7"},
    "result": {"path": str(CALENDAR_ROOT / "RESULT.json"),
               "sha256": "e590bc92c7e26c5292b8c4e8ccd039e4402643b5532b0daa16fd6e72acdf4083"},
    "terminal": {"path": str(CALENDAR_ROOT / "TERMINAL.json"),
                 "sha256": "a204da0dc4f1e8733702c0b9fc2f516252c419504896a834fa63b61fbeff243e"},
}
M1_REFS = {
    "source": {"path": str(M1_ROOT / "m1_quotes.parquet"),
               "sha256": "54b66c5e9b5a0125f089afcc9d3327c79af9c3543eebdb2644281ed9d3c2734c"},
    "manifest": {"path": str(M1_ROOT / "manifest.json"),
                 "sha256": "42e2b20bbb3b1661c704352003461583eab488bba6dbacbd8404a134f060de1d"},
}
TRAIN_START = "2024-05-01T00:00:00+00:00"
TRAIN_END = "2026-01-01T00:00:00+00:00"
LAST_M1_BAR = "2025-12-31T21:59:00+00:00"
CALENDAR_SHA256 = "7528dc475d89cc67a1b16c1cf2158d9fa008c4f44eaabcdc21b620644c393fda"
ROW_GROUP_START = 1820
ROW_GROUP_END = 2025
EXPECTED_ROWS = 591682
MAX_SMALL_FILE = 16 * 1024 * 1024
MINUTE_NS = 60_000_000_000


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError("FULL_TRAIN_XAU_CLOSURE_" + reason)


def _regular(path: Path) -> tuple[int, ...]:
    require(path.is_absolute() and ".." not in path.parts, "PATH")
    require(not any(part.is_symlink() for part in (path, *path.parents)), "SYMLINK")
    info = path.stat()
    require(stat.S_ISREG(info.st_mode), "REGULAR_FILE")
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _checked_ref(ref: dict, *, max_bytes: int | None = MAX_SMALL_FILE) -> tuple[bytes | None, tuple[int, ...]]:
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "REF_SHAPE")
    path = Path(ref["path"])
    before = _regular(path)
    require(max_bytes is None or before[3] <= max_bytes, "REF_SIZE")
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if max_bytes is not None else None
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
            if chunks is not None:
                chunks.append(block)
    require(before == _regular(path), "REF_CHANGED")
    require(digest.hexdigest() == ref["sha256"], "REF_SHA256")
    return (b"".join(chunks) if chunks is not None else None), before


def _read_json(ref: dict) -> dict:
    raw, _ = _checked_ref(ref)
    require(raw is not None, "JSON_READ")
    value = json.loads(raw)
    require(isinstance(value, dict), "JSON_SHAPE")
    return value


def _ref(path: Path, data: bytes) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


def _encode(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _check_plan(plan: dict, plan_ref: dict, out_dir: Path) -> None:
    require(set(plan) == {"schema_version", "policy_gate", "calendar", "m1",
                          "train_start_utc", "train_end_utc_exclusive", "row_group_start",
                          "row_group_end_exclusive", "expected_rows", "source_owners",
                          "output_root", "budget", "test_data_used"}, "PLAN_SHAPE")
    require(plan["schema_version"] == PLAN_SCHEMA and plan["policy_gate"] == STAGE
            and plan["calendar"] == CALENDAR_REFS and plan["m1"] == M1_REFS
            and plan["train_start_utc"] == TRAIN_START and plan["train_end_utc_exclusive"] == TRAIN_END
            and plan["row_group_start"] == ROW_GROUP_START
            and plan["row_group_end_exclusive"] == ROW_GROUP_END
            and plan["expected_rows"] == EXPECTED_ROWS and plan["budget"] == BUDGET
            and plan["test_data_used"] is False and plan["output_root"] == str(out_dir), "PLAN_BINDING")
    root = Path("/home/andre2/GX1_RUNS")
    require(out_dir.is_absolute() and root in out_dir.parents and ".." not in out_dir.parts
            and not any(path.is_symlink() for path in (out_dir, *out_dir.parents))
            and out_dir.is_dir(), "OUTPUT_SCOPE")
    repo = Path(__file__).resolve().parents[2]
    paths = {"worker": Path(__file__).resolve(),
             "closure_owner": repo / "gx1/contracts/unified_exit_market_closure_authority_v1.py",
             "publication_owner": repo / "gx1/contracts/immutable_event_authority_v1.py",
             "capped_owner": repo / "gx1/contracts/gx1_capped_execution_v1.py"}
    require(isinstance(plan["source_owners"], dict) and set(plan["source_owners"]) == set(paths), "OWNERS")
    for name, path in paths.items():
        require(plan["source_owners"][name].get("path") == str(path), "OWNER_PATH")
        _checked_ref(plan["source_owners"][name])
    _checked_ref(plan_ref)


def _check_claim(out_dir: Path, plan_ref: dict, worker_ref: dict) -> tuple[dict, dict]:
    path = out_dir / "CLAIM.json"
    before = _regular(path)
    require(before[3] <= MAX_SMALL_FILE, "CLAIM_SIZE")
    raw = path.read_bytes()
    require(before == _regular(path), "CLAIM_CHANGED")
    claim_ref = _ref(path, raw)
    claim = json.loads(raw)
    require(isinstance(claim, dict) and set(claim) == {
        "schema_version", "plan", "worker", "policy", "source_head",
        "source_only_closure", "test_data_used",
    } and claim["schema_version"] == CLAIM_SCHEMA and claim["plan"] == plan_ref
            and claim["worker"] == worker_ref and claim["source_only_closure"] is True
            and claim["test_data_used"] is False, "CLAIM_BINDING")
    require({item.name for item in out_dir.iterdir()} == {"CLAIM.json"}, "OUTPUT_NOT_EMPTY")
    return claim, claim_ref


def _check_admission(plan: dict, plan_ref: dict, claim: dict) -> dict:
    repo = Path(__file__).resolve().parents[2]
    require(Path.cwd() == repo, "WORKDIR")
    require(subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
            == "work/gx1-current", "BRANCH")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    require(claim["source_head"] == head and
            subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True) == "", "HEAD_CLEAN")
    policy_ref = claim["policy"]
    require(isinstance(policy_ref, dict) and policy_ref.get("path") == str(repo / "NEXT_RUN_POLICY.json"),
            "POLICY_PATH")
    policy = _read_json(policy_ref)
    gate = policy[STAGE]
    expected = {"decision": "ADMITTED", "plan": plan_ref, "execution_authorized": True,
                "data_reads_authorized": True, "target_builds_authorized": 0,
                "model_calls_authorized": 0, "test_authorized": False,
                "claimed_invocations": 0, "completed_invocations": 0,
                "max_invocations": 1, "training_authorized": False,
                "output_root": plan["output_root"]}
    require(policy["training_enabled"] is False and isinstance(gate, dict)
            and all(gate.get(key) == value for key, value in expected.items()), "POLICY_ADMISSION")
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry,
    )
    limits = require_capped_cpu_audit_execution()
    limits["lock"] = require_capped_lock_ancestry()
    require(all(limits[key] == BUDGET[key] for key in
                ("memory_max", "swap", "pids", "numerical_threads")), "CAPS")
    return limits


def _utc(text: str) -> datetime:
    value = datetime.fromisoformat(text)
    require(value.tzinfo is not None and value.utcoffset() == timedelta(0), "UTC")
    return value


def _calendar_intervals(calendar: dict, owner) -> list[dict]:
    require(calendar["schema_version"] == "gx1_dynamic_mtf_state_published_calendar_v1"
            and calendar["calendar_sha256"] == CALENDAR_SHA256
            and owner.canonical_sha256({key: value for key, value in calendar.items()
                                        if key != "calendar_sha256"}) == CALENDAR_SHA256
            and calendar["instrument"] == "XAU_USD" and calendar["timezone"] == "America/New_York"
            and calendar["interval_semantics"] == "left_closed_right_open_utc"
            and calendar["regular_rule"] == {"session_open_local": "18:05",
                                             "session_close_local": "16:59",
                                             "weekly_sessions": "Sunday_to_Friday"}
            and calendar["coverage_start_utc"] <= TRAIN_START
            and calendar["coverage_end_utc_exclusive"] == TRAIN_END
            and calendar["test_data_used"] is False, "CALENDAR_HEADER")
    tz_path = Path("/usr/share/zoneinfo/America/New_York")
    _checked_ref({"path": str(tz_path), "sha256": calendar["timezone_database_file_sha256"]})
    with tz_path.open("rb") as stream:
        ny = ZoneInfo.from_file(stream, key="America/New_York")
    veto = {date.fromisoformat(value) for value in calendar["holiday_veto_dates"]}
    require(len(veto) == len(calendar["holiday_veto_dates"]) == 23, "CALENDAR_VETO")
    documents = calendar["source_documents"]
    require(len(documents) == 16 and {document["role"] for document in documents}
            == {"regular", "holiday"}, "CALENDAR_DOCUMENTS")
    for document in documents:
        ref = {key: document[key] for key in ("path", "sha256")}
        raw, _ = _checked_ref(ref)
        require(raw is not None and base64.b32encode(hashlib.sha1(raw).digest()).decode().rstrip("=")
                == document["cdx_sha1_digest"] and document["archive_url"].startswith(
                    "https://web.archive.org/web/" + document["capture_timestamp"])
                and document["original_url"] in {
                    "https://www.oanda.com/uk-en/trading/hours-of-operation/",
                    "https://www.oanda.com/uk-en/trading/holiday-trading-hours/",
                }, "CALENDAR_SOURCE")
    intervals = calendar["intervals"]
    require(len(intervals) == 415, "CALENDAR_INTERVAL_COUNT")
    previous_end = _utc(calendar["coverage_start_utc"])
    for item in intervals:
        require(set(item) == {"kind", "start_utc", "end_utc_exclusive", "source_event_id"}, "INTERVAL_SHAPE")
        left, right = _utc(item["start_utc"]), _utc(item["end_utc_exclusive"])
        require(previous_end <= left < right <= _utc(calendar["coverage_end_utc_exclusive"])
                and isinstance(item["source_event_id"], str) and bool(item["source_event_id"]), "INTERVAL_ORDER")
        local_left, local_right = left.astimezone(ny), right.astimezone(ny)
        require(local_left.strftime("%H:%M:%S") == "16:59:00"
                and local_right.strftime("%H:%M:%S") == "18:05:00", "LOCAL_BOUNDARY")
        daily = (item["kind"] == "daily_maintenance" and local_left.weekday() < 4
                 and local_left.date() == local_right.date())
        weekend = (item["kind"] == "weekend" and local_left.weekday() == 4
                   and local_right.date() == local_left.date() + timedelta(days=2))
        require(daily or weekend, "REGULAR_PERIOD")
        touched = {local_left.date() + timedelta(days=n)
                   for n in range((local_right.date() - local_left.date()).days + 1)}
        require(not touched & veto, "HOLIDAY_VETO")
        previous_end = right
    return intervals


def _load_train_clock(plan: dict, owner):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    manifest = _read_json(plan["m1"]["manifest"])
    require(manifest["instrument"] == "XAU_USD" and manifest["timeframe"] == "M1"
            and manifest["timestamp_semantics"] == "bar_start_utc"
            and manifest["output_parquet"] == plan["m1"]["source"]["path"]
            and manifest["output_parquet_sha256"] == plan["m1"]["source"]["sha256"]
            and manifest["row_count"] == 5959045 and manifest["quote_complete_m1"] is True
            and manifest["test_boundary_utc"] == "2026-07-01T00:00:00+00:00"
            and manifest["test_accessed"] is False and manifest["output_columns"][0] == "time", "M1_MANIFEST")
    _, source_stat = _checked_ref(plan["m1"]["source"], max_bytes=None)
    parquet = pq.ParquetFile(plan["m1"]["source"]["path"])
    metadata = parquet.metadata
    require(metadata.num_rows == manifest["row_count"] and metadata.num_row_groups > ROW_GROUP_END
            and str(parquet.schema_arrow.field("time").type) == "timestamp[ns, tz=UTC]", "M1_PARQUET")
    start, end = _utc(TRAIN_START), _utc(TRAIN_END)
    previous = metadata.row_group(ROW_GROUP_START - 1).column(0).statistics
    following = metadata.row_group(ROW_GROUP_END).column(0).statistics
    require(previous is not None and previous.max < start and following is not None
            and following.min >= end, "M1_GROUP_BOUNDARY")
    row_count, previous_max, pieces = 0, None, []
    physical_start = sum(metadata.row_group(index).num_rows for index in range(ROW_GROUP_START))
    for index in range(ROW_GROUP_START, ROW_GROUP_END):
        group = metadata.row_group(index)
        stats = group.column(0).statistics
        require(stats is not None and start <= stats.min <= stats.max < end
                and (previous_max is None or previous_max < stats.min), "M1_GROUP_STATS")
        values = pd.DatetimeIndex(parquet.read_row_group(index, columns=["time"]).column("time").to_pandas())
        require(len(values) == group.num_rows and values[0].to_pydatetime() == stats.min
                and values[-1].to_pydatetime() == stats.max, "M1_GROUP_ROWS")
        pieces.append(np.asarray(values.asi8, dtype="<i8"))
        row_count += group.num_rows
        previous_max = stats.max
    require(row_count == EXPECTED_ROWS, "M1_ROW_COUNT")
    clock = pd.DatetimeIndex(np.concatenate(pieces), tz="UTC")
    require(clock[0].isoformat() == TRAIN_START and clock[-1].isoformat() == LAST_M1_BAR,
            "M1_CLOCK_BOUNDARY")
    clock_sha = owner.m1_clock_sha256(clock)
    require(_regular(Path(plan["m1"]["source"]["path"])) == source_stat, "M1_SOURCE_CHANGED")
    return clock, clock_sha, physical_start, source_stat


def _certify_gaps(clock, calendar: dict, intervals: list[dict], physical_start: int, owner):
    import numpy as np
    import pandas as pd

    starts = [pd.Timestamp(item["start_utc"]).value for item in intervals]
    ends = [pd.Timestamp(item["end_utc_exclusive"]).value for item in intervals]
    times = np.asarray(clock.asi8, dtype=np.int64)
    gap_rows = np.flatnonzero(np.diff(times) != MINUTE_NS)
    certified, exact = [], []
    for raw_row in gap_rows:
        row = int(raw_row)
        left, right = int(times[row] + MINUTE_NS), int(times[row + 1])
        position = bisect_right(starts, left) - 1
        if position < 0 or right > ends[position]:
            continue
        item = intervals[position]
        gap = {"start_utc": pd.Timestamp(left, tz="UTC").isoformat(),
               "end_utc_exclusive": pd.Timestamp(right, tz="UTC").isoformat()}
        proof = {"gap_after_slice_row": row,
                 "gap_after_physical_source_row": physical_start + row,
                 "observed_gap": gap, "published_full_closed_period": dict(item),
                 "published_calendar_sha256": calendar["calendar_sha256"],
                 "relation": "observed_gap_fully_contained_in_published_closed_period"}
        event_id = owner.canonical_sha256(proof)
        certified.append({**proof, "source_event_id": event_id})
        exact.append({"kind": item["kind"], **gap, "source_event_id": event_id})
    require(bool(certified), "NO_CERTIFIED_GAP")
    return certified, exact, len(gap_rows)


def run(plan_path: Path, plan_sha256: str, out_dir: Path) -> dict:
    plan_ref = {"path": str(plan_path), "sha256": plan_sha256}
    plan = _read_json(plan_ref)
    _check_plan(plan, plan_ref, out_dir)
    claim, claim_ref = _check_claim(out_dir, plan_ref, plan["source_owners"]["worker"])
    limits = _check_admission(plan, plan_ref, claim)
    def deadline(*_args):
        raise TimeoutError("FULL_TRAIN_CLOSURE_DEADLINE")
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(BUDGET["max_wall_seconds"])
    from gx1.contracts import unified_exit_market_closure_authority_v1 as owner
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    result = _read_json(plan["calendar"]["result"])
    terminal = _read_json(plan["calendar"]["terminal"])
    calendar = _read_json(plan["calendar"]["published_calendar"])
    require(result["decision"] == "PASS_SOURCE_CALENDAR_ONLY"
            and result["published_calendar"] == plan["calendar"]["published_calendar"]
            and result["target_data_read"] is False and result["test_data_used"] is False
            and terminal["status"] == "COMPLETE" and terminal["exit_code"] == 0
            and terminal["result"] == plan["calendar"]["result"]
            and terminal["published_calendar"] == plan["calendar"]["published_calendar"]
            and terminal["test_data_used"] is False, "CALENDAR_TERMINAL")
    intervals = _calendar_intervals(calendar, owner)
    clock, clock_sha, physical_start, source_stat = _load_train_clock(plan, owner)
    certified, exact, observed = _certify_gaps(clock, calendar, intervals, physical_start, owner)
    source_reference = {
        "schema_version": "gx1_full_train_xau_external_gap_coverage_reference_v1",
        "calendar": plan["calendar"]["published_calendar"],
        "calendar_result": plan["calendar"]["result"],
        "calendar_terminal": plan["calendar"]["terminal"],
        "published_calendar_sha256": calendar["calendar_sha256"],
        "m1_source": plan["m1"]["source"], "m1_manifest": plan["m1"]["manifest"],
        "m1_clock_sha256": clock_sha, "physical_source_row_start": physical_start,
        "selected_row_groups": [ROW_GROUP_START, ROW_GROUP_END],
        "coverage_start_utc": TRAIN_START, "coverage_end_utc_exclusive": TRAIN_END,
        "certified_gaps": certified, "uncovered_gap_count": observed - len(certified),
        "semantics": "Only observed gaps wholly contained in independently published regular closures are exact. All other gaps remain unknown; no holiday endpoint or automatic FLAT is inferred.",
        "test_data_used": False, "target_data_read": False, "model_calls": 0,
    }
    source_path = out_dir / "SOURCE_REFERENCE.json"
    source_bytes = _encode(source_reference)
    source_ref = _ref(source_path, source_bytes)
    schedule = owner.seal_exact_market_schedule({
        "schema_version": owner.MARKET_CLOSURE_SCHEDULE_SCHEMA_VERSION,
        "decision": "PASS", "instrument": "XAU_USD", "timeframe": "M1",
        "coverage_start_utc": TRAIN_START, "coverage_end_utc_exclusive": TRAIN_END,
        "interval_semantics": "left_closed_right_open_utc",
        "source_method": "externally_sourced_exact_xau_utc_closure_intervals_v1",
        "source_reference_sha256": source_ref["sha256"], "intervals": exact,
        "test_data_used": False,
    })
    schedule_path = out_dir / "exact_schedule.json"
    schedule_bytes = _encode(schedule)
    schedule_ref = _ref(schedule_path, schedule_bytes)
    authority = owner.build_market_closure_authority(
        m1_times=clock, m1_source_path=Path(plan["m1"]["source"]["path"]),
        m1_source_sha256=plan["m1"]["source"]["sha256"],
        m1_source_manifest_path=Path(plan["m1"]["manifest"]["path"]),
        m1_source_manifest_sha256=plan["m1"]["manifest"]["sha256"],
        exact_schedule=schedule, exact_schedule_path=schedule_path,
        exact_schedule_file_sha256=schedule_ref["sha256"],
    )
    require(authority["observed_gap_count"] == observed
            and authority["known_market_closure_count"] == len(exact)
            and authority["unknown_source_gap_count"] == observed - len(exact), "AUTHORITY_PARITY")
    authority_path = out_dir / "closure_authority.json"
    authority_bytes = _encode(authority)
    authority_ref = _ref(authority_path, authority_bytes)
    output = {
        "schema_version": "gx1_full_train_xau_closure_authority_result_v1",
        "decision": "PASS_SOURCE_GAP_AUTHORITY_ONLY", "plan": plan_ref, "claim": claim_ref,
        "policy": claim["policy"], "source_head": claim["source_head"], "limits": limits,
        "calendar": plan["calendar"], "m1": plan["m1"],
        "coverage_start_utc": TRAIN_START, "coverage_end_utc_exclusive": TRAIN_END,
        "first_m1_bar_utc": clock[0].isoformat(), "last_m1_bar_utc": clock[-1].isoformat(),
        "m1_clock_sha256": clock_sha, "m1_rows": len(clock),
        "observed_gap_count": observed, "known_market_closure_count": len(exact),
        "unknown_source_gap_count": observed - len(exact),
        "source_reference": source_ref, "exact_schedule": schedule_ref,
        "closure_authority": authority_ref,
        "test_data_used": False, "target_data_read": False, "model_calls": 0,
        "source_read_scope": "Whole pre-TEST Parquet hashed as opaque bytes for integrity, including price and VAL bytes; only time from TRAIN row groups 1820..2024 decoded and used. No TEST rows decoded or used.",
        "limit": "Published regular closure plus actual M1 gap containment certifies only those source discontinuities; unknown gaps right-censor. No target, forecast, edge or trade claim.",
    }
    result_path = out_dir / "RESULT.json"
    result_bytes = _encode(output)
    require(sum(map(len, (source_bytes, schedule_bytes, authority_bytes, result_bytes)))
            <= BUDGET["max_json_bytes"], "JSON_BUDGET")
    _check_plan(plan, plan_ref, out_dir)
    require(_check_claim(out_dir, plan_ref, plan["source_owners"]["worker"]) == (claim, claim_ref),
            "CLAIM_CHANGED")
    require(_check_admission(plan, plan_ref, claim) == limits, "CAPS_CHANGED")
    require(_regular(Path(plan["m1"]["source"]["path"])) == source_stat, "M1_SOURCE_CHANGED")
    for binding in (*plan["calendar"].values(), plan["m1"]["manifest"]):
        _checked_ref(binding)
    require(_calendar_intervals(calendar, owner) == intervals, "CALENDAR_CHANGED")
    for path, data in ((source_path, source_bytes), (schedule_path, schedule_bytes),
                       (authority_path, authority_bytes), (result_path, result_bytes)):
        _publish_bytes(path, data)
        path.chmod(0o444)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.plan, args.plan_sha256, args.out_dir), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
