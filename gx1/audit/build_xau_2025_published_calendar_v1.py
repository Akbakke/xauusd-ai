"""Extend the verified OANDA XAU regular-closure calendar with archived sources.

This source-only worker publishes a calendar, not target labels or a model. Holiday
exceptions veto whole regular periods; their undocumented endpoints stay unknown.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
from html.parser import HTMLParser
import json
import re
import stat
import subprocess
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zoneinfo import ZoneInfo


SCHEMA = "gx1_xau_2025_published_calendar_extension_plan_v1"
POLICY_GATE_PREFIX = "xau_2025_published_calendar_extension_20261106_"
CLAIM_SCHEMA = "gx1_xau_2025_published_calendar_extension_claim_v1"
MAX_SOURCE_BYTES = 16 * 1024 * 1024
OLD_CALENDAR_PATH = Path(
    "/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/"
    "OANDA_CALENDAR_DOWNLOAD_001/PUBLISHED_CALENDAR.json"
)
OLD_FILE_SHA256 = "381f1879f97c1cd8d0103ab137d23076c0da3729ae19ec10e737461229965eda"
OLD_CALENDAR_SHA256 = "2e8eb6fbfcc0a410f63cbdfc6574beb4caa1c88ff95ffb305c5fd44d219a007e"
REGULAR_URL = "https://www.oanda.com/uk-en/trading/hours-of-operation/"
HOLIDAY_URL = "https://www.oanda.com/uk-en/trading/holiday-trading-hours/"
ARCHIVES = {
    "20250719220103": ("holiday", HOLIDAY_URL, "LAQ6PW5ROAJCSEAE3DJGJAIL7CJE2EBM"),
    "20251118074941": ("holiday", HOLIDAY_URL, "EKIJ3S4SNNYH6YG4XTFDC4GBJ4LY3CEB"),
    "20251222073046": ("holiday", HOLIDAY_URL, "STWMEWX3GT24VYGPOGDYE2ZSPZ5L3XCT"),
    "20260208124247": ("holiday", HOLIDAY_URL, "72NO6QGJ5WEYXXTQ43LWAP64YQOPA4B5"),
    "20250708202954": ("regular", REGULAR_URL, "UHCN3JEXZS74OZ4MPZG76CFJYWHYLAAW"),
    "20251115023204": ("regular", REGULAR_URL, "7BIVLQ6SFRXXE5UCYOTJRMSGNV4F2SUC"),
    "20251213234655": ("regular", REGULAR_URL, "47ZY2CZIZUTKCQGI3YNBTEFPGQYYL6TJ"),
}
METAL_DATES = (
    "2025-04-17", "2025-04-18", "2025-05-26", "2025-06-19", "2025-07-04",
    "2025-09-01", "2025-11-27", "2025-11-28", "2025-12-24", "2025-12-25",
    "2025-12-31",
)
SEEN_BY_CAPTURE = {
    "20250719220103": frozenset(METAL_DATES[:5]),
    "20251118074941": frozenset(METAL_DATES[:6]),
    "20251222073046": frozenset(METAL_DATES[:8]),
    "20260208124247": frozenset(METAL_DATES),
}
ACTION_FRAGMENTS = {
    "2025-04-17": ("us energies and metals", "close early at 17:00"),
    "2025-04-18": ("us energies and metals", "are closed"),
    "2025-05-26": ("us energies and metals", "close early at 2:30", "reopen at 6:00"),
    "2025-06-19": ("us energies and metals", "close early at 2:30", "reopen at 6:00"),
    "2025-07-04": ("us energies and metals", "close early at 1 pm"),
    "2025-09-01": ("us energies and metals", "close early at 2:30", "reopen at 6:00"),
    "2025-11-27": ("us energies and metals", "close early at 2:30", "reopen at 6:00"),
    "2025-11-28": ("us energies and metals", "close early at 2:45"),
    "2025-12-24": ("metals", "close at 13:44"),
    "2025-12-25": ("metals", "open at 18:01"),
    "2025-12-31": ("metals", "close at 16:59"),
}
MONTHS = {month.lower(): number for number, month in enumerate(
    ("January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"), 1
)}
_MONTH = "|".join(MONTHS)
_DATE_PATTERNS = (
    re.compile(rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<month>{_MONTH})\s+(?P<year>20\d{{2}})\b", re.I),
    re.compile(rf"\b(?P<month>{_MONTH})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?[,]?\s+(?P<year>20\d{{2}})\b", re.I),
)


def require(condition: bool, code: str) -> None:
    if not condition:
        raise RuntimeError("XAU_2025_CALENDAR_" + code)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _cdx_sha1(data: bytes) -> str:
    return base64.b32encode(hashlib.sha1(data).digest()).decode("ascii").rstrip("=")


def _regular_path(path: Path) -> None:
    require(path.is_absolute() and ".." not in path.parts, "PATH")
    require(not any(part.is_symlink() for part in (path, *path.parents)), "SYMLINK")
    require(stat.S_ISREG(path.stat().st_mode), "REGULAR_FILE")


def _read_ref(binding: dict, *, max_bytes: int = MAX_SOURCE_BYTES) -> bytes:
    require(isinstance(binding, dict) and set(binding) == {"path", "sha256"}, "REF_SHAPE")
    path = Path(binding["path"])
    _regular_path(path)
    before = path.stat()
    require(before.st_size <= max_bytes, "REF_SIZE")
    data = path.read_bytes()
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), "REF_CHANGED")
    require(_sha256(data) == binding["sha256"], "REF_SHA256")
    return data


def _ref(path: Path) -> dict[str, str]:
    _regular_path(path)
    return {"path": str(path), "sha256": _sha256(path.read_bytes())}


def _archive_url(timestamp: str, original_url: str) -> str:
    return f"https://web.archive.org/web/{timestamp}id_/{original_url}"


def _check_plan(plan: dict, plan_ref: dict, out_dir: Path) -> None:
    require(isinstance(plan, dict) and set(plan) == {
        "schema_version", "old_calendar", "archive_documents", "source_owners",
        "timezone_database", "output_root", "coverage_end_utc_exclusive", "policy_gate", "test_data_used",
    }, "PLAN_SHAPE")
    require(plan["schema_version"] == SCHEMA and re.fullmatch(POLICY_GATE_PREFIX + r"[0-9]{3}", plan["policy_gate"]) is not None
            and plan["test_data_used"] is False, "PLAN_SCOPE")
    require(plan["old_calendar"] == {"path": str(OLD_CALENDAR_PATH), "sha256": OLD_FILE_SHA256}, "OLD_REF")
    require(plan["coverage_end_utc_exclusive"] == "2026-01-01T00:00:00+00:00", "CUTOFF")
    require(plan["output_root"] == str(out_dir), "OUTPUT_BINDING")
    root = Path("/home/andre2/GX1_RUNS")
    require(out_dir.is_absolute() and root in out_dir.parents and ".." not in out_dir.parts, "OUTPUT_SCOPE")
    require(not any(part.is_symlink() for part in (out_dir, *out_dir.parents)), "OUTPUT_SYMLINK")
    require(out_dir.is_dir(), "OUTPUT_DIRECTORY")
    docs = plan["archive_documents"]
    require(isinstance(docs, list) and len(docs) == len(ARCHIVES), "ARCHIVE_COUNT")
    require([doc.get("capture_timestamp") for doc in docs] == list(ARCHIVES), "ARCHIVE_ORDER")
    for doc in docs:
        timestamp = doc["capture_timestamp"]
        role, original_url, digest = ARCHIVES[timestamp]
        require(set(doc) == {"role", "original_url", "archive_url", "capture_timestamp", "cdx_sha1_digest"}
                and doc == {"role": role, "original_url": original_url,
                            "archive_url": _archive_url(timestamp, original_url),
                            "capture_timestamp": timestamp, "cdx_sha1_digest": digest}, "ARCHIVE_BINDING")
    owners = plan["source_owners"]
    repo = Path(__file__).resolve().parents[2]
    expected_paths = {
        "worker": Path(__file__).resolve(),
        "closure_owner": repo / "gx1/contracts/unified_exit_market_closure_authority_v1.py",
        "publication_owner": repo / "gx1/contracts/immutable_event_authority_v1.py",
        "capped_owner": repo / "gx1/contracts/gx1_capped_execution_v1.py",
    }
    require(isinstance(owners, dict) and set(owners) == set(expected_paths), "OWNERS")
    for name, path in expected_paths.items():
        require(owners[name].get("path") == str(path), "OWNER_PATH")
        _read_ref(owners[name])
    require(plan["timezone_database"].get("path") == "/usr/share/zoneinfo/America/New_York", "TZ_PATH")
    _read_ref(plan["timezone_database"])
    require(plan_ref["sha256"] == _ref(Path(plan_ref["path"]))["sha256"], "PLAN_CHANGED")


def _check_claim(out_dir: Path, plan_ref: dict, worker_ref: dict) -> tuple[dict, dict]:
    claim_path = out_dir / "CLAIM.json"
    claim_ref = _ref(claim_path)
    claim = json.loads(_read_ref(claim_ref))
    require(isinstance(claim, dict) and set(claim) == {
        "schema_version", "plan", "worker", "policy", "source_head",
        "source_only_calendar", "test_data_used",
    } and claim["schema_version"] == CLAIM_SCHEMA and claim["plan"] == plan_ref
            and claim["worker"] == worker_ref and claim["source_only_calendar"] is True
            and claim["test_data_used"] is False, "CLAIM_BINDING")
    require({path.name for path in out_dir.iterdir()} == {"CLAIM.json"}, "OUTPUT_NOT_EMPTY")
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
    policy = json.loads(_read_ref(policy_ref))
    gate = policy[plan["policy_gate"]]
    expected = {"decision": "ADMITTED", "plan": plan_ref, "execution_authorized": True,
                "source_downloads_authorized": len(ARCHIVES), "target_data_reads_authorized": False,
                "target_builds_authorized": 0, "model_calls_authorized": 0, "test_authorized": False,
                "claimed_invocations": 0, "completed_invocations": 0, "max_invocations": 1,
                "training_authorized": False}
    require(policy["training_enabled"] is False and isinstance(gate, dict)
            and all(gate.get(key) == value for key, value in expected.items()), "POLICY_ADMISSION")
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry,
    )
    limits = require_capped_cpu_audit_execution()
    limits["lock"] = require_capped_lock_ancestry()
    require(limits["memory_max"] <= 4 * 1024**3 and limits["swap"] <= 512 * 1024**2
            and limits["pids"] <= 64 and limits["numerical_threads"] == 1, "CAPS")
    return limits


class _VisibleText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.lines: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in {"script", "style", "nav", "footer"}:
            self.skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav", "footer"} and self.skip:
            self.skip -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip:
            self.lines.extend(" ".join(line.split()) for line in data.splitlines() if line.strip())


def _visible_lines(raw: bytes) -> list[str]:
    if raw[:2] == b"\x1f\x8b":
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            raw = stream.read(MAX_SOURCE_BYTES + 1)
    require(len(raw) <= MAX_SOURCE_BYTES, "DECODED_SIZE")
    parser = _VisibleText()
    parser.feed(raw.decode("utf-8"))
    return parser.lines


def _check_regular(lines: list[str]) -> None:
    rows = [lines[i:i + 5] for i, line in enumerate(lines) if line == "XAU/USD"]
    require(rows == [["XAU/USD", "USD", "New York", "100", "Sun-Fri: 18:05 - 16:59"]], "XAU_RULE")


def _date_heading(line: str) -> str | None:
    for pattern in _DATE_PATTERNS:
        match = pattern.search(line)
        if match:
            return date(int(match.group("year")), MONTHS[match.group("month").lower()],
                        int(match.group("day"))).isoformat()
    return None


def _check_holiday(lines: list[str], capture: str) -> None:
    require(any("all times are in Eastern Time" in line for line in lines), "HOLIDAY_TIMEZONE")
    current: str | None = None
    found: dict[str, list[str]] = {}
    for index, line in enumerate(lines):
        heading = _date_heading(line)
        if heading or re.fullmatch(r"20\d{2}", line):
            current = heading
        if current and "2025-04-01" <= current < "2026-01-01" and re.search(r"\bmetals?\b", line, re.I):
            action = line + (" " + lines[index + 1] if line.endswith(":") and index + 1 < len(lines) else "")
            found.setdefault(current, []).append(action.lower())
    require(set(found) == SEEN_BY_CAPTURE[capture], "HOLIDAY_DATE_SET")
    for day, actions in found.items():
        require(len(actions) == 1 and all(fragment in actions[0] for fragment in ACTION_FRAGMENTS[day]),
                "HOLIDAY_METAL_ACTION")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise RuntimeError("XAU_2025_CALENDAR_ARCHIVE_REDIRECT")


def _fetch_archives(plan: dict) -> tuple[list[dict], dict[str, bytes]]:
    opener = build_opener(_NoRedirect)
    records: list[dict] = []
    payloads: dict[str, bytes] = {}
    for document in plan["archive_documents"]:
        url = document["archive_url"]
        with opener.open(Request(url, headers={"Accept-Encoding": "identity"}), timeout=30) as response:
            require(response.status == 200 and response.geturl() == url, "ARCHIVE_HTTP")
            require(response.headers.get_content_type() == "text/html", "ARCHIVE_TYPE")
            raw = response.read(MAX_SOURCE_BYTES + 1)
        require(len(raw) <= MAX_SOURCE_BYTES, "ARCHIVE_SIZE")
        require(_cdx_sha1(raw) == document["cdx_sha1_digest"], "ARCHIVE_CDX_SHA1")
        capture = document["capture_timestamp"]
        lines = _visible_lines(raw)
        if document["role"] == "regular":
            _check_regular(lines)
        else:
            _check_holiday(lines, capture)
        suffix = ".html.gz" if raw[:2] == b"\x1f\x8b" else ".html"
        name = f"oanda_{document['role']}_hours_{capture}{suffix}"
        records.append({"path": str(Path(plan["output_root"]) / name), "sha256": _sha256(raw), **document})
        payloads[name] = raw
    return records, payloads


def _build_calendar(old: dict, plan: dict, records: list[dict], owner) -> dict:
    with Path(plan["timezone_database"]["path"]).open("rb") as stream:
        ny = ZoneInfo.from_file(stream, key="America/New_York")
    veto = set(old["holiday_veto_dates"]) | set(METAL_DATES)
    intervals = list(old["intervals"])
    excluded = list(old["excluded_published_regular_intervals_due_to_holiday"])
    regular_sha = owner.canonical_sha256([doc["sha256"] for doc in records if doc["role"] == "regular"])
    current = date(2025, 4, 2)
    while current <= date(2025, 12, 31):
        weekday = current.weekday()
        if weekday < 5:
            finish_date = current + timedelta(days=2 if weekday == 4 else 0)
            left = datetime.combine(current, time(16, 59), ny).astimezone(timezone.utc).isoformat()
            right = datetime.combine(finish_date, time(18, 5), ny).astimezone(timezone.utc).isoformat()
            require(right <= plan["coverage_end_utc_exclusive"], "PERIOD_CUTOFF")
            item = {"kind": "weekend" if weekday == 4 else "daily_maintenance",
                    "start_utc": left, "end_utc_exclusive": right}
            touched = sorted(day.isoformat() for day in (current + timedelta(days=n)
                             for n in range((finish_date - current).days + 1)) if day.isoformat() in veto)
            if touched:
                excluded.append({"start_utc": left, "end_utc_exclusive": right, "veto_dates": touched})
            else:
                item["source_event_id"] = owner.canonical_sha256({**item, "regular_source_sha256": regular_sha})
                intervals.append(item)
        current += timedelta(days=1)
    require(intervals[:len(old["intervals"])] == old["intervals"] and
            all(intervals[i]["end_utc_exclusive"] <= intervals[i + 1]["start_utc"]
                for i in range(len(intervals) - 1)), "INTERVAL_ORDER")
    calendar = {**old,
        "coverage_end_utc_exclusive": plan["coverage_end_utc_exclusive"],
        "source_documents": [*old["source_documents"], *records],
        "holiday_veto_dates": sorted(veto), "intervals": intervals,
        "excluded_published_regular_intervals_due_to_holiday": excluded,
        "historical_rule_evidence": "The verified original six OANDA regular-hour archives end 2025-04-19; the bound 2025-07-08, 2025-11-15 and 2025-12-13 archives repeat XAU/USD New York Sun-Fri 18:05-16:59. Point-in-time captures do not prove uninterrupted validity between captures.",
        "holiday_evidence": "Bound OANDA holiday archives from 2025-07-19, 2025-11-18, 2025-12-22 and retrospective 2026-02-08 list eleven 2025-04-17..2025-12-31 US-metal exceptions. Every touching regular period is vetoed; no holiday endpoint is inferred.",
    }
    calendar.pop("calendar_sha256")
    calendar["calendar_sha256"] = owner.canonical_sha256(calendar)
    return calendar


def run(plan_path: Path, plan_sha256: str, out_dir: Path) -> dict:
    plan_ref = {"path": str(plan_path), "sha256": plan_sha256}
    plan = json.loads(_read_ref(plan_ref))
    _check_plan(plan, plan_ref, out_dir)
    claim, claim_ref = _check_claim(out_dir, plan_ref, plan["source_owners"]["worker"])
    limits = _check_admission(plan, plan_ref, claim)
    old = json.loads(_read_ref(plan["old_calendar"]))
    from gx1.contracts import unified_exit_market_closure_authority_v1 as owner
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    require(old["schema_version"] == "gx1_dynamic_mtf_state_published_calendar_v1"
            and old["calendar_sha256"] == OLD_CALENDAR_SHA256
            and owner.canonical_sha256({k: v for k, v in old.items() if k != "calendar_sha256"}) == OLD_CALENDAR_SHA256
            and old["coverage_end_utc_exclusive"] == "2025-04-02T00:00:00+00:00"
            and old["timezone_database_file_sha256"] == plan["timezone_database"]["sha256"]
            and old["regular_rule"] == {"session_open_local": "18:05", "session_close_local": "16:59",
                                        "weekly_sessions": "Sunday_to_Friday"}, "OLD_CALENDAR")
    for document in old["source_documents"]:
        raw = _read_ref({key: document[key] for key in ("path", "sha256")})
        require(_cdx_sha1(raw) == document["cdx_sha1_digest"], "OLD_SOURCE_CDX")
    records, payloads = _fetch_archives(plan)
    calendar = _build_calendar(old, plan, records, owner)
    _check_plan(plan, plan_ref, out_dir)
    require(_check_claim(out_dir, plan_ref, plan["source_owners"]["worker"]) == (claim, claim_ref),
            "CLAIM_CHANGED")
    require(_check_admission(plan, plan_ref, claim) == limits, "CAPS_CHANGED")
    _read_ref(plan["old_calendar"])
    for document in old["source_documents"]:
        raw = _read_ref({key: document[key] for key in ("path", "sha256")})
        require(_cdx_sha1(raw) == document["cdx_sha1_digest"], "OLD_SOURCE_CHANGED")
    for name, raw in payloads.items():
        path = out_dir / name
        _publish_bytes(path, raw)
        path.chmod(0o444)
    calendar_path = out_dir / "PUBLISHED_CALENDAR.json"
    encoded = (json.dumps(calendar, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _publish_bytes(calendar_path, encoded)
    calendar_path.chmod(0o444)
    result = {
        "schema_version": "gx1_xau_2025_published_calendar_extension_result_v1",
        "decision": "PASS_SOURCE_CALENDAR_ONLY", "plan": plan_ref, "claim": claim_ref,
        "old_calendar": plan["old_calendar"], "source_owners": plan["source_owners"],
        "policy": claim["policy"], "source_head": claim["source_head"], "limits": limits,
        "archive_documents": records, "published_calendar": _ref(calendar_path),
        "coverage_start_utc": calendar["coverage_start_utc"],
        "coverage_end_utc_exclusive": calendar["coverage_end_utc_exclusive"],
        "old_interval_count": len(old["intervals"]), "new_interval_count": len(calendar["intervals"]) - len(old["intervals"]),
        "added_veto_dates": list(METAL_DATES), "test_data_used": False,
        "target_data_read": False, "model_calls": 0,
        "limit": "Public calendar sources prove sampled published rules and listed exceptions; continuous validity and unlisted interruptions remain unproved. Only full regular-closure containment may later certify observed M1 gaps.",
    }
    result_path = out_dir / "RESULT.json"
    _publish_bytes(result_path, (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
    result_path.chmod(0o444)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.plan, args.plan_sha256, args.out_dir)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
