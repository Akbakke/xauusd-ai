"""Immutable PRETEST market-closure authority for XAU_USD M1 lifecycles.

The authority classifies each observed discontinuity in an immutable quote
clock against an exact UTC schedule artifact.  Only an exact scheduled
weekend or holiday interval may carry a trade state across the source gap.
Every other discontinuity remains unknown and must right-censor the lifecycle.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MARKET_CLOSURE_SCHEDULE_SCHEMA_VERSION = "gx1_xau_exact_market_closure_schedule_v1"
MARKET_CLOSURE_AUTHORITY_SCHEMA_VERSION = "gx1_unified_exit_market_closure_authority_v1"
MARKET_CLOSURE_INTERVAL_SCHEMA_VERSION = "gx1_unified_exit_market_closure_interval_v1"
KNOWN_CLOSURE_KINDS = ("daily_maintenance", "weekend", "holiday")
PROJECT_CLOSURE_POLICY_SCHEMA_VERSION = (
    "gx1_project_inferred_pretest_closure_policy_v1"
)
_SCHEDULE_SOURCE_METHODS = frozenset(
    {
        "externally_sourced_exact_xau_utc_closure_intervals_v1",
        "project_inferred_pretest_closure_policy_v1",
    }
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def m1_clock_sha256(times: Sequence[Any]) -> str:
    clock = _require_clock(times)
    return hashlib.sha256(np.asarray(clock.asi8, dtype="<i8").tobytes()).hexdigest()


def _require_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"UNIFIED_EXIT_MARKET_CLOSURE_{label}_SHA_INVALID")
    return value


def _require_utc(value: Any, label: str) -> pd.Timestamp:
    observed = pd.Timestamp(value)
    if pd.isna(observed) or observed.tz is None or observed.utcoffset() != pd.Timedelta(0):
        raise RuntimeError(f"UNIFIED_EXIT_MARKET_CLOSURE_{label}_TIME_INVALID")
    return observed.as_unit("ns")


def _require_clock(values: Sequence[Any]) -> pd.DatetimeIndex:
    clock = pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors="coerce")).as_unit(
        "ns"
    )
    if (
        clock.empty
        or clock.hasnans
        or not clock.is_unique
        or not clock.is_monotonic_increasing
        or not clock.floor("60s").equals(clock)
    ):
        raise RuntimeError("UNIFIED_EXIT_MARKET_CLOSURE_M1_CLOCK_INVALID")
    return clock


def seal_exact_market_schedule(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "decision",
        "instrument",
        "timeframe",
        "coverage_start_utc",
        "coverage_end_utc_exclusive",
        "interval_semantics",
        "source_method",
        "source_reference_sha256",
        "intervals",
        "test_data_used",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise RuntimeError("UNIFIED_EXIT_MARKET_CLOSURE_SCHEDULE_INVALID")
    observed = dict(value)
    start = _require_utc(observed["coverage_start_utc"], "SCHEDULE_START")
    end = _require_utc(observed["coverage_end_utc_exclusive"], "SCHEDULE_END")
    if (
        observed["schema_version"] != MARKET_CLOSURE_SCHEDULE_SCHEMA_VERSION
        or observed["decision"] != "PASS"
        or observed["instrument"] != "XAU_USD"
        or observed["timeframe"] != "M1"
        or observed["interval_semantics"] != "left_closed_right_open_utc"
        or observed["source_method"] not in _SCHEDULE_SOURCE_METHODS
        or observed["test_data_used"] is not False
        or end <= start
    ):
        raise RuntimeError("UNIFIED_EXIT_MARKET_CLOSURE_SCHEDULE_INVALID")
    _require_sha(observed["source_reference_sha256"], "SCHEDULE_SOURCE")
    raw_intervals = observed["intervals"]
    if not isinstance(raw_intervals, list):
        raise RuntimeError("UNIFIED_EXIT_MARKET_CLOSURE_SCHEDULE_INVALID")
    intervals: list[dict[str, Any]] = []
    previous_end: pd.Timestamp | None = None
    for item in raw_intervals:
        if not isinstance(item, Mapping) or set(item) != {
            "kind",
            "start_utc",
            "end_utc_exclusive",
            "source_event_id",
        }:
            raise RuntimeError("UNIFIED_EXIT_MARKET_CLOSURE_SCHEDULE_INTERVAL_INVALID")
        interval = dict(item)
        left = _require_utc(interval["start_utc"], "SCHEDULE_INTERVAL_START")
        right = _require_utc(interval["end_utc_exclusive"], "SCHEDULE_INTERVAL_END")
        if (
            interval["kind"] not in KNOWN_CLOSURE_KINDS
            or not isinstance(interval["source_event_id"], str)
            or not interval["source_event_id"]
            or left < start
            or right > end
            or right <= left
            or (previous_end is not None and left < previous_end)
        ):
            raise RuntimeError("UNIFIED_EXIT_MARKET_CLOSURE_SCHEDULE_INTERVAL_INVALID")
        previous_end = right
        intervals.append(
            {
                **interval,
                "start_utc": left.isoformat(),
                "end_utc_exclusive": right.isoformat(),
            }
        )
    sealed = {
        **observed,
        "coverage_start_utc": start.isoformat(),
        "coverage_end_utc_exclusive": end.isoformat(),
        "intervals": intervals,
    }
    sealed["schedule_sha256"] = canonical_sha256(sealed)
    return sealed


def _observed_gap_records(clock: pd.DatetimeIndex) -> list[dict[str, Any]]:
    clock_ns = np.asarray(clock.asi8, dtype=np.int64)
    rows = np.flatnonzero(np.diff(clock_ns) != 60_000_000_000)
    records: list[dict[str, Any]] = []
    for raw_row in rows.tolist():
        row = int(raw_row)
        previous = clock[row]
        next_bar = clock[row + 1]
        closure_start = previous + pd.Timedelta(minutes=1)
        closure_seconds = int((next_bar - closure_start).total_seconds())
        signature = {
            "previous_weekday_utc": int(previous.weekday()),
            "previous_time_utc": previous.strftime("%H:%M"),
            "next_weekday_utc": int(next_bar.weekday()),
            "next_time_utc": next_bar.strftime("%H:%M"),
            "closure_seconds": closure_seconds,
        }
        record = {
            "gap_after_m1_row": row,
            "previous_bar_start_utc": previous.isoformat(),
            "next_bar_start_utc": next_bar.isoformat(),
            "closure_start_utc": closure_start.isoformat(),
            "closure_end_utc_exclusive": next_bar.isoformat(),
            "signature": signature,
        }
        record["evidence_sha256"] = canonical_sha256(record)
        records.append(record)
    return records


def _project_inferred_kind(signature: Mapping[str, Any]) -> str | None:
    seconds = signature["closure_seconds"]
    previous_weekday = signature["previous_weekday_utc"]
    next_weekday = signature["next_weekday_utc"]
    if (
        3_600 <= seconds <= 3 * 3_600
        and previous_weekday in {0, 1, 2, 3}
        and next_weekday in {previous_weekday, (previous_weekday + 1) % 7}
    ):
        return "daily_maintenance"
    if (
        24 * 3_600 <= seconds <= 96 * 3_600
        and previous_weekday == 4
        and next_weekday in {6, 0}
    ):
        return "weekend"
    return None


def build_project_inferred_closure_policy(
    *,
    train_m1_times: Sequence[Any],
    train_m1_source_sha256: str,
    minimum_daily_support: int,
    minimum_weekend_support: int,
) -> dict[str, Any]:
    """Fit recurring closure signatures from TRAIN clock structure only."""

    clock = _require_clock(train_m1_times)
    source_sha = _require_sha(train_m1_source_sha256, "TRAIN_M1_SOURCE")
    if (
        isinstance(minimum_daily_support, bool)
        or not isinstance(minimum_daily_support, int)
        or minimum_daily_support < 2
        or isinstance(minimum_weekend_support, bool)
        or not isinstance(minimum_weekend_support, int)
        or minimum_weekend_support < 2
    ):
        raise RuntimeError("UNIFIED_EXIT_MARKET_CLOSURE_SUPPORT_INVALID")
    records = _observed_gap_records(clock)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        kind = _project_inferred_kind(record["signature"])
        if kind is None:
            continue
        key = (kind, canonical_sha256(record["signature"]))
        grouped.setdefault(key, []).append(record)
    rules: list[dict[str, Any]] = []
    for (kind, signature_sha), support in sorted(grouped.items()):
        minimum = (
            minimum_daily_support
            if kind == "daily_maintenance"
            else minimum_weekend_support
        )
        if len(support) < minimum:
            continue
        rule = {
            "closure_kind": kind,
            "signature": support[0]["signature"],
            "signature_sha256": signature_sha,
            "support_count": len(support),
            "minimum_support_count": minimum,
            "train_evidence_sha256": canonical_sha256(
                [item["evidence_sha256"] for item in support]
            ),
        }
        rule["rule_sha256"] = canonical_sha256(rule)
        rules.append(rule)
    policy = {
        "schema_version": PROJECT_CLOSURE_POLICY_SCHEMA_VERSION,
        "decision": "PASS" if rules else "BLOCK",
        "fit_splits": ["train"],
        "application_splits": ["train", "val"],
        "validation_or_test_used_for_fit": False,
        "train_m1_source_sha256": source_sha,
        "train_m1_clock_sha256": m1_clock_sha256(clock),
        "train_m1_row_count": len(clock),
        "minimum_daily_support": minimum_daily_support,
        "minimum_weekend_support": minimum_weekend_support,
        "rule_semantics": "exact_utc_weekday_time_and_closure_duration_signature",
        "false_positive_guards": {
            "minimum_allowed_closure_seconds": 3_600,
            "daily_maximum_closure_seconds": 10_800,
            "weekend_minimum_closure_seconds": 86_400,
            "weekend_maximum_closure_seconds": 345_600,
            "under_one_hour_is_unknown": True,
            "holiday_or_irregular_gap_is_unknown": True,
            "unseen_signature_is_unknown": True,
        },
        "rules": rules,
        "observed_train_gap_count": len(records),
        "qualified_train_gap_count": sum(
            len(grouped[(rule["closure_kind"], rule["signature_sha256"])])
            for rule in rules
        ),
        "test_data_used": False,
    }
    policy["policy_sha256"] = canonical_sha256(policy)
    return policy


def require_project_inferred_closure_policy(
    value: Mapping[str, Any], *, expected_train_m1_source_sha256: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or "policy_sha256" not in value:
        raise RuntimeError("UNIFIED_EXIT_MARKET_CLOSURE_PROJECT_POLICY_INVALID")
    observed = dict(value)
    claimed = observed.pop("policy_sha256")
    required = {
        "schema_version",
        "decision",
        "fit_splits",
        "application_splits",
        "validation_or_test_used_for_fit",
        "train_m1_source_sha256",
        "train_m1_clock_sha256",
        "train_m1_row_count",
        "minimum_daily_support",
        "minimum_weekend_support",
        "rule_semantics",
        "false_positive_guards",
        "rules",
        "observed_train_gap_count",
        "qualified_train_gap_count",
        "test_data_used",
    }
    if (
        set(observed) != required
        or claimed != canonical_sha256(observed)
        or observed["schema_version"] != PROJECT_CLOSURE_POLICY_SCHEMA_VERSION
        or observed["decision"] != "PASS"
        or observed["fit_splits"] != ["train"]
        or observed["application_splits"] != ["train", "val"]
        or observed["validation_or_test_used_for_fit"] is not False
        or observed["test_data_used"] is not False
        or observed["train_m1_source_sha256"]
        != _require_sha(expected_train_m1_source_sha256, "EXPECTED_TRAIN_M1_SOURCE")
        or not isinstance(observed["rules"], list)
        or not observed["rules"]
    ):
        raise RuntimeError("UNIFIED_EXIT_MARKET_CLOSURE_PROJECT_POLICY_INVALID")
    seen: set[tuple[str, str]] = set()
    for rule in observed["rules"]:
        if not isinstance(rule, Mapping) or set(rule) != {
            "closure_kind",
            "signature",
            "signature_sha256",
            "support_count",
            "minimum_support_count",
            "train_evidence_sha256",
            "rule_sha256",
        }:
            raise RuntimeError("UNIFIED_EXIT_MARKET_CLOSURE_PROJECT_RULE_INVALID")
        raw = dict(rule)
        rule_sha = raw.pop("rule_sha256")
        identity = (str(rule["closure_kind"]), str(rule["signature_sha256"]))
        if (
            identity in seen
            or rule["closure_kind"] not in {"daily_maintenance", "weekend"}
            or rule["signature_sha256"] != canonical_sha256(rule["signature"])
            or _project_inferred_kind(rule["signature"]) != rule["closure_kind"]
            or rule["support_count"] < rule["minimum_support_count"]
            or rule_sha != canonical_sha256(raw)
        ):
            raise RuntimeError("UNIFIED_EXIT_MARKET_CLOSURE_PROJECT_RULE_INVALID")
        seen.add(identity)
    observed["policy_sha256"] = claimed
    return observed


def exact_schedule_from_project_policy(
    *,
    policy: Mapping[str, Any],
    expected_train_m1_source_sha256: str,
    target_split: str,
    target_m1_times: Sequence[Any],
) -> dict[str, Any]:
    """Apply frozen TRAIN signatures to TRAIN/VAL; unknown gaps stay absent."""

    checked = require_project_inferred_closure_policy(
        policy, expected_train_m1_source_sha256=expected_train_m1_source_sha256
    )
    if target_split not in checked["application_splits"]:
        raise RuntimeError("UNIFIED_EXIT_MARKET_CLOSURE_PROJECT_SPLIT_INVALID")
    clock = _require_clock(target_m1_times)
    rules = {
        (rule["closure_kind"], rule["signature_sha256"]): rule
        for rule in checked["rules"]
    }
    intervals: list[dict[str, Any]] = []
    for record in _observed_gap_records(clock):
        kind = _project_inferred_kind(record["signature"])
        rule = rules.get((str(kind), canonical_sha256(record["signature"])))
        if rule is None:
            continue
        intervals.append(
            {
                "kind": kind,
                "start_utc": record["closure_start_utc"],
                "end_utc_exclusive": record["closure_end_utc_exclusive"],
                "source_event_id": canonical_sha256(
                    {
                        "policy_sha256": checked["policy_sha256"],
                        "target_split": target_split,
                        "gap_evidence_sha256": record["evidence_sha256"],
                        "rule_sha256": rule["rule_sha256"],
                    }
                ),
            }
        )
    return seal_exact_market_schedule(
        {
            "schema_version": MARKET_CLOSURE_SCHEDULE_SCHEMA_VERSION,
            "decision": "PASS",
            "instrument": "XAU_USD",
            "timeframe": "M1",
            "coverage_start_utc": clock[0].isoformat(),
            "coverage_end_utc_exclusive": (
                clock[-1] + pd.Timedelta(minutes=1)
            ).isoformat(),
            "interval_semantics": "left_closed_right_open_utc",
            "source_method": "project_inferred_pretest_closure_policy_v1",
            "source_reference_sha256": checked["policy_sha256"],
            "intervals": intervals,
            "test_data_used": False,
        }
    )


def require_exact_market_schedule(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or "schedule_sha256" not in value:
        raise RuntimeError("UNIFIED_EXIT_MARKET_CLOSURE_SCHEDULE_INVALID")
    raw = dict(value)
    claimed = raw.pop("schedule_sha256")
    sealed = seal_exact_market_schedule(raw)
    if claimed != sealed["schedule_sha256"]:
        raise RuntimeError("UNIFIED_EXIT_MARKET_CLOSURE_SCHEDULE_HASH_INVALID")
    return sealed


def build_market_closure_authority(
    *,
    m1_times: Sequence[Any],
    m1_source_path: Path,
    m1_source_sha256: str,
    m1_source_manifest_path: Path,
    m1_source_manifest_sha256: str,
    exact_schedule: Mapping[str, Any],
    exact_schedule_path: Path,
    exact_schedule_file_sha256: str,
) -> dict[str, Any]:
    """Classify every observed clock gap against the exact schedule."""

    clock = _require_clock(m1_times)
    source_sha = _require_sha(m1_source_sha256, "M1_SOURCE")
    source_manifest_sha = _require_sha(
        m1_source_manifest_sha256, "M1_SOURCE_MANIFEST"
    )
    schedule_file_sha = _require_sha(exact_schedule_file_sha256, "SCHEDULE_FILE")
    schedule = require_exact_market_schedule(exact_schedule)
    coverage_start = _require_utc(schedule["coverage_start_utc"], "COVERAGE_START")
    coverage_end = _require_utc(schedule["coverage_end_utc_exclusive"], "COVERAGE_END")
    if clock[0] < coverage_start or clock[-1] + pd.Timedelta(minutes=1) > coverage_end:
        raise RuntimeError("UNIFIED_EXIT_MARKET_CLOSURE_SCHEDULE_COVERAGE_INVALID")
    schedule_by_interval = {
        (
            _require_utc(item["start_utc"], "SCHEDULE_INTERVAL_START").value,
            _require_utc(item["end_utc_exclusive"], "SCHEDULE_INTERVAL_END").value,
        ): item
        for item in schedule["intervals"]
    }
    delta_ns = int(pd.Timedelta(minutes=1).value)
    clock_ns = np.asarray(clock.asi8, dtype=np.int64)
    gap_after_rows = np.flatnonzero(np.diff(clock_ns) != delta_ns)
    records: list[dict[str, Any]] = []
    matched: set[tuple[int, int]] = set()
    for raw_row in gap_after_rows.tolist():
        row = int(raw_row)
        interval_start_ns = int(clock_ns[row] + delta_ns)
        interval_end_ns = int(clock_ns[row + 1])
        schedule_item = schedule_by_interval.get((interval_start_ns, interval_end_ns))
        if schedule_item is None:
            classification = "unknown_source_absence"
            kind = "unknown"
            source_event_id = None
            successor_allowed = False
        else:
            matched.add((interval_start_ns, interval_end_ns))
            kind = str(schedule_item["kind"])
            classification = f"declared_{kind}_market_closure"
            source_event_id = str(schedule_item["source_event_id"])
            successor_allowed = True
        record = {
            "schema_version": MARKET_CLOSURE_INTERVAL_SCHEMA_VERSION,
            "gap_after_m1_row": row,
            "previous_bar_start_utc": clock[row].isoformat(),
            "next_bar_start_utc": clock[row + 1].isoformat(),
            "closure_start_utc": pd.Timestamp(interval_start_ns, tz="UTC").isoformat(),
            "closure_end_utc_exclusive": pd.Timestamp(
                interval_end_ns, tz="UTC"
            ).isoformat(),
            "closure_seconds": int((interval_end_ns - interval_start_ns) // 1_000_000_000),
            "classification": classification,
            "closure_kind": kind,
            "source_event_id": source_event_id,
            "successor_across_gap_allowed": successor_allowed,
            "schedule_sha256": schedule["schedule_sha256"],
        }
        record["interval_sha256"] = canonical_sha256(record)
        records.append(record)
    if matched != set(schedule_by_interval):
        raise RuntimeError("UNIFIED_EXIT_MARKET_CLOSURE_SCHEDULE_INTERVAL_UNOBSERVED")
    authority = {
        "schema_version": MARKET_CLOSURE_AUTHORITY_SCHEMA_VERSION,
        "decision": "PASS",
        "instrument": "XAU_USD",
        "timeframe": "M1",
        "timestamp_semantics": "bar_start_utc",
        "m1_source_path": str(m1_source_path),
        "m1_source_sha256": source_sha,
        "m1_source_manifest_path": str(m1_source_manifest_path),
        "m1_source_manifest_sha256": source_manifest_sha,
        "m1_clock_sha256": m1_clock_sha256(clock),
        "m1_row_count": len(clock),
        "schedule_path": str(exact_schedule_path),
        "schedule_file_sha256": schedule_file_sha,
        "schedule_sha256": schedule["schedule_sha256"],
        "coverage_start_utc": coverage_start.isoformat(),
        "coverage_end_utc_exclusive": coverage_end.isoformat(),
        "observed_gap_count": len(records),
        "known_market_closure_count": sum(
            bool(item["successor_across_gap_allowed"]) for item in records
        ),
        "unknown_source_gap_count": sum(
            not bool(item["successor_across_gap_allowed"]) for item in records
        ),
        "intervals": records,
        "interval_stream_sha256": canonical_sha256(
            [item["interval_sha256"] for item in records]
        ),
        "known_closure_semantics": "successor_allowed_with_wall_clock_economics",
        "unknown_gap_semantics": "right_censor_before_gap",
        "test_data_used": False,
    }
    authority["artifact_sha256"] = canonical_sha256(authority)
    return require_market_closure_authority(
        authority,
        expected_m1_source_sha256=source_sha,
        expected_m1_clock_sha256=m1_clock_sha256(clock),
    )


def require_market_closure_authority(
    value: Mapping[str, Any],
    *,
    expected_m1_source_sha256: str,
    expected_m1_clock_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("UNIFIED_EXIT_MARKET_CLOSURE_AUTHORITY_INVALID")
    observed = dict(value)
    required = {
        "schema_version",
        "decision",
        "instrument",
        "timeframe",
        "timestamp_semantics",
        "m1_source_path",
        "m1_source_sha256",
        "m1_source_manifest_path",
        "m1_source_manifest_sha256",
        "m1_clock_sha256",
        "m1_row_count",
        "schedule_path",
        "schedule_file_sha256",
        "schedule_sha256",
        "coverage_start_utc",
        "coverage_end_utc_exclusive",
        "observed_gap_count",
        "known_market_closure_count",
        "unknown_source_gap_count",
        "intervals",
        "interval_stream_sha256",
        "known_closure_semantics",
        "unknown_gap_semantics",
        "test_data_used",
        "artifact_sha256",
    }
    if set(observed) != required:
        raise RuntimeError("UNIFIED_EXIT_MARKET_CLOSURE_AUTHORITY_INVALID")
    if (
        observed["schema_version"] != MARKET_CLOSURE_AUTHORITY_SCHEMA_VERSION
        or observed["decision"] != "PASS"
        or observed["instrument"] != "XAU_USD"
        or observed["timeframe"] != "M1"
        or observed["timestamp_semantics"] != "bar_start_utc"
        or observed["m1_source_sha256"]
        != _require_sha(expected_m1_source_sha256, "EXPECTED_M1_SOURCE")
        or observed["m1_clock_sha256"]
        != _require_sha(expected_m1_clock_sha256, "EXPECTED_M1_CLOCK")
        or observed["known_closure_semantics"]
        != "successor_allowed_with_wall_clock_economics"
        or observed["unknown_gap_semantics"] != "right_censor_before_gap"
        or observed["test_data_used"] is not False
        or isinstance(observed["m1_row_count"], bool)
        or not isinstance(observed["m1_row_count"], int)
        or observed["m1_row_count"] < 1
    ):
        raise RuntimeError("UNIFIED_EXIT_MARKET_CLOSURE_AUTHORITY_INVALID")
    for key in (
        "m1_source_manifest_sha256",
        "schedule_file_sha256",
        "schedule_sha256",
        "interval_stream_sha256",
        "artifact_sha256",
    ):
        _require_sha(observed[key], key.upper())
    intervals = observed["intervals"]
    if not isinstance(intervals, list) or observed["observed_gap_count"] != len(intervals):
        raise RuntimeError("UNIFIED_EXIT_MARKET_CLOSURE_AUTHORITY_INTERVAL_INVALID")
    known = 0
    unknown = 0
    prior_row = -1
    hashes: list[str] = []
    for item in intervals:
        if not isinstance(item, Mapping) or set(item) != {
            "schema_version",
            "gap_after_m1_row",
            "previous_bar_start_utc",
            "next_bar_start_utc",
            "closure_start_utc",
            "closure_end_utc_exclusive",
            "closure_seconds",
            "classification",
            "closure_kind",
            "source_event_id",
            "successor_across_gap_allowed",
            "schedule_sha256",
            "interval_sha256",
        }:
            raise RuntimeError("UNIFIED_EXIT_MARKET_CLOSURE_AUTHORITY_INTERVAL_INVALID")
        record = dict(item)
        claimed = record.pop("interval_sha256")
        row = record["gap_after_m1_row"]
        allowed = record["successor_across_gap_allowed"]
        previous = _require_utc(record["previous_bar_start_utc"], "INTERVAL_PREVIOUS")
        next_bar = _require_utc(record["next_bar_start_utc"], "INTERVAL_NEXT")
        closure_start = _require_utc(record["closure_start_utc"], "INTERVAL_START")
        closure_end = _require_utc(record["closure_end_utc_exclusive"], "INTERVAL_END")
        if (
            isinstance(row, bool)
            or not isinstance(row, int)
            or row <= prior_row
            or record["schema_version"] != MARKET_CLOSURE_INTERVAL_SCHEMA_VERSION
            or record["schedule_sha256"] != observed["schedule_sha256"]
            or type(allowed) is not bool
            or claimed != canonical_sha256(record)
            or row + 1 >= observed["m1_row_count"]
            or closure_start != previous + pd.Timedelta(minutes=1)
            or closure_end != next_bar
            or closure_end <= closure_start
            or record["closure_seconds"]
            != int((closure_end - closure_start).total_seconds())
        ):
            raise RuntimeError("UNIFIED_EXIT_MARKET_CLOSURE_AUTHORITY_INTERVAL_INVALID")
        if allowed:
            known += 1
            if (
                record["closure_kind"] not in KNOWN_CLOSURE_KINDS
                or record["classification"]
                != f"declared_{record['closure_kind']}_market_closure"
                or not isinstance(record["source_event_id"], str)
                or not record["source_event_id"]
            ):
                raise RuntimeError("UNIFIED_EXIT_MARKET_CLOSURE_AUTHORITY_INTERVAL_INVALID")
        else:
            unknown += 1
            if (
                record["closure_kind"] != "unknown"
                or record["classification"] != "unknown_source_absence"
                or record["source_event_id"] is not None
            ):
                raise RuntimeError("UNIFIED_EXIT_MARKET_CLOSURE_AUTHORITY_INTERVAL_INVALID")
        prior_row = row
        hashes.append(claimed)
    if (
        observed["known_market_closure_count"] != known
        or observed["unknown_source_gap_count"] != unknown
        or observed["interval_stream_sha256"] != canonical_sha256(hashes)
        or observed["artifact_sha256"]
        != canonical_sha256(
            {key: item for key, item in observed.items() if key != "artifact_sha256"}
        )
    ):
        raise RuntimeError("UNIFIED_EXIT_MARKET_CLOSURE_AUTHORITY_HASH_INVALID")
    return observed


def closure_intervals_by_gap_after_row(
    authority: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    return {
        int(item["gap_after_m1_row"]): dict(item)
        for item in authority["intervals"]
    }


__all__ = (
    "KNOWN_CLOSURE_KINDS",
    "MARKET_CLOSURE_AUTHORITY_SCHEMA_VERSION",
    "MARKET_CLOSURE_INTERVAL_SCHEMA_VERSION",
    "MARKET_CLOSURE_SCHEDULE_SCHEMA_VERSION",
    "PROJECT_CLOSURE_POLICY_SCHEMA_VERSION",
    "build_market_closure_authority",
    "build_project_inferred_closure_policy",
    "canonical_sha256",
    "closure_intervals_by_gap_after_row",
    "exact_schedule_from_project_policy",
    "file_sha256",
    "m1_clock_sha256",
    "require_exact_market_schedule",
    "require_market_closure_authority",
    "require_project_inferred_closure_policy",
    "seal_exact_market_schedule",
)
