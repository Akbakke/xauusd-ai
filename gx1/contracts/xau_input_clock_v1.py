"""Input-only physical clock evidence, separate from historical bar-close clocks.

Only capture_clock_sample reads the local clock and /proc. Assessment is pure:
the caller must verify referenced files and external UTC evidence. A response
callback is an upper bound on receipt, not wire time or market event time.
No UTC accuracy bound is inferred from timezone settings or clock samples.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import time
import uuid

RECEIPT_SCHEMA = "gx1_xau_input_clock_receipt_v1"
TIMEFRAMES = ("M5", "M15", "H1", "H4", "D1")
_SAMPLE_KEYS = frozenset(("utc_ns", "monotonic_before_ns", "monotonic_after_ns",
                          "boot_id", "process_id", "process_start_ticks"))
_SHA = re.compile(r"[0-9a-f]{64}")


class InputClockError(ValueError):
    """Malformed, contradictory or differently bound physical timing evidence."""


def _require(ok, reason):
    if not ok:
        raise InputClockError("XAU_INPUT_CLOCK_" + reason)


def _keys(value, keys, label):
    _require(isinstance(value, Mapping) and set(value) == set(keys), label + "_FIELDS")


def _integer(value, label, *, minimum=0):
    _require(type(value) is int and value >= minimum, label + "_INTEGER")
    return value


def _sha(value, label):
    _require(isinstance(value, str) and _SHA.fullmatch(value) is not None, label + "_SHA256")
    return value


def _ref(value, label):
    _keys(value, ("path", "sha256"), label)
    path = value["path"]
    _require(isinstance(path, str) and PurePosixPath(path).is_absolute()
             and ".." not in PurePosixPath(path).parts
             and not any(char in path for char in ("\0", "\n", "\r")), label + "_PATH")
    return {"path": path, "sha256": _sha(value["sha256"], label)}


def _sample(value, label):
    _require(isinstance(value, Mapping) and _SAMPLE_KEYS <= set(value)
             and set(value) <= _SAMPLE_KEYS | {"utc_interval"}, label + "_FIELDS")
    for key in ("utc_ns", "monotonic_before_ns", "monotonic_after_ns", "process_start_ticks"):
        _integer(value[key], label + "_" + key.upper())
    _integer(value["process_id"], label + "_PROCESS_ID", minimum=1)
    _require(value["monotonic_before_ns"] <= value["monotonic_after_ns"],
             label + "_MONOTONIC_BRACKET")
    try:
        valid_boot = isinstance(value["boot_id"], str) and str(uuid.UUID(value["boot_id"])) == value["boot_id"]
    except (ValueError, AttributeError):
        valid_boot = False
    _require(valid_boot, label + "_BOOT_ID")
    if "utc_interval" in value:
        interval = value["utc_interval"]
        _keys(interval, ("lower_ns", "upper_ns", "evidence_ref"), label + "_UTC_INTERVAL")
        low = _integer(interval["lower_ns"], label + "_UTC_LOWER")
        high = _integer(interval["upper_ns"], label + "_UTC_UPPER")
        _require(low <= high, label + "_UTC_INTERVAL_ORDER")
        _ref(interval["evidence_ref"], label + "_UTC_EVIDENCE")
        # Bounds describe true UTC under the referenced external evidence. The
        # raw local utc_ns need not lie inside them if that evidence corrects offset.
    return value


def capture_clock_sample():
    """Bracket one UTC reading with monotonic readings and actual process identity.

    An absent utc_interval means UNKNOWN absolute clock accuracy. The sampler
    never manufactures that interval. Monotonic values are comparable only
    within the exact boot/process/start-ticks epoch used by this contract.
    """
    pid = os.getpid()
    stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace").strip()
    left, right = stat.find("("), stat.rfind(")")
    _require(left > 0 and right > left and int(stat[:left].strip()) == pid, "PROC_STAT")
    fields = stat[right + 1:].split()  # field 3 onward; field 22 is starttime.
    _require(len(fields) >= 20, "PROC_STAT_FIELDS")
    start_ticks = int(fields[19])
    boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    before = time.monotonic_ns()
    utc = time.time_ns()
    after = time.monotonic_ns()
    result = {"utc_ns": utc, "monotonic_before_ns": before, "monotonic_after_ns": after,
              "boot_id": boot, "process_id": pid, "process_start_ticks": start_ticks}
    _sample(result, "CAPTURE")
    return result


def _binding_payload(receipt):
    """Exact value/artifact identities; physical stamps are validated separately."""
    m1, m5, atr, ready = (receipt[key] for key in ("m1", "m5", "atr", "input_ready"))
    return {
        "schema_version": RECEIPT_SCHEMA,
        "issue_ns": _integer(receipt["issue_ns"], "ISSUE"),
        "m1": {"ref": _ref(m1["ref"], "M1"), "bar_close_ns": _integer(m1["bar_close_ns"], "M1_BAR_CLOSE")},
        "m5": {"ref": _ref(m5["ref"], "M5"), "bar_close_ns": _integer(m5["bar_close_ns"], "M5_BAR_CLOSE")},
        "atr": {"source_m5_ref": _ref(atr["source_m5_ref"], "ATR_SOURCE_M5"),
                "contract_sha256": _sha(atr["contract_sha256"], "ATR_CONTRACT"),
                "value_sha256": _sha(atr["value_sha256"], "ATR_VALUE"),
                "bar_close_ns": dict(atr["bar_close_ns"])},
        "input_sha256": _sha(ready["input_sha256"], "INPUT"),
    }


def input_binding_sha256(receipt):
    """Bind source manifests, ATR snapshot/contract/closes and published input SHA.

    input_sha256 must identify the actual published input-only snapshot, not a
    model batch. This helper does not read that snapshot or any referenced file.
    The receipt's binding_sha256 may be absent while constructing the receipt.
    """
    return hashlib.sha256(json.dumps(_binding_payload(receipt), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _epoch(sample):
    return tuple(sample[key] for key in ("boot_id", "process_id", "process_start_ticks"))


def _relation(before, after, label):
    if _epoch(before) != _epoch(after):
        return  # restart/another process is UNKNOWN, never inferred ordering.
    _require(before["monotonic_after_ns"] <= after["monotonic_before_ns"],
             label + "_MONOTONIC_ORDER")
    _require(before["utc_ns"] <= after["utc_ns"], label + "_UTC_BACKWARD")
    if "utc_interval" in before and "utc_interval" in after:
        _require(before["utc_interval"]["lower_ns"] <= after["utc_interval"]["upper_ns"],
                 label + "_UTC_INTERVAL_CONTRADICTION")


def _deadline_status(sample, issue):
    interval = sample.get("utc_interval")
    if interval is None:
        return "UNKNOWN"
    if interval["upper_ns"] <= issue:
        return "ON_TIME"
    if interval["lower_ns"] > issue:
        return "LATE"
    return "UNKNOWN"


def assess_input_timing(receipt, *, expected_m1_ref, expected_m5_ref,
                        expected_atr_contract_sha256):
    """Assess one declared slot, without values, outcomes, models or file I/O.

    Both source callbacks must precede their own publication. M5 publication
    precedes ATR completion; both publications and ATR precede input readiness.
    M1 and M5 need no mutual order. Missing UTC evidence or any epoch change
    yields UNKNOWN. Impossible order, wrong metadata and wrong bindings raise.
    """
    _keys(receipt, ("schema_version", "issue_ns", "m1", "m5", "atr", "input_ready"), "RECEIPT")
    _require(receipt["schema_version"] == RECEIPT_SCHEMA, "SCHEMA")
    issue = _integer(receipt["issue_ns"], "ISSUE")
    m1, m5, atr, ready = (receipt[key] for key in ("m1", "m5", "atr", "input_ready"))
    for name, source, expected in (("M1", m1, expected_m1_ref), ("M5", m5, expected_m5_ref)):
        _keys(source, ("ref", "bar_close_ns", "received", "published"), name)
        _require(_ref(source["ref"], name) == _ref(expected, "EXPECTED_" + name), name + "_SOURCE_BINDING")
        _require(_integer(source["bar_close_ns"], name + "_BAR_CLOSE") <= issue, name + "_BAR_AFTER_ISSUE")
    _keys(atr, ("source_m5_ref", "contract_sha256", "value_sha256", "bar_close_ns", "ready"), "ATR")
    _require(_ref(atr["source_m5_ref"], "ATR_SOURCE") == _ref(expected_m5_ref, "EXPECTED_M5"),
             "ATR_M5_SOURCE_BINDING")
    _require(_sha(atr["contract_sha256"], "ATR_CONTRACT") ==
             _sha(expected_atr_contract_sha256, "EXPECTED_ATR_CONTRACT"), "ATR_CONTRACT_BINDING")
    _sha(atr["value_sha256"], "ATR_VALUE")
    _keys(atr["bar_close_ns"], TIMEFRAMES, "ATR_TIMEFRAMES")
    for tf in TIMEFRAMES:
        _require(_integer(atr["bar_close_ns"][tf], "ATR_" + tf + "_BAR_CLOSE") <= issue,
                 "ATR_" + tf + "_BAR_AFTER_ISSUE")
    _keys(ready, ("stamp", "input_sha256", "binding_sha256"), "INPUT_READY")
    _sha(ready["input_sha256"], "INPUT")
    binding = input_binding_sha256(receipt)
    _require(_sha(ready["binding_sha256"], "INPUT_BINDING") == binding, "INPUT_BINDING_MISMATCH")
    stamps = {
        "m1.received": m1["received"], "m1.published": m1["published"],
        "m5.received": m5["received"], "m5.published": m5["published"],
        "atr.ready": atr["ready"], "input_ready": ready["stamp"],
    }
    for name, stamp in stamps.items():
        _sample(stamp, name.upper())
    edges = (("m1.received", "m1.published"), ("m5.received", "m5.published"),
             ("m5.published", "atr.ready"), ("m1.published", "input_ready"),
             ("m5.published", "input_ready"), ("atr.ready", "input_ready"))
    for first, last in edges:
        _relation(stamps[first], stamps[last], first.upper() + "_TO_" + last.upper())
    # Catch backwards UTC even between independent M1 and M5 branches, without
    # inventing an order between overlapping monotonic brackets.
    items = list(stamps.items())
    for i, (first_name, first) in enumerate(items):
        for last_name, last in items[i + 1:]:
            if _epoch(first) != _epoch(last):
                continue
            if first["monotonic_after_ns"] <= last["monotonic_before_ns"]:
                _relation(first, last, first_name.upper() + "_TO_" + last_name.upper())
            elif last["monotonic_after_ns"] <= first["monotonic_before_ns"]:
                _relation(last, first, last_name.upper() + "_TO_" + first_name.upper())
    reasons = []
    if len({_epoch(stamp) for stamp in stamps.values()}) != 1:
        reasons.append("CLOCK_EPOCH_CHANGED_OR_RESTART_UNQUALIFIED")
    missing = [name for name, stamp in stamps.items() if "utc_interval" not in stamp]
    reasons.extend("UTC_INTERVAL_UNKNOWN:" + name for name in missing)
    status = "UNKNOWN" if reasons else _deadline_status(ready["stamp"], issue)
    if not reasons and status == "UNKNOWN":
        reasons.append("INPUT_READY_UTC_INTERVAL_STRADDLES_ISSUE")
    elif status == "LATE":
        reasons.append("INPUT_READY_AFTER_ISSUE")
    return {
        "schema_version": RECEIPT_SCHEMA + "_assessment",
        "status": status, "reasons": reasons, "issue_ns": issue,
        "input_binding_sha256": binding,
        "component_status": {name: _deadline_status(stamp, issue) for name, stamp in stamps.items()},
        "assessment_scope": "supplied_identity_and_external_clock_bounds_only_no_file_or_clock_evidence_verification",
    }
