"""Structural audit of declared Databento GC files; never predictor admission.

Input is the vendor CSV encoding with integer nanosecond timestamps and fixed
point prices (pretty_ts=false, pretty_px=false). No trade direction is inferred
from a price change. TBBO cannot establish order-book OFI between trades.

Format owner: https://databento.com/docs/standards-and-conventions/common-fields-enums-types
Schema owner: https://databento.com/docs/schemas-and-data-formats/mbp-1
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
import re
from typing import Any

from gx1.contracts.oanda_history_ingest_approval_v1 import OANDA_HISTORY_PRETEST_END_UTC


GC_SOURCE_SCHEMA = "gx1_gc_order_flow_source_audit_v1"
GC_RESEARCH_ID = "GC_ORDER_FLOW_RESEARCH_001"
GC_DATASET = "GLBX.MDP3"
GC_LOCAL_SOURCE_ROOT = Path("/home/andre2/GX1_DATA/research") / GC_RESEARCH_ID
GC_AUDIT_OUTPUT_ROOT = Path("/home/andre2/GX1_RUNS") / GC_RESEARCH_ID
GC_PRETEST_END_NS = int(datetime.fromisoformat(OANDA_HISTORY_PRETEST_END_UTC.replace("Z", "+00:00")).timestamp()) * 1_000_000_000
GC_SCHEMAS = frozenset({"trades", "tbbo", "mbp-1"})
GC_RAW_SYMBOL = re.compile(r"GC[FGHJKMNQUVXZ][0-9]{1,2}\Z")
# Values belong to the vendor's common-fields contract, not fitted thresholds.
UNDEF_PRICE = (1 << 63) - 1
UNDEF_TIMESTAMP = (1 << 64) - 1
F_SNAPSHOT = 1 << 5
F_BAD_TS_RECV = 1 << 3
F_MAYBE_BAD_BOOK = 1 << 2
COMMON_FIELDS = (
    "ts_recv", "ts_event", "rtype", "publisher_id", "instrument_id",
    "action", "side", "price", "size", "flags", "sequence",
)
BOOK_FIELDS = ("bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00")
FIELD_MAXIMUM = {
    "ts_recv": UNDEF_TIMESTAMP, "ts_event": UNDEF_TIMESTAMP,
    "rtype": (1 << 8) - 1, "flags": (1 << 8) - 1,
    "publisher_id": (1 << 16) - 1, "instrument_id": (1 << 32) - 1,
    "price": UNDEF_PRICE, "bid_px_00": UNDEF_PRICE, "ask_px_00": UNDEF_PRICE,
    "size": (1 << 32) - 1, "sequence": (1 << 32) - 1,
    "bid_sz_00": (1 << 32) - 1, "ask_sz_00": (1 << 32) - 1,
}


def _integer(value: Any, name: str, *, maximum: int = UNDEF_TIMESTAMP) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"GC_INTEGER_REQUIRED: {name}")
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value) is None:
        raise ValueError(f"GC_INTEGER_REQUIRED: {name}")
    number = int(value)
    if not 0 <= number <= maximum:
        raise ValueError(f"GC_INTEGER_RANGE: {name}")
    return number


def required_gc_fields(schema: str) -> tuple[str, ...]:
    if schema not in GC_SCHEMAS:
        raise ValueError("GC_SCHEMA_UNSUPPORTED")
    return COMMON_FIELDS + (() if schema == "trades" else BOOK_FIELDS)


def require_gc_file_declaration(item: Mapping[str, Any]) -> dict[str, Any]:
    """Require one named outright contract, exact identity and finite audit size."""
    if item.get("provider") != "databento" or item.get("dataset") != GC_DATASET:
        raise ValueError("GC_PROVIDER_DATASET_REQUIRED")
    schema = item.get("schema")
    required_gc_fields(schema)
    if not isinstance(item.get("raw_symbol"), str) or GC_RAW_SYMBOL.fullmatch(item["raw_symbol"]) is None:
        raise ValueError("GC_OUTRIGHT_RAW_SYMBOL_REQUIRED")
    if item.get("pretty_ts") is not False or item.get("pretty_px") is not False:
        raise ValueError("GC_RAW_INTEGER_ENCODING_REQUIRED")
    if item.get("compression") not in {"none", "gzip"}:
        raise ValueError("GC_COMPRESSION_UNSUPPORTED")
    result = dict(item)
    for name in ("start_ns", "end_ns", "instrument_id", "publisher_id", "bytes", "max_records"):
        maximum = FIELD_MAXIMUM[name] if name in FIELD_MAXIMUM else UNDEF_TIMESTAMP
        result[name] = _integer(item.get(name), name, maximum=maximum)
    if (result["start_ns"] == 0 or result["start_ns"] >= result["end_ns"]
            or result["end_ns"] == UNDEF_TIMESTAMP or result["instrument_id"] == 0
            or result["publisher_id"] == 0 or result["bytes"] == 0 or result["max_records"] == 0):
        raise ValueError("GC_FINITE_INTERVAL_AND_SIZE_REQUIRED")
    if result["end_ns"] > GC_PRETEST_END_NS:
        raise ValueError("GC_SEALED_TEST_BOUNDARY")
    for name in ("path", "receipt_path"):
        if not isinstance(item.get(name), str) or not item[name]:
            raise ValueError(f"GC_PATH_REQUIRED: {name}")
    for name in ("sha256", "receipt_sha256"):
        if not isinstance(item.get(name), str) or re.fullmatch(r"[0-9a-f]{64}", item[name]) is None:
            raise ValueError(f"GC_SHA256_REQUIRED: {name}")
    return result


def audit_gc_records(rows: Iterable[Mapping[str, Any]], declaration: Mapping[str, Any]) -> dict[str, Any]:
    """Streaming structural counters, with a hard bound and no admission claim.

    Sequence jumps are not missing-packet proof on an instrument-filtered feed.
    Unknown aggressors are counted separately and never assigned zero delta.
    A hash-bound receipt is provenance material, not proof of complete coverage.
    """
    item = require_gc_file_declaration(declaration)
    fields = required_gc_fields(item["schema"])
    counts: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    volumes: Counter[str] = Counter()
    previous_recv = previous_event = previous_sequence = None
    previous_row = None
    first_recv = last_recv = maximum_gap = None
    for row in rows:
        if counts["records"] >= item["max_records"]:
            raise ValueError("GC_RECORD_BUDGET_EXCEEDED_NO_PARTIAL_SUCCESS")
        if any(name not in row for name in fields) or None in row:
            raise ValueError("GC_REQUIRED_FIELDS_MISSING_OR_EXTRA_VALUES")
        action, side = row["action"], row["side"]
        allowed_actions = {"T"} if item["schema"] in {"trades", "tbbo"} else {"A", "C", "M", "R", "T", "N"}
        if action not in allowed_actions or side not in {"A", "B", "N"}:
            raise ValueError("GC_ACTION_OR_SIDE_INVALID")
        if action == "R" and side != "N":
            raise ValueError("GC_CLEAR_BOOK_SIDE_INVALID")
        numbers = {name: _integer(row[name], name, maximum=FIELD_MAXIMUM[name])
                   for name in fields if name not in {"action", "side"}}
        recv, event = numbers["ts_recv"], numbers["ts_event"]
        if (recv == 0 or event == 0 or recv == UNDEF_TIMESTAMP or event == UNDEF_TIMESTAMP
                or not item["start_ns"] <= recv < item["end_ns"]):
            raise ValueError("GC_TIMESTAMP_OR_DECLARED_INTERVAL_INVALID")
        if numbers["instrument_id"] != item["instrument_id"] or numbers["publisher_id"] != item["publisher_id"]:
            raise ValueError("GC_INSTRUMENT_OR_PUBLISHER_MISMATCH")
        if numbers["rtype"] != (0 if item["schema"] == "trades" else 1) or numbers["flags"] > 255:
            raise ValueError("GC_RECORD_TYPE_OR_FLAGS_INVALID")
        if previous_recv is not None:
            if recv < previous_recv:
                raise ValueError("GC_RECEIVE_TIME_REVERSAL")
            maximum_gap = max(maximum_gap or 0, recv - previous_recv)
            counts["event_time_reversals"] += int(event < previous_event)
            counts["sequence_reversals"] += int(numbers["sequence"] < previous_sequence)
            counts["sequence_jumps"] += int(numbers["sequence"] > previous_sequence + 1)
        if first_recv is None:
            first_recv = recv
        last_recv = recv
        counts["records"] += 1
        actions[action] += 1
        flags = numbers["flags"]
        counts["snapshot_records"] += bool(flags & F_SNAPSHOT)
        counts["bad_receive_timestamp_records"] += bool(flags & F_BAD_TS_RECV)
        counts["possible_bad_book_records"] += bool(flags & F_MAYBE_BAD_BOOK)
        counts["event_after_receive_records"] += int(event > recv)
        fingerprint = tuple(row[name] for name in fields)
        counts["consecutive_identical_records"] += int(fingerprint == previous_row)
        if action == "T":
            if not 0 < numbers["price"] < UNDEF_PRICE or numbers["size"] == 0:
                raise ValueError("GC_EXECUTED_TRADE_PRICE_OR_SIZE_INVALID")
            counts["trade_records"] += 1
            volumes[side] += numbers["size"]
            counts["unknown_aggressor_trades"] += int(side == "N")
        if item["schema"] != "trades":
            bid, ask = numbers["bid_px_00"], numbers["ask_px_00"]
            missing = not (0 < bid < UNDEF_PRICE and 0 < ask < UNDEF_PRICE
                           and numbers["bid_sz_00"] > 0 and numbers["ask_sz_00"] > 0)
            counts["unavailable_two_sided_book_records"] += missing
            if not missing:
                counts["crossed_book_records"] += int(bid > ask)
                counts["locked_book_records"] += int(bid == ask)
        previous_recv, previous_event, previous_sequence = recv, event, numbers["sequence"]
        previous_row = fingerprint
    if not counts["records"]:
        raise ValueError("GC_NO_RECORDS")
    result = {
        "schema_version": GC_SOURCE_SCHEMA,
        "status": "STRUCTURAL_AUDIT_COMPLETE_NOT_ADMITTED",
        "source_schema": item["schema"], "raw_symbol": item["raw_symbol"],
        "counters": dict(counts), "actions": dict(actions),
        "first_ts_recv_ns": first_recv, "last_ts_recv_ns": last_recv,
        "buy_aggressor_volume_contracts": volumes["B"],
        "sell_aggressor_volume_contracts": volumes["A"],
        "unknown_aggressor_volume_contracts": volumes["N"],
        "known_aggressor_delta_contracts": volumes["B"] - volumes["A"],
        "contains_event_book_schema": item["schema"] == "mbp-1",
        "ofi_computed": False, "continuous_coverage_proven": False,
        "contract_mapping_proven": False, "source_receipt_semantics_verified": False,
        "admitted_to_model_or_economic_test": False,
        "sequence_jumps_are_gap_proof": False,
    }
    if maximum_gap is not None:
        result["max_interrecord_receive_gap_ns"] = maximum_gap
    total = sum(volumes.values())
    if total:
        result["known_aggressor_volume_share"] = (volumes["B"] + volumes["A"]) / total
    return result
