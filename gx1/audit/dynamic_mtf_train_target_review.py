"""One TRAIN-prefix measurement for the preregistered dynamic-state target review."""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target
from gx1.contracts.gx1_capped_execution_v1 import (
    require_capped_cpu_audit_execution, require_capped_lock_ancestry,
)
from gx1.utils.artifact_primitives_v1 import require_immutable_artifact, sha256_file


def require(condition, message):
    if not condition:
        raise ValueError(message)


def supported_moves(opening, high, low, close):
    """Largest individually proven rise and fall in each row of ordered M1 bars."""
    earlier_low = np.minimum.accumulate(np.column_stack((opening[:, 0], low[:, :-1])), axis=1)
    earlier_high = np.maximum.accumulate(np.column_stack((opening[:, 0], high[:, :-1])), axis=1)
    rise = np.maximum(high - np.minimum(earlier_low, opening), close - low)
    fall = np.maximum(np.maximum(earlier_high, opening) - low, high - close)
    return rise.max(axis=1), fall.max(axis=1)


def summarize(tf, lead, issues, axis, prices, arrays, cutoff):
    col = target.TIMEFRAMES.index(tf)
    valid = issues + lead * target.MINUTE_NS
    mature = valid < cutoff
    requested = valid[mature]
    pos = np.searchsorted(axis, requested)
    present = (pos < len(axis)) & (axis[np.minimum(pos, len(axis) - 1)] == requested)
    rows = pos[present]
    reason = arrays["reason"][rows, col]
    blocked = target.ATR_UNAVAILABLE | target.LOCAL_SPAN_INCOMPLETE | target.UNKNOWN_SOURCE_GAP
    complete = (reason & blocked) == 0
    require(np.array_equal(arrays["maturity_ns"][rows, col], requested[present]), "maturity mismatch")
    require(np.array_equal(arrays["observed"][rows, col], arrays["state"][rows, col] != target.UNKNOWN),
            "state mask mismatch")
    endpoint_known = int(arrays["observed"][rows, col].sum())
    rows = rows[complete]
    theta = arrays["threshold"][rows, col]
    span = arrays["local_range"][rows, col]
    labels = arrays["state"][rows, col]
    require(np.all(np.isfinite(theta) & (theta > 0)) and np.all(np.isfinite(span)), "unsupported scale")
    require(np.array_equal(span < theta, labels == target.FLAT), "local-range / FLAT mismatch")
    window = target.LOCAL_WINDOW_M1_BARS[tf]
    require(len(rows) and rows.min() >= window - 1, "incomplete history in complete set")
    opening, high, low, close = prices
    both = np.zeros(len(rows), dtype=bool)
    # Bound the largest working array for D1. The chunk size does not select observations.
    offsets = np.arange(1 - window, 1)
    for start in range(0, len(rows), 128):
        sl = slice(start, start + 128)
        idx = rows[sl, None] + offsets
        rise, fall = supported_moves(opening[idx], high[idx], low[idx], close[idx])
        require(np.all(np.maximum(rise, fall) <= span[sl]), "ordered price witness exceeds target range")
        both[sl] = np.minimum(rise, fall) >= theta[sl]
    names = ("UNKNOWN", *target.STATE_NAMES)
    classes = (target.UNKNOWN, target.DOWN, target.FLAT, target.UP)
    counts = {name: int((labels == k).sum()) for name, k in zip(names, classes)}
    witnesses = {name: int((both & (labels == k)).sum()) for name, k in zip(names, classes)}
    require(sum(counts.values()) == len(rows) and witnesses["FLAT"] == 0, "target partition")
    nonflat = len(rows) - counts["FLAT"]
    report = {
        "timeframe": tf, "lead_minutes": lead, "parents": len(issues),
        "immature": int((~mature).sum()), "missing_mature_m1_endpoint": int((~present).sum()),
        "exact_mature_m1_endpoint": len(complete), "all_known_endpoint_classes": endpoint_known,
        "incomplete_local_window_or_atr": int((~complete).sum()),
        "complete_local_windows": len(rows), "complete_window_endpoint_counts": counts,
        "both_moves_proven_by_endpoint_class": witnesses, "nonflat_complete": nonflat,
        "up_share_identification_bounds_among_nonflat": (
            [counts["UP"] / nonflat, (counts["UP"] + counts["UNKNOWN"]) / nonflat] if nonflat else None),
    }
    if tf == "M5":
        interior_wide = ((high[rows] - low[rows] >= theta)
                         & (close[rows] > low[rows]) & (close[rows] < high[rows]))
        unknown = labels == target.UNKNOWN
        require(np.all(~interior_wide | unknown), "interior of wide M5 end bar assigned a direction")
        report["unknown_with_wide_interior_end_bar"] = int((unknown & interior_wide).sum())
        report["other_unknown"] = int((unknown & ~interior_wide).sum())
        report["unknown_with_intraminute_order_flag"] = int((
            unknown & ((reason[complete] & target.INTRAMINUTE_ORDER_UNKNOWN) != 0)).sum())
    return report


def main():
    guard = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    policy = json.loads(Path("NEXT_RUN_POLICY.json").read_bytes())
    gate = policy["current_work"]["dynamic_mtf_train_gates_20261104"]
    spec = gate["specification"]
    require(sha256_file(Path(spec["path"])) == spec["sha256"] and not policy["training_enabled"]
            and gate["status"] == "SPECIFICATION_FROZEN_NO_NATIVE_ADMISSION"
            and not gate["model_calls_authorized"], "closed preregistration required")
    # A digest may cover a mixed development artifact. Only data inside the fit prefix are decoded for analysis.
    stamps = {}

    def stamp(path):
        st = path.stat()
        return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns

    def checked(ref):
        path = Path(ref["path"])
        initial = stamp(path)
        require_immutable_artifact(path, expected_sha256=ref["sha256"], context="TRAIN_TARGET_REVIEW")
        require(stamp(path) == initial, "source changed during verification")
        stamps[path] = initial
        return path

    source_receipt = policy["dynamic_mtf_state_target_audit_20261102_005"]["result"]
    require(source_receipt["sha256"] == "007b1518c9ed1bfdd3eb8024b60421511dd158dbb2c16d2532bba4328214aa6b",
            "preregistered target identity")
    source = json.loads(checked(source_receipt).read_bytes())
    fold = source["fold"]
    start = pd.Timestamp(source["clock_slice_proof"]["requested_start_ns"], tz="UTC")
    cutoff = pd.Timestamp(gate["fit_cutoff_utc_exclusive"])
    require(cutoff.value < fold["train_cutoff_ns"] == pd.Timestamp(gate["train_oof_cutoff_utc_exclusive"]).value,
            "TRAIN bounds")
    manifest = target.read_dynamic_mtf_state_series_metadata(
        source["series"], expected_source_refs=source["target_sources"], expected_cutoff_ns=fold["series_cutoff_ns"])
    raw = checked(source["target_sources"]["m1"])
    fields = ["time", "open", "high", "low", "close"]
    exclusive = cutoff - pd.Timedelta(target.MINUTE_NS, unit="ns")
    table = pq.read_table(raw, columns=fields, filters=[
        ("time", ">=", start.to_pydatetime()), ("time", "<", exclusive.to_pydatetime())], use_threads=False)
    bars = table.to_pandas()
    axis = pd.DatetimeIndex(bars["time"]).asi8 + target.MINUTE_NS
    require(len(axis) and axis[0] >= start.value + target.MINUTE_NS and axis[-1] < cutoff.value
            and np.all(np.diff(axis) > 0), "raw M1 time axis")
    prices = tuple(bars[name].to_numpy() for name in fields[1:])
    opening, high, low, close = prices
    require(np.isfinite(prices).all() and np.all(low <= np.minimum(opening, close))
            and np.all(high >= np.maximum(opening, close)), "invalid source OHLC")

    def array(name):
        return np.load(checked(manifest["arrays"][name]), mmap_mode="r", allow_pickle=False)

    prefix = gate["fit_parents"]
    issue_axis = array("issue_ns")
    issues = issue_axis[:prefix]
    require(len(issues) == prefix and np.all(np.diff(issues) > 0) and issues[-1] < cutoff.value <= issue_axis[prefix]
            and np.array_equal(array("parent_rows")[:prefix], np.arange(prefix)), "prefit geometry")
    saved_axis = array("validity_ns")
    require(np.array_equal(axis, saved_axis[:len(axis)]) and saved_axis[len(axis)] >= cutoff.value,
            "source / target clock mismatch")
    arrays = {key: array(key)[:len(axis)] for key in ("state", "observed", "maturity_ns", "reason", "threshold", "local_range")}
    panels = [summarize(tf, target.LOCAL_WINDOW_M1_BARS[tf], issues, axis, prices, arrays, cutoff.value)
              for tf in target.TIMEFRAMES]
    first = summarize("M5", 1, issues, axis, prices, arrays, cutoff.value)
    require(all(stamp(path) == identity for path, identity in stamps.items())
            and require_capped_lock_ancestry() == lock, "changed input or lock")
    report = {
        "schema": "dynamic_mtf_train_target_review_20261104", "observed_utc": datetime.now(timezone.utc).isoformat(),
        "specification": spec, "audit_source_sha256": sha256_file(Path(__file__)), "guard": guard,
        "source_receipt": source_receipt, "m1_sha256": source["target_sources"]["m1"]["sha256"],
        "series_sha256": source["series"]["sha256"], "target_contract_sha256": source["target_contract_sha256"],
        "target_rule_locked": False, "train_fit_cutoff_exclusive": cutoff.isoformat(), "parents": prefix,
        "first_issue_utc": pd.Timestamp(issues[0], tz="UTC").isoformat(),
        "last_issue_utc": pd.Timestamp(issues[-1], tz="UTC").isoformat(), "m1_prefix_rows_including_context": len(axis),
        "native_leads": panels, "m5_first_minute": first, "model_calls": 0, "optimizer_updates": 0,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
