"""Derive fixed 30/60-minute direction labels from saved endpoint differences.

Pure array mathematics: no loading, fitting, model, or execution authority.
Shared price-row identities make adjacent differences telescope. Outward FP64
intervals certify sign without reconstructing price levels or assuming exact sums.
"""
from collections.abc import Mapping

import numpy as np

SCHEMA_VERSION = "gx1_direction_saved_horizons_v1"
BASE_HORIZONS = (1, 5, 15)
HORIZONS = (30, 60)
MINUTE_NS = 60 * 10**9
MEMBERS = frozenset((
    "parent_rows", "bar_start_ns", "dynamic_entry_m1_row", "decision_ns",
    "terminal_time_ns", "elapsed_wall_seconds", "terminal_source_row",
    "observed", "mid_move", "mid_return_bps", "sign",
))
CHAIN_REASONS = {0: "observed_chain", 1: "right_censored",
                 2: "missing_intermediate_parent", 3: "unobserved_component"}


def _require(value, reason):
    if not value:
        raise RuntimeError("DIRECTION_SAVED_HORIZONS_" + reason)


def _validate(saved, cutoff_ns):
    _require(isinstance(saved, Mapping) and set(saved) == MEMBERS, "MEMBERS")
    _require(type(cutoff_ns) is int and 0 < cutoff_ns <= np.iinfo(np.int64).max, "CUTOFF")
    parent = saved["parent_rows"]
    _require(isinstance(parent, np.ndarray) and parent.dtype == np.int64
             and parent.ndim == 1 and len(parent) > 0, "PARENT_HEADER")
    n = len(parent)
    headers = {
        "parent_rows": (np.int64, (n,)), "bar_start_ns": (np.int64, (n,)),
        "dynamic_entry_m1_row": (np.int64, (n,)), "decision_ns": (np.int64, (n,)),
        "terminal_time_ns": (np.int64, (n, 3)), "elapsed_wall_seconds": (np.int64, (n, 3)),
        "terminal_source_row": (np.int64, (n, 3)), "observed": (np.bool_, (n, 3)),
        "mid_move": (np.float64, (n, 3)), "mid_return_bps": (np.float64, (n, 3)),
        "sign": (np.int8, (n, 3)),
    }
    _require(all(isinstance(saved[k], np.ndarray) and saved[k].dtype == dtype
                 and saved[k].shape == shape and np.isfinite(saved[k]).all()
                 for k, (dtype, shape) in headers.items()), "HEADERS_FINITE")
    decision = saved["decision_ns"]
    _require(np.all(parent >= 0) and np.all(np.diff(parent) > 0)
             and np.all(decision > 0) and np.all(np.diff(decision) > 0)
             and np.all(decision % (5 * MINUTE_NS) == 0)
             and np.all(decision <= np.iinfo(np.int64).max - 60 * MINUTE_NS)
             and np.array_equal(decision, saved["bar_start_ns"] + 5 * MINUTE_NS),
             "PARENT_DECISION_CLOCK")
    horizons = np.asarray(BASE_HORIZONS, dtype=np.int64)
    _require(np.array_equal(saved["terminal_time_ns"], decision[:, None] + horizons * MINUTE_NS)
             and np.array_equal(saved["elapsed_wall_seconds"],
                                np.broadcast_to(horizons * 60, (n, 3))), "EXACT_BASE_ENDPOINT_CLOCKS")
    observed = saved["observed"]
    _require(np.all(saved["terminal_time_ns"][observed] < cutoff_ns)
             and np.all(saved["terminal_source_row"][observed] >= 0)
             and np.all(saved["dynamic_entry_m1_row"][np.any(observed, axis=1)] >= 0),
             "OBSERVED_SOURCE_OR_CUTOFF")
    _require(np.array_equal(saved["sign"], np.sign(saved["mid_move"]).astype(np.int8))
             and np.all(saved["mid_move"][~observed] == 0)
             and np.all(saved["mid_return_bps"][~observed] == 0)
             and np.all(saved["terminal_source_row"][~observed] == -1), "MASKED_STORAGE_OR_SIGN")
    return n


def outward_sum_bounds(parts):
    """Enclose the exact sum of differences rounded once into each saved value."""
    _require(isinstance(parts, np.ndarray) and parts.dtype == np.float64
             and parts.ndim == 2 and parts.shape[0] > 0 and np.isfinite(parts).all(), "PARTS")
    low = np.zeros(parts.shape[1], dtype=np.float64)
    high = low.copy()
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        for part in parts:
            low = np.nextafter(low + np.nextafter(part, -np.inf), -np.inf)
            high = np.nextafter(high + np.nextafter(part, np.inf), np.inf)
    _require(np.isfinite(low).all() and np.isfinite(high).all()
             and np.all(low <= high), "INTERVAL_OVERFLOW")
    return low, high


def _compose(saved, *, step, count, cutoff_ns):
    decision = saved["decision_ns"]
    n = len(decision)
    column = BASE_HORIZONS.index(step)
    end = decision + step * count * MINUTE_NS
    active = end < cutoff_ns
    reason = np.where(active, 0, 1).astype(np.int8)
    positions, present, observed = [], [], []
    for j in range(count):
        expected = decision + j * step * MINUTE_NS
        at = np.searchsorted(decision, expected)
        safe = np.minimum(at, n - 1)
        exists = (at < n) & (decision[safe] == expected)
        seen = exists & saved["observed"][safe, column]
        reason[active & ~exists] = 2
        active &= exists
        reason[active & ~seen] = 3
        active &= seen
        positions.append(safe)
        present.append(exists)
        observed.append(seen)
        if j:
            joined = (end < cutoff_ns) & observed[j - 1] & seen
            _require(np.array_equal(
                saved["terminal_source_row"][positions[j - 1], column][joined],
                saved["dynamic_entry_m1_row"][safe][joined]), "SHARED_PRICE_ROW")
    values = np.stack([saved["mid_move"][at, column] for at in positions])
    low = np.zeros(n, dtype=np.float64)
    high = low.copy()
    low[active], high[active] = outward_sum_bounds(np.ascontiguousarray(values[:, active]))
    up, down = active & (low > 0), active & (high < 0)
    # Subtracting two finite FP64 values gives zero only when they are equal.
    unchanged = active & np.all(values == 0, axis=0)
    resolved = up | down | unchanged
    sign = np.zeros(n, dtype=np.int8)
    sign[up], sign[down] = 1, -1
    source_row = np.full(n, -1, dtype=np.int64)
    source_row[active] = saved["terminal_source_row"][positions[-1], column][active]
    return {
        "terminal_time_ns": end, "terminal_source_row": source_row,
        "observed_chain": active, "label_resolved": resolved, "sign": sign,
        "mid_move_lower": low, "mid_move_upper": high, "chain_reason": reason,
        "rounding_ambiguous": active & ~resolved, "certified_zero": unchanged,
    }


def derive_direction_horizons(saved, *, cutoff_ns):
    """Keep every parent; derive only fixed 30/60 labels and a direct 15m control."""
    n = _validate(saved, cutoff_ns)
    control = _compose(saved, step=5, count=3, cutoff_ns=cutoff_ns)
    seen = control["observed_chain"]
    _require(np.all(saved["observed"][seen, 2])
             and np.array_equal(control["terminal_source_row"][seen],
                                saved["terminal_source_row"][seen, 2]), "CONTROL15_ENDPOINT_IDENTITY")
    direct = saved["mid_move"][:, 2]
    _require(np.all(control["mid_move_lower"][seen] <= np.nextafter(direct[seen], np.inf))
             and np.all(control["mid_move_upper"][seen] >= np.nextafter(direct[seen], -np.inf)),
             "CONTROL15_DIFFERENCE_INTERVAL")
    resolved = control["label_resolved"]
    _require(np.array_equal(control["sign"][resolved], saved["sign"][resolved, 2]),
             "CONTROL15_SIGN_PARITY")
    parts = [_compose(saved, step=15, count=h // 15, cutoff_ns=cutoff_ns) for h in HORIZONS]
    arrays = {name: np.ascontiguousarray(np.column_stack([part[name] for part in parts]))
              for name in parts[0]}
    arrays.update(parent_rows=saved["parent_rows"].copy(), decision_ns=saved["decision_ns"].copy())
    counts = []
    for h, part in zip(HORIZONS, parts):
        counts.append({
            "horizon_minutes": h, "parent_rows": n,
            **{name: int(np.count_nonzero(part["chain_reason"] == code))
               for code, name in CHAIN_REASONS.items()},
            "resolved_nonzero": int(np.count_nonzero(part["label_resolved"] & (part["sign"] != 0))),
            "certified_zero": int(np.count_nonzero(part["certified_zero"])),
            "rounding_ambiguous": int(np.count_nonzero(part["rounding_ambiguous"])),
        })
    for value in arrays.values():
        value.setflags(write=False)
    return {
        "schema_version": SCHEMA_VERSION, "horizons_minutes": HORIZONS, "arrays": arrays, "counts": counts,
        "control15": {
            "parent_rows": n, "direct_observed": int(np.count_nonzero(saved["observed"][:, 2])),
            "chain_observed": int(np.count_nonzero(seen)),
            "resolved_signs_checked": int(np.count_nonzero(resolved)),
            "rounding_ambiguous": int(np.count_nonzero(control["rounding_ambiguous"])),
            "endpoint_identity_passed": True, "difference_intervals_passed": True, "sign_parity_passed": True,
        },
        "population_scope": "chain_observed_with_separate_rounding_certification",
        "price_levels_reconstructed": False, "model_input_authority": False,
    }
