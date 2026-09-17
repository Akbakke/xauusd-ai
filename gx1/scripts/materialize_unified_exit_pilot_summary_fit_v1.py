#!/usr/bin/env python3
"""Materialize outcome-blind lifecycle counts and TRAIN lifetime summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gx1.contracts.unified_exit_market_closure_authority_v1 import (
    m1_clock_sha256,
    require_market_closure_authority,
)
from gx1.contracts.unified_exit_pilot_normalization_v1 import (
    build_physical_summary_sample_authority,
    canonical_sha256,
    fit_lifetime_summary_normalization,
    iter_physical_summary_samples,
)


SCHEMA_VERSION = "gx1_unified_exit_pilot_summary_fit_inputs_v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError("PILOT_SUMMARY_JSON_INVALID")
    return value


def _signed_log1p(value: float) -> float:
    return float(np.copysign(np.log1p(abs(value)), value))


class _SparseRange:
    def __init__(self, values: np.ndarray, *, maximum: bool) -> None:
        base = np.asarray(values, dtype="<f8")
        self.maximum = maximum
        self.values = [base]
        self.indices = [np.arange(len(base), dtype=np.int64)]
        width = 2
        while width <= len(base):
            half = width // 2
            left = self.values[-1][:-half]
            right = self.values[-1][half:]
            left_i = self.indices[-1][:-half]
            right_i = self.indices[-1][half:]
            choose_left = left >= right if maximum else left <= right
            self.values.append(np.where(choose_left, left, right))
            self.indices.append(np.where(choose_left, left_i, right_i))
            width *= 2

    def query(self, left: int, right_exclusive: int) -> tuple[float, int]:
        length = right_exclusive - left
        level = length.bit_length() - 1
        width = 1 << level
        right_start = right_exclusive - width
        lv, li = self.values[level], self.indices[level]
        a, b = float(lv[left]), float(lv[right_start])
        ai, bi = int(li[left]), int(li[right_start])
        if (a > b if self.maximum else a < b) or (a == b and ai <= bi):
            return a, ai
        return b, bi


def _prefix_fit_state_stops(
    *, times: pd.DatetimeIndex, starts: np.ndarray, counts: np.ndarray,
    entry_rows: np.ndarray, cutoff_time_ns: int,
) -> np.ndarray:
    """Bound normalization observations only; retain market successor counts."""
    if (type(cutoff_time_ns) is not int or cutoff_time_ns <= 0
            or entry_rows.dtype != np.dtype("int64") or entry_rows.ndim != 1
            or not len(entry_rows) or np.any(entry_rows < 0)
            or np.any(entry_rows >= len(starts)) or np.any(np.diff(entry_rows) <= 0)
            or times.empty or times.hasnans or not times.is_unique or not times.is_monotonic_increasing
            or starts.ndim != 1 or counts.shape != starts.shape):
        raise RuntimeError("PILOT_SUMMARY_PREFIX_FIT_POPULATION_INVALID")
    # M1 timestamps are bar starts: a row is available only at bar close.
    stop = int(np.searchsorted(times.asi8, cutoff_time_ns - 60_000_000_000, side="right"))
    limits = np.zeros(len(counts), dtype="<i8")
    limits[entry_rows] = np.minimum(counts[entry_rows], stop - starts[entry_rows])
    if np.any(limits[entry_rows] < 1):
        raise RuntimeError("PILOT_SUMMARY_PREFIX_ENTRY_AFTER_CUTOFF")
    return limits


def materialize(
    *,
    split: str,
    child_admission_path: Path,
    m1_path: Path,
    m1_manifest_path: Path,
    closure_path: Path,
    output_dir: Path,
    publish: bool,
    fit_entry_rows_path: Path | None = None,
    fit_cutoff_time_ns: int | None = None,
) -> dict[str, Any]:
    prefix_fit = fit_entry_rows_path is not None
    if (prefix_fit != (fit_cutoff_time_ns is not None)
            or (prefix_fit and split != "train")):
        raise RuntimeError("PILOT_SUMMARY_PREFIX_SCOPE_INVALID")
    if split not in {"train", "val"}:
        raise RuntimeError("PILOT_SUMMARY_SPLIT_INVALID")
    admission = _json(child_admission_path)
    child = admission["splits"][split]
    m1_manifest = _json(m1_manifest_path)
    m1_sha = _sha(m1_path)
    if (
        admission.get("decision") != "PASS"
        or type(child["rows"]) is not int or child["rows"] < 1
        or (split == "val" and child["rows"] != 5_508)
        or _sha(Path(child["parquet_path"])) != child["parquet_sha256"]
        or m1_manifest.get("split") != split
        or m1_manifest.get("output_parquet_sha256") != m1_sha
        or m1_manifest.get("right_censor_time_utc_exclusive")
        != ({"train": "2026-06-01T00:00:00+00:00", "val": "2026-07-01T00:00:00+00:00"}[split])
    ):
        raise RuntimeError("PILOT_SUMMARY_SOURCE_INVALID")
    table = pq.read_table(
        m1_path,
        columns=["time", "bid_open", "bid_high", "bid_low", "bid_close", "ask_open", "ask_high", "ask_low", "ask_close"],
    )
    times = pd.DatetimeIndex(table["time"].to_pandas()).as_unit("ns")
    closure = require_market_closure_authority(
        _json(closure_path),
        expected_m1_source_sha256=m1_sha,
        expected_m1_clock_sha256=m1_clock_sha256(times),
    )
    entry_times = pd.DatetimeIndex(
        pq.read_table(child["parquet_path"], columns=["time"])["time"].to_pandas()
    ).as_unit("ns")
    # The admitted child determines TRAIN scope, including the full five-year view.
    # Verify its complete physical row population instead of a pilot-size constant.
    if (len(entry_times) != child["rows"] or entry_times.hasnans
            or not entry_times.is_unique or not entry_times.is_monotonic_increasing):
        raise RuntimeError("PILOT_SUMMARY_ADMITTED_ENTRY_POPULATION_INVALID")
    starts = np.searchsorted(times.asi8, entry_times.asi8 + 300_000_000_000)
    if np.any(starts >= len(times)) or not np.array_equal(times.asi8[starts], entry_times.asi8 + 300_000_000_000):
        raise RuntimeError("PILOT_SUMMARY_FIRST_STATE_INVALID")
    unknown = np.asarray(
        [item["gap_after_m1_row"] for item in closure["intervals"] if not item["successor_across_gap_allowed"]],
        dtype=np.int64,
    )
    counts = np.empty(len(starts), dtype="<i8")
    for index, start in enumerate(starts):
        offset = int(np.searchsorted(unknown, start))
        stop = len(times)
        if offset < len(unknown):
            stop = min(stop, int(unknown[offset]) + 1)
        counts[index] = stop - int(start) - 1
    if np.any(counts < 1):
        raise RuntimeError("PILOT_SUMMARY_SUCCESSOR_POPULATION_INVALID")
    lineage = canonical_sha256(
        {
            "split": split,
            "child_admission_sha256": _sha(child_admission_path),
            "child_parquet_sha256": child["parquet_sha256"],
            "m1_source_sha256": m1_sha,
            "m1_manifest_sha256": _sha(m1_manifest_path),
            "closure_authority_sha256": closure["artifact_sha256"],
            "right_censor_time_utc_exclusive": m1_manifest["right_censor_time_utc_exclusive"],
        }
    )
    fit_limits = None
    fit_population = None
    if prefix_fit:
        rows_path = fit_entry_rows_path.expanduser().absolute()
        if not rows_path.is_file() or rows_path.is_symlink() or rows_path.resolve() != rows_path:
            raise RuntimeError("PILOT_SUMMARY_PREFIX_ROWS_BINDING_INVALID")
        entry_rows = np.load(rows_path, allow_pickle=False)
        fit_limits = _prefix_fit_state_stops(
            times=times, starts=starts, counts=counts, entry_rows=entry_rows,
            cutoff_time_ns=fit_cutoff_time_ns)
        ends = starts[entry_rows] + fit_limits[entry_rows] - 1
        fit_population = {
            "entry_rows": {"path": str(rows_path), "sha256": _sha(rows_path)},
            "cutoff_time_ns": fit_cutoff_time_ns,
            "maximum_observed_decision_time_ns": int(times.asi8[ends].max()) + 60_000_000_000,
            "fit_state_stop_exclusive_by_entry_sha256": hashlib.sha256(fit_limits.tobytes()).hexdigest(),
            "market_successor_counts_preserved": True,
        }
        fit_limits = [int(value) for value in fit_limits]
    authority = build_physical_summary_sample_authority(
        successor_transition_count_by_entry=[int(value) for value in counts],
        source_lineage_sha256=lineage,
        fit_state_stop_exclusive_by_entry=fit_limits,
    )
    summary_values: np.ndarray | None = None
    normalization: dict[str, Any] | None = None
    if split == "train":
        arrays = {name: table[name].to_numpy(zero_copy_only=False) for name in table.column_names if name != "time"}
        bid_high = _SparseRange(arrays["bid_high"], maximum=True)
        bid_low = _SparseRange(arrays["bid_low"], maximum=False)
        ask_high = _SparseRange(arrays["ask_high"], maximum=True)
        ask_low = _SparseRange(arrays["ask_low"], maximum=False)
        summary_values = np.empty((authority["fit_row_count"], 7), dtype="<f8")
        row = 0
        for sample in iter_physical_summary_samples(successor_transition_count_by_entry=[int(value) for value in counts], source_lineage_sha256=lineage, fit_state_stop_exclusive_by_entry=fit_limits):
            entry_index = sample["entry_row_index"]
            start = int(starts[entry_index])
            state = int(sample["state_index"])
            current = start + state
            entry_bid = float(arrays["bid_open"][start])
            entry_ask = float(arrays["ask_open"][start])
            long_peak, long_peak_row = bid_high.query(start, current + 1)
            long_trough, _ = bid_low.query(start, current + 1)
            short_peak_price, short_peak_row = ask_low.query(start, current + 1)
            short_trough_price, _ = ask_high.query(start, current + 1)
            elapsed = int((int(times.asi8[current]) + 60_000_000_000 - int(times.asi8[start])) // 1_000_000_000)
            bars = state + 1
            for side in (0, 1):
                if side == 0:
                    pnl = (float(arrays["bid_close"][current]) - entry_ask) / entry_ask * 10_000.0
                    mfe = max(0.0, (long_peak - entry_ask) / entry_ask * 10_000.0)
                    mae = min(0.0, (long_trough - entry_ask) / entry_ask * 10_000.0)
                    peak_row = long_peak_row
                else:
                    pnl = (entry_bid - float(arrays["ask_close"][current])) / entry_bid * 10_000.0
                    mfe = max(0.0, (entry_bid - short_peak_price) / entry_bid * 10_000.0)
                    mae = min(0.0, (entry_bid - short_trough_price) / entry_bid * 10_000.0)
                    peak_row = short_peak_row
                bars_since = current - peak_row if mfe > 0.0 else bars
                summary_values[row] = [np.log1p(bars), np.log1p(elapsed), np.log1p(mfe), _signed_log1p(mae), np.log1p(max(0.0, mfe - pnl)), np.log1p(bars_since), _signed_log1p(pnl)]
                row += 1
        normalization = fit_lifetime_summary_normalization(values=summary_values, sample_authority=authority)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "decision": "PASS",
        "split": split,
        "source_lineage_sha256": lineage,
        "child_admission_sha256": _sha(child_admission_path),
        "child_parquet_sha256": child["parquet_sha256"],
        "m1_source_sha256": m1_sha,
        "m1_manifest_sha256": _sha(m1_manifest_path),
        "closure_authority_file_sha256": _sha(closure_path),
        "closure_authority_sha256": closure["artifact_sha256"],
        "entry_pair_population": len(counts),
        "successor_counts_sha256": hashlib.sha256(counts.tobytes()).hexdigest(),
        "successor_transition_total": int(counts.sum()),
        "summary_sample_authority": authority,
        "lifetime_summary_normalization": normalization,
        "val_fit_rows": 0,
        "test_fit_rows": 0,
        "test_accessed": False,
    }
    if fit_population is not None:
        manifest["normalization_fit_population"] = fit_population
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    if not publish:
        return {"mode": "validate_no_publish", "published": False, "manifest": manifest}
    if output_dir.exists() or output_dir.is_symlink():
        raise RuntimeError("PILOT_SUMMARY_OUTPUT_EXISTS")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging.", dir=output_dir.parent))
    try:
        np.save(staging / "successor_transition_counts.npy", counts, allow_pickle=False)
        if fit_limits is not None:
            np.save(staging / "fit_state_stop_exclusive_by_entry.npy", np.asarray(fit_limits, dtype="<i8"), allow_pickle=False)
        if summary_values is not None:
            np.save(staging / "lifetime_summary_fit_values.npy", summary_values, allow_pickle=False)
        (staging / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n")
        os.rename(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"mode": "publish", "published": True, "output_dir": str(output_dir), "manifest_sha256": manifest["manifest_sha256"], "entry_pair_population": len(counts), "successor_transition_total": int(counts.sum()), "sample_count": authority["sample_count"], "normalization_sha256": normalization["normalization_sha256"] if normalization else None}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--child-admission", type=Path, required=True)
    parser.add_argument("--m1-source", type=Path, required=True)
    parser.add_argument("--m1-manifest", type=Path, required=True)
    parser.add_argument("--closure-authority", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--fit-entry-rows", type=Path)
    parser.add_argument("--fit-cutoff-time-ns", type=int)
    args = parser.parse_args(argv)
    print(json.dumps(materialize(split=args.split, child_admission_path=args.child_admission.resolve(), m1_path=args.m1_source.resolve(), m1_manifest_path=args.m1_manifest.resolve(), closure_path=args.closure_authority.resolve(), output_dir=args.output_dir.resolve(), publish=args.publish, fit_entry_rows_path=args.fit_entry_rows, fit_cutoff_time_ns=args.fit_cutoff_time_ns), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
