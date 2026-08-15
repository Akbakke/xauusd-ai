"""Deterministic scalability/liveness benchmark for the level registry owner.

Synthetic paths prove runtime and structural invariants only.  They make no
claim about XAUUSD quality; production fitting must still consume every row of
the declared immutable TRAIN tape.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from gx1.features.level_registry_v1 import (
    LEVEL_REGISTRY_M5_FEATURE_NAMES,
    compute_level_registry_m5_block_v1,
    fit_level_registry_hyperparameters_v1,
)
from gx1.features.technical_indicators_v1 import wilder_atr


def _frame(rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(20260814)
    increments = rng.normal(0.0, 0.32, rows)
    close = 2200.0 + np.cumsum(increments)
    upper = rng.uniform(0.08, 0.75, rows)
    lower = rng.uniform(0.08, 0.75, rows)
    index = pd.date_range("2020-01-01", periods=rows, freq="min", tz="UTC")
    frame = pd.DataFrame(
        {"high": close + upper, "low": close - lower, "close": close},
        index=index,
    )
    frame["atr"] = wilder_atr(
        frame["high"], frame["low"], frame["close"], 14
    )
    return frame


def _source(root: Path, *, frame: pd.DataFrame) -> dict[str, str]:
    paths: dict[str, Path] = {}
    for name in ("source", "tape", "pair"):
        path = (root / f"{name}.json").resolve()
        path.write_text(json.dumps({"name": name}) + "\n", encoding="utf-8")
        paths[name] = path

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    return {
        "source_artifact": str(paths["source"]),
        "source_sha256": sha(paths["source"]),
        "source_schema_version": "synthetic_closed_ohlcv_benchmark_v1",
        "source_lane": "M1",
        "tape_manifest_artifact": str(paths["tape"]),
        "tape_manifest_sha256": sha(paths["tape"]),
        "pair_manifest_artifact": str(paths["pair"]),
        "pair_manifest_sha256": sha(paths["pair"]),
        "train_split_id": "synthetic_benchmark_chronological_train_only",
        "declared_train_window_start": frame.index[0].isoformat(),
        "declared_train_window_end": frame.index[-1].isoformat(),
    }


def _eligibility_counts(
    lifecycle: list[tuple], *, rows: int
) -> tuple[np.ndarray, np.ndarray]:
    by_row: dict[int, list[tuple]] = {}
    for event in lifecycle:
        if len(event) > 1 and isinstance(event[1], (int, np.integer)):
            by_row.setdefault(int(event[1]), []).append(event)
    active: set[int] = set()
    pending: set[int] = set()
    active_rows = np.zeros(rows, dtype=np.int64)
    pending_rows = np.zeros(rows, dtype=np.int64)
    for row in range(rows):
        for event in by_row.get(row, ()):
            kind = str(event[0])
            if kind == "create":
                active.add(int(event[4]))
            elif kind == "break":
                identity = int(event[2])
                active.discard(identity)
                pending.add(identity)
            elif kind == "flip":
                identity = int(event[2])
                pending.discard(identity)
                active.add(identity)
            elif kind == "retest_fail":
                pending.discard(int(event[2]))
            elif kind == "eligibility_expiry":
                identity = int(event[2])
                active.discard(identity)
                pending.discard(identity)
        active_rows[row] = len(active)
        pending_rows[row] = len(pending)
    return active_rows, pending_rows


def _percentiles(values: np.ndarray) -> dict[str, float | int]:
    return {
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "max": int(values.max(initial=0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--fit", action="store_true")
    args = parser.parse_args()
    if args.rows < 1000:
        raise RuntimeError("[LEVEL_REGISTRY_BENCHMARK_ROWS_TOO_SMALL]")
    frame = _frame(args.rows)
    fit_seconds: float | None = None
    with tempfile.TemporaryDirectory(prefix="gx1-level-registry-bench-") as tmp:
        if args.fit:
            started = time.perf_counter()
            payload = fit_level_registry_hyperparameters_v1(
                frame,
                tf="m1",
                inner_fit_end_exclusive=int(args.rows * 0.7),
                source_provenance=_source(Path(tmp), frame=frame),
            )
            fit_seconds = time.perf_counter() - started
            threshold = float(payload["selected_threshold_atr"])
            lifetime = int(payload["learned_expiry_bars"])
            population = payload["population_configuration"]
        else:
            threshold = 0.5
            lifetime = 240
            population = None
        lifecycle: list[tuple] = []
        started = time.perf_counter()
        matrix, names, state = compute_level_registry_m5_block_v1(
            frame,
            recurrence_threshold_atr=threshold,
            max_evidence_age_bars=lifetime,
            decision_clock="m1",
            runtime_lifecycle_log=lifecycle,
            return_registry_state=True,
        )
        serve_seconds = time.perf_counter() - started

    anchors = len(state["levels"])
    carry_bytes = len(
        json.dumps(
            state,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    identity_displacements = [
        abs(float(level["center_price"]) - float(level["member_pivot_prices"][0]))
        for level in state["levels"]
    ]
    active_rows, pending_rows = _eligibility_counts(
        lifecycle, rows=len(frame)
    )
    finite = np.isfinite(matrix).all(axis=1)
    suffix = matrix[finite]
    nonzero = [
        name
        for index, name in enumerate(names)
        if suffix.size and np.any(suffix[:, index] != 0.0)
    ]
    nonconstant = [
        name
        for index, name in enumerate(names)
        if suffix.size and float(np.var(suffix[:, index], dtype=np.float64)) > 0.0
    ]
    event_counts = Counter(str(event[0]) for event in lifecycle)
    rows_per_anchor = float(args.rows) / float(anchors) if anchors else 0.0
    full_m1_rows = 2_660_000
    result = {
        "rows": int(args.rows),
        "fit_enabled": bool(args.fit),
        "fit_seconds": fit_seconds,
        "serve_seconds": serve_seconds,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "threshold_atr": threshold,
        "lifetime_bars": lifetime,
        "confirmed_pivots": int(event_counts["create"]),
        "identities_created": anchors,
        "high_identities": sum(
            level["birth_side"] == "high_pivot" for level in state["levels"]
        ),
        "low_identities": sum(
            level["birth_side"] == "low_pivot" for level in state["levels"]
        ),
        "merge_count": 0,
        "rows_per_anchor": rows_per_anchor,
        "center_displacement_max_price": max(identity_displacements, default=0.0),
        "center_displacement_max_atr": 0.0,
        "event_counts": dict(sorted(event_counts.items())),
        "active_eligible": _percentiles(active_rows),
        "pending_eligible": _percentiles(pending_rows),
        "carry_serialized_bytes": carry_bytes,
        "carry_serialized_bytes_per_anchor": (
            float(carry_bytes) / float(anchors) if anchors else 0.0
        ),
        "full_m1_carry_estimate_bytes": int(
            carry_bytes / float(args.rows) * full_m1_rows
        ),
        "finite_suffix_rows": int(finite.sum()),
        "feature_count": len(LEVEL_REGISTRY_M5_FEATURE_NAMES),
        "nonzero_feature_count": len(nonzero),
        "nonconstant_feature_count": len(nonconstant),
        "zero_only_features": sorted(set(names) - set(nonzero)),
        "constant_features": sorted(set(names) - set(nonconstant)),
    }
    if population is not None:
        result["fit_recurrence_population_sha256"] = population[
            "final_fit_recurrence_population_sha256"
        ]
        result["serve_recurrence_population_sha256"] = population[
            "final_serve_recurrence_population_sha256"
        ]
        result["canonical_outcome_population_sha256"] = population[
            "canonical_fit_outcome_population_sha256"
        ]
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
