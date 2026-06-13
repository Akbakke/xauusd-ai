"""ONE-TRUTH generator for the frozen regime bucket edges.

Writes gx1.execution.v12_ctx_augment_live.REGIME_BUCKET_EDGES_PATH = the full-history quantile
edges (q20/40/60/80) of atr_bps (and spread_bps when non-degenerate), computed from the cemented /
cutover canonical_v3 prebuilt via the SAME _add_spread_atr_bps one-truth formula the live augment
uses. Digitizing against these fixed edges makes vol_regime_id / atr_bucket FRAME-INVARIANT — the
daemon (any rolling window), the entry serve (augment_canonical_v3), the exit (base34), and the
build all produce the IDENTICAL bucket = training.

RUN at every cement / cutover / canonical_v3 rebuild (the edges are a snapshot of the training-time
volatility distribution; the rule-9 KS-drift leg flags when that distribution has drifted enough to
warrant a retrain VEDTAK + fresh edges — rule 3, never auto).

Birthed 2026-06-13 audit: the daemon's 420d-window percentile rank disagreed with full-history
training on ~31% of appends (a wrong-by-one vol bucket the entry+exit IQL one-hots act on).
No existing file owned "persist the regime bucket edges", so this is a new one-truth helper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gx1.execution.v12_ctx_augment_live import _add_spread_atr_bps, REGIME_BUCKET_EDGES_PATH

DEFAULT_CV3 = (
    "/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_PREBUILT/"
    "xauusd_m5_CANONICAL_V3_2020_2026.parquet"
)


def _edges(values: np.ndarray) -> list | None:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0 or float(np.std(v)) < 1e-9:
        return None  # degenerate/constant (e.g. spread_bps on XAU) — rank/digitize both invariant
    return [round(float(q), 6) for q in np.quantile(v, [0.2, 0.4, 0.6, 0.8])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cv3", default=DEFAULT_CV3, help="full-history canonical_v3 prebuilt")
    ap.add_argument("--out", default=REGIME_BUCKET_EDGES_PATH)
    args = ap.parse_args()

    df = pd.read_parquet(args.cv3)
    if "time" in df.columns:
        df = df.set_index(pd.to_datetime(df["time"], utc=True)).sort_index()
    _add_spread_atr_bps(df)  # one-truth atr_bps / spread_bps (same formula the live augment uses)

    out = {
        "atr_bps_edges": _edges(df["atr_bps"].to_numpy(dtype=float)),
        "spread_bps_edges": _edges(df["spread_bps"].to_numpy(dtype=float)),
        "method": "full_history_quantile_q20_40_60_80_digitize_right_False_clip_0_4",
        "source_cv3": str(args.cv3),
        "source_sha256": hashlib.sha256(Path(args.cv3).read_bytes()).hexdigest(),
        "source_rows": int(len(df)),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}\n{json.dumps(out, indent=2)}")


if __name__ == "__main__":
    main()
