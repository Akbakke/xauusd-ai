"""V12 Phase 4: Add V3-tracking features to V12_PER_BAR_V3_AUGMENTED dataset.

Per-bar features computed from `v3_v8_should_exit_prob` (added in Phase 3 by
score_v3_v8_on_per_bar_v1.py). Per V12 design (handover protocol §5):

  v3_should_exit_decision  = (v3_v8_should_exit_prob > 0.5)
  v3_decision_confidence   = abs(v3_v8_should_exit_prob - 0.5)
  v3_max_prob_in_trade     = running max v3_v8_should_exit_prob since entry
  v3_consecutive_exits     = consecutive bars with prob > 0.5 ending at this bar
  v3_signal_acceleration   = Δ v3_v8_should_exit_prob bar-to-bar

Also adds these summary features (constant per candidate, useful as snapshots):
  v3_total_exit_decisions  = count of bars with prob > 0.5 in trade
  v3_max_consecutive_exits = max run of consecutive prob>0.5 anywhere in trade

`v3_iql_disagreement_so_far` is NOT computed here — it depends on IQL's
prior decisions during training rollout (per V12 design, computed in trainer).

Run:
    PYTHONPATH=/home/andre2/src/GX1_ENGINE python3 \\
        /tmp/v12_phase4_v3_tracking_features.py \\
        --in-dir <V12_PER_BAR_V3_AUGMENTED>/per_week \\
        --out-dir <V12_PER_BAR_V3_TRACKED>/per_week \\
        [--week WEEK] [--max-weeks N]
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

V3_PROB_COL = "v3_v8_should_exit_prob"
EXIT_THRESHOLD = 0.5
GROUP_KEY = "candidate_uid"
SORT_KEY = "bar_idx_v1"


def add_v3_tracking_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add V3-tracking features per V12 design. Mutates+returns df."""
    if V3_PROB_COL not in df.columns:
        raise ValueError(f"Missing column {V3_PROB_COL!r} — run Phase 3 (V3 v8 augment) first")

    # Sort once globally so groupby preserves order (groupby preserves original order
    # within each group when sort=False).
    df = df.sort_values([GROUP_KEY, SORT_KEY], kind="stable").reset_index(drop=True)

    prob = df[V3_PROB_COL].astype(np.float32).fillna(0.5).to_numpy()
    decision = (prob > EXIT_THRESHOLD)
    confidence = np.abs(prob - EXIT_THRESHOLD).astype(np.float32)

    # Running max per candidate
    df["v3_should_exit_decision_v1"] = decision
    df["v3_decision_confidence_v1"] = confidence
    df["v3_max_prob_in_trade_v1"] = (
        df.groupby(GROUP_KEY, sort=False)[V3_PROB_COL].cummax().astype(np.float32)
    )

    # Consecutive exits (resets on prob<=0.5).
    # Trick: per-group running count where consecutive True increments, False resets.
    # Use cumsum of "reset markers" trick.
    g = df.groupby(GROUP_KEY, sort=False)
    is_exit = decision.astype(np.int32)
    # group_id integer encoding to vectorize across groups
    grp_ids = g.ngroup().to_numpy()
    consec = np.zeros(len(df), dtype=np.int32)
    last_grp = -1
    run = 0
    for i in range(len(df)):
        if grp_ids[i] != last_grp:
            run = 0
            last_grp = grp_ids[i]
        run = run + 1 if is_exit[i] else 0
        consec[i] = run
    df["v3_consecutive_exits_v1"] = consec

    # Acceleration = Δ prob bar-to-bar within group; reset at group start = 0
    delta = np.zeros(len(df), dtype=np.float32)
    last_grp = -1
    last_prob = 0.0
    for i in range(len(df)):
        if grp_ids[i] != last_grp:
            last_grp = grp_ids[i]
            last_prob = prob[i]
            delta[i] = 0.0
        else:
            delta[i] = prob[i] - last_prob
            last_prob = prob[i]
    df["v3_signal_acceleration_v1"] = delta

    # Summary features (constant per candidate, broadcast back to bars)
    df["v3_total_exit_decisions_v1"] = (
        df.groupby(GROUP_KEY, sort=False)["v3_should_exit_decision_v1"]
          .transform(lambda s: int(s.sum()))
    ).astype(np.int32)
    # max consecutive within trade — group max of running consec
    df["v3_max_consecutive_exits_v1"] = (
        df.groupby(GROUP_KEY, sort=False)["v3_consecutive_exits_v1"]
          .transform("max")
          .astype(np.int32)
    )

    return df


def process_week(in_path: Path, out_path: Path) -> dict:
    df = pd.read_parquet(in_path)
    n_in = len(df)
    if n_in == 0 or V3_PROB_COL not in df.columns:
        df.to_parquet(out_path, index=False)
        return {
            "input": str(in_path), "output": str(out_path),
            "rows": n_in, "status": "EMPTY_OR_NO_V3_COL",
        }
    df = add_v3_tracking_features(df)
    df.to_parquet(out_path, index=False)
    n_decisions = int(df["v3_should_exit_decision_v1"].sum())
    return {
        "input": str(in_path), "output": str(out_path),
        "rows": n_in,
        "n_v3_exit_decisions": n_decisions,
        "exit_decision_frac": float(n_decisions / max(1, n_in)),
        "status": "OK",
    }


def main() -> int:
    p = argparse.ArgumentParser(description="V12 Phase 4: V3-tracking features")
    p.add_argument("--in-dir", required=True, help="dir with V12_PER_BAR_V3_AUGMENTED per-week parquets")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--week", action="append", default=None, help="week stem (repeatable)")
    p.add_argument("--max-weeks", type=int, default=None)
    p.add_argument("--no-skip-existing", action="store_true")
    args = p.parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.parquet"))
    files = [f for f in files if not f.stem.endswith("_stats")]
    if args.week:
        only = set(args.week)
        files = [f for f in files if any(w in f.stem for w in only)]
    if args.max_weeks:
        files = files[: args.max_weeks]

    print(f"V12 PHASE 4 — V3-tracking features")
    print(f"  in:  {in_dir}")
    print(f"  out: {out_dir}")
    print(f"  weeks: {len(files)}")

    stats = []
    t0 = time.time()
    for i, fp in enumerate(files, start=1):
        out_path = out_dir / fp.name
        if not args.no_skip_existing and out_path.exists():
            stats.append({"input": str(fp), "output": str(out_path), "status": "SKIPPED_EXISTING"})
            continue
        s = process_week(fp, out_path)
        stats.append(s)
        if i <= 3 or i % 25 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] {fp.stem}: rows={s.get('rows', 0):,} "
                  f"exits={s.get('n_v3_exit_decisions', 0):,} "
                  f"frac={s.get('exit_decision_frac', 0):.3f} "
                  f"status={s['status']}  elapsed={time.time()-t0:.0f}s")

    summary = {
        "action": "V12_PHASE4_V3_TRACKING_FEATURES",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "n_weeks": len(files),
        "n_ok": sum(1 for s in stats if s["status"] == "OK"),
        "elapsed_sec": time.time() - t0,
    }
    (out_dir.parent / "phase4_summary.json").write_text(
        json.dumps({**summary, "per_week_stats": stats}, indent=2, default=str)
    )
    print(f"\n✅ Done. weeks_ok={summary['n_ok']}/{summary['n_weeks']} "
          f"elapsed={summary['elapsed_sec']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
