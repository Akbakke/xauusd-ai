#!/usr/bin/env python3
"""Build online-IQL replay buffer from live paper-runner journals.

Reads v12_paper_journal_*.jsonl files in a date range, extracts each event's
captured `entry_iql_state_v1` (192-dim raw feature vector) and computes the
counterfactual reward per [SKIP, LONG, SHORT] × K-horizon by replaying the
forward M1 window.

Output: a single parquet bundle the warm-start trainer consumes directly.

Designed-not-forked:
  - State extraction reuses the per-poll dump we added to v12_pipeline.py
    (entry_iql_state_v1 + entry_iql_q_per_action_per_k_v1).
  - Forward-outcome reuses compute_forward_outcome() in
    v12_counterfactual_replay.py (extended 2026-05-31 with MAE-before-MFE).
  - Reward formula matches materialize_build_entry_iql_v2 line 489+ exactly so
    a warm-start update is consistent with cement training.

Usage:
  python -m gx1.scripts.build_online_replay_buffer \
      --from 20260601 --to 20260607 \
      --variant R_WAIT_OPP_K96_LAM50 \
      --out /home/andre2/GX1_DATA/reports/online_replay/replay_W2026_23.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gx1.execution.v12_counterfactual_replay import (
    compute_forward_outcome,
    load_m1_window,
)

JOURNAL_DIR = Path("/home/andre2/GX1_DATA/reports/v12_paper_runs")
K_HORIZONS_DEFAULT = (24, 48, 96, 144, 240)

# Reward configs mirror materialize_build_entry_iql_v2.py:489+ exactly.
WAIT_OPP_LAMBDA = {
    "R_WAIT_OPP_K96_LAM05": (96, 0.5),
    "R_WAIT_OPP_K96_LAM10": (96, 1.0),
    "R_WAIT_OPP_K96_LAM20": (96, 2.0),
    "R_WAIT_OPP_K96_LAM30": (96, 3.0),
    "R_WAIT_OPP_K96_LAM50": (96, 5.0),
    "R_WAIT_OPP_K48_LAM05": (48, 0.5),
    "R_WAIT_OPP_K48_LAM10": (48, 1.0),
}

logging.basicConfig(level=logging.INFO, format="[replay_buffer] %(message)s")
LOG = logging.getLogger("build_online_replay_buffer")


def iter_journals(date_from: str, date_to: str, suffix: str) -> list[Path]:
    glob = f"v12_paper_journal_*_{suffix}.jsonl"
    out = []
    for p in sorted(JOURNAL_DIR.glob(glob)):
        # extract YYYYMMDD from filename
        try:
            day = p.name.split("_")[3]
            if date_from <= day <= date_to:
                out.append(p)
        except IndexError:
            continue
    return out


def compute_rewards_wait_opp(cf: dict, k: int, lam: float, entry_spread_bps: float) -> tuple[float, float, float]:
    """Replicate R_WAIT_OPP reward formula from materialize_build_entry_iql_v2.

    Returns (r_skip, r_long, r_short) all in bps space, clipped to [-500, 500].
    """
    tl = cf.get(f"long_terminal_K{k}", 0.0)
    ts = cf.get(f"short_terminal_K{k}", 0.0)
    ml = max(0.0, cf.get(f"long_mae_before_mfe_K{k}", 0.0))
    ms = max(0.0, cf.get(f"short_mae_before_mfe_K{k}", 0.0))

    r_long = tl - lam * ml - 2.0 * entry_spread_bps
    r_short = ts - lam * ms - 2.0 * entry_spread_bps

    # SKIP credit when waiting (not taking) would have been net positive
    # vs the best take side. Counterfactual "wait" = take at a later (next M5)
    # bar — we approximate via the same K-window with no cost. This is the
    # online-buffer approximation; cement parquet has the explicit wait_pnl
    # but live journals don't, so we use 0 as wait baseline.
    best_take = max(r_long, r_short)
    r_skip = max(0.0, -best_take)  # gain by not taking when take is negative

    r_long = float(np.clip(r_long, -500, 500))
    r_short = float(np.clip(r_short, -500, 500))
    r_skip = float(np.clip(r_skip, 0, 500))
    return r_skip, r_long, r_short


def build_buffer(
    journals: list[Path],
    m1: pd.DataFrame,
    variant: str,
    k_horizons: tuple[int, ...],
) -> pd.DataFrame:
    if variant not in WAIT_OPP_LAMBDA:
        raise ValueError(
            f"variant {variant!r} not supported by online buffer yet. "
            f"Supported: {sorted(WAIT_OPP_LAMBDA)}"
        )
    k_reward, lam = WAIT_OPP_LAMBDA[variant]
    if k_reward not in k_horizons:
        raise ValueError(
            f"variant {variant} needs K={k_reward} in --k-horizons; got {k_horizons}"
        )

    # Pre-build M1 index for fast lookup of entry_idx from decision_ts
    m1_ts_index = pd.Index(pd.to_datetime(m1["time"], utc=True))

    rows = []
    seen_ts = set()
    skipped = {"no_state": 0, "no_decision": 0, "ts_not_found": 0, "fwd_window_short": 0, "duplicate": 0}

    for jpath in journals:
        with jpath.open() as fh:
            for line in fh:
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                dec = ev.get("v12_decision")
                if not isinstance(dec, dict):
                    skipped["no_decision"] += 1
                    continue
                state = dec.get("entry_iql_state_v1")
                if not isinstance(state, list) or not state:
                    skipped["no_state"] += 1
                    continue
                decision_ts = dec.get("decision_ts") or ev.get("ts_utc")
                if not decision_ts:
                    skipped["ts_not_found"] += 1
                    continue
                ts = pd.Timestamp(decision_ts, tz="UTC")
                if ts in seen_ts:
                    skipped["duplicate"] += 1
                    continue
                seen_ts.add(ts)

                pos = m1_ts_index.searchsorted(ts)
                if pos >= len(m1_ts_index) or pos + max(k_horizons) >= len(m1):
                    skipped["fwd_window_short"] += 1
                    continue
                entry_idx = int(pos)

                cf = compute_forward_outcome(m1, entry_idx, list(k_horizons))
                if not cf or f"long_terminal_K{k_reward}" not in cf:
                    skipped["fwd_window_short"] += 1
                    continue

                spread_bps = float(ev.get("spread_bps", 0.0) or 0.0)
                r_skip, r_long, r_short = compute_rewards_wait_opp(cf, k_reward, lam, spread_bps)

                row = {
                    "ts_utc": ts.isoformat(),
                    "entry_idx": entry_idx,
                    "spread_bps_at_decision": spread_bps,
                    "live_action": dec.get("action", "UNKNOWN"),
                    "live_advantage_over_skip": float(dec.get("advantage_over_skip", 0.0) or 0.0),
                }
                # State vector
                for i, v in enumerate(state):
                    row[f"s{i:03d}"] = float(v)
                # Per-K rewards (only the reward-K is used for training;
                # others kept for variant-mixing experiments later)
                for k in k_horizons:
                    row[f"r_skip_K{k}"] = (
                        compute_rewards_wait_opp(cf, k, lam, spread_bps)[0]
                        if f"long_terminal_K{k}" in cf else 0.0
                    )
                    row[f"r_long_K{k}"] = (
                        compute_rewards_wait_opp(cf, k, lam, spread_bps)[1]
                        if f"long_terminal_K{k}" in cf else 0.0
                    )
                    row[f"r_short_K{k}"] = (
                        compute_rewards_wait_opp(cf, k, lam, spread_bps)[2]
                        if f"long_terminal_K{k}" in cf else 0.0
                    )
                rows.append(row)

    LOG.info(f"buffer rows: {len(rows)}  skipped: {skipped}")
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="date_from", type=str, required=True,
                   help="Start date YYYYMMDD (inclusive)")
    p.add_argument("--to", dest="date_to", type=str, required=True,
                   help="End date YYYYMMDD (inclusive)")
    p.add_argument("--suffix", type=str, default="costfix_pure_phase6",
                   help="Journal filename suffix (after the date)")
    p.add_argument("--variant", type=str, default="R_WAIT_OPP_K96_LAM50",
                   choices=sorted(WAIT_OPP_LAMBDA),
                   help="Reward variant — must match cement variant for warm-start")
    p.add_argument("--k-horizons", type=str, default=",".join(str(k) for k in K_HORIZONS_DEFAULT))
    p.add_argument("--out", type=Path, required=True,
                   help="Output parquet path")
    args = p.parse_args()

    k_horizons = tuple(int(k) for k in args.k_horizons.split(","))

    journals = iter_journals(args.date_from, args.date_to, args.suffix)
    if not journals:
        LOG.error(f"No journals matched suffix={args.suffix} between {args.date_from} and {args.date_to}")
        return 2
    LOG.info(f"found {len(journals)} journal(s): {[j.name for j in journals]}")

    # Load M1 window covering the journal date-range + a tail buffer for the
    # longest K-horizon (so the last poll's forward window doesn't truncate).
    start_ts = pd.Timestamp(args.date_from, tz="UTC")
    end_ts = pd.Timestamp(args.date_to, tz="UTC") + pd.Timedelta(days=1) + pd.Timedelta(minutes=max(k_horizons) + 60)
    LOG.info(f"loading M1 window: {start_ts} → {end_ts}")
    m1 = load_m1_window(start_ts, end_ts)
    if m1.empty:
        LOG.error("M1 window empty — check that collector parquets + M1 tape cover the date range")
        return 4
    LOG.info(f"M1 bars loaded: {len(m1):,}  first={m1['time'].iloc[0]}  last={m1['time'].iloc[-1]}")

    df = build_buffer(journals, m1, args.variant, k_horizons)
    if df.empty:
        LOG.error("buffer empty — nothing to write. Check that journals contain entry_iql_state_v1.")
        return 3

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "variant": args.variant,
        "k_horizons": list(k_horizons),
        "date_from": args.date_from,
        "date_to": args.date_to,
        "n_rows": int(len(df)),
        "state_dim": sum(1 for c in df.columns if c.startswith("s") and c[1:].isdigit()),
        "journals": [j.name for j in journals],
    }, indent=2))
    LOG.info(f"wrote {len(df):,} rows → {args.out}")
    LOG.info(f"wrote metadata → {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
