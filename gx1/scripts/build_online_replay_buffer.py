#!/usr/bin/env python3
"""Build online-IQL replay buffer from live paper-runner journals.

Reads v12_paper_journal_*.jsonl files in a date range, extracts each event's
captured `entry_iql_state_v1` (197-dim raw feature vector) and computes the
counterfactual reward per [SKIP, LONG, SHORT] × K-horizon by replaying the
forward M1 window.

Output: a single parquet bundle the warm-start trainer consumes directly.

Designed-not-forked:
  - State extraction reuses the per-poll dump we added to v12_pipeline.py
    (entry_iql_state_v1 + entry_iql_q_per_action_per_k_v1).
  - Forward-outcome reuses compute_forward_outcome() in
    v12_counterfactual_replay.py (extended 2026-05-31 with MAE-before-MFE).
  - Reward formula matches materialize_build_entry_iql_v2.build_reward_matrix
    exactly (incl. the _SYM family + bad-path posgate) so a warm-start update
    is consistent with cement training.

UNITS (2026-06-12 fix — this file had a latent 5x horizon-parity bug):
  Cement K-horizons are in M5 BARS ([12,24,48,96,144,192]; K96 = 8h). The live
  M1 tape is per-minute, so every cement K converts to K*5 M1 bars before
  windowing. The original scaffold passed cement K straight into the M1 window
  (K96 → 96 minutes ≠ 8h) — a refit trained on that would have silently
  optimized 5x-shorter horizons than the bundle it warm-started.

PARITY with the ACTIVE bundle (entry_iql_volbal_20260611):
  - variant R_WAIT_OPP_K96_LAM50_SYM: true per-K rewards, wait-side penalized
    with ITS OWN MAE (like-for-like r_skip), spread_coef=0 (terminal PnL is
    already bid/ask round-trip).
  - GX1_REWARD_BADPATH_POSGATE defaults to "1" HERE (builder default is "0"):
    the volbal bundle was verified posgate-trained (reward-matrix mean
    fingerprint −54.98 ≈ logged −55.635; posgate=0 gives −17.6). Newer bundles
    stamp `reward_env_v1` in the ckpt — check it before refitting.
  - Truncation guard (GX1_REWARD_TRUNC_MASK analog): rows whose forward window
    (incl. the +30-M1-bar wait anchor) is short or crosses a >180-min gap are
    DROPPED — the cement builder masks truncated labels the same way.

Usage:
  python -m gx1.scripts.build_online_replay_buffer \
      --from 20260601 --to 20260607 \
      --variant R_WAIT_OPP_K96_LAM50_SYM \
      --out /home/andre2/GX1_DATA/reports/online_replay/replay_W2026_23.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gx1.execution.v12_counterfactual_replay import (
    compute_forward_outcome,
    load_m1_window,
)

JOURNAL_DIR = Path("/home/andre2/GX1_DATA/reports/v12_paper_runs")
# Cement K-horizons in M5 BARS — must match the ACTIVE bundle ckpt's k_horizons.
K_HORIZONS_M5_DEFAULT = (12, 24, 48, 96, 144, 192)
M1_PER_M5 = 5
WAIT_BARS_M5 = 6                     # cement WAIT anchor: +6 M5 bars
WAIT_BARS_M1 = WAIT_BARS_M5 * M1_PER_M5
MAX_INTRA_GAP_MINUTES = 180.0        # cement forward-outcome gap-truncation rule

# Reward configs mirror materialize_build_entry_iql_v2._CFG exactly.
# name → (K_primary_m5, lambda). The _SYM suffix is handled structurally
# (own-MAE wait penalty, spread_coef=0, true per-K) like the cement builder.
WAIT_OPP_LAMBDA = {
    "R_WAIT_OPP_K96_LAM025": (96, 0.25),
    "R_WAIT_OPP_K96_LAM05": (96, 0.5),
    "R_WAIT_OPP_K96_LAM10": (96, 1.0),
    "R_WAIT_OPP_K96_LAM20": (96, 2.0),
    "R_WAIT_OPP_K96_LAM30": (96, 3.0),
    "R_WAIT_OPP_K96_LAM50": (96, 5.0),
    "R_WAIT_OPP_K48_LAM05": (48, 0.5),
    "R_WAIT_OPP_K48_LAM10": (48, 1.0),
}
SYM_VARIANTS = {f"{base}_SYM": cfg for base, cfg in WAIT_OPP_LAMBDA.items()}
ALL_VARIANTS = {**WAIT_OPP_LAMBDA, **SYM_VARIANTS}

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


def compute_rewards_for_row(
    cf_now: dict,
    cf_wait: dict,
    k_m5: int,
    lam: float,
    sym: bool,
    entry_spread_bps: float,
    bad_path_prob: float,
    posgate: bool,
) -> tuple[float, float, float] | None:
    """Cement-parity reward at ONE cement K (M5 bars) from live M1 counterfactuals.

    Mirrors materialize_build_entry_iql_v2._wait_opp_at_K + _gate_take:
      r_side  = terminal − λ·mae_before_mfe − spread_coef·spread   (coef 0 if SYM)
      wait    = wait_terminal − λ·wait_mae (SYM; unpenalized for base family)
      r_skip  = clip(max(0, max(wait_l, wait_s) − max(r_l, r_s)), 0, 500)
      takes   = clip ±500, then bad-path gate (posgate: positive part only).
    Returns (r_skip, r_long, r_short) or None when the K-window is unavailable.
    """
    k_m1 = k_m5 * M1_PER_M5
    tl = cf_now.get(f"long_terminal_K{k_m1}")
    ts = cf_now.get(f"short_terminal_K{k_m1}")
    wtl = cf_wait.get(f"long_terminal_K{k_m1}")
    wts = cf_wait.get(f"short_terminal_K{k_m1}")
    if tl is None or ts is None or wtl is None or wts is None:
        return None
    ml = max(0.0, cf_now.get(f"long_mae_before_mfe_K{k_m1}", 0.0))
    ms = max(0.0, cf_now.get(f"short_mae_before_mfe_K{k_m1}", 0.0))

    spread_coef = 0.0 if sym else 2.0
    r_long = tl - lam * ml - spread_coef * entry_spread_bps
    r_short = ts - lam * ms - spread_coef * entry_spread_bps

    wl, ws = float(wtl), float(wts)
    if sym:
        wml = max(0.0, cf_wait.get(f"long_mae_before_mfe_K{k_m1}", 0.0))
        wms = max(0.0, cf_wait.get(f"short_mae_before_mfe_K{k_m1}", 0.0))
        wl = wl - lam * wml - spread_coef * entry_spread_bps
        ws = ws - lam * wms - spread_coef * entry_spread_bps

    r_skip = float(np.clip(max(0.0, max(wl, ws) - max(r_long, r_short)), 0.0, 500.0))

    gate = 1.0 - float(np.clip(bad_path_prob, 0.0, 1.0))

    def _gate_take(r: float) -> float:
        r = float(np.clip(r, -500.0, 500.0))
        if posgate:
            return r * gate if r > 0 else r
        return r * gate

    return r_skip, _gate_take(r_long), _gate_take(r_short)


def build_buffer(
    journals: list[Path],
    m1: pd.DataFrame,
    variant: str,
    k_horizons_m5: tuple[int, ...],
    posgate: bool,
) -> pd.DataFrame:
    if variant not in ALL_VARIANTS:
        raise ValueError(
            f"variant {variant!r} not supported by online buffer. "
            f"Supported: {sorted(ALL_VARIANTS)}"
        )
    k_primary_m5, lam = ALL_VARIANTS[variant]
    sym = variant.endswith("_SYM")
    if k_primary_m5 not in k_horizons_m5:
        raise ValueError(
            f"variant {variant} needs K={k_primary_m5} (M5) in --k-horizons; got {k_horizons_m5}"
        )
    k_m1_list = [k * M1_PER_M5 for k in k_horizons_m5]
    max_k_m1 = max(k_m1_list)

    # Pre-build M1 index + gap prefix-sum for the truncation guard. A window
    # [a, b] crosses a >180-min gap iff gap_cum[b] > gap_cum[a].
    m1_ts_index = pd.Index(pd.to_datetime(m1["time"], utc=True))
    diffs_min = np.diff(m1_ts_index.asi8) / 60e9
    gap_cum = np.concatenate([[0], np.cumsum(diffs_min > MAX_INTRA_GAP_MINUTES)])

    rows = []
    seen_ts = set()
    skipped = {"no_state": 0, "no_decision": 0, "ts_not_found": 0,
               "fwd_window_short": 0, "gap_truncated": 0, "duplicate": 0}

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

                pos = int(m1_ts_index.searchsorted(ts))
                end_needed = pos + WAIT_BARS_M1 + max_k_m1
                if pos >= len(m1) or end_needed >= len(m1):
                    skipped["fwd_window_short"] += 1
                    continue
                # GX1_REWARD_TRUNC_MASK analog: drop rows whose forward window
                # (incl. the wait anchor's window) crosses a >180-min gap —
                # bar-indexed windows would otherwise silently span the weekend.
                if gap_cum[end_needed] > gap_cum[pos]:
                    skipped["gap_truncated"] += 1
                    continue

                cf_now = compute_forward_outcome(m1, pos, k_m1_list)
                cf_wait = compute_forward_outcome(m1, pos + WAIT_BARS_M1, k_m1_list)
                if not cf_now or f"long_terminal_K{k_primary_m5 * M1_PER_M5}" not in cf_now:
                    skipped["fwd_window_short"] += 1
                    continue

                spread_bps = float(ev.get("spread_bps", 0.0) or 0.0)
                v10 = dec.get("_v10_snapshot") or {}
                bad_path_prob = float(v10.get("bad_path_prob", 0.0) or 0.0)

                adv_l = dec.get("advantage_over_skip_long")
                adv_s = dec.get("advantage_over_skip_short")
                adv_best = max(float(adv_l or 0.0), float(adv_s or 0.0))

                row = {
                    "ts_utc": ts.isoformat(),
                    "entry_idx": pos,
                    "spread_bps_at_decision": spread_bps,
                    "bad_path_prob": bad_path_prob,
                    "live_action": dec.get("action", "UNKNOWN"),
                    "live_advantage_over_skip": adv_best,
                }
                # State vector
                for i, v in enumerate(state):
                    row[f"s{i:03d}"] = float(v)
                # True per-K rewards — column K names stay in cement M5-bar units
                # (r_*_K96 == cement K96 == 8h) for trainer compatibility.
                ok = True
                for k_m5 in k_horizons_m5:
                    r = compute_rewards_for_row(
                        cf_now, cf_wait, k_m5, lam, sym,
                        spread_bps, bad_path_prob, posgate,
                    )
                    if r is None:
                        ok = False
                        break
                    row[f"r_skip_K{k_m5}"] = r[0]
                    row[f"r_long_K{k_m5}"] = r[1]
                    row[f"r_short_K{k_m5}"] = r[2]
                if not ok:
                    skipped["fwd_window_short"] += 1
                    continue
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
    p.add_argument("--suffix", type=str, default="conviction67sized_skipasia_pure_phase6",
                   help="Journal filename suffix (after the date)")
    p.add_argument("--variant", type=str, default="R_WAIT_OPP_K96_LAM50_SYM",
                   choices=sorted(ALL_VARIANTS),
                   help="Reward variant — must match cement variant for warm-start")
    p.add_argument("--k-horizons", type=str,
                   default=",".join(str(k) for k in K_HORIZONS_M5_DEFAULT),
                   help="Cement K-horizons in M5 BARS (must match the bundle ckpt)")
    p.add_argument("--out", type=Path, required=True,
                   help="Output parquet path")
    args = p.parse_args()

    k_horizons_m5 = tuple(int(k) for k in args.k_horizons.split(","))

    # Parity default: the ACTIVE volbal bundle is posgate-trained (see header).
    posgate = os.environ.get("GX1_REWARD_BADPATH_POSGATE", "1") == "1"
    LOG.info(f"reward gate: GX1_REWARD_BADPATH_POSGATE={'1' if posgate else '0'} "
             f"({'positive-part gate' if posgate else 'full multiplicative gate'})")

    journals = iter_journals(args.date_from, args.date_to, args.suffix)
    if not journals:
        LOG.error(f"No journals matched suffix={args.suffix} between {args.date_from} and {args.date_to}")
        return 2
    LOG.info(f"found {len(journals)} journal(s): {[j.name for j in journals]}")

    # Load M1 window covering the journal date-range + a tail buffer for the
    # longest K-horizon (in M1 minutes!) + the wait anchor.
    max_k_m1 = max(k_horizons_m5) * M1_PER_M5
    start_ts = pd.Timestamp(args.date_from, tz="UTC")
    end_ts = (pd.Timestamp(args.date_to, tz="UTC") + pd.Timedelta(days=1)
              + pd.Timedelta(minutes=max_k_m1 + WAIT_BARS_M1 + 60))
    LOG.info(f"loading M1 window: {start_ts} → {end_ts}")
    m1 = load_m1_window(start_ts, end_ts)
    if m1.empty:
        LOG.error("M1 window empty — check that collector parquets + M1 tape cover the date range")
        return 4
    LOG.info(f"M1 bars loaded: {len(m1):,}  first={m1['time'].iloc[0]}  last={m1['time'].iloc[-1]}")

    df = build_buffer(journals, m1, args.variant, k_horizons_m5, posgate)
    if df.empty:
        LOG.error("buffer empty — nothing to write. Check that journals contain entry_iql_state_v1.")
        return 3

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "variant": args.variant,
        "sym": args.variant.endswith("_SYM"),
        "k_horizons_m5": list(k_horizons_m5),
        "k_horizons_m1": [k * M1_PER_M5 for k in k_horizons_m5],
        "wait_bars_m1": WAIT_BARS_M1,
        "max_intra_gap_minutes": MAX_INTRA_GAP_MINUTES,
        "reward_env_v1": {
            "GX1_REWARD_BADPATH_POSGATE": "1" if posgate else "0",
            "GX1_REWARD_TRUNC_MASK": "1 (structural: short/gapped windows dropped)",
        },
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
