#!/usr/bin/env python3
"""Variant + threshold calibration analyzer.

For each (variant, threshold) combination, simulate the policy:
  TAKE if Q_take - Q_skip > threshold, else SKIP.

Then compute realized PnL by joining with actual M1 price-action from the live
collector. Outputs a Pareto frontier: take-rate vs mean-bps-per-take.

This is OFFLINE analysis. Reuses cement bundle Q-nets + variant infrastructure
from v12_counterfactual_replay. No live runtime touched.

Usage:
  python -m gx1.scripts.analyze_variant_threshold_calibration \
      --journal-date 20260601 --suffix costfix_pure_phase6 \
      --variants R_WAIT_OPP_K96_LAM50,R_WAIT_OPP_K96_LAM30,R_WAIT_OPP_K96_LAM20,R_WAIT_OPP_K96_LAM10,R_HYBRID_K96_TOL20 \
      --thresholds -200,-150,-100,-50,-25,0 \
      --out /tmp/variant_calibration_20260601.json
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/andre2/src/GX1_ENGINE")
from gx1.execution.v12_counterfactual_replay import (
    compute_forward_outcome, load_m1_window,
)
from gx1.runtime.entry_iql_v2_adapter import EntryIQLV2Adapter
from gx1_guards.artifacts import load_decision_artifact

JOURNAL_DIR = Path("/home/andre2/GX1_DATA/reports/v12_paper_runs")
K_HORIZONS = (24, 48, 96, 144, 240)


def load_journal_with_state(journal_path: Path) -> list[dict]:
    out = []
    with open(journal_path) as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            d = e.get("v12_decision", {})
            if not isinstance(d, dict) or "entry_iql_state_v1" not in d:
                continue
            state = d["entry_iql_state_v1"]
            if not isinstance(state, list) or not state:
                continue
            decision_ts = d.get("decision_ts") or e.get("ts_utc")
            out.append({
                "ts": pd.Timestamp(decision_ts, tz="UTC"),
                "state_v1": np.array(state, dtype=np.float32),
                "spread_bps": float(e.get("spread_bps", 0.0) or 0.0),
                "cement_q_skip": float(d.get("q_skip", 0.0) or 0.0),
                "cement_q_long": float(d.get("q_take_long", 0.0) or 0.0),
                "cement_q_short": float(d.get("q_take_short", 0.0) or 0.0),
                "cement_action": d.get("action", "?"),
            })
    return out


def variant_q_inference(events: list[dict], bundle_dir: Path,
                        variants: list[str], fold: str = "FOLD_1",
                        aggregator: str = "mean") -> dict[str, np.ndarray]:
    """For each variant, run Q-net on every event's state. Returns
    {variant: array[n_events, 3]} (cols: q_skip, q_long, q_short).
    """
    out = {}
    for variant in variants:
        print(f"  loading {variant}/{fold}...")
        try:
            adapter = EntryIQLV2Adapter.load(
                artifact_root=bundle_dir,
                variant=variant, fold_id=fold,
                aggregator=aggregator, beta=0.0,
                prefer_cuda=False,
            )
        except Exception as exc:
            print(f"    SKIP — load failed: {exc}")
            continue
        n_events = len(events)
        Q = np.zeros((n_events, 3), dtype=np.float32)
        means = adapter.model.feature_means
        stds = adapter.model.feature_stds
        # Stack states + normalize
        states = np.stack([e["state_v1"] for e in events])  # (n, dim)
        if states.shape[1] != means.shape[0]:
            print(f"    SKIP — state_dim mismatch: events={states.shape[1]} variant={means.shape[0]}")
            continue
        normed = (states - means) / np.maximum(stds, 1e-6)
        normed = np.clip(normed, -5.0, 5.0)
        normed = np.nan_to_num(normed, 0.0)
        # Forward pass through Q-net
        import torch
        device = adapter.model.device
        with torch.no_grad():
            X = torch.from_numpy(normed).to(device)
            q_full = adapter.model.q_net(X).view(n_events, adapter.model.n_actions, adapter.model.n_k)
            # Aggregate over K-horizons per aggregator
            if aggregator == "mean":
                q_agg = q_full.mean(dim=-1).cpu().numpy()
            elif aggregator == "max":
                q_agg = q_full.max(dim=-1).values.cpu().numpy()
            else:
                q_agg = q_full.mean(dim=-1).cpu().numpy()
        Q[:, 0] = q_agg[:, 0]  # SKIP
        Q[:, 1] = q_agg[:, 1]  # LONG
        Q[:, 2] = q_agg[:, 2]  # SHORT
        out[variant] = Q
        print(f"    OK — Q shape {Q.shape}")
    return out


def compute_realized_outcomes(events: list[dict], m1: pd.DataFrame,
                              k_horizons: tuple[int, ...]) -> list[dict]:
    """For each event, compute realized forward-outcome (long_terminal,
    short_terminal, mae_before_mfe per side per K)."""
    m1_ts = pd.Index(pd.to_datetime(m1["time"], utc=True))
    out = []
    for ev in events:
        ts = ev["ts"]
        pos = m1_ts.searchsorted(ts)
        if pos >= len(m1_ts) - max(k_horizons):
            out.append(None)
            continue
        cf = compute_forward_outcome(m1, int(pos), list(k_horizons))
        out.append(cf if cf else None)
    return out


def simulate_policy(Q: np.ndarray, threshold: float) -> np.ndarray:
    """Return action per event given Q-matrix and threshold.
    Action: 0=SKIP, 1=LONG, 2=SHORT.
    TAKE if max(Q_long, Q_short) - Q_skip > threshold."""
    q_skip = Q[:, 0]
    q_long = Q[:, 1]
    q_short = Q[:, 2]
    adv_long = q_long - q_skip
    adv_short = q_short - q_skip
    adv_max = np.maximum(adv_long, adv_short)
    actions = np.zeros(len(Q), dtype=np.int32)  # SKIP default
    take_mask = adv_max > threshold
    long_better = adv_long >= adv_short
    actions[take_mask & long_better] = 1
    actions[take_mask & ~long_better] = 2
    return actions


def realized_pnl(actions: np.ndarray, outcomes: list[dict],
                 spreads: np.ndarray, k_horizon: int = 96) -> dict:
    """Given actions + per-event outcomes, compute aggregate stats."""
    n_take = 0
    total_bps = 0.0
    bps_list = []
    n_wins = 0
    n_loss = 0
    for i, action in enumerate(actions):
        if action == 0:
            continue  # SKIP
        cf = outcomes[i]
        if cf is None:
            continue  # no fwd-outcome available
        if action == 1:
            pnl = cf.get(f"long_terminal_K{k_horizon}", 0.0) - 2.0 * spreads[i]
        else:  # SHORT
            pnl = cf.get(f"short_terminal_K{k_horizon}", 0.0) - 2.0 * spreads[i]
        n_take += 1
        total_bps += pnl
        bps_list.append(pnl)
        if pnl > 0: n_wins += 1
        else: n_loss += 1
    return {
        "n_take": n_take,
        "total_bps": total_bps,
        "mean_bps_per_take": (total_bps / n_take) if n_take > 0 else 0.0,
        "win_rate": (n_wins / n_take) if n_take > 0 else 0.0,
        "median_bps": float(np.median(bps_list)) if bps_list else 0.0,
        "p10_bps": float(np.percentile(bps_list, 10)) if bps_list else 0.0,
        "p90_bps": float(np.percentile(bps_list, 90)) if bps_list else 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--journal-date", required=True)
    p.add_argument("--suffix", default="costfix_pure_phase6")
    p.add_argument("--variants", required=True, help="comma-sep variant list")
    p.add_argument("--thresholds", default="-200,-150,-100,-50,-25,0,50")
    p.add_argument("--fold", default="FOLD_1")
    p.add_argument("--aggregator", default="mean")
    p.add_argument("--k-horizon", type=int, default=96, help="K for realized PnL")
    p.add_argument("--bundle-dir", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    variants = [v.strip() for v in args.variants.split(",")]
    thresholds = [float(t) for t in args.thresholds.split(",")]

    journal_path = JOURNAL_DIR / f"v12_paper_journal_{args.journal_date}_{args.suffix}.jsonl"
    print(f"=== Loading journal: {journal_path.name} ===")
    events = load_journal_with_state(journal_path)
    print(f"  events with state_v1: {len(events)}")
    if not events:
        print("FATAL: no replayable events. Was journal pre-patch (before 2026-05-31)?")
        return 1

    bundle_dir = Path(args.bundle_dir) if args.bundle_dir else Path(load_decision_artifact("entry_iql"))
    print(f"=== Bundle: {bundle_dir.name} ===")

    print(f"\n=== Loading M1 window ===")
    start_ts = events[0]["ts"] - pd.Timedelta(hours=1)
    end_ts = events[-1]["ts"] + pd.Timedelta(hours=24)
    m1 = load_m1_window(start_ts, end_ts)
    print(f"  M1 bars: {len(m1):,}")

    print(f"\n=== Computing realized forward-outcomes ===")
    outcomes = compute_realized_outcomes(events, m1, K_HORIZONS)
    n_with_outcome = sum(1 for o in outcomes if o is not None)
    print(f"  events with fwd-outcome: {n_with_outcome}/{len(events)}")

    print(f"\n=== Running variant Q-inference ===")
    Q_per_variant = variant_q_inference(events, bundle_dir, variants,
                                         fold=args.fold, aggregator=args.aggregator)

    spreads = np.array([e["spread_bps"] for e in events], dtype=np.float32)

    print(f"\n=== Threshold grid-search ===")
    results = []
    for variant in variants:
        if variant not in Q_per_variant:
            continue
        Q = Q_per_variant[variant]
        for theta in thresholds:
            actions = simulate_policy(Q, theta)
            stats = realized_pnl(actions, outcomes, spreads, args.k_horizon)
            take_rate = stats["n_take"] / len(actions)
            results.append({
                "variant": variant,
                "threshold": theta,
                "n_events": len(actions),
                "take_rate": take_rate,
                **stats,
            })
            print(f"  {variant} θ={theta:+5.0f}: take_rate={take_rate*100:5.1f}%  "
                  f"n_take={stats['n_take']:4d}  mean_bps={stats['mean_bps_per_take']:+7.1f}  "
                  f"win_rate={stats['win_rate']*100:4.1f}%  "
                  f"total={stats['total_bps']:+8.0f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "journal_date": args.journal_date,
        "k_horizon": args.k_horizon,
        "n_events": len(events),
        "n_with_outcome": n_with_outcome,
        "variants": variants,
        "thresholds": thresholds,
        "results": results,
    }, indent=2, default=float))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
