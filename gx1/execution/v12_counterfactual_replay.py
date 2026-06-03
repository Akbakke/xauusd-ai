#!/usr/bin/env python3
"""V12 counterfactual replay — daily what-if analysis on paper-runner journal.

For each candidate logged by v12_paper_runner.py (TAKE + SKIP + BLOCKED), look
up what the actual forward-outcome would have been at K_HORIZONS using collected
M1 data (or backfill via OANDA). Reports:

  - "Missed opportunities": SKIP/BLOCKED candidates where forward MFE > 50 bps
  - "Correct skips": SKIP candidates where forward outcome was loss
  - "False takes": TAKE trades with negative outcome
  - PnL distribution per decision-class

Output: daily counterfactual report + JSONL with per-candidate what-if metrics
        for offline analysis and future ML retraining feedback.

Run (once per day, after K=1440 bars / 24h have passed):
    PYTHONPATH=/home/andre2/src/GX1_ENGINE python3 \\
        gx1/execution/v12_counterfactual_replay.py \\
        --journal-date 20260511 [--journal-suffix shadow]

Reads: /home/andre2/GX1_DATA/reports/v12_paper_runs/v12_paper_journal_*.jsonl
M1 source: /home/andre2/GX1_DATA/reports/v12_live_data/xauusd_m1_*.parquet
           (collector output) + canonical M1 tape as fallback.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOG = logging.getLogger("v12_cf_replay")
PAPER_DIR = Path("/home/andre2/GX1_DATA/reports/v12_paper_runs")
LIVE_DATA_DIR = Path("/home/andre2/GX1_DATA/reports/v12_live_data")
M1_TAPE_DIR = Path("/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m1_bid_ask__CANONICAL")

K_HORIZONS_BPS = [12, 60, 240, 480, 1440]   # M1 minutes — match V12 trainer
HIGH_CONVICTION_BPS_THRESHOLD = 50.0


def load_m1_window(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """Concatenate M1 bars from live collector + canonical tape covering [start, end].

    Live collector parquets cover from when collector started; canonical M1 tape
    covers historical. Use both, dedupe by time.
    """
    parts = []
    # Live collector
    for fp in sorted(LIVE_DATA_DIR.glob("xauusd_m1_*.parquet")):
        df = pd.read_parquet(fp)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        sub = df[(df["time"] >= start_ts) & (df["time"] <= end_ts)]
        if len(sub) > 0:
            parts.append(sub)
    # Canonical M1 tape (year partitions)
    for year in range(start_ts.year, end_ts.year + 1):
        fp = M1_TAPE_DIR / f"year={year}" / "part-000.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        sub = df[(df["time"] >= start_ts) & (df["time"] <= end_ts)]
        if len(sub) > 0:
            parts.append(sub)
    if not parts:
        return pd.DataFrame()
    return (pd.concat(parts, ignore_index=True)
            .drop_duplicates(subset=["time"], keep="last")
            .sort_values("time")
            .reset_index(drop=True))


def compute_forward_outcome(m1: pd.DataFrame, entry_idx: int,
                             k_horizons: list[int]) -> dict[str, float]:
    """For a hypothetical trade at entry_idx, compute forward PnL stats per K-horizon.

    Returns max-MFE, terminal-PnL, and giveback per K for both LONG and SHORT.
    Spread-aware (long: ask_open entry, bid_close exit; short: reversed).
    """
    out: dict[str, float] = {}
    if entry_idx >= len(m1):
        return out
    entry_bar = m1.iloc[entry_idx]
    long_entry = float(entry_bar["ask_open"])
    short_entry = float(entry_bar["bid_open"])
    if long_entry <= 0 or short_entry <= 0:
        return out

    for K in k_horizons:
        end_idx = min(entry_idx + K, len(m1) - 1)
        if end_idx <= entry_idx:
            continue
        window = m1.iloc[entry_idx + 1: end_idx + 1]
        if len(window) == 0:
            continue
        # LONG: peak = max bid_high, terminal = bid_close at K
        long_peak = float(window["bid_high"].max())
        long_terminal = float(window["bid_close"].iloc[-1])
        long_max_mfe_bps = (long_peak - long_entry) / long_entry * 10000.0
        long_terminal_bps = (long_terminal - long_entry) / long_entry * 10000.0
        # SHORT: peak (favorable) = min ask_low (lower price), terminal = ask_close
        short_peak = float(window["ask_low"].min())
        short_terminal = float(window["ask_close"].iloc[-1])
        short_max_mfe_bps = (short_entry - short_peak) / short_entry * 10000.0
        short_terminal_bps = (short_entry - short_terminal) / short_entry * 10000.0

        # MAE-before-MFE (online-IQL prep 2026-05-31): the worst adverse move
        # that occurred BEFORE the bar with peak MFE. R_WAIT_OPP variant penalty:
        # this is the "motgang før medgang" the policy is trained to minimize.
        long_mfe_idx = int(window["bid_high"].values.argmax())
        long_pre_window = window.iloc[: long_mfe_idx + 1]
        long_mae_before_mfe_bps = max(0.0, (long_entry - float(long_pre_window["bid_low"].min())) / long_entry * 10000.0)
        short_mfe_idx = int(window["ask_low"].values.argmin())
        short_pre_window = window.iloc[: short_mfe_idx + 1]
        short_mae_before_mfe_bps = max(0.0, (float(short_pre_window["ask_high"].max()) - short_entry) / short_entry * 10000.0)

        out[f"long_max_mfe_K{K}"] = long_max_mfe_bps
        out[f"long_terminal_K{K}"] = long_terminal_bps
        out[f"short_max_mfe_K{K}"] = short_max_mfe_bps
        out[f"short_terminal_K{K}"] = short_terminal_bps
        out[f"long_mae_before_mfe_K{K}"] = long_mae_before_mfe_bps
        out[f"short_mae_before_mfe_K{K}"] = short_mae_before_mfe_bps

    return out


def replay_journal(journal_path: Path) -> tuple[list[dict], dict]:
    """Replay every event in journal; compute counterfactual + classify outcomes."""
    if not journal_path.exists():
        LOG.error(f"journal not found: {journal_path}")
        return [], {}

    events = []
    with journal_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    if not events:
        return [], {}

    # Determine M1 window we need
    ts_strs = [e.get("ts_utc") for e in events if e.get("ts_utc")]
    earliest = pd.to_datetime(min(ts_strs), utc=True)
    latest = pd.to_datetime(max(ts_strs), utc=True)
    fetch_end = latest + pd.Timedelta(minutes=max(K_HORIZONS_BPS) + 60)
    LOG.info(f"loading M1 window {earliest} → {fetch_end} for {len(events)} candidates")
    m1 = load_m1_window(earliest, fetch_end)
    if m1.empty:
        LOG.error("no M1 data available — backfill collector or wait 24h post-trades")
        return events, {"error": "no_m1_data"}
    m1_time_arr = m1["time"].values  # numpy datetime64 for fast searchsorted

    enriched = []
    for ev in events:
        ts_str = ev.get("ts_utc")
        if not ts_str:
            enriched.append(ev)
            continue
        ts = pd.to_datetime(ts_str, utc=True)
        idx = int(np.searchsorted(m1_time_arr, ts.value, side="left"))
        if idx >= len(m1):
            enriched.append(ev)
            continue
        cf = compute_forward_outcome(m1, idx, K_HORIZONS_BPS)
        ev["counterfactual"] = cf
        # Classify: best_what_if_pnl, missed_opportunity flag (direction-aware)
        if cf:
            best_long_mfe = max(cf.get(f"long_max_mfe_K{K}", -9999) for K in K_HORIZONS_BPS)
            best_short_mfe = max(cf.get(f"short_max_mfe_K{K}", -9999) for K in K_HORIZONS_BPS)
            best_what_if = max(best_long_mfe, best_short_mfe)
            ev["best_what_if_pnl_bps"] = best_what_if
            ev["best_what_if_side"] = "long" if best_long_mfe >= best_short_mfe else "short"
            decision = ev.get("v12_decision", {}).get("action", "UNKNOWN")
            order_status = ev.get("order_status", "UNKNOWN")
            # Direction-aware proposed-side MFE:
            #   TAKE_LONG_NOW gated by spread → use long_max_mfe (V12 wanted long)
            #   TAKE_SHORT_NOW gated → use short_max_mfe
            #   SKIP → use best of either side (V12 had no proposed direction)
            if decision == "TAKE_LONG_NOW":
                proposed_side_mfe = best_long_mfe
                proposed_side = "long"
            elif decision == "TAKE_SHORT_NOW":
                proposed_side_mfe = best_short_mfe
                proposed_side = "short"
            else:
                proposed_side_mfe = best_what_if
                proposed_side = ev["best_what_if_side"]
            ev["proposed_side_mfe_bps"] = proposed_side_mfe
            ev["proposed_side"] = proposed_side
            ev["missed_opportunity"] = (
                (decision == "SKIP" or order_status == "BLOCKED_BY_GATE")
                and proposed_side_mfe >= HIGH_CONVICTION_BPS_THRESHOLD
            )
        enriched.append(ev)

    # Aggregate stats
    stats = {
        "total_events": len(enriched),
        "decisions": {},
        "missed_opportunity_count": 0,
        "missed_opportunity_total_bps": 0.0,
        "high_value_missed_50plus": 0,
        "high_value_missed_100plus": 0,
    }
    # Per-regime + per-session breakdown of missed opportunities
    regime_breakdown: dict[str, dict[str, int]] = {}
    session_breakdown: dict[str, dict[str, int]] = {}
    for ev in enriched:
        d = ev.get("v12_decision", {}).get("action", "UNKNOWN")
        stats["decisions"].setdefault(d, 0)
        stats["decisions"][d] += 1
        feats = ev.get("features") or {}
        regime = feats.get("vol_regime", "unknown")
        session = feats.get("session", "unknown")
        regime_breakdown.setdefault(regime, {"total": 0, "missed": 0})
        session_breakdown.setdefault(session, {"total": 0, "missed": 0})
        regime_breakdown[regime]["total"] += 1
        session_breakdown[session]["total"] += 1
        if ev.get("missed_opportunity"):
            stats["missed_opportunity_count"] += 1
            bps = ev.get("best_what_if_pnl_bps", 0)
            stats["missed_opportunity_total_bps"] += bps
            regime_breakdown[regime]["missed"] += 1
            session_breakdown[session]["missed"] += 1
            if bps >= 50:
                stats["high_value_missed_50plus"] += 1
            if bps >= 100:
                stats["high_value_missed_100plus"] += 1
    if stats["missed_opportunity_count"] > 0:
        stats["missed_opportunity_mean_bps"] = (
            stats["missed_opportunity_total_bps"] / stats["missed_opportunity_count"]
        )
    stats["regime_breakdown"] = regime_breakdown
    stats["session_breakdown"] = session_breakdown
    return enriched, stats


def replay_variants(journal_path: Path,
                    bundle_dir: Path,
                    variants: list[str],
                    fold_id: str = "FOLD_1",
                    aggregator: str = "mean",
                    max_events: int | None = None) -> dict:
    """For each journal event with `entry_iql_state_v1`, run that state through
    every requested variant's Q-net + aggregator and report which action it
    would have taken. Answers "would variant X have TAKEN this poll where the
    cement SKIPPED?" — supports the I) Loosening track from the
    2026-05-30 online-IQL prep design.

    Pure offline — no model retraining, no live runtime change. Loads variants
    from the SAME bundle dir as the active cement (per
    PROJECT_STATE_artifacts.json), so the only difference is reward variant
    + fold.

    Returns dict with per-variant action distribution + max adv_over_skip.
    Requires journal events to have `entry_iql_state_v1` captured (added
    2026-05-30 to v12_pipeline.make_entry_decision). Journals from before that
    date will report "0 replayable events".
    """
    import torch
    from collections import Counter
    from gx1.scripts import entry_iql_multi_head_gpu_core_v1 as iql_core

    ACTION_LABELS = {0: "SKIP", 1: "TAKE_LONG_NOW", 2: "TAKE_SHORT_NOW"}
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    states: list[list[float]] = []
    cement_actions: list[str] = []
    n_total = 0
    with open(journal_path) as f:
        for line in f:
            try: ev = json.loads(line)
            except json.JSONDecodeError: continue
            n_total += 1
            d = ev.get("v12_decision") or {}
            sv = d.get("entry_iql_state_v1")
            if sv is None:
                continue
            states.append(sv)
            cement_actions.append(d.get("action", "?"))
            if max_events is not None and len(states) >= max_events:
                break

    LOG.info(f"variants replay: {n_total} events scanned, "
             f"{len(states)} have entry_iql_state_v1 (replayable)")
    if not states:
        return {"n_total": n_total, "n_replayable": 0,
                "note": "no events have entry_iql_state_v1 — journal predates 2026-05-30 patch"}

    states_arr = np.asarray(states, dtype=np.float32)
    cement_dist = Counter(cement_actions)

    per_variant = {}
    for variant in variants:
        ckpt_path = bundle_dir / "trained_models_v1" / f"{variant}_{fold_id}.pt"
        if not ckpt_path.is_file():
            LOG.warning(f"  {variant}: checkpoint not found at {ckpt_path}, skipping")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dim = int(ckpt["state_dim"])
        n_actions = int(ckpt["n_actions"])
        n_k = len(list(ckpt["k_horizons"]))
        hidden_dim = int(ckpt.get("hidden_dim", 128))
        n_hidden = int(ckpt.get("n_hidden", 2))
        dropout = float(ckpt.get("dropout", iql_core.DEFAULT_DROPOUT))
        q_net = iql_core.MLP(state_dim, n_actions * n_k,
                             hidden_dim=hidden_dim, n_hidden=n_hidden, dropout=dropout).to(device)
        q_net.load_state_dict(ckpt["q_state_dict"]); q_net.eval()
        feature_means = np.asarray(ckpt["feature_means"], dtype=np.float32)
        feature_stds = np.asarray(ckpt["feature_stds"], dtype=np.float32)

        # PARITY FIX 2026-06-03: replicate the live/training normalization EXACTLY,
        # which clamps z-scores to ±z_clamp sigma and NaN-fills to 0.0
        # (entry_iql_multi_head_gpu_core_v1.py:130-135, z_clamp default 5.0). Before
        # this fix the replay omitted the clamp, so OOD live features (dead std=1e-6
        # cols firing nonzero -> |z| up to ~6e6) blew Q up to +1234 bps in the
        # shadow reports while the live runtime, which clamps, sat at q_skip ~120.
        # That artifact (and the false "LAM30 takes 77-100%") drove the 2026-06-03
        # live-vs-history alarm. Keep this in lockstep with the live core's _normalize.
        z_clamp = float(ckpt.get("z_clamp", 5.0))
        states_norm = (states_arr - feature_means) / np.maximum(feature_stds, 1e-6)
        states_norm = np.clip(states_norm, -z_clamp, z_clamp)
        states_norm = np.where(np.isnan(states_norm), 0.0, states_norm).astype(np.float32)
        with torch.no_grad():
            q_flat = q_net(torch.from_numpy(states_norm).to(device)).cpu().numpy()
        q_full = q_flat.reshape(-1, n_actions, n_k)
        if aggregator == "mean":   q_agg = q_full.mean(axis=2)
        elif aggregator == "max":  q_agg = q_full.max(axis=2)
        else: raise ValueError(f"unsupported aggregator {aggregator!r}")
        actions = q_agg.argmax(axis=1)
        labels = [ACTION_LABELS.get(int(a), str(a)) for a in actions]
        adv_max = (q_agg[:, 1:] - q_agg[:, [0]]).max(axis=1)
        n_would_take = int((adv_max > 0).sum())
        per_variant[variant] = {
            "distribution": dict(Counter(labels)),
            "n_would_take": n_would_take,
            "take_rate": n_would_take / len(states),
            "mean_q_skip": float(q_agg[:,0].mean()),
            "mean_q_take_long": float(q_agg[:,1].mean()),
            "mean_q_take_short": float(q_agg[:,2].mean()),
            "adv_over_skip_max": float(adv_max.max()),
        }
        LOG.info(f"  {variant}: take_rate={n_would_take}/{len(states)} "
                 f"({100*n_would_take/len(states):.2f}%) "
                 f"max_adv={adv_max.max():+.2f}")

    return dict(
        n_total=n_total,
        n_replayable=len(states),
        aggregator=aggregator,
        fold=fold_id,
        cement_distribution=dict(cement_dist),
        per_variant=per_variant,
    )


# ── Short-in-strong-uptrend stress test (2026-06-03, RR2) ──────────────────
# The explicit -2000 tripwire: on STRONG-UPTREND + OVERBOUGHT bars (the model
# shorted price +18..+89 bps OVER EMA200 in a D1 uptrend on 02.06 -> 16 stacked
# counter-trend shorts -> -2000), assert the model does NOT aggressively short.
# Runs over the PRE-GATE bar population (every journal poll), not just takes.
def stress_test_short_in_uptrend(
    journal_path: Path,
    overbought_bps: float = 50.0,        # pos_vs_ema200 >= this = strongly overbought (worst 02.06 shorts were +87..+89)
    max_short_take_rate: float = 0.01,   # at most ~1% of strong-uptrend-overbought bars may short
    max_mean_short_adv: float = 0.0,     # mean short Q-advantage must be <= 0 in this regime
) -> dict:
    """Read a paper journal; on D1-uptrend & overbought bars, measure the model's
    SHORT behavior. Returns metrics + passed:bool. Caller fail-closes on not-passed."""
    n_bars = n_regime = n_short = 0
    adv_short_sum = 0.0
    adv_short_n = 0
    worst = []
    for line in open(journal_path):
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        ft = ev.get("features") or {}
        dec = ev.get("v12_decision") or {}
        n_bars += 1
        stack = ft.get("d1_ema_stack_aligned_v2")
        pos = ft.get("pos_vs_ema200")
        if stack is None or pos is None:
            continue
        # strong uptrend (D1 stack aligned up) AND overbought (price well over EMA200)
        if not (float(stack) >= 1.0 and float(pos) >= overbought_bps):
            continue
        n_regime += 1
        if dec.get("action") == "TAKE_SHORT_NOW":
            n_short += 1
            worst.append({"ts": ev.get("ts_utc"), "pos_vs_ema200": round(float(pos), 1),
                          "adv_short": dec.get("advantage_over_skip_short")})
        a = dec.get("advantage_over_skip_short")
        if a is not None:
            adv_short_sum += float(a); adv_short_n += 1
    short_take_rate = (n_short / n_regime) if n_regime else 0.0
    mean_short_adv = (adv_short_sum / adv_short_n) if adv_short_n else 0.0
    passed = (n_regime == 0) or (
        short_take_rate <= max_short_take_rate and mean_short_adv <= max_mean_short_adv
    )
    return {
        "test": "short_in_strong_uptrend",
        "overbought_bps": overbought_bps,
        "n_bars": n_bars, "n_regime_bars": n_regime, "n_short_takes": n_short,
        "short_take_rate": round(short_take_rate, 4),
        "mean_short_advantage": round(mean_short_adv, 2),
        "ceilings": {"max_short_take_rate": max_short_take_rate, "max_mean_short_adv": max_mean_short_adv},
        "passed": bool(passed),
        "worst_short_takes": worst[:10],
    }


def main() -> int:
    p = argparse.ArgumentParser(description="V12 counterfactual replay (daily what-if analysis)")
    p.add_argument("--journal-date", type=str, required=True,
                   help="Date in YYYYMMDD format (e.g., 20260511)")
    p.add_argument("--journal-suffix", type=str, default="",
                   help="Suffix matching paper-runner --journal-suffix")
    p.add_argument("--out-dir", type=str, default=str(PAPER_DIR / "counterfactual_reports"))
    # 2026-05-30: alternate-variant replay extension. When --mode includes
    # 'variants' we also run each event's entry_iql_state_v1 through every
    # requested Entry-IQL variant and report take-rate vs the cement.
    p.add_argument("--mode", type=str, default="forward_outcome",
                   choices=["forward_outcome", "variants", "both", "stress"],
                   help="forward_outcome (default): what would the price have done? "
                        "variants: what would alternate IQL variants have decided? "
                        "both: run and report both. "
                        "stress: short-in-strong-uptrend regime-robustness gate (fail-closed, RR2).")
    p.add_argument("--stress-overbought-bps", type=float, default=50.0,
                   help="pos_vs_ema200 threshold for the strong-uptrend-overbought regime (stress mode).")
    p.add_argument("--variants", type=str,
                   default="R_NET_REAL,R_HYBRID_K96_TOL40,R_ASYMMETRIC_K96_LAM10",
                   help="Comma-separated Entry-IQL variants to replay (--mode variants/both).")
    p.add_argument("--fold", type=str, default="FOLD_1")
    p.add_argument("--aggregator", type=str, default="mean", choices=["mean","max"])
    p.add_argument("--bundle-dir", type=str, default=None,
                   help="Override Entry-IQL bundle dir. Default: ACTIVE entry_iql from PROJECT_STATE.")
    p.add_argument("--max-events", type=int, default=None,
                   help="Cap variant replay to N events (for smoke).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%SZ")

    suf = f"_{args.journal_suffix}" if args.journal_suffix else ""
    journal_path = PAPER_DIR / f"v12_paper_journal_{args.journal_date}{suf}.jsonl"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_journal = out_dir / f"counterfactual_{args.journal_date}{suf}.jsonl"
    out_summary = out_dir / f"counterfactual_summary_{args.journal_date}{suf}.json"

    # Stress mode (2026-06-03, RR2): short-in-strong-uptrend regime-robustness gate.
    # Fail-closed — returns nonzero if the model aggressively shorts a strong overbought
    # uptrend (the -2000 failure mode). Run pre-deploy on the retrained model's OOT journal.
    if args.mode == "stress":
        st = stress_test_short_in_uptrend(journal_path, overbought_bps=args.stress_overbought_bps)
        out_stress = out_dir / f"stress_short_in_uptrend_{args.journal_date}{suf}.json"
        out_stress.write_text(json.dumps(st, indent=2, default=str))
        print(json.dumps(st, indent=2, default=str))
        print(f"\nSHORT-IN-UPTREND STRESS TEST: {'PASS' if st['passed'] else 'FAIL'} -> {out_stress}")
        return 0 if st["passed"] else 1

    # Variant replay (2026-05-30 extension) — runs alongside forward-outcome
    # if --mode is 'variants' or 'both'. Skipped when --mode forward_outcome.
    variant_stats = None
    if args.mode in ("variants", "both"):
        if args.bundle_dir:
            bundle_dir = Path(args.bundle_dir).expanduser().resolve()
        else:
            from gx1_guards.artifacts import load_decision_artifact
            bundle_dir = Path(load_decision_artifact("entry_iql"))
            LOG.info(f"variants: bundle resolved from PROJECT_STATE → {bundle_dir.name}")
        variant_list = [v.strip() for v in args.variants.split(",") if v.strip()]
        variant_stats = replay_variants(
            journal_path, bundle_dir, variant_list,
            fold_id=args.fold, aggregator=args.aggregator, max_events=args.max_events,
        )
        out_variants = out_dir / f"counterfactual_variants_{args.journal_date}{suf}.json"
        out_variants.write_text(json.dumps(variant_stats, indent=2, default=str))
        print("\n=== Variant-replay summary ===")
        print(json.dumps(variant_stats, indent=2, default=str))
        if args.mode == "variants":
            return 0  # skip forward-outcome path entirely

    LOG.info(f"replaying {journal_path}")
    enriched, stats = replay_journal(journal_path)

    with out_journal.open("w") as f:
        for ev in enriched:
            f.write(json.dumps(ev, default=str) + "\n")
    out_summary.write_text(json.dumps(stats, indent=2, default=str))

    print("\n=== V12 Counterfactual Daily Report ===")
    print(json.dumps(stats, indent=2, default=str))
    print(f"\nFull enriched journal: {out_journal}")

    # Highlight high-value missed opportunities — surface regime + session for ML feedback
    missed = [e for e in enriched if e.get("missed_opportunity")]
    if missed:
        print(f"\nTop 10 missed opportunities (>= 50 bps forward MFE):")
        missed_sorted = sorted(missed, key=lambda e: e.get("best_what_if_pnl_bps", 0), reverse=True)
        for ev in missed_sorted[:10]:
            ts = ev.get("ts_utc", "?")
            bps = ev.get("best_what_if_pnl_bps", 0)
            side = ev.get("best_what_if_side", "?")
            decision = ev.get("v12_decision", {}).get("action", "?")
            order = ev.get("order_status", "?")
            spread = ev.get("spread_bps", 0)
            feats = ev.get("features") or {}
            regime = feats.get("vol_regime", "?")
            session = feats.get("session", "?")
            atr = feats.get("atr14_m5_bps", 0.0)
            stack = feats.get("ema_stack_aligned", 0)
            print(f"  {ts}  {side:5s} +{bps:6.1f} bps   V12={decision} "
                  f"({order}) spread={spread:.1f}  regime={regime} session={session} "
                  f"atr={atr:.1f}bps stack={stack:+d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
