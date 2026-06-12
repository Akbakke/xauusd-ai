#!/usr/bin/env python3
"""Build EXIT-side online replay buffer from live paper-runner journals.

Reads v12_paper_journal_*.jsonl files in a date range and extracts one
TRANSITION row per (trade_id, M1 bar) from `open_trade_records[].exit_decision`
— the captured 209-dim Exit-IQL state, the behavior action (HOLD/EXIT_NOW),
the per-bar Q-values, and trade-state — joined with the trade journal's
realized terminal outcome and the bounded post-exit counterfactual regret.

Output: a single parquet (e000..e208 state columns + labels) + .meta.json
sidecar. This is the EXIT analog of the entry-side buffer in the
continuous-learning ladder (nightly_learning loop).

Designed-not-forked:
  - EXTENSION CONSIDERED: gx1/scripts/build_online_replay_buffer.py (the
    entry-side buffer). It did NOT fit: the entry buffer consumes one row per
    journal POLL from `v12_decision.entry_iql_state_v1` (a 197-dim
    pre-extracted vector) and LABELS each row by recomputing the cement
    counterfactual reward matrix (SKIP/LONG/SHORT × K) from forward M1
    windows. The exit side is a different schema and different label
    semantics: one row per (trade_id, M1 bar) from
    `open_trade_records[].exit_decision.bar_state` (a NAMED ~615-key
    diagnostic superset that must be projected onto the bundle's ordered
    209-dim subset), 2-action HOLD/EXIT_NOW, and trade-TERMINAL labels
    (realized PnL, pnl-to-go, post-exit regret) instead of per-K
    counterfactual rewards. Forcing both into one script would tangle two
    unrelated event schemas; the shared pieces (journal iteration pattern,
    load_m1_window) are reused, not duplicated.
  - State projection mirrors ExitIQLV2Adapter EXACTLY: feature names come
    from the ACTIVE exit_iql bundle's summary_v1.json `feature_names_v1`
    (exit_iql_v2_adapter.py:96-103), bundle resolved ONLY via
    gx1_guards.artifacts.load_decision_entry (rule 8 — no glob/hardcoded
    path). Encoding mirrors build_state_vector
    (exit_iql_v2_adapter.py:158-177) incl. the `cat__val` one-hot partition.
    Missingness is STRICTER than the serve adapter: serve 0-fills
    sub-threshold-std features with a warning; a TRAINING buffer must never
    fabricate state, so ANY missing feature key ⇒ the row is DROPPED and
    counted loud in the skipped-dict (fail-closed, never silent fill).
  - Post-exit regret mirrors judge_take() in
    gx1/execution/v12_counterfactual_replay.py:199-228 — the one-truth port
    of replay_merge._write_post_exit_regret_audit (replay_merge.py:1326,
    "primary" decision-relevant formula): bounded POST_EXIT_HORIZON_BARS=24
    M1-bar follow-through after exit, favorable move from exit_bid (long) /
    exit_ask (short) on bid_close/ask_close, >= 10 bps == early_exit_regret.
    The constants are IMPORTED from v12_counterfactual_replay so the three
    sites stay in lockstep.

SCOPE — TRANSITIONS + REGRET LABELS ONLY:
  This builder does NOT construct training rewards. Reward-matrix
  construction for an exit-IQL refit MUST go through the exit builder's
  one-truth reward defs: build_reward_matrix() in
  gx1/scripts/materialize_build_exit_iql_v3_m1.py:282 (the layer
  BUILD_EXIT_IQL_V3_M1 that produced the ACTIVE bundle; R_NET_REAL
  coefficients imported from materialize_build_exit_iql_v2.py), under the
  locked MDP/reward contract
  gx1/scripts/materialize_exit_hold_exit_now_mdp_reward_contract_v1.py.
  `pnl_to_go_bps` / `post_exit_mfe_bps` here are VALUE/REGRET signals for
  analysis and critic accumulation, not drop-in IQL rewards.

Usage:
  python -m gx1.scripts.build_online_exit_replay_buffer \
      --from 20260611 --to 20260611 \
      --out /tmp/ladder_build/exit_buffer/smoke.parquet
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
    POST_EXIT_HORIZON_BARS,
    POST_EXIT_REGRET_THR_BPS,
    load_m1_window,
)
from gx1_guards.artifacts import load_decision_entry

JOURNAL_DIR = Path("/home/andre2/GX1_DATA/reports/v12_paper_runs")
TRADES_DIR = JOURNAL_DIR / "trade_journal" / "trades"

logging.basicConfig(level=logging.INFO, format="[exit_replay_buffer] %(message)s")
LOG = logging.getLogger("build_online_exit_replay_buffer")


def iter_journals(date_from: str, date_to: str, suffix: str) -> list[Path]:
    # Same filename convention as build_online_replay_buffer.iter_journals.
    glob = f"v12_paper_journal_*_{suffix}.jsonl"
    out = []
    for p in sorted(JOURNAL_DIR.glob(glob)):
        try:
            day = p.name.split("_")[3]
            if date_from <= day <= date_to:
                out.append(p)
        except IndexError:
            continue
    return out


def load_exit_iql_feature_names() -> tuple[list[str], str]:
    """Ordered 209 feature names from the ACTIVE exit_iql bundle.

    Mirrors ExitIQLV2Adapter.load (exit_iql_v2_adapter.py:96-103): the names
    are summary_v1.json `feature_names_v1` in file order — that order IS the
    serve-time state-vector order, so e000..e208 here == adapter state_v1.
    Bundle resolved via the rule-8 contract only.
    """
    entry = load_decision_entry("exit_iql")
    artifact_root = Path(entry["path"])
    summary_path = artifact_root / "summary_v1.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"summary missing: {summary_path}")
    summary = json.loads(summary_path.read_text())
    feature_names = list(summary["feature_names_v1"])
    if not feature_names:
        raise RuntimeError(f"empty feature_names_v1 in {summary_path}")
    return feature_names, artifact_root.name


def encode_state(bar_state: dict, feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Project the journal's named bar_state superset onto the ordered subset.

    Encoding mirrors ExitIQLV2Adapter.build_state_vector
    (exit_iql_v2_adapter.py:158-177): `cat__val` one-hot partition on "__",
    float coercion with non-numeric → 0.0. Returns (vector, missing_names);
    the CALLER drops the row when missing is non-empty (stricter than the
    serve adapter's std-gated 0-fill — see header).
    """
    v = np.zeros(len(feature_names), dtype=np.float32)
    missing: list[str] = []
    for i, fname in enumerate(feature_names):
        if "__" in fname:
            cat_col, _, cat_val = fname.partition("__")
            runtime_val = bar_state.get(cat_col)
            if runtime_val is None:
                missing.append(fname)
                continue
            if str(runtime_val) == cat_val:
                v[i] = 1.0
        else:
            raw = bar_state.get(fname)
            if raw is None:
                missing.append(fname)
                continue
            try:
                v[i] = float(raw)
            except (TypeError, ValueError):
                v[i] = 0.0
    return v, missing


def load_trade_terminal(trade_id: str) -> dict | None:
    """exit_summary from the per-trade journal file; None if unresolved."""
    fp = TRADES_DIR / f"{trade_id}.json"
    if not fp.exists():
        return None
    try:
        trade = json.loads(fp.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    es = trade.get("exit_summary") or {}
    if not es or not np.isfinite(float(es.get("realized_pnl_bps", np.nan))):
        return None
    return es


def compute_post_exit_regret(
    es: dict, side: str, m1: pd.DataFrame, m1_time_arr: np.ndarray
) -> tuple[float | None, bool]:
    """One-truth post-exit follow-through — mirrors judge_take
    (v12_counterfactual_replay.py:199-228), itself the port of
    replay_merge._write_post_exit_regret_audit primary formula
    (replay_merge.py:1326): exit price = exit_price_used if finite else
    exit_bid (long) / exit_ask (short), favorable move over the NEXT
    POST_EXIT_HORIZON_BARS M1 bars on bid_close/ask_close.
    Returns (post_exit_mfe_bps, observed)."""
    exit_ts = pd.to_datetime(es.get("exit_time"), utc=True, errors="coerce")
    px = es.get("exit_price_used")
    if px is None or not np.isfinite(float(px or np.nan)):
        px = es.get("exit_bid") if side == "long" else es.get("exit_ask")
    if pd.isna(exit_ts) or px is None or not np.isfinite(float(px)) or float(px) <= 0:
        return None, False
    px = float(px)
    exit_idx = int(np.searchsorted(m1_time_arr, exit_ts.value, side="right") - 1)
    start = max(0, exit_idx + 1)
    end = min(len(m1), start + POST_EXIT_HORIZON_BARS)
    if start >= end:
        return None, False
    window = m1.iloc[start:end]
    if side == "long":
        fav = float(window["bid_close"].max())
        val = (fav - px) / px * 10000.0
    else:
        fav = float(window["ask_close"].min())
        val = (px - fav) / px * 10000.0
    if not np.isfinite(val):
        return None, False
    return max(0.0, val), True


def collect_bars(journals: list[Path], skipped: dict) -> list[dict]:
    """Scan journals → one raw record per (trade_id, bars_in_trade) in-trade bar.

    UNIQUE KEY = (trade_id, bars_in_trade), NOT (trade_id, ts_utc): the runner
    journals one exit evaluation per poll at `current_minute`
    (v12_paper_runner.py:574/716) and multiple polls can land in the same
    wall-clock minute while trade.bars_in_trade advances per evaluation —
    verified 20260611: 138 records, 109 unique minutes, 138 unique
    (trade_id, bars_in_trade). ts_utc is the poll wall-clock, kept as-is.
    """
    bars: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for jpath in journals:
        with jpath.open() as fh:
            for line in fh:
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    skipped["corrupt_line"] += 1
                    continue
                otrs = ev.get("open_trade_records")
                if not otrs:
                    continue
                ts_str = ev.get("ts_utc")
                if not ts_str:
                    skipped["no_ts"] += len(otrs)
                    continue
                for rec in otrs:
                    tid = rec.get("trade_id")
                    ed = rec.get("exit_decision")
                    if tid is None or not isinstance(ed, dict):
                        skipped["no_exit_decision"] += 1
                        continue
                    tid = str(tid)
                    bit = ed.get("bars_in_trade", rec.get("bars_in_trade"))
                    if bit is None:
                        skipped["no_bars_in_trade"] += 1
                        continue
                    bit = int(bit)
                    if (tid, bit) in seen:
                        skipped["duplicate"] += 1
                        continue
                    bar_state = ed.get("bar_state")
                    if not isinstance(bar_state, dict) or not bar_state:
                        skipped["no_bar_state"] += 1
                        continue
                    action = ed.get("action")
                    q_hold, q_exit = ed.get("q_hold"), ed.get("q_exit")
                    if action not in ("HOLD", "EXIT_NOW") or q_hold is None or q_exit is None:
                        skipped["no_iql_action_or_q"] += 1
                        continue
                    seen.add((tid, bit))
                    bars.append({
                        "trade_id": tid,
                        "ts_utc": ts_str,
                        "side": str(rec.get("side", "")).lower(),
                        "bars_in_trade": bit,
                        "action_taken": str(action),
                        "decision_source": str(ed.get("decision_source", "")),
                        "q_hold": float(q_hold),
                        "q_exit": float(q_exit),
                        "current_pnl_bps": float(ed.get("current_pnl_bps", np.nan)),
                        "cum_mfe_bps": float(ed.get("cum_mfe_bps", np.nan)),
                        "cum_mae_bps": float(ed.get("cum_mae_bps", np.nan)),
                        "bar_state": bar_state,
                    })
    return bars


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="date_from", type=str, required=True,
                   help="Start date YYYYMMDD (inclusive)")
    p.add_argument("--to", dest="date_to", type=str, required=True,
                   help="End date YYYYMMDD (inclusive)")
    p.add_argument("--suffix", type=str, default="conviction67sized_skipasia_pure_phase6",
                   help="Journal filename suffix (after the date)")
    p.add_argument("--out", type=Path, required=True,
                   help="Output parquet path")
    args = p.parse_args()

    feature_names, bundle_name = load_exit_iql_feature_names()
    state_dim = len(feature_names)
    LOG.info(f"exit_iql bundle (contract-resolved): {bundle_name}  state_dim={state_dim}")

    journals = iter_journals(args.date_from, args.date_to, args.suffix)
    if not journals:
        LOG.error(f"No journals matched suffix={args.suffix} between {args.date_from} and {args.date_to}")
        return 2
    LOG.info(f"found {len(journals)} journal(s): {[j.name for j in journals]}")

    skipped = {"corrupt_line": 0, "no_ts": 0, "no_exit_decision": 0, "duplicate": 0,
               "no_bars_in_trade": 0, "no_bar_state": 0, "no_iql_action_or_q": 0,
               "unknown_side": 0, "trade_unresolved": 0, "state_missing_features": 0}
    bars = collect_bars(journals, skipped)
    if not bars:
        LOG.error("no in-trade bars found — journals contain no open_trade_records "
                  "with exit_decision.bar_state")
        return 3

    trade_ids = sorted({b["trade_id"] for b in bars}, key=lambda t: (len(t), t))
    LOG.info(f"in-trade bars: {len(bars)} across {len(trade_ids)} trade(s): {trade_ids}")

    # Terminal outcomes per trade — fail-closed: a trade with no resolvable
    # exit_summary (still open / no realized pnl) contributes ZERO rows.
    terminals: dict[str, dict] = {}
    for tid in trade_ids:
        es = load_trade_terminal(tid)
        if es is None:
            n_dropped = sum(1 for b in bars if b["trade_id"] == tid)
            skipped["trade_unresolved"] += n_dropped
            LOG.warning(f"trade {tid}: no exit_summary / realized pnl — dropping {n_dropped} bar(s)")
            continue
        terminals[tid] = es
    if not terminals:
        LOG.error("no trades with resolved exit_summary — buffer would be empty")
        return 3

    # M1 window for the post-exit regret leg: journal bars → last exit + the
    # bounded follow-through horizon (+60 min slack for tape edges).
    bar_ts = [pd.Timestamp(b["ts_utc"], tz="UTC") for b in bars]
    exit_ts = [pd.to_datetime(es.get("exit_time"), utc=True, errors="coerce")
               for es in terminals.values()]
    exit_ts = [t for t in exit_ts if pd.notna(t)]
    start_ts = min(bar_ts)
    end_ts = max(exit_ts + bar_ts) + pd.Timedelta(minutes=POST_EXIT_HORIZON_BARS + 60)
    LOG.info(f"loading M1 window: {start_ts} → {end_ts}")
    m1 = load_m1_window(start_ts, end_ts)
    if m1.empty:
        LOG.error("M1 window empty — collector parquets + M1 tape do not cover the range")
        return 4
    m1_time_arr = m1["time"].values
    LOG.info(f"M1 bars loaded: {len(m1):,}  first={m1['time'].iloc[0]}  last={m1['time'].iloc[-1]}")

    # Trade-level regret labels (repeated per row).
    regret: dict[str, dict] = {}
    for tid, es in terminals.items():
        side = str((next(b for b in bars if b["trade_id"] == tid))["side"])
        post_mfe, observed = compute_post_exit_regret(es, side, m1, m1_time_arr)
        regret[tid] = {
            "terminal_realized_pnl_bps": float(es["realized_pnl_bps"]),
            "exit_reason": str(es.get("exit_reason", "")),
            "post_exit_mfe_bps": post_mfe if post_mfe is not None else np.nan,
            "post_exit_observed": observed,
            "early_exit_regret": bool(observed and post_mfe >= POST_EXIT_REGRET_THR_BPS),
        }
        LOG.info(f"trade {tid} ({side}): realized={es['realized_pnl_bps']:+.1f} bps  "
                 f"post_exit_mfe={post_mfe if post_mfe is not None else 'n/a'}  "
                 f"early_exit_regret={regret[tid]['early_exit_regret']}")

    rows = []
    missing_feature_counts: dict[str, int] = {}
    for b in bars:
        tid = b["trade_id"]
        if tid not in regret:
            continue  # counted above as trade_unresolved
        if b["side"] not in ("long", "short"):
            skipped["unknown_side"] += 1
            continue
        state, missing = encode_state(b["bar_state"], feature_names)
        if missing:
            # Fail-closed: NEVER silently fill a training state (header note).
            skipped["state_missing_features"] += 1
            for f in missing:
                missing_feature_counts[f] = missing_feature_counts.get(f, 0) + 1
            continue
        r = regret[tid]
        row = {
            "trade_id": tid,
            "ts_utc": pd.Timestamp(b["ts_utc"], tz="UTC").isoformat(),
            "side": b["side"],
            "bars_in_trade": b["bars_in_trade"],
            "action_taken": b["action_taken"],
            "decision_source": b["decision_source"],
            "q_hold": b["q_hold"],
            "q_exit": b["q_exit"],
            "current_pnl_bps": b["current_pnl_bps"],
            "cum_mfe_bps": b["cum_mfe_bps"],
            "cum_mae_bps": b["cum_mae_bps"],
            "terminal_realized_pnl_bps": r["terminal_realized_pnl_bps"],
            # hold-vs-exit value signal: what the trade still had in it
            "pnl_to_go_bps": r["terminal_realized_pnl_bps"] - b["current_pnl_bps"],
            "post_exit_mfe_bps": r["post_exit_mfe_bps"],
            "post_exit_observed": r["post_exit_observed"],
            "early_exit_regret": r["early_exit_regret"],
            "exit_reason": r["exit_reason"],
        }
        for i in range(state_dim):
            row[f"e{i:03d}"] = float(state[i])
        rows.append(row)

    if missing_feature_counts:
        top = sorted(missing_feature_counts.items(), key=lambda kv: -kv[1])[:20]
        LOG.warning(f"rows dropped for missing state features — per-feature counts (top 20): {top}")
    LOG.info(f"buffer rows: {len(rows)}  skipped: {skipped}")
    if not rows:
        LOG.error("buffer empty after fail-closed filtering — nothing to write")
        return 3

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "schema_v1": "ONLINE_EXIT_REPLAY_BUFFER_V1",
        "n_trades": int(df["trade_id"].nunique()),
        "n_rows": int(len(df)),
        "state_dim": state_dim,
        "feature_name_source": {
            "bundle": bundle_name,
            "mechanism": "summary_v1.json feature_names_v1 via gx1_guards.load_decision_entry('exit_iql') "
                         "(mirrors exit_iql_v2_adapter.py:96-103)",
        },
        # e-column → feature-name map: load-bearing for any downstream refit.
        "feature_names_v1": feature_names,
        "post_exit_regret": {
            "horizon_bars_m1": POST_EXIT_HORIZON_BARS,
            "threshold_bps": POST_EXIT_REGRET_THR_BPS,
            "one_truth": "v12_counterfactual_replay.judge_take ← replay_merge._write_post_exit_regret_audit (primary)",
        },
        "reward_construction": "NOT here — materialize_build_exit_iql_v3_m1.build_reward_matrix "
                               "+ materialize_exit_hold_exit_now_mdp_reward_contract_v1 (see header)",
        "skipped": skipped,
        "missing_feature_counts": missing_feature_counts,
        "date_from": args.date_from,
        "date_to": args.date_to,
        "suffix": args.suffix,
        "journals": [j.name for j in journals],
    }, indent=2))
    LOG.info(f"wrote {len(df):,} rows × {len(df.columns)} cols → {args.out}")
    LOG.info(f"wrote metadata → {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
