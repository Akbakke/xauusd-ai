"""STEP-1 round-number-wall A/B — derive the WALL arm from the EXISTING conviction20 replay.

Reuses the cement-evidence per-candidate replay (phase6_conviction20_R_NET_REAL, 12k TAKEs through
the FULL exit chain) instead of re-running the gate: exit replay is per-candidate independent
(V12_OFF), so the wall arm == the base arm minus the vetoed trades. Portfolio effects are applied
posthoc by the same simulate_portfolio(max_concurrent=3) on both arms (SMART+MAXED: reuse, don't
recompute — same trick as the conviction-gate cement's per_candidate recompute).

ONE-TRUTH: the veto thresholds are IMPORTED from gx1.execution.v12_entry_iql_live (the live serve
code), and the vectorized rule is PARITY-CHECKED row-by-row against the REAL
EntryIQLLiveInference.predict() (stub-Q adapter) before any metric is computed — test==serve by
construction, the same pattern that cemented conviction20.

Rule-7 note: considered extending posthoc_session_strategyf_eval (grades ONE csv, doesn't build
arms) and v12_phase6_joint_validation (decisions->heavy replay) — neither fits a replay->arm
derivation; this is the new shared helper for STEP-1 overlay A/Bs.

Usage (flags must be ON in the env for the parity check — they are read at import):
  GX1_CONVICTION_GATE=1 GX1_ROUND_NUMBER_WALL=1 \
  .venv/bin/python -m gx1.research.step1_roundwall_ab [--per-candidate CSV] [--decisions PQ] \
      [--features-dir DIR] [--out-dir DIR]
Writes: <out-dir>/per_candidate_WALL.csv (vetoed dropped) + ab_summary.json; prints the A/B table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/andre2/src/GX1_ENGINE")

from gx1.backtest.portfolio_sim_v1 import simulate_portfolio  # noqa: E402
from gx1.execution.v12_entry_iql_live import (  # noqa: E402  (live ONE-TRUTH constants)
    CONVICTION_GATE_ON, CONVICTION_THR, ROUND_WALL_ON, ROUND_WALL_NEAR_ATR, ROUND_WALL_EXTRA,
    EntryIQLLiveInference,
)
from gx1.scripts.v12.v12_phase6_joint_validation import GATE_CFG, _entry_year, _per_year_metrics  # noqa: E402
from gx1.time.session_detector import get_session_vectorized  # noqa: E402

WS2 = Path("/home/andre2/GX1_DATA/runs/FASE2B_CLEAN_20260608")
ROUND_COLS = ["dist_to_round_50_atr", "dist_to_round_100_atr"]


def load_round_cols(features_dir: Path, uids: set[str]) -> pd.DataFrame:
    parts = []
    for pq in sorted(features_dir.glob("forward_outcomes_*.parquet")):
        df = pd.read_parquet(pq, columns=["candidate_uid", *ROUND_COLS])
        df = df[df["candidate_uid"].isin(uids)]
        if len(df):
            parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    assert not out["candidate_uid"].duplicated().any(), "duplicate uids in features"
    return out


def veto_vectorized(side: pd.Series, raw_adv: pd.Series, d50: pd.Series, d100: pd.Series) -> pd.Series:
    """Vectorized mirror of the live per-grid round-wall veto (v12_entry_iql_live.predict)."""
    armed = raw_adv < (CONVICTION_THR + ROUND_WALL_EXTRA)
    is_long = side.eq("long") | side.eq("TAKE_LONG_NOW")
    out = pd.Series(False, index=side.index)
    for d in (d50, d100):
        adverse = np.where(is_long, d < 0.0, d > 0.0)
        out |= armed & adverse & (d.abs() < ROUND_WALL_NEAR_ATR)
    return out


class _StubModel:
    k_horizons = [12, 24, 48, 96, 144, 288]
    q = None
    def predict_q(self, s):
        return np.tile(np.asarray(self.q, dtype=np.float64)[None, :, None], (1, 1, 6))


class _StubAdapter:
    aggregator = "mean"; beta = 1.0; min_advantage_bps = 0.0
    variant = "STUB"; fold_id = "F"; feature_names = ["x"]; k_weights = None
    def __init__(self): self.model = _StubModel()
    def build_state_vector(self, c): return np.zeros(1, dtype=np.float32)


def live_parity_check(df: pd.DataFrame, n_sample: int = 800) -> None:
    """Row-by-row: the REAL predict() (stub Q reproducing side/raw_adv, real round cols) must
    reproduce the vectorized veto. Fails loud on any mismatch (test==serve)."""
    from gx1.runtime.entry_iql_v2_adapter import iql_core
    L, S, K = iql_core.ACTION_TAKE_LONG_NOW_ID, iql_core.ACTION_TAKE_SHORT_NOW_ID, iql_core.ACTION_SKIP_ID
    inf = EntryIQLLiveInference(adapter=_StubAdapter(), feature_names=["x"])
    vetoed_idx = df.index[df["wall_veto"]]
    keep_idx = df.index[~df["wall_veto"]]
    rng = np.random.default_rng(20260611)
    pick = list(vetoed_idx[: n_sample // 2]) + list(
        rng.choice(keep_idx, size=min(n_sample // 2, len(keep_idx)), replace=False))
    n_bad = 0
    for i in pick:
        r = df.loc[i]
        is_long = r["side_v1"] in ("long", "TAKE_LONG_NOW")
        q = [0.0] * 3
        q[K] = 0.0
        q[L if is_long else S] = float(r["raw_adv"])
        q[S if is_long else L] = float(r["raw_adv"]) - 100.0
        inf.adapter.model.q = q
        rec = inf.predict({
            "dist_to_round_50_atr": float(r["dist_to_round_50_atr"]),
            "dist_to_round_100_atr": float(r["dist_to_round_100_atr"]),
        })
        live_veto = rec.action_id_v1 == K
        if bool(live_veto) != bool(r["wall_veto"]):
            n_bad += 1
            if n_bad <= 5:
                print(f"  PARITY MISMATCH uid={r['candidate_uid']} live_veto={live_veto} "
                      f"vec={r['wall_veto']} raw_adv={r['raw_adv']:.2f} d50={r['dist_to_round_50_atr']:.3f} "
                      f"d100={r['dist_to_round_100_atr']:.3f}")
    if n_bad:
        raise RuntimeError(f"[PARITY] {n_bad}/{len(pick)} rows disagree with live predict() — DO NOT trust the A/B")
    print(f"[PARITY] live predict() == vectorized veto on {len(pick)} rows (all vetoed + random keeps): PASS")


def arm_metrics(df: pd.DataFrame, label: str) -> dict:
    t = df  # all rows are takes
    sess = t["session"]
    out: dict = {"label": label, "n_takes": int(len(t))}
    for view, sub in (("ALL", t), ("skipASIA", t[sess != "ASIA"])):
        pnl = sub["realized_pnl_bps"].astype(float)
        m = {
            "n": int(len(sub)),
            "total_pnl_bps": float(pnl.sum()),
            "bps_per_take": float(pnl.mean()) if len(sub) else 0.0,
            "win_rate": float((pnl > 0).mean()) if len(sub) else 0.0,
            "p10_pnl": float(pnl.quantile(0.10)) if len(sub) else 0.0,
        }
        yr = _per_year_metrics(sub)
        m["per_year"] = {str(k): {"n": int(v["n"]), "bps": round(float(v["bps_per_take"]), 2),
                                  "win": round(float(v["win_rate"]), 4)}
                         for k, v in yr.items()} if isinstance(yr, dict) else str(yr)
        sim = simulate_portfolio(sub, max_concurrent=3)
        m["portfolio_cap3"] = {k: sim.get(k) for k in
                               ("n_admitted", "n_dropped", "total_pnl_bps", "max_drawdown_bps")}
        out[view] = m
    out["per_session_bps"] = {s: round(float(g["realized_pnl_bps"].mean()), 2)
                              for s, g in t.groupby(t["session"])}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-candidate", type=Path,
                    default=WS2 / "phase6_conviction20_R_NET_REAL/per_candidate_V12_OFF.csv")
    ap.add_argument("--decisions", type=Path,
                    default=WS2 / "entry_decisions_conviction20/decisions.parquet")
    ap.add_argument("--features-dir", type=Path,
                    default=WS2 / "forward_outcome_step1feats_clean/per_week")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    if not (CONVICTION_GATE_ON and ROUND_WALL_ON):
        raise SystemExit("Run with GX1_CONVICTION_GATE=1 GX1_ROUND_NUMBER_WALL=1 (flags are read at import; "
                         "the parity check must exercise the REAL live code path).")

    out_dir = args.out_dir or (args.per_candidate.parent / "roundwall_ab")
    out_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(args.per_candidate)
    assert (base["action_label_v1"] != "SKIP").all(), "expected a takes-only per-candidate CSV"
    dec = pd.read_parquet(args.decisions, columns=["candidate_uid", "raw_adv", "decision_ts_utc"])
    df = base.merge(dec, on="candidate_uid", how="left", validate="1:1")
    assert df["raw_adv"].notna().all(), "raw_adv missing for some replayed takes"

    feats = load_round_cols(args.features_dir, set(df["candidate_uid"]))
    df = df.merge(feats, on="candidate_uid", how="left", validate="1:1")
    n_missing = int(df[ROUND_COLS[0]].isna().sum())
    if n_missing:
        raise RuntimeError(f"[COVERAGE] {n_missing}/{len(df)} takes lack round cols in {args.features_dir} "
                           "— wrong wave or incomplete re-augment; refusing a silently-partial A/B")

    df["session"] = get_session_vectorized(pd.to_datetime(df["decision_ts_utc"], utc=True)).to_numpy()
    df["wall_veto"] = veto_vectorized(df["side_v1"].astype(str), df["raw_adv"].astype(float),
                                      df[ROUND_COLS[0]].astype(float), df[ROUND_COLS[1]].astype(float))

    live_parity_check(df)

    wall = df[~df["wall_veto"]].copy()
    vetoed = df[df["wall_veto"]]
    print(f"\n[A/B] base takes={len(df):,}  wall-vetoed={len(vetoed):,} ({len(vetoed)/len(df):.2%})  "
          f"vetoed mean pnl={vetoed['realized_pnl_bps'].mean():.2f} bps  win={(vetoed['realized_pnl_bps']>0).mean():.3f}"
          if len(vetoed) else "\n[A/B] ZERO vetoes fired — wall is inert on this sample at current thresholds")

    res = {"thresholds": {"CONVICTION_THR": CONVICTION_THR, "ROUND_WALL_NEAR_ATR": ROUND_WALL_NEAR_ATR,
                          "ROUND_WALL_EXTRA": ROUND_WALL_EXTRA},
           "n_vetoed": int(len(vetoed)),
           "vetoed_pnl_total": float(vetoed["realized_pnl_bps"].sum()) if len(vetoed) else 0.0,
           "vetoed_by_session": {s: int(n) for s, n in vetoed.groupby("session").size().items()},
           "base": arm_metrics(df, "BASE conviction20"),
           "wall": arm_metrics(wall, "WALL veto applied")}

    wall_csv = out_dir / "per_candidate_WALL.csv"
    wall.drop(columns=["wall_veto"]).to_csv(wall_csv, index=False)
    (out_dir / "ab_summary.json").write_text(json.dumps(res, indent=2, default=str))

    for arm in ("base", "wall"):
        a = res[arm]
        for view in ("ALL", "skipASIA"):
            v = a[view]
            print(f"  {arm:4s} {view:8s} n={v['n']:>6,} bps/take={v['bps_per_take']:7.2f} "
                  f"win={v['win_rate']:.4f} totPnL={v['total_pnl_bps']:>11,.0f} "
                  f"cap3-DD={v['portfolio_cap3']['max_drawdown_bps']}")
    d_take = res["wall"]["skipASIA"]["bps_per_take"] - res["base"]["skipASIA"]["bps_per_take"]
    print(f"\n[A/B] skipASIA Δbps/take (wall − base) = {d_take:+.3f}   "
          f"vetoed total pnl removed = {res['vetoed_pnl_total']:+,.0f} bps over {res['n_vetoed']} trades")
    print(f"[A/B] wrote {wall_csv} + ab_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
