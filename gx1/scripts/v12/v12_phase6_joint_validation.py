"""V12 Phase 6: Joint Entry-IQL + Exit-IQL v5 validation gate.

Per V12 design (handover protocol §5 Stage 4) we evaluate three configurations
on the same candidate set:

  V12_OFF: pure V12 stack — Entry-IQL v2 → Exit-IQL v5 (no V3 override)
  V12_ON : V12 stack + V3 fail-safe — if v3_v8_should_exit_prob > 0.95 then
           force EXIT_NOW that bar, regardless of IQL Q-values.

Optional: V9 baseline comparison can be done separately by joining against the
per_candidate_joint_eval CSV from Wave 2's Phase 7 LOCK (different candidate
pool, different iteration — left as a downstream side-by-side analysis).

Pipeline
--------
For each candidate in the V12 tracked dataset:
  1. Lookup Entry-IQL Phase 1 decision (TAKE_LONG / TAKE_SHORT / SKIP).
  2. SKIP → joint_pnl = 0, exit_reason = ENTRY_IQL_SKIP.
  3. TAKE → simulate exit:
       - For bar t in 1..max_bars: build bar_state from V12 row, run
         Exit-IQL v5 → action ∈ {HOLD, EXIT_NOW}.
       - V12_ON: also check v3_v8_should_exit_prob[t] > 0.95 → force EXIT_NOW.
       - Take FIRST EXIT_NOW bar; realized_pnl = current_unrealized_pnl_bps[t].
       - If no exit → forced terminal at last bar of trajectory (≤ 1440).
  4. Record realized_pnl_bps, exit_bar, exit_reason per candidate.

Summary metrics (per config):
  joint_mean_pnl_bps, joint_total_pnl_bps, n_take, n_skip, win_rate,
  exit_iql_active_frac, v3_override_active_frac (V12_ON only),
  mean_bars_held, mean_pnl_per_take

Run:
    PYTHONPATH=/home/andre2/src/GX1_ENGINE python3 /tmp/v12_phase6_joint_validation.py \\
        --v3tracked-lock /home/andre2/GX1_DATA/.../V3TRACKED_LOCK \\
        --exit-iql-v5-lock /home/andre2/GX1_DATA/.../EXIT_IQL_V5_V12_TRAINED_LOCK \\
        --entry-iql-decisions /home/andre2/GX1_DATA/.../ENTRY_IQL_INFERENCE_FOR_V12_*/decisions.parquet \\
        [--max-candidates 5000] [--v3-override-threshold 0.95]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path("/home/andre2/src/GX1_ENGINE")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.runtime.exit_iql_v2_adapter import ExitIQLV2Adapter  # noqa: E402

ACTION = "JOINT_ENTRY_EXIT_IQL_V5_V12_VALIDATION_GATE"
DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
V3_OVERRIDE_DEFAULT = 0.95
V12_MAX_BARS_DEFAULT = 1440  # match Phase 2 builder cap
EXIT_HOLD_ID = 0
EXIT_NOW_ID = 1


def load_v12_dataset(v3tracked_dir: Path) -> pd.DataFrame:
    """Load all per_week parquets from V12_PER_BAR_V3TRACKED LOCK."""
    files = sorted((v3tracked_dir / "per_week").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquets under {v3tracked_dir}/per_week")
    print(f"  Loading {len(files)} weekly V12 parquets...")
    t0 = time.time()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
    return df


def load_entry_iql_decisions(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["candidate_uid", "action_label_v1"])
    print(f"  Entry-IQL decisions: {len(df):,} candidates")
    return df


def simulate_one_candidate(
    candidate_bars: pd.DataFrame,
    *,
    side: str,
    exit_adapter: ExitIQLV2Adapter,
    v3_override_threshold: float | None,
    max_bars: int,
) -> dict[str, Any]:
    """Simulate exit decisions per bar for a single candidate.

    Args:
      candidate_bars: rows for ONE candidate sorted by bar_idx_v1
      side: "long" or "short"
      v3_override_threshold: if set (e.g. 0.95), force EXIT_NOW when
        v3_v8_should_exit_prob exceeds this. None = pure IQL.
      max_bars: hard cap on bars to consider (matches V12 design 1440)
    """
    if len(candidate_bars) == 0:
        return {"realized_pnl_bps": 0.0, "exit_bar": -1, "exit_reason": "NO_BARS",
                "iql_action": -1, "v3_override_fired": False}

    n_bars = min(len(candidate_bars), max_bars)
    sub = candidate_bars.iloc[:n_bars]

    # V3 override check (vectorized — first row where prob > threshold)
    v3_override_bar = -1
    if v3_override_threshold is not None and "v3_v8_should_exit_prob" in sub.columns:
        v3_probs = pd.to_numeric(sub["v3_v8_should_exit_prob"], errors="coerce").fillna(0.0).to_numpy()
        v3_hits = np.where(v3_probs > v3_override_threshold)[0]
        if len(v3_hits) > 0:
            v3_override_bar = int(v3_hits[0])

    # Build bar_states for batch IQL inference
    bar_states: list[dict[str, Any]] = []
    candidate_row = sub.iloc[0].to_dict()
    for _, row in sub.iterrows():
        s = row.to_dict()
        s["side_v1"] = side
        bar_states.append(s)

    recs = exit_adapter.predict(bar_states)
    iql_actions = np.array([r.action_id_v1 for r in recs], dtype=np.int32)
    iql_exit_indices = np.where(iql_actions == EXIT_NOW_ID)[0]
    iql_exit_bar = int(iql_exit_indices[0]) if len(iql_exit_indices) > 0 else -1

    # Resolve final exit: earliest of (V3 override, IQL EXIT_NOW)
    candidates_for_exit = [b for b in (v3_override_bar, iql_exit_bar) if b >= 0]
    if candidates_for_exit:
        first_exit = min(candidates_for_exit)
        exit_reason = (
            "V3_OVERRIDE" if first_exit == v3_override_bar and v3_override_bar >= 0 and (iql_exit_bar < 0 or v3_override_bar < iql_exit_bar)
            else "EXIT_IQL_SIGNAL"
        )
    else:
        first_exit = n_bars - 1
        exit_reason = "FORCED_TERMINAL"

    pnl_at_exit = float(sub["current_unrealized_pnl_bps_v1"].iloc[first_exit])
    return {
        "realized_pnl_bps": pnl_at_exit,
        "exit_bar": first_exit + 1,  # 1-based
        "exit_reason": exit_reason,
        "iql_exit_bar": iql_exit_bar + 1 if iql_exit_bar >= 0 else -1,
        "v3_override_bar": v3_override_bar + 1 if v3_override_bar >= 0 else -1,
        "v3_override_fired": exit_reason == "V3_OVERRIDE",
        "n_bars_considered": int(n_bars),
    }


def evaluate_config(
    df: pd.DataFrame,
    decisions: pd.DataFrame,
    exit_adapter: ExitIQLV2Adapter,
    *,
    config_name: str,
    v3_override_threshold: float | None,
    max_bars: int,
    max_candidates: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run validation pass for one config (V12_OFF or V12_ON)."""
    print(f"\n[{config_name}] Joining V12 dataset with Entry-IQL decisions...")
    merged = df.merge(decisions, on="candidate_uid", how="inner")
    candidate_uids = merged["candidate_uid"].unique()
    if max_candidates and len(candidate_uids) > max_candidates:
        rng = np.random.default_rng(20260508)
        candidate_uids = rng.choice(candidate_uids, size=max_candidates, replace=False)
        merged = merged[merged["candidate_uid"].isin(candidate_uids)]
    print(f"[{config_name}] candidates: {len(candidate_uids):,}, total bars: {len(merged):,}")

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for i, uid in enumerate(candidate_uids):
        cand_bars = merged[merged["candidate_uid"] == uid].sort_values("bar_idx_v1").reset_index(drop=True)
        if len(cand_bars) == 0:
            continue
        action_label = cand_bars["action_label_v1"].iloc[0]
        side = "long" if action_label == "TAKE_LONG_NOW" else "short" if action_label == "TAKE_SHORT_NOW" else None

        if side is None:
            row = {"candidate_uid": uid, "action_label_v1": action_label,
                   "realized_pnl_bps": 0.0, "exit_bar": -1, "exit_reason": "ENTRY_IQL_SKIP",
                   "v3_override_fired": False, "n_bars_considered": 0}
        else:
            sim = simulate_one_candidate(
                cand_bars, side=side, exit_adapter=exit_adapter,
                v3_override_threshold=v3_override_threshold, max_bars=max_bars,
            )
            row = {"candidate_uid": uid, "action_label_v1": action_label, "side_v1": side, **sim}
        rows.append(row)
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(1, elapsed)
            eta = (len(candidate_uids) - i - 1) / max(1, rate) / 60
            print(f"[{config_name}] [{i+1}/{len(candidate_uids)}] {rate:.1f} cand/s ETA {eta:.1f}min")

    out = pd.DataFrame(rows)
    n_total = len(out)
    n_skip = int((out["exit_reason"] == "ENTRY_IQL_SKIP").sum())
    n_take = n_total - n_skip
    n_iql = int((out["exit_reason"] == "EXIT_IQL_SIGNAL").sum())
    n_v3_override = int((out["exit_reason"] == "V3_OVERRIDE").sum()) if "v3_override_fired" in out.columns else 0
    n_forced = int((out["exit_reason"] == "FORCED_TERMINAL").sum())

    pnl_take = out[out["exit_reason"] != "ENTRY_IQL_SKIP"]["realized_pnl_bps"]
    bars_take = out[out["exit_reason"] != "ENTRY_IQL_SKIP"]["exit_bar"]
    summary = {
        "config": config_name,
        "n_total": n_total, "n_take": n_take, "n_skip": n_skip,
        "n_exit_iql": n_iql, "n_v3_override": n_v3_override, "n_forced_terminal": n_forced,
        "exit_iql_active_frac": float(n_iql / max(1, n_take)),
        "v3_override_frac": float(n_v3_override / max(1, n_take)),
        "forced_terminal_frac": float(n_forced / max(1, n_take)),
        "joint_mean_pnl_bps": float(out["realized_pnl_bps"].mean()),
        "joint_total_pnl_bps": float(out["realized_pnl_bps"].sum()),
        "mean_pnl_per_take": float(pnl_take.mean()) if n_take > 0 else 0.0,
        "win_rate_take": float((pnl_take > 0).mean()) if n_take > 0 else 0.0,
        "loss_rate_take": float((pnl_take < 0).mean()) if n_take > 0 else 0.0,
        "p25_pnl_take": float(pnl_take.quantile(0.25)) if n_take > 0 else 0.0,
        "median_pnl_take": float(pnl_take.median()) if n_take > 0 else 0.0,
        "p75_pnl_take": float(pnl_take.quantile(0.75)) if n_take > 0 else 0.0,
        "mean_bars_held_take": float(bars_take.mean()) if n_take > 0 else 0.0,
    }
    return out, summary


def main() -> int:
    p = argparse.ArgumentParser(description=ACTION)
    p.add_argument("--v3tracked-lock", required=True, help="Path to V12 V3TRACKED LOCK (Phase 4 output)")
    p.add_argument("--exit-iql-v5-lock", required=True, help="Path to EXIT_IQL_V5_V12_TRAINED LOCK")
    p.add_argument("--entry-iql-decisions", required=True, help="Path to Phase 1 decisions.parquet")
    p.add_argument("--variant", default="R_V12", help="Reward variant for Exit-IQL adapter (R_V12, R_NET_REAL, R_REGRET)")
    p.add_argument("--fold-id", default="FOLD_1")
    p.add_argument("--out-root", default=None)
    p.add_argument("--max-candidates", type=int, default=None, help="Subsample for fast eval")
    p.add_argument("--v3-override-threshold", type=float, default=V3_OVERRIDE_DEFAULT)
    p.add_argument("--max-bars", type=int, default=V12_MAX_BARS_DEFAULT)
    p.add_argument("--skip-v12-on", action="store_true", help="Only run V12_OFF (no V3 fail-safe)")
    p.add_argument("--skip-v12-off", action="store_true", help="Only run V12_ON (V3 fail-safe)")
    args = p.parse_args()

    v3tracked = Path(args.v3tracked_lock).expanduser().resolve()
    iql_lock = Path(args.exit_iql_v5_lock).expanduser().resolve()
    decisions_path = Path(args.entry_iql_decisions).expanduser().resolve()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = (Path(args.out_root).expanduser().resolve() if args.out_root
                else DEFAULT_REPORTS_ROOT / f"{ACTION}_{timestamp}_LOCK")
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"V12 PHASE 6 — Joint validation gate")
    print(f"  v3tracked    = {v3tracked}")
    print(f"  exit-iql-v5  = {iql_lock}")
    print(f"  decisions    = {decisions_path}")
    print(f"  out          = {out_root}")
    print(f"  v3 threshold = {args.v3_override_threshold}")
    print(f"  max_bars     = {args.max_bars}")

    # Load Exit-IQL v5 adapter
    print(f"\n[1/3] Loading Exit-IQL v5 adapter (variant={args.variant}, fold={args.fold_id})...")
    exit_adapter = ExitIQLV2Adapter.load(
        artifact_root=iql_lock,
        variant=args.variant, fold_id=args.fold_id,
        prefer_cuda=True,
    )
    print(f"  features={len(exit_adapter.feature_names)}, device={exit_adapter.model.device}")

    # Load V12 dataset + Entry-IQL decisions
    print(f"\n[2/3] Loading V12 V3TRACKED dataset + Entry-IQL decisions...")
    df = load_v12_dataset(v3tracked)
    decisions = load_entry_iql_decisions(decisions_path)

    # Run configs
    summaries: list[dict[str, Any]] = []
    print(f"\n[3/3] Running validation configs...")
    if not args.skip_v12_off:
        rows_off, sum_off = evaluate_config(
            df, decisions, exit_adapter,
            config_name="V12_OFF", v3_override_threshold=None,
            max_bars=args.max_bars, max_candidates=args.max_candidates,
        )
        rows_off.to_csv(out_root / "per_candidate_V12_OFF.csv", index=False)
        summaries.append(sum_off)

    if not args.skip_v12_on:
        rows_on, sum_on = evaluate_config(
            df, decisions, exit_adapter,
            config_name="V12_ON", v3_override_threshold=args.v3_override_threshold,
            max_bars=args.max_bars, max_candidates=args.max_candidates,
        )
        rows_on.to_csv(out_root / "per_candidate_V12_ON.csv", index=False)
        summaries.append(sum_on)

    # Summary report
    print("\n" + "=" * 80)
    print("PHASE 6 RESULTS")
    print("=" * 80)
    for s in summaries:
        print(f"\n[{s['config']}]")
        for k, v in s.items():
            if k == "config":
                continue
            print(f"  {k:30s} = {v}")

    summary_full = {
        "action_v1": ACTION,
        "built_at_utc_v1": datetime.now(timezone.utc).isoformat(),
        "v3tracked_lock": str(v3tracked),
        "exit_iql_v5_lock": str(iql_lock),
        "entry_iql_decisions": str(decisions_path),
        "variant": args.variant,
        "fold_id": args.fold_id,
        "v3_override_threshold": args.v3_override_threshold,
        "max_bars": args.max_bars,
        "max_candidates": args.max_candidates,
        "configs": summaries,
    }
    (out_root / "summary_v1.json").write_text(json.dumps(summary_full, indent=2, default=str))
    print(f"\n✅ Wrote: {out_root}/summary_v1.json + per_candidate_V12_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
