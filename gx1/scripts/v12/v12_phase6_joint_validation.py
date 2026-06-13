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
import os
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
from gx1.execution.v12_exit_iql_live import strategy_f_decision  # noqa: E402  (one-truth L7A overlay)

ACTION = "JOINT_ENTRY_EXIT_IQL_V5_V12_VALIDATION_GATE"
DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
V3_OVERRIDE_DEFAULT = 0.95
V12_MAX_BARS_DEFAULT = 1440  # match Phase 2 builder cap
EXIT_HOLD_ID = 0
EXIT_NOW_ID = 1


def load_v12_dataset(v3tracked_dir: Path,
                     candidate_uids: set[str] | None = None,
                     reports_root: Path | None = None,
                     canonical_path: Path | None = None) -> pd.DataFrame:
    """Load V12 V3TRACKED parquets WITH lazy-join of chunk_0 + canonical features.

    Critical: IQL-adapter expects 201 features including 106 chunk0 + 84 canonical
    columns that V12 V3TRACKED parquets do NOT contain. Without this lazy-join,
    adapter sees 0 for missing features and produces garbled Q-values (manifesting
    as exit_iql_active_frac=0% even when trained model is correct).

    Mimics `materialize_build_exit_iql_v3_m1.load_per_bar_dataset_lazy_join` but
    filters by candidate_uids to bound memory.
    """
    sys.path.insert(0, "/home/andre2/src/GX1_ENGINE")
    from gx1.scripts import materialize_build_candidate_forward_outcome_dataset_v1 as fwd_pipe
    from gx1.scripts import materialize_build_exit_iql_v2 as v2_train
    from gx1.scripts import materialize_build_exit_iql_v3_m1 as v3_m1

    files = sorted((v3tracked_dir / "per_week").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquets under {v3tracked_dir}/per_week")

    if reports_root is None:
        reports_root = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
    if canonical_path is None:
        canonical_path = v3_m1.DEFAULT_CANONICAL_FEATURES_PATH
    # FAIL LOUD on a missing canonical (rule-8/9): the Exit-IQL state includes the _v1_*_canon_v1 features.
    # _load_canonical_features() returns None on a missing file, which silently SKIPS the join → the 144
    # canonical cols never enter bar_state → FEATURE_COVERAGE_FATAL at predict (caught 2026-06-10: the gate
    # fell back to a DEAD default path and silently dropped all 144). Never silently fall back to a vintage.
    canonical_path = Path(canonical_path)
    if not canonical_path.exists():
        raise FileNotFoundError(
            f"[GATE] canonical features parquet missing: {canonical_path} — refusing to silently skip the "
            f"_v1_*_canon_v1 join (would FEATURE_COVERAGE_FATAL on 144 cols at predict). Pass "
            f"--canonical-features pointing at the SAME canonical the Exit-IQL was built against.")

    print(f"  Loading {len(files)} weekly V12 parquets"
          + (f" (filtered to {len(candidate_uids):,} candidates)..." if candidate_uids else "..."))
    print(f"  [lazy-join] canonical features: {canonical_path}")
    canonical = fwd_pipe._load_canonical_features(canonical_path)
    canonical_suf = v3_m1._suffix_canonical(canonical)
    # 2026-06-08: the train loader (load_per_bar_dataset_lazy_join) ALSO computes + merges the 64
    # AUG64 exit features (gated by GX1_EXIT_AUGMENT_64); this gate loader had omitted it, so an
    # AUG64-trained Exit-IQL hit FEATURE_COVERAGE_FATAL on the 64 missing cols. Mirror the train
    # loader exactly so gate state == train/serve state. No-op when GX1_EXIT_AUGMENT_64!=1.
    aug64 = v3_m1._compute_exit_aug64(canonical_path)
    if aug64 is not None:
        print(f"  [aug64] computed {len(v3_m1.NUMERIC_STATE_COLS_AUG64)} declared exit features (merge_asof per week)")

    t0 = time.time()
    parts = []
    weeks_with_chunk0 = 0
    for f in files:
        if candidate_uids is not None:
            df_p = pd.read_parquet(f, filters=[("candidate_uid", "in", candidate_uids)])
        else:
            df_p = pd.read_parquet(f)
        if len(df_p) == 0:
            continue
        # Per-week chunk_0 lazy-join (matches v3_m1.load_per_bar_dataset_lazy_join)
        week_name = f.stem.removeprefix("exit_per_bar_m1_")
        week_dir = reports_root / week_name
        chunk0 = fwd_pipe._load_chunk0_features(week_dir)
        chunk0_suf = v3_m1._suffix_chunk0(chunk0)
        if chunk0_suf is not None:
            df_p = v3_m1._merge_asof_features(df_p, chunk0_suf)
            weeks_with_chunk0 += 1
        else:
            for col in v2_train.NUMERIC_STATE_COLS_CHUNK0:
                df_p[col] = np.nan
        if canonical_suf is not None:
            df_p = v3_m1._merge_asof_features(df_p, canonical_suf)
        if aug64 is not None:
            df_p = v3_m1._merge_asof_features(df_p, aug64)
        parts.append(df_p)

    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    # 2026-06-08: one-hot expand the categoricals EXACTLY as the train build_state_matrix does
    # (get_dummies + pinned regime set). The gate bar_state is the raw per-bar row, so without this
    # the adapter misses session_*/vol_regime_*/trend_regime_*/side_v1_* (the last coverage layer).
    if len(df) > 0:
        _ONE_HOT_PIN = {"vol_regime": ["LOW", "MEDIUM", "HIGH", "EXTREME"],
                        "trend_regime": ["TREND_UP", "TREND_NEUTRAL", "TREND_DOWN"]}
        for c in v2_train.ONE_HOT_COLS:
            if c in df.columns:
                dummies = pd.get_dummies(df[c].astype(str), prefix=c, dummy_na=False)
                if c in _ONE_HOT_PIN:
                    dummies = dummies.reindex(columns=[f"{c}_{lbl}" for lbl in _ONE_HOT_PIN[c]], fill_value=0)
                for col in dummies.columns:
                    df[col] = dummies[col].astype(np.float32)
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} cols in {time.time()-t0:.1f}s "
          f"(chunk0-joined weeks: {weeks_with_chunk0}/{len(files)})")
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
    strategy_f: bool = False,
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
        return {"realized_pnl_bps": 0.0, "raw_fwd_pnl_bps": 0.0, "mae_bps": 0.0, "mfe_bps": 0.0,
                "exit_bar": -1, "exit_reason": "NO_BARS",
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

    # Strategy-F overlay (2026-06-13 vedtak L7A): score the +Strategy-F policy LIVE actually runs
    # (~55% of live exits). The 4-rule overlay (profit-lock / breakeven-cut / hold-horizon-expired,
    # suppressed by strong-hold) can force an EARLIER EXIT_NOW than the base IQL/v3 exit. ONE-TRUTH
    # strategy_f_decision shared with v12_exit_iql_live. Only scan up to the base exit (Strategy-F
    # later than the base exit is irrelevant) for cost.
    strategy_f_bar = -1
    if strategy_f:
        pnl_arr = pd.to_numeric(sub["current_unrealized_pnl_bps_v1"], errors="coerce").to_numpy(dtype=float)
        run_mfe = np.maximum.accumulate(np.where(np.isfinite(pnl_arr), pnl_arr, -np.inf))
        hold_pred = pd.to_numeric(pd.Series([candidate_row.get("hold_horizon_pred", -1.0)]),
                                  errors="coerce").iloc[0]
        hold_pred = float(hold_pred) if pd.notna(hold_pred) else -1.0
        _base = [b for b in (v3_override_bar, iql_exit_bar) if b >= 0]
        _scan_to = (min(_base) if _base else n_bars - 1) + 1
        for t in range(min(n_bars, _scan_to)):
            q_adv = float(recs[t].iql_recommendation_v1.advantage_exit_over_hold_v1 or 0.0)
            _mfe_t = float(run_mfe[t]) if np.isfinite(run_mfe[t]) else 0.0
            _pnl_t = float(pnl_arr[t]) if np.isfinite(pnl_arr[t]) else 0.0
            force, _r = strategy_f_decision(_mfe_t, _pnl_t, q_adv, hold_pred, t)
            if force:
                strategy_f_bar = t
                break

    # Resolve final exit: earliest of (V3 override, IQL EXIT_NOW, Strategy-F). V3 wins ties (Stage 4),
    # then IQL, then Strategy-F.
    candidates_for_exit = [b for b in (v3_override_bar, iql_exit_bar, strategy_f_bar) if b >= 0]
    if candidates_for_exit:
        first_exit = min(candidates_for_exit)
        if v3_override_bar >= 0 and first_exit == v3_override_bar:
            exit_reason = "V3_OVERRIDE"
        elif iql_exit_bar >= 0 and first_exit == iql_exit_bar:
            exit_reason = "EXIT_IQL_SIGNAL"
        else:
            exit_reason = "STRATEGY_F"
    else:
        first_exit = n_bars - 1
        exit_reason = "FORCED_TERMINAL"

    pnl_at_exit = float(sub["current_unrealized_pnl_bps_v1"].iloc[first_exit])
    # MAE/MFE over the ACTUAL hold [entry .. first_exit] (intra-trade excursion the
    # trade really experienced before exiting). 2026-06-03: Phase 6 previously only
    # recorded PnL at exit, so a take that profited at K could hide a deep adverse
    # drawdown — invisible to validation while the user's goal is MINIMAL MAE.
    held = pd.to_numeric(
        sub["current_unrealized_pnl_bps_v1"].iloc[: first_exit + 1], errors="coerce"
    )
    mae_bps = float(held.min())  # most adverse excursion (<=0 typically)
    mfe_bps = float(held.max())  # best favorable excursion before exit
    # RAW forward PnL at a FIXED horizon (K=144 bars), independent of the exit policy.
    # 2026-06-03 (GATE-4): the per-side Spearman calibration check must correlate the entry
    # advantage against RAW forward PnL — NOT realized_pnl_bps (the exit policy masks entry
    # anti-calibration). Used by the gate's per-side rank-calibration floor.
    _kfix = min(143, len(sub) - 1)
    raw_fwd_pnl_bps = float(pd.to_numeric(sub["current_unrealized_pnl_bps_v1"], errors="coerce").iloc[_kfix])
    return {
        "realized_pnl_bps": pnl_at_exit,
        "raw_fwd_pnl_bps": raw_fwd_pnl_bps,
        "mae_bps": mae_bps,
        "mfe_bps": mfe_bps,
        "exit_bar": first_exit + 1,  # 1-based
        "exit_reason": exit_reason,
        "iql_exit_bar": iql_exit_bar + 1 if iql_exit_bar >= 0 else -1,
        "v3_override_bar": v3_override_bar + 1 if v3_override_bar >= 0 else -1,
        "v3_override_fired": exit_reason == "V3_OVERRIDE",
        "strategy_f_bar": strategy_f_bar + 1 if strategy_f_bar >= 0 else -1,
        "strategy_f_fired": exit_reason == "STRATEGY_F",
        "n_bars_considered": int(n_bars),
    }


# ── 2026-06-11 parallel-replay worker plumbing (AGENTS.md SMART+MAXED #6) ──────────────
# Fork-inherited shared state: the parent sets _REPLAY_CTX BEFORE Pool creation; children get
# the frame + adapter via copy-on-write (no pickling, no per-worker RAM copy). Workers must
# treat everything in _REPLAY_CTX as READ-ONLY.
_REPLAY_CTX: dict | None = None


def _replay_worker_init() -> None:
    try:
        import torch
        torch.set_num_threads(1)   # one core per worker — the pool provides the parallelism
    except Exception:
        pass


def _replay_eval_chunk(uids) -> list[dict]:
    ctx = _REPLAY_CTX
    merged, groups = ctx["merged"], ctx["groups"]
    rows: list[dict] = []
    for uid in uids:
        idx = groups.get(uid)
        if idx is None or len(idx) == 0:
            continue
        cand_bars = merged.iloc[idx].sort_values("bar_idx_v1").reset_index(drop=True)
        action_label = cand_bars["action_label_v1"].iloc[0]
        side = "long" if action_label == "TAKE_LONG_NOW" else "short" if action_label == "TAKE_SHORT_NOW" else None
        if side is None:
            row = {"candidate_uid": uid, "action_label_v1": action_label,
                   "realized_pnl_bps": 0.0, "exit_bar": -1, "exit_reason": "ENTRY_IQL_SKIP",
                   "v3_override_fired": False, "n_bars_considered": 0}
        else:
            sim = simulate_one_candidate(
                cand_bars, side=side, exit_adapter=ctx["exit_adapter"],
                v3_override_threshold=ctx["v3_thr"], max_bars=ctx["max_bars"],
                strategy_f=ctx.get("strategy_f", False),
            )
            row = {"candidate_uid": uid, "action_label_v1": action_label, "side_v1": side, **sim}
        rows.append(row)
    return rows


def evaluate_config(
    df: pd.DataFrame,
    decisions: pd.DataFrame,
    exit_adapter: ExitIQLV2Adapter,
    *,
    config_name: str,
    v3_override_threshold: float | None,
    max_bars: int,
    max_candidates: int | None,
    strategy_f: bool = False,
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

    # 2026-06-11 PARALLEL REPLAY (AGENTS.md SMART+MAXED #6, user vedtak — the 5×1h/1-core lesson).
    # Per-candidate replay is embarrassingly parallel; fork shares `merged` + the adapter COW.
    # GX1_REPLAY_WORKERS=1 forces the ORIGINAL serial path (bit-reference for determinism diffs).
    _n_workers = int(os.environ.get("GX1_REPLAY_WORKERS",
                                    str(min(12, max(1, (os.cpu_count() or 2) - 4)))))
    t0 = time.time()
    if _n_workers <= 1:
        rows: list[dict[str, Any]] = []
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
                    strategy_f=strategy_f,
                )
                row = {"candidate_uid": uid, "action_label_v1": action_label, "side_v1": side, **sim}
            rows.append(row)
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / max(1, elapsed)
                eta = (len(candidate_uids) - i - 1) / max(1, rate) / 60
                print(f"[{config_name}] [{i+1}/{len(candidate_uids)}] {rate:.1f} cand/s ETA {eta:.1f}min")
    else:
        import multiprocessing as _mp
        global _REPLAY_CTX
        # groupby index = O(1) per-candidate lookup (the serial path's boolean scan over the full
        # frame was O(N) × n_candidates — the second half of the leak). Same rows, same order.
        _REPLAY_CTX = {
            "merged": merged, "groups": merged.groupby("candidate_uid", sort=False).indices,
            "exit_adapter": exit_adapter, "v3_thr": v3_override_threshold, "max_bars": max_bars,
            "strategy_f": strategy_f,
        }
        chunks = np.array_split(np.asarray(candidate_uids, dtype=object), _n_workers * 4)
        chunks = [c for c in chunks if len(c)]
        print(f"[{config_name}] PARALLEL replay: {_n_workers} workers × {len(chunks)} chunks "
              f"(GX1_REPLAY_WORKERS; fork/COW shared frame)")
        ctx = _mp.get_context("fork")
        with ctx.Pool(_n_workers, initializer=_replay_worker_init) as pool:
            chunk_rows = pool.map(_replay_eval_chunk, chunks)
        rows = [r for cr in chunk_rows for r in cr]
        _REPLAY_CTX = None
        print(f"[{config_name}] PARALLEL replay done: {len(rows):,} rows in {(time.time()-t0)/60:.1f}min")

    out = pd.DataFrame(rows)
    n_total = len(out)
    n_skip = int((out["exit_reason"] == "ENTRY_IQL_SKIP").sum())
    n_take = n_total - n_skip
    n_iql = int((out["exit_reason"] == "EXIT_IQL_SIGNAL").sum())
    n_v3_override = int((out["exit_reason"] == "V3_OVERRIDE").sum()) if "v3_override_fired" in out.columns else 0
    n_forced = int((out["exit_reason"] == "FORCED_TERMINAL").sum())

    take_mask = out["exit_reason"] != "ENTRY_IQL_SKIP"
    pnl_take = out[take_mask]["realized_pnl_bps"]
    bars_take = out[take_mask]["exit_bar"]
    # MAE/MFE aggregates over takes (2026-06-03). MAE is negative-good→bad; report the
    # mean and the deep-drawdown tail (p05 / worst) plus the fraction of takes that dip
    # below -20 bps, and PnL-per-unit-MAE efficiency. Directly serves the minimal-MAE goal.
    mae_take = pd.to_numeric(out[take_mask].get("mae_bps"), errors="coerce") if "mae_bps" in out.columns else pd.Series(dtype=float)
    mfe_take = pd.to_numeric(out[take_mask].get("mfe_bps"), errors="coerce") if "mfe_bps" in out.columns else pd.Series(dtype=float)
    mae_take = mae_take.dropna()
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
        # --- MAE / MFE (intra-trade excursion to exit) ---
        "mean_mae_bps_take": float(mae_take.mean()) if len(mae_take) else 0.0,
        "median_mae_bps_take": float(mae_take.median()) if len(mae_take) else 0.0,
        "p05_mae_bps_take": float(mae_take.quantile(0.05)) if len(mae_take) else 0.0,
        "worst_mae_bps_take": float(mae_take.min()) if len(mae_take) else 0.0,
        "frac_take_mae_below_-20bps": float((mae_take < -20.0).mean()) if len(mae_take) else 0.0,
        "mean_mfe_bps_take": float(mfe_take.dropna().mean()) if len(mfe_take.dropna()) else 0.0,
        "mean_pnl_per_abs_mae": (
            float(pnl_take.mean() / abs(mae_take.mean()))
            if len(mae_take) and abs(mae_take.mean()) > 1e-9 else 0.0
        ),
    }
    return out, summary


# ── Phase 6 acceptance gate (2026-06-03, S6) ──────────────────────────────

import re as _re

_UID_TS_RE = _re.compile(r"(\d{8}T\d{6})")

# Absolute floors + relative tolerances. Relative checks apply only when an anchor
# (prior clean summary) is supplied. Per-year floors stop a single weak regime (e.g.
# 2026) from auto-failing a real improvement while still catching collapse, and stop a
# pile-up from hiding inside a pooled mean.
GATE_CFG = {
    "min_year_mean_bps": 12.0,
    "min_year_win": 0.85,
    "min_admit_rate": 0.60,
    "max_opposing_overlaps": 0,
    "rel_pnl": 0.95,         # mean_pnl_per_take >= 0.95x anchor
    "rel_drawdown": 1.15,    # |drawdown| <= 1.15x anchor
    "rel_mae": 1.20,         # |median MAE| / frac<-20 <= 1.20x anchor
    "min_side_spearman": 0.10,  # GATE-4: per-side Spearman(adv, RAW fwd PnL) floor (long & short)
}


def _entry_year(uid: str):
    m = _UID_TS_RE.findall(str(uid))
    if not m:
        return None
    try:
        return int(m[-1][:4])
    except ValueError:
        return None


def _per_year_metrics(rows: "pd.DataFrame") -> dict:
    """Per-year mean/median/win over TAKE rows (parsed from candidate_uid)."""
    take = rows[rows["exit_reason"] != "ENTRY_IQL_SKIP"].copy()
    if take.empty:
        return {}
    take["_year"] = take["candidate_uid"].map(_entry_year)
    out = {}
    for y, g in take.groupby("_year"):
        if y is None:
            continue
        p = pd.to_numeric(g["realized_pnl_bps"], errors="coerce")
        out[int(y)] = {"n": int(len(g)), "mean": float(p.mean()),
                       "median": float(p.median()), "win": float((p > 0).mean())}
    return out


def evaluate_phase6_gate(summary: dict, per_year: dict, portfolio_live: dict,
                         anchor: dict | None, cfg: dict = GATE_CFG,
                         take_calib: "pd.DataFrame | None" = None) -> tuple[bool, list]:
    """Return (passed, checks). checks = list of {name, ok, detail}. Fail-closed:
    any failed check -> not passed. Relative checks only when anchor is given.
    take_calib (optional): per-take frame with [side_v1, advantage, raw_fwd_pnl_bps] for the
    per-side rank-calibration floor (RR3)."""
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    # (1) per-year floors (absolute) — catch regime collapse + hidden pile-ups
    if per_year:
        worst_mean = min(v["mean"] for v in per_year.values())
        worst_win = min(v["win"] for v in per_year.values())
        add("per_year_min_mean_bps", worst_mean >= cfg["min_year_mean_bps"],
            f"min-year mean {worst_mean:.1f} >= {cfg['min_year_mean_bps']}")
        add("per_year_min_win", worst_win >= cfg["min_year_win"],
            f"min-year win {worst_win:.3f} >= {cfg['min_year_win']}")
    else:
        add("per_year_metrics_present", False, "no per-year metrics (cannot gate)")

    # (2) portfolio invariants + realistic drawdown/admit under the live safety layer
    add("no_opposing_overlaps", portfolio_live.get("n_opposing_side_overlaps", 1) <= cfg["max_opposing_overlaps"],
        f"opposing_overlaps={portfolio_live.get('n_opposing_side_overlaps')} <= {cfg['max_opposing_overlaps']}")
    add("max_same_side_within_cap",
        portfolio_live.get("max_same_side_concurrent", 10**9) <= (portfolio_live.get("max_concurrent_cap") or 10**9),
        f"max_same_side={portfolio_live.get('max_same_side_concurrent')} <= cap {portfolio_live.get('max_concurrent_cap')}")
    add("admit_rate", portfolio_live.get("admit_rate", 0.0) >= cfg["min_admit_rate"],
        f"admit_rate={portfolio_live.get('admit_rate')} >= {cfg['min_admit_rate']}")

    # (3) relative-to-anchor (only if anchor supplied)
    if anchor:
        a_mean = anchor.get("mean_pnl_per_take", 0.0)
        if a_mean > 0:
            add("rel_pnl_vs_anchor", summary.get("mean_pnl_per_take", 0.0) >= cfg["rel_pnl"] * a_mean,
                f"mean_pnl {summary.get('mean_pnl_per_take'):.1f} >= {cfg['rel_pnl']}x anchor {a_mean:.1f}")
        a_med_mae = abs(anchor.get("median_mae_bps_take", 0.0))
        s_med_mae = abs(summary.get("median_mae_bps_take", 0.0))
        if a_med_mae > 1e-9:
            add("rel_mae_vs_anchor", s_med_mae <= cfg["rel_mae"] * a_med_mae,
                f"|median MAE| {s_med_mae:.1f} <= {cfg['rel_mae']}x anchor {a_med_mae:.1f}")
        a_dd = abs(anchor.get("_portfolio_drawdown_bps", 0.0))
        s_dd = abs(portfolio_live.get("max_account_drawdown_bps", 0.0))
        if a_dd > 1e-9:
            add("rel_drawdown_vs_anchor", s_dd <= cfg["rel_drawdown"] * a_dd,
                f"|drawdown| {s_dd:.0f} <= {cfg['rel_drawdown']}x anchor {a_dd:.0f}")
    else:
        add("anchor_supplied", True, "no anchor — relative checks skipped (first clean baseline run)")

    # (4) PER-SIDE RANK-CALIBRATION (RR3/GATE-4): Spearman(entry advantage, RAW forward PnL),
    # computed separately for long-takes and short-takes, BOTH >= floor. The -2000 model had
    # short rho -0.28 (higher conviction -> worse trade). Correlate vs RAW fwd PnL, not joint-exit.
    if take_calib is not None and len(take_calib) and {"side_v1", "advantage", "raw_fwd_pnl_bps"}.issubset(take_calib.columns):
        floor = float(cfg.get("min_side_spearman", 0.10))
        for side in ("long", "short"):
            s = take_calib[take_calib["side_v1"] == side]
            a = pd.to_numeric(s["advantage"], errors="coerce")
            r = pd.to_numeric(s["raw_fwd_pnl_bps"], errors="coerce")
            ok_n = len(s) >= 50 and a.nunique() >= 5
            rho = float(a.rank().corr(r.rank())) if ok_n else float("nan")
            if not ok_n:
                add(f"side_spearman_{side}", True, f"{side}: n={len(s)} <50 — skipped (insufficient takes)")
            else:
                add(f"side_spearman_{side}", rho >= floor, f"{side} Spearman(adv,raw_fwd_pnl)={rho:.3f} >= {floor}")
    else:
        add("side_calibration_available", True, "no take_calib frame — per-side Spearman skipped")

    passed = all(c["ok"] for c in checks)
    return passed, checks


def main() -> int:
    p = argparse.ArgumentParser(description=ACTION)
    p.add_argument("--v3tracked-lock", required=True, help="Path to V12 V3TRACKED LOCK (Phase 4 output)")
    p.add_argument("--exit-iql-v5-lock", required=True, help="Path to EXIT_IQL_V5_V12_TRAINED LOCK")
    p.add_argument("--entry-iql-decisions", required=True, help="Path to Phase 1 decisions.parquet")
    p.add_argument("--variant", default="R_V12", help="Reward variant for Exit-IQL adapter (R_V12, R_NET_REAL, R_REGRET)")
    p.add_argument("--fold-id", default="FOLD_1")
    p.add_argument("--out-root", default=None)
    p.add_argument("--canonical-features", default=None,
                   help="Canonical features parquet to lazy-join (MUST match the one the Exit-IQL was built "
                        "against). Fail-closed if the resolved path is missing — no silent fallback (rule 8/9).")
    p.add_argument("--max-candidates", type=int, default=None, help="Subsample for fast eval")
    p.add_argument("--v3-override-threshold", type=float, default=V3_OVERRIDE_DEFAULT)
    p.add_argument("--max-bars", type=int, default=V12_MAX_BARS_DEFAULT)
    p.add_argument("--skip-v12-on", action="store_true", help="Only run V12_OFF (no V3 fail-safe)")
    p.add_argument("--skip-v12-off", action="store_true", help="Only run V12_ON (V3 fail-safe)")
    # 2026-06-03 (S6): turn Phase 6 into a real PASS/FAIL gate. Opt-in to preserve
    # back-compat for exploratory runs; the retrain-vedtak procedure MUST pass --gate.
    p.add_argument("--gate", action="store_true",
                   help="Enforce PASS/FAIL acceptance gate; main() returns nonzero on FAIL.")
    p.add_argument("--gate-anchor", default=None,
                   help="Path to a prior CLEAN summary_v1.json for relative thresholds "
                        "(pnl>=0.95x, drawdown<=1.15x, MAE<=1.2x). Omit on the first clean "
                        "baseline run (absolute floors only).")
    p.add_argument("--gate-hard-cap", type=int, default=2,
                   help="Same-side concurrency hard cap modeled in the portfolio gate (matches live HARD_MAX_SAME_SIDE).")
    p.add_argument("--strategy-f", dest="strategy_f", action="store_true",
                   default=(os.environ.get("GX1_GATE_STRATEGY_F", "0") == "1"),
                   help="2026-06-13 L7A: apply the live +Strategy-F exit overlay in the gate "
                        "(scores the policy live actually runs). Default off = pure Exit-IQL+v3 (the "
                        "historical cement-validation policy). Env GX1_GATE_STRATEGY_F=1 also enables.")
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
    # 2026-06-11 parallel replay: forked workers cannot touch a parent CUDA context
    # ("Cannot re-initialize CUDA in forked subprocess") — load the tiny Q-net on CPU
    # when the worker pool is active (12 × CPU beats 1 × GPU here; V3 probs are
    # pre-scored in the frame, the adapter is a small MLP).
    _replay_workers = int(os.environ.get("GX1_REPLAY_WORKERS",
                                         str(min(12, max(1, (os.cpu_count() or 2) - 4)))))
    exit_adapter = ExitIQLV2Adapter.load(
        artifact_root=iql_lock,
        variant=args.variant, fold_id=args.fold_id,
        prefer_cuda=(_replay_workers <= 1),
    )
    print(f"  features={len(exit_adapter.feature_names)}, device={exit_adapter.model.device} "
          f"(replay_workers={_replay_workers})")

    # Load V12 dataset + Entry-IQL decisions
    print(f"\n[2/3] Loading V12 V3TRACKED dataset + Entry-IQL decisions...")
    decisions = load_entry_iql_decisions(decisions_path)
    # Pre-sample candidate_uids to bound memory: full 63.6M-row load OOMs on 15GB host.
    # Filter to TAKE actions, then optionally sub-sample to args.max_candidates.
    take_uids = decisions[decisions["action_label_v1"].isin(
        ["TAKE_LONG_NOW", "TAKE_SHORT_NOW"])]["candidate_uid"].astype(str).tolist()
    print(f"  TAKE candidates available: {len(take_uids):,}")
    if args.max_candidates and len(take_uids) > args.max_candidates:
        rng = np.random.default_rng(20260509)
        take_uids = list(rng.choice(take_uids, size=args.max_candidates, replace=False))
        print(f"  Sub-sampled to {len(take_uids):,} candidates for memory safety")
    df = load_v12_dataset(v3tracked, candidate_uids=set(take_uids),
                          canonical_path=(Path(args.canonical_features).expanduser().resolve()
                                          if args.canonical_features else None))

    # Run configs
    summaries: list[dict[str, Any]] = []
    print(f"\n[3/3] Running validation configs...")
    if not args.skip_v12_off:
        rows_off, sum_off = evaluate_config(
            df, decisions, exit_adapter,
            config_name="V12_OFF", v3_override_threshold=None,
            max_bars=args.max_bars, max_candidates=args.max_candidates,
            strategy_f=args.strategy_f,
        )
        rows_off.to_csv(out_root / "per_candidate_V12_OFF.csv", index=False)
        summaries.append(sum_off)

    if not args.skip_v12_on:
        rows_on, sum_on = evaluate_config(
            df, decisions, exit_adapter,
            config_name="V12_ON", v3_override_threshold=args.v3_override_threshold,
            max_bars=args.max_bars, max_candidates=args.max_candidates,
            strategy_f=args.strategy_f,
        )
    if args.strategy_f:
        print("[L7A] Strategy-F overlay ACTIVE in the gate (scoring the live +Strategy-F policy).")
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

    # Companion: sequential MAE/drawdown-aware portfolio sim (2026-06-03). Phase 6 above
    # scores candidates independently (up to ~35 concurrent, unseen); this shows realized
    # edge + account drawdown under a concurrency cap. Non-fatal: the primary Phase 6
    # output is already on disk — a sim bug must never discard a multi-hour run.
    gate_passed = None
    try:
        from gx1.backtest.portfolio_sim_v1 import cap_sweep, simulate_portfolio
        for cfg in ("V12_OFF", "V12_ON"):
            csv_path = out_root / f"per_candidate_{cfg}.csv"
            if not csv_path.is_file():
                continue
            sweep = cap_sweep(pd.read_csv(csv_path))
            (out_root / f"portfolio_sim_{cfg}.json").write_text(json.dumps(sweep, indent=2, default=str))
            print(f"   + portfolio_sim_{cfg}.json (concurrency-cap sweep)")

        # ── Acceptance GATE (S6) — gate on V12_OFF (primary), else V12_ON ──
        gate_cfg_name = "V12_OFF" if (out_root / "per_candidate_V12_OFF.csv").is_file() else "V12_ON"
        gate_csv = out_root / f"per_candidate_{gate_cfg_name}.csv"
        if gate_csv.is_file():
            gate_rows = pd.read_csv(gate_csv)
            per_year = _per_year_metrics(gate_rows)
            gate_summary = next((s for s in summaries if s.get("config") == gate_cfg_name), summaries[0])
            anchor = None
            if args.gate_anchor:
                a = json.loads(Path(args.gate_anchor).read_text())
                anchor = (a.get("configs", [a])[0]) if isinstance(a, dict) else None
                if isinstance(a, dict) and "_portfolio_drawdown_bps" in a:
                    anchor = {**(anchor or {}), "_portfolio_drawdown_bps": a["_portfolio_drawdown_bps"]}
            try:
                portfolio_live = simulate_portfolio(
                    gate_rows, max_concurrent=args.gate_hard_cap,
                    enforce_invariants=True, require_mae=True)
            except ValueError as ve:
                # No MAE column -> cannot honestly gate. Fail-closed when --gate.
                portfolio_live = {"error": str(ve), "n_opposing_side_overlaps": 1,
                                  "admit_rate": 0.0, "max_account_drawdown_bps": 0.0}
            # GATE-4: per-side calibration frame — join take rows (side + RAW fwd PnL) with the
            # entry advantage from decisions.parquet (by candidate_uid).
            take_calib = None
            try:
                _adv = decisions[["candidate_uid", "advantage_over_skip_v1"]].rename(
                    columns={"advantage_over_skip_v1": "advantage"})
                _tk = gate_rows[gate_rows["exit_reason"] != "ENTRY_IQL_SKIP"][
                    ["candidate_uid", "side_v1", "raw_fwd_pnl_bps"]]
                take_calib = _tk.merge(_adv, on="candidate_uid", how="inner")
            except Exception as _ce:  # noqa: BLE001 — calibration check is best-effort, gate degrades gracefully
                print(f"   (per-side calib frame unavailable: {_ce})")
            gate_passed, checks = evaluate_phase6_gate(
                gate_summary, per_year, portfolio_live, anchor, take_calib=take_calib)
            gate_out = {
                "gated_config": gate_cfg_name, "passed": bool(gate_passed),
                "checks": checks, "per_year": per_year,
                "portfolio_live_modeled": portfolio_live,
                "_portfolio_drawdown_bps": portfolio_live.get("max_account_drawdown_bps"),
                "anchor": args.gate_anchor, "gate_cfg": GATE_CFG,
            }
            (out_root / "phase6_gate_v1.json").write_text(json.dumps(gate_out, indent=2, default=str))
            print(f"\n{'='*80}\nPHASE 6 GATE [{gate_cfg_name}]: "
                  f"{'✅ PASS' if gate_passed else '❌ FAIL'}\n{'='*80}")
            for c in checks:
                print(f"  {'✓' if c['ok'] else '✗'} {c['name']:28s} {c['detail']}")
            print(f"   + phase6_gate_v1.json")
    except Exception as e:  # noqa: BLE001 — companion diagnostic, never fatal
        print(f"   ⚠ portfolio sim / gate skipped: {e}")

    # Hard exit only when --gate is set (back-compat: exploratory runs always return 0).
    if args.gate and gate_passed is False:
        print("\n❌ --gate: acceptance gate FAILED → returning nonzero (retrain NOT validated).")
        return 1
    if args.gate and gate_passed is None:
        print("\n❌ --gate: gate could not be evaluated → returning nonzero (fail-closed).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
