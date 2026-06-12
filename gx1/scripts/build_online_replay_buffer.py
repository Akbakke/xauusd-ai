#!/usr/bin/env python3
"""Build online-IQL replay buffer from live paper-runner journals.

Reads v12_paper_journal_*.jsonl files in a date range, extracts each event's
captured `entry_iql_state_v1` (197-dim raw feature vector) and computes the
counterfactual reward per [SKIP, LONG, SHORT] × K-horizon on the CEMENT M5
label grid.

Output: a single parquet bundle the warm-start trainer consumes directly.

Designed-not-forked:
  - State extraction reuses the per-poll dump we added to v12_pipeline.py
    (entry_iql_state_v1 + entry_iql_q_per_action_per_k_v1).
  - Window/excursion math IS the cement one-truth: slice_forward_window +
    compute_outcomes_for_window imported from
    materialize_build_candidate_forward_outcome_dataset_v1 — this file only
    anchors decisions and aggregates the live M1 tape to M5 via the canonical
    downsampler (v12_m1_to_m5_downsample.m1_to_m5).
  - Reward formula matches materialize_build_entry_iql_v2.build_reward_matrix
    exactly (incl. the _SYM family + bad-path posgate) so a warm-start update
    is consistent with cement training.

LABEL CONVENTION = cement_m5_v1 (2026-06-12 re-anchor fix):
  The cement labels the warm-start's weights AND the --mix-cement anchor were
  trained on are M5-GRID: entry at the NEXT M5 bar's ask_open/bid_open
  (searchsorted side='right' on decision_ts), entry bar INCLUDED in the
  excursion window, K in M5 bars, truncate-don't-drop at the first >180-min
  gap, row kept iff >=48/96 forward bars at K96. The previous buffer computed
  M1-grid rewards (decision-minute anchor, entry bar EXCLUDED, full-window
  gap-DROP) — adversarially measured mean |Δ| 54 bps vs cement on real rows,
  i.e. one Q-net trained on two label definitions. This file now mirrors the
  cement convention bar-for-bar; the meta stamps `label_convention` and
  online_iql_warmstart refuses to --mix-cement against anything else.
  (The 2026-06-12 5x M1/M5 horizon-unit fix is subsumed: K is M5-native here.)

PARITY with the ACTIVE bundle (entry_iql_volbal_20260611):
  - variant R_WAIT_OPP_K96_LAM50_SYM: true per-K rewards, wait-side penalized
    with ITS OWN MAE (like-for-like r_skip), spread_coef=0 (terminal PnL is
    already bid/ask round-trip).
  - GX1_REWARD_BADPATH_POSGATE defaults to "1" HERE (builder default is "0"):
    the volbal bundle was verified posgate-trained (reward-matrix mean
    fingerprint −54.98 ≈ logged −55.635; posgate=0 gives −17.6). Newer bundles
    stamp `reward_env_v1` in the ckpt — check it before refitting.
  - Truncation (GX1_REWARD_TRUNC_MASK parity, materialize_build_entry_iql_v2
    write_artifacts:1547-1553): windows TRUNCATE at the first >180-min gap
    (cement slice_forward_window) and the row is kept iff
    forward_bars_used_at_K96 >= 48. The old full-window gap-DROP censored
    78.6% of Friday bars the cement would have trained on.

ANTI-FORGETTING export mode (2026-06-12, --export-cement-sample):
  The nightly warm-start trains on hundreds of live rows — low LR + 2 epochs is
  the only forgetting protection. This mode exports a frozen, seeded sample of
  the CEMENT training distribution (states + rewards) in buffer-compatible
  columns so online_iql_warmstart --mix-cement can anchor the Q-net while it
  adapts. It IGNORES journals entirely: it replays the cement build's own
  load → trunc-mask → state → reward path via the ONE-TRUTH functions in
  materialize_build_entry_iql_v2 (never a re-implementation), under the same
  env the ACTIVE volbal bundle was trained with, then validates the produced
  feature names + ORDER against the contract-resolved ACTIVE entry_iql bundle.

Usage:
  python -m gx1.scripts.build_online_replay_buffer \
      --from 20260601 --to 20260607 \
      --variant R_WAIT_OPP_K96_LAM50_SYM \
      --out /home/andre2/GX1_DATA/reports/online_replay/replay_W2026_23.parquet

  python -m gx1.scripts.build_online_replay_buffer \
      --export-cement-sample \
      --forward-outcome-dir .../forward_outcome_clean/per_week \
      --sample-n 20000 --variant R_WAIT_OPP_K96_LAM50_SYM \
      --out .../cement_replay_sample_v1.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.execution.v12_counterfactual_replay import load_m1_window
from gx1.execution.v12_m1_to_m5_downsample import m1_to_m5
# ONE-TRUTH cement label construction — windowing/excursion math is IMPORTED,
# never re-implemented (the 2026-06-12 re-anchor fix exists because a parallel
# M1 implementation drifted 54 bps mean |Δ| from these exact functions).
from gx1.scripts.materialize_build_candidate_forward_outcome_dataset_v1 import (
    K_HORIZONS as CEMENT_K_HORIZONS,
    MAX_INTRA_GAP_MINUTES,
    MAX_K,
    WAIT_BARS,
    compute_outcomes_for_window,
    slice_forward_window,
)

JOURNAL_DIR = Path("/home/andre2/GX1_DATA/reports/v12_paper_runs")
# Cement K-horizons in M5 BARS — must match the ACTIVE bundle ckpt's k_horizons.
K_HORIZONS_M5_DEFAULT = tuple(CEMENT_K_HORIZONS)
M1_PER_M5 = 5
# Stamped in the buffer meta; online_iql_warmstart hard-requires it on --mix-cement.
LABEL_CONVENTION_CEMENT_M5 = "cement_m5_v1"
_M5_PRICE_COLS = ("bid_open", "bid_high", "bid_low", "bid_close",
                  "ask_open", "ask_high", "ask_low", "ask_close")
_MAX_GAP_NS = int(MAX_INTRA_GAP_MINUTES) * 60 * 1_000_000_000

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


def compute_cement_counterfactuals(
    m5_time_ns: np.ndarray,
    m5_arrays: dict[str, np.ndarray],
    ts_ns: int,
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]] | None:
    """take_now + wait counterfactuals for ONE decision ts on the CEMENT M5 grid.

    Exact cement semantics (materialize_build_candidate_forward_outcome_dataset_v1):
      anchor   — first M5 bar STRICTLY AFTER decision_ts (process_week:713,
                 searchsorted side='right'); entry fill = that bar's ask_open
                 (long) / bid_open (short) (compute_side_metrics:304/315);
      window   — [anchor, anchor+K) in M5 bars: the entry bar IS bar 0 of the
                 excursion window (slice_forward_window:248);
      truncate — at the first intra-window gap > 180 min or tape end
                 (slice_forward_window:251-267) — row KEPT, forward_bars_used
                 records the effective coverage;
      wait     — same windowing re-anchored at anchor+WAIT_BARS(=6) M5 bars
                 (process_week:721-722).
    The metric math IS the imported one-truth compute_outcomes_for_window; this
    function only anchors and slices. Returns ({side}_-prefixed cf_now, cf_wait,
    window info) or None when the anchor is past the tape end.
    """
    n_m5 = len(m5_time_ns)
    idx_start = int(np.searchsorted(m5_time_ns, ts_ns, side="right"))
    if idx_start >= n_m5:
        return None
    s_t, e_t, status_t = slice_forward_window(m5_time_ns, n_m5, idx_start, MAX_K, _MAX_GAP_NS)
    wait_idx_start = idx_start + WAIT_BARS
    s_w, e_w, status_w = slice_forward_window(m5_time_ns, n_m5, wait_idx_start, MAX_K, _MAX_GAP_NS)

    cf_now: dict[str, float] = {}
    cf_wait: dict[str, float] = {}
    for side in ("long", "short"):
        m_now = compute_outcomes_for_window(
            *(m5_arrays[c][s_t:e_t] for c in _M5_PRICE_COLS), side)
        for k, v in m_now.items():
            cf_now[f"{side}_{k}"] = v
        m_wait = compute_outcomes_for_window(
            *(m5_arrays[c][s_w:e_w] for c in _M5_PRICE_COLS), side)
        for k, v in m_wait.items():
            cf_wait[f"{side}_{k}"] = v
    info = {"m5_anchor_idx": idx_start,
            "take_now_window_status": status_t,
            "wait_window_status": status_w}
    return cf_now, cf_wait, info


def compute_rewards_for_row(
    cf_now: dict,
    cf_wait: dict,
    k_m5: int,
    lam: float,
    sym: bool,
    entry_spread_bps: float,
    bad_path_prob: float,
    posgate: bool,
) -> tuple[float, float, float]:
    """Cement-parity reward at ONE cement K (M5 bars) from cement-grid counterfactuals.

    Mirrors materialize_build_entry_iql_v2._wait_opp_at_K (581-598) + _gate_take (544-548):
      r_side  = terminal − λ·mae_before_mfe − spread_coef·spread   (coef 0 if SYM)
      wait    = wait_terminal − λ·wait_mae (SYM; unpenalized for base family)
      r_skip  = clip(max(0, max(wait_l, wait_s) − max(r_l, r_s)), 0, 500)
      takes   = clip ±500, then bad-path gate (posgate: positive part only).
    cf keys are compute_outcomes_for_window names ({side}_ prefixed, cement M5 K).
    NaN inputs → 0.0 (cement's .fillna(0.0)); a MISSING key = wrong K = hard error.
    """
    def _val(cf: dict, key: str) -> float:
        if key not in cf:
            raise KeyError(
                f"counterfactual key {key!r} missing — K={k_m5} not in cement "
                f"K_HORIZONS {CEMENT_K_HORIZONS}?"
            )
        v = float(cf[key])
        return v if np.isfinite(v) else 0.0

    tl = _val(cf_now, f"long_terminal_pnl_at_K{k_m5}")
    ts = _val(cf_now, f"short_terminal_pnl_at_K{k_m5}")
    wl = _val(cf_wait, f"long_terminal_pnl_at_K{k_m5}")
    ws = _val(cf_wait, f"short_terminal_pnl_at_K{k_m5}")
    ml = max(0.0, _val(cf_now, f"long_mae_before_mfe_bps_at_K{k_m5}"))
    ms = max(0.0, _val(cf_now, f"short_mae_before_mfe_bps_at_K{k_m5}"))

    spread_coef = 0.0 if sym else 2.0
    r_long = tl - lam * ml - spread_coef * entry_spread_bps
    r_short = ts - lam * ms - spread_coef * entry_spread_bps

    if sym:
        wml = max(0.0, _val(cf_wait, f"long_mae_before_mfe_bps_at_K{k_m5}"))
        wms = max(0.0, _val(cf_wait, f"short_mae_before_mfe_bps_at_K{k_m5}"))
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
    m5: pd.DataFrame,
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
    unknown_k = [k for k in k_horizons_m5 if k not in CEMENT_K_HORIZONS]
    if unknown_k:
        raise ValueError(
            f"--k-horizons {unknown_k} not in cement K_HORIZONS {CEMENT_K_HORIZONS} — "
            f"cement-grid labels exist only at the cement horizons."
        )

    m5_time_ns = pd.to_datetime(m5["time"], utc=True).astype("int64").to_numpy()
    m5_arrays = {c: m5[c].astype(np.float64).to_numpy() for c in _M5_PRICE_COLS}

    rows = []
    seen_ts = set()
    skipped = {"no_state": 0, "no_decision": 0, "ts_not_found": 0,
               "anchor_past_tape_end": 0, "trunc_lt48_at_K96": 0, "duplicate": 0}

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

                cf = compute_cement_counterfactuals(m5_time_ns, m5_arrays, int(ts.value))
                if cf is None:
                    skipped["anchor_past_tape_end"] += 1
                    continue
                cf_now, cf_wait, info = cf
                # Cement trunc-mask (materialize_build_entry_iql_v2.write_artifacts:
                # 1547-1553): keep iff >=48/96 forward M5 bars at K96, BOTH sides
                # (same window; symmetric check kept for cement parity). Truncated-
                # but-covered rows SURVIVE — the old full-window gap-DROP censored
                # 78.6% of Friday bars the cement would have trained on.
                bars_l = float(cf_now["long_forward_bars_used_at_K96"])
                bars_s = float(cf_now["short_forward_bars_used_at_K96"])
                if bars_l < 48 or bars_s < 48:
                    skipped["trunc_lt48_at_K96"] += 1
                    continue

                spread_bps = float(ev.get("spread_bps", 0.0) or 0.0)
                v10 = dec.get("_v10_snapshot") or {}
                bad_path_prob = float(v10.get("bad_path_prob", 0.0) or 0.0)

                adv_l = dec.get("advantage_over_skip_long")
                adv_s = dec.get("advantage_over_skip_short")
                adv_best = max(float(adv_l or 0.0), float(adv_s or 0.0))

                row = {
                    "ts_utc": ts.isoformat(),
                    "m5_anchor_idx": int(info["m5_anchor_idx"]),
                    "take_now_window_status": info["take_now_window_status"],
                    "forward_bars_used_at_K96": int(bars_l),
                    "spread_bps_at_decision": spread_bps,
                    "bad_path_prob": bad_path_prob,
                    "live_action": dec.get("action", "UNKNOWN"),
                    "live_advantage_over_skip": adv_best,
                }
                # State vector
                for i, v in enumerate(state):
                    row[f"s{i:03d}"] = float(v)
                # True per-K rewards on the cement M5 grid (r_*_K96 == cement K96 == 8h).
                for k_m5 in k_horizons_m5:
                    r_skip, r_long, r_short = compute_rewards_for_row(
                        cf_now, cf_wait, k_m5, lam, sym,
                        spread_bps, bad_path_prob, posgate,
                    )
                    row[f"r_skip_K{k_m5}"] = r_skip
                    row[f"r_long_K{k_m5}"] = r_long
                    row[f"r_short_K{k_m5}"] = r_short
                rows.append(row)

    LOG.info(f"buffer rows: {len(rows)}  skipped: {skipped}")
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def export_cement_sample(
    forward_outcome_dir: Path,
    variant: str,
    sample_n: int,
    seed: int,
    out: Path,
) -> int:
    """Export a frozen sample of the cement training distribution (states + rewards).

    Parity is the law here: same load → trunc-mask → state → reward sequence as
    the cement build (materialize_build_entry_iql_v2.write_artifacts:1540-1566),
    using its one-truth functions — this function only orders the calls, applies
    the masks the cement applied, and samples AFTER everything is built.
    """
    # Cement env — forced BEFORE importing materialize_build_entry_iql_v2:
    #   GX1_BUILD_IQL_V2=0  read at module IMPORT (line 395); the live 197-dim
    #                     state = V1 mode (185 numeric + 12 one-hot — verified
    #                     vs volbal feature_names_v1); =1 would swap in the
    #                     297-col V2 set and fail the dead-zero guard on this
    #                     (non-augmented) forward-outcome frame;
    #   GX1_REGIME_V4=1   pins the regime one-hot categories/ORDER at
    #                     build_state_matrix:1043-1052 (cement order LOW/MEDIUM/
    #                     HIGH/EXTREME — sorted(observed) would reorder them);
    #   POSGATE/TRUNC_MASK are the volbal reward env (posgate fingerprint-
    #                     verified — see module header). Conflicting pre-set
    #                     values = hard error, never a silent override.
    cement_env = {
        "GX1_BUILD_IQL_V2": "0",
        "GX1_REGIME_V4": "1",
        "GX1_REWARD_BADPATH_POSGATE": "1",
        "GX1_REWARD_TRUNC_MASK": "1",
    }
    for k, v in cement_env.items():
        cur = os.environ.get(k)
        if cur is not None and cur != v:
            raise RuntimeError(
                f"env conflict: {k}={cur!r} is set but cement-sample export requires {v!r} "
                f"(ACTIVE volbal bundle parity). Unset it before --export-cement-sample."
            )
        os.environ[k] = v
    LOG.info(f"cement-sample env FORCED (volbal cement parity): {cement_env}")

    from gx1.scripts.materialize_build_entry_iql_v2 import (
        K_HORIZONS,
        build_reward_matrix,
        build_state_matrix,
        load_forward_outcome_dataset,
    )
    from gx1_guards.artifacts import load_decision_artifact

    LOG.info(f"loading forward-outcome dataset from {forward_outcome_dir}")
    df = load_forward_outcome_dataset(forward_outcome_dir)
    n_loaded = len(df)
    # Trunc-mask EXACTLY as the cement applied it (write_artifacts:1547-1553):
    # drop Friday/week-boundary rows with <48/96 forward bars at K96.
    _bl = pd.to_numeric(df.get("take_now_long_forward_bars_used_at_K96_v1"), errors="coerce")
    _bs = pd.to_numeric(df.get("take_now_short_forward_bars_used_at_K96_v1"), errors="coerce")
    _keep = (_bl.fillna(0) >= 48) & (_bs.fillna(0) >= 48)
    LOG.info(f"trunc-mask (GX1_REWARD_TRUNC_MASK=1): dropping {int((~_keep).sum()):,}/{n_loaded:,} "
             f"truncated rows (<48/96 forward bars at K96)")
    df = df[_keep].reset_index(drop=True)

    # Pre-apply build_state_matrix's completeness mask (its lines 945-952) so X
    # and R are built from the SAME rows: build_state_matrix drops incomplete
    # rows INTERNALLY, which would silently misalign X against an R built from
    # the unmasked df. Pre-masking makes the internal drop a no-op.
    if "_canonical_features_complete_v1" in df.columns:
        complete = df["_canonical_features_complete_v1"].fillna(False).astype(bool)
        if "_chunk0_features_complete_v1" in df.columns:
            complete &= df["_chunk0_features_complete_v1"].fillna(False).astype(bool)
        if (~complete).any():
            LOG.info(f"completeness mask: dropping {int((~complete).sum()):,} incomplete-feature rows")
            df = df[complete].reset_index(drop=True)

    # Build state + reward on the FULL df, sample AFTER — build_state_matrix's
    # attach_group_a_dip_struct_ctx_columns computes rolling/window features
    # whose values CHANGE if computed on a subsample (NEVER subsample first).
    X, feature_names = build_state_matrix(df)
    if X.shape[0] != len(df):
        raise RuntimeError(
            f"state/reward row misalignment: build_state_matrix returned {X.shape[0]} rows "
            f"for a {len(df)}-row df despite pre-masking — investigate before exporting."
        )
    LOG.info(f"state matrix: {X.shape}, features={len(feature_names)}")
    # R shape (n, 3, n_K); action layout per build_reward_matrix:500-509:
    # 0=SKIP, 1=TAKE_LONG, 2=TAKE_SHORT — the exact columns buffer_to_tensors
    # (online_iql_warmstart.py:74-77) reads back as r_skip/r_long/r_short.
    R = build_reward_matrix(df, variant=variant)
    LOG.info(f"reward matrix {variant}: shape={R.shape} mean={float(R.mean()):.3f}")

    # rule 8: the sample anchors a warm-start of the ACTIVE entry_iql bundle —
    # verify the state contract (names AND order; a same-width reorder would
    # pass a dim check silently) against the contract-resolved bundle summary.
    bundle = load_decision_artifact("entry_iql")
    bundle_names = json.loads((bundle / "summary_v1.json").read_text())["feature_names_v1"]
    if list(feature_names) != list(bundle_names):
        diffs = [(i, a, b) for i, (a, b) in enumerate(zip(feature_names, bundle_names)) if a != b]
        raise RuntimeError(
            f"state contract mismatch vs ACTIVE entry_iql bundle {bundle.name}: built "
            f"{len(feature_names)} features vs bundle {len(bundle_names)}; first order diffs: "
            f"{diffs[:5]} — wrong env or wrong forward-outcome wave for this bundle."
        )
    LOG.info(f"state contract verified vs ACTIVE bundle {bundle.name}: "
             f"{len(feature_names)} features, names+order identical")

    n = len(df)
    take = min(int(sample_n), n)
    rng = np.random.default_rng(seed)                      # seeded — reproducible sample
    idx = np.sort(rng.choice(n, size=take, replace=False))  # sorted = keep chronology

    out_cols: dict[str, np.ndarray] = {}
    for i in range(X.shape[1]):
        out_cols[f"s{i:03d}"] = X[idx, i]
    for ki, k in enumerate(K_HORIZONS):
        out_cols[f"r_skip_K{k}"] = R[idx, 0, ki]
        out_cols[f"r_long_K{k}"] = R[idx, 1, ki]
        out_cols[f"r_short_K{k}"] = R[idx, 2, ki]
    out_df = pd.DataFrame(out_cols)
    out_df["source"] = "cement"
    ts_col = next((c for c in ("decision_ts_utc", "decision_ts", "time") if c in df.columns), None)
    if ts_col is not None:
        out_df["ts_utc"] = pd.to_datetime(df[ts_col].iloc[idx], utc=True).astype(str).to_numpy()
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)

    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "mode": "cement_sample_export_v1",
        "variant": variant,
        "sym": variant.endswith("_SYM"),
        "k_horizons_m5": list(K_HORIZONS),
        "n_rows": int(take),
        "n_source_rows_after_masks": int(n),
        "n_loaded_rows": int(n_loaded),
        "sample_seed": int(seed),
        "source": "cement",
        "forward_outcome_dir": str(forward_outcome_dir),
        "active_bundle_validated": str(bundle),
        "state_dim": int(X.shape[1]),
        "feature_names": list(feature_names),
        "reward_env_v1": {
            "GX1_REWARD_BADPATH_POSGATE": "1",
            "GX1_REWARD_TRUNC_MASK": "1",
        },
    }, indent=2))
    LOG.info(f"wrote {take:,} cement-sample rows → {out}")
    LOG.info(f"wrote metadata → {meta_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="date_from", type=str, default=None,
                   help="Start date YYYYMMDD (inclusive). Required unless --export-cement-sample.")
    p.add_argument("--to", dest="date_to", type=str, default=None,
                   help="End date YYYYMMDD (inclusive). Required unless --export-cement-sample.")
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
    # ANTI-FORGETTING export mode (header): journals ignored, cement sample out.
    p.add_argument("--export-cement-sample", action="store_true",
                   help="Export a frozen sample of the CEMENT training distribution "
                        "(states+rewards, buffer-compatible) for --mix-cement anchoring.")
    p.add_argument("--forward-outcome-dir", type=Path, default=None,
                   help="per_week dir of the cement forward-outcome dataset "
                        "(required with --export-cement-sample)")
    p.add_argument("--sample-n", type=int, default=20000,
                   help="Cement-sample rows to export (seeded, sampled AFTER full build)")
    p.add_argument("--seed", type=int, default=20260612,
                   help="RNG seed for the cement sample (reproducible, never wall-clock)")
    args = p.parse_args()

    if args.export_cement_sample:
        if args.forward_outcome_dir is None:
            p.error("--export-cement-sample requires --forward-outcome-dir")
        return export_cement_sample(
            forward_outcome_dir=args.forward_outcome_dir,
            variant=args.variant,
            sample_n=args.sample_n,
            seed=args.seed,
            out=args.out,
        )
    if not args.date_from or not args.date_to:
        p.error("--from/--to are required (unless --export-cement-sample)")

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
    # longest cement K-horizon + wait anchor, in minutes (M5 bars × 5).
    tail_min = (max(k_horizons_m5) + WAIT_BARS) * M1_PER_M5 + 60
    start_ts = pd.Timestamp(args.date_from, tz="UTC")
    end_ts = (pd.Timestamp(args.date_to, tz="UTC") + pd.Timedelta(days=1)
              + pd.Timedelta(minutes=tail_min))
    LOG.info(f"loading M1 window: {start_ts} → {end_ts}")
    m1 = load_m1_window(start_ts, end_ts)
    if m1.empty:
        LOG.error("M1 window empty — check that collector parquets + M1 tape cover the date range")
        return 4
    # Cement labels live on the M5 grid — aggregate through the ONE-TRUTH
    # canonical M5 downsampler (v12_m1_to_m5_downsample.m1_to_m5: open=first/
    # high=max/low=min/close=last per 5-min UTC floor; every provably-complete
    # bucket emitted, only the live tape-edge suppressed — same rule that
    # builds the canonical M5 tape). tape_end=None is correct here: the slice
    # ends at the loaded tail, beyond every journal decision's K192+wait window.
    m5 = m1_to_m5(m1)
    if m5.empty:
        LOG.error("M5 aggregation empty — M1 window too sparse for full 5-min buckets")
        return 4
    LOG.info(f"M5 bars: {len(m5):,} (from {len(m1):,} M1)  "
             f"first={m5['time'].iloc[0]}  last={m5['time'].iloc[-1]}")

    df = build_buffer(journals, m5, args.variant, k_horizons_m5, posgate)
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
        # cement_m5_v1 convention stamp — online_iql_warmstart hard-requires it
        # before mixing these rows with the cement anchor sample.
        "label_convention": LABEL_CONVENTION_CEMENT_M5,
        "entry_anchor": "next_m5_bar_open (searchsorted side='right' on decision_ts; "
                        "long=ask_open, short=bid_open)",
        "entry_bar_included": True,
        "gap_rule": "truncate>=48",
        "wait_bars_m5": WAIT_BARS,
        "max_intra_gap_minutes": MAX_INTRA_GAP_MINUTES,
        "reward_env_v1": {
            "GX1_REWARD_BADPATH_POSGATE": "1" if posgate else "0",
            "GX1_REWARD_TRUNC_MASK": "1 (cement parity: forward_bars_used_at_K96 >= 48)",
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
