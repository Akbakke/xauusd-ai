#!/usr/bin/env python3
"""Build exit-IQL V3-M1: 2-action multi-head Q(s, a, K) on per-M1-bar samples.

Mission
-------
M1-cadence successor to ``materialize_build_exit_iql_v2.py``. The v2 trainer
runs over M5-cadence per-bar samples (decision every 20 minutes); this v3-M1
trainer runs over M1-cadence per-bar samples (decision every minute) so the
override head can match V3 transformer's per-M1 inference cadence.

Two architectural deltas vs v2:

  1. K_HORIZONS = [5, 20, 60, 120, 240, 480] M1-bars
     (same wall-clock horizons as v2's [1, 4, 12, 24, 48, 96] M5-bars).
     K_PRIMARY = 120 (2h M1) ≡ v2's K_PRIMARY = 24 (2h M5).

  2. LAZY-JOIN data loading. The v2_m1 builder writes thin per-week parquets
     (~65 cols, no chunk_0 / canonical denorm). This trainer joins those at
     load time via merge_asof on bar_ts_ns_v1 → floor_to_M5(bar_ts_ns_v1):
       - chunk_0 features: per-week (one parquet per truth-week)
       - canonical_features_v1: global (one parquet for all years)
     The merge_asof direction is "backward" so each M1 bar gets the most
     recent M5-aligned context — same semantics as the live exit_manager.

Action space (2)
----------------
  HOLD, EXIT_NOW

Reward variants
---------------
Inherited from v2: R_NET_REAL, R_GATED, R_REGRET (R_RAW + R_NET05 omitted —
they degenerate to "always HOLD" because their HOLD reward = max future MFE
with no realization cost).

This is RESEARCH-ONLY. No runtime promotion.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.scripts import (  # noqa: E402
    materialize_build_exit_iql_per_bar_dataset_v2_m1 as exit_pipe_m1,
)
from gx1.scripts import (  # noqa: E402
    materialize_build_candidate_forward_outcome_dataset_v1 as fwd_pipe,
)
from gx1.scripts import exit_iql_artifact_primitives_v1 as contract_gate  # noqa: E402
from gx1.scripts import materialize_build_exit_iql_v2 as v2_train  # noqa: E402


# ---------------------------------------------------------------------------
# M1 configuration overrides
# ---------------------------------------------------------------------------

ACTION = "BUILD_EXIT_IQL_V3_M1"
DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
DEFAULT_CANONICAL_FEATURES_PATH = fwd_pipe.DEFAULT_CANONICAL_FEATURES_PATH

# K horizons — M1 units (same wall-clock as v2's M5 [1,4,12,24,48,96]).
K_HORIZONS = list(exit_pipe_m1.K_HORIZONS_EXIT_M1)  # [5, 20, 60, 120, 240, 480]
N_K = len(K_HORIZONS)
K_PRIMARY = 120  # 2h in M1 — same as v2's K_PRIMARY=24 (2h in M5)

# M1-specific extra features written by the v2_m1 builder.
M1_MICRO_COLS = [
    "m1_last_5bar_return_bps_v1",
    "m1_last_15bar_return_bps_v1",
    "m1_last_60bar_return_bps_v1",
    "m1_realized_vol_15bar_bps_v1",
    "m1_realized_vol_60bar_bps_v1",
]
M5_PHASE_COLS = [f"m5_phase_{i}_v1" for i in range(5)]

# V3 v8 transformer outputs (added by score_v3_v8_on_per_bar_v1.py).
# These are present only in V8AUG per-bar datasets; build_state_matrix
# gracefully handles missing columns by filling with 0.0. So including
# them here is a no-op for V6 datasets but adds value for V8AUG.
V3_V8_OUTPUT_COLS = [
    "v3_v8_should_exit_prob",
    "v3_v8_profit_protect_prob",
    "v3_v8_family_argmax",
    "v3_v8_family_logit_max",
]

# Per-bar trade-state columns — extends v2's list with M1-specific features
# and V3 v8 transformer outputs.
NUMERIC_STATE_COLS_PER_BAR = (
    list(v2_train.NUMERIC_STATE_COLS_PER_BAR)
    + M1_MICRO_COLS
    + M5_PHASE_COLS
    + V3_V8_OUTPUT_COLS
)

# Candidate / chunk0 / canonical / derivative / entry-snapshot column lists
# come from v2 unchanged — chunk0 / canonical are populated by lazy-join.
NUMERIC_STATE_COLS_CANDIDATE = list(v2_train.NUMERIC_STATE_COLS_CANDIDATE)
ONE_HOT_COLS = list(v2_train.ONE_HOT_COLS)

# B2/H7 fix (vedtak_b2h7_exit_features_20260604): the 64 volume/group_a/dip_struct features the
# V7/V8 exit contract DECLARES were never materialised into the per-bar dataset (Exit-IQL trained
# on zeros). They are not stored anywhere — computed on-the-fly from OHLC by one-truth helpers.
# Compute them via the SAME helpers the entry build + live serve use, on the raw M5 tape (the
# canonical loader drops OHLC), and merge_asof onto each per-bar row (M5 decision-context
# broadcast across the M1 window — matches exit_io_v7 semantics). Flag-gated, default OFF =
# cement bit-parity.
_AUG64_ENABLED = os.environ.get("GX1_EXIT_AUGMENT_64", "0") == "1"
try:
    from gx1.contracts.signal_bridge_v3 import (
        ORDERED_CTX_CONT_DIP_STRUCT as _DS64,
    )
    from gx1.contracts.signal_bridge_v3 import (
        ORDERED_CTX_CONT_GROUP_A_PARITY as _GA64,
    )
    from gx1.features.volume_features import VOLUME_FEATURE_NAMES as _VOL64

    NUMERIC_STATE_COLS_AUG64 = list(_VOL64) + list(_GA64) + list(_DS64)  # 4 + 24 + 36 = 64
except Exception:  # noqa: BLE001 — import-time guard; absent only on a broken tree
    NUMERIC_STATE_COLS_AUG64 = []


# ── DEFERRAL RELABEL (user vedtak EXIT_IQL_DEFERRAL_RELABEL_20260707, default OFF) ──────────────
# Teach the Q-net the value of DEFERRING a premature Strategy-F hand-rule exit. At every row where
# the ONE-TRUTH Strategy-F fire formulas trigger (profit_lock | breakeven_cut — constants imported
# from gx1.execution.v12_exit_iql_live, so an env-pinned build uses the exact live operating point),
# the HOLD reward is RELABELED to the realized continuation value:
#     r_hold(K) = hold_max_pnl_K{K}_v1 - GAMMA*spread     (alpha->1.0, MAE-penalty->0, fire bars only)
# EXIT reward is untouched everywhere ("reward for EXIT on a correct fire unchanged"). Premature
# fires (real continuation ahead) get HOLD >> EXIT; correct fires (no continuation) keep EXIT >= HOLD
# — the separation is purely realized-outcome data, no classifier, no new hand rule. All K horizons
# are <= 240 M1 bars == the binding deferral horizon cap (serve mirror: GX1_STRATEGY_F_DEFER_CAP_BARS).
# Relabeling covers ALL trigger bars (not only first fires): serve re-consults the strong-hold veto
# on every trigger bar, and first-fire rows are state-indistinguishable from later trigger rows — an
# MSE fit on conflicting targets would average a first-fire-only relabel away. Pregate evidence:
# strategyf_lwr_pregate_20260707 (OOT AUC 0.74-0.88 both directions). Default OFF = cement bit-parity.
_DEFERRAL_RELABEL = os.environ.get("GX1_EXIT_DEFERRAL_RELABEL", "0") == "1"


def _compute_exit_aug64(canonical_path: Path) -> "pd.DataFrame | None":
    """Compute the 64 declared exit features (volume/group_a/dip_struct) on the raw M5 tape via
    the one-truth helpers; returns `_time_ns` + the 64 base-named cols for merge_asof. Flag-gated."""
    if not _AUG64_ENABLED:
        return None
    if not canonical_path.exists():
        raise RuntimeError(f"[{ACTION}] GX1_EXIT_AUGMENT_64=1 but canonical tape missing: {canonical_path}")
    from gx1.features.volume_features import add_volume_features
    from gx1.features.htf_features import build_multi_tf_per_bar_features_v2
    from gx1.scripts.augment_forward_outcome_v2 import attach_group_a_dip_struct_ctx_columns
    tape = pd.read_parquet(canonical_path)  # full cols incl OHLC + volume + smc_swing_state
    tape["time"] = pd.to_datetime(tape["time"], utc=True)
    required_source = ["open", "high", "low", "close", "volume", "smc_swing_state"]
    missing_source = [name for name in required_source if name not in tape.columns]
    if missing_source:
        raise RuntimeError(f"[{ACTION}] canonical aug64 sources missing: {missing_source}")
    mtf_source = tape.set_index("time")[required_source[:5]].copy()
    multi_tf = build_multi_tf_per_bar_features_v2(mtf_source)
    tape = add_volume_features(tape)
    tape = attach_group_a_dip_struct_ctx_columns(
        tape,
        journal_label="exit_aug64",
        multi_tf=multi_tf,
    )
    missing = [c for c in NUMERIC_STATE_COLS_AUG64 if c not in tape.columns]
    if missing:
        raise RuntimeError(
            f"[{ACTION}] aug64 helpers did not produce {len(missing)}/64 cols: {missing[:10]} "
            f"— verify OHLC/volume/smc_swing_state on the tape.")
    tape["_time_ns"] = tape["time"].astype("int64")
    return tape[["_time_ns"] + NUMERIC_STATE_COLS_AUG64].copy()


# ---------------------------------------------------------------------------
# Lazy-join data loader
# ---------------------------------------------------------------------------


def _suffix_chunk0(chunk0: pd.DataFrame | None) -> pd.DataFrame | None:
    """Rename chunk_0 feature columns to add CHUNK0_SUFFIX, ready for merge."""
    if chunk0 is None or len(chunk0) == 0:
        return None
    rename_map = {
        c: f"{c}{fwd_pipe.CHUNK0_SUFFIX}"
        for c in fwd_pipe.CHUNK0_FEATURE_COLS
        if c in chunk0.columns
    }
    cols = ["_time_ns"] + list(rename_map.keys())
    out = chunk0[cols].rename(columns=rename_map).copy()
    return out


def _suffix_canonical(canonical: pd.DataFrame | None) -> pd.DataFrame | None:
    """Rename canonical feature columns to add CANONICAL_FEATURES_SUFFIX."""
    if canonical is None or len(canonical) == 0:
        return None
    feat_cols = [c for c in canonical.columns if c != "_time_ns"]
    rename_map = {c: f"{c}{fwd_pipe.CANONICAL_FEATURES_SUFFIX}" for c in feat_cols}
    cols = ["_time_ns"] + feat_cols
    out = canonical[cols].rename(columns=rename_map).copy()
    return out


def _merge_asof_features(df: pd.DataFrame, feats: pd.DataFrame | None) -> pd.DataFrame:
    """Join the exact last-closed M5 state at each closed M1 decision."""
    if feats is None or len(feats) == 0:
        raise RuntimeError("EXIT_IQL_M5_FEATURE_SOURCE_MISSING")
    left = df.copy()
    bar_ns = pd.to_numeric(
        left["bar_ts_ns_v1"],
        errors="raise",
    ).to_numpy(dtype=np.int64)
    minute_ns = 60 * 1_000_000_000
    five_minutes_ns = 5 * minute_ns
    decision_available_ns = bar_ns + minute_ns
    left["_expected_m5_time_ns"] = (
        (decision_available_ns // five_minutes_ns) * five_minutes_ns
        - five_minutes_ns
    )
    right = feats.copy()
    right["_time_ns"] = pd.to_numeric(
        right["_time_ns"],
        errors="raise",
    ).astype(np.int64)
    if right["_time_ns"].duplicated().any():
        raise RuntimeError("EXIT_IQL_M5_FEATURE_SOURCE_DUPLICATE_TIME")
    merged = left.merge(
        right,
        how="left",
        left_on="_expected_m5_time_ns",
        right_on="_time_ns",
        validate="many_to_one",
        indicator=True,
    )
    if not bool((merged["_merge"] == "both").all()):
        missing = merged.loc[
            merged["_merge"] != "both",
            "_expected_m5_time_ns",
        ].iloc[0]
        raise RuntimeError(
            "EXIT_IQL_M5_FEATURE_EXACT_TIME_MISSING: "
            f"{pd.Timestamp(int(missing), tz='UTC')}"
        )
    feature_columns = [
        column
        for column in right.columns
        if column != "_time_ns"
    ]
    numeric = merged[feature_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("EXIT_IQL_M5_FEATURE_NONFINITE")
    merged = merged.drop(
        columns=["_expected_m5_time_ns", "_time_ns", "_merge"]
    )
    return merged


def load_per_bar_dataset_lazy_join(
    per_week_dir: Path,
    *,
    reports_root: Path,
    canonical_path: Path,
    sample_n_rows: int | None = None,
    seed: int = 20260501,
) -> pd.DataFrame:
    """Load per-week M1 parquets, sub-sample, lazy-join chunk_0 + canonical."""
    parquets = sorted(per_week_dir.glob("exit_per_bar_m1_*.parquet"))
    if not parquets:
        raise RuntimeError(f"[{ACTION}] no exit_per_bar_m1_*.parquet found in {per_week_dir}")

    print(f"[{ACTION}] loading canonical features from {canonical_path}", flush=True)
    canonical = fwd_pipe._load_canonical_features(canonical_path)
    canonical_suf = _suffix_canonical(canonical)
    if canonical_suf is not None:
        print(f"[{ACTION}] canonical: {len(canonical_suf):,} rows × "
              f"{len(canonical_suf.columns) - 1} features (suffixed)", flush=True)
    else:
        # FAIL-CLOSED (2026-06-03 audit): canonical None -> the ~76 canonical state cols
        # (63% of the state matrix) would silently zero-fill -> a wrongly-trained Exit-IQL.
        raise RuntimeError(
            f"[{ACTION}] canonical features unavailable ({canonical_path}) — refusing to "
            f"build Exit-IQL state on a ~63%-zero-filled matrix. Fix the canonical input.")

    aug64 = _compute_exit_aug64(canonical_path)
    if aug64 is None or len(aug64) == 0:
        raise RuntimeError(
            f"[{ACTION}] AUG64 feature source unavailable ({canonical_path}) — "
            "refusing to build an incomplete Exit-IQL state."
        )
    print(f"[{ACTION}] aug64: computed {len(NUMERIC_STATE_COLS_AUG64)} declared exit features "
          f"on the M5 tape (one-truth helpers) — will exact-join onto per-bar rows", flush=True)

    rng = np.random.default_rng(seed)
    per_file_target: int | None = None
    if sample_n_rows is not None and sample_n_rows > 0:
        per_file_target = max(1, int(sample_n_rows / len(parquets)) + 1)
        print(f"[{ACTION}] sub-sample target: ~{per_file_target} rows per week × {len(parquets)} weeks", flush=True)

    parts: list[pd.DataFrame] = []
    weeks_with_chunk0 = 0
    for p_idx, p in enumerate(parquets, start=1):
        df_p = pd.read_parquet(p)
        if len(df_p) == 0:
            continue
        if per_file_target is not None and len(df_p) > per_file_target:
            idx = rng.choice(len(df_p), size=per_file_target, replace=False)
            df_p = df_p.iloc[idx].reset_index(drop=True)

        # Derive week dir from parquet filename: exit_per_bar_m1_TRUTH_MONFRI_WEEK_*.parquet
        week_name = p.stem.removeprefix("exit_per_bar_m1_")
        week_dir = reports_root / week_name
        chunk0 = fwd_pipe._load_chunk0_features(week_dir)
        chunk0_suf = _suffix_chunk0(chunk0)
        if chunk0_suf is not None:
            df_p = _merge_asof_features(df_p, chunk0_suf)
            weeks_with_chunk0 += 1
        else:
            raise RuntimeError(
                f"[{ACTION}] chunk_0 feature source unavailable for {week_name} — "
                "refusing to manufacture an incomplete Exit-IQL state."
            )

        if canonical_suf is not None:
            df_p = _merge_asof_features(df_p, canonical_suf)
        df_p = _merge_asof_features(df_p, aug64)

        parts.append(df_p)
        if p_idx % 25 == 0 or p_idx == len(parquets):
            running = sum(len(part) for part in parts)
            print(f"[{ACTION}] [{p_idx}/{len(parquets)}] loaded weeks; running rows={running:,}", flush=True)

    print(f"[{ACTION}] weeks with exact chunk_0: {weeks_with_chunk0}", flush=True)
    df = pd.concat(parts, ignore_index=True)
    if sample_n_rows is not None and len(df) > sample_n_rows:
        df = df.sample(n=sample_n_rows, random_state=seed).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Reward / state matrices — reuse v2's logic but with M1 K_HORIZONS
# ---------------------------------------------------------------------------


def build_reward_matrix(df: pd.DataFrame, *, variant: str) -> np.ndarray:
    """M1 reward matrix builder. Identical math to v2 but K_HORIZONS in M1 units."""
    n = len(df)
    R = np.zeros((n, v2_train.N_ACTIONS_EXIT, N_K), dtype=np.float32)

    exit_now = pd.to_numeric(df["exit_now_reward_bps_v1"], errors="coerce").fillna(0.0).to_numpy()
    current_mae = pd.to_numeric(df["current_mae_bps_v1"], errors="coerce").fillna(0.0).to_numpy()
    drawdown = pd.to_numeric(df["pnl_drawdown_from_peak_v1"], errors="coerce").fillna(0.0).to_numpy()
    spread = (
        pd.to_numeric(df.get("entry_spread_bps"), errors="coerce")
        .fillna(1.5).clip(lower=0.0).astype(float).to_numpy()
        if "entry_spread_bps" in df.columns
        else np.full(n, 1.5, dtype=float)
    )
    # V9 Issue C: bars_held for R_NET_V2 capital cost.
    bars_held_v2 = (
        pd.to_numeric(df.get("bars_in_trade_v1"), errors="coerce").fillna(0.0).to_numpy()
        if "bars_in_trade_v1" in df.columns else np.zeros(n, dtype=float)
    )

    # DEFERRAL RELABEL fire mask (vedtak EXIT_IQL_DEFERRAL_RELABEL_20260707; see module comment).
    fire_mask = None
    if _DEFERRAL_RELABEL:
        from gx1.execution.v12_exit_iql_live import (  # one-truth Strategy-F constants (env-pinned)
            BREAKEVEN_MIN_MFE,
            BREAKEVEN_RATIO,
            MFE_GIVEBACK_MIN_MFE_BPS,
            MFE_GIVEBACK_PCT,
        )
        _mfe = pd.to_numeric(df["current_mfe_bps_v1"], errors="coerce").fillna(0.0).to_numpy()
        _pnl = pd.to_numeric(df["current_unrealized_pnl_bps_v1"], errors="coerce").fillna(0.0).to_numpy()
        _dd = np.maximum(0.0, _mfe - _pnl)
        _profit_lock = (_mfe >= MFE_GIVEBACK_MIN_MFE_BPS) & (_dd >= MFE_GIVEBACK_PCT * _mfe) & (_mfe > 0)
        _breakeven = (_mfe >= BREAKEVEN_MIN_MFE) & (_pnl < BREAKEVEN_RATIO * _mfe)
        fire_mask = _profit_lock | _breakeven
        print(f"[{ACTION}] DEFERRAL_RELABEL({variant}): {int(fire_mask.sum()):,}/{n:,} Strategy-F "
              f"trigger bars ({100.0 * float(fire_mask.mean()):.1f}%) -> r_hold(K) = realized "
              f"continuation (hold_K - gamma*spread); r_exit unchanged", flush=True)

    # R_PEAK_QUALITY family (ported from v2 builder, ONE-truth math): scale-invariant
    # giveback-quality needs the running MFE peak. Fail-closed (rule 9 / 2026-06-03 audit):
    # refuse a degraded reward if the peak column is absent rather than silently filling 0
    # (which would collapse quality->0 and force a degenerate always-EXIT policy).
    peak_mfe_safe = None
    if variant in ("R_PEAK_QUALITY", "R_PEAK_QUALITY_QUAD"):
        if "current_mfe_bps_v1" not in df.columns:
            raise RuntimeError(f"[{ACTION}] reward '{variant}' needs 'current_mfe_bps_v1' "
                               f"(running MFE peak) — refusing to substitute a degraded reward.")
        peak_mfe = pd.to_numeric(df["current_mfe_bps_v1"], errors="coerce").fillna(0.0).to_numpy()
        peak_mfe_safe = np.maximum(peak_mfe, 1.0)  # avoid div0; if no peak yet, quality≈1

    for ki, K in enumerate(K_HORIZONS):
        hold_col = f"hold_max_pnl_K{K}_v1"
        # FAIL-CLOSED (2026-06-03 audit): if the HOLD-reward column is missing it must NOT
        # silently fall back to the EXIT reward (collapses HOLD vs EXIT-NOW into the same
        # target -> degenerate policy). The dataset must carry hold_max_pnl_K{K}_v1.
        if hold_col not in df.columns:
            raise RuntimeError(f"[{ACTION}] HOLD-reward column '{hold_col}' missing from "
                               f"dataset — refusing to substitute the EXIT reward.")
        hold_K = pd.to_numeric(df[hold_col], errors="coerce").fillna(0.0).to_numpy()

        if variant == "R_NET_REAL":
            r_hold = (v2_train.R_NET_REAL_ALPHA * hold_K
                      - v2_train.R_NET_REAL_BETA * np.abs(current_mae)
                      - v2_train.R_NET_REAL_GAMMA * spread)
            r_exit = exit_now - v2_train.R_NET_REAL_GAMMA * spread
        elif variant == "R_NET_V2":
            # R_NET_REAL + giveback + bars cost (incremental over R_NET_REAL).
            r_hold = (
                v2_train.R_NET_REAL_ALPHA * hold_K
                - v2_train.R_NET_REAL_BETA * np.abs(current_mae)
                - v2_train.R_NET_REAL_GAMMA * spread
                - v2_train.R_NET_V2_GIVEBACK * drawdown
                - v2_train.R_NET_V2_BARS_COST * bars_held_v2
            )
            r_exit = exit_now - v2_train.R_NET_REAL_GAMMA * spread
        elif variant == "R_GATED":
            r_hold = np.where(hold_K > v2_train.GATED_THRESHOLD_BPS, hold_K, 0.0)
            r_exit = exit_now
        elif variant == "R_REGRET":
            oracle = np.maximum(exit_now, hold_K)
            r_hold = hold_K - oracle
            r_exit = exit_now - oracle
        elif variant in ("R_PEAK_QUALITY", "R_PEAK_QUALITY_QUAD"):
            # Scale-invariant exit-quality (ported from v2 builder :508-521, identical math).
            # quality = clip(1 - drawdown_from_peak / peak_mfe, 0, 1): at peak quality=1 (full
            # HOLD reward); full giveback quality=0 (-> EXIT_NOW triggered). Self-balancing —
            # makes giveback-protection LEARNED, the smart-AI replacement for the hardcoded
            # Strategy-F overlay (no tuned giveback/bars_cost lambdas).
            quality = np.clip(1.0 - drawdown / peak_mfe_safe, 0.0, 1.0)
            if variant == "R_PEAK_QUALITY_QUAD":
                quality = quality ** 2  # steeper falloff on giveback
            r_hold = hold_K * quality - v2_train.R_NET_REAL_GAMMA * spread
            r_exit = exit_now - v2_train.R_NET_REAL_GAMMA * spread
        else:
            raise ValueError(f"unknown reward variant for M1 trainer: {variant}")

        if fire_mask is not None:
            # Deferral relabel: at Strategy-F trigger bars the HOLD reward IS the realized
            # continuation value net of the same spread cost as EXIT (K <= 240 == horizon cap).
            r_hold = np.where(fire_mask, hold_K - v2_train.R_NET_REAL_GAMMA * spread, r_hold)

        r_hold = np.clip(r_hold, -500.0, 500.0)
        r_exit = np.clip(r_exit, -500.0, 500.0)
        R[:, v2_train.ACTION_HOLD_ID, ki] = r_hold.astype(np.float32)
        R[:, v2_train.ACTION_EXIT_NOW_ID, ki] = r_exit.astype(np.float32)
    return R


def maybe_year_sample_weights(df: pd.DataFrame) -> "np.ndarray | None":
    """Per-row loss weight by calendar year of decision_ts_utc (mean-normalised to 1.0).

    GX1_EXIT_IQL_YEAR_WEIGHT="2026:4.0[,2025:1.5]" upweights named years so a decayed recent
    regime (pregate: 2026 AUC decay) carries more gradient than its ~7.6% row share. Unset =
    None = cement bit-parity. Fail-closed on a missing decision_ts_utc column. Composes
    multiplicatively with maybe_per_trade_sample_weights in write_artifacts.
    (Vedtak EXIT_IQL_DEFERRAL_RELABEL_20260707 — documented 2026-slice weighting.)"""
    spec = os.environ.get("GX1_EXIT_IQL_YEAR_WEIGHT", "").strip()
    if not spec:
        return None
    if "decision_ts_utc" not in df.columns:
        raise RuntimeError(
            "GX1_EXIT_IQL_YEAR_WEIGHT set but 'decision_ts_utc' is absent from the per-bar df — "
            "cannot compute year weights; fix the dataset or unset the flag.")
    wmap: dict[int, float] = {}
    for part in spec.split(","):
        y_s, w_s = part.split(":")
        wmap[int(y_s)] = float(w_s)
    years = pd.to_datetime(df["decision_ts_utc"], utc=True).dt.year.to_numpy()
    w = np.ones(len(df), dtype=np.float64)
    for y, wv in wmap.items():
        w[years == y] = wv
    n_up = int((w != 1.0).sum())
    w = w * (float(len(w)) / float(w.sum()))  # mean-normalise (loss magnitude / eff-LR unchanged)
    print(f"[{ACTION}] YEAR_WEIGHT {wmap}: {n_up:,}/{len(w):,} rows re-weighted "
          f"(mean-normalised)", flush=True)
    return w.astype(np.float32)


def _maybe_warmstart_state_dicts(
    variant: str, fold_id: str, feature_names: list[str],
) -> "tuple[dict | None, dict | None]":
    """WARM-START source resolution (vedtak EXIT_IQL_DEFERRAL_RELABEL_20260707, default OFF).

    GX1_EXIT_IQL_WARMSTART_FROM_CONTRACT=1 -> init Q/V weights from the ACTIVE exit_iql bundle
    resolved via gx1_guards.load_decision_entry('exit_iql') (rule 8 — NEVER a hardcoded path).
    Fail-closed guards: (a) the warm-start bundle's feature_names_v1 must be IDENTICAL (same
    order — weights are input-order-bound) to this build's; (b) arch (hidden/n_hidden) must match
    the effective training budget; (c) a missing per-(variant,fold) checkpoint is a hard error,
    never a silent cold start."""
    if os.environ.get("GX1_EXIT_IQL_WARMSTART_FROM_CONTRACT", "0") != "1":
        return None, None
    import torch
    from gx1_guards.artifacts import load_decision_entry
    entry = load_decision_entry("exit_iql")
    ws_root = Path(entry["path"])
    ws_summary = json.loads((ws_root / "summary_v1.json").read_text())
    ws_feats = list(ws_summary.get("feature_names_v1") or [])
    if ws_feats != list(feature_names):
        diff_a = [c for c in ws_feats if c not in feature_names][:5]
        diff_b = [c for c in feature_names if c not in ws_feats][:5]
        raise RuntimeError(
            f"[{ACTION}] WARM-START feature-name mismatch vs ACTIVE bundle {ws_root.name}: "
            f"{len(ws_feats)} vs {len(feature_names)} names (bundle-only={diff_a}, "
            f"build-only={diff_b}) — weights are input-order-bound; refusing a silently "
            f"mis-mapped warm start.")
    ckpt_path = ws_root / "trained_models_v1" / f"{variant}_{fold_id}.pt"
    if not ckpt_path.exists():
        raise RuntimeError(f"[{ACTION}] WARM-START checkpoint missing: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if int(ck.get("hidden_dim", -1)) != int(v2_train.TRAIN_HIDDEN_DIM) or \
            int(ck.get("n_hidden", -1)) != int(v2_train.TRAIN_N_HIDDEN):
        raise RuntimeError(
            f"[{ACTION}] WARM-START arch mismatch: ckpt hidden={ck.get('hidden_dim')}/"
            f"n_hidden={ck.get('n_hidden')} vs budget {v2_train.TRAIN_HIDDEN_DIM}/"
            f"{v2_train.TRAIN_N_HIDDEN} — pick the matching --budget or disable warm start.")
    print(f"[{ACTION}] WARM-START init from {ckpt_path} (contract-resolved ACTIVE exit_iql)",
          flush=True)
    return ck["q_state_dict"], ck["v_state_dict"]


def build_state_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """M1 state matrix builder. Drops v2's completion-flag filter (lazy-join always populates)."""
    parts: list[pd.DataFrame] = []
    feature_names: list[str] = []
    nan_warnings: list[tuple[str, float]] = []
    missing_cols: list[str] = []
    constant_cols: list[str] = []
    for c in NUMERIC_STATE_COLS_PER_BAR + NUMERIC_STATE_COLS_CANDIDATE + (
        NUMERIC_STATE_COLS_AUG64 if _AUG64_ENABLED else []
    ):
        if c in df.columns:
            col = pd.to_numeric(df[c], errors="coerce")
        else:
            # FAIL-CLOSED (2026-06-03 audit): a required state column missing means a
            # corrupt/wrong dataset (e.g. canonical block absent -> 63% zero-fill). Do
            # NOT silently zero-fill the model's input; collect and raise below.
            missing_cols.append(c)
            col = pd.Series(np.zeros(len(df)), index=df.index)
        nan_frac = float(col.isna().mean()) if len(col) > 0 else 0.0
        if nan_frac > 0.05:
            nan_warnings.append((c, nan_frac))
        if c not in missing_cols and float(col.fillna(0.0).std()) < 1e-4:
            constant_cols.append(c)  # P4: silent-ignore/zeroed-handoff guard (raised below)
        parts.append(col.fillna(0.0).rename(c))
        feature_names.append(c)
    # EX3 (2026-06-04 train==serve parity): PIN the regime one-hot categories to the FULL fixed label set so
    # the build column-set is deterministic + matches the live serve (v12_exit_iql_live.py:425-428), even when
    # a category is absent in the data. get_dummies emits OBSERVED-only -> the degenerate trend_regime_id
    # (const=1) emits ONLY trend_regime_TREND_NEUTRAL, dropping TREND_UP/DOWN; serve always sets all 3, so a
    # retrained model would get an all-zero trend block live it never saw in training (regime-blind). Activates
    # at the regime retrain (cement bundle unchanged; the adapter is by-name so col order is irrelevant).
    _ONE_HOT_PIN = {
        "vol_regime": ["LOW", "MEDIUM", "HIGH", "EXTREME"],
        "trend_regime": ["TREND_UP", "TREND_NEUTRAL", "TREND_DOWN"],
    }
    for c in ONE_HOT_COLS:
        if c in df.columns:
            dummies = pd.get_dummies(df[c].astype(str), prefix=c, dummy_na=False)
            if c in _ONE_HOT_PIN:
                dummies = dummies.reindex(columns=[f"{c}_{lbl}" for lbl in _ONE_HOT_PIN[c]], fill_value=0)
            parts.append(dummies)
            feature_names.extend(dummies.columns.tolist())
    # FAIL-CLOSED (2026-06-03 audit): missing required cols or >5% NaN = degraded substrate.
    # Was print-only; now hard RuntimeError. A clean
    # build off the repaired substrate has neither, so this only fires on a bad dataset.
    if missing_cols:
        raise RuntimeError(
            f"[{ACTION}] {len(missing_cols)} required state columns MISSING from dataset "
            f"(would be silently zero-filled): {missing_cols[:20]}"
            f"{' ...' if len(missing_cols) > 20 else ''}")
    if nan_warnings:
        raise RuntimeError(
            f"[{ACTION}] {len(nan_warnings)} feature(s) exceed 5% NaN (degraded substrate): "
            f"{[(c, round(f, 4)) for c, f in nan_warnings[:20]]}"
            f"{' ...' if len(nan_warnings) > 20 else ''}")
    # P4 (2026-06-06): a CONSTANT state col is fail-closed UNLESS on the one-truth feature-liveness
    # allowlist — a non-allowlisted constant = a silent-ignore / zeroed-handoff regression (e.g. a V10
    # head or the V3 score collapsing to const). Mirrors the Entry-IQL build P4 guard.
    if constant_cols:
        from gx1.audit.feature_liveness import KNOWN_ALLOWED_DEAD
        _new_const = [c for c in constant_cols if c not in KNOWN_ALLOWED_DEAD]
        if _new_const:
            raise RuntimeError(
                f"[{ACTION}] {len(_new_const)} state feature(s) CONSTANT (std~0) and NOT on the "
                f"feature-liveness allowlist — silent-ignore/zeroed-handoff regression: {_new_const[:20]}"
                f"{' ...' if len(_new_const) > 20 else ''}. Fix the hand-off or add to KNOWN_ALLOWED_DEAD with a reason.")
    X = pd.concat(parts, axis=1).astype(np.float32).to_numpy()
    return X, feature_names


# ---------------------------------------------------------------------------
# Orchestration — delegate fold-evaluation to v2 helpers, swap K_HORIZONS
# ---------------------------------------------------------------------------


def write_artifacts(
    per_bar_dir: Path,
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    canonical_path: Path = DEFAULT_CANONICAL_FEATURES_PATH,
    out_root: Path | None = None,
    sample_n_rows: int | None = None,
    variants_subset: list[str] | None = None,
    built_at_utc: str | None = None,
    k_primary_override: int | None = None,
    exit_reward_multiplier: float = 1.0,
) -> dict[str, Any]:
    timestamp = built_at_utc or v2_train._stamp()
    artifact_root = out_root or (DEFAULT_REPORTS_ROOT / f"{ACTION}_{timestamp}_LOCK")
    artifact_root.mkdir(parents=True, exist_ok=True)

    per_week_dir = per_bar_dir / "per_week"
    if not per_week_dir.exists():
        return {
            "artifact_root": str(artifact_root),
            "summary": {
                "final_status_v1": "EXIT_IQL_V3_M1_BLOCKED_BY_INPUT_LOCK_MISSING",
                "next_action_v1": "REPAIR_EXIT_IQL_V3_M1_BEFORE_FURTHER_WORK",
                "missing_input_v1": str(per_week_dir),
            },
        }

    print(f"[{ACTION}] loading per-bar M1 dataset from {per_week_dir} (lazy-join)", flush=True)
    df = load_per_bar_dataset_lazy_join(
        per_week_dir,
        reports_root=reports_root,
        canonical_path=canonical_path,
        sample_n_rows=sample_n_rows,
    )
    print(f"[{ACTION}] loaded rows={len(df):,} cols={len(df.columns)}", flush=True)

    X, feature_names = build_state_matrix(df)
    print(f"[{ACTION}] state matrix: {X.shape}, features={len(feature_names)}", flush=True)

    variants = variants_subset or v2_train.REWARD_VARIANTS
    R_by_variant: dict[str, np.ndarray] = {}
    for variant in variants:
        R_by_variant[variant] = build_reward_matrix(df, variant=variant)
        print(f"[{ACTION}] reward matrix {variant}: shape={R_by_variant[variant].shape} "
              f"mean={float(R_by_variant[variant].mean()):.3f}", flush=True)

    # V9 Issue 2: optional K_PRIMARY override (default 120 = 2h M1).
    # Wave 2 final used K=120 and Exit-IQL was active only 3.8% — most trades hit
    # forced terminal. Test K=240 (4h) to see if longer horizon lets Exit-IQL
    # learn meaningful EXIT_NOW decisions on multi-hour trends.
    effective_k_primary = int(k_primary_override) if k_primary_override is not None else K_PRIMARY
    if effective_k_primary not in K_HORIZONS:
        raise ValueError(f"k_primary_override={effective_k_primary} not in K_HORIZONS={K_HORIZONS}")

    # V9 Issue 2: optional reward rebalance — boost EXIT_NOW reward to make it
    # more attractive when the model is over-conservatively HOLD-biased.
    if exit_reward_multiplier != 1.0:
        for variant in variants:
            R_by_variant[variant][:, v2_train.ACTION_EXIT_NOW_ID, :] *= float(exit_reward_multiplier)
        print(f"[{ACTION}] applied exit_reward_multiplier={exit_reward_multiplier} to EXIT_NOW reward",
              flush=True)

    # Patch v2's K_HORIZONS / K_PRIMARY for the duration of this run so its
    # downstream helpers (derive_oracle_action, evaluate_one_fold,
    # build_stratified_folds, determine_status, write_*) operate over M1 K's.
    saved_k_horizons = v2_train.K_HORIZONS
    saved_n_k = v2_train.N_K
    saved_k_primary = v2_train.K_PRIMARY
    v2_train.K_HORIZONS = K_HORIZONS
    v2_train.N_K = N_K
    v2_train.K_PRIMARY = effective_k_primary
    try:
        R_for_strat = R_by_variant.get("R_NET_REAL", next(iter(R_by_variant.values())))
        K_primary_idx = K_HORIZONS.index(effective_k_primary)
        oracle_action = v2_train.derive_oracle_action(R_for_strat, K_primary_idx)
        oracle_dist = {ai: int((oracle_action == ai).sum()) for ai in range(v2_train.N_ACTIONS_EXIT)}
        print(f"[{ACTION}] oracle action (R_NET_REAL K={effective_k_primary}) distribution: {oracle_dist}", flush=True)

        folds = v2_train.resolve_exit_folds(
            df, n_rows=len(df), oracle_action=oracle_action,  # R16: GX1_EXIT_IQL_SPLIT_MODE = stratified(cement)|group|chronological
        )

        per_fold_results: list[dict[str, Any]] = []
        flat_evaluations: list[dict[str, Any]] = []
        _exit_sample_weights = v2_train.maybe_per_trade_sample_weights(df)  # EXIT-8 part 2 (None = cement)
        _year_weights = maybe_year_sample_weights(df)  # vedtak EXIT_IQL_DEFERRAL_RELABEL_20260707 (None = cement)
        if _year_weights is not None:
            if _exit_sample_weights is None:
                _exit_sample_weights = _year_weights
            else:  # compose multiplicatively, re-mean-normalise
                _w = _exit_sample_weights.astype(np.float64) * _year_weights.astype(np.float64)
                _exit_sample_weights = (_w * (float(len(_w)) / float(_w.sum()))).astype(np.float32)
        _exit_loss_mask = v2_train.maybe_degenerate_loss_mask(df, v2_train.N_ACTIONS_EXIT, list(K_HORIZONS))  # EXIT-9
        for variant in variants:
            for fold in folds:
                _init_q, _init_v = _maybe_warmstart_state_dicts(
                    variant, fold["fold_id_v1"], feature_names)
                r = v2_train.evaluate_one_fold(
                    fold, X, R_by_variant[variant], oracle_action,
                    variant=variant, artifact_root=artifact_root,
                    feature_names=feature_names,
                    sample_weights=_exit_sample_weights,
                    loss_mask=_exit_loss_mask,
                    init_q_state_dict=_init_q,
                    init_v_state_dict=_init_v,
                )
                per_fold_results.append(r)
                flat_evaluations.extend(r.get("all_evaluations_v1", []))

        v2_train._write_rows(artifact_root / "per_fold_per_variant_evaluations_v1.csv", flat_evaluations)
        status, next_action, recommendation, headline = v2_train.determine_status(per_fold_results)
    finally:
        v2_train.K_HORIZONS = saved_k_horizons
        v2_train.N_K = saved_n_k
        v2_train.K_PRIMARY = saved_k_primary

    # Re-label v2 status strings for the M1 layer (best-effort string substitution)
    status = status.replace("EXIT_IQL_V2", "EXIT_IQL_V3_M1")
    next_action = next_action.replace("EXIT_IQL_V2", "EXIT_IQL_V3_M1")

    summary = {
        "layer_name": "EXIT_IQL_V3_M1_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": v2_train._utc_now(),
        "cadence_v1": "M1",
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "headline_v1": headline,
        "input_per_bar_dir_v1": str(per_bar_dir),
        "n_rows_v1": int(len(df)),
        "n_features_v1": int(X.shape[1]),
        "feature_names_v1": feature_names,
        "feature_names_sha256_v1": contract_gate.ordered_feature_names_sha256(
            feature_names
        ),
        "k_horizons_v1": K_HORIZONS,
        "k_primary_v1": K_PRIMARY,
        "n_actions_v1": v2_train.N_ACTIONS_EXIT,
        "action_labels_v1": v2_train.ACTION_LABELS_EXIT,
        "reward_variants_v1": list(R_by_variant.keys()),
        "oracle_action_distribution_v1": oracle_dist,
        "n_folds_v1": v2_train.N_FOLDS,
        "seed_v1": v2_train.SEED_V1,
        "per_fold_summary_v1": [
            {"fold_id_v1": r["fold_id_v1"], "variant_v1": r["variant_v1"],
             "best_combo_v1": r["best_combo_v1"], "best_test_metric_v1": r["best_test_metric_v1"],
             "val_class_guard_status_v1": r["val_class_guard_status_v1"],
             "val_action_counts_v1": r["val_action_counts_v1"],
             "test_baselines_v1": r["test_baselines_v1"]}
            for r in per_fold_results
        ],
        "research_only_v1": True,
        "iql_production_allowed_v1": False,
        # Wave-flag provenance (vedtak EXIT_IQL_DEFERRAL_RELABEL_20260707); all default-OFF.
        "deferral_relabel_v1": bool(_DEFERRAL_RELABEL),
        "warmstart_from_contract_v1": os.environ.get("GX1_EXIT_IQL_WARMSTART_FROM_CONTRACT", "0") == "1",
        "year_weight_spec_v1": os.environ.get("GX1_EXIT_IQL_YEAR_WEIGHT", "") or None,
    }
    v2_train._write_json(artifact_root / "summary_v1.json", summary)
    v2_train._write_json(artifact_root / "status_v1.json", {
        "layer_name": "EXIT_IQL_V3_M1_STATUS_V1",
        "status_v1": "MATERIALIZED_RESEARCH_ONLY_GATE",
        "final_status_v1": status, "next_action_v1": next_action,
    })
    v2_train._write_json(artifact_root / "manifest_v1.json", {
        "layer_id_v1": ACTION,
        "built_at_utc_v1": summary["built_at_utc_v1"],
        "output_dir_v1": str(artifact_root),
        "artifact_paths_v1": {
            "summary": str(artifact_root / "summary_v1.json"),
            "status": str(artifact_root / "status_v1.json"),
            "per_fold_csv": str(artifact_root / "per_fold_per_variant_evaluations_v1.csv"),
            "trained_models_dir": str(artifact_root / "trained_models_v1"),
        },
    })
    return {"artifact_root": str(artifact_root), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Materialize {ACTION}.")
    parser.add_argument("--per-bar-dir", type=str, required=True,
                        help="Path to BUILD_EXIT_IQL_PER_BAR_DATASET_V2_M1_*_LOCK directory")
    parser.add_argument("--reports-root", type=str, default=str(DEFAULT_REPORTS_ROOT))
    parser.add_argument("--canonical-features", type=str, default=str(DEFAULT_CANONICAL_FEATURES_PATH))
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--sample-n-rows", type=int, default=None)
    parser.add_argument("--variants", type=str, default=None,
                        help="Comma-separated subset of REWARD_VARIANTS "
                             "(R_NET_REAL,R_NET_V2,R_GATED,R_REGRET,R_PEAK_QUALITY,R_PEAK_QUALITY_QUAD)")
    parser.add_argument("--budget", type=str, default="fast",
                        choices=list(v2_train.BUDGET_PRESETS.keys()))
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override the preset batch. The tiny IQL MLP trains via a manual minibatch "
                             "loop, so a LARGE batch (>=4096) is mandatory — 256 was a 16x throughput "
                             "leak (2026-06-10). Presets now default to 4096; raise further if VRAM allows.")
    parser.add_argument("--k-primary", type=int, default=None,
                        help=f"V9 Issue 2: K_PRIMARY override in M1 bars. "
                             f"Allowed values: {K_HORIZONS}. Default 120 (2h). "
                             f"Try 240 (4h) or 480 (8h) for longer-trend trades.")
    parser.add_argument("--exit-reward-multiplier", type=float, default=1.0,
                        help="V9 Issue 2: multiply EXIT_NOW reward by this factor. "
                             "Wave 2 final showed Exit-IQL active only 3.8%% (96.2%% forced "
                             "terminal). Try 1.2-1.5 to push EXIT_NOW above HOLD when "
                             "outcomes are similar.")
    parser.add_argument("--built-at-utc", type=str, default=None)
    parser.add_argument("--vedtak", type=str, default=None,
                        help="REQUIRED retrain vedtak (gx1_guards gate). Short reason string.")
    args = parser.parse_args()

    # Retrain-vedtak gate (no auto-retrains).
    try:
        from gx1_guards.gates import require_retrain_vedtak, GateError
        try:
            require_retrain_vedtak(args.vedtak)
        except GateError as e:
            parser.error(str(e))
    except ImportError:
        if not args.vedtak:
            parser.error("--vedtak is required (gx1_guards unavailable; pass --vedtak anyway).")

    # Apply training budget into v2 globals (same pattern as v2 main).
    preset = v2_train.BUDGET_PRESETS[args.budget]
    v2_train.TRAIN_EPOCHS_Q = preset["epochs_q"]
    v2_train.TRAIN_EPOCHS_V = preset["epochs_v"]
    v2_train.TRAIN_K_VQ_ITERATIONS = preset["k_iter"]
    v2_train.TRAIN_BATCH_SIZE = preset["batch"]
    v2_train.TRAIN_HIDDEN_DIM = preset["hidden"]
    v2_train.TRAIN_N_HIDDEN = preset["n_hidden"]
    if args.batch_size:  # explicit override (tiny IQL => big batch; SMART+MAXED 2026-06-10)
        v2_train.TRAIN_BATCH_SIZE = int(args.batch_size)
    print(f"[{ACTION}] training budget '{args.budget}': {preset} | effective batch="
          f"{v2_train.TRAIN_BATCH_SIZE}", flush=True)

    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else None
    variants_subset = args.variants.split(",") if args.variants else None
    result = write_artifacts(
        per_bar_dir=Path(args.per_bar_dir).expanduser().resolve(),
        reports_root=Path(args.reports_root).expanduser().resolve(),
        canonical_path=Path(args.canonical_features).expanduser().resolve(),
        out_root=out_root,
        sample_n_rows=args.sample_n_rows,
        variants_subset=variants_subset,
        built_at_utc=args.built_at_utc,
        k_primary_override=args.k_primary,
        exit_reward_multiplier=args.exit_reward_multiplier,
    )
    print(json.dumps(v2_train._jsonable(result["summary"]), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
