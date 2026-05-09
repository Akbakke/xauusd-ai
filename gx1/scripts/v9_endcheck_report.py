"""
V9 endcheck report generator.

Takes a Phase 7 LOCK directory and produces a comprehensive evaluation report:
  - Overall metrics (PnL, win/loss rate, action distribution)
  - Action-level breakdown (LONG/SHORT/SKIP)
  - Exit-reason breakdown (FORCED_TERMINAL vs EXIT_IQL_SIGNAL)
  - Side / hour / vol_regime / session slicing
  - Survivorship analysis on EXIT_IQL_SIGNAL fires
  - **Q-advantage filter sweep** (virtual, no Phase 7 re-runs)
  - Comparison vs wave 2 baseline (if available)

Outputs:
  - <out>/v9_endcheck_report.md  (human-readable markdown)
  - <out>/v9_endcheck_metrics.json (structured data)

Usage:
  python -m gx1.scripts.v9_endcheck_report \\
      --phase7-dir /path/to/JOINT_*_LOCK \\
      [--baseline-dir /path/to/wave2_phase7_LOCK] \\
      [--out /path/to/out_dir]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


WAVE2_BASELINE_DIR = Path(
    "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/"
    "JOINT_ENTRY_EXIT_IQL_VALIDATION_GATE_V2_20260506T133947Z_LOCK"
)
QADV_THRESHOLDS = [0.0, 3.6, 6.8, 10.4, 15.1, 18.8, 25.5, 35.0]

# Forward-outcome dataset (same population as Phase 7 candidates → high join rate).
FORWARD_OUTCOME_DIR = Path(
    "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/CANDIDATE_FORWARD_OUTCOME_RUN/v1_full"
)

# Entry-bar feature columns to join from forward-outcome parquet.
# These are all market-state at trade-decision moment.
ENTRY_FEATURE_COLS = [
    "atr_bps", "entry_spread_bps",
    "session", "vol_regime", "trend_regime",
    "hour_utc", "weekday_utc",
    "p_long", "p_short", "p_hat", "margin", "uncertainty_score",
    "tradable_prob", "bad_path_prob",
    "mfe_first_n_pred", "path_quality_pred",
    "direction_logit_long", "direction_logit_short", "direction_logit_flat",
]


def load_phase7(phase7_dir: Path) -> Dict[str, Any]:
    summary = json.loads((phase7_dir / "summary_v1.json").read_text())
    df = pd.read_csv(phase7_dir / "per_candidate_joint_eval_v1.csv")
    return {"summary": summary, "df": df, "dir": phase7_dir}


def join_entry_features(df: pd.DataFrame, forward_outcome_dir: Path) -> pd.DataFrame:
    """Join per-candidate eval with entry-time features from forward-outcome dataset.

    forward-outcome covers the SAME population as Phase 7 candidates (both
    derive from the inference batch), so high join rate expected.

    Returns df augmented with entry-time market state suffixed _ent.
    """
    per_week = forward_outcome_dir / "per_week"
    if not per_week.exists():
        print(f"[v9-endcheck] WARN forward-outcome dir not found: {per_week}", flush=True)
        return df
    parquets = sorted(per_week.glob("*.parquet"))
    if not parquets:
        print(f"[v9-endcheck] WARN no forward-outcome parquets in {per_week}", flush=True)
        return df

    candidate_uids = set(df["candidate_uid_v1"].astype(str))
    entry_parts = []
    cols_to_load = ["candidate_uid"] + ENTRY_FEATURE_COLS
    for p in parquets:
        try:
            fwd = pd.read_parquet(p, columns=cols_to_load)
        except Exception:
            try:
                fwd = pd.read_parquet(p)
            except Exception:
                continue
        # Filter to candidates of interest (saves memory)
        fwd = fwd[fwd["candidate_uid"].astype(str).isin(candidate_uids)]
        if len(fwd) > 0:
            keep_cols = ["candidate_uid"] + [c for c in ENTRY_FEATURE_COLS if c in fwd.columns]
            entry_parts.append(fwd[keep_cols].copy())
    if not entry_parts:
        print(f"[v9-endcheck] WARN no matching forward-outcome rows found", flush=True)
        return df
    entry_df = pd.concat(entry_parts, ignore_index=True)
    entry_df = entry_df.drop_duplicates("candidate_uid", keep="first")
    n_features = len([c for c in ENTRY_FEATURE_COLS if c in entry_df.columns])
    print(f"[v9-endcheck] joined {len(entry_df):,} candidates × {n_features} entry features",
          flush=True)
    # Suffix to avoid collision with per-candidate cols
    rename = {c: f"{c}_ent" for c in ENTRY_FEATURE_COLS if c in entry_df.columns}
    entry_df = entry_df.rename(columns=rename)
    merged = df.merge(
        entry_df, left_on="candidate_uid_v1", right_on="candidate_uid", how="left"
    )
    return merged


def compute_overall(df: pd.DataFrame) -> Dict[str, Any]:
    n_total = len(df)
    pnl_mean = float(df["joint_pnl_bps_v1"].mean())
    pnl_total = float(df["joint_pnl_bps_v1"].sum())
    n_skip = int((df["entry_action_label_v1"] == "SKIP").sum())
    n_take = n_total - n_skip
    pnl_per_taken = pnl_total / n_take if n_take > 0 else 0.0
    losers = df[df["joint_pnl_bps_v1"] < 0]
    wins = df[df["joint_pnl_bps_v1"] > 0]
    return {
        "n_total": n_total,
        "n_take": n_take,
        "n_skip": n_skip,
        "pnl_mean_bps": pnl_mean,
        "pnl_total_bps": pnl_total,
        "pnl_per_taken_bps": pnl_per_taken,
        "win_rate_overall": float(len(wins) / n_total),
        "loss_rate_overall": float(len(losers) / n_total),
        "loser_total_bps": float(losers["joint_pnl_bps_v1"].sum()),
        "loser_mean_bps": float(losers["joint_pnl_bps_v1"].mean()) if len(losers) else 0.0,
    }


def compute_action_breakdown(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out = {}
    for label, sub in df.groupby("entry_action_label_v1"):
        out[label] = {
            "count": int(len(sub)),
            "mean_pnl_bps": float(sub["joint_pnl_bps_v1"].mean()),
            "total_pnl_bps": float(sub["joint_pnl_bps_v1"].sum()),
            "win_rate": float((sub["joint_pnl_bps_v1"] > 0).mean()),
        }
    return out


def compute_exit_reason_breakdown(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out = {}
    for reason, sub in df.groupby("exit_reason_v1"):
        out[reason] = {
            "count": int(len(sub)),
            "mean_pnl_bps": float(sub["joint_pnl_bps_v1"].mean()),
            "median_pnl_bps": float(sub["joint_pnl_bps_v1"].median()),
            "p25_bps": float(sub["joint_pnl_bps_v1"].quantile(0.25)),
            "p75_bps": float(sub["joint_pnl_bps_v1"].quantile(0.75)),
            "loss_rate": float((sub["joint_pnl_bps_v1"] < 0).mean()),
        }
    return out


def compute_side_breakdown(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out = {}
    if "side_used_v1" not in df.columns:
        return out
    for side, sub in df.groupby("side_used_v1"):
        if str(side) not in ("long", "short"):
            continue
        out[str(side)] = {
            "count": int(len(sub)),
            "mean_pnl_bps": float(sub["joint_pnl_bps_v1"].mean()),
            "win_rate": float((sub["joint_pnl_bps_v1"] > 0).mean()),
        }
    return out


def compute_hour_breakdown(df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
    if "decision_ts_utc" not in df.columns:
        return {}
    hours = pd.to_datetime(df["decision_ts_utc"]).dt.hour
    out = {}
    for h in range(24):
        sub = df[hours == h]
        if len(sub) > 20:
            out[int(h)] = {
                "count": int(len(sub)),
                "mean_pnl_bps": float(sub["joint_pnl_bps_v1"].mean()),
            }
    return out


def compute_strat_breakdown(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out = {}
    if "vol_regime" not in df.columns or "session" not in df.columns:
        return out
    for (vol, sess), sub in df.groupby(["vol_regime", "session"]):
        out[f"{vol}|{sess}"] = {
            "count": int(len(sub)),
            "mean_pnl_bps": float(sub["joint_pnl_bps_v1"].mean()),
            "win_rate": float((sub["joint_pnl_bps_v1"] > 0).mean()),
        }
    return out


def compute_exit_iql_survivorship(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if "exit_reason_v1" not in df.columns or "exit_bar_v1" not in df.columns:
        return {}
    exit_iql = df[df["exit_reason_v1"] == "EXIT_IQL_SIGNAL"]
    if len(exit_iql) == 0:
        return {}
    buckets = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 96)]
    out = {}
    for lo, hi in buckets:
        sub = exit_iql[(exit_iql["exit_bar_v1"] >= lo) & (exit_iql["exit_bar_v1"] < hi)]
        if len(sub) > 0:
            out[f"{lo}-{hi}"] = {
                "count": int(len(sub)),
                "mean_pnl_bps": float(sub["joint_pnl_bps_v1"].mean()),
            }
    return out


def _percentile_bins(s: pd.Series, n_bins: int = 5) -> pd.Series:
    """Bin a numeric series into n_bins quantile bins. Returns labels P0-P20, etc."""
    try:
        return pd.qcut(s, q=n_bins, duplicates="drop", labels=False)
    except Exception:
        return pd.Series([np.nan] * len(s), index=s.index)


def compute_conditional_breakdown(df: pd.DataFrame, col: str, *, n_bins: int = 5,
                                   is_categorical: bool = False) -> Dict[str, Dict[str, float]]:
    """Compute mean PnL + win rate sliced by a feature.

    For continuous features: bin into n_bins quantile buckets.
    For categorical: use raw values.
    """
    if col not in df.columns:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    if is_categorical:
        for val, sub in df.groupby(col, dropna=False):
            key = str(val) if not pd.isna(val) else "(nan)"
            if len(sub) < 10:
                continue
            out[key] = {
                "count": int(len(sub)),
                "mean_pnl_bps": float(sub["joint_pnl_bps_v1"].mean()),
                "win_rate": float((sub["joint_pnl_bps_v1"] > 0).mean()),
                "loss_rate": float((sub["joint_pnl_bps_v1"] < 0).mean()),
            }
    else:
        ser = pd.to_numeric(df[col], errors="coerce")
        bins = _percentile_bins(ser, n_bins=n_bins)
        for b_idx in range(n_bins):
            sub_mask = (bins == b_idx)
            sub = df[sub_mask]
            if len(sub) < 10:
                continue
            lo = float(ser[sub_mask].min())
            hi = float(ser[sub_mask].max())
            key = f"P{b_idx*100//n_bins:02d}-P{(b_idx+1)*100//n_bins:02d} [{lo:.2f}-{hi:.2f}]"
            out[key] = {
                "count": int(len(sub)),
                "mean_pnl_bps": float(sub["joint_pnl_bps_v1"].mean()),
                "win_rate": float((sub["joint_pnl_bps_v1"] > 0).mean()),
                "loss_rate": float((sub["joint_pnl_bps_v1"] < 0).mean()),
            }
    return out


def compute_cross_breakdown(df: pd.DataFrame, col_a: str, col_b: str) -> Dict[str, Dict[str, float]]:
    """Cross-tabulate two categorical features. Returns 'a|b' → metrics."""
    if col_a not in df.columns or col_b not in df.columns:
        return {}
    out = {}
    for (va, vb), sub in df.groupby([col_a, col_b], dropna=False):
        if len(sub) < 20:
            continue
        key = f"{va}|{vb}"
        out[key] = {
            "count": int(len(sub)),
            "mean_pnl_bps": float(sub["joint_pnl_bps_v1"].mean()),
            "win_rate": float((sub["joint_pnl_bps_v1"] > 0).mean()),
        }
    return out


def compute_conditional_analyses(df: pd.DataFrame) -> Dict[str, Any]:
    """Build all conditional breakdowns from joined entry features."""
    out = {}
    out["by_atr_bps_ent"] = compute_conditional_breakdown(df, "atr_bps_ent", n_bins=5)
    out["by_entry_spread_bps_ent"] = compute_conditional_breakdown(df, "entry_spread_bps_ent", n_bins=5)
    out["by_p_long_ent"] = compute_conditional_breakdown(df, "p_long_ent", n_bins=5)
    out["by_tradable_prob_ent"] = compute_conditional_breakdown(df, "tradable_prob_ent", n_bins=5)
    out["by_bad_path_prob_ent"] = compute_conditional_breakdown(df, "bad_path_prob_ent", n_bins=5)
    out["by_path_quality_pred_ent"] = compute_conditional_breakdown(df, "path_quality_pred_ent", n_bins=5)
    out["by_mfe_first_n_pred_ent"] = compute_conditional_breakdown(df, "mfe_first_n_pred_ent", n_bins=5)
    if "v3_v8_should_exit_prob_ent" in df.columns:
        out["by_v3_v8_should_exit_at_entry"] = compute_conditional_breakdown(
            df, "v3_v8_should_exit_prob_ent", n_bins=5)
    out["by_session_ent"] = compute_conditional_breakdown(df, "session_ent", is_categorical=True)
    out["by_vol_regime_ent"] = compute_conditional_breakdown(df, "vol_regime_ent", is_categorical=True)
    out["by_trend_regime_ent"] = compute_conditional_breakdown(df, "trend_regime_ent", is_categorical=True)
    out["by_weekday_ent"] = compute_conditional_breakdown(df, "weekday_utc_ent", is_categorical=True)
    out["session_x_side"] = compute_cross_breakdown(df, "session_ent", "side_ent")
    out["regime_x_session"] = compute_cross_breakdown(df, "vol_regime_ent", "session_ent")
    return out


def compute_qadv_sweep(df: pd.DataFrame, thresholds: List[float]) -> List[Dict[str, Any]]:
    """Virtual Q-advantage filter sweep — no Phase 7 re-runs needed.

    For each threshold T, treat trades where:
      action == SKIP OR entry_advantage_over_skip < T → as filtered (PnL = 0)
      else → keep as-is.
    """
    n_total = len(df)
    if "entry_advantage_over_skip_bps_v1" not in df.columns:
        return []
    advantage = pd.to_numeric(df["entry_advantage_over_skip_bps_v1"], errors="coerce").fillna(0.0)
    actions = df["entry_action_label_v1"]
    pnl = df["joint_pnl_bps_v1"]

    rows = []
    for T in thresholds:
        # A trade is "kept" if it's not SKIP and advantage >= T
        is_take = (actions != "SKIP")
        is_above = advantage >= T
        keep_mask = is_take & is_above
        n_keep = int(keep_mask.sum())
        if n_keep == 0:
            rows.append({
                "threshold": T, "n_take": 0, "n_skip": int(n_total - n_keep),
                "pnl_total_bps": 0.0, "pnl_per_taken_bps": 0.0,
                "pnl_per_total_bps": 0.0,
                "win_rate_taken": 0.0, "loss_rate_taken": 0.0,
            })
            continue
        kept_pnl = pnl[keep_mask]
        rows.append({
            "threshold": float(T),
            "n_take": n_keep,
            "n_skip": int(n_total - n_keep),
            "pnl_total_bps": float(kept_pnl.sum()),
            "pnl_per_taken_bps": float(kept_pnl.mean()),
            "pnl_per_total_bps": float(kept_pnl.sum() / n_total),
            "win_rate_taken": float((kept_pnl > 0).mean()),
            "loss_rate_taken": float((kept_pnl < 0).mean()),
        })
    return rows


def render_markdown(metrics: Dict[str, Any], baseline_metrics: Optional[Dict[str, Any]] = None) -> str:
    """Render metrics as markdown report."""
    lines: List[str] = []
    lines.append("# V9 Endcheck Report")
    lines.append("")
    lines.append(f"**Source**: `{metrics['source_dir']}`")
    lines.append(f"**Generated**: {metrics['generated_at']}")
    lines.append("")

    o = metrics["overall"]
    lines.append("## Overall")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| n_candidates | {o['n_total']:,} |")
    lines.append(f"| n_take | {o['n_take']:,} |")
    lines.append(f"| n_skip | {o['n_skip']:,} |")
    lines.append(f"| **PnL/total** | **{o['pnl_mean_bps']:.2f} bps** |")
    lines.append(f"| **PnL/taken** | **{o['pnl_per_taken_bps']:.2f} bps** |")
    lines.append(f"| Total PnL | {o['pnl_total_bps']:,.0f} bps |")
    lines.append(f"| Win rate (overall) | {o['win_rate_overall']*100:.2f}% |")
    lines.append(f"| Loss rate (overall) | {o['loss_rate_overall']*100:.2f}% |")
    if baseline_metrics:
        bo = baseline_metrics["overall"]
        delta_total = o["pnl_mean_bps"] - bo["pnl_mean_bps"]
        delta_taken = o["pnl_per_taken_bps"] - bo["pnl_per_taken_bps"]
        lines.append("")
        lines.append("**vs baseline**:")
        lines.append(f"- PnL/total: {bo['pnl_mean_bps']:.2f} → {o['pnl_mean_bps']:.2f} ({delta_total:+.2f})")
        lines.append(f"- PnL/taken: {bo['pnl_per_taken_bps']:.2f} → {o['pnl_per_taken_bps']:.2f} ({delta_taken:+.2f})")
    lines.append("")

    lines.append("## Action breakdown")
    lines.append("")
    lines.append("| Action | Count | Mean PnL | Total PnL | Win rate |")
    lines.append("|---|---|---|---|---|")
    for label, m in metrics["action_breakdown"].items():
        lines.append(f"| {label} | {m['count']:,} | {m['mean_pnl_bps']:.2f} | {m['total_pnl_bps']:,.0f} | {m['win_rate']*100:.1f}% |")
    lines.append("")

    if metrics["exit_reason_breakdown"]:
        lines.append("## Exit reason breakdown")
        lines.append("")
        lines.append("| Reason | Count | Mean | P25 | Median | P75 | Loss% |")
        lines.append("|---|---|---|---|---|---|---|")
        for r, m in metrics["exit_reason_breakdown"].items():
            lines.append(f"| {r} | {m['count']:,} | {m['mean_pnl_bps']:.2f} | {m['p25_bps']:.0f} | {m['median_pnl_bps']:.0f} | {m['p75_bps']:.0f} | {m['loss_rate']*100:.1f}% |")
        lines.append("")

    if metrics["side_breakdown"]:
        lines.append("## Side breakdown")
        lines.append("")
        lines.append("| Side | Count | Mean PnL | Win rate |")
        lines.append("|---|---|---|---|")
        for s, m in metrics["side_breakdown"].items():
            lines.append(f"| {s} | {m['count']:,} | {m['mean_pnl_bps']:.2f} | {m['win_rate']*100:.1f}% |")
        lines.append("")

    if metrics["strat_breakdown"]:
        lines.append("## vol_regime × session breakdown")
        lines.append("")
        lines.append("| Stratum | Count | Mean PnL | Win rate |")
        lines.append("|---|---|---|---|")
        for k, m in sorted(metrics["strat_breakdown"].items()):
            lines.append(f"| {k} | {m['count']:,} | {m['mean_pnl_bps']:.2f} | {m['win_rate']*100:.1f}% |")
        lines.append("")

    if metrics["hour_breakdown"]:
        lines.append("## Hour-of-day breakdown")
        lines.append("")
        lines.append("| UTC hour | Count | Mean PnL |")
        lines.append("|---|---|---|")
        for h in sorted(metrics["hour_breakdown"].keys()):
            m = metrics["hour_breakdown"][h]
            lines.append(f"| {h:02d} | {m['count']:,} | {m['mean_pnl_bps']:.2f} |")
        lines.append("")

    if metrics["exit_iql_survivorship"]:
        lines.append("## Exit-IQL survivorship (when does it fire?)")
        lines.append("")
        lines.append("| Bars when fired | Count | Mean PnL |")
        lines.append("|---|---|---|")
        for k in sorted(metrics["exit_iql_survivorship"].keys(), key=lambda x: int(x.split("-")[0])):
            m = metrics["exit_iql_survivorship"][k]
            lines.append(f"| {k} bars | {m['count']:,} | {m['mean_pnl_bps']:.2f} |")
        lines.append("")

    if metrics.get("conditional"):
        cond = metrics["conditional"]
        lines.append("## Conditional analysis (entry-time market state)")
        lines.append("")
        lines.append("Slicing by features captured at trade-entry moment. Helps answer: "
                     "*\"In what market conditions does this stack make money?\"*")
        lines.append("")
        sections = [
            ("by_atr_bps_ent", "ATR (bps) at entry"),
            ("by_entry_spread_bps_ent", "Spread (bps) at entry"),
            ("by_p_long_ent", "p_long (XGB) at entry"),
            ("by_tradable_prob_ent", "tradable_prob (V10) at entry"),
            ("by_bad_path_prob_ent", "bad_path_prob (V10) at entry"),
            ("by_path_quality_pred_ent", "path_quality_pred (V10) at entry"),
            ("by_mfe_first_n_pred_ent", "mfe_first_n_pred (V10) at entry"),
            ("by_v3_v8_should_exit_at_entry", "v3_v8_should_exit_prob (V3 v8) at entry"),
            ("by_session_ent", "Session"),
            ("by_vol_regime_ent", "Volatility regime"),
            ("by_trend_regime_ent", "Trend regime"),
            ("by_weekday_ent", "Day of week (0=Mon)"),
        ]
        for key, label in sections:
            data = cond.get(key, {})
            if not data:
                continue
            lines.append(f"### {label}")
            lines.append("")
            lines.append("| Bin | Count | Mean PnL | Win% | Loss% |")
            lines.append("|---|---|---|---|---|")
            sorted_keys = sorted(data.keys()) if not key.startswith("by_") or "ent" in key and isinstance(next(iter(data.keys()), ""), str) else list(data.keys())
            for k in sorted_keys:
                m = data[k]
                lines.append(f"| {k} | {m['count']:,} | {m['mean_pnl_bps']:.2f} | "
                             f"{m['win_rate']*100:.1f}% | {m['loss_rate']*100:.1f}% |")
            lines.append("")

        if cond.get("session_x_side"):
            lines.append("### Session × Side")
            lines.append("")
            lines.append("| Session\\|Side | Count | Mean PnL | Win% |")
            lines.append("|---|---|---|---|")
            for k, m in sorted(cond["session_x_side"].items()):
                lines.append(f"| {k} | {m['count']:,} | {m['mean_pnl_bps']:.2f} | "
                             f"{m['win_rate']*100:.1f}% |")
            lines.append("")

        if cond.get("regime_x_session"):
            lines.append("### Regime × Session")
            lines.append("")
            lines.append("| Regime\\|Session | Count | Mean PnL | Win% |")
            lines.append("|---|---|---|---|")
            for k, m in sorted(cond["regime_x_session"].items()):
                lines.append(f"| {k} | {m['count']:,} | {m['mean_pnl_bps']:.2f} | "
                             f"{m['win_rate']*100:.1f}% |")
            lines.append("")

    lines.append("## Q-advantage filter sweep")
    lines.append("")
    lines.append("| Threshold | n_take | PnL/taken | PnL/total | Win% | Loss% |")
    lines.append("|---|---|---|---|---|---|")
    for r in metrics["qadv_sweep"]:
        lines.append(
            f"| {r['threshold']:.1f} | {r['n_take']:,} | "
            f"{r['pnl_per_taken_bps']:.2f} | {r['pnl_per_total_bps']:.2f} | "
            f"{r['win_rate_taken']*100:.1f}% | {r['loss_rate_taken']*100:.1f}% |"
        )
    lines.append("")
    lines.append("**Best per-taken PnL**: " +
                 ", ".join(f"T={r['threshold']:.1f} → {r['pnl_per_taken_bps']:.2f}"
                           for r in sorted(metrics["qadv_sweep"], key=lambda x: -x["pnl_per_taken_bps"])[:3]))
    lines.append("")

    return "\n".join(lines)


def compute_metrics(p7: Dict[str, Any], per_bar_dir: Optional[Path] = None) -> Dict[str, Any]:
    df = p7["df"]
    # Optionally join entry-bar features for conditional analysis
    if per_bar_dir is not None:
        df_joined = join_entry_features(df, per_bar_dir)
    else:
        df_joined = df

    has_entry_features = any(c.endswith("_ent") for c in df_joined.columns)
    metrics = {
        "source_dir": str(p7["dir"]),
        "summary_keys": sorted(p7["summary"].keys()),
        "overall": compute_overall(df),
        "action_breakdown": compute_action_breakdown(df),
        "exit_reason_breakdown": compute_exit_reason_breakdown(df),
        "side_breakdown": compute_side_breakdown(df),
        "hour_breakdown": compute_hour_breakdown(df),
        "strat_breakdown": compute_strat_breakdown(df),
        "exit_iql_survivorship": compute_exit_iql_survivorship(df),
        "qadv_sweep": compute_qadv_sweep(df, QADV_THRESHOLDS),
    }
    if has_entry_features:
        metrics["conditional"] = compute_conditional_analyses(df_joined)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase7-dir", type=str, required=True,
                        help="Path to JOINT_..._LOCK directory")
    parser.add_argument("--baseline-dir", type=str, default=str(WAVE2_BASELINE_DIR),
                        help="Path to baseline Phase 7 LOCK for comparison (default: wave 2)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory (default: phase7-dir/v9_endcheck)")
    parser.add_argument("--per-bar-dir", type=str, default=None,
                        help="Forward-outcome dir override for entry-feature join "
                             "(default: CANDIDATE_FORWARD_OUTCOME_RUN/v1_full)")
    parser.add_argument("--no-conditional", action="store_true",
                        help="Skip conditional analysis (faster; no entry-feature join)")
    args = parser.parse_args()

    phase7_dir = Path(args.phase7_dir).expanduser().resolve()
    out_dir = Path(args.out) if args.out else phase7_dir / "v9_endcheck"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v9-endcheck] loading phase 7 from {phase7_dir}", flush=True)
    p7 = load_phase7(phase7_dir)

    # Resolve forward-outcome dir for conditional analysis
    fwd_dir: Optional[Path] = None
    if not args.no_conditional:
        fwd_dir = Path(args.per_bar_dir).expanduser().resolve() if args.per_bar_dir else FORWARD_OUTCOME_DIR
        if fwd_dir.exists():
            print(f"[v9-endcheck] forward-outcome dir for conditional analysis: {fwd_dir}", flush=True)
        else:
            fwd_dir = None
            print(f"[v9-endcheck] no forward-outcome dir, skipping conditional", flush=True)

    print(f"[v9-endcheck] computing metrics...", flush=True)
    import datetime
    metrics = compute_metrics(p7, per_bar_dir=fwd_dir)
    metrics["generated_at"] = datetime.datetime.utcnow().isoformat() + "Z"

    baseline_metrics = None
    baseline_dir = Path(args.baseline_dir).expanduser().resolve() if args.baseline_dir else None
    if baseline_dir and baseline_dir != phase7_dir and baseline_dir.exists():
        try:
            print(f"[v9-endcheck] loading baseline from {baseline_dir}", flush=True)
            baseline_p7 = load_phase7(baseline_dir)
            # Don't load entry features for baseline (saves time + same conclusions)
            baseline_metrics = compute_metrics(baseline_p7, per_bar_dir=None)
            baseline_metrics["generated_at"] = metrics["generated_at"]
        except Exception as e:
            print(f"[v9-endcheck] baseline load failed: {e}", flush=True)

    md = render_markdown(metrics, baseline_metrics)
    md_path = out_dir / "v9_endcheck_report.md"
    md_path.write_text(md)
    print(f"[v9-endcheck] markdown -> {md_path}", flush=True)

    json_path = out_dir / "v9_endcheck_metrics.json"
    out_metrics = {"current": metrics}
    if baseline_metrics:
        out_metrics["baseline"] = baseline_metrics
    json_path.write_text(json.dumps(out_metrics, indent=2, default=str))
    print(f"[v9-endcheck] metrics -> {json_path}", flush=True)

    o = metrics["overall"]
    print(f"\n=== Quick view ===")
    print(f"PnL/total: {o['pnl_mean_bps']:.2f}, PnL/taken: {o['pnl_per_taken_bps']:.2f}")
    print(f"Win rate: {o['win_rate_overall']*100:.1f}%, Loss rate: {o['loss_rate_overall']*100:.1f}%")
    if baseline_metrics:
        bo = baseline_metrics["overall"]
        print(f"vs baseline: PnL/total {bo['pnl_mean_bps']:.2f} → {o['pnl_mean_bps']:.2f} ({o['pnl_mean_bps']-bo['pnl_mean_bps']:+.2f})")


if __name__ == "__main__":
    main()
