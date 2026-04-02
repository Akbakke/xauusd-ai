#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report Truth Decomposition: Payoff Shapes (Delayed Edge vs Instant Fail)

DEL 4: Analyze payoff patterns per session and ATR bucket.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from gx1.execution.live_features import infer_session_tag

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def analyze_payoff_shapes(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze payoff shapes per session and ATR bucket.
    
    Returns nested dict: results[session][atr_bucket] = metrics
    """
    # Create ATR buckets
    if "atr_bps" in df.columns and df["atr_bps"].notna().sum() > 0:
        atr_quantiles = df["atr_bps"].quantile([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        df["atr_bucket"] = pd.cut(
            df["atr_bps"],
            bins=atr_quantiles.values,
            labels=["Q0-Q20", "Q20-Q40", "Q40-Q60", "Q60-Q80", "Q80-Q100"],
            include_lowest=True,
        )
    else:
        df["atr_bucket"] = "ALL"
    
    results = {}
    sessions = ["ASIA", "EU", "OVERLAP", "US"]
    
    for session in sessions:
        df_session = df[df["entry_session"] == session].copy()
        
        if len(df_session) == 0:
            results[session] = {}
            continue
        
        session_results = {}
        
        for atr_bucket in df_session["atr_bucket"].unique():
            df_bucket = df_session[df_session["atr_bucket"] == atr_bucket].copy()
            
            winners = df_bucket[df_bucket["pnl_bps"] > 0]
            losers = df_bucket[df_bucket["pnl_bps"] <= 0]
            
            if len(winners) == 0 and len(losers) == 0:
                continue
            
            # Metrics
            winner_bars_held = winners["bars_held"].values if len(winners) > 0 else np.array([])
            loser_bars_held = losers["bars_held"].values if len(losers) > 0 else np.array([])
            
            # Quick-fail rate: % losers with holding_bars <= 3
            quick_fail_rate = (loser_bars_held <= 3).mean() if len(loser_bars_held) > 0 else 0.0
            
            # Delayed-payoff rate: % winners with holding_bars >= 10
            delayed_payoff_rate = (winner_bars_held >= 10).mean() if len(winner_bars_held) > 0 else 0.0
            
            session_results[str(atr_bucket)] = {
                "n_trades": len(df_bucket),
                "n_winners": len(winners),
                "n_losers": len(losers),
                "median_winner_bars_held": float(np.median(winner_bars_held)) if len(winner_bars_held) > 0 else 0.0,
                "median_loser_bars_held": float(np.median(loser_bars_held)) if len(loser_bars_held) > 0 else 0.0,
                "quick_fail_rate": float(quick_fail_rate),
                "delayed_payoff_rate": float(delayed_payoff_rate),
            }
        
        results[session] = session_results
    
    return results


def generate_report(
    results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate DEL 4 markdown report."""
    md_path = output_dir / "TRUTH_DECOMP_PAYOFF_SHAPES.md"
    
    with open(md_path, "w") as f:
        f.write("# Truth Decomposition: Payoff Shapes (Delayed Edge vs Instant Fail)\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().isoformat()}\n\n")
        
        for session in ["ASIA", "EU", "OVERLAP", "US"]:
            session_results = results.get(session, {})
            
            if not session_results:
                continue
            
            f.write(f"## {session}\n\n")
            
            f.write("| ATR Bucket | Trades | Winners | Losers | Winner Median Bars | Loser Median Bars | Quick-Fail % | Delayed-Payoff % |\n")
            f.write("|------------|--------|---------|--------|-------------------|------------------|--------------|------------------|\n")
            
            for atr_bucket, metrics in sorted(session_results.items()):
                f.write(
                    f"| {atr_bucket} | {metrics['n_trades']:,} | {metrics['n_winners']:,} | "
                    f"{metrics['n_losers']:,} | {metrics['median_winner_bars_held']:.1f} | "
                    f"{metrics['median_loser_bars_held']:.1f} | {metrics['quick_fail_rate']:.1%} | "
                    f"{metrics['delayed_payoff_rate']:.1%} |\n"
                )
            f.write("\n")
        
        # Special focus on OVERLAP
        f.write("## OVERLAP Pattern Confirmation\n\n")
        overlap_results = results.get("OVERLAP", {})
        if overlap_results:
            f.write("**Hypothesis:** Losers die fast, winners need time\n\n")
            
            all_winners = []
            all_losers = []
            for metrics in overlap_results.values():
                if metrics["n_winners"] > 0:
                    all_winners.append(metrics["median_winner_bars_held"])
                if metrics["n_losers"] > 0:
                    all_losers.append(metrics["median_loser_bars_held"])
            
            if all_winners and all_losers:
                overall_winner_median = np.median(all_winners)
                overall_loser_median = np.median(all_losers)
                
                f.write(f"- **Overall Winner Median Bars:** {overall_winner_median:.1f}\n")
                f.write(f"- **Overall Loser Median Bars:** {overall_loser_median:.1f}\n")
                f.write(f"- **Ratio:** {overall_winner_median / overall_loser_median:.2f}x\n\n")
                
                if overall_winner_median > overall_loser_median * 1.5:
                    f.write("✅ **CONFIRMED:** Winners take significantly longer than losers\n\n")
                else:
                    f.write("⚠️ **PARTIAL:** Pattern exists but not as strong as expected\n\n")
    
    log.info(f"✅ Wrote payoff shapes report: {md_path}")


def append_h2_overlap_auc_decomposition(run_root: Path, run_id: str) -> None:
    """Append H2_OVERLAP_AUC_DECOMPOSITION section to H2_OVERLAP_SIGNAL_REPORT.md (postrun only)."""
    pred_path = run_root / f"xgb_multi_horizon_predictions_{run_id}.parquet"
    report_path = run_root / "H2_OVERLAP_SIGNAL_REPORT.md"
    if not pred_path.exists():
        print(f"[H2_OVERLAP_AUC_DECOMPOSITION] skipped=missing_predictions path={pred_path}")
        return

    df_full = pd.read_parquet(pred_path)
    print(
        "[H2_RUN_DATA_PROOF] "
        f"run_id={run_id} "
        f"pred_path={pred_path} "
        f"n_rows_total={len(df_full)} "
        f"ts_min={df_full['ts'].min()} "
        f"ts_max={df_full['ts'].max()}",
        flush=True,
    )
    if df_full.empty:
        print("[H2_OVERLAP_AUC_DECOMPOSITION] skipped=empty_predictions")
        return

    df_full = df_full[(df_full["head"] == "OVERLAP") & (df_full["horizon_bars"] == 24)].copy()
    if df_full.empty:
        print("[H2_OVERLAP_AUC_DECOMPOSITION] skipped=no_overlap_h2")
        return

    def _atr_bucket_name(val: Any) -> str:
        if pd.isna(val):
            return "N/A"
        sval = str(val).upper()
        if sval in ("LOW", "MID", "HIGH"):
            return sval
        try:
            iv = int(val)
        except Exception:
            return "N/A"
        if iv <= 0:
            return "LOW"
        if iv == 1:
            return "MID"
        if iv >= 2:
            return "HIGH"
        return "N/A"

    def _trend_regime_name(val: Any) -> str:
        if pd.isna(val):
            return "N/A"
        try:
            return f"R{int(val)}"
        except Exception:
            return "N/A"

    try:
        from gx1.execution.live_features import infer_session_tag  # type: ignore
    except Exception:
        infer_session_tag = None

    if "atr_bucket" in df_full.columns:
        df_full["atr_bucket_name"] = df_full["atr_bucket"].map(_atr_bucket_name)
    else:
        df_full["atr_bucket_name"] = "N/A"

    if "trend_regime" in df_full.columns:
        df_full["trend_regime_name"] = df_full["trend_regime"].map(_trend_regime_name)
    elif "trend_regime_id" in df_full.columns:
        df_full["trend_regime_name"] = df_full["trend_regime_id"].map(_trend_regime_name)
    else:
        df_full["trend_regime_name"] = "N/A"

    if "session_tag" not in df_full.columns:
        if infer_session_tag is not None:
            df_full["session_tag"] = df_full["ts"].apply(lambda ts: str(infer_session_tag(ts)).upper())
        else:
            df_full["session_tag"] = "N/A"

    full_session_counts = {str(k): int(v) for k, v in df_full["session_tag"].value_counts(dropna=False).items()}

    df = df_full.copy()
    n_pred_total = len(df)
    n_ts_missing = 0  # not tracked here (tape join already enforced upstream)
    n_horizon_oob = 0  # not tracked here (label writer enforces bounds)

    # For OVERLAP we use binary LONG label (y_true == 1) and score = p_long
    df["y_true_num"] = pd.to_numeric(df["y_true"], errors="coerce")
    df = df.dropna(subset=["y_true_num", "p_long"]).copy()
    df["y_true_num"] = df["y_true_num"].astype(int)
    if not set(df["y_true_num"].unique()).issubset({0, 1}):
        raise RuntimeError("[H2_OVERLAP_AUC_DECOMPOSITION] y_true_num contains values outside {0,1}")

    n_ytrue_ok = int(df["y_true_num"].notna().sum())
    if df.empty:
        print(
            "[PRED_LABEL_COVERAGE_PROOF] total=%d ts_missing=%d horizon_oob=%d ytrue_ok=%d after_filters=0 bad_ts=%s"
            % (n_pred_total, n_ts_missing, n_horizon_oob, n_ytrue_ok, [])
        )
        print("[H2_OVERLAP_AUC_DECOMPOSITION] skipped=empty_after_ytrue_map")
        return
    n_after_filters = len(df)
    print(
        "[PRED_LABEL_COVERAGE_PROOF] total=%d ts_missing=%d horizon_oob=%d ytrue_ok=%d after_filters=%d"
        % (n_pred_total, n_ts_missing, n_horizon_oob, n_ytrue_ok, n_after_filters)
    )

    if "p_short" not in df.columns:
        df["p_short"] = 1.0 - df["p_long"]
    df["hit"] = (df["y_true_num"] == 1).astype(int)
    y_true_num = df["y_true_num"]
    print(
        "[H2_CONDITIONAL_SCORE_PROOF] "
        f"mean_p_long_pos={df.loc[y_true_num==1, 'p_long'].mean():.6f} "
        f"mean_p_long_neg={df.loc[y_true_num==0, 'p_long'].mean():.6f}"
    )
    print(
        "[H2_CONDITIONAL_SCORE_SHORT_PROOF] "
        f"mean_p_short_pos={df.loc[y_true_num==1,'p_short'].mean():.6f} "
        f"mean_p_short_neg={df.loc[y_true_num==0,'p_short'].mean():.6f}"
    )
    subset_session_counts = {str(k): int(v) for k, v in df["session_tag"].value_counts(dropna=False).items()}
    print("[H2_OVERLAP_SESSION_COVERAGE_PROOF] subset=%s" % subset_session_counts)
    print("[H2_OVERLAP_SESSION_COVERAGE_PROOF] full=%s" % full_session_counts)

    score_cols = ["p_long"]

    def _group_metrics(group_name: str, frame: pd.DataFrame) -> Dict[str, Any]:
        n_total = int(len(frame))
        n_label_nan = int(frame["y_true_num"].isna().sum())
        n_score_nan = int(frame[score_cols].isna().any(axis=1).sum()) if n_total > 0 else 0

        valid_mask = frame["y_true_num"].notna()
        if n_total > 0:
            valid_mask &= ~frame[score_cols].isna().any(axis=1)
        valid_frame = frame[valid_mask].copy()

        valid_labels = valid_frame["y_true_num"].astype(int) if not valid_frame.empty else pd.Series(dtype=int)
        n_pos = int((valid_labels == 1).sum())
        n_neg = int((valid_labels == 0).sum())

        true_scores = valid_frame["p_long"].to_numpy() if not valid_frame.empty else np.array([])
        score_std = float(np.nanstd(true_scores)) if len(true_scores) > 0 else float("nan")

        if n_label_nan > 0 or n_score_nan > 0:
            auc_status = "NAN_IN_INPUT"
        elif n_pos == 0:
            auc_status = "TOO_FEW_POS"
        elif n_neg == 0:
            auc_status = "TOO_FEW_NEG"
        elif not np.isfinite(score_std) or score_std == 0:
            auc_status = "CONST_SCORE"
        else:
            auc_status = "OK"

        auc_ovr = float("nan")
        if auc_status == "OK":
            try:
                # Binary AUC using p_long vs (1 - p_long)
                y = valid_labels.astype(int)
                scores = valid_frame["p_long"].astype(float)
                auc_ovr = float(roc_auc_score(y, scores))
            except Exception:
                auc_ovr = float("nan")
                auc_status = "NAN_IN_INPUT"

        base_rate = float(frame["hit"].mean()) if n_total > 0 else float("nan")

        if len(true_scores) > 0:
            hit_mask = valid_frame["hit"].to_numpy(dtype=bool)
            mean_score_pos = float(np.nanmean(true_scores[hit_mask])) if hit_mask.any() else float("nan")
            mean_score_neg = float(np.nanmean(true_scores[~hit_mask])) if (~hit_mask).any() else float("nan")
        else:
            mean_score_pos = float("nan")
            mean_score_neg = float("nan")

        delta_mean = (
            float(mean_score_pos - mean_score_neg) if np.isfinite(mean_score_pos) and np.isfinite(mean_score_neg) else float("nan")
        )

        return {
            "group": group_name,
            "n": n_total,
            "base_rate": base_rate,
            "auc_ovr": auc_ovr,
            "auc_status": auc_status,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "score_std": score_std,
            "delta_mean": delta_mean,
        }

    rows: List[Dict[str, Any]] = []
    rows.append(_group_metrics("overall", df))
    _overall_auc = rows[0].get("auc_ovr", float("nan"))
    eps = 1e-12
    try:
        overall_auc_inverted = float(roc_auc_score(df["y_true_num"].to_numpy(), 1 - df["p_long"].to_numpy()))
    except Exception:
        overall_auc_inverted = float("nan")
    print(f"[H2_INVERTED_AUC_PROOF] overall_auc_inverted={overall_auc_inverted}", flush=True)
    if np.isfinite(_overall_auc) and np.isfinite(overall_auc_inverted):
        polarity_long = "INVERT" if overall_auc_inverted > _overall_auc + eps else "NORMAL"
    else:
        polarity_long = "UNKNOWN"
    print(
        "[H2_POLARITY_PROOF] "
        f"overall_auc={_overall_auc} overall_auc_inverted={overall_auc_inverted} polarity_long={polarity_long}",
        flush=True,
    )
    short_label = 1 - df["y_true_num"].to_numpy()
    overall_auc_short = float(roc_auc_score(short_label, df["p_short"].to_numpy()))
    print(f"[H2_SHORT_AUC_PROOF] overall_auc_short={overall_auc_short:.4f}", flush=True)
    try:
        overall_auc_short_inverted = float(roc_auc_score(df["y_true_num"].to_numpy(), 1.0 - df["p_short"].to_numpy()))
    except Exception:
        overall_auc_short_inverted = float("nan")
    print(f"[H2_SHORT_INVERTED_AUC_PROOF] overall_auc_short_inverted={overall_auc_short_inverted}", flush=True)

    atr_groups = ["LOW", "MID", "HIGH"]
    for g in atr_groups:
        frame = df[df["atr_bucket_name"] == g]
        rows.append(_group_metrics(f"atr_bucket={g}", frame))

    trend_groups = sorted({str(v) for v in df["trend_regime_name"].unique()})
    if not trend_groups:
        trend_groups = ["N/A"]
    for g in trend_groups:
        frame = df[df["trend_regime_name"] == g]
        rows.append(_group_metrics(f"trend_regime={g}", frame))

    session_tags = list(dict.fromkeys(df["session_tag"].astype(str)))  # preserve order of appearance
    for g in session_tags:
        frame = df[df["session_tag"] == g]
        rows.append(_group_metrics(f"session={g}", frame))

    def _auc_delta(prefix: str) -> Tuple[float, str]:
        vals = [(r["group"], r["auc_ovr"]) for r in rows if r["group"].startswith(prefix) and r["auc_status"] == "OK"]
        auc_vals = [v for _, v in vals if np.isfinite(v)]
        if not auc_vals:
            return float("nan"), "N/A"
        max_auc = max(auc_vals)
        min_auc = min(auc_vals)
        max_group = next((g for g, v in vals if v == max_auc), "N/A")
        return float(max_auc - min_auc), max_group

    atr_delta, atr_max_group = _auc_delta("atr_bucket=")
    trend_delta, trend_max_group = _auc_delta("trend_regime=")

    eligible_sessions = {"EU", "OVERLAP", "US"}
    session_rows_ok: List[Tuple[str, float]] = []
    for r in rows:
        if r["group"].startswith("session=") and r["auc_status"] == "OK" and np.isfinite(r["auc_ovr"]):
            sess = r["group"].split("=", 1)[1]
            if sess in eligible_sessions:
                session_rows_ok.append((sess, r["auc_ovr"]))
    if session_rows_ok:
        vals = [v for _, v in session_rows_ok]
        session_delta = float(max(vals) - min(vals))
        max_group = next((g for g, v in session_rows_ok if v == max(vals)), "N/A")
        session_max_group = f"session={max_group}"
    else:
        session_delta = float("nan")
        session_max_group = "N/A"
    atr_auc_vals = [
        r["auc_ovr"]
        for r in rows
        if r["group"].startswith("atr_bucket=") and r["auc_status"] == "OK" and np.isfinite(r["auc_ovr"])
    ]
    atr_var_auc = float(np.var(atr_auc_vals)) if atr_auc_vals else float("nan")

    atr_delta_mean_candidates = [
        r["delta_mean"]
        for r in rows
        if r["group"].startswith("atr_bucket=")
        and r["n_pos"] > 0
        and r["n_neg"] > 0
        and np.isfinite(r["delta_mean"])
    ]
    atr_var_delta_mean = (
        float(max(atr_delta_mean_candidates) - min(atr_delta_mean_candidates)) if atr_delta_mean_candidates else float("nan")
    )

    verdict_label = "concentrated" if np.isfinite(atr_delta) and atr_delta >= 0.05 else "even"

    groups_total = len(rows)
    groups_ok = len([r for r in rows if r["auc_status"] == "OK"])
    groups_invalid = groups_total - groups_ok
    print(
        "[XGB_MULTI_HORIZON_AUC_STATUS_PROOF] groups_total=%d groups_ok=%d groups_invalid=%d"
        % (groups_total, groups_ok, groups_invalid)
    )

    lines = [
        "## H2_OVERLAP_AUC_DECOMPOSITION",
        "",
        f"- buckets: atr={','.join(atr_groups)} trend={','.join(trend_groups)} session=EU,US",
        "",
        "| group | n | base_rate | auc_ovr | auc_status | n_pos | n_neg | score_std | delta_mean |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        base_rate = "nan" if not np.isfinite(r["base_rate"]) else f"{r['base_rate']:.4f}"
        auc_ovr = "nan" if not np.isfinite(r["auc_ovr"]) else f"{r['auc_ovr']:.4f}"
        score_std = "nan" if not np.isfinite(r["score_std"]) else f"{r['score_std']:.6f}"
        delta_mean = "nan" if not np.isfinite(r["delta_mean"]) else f"{r['delta_mean']:.6f}"
        lines.append(
            f"| {r['group']} | {r['n']} | {base_rate} | {auc_ovr} | {r['auc_status']} | {r['n_pos']} | "
            f"{r['n_neg']} | {score_std} | {delta_mean} |"
        )

    no_valid_auc = groups_ok == 0
    session_note_parts: List[str] = []
    if "US" not in session_tags:
        session_note_parts.append("US not present")
    if "ASIA" in session_tags:
        session_note_parts.append("ASIA excluded (NO-TRADE)")
    session_note = (
        " ".join(session_note_parts) + " -> session_delta computed over eligible sessions only"
        if session_note_parts
        else ""
    )

    def _compute_hist_metric(pred_path_hist: Path) -> Tuple[Optional[float], str]:
        if not pred_path_hist.exists():
            return None, "missing_predictions"
        df_hist = pd.read_parquet(pred_path_hist)
        if df_hist.empty:
            return None, "wrong_slice"
        df_hist = df_hist[(df_hist["head"] == "OVERLAP") & (df_hist["horizon_bars"] == 24)].copy()
        if df_hist.empty:
            return None, "wrong_slice"

        required_cols = {"y_true", "p_long"}
        if not required_cols.issubset(set(df_hist.columns)):
            return None, "missing_cols"
        if "p_short" not in df_hist.columns:
            df_hist["p_short"] = 1.0 - df_hist["p_long"]
        if "atr_bucket" in df_hist.columns:
            df_hist["atr_bucket_name"] = df_hist["atr_bucket"].map(_atr_bucket_name)
        else:
            df_hist["atr_bucket_name"] = "N/A"
        if "trend_regime" in df_hist.columns:
            df_hist["trend_regime_name"] = df_hist["trend_regime"].map(_trend_regime_name)
        elif "trend_regime_id" in df_hist.columns:
            df_hist["trend_regime_name"] = df_hist["trend_regime_id"].map(_trend_regime_name)
        else:
            df_hist["trend_regime_name"] = "N/A"
        if "session_tag" not in df_hist.columns:
            if infer_session_tag is not None:
                df_hist["session_tag"] = df_hist["ts"].apply(lambda ts: str(infer_session_tag(ts)).upper())
            else:
                df_hist["session_tag"] = "N/A"
        df_hist["y_true_num"] = pd.to_numeric(df_hist["y_true"], errors="coerce")
        df_hist = df_hist.dropna(subset=["y_true_num", "p_long"]).copy()
        df_hist["y_true_num"] = df_hist["y_true_num"].astype(int)
        if not set(df_hist["y_true_num"].unique()).issubset({0, 1}):
            return None, "missing_cols"
        df_hist["hit"] = (df_hist["y_true_num"].astype(int) == 1).astype(int)

        rows_hist: List[Dict[str, Any]] = []
        rows_hist.append(_group_metrics("overall", df_hist))
        for g in ["LOW", "MID", "HIGH"]:
            frame = df_hist[df_hist["atr_bucket_name"] == g]
            rows_hist.append(_group_metrics(f"atr_bucket={g}", frame))
        for g in sorted({str(v) for v in df_hist["trend_regime_name"].unique()} or ["N/A"]):
            frame = df_hist[df_hist["trend_regime_name"] == g]
            rows_hist.append(_group_metrics(f"trend_regime={g}", frame))
        for g in dict.fromkeys(df_hist["session_tag"].astype(str)):
            frame = df_hist[df_hist["session_tag"] == g]
            rows_hist.append(_group_metrics(f"session={g}", frame))

        atr_deltas = [
            r["delta_mean"]
            for r in rows_hist
            if r["group"].startswith("atr_bucket=")
            and r["n_pos"] > 0
            and r["n_neg"] > 0
            and np.isfinite(r["delta_mean"])
        ]
        if not atr_deltas:
            return None, "no_valid"
        return float(max(atr_deltas) - min(atr_deltas)), "ok"

    runs_root = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
    if not runs_root.exists():
        raise RuntimeError(f"[H2_OVERLAP_ATR_VAR_CONTEXT] runs_root missing: {runs_root}")
    hist_vals: List[float] = []
    reject_counts = {
        "missing_predictions": 0,
        "missing_cols": 0,
        "wrong_slice": 0,
        "no_valid": 0,
        "ok": 0,
    }
    scanned = 0
    for p in runs_root.iterdir():
        if not p.is_dir():
            continue
        run_id_hist = p.name
        if p.resolve() == run_root.resolve():
            continue
        pred_hist = p / f"xgb_multi_horizon_predictions_{run_id_hist}.parquet"
        scanned += 1
        hist_val, status = _compute_hist_metric(pred_hist)
        if status in reject_counts:
            reject_counts[status] += 1
        if hist_val is not None and np.isfinite(hist_val):
            hist_vals.append(hist_val)
            reject_counts["ok"] += 0  # already counted in status

    if hist_vals and np.isfinite(atr_var_delta_mean):
        pctl_rank = float(np.mean([1.0 if v < atr_var_delta_mean else 0.0 for v in hist_vals]))
    else:
        pctl_rank = float("nan")

    lines += [
        "",
        (
            f"Verdict: atr_delta={atr_delta:.4f} (max={atr_max_group}) trend_delta={trend_delta:.4f} (max={trend_max_group}) "
            f"session_delta={session_delta:.4f} (max={session_max_group}) "
            f"{'atr_var_auc=' + f'{atr_var_auc:.6f}' if np.isfinite(atr_var_auc) else 'atr_var_delta_mean=' + (f'{atr_var_delta_mean:.6f}' if np.isfinite(atr_var_delta_mean) else 'nan')}"
            f" -> {verdict_label}"
            f"{' no valid AUC groups' if no_valid_auc else ''}"
            f"{' ' + session_note if session_note else ''}"
        ),
        f"[H2_POLARITY] polarity_long={polarity_long} (interpretation: p_long is aligned with LONG label if NORMAL; if INVERT then use (1-p_long) as LONG-score in analysis only)",
        "",
        f"- Session coverage (full head=OVERLAP,h2=24): {full_session_counts}",
        f"- Session coverage (subset used in AUC): {subset_session_counts}",
        f"[H2_OVERLAP_ATR_VAR_CONTEXT] current={atr_var_delta_mean if np.isfinite(atr_var_delta_mean) else float('nan'):.6f} pctl_rank={pctl_rank if np.isfinite(pctl_rank) else float('nan'):.2f} n_runs={len(hist_vals)}",
        f"[H2_OVERLAP_ATR_VAR_CONTEXT_PROOF] scanned={scanned} ok={reject_counts.get('ok', 0)} missing_predictions={reject_counts.get('missing_predictions', 0)} missing_cols={reject_counts.get('missing_cols', 0)} wrong_slice={reject_counts.get('wrong_slice', 0)} no_valid={reject_counts.get('no_valid', 0)}",
        "",
    ]

    if report_path.exists():
        existing = report_path.read_text(encoding="utf-8")
        if "## H2_OVERLAP_AUC_DECOMPOSITION" in existing:
            existing = existing.split("## H2_OVERLAP_AUC_DECOMPOSITION")[0].rstrip()
        report_path.write_text(existing.rstrip() + "\n\n" + "\n".join(lines), encoding="utf-8")
    else:
        report_path.write_text("\n".join(lines), encoding="utf-8")

    print("[H2_OVERLAP_AUC_DECOMPOSITION_PROOF] run_id=%s path=%s n=%d" % (run_id, report_path, len(df)))


def _cvar(series: pd.Series, alpha: float) -> float:
    s = series.dropna()
    if len(s) == 0:
        return float("nan")
    q = np.quantile(s, alpha)
    tail = s[s <= q]
    return float(tail.mean()) if len(tail) else float("nan")


def _tail_loss_rates(pnl: pd.Series, thresholds: List[float]) -> Dict[str, float]:
    out = {}
    for t in thresholds:
        out[f"tail_rate_{int(abs(t))}"] = float((pnl <= t).mean()) if len(pnl) else float("nan")
    return out


def _score_margin(df: pd.DataFrame) -> pd.Series:
    probs = df[["p_long", "p_short", "p_flat"]].values
    if probs.size == 0:
        return pd.Series([], dtype=float)
    # margin = max - second max
    sorted_probs = np.sort(probs, axis=1)
    return pd.Series(sorted_probs[:, -1] - sorted_probs[:, -2], index=df.index)


def _build_session_deciles(
    trades: pd.DataFrame,
    score_col: str,
    score_name: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    rows = []
    for session in ["EU", "OVERLAP", "US"]:
        sdf = trades[trades["session"] == session].copy()
        if sdf.empty or score_col not in sdf.columns:
            continue
        scores = sdf[score_col].dropna()
        if scores.empty:
            continue
        try:
            deciles = pd.qcut(scores, q=n_bins, labels=False, duplicates="drop")
        except Exception:
            continue
        sdf = sdf.loc[scores.index].copy()
        sdf["decile"] = deciles
        grp = sdf.groupby("decile", dropna=True)
        for decile, g in grp:
            pnl = g["pnl_bps"]
            mfe_med = g["mfe_bps"].median()
            tail_rates = _tail_loss_rates(pnl, [-50, -100, -200])
            rows.append(
                {
                    "session": session,
                    "score": score_name,
                    "decile": int(decile),
                    "trade_count": int(len(g)),
                    "pnl_sum_bps": float(pnl.sum()),
                    "pnl_mean_bps": float(pnl.mean()),
                    "pnl_median_bps": float(pnl.median()),
                    "win_rate": float((pnl > 0).mean()),
                    "median_MFE_bps": float(mfe_med),
                    "median_MAE_bps": float(g["mae_bps"].median()),
                    "EdgeCaptureRatio": float(pnl.median() / mfe_med) if mfe_med != 0 else float("nan"),
                    "max_loss": float(pnl.min()),
                    "CVaR95": float(_cvar(pnl, 0.05)),
                    "CVaR99": float(_cvar(pnl, 0.01)),
                    "score_min": float(g[score_col].min()),
                    "score_max": float(g[score_col].max()),
                    **tail_rates,
                }
            )
    return pd.DataFrame(rows)


def _build_tail_risk_attribution(trades: pd.DataFrame) -> pd.DataFrame:
    df = trades.copy()
    df["exit_session"] = df["close_ts_utc"].apply(lambda ts: infer_session_tag(ts).upper() if pd.notna(ts) else "UNKNOWN")
    df["session_transition"] = df["exit_session"] != df["session"]
    df["distance_from_peak_mfe_bps"] = df["mfe_bps"] - df["pnl_bps"]
    df["giveback_ratio"] = np.where(df["mfe_bps"] > 0, df["distance_from_peak_mfe_bps"] / df["mfe_bps"], np.nan)
    df["positive_mfe"] = df["mfe_bps"] > 0

    worst_n = max(20, int(len(df) * 0.01))
    worst = df.nsmallest(worst_n, "pnl_bps").copy()

    rows = []
    for session in ["EU", "OVERLAP", "US"]:
        s = worst[worst["session"] == session]
        if s.empty:
            continue
        rows.append(
            {
                "entry_session": session,
                "worst_count": int(len(s)),
                "median_bars_held": float(s["bars_in_trade"].median()),
                "median_distance_from_peak_mfe_bps": float(s["distance_from_peak_mfe_bps"].median()),
                "median_giveback_ratio": float(np.nanmedian(s["giveback_ratio"])),
                "pct_positive_mfe": float(s["positive_mfe"].mean()),
                "pct_session_transition": float(s["session_transition"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _load_run_artifacts(run_root: Path, run_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    run_dir = run_root / run_id
    trade_path = run_dir / f"trade_journal_{run_id}_MERGED.parquet"
    pred_path = run_dir / f"xgb_multi_horizon_predictions_{run_id}.parquet"
    if not trade_path.exists():
        raise RuntimeError(f"[RUN_ARTIFACTS] missing trade_journal: {trade_path}")
    if not pred_path.exists():
        raise RuntimeError(f"[RUN_ARTIFACTS] missing xgb predictions: {pred_path}")
    trades = pd.read_parquet(trade_path)
    preds = pd.read_parquet(pred_path)
    return trades, preds


def generate_xgb_session_decile_report(
    run_root: Path,
    run_ids: List[str],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    trade_frames = []
    pred_frames = []
    for run_id in run_ids:
        trades, preds = _load_run_artifacts(run_root, run_id)
        trades = trades.copy()
        trades["run_id"] = run_id
        preds = preds.copy()
        preds["run_id"] = run_id
        trade_frames.append(trades)
        pred_frames.append(preds)

    trades = pd.concat(trade_frames, ignore_index=True)
    preds = pd.concat(pred_frames, ignore_index=True)

    # Filter predictions to horizon 24 and heads
    preds = preds[(preds["horizon_bars"] == 24) & (preds["head"].isin(["EU", "OVERLAP", "US"]))].copy()
    preds["score_margin"] = _score_margin(preds)
    preds["score_p_hat"] = preds["p_hat"]

    # Join predictions to trades by entry timestamp and session-head
    trades["entry_ts"] = pd.to_datetime(trades["open_ts_utc"], utc=True)
    preds["ts"] = pd.to_datetime(preds["ts"], utc=True)
    preds["session_key"] = preds["session_tag"] if "session_tag" in preds.columns else preds["head"]

    merged = trades.merge(
        preds,
        left_on=["entry_ts", "session"],
        right_on=["ts", "session_key"],
        how="left",
        suffixes=("", "_pred"),
    )

    match_rate = float(merged["score_margin"].notna().mean())
    head_mismatch_rate = float((merged["head"].notna() & (merged["head"] != merged["session"])).mean())
    deciles = _build_session_deciles(merged, "score_margin", "margin_top1_top2", n_bins=10)
    tail_attr = _build_tail_risk_attribution(trades)

    # Write artifacts
    deciles_csv = output_dir / "ENTRY_XGB_SESSION_DECILES.csv"
    tail_csv = output_dir / "ENTRY_TAIL_RISK_ATTRIBUTION.csv"
    deciles.to_csv(deciles_csv, index=False)
    tail_attr.to_csv(tail_csv, index=False)

    md_path = output_dir / "ENTRY_XGB_SESSION_DECILES_REPORT.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# ENTRY/XGB Session-Head Decile Audit\n\n")
        f.write(f"Generated: {pd.Timestamp.now().isoformat()}\n\n")
        f.write(f"Runs: {', '.join(run_ids)}\n\n")
        f.write(f"Predictions horizon=24, head in EU/OVERLAP/US\n")
        f.write(f"Trade join match_rate={match_rate:.3f}\n")
        f.write(f"Head mismatch rate (head != entry session)={head_mismatch_rate:.3f}\n\n")
        if not deciles.empty:
            f.write("## Deciles by Session (score=margin_top1_top2)\n\n")
            f.write(deciles.to_string(index=False))
            f.write("\n\n")
        else:
            f.write("## Deciles by Session\n\nNo decile output (insufficient matches)\n\n")

        f.write("## Tail-Risk Attribution (worst 1% by pnl)\n\n")
        f.write(tail_attr.to_string(index=False))
        f.write("\n\n")
        f.write("Notes:\n")
        f.write("- time_since_mfe_bars not available in artifacts\n")
        f.write("- giveback_ratio computed as (mfe_bps - pnl_bps) / mfe_bps when mfe_bps > 0\n")
        f.write("- exit_session inferred from close_ts_utc via infer_session_tag\n")

    log.info(f"✅ Wrote ENTRY/XGB decile report: {md_path}")
    return md_path


def main():
    parser = argparse.ArgumentParser(description="Report Truth Decomposition: Payoff Shapes")
    parser.add_argument(
        "--trade-table",
        type=Path,
        help="Path to canonical trade table parquet file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for reports",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Path to append to JSON (optional)",
    )
    parser.add_argument(
        "--xgb-session-deciles",
        action="store_true",
        help="Generate ENTRY/XGB session-head decile audit using replay artifacts",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        help="Run root for artifacts (e.g., /home/andre2/GX1_DATA/reports/truth_e2e_sanity)",
    )
    parser.add_argument(
        "--run-ids",
        nargs="+",
        help="Run IDs to include in decile audit",
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    workspace_root = Path(__file__).parent.parent.parent
    if args.trade_table and not args.trade_table.is_absolute():
        args.trade_table = workspace_root / args.trade_table
    if not args.output_dir.is_absolute():
        args.output_dir = workspace_root / args.output_dir
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.xgb_session_deciles:
        if not args.run_root or not args.run_ids:
            raise RuntimeError("--xgb-session-deciles requires --run-root and --run-ids")
        run_root = args.run_root if args.run_root.is_absolute() else workspace_root / args.run_root
        generate_xgb_session_decile_report(run_root, args.run_ids, args.output_dir)
        return 0

    if not args.trade_table:
        raise RuntimeError("--trade-table is required unless --xgb-session-deciles is set")

    log.info("=" * 60)
    log.info("PAYOFF SHAPES (DELAYED EDGE vs INSTANT FAIL)")
    log.info("=" * 60)
    log.info(f"Trade table: {args.trade_table}")
    log.info(f"Output dir: {args.output_dir}")
    log.info("")

    # Load trade table
    log.info("Loading trade table...")
    df = pd.read_parquet(args.trade_table)
    log.info(f"Loaded {len(df):,} trades")

    # Analyze payoff shapes
    log.info("Analyzing payoff shapes...")
    results = analyze_payoff_shapes(df)

    # Generate report
    generate_report(results, args.output_dir)

    # Append to JSON if requested
    if args.json_output:
        json_path = workspace_root / args.json_output if not args.json_output.is_absolute() else args.json_output
        json_path.parent.mkdir(parents=True, exist_ok=True)

        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
        else:
            data = {}

        data["payoff_shapes"] = results

        with open(json_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        log.info(f"✅ Appended to JSON: {json_path}")

    log.info("✅ Payoff shapes analysis complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
